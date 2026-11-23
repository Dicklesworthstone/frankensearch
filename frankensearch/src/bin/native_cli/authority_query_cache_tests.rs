use super::*;

const CACHED_QUERY: &str = "retry cached document edit";

fn cache_config(entries: &str) -> query_cache::Config {
    query_cache::Config::parse(entries).unwrap()
}

#[test]
fn authority_cache_reuses_bound_queries_in_each_primary_tier_and_evicts_at_capacity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for tier in [EmbedderTierTag::Fast, EmbedderTierTag::Quality] {
            for enabled in [false, true] {
                let parent = private_parent("query-cache-primary");
                let model = provider();
                let selected = publish(&cx, &parent, None, &model, tier, "first", false).await;
                let mut session = open_session(&cx, selected, &model, false)
                    .with_query_cache(&cx, cache_config(if enabled { "1" } else { "0" }))
                    .unwrap();
                let before = model.calls.load(Ordering::Relaxed);
                let output = exchange(&cx, &mut session, &[
                    serde_json::json!({"query":CACHED_QUERY, "limit":1}),
                    serde_json::json!({"query":CACHED_QUERY, "limit":2, "stream":true, "filter":{"id_prefix":"retry-"}}),
                ]).await;
                assert_eq!(output[0]["query_cache_entries"], usize::from(enabled));
                assert!(output[1..].iter().all(|frame| frame["event"] == "results"));
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-first");
                assert_eq!(output[2]["results"].as_array().unwrap().len(), 1);
                assert_eq!(
                    model.calls.load(Ordering::Relaxed) - before,
                    if enabled { 1 } else { 2 }
                );
                let churn = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"query":"flowers", "limit":1}),
                        serde_json::json!({"query":CACHED_QUERY, "limit":1}),
                        serde_json::json!({"query":"flowers", "limit":1}),
                    ],
                )
                .await;
                assert!(churn[1..].iter().all(|frame| frame["event"] == "results"));
                assert_eq!(churn[1]["results"][0]["doc_id"], "garden-first");
                assert_eq!(churn[3]["results"][0]["doc_id"], "garden-first");
                let inference_calls = model.calls.load(Ordering::Relaxed) - before;
                if enabled {
                    // One entry cannot retain both distinct queries. Which key
                    // survives is the existing cache's admission policy; this
                    // serving test only pins reuse and the configured capacity.
                    assert!((3..=4).contains(&inference_calls));
                } else {
                    assert_eq!(inference_calls, 5);
                }
            }
        }
    });
}

#[test]
fn same_producer_data_activation_keeps_query_cache_but_retrieves_the_new_snapshot() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("query-cache-data-activation");
        let model = provider();
        let first = publish(
            &cx,
            &parent,
            None,
            &model,
            EmbedderTierTag::Fast,
            "first",
            false,
        )
        .await;
        let mut session = open_session(&cx, first.clone(), &model, true)
            .with_query_cache(&cx, cache_config("4"))
            .unwrap();
        let before = exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(before[1]["results"][0]["doc_id"], "retry-first");
        let cached = Arc::clone(&session.query_models().models().unwrap().fast);
        let next = publish(
            &cx,
            &parent,
            Some(&session.snapshot),
            &model,
            EmbedderTierTag::Fast,
            "next",
            true,
        )
        .await;
        let calls = model.calls.load(Ordering::Relaxed);
        session
            .activate(
                &cx,
                &AuthorityActivateRequest {
                    op: "activate".to_owned(),
                    id: None,
                    receipt: parent.join("next.json"),
                    expected_authority: first.authority,
                },
            )
            .unwrap();
        assert!(Arc::ptr_eq(
            &cached,
            &session.query_models().models().unwrap().fast
        ));
        let after = exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(after[1]["results"][0]["doc_id"], "retry-next");
        assert_eq!(after[1]["authority"], serde_json::json!(next.authority));
        assert_eq!(model.calls.load(Ordering::Relaxed), calls);
    });
}

#[test]
fn quality_completion_preserves_fast_cache_and_updates_bypass_both_query_caches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut fixture =
            QualityActivationFixture::new(&cx, "query-cache-quality-completion", false, true).await;
        fixture.session = fixture
            .session
            .with_query_cache(&cx, cache_config("4"))
            .unwrap();
        let before = exchange(
            &cx,
            &mut fixture.session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(before[1]["results"][0]["doc_id"], "retry-first");
        let fast_cache = Arc::clone(&fixture.session.query_models().models().unwrap().fast);
        let request = activation_request(&fixture.message());
        let fast_calls = fixture.fast.calls.load(Ordering::Relaxed);
        let quality_calls = fixture.quality.calls.load(Ordering::Relaxed);
        assert!(!fixture.fast.bound_batch_is_native());
        assert!(!fixture.quality.bound_batch_is_native());
        let wrong: Arc<dyn Embedder> = fixture.fast.clone();
        assert!(
            fixture
                .session
                .activate_model_with_loader(&cx, &request, |_, _, _, _| Ok(Arc::clone(&wrong)))
                .is_err()
        );
        assert!(Arc::ptr_eq(
            &fast_cache,
            &fixture.session.query_models().models().unwrap().fast
        ));
        let preserved = exchange(
            &cx,
            &mut fixture.session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(preserved[1]["phase"], "initial");
        assert_eq!(fixture.fast.calls.load(Ordering::Relaxed), fast_calls);
        let quality: Arc<dyn Embedder> = fixture.quality.clone();
        fixture
            .session
            .activate_model_with_loader(&cx, &request, |_, _, tier, _| {
                assert_eq!(tier, EmbedderTierTag::Quality);
                Ok(Arc::clone(&quality))
            })
            .unwrap();
        assert!(Arc::ptr_eq(
            &fast_cache,
            &fixture.session.query_models().models().unwrap().fast
        ));
        let quality_cache = Arc::clone(
            fixture
                .session
                .query_models()
                .models()
                .unwrap()
                .quality
                .as_ref()
                .unwrap(),
        );
        // This full-mode stream emits two phases. The primary-tier exchange
        // helper requires one response per request, so inspect the real wire
        // frames here without discarding either progressive phase.
        let mut input = Vec::new();
        for request in [
            serde_json::json!({"query":CACHED_QUERY, "stream":true}),
            serde_json::json!({"query":CACHED_QUERY}),
            serde_json::json!({"query":CACHED_QUERY, "mode":"quality"}),
            serde_json::json!({"query":CACHED_QUERY, "mode":"lexical"}),
        ] {
            serde_json::to_writer(&mut input, &request).unwrap();
            input.push(b'\n');
        }
        let mut bytes = Vec::new();
        serve_session(
            &cx,
            &mut fixture.session,
            &mut Cursor::new(input),
            &mut bytes,
        )
        .await
        .unwrap();
        let output = frames(&bytes);
        assert_eq!(output.len(), 6);
        assert_eq!(output[0]["event"], "ready");
        assert_eq!(output[1]["phase"], "initial");
        assert_eq!(output[2]["phase"], "refined");
        assert_eq!(output[3]["phase"], "refined");
        assert_eq!(output[4]["phase"], "quality");
        assert_eq!(output[5]["phase"], "lexical");
        assert_eq!(fixture.fast.calls.load(Ordering::Relaxed), fast_calls);
        assert_eq!(
            fixture.quality.calls.load(Ordering::Relaxed),
            quality_calls + 1
        );

        // These providers have no native batch path: document building calls
        // embed_bound once per changed document. The exact cached query text
        // must still be embedded for the update, without reading/filling caches.
        let update = update_message(
            &fixture.parent,
            "cached-document-update",
            &fixture.completed,
            8,
            &serde_json::json!([
                {"op":"upsert", "id":"retry-first", "content":CACHED_QUERY, "metadata":{"revision":"updated"}}
            ]),
        );
        let updated = fixture
            .session
            .update(&cx, serde_json::from_value(update).unwrap())
            .await
            .unwrap();
        assert_eq!(updated["publication"], "committed");
        assert_eq!(fixture.fast.calls.load(Ordering::Relaxed), fast_calls + 1);
        assert_eq!(
            fixture.quality.calls.load(Ordering::Relaxed),
            quality_calls + 2
        );
        assert!(Arc::ptr_eq(
            &fast_cache,
            &fixture.session.query_models().models().unwrap().fast
        ));
        assert!(Arc::ptr_eq(
            &quality_cache,
            fixture
                .session
                .query_models()
                .models()
                .unwrap()
                .quality
                .as_ref()
                .unwrap()
        ));
        let after = exchange(
            &cx,
            &mut fixture.session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(after[1]["authority"], updated["authority"]);
        assert_eq!(after[1]["results"][0]["doc_id"], "retry-first");
        assert_eq!(fixture.fast.calls.load(Ordering::Relaxed), fast_calls + 1);
        assert_eq!(
            fixture.quality.calls.load(Ordering::Relaxed),
            quality_calls + 2
        );
    });
}

#[test]
fn model_activation_replaces_query_caches_only_after_successful_complete_admission() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("query-cache-model-activation");
        let old = provider();
        let next_model = replacement_provider(false);
        assert_eq!(
            old.id(),
            next_model.id(),
            "display names deliberately match"
        );
        let first = publish(
            &cx,
            &parent,
            None,
            &old,
            EmbedderTierTag::Fast,
            "first",
            false,
        )
        .await;
        let mut session = open_session(&cx, first.clone(), &old, false)
            .with_query_cache(&cx, cache_config("4"))
            .unwrap();
        session.allow_model_activation = true;
        exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        let old_cache = Arc::clone(&session.query_models().models().unwrap().fast);
        let next = publish(
            &cx,
            &parent,
            Some(&session.snapshot),
            &next_model,
            EmbedderTierTag::Fast,
            "next",
            false,
        )
        .await;
        let old_calls = old.calls.load(Ordering::Relaxed);
        let next_calls = next_model.calls.load(Ordering::Relaxed);
        let request = activation_request(&activation_message(&parent, &first, "next"));
        for cancel in [false, true] {
            assert!(
                session
                    .activate_model_with_loader(&cx, &request, |cx, _, _, _| {
                        if cancel {
                            cx.set_cancel_requested(true);
                        }
                        Err(bad("private failed activation"))
                    })
                    .is_err()
            );
            cx.set_cancel_requested(false);
            assert!(Arc::ptr_eq(
                &old_cache,
                &session.query_models().models().unwrap().fast
            ));
            let after = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({"query":CACHED_QUERY})],
            )
            .await;
            assert_eq!(after[1]["results"][0]["doc_id"], "retry-first");
            assert_eq!(old.calls.load(Ordering::Relaxed), old_calls);
        }
        session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| Ok(next_model.clone()))
            .unwrap();
        assert!(!Arc::ptr_eq(
            &old_cache,
            &session.query_models().models().unwrap().fast
        ));
        let after = exchange(
            &cx,
            &mut session,
            &[
                serde_json::json!({"query":CACHED_QUERY}),
                serde_json::json!({"query":CACHED_QUERY}),
            ],
        )
        .await;
        assert_eq!(after[1]["authority"], serde_json::json!(next.authority));
        assert_eq!(after[1]["results"][0]["doc_id"], "retry-next");
        assert_eq!(next_model.calls.load(Ordering::Relaxed), next_calls + 1);
        assert_eq!(old.calls.load(Ordering::Relaxed), old_calls);
    });
}

struct DriftingCachedQueryProvider {
    inner: Arc<Provider>,
    changed: EmbeddingIdentityBundleV1,
    drift: std::sync::atomic::AtomicBool,
    unavailable: std::sync::atomic::AtomicBool,
    identity_calls: AtomicUsize,
}

impl DriftingCachedQueryProvider {
    fn new(inner: Arc<Provider>) -> Arc<Self> {
        let mut changed = inner.identity.clone();
        changed
            .producer
            .implementation_revision
            .push_str("-foreign-producer");
        changed.validate().unwrap();
        Arc::new(Self {
            inner,
            changed,
            drift: std::sync::atomic::AtomicBool::new(false),
            unavailable: std::sync::atomic::AtomicBool::new(false),
            identity_calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> (usize, usize) {
        (
            self.identity_calls.load(Ordering::Relaxed),
            self.inner.calls.load(Ordering::Relaxed),
        )
    }
}

impl Embedder for DriftingCachedQueryProvider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.identity_calls.fetch_add(1, Ordering::Relaxed);
        if self.unavailable.load(Ordering::Relaxed) {
            return Err(SearchError::EmbedderUnavailable {
                model: self.id().to_owned(),
                reason: "query identity is unavailable".to_owned(),
            });
        }
        Ok(if self.drift.load(Ordering::Relaxed) {
            &self.changed
        } else {
            &self.inner.identity
        })
    }
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
    fn is_semantic(&self) -> bool {
        true
    }
}

fn full_drifting_session(
    cx: &Cx,
    fixture: &QualityActivationFixture,
    cached: bool,
) -> (
    AuthoritySession,
    Arc<DriftingCachedQueryProvider>,
    Arc<DriftingCachedQueryProvider>,
) {
    let fast = DriftingCachedQueryProvider::new(Arc::clone(&fixture.fast));
    let quality = DriftingCachedQueryProvider::new(Arc::clone(&fixture.quality));
    let receipt = fixture.completed.clone();
    let root = admit_root(&receipt.root).unwrap();
    let snapshot = receipt
        .open_on_root(cx, &root, DEFAULT_IMAGE_BYTES)
        .unwrap();
    let session = AuthoritySession::new(
        cx,
        root,
        receipt,
        snapshot,
        Models {
            fast: fast.clone(),
            quality: Some(quality.clone()),
        },
        (1, DEFAULT_IMAGE_BYTES),
        false,
    )
    .unwrap()
    .with_query_cache(cx, cache_config(if cached { "4" } else { "0" }))
    .unwrap();
    (session, fast, quality)
}

#[test]
fn authority_query_mode_isolates_single_tier_producers_with_and_without_cache() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = QualityActivationFixture::new(&cx, "query-mode-isolation", false, true).await;
        for cached in [false, true] {
            let (session, fast, quality) = full_drifting_session(&cx, &fixture, cached);
            for (mode, phase, selected, unused) in [
                ("fast", "initial", &fast, &quality),
                ("quality", "quality", &quality, &fast),
            ] {
                let request: AuthoritySearchRequest = serde_json::from_value(
                    serde_json::json!({"query":CACHED_QUERY, "mode":mode, "stream":true}),
                )
                .unwrap();
                let mut baseline = Vec::new();
                session
                    .query(&cx, &request, 1, &mut baseline)
                    .await
                    .unwrap();
                let baseline = frames(&baseline);
                assert_eq!(baseline.len(), 1);
                assert_eq!(baseline[0]["results"][0]["doc_id"], "retry-first");
                for (drift, unavailable) in [(true, false), (false, true)] {
                    unused.drift.store(drift, Ordering::Relaxed);
                    unused.unavailable.store(unavailable, Ordering::Relaxed);
                    let before_selected = selected.calls();
                    let before_unused = unused.calls();
                    let mut output = Vec::new();
                    session.query(&cx, &request, 2, &mut output).await.unwrap();
                    let output = frames(&output);
                    assert_eq!(output.len(), 1);
                    assert_eq!(output[0]["results"], baseline[0]["results"]);
                    assert_eq!(output[0]["authority"], baseline[0]["authority"]);
                    assert_eq!(output[0]["mode"], mode);
                    assert_eq!(output[0]["phase"], phase);
                    assert!(selected.calls().0 > before_selected.0);
                    assert_eq!(selected.calls().1 - before_selected.1, usize::from(!cached));
                    assert_eq!(unused.calls(), before_unused);
                    unused.drift.store(false, Ordering::Relaxed);
                    unused.unavailable.store(false, Ordering::Relaxed);
                }
            }
        }
    });
}

#[test]
fn authority_query_mode_admits_requested_producers_before_cache_hits_and_empty_scopes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = QualityActivationFixture::new(&cx, "query-mode-admission", true, false).await;
        for cached in [false, true] {
            let (session, fast, quality) = full_drifting_session(&cx, &fixture, cached);
            for (mode, selected, unused) in
                [("fast", &fast, &quality), ("quality", &quality, &fast)]
            {
                let request: AuthoritySearchRequest =
                    serde_json::from_value(serde_json::json!({"query":CACHED_QUERY, "mode":mode}))
                        .unwrap();
                session
                    .query(&cx, &request, 1, &mut Vec::new())
                    .await
                    .unwrap();
                for empty_scope in [false, true] {
                    let mut request: AuthoritySearchRequest = serde_json::from_value(
                        serde_json::json!({"query":CACHED_QUERY, "mode":mode}),
                    )
                    .unwrap();
                    if empty_scope {
                        request.filter =
                            Some(Filter::parse(r#"{"id_prefix":"missing-"}"#).unwrap());
                    }
                    for (drift, unavailable, code) in [
                        (true, false, "producer_identity_mismatch"),
                        (false, true, "producer_identity_unavailable"),
                    ] {
                        selected.drift.store(drift, Ordering::Relaxed);
                        selected.unavailable.store(unavailable, Ordering::Relaxed);
                        let before_selected = selected.calls();
                        let before_unused = unused.calls();
                        let mut output = Vec::new();
                        let error = session
                            .query(&cx, &request, 2, &mut output)
                            .await
                            .unwrap_err();
                        assert!(error.to_string().contains(code), "{error}");
                        assert!(output.is_empty());
                        assert!(selected.calls().0 > before_selected.0);
                        assert_eq!(selected.calls().1, before_selected.1);
                        assert_eq!(unused.calls(), before_unused);
                        selected.drift.store(false, Ordering::Relaxed);
                        selected.unavailable.store(false, Ordering::Relaxed);
                    }
                    let before_selected = selected.calls();
                    let before_unused = unused.calls();
                    let mut recovered = Vec::new();
                    session
                        .query(&cx, &request, 3, &mut recovered)
                        .await
                        .unwrap();
                    let recovered = frames(&recovered);
                    assert_eq!(recovered.len(), 1);
                    assert_eq!(
                        recovered[0]["authority"],
                        serde_json::json!(fixture.completed.authority)
                    );
                    if empty_scope {
                        assert!(recovered[0]["results"].as_array().unwrap().is_empty());
                        assert_eq!(selected.calls().1, before_selected.1);
                    } else {
                        assert_eq!(recovered[0]["results"][0]["doc_id"], "retry-first");
                        assert_eq!(selected.calls().1 - before_selected.1, usize::from(!cached));
                    }
                    assert!(selected.calls().0 > before_selected.0);
                    assert_eq!(unused.calls(), before_unused);
                }
            }
        }
    });
}

#[test]
fn authority_query_mode_preserves_full_startup_lexical_and_protocol_boundaries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture =
            QualityActivationFixture::new(&cx, "query-mode-boundaries", false, false).await;
        for cached in [false, true] {
            let (session, fast, quality) = full_drifting_session(&cx, &fixture, cached);
            let request: AuthoritySearchRequest = serde_json::from_value(
                serde_json::json!({"query":CACHED_QUERY, "mode":"full", "stream":true}),
            )
            .unwrap();
            let mut output = Vec::new();
            session.query(&cx, &request, 1, &mut output).await.unwrap();
            assert_eq!(frames(&output).len(), 2);
            for selected in [&fast, &quality] {
                selected.drift.store(true, Ordering::Relaxed);
                let before_fast = fast.calls().1;
                let before_quality = quality.calls().1;
                let mut output = Vec::new();
                let error = session
                    .query(&cx, &request, 2, &mut output)
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("producer_identity_mismatch"));
                assert!(output.is_empty());
                let error = AuthoritySession::new(
                    &cx,
                    admit_root(&session.receipt.root).unwrap(),
                    session.receipt.clone(),
                    Arc::clone(&session.snapshot),
                    session.models.models().unwrap().clone(),
                    (1, DEFAULT_IMAGE_BYTES),
                    false,
                )
                .err()
                .expect("startup must admit both producers");
                assert!(error.to_string().contains("producer_identity_mismatch"));
                assert_eq!(fast.calls().1, before_fast);
                assert_eq!(quality.calls().1, before_quality);
                selected.drift.store(false, Ordering::Relaxed);
            }
            fast.unavailable.store(true, Ordering::Relaxed);
            quality.unavailable.store(true, Ordering::Relaxed);
            let before_fast = fast.calls();
            let before_quality = quality.calls();
            let lexical: AuthoritySearchRequest =
                serde_json::from_value(serde_json::json!({"query":CACHED_QUERY, "mode":"lexical"}))
                    .unwrap();
            let mut output = Vec::new();
            session.query(&cx, &lexical, 3, &mut output).await.unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), 1);
            assert_eq!(output[0]["phase"], "lexical");
            assert_eq!(output[0]["results"][0]["doc_id"], "retry-first");
            for (query, limit, code) in [
                (" ", 1, "invalid_query"),
                (CACHED_QUERY, 0, "invalid_limit"),
                (CACHED_QUERY, 1_001, "invalid_limit"),
            ] {
                let invalid: AuthoritySearchRequest = serde_json::from_value(
                    serde_json::json!({"query":query, "limit":limit, "mode":"fast"}),
                )
                .unwrap();
                let mut output = Vec::new();
                let error = session
                    .query(&cx, &invalid, 4, &mut output)
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains(code), "{error}");
                assert!(output.is_empty());
            }
            assert_eq!(fast.calls(), before_fast);
            assert_eq!(quality.calls(), before_quality);
        }
    });
}

#[test]
fn cached_authority_query_still_refuses_identity_drift_and_cancellation_before_a_hit() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("query-cache-admission");
        let model = provider();
        let first = publish(
            &cx,
            &parent,
            None,
            &model,
            EmbedderTierTag::Fast,
            "first",
            false,
        )
        .await;
        let drifting = DriftingCachedQueryProvider::new(Arc::clone(&model));
        let root = admit_root(&first.root).unwrap();
        let snapshot = first.open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES).unwrap();
        let mut session = AuthoritySession::new(
            &cx,
            root,
            first,
            snapshot,
            Models {
                fast: drifting.clone(),
                quality: None,
            },
            (1, DEFAULT_IMAGE_BYTES),
            false,
        )
        .unwrap()
        .with_query_cache(&cx, cache_config("4"))
        .unwrap();
        exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        let calls = model.calls.load(Ordering::Relaxed);
        let request: AuthoritySearchRequest =
            serde_json::from_value(serde_json::json!({"query":CACHED_QUERY})).unwrap();
        drifting.drift.store(true, Ordering::Relaxed);
        let mut output = Vec::new();
        let error = session
            .query(&cx, &request, 1, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("producer_identity_mismatch"));
        assert!(output.is_empty());
        drifting.drift.store(false, Ordering::Relaxed);
        cx.set_cancel_requested(true);
        let error = session
            .query(&cx, &request, 2, &mut output)
            .await
            .unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::Cancelled { .. })
        ));
        assert!(output.is_empty());
        let recovered = exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":CACHED_QUERY})],
        )
        .await;
        assert_eq!(recovered[1]["results"][0]["doc_id"], "retry-first");
        assert_eq!(model.calls.load(Ordering::Relaxed), calls);
    });
}
