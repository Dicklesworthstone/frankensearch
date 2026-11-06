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
        let output = exchange(
            &cx,
            &mut fixture.session,
            &[
                serde_json::json!({"query":CACHED_QUERY, "stream":true}),
                serde_json::json!({"query":CACHED_QUERY}),
                serde_json::json!({"query":CACHED_QUERY, "mode":"quality"}),
                serde_json::json!({"query":CACHED_QUERY, "mode":"lexical"}),
            ],
        )
        .await;
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
}

impl Embedder for DriftingCachedQueryProvider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
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
        let mut changed = model.identity.clone();
        changed
            .producer
            .implementation_revision
            .push_str("-foreign-producer");
        changed.validate().unwrap();
        let drifting = Arc::new(DriftingCachedQueryProvider {
            inner: model.clone(),
            changed,
            drift: std::sync::atomic::AtomicBool::new(false),
        });
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
