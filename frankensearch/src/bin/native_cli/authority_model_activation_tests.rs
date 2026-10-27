use super::*;

fn replacement_provider(change_space: bool) -> Arc<Provider> {
    let mut identity = identity().identity;
    identity.producer.implementation_revision = "authority-activated-producer".to_owned();
    if change_space {
        identity.space.immutable_revision = "authority-activated-weights".to_owned();
        identity.space.artifacts[0].sha256 = "c".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
    }
    identity.validate().unwrap();
    Arc::new(Provider {
        identity,
        calls: AtomicUsize::new(0),
    })
}

fn activation_message(parent: &Path, before: &Receipt, label: &str) -> serde_json::Value {
    let mut message = serde_json::json!({
        "op": "activate_model", "id": "change-producer",
        "receipt": parent.join(format!("{label}.json")),
        "expected_authority": before.authority,
    });
    match before.tier {
        EmbedderTierTag::Fast => {
            message["model_dir"] = serde_json::json!(parent.join("explicit-model-root"));
        }
        EmbedderTierTag::Quality => {
            message["quality_backend"] = serde_json::json!("native-int8");
            message["quality_model_dir"] = serde_json::json!(parent.join("explicit-native-model"));
        }
    }
    message
}

fn activation_request(message: &serde_json::Value) -> model_activation::Request {
    serde_json::from_value(message.clone()).unwrap()
}

fn assert_selection_unchanged(
    session: &AuthoritySession,
    selected: &Receipt,
    snapshot: &Arc<NativeGenerationSnapshot>,
    model: &Arc<dyn Embedder>,
) {
    assert_eq!(session.receipt.authority, selected.authority);
    assert_eq!(session.receipt.identity, selected.identity);
    assert!(Arc::ptr_eq(&session.snapshot, snapshot));
    assert!(Arc::ptr_eq(&session.models.models().unwrap().fast, model));
}

#[test]
fn model_activation_installs_the_whole_selection_and_updates_use_the_new_producer() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (label, tier, ann, change_space) in [
            ("activate-fast-exact", EmbedderTierTag::Fast, false, false),
            (
                "activate-quality-exact",
                EmbedderTierTag::Quality,
                false,
                true,
            ),
            ("activate-fast-ann", EmbedderTierTag::Fast, true, true),
            (
                "activate-quality-ann",
                EmbedderTierTag::Quality,
                true,
                false,
            ),
        ] {
            let parent = private_parent(label);
            let old_model = provider();
            let new_model = replacement_provider(change_space);
            let first = publish(&cx, &parent, None, &old_model, tier, "first", ann).await;
            let mut session = open_session(&cx, first.clone(), &old_model, false)
                .with_filter(
                    &cx,
                    Some(Filter::parse(r#"{"id_prefix":"retry-"}"#).unwrap()),
                    MAX_INPUT_BYTES,
                )
                .unwrap()
                .with_updates(2, MAX_INPUT_BYTES)
                .unwrap();
            session.allow_model_activation = true;
            let retained = Arc::clone(&session.snapshot);
            let before = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({"query":"retry", "limit":2})],
            )
            .await;
            assert_eq!(before[1]["results"][0]["doc_id"], "retry-first");
            let next = publish(
                &cx,
                &parent,
                Some(&retained),
                &new_model,
                tier,
                "next",
                !ann,
            )
            .await;
            let authority_bytes = fs::read(first.root.join("AUTHORITY")).unwrap();
            let receipt_bytes = fs::read(parent.join("next.json")).unwrap();
            let old_calls = old_model.calls.load(Ordering::Relaxed);
            let new_calls = new_model.calls.load(Ordering::Relaxed);
            let replacement: Arc<dyn Embedder> = new_model.clone();
            let request = activation_request(&activation_message(&parent, &first, "next"));
            session
                .activate_model_with_loader(&cx, &request, |_, path, selected_tier, options| {
                    assert_eq!(selected_tier, tier);
                    match tier {
                        EmbedderTierTag::Fast => {
                            assert_eq!(path, Some(parent.join("explicit-model-root").as_path()));
                            assert!(options.backend.is_none());
                            assert!(options.directory.is_none());
                        }
                        EmbedderTierTag::Quality => {
                            assert!(path.is_none());
                            assert_eq!(options.backend, Some(models::QualityBackend::NativeInt8));
                            assert_eq!(
                                options.directory,
                                Some(parent.join("explicit-native-model"))
                            );
                        }
                    }
                    Ok(Arc::clone(&replacement))
                })
                .unwrap();
            assert_eq!(session.receipt.authority, next.authority);
            assert_eq!(session.receipt.identity, next.identity);
            assert!(Arc::ptr_eq(
                &session.models.models().unwrap().fast,
                &replacement
            ));
            assert_eq!(session.scope.as_ref().unwrap().len(), 1);
            assert!(
                !session.allow_activation,
                "model activation is separately granted"
            );
            assert_eq!(
                fs::read(first.root.join("AUTHORITY")).unwrap(),
                authority_bytes
            );
            assert_eq!(fs::read(parent.join("next.json")).unwrap(), receipt_bytes);
            assert_eq!(old_model.calls.load(Ordering::Relaxed), old_calls);
            assert_eq!(new_model.calls.load(Ordering::Relaxed), new_calls);

            let after = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({"query":"retry", "limit":2})],
            )
            .await;
            assert_eq!(after[1]["results"].as_array().unwrap().len(), 1);
            assert_eq!(after[1]["results"][0]["doc_id"], "retry-next");
            assert_eq!(after[1]["authority"], serde_json::json!(next.authority));
            assert_eq!(new_model.calls.load(Ordering::Relaxed), new_calls + 1);
            assert_eq!(old_model.calls.load(Ordering::Relaxed), old_calls);
            assert_eq!(
                retained
                    .search_text(&cx, old_model.as_ref(), "retry", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-first"
            );

            let update = update_message(
                &parent,
                "after-model-change",
                &next,
                8,
                serde_json::json!([
                    {"op":"upsert", "id":"retry-next", "content":"retry with new producer", "metadata":{"version":"updated"}}
                ]),
            );
            let response = session
                .update(&cx, serde_json::from_value(update).unwrap())
                .await
                .unwrap();
            assert_eq!(response["publication"], "committed");
            assert_eq!(response["serving"], "activated");
            assert_eq!(session.receipt.identity, next.identity);
            assert_eq!(new_model.calls.load(Ordering::Relaxed), new_calls + 2);
            assert_eq!(old_model.calls.load(Ordering::Relaxed), old_calls + 1);
            assert_eq!(session.scope.as_ref().unwrap().len(), 1);
            let reopened = session.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert!(
                reopened
                    .search_text(&cx, old_model.as_ref(), "retry", 1)
                    .await
                    .is_err()
            );
            assert_eq!(
                reopened
                    .search_text(&cx, new_model.as_ref(), "retry", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-next"
            );
        }
    });
}

#[test]
fn model_activation_grants_and_preflight_refuse_before_loading_and_keep_the_old_scope() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("model-activation-preflight");
        let old_model = provider();
        let new_model = replacement_provider(true);
        let first = publish(
            &cx,
            &parent,
            None,
            &old_model,
            EmbedderTierTag::Fast,
            "first",
            false,
        )
        .await;
        let mut session = open_session(&cx, first.clone(), &old_model, true)
            .with_filter(
                &cx,
                Some(Filter::parse(r#"{"id_prefix":"retry-"}"#).unwrap()),
                MAX_INPUT_BYTES,
            )
            .unwrap();
        let mut lexical = AuthoritySession::new(
            &cx,
            admit_root(&first.root).unwrap(),
            first.clone(),
            first.open(&cx, DEFAULT_IMAGE_BYTES).unwrap(),
            AuthorityModels::Lexical,
            (1, DEFAULT_IMAGE_BYTES),
            true,
        )
        .unwrap();
        lexical.allow_model_activation = true;
        let retained = Arc::clone(&session.snapshot);
        let retained_model = Arc::clone(&session.models.models().unwrap().fast);
        let next = publish(
            &cx,
            &parent,
            Some(&retained),
            &new_model,
            EmbedderTierTag::Fast,
            "next",
            true,
        )
        .await;
        let message = activation_message(&parent, &first, "next");
        let request = activation_request(&message);
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                panic!("ordinary activation permission cannot load a replacement model")
            })
            .unwrap_err();
        assert!(error.to_string().contains("model_activation_disabled"));
        let error = lexical
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                panic!("lexical-only sessions cannot acquire a model")
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("model_activation_requires_hybrid")
        );
        session.allow_model_activation = true;

        let mut wrong_tier = next.clone();
        wrong_tier.tier = EmbedderTierTag::Quality;
        save_receipt(&wrong_tier, &parent.join("wrong-tier.json")).unwrap();
        let mut wrong_input = next.clone();
        let mut changed = wrong_input.identity.identity.clone();
        changed.input.canonicalization.push_str("-changed");
        changed.space.input_contract_fingerprint = changed.input.fingerprint();
        changed.producer.space_fingerprint = changed.space.fingerprint();
        wrong_input.identity = changed.freeze().unwrap();
        save_receipt(&wrong_input, &parent.join("wrong-input.json")).unwrap();
        let mut wrong_binding = next.clone();
        wrong_binding.generation = new_generation(next.generation.sequence + 1).unwrap();
        save_receipt(&wrong_binding, &parent.join("wrong-generation.json")).unwrap();
        let mut wrong_root = next.clone();
        wrong_root.root = parent.join("other-root");
        save_receipt(&wrong_root, &parent.join("wrong-root.json")).unwrap();

        for (field, value, code) in [
            (
                "expected_authority",
                serde_json::json!(next.authority),
                "stale_expected_authority",
            ),
            (
                "receipt",
                serde_json::json!(parent.join("first.json")),
                "model_activation_not_newer",
            ),
            (
                "receipt",
                serde_json::json!(parent.join("wrong-tier.json")),
                "model_activation_root_or_tier_mismatch",
            ),
            (
                "receipt",
                serde_json::json!(parent.join("wrong-root.json")),
                "model_activation_root_or_tier_mismatch",
            ),
            (
                "receipt",
                serde_json::json!(parent.join("wrong-input.json")),
                "model_activation_input_contract_mismatch",
            ),
            (
                "receipt",
                serde_json::json!(parent.join("wrong-generation.json")),
                "model_activation_admission_failed",
            ),
            (
                "model_dir",
                serde_json::Value::Null,
                "model_activation_model_path_required",
            ),
        ] {
            let mut refused = message.clone();
            refused[field] = value;
            let refused = activation_request(&refused);
            let error = session
                .activate_model_with_loader(&cx, &refused, |_, _, _, _| {
                    panic!("preflight refusal must precede model loading")
                })
                .unwrap_err();
            assert!(error.to_string().contains(code), "{error}");
            assert_selection_unchanged(&session, &first, &retained, &retained_model);
            assert_eq!(session.scope.as_ref().unwrap().len(), 1);
        }
        session.max_source_bytes = 1;
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                panic!("fixed-scope source admission must precede model loading")
            })
            .unwrap_err();
        assert!(error.to_string().contains("model_activation_scope_failed"));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
        session.max_source_bytes = MAX_INPUT_BYTES;

        // Exercise the real protocol parser/dispatcher with only the ordinary
        // activation grant: the failed request must leave the following query usable.
        session.allow_model_activation = false;
        let replies = exchange(
            &cx,
            &mut session,
            &[message, serde_json::json!({"query":"retry"})],
        )
        .await;
        assert_eq!(replies[1]["code"], "model_activation_disabled");
        assert_eq!(replies[2]["results"][0]["doc_id"], "retry-first");
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
    });
}

struct DriftingActivationProvider {
    provider: Arc<Provider>,
    changed: EmbeddingIdentityBundleV1,
    identity_reads: AtomicUsize,
}

impl Embedder for DriftingActivationProvider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(
            if self.identity_reads.fetch_add(1, Ordering::Relaxed) == 0 {
                &self.provider.identity
            } else {
                &self.changed
            },
        )
    }
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.provider.embed(cx, text)
    }
    fn dimension(&self) -> usize {
        self.provider.dimension()
    }
    fn id(&self) -> &str {
        self.provider.id()
    }
    fn model_name(&self) -> &str {
        self.provider.model_name()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        self.provider.category()
    }
}

#[test]
fn model_activation_load_failure_wrong_producer_drift_and_cancellation_preserve_the_selection() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("model-activation-load-failures");
        let old_model = provider();
        let new_model = replacement_provider(true);
        let first = publish(
            &cx,
            &parent,
            None,
            &old_model,
            EmbedderTierTag::Quality,
            "first",
            true,
        )
        .await;
        let mut session = open_session(&cx, first.clone(), &old_model, false);
        session.allow_model_activation = true;
        let retained = Arc::clone(&session.snapshot);
        let retained_model = Arc::clone(&session.models.models().unwrap().fast);
        publish(
            &cx,
            &parent,
            Some(&retained),
            &new_model,
            EmbedderTierTag::Quality,
            "next",
            false,
        )
        .await;
        let authority_bytes = fs::read(first.root.join("AUTHORITY")).unwrap();
        let request = activation_request(&activation_message(&parent, &first, "next"));
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                Err(bad("private-model-path-and-provider-error"))
            })
            .unwrap_err();
        assert!(error.to_string().contains("model_activation_load_failed"));
        assert!(!error.to_string().contains("private-model"));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);

        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| Ok(old_model.clone()))
            .unwrap_err();
        assert!(error.to_string().contains("producer_identity_mismatch"));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
        let mut changed = new_model.identity.clone();
        changed.producer.implementation_revision.push_str("-drift");
        let drifting = Arc::new(DriftingActivationProvider {
            provider: Arc::clone(&new_model),
            changed,
            identity_reads: AtomicUsize::new(0),
        });
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| Ok(drifting.clone()))
            .unwrap_err();
        assert!(error.to_string().contains("producer_identity_mismatch"));
        assert_eq!(drifting.identity_reads.load(Ordering::Relaxed), 2);
        assert_selection_unchanged(&session, &first, &retained, &retained_model);

        let error = session
            .activate_model_with_loader(&cx, &request, |cx, _, _, _| {
                cx.set_cancel_requested(true);
                Ok(new_model.clone())
            })
            .unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::Cancelled { .. })
        ));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
        cx.set_cancel_requested(true);
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                panic!("an already cancelled request must not load a model")
            })
            .unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::Cancelled { .. })
        ));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
        assert_eq!(
            fs::read(first.root.join("AUTHORITY")).unwrap(),
            authority_bytes
        );
        let replies = exchange(&cx, &mut session, &[serde_json::json!({"query":"retry"})]).await;
        assert_eq!(replies[1]["results"][0]["doc_id"], "retry-first");
    });
}

#[test]
fn model_activation_rechecks_the_selected_head_after_loading() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("model-activation-head-race");
        let old_model = provider();
        let new_model = replacement_provider(true);
        let first = publish(
            &cx,
            &parent,
            None,
            &old_model,
            EmbedderTierTag::Fast,
            "first",
            false,
        )
        .await;
        let mut session = open_session(&cx, first.clone(), &old_model, false);
        session.allow_model_activation = true;
        let retained = Arc::clone(&session.snapshot);
        let retained_model = Arc::clone(&session.models.models().unwrap().fast);
        let next = publish(
            &cx,
            &parent,
            Some(&retained),
            &new_model,
            EmbedderTierTag::Fast,
            "next",
            true,
        )
        .await;
        let root = admit_root(&first.root).unwrap();
        let next_reader = next.open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES).unwrap();
        let model: Arc<dyn Embedder> = new_model.clone();
        let later =
            NativeIndexBuilder::new(parent.join("later-work"), new_generation(3).unwrap(), model)
                .unwrap()
                .add_documents([IndexableDocument::new("retry-later", "retry later head")])
                .build_exact_generation(
                    &cx,
                    EmbedderTierTag::Fast,
                    CommitRange { low: 1, high: 8 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
        let request = activation_request(&activation_message(&parent, &first, "next"));
        let error = session
            .activate_model_with_loader(&cx, &request, |cx, _, _, _| {
                // A competing publisher wins while the model-loading operation is in
                // progress. The candidate was valid before this callback was entered.
                stage_and_publish(
                    cx,
                    &root,
                    &first.root,
                    &parent.join("later.json"),
                    &later,
                    Some(&next_reader),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap();
                Ok(new_model.clone())
            })
            .unwrap_err();
        assert!(error.to_string().contains("model_activation_head_changed"));
        assert_selection_unchanged(&session, &first, &retained, &retained_model);
        let later = Receipt::read(&parent.join("later.json")).unwrap();
        assert_eq!(
            later.open(&cx, DEFAULT_IMAGE_BYTES).unwrap().authority(),
            later.authority().unwrap()
        );
        assert_eq!(
            later.authority().unwrap().sequence,
            next.authority().unwrap().sequence + 1
        );
        let replies = exchange(&cx, &mut session, &[serde_json::json!({"query":"retry"})]).await;
        assert_eq!(replies[1]["results"][0]["doc_id"], "retry-first");
    });
}

async fn publish_progressive_pair(
    cx: &Cx,
    parent: &Path,
    previous: Option<&NativeGenerationSnapshot>,
    providers: (&Arc<Provider>, &Arc<Provider>),
    label: &str,
    ann: (bool, bool),
) -> Receipt {
    let root_path = parent.join("root");
    let root = if previous.is_some() {
        admit_root(&root_path).unwrap()
    } else {
        create_root(&root_path).expect("qualified Linux root is required")
    };
    let sequence = previous.map_or(1, |snapshot| snapshot.authority().sequence + 1);
    let retrieval = |enabled| {
        if enabled {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 27,
            }
        } else {
            NativeBuildRetrieval::Exact
        }
    };
    let built = NativeIndexBuilder::new(
        parent.join(format!("{label}-work")),
        new_generation(sequence).unwrap(),
        providers.0.clone(),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F32, retrieval(ann.0))
    .with_quality_embedder(providers.1.clone())
    .unwrap()
    .with_quality_storage(NativeBuildPrecision::F32, retrieval(ann.1))
    .unwrap()
    .add_documents([
        IndexableDocument::new(format!("retry-{label}"), "retry network requests")
            .with_metadata("tenant", "blue"),
        IndexableDocument::new(format!("garden-{label}"), "garden flowers")
            .with_metadata("tenant", "red"),
    ])
    .build_progressive_generation(
        cx,
        CommitRange {
            low: 1,
            high: sequence + 4,
        },
        DEFAULT_IMAGE_BYTES,
    )
    .await
    .unwrap();
    stage_and_publish(
        cx,
        &root,
        &root_path,
        &parent.join(format!("{label}.json")),
        &built,
        previous,
        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
    )
    .unwrap()
    .receipt
}

#[test]
fn full_model_activation_validates_both_independent_producers_before_installing_either() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent("full-model-activation");
        let old_fast = provider();
        let old_quality = replacement_provider(false);
        let new_fast = replacement_provider(true);
        let new_quality = Arc::new(Provider {
            identity: {
                let mut changed = new_fast.identity.clone();
                changed.space.immutable_revision = "independent-next-quality".to_owned();
                changed.producer.implementation_revision = "independent-next-quality".to_owned();
                changed.producer.space_fingerprint = changed.space.fingerprint();
                changed.validate().unwrap();
                changed
            },
            calls: AtomicUsize::new(0),
        });
        let first = publish_progressive_pair(
            &cx,
            &parent,
            None,
            (&old_fast, &old_quality),
            "first",
            (false, true),
        )
        .await;
        let mut session = AuthoritySession::new(
            &cx,
            admit_root(&first.root).unwrap(),
            first.clone(),
            first.open(&cx, DEFAULT_IMAGE_BYTES).unwrap(),
            Models {
                fast: old_fast.clone(),
                quality: Some(old_quality.clone()),
            },
            (2, DEFAULT_IMAGE_BYTES),
            false,
        )
        .unwrap()
        .with_filter(
            &cx,
            Some(Filter::parse(r#"{"metadata":{"tenant":"blue"}}"#).unwrap()),
            MAX_INPUT_BYTES,
        )
        .unwrap()
        .with_updates(2, MAX_INPUT_BYTES)
        .unwrap();
        session.allow_model_activation = true;
        let retained = Arc::clone(&session.snapshot);
        let next = publish_progressive_pair(
            &cx,
            &parent,
            Some(&retained),
            (&new_fast, &new_quality),
            "next",
            (true, false),
        )
        .await;
        let authority_bytes = fs::read(first.root.join("AUTHORITY")).unwrap();
        let mut message = activation_message(&parent, &first, "next");
        message["quality_backend"] = serde_json::json!("native-int8");
        message["quality_model_dir"] = serde_json::json!(parent.join("new-quality-model"));
        let request = activation_request(&message);
        let mut loaded_tiers = Vec::new();
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, tier, _| {
                loaded_tiers.push(tier);
                match tier {
                    EmbedderTierTag::Fast => Ok(new_fast.clone()),
                    EmbedderTierTag::Quality => Err(bad("private-quality-load-failure")),
                }
            })
            .unwrap_err();
        assert_eq!(
            loaded_tiers,
            [EmbedderTierTag::Fast, EmbedderTierTag::Quality]
        );
        assert!(error.to_string().contains("model_activation_load_failed"));
        assert!(!error.to_string().contains("private-quality"));
        assert!(Arc::ptr_eq(&session.snapshot, &retained));
        assert_eq!(session.receipt.identity, first.identity);
        assert_eq!(session.receipt.quality_identity, first.quality_identity);

        let mut quality_input_changed = next.clone();
        let mut changed = quality_input_changed
            .quality_identity
            .as_ref()
            .unwrap()
            .identity
            .clone();
        changed.input.canonicalization.push_str("-changed");
        changed.space.input_contract_fingerprint = changed.input.fingerprint();
        changed.producer.space_fingerprint = changed.space.fingerprint();
        quality_input_changed.quality_identity = Some(changed.freeze().unwrap());
        save_receipt(
            &quality_input_changed,
            &parent.join("changed-quality-input.json"),
        )
        .unwrap();
        let mut refused = message.clone();
        refused["receipt"] = serde_json::json!(parent.join("changed-quality-input.json"));
        let error = session
            .activate_model_with_loader(&cx, &activation_request(&refused), |_, _, _, _| {
                panic!("a changed quality input contract cannot load either producer")
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("model_activation_input_contract_mismatch")
        );
        let mut missing_quality = next.clone();
        missing_quality.quality_identity = None;
        missing_quality.schema = RECEIPT_SCHEMA.to_owned();
        save_receipt(&missing_quality, &parent.join("missing-quality.json")).unwrap();
        refused["receipt"] = serde_json::json!(parent.join("missing-quality.json"));
        let error = session
            .activate_model_with_loader(&cx, &activation_request(&refused), |_, _, _, _| {
                panic!("model activation cannot remove the quality tier")
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("model_activation_topology_mismatch")
        );

        let mut changed = new_quality.identity.clone();
        changed
            .producer
            .implementation_revision
            .push_str("-late-drift");
        let drifting = Arc::new(DriftingActivationProvider {
            provider: Arc::clone(&new_quality),
            changed,
            identity_reads: AtomicUsize::new(0),
        });
        let error = session
            .activate_model_with_loader(&cx, &request, |_, _, tier, _| match tier {
                EmbedderTierTag::Fast => Ok(new_fast.clone()),
                EmbedderTierTag::Quality => Ok(drifting.clone()),
            })
            .unwrap_err();
        assert!(error.to_string().contains("producer_identity_mismatch"));
        assert_eq!(drifting.identity_reads.load(Ordering::Relaxed), 2);
        assert!(Arc::ptr_eq(&session.snapshot, &retained));
        let old_calls = (
            old_fast.calls.load(Ordering::Relaxed),
            old_quality.calls.load(Ordering::Relaxed),
        );
        let before = exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":"how do I retry network requests", "mode":"full"})],
        )
        .await;
        assert_eq!(before[1]["results"][0]["doc_id"], "retry-first");
        assert_eq!(old_fast.calls.load(Ordering::Relaxed), old_calls.0 + 1);
        assert_eq!(old_quality.calls.load(Ordering::Relaxed), old_calls.1 + 1);

        loaded_tiers.clear();
        session
            .activate_model_with_loader(&cx, &request, |_, root, tier, options| {
                assert_eq!(root, Some(parent.join("explicit-model-root").as_path()));
                loaded_tiers.push(tier);
                match tier {
                    EmbedderTierTag::Fast => {
                        assert!(options.backend.is_none());
                        Ok(new_fast.clone())
                    }
                    EmbedderTierTag::Quality => {
                        assert_eq!(options.backend, Some(models::QualityBackend::NativeInt8));
                        assert_eq!(options.directory, Some(parent.join("new-quality-model")));
                        Ok(new_quality.clone())
                    }
                }
            })
            .unwrap();
        assert_eq!(
            loaded_tiers,
            [EmbedderTierTag::Fast, EmbedderTierTag::Quality]
        );
        assert_eq!(session.receipt.authority, next.authority);
        assert_eq!(session.receipt.identity, next.identity);
        assert_eq!(session.receipt.quality_identity, next.quality_identity);
        assert_eq!(session.scope.as_ref().unwrap().len(), 1);
        assert_eq!(
            fs::read(first.root.join("AUTHORITY")).unwrap(),
            authority_bytes
        );
        let new_calls = (
            new_fast.calls.load(Ordering::Relaxed),
            new_quality.calls.load(Ordering::Relaxed),
        );
        let after = exchange(
            &cx,
            &mut session,
            &[serde_json::json!({"query":"how do I retry network requests", "mode":"full"})],
        )
        .await;
        assert_eq!(after[1]["phase"], "refined");
        assert_eq!(after[1]["results"].as_array().unwrap().len(), 1);
        assert_eq!(after[1]["results"][0]["doc_id"], "retry-next");
        assert_eq!(new_fast.calls.load(Ordering::Relaxed), new_calls.0 + 1);
        assert_eq!(new_quality.calls.load(Ordering::Relaxed), new_calls.1 + 1);
        assert_eq!(old_fast.calls.load(Ordering::Relaxed), old_calls.0 + 1);
        assert_eq!(old_quality.calls.load(Ordering::Relaxed), old_calls.1 + 1);

        let update = update_message(
            &parent,
            "full-after-model-change",
            &next,
            8,
            serde_json::json!([
                {"op":"upsert", "id":"retry-next", "content":"retry with both new producers", "metadata":{"tenant":"blue"}}
            ]),
        );
        let response = session
            .update(&cx, serde_json::from_value(update).unwrap())
            .await
            .unwrap();
        assert_eq!(response["publication"], "committed");
        assert_eq!(new_fast.calls.load(Ordering::Relaxed), new_calls.0 + 2);
        assert_eq!(new_quality.calls.load(Ordering::Relaxed), new_calls.1 + 2);
        assert_eq!(session.receipt.identity, next.identity);
        assert_eq!(session.receipt.quality_identity, next.quality_identity);
    });
}
