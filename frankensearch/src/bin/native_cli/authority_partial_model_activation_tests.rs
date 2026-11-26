use super::*;
use sha2::Sha256;

const QUERY: &str = "retry partial model activation";

fn replacement(previous: &Arc<Provider>, label: &str) -> Arc<Provider> {
    let mut identity = previous.identity.clone();
    identity.producer.implementation_revision.push_str(label);
    identity.space.immutable_revision.push_str(label);
    identity.space.artifacts[0].sha256 = "c".repeat(64);
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.validate().unwrap();
    Arc::new(Provider {
        identity,
        calls: AtomicUsize::new(0),
    })
}

fn independent_input(previous: &Arc<Provider>) -> Arc<Provider> {
    let mut identity = previous.identity.clone();
    identity
        .input
        .canonicalization
        .push_str("-independent-input");
    identity.space.input_contract_fingerprint = identity.input.fingerprint();
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity.validate().unwrap();
    Arc::new(Provider {
        identity,
        calls: AtomicUsize::new(0),
    })
}

fn session(
    cx: &Cx,
    selected: &Receipt,
    fast: &Arc<Provider>,
    quality: &Arc<Provider>,
) -> AuthoritySession {
    let root = admit_root(&selected.root).unwrap();
    let snapshot = selected
        .open_on_root(cx, &root, DEFAULT_IMAGE_BYTES)
        .unwrap();
    let mut session = AuthoritySession::new(
        cx,
        root,
        selected.clone(),
        snapshot,
        models(fast, quality),
        (4, DEFAULT_IMAGE_BYTES),
        false,
    )
    .unwrap()
    .with_query_cache(cx, query_cache::Config::parse("4").unwrap())
    .unwrap()
    .with_filter(
        cx,
        Some(Filter::parse(r#"{"ids":["quality","fast-only","new"]}"#).unwrap()),
        MAX_INPUT_BYTES,
    )
    .unwrap()
    .with_updates(2, MAX_INPUT_BYTES)
    .unwrap();
    session.allow_model_activation = true;
    session
}

fn request(parent: &Path, selected: &Receipt, label: &str) -> model_activation::Request {
    serde_json::from_value(serde_json::json!({
        "op":"activate_model", "id":"partial-model-change",
        "receipt":parent.join(format!("{label}.json")),
        "expected_authority":selected.authority,
        "model_dir":parent.join("explicit-fast-model"),
        "quality_backend":"native-f32",
        "quality_model_dir":parent.join("explicit-quality-model"),
    }))
    .unwrap()
}

fn cache_pair(session: &AuthoritySession) -> [Arc<dyn Embedder>; 2] {
    let models = session.query_models().models().unwrap();
    [
        Arc::clone(&models.fast),
        Arc::clone(models.quality.as_ref().unwrap()),
    ]
}

fn assert_preserved(
    session: &AuthoritySession,
    selected: &Receipt,
    snapshot: &Arc<NativeGenerationSnapshot>,
    providers: &Models,
    caches: &[Arc<dyn Embedder>; 2],
) {
    assert_eq!(session.receipt.authority, selected.authority);
    assert_eq!(session.receipt.identity, selected.identity);
    assert_eq!(session.receipt.quality_identity, selected.quality_identity);
    assert!(Arc::ptr_eq(&session.snapshot, snapshot));
    let current = session.models.models().unwrap();
    assert!(Arc::ptr_eq(&current.fast, &providers.fast));
    assert!(Arc::ptr_eq(
        current.quality.as_ref().unwrap(),
        providers.quality.as_ref().unwrap()
    ));
    let current = cache_pair(session);
    assert!(Arc::ptr_eq(&caches[0], &current[0]));
    assert!(Arc::ptr_eq(&caches[1], &current[1]));
    assert_eq!(session.scope.as_ref().unwrap().len(), 2);
}

async fn migrate(
    cx: &Cx,
    parent: &Path,
    previous: &NativeGenerationSnapshot,
    fast: &Arc<Provider>,
    quality: &Arc<Provider>,
    label: &str,
) -> NativeBuiltExactGeneration {
    previous
        .begin_partial_quality_model_migration(
            cx,
            parent.join(format!("{label}-work")),
            new_generation(
                previous
                    .retained_snapshot()
                    .manifest()
                    .generation()
                    .sequence
                    + 1,
            )
            .unwrap(),
            fast.clone(),
            quality.clone(),
            MAX_INPUT_BYTES,
        )
        .unwrap()
        .build(
            cx,
            previous.metadata().commit_range.clone(),
            DEFAULT_IMAGE_BYTES,
        )
        .await
        .unwrap()
}

fn publish_candidate(
    cx: &Cx,
    parent: &Path,
    previous: &NativeGenerationSnapshot,
    built: &NativeBuiltExactGeneration,
    label: &str,
    enforce_rebuild_policy: bool,
) -> Receipt {
    let root_path = parent.join("root");
    let root = admit_root(&root_path).unwrap();
    let prepared = if enforce_rebuild_policy {
        prepare_partial_quality_rebuild(cx, built, previous, DEFAULT_IMAGE_BYTES).unwrap()
    } else {
        // The core publisher accepts independently admitted components. These
        // negative cases test warm admission independently of the CLI rebuild
        // policy, without forging a receipt or weakening component validation.
        let writer = WriterIdentityV1::for_current_process().unwrap();
        let object = new_generation(1).unwrap().nonce;
        let mut fence = Sha256::new();
        fence.update(b"partial-model-activation-negative");
        fence.update(writer.writer_id);
        fence.update(writer.nonce);
        fence.update(object);
        PreparedPublication {
            candidate: built
                .candidate(
                    object,
                    Some(previous.authority()),
                    GenerationAuthorityActionV1::Activate,
                    fence.finalize().into(),
                )
                .unwrap(),
            root_id: previous.retained_snapshot().root_id(),
            expected: previous.retained_snapshot().authority_pair(),
            writer,
        }
    };
    stage_prepared_and_publish(
        cx,
        &root,
        &root_path,
        &parent.join(format!("{label}.json")),
        built,
        prepared,
        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
    )
    .unwrap()
    .receipt
}

#[test]
fn partial_model_activation_replaces_spaces_caches_and_preserves_scopes_and_future_updates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (change_fast, change_quality, ann) in [
            (false, true, false),
            (true, false, true),
            (true, true, true),
        ] {
            let parent = private_parent();
            let fast = provider(false);
            let quality = independent_input(&provider(true));
            let (_, selected) = publish_partial_cli(&cx, &parent, &fast, &quality, ann).await;
            let mut session = session(&cx, &selected, &fast, &quality);
            let retained = Arc::clone(&session.snapshot);
            assert_ne!(
                selected.identity.identity.input,
                selected.quality_identity.as_ref().unwrap().identity.input
            );
            let query = serde_json::json!({"query":QUERY,"mode":"partial","stream":true});
            let before = exchange(&cx, &mut session, std::slice::from_ref(&query)).await;
            assert_eq!(before[2]["results"].as_array().unwrap().len(), 2);
            let old_caches = cache_pair(&session);
            let next_fast = if change_fast {
                replacement(&fast, "-next-fast")
            } else {
                Arc::clone(&fast)
            };
            let next_quality = if change_quality {
                replacement(&quality, "-next-quality")
            } else {
                Arc::clone(&quality)
            };
            let calls_before_build = calls(&next_fast, &next_quality);
            let built = migrate(
                &cx,
                &parent,
                &retained,
                &next_fast,
                &next_quality,
                "migrated",
            )
            .await;
            assert_eq!(
                calls(&next_fast, &next_quality),
                (
                    calls_before_build.0 + if change_fast { 3 } else { 0 },
                    calls_before_build.1 + usize::from(change_quality),
                )
            );
            let next = publish_candidate(&cx, &parent, &retained, &built, "migrated", true);
            let authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
            let calls_before_activation = calls(&next_fast, &next_quality);
            let mut loads = 0;
            session
                .activate_model_with_loader(
                    &cx,
                    &request(&parent, &selected, "migrated"),
                    |_, _, tier, _| {
                        loads += 1;
                        let provider: Arc<dyn Embedder> = match tier {
                            EmbedderTierTag::Fast => next_fast.clone(),
                            EmbedderTierTag::Quality => next_quality.clone(),
                        };
                        Ok(provider)
                    },
                )
                .unwrap();
            assert_eq!(loads, 2);
            assert_eq!(calls(&next_fast, &next_quality), calls_before_activation);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority
            );
            assert_eq!(session.receipt.authority, next.authority);
            assert_eq!(session.snapshot.topology(), selected.topology());
            assert_eq!(
                session.snapshot.partial_quality_document_ids(&cx).unwrap(),
                ["quality"]
            );
            assert_eq!(session.scope.as_ref().unwrap().len(), 2);
            let next_caches = cache_pair(&session);
            assert!(!Arc::ptr_eq(&old_caches[0], &next_caches[0]));
            assert!(!Arc::ptr_eq(&old_caches[1], &next_caches[1]));
            let replies = exchange(&cx, &mut session, &[query.clone(), query.clone()]).await;
            assert_eq!(replies[2]["results"], replies[4]["results"]);
            assert_eq!(
                calls(&next_fast, &next_quality),
                (calls_before_activation.0 + 1, calls_before_activation.1 + 1)
            );
            let covered = replies[2]["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|hit| hit["doc_id"] == "quality")
                .unwrap();
            assert!(!covered["quality_score"].is_null());
            let uncovered = replies[2]["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|hit| hit["doc_id"] == "fast-only")
                .unwrap();
            assert!(uncovered["quality_score"].is_null());

            let before_update = calls(&next_fast, &next_quality);
            let receipt_path = parent.join("after-activation-update.json");
            let response = exchange(&cx, &mut session, &[serde_json::json!({
                "op":"update", "id":"after-model-change", "expected_authority":next.authority,
                "work_dir":parent.join("after-activation-update-work"), "new_receipt":receipt_path,
                "source_low":retained.metadata().commit_range.low,
                "source_high":retained.metadata().commit_range.high + 1,
                "changes":[
                    {"op":"upsert","id":"quality","content":"retry changed covered document"},
                    {"op":"upsert","id":"new","content":"retry new document remains uncovered"}
                ]
            })]).await;
            assert_eq!(response[1]["event"], "updated");
            assert_eq!(response[1]["scope"]["eligible_documents"], 3);
            assert_eq!(
                calls(&next_fast, &next_quality),
                (before_update.0 + 2, before_update.1 + 1)
            );
            let updated = Receipt::read(&receipt_path).unwrap();
            assert_eq!(updated.identity, next.identity);
            assert_eq!(updated.quality_identity, next.quality_identity);
            assert_eq!(updated.quality_documents, Some(1));
            updated.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let current_caches = cache_pair(&session);
            assert!(Arc::ptr_eq(&next_caches[0], &current_caches[0]));
            assert!(Arc::ptr_eq(&next_caches[1], &current_caches[1]));
            let after_update = calls(&next_fast, &next_quality);
            let mut filtered = query;
            filtered["filter"] = serde_json::json!({"ids":["new","garden"]});
            let response = exchange(&cx, &mut session, &[filtered]).await;
            assert_eq!(response[2]["results"].as_array().unwrap().len(), 1);
            assert_eq!(response[2]["results"][0]["doc_id"], "new");
            assert!(response[2]["results"][0]["quality_score"].is_null());
            assert_eq!(calls(&next_fast, &next_quality), after_update);
            let old = retained
                .search_progressive_with_partial_quality(
                    &cx,
                    fast.as_ref(),
                    quality.as_ref(),
                    QUERY,
                    3,
                )
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(old.len(), 3);
            assert!(!old.iter().any(|hit| hit.doc_id == "new"));
            assert_eq!(
                retained.partial_quality_document_ids(&cx).unwrap(),
                ["quality"]
            );
        }
    });
}

#[test]
fn partial_model_activation_refuses_subset_source_and_input_changes_before_loading() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for case in [
            "same-count-subset",
            "expanded-subset",
            "changed-source",
            "backward-source",
            "advanced-source",
            "quality-input",
            "fast-input",
        ] {
            let parent = private_parent();
            let fast = provider(false);
            let quality = independent_input(&provider(true));
            let (_, selected) = publish_partial_cli(&cx, &parent, &fast, &quality, false).await;
            let mut session = session(&cx, &selected, &fast, &quality);
            let retained = Arc::clone(&session.snapshot);
            let query = serde_json::json!({"query":QUERY,"mode":"partial","stream":true});
            exchange(&cx, &mut session, std::slice::from_ref(&query)).await;
            let caches = cache_pair(&session);
            let providers = models(&fast, &quality);
            let mut documents = retained.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
            let mut source = retained.metadata().commit_range.clone();
            let mut selected_ids = vec!["quality".to_owned()];
            let mut next_fast = replacement(&fast, "-negative-fast");
            let mut next_quality = replacement(&quality, "-negative-quality");
            let expected = match case {
                "same-count-subset" => {
                    selected_ids = vec!["fast-only".to_owned()];
                    "model_activation_partial_coverage_mismatch"
                }
                "expanded-subset" => {
                    selected_ids.push("fast-only".to_owned());
                    "model_activation_topology_mismatch"
                }
                "changed-source" => {
                    documents
                        .iter_mut()
                        .find(|doc| doc.id == "fast-only")
                        .unwrap()
                        .content
                        .push_str(" changed");
                    "model_activation_partial_source_mismatch"
                }
                "backward-source" => {
                    source.high -= 1;
                    "model_activation_partial_source_mismatch"
                }
                "advanced-source" => {
                    source.high += 1;
                    "model_activation_partial_source_mismatch"
                }
                "quality-input" => {
                    next_quality = independent_input(&next_quality);
                    "model_activation_input_contract_mismatch"
                }
                "fast-input" => {
                    next_fast = independent_input(&next_fast);
                    "model_activation_input_contract_mismatch"
                }
                _ => unreachable!(),
            };
            let built = NativeIndexBuilder::new(
                parent.join("rejected-work"),
                new_generation(2).unwrap(),
                next_fast.clone(),
            )
            .unwrap()
            .with_quality_embedder(next_quality.clone())
            .unwrap()
            .add_documents(documents)
            .build_partial_quality_generation(&cx, source, selected_ids, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
            let candidate = publish_candidate(&cx, &parent, &retained, &built, "rejected", false);
            candidate.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
            let mut loads = 0;
            let error = session
                .activate_model_with_loader(
                    &cx,
                    &request(&parent, &selected, "rejected"),
                    |_, _, _, _| {
                        loads += 1;
                        Err(bad("unreachable rejected partial model load"))
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains(expected), "case {case}: {error}");
            assert_eq!(loads, 0, "case {case}");
            assert_preserved(&session, &selected, &retained, &providers, &caches);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority
            );
            let before = calls(&fast, &quality);
            let response = exchange(&cx, &mut session, &[query]).await;
            assert_eq!(response[2]["results"].as_array().unwrap().len(), 2);
            assert_eq!(
                calls(&fast, &quality),
                before,
                "retained query caches survive refusal"
            );
        }
    });
}

#[test]
fn partial_model_activation_loader_failures_and_racing_head_preserve_the_warm_selection() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent();
        let fast = provider(false);
        let quality = independent_input(&provider(true));
        let (_, selected) = publish_partial_cli(&cx, &parent, &fast, &quality, true).await;
        let mut session = session(&cx, &selected, &fast, &quality);
        let query = serde_json::json!({"query":QUERY,"mode":"partial","stream":true});
        exchange(&cx, &mut session, std::slice::from_ref(&query)).await;
        let retained = Arc::clone(&session.snapshot);
        let caches = cache_pair(&session);
        let providers = models(&fast, &quality);
        let next_fast = replacement(&fast, "-next-fast");
        let next_quality = replacement(&quality, "-next-quality");
        let built = migrate(
            &cx,
            &parent,
            &retained,
            &next_fast,
            &next_quality,
            "migrated",
        )
        .await;
        let next = publish_candidate(&cx, &parent, &retained, &built, "migrated", true);
        let request = request(&parent, &selected, "migrated");
        let authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
        for case in [
            "wrong-fast",
            "wrong-quality",
            "load-error",
            "cancelled",
            "scope-budget",
        ] {
            if case == "scope-budget" {
                session.max_source_bytes = 1;
            }
            let mut loads = 0;
            let error = session
                .activate_model_with_loader(&cx, &request, |cx, _, tier, _| {
                    loads += 1;
                    if case == "load-error" {
                        return Err(bad("private-model-load-diagnostic"));
                    }
                    if case == "cancelled" {
                        cx.set_cancel_requested(true);
                    }
                    let provider: Arc<dyn Embedder> = match tier {
                        EmbedderTierTag::Fast if case == "wrong-fast" => fast.clone(),
                        EmbedderTierTag::Quality if case == "wrong-quality" => quality.clone(),
                        EmbedderTierTag::Fast => next_fast.clone(),
                        EmbedderTierTag::Quality => next_quality.clone(),
                    };
                    Ok(provider)
                })
                .unwrap_err();
            cx.set_cancel_requested(false);
            session.max_source_bytes = MAX_INPUT_BYTES;
            if case == "cancelled" {
                assert!(matches!(
                    error.downcast_ref::<SearchError>(),
                    Some(SearchError::Cancelled { .. })
                ));
            } else {
                let expected = match case {
                    "load-error" => "model_activation_load_failed",
                    "scope-budget" => "model_activation_scope_failed",
                    _ => "producer_identity_mismatch",
                };
                assert!(error.to_string().contains(expected), "case {case}: {error}");
                assert!(!error.to_string().contains("private-model-load-diagnostic"));
            }
            if case == "scope-budget" {
                assert_eq!(loads, 0);
            }
            assert_preserved(&session, &selected, &retained, &providers, &caches);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority
            );
        }
        let winner_source = next.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
        let winner = migrate(
            &cx,
            &parent,
            &winner_source,
            &next_fast,
            &next_quality,
            "winner",
        )
        .await;
        let mut advanced = false;
        let error = session
            .activate_model_with_loader(&cx, &request, |cx, _, tier, _| {
                if !advanced {
                    publish_candidate(cx, &parent, &winner_source, &winner, "winner", true);
                    advanced = true;
                }
                let provider: Arc<dyn Embedder> = match tier {
                    EmbedderTierTag::Fast => next_fast.clone(),
                    EmbedderTierTag::Quality => next_quality.clone(),
                };
                Ok(provider)
            })
            .unwrap_err();
        assert!(advanced);
        assert!(error.to_string().contains("model_activation_head_changed"));
        assert_preserved(&session, &selected, &retained, &providers, &caches);
        Receipt::read(&parent.join("winner.json"))
            .unwrap()
            .open(&cx, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let before = calls(&fast, &quality);
        let response = exchange(&cx, &mut session, &[query]).await;
        assert_eq!(response[2]["results"].as_array().unwrap().len(), 2);
        assert_eq!(calls(&fast, &quality), before);
    });
}

struct RollbackActivationFixture {
    parent: PathBuf,
    fast: Arc<Provider>,
    quality: Arc<Provider>,
    partial: Receipt,
    partial_snapshot: Arc<NativeGenerationSnapshot>,
    full: Receipt,
    session: AuthoritySession,
}

impl RollbackActivationFixture {
    async fn new(cx: &Cx, ann: bool) -> Self {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let mut initial = partial_index_options(&parent, ann);
        initial.partial_quality = false;
        initial.quality_ids = None;
        initial.receipt = parent.join("fast.json");
        index_with_loader(cx, &initial, &mut Vec::new(), |mode| {
            assert_eq!(mode, QueryMode::Fast);
            Ok(Models {
                fast: fast.clone(),
                quality: None,
            })
        })
        .await
        .unwrap();
        let first = Receipt::read(&initial.receipt).unwrap();
        let root = admit_root(&first.root).unwrap();
        let retained = first.open_on_root(cx, &root, DEFAULT_IMAGE_BYTES).unwrap();
        let built = retained
            .begin_quality_completion(
                cx,
                parent.join("rollback-partial-work"),
                new_generation(2).unwrap(),
                fast.clone(),
                quality.clone(),
                MAX_INPUT_BYTES,
            )
            .unwrap()
            .with_quality_storage(
                NativeBuildPrecision::F32,
                retained.retrieval_configuration(),
            )
            .unwrap()
            .build_partial(cx, ["quality".to_owned()], DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
        let prepared =
            prepare_partial_quality_completion(cx, &built, &retained, DEFAULT_IMAGE_BYTES)
                .unwrap();
        let partial = stage_prepared_and_publish(
            cx,
            &root,
            &first.root,
            &parent.join("partial.json"),
            &built,
            prepared,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        let built = partial
            .snapshot
            .begin_quality_completion(
                cx,
                parent.join("rollback-full-work"),
                new_generation(3).unwrap(),
                fast.clone(),
                quality.clone(),
                MAX_INPUT_BYTES,
            )
            .unwrap()
            .build(cx, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
        let prepared = prepare_publication_with_intent(
            &built,
            Some(&partial.snapshot),
            DEFAULT_IMAGE_BYTES,
            PublicationIntent::CompleteQuality,
        )
        .unwrap();
        let full = stage_prepared_and_publish(
            cx,
            &root,
            &first.root,
            &parent.join("full.json"),
            &built,
            prepared,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        let session = session(cx, &full.receipt, &fast, &quality);
        Self {
            parent,
            fast,
            quality,
            partial: partial.receipt,
            partial_snapshot: partial.snapshot,
            full: full.receipt,
            session,
        }
    }
}

fn publish_rollback_selection(
    cx: &Cx,
    parent: &Path,
    current: &Path,
    target: &str,
    label: &str,
) -> Receipt {
    let mut options = parse(&rollback_arguments()).unwrap().unwrap();
    options.from_receipt = Some(current.to_path_buf());
    options.target_receipt = Some(parent.join(format!("{target}.json")));
    options.receipt = parent.join(format!("{label}.json"));
    rollback(cx, &options, &mut Vec::new()).unwrap();
    Receipt::read(&options.receipt).unwrap()
}

fn rollback_activation_message(
    parent: &Path,
    selected: &Receipt,
    label: &str,
) -> serde_json::Value {
    serde_json::json!({
        "op":"activate_rollback", "id":"restore-selection",
        "receipt":parent.join(format!("{label}.json")),
        "expected_authority":selected.authority,
    })
}

#[test]
fn rollback_activation_restores_topologies_scopes_and_warm_caches_through_the_protocol() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let mut fixture = RollbackActivationFixture::new(&cx, ann).await;
            let source = fixture.session.snapshot.metadata().commit_range.clone();
            let expanded = exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({
                    "op":"update", "expected_authority":fixture.full.authority,
                    "work_dir":fixture.parent.join("expanded-work"),
                    "new_receipt":fixture.parent.join("expanded.json"),
                    "source_low":source.low, "source_high":source.high + 1,
                    "changes":[{"op":"upsert", "id":"new", "content":"retry new scoped source"}],
                })],
            )
            .await;
            assert_eq!(expanded[1]["event"], "updated");
            assert_eq!(expanded[1]["documents"], 4);
            assert_eq!(fixture.session.scope.as_ref().unwrap().len(), 3);
            let retained_full = Arc::clone(&fixture.session.snapshot);
            let primed = exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({"query":QUERY,"mode":"full"})],
            )
            .await;
            assert_eq!(primed[0]["rollback_activation_enabled"], true);
            assert_eq!(primed[1]["results"].as_array().unwrap().len(), 3);
            let mut current = fixture.parent.join("expanded.json");
            let mut retired_quality_cache = None;
            for (step, target) in ["partial", "fast", "partial", "full", "fast", "full", "partial"]
                .into_iter()
                .enumerate()
            {
                let previous = fixture.session.receipt.clone();
                let previous_reader = Arc::clone(&fixture.session.snapshot);
                let providers = fixture.session.models.models().unwrap().clone();
                let caches = fixture.session.query_models().models().unwrap().clone();
                let label = format!("rollback-{step}");
                let restored =
                    publish_rollback_selection(&cx, &fixture.parent, &current, target, &label);
                let target_receipt =
                    Receipt::read(&fixture.parent.join(format!("{target}.json"))).unwrap();
                assert_eq!(restored.generation, target_receipt.generation);
                assert!(
                    restored.authority().unwrap().sequence
                        > previous.authority().unwrap().sequence
                );
                let mut message = rollback_activation_message(&fixture.parent, &previous, &label);
                let needs_quality = previous.quality_identity.is_none()
                    && restored.quality_identity.is_some();
                let mode = if restored.quality_documents.is_some() {
                    "partial"
                } else if restored.quality_identity.is_some() {
                    "full"
                } else {
                    "fast"
                };
                let query = serde_json::json!({"query":QUERY,"mode":mode});
                let before = calls(&fixture.fast, &fixture.quality);
                let replies = if needs_quality {
                    message["quality_backend"] = serde_json::json!("native-f32");
                    message["quality_model_dir"] =
                        serde_json::json!(fixture.parent.join("explicit-restored-quality"));
                    let request: model_activation::Request =
                        serde_json::from_value(message).unwrap();
                    let mut loads = 0;
                    let activated = fixture
                        .session
                        .activate_model_with_loader(&cx, &request, |_, directory, tier, options| {
                            loads += 1;
                            assert_eq!(tier, EmbedderTierTag::Quality);
                            assert!(directory.is_none());
                            assert_eq!(options.backend, Some(models::QualityBackend::NativeF32));
                            assert_eq!(
                                options.directory,
                                Some(fixture.parent.join("explicit-restored-quality"))
                            );
                            Ok(fixture.quality.clone())
                        })
                        .unwrap();
                    assert_eq!(loads, 1);
                    assert_eq!(activated.models_loaded, 1);
                    assert!(activated.primary_retained);
                    assert!(!activated.quality_retained);
                    assert_eq!(calls(&fixture.fast, &fixture.quality), before);
                    exchange(&cx, &mut fixture.session, &[query]).await
                } else {
                    let replies = exchange(&cx, &mut fixture.session, &[message, query]).await;
                    assert_eq!(replies[1]["event"], "rollback_activated");
                    assert_eq!(replies[1]["operation"], "activate_rollback");
                    assert_eq!(replies[1]["models_loaded"], 0);
                    assert_eq!(replies[1]["primary_model_retained"], true);
                    assert_eq!(
                        replies[1]["quality_model_retained"],
                        restored.quality_identity.is_some()
                    );
                    assert_eq!(replies[1]["authority"], serde_json::json!(restored.authority));
                    replies
                };
                let capabilities = &replies[usize::from(!needs_quality)];
                assert_eq!(
                    capabilities["default_mode"],
                    if mode == "full" { "full" } else { "fast" }
                );
                let modes = match mode {
                    "partial" => serde_json::json!(["fast", "partial", "lexical"]),
                    "full" => serde_json::json!(["fast", "quality", "full", "lexical"]),
                    _ => serde_json::json!(["fast", "lexical"]),
                };
                assert_eq!(capabilities["query_modes"], modes);
                let result = replies.last().unwrap();
                assert_eq!(result["event"], "results");
                assert_eq!(result["authority"], serde_json::json!(restored.authority));
                assert_eq!(result["results"].as_array().unwrap().len(), 2);
                assert!(
                    result["results"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|hit| hit["doc_id"] != "garden" && hit["doc_id"] != "new")
                );
                assert_eq!(fixture.session.scope.as_ref().unwrap().len(), 2);
                assert_eq!(fixture.session.receipt.topology(), restored.topology());
                assert!(Arc::ptr_eq(
                    &providers.fast,
                    &fixture.session.models.models().unwrap().fast
                ));
                let next_caches = fixture.session.query_models().models().unwrap();
                assert!(Arc::ptr_eq(&caches.fast, &next_caches.fast));
                match (&caches.quality, &next_caches.quality) {
                    (Some(old), Some(next)) => assert!(Arc::ptr_eq(old, next)),
                    (Some(old), None) => retired_quality_cache = Some(Arc::clone(old)),
                    (None, Some(next)) => assert!(!Arc::ptr_eq(
                        retired_quality_cache.as_ref().unwrap(),
                        next
                    )),
                    (None, None) => unreachable!(),
                }
                assert_eq!(
                    calls(&fixture.fast, &fixture.quality),
                    (before.0, before.1 + usize::from(needs_quality))
                );
                assert_eq!(previous_reader.topology(), previous.topology());
                assert_eq!(
                    retained_full.topology(),
                    NativeGenerationTopology::FullProgressive
                );
                assert_eq!(retained_full.metadata().total_documents, 4);
                if restored.quality_documents.is_some() {
                    assert_eq!(
                        fixture.session.snapshot.partial_quality_document_ids(&cx).unwrap(),
                        ["quality"]
                    );
                }
                current = fixture.parent.join(format!("{label}.json"));
            }
        }
    });
}

#[test]
fn rollback_activation_requires_a_current_rollback_and_preserves_state_on_preflight_refusal() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let mut fixture = RollbackActivationFixture::new(&cx, ann).await;
            let mut older = AuthoritySession::new(
                &cx,
                admit_root(&fixture.partial.root).unwrap(),
                fixture.partial.clone(),
                Arc::clone(&fixture.partial_snapshot),
                models(&fixture.fast, &fixture.quality),
                (4, DEFAULT_IMAGE_BYTES),
                false,
            )
            .unwrap();
            older.allow_model_activation = true;
            let nonrollback: model_activation::Request = serde_json::from_value(
                rollback_activation_message(&fixture.parent, &fixture.partial, "full"),
            )
            .unwrap();
            let error = older
                .activate_model_with_loader(&cx, &nonrollback, |_, _, _, _| {
                    panic!("an ordinary completion cannot authorize rollback activation")
                })
                .unwrap_err();
            assert!(
                error.to_string().contains("rollback_activation_requires_rollback")
            );
            assert_eq!(older.receipt.authority, fixture.partial.authority);
            assert!(Arc::ptr_eq(&older.snapshot, &fixture.partial_snapshot));

            exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({"query":QUERY,"mode":"full"})],
            )
            .await;
            let retained = Arc::clone(&fixture.session.snapshot);
            let providers = fixture.session.models.models().unwrap().clone();
            let caches = cache_pair(&fixture.session);
            let restored = publish_rollback_selection(
                &cx,
                &fixture.parent,
                &fixture.parent.join("full.json"),
                "partial",
                "restore-partial",
            );
            let authority = fs::read(restored.root.join("AUTHORITY")).unwrap();
            let message =
                rollback_activation_message(&fixture.parent, &fixture.full, "restore-partial");
            for case in [
                "disabled", "stale", "root", "root-id", "producer", "coverage", "paths",
                "budget", "scope", "cancelled",
            ] {
                let mut refused = message.clone();
                let mut forged = restored.clone();
                let expected = match case {
                    "disabled" => {
                        fixture.session.allow_model_activation = false;
                        "model_activation_disabled"
                    }
                    "stale" => {
                        refused["expected_authority"] = serde_json::json!(restored.authority);
                        "stale_expected_authority"
                    }
                    "root" => {
                        forged.root = fixture.parent.join("different-root");
                        "model_activation_root_or_tier_mismatch"
                    }
                    "root-id" => {
                        forged.root_id[0] ^= 1;
                        "model_activation_root_or_tier_mismatch"
                    }
                    "producer" => {
                        let mut identity =
                            forged.quality_identity.as_ref().unwrap().identity.clone();
                        identity.producer.implementation_revision.push_str("-forged");
                        forged.quality_identity = Some(identity.freeze().unwrap());
                        "model_activation_admission_failed"
                    }
                    "coverage" => {
                        forged.quality_documents = Some(2);
                        "model_activation_admission_failed"
                    }
                    "paths" => {
                        refused["model_dir"] =
                            serde_json::json!(fixture.parent.join("unused-model"));
                        "rollback_activation_unused_model_options"
                    }
                    "budget" => {
                        fixture.session.max_image_bytes = 1;
                        "model_activation_admission_failed"
                    }
                    "scope" => {
                        fixture.session.max_source_bytes = 1;
                        "model_activation_scope_failed"
                    }
                    "cancelled" => {
                        cx.set_cancel_requested(true);
                        "cancelled"
                    }
                    _ => unreachable!(),
                };
                if matches!(case, "root" | "root-id" | "producer" | "coverage") {
                    let path = fixture.parent.join(format!("forged-{case}.json"));
                    save_receipt(&forged, &path).unwrap();
                    refused["receipt"] = serde_json::json!(path);
                }
                let request: model_activation::Request =
                    serde_json::from_value(refused).unwrap();
                let error = fixture
                    .session
                    .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                        panic!("rollback preflight must finish before any model loader")
                    })
                    .unwrap_err();
                cx.set_cancel_requested(false);
                fixture.session.allow_model_activation = true;
                fixture.session.max_image_bytes = DEFAULT_IMAGE_BYTES;
                fixture.session.max_source_bytes = MAX_INPUT_BYTES;
                if case == "cancelled" {
                    assert!(matches!(
                        error.downcast_ref::<SearchError>(),
                        Some(SearchError::Cancelled { .. })
                    ));
                } else {
                    assert!(error.to_string().contains(expected), "{case}: {error}");
                }
                assert_preserved(
                    &fixture.session,
                    &fixture.full,
                    &retained,
                    &providers,
                    &caches,
                );
                assert_eq!(fs::read(restored.root.join("AUTHORITY")).unwrap(), authority);
            }
            let before = calls(&fixture.fast, &fixture.quality);
            let replies = exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({"query":QUERY,"mode":"full"})],
            )
            .await;
            assert_eq!(replies[1]["authority"], serde_json::json!(fixture.full.authority));
            assert_eq!(calls(&fixture.fast, &fixture.quality), before);
            restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
        }
    });
}

#[test]
fn rollback_activation_load_failures_and_racing_publication_preserve_the_old_fast_selection() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let mut fixture = RollbackActivationFixture::new(&cx, ann).await;
            exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({"query":QUERY,"mode":"full"})],
            )
            .await;
            let fast_selected = publish_rollback_selection(
                &cx,
                &fixture.parent,
                &fixture.parent.join("full.json"),
                "fast",
                "restore-fast",
            );
            let message =
                rollback_activation_message(&fixture.parent, &fixture.full, "restore-fast");
            let request: model_activation::Request = serde_json::from_value(message).unwrap();
            fixture
                .session
                .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                    panic!("restoring the retained fast producer needs no model load")
                })
                .unwrap();
            let retained = Arc::clone(&fixture.session.snapshot);
            let fast = Arc::clone(&fixture.session.models.models().unwrap().fast);
            let cache = Arc::clone(&fixture.session.query_models().models().unwrap().fast);
            let restored = publish_rollback_selection(
                &cx,
                &fixture.parent,
                &fixture.parent.join("restore-fast.json"),
                "partial",
                "restore-partial",
            );
            let authority = fs::read(restored.root.join("AUTHORITY")).unwrap();
            let mut message =
                rollback_activation_message(&fixture.parent, &fast_selected, "restore-partial");
            message["quality_backend"] = serde_json::json!("native-f32");
            message["quality_model_dir"] =
                serde_json::json!(fixture.parent.join("explicit-restored-quality"));
            for case in ["wrong-quality", "load-error", "cancelled", "scope", "missing-path"] {
                let mut refused = message.clone();
                if case == "scope" {
                    fixture.session.max_source_bytes = 1;
                }
                if case == "missing-path" {
                    refused.as_object_mut().unwrap().remove("quality_backend");
                    refused.as_object_mut().unwrap().remove("quality_model_dir");
                }
                let request: model_activation::Request =
                    serde_json::from_value(refused).unwrap();
                let mut loads = 0;
                let error = fixture
                    .session
                    .activate_model_with_loader(&cx, &request, |cx, _, tier, _| {
                        loads += 1;
                        assert_eq!(tier, EmbedderTierTag::Quality);
                        if case == "load-error" {
                            return Err(bad("private-rollback-model-diagnostic"));
                        }
                        if case == "cancelled" {
                            cx.set_cancel_requested(true);
                        }
                        let model: Arc<dyn Embedder> = if case == "wrong-quality" {
                            fixture.fast.clone()
                        } else {
                            fixture.quality.clone()
                        };
                        Ok(model)
                    })
                    .unwrap_err();
                cx.set_cancel_requested(false);
                fixture.session.max_source_bytes = MAX_INPUT_BYTES;
                if case == "cancelled" {
                    assert!(matches!(
                        error.downcast_ref::<SearchError>(),
                        Some(SearchError::Cancelled { .. })
                    ));
                } else {
                    let expected = match case {
                        "wrong-quality" => "producer_identity_mismatch",
                        "load-error" => "model_activation_load_failed",
                        "scope" => "model_activation_scope_failed",
                        "missing-path" => "model_activation_model_path_required",
                        _ => unreachable!(),
                    };
                    assert!(error.to_string().contains(expected), "{case}: {error}");
                    assert!(!error.to_string().contains("private-rollback-model-diagnostic"));
                }
                assert_eq!(loads, usize::from(!matches!(case, "scope" | "missing-path")));
                assert_eq!(fixture.session.receipt.authority, fast_selected.authority);
                assert!(Arc::ptr_eq(&fixture.session.snapshot, &retained));
                assert!(Arc::ptr_eq(
                    &fixture.session.models.models().unwrap().fast,
                    &fast
                ));
                assert!(Arc::ptr_eq(
                    &fixture.session.query_models().models().unwrap().fast,
                    &cache
                ));
                assert!(fixture.session.models.models().unwrap().quality.is_none());
                assert!(fixture.session.query_models().models().unwrap().quality.is_none());
                assert_eq!(fixture.session.scope.as_ref().unwrap().len(), 2);
                assert_eq!(fs::read(restored.root.join("AUTHORITY")).unwrap(), authority);
            }
            let request: model_activation::Request = serde_json::from_value(message).unwrap();
            let mut winner = None;
            let error = fixture
                .session
                .activate_model_with_loader(&cx, &request, |cx, _, tier, _| {
                    assert_eq!(tier, EmbedderTierTag::Quality);
                    winner = Some(publish_rollback_selection(
                        cx,
                        &fixture.parent,
                        &fixture.parent.join("restore-partial.json"),
                        "full",
                        "winner",
                    ));
                    Ok(fixture.quality.clone())
                })
                .unwrap_err();
            assert!(error.to_string().contains("model_activation_head_changed"));
            let winner = winner.unwrap();
            winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(fixture.session.receipt.authority, fast_selected.authority);
            assert!(Arc::ptr_eq(&fixture.session.snapshot, &retained));
            assert!(Arc::ptr_eq(
                &fixture.session.models.models().unwrap().fast,
                &fast
            ));
            assert!(Arc::ptr_eq(
                &fixture.session.query_models().models().unwrap().fast,
                &cache
            ));
            assert!(fixture.session.models.models().unwrap().quality.is_none());
            assert!(fixture.session.query_models().models().unwrap().quality.is_none());
            let mut retry =
                rollback_activation_message(&fixture.parent, &fast_selected, "winner");
            retry["quality_backend"] = serde_json::json!("native-f32");
            retry["quality_model_dir"] =
                serde_json::json!(fixture.parent.join("explicit-restored-quality"));
            let request: model_activation::Request = serde_json::from_value(retry).unwrap();
            let activated = fixture
                .session
                .activate_model_with_loader(&cx, &request, |_, _, tier, _| {
                    assert_eq!(tier, EmbedderTierTag::Quality);
                    Ok(fixture.quality.clone())
                })
                .unwrap();
            assert_eq!(activated.models_loaded, 1);
            assert!(activated.primary_retained);
            assert_eq!(fixture.session.receipt.authority, winner.authority);
            assert!(Arc::ptr_eq(
                &fixture.session.query_models().models().unwrap().fast,
                &cache
            ));
            let replies = exchange(
                &cx,
                &mut fixture.session,
                &[serde_json::json!({"query":QUERY,"mode":"full"})],
            )
            .await;
            assert_eq!(replies[0]["default_mode"], "full");
            assert_eq!(replies[1]["authority"], serde_json::json!(winner.authority));
            assert_eq!(replies[1]["results"].as_array().unwrap().len(), 2);
        }
    });
}

struct RollbackUnavailableQuality {
    inner: Arc<Provider>,
    unavailable: std::sync::atomic::AtomicBool,
}

impl Embedder for RollbackUnavailableQuality {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        if self.unavailable.load(Ordering::Relaxed) {
            return Err(SearchError::SubsystemError {
                subsystem: "rollback_quality_identity",
                source: "quality producer became unavailable".into(),
            });
        }
        self.inner.identity()
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

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

#[test]
fn rollback_activation_can_discard_an_unavailable_quality_provider_and_its_cache() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut fixture = RollbackActivationFixture::new(&cx, true).await;
        let quality = Arc::new(RollbackUnavailableQuality {
            inner: Arc::clone(&fixture.quality),
            unavailable: std::sync::atomic::AtomicBool::new(false),
        });
        fixture.session = AuthoritySession::new(
            &cx,
            admit_root(&fixture.full.root).unwrap(),
            fixture.full.clone(),
            fixture.full.open(&cx, DEFAULT_IMAGE_BYTES).unwrap(),
            Models {
                fast: fixture.fast.clone(),
                quality: Some(quality.clone()),
            },
            (4, DEFAULT_IMAGE_BYTES),
            false,
        )
        .unwrap()
        .with_query_cache(&cx, query_cache::Config::parse("4").unwrap())
        .unwrap()
        .with_filter(
            &cx,
            Some(Filter::parse(r#"{"ids":["quality","fast-only","new"]}"#).unwrap()),
            MAX_INPUT_BYTES,
        )
        .unwrap();
        fixture.session.allow_model_activation = true;
        exchange(
            &cx,
            &mut fixture.session,
            &[serde_json::json!({"query":QUERY,"mode":"full"})],
        )
        .await;
        let cache = Arc::clone(&fixture.session.query_models().models().unwrap().fast);
        quality.unavailable.store(true, Ordering::Relaxed);
        let restored = publish_rollback_selection(
            &cx,
            &fixture.parent,
            &fixture.parent.join("full.json"),
            "fast",
            "restore-fast",
        );
        let request: model_activation::Request = serde_json::from_value(
            rollback_activation_message(&fixture.parent, &fixture.full, "restore-fast"),
        )
        .unwrap();
        let before = calls(&fixture.fast, &fixture.quality);
        let activated = fixture
            .session
            .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                panic!("discarded quality must not be admitted or replaced")
            })
            .unwrap();
        assert_eq!(activated.models_loaded, 0);
        assert!(activated.primary_retained);
        assert!(!activated.quality_retained);
        assert_eq!(fixture.session.receipt.authority, restored.authority);
        assert!(fixture.session.models.models().unwrap().quality.is_none());
        assert!(fixture.session.query_models().models().unwrap().quality.is_none());
        assert!(Arc::ptr_eq(
            &fixture.session.query_models().models().unwrap().fast,
            &cache
        ));
        assert!(quality.identity().is_err());
        let replies = exchange(
            &cx,
            &mut fixture.session,
            &[
                serde_json::json!({"query":QUERY,"mode":"full"}),
                serde_json::json!({"query":QUERY,"mode":"fast"}),
            ],
        )
        .await;
        assert_eq!(replies[0]["default_mode"], "fast");
        assert_eq!(replies[1]["event"], "failed");
        assert_eq!(replies[2]["event"], "results");
        assert_eq!(replies[2]["authority"], serde_json::json!(restored.authority));
        assert_eq!(calls(&fixture.fast, &fixture.quality), before);
    });
}
