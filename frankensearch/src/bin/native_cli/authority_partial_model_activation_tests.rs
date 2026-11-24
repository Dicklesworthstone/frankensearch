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
