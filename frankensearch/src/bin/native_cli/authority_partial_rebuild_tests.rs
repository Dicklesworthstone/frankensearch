use super::*;

fn replacement(previous: &Provider, label: &str) -> Arc<Provider> {
    let mut identity = previous.identity.clone();
    identity.producer.implementation_revision = label.to_owned();
    identity.validate().unwrap();
    Arc::new(Provider {
        identity,
        calls: AtomicUsize::new(0),
    })
}

fn options(parent: &Path, predecessor: &Path, label: &str) -> Options {
    let mut options = parse(&retained_rebuild_arguments()).unwrap().unwrap();
    options.from_receipt = Some(predecessor.to_path_buf());
    options.work = Some(parent.join(format!("{label}-work")));
    options.receipt = parent.join(format!("{label}.json"));
    options
}

#[test]
fn mixed_retrieval_index_and_rebuild_reopen_and_search_partial_coverage() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let mut index_args = index_arguments();
        let tier = index_args
            .iter()
            .position(|value| *value == "fast")
            .unwrap();
        index_args[tier] = "partial";
        index_args.extend([
            "--quality-ids",
            r#"["quality"]"#,
            "--retrieval",
            "hnsw",
            "--quality-retrieval",
            "exact",
        ]);
        let mut initial = parse(&index_args).unwrap().unwrap();
        let fixture = partial_index_options(&parent, false);
        initial.root = fixture.root;
        initial.work = fixture.work;
        initial.receipt = fixture.receipt;
        initial.input = fixture.input;
        let mut output = Vec::new();
        index_with_loader(&cx, &initial, &mut output, |_| Ok(models(&fast, &quality)))
            .await
            .unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["retrieval"], "mixed");
        assert_eq!(
            report["retrieval_by_tier"],
            serde_json::json!({"fast":"hnsw","quality":"exact"})
        );
        let original = Receipt::read(&initial.receipt)
            .unwrap()
            .open(&cx, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let graph = original.retrieval_configuration();
        assert!(matches!(graph, NativeBuildRetrieval::Hnsw { .. }));
        assert_eq!(
            original.quality_retrieval_configuration(),
            Some(NativeBuildRetrieval::Exact)
        );

        let mut rebuild_args = retained_rebuild_arguments();
        rebuild_args.extend(["--retrieval", "exact", "--quality-retrieval", "hnsw"]);
        let mut rebuild = parse(&rebuild_args).unwrap().unwrap();
        rebuild.work = Some(parent.join("reverse-work"));
        rebuild.from_receipt = Some(initial.receipt);
        rebuild.receipt = parent.join("reverse.json");
        let before = calls(&fast, &quality);
        output.clear();
        rebuild_with_loader(&cx, &rebuild, &mut output, |_| Ok(models(&fast, &quality)))
            .await
            .unwrap();
        assert_eq!(
            calls(&fast, &quality),
            before,
            "changing graphs reuses partial embeddings"
        );
        let reopened = Receipt::read(&rebuild.receipt)
            .unwrap()
            .open(&cx, DEFAULT_IMAGE_BYTES)
            .unwrap();
        assert_eq!(
            reopened.retrieval_configuration(),
            NativeBuildRetrieval::Exact
        );
        assert_eq!(reopened.quality_retrieval_configuration(), Some(graph));
        assert_eq!(
            reopened.partial_quality_document_ids(&cx).unwrap(),
            ["quality"]
        );
        output.clear();
        let query = partial_search_options(&rebuild.receipt, Some(QueryMode::Partial));
        search_with_loader(&cx, &query, &mut output, |tier| {
            Ok(if tier == EmbedderTierTag::Fast {
                fast.clone()
            } else {
                quality.clone()
            })
        })
        .await
        .unwrap();
        let phases = frames(&output);
        assert_eq!(phases[1]["phase"], "refined");
        assert_eq!(
            phases[1]["retrieval_by_tier"],
            serde_json::json!({"fast":"exact","quality":"hnsw"})
        );
        assert_eq!(phases[1]["results"].as_array().unwrap().len(), 3);
    });
}

#[test]
fn partial_retained_rebuild_migrates_each_producer_without_expanding_coverage() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            for (change_fast, change_quality) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let parent = private_parent();
                let fast = provider(false);
                let quality = provider(true);
                let (initial, selected) =
                    publish_partial_cli(&cx, &parent, &fast, &quality, ann).await;
                let retained = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let mut source = retained.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
                source.sort_by(|left, right| left.id.cmp(&right.id));
                let original_receipt = fs::read(&initial.receipt).unwrap();
                fs::rename(
                    initial.input.as_ref().unwrap(),
                    parent.join("archived-input"),
                )
                .unwrap();
                fs::rename(initial.work.as_ref().unwrap(), parent.join("archived-work")).unwrap();
                let next_fast = if change_fast {
                    replacement(&fast, "partial-new-fast")
                } else {
                    Arc::clone(&fast)
                };
                let next_quality = if change_quality {
                    replacement(&quality, "partial-new-quality")
                } else {
                    Arc::clone(&quality)
                };
                let mut rebuild = options(&parent, &initial.receipt, "rebuilt");
                rebuild.migrate_model = change_fast || change_quality;
                // Changing graph policy alone requires no document inference.
                rebuild.retrieval = Some(if ann {
                    NativeBuildRetrieval::Exact
                } else {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 42,
                    }
                });
                let before = calls(&next_fast, &next_quality);
                let mut output = Vec::new();
                rebuild_with_loader(&cx, &rebuild, &mut output, |mode| {
                    assert_eq!(
                        mode,
                        QueryMode::Full,
                        "both partial producers must be loaded"
                    );
                    Ok(models(&next_fast, &next_quality))
                })
                .await
                .unwrap();
                assert_eq!(
                    calls(&next_fast, &next_quality),
                    (
                        before.0 + if change_fast { 3 } else { 0 },
                        before.1 + usize::from(change_quality),
                    )
                );
                assert_eq!(calls(&fast, &quality), (3, 1));
                let current = Receipt::read(&rebuild.receipt).unwrap();
                let reopened = current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(current.schema, PARTIAL_QUALITY_RECEIPT_SCHEMA);
                assert_eq!(current.documents, 3);
                assert_eq!(current.quality_documents, Some(1));
                assert_eq!(current.identity == selected.identity, !change_fast);
                assert_eq!(
                    current.quality_identity == selected.quality_identity,
                    !change_quality
                );
                assert_eq!(
                    reopened.partial_quality_document_ids(&cx).unwrap(),
                    ["quality"]
                );
                let mut actual_source = reopened.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
                actual_source.sort_by(|left, right| left.id.cmp(&right.id));
                assert_eq!(
                    serde_json::to_value(actual_source).unwrap(),
                    serde_json::to_value(&source).unwrap(),
                );
                assert_eq!(
                    reopened.metadata().commit_range,
                    retained.metadata().commit_range
                );
                assert_eq!(
                    reopened.retained_snapshot().manifest().components().lexical,
                    retained.retained_snapshot().manifest().components().lexical
                );
                assert_eq!(
                    reopened.retrieval_configuration(),
                    rebuild.build_retrieval(None)
                );
                assert_eq!(
                    reopened.quality_retrieval_configuration(),
                    Some(rebuild.build_retrieval(None))
                );
                assert_eq!(fs::read(&initial.receipt).unwrap(), original_receipt);
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "rebuilt");
                assert_eq!(report["model_migrated"], change_fast || change_quality);
                assert_eq!(report["selection"]["quality_documents"], 1);
                let query = partial_search_options(&rebuild.receipt, Some(QueryMode::Partial));
                output.clear();
                search_with_loader(&cx, &query, &mut output, |tier| {
                    Ok(match tier {
                        EmbedderTierTag::Fast => next_fast.clone(),
                        EmbedderTierTag::Quality => next_quality.clone(),
                    })
                })
                .await
                .unwrap();
                let response = frames(&output);
                assert_eq!(response.len(), 2);
                assert_eq!(response[1]["phase"], "refined");
                assert_eq!(response[1]["results"].as_array().unwrap().len(), 3);
                assert_eq!(response[1]["quality_coverage"]["quality_documents"], 1);
                assert!(
                    !retained
                        .search_text(&cx, fast.as_ref(), "retry", 3)
                        .await
                        .unwrap()
                        .is_empty()
                );
            }
        }
    });
}

#[test]
fn partial_rebuild_preflight_and_inference_failure_leave_the_incumbent_selected() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let (initial, selected) = publish_partial_cli(&cx, &parent, &fast, &quality, false).await;
        let authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
        for (case, expected) in [
            ("fresh", "partial_quality_rebuild_requires_retained_source"),
            (
                "query",
                "partial_quality_rebuild_query_requires_partial_mode",
            ),
            ("budget", "stored-payload-limit"),
        ] {
            let mut rebuild = options(&parent, &initial.receipt, case);
            match case {
                "fresh" => {
                    rebuild.retained_source = false;
                    rebuild.input = Some(parent.join("unread-input"));
                }
                "query" => {
                    rebuild.query = Some("retry".to_owned());
                    rebuild.migrate_model = true;
                    rebuild.rollback_receipt = Some(parent.join("unwritten-rollback.json"));
                }
                "budget" => rebuild.max_source_bytes = 1,
                _ => unreachable!(),
            }
            let mut output = Vec::new();
            let error = rebuild_with_loader(&cx, &rebuild, &mut output, |_| {
                panic!("partial rebuild preflight must precede model loading")
            })
            .await
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{case}: {error}");
            assert!(output.is_empty());
            assert!(!rebuild.work.as_ref().unwrap().exists());
            assert!(!rebuild.receipt.exists());
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority
            );
        }
        let changed = replacement(&quality, "partial-rebuild-changed-quality");
        let denied = options(&parent, &initial.receipt, "no-migration-grant");
        assert!(
            rebuild_with_loader(&cx, &denied, &mut Vec::new(), |_| {
                Ok(models(&fast, &changed))
            })
            .await
            .is_err()
        );
        assert_eq!(changed.calls.load(Ordering::Relaxed), 0);
        assert!(!denied.work.as_ref().unwrap().exists());
        assert!(!denied.receipt.exists());

        let mut failure = options(&parent, &initial.receipt, "inference-failure");
        failure.migrate_model = true;
        let failed: Arc<dyn Embedder> = Arc::new(FailingQuality {
            inner: Arc::clone(&changed),
        });
        assert!(
            rebuild_with_loader(&cx, &failure, &mut Vec::new(), |_| Ok(Models {
                fast: fast.clone(),
                quality: Some(Arc::clone(&failed)),
            }))
            .await
            .is_err()
        );
        assert_eq!(
            fast.calls.load(Ordering::Relaxed),
            3,
            "unchanged fast tier is reused even when quality fails"
        );
        assert_eq!(quality.calls.load(Ordering::Relaxed), 1);
        assert_eq!(changed.calls.load(Ordering::Relaxed), 1);
        assert!(!failure.receipt.exists());
        assert_eq!(
            fs::read(selected.root.join("AUTHORITY")).unwrap(),
            authority
        );
        let reopened = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
        assert_eq!(
            reopened.partial_quality_document_ids(&cx).unwrap(),
            ["quality"]
        );
        assert!(
            !reopened
                .search_text(&cx, fast.as_ref(), "retry", 3)
                .await
                .unwrap()
                .is_empty()
        );
    });
}

#[test]
fn partial_rebuild_publication_checks_actual_membership_and_fences_stale_candidates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let (_, selected) = publish_partial_cli(&cx, &parent, &fast, &quality, false).await;
        let root = admit_root(&selected.root).unwrap();
        let retained = selected
            .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
        let changed = replacement(&quality, "partial-publish-new-quality");
        let source = retained.metadata().commit_range.clone();
        for (label, covered, alter_source, advance_cut) in [
            ("different-subset", "fast-only", false, false),
            ("different-body", "quality", true, false),
            ("different-checkpoint", "quality", false, true),
        ] {
            let mut documents = retained.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
            if alter_source {
                documents[0]
                    .content
                    .push_str(" changed without source authorization");
            }
            let mut candidate_source = source.clone();
            if advance_cut {
                candidate_source.high += 1;
            }
            let built = NativeIndexBuilder::new(
                parent.join(label),
                new_generation(2).unwrap(),
                fast.clone(),
            )
            .unwrap()
            .with_quality_embedder(changed.clone())
            .unwrap()
            .add_documents(documents)
            .build_partial_quality_generation(
                &cx,
                candidate_source,
                [covered.to_owned()],
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            assert_eq!(
                built.metadata().total_documents,
                retained.metadata().total_documents
            );
            assert!(
                prepare_partial_quality_rebuild(&cx, &built, &retained, DEFAULT_IMAGE_BYTES)
                    .is_err()
            );
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority
            );
        }

        let candidate = retained
            .begin_partial_quality_model_migration(
                &cx,
                parent.join("stale-work"),
                new_generation(2).unwrap(),
                fast.clone(),
                changed.clone(),
                MAX_INPUT_BYTES,
            )
            .unwrap()
            .build(&cx, source.clone(), DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
        let prepared =
            prepare_partial_quality_rebuild(&cx, &candidate, &retained, DEFAULT_IMAGE_BYTES)
                .unwrap();
        let winner = retained
            .begin_partial_quality_update(
                &cx,
                parent.join("winning-work"),
                new_generation(2).unwrap(),
                fast.clone(),
                quality.clone(),
                MAX_INPUT_BYTES,
            )
            .unwrap()
            .build(&cx, source, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
        let outcome = stage_and_publish(
            &cx,
            &root,
            &selected.root,
            &parent.join("winner.json"),
            &winner,
            Some(&retained),
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        let winner_authority = fs::read(selected.root.join("AUTHORITY")).unwrap();
        assert!(
            stage_prepared_and_publish(
                &cx,
                &root,
                &selected.root,
                &parent.join("stale.json"),
                &candidate,
                prepared,
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .is_err()
        );
        assert_eq!(
            fs::read(selected.root.join("AUTHORITY")).unwrap(),
            winner_authority
        );
        let reopened = outcome.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
        assert_eq!(
            reopened.partial_quality_document_ids(&cx).unwrap(),
            ["quality"]
        );
        assert_eq!(reopened.metadata().embedders, retained.metadata().embedders);
        assert!(
            !retained
                .search_text(&cx, fast.as_ref(), "retry", 3)
                .await
                .unwrap()
                .is_empty()
        );
    });
}
