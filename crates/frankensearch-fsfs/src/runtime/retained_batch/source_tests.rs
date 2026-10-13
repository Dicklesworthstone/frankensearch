use asupersync::test_utils::run_test_with_cx;

use super::*;

fn source(class: IngestionClass) -> SourceAttributes {
    SourceAttributes {
        modified_ms: 42,
        ingestion_class: class,
        title: Some("alpha.md".to_owned()),
        metadata: HashMap::from([
            ("source_path".to_owned(), "/source/alpha.md".to_owned()),
            ("source_modified_ms".to_owned(), "42".to_owned()),
            (
                "ingestion_class".to_owned(),
                ingestion_class_label(class).to_owned(),
            ),
        ]),
    }
}

fn operations() -> Vec<RetainedMutation> {
    vec![
        RetainedMutation::Upsert {
            id: "alpha.md".to_owned(),
            text: "alpha replacement".to_owned(),
        },
        RetainedMutation::Delete {
            id: "old.md".to_owned(),
        },
    ]
}

#[test]
fn source_attributes_preserve_class_timestamp_and_the_original_document_api() {
    run_test_with_cx(|cx| async move {
        let mut documents = prepare(&cx, &operations()).unwrap();
        let ordinary = documents["alpha.md"].as_ref().unwrap();
        assert!(ordinary.source.is_none());
        assert!(ordinary.semantic());
        assert_eq!(ordinary.revision(99), 99);
        assert_eq!(
            ordinary.reason(IngestionClass::FullSemanticLexical),
            "retained_batch"
        );
        attach_sources(
            &cx,
            &mut documents,
            &BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::LexicalOnly))]),
        )
        .unwrap();
        let body = documents["alpha.md"].as_ref().unwrap();
        assert!(!body.semantic());
        assert_eq!(body.revision(99), 42);
        assert_eq!(
            body.reason(IngestionClass::FullSemanticLexical),
            "index.plan.lexical_only"
        );
        assert_eq!(
            body.source.as_ref().unwrap().title.as_deref(),
            Some("alpha.md")
        );
        assert_eq!(
            body.source.as_ref().unwrap().metadata["source_modified_ms"],
            "42"
        );
        assert!(documents["old.md"].is_none());
    });
}

#[test]
fn source_attributes_must_match_final_upsert_membership_exactly() {
    run_test_with_cx(|cx| async move {
        for attributes in [
            BTreeMap::new(),
            BTreeMap::from([("old.md".to_owned(), source(IngestionClass::LexicalOnly))]),
            BTreeMap::from([
                ("alpha.md".to_owned(), source(IngestionClass::LexicalOnly)),
                ("other.md".to_owned(), source(IngestionClass::LexicalOnly)),
            ]),
        ] {
            let mut documents = prepare(&cx, &operations()).unwrap();
            assert!(attach_sources(&cx, &mut documents, &attributes).is_err());
        }
    });
}

#[test]
fn unsupported_source_policy_timestamp_and_metadata_are_rejected_before_io() {
    run_test_with_cx(|cx| async move {
        let mut invalid_sources = vec![
            source(IngestionClass::MetadataOnly),
            source(IngestionClass::Skip),
        ];
        let mut overflow = source(IngestionClass::LexicalOnly);
        overflow.modified_ms = u64::MAX;
        invalid_sources.push(overflow);
        let mut too_many = source(IngestionClass::LexicalOnly);
        too_many.metadata = (0..65).map(|n| (n.to_string(), "v".to_owned())).collect();
        invalid_sources.push(too_many);
        for invalid_source in invalid_sources {
            let mut documents = prepare(&cx, &operations()).unwrap();
            assert!(
                attach_sources(
                    &cx,
                    &mut documents,
                    &BTreeMap::from([("alpha.md".to_owned(), invalid_source),])
                )
                .is_err()
            );
        }
    });
}

#[test]
fn source_attachment_observes_cancellation() {
    run_test_with_cx(|cx| async move {
        let mut documents = prepare(&cx, &operations()).unwrap();
        cx.set_cancel_requested(true);
        let result = attach_sources(
            &cx,
            &mut documents,
            &BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::LexicalOnly))]),
        );
        cx.set_cancel_requested(false);
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert!(documents["alpha.md"].as_ref().unwrap().source.is_none());
    });
}

#[test]
fn lexical_source_batch_persists_metadata_and_preserves_the_retained_reader() {
    use crate::adapters::live_search::LiveSearchConfig;
    use crate::adapters::live_search_session::LiveSearchRefreshConfig;
    use crate::adapters::quill_live_search::QuillLiveSearchSession;
    use crate::{CliCommand, CliInput, FsfsConfig};
    use frankensearch_quill::QuillConfig;

    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source");
        let root = directory.path().join("store");
        std::fs::create_dir(&source_path).unwrap();
        std::fs::write(source_path.join("alpha.md"), "alpha old body").unwrap();
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source_path.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        runtime.lexical_only_indexing = true;
        let GenerationPublication::Durable(first) = runtime
            .rebuild_retained_generation(&cx, &root)
            .await
            .unwrap()
        else {
            panic!("fixture must be durable");
        }; // ubs:ignore — test assertion.
        let pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
        let store = CompleteGenerationStore::open(&cx, &root).unwrap();
        let batch = prepare_source_batch(
            &cx,
            &operations(),
            &BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::LexicalOnly))]),
        )
        .unwrap()
        .unwrap();
        let outcome = runtime
            .apply_retained_source_batch_with_precommit(&cx, &root, &first, batch, |_| Ok(()))
            .await
            .unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (1, 0));
        let Some(GenerationPublication::Durable(next)) = outcome.publication else {
            panic!("source batch must be durable");
        }; // ubs:ignore — test assertion.
        assert_ne!(next, first);
        assert_eq!(pinned.generation(), &first);
        assert_eq!(
            std::fs::read_to_string(source_path.join("alpha.md")).unwrap(),
            "alpha old body"
        );
        let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
            .unwrap()
            .unwrap();
        assert_eq!(manifests["alpha.md"].revision, 42);
        assert_eq!(manifests["alpha.md"].ingestion_class, "lexical_only");
        assert_eq!(manifests["alpha.md"].reason_code, "index.plan.lexical_only");
        let mut session = QuillLiveSearchSession::new(
            store.clone(),
            "replacement",
            LiveSearchConfig::default(),
            LiveSearchRefreshConfig::default(),
            QuillConfig::default(),
        )
        .unwrap();
        session
            .poll(&cx, std::time::Instant::now())
            .await
            .unwrap()
            .unwrap();
        let hits = session.tracker().results();
        assert_eq!(hits.len(), 1);
        let metadata = hits[0].hit.item.as_ref().unwrap();
        assert_eq!(metadata["source_path"], "/source/alpha.md");
        assert_eq!(metadata["source_modified_ms"], "42");
        assert_eq!(metadata["ingestion_class"], "lexical_only");
        assert_eq!(store.active(&cx).unwrap(), Some(next));
        drop(store.begin(&cx).unwrap());
    });
}

#[test]
fn metadata_sources_are_path_only_and_never_certify_file_content() {
    run_test_with_cx(|cx| async move {
        let operations = [
            RetainedMutation::Upsert {
                id: "alpha.md".to_owned(),
                text: "alpha.md".to_owned(),
            },
            RetainedMutation::Upsert {
                id: "text.md".to_owned(),
                text: "actual source text".to_owned(),
            },
        ];
        let attributes = BTreeMap::from([
            ("alpha.md".to_owned(), source(IngestionClass::MetadataOnly)),
            ("text.md".to_owned(), source(IngestionClass::LexicalOnly)),
        ]);
        // The opaque document API still treats the same text as an ordinary
        // upsert; source-only policy never leaks into that public interface.
        assert!(
            prepare(&cx, &operations).unwrap()["alpha.md"]
                .as_ref()
                .unwrap()
                .semantic()
        );
        let batch = prepare_source_batch(&cx, &operations, &attributes)
            .unwrap()
            .unwrap();
        let body = batch.documents["alpha.md"].as_ref().unwrap();
        assert!(!body.semantic());
        assert_eq!(body.embedding, "alpha.md");
        assert_eq!(body.lexical, "alpha.md");
        assert_eq!(body.revision(0), 42);
        assert_eq!(
            body.reason(IngestionClass::LexicalOnly),
            "index.plan.metadata_only"
        );
        assert_eq!(
            body.source.as_ref().unwrap().metadata["ingestion_class"],
            "metadata_only"
        );
        assert_eq!(
            batch
                .source_inputs
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["text.md"],
            "the path-only operation cannot stand in for a raw file-byte witness"
        );
    });
}

#[test]
fn metadata_sources_reject_content_even_in_a_superseded_input() {
    run_test_with_cx(|cx| async move {
        let attributes =
            BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::MetadataOnly))]);
        for text in ["secret file contents", " alpha.md "] {
            let operations = [
                RetainedMutation::Upsert {
                    id: "alpha.md".to_owned(),
                    text: text.to_owned(),
                },
                RetainedMutation::Upsert {
                    id: "alpha.md".to_owned(),
                    text: "alpha.md".to_owned(),
                },
            ];
            assert!(matches!(
                prepare_source_batch(&cx, &operations, &attributes),
                Err(SearchError::InvalidConfig { .. })
            ));
        }
    });
}

#[test]
fn metadata_sources_keep_timestamp_metadata_operation_and_cancel_bounds() {
    run_test_with_cx(|cx| async move {
        let operation = RetainedMutation::Upsert {
            id: "alpha.md".to_owned(),
            text: "alpha.md".to_owned(),
        };
        let mut overflow = source(IngestionClass::MetadataOnly);
        overflow.modified_ms = u64::MAX;
        let mut excess = source(IngestionClass::MetadataOnly);
        excess.metadata = (0..65).map(|n| (n.to_string(), "v".to_owned())).collect();
        for attributes in [overflow, excess] {
            assert!(
                prepare_source_batch(
                    &cx,
                    std::slice::from_ref(&operation),
                    &BTreeMap::from([("alpha.md".to_owned(), attributes)]),
                )
                .is_err()
            );
        }
        let attributes =
            BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::MetadataOnly))]);
        assert!(
            prepare_source_batch(
                &cx,
                &vec![operation.clone(); MAX_OPERATIONS + 1],
                &attributes
            )
            .unwrap()
            .is_none()
        );
        cx.set_cancel_requested(true);
        let cancelled = prepare_source_batch(&cx, &[operation], &attributes);
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn metadata_demotion_publishes_path_only_and_preserves_the_old_reader() {
    use crate::config::DegradationOverrideMode;
    use crate::{CliCommand, CliInput, FsfsConfig};

    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source");
        let root = directory.path().join("store");
        std::fs::create_dir(&source_path).unwrap();
        std::fs::write(source_path.join("alpha.md"), "privatebody original text").unwrap();
        std::fs::write(source_path.join("stable.md"), "untouchedword original text").unwrap();
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source_path.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        runtime.lexical_only_indexing = true;
        let GenerationPublication::Durable(first) = runtime
            .rebuild_retained_generation(&cx, &root)
            .await
            .unwrap()
        else {
            panic!("fixture must be durable");
        }; // ubs:ignore — test assertion.
        let mut query_runtime = runtime.clone();
        query_runtime.config.pressure.degradation_override =
            DegradationOverrideMode::ForceLexicalOnly;
        let mut pinned = query_runtime
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        let before = FsfsRuntime::read_matching_manifest_generation(first.path())
            .unwrap()
            .unwrap();
        let operations = [RetainedMutation::Upsert {
            id: "alpha.md".to_owned(),
            text: "alpha.md".to_owned(),
        }];
        let attributes =
            BTreeMap::from([("alpha.md".to_owned(), source(IngestionClass::MetadataOnly))]);
        let batch = prepare_source_batch(&cx, &operations, &attributes)
            .unwrap()
            .unwrap();
        assert!(batch.source_inputs.is_empty());
        let outcome = runtime
            .apply_retained_source_batch_with_precommit(&cx, &root, &first, batch, |_| Ok(()))
            .await
            .unwrap();
        assert_eq!((outcome.upserted, outcome.deleted), (1, 0));
        let Some(GenerationPublication::Durable(next)) = outcome.publication else {
            panic!("metadata update must be durable");
        }; // ubs:ignore — test assertion.
        let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
            .unwrap()
            .unwrap();
        let metadata = &manifests["alpha.md"];
        assert_eq!(metadata.ingestion_class, "metadata_only");
        assert_eq!(metadata.reason_code, "index.plan.metadata_only");
        assert_eq!(metadata.revision, 42);
        assert!(metadata.fast_windows.is_none());
        assert_eq!(
            serde_json::to_value(&manifests["stable.md"]).unwrap(),
            serde_json::to_value(&before["stable.md"]).unwrap()
        );
        let mut reader = query_runtime
            .open_retained_search(&cx, &root)
            .await
            .unwrap();
        let hits = reader.search(&cx, "alpha", 10).await.unwrap();
        assert!(
            hits.last()
                .unwrap()
                .hits
                .iter()
                .any(|hit| hit.path.ends_with("alpha.md"))
        );
        assert!(
            reader
                .search(&cx, "privatebody", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .is_empty()
        );
        assert_eq!(
            pinned
                .search(&cx, "privatebody", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .len(),
            1
        );
        assert_eq!(
            reader
                .search(&cx, "untouchedword", 10)
                .await
                .unwrap()
                .last()
                .unwrap()
                .hits
                .len(),
            1
        );
        assert_eq!(
            std::fs::read_to_string(source_path.join("alpha.md")).unwrap(),
            "privatebody original text"
        );
        assert_eq!(pinned.generation(), &first);

        // A failed final source check must not install even a valid path-only
        // candidate. The prior metadata-only publication remains selected.
        let batch = prepare_source_batch(&cx, &operations, &attributes)
            .unwrap()
            .unwrap()
            .with_source_check(|_| Err(invalid("source changed before publication")));
        assert!(
            runtime
                .apply_retained_source_batch_with_precommit(&cx, &root, &next, batch, |_| Ok(()),)
                .await
                .is_err()
        );
        assert_eq!(
            CompleteGenerationStore::open(&cx, &root)
                .unwrap()
                .active(&cx)
                .unwrap(),
            Some(next)
        );
    });
}
