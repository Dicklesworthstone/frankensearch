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
