use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use super::*;
use crate::runtime::{
    CliCommand, CliInput, read_indexed_document_text, set_test_quality_embedder,
    test_quality_embedder_override,
};

struct RestoreQuality(Option<Arc<dyn Embedder>>);
impl Drop for RestoreQuality {
    fn drop(&mut self) {
        set_test_quality_embedder(self.0.take());
    }
}

async fn fixture(
    cx: &Cx,
    parent: &Path,
    windows: usize,
) -> (FsfsRuntime, CompleteGenerationStore, PublishedGeneration) {
    let source = parent.join("source");
    fs::create_dir(&source).unwrap();
    for id in ["alpha.md", "beta.md", "beta-notes.md"] {
        fs::write(source.join(id), format!("sharedtoken {id}")).unwrap();
    }
    let mut config = crate::FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.fast_window_max_per_file = windows;
    config.indexing.quality_model.clear();
    config.search.fast_only = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    let root = parent.join("store");
    let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Index,
        target_path: Some(source),
        index_dir: Some(root.clone()),
        quiet: true,
        ..CliInput::default()
    });
    let store = CompleteGenerationStore::create(cx, &root).unwrap();
    let build = store.begin(cx).unwrap();
    let mut input = runtime.cli_input.clone();
    input.index_dir = Some(build.path().to_path_buf());
    let candidate = runtime.clone().with_cli_input(input);
    candidate
        .run_retained_index_with_reuse(cx, &store, build.path())
        .await
        .unwrap();
    let identity = TestProducer::new(Reply::Good).identity;
    let mut quality = VectorIndex::create_with_revision(
        &build.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE),
        "retained-batch-test",
        &identity.fingerprint(),
        3,
        frankensearch_index::Quantization::F16,
    )
    .unwrap();
    for id in ["alpha.md", "beta.md"] {
        quality.write_record(id, &[1.0, 0.0, 0.0]).unwrap();
    }
    quality.finish().unwrap();
    let storage = Storage::open(PipelineStorageConfig {
        db_path: candidate.resolve_storage_db_path().unwrap(),
        ..PipelineStorageConfig::default()
    })
    .unwrap();
    for id in ["alpha.md", "beta.md", "beta-notes.md"] {
        storage
            .upsert_document(&frankensearch_storage::DocumentRecord::new(
                id, "old", [7; 32], 3, 1, 1,
            ))
            .unwrap();
    }
    drop(storage);
    assert!(protect_vector_generations(build.path(), "batch fixture").is_empty());
    let GenerationPublication::Durable(generation) = build
        .publish(cx, |_, path| {
            FsfsRuntime::validate_search_generation_at_root(path, SearchExecutionMode::Full)
        })
        .unwrap()
    else {
        panic!("fixture publication");
    }; // ubs:ignore — test assertion.
    (runtime, store, generation)
}

fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut result = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                result.insert(
                    entry.path().strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(entry.path()).unwrap(),
                );
            }
        }
    }
    result
}

fn ids(root: &Path, relative: &str) -> BTreeSet<String> {
    VectorIndex::open_read_only(&root.join(relative))
        .unwrap()
        .live_doc_ids()
        .unwrap()
        .into_iter()
        .collect()
}

fn durable(result: RetainedBatchResult) -> PublishedGeneration {
    let Some(GenerationPublication::Durable(generation)) = result.publication else {
        panic!("expected durable batch");
    }; // ubs:ignore — test assertion.
    generation
}

#[test]
fn mixed_batch_publishes_once_and_preserves_untouched_documents_and_old_readers() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let _restore = RestoreQuality(test_quality_embedder_override());
        let quality = Arc::new(TestProducer::new(Reply::Good));
        set_test_quality_embedder(Some(quality.clone()));
        let (runtime, store, first) = fixture(&cx, parent.path(), 1).await;
        let before = files(first.path());
        let old = runtime
            .open_retained_search(&cx, store.root())
            .await
            .unwrap();
        fs::rename(
            parent.path().join("source/alpha.md"),
            parent.path().join("removed-source"),
        )
        .unwrap();
        quality.calls.store(0, Ordering::SeqCst);
        let result = runtime
            .apply_retained_batch(
                &cx,
                store.root(),
                &first,
                &[
                    upsert("alpha.md", "superseded"),
                    delete("beta.md"),
                    upsert("new.md", "noveltoken new"),
                    upsert("alpha.md", "noveltoken replacement"),
                    delete("absent.md"),
                ],
            )
            .await
            .unwrap();
        assert_eq!((result.upserted, result.deleted), (2, 1));
        assert_eq!(
            quality.calls.load(Ordering::SeqCst),
            2,
            "only final upserts infer"
        );
        let next = durable(result);
        assert_eq!(store.active(&cx).unwrap(), Some(next.clone()));
        assert_eq!(
            fs::read_dir(store.root().join("generations"))
                .unwrap()
                .count(),
            2
        );
        assert_eq!(files(first.path()), before);
        assert_eq!(
            ids(next.path(), FSFS_VECTOR_INDEX_FILE),
            BTreeSet::from([
                "alpha.md".to_owned(),
                "beta-notes.md".to_owned(),
                "new.md".to_owned(),
            ])
        );
        assert_eq!(
            ids(next.path(), FSFS_VECTOR_QUALITY_INDEX_FILE),
            BTreeSet::from(["alpha.md".to_owned(), "new.md".to_owned(),])
        );
        let fresh = runtime
            .open_retained_search(&cx, store.root())
            .await
            .unwrap();
        assert_eq!(
            read_indexed_document_text(
                &cx,
                old.resources.lexical_index.as_ref().unwrap(),
                "alpha.md"
            )
            .unwrap(),
            "sharedtoken alpha.md"
        );
        assert_eq!(
            read_indexed_document_text(
                &cx,
                fresh.resources.lexical_index.as_ref().unwrap(),
                "alpha.md"
            )
            .unwrap(),
            "noveltoken replacement"
        );
        for relative in [FSFS_VECTOR_INDEX_FILE, FSFS_VECTOR_QUALITY_INDEX_FILE] {
            let index = VectorIndex::open_read_only(&next.path().join(relative)).unwrap();
            assert_eq!(index.wal_record_count(), 0);
            assert_eq!(index.tombstone_count(), 0);
        }
        let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
            .unwrap()
            .unwrap();
        assert!(!manifests.contains_key("beta.md"));
        assert_eq!(manifests.len(), 3);
        let previous = FsfsRuntime::read_matching_manifest_generation(first.path())
            .unwrap()
            .unwrap();
        assert_eq!(
            manifests["beta-notes.md"].revision,
            previous["beta-notes.md"].revision
        );
        assert!(manifests["alpha.md"].revision > previous["alpha.md"].revision);
        // Inspect a copy: opening SQLite must never mutate a sealed bundle.
        let catalog = parent.path().join("inspect");
        fs::create_dir(&catalog).unwrap();
        for entry in fs::read_dir(next.path()).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("catalog.sqlite")
            {
                fs::copy(entry.path(), catalog.join(entry.file_name())).unwrap();
            }
        }
        let storage = Storage::open(PipelineStorageConfig {
            db_path: catalog.join("catalog.sqlite"),
            ..PipelineStorageConfig::default()
        })
        .unwrap();
        assert!(storage.get_document("beta.md").unwrap().is_none());
        assert_eq!(
            storage
                .get_document("alpha.md")
                .unwrap()
                .unwrap()
                .content_preview,
            "noveltoken replacement"
        );
    });
}

#[test]
fn quality_failure_cannot_publish_deletions_or_a_valid_upsert_prefix() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let _restore = RestoreQuality(test_quality_embedder_override());
        set_test_quality_embedder(Some(Arc::new(TestProducer::new(Reply::Good))));
        let (runtime, store, first) = fixture(&cx, parent.path(), 1).await;
        let before = files(first.path());
        let result = runtime
            .apply_retained_batch(
                &cx,
                store.root(),
                &first,
                &[
                    upsert("alpha.md", "valid prefix"),
                    delete("beta.md"),
                    upsert("z.md", "rejectquality"),
                ],
            )
            .await;
        assert!(matches!(result, Err(SearchError::EmbeddingFailed { .. })));
        assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
        assert_eq!(files(first.path()), before);
        drop(store.begin(&cx).unwrap());
    });
}

#[test]
fn stale_expected_base_is_a_conflict_even_for_absent_deletions() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let _restore = RestoreQuality(test_quality_embedder_override());
        set_test_quality_embedder(Some(Arc::new(TestProducer::new(Reply::Good))));
        let (runtime, store, first) = fixture(&cx, parent.path(), 1).await;
        let second = durable(
            runtime
                .apply_retained_batch(&cx, store.root(), &first, &[delete("beta.md")])
                .await
                .unwrap(),
        );
        let error = runtime
            .apply_retained_batch(&cx, store.root(), &first, &[delete("absent")])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("expected batch base"));
        assert_eq!(store.active(&cx).unwrap(), Some(second));
    });
}

#[test]
fn failed_source_precommit_and_cancellation_preserve_the_entire_predecessor() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let _restore = RestoreQuality(test_quality_embedder_override());
        set_test_quality_embedder(Some(Arc::new(TestProducer::new(Reply::Good))));
        let (runtime, store, first) = fixture(&cx, parent.path(), 1).await;
        let before = files(first.path());
        for cancel in [false, true] {
            let result = runtime
                .apply_retained_batch_with_precommit(
                    &cx,
                    store.root(),
                    &first,
                    &[upsert("alpha.md", "new"), delete("beta.md")],
                    |cx| {
                        if cancel {
                            cx.set_cancel_requested(true);
                            Ok(())
                        } else {
                            Err(invalid("injected source authority refusal"))
                        }
                    },
                )
                .await;
            cx.set_cancel_requested(false);
            if cancel {
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("source authority refusal")
                );
            }
            assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
            assert_eq!(files(first.path()), before);
        }
    });
}

#[test]
fn shrinking_a_windowed_document_and_deleting_another_leave_no_orphan_rows() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let _restore = RestoreQuality(test_quality_embedder_override());
        set_test_quality_embedder(Some(Arc::new(TestProducer::new(Reply::Good))));
        let (runtime, store, first) = fixture(&cx, parent.path(), 4).await;
        let long = (0..2000).fold(String::new(), |mut long, line| {
            writeln!(
                long,
                "windowed alpha line {line} with unique searchable content"
            )
            .unwrap();
            long
        });
        let wide = durable(
            runtime
                .apply_retained_batch(&cx, store.root(), &first, &[upsert("alpha.md", &long)])
                .await
                .unwrap(),
        );
        assert!(
            ids(wide.path(), FSFS_VECTOR_INDEX_FILE)
                .iter()
                .filter(|row| semantic_windows::source_id(row) == "alpha.md")
                .count()
                > 1
        );
        let small = durable(
            runtime
                .apply_retained_batch(
                    &cx,
                    store.root(),
                    &wide,
                    &[upsert("alpha.md", "short replacement"), delete("beta.md")],
                )
                .await
                .unwrap(),
        );
        assert_eq!(
            ids(small.path(), FSFS_VECTOR_INDEX_FILE),
            BTreeSet::from([
                semantic_windows::row_id("alpha.md", 0),
                semantic_windows::row_id("beta-notes.md", 0),
            ])
        );
        assert_eq!(
            ids(small.path(), FSFS_VECTOR_QUALITY_INDEX_FILE),
            BTreeSet::from(["alpha.md".to_owned()])
        );
    });
}
