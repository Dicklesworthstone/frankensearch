use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
};
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::native_hnsw::HnswParams;

use crate::native_ann::NativeShardedSearchPhase;
use crate::native_ann::builder::NativeIndexBuilder;
use crate::{Embedder, ModelCategory, SearchFuture};

// Identified deterministic providers exercise real source/FSVI/graph/Quill
// ownership, not model relevance, numerical equivalence or speed.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    documents: AtomicUsize,
    queries: AtomicUsize,
    dropped: AtomicUsize,
    hold_queries: AtomicBool,
    hold_batches: AtomicBool,
    fail_late: AtomicBool,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        Arc::new(Self {
            identity,
            documents: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
            hold_queries: AtomicBool::new(false),
            hold_batches: AtomicBool::new(false),
            fail_late: AtomicBool::new(false),
        })
    }

    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text.contains("vertical"))] = 1.0;
        values
    }
}

struct CountDrop<'a>(&'a AtomicUsize);
impl Drop for CountDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            let _guard = CountDrop(&self.dropped);
            if self.hold_queries.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            Ok(self.values(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.documents.fetch_add(texts.len(), Ordering::SeqCst);
            let _guard = CountDrop(&self.dropped);
            if self.hold_batches.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            if self.fail_late.load(Ordering::SeqCst)
                && texts.iter().any(|text| text.contains("failuretoken"))
            {
                return Err(invalid("sharded_live.fixture", "failed", "late quality failure"));
            }
            Ok(texts.iter().map(|text| IdentityBoundEmbedding {
                identity: self.identity.clone(),
                values: self.values(text),
            }).collect())
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [0x62; 16]).unwrap()
}

fn sources() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("a", "horizontal").with_metadata("version", "old"),
        IndexableDocument::new("b", "legacytoken vertical"),
        IndexableDocument::new("c", "horizontal"),
        IndexableDocument::new("d", "horizontal"),
    ]
}

async fn build(
    cx: &Cx,
    path: &Path,
    sequence: u64,
    fast: &Arc<Provider>,
    quality: Option<&Arc<Provider>>,
    ann: bool,
) -> NativeBuiltShardedHybridIndex {
    let retrieval = if ann {
        NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 7 }
    } else {
        NativeBuildRetrieval::Exact
    };
    let mut builder = NativeIndexBuilder::new(path, generation(sequence), fast.clone())
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F16, retrieval)
        .add_documents(sources());
    if let Some(quality) = quality {
        builder = builder.with_quality_embedder(quality.clone()).unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval).unwrap();
    }
    builder.build_sharded_hybrid(cx, 2).await.unwrap()
}

async fn fixture(
    cx: &Cx,
    root: &Path,
    ann: bool,
) -> (NativeLiveShardedHybridIndex, Arc<Provider>, Arc<Provider>) {
    let fast = Provider::new("sharded-live-fast", 2);
    let quality = Provider::new("sharded-live-quality", 3);
    let index = build(cx, &root.join("initial"), 1, &fast, Some(&quality), ann).await;
    (NativeLiveShardedHybridIndex::new(cx, index).unwrap(), fast, quality)
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = BTreeMap::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.insert(path.strip_prefix(root).unwrap().to_path_buf(), fs::read(path).unwrap());
            }
        }
    }
    files
}

fn assert_rows(snapshot: &NativeShardedSnapshot, hits: &[NativeShardedResult]) {
    let mut resolved = 0;
    for hit in hits {
        assert!(hit.result.index.is_none(), "a bare row is ambiguous across partitions");
        assert!(snapshot.index().vectors().document(&hit.result.doc_id).is_some());
        for (row, quality) in [(hit.fast_row, false), (hit.quality_row, true)] {
            if let Some(row) = row {
                let partition = &snapshot.index().vectors().partitions()[row.shard];
                let tier = if quality { partition.quality().unwrap() } else { partition.fast() };
                assert_eq!(
                    tier.index.owner.doc_id_at(usize::try_from(row.physical_row).unwrap()).unwrap(),
                    hit.result.doc_id
                );
                resolved += 1;
            }
        }
    }
    assert!(resolved > 0, "fixture must exercise physical provenance");
}

fn assert_refusal(error: SearchError, field: &str) {
    assert!(matches!(error, SearchError::InvalidConfig { field: actual, .. } if actual == field));
}

#[test]
fn repartitioned_install_keeps_old_phases_and_pages_on_their_original_owners() {
    run_test_with_cx(|cx| async move {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<NativeLiveShardedHybridIndex>();
        send_sync::<NativeShardedSnapshot>();
        for ann in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, root.path(), ann).await;
            let old = live.snapshot(&cx).await.unwrap();
            let old_image = image(old.index().vectors().directory());
            let old_page = live.search(&cx, "legacytoken", 4).await.unwrap();
            let mut phases = old.index().progressive(&cx, "legacytoken", 4).unwrap();
            let Some(NativeShardedSearchPhase::Initial { results, .. }) = phases.next_phase().await.unwrap() else {
                panic!("Initial must precede refinement");
            };
            assert!(results.iter().any(|hit| hit.result.doc_id == "b"));
            assert_rows(&old, &results);
            let candidate = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
                .delete_document("b")
                .upsert_documents([
                    IndexableDocument::new("a", "horizontal").with_metadata("version", "new"),
                    IndexableDocument::new("aa", "arrival vertical"),
                    IndexableDocument::new("c", "replacement vertical"),
                ])
                .with_batch_size(1).unwrap()
                .with_max_batch_input_bytes(1024).unwrap()
                .build(&cx, 1).await.unwrap();
            assert_eq!(fast.documents.load(Ordering::SeqCst), 6);
            assert_eq!(quality.documents.load(Ordering::SeqCst), 6);
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
            assert_eq!(old.index().vectors().partitions().len(), 2);
            assert_eq!(candidate.index().vectors().partitions().len(), 4);
            let installed = live.install(&cx, &candidate).await.unwrap();
            assert_eq!(installed.generation(), generation(2));
            assert!(installed.index().vectors().document("b").is_none());
            assert_eq!(installed.index().vectors().document("a").unwrap().metadata["version"], "new");
            assert_eq!(old.index().vectors().document("a").unwrap().metadata["version"], "old");
            assert!(installed.index().lexical().search(&cx, "legacytoken", 10).await.unwrap().is_empty());
            assert_eq!(old.index().lexical().search(&cx, "legacytoken", 10).await.unwrap().len(), 1);
            let before_fast = fast.queries.load(Ordering::SeqCst);
            let before_quality = quality.queries.load(Ordering::SeqCst);
            let fast_page = live.search(&cx, "vertical", 4).await.unwrap();
            assert_eq!(fast.queries.load(Ordering::SeqCst), before_fast + 1);
            assert_eq!(quality.queries.load(Ordering::SeqCst), before_quality);
            let quality_page = live.search_quality(&cx, "vertical", 4).await.unwrap();
            assert_eq!(fast.queries.load(Ordering::SeqCst), before_fast + 1);
            assert_eq!(quality.queries.load(Ordering::SeqCst), before_quality + 1);
            let refined_page = live.search_refined(&cx, "vertical", 4).await.unwrap();
            for page in [fast_page, quality_page, refined_page] {
                assert!(Arc::ptr_eq(&page.snapshot.index, &installed.index));
                assert!(page.results.iter().all(|hit| hit.result.doc_id != "b"));
                assert_rows(&page.snapshot, &page.results);
            }
            let Some(NativeShardedSearchPhase::Refined { results, .. }) = phases.next_phase().await.unwrap() else {
                panic!("old refinement must retain the original cohort");
            };
            assert!(results.iter().any(|hit| hit.result.doc_id == "b"));
            assert_rows(&old, &results);
            assert_rows(&old_page.snapshot, &old_page.results);
            assert_eq!(old_page.snapshot.generation(), generation(1));
            assert_eq!(image(old.index().vectors().directory()), old_image);
        }
    });
}

#[test]
fn newer_sequences_and_foreign_handles_cannot_overwrite_a_winning_update() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, quality) = fixture(&cx, root.path(), false).await;
        let old = live.snapshot(&cx).await.unwrap();
        let other = NativeLiveShardedHybridIndex::new(&cx,
            build(&cx, &root.path().join("other"), 1, &fast, Some(&quality), false).await).unwrap();
        let foreign = other.snapshot(&cx).await.unwrap();
        assert_eq!(foreign.generation(), old.generation());
        assert_eq!(foreign.index().vectors().fast().owner_witnesses().collect::<Vec<_>>(),
            old.index().vectors().fast().owner_witnesses().collect::<Vec<_>>());
        let winner = old.begin_update(&cx, root.path().join("winner"), generation(2)).unwrap()
            .upsert_document(IndexableDocument::new("winner", "horizontal"))
            .build(&cx, 3).await.unwrap();
        let late = old.begin_update(&cx, root.path().join("late"), generation(9)).unwrap()
            .upsert_document(IndexableDocument::new("late", "vertical"))
            .build(&cx, 1).await.unwrap();
        assert_refusal(other.install(&cx, &winner).await.unwrap_err(), "native_ann.sharded_live.expected_current");
        assert!(Arc::ptr_eq(&other.snapshot(&cx).await.unwrap().index, &foreign.index));
        let installed = live.install(&cx, &winner).await.unwrap();
        for candidate in [&late, &winner] {
            assert_refusal(live.install(&cx, candidate).await.unwrap_err(), "native_ann.sharded_live.expected_current");
        }
        assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &installed.index));
        assert!(installed.index().vectors().document("winner").is_some());
        assert!(installed.index().vectors().document("late").is_none());
        assert!(late.index().vectors().document("late").is_some());
        let rebased = installed.begin_update(&cx, root.path().join("rebased"), generation(10)).unwrap()
            .upsert_document(IndexableDocument::new("late", "vertical"))
            .build(&cx, 2).await.unwrap();
        let current = live.install(&cx, &rebased).await.unwrap();
        assert!(current.index().vectors().document("winner").is_some());
        assert!(current.index().vectors().document("late").is_some());
    });
}

#[test]
fn pending_inference_and_builds_do_not_hold_the_selection_lock() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, quality) = fixture(&cx, root.path(), false).await;
        let old = live.snapshot(&cx).await.unwrap();
        let candidate = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
            .delete_document("b").build(&cx, 1).await.unwrap();
        fast.hold_queries.store(true, Ordering::SeqCst);
        let mut query = Box::pin(live.search(&cx, "vertical", 4));
        assert!(query.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        let mut install = Box::pin(live.install(&cx, &candidate));
        let Poll::Ready(Ok(current)) = install.as_mut().poll(&mut Context::from_waker(Waker::noop())) else {
            panic!("a pending query must not block the whole-cohort swap");
        };
        drop(install);
        fast.hold_queries.store(false, Ordering::SeqCst);
        let page = live.search_refined(&cx, "horizontal", 4).await.unwrap();
        assert!(Arc::ptr_eq(&page.snapshot.index, &current.index));
        assert!(page.results.iter().all(|hit| hit.result.doc_id != "b"));
        let before = fast.dropped.load(Ordering::SeqCst);
        drop(query);
        assert_eq!(fast.dropped.load(Ordering::SeqCst), before + 1);

        quality.hold_batches.store(true, Ordering::SeqCst);
        let update = current.begin_update(&cx, root.path().join("pending"), generation(3)).unwrap()
            .upsert_document(IndexableDocument::new("z", "vertical"));
        let mut building = Box::pin(update.build(&cx, 1));
        assert!(building.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        let page = live.search_quality(&cx, "horizontal", 4).await.unwrap();
        assert!(Arc::ptr_eq(&page.snapshot.index, &current.index));
        let before = quality.dropped.load(Ordering::SeqCst);
        drop(building);
        assert_eq!(quality.dropped.load(Ordering::SeqCst), before + 1);
        assert!(!root.path().join("pending/native.sharded-hybrid.json").exists());
        assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &current.index));
    });
}

#[test]
fn cancelled_and_dropped_lock_waiters_preserve_selection_and_candidate() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = fixture(&cx, root.path(), false).await;
        let old = live.snapshot(&cx).await.unwrap();
        let candidate = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
            .delete_document("b").build(&cx, 1).await.unwrap();
        let held = live.current.write(&cx).await.unwrap();
        let mut reader = Box::pin(live.snapshot(&cx));
        assert!(reader.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(reader);
        let mut abandoned = Box::pin(live.install(&cx, &candidate));
        assert!(abandoned.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(abandoned);
        let mut cancelled = Box::pin(live.install(&cx, &candidate));
        assert!(cancelled.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        cx.set_cancel_requested(true);
        assert!(matches!(cancelled.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::Cancelled { .. }))));
        drop(cancelled);
        drop(held);
        cx.set_cancel_requested(false);
        assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
        live.install(&cx, &candidate).await.unwrap();
        cx.set_cancel_requested(true);
        assert!(matches!(live.snapshot(&cx).await, Err(SearchError::Cancelled { .. })));
        assert!(matches!(live.install(&cx, &candidate).await, Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(2));
    });
}

#[test]
fn late_required_partition_failure_returns_no_installable_prefix() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, quality) = fixture(&cx, root.path(), true).await;
        let old = live.snapshot(&cx).await.unwrap();
        let before = image(old.index().vectors().directory());
        quality.fail_late.store(true, Ordering::SeqCst);
        let error = old.begin_update(&cx, root.path().join("failed"), generation(2)).unwrap()
            .upsert_document(IndexableDocument::new("z", "failuretoken vertical"))
            .build(&cx, 1).await.unwrap_err();
        assert_refusal(error, "native_ann.sharded_live.fixture");
        assert!(root.path().join("failed/shard-000000/fast.fsvi").is_file(),
            "failure must follow a completed early partition");
        assert!(!root.path().join("failed/native.sharded-hybrid.json").exists());
        assert!(!root.path().join("failed/lexical").exists());
        assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
        assert_eq!(image(old.index().vectors().directory()), before);
        assert!(live.search_refined(&cx, "legacytoken", 4).await.unwrap()
            .results.iter().any(|hit| hit.result.doc_id == "b"));
        quality.fail_late.store(false, Ordering::SeqCst);
        let retry = old.begin_update(&cx, root.path().join("retry"), generation(2)).unwrap()
            .upsert_document(IndexableDocument::new("z", "failuretoken vertical"))
            .build(&cx, 1).await.unwrap();
        assert!(live.install(&cx, &retry).await.unwrap().index().vectors().document("z").is_some());
    });
}

#[test]
fn invalid_generations_partition_sizes_and_empty_successors_keep_identity_contracts() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, quality) = fixture(&cx, root.path(), false).await;
        let old = live.snapshot(&cx).await.unwrap();
        let refused = root.path().join("refused");
        assert!(old.begin_update(&cx, &refused, generation(1)).is_err());
        assert!(old.begin_update(&cx, &refused,
            ArtifactGenerationIdentityV1::new(1, [0x63; 16]).unwrap()).is_err());
        assert!(old.begin_update(&cx, &refused, generation(2)).unwrap().build(&cx, 0).await.is_err());
        assert!(!refused.exists());
        assert_eq!(fast.documents.load(Ordering::SeqCst), 4);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 4);
        let changed_precision = old.begin_update(&cx, root.path().join("precision"), generation(2)).unwrap()
            .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact).unwrap()
            .build(&cx, 1).await.unwrap();
        assert_eq!(fast.documents.load(Ordering::SeqCst), 8);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 4);
        let current = live.install(&cx, &changed_precision).await.unwrap();
        let mut delete_all = current.begin_update(&cx, root.path().join("empty"), generation(3)).unwrap();
        for document in sources() {
            delete_all = delete_all.delete_document(document.id);
        }
        let empty = delete_all.build(&cx, 3).await.unwrap();
        let empty = live.install(&cx, &empty).await.unwrap();
        assert_eq!(empty.index().vectors().document_count(), 0);
        assert_eq!(empty.index().vectors().partitions().len(), 1);
        assert!(empty.index().vectors().quality().is_some());
        assert!(live.search_refined(&cx, "vertical", 4).await.unwrap().results.is_empty());
        assert_eq!(fast.documents.load(Ordering::SeqCst), 8);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 4);
        assert_eq!(old.index().vectors().document_count(), 4);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn selected_activation_retains_models_and_rejects_stale_replay_after_repartitioning() {
    run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, root.path(), ann).await;
            let old = live.snapshot(&cx).await.unwrap();
            let path = root.path().join("selected");
            let candidate = old
                .begin_update(&cx, &path, generation(2))
                .unwrap()
                .delete_document("b")
                .upsert_document(IndexableDocument::new("new", "arrival vertical"))
                .build(&cx, 1)
                .await
                .unwrap();
            let receipt = candidate.index().seal_for_reopen(&cx).unwrap();
            drop(candidate);
            let before = image(&path);
            let counts = (
                fast.documents.load(Ordering::SeqCst),
                quality.documents.load(Ordering::SeqCst),
            );
            let limits = NativeShardedHybridReopenLimits::default();
            let selected = old.prepare_selected(&cx, &path, &receipt, limits).await.unwrap();
            let late = old.prepare_selected(&cx, &path, &receipt, limits).await.unwrap();
            assert_eq!(selected.index().vectors().partitions().len(), 4);
            for partition in selected.index().vectors().partitions() {
                let original = &old.index().vectors().partitions()[0];
                assert!(Arc::ptr_eq(&partition.fast.embedder, &original.fast.embedder));
                assert!(Arc::ptr_eq(
                    &partition.quality.as_ref().unwrap().embedder,
                    &original.quality.as_ref().unwrap().embedder,
                ));
            }
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            assert_eq!(
                (
                    fast.documents.load(Ordering::SeqCst),
                    quality.documents.load(Ordering::SeqCst),
                ),
                counts
            );
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
            let current = live.install(&cx, &selected).await.unwrap();
            assert_refusal(
                live.install(&cx, &late).await.unwrap_err(),
                "native_ann.sharded_live.expected_current",
            );
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &current.index));
            let page = live.search_refined(&cx, "arrival", 4).await.unwrap();
            assert!(page.results.iter().any(|hit| hit.result.doc_id == "new"));
            assert!(page.results.iter().all(|hit| hit.result.doc_id != "b"));
            assert_rows(&page.snapshot, &page.results);
            assert!(old.index().vectors().document("b").is_some());
            assert_refusal(
                current.prepare_selected(&cx, &path, &receipt, limits).await.unwrap_err(),
                "native_ann.sharded_live.generation",
            );
            let restarted = NativeBuiltShardedHybridIndex::open_selected(
                &cx, &path, &receipt, fast.clone(), Some(quality.clone()), limits,
            ).await.unwrap();
            let restarted = NativeLiveShardedHybridIndex::new(&cx, restarted).unwrap();
            let page = restarted.search_quality(&cx, "arrival", 4).await.unwrap();
            assert_eq!(page.snapshot.generation(), current.generation());
            assert_rows(&page.snapshot, &page.results);
            assert_eq!(image(&path), before);
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn selected_admission_requires_late_artifacts_and_aggregate_limits_before_installation() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, quality) = fixture(&cx, root.path(), true).await;
        let old = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("selected");
        let candidate = old.begin_update(&cx, &path, generation(2)).unwrap()
            .upsert_document(IndexableDocument::new("z", "vertical"))
            .build(&cx, 1).await.unwrap();
        let receipt = candidate.index().seal_for_reopen(&cx).unwrap();
        let last = candidate.index().vectors().partitions().last().unwrap();
        let artifacts = [
            last.fast().vector_path().to_path_buf(),
            last.fast().graph_path().unwrap().to_path_buf(),
            last.directory().join("native.source.jsonl"),
            path.join("lexical"),
        ];
        drop(candidate);
        let before = image(&path);
        let limits = NativeShardedHybridReopenLimits::default();
        for (ordinal, artifact) in artifacts.iter().enumerate() {
            let saved = root.path().join(format!("saved-artifact-{ordinal}"));
            fs::rename(artifact, &saved).unwrap();
            assert!(old.prepare_selected(&cx, &path, &receipt, limits).await.is_err());
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
            fs::rename(saved, artifact).unwrap();
        }
        let mut wrong = receipt;
        wrong.sha256[0] ^= 1;
        assert!(old.prepare_selected(&cx, &path, &wrong, limits).await.is_err());
        for fault in 0..3 {
            let mut limited = limits;
            match fault {
                0 => limited.vectors.max_documents = 4,
                1 => limited.vectors.max_shards = 1,
                _ => limited.vectors.max_artifact_bytes = 1,
            }
            assert!(old.prepare_selected(&cx, &path, &receipt, limited).await.is_err());
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
        }
        cx.set_cancel_requested(true);
        assert!(matches!(
            old.prepare_selected(&cx, root.path().join("not-created"), &receipt, limits).await,
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
        assert!(!root.path().join("not-created").exists());
        let counts = (
            fast.documents.load(Ordering::SeqCst),
            quality.documents.load(Ordering::SeqCst),
        );
        let selected = old.prepare_selected(&cx, &path, &receipt, limits).await.unwrap();
        live.install(&cx, &selected).await.unwrap();
        assert_eq!(image(&path), before);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert_eq!((fast.documents.load(Ordering::SeqCst), quality.documents.load(Ordering::SeqCst)), counts);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn selected_topology_producer_and_rollback_failures_leave_the_complete_head_untouched() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, quality) = fixture(&cx, root.path(), false).await;
        let old = live.snapshot(&cx).await.unwrap();
        let old_receipt = old.index().seal_for_reopen(&cx).unwrap();
        let foreign = Provider::new("same-dimension-other-producer", 2);
        for (name, provider, required_quality) in [
            ("foreign", &foreign, Some(&quality)),
            ("missing-quality", &fast, None),
        ] {
            let path = root.path().join(name);
            let other = build(&cx, &path, 2, provider, required_quality, false).await;
            let receipt = other.seal_for_reopen(&cx).unwrap();
            drop(other);
            assert!(old.prepare_selected(
                &cx, &path, &receipt, NativeShardedHybridReopenLimits::default(),
            ).await.is_err());
            assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &old.index));
        }
        let candidate = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
            .delete_document("b").build(&cx, 1).await.unwrap();
        let current = live.install(&cx, &candidate).await.unwrap();
        assert_refusal(
            current.prepare_selected(
                &cx, old.index().vectors().directory(), &old_receipt,
                NativeShardedHybridReopenLimits::default(),
            ).await.unwrap_err(),
            "native_ann.sharded_live.generation",
        );
        assert!(Arc::ptr_eq(&live.snapshot(&cx).await.unwrap().index, &current.index));
        assert!(current.index().vectors().document("b").is_none());
        assert!(old.index().vectors().document("b").is_some());
    });
}

#[test]
fn old_reranking_keeps_source_text_and_both_row_maps_after_live_replacement() {
    use std::sync::Mutex;
    use crate::{RerankDocument, RerankScore, Reranker};

    #[derive(Default)]
    struct Scorer {
        seen: Mutex<Vec<(String, String)>>,
    }
    impl Reranker for Scorer {
        fn id(&self) -> &'static str { "sharded-live-test-reranker" }
        fn model_name(&self) -> &str { self.id() }
        fn rerank<'a>(
            &'a self, _cx: &'a Cx, _query: &'a str, documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                *self.seen.lock().unwrap() = documents.iter()
                    .map(|document| (document.doc_id.clone(), document.text.clone())).collect();
                Ok(documents.iter().enumerate().map(|(original_rank, document)| RerankScore {
                    doc_id: document.doc_id.clone(), original_rank,
                    score: if document.doc_id == "b" { 1.0 } else { 0.0 }, raw_logit: None,
                }).collect())
            })
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = fixture(&cx, root.path(), true).await;
        let old = live.snapshot(&cx).await.unwrap();
        let scorer = Scorer::default();
        let mut phases = old.index().progressive_with_reranker(&cx, "vertical", 1, &scorer, 4).unwrap();
        assert!(matches!(phases.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { .. })));
        assert!(matches!(phases.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Refined { .. })));
        assert!(scorer.seen.lock().unwrap().is_empty(), "reranking must still be lazy");
        let next = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
            .delete_document("b")
            .upsert_document(IndexableDocument::new("a", "replacement vertical"))
            .build(&cx, 1).await.unwrap();
        let current = live.install(&cx, &next).await.unwrap();
        let Some(NativeShardedSearchPhase::Reranked { results, evaluated, .. }) =
            phases.next_phase().await.unwrap() else { panic!("retained rerank phase"); };
        assert_eq!(evaluated, 4);
        assert_eq!(results[0].result.doc_id, "b");
        assert_rows(&old, &results);
        let seen: BTreeMap<_, _> = scorer.seen.lock().unwrap().iter().cloned().collect();
        assert_eq!(seen["a"], "horizontal");
        assert_eq!(seen["b"], "legacytoken vertical");
        assert!(current.index().vectors().document("b").is_none());
        assert_eq!(current.index().vectors().document("a").unwrap().content, "replacement vertical");
        assert!(phases.next_phase().await.unwrap().is_none());
    });
}
