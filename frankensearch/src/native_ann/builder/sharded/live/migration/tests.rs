use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::native_hnsw::HnswParams;

use crate::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
use crate::native_ann::{NativeShardedResult, NativeShardedSearchPhase};
use crate::{IndexableDocument, ModelCategory, SearchError, SearchFuture};

const FAIL_LATE: usize = 1;
const FOREIGN_LATE: usize = 2;
const CANCEL_LATE: usize = 3;
const PENDING_LATE: usize = 4;
const DRIFT_LATE: usize = 5;

// Deterministic identified control providers drive actual FSVI, HNSW and Quill
// builders. These are ownership/dispatch tests, not real-model quality evidence.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    unavailable: AtomicBool,
    changed: AtomicBool,
    fault: AtomicUsize,
    batches: Mutex<Vec<Vec<String>>>,
    queries: AtomicUsize,
    dropped: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        let mut foreign = identity.clone();
        "private-producer-canary".clone_into(&mut foreign.producer.backend);
        Self {
            identity,
            foreign,
            unavailable: AtomicBool::new(false),
            changed: AtomicBool::new(false),
            fault: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
            queries: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }

    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text.contains("vertical"))] = 1.0;
        values
    }

    fn submitted(&self) -> Vec<String> {
        self.batches
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }

    fn failure() -> SearchError {
        SearchError::EmbeddingFailed {
            model: "migration-fixture".to_owned(),
            source: "required target model failed".into(),
        }
    }
}

struct PendingDrop<'a>(&'a AtomicUsize);
impl Drop for PendingDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            if self.unavailable.load(Ordering::SeqCst) {
                return Err(Self::failure());
            }
            Ok(self.values(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.batches
                .lock()
                .unwrap()
                .push(texts.iter().map(|text| (*text).to_owned()).collect());
            let fault = if texts.iter().any(|text| text.contains("lastdoc")) {
                self.fault.load(Ordering::SeqCst)
            } else {
                0
            };
            match fault {
                FAIL_LATE => return Err(Self::failure()),
                CANCEL_LATE => cx.set_cancel_requested(true),
                PENDING_LATE => {
                    let _guard = PendingDrop(&self.dropped);
                    return std::future::pending().await;
                }
                DRIFT_LATE => {
                    self.changed.store(true, Ordering::SeqCst);
                    return Err(Self::failure());
                }
                _ => {}
            }
            Ok(texts
                .iter()
                .map(|text| IdentityBoundEmbedding {
                    values: self.values(text),
                    identity: if fault == FOREIGN_LATE {
                        self.foreign.clone()
                    } else {
                        self.identity.clone()
                    },
                })
                .collect())
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        true
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        if self.unavailable.load(Ordering::SeqCst) {
            return Err(Self::failure());
        }
        Ok(if self.changed.load(Ordering::SeqCst) {
            &self.foreign
        } else {
            &self.identity
        })
    }
    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [0x57; 16]).unwrap()
}

fn sources() -> Vec<IndexableDocument> {
    [
        "horizontal",
        "legacytoken vertical",
        "horizontal",
        "vertical",
        "horizontal",
        "vertical",
        "lastdoc",
    ]
    .into_iter()
    .enumerate()
    .map(|(number, text)| {
        IndexableDocument::new(format!("doc-{number}"), text)
            .with_title(format!("Title {number}"))
            .with_metadata("version", format!("original-{number}"))
    })
    .collect()
}

async fn initial(
    cx: &Cx,
    root: &Path,
    documents: Vec<IndexableDocument>,
    quality: bool,
) -> (NativeLiveShardedHybridIndex, Arc<Provider>, Arc<Provider>) {
    let fast = Arc::new(Provider::new("old-fast", 2));
    let quality_model = Arc::new(Provider::new("old-quality", 3));
    let mut builder = NativeIndexBuilder::new(root.join("initial"), generation(1), fast.clone())
        .unwrap()
        .add_documents(documents);
    if quality {
        builder = builder
            .with_quality_embedder(quality_model.clone())
            .unwrap();
    }
    let built = builder.build_sharded_hybrid(cx, 2).await.unwrap();
    (
        NativeLiveShardedHybridIndex::new(cx, built).unwrap(),
        fast,
        quality_model,
    )
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
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(path).unwrap(),
                );
            }
        }
    }
    files
}

fn assert_rows(snapshot: &NativeShardedSnapshot, hits: &[NativeShardedResult]) {
    let mut resolved = 0;
    for hit in hits {
        assert!(hit.result.index.is_none());
        for (row, quality) in [(hit.fast_row, false), (hit.quality_row, true)] {
            if let Some(row) = row {
                let partition = &snapshot.index().vectors().partitions()[row.shard];
                let tier = if quality {
                    partition.quality().unwrap()
                } else {
                    partition.fast()
                };
                assert_eq!(
                    tier.index
                        .owner
                        .doc_id_at(usize::try_from(row.physical_row).unwrap())
                        .unwrap(),
                    hit.result.doc_id
                );
                resolved += 1;
            }
        }
    }
    assert!(resolved > 0, "must exercise original physical shard coordinates");
}

fn graph(seed: u64) -> NativeBuildRetrieval {
    NativeBuildRetrieval::Hnsw {
        params: HnswParams::default(),
        seed,
    }
}

#[test]
fn migration_repartitions_every_target_tier_and_keeps_old_progressive_queries() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, old_fast, old_quality) = initial(&cx, root.path(), sources(), true).await;
        let old = live.snapshot(&cx).await.unwrap();
        let old_image = image(old.index().vectors().directory());
        let mut phases = old.index().progressive(&cx, "legacytoken", 7).unwrap();
        let Some(NativeShardedSearchPhase::Initial { results, .. }) =
            phases.next_phase().await.unwrap()
        else {
            panic!("expected Initial");
        };
        assert_rows(&old, &results);
        let fast = Arc::new(Provider::new("new-fast", 4));
        let quality = Arc::new(Provider::new("new-quality", 5));
        let candidate = old
            .begin_model_migration(
                &cx,
                root.path().join("migrated"),
                generation(2),
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap()
            .with_batch_size(2)
            .unwrap()
            .with_fast_storage(NativeBuildPrecision::F16, graph(17))
            .with_quality_storage(NativeBuildPrecision::F32, graph(29))
            .unwrap()
            .build(&cx, 3)
            .await
            .unwrap();
        let expected: Vec<_> = sources().into_iter().map(|doc| doc.content).collect();
        for provider in [&fast, &quality] {
            assert_eq!(provider.submitted(), expected);
            assert_eq!(
                provider
                    .batches
                    .lock()
                    .unwrap()
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>(),
                [2, 1, 2, 1, 1]
            );
            assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
        }
        assert_eq!(old.index().vectors().partitions().len(), 4);
        assert_eq!(candidate.index().vectors().partitions().len(), 3);
        assert_eq!(candidate.index().lexical().doc_count().unwrap(), 7);
        let fresh = live.install(&cx, &candidate).await.unwrap();
        for partition in fresh.index().vectors().partitions() {
            assert_eq!(partition.fast().producer_identity, fast.identity);
            assert_eq!(partition.quality().unwrap().producer_identity, quality.identity);
            assert_eq!(partition.fast().precision, NativeBuildPrecision::F16);
            assert!(partition.fast().graph_path().is_some());
            assert!(partition.quality().unwrap().graph_path().is_some());
        }
        let Some(NativeShardedSearchPhase::Refined { results, .. }) =
            phases.next_phase().await.unwrap()
        else {
            panic!("old query must finish on the old quality model");
        };
        assert_rows(&old, &results);
        assert!(old_fast.queries.load(Ordering::SeqCst) > 0);
        assert!(old_quality.queries.load(Ordering::SeqCst) > 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        let page = live.search_refined(&cx, "legacytoken", 7).await.unwrap();
        assert_eq!(page.snapshot.generation(), generation(2));
        assert_rows(&page.snapshot, &page.results);
        assert_eq!(image(old.index().vectors().directory()), old_image);
    });
}

#[test]
fn migration_can_explicitly_add_and_remove_quality_without_reusing_fast_vectors() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, _) = initial(&cx, root.path(), sources(), false).await;
        let base = live.snapshot(&cx).await.unwrap();
        let submitted = fast.submitted().len();
        let quality = Arc::new(Provider::new("added-quality", 4));
        let added = base
            .begin_model_migration(
                &cx,
                root.path().join("added"),
                generation(2),
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap()
            .build(&cx, 3)
            .await
            .unwrap();
        assert_eq!(
            fast.submitted().len(),
            submitted + 7,
            "unchanged models must also rebuild"
        );
        assert_eq!(quality.submitted().len(), 7);
        let with_quality = live.install(&cx, &added).await.unwrap();
        let replacement = Arc::new(Provider::new("fast-only", 6));
        let removed = with_quality
            .begin_model_migration(
                &cx,
                root.path().join("removed"),
                generation(3),
                replacement.clone(),
                None,
            )
            .unwrap()
            .build(&cx, 4)
            .await
            .unwrap();
        let without_quality = live.install(&cx, &removed).await.unwrap();
        assert!(without_quality.index().vectors().quality().is_none());
        assert!(with_quality.index().vectors().quality().is_some());
        assert_eq!(replacement.submitted().len(), 7);
        assert!(without_quality.index().search_quality(&cx, "vertical", 3).await.is_err());
        assert!(!with_quality.index().search_quality(&cx, "vertical", 3).await.unwrap().is_empty());
    });
}

#[test]
fn migration_and_source_updates_share_the_exact_predecessor_fence() {
    run_test_with_cx(|cx| async move {
        for update_first in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let (live, _, _) = initial(&cx, root.path(), sources(), false).await;
            let base = live.snapshot(&cx).await.unwrap();
            let migration = base
                .begin_model_migration(
                    &cx,
                    root.path().join("migration"),
                    generation(50),
                    Arc::new(Provider::new("new", 4)),
                    None,
                )
                .unwrap()
                .build(&cx, 3)
                .await
                .unwrap();
            let update = base
                .begin_update(&cx, root.path().join("edit"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("doc-6", "newsource"))
                .build(&cx, 2)
                .await
                .unwrap();
            let (winner, loser) = if update_first {
                (&update, &migration)
            } else {
                (&migration, &update)
            };
            let installed = live.install(&cx, winner).await.unwrap();
            let error = live.install(&cx, loser).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.sharded_live.expected_current"));
            assert!(Arc::ptr_eq(&installed.index, &live.snapshot(&cx).await.unwrap().index));
            assert_eq!(
                installed.index().vectors().document("doc-6").unwrap().content,
                if update_first { "newsource" } else { "lastdoc" }
            );
        }
    });
}

#[test]
fn migration_late_required_quality_failure_never_installs_a_successful_prefix() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let original = image(base.index().vectors().directory());
        for fault in [FAIL_LATE, FOREIGN_LATE, CANCEL_LATE, DRIFT_LATE] {
            let fast = Arc::new(Provider::new("target-fast", 4));
            let quality = Arc::new(Provider::new("target-quality", 5));
            quality.fault.store(fault, Ordering::SeqCst);
            let error = base
                .begin_model_migration(
                    &cx,
                    root.path().join(format!("failed-{fault}")),
                    generation(2),
                    fast,
                    Some(quality.clone()),
                )
                .unwrap()
                .with_batch_failure_splitting()
                .build(&cx, 2)
                .await
                .unwrap_err();
            cx.set_cancel_requested(false);
            assert_eq!(quality.submitted().len(), 7, "failure must be in the last partition");
            if fault == CANCEL_LATE {
                assert!(matches!(error, SearchError::Cancelled { .. }));
            } else if fault == FAIL_LATE {
                assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
            } else {
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            }
            assert!(Arc::ptr_eq(&base.index, &live.snapshot(&cx).await.unwrap().index));
            assert_eq!(image(base.index().vectors().directory()), original);
        }
    });
}

#[test]
fn dropped_migration_releases_the_pending_model_without_holding_selection() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), false).await;
        let base = live.snapshot(&cx).await.unwrap();
        let fast = Arc::new(Provider::new("pending", 4));
        fast.fault.store(PENDING_LATE, Ordering::SeqCst);
        let intent = base
            .begin_model_migration(
                &cx,
                root.path().join("pending"),
                generation(2),
                fast.clone(),
                None,
            )
            .unwrap();
        let mut future = Box::pin(intent.build(&cx, 2));
        {
            let mut context = Context::from_waker(Waker::noop());
            assert!(future.as_mut().poll(&mut context).is_pending());
        }
        assert_eq!(fast.submitted().len(), 7);
        assert!(Arc::ptr_eq(&base.index, &live.snapshot(&cx).await.unwrap().index));
        drop(future);
        assert_eq!(fast.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(fast.submitted().len(), 7);
        assert!(!live.search(&cx, "legacytoken", 3).await.unwrap().results.is_empty());
    });
}

#[test]
fn migration_admits_source_contracts_and_all_limits_before_creating_partitions() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let mut wrong = Provider::new("different-input", 4);
        wrong.identity.input.document_instruction = "private-input-canary".to_owned();
        let wrong = Arc::new(wrong);
        let path = root.path().join("wrong-input");
        let error = base
            .begin_model_migration(&cx, &path, generation(2), wrong.clone(), None)
            .err()
            .unwrap();
        assert!(matches!(&error, SearchError::InvalidConfig { field, .. }
            if field == "native_ann.sharded_live.migration.input"));
        assert!(!error.to_string().contains("private-input-canary"));
        assert!(!path.exists());
        assert!(wrong.submitted().is_empty());
        let fast = Arc::new(Provider::new("target", 4));
        assert!(base.begin_model_migration(&cx, &path, generation(1), fast.clone(), None).is_err());
        for (suffix, shard_size, byte_limit) in [("zero", 0, 1024), ("bytes", 2, 1)] {
            let path = root.path().join(suffix);
            assert!(base
                .begin_model_migration(&cx, &path, generation(2), fast.clone(), None)
                .unwrap()
                .with_max_batch_input_bytes(byte_limit)
                .unwrap()
                .build(&cx, shard_size)
                .await
                .is_err());
            assert!(!path.exists());
        }
        assert!(fast.submitted().is_empty());
        cx.set_cancel_requested(true);
        assert!(matches!(
            base.begin_model_migration(&cx, &path, generation(2), fast, None),
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
    });
}

#[test]
fn empty_migrations_preserve_an_identity_bearing_partition_without_inference() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), Vec::new(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let fast = Arc::new(Provider::new("new-empty", 4));
        let candidate = base
            .begin_model_migration(
                &cx,
                root.path().join("empty"),
                generation(2),
                fast.clone(),
                None,
            )
            .unwrap()
            .build(&cx, 3)
            .await
            .unwrap();
        let installed = live.install(&cx, &candidate).await.unwrap();
        assert_eq!(installed.index().vectors().document_count(), 0);
        assert_eq!(installed.index().vectors().partitions().len(), 1);
        assert_eq!(installed.index().lexical().doc_count().unwrap(), 0);
        assert!(installed.index().vectors().quality().is_none());
        assert!(fast.submitted().is_empty());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(
            installed.index().vectors().partitions()[0].fast().producer_identity,
            fast.identity
        );
    });
}

#[test]
fn failed_old_models_do_not_block_rebuilding_the_retained_source() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, old_fast, old_quality) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        old_fast.unavailable.store(true, Ordering::SeqCst);
        old_quality.unavailable.store(true, Ordering::SeqCst);
        let fast = Arc::new(Provider::new("replacement", 4));
        let candidate = base
            .begin_model_migration(
                &cx,
                root.path().join("recovery"),
                generation(2),
                fast,
                None,
            )
            .unwrap()
            .build(&cx, 2)
            .await
            .unwrap();
        live.install(&cx, &candidate).await.unwrap();
        assert!(!live.search(&cx, "legacytoken", 3).await.unwrap().results.is_empty());
        assert_eq!(old_fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(old_quality.queries.load(Ordering::SeqCst), 0);
        assert_eq!(old_fast.submitted().len(), 7);
        assert_eq!(old_quality.submitted().len(), 7);
    });
}
