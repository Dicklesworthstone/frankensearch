use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use asupersync::runtime::RuntimeBuilder;
use asupersync::runtime::blocking_pool::BlockingPool;

use crate::{ModelCategory, SearchError, SearchFuture};

#[derive(Clone, Copy)]
pub(super) enum Reply {
    Correct,
    Short,
    Foreign,
    NonFinite,
    WrongDimension,
    Cancel,
    Pending,
    FailSecond,
}

pub(super) struct Provider {
    identity: EmbeddingIdentityBundleV1,
    reply: Reply,
    batches: AtomicUsize,
    pub(super) queries: AtomicUsize,
    max_batch: AtomicUsize,
    dropped: AtomicUsize,
}

impl Provider {
    pub(super) fn new(name: &str, dimension: u32, reply: Reply) -> Self {
        Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            reply,
            queries: AtomicUsize::new(0),
            batches: AtomicUsize::new(0),
            max_batch: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }
    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text == "vertical")] = 1.0;
        values
    }
}

struct DropCount<'a>(&'a AtomicUsize);
impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            Ok(self.values(text))
        })
    }
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let call = self.batches.fetch_add(1, Ordering::SeqCst);
            self.max_batch.fetch_max(texts.len(), Ordering::SeqCst);
            let _guard = DropCount(&self.dropped);
            if matches!(self.reply, Reply::Pending) {
                return std::future::pending().await;
            }
            if matches!(self.reply, Reply::FailSecond) && call == 1 {
                return Err(invalid("test.batch", "failed", "second batch failed"));
            }
            let mut outputs: Vec<_> = texts
                .iter()
                .map(|text| IdentityBoundEmbedding {
                    values: self.values(text),
                    identity: self.identity.clone(),
                })
                .collect();
            match self.reply {
                Reply::Short => {
                    let _ = outputs.pop();
                }
                Reply::Foreign => {
                    outputs.last_mut().unwrap().identity.producer.backend = "different".to_owned();
                }
                Reply::NonFinite => outputs.last_mut().unwrap().values[0] = f32::NAN,
                Reply::WrongDimension => {
                    let _ = outputs.last_mut().unwrap().values.pop();
                }
                Reply::Cancel => {
                    cx.cancel_with(asupersync::CancelKind::User, Some("cancel native build"));
                }
                _ => {}
            }
            Ok(outputs)
        })
    }
    // The bound batch above binds the same values as `embed` under the same
    // identity; faults deliberately violate that advertised native contract.
    fn bound_batch_is_native(&self) -> bool {
        true
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
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

pub(super) fn generation() -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(9, [0x46; 16]).unwrap()
}
pub(super) fn documents() -> Vec<IndexableDocument> {
    [
        ("z", "horizontal"),
        ("b", "vertical"),
        ("a", "horizontal"),
        ("x", "horizontal"),
        ("c", "horizontal"),
    ]
    .into_iter()
    .map(|(id, text)| IndexableDocument::new(id, text))
    .collect()
}
fn builder(path: &Path, provider: &Arc<Provider>) -> NativeIndexBuilder {
    NativeIndexBuilder::new(path, generation(), provider.clone())
        .unwrap()
        .add_documents(documents())
}

fn run_with_pool<F, Fut, T>(body: F) -> T
where
    F: FnOnce(Cx, BlockingPoolHandle) -> Fut,
    Fut: Future<Output = T>,
{
    let runtime = RuntimeBuilder::current_thread().build().unwrap();
    let pool = BlockingPool::new(0, 1);
    let handle = pool.handle();
    let result = runtime
        .block_on(async move { body(Cx::current().expect("runtime context"), handle).await });
    assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
    result
}

async fn wait_for_worker(entered: mpsc::Receiver<()>) {
    let started = Instant::now();
    std::future::poll_fn(move |task| match entered.try_recv() {
        Ok(()) => Poll::Ready(()),
        Err(mpsc::TryRecvError::Empty) => {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "worker did not start"
            );
            task.waker().wake_by_ref();
            Poll::Pending
        }
        Err(mpsc::TryRecvError::Disconnected) => panic!("worker exited before starting"),
    })
    .await;
}

#[test]
fn pooled_build_yields_before_sealing_and_roundtrips_both_native_tiers() {
    run_with_pool(|cx, pool| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generation");
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let graph = NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 7,
        };
        let (started, entered) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let mut blocker = Box::pin(NativeWorker::BUILD.run(&cx, pool.clone(), move |_| {
            assert!(!Cx::is_active());
            started.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(())
        }));
        assert!(
            blocker
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        wait_for_worker(entered).await;
        let mut pending = Box::pin(
            builder(&path, &fast)
                .with_quality_embedder(quality.clone())
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, graph)
                .with_quality_storage(NativeBuildPrecision::F16, graph)
                .unwrap()
                .with_blocking_pool(pool)
                .build(&cx),
        );
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert!(path.is_dir());
        assert_eq!(fast.batches.load(Ordering::SeqCst), 1);
        assert_eq!(quality.batches.load(Ordering::SeqCst), 1);
        assert!(!path.join("fast.fsvi").exists());
        assert!(!path.join("quality.fsvi").exists());
        // Finalization is queued on the real occupied pool, while unrelated
        // work on this single-thread executor continues to make progress.
        assert_eq!(std::future::ready(37_u8).await, 37);
        release.send(()).unwrap();
        blocker.await.unwrap();
        let built = pending.await.unwrap();
        assert!(built.fast().graph_path().unwrap().is_file());
        assert!(built.quality().unwrap().graph_path().unwrap().is_file());
        let seal = built.seal_for_reopen(&cx).unwrap();
        let reopened =
            NativeBuiltIndex::open_selected(&cx, &path, &seal, fast, Some(quality)).unwrap();
        for (before, after) in [
            (built.fast(), reopened.fast()),
            (built.quality().unwrap(), reopened.quality().unwrap()),
        ] {
            let hits = before.search(&cx, "vertical", 5).await.unwrap();
            assert_eq!(hits[0].doc_id, "b");
            assert_eq!(hits, after.search(&cx, "vertical", 5).await.unwrap());
        }
    });
}

#[test]
fn stopped_build_pools_refuse_initial_sharded_and_update_work_before_side_effects() {
    run_with_pool(|cx, _pool| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let original = builder(&dir.path().join("original"), &fast)
            .build(&cx)
            .await
            .unwrap();
        let calls = fast.batches.load(Ordering::SeqCst);
        let stopped = BlockingPool::new(0, 1);
        let handle = stopped.handle();
        assert!(stopped.shutdown_and_wait(Duration::from_secs(10)));
        let inherited = cx.clone().with_blocking_pool_handle(Some(handle.clone()));
        let single = dir.path().join("single");
        let shards = dir.path().join("shards");
        let successor = dir.path().join("successor");
        assert!(matches!(
            builder(&single, &fast)
                .with_blocking_pool(handle.clone())
                .build(&cx)
                .await,
            Err(SearchError::SubsystemError {
                subsystem: "native_ann.build_worker",
                ..
            })
        ));
        assert!(matches!(
            builder(&shards, &fast).build_sharded(&inherited, 2).await,
            Err(SearchError::SubsystemError {
                subsystem: "native_ann.build_worker",
                ..
            })
        ));
        let mut newer = generation();
        newer.sequence += 1;
        assert!(matches!(
            original
                .begin_update(&cx, &successor, newer)
                .unwrap()
                .with_blocking_pool(handle)
                .upsert_document(IndexableDocument::new("new", "vertical"))
                .build(&cx)
                .await,
            Err(SearchError::SubsystemError {
                subsystem: "native_ann.build_worker",
                ..
            })
        ));
        assert_eq!(fast.batches.load(Ordering::SeqCst), calls);
        assert!(!single.exists());
        assert!(!shards.exists());
        assert!(!successor.exists());
        assert_eq!(
            original.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
    });
}

#[test]
fn abandoned_or_cancelled_queued_builds_never_seal_or_deliver_a_generation() {
    for cancel in [false, true] {
        let (dir, path) = run_with_pool(|cx, pool| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("candidate");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let (started, entered) = mpsc::channel();
            let (release, gate) = mpsc::channel();
            let mut blocker = Box::pin(NativeWorker::BUILD.run(&cx, pool.clone(), move |_| {
                started.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(())
            }));
            assert!(
                blocker
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            wait_for_worker(entered).await;
            let mut queued = Box::pin(builder(&path, &fast).with_blocking_pool(pool).build(&cx));
            assert!(
                queued
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(fast.batches.load(Ordering::SeqCst), 1);
            if cancel {
                cx.set_cancel_requested(true);
                // Cancellation must still join the owned queued worker.
                assert!(
                    queued
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                release.send(()).unwrap();
                assert!(matches!(queued.await, Err(SearchError::Cancelled { .. })));
                assert!(matches!(blocker.await, Err(SearchError::Cancelled { .. })));
            } else {
                drop(queued);
                release.send(()).unwrap();
                blocker.await.unwrap();
                assert!(!cx.is_cancel_requested());
            }
            (dir, path)
        });
        // Observe after the pool has drained, so a late writer cannot escape
        // the assertion merely because its awaiting future was dropped.
        assert!(path.is_dir());
        assert!(!path.join("fast.fsvi").exists());
        assert!(!path.join("fast.fshnsw").exists());
        drop(dir);
    }
}

#[test]
fn inherited_pool_builds_every_partition_and_incremental_successor() {
    run_with_pool(|cx, pool| async move {
        let cx = cx.with_blocking_pool_handle(Some(pool));
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let original = builder(&dir.path().join("original"), &fast)
            .with_fast_storage(
                NativeBuildPrecision::F32,
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 7,
                },
            )
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        assert_eq!(original.partitions().len(), 3);
        let calls = fast.batches.load(Ordering::SeqCst);
        let mut newer = generation();
        newer.sequence += 1;
        let successor = original
            .begin_update(&cx, dir.path().join("successor"), newer)
            .unwrap()
            .delete_document("a")
            .upsert_document(IndexableDocument::new("b", "horizontal"))
            .upsert_document(IndexableDocument::new("new", "vertical"))
            .build_sharded(&cx, 3)
            .await
            .unwrap();
        assert_eq!(successor.document_count(), 5);
        assert_eq!(successor.partitions().len(), 2);
        assert!(fast.batches.load(Ordering::SeqCst) > calls);
        for partition in successor.partitions() {
            assert!(partition.fast().graph_path().unwrap().is_file());
            assert_eq!(partition.fast().binding().generation(), newer);
        }
        assert_eq!(original.document("b").unwrap().content, "vertical");
        assert_eq!(successor.document("b").unwrap().content, "horizontal");
        assert!(successor.document("a").is_none());
        assert_eq!(successor.document("new").unwrap().content, "vertical");
    });
}

#[test]
fn native_document_build_batches_both_tiers_and_retains_the_source_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("generation"), &fast)
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_batch_size(2)
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        assert_eq!(fast.batches.load(Ordering::SeqCst), 3);
        assert_eq!(quality.batches.load(Ordering::SeqCst), 3);
        assert_eq!(fast.max_batch.load(Ordering::SeqCst), 2);
        assert_eq!(quality.max_batch.load(Ordering::SeqCst), 2);
        assert_eq!(
            index
                .documents()
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c", "x", "z"]
        );
        assert_eq!(index.fast().index().live_count(), 5);
        assert_eq!(index.quality().unwrap().index().live_count(), 5);
        assert_eq!(
            index.fast().index().owner_witness().generation,
            index.quality().unwrap().index().owner_witness().generation
        );
        assert_eq!(index.document("b").unwrap().content, "vertical");
        assert!(index.document("absent").is_none());
        drop(dir); // Search and source hydration must not reopen these paths.
        assert_eq!(
            index.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
        assert_eq!(
            index
                .quality()
                .unwrap()
                .search(&cx, "vertical", 1)
                .await
                .unwrap()[0]
                .doc_id,
            "b"
        );
    });
}

#[test]
fn built_native_sidecars_reopen_with_the_exact_written_v2_binding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for precision in [NativeBuildPrecision::F32, NativeBuildPrecision::F16] {
            let dir = tempfile::tempdir().unwrap();
            let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let index = builder(&dir.path().join("generation"), &provider)
                .with_fast_storage(
                    precision,
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 7,
                    },
                )
                .build(&cx)
                .await
                .unwrap();
            let tier = index.fast();
            let bytes: Arc<[u8]> = std::fs::read(tier.vector_path()).unwrap().into();
            let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, tier.binding()).unwrap());
            assert_eq!(owner.witness(), tier.index().owner_witness());
            let reopened = NativeAnnIndex::load(&cx, owner, tier.graph_path().unwrap()).unwrap();
            assert_eq!(
                reopened
                    .search_text(&cx, provider.as_ref(), "vertical", 5, None)
                    .await
                    .unwrap(),
                tier.search(&cx, "vertical", 5).await.unwrap()
            );
        }
    });
}

#[test]
fn document_order_does_not_change_vector_bytes_or_row_identity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let first = builder(&dir.path().join("one"), &provider)
            .build(&cx)
            .await
            .unwrap();
        let mut docs = documents();
        docs.reverse();
        let second =
            NativeIndexBuilder::new(dir.path().join("two"), generation(), provider.clone())
                .unwrap()
                .add_documents(docs)
                .build(&cx)
                .await
                .unwrap();
        assert_eq!(
            std::fs::read(first.fast().vector_path()).unwrap(),
            std::fs::read(second.fast().vector_path()).unwrap()
        );
        assert_eq!(
            first.fast().search(&cx, "vertical", 5).await.unwrap(),
            second.fast().search(&cx, "vertical", 5).await.unwrap()
        );
    });
}

#[test]
fn invalid_source_and_configuration_start_no_provider_or_directory_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generation");
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        assert!(builder(&path, &provider).with_batch_size(0).is_err());
        for id in ["", "a"] {
            assert!(
                builder(&path, &provider)
                    .add_document(IndexableDocument::new(id, "extra"))
                    .build(&cx)
                    .await
                    .is_err()
            );
            assert!(!path.exists());
        }
        let params = HnswParams {
            ef_search: 0,
            ..HnswParams::default()
        };
        assert!(
            builder(&path, &provider)
                .with_fast_storage(
                    NativeBuildPrecision::F32,
                    NativeBuildRetrieval::Hnsw { params, seed: 7 }
                )
                .build(&cx)
                .await
                .is_err()
        );
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
        assert!(!path.exists());
    });
}

#[test]
fn existing_destination_is_never_adopted_or_truncated() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("fast.fsvi"), b"existing vector bytes").unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        assert!(builder(dir.path(), &provider).build(&cx).await.is_err());
        assert_eq!(
            std::fs::read(dir.path().join("fast.fsvi")).unwrap(),
            b"existing vector bytes"
        );
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn malformed_batch_or_later_failure_returns_no_partial_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for reply in [
            Reply::Short,
            Reply::Foreign,
            Reply::NonFinite,
            Reply::WrongDimension,
            Reply::FailSecond,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("generation");
            let provider = Arc::new(Provider::new("fast", 2, reply));
            assert!(
                builder(&path, &provider)
                    .with_batch_size(2)
                    .unwrap()
                    .build(&cx)
                    .await
                    .is_err()
            );
            // The failed attempt may leave files for diagnostics. Directory
            // existence is not evidence of a completed, selectable generation.
            assert!(path.is_dir());
        }
    });
}

#[test]
fn required_quality_failure_never_degrades_to_a_fast_only_build() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Foreign));
        assert!(
            builder(&dir.path().join("generation"), &fast)
                .with_quality_embedder(quality.clone())
                .unwrap()
                .build(&cx)
                .await
                .is_err()
        );
        assert_eq!(fast.batches.load(Ordering::SeqCst), 1);
        assert_eq!(quality.batches.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn actual_context_cancellation_dominates_successful_batch_output() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Cancel));
        let result = builder(&dir.path().join("generation"), &provider)
            .build(&cx)
            .await;
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn dropping_pending_build_drops_provider_work_without_retry() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generation");
        let provider = Arc::new(Provider::new("fast", 2, Reply::Pending));
        let mut future = Box::pin(builder(&path, &provider).build(&cx));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        drop(future);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(provider.dropped.load(Ordering::SeqCst), 1);
        assert!(path.is_dir());
    });
}

#[test]
fn empty_cohort_is_admitted_without_empty_provider_batches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Pending));
        let index =
            NativeIndexBuilder::new(dir.path().join("empty"), generation(), provider.clone())
                .unwrap()
                .build(&cx)
                .await
                .unwrap();
        assert!(index.documents().is_empty());
        assert!(
            index
                .fast()
                .search(&cx, "anything", 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn source_join_uses_document_ids_not_fsvi_physical_sort_positions() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("generation"), &provider)
            .with_batch_size(2)
            .unwrap()
            .build(&cx)
            .await
            .unwrap();
        let owner = &index.fast.index.owner;
        let physical: Vec<_> = (0..owner.record_count())
            .map(|position| owner.row(position).unwrap().doc_id().to_owned())
            .collect();
        let source: Vec<_> = index.documents().iter().map(|doc| doc.id.clone()).collect();
        assert_ne!(
            physical, source,
            "fixture must distinguish physical hash order from source ID order"
        );
        for position in 0..owner.record_count() {
            let row = owner.row(position).unwrap();
            let source = index.document(row.doc_id()).unwrap();
            assert_eq!(
                owner.vector_at_f32(position).unwrap(),
                provider.values(&source.content)
            );
        }
    });
}
