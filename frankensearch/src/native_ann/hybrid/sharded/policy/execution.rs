//! Caller-owned blocking placement for retained native vector queries.
//!
//! A job owns the exact admitted index and the actual bound query. It does not
//! open files, reconstruct models, select a generation or publish results into
//! a live handle. One shard-set job uses the existing bounded global merger;
//! this is not a fan-out of one task or one embedding request per partition.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use asupersync::runtime::JoinError;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch_core::BoundQueryEmbedding;

use crate::native_ann::{NativeAnnIndex, NativeShardHit, NativeShardSet, checkpoint};
use crate::{Cx, SearchError, SearchResult, VectorHit};

fn worker_error(reason: &'static str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "native_ann.query_worker",
        source: reason.into(),
    }
}

fn cancelled() -> SearchError {
    SearchError::Cancelled {
        phase: "native_ann.query_worker".to_owned(),
        reason: "native query work was abandoned or cancelled".to_owned(),
    }
}

fn require_pool(cx: &Cx, pool: &BlockingPoolHandle) -> SearchResult<()> {
    checkpoint(cx, "native_ann.query_worker.admission")?;
    if pool.is_shutdown() {
        return Err(worker_error("the caller-owned query pool is shut down"));
    }
    Ok(())
}

// A queued closure can outlive its join future. Do not run expensive graph or
// caller predicate work when the receiver has already abandoned the request.
// This flag never cancels the caller's Cx or other requests sharing the pool.
struct Awaiting(Arc<AtomicBool>);

impl Drop for Awaiting {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn worker_checkpoint(cx: &Cx, child: &Cx, abandoned: &AtomicBool) -> SearchResult<()> {
    checkpoint(cx, "native_ann.query_worker.request")?;
    checkpoint(child, "native_ann.query_worker.child")?;
    if abandoned.load(Ordering::Acquire) {
        return Err(cancelled());
    }
    Ok(())
}

async fn on_pool<T, F>(cx: &Cx, pool: BlockingPoolHandle, search: F) -> SearchResult<T>
where
    T: Send + 'static,
    F: FnOnce(&Cx) -> SearchResult<T> + Send + 'static,
{
    require_pool(cx, &pool)?;
    let awaiting = Awaiting(Arc::new(AtomicBool::new(false)));
    let abandoned = Arc::clone(&awaiting.0);
    let request = cx.clone();
    let worker_cx = cx.clone().with_blocking_pool_handle(Some(pool));
    let mut worker = worker_cx
        .spawn_blocking(move |child: Cx| {
            worker_checkpoint(&request, &child, &abandoned)?;
            // Shutdown can race submission. Asupersync can recover rejected
            // work on an executor task; this API must not silently run a graph
            // on that lane. Refuse before invoking search or its predicate.
            if Cx::is_active() {
                return Err(worker_error(
                    "native query work cannot run on the async executor",
                ));
            }
            let result = search(&request);
            worker_checkpoint(&request, &child, &abandoned)?;
            result
        })
        .map_err(|_| worker_error("cannot admit native query work"))?;
    let result = worker.join(cx).await;
    // The caller's cancellation wins over ready work AND worker failure. The
    // worker never owns a result-delivery callback or a live selection handle.
    checkpoint(cx, "native_ann.query_worker.delivery")?;
    let result = result.map_err(|error| match error {
        JoinError::Cancelled(_) => cancelled(),
        _ => worker_error("native query worker did not complete"),
    })?;
    drop(awaiting);
    result
}

impl NativeAnnIndex {
    /// Search an Arc-retained owner on the caller's blocking pool.
    ///
    /// The actual identity-bound query moves to the worker. The graph and FSVI
    /// bytes are shared, not copied, decoded or reopened. The ordinary search
    /// kernel, beam policy, exact fallback reason, scores and physical rows are
    /// unchanged. No worker is needed for an admitted zero-k or empty owner.
    /// Both query identity and pool availability are checked on those noops.
    ///
    /// The caller owns and drains the pool, and bounds aggregate concurrency.
    /// Dropping an unpolled future starts nothing. Dropping pending work prevents
    /// queued graph work from starting; a running synchronous search may drain
    /// before releasing its owners. It cannot publish a late result or retry.
    /// This is not a hard deadline, RSS bound or guarantee to preempt traversal.
    ///
    /// # Errors
    /// Propagates ordinary search/admission errors, cancellation and worker
    /// refusal. A stopped pool or executor-lane fallback never runs the graph.
    pub async fn search_with_pool(
        self: &Arc<Self>,
        cx: &Cx,
        query: BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_filtered_with_pool(cx, query, k, ef, pool, |_| true)
            .await
    }

    /// Execute the existing expanding filter policy off the async executor.
    ///
    /// The owned predicate must be stable for the request and may be called
    /// repeatedly during widening. Capture immutable source membership, not
    /// current mutable state. It is never called for a refused identity/noop.
    /// Unlike borrowing synchronous filters, its captures must be Send + 'static
    /// because already-running blocking work can outlive the awaiting future.
    ///
    /// # Errors
    /// Has the same failure/ownership contract as [`Self::search_with_pool`].
    #[allow(clippy::too_many_arguments)]
    pub async fn search_filtered_with_pool<F>(
        self: &Arc<Self>,
        cx: &Cx,
        query: BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        pool: BlockingPoolHandle,
        accept: F,
    ) -> SearchResult<Vec<VectorHit>>
    where
        F: Fn(&str) -> bool + Send + 'static,
    {
        checkpoint(cx, "native_ann.query_worker.single")?;
        self.admit_identity(query.identity())?;
        require_pool(cx, &pool)?;
        if k == 0 || self.live_count() == 0 {
            return self.search_filtered(cx, &query, k, ef, accept);
        }
        let index = Arc::clone(self);
        on_pool(cx, pool, move |request| {
            index.search_filtered(request, &query, k, ef, accept)
        })
        .await
    }
}

impl NativeShardSet {
    /// Search the complete retained partition inventory off the async executor.
    ///
    /// One worker executes the existing serial partition traversal and O(k)
    /// global winner heap. It does not allocate a task or embedding per shard,
    /// concatenate all local windows, copy graphs or omit failed partitions.
    /// Concurrent requests can use the caller's pool without blocking their
    /// polling executor. This does not claim parallelism within one shard set.
    /// Both query identity and pool availability are checked even on noops.
    ///
    /// Keep the originating set/snapshot while interpreting returned coordinates.
    /// The worker retains its original ordered owners even if a live selection
    /// changes or the caller drops another clone. No files or models are opened.
    /// The caller must retain/drain the pool; running traversal is not preempted.
    ///
    /// # Errors
    /// Returns the original query failure, cancellation or worker refusal. No
    /// successful prefix, fallback executor execution or late publication occurs.
    pub async fn search_with_pool(
        &self,
        cx: &Cx,
        query: BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<NativeShardHit>> {
        self.search_filtered_with_pool(cx, query, k, ef, pool, |_| true)
            .await
    }

    /// Apply the same stable owned filter before each shard's global merge.
    ///
    /// Filtering and widening remain in the existing native/exact kernels.
    /// Captures must remain valid on the worker after future drop. The predicate
    /// is not called on a refused identity or an admitted zero-result request.
    ///
    /// # Errors
    /// Has the same all-partition failure contract as [`Self::search_with_pool`].
    #[allow(clippy::too_many_arguments)]
    pub async fn search_filtered_with_pool<F>(
        &self,
        cx: &Cx,
        query: BoundQueryEmbedding,
        k: usize,
        ef: Option<usize>,
        pool: BlockingPoolHandle,
        accept: F,
    ) -> SearchResult<Vec<NativeShardHit>>
    where
        F: Fn(&str) -> bool + Send + 'static,
    {
        checkpoint(cx, "native_ann.query_worker.shards")?;
        // Admission guarantees a nonempty inventory, including empty corpora.
        self.shard(0)
            .ok_or_else(|| worker_error("missing admitted shard inventory"))?
            .admit_identity(query.identity())?;
        require_pool(cx, &pool)?;
        if k == 0 || self.live_count() == 0 {
            return self.search_filtered(cx, &query, k, ef, accept);
        }
        let index = self.clone();
        on_pool(cx, pool, move |request| {
            index.search_filtered(request, &query, k, ef, accept)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Mutex, mpsc};
    use std::task::{Context, Poll, Waker};
    use std::time::{Duration, Instant};

    use asupersync::runtime::RuntimeBuilder;
    use asupersync::runtime::blocking_pool::BlockingPool;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    fn run<F, Fut, T>(workers: usize, body: F) -> T
    where
        F: FnOnce(Cx, BlockingPoolHandle) -> Fut,
        Fut: Future<Output = T>,
    {
        let runtime = RuntimeBuilder::current_thread().build().unwrap();
        let pool = BlockingPool::new(0, workers);
        let handle = pool.handle();
        let result = runtime.block_on(async move {
            body(Cx::current().expect("runtime context"), handle).await
        });
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
        result
    }

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("native-query-pool", 2)
    }

    fn query() -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(vec![1.0, 0.0], identity()).unwrap()
    }

    // The ordinary writer/admission/search path, not a mock graph. Files can
    // leave scope because the admitted vector owner contains immutable bytes.
    fn index(
        cx: &Cx,
        precision: QuantizationFormat,
        ann: bool,
        rows: &[(&str, [f32; 2], bool)],
    ) -> Arc<NativeAnnIndex> {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("vectors.fsvi");
        let mut stored = identity();
        stored.storage.format = "fsvi-v2".to_owned();
        stored.storage.endianness = "little-endian".to_owned();
        stored.storage.quantization = precision;
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(1, [0x67; 16]).unwrap(),
            stored.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, values, live) in rows {
            if live {
                writer.write_record(id, &values).unwrap();
            } else {
                writer.write_tombstone_record(id, &values).unwrap();
            }
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        Arc::new(if ann {
            NativeAnnIndex::build(cx, owner, HnswParams::default(), 31).unwrap()
        } else {
            NativeAnnIndex::exact(cx, owner).unwrap()
        })
    }

    fn rows() -> Vec<(&'static str, [f32; 2], bool)> {
        vec![
            ("deleted", [1.0, 0.0], false),
            ("z-tie", [0.8, 0.2], true),
            ("a-tie", [0.8, 0.2], true),
            ("vertical", [0.0, 1.0], true),
            ("negative", [-1.0, 0.0], true),
        ]
    }

    fn assert_hits(expected: &[VectorHit], actual: &[VectorHit]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(expected.doc_id, actual.doc_id);
            assert_eq!(expected.index, actual.index);
            assert_eq!(expected.score.to_bits(), actual.score.to_bits());
        }
    }

    // Cx submission first schedules a region-owned wrapper. Waiting on a
    // blocking receiver here would prevent a current-thread runtime from ever
    // dispatching that wrapper to the pool. Yield while observing the worker's
    // real start signal; wall time is only a test-hang guard, not a speed gate.
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
    fn pooled_single_search_preserves_precision_beams_filters_and_physical_rows() {
        run(2, |cx, pool| async move {
            for precision in [QuantizationFormat::F16, QuantizationFormat::F32] {
                for ann in [false, true] {
                    let index = index(&cx, precision, ann, &rows());
                    for k in [0, 1, 3, 10] {
                        for ef in [None, Some(1), Some(5)] {
                            let expected = index.search(&cx, &query(), k, ef).unwrap();
                            let actual = index
                                .search_with_pool(&cx, query(), k, ef, pool.clone())
                                .await
                                .unwrap();
                            assert_hits(&expected, &actual);
                            let expected = index
                                .search_filtered(&cx, &query(), k, ef, |id| id != "a-tie")
                                .unwrap();
                            let actual = index
                                .search_filtered_with_pool(
                                    &cx,
                                    query(),
                                    k,
                                    ef,
                                    pool.clone(),
                                    |id| {
                                        assert!(!Cx::is_active());
                                        id != "a-tie"
                                    },
                                )
                                .await
                                .unwrap();
                            assert_hits(&expected, &actual);
                            for hit in &actual {
                                assert_ne!(hit.doc_id, "deleted");
                                assert_eq!(
                                    index.owner.doc_id_at(hit.index as usize).unwrap(),
                                    hit.doc_id
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn pooled_shards_use_the_original_global_heap_and_mixed_owner_coordinates() {
        run(2, |cx, pool| async move {
            let parts = vec![
                index(&cx, QuantizationFormat::F16, true, &rows()),
                index(
                    &cx,
                    QuantizationFormat::F32,
                    false,
                    &[("first", [1.0, 0.0], true), ("other", [0.5, 0.5], true)],
                ),
                index(&cx, QuantizationFormat::F32, true, &[]),
            ];
            let witnesses = parts
                .iter()
                .map(|part| part.owner_witness().clone())
                .collect::<Vec<_>>();
            let shards = NativeShardSet::admit(&cx, &witnesses, parts).unwrap();
            for k in [0, 1, 4, 20] {
                let expected = shards.search(&cx, &query(), k, Some(8)).unwrap();
                let actual = shards
                    .search_with_pool(&cx, query(), k, Some(8), pool.clone())
                    .await
                    .unwrap();
                assert_eq!(actual, expected);
                let expected = shards
                    .search_filtered(&cx, &query(), k, Some(1), |id| id != "first")
                    .unwrap();
                let actual = shards
                    .search_filtered_with_pool(
                        &cx,
                        query(),
                        k,
                        Some(1),
                        pool.clone(),
                        |id| {
                            assert!(!Cx::is_active());
                            id != "first"
                        },
                    )
                    .await
                    .unwrap();
                assert_eq!(actual, expected);
                for (hit, expected) in actual.iter().zip(&expected) {
                    assert_eq!(hit.score.to_bits(), expected.score.to_bits());
                    let part = shards.shard(hit.row.shard).unwrap();
                    assert_eq!(
                        part.owner.doc_id_at(hit.row.physical_row as usize).unwrap(),
                        hit.doc_id
                    );
                }
            }
        });
    }

    #[test]
    fn refused_noop_and_cancelled_queries_never_invoke_an_owned_predicate() {
        run(1, |cx, pool| async move {
            let index = index(&cx, QuantizationFormat::F32, false, &rows());
            for k in [0, 1] {
                let foreign = BoundQueryEmbedding::new(
                    vec![1.0, 0.0],
                    EmbeddingIdentityBundleV1::explicit_test_model("foreign", 2),
                )
                .unwrap();
                assert!(index.search_filtered_with_pool(
                    &cx, foreign, k, None, pool.clone(), |_| panic!("foreign query predicate"),
                ).await.is_err());
            }
            assert!(index.search_filtered_with_pool(
                &cx, query(), 0, None, pool.clone(), |_| panic!("zero-k predicate"),
            ).await.unwrap().is_empty());
            cx.set_cancel_requested(true);
            assert!(matches!(index.search_filtered_with_pool(
                &cx, query(), 0, None, pool, |_| panic!("cancelled predicate"),
            ).await, Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn stopped_pool_refuses_search_without_executor_fallback() {
        run(1, |cx, _pool| async move {
            let index = index(&cx, QuantizationFormat::F32, false, &rows());
            let stopped = BlockingPool::new(0, 1);
            let handle = stopped.handle();
            assert!(stopped.shutdown_and_wait(Duration::from_secs(10)));
            for k in [0, 1] {
                let error = index.search_filtered_with_pool(
                    &cx, query(), k, None, handle.clone(), |_| panic!("inline fallback"),
                ).await.unwrap_err();
                assert!(matches!(error, SearchError::SubsystemError {
                    subsystem: "native_ann.query_worker", ..
                }));
            }
        });
    }

    #[test]
    fn abandoning_a_queued_search_prevents_late_graph_and_predicate_work() {
        let calls = run(1, |cx, pool| async move {
            let (started, entered) = mpsc::channel();
            let (release, gate) = mpsc::channel();
            let mut blocker = Box::pin(on_pool(&cx, pool.clone(), move |_| {
                assert!(!Cx::is_active());
                started.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(())
            }));
            assert!(blocker.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            wait_for_worker(entered).await;
            let index = index(&cx, QuantizationFormat::F32, true, &rows());
            let calls = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&calls);
            let mut queued = Box::pin(index.search_filtered_with_pool(
                &cx, query(), 2, None, pool.clone(), move |_| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    true
                },
            ));
            assert!(queued.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            drop(queued);
            release.send(()).unwrap();
            blocker.await.unwrap();
            assert!(!cx.is_cancel_requested());
            // A later request is valid and does not need to reset a shared Cx.
            assert!(!index.search_with_pool(&cx, query(), 1, None, pool).await.unwrap().is_empty());
            calls
        });
        // Check after runtime AND pool drain, not before queued work could run.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn running_search_yields_the_executor_and_cancellation_precedes_ready_hits() {
        run(1, |cx, pool| async move {
            let index = index(&cx, QuantizationFormat::F32, false, &rows());
            let (started, entered) = mpsc::channel();
            let (release, gate) = mpsc::channel();
            let gate = Mutex::new(gate);
            let first = AtomicBool::new(true);
            let mut pending = Box::pin(index.search_filtered_with_pool(
                &cx, query(), 3, None, pool.clone(), move |_| {
                    assert!(!Cx::is_active());
                    if first.swap(false, Ordering::SeqCst) {
                        started.send(()).unwrap();
                        gate.lock().unwrap().recv_timeout(Duration::from_secs(10)).unwrap();
                    }
                    true
                },
            ));
            assert!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            wait_for_worker(entered).await;
            // The real worker is still gated; this unrelated executor future
            // completes without waiting for graph work or moving the gate.
            assert_eq!(std::future::ready(37_u8).await, 37);
            cx.set_cancel_requested(true);
            release.send(()).unwrap();
            assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            assert!(!index.search_with_pool(&cx, query(), 1, None, pool).await.unwrap().is_empty());
        });
    }

    #[test]
    fn pending_drop_releases_the_retained_owner_after_the_running_worker_drains() {
        let weak = run(1, |cx, pool| async move {
            let index = index(&cx, QuantizationFormat::F32, false, &rows());
            let weak = Arc::downgrade(&index.owner);
            let (started, entered) = mpsc::channel();
            let (release, gate) = mpsc::channel();
            let gate = Mutex::new(gate);
            let first = AtomicBool::new(true);
            let mut pending = Box::pin(index.search_filtered_with_pool(
                &cx, query(), 1, None, pool, move |_| {
                    if first.swap(false, Ordering::SeqCst) {
                        started.send(()).unwrap();
                        gate.lock().unwrap().recv_timeout(Duration::from_secs(10)).unwrap();
                    }
                    true
                },
            ));
            assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            wait_for_worker(entered).await;
            drop(pending);
            drop(index);
            assert!(weak.upgrade().is_some(), "running worker must retain its exact bytes");
            assert!(!cx.is_cancel_requested());
            release.send(()).unwrap();
            weak
        });
        assert!(weak.upgrade().is_none(), "pool drain must release abandoned query owners");
    }
}
