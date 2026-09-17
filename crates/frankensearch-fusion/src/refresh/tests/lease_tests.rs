//! Deterministic ownership and publication tests for the actual refresh cycle.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll, Waker};

use super::*;

const DIMENSION: usize = 16;

enum BatchBehavior {
    Pending,
    Gated(Arc<AtomicBool>),
    InvalidPartial,
    Failed,
    Cancelled,
    Scripted(Arc<BatchScript>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchAction {
    Pass,
    Pending,
    Fail,
    Empty,
    MissingMiddle,
    Extra,
}

struct BatchScript {
    calls: Mutex<Vec<Vec<String>>>,
    actions: Vec<(usize, BatchAction)>,
}

impl BatchScript {
    fn shared(action: BatchAction, at: usize) -> Arc<Self> {
        Self::sequence(&[(at, action)])
    }

    fn sequence(actions: &[(usize, BatchAction)]) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            actions: actions.to_vec(),
        })
    }

    fn calls(&self) -> Vec<Vec<String>> {
        self.calls.lock().expect("batch log lock").clone()
    }

    fn record(&self, texts: &[&str]) -> BatchAction {
        let mut calls = self.calls.lock().expect("batch log lock");
        let index = calls.len();
        calls.push(texts.iter().map(|text| (*text).to_owned()).collect());
        drop(calls);
        self.actions
            .iter()
            .find_map(|(at, action)| (*at == index).then_some(*action))
            .unwrap_or(BatchAction::Pass)
    }
}

struct ControlledEmbedder {
    inner: StubEmbedder,
    behavior: BatchBehavior,
}

impl ControlledEmbedder {
    fn new(id: &'static str, behavior: BatchBehavior) -> Self {
        Self {
            inner: StubEmbedder::new(id, DIMENSION),
            behavior,
        }
    }
}

impl Embedder for ControlledEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            match &self.behavior {
                BatchBehavior::Scripted(script) => {
                    // record() releases the log mutex before any suspension.
                    match script.record(texts) {
                        BatchAction::Pass => self.inner.embed_batch_bound(cx, texts).await,
                        BatchAction::Pending => std::future::pending().await,
                        BatchAction::Fail => Err(SearchError::EmbeddingFailed {
                            model: "scripted-embedder".into(),
                            source: Box::new(std::io::Error::other("scripted batch failure")),
                        }),
                        BatchAction::Empty => Ok(Vec::new()),
                        BatchAction::MissingMiddle => {
                            let mut output = self.inner.embed_batch_bound(cx, texts).await?;
                            if !output.is_empty() {
                                let middle = output.len() / 2;
                                let _ = output.remove(middle);
                            }
                            Ok(output)
                        }
                        BatchAction::Extra => {
                            let mut output = self.inner.embed_batch_bound(cx, texts).await?;
                            output.extend(self.inner.embed_batch_bound(cx, texts).await?);
                            Ok(output)
                        }
                    }
                }
                BatchBehavior::Pending => std::future::pending().await,
                BatchBehavior::Gated(ready) => {
                    // Tests explicitly re-poll after opening the gate. No
                    // detached task, timer or scheduler race is involved.
                    std::future::poll_fn(|_| {
                        if ready.load(Ordering::SeqCst) {
                            Poll::Ready(())
                        } else {
                            Poll::Pending
                        }
                    })
                    .await;
                    self.inner.embed_batch_bound(cx, texts).await
                }
                BatchBehavior::InvalidPartial => {
                    let mut output = self.inner.embed_batch_bound(cx, texts).await?;
                    // Keep cardinality valid so this still exercises a per-row
                    // bind failure followed by a distinct rebuild failure.
                    for bound in &mut output {
                        bound.identity = StubEmbedder::new("foreign-space", DIMENSION)
                            .identity_bundle()
                            .clone();
                    }
                    if let Some(first) = output.first_mut() {
                        first.identity.storage.format.clear();
                    }
                    Ok(output)
                }
                BatchBehavior::Cancelled => Err(SearchError::Cancelled {
                    phase: "lease-test.bound".to_owned(),
                    reason: "controlled provider cancellation".to_owned(),
                }),
                BatchBehavior::Failed => Err(SearchError::EmbeddingFailed {
                    model: "controlled-embedder".into(),
                    source: Box::new(std::io::Error::other("controlled embedding failure")),
                }),
            }
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

fn poll_once<T>(future: Pin<&mut impl Future<Output = SearchResult<T>>>) -> Poll<SearchResult<T>> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}

fn assert_pending_drop_restores_work(quality_pending: bool) {
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let dir = temp_index_dir("leased-drop");
        let queue = make_queue(4);
        submit(&queue, "a", "older payload");
        submit(&queue, "b", "second payload");
        let cache = if quality_pending {
            make_cache_with_quality(&dir, DIMENSION, DIMENSION)
        } else {
            make_cache(&dir, DIMENSION)
        };
        let fast: Arc<dyn Embedder> = if quality_pending {
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION))
        } else {
            Arc::new(ControlledEmbedder::new("stub-fast", BatchBehavior::Pending))
        };
        let mut worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir).with_max_docs_per_cycle(2),
            queue.clone(),
            fast,
            cache.clone(),
        );
        if quality_pending {
            worker = worker.with_quality_embedder(Arc::new(ControlledEmbedder::new(
                "stub-quality",
                BatchBehavior::Pending,
            )));
        }

        let mut cycle = Box::pin(worker.run_cycle(&cx));
        assert!(poll_once(cycle.as_mut()).is_pending());
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.in_flight_count(), 2);
        assert_eq!(queue.outstanding_count(), 2);

        // Queue capacity includes suspended work. Newer pending content wins
        // over the old leased version when the suspended cycle is dropped.
        submit(&queue, "a", "newest payload");
        submit(&queue, "c", "third payload");
        let full = queue.submit(EmbeddingRequest {
            doc_id: "overflow".into(),
            text: "must back off".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        });
        assert!(matches!(full, Err(SearchError::QueueFull { .. })));
        assert_eq!(queue.outstanding_count(), 4);

        drop(cycle);
        assert_eq!(queue.in_flight_count(), 0);
        assert_eq!(queue.outstanding_count(), 3);
        let jobs = queue.drain_batch_up_to(4);
        let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(ids, ["b", "a", "c"]);
        assert_eq!(jobs[1].canonical_text, "newest payload");
        assert!(jobs.iter().all(|job| job.retry_count == 0));
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 0);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn dropped_fast_embedding_preserves_capacity_order_and_newer_updates() {
    assert_pending_drop_restores_work(false);
}

#[test]
fn dropped_quality_embedding_preserves_capacity_order_and_newer_updates() {
    assert_pending_drop_restores_work(true);
}

#[test]
fn restored_batch_can_be_published_by_the_next_cycle() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("leased-resume");
        let queue = make_queue(2);
        submit(&queue, "a", "first payload");
        submit(&queue, "b", "second payload");
        let cache = make_cache(&dir, DIMENSION);
        let suspended = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new("stub-fast", BatchBehavior::Pending)),
            cache.clone(),
        );
        let mut cycle = Box::pin(suspended.run_cycle(&cx));
        assert!(poll_once(cycle.as_mut()).is_pending());
        drop(cycle);

        let resumed = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION)),
            cache.clone(),
        );
        assert_eq!(
            resumed.run_cycle(&cx).await.expect("publish restored work"),
            2
        );
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.in_flight_count(), 0);
        assert_eq!(queue.outstanding_count(), 0);
        assert_eq!(cache.current().doc_count(), 2);
        let outcome = queue
            .submit(EmbeddingRequest {
                doc_id: "a".into(),
                text: "first payload".to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .expect("published hash lookup");
        assert_eq!(outcome, JobOutcome::SkippedUnchanged);
    });
}

#[test]
fn a_competing_cycle_cannot_drain_past_an_active_lease() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("leased-competing-cycle");
        let queue = make_queue(3);
        submit(&queue, "a", "first payload");
        submit(&queue, "b", "second payload");
        let cache = make_cache(&dir, DIMENSION);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new("stub-fast", BatchBehavior::Pending)),
            cache,
        );
        let mut first = Box::pin(worker.run_cycle(&cx));
        assert!(poll_once(first.as_mut()).is_pending());
        submit(&queue, "c", "new work");
        assert_eq!(worker.run_cycle(&cx).await.expect("competing cycle"), 0);
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.in_flight_count(), 2);
        drop(first);
        assert_eq!(queue.pending_count(), 3);
        assert_eq!(queue.in_flight_count(), 0);
    });
}

#[test]
fn cache_contract_refusal_does_not_acknowledge_or_spend_retries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let cache_dir = temp_index_dir("leased-cache-contract");
        let writer_dir = temp_index_dir("leased-writer-contract");
        let queue = make_queue(1);
        submit(&queue, "a", "unpublished payload");
        let cache = make_cache(&cache_dir, DIMENSION);
        let retained = cache.current();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&writer_dir),
            queue.clone(),
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION)),
            cache.clone(),
        );
        let error = worker
            .run_cycle(&cx)
            .await
            .expect_err("different cache path");
        assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
            if field == "index_cache.replace"));
        assert!(Arc::ptr_eq(&cache.current(), &retained));
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 0);
        assert_eq!(worker.metrics().rebuild_failures.load(Ordering::Relaxed), 1);

        // The disk candidate was built. A cache refusal is not a rollback of
        // those bytes, nor is it permission to acknowledge the queued work.
        assert_eq!(
            TwoTierIndex::open(&writer_dir, TwoTierConfig::default())
                .expect("completed disk candidate")
                .doc_count(),
            1
        );
        assert_eq!(queue.in_flight_count(), 0);
        let retry = queue.drain_batch();
        assert_eq!(retry.len(), 1);
        assert_eq!(retry[0].retry_count, 0);
        let outcome = queue
            .submit(EmbeddingRequest {
                doc_id: "a".into(),
                text: "unpublished payload".to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .expect("unacknowledged content remains eligible");
        assert_ne!(outcome, JobOutcome::SkippedUnchanged);
    });
}

#[test]
fn a_new_cache_snapshot_wins_over_a_suspended_refresh() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("leased-cache-race");
        let queue = make_queue(1);
        submit(&queue, "a", "first payload");
        let cache = make_cache(&dir, DIMENSION);
        let gate = Arc::new(AtomicBool::new(false));
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let seed_bytes = std::fs::read(&fast_path).expect("read seed before inference");
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Gated(gate.clone()),
            )),
            cache.clone(),
        );
        let mut cycle = Box::pin(worker.run_cycle(&cx));
        assert!(poll_once(cycle.as_mut()).is_pending());
        cache.reload().expect("independent cache replacement");
        let winner = cache.current();
        gate.store(true, Ordering::SeqCst);
        let error = cycle
            .await
            .expect_err("obsolete refresh must not replace winner");
        assert_invalid_config(&error, "refresh.cache_publication", "superseded");
        assert_eq!(
            std::fs::read(&fast_path).expect("obsolete cycle did not rebuild"),
            seed_bytes
        );
        assert!(Arc::ptr_eq(&cache.current(), &winner));
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].retry_count, 0);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn partial_embedding_and_rebuild_failure_retry_each_job_only_once() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("leased-partial-retry");
        let queue = make_queue(2);
        submit(&queue, "a", "first payload");
        submit(&queue, "b", "second payload");
        let cache = make_cache(&dir, DIMENSION);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::InvalidPartial,
            )),
            cache.clone(),
        );
        worker
            .run_cycle(&cx)
            .await
            .expect_err("producer join fails at rebuild");
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch();
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|job| job.retry_count == 1));
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 1);
        assert_eq!(worker.metrics().rebuild_failures.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn explicit_retry_exhaustion_is_not_resurrected_by_lease_drop() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("leased-exhaustion");
        let queue = Arc::new(EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 1,
                batch_size: 1,
                max_retries: 0,
            },
            Box::new(DefaultCanonicalizer::default()),
        ));
        submit(&queue, "a", "first payload");
        let cache = make_cache(&dir, DIMENSION);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new("stub-fast", BatchBehavior::Failed)),
            cache,
        );
        assert_eq!(
            worker
                .run_cycle(&cx)
                .await
                .expect("handled embedding failure"),
            0
        );
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.in_flight_count(), 0);
        assert_eq!(queue.outstanding_count(), 0);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 1);
    });
}

fn assert_provider_cancellation_retains_work(quality_cancelled: bool) {
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let dir = temp_index_dir("leased-provider-cancel");
        let queue = Arc::new(EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 1,
                batch_size: 1,
                max_retries: 0,
            },
            Box::new(DefaultCanonicalizer::default()),
        ));
        submit(&queue, "a", "cancelled payload");
        let cache = if quality_cancelled {
            make_cache_with_quality(&dir, DIMENSION, DIMENSION)
        } else {
            make_cache(&dir, DIMENSION)
        };
        let retained = cache.current();
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let fast_before = std::fs::read(&fast_path).expect("read seed");
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let quality_before =
            quality_cancelled.then(|| std::fs::read(&quality_path).expect("read quality seed"));
        let fast: Arc<dyn Embedder> = if quality_cancelled {
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION))
        } else {
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Cancelled,
            ))
        };
        let mut worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            fast,
            cache.clone(),
        );
        if quality_cancelled {
            worker = worker.with_quality_embedder(Arc::new(ControlledEmbedder::new(
                "stub-quality",
                BatchBehavior::Cancelled,
            )));
        }

        // Even with zero retry allowance, repeated cancellation must preserve
        // the job. Quality cancellation must not publish a fast-only downgrade.
        for _ in 0..3 {
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("propagate cancellation");
            assert!(
                matches!(error, SearchError::Cancelled { ref phase, ref reason, .. }
                if phase == "lease-test.bound" && reason == "controlled provider cancellation")
            );
            assert_eq!(queue.pending_count(), 1);
            assert_eq!(queue.in_flight_count(), 0);
            assert_eq!(queue.outstanding_count(), 1);
        }
        assert!(Arc::ptr_eq(&cache.current(), &retained));
        assert_eq!(
            std::fs::read(&fast_path).expect("unchanged seed"),
            fast_before
        );
        if let Some(before) = quality_before {
            assert_eq!(
                std::fs::read(&quality_path).expect("unchanged quality"),
                before
            );
        }
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 0);
        assert_eq!(worker.metrics().docs_embedded.load(Ordering::Relaxed), 0);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 0);
        assert_eq!(queue.drain_batch()[0].retry_count, 0);
    });
}

#[test]
fn fast_provider_cancellation_does_not_consume_retry_budget() {
    assert_provider_cancellation_retains_work(false);
}

#[test]
fn quality_provider_cancellation_does_not_publish_a_fast_only_replacement() {
    assert_provider_cancellation_retains_work(true);
}

fn queue_with_batches(capacity: usize, batch_size: usize, max_retries: u32) -> Arc<EmbeddingQueue> {
    Arc::new(EmbeddingQueue::new(
        EmbeddingQueueConfig {
            capacity,
            batch_size,
            max_retries,
        },
        Box::new(DefaultCanonicalizer::default()),
    ))
}

fn submit_documents(queue: &EmbeddingQueue, ids: &[&str]) {
    for id in ids {
        submit(queue, id, &format!("payload {id}"));
    }
}

fn batch_sizes(script: &BatchScript) -> Vec<usize> {
    script.calls().iter().map(Vec::len).collect()
}

#[test]
fn cycle_bounds_both_model_calls_without_splitting_publication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("bounded-refresh-both-tiers");
        let queue = queue_with_batches(5, 2, 3);
        submit_documents(&queue, &["a", "b", "c", "d", "e"]);
        let cache = make_cache_with_quality(&dir, DIMENSION, DIMENSION);
        let fast = BatchScript::shared(BatchAction::Pass, 0);
        let quality = BatchScript::shared(BatchAction::Pass, 0);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(fast.clone()),
            )),
            cache.clone(),
        )
        .with_quality_embedder(Arc::new(ControlledEmbedder::new(
            "stub-quality",
            BatchBehavior::Scripted(quality.clone()),
        )));

        assert_eq!(worker.run_cycle(&cx).await.expect("bounded cycle"), 5);
        assert_eq!(batch_sizes(&fast), [2, 2, 1]);
        assert_eq!(fast.calls(), quality.calls());
        assert_eq!(
            fast.calls().concat(),
            [
                "payload a",
                "payload b",
                "payload c",
                "payload d",
                "payload e"
            ]
        );
        assert_eq!(cache.current().doc_count(), 5);
        assert_eq!(queue.outstanding_count(), 0);
        assert_eq!(queue.metrics().total_batches.load(Ordering::Relaxed), 1);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 5);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn inference_batch_limit_and_cycle_budget_are_independent() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("bounded-refresh-cycle-budget");
        let queue = queue_with_batches(5, 2, 3);
        submit_documents(&queue, &["a", "b", "c", "d", "e"]);
        let cache = make_cache(&dir, DIMENSION);
        let script = BatchScript::shared(BatchAction::Pass, 0);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir).with_max_docs_per_cycle(3),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(script.clone()),
            )),
            cache.clone(),
        );

        assert_eq!(worker.run_cycle(&cx).await.expect("bounded cycle"), 3);
        assert_eq!(batch_sizes(&script), [2, 1]);
        assert_eq!(cache.current().doc_count(), 3);
        let jobs = queue.drain_batch_up_to(5);
        let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(ids, ["d", "e"]);
        assert!(jobs.iter().all(|job| job.retry_count == 0));
    });
}

#[test]
fn zero_inference_batch_size_preserves_work_before_admission() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("bounded-refresh-zero-budget");
        let queue = queue_with_batches(1, 0, 3);
        submit_documents(&queue, &["a"]);
        let cache = make_cache(&dir, DIMENSION);
        let before = cache.current();
        let script = BatchScript::shared(BatchAction::Pass, 0);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(script.clone()),
            )),
            cache.clone(),
        );

        let error = worker.run_cycle(&cx).await.expect_err("zero batch size");
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, .. } if field == "embedding_queue.batch_size"
        ));
        assert!(script.calls().is_empty());
        assert!(Arc::ptr_eq(&before, &cache.current()));
        assert_eq!(queue.metrics().total_batches.load(Ordering::Relaxed), 0);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch_up_to(1);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].retry_count, 0);
    });
}

#[test]
fn failed_inference_chunk_does_not_retry_healthy_chunks() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("bounded-refresh-isolated-failure");
        let queue = queue_with_batches(6, 2, 3);
        submit_documents(&queue, &["a", "b", "c", "d", "e", "f"]);
        let cache = make_cache(&dir, DIMENSION);
        let script = BatchScript::shared(BatchAction::Fail, 1);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(script.clone()),
            )),
            cache.clone(),
        );

        assert_eq!(
            worker.run_cycle(&cx).await.expect("healthy chunks publish"),
            4
        );
        assert_eq!(batch_sizes(&script), [2, 2, 2]);
        assert_eq!(cache.current().doc_count(), 4);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 4);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 2);
        let jobs = queue.drain_batch_up_to(6);
        let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(ids, ["c", "d"]);
        assert!(jobs.iter().all(|job| job.retry_count == 1));
    });
}

fn assert_later_chunk_cancellation_restores_all_unpublished_work(quality_pending: bool) {
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let dir = temp_index_dir("bounded-refresh-later-cancellation");
        let queue = queue_with_batches(5, 2, 3);
        submit_documents(&queue, &["a", "b", "c", "d", "e"]);
        let cache = if quality_pending {
            make_cache_with_quality(&dir, DIMENSION, DIMENSION)
        } else {
            make_cache(&dir, DIMENSION)
        };
        let fast = BatchScript::shared(
            if quality_pending {
                BatchAction::Pass
            } else {
                BatchAction::Pending
            },
            1,
        );
        let quality = BatchScript::shared(BatchAction::Pending, 1);
        let mut worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(fast.clone()),
            )),
            cache.clone(),
        );
        if quality_pending {
            worker = worker.with_quality_embedder(Arc::new(ControlledEmbedder::new(
                "stub-quality",
                BatchBehavior::Scripted(quality.clone()),
            )));
        }

        let mut cycle = Box::pin(worker.run_cycle(&cx));
        assert!(poll_once(cycle.as_mut()).is_pending());
        assert_eq!(batch_sizes(&fast), [2, 2]);
        if quality_pending {
            assert_eq!(batch_sizes(&quality), [2, 2]);
        }
        assert_eq!(queue.in_flight_count(), 5);
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 0);
        drop(cycle);

        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch_up_to(5);
        let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c", "d", "e"]);
        assert!(jobs.iter().all(|job| job.retry_count == 0));
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn later_fast_chunk_cancellation_restores_earlier_successful_chunks() {
    assert_later_chunk_cancellation_restores_all_unpublished_work(false);
}

#[test]
fn later_quality_chunk_cancellation_restores_earlier_successful_chunks() {
    assert_later_chunk_cancellation_restores_all_unpublished_work(true);
}

fn assert_malformed_batch_is_not_published(
    action: BatchAction,
    malformed_quality: bool,
    max_retries: u32,
) {
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let dir = temp_index_dir("refresh-batch-cardinality");
        let queue = queue_with_batches(6, 2, max_retries);
        submit_documents(&queue, &["a", "b", "c", "d", "e", "f"]);
        let cache = if malformed_quality {
            make_cache_with_quality(&dir, DIMENSION, DIMENSION)
        } else {
            make_cache(&dir, DIMENSION)
        };
        let fast = BatchScript::shared(
            if malformed_quality {
                BatchAction::Pass
            } else {
                action
            },
            1,
        );
        let quality = BatchScript::shared(action, 1);
        let mut worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(fast.clone()),
            )),
            cache.clone(),
        );
        if malformed_quality {
            worker = worker.with_quality_embedder(Arc::new(ControlledEmbedder::new(
                "stub-quality",
                BatchBehavior::Scripted(quality.clone()),
            )));
        }

        assert_eq!(
            worker.run_cycle(&cx).await.expect("healthy chunks publish"),
            4
        );
        assert_eq!(batch_sizes(&fast), [2, 2, 2]);
        if malformed_quality {
            assert_eq!(fast.calls(), quality.calls());
        }
        assert_eq!(cache.current().doc_count(), 4);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 2);
        assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 1);
        assert_eq!(worker.metrics().rebuild_failures.load(Ordering::Relaxed), 0);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 4);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch_up_to(6);
        if max_retries == 0 {
            assert!(jobs.is_empty());
            assert_eq!(queue.metrics().total_failed.load(Ordering::Relaxed), 2);
        } else {
            let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
            assert_eq!(ids, ["c", "d"]);
            assert!(jobs.iter().all(|job| job.retry_count == 1));
        }
    });
}

#[test]
fn malformed_fast_batch_counts_do_not_publish_a_positional_prefix() {
    for action in [
        BatchAction::Empty,
        BatchAction::MissingMiddle,
        BatchAction::Extra,
    ] {
        assert_malformed_batch_is_not_published(action, false, 3);
    }
}

#[test]
fn malformed_quality_batch_counts_do_not_acknowledge_fast_only_success() {
    for action in [
        BatchAction::Empty,
        BatchAction::MissingMiddle,
        BatchAction::Extra,
    ] {
        assert_malformed_batch_is_not_published(action, true, 3);
    }
}

#[test]
fn malformed_batch_retry_exhaustion_does_not_resurrect_rejected_work() {
    for quality in [false, true] {
        assert_malformed_batch_is_not_published(BatchAction::MissingMiddle, quality, 0);
    }
}

#[test]
fn cancellation_after_a_malformed_chunk_preserves_each_retry_decision() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("refresh-malformed-then-cancelled");
        let queue = queue_with_batches(6, 2, 3);
        submit_documents(&queue, &["a", "b", "c", "d", "e", "f"]);
        let cache = make_cache(&dir, DIMENSION);
        let script =
            BatchScript::sequence(&[(0, BatchAction::MissingMiddle), (1, BatchAction::Pending)]);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(script.clone()),
            )),
            cache.clone(),
        );

        let mut cycle = Box::pin(worker.run_cycle(&cx));
        assert!(poll_once(cycle.as_mut()).is_pending());
        assert_eq!(batch_sizes(&script), [2, 2]);
        assert_eq!(queue.pending_count(), 2);
        assert_eq!(queue.in_flight_count(), 4);
        assert_eq!(queue.outstanding_count(), 6);
        drop(cycle);

        assert_eq!(queue.in_flight_count(), 0);
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 0);
        let jobs = queue.drain_batch_up_to(6);
        let ids: Vec<&str> = jobs.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(ids, ["c", "d", "e", "f", "a", "b"]);
        for job in &jobs {
            let expected_retries = u32::from(matches!(job.doc_id.as_str(), "a" | "b"));
            assert_eq!(job.retry_count, expected_retries);
        }
        assert_eq!(queue.metrics().total_retryable.load(Ordering::Relaxed), 2);
        assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 2);
        assert_eq!(worker.metrics().rebuild_failures.load(Ordering::Relaxed), 0);
    });
}

fn staged_jobs(ids: &[&str]) -> Vec<EmbeddingJob> {
    let queue = queue_with_batches(ids.len(), 2, 3);
    submit_documents(&queue, ids);
    queue.drain_batch_up_to(ids.len())
}

fn assert_staging_left_queue_untouched(queue: &EmbeddingQueue) {
    assert_eq!(queue.pending_count(), 1);
    assert_eq!(queue.in_flight_count(), 0);
    assert_eq!(queue.metrics().total_batches.load(Ordering::Relaxed), 0);
    assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 0);
    assert_eq!(queue.metrics().total_retryable.load(Ordering::Relaxed), 0);
    assert_eq!(queue.metrics().total_failed.load(Ordering::Relaxed), 0);
}

#[test]
fn later_staging_attempt_preserves_prior_artifact_paths_and_bytes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("staging-retained-attempts");
        let queue = queue_with_batches(8, 2, 3);
        submit_documents(&queue, &["still-pending"]);
        let cache = make_cache_with_quality(&dir, DIMENSION, DIMENSION);
        let canonical_fast = std::fs::read(dir.join(VECTOR_INDEX_FAST_FILENAME)).unwrap();
        let canonical_quality = std::fs::read(dir.join(VECTOR_INDEX_QUALITY_FILENAME)).unwrap();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION)),
            cache.clone(),
        )
        .with_quality_embedder(Arc::new(StubEmbedder::new("stub-quality", DIMENSION)));

        let first = worker
            .stage_identity_bound_generation(&cx, &staged_jobs(&["first"]))
            .await
            .expect("first stage");
        let first_fast = std::fs::read(&first.fast_path).unwrap();
        let first_quality_path = first.quality_path.as_ref().expect("first quality path");
        let first_quality = std::fs::read(first_quality_path).unwrap();
        let second = worker
            .stage_identity_bound_generation(&cx, &staged_jobs(&["second"]))
            .await
            .expect("second stage");

        assert_ne!(first.fast_path.parent(), second.fast_path.parent());
        assert_eq!(first.fast_path.parent(), first_quality_path.parent());
        assert_eq!(std::fs::read(&first.fast_path).unwrap(), first_fast);
        assert_eq!(std::fs::read(first_quality_path).unwrap(), first_quality);
        for (stage, expected) in [(&first, "first"), (&second, "second")] {
            for owner in [stage.fast_admitted_owner(), stage.quality_admitted_owner()] {
                let owner = owner.expect("retained admitted owner");
                assert_eq!(owner.record_count(), 1);
                assert_eq!(owner.row(0).expect("admitted row").doc_id(), expected);
            }
            assert!(matches!(
                worker.publish_staged_canonical(stage),
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "refresh.canonical_publication"
            ));
        }
        assert_eq!(cache.current().doc_count(), 0);
        assert_eq!(
            std::fs::read(dir.join(VECTOR_INDEX_FAST_FILENAME)).unwrap(),
            canonical_fast
        );
        assert_eq!(
            std::fs::read(dir.join(VECTOR_INDEX_QUALITY_FILENAME)).unwrap(),
            canonical_quality
        );
        assert_staging_left_queue_untouched(&queue);
    });
}

#[test]
fn staging_preserves_legacy_flat_artifacts_without_reusing_them() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("staging-legacy-flat");
        let queue = queue_with_batches(8, 2, 3);
        submit_documents(&queue, &["still-pending"]);
        let cache = make_cache(&dir, DIMENSION);
        let root = dir.join(STAGED_V2_DIR_NAME);
        std::fs::create_dir_all(&root).unwrap();
        let legacy_fast = root.join(VECTOR_INDEX_FAST_FILENAME);
        let legacy_quality = root.join(VECTOR_INDEX_QUALITY_FILENAME);
        std::fs::write(&legacy_fast, b"retained legacy fast artifact").unwrap();
        std::fs::write(&legacy_quality, b"retained legacy quality artifact").unwrap();
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(StubEmbedder::new("stub-fast", DIMENSION)),
            cache,
        );

        let staged = worker
            .stage_identity_bound_generation(&cx, &staged_jobs(&["new-document"]))
            .await
            .expect("stage does not need to touch legacy artifacts");
        let attempt = staged.fast_path.parent().expect("attempt directory");
        assert_eq!(attempt.parent(), Some(root.as_path()));
        assert_eq!(
            std::fs::read(legacy_fast).unwrap(),
            b"retained legacy fast artifact"
        );
        assert_eq!(
            std::fs::read(legacy_quality).unwrap(),
            b"retained legacy quality artifact"
        );
        assert_eq!(staged.index.doc_count(), 1);
        assert_staging_left_queue_untouched(&queue);
    });
}

#[test]
fn strict_staging_bounds_both_model_calls_without_consuming_queued_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("staging-bounded-inference");
        let queue = queue_with_batches(8, 2, 3);
        submit_documents(&queue, &["still-pending"]);
        let cache = make_cache_with_quality(&dir, DIMENSION, DIMENSION);
        let fast = BatchScript::sequence(&[]);
        let quality = BatchScript::sequence(&[]);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(fast.clone()),
            )),
            cache.clone(),
        )
        .with_quality_embedder(Arc::new(ControlledEmbedder::new(
            "stub-quality",
            BatchBehavior::Scripted(quality.clone()),
        )));

        let staged = worker
            .stage_identity_bound_generation(&cx, &staged_jobs(&["a", "b", "c", "d", "e"]))
            .await
            .expect("all strict chunks stage together");
        assert_eq!(batch_sizes(&fast), [2, 2, 1]);
        assert_eq!(fast.calls(), quality.calls());
        for owner in [
            staged.fast_admitted_owner(),
            staged.quality_admitted_owner(),
        ] {
            let owner = owner.expect("admitted tier");
            let mut ids = (0..owner.record_count())
                .map(|index| owner.row(index).expect("row").doc_id().to_owned())
                .collect::<Vec<_>>();
            ids.sort();
            assert_eq!(ids, ["a", "b", "c", "d", "e"]);
        }
        assert_eq!(cache.current().doc_count(), 0);
        assert_staging_left_queue_untouched(&queue);
    });
}

fn assert_dropped_staging_does_not_write(quality_pending: bool) {
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let dir = temp_index_dir("staging-dropped-later-chunk");
        let queue = queue_with_batches(8, 2, 3);
        submit_documents(&queue, &["still-pending"]);
        let cache = make_cache_with_quality(&dir, DIMENSION, DIMENSION);
        let fast = BatchScript::shared(
            if quality_pending {
                BatchAction::Pass
            } else {
                BatchAction::Pending
            },
            1,
        );
        let quality = BatchScript::shared(BatchAction::Pending, 1);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(fast.clone()),
            )),
            cache.clone(),
        )
        .with_quality_embedder(Arc::new(ControlledEmbedder::new(
            "stub-quality",
            BatchBehavior::Scripted(quality.clone()),
        )));
        let jobs = staged_jobs(&["a", "b", "c", "d", "e"]);
        let mut future = Box::pin(worker.stage_identity_bound_generation(&cx, &jobs));
        assert!(poll_once(future.as_mut()).is_pending());
        assert_eq!(batch_sizes(&fast), [2, 2]);
        assert_eq!(
            batch_sizes(&quality),
            if quality_pending { vec![2, 2] } else { vec![2] }
        );
        assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
        drop(future);
        assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
        assert_eq!(cache.current().doc_count(), 0);
        assert!(jobs.iter().all(|job| job.retry_count == 0));
        assert_staging_left_queue_untouched(&queue);
    });
}

#[test]
fn dropped_later_fast_staging_chunk_leaves_no_partial_generation() {
    assert_dropped_staging_does_not_write(false);
}

#[test]
fn dropped_later_quality_staging_chunk_leaves_no_partial_generation() {
    assert_dropped_staging_does_not_write(true);
}

#[test]
fn malformed_later_staging_chunk_never_writes_a_successful_prefix() {
    for malformed_quality in [false, true] {
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let dir = temp_index_dir("staging-malformed-later-chunk");
            let queue = queue_with_batches(8, 2, 3);
            submit_documents(&queue, &["still-pending"]);
            let cache = make_cache_with_quality(&dir, DIMENSION, DIMENSION);
            let fast = BatchScript::shared(
                if malformed_quality {
                    BatchAction::Pass
                } else {
                    BatchAction::MissingMiddle
                },
                1,
            );
            let quality = BatchScript::shared(BatchAction::MissingMiddle, 1);
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                Arc::new(ControlledEmbedder::new(
                    "stub-fast",
                    BatchBehavior::Scripted(fast.clone()),
                )),
                cache.clone(),
            )
            .with_quality_embedder(Arc::new(ControlledEmbedder::new(
                "stub-quality",
                BatchBehavior::Scripted(quality),
            )));
            let error = worker
                .stage_identity_bound_generation(&cx, &staged_jobs(&["a", "b", "c", "d", "e"]))
                .await
                .expect_err("strict count mismatch invalidates the whole attempt");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, .. } if field == "refresh.staged_embedding"
            ));
            assert_eq!(batch_sizes(&fast), [2, 2]);
            assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
            assert_eq!(cache.current().doc_count(), 0);
            assert_staging_left_queue_untouched(&queue);
        });
    }
}

#[test]
fn zero_staging_batch_size_fails_before_inference_or_staging_files() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_index_dir("staging-zero-batch");
        let queue = queue_with_batches(8, 0, 3);
        submit_documents(&queue, &["still-pending"]);
        let cache = make_cache(&dir, DIMENSION);
        let calls = BatchScript::sequence(&[]);
        let worker = RefreshWorker::new(
            RefreshWorkerConfig::new(&dir),
            queue.clone(),
            Arc::new(ControlledEmbedder::new(
                "stub-fast",
                BatchBehavior::Scripted(calls.clone()),
            )),
            cache,
        );
        let error = worker
            .stage_identity_bound_generation(&cx, &staged_jobs(&["a"]))
            .await
            .expect_err("zero batch size is invalid");
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, .. } if field == "embedding_queue.batch_size"
        ));
        assert!(calls.calls().is_empty());
        assert!(!dir.join(STAGED_V2_DIR_NAME).exists());
        assert_staging_left_queue_untouched(&queue);
    });
}
