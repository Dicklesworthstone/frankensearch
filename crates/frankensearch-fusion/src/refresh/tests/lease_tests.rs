//! Deterministic ownership and publication tests for the actual refresh cycle.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll, Wake, Waker};

use super::*;

const DIMENSION: usize = 16;

enum BatchBehavior {
    Pending,
    Gated(Arc<AtomicBool>),
    InvalidPartial,
    Failed,
    Cancelled,
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
                    let _ = output.pop();
                    // One job gets an embedding retry; the remaining record
                    // passes bind-time validation but fails the writer's
                    // producer/space join, forcing the rebuild failure path.
                    for bound in &mut output {
                        bound.identity = StubEmbedder::new("foreign-space", DIMENSION)
                            .identity_bundle()
                            .clone();
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

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn poll_once(future: Pin<&mut impl Future<Output = SearchResult<usize>>>) -> Poll<SearchResult<usize>> {
    let waker = Waker::from(Arc::new(NoopWake));
    future.poll(&mut Context::from_waker(&waker))
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
        assert_eq!(resumed.run_cycle(&cx).await.expect("publish restored work"), 2);
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
        let error = worker.run_cycle(&cx).await.expect_err("different cache path");
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
            Arc::new(ControlledEmbedder::new("stub-fast", BatchBehavior::Cancelled))
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
            assert!(matches!(error, SearchError::Cancelled { ref phase, ref reason, .. }
                if phase == "lease-test.bound" && reason == "controlled provider cancellation"));
            assert_eq!(queue.pending_count(), 1);
            assert_eq!(queue.in_flight_count(), 0);
            assert_eq!(queue.outstanding_count(), 1);
        }
        assert!(Arc::ptr_eq(&cache.current(), &retained));
        assert_eq!(std::fs::read(&fast_path).expect("unchanged seed"), fast_before);
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
