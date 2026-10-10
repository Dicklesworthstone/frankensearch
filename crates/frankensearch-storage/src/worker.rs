//! Caller-owned, continuously supervised embedding workers.
//!
//! `StorageBackedJobRunner::run_worker` is an idle-bounded helper. This service
//! deliberately stays alive through empty polls and delayed retries, and checks
//! for abandoned leases throughout its lifetime, including while work is busy.
//! It constructs the queue and runner together so recovery cannot accidentally
//! target a different database or queue. No runtime, task, or thread is created.
//!
//! ```no_run
//! use std::sync::atomic::AtomicBool;
//! use asupersync::Cx;
//! use frankensearch_core::SearchResult;
//! use frankensearch_storage::{IngestRequest, PersistentEmbeddingWorker};
//!
//! async fn serve(cx: &Cx, worker: &PersistentEmbeddingWorker, stop: &AtomicBool)
//!     -> SearchResult<()>
//! {
//!     worker.runner().ingest(IngestRequest::new("doc", "complete document text"))?;
//!     worker.run(cx, "embedding-worker", stop).await?;
//!     Ok(())
//! }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use asupersync::Cx;
use asupersync::types::Time;
use frankensearch_core::{Canonicalizer, Embedder, SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::{
    BatchProcessResult, EmbeddingVectorSink, JobQueueConfig, PersistentJobQueue, Storage,
    StorageBackedJobRunner, WorkerReport,
};

/// Scheduling policy for a persistent worker, independent of the runner's
/// idle-bounded helper settings. All durations use the caller's runtime clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerServiceConfig {
    /// Maximum delay between empty-queue polls. Must be positive.
    pub idle_poll_interval: Duration,
    /// Interval between abandoned-lease scans, even under continuous load.
    /// Must be positive. Lease expiry itself remains the queue's wall-clock
    /// contract; this interval only schedules scans, never decides ownership.
    pub recovery_interval: Duration,
}

impl Default for WorkerServiceConfig {
    fn default() -> Self {
        Self {
            idle_poll_interval: Duration::from_millis(25),
            recovery_interval: Duration::from_secs(1),
        }
    }
}

impl WorkerServiceConfig {
    /// Reject zero intervals and durations outside the runtime clock domain.
    pub fn validate(self) -> SearchResult<()> {
        positive_nanos(self.idle_poll_interval, "worker.idle_poll_interval")?;
        positive_nanos(self.recovery_interval, "worker.recovery_interval")?;
        Ok(())
    }
}

/// Progress observed by one service invocation. Queue/catalog transitions are
/// performed by the existing fenced runner and recovery transactions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistentWorkerReport {
    /// Batch outcomes, with the original startup-recovery count kept distinct.
    pub work: WorkerReport,
    /// Successful lease scans, including the startup scan.
    pub recovery_passes: usize,
    /// Rows recovered after startup: retried, terminally failed, or superseded.
    pub reclaimed_after_startup: usize,
    /// Total empty polls, not just the final consecutive idle streak.
    pub idle_polls: usize,
}

/// A runner and its exact queue under one persistent service owner.
///
/// Use `runner().ingest(...)` to submit work and `run(...)` inside the caller's
/// asupersync region. Sharing these handles preserves the storage crate's
/// single-connection ownership contract; this is not a cross-process queue.
#[derive(Debug)]
pub struct PersistentEmbeddingWorker {
    runner: StorageBackedJobRunner,
    queue: Arc<PersistentJobQueue>,
    config: WorkerServiceConfig,
}

impl PersistentEmbeddingWorker {
    /// Construct a service and its runner over exactly one queue/connection.
    /// A zero claim batch is rejected instead of creating a permanently idle
    /// worker that can never make progress. No database work is performed here.
    // The runner's shared handles intentionally remain on one storage owner.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(
        storage: Arc<Storage>,
        queue_config: JobQueueConfig,
        canonicalizer: Arc<dyn Canonicalizer>,
        fast_embedder: Arc<dyn Embedder>,
        vector_sink: Arc<dyn EmbeddingVectorSink>,
    ) -> SearchResult<Self> {
        if queue_config.batch_size == 0 {
            return Err(invalid_config("queue.batch_size", "must be positive"));
        }
        let queue = Arc::new(PersistentJobQueue::new(Arc::clone(&storage), queue_config));
        let runner = StorageBackedJobRunner::new(
            storage,
            Arc::clone(&queue),
            canonicalizer,
            fast_embedder,
            vector_sink,
        );
        Ok(Self {
            runner,
            queue,
            config: WorkerServiceConfig::default(),
        })
    }

    /// Configure the quality tier on the same ingestion and processing runner.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.runner = self.runner.with_quality_embedder(embedder);
        self
    }

    /// Replace the service's polling/recovery policy after validation.
    pub fn with_config(mut self, config: WorkerServiceConfig) -> SearchResult<Self> {
        config.validate()?;
        self.config = config;
        Ok(self)
    }

    /// Bound one processing batch; the queue's own batch cap still applies.
    pub fn with_batch_size(mut self, batch_size: usize) -> SearchResult<Self> {
        if batch_size == 0 {
            return Err(invalid_config("worker.batch_size", "must be positive"));
        }
        let mut config = *self.runner.config();
        config.process_batch_size = batch_size;
        self.runner = self.runner.with_config(config);
        Ok(self)
    }

    /// The exact runner used by this service, including its atomic ingest API.
    #[must_use]
    pub const fn runner(&self) -> &StorageBackedJobRunner {
        &self.runner
    }

    /// The exact queue reclaimed and processed by this service.
    #[must_use]
    pub fn queue(&self) -> &PersistentJobQueue {
        &self.queue
    }

    /// Run until graceful shutdown or structured cancellation.
    ///
    /// Unlike the idle-bounded helper, an empty poll never means completion:
    /// delayed retries and leases which expire after startup remain recoverable.
    /// Successful batches yield cooperatively, even for immediately-ready
    /// embedders, so continuous arrivals cannot monopolize a runtime worker.
    ///
    /// Graceful shutdown is observed between batches and before recovery. It
    /// does not interrupt a batch already in progress. Cancellation and storage
    /// errors propagate; uncompleted claims retain the queue's recovery lease.
    /// No database transaction is held over inference or an await point.
    #[allow(clippy::future_not_send)]
    pub async fn run(
        &self,
        cx: &Cx,
        worker_id: &str,
        shutdown: &AtomicBool,
    ) -> SearchResult<PersistentWorkerReport> {
        if worker_id.trim().is_empty() {
            return Err(invalid_config("worker_id", "must not be empty"));
        }
        let mut report = PersistentWorkerReport::default();
        let mut schedule = RecoverySchedule::new(self.config.recovery_interval)?;
        while !should_stop(cx, shutdown)? {
            self.recover_if_due(cx.now(), &mut schedule, &mut report)?;
            // A synchronous scan can take time; do not start another batch if
            // shutdown or cancellation arrived while recovery was committing.
            if should_stop(cx, shutdown)? {
                break;
            }
            let batch = self.runner.process_batch(cx, worker_id).await?;
            record_batch(&mut report, batch);
            if should_stop(cx, shutdown)? {
                break;
            }
            if batch.jobs_claimed == 0 {
                let wait = self
                    .config
                    .idle_poll_interval
                    .min(schedule.remaining(cx.now()));
                if !wait.is_zero() {
                    asupersync::time::sleep(cx.now(), wait).await;
                } else {
                    asupersync::runtime::yield_now().await;
                }
            } else {
                asupersync::runtime::yield_now().await;
            }
        }
        Ok(report)
    }

    fn recover_if_due(
        &self,
        now: Time,
        schedule: &mut RecoverySchedule,
        report: &mut PersistentWorkerReport,
    ) -> SearchResult<()> {
        if !schedule.due(now) {
            return Ok(());
        }
        let reclaimed = self.queue.reclaim_stale_jobs()?;
        // Advance the schedule and counters only after a successful transaction.
        schedule.last = Some(now);
        if report.recovery_passes == 0 {
            report.work.reclaimed_on_startup = reclaimed;
        } else {
            report.reclaimed_after_startup =
                report.reclaimed_after_startup.saturating_add(reclaimed);
        }
        report.recovery_passes = report.recovery_passes.saturating_add(1);
        self.runner.metrics().total_reclaimed.fetch_add(
            u64::try_from(reclaimed).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(())
    }
}

#[derive(Debug)]
struct RecoverySchedule {
    last: Option<Time>,
    interval_nanos: u64,
}

impl RecoverySchedule {
    fn new(interval: Duration) -> SearchResult<Self> {
        Ok(Self {
            last: None,
            interval_nanos: positive_nanos(interval, "worker.recovery_interval")?,
        })
    }

    fn due(&self, now: Time) -> bool {
        self.remaining(now).is_zero()
    }

    fn remaining(&self, now: Time) -> Duration {
        let Some(last) = self.last else {
            return Duration::ZERO;
        };
        // Rebase on clock regression rather than postponing recovery until an
        // old epoch is reached again. Queue lease timestamps remain untouched.
        if now < last {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.interval_nanos.saturating_sub(now.duration_since(last)))
    }
}

fn record_batch(report: &mut PersistentWorkerReport, batch: BatchProcessResult) {
    if batch.jobs_claimed == 0 {
        report.idle_polls = report.idle_polls.saturating_add(1);
        report.work.idle_cycles = report.work.idle_cycles.saturating_add(1);
        return;
    }
    report.work.idle_cycles = 0;
    report.work.batches_processed = report.work.batches_processed.saturating_add(1);
    report.work.jobs_completed = report.work.jobs_completed.saturating_add(batch.jobs_completed);
    report.work.jobs_failed = report.work.jobs_failed.saturating_add(batch.jobs_failed);
    report.work.jobs_skipped = report.work.jobs_skipped.saturating_add(batch.jobs_skipped);
    report.work.jobs_suppressed = report.work.jobs_suppressed.saturating_add(batch.jobs_suppressed);
    report.work.terminal_failures_encountered = report
        .work
        .terminal_failures_encountered
        .saturating_add(batch.terminal_failures);
}

fn should_stop(cx: &Cx, shutdown: &AtomicBool) -> SearchResult<bool> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.worker_service".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(shutdown.load(Ordering::Acquire))
}

fn positive_nanos(duration: Duration, field: &str) -> SearchResult<u64> {
    let nanos = u64::try_from(duration.as_nanos())
        .map_err(|_| invalid_config(field, "must fit the runtime's u64 nanosecond domain"))?;
    if nanos == 0 {
        return Err(invalid_config(field, "must be positive"));
    }
    Ok(nanos)
}

fn invalid_config(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: "worker service configuration".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::future::{Future, poll_fn};
    use std::sync::atomic::AtomicUsize;
    use std::task::Poll;

    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use frankensearch_core::EmbeddingIdentityBundleV1;
    use fsqlite_types::value::SqliteValue;

    use super::*;
    use crate::{ClaimOutcome, InMemoryVectorSink, IngestRequest};

    struct Probe {
        identity: EmbeddingIdentityBundleV1,
        failures: AtomicUsize,
    }

    impl Embedder for Probe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                if self
                    .failures
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1))
                    .is_ok()
                {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(),
                        source: std::io::Error::other("transient test failure").into(),
                    });
                }
                Ok(vec![0.6, 0.8])
            })
        }

        fn dimension(&self) -> usize {
            2
        }

        fn id(&self) -> &str {
            "worker-probe"
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    struct Fixture {
        worker: PersistentEmbeddingWorker,
        storage: Arc<Storage>,
        sink: Arc<InMemoryVectorSink>,
        probe: Arc<Probe>,
    }

    fn fixture(queue_config: JobQueueConfig) -> Fixture {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let sink = Arc::new(InMemoryVectorSink::default());
        let probe = Arc::new(Probe {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("worker-probe", 2),
            failures: AtomicUsize::new(0),
        });
        let worker = PersistentEmbeddingWorker::new(
            Arc::clone(&storage),
            queue_config,
            Arc::new(DefaultCanonicalizer::default()),
            probe.clone(),
            sink.clone(),
        )
        .unwrap()
        .with_batch_size(1)
        .unwrap()
        .with_config(WorkerServiceConfig {
            idle_poll_interval: Duration::from_millis(1),
            recovery_interval: Duration::from_millis(10),
        })
        .unwrap();
        Fixture { worker, storage, sink, probe }
    }

    fn expire_claims(storage: &Storage) {
        storage
            .connection()
            .execute_sync("UPDATE embedding_jobs SET started_at = 0 WHERE status = 'processing';")
            .unwrap();
    }

    #[test]
    fn service_rejects_non_progressing_configuration() {
        for bad in [Duration::ZERO, Duration::MAX] {
            assert!(WorkerServiceConfig {
                idle_poll_interval: bad,
                ..WorkerServiceConfig::default()
            }.validate().is_err());
            assert!(WorkerServiceConfig {
                recovery_interval: bad,
                ..WorkerServiceConfig::default()
            }.validate().is_err());
        }
        let f = fixture(JobQueueConfig::default());
        assert!(f.worker.with_batch_size(0).is_err());
        let f = fixture(JobQueueConfig::default());
        assert!(PersistentEmbeddingWorker::new(
            f.storage,
            JobQueueConfig { batch_size: 0, ..JobQueueConfig::default() },
            Arc::new(DefaultCanonicalizer::default()),
            f.probe,
            f.sink,
        ).is_err());
    }

    #[test]
    fn recovery_schedule_uses_runtime_time_and_rebases_regression() {
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        assert!(schedule.due(Time::from_millis(100)));
        schedule.last = Some(Time::from_millis(100));
        assert_eq!(schedule.remaining(Time::from_millis(104)), Duration::from_millis(6));
        assert!(!schedule.due(Time::from_millis(109)));
        assert!(schedule.due(Time::from_millis(110)));
        assert!(schedule.due(Time::from_millis(99)));
        schedule.last = Some(Time::from_nanos(u64::MAX - 2));
        assert_eq!(schedule.remaining(Time::MAX), Duration::from_nanos(9_999_998));
    }

    #[test]
    fn late_expiring_claim_is_recovered_and_processed_without_restarting() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker.runner().ingest(IngestRequest::new("late", "durable input")).unwrap();
            let old = f.worker.queue().claim_batch("old-owner", 1).unwrap().pop().unwrap();
            let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
            let mut report = PersistentWorkerReport::default();
            f.worker.recover_if_due(Time::ZERO, &mut schedule, &mut report).unwrap();
            assert_eq!(report.work.reclaimed_on_startup, 0);
            expire_claims(&f.storage);
            f.worker.recover_if_due(Time::from_millis(9), &mut schedule, &mut report).unwrap();
            assert_eq!(f.worker.queue().queue_depth().unwrap().processing, 1);
            f.worker.recover_if_due(Time::from_millis(10), &mut schedule, &mut report).unwrap();
            assert_eq!(report.recovery_passes, 2);
            assert_eq!(report.reclaimed_after_startup, 1);
            assert_eq!(f.worker.runner().metrics().snapshot().total_reclaimed, 1);
            let batch = f.worker.runner().process_batch(&cx, "new-owner").await.unwrap();
            assert_eq!(batch.jobs_completed, 1);
            assert_eq!(f.sink.entries().len(), 1);
            assert_eq!(
                f.worker.queue().complete(&old, &old.content_hash.unwrap()).unwrap(),
                ClaimOutcome::LostClaim
            );
        });
    }

    #[test]
    fn ongoing_recovery_honors_the_persisted_crash_retry_budget() {
        let f = fixture(JobQueueConfig { max_retries: 0, ..JobQueueConfig::default() });
        f.worker.runner().ingest(IngestRequest::new("exhausted", "input")).unwrap();
        assert_eq!(f.worker.queue().claim_batch("old", 1).unwrap().len(), 1);
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        let mut report = PersistentWorkerReport::default();
        f.worker.recover_if_due(Time::ZERO, &mut schedule, &mut report).unwrap();
        expire_claims(&f.storage);
        f.worker.recover_if_due(Time::from_millis(10), &mut schedule, &mut report).unwrap();
        assert_eq!(report.reclaimed_after_startup, 1);
        assert_eq!(f.worker.queue().queue_depth().unwrap().failed, 1);
        assert_eq!(f.storage.count_by_status("worker-probe").unwrap().failed, 1);
        assert!(f.worker.queue().claim_batch("next", 1).unwrap().is_empty());
    }

    #[test]
    fn failed_recovery_does_not_advance_schedule_or_metrics() {
        let f = fixture(JobQueueConfig::default());
        f.storage.connection().execute_sync(
            "ALTER TABLE embedding_jobs RENAME COLUMN started_at TO unavailable_started_at;",
        ).unwrap();
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        let mut report = PersistentWorkerReport::default();
        let metrics = f.worker.runner().metrics().snapshot();
        assert!(f.worker.recover_if_due(Time::ZERO, &mut schedule, &mut report).is_err());
        assert!(schedule.last.is_none());
        assert_eq!(report, PersistentWorkerReport::default());
        assert_eq!(f.worker.runner().metrics().snapshot(), metrics);
        f.storage.connection().execute_sync(
            "ALTER TABLE embedding_jobs RENAME COLUMN unavailable_started_at TO started_at;",
        ).unwrap();
        f.worker.recover_if_due(Time::ZERO, &mut schedule, &mut report).unwrap();
        assert_eq!(report.recovery_passes, 1);
    }

    #[test]
    fn stopped_or_cancelled_service_does_not_reclaim_or_claim_jobs() {
        for cancel in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let f = fixture(JobQueueConfig::default());
                f.worker.runner().ingest(IngestRequest::new("stop", "input")).unwrap();
                assert_eq!(f.worker.queue().claim_batch("old", 1).unwrap().len(), 1);
                expire_claims(&f.storage);
                let before = f.worker.queue().queue_depth().unwrap();
                let metrics = f.worker.queue().metrics().snapshot();
                if cancel {
                    cx.cancel_with(asupersync::CancelKind::User, Some("service cancellation"));
                }
                let result = f.worker.run(&cx, "new", &AtomicBool::new(true)).await;
                if cancel {
                    assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                } else {
                    assert_eq!(result.unwrap(), PersistentWorkerReport::default());
                }
                assert_eq!(f.worker.queue().queue_depth().unwrap(), before);
                assert_eq!(f.worker.queue().metrics().snapshot(), metrics);
            });
        }
    }

    #[test]
    fn busy_service_yields_and_observes_shutdown_between_batches() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            for id in ["first", "second"] {
                f.worker.runner().ingest(IngestRequest::new(id, "input")).unwrap();
            }
            let shutdown = AtomicBool::new(false);
            let mut future = std::pin::pin!(f.worker.run(&cx, "busy", &shutdown));
            let mut yields = 0;
            let report = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    yields += 1;
                    shutdown.store(true, Ordering::Release);
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            }).await.unwrap();
            assert!(yields > 0, "immediately-ready embedders must still yield");
            assert_eq!(report.work.jobs_completed, 1);
            assert_eq!(f.sink.entries().len(), 1);
            assert_eq!(f.worker.queue().queue_depth().unwrap().pending, 1);
        });
    }

    #[test]
    fn persistent_service_waits_through_empty_polls_for_delayed_retries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                retry_base_delay_ms: 30_000,
                ..JobQueueConfig::default()
            });
            f.probe.failures.store(1, Ordering::SeqCst);
            f.worker.runner().ingest(IngestRequest::new("retry", "input")).unwrap();
            let shutdown = AtomicBool::new(false);
            let mut future = std::pin::pin!(f.worker.run(&cx, "retry-worker", &shutdown));
            let mut polls = 0;
            let report = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    polls += 1;
                    // First Pending is the busy-batch yield. The second can
                    // occur only after the delayed job produced an empty poll.
                    // Mature the persisted retry deterministically, not by a
                    // wall-clock sleep or by changing the production backoff.
                    if polls == 2 {
                        assert_eq!(f.worker.queue().queue_depth().unwrap().ready_pending, 0);
                        f.storage.connection().execute_with_params_sync(
                            "UPDATE embedding_jobs SET submitted_at = ?1 WHERE status = 'pending';",
                            &[SqliteValue::Integer(0)],
                        ).unwrap();
                    }
                    if !f.sink.entries().is_empty() {
                        shutdown.store(true, Ordering::Release);
                    }
                    assert!(polls < 10_000, "worker must eventually process the mature retry");
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            }).await.unwrap();
            assert!(report.idle_polls > 0);
            assert_eq!(report.work.jobs_failed, 1);
            assert_eq!(report.work.jobs_completed, 1);
            assert_eq!(report.work.terminal_failures_encountered, 0);
            assert_eq!(f.sink.entries().len(), 1);
        });
    }

    #[test]
    fn processing_storage_error_escapes_without_an_unbounded_restart_loop() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker.runner().ingest(IngestRequest::new("corrupt", "input")).unwrap();
            f.storage.connection().execute_sync(
                "UPDATE documents SET content_hash = X'01' WHERE doc_id = 'corrupt';",
            ).unwrap();
            let result = f.worker.run(&cx, "worker", &AtomicBool::new(false)).await;
            assert!(result.is_err());
            assert!(f.sink.entries().is_empty());
            let depth = f.worker.queue().queue_depth().unwrap();
            assert_eq!(depth.processing, 1, "failed batch retains its recoverable claim");
            assert_eq!(depth.failed + depth.completed + depth.pending, 0);
            assert_eq!(f.worker.queue().metrics().snapshot().total_retried, 0);
        });
    }
}
