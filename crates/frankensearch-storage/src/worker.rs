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
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::types::Time;
use frankensearch_core::{Canonicalizer, Embedder, SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::embedding_publication::PublicationAttempt;
use crate::{
    BatchProcessResult, ClaimOutcome, ClaimedJob, EmbeddingVectorSink, FailResult, JobQueueConfig,
    PersistentJobQueue, QueueDepth, Storage, StorageBackedJobRunner, WorkerReport,
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

/// Why a finite drain stopped.
///
/// A drained queue can still contain terminal failures: inspect
/// `remaining.failed` and the report rather than treating queue quiescence as
/// a claim that every document embedded successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerDrainOutcome {
    /// A queue read observed no pending (including delayed) or processing jobs.
    /// This is an observation, not an ingestion barrier: later submissions are
    /// new work and may require another drain.
    Drained {
        /// Progress made by this invocation.
        report: PersistentWorkerReport,
        /// Queue state at the quiescence observation, including terminal rows.
        remaining: QueueDepth,
    },
    /// Graceful shutdown was observed before starting more work.
    Stopped {
        /// Progress already committed by this invocation.
        report: PersistentWorkerReport,
    },
    /// The cooperative between-batch deadline was reached.
    DeadlineReached {
        /// Progress already committed; successful work is never relabelled.
        report: PersistentWorkerReport,
        /// Outstanding and terminal rows when the deadline was observed.
        remaining: QueueDepth,
    },
}

impl WorkerDrainOutcome {
    /// Return progress regardless of the reason the invocation stopped.
    #[must_use]
    pub const fn report(self) -> PersistentWorkerReport {
        match self {
            Self::Drained { report, .. }
            | Self::Stopped { report }
            | Self::DeadlineReached { report, .. } => report,
        }
    }
}

/// A runner and its exact queue under one persistent service owner.
///
/// Use `runner().ingest(...)` to submit work and `run(...)` inside the caller's
/// asupersync region. Sharing these handles preserves the storage crate's
/// single-connection ownership contract; this is not a cross-process queue.
pub struct PersistentEmbeddingWorker {
    runner: StorageBackedJobRunner,
    queue: Arc<PersistentJobQueue>,
    config: WorkerServiceConfig,
    storage: Arc<Storage>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    database_publication: bool,
}

impl std::fmt::Debug for PersistentEmbeddingWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentEmbeddingWorker")
            .field("runner", &self.runner)
            .field("config", &self.config)
            .field("database_publication", &self.database_publication)
            .finish_non_exhaustive()
    }
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
            Arc::clone(&storage),
            Arc::clone(&queue),
            canonicalizer,
            Arc::clone(&fast_embedder),
            vector_sink,
        );
        Ok(Self {
            runner,
            queue,
            config: WorkerServiceConfig::default(),
            storage,
            fast_embedder,
            quality_embedder: None,
            database_publication: false,
        })
    }

    /// Select atomic database publication for this worker's processing methods.
    ///
    /// `run`, `drain`, and `process_batch` publish vectors, catalog status, and
    /// queue completion in one transaction. They never call the external sink
    /// supplied at construction. The runner remains available for ingestion;
    /// calling `runner().process_batch(...)` directly still selects its external
    /// sink and deliberately bypasses this worker policy.
    ///
    /// Provider and missing-input rejections use the exact captured revision and
    /// the queue's bounded retry policy. Database errors and corrupt retained
    /// records propagate instead of consuming retry budgets. External FSVI/ANN
    /// artifacts are not published by this mode.
    #[must_use]
    pub const fn with_database_publication(mut self) -> Self {
        self.database_publication = true;
        self
    }

    /// Configure the quality tier on the same ingestion and processing runner.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(Arc::clone(&embedder));
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

    /// The runner holding this service's ingestion policy and shared metrics.
    /// Use the worker's `process_batch` to honor database-publication selection;
    /// the runner's direct processing methods retain their external-sink policy.
    #[must_use]
    pub const fn runner(&self) -> &StorageBackedJobRunner {
        &self.runner
    }

    /// The exact queue reclaimed and processed by this service.
    #[must_use]
    pub fn queue(&self) -> &PersistentJobQueue {
        &self.queue
    }

    /// Process one batch under the selected publication policy.
    ///
    /// Database publication uses the same claim limits, fenced failures, and
    /// metrics as the service loop. Every committed transition is counted before
    /// the next attempt, so a later storage error cannot erase earlier progress
    /// from the metrics. A failed publication transaction is propagated intact.
    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    pub async fn process_batch(
        &self,
        cx: &Cx,
        worker_id: &str,
    ) -> SearchResult<BatchProcessResult> {
        if !self.database_publication {
            return self.runner.process_batch(cx, worker_id).await;
        }
        worker_checkpoint(cx)?;
        if worker_id.trim().is_empty() {
            return Err(invalid_config("worker_id", "must not be empty"));
        }
        let started = Instant::now();
        let claims = self
            .queue
            .claim_batch(worker_id, self.runner.config().process_batch_size)?;
        let mut result = BatchProcessResult {
            jobs_claimed: claims.len(),
            ..BatchProcessResult::default()
        };
        self.runner.metrics().total_jobs_claimed.fetch_add(
            u64::try_from(claims.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        for claim in &claims {
            worker_checkpoint(cx)?;
            let document = match self.queue.check_claim(claim, None)? {
                ClaimOutcome::Applied(document) => document,
                ClaimOutcome::LostClaim | ClaimOutcome::Superseded => {
                    self.record_suppressed(&mut result);
                    continue;
                }
            };
            if frankensearch_core::is_hash_generation_id(&claim.embedder_id) {
                match self.queue.skip(
                    claim,
                    &document.content_hash,
                    "hash embeddings computed on-the-fly",
                )? {
                    ClaimOutcome::Applied(()) => {
                        result.jobs_skipped += 1;
                        self.runner
                            .metrics()
                            .total_jobs_skipped
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    ClaimOutcome::LostClaim | ClaimOutcome::Superseded => {
                        self.record_suppressed(&mut result);
                    }
                }
                continue;
            }
            let producer = if self.fast_embedder.id() == claim.embedder_id {
                Some(self.fast_embedder.as_ref())
            } else {
                self.quality_embedder
                    .as_deref()
                    .filter(|model| model.id() == claim.embedder_id)
            };
            // Even routing metadata can execute provider callbacks.
            worker_checkpoint(cx)?;
            let Some(producer) = producer else {
                self.record_rejected(
                    claim,
                    &document.content_hash,
                    &invalid_config(
                        "worker.embedder",
                        "no provider configured for the claimed embedder",
                    ),
                    &mut result,
                )?;
                continue;
            };
            let inference_started = Instant::now();
            let attempted = self
                .storage
                .attempt_claimed_publication(cx, claim, producer)
                .await;
            let elapsed = inference_started.elapsed();
            result.embed_time += elapsed;
            self.runner.metrics().total_embed_time_us.fetch_add(
                u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            match attempted? {
                ClaimOutcome::Applied(PublicationAttempt::Published) => {
                    // Publication already completed this claim in the same
                    // transaction as the vector. Never call queue.complete.
                    self.queue
                        .metrics()
                        .total_completed
                        .fetch_add(1, Ordering::Relaxed);
                    self.runner
                        .metrics()
                        .total_jobs_completed
                        .fetch_add(1, Ordering::Relaxed);
                    result.jobs_completed += 1;
                }
                ClaimOutcome::Applied(PublicationAttempt::Rejected {
                    document_hash,
                    error,
                }) => {
                    worker_checkpoint(cx)?;
                    self.record_rejected(claim, &document_hash, &error, &mut result)?;
                }
                ClaimOutcome::LostClaim | ClaimOutcome::Superseded => {
                    self.record_suppressed(&mut result);
                }
            }
        }
        result.total_time = started.elapsed();
        Ok(result)
    }

    fn record_rejected(
        &self,
        claim: &ClaimedJob,
        document_hash: &[u8; 32],
        error: &SearchError,
        result: &mut BatchProcessResult,
    ) -> SearchResult<()> {
        match self.queue.fail(claim, document_hash, &error.to_string())? {
            ClaimOutcome::Applied(failure) => {
                result.jobs_failed += 1;
                result.terminal_failures +=
                    usize::from(matches!(failure, FailResult::TerminalFailed { .. }));
                self.runner
                    .metrics()
                    .total_jobs_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            ClaimOutcome::LostClaim | ClaimOutcome::Superseded => self.record_suppressed(result),
        }
        Ok(())
    }

    fn record_suppressed(&self, result: &mut BatchProcessResult) {
        result.jobs_suppressed += 1;
        self.runner
            .metrics()
            .total_jobs_suppressed
            .fetch_add(1, Ordering::Relaxed);
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
        self.run_loop(cx, worker_id, shutdown, None)
            .await
            .map(WorkerDrainOutcome::report)
    }

    /// Process until no pending or processing jobs remain, shutdown is
    /// requested, or a cooperative runtime-clock budget expires.
    ///
    /// A delayed pending retry or a still-live claim is NOT an empty queue.
    /// This mode continues polling and recovering late-expiring leases through
    /// both, instead of returning after the first empty claim batch. Terminal
    /// failures do not prevent quiescence and are reported explicitly.
    ///
    /// The budget is checked before recovery, before each new batch, and after
    /// each batch. It does not preempt synchronous work or a provider await
    /// already in progress; provider deadlines still belong on the caller's
    /// `Cx`/embedder. Idle sleeps are capped by the remaining budget. A zero
    /// budget samples the remaining queue without claiming or recovering work.
    /// Storage errors and structured cancellation propagate, not masquerade as
    /// a successful drain or a deadline result.
    #[allow(clippy::future_not_send)]
    pub async fn drain(
        &self,
        cx: &Cx,
        worker_id: &str,
        shutdown: &AtomicBool,
        budget: Duration,
    ) -> SearchResult<WorkerDrainOutcome> {
        let deadline = drain_deadline(cx.now(), budget)?;
        self.run_loop(cx, worker_id, shutdown, Some(deadline)).await
    }

    #[allow(clippy::future_not_send)]
    async fn run_loop(
        &self,
        cx: &Cx,
        worker_id: &str,
        shutdown: &AtomicBool,
        deadline: Option<Time>,
    ) -> SearchResult<WorkerDrainOutcome> {
        if worker_id.trim().is_empty() {
            return Err(invalid_config("worker_id", "must not be empty"));
        }
        let mut report = PersistentWorkerReport::default();
        let mut schedule = RecoverySchedule::new(self.config.recovery_interval)?;
        loop {
            if should_stop(cx, shutdown)? {
                return Ok(WorkerDrainOutcome::Stopped { report });
            }
            if deadline.is_some_and(|limit| cx.now() >= limit) {
                let remaining = self.queue.queue_depth()?;
                if should_stop(cx, shutdown)? {
                    return Ok(WorkerDrainOutcome::Stopped { report });
                }
                return Ok(WorkerDrainOutcome::DeadlineReached { report, remaining });
            }
            self.recover_if_due(cx.now(), &mut schedule, &mut report)?;
            // A synchronous scan can take time; do not start another batch if
            // shutdown or cancellation arrived while recovery was committing.
            if should_stop(cx, shutdown)? {
                return Ok(WorkerDrainOutcome::Stopped { report });
            }
            if deadline.is_some_and(|limit| cx.now() >= limit) {
                let remaining = self.queue.queue_depth()?;
                if should_stop(cx, shutdown)? {
                    return Ok(WorkerDrainOutcome::Stopped { report });
                }
                return Ok(WorkerDrainOutcome::DeadlineReached { report, remaining });
            }
            let batch = self.process_batch(cx, worker_id).await?;
            record_batch(&mut report, batch);
            if should_stop(cx, shutdown)? {
                return Ok(WorkerDrainOutcome::Stopped { report });
            }
            if let Some(limit) = deadline {
                let remaining = self.queue.queue_depth()?;
                if should_stop(cx, shutdown)? {
                    return Ok(WorkerDrainOutcome::Stopped { report });
                }
                if remaining.pending == 0 && remaining.processing == 0 {
                    return Ok(WorkerDrainOutcome::Drained { report, remaining });
                }
                if cx.now() >= limit {
                    return Ok(WorkerDrainOutcome::DeadlineReached { report, remaining });
                }
            }
            if batch.jobs_claimed == 0 {
                let wait = idle_wait(self.config, &schedule, cx.now(), deadline);
                if !wait.is_zero() {
                    asupersync::time::sleep(cx.now(), wait).await;
                }
            }
            // Even a timer that is already ready must not turn idle polls
            // into a non-yielding loop that starves shutdown or new work.
            asupersync::runtime::yield_now().await;
        }
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

fn drain_deadline(now: Time, budget: Duration) -> SearchResult<Time> {
    let nanos = u64::try_from(budget.as_nanos())
        .ok()
        .and_then(|budget| now.as_nanos().checked_add(budget))
        .ok_or_else(|| invalid_config("worker.drain_budget", "deadline exceeds runtime clock"))?;
    Ok(Time::from_nanos(nanos))
}

fn idle_wait(
    config: WorkerServiceConfig,
    schedule: &RecoverySchedule,
    now: Time,
    deadline: Option<Time>,
) -> Duration {
    let wait = config.idle_poll_interval.min(schedule.remaining(now));
    deadline.map_or(wait, |limit| {
        wait.min(Duration::from_nanos(limit.duration_since(now)))
    })
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
    report.work.jobs_completed = report
        .work
        .jobs_completed
        .saturating_add(batch.jobs_completed);
    report.work.jobs_failed = report.work.jobs_failed.saturating_add(batch.jobs_failed);
    report.work.jobs_skipped = report.work.jobs_skipped.saturating_add(batch.jobs_skipped);
    report.work.jobs_suppressed = report
        .work
        .jobs_suppressed
        .saturating_add(batch.jobs_suppressed);
    report.work.terminal_failures_encountered = report
        .work
        .terminal_failures_encountered
        .saturating_add(batch.terminal_failures);
}

fn should_stop(cx: &Cx, shutdown: &AtomicBool) -> SearchResult<bool> {
    worker_checkpoint(cx)?;
    Ok(shutdown.load(Ordering::Acquire))
}

fn worker_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.worker_service".to_owned(),
        reason: error.to_string(),
    })
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
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;
    use std::task::Poll;

    use frankensearch_core::EmbeddingIdentityBundleV1;
    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use fsqlite_types::value::SqliteValue;

    use super::*;
    use crate::{ClaimOutcome, InMemoryVectorSink, IngestRequest};

    struct Probe {
        id: &'static str,
        identity: EmbeddingIdentityBundleV1,
        failures: AtomicUsize,
        cancel: AtomicBool,
        inputs: Mutex<Vec<String>>,
    }

    impl Embedder for Probe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.inputs.lock().unwrap().push(text.to_owned());
                if self.cancel.load(Ordering::SeqCst) {
                    return Err(SearchError::Cancelled {
                        phase: "worker.probe".to_owned(),
                        reason: "provider cancelled its operation".to_owned(),
                    });
                }
                if self
                    .failures
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                        left.checked_sub(1)
                    })
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

        fn id(&self) -> &'static str {
            self.id
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
            id: "worker-probe",
            identity: EmbeddingIdentityBundleV1::explicit_test_model("worker-probe", 2),
            failures: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            inputs: Mutex::new(Vec::new()),
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
        Fixture {
            worker,
            storage,
            sink,
            probe,
        }
    }

    fn expire_claims(storage: &Storage) {
        storage
            .connection()
            .execute_sync("UPDATE embedding_jobs SET started_at = 0 WHERE status = 'processing';")
            .unwrap();
    }

    #[test]
    fn database_publication_drains_both_tiers_without_calling_the_external_sink() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            let quality = Arc::new(Probe {
                id: "worker-quality",
                identity: EmbeddingIdentityBundleV1::explicit_test_model("worker-quality", 2),
                failures: AtomicUsize::new(0),
                cancel: AtomicBool::new(false),
                inputs: Mutex::new(Vec::new()),
            });
            let worker = f
                .worker
                .with_database_publication()
                .with_quality_embedder(quality.clone());
            let text = "searchable café 数据 complete content ".repeat(80);
            worker
                .runner()
                .ingest(IngestRequest::new("doc", &text))
                .unwrap();
            let expected = f.storage.get_document_content("doc").unwrap().unwrap();
            assert!(expected.chars().count() > 400);
            let outcome = worker
                .drain(
                    &cx,
                    "worker",
                    &AtomicBool::new(false),
                    Duration::from_secs(2),
                )
                .await
                .unwrap();
            let WorkerDrainOutcome::Drained { report, remaining } = outcome else {
                panic!("expected a complete database drain: {outcome:?}");
            };
            assert_eq!(report.work.jobs_completed, 2);
            assert_eq!(remaining.completed, 2);
            assert_eq!(
                remaining.failed + remaining.pending + remaining.processing,
                0
            );
            assert!(f.sink.entries().is_empty());
            assert_eq!(
                f.probe.inputs.lock().unwrap().as_slice(),
                std::slice::from_ref(&expected)
            );
            assert_eq!(quality.inputs.lock().unwrap().as_slice(), [expected]);
            for probe in [&f.probe, &quality] {
                let published = f
                    .storage
                    .get_published_embedding("doc", probe.id, &probe.identity)
                    .unwrap()
                    .unwrap();
                assert_eq!(published.embedding.values, vec![0.6, 0.8]);
            }
            assert_eq!(worker.queue().metrics().snapshot().total_completed, 2);
            assert_eq!(worker.runner().metrics().snapshot().total_jobs_completed, 2);
        });
    }

    #[test]
    fn database_publication_retries_provider_failures_without_waiting_for_a_lease() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                retry_base_delay_ms: 5,
                ..JobQueueConfig::default()
            });
            f.probe.failures.store(1, Ordering::SeqCst);
            let worker = f.worker.with_database_publication();
            worker
                .runner()
                .ingest(IngestRequest::new("retry", "complete input"))
                .unwrap();
            let outcome = worker
                .drain(
                    &cx,
                    "worker",
                    &AtomicBool::new(false),
                    Duration::from_secs(2),
                )
                .await
                .unwrap();
            let WorkerDrainOutcome::Drained { report, remaining } = outcome else {
                panic!("delayed retry must drain without lease expiry: {outcome:?}");
            };
            assert_eq!(report.work.jobs_completed, 1);
            assert_eq!(report.work.jobs_failed, 1);
            assert_eq!(report.work.terminal_failures_encountered, 0);
            assert_eq!(remaining.completed, 1);
            assert_eq!(
                remaining.failed + remaining.processing + remaining.pending,
                0
            );
            assert!(f.sink.entries().is_empty());
            let metrics = worker.queue().metrics().snapshot();
            assert_eq!(metrics.total_completed, 1);
            assert_eq!(metrics.total_retried, 1);
            assert_eq!(metrics.total_failed, 0);
            assert_eq!(worker.runner().metrics().snapshot().total_jobs_claimed, 2);
            assert!(
                f.storage
                    .get_published_embedding("retry", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[test]
    fn database_publication_terminal_failure_does_not_block_the_next_job() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            });
            f.probe.failures.store(1, Ordering::SeqCst);
            let worker = f
                .worker
                .with_database_publication()
                .with_batch_size(2)
                .unwrap();
            for id in ["reject", "good"] {
                worker.runner().ingest(IngestRequest::new(id, id)).unwrap();
            }
            f.storage
                .connection()
                .execute_sync("UPDATE embedding_jobs SET priority = 100 WHERE doc_id = 'reject';")
                .unwrap();
            let result = worker.process_batch(&cx, "worker").await.unwrap();
            assert_eq!(result.jobs_claimed, 2);
            assert_eq!(result.jobs_completed, 1);
            assert_eq!(result.jobs_failed, 1);
            assert_eq!(result.terminal_failures, 1);
            assert!(f.sink.entries().is_empty());
            assert!(
                f.storage
                    .get_published_embedding("reject", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_none()
            );
            assert!(
                f.storage
                    .get_published_embedding("good", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_some()
            );
            let queue = worker.queue().metrics().snapshot();
            assert_eq!(queue.total_failed, 1);
            assert_eq!(queue.total_completed, 1);
            let pipeline = worker.runner().metrics().snapshot();
            assert_eq!(pipeline.total_jobs_failed, 1);
            assert_eq!(pipeline.total_jobs_completed, 1);
        });
    }

    #[test]
    fn database_publication_missing_input_consumes_only_the_bounded_job_retry() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            });
            let worker = f.worker.with_database_publication();
            worker
                .runner()
                .ingest(IngestRequest::new("missing", "input"))
                .unwrap();
            f.storage
                .connection()
                .execute_sync("DELETE FROM document_contents;")
                .unwrap();
            let result = worker.process_batch(&cx, "worker").await.unwrap();
            assert_eq!(result.terminal_failures, 1);
            assert_eq!(result.jobs_completed, 0);
            assert!(f.probe.inputs.lock().unwrap().is_empty());
            assert!(f.sink.entries().is_empty());
            assert_eq!(worker.queue().queue_depth().unwrap().failed, 1);
        });
    }

    #[test]
    fn database_publication_failure_rolls_back_without_becoming_a_provider_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            });
            let worker = f.worker.with_database_publication();
            worker
                .runner()
                .ingest(IngestRequest::new("rollback", "input"))
                .unwrap();
            // The last catalog write fails, after vector and queue writes.
            f.storage.connection().execute_sync("ALTER TABLE embedding_status RENAME COLUMN embedded_at TO unavailable_embedded_at;").unwrap();
            assert!(worker.process_batch(&cx, "worker").await.is_err());
            let depth = worker.queue().queue_depth().unwrap();
            assert_eq!(depth.processing, 1);
            assert_eq!(depth.completed + depth.failed + depth.pending, 0);
            assert!(
                f.storage
                    .connection()
                    .query_sync("SELECT doc_id FROM published_embeddings;")
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(worker.queue().metrics().snapshot().total_completed, 0);
            assert_eq!(worker.queue().metrics().snapshot().total_failed, 0);
            assert_eq!(worker.runner().metrics().snapshot().total_jobs_failed, 0);
            assert_eq!(worker.runner().metrics().snapshot().total_jobs_completed, 0);
            assert!(f.sink.entries().is_empty());
        });
    }

    #[test]
    fn database_publication_counts_prior_commits_when_later_input_is_corrupt() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            let worker = f
                .worker
                .with_database_publication()
                .with_batch_size(2)
                .unwrap();
            for id in ["good", "corrupt"] {
                worker.runner().ingest(IngestRequest::new(id, id)).unwrap();
            }
            f.storage
                .connection()
                .execute_sync("UPDATE embedding_jobs SET priority = 100 WHERE doc_id = 'good';")
                .unwrap();
            f.storage.connection().execute_sync("UPDATE document_contents SET canonical_text = 'wrong bytes' WHERE doc_id = 'corrupt';").unwrap();
            assert!(worker.process_batch(&cx, "worker").await.is_err());
            let depth = worker.queue().queue_depth().unwrap();
            assert_eq!(depth.completed, 1);
            assert_eq!(depth.processing, 1);
            assert_eq!(depth.failed + depth.pending, 0);
            assert_eq!(worker.queue().metrics().snapshot().total_completed, 1);
            let metrics = worker.runner().metrics().snapshot();
            assert_eq!(metrics.total_jobs_completed, 1);
            assert_eq!(metrics.total_jobs_failed, 0);
            assert_eq!(metrics.total_jobs_claimed, 2);
            assert!(
                f.storage
                    .get_published_embedding("good", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_some()
            );
            assert!(f.sink.entries().is_empty());
        });
    }

    #[test]
    fn database_publication_propagates_typed_cancellation_without_consuming_retry_budget() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            });
            f.probe.cancel.store(true, Ordering::SeqCst);
            let worker = f.worker.with_database_publication();
            worker
                .runner()
                .ingest(IngestRequest::new("cancel", "input"))
                .unwrap();
            assert!(matches!(
                worker.process_batch(&cx, "worker").await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(worker.queue().queue_depth().unwrap().processing, 1);
            assert_eq!(worker.queue().metrics().snapshot().total_failed, 0);
            assert_eq!(worker.runner().metrics().snapshot().total_jobs_failed, 0);
            assert!(f.sink.entries().is_empty());
            assert!(
                f.storage
                    .get_published_embedding("cancel", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn external_worker_processing_keeps_its_explicit_sink_policy() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker
                .runner()
                .ingest(IngestRequest::new("external", "input"))
                .unwrap();
            assert_eq!(
                f.worker
                    .process_batch(&cx, "worker")
                    .await
                    .unwrap()
                    .jobs_completed,
                1
            );
            assert_eq!(f.sink.entries().len(), 1);
            assert!(
                f.storage
                    .get_published_embedding("external", f.probe.id, &f.probe.identity)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn service_rejects_non_progressing_configuration() {
        for bad in [Duration::ZERO, Duration::MAX] {
            assert!(
                WorkerServiceConfig {
                    idle_poll_interval: bad,
                    ..WorkerServiceConfig::default()
                }
                .validate()
                .is_err()
            );
            assert!(
                WorkerServiceConfig {
                    recovery_interval: bad,
                    ..WorkerServiceConfig::default()
                }
                .validate()
                .is_err()
            );
        }
        let f = fixture(JobQueueConfig::default());
        assert!(f.worker.with_batch_size(0).is_err());
        let f = fixture(JobQueueConfig::default());
        assert!(
            PersistentEmbeddingWorker::new(
                f.storage,
                JobQueueConfig {
                    batch_size: 0,
                    ..JobQueueConfig::default()
                },
                Arc::new(DefaultCanonicalizer::default()),
                f.probe,
                f.sink,
            )
            .is_err()
        );
    }

    #[test]
    fn recovery_schedule_uses_runtime_time_and_rebases_regression() {
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        assert!(schedule.due(Time::from_millis(100)));
        schedule.last = Some(Time::from_millis(100));
        assert_eq!(
            schedule.remaining(Time::from_millis(104)),
            Duration::from_millis(6)
        );
        assert!(!schedule.due(Time::from_millis(109)));
        assert!(schedule.due(Time::from_millis(110)));
        assert!(schedule.due(Time::from_millis(99)));
        schedule.last = Some(Time::from_nanos(u64::MAX - 2));
        assert_eq!(
            schedule.remaining(Time::MAX),
            Duration::from_nanos(9_999_998)
        );
    }

    #[test]
    fn late_expiring_claim_is_recovered_and_processed_without_restarting() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker
                .runner()
                .ingest(IngestRequest::new("late", "durable input"))
                .unwrap();
            let old = f
                .worker
                .queue()
                .claim_batch("old-owner", 1)
                .unwrap()
                .pop()
                .unwrap();
            let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
            let mut report = PersistentWorkerReport::default();
            f.worker
                .recover_if_due(Time::ZERO, &mut schedule, &mut report)
                .unwrap();
            assert_eq!(report.work.reclaimed_on_startup, 0);
            expire_claims(&f.storage);
            f.worker
                .recover_if_due(Time::from_millis(9), &mut schedule, &mut report)
                .unwrap();
            assert_eq!(f.worker.queue().queue_depth().unwrap().processing, 1);
            f.worker
                .recover_if_due(Time::from_millis(10), &mut schedule, &mut report)
                .unwrap();
            assert_eq!(report.recovery_passes, 2);
            assert_eq!(report.reclaimed_after_startup, 1);
            assert_eq!(f.worker.runner().metrics().snapshot().total_reclaimed, 1);
            let batch = f
                .worker
                .runner()
                .process_batch(&cx, "new-owner")
                .await
                .unwrap();
            assert_eq!(batch.jobs_completed, 1);
            assert_eq!(f.sink.entries().len(), 1);
            assert_eq!(
                f.worker
                    .queue()
                    .complete(&old, &old.content_hash.unwrap())
                    .unwrap(),
                ClaimOutcome::LostClaim
            );
        });
    }

    #[test]
    fn ongoing_recovery_honors_the_persisted_crash_retry_budget() {
        let f = fixture(JobQueueConfig {
            max_retries: 0,
            ..JobQueueConfig::default()
        });
        f.worker
            .runner()
            .ingest(IngestRequest::new("exhausted", "input"))
            .unwrap();
        assert_eq!(f.worker.queue().claim_batch("old", 1).unwrap().len(), 1);
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        let mut report = PersistentWorkerReport::default();
        f.worker
            .recover_if_due(Time::ZERO, &mut schedule, &mut report)
            .unwrap();
        expire_claims(&f.storage);
        f.worker
            .recover_if_due(Time::from_millis(10), &mut schedule, &mut report)
            .unwrap();
        assert_eq!(report.reclaimed_after_startup, 1);
        assert_eq!(f.worker.queue().queue_depth().unwrap().failed, 1);
        assert_eq!(f.storage.count_by_status("worker-probe").unwrap().failed, 1);
        assert!(f.worker.queue().claim_batch("next", 1).unwrap().is_empty());
    }

    #[test]
    fn failed_recovery_does_not_advance_schedule_or_metrics() {
        let f = fixture(JobQueueConfig::default());
        f.storage
            .connection()
            .execute_sync(
                "ALTER TABLE embedding_jobs RENAME COLUMN started_at TO unavailable_started_at;",
            )
            .unwrap();
        let mut schedule = RecoverySchedule::new(Duration::from_millis(10)).unwrap();
        let mut report = PersistentWorkerReport::default();
        let metrics = f.worker.runner().metrics().snapshot();
        assert!(
            f.worker
                .recover_if_due(Time::ZERO, &mut schedule, &mut report)
                .is_err()
        );
        assert!(schedule.last.is_none());
        assert_eq!(report, PersistentWorkerReport::default());
        assert_eq!(f.worker.runner().metrics().snapshot(), metrics);
        f.storage
            .connection()
            .execute_sync(
                "ALTER TABLE embedding_jobs RENAME COLUMN unavailable_started_at TO started_at;",
            )
            .unwrap();
        f.worker
            .recover_if_due(Time::ZERO, &mut schedule, &mut report)
            .unwrap();
        assert_eq!(report.recovery_passes, 1);
    }

    #[test]
    fn stopped_or_cancelled_service_does_not_reclaim_or_claim_jobs() {
        for cancel in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let f = fixture(JobQueueConfig::default());
                f.worker
                    .runner()
                    .ingest(IngestRequest::new("stop", "input"))
                    .unwrap();
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
                f.worker
                    .runner()
                    .ingest(IngestRequest::new(id, "input"))
                    .unwrap();
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
            })
            .await
            .unwrap();
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
            f.worker
                .runner()
                .ingest(IngestRequest::new("retry", "input"))
                .unwrap();
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
                    assert!(
                        polls < 10_000,
                        "worker must eventually process the mature retry"
                    );
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            })
            .await
            .unwrap();
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
            f.worker
                .runner()
                .ingest(IngestRequest::new("corrupt", "input"))
                .unwrap();
            f.storage
                .connection()
                .execute_sync("UPDATE documents SET content_hash = X'01' WHERE doc_id = 'corrupt';")
                .unwrap();
            let result = f.worker.run(&cx, "worker", &AtomicBool::new(false)).await;
            assert!(result.is_err());
            assert!(f.sink.entries().is_empty());
            let depth = f.worker.queue().queue_depth().unwrap();
            assert_eq!(
                depth.processing, 1,
                "failed batch retains its recoverable claim"
            );
            assert_eq!(depth.failed + depth.completed + depth.pending, 0);
            assert_eq!(f.worker.queue().metrics().snapshot().total_retried, 0);
        });
    }

    #[test]
    fn drain_reports_terminal_failures_without_leaving_runnable_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            });
            f.probe.failures.store(1, Ordering::SeqCst);
            for id in ["one", "two", "three"] {
                f.worker
                    .runner()
                    .ingest(IngestRequest::new(id, "input"))
                    .unwrap();
            }
            let outcome = f
                .worker
                .drain(
                    &cx,
                    "drain",
                    &AtomicBool::new(false),
                    Duration::from_secs(5),
                )
                .await
                .unwrap();
            let WorkerDrainOutcome::Drained { report, remaining } = outcome else {
                panic!("expected a fully processed queue, got {outcome:?}");
            };
            assert_eq!(report.work.jobs_completed, 2);
            assert_eq!(report.work.jobs_failed, 1);
            assert_eq!(report.work.terminal_failures_encountered, 1);
            assert_eq!(remaining.pending + remaining.processing, 0);
            assert_eq!(remaining.failed, 1, "drained is not an all-success claim");
            assert_eq!(f.sink.entries().len(), 2);
        });
    }

    #[test]
    fn zero_budget_samples_pending_and_expired_claims_without_mutation() {
        for claimed in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let f = fixture(JobQueueConfig::default());
                f.worker
                    .runner()
                    .ingest(IngestRequest::new("budget", "input"))
                    .unwrap();
                if claimed {
                    assert_eq!(f.worker.queue().claim_batch("old", 1).unwrap().len(), 1);
                    expire_claims(&f.storage);
                }
                let depth = f.worker.queue().queue_depth().unwrap();
                let metrics = f.worker.queue().metrics().snapshot();
                let outcome = f
                    .worker
                    .drain(&cx, "drain", &AtomicBool::new(false), Duration::ZERO)
                    .await
                    .unwrap();
                assert_eq!(
                    outcome,
                    WorkerDrainOutcome::DeadlineReached {
                        report: PersistentWorkerReport::default(),
                        remaining: depth,
                    }
                );
                assert_eq!(f.worker.queue().queue_depth().unwrap(), depth);
                assert_eq!(f.worker.queue().metrics().snapshot(), metrics);
                assert!(f.sink.entries().is_empty());
            });
        }
    }

    #[test]
    fn drain_waits_for_a_live_claim_instead_of_declaring_an_empty_queue() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker
                .runner()
                .ingest(IngestRequest::new("live", "input"))
                .unwrap();
            let claim = f
                .worker
                .queue()
                .claim_batch("other", 1)
                .unwrap()
                .pop()
                .unwrap();
            let shutdown = AtomicBool::new(false);
            let mut future =
                std::pin::pin!(
                    f.worker
                        .drain(&cx, "drain", &shutdown, Duration::from_secs(5),)
                );
            let outcome = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    shutdown.store(true, Ordering::Release);
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            })
            .await
            .unwrap();
            let WorkerDrainOutcome::Stopped { report } = outcome else {
                panic!("a live claim is outstanding work, got {outcome:?}");
            };
            assert_eq!(report.idle_polls, 1);
            assert_eq!(f.worker.queue().queue_depth().unwrap().processing, 1);
            assert_eq!(
                f.worker
                    .queue()
                    .complete(&claim, &claim.content_hash.unwrap())
                    .unwrap(),
                ClaimOutcome::Applied(())
            );
        });
    }

    #[test]
    fn stopped_drain_resumes_remaining_jobs_without_repeating_completed_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            for id in ["first", "second"] {
                f.worker
                    .runner()
                    .ingest(IngestRequest::new(id, "input"))
                    .unwrap();
            }
            let shutdown = AtomicBool::new(false);
            let mut future =
                std::pin::pin!(
                    f.worker
                        .drain(&cx, "drain", &shutdown, Duration::from_secs(5),)
                );
            let outcome = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    shutdown.store(true, Ordering::Release);
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            })
            .await
            .unwrap();
            assert!(matches!(outcome, WorkerDrainOutcome::Stopped { .. }));
            assert_eq!(outcome.report().work.jobs_completed, 1);
            shutdown.store(false, Ordering::Release);
            let resumed = f
                .worker
                .drain(&cx, "drain", &shutdown, Duration::from_secs(5))
                .await
                .unwrap();
            assert!(matches!(resumed, WorkerDrainOutcome::Drained { .. }));
            assert_eq!(resumed.report().work.jobs_completed, 1);
            assert_eq!(f.sink.entries().len(), 2);
            assert_eq!(f.worker.queue().queue_depth().unwrap().completed, 2);
        });
    }

    #[test]
    fn drain_waits_for_delayed_retry_then_finishes_without_a_stop_signal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig {
                retry_base_delay_ms: 30_000,
                ..JobQueueConfig::default()
            });
            f.probe.failures.store(1, Ordering::SeqCst);
            f.worker
                .runner()
                .ingest(IngestRequest::new("retry", "input"))
                .unwrap();
            let shutdown = AtomicBool::new(false);
            let mut future =
                std::pin::pin!(
                    f.worker
                        .drain(&cx, "drain", &shutdown, Duration::from_secs(5),)
                );
            let mut polls = 0;
            let outcome = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    polls += 1;
                    if polls == 2 {
                        assert_eq!(f.worker.queue().queue_depth().unwrap().ready_pending, 0);
                        f.storage.connection().execute_sync(
                            "UPDATE embedding_jobs SET submitted_at = 0 WHERE status = 'pending';",
                        ).unwrap();
                    }
                    assert!(polls < 10_000, "drain must finish after the retry matures");
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            })
            .await
            .unwrap();
            let WorkerDrainOutcome::Drained { report, remaining } = outcome else {
                panic!("expected quiescence after retry, got {outcome:?}");
            };
            assert!(report.idle_polls > 0);
            assert_eq!(report.work.jobs_failed, 1);
            assert_eq!(report.work.jobs_completed, 1);
            assert_eq!(
                remaining.pending + remaining.processing + remaining.failed,
                0
            );
            assert_eq!(f.sink.entries().len(), 1);
        });
    }

    #[test]
    fn drain_deadline_is_checked_and_caps_idle_sleep() {
        let config = WorkerServiceConfig {
            idle_poll_interval: Duration::from_secs(5),
            recovery_interval: Duration::from_secs(10),
        };
        let mut schedule = RecoverySchedule::new(config.recovery_interval).unwrap();
        schedule.last = Some(Time::ZERO);
        let deadline = drain_deadline(Time::ZERO, Duration::from_millis(3)).unwrap();
        assert_eq!(
            idle_wait(config, &schedule, Time::ZERO, Some(deadline)),
            Duration::from_millis(3)
        );
        assert_eq!(
            idle_wait(config, &schedule, deadline, Some(deadline)),
            Duration::ZERO
        );
        assert_eq!(
            drain_deadline(Time::MAX, Duration::ZERO).unwrap(),
            Time::MAX
        );
        assert!(drain_deadline(Time::MAX, Duration::from_nanos(1)).is_err());
        assert!(drain_deadline(Time::ZERO, Duration::MAX).is_err());
    }

    #[test]
    fn drain_budget_expires_while_delayed_work_remains_outstanding() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let f = fixture(JobQueueConfig::default());
            f.worker
                .runner()
                .ingest(IngestRequest::new("future", "input"))
                .unwrap();
            f.storage
                .connection()
                .execute_with_params_sync(
                    "UPDATE embedding_jobs SET submitted_at = ?1 WHERE status = 'pending';",
                    &[SqliteValue::Integer(i64::MAX)],
                )
                .unwrap();
            let shutdown = AtomicBool::new(false);
            let mut future =
                std::pin::pin!(
                    f.worker
                        .drain(&cx, "drain", &shutdown, Duration::from_millis(2),)
                );
            let mut polls = 0;
            let outcome = poll_fn(|task_cx| match future.as_mut().poll(task_cx) {
                Poll::Pending => {
                    polls += 1;
                    assert!(polls < 10_000, "deadline must stop the unready drain");
                    Poll::Pending
                }
                Poll::Ready(result) => Poll::Ready(result),
            })
            .await
            .unwrap();
            let WorkerDrainOutcome::DeadlineReached { report, remaining } = outcome else {
                panic!("unready work must not be reported drained: {outcome:?}");
            };
            assert_eq!(remaining.pending, 1);
            assert_eq!(
                remaining.processing + remaining.failed + remaining.completed,
                0
            );
            assert_eq!(report.work.jobs_completed + report.work.jobs_failed, 0);
            assert!(f.sink.entries().is_empty());
        });
    }
}
