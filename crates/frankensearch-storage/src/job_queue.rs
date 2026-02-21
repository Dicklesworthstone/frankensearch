use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

use crate::connection::{Storage, map_storage_error};

const SUBSYSTEM: &str = "storage";
const HASH_EMBEDDER_PREFIX: &str = "fnv1a-";
const JL_EMBEDDER_PREFIX: &str = "jl-";
const MAX_BACKOFF_EXPONENT: u32 = 20;
const MAX_RETRY_DELAY_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Skipped,
}

impl JobStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "processing" => Some(Self::Processing),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "skipped" => Some(Self::Skipped),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueErrorKind {
    NotFound,
    Conflict,
    Validation,
}

impl QueueErrorKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Validation => "validation",
        }
    }
}

#[derive(Debug)]
struct QueueError {
    kind: QueueErrorKind,
    message: String,
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for QueueError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnqueueRequest {
    pub doc_id: String,
    pub embedder_id: String,
    pub content_hash: [u8; 32],
    pub priority: i32,
}

impl EnqueueRequest {
    #[must_use]
    pub fn new(
        doc_id: impl Into<String>,
        embedder_id: impl Into<String>,
        content_hash: [u8; 32],
        priority: i32,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            embedder_id: embedder_id.into(),
            content_hash,
            priority,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEnqueueResult {
    pub inserted: u64,
    pub replaced: u64,
    pub deduplicated: u64,
    pub skipped_hash_embedder: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimedJob {
    pub job_id: i64,
    pub doc_id: String,
    pub embedder_id: String,
    pub priority: i32,
    pub retry_count: u32,
    pub max_retries: u32,
    pub submitted_at: i64,
    pub content_hash: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailResult {
    Retried {
        retry_count: u32,
        delay_ms: u64,
        next_attempt_at_ms: i64,
    },
    TerminalFailed {
        retry_count: u32,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueDepth {
    pub pending: usize,
    pub ready_pending: usize,
    pub processing: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobQueueConfig {
    pub batch_size: usize,
    pub visibility_timeout_ms: u64,
    pub max_retries: u32,
    pub retry_base_delay_ms: u64,
    pub stale_job_threshold_ms: u64,
    pub backpressure_threshold: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            visibility_timeout_ms: 30_000,
            max_retries: 3,
            retry_base_delay_ms: 100,
            stale_job_threshold_ms: 300_000,
            backpressure_threshold: 10_000,
        }
    }
}

#[derive(Debug, Default)]
pub struct JobQueueMetrics {
    pub total_enqueued: AtomicU64,
    pub total_completed: AtomicU64,
    pub total_failed: AtomicU64,
    pub total_skipped: AtomicU64,
    pub total_retried: AtomicU64,
    pub total_deduplicated: AtomicU64,
    pub total_batches_processed: AtomicU64,
    pub total_embed_time_us: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobQueueMetricsSnapshot {
    pub total_enqueued: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub total_skipped: u64,
    pub total_retried: u64,
    pub total_deduplicated: u64,
    pub total_batches_processed: u64,
    pub total_embed_time_us: u64,
}

impl JobQueueMetrics {
    #[must_use]
    pub fn snapshot(&self) -> JobQueueMetricsSnapshot {
        JobQueueMetricsSnapshot {
            total_enqueued: self.total_enqueued.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_skipped: self.total_skipped.load(Ordering::Relaxed),
            total_retried: self.total_retried.load(Ordering::Relaxed),
            total_deduplicated: self.total_deduplicated.load(Ordering::Relaxed),
            total_batches_processed: self.total_batches_processed.load(Ordering::Relaxed),
            total_embed_time_us: self.total_embed_time_us.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PersistentJobQueue {
    storage: Arc<Storage>,
    config: JobQueueConfig,
    metrics: Arc<JobQueueMetrics>,
}

impl PersistentJobQueue {
    #[must_use]
    pub fn new(storage: Arc<Storage>, config: JobQueueConfig) -> Self {
        Self {
            storage,
            config,
            metrics: Arc::new(JobQueueMetrics::default()),
        }
    }

    #[must_use]
    pub fn with_metrics(
        storage: Arc<Storage>,
        config: JobQueueConfig,
        metrics: Arc<JobQueueMetrics>,
    ) -> Self {
        Self {
            storage,
            config,
            metrics,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &JobQueueConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics(&self) -> &JobQueueMetrics {
        self.metrics.as_ref()
    }

    pub fn enqueue(
        &self,
        doc_id: &str,
        embedder_id: &str,
        content_hash: &[u8; 32],
        priority: i32,
    ) -> SearchResult<bool> {
        let request = EnqueueRequest::new(doc_id, embedder_id, *content_hash, priority);
        let submitted_at = unix_timestamp_ms()?;
        let outcome = self.storage.transaction(|conn| {
            enqueue_inner(conn, &request, submitted_at, self.config.max_retries)
        })?;
        self.record_enqueue_outcome(outcome);

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.enqueue",
            doc_id,
            embedder_id,
            outcome = ?outcome,
            "embedding job enqueue completed"
        );

        Ok(matches!(
            outcome,
            EnqueueOutcome::Inserted | EnqueueOutcome::Replaced
        ))
    }

    pub fn enqueue_batch(&self, jobs: &[EnqueueRequest]) -> SearchResult<BatchEnqueueResult> {
        if jobs.is_empty() {
            return Ok(BatchEnqueueResult::default());
        }

        let submitted_base = unix_timestamp_ms()?;
        let max_retries = self.config.max_retries;
        let summary = self.storage.transaction(|conn| {
            let mut summary = BatchEnqueueResult::default();
            for (index, job) in jobs.iter().enumerate() {
                let submitted_at = submitted_base.saturating_add(usize_to_i64(index)?);
                let outcome = enqueue_inner(conn, job, submitted_at, max_retries)?;
                summary.record(outcome);
            }
            Ok(summary)
        })?;

        if summary.inserted > 0 || summary.replaced > 0 {
            self.metrics
                .total_enqueued
                .fetch_add(summary.inserted + summary.replaced, Ordering::Relaxed);
        }
        if summary.deduplicated > 0 || summary.skipped_hash_embedder > 0 {
            self.metrics.total_deduplicated.fetch_add(
                summary.deduplicated + summary.skipped_hash_embedder,
                Ordering::Relaxed,
            );
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.enqueue_batch",
            requested = jobs.len(),
            inserted = summary.inserted,
            replaced = summary.replaced,
            deduplicated = summary.deduplicated,
            skipped_hash_embedder = summary.skipped_hash_embedder,
            "embedding job batch enqueue completed"
        );

        Ok(summary)
    }

    pub fn claim_batch(&self, worker_id: &str, batch_size: usize) -> SearchResult<Vec<ClaimedJob>> {
        ensure_non_empty(worker_id, "worker_id")?;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let batch_limit = batch_size.min(self.config.batch_size);
        let now_ms = unix_timestamp_ms()?;
        let limit = usize_to_i64(batch_limit)?;

        // Use BEGIN IMMEDIATE for defense-in-depth (acquires a write lock
        // eagerly on backends that support cross-connection locking).
        //
        // IMPORTANT: FrankenSQLite connections maintain isolated in-memory
        // snapshots. Multiple connections to the same database file will NOT
        // see each other's committed writes within the same process.
        // Callers MUST route all claim_batch calls through a single
        // PersistentJobQueue instance backed by one Storage connection to
        // prevent duplicate assignments.
        let claimed = self.storage.immediate_transaction(|conn| {
            let claim_params = [SqliteValue::Integer(now_ms), SqliteValue::Integer(limit)];
            let candidates = conn
                .query_with_params(
                    "SELECT job_id, doc_id, embedder_id, priority, retry_count, max_retries, content_hash, submitted_at \
                     FROM embedding_jobs \
                     WHERE status = 'pending' \
                       AND submitted_at <= ?1 \
                       AND NOT EXISTS ( \
                           SELECT 1 FROM embedding_jobs active \
                           WHERE active.doc_id = embedding_jobs.doc_id \
                             AND active.embedder_id = embedding_jobs.embedder_id \
                             AND active.status = 'processing' \
                       ) \
                     ORDER BY priority DESC, submitted_at ASC \
                     LIMIT ?2;",
                    &claim_params,
                )
                .map_err(map_storage_error)?;

            let mut claimed = Vec::with_capacity(candidates.len());
            for row in &candidates {
                let job_id = row_i64(row, 0, "embedding_jobs.job_id")?;
                let update_params = [
                    SqliteValue::Text(JobStatus::Processing.as_str().to_owned()),
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(worker_id.to_owned()),
                    SqliteValue::Integer(job_id),
                ];
                let updated = conn
                    .execute_with_params(
                        "UPDATE embedding_jobs \
                         SET status = ?1, started_at = ?2, worker_id = ?3, error_message = NULL \
                         WHERE job_id = ?4 AND status = 'pending';",
                        &update_params,
                    )
                    .map_err(map_storage_error)?;
                if updated != 1 {
                    continue;
                }

                claimed.push(ClaimedJob {
                    job_id,
                    doc_id: row_text(row, 1, "embedding_jobs.doc_id")?.to_owned(),
                    embedder_id: row_text(row, 2, "embedding_jobs.embedder_id")?.to_owned(),
                    priority: row_i32(row, 3, "embedding_jobs.priority")?,
                    retry_count: row_u32(row, 4, "embedding_jobs.retry_count")?,
                    max_retries: row_u32(row, 5, "embedding_jobs.max_retries")?,
                    content_hash: row_optional_blob_32(row, 6, "embedding_jobs.content_hash")?,
                    submitted_at: row_i64(row, 7, "embedding_jobs.submitted_at")?,
                });
            }
            Ok(claimed)
        })?;

        if !claimed.is_empty() {
            self.metrics
                .total_batches_processed
                .fetch_add(1, Ordering::Relaxed);
        }
        let elapsed_us = duration_as_u64(start.elapsed().as_micros());
        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.claim_batch",
            worker_id,
            requested = batch_size,
            effective_batch_size = batch_limit,
            claimed = claimed.len(),
            claim_latency_us = elapsed_us,
            "embedding job claim completed"
        );

        Ok(claimed)
    }

    pub fn complete(&self, job_id: i64) -> SearchResult<()> {
        let now_ms = unix_timestamp_ms()?;
        let started_at = self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if state.status != JobStatus::Processing {
                return Err(conflict_error(format!(
                    "job {job_id} is not processing (status={})",
                    state.status.as_str()
                )));
            }

            let target_status = JobStatus::Completed.as_str();
            let delete_params = [
                SqliteValue::Text(state.doc_id.clone()),
                SqliteValue::Text(state.embedder_id.clone()),
                SqliteValue::Text(target_status.to_owned()),
            ];
            conn.execute_with_params(
                "DELETE FROM embedding_jobs \
                 WHERE doc_id = ?1 AND embedder_id = ?2 AND status = ?3;",
                &delete_params,
            )
            .map_err(map_storage_error)?;

            let params = [
                SqliteValue::Text(target_status.to_owned()),
                SqliteValue::Integer(now_ms),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, completed_at = ?2, worker_id = NULL, error_message = NULL \
                     WHERE job_id = ?3 AND status = 'processing';",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during completion"
                )));
            }
            Ok(state.started_at)
        })?;

        self.metrics.total_completed.fetch_add(1, Ordering::Relaxed);
        if let Some(started_at_ms) = started_at {
            let elapsed_ms = now_ms.saturating_sub(started_at_ms);
            if let Ok(elapsed_ms_u64) = u64::try_from(elapsed_ms) {
                self.metrics
                    .total_embed_time_us
                    .fetch_add(elapsed_ms_u64.saturating_mul(1_000), Ordering::Relaxed);
            }
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.complete",
            job_id,
            "embedding job marked completed"
        );
        Ok(())
    }

    pub fn fail(&self, job_id: i64, error: &str) -> SearchResult<FailResult> {
        ensure_non_empty(error, "error")?;

        let now_ms = unix_timestamp_ms()?;
        let retry_base_delay_ms = self.config.retry_base_delay_ms;
        let result = self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if state.status != JobStatus::Processing {
                return Err(conflict_error(format!(
                    "job {job_id} is not processing (status={})",
                    state.status.as_str()
                )));
            }

            let retry_count = state.retry_count.saturating_add(1);
            if retry_count > state.max_retries {
                let target_status = JobStatus::Failed.as_str();
                let delete_params = [
                    SqliteValue::Text(state.doc_id.clone()),
                    SqliteValue::Text(state.embedder_id.clone()),
                    SqliteValue::Text(target_status.to_owned()),
                ];
                conn.execute_with_params(
                    "DELETE FROM embedding_jobs \
                     WHERE doc_id = ?1 AND embedder_id = ?2 AND status = ?3;",
                    &delete_params,
                )
                .map_err(map_storage_error)?;

                let params = [
                    SqliteValue::Text(target_status.to_owned()),
                    SqliteValue::Integer(i64::from(retry_count)),
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(error.to_owned()),
                    SqliteValue::Integer(job_id),
                ];
                let updated = conn
                    .execute_with_params(
                        "UPDATE embedding_jobs \
                         SET status = ?1, retry_count = ?2, completed_at = ?3, error_message = ?4, worker_id = NULL \
                         WHERE job_id = ?5 AND status = 'processing';",
                        &params,
                    )
                    .map_err(map_storage_error)?;
                if updated != 1 {
                    return Err(conflict_error(format!(
                        "job {job_id} changed status during fail/terminal transition"
                    )));
                }
                return Ok(FailResult::TerminalFailed { retry_count });
            }

            let delay_ms =
                compute_retry_delay_ms(retry_base_delay_ms, retry_count.saturating_sub(1));
            let next_attempt_at_ms = now_ms.saturating_add(i64::try_from(delay_ms).unwrap_or(i64::MAX));
            let params = [
                SqliteValue::Text(JobStatus::Pending.as_str().to_owned()),
                SqliteValue::Integer(i64::from(retry_count)),
                SqliteValue::Integer(next_attempt_at_ms),
                SqliteValue::Text(error.to_owned()),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, retry_count = ?2, submitted_at = ?3, started_at = NULL, completed_at = NULL, \
                         error_message = ?4, worker_id = NULL \
                     WHERE job_id = ?5 AND status = 'processing';",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during fail/retry transition"
                )));
            }
            Ok(FailResult::Retried {
                retry_count,
                delay_ms,
                next_attempt_at_ms,
            })
        })?;

        match result {
            FailResult::Retried { .. } => {
                self.metrics.total_retried.fetch_add(1, Ordering::Relaxed);
            }
            FailResult::TerminalFailed { .. } => {
                self.metrics.total_failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.fail",
            job_id,
            ?result,
            "embedding job failure transition completed"
        );

        Ok(result)
    }

    pub fn skip(&self, job_id: i64, reason: &str) -> SearchResult<()> {
        ensure_non_empty(reason, "reason")?;
        let now_ms = unix_timestamp_ms()?;
        self.storage.transaction(|conn| {
            let Some(state) = load_job_state(conn, job_id)? else {
                return Err(not_found_error("embedding_jobs", &job_id.to_string()));
            };
            if !matches!(state.status, JobStatus::Pending | JobStatus::Processing) {
                return Err(conflict_error(format!(
                    "job {job_id} cannot be skipped from status {}",
                    state.status.as_str()
                )));
            }

            let target_status = JobStatus::Skipped.as_str();
            let delete_params = [
                SqliteValue::Text(state.doc_id.clone()),
                SqliteValue::Text(state.embedder_id.clone()),
                SqliteValue::Text(target_status.to_owned()),
            ];
            conn.execute_with_params(
                "DELETE FROM embedding_jobs \
                 WHERE doc_id = ?1 AND embedder_id = ?2 AND status = ?3;",
                &delete_params,
            )
            .map_err(map_storage_error)?;

            let params = [
                SqliteValue::Text(target_status.to_owned()),
                SqliteValue::Integer(now_ms),
                SqliteValue::Text(reason.to_owned()),
                SqliteValue::Integer(job_id),
            ];
            let updated = conn
                .execute_with_params(
                    "UPDATE embedding_jobs \
                     SET status = ?1, completed_at = ?2, worker_id = NULL, error_message = ?3 \
                     WHERE job_id = ?4 AND status IN ('pending', 'processing');",
                    &params,
                )
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(conflict_error(format!(
                    "job {job_id} changed status during skip transition"
                )));
            }
            Ok(())
        })?;

        self.metrics.total_skipped.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            target: "frankensearch.storage",
            op = "queue.skip",
            job_id,
            "embedding job marked skipped"
        );
        Ok(())
    }

    pub fn reclaim_stale_jobs(&self) -> SearchResult<usize> {
        let now_ms = unix_timestamp_ms()?;
        let reclaim_after_ms = self
            .config
            .visibility_timeout_ms
            .min(self.config.stale_job_threshold_ms);
        let cutoff = now_ms.saturating_sub(i64::try_from(reclaim_after_ms).unwrap_or(i64::MAX));
        let (reclaimed_pending, superseded) = self.storage.transaction(|conn| {
            let stale_params = [SqliteValue::Integer(cutoff)];
            let stale_rows = conn
                .query_with_params(
                    "SELECT job_id, doc_id, embedder_id \
                     FROM embedding_jobs \
                     WHERE status = 'processing' \
                       AND (started_at IS NULL OR started_at <= ?1);",
                    &stale_params,
                )
                .map_err(map_storage_error)?;

            let mut reclaimed_pending = 0_usize;
            let mut superseded = 0_usize;
            for row in &stale_rows {
                let job_id = row_i64(row, 0, "embedding_jobs.job_id")?;
                let doc_id = row_text(row, 1, "embedding_jobs.doc_id")?.to_owned();
                let embedder_id = row_text(row, 2, "embedding_jobs.embedder_id")?.to_owned();

                let pending_params = [
                    SqliteValue::Text(doc_id.clone()),
                    SqliteValue::Text(embedder_id.clone()),
                ];
                let pending_exists = !conn
                    .query_with_params(
                        "SELECT job_id \
                         FROM embedding_jobs \
                         WHERE doc_id = ?1 AND embedder_id = ?2 AND status = 'pending' \
                         LIMIT 1;",
                        &pending_params,
                    )
                    .map_err(map_storage_error)?
                    .is_empty();

                if pending_exists {
                    let delete_params = [SqliteValue::Integer(job_id)];
                    let deleted = conn
                        .execute_with_params(
                            "DELETE FROM embedding_jobs \
                             WHERE job_id = ?1 AND status = 'processing';",
                            &delete_params,
                        )
                        .map_err(map_storage_error)?;
                    if deleted == 1 {
                        superseded += 1;
                    }
                } else {
                    let update_params = [
                        SqliteValue::Text(JobStatus::Pending.as_str().to_owned()),
                        SqliteValue::Integer(now_ms),
                        SqliteValue::Text("reclaimed stale lease".to_owned()),
                        SqliteValue::Integer(job_id),
                    ];
                    let updated = conn
                        .execute_with_params(
                            "UPDATE embedding_jobs \
                             SET status = ?1, submitted_at = ?2, started_at = NULL, worker_id = NULL, error_message = ?3 \
                             WHERE job_id = ?4 AND status = 'processing';",
                            &update_params,
                        )
                        .map_err(map_storage_error)?;
                    if updated == 1 {
                        reclaimed_pending += 1;
                    }
                }
            }

            Ok((reclaimed_pending, superseded))
        })?;
        let reclaimed = reclaimed_pending.saturating_add(superseded);

        if reclaimed_pending > 0 {
            self.metrics
                .total_retried
                .fetch_add(usize_to_u64(reclaimed_pending), Ordering::Relaxed);
        }

        if reclaimed > 0 {
            tracing::warn!(
                target: "frankensearch.storage",
                op = "queue.reclaim_stale_jobs",
                reclaimed,
                reclaimed_pending,
                superseded,
                cutoff_ms = cutoff,
                "reclaimed stale embedding jobs"
            );
        } else {
            tracing::trace!(
                target: "frankensearch.storage",
                op = "queue.reclaim_stale_jobs",
                reclaimed = 0,
                "no stale embedding jobs found"
            );
        }

        Ok(reclaimed)
    }

    pub fn is_backpressured(&self) -> SearchResult<bool> {
        let depth = self.queue_depth()?;
        Ok(depth.ready_pending > self.config.backpressure_threshold)
    }

    pub fn queue_depth(&self) -> SearchResult<QueueDepth> {
        fetch_queue_depth(self.storage.connection())
    }

    fn record_enqueue_outcome(&self, outcome: EnqueueOutcome) {
        match outcome {
            EnqueueOutcome::Inserted | EnqueueOutcome::Replaced => {
                self.metrics.total_enqueued.fetch_add(1, Ordering::Relaxed);
            }
            EnqueueOutcome::Deduplicated | EnqueueOutcome::HashEmbedderSkipped => {
                self.metrics
                    .total_deduplicated
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

pub(crate) fn fetch_queue_depth(conn: &Connection) -> SearchResult<QueueDepth> {
    let mut depth = QueueDepth::default();
    let rows = conn
        .query("SELECT status, COUNT(*) FROM embedding_jobs GROUP BY status;")
        .map_err(map_storage_error)?;
    for row in &rows {
        let status = row_text(row, 0, "embedding_jobs.status")?;
        let count = i64_to_usize(row_i64(row, 1, "embedding_jobs.count")?)?;
        match JobStatus::from_str(status) {
            Some(JobStatus::Pending) => depth.pending = count,
            Some(JobStatus::Processing) => depth.processing = count,
            Some(JobStatus::Completed) => depth.completed = count,
            Some(JobStatus::Failed) => depth.failed = count,
            Some(JobStatus::Skipped) => depth.skipped = count,
            None => {
                return Err(queue_error(
                    QueueErrorKind::Validation,
                    format!("unknown queue status value: {status:?}"),
                ));
            }
        }
    }

    let now_ms = unix_timestamp_ms()?;
    let ready_params = [SqliteValue::Integer(now_ms)];
    let ready_rows = conn
        .query_with_params(
            "SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending' AND submitted_at <= ?1;",
            &ready_params,
        )
        .map_err(map_storage_error)?;
    if let Some(row) = ready_rows.first() {
        depth.ready_pending = i64_to_usize(row_i64(row, 0, "embedding_jobs.ready_pending")?)?;
    }

    Ok(depth)
}

impl BatchEnqueueResult {
    fn record(&mut self, outcome: EnqueueOutcome) {
        match outcome {
            EnqueueOutcome::Inserted => self.inserted += 1,
            EnqueueOutcome::Replaced => self.replaced += 1,
            EnqueueOutcome::Deduplicated => self.deduplicated += 1,
            EnqueueOutcome::HashEmbedderSkipped => self.skipped_hash_embedder += 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnqueueOutcome {
    Inserted,
    Replaced,
    Deduplicated,
    HashEmbedderSkipped,
}

#[derive(Debug, Clone)]
struct JobState {
    status: JobStatus,
    retry_count: u32,
    max_retries: u32,
    started_at: Option<i64>,
    doc_id: String,
    embedder_id: String,
}

pub(crate) fn enqueue_inner(
    conn: &Connection,
    request: &EnqueueRequest,
    submitted_at: i64,
    max_retries: u32,
) -> SearchResult<EnqueueOutcome> {
    ensure_non_empty(&request.doc_id, "doc_id")?;
    ensure_non_empty(&request.embedder_id, "embedder_id")?;

    if !document_exists(conn, &request.doc_id)? {
        return Err(not_found_error("documents", &request.doc_id));
    }

    if is_hash_embedder(&request.embedder_id) {
        return Ok(EnqueueOutcome::HashEmbedderSkipped);
    }

    let active_params = [
        SqliteValue::Text(request.doc_id.clone()),
        SqliteValue::Text(request.embedder_id.clone()),
    ];
    let active_rows = conn
        .query_with_params(
            "SELECT job_id, content_hash \
             FROM embedding_jobs \
             WHERE doc_id = ?1 AND embedder_id = ?2 AND status IN ('pending', 'processing');",
            &active_params,
        )
        .map_err(map_storage_error)?;

    let mut has_active_job = false;
    for row in &active_rows {
        has_active_job = true;
        let existing_hash = row_optional_blob_32(row, 1, "embedding_jobs.content_hash")?;
        if existing_hash.as_ref() == Some(&request.content_hash) {
            return Ok(EnqueueOutcome::Deduplicated);
        }
    }

    if has_active_job {
        conn.execute_with_params(
            "DELETE FROM embedding_jobs \
             WHERE doc_id = ?1 AND embedder_id = ?2 AND status = 'pending';",
            &active_params,
        )
        .map_err(map_storage_error)?;
    }

    let insert_params = [
        SqliteValue::Text(request.doc_id.clone()),
        SqliteValue::Text(request.embedder_id.clone()),
        SqliteValue::Integer(i64::from(request.priority)),
        SqliteValue::Integer(submitted_at),
        SqliteValue::Integer(i64::from(max_retries)),
        SqliteValue::Blob(request.content_hash.to_vec()),
    ];
    conn.execute_with_params(
        "INSERT INTO embedding_jobs (\
            doc_id, embedder_id, priority, submitted_at, status, retry_count, max_retries, content_hash\
         ) VALUES (?1, ?2, ?3, ?4, 'pending', 0, ?5, ?6);",
        &insert_params,
    )
    .map_err(map_storage_error)?;

    Ok(if has_active_job {
        EnqueueOutcome::Replaced
    } else {
        EnqueueOutcome::Inserted
    })
}

fn load_job_state(conn: &Connection, job_id: i64) -> SearchResult<Option<JobState>> {
    let params = [SqliteValue::Integer(job_id)];
    let rows = conn
        .query_with_params(
            "SELECT status, retry_count, max_retries, started_at, doc_id, embedder_id \
             FROM embedding_jobs \
             WHERE job_id = ?1 \
             LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };

    let status_value = row_text(row, 0, "embedding_jobs.status")?;
    let status = JobStatus::from_str(status_value).ok_or_else(|| {
        queue_error(
            QueueErrorKind::Validation,
            format!("unknown queue status value: {status_value:?}"),
        )
    })?;

    Ok(Some(JobState {
        status,
        retry_count: row_u32(row, 1, "embedding_jobs.retry_count")?,
        max_retries: row_u32(row, 2, "embedding_jobs.max_retries")?,
        started_at: row_optional_i64(row, 3)?,
        doc_id: row_text(row, 4, "embedding_jobs.doc_id")?.to_owned(),
        embedder_id: row_text(row, 5, "embedding_jobs.embedder_id")?.to_owned(),
    }))
}

fn document_exists(conn: &Connection, doc_id: &str) -> SearchResult<bool> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT doc_id FROM documents WHERE doc_id = ?1 LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    Ok(!rows.is_empty())
}

fn is_hash_embedder(embedder_id: &str) -> bool {
    embedder_id.starts_with(HASH_EMBEDDER_PREFIX)
        || embedder_id.starts_with(JL_EMBEDDER_PREFIX)
        || embedder_id == "hash/fnv1a"
}

fn compute_retry_delay_ms(base_delay_ms: u64, exponent: u32) -> u64 {
    let shift = exponent.min(MAX_BACKOFF_EXPONENT);
    let factor = 1_u64.checked_shl(shift).unwrap_or(u64::MAX);
    base_delay_ms.saturating_mul(factor).min(MAX_RETRY_DELAY_MS)
}

fn ensure_non_empty(value: &str, field: &str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(queue_error(
            QueueErrorKind::Validation,
            format!("{field} must not be empty"),
        ));
    }
    Ok(())
}

fn not_found_error(entity: &str, key: &str) -> SearchError {
    queue_error(
        QueueErrorKind::NotFound,
        format!("{entity} record not found for key {key:?}"),
    )
}

fn conflict_error(message: String) -> SearchError {
    queue_error(QueueErrorKind::Conflict, message)
}

pub(crate) fn is_queue_conflict(error: &SearchError) -> bool {
    let SearchError::SubsystemError { subsystem, source } = error else {
        return false;
    };
    if *subsystem != SUBSYSTEM {
        return false;
    }
    source
        .downcast_ref::<QueueError>()
        .is_some_and(|queue_error| queue_error.kind == QueueErrorKind::Conflict)
}

fn queue_error(kind: QueueErrorKind, message: String) -> SearchError {
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(QueueError { kind, message }),
    }
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_i32(row: &Row, index: usize, field: &str) -> SearchResult<i32> {
    let value = row_i64(row, index, field)?;
    i32::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("{field} value {value} does not fit into i32"),
        )
    })
}

fn row_u32(row: &Row, index: usize, field: &str) -> SearchResult<u32> {
    let value = row_i64(row, index, field)?;
    u32::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("{field} value {value} does not fit into u32"),
        )
    })
}

fn row_optional_i64(row: &Row, index: usize) -> SearchResult<Option<i64>> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected optional i64 type: {:?}",
                other
            ))),
        }),
    }
}

fn row_optional_blob_32(row: &Row, index: usize, field: &str) -> SearchResult<Option<[u8; 32]>> {
    match row.get(index) {
        Some(SqliteValue::Blob(value)) => {
            if value.len() != 32 {
                return Err(queue_error(
                    QueueErrorKind::Validation,
                    format!("{field} expected 32-byte hash, found {} bytes", value.len()),
                ));
            }
            let mut hash = [0_u8; 32];
            hash.copy_from_slice(value);
            Ok(Some(hash))
        }
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected optional blob type for {field}: {:?}",
                other
            ))),
        }),
    }
}

fn usize_to_i64(value: usize) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("value {value} does not fit into i64"),
        )
    })
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn i64_to_usize(value: i64) -> SearchResult<usize> {
    usize::try_from(value).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            format!("value {value} does not fit into usize"),
        )
    })
}

fn duration_as_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(map_storage_error)?;
    i64::try_from(duration.as_millis()).map_err(|_| {
        queue_error(
            QueueErrorKind::Validation,
            "system timestamp overflowed i64 milliseconds".to_owned(),
        )
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::process;
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsqlite_types::value::SqliteValue;

    use crate::connection::{Storage, StorageConfig};
    use crate::document::DocumentRecord;

    use super::{
        ClaimedJob, EnqueueRequest, FailResult, JobQueueConfig, JobStatus, PersistentJobQueue,
        QueueDepth, unix_timestamp_ms,
    };

    struct TempDbPath {
        path: PathBuf,
    }

    impl TempDbPath {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "frankensearch-job-queue-{tag}-{}-{nanos}.sqlite3",
                process::id()
            ));
            Self { path }
        }

        fn config(&self) -> StorageConfig {
            StorageConfig {
                db_path: self.path.clone(),
                ..StorageConfig::default()
            }
        }
    }

    impl Drop for TempDbPath {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let candidate = if suffix.is_empty() {
                    self.path.clone()
                } else {
                    PathBuf::from(format!("{}{}", self.path.display(), suffix))
                };
                let _ = std::fs::remove_file(candidate);
            }
        }
    }

    fn queue_fixture(config: JobQueueConfig) -> (PersistentJobQueue, Arc<Storage>) {
        let storage = Arc::new(Storage::open_in_memory().expect("in-memory storage should open"));
        let queue = PersistentJobQueue::new(Arc::clone(&storage), config);
        (queue, storage)
    }

    fn insert_document(storage: &Storage, doc_id: &str, hash_seed: u8) {
        let mut hash = [0_u8; 32];
        hash.fill(hash_seed);
        let doc = DocumentRecord {
            doc_id: doc_id.to_owned(),
            source_path: Some(format!("tests://{doc_id}")),
            content_preview: format!("content for {doc_id}"),
            content_hash: hash,
            content_length: 32,
            created_at: 1_739_499_200,
            updated_at: 1_739_499_200,
            metadata: None,
        };
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
    }

    fn status_counts(storage: &Storage) -> QueueDepth {
        let rows = storage
            .connection()
            .query("SELECT status, COUNT(*) FROM embedding_jobs GROUP BY status;")
            .expect("status query should succeed");
        let mut depth = QueueDepth::default();
        for row in &rows {
            let status = row
                .get(0)
                .and_then(|value| match value {
                    SqliteValue::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .expect("status column should be text");
            let raw_count = row
                .get(1)
                .and_then(|value| match value {
                    SqliteValue::Integer(count) => Some(*count),
                    _ => None,
                })
                .expect("count column should be integer");
            let count = usize::try_from(raw_count).expect("count value should fit into usize");
            match JobStatus::from_str(&status).expect("queue status should be known") {
                JobStatus::Pending => depth.pending = count,
                JobStatus::Processing => depth.processing = count,
                JobStatus::Completed => depth.completed = count,
                JobStatus::Failed => depth.failed = count,
                JobStatus::Skipped => depth.skipped = count,
            }
        }
        depth
    }

    fn claim_single(queue: &PersistentJobQueue, worker_id: &str) -> ClaimedJob {
        let claimed = queue
            .claim_batch(worker_id, 1)
            .expect("claim should succeed");
        assert_eq!(claimed.len(), 1, "exactly one job should be claimed");
        claimed
            .into_iter()
            .next()
            .expect("claim result should contain a job")
    }

    #[test]
    fn enqueue_deduplicates_same_job() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-1", 1);

        let hash = [9_u8; 32];
        assert!(
            queue
                .enqueue("doc-1", "all-MiniLM-L6-v2", &hash, 7)
                .expect("initial enqueue should succeed")
        );
        assert!(
            !queue
                .enqueue("doc-1", "all-MiniLM-L6-v2", &hash, 7)
                .expect("duplicate enqueue should succeed")
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 1);
        assert_eq!(depth.processing, 0);

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_enqueued, 1);
        assert_eq!(metrics.total_deduplicated, 1);
    }

    #[test]
    fn enqueue_replaces_active_job_when_hash_changes() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-2", 2);

        let hash_a = [1_u8; 32];
        let hash_b = [2_u8; 32];
        assert!(
            queue
                .enqueue("doc-2", "all-MiniLM-L6-v2", &hash_a, 1)
                .expect("first enqueue should succeed")
        );
        assert!(
            queue
                .enqueue("doc-2", "all-MiniLM-L6-v2", &hash_b, 1)
                .expect("replacement enqueue should succeed")
        );

        let depth = status_counts(storage.as_ref());
        assert_eq!(
            depth.pending, 1,
            "only replacement pending job should remain"
        );
        assert_eq!(depth.processing, 0);

        let params = [SqliteValue::Text("doc-2".to_owned())];
        let rows = storage
            .connection()
            .query_with_params(
                "SELECT content_hash FROM embedding_jobs WHERE doc_id = ?1 AND status = 'pending' LIMIT 1;",
                &params,
            )
            .expect("pending row query should succeed");
        assert_eq!(rows.len(), 1);
        let pending_hash = rows[0]
            .get(0)
            .and_then(|value| match value {
                SqliteValue::Blob(bytes) => Some(bytes.clone()),
                _ => None,
            })
            .expect("pending hash should be blob");
        assert_eq!(pending_hash, hash_b.to_vec());
    }

    #[test]
    fn hash_embedder_jobs_are_skipped() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-3", 3);

        let hash = [3_u8; 32];
        assert!(
            !queue
                .enqueue("doc-3", "fnv1a-384", &hash, 0)
                .expect("hash enqueue should succeed")
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 0);
        assert_eq!(queue.metrics().snapshot().total_deduplicated, 1);
    }

    #[test]
    fn claim_batch_assigns_disjoint_jobs() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            batch_size: 4,
            ..JobQueueConfig::default()
        });

        insert_document(storage.as_ref(), "doc-a", 4);
        insert_document(storage.as_ref(), "doc-b", 5);
        insert_document(storage.as_ref(), "doc-c", 6);

        let hash_a = [4_u8; 32];
        let hash_b = [5_u8; 32];
        let hash_c = [6_u8; 32];
        queue
            .enqueue("doc-a", "all-MiniLM-L6-v2", &hash_a, 0)
            .expect("enqueue a");
        queue
            .enqueue("doc-b", "all-MiniLM-L6-v2", &hash_b, 10)
            .expect("enqueue b");
        queue
            .enqueue("doc-c", "all-MiniLM-L6-v2", &hash_c, 5)
            .expect("enqueue c");

        let first = queue
            .claim_batch("worker-a", 2)
            .expect("first claim should succeed");
        assert_eq!(first.len(), 2);
        assert!(first[0].priority >= first[1].priority);

        let second = queue
            .claim_batch("worker-b", 2)
            .expect("second claim should succeed");
        assert_eq!(second.len(), 1);

        let mut seen = HashSet::new();
        for job in first.iter().chain(second.iter()) {
            assert!(
                seen.insert(job.job_id),
                "duplicate claim for job {}",
                job.job_id
            );
        }

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 3);
    }

    #[test]
    fn enqueue_replacement_keeps_inflight_processing_job() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-inflight", 7);

        let old_hash = [7_u8; 32];
        let new_hash = [8_u8; 32];

        queue
            .enqueue("doc-inflight", "all-MiniLM-L6-v2", &old_hash, 0)
            .expect("initial enqueue should succeed");
        let inflight = claim_single(&queue, "worker-inflight");
        assert_eq!(
            inflight.content_hash,
            Some(old_hash),
            "claimed processing job should keep original hash"
        );

        assert!(
            queue
                .enqueue("doc-inflight", "all-MiniLM-L6-v2", &new_hash, 5)
                .expect("replacement enqueue should succeed")
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(
            depth.processing, 1,
            "in-flight job should remain processing"
        );
        assert_eq!(depth.pending, 1, "replacement should be queued as pending");

        queue
            .complete(inflight.job_id)
            .expect("completing inflight job should still succeed");
        let replacement = claim_single(&queue, "worker-replacement");
        assert_ne!(
            replacement.job_id, inflight.job_id,
            "replacement must be a distinct job row"
        );
        assert_eq!(
            replacement.content_hash,
            Some(new_hash),
            "replacement job should carry new hash"
        );
    }

    #[test]
    fn fail_transitions_retry_then_terminal_failure() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            max_retries: 1,
            retry_base_delay_ms: 0,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-fail", 8);

        let hash = [8_u8; 32];
        queue
            .enqueue("doc-fail", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let first_claim = claim_single(&queue, "worker-f1");

        let first_fail = queue
            .fail(first_claim.job_id, "transient failure")
            .expect("first fail should succeed");
        assert!(matches!(
            first_fail,
            FailResult::Retried { retry_count: 1, .. }
        ));

        let second_claim = claim_single(&queue, "worker-f2");
        assert_eq!(second_claim.job_id, first_claim.job_id);

        let second_fail = queue
            .fail(second_claim.job_id, "permanent failure")
            .expect("second fail should succeed");
        assert!(matches!(
            second_fail,
            FailResult::TerminalFailed { retry_count: 2 }
        ));

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.failed, 1);
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 0);

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_retried, 1);
        assert_eq!(metrics.total_failed, 1);
    }

    #[test]
    fn reclaim_stale_jobs_restores_processing_work() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            visibility_timeout_ms: 10,
            stale_job_threshold_ms: 10,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-stale", 9);

        let hash = [9_u8; 32];
        queue
            .enqueue("doc-stale", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue, "worker-stale");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(1_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");

        let reclaimed = queue
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed");
        assert_eq!(reclaimed, 1);

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 1);
        assert_eq!(depth.processing, 0);

        let reclaimed_claim = claim_single(&queue, "worker-restored");
        assert_eq!(reclaimed_claim.job_id, claim.job_id);
    }

    #[test]
    fn reclaim_stale_jobs_uses_visibility_timeout_when_stale_threshold_is_larger() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            visibility_timeout_ms: 25,
            stale_job_threshold_ms: 60_000,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-vis-timeout", 22);

        let hash = [22_u8; 32];
        queue
            .enqueue("doc-vis-timeout", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue, "worker-visibility");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(1_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");

        let reclaimed = queue
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed");
        assert_eq!(reclaimed, 1, "job should reclaim after visibility timeout");
    }

    #[test]
    fn restart_recovery_preserves_and_reclaims_jobs() {
        let tmp = TempDbPath::new("restart");
        let queue_config = JobQueueConfig {
            visibility_timeout_ms: 10,
            stale_job_threshold_ms: 10,
            ..JobQueueConfig::default()
        };

        let storage_a =
            Arc::new(Storage::open(tmp.config()).expect("initial storage open should succeed"));
        insert_document(storage_a.as_ref(), "doc-restart", 10);
        let queue_a = PersistentJobQueue::new(Arc::clone(&storage_a), queue_config);
        let hash = [10_u8; 32];
        queue_a
            .enqueue("doc-restart", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue_a, "worker-before-restart");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(5_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage_a
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");
        drop(queue_a);
        drop(storage_a);

        let storage_b =
            Arc::new(Storage::open(tmp.config()).expect("reopened storage should succeed"));
        let queue_b = PersistentJobQueue::new(Arc::clone(&storage_b), queue_config);

        let before_reclaim = queue_b.queue_depth().expect("queue depth should load");
        assert_eq!(before_reclaim.processing, 1);
        assert_eq!(before_reclaim.pending, 0);

        let reclaimed = queue_b
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed");
        assert_eq!(reclaimed, 1);

        let recovered = claim_single(&queue_b, "worker-after-restart");
        assert_eq!(recovered.job_id, claim.job_id);
    }

    #[test]
    fn reclaim_stale_jobs_removes_processing_job_when_replacement_pending_exists() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            visibility_timeout_ms: 10,
            stale_job_threshold_ms: 10,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-superseded", 44);

        let old_hash = [44_u8; 32];
        let new_hash = [45_u8; 32];
        queue
            .enqueue("doc-superseded", "all-MiniLM-L6-v2", &old_hash, 0)
            .expect("initial enqueue should succeed");
        let claim = claim_single(&queue, "worker-superseded");
        queue
            .enqueue("doc-superseded", "all-MiniLM-L6-v2", &new_hash, 0)
            .expect("replacement enqueue should succeed");

        let stale_started_at = unix_timestamp_ms()
            .expect("timestamp should resolve")
            .saturating_sub(5_000);
        let params = [
            SqliteValue::Integer(stale_started_at),
            SqliteValue::Integer(claim.job_id),
        ];
        storage
            .connection()
            .execute_with_params(
                "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                &params,
            )
            .expect("stale timestamp update should succeed");

        let reclaimed = queue
            .reclaim_stale_jobs()
            .expect("stale reclaim should succeed with replacement pending");
        assert_eq!(
            reclaimed, 1,
            "stale superseded processing row should be removed"
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.processing, 0);
        assert_eq!(depth.pending, 1);

        let replacement = claim_single(&queue, "worker-replacement");
        assert_ne!(replacement.job_id, claim.job_id);
        assert_eq!(replacement.content_hash, Some(new_hash));
    }

    #[test]
    fn enqueue_batch_is_atomic_on_partial_failure() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-ok", 11);

        let jobs = vec![
            EnqueueRequest::new("doc-ok", "all-MiniLM-L6-v2", [11_u8; 32], 0),
            EnqueueRequest::new("doc-missing", "all-MiniLM-L6-v2", [12_u8; 32], 0),
        ];
        let err = queue
            .enqueue_batch(&jobs)
            .expect_err("batch enqueue should fail when one row is invalid");
        assert!(
            err.to_string().contains("not_found"),
            "error should classify as not_found: {err}"
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(
            depth.pending, 0,
            "transaction should rollback partial batch insert"
        );
    }

    #[test]
    fn enqueue_batch_reports_insert_replace_dedup_and_hash_skip() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-batch-1", 13);
        insert_document(storage.as_ref(), "doc-batch-2", 14);

        let jobs = vec![
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [13_u8; 32], 0),
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [13_u8; 32], 0),
            EnqueueRequest::new("doc-batch-1", "all-MiniLM-L6-v2", [15_u8; 32], 0),
            EnqueueRequest::new("doc-batch-2", "fnv1a-384", [14_u8; 32], 0),
        ];
        let summary = queue
            .enqueue_batch(&jobs)
            .expect("batch enqueue should succeed");
        assert_eq!(summary.inserted, 1);
        assert_eq!(summary.replaced, 1);
        assert_eq!(summary.deduplicated, 1);
        assert_eq!(summary.skipped_hash_embedder, 1);

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(
            depth.pending, 1,
            "only one pending semantic job should remain"
        );

        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_enqueued, 2);
        assert_eq!(metrics.total_deduplicated, 2);
    }

    #[test]
    fn claim_batch_orders_by_priority_then_fifo_submission() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            batch_size: 8,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-priority-a", 61);
        insert_document(storage.as_ref(), "doc-priority-b", 62);
        insert_document(storage.as_ref(), "doc-priority-c", 63);

        let jobs = vec![
            EnqueueRequest::new("doc-priority-a", "all-MiniLM-L6-v2", [61_u8; 32], 10),
            EnqueueRequest::new("doc-priority-b", "all-MiniLM-L6-v2", [62_u8; 32], 10),
            EnqueueRequest::new("doc-priority-c", "all-MiniLM-L6-v2", [63_u8; 32], 5),
        ];
        queue
            .enqueue_batch(&jobs)
            .expect("batch enqueue should succeed");
        std::thread::sleep(std::time::Duration::from_millis(5));

        let claimed = queue
            .claim_batch("worker-priority", 8)
            .expect("claim should succeed");
        let claimed_doc_ids: Vec<&str> = claimed.iter().map(|job| job.doc_id.as_str()).collect();
        assert_eq!(
            claimed_doc_ids,
            vec!["doc-priority-a", "doc-priority-b", "doc-priority-c"]
        );
    }

    #[test]
    fn queue_depth_tracks_ready_pending_for_delayed_retries() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            backpressure_threshold: 0,
            max_retries: 3,
            retry_base_delay_ms: 1_000,
            ..JobQueueConfig::default()
        });
        insert_document(storage.as_ref(), "doc-delay", 71);

        let hash = [71_u8; 32];
        queue
            .enqueue("doc-delay", "all-MiniLM-L6-v2", &hash, 0)
            .expect("enqueue should succeed");
        let claim = claim_single(&queue, "worker-delay");
        let retry = queue
            .fail(claim.job_id, "transient")
            .expect("fail should schedule retry");
        assert!(
            matches!(retry, FailResult::Retried { delay_ms, .. } if delay_ms >= 1_000),
            "retry should include a future delay"
        );

        let depth = queue.queue_depth().expect("queue depth should load");
        assert_eq!(depth.pending, 1);
        assert_eq!(
            depth.ready_pending, 0,
            "delayed retry should not appear as ready pending yet"
        );
        assert!(
            !queue
                .is_backpressured()
                .expect("backpressure check should succeed"),
            "delayed-only pending jobs should not trigger backpressure when ready_pending is zero"
        );
    }

    #[test]
    fn backpressure_trips_only_when_ready_pending_exceeds_threshold() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            backpressure_threshold: 2,
            ..JobQueueConfig::default()
        });

        for (doc_id, seed) in [("doc-bp-1", 81_u8), ("doc-bp-2", 82), ("doc-bp-3", 83)] {
            insert_document(storage.as_ref(), doc_id, seed);
            queue
                .enqueue(doc_id, "all-MiniLM-L6-v2", &[seed; 32], 0)
                .expect("enqueue should succeed");
        }

        assert!(
            queue
                .is_backpressured()
                .expect("backpressure check should work"),
            "3 ready pending jobs should exceed threshold 2"
        );

        let _ = queue
            .claim_batch("worker-bp", 1)
            .expect("claim should reduce pending depth");
        assert!(
            !queue
                .is_backpressured()
                .expect("backpressure check should work"),
            "2 ready pending jobs should not exceed threshold 2"
        );
    }

    #[test]
    fn round_robin_multi_worker_claims_never_double_assign_jobs() {
        const JOB_COUNT: usize = 50;

        let queue_config = JobQueueConfig {
            batch_size: 5,
            ..JobQueueConfig::default()
        };
        let storage = Arc::new(Storage::open_in_memory().expect("seed storage should open"));
        let queue = PersistentJobQueue::new(Arc::clone(&storage), queue_config);
        for index in 0..JOB_COUNT {
            let doc_id = format!("doc-concurrent-{index}");
            let seed = u8::try_from(index % 200).expect("seed should fit into u8");
            insert_document(storage.as_ref(), &doc_id, seed);
            queue
                .enqueue(&doc_id, "all-MiniLM-L6-v2", &[seed; 32], 0)
                .expect("seed enqueue should succeed");
        }

        let mut unique_ids = HashSet::new();
        let mut total_claimed = 0_usize;
        for worker_idx in 0..4 {
            let worker_id = format!("worker-{worker_idx}");
            loop {
                let batch = queue
                    .claim_batch(&worker_id, 3)
                    .expect("claim should succeed");
                if batch.is_empty() {
                    break;
                }
                for job in batch {
                    assert!(
                        unique_ids.insert(job.job_id),
                        "job {} was claimed more than once",
                        job.job_id
                    );
                    total_claimed = total_claimed.saturating_add(1);
                }
            }
        }

        assert_eq!(
            total_claimed, JOB_COUNT,
            "all seeded jobs must be claimed exactly once"
        );
    }

    /// Verifies that concurrent workers claiming through a single queue
    /// (the production pattern) never double-assign jobs.
    ///
    /// Previously tracked as bd-2cnc: the race occurred when separate
    /// connections each read stale snapshots. The fix documents and enforces
    /// that all `claim_batch` calls MUST flow through a single
    /// `PersistentJobQueue` instance backed by one `Storage` connection.
    ///
    /// Since `Storage` is `!Send` (`FrankenSQLite` `Connection` uses `Rc`),
    /// the test uses a channel pattern: worker threads send claim requests
    /// to a dispatcher thread that owns the queue, matching the production
    /// pattern where an asupersync event loop owns the storage.
    #[test]
    fn concurrent_claim_once_through_shared_queue_has_no_double_assignment() {
        const JOB_COUNT: usize = 50;
        const WORKER_COUNT: usize = 8;
        const BATCH_SIZE: usize = 10;
        type ClaimRequest = (String, mpsc::Sender<Vec<ClaimedJob>>);

        let queue_config = JobQueueConfig {
            batch_size: BATCH_SIZE,
            ..JobQueueConfig::default()
        };

        // All queue operations happen on this single thread (the test main
        // thread), which owns Storage. Workers communicate via channels.
        let storage = Arc::new(Storage::open_in_memory().expect("in-memory storage should open"));
        let queue = PersistentJobQueue::new(Arc::clone(&storage), queue_config);

        for index in 0..JOB_COUNT {
            let doc_id = format!("doc-cc-{index}");
            let seed = u8::try_from(index % 200).expect("seed should fit into u8");
            insert_document(storage.as_ref(), &doc_id, seed);
            queue
                .enqueue(&doc_id, "all-MiniLM-L6-v2", &[seed; 32], 0)
                .expect("seed enqueue should succeed");
        }

        // Each worker thread sends (worker_id, response_sender) to the
        // dispatcher. The dispatcher calls claim_batch and sends back results.
        let (request_tx, request_rx) = mpsc::channel::<ClaimRequest>();

        // Shared gate so workers start racing at the same time.
        let barrier = Arc::new(Barrier::new(WORKER_COUNT));
        let mut handles = Vec::with_capacity(WORKER_COUNT);
        for worker in 0..WORKER_COUNT {
            let tx = request_tx.clone();
            let start = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                start.wait();
                let mut all_claimed = Vec::new();
                loop {
                    let (resp_tx, resp_rx) = mpsc::channel();
                    tx.send((format!("worker-cc-{worker}"), resp_tx))
                        .expect("request send should succeed");
                    let batch = resp_rx.recv().expect("response recv should succeed");
                    if batch.is_empty() {
                        break;
                    }
                    all_claimed.extend(batch.into_iter().map(|job| job.job_id));
                }
                all_claimed
            }));
        }
        drop(request_tx);

        // Dispatch loop: process claim requests sequentially on the queue-
        // owning thread. This serializes claim_batch calls through a single
        // Storage connection, preventing the TOCTOU duplicate-claim race.
        for (worker_id, resp_tx) in request_rx {
            let batch = queue
                .claim_batch(&worker_id, BATCH_SIZE)
                .expect("claim should succeed");
            let _ = resp_tx.send(batch);
        }

        let mut claimed_ids = Vec::new();
        for handle in handles {
            let claimed = handle.join().expect("worker claim thread should join");
            claimed_ids.extend(claimed);
        }

        assert_eq!(
            claimed_ids.len(),
            JOB_COUNT,
            "all pending jobs should be claimed exactly once across workers"
        );
        let unique: HashSet<i64> = claimed_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            JOB_COUNT,
            "no job id should appear in more than one claimed batch"
        );
    }

    // ─── Utility function tests (bd-1erd) ───────────────────────────────

    #[test]
    fn job_status_as_str_all_variants() {
        assert_eq!(JobStatus::Pending.as_str(), "pending");
        assert_eq!(JobStatus::Processing.as_str(), "processing");
        assert_eq!(JobStatus::Completed.as_str(), "completed");
        assert_eq!(JobStatus::Failed.as_str(), "failed");
        assert_eq!(JobStatus::Skipped.as_str(), "skipped");
    }

    #[test]
    fn job_status_from_str_all_valid() {
        assert_eq!(JobStatus::from_str("pending"), Some(JobStatus::Pending));
        assert_eq!(
            JobStatus::from_str("processing"),
            Some(JobStatus::Processing)
        );
        assert_eq!(JobStatus::from_str("completed"), Some(JobStatus::Completed));
        assert_eq!(JobStatus::from_str("failed"), Some(JobStatus::Failed));
        assert_eq!(JobStatus::from_str("skipped"), Some(JobStatus::Skipped));
    }

    #[test]
    fn job_status_from_str_invalid() {
        assert!(JobStatus::from_str("").is_none());
        assert!(JobStatus::from_str("unknown").is_none());
        assert!(JobStatus::from_str("PENDING").is_none());
    }

    #[test]
    fn job_status_serde_roundtrip() {
        for status in [
            JobStatus::Pending,
            JobStatus::Processing,
            JobStatus::Completed,
            JobStatus::Failed,
            JobStatus::Skipped,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: JobStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn queue_error_kind_as_str() {
        use super::QueueErrorKind;
        assert_eq!(QueueErrorKind::NotFound.as_str(), "not_found");
        assert_eq!(QueueErrorKind::Conflict.as_str(), "conflict");
        assert_eq!(QueueErrorKind::Validation.as_str(), "validation");
    }

    #[test]
    fn queue_error_display_format() {
        use super::QueueError;
        let err = QueueError {
            kind: super::QueueErrorKind::NotFound,
            message: "missing doc".to_owned(),
        };
        assert_eq!(err.to_string(), "not_found: missing doc");
    }

    #[test]
    fn compute_retry_delay_ms_zero_exponent() {
        use super::compute_retry_delay_ms;
        assert_eq!(compute_retry_delay_ms(100, 0), 100);
    }

    #[test]
    fn compute_retry_delay_ms_exponential_growth() {
        use super::compute_retry_delay_ms;
        assert_eq!(compute_retry_delay_ms(100, 1), 200);
        assert_eq!(compute_retry_delay_ms(100, 2), 400);
        assert_eq!(compute_retry_delay_ms(100, 3), 800);
    }

    #[test]
    fn compute_retry_delay_ms_capped_at_max() {
        use super::compute_retry_delay_ms;
        // At high exponent, should cap at MAX_RETRY_DELAY_MS (30_000)
        let result = compute_retry_delay_ms(100, 15);
        assert!(result <= 30_000, "delay {result} should not exceed 30_000");
    }

    #[test]
    fn compute_retry_delay_ms_at_max_backoff_exponent() {
        use super::compute_retry_delay_ms;
        // Exponent beyond MAX_BACKOFF_EXPONENT (20) is clamped
        let at_max = compute_retry_delay_ms(1, 20);
        let beyond_max = compute_retry_delay_ms(1, 25);
        assert_eq!(at_max, beyond_max);
    }

    #[test]
    fn compute_retry_delay_ms_zero_base() {
        use super::compute_retry_delay_ms;
        assert_eq!(compute_retry_delay_ms(0, 5), 0);
    }

    #[test]
    fn is_hash_embedder_matches() {
        use super::is_hash_embedder;
        assert!(is_hash_embedder("fnv1a-384"));
        assert!(is_hash_embedder("fnv1a-256"));
        assert!(is_hash_embedder("fnv1a-"));
        assert!(is_hash_embedder("hash/fnv1a"));
    }

    #[test]
    fn is_hash_embedder_non_match() {
        use super::is_hash_embedder;
        assert!(!is_hash_embedder("all-MiniLM-L6-v2"));
        assert!(!is_hash_embedder(""));
        assert!(!is_hash_embedder("fnv1a"));
    }

    #[test]
    fn ensure_non_empty_rejects_empty() {
        use super::ensure_non_empty;
        let err = ensure_non_empty("", "field").unwrap_err();
        assert!(err.to_string().contains("field"));
    }

    #[test]
    fn ensure_non_empty_rejects_whitespace() {
        use super::ensure_non_empty;
        let err = ensure_non_empty("   \t\n", "worker_id").unwrap_err();
        assert!(err.to_string().contains("worker_id"));
    }

    #[test]
    fn ensure_non_empty_accepts_valid() {
        use super::ensure_non_empty;
        assert!(ensure_non_empty("hello", "field").is_ok());
    }

    #[test]
    fn usize_to_i64_normal() {
        use super::usize_to_i64;
        assert_eq!(usize_to_i64(0).unwrap(), 0_i64);
        assert_eq!(usize_to_i64(42).unwrap(), 42_i64);
    }

    #[test]
    fn i64_to_usize_normal() {
        use super::i64_to_usize;
        assert_eq!(i64_to_usize(0).unwrap(), 0_usize);
        assert_eq!(i64_to_usize(100).unwrap(), 100_usize);
    }

    #[test]
    fn i64_to_usize_negative_fails() {
        use super::i64_to_usize;
        assert!(i64_to_usize(-1).is_err());
    }

    #[test]
    fn usize_to_u64_normal_value() {
        use super::usize_to_u64;
        assert_eq!(usize_to_u64(0), 0_u64);
        assert_eq!(usize_to_u64(99), 99_u64);
    }

    #[test]
    fn duration_as_u64_normal_value() {
        use super::duration_as_u64;
        assert_eq!(duration_as_u64(0), 0_u64);
        assert_eq!(duration_as_u64(1_000_000), 1_000_000_u64);
    }

    #[test]
    fn duration_as_u64_overflow() {
        use super::duration_as_u64;
        assert_eq!(duration_as_u64(u128::MAX), u64::MAX);
    }

    // ─── Type defaults and serde (bd-1erd) ──────────────────────────────

    #[test]
    fn enqueue_request_new_fields() {
        let req = EnqueueRequest::new("d1", "emb-1", [5_u8; 32], 3);
        assert_eq!(req.doc_id, "d1");
        assert_eq!(req.embedder_id, "emb-1");
        assert_eq!(req.content_hash, [5_u8; 32]);
        assert_eq!(req.priority, 3);
    }

    #[test]
    fn enqueue_request_serde_roundtrip() {
        let req = EnqueueRequest::new("doc", "emb", [7_u8; 32], 1);
        let json = serde_json::to_string(&req).unwrap();
        let decoded: EnqueueRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn batch_enqueue_result_default() {
        let r = super::BatchEnqueueResult::default();
        assert_eq!(r.inserted, 0);
        assert_eq!(r.replaced, 0);
        assert_eq!(r.deduplicated, 0);
        assert_eq!(r.skipped_hash_embedder, 0);
    }

    #[test]
    fn batch_enqueue_result_record_all_outcomes() {
        use super::{BatchEnqueueResult, EnqueueOutcome};
        let mut r = BatchEnqueueResult::default();
        r.record(EnqueueOutcome::Inserted);
        r.record(EnqueueOutcome::Replaced);
        r.record(EnqueueOutcome::Deduplicated);
        r.record(EnqueueOutcome::HashEmbedderSkipped);
        assert_eq!(r.inserted, 1);
        assert_eq!(r.replaced, 1);
        assert_eq!(r.deduplicated, 1);
        assert_eq!(r.skipped_hash_embedder, 1);
    }

    #[test]
    fn queue_depth_default_all_zeros() {
        let d = QueueDepth::default();
        assert_eq!(d.pending, 0);
        assert_eq!(d.ready_pending, 0);
        assert_eq!(d.processing, 0);
        assert_eq!(d.completed, 0);
        assert_eq!(d.failed, 0);
        assert_eq!(d.skipped, 0);
    }

    #[test]
    fn queue_depth_serde_roundtrip() {
        let d = QueueDepth {
            pending: 5,
            ready_pending: 3,
            processing: 2,
            completed: 10,
            failed: 1,
            skipped: 0,
        };
        let json = serde_json::to_string(&d).unwrap();
        let decoded: QueueDepth = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, d);
    }

    #[test]
    fn job_queue_config_default_values() {
        let cfg = JobQueueConfig::default();
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.visibility_timeout_ms, 30_000);
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.retry_base_delay_ms, 100);
        assert_eq!(cfg.stale_job_threshold_ms, 300_000);
        assert_eq!(cfg.backpressure_threshold, 10_000);
    }

    #[test]
    fn job_queue_config_serde_roundtrip() {
        let cfg = JobQueueConfig {
            batch_size: 64,
            visibility_timeout_ms: 5_000,
            max_retries: 5,
            retry_base_delay_ms: 200,
            stale_job_threshold_ms: 60_000,
            backpressure_threshold: 1_000,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: JobQueueConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn job_queue_metrics_default_snapshot_all_zeros() {
        use super::JobQueueMetrics;
        let metrics = JobQueueMetrics::default();
        let snap = metrics.snapshot();
        assert_eq!(snap.total_enqueued, 0);
        assert_eq!(snap.total_completed, 0);
        assert_eq!(snap.total_failed, 0);
        assert_eq!(snap.total_skipped, 0);
        assert_eq!(snap.total_retried, 0);
        assert_eq!(snap.total_deduplicated, 0);
        assert_eq!(snap.total_batches_processed, 0);
        assert_eq!(snap.total_embed_time_us, 0);
    }

    #[test]
    fn job_queue_metrics_snapshot_default() {
        use super::JobQueueMetricsSnapshot;
        let snap = JobQueueMetricsSnapshot::default();
        assert_eq!(snap.total_enqueued, 0);
        assert_eq!(snap.total_completed, 0);
    }

    #[test]
    fn fail_result_serde_roundtrip_retried() {
        let r = FailResult::Retried {
            retry_count: 2,
            delay_ms: 200,
            next_attempt_at_ms: 1_000_000,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: FailResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn fail_result_serde_roundtrip_terminal() {
        let r = FailResult::TerminalFailed { retry_count: 4 };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: FailResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn claimed_job_serde_roundtrip() {
        let job = ClaimedJob {
            job_id: 42,
            doc_id: "doc-1".to_owned(),
            embedder_id: "emb-1".to_owned(),
            priority: 5,
            retry_count: 1,
            max_retries: 3,
            submitted_at: 1_000_000,
            content_hash: Some([9_u8; 32]),
        };
        let json = serde_json::to_string(&job).unwrap();
        let decoded: ClaimedJob = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, job);
    }

    #[test]
    fn claimed_job_without_content_hash() {
        let job = ClaimedJob {
            job_id: 1,
            doc_id: "d".to_owned(),
            embedder_id: "e".to_owned(),
            priority: 0,
            retry_count: 0,
            max_retries: 0,
            submitted_at: 0,
            content_hash: None,
        };
        let json = serde_json::to_string(&job).unwrap();
        let decoded: ClaimedJob = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content_hash, None);
    }

    // ─── Edge case tests requiring storage (bd-1erd) ────────────────────

    #[test]
    fn enqueue_batch_empty_returns_default() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let result = queue
            .enqueue_batch(&[])
            .expect("empty batch should succeed");
        assert_eq!(result, super::BatchEnqueueResult::default());
    }

    #[test]
    fn claim_batch_zero_size_returns_empty() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let claimed = queue
            .claim_batch("worker", 0)
            .expect("zero batch should succeed");
        assert!(claimed.is_empty());
    }

    #[test]
    fn claim_batch_empty_worker_id_rejected() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .claim_batch("", 1)
            .expect_err("empty worker_id should fail");
        assert!(err.to_string().contains("worker_id"), "error: {err}");
    }

    #[test]
    fn claim_batch_whitespace_worker_id_rejected() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .claim_batch("   ", 1)
            .expect_err("whitespace worker_id should fail");
        assert!(err.to_string().contains("worker_id"), "error: {err}");
    }

    #[test]
    fn complete_missing_job_returns_error() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .complete(99999)
            .expect_err("completing nonexistent job should fail");
        assert!(err.to_string().contains("not_found"), "error: {err}");
    }

    #[test]
    fn complete_pending_job_returns_conflict() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-cp", 50);

        let hash = [50_u8; 32];
        queue.enqueue("doc-cp", "emb", &hash, 0).unwrap();

        // Get job_id from the pending row
        let rows = storage
            .connection()
            .query("SELECT job_id FROM embedding_jobs WHERE status = 'pending' LIMIT 1;")
            .expect("query should succeed");
        let job_id = match rows[0].get(0) {
            Some(SqliteValue::Integer(id)) => *id,
            _ => panic!("expected integer job_id"),
        };

        let err = queue
            .complete(job_id)
            .expect_err("completing pending job should fail");
        assert!(err.to_string().contains("not processing"), "error: {err}");
    }

    #[test]
    fn fail_missing_job_returns_error() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .fail(99999, "error message")
            .expect_err("failing nonexistent job should fail");
        assert!(err.to_string().contains("not_found"), "error: {err}");
    }

    #[test]
    fn fail_empty_error_rejected() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .fail(1, "")
            .expect_err("empty error message should be rejected");
        assert!(err.to_string().contains("error"), "error: {err}");
    }

    #[test]
    fn skip_missing_job_returns_error() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .skip(99999, "reason")
            .expect_err("skipping nonexistent job should fail");
        assert!(err.to_string().contains("not_found"), "error: {err}");
    }

    #[test]
    fn skip_empty_reason_rejected() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .skip(1, "")
            .expect_err("empty reason should be rejected");
        assert!(err.to_string().contains("reason"), "error: {err}");
    }

    #[test]
    fn skip_completed_job_returns_conflict() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-sc", 51);

        let hash = [51_u8; 32];
        queue.enqueue("doc-sc", "emb", &hash, 0).unwrap();
        let claimed = claim_single(&queue, "worker-sc");
        queue.complete(claimed.job_id).unwrap();

        let err = queue
            .skip(claimed.job_id, "too late")
            .expect_err("skipping completed job should fail");
        assert!(
            err.to_string().contains("cannot be skipped"),
            "error: {err}"
        );
    }

    #[test]
    fn skip_pending_job_succeeds() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-sp", 52);

        let hash = [52_u8; 32];
        queue.enqueue("doc-sp", "emb", &hash, 0).unwrap();

        // Get job_id from pending row
        let rows = storage
            .connection()
            .query("SELECT job_id FROM embedding_jobs WHERE status = 'pending' LIMIT 1;")
            .expect("query should succeed");
        let job_id = match rows[0].get(0) {
            Some(SqliteValue::Integer(id)) => *id,
            _ => panic!("expected integer job_id"),
        };

        queue.skip(job_id, "not needed").unwrap();

        let depth = queue.queue_depth().unwrap();
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.skipped, 1);
    }

    #[test]
    fn reclaim_stale_jobs_no_stale_returns_zero() {
        let (queue, storage) = queue_fixture(JobQueueConfig::default());
        insert_document(storage.as_ref(), "doc-ns", 53);

        let hash = [53_u8; 32];
        queue.enqueue("doc-ns", "emb", &hash, 0).unwrap();

        // Don't claim — jobs are pending, not processing
        let reclaimed = queue.reclaim_stale_jobs().unwrap();
        assert_eq!(reclaimed, 0);
    }

    #[test]
    fn queue_config_accessor_returns_configured_values() {
        let cfg = JobQueueConfig {
            batch_size: 99,
            ..JobQueueConfig::default()
        };
        let (queue, _storage) = queue_fixture(cfg);
        assert_eq!(queue.config().batch_size, 99);
    }

    #[test]
    fn queue_with_metrics_uses_shared_metrics() {
        use super::JobQueueMetrics;
        let storage = Arc::new(Storage::open_in_memory().expect("storage"));
        let metrics = Arc::new(JobQueueMetrics::default());
        let queue = PersistentJobQueue::with_metrics(
            storage,
            JobQueueConfig::default(),
            Arc::clone(&metrics),
        );

        // Verify the queue uses the shared metrics
        insert_document(queue.storage.as_ref(), "doc-wm", 54);
        queue.enqueue("doc-wm", "emb", &[54_u8; 32], 0).unwrap();
        assert_eq!(metrics.snapshot().total_enqueued, 1);
    }

    #[test]
    fn enqueue_missing_document_rejected() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let err = queue
            .enqueue("nonexistent", "emb", &[1_u8; 32], 0)
            .expect_err("enqueue for missing doc should fail");
        assert!(err.to_string().contains("not_found"), "error: {err}");
    }

    #[test]
    fn claim_batch_respects_config_batch_size_cap() {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            batch_size: 2, // cap
            ..JobQueueConfig::default()
        });

        for i in 0u8..5 {
            let doc_id = format!("doc-cap-{i}");
            insert_document(storage.as_ref(), &doc_id, i);
            queue.enqueue(&doc_id, "emb", &[i; 32], 0).unwrap();
        }

        // Request 10 but cap is 2
        let claimed = queue.claim_batch("worker", 10).unwrap();
        assert_eq!(claimed.len(), 2, "should be capped at config batch_size");
    }

    #[test]
    fn queue_depth_empty_queue() {
        let (queue, _storage) = queue_fixture(JobQueueConfig::default());
        let depth = queue.queue_depth().unwrap();
        assert_eq!(depth, QueueDepth::default());
    }

    #[test]
    fn not_found_error_contains_entity_and_key() {
        use super::not_found_error;
        let err = not_found_error("documents", "doc-99");
        assert!(err.to_string().contains("documents"), "error: {err}");
        assert!(err.to_string().contains("doc-99"), "error: {err}");
    }

    #[test]
    fn conflict_error_contains_message() {
        use super::conflict_error;
        let err = conflict_error("job changed status".to_owned());
        assert!(
            err.to_string().contains("job changed status"),
            "error: {err}"
        );
    }

    #[test]
    fn is_queue_conflict_detects_conflict_errors() {
        use super::{conflict_error, is_queue_conflict};
        let err = conflict_error("job changed status".to_owned());
        assert!(is_queue_conflict(&err));
    }

    #[test]
    fn is_queue_conflict_ignores_non_conflict_errors() {
        use super::{QueueErrorKind, is_queue_conflict, queue_error};
        let err = queue_error(QueueErrorKind::Validation, "bad input".to_owned());
        assert!(!is_queue_conflict(&err));
    }

    #[test]
    fn queue_error_creates_subsystem_error() {
        use super::{QueueErrorKind, queue_error};
        let err = queue_error(QueueErrorKind::Validation, "bad input".to_owned());
        match err {
            frankensearch_core::SearchError::SubsystemError { subsystem, .. } => {
                assert_eq!(subsystem, "storage");
            }
            other => panic!("expected SubsystemError, got {other:?}"),
        }
    }

    #[test]
    fn batch_enqueue_result_serde_roundtrip() {
        use super::BatchEnqueueResult;
        let r = BatchEnqueueResult {
            inserted: 5,
            replaced: 2,
            deduplicated: 3,
            skipped_hash_embedder: 1,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: BatchEnqueueResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn job_queue_metrics_snapshot_serde_roundtrip() {
        use super::JobQueueMetricsSnapshot;
        let snap = JobQueueMetricsSnapshot {
            total_enqueued: 10,
            total_completed: 8,
            total_failed: 1,
            total_skipped: 1,
            total_retried: 2,
            total_deduplicated: 3,
            total_batches_processed: 5,
            total_embed_time_us: 42_000,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let decoded: JobQueueMetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, snap);
    }

    #[test]
    fn queue_error_kind_serde_roundtrip() {
        use super::QueueErrorKind;
        for kind in [
            QueueErrorKind::NotFound,
            QueueErrorKind::Conflict,
            QueueErrorKind::Validation,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let decoded: QueueErrorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, kind);
        }
    }
}
