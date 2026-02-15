//! Embedding job queue with backpressure.
//!
//! [`EmbeddingQueue`] is a bounded, dedup-aware job queue for incremental
//! index building. Documents are submitted for embedding, canonicalized, and
//! content-hashed. The queue skips re-embedding when a document's content
//! has not changed.
//!
//! Backpressure: when the queue reaches capacity, [`EmbeddingQueue::submit`]
//! returns [`SearchError::QueueFull`].
//!
//! The `EmbeddingJobRunner` drains batches from the queue, embeds them via
//! an `Embedder`, and writes vectors to a `VectorIndexWriter`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use frankensearch_core::canonicalize::Canonicalizer;
use frankensearch_core::{SearchError, SearchResult};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the embedding job queue.
#[derive(Debug, Clone)]
pub struct EmbeddingQueueConfig {
    /// Maximum number of pending jobs. Default: 1000.
    pub capacity: usize,
    /// Maximum batch size for processing. Default: 32.
    pub batch_size: usize,
    /// Maximum retries per job before permanent failure. Default: 3.
    pub max_retries: u32,
}

impl Default for EmbeddingQueueConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            batch_size: 32,
            max_retries: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Job types
// ---------------------------------------------------------------------------

/// A request to embed a document.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Unique document identifier.
    pub doc_id: String,
    /// Raw (pre-canonicalization) document text.
    pub text: String,
    /// Optional metadata to carry through the pipeline.
    pub metadata: Option<serde_json::Value>,
    /// When this request was submitted.
    pub submitted_at: Instant,
}

/// An embedding job after canonicalization and content hashing.
#[derive(Debug, Clone)]
pub struct EmbeddingJob {
    /// Document identifier.
    pub doc_id: String,
    /// Canonicalized text ready for embedding.
    pub canonical_text: String,
    /// SHA-256 hex digest of the canonical text.
    pub content_hash: String,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
    /// When the original request was submitted.
    pub submitted_at: Instant,
    /// Number of times this job has been retried.
    pub retry_count: u32,
}

/// Result of processing a single embedding job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobOutcome {
    /// Successfully embedded and indexed.
    Succeeded,
    /// Skipped: content hash unchanged since last embedding.
    SkippedUnchanged,
    /// Skipped: canonicalized text was empty (low-signal content).
    SkippedEmpty,
    /// Failed: will be retried.
    Retryable,
    /// Failed permanently: max retries exceeded.
    Failed,
}

// ---------------------------------------------------------------------------
// Queue metrics
// ---------------------------------------------------------------------------

/// Lock-free counters for queue telemetry.
#[derive(Debug, Default)]
pub struct QueueMetrics {
    /// Total jobs submitted (including duplicates).
    pub total_submitted: AtomicU64,
    /// Jobs skipped due to dedup (same content hash).
    pub total_deduped: AtomicU64,
    /// Jobs successfully embedded and indexed.
    pub total_succeeded: AtomicU64,
    /// Jobs that failed but can be retried.
    pub total_retryable: AtomicU64,
    /// Jobs that permanently failed.
    pub total_failed: AtomicU64,
    /// Jobs skipped (empty canonicalized text).
    pub total_skipped: AtomicU64,
    /// Batches processed.
    pub total_batches: AtomicU64,
    /// Total embedding time in microseconds.
    pub total_embed_time_us: AtomicU64,
}

impl QueueMetrics {
    /// Record a job outcome.
    #[allow(clippy::too_many_lines)]
    pub fn record(&self, outcome: JobOutcome) {
        match outcome {
            JobOutcome::Succeeded => {
                self.total_succeeded.fetch_add(1, Ordering::Relaxed);
            }
            JobOutcome::SkippedUnchanged => {
                self.total_deduped.fetch_add(1, Ordering::Relaxed);
            }
            JobOutcome::SkippedEmpty => {
                self.total_skipped.fetch_add(1, Ordering::Relaxed);
            }
            JobOutcome::Retryable => {
                self.total_retryable.fetch_add(1, Ordering::Relaxed);
            }
            JobOutcome::Failed => {
                self.total_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Current pending count snapshot (requires external tracking).
    /// This is informational; the actual queue depth is tracked by the queue itself.
    pub fn total_submitted(&self) -> u64 {
        self.total_submitted.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Embedding queue
// ---------------------------------------------------------------------------

/// Internal queue state behind a mutex.
#[derive(Debug)]
struct QueueState {
    /// Pending jobs in submission order.
    jobs: VecDeque<EmbeddingJob>,
    /// Tracks which `doc_ids` are currently in the queue (for dedup).
    pending_ids: HashMap<String, usize>,
    /// Content hashes of recently embedded documents (for skip-unchanged).
    known_hashes: HashMap<String, KnownHashEntry>,
    /// Approximate LRU queue for pruning known content hashes.
    ///
    /// The tuple stores `(doc_id, sequence)` to guard against stale queue entries.
    known_hash_order: VecDeque<(String, usize)>,
    /// Monotonically increasing sequence for dedup ordering.
    sequence: usize,
}

/// Cached hash entry with monotonic sequence for LRU pruning.
#[derive(Debug)]
struct KnownHashEntry {
    hash: String,
    sequence: usize,
}

/// Bounded embedding job queue with content-hash dedup and backpressure.
///
/// Thread-safe: all mutations go through an internal `Mutex`.
///
/// # Backpressure
///
/// When the queue is at capacity, [`submit`](Self::submit) returns
/// [`SearchError::QueueFull`]. Callers should back off and retry.
///
/// # Dedup
///
/// If a document with the same `doc_id` is already pending, the new
/// request replaces it (latest text wins). If the content hash matches
/// a previously embedded version, the job is skipped entirely.
pub struct EmbeddingQueue {
    config: EmbeddingQueueConfig,
    state: Mutex<QueueState>,
    canonicalizer: Box<dyn Canonicalizer>,
    metrics: Arc<QueueMetrics>,
}

impl std::fmt::Debug for EmbeddingQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingQueue")
            .field("config", &self.config)
            .field("pending", &self.pending_count())
            .finish_non_exhaustive()
    }
}

impl EmbeddingQueue {
    /// Create a new queue with the given configuration and canonicalizer.
    #[must_use]
    pub fn new(config: EmbeddingQueueConfig, canonicalizer: Box<dyn Canonicalizer>) -> Self {
        Self {
            state: Mutex::new(QueueState {
                jobs: VecDeque::with_capacity(config.capacity),
                pending_ids: HashMap::new(),
                known_hashes: HashMap::new(),
                known_hash_order: VecDeque::new(),
                sequence: 0,
            }),
            config,
            canonicalizer,
            metrics: Arc::new(QueueMetrics::default()),
        }
    }

    /// Submit a document for embedding.
    ///
    /// The text is canonicalized and content-hashed immediately. If the
    /// canonical text is empty, the request is silently skipped.
    ///
    /// # Dedup behavior
    ///
    /// - Same `doc_id` already pending: the older request is replaced.
    /// - Content hash matches a previously embedded version: skipped.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::QueueFull`] when the queue is at capacity.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[allow(clippy::too_many_lines)]
    pub fn submit(&self, request: EmbeddingRequest) -> SearchResult<JobOutcome> {
        self.metrics.total_submitted.fetch_add(1, Ordering::Relaxed);

        // Canonicalize text
        let canonical = self.canonicalizer.canonicalize(&request.text);
        if canonical.trim().is_empty() {
            debug!(
                target: "frankensearch.queue",
                doc_id = %request.doc_id,
                "skipping empty canonicalized text"
            );
            self.metrics.record(JobOutcome::SkippedEmpty);
            return Ok(JobOutcome::SkippedEmpty);
        }

        // Compute content hash
        let content_hash = sha256_hex(&canonical);
        let EmbeddingRequest {
            doc_id,
            metadata,
            submitted_at,
            ..
        } = request;

        let mut state = self.state.lock().expect("queue lock poisoned");

        // Handle an existing pending job first so "latest submission wins" is honored.
        if let Some(&old_seq) = state.pending_ids.get(&doc_id) {
            if let Some(position) = state.jobs.iter().position(|job| job.doc_id == doc_id) {
                // Duplicate of the pending payload: keep one job but refresh metadata/timestamp.
                if state.jobs[position].content_hash == content_hash {
                    let existing = state
                        .jobs
                        .get_mut(position)
                        .expect("position came from VecDeque::position");
                    existing.metadata = metadata;
                    existing.submitted_at = submitted_at;
                    existing.retry_count = 0;
                    debug!(
                        target: "frankensearch.queue",
                        doc_id = %doc_id,
                        seq = old_seq,
                        "deduped duplicate submission for pending job"
                    );
                    return Ok(JobOutcome::Succeeded);
                }

                // If latest payload matches the already-embedded hash, drop pending update.
                if state
                    .known_hashes
                    .get(&doc_id)
                    .is_some_and(|known| known.hash == content_hash)
                {
                    let _ = state.jobs.remove(position);
                    state.pending_ids.remove(&doc_id);
                    debug!(
                        target: "frankensearch.queue",
                        doc_id = %doc_id,
                        seq = old_seq,
                        "latest submission matches embedded content; removing pending update"
                    );
                    self.metrics.record(JobOutcome::SkippedUnchanged);
                    return Ok(JobOutcome::SkippedUnchanged);
                }

                // Replace pending job with newer content.
                let existing = state
                    .jobs
                    .get_mut(position)
                    .expect("position came from VecDeque::position");
                existing.canonical_text = canonical;
                existing.content_hash = content_hash;
                existing.metadata = metadata;
                existing.submitted_at = submitted_at;
                existing.retry_count = 0;
                debug!(
                    target: "frankensearch.queue",
                    doc_id = %doc_id,
                    seq = old_seq,
                    "replaced pending job with newer text"
                );
                return Ok(JobOutcome::Succeeded);
            }

            // Defensive recovery: pending index said present, queue entry missing.
            state.pending_ids.remove(&doc_id);
            warn!(
                target: "frankensearch.queue",
                doc_id = %doc_id,
                seq = old_seq,
                "recovered stale pending_ids entry without matching queue job"
            );
        }

        // Skip content that is unchanged from the most recently embedded version.
        if state
            .known_hashes
            .get(&doc_id)
            .is_some_and(|known_hash| known_hash.hash == content_hash)
        {
            debug!(
                target: "frankensearch.queue",
                doc_id = %doc_id,
                "skipping unchanged content (hash match)"
            );
            self.metrics.record(JobOutcome::SkippedUnchanged);
            return Ok(JobOutcome::SkippedUnchanged);
        }

        // Check capacity
        if state.jobs.len() >= self.config.capacity {
            return Err(SearchError::QueueFull {
                pending: state.jobs.len(),
                capacity: self.config.capacity,
            });
        }

        let seq = state.sequence;
        state.sequence = state.sequence.wrapping_add(1);

        let job = EmbeddingJob {
            doc_id: doc_id.clone(),
            canonical_text: canonical,
            content_hash,
            metadata,
            submitted_at,
            retry_count: 0,
        };

        state.pending_ids.insert(doc_id, seq);
        state.jobs.push_back(job);

        debug!(
            target: "frankensearch.queue",
            doc_id = %state.jobs.back().unwrap().doc_id,
            pending = state.jobs.len(),
            "job enqueued"
        );

        Ok(JobOutcome::Succeeded)
    }

    fn drain_with_limit(&self, limit: usize) -> Vec<EmbeddingJob> {
        if limit == 0 {
            return Vec::new();
        }

        let mut state = self.state.lock().expect("queue lock poisoned");
        let count = state.jobs.len().min(limit);
        let mut batch = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(job) = state.jobs.pop_front() {
                state.pending_ids.remove(&job.doc_id);
                batch.push(job);
            }
        }

        if !batch.is_empty() {
            self.metrics.total_batches.fetch_add(1, Ordering::Relaxed);
            debug!(
                target: "frankensearch.queue",
                batch_size = batch.len(),
                remaining = state.jobs.len(),
                "drained batch"
            );
        }

        batch
    }

    /// Drain up to `batch_size` jobs from the queue.
    ///
    /// Returns an empty vec if no jobs are pending.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn drain_batch(&self) -> Vec<EmbeddingJob> {
        self.drain_with_limit(self.config.batch_size)
    }

    /// Drain at most `limit` jobs from the queue.
    ///
    /// Useful for consumers that want to cap how many documents they
    /// process without re-enqueuing already fetched work.
    #[must_use]
    pub fn drain_batch_up_to(&self, limit: usize) -> Vec<EmbeddingJob> {
        self.drain_with_limit(limit)
    }

    /// Re-enqueue a failed job for retry (increments retry count).
    ///
    /// If the job has exceeded `max_retries`, it is not re-enqueued and
    /// `JobOutcome::Failed` is returned.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn requeue(&self, mut job: EmbeddingJob) -> JobOutcome {
        job.retry_count += 1;

        if job.retry_count > self.config.max_retries {
            warn!(
                target: "frankensearch.queue",
                doc_id = %job.doc_id,
                retries = job.retry_count,
                "job permanently failed (max retries exceeded)"
            );
            self.metrics.record(JobOutcome::Failed);
            return JobOutcome::Failed;
        }

        let mut state = self.state.lock().expect("queue lock poisoned");

        // If queue is full, drop the retry (backpressure)
        if state.jobs.len() >= self.config.capacity {
            warn!(
                target: "frankensearch.queue",
                doc_id = %job.doc_id,
                "dropping retry: queue full"
            );
            self.metrics.record(JobOutcome::Failed);
            return JobOutcome::Failed;
        }

        // If a newer submission for this doc_id was enqueued while we were
        // processing the batch, drop the stale retry to preserve the dedup
        // invariant (one queue entry per doc_id).
        if state.pending_ids.contains_key(&job.doc_id) {
            debug!(
                target: "frankensearch.queue",
                doc_id = %job.doc_id,
                "dropping retry: newer submission already pending"
            );
            self.metrics.record(JobOutcome::Failed);
            return JobOutcome::Failed;
        }

        // Defensive guard: if bookkeeping drift left `pending_ids` stale,
        // still avoid requeueing a duplicate doc_id into the live queue.
        if state.jobs.iter().any(|queued| queued.doc_id == job.doc_id) {
            warn!(
                target: "frankensearch.queue",
                doc_id = %job.doc_id,
                "dropping retry: duplicate doc_id already present in queue"
            );
            self.metrics.record(JobOutcome::Failed);
            return JobOutcome::Failed;
        }

        let seq = state.sequence;
        state.sequence = state.sequence.wrapping_add(1);
        state.pending_ids.insert(job.doc_id.clone(), seq);
        state.jobs.push_back(job);
        drop(state);

        self.metrics.record(JobOutcome::Retryable);
        JobOutcome::Retryable
    }

    /// Record that a document was successfully embedded with a given content hash.
    ///
    /// Future submissions with the same `doc_id` and hash will be skipped.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn record_embedded(&self, doc_id: &str, content_hash: &str) {
        let mut state = self.state.lock().expect("queue lock poisoned");
        self.record_known_hash_locked(&mut state, doc_id, content_hash);
        self.metrics.record(JobOutcome::Succeeded);
    }

    /// Number of jobs currently pending.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.state.lock().expect("queue lock poisoned").jobs.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending_count() == 0
    }

    /// Queue capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Shared reference to the metrics counters.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<QueueMetrics> {
        &self.metrics
    }

    /// Configuration reference.
    #[must_use]
    pub const fn config(&self) -> &EmbeddingQueueConfig {
        &self.config
    }

    /// Clear all known content hashes (e.g., after a full rebuild).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn clear_known_hashes(&self) {
        let mut state = self.state.lock().expect("queue lock poisoned");
        state.known_hashes.clear();
        state.known_hash_order.clear();
    }

    /// Upper bound for remembered content hashes.
    ///
    /// We keep a multiple of queue capacity so dedup remains effective while
    /// preventing unbounded growth from one-off document IDs.
    const KNOWN_HASH_CAPACITY_FACTOR: usize = 8;

    const fn known_hash_limit(&self) -> usize {
        self.config
            .capacity
            .saturating_mul(Self::KNOWN_HASH_CAPACITY_FACTOR)
    }

    fn record_known_hash_locked(&self, state: &mut QueueState, doc_id: &str, content_hash: &str) {
        let seq = state.sequence;
        state.sequence = state.sequence.wrapping_add(1);

        state.known_hashes.insert(
            doc_id.to_owned(),
            KnownHashEntry {
                hash: content_hash.to_owned(),
                sequence: seq,
            },
        );
        state.known_hash_order.push_back((doc_id.to_owned(), seq));
        self.prune_known_hashes_locked(state);
    }

    fn prune_known_hashes_locked(&self, state: &mut QueueState) {
        let limit = self.known_hash_limit();
        while state.known_hashes.len() > limit {
            let Some((doc_id, seq)) = state.known_hash_order.pop_front() else {
                // Defensive recovery if ordering metadata is lost.
                state.known_hashes.clear();
                break;
            };

            if state
                .known_hashes
                .get(&doc_id)
                .is_some_and(|entry| entry.sequence == seq)
            {
                state.known_hashes.remove(&doc_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute SHA-256 hex digest of a string.
fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    hex_encode(&result)
}

/// Encode bytes as lowercase hex.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{byte:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use frankensearch_core::canonicalize::DefaultCanonicalizer;

    use super::*;

    fn make_queue(capacity: usize) -> EmbeddingQueue {
        EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity,
                batch_size: 32,
                max_retries: 3,
            },
            Box::new(DefaultCanonicalizer::default()),
        )
    }

    fn request(doc_id: &str, text: &str) -> EmbeddingRequest {
        EmbeddingRequest {
            doc_id: doc_id.to_owned(),
            text: text.to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
        }
    }

    // ── Basic submit and drain ──────────────────────────────────────

    #[test]
    fn submit_and_drain_single_job() {
        let queue = make_queue(10);
        let outcome = queue.submit(request("doc-1", "Hello world")).unwrap();
        assert_eq!(outcome, JobOutcome::Succeeded);
        assert_eq!(queue.pending_count(), 1);

        let batch = queue.drain_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].doc_id, "doc-1");
        assert!(!batch[0].canonical_text.is_empty());
        assert!(!batch[0].content_hash.is_empty());
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn submit_multiple_and_drain_in_order() {
        let queue = make_queue(10);
        queue.submit(request("doc-a", "First document")).unwrap();
        queue.submit(request("doc-b", "Second document")).unwrap();
        queue.submit(request("doc-c", "Third document")).unwrap();

        let batch = queue.drain_batch();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].doc_id, "doc-a");
        assert_eq!(batch[1].doc_id, "doc-b");
        assert_eq!(batch[2].doc_id, "doc-c");
    }

    #[test]
    fn drain_respects_batch_size() {
        let queue = EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 100,
                batch_size: 2,
                max_retries: 3,
            },
            Box::new(DefaultCanonicalizer::default()),
        );

        for i in 0..5 {
            queue
                .submit(request(&format!("doc-{i}"), &format!("Content {i}")))
                .unwrap();
        }

        let batch1 = queue.drain_batch();
        assert_eq!(batch1.len(), 2);
        assert_eq!(queue.pending_count(), 3);

        let batch2 = queue.drain_batch();
        assert_eq!(batch2.len(), 2);
        assert_eq!(queue.pending_count(), 1);

        let batch3 = queue.drain_batch();
        assert_eq!(batch3.len(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn drain_empty_queue_returns_empty() {
        let queue = make_queue(10);
        let batch = queue.drain_batch();
        assert!(batch.is_empty());
    }

    // ── Backpressure ────────────────────────────────────────────────

    #[test]
    fn backpressure_when_full() {
        let queue = make_queue(3);
        queue.submit(request("doc-1", "Text one")).unwrap();
        queue.submit(request("doc-2", "Text two")).unwrap();
        queue.submit(request("doc-3", "Text three")).unwrap();

        let err = queue
            .submit(request("doc-4", "Text four"))
            .expect_err("should be full");
        assert!(
            matches!(
                err,
                SearchError::QueueFull {
                    pending: 3,
                    capacity: 3
                }
            ),
            "expected QueueFull, got {err:?}"
        );
    }

    #[test]
    fn drain_frees_capacity() {
        let queue = make_queue(2);
        queue.submit(request("doc-1", "Text one")).unwrap();
        queue.submit(request("doc-2", "Text two")).unwrap();

        // Queue is full
        assert!(queue.submit(request("doc-3", "Text three")).is_err());

        // Drain one batch
        let _ = queue.drain_batch();
        assert!(queue.is_empty());

        // Can submit again
        queue.submit(request("doc-3", "Text three")).unwrap();
        assert_eq!(queue.pending_count(), 1);
    }

    // ── Dedup: same doc_id replaces ─────────────────────────────────

    #[test]
    fn same_doc_id_replaces_pending_job() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Original text")).unwrap();
        queue.submit(request("doc-1", "Updated text")).unwrap();

        // Should still only have 1 pending job
        assert_eq!(queue.pending_count(), 1);

        let batch = queue.drain_batch();
        assert_eq!(batch.len(), 1);
        // Should have the updated text
        assert!(batch[0].canonical_text.contains("Updated"));
    }

    // ── Dedup: content hash skip ────────────────────────────────────

    #[test]
    fn content_hash_skip_when_unchanged() {
        let queue = make_queue(10);

        // Submit and drain a document
        queue.submit(request("doc-1", "Persistent text")).unwrap();
        let batch = queue.drain_batch();
        assert_eq!(batch.len(), 1);

        // Record it as embedded
        queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

        // Submit the same document again with identical text
        let outcome = queue.submit(request("doc-1", "Persistent text")).unwrap();
        assert_eq!(outcome, JobOutcome::SkippedUnchanged);
        assert!(queue.is_empty());
    }

    #[test]
    fn content_hash_skip_does_not_apply_to_changed_text() {
        let queue = make_queue(10);

        // Submit and drain
        queue.submit(request("doc-1", "Original text")).unwrap();
        let batch = queue.drain_batch();
        queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

        // Submit same doc_id with different text
        let outcome = queue
            .submit(request("doc-1", "Completely different text"))
            .unwrap();
        assert_eq!(outcome, JobOutcome::Succeeded);
        assert_eq!(queue.pending_count(), 1);
    }

    // ── Empty text skipping ─────────────────────────────────────────

    #[test]
    fn empty_text_skipped() {
        let queue = make_queue(10);
        let outcome = queue.submit(request("doc-1", "")).unwrap();
        assert_eq!(outcome, JobOutcome::SkippedEmpty);
        assert!(queue.is_empty());
    }

    #[test]
    fn whitespace_only_text_skipped() {
        let queue = make_queue(10);
        let outcome = queue.submit(request("doc-1", "   \n\t  \n  ")).unwrap();
        assert_eq!(outcome, JobOutcome::SkippedEmpty);
        assert!(queue.is_empty());
    }

    // ── Retry logic ─────────────────────────────────────────────────

    #[test]
    fn requeue_increments_retry_count() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Some text")).unwrap();
        let batch = queue.drain_batch();
        assert_eq!(batch[0].retry_count, 0);

        let outcome = queue.requeue(batch.into_iter().next().unwrap());
        assert_eq!(outcome, JobOutcome::Retryable);

        let batch = queue.drain_batch();
        assert_eq!(batch[0].retry_count, 1);
    }

    #[test]
    fn requeue_fails_after_max_retries() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Failing text")).unwrap();

        let mut job = queue.drain_batch().into_iter().next().unwrap();
        job.retry_count = 3; // Already at max

        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Failed);
        assert!(queue.is_empty());
    }

    #[test]
    fn requeue_drops_when_queue_full() {
        let queue = make_queue(1);
        queue
            .submit(request("doc-1", "Occupying the queue"))
            .unwrap();

        let job = EmbeddingJob {
            doc_id: "doc-retry".to_owned(),
            canonical_text: "retry text".to_owned(),
            content_hash: "hash".to_owned(),
            metadata: None,
            submitted_at: Instant::now(),
            retry_count: 0,
        };

        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Failed);
    }

    #[test]
    fn requeue_drops_when_duplicate_doc_id_already_queued_even_with_stale_pending_index() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "stale payload")).unwrap();
        let stale_job = queue.drain_batch().into_iter().next().unwrap();

        queue.submit(request("doc-1", "fresh payload")).unwrap();
        {
            let mut state = queue.state.lock().expect("queue lock poisoned");
            state.pending_ids.clear();
            assert_eq!(state.jobs.len(), 1);
            drop(state);
        }

        let outcome = queue.requeue(stale_job);
        assert_eq!(outcome, JobOutcome::Failed);

        let batch = queue.drain_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].doc_id, "doc-1");
        assert_eq!(batch[0].retry_count, 0);
    }

    // ── Content hash computation ────────────────────────────────────

    #[test]
    fn content_hash_is_deterministic() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Same text")).unwrap();
        queue.submit(request("doc-2", "Same text")).unwrap();

        let batch = queue.drain_batch();
        assert_eq!(batch[0].content_hash, batch[1].content_hash);
    }

    #[test]
    fn content_hash_differs_for_different_text() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Text A")).unwrap();
        queue.submit(request("doc-2", "Text B")).unwrap();

        let batch = queue.drain_batch();
        assert_ne!(batch[0].content_hash, batch[1].content_hash);
    }

    #[test]
    fn sha256_hex_format() {
        let hash = sha256_hex("hello");
        assert_eq!(hash.len(), 64); // 256 bits = 32 bytes = 64 hex chars
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── Metrics ─────────────────────────────────────────────────────

    #[test]
    fn metrics_track_submissions() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Text")).unwrap();
        queue.submit(request("doc-2", "")).unwrap(); // empty -> skipped

        assert_eq!(queue.metrics().total_submitted(), 2);
        assert_eq!(queue.metrics().total_skipped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metrics_track_dedup() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Text")).unwrap();
        let batch = queue.drain_batch();
        queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

        queue.submit(request("doc-1", "Text")).unwrap(); // deduped
        assert_eq!(queue.metrics().total_deduped.load(Ordering::Relaxed), 1);
    }

    // ── Accessors ───────────────────────────────────────────────────

    #[test]
    fn capacity_accessor() {
        let queue = make_queue(42);
        assert_eq!(queue.capacity(), 42);
    }

    #[test]
    fn clear_known_hashes_allows_re_embedding() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Text")).unwrap();
        let batch = queue.drain_batch();
        queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

        // Would normally skip
        let outcome = queue.submit(request("doc-1", "Text")).unwrap();
        assert_eq!(outcome, JobOutcome::SkippedUnchanged);

        // Clear hashes
        queue.clear_known_hashes();

        // Now it should enqueue again
        let outcome = queue.submit(request("doc-1", "Text")).unwrap();
        assert_eq!(outcome, JobOutcome::Succeeded);
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn latest_submission_matching_embedded_hash_cancels_pending_update() {
        let queue = make_queue(10);

        queue.submit(request("doc-1", "Original text")).unwrap();
        let batch = queue.drain_batch();
        queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);

        // Enqueue a changed version.
        let outcome = queue.submit(request("doc-1", "Changed text")).unwrap();
        assert_eq!(outcome, JobOutcome::Succeeded);
        assert_eq!(queue.pending_count(), 1);

        // Submit latest text matching embedded content; should remove pending update.
        let outcome = queue.submit(request("doc-1", "Original text")).unwrap();
        assert_eq!(outcome, JobOutcome::SkippedUnchanged);
        assert!(queue.is_empty());
    }

    #[test]
    fn known_hash_cache_is_pruned_and_reembeds_evicted_entries() {
        // capacity=1 => known hash cache limit is 8 entries.
        let queue = make_queue(1);

        for i in 0..12 {
            queue
                .submit(request(&format!("doc-{i}"), "Stable payload"))
                .unwrap();
            let batch = queue.drain_batch();
            queue.record_embedded(&batch[0].doc_id, &batch[0].content_hash);
        }

        // doc-0 should have been pruned from known hashes and thus re-enqueued.
        let outcome = queue.submit(request("doc-0", "Stable payload")).unwrap();
        assert_eq!(outcome, JobOutcome::Succeeded);
    }

    #[test]
    fn zero_capacity_queue_immediately_full() {
        let queue = make_queue(0);
        let err = queue
            .submit(request("doc-1", "Any text"))
            .expect_err("should be full");
        assert!(matches!(
            err,
            SearchError::QueueFull {
                pending: 0,
                capacity: 0
            }
        ));
    }

    #[test]
    fn metadata_preserved_through_pipeline() {
        let queue = make_queue(10);
        let mut req = request("doc-1", "Some real content");
        req.metadata = Some(serde_json::json!({"source": "test", "page": 42}));
        queue.submit(req).unwrap();

        let batch = queue.drain_batch();
        let meta = batch[0].metadata.as_ref().expect("metadata should exist");
        assert_eq!(meta["source"], "test");
        assert_eq!(meta["page"], 42);
    }

    #[test]
    fn requeue_at_boundary_succeeds_then_fails() {
        let queue = EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 100,
                batch_size: 32,
                max_retries: 2,
            },
            Box::new(DefaultCanonicalizer::default()),
        );
        queue.submit(request("doc-1", "Text")).unwrap();
        let mut job = queue.drain_batch().into_iter().next().unwrap();
        assert_eq!(job.retry_count, 0);

        // First requeue: retry_count becomes 1 (< max_retries=2)
        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Retryable);

        job = queue.drain_batch().into_iter().next().unwrap();
        assert_eq!(job.retry_count, 1);

        // Second requeue: retry_count becomes 2 (== max_retries=2)
        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Retryable);

        job = queue.drain_batch().into_iter().next().unwrap();
        assert_eq!(job.retry_count, 2);

        // Third requeue: retry_count becomes 3 (> max_retries=2) -> Failed
        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Failed);
    }

    #[test]
    fn drain_empty_twice_is_stable() {
        let queue = make_queue(10);
        let b1 = queue.drain_batch();
        let b2 = queue.drain_batch();
        assert!(b1.is_empty());
        assert!(b2.is_empty());
        assert!(queue.is_empty());
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn pending_count_and_is_empty_are_consistent() {
        let queue = make_queue(10);
        assert!(queue.is_empty());
        assert_eq!(queue.pending_count(), 0);

        queue.submit(request("doc-1", "Content")).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.pending_count(), 1);

        let _ = queue.drain_batch();
        assert!(queue.is_empty());
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn metrics_track_retries_and_failures() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Text")).unwrap();
        let job = queue.drain_batch().into_iter().next().unwrap();

        // Requeue -> Retryable
        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Retryable);
        assert_eq!(queue.metrics().total_retryable.load(Ordering::Relaxed), 1);

        // Drain and exceed max retries
        let mut job = queue.drain_batch().into_iter().next().unwrap();
        job.retry_count = 3; // at max
        let outcome = queue.requeue(job);
        assert_eq!(outcome, JobOutcome::Failed);
        assert_eq!(queue.metrics().total_failed.load(Ordering::Relaxed), 1);
    }

    // ─── bd-yvte tests begin ───

    #[test]
    fn embedding_queue_config_default_exact_values() {
        let config = EmbeddingQueueConfig::default();
        assert_eq!(config.capacity, 1000);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn embedding_queue_debug_format() {
        let queue = make_queue(42);
        queue.submit(request("doc-1", "Some content")).unwrap();
        let dbg = format!("{queue:?}");
        assert!(dbg.contains("EmbeddingQueue"));
        assert!(dbg.contains("42")); // capacity in config
        assert!(dbg.contains('1')); // pending count
    }

    #[test]
    fn config_accessor_returns_construction_config() {
        let queue = EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 77,
                batch_size: 11,
                max_retries: 5,
            },
            Box::new(DefaultCanonicalizer::default()),
        );
        assert_eq!(queue.config().capacity, 77);
        assert_eq!(queue.config().batch_size, 11);
        assert_eq!(queue.config().max_retries, 5);
    }

    #[test]
    fn drain_batch_up_to_limits_correctly() {
        let queue = make_queue(100);
        for i in 0..10 {
            queue
                .submit(request(&format!("doc-{i}"), &format!("Content {i}")))
                .unwrap();
        }

        let batch = queue.drain_batch_up_to(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(queue.pending_count(), 7);

        let batch = queue.drain_batch_up_to(100);
        assert_eq!(batch.len(), 7);
        assert!(queue.is_empty());
    }

    #[test]
    fn drain_batch_up_to_zero_returns_empty() {
        let queue = make_queue(10);
        queue.submit(request("doc-1", "Content")).unwrap();
        let batch = queue.drain_batch_up_to(0);
        assert!(batch.is_empty());
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn metrics_track_batch_count() {
        let queue = EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity: 100,
                batch_size: 2,
                max_retries: 3,
            },
            Box::new(DefaultCanonicalizer::default()),
        );
        for i in 0..5 {
            queue
                .submit(request(&format!("doc-{i}"), &format!("Text {i}")))
                .unwrap();
        }

        let _ = queue.drain_batch(); // batch 1 (2 items)
        let _ = queue.drain_batch(); // batch 2 (2 items)
        let _ = queue.drain_batch(); // batch 3 (1 item)

        assert_eq!(queue.metrics().total_batches.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn queue_metrics_record_all_outcomes() {
        let metrics = QueueMetrics::default();
        metrics.record(JobOutcome::Succeeded);
        metrics.record(JobOutcome::SkippedUnchanged);
        metrics.record(JobOutcome::SkippedEmpty);
        metrics.record(JobOutcome::Retryable);
        metrics.record(JobOutcome::Failed);

        assert_eq!(metrics.total_succeeded.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_deduped.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_skipped.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_retryable.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.total_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn job_outcome_copy_and_eq() {
        let a = JobOutcome::Succeeded;
        let b = a; // Copy
        assert_eq!(a, b);

        // All variants are distinct.
        let variants = [
            JobOutcome::Succeeded,
            JobOutcome::SkippedUnchanged,
            JobOutcome::SkippedEmpty,
            JobOutcome::Retryable,
            JobOutcome::Failed,
        ];
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2);
                } else {
                    assert_ne!(v1, v2);
                }
            }
        }
    }

    #[test]
    fn same_doc_id_same_content_refreshes_metadata() {
        let queue = make_queue(10);
        let mut req1 = request("doc-1", "Same text");
        req1.metadata = Some(serde_json::json!({"version": 1}));
        queue.submit(req1).unwrap();

        // Submit again with same text but different metadata.
        let mut req2 = request("doc-1", "Same text");
        req2.metadata = Some(serde_json::json!({"version": 2}));
        queue.submit(req2).unwrap();

        assert_eq!(queue.pending_count(), 1);
        let batch = queue.drain_batch();
        let meta = batch[0].metadata.as_ref().expect("metadata");
        assert_eq!(meta["version"], 2);
    }

    #[test]
    fn sha256_hex_known_value() {
        // SHA-256 of "hello" is well-known.
        let hash = sha256_hex("hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    // ─── bd-yvte tests end ───
}
