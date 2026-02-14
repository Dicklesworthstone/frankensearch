//! Embedding batch coalescing with deadline-aware scheduling.
//!
//! When multiple concurrent callers request embeddings (e.g., during index
//! building or concurrent searches), [`BatchCoalescer`] groups their requests
//! into optimal batches for [`Embedder::embed_batch()`]. This amortises the
//! high fixed overhead of ONNX model inference across many inputs.
//!
//! # Scheduling algorithm
//!
//! 1. Requests arrive via [`BatchCoalescer::submit`] with a [`Priority`] level.
//! 2. **Interactive** requests trigger early dispatch: if the oldest pending
//!    request has waited longer than `max_wait_ms / 2`, the batch fires.
//! 3. **Background** requests accumulate until `max_batch_size` or `max_wait_ms`.
//! 4. Mixed batches use the tightest deadline across all pending requests.
//! 5. The formed [`CoalescedBatch`] is dispatched to `Embedder::embed_batch()`.
//!
//! # Performance model
//!
//! - `FastEmbed` (MiniLM-L6-v2): 128 ms for 1 text, ~140 ms for 32 texts
//!   = 4.4 ms/text batched vs 128 ms/text unbatched = **29× throughput**.
//! - `Model2Vec` (potion): 0.57 ms for 1 text, ~2 ms for 32 texts
//!   = 0.06 ms/text batched.
//!
//! # Thread model
//!
//! The coalescer uses [`std::sync::Mutex`] + [`std::sync::Condvar`] for
//! coordination. The dispatch loop ([`BatchCoalescer::wait_for_batch`])
//! blocks the calling thread until batch conditions are met. The consumer
//! is responsible for running the dispatch loop on an appropriate thread
//! (rayon, asupersync `scope.spawn`, etc.) and bridging into the async
//! runtime for the actual `embed_batch()` call.
//!
//! Result delivery uses [`std::sync::mpsc::sync_channel`] (capacity 1)
//! per request — **not** tokio channels.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`BatchCoalescer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalescerConfig {
    /// Maximum texts per batch. Default: 32.
    pub max_batch_size: usize,
    /// Maximum time (ms) to wait for a batch to fill. Default: 10.
    pub max_wait_ms: u64,
    /// Minimum batch size before time-based dispatch fires. Default: 4.
    pub min_batch_size: usize,
    /// Enable Interactive/Background priority separation. Default: true.
    pub use_priority_lanes: bool,
}

impl Default for CoalescerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 10,
            min_batch_size: 4,
            use_priority_lanes: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Priority level for an embedding request.
///
/// Interactive requests have tighter latency budgets and trigger early
/// batch dispatch. Background requests (e.g., index building) can wait
/// for fuller batches to maximise throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Search query path — tight deadline (~15 ms budget).
    Interactive,
    /// Index building path — can wait for full batch.
    Background,
}

// ---------------------------------------------------------------------------
// Internal request
// ---------------------------------------------------------------------------

/// A pending embedding request waiting to be batched.
struct PendingRequest {
    /// Text to embed.
    text: String,
    /// Caller priority.
    priority: Priority,
    /// Absolute deadline: the request MUST be dispatched by this instant.
    deadline: Instant,
    /// When the request was submitted (for wait-time tracking).
    submitted_at: Instant,
    /// Channel for delivering the result back to the caller.
    result_tx: SyncSender<SearchResult<Vec<f32>>>,
}

// ---------------------------------------------------------------------------
// Coalesced batch
// ---------------------------------------------------------------------------

/// A formed batch of embedding requests ready for dispatch.
///
/// The consumer extracts texts via [`texts()`](Self::texts), calls
/// `Embedder::embed_batch()`, then delivers results via
/// [`deliver()`](Self::deliver).
pub struct CoalescedBatch {
    requests: Vec<PendingRequest>,
}

impl CoalescedBatch {
    /// Borrow the texts to embed, in submission order.
    #[must_use]
    pub fn texts(&self) -> Vec<&str> {
        self.requests.iter().map(|r| r.text.as_str()).collect()
    }

    /// Number of texts in the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Whether the batch contains any interactive-priority requests.
    #[must_use]
    pub fn has_interactive(&self) -> bool {
        self.requests
            .iter()
            .any(|r| r.priority == Priority::Interactive)
    }

    /// Deliver embedding results to the callers who submitted requests.
    ///
    /// On success, each caller receives its corresponding embedding vector.
    /// On error, every caller receives a descriptive error.
    ///
    /// Callers whose receivers have been dropped are silently skipped.
    pub fn deliver(self, results: SearchResult<Vec<Vec<f32>>>) {
        match results {
            Ok(vectors) => {
                // Zip results with senders. If lengths don't match (shouldn't
                // happen), extra callers get an error, extra vectors are dropped.
                let mut vec_iter = vectors.into_iter();
                for req in self.requests {
                    let result = vec_iter.next().map_or_else(
                        || {
                            Err(SearchError::EmbeddingFailed {
                                model: "batch_coalescer".into(),
                                source: "batch result count mismatch".into(),
                            })
                        },
                        Ok,
                    );
                    let _ = req.result_tx.send(result);
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for req in self.requests {
                    let _ = req.result_tx.send(Err(SearchError::EmbeddingFailed {
                        model: "batch_coalescer".into(),
                        source: msg.clone().into(),
                    }));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Lock-free counters for batch coalescer telemetry.
#[derive(Debug, Default)]
pub struct CoalescerMetrics {
    /// Total individual embedding requests submitted.
    pub total_submitted: AtomicU64,
    /// Total batches formed and dispatched.
    pub total_batches: AtomicU64,
    /// Total texts included in dispatched batches.
    pub total_texts_batched: AtomicU64,
    /// Interactive-priority submissions.
    pub interactive_submissions: AtomicU64,
    /// Background-priority submissions.
    pub background_submissions: AtomicU64,
    /// Batches dispatched early due to interactive priority.
    pub early_dispatches: AtomicU64,
    /// Batches dispatched because a request hit its deadline.
    pub deadline_dispatches: AtomicU64,
    /// Batches dispatched because `max_batch_size` was reached.
    pub full_batch_dispatches: AtomicU64,
    /// Batches dispatched on timeout (`max_wait_ms` elapsed).
    pub timeout_dispatches: AtomicU64,
}

impl CoalescerMetrics {
    /// Average batch size (0.0 if no batches).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // telemetry counters won't reach 2^52
    pub fn avg_batch_size(&self) -> f64 {
        let batches = self.total_batches.load(Ordering::Relaxed);
        if batches == 0 {
            return 0.0;
        }
        self.total_texts_batched.load(Ordering::Relaxed) as f64 / batches as f64
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Mutable state behind the coalescer's mutex.
struct CoalescerState {
    /// Pending requests in submission order.
    pending: VecDeque<PendingRequest>,
    /// Whether `shutdown()` has been called.
    shutdown: bool,
}

// ---------------------------------------------------------------------------
// Batch dispatch reason (for metrics)
// ---------------------------------------------------------------------------

/// Why a batch was dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchReason {
    /// Batch reached `max_batch_size`.
    Full,
    /// A request hit its absolute deadline.
    Deadline,
    /// Interactive priority triggered early dispatch.
    InteractiveEarly,
    /// `max_wait_ms` elapsed with pending requests.
    Timeout,
    /// Shutdown: drain remaining requests.
    Shutdown,
}

// ---------------------------------------------------------------------------
// BatchCoalescer
// ---------------------------------------------------------------------------

/// Groups concurrent embedding requests into optimal batches.
///
/// See the [module-level documentation](self) for the scheduling algorithm
/// and performance model.
///
/// # Usage
///
/// ```ignore
/// let coalescer = BatchCoalescer::new(CoalescerConfig::default());
///
/// // On a worker thread:
/// while let Some(batch) = coalescer.wait_for_batch() {
///     let texts = batch.texts();
///     let results = embedder.embed_batch(cx, &texts).await;
///     batch.deliver(results);
/// }
///
/// // From any thread/task:
/// let rx = coalescer.submit("hello world".into(), Priority::Interactive);
/// let embedding = rx.recv().unwrap()?;
/// ```
pub struct BatchCoalescer {
    config: CoalescerConfig,
    state: Mutex<CoalescerState>,
    notify: Condvar,
    metrics: Arc<CoalescerMetrics>,
}

impl std::fmt::Debug for BatchCoalescer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchCoalescer")
            .field("config", &self.config)
            .field("pending", &self.pending_count())
            .finish_non_exhaustive()
    }
}

impl BatchCoalescer {
    /// Create a new batch coalescer with the given configuration.
    #[must_use]
    pub fn new(config: CoalescerConfig) -> Self {
        let mut config = config;
        if config.max_batch_size == 0 {
            warn!(
                target: "frankensearch.coalescer",
                "invalid max_batch_size=0; clamping to 1"
            );
            config.max_batch_size = 1;
        }

        Self {
            state: Mutex::new(CoalescerState {
                pending: VecDeque::with_capacity(config.max_batch_size),
                shutdown: false,
            }),
            notify: Condvar::new(),
            metrics: Arc::new(CoalescerMetrics::default()),
            config,
        }
    }

    // ── Submit ───────────────────────────────────────────────────────

    /// Submit a text for coalesced embedding.
    ///
    /// Returns a receiver that will contain the embedding result once the
    /// batch containing this request is processed. The caller should call
    /// `.recv()` (blocking) or `.try_recv()` (non-blocking) on the receiver.
    ///
    /// The `priority` controls scheduling: [`Priority::Interactive`] requests
    /// trigger early batch dispatch to meet latency SLOs.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn submit(
        &self,
        text: String,
        priority: Priority,
    ) -> mpsc::Receiver<SearchResult<Vec<f32>>> {
        let (tx, rx) = mpsc::sync_channel(1);
        let now = Instant::now();

        let deadline = now
            + Duration::from_millis(match priority {
                Priority::Interactive => self.config.max_wait_ms / 2,
                Priority::Background => self.config.max_wait_ms,
            });

        let request = PendingRequest {
            text,
            priority,
            deadline,
            submitted_at: now,
            result_tx: tx,
        };

        // Track metrics
        self.metrics.total_submitted.fetch_add(1, Ordering::Relaxed);
        match priority {
            Priority::Interactive => {
                self.metrics
                    .interactive_submissions
                    .fetch_add(1, Ordering::Relaxed);
            }
            Priority::Background => {
                self.metrics
                    .background_submissions
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        {
            let mut state = self.state.lock().expect("coalescer lock poisoned");
            state.pending.push_back(request);
            trace!(
                target: "frankensearch.coalescer",
                pending = state.pending.len(),
                ?priority,
                "request submitted"
            );
        }

        self.notify.notify_all();
        rx
    }

    // ── Batch formation ──────────────────────────────────────────────

    /// Blocking wait for a batch to become ready.
    ///
    /// Returns `Some(batch)` when scheduling conditions are met, or
    /// `None` when [`shutdown()`](Self::shutdown) has been called and no
    /// pending requests remain.
    ///
    /// This method blocks the calling thread (via [`Condvar::wait_timeout`])
    /// and should be run on a dedicated worker thread.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn wait_for_batch(&self) -> Option<CoalescedBatch> {
        let mut state = self.state.lock().expect("coalescer lock poisoned");

        loop {
            // Shutdown with empty queue → done
            if state.shutdown && state.pending.is_empty() {
                return None;
            }

            // Check if a batch is ready
            if let Some(reason) = self.batch_ready_reason(&state) {
                let batch = self.form_batch(&mut state, reason);
                return Some(batch);
            }

            // Nothing ready yet → wait
            let timeout = self.next_timeout(&state);
            let (new_state, _timeout_result) = self
                .notify
                .wait_timeout(state, timeout)
                .expect("coalescer lock poisoned");
            state = new_state;
        }
    }

    /// Non-blocking batch formation.
    ///
    /// Returns `Some(batch)` if scheduling conditions are currently met,
    /// `None` otherwise. Does not wait.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn try_form_batch(&self) -> Option<CoalescedBatch> {
        let mut state = self.state.lock().expect("coalescer lock poisoned");
        let reason = self.batch_ready_reason(&state)?;
        let batch = self.form_batch(&mut state, reason);
        drop(state);
        Some(batch)
    }

    // ── Shutdown ─────────────────────────────────────────────────────

    /// Signal the coalescer to shut down.
    ///
    /// After this call, [`wait_for_batch`](Self::wait_for_batch) will drain
    /// any remaining pending requests as a final batch and then return `None`.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn shutdown(&self) {
        let mut state = self.state.lock().expect("coalescer lock poisoned");
        state.shutdown = true;
        debug!(
            target: "frankensearch.coalescer",
            pending = state.pending.len(),
            "shutdown requested"
        );
        drop(state);
        self.notify.notify_all();
    }

    /// Whether shutdown has been requested.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.state.lock().expect("coalescer lock poisoned").shutdown
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Number of requests currently pending.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.state
            .lock()
            .expect("coalescer lock poisoned")
            .pending
            .len()
    }

    /// Shared reference to the metrics counters.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<CoalescerMetrics> {
        &self.metrics
    }

    /// Configuration reference.
    #[must_use]
    pub const fn config(&self) -> &CoalescerConfig {
        &self.config
    }

    // ── Internal: scheduling logic ───────────────────────────────────

    /// Determine whether a batch should be dispatched, and why.
    fn batch_ready_reason(&self, state: &CoalescerState) -> Option<DispatchReason> {
        if state.pending.is_empty() {
            return None;
        }

        // Shutdown with pending requests → drain everything
        if state.shutdown {
            return Some(DispatchReason::Shutdown);
        }

        let now = Instant::now();
        let len = state.pending.len();

        // Rule 1: batch is full
        if len >= self.config.max_batch_size {
            return Some(DispatchReason::Full);
        }

        // Rule 2: interactive priority triggers early dispatch.
        // Checked before general deadline so interactive requests get the
        // specific `InteractiveEarly` reason (their deadline == max_wait_ms/2).
        if self.config.use_priority_lanes {
            let has_interactive = state
                .pending
                .iter()
                .any(|r| r.priority == Priority::Interactive);
            if has_interactive && let Some(oldest) = state.pending.front() {
                let waited = now.saturating_duration_since(oldest.submitted_at);
                if waited >= Duration::from_millis(self.config.max_wait_ms / 2) {
                    return Some(DispatchReason::InteractiveEarly);
                }
            }
        }

        // Rule 3: timeout with at least min_batch_size pending.
        // Checked before general deadline so background batches that meet
        // min_batch_size get the `Timeout` reason rather than `Deadline`.
        if len >= self.config.min_batch_size
            && let Some(oldest) = state.pending.front()
        {
            let waited = now.saturating_duration_since(oldest.submitted_at);
            if waited >= Duration::from_millis(self.config.max_wait_ms) {
                return Some(DispatchReason::Timeout);
            }
        }

        // Rule 4: any request has passed its absolute deadline
        if state.pending.iter().any(|r| now >= r.deadline) {
            return Some(DispatchReason::Deadline);
        }

        None
    }

    /// Calculate the duration to wait before re-checking batch conditions.
    fn next_timeout(&self, state: &CoalescerState) -> Duration {
        if state.pending.is_empty() {
            // Nothing pending — wait for a submit to wake us
            return Duration::from_millis(self.config.max_wait_ms);
        }

        let now = Instant::now();

        // Find the earliest deadline among pending requests
        let earliest_deadline = state
            .pending
            .iter()
            .map(|r| r.deadline)
            .min()
            .unwrap_or(now);

        if earliest_deadline <= now {
            // Already past deadline — wake immediately
            return Duration::ZERO;
        }

        // Wait until the earliest deadline, but cap at max_wait_ms
        let until_deadline = earliest_deadline.saturating_duration_since(now);
        let max_wait = Duration::from_millis(self.config.max_wait_ms);
        until_deadline.min(max_wait)
    }

    /// Drain pending requests into a batch (up to `max_batch_size`).
    fn form_batch(&self, state: &mut CoalescerState, reason: DispatchReason) -> CoalescedBatch {
        let count = state.pending.len().min(self.config.max_batch_size);
        let mut requests = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(req) = state.pending.pop_front() {
                requests.push(req);
            }
        }

        // Update metrics
        self.metrics.total_batches.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_texts_batched
            .fetch_add(requests.len() as u64, Ordering::Relaxed);

        match reason {
            DispatchReason::Full => {
                self.metrics
                    .full_batch_dispatches
                    .fetch_add(1, Ordering::Relaxed);
            }
            DispatchReason::Deadline => {
                self.metrics
                    .deadline_dispatches
                    .fetch_add(1, Ordering::Relaxed);
            }
            DispatchReason::InteractiveEarly => {
                self.metrics
                    .early_dispatches
                    .fetch_add(1, Ordering::Relaxed);
            }
            DispatchReason::Timeout | DispatchReason::Shutdown => {
                self.metrics
                    .timeout_dispatches
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        debug!(
            target: "frankensearch.coalescer",
            batch_size = requests.len(),
            remaining = state.pending.len(),
            ?reason,
            "batch formed"
        );

        CoalescedBatch { requests }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    // ── Config defaults ──────────────────────────────────────────────

    #[test]
    fn default_config() {
        let config = CoalescerConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_wait_ms, 10);
        assert_eq!(config.min_batch_size, 4);
        assert!(config.use_priority_lanes);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = CoalescerConfig {
            max_batch_size: 16,
            max_wait_ms: 20,
            min_batch_size: 2,
            use_priority_lanes: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CoalescerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_batch_size, 16);
        assert_eq!(decoded.max_wait_ms, 20);
        assert_eq!(decoded.min_batch_size, 2);
        assert!(!decoded.use_priority_lanes);
    }

    #[test]
    fn zero_max_batch_size_is_clamped_to_one() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 0,
            max_wait_ms: 50,
            min_batch_size: 1,
            use_priority_lanes: true,
        });
        assert_eq!(coalescer.config().max_batch_size, 1);

        let _rx = coalescer.submit("hello".into(), Priority::Background);
        let batch = coalescer.try_form_batch().expect("batch formed");
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    // ── Priority serde ───────────────────────────────────────────────

    #[test]
    fn priority_serde_roundtrip() {
        let json = serde_json::to_string(&Priority::Interactive).unwrap();
        let decoded: Priority = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, Priority::Interactive);

        let json = serde_json::to_string(&Priority::Background).unwrap();
        let decoded: Priority = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, Priority::Background);
    }

    // ── Single request dispatch ──────────────────────────────────────

    #[test]
    fn single_request_dispatched_within_max_wait() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 50,
            min_batch_size: 1,
            use_priority_lanes: true,
        });

        let rx = coalescer.submit("hello world".into(), Priority::Background);

        // Wait for deadline to pass
        thread::sleep(Duration::from_millis(60));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.texts(), vec!["hello world"]);
        assert!(!batch.has_interactive());

        // Deliver results
        batch.deliver(Ok(vec![vec![1.0, 2.0, 3.0]]));
        let result = rx.recv().unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    // ── Full batch dispatch ──────────────────────────────────────────

    #[test]
    fn full_batch_dispatched_immediately() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 3,
            max_wait_ms: 1000, // Long timeout — shouldn't matter
            min_batch_size: 1,
            use_priority_lanes: true,
        });

        let rx1 = coalescer.submit("text one".into(), Priority::Background);
        let rx2 = coalescer.submit("text two".into(), Priority::Background);
        let rx3 = coalescer.submit("text three".into(), Priority::Background);

        // Batch should be ready immediately (3 = max_batch_size)
        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.texts(), vec!["text one", "text two", "text three"]);

        batch.deliver(Ok(vec![vec![1.0], vec![2.0], vec![3.0]]));

        assert_eq!(rx1.recv().unwrap().unwrap(), vec![1.0]);
        assert_eq!(rx2.recv().unwrap().unwrap(), vec![2.0]);
        assert_eq!(rx3.recv().unwrap().unwrap(), vec![3.0]);

        // Metrics
        assert_eq!(
            coalescer
                .metrics()
                .full_batch_dispatches
                .load(Ordering::Relaxed),
            1
        );
    }

    // ── Interactive priority triggers early dispatch ─────────────────

    #[test]
    fn interactive_triggers_early_dispatch() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 100,
            min_batch_size: 4,
            use_priority_lanes: true,
        });

        // Submit an interactive request
        let _rx = coalescer.submit("urgent query".into(), Priority::Interactive);

        // Wait for max_wait_ms / 2 = 50ms
        thread::sleep(Duration::from_millis(55));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.len(), 1);
        assert!(batch.has_interactive());

        batch.deliver(Ok(vec![vec![42.0]]));

        assert_eq!(
            coalescer.metrics().early_dispatches.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            coalescer
                .metrics()
                .interactive_submissions
                .load(Ordering::Relaxed),
            1
        );
    }

    // ── Deadline enforcement ─────────────────────────────────────────

    #[test]
    fn deadline_enforcement() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 30,
            min_batch_size: 10, // high min so timeout rule doesn't fire first
            use_priority_lanes: false,
        });

        let _rx = coalescer.submit("waiting text".into(), Priority::Background);

        // Not ready yet
        assert!(coalescer.try_form_batch().is_none());

        // Wait past the background deadline (max_wait_ms = 30ms)
        thread::sleep(Duration::from_millis(35));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 1);

        assert_eq!(
            coalescer
                .metrics()
                .deadline_dispatches
                .load(Ordering::Relaxed),
            1
        );
    }

    // ── Mixed priority batch ─────────────────────────────────────────

    #[test]
    fn mixed_priority_batch() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 100,
            min_batch_size: 1,
            use_priority_lanes: true,
        });

        let _rx1 = coalescer.submit("bg task".into(), Priority::Background);
        let _rx2 = coalescer.submit("urgent query".into(), Priority::Interactive);

        // Interactive deadline is max_wait_ms / 2 = 50ms
        thread::sleep(Duration::from_millis(55));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        // Both should be in the batch
        assert_eq!(batch.len(), 2);
        assert!(batch.has_interactive());
    }

    // ── Timeout dispatch with min_batch_size ─────────────────────────

    #[test]
    fn timeout_dispatch_with_min_batch() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 30,
            min_batch_size: 2,
            use_priority_lanes: false,
        });

        // Submit 2 requests (= min_batch_size) but don't fill max
        let _rx1 = coalescer.submit("text a".into(), Priority::Background);
        let _rx2 = coalescer.submit("text b".into(), Priority::Background);

        // Not ready yet (min_batch_size met but max_wait_ms not elapsed)
        assert!(coalescer.try_form_batch().is_none());

        thread::sleep(Duration::from_millis(35));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);

        assert_eq!(
            coalescer
                .metrics()
                .timeout_dispatches
                .load(Ordering::Relaxed),
            1
        );
    }

    // ── Below min_batch_size waits for deadline ──────────────────────

    #[test]
    fn below_min_batch_waits_for_deadline() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 50,
            min_batch_size: 4,
            use_priority_lanes: false,
        });

        // Submit 2 requests (below min_batch_size of 4)
        let _rx1 = coalescer.submit("text a".into(), Priority::Background);
        let _rx2 = coalescer.submit("text b".into(), Priority::Background);

        // After max_wait_ms, timeout rule doesn't fire (below min_batch_size)
        // but deadline rule fires
        thread::sleep(Duration::from_millis(55));

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        // Dispatched by deadline, not timeout
        assert_eq!(
            coalescer
                .metrics()
                .deadline_dispatches
                .load(Ordering::Relaxed),
            1
        );
    }

    // ── Shutdown drains remaining ────────────────────────────────────

    #[test]
    fn shutdown_drains_remaining() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 10_000, // Very long — shouldn't trigger
            min_batch_size: 32,
            use_priority_lanes: false,
        });

        let _rx1 = coalescer.submit("remaining 1".into(), Priority::Background);
        let _rx2 = coalescer.submit("remaining 2".into(), Priority::Background);

        // Not ready under normal rules
        assert!(coalescer.try_form_batch().is_none());

        // Shutdown
        coalescer.shutdown();
        assert!(coalescer.is_shutdown());

        // Now should drain
        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);

        // Next call returns None (empty after drain)
        assert!(coalescer.try_form_batch().is_none());
    }

    // ── wait_for_batch with shutdown ─────────────────────────────────

    #[test]
    fn wait_for_batch_returns_none_on_shutdown() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig::default()));

        let c = Arc::clone(&coalescer);
        let handle = thread::spawn(move || c.wait_for_batch());

        // Give the worker time to block
        thread::sleep(Duration::from_millis(20));
        coalescer.shutdown();

        let result = handle.join().unwrap();
        assert!(result.is_none());
    }

    // ── wait_for_batch dispatches on full batch ──────────────────────

    #[test]
    fn wait_for_batch_dispatches_full_batch() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 2,
            max_wait_ms: 10_000,
            min_batch_size: 2,
            use_priority_lanes: false,
        }));

        let c = Arc::clone(&coalescer);
        let handle = thread::spawn(move || c.wait_for_batch());

        // Submit enough to fill the batch
        thread::sleep(Duration::from_millis(5));
        coalescer.submit("text a".into(), Priority::Background);
        coalescer.submit("text b".into(), Priority::Background);

        let batch = handle.join().unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
    }

    // ── Concurrent submitters ────────────────────────────────────────

    #[test]
    fn concurrent_callers_receive_correct_results() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 4,
            max_wait_ms: 100,
            min_batch_size: 4,
            use_priority_lanes: false,
        }));

        // Spawn 4 concurrent submitters
        let mut receivers = Vec::new();
        let mut handles = Vec::new();

        for i in 0..4 {
            let c = Arc::clone(&coalescer);
            let (done_tx, done_rx) = mpsc::sync_channel(1);
            let handle = thread::spawn(move || {
                let rx = c.submit(format!("text-{i}"), Priority::Background);
                done_tx.send(rx).unwrap();
            });
            handles.push(handle);
            receivers.push(done_rx);
        }

        // Collect all submit receivers
        for h in handles {
            h.join().unwrap();
        }
        let rxs: Vec<_> = receivers.into_iter().map(|r| r.recv().unwrap()).collect();

        // Form and deliver the batch
        // Wait briefly for all submissions to land
        thread::sleep(Duration::from_millis(10));
        let batch = coalescer.try_form_batch().unwrap();
        assert_eq!(batch.len(), 4);

        // Generate distinct result vectors matching batch size
        let results = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        batch.deliver(Ok(results));

        // Each caller gets exactly one result (thread ordering is
        // non-deterministic, so we just verify all succeed with len 1)
        let mut received_values: Vec<f32> = Vec::new();
        for rx in rxs {
            let result = rx.recv().unwrap().unwrap();
            assert_eq!(result.len(), 1);
            received_values.push(result[0]);
        }
        // All 4 callers received results
        assert_eq!(received_values.len(), 4);
    }

    // ── Error delivery ───────────────────────────────────────────────

    #[test]
    fn error_delivered_to_all_callers() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 2,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: false,
        });

        let rx1 = coalescer.submit("text a".into(), Priority::Background);
        let rx2 = coalescer.submit("text b".into(), Priority::Background);

        let batch = coalescer.try_form_batch().unwrap();
        batch.deliver(Err(SearchError::EmbeddingFailed {
            model: "test".into(),
            source: "onnx crashed".into(),
        }));

        let r1 = rx1.recv().unwrap();
        let r2 = rx2.recv().unwrap();
        assert!(r1.is_err());
        assert!(r2.is_err());
        assert!(r1.unwrap_err().to_string().contains("onnx crashed"));
        assert!(r2.unwrap_err().to_string().contains("onnx crashed"));
    }

    // ── Result count mismatch ────────────────────────────────────────

    #[test]
    fn result_count_mismatch_sends_error_for_extras() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 3,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: false,
        });

        let rx1 = coalescer.submit("text a".into(), Priority::Background);
        let rx2 = coalescer.submit("text b".into(), Priority::Background);
        let rx3 = coalescer.submit("text c".into(), Priority::Background);

        let batch = coalescer.try_form_batch().unwrap();
        // Only deliver 2 results for 3 requests
        batch.deliver(Ok(vec![vec![1.0], vec![2.0]]));

        assert!(rx1.recv().unwrap().is_ok());
        assert!(rx2.recv().unwrap().is_ok());
        // Third caller gets an error
        let r3 = rx3.recv().unwrap();
        assert!(r3.is_err());
        assert!(r3.unwrap_err().to_string().contains("mismatch"));
    }

    // ── Dropped receiver is tolerated ────────────────────────────────

    #[test]
    fn dropped_receiver_does_not_panic() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 2,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: false,
        });

        let rx1 = coalescer.submit("text a".into(), Priority::Background);
        let rx2 = coalescer.submit("text b".into(), Priority::Background);

        // Drop rx1's receiver (it goes out of scope)
        drop(rx1);

        let batch = coalescer.try_form_batch().unwrap();
        // Should not panic even though rx1 is dropped
        batch.deliver(Ok(vec![vec![1.0], vec![2.0]]));

        // rx2 still gets its result
        assert!(rx2.recv().unwrap().is_ok());
    }

    // ── Empty batch ──────────────────────────────────────────────────

    #[test]
    fn coalesced_batch_empty() {
        let batch = CoalescedBatch {
            requests: Vec::new(),
        };
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert!(!batch.has_interactive());
        assert!(batch.texts().is_empty());
    }

    // ── Metrics tracking ─────────────────────────────────────────────

    #[test]
    fn metrics_track_submissions_and_batches() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 2,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: true,
        });

        coalescer.submit("a".into(), Priority::Interactive);
        coalescer.submit("b".into(), Priority::Background);

        let batch = coalescer.try_form_batch().unwrap();
        batch.deliver(Ok(vec![vec![1.0], vec![2.0]]));

        let m = coalescer.metrics();
        assert_eq!(m.total_submitted.load(Ordering::Relaxed), 2);
        assert_eq!(m.interactive_submissions.load(Ordering::Relaxed), 1);
        assert_eq!(m.background_submissions.load(Ordering::Relaxed), 1);
        assert_eq!(m.total_batches.load(Ordering::Relaxed), 1);
        assert_eq!(m.total_texts_batched.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn avg_batch_size_computation() {
        let m = CoalescerMetrics::default();
        assert!(m.avg_batch_size().abs() < f64::EPSILON);

        m.total_batches.store(2, Ordering::Relaxed);
        m.total_texts_batched.store(10, Ordering::Relaxed);
        assert!((m.avg_batch_size() - 5.0).abs() < f64::EPSILON);
    }

    // ── Accessors ────────────────────────────────────────────────────

    #[test]
    fn pending_count_tracks_state() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 10_000,
            min_batch_size: 32,
            use_priority_lanes: false,
        });

        assert_eq!(coalescer.pending_count(), 0);
        coalescer.submit("a".into(), Priority::Background);
        assert_eq!(coalescer.pending_count(), 1);
        coalescer.submit("b".into(), Priority::Background);
        assert_eq!(coalescer.pending_count(), 2);
    }

    #[test]
    fn config_accessor() {
        let config = CoalescerConfig {
            max_batch_size: 16,
            ..CoalescerConfig::default()
        };
        let coalescer = BatchCoalescer::new(config);
        assert_eq!(coalescer.config().max_batch_size, 16);
    }

    #[test]
    fn debug_format() {
        let coalescer = BatchCoalescer::new(CoalescerConfig::default());
        let debug_str = format!("{coalescer:?}");
        assert!(debug_str.contains("BatchCoalescer"));
        assert!(debug_str.contains("pending"));
    }

    // ── Priority lanes disabled ──────────────────────────────────────

    #[test]
    fn priority_lanes_disabled_no_early_dispatch() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 100,
            min_batch_size: 32,
            use_priority_lanes: false,
        });

        let _rx = coalescer.submit("urgent".into(), Priority::Interactive);

        // At max_wait_ms / 2, no early dispatch because lanes are disabled
        thread::sleep(Duration::from_millis(55));
        // Interactive deadline (max_wait_ms / 2 = 50ms) should have passed though
        // Since use_priority_lanes is false, the InteractiveEarly rule doesn't fire
        // But the deadline rule still fires because the request's deadline passed
        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        // Dispatched by deadline, not early
        assert_eq!(
            coalescer.metrics().early_dispatches.load(Ordering::Relaxed),
            0
        );
    }

    // ── Multiple batches ─────────────────────────────────────────────

    #[test]
    fn multiple_batches_formed_sequentially() {
        let coalescer = BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 2,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: false,
        });

        // Submit 5 requests
        for i in 0..5 {
            coalescer.submit(format!("text-{i}"), Priority::Background);
        }

        // Should form batches of 2, 2, then 1 remaining
        let b1 = coalescer.try_form_batch().unwrap();
        assert_eq!(b1.len(), 2);
        b1.deliver(Ok(vec![vec![1.0], vec![2.0]]));

        let b2 = coalescer.try_form_batch().unwrap();
        assert_eq!(b2.len(), 2);
        b2.deliver(Ok(vec![vec![3.0], vec![4.0]]));

        // Last request (below max_batch_size) — only dispatches after deadline
        assert_eq!(coalescer.pending_count(), 1);

        assert_eq!(coalescer.metrics().total_batches.load(Ordering::Relaxed), 2);
        assert_eq!(
            coalescer
                .metrics()
                .total_texts_batched
                .load(Ordering::Relaxed),
            4
        );
    }

    // ── Shutdown with pending → wait_for_batch returns batch then None ─

    #[test]
    fn shutdown_with_pending_returns_batch_then_none() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 10_000,
            min_batch_size: 32,
            use_priority_lanes: false,
        }));

        coalescer.submit("leftover".into(), Priority::Background);
        coalescer.shutdown();

        // First call returns the pending batch
        let batch = coalescer.wait_for_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 1);

        // Second call returns None (empty + shutdown)
        let batch = coalescer.wait_for_batch();
        assert!(batch.is_none());
    }

    // ── Empty text handled gracefully ─────────────────────────────────

    #[test]
    fn empty_text_handled_gracefully() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 1, // Full rule fires immediately
            max_wait_ms: 100,
            min_batch_size: 1,
            use_priority_lanes: false,
        }));

        let rx = coalescer.submit(String::new(), Priority::Background);

        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.texts(), vec![""]);

        // Deliver a valid embedding for the empty text.
        batch.deliver(Ok(vec![vec![0.0]]));
        let result = rx.recv().expect("should receive result");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.0]);
    }

    // ── max_batch_size is never exceeded ──────────────────────────────

    #[test]
    fn max_batch_size_never_exceeded() {
        let max = 4;
        let total = max * 3; // exact multiple, so all batches fire via Full rule
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: max,
            max_wait_ms: 10_000,
            min_batch_size: 1,
            use_priority_lanes: false,
        }));

        // Submit exact multiple of max_batch_size requests.
        for i in 0..total {
            coalescer.submit(format!("text-{i}"), Priority::Background);
        }

        // Each formed batch must have exactly max_batch_size entries.
        let mut total_dispatched = 0;
        while total_dispatched < total {
            let batch = coalescer.try_form_batch();
            if let Some(b) = batch {
                assert!(b.len() <= max, "batch size {} exceeds max {}", b.len(), max);
                total_dispatched += b.len();
                let results: Vec<Vec<f32>> = (0..b.len()).map(|_| vec![0.0]).collect();
                b.deliver(Ok(results));
            } else {
                break;
            }
        }
        assert_eq!(total_dispatched, total);
    }

    // ── Graceful shutdown delivers to all pending callers ─────────────

    #[test]
    fn graceful_shutdown_delivers_to_all_pending() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 32,
            max_wait_ms: 60_000,
            min_batch_size: 32,
            use_priority_lanes: false,
        }));

        // Submit 5 requests (below min_batch and max_batch).
        let receivers: Vec<_> = (0..5)
            .map(|i| coalescer.submit(format!("pending-{i}"), Priority::Background))
            .collect();

        // Shutdown should make all pending requests available.
        coalescer.shutdown();

        // Drain all batches and deliver results.
        loop {
            let batch = coalescer.wait_for_batch();
            match batch {
                Some(b) => {
                    let results: Vec<Vec<f32>> = (0..b.len()).map(|_| vec![1.0]).collect();
                    b.deliver(Ok(results));
                }
                None => break,
            }
        }

        // All 5 callers should have received their results.
        for (i, rx) in receivers.iter().enumerate() {
            let result = rx
                .recv()
                .unwrap_or_else(|_| panic!("caller {i} should receive result"));
            assert!(result.is_ok(), "caller {i} result should be Ok");
            assert_eq!(result.unwrap(), vec![1.0]);
        }
    }

    // ── Multiple priorities interleaved correctly ─────────────────────

    #[test]
    fn mixed_interactive_background_preserves_submission_order() {
        let coalescer = Arc::new(BatchCoalescer::new(CoalescerConfig {
            max_batch_size: 10,
            max_wait_ms: 100,
            min_batch_size: 1,
            use_priority_lanes: true,
        }));

        coalescer.submit("bg-1".into(), Priority::Background);
        coalescer.submit("int-1".into(), Priority::Interactive);
        coalescer.submit("bg-2".into(), Priority::Background);
        coalescer.submit("int-2".into(), Priority::Interactive);

        // With interactive present and priority lanes enabled, should dispatch.
        std::thread::sleep(Duration::from_millis(60));
        let batch = coalescer.try_form_batch();
        assert!(batch.is_some());
        let batch = batch.unwrap();

        // All 4 should be in the batch, in submission order.
        let texts = batch.texts();
        assert_eq!(texts.len(), 4);
        assert_eq!(texts[0], "bg-1");
        assert_eq!(texts[1], "int-1");
        assert_eq!(texts[2], "bg-2");
        assert_eq!(texts[3], "int-2");
        assert!(batch.has_interactive());
    }
}
