//! Filesystem watcher for live incremental re-indexing.
//!
//! The watcher keeps fsfs indexes fresh by:
//! - coalescing rapid filesystem events via debounce windows,
//! - classifying changed files through discovery policy before ingest,
//! - adapting behavior based on pressure state,
//! - providing deterministic snapshot diffing for crash-recovery catch-up.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use asupersync::runtime::TaskHandle;
use asupersync::types::CancelReason;
use frankensearch_core::{SearchError, SearchResult};
use notify::event::{ModifyKind, RenameMode};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{debug, warn};

use crate::config::{
    DiscoveryCandidate, DiscoveryConfig, DiscoveryScopeDecision, FsfsConfig, IngestionClass,
};
use crate::mount_info::{FsCategory, MountTable, read_system_mounts};
use crate::pressure::PressureState;
use crate::stream_protocol::is_retryable_error;

pub const DEFAULT_DEBOUNCE_MS: u64 = 500;
pub const DEFAULT_BATCH_SIZE: usize = 100;
const WATCHER_SUBSYSTEM: &str = "fsfs_watcher";

/// One normalized filesystem change event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    pub path: PathBuf,
    pub kind: WatchEventKind,
    pub observed_at_ms: u64,
    pub byte_len: Option<u64>,
    pub is_symlink: bool,
    pub mount_category: Option<FsCategory>,
}

impl WatchEvent {
    #[must_use]
    pub fn created(path: impl Into<PathBuf>, observed_at_ms: u64, byte_len: Option<u64>) -> Self {
        Self {
            path: path.into(),
            kind: WatchEventKind::Created,
            observed_at_ms,
            byte_len,
            is_symlink: false,
            mount_category: None,
        }
    }

    #[must_use]
    pub fn modified(path: impl Into<PathBuf>, observed_at_ms: u64, byte_len: Option<u64>) -> Self {
        Self {
            path: path.into(),
            kind: WatchEventKind::Modified,
            observed_at_ms,
            byte_len,
            is_symlink: false,
            mount_category: None,
        }
    }

    #[must_use]
    pub fn deleted(path: impl Into<PathBuf>, observed_at_ms: u64) -> Self {
        Self {
            path: path.into(),
            kind: WatchEventKind::Deleted,
            observed_at_ms,
            byte_len: None,
            is_symlink: false,
            mount_category: None,
        }
    }
}

/// Event kind used by debounce + ingestion planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchEventKind {
    Created,
    Modified,
    Deleted,
}

/// Ingest operation emitted by watcher processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchIngestOp {
    Upsert {
        file_key: String,
        revision: i64,
        ingestion_class: IngestionClass,
    },
    Delete {
        file_key: String,
        revision: i64,
    },
}

/// One processed batch outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WatchBatchOutcome {
    pub accepted: usize,
    pub reindexed: usize,
    pub skipped: usize,
}

/// Boxed ingest future, dyn-safe like [`frankensearch_core::SearchFuture`] but
/// deliberately without its `Send` bound.
///
/// `SearchFuture` is the pattern this follows and would be the default choice,
/// but the live sink's `apply_batch_inner` is declared
/// `#[allow(clippy::future_not_send)]`: its future is genuinely not `Send`, so
/// it cannot coerce into a `Send`-bounded box. Dropping the bound is sound
/// here because an ingest future is always created and driven on one thread —
/// either the caller's task in [`FsWatcher::process_events_now`] or the local
/// task spawned by [`FsWatcher::start`] — and never handed to another. Making
/// ingest `Send` is an ingest-internals change (the vector-index mutex guard
/// spans awaits), not part of this conversion.
pub type WatchIngestFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = SearchResult<T>> + 'a>>;

/// Ingest sink contract consumed by the watcher.
///
/// The methods take the caller's [`Cx`] and return a boxed future, so `Arc<dyn
/// WatchIngestPipeline>` keeps working while the sink stays async. Handing the
/// sink a runtime instead of a `Cx` is what previously forced a `block_on` at
/// every call site and left ingest with no way to observe the caller's
/// cancellation.
pub trait WatchIngestPipeline: Send + Sync {
    /// Apply one watcher-produced batch under the caller's `Cx`.
    ///
    /// Returns the number of successfully reindexed files.
    ///
    /// # Errors
    ///
    /// Returns any ingest/indexing failure from the downstream pipeline, or a
    /// cancellation error once `cx` is cancelled.
    fn apply_batch<'a>(
        &'a self,
        cx: &'a Cx,
        batch: &'a [WatchIngestOp],
    ) -> WatchIngestFuture<'a, usize>;

    /// Poll an out-of-band durable flush request.
    ///
    /// The default is a no-op for test and dry-run sinks. Live index writers
    /// use this hook so a separate CLI process can request a publication
    /// barrier without acquiring the writer lease itself.
    ///
    /// # Errors
    ///
    /// Returns any durable publication or acknowledgement failure.
    fn poll_flush_barrier<'a>(&'a self, _cx: &'a Cx) -> WatchIngestFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }
}

/// No-op ingest sink used by tests and dry-run scenarios.
#[derive(Debug, Default)]
pub struct NoopWatchIngestPipeline;

impl WatchIngestPipeline for NoopWatchIngestPipeline {
    fn apply_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        _batch: &'a [WatchIngestOp],
    ) -> WatchIngestFuture<'a, usize> {
        Box::pin(async { Ok(0) })
    }
}

/// Effective execution policy derived from pressure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatcherExecutionPolicy {
    pub debounce_ms: u64,
    pub batch_size: usize,
    pub watching_enabled: bool,
}

impl WatcherExecutionPolicy {
    #[must_use]
    pub fn for_pressure(
        state: PressureState,
        base_debounce_ms: u64,
        base_batch_size: usize,
    ) -> Self {
        let base_debounce_ms = base_debounce_ms.max(1);
        let base_batch_size = base_batch_size.max(1);

        match state {
            PressureState::Normal => Self {
                debounce_ms: base_debounce_ms,
                batch_size: base_batch_size,
                watching_enabled: true,
            },
            PressureState::Constrained => Self {
                debounce_ms: base_debounce_ms.saturating_mul(2),
                batch_size: reduce_batch_size(base_batch_size, 2),
                watching_enabled: true,
            },
            PressureState::Degraded => Self {
                debounce_ms: base_debounce_ms.saturating_mul(10),
                batch_size: reduce_batch_size(base_batch_size, 10),
                watching_enabled: false,
            },
            PressureState::Emergency => Self {
                debounce_ms: base_debounce_ms.saturating_mul(20),
                batch_size: 1,
                watching_enabled: false,
            },
        }
    }
}

/// Snapshot map used for crash-recovery catch-up.
pub type FileSnapshot = BTreeMap<PathBuf, u64>;

/// Whether an authoritative scan actually observed every in-scope path.
///
/// A snapshot is only evidence of absence where the scan was allowed to look.
/// A directory the walk could not read, and a root that was not there at all,
/// both produce a *short* snapshot that is indistinguishable from deletion if
/// the caller forgets to ask. Deletion derivation therefore consumes this
/// receipt, and an incomplete scan derives no deletes and leaves the prior
/// baseline authoritative.
///
/// The recorded paths are the ones the scan could not resolve, not the ones it
/// skipped by policy: an excluded root or a filtered path is a complete
/// observation that the path is out of scope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanCompleteness {
    unresolved: BTreeSet<PathBuf>,
}

impl ScanCompleteness {
    /// Whether every in-scope path was resolved.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.unresolved.is_empty()
    }

    /// Paths the scan could not resolve, in deterministic order.
    pub fn unresolved_paths(&self) -> impl ExactSizeIterator<Item = &Path> {
        self.unresolved.iter().map(PathBuf::as_path)
    }

    /// Number of unresolved paths.
    #[must_use]
    pub fn unresolved_count(&self) -> usize {
        self.unresolved.len()
    }

    /// Record a path whose contents or existence could not be established.
    fn record_unresolved(&mut self, path: impl Into<PathBuf>) {
        self.unresolved.insert(path.into());
    }
}

type SnapshotCollector = dyn Fn(&[PathBuf], &DiscoveryConfig) -> SearchResult<(FileSnapshot, ScanCompleteness)>
    + Send
    + Sync;

/// Public watcher statistics snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatcherStats {
    pub watching_dirs: usize,
    pub events_received: u64,
    pub events_debounced: u64,
    pub files_reindexed: u64,
    pub files_skipped: u64,
    pub errors: u64,
    pub events_dropped_pressure: u64,
    pub worker_restarts: u64,
    pub last_event_at: Option<SystemTime>,
}

#[derive(Debug, Default)]
struct WatcherStatsInner {
    watching_dirs: AtomicUsize,
    events_received: AtomicU64,
    events_debounced: AtomicU64,
    files_reindexed: AtomicU64,
    files_skipped: AtomicU64,
    errors: AtomicU64,
    events_dropped_pressure: AtomicU64,
    worker_restarts: AtomicU64,
    last_event_at_ms: AtomicU64,
}

impl WatcherStatsInner {
    fn mark_event(&self, observed_at_ms: u64) {
        self.events_received.fetch_add(1, Ordering::Relaxed);
        self.last_event_at_ms
            .store(observed_at_ms, Ordering::Relaxed);
    }

    fn add_debounced(&self, count: usize) {
        self.events_debounced
            .fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    fn add_skipped(&self, count: usize) {
        self.files_skipped
            .fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    fn add_reindexed(&self, count: usize) {
        self.files_reindexed
            .fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    fn add_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> WatcherStats {
        let raw_last = self.last_event_at_ms.load(Ordering::Relaxed);
        let last_event_at = if raw_last == 0 {
            None
        } else {
            UNIX_EPOCH.checked_add(Duration::from_millis(raw_last))
        };

        WatcherStats {
            watching_dirs: self.watching_dirs.load(Ordering::Relaxed),
            events_received: self.events_received.load(Ordering::Relaxed),
            events_debounced: self.events_debounced.load(Ordering::Relaxed),
            files_reindexed: self.files_reindexed.load(Ordering::Relaxed),
            files_skipped: self.files_skipped.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            events_dropped_pressure: self.events_dropped_pressure.load(Ordering::Relaxed),
            worker_restarts: self.worker_restarts.load(Ordering::Relaxed),
            last_event_at,
        }
    }
}

type ReadyBatchQueue = Arc<Mutex<VecDeque<Vec<WatchEvent>>>>;

type ProducerHandle = thread::JoinHandle<SearchResult<()>>;

#[derive(Default)]
struct WatcherStop {
    requested: AtomicBool,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl WatcherStop {
    fn request(&self) {
        // The store and the notify must both happen under `wait_lock`, not just
        // the notify. `wait_or_stopped` tests the flag while holding that lock
        // and only then hands it to `wait_timeout`. Publishing the flag outside
        // the lock lets the store and the wakeup both land inside that window:
        // the waiter reads `false`, we set `true` and notify with no waiter
        // registered, the notification is dropped, and the waiter then sleeps
        // the entire backoff — up to MAX_BACKOFF_MS — with stop already
        // requested. Holding the lock across both serialises this against the
        // waiter's check-then-wait, so the waiter either observes the flag or
        // is already parked and receives the notify.
        let _publication = lock_or_recover(&self.wait_lock);
        self.requested.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn wait_or_stopped(&self, duration: Duration) -> bool {
        if self.is_requested() {
            return true;
        }
        let guard = lock_or_recover(&self.wait_lock);
        if !self.is_requested() {
            let _guard = self
                .wait_cv
                .wait_timeout(guard, duration)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        self.is_requested()
    }
}

#[derive(Default)]
struct ReconciliationState {
    indexed_snapshot: FileSnapshot,
    baseline_initialized: bool,
    required: bool,
    affected_paths: BTreeSet<PathBuf>,
    epoch: u64,
}

type ReconciliationTracker = Arc<Mutex<ReconciliationState>>;

impl ReconciliationState {
    fn require_for_events(&mut self, events: &[WatchEvent]) {
        self.required = true;
        self.epoch = self.epoch.saturating_add(1);
        self.affected_paths
            .extend(events.iter().map(|event| event.path.clone()));
    }

    fn require_full_scan(&mut self) {
        self.required = true;
        self.epoch = self.epoch.saturating_add(1);
    }
}

enum WatcherLifecycle {
    Stopped,
    Starting {
        generation: u64,
        stop: Arc<WatcherStop>,
    },
    Running {
        generation: u64,
        stop: Arc<WatcherStop>,
        producer: ProducerHandle,
        ingest_task: TaskHandle<SearchResult<()>>,
    },
    Stopping {
        generation: u64,
        stop: Arc<WatcherStop>,
    },
}

impl Default for WatcherLifecycle {
    fn default() -> Self {
        Self::Stopped
    }
}

#[derive(Default)]
struct WatcherControl {
    next_generation: u64,
    lifecycle: WatcherLifecycle,
}

enum StartDecision {
    AlreadyRunning,
    ObservedGenerationCompleted,
    Wait(u64),
    Begin {
        generation: u64,
        stop: Arc<WatcherStop>,
    },
    Drain {
        generation: u64,
        stop: Arc<WatcherStop>,
        producer: ProducerHandle,
        ingest_task: TaskHandle<SearchResult<()>>,
    },
}

enum StopDecision {
    Done,
    Wait,
    Drain {
        generation: u64,
        stop: Arc<WatcherStop>,
        producer: ProducerHandle,
        ingest_task: TaskHandle<SearchResult<()>>,
    },
}

impl WatcherControl {
    fn start_decision(&mut self, observed_generation: Option<u64>) -> StartDecision {
        match &self.lifecycle {
            WatcherLifecycle::Stopped
                if observed_generation.is_some_and(|observed| self.next_generation >= observed) =>
            {
                StartDecision::ObservedGenerationCompleted
            }
            WatcherLifecycle::Stopped => {
                self.next_generation = self.next_generation.saturating_add(1);
                let generation = self.next_generation;
                let stop = Arc::new(WatcherStop::default());
                self.lifecycle = WatcherLifecycle::Starting {
                    generation,
                    stop: Arc::clone(&stop),
                };
                StartDecision::Begin { generation, stop }
            }
            WatcherLifecycle::Starting { generation, .. }
            | WatcherLifecycle::Stopping { generation, .. } => StartDecision::Wait(*generation),
            WatcherLifecycle::Running {
                producer,
                ingest_task,
                ..
            } if !producer.is_finished() && !ingest_task.is_finished() => {
                StartDecision::AlreadyRunning
            }
            WatcherLifecycle::Running { .. } => {
                let old = std::mem::take(&mut self.lifecycle);
                let WatcherLifecycle::Running {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                } = old
                else {
                    unreachable!("running lifecycle was just matched")
                };
                stop.request();
                self.lifecycle = WatcherLifecycle::Stopping {
                    generation,
                    stop: Arc::clone(&stop),
                };
                StartDecision::Drain {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                }
            }
        }
    }

    fn stop_decision(&mut self) -> StopDecision {
        match &self.lifecycle {
            WatcherLifecycle::Stopped => StopDecision::Done,
            WatcherLifecycle::Starting { generation, stop } => {
                let generation = *generation;
                let stop = Arc::clone(stop);
                stop.request();
                self.lifecycle = WatcherLifecycle::Stopping { generation, stop };
                StopDecision::Wait
            }
            WatcherLifecycle::Stopping { .. } => StopDecision::Wait,
            WatcherLifecycle::Running { .. } => {
                let old = std::mem::take(&mut self.lifecycle);
                let WatcherLifecycle::Running {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                } = old
                else {
                    unreachable!("running lifecycle was just matched")
                };
                stop.request();
                self.lifecycle = WatcherLifecycle::Stopping {
                    generation,
                    stop: Arc::clone(&stop),
                };
                StopDecision::Drain {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                }
            }
        }
    }

    fn generation_can_publish(&self, generation: u64) -> bool {
        matches!(
            &self.lifecycle,
            WatcherLifecycle::Starting {
                generation: active,
                ..
            } if *active == generation
        )
    }

    fn complete_generation(&mut self, generation: u64) {
        let belongs_to_generation = matches!(
            &self.lifecycle,
            WatcherLifecycle::Starting {
                generation: active,
                ..
            } | WatcherLifecycle::Stopping {
                generation: active,
                ..
            } if *active == generation
        );
        if belongs_to_generation {
            self.lifecycle = WatcherLifecycle::Stopped;
        }
    }
}

/// Filesystem watcher service for live incremental re-indexing.
pub struct FsWatcher {
    roots: Vec<PathBuf>,
    discovery: DiscoveryConfig,
    ingest: Arc<dyn WatchIngestPipeline>,
    base_debounce_ms: u64,
    base_batch_size: usize,
    pressure_state: Arc<AtomicU8>,
    stats: Arc<WatcherStatsInner>,
    ready_batches: ReadyBatchQueue,
    reconciliation: ReconciliationTracker,
    control: Mutex<WatcherControl>,
}

impl FsWatcher {
    #[must_use]
    pub fn new(
        roots: Vec<PathBuf>,
        discovery: DiscoveryConfig,
        ingest: Arc<dyn WatchIngestPipeline>,
    ) -> Self {
        Self {
            roots,
            discovery,
            ingest,
            base_debounce_ms: DEFAULT_DEBOUNCE_MS,
            base_batch_size: DEFAULT_BATCH_SIZE,
            pressure_state: Arc::new(AtomicU8::new(pressure_state_to_code(PressureState::Normal))),
            stats: Arc::new(WatcherStatsInner::default()),
            ready_batches: Arc::new(Mutex::new(VecDeque::new())),
            reconciliation: Arc::new(Mutex::new(ReconciliationState::default())),
            control: Mutex::new(WatcherControl::default()),
        }
    }

    #[must_use]
    pub fn from_config(config: &FsfsConfig, ingest: Arc<dyn WatchIngestPipeline>) -> Self {
        let roots = config.discovery.roots.iter().map(PathBuf::from).collect();
        Self::new(roots, config.discovery.clone(), ingest)
    }

    #[must_use]
    pub fn with_debounce_ms(mut self, debounce_ms: u64) -> Self {
        self.base_debounce_ms = debounce_ms.max(1);
        self
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.base_batch_size = batch_size.max(1);
        self
    }

    #[must_use]
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    #[must_use]
    pub fn execution_policy(&self) -> WatcherExecutionPolicy {
        WatcherExecutionPolicy::for_pressure(
            self.pressure_state(),
            self.base_debounce_ms,
            self.base_batch_size,
        )
    }

    #[must_use]
    pub fn pressure_state(&self) -> PressureState {
        pressure_state_from_code(self.pressure_state.load(Ordering::Acquire))
    }

    pub fn apply_pressure_state(&self, state: PressureState) {
        self.pressure_state
            .store(pressure_state_to_code(state), Ordering::Release);
    }

    #[must_use]
    pub fn stats(&self) -> WatcherStats {
        self.stats.snapshot()
    }

    /// Report whether a started watcher generation has reached a terminal task
    /// outcome that its owner must join with [`Self::stop_checked`].
    #[must_use]
    pub fn has_terminal_task_outcome(&self) -> bool {
        let control = lock_or_recover(&self.control);
        match &control.lifecycle {
            WatcherLifecycle::Stopped => control.next_generation > 0,
            WatcherLifecycle::Running {
                producer,
                ingest_task,
                ..
            } => producer.is_finished() || ingest_task.is_finished(),
            WatcherLifecycle::Starting { .. } | WatcherLifecycle::Stopping { .. } => false,
        }
    }

    /// Start background watch processing.
    ///
    /// # Errors
    ///
    /// Returns an error if the watcher backend cannot be created or started.
    pub async fn start(&self, cx: &Cx) -> SearchResult<()> {
        if cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "watch.start".to_owned(),
                reason: "cancel requested before start".to_owned(),
            });
        }

        let mut observed_generation = None;
        loop {
            let decision = lock_or_recover(&self.control).start_decision(observed_generation);
            match decision {
                StartDecision::AlreadyRunning | StartDecision::ObservedGenerationCompleted => {
                    return Ok(());
                }
                StartDecision::Wait(generation) => {
                    observed_generation = Some(generation);
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                StartDecision::Drain {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                } => {
                    stop.request();
                    let result = finish_watcher_tasks(cx, Some(producer), Some(ingest_task)).await;
                    lock_or_recover(&self.control).complete_generation(generation);
                    result?;
                }
                StartDecision::Begin { generation, stop } => {
                    return self.start_generation(cx, generation, stop).await;
                }
            }
        }
    }

    async fn start_generation(
        &self,
        cx: &Cx,
        generation: u64,
        stop: Arc<WatcherStop>,
    ) -> SearchResult<()> {
        const ADMISSION_POLLS: usize = 1_000;

        let admitted = Arc::new(AtomicBool::new(false));
        let admitted_for_task = Arc::clone(&admitted);
        let producer_done = Arc::new(AtomicBool::new(false));
        let producer_done_for_task = Arc::clone(&producer_done);
        let ingest = Arc::clone(&self.ingest);
        let ingest_discovery = self.discovery.clone();
        let ingest_stats = Arc::clone(&self.stats);
        let ingest_queue = Arc::clone(&self.ready_batches);
        let ingest_stop = Arc::clone(&stop);
        let ingest_reconciliation = Arc::clone(&self.reconciliation);
        let ingest_roots = self.roots.clone();
        let ingest_batch_size = self.base_batch_size;
        let ingest_task = match cx.spawn_local(move |child_cx| async move {
            admitted_for_task.store(true, Ordering::Release);
            let _stop_producer_on_exit = IngestTaskStopGuard {
                stop: Arc::clone(&ingest_stop),
            };
            run_ingest_loop(
                &child_cx,
                &ingest_roots,
                &ingest_discovery,
                ingest.as_ref(),
                &ingest_queue,
                &ingest_stop,
                &ingest_stats,
                &ingest_reconciliation,
                ingest_batch_size,
                &producer_done_for_task,
                &collect_snapshot_from_roots,
            )
            .await
        }) {
            Ok(task) => task,
            Err(error) => {
                let error = watcher_task_error(format!(
                    "failed to spawn caller-owned ingest task: {error}"
                ));
                lock_or_recover(&self.control).complete_generation(generation);
                return Err(error);
            }
        };

        for _ in 0..ADMISSION_POLLS {
            if admitted.load(Ordering::Acquire) {
                break;
            }
            let still_starting = lock_or_recover(&self.control).generation_can_publish(generation);
            if stop.is_requested() || !still_starting {
                stop.request();
                ingest_task.abort_with_reason(CancelReason::shutdown());
                let _ = finish_watcher_tasks(cx, None, Some(ingest_task)).await;
                lock_or_recover(&self.control).complete_generation(generation);
                return Ok(());
            }
            if ingest_task.is_finished() {
                let result = finish_watcher_tasks(cx, None, Some(ingest_task)).await;
                lock_or_recover(&self.control).complete_generation(generation);
                return result;
            }
            asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
        }

        if !admitted.load(Ordering::Acquire) {
            stop.request();
            ingest_task.abort_with_reason(CancelReason::user(
                "watcher ingest admission handshake timed out",
            ));
            let _ = finish_watcher_tasks(cx, None, Some(ingest_task)).await;
            let error =
                watcher_task_error("watcher ingest task was not admitted within 1000 polls");
            lock_or_recover(&self.control).complete_generation(generation);
            return Err(error);
        }

        if ingest_task.is_finished() {
            let result = finish_watcher_tasks(cx, None, Some(ingest_task)).await;
            lock_or_recover(&self.control).complete_generation(generation);
            return result;
        }

        let producer_context = ProducerContext {
            roots: self.roots.clone(),
            discovery: self.discovery.clone(),
            stats: Arc::clone(&self.stats),
            pressure_state: Arc::clone(&self.pressure_state),
            stop: Arc::clone(&stop),
            ready_batches: Arc::clone(&self.ready_batches),
            reconciliation: Arc::clone(&self.reconciliation),
            producer_done,
            base_debounce_ms: self.base_debounce_ms,
            base_batch_size: self.base_batch_size,
        };

        let producer = match thread::Builder::new()
            .name("fsfs-watcher".to_owned())
            .spawn(move || run_producer_supervisor(&producer_context))
        {
            Ok(producer) => producer,
            Err(error) => {
                stop.request();
                ingest_task.abort_with_reason(CancelReason::shutdown());
                let _ = finish_watcher_tasks(cx, None, Some(ingest_task)).await;
                let error =
                    watcher_task_error(format!("failed to spawn watcher producer: {error}"));
                lock_or_recover(&self.control).complete_generation(generation);
                return Err(error);
            }
        };

        let mut pair = Some((producer, ingest_task));
        let published = {
            let mut control = lock_or_recover(&self.control);
            if control.generation_can_publish(generation) && !stop.is_requested() {
                let (producer, ingest_task) = pair.take().expect("unpublished watcher pair");
                control.lifecycle = WatcherLifecycle::Running {
                    generation,
                    stop: Arc::clone(&stop),
                    producer,
                    ingest_task,
                };
                true
            } else {
                false
            }
        };

        if published {
            return Ok(());
        }

        stop.request();
        let (producer, ingest_task) = pair.expect("unpublished watcher pair remains owned");
        let result = finish_watcher_tasks(cx, Some(producer), Some(ingest_task)).await;
        lock_or_recover(&self.control).complete_generation(generation);
        result
    }

    /// Stop background watch processing, logging any terminal task failure.
    pub async fn stop(&self, cx: &Cx) {
        if let Err(error) = self.stop_checked(cx).await {
            warn!(error = %error, "fsfs watcher stopped after a task failure");
        }
    }

    /// Stop background watch processing and return any terminal task failure.
    ///
    /// # Errors
    ///
    /// Returns a producer or ingest task's terminal failure after both tasks
    /// have been joined.
    pub async fn stop_checked(&self, cx: &Cx) -> SearchResult<()> {
        loop {
            match lock_or_recover(&self.control).stop_decision() {
                StopDecision::Done => return Ok(()),
                StopDecision::Wait => {
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                StopDecision::Drain {
                    generation,
                    stop,
                    producer,
                    ingest_task,
                } => {
                    stop.request();
                    let result = finish_watcher_tasks(cx, Some(producer), Some(ingest_task)).await;
                    lock_or_recover(&self.control).complete_generation(generation);
                    return result;
                }
            }
        }
    }

    /// Process one explicit event batch immediately (without debounce).
    ///
    /// # Errors
    ///
    /// Returns any downstream ingest error.
    pub async fn process_events_now(
        &self,
        cx: &Cx,
        events: &[WatchEvent],
    ) -> SearchResult<WatchBatchOutcome> {
        for event in events {
            self.stats.mark_event(event.observed_at_ms);
        }

        let policy = self.execution_policy();
        if !policy.watching_enabled {
            self.stats.add_skipped(events.len());
            return Ok(WatchBatchOutcome {
                accepted: 0,
                reindexed: 0,
                skipped: events.len(),
            });
        }

        if lock_or_recover(&self.reconciliation).required {
            run_authoritative_reconciliation(
                cx,
                &self.roots,
                &self.discovery,
                self.ingest.as_ref(),
                &self.reconciliation,
                &self.ready_batches,
                &self.stats,
                self.base_batch_size,
                &collect_snapshot_from_roots,
            )
            .await?;
        }

        // Runs on the caller's executor under the caller's `Cx`: this path
        // builds no runtime of its own and therefore cannot strand ingest work
        // on a private one that the caller cannot cancel.
        let prepared = prepare_event_batch(&self.discovery, events);
        if prepared.ops.is_empty() {
            let outcome = prepared.outcome(0);
            self.stats.add_skipped(outcome.skipped);
            return Ok(outcome);
        }

        let mut guard = DirectApplyGuard::new(&self.reconciliation, events);
        let reindexed = self.ingest.apply_batch(cx, &prepared.ops).await?;
        if let Err(error) = record_successful_events(
            &self.roots,
            &self.discovery,
            &self.reconciliation,
            events,
            &collect_snapshot_from_roots,
        ) {
            return Err(error);
        }
        guard.commit();
        let outcome = prepared.outcome(reindexed);
        self.stats.add_reindexed(outcome.reindexed);
        self.stats.add_skipped(outcome.skipped);
        Ok(outcome)
    }

    /// Collect a filtered file snapshot for crash-recovery comparisons.
    ///
    /// # Errors
    ///
    /// Returns errors from filesystem traversal that are not safe to ignore.
    pub fn collect_snapshot(&self) -> SearchResult<(FileSnapshot, ScanCompleteness)> {
        collect_snapshot_from_roots(&self.roots, &self.discovery)
    }

    /// Build catch-up events by diffing prior and current snapshots.
    ///
    /// An incomplete scan still reports the creates and modifies it observed —
    /// those paths demonstrably exist — but derives no deletes, and leaves
    /// reconciliation required so a later complete scan can settle the
    /// difference.
    ///
    /// # Errors
    ///
    /// Returns errors from current snapshot collection.
    pub fn build_catchup_events(&self, previous: &FileSnapshot) -> SearchResult<Vec<WatchEvent>> {
        let (current, completeness) = self.collect_snapshot()?;
        if !completeness.is_complete() {
            lock_or_recover(&self.reconciliation).require_full_scan();
        }
        Ok(Self::diff_snapshots(
            previous,
            &current,
            now_millis(),
            &completeness,
        ))
    }

    /// Deterministically diff two snapshots into create/modify/delete events.
    ///
    /// Deletes are derived only from a complete scan. `current` is a statement
    /// about absence only where the scan was able to look, so an incomplete
    /// receipt suppresses every delete rather than inventing one for a path
    /// the scan never resolved.
    #[must_use]
    pub fn diff_snapshots(
        previous: &FileSnapshot,
        current: &FileSnapshot,
        observed_at_ms: u64,
        completeness: &ScanCompleteness,
    ) -> Vec<WatchEvent> {
        let derive_deletes = completeness.is_complete();
        let mut events = Vec::new();
        let mut prev_iter = previous.iter();
        let mut curr_iter = current.iter();
        let mut p_next = prev_iter.next();
        let mut c_next = curr_iter.next();

        while let (Some((p_path, p_time)), Some((c_path, c_time))) = (p_next, c_next) {
            match p_path.cmp(c_path) {
                std::cmp::Ordering::Less => {
                    if derive_deletes {
                        events.push(WatchEvent::deleted(p_path, observed_at_ms));
                    }
                    p_next = prev_iter.next();
                }
                std::cmp::Ordering::Greater => {
                    events.push(WatchEvent::created(c_path, observed_at_ms, None));
                    c_next = curr_iter.next();
                }
                std::cmp::Ordering::Equal => {
                    if p_time != c_time {
                        events.push(WatchEvent::modified(c_path, observed_at_ms, None));
                    }
                    p_next = prev_iter.next();
                    c_next = curr_iter.next();
                }
            }
        }

        while let Some((p_path, _)) = p_next {
            if derive_deletes {
                events.push(WatchEvent::deleted(p_path, observed_at_ms));
            }
            p_next = prev_iter.next();
        }

        while let Some((c_path, _)) = c_next {
            events.push(WatchEvent::created(c_path, observed_at_ms, None));
            c_next = curr_iter.next();
        }

        events
    }
}

impl Drop for FsWatcher {
    fn drop(&mut self) {
        let mut control = lock_or_recover(&self.control);
        match &mut control.lifecycle {
            WatcherLifecycle::Stopped => {}
            WatcherLifecycle::Starting { stop, .. } | WatcherLifecycle::Stopping { stop, .. } => {
                stop.request()
            }
            WatcherLifecycle::Running {
                stop, ingest_task, ..
            } => {
                stop.request();
                ingest_task.abort_with_reason(CancelReason::shutdown());
            }
        }
    }
}

struct ProducerContext {
    roots: Vec<PathBuf>,
    discovery: DiscoveryConfig,
    stats: Arc<WatcherStatsInner>,
    pressure_state: Arc<AtomicU8>,
    stop: Arc<WatcherStop>,
    ready_batches: ReadyBatchQueue,
    reconciliation: ReconciliationTracker,
    producer_done: Arc<AtomicBool>,
    base_debounce_ms: u64,
    base_batch_size: usize,
}

fn run_producer_supervisor(context: &ProducerContext) -> SearchResult<()> {
    const MAX_RESTARTS: usize = 10;
    const MIN_BACKOFF_MS: u64 = 500;
    const MAX_BACKOFF_MS: u64 = 30_000;

    let mut restarts = 0_usize;
    let terminal = loop {
        match run_producer_loop(context) {
            Ok(()) => break Ok(()),
            Err(error) => {
                context.stats.add_error();
                restarts = restarts.saturating_add(1);
                if context.stop.is_requested() {
                    debug!(error = %error, "watcher producer exited with error after stop signal");
                    break Err(error);
                }
                if restarts > MAX_RESTARTS || !is_retryable_error(&error) {
                    warn!(
                        error = %error,
                        restarts,
                        "watcher producer reached a terminal failure"
                    );
                    break Err(error);
                }
                context
                    .stats
                    .worker_restarts
                    .fetch_add(1, Ordering::Relaxed);
                let backoff_ms = MIN_BACKOFF_MS
                    .saturating_mul(1_u64 << restarts.min(6))
                    .min(MAX_BACKOFF_MS);
                warn!(
                    error = %error,
                    restart_attempt = restarts,
                    backoff_ms,
                    "watcher producer failed; restarting after backoff"
                );
                if context
                    .stop
                    .wait_or_stopped(Duration::from_millis(backoff_ms))
                {
                    break Err(error);
                }
            }
        }
    };
    context.stats.watching_dirs.store(0, Ordering::Relaxed);
    context.producer_done.store(true, Ordering::Release);
    context.stop.request();
    terminal
}

#[allow(clippy::too_many_lines)]
fn run_producer_loop(context: &ProducerContext) -> SearchResult<()> {
    let (event_tx, event_rx) = std::sync::mpsc::channel::<notify::Result<Event>>();
    let mut watcher = build_notify_watcher(event_tx)?;
    let mount_table = build_mount_table(&context.discovery);

    let mut watched_dirs = 0_usize;
    for root in &context.roots {
        if !root.exists() {
            continue;
        }
        watcher
            .watch(root, RecursiveMode::Recursive)
            .map_err(|error| watcher_error(&error))?;
        watched_dirs = watched_dirs.saturating_add(1);
    }
    context
        .stats
        .watching_dirs
        .store(watched_dirs, Ordering::Relaxed);

    if watched_dirs == 0 {
        return Ok(());
    }

    let (baseline, baseline_completeness) =
        collect_snapshot_from_roots(&context.roots, &context.discovery)?;
    {
        let mut reconciliation = lock_or_recover(&context.reconciliation);
        // A short startup baseline would make the first authoritative rescan
        // read every unobserved path as a creation and, once promoted, every
        // later disappearance as a delete. Adopt it, but require a rescan so
        // it is replaced by a complete one.
        if !reconciliation.baseline_initialized {
            reconciliation.indexed_snapshot = baseline;
            reconciliation.baseline_initialized = true;
        }
        if !baseline_completeness.is_complete() {
            reconciliation.require_full_scan();
        }
    }

    let mut pending = PendingEvents::default();
    let mut pressure_was_disabled = false;
    let mut disconnected = false;
    while !context.stop.is_requested() {
        let policy = WatcherExecutionPolicy::for_pressure(
            pressure_state_from_code(context.pressure_state.load(Ordering::Acquire)),
            context.base_debounce_ms,
            context.base_batch_size,
        );

        let timeout = pending.earliest_observed_at().map_or_else(
            || Duration::from_millis(100),
            |earliest| {
                let now = now_millis();
                let ready_at = earliest.saturating_add(policy.debounce_ms);
                let wait = ready_at.saturating_sub(now);
                // Cap at 100ms to check stop flag, but allow short waits for debounce
                Duration::from_millis(wait.min(100))
            },
        );

        match event_rx.recv_timeout(timeout) {
            Ok(event) => process_notify_result(
                event,
                policy,
                &context.stats,
                &mut pending,
                Some(&mount_table),
            ),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                disconnected = true;
                break;
            }
        }
        while let Ok(event) = event_rx.try_recv() {
            process_notify_result(
                event,
                policy,
                &context.stats,
                &mut pending,
                Some(&mount_table),
            );
        }

        observe_pressure_transition(
            policy.watching_enabled,
            &mut pressure_was_disabled,
            &context.reconciliation,
        );
        if !policy.watching_enabled {
            let dropped = pending.clear();
            if dropped > 0 {
                context.stats.add_skipped(dropped);
                context.stats.events_dropped_pressure.fetch_add(
                    u64::try_from(dropped).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                debug!(
                    dropped,
                    pressure_state = ?pressure_state_from_code(context.pressure_state.load(Ordering::Acquire)),
                    "watcher dropped pending events while disabled by pressure"
                );
            }
            continue;
        }

        let ready = pending.drain_ready(now_millis(), policy.debounce_ms, policy.batch_size);
        if ready.is_empty() {
            continue;
        }

        lock_or_recover(&context.ready_batches).push_back(ready);
    }

    // Stop delivery before draining the channel so no callback can race the
    // final debounce flush. Events not yet old enough are still real work and
    // must be queued rather than discarded at shutdown.
    drop(watcher);
    let final_policy = WatcherExecutionPolicy::for_pressure(
        pressure_state_from_code(context.pressure_state.load(Ordering::Acquire)),
        context.base_debounce_ms,
        context.base_batch_size,
    );
    drain_notify_channel(
        &event_rx,
        final_policy,
        &context.stats,
        &mut pending,
        Some(&mount_table),
    );
    if !final_policy.watching_enabled || pressure_was_disabled {
        lock_or_recover(&context.reconciliation).require_full_scan();
    }
    flush_pending_batches(
        &mut pending,
        &context.ready_batches,
        context.base_batch_size,
    );

    if disconnected && !context.stop.is_requested() {
        return Err(watcher_task_error(
            "notify channel disconnected before watcher shutdown",
        ));
    }
    Ok(())
}

fn drain_notify_channel(
    event_rx: &std::sync::mpsc::Receiver<notify::Result<Event>>,
    policy: WatcherExecutionPolicy,
    stats: &WatcherStatsInner,
    pending: &mut PendingEvents,
    mount_table: Option<&MountTable>,
) {
    while let Ok(event) = event_rx.try_recv() {
        process_notify_result(event, policy, stats, pending, mount_table);
    }
}

fn observe_pressure_transition(
    watching_enabled: bool,
    pressure_was_disabled: &mut bool,
    reconciliation: &ReconciliationTracker,
) {
    if !watching_enabled {
        *pressure_was_disabled = true;
    } else if std::mem::take(pressure_was_disabled) {
        lock_or_recover(reconciliation).require_full_scan();
        debug!("pressure relieved; requiring authoritative watcher rescan");
    }
}

struct IngestTaskStopGuard {
    stop: Arc<WatcherStop>,
}

impl Drop for IngestTaskStopGuard {
    fn drop(&mut self) {
        self.stop.request();
    }
}

struct PendingBatchLease {
    queue: ReadyBatchQueue,
    reconciliation: ReconciliationTracker,
    batch: Option<Vec<WatchEvent>>,
    live_apply_started: bool,
}

impl PendingBatchLease {
    fn acquire(queue: &ReadyBatchQueue, reconciliation: &ReconciliationTracker) -> Option<Self> {
        let batch = lock_or_recover(queue).pop_front()?;
        Some(Self {
            queue: Arc::clone(queue),
            reconciliation: Arc::clone(reconciliation),
            batch: Some(batch),
            live_apply_started: false,
        })
    }

    fn events(&self) -> &[WatchEvent] {
        self.batch.as_deref().unwrap_or_default()
    }

    fn begin_live_apply(&mut self) {
        self.live_apply_started = true;
    }

    fn commit(mut self) {
        self.batch = None;
    }
}

impl Drop for PendingBatchLease {
    fn drop(&mut self) {
        if let Some(batch) = self.batch.take() {
            if self.live_apply_started {
                lock_or_recover(&self.reconciliation).require_for_events(&batch);
            } else {
                lock_or_recover(&self.queue).push_front(batch);
            }
        }
    }
}

struct DirectApplyGuard {
    reconciliation: ReconciliationTracker,
    events: Vec<WatchEvent>,
    committed: bool,
}

impl DirectApplyGuard {
    fn new(reconciliation: &ReconciliationTracker, events: &[WatchEvent]) -> Self {
        Self {
            reconciliation: Arc::clone(reconciliation),
            events: events.to_vec(),
            committed: false,
        }
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for DirectApplyGuard {
    fn drop(&mut self) {
        if !self.committed {
            lock_or_recover(&self.reconciliation).require_for_events(&self.events);
        }
    }
}

struct PreparedWatchBatch {
    ops: Vec<WatchIngestOp>,
    accepted: usize,
    skipped: usize,
}

impl PreparedWatchBatch {
    const fn outcome(&self, reindexed: usize) -> WatchBatchOutcome {
        WatchBatchOutcome {
            accepted: self.accepted,
            reindexed,
            skipped: self.skipped,
        }
    }
}

async fn run_ingest_loop(
    cx: &Cx,
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    ready_batches: &ReadyBatchQueue,
    stop: &WatcherStop,
    stats: &WatcherStatsInner,
    reconciliation: &ReconciliationTracker,
    batch_size: usize,
    producer_done: &AtomicBool,
    snapshot_collector: &SnapshotCollector,
) -> SearchResult<()> {
    const IDLE_POLL: Duration = Duration::from_millis(10);
    const MAX_RECONCILIATION_ATTEMPTS: usize = 3;
    let mut reconciliation_attempts = 0_usize;

    loop {
        if cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "watch.ingest".to_owned(),
                reason: cx.cancel_reason().map_or_else(
                    || "caller-owned ingest task cancelled".to_owned(),
                    |reason| reason.to_string(),
                ),
            });
        }

        if lock_or_recover(reconciliation).required {
            match run_authoritative_reconciliation(
                cx,
                roots,
                discovery,
                ingest,
                reconciliation,
                ready_batches,
                stats,
                batch_size,
                snapshot_collector,
            )
            .await
            {
                Ok(()) => {
                    reconciliation_attempts = 0;
                    continue;
                }
                Err(error) => {
                    stats.add_error();
                    reconciliation_attempts = reconciliation_attempts.saturating_add(1);
                    if !is_retryable_error(&error)
                        || cx.is_cancel_requested()
                        || reconciliation_attempts >= MAX_RECONCILIATION_ATTEMPTS
                    {
                        return Err(error);
                    }
                    warn!(error = %error, "watcher reconciliation failed; retrying full rescan");
                    asupersync::time::sleep(cx.now(), IDLE_POLL).await;
                    continue;
                }
            }
        }

        match ingest.poll_flush_barrier(cx).await {
            Ok(true) => debug!("watcher acknowledged a durable flush barrier"),
            Ok(false) => {}
            Err(error) => {
                stats.add_error();
                warn!(error = %error, "watcher failed to acknowledge a durable flush barrier");
                if !is_retryable_error(&error) {
                    return Err(error);
                }
            }
        }

        let Some(mut lease) = PendingBatchLease::acquire(ready_batches, reconciliation) else {
            if stop.is_requested() && producer_done.load(Ordering::Acquire) {
                return Ok(());
            }
            asupersync::time::sleep(cx.now(), IDLE_POLL).await;
            continue;
        };

        let prepared = prepare_event_batch(discovery, lease.events());
        if prepared.ops.is_empty() {
            stats.add_skipped(prepared.skipped);
            lease.commit();
            continue;
        }

        // This is the point of no replay. Once the sink future is created it
        // may mutate lexical, vector, or storage state before returning. Any
        // non-success from here requires a filesystem-authoritative rescan.
        lease.begin_live_apply();
        match ingest.apply_batch(cx, &prepared.ops).await {
            Ok(reindexed) => {
                let events = lease.events().to_vec();
                let outcome = prepared.outcome(reindexed);
                if let Err(error) = record_successful_events(
                    roots,
                    discovery,
                    reconciliation,
                    &events,
                    snapshot_collector,
                ) {
                    stats.add_error();
                    if !is_retryable_error(&error) || cx.is_cancel_requested() {
                        return Err(error);
                    }
                    asupersync::time::sleep(cx.now(), IDLE_POLL).await;
                    continue;
                }
                lease.commit();
                stats.add_reindexed(outcome.reindexed);
                stats.add_skipped(outcome.skipped);
            }
            Err(error) => {
                stats.add_error();
                warn!(
                    error = %error,
                    "watcher ingest failed after mutation boundary; requiring full rescan"
                );
                drop(lease);
                if !is_retryable_error(&error) || cx.is_cancel_requested() {
                    return Err(error);
                }
                asupersync::time::sleep(cx.now(), IDLE_POLL).await;
            }
        }
    }
}

async fn finish_watcher_tasks(
    cx: &Cx,
    producer: Option<ProducerHandle>,
    ingest_task: Option<TaskHandle<SearchResult<()>>>,
) -> SearchResult<()> {
    let mut failure = None;
    if let Some(producer) = producer {
        match producer.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => failure = Some(error),
            Err(error) => {
                failure = Some(watcher_task_error(format!(
                    "fsfs watcher producer panicked during shutdown: {error:?}"
                )));
            }
        }
    }

    if let Some(mut ingest_task) = ingest_task {
        match ingest_task.join(cx).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                failure.get_or_insert(error);
            }
            Err(error) => {
                failure.get_or_insert_with(|| {
                    watcher_task_error(format!("fsfs watcher ingest task terminated: {error}"))
                });
            }
        }
    }

    failure.map_or(Ok(()), Err)
}

#[allow(clippy::too_many_arguments)]
async fn run_authoritative_reconciliation(
    cx: &Cx,
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    reconciliation: &ReconciliationTracker,
    ready_batches: &ReadyBatchQueue,
    stats: &WatcherStatsInner,
    batch_size: usize,
    snapshot_collector: &SnapshotCollector,
) -> SearchResult<()> {
    let (epoch, indexed_snapshot, affected_paths) = {
        let state = lock_or_recover(reconciliation);
        (
            state.epoch,
            state.indexed_snapshot.clone(),
            state.affected_paths.clone(),
        )
    };
    // Every batch already visible here predates the authoritative snapshot
    // below. Dropping it is safe: the rescan covers its final filesystem
    // state, while batches produced after this clear remain queued and are
    // applied after the rescan.
    lock_or_recover(ready_batches).clear();
    let (current, completeness) = snapshot_collector(roots, discovery)?;
    let observed_at_ms = now_millis();
    let mount_table = build_mount_table(discovery);
    let mut events = current
        .keys()
        .cloned()
        .map(|path| {
            build_watch_event(
                path,
                WatchEventKind::Modified,
                observed_at_ms,
                Some(&mount_table),
            )
        })
        .collect::<Vec<_>>();

    // An incomplete rescan is not authoritative about absence. It still
    // reindexes everything it did observe above, but it derives no deletes:
    // a path missing from a short snapshot may simply be one the scan could
    // not read.
    if completeness.is_complete() {
        let mut deletion_candidates = indexed_snapshot.keys().cloned().collect::<BTreeSet<_>>();
        deletion_candidates.extend(affected_paths);
        events.extend(
            deletion_candidates
                .into_iter()
                .filter(|path| !current.contains_key(path))
                .map(|path| WatchEvent::deleted(path, observed_at_ms)),
        );
    } else {
        warn!(
            unresolved_paths = completeness.unresolved_count(),
            "watcher rescan could not resolve every path; deriving no deletions and keeping \
             reconciliation required"
        );
    }

    for event_batch in events.chunks(batch_size.max(1)) {
        if cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "watch.reconcile".to_owned(),
                reason: cx.cancel_reason().map_or_else(
                    || "watcher reconciliation cancelled".to_owned(),
                    |reason| reason.to_string(),
                ),
            });
        }
        let prepared = prepare_event_batch(discovery, event_batch);
        if prepared.ops.is_empty() {
            stats.add_skipped(prepared.skipped);
            continue;
        }
        let reindexed = ingest.apply_batch(cx, &prepared.ops).await?;
        let outcome = prepared.outcome(reindexed);
        stats.add_reindexed(outcome.reindexed);
        stats.add_skipped(outcome.skipped);
    }

    let mut state = lock_or_recover(reconciliation);
    if state.epoch == epoch {
        if completeness.is_complete() {
            state.indexed_snapshot = current;
            state.baseline_initialized = true;
            state.required = false;
            state.affected_paths.clear();
        } else {
            // Promoting a short snapshot to the baseline would make the next
            // complete scan diff against it and delete everything the failed
            // scan could not see. Keep the old baseline and stay required.
            state.required = true;
        }
    }
    Ok(())
}

fn record_successful_events(
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    reconciliation: &ReconciliationTracker,
    events: &[WatchEvent],
    snapshot_collector: &SnapshotCollector,
) -> SearchResult<()> {
    let (current, completeness) = snapshot_collector(roots, discovery)?;
    let mut state = lock_or_recover(reconciliation);
    if !state.baseline_initialized {
        state.indexed_snapshot = current.clone();
        state.baseline_initialized = true;
    }
    if !completeness.is_complete() {
        // The per-event updates below stay correct — they are keyed on paths
        // this batch actually touched — but the snapshot behind them is short,
        // so the baseline must still be re-established authoritatively.
        state.required = true;
    }
    for event in events {
        if let Some(modified_at_ms) = current.get(&event.path) {
            state
                .indexed_snapshot
                .insert(event.path.clone(), *modified_at_ms);
        } else {
            state.indexed_snapshot.remove(&event.path);
        }
    }
    Ok(())
}

fn flush_pending_batches(
    pending: &mut PendingEvents,
    ready_batches: &ReadyBatchQueue,
    batch_size: usize,
) {
    loop {
        let batch = pending.drain_all(batch_size.max(1));
        if batch.is_empty() {
            break;
        }
        lock_or_recover(ready_batches).push_back(batch);
    }
}

fn watcher_task_error(message: impl Into<String>) -> SearchError {
    SearchError::SubsystemError {
        subsystem: WATCHER_SUBSYSTEM,
        source: Box::new(io::Error::other(message.into())),
    }
}

fn process_notify_result(
    event: notify::Result<Event>,
    policy: WatcherExecutionPolicy,
    stats: &WatcherStatsInner,
    pending: &mut PendingEvents,
    mount_table: Option<&MountTable>,
) {
    match event {
        Ok(event) => {
            let mapped_events = map_notify_event_with_mount_table(event, mount_table);
            if mapped_events.is_empty() {
                return;
            }

            for watch_event in mapped_events {
                stats.mark_event(watch_event.observed_at_ms);
                if !policy.watching_enabled {
                    stats.add_skipped(1);
                    continue;
                }
                if pending.push(watch_event) {
                    stats.add_debounced(1);
                }
            }
        }
        Err(error) => {
            stats.add_error();
            warn!(error = %error, "watch backend emitted error");
        }
    }
}

fn prepare_event_batch(discovery: &DiscoveryConfig, events: &[WatchEvent]) -> PreparedWatchBatch {
    let mut ops = Vec::new();
    let mut skipped = 0_usize;

    for event in events {
        if let Some(op) = event_to_ingest_op(discovery, event) {
            ops.push(op);
        } else {
            skipped = skipped.saturating_add(1);
        }
    }

    PreparedWatchBatch {
        accepted: ops.len(),
        ops,
        skipped,
    }
}

fn event_to_ingest_op(discovery: &DiscoveryConfig, event: &WatchEvent) -> Option<WatchIngestOp> {
    let revision = i64::try_from(event.observed_at_ms).unwrap_or(i64::MAX);
    let file_key = normalize_file_key(&event.path);

    if matches!(event.kind, WatchEventKind::Deleted) {
        return Some(WatchIngestOp::Delete { file_key, revision });
    }

    let byte_len = event.byte_len.unwrap_or_else(|| {
        std::fs::symlink_metadata(&event.path)
            .map(|m| m.len())
            .unwrap_or(0)
    });
    let mut candidate =
        DiscoveryCandidate::new(&event.path, byte_len).with_symlink(event.is_symlink);
    if let Some(category) = event.mount_category {
        candidate = candidate.with_mount_category(category);
    }

    let decision = discovery.evaluate_candidate(&candidate);
    if matches!(decision.scope, DiscoveryScopeDecision::Exclude)
        || !decision.ingestion_class.is_indexed()
    {
        return None;
    }

    Some(WatchIngestOp::Upsert {
        file_key,
        revision,
        ingestion_class: decision.ingestion_class,
    })
}

#[cfg(test)]
fn map_notify_event(event: Event) -> Vec<WatchEvent> {
    map_notify_event_with_mount_table(event, None)
}

fn map_notify_event_with_mount_table(
    event: Event,
    mount_table: Option<&MountTable>,
) -> Vec<WatchEvent> {
    let Event { kind, paths, .. } = event;
    let observed_at_ms = now_millis();
    if let EventKind::Modify(ModifyKind::Name(mode)) = kind {
        return map_rename_notify_event(paths, mode, observed_at_ms, mount_table);
    }

    let Some(kind) = map_notify_kind(kind) else {
        return Vec::new();
    };

    paths
        .into_iter()
        .map(|path| build_watch_event(path, kind, observed_at_ms, mount_table))
        .collect()
}

const fn map_notify_kind(kind: EventKind) -> Option<WatchEventKind> {
    match kind {
        EventKind::Create(_) => Some(WatchEventKind::Created),
        EventKind::Modify(_) => Some(WatchEventKind::Modified),
        EventKind::Remove(_) => Some(WatchEventKind::Deleted),
        _ => None,
    }
}

fn map_rename_notify_event(
    paths: Vec<PathBuf>,
    mode: RenameMode,
    observed_at_ms: u64,
    mount_table: Option<&MountTable>,
) -> Vec<WatchEvent> {
    match mode {
        RenameMode::Both => {
            let mut events = Vec::with_capacity(2);
            if let Some(from) = paths.first() {
                events.push(build_watch_event(
                    from.clone(),
                    WatchEventKind::Deleted,
                    observed_at_ms,
                    mount_table,
                ));
            }
            // Use get(1) — not last() — to reliably pick the destination
            // path even if the event carries more than two entries.
            if let Some(to) = paths.get(1) {
                events.push(build_watch_event(
                    to.clone(),
                    WatchEventKind::Created,
                    observed_at_ms,
                    mount_table,
                ));
            }
            events
        }
        RenameMode::From => paths
            .into_iter()
            .map(|path| {
                build_watch_event(path, WatchEventKind::Deleted, observed_at_ms, mount_table)
            })
            .collect(),
        RenameMode::To => paths
            .into_iter()
            .map(|path| {
                build_watch_event(path, WatchEventKind::Created, observed_at_ms, mount_table)
            })
            .collect(),
        RenameMode::Any | RenameMode::Other => paths
            .into_iter()
            .map(|path| {
                let kind = if fs::symlink_metadata(&path).is_ok() {
                    WatchEventKind::Created
                } else {
                    WatchEventKind::Deleted
                };
                build_watch_event(path, kind, observed_at_ms, mount_table)
            })
            .collect(),
    }
}

fn build_watch_event(
    path: PathBuf,
    kind: WatchEventKind,
    observed_at_ms: u64,
    mount_table: Option<&MountTable>,
) -> WatchEvent {
    let mount_category = lookup_mount_category(mount_table, &path);
    let metadata = if matches!(kind, WatchEventKind::Deleted) {
        None
    } else {
        fs::symlink_metadata(&path).ok()
    };
    let byte_len = metadata.as_ref().map(std::fs::Metadata::len);
    let is_symlink = metadata
        .as_ref()
        .is_some_and(|meta| meta.file_type().is_symlink());

    WatchEvent {
        path,
        kind,
        observed_at_ms,
        byte_len,
        is_symlink,
        mount_category,
    }
}

fn build_mount_table(discovery: &DiscoveryConfig) -> MountTable {
    let overrides = discovery.mount_override_map();
    MountTable::new(read_system_mounts(), &overrides)
}

fn lookup_mount_category(mount_table: Option<&MountTable>, path: &Path) -> Option<FsCategory> {
    mount_table.and_then(|table| table.lookup(path).map(|(entry, _)| entry.category))
}

fn build_notify_watcher(
    event_tx: std::sync::mpsc::Sender<notify::Result<Event>>,
) -> SearchResult<RecommendedWatcher> {
    notify::recommended_watcher(move |event| {
        if event_tx.send(event).is_err() {
            debug!("watch event dropped because worker channel is closed");
        }
    })
    .map_err(|error| watcher_error(&error))
}

fn watcher_error(error: &notify::Error) -> SearchError {
    SearchError::SubsystemError {
        subsystem: WATCHER_SUBSYSTEM,
        source: Box::new(io::Error::other(format!("watch backend error: {error}"))),
    }
}

fn collect_snapshot_from_roots(
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
) -> SearchResult<(FileSnapshot, ScanCompleteness)> {
    let mut snapshot = FileSnapshot::new();
    let mut completeness = ScanCompleteness::default();
    let mount_table = build_mount_table(discovery);
    for root in roots {
        collect_snapshot_for_root(
            root,
            discovery,
            Some(&mount_table),
            &mut snapshot,
            &mut completeness,
        )?;
    }
    Ok((snapshot, completeness))
}

fn collect_snapshot_for_root(
    root: &Path,
    discovery: &DiscoveryConfig,
    mount_table: Option<&MountTable>,
    snapshot: &mut FileSnapshot,
    completeness: &mut ScanCompleteness,
) -> SearchResult<()> {
    if !root.exists() {
        // A root that is not there is not the same claim as a root that is
        // there and empty. An unmounted or not-yet-created root would
        // otherwise diff as the deletion of everything beneath it.
        completeness.record_unresolved(root);
        return Ok(());
    }

    let root_decision = discovery.evaluate_root(root, lookup_mount_category(mount_table, root));
    if matches!(root_decision.scope, DiscoveryScopeDecision::Exclude) {
        return Ok(());
    }

    // Handle single-file roots explicitly to avoid walk errors
    let symlink_meta = fs::symlink_metadata(root).map_err(SearchError::Io)?;
    let (metadata, is_symlink) = if symlink_meta.is_symlink() {
        match fs::metadata(root) {
            Ok(target) if target.is_file() => (target, true),
            Ok(target) => (target, true), // Directory or other: preserve symlink identity
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()), // Broken link
            Err(e) => return Err(e.into()),
        }
    } else {
        (symlink_meta, false)
    };

    if is_symlink && !discovery.follow_symlinks {
        return Ok(());
    }

    if metadata.is_file() {
        let mut candidate = DiscoveryCandidate::new(root, metadata.len()).with_symlink(is_symlink);
        if let Some(category) = lookup_mount_category(mount_table, root) {
            candidate = candidate.with_mount_category(category);
        }
        let decision = discovery.evaluate_candidate(&candidate);
        if !matches!(decision.scope, DiscoveryScopeDecision::Exclude)
            && decision.ingestion_class.is_indexed()
        {
            let modified = metadata
                .modified()
                .ok()
                .map(system_time_to_ms)
                .unwrap_or_default();
            snapshot.insert(root.to_path_buf(), modified);
        }
        return Ok(());
    }

    let mut stack = vec![root.to_path_buf()];
    let mut visited_dirs = HashSet::new();
    while let Some(dir_path) = stack.pop() {
        let canonical_dir = dir_path.canonicalize().unwrap_or_else(|_| dir_path.clone());
        if !visited_dirs.insert(canonical_dir) {
            continue;
        }

        // Every skip below leaves part of the tree unobserved, so each one
        // records the path it could not resolve. Skipping silently is what
        // turns an unreadable directory into a subtree of phantom deletes.
        let dir_entries = match fs::read_dir(&dir_path) {
            Ok(entries) => entries,
            Err(error) if is_ignorable_walk_error(&error) => {
                completeness.record_unresolved(&dir_path);
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        for entry in dir_entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if is_ignorable_walk_error(&error) => {
                    // The directory's remaining entries are unknown, so the
                    // directory itself is what went unresolved.
                    completeness.record_unresolved(&dir_path);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) if is_ignorable_walk_error(&error) => {
                    completeness.record_unresolved(&path);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };

            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                // A `NotFound` here is the one skip that is a complete
                // observation: the entry was listed and has since gone, which
                // is exactly the absence a delete should be derived from.
                // Treating it as unresolved would make every scan of a
                // changing tree incomplete and suppress deletes forever.
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) if is_ignorable_walk_error(&error) => {
                    completeness.record_unresolved(&path);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };

            let is_symlink = file_type.is_symlink();
            if is_symlink && !discovery.follow_symlinks {
                continue;
            }

            if metadata.is_dir() {
                let mut directory_candidate =
                    DiscoveryCandidate::new(&path, 0).with_symlink(is_symlink);
                if let Some(category) = lookup_mount_category(mount_table, &path) {
                    directory_candidate = directory_candidate.with_mount_category(category);
                }
                let directory_decision = discovery.evaluate_candidate(&directory_candidate);
                if matches!(directory_decision.scope, DiscoveryScopeDecision::Exclude) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            if !metadata.is_file() {
                continue;
            }

            let mut candidate =
                DiscoveryCandidate::new(&path, metadata.len()).with_symlink(is_symlink);
            if let Some(category) = lookup_mount_category(mount_table, &path) {
                candidate = candidate.with_mount_category(category);
            }
            let decision = discovery.evaluate_candidate(&candidate);
            if matches!(decision.scope, DiscoveryScopeDecision::Exclude)
                || !decision.ingestion_class.is_indexed()
            {
                continue;
            }

            let modified = metadata
                .modified()
                .ok()
                .map(system_time_to_ms)
                .unwrap_or_default();
            snapshot.insert(path, modified);
        }
    }

    Ok(())
}

fn is_ignorable_walk_error(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied | io::ErrorKind::Interrupted
    )
}

fn system_time_to_ms(time: SystemTime) -> u64 {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn now_millis() -> u64 {
    system_time_to_ms(SystemTime::now())
}

fn reduce_batch_size(base_batch_size: usize, divisor: usize) -> usize {
    base_batch_size.saturating_div(divisor.max(1)).max(1)
}

fn normalize_file_key(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

const fn pressure_state_to_code(state: PressureState) -> u8 {
    match state {
        PressureState::Normal => 0,
        PressureState::Constrained => 1,
        PressureState::Degraded => 2,
        PressureState::Emergency => 3,
    }
}

const fn pressure_state_from_code(code: u8) -> PressureState {
    match code {
        1 => PressureState::Constrained,
        2 => PressureState::Degraded,
        3 => PressureState::Emergency,
        _ => PressureState::Normal,
    }
}

#[derive(Default)]
struct PendingEvents {
    by_path: HashMap<PathBuf, WatchEvent>,
    by_time: BTreeMap<u64, BTreeSet<PathBuf>>,
}

impl PendingEvents {
    fn push(&mut self, event: WatchEvent) -> bool {
        let old_event = self.by_path.insert(event.path.clone(), event.clone());
        if let Some(old) = old_event {
            if let Some(paths) = self.by_time.get_mut(&old.observed_at_ms) {
                paths.remove(&old.path);
                if paths.is_empty() {
                    self.by_time.remove(&old.observed_at_ms);
                }
            }
            // Return true because we debounced (replaced) an existing event
            self.by_time
                .entry(event.observed_at_ms)
                .or_default()
                .insert(event.path);
            true
        } else {
            self.by_time
                .entry(event.observed_at_ms)
                .or_default()
                .insert(event.path);
            false
        }
    }

    fn clear(&mut self) -> usize {
        let count = self.by_path.len();
        self.by_path.clear();
        self.by_time.clear();
        count
    }

    fn drain_ready(&mut self, now_ms: u64, debounce_ms: u64, batch_size: usize) -> Vec<WatchEvent> {
        if batch_size == 0 {
            return Vec::new();
        }

        let cutoff = now_ms.saturating_sub(debounce_ms);
        let mut ready_events = Vec::new();

        // Split off everything up to (and including) cutoff.
        // split_off returns keys >= cutoff + 1 (strictly greater than cutoff).
        // So we keep the "future" part in self.by_time, and take the "past" part.
        // Wait, split_off returns everything AFTER the key.
        // We want to remove everything BEFORE the key.
        // BTreeMap doesn't have split_off_before.
        // We have to iterate keys.

        // Since we want to limit by batch_size, we can't just take everything.
        // We must iterate and stop when we hit batch_size.

        let mut timestamps_to_remove = Vec::new();
        let mut paths_to_remove = Vec::new();

        'outer: for (&ts, paths) in &self.by_time {
            if ts > cutoff {
                break;
            }

            for path in paths {
                if ready_events.len() >= batch_size {
                    break 'outer;
                }
                if let Some(event) = self.by_path.remove(path) {
                    ready_events.push(event);
                    paths_to_remove.push((ts, path.clone()));
                }
            }

            // If we didn't break 'outer, it means we consumed all paths for this timestamp.
            // We can mark the timestamp for removal (if we are sure we took all paths).
            // But if we broke 'outer inside the inner loop, we might have left some paths.
            // It's safer to remove paths individually or check if paths is empty.
        }

        // Cleanup by_time
        for (ts, path) in paths_to_remove {
            if let Some(paths) = self.by_time.get_mut(&ts) {
                paths.remove(&path);
                if paths.is_empty() {
                    timestamps_to_remove.push(ts);
                }
            }
        }

        // Use a set to dedup timestamps to remove, though order matters for remove? No.
        // But timestamps_to_remove might contain duplicates if we iterate multiple paths.
        // BTreeMap remove is safe.
        for ts in timestamps_to_remove {
            // Check again if empty, because we might have added it multiple times
            if let Some(paths) = self.by_time.get(&ts)
                && paths.is_empty()
            {
                self.by_time.remove(&ts);
            }
        }

        ready_events
    }

    fn drain_all(&mut self, batch_size: usize) -> Vec<WatchEvent> {
        self.drain_ready(u64::MAX, 0, batch_size)
    }

    fn earliest_observed_at(&self) -> Option<u64> {
        self.by_time.keys().next().copied()
    }
}

fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BATCH_SIZE, DEFAULT_DEBOUNCE_MS, FileSnapshot, FsWatcher, NoopWatchIngestPipeline,
        PendingBatchLease, PendingEvents, ReadyBatchQueue, ReconciliationState,
        ReconciliationTracker, ScanCompleteness, WatchBatchOutcome, WatchEvent, WatchEventKind,
        WatchIngestFuture, WatchIngestOp, WatchIngestPipeline, WatcherExecutionPolicy,
        WatcherLifecycle, WatcherStop, collect_snapshot_from_roots, drain_notify_channel,
        flush_pending_batches, normalize_file_key, now_millis, observe_pressure_transition,
        run_authoritative_reconciliation, run_ingest_loop,
    };
    use crate::config::DiscoveryConfig;

    /// `WatcherStop::request` must publish the flag and the notify inside
    /// `wait_lock`, so a waiter cannot be skipped between its check and its
    /// park.
    ///
    /// The interleaving is forced rather than raced: the test holds `wait_lock`
    /// itself, which is exactly the state `wait_or_stopped` is in after it has
    /// read the flag as `false` and before `wait_timeout` releases the lock.
    /// A `request()` that publishes outside the lock returns immediately in
    /// that state — its store and its notify both land in the waiter's blind
    /// window, and the real waiter then sleeps the full backoff with stop
    /// already requested. The corrected `request()` cannot return until the
    /// lock is free, so the observation below is deterministic in both
    /// directions: `false` before release, `true` after the join.
    #[test]
    fn stop_request_cannot_publish_inside_the_waiters_check_then_park_window() {
        let stop = Arc::new(WatcherStop::default());
        let published = Arc::new(AtomicBool::new(false));

        let held = super::lock_or_recover(&stop.wait_lock);

        let requester = {
            let stop = Arc::clone(&stop);
            let published = Arc::clone(&published);
            thread::spawn(move || {
                stop.request();
                published.store(true, Ordering::Release);
            })
        };

        // Give the requester every chance to publish. On the pre-fix code it
        // does, because nothing stops it from storing and notifying while this
        // thread owns the window; that is the lost wakeup.
        thread::sleep(Duration::from_millis(50));
        assert!(
            !published.load(Ordering::Acquire),
            "stop publication escaped the waiter's check-then-park window; a waiter \
             parked here would miss the notify and sleep the entire backoff"
        );
        assert!(
            !stop.is_requested(),
            "the stop flag became visible before the wait window closed"
        );

        drop(held);
        requester.join().expect("stop requester thread");
        assert!(published.load(Ordering::Acquire));
        assert!(stop.is_requested());
    }

    /// A waiter already parked in `wait_or_stopped` returns on the notify
    /// rather than serving out its timeout.
    #[test]
    fn parked_waiter_wakes_on_stop_instead_of_sleeping_the_backoff() {
        const BACKOFF: Duration = Duration::from_secs(30);
        let stop = Arc::new(WatcherStop::default());

        let waiter = {
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let started = std::time::Instant::now();
                let stopped = stop.wait_or_stopped(BACKOFF);
                (stopped, started.elapsed())
            })
        };

        thread::sleep(Duration::from_millis(50));
        stop.request();

        let (stopped, elapsed) = waiter.join().expect("stop waiter thread");
        assert!(stopped, "wait_or_stopped must report the requested stop");
        assert!(
            elapsed < BACKOFF / 2,
            "waiter served {elapsed:?} of a {BACKOFF:?} backoff instead of waking on stop"
        );
    }
    use crate::pressure::PressureState;
    use asupersync::Cx;
    use asupersync::runtime::RuntimeBuilder;
    use asupersync::test_utils::run_test_with_cx;
    use asupersync::types::CancelKind;
    use frankensearch_core::SearchError;
    use notify::event::{CreateKind, ModifyKind, RenameMode};
    use notify::{Event, EventKind};
    use std::collections::{BTreeSet, HashMap, VecDeque};
    use std::fs;
    use std::future::Future;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    use crate::mount_info::{FsCategory, MountTable};

    #[derive(Default)]
    struct RecordingPipeline {
        batches: Mutex<Vec<Vec<WatchIngestOp>>>,
        attempts: Mutex<Vec<Vec<WatchIngestOp>>>,
        fail_next: AtomicBool,
        stop_on_success: Mutex<Option<Arc<WatcherStop>>>,
        /// Cancellation state of the `Cx` the sink was actually handed.
        observed_cancelled: AtomicBool,
    }

    impl RecordingPipeline {
        fn all_ops(&self) -> Vec<WatchIngestOp> {
            lock_or_recover(&self.batches)
                .iter()
                .flat_map(|batch| batch.iter().cloned())
                .collect()
        }
    }

    impl WatchIngestPipeline for RecordingPipeline {
        fn apply_batch<'a>(
            &'a self,
            cx: &'a Cx,
            batch: &'a [WatchIngestOp],
        ) -> WatchIngestFuture<'a, usize> {
            Box::pin(async move {
                lock_or_recover(&self.attempts).push(batch.to_vec());
                // Recording the observed cancellation state is what lets the
                // lineage test below prove the caller's `Cx` actually arrives
                // here rather than a freshly minted one.
                self.observed_cancelled
                    .store(cx.is_cancel_requested(), Ordering::Release);
                if self.fail_next.swap(false, Ordering::AcqRel) {
                    return Err(frankensearch_core::SearchError::SubsystemError {
                        subsystem: "test",
                        source: Box::new(io::Error::other("forced failure")),
                    });
                }

                lock_or_recover(&self.batches).push(batch.to_vec());
                if let Some(stop) = lock_or_recover(&self.stop_on_success).as_ref() {
                    stop.request();
                }
                Ok(batch.len())
            })
        }
    }

    #[derive(Default)]
    struct PartialMutationPipeline {
        attempts: AtomicUsize,
        applied: Mutex<Vec<WatchIngestOp>>,
        fail_after_first: AtomicBool,
        stop_on_success: Mutex<Option<Arc<WatcherStop>>>,
    }

    impl WatchIngestPipeline for PartialMutationPipeline {
        fn apply_batch<'a>(
            &'a self,
            _cx: &'a Cx,
            batch: &'a [WatchIngestOp],
        ) -> WatchIngestFuture<'a, usize> {
            Box::pin(async move {
                self.attempts.fetch_add(1, Ordering::AcqRel);
                if self.fail_after_first.swap(false, Ordering::AcqRel) {
                    if let Some(first) = batch.first() {
                        lock_or_recover(&self.applied).push(first.clone());
                    }
                    return Err(SearchError::SubsystemError {
                        subsystem: "test",
                        source: Box::new(io::Error::other("failed after first live mutation")),
                    });
                }
                lock_or_recover(&self.applied).extend_from_slice(batch);
                if let Some(stop) = lock_or_recover(&self.stop_on_success).as_ref() {
                    stop.request();
                }
                Ok(batch.len())
            })
        }
    }

    #[derive(Default)]
    struct PermanentFailurePipeline {
        attempts: AtomicUsize,
    }

    impl WatchIngestPipeline for PermanentFailurePipeline {
        fn apply_batch<'a>(
            &'a self,
            _cx: &'a Cx,
            _batch: &'a [WatchIngestOp],
        ) -> WatchIngestFuture<'a, usize> {
            Box::pin(async move {
                self.attempts.fetch_add(1, Ordering::AcqRel);
                Err(SearchError::InvalidConfig {
                    field: "watch.test".to_owned(),
                    value: "poison".to_owned(),
                    reason: "permanent batch failure".to_owned(),
                })
            })
        }
    }

    struct EpochAdvancingPipeline {
        reconciliation: ReconciliationTracker,
        late_event: WatchEvent,
        advance_once: AtomicBool,
        attempts: AtomicUsize,
    }

    impl WatchIngestPipeline for EpochAdvancingPipeline {
        fn apply_batch<'a>(
            &'a self,
            _cx: &'a Cx,
            batch: &'a [WatchIngestOp],
        ) -> WatchIngestFuture<'a, usize> {
            Box::pin(async move {
                self.attempts.fetch_add(1, Ordering::AcqRel);
                if self.advance_once.swap(false, Ordering::AcqRel) {
                    lock_or_recover(&self.reconciliation)
                        .require_for_events(std::slice::from_ref(&self.late_event));
                }
                Ok(batch.len())
            })
        }
    }

    #[derive(Default)]
    struct CancellationProbePipeline {
        child_cx: Mutex<Option<Cx>>,
        started: AtomicBool,
        future_dropped: AtomicBool,
        attempts: AtomicUsize,
    }

    struct CancellationProbeDropGuard<'a> {
        dropped: &'a AtomicBool,
    }

    impl Drop for CancellationProbeDropGuard<'_> {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    impl WatchIngestPipeline for CancellationProbePipeline {
        fn apply_batch<'a>(
            &'a self,
            cx: &'a Cx,
            _batch: &'a [WatchIngestOp],
        ) -> WatchIngestFuture<'a, usize> {
            Box::pin(async move {
                let _drop_guard = CancellationProbeDropGuard {
                    dropped: &self.future_dropped,
                };
                self.attempts.fetch_add(1, Ordering::AcqRel);
                *lock_or_recover(&self.child_cx) = Some(cx.clone());
                self.started.store(true, Ordering::Release);

                loop {
                    if cx.is_cancel_requested() {
                        return Err(SearchError::Cancelled {
                            phase: "watch.test-probe".to_owned(),
                            reason: cx.cancel_reason().map_or_else(
                                || "probe cancelled".to_owned(),
                                |reason| reason.to_string(),
                            ),
                        });
                    }
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
            })
        }
    }

    #[test]
    fn debounce_queue_coalesces_rapid_events_for_same_path() {
        let mut pending = PendingEvents::default();
        let path = PathBuf::from("/tmp/doc.md");

        assert!(!pending.push(WatchEvent::created(path.clone(), 100, Some(10))));
        assert!(pending.push(WatchEvent::modified(path, 120, Some(20))));

        let ready = pending.drain_ready(700, 500, 10);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].kind, WatchEventKind::Modified);
        assert_eq!(ready[0].observed_at_ms, 120);
    }

    #[test]
    fn exclusion_patterns_filter_node_modules_git_and_target() {
        run_test_with_cx(|cx| async move {
            let pipeline = Arc::new(RecordingPipeline::default());
            let watcher = FsWatcher::new(
                vec![PathBuf::from("/tmp/repo")],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );

            let events = [
                WatchEvent::modified("/tmp/repo/node_modules/pkg/index.js", 1_000, Some(128)),
                WatchEvent::modified("/tmp/repo/.git/config", 1_001, Some(32)),
                WatchEvent::modified("/tmp/repo/target/debug/app", 1_002, Some(64)),
            ];
            let outcome = watcher
                .process_events_now(&cx, &events)
                .await
                .expect("process excluded paths");

            assert_eq!(
                outcome,
                WatchBatchOutcome {
                    accepted: 0,
                    reindexed: 0,
                    skipped: 3
                }
            );
            assert!(pipeline.all_ops().is_empty());
        });
    }

    #[test]
    fn binary_files_are_filtered_by_discovery_classifier() {
        run_test_with_cx(|cx| async move {
            let pipeline = Arc::new(RecordingPipeline::default());
            let watcher = FsWatcher::new(
                vec![PathBuf::from("/tmp/repo")],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );

            let event = WatchEvent::modified("/tmp/repo/assets/image.png", 1_000, Some(2048));
            let outcome = watcher
                .process_events_now(&cx, &[event])
                .await
                .expect("process binary");
            assert_eq!(outcome.accepted, 0);
            assert_eq!(outcome.reindexed, 0);
            assert_eq!(outcome.skipped, 1);
            assert!(pipeline.all_ops().is_empty());
        });
    }

    #[test]
    fn notify_event_mount_category_lookup_uses_mount_table() {
        let mount_table = MountTable::new(
            vec![crate::mount_info::MountEntry {
                device: "server:/share".to_owned(),
                mount_point: PathBuf::from("/mnt/nfs"),
                fstype: "nfs".to_owned(),
                category: FsCategory::Nfs,
                options: "rw".to_owned(),
            }],
            &HashMap::new(),
        );

        let event = Event::new(EventKind::Create(CreateKind::Any))
            .add_path(PathBuf::from("/mnt/nfs/project/src/lib.rs"));

        let mapped = super::map_notify_event_with_mount_table(event, Some(&mount_table));
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].mount_category, Some(FsCategory::Nfs));
    }

    #[test]
    fn watcher_stats_track_received_reindexed_and_skipped() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline,
        );

        let events = [
            WatchEvent::modified("/tmp/repo/src/lib.rs", 1_100, Some(256)),
            WatchEvent::modified("/tmp/repo/node_modules/pkg/index.js", 1_101, Some(128)),
        ];
        run_test_with_cx(|cx| async move {
            let outcome = watcher
                .process_events_now(&cx, &events)
                .await
                .expect("process events");
            assert_eq!(outcome.accepted, 1);
            assert_eq!(outcome.reindexed, 1);
            assert_eq!(outcome.skipped, 1);

            let stats = watcher.stats();
            assert_eq!(stats.events_received, 2);
            assert_eq!(stats.files_reindexed, 1);
            assert_eq!(stats.files_skipped, 1);
            assert_eq!(stats.errors, 0);
            assert!(stats.last_event_at.is_some());
        });
    }

    #[test]
    fn pressure_policy_scales_and_disables_watching_when_degraded() {
        let policy_normal = WatcherExecutionPolicy::for_pressure(
            PressureState::Normal,
            DEFAULT_DEBOUNCE_MS,
            DEFAULT_BATCH_SIZE,
        );
        assert_eq!(policy_normal.debounce_ms, DEFAULT_DEBOUNCE_MS);
        assert_eq!(policy_normal.batch_size, DEFAULT_BATCH_SIZE);
        assert!(policy_normal.watching_enabled);

        let policy_constrained = WatcherExecutionPolicy::for_pressure(
            PressureState::Constrained,
            DEFAULT_DEBOUNCE_MS,
            DEFAULT_BATCH_SIZE,
        );
        assert_eq!(policy_constrained.debounce_ms, DEFAULT_DEBOUNCE_MS * 2);
        assert_eq!(policy_constrained.batch_size, DEFAULT_BATCH_SIZE / 2);
        assert!(policy_constrained.watching_enabled);

        let policy_degraded = WatcherExecutionPolicy::for_pressure(
            PressureState::Degraded,
            DEFAULT_DEBOUNCE_MS,
            DEFAULT_BATCH_SIZE,
        );
        assert!(!policy_degraded.watching_enabled);
        assert_eq!(policy_degraded.batch_size, DEFAULT_BATCH_SIZE / 10);
    }

    #[test]
    fn process_events_short_circuits_when_pressure_disables_watching() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );
        watcher.apply_pressure_state(PressureState::Degraded);

        run_test_with_cx(|cx| async move {
            let event = WatchEvent::modified("/tmp/repo/src/lib.rs", now_millis(), Some(128));
            let outcome = watcher
                .process_events_now(&cx, &[event])
                .await
                .expect("degraded process");
            assert_eq!(outcome.accepted, 0);
            assert_eq!(outcome.reindexed, 0);
            assert_eq!(outcome.skipped, 1);
            assert!(pipeline.all_ops().is_empty());
        });
    }

    #[test]
    fn diff_snapshots_detects_create_modify_and_delete() {
        let mut previous = FileSnapshot::new();
        previous.insert(PathBuf::from("/repo/a.rs"), 10);
        previous.insert(PathBuf::from("/repo/b.rs"), 20);

        let mut current = FileSnapshot::new();
        current.insert(PathBuf::from("/repo/a.rs"), 11);
        current.insert(PathBuf::from("/repo/c.rs"), 30);

        let events =
            FsWatcher::diff_snapshots(&previous, &current, 1_000, &ScanCompleteness::default());
        assert_eq!(events.len(), 3);

        let mut kinds = events
            .iter()
            .map(|event| (event.path.clone(), event.kind))
            .collect::<Vec<_>>();
        kinds.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(
            kinds,
            vec![
                (PathBuf::from("/repo/a.rs"), WatchEventKind::Modified),
                (PathBuf::from("/repo/b.rs"), WatchEventKind::Deleted),
                (PathBuf::from("/repo/c.rs"), WatchEventKind::Created),
            ]
        );
    }

    #[test]
    fn collect_snapshot_excludes_binary_and_noise_paths() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let src_dir = root.join("src");
        let node_modules_dir = root.join("node_modules").join("pkg");

        fs::create_dir_all(&src_dir).expect("create src");
        fs::create_dir_all(&node_modules_dir).expect("create node_modules");
        fs::write(src_dir.join("lib.rs"), "fn main() {}\n").expect("write source");
        fs::write(node_modules_dir.join("index.js"), "module.exports = 1;\n").expect("write js");
        fs::write(root.join("image.png"), [0_u8, 1, 2, 3]).expect("write png");

        let watcher = FsWatcher::new(
            vec![root.clone()],
            DiscoveryConfig::default(),
            Arc::new(NoopWatchIngestPipeline),
        );
        let (snapshot, completeness) = watcher.collect_snapshot().expect("collect snapshot");
        assert!(
            completeness.is_complete(),
            "readable fixture scans complete"
        );

        assert!(snapshot.contains_key(&src_dir.join("lib.rs")));
        assert!(!snapshot.contains_key(&node_modules_dir.join("index.js")));
        assert!(!snapshot.contains_key(&root.join("image.png")));
    }

    #[test]
    fn collect_snapshot_skips_network_root_when_category_is_network() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).expect("create src");
        fs::write(src_dir.join("lib.rs"), "fn main() {}\n").expect("write source");

        let discovery = DiscoveryConfig {
            skip_network_mounts: true,
            ..DiscoveryConfig::default()
        };
        let mount_table = MountTable::new(
            vec![crate::mount_info::MountEntry {
                device: "server:/share".to_owned(),
                mount_point: root.clone(),
                fstype: "nfs".to_owned(),
                category: FsCategory::Nfs,
                options: "rw".to_owned(),
            }],
            &HashMap::new(),
        );

        let mut snapshot = FileSnapshot::new();
        let mut completeness = ScanCompleteness::default();
        super::collect_snapshot_for_root(
            &root,
            &discovery,
            Some(&mount_table),
            &mut snapshot,
            &mut completeness,
        )
        .expect("collect snapshot");
        assert!(snapshot.is_empty(), "network root should be excluded");
        assert!(
            completeness.is_complete(),
            "an excluded root is a complete observation that it is out of scope"
        );
    }

    #[cfg(unix)]
    #[test]
    fn collect_snapshot_skips_root_directory_symlink_when_follow_disabled() {
        let temp = tempdir().expect("tempdir");
        let target_root = temp.path().join("target");
        fs::create_dir_all(&target_root).expect("create target");
        fs::write(target_root.join("lib.rs"), "fn main() {}\n").expect("write source");

        let symlink_root = temp.path().join("linked-root");
        std::os::unix::fs::symlink(&target_root, &symlink_root).expect("create symlink");

        let discovery = DiscoveryConfig {
            follow_symlinks: false,
            ..DiscoveryConfig::default()
        };
        let mut snapshot = FileSnapshot::new();
        let mut completeness = ScanCompleteness::default();
        super::collect_snapshot_for_root(
            &symlink_root,
            &discovery,
            None,
            &mut snapshot,
            &mut completeness,
        )
        .expect("collect snapshot");
        assert!(completeness.is_complete());
        assert!(
            snapshot.is_empty(),
            "root symlink should be skipped when follow_symlinks=false"
        );
    }

    #[cfg(unix)]
    #[test]
    fn collect_snapshot_includes_root_directory_symlink_when_follow_enabled() {
        let temp = tempdir().expect("tempdir");
        let target_root = temp.path().join("target");
        fs::create_dir_all(&target_root).expect("create target");
        fs::write(target_root.join("lib.rs"), "fn main() {}\n").expect("write source");

        let symlink_root = temp.path().join("linked-root");
        std::os::unix::fs::symlink(&target_root, &symlink_root).expect("create symlink");

        let discovery = DiscoveryConfig {
            follow_symlinks: true,
            ..DiscoveryConfig::default()
        };
        let mut snapshot = FileSnapshot::new();
        let mut completeness = ScanCompleteness::default();
        super::collect_snapshot_for_root(
            &symlink_root,
            &discovery,
            None,
            &mut snapshot,
            &mut completeness,
        )
        .expect("collect snapshot");
        assert!(completeness.is_complete());
        assert!(
            snapshot.contains_key(&symlink_root.join("lib.rs")),
            "root symlink contents should be indexed when follow_symlinks=true"
        );
    }

    #[test]
    fn deleted_event_emits_delete_ingest_operation() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );

        run_test_with_cx(|cx| async move {
            let event = WatchEvent::deleted("/tmp/repo/src/lib.rs", 9_999);
            let outcome = watcher
                .process_events_now(&cx, &[event])
                .await
                .expect("delete processing");
            assert_eq!(outcome.accepted, 1);
            assert_eq!(outcome.reindexed, 1);
            assert_eq!(outcome.skipped, 0);

            let ops = pipeline.all_ops();
            assert_eq!(ops.len(), 1);
            assert!(matches!(ops[0], WatchIngestOp::Delete { .. }));
        });
    }

    #[test]
    fn deleted_event_for_excluded_path_still_emits_delete_operation() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );

        run_test_with_cx(|cx| async move {
            let event = WatchEvent::deleted("/tmp/repo/node_modules/pkg/index.js", 7_777);
            let outcome = watcher
                .process_events_now(&cx, &[event])
                .await
                .expect("delete for excluded path");
            assert_eq!(outcome.accepted, 1);
            assert_eq!(outcome.reindexed, 1);
            assert_eq!(outcome.skipped, 0);

            let ops = pipeline.all_ops();
            assert_eq!(ops.len(), 1);
            assert!(matches!(ops[0], WatchIngestOp::Delete { .. }));
        });
    }

    #[test]
    fn rename_notify_event_maps_to_delete_then_create() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path(PathBuf::from("/tmp/repo/src/old.rs"))
            .add_path(PathBuf::from("/tmp/repo/src/new.rs"));

        let mapped = super::map_notify_event(event);
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].kind, WatchEventKind::Deleted);
        assert_eq!(mapped[0].path, PathBuf::from("/tmp/repo/src/old.rs"));
        assert_eq!(mapped[1].kind, WatchEventKind::Created);
        assert_eq!(mapped[1].path, PathBuf::from("/tmp/repo/src/new.rs"));
    }

    #[test]
    fn rename_notify_event_from_maps_to_delete() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::From)))
            .add_path(PathBuf::from("/tmp/repo/src/old.rs"));

        let mapped = super::map_notify_event(event);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].kind, WatchEventKind::Deleted);
        assert_eq!(mapped[0].path, PathBuf::from("/tmp/repo/src/old.rs"));
    }

    #[test]
    fn rename_notify_event_to_maps_to_create() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::To)))
            .add_path(PathBuf::from("/tmp/repo/src/new.rs"));

        let mapped = super::map_notify_event(event);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].kind, WatchEventKind::Created);
        assert_eq!(mapped[0].path, PathBuf::from("/tmp/repo/src/new.rs"));
    }

    #[test]
    fn rename_notify_event_preserves_delete_then_upsert_ingest_mapping() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).expect("create src");

        let old_path = src_dir.join("old.rs");
        let new_path = src_dir.join("new.rs");
        fs::write(&new_path, "fn renamed_symbol() {}\n").expect("write new path");

        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path(old_path.clone())
            .add_path(new_path.clone());
        let mapped = super::map_notify_event(event);

        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(vec![root], DiscoveryConfig::default(), pipeline.clone());
        run_test_with_cx(|cx| async move {
            let outcome = watcher
                .process_events_now(&cx, &mapped)
                .await
                .expect("process rename mapping");
            assert_eq!(outcome.accepted, 2);
            assert_eq!(outcome.reindexed, 2);
            assert_eq!(outcome.skipped, 0);

            let ops = pipeline.all_ops();
            assert_eq!(ops.len(), 2);
            assert!(
                matches!(
                    &ops[0],
                    WatchIngestOp::Delete { file_key, .. }
                        if file_key == &normalize_file_key(&old_path)
                ),
                "rename old path should map to delete op"
            );
            assert!(
                matches!(
                    &ops[1],
                    WatchIngestOp::Upsert { file_key, .. }
                        if file_key == &normalize_file_key(&new_path)
                ),
                "rename new path should map to upsert op"
            );
        });
    }

    /// The caller's `Cx` must reach the ingest sink itself.
    ///
    /// This is the regression guard for the defect this conversion fixes: the
    /// sink used to be handed a `Runtime` and mint `Cx::for_request()` per
    /// batch, so a cancelled caller was indistinguishable from a live one at
    /// the point where indexing actually happens. Asserting on a *cancelled*
    /// context is what makes the check non-vacuous — a freshly minted root
    /// context always reports `false` here, so the old code fails this test
    /// while satisfying every other assertion in this module.
    #[test]
    fn ingest_sink_observes_the_callers_cancellation_not_a_fresh_context() {
        run_test_with_cx(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let file = temp.path().join("lib.rs");
            fs::write(&file, "fn live() {}\n").expect("write live file");
            let pipeline = Arc::new(RecordingPipeline::default());
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );
            let event = WatchEvent::modified(file, now_millis(), Some(128));

            watcher
                .process_events_now(&cx, std::slice::from_ref(&event))
                .await
                .expect("live context processes the batch");
            assert!(
                !pipeline.observed_cancelled.load(Ordering::Acquire),
                "a live caller context must arrive at the sink as live"
            );

            cx.set_cancel_requested(true);
            let _ = watcher.process_events_now(&cx, &[event]).await;
            assert!(
                pipeline.observed_cancelled.load(Ordering::Acquire),
                "the sink must observe the caller's cancellation, which a freshly \
                 minted per-batch context can never report"
            );
        });
    }

    #[test]
    fn rename_both_single_path_emits_only_delete() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path(PathBuf::from("/tmp/repo/src/only.rs"));

        let mapped = super::map_notify_event(event);
        assert_eq!(
            mapped.len(),
            1,
            "single-path Both should produce delete only"
        );
        assert_eq!(mapped[0].kind, WatchEventKind::Deleted);
        assert_eq!(mapped[0].path, PathBuf::from("/tmp/repo/src/only.rs"));
    }

    #[test]
    fn rename_any_existing_file_maps_to_created() {
        let temp = tempdir().expect("tempdir");
        let file = temp.path().join("exists.rs");
        fs::write(&file, "fn main() {}\n").expect("write");

        let event =
            Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Any))).add_path(file.clone());

        let mapped = super::map_notify_event(event);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].kind, WatchEventKind::Created);
        assert_eq!(mapped[0].path, file);
    }

    #[test]
    fn rename_any_missing_file_maps_to_deleted() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Any)))
            .add_path(PathBuf::from("/tmp/nonexistent_rename_target_98765.rs"));

        let mapped = super::map_notify_event(event);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].kind, WatchEventKind::Deleted);
    }

    #[test]
    fn rename_events_survive_debounce_independently() {
        let mut pending = PendingEvents::default();
        let old_path = PathBuf::from("/tmp/repo/src/old.rs");
        let new_path = PathBuf::from("/tmp/repo/src/new.rs");

        // Simulate rename: delete old, create new — distinct paths, no coalescing.
        pending.push(WatchEvent::deleted(old_path, 100));
        pending.push(WatchEvent::created(new_path, 100, Some(42)));

        let ready = pending.drain_ready(700, 500, 10);
        assert_eq!(
            ready.len(),
            2,
            "both rename events should drain independently"
        );

        let kinds: Vec<_> = ready.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&WatchEventKind::Deleted));
        assert!(kinds.contains(&WatchEventKind::Created));
    }

    #[test]
    fn pending_batch_lease_requeues_only_before_live_apply_starts() {
        let first = vec![
            WatchEvent::modified("/tmp/repo/src/a.rs", 100, Some(10)),
            WatchEvent::modified("/tmp/repo/src/b.rs", 110, Some(20)),
        ];
        let second = vec![WatchEvent::deleted("/tmp/repo/src/c.rs", 120)];
        let queue: ReadyBatchQueue =
            Arc::new(Mutex::new(VecDeque::from([first.clone(), second.clone()])));
        let reconciliation: ReconciliationTracker =
            Arc::new(Mutex::new(ReconciliationState::default()));

        let lease = PendingBatchLease::acquire(&queue, &reconciliation).expect("first batch lease");
        assert_eq!(lease.events(), first);
        assert_eq!(lock_or_recover(&queue).front(), Some(&second));
        drop(lease);

        assert_eq!(
            lock_or_recover(&queue).iter().cloned().collect::<Vec<_>>(),
            vec![first, second],
            "dropping a lease must restore the complete batch before later work"
        );
        assert!(!lock_or_recover(&reconciliation).required);
    }

    #[test]
    fn live_apply_drop_disarms_replay_and_requires_reconciliation() {
        let batch = vec![WatchEvent::deleted("/tmp/repo/src/dropped.rs", 200)];
        let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([batch.clone()])));
        let reconciliation: ReconciliationTracker =
            Arc::new(Mutex::new(ReconciliationState::default()));

        let mut lease = PendingBatchLease::acquire(&queue, &reconciliation).expect("batch lease");
        lease.begin_live_apply();
        drop(lease);

        assert!(lock_or_recover(&queue).is_empty());
        let state = lock_or_recover(&reconciliation);
        assert!(state.required);
        assert!(state.affected_paths.contains(&batch[0].path));
    }

    #[test]
    fn panic_after_live_apply_boundary_requires_reconciliation_without_replay() {
        let batch = vec![WatchEvent::deleted("/tmp/repo/src/panic.rs", 300)];
        let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([batch.clone()])));
        let reconciliation: ReconciliationTracker =
            Arc::new(Mutex::new(ReconciliationState::default()));

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let queue = Arc::clone(&queue);
            let reconciliation = Arc::clone(&reconciliation);
            move || {
                let mut lease =
                    PendingBatchLease::acquire(&queue, &reconciliation).expect("batch lease");
                lease.begin_live_apply();
                panic!("hostile ingest unwind after first possible mutation");
            }
        }));
        assert!(unwind.is_err());
        assert!(lock_or_recover(&queue).is_empty());
        assert!(lock_or_recover(&reconciliation).required);
    }

    #[test]
    fn post_first_mutation_failure_converges_through_full_rescan() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let first_path = temp.path().join("a.rs");
            let second_path = temp.path().join("b.rs");
            fs::write(&first_path, "fn a() {}\n").expect("write first");
            fs::write(&second_path, "fn b() {}\n").expect("write second");

            let pipeline = Arc::new(PartialMutationPipeline::default());
            pipeline.fail_after_first.store(true, Ordering::Release);
            let stop = Arc::new(WatcherStop::default());
            *lock_or_recover(&pipeline.stop_on_success) = Some(Arc::clone(&stop));

            let events = vec![
                WatchEvent::modified(&first_path, 100, Some(10)),
                WatchEvent::modified(&second_path, 110, Some(10)),
            ];
            let stale_later_delete = WatchEvent::deleted(&first_path, 120);
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([
                events,
                vec![stale_later_delete],
            ])));
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let stats = Arc::new(super::WatcherStatsInner::default());
            let discovery = DiscoveryConfig::default();
            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stop_for_task = Arc::clone(&stop);
            let stats_for_task = Arc::clone(&stats);
            let reconciliation_for_task = Arc::clone(&reconciliation);
            let root = temp.path().to_path_buf();
            let producer_done = Arc::new(AtomicBool::new(true));
            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    run_ingest_loop(
                        &child_cx,
                        &[root],
                        &discovery,
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                        &reconciliation_for_task,
                        100,
                        &producer_done,
                        &collect_snapshot_from_roots,
                    )
                    .await
                })
                .expect("spawn local ingest task");

            task.join(&cx)
                .await
                .expect("ingest task terminal result")
                .expect("full rescan should converge after partial mutation");

            assert_eq!(pipeline.attempts.load(Ordering::Acquire), 2);
            let applied = lock_or_recover(&pipeline.applied);
            assert_eq!(applied.len(), 3);
            assert!(
                applied
                    .iter()
                    .all(|op| matches!(op, WatchIngestOp::Upsert { .. })),
                "the stale queued delete must be subsumed by the rescan"
            );
            let upserted_files = applied
                .iter()
                .filter_map(|op| match op {
                    WatchIngestOp::Upsert { file_key, .. } => Some(file_key.clone()),
                    WatchIngestOp::Delete { .. } => None,
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(upserted_files.len(), 2);
            drop(applied);
            assert!(lock_or_recover(&queue).is_empty());
            assert!(!lock_or_recover(&reconciliation).required);
            assert_eq!(stats.snapshot().errors, 1);
        });
    }

    #[test]
    fn retryable_record_failure_counts_only_the_reconciled_commit() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let path = temp.path().join("record-retry.rs");
            fs::write(&path, "fn retry() {}\n").expect("write retry fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let stop = Arc::new(WatcherStop::default());
            *lock_or_recover(&pipeline.stop_on_success) = Some(Arc::clone(&stop));
            let queue: ReadyBatchQueue =
                Arc::new(Mutex::new(VecDeque::from([vec![WatchEvent::modified(
                    &path,
                    200,
                    Some(14),
                )]])));
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let stats = Arc::new(super::WatcherStatsInner::default());
            let producer_done = Arc::new(AtomicBool::new(true));
            let snapshot_attempts = Arc::new(AtomicUsize::new(0));

            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stop_for_task = Arc::clone(&stop);
            let reconciliation_for_task = Arc::clone(&reconciliation);
            let stats_for_task = Arc::clone(&stats);
            let snapshot_attempts_for_task = Arc::clone(&snapshot_attempts);
            let root = temp.path().to_path_buf();
            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    let discovery = DiscoveryConfig::default();
                    let snapshot_collector =
                        move |roots: &[PathBuf], discovery: &DiscoveryConfig| {
                            if snapshot_attempts_for_task.fetch_add(1, Ordering::AcqRel) == 0 {
                                return Err(SearchError::Io(io::Error::other(
                                    "injected retryable post-commit snapshot failure",
                                )));
                            }
                            collect_snapshot_from_roots(roots, discovery)
                        };
                    run_ingest_loop(
                        &child_cx,
                        &[root],
                        &discovery,
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                        &reconciliation_for_task,
                        100,
                        &producer_done,
                        &snapshot_collector,
                    )
                    .await
                })
                .expect("spawn retry accounting task");

            task.join(&cx)
                .await
                .expect("retry accounting task terminal result")
                .expect("retryable record failure should reconcile");

            assert_eq!(snapshot_attempts.load(Ordering::Acquire), 2);
            assert_eq!(lock_or_recover(&pipeline.attempts).len(), 2);
            let snapshot = stats.snapshot();
            assert_eq!(snapshot.files_reindexed, 1);
            assert_eq!(snapshot.files_skipped, 0);
            assert_eq!(snapshot.errors, 1);
            assert!(!lock_or_recover(&reconciliation).required);
        });
    }

    #[test]
    fn epoch_mismatch_preserves_baseline_until_the_next_rescan() {
        run_test_with_cx(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let current_path = temp.path().join("current.rs");
            let late_path = temp.path().join("late.rs");
            let prior_path = temp.path().join("prior.rs");
            fs::write(&current_path, "fn current() {}\n").expect("write current fixture");

            let prior_snapshot = FileSnapshot::from([(prior_path.clone(), 7)]);
            let reconciliation: ReconciliationTracker = Arc::new(Mutex::new(ReconciliationState {
                indexed_snapshot: prior_snapshot.clone(),
                baseline_initialized: true,
                required: true,
                affected_paths: BTreeSet::from([prior_path.clone()]),
                epoch: 1,
            }));
            let pipeline = EpochAdvancingPipeline {
                reconciliation: Arc::clone(&reconciliation),
                late_event: WatchEvent::modified(&late_path, 300, Some(12)),
                advance_once: AtomicBool::new(true),
                attempts: AtomicUsize::new(0),
            };
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = super::WatcherStatsInner::default();
            let roots = vec![temp.path().to_path_buf()];
            let discovery = DiscoveryConfig::default();

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                &pipeline,
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
            )
            .await
            .expect("first rescan applies before the injected epoch advance");

            {
                let state = lock_or_recover(&reconciliation);
                assert_eq!(state.epoch, 2);
                assert!(state.required);
                assert_eq!(state.indexed_snapshot, prior_snapshot);
                assert!(state.baseline_initialized);
                assert!(state.affected_paths.contains(&prior_path));
                assert!(state.affected_paths.contains(&late_path));
            }

            fs::write(&late_path, "fn late() {}\n").expect("write late fixture");
            let (expected_snapshot, expected_completeness) =
                collect_snapshot_from_roots(&roots, &discovery)
                    .expect("collect expected second baseline");
            assert!(expected_completeness.is_complete());
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                &pipeline,
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
            )
            .await
            .expect("second rescan advances the stable epoch");

            let state = lock_or_recover(&reconciliation);
            assert_eq!(pipeline.attempts.load(Ordering::Acquire), 2);
            assert_eq!(state.indexed_snapshot, expected_snapshot);
            assert!(state.baseline_initialized);
            assert!(!state.required);
            assert!(state.affected_paths.is_empty());
        });
    }

    #[test]
    fn same_region_task_cancellation_reports_the_natural_user_reason() {
        run_on_runtime_task(|cx| async move {
            let pipeline = Arc::new(CancellationProbePipeline::default());
            let temp = tempdir().expect("tempdir");
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );
            watcher.start(&cx).await.expect("start watcher");
            let batch = vec![WatchEvent::deleted(temp.path().join("cancel.rs"), 400)];
            lock_or_recover(&watcher.ready_batches).push_back(batch.clone());

            for _ in 0..1_000 {
                if pipeline.started.load(Ordering::Acquire) {
                    break;
                }
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            if !pipeline.started.load(Ordering::Acquire) {
                watcher.stop(&cx).await;
                panic!("ingest task did not acquire the queued batch");
            }

            {
                let control = lock_or_recover(&watcher.control);
                let WatcherLifecycle::Running { ingest_task, .. } = &control.lifecycle else {
                    panic!("watcher should be running");
                };
                ingest_task.abort();
            }
            let _stop_error = watcher
                .stop_checked(&cx)
                .await
                .expect_err("cancel must surface");

            let child_cx = lock_or_recover(&pipeline.child_cx)
                .clone()
                .expect("sink observed child context");
            assert_eq!(
                child_cx.cancel_reason().map(|reason| reason.kind),
                Some(CancelKind::User)
            );
            assert!(child_cx.cancelled_by(CancelKind::User));
            assert!(!child_cx.cancelled_by(CancelKind::ParentCancelled));
            let chain = child_cx
                .cancel_chain()
                .map(|reason| reason.kind)
                .collect::<Vec<_>>();
            assert_eq!(chain, vec![CancelKind::User]);
            assert!(pipeline.future_dropped.load(Ordering::Acquire));
            assert_eq!(pipeline.attempts.load(Ordering::Acquire), 1);
            assert!(lock_or_recover(&watcher.ready_batches).is_empty());
            assert!(lock_or_recover(&watcher.reconciliation).required);
            assert!(matches!(
                &lock_or_recover(&watcher.control).lifecycle,
                WatcherLifecycle::Stopped
            ));
        });
    }

    #[test]
    fn concurrent_starts_and_stop_publish_one_generation_and_terminate() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let watcher = Arc::new(FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            ));

            let first_watcher = Arc::clone(&watcher);
            let second_watcher = Arc::clone(&watcher);
            let second_entered = Arc::new(AtomicBool::new(false));
            let second_entered_for_task = Arc::clone(&second_entered);
            let mut first = cx
                .spawn_local(move |child_cx| async move { first_watcher.start(&child_cx).await })
                .expect("spawn first start");
            let mut second = cx
                .spawn_local(move |child_cx| async move {
                    second_entered_for_task.store(true, Ordering::Release);
                    second_watcher.start(&child_cx).await
                })
                .expect("spawn second start");

            for _ in 0..1_000 {
                if second_entered.load(Ordering::Acquire) {
                    break;
                }
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            let stop_watcher = Arc::clone(&watcher);
            let mut stop = cx
                .spawn_local(
                    move |child_cx| async move { stop_watcher.stop_checked(&child_cx).await },
                )
                .expect("spawn concurrent stop");

            first
                .join(&cx)
                .await
                .expect("first start task")
                .expect("first start result");
            second
                .join(&cx)
                .await
                .expect("second start task")
                .expect("second start result");
            stop.join(&cx)
                .await
                .expect("stop task")
                .expect("stop result");
            assert_eq!(lock_or_recover(&watcher.control).next_generation, 1);

            let control = lock_or_recover(&watcher.control);
            assert!(matches!(&control.lifecycle, WatcherLifecycle::Stopped));
        });
    }

    #[test]
    fn permanent_poison_batch_is_not_retried_and_stop_surfaces_failure() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let file = temp.path().join("poison.rs");
            fs::write(&file, "fn poison() {}\n").expect("write poison");
            let pipeline = Arc::new(PermanentFailurePipeline::default());
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );
            watcher.start(&cx).await.expect("start watcher");
            lock_or_recover(&watcher.ready_batches).push_back(vec![WatchEvent::modified(
                file,
                500,
                Some(16),
            )]);

            for _ in 0..1_000 {
                if pipeline.attempts.load(Ordering::Acquire) > 0 {
                    break;
                }
                asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
            }
            let error = watcher
                .stop_checked(&cx)
                .await
                .expect_err("poison failure must surface");
            assert!(error.to_string().contains("permanent batch failure"));
            assert_eq!(pipeline.attempts.load(Ordering::Acquire), 1);
            assert!(lock_or_recover(&watcher.reconciliation).required);
        });
    }

    #[test]
    fn stop_flushes_events_that_have_not_reached_the_debounce_deadline() {
        let temp = tempdir().expect("tempdir");
        let channel_path = temp.path().join("channel.rs");
        fs::write(&channel_path, "fn channel() {}\n").expect("write channel file");
        let mut pending = PendingEvents::default();
        let event = WatchEvent::modified("/tmp/repo/src/debounce.rs", 1_000, Some(8));
        pending.push(event.clone());
        assert!(pending.drain_ready(1_001, 500, 10).is_empty());
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(Event::new(EventKind::Modify(ModifyKind::Data(
            notify::event::DataChange::Any,
        )))
        .add_path(channel_path.clone())))
            .expect("queue channel event");
        let stats = super::WatcherStatsInner::default();
        drain_notify_channel(
            &rx,
            WatcherExecutionPolicy::for_pressure(PressureState::Normal, 500, 10),
            &stats,
            &mut pending,
            None,
        );
        let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));

        flush_pending_batches(&mut pending, &queue, 10);

        let drained = lock_or_recover(&queue)
            .iter()
            .flatten()
            .map(|event| event.path.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(drained, BTreeSet::from([event.path, channel_path]));
        assert!(pending.by_path.is_empty());
    }

    #[test]
    fn pressure_recovery_requires_an_authoritative_rescan() {
        let reconciliation: ReconciliationTracker =
            Arc::new(Mutex::new(ReconciliationState::default()));
        let mut pressure_was_disabled = false;

        observe_pressure_transition(false, &mut pressure_was_disabled, &reconciliation);
        assert!(pressure_was_disabled);
        assert!(!lock_or_recover(&reconciliation).required);
        observe_pressure_transition(true, &mut pressure_was_disabled, &reconciliation);

        assert!(!pressure_was_disabled);
        assert!(lock_or_recover(&reconciliation).required);
    }

    #[test]
    fn stop_signal_interrupts_the_maximum_producer_backoff() {
        let stop = Arc::new(WatcherStop::default());
        let waiter_stop = Arc::clone(&stop);
        let waiter = std::thread::spawn(move || {
            let started = Instant::now();
            assert!(waiter_stop.wait_or_stopped(Duration::from_secs(30)));
            started.elapsed()
        });

        std::thread::sleep(Duration::from_millis(10));
        stop.request();
        let elapsed = waiter.join().expect("backoff waiter must not panic");
        assert!(
            elapsed < Duration::from_secs(1),
            "stop should wake producer backoff promptly, took {elapsed:?}"
        );
    }

    #[test]
    fn dropping_a_running_watcher_signals_its_generation() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );
            watcher.start(&cx).await.expect("start watcher");
            let stop = {
                let control = lock_or_recover(&watcher.control);
                let WatcherLifecycle::Running { stop, .. } = &control.lifecycle else {
                    panic!("existing root should keep watcher generation running");
                };
                Arc::clone(stop)
            };

            drop(watcher);

            assert!(stop.is_requested());
        });
    }

    #[test]
    fn start_replaces_finished_producer_and_ingest_task() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("watched");
            let watcher = FsWatcher::new(
                vec![root.clone()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );

            // Missing root => producer exits quickly and tells ingest to stop.
            watcher.start(&cx).await.expect("initial start");
            asupersync::time::sleep(cx.now(), Duration::from_millis(100)).await;

            {
                let control = lock_or_recover(&watcher.control);
                assert!(match &control.lifecycle {
                    WatcherLifecycle::Stopped => true,
                    WatcherLifecycle::Running { producer, .. } => producer.is_finished(),
                    WatcherLifecycle::Starting { .. } | WatcherLifecycle::Stopping { .. } => false,
                });
            }

            // Create the root and start again. Both terminal handles are drained
            // before the replacement producer/task pair is installed.
            fs::create_dir_all(&root).expect("create watcher root");
            watcher.start(&cx).await.expect("restart watcher");
            asupersync::time::sleep(cx.now(), Duration::from_millis(50)).await;

            {
                let control = lock_or_recover(&watcher.control);
                let finished = match &control.lifecycle {
                    WatcherLifecycle::Running { producer, .. } => producer.is_finished(),
                    _ => panic!("producer handle should exist after restart"),
                };
                assert!(
                    !finished,
                    "watcher should replace finished producer handle on restart"
                );
            }

            watcher
                .stop_checked(&cx)
                .await
                .expect("stop restarted watcher");
        });
    }

    /// An unreadable subtree must not be reported as a complete scan, and the
    /// diff against it must derive no deletes for the files it hid.
    ///
    /// The fixture is deliberately *readable first*: the same tree is scanned
    /// once with the subtree readable and once with it chmod-0, so the only
    /// difference between the two runs is the permission bit. Without the
    /// completeness receipt the second snapshot is simply shorter, and the
    /// diff turns every hidden file into a deletion — which is the failure
    /// this asserts against, not a hypothetical one.
    #[cfg(unix)]
    #[test]
    fn unreadable_subtree_is_incomplete_and_derives_no_deletes() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempdir().expect("tempdir");
        let root = temp.path().join("g3-unreadable-subtree");
        let open_dir = root.join("open");
        let closed_dir = root.join("closed");
        fs::create_dir_all(&open_dir).expect("create open dir");
        fs::create_dir_all(&closed_dir).expect("create closed dir");
        let open_file = open_dir.join("visible.rs");
        let hidden_file = closed_dir.join("hidden.rs");
        fs::write(&open_file, "fn visible() {}\n").expect("write visible fixture");
        fs::write(&hidden_file, "fn hidden() {}\n").expect("write hidden fixture");

        let discovery = DiscoveryConfig::default();
        let roots = vec![root.clone()];

        let (baseline, baseline_completeness) =
            collect_snapshot_from_roots(&roots, &discovery).expect("baseline scan");
        assert!(
            baseline_completeness.is_complete(),
            "the readable control must scan complete"
        );
        assert!(baseline.contains_key(&open_file));
        assert!(
            baseline.contains_key(&hidden_file),
            "the control must see the file the hostile run will hide"
        );

        // Close the subtree without removing or rewriting anything.
        let original = fs::metadata(&closed_dir)
            .expect("read closed dir metadata")
            .permissions();
        fs::set_permissions(&closed_dir, fs::Permissions::from_mode(0o000)).expect("close subtree");

        // Probe the precondition through the same syscall the walk uses. A
        // privileged uid ignores the mode bits, and the hostile condition
        // would silently not exist — the test must fail loudly there rather
        // than pass without ever hiding anything.
        let denial_is_enforced = fs::read_dir(&closed_dir).is_err();

        let (short, short_completeness) =
            collect_snapshot_from_roots(&roots, &discovery).expect("hostile scan still succeeds");

        // Restore before asserting so a failure cannot leave the temp tree
        // undeletable for the harness.
        fs::set_permissions(&closed_dir, original).expect("restore subtree permissions");

        assert!(
            denial_is_enforced,
            "fixture precondition failed: mode 0o000 did not deny read_dir (running as root?), \
             so this test could not hide anything and proves nothing"
        );
        assert!(
            !short_completeness.is_complete(),
            "an unreadable directory must be reported as unresolved, not as an empty one"
        );
        assert!(
            short_completeness
                .unresolved_paths()
                .any(|path| path == closed_dir),
            "the unresolved path must name the directory that could not be read"
        );
        assert!(
            !short.contains_key(&hidden_file),
            "the hostile scan really is short — otherwise this proves nothing"
        );

        let events = FsWatcher::diff_snapshots(&baseline, &short, 1_000, &short_completeness);
        assert!(
            !events
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted),
            "an incomplete scan must derive no deletions, got {events:?}"
        );

        // The same short snapshot with a complete receipt WOULD delete, which
        // is what makes the gate load-bearing rather than decorative.
        let unguarded =
            FsWatcher::diff_snapshots(&baseline, &short, 1_000, &ScanCompleteness::default());
        assert!(
            unguarded
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted && event.path == hidden_file),
            "control: a complete receipt over the same pair does derive the delete"
        );
    }

    /// A root that is not there is unresolved, never an empty directory.
    ///
    /// This is the unmounted-root case: the previous baseline holds real
    /// files, the root then disappears, and a scan that called that "complete
    /// and empty" would delete the entire index.
    #[test]
    fn vanished_root_is_incomplete_and_derives_no_deletes() {
        let temp = tempdir().expect("tempdir");
        let present_root = temp.path().join("g3-present-root");
        let absent_root = temp.path().join("g3-absent-root");
        fs::create_dir_all(&present_root).expect("create present root");
        let present_file = present_root.join("kept.rs");
        fs::write(&present_file, "fn kept() {}\n").expect("write present fixture");

        let discovery = DiscoveryConfig::default();
        let roots = vec![present_root.clone(), absent_root.clone()];

        // `absent_root` is never created: nothing is deleted by this test.
        assert!(!absent_root.exists(), "the absent root must stay absent");

        let (snapshot, completeness) =
            collect_snapshot_from_roots(&roots, &discovery).expect("scan with an absent root");

        assert!(
            !completeness.is_complete(),
            "a missing root must be unresolved, not silently empty"
        );
        assert!(
            completeness
                .unresolved_paths()
                .any(|path| path == absent_root),
            "the unresolved path must name the missing root"
        );
        assert!(
            snapshot.contains_key(&present_file),
            "the readable root is still scanned"
        );

        let mut baseline = snapshot.clone();
        let vanished_file = absent_root.join("was-indexed.rs");
        baseline.insert(vanished_file.clone(), 10);

        let events = FsWatcher::diff_snapshots(&baseline, &snapshot, 2_000, &completeness);
        assert!(
            !events
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted),
            "a missing root must not delete what it used to hold, got {events:?}"
        );

        let unguarded =
            FsWatcher::diff_snapshots(&baseline, &snapshot, 2_000, &ScanCompleteness::default());
        assert!(
            unguarded
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted && event.path == vanished_file),
            "control: a complete receipt over the same pair does derive the delete"
        );
    }

    #[test]
    fn collect_snapshot_supports_file_root() {
        let temp = tempdir().expect("tempdir");
        let file_root = temp.path().join("single.rs");
        fs::write(&file_root, "fn main() {}").expect("write");

        let watcher = FsWatcher::new(
            vec![file_root.clone()],
            DiscoveryConfig::default(),
            Arc::new(NoopWatchIngestPipeline),
        );
        let (snapshot, completeness) = watcher.collect_snapshot().expect("collect snapshot");
        assert!(
            completeness.is_complete(),
            "readable fixture scans complete"
        );

        assert!(snapshot.contains_key(&file_root));
        assert_eq!(snapshot.len(), 1);
    }

    fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        match mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn run_on_runtime_task<F, Fut>(test: F)
    where
        F: FnOnce(Cx) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let scheduler = RuntimeBuilder::current_thread()
            .build()
            .expect("build watcher test runtime");
        let test_task = scheduler.handle().spawn(async move {
            let cx = Cx::current().expect("runtime task installs a spawn-capable Cx");
            test(cx).await;
        });
        scheduler.block_on(test_task);
    }
}
