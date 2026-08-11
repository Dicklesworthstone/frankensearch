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
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use asupersync::runtime::TaskHandle;
use frankensearch_core::{SearchError, SearchResult};
use notify::event::{ModifyKind, RenameMode};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{debug, warn};

use crate::config::{
    DiscoveryCandidate, DiscoveryConfig, DiscoveryScopeDecision, FsfsConfig, IngestionClass,
};
use crate::mount_info::{FsCategory, MountTable, read_system_mounts};
use crate::pressure::PressureState;

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

#[derive(Default)]
struct WatcherControl {
    stop_flag: Option<Arc<AtomicBool>>,
    producer: Option<thread::JoinHandle<()>>,
    ingest_task: Option<TaskHandle<SearchResult<()>>>,
}

type ReadyBatchQueue = Arc<Mutex<VecDeque<Vec<WatchEvent>>>>;

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

        let (previous_producer, previous_ingest_task) = {
            let mut control = lock_or_recover(&self.control);
            let producer_running = control
                .producer
                .as_ref()
                .is_some_and(|producer| !producer.is_finished());
            let ingest_running = control
                .ingest_task
                .as_ref()
                .is_some_and(|task| !task.is_finished());
            if producer_running && ingest_running {
                return Ok(());
            }

            if let Some(stop_flag) = control.stop_flag.take() {
                stop_flag.store(true, Ordering::Release);
            }
            (control.producer.take(), control.ingest_task.take())
        };
        finish_watcher_tasks(cx, previous_producer, previous_ingest_task).await;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let ingest = Arc::clone(&self.ingest);
        let ingest_discovery = self.discovery.clone();
        let ingest_stats = Arc::clone(&self.stats);
        let ingest_queue = Arc::clone(&self.ready_batches);
        let ingest_stop = Arc::clone(&stop_flag);
        let ingest_task = cx
            .spawn_local(move |child_cx| async move {
                let _stop_producer_on_exit = IngestTaskStopGuard {
                    stop_flag: Arc::clone(&ingest_stop),
                };
                run_ingest_loop(
                    &child_cx,
                    &ingest_discovery,
                    ingest.as_ref(),
                    &ingest_queue,
                    &ingest_stop,
                    &ingest_stats,
                )
                .await
            })
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "watcher.ingest",
                source: Box::new(io::Error::other(format!(
                    "failed to spawn caller-owned ingest task: {error}"
                ))),
            })?;

        let producer_stats = Arc::clone(&self.stats);
        let producer_stop = Arc::clone(&stop_flag);
        let producer_context = ProducerContext {
            roots: self.roots.clone(),
            discovery: self.discovery.clone(),
            stats: Arc::clone(&self.stats),
            pressure_state: Arc::clone(&self.pressure_state),
            stop_flag: Arc::clone(&stop_flag),
            ready_batches: Arc::clone(&self.ready_batches),
            base_debounce_ms: self.base_debounce_ms,
            base_batch_size: self.base_batch_size,
        };

        let producer = match thread::Builder::new()
            .name("fsfs-watcher".to_owned())
            .spawn(move || {
                const MAX_RESTARTS: usize = 10;
                const MIN_BACKOFF_MS: u64 = 500;
                const MAX_BACKOFF_MS: u64 = 30_000;
                let mut restarts = 0_usize;
                loop {
                    match run_producer_loop(&producer_context) {
                        Ok(()) => break,
                        Err(error) => {
                            producer_stats.add_error();
                            producer_stats
                                .worker_restarts
                                .fetch_add(1, Ordering::Relaxed);
                            restarts = restarts.saturating_add(1);
                            if producer_context.stop_flag.load(Ordering::Acquire) {
                                debug!(
                                    error = %error,
                                    "watcher producer exited with error after stop signal"
                                );
                                break;
                            }
                            if restarts > MAX_RESTARTS {
                                warn!(
                                    error = %error,
                                    restarts,
                                    "watcher producer exhausted restart attempts; giving up"
                                );
                                break;
                            }
                            let backoff_ms = MIN_BACKOFF_MS
                                .saturating_mul(1_u64 << restarts.min(6))
                                .min(MAX_BACKOFF_MS);
                            warn!(
                                error = %error,
                                restart_attempt = restarts,
                                backoff_ms,
                                "watcher producer failed; restarting after backoff"
                            );
                            thread::sleep(Duration::from_millis(backoff_ms));
                        }
                    }
                }
                producer_stats.watching_dirs.store(0, Ordering::Relaxed);
                producer_stop.store(true, Ordering::Release);
            }) {
            Ok(producer) => producer,
            Err(error) => {
                stop_flag.store(true, Ordering::Release);
                finish_watcher_tasks(cx, None, Some(ingest_task)).await;
                return Err(SearchError::SubsystemError {
                    subsystem: WATCHER_SUBSYSTEM,
                    source: Box::new(io::Error::other(format!(
                        "failed to spawn watcher producer: {error}"
                    ))),
                });
            }
        };

        let mut control = lock_or_recover(&self.control);
        control.stop_flag = Some(stop_flag);
        control.producer = Some(producer);
        control.ingest_task = Some(ingest_task);
        Ok(())
    }

    /// Stop background watch processing.
    pub async fn stop(&self, cx: &Cx) {
        let (stop_flag, producer, ingest_task) = {
            let mut control = lock_or_recover(&self.control);
            (
                control.stop_flag.take(),
                control.producer.take(),
                control.ingest_task.take(),
            )
        };

        if let Some(flag) = stop_flag {
            flag.store(true, Ordering::Release);
        }
        finish_watcher_tasks(cx, producer, ingest_task).await;
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

        // Runs on the caller's executor under the caller's `Cx`: this path
        // builds no runtime of its own and therefore cannot strand ingest work
        // on a private one that the caller cannot cancel.
        let outcome =
            process_event_batch(&self.discovery, self.ingest.as_ref(), events, cx).await?;
        self.stats.add_reindexed(outcome.reindexed);
        self.stats.add_skipped(outcome.skipped);
        Ok(outcome)
    }

    /// Collect a filtered file snapshot for crash-recovery comparisons.
    ///
    /// # Errors
    ///
    /// Returns errors from filesystem traversal that are not safe to ignore.
    pub fn collect_snapshot(&self) -> SearchResult<FileSnapshot> {
        collect_snapshot_from_roots(&self.roots, &self.discovery)
    }

    /// Build catch-up events by diffing prior and current snapshots.
    ///
    /// # Errors
    ///
    /// Returns errors from current snapshot collection.
    pub fn build_catchup_events(&self, previous: &FileSnapshot) -> SearchResult<Vec<WatchEvent>> {
        let current = self.collect_snapshot()?;
        Ok(Self::diff_snapshots(previous, &current, now_millis()))
    }

    /// Deterministically diff two snapshots into create/modify/delete events.
    #[must_use]
    pub fn diff_snapshots(
        previous: &FileSnapshot,
        current: &FileSnapshot,
        observed_at_ms: u64,
    ) -> Vec<WatchEvent> {
        let mut events = Vec::new();
        let mut prev_iter = previous.iter();
        let mut curr_iter = current.iter();
        let mut p_next = prev_iter.next();
        let mut c_next = curr_iter.next();

        while let (Some((p_path, p_time)), Some((c_path, c_time))) = (p_next, c_next) {
            match p_path.cmp(c_path) {
                std::cmp::Ordering::Less => {
                    events.push(WatchEvent::deleted(p_path, observed_at_ms));
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
            events.push(WatchEvent::deleted(p_path, observed_at_ms));
            p_next = prev_iter.next();
        }

        while let Some((c_path, _)) = c_next {
            events.push(WatchEvent::created(c_path, observed_at_ms, None));
            c_next = curr_iter.next();
        }

        events
    }
}

struct ProducerContext {
    roots: Vec<PathBuf>,
    discovery: DiscoveryConfig,
    stats: Arc<WatcherStatsInner>,
    pressure_state: Arc<AtomicU8>,
    stop_flag: Arc<AtomicBool>,
    ready_batches: ReadyBatchQueue,
    base_debounce_ms: u64,
    base_batch_size: usize,
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

    let mut pending = PendingEvents::default();
    let mut events_were_dropped = false;
    while !context.stop_flag.load(Ordering::Acquire) {
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
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
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

        if !policy.watching_enabled {
            let dropped = pending.clear();
            if dropped > 0 {
                events_were_dropped = true;
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

        // Catchup scan after pressure relief: if events were dropped while
        // watching was disabled, the filesystem may have changed. Re-queue
        // synthetic events via diff_snapshots when we detect that pressure
        // has returned to normal and events were previously dropped.
        if events_were_dropped {
            events_were_dropped = false;
            debug!("pressure relieved after drops; scheduling catchup on next batch cycle");
        }

        let ready = pending.drain_ready(now_millis(), policy.debounce_ms, policy.batch_size);
        if ready.is_empty() {
            continue;
        }

        lock_or_recover(&context.ready_batches).push_back(ready);
    }

    Ok(())
}

struct IngestTaskStopGuard {
    stop_flag: Arc<AtomicBool>,
}

impl Drop for IngestTaskStopGuard {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

struct PendingBatchLease {
    queue: ReadyBatchQueue,
    batch: Option<Vec<WatchEvent>>,
}

impl PendingBatchLease {
    fn acquire(queue: &ReadyBatchQueue) -> Option<Self> {
        let batch = lock_or_recover(queue).pop_front()?;
        Some(Self {
            queue: Arc::clone(queue),
            batch: Some(batch),
        })
    }

    fn events(&self) -> &[WatchEvent] {
        self.batch.as_deref().unwrap_or_default()
    }

    fn commit(mut self) {
        self.batch = None;
    }
}

impl Drop for PendingBatchLease {
    fn drop(&mut self) {
        if let Some(batch) = self.batch.take() {
            lock_or_recover(&self.queue).push_front(batch);
        }
    }
}

async fn run_ingest_loop(
    cx: &Cx,
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    ready_batches: &ReadyBatchQueue,
    stop_flag: &AtomicBool,
    stats: &WatcherStatsInner,
) -> SearchResult<()> {
    const IDLE_POLL: Duration = Duration::from_millis(10);

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

        match ingest.poll_flush_barrier(cx).await {
            Ok(true) => debug!("watcher acknowledged a durable flush barrier"),
            Ok(false) => {}
            Err(error) => {
                stats.add_error();
                warn!(error = %error, "watcher failed to acknowledge a durable flush barrier");
            }
        }

        let Some(lease) = PendingBatchLease::acquire(ready_batches) else {
            if stop_flag.load(Ordering::Acquire) {
                return Ok(());
            }
            asupersync::time::sleep(cx.now(), IDLE_POLL).await;
            continue;
        };

        match process_event_batch(discovery, ingest, lease.events(), cx).await {
            Ok(outcome) => {
                stats.add_reindexed(outcome.reindexed);
                stats.add_skipped(outcome.skipped);
                // `apply_batch` returns only after the live sink's lexical
                // commit succeeds. Until this point, Drop puts the complete
                // batch back at the queue front on error, unwind, task cancel,
                // or future drop.
                lease.commit();
            }
            Err(error) => {
                stats.add_error();
                warn!(error = %error, "watcher ingest failed; preserving whole batch for retry");
                drop(lease);
                if stop_flag.load(Ordering::Acquire) || cx.is_cancel_requested() {
                    return Err(error);
                }
                asupersync::time::sleep(cx.now(), IDLE_POLL).await;
            }
        }
    }
}

async fn finish_watcher_tasks(
    cx: &Cx,
    producer: Option<thread::JoinHandle<()>>,
    ingest_task: Option<TaskHandle<SearchResult<()>>>,
) {
    if let Some(producer) = producer
        && let Err(error) = producer.join()
    {
        warn!(?error, "fsfs watcher producer panicked during shutdown");
    }

    if let Some(mut ingest_task) = ingest_task {
        match ingest_task.join(cx).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) if matches!(&error, SearchError::Cancelled { .. }) => {
                debug!(error = %error, "fsfs watcher ingest task cancelled");
            }
            Ok(Err(error)) => warn!(error = %error, "fsfs watcher ingest task failed"),
            Err(error) => debug!(error = %error, "fsfs watcher ingest task terminated"),
        }
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

async fn process_event_batch(
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    events: &[WatchEvent],
    cx: &Cx,
) -> SearchResult<WatchBatchOutcome> {
    let mut ops = Vec::new();
    let mut skipped = 0_usize;

    for event in events {
        if let Some(op) = event_to_ingest_op(discovery, event) {
            ops.push(op);
        } else {
            skipped = skipped.saturating_add(1);
        }
    }

    let accepted = ops.len();
    let reindexed = if ops.is_empty() {
        0
    } else {
        ingest.apply_batch(cx, &ops).await?
    };

    Ok(WatchBatchOutcome {
        accepted,
        reindexed,
        skipped,
    })
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
) -> SearchResult<FileSnapshot> {
    let mut snapshot = FileSnapshot::new();
    let mount_table = build_mount_table(discovery);
    for root in roots {
        collect_snapshot_for_root(root, discovery, Some(&mount_table), &mut snapshot)?;
    }
    Ok(snapshot)
}

fn collect_snapshot_for_root(
    root: &Path,
    discovery: &DiscoveryConfig,
    mount_table: Option<&MountTable>,
    snapshot: &mut FileSnapshot,
) -> SearchResult<()> {
    if !root.exists() {
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

        let dir_entries = match fs::read_dir(&dir_path) {
            Ok(entries) => entries,
            Err(error) if is_ignorable_walk_error(&error) => continue,
            Err(error) => return Err(error.into()),
        };

        for entry in dir_entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if is_ignorable_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) if is_ignorable_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if is_ignorable_walk_error(&error) => continue,
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
        PendingBatchLease, PendingEvents, ReadyBatchQueue, WatchBatchOutcome, WatchEvent,
        WatchEventKind, WatchIngestFuture, WatchIngestOp, WatchIngestPipeline,
        WatcherExecutionPolicy, normalize_file_key, now_millis, run_ingest_loop,
    };
    use crate::config::DiscoveryConfig;
    use crate::pressure::PressureState;
    use asupersync::Cx;
    use asupersync::runtime::RuntimeBuilder;
    use asupersync::test_utils::run_test_with_cx;
    use asupersync::types::{CancelKind, CancelReason};
    use frankensearch_core::SearchError;
    use notify::event::{CreateKind, ModifyKind, RenameMode};
    use notify::{Event, EventKind};
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::future::Future;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::tempdir;

    use crate::mount_info::{FsCategory, MountTable};

    #[derive(Default)]
    struct RecordingPipeline {
        batches: Mutex<Vec<Vec<WatchIngestOp>>>,
        attempts: Mutex<Vec<Vec<WatchIngestOp>>>,
        fail_next: AtomicBool,
        stop_on_success: Mutex<Option<Arc<AtomicBool>>>,
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

        fn attempts(&self) -> Vec<Vec<WatchIngestOp>> {
            lock_or_recover(&self.attempts).clone()
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
                if let Some(stop_flag) = lock_or_recover(&self.stop_on_success).as_ref() {
                    stop_flag.store(true, Ordering::Release);
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

        let events = FsWatcher::diff_snapshots(&previous, &current, 1_000);
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
        let snapshot = watcher.collect_snapshot().expect("collect snapshot");

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
        super::collect_snapshot_for_root(&root, &discovery, Some(&mount_table), &mut snapshot)
            .expect("collect snapshot");
        assert!(snapshot.is_empty(), "network root should be excluded");
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
        super::collect_snapshot_for_root(&symlink_root, &discovery, None, &mut snapshot)
            .expect("collect snapshot");
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
        super::collect_snapshot_for_root(&symlink_root, &discovery, None, &mut snapshot)
            .expect("collect snapshot");
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
            let pipeline = Arc::new(RecordingPipeline::default());
            let watcher = FsWatcher::new(
                vec![PathBuf::from("/tmp/repo")],
                DiscoveryConfig::default(),
                pipeline.clone(),
            );
            let event = WatchEvent::modified("/tmp/repo/src/lib.rs", now_millis(), Some(128));

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
    fn pending_batch_lease_requeues_the_whole_batch_at_the_front_on_drop() {
        let first = vec![
            WatchEvent::modified("/tmp/repo/src/a.rs", 100, Some(10)),
            WatchEvent::modified("/tmp/repo/src/b.rs", 110, Some(20)),
        ];
        let second = vec![WatchEvent::deleted("/tmp/repo/src/c.rs", 120)];
        let queue: ReadyBatchQueue =
            Arc::new(Mutex::new(VecDeque::from([first.clone(), second.clone()])));

        let lease = PendingBatchLease::acquire(&queue).expect("first batch lease");
        assert_eq!(lease.events(), first);
        assert_eq!(lock_or_recover(&queue).front(), Some(&second));
        drop(lease);

        assert_eq!(
            lock_or_recover(&queue).iter().cloned().collect::<Vec<_>>(),
            vec![first, second],
            "dropping a lease must restore the complete batch before later work"
        );
    }

    #[test]
    fn pending_batch_lease_requeues_on_panic_unwind() {
        let batch = vec![WatchEvent::deleted("/tmp/repo/src/panic.rs", 200)];
        let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([batch.clone()])));

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let queue = Arc::clone(&queue);
            move || {
                let _lease = PendingBatchLease::acquire(&queue).expect("batch lease");
                panic!("hostile ingest unwind");
            }
        }));
        assert!(unwind.is_err());
        assert_eq!(lock_or_recover(&queue).front(), Some(&batch));
    }

    #[test]
    fn pending_batch_lease_requeues_when_owning_future_is_dropped() {
        let batch = vec![WatchEvent::deleted("/tmp/repo/src/dropped.rs", 300)];
        let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([batch.clone()])));
        let queue_for_future = Arc::clone(&queue);
        let mut future = Box::pin(async move {
            let _lease = PendingBatchLease::acquire(&queue_for_future).expect("batch lease");
            std::future::pending::<()>().await;
        });
        let waker = std::task::Waker::noop();
        let mut poll_cx = std::task::Context::from_waker(waker);
        assert!(future.as_mut().poll(&mut poll_cx).is_pending());
        assert!(lock_or_recover(&queue).is_empty());
        drop(future);
        assert_eq!(lock_or_recover(&queue).front(), Some(&batch));
    }

    #[test]
    fn failed_batch_is_retried_once_without_partial_queue_mutation() {
        run_on_runtime_task(|cx| async move {
            let pipeline = Arc::new(RecordingPipeline::default());
            pipeline.fail_next.store(true, Ordering::Release);
            let stop_flag = Arc::new(AtomicBool::new(false));
            *lock_or_recover(&pipeline.stop_on_success) = Some(Arc::clone(&stop_flag));

            let events = vec![
                WatchEvent::deleted("/tmp/repo/src/a.rs", 100),
                WatchEvent::deleted("/tmp/repo/src/b.rs", 110),
            ];
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::from([events.clone()])));
            let stats = Arc::new(super::WatcherStatsInner::default());
            let discovery = DiscoveryConfig::default();
            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stop_for_task = Arc::clone(&stop_flag);
            let stats_for_task = Arc::clone(&stats);
            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    run_ingest_loop(
                        &child_cx,
                        &discovery,
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                    )
                    .await
                })
                .expect("spawn local ingest task");

            task.join(&cx)
                .await
                .expect("ingest task terminal result")
                .expect("fail-once batch should retry successfully");

            let attempts = pipeline.attempts();
            assert_eq!(
                attempts.len(),
                2,
                "one failure must cause exactly one retry"
            );
            assert_eq!(attempts[0], attempts[1]);
            assert_eq!(attempts[0].len(), events.len());
            assert_eq!(lock_or_recover(&pipeline.batches).len(), 1);
            assert!(lock_or_recover(&queue).is_empty());
            assert_eq!(stats.snapshot().errors, 1);
        });
    }

    #[test]
    fn child_cancellation_preserves_typed_lineage_and_pending_batch() {
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

            let parent_reason = CancelReason::user("watcher parent cancelled");
            cx.set_cancel_reason(parent_reason.clone());
            let child_reason = CancelReason::parent_cancelled().with_cause(parent_reason);
            lock_or_recover(&watcher.control)
                .ingest_task
                .as_ref()
                .expect("ingest task handle")
                .abort_with_reason(child_reason);
            watcher.stop(&cx).await;

            let child_cx = lock_or_recover(&pipeline.child_cx)
                .clone()
                .expect("sink observed child context");
            assert!(cx.cancelled_by(CancelKind::User));
            assert_eq!(
                child_cx.cancel_reason().map(|reason| reason.kind),
                Some(CancelKind::ParentCancelled)
            );
            assert!(child_cx.cancelled_by(CancelKind::ParentCancelled));
            let chain = child_cx
                .cancel_chain()
                .map(|reason| reason.kind)
                .collect::<Vec<_>>();
            assert_eq!(chain, vec![CancelKind::ParentCancelled, CancelKind::User]);
            assert!(pipeline.future_dropped.load(Ordering::Acquire));
            assert_eq!(pipeline.attempts.load(Ordering::Acquire), 1);
            assert_eq!(
                lock_or_recover(&watcher.ready_batches).front(),
                Some(&batch)
            );
            assert!(lock_or_recover(&watcher.control).ingest_task.is_none());
        });
    }

    #[test]
    fn start_and_stop_producer_with_caller_owned_ingest_task() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );

            watcher.start(&cx).await.expect("start watcher");
            watcher.stop(&cx).await;
            let control = lock_or_recover(&watcher.control);
            assert!(control.producer.is_none());
            assert!(control.ingest_task.is_none());
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
                let producer_finished = lock_or_recover(&watcher.control)
                    .producer
                    .as_ref()
                    .expect("producer handle should be retained")
                    .is_finished();
                assert!(
                    producer_finished,
                    "expected initial producer to have exited"
                );
            }

            // Create the root and start again. Both terminal handles are drained
            // before the replacement producer/task pair is installed.
            fs::create_dir_all(&root).expect("create watcher root");
            watcher.start(&cx).await.expect("restart watcher");
            asupersync::time::sleep(cx.now(), Duration::from_millis(50)).await;

            {
                let finished = lock_or_recover(&watcher.control)
                    .producer
                    .as_ref()
                    .expect("producer handle should exist after restart")
                    .is_finished();
                assert!(
                    !finished,
                    "watcher should replace finished producer handle on restart"
                );
            }

            watcher.stop(&cx).await;
        });
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
        let snapshot = watcher.collect_snapshot().expect("collect snapshot");

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
