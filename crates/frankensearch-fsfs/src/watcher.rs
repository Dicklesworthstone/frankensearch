//! Filesystem watcher for live incremental re-indexing.
//!
//! The watcher keeps fsfs indexes fresh by:
//! - coalescing rapid filesystem events via debounce windows,
//! - classifying changed files through discovery policy before ingest,
//! - adapting behavior based on pressure state,
//! - providing deterministic snapshot diffing for crash-recovery catch-up.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
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
const WATCHER_POLL_INTERVAL: Duration = Duration::from_millis(25);
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

/// Ingest sink contract consumed by the watcher.
pub trait WatchIngestPipeline: Send + Sync {
    /// Apply one watcher-produced batch.
    ///
    /// Returns the number of successfully reindexed files.
    ///
    /// # Errors
    ///
    /// Returns any ingest/indexing failure from the downstream pipeline.
    fn apply_batch(&self, batch: &[WatchIngestOp]) -> SearchResult<usize>;
}

/// No-op ingest sink used by tests and dry-run scenarios.
#[derive(Debug, Default)]
pub struct NoopWatchIngestPipeline;

impl WatchIngestPipeline for NoopWatchIngestPipeline {
    fn apply_batch(&self, _batch: &[WatchIngestOp]) -> SearchResult<usize> {
        Ok(0)
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
            last_event_at,
        }
    }
}

#[derive(Default)]
struct WatcherControl {
    stop_flag: Option<Arc<AtomicBool>>,
    worker: Option<thread::JoinHandle<()>>,
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
    #[allow(clippy::unused_async)]
    pub async fn start(&self, cx: &Cx) -> SearchResult<()> {
        if cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "watch.start".to_owned(),
                reason: "cancel requested before start".to_owned(),
            });
        }

        let mut control = lock_or_recover(&self.control);
        if let Some(worker) = control.worker.take() {
            if worker.is_finished() {
                if let Err(error) = worker.join() {
                    warn!(?error, "previous fsfs watcher worker panicked");
                }
                control.stop_flag = None;
            } else {
                control.worker = Some(worker);
                return Ok(());
            }
        }

        let stop_flag = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop_flag);

        let roots = self.roots.clone();
        let discovery = self.discovery.clone();
        let ingest = Arc::clone(&self.ingest);
        let stats = Arc::clone(&self.stats);
        let pressure_state = Arc::clone(&self.pressure_state);
        let base_debounce_ms = self.base_debounce_ms;
        let base_batch_size = self.base_batch_size;

        let worker_stats = Arc::clone(&stats);
        let worker_context = WorkerContext {
            roots,
            discovery,
            ingest,
            stats,
            pressure_state,
            stop_flag: worker_stop,
            base_debounce_ms,
            base_batch_size,
        };

        let worker = thread::Builder::new()
            .name("fsfs-watcher".to_owned())
            .spawn(move || {
                if let Err(error) = run_worker_loop(&worker_context) {
                    worker_stats.add_error();
                    warn!(error = %error, "fsfs watcher worker exited with initialization error");
                }
                worker_stats.watching_dirs.store(0, Ordering::Relaxed);
            })
            .map_err(|error| SearchError::SubsystemError {
                subsystem: WATCHER_SUBSYSTEM,
                source: Box::new(io::Error::other(format!(
                    "failed to spawn watcher worker: {error}"
                ))),
            })?;

        control.stop_flag = Some(stop_flag);
        control.worker = Some(worker);
        drop(control);
        Ok(())
    }

    /// Stop background watch processing.
    #[allow(clippy::unused_async)]
    pub async fn stop(&self) {
        let (stop_flag, worker) = {
            let mut control = lock_or_recover(&self.control);
            (control.stop_flag.take(), control.worker.take())
        };

        if let Some(flag) = stop_flag {
            flag.store(true, Ordering::Release);
        }

        if let Some(worker) = worker
            && let Err(error) = worker.join()
        {
            warn!(?error, "fsfs watcher worker panicked during shutdown");
        }
    }

    /// Process one explicit event batch immediately (without debounce).
    ///
    /// # Errors
    ///
    /// Returns any downstream ingest error.
    pub fn process_events_now(&self, events: &[WatchEvent]) -> SearchResult<WatchBatchOutcome> {
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

        let outcome = process_event_batch(&self.discovery, self.ingest.as_ref(), events)?;
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
        let mut all_paths = BTreeSet::new();
        all_paths.extend(previous.keys().cloned());
        all_paths.extend(current.keys().cloned());

        for path in all_paths {
            match (previous.get(&path), current.get(&path)) {
                (None, Some(_)) => {
                    events.push(WatchEvent::created(path, observed_at_ms, None));
                }
                (Some(_), None) => {
                    events.push(WatchEvent::deleted(path, observed_at_ms));
                }
                (Some(before), Some(after)) if before != after => {
                    events.push(WatchEvent::modified(path, observed_at_ms, None));
                }
                _ => {}
            }
        }

        events
    }
}

struct WorkerContext {
    roots: Vec<PathBuf>,
    discovery: DiscoveryConfig,
    ingest: Arc<dyn WatchIngestPipeline>,
    stats: Arc<WatcherStatsInner>,
    pressure_state: Arc<AtomicU8>,
    stop_flag: Arc<AtomicBool>,
    base_debounce_ms: u64,
    base_batch_size: usize,
}

fn run_worker_loop(context: &WorkerContext) -> SearchResult<()> {
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
    while !context.stop_flag.load(Ordering::Acquire) {
        let policy = WatcherExecutionPolicy::for_pressure(
            pressure_state_from_code(context.pressure_state.load(Ordering::Acquire)),
            context.base_debounce_ms,
            context.base_batch_size,
        );

        match event_rx.recv_timeout(WATCHER_POLL_INTERVAL) {
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
                context.stats.add_skipped(dropped);
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

        match process_event_batch(&context.discovery, context.ingest.as_ref(), &ready) {
            Ok(outcome) => {
                context.stats.add_reindexed(outcome.reindexed);
                context.stats.add_skipped(outcome.skipped);
            }
            Err(error) => {
                context.stats.add_error();
                if should_retry_ingest_error(&error) {
                    let retried = requeue_failed_ready_events(&mut pending, ready);
                    warn!(error = %error, "watcher failed to apply ingest batch");
                    debug!(retried, "watcher requeued failed batch for retry");
                } else {
                    let dropped = ready.len();
                    context.stats.add_skipped(dropped);
                    warn!(
                        error = %error,
                        dropped,
                        "watcher dropped non-retryable ingest batch"
                    );
                }
            }
        }
    }

    Ok(())
}

fn requeue_failed_ready_events(pending: &mut PendingEvents, ready: Vec<WatchEvent>) -> usize {
    let retry_observed_at_ms = now_millis();
    let mut count = 0_usize;
    for mut event in ready {
        event.observed_at_ms = retry_observed_at_ms;
        let _ = pending.push(event);
        count = count.saturating_add(1);
    }
    count
}

fn should_retry_ingest_error(error: &SearchError) -> bool {
    matches!(
        error,
        SearchError::Io(_)
            | SearchError::EmbeddingFailed { .. }
            | SearchError::SearchTimeout { .. }
            | SearchError::Cancelled { .. }
            | SearchError::QueueFull { .. }
            | SearchError::SubsystemError { .. }
    )
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

fn process_event_batch(
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    events: &[WatchEvent],
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
        ingest.apply_batch(&ops)?
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

    let byte_len = event.byte_len.unwrap_or(0);
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
    let metadata = fs::symlink_metadata(root).map_err(SearchError::Io)?;
    if metadata.is_file() {
        let mut candidate = DiscoveryCandidate::new(root, metadata.len());
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
    while let Some(dir_path) = stack.pop() {
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

            if file_type.is_dir() {
                let mut directory_candidate = DiscoveryCandidate::new(&path, 0);
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

            if !file_type.is_file() && !file_type.is_symlink() {
                continue;
            }

            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if is_ignorable_walk_error(&error) => continue,
                Err(error) => return Err(error.into()),
            };

            let mut candidate =
                DiscoveryCandidate::new(&path, metadata.len()).with_symlink(file_type.is_symlink());
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
    by_path: BTreeMap<PathBuf, WatchEvent>,
}

impl PendingEvents {
    fn push(&mut self, event: WatchEvent) -> bool {
        self.by_path.insert(event.path.clone(), event).is_some()
    }

    fn clear(&mut self) -> usize {
        let count = self.by_path.len();
        self.by_path.clear();
        count
    }

    fn drain_ready(&mut self, now_ms: u64, debounce_ms: u64, batch_size: usize) -> Vec<WatchEvent> {
        if batch_size == 0 {
            return Vec::new();
        }

        let mut ready_paths = Vec::new();
        for (path, event) in &self.by_path {
            if ready_paths.len() >= batch_size {
                break;
            }
            let elapsed = now_ms.saturating_sub(event.observed_at_ms);
            if elapsed >= debounce_ms {
                ready_paths.push(path.clone());
            }
        }

        let mut ready_events = Vec::with_capacity(ready_paths.len());
        for path in ready_paths {
            if let Some(event) = self.by_path.remove(&path) {
                ready_events.push(event);
            }
        }
        ready_events
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
        PendingEvents, WatchBatchOutcome, WatchEvent, WatchEventKind, WatchIngestOp,
        WatchIngestPipeline, WatcherExecutionPolicy, normalize_file_key, now_millis,
        requeue_failed_ready_events, should_retry_ingest_error,
    };
    use crate::config::DiscoveryConfig;
    use crate::pressure::PressureState;
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::SearchResult;
    use notify::event::{CreateKind, ModifyKind, RenameMode};
    use notify::{Event, EventKind};
    use std::collections::HashMap;
    use std::fs;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    use crate::mount_info::{FsCategory, MountTable};

    #[derive(Default)]
    struct RecordingPipeline {
        batches: Mutex<Vec<Vec<WatchIngestOp>>>,
        fail_next: AtomicBool,
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
        fn apply_batch(&self, batch: &[WatchIngestOp]) -> SearchResult<usize> {
            if self.fail_next.swap(false, Ordering::AcqRel) {
                return Err(frankensearch_core::SearchError::SubsystemError {
                    subsystem: "test",
                    source: Box::new(io::Error::other("forced failure")),
                });
            }

            lock_or_recover(&self.batches).push(batch.to_vec());
            Ok(batch.len())
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
            .process_events_now(&events)
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
    }

    #[test]
    fn binary_files_are_filtered_by_discovery_classifier() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );

        let event = WatchEvent::modified("/tmp/repo/assets/image.png", 1_000, Some(2048));
        let outcome = watcher
            .process_events_now(&[event])
            .expect("process binary");
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.reindexed, 0);
        assert_eq!(outcome.skipped, 1);
        assert!(pipeline.all_ops().is_empty());
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
        let outcome = watcher.process_events_now(&events).expect("process events");
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.reindexed, 1);
        assert_eq!(outcome.skipped, 1);

        let stats = watcher.stats();
        assert_eq!(stats.events_received, 2);
        assert_eq!(stats.files_reindexed, 1);
        assert_eq!(stats.files_skipped, 1);
        assert_eq!(stats.errors, 0);
        assert!(stats.last_event_at.is_some());
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

        let event = WatchEvent::modified("/tmp/repo/src/lib.rs", now_millis(), Some(128));
        let outcome = watcher
            .process_events_now(&[event])
            .expect("degraded process");
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.reindexed, 0);
        assert_eq!(outcome.skipped, 1);
        assert!(pipeline.all_ops().is_empty());
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

    #[test]
    fn deleted_event_emits_delete_ingest_operation() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );

        let event = WatchEvent::deleted("/tmp/repo/src/lib.rs", 9_999);
        let outcome = watcher
            .process_events_now(&[event])
            .expect("delete processing");
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.reindexed, 1);
        assert_eq!(outcome.skipped, 0);

        let ops = pipeline.all_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], WatchIngestOp::Delete { .. }));
    }

    #[test]
    fn deleted_event_for_excluded_path_still_emits_delete_operation() {
        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = FsWatcher::new(
            vec![PathBuf::from("/tmp/repo")],
            DiscoveryConfig::default(),
            pipeline.clone(),
        );

        let event = WatchEvent::deleted("/tmp/repo/node_modules/pkg/index.js", 7_777);
        let outcome = watcher
            .process_events_now(&[event])
            .expect("delete for excluded path");
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.reindexed, 1);
        assert_eq!(outcome.skipped, 0);

        let ops = pipeline.all_ops();
        assert_eq!(ops.len(), 1);
        assert!(matches!(ops[0], WatchIngestOp::Delete { .. }));
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
        let outcome = watcher
            .process_events_now(&mapped)
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
    fn failed_ready_batch_is_requeued_with_fresh_timestamp() {
        let mut pending = PendingEvents::default();
        let ready = vec![
            WatchEvent::modified("/tmp/repo/src/a.rs", 100, Some(10)),
            WatchEvent::modified("/tmp/repo/src/b.rs", 110, Some(20)),
        ];

        let requeued = requeue_failed_ready_events(&mut pending, ready);
        assert_eq!(requeued, 2);

        let immediately_ready = pending.drain_ready(now_millis(), 500, 10);
        assert!(
            immediately_ready.is_empty(),
            "failed events should be delayed by debounce when requeued"
        );

        let eventually_ready = pending.drain_ready(now_millis().saturating_add(600), 500, 10);
        assert_eq!(eventually_ready.len(), 2);
    }

    #[test]
    fn retry_policy_treats_invalid_config_as_non_retryable() {
        let invalid = SearchError::InvalidConfig {
            field: "file_key".to_owned(),
            value: "../etc/passwd".to_owned(),
            reason: "path escapes target root".to_owned(),
        };
        assert!(!should_retry_ingest_error(&invalid));

        let dimension_mismatch = SearchError::DimensionMismatch {
            expected: 384,
            found: 768,
        };
        assert!(!should_retry_ingest_error(&dimension_mismatch));

        let io_error = SearchError::Io(io::Error::other("temporary failure"));
        assert!(should_retry_ingest_error(&io_error));
    }

    #[test]
    fn start_and_stop_worker_without_runtime_integration() {
        run_test_with_cx(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );

            watcher.start(&cx).await.expect("start watcher");
            watcher.stop().await;
        });
    }

    #[test]
    fn start_replaces_finished_worker_handle() {
        run_test_with_cx(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("watched");
            let watcher = FsWatcher::new(
                vec![root.clone()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );

            // Missing root => worker exits quickly with zero watched dirs.
            watcher.start(&cx).await.expect("initial start");
            std::thread::sleep(std::time::Duration::from_millis(100));

            {
                let control = lock_or_recover(&watcher.control);
                let worker = control
                    .worker
                    .as_ref()
                    .expect("worker handle should be retained");
                assert!(worker.is_finished(), "expected initial worker to have exited");
            }

            // Create the root and start again. This should replace the finished handle.
            fs::create_dir_all(&root).expect("create watcher root");
            watcher.start(&cx).await.expect("restart watcher");
            std::thread::sleep(std::time::Duration::from_millis(50));

            {
                let control = lock_or_recover(&watcher.control);
                let worker = control
                    .worker
                    .as_ref()
                    .expect("worker handle should exist after restart");
                assert!(
                    !worker.is_finished(),
                    "watcher should replace finished worker handle on restart"
                );
            }

            watcher.stop().await;
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
}
