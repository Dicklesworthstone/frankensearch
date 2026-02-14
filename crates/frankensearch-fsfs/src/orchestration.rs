//! Watcher/backfill orchestration with crash-safe resume semantics.
//!
//! This module defines a deterministic state machine for fsfs indexing:
//! 1. Bootstrap/backfill startup strategy for large machines.
//! 2. Bounded queue semantics with explicit backpressure behavior.
//! 3. Replay/resume handling driven by monotonic stream sequence checkpoints.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use crate::catalog::{ReplayDecision, classify_replay_sequence};
use crate::config::{DiscoveryCandidate, DiscoveryConfig, DiscoveryScopeDecision};
use crate::mount_info::ChangeDetectionStrategy;
use crate::pressure::{DegradationStage, PressureState};
use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

/// Runtime orchestration phase for crawl/watch indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestrationPhase {
    Bootstrap,
    Backfill,
    Recovering,
    Draining,
    Watch,
}

/// Backpressure state derived from queue depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureMode {
    Normal,
    HighWatermark,
    Saturated,
}

/// Bounded queue policy for orchestration work items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuePolicy {
    pub high_watermark: usize,
    pub hard_limit: usize,
    pub drop_oldest_on_saturation: bool,
    pub scheduler: SchedulerPolicy,
}

impl QueuePolicy {
    #[must_use]
    pub const fn normalized(self) -> Self {
        let high_watermark = if self.high_watermark == 0 {
            1
        } else {
            self.high_watermark
        };
        let hard_limit = if self.hard_limit < high_watermark {
            high_watermark
        } else {
            self.hard_limit
        };
        Self {
            high_watermark,
            hard_limit,
            drop_oldest_on_saturation: self.drop_oldest_on_saturation,
            scheduler: self.scheduler.normalized(hard_limit),
        }
    }
}

impl Default for QueuePolicy {
    fn default() -> Self {
        Self {
            high_watermark: 4_096,
            hard_limit: 8_192,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy::default(),
        }
    }
}

/// Scheduling mode used when selecting the next queued item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerMode {
    FairShare,
    LatencySensitive,
}

/// Per-lane admission caps for bounded queueing semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneBudget {
    pub backfill: usize,
    pub watch_event: usize,
    pub replay: usize,
}

impl LaneBudget {
    #[must_use]
    pub const fn normalized(self, hard_limit: usize) -> Self {
        let minimum = if hard_limit == 0 { 1 } else { hard_limit };
        let backfill = if self.backfill == 0 {
            minimum
        } else {
            self.backfill
        };
        let watch_event = if self.watch_event == 0 {
            minimum
        } else {
            self.watch_event
        };
        let replay = if self.replay == 0 {
            minimum
        } else {
            self.replay
        };
        Self {
            backfill,
            watch_event,
            replay,
        }
    }
}

impl Default for LaneBudget {
    fn default() -> Self {
        Self {
            backfill: 8_192,
            watch_event: 8_192,
            replay: 8_192,
        }
    }
}

/// Scheduler policy for fairness, starvation protection, and bounded admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerPolicy {
    pub mode: SchedulerMode,
    pub starvation_guard: u8,
    pub lane_budget: LaneBudget,
}

impl SchedulerPolicy {
    #[must_use]
    pub const fn normalized(self, hard_limit: usize) -> Self {
        Self {
            mode: self.mode,
            starvation_guard: if self.starvation_guard == 0 {
                1
            } else {
                self.starvation_guard
            },
            lane_budget: self.lane_budget.normalized(hard_limit),
        }
    }
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self {
            mode: SchedulerMode::FairShare,
            starvation_guard: 3,
            lane_budget: LaneBudget::default(),
        }
    }
}

/// Startup bootstrap plan tuned for corpus scale and CPU budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupBootstrapPlan {
    pub parallel_backfill_workers: usize,
    pub initial_backfill_batch_size: usize,
    pub watcher_activation_threshold_pct: u8,
}

impl StartupBootstrapPlan {
    /// Compute a deterministic startup strategy for a machine's expected scale.
    #[must_use]
    pub fn for_machine(total_roots: usize, cpu_budget_pct: u8) -> Self {
        let cpu_slots = usize::from(cpu_budget_pct.max(10))
            .saturating_div(10)
            .max(1);
        let parallel_backfill_workers = total_roots.max(1).min(cpu_slots.max(1));
        if total_roots > 100_000 {
            return Self {
                parallel_backfill_workers,
                initial_backfill_batch_size: 2_000,
                watcher_activation_threshold_pct: 60,
            };
        }
        if total_roots > 10_000 {
            return Self {
                parallel_backfill_workers,
                initial_backfill_batch_size: 1_000,
                watcher_activation_threshold_pct: 75,
            };
        }
        Self {
            parallel_backfill_workers,
            initial_backfill_batch_size: 256,
            watcher_activation_threshold_pct: 90,
        }
    }
}

/// Source lane that produced a queued item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    Backfill,
    WatchEvent,
    Replay,
}

/// One monotonic work item keyed by stream sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkItem {
    pub stream_seq: i64,
    pub file_key: String,
    pub revision: i64,
    pub kind: WorkKind,
    pub event_ts_ms: u64,
}

/// Crash-safe resume checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeToken {
    pub last_applied_seq: i64,
    pub backlog_depth: usize,
    pub backpressure_mode: BackpressureMode,
    pub phase: OrchestrationPhase,
    pub generated_at_ms: u64,
}

/// Queue push result including backpressure transition state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueuePushResult {
    Enqueued {
        depth: usize,
        mode: BackpressureMode,
    },
    DroppedOldest {
        dropped_seq: i64,
        depth: usize,
        mode: BackpressureMode,
    },
    Rejected {
        reason_code: &'static str,
        mode: BackpressureMode,
    },
}

/// Deterministic watcher/backfill orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestrationState {
    phase: OrchestrationPhase,
    backpressure_mode: BackpressureMode,
    queue: VecDeque<WorkItem>,
    queue_policy: QueuePolicy,
    last_dispatched_kind: Option<WorkKind>,
    consecutive_dispatch_count: u8,
    last_applied_seq: i64,
    startup_plan: StartupBootstrapPlan,
}

impl OrchestrationState {
    #[allow(clippy::missing_const_for_fn)]
    #[must_use]
    pub fn new(queue_policy: QueuePolicy, startup_plan: StartupBootstrapPlan) -> Self {
        Self {
            phase: OrchestrationPhase::Bootstrap,
            backpressure_mode: BackpressureMode::Normal,
            queue: VecDeque::new(),
            queue_policy: queue_policy.normalized(),
            last_dispatched_kind: None,
            consecutive_dispatch_count: 0,
            last_applied_seq: 0,
            startup_plan,
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    #[must_use]
    pub fn from_resume(
        queue_policy: QueuePolicy,
        startup_plan: StartupBootstrapPlan,
        token: &ResumeToken,
    ) -> Self {
        Self {
            phase: OrchestrationPhase::Recovering,
            backpressure_mode: token.backpressure_mode,
            queue: VecDeque::new(),
            queue_policy: queue_policy.normalized(),
            last_dispatched_kind: None,
            consecutive_dispatch_count: 0,
            last_applied_seq: token.last_applied_seq,
            startup_plan,
        }
    }

    #[must_use]
    pub const fn phase(&self) -> OrchestrationPhase {
        self.phase
    }

    #[must_use]
    pub const fn backpressure_mode(&self) -> BackpressureMode {
        self.backpressure_mode
    }

    #[must_use]
    pub const fn last_applied_seq(&self) -> i64 {
        self.last_applied_seq
    }

    #[must_use]
    pub const fn startup_plan(&self) -> StartupBootstrapPlan {
        self.startup_plan
    }

    #[must_use]
    pub fn backlog_depth(&self) -> usize {
        self.queue.len()
    }

    #[allow(clippy::missing_const_for_fn)]
    fn mode_for_depth(&self, depth: usize) -> BackpressureMode {
        if depth >= self.queue_policy.hard_limit {
            return BackpressureMode::Saturated;
        }
        if depth >= self.queue_policy.high_watermark {
            return BackpressureMode::HighWatermark;
        }
        BackpressureMode::Normal
    }

    fn refresh_mode(&mut self) {
        self.backpressure_mode = self.mode_for_depth(self.queue.len());
    }

    fn lane_depth(&self, kind: WorkKind) -> usize {
        self.queue.iter().filter(|item| item.kind == kind).count()
    }

    const fn lane_limit(&self, kind: WorkKind) -> usize {
        match kind {
            WorkKind::Backfill => self.queue_policy.scheduler.lane_budget.backfill,
            WorkKind::WatchEvent => self.queue_policy.scheduler.lane_budget.watch_event,
            WorkKind::Replay => self.queue_policy.scheduler.lane_budget.replay,
        }
    }

    fn fairness_guard_requires_switch(&self) -> bool {
        let Some(last_kind) = self.last_dispatched_kind else {
            return false;
        };
        if self.consecutive_dispatch_count < self.queue_policy.scheduler.starvation_guard {
            return false;
        }
        self.queue.iter().any(|item| item.kind != last_kind)
    }

    fn pick_next_index(&self) -> Option<usize> {
        if self.queue.is_empty() {
            return None;
        }

        if matches!(
            self.queue_policy.scheduler.mode,
            SchedulerMode::LatencySensitive
        ) {
            if let Some(idx) = self
                .queue
                .iter()
                .position(|item| item.kind == WorkKind::WatchEvent)
            {
                return Some(idx);
            }
            if let Some(idx) = self
                .queue
                .iter()
                .position(|item| item.kind == WorkKind::Replay)
            {
                return Some(idx);
            }
            return Some(0);
        }

        if self.fairness_guard_requires_switch()
            && let Some(last_kind) = self.last_dispatched_kind
            && let Some(idx) = self.queue.iter().position(|item| item.kind != last_kind)
        {
            return Some(idx);
        }

        Some(0)
    }

    /// Push one queued work item with deterministic monotonic ordering checks.
    pub fn push_work(&mut self, item: WorkItem) -> QueuePushResult {
        if let Some(last) = self.queue.back()
            && item.stream_seq <= last.stream_seq
        {
            return QueuePushResult::Rejected {
                reason_code: "orchestration.reject.non_monotonic_seq",
                mode: self.backpressure_mode,
            };
        }

        if self.lane_depth(item.kind) >= self.lane_limit(item.kind) {
            return QueuePushResult::Rejected {
                reason_code: "orchestration.reject.lane_budget_exhausted",
                mode: self.backpressure_mode,
            };
        }

        match item.kind {
            WorkKind::Backfill => {
                if matches!(self.phase, OrchestrationPhase::Bootstrap) {
                    self.phase = OrchestrationPhase::Backfill;
                }
            }
            WorkKind::WatchEvent => {
                if matches!(self.phase, OrchestrationPhase::Bootstrap) {
                    self.phase = OrchestrationPhase::Watch;
                }
            }
            WorkKind::Replay => {
                self.phase = OrchestrationPhase::Recovering;
            }
        }

        if self.queue.len() < self.queue_policy.hard_limit {
            self.queue.push_back(item);
            self.refresh_mode();
            return QueuePushResult::Enqueued {
                depth: self.queue.len(),
                mode: self.backpressure_mode,
            };
        }

        if !self.queue_policy.drop_oldest_on_saturation {
            self.backpressure_mode = BackpressureMode::Saturated;
            return QueuePushResult::Rejected {
                reason_code: "orchestration.reject.queue_saturated",
                mode: self.backpressure_mode,
            };
        }

        let dropped_seq = self
            .queue
            .pop_front()
            .map_or(-1, |dropped| dropped.stream_seq);
        self.queue.push_back(item);
        self.refresh_mode();
        QueuePushResult::DroppedOldest {
            dropped_seq,
            depth: self.queue.len(),
            mode: self.backpressure_mode,
        }
    }

    /// Pop the next queued item in deterministic stream order.
    pub fn pop_work(&mut self) -> Option<WorkItem> {
        let item = self
            .pick_next_index()
            .and_then(|index| self.queue.remove(index));
        self.refresh_mode();
        if let Some(ref popped) = item {
            if Some(popped.kind) == self.last_dispatched_kind {
                self.consecutive_dispatch_count = self.consecutive_dispatch_count.saturating_add(1);
            } else {
                self.last_dispatched_kind = Some(popped.kind);
                self.consecutive_dispatch_count = 1;
            }
        }
        if item.is_none() && matches!(self.phase, OrchestrationPhase::Draining) {
            self.phase = OrchestrationPhase::Watch;
        }
        item
    }

    /// Cancel one queued work item by stream sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence is not currently queued.
    pub fn cancel_work(&mut self, stream_seq: i64) -> Result<WorkItem, &'static str> {
        let Some(index) = self
            .queue
            .iter()
            .position(|item| item.stream_seq == stream_seq)
        else {
            return Err("orchestration.cancel.not_found");
        };
        let cancelled = self
            .queue
            .remove(index)
            .ok_or("orchestration.cancel.not_found")?;
        self.refresh_mode();
        if self.queue.is_empty() && matches!(self.phase, OrchestrationPhase::Draining) {
            self.phase = OrchestrationPhase::Watch;
        }
        Ok(cancelled)
    }

    /// Signal that startup backfill enqueueing is complete.
    pub fn mark_backfill_complete(&mut self) {
        self.phase = if self.queue.is_empty() {
            OrchestrationPhase::Watch
        } else {
            OrchestrationPhase::Draining
        };
    }

    /// Classify one replay sequence against the current checkpoint.
    #[must_use]
    pub const fn classify_replay(&self, incoming_seq: i64) -> ReplayDecision {
        classify_replay_sequence(self.last_applied_seq, incoming_seq)
    }

    /// Apply replay sequence with gap rejection semantics.
    ///
    /// # Errors
    ///
    /// Returns an error code when the replay stream has a gap.
    #[allow(clippy::missing_const_for_fn)]
    pub fn apply_replay_seq(&mut self, incoming_seq: i64) -> Result<(), &'static str> {
        match self.classify_replay(incoming_seq) {
            ReplayDecision::ApplyNext { next_checkpoint } => {
                self.last_applied_seq = next_checkpoint;
                Ok(())
            }
            ReplayDecision::Duplicate { .. } => Ok(()),
            ReplayDecision::Gap { .. } => Err("orchestration.replay.gap"),
        }
    }

    /// Capture deterministic resume metadata for crash-safe restart.
    #[must_use]
    pub fn snapshot_resume_token(&self, generated_at_ms: u64) -> ResumeToken {
        ResumeToken {
            last_applied_seq: self.last_applied_seq,
            backlog_depth: self.queue.len(),
            backpressure_mode: self.backpressure_mode,
            phase: self.phase,
            generated_at_ms,
        }
    }
}

/// Filesystem change event kind consumed by [`FsWatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchEventKind {
    Create,
    Modify,
    Delete,
}

/// One incoming filesystem change event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    pub path: PathBuf,
    pub kind: WatchEventKind,
    pub event_ts_ms: u64,
    pub byte_len: u64,
    pub change_detection: ChangeDetectionStrategy,
}

impl WatchEvent {
    #[must_use]
    pub fn new(
        path: impl Into<PathBuf>,
        kind: WatchEventKind,
        event_ts_ms: u64,
        byte_len: u64,
    ) -> Self {
        Self {
            path: path.into(),
            kind,
            event_ts_ms,
            byte_len,
            change_detection: ChangeDetectionStrategy::Watch,
        }
    }

    #[must_use]
    pub const fn with_change_detection(
        mut self,
        change_detection: ChangeDetectionStrategy,
    ) -> Self {
        self.change_detection = change_detection;
        self
    }
}

/// Runtime knobs for watcher debounce and dispatch pacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsWatcherConfig {
    pub debounce_ms: u64,
    pub batch_size: usize,
    pub poll_interval_ms: u64,
}

impl Default for FsWatcherConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 500,
            batch_size: 100,
            poll_interval_ms: 2_000,
        }
    }
}

impl FsWatcherConfig {
    #[must_use]
    pub const fn normalized(self) -> Self {
        let debounce_ms = if self.debounce_ms == 0 {
            1
        } else {
            self.debounce_ms
        };
        let batch_size = if self.batch_size == 0 {
            1
        } else {
            self.batch_size
        };
        let poll_interval_ms = if self.poll_interval_ms == 0 {
            1
        } else {
            self.poll_interval_ms
        };
        Self {
            debounce_ms,
            batch_size,
            poll_interval_ms,
        }
    }

    #[must_use]
    pub const fn throttle_for(
        self,
        pressure_state: PressureState,
        degrade_stage: DegradationStage,
    ) -> WatcherThrottle {
        if matches!(
            degrade_stage,
            DegradationStage::MetadataOnly | DegradationStage::Paused
        ) {
            return WatcherThrottle {
                debounce_ms: self.debounce_ms.saturating_mul(20),
                batch_size: 1,
                suspended: true,
                reason_code: "watcher.throttle.suspended",
            };
        }

        if matches!(degrade_stage, DegradationStage::LexicalOnly)
            || matches!(
                pressure_state,
                PressureState::Degraded | PressureState::Emergency
            )
        {
            let batch_size = if self.batch_size < 10 {
                self.batch_size
            } else {
                10
            };
            return WatcherThrottle {
                debounce_ms: self.debounce_ms.saturating_mul(10),
                batch_size,
                suspended: false,
                reason_code: "watcher.throttle.heavy_pressure",
            };
        }

        if matches!(degrade_stage, DegradationStage::EmbedDeferred)
            || matches!(pressure_state, PressureState::Constrained)
        {
            let batch_size = if self.batch_size < 25 {
                self.batch_size
            } else {
                25
            };
            return WatcherThrottle {
                debounce_ms: self.debounce_ms.saturating_mul(4),
                batch_size,
                suspended: false,
                reason_code: "watcher.throttle.constrained",
            };
        }

        WatcherThrottle {
            debounce_ms: self.debounce_ms,
            batch_size: self.batch_size,
            suspended: false,
            reason_code: "watcher.throttle.normal",
        }
    }
}

/// Effective watcher cadence derived from pressure/degradation signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatcherThrottle {
    pub debounce_ms: u64,
    pub batch_size: usize,
    pub suspended: bool,
    pub reason_code: &'static str,
}

/// Watcher processing counters for control-plane telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WatcherStats {
    pub watching_dirs: usize,
    pub events_received: u64,
    pub events_debounced: u64,
    pub files_reindexed: u64,
    pub files_skipped: u64,
    pub errors: u64,
    pub last_event_ts_ms: Option<u64>,
}

/// Post-debounce action kind emitted to incremental indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatcherActionKind {
    Reindex,
    Tombstone,
}

/// One deterministic watcher dispatch action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatcherAction {
    pub path: PathBuf,
    pub kind: WatcherActionKind,
    pub reason_code: &'static str,
}

/// Result of one watcher flush cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatcherDispatch {
    pub throttle: WatcherThrottle,
    pub reason_code: &'static str,
    pub actions: Vec<WatcherAction>,
    pub deferred_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DebouncedWatchEvent {
    kind: WatchEventKind,
    last_event_ts_ms: u64,
}

/// Checkpoint row for crash catch-up planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedFileCheckpoint {
    pub path: PathBuf,
    pub indexed_mtime_ms: u64,
}

impl IndexedFileCheckpoint {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, indexed_mtime_ms: u64) -> Self {
        Self {
            path: path.into(),
            indexed_mtime_ms,
        }
    }
}

/// Current filesystem observation for catch-up planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedFileState {
    pub path: PathBuf,
    pub observed_mtime_ms: Option<u64>,
}

impl ObservedFileState {
    #[must_use]
    pub fn present(path: impl Into<PathBuf>, observed_mtime_ms: u64) -> Self {
        Self {
            path: path.into(),
            observed_mtime_ms: Some(observed_mtime_ms),
        }
    }

    #[must_use]
    pub fn missing(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            observed_mtime_ms: None,
        }
    }
}

/// Deterministic watcher core for debounce/coalescing and staged dispatch.
#[derive(Debug, Clone)]
pub struct FsWatcher {
    config: FsWatcherConfig,
    discovery: DiscoveryConfig,
    pending: HashMap<PathBuf, DebouncedWatchEvent>,
    stats: WatcherStats,
    running: bool,
}

impl FsWatcher {
    #[must_use]
    pub fn new(config: FsWatcherConfig, discovery: DiscoveryConfig) -> Self {
        Self {
            config: config.normalized(),
            discovery,
            pending: HashMap::new(),
            stats: WatcherStats::default(),
            running: false,
        }
    }

    /// Start watcher processing.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `watching_dirs` is zero.
    #[allow(clippy::unused_async)]
    pub async fn start(&mut self, cx: &Cx, watching_dirs: usize) -> SearchResult<()> {
        let _ = cx;
        if watching_dirs == 0 {
            return Err(SearchError::InvalidConfig {
                field: "watcher.watching_dirs".to_owned(),
                value: "0".to_owned(),
                reason: "must be >= 1".to_owned(),
            });
        }
        self.running = true;
        self.stats.watching_dirs = watching_dirs;
        Ok(())
    }

    /// Stop watcher processing and clear in-memory debounce state.
    #[allow(clippy::unused_async)]
    pub async fn stop(&mut self, cx: &Cx) {
        let _ = cx;
        self.running = false;
        self.stats.watching_dirs = 0;
        self.pending.clear();
    }

    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    #[must_use]
    pub fn pending_depth(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub const fn stats(&self) -> &WatcherStats {
        &self.stats
    }

    /// Ingest one filesystem event into the debounce map.
    ///
    /// Returns the deterministic reason code for queue/skip decisions.
    pub fn enqueue_event(&mut self, event: WatchEvent) -> &'static str {
        self.stats.events_received = self.stats.events_received.saturating_add(1);
        self.stats.last_event_ts_ms = Some(event.event_ts_ms);

        if !self.running {
            self.stats.errors = self.stats.errors.saturating_add(1);
            return "watcher.reject.not_running";
        }

        if matches!(event.change_detection, ChangeDetectionStrategy::Static) {
            self.stats.files_skipped = self.stats.files_skipped.saturating_add(1);
            return "watcher.skip.static_mount";
        }

        if !matches!(event.kind, WatchEventKind::Delete) {
            let candidate = DiscoveryCandidate::new(event.path.as_path(), event.byte_len);
            let decision = self.discovery.evaluate_candidate(&candidate);
            if matches!(decision.scope, DiscoveryScopeDecision::Exclude)
                || !decision.ingestion_class.is_indexed()
            {
                self.stats.files_skipped = self.stats.files_skipped.saturating_add(1);
                return "watcher.skip.discovery_policy";
            }
        }

        let next = DebouncedWatchEvent {
            kind: event.kind,
            last_event_ts_ms: event.event_ts_ms,
        };

        if let Some(previous) = self.pending.insert(event.path, next)
            && previous.last_event_ts_ms <= next.last_event_ts_ms
        {
            self.stats.events_debounced = self.stats.events_debounced.saturating_add(1);
        }

        "watcher.event.queued"
    }

    /// Flush ready debounced events into bounded incremental actions.
    #[must_use]
    pub fn flush_ready(
        &mut self,
        now_ms: u64,
        pressure_state: PressureState,
        degrade_stage: DegradationStage,
    ) -> WatcherDispatch {
        let throttle = self.config.throttle_for(pressure_state, degrade_stage);
        if !self.running {
            return WatcherDispatch {
                throttle,
                reason_code: "watcher.flush.not_running",
                actions: Vec::new(),
                deferred_count: 0,
            };
        }

        if throttle.suspended {
            return WatcherDispatch {
                throttle,
                reason_code: "watcher.flush.suspended",
                actions: Vec::new(),
                deferred_count: 0,
            };
        }

        let mut ready = self
            .pending
            .iter()
            .filter_map(|(path, pending)| {
                let age = now_ms.saturating_sub(pending.last_event_ts_ms);
                (age >= throttle.debounce_ms).then(|| (path.clone(), *pending))
            })
            .collect::<Vec<_>>();

        ready.sort_by(|(left_path, left), (right_path, right)| {
            left.last_event_ts_ms
                .cmp(&right.last_event_ts_ms)
                .then_with(|| left_path.cmp(right_path))
        });

        let deferred_count = ready.len().saturating_sub(throttle.batch_size);
        ready.truncate(throttle.batch_size);

        let mut actions = Vec::with_capacity(ready.len());
        for (path, pending) in ready {
            self.pending.remove(&path);
            let (kind, reason_code) = match pending.kind {
                WatchEventKind::Delete => {
                    (WatcherActionKind::Tombstone, "watcher.action.tombstone")
                }
                WatchEventKind::Create | WatchEventKind::Modify => {
                    (WatcherActionKind::Reindex, "watcher.action.reindex")
                }
            };
            actions.push(WatcherAction {
                path,
                kind,
                reason_code,
            });
        }

        self.stats.files_reindexed = self
            .stats
            .files_reindexed
            .saturating_add(u64::try_from(actions.len()).unwrap_or(u64::MAX));

        let reason_code = if actions.is_empty() {
            "watcher.flush.no_ready_events"
        } else {
            "watcher.flush.dispatched"
        };

        WatcherDispatch {
            throttle,
            reason_code,
            actions,
            deferred_count,
        }
    }

    /// Build catch-up actions for files that changed while watcher was down.
    #[must_use]
    pub fn plan_catchup(
        indexed: &[IndexedFileCheckpoint],
        observed: &[ObservedFileState],
    ) -> Vec<WatcherAction> {
        let mut indexed_by_path = HashMap::new();
        for checkpoint in indexed {
            indexed_by_path.insert(checkpoint.path.clone(), checkpoint.indexed_mtime_ms);
        }

        let mut observed_by_path = HashMap::new();
        for entry in observed {
            observed_by_path.insert(entry.path.clone(), entry.observed_mtime_ms);
        }

        let mut actions = Vec::new();
        for (path, observed_mtime_ms) in &observed_by_path {
            match observed_mtime_ms {
                Some(observed_mtime_ms) => match indexed_by_path.get(path) {
                    Some(indexed_mtime_ms) if observed_mtime_ms <= indexed_mtime_ms => {}
                    Some(_) => actions.push(WatcherAction {
                        path: path.clone(),
                        kind: WatcherActionKind::Reindex,
                        reason_code: "watcher.catchup.mtime_advanced",
                    }),
                    None => actions.push(WatcherAction {
                        path: path.clone(),
                        kind: WatcherActionKind::Reindex,
                        reason_code: "watcher.catchup.new_file",
                    }),
                },
                None => {
                    if indexed_by_path.contains_key(path) {
                        actions.push(WatcherAction {
                            path: path.clone(),
                            kind: WatcherActionKind::Tombstone,
                            reason_code: "watcher.catchup.deleted_since_checkpoint",
                        });
                    }
                }
            }
        }

        for path in indexed_by_path.keys() {
            if !observed_by_path.contains_key(path) {
                actions.push(WatcherAction {
                    path: path.clone(),
                    kind: WatcherActionKind::Tombstone,
                    reason_code: "watcher.catchup.missing_from_snapshot",
                });
            }
        }

        actions.sort_by(|left, right| left.path.cmp(&right.path));
        actions
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use asupersync::test_utils::run_test_with_cx;

    use super::{
        BackpressureMode, FsWatcher, FsWatcherConfig, IndexedFileCheckpoint, LaneBudget,
        ObservedFileState, OrchestrationPhase, OrchestrationState, QueuePolicy, QueuePushResult,
        SchedulerMode, SchedulerPolicy, StartupBootstrapPlan, WatchEvent, WatchEventKind,
        WatcherActionKind, WorkItem, WorkKind,
    };
    use crate::config::DiscoveryConfig;
    use crate::pressure::{DegradationStage, PressureState};

    #[test]
    fn startup_plan_scales_for_large_machine() {
        let plan = StartupBootstrapPlan::for_machine(250_000, 80);
        assert_eq!(plan.initial_backfill_batch_size, 2_000);
        assert_eq!(plan.watcher_activation_threshold_pct, 60);
        assert!(plan.parallel_backfill_workers >= 1);
        assert!(plan.parallel_backfill_workers <= 8);
    }

    #[test]
    fn queue_drops_oldest_when_saturated_if_policy_allows() {
        let policy = QueuePolicy {
            high_watermark: 2,
            hard_limit: 3,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy::default(),
        };
        let startup = StartupBootstrapPlan::for_machine(10_000, 70);
        let mut state = OrchestrationState::new(policy, startup);

        for seq in 1_i64..=3_i64 {
            let result = state.push_work(WorkItem {
                stream_seq: seq,
                file_key: format!("doc/{seq}.md"),
                revision: seq,
                kind: WorkKind::Backfill,
                event_ts_ms: 100 + u64::try_from(seq).unwrap_or(0),
            });
            assert!(matches!(result, QueuePushResult::Enqueued { .. }));
        }
        assert_eq!(state.backpressure_mode(), BackpressureMode::Saturated);
        assert_eq!(state.phase(), OrchestrationPhase::Backfill);

        let dropped = state.push_work(WorkItem {
            stream_seq: 4,
            file_key: "doc/4.md".to_owned(),
            revision: 4,
            kind: WorkKind::WatchEvent,
            event_ts_ms: 104,
        });
        assert_eq!(
            dropped,
            QueuePushResult::DroppedOldest {
                dropped_seq: 1,
                depth: 3,
                mode: BackpressureMode::Saturated
            }
        );

        let non_monotonic = state.push_work(WorkItem {
            stream_seq: 3,
            file_key: "doc/3.md".to_owned(),
            revision: 3,
            kind: WorkKind::WatchEvent,
            event_ts_ms: 105,
        });
        assert_eq!(
            non_monotonic,
            QueuePushResult::Rejected {
                reason_code: "orchestration.reject.non_monotonic_seq",
                mode: BackpressureMode::Saturated
            }
        );
    }

    #[test]
    fn queue_rejects_when_drop_oldest_disabled() {
        let policy = QueuePolicy {
            high_watermark: 1,
            hard_limit: 2,
            drop_oldest_on_saturation: false,
            scheduler: SchedulerPolicy::default(),
        };
        let startup = StartupBootstrapPlan::for_machine(1_000, 40);
        let mut state = OrchestrationState::new(policy, startup);

        for seq in 1_i64..=2_i64 {
            let result = state.push_work(WorkItem {
                stream_seq: seq,
                file_key: format!("doc/{seq}.md"),
                revision: seq,
                kind: WorkKind::Backfill,
                event_ts_ms: 200 + u64::try_from(seq).unwrap_or(0),
            });
            assert!(matches!(result, QueuePushResult::Enqueued { .. }));
        }

        let rejected = state.push_work(WorkItem {
            stream_seq: 3,
            file_key: "doc/3.md".to_owned(),
            revision: 3,
            kind: WorkKind::Backfill,
            event_ts_ms: 203,
        });
        assert_eq!(
            rejected,
            QueuePushResult::Rejected {
                reason_code: "orchestration.reject.queue_saturated",
                mode: BackpressureMode::Saturated
            }
        );
    }

    #[test]
    fn resume_token_roundtrip_and_gap_detection() {
        let policy = QueuePolicy::default();
        let startup = StartupBootstrapPlan::for_machine(50_000, 70);
        let mut state = OrchestrationState::new(policy, startup);
        state
            .apply_replay_seq(1)
            .expect("first replay sequence should apply");
        state
            .apply_replay_seq(2)
            .expect("second replay sequence should apply");

        let generated_at_ms = 42_u64;
        let resume_checkpoint = state.snapshot_resume_token(generated_at_ms);
        let mut resumed = OrchestrationState::from_resume(policy, startup, &resume_checkpoint);
        assert_eq!(resumed.phase(), OrchestrationPhase::Recovering);
        assert_eq!(resumed.last_applied_seq(), 2);
        assert!(resumed.apply_replay_seq(3).is_ok());
        assert_eq!(resumed.last_applied_seq(), 3);
        assert!(resumed.apply_replay_seq(5).is_err());
    }

    #[test]
    fn draining_phase_transitions_to_watch_after_queue_clears() {
        let policy = QueuePolicy {
            high_watermark: 2,
            hard_limit: 4,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy::default(),
        };
        let startup = StartupBootstrapPlan::for_machine(2_000, 60);
        let mut state = OrchestrationState::new(policy, startup);
        for seq in 1_i64..=2_i64 {
            let _ = state.push_work(WorkItem {
                stream_seq: seq,
                file_key: format!("doc/{seq}.md"),
                revision: seq,
                kind: WorkKind::Backfill,
                event_ts_ms: 300 + u64::try_from(seq).unwrap_or(0),
            });
        }
        state.mark_backfill_complete();
        assert_eq!(state.phase(), OrchestrationPhase::Draining);

        while state.pop_work().is_some() {}
        assert_eq!(state.phase(), OrchestrationPhase::Watch);
    }

    #[test]
    fn latency_sensitive_mode_prioritizes_watch_events() {
        let policy = QueuePolicy {
            high_watermark: 8,
            hard_limit: 16,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy {
                mode: SchedulerMode::LatencySensitive,
                starvation_guard: 3,
                lane_budget: LaneBudget::default(),
            },
        };
        let startup = StartupBootstrapPlan::for_machine(5_000, 60);
        let mut state = OrchestrationState::new(policy, startup);
        let _ = state.push_work(WorkItem {
            stream_seq: 1,
            file_key: "doc/backfill.md".to_owned(),
            revision: 1,
            kind: WorkKind::Backfill,
            event_ts_ms: 1,
        });
        let _ = state.push_work(WorkItem {
            stream_seq: 2,
            file_key: "doc/watch.md".to_owned(),
            revision: 2,
            kind: WorkKind::WatchEvent,
            event_ts_ms: 2,
        });

        let first = state.pop_work().expect("watch event should be prioritized");
        assert_eq!(first.kind, WorkKind::WatchEvent);
    }

    #[test]
    fn fair_share_starvation_guard_switches_lanes() {
        let policy = QueuePolicy {
            high_watermark: 8,
            hard_limit: 16,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy {
                mode: SchedulerMode::FairShare,
                starvation_guard: 2,
                lane_budget: LaneBudget::default(),
            },
        };
        let startup = StartupBootstrapPlan::for_machine(5_000, 60);
        let mut state = OrchestrationState::new(policy, startup);
        let _ = state.push_work(WorkItem {
            stream_seq: 1,
            file_key: "doc/backfill-1.md".to_owned(),
            revision: 1,
            kind: WorkKind::Backfill,
            event_ts_ms: 1,
        });
        let _ = state.push_work(WorkItem {
            stream_seq: 2,
            file_key: "doc/backfill-2.md".to_owned(),
            revision: 2,
            kind: WorkKind::Backfill,
            event_ts_ms: 2,
        });
        let _ = state.push_work(WorkItem {
            stream_seq: 3,
            file_key: "doc/watch.md".to_owned(),
            revision: 3,
            kind: WorkKind::WatchEvent,
            event_ts_ms: 3,
        });

        let first = state.pop_work().expect("first pop");
        let second = state.pop_work().expect("second pop");
        let third = state.pop_work().expect("third pop");

        assert_eq!(first.kind, WorkKind::Backfill);
        assert_eq!(second.kind, WorkKind::Backfill);
        assert_eq!(third.kind, WorkKind::WatchEvent);
    }

    #[test]
    fn lane_budget_bounds_admission() {
        let policy = QueuePolicy {
            high_watermark: 8,
            hard_limit: 16,
            drop_oldest_on_saturation: true,
            scheduler: SchedulerPolicy {
                mode: SchedulerMode::FairShare,
                starvation_guard: 2,
                lane_budget: LaneBudget {
                    backfill: 1,
                    watch_event: 16,
                    replay: 16,
                },
            },
        };
        let startup = StartupBootstrapPlan::for_machine(2_000, 60);
        let mut state = OrchestrationState::new(policy, startup);
        assert!(matches!(
            state.push_work(WorkItem {
                stream_seq: 1,
                file_key: "doc/backfill-1.md".to_owned(),
                revision: 1,
                kind: WorkKind::Backfill,
                event_ts_ms: 1,
            }),
            QueuePushResult::Enqueued { .. }
        ));

        let rejected = state.push_work(WorkItem {
            stream_seq: 2,
            file_key: "doc/backfill-2.md".to_owned(),
            revision: 2,
            kind: WorkKind::Backfill,
            event_ts_ms: 2,
        });
        assert_eq!(
            rejected,
            QueuePushResult::Rejected {
                reason_code: "orchestration.reject.lane_budget_exhausted",
                mode: BackpressureMode::Normal
            }
        );
    }

    #[test]
    fn cancel_work_removes_item_and_updates_depth() {
        let policy = QueuePolicy::default();
        let startup = StartupBootstrapPlan::for_machine(2_000, 60);
        let mut state = OrchestrationState::new(policy, startup);
        let _ = state.push_work(WorkItem {
            stream_seq: 1,
            file_key: "doc/backfill-1.md".to_owned(),
            revision: 1,
            kind: WorkKind::Backfill,
            event_ts_ms: 1,
        });
        let _ = state.push_work(WorkItem {
            stream_seq: 2,
            file_key: "doc/watch.md".to_owned(),
            revision: 2,
            kind: WorkKind::WatchEvent,
            event_ts_ms: 2,
        });

        let cancelled = state.cancel_work(1).expect("cancel first item");
        assert_eq!(cancelled.stream_seq, 1);
        assert_eq!(state.backlog_depth(), 1);
        assert!(state.cancel_work(1).is_err());
    }

    #[test]
    fn watcher_debounce_coalesces_rapid_events_for_same_path() {
        run_test_with_cx(|cx| async move {
            let mut watcher =
                FsWatcher::new(FsWatcherConfig::default(), DiscoveryConfig::default());
            watcher.start(&cx, 3).await.expect("watcher start");

            let path = PathBuf::from("/home/tester/src/lib.rs");
            assert_eq!(
                watcher.enqueue_event(WatchEvent::new(
                    path.clone(),
                    WatchEventKind::Create,
                    100,
                    64
                )),
                "watcher.event.queued"
            );
            assert_eq!(
                watcher.enqueue_event(WatchEvent::new(
                    path.clone(),
                    WatchEventKind::Modify,
                    220,
                    72
                )),
                "watcher.event.queued"
            );

            assert_eq!(watcher.pending_depth(), 1);
            assert_eq!(watcher.stats().events_received, 2);
            assert_eq!(watcher.stats().events_debounced, 1);

            let not_ready = watcher.flush_ready(719, PressureState::Normal, DegradationStage::Full);
            assert!(not_ready.actions.is_empty());
            assert_eq!(not_ready.reason_code, "watcher.flush.no_ready_events");

            let ready = watcher.flush_ready(720, PressureState::Normal, DegradationStage::Full);
            assert_eq!(ready.actions.len(), 1);
            assert_eq!(ready.actions[0].path, path);
            assert_eq!(ready.actions[0].kind, WatcherActionKind::Reindex);
            assert_eq!(watcher.pending_depth(), 0);
            assert_eq!(watcher.stats().files_reindexed, 1);

            watcher.stop(&cx).await;
            assert!(!watcher.is_running());
        });
    }

    #[test]
    fn watcher_filters_excluded_and_binary_candidates() {
        run_test_with_cx(|cx| async move {
            let mut watcher =
                FsWatcher::new(FsWatcherConfig::default(), DiscoveryConfig::default());
            watcher.start(&cx, 2).await.expect("watcher start");

            let excluded = watcher.enqueue_event(WatchEvent::new(
                "/home/tester/node_modules/pkg/index.js",
                WatchEventKind::Modify,
                100,
                128,
            ));
            let binary = watcher.enqueue_event(WatchEvent::new(
                "/home/tester/media/logo.png",
                WatchEventKind::Modify,
                120,
                128,
            ));

            assert_eq!(excluded, "watcher.skip.discovery_policy");
            assert_eq!(binary, "watcher.skip.discovery_policy");
            assert_eq!(watcher.pending_depth(), 0);
            assert_eq!(watcher.stats().files_skipped, 2);

            watcher.stop(&cx).await;
        });
    }

    #[test]
    fn watcher_delete_event_maps_to_tombstone_action() {
        run_test_with_cx(|cx| async move {
            let mut watcher =
                FsWatcher::new(FsWatcherConfig::default(), DiscoveryConfig::default());
            watcher.start(&cx, 1).await.expect("watcher start");

            assert_eq!(
                watcher.enqueue_event(WatchEvent::new(
                    "/home/tester/docs/readme.md",
                    WatchEventKind::Delete,
                    100,
                    0,
                )),
                "watcher.event.queued"
            );

            let dispatch = watcher.flush_ready(601, PressureState::Normal, DegradationStage::Full);
            assert_eq!(dispatch.actions.len(), 1);
            assert_eq!(dispatch.actions[0].kind, WatcherActionKind::Tombstone);
            assert_eq!(dispatch.actions[0].reason_code, "watcher.action.tombstone");
            assert_eq!(watcher.stats().files_reindexed, 1);
        });
    }

    #[test]
    fn watcher_pressure_throttle_reduces_batch_and_increases_debounce() {
        run_test_with_cx(|cx| async move {
            let mut watcher =
                FsWatcher::new(FsWatcherConfig::default(), DiscoveryConfig::default());
            watcher.start(&cx, 8).await.expect("watcher start");

            for idx in 0..25_u32 {
                let _ = watcher.enqueue_event(WatchEvent::new(
                    format!("/home/tester/src/file-{idx}.rs"),
                    WatchEventKind::Modify,
                    100,
                    256,
                ));
            }

            let constrained = watcher.flush_ready(
                6_000,
                PressureState::Degraded,
                DegradationStage::LexicalOnly,
            );
            assert_eq!(constrained.throttle.debounce_ms, 5_000);
            assert_eq!(constrained.throttle.batch_size, 10);
            assert!(constrained.actions.len() <= 10);
            assert!(constrained.deferred_count >= 15);
        });
    }

    #[test]
    fn watcher_catchup_plans_new_changed_and_deleted_files() {
        let indexed = vec![
            IndexedFileCheckpoint::new("/corpus/a.md", 100),
            IndexedFileCheckpoint::new("/corpus/b.md", 200),
            IndexedFileCheckpoint::new("/corpus/c.md", 300),
        ];
        let observed = vec![
            ObservedFileState::present("/corpus/a.md", 150),
            ObservedFileState::present("/corpus/b.md", 200),
            ObservedFileState::present("/corpus/d.md", 50),
            ObservedFileState::missing("/corpus/c.md"),
        ];

        let actions = FsWatcher::plan_catchup(&indexed, &observed);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].path, PathBuf::from("/corpus/a.md"));
        assert_eq!(actions[0].kind, WatcherActionKind::Reindex);
        assert_eq!(actions[1].path, PathBuf::from("/corpus/c.md"));
        assert_eq!(actions[1].kind, WatcherActionKind::Tombstone);
        assert_eq!(actions[2].path, PathBuf::from("/corpus/d.md"));
        assert_eq!(actions[2].kind, WatcherActionKind::Reindex);
    }
}
