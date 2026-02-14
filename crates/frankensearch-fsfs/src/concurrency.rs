//! Concurrency model, lock ordering, and contention policy for the fsfs indexing pipeline.
//!
//! This module defines the canonical concurrency strategy for all shared resources
//! in the fsfs pipeline: `FrankenSQLite` catalog, FSVI vector indices, Tantivy lexical
//! indices, the embedding queue, and the index cache. It codifies lock ordering to
//! prevent deadlocks, provides contention mitigation via backoff, and includes
//! crash-recovery primitives for stale lock detection.
//!
//! # Lock Ordering Convention
//!
//! All locks must be acquired in ascending [`LockLevel`] order. Acquiring a lock
//! at a level lower than or equal to an already-held lock is a programming error
//! and will panic in debug builds via [`LockOrderGuard`].
//!
//! The canonical order is:
//!
//! 1. **Catalog** — `FrankenSQLite` metadata database (row-level MVCC, rarely contended)
//! 2. **`EmbeddingQueue`** — In-memory job queue (short critical sections)
//! 3. **`IndexCache`** — Atomic index snapshot (Arc swap under `RwLock`)
//! 4. **`FsviSegment`** — Per-segment vector index file locks
//! 5. **`TantivyWriter`** — Single `IndexWriter` per directory (long-held during commits)
//! 6. **`AdaptiveState`** — Fusion parameter updates (rare writes)
//!
//! # Reader/Writer Isolation
//!
//! - **`FrankenSQLite`**: Page-level MVCC via `BEGIN CONCURRENT`. Readers never block
//!   writers; writers serialize at commit time only if pages conflict.
//! - **FSVI**: Append-only segments. Readers see consistent snapshots via `Arc` cloning
//!   from [`IndexCache`]. The [`RefreshWorker`] is the single writer.
//! - **Tantivy**: Built-in single-writer model. fsfs ensures exactly one `IndexWriter`
//!   per index directory via [`ResourceToken`].

#![allow(clippy::module_name_repetitions)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use tracing::{debug, warn};

// ─── Lock Ordering ──────────────────────────────────────────────────────────

/// Canonical lock levels in acquisition order. A thread must never acquire a
/// lock at a level ≤ to any lock it already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LockLevel {
    /// `FrankenSQLite` catalog (metadata, job queue, staleness).
    Catalog = 1,
    /// In-memory embedding job queue.
    EmbeddingQueue = 2,
    /// Index cache (`Arc<RwLock<Arc<TwoTierIndex>>>`).
    IndexCache = 3,
    /// Per-segment FSVI vector index file.
    FsviSegment = 4,
    /// Tantivy `IndexWriter` (single-writer, long-held).
    TantivyWriter = 5,
    /// Adaptive fusion parameter state.
    AdaptiveState = 6,
}

impl LockLevel {
    /// Human-readable name for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::EmbeddingQueue => "embedding_queue",
            Self::IndexCache => "index_cache",
            Self::FsviSegment => "fsvi_segment",
            Self::TantivyWriter => "tantivy_writer",
            Self::AdaptiveState => "adaptive_state",
        }
    }
}

impl std::fmt::Display for LockLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name(), *self as u8)
    }
}

// ─── Lock Order Guard ───────────────────────────────────────────────────────

thread_local! {
    /// Tracks the highest lock level held by the current thread.
    /// 0 means no lock is held.
    static HELD_LEVEL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// RAII guard that enforces lock ordering in debug builds. On construction it
/// verifies the requested level exceeds any currently held level; on drop it
/// restores the previous level.
///
/// In release builds the check is compiled away entirely.
pub struct LockOrderGuard {
    level: u8,
    previous: u8,
}

impl LockOrderGuard {
    /// Create a new guard for the given lock level. Panics in debug builds if
    /// the ordering invariant is violated.
    ///
    /// # Panics
    ///
    /// Panics (in debug builds) if `level` is less than or equal to any lock
    /// level already held by the current thread.
    #[must_use]
    #[inline]
    pub fn acquire(level: LockLevel) -> Self {
        let level_u8 = level as u8;
        let previous = HELD_LEVEL.with(std::cell::Cell::get);

        #[cfg(debug_assertions)]
        {
            assert!(
                level_u8 > previous,
                "Lock ordering violation: attempting to acquire {level} (level {level_u8}) \
                 while already holding a lock at level {previous}. \
                 Locks must be acquired in ascending LockLevel order.",
            );
        }

        HELD_LEVEL.with(|h| h.set(level_u8));
        Self {
            level: level_u8,
            previous,
        }
    }

    /// The lock level this guard represents.
    #[must_use]
    pub const fn level(&self) -> u8 {
        self.level
    }
}

impl Drop for LockOrderGuard {
    fn drop(&mut self) {
        HELD_LEVEL.with(|h| h.set(self.previous));
    }
}

// ─── Resource Tokens ────────────────────────────────────────────────────────

/// Identifies a specific shared resource instance for locking/reservation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceId {
    /// The `FrankenSQLite` catalog database at the given path.
    Catalog(PathBuf),
    /// A specific FSVI segment file.
    FsviSegment(PathBuf),
    /// The Tantivy index directory.
    TantivyIndex(PathBuf),
    /// The in-memory embedding queue (singleton).
    EmbeddingQueue,
    /// The index cache (singleton).
    IndexCache,
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Catalog(p) => write!(f, "catalog:{}", p.display()),
            Self::FsviSegment(p) => write!(f, "fsvi:{}", p.display()),
            Self::TantivyIndex(p) => write!(f, "tantivy:{}", p.display()),
            Self::EmbeddingQueue => write!(f, "embedding_queue"),
            Self::IndexCache => write!(f, "index_cache"),
        }
    }
}

/// Token representing exclusive access to a resource, used by the single-writer
/// guarantee system. Callers must hold the token to perform write operations.
#[derive(Debug)]
pub struct ResourceToken {
    resource: ResourceId,
    acquired_at: Instant,
    holder: String,
}

impl ResourceToken {
    /// Create a token for a resource, recording the holder identity and time.
    #[must_use]
    pub fn new(resource: ResourceId, holder: impl Into<String>) -> Self {
        Self {
            resource,
            acquired_at: Instant::now(),
            holder: holder.into(),
        }
    }

    /// Which resource this token grants access to.
    #[must_use]
    pub const fn resource(&self) -> &ResourceId {
        &self.resource
    }

    /// Who holds this token.
    #[must_use]
    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// How long this token has been held.
    #[must_use]
    pub fn held_duration(&self) -> Duration {
        self.acquired_at.elapsed()
    }
}

// ─── Contention Policy ──────────────────────────────────────────────────────

/// Configuration for backoff and contention mitigation.
#[derive(Debug, Clone)]
pub struct ContentionPolicy {
    /// Initial backoff delay for retry loops.
    pub initial_backoff: Duration,
    /// Maximum backoff delay (cap for exponential growth).
    pub max_backoff: Duration,
    /// Multiplier for exponential backoff (typically 2.0).
    pub backoff_multiplier: f64,
    /// Maximum number of retries before giving up.
    pub max_retries: u32,
    /// Queue depth at which backpressure kicks in.
    pub backpressure_threshold: usize,
    /// Maximum time to wait for a resource before timeout.
    pub acquisition_timeout: Duration,
}

impl Default for ContentionPolicy {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            max_retries: 8,
            backpressure_threshold: 10_000,
            acquisition_timeout: Duration::from_secs(30),
        }
    }
}

impl ContentionPolicy {
    /// Compute the backoff delay for the given retry attempt (0-indexed).
    #[must_use]
    pub fn backoff_delay(&self, attempt: u32) -> Duration {
        let multiplier = self.backoff_multiplier.powi(attempt.cast_signed());
        let delay = self.initial_backoff.as_secs_f64() * multiplier;
        let capped = delay.min(self.max_backoff.as_secs_f64());
        Duration::from_secs_f64(capped)
    }

    /// Whether the given queue depth exceeds the backpressure threshold.
    #[must_use]
    pub const fn is_backpressured(&self, queue_depth: usize) -> bool {
        queue_depth >= self.backpressure_threshold
    }

    /// Whether the given attempt number exceeds max retries.
    #[must_use]
    pub const fn is_exhausted(&self, attempt: u32) -> bool {
        attempt >= self.max_retries
    }
}

// ─── Contention Metrics ─────────────────────────────────────────────────────

/// Lock-free metrics tracking contention events across the pipeline.
#[derive(Debug, Default)]
pub struct ContentionMetrics {
    /// Total lock acquisition attempts.
    pub acquisitions: AtomicU64,
    /// Total times a lock acquisition had to wait (contended).
    pub contentions: AtomicU64,
    /// Total times a lock acquisition timed out.
    pub timeouts: AtomicU64,
    /// Total backoff retries.
    pub retries: AtomicU64,
    /// Total backpressure events.
    pub backpressure_events: AtomicU64,
    /// Total stale locks recovered.
    pub stale_locks_recovered: AtomicU64,
}

impl ContentionMetrics {
    /// Create a new zeroed metrics tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            acquisitions: AtomicU64::new(0),
            contentions: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
            stale_locks_recovered: AtomicU64::new(0),
        }
    }

    /// Record a successful acquisition (with optional contention flag).
    pub fn record_acquisition(&self, was_contended: bool) {
        self.acquisitions.fetch_add(1, Ordering::Relaxed);
        if was_contended {
            self.contentions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a timeout.
    pub fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a retry.
    pub fn record_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Contention rate (0.0 = no contention, 1.0 = all acquisitions contended).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn contention_rate(&self) -> f64 {
        let total = self.acquisitions.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.contentions.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Take a snapshot of all metrics.
    #[must_use]
    pub fn snapshot(&self) -> ContentionSnapshot {
        ContentionSnapshot {
            acquisitions: self.acquisitions.load(Ordering::Relaxed),
            contentions: self.contentions.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            backpressure_events: self.backpressure_events.load(Ordering::Relaxed),
            stale_locks_recovered: self.stale_locks_recovered.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of contention metrics for reporting.
#[derive(Debug, Clone, Copy)]
pub struct ContentionSnapshot {
    pub acquisitions: u64,
    pub contentions: u64,
    pub timeouts: u64,
    pub retries: u64,
    pub backpressure_events: u64,
    pub stale_locks_recovered: u64,
}

impl ContentionSnapshot {
    /// Contention rate (0.0–1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn contention_rate(&self) -> f64 {
        if self.acquisitions == 0 {
            return 0.0;
        }
        self.contentions as f64 / self.acquisitions as f64
    }
}

// ─── Stale Lock Detection ───────────────────────────────────────────────────

/// Sentinel file written to disk to claim exclusive write access to a resource.
///
/// If the process crashes, the sentinel remains. A new process can detect the
/// stale sentinel via PID liveness checking and mtime staleness.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LockSentinel {
    /// PID of the process that created this sentinel.
    pub pid: u32,
    /// Hostname of the machine.
    pub hostname: String,
    /// When the sentinel was created (Unix timestamp millis).
    pub created_at_ms: u64,
    /// Resource being locked.
    pub resource: String,
    /// Description of the holder (e.g., "`RefreshWorker`").
    pub holder: String,
}

impl LockSentinel {
    /// Create a sentinel for the current process.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn current(resource: impl Into<String>, holder: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            pid: std::process::id(),
            hostname: hostname(),
            created_at_ms: now,
            resource: resource.into(),
            holder: holder.into(),
        }
    }

    /// Whether this sentinel's PID is still alive on this host.
    #[must_use]
    pub fn is_holder_alive(&self) -> bool {
        // Only valid if same hostname.
        if self.hostname != hostname() {
            // Can't verify cross-host; assume alive to be safe.
            return true;
        }
        is_pid_alive(self.pid)
    }

    /// Whether the sentinel is older than the given threshold.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn is_stale(&self, threshold: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let age_ms = now.saturating_sub(self.created_at_ms);
        Duration::from_millis(age_ms) > threshold
    }
}

/// Write a lock sentinel file to the given path.
///
/// # Errors
///
/// Returns `Err` if the file cannot be written.
pub fn write_sentinel(path: &Path, sentinel: &LockSentinel) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(sentinel)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json.as_bytes())
}

/// Read and parse a lock sentinel file.
///
/// # Errors
///
/// Returns `Err` if the file doesn't exist, can't be read, or contains invalid JSON.
pub fn read_sentinel(path: &Path) -> std::io::Result<LockSentinel> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Remove a sentinel file.
///
/// # Errors
///
/// Returns `Err` if the file exists but cannot be removed.
pub fn remove_sentinel(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Attempt to acquire a sentinel-based lock. If a stale sentinel exists (dead
/// PID or exceeds timeout), it is cleaned up and the lock is granted.
///
/// Returns `Ok(LockSentinel)` on success, `Err` if the resource is already
/// legitimately locked by another process.
///
/// # Errors
///
/// Returns `Err` if another live process holds the sentinel or if I/O fails.
pub fn try_acquire_sentinel(
    path: &Path,
    resource: &str,
    holder: &str,
    stale_threshold: Duration,
) -> std::io::Result<LockSentinel> {
    // Check for existing sentinel.
    match read_sentinel(path) {
        Ok(existing) => {
            if !existing.is_holder_alive() {
                warn!(
                    pid = existing.pid,
                    resource = existing.resource,
                    "Recovering stale lock sentinel (holder PID is dead)"
                );
                remove_sentinel(path)?;
            } else if existing.is_stale(stale_threshold) {
                warn!(
                    pid = existing.pid,
                    age_ms = existing.created_at_ms,
                    resource = existing.resource,
                    "Recovering stale lock sentinel (exceeded timeout)"
                );
                remove_sentinel(path)?;
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    format!(
                        "Resource '{}' is locked by PID {} ({})",
                        existing.resource, existing.pid, existing.holder
                    ),
                ));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No sentinel — proceed.
        }
        Err(e) => return Err(e),
    }

    let sentinel = LockSentinel::current(resource, holder);
    write_sentinel(path, &sentinel)?;
    debug!(
        pid = sentinel.pid,
        resource, holder, "Lock sentinel acquired"
    );
    Ok(sentinel)
}

// ─── Pipeline Access Model ──────────────────────────────────────────────────

/// Documents the read/write access pattern for each pipeline stage.
/// This is informational — used by diagnostics and documentation, not enforced
/// at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Read-only access (queries, status checks).
    ReadOnly,
    /// Write access (indexing, embedding, compaction).
    ReadWrite,
    /// No access needed (stage doesn't touch this resource).
    None,
}

/// A pipeline stage and its resource access requirements.
#[derive(Debug, Clone)]
pub struct PipelineStageAccess {
    /// Name of the pipeline stage.
    pub stage: &'static str,
    /// Access to `FrankenSQLite` catalog.
    pub catalog: AccessMode,
    /// Access to embedding queue.
    pub queue: AccessMode,
    /// Access to FSVI vector indices.
    pub fsvi: AccessMode,
    /// Access to Tantivy lexical index.
    pub tantivy: AccessMode,
    /// Access to index cache.
    pub cache: AccessMode,
}

/// All pipeline stages and their access patterns.
#[must_use]
pub const fn pipeline_access_matrix() -> &'static [PipelineStageAccess] {
    &[
        PipelineStageAccess {
            stage: "crawl",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::None,
            tantivy: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "classify",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::None,
            tantivy: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "embed_fast",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            tantivy: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "embed_quality",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            tantivy: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "lexical_index",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::None,
            tantivy: AccessMode::ReadWrite,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "serve_queries",
            catalog: AccessMode::ReadOnly,
            queue: AccessMode::None,
            fsvi: AccessMode::ReadOnly,
            tantivy: AccessMode::ReadOnly,
            cache: AccessMode::ReadOnly,
        },
        PipelineStageAccess {
            stage: "refresh_worker",
            catalog: AccessMode::ReadOnly,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            tantivy: AccessMode::None,
            cache: AccessMode::ReadWrite,
        },
        PipelineStageAccess {
            stage: "compaction",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::ReadWrite,
            tantivy: AccessMode::ReadWrite,
            cache: AccessMode::ReadWrite,
        },
    ]
}

// ─── Workload Budget Scheduling ─────────────────────────────────────────────

/// Workload classes competing for shared fsfs compute budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    Ingest,
    Embed,
    Query,
}

impl WorkloadClass {
    const COUNT: usize = 3;

    #[must_use]
    const fn index(self) -> usize {
        match self {
            Self::Ingest => 0,
            Self::Embed => 1,
            Self::Query => 2,
        }
    }

    #[must_use]
    const fn all() -> [Self; Self::COUNT] {
        [Self::Ingest, Self::Embed, Self::Query]
    }
}

/// Scheduler mode toggle for balancing fairness vs query latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSchedulerMode {
    FairShare,
    LatencySensitive,
}

/// Pending demand snapshot for one scheduling cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkloadDemand {
    pub ingest: usize,
    pub embed: usize,
    pub query: usize,
}

impl WorkloadDemand {
    #[must_use]
    const fn get(self, class: WorkloadClass) -> usize {
        match class {
            WorkloadClass::Ingest => self.ingest,
            WorkloadClass::Embed => self.embed,
            WorkloadClass::Query => self.query,
        }
    }

    const fn set(&mut self, class: WorkloadClass, value: usize) {
        match class {
            WorkloadClass::Ingest => self.ingest = value,
            WorkloadClass::Embed => self.embed = value,
            WorkloadClass::Query => self.query = value,
        }
    }

    #[must_use]
    const fn total(self) -> usize {
        self.ingest + self.embed + self.query
    }

    #[must_use]
    const fn active_classes(self) -> usize {
        (self.ingest > 0) as usize + (self.embed > 0) as usize + (self.query > 0) as usize
    }
}

/// Scheduler configuration and safety bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSchedulerPolicy {
    /// Total slots available per scheduling cycle.
    pub total_slots: usize,
    /// Upper bound on admitted units per cycle.
    pub admission_limit: usize,
    /// Query floor when in latency-sensitive mode (percent of total slots).
    pub latency_query_reserve_pct: u8,
    /// Cycles a non-empty class can receive zero slots before guard activates.
    pub starvation_guard_cycles: u8,
}

impl BudgetSchedulerPolicy {
    #[must_use]
    pub const fn normalized(self) -> Self {
        let total_slots = if self.total_slots == 0 {
            1
        } else {
            self.total_slots
        };
        let admission_limit = if self.admission_limit == 0 {
            total_slots
        } else {
            self.admission_limit
        };
        let latency_query_reserve_pct = if self.latency_query_reserve_pct > 100 {
            100
        } else {
            self.latency_query_reserve_pct
        };
        let starvation_guard_cycles = if self.starvation_guard_cycles == 0 {
            1
        } else {
            self.starvation_guard_cycles
        };
        Self {
            total_slots,
            admission_limit,
            latency_query_reserve_pct,
            starvation_guard_cycles,
        }
    }
}

impl Default for BudgetSchedulerPolicy {
    fn default() -> Self {
        Self {
            total_slots: 24,
            admission_limit: 48,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        }
    }
}

/// Deterministic per-cycle allocation output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadAllocation {
    pub ingest: usize,
    pub embed: usize,
    pub query: usize,
    pub mode: BudgetSchedulerMode,
    pub admission_capped: bool,
    pub starvation_guard_applied: bool,
    pub reason_code: &'static str,
}

impl WorkloadAllocation {
    #[must_use]
    pub const fn get(self, class: WorkloadClass) -> usize {
        match class {
            WorkloadClass::Ingest => self.ingest,
            WorkloadClass::Embed => self.embed,
            WorkloadClass::Query => self.query,
        }
    }

    const fn set(&mut self, class: WorkloadClass, value: usize) {
        match class {
            WorkloadClass::Ingest => self.ingest = value,
            WorkloadClass::Embed => self.embed = value,
            WorkloadClass::Query => self.query = value,
        }
    }

    #[must_use]
    pub const fn total(self) -> usize {
        self.ingest + self.embed + self.query
    }
}

/// Two-phase permit state for cancellation-correct admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermitState {
    Reserved,
    Committed,
    Cancelled,
}

/// Reserve/commit token for one workload class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadPermit {
    class: WorkloadClass,
    slots: usize,
    state: PermitState,
}

impl WorkloadPermit {
    #[must_use]
    pub const fn class(self) -> WorkloadClass {
        self.class
    }

    #[must_use]
    pub const fn slots(self) -> usize {
        self.slots
    }

    #[must_use]
    pub const fn state(self) -> PermitState {
        self.state
    }
}

/// Stateful scheduler with fairness and starvation tracking.
#[derive(Debug, Clone)]
pub struct WorkloadBudgetScheduler {
    policy: BudgetSchedulerPolicy,
    starvation_counters: [u8; WorkloadClass::COUNT],
    rr_cursor: usize,
    reserved: [usize; WorkloadClass::COUNT],
    inflight: [usize; WorkloadClass::COUNT],
}

impl WorkloadBudgetScheduler {
    /// Create a scheduler with normalized policy.
    #[must_use]
    pub const fn new(policy: BudgetSchedulerPolicy) -> Self {
        Self {
            policy: policy.normalized(),
            starvation_counters: [0; WorkloadClass::COUNT],
            rr_cursor: 0,
            reserved: [0; WorkloadClass::COUNT],
            inflight: [0; WorkloadClass::COUNT],
        }
    }

    /// Plan one deterministic allocation cycle.
    #[must_use]
    pub fn plan_cycle(
        &mut self,
        demand: WorkloadDemand,
        mode: BudgetSchedulerMode,
    ) -> WorkloadAllocation {
        let mut capped = false;
        let mut demand = demand;
        if demand.total() > self.policy.admission_limit {
            demand = cap_demand(demand, self.policy.admission_limit, mode, self.rr_cursor);
            capped = true;
        }

        let mut allocation = match mode {
            BudgetSchedulerMode::FairShare => {
                fair_share_allocate(demand, self.policy.total_slots, self.rr_cursor)
            }
            BudgetSchedulerMode::LatencySensitive => {
                latency_sensitive_allocate(demand, self.policy.total_slots, self.policy)
            }
        };
        self.bump_cursor();

        let starvation_guard_applied = self.apply_starvation_guard(demand, &mut allocation);
        self.update_starvation_counters(demand, allocation);

        WorkloadAllocation {
            ingest: allocation.ingest,
            embed: allocation.embed,
            query: allocation.query,
            mode,
            admission_capped: capped,
            starvation_guard_applied,
            reason_code: allocation_reason(mode, capped, starvation_guard_applied),
        }
    }

    /// Reserve slots in two phases. Call commit/cancel explicitly.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when reservation cannot be admitted.
    pub fn reserve(
        &mut self,
        class: WorkloadClass,
        slots: usize,
    ) -> Result<WorkloadPermit, &'static str> {
        if slots == 0 {
            return Err("scheduler.reserve.zero_slots");
        }
        if self.total_reserved().saturating_add(slots) > self.policy.admission_limit {
            return Err("scheduler.reserve.admission_limited");
        }
        if self.total_inflight_reserved().saturating_add(slots) > self.policy.total_slots {
            return Err("scheduler.reserve.capacity_exhausted");
        }

        self.reserved[class.index()] = self.reserved[class.index()].saturating_add(slots);
        Ok(WorkloadPermit {
            class,
            slots,
            state: PermitState::Reserved,
        })
    }

    /// Commit a previously reserved permit.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when the permit is not in reserved state.
    pub const fn commit_permit(&mut self, permit: &mut WorkloadPermit) -> Result<(), &'static str> {
        if !matches!(permit.state, PermitState::Reserved) {
            return Err("scheduler.commit.invalid_state");
        }
        let idx = permit.class.index();
        self.reserved[idx] = self.reserved[idx].saturating_sub(permit.slots);
        self.inflight[idx] = self.inflight[idx].saturating_add(permit.slots);
        permit.state = PermitState::Committed;
        Ok(())
    }

    /// Cancel a previously reserved permit and release capacity.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when the permit is not in reserved state.
    pub const fn cancel_permit(&mut self, permit: &mut WorkloadPermit) -> Result<(), &'static str> {
        if !matches!(permit.state, PermitState::Reserved) {
            return Err("scheduler.cancel.invalid_state");
        }
        let idx = permit.class.index();
        self.reserved[idx] = self.reserved[idx].saturating_sub(permit.slots);
        permit.state = PermitState::Cancelled;
        Ok(())
    }

    /// Mark committed work complete and release inflight capacity.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when completion exceeds inflight slots.
    pub const fn complete(
        &mut self,
        class: WorkloadClass,
        slots: usize,
    ) -> Result<(), &'static str> {
        let idx = class.index();
        if slots > self.inflight[idx] {
            return Err("scheduler.complete.underflow");
        }
        self.inflight[idx] -= slots;
        Ok(())
    }

    #[must_use]
    pub const fn policy(&self) -> BudgetSchedulerPolicy {
        self.policy
    }

    #[must_use]
    pub const fn reserved_for(&self, class: WorkloadClass) -> usize {
        self.reserved[class.index()]
    }

    #[must_use]
    pub const fn inflight_for(&self, class: WorkloadClass) -> usize {
        self.inflight[class.index()]
    }

    const fn bump_cursor(&mut self) {
        self.rr_cursor = (self.rr_cursor + 1) % WorkloadClass::COUNT;
    }

    fn total_reserved(&self) -> usize {
        self.reserved.iter().sum()
    }

    fn total_inflight_reserved(&self) -> usize {
        self.inflight.iter().sum::<usize>() + self.total_reserved()
    }

    fn apply_starvation_guard(
        &self,
        demand: WorkloadDemand,
        allocation: &mut WorkloadAllocation,
    ) -> bool {
        let mut applied = false;
        for class in WorkloadClass::all() {
            let idx = class.index();
            if demand.get(class) == 0
                || allocation.get(class) > 0
                || self.starvation_counters[idx] < self.policy.starvation_guard_cycles
            {
                continue;
            }

            if let Some(donor) = donor_class(*allocation, class) {
                allocation.set(class, allocation.get(class) + 1);
                allocation.set(donor, allocation.get(donor).saturating_sub(1));
                applied = true;
            }
        }
        applied
    }

    fn update_starvation_counters(
        &mut self,
        demand: WorkloadDemand,
        allocation: WorkloadAllocation,
    ) {
        for class in WorkloadClass::all() {
            let idx = class.index();
            if demand.get(class) == 0 {
                self.starvation_counters[idx] = 0;
            } else if allocation.get(class) == 0 {
                self.starvation_counters[idx] = self.starvation_counters[idx].saturating_add(1);
            } else {
                self.starvation_counters[idx] = 0;
            }
        }
    }
}

const fn allocation_reason(
    mode: BudgetSchedulerMode,
    admission_capped: bool,
    starvation_guard_applied: bool,
) -> &'static str {
    match (mode, admission_capped, starvation_guard_applied) {
        (BudgetSchedulerMode::FairShare, false, false) => "scheduler.plan.fair_share",
        (BudgetSchedulerMode::FairShare, true, false) => "scheduler.plan.fair_share.capped",
        (BudgetSchedulerMode::FairShare, false, true) => {
            "scheduler.plan.fair_share.starvation_guard"
        }
        (BudgetSchedulerMode::FairShare, true, true) => {
            "scheduler.plan.fair_share.capped_starvation_guard"
        }
        (BudgetSchedulerMode::LatencySensitive, false, false) => "scheduler.plan.latency_sensitive",
        (BudgetSchedulerMode::LatencySensitive, true, false) => {
            "scheduler.plan.latency_sensitive.capped"
        }
        (BudgetSchedulerMode::LatencySensitive, false, true) => {
            "scheduler.plan.latency_sensitive.starvation_guard"
        }
        (BudgetSchedulerMode::LatencySensitive, true, true) => {
            "scheduler.plan.latency_sensitive.capped_starvation_guard"
        }
    }
}

fn fair_share_allocate(
    demand: WorkloadDemand,
    total_slots: usize,
    rr_cursor: usize,
) -> WorkloadAllocation {
    let active = demand.active_classes();
    if active == 0 || total_slots == 0 {
        return WorkloadAllocation {
            ingest: 0,
            embed: 0,
            query: 0,
            mode: BudgetSchedulerMode::FairShare,
            admission_capped: false,
            starvation_guard_applied: false,
            reason_code: "scheduler.plan.fair_share",
        };
    }

    let mut allocation = WorkloadAllocation {
        ingest: 0,
        embed: 0,
        query: 0,
        mode: BudgetSchedulerMode::FairShare,
        admission_capped: false,
        starvation_guard_applied: false,
        reason_code: "scheduler.plan.fair_share",
    };
    let base = total_slots / active;
    let mut remainder = total_slots % active;

    let order = class_order(rr_cursor);
    for class in order {
        if demand.get(class) == 0 {
            continue;
        }
        let mut slots = base.min(demand.get(class));
        if remainder > 0 && slots < demand.get(class) {
            slots += 1;
            remainder -= 1;
        }
        allocation.set(class, slots);
    }

    while allocation.total() < total_slots {
        let mut progressed = false;
        for class in class_order(rr_cursor) {
            if allocation.get(class) >= demand.get(class) {
                continue;
            }
            allocation.set(class, allocation.get(class) + 1);
            progressed = true;
            if allocation.total() == total_slots {
                break;
            }
        }
        if !progressed {
            break;
        }
    }

    allocation
}

fn latency_sensitive_allocate(
    demand: WorkloadDemand,
    total_slots: usize,
    policy: BudgetSchedulerPolicy,
) -> WorkloadAllocation {
    if total_slots == 0 || demand.total() == 0 {
        return WorkloadAllocation {
            ingest: 0,
            embed: 0,
            query: 0,
            mode: BudgetSchedulerMode::LatencySensitive,
            admission_capped: false,
            starvation_guard_applied: false,
            reason_code: "scheduler.plan.latency_sensitive",
        };
    }

    let mut allocation = WorkloadAllocation {
        ingest: 0,
        embed: 0,
        query: 0,
        mode: BudgetSchedulerMode::LatencySensitive,
        admission_capped: false,
        starvation_guard_applied: false,
        reason_code: "scheduler.plan.latency_sensitive",
    };

    let mut remaining = total_slots;
    let query_floor =
        (total_slots.saturating_mul(usize::from(policy.latency_query_reserve_pct))) / 100;
    let query_reserved = demand
        .query
        .min(query_floor.max(usize::from(demand.query > 0)));
    allocation.query = query_reserved;
    remaining = remaining.saturating_sub(query_reserved);

    let non_query_demand = WorkloadDemand {
        ingest: demand.ingest,
        embed: demand.embed,
        query: 0,
    };
    let non_query_allocation = fair_share_allocate(non_query_demand, remaining, 0);
    allocation.ingest = non_query_allocation.ingest;
    allocation.embed = non_query_allocation.embed;
    remaining = remaining.saturating_sub(non_query_allocation.total());

    if remaining > 0 {
        let extra_query = demand.query.saturating_sub(allocation.query).min(remaining);
        allocation.query += extra_query;
        remaining -= extra_query;
    }

    if remaining > 0 {
        for class in [
            WorkloadClass::Ingest,
            WorkloadClass::Embed,
            WorkloadClass::Query,
        ] {
            if remaining == 0 {
                break;
            }
            let headroom = demand.get(class).saturating_sub(allocation.get(class));
            if headroom == 0 {
                continue;
            }
            let take = headroom.min(remaining);
            allocation.set(class, allocation.get(class) + take);
            remaining -= take;
        }
    }

    allocation
}

fn cap_demand(
    mut demand: WorkloadDemand,
    limit: usize,
    mode: BudgetSchedulerMode,
    rr_cursor: usize,
) -> WorkloadDemand {
    while demand.total() > limit {
        let class = match mode {
            BudgetSchedulerMode::LatencySensitive => {
                if demand.ingest > 0 {
                    WorkloadClass::Ingest
                } else if demand.embed > 0 {
                    WorkloadClass::Embed
                } else {
                    WorkloadClass::Query
                }
            }
            BudgetSchedulerMode::FairShare => highest_demand_class(demand, rr_cursor),
        };
        demand.set(class, demand.get(class).saturating_sub(1));
    }
    demand
}

fn highest_demand_class(demand: WorkloadDemand, rr_cursor: usize) -> WorkloadClass {
    let mut best = WorkloadClass::Ingest;
    let mut best_value = 0usize;
    for class in class_order(rr_cursor) {
        let value = demand.get(class);
        if value > best_value {
            best = class;
            best_value = value;
        }
    }
    best
}

fn donor_class(allocation: WorkloadAllocation, starving: WorkloadClass) -> Option<WorkloadClass> {
    let mut donor = None;
    let mut donor_slots = 0usize;
    for class in WorkloadClass::all() {
        if class == starving {
            continue;
        }
        let slots = allocation.get(class);
        if slots > donor_slots {
            donor = Some(class);
            donor_slots = slots;
        }
    }
    donor.filter(|class| allocation.get(*class) > 0)
}

const fn class_order(rr_cursor: usize) -> [WorkloadClass; WorkloadClass::COUNT] {
    let classes = WorkloadClass::all();
    [
        classes[rr_cursor % WorkloadClass::COUNT],
        classes[(rr_cursor + 1) % WorkloadClass::COUNT],
        classes[(rr_cursor + 2) % WorkloadClass::COUNT],
    ]
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Get the system hostname, or "unknown" if it can't be determined.
fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".into())
}

/// Check whether a PID is alive on the local system.
fn is_pid_alive(pid: u32) -> bool {
    // On Linux, we can check /proc/<pid>/status.
    // signal(0) is the canonical check but requires unsafe.
    // We use /proc existence as a safe alternative.
    Path::new(&format!("/proc/{pid}")).exists()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    // ── Lock Ordering ──

    #[test]
    fn lock_levels_are_ordered() {
        assert!(LockLevel::Catalog < LockLevel::EmbeddingQueue);
        assert!(LockLevel::EmbeddingQueue < LockLevel::IndexCache);
        assert!(LockLevel::IndexCache < LockLevel::FsviSegment);
        assert!(LockLevel::FsviSegment < LockLevel::TantivyWriter);
        assert!(LockLevel::TantivyWriter < LockLevel::AdaptiveState);
    }

    #[test]
    fn lock_order_guard_acquires_ascending() {
        let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
        let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
        let _g3 = LockOrderGuard::acquire(LockLevel::IndexCache);
        // All ascending — should not panic.
    }

    #[test]
    fn lock_order_guard_restores_on_drop() {
        {
            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
        }
        // After drop, level should be reset.
        let _g3 = LockOrderGuard::acquire(LockLevel::Catalog);
        // Should not panic — level was restored.
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Lock ordering violation")]
    fn lock_order_violation_panics_in_debug() {
        let _g1 = LockOrderGuard::acquire(LockLevel::IndexCache);
        let _g2 = LockOrderGuard::acquire(LockLevel::Catalog); // Lower level — violation!
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Lock ordering violation")]
    fn same_level_acquisition_panics() {
        let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
        let _g2 = LockOrderGuard::acquire(LockLevel::Catalog); // Same level — violation!
    }

    #[test]
    fn lock_level_display() {
        assert_eq!(format!("{}", LockLevel::Catalog), "catalog(1)");
        assert_eq!(format!("{}", LockLevel::TantivyWriter), "tantivy_writer(5)");
    }

    #[test]
    fn lock_level_name() {
        assert_eq!(LockLevel::Catalog.name(), "catalog");
        assert_eq!(LockLevel::AdaptiveState.name(), "adaptive_state");
    }

    // ── Resource Tokens ──

    #[test]
    fn resource_token_tracks_holder() {
        let token = ResourceToken::new(ResourceId::EmbeddingQueue, "refresh_worker");
        assert_eq!(token.holder(), "refresh_worker");
        assert!(matches!(token.resource(), ResourceId::EmbeddingQueue));
        assert!(token.held_duration() < Duration::from_secs(1));
    }

    #[test]
    fn resource_id_display() {
        let id = ResourceId::Catalog(PathBuf::from("/data/db.sqlite"));
        assert_eq!(format!("{id}"), "catalog:/data/db.sqlite");

        let id = ResourceId::EmbeddingQueue;
        assert_eq!(format!("{id}"), "embedding_queue");
    }

    // ── Contention Policy ──

    #[test]
    fn backoff_delay_exponential() {
        let policy = ContentionPolicy::default();

        let d0 = policy.backoff_delay(0);
        let d1 = policy.backoff_delay(1);
        let d2 = policy.backoff_delay(2);

        assert_eq!(d0, Duration::from_millis(10));
        assert_eq!(d1, Duration::from_millis(20));
        assert_eq!(d2, Duration::from_millis(40));
    }

    #[test]
    fn backoff_delay_caps_at_max() {
        let policy = ContentionPolicy {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            ..ContentionPolicy::default()
        };

        // 100 * 2^3 = 800, capped at 500.
        let d3 = policy.backoff_delay(3);
        assert_eq!(d3, Duration::from_millis(500));
    }

    #[test]
    fn backpressure_detection() {
        let policy = ContentionPolicy {
            backpressure_threshold: 100,
            ..ContentionPolicy::default()
        };

        assert!(!policy.is_backpressured(99));
        assert!(policy.is_backpressured(100));
        assert!(policy.is_backpressured(101));
    }

    #[test]
    fn retry_exhaustion() {
        let policy = ContentionPolicy {
            max_retries: 3,
            ..ContentionPolicy::default()
        };

        assert!(!policy.is_exhausted(0));
        assert!(!policy.is_exhausted(2));
        assert!(policy.is_exhausted(3));
        assert!(policy.is_exhausted(4));
    }

    // ── Contention Metrics ──

    #[test]
    #[allow(clippy::float_cmp)]
    fn contention_metrics_tracking() {
        let metrics = ContentionMetrics::new();
        assert_eq!(metrics.contention_rate(), 0.0);

        metrics.record_acquisition(false);
        metrics.record_acquisition(false);
        metrics.record_acquisition(true);
        metrics.record_acquisition(true);

        let rate = metrics.contention_rate();
        assert!(
            (rate - 0.5).abs() < f64::EPSILON,
            "expected 0.5, got {rate}"
        );
    }

    #[test]
    fn contention_snapshot() {
        let metrics = ContentionMetrics::new();
        metrics.record_acquisition(true);
        metrics.record_timeout();
        metrics.record_retry();

        let snap = metrics.snapshot();
        assert_eq!(snap.acquisitions, 1);
        assert_eq!(snap.contentions, 1);
        assert_eq!(snap.timeouts, 1);
        assert_eq!(snap.retries, 1);
        assert_eq!(snap.backpressure_events, 0);
        assert!((snap.contention_rate() - 1.0).abs() < f64::EPSILON);
    }

    // ── Lock Sentinel ──

    #[test]
    fn sentinel_current_captures_pid() {
        let sentinel = LockSentinel::current("test_resource", "test_holder");
        assert_eq!(sentinel.pid, std::process::id());
        assert_eq!(sentinel.resource, "test_resource");
        assert_eq!(sentinel.holder, "test_holder");
        assert!(sentinel.created_at_ms > 0);
    }

    #[test]
    fn sentinel_is_alive_for_current_process() {
        let sentinel = LockSentinel::current("test", "test");
        assert!(sentinel.is_holder_alive());
    }

    #[test]
    fn sentinel_is_stale_checks_age() {
        let mut sentinel = LockSentinel::current("test", "test");
        assert!(!sentinel.is_stale(Duration::from_mins(1)));

        // Make it old.
        sentinel.created_at_ms = sentinel.created_at_ms.saturating_sub(120_000);
        assert!(sentinel.is_stale(Duration::from_mins(1)));
    }

    #[test]
    fn sentinel_roundtrip_via_file() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.lock");

        let original = LockSentinel::current("my_resource", "worker_1");
        write_sentinel(&path, &original).expect("write");

        let loaded = read_sentinel(&path).expect("read");
        assert_eq!(loaded.pid, original.pid);
        assert_eq!(loaded.resource, "my_resource");
        assert_eq!(loaded.holder, "worker_1");
    }

    #[test]
    fn sentinel_remove_nonexistent_is_ok() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("nonexistent.lock");
        assert!(remove_sentinel(&path).is_ok());
    }

    #[test]
    fn try_acquire_sentinel_creates_new() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.lock");

        let sentinel = try_acquire_sentinel(&path, "resource", "holder", Duration::from_mins(5))
            .expect("acquire");

        assert_eq!(sentinel.pid, std::process::id());
        assert!(path.exists());

        // Cleanup.
        remove_sentinel(&path).expect("cleanup");
    }

    #[test]
    fn try_acquire_sentinel_blocks_when_held() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.lock");

        // First acquisition succeeds.
        let _s1 = try_acquire_sentinel(&path, "resource", "holder1", Duration::from_mins(5))
            .expect("first acquire");

        // Second acquisition fails (same PID, but sentinel exists).
        let result = try_acquire_sentinel(&path, "resource", "holder2", Duration::from_mins(5));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);

        // Cleanup.
        remove_sentinel(&path).expect("cleanup");
    }

    #[test]
    fn try_acquire_sentinel_recovers_stale_by_timeout() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.lock");

        // Write an old sentinel from current PID but with very old timestamp.
        let mut old = LockSentinel::current("resource", "old_holder");
        old.created_at_ms = 1_000; // Very old.
        write_sentinel(&path, &old).expect("write old");

        // Acquire with short stale threshold — should recover.
        let sentinel =
            try_acquire_sentinel(&path, "resource", "new_holder", Duration::from_secs(1))
                .expect("recover stale");
        assert_eq!(sentinel.holder, "new_holder");

        remove_sentinel(&path).expect("cleanup");
    }

    #[test]
    fn try_acquire_sentinel_recovers_dead_pid() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.lock");

        // Write a sentinel with a PID that definitely doesn't exist.
        let mut dead = LockSentinel::current("resource", "dead_holder");
        dead.pid = 999_999_999; // Very unlikely to exist.
        write_sentinel(&path, &dead).expect("write dead");

        // Should recover because PID is dead.
        let sentinel =
            try_acquire_sentinel(&path, "resource", "new_holder", Duration::from_mins(5))
                .expect("recover dead");
        assert_eq!(sentinel.holder, "new_holder");

        remove_sentinel(&path).expect("cleanup");
    }

    // ── Pipeline Access Matrix ──

    #[test]
    fn pipeline_access_matrix_has_all_stages() {
        let matrix = pipeline_access_matrix();
        let stages: Vec<&str> = matrix.iter().map(|s| s.stage).collect();
        assert!(stages.contains(&"crawl"));
        assert!(stages.contains(&"serve_queries"));
        assert!(stages.contains(&"refresh_worker"));
        assert!(stages.contains(&"compaction"));
    }

    #[test]
    fn serve_queries_is_read_only() {
        let matrix = pipeline_access_matrix();
        let query_stage = matrix.iter().find(|s| s.stage == "serve_queries").unwrap();
        assert_eq!(query_stage.catalog, AccessMode::ReadOnly);
        assert_eq!(query_stage.fsvi, AccessMode::ReadOnly);
        assert_eq!(query_stage.tantivy, AccessMode::ReadOnly);
        assert_eq!(query_stage.cache, AccessMode::ReadOnly);
        assert_eq!(query_stage.queue, AccessMode::None);
    }

    #[test]
    fn crawl_writes_catalog_and_queue() {
        let matrix = pipeline_access_matrix();
        let crawl = matrix.iter().find(|s| s.stage == "crawl").unwrap();
        assert_eq!(crawl.catalog, AccessMode::ReadWrite);
        assert_eq!(crawl.queue, AccessMode::ReadWrite);
        assert_eq!(crawl.fsvi, AccessMode::None);
    }

    // ── Workload Budget Scheduler ──

    #[test]
    fn fair_share_scheduler_balances_active_classes() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 9,
            admission_limit: 64,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        });
        let allocation = scheduler.plan_cycle(
            WorkloadDemand {
                ingest: 20,
                embed: 20,
                query: 20,
            },
            BudgetSchedulerMode::FairShare,
        );

        assert_eq!(allocation.total(), 9);
        assert_eq!(allocation.ingest, 3);
        assert_eq!(allocation.embed, 3);
        assert_eq!(allocation.query, 3);
        assert_eq!(allocation.reason_code, "scheduler.plan.fair_share");
    }

    #[test]
    fn latency_sensitive_scheduler_prioritizes_query_budget() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 10,
            admission_limit: 32,
            latency_query_reserve_pct: 60,
            starvation_guard_cycles: 2,
        });
        let allocation = scheduler.plan_cycle(
            WorkloadDemand {
                ingest: 10,
                embed: 10,
                query: 10,
            },
            BudgetSchedulerMode::LatencySensitive,
        );

        assert_eq!(allocation.total(), 10);
        assert!(allocation.query >= 6);
        assert!(allocation.query >= allocation.ingest);
        assert!(allocation.query >= allocation.embed);
        assert_eq!(allocation.reason_code, "scheduler.plan.latency_sensitive");
    }

    #[test]
    fn starvation_guard_forces_minimum_progress() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 1,
            admission_limit: 16,
            latency_query_reserve_pct: 100,
            starvation_guard_cycles: 1,
        });

        let first = scheduler.plan_cycle(
            WorkloadDemand {
                ingest: 5,
                embed: 0,
                query: 5,
            },
            BudgetSchedulerMode::LatencySensitive,
        );
        assert_eq!(first.query, 1);
        assert_eq!(first.ingest, 0);

        let second = scheduler.plan_cycle(
            WorkloadDemand {
                ingest: 5,
                embed: 0,
                query: 5,
            },
            BudgetSchedulerMode::LatencySensitive,
        );
        assert_eq!(second.ingest, 1);
        assert!(second.starvation_guard_applied);
        assert_eq!(
            second.reason_code,
            "scheduler.plan.latency_sensitive.starvation_guard"
        );
    }

    #[test]
    fn admission_cap_is_reflected_in_allocation() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 8,
            admission_limit: 5,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        });
        let allocation = scheduler.plan_cycle(
            WorkloadDemand {
                ingest: 10,
                embed: 10,
                query: 10,
            },
            BudgetSchedulerMode::FairShare,
        );

        assert_eq!(allocation.total(), 5);
        assert!(allocation.admission_capped);
        assert_eq!(allocation.reason_code, "scheduler.plan.fair_share.capped");
    }

    #[test]
    fn permit_cancel_releases_reserved_capacity() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 4,
            admission_limit: 4,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        });
        let mut permit = scheduler
            .reserve(WorkloadClass::Ingest, 3)
            .expect("reservation should succeed");
        assert_eq!(permit.state(), PermitState::Reserved);
        assert_eq!(scheduler.reserved_for(WorkloadClass::Ingest), 3);

        scheduler
            .cancel_permit(&mut permit)
            .expect("cancel should release reserved slots");
        assert_eq!(permit.state(), PermitState::Cancelled);
        assert_eq!(scheduler.reserved_for(WorkloadClass::Ingest), 0);
    }

    #[test]
    fn permit_commit_and_complete_preserve_invariants() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 6,
            admission_limit: 6,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        });
        let mut permit = scheduler
            .reserve(WorkloadClass::Query, 2)
            .expect("query reservation should succeed");
        scheduler
            .commit_permit(&mut permit)
            .expect("commit should move reservation to inflight");

        assert_eq!(permit.state(), PermitState::Committed);
        assert_eq!(scheduler.reserved_for(WorkloadClass::Query), 0);
        assert_eq!(scheduler.inflight_for(WorkloadClass::Query), 2);

        scheduler
            .complete(WorkloadClass::Query, 2)
            .expect("completion should release inflight slots");
        assert_eq!(scheduler.inflight_for(WorkloadClass::Query), 0);
    }

    #[test]
    fn reserve_enforces_bounded_admission() {
        let mut scheduler = WorkloadBudgetScheduler::new(BudgetSchedulerPolicy {
            total_slots: 8,
            admission_limit: 3,
            latency_query_reserve_pct: 50,
            starvation_guard_cycles: 2,
        });
        let result = scheduler.reserve(WorkloadClass::Embed, 4);
        assert_eq!(result, Err("scheduler.reserve.admission_limited"));
    }

    // ── Helpers ──

    #[test]
    fn current_pid_is_alive() {
        assert!(is_pid_alive(std::process::id()));
    }

    #[test]
    fn nonexistent_pid_is_not_alive() {
        assert!(!is_pid_alive(999_999_999));
    }
}
