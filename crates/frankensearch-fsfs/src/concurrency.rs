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
