//! Concurrency model, lock ordering, and contention policy for the fsfs indexing pipeline.
//!
//! This module defines the canonical concurrency strategy for all shared resources
//! in the fsfs pipeline: `FrankenSQLite` catalog, FSVI vector indices, lexical
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
//! 5. **`LexicalWriter`** — Single lexical writer per directory (long-held during commits)
//! 6. **`AdaptiveState`** — Fusion parameter updates (rare writes)
//!
//! # Reader/Writer Isolation
//!
//! - **`FrankenSQLite`**: Page-level MVCC via `BEGIN CONCURRENT`. Readers never block
//!   writers; writers serialize at commit time only if pages conflict.
//! - **FSVI**: Append-only segments. Readers see consistent snapshots via `Arc` cloning
//!   from `IndexCache`. The `RefreshWorker` is the single writer.
//! - **Lexical index**: fsfs ensures exactly one writer
//!   per index directory via [`ResourceToken`].

#![allow(clippy::module_name_repetitions)]

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use sha2::{Digest, Sha256};
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
    /// Lexical index writer (single-writer, long-held).
    LexicalWriter = 5,
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
            Self::LexicalWriter => "lexical_writer",
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
    /// The lexical index directory.
    LexicalIndex(PathBuf),
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
            Self::LexicalIndex(p) => write!(f, "lexical:{}", p.display()),
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
        #[allow(clippy::cast_possible_wrap)]
        let multiplier = self.backoff_multiplier.powi(attempt as i32);
        let delay = self.initial_backoff.as_secs_f64() * multiplier;
        let capped = delay.min(self.max_backoff.as_secs_f64());
        // Guard against negative or NaN from misconfigured multiplier —
        // Duration::from_secs_f64 panics on negative/NaN/infinite values.
        if !capped.is_finite() || capped < 0.0 {
            return self.initial_backoff;
        }
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

// ─── Index-root writer lease ───────────────────────────────────────────────

/// Persistent lock inode used to serialize every fsfs mutation rooted at an
/// index directory.
///
/// The inode is deliberately never unlinked during normal release. Kernel
/// `flock` ownership, rather than wall-clock age or a PID-file deletion race,
/// is the sole authority for writer admission.
pub const INDEX_ROOT_WRITER_LOCK_FILE: &str = ".fsfs-writer.lock";

const INDEX_ROOT_WRITER_RECORD_MAGIC: [u8; 8] = *b"FSFSWLK1";
const INDEX_ROOT_WRITER_RECORD_VERSION: u32 = 1;
const INDEX_ROOT_WRITER_RECORD_PREFIX_BYTES: usize = 96;
const INDEX_ROOT_WRITER_RECORD_BYTES: usize = 128;

/// Typed, fail-closed failure from index-root writer admission or fencing.
#[derive(Debug)]
pub enum IndexRootWriterLeaseError {
    /// Another cooperating process currently owns the kernel flock.
    Busy {
        /// Persistent lock pathname.
        path: PathBuf,
        /// Best-effort owner PID from the sealed record. The flock remains the
        /// authority even when the record is absent or being written.
        owner_pid: Option<u32>,
    },
    /// The root or lock inode failed an identity, ownership, or record check.
    Invalid {
        /// Path whose capability could not be proved.
        path: PathBuf,
        /// Stable operator-facing explanation.
        detail: String,
    },
    /// A filesystem operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Path involved in the failure.
        path: PathBuf,
        /// Underlying I/O failure.
        source: std::io::Error,
    },
    /// This target cannot provide the required no-follow flock contract.
    Unsupported {
        /// Requested index root.
        path: PathBuf,
    },
}

impl std::fmt::Display for IndexRootWriterLeaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy { path, owner_pid } => {
                write!(formatter, "index root {} is busy", path.display())?;
                if let Some(pid) = owner_pid {
                    write!(formatter, " (writer pid {pid})")?;
                }
                Ok(())
            }
            Self::Invalid { path, detail } => {
                write!(
                    formatter,
                    "index-root writer lease rejected {}: {detail}",
                    path.display()
                )
            }
            Self::Io {
                operation,
                path,
                source,
            } => write!(formatter, "{operation} {}: {source}", path.display()),
            Self::Unsupported { path } => write!(
                formatter,
                "index-root writer lease at {} requires no-follow Unix flock semantics",
                path.display()
            ),
        }
    }
}

impl std::error::Error for IndexRootWriterLeaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Busy { .. } | Self::Invalid { .. } | Self::Unsupported { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexRootWriterOwnerRecord {
    pid: u32,
    owner_uid: u32,
    process_start: u64,
    host_id: u64,
    boot_id: u64,
    nonce_high: u64,
    nonce_low: u64,
    root_device: u64,
    root_inode: u64,
    lock_device: u64,
    lock_inode: u64,
}

impl IndexRootWriterOwnerRecord {
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    fn current(
        index_root: &Path,
        owner_uid: u32,
        root_device: u64,
        root_inode: u64,
        lock_device: u64,
        lock_inode: u64,
    ) -> Result<Self, IndexRootWriterLeaseError> {
        let pid = std::process::id();
        let system_pid = sysinfo::Pid::from_u32(pid);
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[system_pid]), true);
        let process_start = system
            .process(system_pid)
            .map(sysinfo::Process::start_time)
            .filter(|value| *value != 0)
            .ok_or_else(|| IndexRootWriterLeaseError::Invalid {
                path: index_root.to_path_buf(),
                detail: "current process start identity is unavailable".to_owned(),
            })?;
        let boot_id = sysinfo::System::boot_time();
        if boot_id == 0 {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: index_root.to_path_buf(),
                detail: "current boot identity is unavailable".to_owned(),
            });
        }
        let host_id = stable_identity_hash(hostname().as_bytes());
        let nonce_input = (
            pid,
            owner_uid,
            process_start,
            boot_id,
            root_device,
            root_inode,
            lock_device,
            lock_inode,
            index_root,
        );
        let nonce_high = ahash::RandomState::new().hash_one(nonce_input);
        let nonce_low = ahash::RandomState::new().hash_one(nonce_input);
        if nonce_high == 0 && nonce_low == 0 {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: index_root.to_path_buf(),
                detail: "random writer nonce generation returned the reserved zero identity"
                    .to_owned(),
            });
        }

        Ok(Self {
            pid,
            owner_uid,
            process_start,
            host_id,
            boot_id,
            nonce_high,
            nonce_low,
            root_device,
            root_inode,
            lock_device,
            lock_inode,
        })
    }

    fn to_bytes(self) -> [u8; INDEX_ROOT_WRITER_RECORD_BYTES] {
        let mut bytes = [0_u8; INDEX_ROOT_WRITER_RECORD_BYTES];
        bytes[..8].copy_from_slice(&INDEX_ROOT_WRITER_RECORD_MAGIC);
        put_u32(&mut bytes, 8, INDEX_ROOT_WRITER_RECORD_VERSION);
        put_u32(&mut bytes, 12, self.pid);
        put_u32(&mut bytes, 16, self.owner_uid);
        put_u32(&mut bytes, 20, 0);
        for (offset, value) in [
            (24, self.process_start),
            (32, self.host_id),
            (40, self.boot_id),
            (48, self.nonce_high),
            (56, self.nonce_low),
            (64, self.root_device),
            (72, self.root_inode),
            (80, self.lock_device),
            (88, self.lock_inode),
        ] {
            put_u64(&mut bytes, offset, value);
        }
        let checksum = Sha256::digest(&bytes[..INDEX_ROOT_WRITER_RECORD_PREFIX_BYTES]);
        bytes[INDEX_ROOT_WRITER_RECORD_PREFIX_BYTES..].copy_from_slice(&checksum);
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != INDEX_ROOT_WRITER_RECORD_BYTES {
            return Err(format!(
                "v1 writer record must be exactly {INDEX_ROOT_WRITER_RECORD_BYTES} bytes, found {}",
                bytes.len()
            ));
        }
        if bytes[..8] != INDEX_ROOT_WRITER_RECORD_MAGIC {
            return Err("invalid index-root writer record magic".to_owned());
        }
        let version = get_u32(bytes, 8);
        if version != INDEX_ROOT_WRITER_RECORD_VERSION {
            return Err(format!(
                "unsupported index-root writer record version {version}"
            ));
        }
        if get_u32(bytes, 20) != 0 {
            return Err("index-root writer record reserved bits are nonzero".to_owned());
        }
        let expected = Sha256::digest(&bytes[..INDEX_ROOT_WRITER_RECORD_PREFIX_BYTES]);
        if bytes[INDEX_ROOT_WRITER_RECORD_PREFIX_BYTES..] != expected[..] {
            return Err("index-root writer record checksum mismatch".to_owned());
        }

        let record = Self {
            pid: get_u32(bytes, 12),
            owner_uid: get_u32(bytes, 16),
            process_start: get_u64(bytes, 24),
            host_id: get_u64(bytes, 32),
            boot_id: get_u64(bytes, 40),
            nonce_high: get_u64(bytes, 48),
            nonce_low: get_u64(bytes, 56),
            root_device: get_u64(bytes, 64),
            root_inode: get_u64(bytes, 72),
            lock_device: get_u64(bytes, 80),
            lock_inode: get_u64(bytes, 88),
        };
        if record.pid == 0 {
            return Err("index-root writer record has pid zero".to_owned());
        }
        if record.process_start == 0 || record.boot_id == 0 {
            return Err("index-root writer record lacks process/boot identity".to_owned());
        }
        if record.nonce_high == 0 && record.nonce_low == 0 {
            return Err("index-root writer record has the reserved zero nonce".to_owned());
        }
        Ok(record)
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed writer-record u32 range"),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed writer-record u64 range"),
    )
}

fn stable_identity_hash(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is exactly eight bytes"),
    )
}

/// An owned, kernel-backed capability for mutating one fsfs index root.
///
/// Call [`Self::fence`] immediately before each shared mutation boundary. The
/// check binds the still-open root and lock descriptors, their pathname
/// identities and owners, and the sealed owner record written after acquiring
/// the flock.
#[derive(Debug)]
pub struct IndexRootWriterLease {
    index_root: PathBuf,
    lock_path: PathBuf,
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    root_file: File,
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    lock_file: File,
    owner_record: IndexRootWriterOwnerRecord,
}

impl IndexRootWriterLease {
    /// Acquire the nonblocking exclusive writer flock for `index_root`.
    ///
    /// The root directory is the only artifact this function may create before
    /// writer admission; it is needed to host the persistent lock inode.
    ///
    /// # Errors
    ///
    /// Returns [`IndexRootWriterLeaseError::Busy`] if another process owns the
    /// flock, or a fail-closed identity/I/O error if the root, lock inode, or
    /// sealed owner record cannot be proved.
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    pub fn acquire(index_root: &Path) -> Result<Self, IndexRootWriterLeaseError> {
        use rustix::fs::{FlockOperation, Mode, OFlags, flock, open, openat};
        use std::os::unix::fs::MetadataExt;

        std::fs::create_dir_all(index_root).map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "create index root for writer lease",
            path: index_root.to_path_buf(),
            source,
        })?;
        let root = open(
            index_root,
            OFlags::RDONLY
                | OFlags::CLOEXEC
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)
        .map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "open no-follow index root",
            path: index_root.to_path_buf(),
            source,
        })?;
        let root_file = File::from(root);
        let root_metadata =
            root_file
                .metadata()
                .map_err(|source| IndexRootWriterLeaseError::Io {
                    operation: "inspect index-root descriptor",
                    path: index_root.to_path_buf(),
                    source,
                })?;
        if !root_metadata.is_dir() {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: index_root.to_path_buf(),
                detail: "index root must be a no-follow directory".to_owned(),
            });
        }
        let owner_uid = rustix::process::geteuid().as_raw();
        if root_metadata.uid() != owner_uid {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: index_root.to_path_buf(),
                detail: format!(
                    "index root is owned by uid {}, current effective uid is {owner_uid}",
                    root_metadata.uid()
                ),
            });
        }

        let lock_path = index_root.join(INDEX_ROOT_WRITER_LOCK_FILE);
        let lock = openat(
            &root_file,
            INDEX_ROOT_WRITER_LOCK_FILE,
            OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(std::io::Error::from)
        .map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "open no-follow index-root writer lock",
            path: lock_path.clone(),
            source,
        })?;
        let mut lock_file = File::from(lock);
        let lock_metadata =
            lock_file
                .metadata()
                .map_err(|source| IndexRootWriterLeaseError::Io {
                    operation: "inspect index-root writer lock",
                    path: lock_path.clone(),
                    source,
                })?;
        validate_lock_metadata(&lock_path, &lock_metadata, owner_uid)?;

        if let Err(source) = flock(&lock_file, FlockOperation::NonBlockingLockExclusive) {
            if source == rustix::io::Errno::AGAIN {
                let owner_pid = read_index_root_writer_record(&lock_path, &lock_file)
                    .ok()
                    .flatten()
                    .map(|record| record.pid);
                return Err(IndexRootWriterLeaseError::Busy {
                    path: lock_path,
                    owner_pid,
                });
            }
            return Err(IndexRootWriterLeaseError::Io {
                operation: "acquire index-root writer flock",
                path: lock_path,
                source: std::io::Error::from(source),
            });
        }

        let owner_record = IndexRootWriterOwnerRecord::current(
            index_root,
            owner_uid,
            root_metadata.dev(),
            root_metadata.ino(),
            lock_metadata.dev(),
            lock_metadata.ino(),
        )?;
        write_index_root_writer_record(&lock_path, &mut lock_file, owner_record)?;
        root_file
            .sync_all()
            .map_err(|source| IndexRootWriterLeaseError::Io {
                operation: "sync index-root writer lock directory",
                path: index_root.to_path_buf(),
                source,
            })?;

        let lease = Self {
            index_root: index_root.to_path_buf(),
            lock_path,
            root_file,
            lock_file,
            owner_record,
        };
        lease.fence("writer admission")?;
        Ok(lease)
    }

    /// Fail closed on targets without the required kernel and no-follow
    /// semantics.
    #[cfg(not(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    )))]
    pub fn acquire(index_root: &Path) -> Result<Self, IndexRootWriterLeaseError> {
        Err(IndexRootWriterLeaseError::Unsupported {
            path: index_root.to_path_buf(),
        })
    }

    /// Revalidate the held owner/root/lock capability before a shared mutation.
    ///
    /// # Errors
    ///
    /// Returns a fail-closed error if either pathname was substituted, owner or
    /// file type changed, the persistent lock gained a hard link, or the sealed
    /// owner record no longer matches this holder.
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    pub fn fence(&self, boundary: &'static str) -> Result<(), IndexRootWriterLeaseError> {
        use std::os::unix::fs::MetadataExt;

        let root_descriptor =
            self.root_file
                .metadata()
                .map_err(|source| IndexRootWriterLeaseError::Io {
                    operation: "inspect held index-root descriptor",
                    path: self.index_root.clone(),
                    source,
                })?;
        let root_path = std::fs::symlink_metadata(&self.index_root).map_err(|source| {
            IndexRootWriterLeaseError::Io {
                operation: "resolve held index-root pathname",
                path: self.index_root.clone(),
                source,
            }
        })?;
        if !root_descriptor.is_dir()
            || !root_path.is_dir()
            || root_descriptor.dev() != self.owner_record.root_device
            || root_descriptor.ino() != self.owner_record.root_inode
            || root_path.dev() != self.owner_record.root_device
            || root_path.ino() != self.owner_record.root_inode
            || root_descriptor.uid() != self.owner_record.owner_uid
            || root_path.uid() != self.owner_record.owner_uid
        {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: self.index_root.clone(),
                detail: format!(
                    "{boundary}: index-root pathname/descriptor identity or owner changed"
                ),
            });
        }

        let lock_descriptor =
            self.lock_file
                .metadata()
                .map_err(|source| IndexRootWriterLeaseError::Io {
                    operation: "inspect held index-root lock descriptor",
                    path: self.lock_path.clone(),
                    source,
                })?;
        validate_lock_metadata(
            &self.lock_path,
            &lock_descriptor,
            self.owner_record.owner_uid,
        )?;
        let lock_path = std::fs::symlink_metadata(&self.lock_path).map_err(|source| {
            IndexRootWriterLeaseError::Io {
                operation: "resolve held index-root lock pathname",
                path: self.lock_path.clone(),
                source,
            }
        })?;
        if !lock_path.file_type().is_file()
            || lock_path.dev() != self.owner_record.lock_device
            || lock_path.ino() != self.owner_record.lock_inode
            || lock_descriptor.dev() != self.owner_record.lock_device
            || lock_descriptor.ino() != self.owner_record.lock_inode
            || lock_path.uid() != self.owner_record.owner_uid
        {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: self.lock_path.clone(),
                detail: format!(
                    "{boundary}: writer-lock pathname/descriptor identity or owner changed"
                ),
            });
        }

        let observed = read_index_root_writer_record(&self.lock_path, &self.lock_file)?
            .ok_or_else(|| IndexRootWriterLeaseError::Invalid {
                path: self.lock_path.clone(),
                detail: format!("{boundary}: sealed writer owner record is absent"),
            })?;
        if observed != self.owner_record {
            return Err(IndexRootWriterLeaseError::Invalid {
                path: self.lock_path.clone(),
                detail: format!("{boundary}: sealed writer owner record changed"),
            });
        }
        Ok(())
    }

    /// Fail closed on targets without the required kernel and no-follow
    /// semantics.
    #[cfg(not(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    )))]
    pub fn fence(&self, _: &'static str) -> Result<(), IndexRootWriterLeaseError> {
        Err(IndexRootWriterLeaseError::Unsupported {
            path: self.index_root.clone(),
        })
    }

    /// Path to the index root bound by this lease.
    #[must_use]
    pub fn index_root(&self) -> &Path {
        &self.index_root
    }
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn validate_lock_metadata(
    path: &Path,
    metadata: &std::fs::Metadata,
    owner_uid: u32,
) -> Result<(), IndexRootWriterLeaseError> {
    use std::os::unix::fs::MetadataExt;

    if !metadata.file_type().is_file() {
        return Err(IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail: "persistent writer lock must be a no-follow regular file".to_owned(),
        });
    }
    if metadata.uid() != owner_uid {
        return Err(IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail: format!(
                "writer lock is owned by uid {}, expected uid {owner_uid}",
                metadata.uid()
            ),
        });
    }
    if metadata.nlink() != 1 {
        return Err(IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail: format!(
                "persistent writer lock must have exactly one link, found {}",
                metadata.nlink()
            ),
        });
    }
    if metadata.mode() & 0o077 != 0 {
        return Err(IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail: format!(
                "persistent writer lock permissions must be owner-only, found {:04o}",
                metadata.mode() & 0o7777
            ),
        });
    }
    Ok(())
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn read_index_root_writer_record(
    path: &Path,
    file: &File,
) -> Result<Option<IndexRootWriterOwnerRecord>, IndexRootWriterLeaseError> {
    use std::os::unix::fs::FileExt;

    let length = file
        .metadata()
        .map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "inspect index-root writer record",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length == 0 {
        return Ok(None);
    }
    if length != u64::try_from(INDEX_ROOT_WRITER_RECORD_BYTES).expect("record size fits u64") {
        return Err(IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail: format!(
                "v1 writer record must be exactly {INDEX_ROOT_WRITER_RECORD_BYTES} bytes, found {length}"
            ),
        });
    }
    let mut bytes = [0_u8; INDEX_ROOT_WRITER_RECORD_BYTES];
    file.read_exact_at(&mut bytes, 0)
        .map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "read index-root writer record",
            path: path.to_path_buf(),
            source,
        })?;
    IndexRootWriterOwnerRecord::from_bytes(&bytes)
        .map(Some)
        .map_err(|detail| IndexRootWriterLeaseError::Invalid {
            path: path.to_path_buf(),
            detail,
        })
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn write_index_root_writer_record(
    path: &Path,
    file: &mut File,
    record: IndexRootWriterOwnerRecord,
) -> Result<(), IndexRootWriterLeaseError> {
    let bytes = record.to_bytes();
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
        .and_then(|()| file.write_all(&bytes))
        .and_then(|()| file.sync_all())
        .map_err(|source| IndexRootWriterLeaseError::Io {
            operation: "persist index-root writer record",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
impl Drop for IndexRootWriterLease {
    fn drop(&mut self) {
        use rustix::fs::{FlockOperation, flock};

        // The record is intentionally retained as historical owner context.
        // Acquiring the kernel flock is sufficient to overwrite it; normal
        // release never mutates or unlinks the persistent lock inode.
        let _ = flock(&self.lock_file, FlockOperation::Unlock);
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

/// Write a lock sentinel file to the given path (unconditional overwrite).
///
/// # Errors
///
/// Returns `Err` if the file cannot be written.
pub fn write_sentinel(path: &Path, sentinel: &LockSentinel) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(sentinel)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json.as_bytes())
}

/// Atomically create a sentinel file using `O_CREAT|O_EXCL`.
///
/// Returns `Ok(())` if the file was created, `Err(AlreadyExists)` if another
/// process won the race.
fn write_sentinel_exclusive(path: &Path, sentinel: &LockSentinel) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(sentinel)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
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

/// Maximum retries when racing against stale sentinel cleanup. Two is enough:
/// one for the initial stale removal, one in case a concurrent process also
/// removed the stale file and won the `create_new` race.
const SENTINEL_ACQUIRE_MAX_RETRIES: usize = 2;

/// Attempt to acquire a sentinel-based lock. If a stale sentinel exists (dead
/// PID or exceeds timeout), it is cleaned up and the lock is granted.
///
/// Uses `O_CREAT|O_EXCL` (via [`OpenOptions::create_new`]) to atomically
/// create the sentinel file, eliminating the TOCTOU race between checking
/// for an existing lock and writing a new one.
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
    let sentinel = LockSentinel::current(resource, holder);

    for _ in 0..=SENTINEL_ACQUIRE_MAX_RETRIES {
        // Attempt atomic exclusive creation — only one process can succeed.
        match write_sentinel_exclusive(path, &sentinel) {
            Ok(()) => {
                debug!(
                    pid = sentinel.pid,
                    resource, holder, "Lock sentinel acquired"
                );
                return Ok(sentinel);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // File exists — check if the holder is alive and not stale.
                match read_sentinel(path) {
                    Ok(existing) if !existing.is_holder_alive() => {
                        warn!(
                            pid = existing.pid,
                            resource = existing.resource,
                            "Recovering stale lock sentinel (holder PID is dead)"
                        );
                        remove_sentinel(path)?;
                    }
                    Ok(existing) if existing.is_stale(stale_threshold) => {
                        warn!(
                            pid = existing.pid,
                            age_ms = existing.created_at_ms,
                            resource = existing.resource,
                            "Recovering stale lock sentinel (exceeded timeout)"
                        );
                        remove_sentinel(path)?;
                    }
                    Ok(existing) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::WouldBlock,
                            format!(
                                "Resource '{}' is locked by PID {} ({})",
                                existing.resource, existing.pid, existing.holder
                            ),
                        ));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // Sentinel was removed between our create_new and read
                        // (another process cleaned it up). Retry creation.
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(e) => return Err(e),
        }
    }

    // Exhausted retries — another process is actively contending.
    Err(std::io::Error::new(
        std::io::ErrorKind::WouldBlock,
        format!(
            "Resource '{resource}' acquisition failed after {} retries (contention)",
            SENTINEL_ACQUIRE_MAX_RETRIES + 1
        ),
    ))
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
    /// Access to the lexical index.
    pub lexical: AccessMode,
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
            lexical: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "classify",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::None,
            lexical: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "embed_fast",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            lexical: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "embed_quality",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            lexical: AccessMode::None,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "lexical_index",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::None,
            lexical: AccessMode::ReadWrite,
            cache: AccessMode::None,
        },
        PipelineStageAccess {
            stage: "serve_queries",
            catalog: AccessMode::ReadOnly,
            queue: AccessMode::None,
            fsvi: AccessMode::ReadOnly,
            lexical: AccessMode::ReadOnly,
            cache: AccessMode::ReadOnly,
        },
        PipelineStageAccess {
            stage: "refresh_worker",
            catalog: AccessMode::ReadOnly,
            queue: AccessMode::ReadWrite,
            fsvi: AccessMode::ReadWrite,
            lexical: AccessMode::None,
            cache: AccessMode::ReadWrite,
        },
        PipelineStageAccess {
            stage: "compaction",
            catalog: AccessMode::ReadWrite,
            queue: AccessMode::None,
            fsvi: AccessMode::ReadWrite,
            lexical: AccessMode::ReadWrite,
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
#[cfg(unix)]
#[allow(unsafe_code, clippy::cast_possible_wrap)]
pub(crate) fn is_pid_alive(pid: u32) -> bool {
    // kill(pid, 0) is the canonical Unix check: returns 0 if the process
    // exists (or EPERM if we lack permission, which still means alive).
    // SAFETY: signal 0 does not deliver a signal; it only checks existence.
    let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if ret == 0 {
        return true;
    }
    // EPERM means process exists but we can't signal it — still alive.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
pub(crate) fn is_pid_alive(_pid: u32) -> bool {
    // Conservative: assume alive on unsupported platforms.
    true
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
        assert!(LockLevel::FsviSegment < LockLevel::LexicalWriter);
        assert!(LockLevel::LexicalWriter < LockLevel::AdaptiveState);
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
        assert_eq!(format!("{}", LockLevel::LexicalWriter), "lexical_writer(5)");
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

    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    mod index_root_writer_lease_tests {
        use std::fs;
        use std::process::{Command, Output, Stdio};
        use std::thread;

        use frankensearch_index::VectorIndex;

        use super::*;

        const HELPER_ROOT_ENV: &str = "FSFS_WRITER_LEASE_TEST_ROOT";
        const HELPER_OPERATION_ENV: &str = "FSFS_WRITER_LEASE_TEST_OPERATION";
        const HELPER_GENERATION_ENV: &str = "FSFS_WRITER_LEASE_TEST_GENERATION";

        fn retry_writer_lease(index_root: &Path) -> IndexRootWriterLease {
            for _ in 0..400 {
                match IndexRootWriterLease::acquire(index_root) {
                    Ok(lease) => return lease,
                    Err(IndexRootWriterLeaseError::Busy { .. }) => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("unexpected writer-lease admission failure: {error}"),
                }
            }
            panic!("timed out retrying index-root writer lease")
        }

        fn helper_command(index_root: &Path, operation: &str) -> Command {
            let mut command = Command::new(std::env::current_exe().expect("current test binary"));
            command
                .arg("index_root_writer_lease_subprocess_helper")
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env(HELPER_ROOT_ENV, index_root)
                .env(HELPER_OPERATION_ENV, operation);
            command
        }

        fn assert_helper_output(output: Output, token: &str) {
            assert!(
                output.status.success(),
                "writer-lease helper failed: status={:?}, stdout={}, stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains(token),
                "writer-lease helper did not execute expected branch {token}: stdout={}",
                String::from_utf8_lossy(&output.stdout)
            );
        }

        #[test]
        fn index_root_writer_owner_record_roundtrips_and_rejects_tampering() {
            let record = IndexRootWriterOwnerRecord {
                pid: 42,
                owner_uid: 1000,
                process_start: 17,
                host_id: 23,
                boot_id: 29,
                nonce_high: 31,
                nonce_low: 37,
                root_device: 41,
                root_inode: 43,
                lock_device: 47,
                lock_inode: 53,
            };
            let bytes = record.to_bytes();
            assert_eq!(
                IndexRootWriterOwnerRecord::from_bytes(&bytes).expect("decode sealed record"),
                record
            );

            let mut tampered = bytes;
            tampered[48] ^= 0x80;
            assert!(
                IndexRootWriterOwnerRecord::from_bytes(&tampered)
                    .expect_err("tampered nonce must invalidate checksum")
                    .contains("checksum")
            );
        }

        #[test]
        fn index_root_writer_lease_keeps_one_persistent_inode_across_release() {
            use std::os::unix::fs::MetadataExt;

            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            let first = IndexRootWriterLease::acquire(&index_root).expect("first lease");
            first.fence("first holder").expect("fence first holder");
            let lock_path = index_root.join(INDEX_ROOT_WRITER_LOCK_FILE);
            let first_inode = fs::metadata(&lock_path).expect("first lock metadata").ino();

            assert!(matches!(
                IndexRootWriterLease::acquire(&index_root),
                Err(IndexRootWriterLeaseError::Busy {
                    owner_pid: Some(pid),
                    ..
                }) if pid == std::process::id()
            ));

            drop(first);
            assert!(
                lock_path.is_file(),
                "normal release must retain the persistent lock inode"
            );
            let second = IndexRootWriterLease::acquire(&index_root).expect("second lease");
            second.fence("second holder").expect("fence second holder");
            assert_eq!(
                fs::metadata(&lock_path)
                    .expect("second lock metadata")
                    .ino(),
                first_inode,
                "release/reacquire must not replace the lock inode"
            );
        }

        #[test]
        fn index_root_writer_lease_fails_closed_after_root_substitution() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            let displaced_root = temp.path().join("displaced-index");
            let lease = IndexRootWriterLease::acquire(&index_root).expect("lease");

            fs::rename(&index_root, &displaced_root).expect("displace held root");
            fs::create_dir(&index_root).expect("substitute index root");
            let guarded_marker = index_root.join("must-not-publish");
            if lease.fence("root substitution").is_ok() {
                fs::write(&guarded_marker, b"invalid publication")
                    .expect("write only if fencing incorrectly succeeded");
            }
            assert!(
                !guarded_marker.exists(),
                "failed root identity must prevent the guarded publication"
            );
        }

        #[test]
        fn index_root_writer_lease_fails_closed_after_lock_substitution() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            let lease = IndexRootWriterLease::acquire(&index_root).expect("lease");
            let lock_path = index_root.join(INDEX_ROOT_WRITER_LOCK_FILE);
            let displaced_lock = index_root.join(".fsfs-writer.lock.displaced");

            fs::rename(&lock_path, &displaced_lock).expect("displace held lock");
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
                .expect("substitute writer lock");
            assert!(matches!(
                lease.fence("lock substitution"),
                Err(IndexRootWriterLeaseError::Invalid { .. })
            ));
        }

        #[test]
        fn index_root_writer_lease_fails_closed_after_owner_record_tampering() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            let lease = IndexRootWriterLease::acquire(&index_root).expect("lease");
            let lock_path = index_root.join(INDEX_ROOT_WRITER_LOCK_FILE);
            let mut bytes = fs::read(&lock_path).expect("read sealed owner record");
            bytes[56] ^= 0x40;
            fs::write(&lock_path, bytes).expect("tamper owner record");

            assert!(matches!(
                lease.fence("owner record tampering"),
                Err(IndexRootWriterLeaseError::Invalid { .. })
            ));
        }

        #[test]
        fn fresh_process_observes_busy_then_acquires_while_readers_continue() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            fs::create_dir_all(&index_root).expect("create index root");
            fs::write(index_root.join("readable"), b"stable snapshot").expect("write read fixture");
            let lease = IndexRootWriterLease::acquire(&index_root).expect("parent lease");

            assert_helper_output(
                helper_command(&index_root, "busy")
                    .output()
                    .expect("run busy helper"),
                "FSFS_HELPER_BUSY_OK",
            );
            assert_helper_output(
                helper_command(&index_root, "read")
                    .output()
                    .expect("run reader helper"),
                "FSFS_HELPER_READ_OK",
            );

            drop(lease);
            assert_helper_output(
                helper_command(&index_root, "acquire")
                    .output()
                    .expect("run acquisition helper"),
                "FSFS_HELPER_ACQUIRE_OK",
            );
            assert!(index_root.join("fresh-process-acquired").is_file());
        }

        #[test]
        fn concurrent_publishers_cannot_expose_a_mixed_generation() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");

            let mut first = helper_command(&index_root, "publish");
            first
                .env(HELPER_GENERATION_ENV, "generation-a")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut second = helper_command(&index_root, "publish");
            second
                .env(HELPER_GENERATION_ENV, "generation-b")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let first = first.spawn().expect("spawn first publisher");
            let second = second.spawn().expect("spawn second publisher");
            let first_output = first.wait_with_output().expect("wait first publisher");
            let second_output = second.wait_with_output().expect("wait second publisher");
            assert_helper_output(first_output, "FSFS_HELPER_PUBLISH_OK");
            assert_helper_output(second_output, "FSFS_HELPER_PUBLISH_OK");

            let lexical =
                fs::read_to_string(index_root.join("lexical.generation")).expect("read lexical");
            let vector =
                fs::read_to_string(index_root.join("vector.generation")).expect("read vector");
            assert_eq!(
                lexical, vector,
                "a complete writer critical section must publish one paired generation"
            );
            assert!(matches!(lexical.as_str(), "generation-a" | "generation-b"));
        }

        #[test]
        fn daemon_compaction_cannot_erase_a_concurrent_acknowledged_append() {
            let temp = tempfile::tempdir().expect("tempdir");
            let index_root = temp.path().join("index");
            fs::create_dir_all(&index_root).expect("create index root");
            let vector_path = index_root.join("vectors.fsvi");
            let mut writer =
                VectorIndex::create(&vector_path, "writer-lease-test", 4).expect("create FSVI");
            writer
                .write_record("base", &[1.0, 0.0, 0.0, 0.0])
                .expect("write base record");
            writer.finish().expect("finish FSVI");
            let mut seed = VectorIndex::open(&vector_path).expect("open FSVI for seed WAL");
            seed.append("seed", &[0.0, 1.0, 0.0, 0.0])
                .expect("seed WAL");
            drop(seed);

            let mut compact_command = helper_command(&index_root, "vector-compact");
            compact_command
                .env("FSFS_WRITER_LEASE_TEST_VECTOR", &vector_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let compact_child = compact_command.spawn().expect("spawn compactor");
            let ready_path = index_root.join("compactor-ready");
            for _ in 0..200 {
                if ready_path.is_file() {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                ready_path.is_file(),
                "compactor must hold the lease before append starts"
            );

            let mut append_command = helper_command(&index_root, "vector-append");
            append_command
                .env("FSFS_WRITER_LEASE_TEST_VECTOR", &vector_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let append_child = append_command.spawn().expect("spawn appender");
            assert_helper_output(
                compact_child
                    .wait_with_output()
                    .expect("wait vector compactor"),
                "FSFS_HELPER_VECTOR_COMPACT_OK",
            );
            assert_helper_output(
                append_child
                    .wait_with_output()
                    .expect("wait vector appender"),
                "FSFS_HELPER_VECTOR_APPEND_OK",
            );

            let admitted = VectorIndex::open(&vector_path).expect("reopen final FSVI");
            let hits = admitted
                .search_top_k(&[0.0, 0.0, 1.0, 0.0], 10, None)
                .expect("search final FSVI");
            assert!(
                hits.iter().any(|hit| hit.doc_id == "survivor"),
                "acknowledged append must survive daemon compaction"
            );
        }

        #[test]
        fn index_root_writer_lease_subprocess_helper() {
            let Some(index_root) = std::env::var_os(HELPER_ROOT_ENV).map(PathBuf::from) else {
                return;
            };
            let operation =
                std::env::var(HELPER_OPERATION_ENV).expect("helper operation environment");
            match operation.as_str() {
                "busy" => {
                    assert!(matches!(
                        IndexRootWriterLease::acquire(&index_root),
                        Err(IndexRootWriterLeaseError::Busy { .. })
                    ));
                    println!("FSFS_HELPER_BUSY_OK");
                }
                "read" => {
                    assert_eq!(
                        fs::read(index_root.join("readable")).expect("read shared snapshot"),
                        b"stable snapshot"
                    );
                    println!("FSFS_HELPER_READ_OK");
                }
                "acquire" => {
                    let lease = retry_writer_lease(&index_root);
                    lease.fence("fresh-process acquisition").expect("fence");
                    fs::write(index_root.join("fresh-process-acquired"), b"acquired")
                        .expect("write acquisition marker");
                    println!("FSFS_HELPER_ACQUIRE_OK");
                }
                "publish" => {
                    let generation =
                        std::env::var(HELPER_GENERATION_ENV).expect("publisher generation");
                    let lease = retry_writer_lease(&index_root);
                    lease.fence("paired lexical publication").expect("fence");
                    fs::write(index_root.join("lexical.generation"), &generation)
                        .expect("publish lexical generation");
                    thread::sleep(Duration::from_millis(75));
                    lease.fence("paired vector publication").expect("fence");
                    fs::write(index_root.join("vector.generation"), &generation)
                        .expect("publish vector generation");
                    println!("FSFS_HELPER_PUBLISH_OK");
                }
                "vector-compact" => {
                    let vector_path = PathBuf::from(
                        std::env::var_os("FSFS_WRITER_LEASE_TEST_VECTOR")
                            .expect("vector path environment"),
                    );
                    let lease = retry_writer_lease(&index_root);
                    lease.fence("daemon compactor ready").expect("fence");
                    fs::write(index_root.join("compactor-ready"), b"ready")
                        .expect("write compactor handshake");
                    thread::sleep(Duration::from_millis(100));
                    lease.fence("daemon compaction publication").expect("fence");
                    VectorIndex::open(&vector_path)
                        .expect("open vector for compaction")
                        .compact()
                        .expect("compact vector");
                    println!("FSFS_HELPER_VECTOR_COMPACT_OK");
                }
                "vector-append" => {
                    let vector_path = PathBuf::from(
                        std::env::var_os("FSFS_WRITER_LEASE_TEST_VECTOR")
                            .expect("vector path environment"),
                    );
                    let lease = retry_writer_lease(&index_root);
                    lease.fence("append WAL publication").expect("fence");
                    VectorIndex::open(&vector_path)
                        .expect("open vector for append")
                        .append("survivor", &[0.0, 0.0, 1.0, 0.0])
                        .expect("append surviving vector");
                    println!("FSFS_HELPER_VECTOR_APPEND_OK");
                }
                other => panic!("unknown writer-lease helper operation {other}"),
            }
        }
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
        assert_eq!(query_stage.lexical, AccessMode::ReadOnly);
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
