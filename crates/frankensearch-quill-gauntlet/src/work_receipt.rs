//! QG-1 H2: actual-work, queue, worker-role, and lifecycle receipts.
//!
//! The QG-1 bulk cells configure pool capacity (`spec.threads`) but the
//! decision artifacts have never attested what work was *actually* performed,
//! by which worker roles, at what CPU cost, or whether every background
//! worker was joined before the clock stopped. This module is the typed
//! receipt contract that closes that gap:
//!
//! 1. Configured capacity and observed activity are separate fields
//!    ([`ConcurrencyReceipt::configured_threads`] vs
//!    [`WidthObservation`] / [`RoleCensus`]); a missing collector is a typed
//!    absence, never a fabricated `actual = 1`.
//! 2. Work and byte counters are observed at the harness↔engine boundary
//!    ([`WorkCounters`]), engine-reported quantities carry the exact seam
//!    they were read from ([`EngineByteObservation`]), and quantities no
//!    engine can honestly report stay typed gaps
//!    ([`ByteStageObservation::StructurallyUnobservable`]).
//! 3. The active-concurrency integral is derived from the process CPU-time
//!    identity — the integral of active threads over the window *is* the
//!    process CPU time consumed in the window — so it is observed, never
//!    modeled ([`ActiveConcurrencyIntegral`]). Role-resolved CPU is a
//!    decomposition of that same total with an explicit unattributed
//!    residual, so the numbers can never silently double-count.
//! 4. Terminal lifecycle state (joined, drained, pending-zero) is attested
//!    per engine ([`TerminalJoin`]) and validated fail-closed
//!    ([`WorkReceipt::validate`]), including tamper evidence via a canonical
//!    xxh3 self-digest.
//!
//! The receipts compose with the QG-1 H1 continuous-timing window through
//! the [`LifecycleObserver`] seam (H1 ships the seam; this module ships the
//! first real observer) and also run standalone on the per-call path.

use std::collections::BTreeMap;
use std::time::Instant;

use frankensearch_core::IndexableDocument;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use xxhash_rust::xxh3::Xxh3;

/// Schema tag stamped on every work-receipt evidence object.
pub const WORK_RECEIPT_SCHEMA_VERSION: &str = "qg1-work-receipt-v1";

/// Upper bound on threads retained in one role census sample.
pub const MAX_CENSUS_THREADS: usize = 512;

/// Slack added to the census thread count when bounding plausible
/// concurrency: short-lived threads born and reaped between samples are
/// invisible to the census but real to the scheduler.
pub const CENSUS_SLACK_THREADS: u64 = 4;

/// Upper bound on census samples taken inside one timed window.
pub const MAX_CENSUS_SAMPLES_IN_WINDOW: u64 = 8;

/// The only process-CPU seam the v1 receipt admits.
pub const PROCESS_CPU_SEAM: &str = "linux:/proc/self/stat utime+stime";

/// The only per-thread census seam the v1 receipt admits.
pub const ROLE_CENSUS_SEAM: &str = "linux:/proc/self/task/<tid>/{stat,comm}";

// ─── QG-1 H1 composition seam ───────────────────────────────────────────────
//
// H1 and H2 share these definitions so the continuous clock and the
// actual-work collector cannot drift into parallel lifecycle contracts.

/// Lifecycle boundary reported through the H2 receipts seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecyclePhase {
    /// The first feed call is about to be issued (window origin).
    FirstFeed,
    /// The last feed call returned.
    FeedComplete,
    /// The terminal commit returned.
    CommitComplete,
    /// Searchable count and terminal probe verified.
    SearchableVerified,
    /// Engine quiescence joined (always the final phase).
    QuiescenceJoined,
}

impl LifecyclePhase {
    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FirstFeed => "first_feed",
            Self::FeedComplete => "feed_complete",
            Self::CommitComplete => "commit_complete",
            Self::SearchableVerified => "searchable_verified",
            Self::QuiescenceJoined => "quiescence_joined",
        }
    }

    /// Canonical phase order for one window.
    pub const ORDERED: [Self; 5] = [
        Self::FirstFeed,
        Self::FeedComplete,
        Self::CommitComplete,
        Self::SearchableVerified,
        Self::QuiescenceJoined,
    ];
}

/// QG-1 H2 composition seam.
///
/// The continuous runner (H1) and the per-call bulk runner both report every
/// phase boundary through this trait so receipt collectors attach without
/// changing either runner's lifecycle ownership.
pub trait LifecycleObserver {
    /// Called at each phase boundary with window-relative elapsed time.
    fn on_phase(&mut self, phase: LifecyclePhase, window_elapsed_ns: u64);

    /// Called after one feed batch was accepted at the harness-engine seam.
    fn on_feed_batch(&mut self, _documents: u64, _bytes: u64) {}
}

/// Default observer: records nothing, costs nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLifecycleObserver;

impl LifecycleObserver for NoopLifecycleObserver {
    fn on_phase(&mut self, _phase: LifecyclePhase, _window_elapsed_ns: u64) {}
}

/// Byte count for one lifecycle stage: observed from the actual documents at
/// the stage boundary, or typed as structurally unobservable at this seam.
///
/// The unobservable variant is an explicit, named gap — never a silently
/// copied denominator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteStageObservation {
    /// Byte total observed from the actual documents at this stage.
    Observed(u64),
    /// The stage's byte total cannot be observed at this seam; `seam` names
    /// exactly which boundary lacks the observation.
    StructurallyUnobservable { seam: String },
}

/// Bytes exactly as the sealed corpus manifests count them for one document.
///
/// Covers id, optional title, content, and every metadata key/value. Shared
/// by manifest sealing and window-time observed byte accounting so the two
/// can never drift.
#[must_use]
pub fn document_bytes(document: &IndexableDocument) -> u64 {
    let mut bytes = document.id.len() as u64 + document.content.len() as u64;
    if let Some(title) = &document.title {
        bytes += title.len() as u64;
    }
    for (key, value) in &document.metadata {
        bytes += key.len() as u64 + value.len() as u64;
    }
    bytes
}

// ─── Typed failure surface ──────────────────────────────────────────────────

/// Typed rejection surface for the work-receipt contract.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkReceiptError {
    #[error("work receipt schema {actual:?} does not match {expected:?}")]
    SchemaVersionMismatch { expected: String, actual: String },
    #[error("QUILL_PERF_WORK_RECEIPTS must be `on` or `off`, got {value:?}")]
    InvalidWorkReceiptMode { value: String },
    #[error("receipt binding field {field} is empty")]
    EmptyBindingField { field: &'static str },
    #[error("receipt binding field {field} is not 64 lowercase hex chars: {value:?}")]
    MalformedSha256 { field: &'static str, value: String },
    #[error("receipt engine {engine:?} is not `quill` or `tantivy`")]
    InvalidEngine { engine: String },
    #[error("receipt timing mode {value:?} is not `per-call` or `continuous`")]
    InvalidTimingMode { value: String },
    #[error("receipt window ended_ns {ended_ns} does not follow started_ns {started_ns}")]
    InvalidWindow { started_ns: u64, ended_ns: u64 },
    #[error("work inequality at {stage}: expected {expected} but observed {actual}")]
    CounterInequality {
        stage: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("receipt recorded zero feed calls")]
    NoFeedCalls,
    #[error("terminal commit calls must be exactly 1, got {actual}")]
    TerminalCommitCalls { actual: u64 },
    #[error(
        "{field} names no seam; an unnamed gap is indistinguishable from a skipped observation"
    )]
    UnnamedSeam { field: &'static str },
    #[error("configured width {actual} does not match the cell contract width {expected}")]
    ConfiguredWidthMismatch { expected: u64, actual: u64 },
    #[error("configured width is zero")]
    ZeroConfiguredWidth,
    #[error("observed width of zero threads is a fabrication: {seam}")]
    FabricatedWidth { seam: String },
    #[error(
        "{field} claims platform {platform:?} but the v1 collector exists only for \
         {expected:?}; a sampled observation on another platform is a masquerade"
    )]
    PlatformMasquerade {
        field: &'static str,
        platform: String,
        expected: &'static str,
    },
    #[error("{field} names seam {seam:?} but the v1 receipt admits only {expected:?}")]
    UnknownSeam {
        field: &'static str,
        seam: String,
        expected: &'static str,
    },
    #[error(
        "active-concurrency integral is present while process CPU time is unavailable; \
         an integral without its observed source is fabricated"
    )]
    FabricatedIntegral,
    #[error(
        "active-concurrency integral {integral_thread_ns}ns does not equal observed \
         process CPU {process_cpu_ns}ns; the integral must be the CPU-time identity"
    )]
    IntegralIdentityViolation {
        integral_thread_ns: u64,
        process_cpu_ns: u64,
    },
    #[error(
        "stored mean active concurrency {stored_millithreads} does not match \
         {recomputed_millithreads} recomputed from integral/wall"
    )]
    MeanConcurrencyMismatch {
        stored_millithreads: u64,
        recomputed_millithreads: u64,
    },
    #[error(
        "impossible concurrency: integral {integral_thread_ns}ns over wall {wall_ns}ns \
         exceeds {plausible_threads} plausible threads"
    )]
    ImpossibleConcurrency {
        integral_thread_ns: u64,
        wall_ns: u64,
        plausible_threads: u64,
    },
    #[error(
        "role-resolved CPU is present without a sampled census; roles cannot be resolved blind"
    )]
    RolesWithoutCensus,
    #[error(
        "role CPU decomposition violated: roles {role_sum_ns}ns + unattributed \
         {unattributed_ns}ns != process {process_cpu_ns}ns"
    )]
    RoleCpuIdentityViolation {
        role_sum_ns: u64,
        unattributed_ns: u64,
        process_cpu_ns: u64,
    },
    #[error("role census retains {threads} threads, over the {cap} bound")]
    CensusThreadCapExceeded { threads: usize, cap: usize },
    #[error("truncated census must carry its truncation caveat")]
    TruncationUnflagged,
    #[error("engine {engine} receipt is missing required worker role {role}")]
    MissingRequiredRole { engine: String, role: &'static str },
    #[error("phase timeline holds {actual} samples, expected {expected}")]
    PhaseCountMismatch { expected: usize, actual: usize },
    #[error("phase timeline violates canonical order at {phase}")]
    PhaseOrderViolation { phase: &'static str },
    #[error("terminal phase elapsed {quiescence_ns}ns does not equal window wall {wall_ns}ns")]
    PhaseWindowMismatch { quiescence_ns: u64, wall_ns: u64 },
    #[error(
        "engine {engine:?} paired with terminal join {join}; each engine must attest its own join contract"
    )]
    JoinContractMismatch { engine: String, join: &'static str },
    #[error("terminal join rearmed a writer; terminal lifecycle must leave no armed worker behind")]
    WriterRearmedInTerminalJoin,
    #[error("feed loop did not drain every prepared batch")]
    UndrainedFeed,
    #[error("terminal state is not pending-zero: committed {committed} vs accepted {accepted}")]
    PendingWorkAtTerminal { committed: u64, accepted: u64 },
    #[error("terminal reason is empty")]
    EmptyTerminalReason,
    #[error("terminal reason {reason:?} completed but the retry predicate claims retryable")]
    RetryablePredicateViolation { reason: String },
    #[error("summed measured calls {measured_sum_ns}ns exceed the window wall {wall_ns}ns")]
    MeasuredExceedsWall { measured_sum_ns: u64, wall_ns: u64 },
    #[error("{samples} census samples inside one window exceed the {cap} bound")]
    ExcessiveCensusSamples { samples: u64, cap: u64 },
    #[error(
        "receipt digest mismatch: sealed {expected} but recomputed {actual}; the receipt was tampered with"
    )]
    DigestMismatch { expected: String, actual: String },
    #[error("receipt serialization failed: {detail}")]
    Serialization { detail: String },
}

// ─── Environment gate ───────────────────────────────────────────────────────

/// Whether an invocation collects work receipts.
///
/// Defaults to [`Self::Off`] so the default per-call artifact byte shape and
/// the measured lanes are untouched until a run opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkReceiptMode {
    /// No collectors run; artifacts are byte-identical to the legacy shape.
    Off,
    /// Collectors run on QG-1 bulk cells; receipts ride the gate artifact.
    On,
}

impl WorkReceiptMode {
    /// Environment variable that selects the mode.
    pub const ENV_VAR: &'static str = "QUILL_PERF_WORK_RECEIPTS";

    /// Parse an optional environment value.
    ///
    /// # Errors
    ///
    /// Returns [`WorkReceiptError::InvalidWorkReceiptMode`] for any value
    /// other than `on` or `off`.
    pub fn parse(value: Option<&str>) -> Result<Self, WorkReceiptError> {
        match value {
            None | Some("off") => Ok(Self::Off),
            Some("on") => Ok(Self::On),
            Some(other) => Err(WorkReceiptError::InvalidWorkReceiptMode {
                value: other.to_owned(),
            }),
        }
    }

    /// Read the mode from [`Self::ENV_VAR`].
    ///
    /// # Errors
    ///
    /// Returns [`WorkReceiptError::InvalidWorkReceiptMode`] for an
    /// unrecognized value.
    pub fn from_env() -> Result<Self, WorkReceiptError> {
        let value = std::env::var(Self::ENV_VAR).ok();
        Self::parse(value.as_deref())
    }

    /// Whether collectors are enabled.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::On)
    }

    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

// ─── Platform observations ──────────────────────────────────────────────────

/// One observed CPU-time reading, or a typed absence.
///
/// The `Unavailable` variant is how a missing collector shows up: platform
/// and reason, never a fabricated zero or a copied configuration value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuTimeObservation {
    /// CPU nanoseconds read from a named seam.
    Observed { cpu_ns: u64, seam: String },
    /// No collector exists on this platform.
    Unavailable { platform: String, reason: String },
}

/// Worker role resolved from an observed thread name.
///
/// Attribution is name-based and fail-open to [`Self::Unattributed`]: a
/// thread whose name matches no known worker family is reported as observed
/// but unclassified, never guessed into a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    /// The harness thread driving the feed loop (and, for Quill, the
    /// synchronous writer itself).
    BenchCaller,
    /// A bench-pinned Rayon pool worker (Quill's indexing parallelism).
    RayonWorker,
    /// A Tantivy indexing worker thread.
    TantivyIndexWorker,
    /// Tantivy's segment updater thread.
    TantivySegmentUpdater,
    /// A Tantivy merge worker thread.
    TantivyMergeWorker,
    /// A Tantivy docstore compressor thread (one per store writer; these
    /// exit when their segment store closes, so they are recorded when seen
    /// but never required).
    TantivyDocstoreCompressor,
    /// An asupersync runtime/blocking-pool thread.
    AsupersyncRuntime,
    /// A live thread whose name matches no known worker family.
    Unattributed,
}

impl WorkerRole {
    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BenchCaller => "bench_caller",
            Self::RayonWorker => "rayon_worker",
            Self::TantivyIndexWorker => "tantivy_index_worker",
            Self::TantivySegmentUpdater => "tantivy_segment_updater",
            Self::TantivyMergeWorker => "tantivy_merge_worker",
            Self::TantivyDocstoreCompressor => "tantivy_docstore_compressor",
            Self::AsupersyncRuntime => "asupersync_runtime",
            Self::Unattributed => "unattributed",
        }
    }

    /// Parse a stable artifact label back into a role.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "bench_caller" => Some(Self::BenchCaller),
            "rayon_worker" => Some(Self::RayonWorker),
            "tantivy_index_worker" => Some(Self::TantivyIndexWorker),
            "tantivy_segment_updater" => Some(Self::TantivySegmentUpdater),
            "tantivy_merge_worker" => Some(Self::TantivyMergeWorker),
            "tantivy_docstore_compressor" => Some(Self::TantivyDocstoreCompressor),
            "asupersync_runtime" => Some(Self::AsupersyncRuntime),
            "unattributed" => Some(Self::Unattributed),
            _ => None,
        }
    }
}

/// Thread-name prefix the bench stamps on its pinned Rayon pool workers.
pub const BENCH_RAYON_THREAD_PREFIX: &str = "qg-rayon-";

/// Resolve a worker role from an observed thread name.
///
/// `is_caller` marks the thread id that entered the window (`gettid` of the
/// feeding thread), which needs no name to classify.
#[must_use]
pub fn classify_thread_role(name: &str, is_caller: bool) -> WorkerRole {
    if is_caller {
        return WorkerRole::BenchCaller;
    }
    if name.starts_with(BENCH_RAYON_THREAD_PREFIX) {
        return WorkerRole::RayonWorker;
    }
    // Tantivy 0.26.1 thread names, matched as prefixes because comm(5) caps
    // names at 15 bytes: `thrd-tantivy-index{i}` truncates to
    // `thrd-tantivy-in` (individual worker indices are unrecoverable from
    // comm — workers are counted, never distinguished), `segment_updater`
    // fits exactly, `merge_thread_{i}` fits for i < 100, and
    // `docstore-compressor-thread` truncates to `docstore-compre`.
    if name.starts_with("thrd-tantivy-in") {
        return WorkerRole::TantivyIndexWorker;
    }
    if name.starts_with("segment_updater") {
        return WorkerRole::TantivySegmentUpdater;
    }
    if name.starts_with("merge_thread_") {
        return WorkerRole::TantivyMergeWorker;
    }
    if name.starts_with("docstore-compre") {
        return WorkerRole::TantivyDocstoreCompressor;
    }
    if name.starts_with("asupersync-") || name.starts_with("coordinated-dea") {
        return WorkerRole::AsupersyncRuntime;
    }
    WorkerRole::Unattributed
}

/// One live thread observed by the census, with its in-window CPU delta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadRoleSample {
    /// Kernel thread id.
    pub tid: u64,
    /// Thread name as read from `comm` (15-byte kernel cap).
    pub name: String,
    /// Role resolved by [`classify_thread_role`].
    pub role: WorkerRole,
    /// CPU nanoseconds consumed inside the window (baseline-subtracted).
    pub window_cpu_ns: u64,
}

/// Typed caveat attached to a sampled census.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CensusCaveat {
    /// Per-thread CPU is read at clock-tick granularity; a thread that ran
    /// for less than one tick can read zero CPU while having actually run.
    ClockTickGranularity { tick_ns: u64 },
    /// Threads that exited before the census sample are invisible to
    /// role-resolved CPU; their time is folded into the process total and
    /// surfaces as the unattributed residual.
    ExitedThreadCpuUnattributed,
    /// The live thread list exceeded [`MAX_CENSUS_THREADS`]; only the first
    /// `retained` were kept.
    TruncatedThreadList { observed: u64, retained: u64 },
}

/// One sampled role census.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleCensusSample {
    /// Platform the collector ran on (v1: always `linux`).
    pub platform: String,
    /// Seam the sample was read from (v1: [`ROLE_CENSUS_SEAM`]).
    pub seam: String,
    /// `_SC_CLK_TCK` used to convert ticks to nanoseconds.
    pub clock_tick_hz: u64,
    /// Live threads at sample time (bounded by [`MAX_CENSUS_THREADS`]).
    pub threads: Vec<ThreadRoleSample>,
    /// Whether the live thread list was truncated.
    pub truncated: bool,
    /// Typed caveats bound to this sample.
    pub caveats: Vec<CensusCaveat>,
}

/// A role census: sampled, or a typed absence.
///
/// On platforms without a collector (macOS has no `/proc`), the absence is
/// typed — an Apple run can never fabricate `actual = 1` or copy configured
/// capacity into an "observed" field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleCensus {
    /// A real sample from a real collector.
    Sampled(RoleCensusSample),
    /// No collector exists on this platform.
    Unavailable { platform: String, reason: String },
}

impl RoleCensus {
    /// Role histogram of a sampled census (empty for a typed absence).
    #[must_use]
    pub fn role_counts(&self) -> BTreeMap<WorkerRole, u64> {
        let mut counts = BTreeMap::new();
        if let Self::Sampled(sample) = self {
            for thread in &sample.threads {
                *counts.entry(thread.role).or_insert(0) += 1;
            }
        }
        counts
    }
}

// ─── Linux collectors ───────────────────────────────────────────────────────

/// Raw cumulative CPU reading for one thread or the whole process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawCpuReading {
    /// Cumulative user+system clock ticks.
    pub ticks: u64,
}

/// Parse `utime + stime` (fields 14 and 15) out of a `/proc/.../stat` line.
///
/// The comm field (2) may contain spaces and parentheses; parsing is anchored
/// after the *last* `)` per proc(5).
#[must_use]
pub fn parse_stat_cpu_ticks(stat_line: &str) -> Option<RawCpuReading> {
    let after_comm = stat_line.rfind(')')?;
    let rest = stat_line.get(after_comm + 1..)?;
    let mut fields = rest.split_ascii_whitespace();
    // `rest` starts at field 3 (state); utime is field 14, stime field 15.
    let utime = fields.nth(11)?.parse::<u64>().ok()?;
    let stime = fields.next()?.parse::<u64>().ok()?;
    Some(RawCpuReading {
        ticks: utime.checked_add(stime)?,
    })
}

/// Convert cumulative clock ticks to nanoseconds.
#[must_use]
pub const fn ticks_to_ns(ticks: u64, clock_tick_hz: u64) -> u64 {
    if clock_tick_hz == 0 {
        return 0;
    }
    // ticks/hz seconds → ns; split to avoid overflow for large cumulative
    // readings (whole seconds first, remainder scaled exactly).
    let seconds = ticks / clock_tick_hz;
    let remainder = ticks % clock_tick_hz;
    seconds * 1_000_000_000 + (remainder * 1_000_000_000) / clock_tick_hz
}

#[cfg(target_os = "linux")]
fn clock_tick_hz() -> u64 {
    rustix::param::clock_ticks_per_second()
}

/// Sample cumulative process CPU time (includes exited threads).
///
/// Linux reads `/proc/self/stat`; other platforms return a typed absence.
#[must_use]
pub fn sample_process_cpu() -> CpuTimeObservation {
    #[cfg(target_os = "linux")]
    {
        let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
            return CpuTimeObservation::Unavailable {
                platform: "linux".to_owned(),
                reason: "/proc/self/stat unreadable".to_owned(),
            };
        };
        parse_stat_cpu_ticks(&stat).map_or_else(
            || CpuTimeObservation::Unavailable {
                platform: "linux".to_owned(),
                reason: "/proc/self/stat parse failure".to_owned(),
            },
            |reading| CpuTimeObservation::Observed {
                cpu_ns: ticks_to_ns(reading.ticks, clock_tick_hz()),
                seam: PROCESS_CPU_SEAM.to_owned(),
            },
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        CpuTimeObservation::Unavailable {
            platform: std::env::consts::OS.to_owned(),
            reason: "no /proc; per-process CPU collector not implemented on this platform"
                .to_owned(),
        }
    }
}

/// One raw per-thread reading from a census pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawThreadReading {
    /// Kernel thread id.
    pub tid: u64,
    /// Thread name from `comm`.
    pub name: String,
    /// Cumulative CPU ticks at sample time.
    pub ticks: u64,
}

/// Raw census pass: every live thread's tid, name, and cumulative ticks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawCensus {
    /// Live threads at sample time (possibly truncated).
    Sampled {
        /// Threads retained (bounded by [`MAX_CENSUS_THREADS`]).
        threads: Vec<RawThreadReading>,
        /// Total live threads observed before truncation.
        observed: u64,
        /// `_SC_CLK_TCK`.
        clock_tick_hz: u64,
    },
    /// No collector exists on this platform.
    Unavailable {
        /// `std::env::consts::OS` at sample time.
        platform: String,
        /// Human-readable absence reason.
        reason: String,
    },
}

/// Sample every live thread of this process (Linux `/proc/self/task`).
///
/// Non-Linux platforms return a typed absence — never an empty "success".
#[must_use]
pub fn sample_raw_census() -> RawCensus {
    #[cfg(target_os = "linux")]
    {
        let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
            return RawCensus::Unavailable {
                platform: "linux".to_owned(),
                reason: "/proc/self/task unreadable".to_owned(),
            };
        };
        let mut threads = Vec::new();
        let mut observed = 0_u64;
        for entry in entries.flatten() {
            let Some(tid) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
            else {
                continue;
            };
            observed += 1;
            if threads.len() >= MAX_CENSUS_THREADS {
                continue;
            }
            let base = entry.path();
            // A thread may exit between readdir and the reads below; skip it
            // rather than fabricating a zero sample.
            let Ok(comm) = std::fs::read_to_string(base.join("comm")) else {
                continue;
            };
            let Ok(stat) = std::fs::read_to_string(base.join("stat")) else {
                continue;
            };
            let Some(reading) = parse_stat_cpu_ticks(&stat) else {
                continue;
            };
            threads.push(RawThreadReading {
                tid,
                name: comm.trim_end_matches('\n').to_owned(),
                ticks: reading.ticks,
            });
        }
        RawCensus::Sampled {
            threads,
            observed,
            clock_tick_hz: clock_tick_hz(),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        RawCensus::Unavailable {
            platform: std::env::consts::OS.to_owned(),
            reason: "no /proc; per-thread role census not implemented on this platform".to_owned(),
        }
    }
}

/// Current kernel thread id of the calling thread (Linux), used to mark the
/// bench caller in the census.
#[must_use]
pub fn current_tid() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        u64::try_from(rustix::thread::gettid().as_raw_nonzero().get()).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Resolve a raw census pass against a baseline into a role census with
/// per-thread in-window CPU deltas.
///
/// Threads absent from the baseline are treated as born inside the window
/// (baseline zero). Threads that exited before this pass are structurally
/// invisible here; their CPU surfaces in the process-level unattributed
/// residual, and the caveat says so.
#[must_use]
pub fn resolve_role_census(
    raw: &RawCensus,
    baseline: &RawCensus,
    caller_tid: Option<u64>,
) -> RoleCensus {
    match raw {
        RawCensus::Unavailable { platform, reason } => RoleCensus::Unavailable {
            platform: platform.clone(),
            reason: reason.clone(),
        },
        RawCensus::Sampled {
            threads,
            observed,
            clock_tick_hz,
        } => {
            let baseline_ticks: BTreeMap<u64, u64> = match baseline {
                RawCensus::Sampled { threads, .. } => threads
                    .iter()
                    .map(|thread| (thread.tid, thread.ticks))
                    .collect(),
                RawCensus::Unavailable { .. } => BTreeMap::new(),
            };
            let tick_ns = ticks_to_ns(1, *clock_tick_hz);
            let truncated = *observed > threads.len() as u64;
            let mut caveats = vec![
                CensusCaveat::ClockTickGranularity { tick_ns },
                CensusCaveat::ExitedThreadCpuUnattributed,
            ];
            if truncated {
                caveats.push(CensusCaveat::TruncatedThreadList {
                    observed: *observed,
                    retained: threads.len() as u64,
                });
            }
            let threads = threads
                .iter()
                .map(|thread| {
                    let baseline = baseline_ticks.get(&thread.tid).copied().unwrap_or(0);
                    ThreadRoleSample {
                        tid: thread.tid,
                        name: thread.name.clone(),
                        role: classify_thread_role(&thread.name, caller_tid == Some(thread.tid)),
                        window_cpu_ns: ticks_to_ns(
                            thread.ticks.saturating_sub(baseline),
                            *clock_tick_hz,
                        ),
                    }
                })
                .collect();
            RoleCensus::Sampled(RoleCensusSample {
                platform: "linux".to_owned(),
                seam: ROLE_CENSUS_SEAM.to_owned(),
                clock_tick_hz: *clock_tick_hz,
                threads,
                truncated,
                caveats,
            })
        }
    }
}

// ─── Observed quantities ────────────────────────────────────────────────────

/// Queue occupancy for one arm: observed, absent by design, or a typed gap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueObservation {
    /// A real depth observation from a named seam.
    Observed {
        high_water_mark: u64,
        terminal_depth: u64,
        seam: String,
    },
    /// The arm has no queue at this seam (synchronous hand-off).
    SynchronousNoQueue { seam: String },
    /// A queue exists but no seam exposes its depth without patching the
    /// engine; `seam` names the exact boundary that lacks the observation.
    StructurallyUnobservable { seam: String },
}

/// Observed worker-pool width, or a typed absence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WidthObservation {
    /// Width read from a named engine seam.
    Observed { threads: u64, seam: String },
    /// No seam reports the width on this platform/arm.
    Unavailable { platform: String, reason: String },
}

/// The active-concurrency integral over the window.
///
/// `∫ active_threads(t) dt` over the window equals the process CPU time
/// consumed in the window — an identity, not a model — so the integral is
/// only ever *derived from* an observed CPU total, and is typed unavailable
/// exactly when that observation is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveConcurrencyIntegral {
    /// Derived from the observed process CPU delta.
    DerivedFromProcessCpuIdentity {
        /// Thread-nanoseconds: the integral value.
        integral_thread_ns: u64,
        /// `integral * 1000 / wall`, fixed-point mean active threads.
        mean_active_millithreads: u64,
    },
    /// The underlying CPU observation is unavailable on this platform.
    Unavailable { platform: String, reason: String },
}

/// Concurrency receipt: configured capacity vs observed activity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcurrencyReceipt {
    /// Pool capacity the cell contract configured (never copied into any
    /// observed field).
    pub configured_threads: u64,
    /// Width observed from the engine seam.
    pub observed_width: WidthObservation,
    /// Window wall time in nanoseconds.
    pub wall_ns: u64,
    /// Process CPU delta over the window (includes exited threads).
    pub process_cpu: CpuTimeObservation,
    /// The active-concurrency integral (CPU-time identity).
    pub active_concurrency_integral: ActiveConcurrencyIntegral,
    /// In-window CPU nanoseconds per resolved role (live threads only).
    pub role_cpu_ns: BTreeMap<String, u64>,
    /// Process CPU not attributable to any censused live thread: exited
    /// threads, post-census join work, and tick-granularity skew.
    pub unattributed_cpu_ns: u64,
    /// The census the roles were resolved from: the union of the in-window
    /// passes (feed-complete and searchable-verified), one entry per thread
    /// id with its last-observed cumulative CPU. Two passes matter because
    /// Tantivy respawns its indexing workers at every commit — a single
    /// terminal pass would fold most ingest CPU into the residual.
    pub live_thread_census: RoleCensus,
    /// Sum of the individually timed measured calls (legacy view); must fit
    /// inside the window wall.
    pub measured_call_sum_ns: u64,
}

/// Doc and byte counters observed at the harness↔engine boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkCounters {
    /// Feed calls issued inside the window.
    pub feed_calls: u64,
    /// Documents handed to the engine (tallied before each feed call).
    pub accepted_docs: u64,
    /// Documents the engine acknowledged (tallied after each successful call).
    pub processed_docs: u64,
    /// Engine-reported document count after the terminal commit.
    pub committed_docs: u64,
    /// Engine-reported document count at the terminal searchable probe.
    pub searchable_docs: u64,
    /// Bytes observed at hand-off, summed over the actual documents fed.
    pub accepted_bytes: u64,
    /// Bytes observed after each successful feed call returned.
    pub processed_bytes: u64,
    /// Corpus-byte total at the committed stage: engines do not retain fed
    /// byte totals, so this stays a typed gap unless a real seam appears.
    pub committed_corpus_bytes: ByteStageObservation,
    /// Corpus-byte total at the searchable stage (same seam situation).
    pub searchable_corpus_bytes: ByteStageObservation,
    /// Visibility-parity commits inside the window, excluding the terminal.
    pub periodic_commits: u64,
    /// Terminal commit calls (must be exactly 1).
    pub terminal_commit_calls: u64,
}

/// Engine-reported byte total read from a named seam, or a typed gap.
///
/// This is a *different quantity* from corpus bytes: it is what the engine
/// says its index occupies, and it is never compared against the fed-byte
/// denominator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineByteObservation {
    /// Byte total the engine reported through a named accessor.
    Observed { bytes: u64, seam: String },
    /// No accessor exposes the total without patching the engine.
    StructurallyUnobservable { seam: String },
}

/// Engine-reported index footprint at the terminal committed state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexFootprint {
    /// Serialized index bytes the engine reports after the terminal commit.
    pub committed_index_bytes: EngineByteObservation,
    /// Searchable segment count the engine reports after the terminal
    /// commit, with the seam it came from.
    pub committed_segment_count: EngineByteObservation,
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

/// One observed phase boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseSample {
    /// The boundary crossed.
    pub phase: LifecyclePhase,
    /// Window-relative elapsed nanoseconds at the boundary.
    pub window_elapsed_ns: u64,
}

/// Engine-specific terminal join attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalJoin {
    /// Quill's terminal commit is synchronous: when it returned there was no
    /// background worker left to join.
    QuillSynchronousCommit,
    /// Tantivy's benchmark join fence ran: operation channel closed, every
    /// indexing worker joined, merging thread completed.
    TantivyWorkersJoined {
        join_elapsed_ns: u64,
        searchable_segments_before: u64,
        searchable_segments_after: u64,
        writer_rearmed: bool,
    },
}

impl TerminalJoin {
    /// Stable label for logs and error messages.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::QuillSynchronousCommit => "quill_synchronous_commit",
            Self::TantivyWorkersJoined { .. } => "tantivy_workers_joined",
        }
    }
}

/// Terminal lifecycle state of one window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalLifecycle {
    /// Engine-specific join attestation.
    pub join: TerminalJoin,
    /// Whether the feed loop consumed every prepared batch.
    pub drained: bool,
    /// Whether no accepted document is still pending at terminal state.
    pub pending_docs_zero: bool,
    /// Bounded terminal reason (`completed` on the success path).
    pub terminal_reason: String,
    /// Retry predicate: whether re-running could change the outcome. A
    /// completed window is never retryable.
    pub retryable: bool,
}

// ─── Binding and overhead ───────────────────────────────────────────────────

/// Identity binding: process tree, machine, source, executable, corpus, and
/// run window for one receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptBinding {
    /// Run id of the invocation.
    pub run_id: String,
    /// Process id the window executed in.
    pub pid: u32,
    /// Parent process id (process-tree binding).
    pub parent_pid: u32,
    /// Machine fingerprint of the host.
    pub machine_fingerprint: String,
    /// Build profile label.
    pub build_profile: String,
    /// SHA-256 of the running bench executable.
    pub executable_sha256: String,
    /// Git revision of the source tree.
    pub git_rev: String,
    /// Gate label.
    pub gate: String,
    /// Cell fixture.
    pub fixture: String,
    /// Cell metric.
    pub metric: String,
    /// Engine arm (`quill` or `tantivy`).
    pub engine: String,
    /// Timing mode of the invocation (`per-call` or `continuous`).
    pub timing_mode: String,
    /// Prepared-corpus identity the window fed from.
    pub corpus_identity: String,
    /// SHA-256 of the prepared corpus manifest.
    pub corpus_manifest_sha256: String,
    /// Window start, nanoseconds relative to the experiment origin.
    pub window_started_ns: u64,
    /// Window end, nanoseconds relative to the experiment origin.
    pub window_ended_ns: u64,
}

/// Collection-overhead calibration for one receipt.
///
/// Receipt collection samples `/proc` at phase boundaries. The cost of one
/// census pass is measured at collector construction (outside any window)
/// and the number of in-window samples is bounded and recorded; both arms
/// sample at identical boundaries, so the overhead is symmetric by
/// construction rather than subtracted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionOverhead {
    /// Median nanoseconds of one census pass, measured at construction.
    pub calibrated_census_sample_ns: u64,
    /// Census passes taken inside the timed window.
    pub census_samples_in_window: u64,
    /// Both arms sample at the same phase boundaries.
    pub bounded_symmetric: bool,
}

// ─── The receipt ────────────────────────────────────────────────────────────

/// Actual-work, queue, worker-role, and lifecycle receipt for one window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkReceipt {
    /// Schema tag ([`WORK_RECEIPT_SCHEMA_VERSION`]).
    pub schema_version: String,
    /// Identity binding.
    pub binding: ReceiptBinding,
    /// Doc/byte counters observed at the harness↔engine boundary.
    pub counters: WorkCounters,
    /// Engine-reported index footprint.
    pub footprint: IndexFootprint,
    /// Queue occupancy observation.
    pub queue: QueueObservation,
    /// Configured-vs-observed concurrency and role-resolved CPU.
    pub concurrency: ConcurrencyReceipt,
    /// Observed phase boundaries (all five, in canonical order).
    pub phases: Vec<PhaseSample>,
    /// Terminal lifecycle attestation.
    pub terminal: TerminalLifecycle,
    /// Collection-overhead calibration.
    pub collection_overhead: CollectionOverhead,
    /// Canonical xxh3 self-digest (tamper evidence).
    pub receipt_xxh3: String,
}

/// Independently known quantities a receipt is validated against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkReceiptExpectation {
    /// Engine arm the harness ran.
    pub engine: String,
    /// Prepared-corpus document count.
    pub doc_count: u64,
    /// Prepared-corpus byte total.
    pub total_bytes: u64,
    /// Pool width the cell contract configured.
    pub configured_threads: u64,
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), WorkReceiptError> {
    if value.trim().is_empty() {
        return Err(WorkReceiptError::EmptyBindingField { field });
    }
    Ok(())
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), WorkReceiptError> {
    let well_formed = value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    if !well_formed {
        return Err(WorkReceiptError::MalformedSha256 {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn require_named_seam(seam: &str, field: &'static str) -> Result<(), WorkReceiptError> {
    if seam.trim().is_empty() {
        return Err(WorkReceiptError::UnnamedSeam { field });
    }
    Ok(())
}

impl WorkReceipt {
    /// Compute the canonical xxh3 self-digest (digest field zeroed).
    ///
    /// # Errors
    ///
    /// Returns [`WorkReceiptError::Serialization`] when canonical JSON
    /// encoding fails.
    pub fn compute_digest(&self) -> Result<String, WorkReceiptError> {
        let mut canonical = self.clone();
        canonical.receipt_xxh3 = String::new();
        let encoded =
            serde_json::to_vec(&canonical).map_err(|error| WorkReceiptError::Serialization {
                detail: error.to_string(),
            })?;
        let mut hasher = Xxh3::new();
        hasher.update(&encoded);
        Ok(format!("{:016x}", hasher.digest()))
    }

    /// Seal the receipt with its canonical self-digest.
    ///
    /// # Errors
    ///
    /// Returns [`WorkReceiptError::Serialization`] when canonical JSON
    /// encoding fails.
    pub fn seal(mut self) -> Result<Self, WorkReceiptError> {
        self.receipt_xxh3 = self.compute_digest()?;
        Ok(self)
    }

    /// Validate the receipt fail-closed against independently known
    /// quantities.
    ///
    /// Every acceptance clause that can be checked structurally is checked:
    /// binding completeness, work/byte equality, pipeline consistency,
    /// configured-as-actual substitution, impossible concurrency, missing
    /// roles, platform masquerades, phase order, unjoined work, terminal
    /// state, bounded collection, and tamper evidence.
    ///
    /// # Errors
    ///
    /// Returns the first [`WorkReceiptError`] violated.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self, expectation: &WorkReceiptExpectation) -> Result<(), WorkReceiptError> {
        if self.schema_version != WORK_RECEIPT_SCHEMA_VERSION {
            return Err(WorkReceiptError::SchemaVersionMismatch {
                expected: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
                actual: self.schema_version.clone(),
            });
        }
        self.validate_binding(expectation)?;
        self.validate_counters(expectation)?;
        self.validate_footprint_and_queue()?;
        self.validate_concurrency(expectation)?;
        self.validate_phases()?;
        self.validate_terminal()?;
        if self.collection_overhead.census_samples_in_window > MAX_CENSUS_SAMPLES_IN_WINDOW {
            return Err(WorkReceiptError::ExcessiveCensusSamples {
                samples: self.collection_overhead.census_samples_in_window,
                cap: MAX_CENSUS_SAMPLES_IN_WINDOW,
            });
        }
        let recomputed = self.compute_digest()?;
        if recomputed != self.receipt_xxh3 {
            return Err(WorkReceiptError::DigestMismatch {
                expected: self.receipt_xxh3.clone(),
                actual: recomputed,
            });
        }
        Ok(())
    }

    fn validate_binding(
        &self,
        expectation: &WorkReceiptExpectation,
    ) -> Result<(), WorkReceiptError> {
        let binding = &self.binding;
        require_nonempty(&binding.run_id, "run_id")?;
        require_nonempty(&binding.machine_fingerprint, "machine_fingerprint")?;
        require_nonempty(&binding.build_profile, "build_profile")?;
        require_nonempty(&binding.git_rev, "git_rev")?;
        require_nonempty(&binding.gate, "gate")?;
        require_nonempty(&binding.fixture, "fixture")?;
        require_nonempty(&binding.metric, "metric")?;
        require_nonempty(&binding.corpus_identity, "corpus_identity")?;
        require_sha256(&binding.executable_sha256, "executable_sha256")?;
        require_sha256(&binding.corpus_manifest_sha256, "corpus_manifest_sha256")?;
        if binding.pid == 0 {
            return Err(WorkReceiptError::EmptyBindingField { field: "pid" });
        }
        if binding.engine != "quill" && binding.engine != "tantivy" {
            return Err(WorkReceiptError::InvalidEngine {
                engine: binding.engine.clone(),
            });
        }
        if binding.engine != expectation.engine {
            return Err(WorkReceiptError::InvalidEngine {
                engine: binding.engine.clone(),
            });
        }
        if binding.timing_mode != "per-call" && binding.timing_mode != "continuous" {
            return Err(WorkReceiptError::InvalidTimingMode {
                value: binding.timing_mode.clone(),
            });
        }
        if binding.window_ended_ns <= binding.window_started_ns {
            return Err(WorkReceiptError::InvalidWindow {
                started_ns: binding.window_started_ns,
                ended_ns: binding.window_ended_ns,
            });
        }
        Ok(())
    }

    fn validate_counters(
        &self,
        expectation: &WorkReceiptExpectation,
    ) -> Result<(), WorkReceiptError> {
        let counters = &self.counters;
        if counters.feed_calls == 0 {
            return Err(WorkReceiptError::NoFeedCalls);
        }
        for (stage, actual) in [
            ("accepted_docs", counters.accepted_docs),
            ("processed_docs", counters.processed_docs),
            ("committed_docs", counters.committed_docs),
            ("searchable_docs", counters.searchable_docs),
        ] {
            if actual != expectation.doc_count {
                return Err(WorkReceiptError::CounterInequality {
                    stage,
                    expected: expectation.doc_count,
                    actual,
                });
            }
        }
        for (stage, actual) in [
            ("accepted_bytes", counters.accepted_bytes),
            ("processed_bytes", counters.processed_bytes),
        ] {
            if actual != expectation.total_bytes {
                return Err(WorkReceiptError::CounterInequality {
                    stage,
                    expected: expectation.total_bytes,
                    actual,
                });
            }
        }
        for (stage, observation) in [
            ("committed_corpus_bytes", &counters.committed_corpus_bytes),
            ("searchable_corpus_bytes", &counters.searchable_corpus_bytes),
        ] {
            match observation {
                ByteStageObservation::Observed(actual) => {
                    if *actual != expectation.total_bytes {
                        return Err(WorkReceiptError::CounterInequality {
                            stage,
                            expected: expectation.total_bytes,
                            actual: *actual,
                        });
                    }
                }
                ByteStageObservation::StructurallyUnobservable { seam } => {
                    require_named_seam(seam, "corpus byte stage")?;
                }
            }
        }
        if counters.terminal_commit_calls != 1 {
            return Err(WorkReceiptError::TerminalCommitCalls {
                actual: counters.terminal_commit_calls,
            });
        }
        Ok(())
    }

    fn validate_footprint_and_queue(&self) -> Result<(), WorkReceiptError> {
        for (field, observation) in [
            (
                "committed_index_bytes",
                &self.footprint.committed_index_bytes,
            ),
            (
                "committed_segment_count",
                &self.footprint.committed_segment_count,
            ),
        ] {
            match observation {
                EngineByteObservation::Observed { seam, .. }
                | EngineByteObservation::StructurallyUnobservable { seam } => {
                    require_named_seam(seam, field)?;
                }
            }
        }
        match &self.queue {
            QueueObservation::Observed { seam, .. }
            | QueueObservation::SynchronousNoQueue { seam }
            | QueueObservation::StructurallyUnobservable { seam } => {
                require_named_seam(seam, "queue")?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn validate_concurrency(
        &self,
        expectation: &WorkReceiptExpectation,
    ) -> Result<(), WorkReceiptError> {
        let concurrency = &self.concurrency;
        if concurrency.configured_threads == 0 {
            return Err(WorkReceiptError::ZeroConfiguredWidth);
        }
        if concurrency.configured_threads != expectation.configured_threads {
            return Err(WorkReceiptError::ConfiguredWidthMismatch {
                expected: expectation.configured_threads,
                actual: concurrency.configured_threads,
            });
        }
        match &concurrency.observed_width {
            WidthObservation::Observed { threads, seam } => {
                require_named_seam(seam, "observed_width")?;
                if *threads == 0 {
                    return Err(WorkReceiptError::FabricatedWidth { seam: seam.clone() });
                }
            }
            WidthObservation::Unavailable { platform, reason } => {
                require_nonempty(platform, "observed_width.platform")?;
                require_nonempty(reason, "observed_width.reason")?;
            }
        }
        let process_cpu_ns = match &concurrency.process_cpu {
            CpuTimeObservation::Observed { cpu_ns, seam } => {
                if seam != PROCESS_CPU_SEAM {
                    return Err(WorkReceiptError::UnknownSeam {
                        field: "process_cpu",
                        seam: seam.clone(),
                        expected: PROCESS_CPU_SEAM,
                    });
                }
                Some(*cpu_ns)
            }
            CpuTimeObservation::Unavailable { platform, reason } => {
                require_nonempty(platform, "process_cpu.platform")?;
                require_nonempty(reason, "process_cpu.reason")?;
                None
            }
        };
        match (&concurrency.active_concurrency_integral, process_cpu_ns) {
            (
                ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                    integral_thread_ns,
                    mean_active_millithreads,
                },
                Some(cpu_ns),
            ) => {
                if *integral_thread_ns != cpu_ns {
                    return Err(WorkReceiptError::IntegralIdentityViolation {
                        integral_thread_ns: *integral_thread_ns,
                        process_cpu_ns: cpu_ns,
                    });
                }
                let recomputed =
                    integral_thread_ns.saturating_mul(1000) / concurrency.wall_ns.max(1);
                if *mean_active_millithreads != recomputed {
                    return Err(WorkReceiptError::MeanConcurrencyMismatch {
                        stored_millithreads: *mean_active_millithreads,
                        recomputed_millithreads: recomputed,
                    });
                }
            }
            (ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity { .. }, None) => {
                return Err(WorkReceiptError::FabricatedIntegral);
            }
            (ActiveConcurrencyIntegral::Unavailable { platform, reason }, _) => {
                require_nonempty(platform, "integral.platform")?;
                require_nonempty(reason, "integral.reason")?;
            }
        }
        // Census structural checks and platform masquerade defense.
        let census_thread_count = match &concurrency.live_thread_census {
            RoleCensus::Sampled(sample) => {
                if sample.platform != "linux" {
                    return Err(WorkReceiptError::PlatformMasquerade {
                        field: "live_thread_census",
                        platform: sample.platform.clone(),
                        expected: "linux",
                    });
                }
                if sample.seam != ROLE_CENSUS_SEAM {
                    return Err(WorkReceiptError::UnknownSeam {
                        field: "live_thread_census",
                        seam: sample.seam.clone(),
                        expected: ROLE_CENSUS_SEAM,
                    });
                }
                if sample.threads.len() > MAX_CENSUS_THREADS {
                    return Err(WorkReceiptError::CensusThreadCapExceeded {
                        threads: sample.threads.len(),
                        cap: MAX_CENSUS_THREADS,
                    });
                }
                if sample.truncated
                    && !sample
                        .caveats
                        .iter()
                        .any(|caveat| matches!(caveat, CensusCaveat::TruncatedThreadList { .. }))
                {
                    return Err(WorkReceiptError::TruncationUnflagged);
                }
                Some(sample.threads.len() as u64)
            }
            RoleCensus::Unavailable { platform, reason } => {
                require_nonempty(platform, "census.platform")?;
                require_nonempty(reason, "census.reason")?;
                None
            }
        };
        // Role CPU requires a census; the decomposition must sum exactly.
        if census_thread_count.is_none() && !concurrency.role_cpu_ns.is_empty() {
            return Err(WorkReceiptError::RolesWithoutCensus);
        }
        for label in concurrency.role_cpu_ns.keys() {
            if WorkerRole::from_label(label).is_none() {
                return Err(WorkReceiptError::UnknownSeam {
                    field: "role_cpu_ns",
                    seam: label.clone(),
                    expected: "a WorkerRole label",
                });
            }
        }
        if let Some(cpu_ns) = process_cpu_ns {
            let role_sum: u64 = concurrency.role_cpu_ns.values().sum();
            if role_sum.saturating_add(concurrency.unattributed_cpu_ns) != cpu_ns {
                return Err(WorkReceiptError::RoleCpuIdentityViolation {
                    role_sum_ns: role_sum,
                    unattributed_ns: concurrency.unattributed_cpu_ns,
                    process_cpu_ns: cpu_ns,
                });
            }
            // Impossible-concurrency bound: the integral cannot exceed
            // (wall + one clock tick) multiplied by every plausibly live
            // thread — each thread's cumulative counter is tick-quantized,
            // so P threads can legitimately over-read by up to P ticks.
            let tick_ns = match &concurrency.live_thread_census {
                RoleCensus::Sampled(sample) => ticks_to_ns(1, sample.clock_tick_hz),
                RoleCensus::Unavailable { .. } => 10_000_000,
            };
            let plausible = census_thread_count
                .unwrap_or_else(|| {
                    let observed = match &concurrency.observed_width {
                        WidthObservation::Observed { threads, .. } => *threads,
                        WidthObservation::Unavailable { .. } => 0,
                    };
                    concurrency.configured_threads.saturating_add(observed)
                })
                .saturating_add(CENSUS_SLACK_THREADS);
            let bound = concurrency
                .wall_ns
                .saturating_add(tick_ns)
                .saturating_mul(plausible);
            if cpu_ns > bound {
                return Err(WorkReceiptError::ImpossibleConcurrency {
                    integral_thread_ns: cpu_ns,
                    wall_ns: concurrency.wall_ns,
                    plausible_threads: plausible,
                });
            }
        }
        // Role presence: the engines' worker families must actually appear.
        // Docstore compressor threads exit with their store writers and the
        // merge pool width is a Tantivy internal default, so those roles are
        // recorded when seen but never required.
        if let RoleCensus::Sampled(sample) = &concurrency.live_thread_census {
            let has_role = |role: WorkerRole| sample.threads.iter().any(|t| t.role == role);
            match self.binding.engine.as_str() {
                "tantivy" => {
                    if !has_role(WorkerRole::TantivyIndexWorker) {
                        return Err(WorkReceiptError::MissingRequiredRole {
                            engine: self.binding.engine.clone(),
                            role: "tantivy_index_worker",
                        });
                    }
                    if !has_role(WorkerRole::TantivySegmentUpdater) {
                        return Err(WorkReceiptError::MissingRequiredRole {
                            engine: self.binding.engine.clone(),
                            role: "tantivy_segment_updater",
                        });
                    }
                }
                "quill" => {
                    // Quill's writer is synchronous on the caller; QG-1 runs
                    // it inside a bench-named Rayon pool.
                    if !has_role(WorkerRole::RayonWorker) && !has_role(WorkerRole::BenchCaller) {
                        return Err(WorkReceiptError::MissingRequiredRole {
                            engine: self.binding.engine.clone(),
                            role: "rayon_worker|bench_caller",
                        });
                    }
                }
                _ => {}
            }
        }
        if concurrency.measured_call_sum_ns > concurrency.wall_ns {
            return Err(WorkReceiptError::MeasuredExceedsWall {
                measured_sum_ns: concurrency.measured_call_sum_ns,
                wall_ns: concurrency.wall_ns,
            });
        }
        Ok(())
    }

    fn validate_phases(&self) -> Result<(), WorkReceiptError> {
        if self.phases.len() != LifecyclePhase::ORDERED.len() {
            return Err(WorkReceiptError::PhaseCountMismatch {
                expected: LifecyclePhase::ORDERED.len(),
                actual: self.phases.len(),
            });
        }
        let mut previous_ns = 0_u64;
        for (sample, expected) in self.phases.iter().zip(LifecyclePhase::ORDERED) {
            if sample.phase != expected {
                return Err(WorkReceiptError::PhaseOrderViolation {
                    phase: expected.label(),
                });
            }
            if sample.window_elapsed_ns < previous_ns {
                return Err(WorkReceiptError::PhaseOrderViolation {
                    phase: sample.phase.label(),
                });
            }
            previous_ns = sample.window_elapsed_ns;
        }
        let quiescence_ns = previous_ns;
        if quiescence_ns != self.concurrency.wall_ns {
            return Err(WorkReceiptError::PhaseWindowMismatch {
                quiescence_ns,
                wall_ns: self.concurrency.wall_ns,
            });
        }
        let window_span = self
            .binding
            .window_ended_ns
            .saturating_sub(self.binding.window_started_ns);
        if window_span != self.concurrency.wall_ns {
            return Err(WorkReceiptError::PhaseWindowMismatch {
                quiescence_ns: window_span,
                wall_ns: self.concurrency.wall_ns,
            });
        }
        Ok(())
    }

    fn validate_terminal(&self) -> Result<(), WorkReceiptError> {
        let terminal = &self.terminal;
        let join_ok = matches!(
            (&self.binding.engine[..], &terminal.join),
            ("quill", TerminalJoin::QuillSynchronousCommit)
                | ("tantivy", TerminalJoin::TantivyWorkersJoined { .. })
        );
        if !join_ok {
            return Err(WorkReceiptError::JoinContractMismatch {
                engine: self.binding.engine.clone(),
                join: terminal.join.label(),
            });
        }
        if let TerminalJoin::TantivyWorkersJoined { writer_rearmed, .. } = &terminal.join {
            if *writer_rearmed {
                return Err(WorkReceiptError::WriterRearmedInTerminalJoin);
            }
        }
        if !terminal.drained {
            return Err(WorkReceiptError::UndrainedFeed);
        }
        if !terminal.pending_docs_zero {
            return Err(WorkReceiptError::PendingWorkAtTerminal {
                committed: self.counters.committed_docs,
                accepted: self.counters.accepted_docs,
            });
        }
        if terminal.terminal_reason.trim().is_empty() {
            return Err(WorkReceiptError::EmptyTerminalReason);
        }
        if terminal.terminal_reason == "completed" && terminal.retryable {
            return Err(WorkReceiptError::RetryablePredicateViolation {
                reason: terminal.terminal_reason.clone(),
            });
        }
        Ok(())
    }

    /// One bounded structured log line binding machine, profile, process
    /// roles, lifecycle, source, queue state, terminal reason, and retry
    /// predicate. Field values are truncated; no environment or secret
    /// material is included.
    #[must_use]
    pub fn bounded_log_line(&self) -> String {
        fn clip(value: &str, max: usize) -> &str {
            &value[..value.len().min(max)]
        }
        let role_counts = self.concurrency.live_thread_census.role_counts();
        let mut roles = String::new();
        for (role, count) in &role_counts {
            if !roles.is_empty() {
                roles.push(',');
            }
            roles.push_str(role.label());
            roles.push(':');
            roles.push_str(&count.to_string());
        }
        if roles.is_empty() {
            roles.push_str("unavailable");
        }
        let integral = match &self.concurrency.active_concurrency_integral {
            ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                mean_active_millithreads,
                ..
            } => format!("{mean_active_millithreads}m"),
            ActiveConcurrencyIntegral::Unavailable { .. } => "unavailable".to_owned(),
        };
        let queue = match &self.queue {
            QueueObservation::Observed {
                high_water_mark, ..
            } => format!("hwm:{high_water_mark}"),
            QueueObservation::SynchronousNoQueue { .. } => "sync_no_queue".to_owned(),
            QueueObservation::StructurallyUnobservable { .. } => "unobservable".to_owned(),
        };
        format!(
            "[qg1-work-receipt] run_id={} pid={} ppid={} machine={} profile={} elf={} git={} \
             gate={} fixture={} metric={} arm={} mode={} corpus={} wall_ns={} configured={} \
             roles={} mean_active={} queue={} join={} drained={} pending_zero={} terminal={} \
             retryable={} digest={}",
            clip(&self.binding.run_id, 64),
            self.binding.pid,
            self.binding.parent_pid,
            clip(&self.binding.machine_fingerprint, 16),
            clip(&self.binding.build_profile, 24),
            clip(&self.binding.executable_sha256, 16),
            clip(&self.binding.git_rev, 16),
            clip(&self.binding.gate, 8),
            clip(&self.binding.fixture, 48),
            clip(&self.binding.metric, 32),
            clip(&self.binding.engine, 8),
            clip(&self.binding.timing_mode, 12),
            clip(&self.binding.corpus_manifest_sha256, 16),
            self.concurrency.wall_ns,
            self.concurrency.configured_threads,
            roles,
            integral,
            queue,
            self.terminal.join.label(),
            self.terminal.drained,
            self.terminal.pending_docs_zero,
            clip(&self.terminal.terminal_reason, 48),
            self.terminal.retryable,
            clip(&self.receipt_xxh3, 16),
        )
    }
}

// ─── Collector ──────────────────────────────────────────────────────────────

/// Identity inputs for one collector (everything the receipt binds that the
/// harness knows before the window runs).
#[derive(Debug, Clone)]
pub struct CollectorBinding {
    pub run_id: String,
    pub machine_fingerprint: String,
    pub build_profile: String,
    pub executable_sha256: String,
    pub git_rev: String,
    pub gate: String,
    pub fixture: String,
    pub metric: String,
    pub engine: String,
    pub timing_mode: String,
    pub corpus_identity: String,
    pub corpus_manifest_sha256: String,
}

/// Live receipt collector for one window.
///
/// Implements [`LifecycleObserver`]: the H1 continuous runner and the
/// per-call bulk runner both drive it through the same seam. Counters are
/// recorded through the inherent methods by whichever loop owns the feed
/// boundary.
#[derive(Debug)]
pub struct WorkReceiptCollector {
    binding: CollectorBinding,
    configured_threads: u64,
    window_started: Option<Instant>,
    baseline_cpu: Option<CpuTimeObservation>,
    baseline_census: Option<RawCensus>,
    caller_tid: Option<u64>,
    /// Union of the in-window census passes: last-observed cumulative ticks
    /// per tid. Tantivy respawns indexing workers at every commit, so a
    /// single terminal pass would miss most ingest CPU.
    window_census: BTreeMap<u64, RawThreadReading>,
    window_census_observed_max: u64,
    window_census_clock_hz: u64,
    window_census_passes: u64,
    census_samples_in_window: u64,
    calibrated_census_sample_ns: u64,
    phases: Vec<PhaseSample>,
    feed_calls: u64,
    accepted_docs: u64,
    processed_docs: u64,
    accepted_bytes: u64,
    processed_bytes: u64,
    periodic_commits: u64,
    committed_docs: u64,
    searchable_docs: u64,
    terminal_commit_calls: u64,
    committed_index_bytes: EngineByteObservation,
    committed_segment_count: EngineByteObservation,
    queue: QueueObservation,
    observed_width: WidthObservation,
    measured_call_sum_ns: u64,
    terminal: Option<TerminalLifecycle>,
}

impl WorkReceiptCollector {
    /// Build a collector and calibrate the census-pass overhead (three timed
    /// passes outside any window; the median is recorded on the receipt).
    #[must_use]
    pub fn new(binding: CollectorBinding, configured_threads: u64) -> Self {
        let mut calibration = [0_u64; 3];
        for slot in &mut calibration {
            let started = Instant::now();
            let _ = sample_raw_census();
            *slot = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
        calibration.sort_unstable();
        Self {
            binding,
            configured_threads,
            window_started: None,
            baseline_cpu: None,
            baseline_census: None,
            caller_tid: current_tid(),
            window_census: BTreeMap::new(),
            window_census_observed_max: 0,
            window_census_clock_hz: 0,
            window_census_passes: 0,
            census_samples_in_window: 0,
            calibrated_census_sample_ns: calibration[1],
            phases: Vec::new(),
            feed_calls: 0,
            accepted_docs: 0,
            processed_docs: 0,
            accepted_bytes: 0,
            processed_bytes: 0,
            periodic_commits: 0,
            committed_docs: 0,
            searchable_docs: 0,
            terminal_commit_calls: 0,
            committed_index_bytes: EngineByteObservation::StructurallyUnobservable {
                seam: "engine footprint not yet recorded".to_owned(),
            },
            committed_segment_count: EngineByteObservation::StructurallyUnobservable {
                seam: "engine segment count not yet recorded".to_owned(),
            },
            queue: QueueObservation::StructurallyUnobservable {
                seam: "queue observation not yet recorded".to_owned(),
            },
            observed_width: WidthObservation::Unavailable {
                platform: std::env::consts::OS.to_owned(),
                reason: "width observation not yet recorded".to_owned(),
            },
            measured_call_sum_ns: 0,
            terminal: None,
        }
    }

    /// Take the pre-window baselines (process CPU, census, wall origin).
    ///
    /// Call immediately before the window's first feed; when driven purely
    /// through [`LifecycleObserver::on_phase`] (the H1 continuous runner),
    /// the baseline is taken lazily at `FirstFeed` instead and counted as an
    /// in-window census sample.
    pub fn begin_window(&mut self) {
        self.baseline_cpu = Some(sample_process_cpu());
        self.baseline_census = Some(sample_raw_census());
        self.window_started = Some(Instant::now());
    }

    /// Record one feed batch observed at the hand-off boundary.
    pub fn record_feed_batch(&mut self, documents: u64, bytes: u64) {
        self.feed_calls += 1;
        self.accepted_docs += documents;
        self.accepted_bytes += bytes;
        // The harness records the batch after the call returned; a failed
        // call panics the bench, so processed == accepted on every live path.
        self.processed_docs += documents;
        self.processed_bytes += bytes;
    }

    /// Record a visibility-parity commit inside the window.
    pub fn record_periodic_commit(&mut self) {
        self.periodic_commits += 1;
    }

    /// Record the engine-reported committed document count and footprint
    /// after the terminal commit.
    pub fn record_committed(
        &mut self,
        committed_docs: u64,
        index_bytes: EngineByteObservation,
        segment_count: EngineByteObservation,
    ) {
        self.committed_docs = committed_docs;
        self.terminal_commit_calls += 1;
        self.committed_index_bytes = index_bytes;
        self.committed_segment_count = segment_count;
    }

    /// Record the engine-reported searchable document count at the terminal
    /// probe.
    pub fn record_searchable(&mut self, searchable_docs: u64) {
        self.searchable_docs = searchable_docs;
    }

    /// Record the queue observation for this arm.
    pub fn record_queue(&mut self, queue: QueueObservation) {
        self.queue = queue;
    }

    /// Record the observed worker-pool width for this arm.
    pub fn record_width(&mut self, width: WidthObservation) {
        self.observed_width = width;
    }

    /// Record the summed individually timed measured calls (legacy view).
    pub fn record_measured_sum_ns(&mut self, measured_sum_ns: u64) {
        self.measured_call_sum_ns = measured_sum_ns;
    }

    /// Record the terminal lifecycle attestation.
    pub fn record_terminal(&mut self, join: TerminalJoin, terminal_reason: &str, retryable: bool) {
        let drained = self.feed_calls > 0;
        let pending_docs_zero = self.committed_docs == self.accepted_docs;
        self.terminal = Some(TerminalLifecycle {
            join,
            drained,
            pending_docs_zero,
            terminal_reason: terminal_reason.to_owned(),
            retryable,
        });
    }

    /// Number of census passes taken inside the window so far.
    #[must_use]
    pub const fn census_samples_in_window(&self) -> u64 {
        self.census_samples_in_window
    }

    /// Take one in-window census pass and merge it into the window union
    /// (last-observed cumulative ticks per tid).
    fn take_window_census_pass(&mut self) {
        self.census_samples_in_window += 1;
        if let RawCensus::Sampled {
            threads,
            observed,
            clock_tick_hz,
        } = sample_raw_census()
        {
            self.window_census_passes += 1;
            self.window_census_observed_max = self.window_census_observed_max.max(observed);
            self.window_census_clock_hz = clock_tick_hz;
            for reading in threads {
                self.window_census.insert(reading.tid, reading);
            }
        }
    }

    /// Assemble and seal the receipt.
    ///
    /// `window_started_rel_origin_ns` anchors the window to the experiment
    /// origin the raw samples use.
    ///
    /// # Errors
    ///
    /// Returns [`WorkReceiptError::Serialization`] when sealing fails, or a
    /// structural error when the collector was driven out of order (no
    /// baseline, no terminal attestation).
    pub fn finish(
        self,
        window_started_rel_origin_ns: u64,
    ) -> Result<WorkReceipt, WorkReceiptError> {
        let window_started = self
            .window_started
            .ok_or(WorkReceiptError::PhaseOrderViolation {
                phase: "first_feed",
            })?;
        let wall_ns = self
            .phases
            .iter()
            .find(|sample| sample.phase == LifecyclePhase::QuiescenceJoined)
            .map_or_else(
                || u64::try_from(window_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                |sample| sample.window_elapsed_ns,
            );
        let terminal = self.terminal.ok_or(WorkReceiptError::PhaseOrderViolation {
            phase: "quiescence_joined",
        })?;
        // Terminal process CPU: read after the join, so merge/join work is
        // inside the delta.
        let terminal_cpu = sample_process_cpu();
        let baseline_cpu = self.baseline_cpu.unwrap_or_else(sample_process_cpu);
        let (process_cpu, integral) = match (&baseline_cpu, &terminal_cpu) {
            (
                CpuTimeObservation::Observed {
                    cpu_ns: baseline_ns,
                    ..
                },
                CpuTimeObservation::Observed {
                    cpu_ns: terminal_ns,
                    seam,
                },
            ) => {
                let delta = terminal_ns.saturating_sub(*baseline_ns);
                (
                    CpuTimeObservation::Observed {
                        cpu_ns: delta,
                        seam: seam.clone(),
                    },
                    ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                        integral_thread_ns: delta,
                        mean_active_millithreads: delta.saturating_mul(1000) / wall_ns.max(1),
                    },
                )
            }
            (_, CpuTimeObservation::Unavailable { platform, reason })
            | (CpuTimeObservation::Unavailable { platform, reason }, _) => (
                CpuTimeObservation::Unavailable {
                    platform: platform.clone(),
                    reason: reason.clone(),
                },
                ActiveConcurrencyIntegral::Unavailable {
                    platform: platform.clone(),
                    reason: reason.clone(),
                },
            ),
        };
        let baseline_census = self
            .baseline_census
            .unwrap_or_else(|| RawCensus::Unavailable {
                platform: std::env::consts::OS.to_owned(),
                reason: "collector finished before any baseline census".to_owned(),
            });
        // Assemble the merged window census: the union of every in-window
        // pass, one entry per tid with its last-observed cumulative ticks.
        let merged_raw = if self.window_census_passes == 0 {
            // Driven without any in-window pass (should not happen through
            // the observer, which samples at FeedComplete): take one now
            // rather than fabricating an empty census.
            sample_raw_census()
        } else {
            RawCensus::Sampled {
                threads: self.window_census.into_values().collect(),
                observed: self.window_census_observed_max,
                clock_tick_hz: self.window_census_clock_hz,
            }
        };
        let census = resolve_role_census(&merged_raw, &baseline_census, self.caller_tid);
        let mut role_cpu_ns: BTreeMap<String, u64> = BTreeMap::new();
        if let RoleCensus::Sampled(sample) = &census {
            for thread in &sample.threads {
                *role_cpu_ns
                    .entry(thread.role.label().to_owned())
                    .or_insert(0) += thread.window_cpu_ns;
            }
        }
        let unattributed_cpu_ns = match &process_cpu {
            CpuTimeObservation::Observed { cpu_ns, .. } => {
                let role_sum: u64 = role_cpu_ns.values().sum();
                cpu_ns.saturating_sub(role_sum)
            }
            CpuTimeObservation::Unavailable { .. } => 0,
        };
        // Census reads and the process-stat read are not atomic, so under
        // tick-granularity skew the per-thread sum can exceed the process
        // delta by a tick or two. Scaling roles down would fabricate values
        // nobody observed; instead the process total is floored at the
        // observed role sum (both readings come from the same kernel
        // counters) so the decomposition stays an exact identity over
        // observed numbers.
        let process_cpu = match process_cpu {
            CpuTimeObservation::Observed { cpu_ns, seam } => {
                let role_sum: u64 = role_cpu_ns.values().sum();
                CpuTimeObservation::Observed {
                    cpu_ns: cpu_ns.max(role_sum),
                    seam,
                }
            }
            unavailable @ CpuTimeObservation::Unavailable { .. } => unavailable,
        };
        let integral = match (&process_cpu, integral) {
            (
                CpuTimeObservation::Observed { cpu_ns, .. },
                ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity { .. },
            ) => ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                integral_thread_ns: *cpu_ns,
                mean_active_millithreads: cpu_ns.saturating_mul(1000) / wall_ns.max(1),
            },
            (_, other) => other,
        };
        let receipt = WorkReceipt {
            schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
            binding: ReceiptBinding {
                run_id: self.binding.run_id,
                pid: std::process::id(),
                parent_pid: parent_pid(),
                machine_fingerprint: self.binding.machine_fingerprint,
                build_profile: self.binding.build_profile,
                executable_sha256: self.binding.executable_sha256,
                git_rev: self.binding.git_rev,
                gate: self.binding.gate,
                fixture: self.binding.fixture,
                metric: self.binding.metric,
                engine: self.binding.engine,
                timing_mode: self.binding.timing_mode,
                corpus_identity: self.binding.corpus_identity,
                corpus_manifest_sha256: self.binding.corpus_manifest_sha256,
                window_started_ns: window_started_rel_origin_ns,
                window_ended_ns: window_started_rel_origin_ns.saturating_add(wall_ns),
            },
            counters: WorkCounters {
                feed_calls: self.feed_calls,
                accepted_docs: self.accepted_docs,
                processed_docs: self.processed_docs,
                committed_docs: self.committed_docs,
                searchable_docs: self.searchable_docs,
                accepted_bytes: self.accepted_bytes,
                processed_bytes: self.processed_bytes,
                committed_corpus_bytes: corpus_byte_stage_gap(),
                searchable_corpus_bytes: corpus_byte_stage_gap(),
                periodic_commits: self.periodic_commits,
                terminal_commit_calls: self.terminal_commit_calls,
            },
            footprint: IndexFootprint {
                committed_index_bytes: self.committed_index_bytes,
                committed_segment_count: self.committed_segment_count,
            },
            queue: self.queue,
            concurrency: ConcurrencyReceipt {
                configured_threads: self.configured_threads,
                observed_width: self.observed_width,
                wall_ns,
                process_cpu,
                active_concurrency_integral: integral,
                role_cpu_ns,
                unattributed_cpu_ns,
                live_thread_census: census,
                measured_call_sum_ns: self.measured_call_sum_ns.min(wall_ns),
            },
            phases: self.phases,
            terminal,
            collection_overhead: CollectionOverhead {
                calibrated_census_sample_ns: self.calibrated_census_sample_ns,
                census_samples_in_window: self.census_samples_in_window,
                bounded_symmetric: true,
            },
            receipt_xxh3: String::new(),
        };
        receipt.seal()
    }
}

impl LifecycleObserver for WorkReceiptCollector {
    fn on_feed_batch(&mut self, documents: u64, bytes: u64) {
        self.record_feed_batch(documents, bytes);
    }

    fn on_phase(&mut self, phase: LifecyclePhase, window_elapsed_ns: u64) {
        match phase {
            LifecyclePhase::FirstFeed => {
                if self.window_started.is_none() {
                    // Driven purely through the seam (H1 continuous runner):
                    // take the baseline now, inside the window, and count it.
                    self.begin_window();
                    self.census_samples_in_window += 1;
                }
            }
            LifecyclePhase::FeedComplete | LifecyclePhase::SearchableVerified => {
                // Two in-window passes: at feed-complete the current
                // generation of indexing workers is still alive with its
                // ingest CPU on the clock, and at searchable-verified the
                // post-commit generation (plus updater/merge threads) is
                // live before the quiescence join tears anything down.
                self.take_window_census_pass();
            }
            LifecyclePhase::CommitComplete | LifecyclePhase::QuiescenceJoined => {}
        }
        self.phases.push(PhaseSample {
            phase,
            window_elapsed_ns,
        });
    }
}

/// The typed corpus-byte gap both terminal stages carry today.
///
/// Engines do not retain fed-byte totals: Quill's `LexicalRead` and
/// `SegmentStatsProvider` report document counts and serialized segment
/// sizes, Tantivy's benchmark seams report segment counts and serialized
/// layout bytes — none of them can re-state the *corpus* bytes that reached
/// the committed or searchable stage, and copying the manifest denominator
/// would be fabrication. The engine-reported serialized footprint rides
/// [`IndexFootprint`] instead.
#[must_use]
pub fn corpus_byte_stage_gap() -> ByteStageObservation {
    ByteStageObservation::StructurallyUnobservable {
        seam: "no engine seam retains fed corpus bytes; LexicalRead exposes doc_count and \
               segment stats expose serialized sizes, a different quantity recorded under \
               index footprint"
            .to_owned(),
    }
}

fn parent_pid() -> u32 {
    #[cfg(target_os = "linux")]
    {
        u32::try_from(rustix::process::getppid().map_or(0, |pid| pid.as_raw_nonzero().get()))
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

// ─── Artifact evidence containers ───────────────────────────────────────────

/// Bounded per-cell work-receipt evidence: round counts plus the final
/// measurement-round receipt per arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkReceiptCellEvidence {
    pub schema_version: String,
    pub fixture: String,
    pub rounds_quill: u64,
    pub rounds_tantivy: u64,
    pub last_quill_receipt: Option<WorkReceipt>,
    pub last_tantivy_receipt: Option<WorkReceipt>,
    /// True only when every receipt in the run validated fail-closed.
    pub all_receipts_validated: bool,
}

/// Additive per-gate work-receipt evidence block.
///
/// Absent (and never serialized) when receipts are off, so the legacy
/// artifact byte shape is unchanged until a run opts in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkReceiptEvidence {
    pub schema_version: String,
    pub mode: String,
    pub cells: Vec<WorkReceiptCellEvidence>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> ReceiptBinding {
        ReceiptBinding {
            run_id: "work-receipt-test".to_owned(),
            pid: 4242,
            parent_pid: 1,
            machine_fingerprint: "machine-test".to_owned(),
            build_profile: "dev".to_owned(),
            executable_sha256: "e".repeat(64),
            git_rev: "abcdef012345".to_owned(),
            gate: "QG-1".to_owned(),
            fixture: "bulk/tiny/1/positions_on".to_owned(),
            metric: "docs_per_second".to_owned(),
            engine: "quill".to_owned(),
            timing_mode: "per-call".to_owned(),
            corpus_identity: "qg1-native/prepared-prefix-v1/test".to_owned(),
            corpus_manifest_sha256: "c".repeat(64),
            window_started_ns: 1_000,
            window_ended_ns: 8_000,
        }
    }

    fn census(engine: &str) -> RoleCensus {
        let mut threads = vec![
            ThreadRoleSample {
                tid: 100,
                name: "perf_matrix".to_owned(),
                role: WorkerRole::BenchCaller,
                window_cpu_ns: 2_000,
            },
            ThreadRoleSample {
                tid: 101,
                name: "qg-rayon-0".to_owned(),
                role: WorkerRole::RayonWorker,
                window_cpu_ns: 1_000,
            },
        ];
        if engine == "tantivy" {
            threads.push(ThreadRoleSample {
                tid: 102,
                name: "thrd-tantivy-in".to_owned(),
                role: WorkerRole::TantivyIndexWorker,
                window_cpu_ns: 1_500,
            });
            threads.push(ThreadRoleSample {
                tid: 103,
                name: "segment_updater".to_owned(),
                role: WorkerRole::TantivySegmentUpdater,
                window_cpu_ns: 300,
            });
            threads.push(ThreadRoleSample {
                tid: 104,
                name: "merge_thread_0".to_owned(),
                role: WorkerRole::TantivyMergeWorker,
                window_cpu_ns: 200,
            });
        }
        RoleCensus::Sampled(RoleCensusSample {
            platform: "linux".to_owned(),
            seam: ROLE_CENSUS_SEAM.to_owned(),
            clock_tick_hz: 100,
            threads,
            truncated: false,
            caveats: vec![
                CensusCaveat::ClockTickGranularity {
                    tick_ns: 10_000_000,
                },
                CensusCaveat::ExitedThreadCpuUnattributed,
            ],
        })
    }

    fn expectation(engine: &str) -> WorkReceiptExpectation {
        WorkReceiptExpectation {
            engine: engine.to_owned(),
            doc_count: 12,
            total_bytes: 4_096,
            configured_threads: 1,
        }
    }

    fn good_receipt(engine: &str) -> WorkReceipt {
        let census = census(engine);
        let mut role_cpu_ns = BTreeMap::new();
        if let RoleCensus::Sampled(sample) = &census {
            for thread in &sample.threads {
                *role_cpu_ns
                    .entry(thread.role.label().to_owned())
                    .or_insert(0_u64) += thread.window_cpu_ns;
            }
        }
        let role_sum: u64 = role_cpu_ns.values().sum();
        let process_cpu_ns = role_sum + 500;
        let wall_ns = 7_000_u64;
        let mut binding = binding();
        binding.engine = engine.to_owned();
        let receipt = WorkReceipt {
            schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
            binding,
            counters: WorkCounters {
                feed_calls: 2,
                accepted_docs: 12,
                processed_docs: 12,
                committed_docs: 12,
                searchable_docs: 12,
                accepted_bytes: 4_096,
                processed_bytes: 4_096,
                committed_corpus_bytes: corpus_byte_stage_gap(),
                searchable_corpus_bytes: corpus_byte_stage_gap(),
                periodic_commits: 0,
                terminal_commit_calls: 1,
            },
            footprint: IndexFootprint {
                committed_index_bytes: EngineByteObservation::Observed {
                    bytes: 65_536,
                    seam: "test seam: engine layout accessor".to_owned(),
                },
                committed_segment_count: EngineByteObservation::Observed {
                    bytes: 1,
                    seam: "test seam: engine segment accessor".to_owned(),
                },
            },
            queue: if engine == "quill" {
                QueueObservation::SynchronousNoQueue {
                    seam: "test seam: synchronous writer".to_owned(),
                }
            } else {
                QueueObservation::StructurallyUnobservable {
                    seam: "test seam: internal channel exposes no depth".to_owned(),
                }
            },
            concurrency: ConcurrencyReceipt {
                configured_threads: 1,
                observed_width: WidthObservation::Observed {
                    threads: 1,
                    seam: "test seam: pool accessor".to_owned(),
                },
                wall_ns,
                process_cpu: CpuTimeObservation::Observed {
                    cpu_ns: process_cpu_ns,
                    seam: PROCESS_CPU_SEAM.to_owned(),
                },
                active_concurrency_integral:
                    ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                        integral_thread_ns: process_cpu_ns,
                        mean_active_millithreads: process_cpu_ns * 1000 / wall_ns,
                    },
                role_cpu_ns,
                unattributed_cpu_ns: 500,
                live_thread_census: census,
                measured_call_sum_ns: 5_000,
            },
            phases: vec![
                PhaseSample {
                    phase: LifecyclePhase::FirstFeed,
                    window_elapsed_ns: 0,
                },
                PhaseSample {
                    phase: LifecyclePhase::FeedComplete,
                    window_elapsed_ns: 3_000,
                },
                PhaseSample {
                    phase: LifecyclePhase::CommitComplete,
                    window_elapsed_ns: 5_000,
                },
                PhaseSample {
                    phase: LifecyclePhase::SearchableVerified,
                    window_elapsed_ns: 6_000,
                },
                PhaseSample {
                    phase: LifecyclePhase::QuiescenceJoined,
                    window_elapsed_ns: 7_000,
                },
            ],
            terminal: TerminalLifecycle {
                join: if engine == "quill" {
                    TerminalJoin::QuillSynchronousCommit
                } else {
                    TerminalJoin::TantivyWorkersJoined {
                        join_elapsed_ns: 900,
                        searchable_segments_before: 3,
                        searchable_segments_after: 3,
                        writer_rearmed: false,
                    }
                },
                drained: true,
                pending_docs_zero: true,
                terminal_reason: "completed".to_owned(),
                retryable: false,
            },
            collection_overhead: CollectionOverhead {
                calibrated_census_sample_ns: 120_000,
                census_samples_in_window: 1,
                bounded_symmetric: true,
            },
            receipt_xxh3: String::new(),
        };
        receipt.seal().expect("seal test receipt")
    }

    #[test]
    fn valid_receipts_pass_for_both_engines() {
        for engine in ["quill", "tantivy"] {
            good_receipt(engine)
                .validate(&expectation(engine))
                .unwrap_or_else(|error| panic!("{engine} receipt must validate: {error}"));
        }
    }

    #[test]
    fn work_receipt_mode_parses_fail_closed() {
        assert_eq!(WorkReceiptMode::parse(None), Ok(WorkReceiptMode::Off));
        assert_eq!(
            WorkReceiptMode::parse(Some("off")),
            Ok(WorkReceiptMode::Off)
        );
        assert_eq!(WorkReceiptMode::parse(Some("on")), Ok(WorkReceiptMode::On));
        assert!(matches!(
            WorkReceiptMode::parse(Some("yes")),
            Err(WorkReceiptError::InvalidWorkReceiptMode { .. })
        ));
        assert!(WorkReceiptMode::On.is_enabled());
        assert!(!WorkReceiptMode::Off.is_enabled());
    }

    #[test]
    fn tampering_any_field_breaks_the_digest() {
        let mut receipt = good_receipt("quill");
        receipt.counters.accepted_docs += 1;
        // The mutation also breaks the doc-count equality, so check digest
        // specifically on a field no other validator pins.
        let mut receipt2 = good_receipt("quill");
        receipt2.collection_overhead.calibrated_census_sample_ns += 1;
        assert!(matches!(
            receipt2.validate(&expectation("quill")),
            Err(WorkReceiptError::DigestMismatch { .. })
        ));
        assert!(receipt.validate(&expectation("quill")).is_err());
    }

    #[test]
    fn configured_as_actual_substitution_is_rejected() {
        // An "observed" width with no seam is a copied configuration value.
        let mut receipt = good_receipt("quill");
        receipt.concurrency.observed_width = WidthObservation::Observed {
            threads: 1,
            seam: "  ".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::UnnamedSeam {
                field: "observed_width"
            })
        ));
        // Observed zero threads is a fabrication, not an observation.
        let mut receipt = good_receipt("quill");
        receipt.concurrency.observed_width = WidthObservation::Observed {
            threads: 0,
            seam: "test seam".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::FabricatedWidth { .. })
        ));
    }

    #[test]
    fn impossible_concurrency_is_rejected() {
        // Use a wall long enough that clock-tick quantization tolerance
        // cannot excuse the claim: 1000 mean-active threads over 2 seconds
        // against a census of a handful of threads.
        let mut receipt = good_receipt("quill");
        let wall = 2_000_000_000_u64;
        receipt.phases = vec![
            PhaseSample {
                phase: LifecyclePhase::FirstFeed,
                window_elapsed_ns: 0,
            },
            PhaseSample {
                phase: LifecyclePhase::FeedComplete,
                window_elapsed_ns: 1_500_000_000,
            },
            PhaseSample {
                phase: LifecyclePhase::CommitComplete,
                window_elapsed_ns: 1_700_000_000,
            },
            PhaseSample {
                phase: LifecyclePhase::SearchableVerified,
                window_elapsed_ns: 1_800_000_000,
            },
            PhaseSample {
                phase: LifecyclePhase::QuiescenceJoined,
                window_elapsed_ns: wall,
            },
        ];
        receipt.concurrency.wall_ns = wall;
        receipt.binding.window_ended_ns = receipt.binding.window_started_ns + wall;
        let absurd = wall.saturating_mul(1_000);
        receipt.concurrency.process_cpu = CpuTimeObservation::Observed {
            cpu_ns: absurd,
            seam: PROCESS_CPU_SEAM.to_owned(),
        };
        receipt.concurrency.active_concurrency_integral =
            ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                integral_thread_ns: absurd,
                mean_active_millithreads: absurd / wall * 1000,
            };
        receipt.concurrency.unattributed_cpu_ns =
            absurd - receipt.concurrency.role_cpu_ns.values().sum::<u64>();
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::ImpossibleConcurrency { .. })
        ));
    }

    #[test]
    fn fabricated_integral_without_cpu_observation_is_rejected() {
        let mut receipt = good_receipt("quill");
        receipt.concurrency.process_cpu = CpuTimeObservation::Unavailable {
            platform: "macos".to_owned(),
            reason: "no /proc".to_owned(),
        };
        receipt.concurrency.role_cpu_ns.clear();
        receipt.concurrency.unattributed_cpu_ns = 0;
        receipt.concurrency.live_thread_census = RoleCensus::Unavailable {
            platform: "macos".to_owned(),
            reason: "no /proc".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::FabricatedIntegral)
        ));
    }

    #[test]
    fn typed_unavailability_validates_when_nothing_is_fabricated() {
        // The honest macOS shape: CPU, census, and integral all typed absent,
        // no role CPU, no fabricated observed width.
        let mut receipt = good_receipt("quill");
        receipt.concurrency.process_cpu = CpuTimeObservation::Unavailable {
            platform: "macos".to_owned(),
            reason: "no /proc; per-process CPU collector not implemented on this platform"
                .to_owned(),
        };
        receipt.concurrency.active_concurrency_integral = ActiveConcurrencyIntegral::Unavailable {
            platform: "macos".to_owned(),
            reason: "process CPU observation unavailable".to_owned(),
        };
        receipt.concurrency.live_thread_census = RoleCensus::Unavailable {
            platform: "macos".to_owned(),
            reason: "no /proc; per-thread role census not implemented on this platform".to_owned(),
        };
        receipt.concurrency.role_cpu_ns.clear();
        receipt.concurrency.unattributed_cpu_ns = 0;
        receipt.concurrency.observed_width = WidthObservation::Unavailable {
            platform: "macos".to_owned(),
            reason: "pool accessor not implemented on this platform".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        receipt
            .validate(&expectation("quill"))
            .expect("typed absence is admissible");
    }

    #[test]
    fn cross_platform_masquerade_is_rejected() {
        let mut receipt = good_receipt("quill");
        if let RoleCensus::Sampled(sample) = &mut receipt.concurrency.live_thread_census {
            sample.platform = "macos".to_owned();
        }
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::PlatformMasquerade { .. })
        ));
        let mut receipt = good_receipt("quill");
        receipt.concurrency.process_cpu = CpuTimeObservation::Observed {
            cpu_ns: 4_500,
            seam: "macos:host_processor_info".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::UnknownSeam {
                field: "process_cpu",
                ..
            })
        ));
    }

    #[test]
    fn roles_without_census_are_rejected() {
        let mut receipt = good_receipt("quill");
        receipt.concurrency.live_thread_census = RoleCensus::Unavailable {
            platform: "macos".to_owned(),
            reason: "no /proc".to_owned(),
        };
        // role_cpu_ns still populated: resolving roles without a census is
        // structurally impossible, so this shape is fabricated.
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::RolesWithoutCensus)
        ));
    }

    #[test]
    fn missing_required_role_is_rejected() {
        let mut receipt = good_receipt("tantivy");
        if let RoleCensus::Sampled(sample) = &mut receipt.concurrency.live_thread_census {
            sample
                .threads
                .retain(|thread| thread.role != WorkerRole::TantivyIndexWorker);
        }
        receipt
            .concurrency
            .role_cpu_ns
            .remove("tantivy_index_worker");
        let role_sum: u64 = receipt.concurrency.role_cpu_ns.values().sum();
        if let CpuTimeObservation::Observed { cpu_ns, .. } = &mut receipt.concurrency.process_cpu {
            *cpu_ns = role_sum + receipt.concurrency.unattributed_cpu_ns;
            let cpu = *cpu_ns;
            receipt.concurrency.active_concurrency_integral =
                ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                    integral_thread_ns: cpu,
                    mean_active_millithreads: cpu * 1000 / receipt.concurrency.wall_ns,
                };
        }
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("tantivy")),
            Err(WorkReceiptError::MissingRequiredRole {
                role: "tantivy_index_worker",
                ..
            })
        ));
        // A census with workers but no segment updater is also incomplete.
        let mut receipt = good_receipt("tantivy");
        if let RoleCensus::Sampled(sample) = &mut receipt.concurrency.live_thread_census {
            sample
                .threads
                .retain(|thread| thread.role != WorkerRole::TantivySegmentUpdater);
        }
        receipt
            .concurrency
            .role_cpu_ns
            .remove("tantivy_segment_updater");
        let role_sum: u64 = receipt.concurrency.role_cpu_ns.values().sum();
        if let CpuTimeObservation::Observed { cpu_ns, .. } = &mut receipt.concurrency.process_cpu {
            *cpu_ns = role_sum + receipt.concurrency.unattributed_cpu_ns;
            let cpu = *cpu_ns;
            receipt.concurrency.active_concurrency_integral =
                ActiveConcurrencyIntegral::DerivedFromProcessCpuIdentity {
                    integral_thread_ns: cpu,
                    mean_active_millithreads: cpu * 1000 / receipt.concurrency.wall_ns,
                };
        }
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("tantivy")),
            Err(WorkReceiptError::MissingRequiredRole {
                role: "tantivy_segment_updater",
                ..
            })
        ));
    }

    #[test]
    fn inconsistent_counters_are_rejected() {
        for (stage, mutate) in [
            (
                "accepted_docs",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.accepted_docs -= 1)
                    as Box<dyn Fn(&mut WorkReceipt)>,
            ),
            (
                "processed_docs",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.processed_docs += 1),
            ),
            (
                "committed_docs",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.committed_docs -= 1),
            ),
            (
                "searchable_docs",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.searchable_docs -= 2),
            ),
            (
                "accepted_bytes",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.accepted_bytes += 7),
            ),
            (
                "processed_bytes",
                Box::new(|receipt: &mut WorkReceipt| receipt.counters.processed_bytes -= 7),
            ),
        ] {
            let mut receipt = good_receipt("quill");
            mutate(&mut receipt);
            let receipt = receipt.seal().expect("reseal");
            assert!(
                matches!(
                    receipt.validate(&expectation("quill")),
                    Err(WorkReceiptError::CounterInequality { stage: got, .. }) if got == stage
                ),
                "stage {stage} must fail closed"
            );
        }
        let mut receipt = good_receipt("quill");
        receipt.counters.terminal_commit_calls = 2;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::TerminalCommitCalls { actual: 2 })
        ));
        let mut receipt = good_receipt("quill");
        receipt.counters.feed_calls = 0;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::NoFeedCalls)
        ));
    }

    #[test]
    fn corpus_byte_stage_fabrication_is_rejected() {
        // Claiming an observed corpus-byte total that disagrees with the
        // manifest denominator is rejected; naming the typed gap passes.
        let mut receipt = good_receipt("quill");
        receipt.counters.committed_corpus_bytes = ByteStageObservation::Observed(1);
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::CounterInequality {
                stage: "committed_corpus_bytes",
                ..
            })
        ));
        let mut receipt = good_receipt("quill");
        receipt.counters.searchable_corpus_bytes = ByteStageObservation::StructurallyUnobservable {
            seam: " ".to_owned(),
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::UnnamedSeam { .. })
        ));
    }

    #[test]
    fn unjoined_work_and_terminal_state_are_rejected() {
        // Engine/join mismatch.
        let mut receipt = good_receipt("tantivy");
        receipt.terminal.join = TerminalJoin::QuillSynchronousCommit;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("tantivy")),
            Err(WorkReceiptError::JoinContractMismatch { .. })
        ));
        // A rearmed writer in the terminal join left an armed worker behind.
        let mut receipt = good_receipt("tantivy");
        receipt.terminal.join = TerminalJoin::TantivyWorkersJoined {
            join_elapsed_ns: 900,
            searchable_segments_before: 3,
            searchable_segments_after: 3,
            writer_rearmed: true,
        };
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("tantivy")),
            Err(WorkReceiptError::WriterRearmedInTerminalJoin)
        ));
        let mut receipt = good_receipt("quill");
        receipt.terminal.drained = false;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::UndrainedFeed)
        ));
        let mut receipt = good_receipt("quill");
        receipt.terminal.pending_docs_zero = false;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::PendingWorkAtTerminal { .. })
        ));
        let mut receipt = good_receipt("quill");
        receipt.terminal.retryable = true;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::RetryablePredicateViolation { .. })
        ));
    }

    #[test]
    fn phase_disorder_is_rejected() {
        let mut receipt = good_receipt("quill");
        receipt.phases.swap(1, 2);
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::PhaseOrderViolation { .. })
        ));
        let mut receipt = good_receipt("quill");
        receipt.phases.pop();
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::PhaseCountMismatch { .. })
        ));
        // A terminal phase that disagrees with the window wall is a clock
        // that stopped somewhere other than quiescence.
        let mut receipt = good_receipt("quill");
        receipt.phases[4].window_elapsed_ns += 1;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::PhaseWindowMismatch { .. })
        ));
    }

    #[test]
    fn stale_source_and_elf_bindings_are_rejected() {
        let mut receipt = good_receipt("quill");
        receipt.binding.executable_sha256 = "not-a-sha".to_owned();
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::MalformedSha256 {
                field: "executable_sha256",
                ..
            })
        ));
        let mut receipt = good_receipt("quill");
        receipt.binding.git_rev = String::new();
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::EmptyBindingField { field: "git_rev" })
        ));
        let mut receipt = good_receipt("quill");
        receipt.binding.corpus_manifest_sha256 = "C".repeat(64);
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::MalformedSha256 {
                field: "corpus_manifest_sha256",
                ..
            })
        ));
    }

    #[test]
    fn role_cpu_identity_must_sum_exactly() {
        let mut receipt = good_receipt("quill");
        receipt.concurrency.unattributed_cpu_ns += 1;
        let receipt = receipt.seal().expect("reseal");
        assert!(matches!(
            receipt.validate(&expectation("quill")),
            Err(WorkReceiptError::RoleCpuIdentityViolation { .. })
        ));
    }

    #[test]
    fn stat_parser_handles_hostile_comm_names() {
        // comm with spaces and a closing paren; utime=7 stime=11.
        let line = "1234 (thrd (evil) x) R 1 1 1 0 -1 4194304 0 0 0 0 7 11 0 0 20 0 1 0 100 0 0";
        let reading = parse_stat_cpu_ticks(line).expect("parse hostile stat");
        assert_eq!(reading.ticks, 18);
        assert_eq!(parse_stat_cpu_ticks("garbage"), None);
        assert_eq!(parse_stat_cpu_ticks(""), None);
    }

    #[test]
    fn ticks_convert_exactly_at_common_hz() {
        assert_eq!(ticks_to_ns(0, 100), 0);
        assert_eq!(ticks_to_ns(1, 100), 10_000_000);
        assert_eq!(ticks_to_ns(150, 100), 1_500_000_000);
        assert_eq!(ticks_to_ns(1, 0), 0);
        // Large cumulative readings must not overflow.
        assert_eq!(
            ticks_to_ns(u64::MAX / 2_000_000_000, 100),
            92_233_720_360_000_000
        );
    }

    #[test]
    fn thread_role_classification_matches_known_families() {
        assert_eq!(
            classify_thread_role("anything", true),
            WorkerRole::BenchCaller
        );
        assert_eq!(
            classify_thread_role("qg-rayon-3", false),
            WorkerRole::RayonWorker
        );
        // Real tantivy 0.26.1 names as they appear in a 15-byte comm.
        assert_eq!(
            classify_thread_role("thrd-tantivy-in", false),
            WorkerRole::TantivyIndexWorker
        );
        assert_eq!(
            classify_thread_role("segment_updater", false),
            WorkerRole::TantivySegmentUpdater
        );
        assert_eq!(
            classify_thread_role("merge_thread_0", false),
            WorkerRole::TantivyMergeWorker
        );
        assert_eq!(
            classify_thread_role("docstore-compre", false),
            WorkerRole::TantivyDocstoreCompressor
        );
        assert_eq!(
            classify_thread_role("asupersync-work", false),
            WorkerRole::AsupersyncRuntime
        );
        assert_eq!(
            classify_thread_role("tokio-runtime-w", false),
            WorkerRole::Unattributed
        );
        assert_eq!(
            classify_thread_role("perf_matrix-abc", false),
            WorkerRole::Unattributed,
            "unnamed threads inheriting the process comm stay unattributed"
        );
        for role in [
            WorkerRole::BenchCaller,
            WorkerRole::RayonWorker,
            WorkerRole::TantivyIndexWorker,
            WorkerRole::TantivySegmentUpdater,
            WorkerRole::TantivyMergeWorker,
            WorkerRole::TantivyDocstoreCompressor,
            WorkerRole::AsupersyncRuntime,
            WorkerRole::Unattributed,
        ] {
            assert_eq!(WorkerRole::from_label(role.label()), Some(role));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_collectors_observe_real_threads_and_cpu() {
        // Real end-to-end pass over the live collectors: burn CPU on a named
        // thread, then require the census to see it with its role resolved.
        let baseline = sample_raw_census();
        let baseline_cpu = sample_process_cpu();
        assert!(matches!(baseline, RawCensus::Sampled { .. }));
        assert!(matches!(baseline_cpu, CpuTimeObservation::Observed { .. }));
        let worker = std::thread::Builder::new()
            .name(format!("{BENCH_RAYON_THREAD_PREFIX}77"))
            .spawn(|| {
                // Burn well past one clock tick (10ms at hz=100) so the
                // tick-granularity caveat cannot zero the reading.
                let started = Instant::now();
                let mut acc = 0_u64;
                while started.elapsed().as_millis() < 60 {
                    acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    std::hint::black_box(acc);
                }
                acc
            })
            .expect("spawn census worker");
        // Sample while the worker is alive.
        std::thread::sleep(std::time::Duration::from_millis(40));
        let live = sample_raw_census();
        worker.join().expect("join census worker");
        let census = resolve_role_census(&live, &baseline, current_tid());
        let RoleCensus::Sampled(sample) = &census else {
            panic!("linux census must sample");
        };
        assert_eq!(sample.platform, "linux");
        assert!(sample.clock_tick_hz > 0);
        let rayon_named: Vec<_> = sample
            .threads
            .iter()
            .filter(|thread| thread.role == WorkerRole::RayonWorker)
            .collect();
        assert!(
            !rayon_named.is_empty(),
            "the named worker thread must be censused with its role resolved"
        );
        assert!(
            sample
                .threads
                .iter()
                .any(|thread| thread.role == WorkerRole::BenchCaller),
            "the calling thread must be censused as bench caller"
        );
        assert!(
            sample
                .caveats
                .iter()
                .any(|caveat| matches!(caveat, CensusCaveat::ClockTickGranularity { .. })),
            "tick granularity caveat must be typed on every sample"
        );
        // The worker burned ≥1 tick of CPU inside the window.
        assert!(
            rayon_named.iter().any(|thread| thread.window_cpu_ns > 0),
            "a worker that burned 60ms must show in-window CPU"
        );
        let terminal_cpu = sample_process_cpu();
        let (
            CpuTimeObservation::Observed { cpu_ns: before, .. },
            CpuTimeObservation::Observed { cpu_ns: after, .. },
        ) = (&baseline_cpu, &terminal_cpu)
        else {
            panic!("linux process CPU must observe");
        };
        assert!(after >= before, "cumulative process CPU is monotone");
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_collectors_are_typed_absent_never_fabricated() {
        assert!(matches!(sample_raw_census(), RawCensus::Unavailable { .. }));
        assert!(matches!(
            sample_process_cpu(),
            CpuTimeObservation::Unavailable { .. }
        ));
        assert_eq!(current_tid(), None);
    }

    #[test]
    fn collector_assembles_a_valid_receipt_end_to_end() {
        let collector_binding = CollectorBinding {
            run_id: "collector-test".to_owned(),
            machine_fingerprint: "machine-test".to_owned(),
            build_profile: "dev".to_owned(),
            executable_sha256: "e".repeat(64),
            git_rev: "abcdef012345".to_owned(),
            gate: "QG-1".to_owned(),
            fixture: "bulk/tiny/1/positions_on".to_owned(),
            metric: "docs_per_second".to_owned(),
            engine: "quill".to_owned(),
            timing_mode: "per-call".to_owned(),
            corpus_identity: "qg1-native/prepared-prefix-v1/test".to_owned(),
            corpus_manifest_sha256: "c".repeat(64),
        };
        let mut collector = WorkReceiptCollector::new(collector_binding, 1);
        collector.begin_window();
        let origin = Instant::now();
        let elapsed = |origin: Instant| u64::try_from(origin.elapsed().as_nanos()).expect("ns");
        collector.on_phase(LifecyclePhase::FirstFeed, 0);
        collector.record_feed_batch(6, 2_048);
        collector.record_feed_batch(6, 2_048);
        // Do a little real work so the window is not degenerate.
        std::thread::sleep(std::time::Duration::from_millis(2));
        collector.on_phase(LifecyclePhase::FeedComplete, elapsed(origin));
        collector.on_phase(LifecyclePhase::CommitComplete, elapsed(origin));
        collector.record_committed(
            12,
            EngineByteObservation::Observed {
                bytes: 65_536,
                seam: "test seam: engine layout accessor".to_owned(),
            },
            EngineByteObservation::Observed {
                bytes: 1,
                seam: "test seam: engine segment accessor".to_owned(),
            },
        );
        collector.record_searchable(12);
        collector.on_phase(LifecyclePhase::SearchableVerified, elapsed(origin));
        collector.record_queue(QueueObservation::SynchronousNoQueue {
            seam: "test seam: synchronous writer".to_owned(),
        });
        collector.record_width(WidthObservation::Observed {
            threads: 1,
            seam: "test seam: pool accessor".to_owned(),
        });
        collector.record_measured_sum_ns(1_000_000);
        collector.record_terminal(TerminalJoin::QuillSynchronousCommit, "completed", false);
        collector.on_phase(LifecyclePhase::QuiescenceJoined, elapsed(origin));
        let receipt = collector.finish(10_000).expect("assemble receipt");
        let expectation = WorkReceiptExpectation {
            engine: "quill".to_owned(),
            doc_count: 12,
            total_bytes: 4_096,
            configured_threads: 1,
        };
        // On Linux the full receipt must validate; elsewhere the collectors
        // are typed absent and the same validation must still pass because
        // nothing was fabricated.
        receipt
            .validate(&expectation)
            .expect("collector-built receipt validates");
        assert_eq!(receipt.counters.feed_calls, 2);
        assert_eq!(receipt.counters.accepted_docs, 12);
        assert_eq!(receipt.counters.terminal_commit_calls, 1);
        // One pass at feed-complete plus one at searchable-verified.
        assert_eq!(receipt.collection_overhead.census_samples_in_window, 2);
        assert!(receipt.terminal.drained);
        assert!(receipt.terminal.pending_docs_zero);
        let line = receipt.bounded_log_line();
        assert!(line.starts_with("[qg1-work-receipt] "));
        assert!(
            line.len() < 1024,
            "log line must stay bounded: {}",
            line.len()
        );
        assert!(line.contains("terminal=completed"));
        assert!(line.contains("retryable=false"));
        assert!(!line.contains('\n'));
    }

    #[test]
    fn evidence_container_round_trips_and_stays_additive() {
        let evidence = WorkReceiptEvidence {
            schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
            mode: WorkReceiptMode::On.label().to_owned(),
            cells: vec![WorkReceiptCellEvidence {
                schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
                fixture: "bulk/tiny/1/positions_on".to_owned(),
                rounds_quill: 3,
                rounds_tantivy: 3,
                last_quill_receipt: Some(good_receipt("quill")),
                last_tantivy_receipt: Some(good_receipt("tantivy")),
                all_receipts_validated: true,
            }],
        };
        let encoded = serde_json::to_string(&evidence).expect("encode evidence");
        let decoded: WorkReceiptEvidence = serde_json::from_str(&encoded).expect("decode evidence");
        assert_eq!(decoded, evidence);
    }
}
