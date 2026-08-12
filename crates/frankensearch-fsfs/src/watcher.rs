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

/// How many one-millisecond polls a lifecycle transition may take before the
/// caller gives up on it.
///
/// `start` and `stop_checked` both wait on a generation another caller owns.
/// An unbounded wait turns one wedged generation into a caller that never
/// returns and a shutdown that never completes; the bound is generous enough
/// that an ordinary handover is never affected and short enough that a wedged
/// one is reported.
const LIFECYCLE_TRANSITION_POLLS: usize = 30_000;

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
    root_identities: BTreeMap<PathBuf, RootIdentity>,
    /// Set when any root could not yield a trustworthy identity. Such a scan
    /// may still upsert what it saw, but it can never authorize a deletion.
    identity_degraded: bool,
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

    /// Record the identity a root had while this scan read it, or mark the
    /// scan degraded when the platform cannot supply one.
    fn record_root_identity(&mut self, root: impl Into<PathBuf>, identity: Option<RootIdentity>) {
        match identity {
            Some(identity) => {
                self.root_identities.insert(root.into(), identity);
            }
            None => self.identity_degraded = true,
        }
    }

    /// Whether this scan may be used to establish deletion authority.
    const fn identity_is_trustworthy(&self) -> bool {
        !self.identity_degraded
    }

    /// Identities observed for the roots this scan resolved.
    const fn root_identities(&self) -> &BTreeMap<PathBuf, RootIdentity> {
        &self.root_identities
    }

    /// Mark every root whose identity differs from the baseline's as
    /// unresolved.
    ///
    /// A replaced root reads as an ordinary, complete, empty directory. Only
    /// the identity distinguishes "the tree is empty" from "this is not the
    /// tree the baseline describes", so the comparison happens before any
    /// deletion is derived. A root absent from `baseline` is new rather than
    /// swapped and is left alone.
    fn reject_swapped_roots(&mut self, baseline: &BTreeMap<PathBuf, RootIdentity>) {
        let swapped = self
            .root_identities
            .iter()
            .filter(|(root, identity)| {
                baseline
                    .get(*root)
                    .is_some_and(|previous| previous != *identity)
            })
            .map(|(root, _)| root.clone())
            .collect::<Vec<_>>();
        for root in swapped {
            self.record_unresolved(root);
        }
    }
}

/// A scan, plus the abort predicate it must honour while walking.
///
/// The predicate is part of the contract rather than a wrapper detail: a
/// collector that cannot be told to stop turns every long walk into a window
/// where a stop is ignored, which is exactly what shipping a hardcoded `false`
/// produced.
type SnapshotCollector = dyn Fn(
        &[PathBuf],
        &DiscoveryConfig,
        &dyn Fn() -> bool,
    ) -> SearchResult<(FileSnapshot, ScanCompleteness)>
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

/// Test-only observer invoked at one exact point in the stop protocol.
///
/// The hooks exist so a test can inspect the lock state *at* the publication
/// and park boundaries instead of inferring it from timing. They are compiled
/// out of shipping builds, and the production paths below are unchanged apart
/// from the two `#[cfg(test)]` call sites.
#[cfg(test)]
type StopObserver = Box<dyn Fn(&WatcherStop) + Send + Sync>;

#[derive(Default)]
struct WatcherStop {
    requested: AtomicBool,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
    /// Invoked at the instant the stop flag is published.
    #[cfg(test)]
    publish_observer: Mutex<Option<StopObserver>>,
    /// Invoked while `wait_lock` is held, immediately before parking.
    #[cfg(test)]
    park_observer: Mutex<Option<StopObserver>>,
    /// Invoked each time the waiter re-evaluates its predicate, which is
    /// exactly when it has processed a wakeup and is deciding whether to
    /// re-park.
    #[cfg(test)]
    wake_observer: Mutex<Option<StopObserver>>,
}

impl WatcherStop {
    /// Observe the exact publication boundary. The observer runs wherever the
    /// flag store runs, so a regression that moves the store out of
    /// `wait_lock` moves the observation with it.
    #[cfg(test)]
    fn set_publish_observer(&self, observer: StopObserver) {
        *lock_or_recover(&self.publish_observer) = Some(observer);
    }

    /// Observe the exact park boundary, while `wait_lock` is still held.
    #[cfg(test)]
    fn set_park_observer(&self, observer: StopObserver) {
        *lock_or_recover(&self.park_observer) = Some(observer);
    }

    /// Observe every predicate re-evaluation, i.e. every processed wakeup.
    #[cfg(test)]
    fn set_wake_observer(&self, observer: StopObserver) {
        *lock_or_recover(&self.wake_observer) = Some(observer);
    }

    /// Publish the stop flag.
    ///
    /// The store lives here, behind a live `wait_lock` guard the caller must
    /// already hold, and the observation of that store sits in the same
    /// function. Keeping them together is the point: as two adjacent
    /// statements in `request`, a change that moved the store out of the lock
    /// could leave the observer behind and the evidence would still read
    /// "published under the lock". Moving the store now means removing it from
    /// a function that cannot be called without a guard.
    fn publish_requested(&self, publication_guard: &std::sync::MutexGuard<'_, ()>) {
        // Borrowed purely to make lock ownership a precondition of the store.
        let () = **publication_guard;
        #[cfg(test)]
        self.notify_observer(&self.publish_observer);
        self.requested.store(true, Ordering::Release);
    }

    #[cfg(test)]
    fn notify_observer(&self, slot: &Mutex<Option<StopObserver>>) {
        let observer = lock_or_recover(slot);
        if let Some(observer) = observer.as_ref() {
            observer(self);
        }
    }

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
        let publication = lock_or_recover(&self.wait_lock);
        self.publish_requested(&publication);
        self.wait_cv.notify_all();
    }

    fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    /// Park for at most `duration`, returning early only for a real stop.
    ///
    /// A single `wait_timeout` is not a wait for `duration` — a condvar may
    /// wake for no reason at all, and any such wakeup made the previous form
    /// return `false` as though the backoff had elapsed. The caller treats
    /// that as "the backoff is over", so a spurious wakeup silently shortened
    /// a restart backoff of up to `MAX_BACKOFF_MS` to nothing and the loop
    /// span. The wait must therefore be a predicate/deadline loop: wake, test
    /// the flag, and re-park for the *remaining* time unless stop is actually
    /// requested.
    ///
    /// `wait_timeout_while` is exactly that loop — it re-checks the predicate
    /// on every wakeup and tracks the deadline across re-parks, so `duration`
    /// is an upper bound on the whole call rather than on one park.
    fn wait_or_stopped(&self, duration: Duration) -> bool {
        if self.is_requested() {
            return true;
        }
        let guard = lock_or_recover(&self.wait_lock);
        // Announced while the lock is still held: a notifier that then
        // acquires `wait_lock` can only do so once this thread has released
        // it into the wait below, which is what makes "parked" observable.
        #[cfg(test)]
        self.notify_observer(&self.park_observer);
        let (_guard, _timed_out) = self
            .wait_cv
            .wait_timeout_while(guard, duration, |()| {
                // Runs on entry and on every wakeup, under `wait_lock`: the
                // one point at which the waiter has demonstrably processed a
                // notification and is choosing whether to re-park.
                #[cfg(test)]
                self.notify_observer(&self.wake_observer);
                !self.is_requested()
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.is_requested()
    }
}

/// The distinct roots a configuration names.
///
/// Coverage is keyed by path and a configuration may name the same root twice.
/// Counting the slice rather than its distinct members demands an identity for
/// a duplicate that a path-keyed map can never hold, so every coverage check
/// would fail for as long as the duplicate stayed in the configuration — and a
/// watcher that can never establish authority can never derive a deletion.
fn distinct_roots(roots: &[PathBuf]) -> BTreeSet<&Path> {
    roots.iter().map(PathBuf::as_path).collect()
}

/// How many consecutive complete passes may end without adjudicating anything
/// before the watcher stops asking for another one.
///
/// A probationary pass owes exactly one confirming pass. If the roots keep
/// changing between observations that confirmation never arrives, and
/// re-arming unconditionally turns it into a full rescan on every iteration of
/// the ingest loop, forever. The bound converts a stuck watcher into a
/// fail-closed hold — no authority, no deletes, no spin — and any new event
/// resets the counter, so real work always gets its passes.
const MAX_UNSETTLED_PASSES: u32 = 2;

#[derive(Default)]
struct ReconciliationState {
    /// Paths believed to be indexed. Bookkeeping for upserts; on its own it is
    /// never a deletion baseline.
    indexed_snapshot: FileSnapshot,
    baseline_initialized: bool,
    required: bool,
    affected_paths: BTreeSet<PathBuf>,
    epoch: u64,
    /// What is known about the tree the index describes.
    authority: DeletionAuthorityState,
    /// Operator opt-in permitting the next adjudicating pass to also settle
    /// the inherited legacy names. One-shot, and consumed only by a pass that
    /// actually adjudicated them.
    rebuild_authorized: bool,
    /// Consecutive complete passes that concluded without adjudicating.
    unsettled_passes: u32,
}

/// What the watcher knows about the tree its index describes.
///
/// The three states are kept apart because one `Option<DeletionAuthority>`
/// could not tell them apart: a snapshot inherited from crash recovery, a
/// first observation awaiting confirmation, and a confirmed authority all
/// shared one slot. Storing the inherited snapshot there — which is what the
/// catch-up path did to retain it — promoted it to full authority on the very
/// next call, so a single pass could delete everything it named.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum DeletionAuthorityState {
    /// Nothing observed and nothing inherited.
    #[default]
    Absent,
    /// Names the watcher inherited without ever observing the identity of the
    /// roots they were read through: a crash-recovery restore, or a caller's
    /// `previous` argument. Absence of historical identity is not evidence of
    /// deletion, so this state can never authorize one.
    UnverifiedLegacy { legacy: FileSnapshot },
    /// One complete, trustworthy scan, not yet confirmed by a second one.
    /// A candidate is evidence, not authority.
    Probationary {
        candidate: DeletionAuthority,
        legacy: Option<FileSnapshot>,
    },
    /// Confirmed. The only state that authorizes a deletion.
    Established {
        authority: DeletionAuthority,
        legacy: Option<FileSnapshot>,
    },
}

impl DeletionAuthorityState {
    /// The established authority, if this state is the one that has any.
    fn established(&self) -> Option<&DeletionAuthority> {
        match self {
            Self::Established { authority, .. } => Some(authority),
            Self::Absent | Self::UnverifiedLegacy { .. } | Self::Probationary { .. } => None,
        }
    }

    /// Names inherited without identity evidence behind them.
    fn legacy(&self) -> Option<&FileSnapshot> {
        match self {
            Self::Absent => None,
            Self::UnverifiedLegacy { legacy } => Some(legacy),
            Self::Probationary { legacy, .. } | Self::Established { legacy, .. } => legacy.as_ref(),
        }
    }

    /// Detach the inherited names so the next state can carry them forward.
    fn take_legacy(&mut self) -> Option<FileSnapshot> {
        match self {
            Self::Absent => None,
            Self::UnverifiedLegacy { legacy } => Some(std::mem::take(legacy)),
            Self::Probationary { legacy, .. } | Self::Established { legacy, .. } => legacy.take(),
        }
    }
}

/// How a completed pass concluded, and therefore what it may commit.
///
/// The previous shape decided this implicitly and then committed
/// unconditionally, so a pass that had deliberately derived no deletes still
/// adopted its own snapshot as the new authority — destroying the baseline it
/// was created to preserve and silently discarding every stale delete. Making
/// the conclusion explicit is what stops the epilogue from contradicting the
/// decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassOutcome {
    /// Deletions were adjudicated: either against an established authority, or
    /// against a probation candidate this pass confirmed, or as the very first
    /// observation with nothing inherited to put at risk. Only this outcome
    /// installs authority.
    Adjudicated,
    /// A candidate was recorded and nothing was adjudicated. Every inherited
    /// name survives untouched and a confirming pass is owed.
    Probationary,
    /// The platform cannot supply trustworthy root identity. Upserts only,
    /// never authority — and never a re-arm, because asking again cannot make
    /// an identity appear.
    Degraded,
}

/// What a complete scan is permitted to conclude, decided before any operation
/// is applied and committed only if the state it was planned against is still
/// the state in front of it.
struct PassPlan {
    outcome: PassOutcome,
    /// The names this pass may adjudicate. `None` derives no deletion at all.
    deletion_baseline: Option<FileSnapshot>,
    /// Whether the inherited legacy names were folded into that baseline, and
    /// therefore whether the one-shot rebuild authority is spent.
    adjudicates_legacy: bool,
    /// Generation of the established authority the plan was made against.
    /// Continuity is proven by this value still being current at the commit.
    expected_generation: Option<u64>,
}

impl PassPlan {
    /// Whether this pass may derive deletions, including for the paths its own
    /// unapplied events named.
    const fn adjudicates(&self) -> bool {
        matches!(self.outcome, PassOutcome::Adjudicated)
    }
}

/// A snapshot plus the exact root identities it was observed through.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DeletionAuthority {
    snapshot: FileSnapshot,
    root_identities: BTreeMap<PathBuf, RootIdentity>,
    generation: u64,
}

impl DeletionAuthority {
    /// Whether this authority can adjudicate deletions for `roots`.
    ///
    /// Every distinct configured root must carry an identity. A partial map
    /// cannot detect a swap of the roots it omits and an empty map cannot
    /// detect one at all, so both are refused rather than treated as "good
    /// enough". There is deliberately no exemption for an empty snapshot: the
    /// candidate set a pass deletes from is the baseline *plus* the paths its
    /// own unapplied events named, so an authority that names nothing can
    /// still delete, and exempting it authorized exactly that.
    fn covers(&self, roots: &[PathBuf]) -> bool {
        let distinct = distinct_roots(roots);
        if distinct.is_empty() {
            // A configuration with no roots observes nothing and may delete
            // nothing; it is coherent only while it also holds nothing.
            return self.root_identities.is_empty() && self.snapshot.is_empty();
        }
        distinct
            .iter()
            .all(|root| self.root_identities.contains_key(*root))
            && self.root_identities.len() == distinct.len()
    }

    /// Whether this snapshot was read through exactly these root identities.
    fn observed_through(&self, identities: &BTreeMap<PathBuf, RootIdentity>) -> bool {
        &self.root_identities == identities
    }
}

/// Identity of a configured root at scan time.
///
/// A root that is unmounted and left as a readable empty directory, or renamed
/// away and replaced by a fresh one, is byte-for-byte a *complete* scan of an
/// empty tree — and would delete the entire index. Comparing the identity
/// against the one recorded when the baseline was taken turns that into an
/// incomplete scan instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RootIdentity {
    device: u64,
    inode: u64,
}

impl RootIdentity {
    /// Read the identity of an opened root, mirroring the device+inode pair
    /// the publication lease already uses for the same purpose.
    #[cfg(unix)]
    fn of(metadata: &fs::Metadata) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;

        Some(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    /// Non-Unix targets expose no stable identity pair here. Returning `None`
    /// rather than a constant is the whole point: a constant compares equal to
    /// every other constant, so a swapped root would read as unchanged and
    /// authorize a mass delete. Absent identity means no deletion authority.
    #[cfg(not(unix))]
    fn of(_metadata: &fs::Metadata) -> Option<Self> {
        None
    }
}

type ReconciliationTracker = Arc<Mutex<ReconciliationState>>;

impl ReconciliationState {
    fn require_for_events(&mut self, events: &[WatchEvent]) {
        self.required = true;
        self.epoch = self.epoch.saturating_add(1);
        // Real work is owed again, so the fail-closed hold below is released:
        // the bound exists to stop a self-perpetuating rescan, not to stop the
        // watcher from reacting to the filesystem.
        self.unsettled_passes = 0;
        self.affected_paths
            .extend(events.iter().map(|event| event.path.clone()));
    }

    fn require_full_scan(&mut self) {
        self.required = true;
        self.epoch = self.epoch.saturating_add(1);
        self.unsettled_passes = 0;
    }

    /// The authority deletions may be derived against, or `None`.
    ///
    /// There is deliberately no fallback: an absent authority is not "use the
    /// working set instead", a probation candidate is not "close enough", and
    /// a partial identity map is not authority at all.
    fn established_authority(&self, roots: &[PathBuf]) -> Option<&DeletionAuthority> {
        self.authority
            .established()
            .filter(|authority| authority.covers(roots))
    }

    fn established_generation(&self) -> Option<u64> {
        self.authority.established().map(|held| held.generation)
    }

    fn established_identities(&self) -> Option<&BTreeMap<PathBuf, RootIdentity>> {
        self.authority
            .established()
            .map(|held| &held.root_identities)
    }

    /// Take in names the watcher never observed itself.
    ///
    /// They are retained rather than merged into any baseline, because they
    /// are exactly the names no observation can adjudicate: nothing records
    /// which roots they were read through, so their absence today cannot be
    /// distinguished from the roots having been replaced since.
    fn inherit_legacy(&mut self, snapshot: &FileSnapshot) {
        if snapshot.is_empty() {
            return;
        }
        // Union, never replace: each inherited record names files some earlier
        // regime indexed, and dropping one to make room for the next would lose
        // exactly the names that have no other evidence behind them.
        let mut retained = self.authority.take_legacy().unwrap_or_default();
        retained.extend(snapshot.iter().map(|(path, at)| (path.clone(), *at)));
        self.authority = match std::mem::take(&mut self.authority) {
            DeletionAuthorityState::Absent | DeletionAuthorityState::UnverifiedLegacy { .. } => {
                DeletionAuthorityState::UnverifiedLegacy { legacy: retained }
            }
            DeletionAuthorityState::Probationary { candidate, .. } => {
                DeletionAuthorityState::Probationary {
                    candidate,
                    legacy: Some(retained),
                }
            }
            DeletionAuthorityState::Established { authority, .. } => {
                DeletionAuthorityState::Established {
                    authority,
                    legacy: Some(retained),
                }
            }
        };
    }

    /// Grant one-shot permission to adjudicate the inherited legacy names.
    fn authorize_rebuild(&mut self) {
        self.rebuild_authorized = true;
    }

    /// Decide what a complete scan may conclude, committing nothing.
    ///
    /// Splitting the decision from the commit is what makes the two
    /// verifiable: the plan records the authority lineage it reasoned about,
    /// and the commit refuses to install anything if that lineage moved while
    /// the operations were being applied.
    fn plan_complete_pass(&self, roots: &[PathBuf], completeness: &ScanCompleteness) -> PassPlan {
        let expected_generation = self.established_generation();
        if !completeness.identity_is_trustworthy() {
            return PassPlan {
                outcome: PassOutcome::Degraded,
                deletion_baseline: None,
                adjudicates_legacy: false,
                expected_generation,
            };
        }
        let identities = completeness.root_identities();
        let adjudicate = |baseline: &FileSnapshot, legacy: Option<&FileSnapshot>| {
            let adjudicates_legacy = self.rebuild_authorized && legacy.is_some();
            let mut deletion_baseline = baseline.clone();
            if let Some(legacy) = legacy.filter(|_| adjudicates_legacy) {
                deletion_baseline.extend(legacy.iter().map(|(path, at)| (path.clone(), *at)));
            }
            PassPlan {
                outcome: PassOutcome::Adjudicated,
                deletion_baseline: Some(deletion_baseline),
                adjudicates_legacy,
                expected_generation,
            }
        };
        match &self.authority {
            // Established, and still covering the configured roots.
            DeletionAuthorityState::Established { authority, legacy }
                if authority.covers(roots) =>
            {
                adjudicate(&authority.snapshot, legacy.as_ref())
            }
            // A second complete scan through the identical roots confirms the
            // candidate, so what that candidate observed may now be
            // adjudicated.
            DeletionAuthorityState::Probationary { candidate, legacy }
                if candidate.covers(roots) && candidate.observed_through(identities) =>
            {
                adjudicate(&candidate.snapshot, legacy.as_ref())
            }
            // Nothing observed and nothing inherited: there is no prior claim
            // to put at risk, so this scan is authority for what it saw.
            DeletionAuthorityState::Absent => adjudicate(&FileSnapshot::new(), None),
            // Everything else — inherited names, an unconfirmed or superseded
            // candidate, an authority that no longer covers the configured
            // roots — derives nothing and records a candidate instead.
            DeletionAuthorityState::UnverifiedLegacy { .. }
            | DeletionAuthorityState::Probationary { .. }
            | DeletionAuthorityState::Established { .. } => PassPlan {
                outcome: PassOutcome::Probationary,
                deletion_baseline: None,
                adjudicates_legacy: false,
                expected_generation,
            },
        }
    }

    /// Install a planned pass, or refuse it because the lineage moved.
    ///
    /// Returns `false` when the established generation is no longer the one
    /// the plan was made against: another writer rebound the roots while this
    /// pass was applying, so installing now would silently adopt a snapshot
    /// read through one set of roots as authority over another.
    fn commit_complete_pass(
        &mut self,
        plan: &PassPlan,
        snapshot: FileSnapshot,
        identities: BTreeMap<PathBuf, RootIdentity>,
    ) -> bool {
        if self.established_generation() != plan.expected_generation {
            return false;
        }
        let generation = plan
            .expected_generation
            .map_or(1, |previous| previous.saturating_add(1));
        match plan.outcome {
            PassOutcome::Adjudicated => {
                let retained = self.authority.take_legacy();
                let legacy = if plan.adjudicates_legacy {
                    // Spent: the names were folded into this pass's baseline
                    // and every survivor of them has just been re-observed.
                    self.rebuild_authorized = false;
                    None
                } else {
                    retained
                };
                self.indexed_snapshot = snapshot.clone();
                self.baseline_initialized = true;
                self.authority = DeletionAuthorityState::Established {
                    authority: DeletionAuthority {
                        snapshot,
                        root_identities: identities,
                        generation,
                    },
                    legacy,
                };
                self.required = false;
                self.affected_paths.clear();
                self.unsettled_passes = 0;
            }
            PassOutcome::Probationary => {
                // The inherited names, and any authority that just lost
                // coverage of the configured roots, survive as legacy: this
                // pass adjudicated neither. The working set is left exactly as
                // it was found — a probationary pass changes nothing but the
                // candidate.
                let legacy = self.retire_into_legacy();
                self.authority = DeletionAuthorityState::Probationary {
                    candidate: DeletionAuthority {
                        snapshot,
                        root_identities: identities,
                        generation,
                    },
                    legacy,
                };
                self.unsettled_passes = self.unsettled_passes.saturating_add(1);
                self.required = self.unsettled_passes < MAX_UNSETTLED_PASSES;
            }
            PassOutcome::Degraded => {
                // Upserts only, and explicitly no re-arm: a platform that
                // cannot identify its roots will not start identifying them
                // because it was asked again, and re-arming turned that into a
                // full rescan on every iteration of the ingest loop.
                self.indexed_snapshot = snapshot;
                self.baseline_initialized = true;
                self.required = false;
                self.unsettled_passes = 0;
            }
        }
        true
    }

    /// Seed the very first authority from a scan that applies nothing.
    ///
    /// Only legal from [`DeletionAuthorityState::Absent`]: with nothing
    /// inherited, nothing held, and no candidate, this scan puts no prior
    /// claim at risk and derives no deletion, so recording it costs nothing
    /// and spares the next pass a redundant walk. Every other state has
    /// something to adjudicate, and adjudication belongs to a pass that can
    /// apply the operations it derives. Returns whether it seeded.
    fn seed_initial_authority(
        &mut self,
        snapshot: FileSnapshot,
        identities: BTreeMap<PathBuf, RootIdentity>,
    ) -> bool {
        if !matches!(self.authority, DeletionAuthorityState::Absent) {
            return false;
        }
        self.indexed_snapshot = snapshot.clone();
        self.baseline_initialized = true;
        self.authority = DeletionAuthorityState::Established {
            authority: DeletionAuthority {
                snapshot,
                root_identities: identities,
                generation: 1,
            },
            legacy: None,
        };
        true
    }

    /// Advance an established authority in place, preserving root continuity.
    ///
    /// Refuses unless the roots this snapshot was read through are the very
    /// ones the authority was established through. A scan that saw different
    /// identities is a scan of a different tree, and adopting it would rebind
    /// the authority to a swapped root — after which every file of the old one
    /// reads as deleted.
    fn advance_established(
        &mut self,
        snapshot: FileSnapshot,
        identities: &BTreeMap<PathBuf, RootIdentity>,
    ) -> bool {
        let DeletionAuthorityState::Established { authority, .. } = &mut self.authority else {
            return false;
        };
        if !authority.observed_through(identities) {
            return false;
        }
        authority.snapshot = snapshot;
        authority.generation = authority.generation.saturating_add(1);
        true
    }

    /// Detach the inherited names, absorbing an authority that just lost
    /// coverage of the configured roots.
    ///
    /// Those names were adjudicable a moment ago; letting them vanish would
    /// silently drop every pending stale delete. Retaining them as legacy
    /// keeps them fail-closed — never deleted by observation alone — while
    /// leaving them recoverable under explicit rebuild authority.
    fn retire_into_legacy(&mut self) -> Option<FileSnapshot> {
        let mut legacy = self.authority.take_legacy();
        if let DeletionAuthorityState::Established { authority, .. } = &self.authority {
            if !authority.snapshot.is_empty() {
                legacy
                    .get_or_insert_with(FileSnapshot::new)
                    .extend(authority.snapshot.clone());
            }
        }
        legacy
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
    /// Returns an error if the watcher backend cannot be created or started,
    /// or if another caller's generation never leaves its transition.
    pub async fn start(&self, cx: &Cx) -> SearchResult<()> {
        self.start_bounded(cx, LIFECYCLE_TRANSITION_POLLS).await
    }

    /// [`Self::start`] with an explicit wait budget.
    ///
    /// The budget is the only difference from the public entry point, which is
    /// what lets a test reach the timeout branch in milliseconds instead of
    /// waiting out a production-sized bound. The shipping path is the same
    /// code with the same constant it always used.
    async fn start_bounded(&self, cx: &Cx, max_waits: usize) -> SearchResult<()> {
        if cx.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "watch.start".to_owned(),
                reason: "cancel requested before start".to_owned(),
            });
        }

        let mut observed_generation = None;
        let mut waits = 0_usize;
        loop {
            let decision = lock_or_recover(&self.control).start_decision(observed_generation);
            match decision {
                StartDecision::AlreadyRunning | StartDecision::ObservedGenerationCompleted => {
                    return Ok(());
                }
                StartDecision::Wait(generation) => {
                    // Bounded: a generation wedged in `Starting` or `Stopping`
                    // must surface as a failure, not as a caller that polls a
                    // millisecond at a time until the process ends.
                    waits = waits.saturating_add(1);
                    if waits > max_waits {
                        return Err(watcher_task_error(format!(
                            "watcher start timed out waiting for generation {generation} to settle"
                        )));
                    }
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
    /// have been joined, or a timeout if the generation another caller owns
    /// never leaves its transition.
    pub async fn stop_checked(&self, cx: &Cx) -> SearchResult<()> {
        self.stop_checked_bounded(cx, LIFECYCLE_TRANSITION_POLLS)
            .await
    }

    /// [`Self::stop_checked`] with an explicit wait budget. See
    /// [`Self::start_bounded`] for why the budget is a parameter.
    async fn stop_checked_bounded(&self, cx: &Cx, max_waits: usize) -> SearchResult<()> {
        let mut waits = 0_usize;
        loop {
            match lock_or_recover(&self.control).stop_decision() {
                StopDecision::Done => return Ok(()),
                StopDecision::Wait => {
                    // Bounded, for the same reason the start path is: the
                    // owner of a wedged generation needs an error it can act
                    // on, not a caller that never returns.
                    waits = waits.saturating_add(1);
                    if waits > max_waits {
                        return Err(watcher_task_error(
                            "watcher stop timed out waiting for its generation to settle",
                        ));
                    }
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
                // Caller-driven path: its `Cx` is the cancellation channel.
                &|| cx.is_cancel_requested(),
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
        collect_snapshot_from_roots(&self.roots, &self.discovery, &|| false)
    }

    /// Grant one-shot authority to rebuild deletion authority over names the
    /// watcher inherited but never observed.
    ///
    /// A snapshot restored from crash recovery, or handed to
    /// [`Self::build_catchup_events`], records which files were indexed but not
    /// which roots they were read through. No amount of later observation can
    /// supply that missing evidence: a file absent today is indistinguishable
    /// from a root that was replaced since, so the watcher refuses to delete
    /// those names on its own. This is the operator saying that the roots in
    /// the configuration really are the roots that corpus was built from, and
    /// it is spent by the next pass that holds authority and adjudicates them.
    pub fn authorize_deletion_authority_rebuild(&self) {
        lock_or_recover(&self.reconciliation).authorize_rebuild();
    }

    /// Whether the watcher is holding inherited names it refuses to delete.
    ///
    /// True until [`Self::authorize_deletion_authority_rebuild`] is granted and
    /// an adjudicating pass settles them.
    #[must_use]
    pub fn holds_unverified_legacy_baseline(&self) -> bool {
        lock_or_recover(&self.reconciliation)
            .authority
            .legacy()
            .is_some_and(|legacy| !legacy.is_empty())
    }

    /// Build catch-up events by diffing prior and current snapshots.
    ///
    /// An incomplete scan still reports the creates and modifies it observed —
    /// those paths demonstrably exist — but derives no deletes, and leaves
    /// reconciliation required so a later complete scan can settle the
    /// difference.
    ///
    /// `previous` is *inherited* evidence: it names files without recording the
    /// roots they were read through. It is retained rather than adjudicated, so
    /// no call here can delete from it until
    /// [`Self::authorize_deletion_authority_rebuild`] says the roots are the
    /// same ones.
    ///
    /// # Errors
    ///
    /// Returns errors from current snapshot collection.
    pub fn build_catchup_events(&self, previous: &FileSnapshot) -> SearchResult<Vec<WatchEvent>> {
        let (current, mut completeness) = self.collect_snapshot()?;
        let baseline = {
            let mut state = lock_or_recover(&self.reconciliation);
            state.inherit_legacy(previous);
            if let Some(authority) = state.established_authority(&self.roots) {
                // A swapped root reads as a complete scan of an empty tree;
                // only the identities bound to the authority tell them apart,
                // and the check has to happen before any delete is derived.
                completeness.reject_swapped_roots(&authority.root_identities);
            }
            if completeness.is_complete() && completeness.identity_is_trustworthy() {
                let plan = state.plan_complete_pass(&self.roots, &completeness);
                let baseline = plan.deletion_baseline.clone().unwrap_or_default();
                // Nothing has been applied between the plan and here, so the
                // lineage cannot have moved and the commit cannot be refused;
                // the state still decides, not this call site.
                let committed = state.commit_complete_pass(
                    &plan,
                    current.clone(),
                    completeness.root_identities().clone(),
                );
                if !committed {
                    state.require_full_scan();
                }
                baseline
            } else {
                // Short or unidentifiable: report what was seen, adjudicate
                // nothing, and leave a complete pass owed.
                state.require_full_scan();
                FileSnapshot::new()
            }
        };
        Ok(Self::diff_snapshots(
            &baseline,
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
        // Completeness alone is not enough. A scan on a platform that cannot
        // identify its roots is a truthful listing of what it saw, but it can
        // never rule out that the roots were replaced, so absence in it is not
        // evidence of deletion.
        let derive_deletes = completeness.is_complete() && completeness.identity_is_trustworthy();
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

    // The startup scan is a full walk too, so it observes the stop flag on the
    // same bounded interval as every other scan.
    let startup_scan = collect_snapshot_from_roots(&context.roots, &context.discovery, &|| {
        context.stop.is_requested()
    });
    let (baseline, baseline_completeness) = match startup_scan {
        Ok(scan) => scan,
        // A stop during startup is an ordinary shutdown, not a watcher fault.
        // Classifying it as a subsystem error would surface a spurious failure
        // from every normal stop that lands mid-scan.
        Err(_) if context.stop.is_requested() => return Ok(()),
        Err(error) => return Err(error),
    };
    {
        let mut reconciliation = lock_or_recover(&context.reconciliation);
        // Startup applies no operation, so it may only seed authority where
        // there is nothing to adjudicate: no inherited names, no candidate, no
        // held authority. Anything else — a retained baseline holding
        // suppressed deletes, a probation candidate awaiting confirmation — is
        // promoted only by a pass that can actually apply the deletes it
        // derives. A restart is not evidence that those files came back.
        if baseline_completeness.is_complete() && baseline_completeness.identity_is_trustworthy() {
            let seeded = reconciliation
                .seed_initial_authority(baseline, baseline_completeness.root_identities().clone());
            if !seeded {
                reconciliation.require_full_scan();
            }
        } else {
            // A short startup scan is not deletion authority. Take it as a
            // working set only if nothing better exists, leave the authority
            // state untouched so no delete can be derived from it, and require
            // a rescan.
            if !reconciliation.baseline_initialized {
                reconciliation.indexed_snapshot = baseline;
                reconciliation.baseline_initialized = true;
            }
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
            return Err(cancelled_ingest_error(cx));
        }

        // A full rescan can take arbitrarily long on a large tree, so a stop
        // that arrived while the previous batch was applying must be honoured
        // before starting one rather than after it finishes. It is not,
        // however, a licence to discard work already produced: the producer
        // flushes its debounce buffer into the ready queue precisely because
        // those events are real, and returning here dropped every one of them.
        // `return`, not `break`: this loop is the function's diverging tail
        // expression, so breaking out of it would leave no value for a
        // `SearchResult<()>`.
        if stop.is_requested() {
            return drain_final_batches(
                cx,
                roots,
                discovery,
                ingest,
                ready_batches,
                stats,
                reconciliation,
                producer_done,
                snapshot_collector,
            )
            .await;
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
                // The walk itself now admits the stop, so a rescan over a
                // large tree cannot hold `stop_checked` for its duration.
                &|| stop.is_requested(),
            )
            .await
            {
                Ok(()) => {
                    reconciliation_attempts = 0;
                    continue;
                }
                Err(error) if is_retryable_error(&error) && !cx.is_cancel_requested() => {
                    if stop.is_requested() {
                        // Abandoned for a stop rather than failed: an
                        // unresolved pass applies nothing, and the requirement
                        // it leaves behind is what the next start honours.
                        // Reporting an ordinary shutdown as a terminal watcher
                        // failure is what made every stop that landed mid-scan
                        // surface an error. The loop head turns this into the
                        // shutdown drain.
                        continue;
                    }
                    stats.add_error();
                    reconciliation_attempts = reconciliation_attempts.saturating_add(1);
                    // A root that stays unavailable fails every attempt, so
                    // the attempt bound is what stops a persistent
                    // incompleteness from looping here forever, and the stop
                    // flag is what keeps `stop_checked` prompt while it does.
                    if reconciliation_attempts >= MAX_RECONCILIATION_ATTEMPTS {
                        return Err(error);
                    }
                    warn!(error = %error, "watcher reconciliation failed; retrying full rescan");
                    // Interruptible and stop-aware: a stop requested during the
                    // backoff ends the wait instead of serving it out, and the
                    // loop head then drains rather than serving out a retry.
                    let _stopped = stop.wait_or_stopped(IDLE_POLL);
                    continue;
                }
                Err(error) => {
                    // Cancellation, or a failure no retry can clear.
                    stats.add_error();
                    return Err(error);
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

/// The cancellation this task must report, carrying the caller's own reason.
fn cancelled_ingest_error(cx: &Cx) -> SearchError {
    SearchError::Cancelled {
        phase: "watch.ingest".to_owned(),
        reason: cx.cancel_reason().map_or_else(
            || "caller-owned ingest task cancelled".to_owned(),
            |reason| reason.to_string(),
        ),
    }
}

/// Apply what is already queued before a stopped generation exits.
///
/// The producer's shutdown path deliberately flushes its debounce buffer into
/// the ready queue — those events are observed filesystem changes that nothing
/// else will re-derive — and returning the moment a stop was seen discarded
/// every one of them. The drain is bounded on both axes, how long it waits for
/// that flush and how many batches it will apply, so a stop stays prompt.
///
/// It deliberately starts no rescan. When one is already owed the queued
/// events are folded into the pending candidate set instead of being applied
/// ahead of it, which keeps the authoritative-rescan ordering the loop above
/// depends on.
#[allow(clippy::too_many_arguments)]
async fn drain_final_batches(
    cx: &Cx,
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    ingest: &dyn WatchIngestPipeline,
    ready_batches: &ReadyBatchQueue,
    stats: &WatcherStatsInner,
    reconciliation: &ReconciliationTracker,
    producer_done: &AtomicBool,
    snapshot_collector: &SnapshotCollector,
) -> SearchResult<()> {
    const PRODUCER_FLUSH_POLLS: usize = 2_000;
    const MAX_FINAL_BATCHES: usize = 4_096;
    const DRAIN_POLL: Duration = Duration::from_millis(1);

    // The final flush reaches the queue only once the producer has left its
    // loop, so waiting for it is what makes this drain complete rather than
    // racy — and the bound is what keeps a producer that never finishes from
    // holding the stop open.
    for _ in 0..PRODUCER_FLUSH_POLLS {
        if producer_done.load(Ordering::Acquire) {
            break;
        }
        if cx.is_cancel_requested() {
            return Err(cancelled_ingest_error(cx));
        }
        asupersync::time::sleep(cx.now(), DRAIN_POLL).await;
    }

    for _ in 0..MAX_FINAL_BATCHES {
        // Cancellation is abortive where a stop is graceful: it must not be
        // served out by finishing the queue first.
        if cx.is_cancel_requested() {
            return Err(cancelled_ingest_error(cx));
        }
        if lock_or_recover(reconciliation).required {
            fold_queue_into_reconciliation(ready_batches, reconciliation);
            return Ok(());
        }
        let Some(mut lease) = PendingBatchLease::acquire(ready_batches, reconciliation) else {
            return Ok(());
        };
        let prepared = prepare_event_batch(discovery, lease.events());
        if prepared.ops.is_empty() {
            stats.add_skipped(prepared.skipped);
            lease.commit();
            continue;
        }
        let events = lease.events().to_vec();
        lease.begin_live_apply();
        match ingest.apply_batch(cx, &prepared.ops).await {
            Ok(reindexed) => {
                match record_successful_events(
                    roots,
                    discovery,
                    reconciliation,
                    &events,
                    snapshot_collector,
                ) {
                    Ok(()) => {
                        lease.commit();
                        let outcome = prepared.outcome(reindexed);
                        stats.add_reindexed(outcome.reindexed);
                        stats.add_skipped(outcome.skipped);
                    }
                    Err(error) => {
                        // The batch landed; only the bookkeeping scan behind it
                        // failed. Dropping the lease past the mutation boundary
                        // leaves a rescan owed, which is what settles it.
                        drop(lease);
                        stats.add_error();
                        if is_retryable_error(&error) && !cx.is_cancel_requested() {
                            return Ok(());
                        }
                        return Err(error);
                    }
                }
            }
            Err(error) => {
                stats.add_error();
                drop(lease);
                warn!(
                    error = %error,
                    "watcher ingest failed while draining a stopped generation"
                );
                if is_retryable_error(&error) && !cx.is_cancel_requested() {
                    // Retrying is unbounded work during a shutdown, and the
                    // dropped lease has already made the rescan owed.
                    return Ok(());
                }
                return Err(error);
            }
        }
    }

    // Bound reached: what is still queued becomes the next pass's work rather
    // than work that silently disappeared.
    fold_queue_into_reconciliation(ready_batches, reconciliation);
    Ok(())
}

/// Move everything still queued into the pending candidate set, so a pass that
/// can adjudicate it sees it.
fn fold_queue_into_reconciliation(
    ready_batches: &ReadyBatchQueue,
    reconciliation: &ReconciliationTracker,
) {
    let queued = {
        let mut queue = lock_or_recover(ready_batches);
        queue.drain(..).flatten().collect::<Vec<_>>()
    };
    if queued.is_empty() {
        return;
    }
    lock_or_recover(reconciliation).require_for_events(&queued);
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
    abort: &dyn Fn() -> bool,
) -> SearchResult<()> {
    let (epoch, affected_paths, authority_identities) = {
        let state = lock_or_recover(reconciliation);
        (
            state.epoch,
            state.affected_paths.clone(),
            state
                .established_authority(roots)
                .map(|authority| authority.root_identities.clone()),
        )
    };
    // Every batch already visible here predates the authoritative snapshot
    // below. Dropping it is safe: the rescan covers its final filesystem
    // state, while batches produced after this clear remain queued and are
    // applied after the rescan.
    lock_or_recover(ready_batches).clear();
    let (current, mut completeness) = snapshot_collector(roots, discovery, abort)?;
    if let Some(identities) = authority_identities.as_ref() {
        // A swapped root reads as a complete scan of an empty tree; only the
        // identities bound to the authority distinguish it from a real one.
        completeness.reject_swapped_roots(identities);
    }
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
    if !completeness.is_complete() {
        // Applying the visible subset and returning `Ok` was the defect in the
        // first correction: it reindexed a partial tree, cleared nothing, and
        // reported success, so the caller had no reason to back off and the
        // rescan ran again immediately on the next pass. An unresolved rescan
        // is a retryable failure of the whole pass — no ops are applied here —
        // and `SubsystemError` is the classification `is_retryable_error`
        // already honours, so the ingest loop's bounded attempts, its
        // interruptible sleep, and its stop/cancel checks all apply unchanged.
        let mut state = lock_or_recover(reconciliation);
        state.required = true;
        drop(state);
        let unresolved = completeness
            .unresolved_paths()
            .map(Path::display)
            .map(|path| path.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        warn!(
            unresolved_paths = completeness.unresolved_count(),
            "watcher rescan could not resolve every path; applying nothing and retrying"
        );
        return Err(SearchError::SubsystemError {
            subsystem: "fsfs-watcher",
            source: Box::new(io::Error::other(format!(
                "authoritative rescan is incomplete; {} unresolved path(s): {unresolved}",
                completeness.unresolved_count()
            ))),
        });
    }

    // From here the scan is complete. What it may *conclude* is decided by the
    // authority state, not by this call site, and it is decided before a single
    // operation is applied.
    let scan_identities = completeness.root_identities().clone();
    let plan = {
        let state = lock_or_recover(reconciliation);
        state.plan_complete_pass(roots, &completeness)
    };
    match plan.outcome {
        PassOutcome::Adjudicated => {}
        PassOutcome::Probationary => warn!(
            "watcher recorded a probationary root observation; a confirming scan must match it \
             before deletions are derived"
        ),
        PassOutcome::Degraded => {
            warn!("watcher has no trustworthy root identity; upserting without deletion authority");
        }
    }

    if let Some(deletion_baseline) = plan.deletion_baseline.as_ref() {
        let mut deletion_candidates = deletion_baseline.keys().cloned().collect::<BTreeSet<_>>();
        // The paths this watcher's own unapplied events named are absences of
        // the same evidence class as the baseline's, and are adjudicated only
        // by a pass that holds authority. Unioning them into a baseline that
        // could not adjudicate anything is what let an authority naming nothing
        // still delete.
        if plan.adjudicates() {
            deletion_candidates.extend(affected_paths);
        }
        events.extend(
            deletion_candidates
                .into_iter()
                .filter(|path| !current.contains_key(path))
                .map(|path| WatchEvent::deleted(path, observed_at_ms)),
        );
    }

    // Telemetry is staged, not published per chunk: a pass that fails a later
    // chunk, or whose epoch advances under it, is retried in full, and counts
    // already published would then be counted a second time.
    let mut staged_reindexed = 0_usize;
    let mut staged_skipped = 0_usize;
    for event_batch in events.chunks(batch_size.max(1)) {
        if cx.is_cancel_requested() || abort() {
            return Err(SearchError::Cancelled {
                phase: "watch.reconcile".to_owned(),
                reason: cx.cancel_reason().map_or_else(
                    || "watcher reconciliation cancelled".to_owned(),
                    |reason| reason.to_string(),
                ),
            });
        }
        // The epoch must hold through the apply, not merely at the commit: a
        // concurrent mutation that advanced it means these events describe a
        // filesystem state that is no longer the one being reconciled.
        if lock_or_recover(reconciliation).epoch != epoch {
            return Err(SearchError::SubsystemError {
                subsystem: "fsfs-watcher",
                source: Box::new(io::Error::other(
                    "reconciliation epoch advanced during apply; rescanning",
                )),
            });
        }
        let prepared = prepare_event_batch(discovery, event_batch);
        if prepared.ops.is_empty() {
            staged_skipped = staged_skipped.saturating_add(prepared.skipped);
            continue;
        }
        let reindexed = ingest.apply_batch(cx, &prepared.ops).await?;
        let outcome = prepared.outcome(reindexed);
        staged_reindexed = staged_reindexed.saturating_add(outcome.reindexed);
        staged_skipped = staged_skipped.saturating_add(outcome.skipped);
    }

    // Only a complete pass reaches here; the incomplete one returned above
    // before applying anything.
    let mut state = lock_or_recover(reconciliation);
    if state.epoch != epoch {
        // The tree moved under this pass. Its operations landed, but its
        // conclusion describes a state that no longer exists, so nothing is
        // installed and nothing is counted — the next pass does both.
        return Ok(());
    }
    if !state.commit_complete_pass(&plan, current, scan_identities) {
        // Authority was rebound while this pass was applying. Fail closed: the
        // conclusion is dropped and a fresh pass is owed.
        state.require_full_scan();
        return Ok(());
    }
    drop(state);
    stats.add_reindexed(staged_reindexed);
    stats.add_skipped(staged_skipped);
    Ok(())
}

fn record_successful_events(
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    reconciliation: &ReconciliationTracker,
    events: &[WatchEvent],
    snapshot_collector: &SnapshotCollector,
) -> SearchResult<()> {
    // Read the lineage this batch is being recorded against *before* the scan,
    // so the checks below compare the authority that was in force when the
    // batch was applied against the one still in force now.
    let (expected_epoch, expected_generation, expected_identities) = {
        let state = lock_or_recover(reconciliation);
        (
            state.epoch,
            state.established_generation(),
            state.established_identities().cloned(),
        )
    };
    let (current, completeness) = snapshot_collector(roots, discovery, &|| false)?;
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

    if !completeness.is_complete() || !completeness.identity_is_trustworthy() {
        return Ok(());
    }
    let observed_identities = completeness.root_identities();
    let roots_unchanged = expected_identities.as_ref() == Some(observed_identities);
    if expected_generation.is_some() && !roots_unchanged {
        // The roots this batch was recorded through are not the roots the
        // authority was established through. Rebinding here is what let a
        // swapped root become authority without any pass ever adjudicating it,
        // after which every file of the old root reads as deleted. Fail closed
        // and hand it to a pass that can prove what happened.
        state.require_full_scan();
        return Ok(());
    }
    if expected_generation.is_none()
        || state.epoch != expected_epoch
        || state.established_generation() != expected_generation
    {
        // No authority to advance, or the lineage moved while this scan ran.
        // Establishing authority is a pass's job, not a batch's: this path
        // applies no deletes and cannot adjudicate anything, so it may only
        // advance an authority that already exists and is still the same one.
        return Ok(());
    }
    // A successful batch over a complete, trustworthy scan advances the
    // deletion authority as one value — the snapshot and the identities it was
    // read through together. Advancing the working set alone would leave the
    // authority describing a tree that no longer exists, and a later pass would
    // derive deletes from it.
    let updated = state.indexed_snapshot.clone();
    if !state.advance_established(updated, observed_identities) {
        state.require_full_scan();
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

/// Walk every root, abandoning the scan promptly when `should_abort` says so.
///
/// The predicate is consulted once per directory rather than once per root:
/// a single root can hold an unbounded tree, so checking only between roots
/// would let one scan block a stop for as long as the walk takes. Abandoning
/// returns a typed interruption instead of a short snapshot, so a stop can
/// never be mistaken for a complete scan of a smaller tree.
fn collect_snapshot_from_roots(
    roots: &[PathBuf],
    discovery: &DiscoveryConfig,
    should_abort: &dyn Fn() -> bool,
) -> SearchResult<(FileSnapshot, ScanCompleteness)> {
    let mut snapshot = FileSnapshot::new();
    let mut completeness = ScanCompleteness::default();
    let mount_table = build_mount_table(discovery);
    // Distinct roots only. A configuration that names the same root twice
    // otherwise walks it twice for byte-identical results, and the second walk
    // is a second window in which the root can be swapped under the scan.
    for root in distinct_roots(roots) {
        if should_abort() {
            return Err(scan_interrupted_error());
        }
        collect_snapshot_for_root(
            root,
            discovery,
            Some(&mount_table),
            &mut snapshot,
            &mut completeness,
            should_abort,
        )?;
    }
    Ok((snapshot, completeness))
}

/// Points inside the walk a test may observe, so a scan can be parked
/// deterministically instead of raced against a sleep.
///
/// Compiled out of shipping builds. The observer receives the path so a test
/// can act only on its own fixture and never perturb a concurrently running
/// test in the same binary.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanProbe {
    /// A directory is about to be listed.
    DirectoryEntered,
    /// An entry has been listed and is about to be stat'd.
    EntryListed,
}

#[cfg(test)]
type ScanObserver = Arc<dyn Fn(ScanProbe, &Path) + Send + Sync>;

/// Registered scan observers, each owned by exactly one live guard.
///
/// A single global slot was two bugs at once: two tests running concurrently
/// in the same binary clobbered each other's observer, and clearing it at the
/// end of one test removed the other's. A test that panicked left its observer
/// installed for every test that followed. The registry keys each registration
/// to its own guard, so installing is additive and removal is scoped to the
/// registration that performed it — including on unwind.
#[cfg(test)]
static SCAN_OBSERVERS: Mutex<Vec<(u64, ScanObserver)>> = Mutex::new(Vec::new());

#[cfg(test)]
static NEXT_SCAN_OBSERVER_ID: AtomicU64 = AtomicU64::new(0);

/// Lifetime of one scan-observer registration.
#[cfg(test)]
struct ScanObserverGuard {
    id: u64,
}

#[cfg(test)]
impl Drop for ScanObserverGuard {
    fn drop(&mut self) {
        lock_or_recover(&SCAN_OBSERVERS).retain(|(id, _)| *id != self.id);
    }
}

#[cfg(test)]
#[must_use = "the observer is removed when the guard is dropped"]
fn install_scan_observer(observer: ScanObserver) -> ScanObserverGuard {
    let id = NEXT_SCAN_OBSERVER_ID.fetch_add(1, Ordering::AcqRel);
    lock_or_recover(&SCAN_OBSERVERS).push((id, observer));
    ScanObserverGuard { id }
}

#[cfg(test)]
fn scan_probe(probe: ScanProbe, path: &Path) {
    // Cloned out before invoking so an observer may itself block without
    // holding the registry lock against another thread.
    let observers = lock_or_recover(&SCAN_OBSERVERS)
        .iter()
        .map(|(_, observer)| Arc::clone(observer))
        .collect::<Vec<_>>();
    for observer in observers {
        observer(probe, path);
    }
}

/// Typed outcome of a scan abandoned for a stop or cancellation.
///
/// Retryable by classification, so a scan abandoned for a stop that is later
/// cleared simply runs again; the loop's stop checks end the task first when
/// the stop is real.
fn scan_interrupted_error() -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs-watcher",
        source: Box::new(io::Error::other(
            "authoritative scan abandoned before completion",
        )),
    }
}

fn collect_snapshot_for_root(
    root: &Path,
    discovery: &DiscoveryConfig,
    mount_table: Option<&MountTable>,
    snapshot: &mut FileSnapshot,
    completeness: &mut ScanCompleteness,
    should_abort: &dyn Fn() -> bool,
) -> SearchResult<()> {
    // One stat, not `exists()` then a walk. `exists()` answers a question
    // about a past instant and swallows every error into `false`: a root that
    // was merely unreadable reported "absent", and between that answer and the
    // walk the root could be replaced anyway. Classifying this single call is
    // both the error fix and the TOCTOU fix.
    // Check-use-recheck, and deliberately not called descriptor-bound: `std`
    // offers no way to enumerate a directory from an open handle, so the walk
    // below still addresses children by path. Opening the root pins the object
    // whose identity is read, and the recheck after the walk fails the scan if
    // that object stopped being the one at this path while we were inside it.
    // The window is closed by detection, not by construction.
    let opened_root = match fs::File::open(root) {
        Ok(handle) => handle,
        Err(error) if is_ignorable_walk_error(&error) => {
            // Absent, unreadable, or interrupted are all "could not look",
            // never "there is nothing there".
            completeness.record_unresolved(root);
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let root_metadata = match opened_root.metadata() {
        Ok(metadata) => metadata,
        Err(error) if is_ignorable_walk_error(&error) => {
            completeness.record_unresolved(root);
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let opened_identity = RootIdentity::of(&root_metadata);
    completeness.record_root_identity(root, opened_identity);

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
        // Bounded interval: one check per directory entered, so a stop is
        // observed within a single directory listing however large the tree.
        #[cfg(test)]
        scan_probe(ScanProbe::DirectoryEntered, &dir_path);
        if should_abort() {
            return Err(scan_interrupted_error());
        }
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

            // Between listing and stat: the window a file may vanish in.
            #[cfg(test)]
            scan_probe(ScanProbe::EntryListed, &path);
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

    // Recheck: the object we opened must still be the object at this path.
    // Comparing the still-open handle against a fresh lookup catches a root
    // unmounted, renamed away, or replaced at any point during the walk — the
    // cases whose snapshots are indistinguishable from a real empty tree.
    let reopened_identity = fs::File::open(root)
        .and_then(|handle| handle.metadata())
        .ok()
        .as_ref()
        .and_then(RootIdentity::of);
    let still_open_identity = opened_root
        .metadata()
        .ok()
        .as_ref()
        .and_then(RootIdentity::of);
    // A platform with no identity is degraded, not unresolved: the scan is a
    // truthful listing and may upsert, it simply cannot authorize deletes.
    // Recording it unresolved instead would fail every pass on such a target.
    if opened_identity.is_some()
        && (still_open_identity != opened_identity || reopened_identity != opened_identity)
    {
        completeness.record_unresolved(root);
    }
    drop(opened_root);

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
        ReconciliationTracker, ScanCompleteness, ScanProbe, WatchBatchOutcome, WatchEvent,
        WatchEventKind, WatchIngestFuture, WatchIngestOp, WatchIngestPipeline,
        WatcherExecutionPolicy, WatcherLifecycle, WatcherStatsInner, WatcherStop,
        collect_snapshot_from_roots, drain_notify_channel, flush_pending_batches,
        install_scan_observer, is_retryable_error, normalize_file_key, now_millis,
        observe_pressure_transition, record_successful_events, run_authoritative_reconciliation,
        run_ingest_loop,
    };
    use crate::config::DiscoveryConfig;

    /// Read the real identities of `roots`, so a fixture that claims prior
    /// authority carries the same evidence a live scan would have produced.
    /// A fixture with invented or empty identities is a different state — the
    /// no-authority one — and must not be used to stand in for this one.
    fn authentic_root_identities(roots: &[PathBuf]) -> BTreeMap<PathBuf, super::RootIdentity> {
        roots
            .iter()
            .map(|root| {
                let metadata = fs::metadata(root).expect("identity fixture root must exist");
                let identity = super::RootIdentity::of(&metadata)
                    .expect("identity fixture requires a platform with root identity");
                (root.clone(), identity)
            })
            .collect()
    }

    /// Build a `DeletionAuthority` fixture bound to real roots.
    fn authority_over(snapshot: FileSnapshot, roots: &[PathBuf]) -> super::DeletionAuthority {
        super::DeletionAuthority {
            snapshot,
            root_identities: authentic_root_identities(roots),
            generation: 1,
        }
    }

    /// `WatcherStop::request` must publish the flag and the notify inside
    /// `wait_lock`, so a waiter cannot be skipped between its check and its
    /// park.
    ///
    /// The interleaving is forced rather than raced: the test holds `wait_lock`
    /// itself, which is exactly the state `wait_or_stopped` is in after it has
    /// read the flag as `false` and before it parks. A `request()` that
    /// publishes outside the lock returns immediately in that state — its
    /// store and its notify both land in the waiter's blind window, and the
    /// real waiter then sleeps the full backoff with stop already requested.
    ///
    /// The verdict is an observation taken *at* the publication, not a race
    /// between two threads. The observer runs at the flag store itself and
    /// asks whether `wait_lock` is held there: `try_lock` fails from the
    /// owning thread, so a publication inside the lock reports "held". A
    /// regression that moves the store back outside the lock moves the
    /// observer with it, `try_lock` then succeeds, and this fails. Single
    /// threaded on purpose — there is no scheduling for a false green to hide
    /// in.
    #[test]
    fn stop_request_publishes_only_while_holding_the_wait_lock() {
        let stop = Arc::new(WatcherStop::default());
        let observed = Arc::new(AtomicUsize::new(0));
        let held_at_publication = Arc::new(AtomicBool::new(false));

        let flag_already_set = Arc::new(AtomicBool::new(false));

        {
            let observed = Arc::clone(&observed);
            let held_at_publication = Arc::clone(&held_at_publication);
            let flag_already_set = Arc::clone(&flag_already_set);
            stop.set_publish_observer(Box::new(move |stop: &WatcherStop| {
                // `try_lock` from the thread that already owns the mutex
                // returns `WouldBlock`; an unheld mutex hands the guard over.
                let unheld = stop.wait_lock.try_lock().is_ok();
                held_at_publication.store(!unheld, Ordering::Release);
                // Pins the observation to the store seam from the other side:
                // an observer that had drifted after the store would see the
                // flag already true.
                flag_already_set.store(stop.is_requested(), Ordering::Release);
                observed.fetch_add(1, Ordering::AcqRel);
            }));
        }

        stop.request();

        assert_eq!(
            observed.load(Ordering::Acquire),
            1,
            "the publication boundary must be reached exactly once"
        );
        assert!(
            held_at_publication.load(Ordering::Acquire),
            "the stop flag was published without holding wait_lock; a waiter between its \
             flag check and its park would miss the notify and sleep the entire backoff"
        );
        assert!(
            !flag_already_set.load(Ordering::Acquire),
            "the observation ran after the store, so it no longer witnesses the seam it claims to"
        );
        assert!(stop.is_requested());
    }

    /// A wakeup that is not a stop must not end the backoff.
    ///
    /// This is the property a single `wait_timeout` cannot hold: a condvar may
    /// wake a waiter for no reason, and the previous form returned `false` on
    /// any such wakeup, which the caller reads as "the backoff elapsed". The
    /// notify below carries no stop — it is issued under `wait_lock` exactly
    /// as `request()` does, but without the store — so it is indistinguishable
    /// from a spurious wakeup. Pre-fix the waiter returns `false` there and
    /// this fails on `stopped`; post-fix it re-parks and only the real stop
    /// releases it.
    ///
    /// "Parked" is proven by two facts in sequence, with no sleeping or
    /// spinning. The park observer runs while the waiter still holds
    /// `wait_lock`, so receiving its signal proves the waiter is inside
    /// `wait_or_stopped` and past its flag check. This thread then blocks
    /// acquiring `wait_lock`, which cannot be granted until the waiter has
    /// released it *into* the condvar wait. Owning the lock therefore means
    /// the waiter is parked, and the notify issued under it is delivered.
    #[test]
    fn spurious_wakeup_does_not_shorten_the_backoff_and_stop_still_wakes_it() {
        const BACKOFF: Duration = Duration::from_secs(30);
        let stop = Arc::new(WatcherStop::default());
        // Signalled from inside the park boundary; waited on by this thread.
        let parking = Arc::new((Mutex::new(false), Condvar::new()));

        {
            let parking = Arc::clone(&parking);
            stop.set_park_observer(Box::new(move |_stop: &WatcherStop| {
                let (announced, ready) = &*parking;
                *super::lock_or_recover(announced) = true;
                ready.notify_all();
            }));
        }

        // Counts processed wakeups, and records the waiter returning. Both are
        // published under one lock so the test can wait for either without
        // sleeping.
        let progress = Arc::new((Mutex::new((0_usize, false)), Condvar::new()));
        {
            let progress = Arc::clone(&progress);
            stop.set_wake_observer(Box::new(move |_stop: &WatcherStop| {
                let (state, changed) = &*progress;
                super::lock_or_recover(state).0 += 1;
                changed.notify_all();
            }));
        }

        let waiter = {
            let stop = Arc::clone(&stop);
            let progress = Arc::clone(&progress);
            thread::spawn(move || {
                let started = Instant::now();
                let stopped = stop.wait_or_stopped(BACKOFF);
                let (state, changed) = &*progress;
                super::lock_or_recover(state).1 = true;
                changed.notify_all();
                (stopped, started.elapsed())
            })
        };

        {
            let (announced, ready) = &*parking;
            let mut announced = super::lock_or_recover(announced);
            while !*announced {
                announced = ready
                    .wait(announced)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
        {
            // Blocks until the waiter releases `wait_lock` into the condvar.
            let _parked = super::lock_or_recover(&stop.wait_lock);
            stop.wait_cv.notify_all();
        }
        assert!(
            !stop.is_requested(),
            "the spurious notify must not have requested a stop"
        );

        // Wait for the waiter to acknowledge the synthetic wakeup before the
        // real stop exists. Without this the real `request()` could land while
        // the waiter is still reacquiring `wait_lock`, and a one-shot
        // `wait_timeout` would then observe an already-true flag and report
        // `stopped` — passing while doing the very thing this rejects. The
        // acknowledgement is the predicate re-evaluation: one on entry, a
        // second for the synthetic notify.
        {
            let (state, changed) = &*progress;
            let mut observed = super::lock_or_recover(state);
            while observed.0 < 2 && !observed.1 {
                observed = changed
                    .wait(observed)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            assert!(
                !observed.1,
                "the waiter returned from a wakeup that carried no stop; the backoff was cut \
                 short before a real stop was ever requested"
            );
            assert!(
                observed.0 >= 2,
                "the synthetic wakeup was never processed, so nothing was exercised"
            );
        }

        stop.request();

        let (stopped, elapsed) = waiter.join().expect("stop waiter thread");
        assert!(
            stopped,
            "a wakeup carrying no stop ended the wait; the backoff was cut short"
        );
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
    use frankensearch_core::{SearchError, SearchResult};
    use notify::event::{CreateKind, ModifyKind, RenameMode};
    use notify::{Event, EventKind};
    use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
    use std::fs;
    use std::future::Future;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
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
            &|| false,
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
            &|| false,
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
            &|| false,
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

    /// `stop_checked` must not wait for a running scan, driven entirely
    /// through the public lifecycle.
    ///
    /// The watcher is started and stopped through `start`/`stop_checked`, with
    /// the scan parked deterministically inside the walk by a test probe
    /// rather than by a sleep. The park is released only once the stop has
    /// been published, so the walk's own poll is the only thing that can end
    /// the scan, and nothing may be applied afterwards.
    ///
    /// One runtime, and the release comes from a plain thread. The parked walk
    /// occupies the producer's own OS thread, and `stop_checked` blocks the
    /// runtime thread joining it, so anything that had to be *polled* to
    /// perform the release could never run — and a watcher started on one
    /// runtime and joined from another is joining a handle whose task that
    /// runtime no longer drives.
    #[test]
    fn public_stop_checked_releases_a_scan_parked_mid_walk() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().join("g3r-public-stop");
        fs::create_dir_all(root.join("nested")).expect("create nested root");
        fs::write(root.join("nested").join("indexed.rs"), "fn indexed() {}\n")
            .expect("write fixture");

        let pipeline = Arc::new(RecordingPipeline::default());
        let watcher = Arc::new(FsWatcher::new(
            vec![root.clone()],
            DiscoveryConfig::default(),
            Arc::clone(&pipeline) as Arc<dyn WatchIngestPipeline>,
        ));

        // Parked = "the walk reached this fixture's directory". Released only
        // by the releaser below, so the scan cannot finish on its own.
        let parked = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let _probe = {
            let parked = Arc::clone(&parked);
            let owned_root = root.clone();
            // Scoped to this guard: a concurrently running test's observer is
            // neither replaced nor removed by it, and a panic below still
            // uninstalls this one.
            install_scan_observer(Arc::new(move |probe: ScanProbe, path: &Path| {
                // Only this test's own fixture, so a concurrently running test
                // in the same binary is never perturbed.
                if probe != ScanProbe::DirectoryEntered || !path.starts_with(&owned_root) {
                    return;
                }
                let (state, changed) = &*parked;
                let mut state = lock_or_recover(state);
                state.0 = true;
                changed.notify_all();
                while !state.1 {
                    state = changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }))
        };

        let stop_result = run_on_runtime_task_with_result({
            let watcher = Arc::clone(&watcher);
            let parked = Arc::clone(&parked);
            |cx| async move {
                watcher.start(&cx).await.expect("public start");
                let stop = {
                    let control = lock_or_recover(&watcher.control);
                    match &control.lifecycle {
                        WatcherLifecycle::Running { stop, .. } => Some(Arc::clone(stop)),
                        WatcherLifecycle::Stopped
                        | WatcherLifecycle::Starting { .. }
                        | WatcherLifecycle::Stopping { .. } => None,
                    }
                }
                .expect("the started generation must be running");

                // Releases the walk only after the stop has actually been
                // published, so the walk's own poll is what ends the scan. It
                // is an ordinary thread because `stop_checked` joins the
                // producer thread and would not poll a task.
                let releaser = {
                    let parked = Arc::clone(&parked);
                    thread::spawn(move || {
                        {
                            let (lock, changed) = &*parked;
                            let mut state = lock_or_recover(lock);
                            while !state.0 {
                                state = changed
                                    .wait(state)
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                            }
                        }
                        while !stop.is_requested() {
                            thread::sleep(Duration::from_millis(1));
                        }
                        let (lock, changed) = &*parked;
                        lock_or_recover(lock).1 = true;
                        changed.notify_all();
                    })
                };

                let stop_result = watcher.stop_checked(&cx).await;
                releaser.join().expect("releaser thread");
                stop_result
            }
        });

        drop(_probe);
        assert!(
            stop_result.is_ok(),
            "stop_checked must not surface a failure for an ordinary stop: {stop_result:?}"
        );
        assert!(
            pipeline.all_ops().is_empty(),
            "a scan released by a stop must apply nothing, got {:?}",
            pipeline.all_ops()
        );
    }

    /// Superseded by the public lifecycle test above; retained as the
    /// component-level check that the loop applies nothing once stopped.
    #[test]
    fn stop_during_a_running_scan_ends_the_loop_without_finishing_it() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3r-stop-scan");
            fs::create_dir_all(&root).expect("create root");
            fs::write(root.join("indexed.rs"), "fn indexed() {}\n").expect("write fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            lock_or_recover(&reconciliation).require_full_scan();
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = Arc::new(WatcherStatsInner::default());
            let stop = Arc::new(WatcherStop::default());
            let producer_done = Arc::new(AtomicBool::new(true));

            // Signals "the scan has started" and blocks until released.
            let scanning = Arc::new((Mutex::new((false, false)), Condvar::new()));
            let completed_scans = Arc::new(AtomicUsize::new(0));

            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stats_for_task = Arc::clone(&stats);
            let reconciliation_for_task = Arc::clone(&reconciliation);
            let stop_for_task = Arc::clone(&stop);
            let scanning_for_task = Arc::clone(&scanning);
            let completed_for_task = Arc::clone(&completed_scans);
            let root_for_task = root.clone();

            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    let discovery = DiscoveryConfig::default();
                    let collector =
                        move |roots: &[PathBuf],
                              discovery: &DiscoveryConfig,
                              abort: &dyn Fn() -> bool| {
                            {
                                let (state, changed) = &*scanning_for_task;
                                let mut state = lock_or_recover(state);
                                state.0 = true;
                                changed.notify_all();
                                while !state.1 {
                                    state = changed
                                        .wait(state)
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                }
                            }
                            let result = collect_snapshot_from_roots(roots, discovery, abort);
                            completed_for_task.fetch_add(1, Ordering::AcqRel);
                            result
                        };
                    run_ingest_loop(
                        &child_cx,
                        &[root_for_task],
                        &discovery,
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                        &reconciliation_for_task,
                        100,
                        &producer_done,
                        &collector,
                    )
                    .await
                })
                .expect("spawn scan-stop task");

            // Wait for the scan to actually be in flight, then stop.
            {
                let (state, changed) = &*scanning;
                let mut state = lock_or_recover(state);
                while !state.0 {
                    state = changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
            stop.request();
            {
                let (state, changed) = &*scanning;
                lock_or_recover(state).1 = true;
                changed.notify_all();
            }

            task.join(&cx)
                .await
                .expect("scan-stop task terminal result")
                .expect("a stop during a scan is not a failure");
            assert!(
                pipeline.all_ops().is_empty(),
                "a stopped pass must not apply anything"
            );
        });
    }

    /// Inherited names are retained, never adjudicated by observation alone,
    /// and settled only by explicit rebuild authority.
    ///
    /// The caller's `previous` records *which* files were indexed and nothing
    /// about the roots they were read through. No later scan can supply that
    /// missing evidence: a file absent today is indistinguishable from a root
    /// replaced since, so any number of complete observations must still
    /// refuse the delete. What the watcher owes instead is retention — the
    /// names must survive so the operator's explicit grant can settle them.
    ///
    /// Every call after the first passes an EMPTY `previous`, so a delete can
    /// only ever come from the retained record and never from the argument.
    /// This drives the public API rather than the collect/diff helpers.
    #[test]
    fn catchup_retains_inherited_names_and_deletes_only_under_rebuild_authority() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().join("g3r-catchup");
        let blocked = temp.path().join("g3r-catchup-blocked");
        fs::create_dir_all(&root).expect("create root");
        fs::create_dir_all(&blocked).expect("create second root");
        fs::write(root.join("kept.rs"), "fn kept() {}\n").expect("write kept fixture");

        let watcher = FsWatcher::new(
            vec![root.clone(), blocked.clone()],
            DiscoveryConfig::default(),
            Arc::new(NoopWatchIngestPipeline),
        );

        // The caller's record names a file that no longer exists, while a root
        // is unavailable.
        let indexed_elsewhere = root.join("already-gone.rs");
        let previous = FileSnapshot::from([(indexed_elsewhere.clone(), 5)]);
        let moved = temp.path().join("g3r-catchup-blocked-moved");
        fs::rename(&blocked, &moved).expect("move the second root aside");

        let events = watcher
            .build_catchup_events(&previous)
            .expect("catch-up still reports what it can see");
        assert!(
            !events
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted),
            "an incomplete catch-up must derive no deletes, got {events:?}"
        );

        {
            let state = lock_or_recover(&watcher.reconciliation);
            assert!(state.required, "an incomplete catch-up stays required");
            assert!(
                matches!(
                    &state.authority,
                    super::DeletionAuthorityState::UnverifiedLegacy { .. }
                ),
                "names with no identity evidence behind them must be held in the \
                 unverified state, not in the one that authorizes deletion"
            );
            assert_eq!(
                state.authority.legacy(),
                Some(&previous),
                "the caller's record must be retained so it stays settleable"
            );
        }

        // Completeness returns, twice. Both passes see every root, and both
        // still refuse: two identical observations prove the roots were stable
        // across them, not that they are the roots this record came from.
        fs::rename(&moved, &blocked).expect("restore the second root");
        for pass in 0..2 {
            let observed = watcher
                .build_catchup_events(&FileSnapshot::new())
                .expect("complete catch-up");
            assert!(
                !observed
                    .iter()
                    .any(|event| event.kind == WatchEventKind::Deleted),
                "complete observation {pass} must not adjudicate an inherited name, got {observed:?}"
            );
            assert!(
                watcher.holds_unverified_legacy_baseline(),
                "the inherited name must still be retained after pass {pass}"
            );
        }

        // The operator states that the configured roots really are the roots
        // that corpus was built from. Only now is the absence adjudicable.
        watcher.authorize_deletion_authority_rebuild();
        let recovered = watcher
            .build_catchup_events(&FileSnapshot::new())
            .expect("catch-up under rebuild authority");
        assert!(
            recovered
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted
                    && event.path == indexed_elsewhere),
            "the retained name must be settled by the explicit grant, and it can only \
             have come from retention because the argument was empty, got {recovered:?}"
        );
        assert!(
            !watcher.holds_unverified_legacy_baseline(),
            "a settled record is no longer held, so the one-shot grant cannot be replayed"
        );

        // And the grant really is one-shot: a fresh inherited name after it is
        // spent is refused again.
        let later = root.join("indexed-after-the-grant.rs");
        let refused = watcher
            .build_catchup_events(&FileSnapshot::from([(later.clone(), 7)]))
            .expect("catch-up after the grant is spent");
        assert!(
            !refused
                .iter()
                .any(|event| event.kind == WatchEventKind::Deleted && event.path == later),
            "the spent grant must not authorize the next inherited name, got {refused:?}"
        );
    }

    /// A probationary pass must leave the legacy baseline and the probation
    /// record exactly as it found them.
    ///
    /// This is the assertion whose absence let the epilogue adopt the current
    /// snapshot on the very pass that had deliberately derived no deletes,
    /// destroying the baseline holding every stale delete. It checks what the
    /// pass did *not* change, which no other test here did.
    #[test]
    fn probationary_pass_preserves_the_legacy_baseline_and_stays_owed() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3s-probation");
            fs::create_dir_all(&root).expect("create root");
            fs::write(root.join("present.rs"), "fn present() {}\n").expect("write fixture");

            // Legacy baseline naming a file that is already gone, with no
            // identities behind it: the crash-recovery shape.
            let stale = root.join("gone-before-restart.rs");
            let legacy = FileSnapshot::from([(stale.clone(), 3)]);
            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker = Arc::new(Mutex::new(ReconciliationState {
                indexed_snapshot: legacy.clone(),
                baseline_initialized: true,
                required: true,
                affected_paths: BTreeSet::from([stale.clone()]),
                epoch: 0,
                authority: super::DeletionAuthorityState::UnverifiedLegacy {
                    legacy: legacy.clone(),
                },
                rebuild_authorized: false,
                unsettled_passes: 0,
            }));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let roots = vec![root.clone()];

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &DiscoveryConfig::default(),
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("a probationary pass still succeeds; it simply derives no deletes");

            let state = lock_or_recover(&reconciliation);
            assert!(
                !pipeline
                    .all_ops()
                    .iter()
                    .any(|op| matches!(op, WatchIngestOp::Delete { .. })),
                "a first observation must derive no deletes, got {:?}",
                pipeline.all_ops()
            );
            assert_eq!(
                state.indexed_snapshot, legacy,
                "the legacy baseline must survive the probationary pass untouched"
            );
            assert_eq!(
                state.authority.legacy(),
                Some(&legacy),
                "the inherited names must be carried forward, not consumed by a pass that \
                 adjudicated nothing"
            );
            assert!(
                matches!(
                    &state.authority,
                    super::DeletionAuthorityState::Probationary { .. }
                ),
                "a first trustworthy observation is a candidate, never authority"
            );
            assert!(state.required, "a second scan is still owed");
            assert!(
                state.affected_paths.contains(&stale),
                "pending delete candidates must not be forgotten by a pass that could not \
                 adjudicate them"
            );
        });
    }

    /// An authority whose identity map cannot cover the configured roots is
    /// not authority, and the names it holds are retired into the state that
    /// refuses to delete them.
    ///
    /// A baseline restored from crash recovery names files and records nothing
    /// about the roots they were read through, so nothing can rule out that
    /// those roots were replaced underneath it. The pass must derive no
    /// deletion, must keep the names rather than dropping them, and must not
    /// leave the watcher believing it holds authority.
    #[test]
    fn uncovered_authority_is_demoted_instead_of_authorizing_deletion() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3r-no-authority");
            fs::create_dir_all(&root).expect("create root");

            let pipeline = Arc::new(RecordingPipeline::default());
            // A baseline naming indexed files, with no root identities: the
            // shape a crash-recovery restore produces.
            let orphaned = root.join("restored-from-crash.rs");
            let held = FileSnapshot::from([(orphaned.clone(), 9)]);
            let reconciliation: ReconciliationTracker = Arc::new(Mutex::new(ReconciliationState {
                indexed_snapshot: held.clone(),
                baseline_initialized: true,
                required: true,
                affected_paths: BTreeSet::new(),
                epoch: 0,
                // The negative case: an authority that names files while
                // carrying no identity able to detect a swap of the root they
                // were read through.
                authority: super::DeletionAuthorityState::Established {
                    authority: super::DeletionAuthority {
                        snapshot: held.clone(),
                        root_identities: BTreeMap::new(),
                        generation: 1,
                    },
                    legacy: None,
                },
                rebuild_authorized: false,
                unsettled_passes: 0,
            }));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();

            run_authoritative_reconciliation(
                &cx,
                std::slice::from_ref(&root),
                &DiscoveryConfig::default(),
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("the pass concludes; it simply concludes that it may not delete");

            assert!(
                pipeline.all_ops().is_empty(),
                "nothing may be applied, and above all nothing deleted, got {:?}",
                pipeline.all_ops()
            );
            let state = lock_or_recover(&reconciliation);
            assert!(state.required, "a confirming pass is owed");
            assert!(
                state
                    .established_authority(std::slice::from_ref(&root))
                    .is_none(),
                "a map that cannot cover the configured roots is not deletion authority"
            );
            assert_eq!(
                state.authority.legacy(),
                Some(&held),
                "the names it held must be retired into the unverified state, not dropped"
            );
        });
    }

    /// An authority naming nothing must not delete the paths its own pending
    /// events named.
    ///
    /// This is the exemption that made emptiness look harmless: the candidate
    /// set a pass deletes from is the baseline *plus* `affected_paths`, so an
    /// authority with an empty snapshot and no identity at all could still
    /// delete — with zero evidence that the roots were ever the right ones.
    /// The control below proves the very same fixture does derive that delete
    /// once the identities genuinely cover the roots.
    #[test]
    fn empty_authority_without_identity_cannot_delete_its_affected_paths() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g4-empty-authority");
            fs::create_dir_all(&root).expect("create root");
            let vanished = root.join("named-by-a-failed-batch.rs");
            let roots = vec![root.clone()];

            let fixture =
                |identities: BTreeMap<PathBuf, super::RootIdentity>| -> ReconciliationTracker {
                    Arc::new(Mutex::new(ReconciliationState {
                        indexed_snapshot: FileSnapshot::new(),
                        baseline_initialized: true,
                        required: true,
                        affected_paths: BTreeSet::from([vanished.clone()]),
                        epoch: 0,
                        authority: super::DeletionAuthorityState::Established {
                            authority: super::DeletionAuthority {
                                snapshot: FileSnapshot::new(),
                                root_identities: identities,
                                generation: 1,
                            },
                            legacy: None,
                        },
                        rebuild_authorized: false,
                        unsettled_passes: 0,
                    }))
                };

            let deletes_for = |pipeline: &RecordingPipeline| {
                pipeline
                    .all_ops()
                    .into_iter()
                    .filter_map(|op| match op {
                        WatchIngestOp::Delete { path, .. } => Some(path),
                        WatchIngestOp::Upsert { .. } => None,
                    })
                    .collect::<Vec<_>>()
            };

            let refusing = Arc::new(RecordingPipeline::default());
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &DiscoveryConfig::default(),
                refusing.as_ref(),
                &fixture(BTreeMap::new()),
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("the pass concludes without adjudicating");
            assert!(
                deletes_for(&refusing).is_empty(),
                "an authority with no identity must not delete an affected path, got {:?}",
                deletes_for(&refusing)
            );

            // Control: identical fixture, real identities. If this did not
            // delete, the assertion above would be vacuous.
            let adjudicating = Arc::new(RecordingPipeline::default());
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &DiscoveryConfig::default(),
                adjudicating.as_ref(),
                &fixture(authentic_root_identities(&roots)),
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("a covering authority adjudicates");
            assert!(
                deletes_for(&adjudicating).contains(&vanished),
                "control: a covering authority does derive the affected-path delete, got {:?}",
                deletes_for(&adjudicating)
            );
        });
    }

    /// One reachable root plus one unavailable root must not reindex the half
    /// it can see. The pass fails retryably, applies nothing, stays required,
    /// and does not stall a stop.
    #[test]
    fn unavailable_root_fails_the_whole_pass_and_stops_promptly() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let present_root = temp.path().join("g3f-present");
            let absent_root = temp.path().join("g3f-absent");
            fs::create_dir_all(&present_root).expect("create present root");
            let visible = present_root.join("visible.rs");
            fs::write(&visible, "fn visible() {}\n").expect("write visible fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let roots = vec![present_root.clone(), absent_root];

            let error = run_authoritative_reconciliation(
                &cx,
                &roots,
                &DiscoveryConfig::default(),
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect_err("an unresolved root must fail the pass");

            assert!(
                is_retryable_error(&error),
                "the outcome must reach the loop's retry path, got {error:?}"
            );
            assert!(
                pipeline.all_ops().is_empty(),
                "no subset may be reindexed from an incomplete scan"
            );
            let snapshot = stats.snapshot();
            assert_eq!(snapshot.files_reindexed, 0);
            let state = lock_or_recover(&reconciliation);
            assert!(state.required, "an incomplete pass stays required");
            assert!(
                state.authority.established().is_none(),
                "an incomplete pass must not seed the authoritative baseline"
            );
        });
    }

    /// A delete suppressed while a root was unavailable is still synthesized
    /// once completeness returns, because the authoritative baseline retained
    /// the path across the incomplete pass.
    #[test]
    fn delete_suppressed_during_incompleteness_is_recovered_when_completeness_returns() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3f-recover");
            let blocked = temp.path().join("g3f-recover-blocked");
            fs::create_dir_all(&root).expect("create root");
            fs::create_dir_all(&blocked).expect("create second root");
            let kept = root.join("kept.rs");
            let doomed = root.join("doomed.rs");
            fs::write(&kept, "fn kept() {}\n").expect("write kept fixture");
            fs::write(&doomed, "fn doomed() {}\n").expect("write doomed fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let discovery = DiscoveryConfig::default();
            let roots = vec![root.clone(), blocked.clone()];

            // Complete pass: establishes the authoritative baseline.
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("first pass resolves every root");
            assert!(
                lock_or_recover(&reconciliation)
                    .authority
                    .established()
                    .expect("the first complete pass establishes authority")
                    .snapshot
                    .contains_key(&doomed)
            );

            // The file goes away while the second root is unavailable, so the
            // delete cannot be derived yet.
            let renamed = temp.path().join("g3f-recover-doomed-moved.rs");
            fs::rename(&doomed, &renamed).expect("move doomed fixture out of the tree");
            let stolen = temp.path().join("g3f-recover-blocked-moved");
            fs::rename(&blocked, &stolen).expect("move the second root out of the way");

            let error = run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect_err("the missing root makes this pass incomplete");
            assert!(is_retryable_error(&error));

            // Completeness returns; the retained baseline still lists the
            // removed path, so this pass finally derives its delete.
            fs::rename(&stolen, &blocked).expect("restore the second root");
            lock_or_recover(&pipeline.batches).clear();
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("the restored pass resolves every root");

            let deletes = pipeline
                .all_ops()
                .into_iter()
                .filter_map(|op| match op {
                    WatchIngestOp::Delete { path, .. } => Some(path),
                    WatchIngestOp::Upsert { .. } => None,
                })
                .collect::<Vec<_>>();
            assert!(
                deletes.contains(&doomed),
                "the suppressed delete must be recovered, got {deletes:?}"
            );
            assert!(
                !deletes.contains(&kept),
                "a surviving file must never be deleted"
            );
        });
    }

    /// A target that disappears between the scan and the apply is a genuine
    /// absence, not an unresolved path: the pass still completes.
    #[test]
    fn target_that_disappears_between_checks_still_completes() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3f-vanishing");
            fs::create_dir_all(&root).expect("create root");
            let stable = root.join("stable.rs");
            let vanishing = root.join("vanishing.rs");
            fs::write(&stable, "fn stable() {}\n").expect("write stable fixture");
            fs::write(&vanishing, "fn vanishing() {}\n").expect("write vanishing fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let discovery = DiscoveryConfig::default();
            let roots = vec![root.clone()];

            // The file is removed from the tree *inside* the listing-to-stat
            // window, which is the window the real race occupies. Moving it
            // after the whole scan proved nothing: the scan had already
            // recorded it, so no `NotFound` was ever exercised.
            let moved_aside = temp.path().join("g3f-vanishing-moved.rs");
            let _probe = {
                let vanishing_for_probe = vanishing.clone();
                let moved_for_probe = moved_aside.clone();
                install_scan_observer(Arc::new(move |probe: ScanProbe, path: &Path| {
                    if probe == ScanProbe::EntryListed
                        && path == vanishing_for_probe
                        && vanishing_for_probe.exists()
                    {
                        fs::rename(&vanishing_for_probe, &moved_for_probe)
                            .expect("move the vanishing fixture aside between listing and stat");
                    }
                }))
            };
            let collector =
                move |roots: &[PathBuf], discovery: &DiscoveryConfig, abort: &dyn Fn() -> bool| {
                    collect_snapshot_from_roots(roots, discovery, abort)
                };

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collector,
                &|| false,
            )
            .await
            .expect("a file vanishing mid-pass is a real absence, not an unresolved path");
            drop(_probe);
            assert!(
                !vanishing.exists(),
                "the probe must actually have removed the file inside the listing window"
            );

            let state = lock_or_recover(&reconciliation);
            assert!(!state.required, "the pass settled");
            assert!(state.authority.established().is_some());
        });
    }

    /// A root replaced by a fresh directory is a different tree, even though
    /// it reads as a perfectly complete scan of an empty one.
    #[test]
    fn swapped_root_identity_is_incomplete_rather_than_a_mass_delete() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g3f-identity");
            fs::create_dir_all(&root).expect("create root");
            let indexed = root.join("indexed.rs");
            fs::write(&indexed, "fn indexed() {}\n").expect("write indexed fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let discovery = DiscoveryConfig::default();
            let roots = vec![root.clone()];

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("baseline pass over the original root");

            // Rename the populated root to a vacant name and create a new,
            // empty directory at the configured path. Nothing is deleted or
            // overwritten: the original tree is intact under a new name.
            let vacated = temp.path().join("g3f-identity-original");
            fs::rename(&root, &vacated).expect("rename the original root aside");
            fs::create_dir(&root).expect("create the replacement root");
            assert!(
                vacated.join("indexed.rs").exists(),
                "the original tree must survive the rename"
            );

            lock_or_recover(&pipeline.batches).clear();
            let error = run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect_err("a replaced root is not a complete scan of an empty tree");

            assert!(is_retryable_error(&error));
            assert!(
                pipeline.all_ops().is_empty(),
                "a swapped root must delete nothing, got {:?}",
                pipeline.all_ops()
            );
            let state = lock_or_recover(&reconciliation);
            assert!(state.required);
            assert!(
                state
                    .authority
                    .established()
                    .expect("baseline from the first pass")
                    .snapshot
                    .contains_key(&indexed),
                "the authoritative baseline must survive the swap"
            );
        });
    }

    /// A scan that cannot identify its roots upserts and then *settles*.
    ///
    /// Re-arming here is the hot loop: the ingest loop reconciles whenever the
    /// requirement is set, a degraded pass returns `Ok`, and the loop
    /// immediately walks the whole tree again — forever, because asking a
    /// platform that has no root identity to look again cannot make one
    /// appear. The pass must therefore apply what it saw, refuse to touch the
    /// authority, and leave nothing owed.
    #[test]
    fn degraded_identity_pass_upserts_and_does_not_rearm_the_rescan() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g4-degraded");
            fs::create_dir_all(&root).expect("create root");
            let present = root.join("present.rs");
            fs::write(&present, "fn present() {}\n").expect("write fixture");
            let roots = vec![root.clone()];

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            lock_or_recover(&reconciliation).require_full_scan();
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let scans = Arc::new(AtomicUsize::new(0));

            // The platform seam, reproduced exactly: a truthful listing whose
            // roots carry no identity, which is what `RootIdentity::of`
            // returns off Unix.
            let scans_for_collector = Arc::clone(&scans);
            let degraded_collector =
                move |roots: &[PathBuf],
                      discovery: &DiscoveryConfig,
                      abort: &dyn Fn() -> bool|
                      -> SearchResult<(FileSnapshot, ScanCompleteness)> {
                    scans_for_collector.fetch_add(1, Ordering::AcqRel);
                    let (snapshot, _) = collect_snapshot_from_roots(roots, discovery, abort)?;
                    let mut completeness = ScanCompleteness::default();
                    for root in roots {
                        completeness.record_root_identity(root, None);
                    }
                    Ok((snapshot, completeness))
                };

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &DiscoveryConfig::default(),
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &degraded_collector,
                &|| false,
            )
            .await
            .expect("a degraded pass still succeeds; it simply cannot delete");

            assert_eq!(scans.load(Ordering::Acquire), 1);
            assert!(
                pipeline
                    .all_ops()
                    .iter()
                    .any(|op| matches!(op, WatchIngestOp::Upsert { .. })),
                "a degraded pass is still a truthful listing and must upsert it"
            );
            assert!(
                !pipeline
                    .all_ops()
                    .iter()
                    .any(|op| matches!(op, WatchIngestOp::Delete { .. })),
                "a scan that cannot identify its roots must never delete"
            );
            let state = lock_or_recover(&reconciliation);
            assert!(
                !state.required,
                "a degraded pass must not re-arm itself; re-arming is the rescan hot loop"
            );
            assert!(
                state.authority.established().is_none(),
                "a degraded scan must never become deletion authority"
            );
            assert!(state.indexed_snapshot.contains_key(&present));
        });
    }

    /// A configuration naming the same root twice still yields authority.
    ///
    /// Coverage is keyed by path, so a duplicate can never have its own entry
    /// in the identity map. Comparing the map against the length of the raw
    /// slice therefore failed forever, and a watcher that can never hold
    /// authority can never derive a deletion — the fail-closed rule turned
    /// into permanent paralysis by a configuration typo.
    #[test]
    fn duplicate_roots_do_not_forfeit_deletion_authority() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g4-duplicate");
            fs::create_dir_all(&root).expect("create root");
            let kept = root.join("kept.rs");
            let doomed = root.join("doomed.rs");
            fs::write(&kept, "fn kept() {}\n").expect("write kept fixture");
            fs::write(&doomed, "fn doomed() {}\n").expect("write doomed fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let queue: ReadyBatchQueue = Arc::new(Mutex::new(VecDeque::new()));
            let stats = WatcherStatsInner::default();
            let discovery = DiscoveryConfig::default();
            // The same root, twice.
            let roots = vec![root.clone(), root.clone()];

            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("first pass over a duplicated root");
            assert!(
                lock_or_recover(&reconciliation)
                    .established_authority(&roots)
                    .is_some(),
                "a duplicated root must still be coverable, or no deletion is ever derivable"
            );

            fs::remove_file(&doomed).expect("remove the doomed fixture");
            lock_or_recover(&pipeline.batches).clear();
            run_authoritative_reconciliation(
                &cx,
                &roots,
                &discovery,
                pipeline.as_ref(),
                &reconciliation,
                &queue,
                &stats,
                100,
                &collect_snapshot_from_roots,
                &|| false,
            )
            .await
            .expect("second pass over a duplicated root");

            let deletes = pipeline
                .all_ops()
                .into_iter()
                .filter_map(|op| match op {
                    WatchIngestOp::Delete { path, .. } => Some(path),
                    WatchIngestOp::Upsert { .. } => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                deletes,
                vec![doomed],
                "the removed file must be derived exactly once, not once per duplicate"
            );
        });
    }

    /// A successful batch may advance authority, never rebind it.
    ///
    /// The bookkeeping scan behind a batch runs with no epoch or generation
    /// guard of its own. If the root is replaced between the batch landing and
    /// that scan, adopting its snapshot silently makes the *new* root the
    /// authority — after which every file of the old one reads as absent, and
    /// the next pass deletes the entire index. It must fail closed instead.
    #[test]
    fn a_recorded_batch_cannot_rebind_authority_to_a_swapped_root() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().join("g4-rebind");
        fs::create_dir_all(&root).expect("create root");
        let indexed = root.join("indexed.rs");
        fs::write(&indexed, "fn indexed() {}\n").expect("write fixture");
        let roots = vec![root.clone()];
        let discovery = DiscoveryConfig::default();

        let (snapshot, completeness) = collect_snapshot_from_roots(&roots, &discovery, &|| false)
            .expect("baseline scan of the original root");
        assert!(completeness.is_complete());
        let reconciliation: ReconciliationTracker =
            Arc::new(Mutex::new(ReconciliationState::default()));
        {
            let mut state = lock_or_recover(&reconciliation);
            assert!(
                state.seed_initial_authority(snapshot, completeness.root_identities().clone()),
                "the fixture must start from real, covering authority"
            );
        }
        let established = lock_or_recover(&reconciliation)
            .authority
            .established()
            .cloned()
            .expect("authority fixture");

        // The root is replaced by a fresh directory: a complete, trustworthy
        // scan of a different tree.
        let vacated = temp.path().join("g4-rebind-original");
        fs::rename(&root, &vacated).expect("rename the original root aside");
        fs::create_dir(&root).expect("create the replacement root");
        let replacement = root.join("new-tree.rs");
        fs::write(&replacement, "fn replacement() {}\n").expect("write replacement fixture");

        record_successful_events(
            &roots,
            &discovery,
            &reconciliation,
            &[WatchEvent::modified(&replacement, 100, Some(12))],
            &collect_snapshot_from_roots,
        )
        .expect("recording succeeds; it simply refuses to rebind");

        let state = lock_or_recover(&reconciliation);
        let held = state
            .authority
            .established()
            .expect("the authority must survive");
        assert_eq!(
            held.root_identities, established.root_identities,
            "authority must still be bound to the roots it was established through"
        );
        assert_eq!(
            held.snapshot, established.snapshot,
            "a scan of a different tree must not become the authority's snapshot"
        );
        assert_eq!(
            held.generation, established.generation,
            "a refused advance must not consume a generation"
        );
        assert!(
            state.required,
            "the rebind attempt must hand the question to a pass that can adjudicate it"
        );
    }

    /// A stop must apply what the producer already flushed.
    ///
    /// The producer's shutdown path drains its debounce buffer into the ready
    /// queue precisely because those events are real observations that nothing
    /// re-derives. Returning the instant a stop was seen discarded every one of
    /// them, so a file modified just before shutdown stayed stale in the index.
    #[test]
    fn stop_applies_the_batches_the_producer_already_flushed() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g4-final-drain");
            fs::create_dir_all(&root).expect("create root");
            let flushed = root.join("flushed-at-shutdown.rs");
            fs::write(&flushed, "fn flushed() {}\n").expect("write fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let queue: ReadyBatchQueue =
                Arc::new(Mutex::new(VecDeque::from([vec![WatchEvent::modified(
                    &flushed,
                    900,
                    Some(14),
                )]])));
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            let stats = Arc::new(WatcherStatsInner::default());
            let stop = Arc::new(WatcherStop::default());
            // The producer has flushed and exited; the stop is already
            // published, exactly as `stop_checked` leaves it.
            let producer_done = Arc::new(AtomicBool::new(true));
            stop.request();

            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stop_for_task = Arc::clone(&stop);
            let stats_for_task = Arc::clone(&stats);
            let reconciliation_for_task = Arc::clone(&reconciliation);
            let producer_done_for_task = Arc::clone(&producer_done);
            let roots = vec![root.clone()];
            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    run_ingest_loop(
                        &child_cx,
                        &roots,
                        &DiscoveryConfig::default(),
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                        &reconciliation_for_task,
                        100,
                        &producer_done_for_task,
                        &collect_snapshot_from_roots,
                    )
                    .await
                })
                .expect("spawn final-drain task");

            task.join(&cx)
                .await
                .expect("final-drain task terminal result")
                .expect("a stop is an ordinary shutdown");

            let upserted = pipeline
                .all_ops()
                .into_iter()
                .filter_map(|op| match op {
                    WatchIngestOp::Upsert { file_key, .. } => Some(file_key),
                    WatchIngestOp::Delete { .. } => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                upserted,
                vec![normalize_file_key(&flushed)],
                "the flushed batch must be applied, not discarded by the stop"
            );
            assert!(
                lock_or_recover(&queue).is_empty(),
                "the drained queue must be empty"
            );
            assert_eq!(stats.snapshot().files_reindexed, 1);
        });
    }

    /// A stop that finds a rescan already owed must not apply queued batches
    /// ahead of it; it records them for the pass that can adjudicate them.
    #[test]
    fn stop_folds_queued_batches_into_an_owed_rescan_instead_of_applying_them() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let root = temp.path().join("g4-final-fold");
            fs::create_dir_all(&root).expect("create root");
            let queued = root.join("queued.rs");
            fs::write(&queued, "fn queued() {}\n").expect("write fixture");

            let pipeline = Arc::new(RecordingPipeline::default());
            let queue: ReadyBatchQueue =
                Arc::new(Mutex::new(VecDeque::from([vec![WatchEvent::modified(
                    &queued,
                    950,
                    Some(13),
                )]])));
            let reconciliation: ReconciliationTracker =
                Arc::new(Mutex::new(ReconciliationState::default()));
            lock_or_recover(&reconciliation).require_full_scan();
            let stats = Arc::new(WatcherStatsInner::default());
            let stop = Arc::new(WatcherStop::default());
            let producer_done = Arc::new(AtomicBool::new(true));
            stop.request();

            let pipeline_for_task = Arc::clone(&pipeline);
            let queue_for_task = Arc::clone(&queue);
            let stop_for_task = Arc::clone(&stop);
            let stats_for_task = Arc::clone(&stats);
            let reconciliation_for_task = Arc::clone(&reconciliation);
            let producer_done_for_task = Arc::clone(&producer_done);
            let roots = vec![root.clone()];
            let mut task = cx
                .spawn_local(move |child_cx| async move {
                    run_ingest_loop(
                        &child_cx,
                        &roots,
                        &DiscoveryConfig::default(),
                        pipeline_for_task.as_ref(),
                        &queue_for_task,
                        &stop_for_task,
                        &stats_for_task,
                        &reconciliation_for_task,
                        100,
                        &producer_done_for_task,
                        &collect_snapshot_from_roots,
                    )
                    .await
                })
                .expect("spawn fold task");

            task.join(&cx)
                .await
                .expect("fold task terminal result")
                .expect("a stop is an ordinary shutdown");

            assert!(
                pipeline.all_ops().is_empty(),
                "nothing may be applied ahead of an owed authoritative pass, got {:?}",
                pipeline.all_ops()
            );
            let state = lock_or_recover(&reconciliation);
            assert!(state.required, "the rescan is still owed");
            assert!(
                state.affected_paths.contains(&queued),
                "the queued work must survive as a candidate for that pass"
            );
            assert!(lock_or_recover(&queue).is_empty());
        });
    }

    /// A wedged lifecycle transition must time out rather than poll forever.
    ///
    /// Both entry points wait on a generation another caller owns. The public
    /// methods pass a production-sized budget; this drives the same loops with
    /// a small one so the timeout branch is reachable in milliseconds.
    #[test]
    fn a_wedged_generation_times_out_instead_of_polling_forever() {
        run_on_runtime_task(|cx| async move {
            let temp = tempdir().expect("tempdir");
            let watcher = FsWatcher::new(
                vec![temp.path().to_path_buf()],
                DiscoveryConfig::default(),
                Arc::new(NoopWatchIngestPipeline),
            );

            // A generation stuck mid-transition, owned by nobody who will ever
            // complete it.
            {
                let mut control = lock_or_recover(&watcher.control);
                control.next_generation = 7;
                control.lifecycle = WatcherLifecycle::Stopping {
                    generation: 7,
                    stop: Arc::new(WatcherStop::default()),
                };
            }

            let stop_error = watcher
                .stop_checked_bounded(&cx, 4)
                .await
                .expect_err("a wedged generation must surface as an error");
            assert!(
                stop_error.to_string().contains("timed out"),
                "the stop timeout must say so, got {stop_error}"
            );

            let start_error = watcher
                .start_bounded(&cx, 4)
                .await
                .expect_err("a wedged generation must surface as an error");
            assert!(
                start_error.to_string().contains("timed out"),
                "the start timeout must say so, got {start_error}"
            );

            // Control: once the transition completes, both settle normally.
            lock_or_recover(&watcher.control).lifecycle = WatcherLifecycle::Stopped;
            watcher
                .stop_checked_bounded(&cx, 4)
                .await
                .expect("a settled lifecycle stops immediately");
        });
    }

    /// One test's scan observer must not be removed by another's.
    ///
    /// The single global slot made installation destructive and removal
    /// global: two tests in the same binary clobbered each other, and a
    /// panicking test left its observer installed for every test that
    /// followed. Registration is scoped to its guard, so this proves both that
    /// a second registration does not displace the first and that dropping one
    /// leaves the other live.
    #[test]
    fn scan_observers_are_scoped_to_their_own_registration() {
        let temp = tempdir().expect("tempdir");
        let root = temp.path().join("g4-observer-scope");
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("seen.rs"), "fn seen() {}\n").expect("write fixture");
        let roots = vec![root.clone()];
        let discovery = DiscoveryConfig::default();

        let mine = Arc::new(AtomicUsize::new(0));
        let theirs = Arc::new(AtomicUsize::new(0));
        let mine_guard = {
            let mine = Arc::clone(&mine);
            let owned = root.clone();
            install_scan_observer(Arc::new(move |_probe: ScanProbe, path: &Path| {
                if path.starts_with(&owned) {
                    mine.fetch_add(1, Ordering::AcqRel);
                }
            }))
        };
        let their_guard = {
            let theirs = Arc::clone(&theirs);
            let owned = root.clone();
            install_scan_observer(Arc::new(move |_probe: ScanProbe, path: &Path| {
                if path.starts_with(&owned) {
                    theirs.fetch_add(1, Ordering::AcqRel);
                }
            }))
        };

        collect_snapshot_from_roots(&roots, &discovery, &|| false).expect("scan with both live");
        let both_mine = mine.load(Ordering::Acquire);
        assert!(
            both_mine > 0 && theirs.load(Ordering::Acquire) > 0,
            "a second registration must not displace the first"
        );

        // The peer finishes — as a panicking test's guard would also do.
        drop(their_guard);
        let their_final = theirs.load(Ordering::Acquire);
        collect_snapshot_from_roots(&roots, &discovery, &|| false).expect("scan after one is gone");
        assert!(
            mine.load(Ordering::Acquire) > both_mine,
            "removing a peer's registration must leave this one live"
        );
        assert_eq!(
            theirs.load(Ordering::Acquire),
            their_final,
            "a dropped registration must stop being called"
        );

        drop(mine_guard);
        let mine_final = mine.load(Ordering::Acquire);
        collect_snapshot_from_roots(&roots, &discovery, &|| false).expect("scan with none live");
        assert_eq!(
            mine.load(Ordering::Acquire),
            mine_final,
            "the last registration must be removed by its own guard"
        );
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
                        move |roots: &[PathBuf],
                              discovery: &DiscoveryConfig,
                              abort: &dyn Fn() -> bool| {
                            if snapshot_attempts_for_task.fetch_add(1, Ordering::AcqRel) == 0 {
                                return Err(SearchError::Io(io::Error::other(
                                    "injected retryable post-commit snapshot failure",
                                )));
                            }
                            collect_snapshot_from_roots(roots, discovery, abort)
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
            let fixture_roots = vec![temp.path().to_path_buf()];
            let reconciliation: ReconciliationTracker = Arc::new(Mutex::new(ReconciliationState {
                indexed_snapshot: prior_snapshot.clone(),
                baseline_initialized: true,
                required: true,
                affected_paths: BTreeSet::from([prior_path.clone()]),
                epoch: 1,
                // Established authority, with the real identities a live scan
                // of these roots would have recorded. This is deliberately the
                // opposite state from the no-authority fixture above: this one
                // may derive deletes, that one must refuse.
                authority: super::DeletionAuthorityState::Established {
                    authority: authority_over(prior_snapshot.clone(), &fixture_roots),
                    legacy: None,
                },
                rebuild_authorized: false,
                unsettled_passes: 0,
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
                &|| false,
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
                collect_snapshot_from_roots(&roots, &discovery, &|| false)
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
                &|| false,
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
            collect_snapshot_from_roots(&roots, &discovery, &|| false).expect("baseline scan");
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
            collect_snapshot_from_roots(&roots, &discovery, &|| false)
                .expect("hostile scan still succeeds");

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

        let (snapshot, completeness) = collect_snapshot_from_roots(&roots, &discovery, &|| false)
            .expect("scan with an absent root");

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
        run_on_runtime_task_with_result(test);
    }

    /// `run_on_runtime_task`, keeping the task's value.
    ///
    /// A lifecycle assertion has to be made *after* the runtime has finished
    /// with the watcher, while the value it asserts on is produced inside the
    /// task — a second runtime would be joining handles the first one owns.
    fn run_on_runtime_task_with_result<F, Fut, T>(test: F) -> T
    where
        F: FnOnce(Cx) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let scheduler = RuntimeBuilder::current_thread()
            .build()
            .expect("build watcher test runtime");
        let test_task = scheduler.handle().spawn(async move {
            let cx = Cx::current().expect("runtime task installs a spawn-capable Cx");
            test(cx).await
        });
        scheduler.block_on(test_task)
    }
}
