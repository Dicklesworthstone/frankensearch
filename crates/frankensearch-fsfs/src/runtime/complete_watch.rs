//! Full-generation watch rebuilding without a serving-index writer lifetime.
//!
//! Native notifications are hints, not an authoritative deletion log. A complete
//! discovery observation drives each rebuild, and a second observation is checked
//! immediately before pointer publication. This is the existing cooperative
//! store protocol, not a filesystem snapshot or hostile-directory v2 authority.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use super::complete_cli::{complete_cli_error, require_durable_publication};
use super::{FsfsRuntime, retained_search_checkpoint, validate_retained_catalog_path};
use crate::OutputFormat;
use crate::config::{DiscoveryCandidate, DiscoveryConfig, DiscoveryScopeDecision};
use crate::generation_store::{
    CompleteGenerationStore, GenerationPublication, PublishedGeneration,
};
use crate::mount_info::{MountTable, read_system_mounts};
use crate::watcher::DEFAULT_DEBOUNCE_MS;

const DEBOUNCE: Duration = Duration::from_millis(DEFAULT_DEBOUNCE_MS);
const MAX_DEBOUNCE: Duration = Duration::from_secs(5);
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);
const IDLE_POLL: Duration = Duration::from_millis(50);
// Pause candidate allocation under sustained churn, not the watch registration.
const MAX_UNSTABLE_BUILDS: usize = 3;
const SETTLE_INTERVAL: Duration = Duration::from_secs(5);
const MAX_HINT_PATHS: usize = 128;
const MAX_HINT_PATH_BYTES: usize = 4096;

#[derive(Debug)]
struct DirtyWindow {
    first: Instant,
    last: Instant,
    force_rebuild: bool,
    paths: BTreeSet<PathBuf>,
}

/// Bounded coalescing state. Overflow becomes one forced reconciliation rather
/// than an unbounded path queue or silently dropped content-change evidence.
#[derive(Default)]
struct Changes {
    dirty: Option<DirtyWindow>,
    // Sticky: a second request cannot clear a failed backend by retrying.
    failure: Option<Arc<notify::Error>>,
}

impl Changes {
    fn check_backend(&self) -> SearchResult<()> {
        if let Some(failure) = &self.failure {
            return Err(SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.watch",
                source: Box::new(BackendFailure(Arc::clone(failure))),
            });
        }
        Ok(())
    }

    fn record(&mut self, now: Instant, force_rebuild: bool) {
        if let Some(window) = &mut self.dirty {
            window.last = now;
            window.force_rebuild |= force_rebuild;
        } else {
            self.dirty = Some(DirtyWindow {
                first: now,
                last: now,
                force_rebuild,
                paths: BTreeSet::new(),
            });
        }
    }

    fn record_path(&mut self, path: &Path) {
        let Some(window) = self.dirty.as_mut() else {
            return;
        };
        if window.force_rebuild {
            return;
        }
        if !path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, Component::ParentDir))
            || path.as_os_str().len() > MAX_HINT_PATH_BYTES
            || (!window.paths.contains(path) && window.paths.len() == MAX_HINT_PATHS)
        {
            window.force_rebuild = true;
            window.paths.clear();
            return;
        }
        window.paths.insert(path.to_path_buf());
    }

    fn take_due(&mut self, now: Instant) -> Option<DirtyWindow> {
        let window = self.dirty.as_ref()?;
        if now.saturating_duration_since(window.last) < DEBOUNCE
            && now.saturating_duration_since(window.first) < MAX_DEBOUNCE
        {
            return None;
        }
        // Consume BEFORE observing/building, never after publication. A callback
        // arriving during either operation belongs to the next generation.
        self.dirty.take()
    }
}

#[derive(Debug)]
struct BackendFailure(Arc<notify::Error>);

impl std::fmt::Display for BackendFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "complete-generation notification backend failed: {}",
            self.0
        )
    }
}

impl std::error::Error for BackendFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[derive(Debug)]
struct SourceChanged;

impl std::fmt::Display for SourceChanged {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("source changed during complete-generation discovery or rebuilding")
    }
}

impl std::error::Error for SourceChanged {}

fn source_changed() -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.watch",
        source: Box::new(SourceChanged),
    }
}

fn is_source_changed(error: &SearchError) -> bool {
    matches!(error, SearchError::SubsystemError { source, .. } if source.is::<SourceChanged>())
}

fn observation_io(error: std::io::Error) -> SearchError {
    if error.kind() == std::io::ErrorKind::NotFound {
        source_changed()
    } else {
        error.into()
    }
}

fn lock_changes(changes: &Mutex<Changes>) -> SearchResult<MutexGuard<'_, Changes>> {
    changes.lock().map_err(|_| {
        complete_cli_error(
            "watch_state",
            "complete-generation notification state was poisoned",
        )
    })
}

fn check_backend(changes: &Mutex<Changes>) -> SearchResult<()> {
    lock_changes(changes)?.check_backend()
}

fn record_notification(changes: &Mutex<Changes>, result: notify::Result<Event>, now: Instant) {
    // The owning lane reports poisoning; a callback must not panic a backend
    // worker or retain a runtime/index writer after the session is dropped.
    let Ok(mut changes) = changes.lock() else {
        return;
    };
    match result {
        Ok(event) => {
            let rescan = event.need_rescan();
            if rescan || !matches!(event.kind, EventKind::Access(_)) {
                changes.record(
                    now,
                    rescan || event.paths.is_empty() || event.paths.len() > MAX_HINT_PATHS,
                );
                for path in event.paths.iter().take(MAX_HINT_PATHS) {
                    changes.record_path(path);
                }
            }
        }
        Err(error) => {
            changes.failure.get_or_insert_with(|| Arc::new(error));
        }
    }
}

/// Keep the original root inode alive so remove/recreate cannot be mistaken
/// for an authoritative empty corpus. All later indexing uses this canonical
/// source path, not a possibly retargeted original CLI alias.
struct SourceRoot {
    path: PathBuf,
    directory: File,
}

impl SourceRoot {
    fn open(path: PathBuf) -> SearchResult<Self> {
        let directory = File::open(&path)?;
        let source = Self { path, directory };
        source.check()?;
        Ok(source)
    }

    fn check(&self) -> SearchResult<()> {
        let original = self.directory.metadata()?;
        let current = fs::symlink_metadata(&self.path)?;
        let original_identity = (original.dev(), original.ino());
        let current_identity = (current.dev(), current.ino());
        if !original.is_dir() || !current.is_dir() || original_identity != current_identity {
            return Err(complete_cli_error(
                "watch_source",
                "the watched source directory was replaced or is not a directory; the selected generation was not discarded",
            ));
        }
        Ok(())
    }

    fn observe(&self, cx: &Cx, discovery: &DiscoveryConfig) -> SearchResult<SourceObservation> {
        self.observe_with_entry_check(cx, discovery, |_| Ok(()))
    }

    // Unlike the mutable watcher's millisecond reconciliation snapshot, this
    // observation needs inode and nanosecond stamps from the same traversal.
    // Reuse its public discovery/mount policy, but fail the entire observation
    // on incomplete traversal: a replacement bundle cannot apply partial data.
    fn observe_with_entry_check<F>(
        &self,
        cx: &Cx,
        discovery: &DiscoveryConfig,
        mut before_entry: F,
    ) -> SearchResult<SourceObservation>
    where
        F: FnMut(&Path) -> SearchResult<()>,
    {
        retained_search_checkpoint(cx)?;
        self.check()?;
        let mounts = MountTable::new(read_system_mounts(), &discovery.mount_override_map());
        let category = mounts.lookup(&self.path).map(|(entry, _)| entry.category);
        if matches!(
            discovery.evaluate_root(&self.path, category).scope,
            DiscoveryScopeDecision::Exclude
        ) {
            return Err(complete_cli_error(
                "watch_discovery",
                "the watched source is excluded by discovery policy; refusing an empty replacement",
            ));
        }
        let mut stamps = BTreeMap::new();
        let mut directories = BTreeMap::new();
        let mut visited = BTreeSet::new();
        let mut stack = vec![(self.path.clone(), self.directory.metadata()?)];
        while let Some((directory, expected)) = stack.pop() {
            retained_search_checkpoint(cx)?;
            // Keep the descriptor alive until the listing and its final check
            // finish; replacement cannot recycle the opened inode underneath us.
            let held = File::open(&directory).map_err(observation_io)?;
            let opened = held.metadata()?;
            if !opened.is_dir() || (opened.dev(), opened.ino()) != (expected.dev(), expected.ino())
            {
                return Err(source_changed());
            }
            let canonical = fs::canonicalize(&directory).map_err(observation_io)?;
            if !visited.insert(canonical) {
                continue;
            }
            directories.insert(directory.clone(), (opened.dev(), opened.ino()));
            let mut entries = fs::read_dir(&directory).map_err(observation_io)?;
            loop {
                retained_search_checkpoint(cx)?;
                let Some(entry) = entries.next() else { break };
                let entry = entry.map_err(observation_io)?;
                let path = entry.path();
                before_entry(&path)?;
                retained_search_checkpoint(cx)?;
                let link = fs::symlink_metadata(&path).map_err(observation_io)?;
                let is_symlink = link.is_symlink();
                if is_symlink && !discovery.follow_symlinks {
                    continue;
                }
                let metadata = if is_symlink {
                    match fs::metadata(&path) {
                        Ok(metadata) => metadata,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(error.into()),
                    }
                } else {
                    link
                };
                let mut candidate = DiscoveryCandidate::new(
                    &path,
                    if metadata.is_dir() { 0 } else { metadata.len() },
                )
                .with_symlink(is_symlink);
                if let Some((mount, _)) = mounts.lookup(&path) {
                    candidate = candidate.with_mount_category(mount.category);
                }
                let decision = discovery.evaluate_candidate(&candidate);
                if matches!(decision.scope, DiscoveryScopeDecision::Exclude) {
                    continue;
                }
                if metadata.is_dir() {
                    stack.push((path, metadata));
                } else if metadata.is_file() && decision.ingestion_class.is_indexed() {
                    stamps.insert(path, SourceStamp::from_metadata(&metadata));
                }
            }
            let after = fs::metadata(&directory).map_err(observation_io)?;
            if SourceStamp::from_metadata(&opened) != SourceStamp::from_metadata(&held.metadata()?)
                || SourceStamp::from_metadata(&opened) != SourceStamp::from_metadata(&after)
            {
                return Err(source_changed());
            }
        }
        self.check()?;
        retained_search_checkpoint(cx)?;
        Ok(SourceObservation {
            stamps,
            directories,
        })
    }
}

/// Extend the legacy scan's millisecond timestamps with inode, length, ctime
/// and nanosecond mtime. Atomic saves and restored-mtime edits are not treated
/// as unchanged just because their coarse timestamps happen to match.
#[derive(Debug, PartialEq, Eq)]
struct SourceStamp {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl SourceStamp {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SourceObservation {
    stamps: BTreeMap<PathBuf, SourceStamp>,
    directories: BTreeMap<PathBuf, (u64, u64)>,
}

fn touches_observed_files(
    hint: &DirtyWindow,
    current: &SourceObservation,
    previous: Option<&SourceObservation>,
) -> bool {
    hint.force_rebuild
        || hint.paths.iter().any(|path| {
            current.stamps.contains_key(path)
                || current.directories.contains_key(path)
                || previous.is_some_and(|observation| {
                    observation.stamps.contains_key(path)
                        || observation.directories.contains_key(path)
                })
        })
}

/// The source baseline belongs to one publication, not just a store path.
/// Reading only the bounded descriptor here avoids rehashing every sealed
/// artifact on an idle poll. The store's ordinary admission still verifies it.
fn check_watch_publication(
    cx: &Cx,
    root: &Path,
    expected: Option<&PublishedGeneration>,
) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    if let Some(expected) = expected {
        let store = CompleteGenerationStore::open(cx, root)?;
        if !store.is_selected(cx, expected)? {
            return Err(complete_cli_error(
                "watch_selection_changed",
                "selection changed outside this watch session; refusing to reuse its source baseline or overwrite another publisher's generation",
            ));
        }
    }
    Ok(())
}

/// A read-only preflight: do not create a staging tree inside the source even
/// transiently, since that could recursively trigger its own notifications.
fn resolve_watch_store(source: &Path, root: &Path) -> SearchResult<PathBuf> {
    let root = match fs::canonicalize(root) {
        Ok(root) => root,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let name = root.file_name().ok_or_else(|| {
                complete_cli_error("store_root", "a store root needs a directory name")
            })?;
            let parent = root
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            fs::canonicalize(parent)?.join(name)
        }
        Err(error) => return Err(error.into()),
    };
    if source.starts_with(&root) || root.starts_with(source) {
        return Err(complete_cli_error(
            "store_root",
            "source and generation-store trees must not overlap",
        ));
    }
    Ok(root)
}

/// Quiet-source observations are not published baselines. They only authorize
/// a fresh catch-up attempt through the ordinary builder and its precommit.
struct SettleWindow {
    next_probe: Instant,
    observation: Option<SourceObservation>,
}

struct CompleteWatchSession {
    runtime: FsfsRuntime,
    store_root: PathBuf,
    source: SourceRoot,
    changes: Arc<Mutex<Changes>>,
    // Registration precedes the initial scan/rebuild and lives across every
    // query/publication. The callback owns only coalescing state, not a writer.
    _watcher: RecommendedWatcher,
    baseline: Option<SourceObservation>,
    publication: Option<PublishedGeneration>,
    settling: Option<SettleWindow>,
    last_reconcile: Instant,
    unstable_builds: usize,
}

impl CompleteWatchSession {
    fn open(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> SearchResult<Self> {
        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&runtime.config.storage.db_path)?;
        let source = SourceRoot::open(fs::canonicalize(runtime.resolve_target_root()?)?)?;
        let store_root = resolve_watch_store(&source.path, root)?;
        // Preserve the store's refusal of a symlink supplied by the caller;
        // canonicalization above is only for overlap checking and anchoring.
        if fs::symlink_metadata(root).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err(complete_cli_error(
                "store_root",
                "symlinked store roots are unsupported",
            ));
        }
        let changes = Arc::new(Mutex::new(Changes::default()));
        let callback_changes = Arc::clone(&changes);
        let mut watcher = notify::recommended_watcher(move |result| {
            record_notification(&callback_changes, result, Instant::now());
        })
        .map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.watch",
            source: Box::new(source),
        })?;
        watcher
            .watch(&source.path, RecursiveMode::Recursive)
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.watch",
                source: Box::new(source),
            })?;
        source.check()?;
        retained_search_checkpoint(cx)?;
        let mut runtime = runtime.clone();
        runtime.cli_input.target_path = Some(source.path.clone());
        Ok(Self {
            runtime,
            store_root,
            source,
            changes,
            _watcher: watcher,
            baseline: None,
            publication: None,
            settling: None,
            last_reconcile: Instant::now(),
            unstable_builds: 0,
        })
    }

    fn record_unstable_source(&mut self, now: Instant) -> SearchResult<()> {
        self.unstable_builds += 1;
        if self.unstable_builds >= MAX_UNSTABLE_BUILDS {
            self.settling = Some(SettleWindow {
                next_probe: now + SETTLE_INTERVAL,
                observation: None,
            });
            tracing::warn!(
                reason_code = "watch.source_settling",
                attempts = self.unstable_builds,
                "paused candidate allocation until two quiet source observations agree"
            );
        } else {
            // The catch-up obligation survives even without another event.
            // Hints received during the failed attempt are never erased.
            let mut changes = lock_changes(&self.changes)?;
            changes.record(now, true);
            if let Some(window) = &mut changes.dirty {
                window.first = now;
                window.paths.clear();
            }
        }
        Ok(())
    }

    fn probe_settling_source(&mut self, cx: &Cx, now: Instant) -> SearchResult<()> {
        self.probe_settling_source_with_entry_check(cx, now, |_| Ok(()))
    }

    fn probe_settling_source_with_entry_check<F>(
        &mut self,
        cx: &Cx,
        now: Instant,
        before_entry: F,
    ) -> SearchResult<()>
    where
        F: FnMut(&Path) -> SearchResult<()>,
    {
        retained_search_checkpoint(cx)?;
        let Some(gate) = self.settling.as_ref() else {
            return Ok(());
        };
        if now < gate.next_probe {
            return Ok(());
        }
        // Consume BEFORE observing. Events delivered during the scan remain
        // pending and veto recovery, including edits with identical metadata.
        let pending = {
            let mut changes = lock_changes(&self.changes)?;
            changes.check_backend()?;
            changes.dirty.take()
        };
        let observed = match self.source.observe_with_entry_check(
            cx,
            &self.runtime.config.discovery,
            before_entry,
        ) {
            Ok(observed) => observed,
            Err(error) if is_source_changed(&error) => {
                self.settling = Some(SettleWindow {
                    next_probe: now.max(Instant::now()) + SETTLE_INTERVAL,
                    observation: None,
                });
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        check_watch_publication(cx, &self.store_root, self.publication.as_ref())?;
        let previous = self
            .settling
            .as_ref()
            .and_then(|gate| gate.observation.as_ref());
        let touched = {
            let changes = lock_changes(&self.changes)?;
            // Check backend failure at the same boundary as hints, not only
            // before traversal. A failed backend cannot certify a quiet source.
            changes.check_backend()?;
            pending
                .as_ref()
                .is_some_and(|hint| touches_observed_files(hint, &observed, previous))
                || changes
                    .dirty
                    .as_ref()
                    .is_some_and(|hint| touches_observed_files(hint, &observed, previous))
        };
        let completed = now.max(Instant::now());
        if !touched && previous == Some(&observed) {
            // Even a source equal to the old baseline needs this catch-up:
            // metadata equality cannot cancel an earlier content-change hint.
            self.settling = None;
            self.unstable_builds = 0;
            lock_changes(&self.changes)?.record(completed, true);
            tracing::info!(
                reason_code = "watch.source_settled",
                "source observations agree; queued retained-generation catch-up"
            );
        } else {
            self.settling = Some(SettleWindow {
                next_probe: completed + SETTLE_INTERVAL,
                // A hint-contaminated scan must not count as the first of the
                // two quiet observations. No candidate is allocated here.
                observation: (!touched).then_some(observed),
            });
        }
        self.last_reconcile = completed;
        Ok(())
    }

    #[allow(clippy::future_not_send)]
    async fn attempt(
        &self,
        cx: &Cx,
        force_rebuild: bool,
        pending: Option<&DirtyWindow>,
    ) -> SearchResult<Option<(SourceObservation, GenerationPublication)>> {
        let observed = self.source.observe(cx, &self.runtime.config.discovery)?;
        let touched = pending
            .is_some_and(|hint| touches_observed_files(hint, &observed, self.baseline.as_ref()));
        if !force_rebuild && !touched && self.baseline.as_ref() == Some(&observed) {
            // Excluded-file notifications do not force re-embedding the corpus.
            return Ok(None);
        }
        // Capture only the read-only authority inputs, not the native watcher
        // handle, whose cross-platform Sync guarantees are not part of this API.
        let source = &self.source;
        let discovery = &self.runtime.config.discovery;
        let changes = &self.changes;
        let expected = &observed;
        let selected = self.publication.as_ref();
        let store_root = &self.store_root;
        let publication = self
            .runtime
            .rebuild_retained_generation_with_precommit(cx, &self.store_root, move |cx| {
                check_backend(changes)?;
                let current = source.observe(cx, discovery)?;
                if &current != expected {
                    return Err(source_changed());
                }
                // A different publisher can win between the last poll and
                // this build's begin(). Its receipt cannot retarget our baseline.
                check_watch_publication(cx, store_root, selected)?;
                // On a coarse-timestamp filesystem even ctime can match. A
                // queued in-scope content hint is still evidence of a raced
                // build, and must not be acknowledged by this publication.
                let changes = lock_changes(changes)?;
                changes.check_backend()?;
                if changes
                    .dirty
                    .as_ref()
                    .is_some_and(|hint| touches_observed_files(hint, &current, Some(expected)))
                {
                    return Err(source_changed());
                }
                drop(changes);
                Ok(())
            })
            .await?;
        Ok(Some((observed, publication)))
    }

    #[allow(clippy::future_not_send)]
    async fn advance(
        &mut self,
        cx: &Cx,
        now: Instant,
    ) -> SearchResult<Option<GenerationPublication>> {
        retained_search_checkpoint(cx)?;
        check_backend(&self.changes)?;
        self.source.check()?;
        check_watch_publication(cx, &self.store_root, self.publication.as_ref())?;
        if self.settling.is_some() {
            self.probe_settling_source(cx, now)?;
            return Ok(None);
        }
        let initial = self.baseline.is_none() && self.unstable_builds == 0;
        let periodic = now.saturating_duration_since(self.last_reconcile) >= RECONCILE_INTERVAL;
        let pending = {
            let mut changes = lock_changes(&self.changes)?;
            if initial || periodic {
                changes.dirty.take()
            } else {
                changes.take_due(now)
            }
        };
        if !initial && !periodic && pending.is_none() {
            return Ok(None);
        }
        self.last_reconcile = now;
        let force_rebuild = initial || pending.as_ref().is_some_and(|window| window.force_rebuild);
        match self.attempt(cx, force_rebuild, pending.as_ref()).await {
            Ok(Some((observation, publication))) => {
                self.baseline = Some(observation);
                self.publication = Some(match &publication {
                    GenerationPublication::Durable(generation)
                    | GenerationPublication::VisibleButDurabilityUncertain { generation, .. } => {
                        generation.clone()
                    }
                });
                self.unstable_builds = 0;
                Ok(Some(publication))
            }
            Ok(None) => {
                self.unstable_builds = 0;
                Ok(None)
            }
            Err(error) if is_source_changed(&error) => {
                self.record_unstable_source(Instant::now())?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

impl FsfsRuntime {
    /// Watch a directory and atomically publish complete replacement generations.
    ///
    /// Native notifications are registered before the initial rebuild. Edits
    /// coalesce for 500 ms (at most 5 s); a 30 s metadata/membership reconciliation
    /// recovers missed observable changes. Bounded path hints also preserve edits
    /// with unchanged coarse timestamps. Every build reuses full-search admission and
    /// validates source membership/metadata before selection. Source and store
    /// must be disjoint; an incomplete scan is never an authoritative deletion.
    ///
    /// Existing retained readers remain usable throughout this future. This is
    /// a replacement-build route using eligible checkpoint-proven embedding
    /// reuse, not delta-only discovery or legacy-watch migration.
    /// The sink runs only after durable publication; its error cannot roll that
    /// publication back. Visibility with uncertain durability returns an explicit
    /// error and stops instead of emitting a false durable receipt.
    ///
    /// # Errors
    /// Returns admission, discovery, backend, output, cancellation or publication
    /// errors. Three consecutive source races pause candidate allocation. Two
    /// quiet observations five seconds apart queue catch-up without requiring a
    /// new notification. Paused probes neither allocate nor delete generations.
    /// External publication changes stop this session rather than silently
    /// associating its source baseline with someone else's selected bundle.
    /// Dropping the future drops its owned native watch registration; no
    /// independent indexing task or runtime is spawned.
    #[allow(clippy::future_not_send)]
    pub async fn watch_retained_generations<F>(
        &self,
        cx: &Cx,
        root: &Path,
        mut on_publication: F,
    ) -> SearchResult<()>
    where
        F: FnMut(&PublishedGeneration) -> SearchResult<()> + Send,
    {
        let mut session = CompleteWatchSession::open(self, cx, root)?;
        loop {
            if let Some(publication) = session.advance(cx, Instant::now()).await? {
                let generation = require_durable_publication(publication)?;
                // Do not insert a cancellation point between a completed rename
                // and its receipt. Cancellation cannot relabel a visible commit.
                on_publication(&generation)?;
            }
            retained_search_checkpoint(cx)?;
            asupersync::time::sleep(cx.now(), IDLE_POLL).await;
        }
    }

    #[allow(clippy::future_not_send)]
    pub(super) async fn run_complete_generation_watch_with_writer<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        writer: &mut W,
    ) -> SearchResult<()> {
        if !matches!(
            self.cli_input.format,
            OutputFormat::Table | OutputFormat::Jsonl | OutputFormat::Toon
        ) {
            return Err(complete_cli_error(
                "watch_format",
                "complete-generation watch requires --format table, jsonl, or toon; a sequence of standalone JSON or CSV documents is not emitted",
            ));
        }
        self.watch_retained_generations(cx, root, |generation| {
            self.emit_complete_generation_receipt(root, generation, "watch", writer)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use notify::event::{AccessKind, DataChange, Flag, ModifyKind};

    #[test]
    fn complete_watch_debounce_does_not_starve_under_a_continuous_burst() {
        let start = Instant::now();
        let mut changes = Changes::default();
        changes.record(start, false);
        for step in 1..50 {
            let now = start + Duration::from_millis(step * 100);
            changes.record(now, false);
            assert!(changes.take_due(now).is_none());
        }
        changes.record(start + MAX_DEBOUNCE, false);
        assert!(changes.take_due(start + MAX_DEBOUNCE).is_some());
        assert!(changes.dirty.is_none());
    }

    #[test]
    fn complete_watch_hint_during_a_build_is_not_acknowledged_by_the_old_ticket() {
        let start = Instant::now();
        let mut changes = Changes::default();
        changes.record(start, false);
        let old_ticket = changes.take_due(start + DEBOUNCE).unwrap();
        let during_build = start + DEBOUNCE + Duration::from_millis(1);
        changes.record(during_build, true);
        assert!(!old_ticket.force_rebuild);
        assert!(changes.take_due(during_build).is_none());
        let next = changes.take_due(during_build + DEBOUNCE).unwrap();
        assert_eq!(next.first, during_build);
        assert!(next.force_rebuild);
    }

    #[test]
    fn complete_watch_access_does_not_rebuild_but_rescan_even_on_access_does() {
        let changes = Mutex::new(Changes::default());
        let now = Instant::now();
        record_notification(
            &changes,
            Ok(Event::new(EventKind::Access(AccessKind::Any))),
            now,
        );
        assert!(lock_changes(&changes).unwrap().dirty.is_none());
        record_notification(
            &changes,
            Ok(Event::new(EventKind::Access(AccessKind::Any)).set_flag(Flag::Rescan)),
            now,
        );
        assert!(
            lock_changes(&changes)
                .unwrap()
                .dirty
                .as_ref()
                .unwrap()
                .force_rebuild
        );
    }

    #[test]
    fn complete_watch_backend_error_is_sticky_and_preserves_the_original_source() {
        let changes = Mutex::new(Changes::default());
        record_notification(
            &changes,
            Err(notify::Error::generic("injected backend failure")),
            Instant::now(),
        );
        for _ in 0..2 {
            let error = check_backend(&changes).unwrap_err();
            let SearchError::SubsystemError { source, .. } = error else {
                panic!("subsystem error") // ubs:ignore — cfg(test) assertion requires the injected backend error to retain its typed source.
            };
            assert!(source.to_string().contains("injected backend failure"));
            assert!(
                source
                    .source()
                    .unwrap()
                    .to_string()
                    .contains("injected backend failure")
            );
        }
    }

    #[test]
    fn complete_watch_coalesces_large_hint_bursts_into_one_window() {
        let changes = Mutex::new(Changes::default());
        let now = Instant::now();
        for _ in 0..10_000 {
            record_notification(
                &changes,
                Ok(
                    Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Any)))
                        .add_path(PathBuf::from("/source/document.md")),
                ),
                now,
            );
        }
        let mut changes = lock_changes(&changes).unwrap();
        assert!(changes.take_due(now + DEBOUNCE).is_some());
        assert!(changes.take_due(now + DEBOUNCE).is_none());
        drop(changes);
    }

    #[test]
    fn complete_watch_hint_overflow_forces_reconciliation_without_growing_a_queue() {
        let mut changes = Changes::default();
        changes.record(Instant::now(), false);
        for index in 0..=MAX_HINT_PATHS {
            changes.record_path(&PathBuf::from(format!("/source/{index}.md")));
        }
        let window = changes.dirty.as_ref().unwrap();
        assert!(window.force_rebuild);
        assert!(window.paths.is_empty());
        changes.record_path(Path::new("/source/another.md"));
        assert!(changes.dirty.as_ref().unwrap().paths.is_empty());
    }

    #[test]
    fn complete_watch_relative_or_oversized_hint_cannot_be_silently_missed() {
        for path in [
            PathBuf::from("relative.md"),
            PathBuf::from("/source/nested/../alpha.md"),
            PathBuf::from(format!("/{}", "x".repeat(MAX_HINT_PATH_BYTES))),
        ] {
            let mut changes = Changes::default();
            changes.record(Instant::now(), false);
            changes.record_path(&path);
            assert!(changes.dirty.as_ref().unwrap().force_rebuild);
            assert!(changes.dirty.as_ref().unwrap().paths.is_empty());
        }
    }

    #[test]
    fn complete_watch_content_hint_is_not_discarded_when_all_metadata_matches() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            let path = source.path.join("alpha.md");
            fs::write(&path, "stable metadata fixture").unwrap();
            let observed = source.observe(&cx, &DiscoveryConfig::default()).unwrap();
            let mut changes = Changes::default();
            changes.record(Instant::now(), false);
            changes.record_path(&path);
            assert!(touches_observed_files(
                changes.dirty.as_ref().unwrap(),
                &observed,
                Some(&observed)
            ));
            let mut excluded = Changes::default();
            excluded.record(Instant::now(), false);
            excluded.record_path(&source.path.join("not-in-the-observation.bin"));
            assert!(!touches_observed_files(
                excluded.dirty.as_ref().unwrap(),
                &observed,
                Some(&observed)
            ));
        });
    }

    #[test]
    fn complete_watch_root_replacement_is_not_an_empty_authoritative_source() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source");
        fs::create_dir(&path).unwrap();
        let source = SourceRoot::open(path.clone()).unwrap();
        fs::rename(&path, directory.path().join("old-source")).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(source.check().is_err());
    }

    #[test]
    fn complete_watch_directory_hint_covers_current_and_renamed_subtrees_not_siblings() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            let nested = source.path.join("nested");
            fs::create_dir(&nested).unwrap();
            fs::write(nested.join("alpha.md"), "sharedtoken directory fixture").unwrap();
            let before = source.observe(&cx, &DiscoveryConfig::default()).unwrap();
            let mut changes = Changes::default();
            changes.record(Instant::now(), false);
            changes.record_path(&nested);
            let hint = changes.dirty.as_ref().unwrap();
            assert!(touches_observed_files(hint, &before, None));
            fs::rename(&nested, source.path.join("renamed")).unwrap();
            let after = source.observe(&cx, &DiscoveryConfig::default()).unwrap();
            assert!(touches_observed_files(hint, &after, Some(&before)));
            assert!(!touches_observed_files(hint, &after, None));
            let mut sibling = Changes::default();
            sibling.record(Instant::now(), false);
            sibling.record_path(&source.path.join("nested-sibling"));
            assert!(!touches_observed_files(
                sibling.dirty.as_ref().unwrap(),
                &after,
                Some(&before)
            ));
        });
    }

    #[test]
    fn complete_watch_overlap_preflight_creates_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let source = fs::canonicalize(directory.path()).unwrap();
        let nested = source.join("store");
        assert!(resolve_watch_store(&source, &nested).is_err());
        assert!(!nested.exists());
        assert!(resolve_watch_store(&nested, &source).is_err());
    }

    #[test]
    fn complete_watch_observation_detects_same_size_same_mtime_atomic_saves() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            let path = source.path.join("alpha.md");
            fs::write(&path, "first").unwrap();
            let modified = fs::metadata(&path).unwrap().modified().unwrap();
            let discovery = DiscoveryConfig::default();
            let before = source.observe(&cx, &discovery).unwrap();
            let replacement = source.path.join("replacement.tmp");
            fs::write(&replacement, "other").unwrap();
            File::options()
                .write(true)
                .open(&replacement)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(modified))
                .unwrap();
            fs::rename(&replacement, &path).unwrap();
            let after = source.observe(&cx, &discovery).unwrap();
            assert_eq!(before.stamps.len(), 1);
            assert_eq!(after.stamps.len(), 1);
            assert_ne!(before, after);
        });
    }

    #[test]
    fn complete_watch_cancelled_discovery_never_reports_an_empty_scan() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                source.observe(&cx, &DiscoveryConfig::default()),
                Err(SearchError::Cancelled { .. })
            ));
        });
    }

    #[test]
    fn complete_watch_cancellation_mid_listing_never_returns_partial_membership() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            for name in ["alpha.md", "beta.md", "gamma.md"] {
                fs::write(directory.path().join(name), "source content").unwrap();
            }
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            let mut visited = 0;
            let error = source
                .observe_with_entry_check(&cx, &DiscoveryConfig::default(), |_| {
                    visited += 1;
                    cx.set_cancel_requested(true);
                    Ok(())
                })
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(
                visited, 1,
                "stop inside a flat listing, not after traversal"
            );
            cx.set_cancel_requested(false);
            assert_eq!(
                source
                    .observe(&cx, &DiscoveryConfig::default())
                    .unwrap()
                    .stamps
                    .len(),
                3
            );
        });
    }

    #[test]
    fn complete_watch_unreadable_entry_preserves_original_error_not_partial_membership() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            fs::write(directory.path().join("alpha.md"), "source content").unwrap();
            let source = SourceRoot::open(fs::canonicalize(directory.path()).unwrap()).unwrap();
            let error = source
                .observe_with_entry_check(&cx, &DiscoveryConfig::default(), |_| {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "injected denied entry",
                    )
                    .into())
                })
                .unwrap_err();
            assert!(matches!(error, SearchError::Io(source)
                if source.kind() == std::io::ErrorKind::PermissionDenied
                    && source.to_string().contains("injected denied entry")));
        });
    }

    #[test]
    fn complete_watch_directory_mutation_mid_listing_refuses_the_observation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("source");
            fs::create_dir(&root).unwrap();
            fs::write(root.join("alpha.md"), "source content").unwrap();
            let source = SourceRoot::open(fs::canonicalize(&root).unwrap()).unwrap();
            let mut replaced = false;
            let error = source
                .observe_with_entry_check(&cx, &DiscoveryConfig::default(), |_| {
                    if !replaced {
                        fs::rename(&root, directory.path().join("retained-original"))?;
                        fs::create_dir(&root)?;
                        fs::write(root.join("alpha.md"), "replacement source")?;
                        replaced = true;
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(replaced);
            assert!(is_source_changed(&error));
            assert!(
                directory
                    .path()
                    .join("retained-original/alpha.md")
                    .is_file()
            );
        });
    }

    #[test]
    fn complete_watch_native_registration_observes_an_actual_write() {
        let directory = tempfile::tempdir().unwrap();
        let changes = Arc::new(Mutex::new(Changes::default()));
        let sink = Arc::clone(&changes);
        let mut watcher = notify::recommended_watcher(move |result| {
            record_notification(&sink, result, Instant::now());
        })
        .unwrap();
        watcher
            .watch(directory.path(), RecursiveMode::Recursive)
            .unwrap();
        fs::write(directory.path().join("alpha.md"), "native notification").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            check_backend(&changes).unwrap();
            if lock_changes(&changes).unwrap().dirty.is_some() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "native watcher produced no write notification"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(all(test, not(feature = "embedded-models")))]
mod lifecycle_tests {
    use super::*;
    use crate::generation_store::{COMPLETE_GENERATION_POINTER, CompleteGenerationStore};
    use crate::{CliCommand, CliInput, FsfsConfig};
    use asupersync::test_utils::run_test_with_cx;

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        let input = CliInput {
            command: CliCommand::Watch,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            format: OutputFormat::Jsonl,
            ..CliInput::default()
        };
        (FsfsRuntime::new(config).with_cli_input(input), source, root)
    }

    async fn publish_tick(session: &mut CompleteWatchSession, cx: &Cx) -> PublishedGeneration {
        // Inject the same coalescing hint used by the native callback. This
        // controls the scheduling boundary; indexing/admission remain real.
        for _ in 0..=MAX_UNSTABLE_BUILDS {
            let now = Instant::now();
            lock_changes(&session.changes).unwrap().record(now, false);
            if let Some(publication) = session.advance(cx, now + MAX_DEBOUNCE).await.unwrap() {
                return require_durable_publication(publication).unwrap();
            }
        }
        panic!("no publication after bounded settling attempts") // ubs:ignore — cfg(test) bounded wait must fail when publication never occurs.
    }

    fn controlled_notifications(session: &mut CompleteWatchSession) {
        let CompleteWatchSession {
            _watcher: watcher,
            source,
            ..
        } = &mut *session;
        watcher.unwatch(&source.path).unwrap();
        // The old callback retains its old state until it is dropped. Tests
        // inject hints into the same owning-lane state without backend timing.
        session.changes = Arc::new(Mutex::new(Changes::default()));
    }

    fn pause_after_races(session: &mut CompleteWatchSession, now: Instant) {
        for attempt in 1..=MAX_UNSTABLE_BUILDS {
            session.record_unstable_source(now).unwrap();
            assert_eq!(session.settling.is_some(), attempt == MAX_UNSTABLE_BUILDS);
        }
    }

    fn candidate_count(root: &Path) -> usize {
        fs::read_dir(root.join("generations")).unwrap().count()
    }

    #[test]
    fn complete_watch_hot_source_pauses_allocations_then_catches_up_without_an_event() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let before = publish_tick(&mut session, &cx).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            let allocated = candidate_count(&root);
            pause_after_races(&mut session, Instant::now());

            for revision in 0..5 {
                let now = session.settling.as_ref().unwrap().next_probe;
                let beta_path = session.source.path.join("beta.md");
                fs::write(&beta_path, format!("sharedtoken beta revision {revision}")).unwrap();
                {
                    let mut changes = lock_changes(&session.changes).unwrap();
                    changes.record(now, false);
                    changes.record_path(&beta_path);
                }
                assert!(session.advance(&cx, now).await.unwrap().is_none());
                assert!(session.settling.as_ref().unwrap().observation.is_none());
                assert_eq!(candidate_count(&root), allocated);
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(before.clone())
                );
            }
            // No event after the source settles. The paused state itself owes
            // a catch-up publication, and still uses the real retained builder.
            let now = session.settling.as_ref().unwrap().next_probe;
            assert!(session.advance(&cx, now).await.unwrap().is_none());
            assert!(session.settling.as_ref().unwrap().observation.is_some());
            let now = session.settling.as_ref().unwrap().next_probe;
            assert!(session.advance(&cx, now).await.unwrap().is_none());
            assert!(session.settling.is_none());
            assert_eq!(candidate_count(&root), allocated);
            let due = session.last_reconcile + DEBOUNCE;
            let publication = session.advance(&cx, due).await.unwrap().unwrap();
            let current = require_durable_publication(publication).unwrap();
            assert_ne!(before, current);
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                reader
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
            assert_eq!(
                pinned
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(pinned.generation(), &before);
        });
    }

    #[test]
    fn complete_watch_quiet_recovery_forces_catch_up_even_when_baseline_matches() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let before = publish_tick(&mut session, &cx).await;
            pause_after_races(&mut session, Instant::now());
            // First probe consumes the outstanding forced retry. It cannot be
            // used as one of the subsequent two quiet observations.
            for step in 0..3 {
                let now = session.settling.as_ref().unwrap().next_probe;
                assert!(session.advance(&cx, now).await.unwrap().is_none());
                assert_eq!(session.settling.is_none(), step == 2);
            }
            assert_eq!(
                session.baseline.as_ref(),
                Some(
                    &session
                        .source
                        .observe(&cx, &runtime.config.discovery)
                        .unwrap()
                )
            );
            assert!(
                lock_changes(&session.changes)
                    .unwrap()
                    .dirty
                    .as_ref()
                    .unwrap()
                    .force_rebuild
            );
            let due = session.last_reconcile + DEBOUNCE;
            let next =
                require_durable_publication(session.advance(&cx, due).await.unwrap().unwrap())
                    .unwrap();
            assert_ne!(
                next, before,
                "metadata equality must not erase the catch-up obligation"
            );
        });
    }

    #[test]
    fn complete_watch_initial_settling_observations_do_not_create_a_store() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            pause_after_races(&mut session, Instant::now());
            for step in 0..3 {
                let now = session.settling.as_ref().unwrap().next_probe;
                assert!(session.advance(&cx, now).await.unwrap().is_none());
                assert!(!root.exists(), "a quiet probe cannot allocate a candidate");
                assert_eq!(session.settling.is_none(), step == 2);
            }
            let due = session.last_reconcile + DEBOUNCE;
            let first = session.advance(&cx, due).await.unwrap().unwrap();
            assert!(require_durable_publication(first).unwrap().path().is_dir());
        });
    }

    #[test]
    fn complete_watch_hint_during_settling_probe_vetoes_quiet_proof_without_losing_hint() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let now = Instant::now();
            session.settling = Some(SettleWindow {
                next_probe: now,
                observation: Some(
                    session
                        .source
                        .observe(&cx, &runtime.config.discovery)
                        .unwrap(),
                ),
            });
            let changes = Arc::clone(&session.changes);
            session
                .probe_settling_source_with_entry_check(&cx, now, |path| {
                    let mut changes = lock_changes(&changes)?;
                    changes.record(now, false);
                    changes.record_path(path);
                    Ok(())
                })
                .unwrap();
            assert!(session.settling.as_ref().unwrap().observation.is_none());
            assert!(lock_changes(&session.changes).unwrap().dirty.is_some());
            assert!(!root.exists());
        });
    }

    #[test]
    fn complete_watch_backend_failure_during_settling_probe_cannot_queue_a_build() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let now = Instant::now();
            session.settling = Some(SettleWindow {
                next_probe: now,
                observation: Some(
                    session
                        .source
                        .observe(&cx, &runtime.config.discovery)
                        .unwrap(),
                ),
            });
            let changes = Arc::clone(&session.changes);
            let error = session
                .probe_settling_source_with_entry_check(&cx, now, |_| {
                    record_notification(
                        &changes,
                        Err(notify::Error::generic("probe backend failed")),
                        now,
                    );
                    Ok(())
                })
                .unwrap_err();
            assert!(error.to_string().contains("probe backend failed"));
            assert!(session.settling.is_some());
            assert!(lock_changes(&session.changes).unwrap().dirty.is_none());
            assert!(!root.exists());
        });
    }

    #[test]
    fn complete_watch_cancelled_settling_preserves_selection_without_allocation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let before = publish_tick(&mut session, &cx).await;
            let allocated = candidate_count(&root);
            pause_after_races(&mut session, Instant::now());
            cx.set_cancel_requested(true);
            let due = session.settling.as_ref().unwrap().next_probe;
            let error = session.advance(&cx, due).await.unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert!(session.settling.is_some());
            assert_eq!(candidate_count(&root), allocated);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(before)
            );
        });
    }

    #[test]
    fn complete_watch_external_publication_cannot_retarget_an_idle_source_baseline() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let first = publish_tick(&mut session, &cx).await;
            let outside = require_durable_publication(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let count = candidate_count(&root);
            let error = session.advance(&cx, Instant::now()).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.watch_selection_changed"));
            assert_eq!(candidate_count(&root), count);
            assert_eq!(session.publication.as_ref(), Some(&first));
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(outside)
            );
        });
    }

    #[test]
    fn complete_watch_candidate_refuses_a_publication_won_since_its_last_poll() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let first = publish_tick(&mut session, &cx).await;
            let outside = require_durable_publication(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let before = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            // Call the actual attempt after the external publication, as when
            // a writer wins after advance's check but before store.begin().
            let error = session.attempt(&cx, true, None).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.watch_selection_changed"));
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                before
            );
            assert_eq!(session.publication.as_ref(), Some(&first));
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap(), Some(outside));
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn complete_watch_missing_selection_while_settling_is_not_recreated() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let first = publish_tick(&mut session, &cx).await;
            pause_after_races(&mut session, Instant::now());
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let preserved = root.join("test-preserved-selection");
            fs::rename(&pointer, &preserved).unwrap();
            let count = candidate_count(&root);
            let due = session.settling.as_ref().unwrap().next_probe;
            assert!(session.advance(&cx, due).await.is_err());
            assert!(!pointer.exists());
            assert!(preserved.is_file());
            assert_eq!(session.publication.as_ref(), Some(&first));
            assert_eq!(candidate_count(&root), count);
        });
    }

    #[test]
    fn complete_watch_selection_change_during_quiet_probe_cannot_queue_catch_up() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            controlled_notifications(&mut session);
            let first = publish_tick(&mut session, &cx).await;
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let first_pointer = fs::read(&pointer).unwrap();
            let outside = require_durable_publication(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let outside_pointer = fs::read(&pointer).unwrap();
            // Both descriptors select real, admitted bundles. Replay them at
            // a deterministic point inside the production discovery traversal.
            let staged = root.join("test-selection-switch");
            fs::write(&staged, first_pointer).unwrap();
            fs::rename(&staged, &pointer).unwrap();
            let now = Instant::now();
            session.settling = Some(SettleWindow {
                next_probe: now,
                observation: Some(
                    session
                        .source
                        .observe(&cx, &runtime.config.discovery)
                        .unwrap(),
                ),
            });
            let error = session
                .probe_settling_source_with_entry_check(&cx, now, |_| {
                    fs::write(&staged, &outside_pointer)?;
                    fs::rename(&staged, &pointer)?;
                    Ok(())
                })
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.watch_selection_changed"));
            assert!(session.settling.is_some());
            assert!(lock_changes(&session.changes).unwrap().dirty.is_none());
            assert_eq!(session.publication.as_ref(), Some(&first));
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(outside)
            );
        });
    }

    #[test]
    fn complete_watch_keeps_readers_alive_across_add_delete_and_rename() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            let first = publish_tick(&mut session, &cx).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            let mut live = runtime.open_live_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            let second = publish_tick(&mut session, &cx).await;
            assert_ne!(first, second);
            assert_eq!(
                pinned
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(
                live.search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
            fs::rename(source.join("alpha.md"), source.join("renamed.md")).unwrap();
            fs::remove_file(source.join("beta.md")).unwrap();
            let third = publish_tick(&mut session, &cx).await;
            let result = live.search(&cx, "sharedtoken", 10).await.unwrap();
            let hits = &result.last().unwrap().hits;
            assert_eq!(hits.len(), 1);
            assert!(hits[0].path.ends_with("renamed.md"));
            assert_eq!(pinned.generation(), &first);
            assert!(first.path().is_dir());
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(third)
            );
            // The native watcher is STILL owned and registered at these reads.
            session.source.check().unwrap();
        });
    }

    #[test]
    fn complete_watch_periodic_reconciliation_recovers_a_missing_notification() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            publish_tick(&mut session, &cx).await;
            let CompleteWatchSession {
                _watcher: watcher,
                source: watched_source,
                ..
            } = &mut session;
            watcher.unwatch(&watched_source.path).unwrap();
            lock_changes(&session.changes).unwrap().dirty = None;
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            let deadline = session.last_reconcile + RECONCILE_INTERVAL;
            let publication = session.advance(&cx, deadline).await.unwrap().unwrap();
            require_durable_publication(publication).unwrap();
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                reader
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
        });
    }

    #[test]
    fn complete_watch_missing_source_preserves_current_instead_of_publishing_deletions() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = CompleteWatchSession::open(&runtime, &cx, &root).unwrap();
            let before = publish_tick(&mut session, &cx).await;
            fs::rename(source, directory.path().join("offline-source")).unwrap();
            assert!(
                session
                    .advance(&cx, Instant::now() + MAX_DEBOUNCE)
                    .await
                    .is_err()
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(before)
            );
        });
    }

    #[test]
    fn complete_watch_sink_failure_does_not_rollback_a_durable_publication() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut received = None;
            let error = runtime
                .watch_retained_generations(&cx, &root, |generation| {
                    received = Some(generation.clone());
                    Err(complete_cli_error("test_sink", "injected output failure"))
                })
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "complete_generation.test_sink")
            );
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap(), received);
            assert!(received.is_some());
            let next = store
                .begin(&cx)
                .expect("writer lease released after sink failure");
            drop(next);
        });
    }

    #[test]
    fn complete_watch_cancel_after_receipt_releases_the_writer_and_preserves_selection() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let mut delivered = 0;
            let error = runtime
                .watch_retained_generations(&cx, &root, |_| {
                    delivered += 1;
                    cx.set_cancel_requested(true);
                    Ok(())
                })
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(delivered, 1);
            cx.set_cancel_requested(false);
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert!(store.active(&cx).unwrap().is_some());
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn complete_watch_pre_cancelled_start_creates_no_store_and_emits_nothing() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            cx.set_cancel_requested(true);
            let mut delivered = 0;
            let error = runtime
                .watch_retained_generations(&cx, &root, |_| {
                    delivered += 1;
                    Ok(())
                })
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(delivered, 0);
            assert!(!root.exists());
        });
    }

    #[test]
    fn complete_watch_source_change_at_precommit_cannot_replace_current() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            require_durable_publication(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            let observed_source = SourceRoot::open(fs::canonicalize(&source).unwrap()).unwrap();
            let observed = observed_source
                .observe(&cx, &runtime.config.discovery)
                .unwrap();
            let error = runtime
                .rebuild_retained_generation_with_precommit(&cx, &root, |cx| {
                    fs::write(source.join("beta.md"), "sharedtoken beta document")?;
                    if observed_source.observe(cx, &runtime.config.discovery)? != observed {
                        return Err(source_changed());
                    }
                    Ok(())
                })
                .await
                .unwrap_err();
            assert!(is_source_changed(&error));
            assert_eq!(fs::read(pointer).unwrap(), before);
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert!(store.active(&cx).unwrap().is_some());
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn complete_watch_machine_output_requires_a_framed_format_before_any_write() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            runtime.cli_input.format = OutputFormat::Json;
            let mut output = Vec::new();
            let error = runtime
                .run_complete_generation_watch_with_writer(&cx, &root, &mut output)
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "complete_generation.watch_format")
            );
            assert!(output.is_empty());
            assert!(!root.exists());
        });
    }
}
