//! Immutable complete-generation publication for the fsfs library.
//!
//! A rebuild writes to a fresh directory, not to the serving generation. After
//! engine-level admission succeeds, an inventory authenticates the entire bundle
//! and one atomic pointer switch publishes it. Within one process, a file whose
//! identity and change stamps are unchanged since this process hashed it is not
//! hashed again (see `digest_memo`). Publication, cancellation and
//! Drop never delete data: predecessors and abandoned builds stay until an
//! explicit retention pass ([`CompleteGenerationStore::collect_retained`])
//! removes those older than the newest kept predecessors, skipping any
//! generation a live reader in any process pins.
//!
//! This is a cooperative, trusted-directory protocol, not protection against a
//! hostile process replacing directory ancestors or mutating files outside the
//! publication-lease protocol. Synchronous filesystem work belongs on a caller's
//! blocking lane. No new runtime or detached worker is created here.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_storage::ContentHasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::lifecycle::PublicationLease;

mod maintenance;
pub use maintenance::PreparedGenerationRestore;

/// A whole-bundle pointer, deliberately distinct from Quill's lexical CURRENT.
pub const COMPLETE_GENERATION_POINTER: &str = "FSFS-CURRENT";
/// The presence of a sealed inventory makes a generation read-only to writers.
pub const COMPLETE_GENERATION_MANIFEST: &str = "FSFS-BUNDLE.json";
const GENERATIONS: &str = "generations";
/// Per-generation reader pin files; never part of a sealed bundle.
const READER_PINS: &str = "readers";
const POINTER_MAGIC: &str = "FSFS-COMPLETE-GENERATION-v1";
const MAX_POINTER_BYTES: u64 = 256;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 100_000;
const MAX_TREE_DEPTH: usize = 64;
const MAX_ACTIVE_OPEN_ATTEMPTS: usize = 4;
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(0);

/// A canonical, stable root containing immutable generations and one pointer.
#[derive(Debug, Clone)]
pub struct CompleteGenerationStore {
    root: PathBuf,
}

/// One admitted generation. Retaining it pins a directory, not a changing CURRENT.
///
/// All reads for an operation must use this path. Do not resolve CURRENT again
/// between opening the lexical, vector, catalog, or content artifacts. While any
/// clone lives, retention in every process leaves this generation on disk.
#[derive(Debug, Clone)]
pub struct PublishedGeneration {
    path: PathBuf,
    id: String,
    manifest_sha256: String,
    /// Held only for its lock; dropping the last clone releases the pin.
    _pin: Option<Arc<GenerationPin>>,
}

impl PartialEq for PublishedGeneration {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.id == other.id
            && self.manifest_sha256 == other.manifest_sha256
    }
}

impl Eq for PublishedGeneration {}

/// A shared `flock(2)` on one generation's pin file under `readers/`, outside
/// every sealed bundle. Retention deletes a generation only while holding the
/// exclusive lock, so a live reader in any process keeps its bundle on disk,
/// and the kernel releases the pin when that process exits.
#[derive(Debug)]
struct GenerationPin {
    _file: File,
}

impl PublishedGeneration {
    /// Stable directory containing this generation's complete artifact set.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Opaque generation identifier, not an ordering or freshness assertion.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Digest binding the selected pointer to the exact bundle inventory.
    #[must_use]
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// The sealed inventory admission authenticated: each artifact's
    /// `/`-separated path within the bundle, with its length and SHA-256 hex.
    ///
    /// # Errors
    /// Returns an error when the inventory no longer matches the admitted digest.
    pub(crate) fn sealed_artifacts(&self) -> SearchResult<BTreeMap<String, (u64, String)>> {
        let bytes = read_bounded_regular(
            &self.path.join(COMPLETE_GENERATION_MANIFEST),
            MAX_MANIFEST_BYTES,
        )?;
        if digest(&bytes) != self.manifest_sha256 {
            return Err(invalid(
                &self.path,
                "bundle inventory digest changed after admission",
            ));
        }
        let manifest: BundleManifest = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(&self.path, &format!("invalid bundle inventory: {error}")))?;
        Ok(manifest
            .files
            .into_iter()
            .map(|artifact| (artifact.path, (artifact.bytes, artifact.sha256)))
            .collect())
    }
}

/// A successful pointer rename is not undone when the following fsync fails.
/// The caller must distinguish visibility from confirmed crash durability.
#[derive(Debug)]
#[must_use]
pub enum GenerationPublication {
    /// The pointer and preceding bundle writes have reached their sync boundary.
    Durable(PublishedGeneration),
    /// The new pointer is visible, but persistence of the directory entry is
    /// uncertain. Do not report this as an aborted or unpublished generation.
    VisibleButDurabilityUncertain {
        /// The generation selected by the completed rename.
        generation: PublishedGeneration,
        /// The failed directory synchronization.
        source: std::io::Error,
    },
}

/// Owns the publisher lease for an entire rebuild, including suspended futures.
/// A dropped value releases the lease and leaves its inert directory untouched.
#[derive(Debug)]
pub struct GenerationBuild {
    store: CompleteGenerationStore,
    path: PathBuf,
    id: String,
    predecessor: Option<Vec<u8>>,
    lease: PublicationLease,
}

impl CompleteGenerationStore {
    /// Create a store beneath an existing parent, or open an existing store,
    /// without changing its selection. A sealed generation and all of its
    /// descendants are refused before creating any directory.
    ///
    /// # Errors
    /// Returns cancellation, unsupported-platform, path, or filesystem errors.
    pub fn create(cx: &Cx, root: &Path) -> SearchResult<Self> {
        checkpoint(cx)?;
        require_supported_platform()?;
        reject_symlink_if_present(root)?;
        crate::lifecycle::reject_sealed_ancestry(root)?;
        checkpoint(cx)?;
        let created = match fs::create_dir(root) {
            Ok(()) => true,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => false,
            Err(error) => return Err(error.into()),
        };
        let store = Self::open(cx, root)?;
        if created {
            File::open(store.root())?.sync_all()?;
            if let Some(parent) = store.root().parent() {
                File::open(parent)?.sync_all()?;
            }
        }
        Ok(store)
    }

    /// Open an existing store without creating directories or lock files.
    ///
    /// # Errors
    /// Returns an error for a missing, non-directory, or symlinked root, or a
    /// root inside a sealed generation.
    pub fn open(cx: &Cx, root: &Path) -> SearchResult<Self> {
        checkpoint(cx)?;
        require_supported_platform()?;
        require_directory(root)?;
        crate::lifecycle::reject_sealed_ancestry(root)?;
        Ok(Self {
            root: fs::canonicalize(root)?,
        })
    }

    /// Canonical absolute store root; later CWD changes cannot retarget it.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve one descriptor and verify exactly the files it authenticates.
    /// An invalid descriptor never falls back to scanning for another generation.
    /// If publication and retention retire that descriptor's bundle before it
    /// can be pinned, retry the newly selected descriptor a bounded number of
    /// times. An admitted predecessor remains a valid result; successful opens
    /// are not restarted merely because a newer generation has been published.
    ///
    /// # Errors
    /// Returns an error for malformed selection, missing/extra/changed artifacts,
    /// cancellation, unsupported filesystem objects, or repeated selection churn.
    pub fn active(&self, cx: &Cx) -> SearchResult<Option<PublishedGeneration>> {
        self.active_with_opener(cx, Self::open_retained)
    }

    fn active_with_opener<F>(
        &self,
        cx: &Cx,
        mut open: F,
    ) -> SearchResult<Option<PublishedGeneration>>
    where
        F: FnMut(&Self, &Cx, &str, &str) -> SearchResult<PublishedGeneration>,
    {
        checkpoint(cx)?;
        let Some(mut pointer) = read_pointer(&self.root)? else {
            return Ok(None);
        };
        for _ in 0..MAX_ACTIVE_OPEN_ATTEMPTS {
            checkpoint(cx)?;
            let (id, manifest_sha256) = decode_pointer(&pointer, &self.root)?;
            let error = match open(self, cx, &id, &manifest_sha256) {
                Ok(generation) => return Ok(Some(generation)),
                Err(error) => error,
            };
            // Only a removed bundle or a collector's exclusive pin can be a
            // publication race. Never retry corruption, permission failures or
            // cancellation against a different generation and hide the error.
            if !matches!(&error, SearchError::Io(source)
                if matches!(source.kind(), ErrorKind::NotFound | ErrorKind::WouldBlock))
            {
                return Err(error);
            }
            checkpoint(cx)?;
            let Some(selected) = read_pointer(&self.root)? else {
                // Losing an observed selection is not an empty, healthy store.
                return Err(error);
            };
            if selected == pointer {
                return Err(error);
            }
            pointer = selected;
        }
        Err(SearchError::Io(std::io::Error::new(
            ErrorKind::WouldBlock,
            "complete-generation selection changed repeatedly before reader admission; retry the command",
        )))
    }

    /// Check whether an already admitted generation is still selected.
    ///
    /// This bounded descriptor read does not rehash the bundle or reopen any
    /// engine. It is a selection-change probe, not an integrity check: retain
    /// the previously admitted immutable reader while it returns true, and use
    /// `active` plus normal engine admission before installing a replacement.
    /// An absent selection returns false; malformed or unreadable selection is
    /// an error rather than permission to continue silently with stale results.
    ///
    /// # Errors
    /// Returns cancellation, invalid-descriptor, or filesystem errors.
    pub fn is_selected(&self, cx: &Cx, generation: &PublishedGeneration) -> SearchResult<bool> {
        checkpoint(cx)?;
        let Some(pointer) = read_pointer(&self.root)? else {
            return Ok(false);
        };
        let (id, manifest_sha256) = decode_pointer(&pointer, &self.root)?;
        Ok(generation.id == id
            && generation.manifest_sha256 == manifest_sha256
            && generation.path == self.root.join(GENERATIONS).join(id))
    }

    /// Reserve a fresh directory while excluding every other cooperating writer.
    ///
    /// # Errors
    /// Returns contention, cancellation, corrupt-current, or filesystem errors.
    pub fn begin(&self, cx: &Cx) -> SearchResult<GenerationBuild> {
        checkpoint(cx)?;
        let lease = PublicationLease::acquire(&self.root)?;
        lease.fence("complete-generation build entry")?;
        // A corrupt predecessor must be repaired explicitly, not silently
        // overwritten just because a new rebuild could be started.
        let _active = self.active(cx)?;
        let predecessor = read_pointer(&self.root)?;
        let parent = self.root.join(GENERATIONS);
        reject_symlink_if_present(&parent)?;
        fs::create_dir_all(&parent)?;
        require_directory(&parent)?;
        for _ in 0..64 {
            checkpoint(cx)?;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| invalid(&self.root, &format!("clock before epoch: {error}")))?;
            let serial = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
            let id = format!(
                "g-{:032x}-{:08x}-{:016x}",
                now.as_nanos(),
                std::process::id(),
                serial,
            );
            let path = parent.join(&id);
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(GenerationBuild {
                        store: self.clone(),
                        path,
                        id,
                        predecessor,
                        lease,
                    });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        Err(invalid(
            &parent,
            "unable to allocate a fresh generation name",
        ))
    }
}

impl GenerationBuild {
    /// The only directory a rebuild may mutate. All writers must be quiescent
    /// before `publish`; do not retain handles that can change sealed files.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Validate the engine-level bundle, seal its complete inventory, and switch
    /// the selection. No predecessor file is overwritten or removed.
    ///
    /// `validate` must run the consumer's real completeness/producer-identity
    /// admission, not merely check file existence. The store's hashes complement
    /// that semantic check; they are not a replacement for it.
    ///
    /// # Errors
    /// An error before the pointer rename leaves the predecessor selected.
    /// A failed sync after rename is returned as a distinct visible outcome.
    pub fn publish<F>(self, cx: &Cx, validate: F) -> SearchResult<GenerationPublication>
    where
        F: FnOnce(&Cx, &Path) -> SearchResult<()>,
    {
        self.publish_with_precommit(cx, validate, |_| Ok(()))
    }

    /// Recheck caller-owned source authority after sealing, before selection.
    ///
    /// `precommit` must be read-only: it may refuse publication but must not
    /// modify the candidate, selection or predecessor. It supplements, never
    /// replaces, engine admission and bundle integrity checks. Cancellation,
    /// the lease fence and predecessor comparison are repeated after this hook.
    pub(crate) fn publish_with_precommit<F, G>(
        self,
        cx: &Cx,
        validate: F,
        precommit: G,
    ) -> SearchResult<GenerationPublication>
    where
        F: FnOnce(&Cx, &Path) -> SearchResult<()>,
        G: FnOnce(&Cx) -> SearchResult<()>,
    {
        checkpoint(cx)?;
        self.lease.fence("complete-generation validation")?;
        reject_published_write(&self.path)?;
        validate(cx, &self.path)?;
        checkpoint(cx)?;
        let files = inventory(cx, &self.path, false)?;
        if files.is_empty() {
            return Err(invalid(&self.path, "cannot publish an empty bundle"));
        }
        let manifest = BundleManifest {
            schema_version: 1,
            generation: self.id.clone(),
            files,
        };
        let bytes = serde_json::to_vec(&manifest)
            .map_err(|error| invalid(&self.path, &format!("cannot encode inventory: {error}")))?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(invalid(
                &self.path,
                "bundle inventory exceeds its size bound",
            ));
        }
        let manifest_sha256 = digest(&bytes);
        write_new_synced(&self.path.join(COMPLETE_GENERATION_MANIFEST), &bytes)?;
        sync_tree(cx, &self.path, 0)?;
        File::open(self.store.root.join(GENERATIONS))?.sync_all()?;
        // Detect modifications during validation/sealing before any selection
        // change. Published readers also recheck this inventory on admission.
        if inventory(cx, &self.path, true)? != manifest.files {
            return Err(invalid(&self.path, "bundle changed while being sealed"));
        }
        let pointer = format!("{POINTER_MAGIC}\n{}\n{manifest_sha256}\n", self.id).into_bytes();
        let temporary = self.store.root.join(format!(".FSFS-CURRENT-{}", self.id));
        write_new_synced(&temporary, &pointer)?;
        precommit(cx)?;
        checkpoint(cx)?;
        self.lease
            .fence("complete-generation pointer publication")?;
        if read_pointer(&self.store.root)? != self.predecessor {
            return Err(invalid(
                &self.store.root,
                "selected predecessor changed during rebuild",
            ));
        }
        let pin = pin_generation(&self.store.root, &self.id)?;
        let generation = PublishedGeneration {
            path: self.path,
            id: self.id,
            manifest_sha256,
            _pin: pin,
        };
        fs::rename(
            &temporary,
            self.store.root.join(COMPLETE_GENERATION_POINTER),
        )?;
        // No cancellation point follows the linearization point. An observed
        // cancel cannot turn an already visible commit into a claimed abort.
        match File::open(&self.store.root).and_then(|directory| directory.sync_all()) {
            Ok(()) => Ok(GenerationPublication::Durable(generation)),
            Err(source) => {
                Ok(GenerationPublication::VisibleButDurabilityUncertain { generation, source })
            }
        }
    }
}

/// Whether `root` carries a sealed-generation marker (a symlink marker counts).
///
/// # Errors
/// Returns an error when the marker cannot be inspected.
pub(crate) fn is_sealed(root: &Path) -> SearchResult<bool> {
    match fs::symlink_metadata(root.join(COMPLETE_GENERATION_MANIFEST)) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// Refuse mutation of a sealed generation, including attempts through the
/// ordinary fsfs publication-lease entry point.
///
/// # Errors
/// Returns an error whenever a sealed marker exists, including a symlink marker.
pub(crate) fn reject_published_write(root: &Path) -> SearchResult<()> {
    if is_sealed(root)? {
        return Err(invalid(
            root,
            "sealed complete generation is read-only; build a successor",
        ));
    }
    Ok(())
}

/// Predecessors a complete-generation command keeps after it publishes.
///
/// A generation copies its predecessor except for the keyword segments the
/// two share, so one keeps a settled store under twice a fresh build while
/// leaving the previous generation available to restore.
pub const RETAINED_PREDECESSORS: usize = 1;

/// What a complete-generation command does with superseded generations after
/// a durable publication (`FSFS_GENERATION_RETENTION`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GenerationRetention {
    /// Remove generations older than the [`RETAINED_PREDECESSORS`] newest
    /// predecessors, and abandoned builds, unless a live reader pins them.
    #[default]
    Collect,
    /// Report what `Collect` would remove; remove nothing.
    Report,
    /// Keep every generation.
    Off,
}

impl GenerationRetention {
    /// Parse `collect`, `report`, or `off` (ASCII case-insensitive).
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "collect" => Some(Self::Collect),
            "report" => Some(Self::Report),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    /// The lowercase name `parse` accepts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Collect => "collect",
            Self::Report => "report",
            Self::Off => "off",
        }
    }
}

/// What one retention pass keeps or would reclaim. Planning never mutates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetentionPlan {
    /// The selected generation, which retention never removes.
    pub active: Option<String>,
    /// Sealed unselected generations kept as the newest predecessors.
    pub retained: Vec<String>,
    /// Generations a collection would remove, oldest first.
    pub reclaimable: Vec<ReclaimableGeneration>,
}

/// One generation directory a retention pass may remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReclaimableGeneration {
    /// Generation identifier (its directory name).
    pub id: String,
    /// Bytes removing it frees: a file it shares with a kept generation is not
    /// counted, and one several reclaimable generations share counts once,
    /// for the oldest.
    pub bytes: u64,
    /// False for an abandoned build that never sealed an inventory.
    pub sealed: bool,
}

/// What one collection removed and what live readers kept.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetentionReport {
    /// Removed generation identifiers, oldest first.
    pub removed: Vec<String>,
    /// Bytes the removed generations held (see [`ReclaimableGeneration::bytes`]).
    pub reclaimed_bytes: u64,
    /// Reclaimable generations a live reader still pins.
    pub pinned: Vec<String>,
}

impl CompleteGenerationStore {
    /// Plan retention without changing anything: keep the selection and the
    /// `keep_predecessors` newest other sealed generations; everything else
    /// under `generations/` (older sealed generations and abandoned unsealed
    /// builds) is reclaimable. Only directories named like generations count.
    ///
    /// # Errors
    /// Returns cancellation, malformed-selection, or filesystem errors. A
    /// malformed selection is never guessed around: nothing is reclaimable
    /// until it is repaired.
    pub fn plan_retention(&self, cx: &Cx, keep_predecessors: usize) -> SearchResult<RetentionPlan> {
        checkpoint(cx)?;
        let active = match read_pointer(&self.root)? {
            Some(pointer) => Some(decode_pointer(&pointer, &self.root)?.0),
            None => None,
        };
        let parent = self.root.join(GENERATIONS);
        let entries = match fs::read_dir(&parent) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(RetentionPlan {
                    active,
                    ..RetentionPlan::default()
                });
            }
            Err(error) => return Err(error.into()),
        };
        let mut sealed = Vec::new();
        let mut abandoned = Vec::new();
        for entry in entries {
            checkpoint(cx)?;
            let entry = entry?;
            let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !is_generation_id(&id) || active.as_deref() == Some(id.as_str()) {
                continue;
            }
            if !fs::symlink_metadata(entry.path())?.is_dir() {
                continue;
            }
            if is_sealed(&entry.path())? {
                sealed.push(id);
            } else {
                abandoned.push(id);
            }
        }
        // Identifiers start with a fixed-width creation timestamp, and builds
        // are serialized by the publication lease, so this is publication order.
        sealed.sort_unstable_by(|left, right| right.cmp(left));
        let older = sealed.split_off(keep_predecessors.min(sealed.len()));
        let mut candidates = older
            .into_iter()
            .map(|id| (id, true))
            .chain(abandoned.into_iter().map(|id| (id, false)))
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        // Shared segments the kept generations link are freed by no removal.
        let mut counted = HashSet::new();
        for id in active.iter().chain(&sealed) {
            match tree_bytes(cx, &parent.join(id), 0, &mut counted) {
                Ok(_) => {}
                Err(SearchError::Io(error)) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        let mut reclaimable = Vec::with_capacity(candidates.len());
        for (id, is_sealed) in candidates {
            let bytes = tree_bytes(cx, &parent.join(&id), 0, &mut counted)?;
            reclaimable.push(ReclaimableGeneration {
                id,
                bytes,
                sealed: is_sealed,
            });
        }
        Ok(RetentionPlan {
            active,
            retained: sealed,
            reclaimable,
        })
    }

    /// Remove what [`Self::plan_retention`] finds reclaimable, under the
    /// publication lease, skipping any generation a live reader pins.
    ///
    /// Holding the lease means no build is in progress, so an unsealed
    /// directory is an abandoned build, never a running one.
    /// The selected bundle must pass full admission before anything is removed.
    /// If selection is absent while sealed generations remain, preserve them
    /// for explicit recovery rather than treating every bundle as garbage.
    ///
    /// # Errors
    /// Returns lease contention (a build is running), cancellation,
    /// absent/corrupt selection, or filesystem errors; generations removed before
    /// an error stay removed.
    pub fn collect_retained(
        &self,
        cx: &Cx,
        keep_predecessors: usize,
    ) -> SearchResult<RetentionReport> {
        checkpoint(cx)?;
        let lease = PublicationLease::acquire(&self.root)?;
        lease.fence("complete-generation retention")?;
        let expected_selection = read_pointer(&self.root)?;
        let selected = self.active(cx)?;
        let plan = self.plan_retention(cx, keep_predecessors)?;
        if read_pointer(&self.root)? != expected_selection
            || plan.active.as_deref() != selected.as_ref().map(PublishedGeneration::id)
        {
            return Err(invalid(
                &self.root,
                "selection changed during retention admission; nothing was removed",
            ));
        }
        if selected.is_none()
            && (!plan.retained.is_empty()
                || plan.reclaimable.iter().any(|generation| generation.sealed))
        {
            return Err(invalid(
                &self.root,
                "sealed generations remain without a selection; restore an explicit trusted generation before collection",
            ));
        }
        let parent = self.root.join(GENERATIONS);
        let mut report = RetentionReport::default();
        for generation in plan.reclaimable {
            checkpoint(cx)?;
            lease.fence("complete-generation retention removal")?;
            // Readers pin only sealed generations; an abandoned build has none.
            let exclusive = if generation.sealed {
                match exclusive_pin(&self.root, &generation.id)? {
                    Some(pin) => Some(pin),
                    None => {
                        report.pinned.push(generation.id);
                        continue;
                    }
                }
            } else {
                None
            };
            let path = parent.join(&generation.id);
            require_directory(&path)?;
            fs::remove_dir_all(&path)?;
            if exclusive.is_some() {
                match fs::remove_file(self.root.join(READER_PINS).join(pin_name(&generation.id))) {
                    Ok(()) => {}
                    Err(error) if error.kind() == ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
            drop(exclusive);
            report.reclaimed_bytes = report.reclaimed_bytes.saturating_add(generation.bytes);
            report.removed.push(generation.id);
        }
        if !report.removed.is_empty() {
            File::open(&parent)?.sync_all()?;
        }
        Ok(report)
    }
}

/// Whether `name` has the exact shape [`CompleteGenerationStore::begin`] mints.
fn is_generation_id(name: &str) -> bool {
    let mut parts = name.split('-');
    parts.next() == Some("g")
        && parts.next().is_some_and(|part| hex(part, 32))
        && parts.next().is_some_and(|part| hex(part, 8))
        && parts.next().is_some_and(|part| hex(part, 16))
        && parts.next().is_none()
}

fn pin_name(id: &str) -> String {
    format!("{id}.pin")
}

#[cfg(unix)]
fn pin_open_flags() -> std::io::Result<i32> {
    // A substituted FIFO must not block before require_regular_pin can refuse it.
    let flags = rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK;
    i32::try_from(flags.bits()).map_err(|_| {
        std::io::Error::new(
            ErrorKind::Unsupported,
            "reader pin flags do not fit the open(2) flag word on this target",
        )
    })
}

#[cfg(unix)]
fn require_pin_directory(directory: &Path) -> std::io::Result<()> {
    if fs::symlink_metadata(directory)?.is_dir() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "reader pin directory is not a non-symlink directory",
        ))
    }
}

#[cfg(unix)]
fn require_regular_pin(file: File) -> std::io::Result<File> {
    if file.metadata()?.file_type().is_file() {
        Ok(file)
    } else {
        Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "reader pin is not a regular file",
        ))
    }
}

/// Open (creating if needed) one generation's pin file read-write. A new pin
/// file is as readable as the rest of the store (mode 0644 under the owner's
/// umask), so a reader without write access can still lock it shared through
/// [`open_existing_pin_read_only`].
#[cfg(unix)]
fn open_pin_file(root: &Path, id: &str) -> std::io::Result<File> {
    let directory = root.join(READER_PINS);
    match fs::create_dir(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    require_pin_directory(&directory)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o644)
        .custom_flags(pin_open_flags()?)
        .open(directory.join(pin_name(id)))?;
    require_regular_pin(file)
}

/// Open an existing pin file read-only: `flock(2)` needs no write access, so
/// this pins as firmly as a read-write handle on the same inode.
#[cfg(unix)]
fn open_existing_pin_read_only(root: &Path, id: &str) -> std::io::Result<File> {
    let directory = root.join(READER_PINS);
    require_pin_directory(&directory)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(pin_open_flags()?)
        .open(directory.join(pin_name(id)))?;
    require_regular_pin(file)
}

/// The pin file handle a reader or collector locks: read-write (creating it)
/// when it may, else the existing file read-only. `Err` carries both causes.
#[cfg(unix)]
fn lockable_pin_file(
    root: &Path,
    id: &str,
) -> Result<File, (std::io::Error, Option<std::io::Error>)> {
    match open_pin_file(root, id) {
        Ok(file) => Ok(file),
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem
            ) =>
        {
            open_existing_pin_read_only(root, id).map_err(|fallback| (error, Some(fallback)))
        }
        Err(error) => Err((error, None)),
    }
}

/// Take a shared pin on `id` for as long as the returned handle lives.
///
/// A reader without write access to the store (another user, a read-only
/// mount) locks the existing pin file read-only. A reader that cannot open it
/// at all is refused rather than admitted unpinned: a writable collector in
/// another process would otherwise be free to remove the generation (GH #60).
/// A collector holding the exclusive pin causes `WouldBlock`, not an unbounded
/// wait inside an otherwise cancellation-aware reader admission.
fn pin_generation(root: &Path, id: &str) -> SearchResult<Option<Arc<GenerationPin>>> {
    #[cfg(unix)]
    {
        let file = match lockable_pin_file(root, id) {
            Ok(file) => file,
            Err((error, None)) => return Err(error.into()),
            Err((write, Some(read))) => {
                return Err(SearchError::Io(std::io::Error::new(
                    write.kind(),
                    format!(
                        "cannot pin complete generation {id} for reading (read-write: {write}; \
                         read-only: {read}); without a pin, retention in another process could \
                         remove it while it is open. Run as the store owner, or have the owner \
                         publish again so that {READER_PINS}/{id}.pin exists and is readable"
                    ),
                )));
            }
        };
        rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockShared)
            .map_err(|errno| SearchError::Io(std::io::Error::from(errno)))?;
        Ok(Some(Arc::new(GenerationPin { _file: file })))
    }
    #[cfg(not(unix))]
    {
        let _ = (root, id);
        Ok(None)
    }
}

/// Try to take the exclusive pin retention needs before removing `id`.
/// `None` means a live reader holds it.
fn exclusive_pin(root: &Path, id: &str) -> SearchResult<Option<File>> {
    #[cfg(unix)]
    {
        let file = lockable_pin_file(root, id).map_err(|(error, _)| SearchError::Io(error))?;
        match rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => Ok(Some(file)),
            Err(rustix::io::Errno::AGAIN) => Ok(None),
            Err(errno) => Err(SearchError::Io(std::io::Error::from(errno))),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (root, id);
        Ok(None)
    }
}

/// Bytes held by regular files under `directory`, without following symlinks.
/// A file with several links counts only if `counted` lacks its inode, which
/// it then records.
fn tree_bytes(
    cx: &Cx,
    directory: &Path,
    depth: usize,
    counted: &mut HashSet<(u64, u64)>,
) -> SearchResult<u64> {
    checkpoint(cx)?;
    if depth > MAX_TREE_DEPTH {
        return Err(invalid(directory, "generation tree exceeds depth limit"));
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_dir() {
            total = total.saturating_add(tree_bytes(cx, &entry.path(), depth + 1, counted)?);
        } else if metadata.is_file() {
            #[cfg(unix)]
            if metadata.nlink() > 1 && !counted.insert((metadata.dev(), metadata.ino())) {
                continue;
            }
            total = total.saturating_add(metadata.len());
        }
    }
    #[cfg(not(unix))]
    let _ = counted;
    Ok(total)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    schema_version: u16,
    generation: String,
    files: Vec<Artifact>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    bytes: u64,
    sha256: String,
}

fn inventory(cx: &Cx, root: &Path, sealed: bool) -> SearchResult<Vec<Artifact>> {
    let mut files = Vec::new();
    inventory_at(cx, root, root, sealed, 0, &mut files)?;
    files.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

fn inventory_at(
    cx: &Cx,
    root: &Path,
    directory: &Path,
    sealed: bool,
    depth: usize,
    files: &mut Vec<Artifact>,
) -> SearchResult<()> {
    checkpoint(cx)?;
    if depth > MAX_TREE_DEPTH {
        return Err(invalid(directory, "bundle tree exceeds depth limit"));
    }
    require_directory(directory)?;
    for entry in fs::read_dir(directory)? {
        checkpoint(cx)?;
        let path = entry?.path();
        if sealed && path == root.join(COMPLETE_GENERATION_MANIFEST) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(&path, "symlinks are not bundle artifacts"));
        }
        if metadata.is_dir() {
            inventory_at(cx, root, &path, sealed, depth + 1, files)?;
        } else if metadata.is_file() {
            if files.len() >= MAX_ARTIFACTS {
                return Err(invalid(root, "too many bundle artifacts"));
            }
            let relative = path
                .strip_prefix(root)
                .map_err(|_| invalid(&path, "artifact escaped bundle root"))?;
            let mut components = Vec::new();
            for component in relative.components() {
                let Component::Normal(name) = component else {
                    return Err(invalid(&path, "invalid artifact path"));
                };
                components.push(
                    name.to_str()
                        .ok_or_else(|| invalid(&path, "artifact path is not UTF-8"))?,
                );
            }
            let (bytes, sha256) = hash_file(cx, &path)?;
            files.push(Artifact {
                path: components.join("/"),
                bytes,
                sha256,
            });
        } else {
            return Err(invalid(
                &path,
                "only regular files and directories may be published",
            ));
        }
    }
    Ok(())
}

fn hash_file(cx: &Cx, path: &Path) -> SearchResult<(u64, String)> {
    let mut file = open_regular(path)?;
    #[cfg(unix)]
    let stamp = digest_memo::Stamp::of(&file.metadata()?);
    #[cfg(unix)]
    if let Some(sha256) = digest_memo::recall(&stamp) {
        return Ok((stamp.bytes(), ContentHasher::to_hex(&sha256)));
    }
    let started = SystemTime::now();
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        checkpoint(cx)?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| invalid(path, "artifact length overflow"))?;
        hasher.update(&buffer[..count]);
    }
    let sha256: [u8; 32] = hasher.finalize().into();
    #[cfg(unix)]
    if total == stamp.bytes() && digest_memo::Stamp::of(&file.metadata()?) == stamp {
        digest_memo::remember(stamp, started, sha256);
    }
    #[cfg(not(unix))]
    let _ = started;
    Ok((total, ContentHasher::to_hex(&sha256)))
}

/// Digests this process computed for bundle files, keyed by file identity and
/// change stamps.
///
/// Every inventory pass used to re-read and re-hash the whole bundle, and one
/// complete-generation watch edit made about nine of them over the same
/// sealed files: `begin`, the batch base, the seed copy before and after, two
/// at sealing, retention (bd-dnqgr). A pass now reuses a digest computed in
/// this process while the file's device, inode, length, mtime and ctime are
/// unchanged, as git's index does. Any write moves ctime, which no caller can
/// set, and a digest is remembered only for a file whose ctime predates the
/// hash by more than the timestamp granularity, so a write in the same clock
/// tick as the previous one cannot go unseen. A new process starts empty and
/// reads every byte. What a memo gives up is noticing media corruption between
/// two passes in one process, which a re-read served from the page cache
/// rarely could; the seed copy still checks every byte it reads against the
/// sealed inventory, and a segment it links instead gets a new ctime, so the
/// predecessor's admission that follows the seeding hashes it again.
#[cfg(unix)]
mod digest_memo {
    use std::collections::HashMap;
    use std::fs::Metadata;
    use std::os::unix::fs::MetadataExt;
    use std::sync::{LazyLock, Mutex, PoisonError};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    /// Far above the kernel's coarse timestamp tick (at most 10 ms).
    const SETTLED: Duration = Duration::from_millis(100);
    /// Whole-second timestamps (ext4 with 128-byte inodes, FAT) need more.
    const SETTLED_COARSE: Duration = Duration::from_secs(2);
    /// A bound, not a working-set estimate: a store keeps a few generations.
    const MAX_ENTRIES: usize = 1 << 18;

    /// Device and inode to the stamp a digest was computed under.
    type Memo = HashMap<(u64, u64), (Stamp, [u8; 32])>;

    static MEMO: LazyLock<Mutex<Memo>> = LazyLock::new(Mutex::default);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct Stamp {
        pub(super) device: u64,
        pub(super) inode: u64,
        pub(super) bytes: u64,
        pub(super) modified: (i64, i64),
        pub(super) changed: (i64, i64),
    }

    impl Stamp {
        pub(super) fn of(metadata: &Metadata) -> Self {
            Self {
                device: metadata.dev(),
                inode: metadata.ino(),
                bytes: metadata.size(),
                modified: (metadata.mtime(), metadata.mtime_nsec()),
                changed: (metadata.ctime(), metadata.ctime_nsec()),
            }
        }

        pub(super) const fn bytes(&self) -> u64 {
            self.bytes
        }

        /// Whether every write after `hashed_at` must change this stamp.
        pub(super) fn settled_before(&self, hashed_at: SystemTime) -> bool {
            let (Ok(seconds), Ok(nanos)) =
                (u64::try_from(self.changed.0), u32::try_from(self.changed.1))
            else {
                return false;
            };
            let margin = if self.changed.1 == 0 && self.modified.1 == 0 {
                SETTLED_COARSE
            } else {
                SETTLED
            };
            UNIX_EPOCH
                .checked_add(Duration::new(seconds, nanos))
                .and_then(|changed| changed.checked_add(margin))
                .is_some_and(|settled| settled < hashed_at)
        }
    }

    pub(super) fn recall(stamp: &Stamp) -> Option<[u8; 32]> {
        let memo = MEMO.lock().unwrap_or_else(PoisonError::into_inner);
        memo.get(&(stamp.device, stamp.inode))
            .filter(|(remembered, _)| remembered == stamp)
            .map(|(_, sha256)| *sha256)
    }

    /// Remember a digest of bytes read from `hashed_at` on, while `stamp`
    /// held throughout.
    pub(super) fn remember(stamp: Stamp, hashed_at: SystemTime, sha256: [u8; 32]) {
        if !stamp.settled_before(hashed_at) {
            return;
        }
        let mut memo = MEMO.lock().unwrap_or_else(PoisonError::into_inner);
        if memo.len() >= MAX_ENTRIES {
            memo.clear();
        }
        memo.insert((stamp.device, stamp.inode), (stamp, sha256));
    }
}

/// Make this process remember `remembered`'s digest for `path` as it is now,
/// standing in for bytes that decayed on disk without a write.
#[cfg(all(test, unix))]
pub(crate) fn remember_digest_for_test(path: &Path, remembered: &[u8]) {
    let stamp = digest_memo::Stamp::of(&fs::metadata(path).unwrap());
    let later = SystemTime::now() + std::time::Duration::from_secs(60);
    digest_memo::remember(stamp, later, Sha256::digest(remembered).into());
}

fn sync_tree(cx: &Cx, root: &Path, depth: usize) -> SearchResult<()> {
    checkpoint(cx)?;
    if depth > MAX_TREE_DEPTH {
        return Err(invalid(root, "bundle tree exceeds depth limit"));
    }
    require_directory(root)?;
    for entry in fs::read_dir(root)? {
        checkpoint(cx)?;
        let path = entry?.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.is_dir() {
            sync_tree(cx, &path, depth + 1)?;
        } else if metadata.is_file() {
            open_regular(&path)?.sync_all()?;
        } else {
            return Err(invalid(
                &path,
                "non-regular artifact encountered during sync",
            ));
        }
    }
    File::open(root)?.sync_all()?;
    Ok(())
}

fn read_pointer(root: &Path) -> SearchResult<Option<Vec<u8>>> {
    let path = root.join(COMPLETE_GENERATION_POINTER);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let bytes = read_bounded_regular(&path, MAX_POINTER_BYTES)?;
    let _decoded = decode_pointer(&bytes, root)?;
    Ok(Some(bytes))
}

fn decode_pointer(bytes: &[u8], root: &Path) -> SearchResult<(String, String)> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid(root, "selection is not UTF-8"))?;
    let fields = text.split('\n').collect::<Vec<_>>();
    if fields.len() != 4 || fields[0] != POINTER_MAGIC || !fields[3].is_empty() {
        return Err(invalid(root, "invalid complete-generation pointer framing"));
    }
    let id = fields[1];
    let parts = id.split('-').collect::<Vec<_>>();
    if parts.len() != 4
        || parts[0] != "g"
        || !hex(parts[1], 32)
        || !hex(parts[2], 8)
        || !hex(parts[3], 16)
        || !hex(fields[2], 64)
    {
        return Err(invalid(
            root,
            "invalid complete-generation identity or digest",
        ));
    }
    Ok((id.to_owned(), fields[2].to_owned()))
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_bounded_regular(path: &Path, limit: u64) -> SearchResult<Vec<u8>> {
    let file = open_regular(path)?;
    if file.metadata()?.len() > limit {
        return Err(invalid(path, "descriptor is not a bounded regular file"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(invalid(path, "descriptor grew beyond its size bound"));
    }
    Ok(bytes)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> SearchResult<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn open_regular(path: &Path) -> SearchResult<File> {
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(invalid(path, "expected a non-symlink regular file"));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        let flags =
            i32::try_from((rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits())
                .map_err(|_| {
                    std::io::Error::new(
                        ErrorKind::Unsupported,
                        "no-follow file flags do not fit this target's open flag word",
                    )
                })?;
        options.custom_flags(flags);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid(path, "expected a regular file"));
    }
    // Never admit a generation whose bytes alias a mutable external file or
    // another generation through a hard link. A Quill segment is the one
    // exception: Quill writes it once and afterwards only renames or unlinks
    // it, so a successor seeded from its predecessor shares it (bd-dnqgr),
    // and every inventory pass still authenticates its bytes.
    #[cfg(unix)]
    if metadata.nlink() != 1 && !path.file_name().is_some_and(is_quill_segment_name) {
        return Err(invalid(
            path,
            "hard-linked bundle artifacts are unsupported",
        ));
    }
    Ok(file)
}

/// Whether `name` is a published Quill segment, `seg-<16 lowercase hex>.fslx`.
pub(crate) fn is_quill_segment_name(name: &std::ffi::OsStr) -> bool {
    name.to_str()
        .and_then(|name| name.strip_prefix("seg-"))
        .and_then(|name| name.strip_suffix(".fslx"))
        .is_some_and(|id| hex(id, 16))
}

fn require_directory(path: &Path) -> SearchResult<()> {
    if !fs::symlink_metadata(path)?.is_dir() {
        return Err(invalid(path, "expected a non-symlink directory"));
    }
    Ok(())
}

fn reject_symlink_if_present(path: &Path) -> SearchResult<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(invalid(path, "symlinked store roots are unsupported"))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.complete_generation".to_owned(),
        reason: error.to_string(),
    })
}

fn require_supported_platform() -> SearchResult<()> {
    if !cfg!(unix) {
        return Err(SearchError::Io(std::io::Error::new(
            ErrorKind::Unsupported,
            "complete-generation publication requires Unix directory fsync semantics",
        )));
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    ContentHasher::to_hex(&Sha256::digest(bytes).into())
}

fn invalid(path: &Path, detail: &str) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.to_owned(),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;

    #[test]
    fn active_retries_when_publication_and_collection_win_before_the_pin() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let retired = publish(&store, &cx, "old").id().to_owned();
            let mut attempts = 0;
            let reader = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    if attempts == 1 {
                        assert_eq!(id, retired);
                        drop(publish(store, cx, "new"));
                        assert_eq!(
                            store.collect_retained(cx, 0)?.removed,
                            std::slice::from_ref(&retired)
                        );
                    }
                    store.open_retained(cx, id, digest)
                })
                .unwrap()
                .unwrap();
            assert_eq!(attempts, 2);
            assert_eq!(store.active(&cx).unwrap(), Some(reader.clone()));
            assert_eq!(
                fs::read_to_string(reader.path().join("content.txt")).unwrap(),
                "new"
            );
            assert!(!root.path().join(GENERATIONS).join(retired).exists());
        });
    }

    #[test]
    fn active_keeps_a_successfully_pinned_predecessor_during_publication() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let old = publish(&store, &cx, "old");
            let mut attempts = 0;
            let reader = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    let pinned = store.open_retained(cx, id, digest)?;
                    drop(publish(store, cx, "new"));
                    assert!(store.collect_retained(cx, 0)?.removed.is_empty());
                    Ok(pinned)
                })
                .unwrap()
                .unwrap();
            assert_eq!(attempts, 1);
            assert_eq!(reader, old);
            assert!(!store.is_selected(&cx, &reader).unwrap());
            assert_eq!(
                fs::read_to_string(reader.path().join("content.txt")).unwrap(),
                "old"
            );
        });
    }

    #[test]
    fn active_never_retries_corruption_against_a_healthy_successor() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let old = publish(&store, &cx, "old");
            let mut attempts = 0;
            let error = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    drop(publish(store, cx, "new"));
                    fs::write(old.path().join("vector.idx"), "corrupt")?;
                    store.open_retained(cx, id, digest)
                })
                .unwrap_err();
            assert_eq!(attempts, 1);
            assert!(matches!(error, SearchError::IndexCorrupted { .. }));
            assert!(store.active(&cx).unwrap().is_some());
        });
    }

    #[test]
    fn active_does_not_retry_io_failures_without_a_different_selection() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "selected"));
            for kind in [
                ErrorKind::NotFound,
                ErrorKind::WouldBlock,
                ErrorKind::PermissionDenied,
            ] {
                let mut attempts = 0;
                let error = store
                    .active_with_opener(&cx, |_, _, _, _| {
                        attempts += 1;
                        Err(SearchError::Io(std::io::Error::new(
                            kind,
                            "injected I/O failure",
                        )))
                    })
                    .unwrap_err();
                assert_eq!(attempts, 1);
                assert!(matches!(error, SearchError::Io(error) if error.kind() == kind));
            }
        });
    }

    #[test]
    fn active_does_not_turn_a_lost_selection_into_an_empty_store() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "old"));
            let mut attempts = 0;
            let error = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    drop(publish(store, cx, "new"));
                    store.collect_retained(cx, 0)?;
                    fs::rename(
                        root.path().join(COMPLETE_GENERATION_POINTER),
                        root.path().join("saved-selection"),
                    )?;
                    store.open_retained(cx, id, digest)
                })
                .unwrap_err();
            assert_eq!(attempts, 1);
            assert!(matches!(error, SearchError::Io(error) if error.kind() == ErrorKind::NotFound));
        });
    }

    #[test]
    fn active_retries_are_bounded_even_when_every_selected_bundle_is_collected() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "initial"));
            let mut attempts = 0;
            let error = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    drop(publish(store, cx, "successor"));
                    store.collect_retained(cx, 0)?;
                    store.open_retained(cx, id, digest)
                })
                .unwrap_err();
            assert_eq!(attempts, MAX_ACTIVE_OPEN_ATTEMPTS);
            assert!(
                matches!(error, SearchError::Io(error) if error.kind() == ErrorKind::WouldBlock)
            );
            assert!(store.active(&cx).unwrap().is_some());
            assert_eq!(generation_names(root.path()).len(), 1);
        });
    }

    #[test]
    fn active_checks_cancellation_before_retrying_a_new_selection() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "old"));
            let mut attempts = 0;
            let error = store
                .active_with_opener(&cx, |store, cx, id, digest| {
                    attempts += 1;
                    drop(publish(store, cx, "new"));
                    store.collect_retained(cx, 0)?;
                    let result = store.open_retained(cx, id, digest);
                    cx.set_cancel_requested(true);
                    result
                })
                .unwrap_err();
            cx.set_cancel_requested(false);
            assert_eq!(attempts, 1);
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert!(store.active(&cx).unwrap().is_some());
        });
    }

    #[test]
    fn a_collectors_exclusive_pin_does_not_block_reader_admission() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let id = publish(&store, &cx, "selected").id().to_owned();
            let exclusive = exclusive_pin(root.path(), &id).unwrap().unwrap();
            std::thread::scope(|scope| {
                let (sender, receiver) = std::sync::mpsc::channel();
                let pin_root = root.path();
                let pin_id = id.as_str();
                let reader = scope.spawn(move || {
                    let refused = matches!(pin_generation(pin_root, pin_id),
                        Err(SearchError::Io(error)) if error.kind() == ErrorKind::WouldBlock);
                    sender.send(refused).unwrap();
                });
                let refusal = receiver.recv_timeout(std::time::Duration::from_secs(5));
                // Release even on timeout so reverting to blocking flock makes
                // the assertion fail instead of hanging the entire test suite.
                drop(exclusive);
                reader.join().unwrap();
                assert!(refusal.expect("reader admission waited for the exclusive pin"));
            });
            assert_eq!(store.active(&cx).unwrap().unwrap().id(), id);
        });
    }

    #[test]
    fn collection_preserves_recovery_candidates_when_selection_is_not_admissible() {
        run_test_with_cx(|cx| async move {
            for fault in [
                "artifact",
                "manifest",
                "bundle",
                "selection",
                "framing",
                "digest",
            ] {
                let root = tempfile::tempdir().unwrap();
                let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
                let old = publish(&store, &cx, "recoverable");
                let old_id = old.id().to_owned();
                let old_digest = old.manifest_sha256().to_owned();
                let old_path = old.path().to_path_buf();
                drop(old);
                let current = publish(&store, &cx, "current");
                let current_path = current.path().to_path_buf();
                let current_id = current.id().to_owned();
                drop(current);
                let pointer = root.path().join(COMPLETE_GENERATION_POINTER);
                match fault {
                    "artifact" => fs::write(current_path.join("vector.idx"), "corrupt").unwrap(),
                    "manifest" => {
                        fs::write(current_path.join(COMPLETE_GENERATION_MANIFEST), "corrupt")
                            .unwrap();
                    }
                    "bundle" => {
                        fs::rename(&current_path, root.path().join("saved-bundle")).unwrap();
                    }
                    "selection" => {
                        fs::rename(&pointer, root.path().join("saved-selection")).unwrap();
                    }
                    "framing" => fs::write(&pointer, "not a selection").unwrap(),
                    "digest" => {
                        fs::write(
                            &pointer,
                            format!("{POINTER_MAGIC}\n{current_id}\n{}\n", "0".repeat(64)),
                        )
                        .unwrap();
                    }
                    _ => unreachable!("fixture fault"), // ubs:ignore — cfg(test) fixture assertion.
                }
                let before = generation_names(root.path());
                for keep in [0, 2] {
                    assert!(
                        store.collect_retained(&cx, keep).is_err(),
                        "{fault}, keep={keep}"
                    );
                    assert_eq!(
                        generation_names(root.path()),
                        before,
                        "{fault}, keep={keep}"
                    );
                    assert_eq!(
                        fs::read_to_string(old_path.join("content.txt")).unwrap(),
                        "recoverable"
                    );
                }
                assert!(store.open_retained(&cx, &old_id, &old_digest).is_ok());
                drop(PublicationLease::acquire(root.path()).unwrap());
            }
        });
    }

    #[test]
    fn collection_can_still_remove_abandoned_unsealed_first_builds() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            assert!(store.collect_retained(&cx, 0).unwrap().removed.is_empty());
            let build = store.begin(&cx).unwrap();
            let abandoned = build.id.clone();
            write_bundle(build.path(), "never published");
            drop(build);
            assert!(store.active(&cx).unwrap().is_none());
            assert_eq!(store.collect_retained(&cx, 0).unwrap().removed, [abandoned]);
            assert!(generation_names(root.path()).is_empty());
        });
    }

    #[test]
    fn complete_watch_precommit_refusal_preserves_current_and_releases_the_lease() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let pointer = root.path().join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            let build = store.begin(&cx).unwrap();
            let candidate = build.path().to_path_buf();
            write_bundle(&candidate, "successor");
            let error = build
                .publish_with_precommit(
                    &cx,
                    |_, _| Ok(()),
                    |_| {
                        assert!(candidate.join(COMPLETE_GENERATION_MANIFEST).is_file());
                        assert_eq!(fs::read(&pointer).unwrap(), before);
                        Err(SearchError::InvalidConfig {
                            field: "test.source_authority".to_owned(),
                            value: String::new(),
                            reason: "source changed after sealing".to_owned(),
                        })
                    },
                )
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "test.source_authority"));
            assert_eq!(fs::read(&pointer).unwrap(), before);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
            assert!(candidate.join(COMPLETE_GENERATION_MANIFEST).is_file());
            drop(
                store
                    .begin(&cx)
                    .expect("failed precommit released the writer lease"),
            );
        });
    }

    #[test]
    fn complete_watch_cancellation_in_precommit_is_checked_before_the_rename() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "successor");
            let error = build
                .publish_with_precommit(
                    &cx,
                    |_, _| Ok(()),
                    |cx| {
                        cx.set_cancel_requested(true);
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn complete_watch_failed_engine_admission_never_reaches_precommit() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "invalid successor");
            let mut invoked = false;
            let error = build
                .publish_with_precommit(
                    &cx,
                    |_, path| Err(invalid(path, "real admission refused the candidate")),
                    |_| {
                        invoked = true;
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(matches!(error, SearchError::IndexCorrupted { .. }));
            assert!(!invoked);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
        });
    }

    #[test]
    fn complete_watch_successful_precommit_retains_the_complete_predecessor() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "successor");
            let candidate = build.path().to_path_buf();
            let mut checked = false;
            let publication = build
                .publish_with_precommit(
                    &cx,
                    |_, _| Ok(()),
                    |_| {
                        assert!(candidate.join(COMPLETE_GENERATION_MANIFEST).is_file());
                        checked = true;
                        Ok(())
                    },
                )
                .unwrap();
            assert!(checked);
            let GenerationPublication::Durable(second) = publication else {
                panic!("publication was not durable"); // ubs:ignore — cfg(test) assertion.
            };
            assert_eq!(store.active(&cx).unwrap(), Some(second));
            assert!(first.path().is_dir());
        });
    }

    #[test]
    fn complete_watch_predecessor_is_rechecked_after_precommit() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let pointer = root.path().join(COMPLETE_GENERATION_POINTER);
            let first_pointer = fs::read(&pointer).unwrap();
            let second = publish(&store, &cx, "second");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "third");
            let error = build
                .publish_with_precommit(
                    &cx,
                    |_, _| Ok(()),
                    |_| {
                        // Fault injection: simulate out-of-protocol replacement
                        // exactly between the source check and selection commit.
                        fs::write(&pointer, &first_pointer)?;
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(matches!(error, SearchError::IndexCorrupted { detail, .. }
                if detail.contains("predecessor changed")));
            assert_eq!(store.active(&cx).unwrap(), Some(first));
            assert!(second.path().is_dir());
            drop(store.begin(&cx).unwrap());
        });
    }

    #[test]
    fn bundle_digests_preserve_standard_sha256_hex() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("artifact");
            for (bytes, expected) in [
                (
                    b"".as_slice(),
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                ),
                (
                    b"abc".as_slice(),
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                ),
            ] {
                fs::write(&path, bytes).unwrap();
                assert_eq!(digest(bytes), expected);
                assert_eq!(
                    hash_file(&cx, &path).unwrap(),
                    (bytes.len() as u64, expected.to_owned())
                );
            }
            // Standard million-'a' vector spans multiple file-read buffers.
            let bytes = vec![b'a'; 1_000_000];
            let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
            fs::write(&path, &bytes).unwrap();
            assert_eq!(digest(&bytes), expected);
            assert_eq!(
                hash_file(&cx, &path).unwrap(),
                (1_000_000, expected.to_owned())
            );
        });
    }

    #[test]
    fn hashing_reuses_a_settled_digest_until_the_file_changes() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("artifact");
            fs::write(&path, b"abc").unwrap();
            std::thread::sleep(std::time::Duration::from_millis(150));
            assert_eq!(hash_file(&cx, &path).unwrap(), (3, digest(b"abc")));
            let stamp = digest_memo::Stamp::of(&fs::metadata(&path).unwrap());
            assert_eq!(
                digest_memo::recall(&stamp).map(|sha256| ContentHasher::to_hex(&sha256)),
                Some(digest(b"abc")),
                "a settled file's digest is remembered"
            );
            // Same length, new bytes: the write moves ctime.
            fs::write(&path, b"xyz").unwrap();
            assert_eq!(hash_file(&cx, &path).unwrap(), (3, digest(b"xyz")));
        });
    }

    #[test]
    fn digest_memo_keeps_only_stamps_a_later_write_must_change() {
        use std::time::Duration;
        let stamp = |seconds, nanos| digest_memo::Stamp {
            device: 1,
            inode: 2,
            bytes: 3,
            modified: (seconds, nanos),
            changed: (seconds, nanos),
        };
        let fine = UNIX_EPOCH + Duration::new(1_800_000_000, 500_000_000);
        assert!(
            !stamp(1_800_000_000, 500_000_000).settled_before(fine + Duration::from_millis(50))
        );
        assert!(
            stamp(1_800_000_000, 500_000_000).settled_before(fine + Duration::from_millis(150))
        );
        // Whole-second timestamps come from filesystems that store no more.
        let coarse = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        assert!(!stamp(1_800_000_000, 0).settled_before(coarse + Duration::from_secs(1)));
        assert!(stamp(1_800_000_000, 0).settled_before(coarse + Duration::from_secs(3)));
        assert!(!stamp(-1, 0).settled_before(SystemTime::now()));
    }

    fn write_bundle(path: &Path, value: &str) {
        fs::create_dir(path.join("lexical")).unwrap();
        for relative in [
            "lexical/MANIFEST",
            "vector.idx",
            "catalog.db",
            "content.txt",
        ] {
            fs::write(path.join(relative), value).unwrap();
        }
    }

    fn publish(store: &CompleteGenerationStore, cx: &Cx, value: &str) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        write_bundle(build.path(), value);
        match build.publish(cx, |_, _| Ok(())).unwrap() {
            GenerationPublication::Durable(generation) => generation,
            GenerationPublication::VisibleButDurabilityUncertain { source, .. } => {
                panic!("sync: {source}"); // ubs:ignore — cfg(test) assertion: uncertain durability must fail the test.
            }
        }
    }

    fn generation_names(root: &Path) -> Vec<String> {
        let mut names = fs::read_dir(root.join(GENERATIONS))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names
    }

    /// bd-2op1d: retention keeps the selection and the newest predecessors,
    /// reclaims older sealed generations and abandoned builds, and leaves
    /// anything not named like a generation alone. Planning changes nothing.
    #[test]
    fn retention_keeps_the_newest_predecessors_and_reclaims_the_rest() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let ids = (0..5)
                .map(|ordinal| publish(&store, &cx, &format!("v{ordinal}")).id().to_owned())
                .collect::<Vec<_>>();
            let abandoned = {
                let build = store.begin(&cx).unwrap();
                write_bundle(build.path(), "never sealed");
                build
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            };
            fs::create_dir(root.path().join(GENERATIONS).join("operator-notes")).unwrap();
            let before = generation_names(root.path());

            let plan = store.plan_retention(&cx, 2).unwrap();
            assert_eq!(plan.active.as_deref(), Some(ids[4].as_str()));
            assert_eq!(plan.retained, [ids[3].clone(), ids[2].clone()]);
            let reclaimable = plan
                .reclaimable
                .iter()
                .map(|generation| (generation.id.clone(), generation.sealed))
                .collect::<Vec<_>>();
            assert_eq!(
                reclaimable,
                [
                    (ids[0].clone(), true),
                    (ids[1].clone(), true),
                    (abandoned.clone(), false)
                ]
            );
            assert!(
                plan.reclaimable
                    .iter()
                    .all(|generation| generation.bytes > 0)
            );
            assert_eq!(
                generation_names(root.path()),
                before,
                "planning is a dry run"
            );

            let report = store.collect_retained(&cx, 2).unwrap();
            assert_eq!(report.removed, [ids[0].clone(), ids[1].clone(), abandoned]);
            assert!(report.pinned.is_empty());
            assert_eq!(
                report.reclaimed_bytes,
                plan.reclaimable
                    .iter()
                    .map(|generation| generation.bytes)
                    .sum::<u64>()
            );
            let mut kept = vec![
                ids[2].clone(),
                ids[3].clone(),
                ids[4].clone(),
                "operator-notes".to_owned(),
            ];
            kept.sort_unstable();
            assert_eq!(generation_names(root.path()), kept);
            assert_eq!(store.active(&cx).unwrap().unwrap().id(), ids[4]);
            assert!(store.plan_retention(&cx, 2).unwrap().reclaimable.is_empty());
        });
    }

    /// bd-2op1d: a generation a live reader holds is never removed, and it
    /// becomes reclaimable as soon as that reader is gone.
    #[test]
    fn retention_never_removes_a_generation_a_live_reader_pins() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "first"));
            let reader = store.active(&cx).unwrap().unwrap();
            let pinned = reader.id().to_owned();
            let second = publish(&store, &cx, "second").id().to_owned();
            drop(publish(&store, &cx, "third"));

            let report = store.collect_retained(&cx, 0).unwrap();
            assert_eq!(report.removed, [second]);
            assert_eq!(report.pinned, std::slice::from_ref(&pinned));
            assert!(reader.path().join(COMPLETE_GENERATION_MANIFEST).is_file());

            drop(reader);
            let report = store.collect_retained(&cx, 0).unwrap();
            assert_eq!(report.removed, [pinned]);
            assert!(report.pinned.is_empty());
        });
    }

    /// GH #60: a reader without write access to its pin file (another user, a
    /// read-only mount) still pins with a read-only handle, so a writable
    /// collector keeps G0 through the publication of G1 and G2.
    #[test]
    fn a_reader_without_pin_write_access_still_blocks_retention() {
        use std::os::unix::fs::PermissionsExt;
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let pinned = publish(&store, &cx, "first").id().to_owned();
            let pin_path = root.path().join(READER_PINS).join(pin_name(&pinned));
            fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o444)).unwrap();
            if !rustix::process::geteuid().is_root() {
                assert_eq!(
                    open_pin_file(root.path(), &pinned).unwrap_err().kind(),
                    ErrorKind::PermissionDenied,
                    "the reader below must take the read-only path"
                );
            }
            let reader = store.active(&cx).unwrap().unwrap();
            assert_eq!(reader.id(), pinned);
            let second = publish(&store, &cx, "second").id().to_owned();
            drop(publish(&store, &cx, "third"));

            let report = store.collect_retained(&cx, 1).unwrap();
            assert!(report.removed.is_empty());
            assert_eq!(report.pinned, std::slice::from_ref(&pinned));
            assert!(reader.path().join(COMPLETE_GENERATION_MANIFEST).is_file());
            let readmitted = store
                .open_retained(&cx, &pinned, reader.manifest_sha256())
                .unwrap();
            assert_eq!(readmitted, reader);
            drop(readmitted);

            drop(reader);
            let report = store.collect_retained(&cx, 1).unwrap();
            assert_eq!(report.removed, [pinned]);
            assert!(report.pinned.is_empty());
            assert!(generation_names(root.path()).contains(&second));
        });
    }

    /// GH #60: a reader that cannot open the pin file at all is refused with
    /// a typed error, never admitted unpinned.
    #[test]
    fn a_reader_that_cannot_open_its_pin_is_refused_not_admitted_unpinned() {
        use std::os::unix::fs::PermissionsExt;
        if rustix::process::geteuid().is_root() {
            // Root opens a mode-0000 file, so the refusal is unreachable here.
            return;
        }
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let id = publish(&store, &cx, "first").id().to_owned();
            let pin_path = root.path().join(READER_PINS).join(pin_name(&id));
            fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o000)).unwrap();

            let error = store.active(&cx).unwrap_err();
            assert!(
                matches!(
                    &error,
                    SearchError::Io(io)
                        if io.kind() == ErrorKind::PermissionDenied
                            && io.to_string().contains("cannot pin complete generation")
                ),
                "{error:?}"
            );

            fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o644)).unwrap();
            assert_eq!(store.active(&cx).unwrap().unwrap().id(), id);
        });
    }

    #[test]
    fn retention_refuses_to_run_while_a_build_holds_the_lease() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            drop(publish(&store, &cx, "first"));
            drop(publish(&store, &cx, "second"));
            let build = store.begin(&cx).unwrap();
            let before = generation_names(root.path());
            assert!(store.collect_retained(&cx, 0).is_err());
            assert_eq!(generation_names(root.path()), before);
            drop(build);
        });
    }

    #[test]
    fn failed_validation_and_dropped_build_preserve_complete_predecessor() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "first");
            let failed = store.begin(&cx).unwrap();
            let abandoned = failed.path().to_path_buf();
            write_bundle(&abandoned, "partial");
            assert!(
                failed
                    .publish(&cx, |_, path| Err(invalid(
                        path,
                        "injected admission failure"
                    )))
                    .is_err()
            );
            assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
            assert!(abandoned.is_dir());
            let dropped = store.begin(&cx).unwrap();
            let dropped_path = dropped.path().to_path_buf();
            drop(dropped);
            assert!(dropped_path.is_dir());
            assert_eq!(store.active(&cx).unwrap(), Some(first));
        });
    }

    #[test]
    fn successful_successor_preserves_old_pinned_artifacts() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let second = publish(&store, &cx, "new");
            assert_ne!(first.id(), second.id());
            assert_eq!(store.active(&cx).unwrap(), Some(second.clone()));
            for relative in [
                "lexical/MANIFEST",
                "vector.idx",
                "catalog.db",
                "content.txt",
            ] {
                assert_eq!(
                    fs::read_to_string(first.path().join(relative)).unwrap(),
                    "old"
                );
                assert_eq!(
                    fs::read_to_string(second.path().join(relative)).unwrap(),
                    "new"
                );
            }
        });
    }

    #[test]
    fn cancellation_before_publication_leaves_pointer_unchanged() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "new");
            cx.set_cancel_requested(true);
            assert!(matches!(
                build.publish(&cx, |_, _| Ok(())),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
        });
    }

    #[test]
    fn changed_predecessor_refuses_lost_update() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let first_pointer = fs::read(root.path().join(COMPLETE_GENERATION_POINTER)).unwrap();
            let second = publish(&store, &cx, "second");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "third");
            // Simulate an out-of-protocol replacement, not another admitted writer.
            fs::write(root.path().join(COMPLETE_GENERATION_POINTER), first_pointer).unwrap();
            assert!(build.publish(&cx, |_, _| Ok(())).is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(first));
            assert!(second.path().is_dir());
        });
    }

    #[test]
    fn corrupt_or_extra_artifacts_fail_closed_without_fallback() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let _old = publish(&store, &cx, "old");
            let active = publish(&store, &cx, "new");
            fs::write(active.path().join("vector.idx"), "bad").unwrap();
            assert!(store.active(&cx).is_err());
            fs::write(active.path().join("vector.idx"), "new").unwrap();
            fs::write(active.path().join("unexpected.idx"), "extra").unwrap();
            assert!(store.active(&cx).is_err());
        });
    }

    #[test]
    fn empty_bundle_and_symlinked_artifact_are_not_publishable() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let empty = store.begin(&cx).unwrap();
            assert!(empty.publish(&cx, |_, _| Ok(())).is_err());
            assert!(store.active(&cx).unwrap().is_none());
            let build = store.begin(&cx).unwrap();
            std::os::unix::fs::symlink("/dev/null", build.path().join("vector.idx")).unwrap();
            assert!(build.publish(&cx, |_, _| Ok(())).is_err());
            assert!(store.active(&cx).unwrap().is_none());
        });
    }

    #[test]
    fn pointer_traversal_wrong_version_and_trailing_data_are_rejected() {
        let root = Path::new("/unused");
        let id = format!("g-{}-{}-{}", "0".repeat(32), "0".repeat(8), "0".repeat(16));
        let hash = "0".repeat(64);
        assert!(
            decode_pointer(format!("{POINTER_MAGIC}\n{id}\n{hash}\n").as_bytes(), root).is_ok()
        );
        for text in [
            format!("{POINTER_MAGIC}\n../outside\n{hash}\n"),
            format!("{POINTER_MAGIC}\n/absolute\n{hash}\n"),
            format!("FSFS-COMPLETE-GENERATION-v2\n{id}\n{hash}\n"),
            format!("{POINTER_MAGIC}\n{id}\n{hash}\ntrailing"),
        ] {
            assert!(decode_pointer(text.as_bytes(), root).is_err());
        }
    }

    #[test]
    fn sealed_generation_refuses_reopening_as_writable_store() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "published");
            assert!(reject_published_write(generation.path()).is_err());
            assert!(CompleteGenerationStore::open(&cx, generation.path()).is_err());
            assert!(PublicationLease::acquire(generation.path()).is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn store_creation_refuses_retained_descendants_before_mkdir() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "retained");
            let pointer = fs::read(root.path().join(COMPLETE_GENERATION_POINTER)).unwrap();
            let before = inventory(&cx, generation.path(), true).unwrap();

            for parent in [
                generation.path().to_path_buf(),
                generation.path().join("lexical"),
            ] {
                let nested = parent.join("new-store");
                let error = CompleteGenerationStore::create(&cx, &nested)
                    .expect_err("retained descendants are immutable");
                assert!(matches!(error, SearchError::IndexCorrupted { .. }));
                assert!(!nested.exists(), "refusal must precede directory creation");
            }

            assert_eq!(inventory(&cx, generation.path(), true).unwrap(), before);
            assert_eq!(
                fs::read(root.path().join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn store_open_and_create_refuse_existing_retained_descendants() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "retained");
            let nested = generation.path().join("lexical");
            let before = inventory(&cx, generation.path(), true).unwrap();

            assert!(CompleteGenerationStore::open(&cx, &nested).is_err());
            assert!(CompleteGenerationStore::create(&cx, &nested).is_err());
            assert!(
                !nested
                    .join(crate::lifecycle::PUBLICATION_LOCK_FILE_NAME)
                    .exists()
            );
            assert_eq!(inventory(&cx, generation.path(), true).unwrap(), before);
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn store_parent_alias_cannot_hide_a_retained_ancestor() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "retained");
            let alias = root.path().join("alias");
            std::os::unix::fs::symlink(generation.path(), &alias).unwrap();
            let nested = alias.join("lexical");
            let before = inventory(&cx, generation.path(), true).unwrap();

            assert!(CompleteGenerationStore::open(&cx, &nested).is_err());
            assert!(CompleteGenerationStore::create(&cx, &nested.join("new-store")).is_err());
            assert!(!generation.path().join("lexical/new-store").exists());
            assert_eq!(inventory(&cx, generation.path(), true).unwrap(), before);
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn store_opened_before_ancestor_sealing_cannot_start_a_writer() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "retained");
            let nested = build.path().join("lexical");
            let opened = CompleteGenerationStore::open(&cx, &nested).unwrap();
            let publication = build.publish(&cx, |_, _| Ok(())).unwrap();
            let GenerationPublication::Durable(generation) = publication else {
                panic!("publication was not durable"); // ubs:ignore — cfg(test) assertion.
            };
            let before = inventory(&cx, generation.path(), true).unwrap();

            assert!(opened.begin(&cx).is_err());
            assert!(
                !nested
                    .join(crate::lifecycle::PUBLICATION_LOCK_FILE_NAME)
                    .exists()
            );
            assert!(!nested.join(GENERATIONS).exists());
            assert_eq!(inventory(&cx, generation.path(), true).unwrap(), before);
            assert_eq!(store.active(&cx).unwrap(), Some(generation));
        });
    }

    #[test]
    fn store_creation_through_a_mutable_parent_alias_still_publishes() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let parent = root.path().join("mutable");
            fs::create_dir(&parent).unwrap();
            let alias = root.path().join("alias");
            std::os::unix::fs::symlink(&parent, &alias).unwrap();
            let store = CompleteGenerationStore::create(&cx, &alias.join("store")).unwrap();
            assert_eq!(
                store.root(),
                fs::canonicalize(parent.join("store")).unwrap().as_path()
            );

            let first = publish(&store, &cx, "first");
            let second = publish(&store, &cx, "second");
            assert_eq!(store.active(&cx).unwrap(), Some(second));
            assert_eq!(
                fs::read_to_string(first.path().join("content.txt")).unwrap(),
                "first"
            );
        });
    }

    #[test]
    fn hard_links_cannot_import_mutable_external_backing() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let outside = tempfile::tempdir().unwrap();
            let external = outside.path().join("external.idx");
            fs::write(&external, "mutable").unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let build = store.begin(&cx).unwrap();
            fs::hard_link(&external, build.path().join("vector.idx")).unwrap();
            assert!(build.publish(&cx, |_, _| Ok(())).is_err());
            assert!(store.active(&cx).unwrap().is_none());
            assert_eq!(fs::read_to_string(external).unwrap(), "mutable");
        });
    }

    #[test]
    fn generations_share_only_quill_segments_and_retention_counts_each_once() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let segment = |id: u64| format!("lexical/quill-v1/seg-{id:016x}.fslx");
            // Each build shares its predecessor's newest segment, as a seed does.
            let mut generations: Vec<PublishedGeneration> = Vec::new();
            for (id, length) in [(1_u64, 100_usize), (2, 1000), (3, 0)] {
                let build = store.begin(&cx).unwrap();
                fs::create_dir_all(build.path().join("lexical/quill-v1")).unwrap();
                fs::write(build.path().join("content.txt"), [b'c'; 10]).unwrap();
                if let Some(previous) = generations.last() {
                    let shared = segment(id - 1);
                    fs::hard_link(previous.path().join(&shared), build.path().join(&shared))
                        .unwrap();
                }
                if length > 0 {
                    fs::write(build.path().join(segment(id)), vec![b's'; length]).unwrap();
                }
                let GenerationPublication::Durable(generation) =
                    build.publish(&cx, |_, _| Ok(())).unwrap()
                else {
                    panic!("fixture publication must be durable"); // ubs:ignore — cfg(test) assertion.
                };
                generations.push(generation);
            }
            let oldest = &generations[0];
            store
                .open_retained(&cx, oldest.id(), oldest.manifest_sha256())
                .unwrap();
            let inventory_bytes = |generation: &PublishedGeneration| {
                fs::metadata(generation.path().join(COMPLETE_GENERATION_MANIFEST))
                    .unwrap()
                    .len()
            };
            // The oldest frees its share of the first segment; the second
            // frees neither segment, since the selection keeps the newer one.
            let expected = [
                (
                    generations[0].id().to_owned(),
                    inventory_bytes(&generations[0]) + 110,
                ),
                (
                    generations[1].id().to_owned(),
                    inventory_bytes(&generations[1]) + 10,
                ),
            ];
            let active = generations.pop().unwrap();
            drop(generations);
            let plan = store.plan_retention(&cx, 0).unwrap();
            let planned = plan
                .reclaimable
                .iter()
                .map(|generation| (generation.id.clone(), generation.bytes))
                .collect::<Vec<_>>();
            assert_eq!(planned, expected);
            let report = store.collect_retained(&cx, 0).unwrap();
            assert_eq!(
                report.reclaimed_bytes,
                expected.iter().map(|(_, bytes)| bytes).sum::<u64>()
            );
            assert_eq!(store.active(&cx).unwrap(), Some(active.clone()));
            assert_eq!(
                fs::metadata(active.path().join(segment(2)))
                    .unwrap()
                    .nlink(),
                1
            );

            // Any other shared artifact is still refused.
            let build = store.begin(&cx).unwrap();
            fs::hard_link(
                active.path().join("content.txt"),
                build.path().join("content.txt"),
            )
            .unwrap();
            assert!(build.publish(&cx, |_, _| Ok(())).is_err());
            for (name, is_segment) in [
                ("seg-0123456789abcdef.fslx", true),
                ("seg-0123456789ABCDEF.fslx", false),
                ("seg-0123.fslx", false),
                ("seg-0123456789abcdef.fslx.fec", false),
                ("MANIFEST", false),
            ] {
                assert_eq!(
                    is_quill_segment_name(std::ffi::OsStr::new(name)),
                    is_segment,
                    "{name}"
                );
            }
        });
    }

    #[test]
    fn readers_keep_the_complete_bundle_while_a_successor_is_being_built() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "complete");
            let pending = store.begin(&cx).unwrap();
            fs::write(pending.path().join("vector.idx"), "incomplete").unwrap();
            for _ in 0..4 {
                assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
                assert_eq!(
                    fs::read_to_string(first.path().join("catalog.db")).unwrap(),
                    "complete"
                );
            }
            drop(pending);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
        });
    }

    #[test]
    fn occupied_pointer_temporary_is_not_overwritten() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let build = store.begin(&cx).unwrap();
            write_bundle(build.path(), "new");
            let temporary = root.path().join(format!(".FSFS-CURRENT-{}", build.id));
            fs::write(&temporary, "retain this evidence").unwrap();
            assert!(build.publish(&cx, |_, _| Ok(())).is_err());
            assert_eq!(
                fs::read_to_string(temporary).unwrap(),
                "retain this evidence"
            );
            assert_eq!(store.active(&cx).unwrap(), Some(first));
        });
    }

    #[test]
    fn dropping_suspended_rebuild_releases_lease_without_publishing() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let mut future = Box::pin(async {
                let build = store.begin(&cx).unwrap();
                fs::write(build.path().join("partial.idx"), "partial").unwrap();
                std::future::pending::<()>().await;
                drop(build);
            });
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            drop(future);
            assert_eq!(store.active(&cx).unwrap(), Some(first));
            let retry = store
                .begin(&cx)
                .expect("abandoned future released its lease");
            drop(retry);
        });
    }

    #[test]
    fn selection_probe_ignores_staging_and_detects_publication() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let first = publish(&store, &cx, "old");
            assert!(store.is_selected(&cx, &first).unwrap());
            let pending = store.begin(&cx).unwrap();
            fs::write(pending.path().join("partial.idx"), "partial").unwrap();
            assert!(store.is_selected(&cx, &first).unwrap());
            drop(pending);
            let second = publish(&store, &cx, "new");
            assert!(!store.is_selected(&cx, &first).unwrap());
            assert!(store.is_selected(&cx, &second).unwrap());
            assert_eq!(
                fs::read_to_string(first.path().join("content.txt")).unwrap(),
                "old"
            );
        });
    }

    #[test]
    fn selection_probe_binds_root_identity_and_manifest_digest() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "complete");
            let other_root = tempfile::tempdir().unwrap();
            let other = CompleteGenerationStore::create(&cx, other_root.path()).unwrap();
            // Even an identical descriptor in another root cannot identify the
            // first store's retained reader as that other store's selection.
            fs::copy(
                store.root().join(COMPLETE_GENERATION_POINTER),
                other.root().join(COMPLETE_GENERATION_POINTER),
            )
            .unwrap();
            assert!(!other.is_selected(&cx, &generation).unwrap());
            let mut wrong_digest = generation.clone();
            wrong_digest.manifest_sha256 = "0".repeat(64);
            assert!(!store.is_selected(&cx, &wrong_digest).unwrap());
            assert!(store.is_selected(&cx, &generation).unwrap());
        });
    }

    #[test]
    fn selection_probe_distinguishes_absence_corruption_and_cancellation() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "complete");
            let pointer = root.path().join(COMPLETE_GENERATION_POINTER);
            let saved = root.path().join("saved-pointer");
            fs::rename(&pointer, &saved).unwrap();
            assert!(!store.is_selected(&cx, &generation).unwrap());
            fs::write(&pointer, "not a complete-generation pointer").unwrap();
            assert!(store.is_selected(&cx, &generation).is_err());
            fs::rename(&saved, &pointer).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                store.is_selected(&cx, &generation),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(store.is_selected(&cx, &generation).unwrap());
        });
    }

    #[test]
    fn selection_probe_is_not_a_substitute_for_bundle_admission() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&store, &cx, "complete");
            // Deliberately violate immutability to distinguish the cheap
            // selection probe from the full integrity gate used on new opens.
            fs::write(generation.path().join("vector.idx"), "tampered").unwrap();
            assert!(store.is_selected(&cx, &generation).unwrap());
            assert!(store.active(&cx).is_err());
        });
    }
}
