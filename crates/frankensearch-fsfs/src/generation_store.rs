//! Immutable complete-generation publication for the fsfs library.
//!
//! A rebuild writes to a fresh directory, not to the serving generation. After
//! engine-level admission succeeds, an inventory authenticates the entire bundle
//! and one atomic pointer switch publishes it. Abandoned builds and predecessors
//! are deliberately retained; neither cancellation nor Drop deletes data.
//!
//! This is a cooperative, trusted-directory protocol, not protection against a
//! hostile process replacing directory ancestors or mutating files outside the
//! publication-lease protocol. Synchronous filesystem work belongs on a caller's
//! blocking lane. No new runtime or detached worker is created here.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_storage::ContentHasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::lifecycle::PublicationLease;

/// A whole-bundle pointer, deliberately distinct from Quill's lexical CURRENT.
pub const COMPLETE_GENERATION_POINTER: &str = "FSFS-CURRENT";
/// The presence of a sealed inventory makes a generation read-only to writers.
pub const COMPLETE_GENERATION_MANIFEST: &str = "FSFS-BUNDLE.json";
const GENERATIONS: &str = "generations";
const POINTER_MAGIC: &str = "FSFS-COMPLETE-GENERATION-v1";
const MAX_POINTER_BYTES: u64 = 256;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 100_000;
const MAX_TREE_DEPTH: usize = 64;
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(0);

/// A canonical, stable root containing immutable generations and one pointer.
#[derive(Debug, Clone)]
pub struct CompleteGenerationStore {
    root: PathBuf,
}

/// One admitted generation. Retaining it pins a directory, not a changing CURRENT.
///
/// All reads for an operation must use this path. Do not resolve CURRENT again
/// between opening the lexical, vector, catalog, or content artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedGeneration {
    path: PathBuf,
    id: String,
    manifest_sha256: String,
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
    /// without changing its selection.
    ///
    /// # Errors
    /// Returns cancellation, unsupported-platform, path, or filesystem errors.
    pub fn create(cx: &Cx, root: &Path) -> SearchResult<Self> {
        checkpoint(cx)?;
        require_supported_platform()?;
        reject_symlink_if_present(root)?;
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
    /// Returns an error for a missing, non-directory, or symlinked root.
    pub fn open(cx: &Cx, root: &Path) -> SearchResult<Self> {
        checkpoint(cx)?;
        require_supported_platform()?;
        require_directory(root)?;
        reject_published_write(root)?;
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
    ///
    /// # Errors
    /// Returns an error for malformed selection, missing/extra/changed artifacts,
    /// cancellation, or unsupported filesystem objects.
    pub fn active(&self, cx: &Cx) -> SearchResult<Option<PublishedGeneration>> {
        checkpoint(cx)?;
        let Some(pointer) = read_pointer(&self.root)? else {
            return Ok(None);
        };
        let (id, manifest_sha256) = decode_pointer(&pointer, &self.root)?;
        let parent = self.root.join(GENERATIONS);
        require_directory(&parent)?;
        let path = parent.join(&id);
        require_directory(&path)?;
        let bytes =
            read_bounded_regular(&path.join(COMPLETE_GENERATION_MANIFEST), MAX_MANIFEST_BYTES)?;
        if digest(&bytes) != manifest_sha256 {
            return Err(invalid(
                &path,
                "bundle inventory digest does not match selection",
            ));
        }
        let manifest: BundleManifest = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(&path, &format!("invalid bundle inventory: {error}")))?;
        if manifest.schema_version != 1 || manifest.generation != id || manifest.files.is_empty() {
            return Err(invalid(
                &path,
                "invalid bundle inventory version, identity, or empty file set",
            ));
        }
        // Deriving a fresh sorted inventory also refuses traversal, duplicate
        // names, symlinks, unexpected files, and a file replaced by a directory.
        let actual = inventory(cx, &path, true)?;
        if actual != manifest.files {
            return Err(invalid(
                &path,
                "bundle files differ from their sealed inventory",
            ));
        }
        Ok(Some(PublishedGeneration {
            path,
            id,
            manifest_sha256,
        }))
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
        let generation = PublishedGeneration {
            path: self.path,
            id: self.id,
            manifest_sha256,
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

/// Refuse mutation of a sealed generation, including attempts through the
/// ordinary fsfs publication-lease entry point.
///
/// # Errors
/// Returns an error whenever a sealed marker exists, including a symlink marker.
pub(crate) fn reject_published_write(root: &Path) -> SearchResult<()> {
    match fs::symlink_metadata(root.join(COMPLETE_GENERATION_MANIFEST)) {
        Ok(_) => Err(invalid(
            root,
            "sealed complete generation is read-only; build a successor",
        )),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
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
    Ok((total, ContentHasher::to_hex(&hasher.finalize().into())))
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
    // another generation through a hard link.
    #[cfg(unix)]
    if metadata.nlink() != 1 {
        return Err(invalid(
            path,
            "hard-linked bundle artifacts are unsupported",
        ));
    }
    Ok(file)
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
            drop(store.begin(&cx).expect("failed precommit released the writer lease"));
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
