//! Explicit recovery and durability barriers for an immutable complete store.
//!
//! No directory scan chooses a recovery target. The caller supplies an admitted
//! generation, or an ID plus manifest digest from a trusted publication receipt.
//! Preparing a restore binds the exact incumbent descriptor (even a malformed
//! one); committing it rechecks both that evidence and the complete target.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind};
use std::path::Path;
use std::sync::atomic::Ordering;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::{
    BundleManifest, COMPLETE_GENERATION_MANIFEST, COMPLETE_GENERATION_POINTER,
    CompleteGenerationStore, GENERATIONS, GenerationPublication, MAX_MANIFEST_BYTES,
    MAX_POINTER_BYTES, NEXT_GENERATION, POINTER_MAGIC, PublicationLease, PublishedGeneration,
    checkpoint, decode_pointer, digest, hex, invalid, inventory, open_regular,
    read_bounded_regular, require_directory, sync_tree, write_new_synced,
};

/// An explicit restore bound to one store, one target and one observed selection.
///
/// This does not hold a publisher lease between preparation and execution. A
/// concurrent publication invalidates the preparation instead of being lost.
/// Restoring an older generation is an intentional rollback, not an automatic
/// recovery policy or the hostile-directory generation-v2 antirollback protocol.
#[must_use]
pub struct PreparedGenerationRestore {
    store: CompleteGenerationStore,
    target: PublishedGeneration,
    expected_selection: Option<Vec<u8>>,
}

impl fmt::Debug for PreparedGenerationRestore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedGenerationRestore")
            .field("target", &self.target)
            .field("selection_present", &self.expected_selection.is_some())
            // A malformed selection can contain arbitrary private bytes.
            .finish_non_exhaustive()
    }
}

impl CompleteGenerationStore {
    /// Reopen an explicitly named retained generation without changing selection.
    ///
    /// `manifest_sha256` must come from a trusted publication receipt, not be
    /// computed from a candidate merely to make it pass. This uses the same
    /// full-inventory gate as [`Self::active`]. It never scans for a fallback or
    /// consults the active descriptor, so an intact predecessor can be admitted
    /// even when the selected generation is corrupt. Engine/producer admission
    /// remains the caller's responsibility, as with `active`.
    ///
    /// # Errors
    /// Returns invalid-identity, missing/changed-artifact, I/O or cancellation errors.
    pub fn open_retained(
        &self,
        cx: &Cx,
        id: &str,
        manifest_sha256: &str,
    ) -> SearchResult<PublishedGeneration> {
        checkpoint(cx)?;
        // Bound untrusted strings before constructing either a path or pointer.
        if id.len() != 60 || !hex(manifest_sha256, 64) {
            return Err(invalid(
                &self.root,
                "invalid retained generation identity or digest",
            ));
        }
        let pointer = format!("{POINTER_MAGIC}\n{id}\n{manifest_sha256}\n");
        let _decoded = decode_pointer(pointer.as_bytes(), &self.root)?;
        let parent = self.root.join(GENERATIONS);
        require_directory(&parent)?;
        let path = parent.join(id);
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
        if inventory(cx, &path, true)? != manifest.files {
            return Err(invalid(
                &path,
                "bundle files differ from their sealed inventory",
            ));
        }
        checkpoint(cx)?;
        Ok(PublishedGeneration {
            path,
            id: id.to_owned(),
            manifest_sha256: manifest_sha256.to_owned(),
        })
    }

    /// Prepare an intentional rollback or repair of a damaged/missing selection.
    ///
    /// The target must belong to this store and still match its trusted digest.
    /// A bounded malformed descriptor may be replaced, but symlinks, hard links,
    /// special files and oversized descriptors remain errors. No generation or
    /// descriptor is changed during preparation.
    ///
    /// # Errors
    /// Returns contention, foreign/changed target, unsafe descriptor, I/O or
    /// cancellation errors. An in-flight build is never implicitly aborted.
    pub fn prepare_restore(
        &self,
        cx: &Cx,
        target: &PublishedGeneration,
    ) -> SearchResult<PreparedGenerationRestore> {
        checkpoint(cx)?;
        if target.path != self.root.join(GENERATIONS).join(&target.id) {
            return Err(invalid(
                &self.root,
                "restore target belongs to another store",
            ));
        }
        let lease = PublicationLease::acquire(&self.root)?;
        lease.fence("complete-generation restore preparation")?;
        let expected_selection = selection_evidence(&self.root)?;
        let target = self.open_retained(cx, target.id(), target.manifest_sha256())?;
        ensure_selection(&self.root, expected_selection.as_deref())?;
        checkpoint(cx)?;
        lease.fence("complete-generation restore preparation complete")?;
        drop(lease);
        Ok(PreparedGenerationRestore {
            store: self.clone(),
            target,
            expected_selection,
        })
    }

    /// Confirm durability of the already selected complete generation.
    ///
    /// Revalidates the complete bundle, syncs its files/directories and selection,
    /// and fences the publisher lease. It neither modifies sealed engine files
    /// nor allocates a generation. This can resolve a prior
    /// [`GenerationPublication::VisibleButDurabilityUncertain`] outcome.
    ///
    /// This is not a source rescan or a request to drain another process's watch
    /// queue. An active publisher causes contention, never a misleading success
    /// for a candidate that has not been published.
    ///
    /// # Errors
    /// Returns contention, absent/corrupt selection, I/O or cancellation errors.
    /// A failed final directory sync preserves the distinct visible outcome.
    pub fn flush_selected(&self, cx: &Cx) -> SearchResult<GenerationPublication> {
        self.flush_with_sync(cx, sync_directory)
    }

    /// Confirm durability only while the exact receipted generation is selected.
    ///
    /// Unlike [`Self::flush_selected`], this binds confirmation to the caller's
    /// expected generation, including its store path, ID and manifest digest.
    /// The comparison happens under publication ownership, before artifact sync,
    /// so a concurrent publisher cannot redirect the confirmation to a successor.
    /// Reopen a trusted receipt with [`Self::open_retained`] after process restart.
    ///
    /// No selection is written, no generation is created or deleted, and no model
    /// is loaded. This confirms filesystem durability, not semantic readiness or
    /// a drained watcher queue. A superseded target requires a new explicit
    /// decision; it is never republished merely to make confirmation succeed.
    ///
    /// # Errors
    /// Returns contention, changed/foreign/absent/corrupt selection, I/O or
    /// cancellation errors. Final-sync failure retains the visible-uncertain
    /// outcome; success is bound to the expected generation at the barrier.
    pub fn confirm_retained_durability(
        &self,
        cx: &Cx,
        expected: &PublishedGeneration,
    ) -> SearchResult<GenerationPublication> {
        self.flush_with_expected_sync(cx, Some(expected), sync_directory)
    }

    fn flush_with_sync<S>(&self, cx: &Cx, final_sync: S) -> SearchResult<GenerationPublication>
    where
        S: FnOnce(&Path) -> io::Result<()>,
    {
        self.flush_with_expected_sync(cx, None, final_sync)
    }

    fn flush_with_expected_sync<S>(
        &self,
        cx: &Cx,
        expected_generation: Option<&PublishedGeneration>,
        final_sync: S,
    ) -> SearchResult<GenerationPublication>
    where
        S: FnOnce(&Path) -> io::Result<()>,
    {
        checkpoint(cx)?;
        let lease = PublicationLease::acquire(&self.root)?;
        lease.fence("complete-generation flush entry")?;
        let expected = selection_evidence(&self.root)?;
        let generation = self.active(cx)?.ok_or_else(|| {
            invalid(
                &self.root,
                "no complete generation is selected; nothing can be flushed",
            )
        })?;
        if expected_generation.is_some_and(|expected| expected != &generation) {
            return Err(invalid(
                &self.root,
                "selected generation differs from the trusted durability target; no confirmation was performed",
            ));
        }
        sync_tree(cx, generation.path(), 0)?;
        sync_directory(&self.root.join(GENERATIONS))?;
        open_regular(&self.root.join(COMPLETE_GENERATION_POINTER))?.sync_all()?;
        // Refuse detected mutation during sync; a checksum is not permission
        // to accept new bytes as the same generation.
        self.open_retained(cx, generation.id(), generation.manifest_sha256())?;
        ensure_selection(&self.root, expected.as_deref())?;
        checkpoint(cx)?;
        lease.fence("complete-generation flush completion")?;
        // Once the barrier completes, a concurrent cancellation cannot revoke it.
        let outcome = synced_outcome(generation, final_sync(&self.root));
        drop(lease);
        Ok(outcome)
    }
}

impl PreparedGenerationRestore {
    /// Exact retained target selected by the caller, not a newly discovered candidate.
    #[must_use]
    pub const fn target(&self) -> &PublishedGeneration {
        &self.target
    }

    /// Re-admit the target and atomically select it without rewriting its artifacts.
    ///
    /// `validate` must perform the consumer's read-only engine completeness and
    /// producer-identity checks, just like [`super::GenerationBuild::publish`].
    /// A failure or cancellation before rename preserves the incumbent. The old
    /// and target directories, including corrupt data, are never deleted.
    ///
    /// # Errors
    /// Returns an error for a stale preparation, changed target, failed consumer
    /// admission, contention, cancellation or pre-rename I/O. After rename, a
    /// failed directory sync returns a visible-but-uncertain publication.
    pub fn restore<F>(self, cx: &Cx, validate: F) -> SearchResult<GenerationPublication>
    where
        F: FnOnce(&Cx, &Path) -> SearchResult<()>,
    {
        self.restore_with_sync(cx, validate, sync_directory)
    }

    fn restore_with_sync<F, S>(
        self,
        cx: &Cx,
        validate: F,
        final_sync: S,
    ) -> SearchResult<GenerationPublication>
    where
        F: FnOnce(&Cx, &Path) -> SearchResult<()>,
        S: FnOnce(&Path) -> io::Result<()>,
    {
        checkpoint(cx)?;
        let lease = PublicationLease::acquire(&self.store.root)?;
        lease.fence("complete-generation restore entry")?;
        ensure_selection(&self.store.root, self.expected_selection.as_deref())?;
        let target =
            self.store
                .open_retained(cx, self.target.id(), self.target.manifest_sha256())?;
        validate(cx, target.path())?;
        checkpoint(cx)?;
        sync_tree(cx, target.path(), 0)?;
        sync_directory(&self.store.root.join(GENERATIONS))?;
        self.store
            .open_retained(cx, target.id(), target.manifest_sha256())?;
        let pointer = format!(
            "{POINTER_MAGIC}\n{}\n{}\n",
            target.id(),
            target.manifest_sha256()
        );
        let temporary = stage_restore_pointer(cx, &self.store.root, pointer.as_bytes())?;
        checkpoint(cx)?;
        lease.fence("complete-generation restore publication")?;
        ensure_selection(&self.store.root, self.expected_selection.as_deref())?;
        fs::rename(
            &temporary,
            self.store.root.join(COMPLETE_GENERATION_POINTER),
        )?;
        // The selected generation is now visible. Never turn a post-rename
        // cancellation or sync error into a claimed abort, and never roll back.
        let outcome = synced_outcome(target, final_sync(&self.store.root));
        drop(lease);
        Ok(outcome)
    }
}

fn synced_outcome(
    generation: PublishedGeneration,
    result: io::Result<()>,
) -> GenerationPublication {
    match result {
        Ok(()) => GenerationPublication::Durable(generation),
        Err(source) => GenerationPublication::VisibleButDurabilityUncertain { generation, source },
    }
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn selection_evidence(root: &Path) -> SearchResult<Option<Vec<u8>>> {
    let path = root.join(COMPLETE_GENERATION_POINTER);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    // Deliberately do not decode: explicit recovery must be able to replace a
    // malformed selection, while still binding its exact bounded bytes.
    read_bounded_regular(&path, MAX_POINTER_BYTES).map(Some)
}

fn ensure_selection(root: &Path, expected: Option<&[u8]>) -> SearchResult<()> {
    if selection_evidence(root)?.as_deref() != expected {
        return Err(invalid(
            root,
            "selection changed since recovery or flush began",
        ));
    }
    Ok(())
}

fn stage_restore_pointer(cx: &Cx, root: &Path, bytes: &[u8]) -> SearchResult<std::path::PathBuf> {
    for _ in 0..64 {
        checkpoint(cx)?;
        let serial = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".FSFS-RESTORE-{:08x}-{serial:016x}",
            std::process::id()
        ));
        match write_new_synced(&path, bytes) {
            Ok(()) => return Ok(path),
            Err(SearchError::Io(error)) if error.kind() == ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(invalid(
        root,
        "cannot allocate an exclusive restore descriptor",
    ))
}

#[cfg(all(test, unix))]
mod tests;

#[cfg(all(test, unix))]
mod receipt_durability_tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::os::unix::fs::MetadataExt;

    // Container-level durability fixtures, not semantic readiness evidence.
    fn publish(store: &CompleteGenerationStore, cx: &Cx, body: &[u8]) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        fs::write(build.path().join("payload"), body).unwrap();
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("fixture publication must be durable"); // ubs:ignore — test assertion.
        };
        generation
    }

    #[test]
    fn confirmation_binds_exact_selected_identity_before_sync() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let old = publish(&store, &cx, b"old");
            let new = publish(&store, &cx, b"new");
            let pointer = directory.path().join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            let inode = fs::metadata(&pointer).unwrap().ino();
            let mut sync_called = false;
            assert!(
                store
                    .flush_with_expected_sync(&cx, Some(&old), |_| {
                        sync_called = true;
                        Ok(())
                    })
                    .is_err()
            );
            assert!(!sync_called);
            assert!(store.confirm_retained_durability(&cx, &old).is_err());
            let mut foreign = new.clone();
            foreign.path = directory.path().join("foreign-store").join(new.id());
            assert!(store.confirm_retained_durability(&cx, &foreign).is_err());
            assert!(matches!(
                store.confirm_retained_durability(&cx, &new).unwrap(),
                GenerationPublication::Durable(value) if value == new
            ));
            assert_eq!(fs::read(&pointer).unwrap(), before);
            assert_eq!(fs::metadata(&pointer).unwrap().ino(), inode);
            assert_eq!(fs::read(old.path().join("payload")).unwrap(), b"old");
            assert_eq!(fs::read(new.path().join("payload")).unwrap(), b"new");
            assert_eq!(
                fs::read_dir(directory.path().join(GENERATIONS))
                    .unwrap()
                    .count(),
                2
            );
        });
    }

    #[test]
    fn uncertain_restore_is_confirmed_without_republishing_after_reopen() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let old = publish(&store, &cx, b"old");
            let _new = publish(&store, &cx, b"new");
            let plan = store.prepare_restore(&cx, &old).unwrap();
            assert!(matches!(
                plan.restore_with_sync(
                    &cx,
                    |_, _| Ok(()),
                    |_| Err(io::Error::other("injected sync failure"))
                )
                .unwrap(),
                GenerationPublication::VisibleButDurabilityUncertain { .. }
            ));
            let pointer = directory.path().join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            let inode = fs::metadata(&pointer).unwrap().ino();
            let reopened = CompleteGenerationStore::open(&cx, directory.path()).unwrap();
            let expected = reopened
                .open_retained(&cx, old.id(), old.manifest_sha256())
                .unwrap();
            let uncertain = reopened
                .flush_with_expected_sync(&cx, Some(&expected), |_| {
                    Err(io::Error::other("still unavailable"))
                })
                .unwrap();
            assert!(matches!(uncertain,
                GenerationPublication::VisibleButDurabilityUncertain { generation, .. } if generation == old));
            assert!(matches!(
                reopened.confirm_retained_durability(&cx, &expected).unwrap(),
                GenerationPublication::Durable(generation) if generation == old
            ));
            assert_eq!(fs::read(&pointer).unwrap(), before);
            assert_eq!(fs::metadata(&pointer).unwrap().ino(), inode);
            assert_eq!(
                fs::read_dir(directory.path().join(GENERATIONS))
                    .unwrap()
                    .count(),
                2
            );
        });
    }

    #[test]
    fn confirmation_cannot_bypass_publisher_ownership_or_cancellation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let target = publish(&store, &cx, b"target");
            let before = fs::read(directory.path().join(COMPLETE_GENERATION_POINTER)).unwrap();
            let build = store.begin(&cx).unwrap();
            assert!(store.confirm_retained_durability(&cx, &target).is_err());
            drop(build);
            cx.set_cancel_requested(true);
            assert!(matches!(
                store.confirm_retained_durability(&cx, &target),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(
                fs::read(directory.path().join(COMPLETE_GENERATION_POINTER)).unwrap(),
                before
            );
            assert_eq!(store.active(&cx).unwrap(), Some(target));
        });
    }

    #[test]
    fn confirmation_never_repairs_missing_or_corrupt_selection_implicitly() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let target = publish(&store, &cx, b"target");
            let pointer = directory.path().join(COMPLETE_GENERATION_POINTER);
            let before = fs::read(&pointer).unwrap();
            fs::rename(&pointer, directory.path().join("saved-pointer")).unwrap();
            assert!(store.confirm_retained_durability(&cx, &target).is_err());
            assert!(!pointer.exists());
            fs::write(&pointer, b"broken selection").unwrap();
            assert!(store.confirm_retained_durability(&cx, &target).is_err());
            assert_eq!(fs::read(&pointer).unwrap(), b"broken selection");
            fs::write(&pointer, &before).unwrap();
            fs::write(target.path().join("payload"), b"damaged").unwrap();
            assert!(store.confirm_retained_durability(&cx, &target).is_err());
            assert_eq!(fs::read(&pointer).unwrap(), before);
            assert_eq!(fs::read(target.path().join("payload")).unwrap(), b"damaged");
        });
    }
}
