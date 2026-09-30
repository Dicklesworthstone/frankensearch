//! Lifecycle management with complete-generation-aware write admission.
//!
//! Keep the native kernel-lock implementation in its existing source module.
//! Every public publication lease additionally refuses a sealed bundle or any
//! of its descendants before touching directories or lock files and at each
//! publication fence. Filesystem aliases cannot hide a sealed ancestor.

#[path = "lifecycle.rs"]
mod kernel;

// Preserve the lifecycle surface while the explicit local PublicationLease
// below supplies the additional immutable-generation admission boundary.
#[allow(clippy::wildcard_imports)]
pub use kernel::*;

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use frankensearch_core::SearchResult;

/// Exclusive publication authority for a mutable fsfs index root.
///
/// Kernel exclusion, owner records, creator-process checks and lock-identity
/// fencing are retained by the native lease. Sealed complete generations may
/// only be read; callers must build a successor rather than mutate one in place.
#[derive(Debug)]
pub struct PublicationLease {
    inner: kernel::PublicationLease,
    root: PathBuf,
}

impl PublicationLease {
    /// Acquire native exclusion after refusing a sealed generation subtree.
    ///
    /// # Errors
    /// Returns the native contention/I/O error or a sealed-generation refusal.
    pub fn acquire(index_root: &Path) -> SearchResult<Self> {
        let root = std::path::absolute(index_root)?;
        reject_sealed_ancestry(&root)?;
        let inner = kernel::PublicationLease::acquire(&root)?;
        reject_sealed_ancestry(&root)?;
        Ok(Self { inner, root })
    }

    /// Revalidate both mutable-generation admission and native lock authority.
    ///
    /// # Errors
    /// Returns an error if the root or an ancestor was sealed, or native
    /// authority was lost.
    pub fn fence(&self, boundary: &'static str) -> SearchResult<()> {
        reject_sealed_ancestry(&self.root)?;
        self.inner.fence(boundary)
    }

    /// Path of the held native lock file.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        self.inner.lock_path()
    }
}

// The native lease creates missing directories and rewrites its owner record.
// Inspect the nearest existing ancestor without creating anything first. Its
// canonical ancestry catches both a nested index root and a symlink pointing
// into a sealed generation; lexical parents alone cannot catch the latter.
// Keep the original absolute root on the lease so the native inode fence still
// detects a pathname retargeted after acquisition.
pub(crate) fn reject_sealed_ancestry(root: &Path) -> SearchResult<()> {
    let absolute = std::path::absolute(root)?;
    for ancestor in absolute.ancestors() {
        match std::fs::canonicalize(ancestor) {
            Ok(existing) => {
                for directory in existing.ancestors() {
                    crate::generation_store::reject_published_write(directory)?;
                }
                return Ok(());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Err(std::io::Error::new(ErrorKind::NotFound, "publication root has no existing ancestor").into())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::generation_store::COMPLETE_GENERATION_MANIFEST;

    #[test]
    fn sealed_admission_does_not_create_or_modify_the_lock_file() {
        let root = tempfile::tempdir().expect("root");
        std::fs::write(root.path().join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        assert!(PublicationLease::acquire(root.path()).is_err());
        let lock = root.path().join(PUBLICATION_LOCK_FILE_NAME);
        assert!(!lock.exists());
        std::fs::write(&lock, b"preserve this record").expect("existing lock");
        assert!(PublicationLease::acquire(root.path()).is_err());
        assert_eq!(
            std::fs::read(lock).expect("retained record"),
            b"preserve this record"
        );
    }

    #[test]
    fn held_lease_fences_out_a_root_that_has_been_sealed() {
        let root = tempfile::tempdir().expect("root");
        let lease = PublicationLease::acquire(root.path()).expect("mutable admission");
        lease.fence("before seal").expect("mutable fence");
        std::fs::write(root.path().join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        assert!(lease.fence("after seal").is_err());
    }

    #[test]
    fn sealed_descendant_admission_preserves_existing_artifacts_and_lock_record() {
        let root = tempfile::tempdir().expect("root");
        let nested = root.path().join("lexical/shard");
        std::fs::create_dir_all(&nested).expect("nested index");
        let artifact = nested.join("segment");
        std::fs::write(&artifact, b"retained segment").expect("artifact");
        let lock = nested.join(PUBLICATION_LOCK_FILE_NAME);
        std::fs::write(&lock, b"retained owner record").expect("lock record");
        std::fs::write(root.path().join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");

        let error = PublicationLease::acquire(&nested).expect_err("sealed ancestor");
        assert!(error.to_string().contains("sealed complete generation"));
        assert_eq!(std::fs::read(&lock).expect("record"), b"retained owner record");
        assert_eq!(
            std::fs::read(&artifact).expect("artifact"),
            b"retained segment"
        );
    }

    #[test]
    fn sealed_descendant_admission_does_not_create_missing_directories() {
        let root = tempfile::tempdir().expect("root");
        std::fs::write(root.path().join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        let missing = root.path().join("new-index");
        assert!(PublicationLease::acquire(&missing.join("nested")).is_err());
        assert!(!missing.exists());
        assert_eq!(std::fs::read_dir(root.path()).expect("inventory").count(), 1);
    }

    #[test]
    fn sealed_descendant_admission_resolves_symlink_aliases_before_any_write() {
        let root = tempfile::tempdir().expect("root");
        let sealed = root.path().join("sealed");
        let nested = sealed.join("lexical");
        std::fs::create_dir_all(&nested).expect("nested index");
        std::fs::write(sealed.join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        let alias = root.path().join("mutable-looking-alias");
        std::os::unix::fs::symlink(&nested, &alias).expect("alias");

        assert!(PublicationLease::acquire(&alias).is_err());
        assert!(PublicationLease::acquire(&alias.join("new/nested")).is_err());
        assert_eq!(std::fs::read_dir(&nested).expect("inventory").count(), 0);
    }

    #[test]
    fn sealed_ancestor_marker_symlink_is_not_followed_or_ignored() {
        let root = tempfile::tempdir().expect("root");
        let nested = root.path().join("lexical");
        std::fs::create_dir(&nested).expect("nested index");
        std::os::unix::fs::symlink(
            "missing-manifest",
            root.path().join(COMPLETE_GENERATION_MANIFEST),
        )
        .expect("dangling seal marker");
        assert!(PublicationLease::acquire(&nested).is_err());
        assert!(!nested.join(PUBLICATION_LOCK_FILE_NAME).exists());
    }

    #[test]
    fn held_lease_fences_out_a_newly_sealed_ancestor() {
        let root = tempfile::tempdir().expect("root");
        let nested = root.path().join("lexical");
        let lease = PublicationLease::acquire(&nested).expect("mutable admission");
        lease.fence("before ancestor seal").expect("mutable fence");
        std::fs::write(root.path().join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        assert!(lease.fence("after ancestor seal").is_err());
    }

    #[test]
    fn mutable_sibling_and_alias_keep_native_exclusion_and_directory_creation() {
        let root = tempfile::tempdir().expect("root");
        let sealed = root.path().join("generation");
        std::fs::create_dir(&sealed).expect("sealed directory");
        std::fs::write(sealed.join(COMPLETE_GENERATION_MANIFEST), b"sealed")
            .expect("seal marker");
        let mutable = root.path().join("generation-other");
        let nested = mutable.join("new/nested");
        let lease = PublicationLease::acquire(&nested).expect("mutable sibling");
        lease.fence("mutable sibling").expect("mutable fence");
        let alias = root.path().join("mutable-alias");
        std::os::unix::fs::symlink(&mutable, &alias).expect("mutable alias");
        let aliased = alias.join("new/nested");
        PublicationLease::acquire(&aliased).expect_err("same native lock remains exclusive");
        drop(lease);
        PublicationLease::acquire(&aliased)
            .expect("mutable alias admitted after release")
            .fence("alias publication")
            .expect("alias fence");
    }
}
