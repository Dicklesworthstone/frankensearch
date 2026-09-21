//! Lifecycle management with complete-generation-aware write admission.
//!
//! Keep the native kernel-lock implementation in its existing source module.
//! Every public publication lease additionally refuses a sealed bundle before
//! touching its lock file and at each publication fence.

#[path = "lifecycle.rs"]
mod kernel;

// Preserve the lifecycle surface while the explicit local PublicationLease
// below supplies the additional immutable-generation admission boundary.
#[allow(clippy::wildcard_imports)]
pub use kernel::*;

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
    /// Acquire native exclusion after refusing a sealed generation.
    ///
    /// # Errors
    /// Returns the native contention/I/O error or a sealed-generation refusal.
    pub fn acquire(index_root: &Path) -> SearchResult<Self> {
        let root = std::path::absolute(index_root)?;
        crate::generation_store::reject_published_write(&root)?;
        let inner = kernel::PublicationLease::acquire(&root)?;
        crate::generation_store::reject_published_write(&root)?;
        Ok(Self { inner, root })
    }

    /// Revalidate both mutable-generation admission and native lock authority.
    ///
    /// # Errors
    /// Returns an error if the root was sealed or native authority was lost.
    pub fn fence(&self, boundary: &'static str) -> SearchResult<()> {
        crate::generation_store::reject_published_write(&self.root)?;
        self.inner.fence(boundary)
    }

    /// Path of the held native lock file.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        self.inner.lock_path()
    }
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
}
