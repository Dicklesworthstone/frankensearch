//! Filesystem object identity comparison without the `same-file` crate.
//!
//! frankensearch used `same_file::is_same_file` for one question, asked
//! six times in `two_tier.rs`: do two paths resolve to the same physical
//! filesystem object (through symlinks and hardlinks)? That is answered
//! directly by the platform identity tuple:
//!
//! * **Unix** — `(st_dev, st_ino)` from a symlink-following `stat`. This
//!   is deliberately `fs::metadata`, not open-then-`fstat`: it needs only
//!   search permission on the parent directories, so unreadable
//!   (mode-`000`) artifacts still compare correctly where an open-based
//!   probe would fail with `EACCES`, and no descriptor is spent per
//!   comparison.
//! * **Windows** — `(volume_serial_number, file_index)` from
//!   [`std::os::windows::fs::MetadataExt`] via an opened handle (the
//!   by-handle fields are only populated for handle-derived metadata).
//!   This uses the nightly `windows_by_handle` std feature — the
//!   workspace already mandates nightly — instead of hand-rolled
//!   `GetFileInformationByHandle` FFI.
//!
//! Both paths follow symlinks, matching `same_file` semantics at every
//! existing call site.

use std::io;
use std::path::Path;

/// Do `left` and `right` refer to the same physical filesystem object?
///
/// Follows symlinks. Two hardlinks to one inode compare equal; two
/// distinct files with identical contents do not.
///
/// # Errors
///
/// Propagates the underlying [`io::Error`] when either path cannot be
/// inspected (including `NotFound` — callers decide whether a missing
/// path means "not the same file" or is fatal, exactly as they did with
/// `same-file`).
pub fn is_same_file(left: &Path, right: &Path) -> io::Result<bool> {
    Ok(identity_of(left)? == identity_of(right)?)
}

#[cfg(unix)]
fn identity_of(path: &Path) -> io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(windows)]
fn identity_of(path: &Path) -> io::Result<(u32, u64)> {
    use std::os::windows::fs::MetadataExt;
    // By-handle identity fields are only populated on metadata obtained
    // from an open handle, so open for metadata access and stat that.
    let handle = std::fs::File::open(path)?;
    let metadata = handle.metadata()?;
    match (metadata.volume_serial_number(), metadata.file_index()) {
        (Some(volume), Some(index)) => Ok((volume, index)),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "platform did not report a by-handle file identity",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::is_same_file;
    use std::fs::File;
    use std::io;

    #[test]
    fn same_path_and_distinct_files_compare_correctly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        File::create(&first).expect("create first");
        File::create(&second).expect("create second");

        assert!(is_same_file(&first, &first).expect("self compare"));
        assert!(!is_same_file(&first, &second).expect("distinct compare"));
    }

    #[test]
    fn missing_paths_surface_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let present = dir.path().join("present");
        File::create(&present).expect("create present");
        let missing = dir.path().join("missing");

        let error = is_same_file(&present, &missing).expect_err("missing must error");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[cfg(unix)]
    #[test]
    fn hardlinks_and_symlinks_alias_their_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("target");
        File::create(&target).expect("create target");

        let hardlink = dir.path().join("hardlink");
        std::fs::hard_link(&target, &hardlink).expect("hardlink");
        assert!(
            is_same_file(&target, &hardlink).expect("hardlink compare"),
            "hardlinks share an inode and must alias"
        );

        let symlink = dir.path().join("symlink");
        std::os::unix::fs::symlink(&target, &symlink).expect("symlink");
        assert!(
            is_same_file(&target, &symlink).expect("symlink compare"),
            "symlinks are followed and must alias their target"
        );
        assert!(
            is_same_file(&hardlink, &symlink).expect("cross compare"),
            "all three names resolve to one object"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_files_still_compare_by_identity() {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let sealed = dir.path().join("sealed");
        File::create(&sealed).expect("create sealed");
        std::fs::set_permissions(&sealed, Permissions::from_mode(0o000)).expect("seal permissions");

        // Open-based probes (same-file's approach) fail here with EACCES
        // for unprivileged users; the stat-based identity must not. Root
        // can open anything, so the assertion holds either way.
        assert!(is_same_file(&sealed, &sealed).expect("stat-based compare"));

        std::fs::set_permissions(&sealed, Permissions::from_mode(0o600))
            .expect("restore permissions so the tempdir can clean up");
    }
}
