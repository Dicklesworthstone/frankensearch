//! Bounded sentinel I/O with complete-file replacement.
//!
//! A sentinel is advisory refresh metadata, not composite generation authority.
//! Writers must still hold their normal index-publication ownership. Atomic
//! replacement prevents readers from observing a truncated/interleaved JSON
//! document; it does not decide which concurrent generation is authoritative.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::SENTINEL_FILENAME;

/// Bounds both encoded publication and admission, including ignored JSON fields.
pub(super) const MAX_SENTINEL_BYTES: usize = 1024 * 1024;
const CREATE_ATTEMPTS: usize = 128;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn too_large() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "sentinel exceeds the 1 MiB byte limit",
    )
}

/// Only an actual `NotFound` error denotes an absent sentinel. Other open/read
/// failures remain errors, rather than being hidden by `Path::exists()`.
pub(super) fn read(dir: &Path) -> io::Result<Option<Vec<u8>>> {
    let file = match File::open(dir.join(SENTINEL_FILENAME)) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !file.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sentinel is not a regular file",
        ));
    }
    read_bounded(file).map(Some)
}

fn read_bounded(reader: impl Read) -> io::Result<Vec<u8>> {
    // The bound is applied to the actual read, not a racy metadata length.
    let mut bytes = Vec::new();
    reader
        .take((MAX_SENTINEL_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_SENTINEL_BYTES {
        return Err(too_large());
    }
    Ok(bytes)
}

/// Publish only a fully written, synced sibling. Failures before rename leave
/// the old sentinel intact. A directory-sync failure after rename is returned
/// as an error even though the complete replacement may already be visible.
pub(super) fn write(dir: &Path, bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_SENTINEL_BYTES {
        return Err(too_large());
    }
    replace_with(dir, |file| file.write_all(bytes))
}

struct PendingSentinel {
    path: Option<PathBuf>,
    file: Option<File>,
}

impl Drop for PendingSentinel {
    fn drop(&mut self) {
        // Windows cannot unlink the temporary file while our handle is open.
        drop(self.file.take());
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn replace_with(
    dir: &Path,
    write_contents: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    let destination = dir.join(SENTINEL_FILENAME);
    let permissions = match fs::symlink_metadata(&destination) {
        Ok(metadata) if metadata.is_file() => Some(metadata.permissions()),
        Ok(_) => None,
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let (path, file) = create_pending(dir)?;
    let mut pending = PendingSentinel {
        path: Some(path.clone()),
        file: Some(file),
    };
    // Borrow the actual owned handle; no path reopen or live-file truncation.
    let file = pending
        .file
        .as_mut()
        .ok_or_else(|| io::Error::other("sentinel staging handle is missing"))?;
    // Preserve a restricted existing mode before writing any private metadata.
    if let Some(permissions) = permissions {
        file.set_permissions(permissions)?;
    }
    write_contents(file)?;
    file.sync_all()?;
    drop(pending.file.take());
    fs::rename(&path, &destination)?;
    pending.path = None;
    // std exposes no portable directory flush on non-Unix platforms; the
    // file itself was synced there, without a directory-durability claim.
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn create_pending(dir: &Path) -> io::Result<(PathBuf, File)> {
    for _ in 0..CREATE_ATTEMPTS {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            "{SENTINEL_FILENAME}.{}.{sequence:016x}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            // A crashed process or PID reuse can leave a sibling behind. Never
            // truncate/remove an unowned collision; try another unique name.
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "cannot reserve a unique sentinel staging file",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(dir: &Path) -> Vec<std::ffi::OsString> {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        entries.sort();
        entries
    }

    #[test]
    fn atomic_replace_preserves_the_old_open_file_and_publishes_complete_bytes() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), b"old generation").unwrap();
        let mut old = File::open(dir.path().join(SENTINEL_FILENAME)).unwrap();
        let before = inventory(dir.path());
        write(dir.path(), b"complete new generation").unwrap();
        assert_eq!(
            read(dir.path()).unwrap().unwrap(),
            b"complete new generation"
        );
        let mut old_bytes = Vec::new();
        old.read_to_end(&mut old_bytes).unwrap();
        assert_eq!(old_bytes, b"old generation");
        assert_eq!(inventory(dir.path()), before);
    }

    #[test]
    fn failed_write_keeps_old_bytes_and_cleans_only_its_own_stage() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), b"old generation").unwrap();
        let foreign = dir.path().join("unrelated.tmp");
        fs::write(&foreign, b"not ours").unwrap();
        let before = inventory(dir.path());
        let error = replace_with(dir.path(), |file| {
            file.write_all(b"partial replacement")?;
            Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "injected write failure",
            ))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        assert_eq!(read(dir.path()).unwrap().unwrap(), b"old generation");
        assert_eq!(fs::read(foreign).unwrap(), b"not ours");
        assert_eq!(inventory(dir.path()), before);
    }

    #[test]
    fn rename_failure_preserves_destination_and_cleans_the_stage() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join(SENTINEL_FILENAME);
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("keep"), b"keep").unwrap();
        let before = inventory(dir.path());
        assert!(write(dir.path(), b"replacement").is_err());
        assert_eq!(fs::read(destination.join("keep")).unwrap(), b"keep");
        assert_eq!(inventory(dir.path()), before);
    }

    #[test]
    fn readers_never_observe_partial_concurrent_publications() {
        let dir = tempfile::tempdir().unwrap();
        let left = vec![b'a'; 4096];
        let right = vec![b'b'; 8192];
        write(dir.path(), &left).unwrap();
        let start = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                start.wait();
                for _ in 0..16 {
                    write(dir.path(), &left).unwrap();
                }
            });
            let second = scope.spawn(|| {
                start.wait();
                for _ in 0..16 {
                    write(dir.path(), &right).unwrap();
                }
            });
            start.wait();
            for _ in 0..64 {
                let bytes = read(dir.path()).unwrap().expect("always published");
                assert!(bytes == left || bytes == right);
            }
            first.join().unwrap();
            second.join().unwrap();
        });
        assert_eq!(
            inventory(dir.path()),
            vec![std::ffi::OsString::from(SENTINEL_FILENAME)]
        );
    }

    #[test]
    fn byte_limit_accepts_exact_boundary_and_rejects_oversize_without_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let exact = vec![b'x'; MAX_SENTINEL_BYTES];
        write(dir.path(), &exact).unwrap();
        assert_eq!(read(dir.path()).unwrap().unwrap(), exact);
        let before = inventory(dir.path());
        let too_big = vec![b'y'; MAX_SENTINEL_BYTES + 1];
        assert_eq!(
            write(dir.path(), &too_big).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(read(dir.path()).unwrap().unwrap(), exact);
        assert_eq!(inventory(dir.path()), before);
        fs::write(dir.path().join(SENTINEL_FILENAME), &too_big).unwrap();
        assert_eq!(
            read(dir.path()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn read_bound_is_enforced_on_the_stream_not_a_length_observation() {
        struct Endless {
            bytes_read: usize,
        }
        impl Read for Endless {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                buffer.fill(b'x');
                self.bytes_read += buffer.len();
                Ok(buffer.len())
            }
        }
        let mut input = Endless { bytes_read: 0 };
        assert_eq!(
            read_bounded(&mut input).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(input.bytes_read, MAX_SENTINEL_BYTES + 1);
    }

    #[test]
    fn only_missing_files_return_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read(dir.path()).unwrap().is_none());
        fs::create_dir(dir.path().join(SENTINEL_FILENAME)).unwrap();
        assert!(read(dir.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn replacement_does_not_follow_or_truncate_a_destination_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("unrelated-file");
        fs::write(&outside, b"private unrelated bytes").unwrap();
        std::os::unix::fs::symlink(&outside, dir.path().join(SENTINEL_FILENAME)).unwrap();
        write(dir.path(), b"new sentinel").unwrap();
        assert_eq!(fs::read(outside).unwrap(), b"private unrelated bytes");
        assert_eq!(read(dir.path()).unwrap().unwrap(), b"new sentinel");
        assert!(
            !fs::symlink_metadata(dir.path().join(SENTINEL_FILENAME))
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
    #[cfg(unix)]
    #[test]
    fn replacement_preserves_restrictive_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), b"old private metadata").unwrap();
        let path = dir.path().join(SENTINEL_FILENAME);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        write(dir.path(), b"new private metadata").unwrap();
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(read(dir.path()).unwrap().unwrap(), b"new private metadata");
    }
}
