//! Shared immutable file-mapping facade.
//!
//! File-backed memory maps are sound only while the mapped file's contents and
//! length remain unchanged. Frankensearch satisfies that requirement for
//! published index artifacts by writing a new temporary file, syncing it, and
//! atomically publishing it under a new generation name. Published files are
//! never modified in place; replacement and garbage collection operate on
//! directory entries instead.

use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

/// An immutable, file-backed byte buffer.
///
/// The mapping owns its operating-system handle and remains valid after the
/// [`File`] used by [`Self::open_published`] is dropped. It exposes no mutable access.
/// Frankensearch callers must map only immutable published artifacts: changing
/// the contents or length of the underlying file while this value is alive
/// violates `memmap2`'s file-mapping safety contract.
#[derive(Debug)]
pub struct ReadOnlyMappedFile {
    mmap: Mmap,
}

impl ReadOnlyMappedFile {
    /// Open and map an immutable published artifact read-only.
    ///
    /// The caller must uphold the repository's publication lifecycle: the file
    /// was completely written and synced before an atomic rename, and no code
    /// may mutate or truncate that published inode while readers can retain it.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or the operating
    /// system cannot create a read-only mapping for it.
    pub fn open_published(path: &Path) -> io::Result<Self> {
        Self::open_published_with_file(path, |mapped, _file| Ok(mapped))
    }

    /// Inspect a mapping together with the exact opened file backing it.
    ///
    /// The file cursor starts at zero. The handle is dropped on return, even
    /// when the mapping is retained by the callback's result. Reads through
    /// this handle do not fault the corresponding mapped pages into the
    /// process. The immutable publication contract of [`Self::open_published`]
    /// still applies; this does not protect against in-place writers.
    ///
    /// # Errors
    ///
    /// Returns an opening/mapping error or the callback's validation error.
    pub fn open_published_with_file<T, E: From<io::Error>>(
        path: &Path,
        inspect: impl FnOnce(Self, &mut File) -> Result<T, E>,
    ) -> Result<T, E> {
        let path_metadata = path.symlink_metadata()?;
        if !path_metadata.file_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "published artifact must be a regular file, not a symlink or special file",
            )
            .into());
        }
        let mut file = open_same_regular_file(path, &path_metadata)?;
        let mmap = map_immutable(&file)?;
        inspect(Self { mmap }, &mut file)
    }

    /// Borrow all mapped bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Return the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Return whether the mapping contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

// The path check alone cannot constrain the subsequent open: publication or
// another directory writer can replace the entry between those operations.
// In particular, a blocking open can wait on a substituted FIFO before fstat
// gets a chance to reject it. A substituted symlink can even resolve back to
// the original inode and pass an identity-only check.
fn open_same_regular_file(path: &Path, expected: &std::fs::Metadata) -> io::Result<File> {
    let mut options = File::options();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    ensure_same_file(expected, &file.metadata()?)?;
    Ok(file)
}

fn ensure_same_file(
    path_metadata: &std::fs::Metadata,
    opened: &std::fs::Metadata,
) -> io::Result<()> {
    // Check the actual descriptor on every platform, before mapping or
    // invoking user validation. Path metadata is only pre-open evidence.
    if !opened.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "published artifact must be a regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if path_metadata.dev() != opened.dev() || path_metadata.ino() != opened.ino() {
            return Err(io::Error::other(
                "published artifact changed while it was being opened",
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = path_metadata;
    Ok(())
}

impl AsRef<[u8]> for ReadOnlyMappedFile {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[allow(unsafe_code)]
fn map_immutable(file: &File) -> io::Result<Mmap> {
    // SAFETY: This is the facade's only unsafe operation. Frankensearch maps
    // only published generation files, whose lifecycle contract forbids
    // in-place writes and truncation for as long as readers can retain them.
    // The returned Mmap owns the mapping and exposes only shared byte slices.
    unsafe { Mmap::map(file) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::Read;

    #[cfg(unix)]
    #[test]
    fn inspected_handle_and_mapping_survive_path_replacement() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        let replacement = directory.path().join("replacement.bin");
        std::fs::write(&path, b"original immutable bytes")?;
        std::fs::write(&replacement, b"unrelated replacement")?;
        let mapped = ReadOnlyMappedFile::open_published_with_file(&path, |mapped, file| {
            std::fs::rename(&replacement, &path)?;
            let mut streamed = Vec::new();
            file.read_to_end(&mut streamed)?;
            assert_eq!(streamed, b"original immutable bytes");
            assert_eq!(mapped.as_bytes(), streamed);
            Ok::<_, io::Error>(mapped)
        })?;
        assert_eq!(std::fs::read(&path)?, b"unrelated replacement");
        assert_eq!(mapped.as_bytes(), b"original immutable bytes");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn inspected_open_releases_descriptors_while_mappings_remain_live() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        std::fs::write(&path, b"retained mapping")?;
        let mut mappings = Vec::new();
        for _ in 0..256 {
            mappings.push(ReadOnlyMappedFile::open_published_with_file(
                &path,
                |mapped, file| {
                    let mut byte = [0_u8; 1];
                    file.read_exact(&mut byte)?;
                    assert_eq!(byte, [b'r']);
                    Ok::<_, io::Error>(mapped)
                },
            )?);
        }
        for entry in std::fs::read_dir("/proc/self/fd")? {
            if let Ok(target) = std::fs::read_link(entry?.path()) {
                assert_ne!(
                    target, path,
                    "construction must release its file descriptor"
                );
            }
        }
        assert!(
            mappings
                .iter()
                .all(|mapped| mapped.as_bytes() == b"retained mapping")
        );
        Ok(())
    }

    #[test]
    fn exposes_exact_file_bytes_through_all_views() {
        let directory = tempfile::tempdir().expect("create temp directory");
        let path = directory.path().join("seg-0000000000000001.fslx");
        let expected = b"published immutable artifact\0with raw bytes";
        std::fs::write(&path, expected).expect("write fixture before mapping");

        let mapped = ReadOnlyMappedFile::open_published(&path).expect("map fixture");

        assert_eq!(mapped.as_bytes(), expected);
        assert_eq!(mapped.as_ref(), expected);
        assert_eq!(mapped.len(), expected.len());
        assert!(!mapped.is_empty());
    }

    #[test]
    fn reports_missing_file_as_io_error() {
        let directory = tempfile::tempdir().expect("create temp directory");
        let path = directory.path().join("missing.bin");

        let error = ReadOnlyMappedFile::open_published(&path).expect_err("missing file must fail");

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symbolic_links() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("create temp directory");
        let target = directory.path().join("target.fslx");
        let path = directory.path().join("seg-0000000000000001.fslx");
        std::fs::write(&target, b"immutable target").expect("write target");
        symlink(&target, &path).expect("create fixture symlink");

        let error = ReadOnlyMappedFile::open_published(&path).expect_err("symlink must fail");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_same_inode_symlink_inserted_after_metadata() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        let retained = directory.path().join("retained.bin");
        std::fs::write(&path, b"immutable original")?;
        let expected = path.symlink_metadata()?;
        std::fs::rename(&path, &retained)?;
        std::os::unix::fs::symlink(&retained, &path)?;

        // File::open would follow this link to the SAME inode: fstat identity
        // agreement alone does not enforce the no-symlink publication rule.
        assert!(open_same_regular_file(&path, &expected).is_err());
        assert_eq!(std::fs::read(&retained)?, b"immutable original");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_different_regular_file_inserted_after_metadata() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        let retained = directory.path().join("retained.bin");
        std::fs::write(&path, b"immutable original")?;
        let expected = path.symlink_metadata()?;
        std::fs::rename(&path, &retained)?;
        std::fs::write(&path, b"different original")?;

        let error = open_same_regular_file(&path, &expected).expect_err("identity changed");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn preserves_regular_hard_link_identity_after_metadata() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        let retained = directory.path().join("retained.bin");
        std::fs::write(&path, b"immutable original")?;
        let expected = path.symlink_metadata()?;
        std::fs::rename(&path, &retained)?;
        std::fs::hard_link(&retained, &path)?;

        let mut file = open_same_regular_file(&path, &expected)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        assert_eq!(bytes, b"immutable original");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn inspected_open_sets_nonblock_before_validation() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        std::fs::write(&path, b"immutable original")?;
        let mapped = ReadOnlyMappedFile::open_published_with_file(&path, |mapped, file| {
            assert!(rustix::fs::fcntl_getfl(file)?.contains(rustix::fs::OFlags::NONBLOCK));
            Ok::<_, io::Error>(mapped)
        })?;
        assert_eq!(mapped.as_bytes(), b"immutable original");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_fifo_inserted_after_metadata() -> io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;

        let directory = tempfile::tempdir()?;
        let path = directory.path().join("published.bin");
        let retained = directory.path().join("retained.bin");
        std::fs::write(&path, b"immutable original")?;
        let expected = path.symlink_metadata()?;
        std::fs::rename(&path, &retained)?;
        rustix::fs::mkfifoat(
            rustix::fs::CWD,
            &path,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        )?;

        // Keep a Linux read/write peer so a regression cannot hang this test.
        // The separate descriptor-flag test proves NONBLOCK is really set.
        let _peer = File::options()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)?;
        let error = open_same_regular_file(&path, &expected).expect_err("FIFO must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    #[test]
    fn is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ReadOnlyMappedFile>();
    }
}
