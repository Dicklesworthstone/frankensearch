//! Bind restart reuse to the actual running executable, not its pathname or
//! package version. This identifies compiled extraction/canonicalization code;
//! the ordinary indexer still admits each model producer and current source hash.
//! Unsupported or unprovable execution keeps the existing process-only policy.

use std::sync::OnceLock;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::retained_search_checkpoint;

static EXECUTABLE: OnceLock<Option<String>> = OnceLock::new();

#[cfg(all(test, target_os = "linux", not(feature = "embedded-models")))]
#[path = "restart_tests.rs"]
mod restart_tests;

pub(super) fn fingerprint() -> Option<&'static str> {
    EXECUTABLE.get().and_then(Option::as_deref)
}

/// Prepare once on the caller-owned blocking lane. No model is loaded and no
/// runtime or worker pool is created. Lack of a usable lane is a cold-reuse
/// decision, not a reason to reject an otherwise valid indexing command.
pub(super) async fn prepare(cx: &Cx) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    if EXECUTABLE.get().is_some() {
        return Ok(());
    }
    let request = cx.clone();
    let worker = cx.spawn_blocking(move |child| {
        retained_search_checkpoint(&request)?;
        retained_search_checkpoint(&child)?;
        // A rejected pool submission can run on an async wrapper. Never hash
        // an executable there: retain session-local reuse instead.
        if Cx::is_active() {
            return Ok(None);
        }
        let result = current_image(&request);
        retained_search_checkpoint(&request)?;
        retained_search_checkpoint(&child)?;
        result
    });
    let candidate = match worker {
        Ok(mut worker) => {
            let joined = worker.join(cx).await;
            retained_search_checkpoint(cx)?;
            match joined {
                Ok(Ok(value)) => value,
                Ok(Err(error @ SearchError::Cancelled { .. })) => return Err(error),
                Err(asupersync::runtime::JoinError::Cancelled(_)) => {
                    return Err(SearchError::Cancelled {
                        phase: "fsfs.complete_generation.reuse_identity".to_owned(),
                        reason: "executable identity worker cancelled".to_owned(),
                    });
                }
                // A failed proof grants no cross-process reuse. Do not persist
                // arbitrary filesystem paths or backend diagnostics in receipts.
                Ok(Err(_)) | Err(_) => None,
            }
        }
        Err(_) => {
            retained_search_checkpoint(cx)?;
            None
        }
    };
    retained_search_checkpoint(cx)?;
    let admitted = EXECUTABLE.get_or_init(|| candidate);
    tracing::debug!(
        cross_process_reuse = admitted.is_some(),
        "established retained-index reuse execution scope"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn current_image(cx: &Cx) -> SearchResult<Option<String>> {
    retained_search_checkpoint(cx)?;
    // current_exe() names a path which may already hold an upgraded binary,
    // not the object this process is executing. Do not guess on other targets.
    Ok(None)
}

#[cfg(target_os = "linux")]
fn current_image(cx: &Cx) -> SearchResult<Option<String>> {
    use std::fs::File;
    use std::os::unix::fs::MetadataExt;

    retained_search_checkpoint(cx)?;
    // This is the kernel's link to the executing inode, even after a package
    // manager replaces/unlinks its original pathname. Following this one
    // kernel link is intentional; never substitute std::env::current_exe().
    let mut file = File::open("/proc/self/exe")?;
    let before = file.metadata()?;
    if !before.is_file() || before.len() == 0 || before.len() > MAX_IMAGE_BYTES {
        return Ok(None);
    }
    if !code_belongs_to_image(cx, &before)? {
        // A host can load FSFS from a shared library. Hashing that host's
        // executable would not identify our compiled indexing implementation.
        return Ok(None);
    }
    let digest = image_digest(cx, &mut file, before.len())?;
    let after = file.metadata()?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return Ok(None);
    }
    retained_search_checkpoint(cx)?;
    Ok(Some(digest))
}

#[cfg(target_os = "linux")]
fn code_belongs_to_image(cx: &Cx, image: &std::fs::Metadata) -> SearchResult<bool> {
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;

    const MAX_MAP_BYTES: u64 = 4 * 1024 * 1024;
    let mut maps = Vec::new();
    std::fs::File::open("/proc/self/maps")?
        .take(MAX_MAP_BYTES + 1)
        .read_to_end(&mut maps)?;
    retained_search_checkpoint(cx)?;
    if maps.len() as u64 > MAX_MAP_BYTES {
        return Ok(false);
    }
    let Ok(maps) = std::str::from_utf8(&maps) else {
        return Ok(false);
    };
    let address = current_image as *const () as usize;
    Ok(mapping_matches_image(
        maps,
        address,
        (libc::major(image.dev()), libc::minor(image.dev())),
        image.ino(),
    ))
}

#[cfg(any(target_os = "linux", test))]
fn mapping_matches_image(maps: &str, address: usize, device_id: (u32, u32), image_inode: u64) -> bool {
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let (Some(range), Some(permissions), Some(_offset), Some(device), Some(inode)) =
            (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            )
        else {
            return false;
        };
        let Some((start, end)) = range.split_once('-') else {
            return false;
        };
        let (Ok(start), Ok(end)) =
            (usize::from_str_radix(start, 16), usize::from_str_radix(end, 16))
        else {
            return false;
        };
        if !(start..end).contains(&address) {
            continue;
        }
        let Some((major, minor)) = device.split_once(':') else {
            return false;
        };
        return permissions.len() == 4
            && permissions.as_bytes().get(2) == Some(&b'x')
            && inode.parse::<u64>().ok() == Some(image_inode)
            && u32::from_str_radix(major, 16).ok() == Some(device_id.0)
            && u32::from_str_radix(minor, 16).ok() == Some(device_id.1);
    }
    false
}

#[cfg(any(target_os = "linux", test))]
const MAX_IMAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[cfg(any(target_os = "linux", test))]
fn image_digest<R: std::io::Read>(cx: &Cx, reader: &mut R, expected: u64) -> SearchResult<String> {
    use sha2::{Digest, Sha256};

    retained_search_checkpoint(cx)?;
    if expected == 0 || expected > MAX_IMAGE_BYTES {
        return Err(super::reuse_error(
            "executable identity exceeds its byte budget",
        ));
    }
    let mut hash = Sha256::new();
    let mut bytes = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut observed = 0_u64;
    loop {
        retained_search_checkpoint(cx)?;
        // Read at most the declared length plus one byte, including a stream
        // that never terminates. EOF itself does not override cancellation.
        let remaining = expected.saturating_sub(observed).saturating_add(1);
        let width = usize::try_from(remaining.min(bytes.len() as u64))
            .map_err(|_| super::reuse_error("executable read budget does not fit usize"))?;
        let outcome = reader.read(&mut bytes[..width]);
        retained_search_checkpoint(cx)?;
        let count = match outcome {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            break;
        }
        observed += count as u64;
        if observed > expected {
            return Err(super::reuse_error(
                "executable grew while establishing reuse identity",
            ));
        }
        hash.update(&bytes[..count]);
    }
    if observed != expected {
        return Err(super::reuse_error(
            "executable shortened while establishing reuse identity",
        ));
    }
    retained_search_checkpoint(cx)?;
    Ok(crate::runtime::sha256_digest_hex(hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::io::{Cursor, Read};

    #[test]
    fn executable_digest_binds_all_bytes_not_only_size_or_labels() {
        run_test_with_cx(|cx| async move {
            let bytes = vec![17_u8; 140_000];
            let expected = image_digest(&cx, &mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
            assert_eq!(expected.len(), 64);
            assert_eq!(
                expected,
                image_digest(&cx, &mut Cursor::new(&bytes), bytes.len() as u64).unwrap()
            );
            for position in [0, 65_535, 65_536, 139_999] {
                let mut changed = bytes.clone();
                changed[position] ^= 1;
                assert_ne!(
                    expected,
                    image_digest(&cx, &mut Cursor::new(&changed), changed.len() as u64).unwrap()
                );
            }
            assert_eq!(
                image_digest(&cx, &mut Cursor::new(b"abc"), 3).unwrap(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
        });
    }

    #[test]
    fn executable_digest_refuses_short_long_empty_and_unbounded_inputs() {
        run_test_with_cx(|cx| async move {
            for expected in [0, 2, 4, MAX_IMAGE_BYTES + 1] {
                assert!(image_digest(&cx, &mut Cursor::new(b"abc"), expected).is_err());
            }
            let mut endless = std::io::repeat(7).take(1_000_000);
            assert!(image_digest(&cx, &mut endless, 3).is_err());
            assert_eq!(endless.limit(), 999_996);
        });
    }

    #[test]
    fn executable_digest_observes_cancellation_before_io_and_on_eof_or_error() {
        struct CancelReader<'a> {
            cx: &'a Cx,
            fail: bool,
        }
        impl Read for CancelReader<'_> {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                self.cx.set_cancel_requested(true);
                if self.fail {
                    Err(std::io::ErrorKind::PermissionDenied.into())
                } else {
                    Ok(0)
                }
            }
        }
        run_test_with_cx(|cx| async move {
            let mut untouched = Cursor::new(b"abc");
            cx.set_cancel_requested(true);
            assert!(matches!(
                image_digest(&cx, &mut untouched, 3),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(untouched.position(), 0);
            cx.set_cancel_requested(false);
            for fail in [false, true] {
                assert!(matches!(
                    image_digest(&cx, &mut CancelReader { cx: &cx, fail }, 3),
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
            }
        });
    }

    #[test]
    fn executable_digest_retries_interrupted_io_but_preserves_other_errors() {
        struct InterruptedOnce {
            first: bool,
            bytes: Cursor<&'static [u8]>,
        }
        impl Read for InterruptedOnce {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if std::mem::take(&mut self.first) {
                    Err(std::io::ErrorKind::Interrupted.into())
                } else {
                    self.bytes.read(buffer)
                }
            }
        }
        struct Denied;
        impl Read for Denied {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::PermissionDenied.into())
            }
        }
        run_test_with_cx(|cx| async move {
            let mut interrupted = InterruptedOnce {
                first: true,
                bytes: Cursor::new(b"abc"),
            };
            assert_eq!(
                image_digest(&cx, &mut interrupted, 3).unwrap(),
                image_digest(&cx, &mut Cursor::new(b"abc"), 3).unwrap()
            );
            assert!(matches!(
                image_digest(&cx, &mut Denied, 3),
                Err(SearchError::Io(error)) if error.kind() == std::io::ErrorKind::PermissionDenied
            ));
        });
    }

    #[test]
    fn shared_library_code_cannot_be_attributed_to_the_host_executable() {
        let maps = "1000-2000 r--p 0000 08:02 7 /binary\n\
                    2000-3000 r-xp 1000 08:02 7 /binary\n\
                    4000-5000 r-xp 0000 08:03 7 /plugin with spaces.so\n";
        assert!(mapping_matches_image(maps, 0x2000, (8, 2), 7));
        assert!(mapping_matches_image(maps, 0x2fff, (8, 2), 7));
        for address in [0x1000, 0x3000, 0x4000, 0x6000] {
            assert!(!mapping_matches_image(maps, address, (8, 2), 7));
        }
        assert!(!mapping_matches_image(maps, 0x2000, (8, 2), 8));
        assert!(!mapping_matches_image(maps, 0x2000, (8, 3), 7));
        assert!(!mapping_matches_image("invalid", 0x2000, (8, 2), 7));
        assert!(!mapping_matches_image("2000-3000 rw-p 0000 08:02 7", 0x2000, (8, 2), 7));
    }
}
