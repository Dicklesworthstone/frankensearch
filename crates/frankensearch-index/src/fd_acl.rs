//! Fd-bound extended-ACL presence probe.
//!
//! In-tree replacement for the `exacl` dependency: generation-root
//! hardening needs exactly one question answered — "does the filesystem
//! object behind this already-open descriptor carry an extended ACL?" —
//! and it must ask through the retained descriptor so no pathname
//! re-resolution can be raced by a concurrent rename/replace. Everything
//! else `exacl` ships (entry decoding, ACL mutation, text round-trips)
//! is surface we do not want to depend on or audit.
//!
//! Platform semantics:
//!
//! * **macOS** — `acl_get_fd_np(fd, ACL_TYPE_EXTENDED)`. `NULL` with
//!   `ENOENT` is the documented "no extended ACL" answer and maps to
//!   [`ExtendedAclPresence::Absent`]; a non-null ACL is released with
//!   `acl_free` (error-checked) and maps to
//!   [`ExtendedAclPresence::Present`]; any other failure is surfaced as
//!   the underlying [`io::Error`].
//! * **Linux** — POSIX ACLs materialize as the `system.posix_acl_access`
//!   / `system.posix_acl_default` extended attributes only when a
//!   non-minimal (extended) ACL is attached, so a zero-length
//!   `fgetxattr` size probe on either name answers presence without
//!   reading or decoding entries. `ENODATA` means the attribute (and
//!   therefore an extended ACL) is absent; `ENOTSUP` means the
//!   filesystem cannot hold POSIX ACLs at all, which is also an honest
//!   [`ExtendedAclPresence::Absent`].
//!
//!   **`O_PATH` descriptors are not probeable**: `fgetxattr` on an
//!   `O_PATH` fd fails with `EBADF`, which this module surfaces as the
//!   error it is. Callers holding `O_PATH` capabilities (the
//!   generation-root admission pipeline does) must probe through a
//!   data descriptor for the same object, never the `O_PATH` handle —
//!   there is a regression test pinning the `EBADF` behavior so this
//!   footgun stays visible.
//! * **Other Unix** — [`io::ErrorKind::Unsupported`]: FreeBSD and
//!   friends have real ACLs we have not audited a probe for, and
//!   silently answering `Absent` there would turn a hardening gate into
//!   a no-op.
//!
//! The result is a point-in-time observation: a sufficiently privileged
//! process may attach an ACL after the probe returns. Callers that need
//! stability must sandwich the probe between object-identity witnesses,
//! as the generation-root gate does.

use std::io;
use std::os::fd::BorrowedFd;

/// Whether the filesystem object behind a descriptor has an extended ACL.
///
/// Security gates consuming this must match **exhaustively on `Absent`**
/// (admit only the known-good state) rather than testing `!= Present` /
/// `== Present`: if a future platform forces a third state, an equality
/// test silently fails open while an exhaustive match fails the build at
/// every admission site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the observed ACL presence must be handled"]
pub enum ExtendedAclPresence {
    /// No extended ACL is attached to the object.
    Absent,
    /// An extended ACL is attached to the object.
    Present,
}

/// Detect whether the object referenced by `fd` carries an extended ACL.
///
/// Queries the retained descriptor directly: no pathname is resolved, and
/// the descriptor is neither duplicated nor closed.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] when the platform query fails for
/// any reason other than the platform's "no extended ACL is attached"
/// answer, and [`io::ErrorKind::Unsupported`] on Unix platforms without an
/// audited probe.
pub fn extended_acl_presence(fd: BorrowedFd<'_>) -> io::Result<ExtendedAclPresence> {
    imp::extended_acl_presence(fd)
}

#[cfg(target_os = "macos")]
mod imp {
    use super::ExtendedAclPresence;
    use std::io;
    use std::os::fd::{AsRawFd, BorrowedFd};
    use std::os::raw::{c_int, c_uint, c_void};

    /// `ACL_TYPE_EXTENDED` from `<sys/acl.h>`: the only ACL type Darwin
    /// implements.
    const ACL_TYPE_EXTENDED: c_uint = 0x0000_0100;

    // The `acl_*` family is not exposed by the `libc` crate for Apple
    // targets, so the two calls the probe needs are declared here against
    // their `<sys/acl.h>` prototypes (`acl_t` is an opaque pointer).
    //
    // AUTHORIZATION (Rule-0): these two named Darwin FFI seams —
    // `acl_get_fd_np` and `acl_free` — are the only in-scope way to read a
    // macOS extended ACL through a retained fd without an external crate
    // (which project policy prohibits) or shelling out. The project owner
    // explicitly authorized this specific unsafe FFI on 2026-07-31, in
    // preference to reintroducing the `exacl` dependency. Landing it into
    // the generation-root ACL gate remains subject to the campaign's
    // physical-M4 proof; nothing beyond these two declarations and their
    // two call sites below is covered by this authorization.
    #[allow(unsafe_code)] // FFI prototypes for Darwin's libc ACL API (owner-authorized, see above).
    unsafe extern "C" {
        fn acl_get_fd_np(fd: c_int, acl_type: c_uint) -> *mut c_void;
        fn acl_free(obj: *mut c_void) -> c_int;
    }

    pub(super) fn extended_acl_presence(fd: BorrowedFd<'_>) -> io::Result<ExtendedAclPresence> {
        // SAFETY: `BorrowedFd` guarantees the descriptor stays open for the
        // duration of this call, Darwin's getter does not retain the
        // descriptor, and `ACL_TYPE_EXTENDED` is the platform's valid ACL
        // type constant.
        #[allow(unsafe_code)]
        let acl = unsafe { acl_get_fd_np(fd.as_raw_fd(), ACL_TYPE_EXTENDED) };
        if acl.is_null() {
            let error = io::Error::last_os_error();
            return if error.raw_os_error() == Some(libc::ENOENT) {
                Ok(ExtendedAclPresence::Absent)
            } else {
                Err(error)
            };
        }

        // SAFETY: `acl` is the non-null allocation returned by
        // `acl_get_fd_np` above; it is freed exactly once and never used
        // after this call.
        #[allow(unsafe_code)]
        let freed = unsafe { acl_free(acl) };
        if freed != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(ExtendedAclPresence::Present)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::ExtendedAclPresence;
    use std::ffi::CStr;
    use std::io;
    use std::os::fd::{AsRawFd, BorrowedFd};

    const ACL_XATTR_NAMES: [&CStr; 2] = [c"system.posix_acl_access", c"system.posix_acl_default"];

    pub(super) fn extended_acl_presence(fd: BorrowedFd<'_>) -> io::Result<ExtendedAclPresence> {
        for name in ACL_XATTR_NAMES {
            // SAFETY: the descriptor is kept open by `BorrowedFd` for the
            // duration of the call, `name` is a NUL-terminated C string,
            // and a null destination buffer with size 0 is the documented
            // fgetxattr size-probe form (no memory is written).
            #[allow(unsafe_code)]
            let size =
                unsafe { libc::fgetxattr(fd.as_raw_fd(), name.as_ptr(), std::ptr::null_mut(), 0) };
            if size >= 0 {
                return Ok(ExtendedAclPresence::Present);
            }
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                // ENODATA: this ACL xattr is not attached. ENOTSUP: the
                // filesystem cannot hold POSIX ACLs, so none is attached.
                Some(libc::ENODATA | libc::ENOTSUP) => {}
                _ => return Err(error),
            }
        }
        Ok(ExtendedAclPresence::Absent)
    }
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "linux"))))]
mod imp {
    use super::ExtendedAclPresence;
    use std::io;
    use std::os::fd::BorrowedFd;

    pub(super) fn extended_acl_presence(_fd: BorrowedFd<'_>) -> io::Result<ExtendedAclPresence> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "extended-ACL presence probing is only audited for macOS and Linux",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtendedAclPresence, extended_acl_presence};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsFd;
    use std::path::Path;
    use std::process::Command;

    /// Run an ACL-mutating platform binary, or return `false` when the
    /// binary is unavailable so the test can skip rather than lie.
    fn run_acl_tool(program: &str, args: &[&str]) -> io::Result<bool> {
        match Command::new(program).args(args).output() {
            Ok(output) if output.status.success() => Ok(true),
            Ok(output) => Err(io::Error::other(format!(
                "{program} {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn assert_presence(path: &Path, expected: ExtendedAclPresence, context: &str) {
        let handle = File::open(path).expect("fixture must open");
        assert_eq!(
            extended_acl_presence(handle.as_fd()).expect("presence probe must succeed"),
            expected,
            "{context}"
        );
    }

    // Gated to the two audited platforms: on other Unix the probe
    // intentionally returns Unsupported, and this test would panic with a
    // misleading "must succeed" message rather than documenting that fact.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn clean_file_and_directory_report_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("clean-file");
        File::create(&file_path).expect("create fixture file");

        assert_presence(
            &file_path,
            ExtendedAclPresence::Absent,
            "a freshly created file must report Absent",
        );
        assert_presence(
            dir.path(),
            ExtendedAclPresence::Absent,
            "a freshly created directory must report Absent",
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_access_acl_toggles_presence_through_a_retained_fd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("acl-file");
        File::create(&file_path).expect("create fixture file");
        let retained = File::open(&file_path).expect("open fixture");
        let path_text = file_path.to_str().expect("utf8 fixture path");

        let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
        let spec = format!("u:{user}:r");
        match run_acl_tool("setfacl", &["-m", &spec, path_text]) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("skipping: setfacl binary not installed");
                return;
            }
            Err(error) => {
                // tmpdirs on exotic filesystems may legitimately refuse.
                eprintln!("skipping: setfacl unusable here: {error}");
                return;
            }
        }
        assert_eq!(
            extended_acl_presence(retained.as_fd()).expect("probe after setfacl"),
            ExtendedAclPresence::Present,
            "an access ACL must be observed through the retained descriptor"
        );

        assert!(
            run_acl_tool("setfacl", &["-b", path_text]).expect("setfacl -b must succeed"),
            "setfacl disappeared mid-test"
        );
        assert_eq!(
            extended_acl_presence(retained.as_fd()).expect("probe after clear"),
            ExtendedAclPresence::Absent,
            "clearing the ACL must return the retained descriptor to Absent"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_default_directory_acl_reports_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sub = dir.path().join("default-acl-dir");
        std::fs::create_dir(&sub).expect("create fixture dir");
        let path_text = sub.to_str().expect("utf8 fixture path");

        let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
        let spec = format!("u:{user}:rx");
        match run_acl_tool("setfacl", &["-d", "-m", &spec, path_text]) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("skipping: setfacl binary not installed");
                return;
            }
            Err(error) => {
                eprintln!("skipping: setfacl unusable here: {error}");
                return;
            }
        }
        assert_presence(
            &sub,
            ExtendedAclPresence::Present,
            "a default ACL alone must report Present for a directory",
        );
    }

    /// `fgetxattr` cannot service `O_PATH` descriptors: the probe must
    /// surface `EBADF` loudly instead of pretending an answer. The
    /// generation-root pipeline holds `O_PATH` capabilities, so this
    /// pin keeps anyone from wiring those directly into the probe.
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_o_path_descriptor_is_rejected_not_misread() {
        use std::os::fd::{AsFd, FromRawFd, OwnedFd};

        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("target");
        File::create(&file_path).expect("create fixture file");
        let path_cstr = std::ffi::CString::new(file_path.to_str().expect("utf8 fixture path"))
            .expect("no interior NUL");

        // SAFETY: open(2) with valid arguments; the raw fd is immediately
        // wrapped in OwnedFd, which closes it on drop.
        #[allow(unsafe_code)]
        let raw = unsafe { libc::open(path_cstr.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
        assert!(raw >= 0, "O_PATH open must succeed");
        #[allow(unsafe_code)]
        let o_path_fd = unsafe { OwnedFd::from_raw_fd(raw) };

        let error = extended_acl_presence(o_path_fd.as_fd())
            .expect_err("an O_PATH descriptor must be rejected, not misread as Absent");
        assert_eq!(
            error.raw_os_error(),
            Some(libc::EBADF),
            "the kernel's EBADF must surface unchanged"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_chmod_acl_toggles_presence_through_a_retained_fd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("acl-file");
        File::create(&file_path).expect("create fixture file");
        let retained = File::open(&file_path).expect("open fixture");
        let path_text = file_path.to_str().expect("utf8 fixture path");

        let user = std::env::var("USER").expect("USER must identify an ACL principal");
        let entry = format!("user:{user} allow read");
        assert!(
            run_acl_tool("/bin/chmod", &["+a", &entry, path_text]).expect("chmod +a must succeed"),
            "/bin/chmod must exist on macOS"
        );
        assert_eq!(
            extended_acl_presence(retained.as_fd()).expect("probe after chmod +a"),
            ExtendedAclPresence::Present,
            "an ALLOW entry must be observed through the retained descriptor"
        );

        assert!(
            run_acl_tool("/bin/chmod", &["-N", path_text]).expect("chmod -N must succeed"),
            "/bin/chmod must exist on macOS"
        );
        assert_eq!(
            extended_acl_presence(retained.as_fd()).expect("probe after chmod -N"),
            ExtendedAclPresence::Absent,
            "clearing the ACL must return the retained descriptor to Absent"
        );
    }
}
