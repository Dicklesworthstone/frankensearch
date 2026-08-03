//! Pure route validation for capability-based generation roots.
//!
//! This module deliberately separates route syntax from filesystem admission.
//! A later descriptor-relative traversal layer may only receive a
//! [`GenerationRootRouteV1`] produced here, so it never needs to reinterpret
//! ambient, absolute, or traversal-bearing paths.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::File;
use std::io::Read as _;
use std::path::{Component, Path};
use std::sync::Arc;

/// Bounded reason why a generation-root route cannot enter capability-based
/// filesystem traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationRootRouteErrorV1 {
    /// The route has no component.
    Empty,
    /// The route contains a separator, dot component, parent component, or
    /// platform prefix that would make its interpretation non-canonical.
    UnsafeComponent,
    /// The route contains an embedded NUL byte.
    NulByte,
}

impl fmt::Display for GenerationRootRouteErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "generation-root route is empty",
            Self::UnsafeComponent => "generation-root route has an unsafe component",
            Self::NulByte => "generation-root route contains a NUL byte",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for GenerationRootRouteErrorV1 {}

/// Bounded reason why a trusted generation-root descriptor cannot admit a
/// descendant directory or immutable file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationRootAdmissionErrorV1 {
    /// The supplied descriptor is not a directory.
    NotDirectory,
    /// A descendant crosses the trusted root's device boundary.
    CrossDevice,
    /// A candidate is not a single-link regular file.
    UnsafeFileType,
    /// A file owner does not match the caller's explicit policy.
    OwnerMismatch,
    /// A file mode does not match the caller's explicit policy.
    ModeMismatch,
    /// A file exceeded the caller's bounded read ceiling.
    FileTooLarge,
    /// Identity metadata changed while the file was being retained.
    IdentityChanged,
    /// The platform does not yet provide the required descriptor admission.
    UnsupportedPlatform,
    /// A descriptor operation failed without exposing path text.
    Io,
}

impl fmt::Display for GenerationRootAdmissionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NotDirectory => "generation-root descriptor is not a directory",
            Self::CrossDevice => "generation-root descendant crosses a device boundary",
            Self::UnsafeFileType => "generation-root candidate is not a safe regular file",
            Self::OwnerMismatch => "generation-root file owner does not match policy",
            Self::ModeMismatch => "generation-root file mode does not match policy",
            Self::FileTooLarge => "generation-root file exceeds the bounded read ceiling",
            Self::IdentityChanged => "generation-root file identity changed while reading",
            Self::UnsupportedPlatform => "generation-root descriptor admission is unsupported",
            Self::Io => "generation-root descriptor operation failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for GenerationRootAdmissionErrorV1 {}

/// One canonical relative route to be resolved from an already trusted
/// directory descriptor.
///
/// The type owns only normal path components. It does not touch the
/// filesystem, follow symlinks, consult the current directory, or normalize a
/// caller-controlled route. Those operations are deferred to the
/// descriptor-relative admission layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRootRouteV1 {
    components: Vec<OsString>,
}

impl GenerationRootRouteV1 {
    /// Parse one canonical, non-empty relative route.
    ///
    /// # Errors
    ///
    /// Returns a bounded error for every ambiguous route shape before any
    /// filesystem operation can occur.
    pub fn parse(route: &Path) -> Result<Self, GenerationRootRouteErrorV1> {
        #[cfg(unix)]
        validate_unix_route_bytes(route.as_os_str())?;

        let mut components = Vec::new();
        for component in route.components() {
            match component {
                Component::Normal(component) => components.push(component.to_owned()),
                Component::CurDir
                | Component::ParentDir
                | Component::RootDir
                | Component::Prefix(_) => return Err(GenerationRootRouteErrorV1::UnsafeComponent),
            }
        }
        if components.is_empty() {
            return Err(GenerationRootRouteErrorV1::Empty);
        }
        Ok(Self { components })
    }

    /// Construct a route containing exactly one normal component.
    ///
    /// # Errors
    ///
    /// Returns the same bounded validation error as [`Self::parse`].
    pub fn from_component(component: &OsStr) -> Result<Self, GenerationRootRouteErrorV1> {
        let route = Self::parse(Path::new(component))?;
        if route.components.len() != 1 {
            return Err(GenerationRootRouteErrorV1::UnsafeComponent);
        }
        Ok(route)
    }

    /// Components to resolve in order from an explicit trusted descriptor.
    #[must_use]
    pub fn components(&self) -> &[OsString] {
        &self.components
    }
}

/// Explicit immutable-file policy supplied by the generation-root owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootFilePolicyV1 {
    /// Required Unix owner identity.
    pub expected_uid: u32,
    /// Required permission bits, excluding the file-type bits.
    pub expected_mode: u32,
    /// Maximum exact byte count admitted into retained memory.
    pub max_bytes: u64,
}

/// Explicit ownership and permission policy for the trusted root descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootDirectoryPolicyV1 {
    /// Required Unix owner identity.
    pub expected_uid: u32,
    /// Required permission bits, excluding the directory type bits.
    pub expected_mode: u32,
}

/// Stable descriptor witness captured before and after a bounded file read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootFileWitnessV1 {
    /// Owning filesystem device number.
    pub device: u64,
    /// Inode number bound to the retained descriptor.
    pub inode: u64,
    /// Full raw mode captured from the descriptor.
    pub mode: u32,
    /// Link count captured from the descriptor.
    pub links: u64,
    /// Owner identity captured from the descriptor.
    pub uid: u32,
    /// Group identity captured from the descriptor.
    pub gid: u32,
    /// Exact retained length.
    pub size: u64,
    /// Modification timestamp seconds and nanoseconds.
    pub mtime: (i64, u64),
    /// Status-change timestamp seconds and nanoseconds.
    pub ctime: (i64, u64),
    /// Last-access timestamp seconds and nanoseconds.
    ///
    /// This is part of the retained descriptor's mutation witness: an
    /// out-of-band reader that mutates access time must invalidate admission
    /// instead of being silently invisible to the before/after comparison.
    pub atime: (i64, u64),
}

/// A descriptor-retained, byte-exact immutable file admitted below a trusted
/// generation-root capability.
#[derive(Debug)]
pub struct RetainedGenerationRootFileV1 {
    #[cfg(target_os = "linux")]
    descriptor: File,
    bytes: Arc<[u8]>,
    witness: GenerationRootFileWitnessV1,
}

impl RetainedGenerationRootFileV1 {
    /// Immutable bytes read from the retained descriptor.
    #[must_use]
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    /// Stable descriptor witness validated before and after the read.
    #[must_use]
    pub const fn witness(&self) -> GenerationRootFileWitnessV1 {
        self.witness
    }
}

#[cfg(target_os = "linux")]
impl std::os::fd::AsFd for RetainedGenerationRootFileV1 {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        std::os::fd::AsFd::as_fd(&self.descriptor)
    }
}

/// A caller-provided directory capability with no ambient path fallback.
#[derive(Debug)]
pub struct GenerationRootCapabilityV1 {
    #[cfg(target_os = "linux")]
    directory: File,
    #[cfg(target_os = "linux")]
    device: u64,
    #[cfg(target_os = "linux")]
    directory_policy: GenerationRootDirectoryPolicyV1,
}

impl GenerationRootCapabilityV1 {
    /// Admit an explicit directory descriptor as a trusted generation root.
    ///
    /// # Errors
    ///
    /// Linux rejects non-directory descriptors. Other targets intentionally
    /// return a typed zero-I/O unsupported result until their platform
    /// qualification is implemented.
    #[cfg(target_os = "linux")]
    pub fn from_trusted_directory(
        directory: File,
        policy: GenerationRootDirectoryPolicyV1,
    ) -> Result<Self, GenerationRootAdmissionErrorV1> {
        let stat = rustix::fs::fstat(&directory).map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
        if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::Directory {
            return Err(GenerationRootAdmissionErrorV1::NotDirectory);
        }
        if stat.st_uid != policy.expected_uid {
            return Err(GenerationRootAdmissionErrorV1::OwnerMismatch);
        }
        if stat.st_mode & 0o7777 != policy.expected_mode {
            return Err(GenerationRootAdmissionErrorV1::ModeMismatch);
        }
        Ok(Self {
            directory,
            device: stat.st_dev,
            directory_policy: policy,
        })
    }

    /// Typed zero-I/O stub for non-Linux qualification targets.
    #[cfg(not(target_os = "linux"))]
    pub fn from_trusted_directory(
        directory: File,
        policy: GenerationRootDirectoryPolicyV1,
    ) -> Result<Self, GenerationRootAdmissionErrorV1> {
        drop(directory);
        let _ = policy;
        Err(GenerationRootAdmissionErrorV1::UnsupportedPlatform)
    }

    /// Resolve one canonical relative route from the retained root descriptor.
    ///
    /// # Errors
    ///
    /// Rejects symlink/non-directory traversal and nested-device routes.
    #[cfg(target_os = "linux")]
    pub fn descend(
        &self,
        route: &GenerationRootRouteV1,
    ) -> Result<Self, GenerationRootAdmissionErrorV1> {
        let mut current = self
            .directory
            .try_clone()
            .map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
        for component in route.components() {
            let descriptor = rustix::fs::openat(
                &current,
                component,
                rustix::fs::OFlags::RDONLY
                    | rustix::fs::OFlags::CLOEXEC
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::DIRECTORY
                    | rustix::fs::OFlags::NONBLOCK,
                rustix::fs::Mode::empty(),
            )
            .map_err(|error| match error {
                rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR => {
                    GenerationRootAdmissionErrorV1::NotDirectory
                }
                _ => GenerationRootAdmissionErrorV1::Io,
            })?;
            let stat =
                rustix::fs::fstat(&descriptor).map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
            if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::Directory
            {
                return Err(GenerationRootAdmissionErrorV1::NotDirectory);
            }
            if stat.st_dev != self.device {
                return Err(GenerationRootAdmissionErrorV1::CrossDevice);
            }
            if stat.st_uid != self.directory_policy.expected_uid {
                return Err(GenerationRootAdmissionErrorV1::OwnerMismatch);
            }
            if stat.st_mode & 0o7777 != self.directory_policy.expected_mode {
                return Err(GenerationRootAdmissionErrorV1::ModeMismatch);
            }
            current = File::from(descriptor);
        }
        Ok(Self {
            directory: current,
            device: self.device,
            directory_policy: self.directory_policy,
        })
    }

    /// Read and retain an immutable single-link regular file below this root.
    ///
    /// # Errors
    ///
    /// Rejects symlinks, special files, policy mismatch, excessive allocation,
    /// and descriptor identity changes without reopening by pathname.
    #[cfg(target_os = "linux")]
    pub fn read_regular_file(
        &self,
        name: &OsStr,
        policy: GenerationRootFilePolicyV1,
    ) -> Result<RetainedGenerationRootFileV1, GenerationRootAdmissionErrorV1> {
        let route = GenerationRootRouteV1::from_component(name)
            .map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
        let component = route
            .components()
            .first()
            .ok_or(GenerationRootAdmissionErrorV1::Io)?;
        let descriptor = rustix::fs::openat(
            &self.directory,
            component,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::CLOEXEC
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK
                // The atime witness must not be mutated by this admission
                // read. Refuse admission if Linux denies this flag.
                | rustix::fs::OFlags::from_bits_retain(libc::O_NOATIME as u32),
            rustix::fs::Mode::empty(),
        )
        .map_err(|error| match error {
            rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR => {
                GenerationRootAdmissionErrorV1::UnsafeFileType
            }
            _ => GenerationRootAdmissionErrorV1::Io,
        })?;
        let mut file = File::from(descriptor);
        let before = file_witness(&file)?;
        validate_file_policy(before, self.device, policy)?;
        let capacity = usize::try_from(before.size)
            .map_err(|_| GenerationRootAdmissionErrorV1::FileTooLarge)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| GenerationRootAdmissionErrorV1::FileTooLarge)?;
        file.by_ref()
            .take(policy.max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
        if u64::try_from(bytes.len()).map_err(|_| GenerationRootAdmissionErrorV1::FileTooLarge)?
            != before.size
        {
            return Err(GenerationRootAdmissionErrorV1::IdentityChanged);
        }
        let after = file_witness(&file)?;
        if before != after {
            return Err(GenerationRootAdmissionErrorV1::IdentityChanged);
        }
        Ok(RetainedGenerationRootFileV1 {
            descriptor: file,
            bytes: Arc::from(bytes),
            witness: before,
        })
    }
}

#[cfg(target_os = "linux")]
fn file_witness(
    file: &File,
) -> Result<GenerationRootFileWitnessV1, GenerationRootAdmissionErrorV1> {
    let stat = rustix::fs::fstat(file).map_err(|_| GenerationRootAdmissionErrorV1::Io)?;
    let size =
        u64::try_from(stat.st_size).map_err(|_| GenerationRootAdmissionErrorV1::UnsafeFileType)?;
    Ok(GenerationRootFileWitnessV1 {
        device: stat.st_dev,
        inode: stat.st_ino,
        mode: stat.st_mode,
        links: stat.st_nlink,
        uid: stat.st_uid,
        gid: stat.st_gid,
        size,
        mtime: (stat.st_mtime, stat.st_mtime_nsec),
        ctime: (stat.st_ctime, stat.st_ctime_nsec),
        atime: (stat.st_atime, stat.st_atime_nsec),
    })
}

#[cfg(target_os = "linux")]
fn validate_file_policy(
    witness: GenerationRootFileWitnessV1,
    root_device: u64,
    policy: GenerationRootFilePolicyV1,
) -> Result<(), GenerationRootAdmissionErrorV1> {
    if witness.device != root_device {
        return Err(GenerationRootAdmissionErrorV1::CrossDevice);
    }
    if rustix::fs::FileType::from_raw_mode(witness.mode) != rustix::fs::FileType::RegularFile
        || witness.links != 1
    {
        return Err(GenerationRootAdmissionErrorV1::UnsafeFileType);
    }
    if witness.uid != policy.expected_uid {
        return Err(GenerationRootAdmissionErrorV1::OwnerMismatch);
    }
    if witness.mode & 0o7777 != policy.expected_mode {
        return Err(GenerationRootAdmissionErrorV1::ModeMismatch);
    }
    if witness.size > policy.max_bytes {
        return Err(GenerationRootAdmissionErrorV1::FileTooLarge);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_unix_route_bytes(route: &OsStr) -> Result<(), GenerationRootRouteErrorV1> {
    use std::os::unix::ffi::OsStrExt as _;

    let bytes = route.as_bytes();
    if bytes.is_empty() {
        return Err(GenerationRootRouteErrorV1::Empty);
    }
    if bytes.contains(&0) {
        return Err(GenerationRootRouteErrorV1::NulByte);
    }
    if bytes
        .split(|byte| *byte == b'/')
        .any(|component| component.is_empty() || matches!(component, b"." | b".."))
    {
        return Err(GenerationRootRouteErrorV1::UnsafeComponent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GenerationRootAdmissionErrorV1, GenerationRootCapabilityV1,
        GenerationRootDirectoryPolicyV1, GenerationRootFilePolicyV1, GenerationRootRouteErrorV1,
        GenerationRootRouteV1,
    };
    use std::ffi::OsStr;
    use std::path::Path;

    #[cfg(target_os = "linux")]
    fn admit_test_root(directory: &tempfile::TempDir) -> GenerationRootCapabilityV1 {
        use std::fs::{self, File};
        use std::os::unix::fs::MetadataExt as _;

        let metadata = fs::metadata(directory.path()).expect("trusted root metadata");
        GenerationRootCapabilityV1::from_trusted_directory(
            File::open(directory.path()).expect("open trusted root descriptor"),
            GenerationRootDirectoryPolicyV1 {
                expected_uid: metadata.uid(),
                expected_mode: metadata.mode() & 0o7777,
            },
        )
        .expect("admit private test root")
    }

    #[test]
    fn route_preserves_normal_component_order_without_filesystem_access() {
        let route = GenerationRootRouteV1::parse(Path::new("generations/0007/AUTHORITY"))
            .expect("normal relative route");
        let names: Vec<_> = route
            .components()
            .iter()
            .map(|component| component.to_string_lossy())
            .collect();
        assert_eq!(names, ["generations", "0007", "AUTHORITY"]);
    }

    #[test]
    fn route_rejects_ambiguous_components_before_descriptor_open() {
        assert_eq!(
            GenerationRootRouteV1::parse(Path::new("")),
            Err(GenerationRootRouteErrorV1::Empty)
        );
        for route in [".", "..", "a/./b", "a/../b", "/absolute", "a//b", "a/"] {
            assert_eq!(
                GenerationRootRouteV1::parse(Path::new(route)),
                Err(GenerationRootRouteErrorV1::UnsafeComponent),
                "route {route:?} must never be normalized into an admitted capability route"
            );
        }
    }

    #[test]
    fn single_component_constructor_refuses_separators() {
        assert_eq!(
            GenerationRootRouteV1::from_component(OsStr::new("AUTHORITY/next")),
            Err(GenerationRootRouteErrorV1::UnsafeComponent)
        );
    }

    #[cfg(unix)]
    #[test]
    fn route_rejects_embedded_nul_without_lossy_conversion() {
        use std::os::unix::ffi::OsStrExt as _;

        let route = OsStr::from_bytes(b"AUTHORITY\0shadow");
        assert_eq!(
            GenerationRootRouteV1::from_component(route),
            Err(GenerationRootRouteErrorV1::NulByte)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn retained_file_is_descriptor_bound_and_policy_checked() {
        use std::fs::{self, File};
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let directory = tempfile::tempdir().expect("temporary trusted root");
        let path = directory.path().join("AUTHORITY");
        fs::write(&path, b"immutable authority bytes").expect("write authority fixture");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .expect("restrict authority fixture mode");
        let metadata = fs::metadata(&path).expect("authority fixture metadata");
        let policy = GenerationRootFilePolicyV1 {
            expected_uid: metadata.uid(),
            expected_mode: 0o600,
            max_bytes: 4_096,
        };
        let root = admit_test_root(&directory);
        let root_metadata = fs::metadata(directory.path()).expect("trusted root metadata");
        assert!(matches!(
            GenerationRootCapabilityV1::from_trusted_directory(
                File::open(directory.path()).expect("reopen trusted root descriptor"),
                GenerationRootDirectoryPolicyV1 {
                    expected_uid: root_metadata.uid(),
                    expected_mode: (root_metadata.mode() & 0o7777) ^ 0o100,
                },
            ),
            Err(GenerationRootAdmissionErrorV1::ModeMismatch)
        ));

        let retained = root
            .read_regular_file(OsStr::new("AUTHORITY"), policy)
            .expect("admit immutable authority file");
        assert_eq!(retained.bytes().as_ref(), b"immutable authority bytes");
        assert_eq!(retained.witness().uid, metadata.uid());
        assert_eq!(retained.witness().size, 25);
        assert_eq!(
            retained.witness().atime,
            (metadata.atime(), metadata.atime_nsec() as u64),
            "the retained descriptor witness includes access-time identity"
        );

        assert!(
            matches!(
                root.read_regular_file(
                    OsStr::new("AUTHORITY"),
                    GenerationRootFilePolicyV1 {
                        expected_uid: metadata.uid(),
                        expected_mode: 0o640,
                        max_bytes: 4_096,
                    },
                ),
                Err(GenerationRootAdmissionErrorV1::ModeMismatch)
            ),
            "mode policy is enforced from the retained descriptor, not path metadata"
        );
        assert!(matches!(
            root.read_regular_file(
                OsStr::new("AUTHORITY"),
                GenerationRootFilePolicyV1 {
                    expected_uid: metadata.uid().saturating_add(1),
                    expected_mode: 0o600,
                    max_bytes: 4_096,
                },
            ),
            Err(GenerationRootAdmissionErrorV1::OwnerMismatch)
        ));
        assert!(matches!(
            root.read_regular_file(
                OsStr::new("AUTHORITY"),
                GenerationRootFilePolicyV1 {
                    expected_uid: metadata.uid(),
                    expected_mode: 0o600,
                    max_bytes: 24,
                },
            ),
            Err(GenerationRootAdmissionErrorV1::FileTooLarge)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_open_refuses_a_final_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary trusted root");
        symlink("elsewhere", directory.path().join("AUTHORITY"))
            .expect("create final symlink fixture");
        let root = admit_test_root(&directory);
        assert!(matches!(
            root.read_regular_file(
                OsStr::new("AUTHORITY"),
                GenerationRootFilePolicyV1 {
                    expected_uid: 0,
                    expected_mode: 0o600,
                    max_bytes: 4_096,
                },
            ),
            Err(GenerationRootAdmissionErrorV1::UnsafeFileType)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_descent_keeps_normal_children_and_rejects_symlink_ancestors() {
        use std::fs;
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _, symlink};

        let directory = tempfile::tempdir().expect("temporary trusted root");
        fs::create_dir(directory.path().join("generations")).expect("create child directory");
        let root_metadata = fs::metadata(directory.path()).expect("trusted root metadata");
        fs::set_permissions(
            directory.path().join("generations"),
            fs::Permissions::from_mode(root_metadata.mode() & 0o7777),
        )
        .expect("align child directory policy with trusted root");
        let root = admit_test_root(&directory);
        let normal = GenerationRootRouteV1::parse(Path::new("generations"))
            .expect("normal descendant route");
        root.descend(&normal)
            .expect("descriptor-relative normal descent");

        symlink("generations", directory.path().join("redirect"))
            .expect("create ancestor symlink fixture");
        let redirected = GenerationRootRouteV1::parse(Path::new("redirect"))
            .expect("syntactically normal route");
        assert!(matches!(
            root.descend(&redirected),
            Err(GenerationRootAdmissionErrorV1::NotDirectory)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_descent_rejects_child_directory_mode_outside_root_policy() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().expect("temporary trusted root");
        let child = directory.path().join("untrusted-child");
        fs::create_dir(&child).expect("create child directory");
        fs::set_permissions(&child, fs::Permissions::from_mode(0o777))
            .expect("make child directory policy-incompatible");
        let root = admit_test_root(&directory);
        let route = GenerationRootRouteV1::parse(Path::new("untrusted-child"))
            .expect("syntactically normal route");

        assert!(matches!(
            root.descend(&route),
            Err(GenerationRootAdmissionErrorV1::ModeMismatch)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descriptor_open_refuses_hardlinked_final_file() {
        use std::fs;
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let directory = tempfile::tempdir().expect("temporary trusted root");
        let source = directory.path().join("source");
        let authority = directory.path().join("AUTHORITY");
        fs::write(&source, b"linked authority").expect("write source fixture");
        fs::set_permissions(&source, fs::Permissions::from_mode(0o600))
            .expect("restrict source fixture mode");
        fs::hard_link(&source, &authority).expect("create hardlink fixture");
        let metadata = fs::metadata(&authority).expect("authority hardlink metadata");
        let root = admit_test_root(&directory);
        assert!(matches!(
            root.read_regular_file(
                OsStr::new("AUTHORITY"),
                GenerationRootFilePolicyV1 {
                    expected_uid: metadata.uid(),
                    expected_mode: 0o600,
                    max_bytes: 4_096,
                },
            ),
            Err(GenerationRootAdmissionErrorV1::UnsafeFileType)
        ));
    }
}
