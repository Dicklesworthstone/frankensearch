//! Descriptor-owned admission for composite search-index generation roots.
//!
//! This module is deliberately narrower than the generation publisher. It
//! qualifies an existing private root, admits existing immutable regular
//! files into retained descriptors plus exact owned bytes, exposes fail-closed
//! durability barriers, and provides non-blocking kernel `flock` guards.
//! Publication, renames, unlinks, truncation, garbage collection, recovery,
//! and consumer activation do not belong here.
//!
//! # Supported targets and process binding
//!
//! Linux admits path-opened roots and final files only on locally qualified,
//! writable ext4 or Btrfs. Apple Silicon admits path-opened local writable APFS
//! roots, but every final artifact/control/lock file must arrive as an owned
//! already-open descriptor from the trusted provider documented by the
//! preopened APIs. Every other target returns
//! [`GenerationRootErrorKind::UnsupportedPlatform`] before route parsing or
//! filesystem I/O. Root, file, control, and lock capabilities are bound to the
//! process that created them; using any operation that revalidates or acts on
//! those capabilities after `fork` returns
//! [`GenerationRootErrorKind::ForkedProcess`].
//!
//! # Synchronous execution and resource boundary
//!
//! Every operation in this module is synchronous. Each exact-name component
//! proof accepts up to [`GENERATION_ROOT_MAX_DIRECTORY_ENTRIES`] names and
//! [`GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES`] aggregate name bytes, and may
//! inspect one overflow entry/name to reject it; one route can require up to
//! [`GENERATION_ROOT_MAX_COMPONENTS`] such proofs.
//! A route walk can invoke more than one exact-name proof per component, and
//! one public operation can walk the same route several times. The component
//! ceiling bounds depth, not the aggregate number of directory scans.
//! Linux mount qualification stores at most
//! [`GENERATION_ROOT_MAX_MOUNTINFO_BYTES`] from the calling thread's
//! task-scoped mount table and may read one overflow sentinel byte.
//! Immutable artifacts can read/hash up to [`GENERATION_ROOT_MAX_FILE_BYTES`]
//! bytes; repeatedly checked control anchors have the smaller
//! [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`] ceiling. Explicit immutable
//! durability performs two full-file reads, while control admission, locking,
//! and locked durability can perform multiple exact hash passes around route,
//! lock, and kernel-barrier checkpoints. These functions are neither async nor
//! cancellation-aware. Async consumers must run them in a bounded blocking
//! pool, propagate cancellation only at operation boundaries, and admit only
//! trusted local generation roots whose total work is independently budgeted.
//! The per-proof ceilings below bound individual work and memory; they are not
//! aggregate admission budgets or latency promises.
//!
//! # Linux retained-capability reopen
//!
//! Linux has one intentional, tightly bounded magic-link exception. After the
//! original route has produced and fully qualified a retained `O_PATH`
//! regular-file capability, the module opens a separately verified procfs root,
//! proves that its `self` link names the calling process and its `thread-self`
//! link names the calling thread, and preflights the exact final
//! `/proc/thread-self/fd/<held-O_PATH-fd>` magic link as a symlink on the same
//! procfs mount before and after deriving the data descriptor. Resolving
//! `thread-self` and the numeric fd entry in the data-open syscall binds the
//! reopen to the accessing thread's descriptor table even after
//! `unshare(CLONE_FILES)`; process-identity checks still reject child-side use
//! after `fork()`.
//! This bridge never re-resolves the original generation route. Descriptor-
//! bound directory traversal performs each final-file route revalidation with
//! an `O_PATH`-only final probe, and the retained probe, derived data
//! descriptor, final procfs probe, and final route probe must retain their
//! respective identities. As with every `/proc`-based capability protocol, the
//! calling mount namespace must be trusted against concurrent `CAP_SYS_ADMIN`
//! remounts; a process able to rewrite that namespace is outside this
//! substrate's authority boundary.

#![forbid(unsafe_code)]

use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// Maximum accepted absolute or relative route length.
pub const GENERATION_ROOT_MAX_ROUTE_BYTES: usize = 4096;
/// Maximum accepted component length.
pub const GENERATION_ROOT_MAX_COMPONENT_BYTES: usize = 255;
/// Maximum accepted component count.
pub const GENERATION_ROOT_MAX_COMPONENTS: usize = 64;
/// Hard ceiling for one owned immutable artifact image.
pub const GENERATION_ROOT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Hard ceiling for one repeatedly hashed mutable control anchor.
pub const GENERATION_ROOT_MAX_CONTROL_FILE_BYTES: u64 = 1024 * 1024;
/// Maximum names inspected while proving one exact directory entry.
pub const GENERATION_ROOT_MAX_DIRECTORY_ENTRIES: usize = 1_000_000;
/// Maximum aggregate name bytes inspected during one exact-name proof.
pub const GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES: usize = 64 * 1024 * 1024;
/// Maximum bytes read from Linux task-scoped procfs `mountinfo` for one qualification.
pub const GENERATION_ROOT_MAX_MOUNTINFO_BYTES: usize = 4 * 1024 * 1024;
/// Required mode for the qualified generation root and its directories.
pub const GENERATION_ROOT_DIRECTORY_MODE: u32 = 0o700;
/// Required sealed mode for immutable generation artifacts.
pub const GENERATION_ROOT_IMMUTABLE_FILE_MODE: u32 = 0o400;
/// Required mode for mutable generation control files.
pub const GENERATION_ROOT_CONTROL_FILE_MODE: u32 = 0o600;
/// Exact top-level name of the persistent generation-root lock anchor.
pub const GENERATION_ROOT_LOCK_FILE_NAME: &str = "LOCK";
/// Exact top-level name of the persistent generation-root authority anchor.
pub const GENERATION_ROOT_AUTHORITY_FILE_NAME: &str = "AUTHORITY";
/// Frozen physical length of the two-slot AUTHORITY inode.
///
/// Slot interpretation belongs to the authority codec. The platform layer
/// enforces only this immutable physical-layout fact.
pub const GENERATION_ROOT_AUTHORITY_FILE_BYTES: u64 = 8192;

/// Operation stage attached to every bounded generation-root failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GenerationRootStage {
    /// Reject an unsupported target before any filesystem access.
    PlatformGate,
    /// Parse and bound an absolute generation-root route.
    ParseRootRoute,
    /// Parse and bound a confined relative route.
    ParseRelativeRoute,
    /// Open the absolute route one component at a time.
    OpenRootRoute,
    /// Qualify root ownership and permissions.
    QualifyRootSecurity,
    /// Qualify the root filesystem and mount.
    QualifyFilesystem,
    /// Inspect an extended ACL through a retained descriptor.
    InspectAcl,
    /// Re-resolve and fence the absolute route.
    RevalidateRootRoute,
    /// Open and qualify a relative directory component.
    OpenRelativeDirectory,
    /// Open and qualify the final regular file.
    OpenRegularFile,
    /// Read one regular file into a bounded owned image.
    ReadRegularFile,
    /// Re-open and fence a relative file route.
    RevalidateRegularFile,
    /// Execute the platform's file durability barrier.
    SyncRegularFile,
    /// Execute the platform's directory durability barrier.
    SyncDirectory,
    /// Acquire a non-blocking kernel file lock.
    AcquireLock,
    /// Explicitly release a kernel file lock.
    ReleaseLock,
}

impl GenerationRootStage {
    const fn code(self) -> &'static str {
        match self {
            Self::PlatformGate => "platform_gate",
            Self::ParseRootRoute => "parse_root_route",
            Self::ParseRelativeRoute => "parse_relative_route",
            Self::OpenRootRoute => "open_root_route",
            Self::QualifyRootSecurity => "qualify_root_security",
            Self::QualifyFilesystem => "qualify_filesystem",
            Self::InspectAcl => "inspect_acl",
            Self::RevalidateRootRoute => "revalidate_root_route",
            Self::OpenRelativeDirectory => "open_relative_directory",
            Self::OpenRegularFile => "open_regular_file",
            Self::ReadRegularFile => "read_regular_file",
            Self::RevalidateRegularFile => "revalidate_regular_file",
            Self::SyncRegularFile => "sync_regular_file",
            Self::SyncDirectory => "sync_directory",
            Self::AcquireLock => "acquire_lock",
            Self::ReleaseLock => "release_lock",
        }
    }
}

/// Stable failure class for generation-root admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GenerationRootErrorKind {
    /// This architecture/OS combination is intentionally unsupported.
    UnsupportedPlatform,
    /// A path is empty, ambiguous, unbounded, absolute where relative was
    /// required, or contains a forbidden component.
    InvalidRoute,
    /// A required kernel primitive is unavailable or filtered.
    UnsupportedKernelFeature,
    /// Apple Silicon final-file admission requires a caller/provider-owned
    /// already-open descriptor so this library never invokes a vnode open
    /// callback before proving the object type.
    PreopenedDescriptorRequired,
    /// A symbolic or magic link was encountered.
    SymbolicLink,
    /// An ancestor or requested root is not a directory.
    NotDirectory,
    /// The final object is not a regular file.
    NotRegularFile,
    /// A regular file has more than one hard link.
    HardLinked,
    /// An object is not owned by the effective user.
    WrongOwner,
    /// An object's permission mode is not the exact private mode.
    WrongMode,
    /// A provider-owned descriptor is not opened with the exact data access
    /// mode required by the API, or carries a metadata/event-only access flag.
    InvalidDescriptorAccess,
    /// A provider-owned descriptor is missing close-on-exec.
    CloseOnExecRequired,
    /// Per-object immutable, append-only, or other write-restricting flags
    /// make the object incompatible with the qualified writable-root profile.
    WriteRestrictedObject,
    /// Apple object flags are nonzero and therefore outside the deliberately
    /// narrow zero-flags APFS admission profile.
    UnsupportedObjectFlags,
    /// A platform extended ACL is present or cannot be inspected exactly
    /// through the retained descriptor.
    AclRejected,
    /// A descendant crossed the qualified root's mount.
    CrossDevice,
    /// The filesystem is remote, synthetic, unknown, or not allowlisted.
    UnsupportedFilesystem,
    /// The filesystem is read-only or ignores ownership.
    ReadOnlyFilesystem,
    /// A retained object or its canonical route changed during admission.
    ObjectChanged,
    /// A file length differs from its exact expected value.
    SizeMismatch,
    /// An immutable-artifact API received a control-file expectation or the
    /// reverse.
    WrongRole,
    /// A caller or on-disk value exceeds a resource ceiling.
    ResourceLimit,
    /// The exact owned bytes do not match the expected SHA-256.
    HashMismatch,
    /// The required durability primitive failed; no weaker barrier was used.
    DurabilityUnavailable,
    /// A non-blocking shared or exclusive kernel lock is contended.
    LockContended,
    /// Admission or a lock operation crossed a `fork()` process-identity
    /// boundary and cannot continue in the child process.
    ForkedProcess,
    /// A bounded operating-system error not covered by a more specific class.
    Io,
}

impl GenerationRootErrorKind {
    /// Stable, machine-readable reason code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UnsupportedPlatform => "unsupported_platform",
            Self::InvalidRoute => "invalid_route",
            Self::UnsupportedKernelFeature => "unsupported_kernel_feature",
            Self::PreopenedDescriptorRequired => "preopened_descriptor_required",
            Self::SymbolicLink => "symbolic_link",
            Self::NotDirectory => "not_directory",
            Self::NotRegularFile => "not_regular_file",
            Self::HardLinked => "hard_linked",
            Self::WrongOwner => "wrong_owner",
            Self::WrongMode => "wrong_mode",
            Self::InvalidDescriptorAccess => "invalid_descriptor_access",
            Self::CloseOnExecRequired => "close_on_exec_required",
            Self::WriteRestrictedObject => "write_restricted_object",
            Self::UnsupportedObjectFlags => "unsupported_object_flags",
            Self::AclRejected => "acl_rejected",
            Self::CrossDevice => "cross_device",
            Self::UnsupportedFilesystem => "unsupported_filesystem",
            Self::ReadOnlyFilesystem => "read_only_filesystem",
            Self::ObjectChanged => "object_changed",
            Self::SizeMismatch => "size_mismatch",
            Self::WrongRole => "wrong_role",
            Self::ResourceLimit => "resource_limit",
            Self::HashMismatch => "hash_mismatch",
            Self::DurabilityUnavailable => "durability_unavailable",
            Self::LockContended => "lock_contended",
            Self::ForkedProcess => "forked_process",
            Self::Io => "io",
        }
    }
}

/// Bounded, path-redacted generation-root failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootError {
    kind: GenerationRootErrorKind,
    stage: GenerationRootStage,
    component_index: Option<u16>,
    raw_os_error: Option<i32>,
    expected: Option<u64>,
    observed: Option<u64>,
}

impl GenerationRootError {
    const fn new(kind: GenerationRootErrorKind, stage: GenerationRootStage) -> Self {
        Self {
            kind,
            stage,
            component_index: None,
            raw_os_error: None,
            expected: None,
            observed: None,
        }
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    fn at_component(mut self, component_index: usize) -> Self {
        self.component_index = Some(u16::try_from(component_index).unwrap_or(u16::MAX));
        self
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    const fn with_raw_os_error(mut self, raw_os_error: i32) -> Self {
        self.raw_os_error = Some(raw_os_error);
        self
    }

    const fn with_counts(mut self, expected: u64, observed: u64) -> Self {
        self.expected = Some(expected);
        self.observed = Some(observed);
        self
    }

    /// Stable failure class.
    #[must_use]
    pub const fn kind(&self) -> GenerationRootErrorKind {
        self.kind
    }

    /// Operation stage at which the failure was observed.
    #[must_use]
    pub const fn stage(&self) -> GenerationRootStage {
        self.stage
    }

    /// Zero-based path component index, when applicable.
    #[must_use]
    pub const fn component_index(&self) -> Option<u16> {
        self.component_index
    }

    /// Raw OS error number, when retaining it is useful and bounded.
    #[must_use]
    pub const fn raw_os_error(&self) -> Option<i32> {
        self.raw_os_error
    }

    /// Expected bounded numeric value, ceiling, count, or bitmask, when
    /// applicable.
    #[must_use]
    pub const fn expected(&self) -> Option<u64> {
        self.expected
    }

    /// Observed bounded numeric value, count, or bitmask, when applicable.
    #[must_use]
    pub const fn observed(&self) -> Option<u64> {
        self.observed
    }
}

impl fmt::Display for GenerationRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "generation-root {} failed: {}",
            self.stage.code(),
            self.kind.code()
        )?;
        if let Some(component_index) = self.component_index {
            write!(formatter, " (component {component_index})")?;
        }
        if let Some(raw_os_error) = self.raw_os_error {
            write!(formatter, " (os error {raw_os_error})")?;
        }
        if let (Some(expected), Some(observed)) = (self.expected, self.observed) {
            write!(formatter, " (expected {expected}, observed {observed})")?;
        }
        if self.kind == GenerationRootErrorKind::AclRejected {
            write!(
                formatter,
                " (only the descriptor-bound absence of a platform extended ACL \
                 is admissible; presence or inspection uncertainty fails closed)"
            )?;
        }
        if self.kind == GenerationRootErrorKind::PreopenedDescriptorRequired {
            write!(
                formatter,
                " (Apple Silicon requires a trusted external provider to supply an owned \
                 already-open regular-file descriptor; use the preopened admission/lock API)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for GenerationRootError {}

/// Result alias for this platform substrate.
pub type GenerationRootResult<T> = Result<T, GenerationRootError>;

/// Filesystem profile admitted for generation-root durability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QualifiedFilesystem {
    /// Linux ext4.
    LinuxExt4,
    /// Linux Btrfs.
    LinuxBtrfs,
    /// Local writable APFS on Apple Silicon.
    AppleApfs,
}

/// Stable metadata snapshot retained for one admitted filesystem object.
///
/// This is an object/metadata witness, not a content authority. Derived
/// equality and hashing compare the raw snapshot, including access time;
/// regular-file admission uses a narrower mutation-stable comparison that
/// deliberately excludes access time on every admitted platform. Root and
/// directory route revalidation is narrower again: it attests only
/// device/inode, mode, owner/group, mount identity, and filesystem profile,
/// because contained publication may legitimately change directory contents,
/// link count, size, and timestamps. Use the owning qualified
/// file/control/lock capability's SHA-256 accessor when content identity is
/// required; never interpret a root/directory revalidation as an attestation
/// of directory contents or timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationRootObjectWitness {
    device: u64,
    inode: u64,
    mode: u32,
    hard_links: u64,
    uid: u32,
    gid: u32,
    byte_len: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
    accessed_seconds: i64,
    accessed_nanoseconds: i64,
    mount_identity: [u8; 32],
    filesystem: QualifiedFilesystem,
}

impl GenerationRootObjectWitness {
    /// Device identifier.
    #[must_use]
    pub const fn device(&self) -> u64 {
        self.device
    }

    /// Inode identifier.
    #[must_use]
    pub const fn inode(&self) -> u64 {
        self.inode
    }

    /// Full persisted mode word.
    #[must_use]
    pub const fn mode(&self) -> u32 {
        self.mode
    }

    /// Hard-link count.
    #[must_use]
    pub const fn hard_links(&self) -> u64 {
        self.hard_links
    }

    /// Owner user id.
    #[must_use]
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// Owner group id.
    #[must_use]
    pub const fn gid(&self) -> u32 {
        self.gid
    }

    /// Exact byte length at admission.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Opaque, descriptor-derived filesystem namespace identity.
    ///
    /// On Btrfs this binds both the VFS mount and the containing subvolume.
    #[must_use]
    pub const fn mount_identity(&self) -> [u8; 32] {
        self.mount_identity
    }

    /// Qualified filesystem profile.
    #[must_use]
    pub const fn filesystem(&self) -> QualifiedFilesystem {
        self.filesystem
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    fn same_object(self, other: Self) -> bool {
        self.device == other.device && self.inode == other.inode
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    fn same_directory_security_identity(self, other: Self) -> bool {
        self.same_object(other)
            && self.mode == other.mode
            && self.uid == other.uid
            && self.gid == other.gid
            && self.mount_identity == other.mount_identity
            && self.filesystem == other.filesystem
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    fn same_file_security_identity(self, other: Self) -> bool {
        self.same_directory_security_identity(other) && self.hard_links == other.hard_links
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    fn same_control_anchor_identity(self, other: Self) -> bool {
        self.same_file_security_identity(other) && self.byte_len == other.byte_len
    }

    #[cfg(target_os = "linux")]
    fn mutation_stable_eq(self, other: Self) -> bool {
        self.same_file_security_identity(other)
            && self.byte_len == other.byte_len
            && self.modified_seconds == other.modified_seconds
            && self.modified_nanoseconds == other.modified_nanoseconds
            && self.changed_seconds == other.changed_seconds
            && self.changed_nanoseconds == other.changed_nanoseconds
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn mutation_stable_eq(self, other: Self) -> bool {
        self.same_file_security_identity(other)
            && self.byte_len == other.byte_len
            && self.modified_seconds == other.modified_seconds
            && self.modified_nanoseconds == other.modified_nanoseconds
            && self.changed_seconds == other.changed_seconds
            && self.changed_nanoseconds == other.changed_nanoseconds
    }
}

/// Fixed physical layout of the named generation-root control anchors.
///
/// The authority length is frozen by the generation-root protocol. The LOCK
/// length is supplied by the separately versioned owner/attempt-frame codec;
/// this platform layer neither parses nor authenticates those frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationRootAnchorLayout {
    lock_byte_len: u64,
}

impl GenerationRootAnchorLayout {
    /// Construct a bounded, non-empty named-anchor layout.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when the LOCK image
    /// is empty or exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn new(lock_byte_len: u64) -> GenerationRootResult<Self> {
        if lock_byte_len == 0 || lock_byte_len > GENERATION_ROOT_MAX_CONTROL_FILE_BYTES {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES, lock_byte_len));
        }
        Ok(Self { lock_byte_len })
    }

    /// Exact LOCK byte length supplied by the owner/attempt-frame codec.
    #[must_use]
    pub const fn lock_byte_len(&self) -> u64 {
        self.lock_byte_len
    }

    /// Frozen physical AUTHORITY byte length.
    #[must_use]
    pub const fn authority_byte_len(&self) -> u64 {
        GENERATION_ROOT_AUTHORITY_FILE_BYTES
    }
}

/// One descriptor-derived root/LOCK/AUTHORITY identity tuple.
///
/// This attests the retained physical objects, not the semantic validity of
/// either mutable control image. Content authority is exposed only through a
/// live read or writer guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationRootAnchorWitness {
    root: GenerationRootObjectWitness,
    lock: GenerationRootObjectWitness,
    authority: GenerationRootObjectWitness,
}

impl GenerationRootAnchorWitness {
    const fn new(
        root: GenerationRootObjectWitness,
        lock: GenerationRootObjectWitness,
        authority: GenerationRootObjectWitness,
    ) -> Self {
        Self {
            root,
            lock,
            authority,
        }
    }

    /// Qualified generation-root directory witness.
    #[must_use]
    pub const fn root(&self) -> GenerationRootObjectWitness {
        self.root
    }

    /// Retained named LOCK witness.
    #[must_use]
    pub const fn lock(&self) -> GenerationRootObjectWitness {
        self.lock
    }

    /// Retained named AUTHORITY witness.
    #[must_use]
    pub const fn authority(&self) -> GenerationRootObjectWitness {
        self.authority
    }
}

/// Sealed role for an admitted generation file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationFileRole {
    /// Immutable artifact, sealed read-only for its owner.
    ImmutableArtifact,
    /// Fixed-size control anchor whose future in-place writes require a
    /// publisher API that advances the admitted witness while holding the
    /// kernel lock.
    ///
    /// This qualification substrate exposes no write operation and rejects
    /// ambient mutation fail-closed.
    MutableControl,
}

/// Exact expected generation-file identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationFileExpectation {
    byte_len: u64,
    sha256: Option<[u8; 32]>,
    role: GenerationFileRole,
}

impl GenerationFileExpectation {
    /// Expect an exact length and SHA-256, subject to the global ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_FILE_BYTES`].
    pub fn immutable(byte_len: u64, sha256: [u8; 32]) -> GenerationRootResult<Self> {
        Self::new(
            byte_len,
            Some(sha256),
            GenerationFileRole::ImmutableArtifact,
        )
    }

    /// Expect the current exact control-file length without precommitting its
    /// content.
    ///
    /// This substrate admits and locks the unchanged image only. A later
    /// publisher must use a witness-advancing mutation API rather than changing
    /// the file through an ambient descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn control(byte_len: u64) -> GenerationRootResult<Self> {
        Self::new(byte_len, None, GenerationFileRole::MutableControl)
    }

    /// Expect an exact control-file length and SHA-256 at lock acquisition.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ResourceLimit`] when `byte_len`
    /// exceeds [`GENERATION_ROOT_MAX_CONTROL_FILE_BYTES`].
    pub fn control_with_sha256(byte_len: u64, sha256: [u8; 32]) -> GenerationRootResult<Self> {
        Self::new(byte_len, Some(sha256), GenerationFileRole::MutableControl)
    }

    fn new(
        byte_len: u64,
        sha256: Option<[u8; 32]>,
        role: GenerationFileRole,
    ) -> GenerationRootResult<Self> {
        let max_byte_len = match role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_MAX_FILE_BYTES,
            GenerationFileRole::MutableControl => GENERATION_ROOT_MAX_CONTROL_FILE_BYTES,
        };
        if byte_len > max_byte_len {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(max_byte_len, byte_len));
        }
        Ok(Self {
            byte_len,
            sha256,
            role,
        })
    }

    /// Exact expected byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Optional expected SHA-256.
    #[must_use]
    pub const fn sha256(&self) -> Option<[u8; 32]> {
        self.sha256
    }

    /// Sealed file role.
    #[must_use]
    pub const fn role(&self) -> GenerationFileRole {
        self.role
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    const fn max_byte_len(self) -> u64 {
        match self.role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_MAX_FILE_BYTES,
            GenerationFileRole::MutableControl => GENERATION_ROOT_MAX_CONTROL_FILE_BYTES,
        }
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    const fn expected_mode(self) -> u32 {
        match self.role {
            GenerationFileRole::ImmutableArtifact => GENERATION_ROOT_IMMUTABLE_FILE_MODE,
            GenerationFileRole::MutableControl => GENERATION_ROOT_CONTROL_FILE_MODE,
        }
    }
}

/// A bounded, strictly relative path under a qualified generation root.
#[derive(Clone)]
pub struct ConfinedGenerationPath {
    inner: platform::RelativePath,
}

impl ConfinedGenerationPath {
    /// Parse a non-empty relative path without dot, dot-dot, root, prefix,
    /// empty, oversized, or NUL-containing components.
    ///
    /// Unsupported targets return
    /// [`GenerationRootErrorKind::UnsupportedPlatform`] before filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns a bounded route or platform error.
    pub fn parse(path: &Path) -> GenerationRootResult<Self> {
        platform::parse_relative(path).map(|inner| Self { inner })
    }

    /// Number of relative components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        platform::relative_component_count(&self.inner)
    }
}

impl fmt::Debug for ConfinedGenerationPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfinedGenerationPath")
            .field("component_count", &self.component_count())
            .finish_non_exhaustive()
    }
}

/// Internal qualified directory capability used while atomically binding the
/// named LOCK and AUTHORITY anchors.
struct QualifiedGenerationDirectory {
    inner: platform::RootHandle,
}

impl QualifiedGenerationDirectory {
    /// Admit an existing absolute private generation-root route.
    ///
    /// The route is walked component-by-component without following links,
    /// then independently re-resolved before this function returns.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, security, mount,
    /// filesystem, ACL, kernel-feature, or identity uncertainty. Unsupported
    /// targets fail with [`GenerationRootErrorKind::UnsupportedPlatform`]
    /// before route parsing or filesystem I/O.
    fn admit(path: &Path) -> GenerationRootResult<Self> {
        platform::admit_root(path).map(|inner| Self { inner })
    }

    /// Root descriptor witness.
    #[must_use]
    fn witness(&self) -> GenerationRootObjectWitness {
        platform::root_witness(&self.inner)
    }

    /// Re-resolve the canonical route and require it to name the retained
    /// private root.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::ObjectChanged`] or the precise
    /// qualification failure when the route no longer resolves identically.
    #[cfg(all(test, target_os = "linux"))]
    fn revalidate_route(&self) -> GenerationRootResult<()> {
        platform::revalidate_root(&self.inner)
    }

    /// Admit one exact regular-file image below this root.
    ///
    /// The returned owner retains the qualified data descriptor and one immutable
    /// `Arc<[u8]>`; header validation and search consumers can therefore use
    /// the same allocation after route replacement. On Linux the data
    /// descriptor is derived from a retained, fully qualified `O_PATH`
    /// capability through the verified `/proc/thread-self/fd` bridge
    /// documented at module level; it is not opened from the generation route
    /// before the final object is proven regular.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, mount, type, owner,
    /// mode, link-count, length, content, or before/after identity mismatch.
    /// On Apple Silicon this path-only API always returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened_file`].
    #[cfg_attr(
        all(target_os = "macos", target_arch = "aarch64"),
        allow(clippy::unused_self)
    )]
    fn admit_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        if expectation.role != GenerationFileRole::ImmutableArtifact {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = path;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::OpenRegularFile,
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let admission = platform::admit_file(&self.inner, &path.inner, expectation)?;
            Ok(QualifiedGenerationFile {
                handle: admission.handle,
                bytes: admission.bytes,
                witness: admission.witness,
                sha256: admission.sha256,
            })
        }
    }

    /// Admit an exact Apple Silicon generation artifact from a descriptor
    /// already opened by a trusted external provider.
    ///
    /// The provider owns the proof that descriptor acquisition itself had no
    /// vnode/device side effects and attests that the descriptor is fresh,
    /// solely transferred, and has no retained duplicate or separately opened
    /// writable alias; POSIX descriptor inspection cannot verify those facts.
    /// This library first rejects non-regular descriptors and proves exact
    /// access status plus `FD_CLOEXEC`; it then qualifies the retained
    /// descriptor's mount, owner, permissions, link count, flags, ACL, and
    /// size before binding the canonical route and reading the exact content.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any descriptor, route, security,
    /// identity, content, or resource uncertainty.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn admit_preopened_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        if expectation.role != GenerationFileRole::ImmutableArtifact {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        let admission =
            platform::admit_preopened_file(&self.inner, &path.inner, descriptor, expectation)?;
        Ok(QualifiedGenerationFile {
            handle: admission.handle,
            bytes: admission.bytes,
            witness: admission.witness,
            sha256: admission.sha256,
        })
    }

    /// Apply the platform's strongest required directory durability barrier.
    ///
    /// Linux uses `fsync`; Apple Silicon uses `F_FULLFSYNC` and never
    /// downgrades to `fsync`.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::DurabilityUnavailable`] when the
    /// required primitive fails.
    fn sync_directory_durable(&self) -> GenerationRootResult<()> {
        platform::sync_root_directory(&self.inner)
    }

    /// Admit an existing fixed-size private control anchor as a persistent
    /// descriptor-owned capability.
    ///
    /// This foundation API deliberately exposes no mutation operation.
    /// Ambient content or metadata changes invalidate the capability. The
    /// publisher layer must add a lock-owned, witness-advancing write protocol
    /// for AUTHORITY/LOCK frames.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any route, mount, type, owner,
    /// mode, link-count, length, or identity mismatch. On Apple Silicon this
    /// path-only API always returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened_control_file`].
    #[cfg_attr(
        all(target_os = "macos", target_arch = "aarch64"),
        allow(clippy::unused_self)
    )]
    fn admit_control_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationControlFile> {
        if expectation.role != GenerationFileRole::MutableControl {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = path;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::OpenRegularFile,
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            platform::admit_control_file(&self.inner, &path.inner, expectation)
                .map(|inner| QualifiedGenerationControlFile { inner })
        }
    }

    /// Admit an Apple Silicon control anchor from a provider-owned descriptor.
    ///
    /// The provider owns no-side-effect acquisition provenance. The supplied
    /// descriptor must be a fresh read-write, close-on-exec open description;
    /// future lock acquisition requires another independently opened
    /// descriptor through [`QualifiedGenerationControlFile::try_lock_preopened`].
    /// Freshness and sole ownership are trusted provider attestations: POSIX
    /// descriptor inspection cannot detect an unseen `dup`.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed error for any descriptor, route, security,
    /// identity, content, or resource uncertainty.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn admit_preopened_control_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationControlFile> {
        if expectation.role != GenerationFileRole::MutableControl {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::WrongRole,
                GenerationRootStage::OpenRegularFile,
            ));
        }
        platform::admit_preopened_control_file(&self.inner, &path.inner, descriptor, expectation)
            .map(|inner| QualifiedGenerationControlFile { inner })
    }
}

/// Provider-owned Apple Silicon descriptors for the two fixed control
/// anchors.
///
/// The trusted provider must transfer fresh, solely owned, read-write,
/// close-on-exec descriptors for the exact top-level LOCK and AUTHORITY files.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub struct PreopenedGenerationRootAnchors {
    lock: std::os::fd::OwnedFd,
    authority: std::os::fd::OwnedFd,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl PreopenedGenerationRootAnchors {
    /// Bind provider-labelled LOCK and AUTHORITY descriptors without exposing
    /// an order-dependent tuple at the root-admission call site.
    #[must_use]
    pub fn new(lock: std::os::fd::OwnedFd, authority: std::os::fd::OwnedFd) -> Self {
        Self { lock, authority }
    }
}

/// Qualified generation root with retained exact-name LOCK and AUTHORITY
/// descriptors.
///
/// Construction returns only after the root and both anchors have passed a
/// combined anti-mix fence. Mutable control content is deliberately not frozen
/// into this long-lived capability: each shared or exclusive guard captures a
/// fresh exact owned image while holding the named LOCK flock.
pub struct QualifiedGenerationRoot {
    directory: QualifiedGenerationDirectory,
    lock: QualifiedGenerationControlFile,
    authority: QualifiedGenerationControlFile,
    layout: GenerationRootAnchorLayout,
}

impl QualifiedGenerationRoot {
    /// Admit an existing private root and its exact top-level LOCK and
    /// AUTHORITY anchors.
    ///
    /// Linux opens both anchors descriptor-relatively. Apple Silicon returns
    /// [`GenerationRootErrorKind::PreopenedDescriptorRequired`]; use
    /// [`Self::admit_preopened`]. Unsupported targets fail before path parsing
    /// or filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed platform, route, security, filesystem,
    /// anchor-identity, length, or content-admission error.
    pub fn admit(path: &Path, layout: GenerationRootAnchorLayout) -> GenerationRootResult<Self> {
        let directory = QualifiedGenerationDirectory::admit(path)?;
        let lock_path = ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_LOCK_FILE_NAME))?;
        let authority_path =
            ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_AUTHORITY_FILE_NAME))?;
        let lock = directory.admit_control_file(
            &lock_path,
            GenerationFileExpectation::control(layout.lock_byte_len())?,
        )?;
        let authority = directory.admit_control_file(
            &authority_path,
            GenerationFileExpectation::control(layout.authority_byte_len())?,
        )?;
        let admitted = Self {
            directory,
            lock,
            authority,
            layout,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }

    /// Admit Apple Silicon named anchors from trusted provider-owned
    /// descriptors.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor, route, security, filesystem,
    /// anchor-identity, length, or content-admission error.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn admit_preopened(
        path: &Path,
        layout: GenerationRootAnchorLayout,
        anchors: PreopenedGenerationRootAnchors,
    ) -> GenerationRootResult<Self> {
        let directory = QualifiedGenerationDirectory::admit(path)?;
        let lock_path = ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_LOCK_FILE_NAME))?;
        let authority_path =
            ConfinedGenerationPath::parse(Path::new(GENERATION_ROOT_AUTHORITY_FILE_NAME))?;
        let lock = directory.admit_preopened_control_file(
            &lock_path,
            anchors.lock,
            GenerationFileExpectation::control(layout.lock_byte_len())?,
        )?;
        let authority = directory.admit_preopened_control_file(
            &authority_path,
            anchors.authority,
            GenerationFileExpectation::control(layout.authority_byte_len())?,
        )?;
        let admitted = Self {
            directory,
            lock,
            authority,
            layout,
        };
        admitted.revalidate()?;
        Ok(admitted)
    }

    /// Root descriptor witness.
    #[must_use]
    pub fn witness(&self) -> GenerationRootObjectWitness {
        self.directory.witness()
    }

    /// Descriptor-derived identity tuple retained by this root capability.
    #[must_use]
    pub fn anchor_witnesses(&self) -> GenerationRootAnchorWitness {
        GenerationRootAnchorWitness::new(
            self.directory.witness(),
            self.lock.witness(),
            self.authority.witness(),
        )
    }

    /// Frozen physical named-anchor layout.
    #[must_use]
    pub const fn anchor_layout(&self) -> GenerationRootAnchorLayout {
        self.layout
    }

    /// Revalidate the canonical root plus both retained named anchors as one
    /// anti-mix tuple.
    ///
    /// Mutable bytes, mtime, and ctime may advance between guards; object,
    /// route, mount, security, link, and fixed-length identity may not.
    ///
    /// # Errors
    ///
    /// Returns the precise fail-closed route or identity error.
    pub fn revalidate(&self) -> GenerationRootResult<()> {
        platform::revalidate_anchor_set(
            &self.directory.inner,
            &self.lock.inner,
            &self.authority.inner,
        )
    }

    /// Admit one exact immutable regular-file image below this root.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as descriptor-owned file admission.
    pub fn admit_file(
        &self,
        path: &ConfinedGenerationPath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        self.directory.admit_file(path, expectation)
    }

    /// Admit one exact Apple Silicon immutable artifact from a provider-owned
    /// descriptor.
    ///
    /// # Errors
    ///
    /// Returns the same typed errors as preopened file admission.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn admit_preopened_file(
        &self,
        path: &ConfinedGenerationPath,
        descriptor: std::os::fd::OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<QualifiedGenerationFile> {
        self.directory
            .admit_preopened_file(path, descriptor, expectation)
    }

    /// Apply the platform's strongest required directory durability barrier.
    ///
    /// # Errors
    ///
    /// Returns a typed route, identity, or durability error.
    pub fn sync_directory_durable(&self) -> GenerationRootResult<()> {
        self.directory.sync_directory_durable()
    }

    /// Acquire a non-blocking shared guard on the exact named LOCK anchor and
    /// capture fresh stable LOCK and AUTHORITY images.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::LockContended`] instead of blocking,
    /// or the precise tuple/content failure. Apple Silicon requires
    /// [`Self::read_guard_preopened`].
    pub fn read_guard(&self) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let flock =
                platform::try_anchor_lock(&self.lock.inner, GenerationRootLockMode::Shared)?;
            self.finish_read_guard(flock)
        }
    }

    /// Acquire an Apple Silicon shared guard using a fresh provider-owned LOCK
    /// open description.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor, contention, tuple, or content
    /// error.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn read_guard_preopened(
        &self,
        fresh_lock_descriptor: std::os::fd::OwnedFd,
    ) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        let flock = platform::try_anchor_lock_preopened(
            &self.lock.inner,
            fresh_lock_descriptor,
            GenerationRootLockMode::Shared,
        )?;
        self.finish_read_guard(flock)
    }

    fn finish_read_guard(
        &self,
        flock: platform::LockHandle,
    ) -> GenerationRootResult<GenerationRootReadGuard<'_>> {
        let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
        Ok(GenerationRootReadGuard {
            root: self,
            flock,
            lock_image,
            authority_image,
        })
    }

    fn capture_stable_anchor_images(
        &self,
    ) -> GenerationRootResult<(platform::AnchorImage, platform::AnchorImage)> {
        self.revalidate()?;
        let first_lock = platform::capture_anchor_image(&self.lock.inner)?;
        let first_authority = platform::capture_anchor_image(&self.authority.inner)?;
        self.revalidate()?;
        let second_lock = platform::capture_anchor_image(&self.lock.inner)?;
        let second_authority = platform::capture_anchor_image(&self.authority.inner)?;
        self.revalidate()?;
        if !platform::anchor_images_match(&first_lock, &second_lock)
            || !platform::anchor_images_match(&first_authority, &second_authority)
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        Ok((second_lock, second_authority))
    }

    // This deliberately lands one bead before the writer-lease consumer.
    #[allow(dead_code)]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "aarch64"),
        allow(clippy::unused_self)
    )]
    pub(crate) fn try_exclusive_anchor_guard(
        &self,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let flock =
                platform::try_anchor_lock(&self.lock.inner, GenerationRootLockMode::Exclusive)?;
            let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
            Ok(GenerationRootExclusiveAnchorGuard {
                root: self,
                flock,
                lock_image,
                authority_image,
            })
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[allow(dead_code)]
    pub(crate) fn try_exclusive_anchor_guard_preopened(
        &self,
        fresh_lock_descriptor: std::os::fd::OwnedFd,
    ) -> GenerationRootResult<GenerationRootExclusiveAnchorGuard<'_>> {
        let flock = platform::try_anchor_lock_preopened(
            &self.lock.inner,
            fresh_lock_descriptor,
            GenerationRootLockMode::Exclusive,
        )?;
        let (lock_image, authority_image) = self.capture_stable_anchor_images()?;
        Ok(GenerationRootExclusiveAnchorGuard {
            root: self,
            flock,
            lock_image,
            authority_image,
        })
    }
}

impl fmt::Debug for QualifiedGenerationRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationRoot")
            .field("anchors", &self.anchor_witnesses())
            .finish_non_exhaustive()
    }
}

/// Shared named-anchor guard carrying fresh exact control images.
///
/// The guard cannot outlive the qualified root that owns the retained anchor
/// descriptors:
///
/// ```compile_fail
/// use frankensearch_index::generation_root::{
///     GenerationRootReadGuard, QualifiedGenerationRoot,
/// };
///
/// fn impossible(root: &QualifiedGenerationRoot) -> GenerationRootReadGuard<'static> {
///     root.read_guard().expect("type-checking must reject this lifetime")
/// }
/// ```
#[must_use = "dropping the guard releases the shared generation-root lock"]
pub struct GenerationRootReadGuard<'root> {
    root: &'root QualifiedGenerationRoot,
    flock: platform::LockHandle,
    lock_image: platform::AnchorImage,
    authority_image: platform::AnchorImage,
}

impl GenerationRootReadGuard<'_> {
    /// Root and fresh LOCK/AUTHORITY witnesses captured under the shared lock.
    #[must_use]
    pub fn witnesses(&self) -> GenerationRootAnchorWitness {
        GenerationRootAnchorWitness::new(
            self.root.witness(),
            self.lock_image.witness,
            self.authority_image.witness,
        )
    }

    /// Borrow the exact named LOCK image.
    #[must_use]
    pub fn lock_bytes(&self) -> &[u8] {
        &self.lock_image.bytes
    }

    /// Clone the exact named LOCK allocation.
    #[must_use]
    pub fn lock_bytes_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.lock_image.bytes)
    }

    /// SHA-256 of the exact named LOCK image.
    #[must_use]
    pub const fn lock_sha256(&self) -> [u8; 32] {
        self.lock_image.sha256
    }

    /// Borrow the exact named AUTHORITY image.
    #[must_use]
    pub fn authority_bytes(&self) -> &[u8] {
        &self.authority_image.bytes
    }

    /// Clone the exact named AUTHORITY allocation.
    #[must_use]
    pub fn authority_bytes_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.authority_image.bytes)
    }

    /// SHA-256 of the exact named AUTHORITY image.
    #[must_use]
    pub const fn authority_sha256(&self) -> [u8; 32] {
        self.authority_image.sha256
    }

    /// Validate both captured images and explicitly release the shared flock.
    ///
    /// Unlock is attempted even when validation detects ambient mutation.
    ///
    /// # Errors
    ///
    /// Returns a typed content, route, process, or unlock error.
    pub fn release(self) -> GenerationRootResult<()> {
        let validation = platform::validate_anchor_image(&self.root.lock.inner, &self.lock_image)
            .and_then(|()| {
                platform::validate_anchor_image(&self.root.authority.inner, &self.authority_image)
            });
        let unlock = platform::unlock(self.flock);
        validation.and(unlock)
    }
}

impl fmt::Debug for GenerationRootReadGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootReadGuard")
            .field("witnesses", &self.witnesses())
            .finish_non_exhaustive()
    }
}

/// Crate-private kernel primitive consumed by the later shared-index writer
/// lease. This is not itself a lease and exposes no mutation operation.
#[must_use = "dropping the guard releases the exclusive generation-root lock"]
#[allow(dead_code)]
pub(crate) struct GenerationRootExclusiveAnchorGuard<'root> {
    root: &'root QualifiedGenerationRoot,
    flock: platform::LockHandle,
    lock_image: platform::AnchorImage,
    authority_image: platform::AnchorImage,
}

impl GenerationRootExclusiveAnchorGuard<'_> {
    #[allow(dead_code)]
    pub(crate) fn release(self) -> GenerationRootResult<()> {
        platform::unlock(self.flock)
    }
}

impl fmt::Debug for GenerationRootExclusiveAnchorGuard<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootExclusiveAnchorGuard")
            .field("root", &self.root.anchor_witnesses())
            .field("lock_sha256", &self.lock_image.sha256)
            .field("authority_sha256", &self.authority_image.sha256)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for QualifiedGenerationDirectory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationDirectory")
            .field("witness", &self.witness())
            .finish_non_exhaustive()
    }
}

/// Retained descriptor plus the exact bytes admitted from it.
pub struct QualifiedGenerationFile {
    handle: platform::FileHandle,
    bytes: Arc<[u8]>,
    witness: GenerationRootObjectWitness,
    sha256: [u8; 32],
}

impl QualifiedGenerationFile {
    /// Borrow the exact owned byte image.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Clone the exact owned byte allocation for a descriptor-independent
    /// consumer such as `ValidatedFsviBytes::from_arc`.
    #[must_use]
    pub fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Descriptor witness captured after complete admission.
    #[must_use]
    pub const fn witness(&self) -> GenerationRootObjectWitness {
        self.witness
    }

    /// SHA-256 of the exact owned byte image.
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Apply the platform's strongest required regular-file durability
    /// barrier to the retained descriptor.
    ///
    /// The descriptor is read and SHA-256 checked against the admitted exact
    /// image both before and after the barrier. This explicit operation
    /// therefore performs two full-file reads.
    ///
    /// # Errors
    ///
    /// Returns a typed route, identity, or durability error. A required
    /// durability primitive never falls back to a weaker barrier.
    pub fn sync_durable(&self) -> GenerationRootResult<()> {
        platform::sync_file(&self.handle, self.witness)
    }
}

impl fmt::Debug for QualifiedGenerationFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationFile")
            .field("witness", &self.witness)
            .finish_non_exhaustive()
    }
}

/// Internal qualified capability for one mutable private control anchor.
struct QualifiedGenerationControlFile {
    inner: platform::ControlHandle,
}

impl QualifiedGenerationControlFile {
    /// Object/metadata witness captured when this control file was admitted.
    ///
    /// This does not attest mutable content; the fixed named-anchor guards
    /// capture a fresh exact image while holding the kernel flock.
    #[must_use]
    fn witness(&self) -> GenerationRootObjectWitness {
        platform::control_witness(&self.inner)
    }

    /// Clone the exact descriptor-admitted control-file image.
    #[must_use]
    #[cfg(test)]
    fn bytes(&self) -> Arc<[u8]> {
        platform::control_bytes(&self.inner)
    }

    /// SHA-256 of the exact descriptor-admitted control-file image.
    #[must_use]
    #[cfg(test)]
    fn sha256(&self) -> [u8; 32] {
        platform::control_sha256(&self.inner)
    }

    /// Attempt a shared or exclusive non-blocking kernel lock on this exact
    /// qualified control-file identity.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationRootErrorKind::LockContended`] rather than
    /// blocking. Route replacement, process-identity drift, and all normal
    /// file-admission errors remain fail-closed.
    #[cfg(test)]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "aarch64"),
        allow(clippy::unused_self)
    )]
    fn try_lock(&self, mode: GenerationRootLockMode) -> GenerationRootResult<GenerationRootLock> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let _ = mode;
            Err(GenerationRootError::new(
                GenerationRootErrorKind::PreopenedDescriptorRequired,
                GenerationRootStage::AcquireLock,
            ))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            platform::try_lock(&self.inner, mode).map(|inner| GenerationRootLock { inner })
        }
    }

    /// Attempt an Apple Silicon lock using a fresh provider-owned open
    /// description for the exact admitted control object.
    ///
    /// A duplicated descriptor is not sufficient: `flock` state is attached
    /// to the open description. The external provider owns acquisition
    /// provenance; this method verifies descriptor type, access, route, and
    /// security before the non-blocking flock, then verifies the candidate's
    /// bound content only after flock succeeds. Exact Darwin status validation
    /// also rejects the persistent `FWASLOCKED` history bit left by a prior
    /// successful flock, but that defense does not prove the absence of unseen
    /// aliases. The provider must transfer the sole live reference and retain
    /// no duplicate. A surviving alias could convert or unlock the flock while
    /// this guard is live; creator-process drop and explicit unlock
    /// nevertheless issue `LOCK_UN` so an inherited alias cannot prolong the
    /// guard after release.
    ///
    /// # Errors
    ///
    /// Returns a typed fail-closed descriptor/admission error or
    /// [`GenerationRootErrorKind::LockContended`].
    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    fn try_lock_preopened(
        &self,
        descriptor: std::os::fd::OwnedFd,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<GenerationRootLock> {
        platform::try_lock_preopened(&self.inner, descriptor, mode)
            .map(|inner| GenerationRootLock { inner })
    }
}

impl fmt::Debug for QualifiedGenerationControlFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedGenerationControlFile")
            .field("witness", &self.witness())
            .finish_non_exhaustive()
    }
}

/// Internal requested kernel lock mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GenerationRootLockMode {
    /// Reader/shared lock.
    Shared,
    /// Writer/exclusive lock.
    #[allow(dead_code)]
    Exclusive,
}

/// Owned non-blocking kernel lock guard.
///
/// In the creator process, dropping the guard issues a best-effort explicit
/// `LOCK_UN` before closing the descriptor. This releases the open-description
/// lock even when a forked child still retains an inherited descriptor.
/// A child-side drop never unlocks the creator's guard. [`Self::unlock`]
/// reports explicit validation or unlock failures.
///
/// Applications must not continue normal multithreaded library execution
/// across `fork()`: the child may only close inherited capabilities or `exec`.
/// Operations that revalidate, synchronize, or unlock in a child fail with
/// [`GenerationRootErrorKind::ForkedProcess`].
#[must_use = "holding this guard keeps the kernel lock acquired"]
#[cfg(test)]
struct GenerationRootLock {
    inner: platform::LockHandle,
}

#[cfg(test)]
impl GenerationRootLock {
    /// Object/metadata witness bound to this lock.
    ///
    /// This does not attest content. Use [`Self::sha256`] or [`Self::bytes`]
    /// for the exact locked image.
    #[must_use]
    fn witness(&self) -> GenerationRootObjectWitness {
        platform::lock_witness(&self.inner)
    }

    /// Clone the exact control-file image validated while the lock was held.
    #[must_use]
    fn bytes(&self) -> Arc<[u8]> {
        platform::lock_bytes(&self.inner)
    }

    /// SHA-256 of the exact control-file image validated while the lock was
    /// held.
    #[must_use]
    fn sha256(&self) -> [u8; 32] {
        platform::lock_sha256(&self.inner)
    }

    /// Lock mode bound to this guard.
    #[must_use]
    fn mode(&self) -> GenerationRootLockMode {
        platform::lock_mode(&self.inner)
    }

    /// Explicitly release this lock.
    ///
    /// # Errors
    ///
    /// Returns a typed process, content, route, or OS error. Explicit
    /// creator-process `LOCK_UN` is the primary release boundary; creator-side
    /// drop retries it best-effort after a reported failure. Closing the owned
    /// descriptor is only a fallback when no inherited or duplicated
    /// open-description reference survives.
    fn unlock(self) -> GenerationRootResult<()> {
        platform::unlock(self.inner)
    }

    /// Apply the required file durability barrier to the locked control file.
    ///
    /// This synchronizes the unchanged admitted image. It neither authorizes
    /// nor absorbs an ambient mutation.
    ///
    /// # Errors
    ///
    /// Returns a typed process, content, route, identity, or durability error.
    /// A required durability primitive never falls back to a weaker barrier.
    fn sync_durable(&self) -> GenerationRootResult<()> {
        platform::sync_lock_file(&self.inner)
    }
}

#[cfg(test)]
impl fmt::Debug for GenerationRootLock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GenerationRootLock")
            .field("held", &true)
            .finish_non_exhaustive()
    }
}

#[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
mod platform {
    #[cfg(target_os = "linux")]
    use super::GENERATION_ROOT_MAX_MOUNTINFO_BYTES;
    use super::{
        GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_DIRECTORY_MODE,
        GENERATION_ROOT_LOCK_FILE_NAME, GENERATION_ROOT_MAX_COMPONENT_BYTES,
        GENERATION_ROOT_MAX_COMPONENTS, GENERATION_ROOT_MAX_DIRECTORY_ENTRIES,
        GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES, GENERATION_ROOT_MAX_ROUTE_BYTES,
        GenerationFileExpectation, GenerationRootError, GenerationRootErrorKind,
        GenerationRootLockMode, GenerationRootObjectWitness, GenerationRootResult,
        GenerationRootStage, QualifiedFilesystem,
    };
    use rustix::fd::{AsFd, OwnedFd};
    use rustix::fs::{FileType, Mode, OFlags, fstat, fstatfs, fstatvfs};
    use rustix::io::{Errno, pread};
    use sha2::{Digest, Sha256};
    use std::ffi::{OsStr, OsString};
    use std::fmt;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::sync::Arc;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use rustix::fs::openat;
    #[cfg(target_os = "linux")]
    use std::os::fd::{AsRawFd, RawFd};

    const MODE_PERMISSIONS_MASK: u32 = 0o7777;
    #[cfg(target_os = "linux")]
    const LINUX_STATX_SUBVOLUME_MASK: u32 = 0x0000_8000;

    #[cfg(target_os = "linux")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum ProcSelfBindingPhase {
        RootQualification,
        MountInfoRead,
        BeforeDataOpen,
        AfterDataOpen,
    }

    #[cfg(target_os = "linux")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum ProcBridgeProbePhase {
        BeforeDataOpen,
        AfterDataOpen,
    }

    #[cfg(test)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum TestBoundary {
        AfterFirstRootQualification,
        BeforeAbsoluteAncestorAclRead {
            index: Option<usize>,
        },
        AfterAbsoluteAncestorAclRead {
            index: Option<usize>,
        },
        BeforeRootComponentOpen {
            index: usize,
        },
        AfterRootComponentOpen {
            index: usize,
        },
        BeforeRelativeDirectoryOpen {
            index: usize,
        },
        AfterRelativeDirectoryOpen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        BeforeRegularFileOpen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        AfterRegularFileOpen {
            index: usize,
            cloexec: bool,
        },
        BeforeRootDescriptorDuplicate,
        AfterRootDescriptorDuplicate,
        #[cfg(target_os = "linux")]
        AfterQualifiedFileOpen,
        BeforeRead {
            offset: u64,
        },
        AfterRead {
            offset: u64,
            byte_count: usize,
        },
        BeforeTrailingByteProbe,
        AfterTrailingByteProbe {
            byte_count: usize,
        },
        AfterExactRead,
        #[cfg(target_os = "linux")]
        BeforeFileRouteReopen,
        #[cfg(target_os = "linux")]
        AfterFileRouteReopen,
        #[cfg(target_os = "linux")]
        BeforeFinalRootRevalidation,
        BeforeFilesystemQualification,
        AfterFilesystemQualification,
        BeforeRouteMountIdentity,
        AfterRouteMountIdentity,
        BeforeObjectWitness,
        AfterObjectWitness,
        #[cfg(target_os = "linux")]
        BeforeMountInfoOpen,
        #[cfg(target_os = "linux")]
        AfterMountInfoOpen,
        #[cfg(target_os = "linux")]
        BeforeMountInfoRead,
        #[cfg(target_os = "linux")]
        AfterMountInfoRead {
            byte_count: usize,
        },
        #[cfg(target_os = "linux")]
        BeforeFinalProbeOpen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        AfterFinalProbeOpen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        AfterProbeQualified {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        BeforeProcCapabilityRootOpen,
        #[cfg(target_os = "linux")]
        AfterProcCapabilityRootOpen,
        #[cfg(target_os = "linux")]
        BeforeProcSelfReadlink {
            phase: ProcSelfBindingPhase,
        },
        #[cfg(target_os = "linux")]
        AfterProcSelfReadlink {
            phase: ProcSelfBindingPhase,
        },
        #[cfg(target_os = "linux")]
        BeforeProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase,
        },
        #[cfg(target_os = "linux")]
        AfterProcThreadSelfReadlink {
            phase: ProcSelfBindingPhase,
        },
        #[cfg(target_os = "linux")]
        BeforeProcBridgeProbe {
            index: usize,
            phase: ProcBridgeProbePhase,
        },
        #[cfg(target_os = "linux")]
        AfterProcBridgeProbe {
            index: usize,
            phase: ProcBridgeProbePhase,
        },
        #[cfg(target_os = "linux")]
        BeforeProcFdReopen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        AfterProcFdReopen {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        BeforeFinalRouteProbe {
            index: usize,
        },
        #[cfg(target_os = "linux")]
        AfterFinalRouteProbe {
            index: usize,
        },
        BeforeLock,
        AfterLock,
        BeforeUnlock,
        AfterUnlock,
        BeforeFileSync,
        AfterFileSync,
        BeforeDirectorySync,
        AfterDirectorySync,
        BeforeExactNameEnumeration,
        AfterExactNameEnumeration,
        BeforeAclRead,
        AfterAclRead,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        BeforePreopenedQualification,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        AfterPreopenedQualification,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        BeforeFinalRouteStat,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        AfterFinalRouteStat,
    }

    #[cfg(test)]
    type TestHook = Box<dyn FnMut(TestBoundary) -> GenerationRootResult<()>>;

    #[cfg(test)]
    std::thread_local! {
        static TEST_HOOK: std::cell::RefCell<Option<TestHook>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    pub(super) struct TestHookGuard;

    #[cfg(test)]
    impl Drop for TestHookGuard {
        fn drop(&mut self) {
            TEST_HOOK.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }

    #[cfg(test)]
    pub(super) fn install_test_hook(
        hook: impl FnMut(TestBoundary) -> GenerationRootResult<()> + 'static,
    ) -> TestHookGuard {
        TEST_HOOK.with(|slot| {
            let replaced = slot.borrow_mut().replace(Box::new(hook));
            assert!(
                replaced.is_none(),
                "only one test hook may be active per thread"
            );
        });
        TestHookGuard
    }

    #[cfg(test)]
    fn test_boundary(boundary: TestBoundary) -> GenerationRootResult<()> {
        TEST_HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            slot.as_mut().map_or(Ok(()), |hook| hook(boundary))
        })
    }

    #[derive(Clone)]
    pub(super) struct RelativePath {
        components: Arc<[OsString]>,
    }

    #[derive(Clone)]
    pub(super) struct RootHandle {
        state: Arc<RootState>,
    }

    struct RootState {
        route: AbsoluteRoute,
        descriptor: OwnedFd,
        witness: GenerationRootObjectWitness,
        route_identities: Arc<[BasicObjectIdentity]>,
        creator_process_id: u32,
    }

    pub(super) struct FileHandle {
        descriptor: OwnedFd,
        root: RootHandle,
        path: RelativePath,
        expectation: GenerationFileExpectation,
        content_witness: ExactContentWitness,
        component_identities: Arc<[BasicObjectIdentity]>,
    }

    pub(super) struct FileAdmission {
        pub(super) handle: FileHandle,
        pub(super) bytes: Arc<[u8]>,
        pub(super) witness: GenerationRootObjectWitness,
        pub(super) sha256: [u8; 32],
    }

    pub(super) struct ControlHandle {
        descriptor: OwnedFd,
        #[cfg(test)]
        bytes: Arc<[u8]>,
        root: RootHandle,
        path: RelativePath,
        expectation: GenerationFileExpectation,
        witness: GenerationRootObjectWitness,
        #[cfg(test)]
        content_witness: ExactContentWitness,
        component_identities: Arc<[BasicObjectIdentity]>,
    }

    pub(super) struct AnchorImage {
        pub(super) bytes: Arc<[u8]>,
        pub(super) witness: GenerationRootObjectWitness,
        pub(super) sha256: [u8; 32],
    }

    pub(super) struct LockHandle {
        descriptor: OwnedFd,
        #[cfg(test)]
        bytes: Arc<[u8]>,
        witness: GenerationRootObjectWitness,
        content_witness: ExactContentWitness,
        root: RootHandle,
        path: RelativePath,
        expectation: GenerationFileExpectation,
        component_identities: Arc<[BasicObjectIdentity]>,
        creator_process_id: u32,
        lock_held: bool,
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        descriptor_status_flags: OFlags,
        #[cfg(test)]
        mode: GenerationRootLockMode,
    }

    struct AcquiredFlockDescriptor {
        descriptor: Option<OwnedFd>,
        creator_process_id: u32,
    }

    impl AcquiredFlockDescriptor {
        fn new(descriptor: OwnedFd, creator_process_id: u32) -> Self {
            Self {
                descriptor: Some(descriptor),
                creator_process_id,
            }
        }

        fn descriptor(&self) -> &OwnedFd {
            let Some(descriptor) = self.descriptor.as_ref() else {
                unreachable!("an armed flock descriptor always owns its descriptor")
            };
            descriptor
        }

        fn into_descriptor(mut self) -> OwnedFd {
            let Some(descriptor) = self.descriptor.take() else {
                unreachable!("an armed flock descriptor transfers exactly once")
            };
            descriptor
        }
    }

    impl Drop for AcquiredFlockDescriptor {
        fn drop(&mut self) {
            if self.creator_process_id == std::process::id()
                && let Some(descriptor) = self.descriptor.as_ref()
            {
                let _ = rustix::fs::flock(descriptor, rustix::fs::FlockOperation::Unlock);
            }
        }
    }

    #[derive(Clone)]
    struct AbsoluteRoute {
        components: Arc<[OsString]>,
    }

    struct OpenedRoute {
        descriptor: OwnedFd,
        route_identities: Vec<BasicObjectIdentity>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BasicObjectIdentity {
        device: u64,
        inode: u64,
        mode: u32,
        uid: u32,
        gid: u32,
        mount_identity: [u8; 32],
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct AbsoluteAncestorSecurityWitness {
        identity: BasicObjectIdentity,
        changed_seconds: i64,
        changed_nanoseconds: i64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExactContentWitness {
        sha256: [u8; 32],
    }

    impl fmt::Debug for RootHandle {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("RootHandle")
                .field("witness", &self.state.witness)
                .finish_non_exhaustive()
        }
    }

    pub(super) fn parse_relative(path: &Path) -> GenerationRootResult<RelativePath> {
        let components = parse_route_bytes(
            path.as_os_str(),
            false,
            GenerationRootStage::ParseRelativeRoute,
        )?;
        Ok(RelativePath {
            components: Arc::from(components),
        })
    }

    pub(super) fn relative_component_count(path: &RelativePath) -> usize {
        path.components.len()
    }

    pub(super) fn admit_root(path: &Path) -> GenerationRootResult<RootHandle> {
        let creator_process_id = std::process::id();
        let route = AbsoluteRoute {
            components: Arc::from(parse_route_bytes(
                path.as_os_str(),
                true,
                GenerationRootStage::ParseRootRoute,
            )?),
        };
        let first = open_absolute_route(&route, GenerationRootStage::OpenRootRoute)?;
        let (filesystem, mount_identity) =
            qualify_filesystem(&first.descriptor, GenerationRootStage::QualifyFilesystem)?;
        let first_witness = object_witness(
            &first.descriptor,
            filesystem,
            mount_identity,
            GenerationRootStage::QualifyRootSecurity,
        )?;
        validate_private_directory(
            &first.descriptor,
            first_witness,
            GenerationRootStage::QualifyRootSecurity,
        )?;
        validate_root_writable(
            &first.descriptor,
            first_witness.filesystem,
            GenerationRootStage::QualifyFilesystem,
        )?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFirstRootQualification)?;

        let second = open_absolute_route(&route, GenerationRootStage::RevalidateRootRoute)?;
        if first.route_identities != second.route_identities {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRootRoute,
            ));
        }
        let (second_filesystem, second_mount_identity) =
            qualify_filesystem(&second.descriptor, GenerationRootStage::RevalidateRootRoute)?;
        let second_witness = object_witness(
            &second.descriptor,
            second_filesystem,
            second_mount_identity,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        validate_private_directory(
            &second.descriptor,
            second_witness,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        validate_root_writable(
            &second.descriptor,
            second_witness.filesystem,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        if !first_witness.same_directory_security_identity(second_witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRootRoute,
            ));
        }
        validate_process_identity(creator_process_id, GenerationRootStage::RevalidateRootRoute)?;

        Ok(RootHandle {
            state: Arc::new(RootState {
                route,
                descriptor: first.descriptor,
                witness: first_witness,
                route_identities: Arc::from(first.route_identities),
                creator_process_id,
            }),
        })
    }

    pub(super) fn root_witness(root: &RootHandle) -> GenerationRootObjectWitness {
        root.state.witness
    }

    pub(super) fn revalidate_root(root: &RootHandle) -> GenerationRootResult<()> {
        validate_process_identity(
            root.state.creator_process_id,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        let retained = object_witness(
            &root.state.descriptor,
            root.state.witness.filesystem,
            root.state.witness.mount_identity,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        if !retained.same_directory_security_identity(root.state.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRootRoute,
            ));
        }
        validate_private_directory(
            &root.state.descriptor,
            retained,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        validate_root_writable(
            &root.state.descriptor,
            retained.filesystem,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        let reopened =
            open_absolute_route(&root.state.route, GenerationRootStage::RevalidateRootRoute)?;
        if reopened.route_identities.as_slice() != root.state.route_identities.as_ref() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRootRoute,
            ));
        }
        let (filesystem, mount_identity) = qualify_filesystem(
            &reopened.descriptor,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        let reopened_witness = object_witness(
            &reopened.descriptor,
            filesystem,
            mount_identity,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        validate_private_directory(
            &reopened.descriptor,
            reopened_witness,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        validate_root_writable(
            &reopened.descriptor,
            reopened_witness.filesystem,
            GenerationRootStage::RevalidateRootRoute,
        )?;
        if !retained.same_directory_security_identity(reopened_witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRootRoute,
            ));
        }
        validate_process_identity(
            root.state.creator_process_id,
            GenerationRootStage::RevalidateRootRoute,
        )
    }

    #[cfg(target_os = "linux")]
    pub(super) fn admit_file(
        root: &RootHandle,
        path: &RelativePath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<FileAdmission> {
        revalidate_root(root)?;
        let opened = open_relative_file(
            root,
            path,
            expectation,
            false,
            GenerationRootStage::OpenRelativeDirectory,
            GenerationRootStage::OpenRegularFile,
        )?;
        let (bytes, after_read) =
            read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        if let Some(expected_sha256) = expectation.sha256 {
            if sha256 != expected_sha256 {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::HashMismatch,
                    GenerationRootStage::ReadRegularFile,
                ));
            }
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFileRouteReopen)?;
        let reopened = open_relative_probe(
            root,
            path,
            expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFileRouteReopen)?;
        if !after_read.mutation_stable_eq(reopened.witness)
            || opened.component_identities != reopened.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFinalRootRevalidation)?;
        revalidate_root(root)?;
        Ok(FileAdmission {
            handle: FileHandle {
                descriptor: opened.descriptor,
                root: root.clone(),
                path: path.clone(),
                expectation,
                content_witness: ExactContentWitness { sha256 },
                component_identities: opened.component_identities,
            },
            bytes: Arc::from(bytes),
            witness: after_read,
            sha256,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn admit_preopened_file(
        root: &RootHandle,
        path: &RelativePath,
        descriptor: OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<FileAdmission> {
        validate_process_identity(
            root.state.creator_process_id,
            GenerationRootStage::OpenRegularFile,
        )?;
        validate_preopened_regular_descriptor(
            &descriptor,
            PreopenedAccess::ReadOnly,
            GenerationRootStage::OpenRegularFile,
        )?;
        revalidate_root(root)?;
        let opened = qualify_preopened_file(
            root,
            path,
            descriptor,
            expectation,
            PreopenedAccess::ReadOnly,
            GenerationRootStage::OpenRegularFile,
        )?;
        let (bytes, after_read) =
            read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        if expectation
            .sha256
            .is_some_and(|expected_sha256| sha256 != expected_sha256)
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::HashMismatch,
                GenerationRootStage::ReadRegularFile,
            ));
        }
        revalidate_preopened_file_route(
            root,
            path,
            expectation,
            &opened.descriptor,
            after_read,
            &opened.component_identities,
            OFlags::RDONLY,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        revalidate_root(root)?;
        Ok(FileAdmission {
            handle: FileHandle {
                descriptor: opened.descriptor,
                root: root.clone(),
                path: path.clone(),
                expectation,
                content_witness: ExactContentWitness { sha256 },
                component_identities: opened.component_identities,
            },
            bytes: Arc::from(bytes),
            witness: after_read,
            sha256,
        })
    }

    pub(super) fn sync_root_directory(root: &RootHandle) -> GenerationRootResult<()> {
        revalidate_root(root)?;
        sync_directory_descriptor(&root.state.descriptor)?;
        revalidate_root(root)
    }

    pub(super) fn sync_file(
        file: &FileHandle,
        expected: GenerationRootObjectWitness,
    ) -> GenerationRootResult<()> {
        revalidate_file_route(file, expected)?;
        let (before_content, before) = read_exact_sha256(
            &file.descriptor,
            expected,
            file.expectation,
            GenerationRootStage::SyncRegularFile,
        )?;
        validate_bound_content(
            before_content,
            file.content_witness,
            GenerationRootStage::SyncRegularFile,
        )?;
        sync_file_descriptor(&file.descriptor)?;
        let (after_content, _) = read_exact_sha256(
            &file.descriptor,
            before,
            file.expectation,
            GenerationRootStage::SyncRegularFile,
        )?;
        validate_bound_content(
            after_content,
            file.content_witness,
            GenerationRootStage::SyncRegularFile,
        )?;
        revalidate_file_route(file, expected)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn admit_control_file(
        root: &RootHandle,
        path: &RelativePath,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<ControlHandle> {
        revalidate_root(root)?;
        let opened = open_relative_file(
            root,
            path,
            expectation,
            true,
            GenerationRootStage::OpenRelativeDirectory,
            GenerationRootStage::OpenRegularFile,
        )?;
        let (bytes, after_read) =
            read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
        let content_witness = ExactContentWitness {
            sha256: Sha256::digest(&bytes).into(),
        };
        let reopened = open_relative_probe(
            root,
            path,
            expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !after_read.mutation_stable_eq(reopened.witness)
            || opened.component_identities != reopened.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let (revalidated_content, revalidated_after) = read_exact_sha256(
            &opened.descriptor,
            after_read,
            expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        validate_bound_content(
            revalidated_content,
            content_witness,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        revalidate_root(root)?;
        Ok(ControlHandle {
            descriptor: opened.descriptor,
            #[cfg(test)]
            bytes: Arc::from(bytes),
            root: root.clone(),
            path: path.clone(),
            expectation,
            witness: revalidated_after,
            #[cfg(test)]
            content_witness: revalidated_content,
            component_identities: opened.component_identities,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn admit_preopened_control_file(
        root: &RootHandle,
        path: &RelativePath,
        descriptor: OwnedFd,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<ControlHandle> {
        validate_process_identity(
            root.state.creator_process_id,
            GenerationRootStage::OpenRegularFile,
        )?;
        validate_preopened_regular_descriptor(
            &descriptor,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::OpenRegularFile,
        )?;
        revalidate_root(root)?;
        let opened = qualify_preopened_file(
            root,
            path,
            descriptor,
            expectation,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::OpenRegularFile,
        )?;
        let (bytes, after_read) =
            read_exact_owned(&opened.descriptor, opened.witness, expectation)?;
        let content_witness = ExactContentWitness {
            sha256: Sha256::digest(&bytes).into(),
        };
        revalidate_preopened_file_route(
            root,
            path,
            expectation,
            &opened.descriptor,
            after_read,
            &opened.component_identities,
            OFlags::RDWR,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        let (revalidated_content, revalidated_after) = read_exact_sha256(
            &opened.descriptor,
            after_read,
            expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        validate_bound_content(
            revalidated_content,
            content_witness,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        revalidate_root(root)?;
        Ok(ControlHandle {
            descriptor: opened.descriptor,
            #[cfg(test)]
            bytes: Arc::from(bytes),
            root: root.clone(),
            path: path.clone(),
            expectation,
            witness: revalidated_after,
            #[cfg(test)]
            content_witness: revalidated_content,
            component_identities: opened.component_identities,
        })
    }

    pub(super) fn control_witness(control: &ControlHandle) -> GenerationRootObjectWitness {
        control.witness
    }

    #[cfg(test)]
    pub(super) fn control_bytes(control: &ControlHandle) -> Arc<[u8]> {
        Arc::clone(&control.bytes)
    }

    #[cfg(test)]
    pub(super) fn control_sha256(control: &ControlHandle) -> [u8; 32] {
        control.content_witness.sha256
    }

    pub(super) fn revalidate_anchor_set(
        root: &RootHandle,
        lock: &ControlHandle,
        authority: &ControlHandle,
    ) -> GenerationRootResult<()> {
        revalidate_root(root)?;
        if !Arc::ptr_eq(&root.state, &lock.root.state)
            || !Arc::ptr_eq(&root.state, &authority.root.state)
            || !relative_path_is_exact_name(&lock.path, GENERATION_ROOT_LOCK_FILE_NAME)
            || !relative_path_is_exact_name(&authority.path, GENERATION_ROOT_AUTHORITY_FILE_NAME)
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_anchor_route(lock)?;
        revalidate_anchor_route(authority)?;
        let root_witness = root.state.witness;
        if lock.witness.same_object(authority.witness)
            || lock.witness.device != root_witness.device
            || authority.witness.device != root_witness.device
            || lock.witness.mount_identity != root_witness.mount_identity
            || authority.witness.mount_identity != root_witness.mount_identity
            || lock.witness.filesystem != root_witness.filesystem
            || authority.witness.filesystem != root_witness.filesystem
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_anchor_route(lock)?;
        revalidate_anchor_route(authority)?;
        revalidate_root(root)
    }

    fn relative_path_is_exact_name(path: &RelativePath, expected: &str) -> bool {
        path.components.len() == 1 && path.components[0].as_os_str() == OsStr::new(expected)
    }

    pub(super) fn capture_anchor_image(
        control: &ControlHandle,
    ) -> GenerationRootResult<AnchorImage> {
        revalidate_anchor_route(control)?;
        let before = object_witness(
            &control.descriptor,
            control.witness.filesystem,
            control.witness.mount_identity,
            GenerationRootStage::AcquireLock,
        )?;
        if !before.same_control_anchor_identity(control.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        let (bytes, witness) = read_exact_owned(&control.descriptor, before, control.expectation)?;
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        if control
            .expectation
            .sha256
            .is_some_and(|expected| expected != sha256)
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::HashMismatch,
                GenerationRootStage::AcquireLock,
            ));
        }
        revalidate_anchor_route(control)?;
        Ok(AnchorImage {
            bytes: Arc::from(bytes),
            witness,
            sha256,
        })
    }

    pub(super) fn anchor_images_match(left: &AnchorImage, right: &AnchorImage) -> bool {
        left.sha256 == right.sha256
            && left.bytes == right.bytes
            && left.witness.mutation_stable_eq(right.witness)
    }

    pub(super) fn validate_anchor_image(
        control: &ControlHandle,
        expected: &AnchorImage,
    ) -> GenerationRootResult<()> {
        let observed = capture_anchor_image(control)?;
        if anchor_images_match(&observed, expected) {
            Ok(())
        } else {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::ReleaseLock,
            ))
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) fn try_anchor_lock(
        control: &ControlHandle,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        let creator_process_id = std::process::id();
        revalidate_anchor_route(control)?;
        let opened = open_relative_file(
            &control.root,
            &control.path,
            control.expectation,
            true,
            GenerationRootStage::AcquireLock,
            GenerationRootStage::AcquireLock,
        )?;
        if !opened.witness.same_control_anchor_identity(control.witness)
            || opened.component_identities != control.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        let operation = match mode {
            GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
            GenerationRootLockMode::Exclusive => {
                rustix::fs::FlockOperation::NonBlockingLockExclusive
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeLock)?;
        let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
            Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
            Err(error) if error == Errno::WOULDBLOCK => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::LockContended,
                    GenerationRootStage::AcquireLock,
                ));
            }
            Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterLock)?;
        validate_process_identity(creator_process_id, GenerationRootStage::AcquireLock)?;

        let (first_bytes, first_after) =
            read_exact_owned(acquired.descriptor(), opened.witness, control.expectation)?;
        let first_content = ExactContentWitness {
            sha256: Sha256::digest(&first_bytes).into(),
        };
        validate_expected_sha256(
            first_content,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_anchor_route(control)?;
        let (second_bytes, second_after) =
            read_exact_owned(acquired.descriptor(), first_after, control.expectation)?;
        let second_content = ExactContentWitness {
            sha256: Sha256::digest(&second_bytes).into(),
        };
        if first_bytes != second_bytes || first_content != second_content {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        revalidate_anchor_route(control)?;
        let descriptor = acquired.into_descriptor();
        Ok(LockHandle {
            descriptor,
            #[cfg(test)]
            bytes: Arc::from(second_bytes),
            witness: second_after,
            content_witness: second_content,
            root: control.root.clone(),
            path: control.path.clone(),
            expectation: control.expectation,
            component_identities: opened.component_identities,
            creator_process_id,
            lock_held: true,
            #[cfg(test)]
            mode,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn try_anchor_lock_preopened(
        control: &ControlHandle,
        descriptor: OwnedFd,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        let creator_process_id = std::process::id();
        validate_process_identity(
            control.root.state.creator_process_id,
            GenerationRootStage::AcquireLock,
        )?;
        validate_preopened_regular_descriptor(
            &descriptor,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_anchor_route(control)?;
        let opened = qualify_preopened_file(
            &control.root,
            &control.path,
            descriptor,
            control.expectation,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::AcquireLock,
        )?;
        if !opened.witness.same_control_anchor_identity(control.witness)
            || opened.component_identities != control.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        let operation = match mode {
            GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
            GenerationRootLockMode::Exclusive => {
                rustix::fs::FlockOperation::NonBlockingLockExclusive
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeLock)?;
        let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
            Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
            Err(error) if error == Errno::WOULDBLOCK => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::LockContended,
                    GenerationRootStage::AcquireLock,
                ));
            }
            Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterLock)?;
        validate_process_identity(creator_process_id, GenerationRootStage::AcquireLock)?;
        let descriptor_status_flags = rustix::fs::fcntl_getfl(acquired.descriptor())
            .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))?;

        let (first_bytes, first_after) =
            read_exact_owned(acquired.descriptor(), opened.witness, control.expectation)?;
        let first_content = ExactContentWitness {
            sha256: Sha256::digest(&first_bytes).into(),
        };
        validate_expected_sha256(
            first_content,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_anchor_route(control)?;
        let (second_bytes, second_after) =
            read_exact_owned(acquired.descriptor(), first_after, control.expectation)?;
        let second_content = ExactContentWitness {
            sha256: Sha256::digest(&second_bytes).into(),
        };
        if first_bytes != second_bytes || first_content != second_content {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        revalidate_anchor_route(control)?;
        let descriptor = acquired.into_descriptor();
        Ok(LockHandle {
            descriptor,
            #[cfg(test)]
            bytes: Arc::from(second_bytes),
            witness: second_after,
            content_witness: second_content,
            root: control.root.clone(),
            path: control.path.clone(),
            expectation: control.expectation,
            component_identities: opened.component_identities,
            creator_process_id,
            lock_held: true,
            descriptor_status_flags,
            #[cfg(test)]
            mode,
        })
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn try_lock(
        control: &ControlHandle,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        let creator_process_id = std::process::id();
        revalidate_control_route(control)?;
        let opened = open_relative_file(
            &control.root,
            &control.path,
            control.expectation,
            true,
            GenerationRootStage::AcquireLock,
            GenerationRootStage::AcquireLock,
        )?;
        if !opened.witness.mutation_stable_eq(control.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        if opened.component_identities != control.component_identities {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        let operation = match mode {
            GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
            GenerationRootLockMode::Exclusive => {
                rustix::fs::FlockOperation::NonBlockingLockExclusive
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeLock)?;
        let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
            Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
            Err(error) if error == Errno::WOULDBLOCK => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::LockContended,
                    GenerationRootStage::AcquireLock,
                ));
            }
            Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterLock)?;
        if creator_process_id != std::process::id() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                GenerationRootStage::AcquireLock,
            ));
        }
        let (observed_content, after) = read_exact_sha256(
            acquired.descriptor(),
            opened.witness,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        validate_bound_content(
            observed_content,
            control.content_witness,
            GenerationRootStage::AcquireLock,
        )?;
        validate_expected_sha256(
            observed_content,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_control_route(control)?;
        let (revalidated_content, revalidated_after) = read_exact_sha256(
            acquired.descriptor(),
            after,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        validate_bound_content(
            revalidated_content,
            control.content_witness,
            GenerationRootStage::AcquireLock,
        )?;
        let descriptor = acquired.into_descriptor();
        Ok(LockHandle {
            descriptor,
            #[cfg(test)]
            bytes: Arc::clone(&control.bytes),
            witness: revalidated_after,
            content_witness: revalidated_content,
            root: control.root.clone(),
            path: control.path.clone(),
            expectation: control.expectation,
            component_identities: opened.component_identities,
            creator_process_id,
            lock_held: true,
            #[cfg(test)]
            mode,
        })
    }

    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn try_lock_preopened(
        control: &ControlHandle,
        descriptor: OwnedFd,
        mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        let creator_process_id = std::process::id();
        validate_process_identity(
            control.root.state.creator_process_id,
            GenerationRootStage::AcquireLock,
        )?;
        validate_preopened_regular_descriptor(
            &descriptor,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_control_route(control)?;
        let opened = qualify_preopened_file(
            &control.root,
            &control.path,
            descriptor,
            control.expectation,
            PreopenedAccess::ReadWrite,
            GenerationRootStage::AcquireLock,
        )?;
        if !opened.witness.mutation_stable_eq(control.witness)
            || opened.component_identities != control.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::AcquireLock,
            ));
        }
        let operation = match mode {
            GenerationRootLockMode::Shared => rustix::fs::FlockOperation::NonBlockingLockShared,
            GenerationRootLockMode::Exclusive => {
                rustix::fs::FlockOperation::NonBlockingLockExclusive
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeLock)?;
        let acquired = match rustix::fs::flock(&opened.descriptor, operation) {
            Ok(()) => AcquiredFlockDescriptor::new(opened.descriptor, creator_process_id),
            Err(error) if error == Errno::WOULDBLOCK => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::LockContended,
                    GenerationRootStage::AcquireLock,
                ));
            }
            Err(error) => return Err(os_error(GenerationRootStage::AcquireLock, error)),
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterLock)?;
        if creator_process_id != std::process::id() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                GenerationRootStage::AcquireLock,
            ));
        }
        let descriptor_status_flags = rustix::fs::fcntl_getfl(acquired.descriptor())
            .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))?;
        let (locked_content, locked_after) = read_exact_sha256(
            acquired.descriptor(),
            opened.witness,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        validate_bound_content(
            locked_content,
            control.content_witness,
            GenerationRootStage::AcquireLock,
        )?;
        validate_expected_sha256(
            locked_content,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        revalidate_control_route(control)?;
        let (revalidated_content, revalidated_after) = read_exact_sha256(
            acquired.descriptor(),
            locked_after,
            control.expectation,
            GenerationRootStage::AcquireLock,
        )?;
        validate_bound_content(
            revalidated_content,
            control.content_witness,
            GenerationRootStage::AcquireLock,
        )?;
        let descriptor = acquired.into_descriptor();
        Ok(LockHandle {
            descriptor,
            #[cfg(test)]
            bytes: Arc::clone(&control.bytes),
            witness: revalidated_after,
            content_witness: revalidated_content,
            root: control.root.clone(),
            path: control.path.clone(),
            expectation: control.expectation,
            component_identities: opened.component_identities,
            creator_process_id,
            lock_held: true,
            descriptor_status_flags,
            #[cfg(test)]
            mode,
        })
    }

    #[cfg(test)]
    pub(super) fn lock_witness(lock: &LockHandle) -> GenerationRootObjectWitness {
        lock.witness
    }

    #[cfg(test)]
    pub(super) fn lock_bytes(lock: &LockHandle) -> Arc<[u8]> {
        Arc::clone(&lock.bytes)
    }

    #[cfg(test)]
    pub(super) fn lock_sha256(lock: &LockHandle) -> [u8; 32] {
        lock.content_witness.sha256
    }

    #[cfg(test)]
    pub(super) fn lock_mode(lock: &LockHandle) -> GenerationRootLockMode {
        lock.mode
    }

    #[cfg(all(
        test,
        any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))
    ))]
    pub(super) fn set_root_creator_process_id_for_test(root: &mut RootHandle, process_id: u32) {
        Arc::get_mut(&mut root.state)
            .expect("test root must have sole state ownership")
            .creator_process_id = process_id;
    }

    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn set_control_root_creator_process_id_for_test(
        control: &mut ControlHandle,
        process_id: u32,
    ) {
        Arc::get_mut(&mut control.root.state)
            .expect("test control root must have sole state ownership")
            .creator_process_id = process_id;
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn set_lock_creator_process_id_for_test(lock: &mut LockHandle, process_id: u32) {
        lock.creator_process_id = process_id;
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn duplicate_lock_descriptor_for_inheritance_for_test(
        lock: &LockHandle,
    ) -> GenerationRootResult<OwnedFd> {
        rustix::io::dup(&lock.descriptor)
            .map_err(|error| os_error(GenerationRootStage::AcquireLock, error))
    }

    pub(super) fn unlock(mut lock: LockHandle) -> GenerationRootResult<()> {
        if lock.creator_process_id != std::process::id() {
            drop(lock);
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                GenerationRootStage::ReleaseLock,
            ));
        }
        let validation_result = revalidate_lock_route(&lock)
            .and_then(|()| validate_lock_content(&lock, GenerationRootStage::ReleaseLock));
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeUnlock)?;
        let unlock_result = rustix::fs::flock(&lock.descriptor, rustix::fs::FlockOperation::Unlock)
            .map_err(|error| os_error(GenerationRootStage::ReleaseLock, error));
        if unlock_result.is_ok() {
            lock.lock_held = false;
        }
        #[cfg(test)]
        let unlock_result = unlock_result.and_then(|()| test_boundary(TestBoundary::AfterUnlock));
        drop(lock);
        validation_result.and(unlock_result)
    }

    impl Drop for LockHandle {
        fn drop(&mut self) {
            if self.lock_held && self.creator_process_id == std::process::id() {
                let _ = rustix::fs::flock(&self.descriptor, rustix::fs::FlockOperation::Unlock);
                self.lock_held = false;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn sync_lock_file(lock: &LockHandle) -> GenerationRootResult<()> {
        validate_lock_process(lock, GenerationRootStage::SyncRegularFile)?;
        revalidate_lock_route(lock)?;
        let (before_content, before) = read_exact_sha256(
            &lock.descriptor,
            lock.witness,
            lock.expectation,
            GenerationRootStage::SyncRegularFile,
        )?;
        validate_bound_content(
            before_content,
            lock.content_witness,
            GenerationRootStage::SyncRegularFile,
        )?;
        sync_file_descriptor(&lock.descriptor)?;
        let (after_content, after) = read_exact_sha256(
            &lock.descriptor,
            before,
            lock.expectation,
            GenerationRootStage::SyncRegularFile,
        )?;
        validate_bound_content(
            after_content,
            lock.content_witness,
            GenerationRootStage::SyncRegularFile,
        )?;
        revalidate_lock_route(lock)?;
        let (revalidated_content, _) = read_exact_sha256(
            &lock.descriptor,
            after,
            lock.expectation,
            GenerationRootStage::SyncRegularFile,
        )?;
        validate_bound_content(
            revalidated_content,
            lock.content_witness,
            GenerationRootStage::SyncRegularFile,
        )
    }

    fn validate_lock_process(
        lock: &LockHandle,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        validate_process_identity(lock.creator_process_id, stage)
    }

    fn validate_process_identity(
        expected_process_id: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        if expected_process_id == std::process::id() {
            Ok(())
        } else {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                stage,
            ))
        }
    }

    #[cfg(target_os = "linux")]
    fn revalidate_anchor_route(control: &ControlHandle) -> GenerationRootResult<()> {
        revalidate_root(&control.root)?;
        let retained = object_witness(
            &control.descriptor,
            control.witness.filesystem,
            control.witness.mount_identity,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        validate_private_regular_file(
            &control.descriptor,
            retained,
            control.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !retained.same_control_anchor_identity(control.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let reopened = open_relative_probe(
            &control.root,
            &control.path,
            control.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !reopened
            .witness
            .same_control_anchor_identity(control.witness)
            || reopened.component_identities != control.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_root(&control.root)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn revalidate_anchor_route(control: &ControlHandle) -> GenerationRootResult<()> {
        revalidate_root(&control.root)?;
        validate_preopened_regular_descriptor_status(
            &control.descriptor,
            OFlags::RDWR,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        let retained = object_witness(
            &control.descriptor,
            control.witness.filesystem,
            control.witness.mount_identity,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        validate_private_regular_file(
            &control.descriptor,
            retained,
            control.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !retained.same_control_anchor_identity(control.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let observed_components = bind_preopened_route(
            &control.root,
            &control.path,
            retained,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if observed_components.as_ref() != control.component_identities.as_ref() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_root(&control.root)
    }

    #[cfg(target_os = "linux")]
    #[cfg(test)]
    fn revalidate_control_route(control: &ControlHandle) -> GenerationRootResult<()> {
        revalidate_root(&control.root)?;
        let retained = object_witness(
            &control.descriptor,
            control.witness.filesystem,
            control.witness.mount_identity,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !retained.mutation_stable_eq(control.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let reopened = open_relative_probe(
            &control.root,
            &control.path,
            control.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !reopened.witness.mutation_stable_eq(control.witness)
            || reopened.component_identities != control.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_root(&control.root)
    }

    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    fn revalidate_control_route(control: &ControlHandle) -> GenerationRootResult<()> {
        revalidate_preopened_file_route(
            &control.root,
            &control.path,
            control.expectation,
            &control.descriptor,
            control.witness,
            &control.component_identities,
            OFlags::RDWR,
            GenerationRootStage::RevalidateRegularFile,
        )
    }

    #[cfg(target_os = "linux")]
    fn revalidate_lock_route(lock: &LockHandle) -> GenerationRootResult<()> {
        validate_lock_process(lock, GenerationRootStage::RevalidateRegularFile)?;
        revalidate_root(&lock.root)?;
        let retained = object_witness(
            &lock.descriptor,
            lock.witness.filesystem,
            lock.witness.mount_identity,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !retained.mutation_stable_eq(lock.witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let reopened = open_relative_probe(
            &lock.root,
            &lock.path,
            lock.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !reopened.witness.mutation_stable_eq(lock.witness)
            || reopened.component_identities != lock.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_root(&lock.root)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn revalidate_lock_route(lock: &LockHandle) -> GenerationRootResult<()> {
        validate_lock_process(lock, GenerationRootStage::RevalidateRegularFile)?;
        revalidate_preopened_file_route(
            &lock.root,
            &lock.path,
            lock.expectation,
            &lock.descriptor,
            lock.witness,
            &lock.component_identities,
            lock.descriptor_status_flags,
            GenerationRootStage::RevalidateRegularFile,
        )
    }

    fn validate_lock_content(
        lock: &LockHandle,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let (observed, _) =
            read_exact_sha256(&lock.descriptor, lock.witness, lock.expectation, stage)?;
        validate_bound_content(observed, lock.content_witness, stage)
    }

    struct OpenedFile {
        descriptor: OwnedFd,
        witness: GenerationRootObjectWitness,
        component_identities: Arc<[BasicObjectIdentity]>,
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[derive(Clone, Copy)]
    enum PreopenedAccess {
        ReadOnly,
        ReadWrite,
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn qualify_preopened_file(
        root: &RootHandle,
        path: &RelativePath,
        descriptor: OwnedFd,
        expectation: GenerationFileExpectation,
        access: PreopenedAccess,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OpenedFile> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforePreopenedQualification)?;
        validate_preopened_regular_descriptor(&descriptor, access, stage)?;
        let witness = object_witness(
            &descriptor,
            root.state.witness.filesystem,
            root.state.witness.mount_identity,
            stage,
        )?;
        validate_private_regular_file(&descriptor, witness, expectation, stage)?;
        let component_identities = bind_preopened_route(root, path, witness, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterPreopenedQualification)?;
        Ok(OpenedFile {
            descriptor,
            witness,
            component_identities,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn validate_preopened_regular_descriptor(
        descriptor: &OwnedFd,
        expected_access: PreopenedAccess,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let expected_status_flags = match expected_access {
            PreopenedAccess::ReadOnly => OFlags::RDONLY,
            PreopenedAccess::ReadWrite => OFlags::RDWR,
        };
        validate_preopened_regular_descriptor_status(descriptor, expected_status_flags, stage)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn validate_preopened_regular_descriptor_status(
        descriptor: &OwnedFd,
        expected_status_flags: OFlags,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::NotRegularFile,
                stage,
            ));
        }
        let observed_flags =
            rustix::fs::fcntl_getfl(descriptor).map_err(|error| os_error(stage, error))?;
        if observed_flags != expected_status_flags {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::InvalidDescriptorAccess,
                stage,
            ));
        }
        let descriptor_flags =
            rustix::io::fcntl_getfd(descriptor).map_err(|error| os_error(stage, error))?;
        if !descriptor_flags.contains(rustix::io::FdFlags::CLOEXEC) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::CloseOnExecRequired,
                stage,
            ));
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn bind_preopened_route(
        root: &RootHandle,
        path: &RelativePath,
        witness: GenerationRootObjectWitness,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<Arc<[BasicObjectIdentity]>> {
        use rustix::fs::{AtFlags, statat};

        let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
            GenerationRootError::new(
                GenerationRootErrorKind::InvalidRoute,
                GenerationRootStage::ParseRelativeRoute,
            )
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
        let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
            .map_err(|error| os_error(stage, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
        let mut component_identities = Vec::new();
        component_identities
            .try_reserve(path.components.len())
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        for (index, component) in ancestors.iter().enumerate() {
            parent = open_relative_directory_component(
                &parent,
                component,
                index,
                root.state.witness,
                stage,
            )?;
            component_identities.push(basic_identity(&parent, stage)?);
        }
        let index = ancestors.len();
        verify_exact_directory_entry(&parent, final_name, stage, index)?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFinalRouteStat)?;
        let route_stat = statat(&parent, final_name, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(|error| macos_open_error(stage, error).at_component(index))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFinalRouteStat)?;
        verify_exact_directory_entry(&parent, final_name, stage, index)?;
        let route_file_type = FileType::from_raw_mode(route_stat.st_mode);
        if route_file_type == FileType::Symlink {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
                    .at_component(index),
            );
        }
        if route_file_type != FileType::RegularFile {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::NotRegularFile, stage)
                    .at_component(index),
            );
        }
        if !macos_route_stat_matches_witness(&route_stat, witness) {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index),
            );
        }
        component_identities.push(BasicObjectIdentity {
            device: witness.device,
            inode: witness.inode,
            mode: witness.mode,
            uid: witness.uid,
            gid: witness.gid,
            mount_identity: witness.mount_identity,
        });
        Ok(Arc::from(component_identities))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn macos_route_stat_matches_witness(
        stat: &rustix::fs::Stat,
        witness: GenerationRootObjectWitness,
    ) -> bool {
        let Ok(byte_len) = u64::try_from(stat.st_size) else {
            return false;
        };
        stat.st_flags == 0
            && stat_device_as_u64(stat.st_dev) == witness.device
            && stat_inode_as_u64(stat.st_ino) == witness.inode
            && stat_mode_as_u32(stat.st_mode) == witness.mode
            && stat_link_count_as_u64(stat.st_nlink) == witness.hard_links
            && stat.st_uid == witness.uid
            && stat.st_gid == witness.gid
            && byte_len == witness.byte_len
            && stat_seconds_as_i64(stat.st_mtime) == witness.modified_seconds
            && stat.st_mtime_nsec == witness.modified_nanoseconds
            && stat_seconds_as_i64(stat.st_ctime) == witness.changed_seconds
            && stat.st_ctime_nsec == witness.changed_nanoseconds
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn revalidate_preopened_file_route(
        root: &RootHandle,
        path: &RelativePath,
        expectation: GenerationFileExpectation,
        descriptor: &OwnedFd,
        expected: GenerationRootObjectWitness,
        component_identities: &[BasicObjectIdentity],
        expected_status_flags: OFlags,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        revalidate_root(root)?;
        validate_preopened_regular_descriptor_status(descriptor, expected_status_flags, stage)?;
        let retained = object_witness(
            descriptor,
            expected.filesystem,
            expected.mount_identity,
            stage,
        )?;
        validate_private_regular_file(descriptor, retained, expectation, stage)?;
        if !retained.mutation_stable_eq(expected) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        let observed_components = bind_preopened_route(root, path, retained, stage)?;
        if observed_components.as_ref() != component_identities {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        revalidate_root(root)
    }

    #[cfg(target_os = "linux")]
    fn revalidate_file_route(
        file: &FileHandle,
        expected: GenerationRootObjectWitness,
    ) -> GenerationRootResult<()> {
        revalidate_root(&file.root)?;
        let retained = object_witness(
            &file.descriptor,
            expected.filesystem,
            expected.mount_identity,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !retained.mutation_stable_eq(expected) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        let reopened = open_relative_probe(
            &file.root,
            &file.path,
            file.expectation,
            GenerationRootStage::RevalidateRegularFile,
        )?;
        if !reopened.witness.mutation_stable_eq(expected)
            || reopened.component_identities != file.component_identities
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::RevalidateRegularFile,
            ));
        }
        revalidate_root(&file.root)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn revalidate_file_route(
        file: &FileHandle,
        expected: GenerationRootObjectWitness,
    ) -> GenerationRootResult<()> {
        revalidate_preopened_file_route(
            &file.root,
            &file.path,
            file.expectation,
            &file.descriptor,
            expected,
            &file.component_identities,
            OFlags::RDONLY,
            GenerationRootStage::RevalidateRegularFile,
        )
    }

    #[cfg(target_os = "linux")]
    fn open_relative_file(
        root: &RootHandle,
        path: &RelativePath,
        expectation: GenerationFileExpectation,
        writable: bool,
        directory_stage: GenerationRootStage,
        regular_stage: GenerationRootStage,
    ) -> GenerationRootResult<OpenedFile> {
        let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
            GenerationRootError::new(
                GenerationRootErrorKind::InvalidRoute,
                GenerationRootStage::ParseRelativeRoute,
            )
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
        let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
            .map_err(|error| os_error(directory_stage, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
        let mut component_identities = Vec::new();
        component_identities
            .try_reserve(path.components.len())
            .map_err(|_| {
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, directory_stage)
            })?;
        for (index, component) in ancestors.iter().enumerate() {
            parent = open_relative_directory_component(
                &parent,
                component,
                index,
                root.state.witness,
                directory_stage,
            )?;
            component_identities.push(basic_identity(&parent, directory_stage)?);
        }
        let descriptor = open_regular_component(
            &parent,
            final_name,
            ancestors.len(),
            root.state.witness,
            expectation,
            writable,
            root.state.creator_process_id,
            regular_stage,
        )?;
        let witness = object_witness(
            &descriptor,
            root.state.witness.filesystem,
            root.state.witness.mount_identity,
            regular_stage,
        )?;
        validate_private_regular_file(&descriptor, witness, expectation, regular_stage)?;
        component_identities.push(basic_identity(&descriptor, regular_stage)?);
        #[cfg(test)]
        test_boundary(TestBoundary::AfterQualifiedFileOpen)?;
        Ok(OpenedFile {
            descriptor,
            witness,
            component_identities: Arc::from(component_identities),
        })
    }

    #[cfg(target_os = "linux")]
    fn open_relative_probe(
        root: &RootHandle,
        path: &RelativePath,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OpenedFile> {
        let (final_name, ancestors) = path.components.split_last().ok_or_else(|| {
            GenerationRootError::new(
                GenerationRootErrorKind::InvalidRoute,
                GenerationRootStage::ParseRelativeRoute,
            )
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRootDescriptorDuplicate)?;
        let mut parent = rustix::io::fcntl_dupfd_cloexec(&root.state.descriptor, 0)
            .map_err(|error| os_error(stage, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRootDescriptorDuplicate)?;
        let mut component_identities = Vec::new();
        component_identities
            .try_reserve(path.components.len())
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        for (index, component) in ancestors.iter().enumerate() {
            parent = open_relative_directory_component(
                &parent,
                component,
                index,
                root.state.witness,
                stage,
            )?;
            component_identities.push(basic_identity(&parent, stage)?);
        }
        let index = ancestors.len();
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFinalRouteProbe { index })?;
        let (descriptor, witness) = probe_regular_component(
            &parent,
            final_name,
            index,
            root.state.witness,
            expectation,
            stage,
        )?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFinalRouteProbe { index })?;
        component_identities.push(basic_identity(&descriptor, stage)?);
        Ok(OpenedFile {
            descriptor,
            witness,
            component_identities: Arc::from(component_identities),
        })
    }

    fn read_exact_owned(
        descriptor: &OwnedFd,
        before: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<(Vec<u8>, GenerationRootObjectWitness)> {
        let max_byte_len = expectation.max_byte_len();
        if expectation.byte_len > max_byte_len {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(max_byte_len, expectation.byte_len));
        }
        let byte_len = usize::try_from(expectation.byte_len).map_err(|_| {
            GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(usize_as_u64(usize::MAX), expectation.byte_len)
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|_| {
            GenerationRootError::new(
                GenerationRootErrorKind::ResourceLimit,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(max_byte_len, expectation.byte_len)
        })?;
        bytes.resize(byte_len, 0);
        let mut offset = 0_usize;
        while offset < bytes.len() {
            let file_offset = usize_as_u64(offset);
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeRead {
                offset: file_offset,
            })?;
            match pread(descriptor, &mut bytes[offset..], file_offset) {
                Ok(0) => {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::SizeMismatch,
                        GenerationRootStage::ReadRegularFile,
                    )
                    .with_counts(expectation.byte_len, file_offset));
                }
                Ok(read) => {
                    #[cfg(test)]
                    test_boundary(TestBoundary::AfterRead {
                        offset: file_offset,
                        byte_count: read,
                    })?;
                    offset += read;
                }
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(os_error(GenerationRootStage::ReadRegularFile, error));
                }
            }
        }
        let mut probe = [0_u8; 1];
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeTrailingByteProbe)?;
        let probe_read = pread(descriptor, &mut probe, expectation.byte_len)
            .map_err(|error| os_error(GenerationRootStage::ReadRegularFile, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterTrailingByteProbe {
            byte_count: probe_read,
        })?;
        if probe_read != 0 {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::SizeMismatch,
                GenerationRootStage::ReadRegularFile,
            )
            .with_counts(expectation.byte_len, expectation.byte_len.saturating_add(1)));
        }
        #[cfg(test)]
        test_boundary(TestBoundary::AfterExactRead)?;
        let after = object_witness(
            descriptor,
            before.filesystem,
            before.mount_identity,
            GenerationRootStage::ReadRegularFile,
        )?;
        if !before.mutation_stable_eq(after) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::ReadRegularFile,
            ));
        }
        Ok((bytes, after))
    }

    fn read_exact_sha256(
        descriptor: &OwnedFd,
        before: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<(ExactContentWitness, GenerationRootObjectWitness)> {
        const HASH_BUFFER_BYTES: usize = 64 * 1024;

        let max_byte_len = expectation.max_byte_len();
        if expectation.byte_len > max_byte_len {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                    .with_counts(max_byte_len, expectation.byte_len),
            );
        }
        let mut digest = Sha256::new();
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(HASH_BUFFER_BYTES)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        buffer.resize(HASH_BUFFER_BYTES, 0_u8);
        let mut offset = 0_u64;
        while offset < expectation.byte_len {
            let remaining = expectation.byte_len.saturating_sub(offset);
            let requested = usize::try_from(remaining)
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeRead { offset })?;
            match pread(descriptor, &mut buffer[..requested], offset) {
                Ok(0) => {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::SizeMismatch,
                        stage,
                    )
                    .with_counts(expectation.byte_len, offset));
                }
                Ok(read) => {
                    #[cfg(test)]
                    test_boundary(TestBoundary::AfterRead {
                        offset,
                        byte_count: read,
                    })?;
                    digest.update(&buffer[..read]);
                    offset = offset.saturating_add(usize_as_u64(read));
                }
                Err(error) if error == Errno::INTR => {}
                Err(error) => return Err(os_error(stage, error)),
            }
        }
        let mut probe = [0_u8; 1];
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeTrailingByteProbe)?;
        let probe_read = pread(descriptor, &mut probe, expectation.byte_len)
            .map_err(|error| os_error(stage, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterTrailingByteProbe {
            byte_count: probe_read,
        })?;
        if probe_read != 0 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::SizeMismatch, stage)
                    .with_counts(expectation.byte_len, expectation.byte_len.saturating_add(1)),
            );
        }
        #[cfg(test)]
        test_boundary(TestBoundary::AfterExactRead)?;
        let after = object_witness(descriptor, before.filesystem, before.mount_identity, stage)?;
        if !before.mutation_stable_eq(after) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        Ok((
            ExactContentWitness {
                sha256: digest.finalize().into(),
            },
            after,
        ))
    }

    fn validate_bound_content(
        observed: ExactContentWitness,
        expected: ExactContentWitness,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        if observed == expected {
            Ok(())
        } else {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ))
        }
    }

    fn validate_expected_sha256(
        observed: ExactContentWitness,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        if expectation
            .sha256
            .is_some_and(|expected| expected != observed.sha256)
        {
            Err(GenerationRootError::new(
                GenerationRootErrorKind::HashMismatch,
                stage,
            ))
        } else {
            Ok(())
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_bound_control_content_for_test(
        observed: [u8; 32],
        expected: [u8; 32],
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        validate_bound_content(
            ExactContentWitness { sha256: observed },
            ExactContentWitness { sha256: expected },
            stage,
        )
    }

    fn parse_route_bytes(
        value: &OsStr,
        absolute: bool,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<Vec<OsString>> {
        let bytes = value.as_bytes();
        if bytes.is_empty() || bytes.len() > GENERATION_ROOT_MAX_ROUTE_BYTES || bytes.contains(&0) {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::InvalidRoute, stage).with_counts(
                    usize_as_u64(GENERATION_ROOT_MAX_ROUTE_BYTES),
                    usize_as_u64(bytes.len()),
                ),
            );
        }
        if absolute != bytes.starts_with(b"/") {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::InvalidRoute,
                stage,
            ));
        }
        let body = if absolute { &bytes[1..] } else { bytes };
        if body.is_empty() || body.starts_with(b"/") || body.ends_with(b"/") {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::InvalidRoute,
                stage,
            ));
        }
        let mut components = Vec::new();
        components
            .try_reserve(GENERATION_ROOT_MAX_COMPONENTS)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        for (index, component) in body.split(|byte| *byte == b'/').enumerate() {
            if component.is_empty()
                || component == b"."
                || component == b".."
                || component.len() > GENERATION_ROOT_MAX_COMPONENT_BYTES
            {
                return Err(
                    GenerationRootError::new(GenerationRootErrorKind::InvalidRoute, stage)
                        .at_component(index)
                        .with_counts(
                            usize_as_u64(GENERATION_ROOT_MAX_COMPONENT_BYTES),
                            usize_as_u64(component.len()),
                        ),
                );
            }
            if index >= GENERATION_ROOT_MAX_COMPONENTS {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::ResourceLimit,
                    stage,
                )
                .at_component(index)
                .with_counts(
                    usize_as_u64(GENERATION_ROOT_MAX_COMPONENTS),
                    usize_as_u64(index.saturating_add(1)),
                ));
            }
            components.push(OsStr::from_bytes(component).to_os_string());
        }
        Ok(components)
    }

    fn basic_identity(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<BasicObjectIdentity> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        let mount_identity = route_mount_identity(descriptor, stage)?;
        Ok(BasicObjectIdentity {
            device: stat_device_as_u64(stat.st_dev),
            inode: stat_inode_as_u64(stat.st_ino),
            mode: stat_mode_as_u32(stat.st_mode),
            uid: stat.st_uid,
            gid: stat.st_gid,
            mount_identity,
        })
    }

    fn absolute_ancestor_security_witness(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<AbsoluteAncestorSecurityWitness> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        #[cfg(target_os = "linux")]
        let changed_nanoseconds = i64::try_from(stat.st_ctime_nsec)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let changed_nanoseconds = stat.st_ctime_nsec;
        Ok(AbsoluteAncestorSecurityWitness {
            identity: basic_identity(descriptor, stage)?,
            changed_seconds: stat_seconds_as_i64(stat.st_ctime),
            changed_nanoseconds,
        })
    }

    fn validate_absolute_ancestor_owner_mode(
        uid: u32,
        mode: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let effective_uid = rustix::process::geteuid().as_raw();
        if uid != effective_uid && uid != 0 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::WrongOwner, stage)
                    .with_counts(u64::from(effective_uid), u64::from(uid)),
            );
        }
        let writable_by_non_owner = mode & 0o022;
        if writable_by_non_owner != 0 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::WrongMode, stage)
                    .with_counts(0, u64::from(writable_by_non_owner)),
            );
        }
        Ok(())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_absolute_ancestor_owner_mode_for_test(
        uid: u32,
        mode: u32,
    ) -> GenerationRootResult<()> {
        validate_absolute_ancestor_owner_mode(uid, mode, GenerationRootStage::QualifyRootSecurity)
    }

    fn at_absolute_component(
        error: GenerationRootError,
        index: Option<usize>,
    ) -> GenerationRootError {
        index.map_or(error, |index| error.at_component(index))
    }

    fn qualify_absolute_ancestor(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
        index: Option<usize>,
    ) -> GenerationRootResult<BasicObjectIdentity> {
        let result = (|| {
            let before = absolute_ancestor_security_witness(descriptor, stage)?;
            #[cfg(target_os = "linux")]
            let raw_mode = before.identity.mode;
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let raw_mode = u16::try_from(before.identity.mode).map_err(|_| {
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
            })?;
            if FileType::from_raw_mode(raw_mode) != FileType::Directory {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::NotDirectory,
                    stage,
                ));
            }
            validate_absolute_ancestor_owner_mode(
                before.identity.uid,
                before.identity.mode,
                stage,
            )?;
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeAbsoluteAncestorAclRead { index })?;
            let acl_presence =
                crate::fd_acl::extended_acl_presence(descriptor.as_fd()).map_err(|error| {
                    GenerationRootError::new(GenerationRootErrorKind::AclRejected, stage)
                        .with_raw_os_error(error.raw_os_error().unwrap_or(libc::EIO))
                })?;
            #[cfg(test)]
            test_boundary(TestBoundary::AfterAbsoluteAncestorAclRead { index })?;
            match acl_presence {
                crate::fd_acl::ExtendedAclPresence::Absent => {}
                crate::fd_acl::ExtendedAclPresence::Present => {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::AclRejected,
                        stage,
                    ));
                }
            }
            let after = absolute_ancestor_security_witness(descriptor, stage)?;
            if before != after {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::ObjectChanged,
                    stage,
                ));
            }
            Ok(after.identity)
        })();
        result.map_err(|error| at_absolute_component(error, index))
    }

    fn validate_private_directory(
        descriptor: &OwnedFd,
        witness: GenerationRootObjectWitness,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::NotDirectory,
                stage,
            ));
        }
        validate_owner_mode(witness, GENERATION_ROOT_DIRECTORY_MODE, stage)?;
        inspect_acl_if_required(descriptor, witness, stage)
    }

    fn validate_private_regular_file(
        descriptor: &OwnedFd,
        witness: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        validate_private_regular_file_metadata(descriptor, witness, expectation, stage)?;
        inspect_acl_if_required(descriptor, witness, stage)
    }

    /// Validate only metadata available through either a data descriptor or a
    /// Linux `O_PATH` probe. ACL inspection is deliberately excluded: Linux
    /// `fgetxattr` rejects `O_PATH` descriptors with `EBADF`, so callers must
    /// invoke [`validate_private_regular_file`] on the proc-derived RDONLY or
    /// RDWR descriptor before content I/O.
    fn validate_private_regular_file_metadata(
        descriptor: &OwnedFd,
        witness: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::NotRegularFile,
                stage,
            ));
        }
        if witness.hard_links != 1 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::HardLinked, stage)
                    .with_counts(1, witness.hard_links),
            );
        }
        validate_owner_mode(witness, expectation.expected_mode(), stage)?;
        if witness.byte_len != expectation.byte_len {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::SizeMismatch, stage)
                    .with_counts(expectation.byte_len, witness.byte_len),
            );
        }
        Ok(())
    }

    fn validate_owner_mode(
        witness: GenerationRootObjectWitness,
        expected_mode: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let effective_uid = rustix::process::geteuid().as_raw();
        if witness.uid != effective_uid {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::WrongOwner, stage)
                    .with_counts(u64::from(effective_uid), u64::from(witness.uid)),
            );
        }
        let actual_mode = witness.mode & MODE_PERMISSIONS_MASK;
        if actual_mode != expected_mode {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::WrongMode, stage)
                    .with_counts(u64::from(expected_mode), u64::from(actual_mode)),
            );
        }
        Ok(())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_owner_mode_for_test(
        witness: GenerationRootObjectWitness,
        expected_mode: u32,
    ) -> GenerationRootResult<()> {
        validate_owner_mode(
            witness,
            expected_mode,
            GenerationRootStage::QualifyRootSecurity,
        )
    }

    fn os_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        GenerationRootError::new(GenerationRootErrorKind::Io, stage)
            .with_raw_os_error(error.raw_os_error())
    }

    fn usize_as_u64(value: usize) -> u64 {
        u64::try_from(value).unwrap_or(u64::MAX)
    }

    pub(super) fn stat_mode_as_u32<T>(mode: T) -> u32
    where
        u32: From<T>,
    {
        u32::from(mode)
    }

    pub(super) fn stat_inode_as_u64<T>(inode: T) -> u64
    where
        u64: From<T>,
    {
        u64::from(inode)
    }

    pub(super) fn stat_link_count_as_u64<T>(link_count: T) -> u64
    where
        u64: From<T>,
    {
        u64::from(link_count)
    }

    pub(super) fn stat_seconds_as_i64<T>(seconds: T) -> i64
    where
        i64: From<T>,
    {
        i64::from(seconds)
    }

    #[cfg(target_os = "linux")]
    fn open_absolute_route(
        route: &AbsoluteRoute,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OpenedRoute> {
        use rustix::fs::{CWD, ResolveFlags, openat2};

        let flags = OFlags::RDONLY
            | OFlags::DIRECTORY
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK;
        let mut descriptor = openat2(
            CWD,
            Path::new("/"),
            flags,
            Mode::empty(),
            ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
        )
        .map_err(|error| linux_open_error(stage, error))?;
        let mut route_identities = Vec::new();
        route_identities
            .try_reserve(route.components.len() + 1)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        route_identities.push(qualify_absolute_ancestor(&descriptor, stage, None)?);
        for (index, component) in route.components.iter().enumerate() {
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeRootComponentOpen { index })?;
            verify_exact_directory_entry(&descriptor, component, stage, index)?;
            let next = openat2(
                &descriptor,
                component,
                flags,
                Mode::empty(),
                ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
            )
            .map_err(|error| {
                linux_component_open_error(&descriptor, component, stage, index, error)
            })?;
            verify_exact_directory_entry(&descriptor, component, stage, index)?;
            descriptor = next;
            let stat = fstat(&descriptor).map_err(|error| os_error(stage, error))?;
            if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
                return Err(
                    GenerationRootError::new(GenerationRootErrorKind::NotDirectory, stage)
                        .at_component(index),
                );
            }
            let identity = if index + 1 == route.components.len() {
                basic_identity(&descriptor, stage)?
            } else {
                qualify_absolute_ancestor(&descriptor, stage, Some(index))?
            };
            #[cfg(test)]
            test_boundary(TestBoundary::AfterRootComponentOpen { index })?;
            route_identities.push(identity);
        }
        Ok(OpenedRoute {
            descriptor,
            route_identities,
        })
    }

    #[cfg(target_os = "linux")]
    fn open_relative_directory_component(
        parent: &OwnedFd,
        component: &OsStr,
        index: usize,
        root: GenerationRootObjectWitness,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OwnedFd> {
        use rustix::fs::{ResolveFlags, openat2};

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRelativeDirectoryOpen { index })?;
        verify_exact_directory_entry(parent, component, stage, index)?;
        let descriptor = openat2(
            parent,
            component,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::CLOEXEC
                | OFlags::NONBLOCK,
            Mode::empty(),
            ResolveFlags::BENEATH
                | ResolveFlags::NO_SYMLINKS
                | ResolveFlags::NO_MAGICLINKS
                | ResolveFlags::NO_XDEV,
        )
        .map_err(|error| linux_component_open_error(parent, component, stage, index, error))?;
        verify_exact_directory_entry(parent, component, stage, index)?;
        let witness = object_witness(&descriptor, root.filesystem, root.mount_identity, stage)?;
        if witness.mount_identity != root.mount_identity {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::CrossDevice, stage)
                    .at_component(index),
            );
        }
        validate_private_directory(&descriptor, witness, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRelativeDirectoryOpen { index })?;
        Ok(descriptor)
    }

    #[cfg(target_os = "linux")]
    fn open_regular_component(
        parent: &OwnedFd,
        component: &OsStr,
        index: usize,
        root: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
        writable: bool,
        creator_process_id: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OwnedFd> {
        use rustix::fs::openat;

        let access = if writable {
            OFlags::RDWR
        } else {
            OFlags::RDONLY
        };
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRegularFileOpen { index })?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFinalProbeOpen { index })?;
        let (probe, probe_witness) =
            probe_regular_component(parent, component, index, root, expectation, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProbeQualified { index })?;
        validate_process_identity(creator_process_id, stage)?;
        let (proc_root, proc_mount_id) = open_verified_proc_root(creator_process_id, stage)?;
        let probe_route = proc_thread_fd_route(probe.as_raw_fd());
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeProcFdReopen { index })?;
        validate_proc_bridge_entry(
            &proc_root,
            &probe_route,
            proc_mount_id,
            index,
            ProcBridgeProbePhase::BeforeDataOpen,
            stage,
        )?;
        validate_proc_self_binding(
            &proc_root,
            creator_process_id,
            ProcSelfBindingPhase::BeforeDataOpen,
            stage,
        )?;
        let descriptor = loop {
            match openat(
                &proc_root,
                &probe_route,
                access | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOATIME,
                Mode::empty(),
            ) {
                Ok(descriptor) => break descriptor,
                Err(error) if error == Errno::INTR => {}
                Err(error) if error == Errno::PERM || error == Errno::ACCESS => {
                    validate_process_identity(creator_process_id, stage)
                        .map_err(|process_error| process_error.at_component(index))?;
                    let retained_probe =
                        object_witness(&probe, root.filesystem, root.mount_identity, stage)
                            .map_err(|witness_error| witness_error.at_component(index))?;
                    if !probe_witness.mutation_stable_eq(retained_probe) {
                        return Err(GenerationRootError::new(
                            GenerationRootErrorKind::ObjectChanged,
                            stage,
                        )
                        .at_component(index));
                    }
                    return Err(proc_bridge_error(stage, error).at_component(index));
                }
                Err(error) => {
                    return Err(proc_bridge_error(stage, error).at_component(index));
                }
            }
        };
        validate_proc_self_binding(
            &proc_root,
            creator_process_id,
            ProcSelfBindingPhase::AfterDataOpen,
            stage,
        )
        .map_err(|error| error.at_component(index))?;
        validate_proc_bridge_entry(
            &proc_root,
            &probe_route,
            proc_mount_id,
            index,
            ProcBridgeProbePhase::AfterDataOpen,
            stage,
        )?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProcFdReopen { index })?;
        let reopened_witness =
            object_witness(&descriptor, root.filesystem, root.mount_identity, stage)?;
        validate_private_regular_file(&descriptor, reopened_witness, expectation, stage)?;
        if !probe_witness.mutation_stable_eq(reopened_witness) {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index),
            );
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFinalRouteProbe { index })?;
        let (_final_probe, final_probe_witness) =
            probe_regular_component(parent, component, index, root, expectation, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFinalRouteProbe { index })?;
        if !probe_witness.mutation_stable_eq(final_probe_witness)
            || !reopened_witness.mutation_stable_eq(final_probe_witness)
        {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index),
            );
        }
        #[cfg(test)]
        {
            let cloexec = rustix::io::fcntl_getfd(&descriptor)
                .map_err(|error| os_error(stage, error).at_component(index))?
                .contains(rustix::io::FdFlags::CLOEXEC);
            test_boundary(TestBoundary::AfterRegularFileOpen { index, cloexec })?;
        }
        validate_process_identity(creator_process_id, stage)?;
        Ok(descriptor)
    }

    #[cfg(target_os = "linux")]
    fn proc_thread_fd_route(raw_fd: RawFd) -> String {
        format!("thread-self/fd/{raw_fd}")
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn proc_thread_fd_route_for_test(raw_fd: RawFd) -> String {
        proc_thread_fd_route(raw_fd)
    }

    #[cfg(target_os = "linux")]
    fn probe_regular_component(
        parent: &OwnedFd,
        component: &OsStr,
        index: usize,
        root: GenerationRootObjectWitness,
        expectation: GenerationFileExpectation,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<(OwnedFd, GenerationRootObjectWitness)> {
        use rustix::fs::{ResolveFlags, openat2};

        verify_exact_directory_entry(parent, component, stage, index)?;
        let probe = openat2(
            parent,
            component,
            OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::BENEATH
                | ResolveFlags::NO_SYMLINKS
                | ResolveFlags::NO_MAGICLINKS
                | ResolveFlags::NO_XDEV,
        )
        .map_err(|error| linux_open_error(stage, error).at_component(index))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFinalProbeOpen { index })?;
        verify_exact_directory_entry(parent, component, stage, index)?;
        validate_regular_descriptor_type(&probe, stage, index)?;
        let witness = object_witness(&probe, root.filesystem, root.mount_identity, stage)?;
        if witness.mount_identity != root.mount_identity {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::CrossDevice, stage)
                    .at_component(index),
            );
        }
        validate_private_regular_file_metadata(&probe, witness, expectation, stage)?;
        Ok((probe, witness))
    }

    #[cfg(target_os = "linux")]
    pub(super) fn validate_regular_descriptor_type(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
        index: usize,
    ) -> GenerationRootResult<()> {
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error).at_component(index))?;
        match FileType::from_raw_mode(stat.st_mode) {
            FileType::RegularFile => Ok(()),
            FileType::Symlink => Err(GenerationRootError::new(
                GenerationRootErrorKind::SymbolicLink,
                stage,
            )
            .at_component(index)),
            _ => Err(
                GenerationRootError::new(GenerationRootErrorKind::NotRegularFile, stage)
                    .at_component(index),
            ),
        }
    }

    #[cfg(target_os = "linux")]
    fn open_verified_proc_root(
        process_id: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<(OwnedFd, u64)> {
        use rustix::fs::{CWD, ResolveFlags, openat2};

        validate_process_identity(process_id, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeProcCapabilityRootOpen)?;
        let descriptor = loop {
            match openat2(
                CWD,
                Path::new("/proc"),
                OFlags::PATH | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
                ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
            ) {
                Ok(descriptor) => break descriptor,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_root_error(stage, error));
                }
            }
        };
        let filesystem = loop {
            match fstatfs(&descriptor) {
                Ok(filesystem) => break filesystem,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_root_error(stage, error));
                }
            }
        };
        if i128::from(filesystem.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            ));
        }
        let stat = loop {
            match fstat(&descriptor) {
                Ok(stat) => break stat,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_root_error(stage, error));
                }
            }
        };
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            ));
        }
        let proc_mount_id = proc_mount_id(&descriptor, stage)?;
        validate_proc_self_binding(
            &descriptor,
            process_id,
            ProcSelfBindingPhase::RootQualification,
            stage,
        )?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProcCapabilityRootOpen)?;
        validate_process_identity(process_id, stage)?;
        Ok((descriptor, proc_mount_id))
    }

    #[cfg(target_os = "linux")]
    fn validate_proc_self_binding(
        proc_root: &OwnedFd,
        process_id: u32,
        phase: ProcSelfBindingPhase,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        use rustix::fs::readlinkat;

        let _ = phase;

        if process_id != std::process::id() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                stage,
            ));
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeProcSelfReadlink { phase })?;
        let target = loop {
            match readlinkat(proc_root, "self", Vec::new()) {
                Ok(target) => break target,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error));
                }
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProcSelfReadlink { phase })?;
        if process_id != std::process::id() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                stage,
            ));
        }
        if target.as_bytes() != process_id.to_string().as_bytes() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        let thread_id = rustix::thread::gettid();
        let expected_thread_target = format!("{process_id}/task/{thread_id}");
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeProcThreadSelfReadlink { phase })?;
        let thread_target = loop {
            match readlinkat(proc_root, "thread-self", Vec::new()) {
                Ok(target) => break target,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error));
                }
            }
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProcThreadSelfReadlink { phase })?;
        if process_id != std::process::id() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ForkedProcess,
                stage,
            ));
        }
        if thread_target.as_bytes() != expected_thread_target.as_bytes() {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn validate_proc_bridge_entry(
        proc_root: &OwnedFd,
        probe_route: &str,
        expected_proc_mount_id: u64,
        index: usize,
        phase: ProcBridgeProbePhase,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        use rustix::fs::{ResolveFlags, openat2};

        let _ = phase;

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeProcBridgeProbe { index, phase })?;
        let descriptor = loop {
            match openat2(
                proc_root,
                probe_route,
                OFlags::PATH | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
                ResolveFlags::NO_XDEV,
            ) {
                Ok(descriptor) => break descriptor,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error).at_component(index));
                }
            }
        };
        let stat = loop {
            match fstat(&descriptor) {
                Ok(stat) => break stat,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error).at_component(index));
                }
            }
        };
        if FileType::from_raw_mode(stat.st_mode) != FileType::Symlink {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index),
            );
        }
        let filesystem = loop {
            match fstatfs(&descriptor) {
                Ok(filesystem) => break filesystem,
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error).at_component(index));
                }
            }
        };
        if i128::from(filesystem.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index),
            );
        }
        let observed_mount_id =
            proc_mount_id(&descriptor, stage).map_err(|error| error.at_component(index))?;
        if observed_mount_id != expected_proc_mount_id {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index)
                    .with_counts(expected_proc_mount_id, observed_mount_id),
            );
        }
        #[cfg(test)]
        test_boundary(TestBoundary::AfterProcBridgeProbe { index, phase })?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn proc_mount_id(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<u64> {
        let extended = proc_statx(descriptor, stage)?;
        validate_proc_mount_id_mask(
            rustix::fs::StatxFlags::from_bits_retain(extended.stx_mask),
            extended.stx_mnt_id,
            stage,
        )
    }

    #[cfg(target_os = "linux")]
    fn proc_statx(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<rustix::fs::Statx> {
        use rustix::fs::{AtFlags, StatxFlags, statx};

        loop {
            match statx(
                descriptor,
                "",
                AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
                StatxFlags::BASIC_STATS | StatxFlags::MNT_ID,
            ) {
                Ok(extended) => return Ok(extended),
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(proc_bridge_error(stage, error));
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn validate_proc_mount_id_mask(
        returned_mask: rustix::fs::StatxFlags,
        mount_id: u64,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<u64> {
        if returned_mask.contains(rustix::fs::StatxFlags::MNT_ID) {
            Ok(mount_id)
        } else {
            Err(
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedKernelFeature, stage)
                    .with_counts(
                        u64::from(rustix::fs::StatxFlags::MNT_ID.bits()),
                        u64::from(returned_mask.bits() & rustix::fs::StatxFlags::MNT_ID.bits()),
                    ),
            )
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_proc_mount_id_mask_for_test(
        returned_mask: rustix::fs::StatxFlags,
        mount_id: u64,
    ) -> GenerationRootResult<u64> {
        validate_proc_mount_id_mask(
            returned_mask,
            mount_id,
            GenerationRootStage::OpenRegularFile,
        )
    }

    #[cfg(target_os = "linux")]
    fn proc_bridge_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
            GenerationRootErrorKind::ResourceLimit
        } else if error == Errno::NOSYS || error == Errno::PERM || error == Errno::ACCESS {
            GenerationRootErrorKind::UnsupportedKernelFeature
        } else if error == Errno::NOENT
            || error == Errno::NOTDIR
            || error == Errno::LOOP
            || error == Errno::XDEV
            || error == Errno::STALE
        {
            GenerationRootErrorKind::ObjectChanged
        } else {
            GenerationRootErrorKind::Io
        };
        GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
    }

    #[cfg(target_os = "linux")]
    fn proc_root_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
            GenerationRootErrorKind::ResourceLimit
        } else if error == Errno::NOSYS
            || error == Errno::PERM
            || error == Errno::ACCESS
            || error == Errno::NOENT
            || error == Errno::NOTDIR
            || error == Errno::LOOP
            || error == Errno::XDEV
        {
            GenerationRootErrorKind::UnsupportedKernelFeature
        } else {
            GenerationRootErrorKind::Io
        };
        GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn proc_bridge_error_for_test(error: Errno) -> GenerationRootError {
        proc_bridge_error(GenerationRootStage::OpenRegularFile, error)
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn proc_root_error_for_test(error: Errno) -> GenerationRootError {
        proc_root_error(GenerationRootStage::OpenRegularFile, error)
    }

    #[cfg(target_os = "linux")]
    fn route_mount_identity(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<[u8; 32]> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRouteMountIdentity)?;
        let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
        let extended = linux_statx(
            descriptor,
            stage,
            linux_filesystem_is_btrfs(filesystem_stat.f_type),
        )?;
        let identity =
            linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRouteMountIdentity)?;
        Ok(identity)
    }

    #[cfg(target_os = "linux")]
    fn qualify_filesystem(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<(QualifiedFilesystem, [u8; 32])> {
        use rustix::fs::StatVfsMountFlags;

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFilesystemQualification)?;
        let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
        let mount_stat = fstatvfs(descriptor).map_err(|error| os_error(stage, error))?;
        if mount_stat.f_flag.contains(StatVfsMountFlags::RDONLY) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ReadOnlyFilesystem,
                stage,
            ));
        }
        let extended = linux_statx(
            descriptor,
            stage,
            linux_filesystem_is_btrfs(filesystem_stat.f_type),
        )?;
        let filesystem =
            linux_filesystem_profile_at_stage(filesystem_stat.f_type, extended.stx_mnt_id, stage)?;
        let mount_identity =
            linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
        let qualified = (filesystem, mount_identity);
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFilesystemQualification)?;
        Ok(qualified)
    }

    #[cfg(target_os = "linux")]
    fn object_witness(
        descriptor: &OwnedFd,
        expected_filesystem: QualifiedFilesystem,
        expected_mount_identity: [u8; 32],
        stage: GenerationRootStage,
    ) -> GenerationRootResult<GenerationRootObjectWitness> {
        use rustix::fs::StatxFlags;

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeObjectWitness)?;
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
        let extended = linux_statx(
            descriptor,
            stage,
            linux_filesystem_is_btrfs(filesystem_stat.f_type),
        )?;
        let mask = StatxFlags::from_bits_retain(extended.stx_mask);
        if !mask.contains(StatxFlags::BASIC_STATS) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            ));
        }
        validate_linux_writable_attributes(
            extended.stx_attributes,
            extended.stx_attributes_mask,
            stage,
        )?;
        if !linux_filesystem_type_matches(expected_filesystem, filesystem_stat.f_type) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::CrossDevice,
                stage,
            ));
        }
        let mount_identity =
            linux_filesystem_namespace_identity(filesystem_stat.f_type, &extended, stage)?;
        if mount_identity != expected_mount_identity {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::CrossDevice,
                stage,
            ));
        }
        let size = u64::try_from(stat.st_size)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        let mode = stat_mode_as_u32(stat.st_mode);
        let inode = stat_inode_as_u64(stat.st_ino);
        let hard_links = stat_link_count_as_u64(stat.st_nlink);
        let modified_seconds = stat_seconds_as_i64(stat.st_mtime);
        let changed_seconds = stat_seconds_as_i64(stat.st_ctime);
        let accessed_seconds = stat_seconds_as_i64(stat.st_atime);
        let modified_nanoseconds = i64::try_from(stat.st_mtime_nsec)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        let changed_nanoseconds = i64::try_from(stat.st_ctime_nsec)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        let accessed_nanoseconds = i64::try_from(stat.st_atime_nsec)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        let stat_device_major = rustix::fs::major(stat.st_dev);
        let stat_device_minor = rustix::fs::minor(stat.st_dev);
        if !linux_device_components_match(
            stat_device_major,
            stat_device_minor,
            extended.stx_dev_major,
            extended.stx_dev_minor,
        ) || u32::from(extended.stx_mode) != mode
            || u64::from(extended.stx_nlink) != hard_links
            || extended.stx_uid != stat.st_uid
            || extended.stx_gid != stat.st_gid
            || extended.stx_ino != inode
            || extended.stx_size != size
            || extended.stx_mtime.tv_sec != modified_seconds
            || i64::from(extended.stx_mtime.tv_nsec) != modified_nanoseconds
            || extended.stx_ctime.tv_sec != changed_seconds
            || i64::from(extended.stx_ctime.tv_nsec) != changed_nanoseconds
            || extended.stx_atime.tv_sec != accessed_seconds
            || i64::from(extended.stx_atime.tv_nsec) != accessed_nanoseconds
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        let witness = GenerationRootObjectWitness {
            device: stat_device_as_u64(stat.st_dev),
            inode,
            mode,
            hard_links,
            uid: stat.st_uid,
            gid: stat.st_gid,
            byte_len: size,
            modified_seconds,
            modified_nanoseconds,
            changed_seconds,
            changed_nanoseconds,
            accessed_seconds,
            accessed_nanoseconds,
            mount_identity,
            filesystem: expected_filesystem,
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterObjectWitness)?;
        Ok(witness)
    }

    #[cfg(target_os = "linux")]
    const fn linux_device_components_match(
        legacy_device_major: u32,
        legacy_device_minor: u32,
        extended_device_major: u32,
        extended_device_minor: u32,
    ) -> bool {
        legacy_device_major == extended_device_major && legacy_device_minor == extended_device_minor
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) const fn linux_device_components_match_for_test(
        legacy_device_major: u32,
        legacy_device_minor: u32,
        extended_device_major: u32,
        extended_device_minor: u32,
    ) -> bool {
        linux_device_components_match(
            legacy_device_major,
            legacy_device_minor,
            extended_device_major,
            extended_device_minor,
        )
    }

    #[cfg(target_os = "linux")]
    fn validate_linux_writable_attributes(
        attributes: rustix::fs::StatxAttributes,
        attributes_mask: rustix::fs::StatxAttributes,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        use rustix::fs::StatxAttributes;

        let required = StatxAttributes::IMMUTABLE | StatxAttributes::APPEND;
        if !attributes_mask.contains(required) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            )
            .with_counts(required.bits(), attributes_mask.bits() & required.bits()));
        }
        let observed = attributes & required;
        if observed.is_empty() {
            Ok(())
        } else {
            Err(
                GenerationRootError::new(GenerationRootErrorKind::WriteRestrictedObject, stage)
                    .with_counts(0, observed.bits()),
            )
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_linux_writable_attributes_for_test(
        attributes: rustix::fs::StatxAttributes,
        attributes_mask: rustix::fs::StatxAttributes,
    ) -> GenerationRootResult<()> {
        validate_linux_writable_attributes(
            attributes,
            attributes_mask,
            GenerationRootStage::QualifyFilesystem,
        )
    }

    #[cfg(target_os = "linux")]
    fn linux_statx(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
        request_subvolume: bool,
    ) -> GenerationRootResult<rustix::fs::Statx> {
        use rustix::fs::{AtFlags, StatxFlags, statx};

        let mut requested = StatxFlags::BASIC_STATS | StatxFlags::MNT_ID;
        if request_subvolume {
            requested |= StatxFlags::from_bits_retain(LINUX_STATX_SUBVOLUME_MASK);
        }
        loop {
            match statx(
                descriptor,
                "",
                AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
                requested,
            ) {
                Ok(extended) => return Ok(extended),
                Err(error) if error == Errno::INTR => {}
                Err(error) => {
                    return Err(
                        if error == Errno::NOSYS
                            || error == Errno::PERM
                            || (request_subvolume && error == Errno::INVAL)
                        {
                            GenerationRootError::new(
                                GenerationRootErrorKind::UnsupportedKernelFeature,
                                stage,
                            )
                            .with_raw_os_error(error.raw_os_error())
                        } else if error == Errno::MFILE
                            || error == Errno::NFILE
                            || error == Errno::NOMEM
                        {
                            GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                                .with_raw_os_error(error.raw_os_error())
                        } else {
                            os_error(stage, error)
                        },
                    );
                }
            }
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn linux_filesystem_profile(
        raw_type: rustix::fs::FsWord,
        mount_id: u64,
    ) -> GenerationRootResult<QualifiedFilesystem> {
        linux_filesystem_profile_at_stage(
            raw_type,
            mount_id,
            GenerationRootStage::QualifyFilesystem,
        )
    }

    #[cfg(target_os = "linux")]
    fn linux_filesystem_profile_at_stage(
        raw_type: rustix::fs::FsWord,
        mount_id: u64,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<QualifiedFilesystem> {
        let raw = i128::from(raw_type);
        let (expected_name, profile) = if raw == i128::from(libc::EXT4_SUPER_MAGIC) {
            (b"ext4".as_slice(), QualifiedFilesystem::LinuxExt4)
        } else if raw == i128::from(libc::BTRFS_SUPER_MAGIC) {
            (b"btrfs".as_slice(), QualifiedFilesystem::LinuxBtrfs)
        } else {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        };
        let observed_name = linux_mountinfo_filesystem(mount_id, stage)?;
        if observed_name.as_slice() != expected_name {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        Ok(profile)
    }

    #[cfg(target_os = "linux")]
    fn linux_filesystem_type_matches(
        expected: QualifiedFilesystem,
        raw_type: rustix::fs::FsWord,
    ) -> bool {
        let raw = i128::from(raw_type);
        match expected {
            QualifiedFilesystem::LinuxExt4 => raw == i128::from(libc::EXT4_SUPER_MAGIC),
            QualifiedFilesystem::LinuxBtrfs => raw == i128::from(libc::BTRFS_SUPER_MAGIC),
            QualifiedFilesystem::AppleApfs => false,
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_filesystem_is_btrfs(raw_type: rustix::fs::FsWord) -> bool {
        i128::from(raw_type) == i128::from(libc::BTRFS_SUPER_MAGIC)
    }

    #[cfg(target_os = "linux")]
    fn linux_filesystem_namespace_identity(
        raw_type: rustix::fs::FsWord,
        extended: &rustix::fs::Statx,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<[u8; 32]> {
        use rustix::fs::StatxFlags;

        let mask = StatxFlags::from_bits_retain(extended.stx_mask);
        if !mask.contains(StatxFlags::MNT_ID) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedKernelFeature,
                stage,
            ));
        }
        let subvolume_id = if linux_filesystem_is_btrfs(raw_type) {
            let subvolume_mask = StatxFlags::from_bits_retain(LINUX_STATX_SUBVOLUME_MASK);
            if !mask.contains(subvolume_mask) {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::UnsupportedKernelFeature,
                    stage,
                ));
            }
            // This is the kernel-authenticated value for the already-open
            // descriptor (`AT_EMPTY_PATH`), not a pathname re-resolution.
            // Keeping it as the sole source of truth also avoids imposing
            // consumer-side bindgen and libclang prerequisites for Btrfs.
            Some(extended.stx_subvol)
        } else {
            None
        };
        Ok(linux_filesystem_namespace_digest(
            raw_type,
            extended.stx_mnt_id,
            subvolume_id,
        ))
    }

    #[cfg(target_os = "linux")]
    fn validate_root_writable(
        descriptor: &OwnedFd,
        filesystem: QualifiedFilesystem,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        use rustix::fs::{Access, AtFlags, accessat};

        if filesystem != QualifiedFilesystem::LinuxBtrfs {
            return Ok(());
        }
        match accessat(
            descriptor,
            ".",
            Access::WRITE_OK,
            AtFlags::EACCESS | AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(()) => Ok(()),
            Err(error) if error == Errno::ROFS => Err(GenerationRootError::new(
                GenerationRootErrorKind::ReadOnlyFilesystem,
                stage,
            )
            .with_raw_os_error(error.raw_os_error())),
            Err(error) if error == Errno::NOSYS || error == Errno::PERM => Err(
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedKernelFeature, stage)
                    .with_raw_os_error(error.raw_os_error()),
            ),
            Err(error) => Err(os_error(stage, error)),
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_mountinfo_filesystem(
        mount_id: u64,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<Vec<u8>> {
        use rustix::fs::{ResolveFlags, openat2};
        use rustix::io::read;

        let process_id = std::process::id();
        let thread_id = rustix::thread::gettid();
        let path = linux_task_mountinfo_route(process_id, thread_id.as_raw_pid());
        validate_process_identity(process_id, stage)?;
        let (proc_root, _proc_mount_id) = open_verified_proc_root(process_id, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeMountInfoOpen)?;
        let descriptor = openat2(
            &proc_root,
            &path,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
            ResolveFlags::BENEATH
                | ResolveFlags::NO_SYMLINKS
                | ResolveFlags::NO_MAGICLINKS
                | ResolveFlags::NO_XDEV,
        )
        .map_err(|error| linux_mountinfo_error(stage, error))?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterMountInfoOpen)?;
        validate_process_identity(process_id, stage)?;
        if rustix::thread::gettid() != thread_id {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                stage,
            ));
        }
        let filesystem_stat =
            fstatfs(&descriptor).map_err(|error| linux_mountinfo_error(stage, error))?;
        if i128::from(filesystem_stat.f_type) != i128::from(libc::PROC_SUPER_MAGIC) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        let stat = fstat(&descriptor).map_err(|error| linux_mountinfo_error(stage, error))?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve(64 * 1024)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        let mut chunk = [0_u8; 8192];
        loop {
            let requested = mountinfo_read_request(bytes.len());
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeMountInfoRead)?;
            let count = match read(&descriptor, &mut chunk[..requested]) {
                Ok(count) => count,
                Err(error) if error == Errno::INTR => continue,
                Err(error) => return Err(linux_mountinfo_error(stage, error)),
            };
            #[cfg(test)]
            test_boundary(TestBoundary::AfterMountInfoRead { byte_count: count })?;
            validate_process_identity(process_id, stage)?;
            if rustix::thread::gettid() != thread_id {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::ObjectChanged,
                    stage,
                ));
            }
            if count == 0 {
                break;
            }
            validate_mountinfo_append(bytes.len(), count, stage)?;
            bytes.extend_from_slice(&chunk[..count]);
        }
        validate_proc_self_binding(
            &proc_root,
            process_id,
            ProcSelfBindingPhase::MountInfoRead,
            stage,
        )?;
        parse_mountinfo_filesystem_at_stage(&bytes, mount_id, stage)
    }

    #[cfg(target_os = "linux")]
    fn linux_mountinfo_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        let kind = if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
            GenerationRootErrorKind::ResourceLimit
        } else {
            GenerationRootErrorKind::Io
        };
        GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn linux_mountinfo_error_for_test(error: Errno) -> GenerationRootError {
        linux_mountinfo_error(GenerationRootStage::QualifyFilesystem, error)
    }

    #[cfg(target_os = "linux")]
    fn linux_task_mountinfo_route(process_id: u32, thread_id: i32) -> std::path::PathBuf {
        std::path::PathBuf::from(process_id.to_string())
            .join("task")
            .join(thread_id.to_string())
            .join("mountinfo")
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn linux_task_mountinfo_route_for_test(
        process_id: u32,
        thread_id: i32,
    ) -> std::path::PathBuf {
        linux_task_mountinfo_route(process_id, thread_id)
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn linux_mountinfo_filesystem_for_test(
        mount_id: u64,
    ) -> GenerationRootResult<Vec<u8>> {
        linux_mountinfo_filesystem(mount_id, GenerationRootStage::QualifyFilesystem)
    }

    #[cfg(target_os = "linux")]
    fn mountinfo_read_request(stored_bytes: usize) -> usize {
        GENERATION_ROOT_MAX_MOUNTINFO_BYTES
            .saturating_sub(stored_bytes)
            .saturating_add(1)
            .min(8192)
    }

    #[cfg(target_os = "linux")]
    fn validate_mountinfo_append(
        stored_bytes: usize,
        incoming_bytes: usize,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let observed = stored_bytes.saturating_add(incoming_bytes);
        if observed > GENERATION_ROOT_MAX_MOUNTINFO_BYTES {
            Err(
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                    .with_counts(
                        usize_as_u64(GENERATION_ROOT_MAX_MOUNTINFO_BYTES),
                        usize_as_u64(observed),
                    ),
            )
        } else {
            Ok(())
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_mountinfo_bound_for_test(
        stored_bytes: usize,
        incoming_bytes: usize,
    ) -> (usize, GenerationRootResult<()>) {
        (
            mountinfo_read_request(stored_bytes),
            validate_mountinfo_append(
                stored_bytes,
                incoming_bytes,
                GenerationRootStage::QualifyFilesystem,
            ),
        )
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn parse_mountinfo_filesystem(
        bytes: &[u8],
        mount_id: u64,
    ) -> GenerationRootResult<Vec<u8>> {
        parse_mountinfo_filesystem_at_stage(bytes, mount_id, GenerationRootStage::QualifyFilesystem)
    }

    #[cfg(target_os = "linux")]
    fn parse_mountinfo_filesystem_at_stage(
        bytes: &[u8],
        mount_id: u64,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<Vec<u8>> {
        let mut matched = None;
        for line in bytes.split(|byte| *byte == b'\n') {
            let mut fields = line.split(|byte| *byte == b' ');
            let Some(observed_mount_id) = fields.next() else {
                continue;
            };
            if parse_ascii_u64(observed_mount_id) != Some(mount_id) {
                continue;
            }
            let filesystem = loop {
                let field = fields.next().ok_or_else(|| {
                    GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
                })?;
                if field == b"-" {
                    break fields.next().ok_or_else(|| {
                        GenerationRootError::new(
                            GenerationRootErrorKind::UnsupportedFilesystem,
                            stage,
                        )
                    })?;
                }
            };
            let mount_source = fields.next().ok_or_else(|| {
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
            })?;
            let super_options = fields.next().ok_or_else(|| {
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
            })?;
            if filesystem.is_empty() || mount_source.is_empty() || super_options.is_empty() {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::UnsupportedFilesystem,
                    stage,
                ));
            }
            let mut owned_filesystem = Vec::new();
            owned_filesystem
                .try_reserve_exact(filesystem.len())
                .map_err(|_| {
                    GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                })?;
            owned_filesystem.extend_from_slice(filesystem);
            if matched.replace(owned_filesystem).is_some() {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::UnsupportedFilesystem,
                    stage,
                ));
            }
        }
        matched.ok_or_else(|| {
            GenerationRootError::new(GenerationRootErrorKind::UnsupportedFilesystem, stage)
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_ascii_u64(bytes: &[u8]) -> Option<u64> {
        if bytes.is_empty() {
            return None;
        }
        let mut value = 0_u64;
        for byte in bytes {
            let digit = byte.checked_sub(b'0')?;
            if digit > 9 {
                return None;
            }
            value = value.checked_mul(10)?.checked_add(u64::from(digit))?;
        }
        Some(value)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn linux_filesystem_namespace_digest(
        raw_type: rustix::fs::FsWord,
        mount_id: u64,
        subvolume_id: Option<u64>,
    ) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"frankensearch.generation-root.linux-namespace.v2");
        digest.update(i128::from(raw_type).to_le_bytes());
        digest.update(mount_id.to_le_bytes());
        match subvolume_id {
            Some(subvolume_id) => {
                digest.update([1]);
                digest.update(subvolume_id.to_le_bytes());
            }
            None => digest.update([0]),
        }
        digest.finalize().into()
    }

    #[cfg(target_os = "linux")]
    const fn stat_device_as_u64(device: u64) -> u64 {
        device
    }

    #[cfg(target_os = "linux")]
    fn linux_component_open_error(
        parent: &OwnedFd,
        component: &OsStr,
        stage: GenerationRootStage,
        index: usize,
        error: Errno,
    ) -> GenerationRootError {
        use rustix::fs::{AtFlags, statat};

        if error == Errno::NOTDIR
            && statat(parent, component, AtFlags::SYMLINK_NOFOLLOW)
                .is_ok_and(|stat| FileType::from_raw_mode(stat.st_mode) == FileType::Symlink)
        {
            return GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
                .at_component(index)
                .with_raw_os_error(error.raw_os_error());
        }
        linux_open_error(stage, error).at_component(index)
    }

    #[cfg(target_os = "linux")]
    fn linux_open_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        let kind = if error == Errno::LOOP {
            GenerationRootErrorKind::SymbolicLink
        } else if error == Errno::NOTDIR {
            GenerationRootErrorKind::NotDirectory
        } else if error == Errno::XDEV {
            GenerationRootErrorKind::CrossDevice
        } else if error == Errno::NOSYS || error == Errno::PERM {
            GenerationRootErrorKind::UnsupportedKernelFeature
        } else if error == Errno::MFILE || error == Errno::NFILE || error == Errno::NOMEM {
            GenerationRootErrorKind::ResourceLimit
        } else {
            GenerationRootErrorKind::Io
        };
        GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn linux_open_error_for_test(error: Errno) -> GenerationRootError {
        linux_open_error(GenerationRootStage::OpenRegularFile, error)
    }

    #[cfg(target_os = "linux")]
    fn inspect_acl_if_required(
        descriptor: &OwnedFd,
        witness: GenerationRootObjectWitness,
        _stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let before = object_witness(
            descriptor,
            witness.filesystem,
            witness.mount_identity,
            GenerationRootStage::InspectAcl,
        )?;
        if !before.mutation_stable_eq(witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::InspectAcl,
            ));
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeAclRead)?;
        let presence =
            crate::fd_acl::extended_acl_presence(descriptor.as_fd()).map_err(|error| {
                GenerationRootError::new(
                    GenerationRootErrorKind::AclRejected,
                    GenerationRootStage::InspectAcl,
                )
                .with_raw_os_error(error.raw_os_error().unwrap_or(libc::EIO))
            })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterAclRead)?;
        match presence {
            crate::fd_acl::ExtendedAclPresence::Absent => {}
            crate::fd_acl::ExtendedAclPresence::Present => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::AclRejected,
                    GenerationRootStage::InspectAcl,
                ));
            }
        }
        let after = object_witness(
            descriptor,
            witness.filesystem,
            witness.mount_identity,
            GenerationRootStage::InspectAcl,
        )?;
        if !witness.mutation_stable_eq(after) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::InspectAcl,
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn sync_file_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFileSync)?;
        rustix::fs::fsync(descriptor).map_err(|error| {
            GenerationRootError::new(
                GenerationRootErrorKind::DurabilityUnavailable,
                GenerationRootStage::SyncRegularFile,
            )
            .with_raw_os_error(error.raw_os_error())
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFileSync)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn sync_directory_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeDirectorySync)?;
        rustix::fs::fsync(descriptor).map_err(|error| {
            GenerationRootError::new(
                GenerationRootErrorKind::DurabilityUnavailable,
                GenerationRootStage::SyncDirectory,
            )
            .with_raw_os_error(error.raw_os_error())
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterDirectorySync)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn open_absolute_route(
        route: &AbsoluteRoute,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OpenedRoute> {
        use rustix::fs::CWD;

        let flags = OFlags::RDONLY
            | OFlags::DIRECTORY
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK;
        let mut descriptor = openat(CWD, Path::new("/"), flags, Mode::empty())
            .map_err(|error| macos_open_error(stage, error))?;
        let mut route_identities = Vec::new();
        route_identities
            .try_reserve(route.components.len() + 1)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage))?;
        route_identities.push(qualify_absolute_ancestor(&descriptor, stage, None)?);
        for (index, component) in route.components.iter().enumerate() {
            #[cfg(test)]
            test_boundary(TestBoundary::BeforeRootComponentOpen { index })?;
            verify_exact_directory_entry(&descriptor, component, stage, index)?;
            let next = openat(&descriptor, component, flags, Mode::empty()).map_err(|error| {
                macos_component_open_error(&descriptor, component, stage, index, error)
            })?;
            verify_exact_directory_entry(&descriptor, component, stage, index)?;
            descriptor = next;
            let stat = fstat(&descriptor).map_err(|error| os_error(stage, error))?;
            if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
                return Err(
                    GenerationRootError::new(GenerationRootErrorKind::NotDirectory, stage)
                        .at_component(index),
                );
            }
            let identity = if index + 1 == route.components.len() {
                basic_identity(&descriptor, stage)?
            } else {
                qualify_absolute_ancestor(&descriptor, stage, Some(index))?
            };
            #[cfg(test)]
            test_boundary(TestBoundary::AfterRootComponentOpen { index })?;
            route_identities.push(identity);
        }
        Ok(OpenedRoute {
            descriptor,
            route_identities,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn open_relative_directory_component(
        parent: &OwnedFd,
        component: &OsStr,
        index: usize,
        root: GenerationRootObjectWitness,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<OwnedFd> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRelativeDirectoryOpen { index })?;
        verify_exact_directory_entry(parent, component, stage, index)?;
        let descriptor = openat(
            parent,
            component,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::CLOEXEC
                | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|error| macos_component_open_error(parent, component, stage, index, error))?;
        verify_exact_directory_entry(parent, component, stage, index)?;
        let witness = object_witness(&descriptor, root.filesystem, root.mount_identity, stage)?;
        if witness.mount_identity != root.mount_identity {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::CrossDevice, stage)
                    .at_component(index),
            );
        }
        validate_private_directory(&descriptor, witness, stage)?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRelativeDirectoryOpen { index })?;
        Ok(descriptor)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn route_mount_identity(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<[u8; 32]> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeRouteMountIdentity)?;
        let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        let identity = macos_mount_identity(&filesystem_stat, stat_device_as_u64(stat.st_dev));
        #[cfg(test)]
        test_boundary(TestBoundary::AfterRouteMountIdentity)?;
        Ok(identity)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn qualify_filesystem(
        descriptor: &OwnedFd,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<(QualifiedFilesystem, [u8; 32])> {
        use rustix::fs::StatVfsMountFlags;

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFilesystemQualification)?;
        let filesystem_stat = fstatfs(descriptor).map_err(|error| os_error(stage, error))?;
        if macos_fixed_name(&filesystem_stat.f_fstypename) != b"apfs" {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        let flags = i64::from(filesystem_stat.f_flags);
        if flags & i64::from(libc::MNT_LOCAL) == 0 {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::UnsupportedFilesystem,
                stage,
            ));
        }
        if flags & i64::from(libc::MNT_RDONLY) != 0
            || flags & i64::from(libc::MNT_IGNORE_OWNERSHIP) != 0
        {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ReadOnlyFilesystem,
                stage,
            ));
        }
        let mount_stat = fstatvfs(descriptor).map_err(|error| os_error(stage, error))?;
        if mount_stat.f_flag.contains(StatVfsMountFlags::RDONLY) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ReadOnlyFilesystem,
                stage,
            ));
        }
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        let qualified = (
            QualifiedFilesystem::AppleApfs,
            macos_mount_identity(&filesystem_stat, stat_device_as_u64(stat.st_dev)),
        );
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFilesystemQualification)?;
        Ok(qualified)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn validate_root_writable(
        descriptor: &OwnedFd,
        filesystem: QualifiedFilesystem,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let (observed, _) = qualify_filesystem(descriptor, stage)?;
        if observed != filesystem {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::CrossDevice,
                stage,
            ));
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn object_witness(
        descriptor: &OwnedFd,
        expected_filesystem: QualifiedFilesystem,
        expected_mount_identity: [u8; 32],
        stage: GenerationRootStage,
    ) -> GenerationRootResult<GenerationRootObjectWitness> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeObjectWitness)?;
        let stat = fstat(descriptor).map_err(|error| os_error(stage, error))?;
        let (filesystem, mount_identity) = qualify_filesystem(descriptor, stage)?;
        if filesystem != expected_filesystem || mount_identity != expected_mount_identity {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::CrossDevice,
                stage,
            ));
        }
        validate_macos_object_flags(stat.st_flags, stage)?;
        let size = u64::try_from(stat.st_size)
            .map_err(|_| GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage))?;
        let witness = GenerationRootObjectWitness {
            device: stat_device_as_u64(stat.st_dev),
            inode: stat_inode_as_u64(stat.st_ino),
            mode: stat_mode_as_u32(stat.st_mode),
            hard_links: stat_link_count_as_u64(stat.st_nlink),
            uid: stat.st_uid,
            gid: stat.st_gid,
            byte_len: size,
            modified_seconds: stat_seconds_as_i64(stat.st_mtime),
            modified_nanoseconds: stat.st_mtime_nsec,
            changed_seconds: stat_seconds_as_i64(stat.st_ctime),
            changed_nanoseconds: stat.st_ctime_nsec,
            accessed_seconds: stat_seconds_as_i64(stat.st_atime),
            accessed_nanoseconds: stat.st_atime_nsec,
            mount_identity,
            filesystem,
        };
        #[cfg(test)]
        test_boundary(TestBoundary::AfterObjectWitness)?;
        Ok(witness)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn validate_macos_object_flags(
        flags: u32,
        stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        if flags == 0 {
            Ok(())
        } else {
            Err(
                GenerationRootError::new(GenerationRootErrorKind::UnsupportedObjectFlags, stage)
                    .with_counts(0, u64::from(flags)),
            )
        }
    }

    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn validate_macos_object_flags_for_test(flags: u32) -> GenerationRootResult<()> {
        validate_macos_object_flags(flags, GenerationRootStage::OpenRegularFile)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn inspect_acl_if_required(
        descriptor: &OwnedFd,
        witness: GenerationRootObjectWitness,
        _stage: GenerationRootStage,
    ) -> GenerationRootResult<()> {
        let before = object_witness(
            descriptor,
            witness.filesystem,
            witness.mount_identity,
            GenerationRootStage::InspectAcl,
        )?;
        if !before.mutation_stable_eq(witness) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::InspectAcl,
            ));
        }
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeAclRead)?;
        let presence =
            crate::fd_acl::extended_acl_presence(descriptor.as_fd()).map_err(|error| {
                GenerationRootError::new(
                    GenerationRootErrorKind::AclRejected,
                    GenerationRootStage::InspectAcl,
                )
                .with_raw_os_error(error.raw_os_error().unwrap_or(libc::EIO))
            })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterAclRead)?;
        match presence {
            crate::fd_acl::ExtendedAclPresence::Absent => {}
            crate::fd_acl::ExtendedAclPresence::Present => {
                return Err(GenerationRootError::new(
                    GenerationRootErrorKind::AclRejected,
                    GenerationRootStage::InspectAcl,
                ));
            }
        }
        let after = object_witness(
            descriptor,
            witness.filesystem,
            witness.mount_identity,
            GenerationRootStage::InspectAcl,
        )?;
        if !witness.mutation_stable_eq(after) {
            return Err(GenerationRootError::new(
                GenerationRootErrorKind::ObjectChanged,
                GenerationRootStage::InspectAcl,
            ));
        }
        Ok(())
    }

    pub(super) fn verify_exact_directory_entry(
        parent: &OwnedFd,
        component: &OsStr,
        stage: GenerationRootStage,
        index: usize,
    ) -> GenerationRootResult<()> {
        use rustix::fs::Dir;

        #[cfg(test)]
        test_boundary(TestBoundary::BeforeExactNameEnumeration)?;
        let mut directory = Dir::read_from(parent).map_err(|error| {
            GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                .at_component(index)
                .with_raw_os_error(error.raw_os_error())
        })?;
        let expected = component.as_bytes();
        let mut entry_count = 0_usize;
        let mut name_bytes = 0_usize;
        let mut exact_matches = 0_u8;
        while let Some(entry) = directory.read() {
            let entry = entry.map_err(|error| {
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index)
                    .with_raw_os_error(error.raw_os_error())
            })?;
            entry_count = entry_count.saturating_add(1);
            let name = entry.file_name().to_bytes();
            name_bytes = name_bytes.saturating_add(name.len());
            validate_directory_scan_counts(entry_count, name_bytes, stage, index)?;
            if name == expected {
                exact_matches = exact_matches.saturating_add(1);
            }
        }
        #[cfg(test)]
        test_boundary(TestBoundary::AfterExactNameEnumeration)?;
        if exact_matches != 1 {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ObjectChanged, stage)
                    .at_component(index)
                    .with_counts(1, u64::from(exact_matches)),
            );
        }
        Ok(())
    }

    fn validate_directory_scan_counts(
        entry_count: usize,
        name_bytes: usize,
        stage: GenerationRootStage,
        index: usize,
    ) -> GenerationRootResult<()> {
        if entry_count > GENERATION_ROOT_MAX_DIRECTORY_ENTRIES {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                    .at_component(index)
                    .with_counts(
                        usize_as_u64(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES),
                        usize_as_u64(entry_count),
                    ),
            );
        }
        if name_bytes > GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES {
            return Err(
                GenerationRootError::new(GenerationRootErrorKind::ResourceLimit, stage)
                    .at_component(index)
                    .with_counts(
                        usize_as_u64(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES),
                        usize_as_u64(name_bytes),
                    ),
            );
        }
        Ok(())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn validate_directory_scan_counts_for_test(
        entry_count: usize,
        name_bytes: usize,
    ) -> GenerationRootResult<()> {
        validate_directory_scan_counts(
            entry_count,
            name_bytes,
            GenerationRootStage::OpenRelativeDirectory,
            7,
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn macos_fixed_name<const N: usize>(value: &[libc::c_char; N]) -> Vec<u8> {
        value
            .iter()
            .map(|character| character.to_ne_bytes()[0])
            .take_while(|byte| *byte != 0)
            .collect()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn macos_mount_identity(stat: &rustix::fs::StatFs, device: u64) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"frankensearch.generation-root.macos-mount.v1");
        digest.update(device.to_le_bytes());
        for character in &stat.f_fstypename {
            digest.update(character.to_ne_bytes());
        }
        for character in &stat.f_mntonname {
            digest.update(character.to_ne_bytes());
        }
        for character in &stat.f_mntfromname {
            digest.update(character.to_ne_bytes());
        }
        digest.finalize().into()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn stat_device_as_u64(device: i32) -> u64 {
        u64::from_ne_bytes(i64::from(device).to_ne_bytes())
    }

    #[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn stat_device_as_u64_for_test(device: i32) -> u64 {
        stat_device_as_u64(device)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn macos_open_error(stage: GenerationRootStage, error: Errno) -> GenerationRootError {
        let kind = if error == Errno::LOOP {
            GenerationRootErrorKind::SymbolicLink
        } else if error == Errno::NOTDIR {
            GenerationRootErrorKind::NotDirectory
        } else {
            GenerationRootErrorKind::Io
        };
        GenerationRootError::new(kind, stage).with_raw_os_error(error.raw_os_error())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn macos_component_open_error(
        parent: &OwnedFd,
        component: &OsStr,
        stage: GenerationRootStage,
        index: usize,
        error: Errno,
    ) -> GenerationRootError {
        use rustix::fs::{AtFlags, statat};

        if matches!(error, Errno::NOTDIR)
            && statat(parent, component, AtFlags::SYMLINK_NOFOLLOW)
                .is_ok_and(|stat| FileType::from_raw_mode(stat.st_mode) == FileType::Symlink)
        {
            return GenerationRootError::new(GenerationRootErrorKind::SymbolicLink, stage)
                .at_component(index)
                .with_raw_os_error(error.raw_os_error());
        }
        macos_open_error(stage, error).at_component(index)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn sync_file_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeFileSync)?;
        rustix::fs::fcntl_fullfsync(descriptor).map_err(|error| {
            GenerationRootError::new(
                GenerationRootErrorKind::DurabilityUnavailable,
                GenerationRootStage::SyncRegularFile,
            )
            .with_raw_os_error(error.raw_os_error())
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterFileSync)?;
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn sync_directory_descriptor(descriptor: &OwnedFd) -> GenerationRootResult<()> {
        #[cfg(test)]
        test_boundary(TestBoundary::BeforeDirectorySync)?;
        rustix::fs::fcntl_fullfsync(descriptor).map_err(|error| {
            GenerationRootError::new(
                GenerationRootErrorKind::DurabilityUnavailable,
                GenerationRootStage::SyncDirectory,
            )
            .with_raw_os_error(error.raw_os_error())
        })?;
        #[cfg(test)]
        test_boundary(TestBoundary::AfterDirectorySync)?;
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))))]
#[allow(dead_code)] // Typed zero-I/O stubs intentionally have no native consumers yet.
mod platform {
    use super::{
        GenerationFileExpectation, GenerationRootError, GenerationRootErrorKind,
        GenerationRootLockMode, GenerationRootObjectWitness, GenerationRootResult,
        GenerationRootStage,
    };
    use std::path::Path;

    #[derive(Clone)]
    pub(super) enum RelativePath {}
    pub(super) enum RootHandle {}
    pub(super) enum FileHandle {}
    pub(super) enum ControlHandle {}
    pub(super) enum LockHandle {}

    pub(super) struct AnchorImage {
        pub(super) bytes: std::sync::Arc<[u8]>,
        pub(super) witness: GenerationRootObjectWitness,
        pub(super) sha256: [u8; 32],
    }

    pub(super) struct FileAdmission {
        pub(super) handle: FileHandle,
        pub(super) bytes: std::sync::Arc<[u8]>,
        pub(super) witness: GenerationRootObjectWitness,
        pub(super) sha256: [u8; 32],
    }

    const fn unsupported<T>() -> GenerationRootResult<T> {
        Err(GenerationRootError::new(
            GenerationRootErrorKind::UnsupportedPlatform,
            GenerationRootStage::PlatformGate,
        ))
    }

    pub(super) fn parse_relative(_path: &Path) -> GenerationRootResult<RelativePath> {
        unsupported()
    }

    pub(super) const fn relative_component_count(path: &RelativePath) -> usize {
        match *path {}
    }

    pub(super) fn admit_root(_path: &Path) -> GenerationRootResult<RootHandle> {
        unsupported()
    }

    pub(super) const fn root_witness(root: &RootHandle) -> GenerationRootObjectWitness {
        match *root {}
    }

    pub(super) fn revalidate_root(root: &RootHandle) -> GenerationRootResult<()> {
        match *root {}
    }

    pub(super) fn admit_file(
        root: &RootHandle,
        _path: &RelativePath,
        _expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<FileAdmission> {
        match *root {}
    }

    pub(super) fn sync_root_directory(root: &RootHandle) -> GenerationRootResult<()> {
        match *root {}
    }

    pub(super) fn sync_file(
        file: &FileHandle,
        _expected: GenerationRootObjectWitness,
    ) -> GenerationRootResult<()> {
        match *file {}
    }

    pub(super) fn admit_control_file(
        root: &RootHandle,
        _path: &RelativePath,
        _expectation: GenerationFileExpectation,
    ) -> GenerationRootResult<ControlHandle> {
        match *root {}
    }

    pub(super) const fn control_witness(control: &ControlHandle) -> GenerationRootObjectWitness {
        match *control {}
    }

    pub(super) fn control_bytes(control: &ControlHandle) -> std::sync::Arc<[u8]> {
        match *control {}
    }

    pub(super) const fn control_sha256(control: &ControlHandle) -> [u8; 32] {
        match *control {}
    }

    pub(super) fn revalidate_anchor_set(
        root: &RootHandle,
        _lock: &ControlHandle,
        _authority: &ControlHandle,
    ) -> GenerationRootResult<()> {
        match *root {}
    }

    pub(super) fn capture_anchor_image(
        control: &ControlHandle,
    ) -> GenerationRootResult<AnchorImage> {
        match *control {}
    }

    pub(super) fn anchor_images_match(left: &AnchorImage, right: &AnchorImage) -> bool {
        left.sha256 == right.sha256 && left.bytes == right.bytes && left.witness == right.witness
    }

    pub(super) fn validate_anchor_image(
        control: &ControlHandle,
        _expected: &AnchorImage,
    ) -> GenerationRootResult<()> {
        match *control {}
    }

    pub(super) fn try_anchor_lock(
        control: &ControlHandle,
        _mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        match *control {}
    }

    pub(super) fn try_lock(
        control: &ControlHandle,
        _mode: GenerationRootLockMode,
    ) -> GenerationRootResult<LockHandle> {
        match *control {}
    }

    pub(super) const fn lock_witness(lock: &LockHandle) -> GenerationRootObjectWitness {
        match *lock {}
    }

    pub(super) fn lock_bytes(lock: &LockHandle) -> std::sync::Arc<[u8]> {
        match *lock {}
    }

    pub(super) const fn lock_sha256(lock: &LockHandle) -> [u8; 32] {
        match *lock {}
    }

    pub(super) const fn lock_mode(lock: &LockHandle) -> GenerationRootLockMode {
        match *lock {}
    }

    pub(super) fn unlock(lock: LockHandle) -> GenerationRootResult<()> {
        match lock {}
    }

    pub(super) fn sync_lock_file(lock: &LockHandle) -> GenerationRootResult<()> {
        match *lock {}
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GENERATION_ROOT_MAX_CONTROL_FILE_BYTES, GENERATION_ROOT_MAX_FILE_BYTES,
        GenerationFileExpectation, GenerationRootErrorKind, GenerationRootStage,
    };

    #[test]
    fn file_expectation_enforces_roles_digests_and_role_specific_limits() {
        let exact = GenerationFileExpectation::immutable(7, [0x5a; 32])
            .expect("small immutable expectation should be admitted");
        assert_eq!(exact.byte_len(), 7);
        assert_eq!(exact.sha256(), Some([0x5a; 32]));
        assert_eq!(exact.role(), super::GenerationFileRole::ImmutableArtifact);

        let control = GenerationFileExpectation::control(8)
            .expect("small control-file expectation should be admitted");
        assert_eq!(control.byte_len(), 8);
        assert_eq!(control.sha256(), None);
        assert_eq!(control.role(), super::GenerationFileRole::MutableControl);

        GenerationFileExpectation::control(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
            .expect("the exact control repeated-hash ceiling must be accepted");
        GenerationFileExpectation::immutable(GENERATION_ROOT_MAX_FILE_BYTES, [0; 32])
            .expect("the exact immutable owned-image ceiling must be accepted");

        let control_limit = GenerationFileExpectation::control(
            GENERATION_ROOT_MAX_CONTROL_FILE_BYTES.saturating_add(1),
        )
        .expect_err("control length above its repeated-hash ceiling must fail");
        assert_eq!(control_limit.kind(), GenerationRootErrorKind::ResourceLimit);
        assert_eq!(
            control_limit.expected(),
            Some(GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
        );

        let artifact_limit = GenerationFileExpectation::immutable(
            GENERATION_ROOT_MAX_FILE_BYTES.saturating_add(1),
            [0; 32],
        )
        .expect_err("immutable length above the owned-image ceiling must fail");
        assert_eq!(
            artifact_limit.kind(),
            GenerationRootErrorKind::ResourceLimit
        );
        assert_eq!(
            artifact_limit.expected(),
            Some(GENERATION_ROOT_MAX_FILE_BYTES)
        );
    }

    #[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn confined_routes_are_bounded_unambiguous_and_non_utf8_safe() {
        use super::{
            ConfinedGenerationPath, GENERATION_ROOT_MAX_COMPONENT_BYTES,
            GENERATION_ROOT_MAX_COMPONENTS, GENERATION_ROOT_MAX_ROUTE_BYTES,
            QualifiedGenerationDirectory as QualifiedGenerationRoot,
        };
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::path::{Path, PathBuf};

        for invalid in ["", ".", "..", "/absolute", "a/", "a//b", "a/./b", "a/../b"] {
            let error = ConfinedGenerationPath::parse(Path::new(invalid))
                .expect_err("ambiguous confined route must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
            assert_eq!(error.stage(), GenerationRootStage::ParseRelativeRoute);
        }

        let nul = Path::new(OsStr::from_bytes(b"a\0b"));
        assert_eq!(
            ConfinedGenerationPath::parse(nul)
                .expect_err("NUL-containing route must fail")
                .kind(),
            GenerationRootErrorKind::InvalidRoute
        );

        let max_component = PathBuf::from("x".repeat(GENERATION_ROOT_MAX_COMPONENT_BYTES));
        assert_eq!(
            ConfinedGenerationPath::parse(&max_component)
                .expect("maximum component length should be accepted")
                .component_count(),
            1
        );
        let oversized_component =
            PathBuf::from("x".repeat(GENERATION_ROOT_MAX_COMPONENT_BYTES.saturating_add(1)));
        assert_eq!(
            ConfinedGenerationPath::parse(&oversized_component)
                .expect_err("oversized component must fail")
                .kind(),
            GenerationRootErrorKind::InvalidRoute
        );

        let max_components = std::iter::repeat_n("x", GENERATION_ROOT_MAX_COMPONENTS)
            .collect::<Vec<_>>()
            .join("/");
        assert_eq!(
            ConfinedGenerationPath::parse(Path::new(&max_components))
                .expect("maximum component count should be accepted")
                .component_count(),
            GENERATION_ROOT_MAX_COMPONENTS
        );
        let too_many_components =
            std::iter::repeat_n("x", GENERATION_ROOT_MAX_COMPONENTS.saturating_add(1))
                .collect::<Vec<_>>()
                .join("/");
        assert_eq!(
            ConfinedGenerationPath::parse(Path::new(&too_many_components))
                .expect_err("too many components must fail")
                .kind(),
            GenerationRootErrorKind::ResourceLimit
        );

        let non_utf8 = Path::new(OsStr::from_bytes(&[0xff, b'x']));
        assert_eq!(
            ConfinedGenerationPath::parse(non_utf8)
                .expect("non-UTF-8 bytes are valid Unix component bytes")
                .component_count(),
            1
        );

        let exact_relative = [
            std::iter::repeat_n("x".repeat(255), 15).collect::<Vec<_>>(),
            vec!["x".repeat(254), "x".to_owned()],
        ]
        .concat()
        .join("/");
        assert_eq!(exact_relative.len(), GENERATION_ROOT_MAX_ROUTE_BYTES);
        assert_eq!(
            ConfinedGenerationPath::parse(Path::new(&exact_relative))
                .expect("the exact relative-route byte ceiling must be accepted")
                .component_count(),
            17
        );
        let oversized_relative = format!("{exact_relative}x");
        let relative_error = ConfinedGenerationPath::parse(Path::new(&oversized_relative))
            .expect_err("one byte above the relative-route ceiling must fail");
        assert_eq!(relative_error.kind(), GenerationRootErrorKind::InvalidRoute);
        assert_eq!(
            relative_error.stage(),
            GenerationRootStage::ParseRelativeRoute
        );
        assert_eq!(
            relative_error.expected(),
            Some(GENERATION_ROOT_MAX_ROUTE_BYTES as u64)
        );
        assert_eq!(
            relative_error.observed(),
            Some(GENERATION_ROOT_MAX_ROUTE_BYTES.saturating_add(1) as u64)
        );

        let exact_absolute_body = [
            std::iter::repeat_n("x".repeat(255), 15).collect::<Vec<_>>(),
            vec!["x".repeat(253), "x".to_owned()],
        ]
        .concat()
        .join("/");
        let exact_absolute = format!("/{exact_absolute_body}");
        assert_eq!(exact_absolute.len(), GENERATION_ROOT_MAX_ROUTE_BYTES);
        let exact_absolute_error = QualifiedGenerationRoot::admit(Path::new(&exact_absolute))
            .expect_err("the parsed but nonexistent exact-limit route must fail during open");
        assert_eq!(
            exact_absolute_error.stage(),
            GenerationRootStage::OpenRootRoute,
            "the exact absolute-route ceiling must pass parsing"
        );
        let oversized_absolute = format!("{exact_absolute}x");
        let absolute_error = QualifiedGenerationRoot::admit(Path::new(&oversized_absolute))
            .expect_err("one byte above the absolute-route ceiling must fail during parsing");
        assert_eq!(absolute_error.kind(), GenerationRootErrorKind::InvalidRoute);
        assert_eq!(absolute_error.stage(), GenerationRootStage::ParseRootRoute);
        assert_eq!(
            absolute_error.expected(),
            Some(GENERATION_ROOT_MAX_ROUTE_BYTES as u64)
        );
        assert_eq!(
            absolute_error.observed(),
            Some(GENERATION_ROOT_MAX_ROUTE_BYTES.saturating_add(1) as u64)
        );
    }

    #[cfg(not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))))]
    #[test]
    fn unsupported_targets_return_typed_platform_errors_without_path_access() {
        use super::{
            ConfinedGenerationPath, GenerationRootAnchorLayout, GenerationRootReadGuard,
            GenerationRootResult, QualifiedGenerationDirectory,
            QualifiedGenerationRoot as NamedGenerationRoot,
        };
        use std::path::Path;

        let missing = Path::new("this-path-must-never-be-opened");
        let confined = ConfinedGenerationPath::parse(missing)
            .expect_err("unsupported target must fail before parsing or I/O");
        assert_eq!(
            confined.kind(),
            GenerationRootErrorKind::UnsupportedPlatform
        );
        assert_eq!(confined.stage(), GenerationRootStage::PlatformGate);

        let absolute_missing = Path::new("/this-path-must-never-be-opened");
        let directory = QualifiedGenerationDirectory::admit(absolute_missing)
            .expect_err("unsupported target must fail before opening the path");
        assert_eq!(
            directory.kind(),
            GenerationRootErrorKind::UnsupportedPlatform
        );
        assert_eq!(directory.stage(), GenerationRootStage::PlatformGate);

        let layout = GenerationRootAnchorLayout::new(8)
            .expect("the public named-root layout should remain platform-independent");
        let named = NamedGenerationRoot::admit(absolute_missing, layout)
            .expect_err("the public named-root wrapper must fail before anchor path access");
        assert_eq!(named.kind(), GenerationRootErrorKind::UnsupportedPlatform);
        assert_eq!(named.stage(), GenerationRootStage::PlatformGate);

        let _read_guard_surface: for<'root> fn(
            &'root NamedGenerationRoot,
        ) -> GenerationRootResult<
            GenerationRootReadGuard<'root>,
        > = NamedGenerationRoot::read_guard;
    }

    #[cfg(target_os = "linux")]
    mod linux {
        use super::super::platform::{
            ProcBridgeProbePhase, ProcSelfBindingPhase, TestBoundary,
            duplicate_lock_descriptor_for_inheritance_for_test, install_test_hook,
            linux_device_components_match_for_test, linux_filesystem_namespace_digest,
            linux_filesystem_profile, linux_mountinfo_error_for_test,
            linux_mountinfo_filesystem_for_test, linux_open_error_for_test,
            linux_task_mountinfo_route_for_test, parse_mountinfo_filesystem,
            proc_bridge_error_for_test, proc_root_error_for_test, proc_thread_fd_route_for_test,
            set_lock_creator_process_id_for_test, set_root_creator_process_id_for_test,
            stat_inode_as_u64, stat_link_count_as_u64, stat_mode_as_u32, stat_seconds_as_i64,
            validate_absolute_ancestor_owner_mode_for_test,
            validate_bound_control_content_for_test, validate_directory_scan_counts_for_test,
            validate_linux_writable_attributes_for_test, validate_mountinfo_bound_for_test,
            validate_owner_mode_for_test, validate_proc_mount_id_mask_for_test,
            validate_regular_descriptor_type, verify_exact_directory_entry,
        };
        use super::super::{
            ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_BYTES,
            GENERATION_ROOT_MAX_DIRECTORY_ENTRIES, GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES,
            GENERATION_ROOT_MAX_MOUNTINFO_BYTES, GenerationFileExpectation,
            GenerationRootAnchorLayout, GenerationRootError, GenerationRootErrorKind,
            GenerationRootLockMode, GenerationRootStage,
            QualifiedGenerationDirectory as QualifiedGenerationRoot,
            QualifiedGenerationRoot as NamedGenerationRoot,
        };
        use sha2::{Digest, Sha256};
        use std::ffi::OsStr;
        use std::fs::{self, DirBuilder, File, OpenOptions};
        use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
        use std::os::fd::{AsFd, AsRawFd};
        use std::os::unix::fs::{
            DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt, symlink,
        };
        use std::path::{Path, PathBuf};
        use std::process::{Command, Stdio};
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Arc, Mutex, OnceLock};
        use std::time::{SystemTime, UNIX_EPOCH};

        static FIXTURE_SERIAL: AtomicU64 = AtomicU64::new(0);
        static FIXTURE_SLOT_SERIAL: AtomicU64 = AtomicU64::new(0);
        static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
        const FIXTURE_SLOT_COUNT: u64 = 512;

        fn fixture_root(label: &str) -> PathBuf {
            let slot = FIXTURE_SLOT_SERIAL.fetch_add(1, Ordering::Relaxed);
            assert!(
                slot < FIXTURE_SLOT_COUNT,
                "the retained fixture-slot pool must cover every test root"
            );
            let slot_path = secure_fixture_base().join(format!("slot-{slot:03}"));
            fixture_root_under(&slot_path, label)
        }

        fn secure_fixture_base() -> &'static Path {
            FIXTURE_BASE
                .get_or_init(|| {
                    let home = std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .expect("HOME must identify the process-private fixture ancestor");
                    assert!(home.is_absolute(), "HOME must be an absolute route");
                    let epoch_nanos = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_or(0, |duration| duration.as_nanos());
                    for attempt in 0_u64..64 {
                        let candidate = home.join(format!(
                            ".frankensearch-generation-root-tests-{}-{epoch_nanos}-{attempt}",
                            std::process::id()
                        ));
                        match DirBuilder::new().mode(0o700).create(&candidate) {
                            Ok(()) => {
                                fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                                    .expect("fixture base mode should be settable");
                                for slot in 0..FIXTURE_SLOT_COUNT {
                                    let slot_path = candidate.join(format!("slot-{slot:03}"));
                                    DirBuilder::new()
                                        .mode(0o700)
                                        .create(&slot_path)
                                        .expect("each retained fixture slot should be creatable");
                                    fs::set_permissions(
                                        &slot_path,
                                        fs::Permissions::from_mode(0o700),
                                    )
                                    .expect("each retained fixture slot mode should be private");
                                }
                                return candidate;
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                            Err(error) => panic!("failed to create retained fixture base: {error}"),
                        }
                    }
                    panic!("failed to allocate a retained process fixture base");
                })
                .as_path()
        }

        fn fixture_root_under(base: &Path, label: &str) -> PathBuf {
            let epoch_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            for attempt in 0_u64..64 {
                let serial = FIXTURE_SERIAL.fetch_add(1, Ordering::Relaxed);
                let candidate = base.join(format!(
                    "{}-{}-{epoch_nanos}-{serial}-{attempt}",
                    std::process::id(),
                    label
                ));
                match DirBuilder::new().mode(0o700).create(&candidate) {
                    Ok(()) => {
                        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                            .expect("fixture root mode should be settable");
                        return candidate;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("failed to create persistent test fixture: {error}"),
                }
            }
            panic!("failed to allocate a unique persistent test fixture");
        }

        fn run_btrfs_fixture_command(arguments: &[&OsStr]) {
            let output = Command::new("btrfs")
                .args(arguments)
                .output()
                .expect("the physical Btrfs receipt requires the btrfs utility");
            assert!(
                output.status.success(),
                "Btrfs fixture command failed: status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        fn run_bind_mount_fixture_command(program: &str, arguments: &[&OsStr]) {
            let output = Command::new(program)
                .args(arguments)
                .output()
                .expect("the physical absolute-route receipt requires mount utilities");
            assert!(
                output.status.success(),
                "bind-mount fixture command failed: program={program}, status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        struct BindMountFixture {
            source: PathBuf,
            target: PathBuf,
            mounted: bool,
        }

        impl BindMountFixture {
            fn new(source: &Path, target: &Path) -> Self {
                run_bind_mount_fixture_command(
                    "mount",
                    &[OsStr::new("--bind"), source.as_os_str(), target.as_os_str()],
                );
                Self {
                    source: source.to_path_buf(),
                    target: target.to_path_buf(),
                    mounted: true,
                }
            }

            fn remount_same_source(&mut self) {
                if self.mounted {
                    run_bind_mount_fixture_command(
                        "umount",
                        &[OsStr::new("--lazy"), self.target.as_os_str()],
                    );
                    self.mounted = false;
                }
                run_bind_mount_fixture_command(
                    "mount",
                    &[
                        OsStr::new("--bind"),
                        self.source.as_os_str(),
                        self.target.as_os_str(),
                    ],
                );
                self.mounted = true;
            }
        }

        impl Drop for BindMountFixture {
            fn drop(&mut self) {
                if self.mounted {
                    let _ = Command::new("umount").arg(&self.target).status();
                    self.mounted = false;
                }
            }
        }

        fn run_chattr_fixture_command(flag: &str, path: &Path) {
            let output = Command::new("chattr")
                .arg(flag)
                .arg(path)
                .output()
                .expect("the physical attribute receipt requires the chattr utility");
            assert!(
                output.status.success(),
                "chattr fixture command failed: flag={flag}, status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        struct ChattrReset {
            flag: &'static str,
            path: PathBuf,
            armed: bool,
        }

        impl ChattrReset {
            fn new(flag: &'static str, path: &Path) -> Self {
                Self {
                    flag,
                    path: path.to_path_buf(),
                    armed: true,
                }
            }

            fn clear(&mut self) {
                if self.armed {
                    run_chattr_fixture_command(self.flag, &self.path);
                    self.armed = false;
                }
            }
        }

        impl Drop for ChattrReset {
            fn drop(&mut self) {
                if self.armed {
                    let _ = Command::new("chattr")
                        .arg(self.flag)
                        .arg(&self.path)
                        .status();
                }
            }
        }

        fn private_dir(parent: &Path, name: &str) -> PathBuf {
            let path = parent.join(name);
            DirBuilder::new()
                .mode(0o700)
                .create(&path)
                .expect("private directory should be creatable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                .expect("private directory mode should be settable");
            path
        }

        fn private_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
            let path = parent.join(name);
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .expect("private file should be creatable");
            file.write_all(bytes)
                .expect("fixture bytes should be writable");
            file.sync_all().expect("fixture bytes should be durable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
                .expect("immutable fixture should be sealed read-only");
            path
        }

        fn control_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
            let path = parent.join(name);
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .expect("control file should be creatable");
            file.write_all(bytes)
                .expect("control bytes should be writable");
            file.sync_all().expect("control bytes should be durable");
            path
        }

        fn named_root_fixture(
            label: &str,
            lock_bytes: &[u8],
            authority_fill: u8,
        ) -> (PathBuf, PathBuf, GenerationRootAnchorLayout) {
            let root = fixture_root(label);
            control_file(&root, "LOCK", lock_bytes);
            let authority = authority_image(authority_fill);
            let authority_path = control_file(&root, "AUTHORITY", &authority);
            let layout = GenerationRootAnchorLayout::new(
                u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
            )
            .expect("the named lock fixture length is bounded");
            (root, authority_path, layout)
        }

        fn authority_image(fill: u8) -> Vec<u8> {
            vec![
                fill;
                usize::try_from(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
                    .expect("the frozen authority length fits usize")
            ]
        }

        fn rewrite_control_file(path: &Path, bytes: &[u8]) {
            let expected_len = u64::try_from(bytes.len()).expect("fixture length fits u64");
            assert_eq!(
                fs::metadata(path)
                    .expect("control metadata should be readable")
                    .len(),
                expected_len,
                "in-place control rewrites must retain the frozen physical length"
            );
            let mut file = OpenOptions::new()
                .write(true)
                .open(path)
                .expect("the mutable control anchor should be writable");
            file.seek(SeekFrom::Start(0))
                .expect("the control anchor should seek to its first byte");
            file.write_all(bytes)
                .expect("the complete in-place control image should be writable");
            file.sync_all()
                .expect("the in-place control rewrite should be durable");
        }

        fn install_linux_acl_xattr(
            descriptor: &File,
            name: &str,
            owner_permissions: u16,
            named_permissions: u16,
            group_permissions: u16,
            mask_permissions: u16,
            other_permissions: u16,
        ) {
            const ACL_UNDEFINED_ID: u32 = u32::MAX;
            const ACL_USER_OBJ: u16 = 0x01;
            const ACL_USER: u16 = 0x02;
            const ACL_GROUP_OBJ: u16 = 0x04;
            const ACL_MASK: u16 = 0x10;
            const ACL_OTHER: u16 = 0x20;

            let mut bytes = Vec::with_capacity(44);
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            for (tag, permissions, id) in [
                (ACL_USER_OBJ, owner_permissions, ACL_UNDEFINED_ID),
                (ACL_USER, named_permissions, 424_242),
                (ACL_GROUP_OBJ, group_permissions, ACL_UNDEFINED_ID),
                (ACL_MASK, mask_permissions, ACL_UNDEFINED_ID),
                (ACL_OTHER, other_permissions, ACL_UNDEFINED_ID),
            ] {
                bytes.extend_from_slice(&tag.to_le_bytes());
                bytes.extend_from_slice(&permissions.to_le_bytes());
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            rustix::fs::fsetxattr(
                descriptor.as_fd(),
                name,
                &bytes,
                rustix::fs::XattrFlags::empty(),
            )
            .expect("raw Linux ACL fixture must install without an external utility");
        }

        fn mapped_control_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
            let path = parent.join(name);
            let mut writer = crate::VectorIndex::create(&path, "generation-root-test", 1)
                .expect("mapped control fixture writer should be creatable");
            writer
                .write_record("control", &[1.0])
                .expect("mapped control fixture record should be valid");
            writer
                .finish()
                .expect("mapped control fixture should be durable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
                .expect("mapped control fixture should use the control mode");
            let mapping = crate::VectorIndex::open(&path)
                .expect("mapped control fixture should retain a shared writable mapping");
            (path, mapping)
        }

        fn mapped_immutable_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
            let (path, mapping) = mapped_control_file(parent, name);
            fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
                .expect("pre-held mapped fixture should seal as an immutable artifact");
            (path, mapping)
        }

        fn mutate_mapping(mapping: &mut crate::VectorIndex) {
            let crate::VectorIndexData::Mutable(bytes) = &mut mapping.data else {
                panic!("legacy VectorIndex::open must retain a writable mapping");
            };
            let byte = bytes
                .last_mut()
                .expect("mapped control fixture must contain at least one byte");
            *byte ^= 0x01;
        }

        fn digest(bytes: &[u8]) -> [u8; 32] {
            Sha256::digest(bytes).into()
        }

        fn immutable_expectation(bytes: &[u8]) -> GenerationFileExpectation {
            GenerationFileExpectation::immutable(
                u64::try_from(bytes.len()).expect("fixture length should fit u64"),
                digest(bytes),
            )
            .expect("small fixture expectation should be accepted")
        }

        fn confined(path: &str) -> ConfinedGenerationPath {
            ConfinedGenerationPath::parse(Path::new(path))
                .expect("static relative fixture path should parse")
        }

        fn injected(stage: GenerationRootStage) -> GenerationRootError {
            GenerationRootError::new(GenerationRootErrorKind::Io, stage)
        }

        #[test]
        fn exact_file_admission_retains_bytes_identity_and_durability() {
            let root_path = fixture_root("happy");
            let generation = private_dir(&root_path, "generation");
            let contents = b"immutable generation image";
            private_file(&generation, "vector.fsvi", contents);

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("ext4 root should qualify");
            assert_eq!(root.witness().mode() & 0o7777, 0o700);
            let admitted = root
                .admit_file(
                    &confined("generation/vector.fsvi"),
                    immutable_expectation(contents),
                )
                .expect("exact immutable file should be admitted");
            assert_eq!(admitted.as_bytes(), contents);
            assert_eq!(admitted.bytes().as_ref(), contents);
            assert_eq!(admitted.sha256(), digest(contents));
            assert_eq!(admitted.witness().byte_len(), contents.len() as u64);
            assert_eq!(admitted.witness().hard_links(), 1);
            assert_eq!(admitted.witness().mode() & 0o7777, 0o400);
            admitted
                .sync_durable()
                .expect("retained regular-file fsync should succeed");
            root.sync_directory_durable()
                .expect("retained root fsync should succeed");

            let debug_root = format!("{root:?}");
            let debug_file = format!("{admitted:?}");
            assert!(!debug_root.contains(root_path.to_string_lossy().as_ref()));
            assert!(!debug_file.contains("vector.fsvi"));
            assert!(!debug_file.contains("immutable generation image"));
        }

        #[test]
        fn route_and_error_diagnostics_never_disclose_path_bytes() {
            let root_path = fixture_root("redaction");
            let secret_component = "secret-customer-generation";
            let path = confined(secret_component);
            let error = QualifiedGenerationRoot::admit(Path::new("relative-root"))
                .expect_err("relative root route must fail");
            let display = error.to_string();
            let debug = format!("{error:?}");
            let path_debug = format!("{path:?}");
            assert!(!display.contains("relative-root"));
            assert!(!debug.contains("relative-root"));
            assert!(!path_debug.contains(secret_component));
            assert!(!path_debug.contains(root_path.to_string_lossy().as_ref()));
        }

        #[test]
        fn exact_directory_enumeration_rejects_case_distinct_component_bytes() {
            use rustix::fs::{Mode, OFlags, open};

            let root_path = fixture_root("exact-component-bytes");
            private_file(&root_path, "ExactName", b"x");
            let descriptor = open(
                &root_path,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .expect("fixture directory should open");
            verify_exact_directory_entry(
                &descriptor,
                OsStr::new("ExactName"),
                GenerationRootStage::OpenRegularFile,
                0,
            )
            .expect("the exact component bytes should be present once");
            assert_eq!(
                verify_exact_directory_entry(
                    &descriptor,
                    OsStr::new("exactname"),
                    GenerationRootStage::OpenRegularFile,
                    0,
                )
                .expect_err("case-distinct bytes must never count as the exact route")
                .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn root_and_descendant_modes_are_exactly_private() {
            let bad_root = fixture_root("bad-root-mode");
            fs::set_permissions(&bad_root, fs::Permissions::from_mode(0o750))
                .expect("fixture mode should change");
            assert_eq!(
                QualifiedGenerationRoot::admit(&bad_root)
                    .expect_err("non-private root must fail")
                    .kind(),
                GenerationRootErrorKind::WrongMode
            );

            let root_path = fixture_root("bad-descendant-mode");
            let directory = private_dir(&root_path, "generation");
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o750))
                .expect("fixture mode should change");
            private_file(&directory, "artifact", b"x");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            assert_eq!(
                root.admit_file(
                    &confined("generation/artifact"),
                    immutable_expectation(b"x")
                )
                .expect_err("non-private descendant directory must fail")
                .kind(),
                GenerationRootErrorKind::WrongMode
            );

            let root_path = fixture_root("bad-file-mode");
            let artifact = private_file(&root_path, "artifact", b"x");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o640))
                .expect("fixture mode should change");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(b"x"))
                    .expect_err("non-private regular file must fail")
                    .kind(),
                GenerationRootErrorKind::WrongMode
            );
        }

        #[test]
        fn linux_rejects_a_granting_default_acl_on_the_root() {
            let root_path = fixture_root("root-default-acl");
            let retained = File::open(&root_path).expect("private root should open");
            install_linux_acl_xattr(&retained, "system.posix_acl_default", 0o7, 0o4, 0, 0o4, 0);
            assert_eq!(
                crate::fd_acl::extended_acl_presence(retained.as_fd())
                    .expect("descriptor-bound default ACL probe should succeed"),
                crate::fd_acl::ExtendedAclPresence::Present
            );
            let error = QualifiedGenerationRoot::admit(&root_path)
                .expect_err("a granting default ACL must fail root admission");
            assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
            assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
        }

        #[test]
        fn linux_rejects_acl_bearing_artifact_control_and_lock_descriptors_before_content_io() {
            let artifact_root = fixture_root("artifact-access-acl");
            let artifact_path = private_file(&artifact_root, "artifact", b"sealed");
            let artifact_acl =
                File::open(&artifact_path).expect("immutable artifact should open for ACL setup");
            install_linux_acl_xattr(&artifact_acl, "system.posix_acl_access", 0o4, 0, 0, 0, 0);
            let artifact_generation = QualifiedGenerationRoot::admit(&artifact_root)
                .expect("the ACL-free root should qualify");
            let artifact_boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_artifact_boundaries = Arc::clone(&artifact_boundaries);
            let artifact_hook = install_test_hook(move |boundary| {
                hook_artifact_boundaries
                    .lock()
                    .expect("artifact boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let artifact_error = artifact_generation
                .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
                .expect_err("an ACL-bearing artifact must fail");
            assert_eq!(artifact_error.kind(), GenerationRootErrorKind::AclRejected);
            assert_eq!(artifact_error.stage(), GenerationRootStage::InspectAcl);
            let artifact_observed = artifact_boundaries
                .lock()
                .expect("artifact boundary log should lock");
            assert!(artifact_observed.contains(&TestBoundary::BeforeAclRead));
            let after_data_reopen = artifact_observed
                .iter()
                .position(|boundary| *boundary == (TestBoundary::AfterProcFdReopen { index: 0 }))
                .expect("Linux ACL inspection requires the real proc-derived data descriptor");
            let before_acl = artifact_observed
                .iter()
                .enumerate()
                .skip(after_data_reopen.saturating_add(1))
                .find_map(|(index, boundary)| {
                    (*boundary == TestBoundary::BeforeAclRead).then_some(index)
                })
                .expect("ACL inspection boundary should be recorded");
            assert!(
                before_acl > after_data_reopen,
                "the O_PATH probe must complete without fgetxattr; ACL inspection belongs only \
                 to the real proc-derived data descriptor"
            );
            assert!(
                !artifact_observed
                    .iter()
                    .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
                "ACL rejection must precede the first content read"
            );
            drop(artifact_observed);
            drop(artifact_hook);

            let control_root = fixture_root("control-access-acl");
            let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
            let control_acl =
                File::open(&control_path).expect("control file should open for ACL setup");
            install_linux_acl_xattr(&control_acl, "system.posix_acl_access", 0o6, 0, 0, 0, 0);
            let control_generation = QualifiedGenerationRoot::admit(&control_root)
                .expect("the ACL-free root should qualify");
            let control_error = control_generation
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect_err("an ACL-bearing control descriptor must fail");
            assert_eq!(control_error.kind(), GenerationRootErrorKind::AclRejected);
            assert_eq!(control_error.stage(), GenerationRootStage::InspectAcl);

            let lock_root = fixture_root("lock-access-acl");
            let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
            let lock_generation = QualifiedGenerationRoot::admit(&lock_root)
                .expect("the ACL-free root should qualify");
            let control = lock_generation
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("the ACL-free control should qualify before target mutation");
            let reached_flock = Arc::new(Mutex::new(false));
            let hook_reached_flock = Arc::clone(&reached_flock);
            let acl_mutated = Arc::new(Mutex::new(false));
            let hook_acl_mutated = Arc::clone(&acl_mutated);
            let hook_lock_path = lock_path.clone();
            let mut after_data_reopen = false;
            let _hook = install_test_hook(move |boundary| {
                if boundary == (TestBoundary::AfterProcFdReopen { index: 0 }) {
                    after_data_reopen = true;
                }
                if after_data_reopen
                    && boundary == TestBoundary::BeforeAclRead
                    && !*hook_acl_mutated.lock().expect("hook state should lock")
                {
                    let descriptor = File::open(&hook_lock_path)
                        .expect("lock target should open for injected ACL mutation");
                    install_linux_acl_xattr(
                        &descriptor,
                        "system.posix_acl_access",
                        0o6,
                        0,
                        0,
                        0,
                        0,
                    );
                    *hook_acl_mutated.lock().expect("hook state should lock") = true;
                }
                if boundary == TestBoundary::BeforeLock {
                    *hook_reached_flock.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            let lock_error = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("an ACL-bearing lock descriptor must fail before flock");
            assert_eq!(lock_error.kind(), GenerationRootErrorKind::AclRejected);
            assert_eq!(lock_error.stage(), GenerationRootStage::InspectAcl);
            assert!(
                *acl_mutated.lock().expect("hook state should lock"),
                "the lock candidate ACL must be injected after the real data reopen"
            );
            assert!(
                !*reached_flock.lock().expect("hook state should lock"),
                "ACL rejection must precede flock"
            );
        }

        #[test]
        fn ancestor_and_final_symlinks_fail_without_touching_decoys() {
            let root_path = fixture_root("symlink");
            let decoy = fixture_root("decoy");
            let decoy_file = private_file(&decoy, "artifact", b"decoy stays exact");
            symlink(&decoy, root_path.join("jump")).expect("ancestor symlink should be creatable");
            symlink(&decoy_file, root_path.join("final"))
                .expect("final symlink should be creatable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");

            for route in ["jump/artifact", "final"] {
                let error = root
                    .admit_file(
                        &confined(route),
                        GenerationFileExpectation::immutable(17, [0; 32])
                            .expect("small expectation should be accepted"),
                    )
                    .expect_err("symlink route must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::SymbolicLink);
            }
            assert_eq!(
                fs::read(&decoy_file).expect("decoy should remain readable"),
                b"decoy stays exact"
            );
        }

        #[test]
        fn hardlinks_and_special_files_are_never_admitted() {
            use std::os::unix::net::UnixListener;

            let root_path = fixture_root("special");
            let linked = private_file(&root_path, "linked", b"linked");
            fs::hard_link(&linked, root_path.join("linked-alias"))
                .expect("fixture hardlink should be creatable");
            private_dir(&root_path, "directory-final");
            rustix::fs::mkfifoat(
                rustix::fs::CWD,
                root_path.join("fifo-final"),
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .expect("fixture FIFO should be creatable");
            let root_descriptor =
                File::open(&root_path).expect("fixture root descriptor should open");
            let short_socket_route =
                format!("/proc/self/fd/{}/socket-final", root_descriptor.as_raw_fd());
            let _socket = UnixListener::bind(&short_socket_route)
                .expect("fixture socket should be creatable through a short retained-fd route");

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            assert_eq!(
                root.admit_file(&confined("linked"), immutable_expectation(b"linked"))
                    .expect_err("hardlinked file must fail")
                    .kind(),
                GenerationRootErrorKind::HardLinked
            );
            assert_eq!(
                root.admit_file(
                    &confined("directory-final"),
                    GenerationFileExpectation::immutable(0, digest(b""))
                        .expect("zero-length expectation should be accepted"),
                )
                .expect_err("directory must not be admitted as a regular file")
                .kind(),
                GenerationRootErrorKind::NotRegularFile
            );
            for route in ["fifo-final", "socket-final"] {
                assert!(
                    root.admit_file(
                        &confined(route),
                        GenerationFileExpectation::immutable(0, digest(b""))
                            .expect("zero-length expectation should be accepted"),
                    )
                    .is_err(),
                    "special file {route} must fail closed"
                );
            }
        }

        #[test]
        fn linux_final_probe_rejects_fifo_before_endpoint_io_procfd_reopen_or_read() {
            let root_path = fixture_root("fifo-preflight");
            let fifo_path = root_path.join("fifo-final");
            rustix::fs::mkfifoat(
                rustix::fs::CWD,
                &fifo_path,
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .expect("fixture FIFO should be creatable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let hook_fifo = fifo_path.clone();
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if boundary == (TestBoundary::AfterFinalProbeOpen { index: 0 }) {
                    let error = OpenOptions::new()
                        .write(true)
                        .custom_flags(libc::O_NONBLOCK)
                        .open(&hook_fifo)
                        .expect_err("O_PATH preflight must not establish a FIFO reader endpoint");
                    assert_eq!(
                        error.raw_os_error(),
                        Some(libc::ENXIO),
                        "a nonblocking FIFO writer must still observe no reader endpoint"
                    );
                }
                Ok(())
            });

            for error in [
                root.admit_file(
                    &confined("fifo-final"),
                    GenerationFileExpectation::immutable(0, digest(b""))
                        .expect("zero-length expectation should be accepted"),
                )
                .expect_err("immutable FIFO must fail at type preflight"),
                root.admit_control_file(
                    &confined("fifo-final"),
                    GenerationFileExpectation::control(0)
                        .expect("zero-length control expectation should be accepted"),
                )
                .expect_err("control FIFO must fail at type preflight"),
            ] {
                assert_eq!(error.kind(), GenerationRootErrorKind::NotRegularFile);
                assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
            }
            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(
                observed.contains(&TestBoundary::AfterFinalProbeOpen { index: 0 }),
                "the no-I/O O_PATH probe should be observable"
            );
            // Root revalidation legitimately opens the verified procfs root
            // to bind task-scoped mountinfo before inspecting the final
            // component. `BeforeProcFdReopen` is the distinct boundary that
            // would reopen this probed FIFO as a data descriptor.
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::BeforeProcFdReopen { .. }
                        | TestBoundary::BeforeRead { .. }
                        | TestBoundary::BeforeLock
                        | TestBoundary::BeforeFileSync
                )),
                "special nodes must fail before final-capability reopen or any content operation"
            );
            drop(observed);
        }

        #[test]
        fn linux_final_probe_rejects_a_real_device_descriptor_without_io_reopen() {
            use rustix::fs::{CWD, ResolveFlags, openat2};

            let descriptor = openat2(
                CWD,
                Path::new("/dev/null"),
                rustix::fs::OFlags::PATH
                    | rustix::fs::OFlags::NOFOLLOW
                    | rustix::fs::OFlags::CLOEXEC,
                rustix::fs::Mode::empty(),
                ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
            )
            .expect("the physical device should support a side-effect-free O_PATH reference");
            let error = validate_regular_descriptor_type(
                &descriptor,
                GenerationRootStage::OpenRegularFile,
                0,
            )
            .expect_err("the real device must fail the production type gate");
            assert_eq!(error.kind(), GenerationRootErrorKind::NotRegularFile);
            assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
        }

        #[test]
        fn linux_regular_to_fifo_swap_after_probe_never_touches_the_replacement() {
            let root_path = fixture_root("regular-to-fifo-after-probe");
            private_file(&root_path, "artifact", b"sealed");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let writer_probe = Arc::new(Mutex::new(None));
            let hook_writer_probe = Arc::clone(&writer_probe);
            let hook_root = root_path.clone();
            let mut swapped = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !swapped && boundary == (TestBoundary::AfterProbeQualified { index: 0 }) {
                    fs::rename(
                        hook_root.join("artifact"),
                        hook_root.join("artifact-original"),
                    )
                    .expect("the probed regular file should move without invalidating its fd");
                    rustix::fs::mkfifoat(
                        rustix::fs::CWD,
                        hook_root.join("artifact"),
                        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
                    )
                    .expect("the canonical route should be replaceable with a FIFO fixture");
                    swapped = true;
                }
                if boundary == (TestBoundary::AfterProcFdReopen { index: 0 }) {
                    let observed_errno = match OpenOptions::new()
                        .write(true)
                        .custom_flags(libc::O_NONBLOCK)
                        .open(hook_root.join("artifact"))
                    {
                        Ok(writer) => {
                            drop(writer);
                            None
                        }
                        Err(error) => error.raw_os_error(),
                    };
                    *hook_writer_probe
                        .lock()
                        .expect("writer probe state should lock") = Some(observed_errno);
                }
                Ok(())
            });
            let error = root
                .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
                .expect_err("the final route reprobe must reject a special-node substitution");
            assert!(
                matches!(
                    error.kind(),
                    GenerationRootErrorKind::NotRegularFile
                        | GenerationRootErrorKind::ObjectChanged
                ),
                "the substitution must fail as a type or identity change"
            );
            assert_eq!(
                *writer_probe.lock().expect("writer probe state should lock"),
                Some(Some(libc::ENXIO)),
                "after the proc-derived data reopen, a nonblocking FIFO writer must still observe \
                 no reader; otherwise the retained descriptor touched the replacement FIFO"
            );
            assert!(
                !boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .iter()
                    .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
                "the replacement FIFO must never reach the content reader"
            );
        }

        #[test]
        fn linux_filtered_procfd_capability_reopen_is_typed_unsupported() {
            let root_path = fixture_root("filtered-procfd");
            private_file(&root_path, "artifact", b"sealed");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let _guard = install_test_hook(|boundary| {
                if boundary == TestBoundary::BeforeProcCapabilityRootOpen {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::UnsupportedKernelFeature,
                        GenerationRootStage::OpenRegularFile,
                    ));
                }
                Ok(())
            });
            let error = root
                .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
                .expect_err("a filtered procfd capability bridge must fail closed");
            assert_eq!(
                error.kind(),
                GenerationRootErrorKind::UnsupportedKernelFeature
            );
            assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
        }

        #[test]
        fn size_hash_grow_and_shrink_fail_closed() {
            let root_path = fixture_root("content");
            let contents = b"abcdefgh";
            let artifact = private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");

            let wrong_size = GenerationFileExpectation::immutable(7, digest(b"abcdefg"))
                .expect("small expectation should be accepted");
            assert_eq!(
                root.admit_file(&confined("artifact"), wrong_size)
                    .expect_err("wrong exact size must fail")
                    .kind(),
                GenerationRootErrorKind::SizeMismatch
            );
            let wrong_hash = GenerationFileExpectation::immutable(8, [0x31; 32])
                .expect("small expectation should be accepted");
            assert_eq!(
                root.admit_file(&confined("artifact"), wrong_hash)
                    .expect_err("wrong exact digest must fail")
                    .kind(),
                GenerationRootErrorKind::HashMismatch
            );

            for grow in [true, false] {
                fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
                    .expect("fixture should become writable for the retained test fd");
                let mut mutation_file = OpenOptions::new()
                    .write(true)
                    .open(&artifact)
                    .expect("retained mutation fd should open");
                fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
                    .expect("fixture should be resealed before admission");
                let guard = install_test_hook(move |boundary| {
                    if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                        if grow {
                            mutation_file
                                .seek(SeekFrom::End(0))
                                .expect("fixture should be seekable");
                            mutation_file.write_all(b"+").expect("fixture should grow");
                        } else {
                            mutation_file.set_len(0).expect("fixture should shrink");
                        }
                        mutation_file
                            .sync_all()
                            .expect("size mutation should be durable");
                    }
                    Ok(())
                });
                assert!(
                    root.admit_file(&confined("artifact"), immutable_expectation(contents))
                        .is_err(),
                    "mid-read size drift must fail closed"
                );
                drop(guard);
                fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
                    .expect("fixture should become writable for reset");
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&artifact)
                    .expect("fixture should be resettable");
                file.write_all(contents).expect("fixture should reset");
                file.sync_all().expect("reset should be durable");
                fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
                    .expect("fixture should be resealed after reset");
            }
        }

        #[test]
        fn same_size_rewrite_and_hardlink_race_fail_after_initial_validation() {
            let contents = b"original";

            let root_path = fixture_root("rewrite-race");
            let artifact = private_file(&root_path, "artifact", contents);
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
                .expect("fixture should become writable for the retained test fd");
            let mut rewrite_file = OpenOptions::new()
                .write(true)
                .open(&artifact)
                .expect("retained rewrite fd should open");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
                .expect("fixture should be resealed before admission");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeFileRouteReopen {
                    rewrite_file
                        .seek(SeekFrom::Start(0))
                        .expect("fixture should be seekable");
                    rewrite_file
                        .write_all(b"rewrite!")
                        .expect("same-size rewrite should succeed");
                    rewrite_file.sync_all().expect("rewrite should be durable");
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect_err("same-size post-read rewrite must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            drop(guard);

            let root_path = fixture_root("nlink-race");
            let artifact = private_file(&root_path, "artifact", contents);
            let alias = root_path.join("late-alias");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let _guard = install_test_hook(move |boundary| {
                if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    fs::hard_link(&artifact, &alias).expect("late hardlink should be creatable");
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect_err("post-open hardlink must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn file_and_root_path_swaps_are_detected_while_retained_bytes_survive() {
            let contents = b"retained exact bytes";
            let root_path = fixture_root("retained");
            let artifact = private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(contents))
                .expect("exact file should be admitted");

            let retained_artifact = root_path.join("artifact-retained");
            fs::rename(&artifact, &retained_artifact)
                .expect("fixture file should move to a fresh retained name");
            private_file(&root_path, "artifact", b"x");
            assert_eq!(admitted.as_bytes(), contents);
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect_err("replacement file must not satisfy original digest")
                    .kind(),
                GenerationRootErrorKind::SizeMismatch
            );

            let retained_root = root_path.with_extension("retained-root");
            fs::rename(&root_path, &retained_root)
                .expect("fixture root should move to a fresh retained name");
            DirBuilder::new()
                .mode(0o700)
                .create(&root_path)
                .expect("attacker root should be creatable");
            fs::set_permissions(&root_path, fs::Permissions::from_mode(0o700))
                .expect("attacker root mode should be settable");
            assert_eq!(
                root.revalidate_route()
                    .expect_err("root pathname replacement must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            assert_eq!(admitted.as_bytes(), contents);
        }

        #[test]
        fn root_swap_during_initial_double_resolution_is_rejected() {
            let root_path = fixture_root("root-admission-race");
            let retained_root = root_path.with_extension("first");
            let swap_source = root_path.clone();
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::AfterFirstRootQualification {
                    fs::rename(&swap_source, &retained_root)
                        .expect("first root should move to a fresh name");
                    DirBuilder::new()
                        .mode(0o700)
                        .create(&swap_source)
                        .expect("replacement root should be creatable");
                    fs::set_permissions(&swap_source, fs::Permissions::from_mode(0o700))
                        .expect("replacement root mode should be settable");
                }
                Ok(())
            });
            assert_eq!(
                QualifiedGenerationRoot::admit(&root_path)
                    .expect_err("mid-admission root swap must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn intermediate_ancestor_substitution_is_detected_when_final_root_inode_is_unchanged() {
            let base = fixture_root("ancestor-chain");
            let ancestor = private_dir(&base, "ancestor");
            let root_path = private_dir(&ancestor, "root");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted_root_witness = root.witness();

            let displaced_ancestor = base.join("displaced-ancestor");
            fs::rename(&ancestor, &displaced_ancestor)
                .expect("original ancestor should move to a fresh retained name");
            let replacement_ancestor = private_dir(&base, "ancestor");
            fs::rename(
                displaced_ancestor.join("root"),
                replacement_ancestor.join("root"),
            )
            .expect("the same admitted root inode should move under the replacement ancestor");

            assert_eq!(
                fs::metadata(&root_path)
                    .expect("canonical root route should resolve again")
                    .ino(),
                admitted_root_witness.inode()
            );
            assert_eq!(
                root.revalidate_route()
                    .expect_err(
                        "whole-route validation must reject a replaced intermediate ancestor"
                    )
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn relative_ancestor_substitution_is_detected_when_final_file_inode_is_unchanged() {
            let root_path = fixture_root("relative-ancestor-chain");
            let ancestor = private_dir(&root_path, "ancestor");
            let nested = private_dir(&ancestor, "nested");
            let contents = b"same retained file";
            private_file(&nested, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(
                    &confined("ancestor/nested/artifact"),
                    immutable_expectation(contents),
                )
                .expect("immutable file should qualify");
            let admitted_inode = admitted.witness().inode();

            let displaced_ancestor = root_path.join("displaced-ancestor");
            fs::rename(&ancestor, &displaced_ancestor)
                .expect("original relative ancestor should move to a fresh retained name");
            let replacement_ancestor = private_dir(&root_path, "ancestor");
            fs::rename(
                displaced_ancestor.join("nested"),
                replacement_ancestor.join("nested"),
            )
            .expect("the original final subtree should move below the replacement ancestor");

            assert_eq!(
                fs::metadata(root_path.join("ancestor/nested/artifact"))
                    .expect("canonical final file route should resolve again")
                    .ino(),
                admitted_inode
            );
            assert_eq!(
                admitted
                    .sync_durable()
                    .expect_err("the full relative identity chain must reject a replaced ancestor")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn control_ancestor_substitution_fails_before_flock_when_final_inode_is_unchanged() {
            let root_path = fixture_root("control-ancestor-chain");
            let ancestor = private_dir(&root_path, "ancestor");
            let nested = private_dir(&ancestor, "nested");
            control_file(&nested, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("ancestor/nested/LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("control file should qualify");
            let admitted_inode = control.witness().inode();

            let displaced_ancestor = root_path.join("displaced-ancestor");
            fs::rename(&ancestor, &displaced_ancestor)
                .expect("original relative ancestor should move to a fresh retained name");
            let replacement_ancestor = private_dir(&root_path, "ancestor");
            fs::rename(
                displaced_ancestor.join("nested"),
                replacement_ancestor.join("nested"),
            )
            .expect("the original control subtree should move below the replacement ancestor");

            assert_eq!(
                fs::metadata(root_path.join("ancestor/nested/LOCK"))
                    .expect("canonical final control route should resolve again")
                    .ino(),
                admitted_inode
            );
            let lock_boundary_reached = Arc::new(Mutex::new(false));
            let hook_lock_boundary_reached = Arc::clone(&lock_boundary_reached);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_lock_boundary_reached
                        .lock()
                        .expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("relative ancestor replacement must fail before flock")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            assert!(
                !*lock_boundary_reached
                    .lock()
                    .expect("hook state should lock"),
                "relative route validation must precede flock"
            );
        }

        #[test]
        fn lock_modes_contend_release_sync_and_validate_optional_digest() {
            let root_path = fixture_root("locks");
            let contents = b"lock-v1!";
            control_file(&root_path, "LOCK", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let route = confined("LOCK");
            let expectation = GenerationFileExpectation::control(8)
                .expect("small control-file expectation should be accepted");
            let control = root
                .admit_control_file(&route, expectation)
                .expect("control file should be admitted once");
            assert_eq!(control.bytes().as_ref(), contents);
            assert_eq!(control.sha256(), digest(contents));

            let shared_one = control
                .try_lock(GenerationRootLockMode::Shared)
                .expect("first shared lock should succeed");
            let shared_two = control
                .try_lock(GenerationRootLockMode::Shared)
                .expect("second shared lock should succeed");
            assert_eq!(
                control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("exclusive lock must contend with shared locks")
                    .kind(),
                GenerationRootErrorKind::LockContended
            );
            shared_one.unlock().expect("shared unlock should succeed");
            shared_two.unlock().expect("shared unlock should succeed");

            let exclusive = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("exclusive lock should succeed after release");
            assert_eq!(exclusive.witness(), control.witness());
            assert_eq!(exclusive.bytes().as_ref(), contents);
            assert_eq!(exclusive.sha256(), control.sha256());
            assert_eq!(exclusive.mode(), GenerationRootLockMode::Exclusive);
            assert_eq!(
                control
                    .try_lock(GenerationRootLockMode::Shared)
                    .expect_err("shared lock must contend with exclusive lock")
                    .kind(),
                GenerationRootErrorKind::LockContended
            );
            exclusive
                .sync_durable()
                .expect("locked control-file fsync should succeed");
            exclusive.unlock().expect("exclusive unlock should succeed");

            let exact_control = root
                .admit_control_file(
                    &route,
                    GenerationFileExpectation::control_with_sha256(8, digest(contents))
                        .expect("small expectation should be accepted"),
                )
                .expect("digest-qualified control capability should be admitted");
            let exact = exact_control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("digest-qualified lock should succeed");
            exact
                .unlock()
                .expect("digest-qualified unlock should succeed");
            let wrong = GenerationFileExpectation::control_with_sha256(8, [0xa5; 32])
                .expect("small expectation should be accepted");
            assert_eq!(
                root.admit_control_file(&route, wrong)
                    .expect("wrong digest is checked only while acquiring the lock")
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("wrong lock-file digest must fail after acquisition")
                    .kind(),
                GenerationRootErrorKind::HashMismatch
            );
        }

        #[test]
        fn immutable_content_drift_fences_sync_before_durability() {
            let root_path = fixture_root("mapped-immutable-before-sync");
            let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
            let admitted_bytes =
                fs::read(&artifact_path).expect("mapped immutable fixture must be readable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(
                    &confined("artifact"),
                    immutable_expectation(&admitted_bytes),
                )
                .expect("mapped immutable fixture should qualify before ambient mutation");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared immutable mapping should remain retained by the hook"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = admitted
                .sync_durable()
                .expect_err("pre-durability immutable content drift must fail closed");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            assert!(
                !boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .contains(&TestBoundary::BeforeFileSync),
                "immutable content validation must fail before the durability syscall"
            );
        }

        #[test]
        fn immutable_content_drift_during_durability_is_detected_after_barrier() {
            let root_path = fixture_root("mapped-immutable-during-sync");
            let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
            let admitted_bytes =
                fs::read(&artifact_path).expect("mapped immutable fixture must be readable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(
                    &confined("artifact"),
                    immutable_expectation(&admitted_bytes),
                )
                .expect("mapped immutable fixture should qualify before ambient mutation");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == TestBoundary::BeforeFileSync {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared immutable mapping should remain retained by the hook"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = admitted
                .sync_durable()
                .expect_err("immutable mutation at durability must fail after the barrier");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            let observed = boundaries.lock().expect("boundary log should lock");
            let after_sync = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::AfterFileSync)
                .expect("the durability barrier should complete before post-check failure");
            assert!(
                observed.iter().enumerate().any(|(index, boundary)| {
                    index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
                }),
                "a second exact immutable content read must follow the durability barrier"
            );
            drop(observed);
        }

        #[test]
        fn control_writer_mapping_is_lock_contended() {
            let root_path = fixture_root("mapped-control-after-flock");
            let (lock_path, _mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let expectation = GenerationFileExpectation::control(byte_len)
                .expect("mapped control fixture length should be accepted");
            let control = root
                .admit_control_file(&confined("LOCK"), expectation)
                .expect("mapped control fixture should qualify before lock acquisition");
            let error = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("retained FSVI writer mapping must block a second exclusive lock");
            assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
        }

        #[test]
        fn control_content_drift_fences_sync_before_durability() {
            let root_path = fixture_root("mapped-control-before-sync");
            let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let error = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(byte_len)
                        .expect("mapped control fixture length should be accepted"),
                )
                .expect("mapped control fixture should qualify")
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("retained FSVI writer mapping must block the durability lock");
            assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            drop(mapping);
        }

        #[test]
        fn control_content_drift_during_durability_is_detected_after_barrier() {
            let root_path = fixture_root("mapped-control-during-sync");
            let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let error = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(byte_len)
                        .expect("mapped control fixture length should be accepted"),
                )
                .expect("mapped control fixture should qualify")
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("retained FSVI writer mapping must block the durability lock");
            assert_eq!(error.kind(), GenerationRootErrorKind::LockContended);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            drop(mapping);
        }

        #[test]
        fn bound_content_mismatch_is_object_changed_even_when_metadata_is_not_consulted() {
            let error = validate_bound_control_content_for_test(
                [0x11; 32],
                [0x22; 32],
                GenerationRootStage::AcquireLock,
            )
            .expect_err("different content witnesses must fail independently of metadata");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            validate_bound_control_content_for_test(
                [0x33; 32],
                [0x33; 32],
                GenerationRootStage::SyncRegularFile,
            )
            .expect("identical content witnesses should validate");
        }

        #[test]
        fn inherited_lock_guard_never_issues_unlock_from_a_different_process_identity() {
            let root_path = fixture_root("forked-lock");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            let mut inherited = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("exclusive lock should succeed");
            set_lock_creator_process_id_for_test(
                &mut inherited.inner,
                std::process::id().wrapping_add(1),
            );
            let unlock_boundary_reached = Arc::new(Mutex::new(false));
            let hook_boundary_reached = Arc::clone(&unlock_boundary_reached);
            let guard = install_test_hook(move |boundary| {
                if matches!(
                    boundary,
                    TestBoundary::BeforeUnlock | TestBoundary::AfterUnlock
                ) {
                    *hook_boundary_reached
                        .lock()
                        .expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                inherited
                    .unlock()
                    .expect_err("simulated fork child must refuse explicit unlock")
                    .kind(),
                GenerationRootErrorKind::ForkedProcess
            );
            assert!(
                !*unlock_boundary_reached
                    .lock()
                    .expect("hook state should lock"),
                "process identity must be checked before the explicit LOCK_UN path"
            );
            drop(guard);

            control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("descriptor close should leave the control file lockable")
                .unlock()
                .expect("fresh same-process lock should unlock");
        }

        #[test]
        fn inherited_root_capability_fails_before_proc_bridge_and_regular_file_io() {
            let root_path = fixture_root("forked-root");
            private_file(&root_path, "artifact", b"fork");
            let mut root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            set_root_creator_process_id_for_test(
                &mut root.inner,
                std::process::id().wrapping_add(1),
            );
            let boundary_reached = Arc::new(Mutex::new(false));
            let hook_boundary_reached = Arc::clone(&boundary_reached);
            let _guard = install_test_hook(move |_| {
                *hook_boundary_reached
                    .lock()
                    .expect("hook state should lock") = true;
                Ok(())
            });

            let file_error = root
                .admit_file(&confined("artifact"), immutable_expectation(b"fork"))
                .expect_err("simulated fork child must reject before the proc-fd bridge");
            assert_eq!(file_error.kind(), GenerationRootErrorKind::ForkedProcess);
            assert_eq!(file_error.stage(), GenerationRootStage::RevalidateRootRoute);
            let route_error = root
                .revalidate_route()
                .expect_err("simulated fork child must reject the retained root");
            assert_eq!(route_error.kind(), GenerationRootErrorKind::ForkedProcess);
            assert_eq!(
                route_error.stage(),
                GenerationRootStage::RevalidateRootRoute
            );
            assert!(
                !*boundary_reached.lock().expect("hook state should lock"),
                "process identity must be checked before proc-root qualification, proc-fd \
                 resolution, reads, flock, sync, or route work"
            );
        }

        #[test]
        #[ignore = "requires native Linux, python3, fork, and procfs"]
        fn physical_linux_proc_bridge_fork_receipt_separates_child_self_from_parent_fd_number() {
            let root_path = fixture_root("physical-fork-proc-bridge");
            let inherited_path = private_file(&root_path, "inherited", b"inherited");
            let parent_decoy_path = private_file(&root_path, "parent-decoy", b"parent-decoy");
            let script = r#"
import json
import os
import sys

inherited_path = os.path.realpath(sys.argv[1])
parent_decoy_path = os.path.realpath(sys.argv[2])
parent_pid = os.getpid()
probe = os.open(inherited_path, os.O_PATH | os.O_CLOEXEC)
probe_number = probe
ready_read, ready_write = os.pipe()
continue_read, continue_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    os.close(continue_write)
    os.write(ready_write, b"r")
    os.close(ready_write)
    if os.read(continue_read, 1) != b"c":
        os._exit(3)
    os.close(continue_read)
    child_self_route = os.readlink(f"/proc/self/fd/{probe_number}")
    parent_same_number_route = os.readlink(
        f"/proc/{parent_pid}/fd/{probe_number}"
    )
    child_ok = (
        child_self_route == inherited_path
        and parent_same_number_route == parent_decoy_path
    )
    print(json.dumps({
        "role": "child",
        "probe_number": probe_number,
        "self_route": child_self_route,
        "parent_same_number_route": parent_same_number_route,
        "self_is_inherited": child_self_route == inherited_path,
        "parent_same_number_is_decoy": (
            parent_same_number_route == parent_decoy_path
        ),
    }), flush=True)
    os._exit(0 if child_ok else 4)

os.close(ready_write)
os.close(continue_read)
if os.read(ready_read, 1) != b"r":
    raise SystemExit(5)
os.close(ready_read)
os.close(probe)
decoy = os.open(parent_decoy_path, os.O_PATH | os.O_CLOEXEC)
if decoy != probe_number:
    os.dup2(decoy, probe_number, inheritable=False)
    os.close(decoy)
parent_self_route = os.readlink(f"/proc/self/fd/{probe_number}")
os.write(continue_write, b"c")
os.close(continue_write)
_, status = os.waitpid(child, 0)
child_exit = os.waitstatus_to_exitcode(status)
print(json.dumps({
    "role": "parent",
    "probe_number": probe_number,
    "self_route": parent_self_route,
    "self_is_decoy": parent_self_route == parent_decoy_path,
    "child_exit": child_exit,
}), flush=True)
if parent_self_route != parent_decoy_path or child_exit != 0:
    raise SystemExit(6)
"#;
            let output = Command::new("python3")
                .arg("-c")
                .arg(script)
                .arg(&inherited_path)
                .arg(&parent_decoy_path)
                .output()
                .expect("native proc-bridge fork receipt requires python3");
            assert!(
                output.status.success(),
                "proc-bridge fork receipt failed: status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8(output.stdout).expect("fork receipt JSON must be UTF-8");
            let mut records = stdout.lines().map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .expect("each fork receipt line must be JSON")
            });
            let child = records.next().expect("child receipt must be present");
            let parent = records.next().expect("parent receipt must be present");
            assert!(
                records.next().is_none(),
                "fork receipt must contain exactly child and parent records"
            );
            assert_eq!(child["role"], "child");
            assert_eq!(child["self_is_inherited"], true);
            assert_eq!(child["parent_same_number_is_decoy"], true);
            assert_eq!(parent["role"], "parent");
            assert_eq!(parent["self_is_decoy"], true);
            assert_eq!(parent["child_exit"], 0);
            assert_eq!(
                child["probe_number"], parent["probe_number"],
                "the receipt must compare the exact same numeric descriptor in both processes"
            );
            assert_ne!(
                child["self_route"], child["parent_same_number_route"],
                "the child's self route must remain bound to its inherited probe while the \
                 parent's same-number route names the decoy"
            );
        }

        #[test]
        #[ignore = "requires native Linux, python3, and fork/flock process support"]
        fn physical_linux_fork_receipt_proves_close_vs_explicit_unlock_with_inherited_ofd() {
            let root_path = fixture_root("physical-fork-ofd");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let script = r#"
import fcntl
import json
import os
import sys

path = sys.argv[1]

def open_lock():
    return os.open(path, os.O_RDWR | os.O_CLOEXEC)

held = open_lock()
fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
read_end, write_end = os.pipe()
child = os.fork()
if child == 0:
    os.close(write_end)
    os.read(read_end, 1)
    os.close(held)
    os._exit(0)
os.close(read_end)
os.close(held)
probe = open_lock()
close_only_contended = False
try:
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    close_only_contended = True
os.write(write_end, b"x")
os.close(write_end)
os.waitpid(child, 0)
fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
fcntl.flock(probe, fcntl.LOCK_UN)
os.close(probe)

held = open_lock()
fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
read_end, write_end = os.pipe()
child = os.fork()
if child == 0:
    os.close(write_end)
    os.read(read_end, 1)
    os.close(held)
    os._exit(0)
os.close(read_end)
fcntl.flock(held, fcntl.LOCK_UN)
os.close(held)
probe = open_lock()
fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
explicit_unlock_released = True
fcntl.flock(probe, fcntl.LOCK_UN)
os.close(probe)
os.write(write_end, b"x")
os.close(write_end)
os.waitpid(child, 0)

print(json.dumps({
    "close_only_contended_while_child_retained_ofd": close_only_contended,
    "explicit_unlock_released_while_child_retained_ofd": explicit_unlock_released,
}))
if not close_only_contended or not explicit_unlock_released:
    raise SystemExit(2)
"#;
            let output = Command::new("python3")
                .arg("-c")
                .arg(script)
                .arg(&lock_path)
                .output()
                .expect("native fork receipt requires python3");
            assert!(
                output.status.success(),
                "native fork/OFD receipt failed: status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            let stdout = String::from_utf8(output.stdout)
                .expect("native fork/OFD receipt must emit UTF-8 JSON");
            assert!(stdout.contains("\"close_only_contended_while_child_retained_ofd\": true"));
            assert!(stdout.contains("\"explicit_unlock_released_while_child_retained_ofd\": true"));
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT on native ext4/Btrfs plus python3"]
        fn physical_generation_lock_drop_and_unlock_release_while_child_retains_same_ofd() {
            struct FdHolder {
                child: Option<std::process::Child>,
            }

            impl FdHolder {
                fn assert_alive(&mut self) {
                    assert!(
                        self.child
                            .as_mut()
                            .expect("fd-holder child must be present")
                            .try_wait()
                            .expect("fd-holder liveness should be observable")
                            .is_none(),
                        "child must still retain the guard OFD"
                    );
                }

                fn release(mut self) {
                    let mut child = self.child.take().expect("fd-holder child must be present");
                    child
                        .stdin
                        .take()
                        .expect("fd-holder stdin must be piped")
                        .write_all(b"x")
                        .expect("fd-holder release byte must be writable");
                    let output = child
                        .wait_with_output()
                        .expect("fd-holder process must be waitable");
                    assert!(
                        output.status.success(),
                        "fd-holder failed: status={:?}, stderr={}",
                        output.status.code(),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }

            impl Drop for FdHolder {
                fn drop(&mut self) {
                    let Some(mut child) = self.child.take() else {
                        return;
                    };
                    if let Some(mut stdin) = child.stdin.take() {
                        let _ = stdin.write_all(b"x");
                    }
                    match child.try_wait() {
                        Ok(Some(_)) => {}
                        Ok(None) | Err(_) => {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }

            fn spawn_inherited_fd_holder(alias: &rustix::fd::OwnedFd) -> FdHolder {
                let mut child = Command::new("python3")
                    .arg("-c")
                    .arg(
                        "import os,sys; fd=int(sys.argv[1]); os.fstat(fd); \
                         print('READY', flush=True); sys.stdin.buffer.read(1)",
                    )
                    .arg(alias.as_raw_fd().to_string())
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("native inherited-OFD receipt requires python3");
                let stdout = child.stdout.take().expect("fd-holder stdout must be piped");
                let mut ready = String::new();
                BufReader::new(stdout)
                    .read_line(&mut ready)
                    .expect("fd-holder readiness must be readable");
                assert_eq!(ready.trim(), "READY", "child must retain the exact alias");
                FdHolder { child: Some(child) }
            }

            let base = std::env::var_os("FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT")
                .map(PathBuf::from)
                .expect("set FRANKENSEARCH_LINUX_LOCK_FORK_TEST_ROOT to native ext4/Btrfs");
            let filesystem =
                rustix::fs::statfs(&base).expect("physical lock fixture base must be statable");
            assert!(
                i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
                    || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
                "physical inherited-OFD receipt requires ext4 or Btrfs"
            );
            let root_path = fixture_root_under(&base, "physical-generation-lock-fork");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("control file should qualify");

            let dropped = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("Rust guard should acquire for Drop receipt");
            let drop_alias = duplicate_lock_descriptor_for_inheritance_for_test(&dropped.inner)
                .expect("test should duplicate the private guard descriptor");
            let mut drop_child = spawn_inherited_fd_holder(&drop_alias);
            drop(drop_alias);
            drop_child.assert_alive();
            drop(dropped);
            control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("creator Drop must LOCK_UN while the child alias survives")
                .unlock()
                .expect("fresh lock after creator Drop should release");
            drop_child.release();

            let explicit = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("Rust guard should acquire for explicit-unlock receipt");
            let explicit_alias =
                duplicate_lock_descriptor_for_inheritance_for_test(&explicit.inner)
                    .expect("test should duplicate the explicit guard descriptor");
            let mut explicit_child = spawn_inherited_fd_holder(&explicit_alias);
            drop(explicit_alias);
            explicit_child.assert_alive();
            explicit
                .unlock()
                .expect("creator explicit LOCK_UN should succeed");
            control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("explicit LOCK_UN must release while the child alias survives")
                .unlock()
                .expect("fresh lock after explicit unlock should release");
            explicit_child.release();
        }

        #[test]
        fn control_route_replacement_fences_lock_acquisition_and_locked_sync() {
            let root_path = fixture_root("control-route-replacement-before-lock");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            fs::rename(&lock_path, root_path.join("retained-LOCK"))
                .expect("original control file should move to a fresh retained route");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let before_lock_reached = Arc::new(Mutex::new(false));
            let hook_before_lock_reached = Arc::clone(&before_lock_reached);
            let guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_before_lock_reached
                        .lock()
                        .expect("hook state should lock") = true;
                }
                Ok(())
            });
            let lock_error = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("replacement route must fence lock acquisition");
            assert_eq!(lock_error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                lock_error.stage(),
                GenerationRootStage::RevalidateRegularFile
            );
            assert!(
                !*before_lock_reached.lock().expect("hook state should lock"),
                "route identity must be checked before flock"
            );
            drop(guard);

            let root_path = fixture_root("control-route-replacement-after-lock");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            let lock = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("exclusive lock should succeed");
            let retained_lock_path = root_path.join("retained-LOCK");
            fs::rename(&lock_path, &retained_lock_path)
                .expect("locked file should move to a fresh retained route");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let sync_boundary_reached = Arc::new(Mutex::new(false));
            let hook_sync_boundary_reached = Arc::clone(&sync_boundary_reached);
            let unlock_boundary_reached = Arc::new(Mutex::new(false));
            let hook_unlock_boundary_reached = Arc::clone(&unlock_boundary_reached);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeFileSync {
                    *hook_sync_boundary_reached
                        .lock()
                        .expect("hook state should lock") = true;
                }
                if boundary == TestBoundary::BeforeUnlock {
                    *hook_unlock_boundary_reached
                        .lock()
                        .expect("hook state should lock") = true;
                }
                Ok(())
            });
            let sync_error = lock
                .sync_durable()
                .expect_err("replacement route must fence locked sync");
            assert_eq!(sync_error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                sync_error.stage(),
                GenerationRootStage::RevalidateRegularFile
            );
            assert!(
                !*sync_boundary_reached
                    .lock()
                    .expect("hook state should lock"),
                "route identity must be checked before the durability syscall"
            );
            let unlock_error = lock
                .unlock()
                .expect_err("route replacement remains visible during explicit unlock");
            assert_eq!(unlock_error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                unlock_error.stage(),
                GenerationRootStage::RevalidateRegularFile
            );
            assert!(
                *unlock_boundary_reached
                    .lock()
                    .expect("hook state should lock"),
                "route-validation failure must not skip the explicit LOCK_UN path"
            );
            let retained_lock = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&retained_lock_path)
                .expect("retained lock inode should reopen after failed validation");
            rustix::fs::flock(
                &retained_lock,
                rustix::fs::FlockOperation::NonBlockingLockExclusive,
            )
            .expect("failed validation must still release the retained inode lock");
            rustix::fs::flock(&retained_lock, rustix::fs::FlockOperation::Unlock)
                .expect("test lock cleanup should succeed");
        }

        #[test]
        fn unsupported_linux_filesystems_fail_before_admission() {
            let tmpfs_base = Path::new("/dev/shm");
            let tmpfs_stat =
                rustix::fs::statfs(tmpfs_base).expect("physical tmpfs fixture must be statable");
            assert_eq!(
                i128::from(tmpfs_stat.f_type),
                i128::from(libc::TMPFS_MAGIC),
                "unsupported-filesystem receipt requires a real tmpfs mount"
            );
            assert_eq!(
                QualifiedGenerationRoot::admit(tmpfs_base)
                    .expect_err("the tmpfs mount itself must fail the physical allowlist")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
            assert_eq!(
                QualifiedGenerationRoot::admit(Path::new("/proc"))
                    .expect_err("procfs root must fail the physical filesystem allowlist")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
        }

        #[test]
        fn xfs_remains_fail_closed_until_its_native_receipt_bead_closes() {
            let raw_type: rustix::fs::FsWord = libc::XFS_SUPER_MAGIC;
            assert_eq!(
                linux_filesystem_profile(raw_type, 1)
                    .expect_err("an unreceipted XFS profile must not reach mountinfo admission")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
        }

        #[test]
        fn statx_writable_truth_requires_and_rejects_immutable_and_append_attributes() {
            use rustix::fs::StatxAttributes;

            let required = StatxAttributes::IMMUTABLE | StatxAttributes::APPEND;
            validate_linux_writable_attributes_for_test(StatxAttributes::empty(), required)
                .expect("a reported writable object should pass");

            let missing_mask = validate_linux_writable_attributes_for_test(
                StatxAttributes::empty(),
                StatxAttributes::IMMUTABLE,
            )
            .expect_err("an unreported append attribute must fail closed");
            assert_eq!(
                missing_mask.kind(),
                GenerationRootErrorKind::UnsupportedKernelFeature
            );
            assert_eq!(missing_mask.stage(), GenerationRootStage::QualifyFilesystem);

            for attribute in [StatxAttributes::IMMUTABLE, StatxAttributes::APPEND] {
                let error = validate_linux_writable_attributes_for_test(attribute, required)
                    .expect_err("immutable or append-only objects are not writable truth");
                assert_eq!(error.kind(), GenerationRootErrorKind::WriteRestrictedObject);
                assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);
            }
        }

        #[test]
        fn stat_and_statx_device_components_must_match_exactly() {
            assert!(linux_device_components_match_for_test(8, 1, 8, 1));
            assert!(!linux_device_components_match_for_test(8, 1, 9, 1));
            assert!(!linux_device_components_match_for_test(8, 1, 8, 2));
        }

        #[test]
        fn proc_bridge_errno_classes_distinguish_limits_policy_and_identity_drift() {
            for error in [
                rustix::io::Errno::MFILE,
                rustix::io::Errno::NFILE,
                rustix::io::Errno::NOMEM,
            ] {
                assert_eq!(
                    proc_bridge_error_for_test(error).kind(),
                    GenerationRootErrorKind::ResourceLimit
                );
                assert_eq!(
                    linux_open_error_for_test(error).kind(),
                    GenerationRootErrorKind::ResourceLimit
                );
                let mountinfo_error = linux_mountinfo_error_for_test(error);
                assert_eq!(
                    mountinfo_error.kind(),
                    GenerationRootErrorKind::ResourceLimit
                );
                assert_eq!(mountinfo_error.raw_os_error(), Some(error.raw_os_error()));
            }
            let mountinfo_io = linux_mountinfo_error_for_test(rustix::io::Errno::IO);
            assert_eq!(mountinfo_io.kind(), GenerationRootErrorKind::Io);
            assert_eq!(
                mountinfo_io.raw_os_error(),
                Some(rustix::io::Errno::IO.raw_os_error())
            );
            for error in [rustix::io::Errno::PERM, rustix::io::Errno::ACCESS] {
                assert_eq!(
                    proc_bridge_error_for_test(error).kind(),
                    GenerationRootErrorKind::UnsupportedKernelFeature
                );
            }
            for error in [
                rustix::io::Errno::NOENT,
                rustix::io::Errno::NOTDIR,
                rustix::io::Errno::LOOP,
                rustix::io::Errno::XDEV,
                rustix::io::Errno::STALE,
            ] {
                assert_eq!(
                    proc_bridge_error_for_test(error).kind(),
                    GenerationRootErrorKind::ObjectChanged
                );
            }
            for error in [
                rustix::io::Errno::NOENT,
                rustix::io::Errno::NOTDIR,
                rustix::io::Errno::LOOP,
                rustix::io::Errno::XDEV,
            ] {
                assert_eq!(
                    proc_root_error_for_test(error).kind(),
                    GenerationRootErrorKind::UnsupportedKernelFeature
                );
            }
        }

        #[test]
        fn proc_bridge_requires_kernel_attestation_of_the_mount_id_field() {
            use rustix::fs::StatxFlags;

            assert_eq!(
                validate_proc_mount_id_mask_for_test(StatxFlags::MNT_ID, 47)
                    .expect("reported mount identity should pass"),
                47
            );
            let error = validate_proc_mount_id_mask_for_test(StatxFlags::BASIC_STATS, 47)
                .expect_err("an unreported mount-id value must never be trusted");
            assert_eq!(
                error.kind(),
                GenerationRootErrorKind::UnsupportedKernelFeature
            );
            assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
            assert_eq!(error.expected(), Some(u64::from(StatxFlags::MNT_ID.bits())));
            assert_eq!(error.observed(), Some(0));
        }

        #[test]
        fn proc_thread_self_binding_and_bridge_route_name_the_current_thread_table() {
            let process_id = std::process::id();
            let thread_id = rustix::thread::gettid();
            assert_eq!(
                fs::read_link("/proc/thread-self")
                    .expect("the kernel thread-self binding must be readable"),
                PathBuf::from(format!("{process_id}/task/{thread_id}"))
            );
            assert_eq!(
                proc_thread_fd_route_for_test(47),
                "thread-self/fd/47",
                "the production bridge must never route through the leader's self/fd table"
            );
            assert_eq!(
                linux_task_mountinfo_route_for_test(process_id, thread_id.as_raw_pid()),
                PathBuf::from(format!("{process_id}/task/{thread_id}/mountinfo")),
                "the verified proc-root-relative route must name the accessing thread namespace"
            );
        }

        #[test]
        #[ignore = "physical Linux CLONE_FILES regression must run as an exact isolated test"]
        #[allow(deprecated)]
        fn physical_linux_thread_self_bridge_survives_an_unshared_fd_table_collision() {
            use rustix::thread::UnshareFlags;
            use std::sync::mpsc;

            let root_path = fixture_root("thread-self-unshared-files");
            let artifact_path = private_file(&root_path, "artifact", b"thread-local capability");
            let (descriptor_sender, descriptor_receiver) = mpsc::sync_channel(0);
            let (continue_sender, continue_receiver) = mpsc::sync_channel(0);
            let worker = std::thread::spawn(move || {
                // rustix 1.1.4 retains this deprecated safe wrapper. This
                // worker owns every descriptor it opens after unsharing and
                // terminates immediately after the receipt, satisfying the
                // API's no-cross-table-descriptor-use invariant.
                rustix::thread::unshare(UnshareFlags::FILES)
                    .expect("the worker file table must unshare without privilege");
                let retained =
                    File::open(&artifact_path).expect("the worker-local artifact must open");
                let raw_fd = retained.as_raw_fd();
                descriptor_sender
                    .send(raw_fd)
                    .expect("the leader must receive the collision descriptor number");
                continue_receiver
                    .recv()
                    .expect("the leader must establish its decoy at the same fd number");

                assert_eq!(
                    fs::read_link(format!("/proc/self/fd/{raw_fd}"))
                        .expect("the leader-table proc route must resolve"),
                    PathBuf::from("/dev/null"),
                    "self/fd must demonstrate the unrelated leader-table collision"
                );
                assert_eq!(
                    fs::read_link(format!("/proc/thread-self/fd/{raw_fd}"))
                        .expect("the worker-table proc route must resolve"),
                    artifact_path,
                    "thread-self/fd must remain bound to the worker-local capability"
                );

                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("the private root must qualify after CLONE_FILES");
                let admitted = root
                    .admit_file(
                        &confined("artifact"),
                        immutable_expectation(b"thread-local capability"),
                    )
                    .expect("production admission must reopen through the worker descriptor table");
                assert_eq!(
                    admitted.bytes().as_ref(),
                    b"thread-local capability",
                    "the admitted bytes must come from the worker-local retained capability"
                );
            });

            let worker_fd = descriptor_receiver
                .recv()
                .expect("the worker must report its retained fd number");
            let leader_decoy = File::open("/dev/null").expect("the leader decoy must open");
            assert_eq!(
                leader_decoy.as_raw_fd(),
                worker_fd,
                "the isolated receipt requires an exact leader/worker fd-number collision"
            );
            continue_sender
                .send(())
                .expect("the worker must be released after collision setup");
            worker
                .join()
                .expect("the unshared-table worker must complete without panic");
        }

        #[test]
        #[ignore = "physical Linux CLONE_NEWNS receipt requires root and exact isolated execution"]
        #[allow(deprecated)]
        fn physical_linux_task_mountinfo_observes_the_calling_threads_private_mount_namespace() {
            use rustix::fs::{AtFlags, CWD, StatxFlags, statx};
            use rustix::thread::UnshareFlags;

            assert!(
                rustix::process::geteuid().is_root(),
                "the task-scoped mountinfo receipt requires CAP_SYS_ADMIN"
            );
            let mountpoint = private_dir(
                &fixture_root("task-mountinfo-private-namespace"),
                "bind-mount",
            );
            let worker = std::thread::spawn(move || {
                // NEWNS does not split the descriptor table. This isolated
                // worker performs all namespace-sensitive work itself and
                // exits immediately, so the private mounts disappear with
                // the last task holding the namespace.
                rustix::thread::unshare(UnshareFlags::NEWNS)
                    .expect("the worker must acquire a private mount namespace");
                let private_output = Command::new("mount")
                    .args(["--make-rprivate", "/"])
                    .output()
                    .expect("the physical receipt requires util-linux mount");
                assert!(
                    private_output.status.success(),
                    "making the worker namespace private failed: status={:?}, stdout={}, stderr={}",
                    private_output.status.code(),
                    String::from_utf8_lossy(&private_output.stdout),
                    String::from_utf8_lossy(&private_output.stderr)
                );
                let bind_output = Command::new("mount")
                    .arg("--bind")
                    .arg(&mountpoint)
                    .arg(&mountpoint)
                    .output()
                    .expect("the physical receipt requires util-linux mount");
                assert!(
                    bind_output.status.success(),
                    "worker-private bind mount failed: status={:?}, stdout={}, stderr={}",
                    bind_output.status.code(),
                    String::from_utf8_lossy(&bind_output.stdout),
                    String::from_utf8_lossy(&bind_output.stderr)
                );

                let extended = statx(
                    CWD,
                    &mountpoint,
                    AtFlags::NO_AUTOMOUNT,
                    StatxFlags::BASIC_STATS | StatxFlags::MNT_ID,
                )
                .expect("the worker-private bind mount must expose statx identity");
                assert!(
                    StatxFlags::from_bits_retain(extended.stx_mask).contains(StatxFlags::MNT_ID),
                    "the kernel must attest the worker-private mount id"
                );
                let filesystem = rustix::fs::statfs(&mountpoint)
                    .expect("the worker-private bind mount must be statable");
                let expected =
                    if i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC) {
                        b"ext4".as_slice()
                    } else if i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC) {
                        b"btrfs".as_slice()
                    } else {
                        panic!("the physical receipt requires ext4 or Btrfs");
                    };

                let leader_bytes = fs::read("/proc/self/mountinfo")
                    .expect("the leader mount table must remain readable");
                assert!(
                    parse_mountinfo_filesystem(&leader_bytes, extended.stx_mnt_id).is_err(),
                    "the leader table must not contain the worker-private bind mount"
                );
                let task_path = linux_task_mountinfo_route_for_test(
                    std::process::id(),
                    rustix::thread::gettid().as_raw_pid(),
                );
                let task_bytes = fs::read(Path::new("/proc").join(&task_path))
                    .expect("the worker task mount table must be readable");
                assert_eq!(
                    parse_mountinfo_filesystem(&task_bytes, extended.stx_mnt_id)
                        .expect("the worker task table must contain its private bind mount"),
                    expected
                );
                assert_eq!(
                    linux_mountinfo_filesystem_for_test(extended.stx_mnt_id)
                        .expect("production mountinfo lookup must use the worker task table"),
                    expected,
                    "the production route must attest the accessing thread's mount namespace"
                );
            });
            worker
                .join()
                .expect("the private-mount-namespace worker must complete without panic");
        }

        #[test]
        #[ignore = "physical nested-PID receipt requires root, util-linux unshare, and exact isolated execution"]
        fn physical_linux_foreign_procfs_pid_namespace_fails_before_mountinfo_open() {
            const CHILD_FLAG: &str = "FRANKENSEARCH_FOREIGN_PROCFS_PIDNS_CHILD";
            const ROOT_PATH: &str = "FRANKENSEARCH_FOREIGN_PROCFS_PIDNS_ROOT";
            const TEST_NAME: &str = "generation_root::tests::linux::physical_linux_foreign_procfs_pid_namespace_fails_before_mountinfo_open";

            assert!(
                rustix::process::geteuid().is_root(),
                "the foreign-procfs PID-namespace receipt requires root"
            );
            if std::env::var_os(CHILD_FLAG).is_none() {
                let root_path = fixture_root("foreign-procfs-pid-namespace");
                let executable =
                    std::env::current_exe().expect("the current test ELF path must be available");
                let output = Command::new("unshare")
                    .args(["--pid", "--fork"])
                    .arg(executable)
                    .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
                    .env(CHILD_FLAG, "1")
                    .env(ROOT_PATH, &root_path)
                    .output()
                    .expect("the physical receipt requires util-linux unshare");
                assert!(
                    output.status.success(),
                    "nested-PID child failed: status={:?}, stdout={}, stderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                return;
            }

            assert_eq!(
                std::process::id(),
                1,
                "the nested PID-namespace test process must be namespace PID 1"
            );
            assert_ne!(
                fs::read_link("/proc/self")
                    .expect("the inherited outer procfs self link must read"),
                PathBuf::from("1"),
                "the receipt must retain a procfs mounted for the outer PID namespace"
            );
            let root_path = PathBuf::from(
                std::env::var_os(ROOT_PATH)
                    .expect("the parent must pass the persistent root fixture"),
            );
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&root_path)
                .expect_err("foreign PID-namespace procfs must fail closed");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);
            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(
                observed.contains(&TestBoundary::BeforeProcCapabilityRootOpen),
                "the verified proc-root qualification must be attempted"
            );
            assert!(
                !observed.contains(&TestBoundary::BeforeMountInfoOpen),
                "foreign procfs must fail binding before the mountinfo file is opened"
            );
            drop(observed);
        }

        #[test]
        #[ignore = "run the test ELF as root with FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT on native ext4/Btrfs"]
        fn physical_linux_proc_fd_overmount_is_rejected_before_data_open() {
            const CHILD_FLAG: &str = "FRANKENSEARCH_PROC_OVERMOUNT_CHILD";
            const TEST_NAME: &str = "generation_root::tests::linux::physical_linux_proc_fd_overmount_is_rejected_before_data_open";

            assert!(
                rustix::process::geteuid().is_root(),
                "physical procfs overmount receipt must run as root"
            );
            let base = std::env::var_os("FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT")
                .map(PathBuf::from)
                .expect(
                    "FRANKENSEARCH_LINUX_PROC_OVERMOUNT_TEST_ROOT must identify native ext4/Btrfs",
                );
            if std::env::var_os(CHILD_FLAG).is_none() {
                let executable =
                    std::env::current_exe().expect("the current test ELF path must be available");
                let output = Command::new("unshare")
                    .args(["--mount", "--fork", "--propagation", "private"])
                    .arg(executable)
                    .args(["--exact", TEST_NAME, "--ignored", "--nocapture"])
                    .env(CHILD_FLAG, "1")
                    .output()
                    .expect("the physical receipt requires util-linux unshare");
                assert!(
                    output.status.success(),
                    "mount-namespace child failed: status={:?}, stdout={}, stderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                return;
            }

            let filesystem =
                rustix::fs::statfs(&base).expect("physical procfs fixture base must be statable");
            assert!(
                i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
                    || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
                "physical procfs receipt requires ext4 or Btrfs"
            );
            let root_path = fixture_root_under(&base, "proc-fd-overmount");
            private_file(&root_path, "artifact", b"sealed");
            let fifo = root_path.join("proc-fd-overmount-fifo");
            rustix::fs::mkfifoat(
                rustix::fs::CWD,
                &fifo,
                rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
            )
            .expect("the hostile overmount fixture FIFO must be creatable");
            let decoy = private_dir(&root_path, "forged-proc-fd-decoy");
            for descriptor in 0_u32..=1024 {
                symlink(&fifo, decoy.join(descriptor.to_string()))
                    .expect("each likely descriptor route must have a forged FIFO symlink");
            }
            let before_error = OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&fifo)
                .expect_err("the hostile FIFO must have no reader before the bridge attempt");
            assert_eq!(
                before_error.raw_os_error(),
                Some(libc::ENXIO),
                "the hostile FIFO fixture must prove its no-reader precondition"
            );
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let process_id = std::process::id();
            let thread_id = rustix::thread::gettid();
            let proc_fd = PathBuf::from(format!("/proc/{process_id}/task/{thread_id}/fd"));
            let mount = Command::new("mount")
                .args(["--bind"])
                .arg(&decoy)
                .arg(&proc_fd)
                .output()
                .expect("the physical receipt requires mount");
            assert!(
                mount.status.success(),
                "proc-fd bind overmount failed: status={:?}, stdout={}, stderr={}",
                mount.status.code(),
                String::from_utf8_lossy(&mount.stdout),
                String::from_utf8_lossy(&mount.stderr)
            );

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let error = root
                .admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
                .expect_err("an overmounted proc-fd bridge must fail closed");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(
                observed.contains(&TestBoundary::BeforeProcBridgeProbe {
                    index: 0,
                    phase: ProcBridgeProbePhase::BeforeDataOpen,
                }),
                "the exact proc-fd bridge preflight must be attempted"
            );
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::AfterProcBridgeProbe { .. }
                        | TestBoundary::AfterProcFdReopen { .. }
                        | TestBoundary::BeforeRead { .. }
                )),
                "the overmount must fail before the data descriptor or content path"
            );
            drop(observed);
            let after_error = OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&fifo)
                .expect_err("the hostile FIFO must still have no reader after bridge rejection");
            assert_eq!(
                after_error.raw_os_error(),
                Some(libc::ENXIO),
                "bridge rejection must not establish a reader on any forged numeric fd route"
            );
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT on native ext4/Btrfs, chattr, and CAP_LINUX_IMMUTABLE"]
        fn physical_statx_immutable_and_append_attributes_fail_before_data_open_or_lock() {
            use rustix::fs::{AtFlags, StatxAttributes, StatxFlags, statx};

            let base = std::env::var_os("FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_LINUX_CHATTR_TEST_ROOT must identify native ext4 or Btrfs");
            assert!(
                base.is_absolute(),
                "the physical attribute fixture route must be absolute"
            );
            let filesystem =
                rustix::fs::statfs(&base).expect("physical attribute mount must be statable");
            assert!(
                i128::from(filesystem.f_type) == i128::from(libc::EXT4_SUPER_MAGIC)
                    || i128::from(filesystem.f_type) == i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on native ext4 or Btrfs"
            );

            let root_path = fixture_root_under(&base, "physical-statx-attributes");
            let immutable_contents = b"immutable physical attribute";
            let immutable_path = private_file(&root_path, "immutable", immutable_contents);
            let append_path = control_file(&root_path, "append-control", b"lock-v1!");
            let root = QualifiedGenerationRoot::admit(&root_path)
                .expect("the writable native root must qualify before object attributes are set");

            run_chattr_fixture_command("+i", &immutable_path);
            let mut immutable_reset = ChattrReset::new("-i", &immutable_path);
            run_chattr_fixture_command("+a", &append_path);
            let mut append_reset = ChattrReset::new("-a", &append_path);

            for (path, attribute) in [
                (&immutable_path, StatxAttributes::IMMUTABLE),
                (&append_path, StatxAttributes::APPEND),
            ] {
                let descriptor =
                    File::open(path).expect("attributed fixture must remain readable for proof");
                let observed = statx(
                    &descriptor,
                    "",
                    AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
                    StatxFlags::BASIC_STATS,
                )
                .expect("physical attributed descriptor must support statx");
                assert!(
                    observed.stx_attributes_mask.contains(attribute),
                    "the kernel must report support for the tested physical attribute"
                );
                assert!(
                    observed.stx_attributes.contains(attribute),
                    "the physical attribute must be set before the rejection receipt"
                );
            }

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let immutable_error = root
                .admit_file(
                    &confined("immutable"),
                    immutable_expectation(immutable_contents),
                )
                .expect_err("a physically immutable artifact must fail writable qualification");
            assert_eq!(
                immutable_error.kind(),
                GenerationRootErrorKind::WriteRestrictedObject
            );
            assert_eq!(
                immutable_error.stage(),
                GenerationRootStage::OpenRegularFile
            );
            let append_error = root
                .admit_control_file(
                    &confined("append-control"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation must be accepted"),
                )
                .expect_err("a physically append-only control file must fail qualification");
            assert_eq!(
                append_error.kind(),
                GenerationRootErrorKind::WriteRestrictedObject
            );
            assert_eq!(append_error.stage(), GenerationRootStage::OpenRegularFile);
            let observed = boundaries.lock().expect("boundary log should lock");
            // Root revalidation legitimately opens the verified procfs root
            // to bind task-scoped mountinfo before inspecting the attributed
            // component. `BeforeProcFdReopen` is the distinct boundary that
            // would derive a data descriptor from the qualified `O_PATH`
            // probe.
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::BeforeProcFdReopen { .. }
                        | TestBoundary::BeforeRead { .. }
                        | TestBoundary::BeforeLock
                )),
                "physical immutable/append rejection must precede data-fd derivation, reads, and \
                 flock; observed={observed:?}"
            );
            drop(observed);
            drop(guard);

            immutable_reset.clear();
            append_reset.clear();
            root.admit_file(
                &confined("immutable"),
                immutable_expectation(immutable_contents),
            )
            .expect("the same artifact must qualify after the immutable attribute is cleared");
            root.admit_control_file(
                &confined("append-control"),
                GenerationFileExpectation::control(8)
                    .expect("small control expectation must be accepted"),
            )
            .expect("the same control must qualify after the append-only attribute is cleared")
            .try_lock(GenerationRootLockMode::Exclusive)
            .expect("the cleared control file must reach flock")
            .unlock()
            .expect("the cleared control lock must release");
        }

        #[test]
        fn stat_field_normalization_is_lossless_across_32_and_64_bit_abi_widths() {
            assert_eq!(stat_mode_as_u32(u16::MAX), u32::from(u16::MAX));
            assert_eq!(stat_mode_as_u32(u32::MAX), u32::MAX);
            assert_eq!(stat_inode_as_u64(u32::MAX), u64::from(u32::MAX));
            assert_eq!(stat_inode_as_u64(u64::MAX), u64::MAX);
            assert_eq!(stat_link_count_as_u64(u32::MAX), u64::from(u32::MAX));
            assert_eq!(stat_link_count_as_u64(u64::MAX), u64::MAX);
            assert_eq!(stat_seconds_as_i64(i32::MIN), i64::from(i32::MIN));
            assert_eq!(stat_seconds_as_i64(i32::MAX), i64::from(i32::MAX));
            assert_eq!(stat_seconds_as_i64(i64::MIN), i64::MIN);
            assert_eq!(stat_seconds_as_i64(i64::MAX), i64::MAX);
        }

        #[test]
        fn btrfs_namespace_identity_binds_mount_and_subvolume() {
            let raw_type: rustix::fs::FsWord = libc::BTRFS_SUPER_MAGIC;
            let top_level = linux_filesystem_namespace_digest(raw_type, 41, Some(5));
            let nested = linux_filesystem_namespace_digest(raw_type, 41, Some(256));
            let other_mount = linux_filesystem_namespace_digest(raw_type, 42, Some(5));

            assert_ne!(
                top_level, nested,
                "one VFS mount must not collapse distinct Btrfs subvolumes"
            );
            assert_ne!(
                top_level, other_mount,
                "the namespace identity must continue to bind the VFS mount"
            );
            assert_eq!(
                top_level,
                linux_filesystem_namespace_digest(raw_type, 41, Some(5)),
                "the namespace identity must be deterministic"
            );
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT plus mount, umount, and CAP_SYS_ADMIN"]
        fn physical_absolute_route_accepts_bind_transition_and_rejects_same_source_remount() {
            let base = std::env::var_os("FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT")
                .map(PathBuf::from)
                .expect(
                    "FRANKENSEARCH_LINUX_ABSOLUTE_MOUNT_TEST_ROOT must identify a secure native ext4/Btrfs ancestor",
                );
            assert!(
                base.is_absolute(),
                "the physical absolute-route receipt must use an absolute trusted ancestor"
            );

            let receipt_root = fixture_root_under(&base, "physical-absolute-bind-transition");
            let source = private_dir(&receipt_root, "source");
            let source_root = private_dir(&source, "qualified");
            let lock_bytes = b"lock-v1!";
            control_file(&source_root, "LOCK", lock_bytes);
            control_file(&source_root, "AUTHORITY", &authority_image(0x5a));
            let layout = GenerationRootAnchorLayout::new(
                u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
            )
            .expect("the named lock fixture length is bounded");
            let target = private_dir(&receipt_root, "mountpoint");
            let bind_mount = BindMountFixture::new(&source, &target);
            let mounted_root = target.join("qualified");

            let direct = QualifiedGenerationRoot::admit(&source_root)
                .expect("the source root should independently qualify");
            let through_bind = QualifiedGenerationRoot::admit(&mounted_root)
                .expect("an absolute route may cross into a qualified adjacent bind mount");
            assert_eq!(
                direct.witness().inode(),
                through_bind.witness().inode(),
                "the bind route must resolve the same underlying final directory"
            );
            assert_ne!(
                direct.witness().mount_identity(),
                through_bind.witness().mount_identity(),
                "the route witness must distinguish the adjacent bind mount"
            );
            let named = NamedGenerationRoot::admit(&mounted_root, layout)
                .expect("fixed anchors should qualify through the adjacent bind mount");
            named
                .read_guard()
                .expect("the named LOCK should acquire through the qualified mount transition")
                .release()
                .expect("the unchanged mounted guard should release");
            drop(named);
            drop(through_bind);
            drop(direct);

            let swapped = Arc::new(Mutex::new(false));
            let hook_swapped = Arc::clone(&swapped);
            let mut hook_mount = bind_mount;
            let hook = install_test_hook(move |boundary| {
                if !*hook_swapped.lock().expect("mount-swap state should lock")
                    && boundary == TestBoundary::AfterFirstRootQualification
                {
                    hook_mount.remount_same_source();
                    *hook_swapped.lock().expect("mount-swap state should lock") = true;
                }
                Ok(())
            });
            let error = NamedGenerationRoot::admit(&mounted_root, layout)
                .expect_err("a new mount identity between the two absolute walks must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::RevalidateRootRoute);
            assert!(
                *swapped.lock().expect("mount-swap state should lock"),
                "the physical receipt must execute the remount boundary"
            );
            drop(hook);
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT on a writable native NFS mount"]
        fn physical_linux_network_mount_fails_before_data_open_or_mutation() {
            let network_root = std::env::var_os("FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_LINUX_NETWORK_TEST_ROOT must identify the native NFS mount");
            assert!(
                network_root.is_absolute(),
                "the physical network receipt must use an absolute provider route"
            );
            let filesystem = rustix::fs::statfs(&network_root)
                .expect("the physical network mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::NFS_SUPER_MAGIC),
                "the provider route must be a real NFS mount"
            );
            let mount = rustix::fs::statvfs(&network_root)
                .expect("the physical network mount flags must be readable");
            assert!(
                !mount.f_flag.contains(rustix::fs::StatVfsMountFlags::RDONLY),
                "the provider NFS mount must be writable so this receipt isolates filesystem-type rejection"
            );

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _hook = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("network boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&network_root)
                .expect_err("a network filesystem must fail qualification");
            assert_eq!(error.kind(), GenerationRootErrorKind::UnsupportedFilesystem);
            assert_eq!(error.stage(), GenerationRootStage::QualifyFilesystem);

            let observed = boundaries.lock().expect("network boundary log should lock");
            assert!(
                observed.contains(&TestBoundary::BeforeFilesystemQualification),
                "the receipt must reach the descriptor-bound filesystem classifier"
            );
            assert!(
                !observed.contains(&TestBoundary::AfterFilesystemQualification),
                "an NFS root must not cross the qualification boundary"
            );
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::BeforeRegularFileOpen { .. }
                        | TestBoundary::BeforeRead { .. }
                        | TestBoundary::BeforeLock
                        | TestBoundary::BeforeFileSync
                        | TestBoundary::BeforeDirectorySync
                )),
                "NFS rejection must precede every regular-file, content, lock, and durability boundary"
            );
            drop(observed);
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT on a writable native Btrfs mount"]
        fn physical_btrfs_retained_files_locks_and_durability_succeed() {
            let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
            assert!(
                base.is_absolute(),
                "the physical Btrfs mount route must be absolute"
            );
            let filesystem =
                rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on a real Btrfs mount"
            );

            let root_path = fixture_root_under(&base, "physical-btrfs");
            let generation = private_dir(&root_path, "generation");
            let contents = b"physical Btrfs generation";
            private_file(&generation, "vector.fsvi", contents);
            control_file(&root_path, "LOCK", b"lock-v1!");

            let root = QualifiedGenerationRoot::admit(&root_path)
                .expect("writable Btrfs root must qualify");
            assert_eq!(
                root.witness().filesystem(),
                super::super::QualifiedFilesystem::LinuxBtrfs
            );
            let admitted = root
                .admit_file(
                    &confined("generation/vector.fsvi"),
                    immutable_expectation(contents),
                )
                .expect("exact immutable Btrfs file must qualify");
            assert_eq!(admitted.as_bytes(), contents);
            admitted
                .sync_durable()
                .expect("retained Btrfs regular-file fsync must succeed");

            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation must be accepted"),
                )
                .expect("Btrfs control file must qualify");
            let lock = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("Btrfs control-file flock must succeed");
            lock.sync_durable()
                .expect("locked Btrfs control-file fsync must succeed");
            lock.unlock()
                .expect("Btrfs control-file unlock must succeed");
            root.sync_directory_durable()
                .expect("retained Btrfs directory fsync must succeed");
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
        fn physical_btrfs_nested_subvolume_is_a_confined_namespace_boundary() {
            let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
            let filesystem =
                rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on a real Btrfs mount"
            );

            let root_path = fixture_root_under(&base, "physical-btrfs-nested-subvolume");
            let nested_path = root_path.join("nested");
            run_btrfs_fixture_command(&[
                OsStr::new("subvolume"),
                OsStr::new("create"),
                nested_path.as_os_str(),
            ]);
            fs::set_permissions(&nested_path, fs::Permissions::from_mode(0o700))
                .expect("nested Btrfs subvolume mode must be private");
            let contents = b"nested subvolume must not cross the root namespace";
            private_file(&nested_path, "vector.fsvi", contents);

            let root = QualifiedGenerationRoot::admit(&root_path)
                .expect("the containing writable Btrfs root must qualify");
            assert_eq!(
                root.admit_file(
                    &confined("nested/vector.fsvi"),
                    immutable_expectation(contents),
                )
                .expect_err("a nested Btrfs subvolume must be a confinement boundary")
                .kind(),
                GenerationRootErrorKind::CrossDevice
            );
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
        fn physical_btrfs_snapshot_substitution_invalidates_the_retained_route() {
            let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
            let filesystem =
                rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on a real Btrfs mount"
            );

            let parent_path = fixture_root_under(&base, "physical-btrfs-snapshot-substitution");
            let live_path = parent_path.join("live");
            run_btrfs_fixture_command(&[
                OsStr::new("subvolume"),
                OsStr::new("create"),
                live_path.as_os_str(),
            ]);
            fs::set_permissions(&live_path, fs::Permissions::from_mode(0o700))
                .expect("live Btrfs subvolume mode must be private");
            private_file(&live_path, "manifest", b"sealed generation");

            let retained = QualifiedGenerationRoot::admit(&live_path)
                .expect("the original Btrfs subvolume must qualify");
            let retained_path = parent_path.join("retained");
            fs::rename(&live_path, &retained_path)
                .expect("the original subvolume must move to an unused persistent route");
            run_btrfs_fixture_command(&[
                OsStr::new("subvolume"),
                OsStr::new("snapshot"),
                retained_path.as_os_str(),
                live_path.as_os_str(),
            ]);

            let replacement = QualifiedGenerationRoot::admit(&live_path)
                .expect("the replacement snapshot is independently well-formed");
            assert_ne!(
                retained.witness().mount_identity(),
                replacement.witness().mount_identity(),
                "snapshot and source must have distinct namespace identities"
            );
            assert_eq!(
                retained
                    .revalidate_route()
                    .expect_err("snapshot substitution must invalidate the retained route")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_BTRFS_TEST_ROOT, btrfs-progs, and CAP_SYS_ADMIN"]
        fn physical_read_only_btrfs_subvolume_fails_on_a_writable_mount() {
            use rustix::fs::StatVfsMountFlags;

            let base = std::env::var_os("FRANKENSEARCH_BTRFS_TEST_ROOT")
                .map(PathBuf::from)
                .expect("FRANKENSEARCH_BTRFS_TEST_ROOT must identify the native Btrfs mount");
            let filesystem =
                rustix::fs::statfs(&base).expect("physical Btrfs mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on a real Btrfs mount"
            );

            let parent_path = fixture_root_under(&base, "physical-btrfs-read-only-subvolume");
            let readonly_path = parent_path.join("readonly");
            run_btrfs_fixture_command(&[
                OsStr::new("subvolume"),
                OsStr::new("create"),
                readonly_path.as_os_str(),
            ]);
            fs::set_permissions(&readonly_path, fs::Permissions::from_mode(0o700))
                .expect("read-only fixture mode must be private before sealing");
            run_btrfs_fixture_command(&[
                OsStr::new("property"),
                OsStr::new("set"),
                OsStr::new("-t"),
                OsStr::new("subvol"),
                readonly_path.as_os_str(),
                OsStr::new("ro"),
                OsStr::new("true"),
            ]);
            let readonly_descriptor =
                File::open(&readonly_path).expect("read-only fixture must be openable");
            let write_error = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(readonly_path.join("writability-probe"))
                .expect_err("the receipt requires a genuinely read-only Btrfs subvolume");
            assert_eq!(
                write_error.raw_os_error(),
                Some(libc::EROFS),
                "the read-only proof must be the kernel EROFS result"
            );
            let mount = rustix::fs::fstatvfs(&readonly_descriptor)
                .expect("Btrfs mount flags must be readable");
            assert!(
                !mount.f_flag.contains(StatVfsMountFlags::RDONLY),
                "the enclosing VFS mount must remain writable so this proves subvolume handling"
            );
            assert_eq!(
                QualifiedGenerationRoot::admit(&readonly_path)
                    .expect_err("read-only Btrfs subvolume must fail admission")
                    .kind(),
                GenerationRootErrorKind::ReadOnlyFilesystem
            );
        }

        #[test]
        #[ignore = "requires FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT on a native read-only Btrfs mount"]
        fn physical_read_only_btrfs_fails_before_security_admission() {
            use rustix::fs::StatVfsMountFlags;

            let base = std::env::var_os("FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT")
                .map(PathBuf::from)
                .expect(
                    "FRANKENSEARCH_BTRFS_READ_ONLY_TEST_ROOT must identify the native Btrfs mount",
                );
            assert!(
                base.is_absolute(),
                "the physical read-only Btrfs mount route must be absolute"
            );
            let filesystem =
                rustix::fs::statfs(&base).expect("physical read-only Btrfs mount must be statable");
            assert_eq!(
                i128::from(filesystem.f_type),
                i128::from(libc::BTRFS_SUPER_MAGIC),
                "the receipt must execute on a real Btrfs mount"
            );
            let descriptor =
                File::open(&base).expect("physical read-only Btrfs root must be openable");
            let mount = rustix::fs::fstatvfs(&descriptor)
                .expect("physical read-only Btrfs mount flags must be readable");
            assert!(
                mount.f_flag.contains(StatVfsMountFlags::RDONLY),
                "the receipt requires a genuinely read-only mount"
            );
            assert_eq!(
                QualifiedGenerationRoot::admit(&base)
                    .expect_err("read-only Btrfs must fail before owner or mode admission")
                    .kind(),
                GenerationRootErrorKind::ReadOnlyFilesystem
            );
        }

        #[test]
        fn mountinfo_parser_requires_one_exact_well_formed_mount_record() {
            let fixture = b"41 1 8:1 / /other rw,relatime - xfs /dev/other rw\n\
                            42 1 8:2 / /target rw,relatime shared:7 - ext4 /dev/root rw\n";
            assert_eq!(
                parse_mountinfo_filesystem(fixture, 42)
                    .expect("one exact mount-id record should qualify"),
                b"ext4"
            );

            for malformed in [
                b"42 1 8:2 / /target rw,relatime ext4 /dev/root rw\n".as_slice(),
                b"42 1 8:2 / /target rw,relatime -  /dev/root rw\n".as_slice(),
                b"42 1 8:2 / /target rw,relatime - ext4  rw\n".as_slice(),
                b"42 1 8:2 / /target rw,relatime - ext4 /dev/root \n".as_slice(),
            ] {
                assert_eq!(
                    parse_mountinfo_filesystem(malformed, 42)
                        .expect_err("malformed matching mountinfo record must fail")
                        .kind(),
                    GenerationRootErrorKind::UnsupportedFilesystem
                );
            }

            let duplicate = b"42 1 8:2 / /a rw - ext4 /dev/a rw\n\
                              42 1 8:2 / /b rw - ext4 /dev/b rw\n";
            assert_eq!(
                parse_mountinfo_filesystem(duplicate, 42)
                    .expect_err("duplicate mount-id records must fail")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
            assert_eq!(
                parse_mountinfo_filesystem(
                    b"18446744073709551616 1 8:2 / /target rw - ext4 /dev/root rw\n",
                    u64::MAX,
                )
                .expect_err("overflowing mount ids must never alias a valid u64")
                .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
            assert_eq!(
                parse_mountinfo_filesystem(fixture, 99)
                    .expect_err("missing mount-id record must fail")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
        }

        #[test]
        fn directory_and_mountinfo_resource_boundaries_report_exact_numeric_counts() {
            validate_directory_scan_counts_for_test(
                GENERATION_ROOT_MAX_DIRECTORY_ENTRIES,
                GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES,
            )
            .expect("both exact directory-scan ceilings should be accepted");

            let entry_overflow = validate_directory_scan_counts_for_test(
                GENERATION_ROOT_MAX_DIRECTORY_ENTRIES.saturating_add(1),
                0,
            )
            .expect_err("one overflow entry must fail");
            assert_eq!(
                entry_overflow.kind(),
                GenerationRootErrorKind::ResourceLimit
            );
            assert_eq!(
                entry_overflow.expected(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES)
                        .expect("entry ceiling fits u64")
                )
            );
            assert_eq!(
                entry_overflow.observed(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_ENTRIES.saturating_add(1))
                        .expect("entry sentinel fits u64")
                )
            );

            let name_overflow = validate_directory_scan_counts_for_test(
                1,
                GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES.saturating_add(1),
            )
            .expect_err("one overflow name byte must fail");
            assert_eq!(name_overflow.kind(), GenerationRootErrorKind::ResourceLimit);
            assert_eq!(
                name_overflow.expected(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES)
                        .expect("name-byte ceiling fits u64")
                )
            );
            assert_eq!(
                name_overflow.observed(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_DIRECTORY_NAME_BYTES.saturating_add(1))
                        .expect("name-byte sentinel fits u64")
                )
            );

            let (last_request, accepted) =
                validate_mountinfo_bound_for_test(GENERATION_ROOT_MAX_MOUNTINFO_BYTES - 1, 1);
            assert_eq!(last_request, 2, "one accepted byte plus one sentinel");
            accepted.expect("the exact mountinfo storage ceiling should be accepted");
            let (sentinel_request, overflow) =
                validate_mountinfo_bound_for_test(GENERATION_ROOT_MAX_MOUNTINFO_BYTES, 1);
            assert_eq!(sentinel_request, 1, "only an overflow sentinel may be read");
            let overflow = overflow.expect_err("the mountinfo sentinel byte must fail");
            assert_eq!(overflow.kind(), GenerationRootErrorKind::ResourceLimit);
            assert_eq!(
                overflow.expected(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_MOUNTINFO_BYTES)
                        .expect("mountinfo ceiling fits u64")
                )
            );
            assert_eq!(
                overflow.observed(),
                Some(
                    u64::try_from(GENERATION_ROOT_MAX_MOUNTINFO_BYTES.saturating_add(1))
                        .expect("mountinfo sentinel fits u64")
                )
            );
        }

        #[test]
        fn every_linux_open_read_lock_and_sync_boundary_is_fault_injectable() {
            for target in [
                TestBoundary::BeforeRootComponentOpen { index: 0 },
                TestBoundary::AfterRootComponentOpen { index: 0 },
                TestBoundary::BeforeExactNameEnumeration,
                TestBoundary::AfterExactNameEnumeration,
                TestBoundary::BeforeRouteMountIdentity,
                TestBoundary::AfterRouteMountIdentity,
                TestBoundary::BeforeFilesystemQualification,
                TestBoundary::AfterFilesystemQualification,
                TestBoundary::BeforeObjectWitness,
                TestBoundary::AfterObjectWitness,
                TestBoundary::BeforeProcCapabilityRootOpen,
                TestBoundary::AfterProcCapabilityRootOpen,
                TestBoundary::BeforeProcSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::AfterProcSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::BeforeProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::AfterProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::BeforeMountInfoOpen,
                TestBoundary::AfterMountInfoOpen,
                TestBoundary::BeforeMountInfoRead,
                TestBoundary::BeforeProcSelfReadlink {
                    phase: ProcSelfBindingPhase::MountInfoRead,
                },
                TestBoundary::AfterProcSelfReadlink {
                    phase: ProcSelfBindingPhase::MountInfoRead,
                },
                TestBoundary::BeforeProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::MountInfoRead,
                },
                TestBoundary::AfterProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::MountInfoRead,
                },
                TestBoundary::AfterFirstRootQualification,
                TestBoundary::BeforeAclRead,
                TestBoundary::AfterAclRead,
            ] {
                let root_path = fixture_root("root-boundary");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::OpenRootRoute));
                    }
                    Ok(())
                });
                let error = QualifiedGenerationRoot::admit(&root_path)
                    .expect_err("injected root-open boundary must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::Io);
            }

            let contents = b"boundary";
            for target in [
                TestBoundary::BeforeRootDescriptorDuplicate,
                TestBoundary::AfterRootDescriptorDuplicate,
                TestBoundary::BeforeRelativeDirectoryOpen { index: 0 },
                TestBoundary::AfterRelativeDirectoryOpen { index: 0 },
                TestBoundary::BeforeRegularFileOpen { index: 1 },
                TestBoundary::BeforeFinalProbeOpen { index: 1 },
                TestBoundary::AfterFinalProbeOpen { index: 1 },
                TestBoundary::AfterProbeQualified { index: 1 },
                TestBoundary::BeforeProcCapabilityRootOpen,
                TestBoundary::AfterProcCapabilityRootOpen,
                TestBoundary::BeforeProcSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::AfterProcSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::BeforeProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::AfterProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::RootQualification,
                },
                TestBoundary::BeforeProcBridgeProbe {
                    index: 1,
                    phase: ProcBridgeProbePhase::BeforeDataOpen,
                },
                TestBoundary::AfterProcBridgeProbe {
                    index: 1,
                    phase: ProcBridgeProbePhase::BeforeDataOpen,
                },
                TestBoundary::BeforeProcSelfReadlink {
                    phase: ProcSelfBindingPhase::BeforeDataOpen,
                },
                TestBoundary::AfterProcSelfReadlink {
                    phase: ProcSelfBindingPhase::BeforeDataOpen,
                },
                TestBoundary::BeforeProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::BeforeDataOpen,
                },
                TestBoundary::AfterProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::BeforeDataOpen,
                },
                TestBoundary::BeforeProcFdReopen { index: 1 },
                TestBoundary::BeforeProcSelfReadlink {
                    phase: ProcSelfBindingPhase::AfterDataOpen,
                },
                TestBoundary::AfterProcSelfReadlink {
                    phase: ProcSelfBindingPhase::AfterDataOpen,
                },
                TestBoundary::BeforeProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::AfterDataOpen,
                },
                TestBoundary::AfterProcThreadSelfReadlink {
                    phase: ProcSelfBindingPhase::AfterDataOpen,
                },
                TestBoundary::BeforeProcBridgeProbe {
                    index: 1,
                    phase: ProcBridgeProbePhase::AfterDataOpen,
                },
                TestBoundary::AfterProcBridgeProbe {
                    index: 1,
                    phase: ProcBridgeProbePhase::AfterDataOpen,
                },
                TestBoundary::AfterProcFdReopen { index: 1 },
                TestBoundary::BeforeFinalRouteProbe { index: 1 },
                TestBoundary::AfterFinalRouteProbe { index: 1 },
                TestBoundary::AfterRegularFileOpen {
                    index: 1,
                    cloexec: true,
                },
                TestBoundary::AfterQualifiedFileOpen,
                TestBoundary::BeforeAclRead,
                TestBoundary::AfterAclRead,
                TestBoundary::BeforeRead { offset: 0 },
                TestBoundary::AfterRead {
                    offset: 0,
                    byte_count: contents.len(),
                },
                TestBoundary::BeforeTrailingByteProbe,
                TestBoundary::AfterTrailingByteProbe { byte_count: 0 },
                TestBoundary::AfterExactRead,
                TestBoundary::BeforeFileRouteReopen,
                TestBoundary::AfterFileRouteReopen,
                TestBoundary::BeforeFinalRootRevalidation,
            ] {
                let root_path = fixture_root("file-boundary");
                let generation = private_dir(&root_path, "generation");
                private_file(&generation, "artifact", contents);
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("private root should qualify before injection");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::ReadRegularFile));
                    }
                    Ok(())
                });
                let error = root
                    .admit_file(
                        &confined("generation/artifact"),
                        immutable_expectation(contents),
                    )
                    .expect_err("injected file boundary must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::Io);
            }

            {
                let root_path = fixture_root("mountinfo-read-complete-boundary");
                let _guard = install_test_hook(|boundary| {
                    if matches!(boundary, TestBoundary::AfterMountInfoRead { .. }) {
                        return Err(injected(GenerationRootStage::QualifyFilesystem));
                    }
                    Ok(())
                });
                assert_eq!(
                    QualifiedGenerationRoot::admit(&root_path)
                        .expect_err("completed mountinfo read boundary must be injectable")
                        .kind(),
                    GenerationRootErrorKind::Io
                );
            }

            for target in [TestBoundary::BeforeLock, TestBoundary::AfterLock] {
                let root_path = fixture_root("lock-boundary");
                control_file(&root_path, "LOCK", b"lock-v1!");
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("private root should qualify before injection");
                let control = root
                    .admit_control_file(
                        &confined("LOCK"),
                        GenerationFileExpectation::control(8)
                            .expect("small expectation should be accepted"),
                    )
                    .expect("control file should qualify before lock injection");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::AcquireLock));
                    }
                    Ok(())
                });
                let error = control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("injected lock boundary must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::Io);
            }

            for target in [TestBoundary::BeforeUnlock, TestBoundary::AfterUnlock] {
                let root_path = fixture_root("unlock-boundary");
                control_file(&root_path, "LOCK", b"lock-v1!");
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("private root should qualify before injection");
                let control = root
                    .admit_control_file(
                        &confined("LOCK"),
                        GenerationFileExpectation::control(8)
                            .expect("small expectation should be accepted"),
                    )
                    .expect("control file should qualify before unlock injection");
                let lock = control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect("exclusive lock should succeed before unlock injection");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::ReleaseLock));
                    }
                    Ok(())
                });
                assert_eq!(
                    lock.unlock()
                        .expect_err("injected unlock boundary must fail")
                        .kind(),
                    GenerationRootErrorKind::Io
                );
            }

            for target in [TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync] {
                let root_path = fixture_root("file-sync-boundary");
                private_file(&root_path, "artifact", contents);
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("private root should qualify before injection");
                let file = root
                    .admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect("file should qualify before sync injection");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::SyncRegularFile));
                    }
                    Ok(())
                });
                assert_eq!(
                    file.sync_durable()
                        .expect_err("injected file-sync boundary must fail")
                        .kind(),
                    GenerationRootErrorKind::Io
                );
            }

            for target in [
                TestBoundary::BeforeDirectorySync,
                TestBoundary::AfterDirectorySync,
            ] {
                let root_path = fixture_root("directory-sync-boundary");
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("private root should qualify before injection");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::SyncDirectory));
                    }
                    Ok(())
                });
                assert_eq!(
                    root.sync_directory_durable()
                        .expect_err("injected directory-sync boundary must fail")
                        .kind(),
                    GenerationRootErrorKind::Io
                );
            }
        }

        #[test]
        fn exact_path_reopen_detects_same_size_inode_substitution() {
            let root_path = fixture_root("inode-substitution");
            let original = private_file(&root_path, "artifact", b"original");
            let replacement_source = private_file(&root_path, "replacement", b"attacker");
            let retained = root_path.join("retained-original");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let observed = Arc::new(Mutex::new(false));
            let hook_observed = Arc::clone(&observed);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeFileRouteReopen {
                    fs::rename(&original, &retained)
                        .expect("original should move to a fresh retained path");
                    fs::rename(&replacement_source, &original)
                        .expect("replacement should move into the original path");
                    *hook_observed.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(b"original"))
                    .expect_err("same-size inode substitution must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            assert!(*observed.lock().expect("hook state should lock"));
        }

        #[test]
        fn root_open_fault_does_not_modify_decoy_tree() {
            let root_path = fixture_root("root-fault-decoy");
            let decoy_path = private_file(&root_path, "decoy", b"unchanged");
            let _guard = install_test_hook(|boundary| {
                if boundary == (TestBoundary::BeforeRootComponentOpen { index: 0 }) {
                    return Err(injected(GenerationRootStage::OpenRootRoute));
                }
                Ok(())
            });
            assert!(QualifiedGenerationRoot::admit(&root_path).is_err());
            assert_eq!(
                fs::read(decoy_path).expect("decoy should remain readable"),
                b"unchanged"
            );
        }

        #[test]
        fn existing_file_descriptor_is_never_replaced_by_ambient_path_changes() {
            let root_path = fixture_root("retained-descriptor");
            let contents = b"descriptor-owned";
            let artifact = private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(contents))
                .expect("file should qualify");
            let moved = root_path.join("artifact-moved");
            fs::rename(&artifact, &moved).expect("artifact should move to a fresh path");
            private_file(&root_path, "artifact", b"ambient-changed!");
            assert_eq!(admitted.as_bytes(), contents);
            assert_eq!(admitted.sha256(), digest(contents));
            assert_eq!(
                admitted
                    .sync_durable()
                    .expect_err("same-size route substitution must fence retained-file sync")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn regular_file_open_rejects_missing_path_without_path_disclosure() {
            let root_path = fixture_root("missing");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let error = root
                .admit_file(
                    &confined("sensitive-missing-name"),
                    GenerationFileExpectation::immutable(0, digest(b""))
                        .expect("zero-length expectation should be accepted"),
                )
                .expect_err("missing file must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert!(!error.to_string().contains("sensitive-missing-name"));
            assert!(!format!("{error:?}").contains("sensitive-missing-name"));
        }

        #[test]
        fn retained_descriptor_sync_detects_post_admission_content_change() {
            let root_path = fixture_root("sync-drift");
            let artifact = private_file(&root_path, "artifact", b"before!!");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
                .expect("fixture should become writable for the retained test fd");
            let mut mutation_file = OpenOptions::new()
                .write(true)
                .open(&artifact)
                .expect("retained mutation fd should open");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
                .expect("fixture should be resealed before admission");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(b"before!!"))
                .expect("file should qualify");
            mutation_file
                .write_all(b"after!!!")
                .expect("same-size mutation should succeed");
            mutation_file
                .sync_all()
                .expect("mutation should be durable");
            assert_eq!(
                admitted
                    .sync_durable()
                    .expect_err("post-admission mutation must fence durability")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn linux_atime_only_drift_is_excluded_without_restoring_metadata() {
            let root_path = fixture_root("atime-drift");
            let artifact = private_file(&root_path, "artifact", b"atime");
            let timestamp_owner = File::open(&artifact).expect("fixture should open for timestamp");
            let modified = timestamp_owner
                .metadata()
                .and_then(|metadata| metadata.modified())
                .expect("fixture mtime should be readable");
            timestamp_owner
                .set_times(
                    std::fs::FileTimes::new()
                        .set_accessed(UNIX_EPOCH)
                        .set_modified(modified),
                )
                .expect("fixture atime should become observably old before admission");

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(b"atime"))
                .expect("file should qualify while O_NOATIME preserves the old atime");
            let before = fs::metadata(&artifact)
                .expect("fixture metadata should load")
                .accessed()
                .expect("fixture atime should be readable");
            let mut ambient_reader = File::open(&artifact).expect("ambient reader should open");
            let mut byte = [0_u8; 1];
            ambient_reader
                .read_exact(&mut byte)
                .expect("ambient read should advance atime");
            let after = fs::metadata(&artifact)
                .expect("fixture metadata should reload")
                .accessed()
                .expect("advanced atime should be readable");
            assert!(
                after > before,
                "fixture must prove an atime-only advancement"
            );
            admitted
                .sync_durable()
                .expect("Linux atime-only drift is not a content mutation");
            assert_eq!(
                fs::metadata(&artifact)
                    .expect("fixture metadata should remain readable")
                    .accessed()
                    .expect("fixture atime should remain readable"),
                after,
                "validation must never restore attacker-observed atime"
            );
        }

        #[test]
        fn open_control_file_requires_write_access_and_private_regular_identity() {
            let root_path = fixture_root("lock-mode");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o400))
                .expect("fixture lock mode should change");
            let error = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect_err("non-private writable control file must fail");
            assert!(
                matches!(
                    error.kind(),
                    GenerationRootErrorKind::WrongMode | GenerationRootErrorKind::Io
                ),
                "write denial must fail before lock ownership"
            );
        }

        #[test]
        fn sealed_file_role_mismatch_precedes_disk_access() {
            let root_path = fixture_root("role-mismatch");
            private_file(&root_path, "artifact", b"01234567");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            assert_eq!(
                root.admit_file(
                    &confined("artifact"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect_err("control role must not enter immutable admission")
                .kind(),
                GenerationRootErrorKind::WrongRole
            );
            assert_eq!(
                root.admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::immutable(8, digest(b"lock-v1!"))
                        .expect("small expectation should be accepted"),
                )
                .expect_err("immutable role must not enter control admission")
                .kind(),
                GenerationRootErrorKind::WrongRole
            );
        }

        #[test]
        fn file_sync_failure_never_falls_back_to_a_weaker_barrier() {
            let root_path = fixture_root("no-sync-fallback");
            private_file(&root_path, "artifact", b"barrier");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(b"barrier"))
                .expect("file should qualify");
            let calls = Arc::new(Mutex::new(Vec::new()));
            let hook_calls = Arc::clone(&calls);
            let _guard = install_test_hook(move |boundary| {
                if matches!(
                    boundary,
                    TestBoundary::BeforeFileSync | TestBoundary::AfterFileSync
                ) {
                    hook_calls
                        .lock()
                        .expect("hook calls should lock")
                        .push(boundary);
                }
                if boundary == TestBoundary::BeforeFileSync {
                    return Err(GenerationRootError::new(
                        GenerationRootErrorKind::DurabilityUnavailable,
                        GenerationRootStage::SyncRegularFile,
                    ));
                }
                Ok(())
            });
            assert_eq!(
                admitted
                    .sync_durable()
                    .expect_err("required barrier failure must propagate")
                    .kind(),
                GenerationRootErrorKind::DurabilityUnavailable
            );
            assert_eq!(
                calls.lock().expect("hook calls should lock").as_slice(),
                &[TestBoundary::BeforeFileSync]
            );
        }

        #[test]
        fn fixture_files_use_single_link_private_modes() {
            let root_path = fixture_root("fixture-self-check");
            let file_path = private_file(&root_path, "artifact", b"x");
            let root_metadata = fs::metadata(&root_path).expect("root metadata should load");
            let file_metadata = fs::metadata(file_path).expect("file metadata should load");
            assert_eq!(root_metadata.permissions().mode() & 0o7777, 0o700);
            assert_eq!(file_metadata.permissions().mode() & 0o7777, 0o400);
        }

        #[test]
        fn root_admission_rejects_non_directory_final_component() {
            let container = fixture_root("root-not-directory");
            let file_path = private_file(&container, "not-a-root", b"x");
            let error = QualifiedGenerationRoot::admit(&file_path)
                .expect_err("regular file cannot be a generation root");
            assert!(
                matches!(
                    error.kind(),
                    GenerationRootErrorKind::NotDirectory | GenerationRootErrorKind::Io
                ),
                "non-directory final route must fail closed"
            );
        }

        #[test]
        fn root_revalidation_detects_security_identity_drift() {
            let root_path = fixture_root("root-security-drift");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            fs::set_permissions(&root_path, fs::Permissions::from_mode(0o750))
                .expect("fixture mode should change");
            assert_eq!(
                root.revalidate_route()
                    .expect_err("root mode drift must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn caller_control_hash_is_checked_only_after_lock_acquisition() {
            let root_path = fixture_root("lock-hash-order");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            let first = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("first exclusive lock should succeed");
            let wrong_hash = GenerationFileExpectation::control_with_sha256(8, [0x11; 32])
                .expect("small expectation should be accepted");
            let wrong_control = root
                .admit_control_file(&confined("LOCK"), wrong_hash)
                .expect("digest-qualified control file should qualify structurally");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            assert_eq!(
                wrong_control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("content must not be inspected through a contended lock")
                    .kind(),
                GenerationRootErrorKind::LockContended
            );
            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(observed.contains(&TestBoundary::BeforeLock));
            assert!(
                !observed
                    .iter()
                    .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
                "a contended lock must return before post-admission content or caller-digest reads"
            );
            drop(observed);
            drop(guard);
            first.unlock().expect("exclusive unlock should succeed");
            assert_eq!(
                wrong_control
                    .try_lock(GenerationRootLockMode::Exclusive)
                    .expect_err("wrong hash must fail after lock acquisition")
                    .kind(),
                GenerationRootErrorKind::HashMismatch
            );
        }

        #[test]
        fn explicit_file_creation_does_not_replace_existing_data() {
            let root_path = fixture_root("create-new");
            let path = private_file(&root_path, "artifact", b"first");
            let result = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path);
            assert!(
                result.is_err(),
                "fixture and production doctrine require create-new semantics"
            );
            assert_eq!(
                fs::read(path).expect("original bytes should remain readable"),
                b"first"
            );
        }

        #[test]
        fn retained_root_directory_sync_revalidates_before_and_after_barrier() {
            let root_path = fixture_root("root-sync-fence");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let moved = root_path.with_extension("moved-before-sync");
            fs::rename(&root_path, &moved).expect("root should move to a fresh path");
            DirBuilder::new()
                .mode(0o700)
                .create(&root_path)
                .expect("replacement root should be creatable");
            fs::set_permissions(&root_path, fs::Permissions::from_mode(0o700))
                .expect("replacement root mode should be settable");
            let error = root
                .sync_directory_durable()
                .expect_err("root route replacement must fail before sync");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::RevalidateRootRoute);
        }

        #[test]
        fn root_component_replacement_during_walk_is_fault_detectable() {
            let root_path = fixture_root("ancestor-walk-race");
            let components = root_path
                .components()
                .filter(|component| !matches!(component, std::path::Component::RootDir))
                .count();
            assert!(components > 0);
            let observed = Arc::new(Mutex::new(false));
            let hook_observed = Arc::clone(&observed);
            let last_index = components.saturating_sub(1);
            let _guard = install_test_hook(move |boundary| {
                if boundary == (TestBoundary::AfterRootComponentOpen { index: last_index }) {
                    *hook_observed.lock().expect("hook state should lock") = true;
                    return Err(injected(GenerationRootStage::OpenRootRoute));
                }
                Ok(())
            });
            assert!(QualifiedGenerationRoot::admit(&root_path).is_err());
            assert!(*observed.lock().expect("hook state should lock"));
        }

        #[test]
        fn read_fault_is_bounded_and_does_not_return_partial_bytes() {
            let root_path = fixture_root("read-fault");
            let contents = b"no partial publication";
            private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let _guard = install_test_hook(|boundary| {
                if boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    return Err(injected(GenerationRootStage::ReadRegularFile));
                }
                Ok(())
            });
            let error = root
                .admit_file(&confined("artifact"), immutable_expectation(contents))
                .expect_err("injected read failure must return no owner");
            assert_eq!(error.kind(), GenerationRootErrorKind::Io);
            assert_eq!(error.stage(), GenerationRootStage::ReadRegularFile);
        }

        #[test]
        fn file_length_probe_rejects_growth_beyond_expected_end() {
            let root_path = fixture_root("end-probe");
            let artifact = private_file(&root_path, "artifact", b"12345678");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o600))
                .expect("fixture should become writable for the retained test fd");
            let mut mutation_file = OpenOptions::new()
                .write(true)
                .open(&artifact)
                .expect("retained mutation fd should open");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o400))
                .expect("fixture should be resealed before admission");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let mutated = Arc::new(Mutex::new(false));
            let hook_mutated = Arc::clone(&mutated);
            let _guard = install_test_hook(move |boundary| {
                if boundary
                    == (TestBoundary::AfterRead {
                        offset: 0,
                        byte_count: 8,
                    })
                    && !*hook_mutated.lock().expect("hook state should lock")
                {
                    mutation_file
                        .seek(SeekFrom::End(0))
                        .expect("fixture should be seekable");
                    mutation_file.write_all(b"9").expect("fixture should grow");
                    mutation_file.sync_all().expect("growth should be durable");
                    *hook_mutated.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(b"12345678"),)
                    .expect_err("one-byte end growth must fail")
                    .kind(),
                GenerationRootErrorKind::SizeMismatch
            );
        }

        #[test]
        fn wrong_owner_witness_is_rejected_without_requiring_privileged_chown() {
            let root_path = fixture_root("owner-contract");
            let root = QualifiedGenerationRoot::admit(&root_path)
                .expect("current-user root should qualify");
            let mut wrong_owner = root.witness();
            wrong_owner.uid = rustix::process::geteuid().as_raw().wrapping_add(1);
            assert_eq!(
                validate_owner_mode_for_test(
                    wrong_owner,
                    super::super::GENERATION_ROOT_DIRECTORY_MODE,
                )
                .expect_err("a non-effective owner witness must fail")
                .kind(),
                GenerationRootErrorKind::WrongOwner
            );
        }

        #[test]
        fn opened_file_descriptor_witness_is_stable_across_arc_clones() {
            let root_path = fixture_root("arc-owner");
            let contents = b"one allocation";
            private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(contents))
                .expect("file should qualify");
            let first = admitted.bytes();
            let second = admitted.bytes();
            assert!(Arc::ptr_eq(&first, &second));
            assert_eq!(first.as_ref(), contents);
        }

        #[test]
        fn file_mode_change_after_admission_fences_sync() {
            let root_path = fixture_root("file-mode-drift");
            let artifact = private_file(&root_path, "artifact", b"mode");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("artifact"), immutable_expectation(b"mode"))
                .expect("file should qualify");
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o640))
                .expect("fixture mode should change");
            assert_eq!(
                admitted
                    .sync_durable()
                    .expect_err("file mode drift must fence sync")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn open_file_missing_write_permission_never_acquires_lock() {
            let root_path = fixture_root("lock-permission");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o400))
                .expect("fixture lock mode should change");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let observed = Arc::new(Mutex::new(false));
            let hook_observed = Arc::clone(&observed);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_observed.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert!(
                root.admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .is_err()
            );
            assert!(
                !*observed.lock().expect("hook state should lock"),
                "mode validation must precede the flock syscall"
            );
        }

        #[test]
        fn root_filesystem_profile_exposes_only_qualified_local_type() {
            let root_path = fixture_root("filesystem-profile");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("ext4 root should qualify");
            assert_eq!(
                root.witness().filesystem(),
                super::super::QualifiedFilesystem::LinuxExt4
            );
            assert_ne!(root.witness().mount_identity(), [0; 32]);
        }

        #[test]
        fn explicit_private_file_is_not_mutated_by_failed_hash_admission() {
            let root_path = fixture_root("hash-decoy");
            let artifact = private_file(&root_path, "artifact", b"unchanged");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let expectation = GenerationFileExpectation::immutable(9, [0xff; 32])
                .expect("small expectation should be accepted");
            assert_eq!(
                root.admit_file(&confined("artifact"), expectation)
                    .expect_err("wrong digest must fail")
                    .kind(),
                GenerationRootErrorKind::HashMismatch
            );
            assert_eq!(
                fs::read(artifact).expect("artifact should remain readable"),
                b"unchanged"
            );
        }

        #[test]
        fn opened_root_requires_absolute_route() {
            let error = QualifiedGenerationRoot::admit(Path::new("relative"))
                .expect_err("relative root must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
            assert_eq!(error.stage(), GenerationRootStage::ParseRootRoute);
        }

        #[test]
        fn root_path_with_repeated_separator_is_rejected_before_open() {
            let error = QualifiedGenerationRoot::admit(Path::new("/data//forbidden"))
                .expect_err("repeated separator must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::InvalidRoute);
            assert_eq!(error.stage(), GenerationRootStage::ParseRootRoute);
        }

        #[test]
        fn lock_file_hardlink_is_rejected_before_flock() {
            let root_path = fixture_root("lock-hardlink");
            let lock = control_file(&root_path, "LOCK", b"lock-v1!");
            fs::hard_link(&lock, root_path.join("LOCK-alias"))
                .expect("fixture hardlink should be creatable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let observed = Arc::new(Mutex::new(false));
            let hook_observed = Arc::clone(&observed);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_observed.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                root.admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect_err("hardlinked lock file must fail")
                .kind(),
                GenerationRootErrorKind::HardLinked
            );
            assert!(!*observed.lock().expect("hook state should lock"));
        }

        #[test]
        fn lock_sync_rechecks_identity_after_barrier() {
            let root_path = fixture_root("lock-sync-fence");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            let lock = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("exclusive lock should succeed");
            let _guard = install_test_hook(|boundary| {
                if boundary == TestBoundary::AfterFileSync {
                    return Err(injected(GenerationRootStage::SyncRegularFile));
                }
                Ok(())
            });
            assert_eq!(
                lock.sync_durable()
                    .expect_err("post-barrier fault must propagate")
                    .kind(),
                GenerationRootErrorKind::Io
            );
            lock.unlock()
                .expect("lock should still be explicitly releasable");
        }

        #[test]
        fn persistent_test_fixtures_never_use_automatic_cleanup() {
            let root_path = fixture_root("persistent");
            private_file(&root_path, "evidence", b"retained");
            assert!(root_path.exists());
            assert_eq!(
                fs::read(root_path.join("evidence")).expect("evidence should persist"),
                b"retained"
            );
        }

        #[test]
        fn root_qualification_rejects_symlink_final_component() {
            let container = fixture_root("root-symlink-container");
            let target = fixture_root("root-symlink-target");
            let link = container.join("root-link");
            symlink(&target, &link).expect("root symlink should be creatable");
            assert_eq!(
                QualifiedGenerationRoot::admit(&link)
                    .expect_err("root symlink must fail")
                    .kind(),
                GenerationRootErrorKind::SymbolicLink
            );
        }

        #[test]
        fn file_reopen_fault_occurs_after_exact_hashing() {
            let root_path = fixture_root("reopen-order");
            let contents = b"hash-before-reopen";
            private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if boundary == TestBoundary::BeforeFileRouteReopen {
                    return Err(injected(GenerationRootStage::RevalidateRegularFile));
                }
                Ok(())
            });
            assert!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .is_err()
            );
            let observed = boundaries.lock().expect("boundary log should lock");
            let exact_read = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::AfterExactRead)
                .expect("exact read boundary should be observed");
            let reopen = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::BeforeFileRouteReopen)
                .expect("reopen boundary should be observed");
            drop(observed);
            assert!(exact_read < reopen);
        }

        #[test]
        fn zero_length_file_is_admitted_without_partial_read_state() {
            let root_path = fixture_root("zero-length");
            private_file(&root_path, "empty", b"");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let admitted = root
                .admit_file(&confined("empty"), immutable_expectation(b""))
                .expect("exact zero-length file should qualify");
            assert!(admitted.as_bytes().is_empty());
            assert_eq!(admitted.sha256(), digest(b""));
        }

        #[test]
        fn lock_guard_debug_never_exposes_descriptor_or_path() {
            let root_path = fixture_root("lock-debug");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect("control file should qualify");
            let lock = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect("exclusive lock should succeed");
            let debug = format!("{lock:?}");
            assert_eq!(debug, "GenerationRootLock { held: true, .. }");
            assert!(!debug.contains("LOCK"));
            assert!(!debug.contains(root_path.to_string_lossy().as_ref()));
            lock.unlock().expect("lock should release");
        }

        #[test]
        fn file_admission_revalidates_root_after_file_reopen() {
            let root_path = fixture_root("final-root-fence");
            let contents = b"root fence";
            private_file(&root_path, "artifact", contents);
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let moved = root_path.with_extension("moved-at-final-fence");
            let swap_path = root_path.clone();
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeFinalRootRevalidation {
                    fs::rename(&swap_path, &moved).expect("root should move to a fresh path");
                    DirBuilder::new()
                        .mode(0o700)
                        .create(&swap_path)
                        .expect("replacement root should be creatable");
                    fs::set_permissions(&swap_path, fs::Permissions::from_mode(0o700))
                        .expect("replacement root mode should be settable");
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect_err("final root replacement must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn regular_file_descriptor_is_opened_cloexec() {
            let root_path = fixture_root("cloexec");
            private_file(&root_path, "artifact", b"cloexec");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let observed = Arc::new(Mutex::new(None));
            let hook_observed = Arc::clone(&observed);
            let _guard = install_test_hook(move |boundary| {
                if let TestBoundary::AfterRegularFileOpen { index: 0, cloexec } = boundary {
                    *hook_observed.lock().expect("hook state should lock") = Some(cloexec);
                }
                Ok(())
            });
            root.admit_file(&confined("artifact"), immutable_expectation(b"cloexec"))
                .expect("file should qualify");
            assert_eq!(
                *observed.lock().expect("hook state should lock"),
                Some(true),
                "the retained descriptor must carry FD_CLOEXEC"
            );
        }

        #[test]
        fn fixture_file_open_is_create_new_not_truncating() {
            let root_path = fixture_root("fixture-create-new");
            let path = private_file(&root_path, "artifact", b"original");
            let error = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .expect_err("create-new should reject an existing fixture");
            assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
            assert_eq!(
                fs::read(path).expect("original should remain readable"),
                b"original"
            );
        }

        #[test]
        fn error_counts_are_bounded_numeric_not_content_bearing() {
            let observed = super::super::GENERATION_ROOT_MAX_CONTROL_FILE_BYTES.saturating_add(1);
            let error = GenerationFileExpectation::control(observed)
                .expect_err("control-file ceiling must fail");
            assert_eq!(
                error.expected(),
                Some(super::super::GENERATION_ROOT_MAX_CONTROL_FILE_BYTES)
            );
            assert_eq!(error.observed(), Some(observed));
            assert!(!error.to_string().contains("artifact"));
        }

        #[test]
        fn file_admission_does_not_reopen_through_symlink_substitution() {
            let root_path = fixture_root("symlink-substitution");
            let contents = b"original";
            let original = private_file(&root_path, "artifact", contents);
            let decoy = private_file(&root_path, "decoy", contents);
            let retained = root_path.join("retained");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeFileRouteReopen {
                    fs::rename(&original, &retained).expect("original should move to a fresh path");
                    symlink(&decoy, &original).expect("attacker symlink should be creatable");
                }
                Ok(())
            });
            assert_eq!(
                root.admit_file(&confined("artifact"), immutable_expectation(contents))
                    .expect_err("symlink substitution must fail")
                    .kind(),
                GenerationRootErrorKind::SymbolicLink
            );
        }

        #[test]
        fn direct_root_and_file_operations_leave_existing_siblings_unchanged() {
            let root_path = fixture_root("siblings");
            let sibling = private_file(&root_path, "sibling", b"sibling bytes");
            private_file(&root_path, "artifact", b"artifact bytes");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            root.admit_file(
                &confined("artifact"),
                immutable_expectation(b"artifact bytes"),
            )
            .expect("artifact should qualify");
            assert_eq!(
                fs::read(sibling).expect("sibling should remain readable"),
                b"sibling bytes"
            );
        }

        #[test]
        fn opened_root_has_single_effective_owner() {
            let root_path = fixture_root("owner");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            assert_eq!(root.witness().uid(), rustix::process::geteuid().as_raw());
            assert_eq!(root.witness().gid(), rustix::process::getegid().as_raw());
        }

        #[test]
        fn exact_control_file_size_is_checked_before_flock() {
            let root_path = fixture_root("lock-size");
            control_file(&root_path, "LOCK", b"short");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let observed = Arc::new(Mutex::new(false));
            let hook_observed = Arc::clone(&observed);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_observed.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            assert_eq!(
                root.admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small expectation should be accepted"),
                )
                .expect_err("wrong lock size must fail")
                .kind(),
                GenerationRootErrorKind::SizeMismatch
            );
            assert!(!*observed.lock().expect("hook state should lock"));
        }

        #[test]
        fn lock_candidate_open_faults_are_reported_as_acquire_lock() {
            let root_path = fixture_root("lock-candidate-stage");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("private root should qualify");
            let control = root
                .admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("control should qualify before the injected drift");
            let reached_flock = Arc::new(Mutex::new(false));
            let hook_reached_flock = Arc::clone(&reached_flock);
            let mut changed = false;
            let _guard = install_test_hook(move |boundary| {
                if !changed && boundary == (TestBoundary::BeforeRegularFileOpen { index: 0 }) {
                    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o644))
                        .expect("lock candidate mode drift should be injectable");
                    changed = true;
                }
                if boundary == TestBoundary::BeforeLock {
                    *hook_reached_flock.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            let error = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("the drifted lock candidate must fail before flock");
            assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            assert!(
                !*reached_flock.lock().expect("hook state should lock"),
                "candidate qualification must precede flock"
            );
        }

        #[test]
        fn root_security_error_identifies_stage_without_path() {
            let root_path = fixture_root("root-stage");
            fs::set_permissions(&root_path, fs::Permissions::from_mode(0o755))
                .expect("fixture mode should change");
            let error =
                QualifiedGenerationRoot::admit(&root_path).expect_err("public root mode must fail");
            assert_eq!(error.stage(), GenerationRootStage::QualifyRootSecurity);
            assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
            assert!(
                !error
                    .to_string()
                    .contains(root_path.to_string_lossy().as_ref())
            );
        }

        #[test]
        fn absolute_ancestor_owner_and_mode_policy_is_fail_closed() {
            let effective_uid = rustix::process::geteuid().as_raw();
            for owner in [0, effective_uid] {
                for mode in [0o700, 0o750, 0o755, 0o500, 0o555] {
                    validate_absolute_ancestor_owner_mode_for_test(owner, mode)
                        .expect("root-owned or euid-owned non-writable ancestors are trusted");
                }
            }
            for mode in [0o720, 0o702, 0o770, 0o707, 0o1777] {
                let error = validate_absolute_ancestor_owner_mode_for_test(effective_uid, mode)
                    .expect_err("sticky or non-sticky group/world writable ancestors must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
            }
            let wrong_owner = if effective_uid == 0 {
                1
            } else {
                effective_uid + 1
            };
            assert_eq!(
                validate_absolute_ancestor_owner_mode_for_test(wrong_owner, 0o755)
                    .expect_err("an unrelated owner must fail")
                    .kind(),
                GenerationRootErrorKind::WrongOwner
            );
        }

        #[test]
        fn absolute_route_qualifies_slash_and_every_non_final_component() {
            let root_path = fixture_root("absolute-ancestor-boundaries");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            QualifiedGenerationRoot::admit(&root_path)
                .expect("the secure retained fixture route should qualify");
            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(
                observed.contains(&TestBoundary::BeforeAbsoluteAncestorAclRead { index: None })
            );
            assert!(observed.contains(&TestBoundary::AfterAbsoluteAncestorAclRead { index: None }));
            let final_index = root_path.components().count().saturating_sub(2);
            assert!(
                observed.contains(&TestBoundary::BeforeRootComponentOpen { index: final_index })
            );
            assert!(
                !observed.contains(&TestBoundary::BeforeAbsoluteAncestorAclRead {
                    index: Some(final_index),
                }),
                "the final root uses the stricter private-root qualifier, not ancestor policy"
            );
            drop(observed);
        }

        #[test]
        fn group_world_and_sticky_writable_absolute_ancestors_are_rejected() {
            for (label, mode) in [("group", 0o770), ("world", 0o707), ("sticky", 0o1777)] {
                let ancestor = fixture_root(&format!("writable-ancestor-{label}"));
                let root_path = private_dir(&ancestor, "root");
                fs::set_permissions(&ancestor, fs::Permissions::from_mode(mode))
                    .expect("hostile ancestor mode should be settable");
                let error = QualifiedGenerationRoot::admit(&root_path)
                    .expect_err("an attacker-writable absolute ancestor must fail");
                assert_eq!(error.kind(), GenerationRootErrorKind::WrongMode);
                assert!(error.component_index().is_some());
            }
        }

        #[test]
        fn access_and_default_acls_on_absolute_ancestors_are_rejected() {
            for (label, xattr_name) in [
                ("access", "system.posix_acl_access"),
                ("default", "system.posix_acl_default"),
            ] {
                let ancestor = fixture_root(&format!("acl-ancestor-{label}"));
                let root_path = private_dir(&ancestor, "root");
                let retained = File::open(&ancestor).expect("ancestor descriptor should open");
                install_linux_acl_xattr(&retained, xattr_name, 0o7, 0o4, 0, 0o4, 0);
                assert_eq!(
                    QualifiedGenerationRoot::admit(&root_path)
                        .expect_err("an ancestor extended ACL must fail")
                        .kind(),
                    GenerationRootErrorKind::AclRejected
                );
            }
        }

        #[test]
        fn ancestor_mutation_inside_the_acl_sandwich_is_rejected() {
            let ancestor = fixture_root("ancestor-acl-sandwich");
            let root_path = private_dir(&ancestor, "root");
            let ancestor_index = ancestor.components().count().saturating_sub(2);
            let mut mutated = false;
            let hook_ancestor = ancestor.clone();
            let _guard = install_test_hook(move |boundary| {
                if !mutated
                    && boundary
                        == (TestBoundary::BeforeAbsoluteAncestorAclRead {
                            index: Some(ancestor_index),
                        })
                {
                    fs::set_permissions(&hook_ancestor, fs::Permissions::from_mode(0o710))
                        .expect("the ancestor security mutation should be injectable");
                    mutated = true;
                }
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&root_path)
                .expect_err("ctime drift inside the ACL sandwich must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                error.component_index(),
                Some(u16::try_from(ancestor_index).expect("fixture depth fits u16"))
            );
        }

        #[test]
        fn ancestor_acl_added_between_full_route_walks_is_rejected() {
            let ancestor = fixture_root("ancestor-acl-between-walks");
            let root_path = private_dir(&ancestor, "root");
            let hook_ancestor = ancestor.clone();
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                if !mutated && boundary == TestBoundary::AfterFirstRootQualification {
                    let retained =
                        File::open(&hook_ancestor).expect("ancestor descriptor should reopen");
                    install_linux_acl_xattr(
                        &retained,
                        "system.posix_acl_default",
                        0o7,
                        0o4,
                        0,
                        0o4,
                        0,
                    );
                    mutated = true;
                }
                Ok(())
            });
            assert_eq!(
                QualifiedGenerationRoot::admit(&root_path)
                    .expect_err("an ACL added before the second route walk must fail")
                    .kind(),
                GenerationRootErrorKind::AclRejected
            );
        }

        #[test]
        fn absolute_ancestor_substitution_is_rejected_but_sibling_churn_is_allowed() {
            let parent = fixture_root("ancestor-substitution-parent");
            let live = private_dir(&parent, "live");
            let live_root = private_dir(&live, "root");
            let decoy = private_dir(&parent, "decoy");
            private_dir(&decoy, "root");
            let retained = parent.join("retained-live");
            let hook_live = live.clone();
            let hook_decoy = decoy.clone();
            let mut swapped = false;
            let swap_guard = install_test_hook(move |boundary| {
                if !swapped && boundary == TestBoundary::AfterFirstRootQualification {
                    fs::rename(&hook_live, &retained)
                        .expect("the original ancestor should move to a fresh retained name");
                    fs::rename(&hook_decoy, &hook_live)
                        .expect("the decoy ancestor should move into the live route");
                    swapped = true;
                }
                Ok(())
            });
            assert_eq!(
                QualifiedGenerationRoot::admit(&live_root)
                    .expect_err("an absolute ancestor substitution must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            drop(swap_guard);

            let churn_parent = fixture_root("ancestor-sibling-churn");
            let churn_root = private_dir(&churn_parent, "root");
            let hook_parent = churn_parent.clone();
            let mut churned = false;
            let _guard = install_test_hook(move |boundary| {
                if !churned && boundary == TestBoundary::AfterFirstRootQualification {
                    control_file(&hook_parent, "unrelated-sibling", b"retained");
                    churned = true;
                }
                Ok(())
            });
            QualifiedGenerationRoot::admit(&churn_root)
                .expect("unrelated sibling churn must not poison persisted route identity");
        }

        #[test]
        fn named_anchor_guard_binds_exact_tuple_and_images() {
            let lock_bytes = b"lock-v1!";
            let (root_path, _authority_path, layout) =
                named_root_fixture("named-anchor-happy", lock_bytes, 0xa5);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the exact named anchor tuple should qualify");
            let witnesses = root.anchor_witnesses();
            assert_eq!(witnesses.root(), root.witness());
            assert_ne!(witnesses.root().inode(), witnesses.lock().inode());
            assert_ne!(witnesses.root().inode(), witnesses.authority().inode());
            assert_ne!(witnesses.lock().inode(), witnesses.authority().inode());
            assert_eq!(
                witnesses.lock().byte_len(),
                u64::try_from(lock_bytes.len()).expect("fixture length fits u64")
            );
            assert_eq!(
                witnesses.authority().byte_len(),
                GENERATION_ROOT_AUTHORITY_FILE_BYTES
            );
            let guard = root
                .read_guard()
                .expect("the shared read guard should acquire");
            assert_eq!(guard.lock_bytes(), lock_bytes);
            assert!(guard.authority_bytes().iter().all(|byte| *byte == 0xa5));
            assert_eq!(guard.lock_sha256(), digest(lock_bytes));
            assert_eq!(guard.authority_sha256(), digest(guard.authority_bytes()));
            let first_lock_arc = guard.lock_bytes_arc();
            let second_lock_arc = guard.lock_bytes_arc();
            assert!(Arc::ptr_eq(&first_lock_arc, &second_lock_arc));
            guard
                .release()
                .expect("the unchanged shared guard should release");
        }

        #[test]
        fn named_authority_legitimate_in_place_rewrite_is_visible_to_the_next_guard() {
            let (root_path, authority_path, layout) =
                named_root_fixture("named-anchor-rewrite", b"lock-v1!", 0x11);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the original named anchors should qualify");
            let first = root.read_guard().expect("the first guard should acquire");
            assert!(first.authority_bytes().iter().all(|byte| *byte == 0x11));
            first.release().expect("the first guard should release");

            let replacement = authority_image(0x22);
            rewrite_control_file(&authority_path, &replacement);
            root.revalidate()
                .expect("same-inode fixed-length authority advancement is legitimate");
            let second = root.read_guard().expect("the next guard should acquire");
            assert_eq!(second.authority_bytes(), replacement);
            second.release().expect("the second guard should release");
        }

        #[test]
        fn shared_guards_coexist_and_exclude_the_exclusive_anchor_guard() {
            let (root_path, _authority_path, layout) =
                named_root_fixture("named-anchor-contention", b"lock-v1!", 0x33);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the named anchors should qualify");
            let first = root.read_guard().expect("the first reader should acquire");
            let second = root.read_guard().expect("a second reader should coexist");
            assert_eq!(
                root.try_exclusive_anchor_guard()
                    .expect_err("an exclusive guard must not bypass active readers")
                    .kind(),
                GenerationRootErrorKind::LockContended
            );
            second.release().expect("the second reader should release");
            first.release().expect("the first reader should release");
            root.try_exclusive_anchor_guard()
                .expect("the exclusive guard should acquire after both readers release")
                .release()
                .expect("the exclusive guard should release");
        }

        #[test]
        fn captured_anchor_arc_is_immutable_and_release_detects_ambient_mutation() {
            let (root_path, authority_path, layout) =
                named_root_fixture("named-anchor-ambient-mutation", b"lock-v1!", 0x44);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the named anchors should qualify");
            let guard = root.read_guard().expect("the shared guard should acquire");
            let captured = guard.authority_bytes_arc();
            let replacement = authority_image(0x55);
            rewrite_control_file(&authority_path, &replacement);
            assert!(captured.iter().all(|byte| *byte == 0x44));
            let error = guard
                .release()
                .expect_err("ambient mutation must invalidate explicit guard release");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::ReleaseLock);
            root.try_exclusive_anchor_guard()
                .expect("failed validation must still unlock the kernel flock")
                .release()
                .expect("the post-failure exclusive guard should release");
        }

        #[test]
        fn drift_between_double_anchor_captures_fails_before_returning_a_guard() {
            let (root_path, authority_path, layout) =
                named_root_fixture("named-anchor-double-capture", b"lock-v1!", 0x66);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the named anchors should qualify");
            let replacement = authority_image(0x77);
            let mut exact_reads = 0_u8;
            let mut injected = false;
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::AfterExactRead {
                    exact_reads = exact_reads.saturating_add(1);
                } else if exact_reads >= 4
                    && !injected
                    && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
                {
                    rewrite_control_file(&authority_path, &replacement);
                    injected = true;
                }
                Ok(())
            });
            let error = root
                .read_guard()
                .expect_err("authority drift between the two captures must fail");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                error.stage(),
                GenerationRootStage::AcquireLock,
                "the completed first image and later image must disagree at the guard boundary"
            );
        }

        #[test]
        fn named_anchor_alias_missing_name_and_route_replacement_fail_closed() {
            let alias_root = fixture_root("named-anchor-hardlink-alias");
            let shared = authority_image(0x88);
            let lock_path = control_file(&alias_root, "LOCK", &shared);
            fs::hard_link(&lock_path, alias_root.join("AUTHORITY"))
                .expect("the hostile hard-link alias should be creatable");
            let layout = GenerationRootAnchorLayout::new(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
                .expect("the alias layout is bounded");
            assert_eq!(
                NamedGenerationRoot::admit(&alias_root, layout)
                    .expect_err("LOCK and AUTHORITY must never alias one inode")
                    .kind(),
                GenerationRootErrorKind::HardLinked
            );

            let missing_root = fixture_root("named-anchor-missing");
            control_file(&missing_root, "lock", b"lock-v1!");
            control_file(&missing_root, "AUTHORITY", &authority_image(0x99));
            assert!(
                NamedGenerationRoot::admit(
                    &missing_root,
                    GenerationRootAnchorLayout::new(8).expect("layout is bounded"),
                )
                .is_err(),
                "the lower-case decoy must not satisfy the exact LOCK route"
            );

            let (root_path, authority_path, layout) =
                named_root_fixture("named-anchor-route-swap", b"lock-v1!", 0xaa);
            let root = NamedGenerationRoot::admit(&root_path, layout)
                .expect("the original named tuple should qualify");
            let retained = root_path.join("AUTHORITY-retained");
            let hook_root = root_path.clone();
            let mut exact_reads = 0_u8;
            let mut injected = false;
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::AfterExactRead {
                    exact_reads = exact_reads.saturating_add(1);
                } else if exact_reads >= 4
                    && !injected
                    && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
                {
                    fs::rename(&authority_path, &retained)
                        .expect("the original authority should move to a retained name");
                    control_file(&hook_root, "AUTHORITY", &authority_image(0xaa));
                    injected = true;
                }
                Ok(())
            });
            assert_eq!(
                root.read_guard()
                    .expect_err("route replacement during tuple capture must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn named_root_and_read_guard_are_send_and_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<NamedGenerationRoot>();
            assert_send_sync::<super::super::GenerationRootReadGuard<'static>>();
        }

        #[test]
        fn private_file_helper_never_opens_read_write_by_default() {
            let root_path = fixture_root("helper-access");
            let path = private_file(&root_path, "artifact", b"readable");
            let mut file = File::open(path).expect("fixture should be readable");
            let position = file
                .stream_position()
                .expect("fixture stream position should be available");
            assert_eq!(position, 0);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod macos_arm64 {
        use super::super::platform::{
            TestBoundary, install_test_hook, set_control_root_creator_process_id_for_test,
            set_root_creator_process_id_for_test, stat_device_as_u64_for_test,
            validate_macos_object_flags_for_test,
        };
        use super::super::{
            ConfinedGenerationPath, GENERATION_ROOT_AUTHORITY_FILE_BYTES,
            GenerationFileExpectation, GenerationRootAnchorLayout, GenerationRootError,
            GenerationRootErrorKind, GenerationRootLockMode, GenerationRootStage,
            PreopenedGenerationRootAnchors, QualifiedFilesystem,
            QualifiedGenerationDirectory as QualifiedGenerationRoot,
            QualifiedGenerationRoot as NamedGenerationRoot,
        };
        use crate::fd_acl::{ExtendedAclPresence, extended_acl_presence};
        use sha2::{Digest, Sha256};
        use std::fs::{self, DirBuilder, File, OpenOptions};
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::os::fd::{AsFd, OwnedFd};
        use std::os::macos::fs::MetadataExt;
        use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt, symlink};
        use std::path::{Path, PathBuf};
        use std::process::Command;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Arc, Mutex, OnceLock};
        use std::time::{Duration, SystemTime, UNIX_EPOCH};

        static FIXTURE_SERIAL: AtomicU64 = AtomicU64::new(0);
        static FIXTURE_SLOT_SERIAL: AtomicU64 = AtomicU64::new(0);
        static FIXTURE_BASE: OnceLock<PathBuf> = OnceLock::new();
        const FIXTURE_SLOT_COUNT: u64 = 512;

        #[test]
        fn apfs_object_flag_profile_accepts_only_zero() {
            validate_macos_object_flags_for_test(0)
                .expect("the narrow APFS object-flag profile accepts zero");
            for flags in [libc::UF_IMMUTABLE, libc::UF_APPEND, libc::UF_NODUMP] {
                let error = validate_macos_object_flags_for_test(flags)
                    .expect_err("every nonzero APFS object flag is outside the profile");
                assert_eq!(
                    error.kind(),
                    GenerationRootErrorKind::UnsupportedObjectFlags
                );
                assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
                assert_eq!(error.expected(), Some(0));
                assert_eq!(error.observed(), Some(u64::from(flags)));
            }
        }

        #[test]
        fn macos_signed_device_ids_are_normalized_by_sign_extension() {
            assert_eq!(stat_device_as_u64_for_test(0), 0);
            assert_eq!(stat_device_as_u64_for_test(1), 1);
            assert_eq!(stat_device_as_u64_for_test(-1), u64::MAX);
            assert_eq!(
                stat_device_as_u64_for_test(i32::MIN),
                u64::from_ne_bytes(i64::from(i32::MIN).to_ne_bytes())
            );
        }

        fn set_empty_acl(path: &Path) {
            let output = Command::new("/bin/chmod")
                .arg("-N")
                .arg(path)
                .output()
                .expect("the physical APFS ACL receipt requires /bin/chmod");
            assert!(
                output.status.success(),
                "chmod -N failed: status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let retained = File::open(path)
                .expect("the cleared APFS fixture must remain openable for ACL proof");
            assert_eq!(
                extended_acl_presence(retained.as_fd())
                    .expect("the descriptor-bound ACL absence probe must succeed"),
                ExtendedAclPresence::Absent,
                "an admissible APFS fixture must have no extended ACL entries"
            );
        }

        fn set_non_empty_acl(path: &Path, allow: bool) {
            let user = std::env::var("USER").expect("USER must identify an ACL principal");
            let disposition = if allow { "allow" } else { "deny" };
            let entry = format!("user:{user} {disposition} write");
            let output = Command::new("/bin/chmod")
                .args(["+a", &entry])
                .arg(path)
                .output()
                .expect("the physical APFS ACL receipt requires /bin/chmod");
            assert!(
                output.status.success(),
                "chmod +a failed: entry={entry:?}, status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let retained = File::open(path)
                .expect("the ACL-bearing APFS fixture must remain openable for ACL proof");
            assert_eq!(
                extended_acl_presence(retained.as_fd())
                    .expect("the descriptor-bound ACL presence probe must succeed"),
                ExtendedAclPresence::Present,
                "the installed {disposition} ACL must be observed through the retained descriptor"
            );
        }

        fn run_chflags(flag: &str, path: &Path) {
            let output = Command::new("/usr/bin/chflags")
                .arg(flag)
                .arg(path)
                .output()
                .expect("the physical APFS object-flag receipt requires chflags");
            assert!(
                output.status.success(),
                "chflags failed: flag={flag}, status={:?}, stdout={}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        fn fixture_root(label: &str) -> PathBuf {
            let slot = FIXTURE_SLOT_SERIAL.fetch_add(1, Ordering::Relaxed);
            assert!(
                slot < FIXTURE_SLOT_COUNT,
                "the retained APFS fixture-slot pool must cover every test root"
            );
            let slot_path = secure_fixture_base().join(format!("slot-{slot:03}"));
            fixture_root_under(&slot_path, label)
        }

        fn secure_fixture_base() -> &'static Path {
            FIXTURE_BASE
                .get_or_init(|| {
                    let private_tmp = std::env::temp_dir()
                        .canonicalize()
                        .expect("the process-private macOS temporary route must canonicalize");
                    assert!(
                        private_tmp.is_absolute(),
                        "the canonical macOS temporary route must be absolute"
                    );
                    let epoch_nanos = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_or(0, |duration| duration.as_nanos());
                    for attempt in 0_u64..64 {
                        let candidate = private_tmp.join(format!(
                            "frankensearch-generation-root-tests-{}-{epoch_nanos}-{attempt}",
                            std::process::id()
                        ));
                        match DirBuilder::new().mode(0o700).create(&candidate) {
                            Ok(()) => {
                                fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                                    .expect("persistent APFS fixture base mode should be settable");
                                set_empty_acl(&candidate);
                                for slot in 0..FIXTURE_SLOT_COUNT {
                                    let slot_path = candidate.join(format!("slot-{slot:03}"));
                                    DirBuilder::new().mode(0o700).create(&slot_path).expect(
                                        "each retained APFS fixture slot should be creatable",
                                    );
                                    fs::set_permissions(
                                        &slot_path,
                                        fs::Permissions::from_mode(0o700),
                                    )
                                    .expect(
                                        "each retained APFS fixture slot mode should be private",
                                    );
                                    set_empty_acl(&slot_path);
                                }
                                return candidate;
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                            Err(error) => {
                                panic!("failed to create retained APFS fixture base: {error}");
                            }
                        }
                    }
                    panic!("failed to allocate a retained process APFS fixture base");
                })
                .as_path()
        }

        fn fixture_root_under(base: &Path, label: &str) -> PathBuf {
            fs::create_dir_all(base).expect("persistent APFS fixture base should be creatable");
            let epoch_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            for attempt in 0_u64..64 {
                let serial = FIXTURE_SERIAL.fetch_add(1, Ordering::Relaxed);
                let candidate = base.join(format!(
                    "{}-{}-{epoch_nanos}-{serial}-{attempt}",
                    std::process::id(),
                    label
                ));
                match DirBuilder::new().mode(0o700).create(&candidate) {
                    Ok(()) => {
                        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o700))
                            .expect("persistent APFS fixture mode should be settable");
                        set_empty_acl(&candidate);
                        return candidate;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => {
                        panic!("failed to create persistent APFS test fixture: {error}");
                    }
                }
            }
            panic!("failed to allocate a unique persistent APFS test fixture");
        }

        fn private_dir(parent: &Path, name: &str) -> PathBuf {
            let path = parent.join(name);
            DirBuilder::new()
                .mode(0o700)
                .create(&path)
                .expect("private APFS directory should be creatable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                .expect("private APFS directory mode should be settable");
            set_empty_acl(&path);
            path
        }

        fn observe_case_sensitive(root: &Path) -> bool {
            private_file(root, "case-probe", b"a");
            let upper = root.join("CASE-PROBE");
            match OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&upper)
            {
                Ok(mut file) => {
                    file.write_all(b"b")
                        .expect("case-sensitivity probe byte should be writable");
                    file.sync_all()
                        .expect("case-sensitivity probe byte should be durable");
                    fs::set_permissions(&upper, fs::Permissions::from_mode(0o400))
                        .expect("case-sensitivity probe should be sealed read-only");
                    set_empty_acl(&upper);
                    true
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => panic!("case-sensitivity probe failed unexpectedly: {error}"),
            }
        }

        fn private_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
            let path = parent.join(name);
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .expect("private APFS file should be creatable");
            file.write_all(bytes)
                .expect("persistent APFS fixture bytes should be writable");
            file.sync_all()
                .expect("persistent APFS fixture bytes should be durable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
                .expect("immutable APFS fixture should be sealed read-only");
            set_empty_acl(&path);
            path
        }

        fn control_file(parent: &Path, name: &str, bytes: &[u8]) -> PathBuf {
            let path = parent.join(name);
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .expect("private APFS control file should be creatable");
            file.write_all(bytes)
                .expect("persistent APFS control bytes should be writable");
            file.sync_all()
                .expect("persistent APFS control bytes should be durable");
            set_empty_acl(&path);
            path
        }

        fn authority_image(fill: u8) -> Vec<u8> {
            vec![
                fill;
                usize::try_from(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
                    .expect("the frozen authority length fits usize")
            ]
        }

        fn named_root_fixture(
            label: &str,
            lock_bytes: &[u8],
            authority_fill: u8,
        ) -> (PathBuf, PathBuf, PathBuf, GenerationRootAnchorLayout) {
            let root = fixture_root(label);
            let lock_path = control_file(&root, "LOCK", lock_bytes);
            let authority_path = control_file(&root, "AUTHORITY", &authority_image(authority_fill));
            let layout = GenerationRootAnchorLayout::new(
                u64::try_from(lock_bytes.len()).expect("the lock fixture length fits u64"),
            )
            .expect("the named lock fixture length is bounded");
            (root, lock_path, authority_path, layout)
        }

        fn admit_named_root(
            root_path: &Path,
            lock_path: &Path,
            authority_path: &Path,
            layout: GenerationRootAnchorLayout,
        ) -> NamedGenerationRoot {
            NamedGenerationRoot::admit_preopened(
                root_path,
                layout,
                PreopenedGenerationRootAnchors::new(
                    preopened_control(lock_path),
                    preopened_control(authority_path),
                ),
            )
            .expect("the exact provider-labelled anchor tuple should qualify")
        }

        fn rewrite_control_file(path: &Path, bytes: &[u8]) {
            let expected_len = u64::try_from(bytes.len()).expect("fixture length fits u64");
            assert_eq!(
                fs::metadata(path)
                    .expect("control metadata should be readable")
                    .len(),
                expected_len,
                "in-place control rewrites must retain the frozen physical length"
            );
            let mut file = OpenOptions::new()
                .write(true)
                .open(path)
                .expect("the mutable APFS control anchor should be writable");
            file.seek(SeekFrom::Start(0))
                .expect("the APFS control anchor should seek to its first byte");
            file.write_all(bytes)
                .expect("the complete in-place APFS control image should be writable");
            file.sync_all()
                .expect("the in-place APFS control rewrite should be durable");
        }

        fn mapped_control_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
            let path = parent.join(name);
            let mut writer = crate::VectorIndex::create(&path, "generation-root-test", 1)
                .expect("mapped APFS control fixture writer should be creatable");
            writer
                .write_record("control", &[1.0])
                .expect("mapped APFS control fixture record should be valid");
            writer
                .finish()
                .expect("mapped APFS control fixture should be durable");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
                .expect("mapped APFS control fixture should use the control mode");
            set_empty_acl(&path);
            let mapping = crate::VectorIndex::open(&path)
                .expect("mapped APFS control fixture should retain a shared writable mapping");
            (path, mapping)
        }

        fn mapped_immutable_file(parent: &Path, name: &str) -> (PathBuf, crate::VectorIndex) {
            let (path, mapping) = mapped_control_file(parent, name);
            fs::set_permissions(&path, fs::Permissions::from_mode(0o400))
                .expect("pre-held APFS mapping should seal as an immutable artifact");
            (path, mapping)
        }

        fn mutate_mapping(mapping: &mut crate::VectorIndex) {
            let crate::VectorIndexData::Mutable(bytes) = &mut mapping.data else {
                panic!("legacy VectorIndex::open must retain a writable mapping");
            };
            let byte = bytes
                .last_mut()
                .expect("mapped APFS control fixture must contain at least one byte");
            *byte ^= 0x01;
        }

        fn preopened_immutable(path: &Path) -> OwnedFd {
            File::open(path)
                .expect("trusted test provider should open the known regular artifact read-only")
                .into()
        }

        fn preopened_control(path: &Path) -> OwnedFd {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .expect("trusted test provider should open the known regular control read-write")
                .into()
        }

        fn preopened_with_status_flag(path: &Path, flag: libc::c_int) -> OwnedFd {
            rustix::fs::open(
                path,
                rustix::fs::OFlags::from_bits_retain(flag.unsigned_abs())
                    | rustix::fs::OFlags::CLOEXEC,
                rustix::fs::Mode::empty(),
            )
            .expect("trusted test provider should open the status-flag fixture")
        }

        fn digest(bytes: &[u8]) -> [u8; 32] {
            Sha256::digest(bytes).into()
        }

        fn immutable_expectation(bytes: &[u8]) -> GenerationFileExpectation {
            GenerationFileExpectation::immutable(
                u64::try_from(bytes.len()).expect("fixture length should fit u64"),
                digest(bytes),
            )
            .expect("small fixture expectation should be accepted")
        }

        fn confined(path: &str) -> ConfinedGenerationPath {
            ConfinedGenerationPath::parse(Path::new(path))
                .expect("static relative APFS fixture path should parse")
        }

        fn injected(stage: GenerationRootStage) -> GenerationRootError {
            GenerationRootError::new(GenerationRootErrorKind::Io, stage)
        }

        #[test]
        fn named_anchor_preopened_guard_binds_exact_tuple_and_advances_authority() {
            let lock_bytes = b"lock-v1!";
            let (root_path, lock_path, authority_path, layout) =
                named_root_fixture("named-anchor-preopened", lock_bytes, 0x31);
            let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
            assert_eq!(
                root.read_guard()
                    .expect_err("Apple Silicon must require a fresh provider-owned LOCK open")
                    .kind(),
                GenerationRootErrorKind::PreopenedDescriptorRequired
            );
            let first = root
                .read_guard_preopened(preopened_control(&lock_path))
                .expect("the first provider-owned shared guard should acquire");
            assert_eq!(first.lock_bytes(), lock_bytes);
            assert!(first.authority_bytes().iter().all(|byte| *byte == 0x31));
            first.release().expect("the unchanged guard should release");

            let replacement = authority_image(0x42);
            rewrite_control_file(&authority_path, &replacement);
            root.revalidate()
                .expect("same-inode fixed-length AUTHORITY advancement should remain admissible");
            let second = root
                .read_guard_preopened(preopened_control(&lock_path))
                .expect("the next provider-owned shared guard should acquire");
            assert_eq!(second.authority_bytes(), replacement);
            second.release().expect("the advanced guard should release");
        }

        #[test]
        fn preopened_shared_guards_coexist_and_exclude_a_fresh_exclusive_open() {
            let (root_path, lock_path, authority_path, layout) =
                named_root_fixture("named-anchor-preopened-contention", b"lock-v1!", 0x53);
            let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
            let first = root
                .read_guard_preopened(preopened_control(&lock_path))
                .expect("the first provider-owned reader should acquire");
            let second = root
                .read_guard_preopened(preopened_control(&lock_path))
                .expect("the second provider-owned reader should coexist");
            assert_eq!(
                root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
                    .expect_err("a fresh exclusive open must contend with active readers")
                    .kind(),
                GenerationRootErrorKind::LockContended
            );
            second.release().expect("the second reader should release");
            first.release().expect("the first reader should release");
            root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
                .expect("the exclusive guard should acquire after both readers release")
                .release()
                .expect("the exclusive guard should release");
        }

        #[test]
        fn preopened_named_anchor_hardlink_alias_and_ancestor_acls_fail_closed() {
            let alias_root = fixture_root("named-anchor-preopened-alias");
            let shared = authority_image(0x64);
            let lock_path = control_file(&alias_root, "LOCK", &shared);
            let authority_path = alias_root.join("AUTHORITY");
            fs::hard_link(&lock_path, &authority_path)
                .expect("the hostile APFS hard-link alias should be creatable");
            set_empty_acl(&authority_path);
            let error = NamedGenerationRoot::admit_preopened(
                &alias_root,
                GenerationRootAnchorLayout::new(GENERATION_ROOT_AUTHORITY_FILE_BYTES)
                    .expect("the alias layout is bounded"),
                PreopenedGenerationRootAnchors::new(
                    preopened_control(&lock_path),
                    preopened_control(&authority_path),
                ),
            )
            .expect_err("LOCK and AUTHORITY must never alias one inode");
            assert_eq!(error.kind(), GenerationRootErrorKind::HardLinked);

            for (label, allow) in [("allow", true), ("deny", false)] {
                let ancestor = fixture_root(&format!("absolute-ancestor-acl-{label}"));
                let root_path = private_dir(&ancestor, "root");
                set_non_empty_acl(&ancestor, allow);
                assert_eq!(
                    QualifiedGenerationRoot::admit(&root_path)
                        .expect_err("every non-empty APFS ancestor ACL must fail")
                        .kind(),
                    GenerationRootErrorKind::AclRejected
                );
            }
        }

        #[test]
        fn preopened_guard_release_detects_live_authority_mutation_and_unlocks() {
            let (root_path, lock_path, authority_path, layout) =
                named_root_fixture("named-anchor-preopened-live-mutation", b"lock-v1!", 0x71);
            let root = admit_named_root(&root_path, &lock_path, &authority_path, layout);
            let guard = root
                .read_guard_preopened(preopened_control(&lock_path))
                .expect("the provider-owned shared guard should acquire");
            let captured = guard.authority_bytes_arc();
            rewrite_control_file(&authority_path, &authority_image(0x72));
            assert!(
                captured.iter().all(|byte| *byte == 0x71),
                "the live guard must retain its immutable APFS authority image"
            );
            let error = guard
                .release()
                .expect_err("ambient APFS authority mutation must invalidate release");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::ReleaseLock);
            root.try_exclusive_anchor_guard_preopened(preopened_control(&lock_path))
                .expect("failed validation must still release the provider-owned flock")
                .release()
                .expect("the post-failure exclusive guard should release");
        }

        #[test]
        fn preopened_double_capture_rejects_content_drift_and_route_replacement() {
            let (drift_root, drift_lock, drift_authority, drift_layout) =
                named_root_fixture("named-anchor-preopened-double-drift", b"lock-v1!", 0x81);
            let root = admit_named_root(&drift_root, &drift_lock, &drift_authority, drift_layout);
            let replacement = authority_image(0x82);
            let mut exact_reads = 0_u8;
            let mut injected = false;
            let drift_hook = install_test_hook(move |boundary| {
                if boundary == TestBoundary::AfterExactRead {
                    exact_reads = exact_reads.saturating_add(1);
                } else if exact_reads >= 4
                    && !injected
                    && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
                {
                    rewrite_control_file(&drift_authority, &replacement);
                    injected = true;
                }
                Ok(())
            });
            let drift_error = root
                .read_guard_preopened(preopened_control(&drift_lock))
                .expect_err("APFS authority drift between captures must fail");
            assert_eq!(drift_error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                drift_error.stage(),
                GenerationRootStage::AcquireLock,
                "the completed first image and later image must disagree at the guard boundary"
            );
            drop(drift_hook);

            let (route_root, route_lock, route_authority, route_layout) =
                named_root_fixture("named-anchor-preopened-route-swap", b"lock-v1!", 0x91);
            let route = admit_named_root(&route_root, &route_lock, &route_authority, route_layout);
            let retained = route_root.join("AUTHORITY-retained");
            let hook_root = route_root.clone();
            let mut route_reads = 0_u8;
            let mut injected = false;
            let route_hook = install_test_hook(move |boundary| {
                if boundary == TestBoundary::AfterExactRead {
                    route_reads = route_reads.saturating_add(1);
                } else if route_reads >= 4
                    && !injected
                    && matches!(boundary, TestBoundary::BeforeRootComponentOpen { .. })
                {
                    fs::rename(&route_authority, &retained)
                        .expect("the original APFS authority should move aside");
                    control_file(&hook_root, "AUTHORITY", &authority_image(0x92));
                    injected = true;
                }
                Ok(())
            });
            assert_eq!(
                route
                    .read_guard_preopened(preopened_control(&route_lock))
                    .expect_err("APFS authority route replacement between captures must fail")
                    .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            drop(route_hook);
        }

        #[test]
        fn absolute_ancestor_acl_mutation_inside_darwin_ctime_sandwich_is_rejected() {
            let ancestor = fixture_root("absolute-ancestor-acl-sandwich");
            let root_path = private_dir(&ancestor, "root");
            let ancestor_index = ancestor.components().count().saturating_sub(2);
            let hook_ancestor = ancestor.clone();
            let mut mutated = false;
            let hook = install_test_hook(move |boundary| {
                if !mutated
                    && boundary
                        == (TestBoundary::AfterAbsoluteAncestorAclRead {
                            index: Some(ancestor_index),
                        })
                {
                    set_non_empty_acl(&hook_ancestor, true);
                    mutated = true;
                }
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&root_path)
                .expect_err("an ACL inserted after the Darwin probe must trip the ctime fence");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                error.component_index(),
                Some(u16::try_from(ancestor_index).expect("fixture depth fits u16"))
            );
            drop(hook);
        }

        #[test]
        fn physical_apfs_empty_acl_exact_names_locks_and_fullsync_succeed() {
            let root_path = fixture_root("physical-success");
            let generation = private_dir(&root_path, "Generation");
            let contents = b"physical APFS generation";
            let artifact_path = private_file(&generation, "Vector.fsvi", contents);
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            assert_eq!(root.witness().filesystem(), QualifiedFilesystem::AppleApfs);
            let admitted = root
                .admit_preopened_file(
                    &confined("Generation/Vector.fsvi"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(contents),
                )
                .expect("empty-ACL exact-case immutable APFS file should qualify");
            assert_eq!(admitted.as_bytes(), contents);
            admitted
                .sync_durable()
                .expect("retained APFS regular file must support F_FULLFSYNC");

            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("empty-ACL APFS control file should qualify");
            assert_eq!(control.bytes().as_ref(), b"lock-v1!");
            assert_eq!(control.sha256(), digest(b"lock-v1!"));
            let lock = control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("APFS control-file flock should succeed");
            assert_eq!(lock.witness(), control.witness());
            assert_eq!(lock.bytes().as_ref(), b"lock-v1!");
            assert_eq!(lock.sha256(), control.sha256());
            assert_eq!(lock.mode(), GenerationRootLockMode::Exclusive);
            lock.sync_durable()
                .expect("locked APFS control file must support F_FULLFSYNC");
            lock.unlock()
                .expect("APFS control-file unlock should succeed");
            root.sync_directory_durable()
                .expect("retained APFS directory must support F_FULLFSYNC");

            assert_eq!(
                root.admit_preopened_file(
                    &confined("generation/Vector.fsvi"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(contents),
                )
                .expect_err("case-folded ancestor spelling must fail even on insensitive APFS")
                .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
            assert_eq!(
                root.admit_preopened_file(
                    &confined("Generation/vector.fsvi"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(contents),
                )
                .expect_err("case-folded final spelling must fail even on insensitive APFS")
                .kind(),
                GenerationRootErrorKind::ObjectChanged
            );

            let decomposed_name = "e\u{301}.fsvi";
            let decomposed_path = private_file(&generation, decomposed_name, b"unicode");
            assert_eq!(
                root.admit_preopened_file(
                    &confined("Generation/\u{e9}.fsvi"),
                    preopened_immutable(&decomposed_path),
                    immutable_expectation(b"unicode"),
                )
                .expect_err("canonically equivalent but byte-distinct spelling must fail")
                .kind(),
                GenerationRootErrorKind::ObjectChanged
            );
        }

        #[test]
        fn macos_independent_lock_contention_and_creator_release_survive_aliases_and_failures() {
            let root_path = fixture_root("lock-lifecycle");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("empty-ACL APFS control should qualify");

            let shared_one = control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Shared,
                )
                .expect("first independent shared lock should succeed");
            let shared_two = control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Shared,
                )
                .expect("second independent shared lock should succeed");
            let contention_boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&contention_boundaries);
            let contention_hook = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let contention_error = control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect_err("an independent exclusive lock must contend with shared locks");
            assert_eq!(
                contention_error.kind(),
                GenerationRootErrorKind::LockContended
            );
            let observed = contention_boundaries
                .lock()
                .expect("boundary log should lock");
            assert!(observed.contains(&TestBoundary::BeforeLock));
            assert!(
                !observed
                    .iter()
                    .any(|boundary| matches!(boundary, TestBoundary::BeforeRead { .. })),
                "a contended APFS lock must return before descriptor content is inspected"
            );
            drop(observed);
            drop(contention_hook);
            drop(shared_one);
            shared_two
                .unlock()
                .expect("the second independent shared lock should release");

            let drop_candidate = preopened_control(&lock_path);
            let surviving_drop_alias = rustix::io::fcntl_dupfd_cloexec(&drop_candidate, 0)
                .expect("the regression fixture should retain a known duplicate");
            let dropped = control
                .try_lock_preopened(drop_candidate, GenerationRootLockMode::Exclusive)
                .expect("the aliased candidate should acquire under the trusted-provider contract");
            drop(dropped);
            control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("creator Drop must explicitly unlock despite a surviving alias")
                .unlock()
                .expect("fresh contender after Drop should release");
            drop(surviving_drop_alias);

            let explicit_candidate = preopened_control(&lock_path);
            let surviving_explicit_alias = rustix::io::fcntl_dupfd_cloexec(&explicit_candidate, 0)
                .expect("the explicit-unlock fixture should retain a known duplicate");
            control
                .try_lock_preopened(explicit_candidate, GenerationRootLockMode::Exclusive)
                .expect("the explicit-unlock fixture should acquire")
                .unlock()
                .expect("explicit unlock should disarm creator Drop deterministically");
            control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("explicit LOCK_UN must release despite a surviving alias")
                .unlock()
                .expect("fresh contender after explicit unlock should release");
            drop(surviving_explicit_alias);

            let failed_candidate = preopened_control(&lock_path);
            let surviving_failure_alias = rustix::io::fcntl_dupfd_cloexec(&failed_candidate, 0)
                .expect("the post-flock failure fixture should retain a known duplicate");
            let hook = install_test_hook(|boundary| {
                if boundary == TestBoundary::AfterLock {
                    return Err(injected(GenerationRootStage::AcquireLock));
                }
                Ok(())
            });
            assert_eq!(
                control
                    .try_lock_preopened(failed_candidate, GenerationRootLockMode::Exclusive)
                    .expect_err("injected post-flock failure must fail")
                    .kind(),
                GenerationRootErrorKind::Io
            );
            drop(hook);
            control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("provisional RAII must unlock after post-flock failure despite an alias")
                .unlock()
                .expect("fresh contender after injected failure should release");
            drop(surviving_failure_alias);
        }

        #[test]
        fn macos_path_final_admission_requires_preopened_capabilities_before_any_io_boundary() {
            let root_path = fixture_root("path-requires-preopened");
            let artifact_path = private_file(&root_path, "artifact", b"sealed");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("trusted preopened control should qualify");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            for error in [
                root.admit_file(&confined("artifact"), immutable_expectation(b"sealed"))
                    .expect_err("path artifact admission must require external provenance"),
                root.admit_control_file(
                    &confined("LOCK"),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect_err("path control admission must require external provenance"),
            ] {
                assert_eq!(
                    error.kind(),
                    GenerationRootErrorKind::PreopenedDescriptorRequired
                );
                assert_eq!(error.stage(), GenerationRootStage::OpenRegularFile);
            }
            let lock_error = control
                .try_lock(GenerationRootLockMode::Exclusive)
                .expect_err("path lock acquisition must require a fresh provider descriptor");
            assert_eq!(
                lock_error.kind(),
                GenerationRootErrorKind::PreopenedDescriptorRequired
            );
            assert_eq!(lock_error.stage(), GenerationRootStage::AcquireLock);
            assert!(
                boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .is_empty(),
                "typed path rejection must precede revalidation, open, read, flock, ACL, and sync"
            );
            assert!(
                artifact_path.exists(),
                "the known regular fixture remains intact"
            );
        }

        #[test]
        fn macos_forked_capabilities_reject_before_provider_descriptor_inspection() {
            let root_path = fixture_root("forked-preopened");
            let artifact_path = private_file(&root_path, "artifact", b"sealed");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let mut root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            set_root_creator_process_id_for_test(
                &mut root.inner,
                std::process::id().wrapping_add(1),
            );
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let artifact_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    File::open(&root_path)
                        .expect("the precedence fixture directory should open")
                        .into(),
                    immutable_expectation(b"sealed"),
                )
                .expect_err(
                    "ForkedProcess must outrank the supplied directory's NotRegularFile error",
                );
            assert_eq!(
                artifact_error.kind(),
                GenerationRootErrorKind::ForkedProcess
            );
            assert_eq!(artifact_error.stage(), GenerationRootStage::OpenRegularFile);
            let control_error = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_immutable(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect_err(
                    "ForkedProcess must outrank the read-only fd's InvalidDescriptorAccess error",
                );
            assert_eq!(control_error.kind(), GenerationRootErrorKind::ForkedProcess);
            assert_eq!(control_error.stage(), GenerationRootStage::OpenRegularFile);
            assert!(
                boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .is_empty(),
                "creator-process rejection must precede provider-fd inspection and every I/O \
                 boundary"
            );
            drop(guard);

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("fresh APFS root should qualify");
            let mut control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("fresh control should qualify");
            drop(root);
            set_control_root_creator_process_id_for_test(
                &mut control.inner,
                std::process::id().wrapping_add(1),
            );
            let lock_boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_lock_boundaries = Arc::clone(&lock_boundaries);
            let _lock_guard = install_test_hook(move |boundary| {
                hook_lock_boundaries
                    .lock()
                    .expect("lock boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let lock_error = control
                .try_lock_preopened(
                    preopened_immutable(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect_err(
                    "ForkedProcess must outrank the lock fd's InvalidDescriptorAccess error",
                );
            assert_eq!(lock_error.kind(), GenerationRootErrorKind::ForkedProcess);
            assert_eq!(lock_error.stage(), GenerationRootStage::AcquireLock);
            assert!(
                lock_boundaries
                    .lock()
                    .expect("lock boundary log should lock")
                    .is_empty(),
                "creator-process rejection must precede lock-candidate inspection, ACL, content, \
                 and flock"
            );
            assert!(
                artifact_path.exists(),
                "the known regular fixture must remain intact"
            );
        }

        #[test]
        fn macos_preopened_type_access_and_cloexec_gates_precede_content_and_lock_operations() {
            let root_path = fixture_root("preopened-gates");
            let artifact_path = private_file(&root_path, "artifact", b"sealed");
            let execute_path = private_file(&root_path, "execute-artifact", b"sealed");
            fs::set_permissions(&execute_path, fs::Permissions::from_mode(0o500))
                .expect("O_EXEC fixture must be owner-executable");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let fifo_path = root_path.join("fifo");
            let fifo_status = Command::new("/usr/bin/mkfifo")
                .arg("-m")
                .arg("600")
                .arg(&fifo_path)
                .status()
                .expect("macOS mkfifo should start for the special-file fixture");
            assert!(
                fifo_status.success(),
                "macOS mkfifo should create the special-file fixture"
            );
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });

            let fifo: OwnedFd = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&fifo_path)
                .expect("trusted provider should supply the FIFO descriptor for rejection")
                .into();
            let fifo_error = root
                .admit_preopened_control_file(
                    &confined("fifo"),
                    fifo,
                    GenerationFileExpectation::control(0)
                        .expect("zero-length control expectation should be accepted"),
                )
                .expect_err("a supplied FIFO must fail at the first fstat type gate");
            assert_eq!(fifo_error.kind(), GenerationRootErrorKind::NotRegularFile);

            let device: OwnedFd = File::open("/dev/null")
                .expect("trusted provider should supply a real device descriptor for rejection")
                .into();
            let device_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    device,
                    immutable_expectation(b"sealed"),
                )
                .expect_err("a supplied device must fail at the first fstat type gate");
            assert_eq!(device_error.kind(), GenerationRootErrorKind::NotRegularFile);

            fs::set_permissions(&artifact_path, fs::Permissions::from_mode(0o600))
                .expect("fixture should become writable before the provider opens the wrong mode");
            let wrong_artifact_access: OwnedFd = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&artifact_path)
                .expect("trusted provider should supply a deliberately wrong access mode")
                .into();
            fs::set_permissions(&artifact_path, fs::Permissions::from_mode(0o400))
                .expect("fixture should be resealed before library admission");
            let access_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    wrong_artifact_access,
                    immutable_expectation(b"sealed"),
                )
                .expect_err("immutable admission requires an exactly read-only descriptor");
            assert_eq!(
                access_error.kind(),
                GenerationRootErrorKind::InvalidDescriptorAccess
            );

            let wrong_control_access = preopened_immutable(&lock_path);
            let control_access_error = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    wrong_control_access,
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect_err("control admission requires an exactly read-write descriptor");
            assert_eq!(
                control_access_error.kind(),
                GenerationRootErrorKind::InvalidDescriptorAccess
            );

            let missing_cloexec = preopened_immutable(&artifact_path);
            rustix::io::fcntl_setfd(&missing_cloexec, rustix::io::FdFlags::empty())
                .expect("test provider should clear close-on-exec");
            let cloexec_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    missing_cloexec,
                    immutable_expectation(b"sealed"),
                )
                .expect_err("missing FD_CLOEXEC must fail before route or content work");
            assert_eq!(
                cloexec_error.kind(),
                GenerationRootErrorKind::CloseOnExecRequired
            );

            let event_only_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_with_status_flag(&artifact_path, libc::O_EVTONLY),
                    immutable_expectation(b"sealed"),
                )
                .expect_err("O_EVTONLY is not a data-readable descriptor");
            assert_eq!(
                event_only_error.kind(),
                GenerationRootErrorKind::InvalidDescriptorAccess
            );

            let execute_only_error = root
                .admit_preopened_file(
                    &confined("execute-artifact"),
                    preopened_with_status_flag(&execute_path, libc::O_EXEC),
                    immutable_expectation(b"sealed"),
                )
                .expect_err("O_EXEC is not a data-readable descriptor");
            assert_eq!(
                execute_only_error.kind(),
                GenerationRootErrorKind::InvalidDescriptorAccess
            );

            let observed = boundaries.lock().expect("boundary log should lock");
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::BeforeRead { .. }
                        | TestBoundary::BeforeLock
                        | TestBoundary::BeforeFileSync
                        | TestBoundary::BeforeAclRead
                        | TestBoundary::BeforeFinalRouteStat
                )),
                "type/access/status/CLOEXEC rejection must precede ACL, route, content, flock, and sync"
            );
            drop(observed);
            drop(guard);
        }

        #[test]
        fn macos_preopened_lock_candidate_rejects_persistent_fwaslocked_history_before_io() {
            const DARWIN_FWASLOCKED: u32 = 0x0000_4000;

            let root_path = fixture_root("preopened-fwaslocked-history");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("fresh control descriptor should qualify");
            let candidate = preopened_control(&lock_path);
            rustix::fs::flock(
                &candidate,
                rustix::fs::FlockOperation::NonBlockingLockExclusive,
            )
            .expect("the fresh candidate should take an uncontended kernel flock");
            rustix::fs::flock(&candidate, rustix::fs::FlockOperation::Unlock)
                .expect("the fixture should explicitly unlock before admission");
            let observed_status =
                rustix::fs::fcntl_getfl(&candidate).expect("candidate status flags should load");
            assert!(
                observed_status.contains(rustix::fs::OFlags::from_bits_retain(DARWIN_FWASLOCKED)),
                "Darwin must expose the kernel-private FWASLOCKED history bit after successful \
                 flock plus unlock"
            );

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let error = control
                .try_lock_preopened(candidate, GenerationRootLockMode::Exclusive)
                .expect_err("a candidate with prior flock history is not fresh");
            assert_eq!(
                error.kind(),
                GenerationRootErrorKind::InvalidDescriptorAccess
            );
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            assert!(
                boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .is_empty(),
                "exact status validation must reject before preopened qualification, ACL, route, \
                 content, or flock boundaries"
            );
        }

        #[test]
        fn macos_preopened_route_binding_rejects_siblings_and_post_admission_substitution() {
            let root_path = fixture_root("preopened-route-binding");
            let artifact_path = private_file(&root_path, "artifact", b"artifact");
            let sibling_path = private_file(&root_path, "sibling", b"sibling!");
            let lock_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");

            let sibling_error = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_immutable(&sibling_path),
                    immutable_expectation(b"sibling!"),
                )
                .expect_err("a valid sibling descriptor must not satisfy another route");
            assert_eq!(sibling_error.kind(), GenerationRootErrorKind::ObjectChanged);

            let admitted = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(b"artifact"),
                )
                .expect("the exact descriptor and route should qualify");
            fs::rename(&artifact_path, root_path.join("artifact-original"))
                .expect("the admitted artifact should move to a retained route");
            private_file(&root_path, "artifact", b"artifact");
            let file_error = admitted
                .sync_durable()
                .expect_err("route substitution must fence retained-descriptor sync");
            assert_eq!(file_error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(
                file_error.stage(),
                GenerationRootStage::RevalidateRegularFile
            );

            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("the exact control descriptor and route should qualify");
            let fresh_lock_descriptor = preopened_control(&lock_path);
            fs::rename(&lock_path, root_path.join("LOCK-original"))
                .expect("the admitted control should move to a retained route");
            control_file(&root_path, "LOCK", b"lock-v1!");
            let lock_reached = Arc::new(Mutex::new(false));
            let hook_lock_reached = Arc::clone(&lock_reached);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeLock {
                    *hook_lock_reached.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            let error = control
                .try_lock_preopened(fresh_lock_descriptor, GenerationRootLockMode::Exclusive)
                .expect_err("route substitution must fence the fresh lock descriptor");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::RevalidateRegularFile);
            assert!(
                !*lock_reached.lock().expect("hook state should lock"),
                "route revalidation must reject before flock"
            );
        }

        #[test]
        fn macos_absolute_root_route_rejects_ancestor_and_final_symlinks() {
            let container = fixture_root("absolute-symlink-container");
            let target_parent = fixture_root("absolute-symlink-target-parent");
            let target_root = private_dir(&target_parent, "qualified-root");
            let ancestor_link = container.join("ancestor-link");
            let final_link = container.join("final-link");
            symlink(&target_parent, &ancestor_link)
                .expect("absolute-route ancestor symlink should be creatable");
            symlink(&target_root, &final_link)
                .expect("absolute-route final symlink should be creatable");

            assert_eq!(
                QualifiedGenerationRoot::admit(&ancestor_link.join("qualified-root"))
                    .expect_err("an absolute-route ancestor symlink must fail")
                    .kind(),
                GenerationRootErrorKind::SymbolicLink
            );
            assert_eq!(
                QualifiedGenerationRoot::admit(&final_link)
                    .expect_err("an absolute-route final symlink must fail")
                    .kind(),
                GenerationRootErrorKind::SymbolicLink
            );
        }

        #[test]
        fn macos_preopened_rejects_ancestor_and_final_symlinks_without_touching_decoys() {
            let root_path = fixture_root("preopened-symlink-routes");
            let decoy_directory = private_dir(&root_path, "decoy-directory");
            let ancestor_target =
                private_file(&decoy_directory, "artifact", b"ancestor decoy stays exact");
            let final_target = private_file(&root_path, "final-target", b"final decoy stays exact");
            symlink(&decoy_directory, root_path.join("ancestor-link"))
                .expect("ancestor symlink should be creatable");
            symlink(&final_target, root_path.join("final-link"))
                .expect("final symlink should be creatable");

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let _hook = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("symlink boundary log should lock")
                    .push(boundary);
                Ok(())
            });
            let ancestor_error = root
                .admit_preopened_file(
                    &confined("ancestor-link/artifact"),
                    preopened_immutable(&ancestor_target),
                    immutable_expectation(b"ancestor decoy stays exact"),
                )
                .expect_err("an ancestor symlink must not bind a trusted descriptor");
            assert_eq!(ancestor_error.kind(), GenerationRootErrorKind::SymbolicLink);
            assert_eq!(ancestor_error.component_index(), Some(0));

            let final_error = root
                .admit_preopened_file(
                    &confined("final-link"),
                    preopened_immutable(&final_target),
                    immutable_expectation(b"final decoy stays exact"),
                )
                .expect_err("a final symlink must not bind a trusted descriptor");
            assert_eq!(final_error.kind(), GenerationRootErrorKind::SymbolicLink);
            assert_eq!(final_error.component_index(), Some(0));

            let observed = boundaries.lock().expect("symlink boundary log should lock");
            assert!(
                !observed.iter().any(|boundary| matches!(
                    boundary,
                    TestBoundary::BeforeRead { .. }
                        | TestBoundary::BeforeTrailingByteProbe
                        | TestBoundary::AfterExactRead
                )),
                "route-bound symlink rejection must precede every content-read boundary"
            );
            drop(observed);

            assert_eq!(
                fs::read(&ancestor_target).expect("ancestor decoy should remain readable"),
                b"ancestor decoy stays exact"
            );
            assert_eq!(
                fs::read(&final_target).expect("final decoy should remain readable"),
                b"final decoy stays exact"
            );
        }

        #[test]
        fn macos_immutable_content_drift_fences_sync_before_durability() {
            let root_path = fixture_root("mapped-immutable-before-sync");
            let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
            let admitted_bytes =
                fs::read(&artifact_path).expect("mapped APFS immutable fixture must be readable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let admitted = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(&admitted_bytes),
                )
                .expect("mapped APFS immutable fixture should qualify before ambient mutation");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared APFS immutable mapping must remain retained"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = admitted
                .sync_durable()
                .expect_err("pre-durability APFS immutable content drift must fail closed");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            assert!(
                !boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .contains(&TestBoundary::BeforeFileSync),
                "APFS immutable validation must fail before F_FULLFSYNC"
            );
        }

        #[test]
        fn macos_immutable_content_drift_during_durability_is_detected_after_barrier() {
            let root_path = fixture_root("mapped-immutable-during-sync");
            let (artifact_path, mapping) = mapped_immutable_file(&root_path, "artifact");
            let admitted_bytes =
                fs::read(&artifact_path).expect("mapped APFS immutable fixture must be readable");
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let admitted = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(&admitted_bytes),
                )
                .expect("mapped APFS immutable fixture should qualify before ambient mutation");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == TestBoundary::BeforeFileSync {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared APFS immutable mapping must remain retained"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = admitted
                .sync_durable()
                .expect_err("APFS immutable mutation at durability must fail after the barrier");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            let observed = boundaries.lock().expect("boundary log should lock");
            let after_sync = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::AfterFileSync)
                .expect("F_FULLFSYNC should complete before post-check failure");
            assert!(
                observed.iter().enumerate().any(|(index, boundary)| {
                    index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
                }),
                "a second exact APFS immutable read must follow F_FULLFSYNC"
            );
            drop(observed);
        }

        #[test]
        fn macos_control_without_caller_digest_rejects_preheld_mapping_mutation_after_flock() {
            let root_path = fixture_root("mapped-control-after-flock");
            let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped APFS control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let expectation = GenerationFileExpectation::control(byte_len)
                .expect("mapped APFS control fixture length should be accepted");
            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    expectation,
                )
                .expect("mapped APFS control fixture should qualify before ambient mutation");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut locked = false;
            let mut mutated = false;
            let guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if boundary == TestBoundary::AfterLock {
                    locked = true;
                }
                if locked && !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared APFS mapping should remain retained by the hook"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = control
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect_err("post-flock APFS mapped mutation must invalidate the bound image");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::AcquireLock);
            let observed = boundaries.lock().expect("boundary log should lock");
            let before_lock = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::BeforeLock)
                .expect("flock attempt boundary should be observed");
            let after_lock = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::AfterLock)
                .expect("successful flock boundary should be observed");
            let first_locked_read = observed
                .iter()
                .enumerate()
                .find_map(|(index, boundary)| {
                    (index > after_lock && *boundary == (TestBoundary::BeforeRead { offset: 0 }))
                        .then_some(index)
                })
                .expect("post-flock content validation should read the descriptor");
            assert!(before_lock < after_lock && after_lock < first_locked_read);
            drop(observed);
            drop(guard);

            let fresh = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    expectation,
                )
                .expect("the mutated APFS image should support a fresh bound admission");
            fresh
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("failed validation must not leak the APFS kernel flock")
                .unlock()
                .expect("fresh APFS lock should release cleanly");
        }

        #[test]
        fn macos_control_content_drift_fences_sync_before_durability() {
            let root_path = fixture_root("mapped-control-before-sync");
            let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped APFS control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let lock = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(byte_len)
                        .expect("mapped APFS control fixture length should be accepted"),
                )
                .expect("mapped APFS control fixture should qualify")
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("mapped APFS control fixture should lock");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == (TestBoundary::BeforeRead { offset: 0 }) {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared APFS mapping should remain retained by the hook"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = lock
                .sync_durable()
                .expect_err("pre-durability APFS content drift must fail closed");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            assert!(
                !boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .contains(&TestBoundary::BeforeFileSync),
                "APFS content validation must fail before the durability syscall"
            );
        }

        #[test]
        fn macos_control_content_drift_during_durability_is_detected_after_barrier() {
            let root_path = fixture_root("mapped-control-during-sync");
            let (lock_path, mapping) = mapped_control_file(&root_path, "LOCK");
            let byte_len = fs::metadata(&lock_path)
                .expect("mapped APFS control fixture metadata should load")
                .len();
            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let lock = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(byte_len)
                        .expect("mapped APFS control fixture length should be accepted"),
                )
                .expect("mapped APFS control fixture should qualify")
                .try_lock_preopened(
                    preopened_control(&lock_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("mapped APFS control fixture should lock");

            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let mut mapping = Some(mapping);
            let mut mutated = false;
            let _guard = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("boundary log should lock")
                    .push(boundary);
                if !mutated && boundary == TestBoundary::BeforeFileSync {
                    mutate_mapping(
                        mapping
                            .as_mut()
                            .expect("shared APFS mapping should remain retained by the hook"),
                    );
                    mutated = true;
                }
                Ok(())
            });
            let error = lock
                .sync_durable()
                .expect_err("APFS mutation at durability must fail after the barrier");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert_eq!(error.stage(), GenerationRootStage::SyncRegularFile);
            let observed = boundaries.lock().expect("boundary log should lock");
            let after_sync = observed
                .iter()
                .position(|boundary| *boundary == TestBoundary::AfterFileSync)
                .expect("the APFS durability barrier should complete before post-check failure");
            assert!(
                observed.iter().enumerate().any(|(index, boundary)| {
                    index > after_sync && *boundary == (TestBoundary::BeforeRead { offset: 0 })
                }),
                "a second exact APFS content read must follow the durability barrier"
            );
            drop(observed);
        }

        #[test]
        fn physical_apfs_rejects_every_non_empty_allow_or_deny_acl() {
            for (label, allow) in [("allow-acl", true), ("deny-acl", false)] {
                let root_path = fixture_root(label);
                set_non_empty_acl(&root_path, allow);

                let error = QualifiedGenerationRoot::admit(&root_path)
                    .expect_err("every non-empty APFS ACL must fail closed");
                assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
                assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
                assert!(
                    error.to_string().contains("descriptor-bound absence"),
                    "operator diagnostic must explain the descriptor-bound fail-closed policy"
                );
            }
        }

        #[test]
        fn physical_apfs_rejects_acl_bearing_artifact_control_and_lock_targets() {
            let artifact_root = fixture_root("artifact-target-acl");
            let artifact_path = private_file(&artifact_root, "artifact", b"sealed");
            set_non_empty_acl(&artifact_path, true);
            let artifact_generation = QualifiedGenerationRoot::admit(&artifact_root)
                .expect("the empty-ACL APFS root should qualify");
            assert_eq!(
                artifact_generation
                    .admit_preopened_file(
                        &confined("artifact"),
                        preopened_immutable(&artifact_path),
                        immutable_expectation(b"sealed"),
                    )
                    .expect_err("an ACL-bearing artifact descriptor must fail")
                    .kind(),
                GenerationRootErrorKind::AclRejected
            );

            let control_root = fixture_root("control-target-acl");
            let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
            set_non_empty_acl(&control_path, true);
            let control_generation = QualifiedGenerationRoot::admit(&control_root)
                .expect("the empty-ACL APFS root should qualify");
            assert_eq!(
                control_generation
                    .admit_preopened_control_file(
                        &confined("LOCK"),
                        preopened_control(&control_path),
                        GenerationFileExpectation::control(8)
                            .expect("small control expectation should be accepted"),
                    )
                    .expect_err("an ACL-bearing control descriptor must fail")
                    .kind(),
                GenerationRootErrorKind::AclRejected
            );

            let lock_root = fixture_root("lock-target-acl");
            let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
            let lock_generation = QualifiedGenerationRoot::admit(&lock_root)
                .expect("the empty-ACL APFS root should qualify");
            let control = lock_generation
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("the empty-ACL control should qualify before target mutation");
            let lock_candidate = preopened_control(&lock_path);
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let hook_lock_path = lock_path.clone();
            let mut mutated = false;
            let _hook = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("hook state should lock")
                    .push(boundary);
                if !mutated && boundary == TestBoundary::BeforePreopenedQualification {
                    set_non_empty_acl(&hook_lock_path, true);
                    mutated = true;
                }
                Ok(())
            });
            assert_eq!(
                control
                    .try_lock_preopened(lock_candidate, GenerationRootLockMode::Exclusive,)
                    .expect_err("the ACL-bearing fresh lock candidate must fail before flock")
                    .kind(),
                GenerationRootErrorKind::AclRejected
            );
            let observed = boundaries.lock().expect("hook state should lock");
            assert!(
                observed.contains(&TestBoundary::BeforePreopenedQualification),
                "the retained control route must revalidate before candidate mutation"
            );
            assert!(
                observed.contains(&TestBoundary::BeforeAclRead),
                "the fresh candidate must reach the descriptor-bound ACL probe"
            );
            assert!(
                !observed.contains(&TestBoundary::BeforeLock),
                "fresh-candidate ACL rejection must precede flock"
            );
            drop(observed);
        }

        #[test]
        fn physical_apfs_rejects_nonzero_flags_on_artifact_control_and_lock_targets() {
            for (label, flag) in [
                ("artifact-immutable-flag", "uchg"),
                ("artifact-nodump-flag", "nodump"),
            ] {
                let root_path = fixture_root(label);
                let artifact_path = private_file(&root_path, "artifact", b"sealed");
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("the zero-flags APFS root should qualify");
                run_chflags(flag, &artifact_path);
                let observed_flags = fs::metadata(&artifact_path)
                    .expect("flagged APFS artifact metadata should load")
                    .st_flags();
                assert_ne!(observed_flags, 0, "chflags must establish the fixture");
                let error = root
                    .admit_preopened_file(
                        &confined("artifact"),
                        preopened_immutable(&artifact_path),
                        immutable_expectation(b"sealed"),
                    )
                    .expect_err("a flagged artifact must be outside the zero-flags profile");
                assert_eq!(
                    error.kind(),
                    GenerationRootErrorKind::UnsupportedObjectFlags
                );
                assert_eq!(error.expected(), Some(0));
                assert_eq!(error.observed(), Some(u64::from(observed_flags)));
            }

            let control_root = fixture_root("control-nodump-flag");
            let control_path = control_file(&control_root, "LOCK", b"lock-v1!");
            let control_generation = QualifiedGenerationRoot::admit(&control_root)
                .expect("the zero-flags APFS root should qualify");
            run_chflags("nodump", &control_path);
            assert_ne!(
                fs::metadata(&control_path)
                    .expect("flagged APFS control metadata should load")
                    .st_flags(),
                0
            );
            assert_eq!(
                control_generation
                    .admit_preopened_control_file(
                        &confined("LOCK"),
                        preopened_control(&control_path),
                        GenerationFileExpectation::control(8)
                            .expect("small control expectation should be accepted"),
                    )
                    .expect_err("a flagged control target must fail")
                    .kind(),
                GenerationRootErrorKind::UnsupportedObjectFlags
            );

            let lock_root = fixture_root("lock-nodump-flag");
            let lock_path = control_file(&lock_root, "LOCK", b"lock-v1!");
            let lock_generation = QualifiedGenerationRoot::admit(&lock_root)
                .expect("the zero-flags APFS root should qualify");
            let control = lock_generation
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&lock_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("the zero-flags control should qualify before target mutation");
            let lock_candidate = preopened_control(&lock_path);
            let boundaries = Arc::new(Mutex::new(Vec::new()));
            let hook_boundaries = Arc::clone(&boundaries);
            let hook_lock_path = lock_path.clone();
            let mut mutated = false;
            let _hook = install_test_hook(move |boundary| {
                hook_boundaries
                    .lock()
                    .expect("hook state should lock")
                    .push(boundary);
                if !mutated && boundary == TestBoundary::BeforePreopenedQualification {
                    run_chflags("nodump", &hook_lock_path);
                    mutated = true;
                }
                Ok(())
            });
            let error = control
                .try_lock_preopened(lock_candidate, GenerationRootLockMode::Exclusive)
                .expect_err("the flagged fresh lock candidate must fail before flock");
            assert_eq!(
                error.kind(),
                GenerationRootErrorKind::UnsupportedObjectFlags
            );
            let observed = boundaries.lock().expect("hook state should lock");
            assert!(
                observed.contains(&TestBoundary::BeforePreopenedQualification),
                "the retained control route must revalidate before candidate mutation"
            );
            assert!(
                !observed.contains(&TestBoundary::BeforeLock),
                "fresh-candidate object-flag rejection must precede flock"
            );
            drop(observed);
        }

        #[test]
        fn physical_apfs_acl_bearing_retained_descriptor_rejects_empty_replacement() {
            let root_path = fixture_root("retained-acl");
            set_non_empty_acl(&root_path, true);
            let displaced = root_path.with_extension("retained-acl-original");
            let canonical = root_path.clone();
            let substituted = Arc::new(Mutex::new(false));
            let hook_substituted = Arc::clone(&substituted);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeAclRead
                    && !*hook_substituted.lock().expect("hook state should lock")
                {
                    fs::rename(&canonical, &displaced)
                        .expect("ACL-bearing root should move to a retained route");
                    DirBuilder::new()
                        .mode(0o700)
                        .create(&canonical)
                        .expect("empty-ACL replacement root should be creatable");
                    fs::set_permissions(&canonical, fs::Permissions::from_mode(0o700))
                        .expect("replacement root mode should be settable");
                    set_empty_acl(&canonical);
                    *hook_substituted.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&root_path).expect_err(
                "the ACL-bearing retained descriptor must reject before route fallback",
            );
            assert_eq!(error.kind(), GenerationRootErrorKind::AclRejected);
            assert_eq!(error.stage(), GenerationRootStage::InspectAcl);
            assert!(*substituted.lock().expect("hook state should lock"));
        }

        #[test]
        fn physical_apfs_empty_retained_descriptor_ignores_acl_bearing_replacement() {
            let root_path = fixture_root("retained-empty-acl");
            let displaced = root_path.with_extension("retained-empty-acl-original");
            let canonical = root_path.clone();
            let substituted = Arc::new(Mutex::new(false));
            let hook_substituted = Arc::clone(&substituted);
            let _guard = install_test_hook(move |boundary| {
                if boundary == TestBoundary::BeforeAclRead
                    && !*hook_substituted.lock().expect("hook state should lock")
                {
                    fs::rename(&canonical, &displaced)
                        .expect("empty-ACL root should move to a retained route");
                    DirBuilder::new()
                        .mode(0o700)
                        .create(&canonical)
                        .expect("replacement root should be creatable");
                    fs::set_permissions(&canonical, fs::Permissions::from_mode(0o700))
                        .expect("replacement root mode should be settable");
                    set_non_empty_acl(&canonical, true);
                    *hook_substituted.lock().expect("hook state should lock") = true;
                }
                Ok(())
            });
            let error = QualifiedGenerationRoot::admit(&root_path)
                .expect_err("canonical route replacement must fail after retained ACL inspection");
            assert_eq!(error.kind(), GenerationRootErrorKind::ObjectChanged);
            assert!(*substituted.lock().expect("hook state should lock"));
        }

        #[test]
        fn physical_apfs_excludes_regular_file_atime_only_drift_and_never_restores_it() {
            let root_path = fixture_root("regular-file-atime-exclusion");
            let artifact_path = private_file(&root_path, "artifact", b"atime");
            let control_path = control_file(&root_path, "LOCK", b"lock-v1!");
            let future_modified = SystemTime::now() + Duration::from_secs(24 * 60 * 60);
            for path in [&artifact_path, &control_path] {
                let timestamp_owner =
                    File::open(path).expect("fixture should open for timestamp setup");
                timestamp_owner
                    .set_times(
                        std::fs::FileTimes::new()
                            .set_accessed(UNIX_EPOCH)
                            .set_modified(future_modified),
                    )
                    .expect(
                        "fixture atime must be old and mtime future-dated before admission so \
                         each physical read remains relatime-eligible",
                    );
            }

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let artifact = root
                .admit_preopened_file(
                    &confined("artifact"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(b"atime"),
                )
                .expect("APFS artifact should qualify");
            let control = root
                .admit_preopened_control_file(
                    &confined("LOCK"),
                    preopened_control(&control_path),
                    GenerationFileExpectation::control(8)
                        .expect("small control expectation should be accepted"),
                )
                .expect("APFS control should qualify");
            let artifact_before = fs::metadata(&artifact_path)
                .expect("artifact metadata should load")
                .accessed()
                .expect("artifact atime should be readable");
            let control_before = fs::metadata(&control_path)
                .expect("control metadata should load")
                .accessed()
                .expect("control atime should be readable");

            std::thread::sleep(Duration::from_millis(20));
            for path in [&artifact_path, &control_path] {
                let mut reader = File::open(path).expect("ambient reader should open");
                let mut byte = [0_u8; 1];
                reader
                    .read_exact(&mut byte)
                    .expect("ambient read should advance atime");
            }
            let artifact_after = fs::metadata(&artifact_path)
                .expect("artifact metadata should reload")
                .accessed()
                .expect("advanced artifact atime should be readable");
            let control_after = fs::metadata(&control_path)
                .expect("control metadata should reload")
                .accessed()
                .expect("advanced control atime should be readable");
            assert!(
                artifact_after > artifact_before,
                "fixture must prove post-admission artifact-atime advancement"
            );
            assert!(
                control_after > control_before,
                "fixture must prove post-admission control-atime advancement"
            );

            artifact
                .sync_durable()
                .expect("APFS artifact atime alone is excluded from the mutation witness");
            control
                .try_lock_preopened(
                    preopened_control(&control_path),
                    GenerationRootLockMode::Exclusive,
                )
                .expect("APFS control atime alone must not fence locking")
                .unlock()
                .expect("atime-only APFS lock should release");
            assert!(
                fs::metadata(&artifact_path)
                    .expect("artifact metadata should remain readable")
                    .accessed()
                    .expect("artifact atime should remain readable")
                    >= artifact_after,
                "validation must never restore the artifact atime"
            );
            assert!(
                fs::metadata(&control_path)
                    .expect("control metadata should remain readable")
                    .accessed()
                    .expect("control atime should remain readable")
                    >= control_after,
                "validation must never restore the control atime"
            );
        }

        #[test]
        fn physical_apfs_excludes_directory_atime_only_drift_and_never_restores_it() {
            let root_path = fixture_root("atime-exclusion");
            private_file(&root_path, "directory-entry", b"atime");
            let timestamp_owner =
                File::open(&root_path).expect("fixture directory should open for timestamp");
            let future_modified = SystemTime::now() + Duration::from_secs(24 * 60 * 60);
            timestamp_owner
                .set_times(
                    std::fs::FileTimes::new()
                        .set_accessed(UNIX_EPOCH)
                        .set_modified(future_modified),
                )
                .expect(
                    "directory atime must be old and mtime future-dated before admission so \
                     each physical readdir remains relatime-eligible",
                );
            drop(timestamp_owner);

            let root =
                QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
            let before = fs::metadata(&root_path)
                .expect("directory metadata should load")
                .accessed()
                .expect("directory atime should be readable");
            let observed_entry = fs::read_dir(&root_path)
                .expect("ambient directory reader should open")
                .next()
                .transpose()
                .expect("ambient directory entry should be readable");
            assert!(
                observed_entry.is_some(),
                "fixture must contain an entry so readdir performs observable work"
            );
            let after = fs::metadata(&root_path)
                .expect("directory metadata should reload")
                .accessed()
                .expect("advanced directory atime should be readable");
            assert!(
                after > before,
                "fixture must prove a post-admission directory-atime advancement"
            );
            root.sync_directory_durable()
                .expect("APFS directory atime alone is excluded from the mutation witness");
            assert_eq!(
                fs::metadata(&root_path)
                    .expect("directory metadata should remain readable")
                    .accessed()
                    .expect("directory atime should remain readable"),
                after,
                "validation must never restore attacker-observed atime"
            );
        }

        #[test]
        fn every_macos_specific_open_acl_and_fullsync_boundary_is_fault_injectable() {
            for target in [
                TestBoundary::BeforeExactNameEnumeration,
                TestBoundary::AfterExactNameEnumeration,
                TestBoundary::BeforeAclRead,
                TestBoundary::AfterAclRead,
            ] {
                let root_path = fixture_root("macos-boundary");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::InspectAcl));
                    }
                    Ok(())
                });
                assert_eq!(
                    QualifiedGenerationRoot::admit(&root_path)
                        .expect_err("injected macOS boundary must fail")
                        .kind(),
                    GenerationRootErrorKind::Io
                );
            }

            for target in [
                TestBoundary::BeforePreopenedQualification,
                TestBoundary::AfterPreopenedQualification,
                TestBoundary::BeforeFinalRouteStat,
                TestBoundary::AfterFinalRouteStat,
            ] {
                let root_path = fixture_root("preopened-boundary");
                let artifact_path = private_file(&root_path, "artifact", b"sealed");
                let root =
                    QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
                let _guard = install_test_hook(move |boundary| {
                    if boundary == target {
                        return Err(injected(GenerationRootStage::OpenRegularFile));
                    }
                    Ok(())
                });
                assert_eq!(
                    root.admit_preopened_file(
                        &confined("artifact"),
                        preopened_immutable(&artifact_path),
                        immutable_expectation(b"sealed"),
                    )
                    .expect_err("injected preopened qualification boundary must fail")
                    .kind(),
                    GenerationRootErrorKind::Io
                );
            }

            for target in [TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync] {
                let root_path = fixture_root("file-fullsync-failure");
                let artifact_path = private_file(&root_path, "artifact", b"barrier");
                let root =
                    QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
                let admitted = root
                    .admit_preopened_file(
                        &confined("artifact"),
                        preopened_immutable(&artifact_path),
                        immutable_expectation(b"barrier"),
                    )
                    .expect("empty-ACL immutable APFS file should qualify");
                let calls = Arc::new(Mutex::new(Vec::new()));
                let hook_calls = Arc::clone(&calls);
                let _guard = install_test_hook(move |boundary| {
                    if matches!(
                        boundary,
                        TestBoundary::BeforeFileSync | TestBoundary::AfterFileSync
                    ) {
                        hook_calls
                            .lock()
                            .expect("hook calls should lock")
                            .push(boundary);
                    }
                    if boundary == target {
                        return Err(GenerationRootError::new(
                            GenerationRootErrorKind::DurabilityUnavailable,
                            GenerationRootStage::SyncRegularFile,
                        ));
                    }
                    Ok(())
                });
                assert_eq!(
                    admitted
                        .sync_durable()
                        .expect_err("required file F_FULLFSYNC boundary failure must propagate")
                        .kind(),
                    GenerationRootErrorKind::DurabilityUnavailable
                );
                let expected = if target == TestBoundary::BeforeFileSync {
                    &[TestBoundary::BeforeFileSync][..]
                } else {
                    &[TestBoundary::BeforeFileSync, TestBoundary::AfterFileSync][..]
                };
                assert_eq!(
                    calls.lock().expect("hook calls should lock").as_slice(),
                    expected,
                    "file F_FULLFSYNC failure must not retry or fall back to a weaker barrier"
                );
            }

            for target in [
                TestBoundary::BeforeDirectorySync,
                TestBoundary::AfterDirectorySync,
            ] {
                let root_path = fixture_root("directory-fullsync-failure");
                let root =
                    QualifiedGenerationRoot::admit(&root_path).expect("local writable APFS root");
                let calls = Arc::new(Mutex::new(Vec::new()));
                let hook_calls = Arc::clone(&calls);
                let _guard = install_test_hook(move |boundary| {
                    if matches!(
                        boundary,
                        TestBoundary::BeforeDirectorySync | TestBoundary::AfterDirectorySync
                    ) {
                        hook_calls
                            .lock()
                            .expect("hook calls should lock")
                            .push(boundary);
                    }
                    if boundary == target {
                        return Err(GenerationRootError::new(
                            GenerationRootErrorKind::DurabilityUnavailable,
                            GenerationRootStage::SyncDirectory,
                        ));
                    }
                    Ok(())
                });
                assert_eq!(
                    root.sync_directory_durable()
                        .expect_err(
                            "required directory F_FULLFSYNC boundary failure must propagate"
                        )
                        .kind(),
                    GenerationRootErrorKind::DurabilityUnavailable
                );
                let expected = if target == TestBoundary::BeforeDirectorySync {
                    &[TestBoundary::BeforeDirectorySync][..]
                } else {
                    &[
                        TestBoundary::BeforeDirectorySync,
                        TestBoundary::AfterDirectorySync,
                    ][..]
                };
                assert_eq!(
                    calls.lock().expect("hook calls should lock").as_slice(),
                    expected,
                    "directory F_FULLFSYNC failure must not retry or fall back to a weaker barrier"
                );
            }
        }

        #[test]
        #[ignore = "physical M4 mount matrix requires explicit host paths"]
        fn physical_m4_external_noowners_and_network_mounts_fail_before_mutation() {
            let external = std::env::var_os("FRANKENSEARCH_M4_EXTERNAL_NOOWNERS_ROOT")
                .expect("set FRANKENSEARCH_M4_EXTERNAL_NOOWNERS_ROOT for the physical receipt");
            let network = std::env::var_os("FRANKENSEARCH_M4_NETWORK_ROOT")
                .expect("set FRANKENSEARCH_M4_NETWORK_ROOT for the physical receipt");

            let external_error = QualifiedGenerationRoot::admit(Path::new(&external))
                .expect_err("APFS with ownership disabled must fail qualification");
            assert!(matches!(
                external_error.kind(),
                GenerationRootErrorKind::ReadOnlyFilesystem
                    | GenerationRootErrorKind::UnsupportedFilesystem
            ));
            assert_eq!(
                QualifiedGenerationRoot::admit(Path::new(&network))
                    .expect_err("network filesystem must fail qualification")
                    .kind(),
                GenerationRootErrorKind::UnsupportedFilesystem
            );
        }

        #[test]
        #[ignore = "physical M4 case-sensitive and case-insensitive APFS bases are explicit"]
        fn physical_m4_case_sensitive_and_insensitive_apfs_both_require_exact_bytes() {
            let insensitive_base = std::env::var_os("FRANKENSEARCH_M4_CASE_INSENSITIVE_APFS_BASE")
                .expect("set the case-insensitive APFS base for the physical receipt");
            let sensitive_base = std::env::var_os("FRANKENSEARCH_M4_CASE_SENSITIVE_APFS_BASE")
                .expect("set the case-sensitive APFS base for the physical receipt");

            for (base, expected_sensitive, label) in [
                (insensitive_base, false, "case-insensitive"),
                (sensitive_base, true, "case-sensitive"),
            ] {
                let root_path = fixture_root_under(Path::new(&base), label);
                assert_eq!(
                    observe_case_sensitive(&root_path),
                    expected_sensitive,
                    "fixture volume sensitivity must match its declared physical lane"
                );
                let artifact_path = private_file(&root_path, "CaseArtifact", b"case");
                let root = QualifiedGenerationRoot::admit(&root_path)
                    .expect("local ownership-aware APFS fixture should qualify");
                root.admit_preopened_file(
                    &confined("CaseArtifact"),
                    preopened_immutable(&artifact_path),
                    immutable_expectation(b"case"),
                )
                .expect("the exact-case route must qualify on each APFS personality");
                assert_eq!(
                    root.admit_preopened_file(
                        &confined("caseartifact"),
                        preopened_immutable(&artifact_path),
                        immutable_expectation(b"case"),
                    )
                    .expect_err("byte-inexact spelling must fail on every APFS personality")
                    .kind(),
                    GenerationRootErrorKind::ObjectChanged
                );
            }
        }
    }
}
