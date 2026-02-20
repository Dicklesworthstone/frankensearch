//! Network filesystem detection and edge-case handling for corpus discovery.
//!
//! Implements bd-2hz.2.6: detects mount types (NFS, SSHFS, FUSE, etc.),
//! classifies filesystem behavior, and provides per-mount policies that
//! the discovery walker uses to adapt its strategy.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ─── Filesystem Type Classification ─────────────────────────────────────────

/// Broad category of a mounted filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsCategory {
    /// Local disk (ext4, xfs, btrfs, apfs, ntfs, etc.).
    Local,
    /// Network-attached via NFS.
    Nfs,
    /// SSH-based remote mount.
    Sshfs,
    /// Generic FUSE mount (rclone, Google Drive Stream, etc.).
    Fuse,
    /// CIFS / SMB network share.
    Cifs,
    /// In-memory filesystem (tmpfs, ramfs).
    Memory,
    /// Virtual / pseudo filesystem (proc, sys, devtmpfs).
    Virtual,
    /// Unknown filesystem type.
    Unknown,
}

impl FsCategory {
    /// Whether this filesystem category is network-based.
    #[must_use]
    pub const fn is_network(self) -> bool {
        matches!(self, Self::Nfs | Self::Sshfs | Self::Fuse | Self::Cifs)
    }

    /// Whether this filesystem should be skipped entirely by default
    /// (pseudo-filesystems that never contain user content).
    #[must_use]
    pub const fn is_virtual(self) -> bool {
        matches!(self, Self::Virtual)
    }

    /// Whether inotify / kqueue / `FSEvents` can be trusted on this mount.
    #[must_use]
    pub const fn supports_reliable_watch(self) -> bool {
        matches!(self, Self::Local | Self::Memory)
    }
}

/// Classify a filesystem type string (from /proc/mounts or statfs) into a category.
#[must_use]
pub fn classify_fstype(fstype: &str) -> FsCategory {
    match fstype {
        // Local disk filesystems.
        "ext2" | "ext3" | "ext4" | "xfs" | "btrfs" | "zfs" | "f2fs" | "reiserfs" | "jfs"
        | "nilfs2" | "bcachefs" => FsCategory::Local,
        // macOS / Windows local.
        "apfs" | "hfs" | "hfsplus" | "ntfs" | "ntfs3" | "vfat" | "fat32" | "exfat" => {
            FsCategory::Local
        }
        // NFS variants.
        "nfs" | "nfs4" | "nfsd" => FsCategory::Nfs,
        // CIFS / SMB.
        "cifs" | "smb" | "smb2" | "smbfs" => FsCategory::Cifs,
        // FUSE-based (may be local or network depending on the underlying driver).
        "fuse" | "fuseblk" | "fuse.sshfs" => {
            // fuse.sshfs is specifically SSHFS.
            if fstype == "fuse.sshfs" {
                FsCategory::Sshfs
            } else {
                FsCategory::Fuse
            }
        }
        // In-memory.
        "tmpfs" | "ramfs" => FsCategory::Memory,
        // Virtual / pseudo filesystems — always skip.
        "proc" | "sysfs" | "devtmpfs" | "devpts" | "securityfs" | "cgroup" | "cgroup2"
        | "pstore" | "debugfs" | "tracefs" | "hugetlbfs" | "mqueue" | "configfs" | "efivarfs"
        | "binfmt_misc" | "fusectl" | "autofs" | "bpf" | "nsfs" | "overlay" => FsCategory::Virtual,
        _ => {
            // Catch FUSE sub-types like "fuse.rclone", "fuse.gocryptfs", etc.
            if fstype.starts_with("fuse.") {
                if fstype == "fuse.sshfs" {
                    FsCategory::Sshfs
                } else {
                    FsCategory::Fuse
                }
            } else {
                FsCategory::Unknown
            }
        }
    }
}

// ─── Mount Entry ─────────────────────────────────────────────────────────────

/// A single mount point with its filesystem classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    /// Device or remote path (e.g., "/dev/sda1" or `"host:/export"`).
    pub device: String,
    /// Local mount point path.
    pub mount_point: PathBuf,
    /// Raw filesystem type string from the OS.
    pub fstype: String,
    /// Classified category.
    pub category: FsCategory,
    /// Mount options (comma-separated in raw form).
    pub options: String,
}

// ─── Mount Policy ────────────────────────────────────────────────────────────

/// What strategy to use for change detection on a given mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeDetectionStrategy {
    /// Use inotify/kqueue/FSEvents for real-time notifications.
    Watch,
    /// Periodically rescan the mount point.
    Poll,
    /// Don't monitor for changes (index once, then ignore).
    Static,
}

/// Per-mount behavioral policy that controls how discovery/indexing interacts
/// with a specific mount point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MountPolicy {
    /// How to detect file changes on this mount.
    pub change_detection: ChangeDetectionStrategy,
    /// Maximum time to wait for a single `stat()` call before considering
    /// the mount slow/unavailable.
    pub stat_timeout: Duration,
    /// Maximum concurrent I/O operations to issue against this mount.
    pub max_concurrent_io: usize,
    /// Interval between poll rescans (only used when `change_detection` == Poll).
    pub poll_interval: Duration,
    /// Whether to include this mount in discovery at all.
    pub enabled: bool,
    /// Whether the mount is currently considered available.
    /// Set to false when repeated stat timeouts occur.
    pub available: bool,
}

impl MountPolicy {
    /// Default policy for local filesystems — fast, reliable, use watch events.
    #[must_use]
    pub const fn local_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Watch,
            stat_timeout: Duration::from_secs(5),
            max_concurrent_io: 64,
            poll_interval: Duration::from_mins(5),
            enabled: true,
            available: true,
        }
    }

    /// Default policy for NFS mounts — polling fallback, moderate timeouts.
    #[must_use]
    pub const fn nfs_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Poll,
            stat_timeout: Duration::from_secs(2),
            max_concurrent_io: 8,
            poll_interval: Duration::from_mins(1),
            enabled: true,
            available: true,
        }
    }

    /// Default policy for SSHFS — polling, longer intervals due to high latency.
    #[must_use]
    pub const fn sshfs_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Poll,
            stat_timeout: Duration::from_secs(5),
            max_concurrent_io: 4,
            poll_interval: Duration::from_mins(2),
            enabled: true,
            available: true,
        }
    }

    /// Default policy for generic FUSE — conservative mode.
    #[must_use]
    pub const fn fuse_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Poll,
            stat_timeout: Duration::from_secs(3),
            max_concurrent_io: 4,
            poll_interval: Duration::from_mins(2),
            enabled: true,
            available: true,
        }
    }

    /// Default policy for CIFS/SMB shares.
    #[must_use]
    pub const fn cifs_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Poll,
            stat_timeout: Duration::from_secs(2),
            max_concurrent_io: 8,
            poll_interval: Duration::from_mins(1),
            enabled: true,
            available: true,
        }
    }

    /// Policy for virtual/pseudo filesystems — disabled by default.
    #[must_use]
    pub const fn virtual_default() -> Self {
        Self {
            change_detection: ChangeDetectionStrategy::Static,
            stat_timeout: Duration::from_millis(500),
            max_concurrent_io: 1,
            poll_interval: Duration::from_hours(1),
            enabled: false,
            available: true,
        }
    }

    /// Select the appropriate default policy for a filesystem category.
    #[must_use]
    pub const fn for_category(category: FsCategory) -> Self {
        match category {
            FsCategory::Local | FsCategory::Memory => Self::local_default(),
            FsCategory::Nfs => Self::nfs_default(),
            FsCategory::Sshfs => Self::sshfs_default(),
            FsCategory::Fuse | FsCategory::Unknown => Self::fuse_default(),
            FsCategory::Cifs => Self::cifs_default(),
            FsCategory::Virtual => Self::virtual_default(),
        }
    }
}

// ─── Configurable Mount Overrides ────────────────────────────────────────────

/// User-supplied per-mount policy override in the config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MountOverride {
    /// Override the filesystem category detection.
    pub category: Option<FsCategory>,
    /// Override change detection strategy.
    pub change_detection: Option<ChangeDetectionStrategy>,
    /// Override stat timeout in milliseconds.
    pub stat_timeout_ms: Option<u64>,
    /// Override max concurrent I/O.
    pub max_concurrent_io: Option<usize>,
    /// Override poll interval in seconds.
    pub poll_interval_secs: Option<u64>,
    /// Explicitly enable or disable this mount.
    pub enabled: Option<bool>,
}

impl MountOverride {
    /// Apply this override to a base policy.
    #[must_use]
    pub const fn apply(&self, mut base: MountPolicy) -> MountPolicy {
        if let Some(strategy) = self.change_detection {
            base.change_detection = strategy;
        }
        if let Some(ms) = self.stat_timeout_ms {
            base.stat_timeout = Duration::from_millis(ms);
        }
        if let Some(max) = self.max_concurrent_io {
            base.max_concurrent_io = max;
        }
        if let Some(secs) = self.poll_interval_secs {
            base.poll_interval = Duration::from_secs(secs);
        }
        if let Some(enabled) = self.enabled {
            base.enabled = enabled;
        }
        base
    }
}

// ─── Mount Table ─────────────────────────────────────────────────────────────

/// Parsed mount table with per-mount policies.
#[derive(Debug, Clone)]
pub struct MountTable {
    entries: Vec<MountEntry>,
    policies: HashMap<PathBuf, MountPolicy>,
}

impl MountTable {
    /// Build a mount table from parsed entries with default policies,
    /// applying any user overrides.
    #[must_use]
    pub fn new(entries: Vec<MountEntry>, overrides: &HashMap<String, MountOverride>) -> Self {
        let mut effective_entries = Vec::with_capacity(entries.len());
        let mut policies = HashMap::with_capacity(entries.len());

        for mut entry in entries {
            let mount_path_str = entry.mount_point.to_string_lossy();
            let override_entry = overrides.get(mount_path_str.as_ref());

            if let Some(category) = override_entry.and_then(|ovr| ovr.category) {
                entry.category = category;
            }

            let base = MountPolicy::for_category(entry.category);
            let policy = if let Some(ovr) = override_entry {
                ovr.apply(base)
            } else {
                base
            };

            policies.insert(entry.mount_point.clone(), policy);
            effective_entries.push(entry);
        }

        Self {
            entries: effective_entries,
            policies,
        }
    }

    /// Look up the mount entry and policy for a given file path.
    ///
    /// Finds the longest-prefix mount point that contains the path.
    #[must_use]
    pub fn lookup(&self, path: &Path) -> Option<(&MountEntry, &MountPolicy)> {
        let mut best: Option<(&MountEntry, &MountPolicy)> = None;
        let mut best_len = 0;

        for entry in &self.entries {
            let mp = &entry.mount_point;
            if path.strip_prefix(mp).is_ok() {
                let len = mp.as_os_str().len();
                if len > best_len {
                    best_len = len;
                    if let Some(policy) = self.policies.get(mp) {
                        best = Some((entry, policy));
                    }
                }
            }
        }

        best
    }

    /// Iterate over all mount entries.
    #[must_use]
    pub fn entries(&self) -> &[MountEntry] {
        &self.entries
    }

    /// Get the policy for a specific mount point.
    #[must_use]
    pub fn policy_for(&self, mount_point: &Path) -> Option<&MountPolicy> {
        self.policies.get(mount_point)
    }

    /// List all network-type mount points.
    #[must_use]
    pub fn network_mounts(&self) -> Vec<&MountEntry> {
        self.entries
            .iter()
            .filter(|e| e.category.is_network())
            .collect()
    }

    /// List all enabled mount points (non-virtual, policy.enabled == true).
    #[must_use]
    pub fn enabled_mounts(&self) -> Vec<(&MountEntry, &MountPolicy)> {
        self.entries
            .iter()
            .filter_map(|entry| {
                let policy = self.policies.get(&entry.mount_point)?;
                if policy.enabled {
                    Some((entry, policy))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Mark a mount as unavailable (e.g., after repeated stat timeouts).
    /// Returns true if the mount was found and updated.
    pub fn mark_unavailable(&mut self, mount_point: &Path) -> bool {
        if let Some(policy) = self.policies.get_mut(mount_point) {
            policy.available = false;
            tracing::warn!(
                target: "frankensearch.fsfs.mount",
                mount = %mount_point.display(),
                "mount marked unavailable due to repeated failures"
            );
            true
        } else {
            false
        }
    }

    /// Mark a mount as available again (e.g., after a successful probe).
    pub fn mark_available(&mut self, mount_point: &Path) -> bool {
        if let Some(policy) = self.policies.get_mut(mount_point) {
            if !policy.available {
                policy.available = true;
                tracing::info!(
                    target: "frankensearch.fsfs.mount",
                    mount = %mount_point.display(),
                    "mount restored to available"
                );
            }
            true
        } else {
            false
        }
    }
}

// ─── Mount Probe (health check) ─────────────────────────────────────────────

/// Result of probing a single mount point for availability.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub mount_point: PathBuf,
    pub available: bool,
    pub latency: Duration,
    pub error: Option<String>,
}

/// Probe a mount point by attempting a `stat()` on it, with a timeout.
///
/// Returns the probe result including latency and any error.
pub fn probe_mount(mount_point: &Path, timeout: Duration) -> ProbeResult {
    let start = Instant::now();

    // We use a simple blocking stat check. In production with asupersync,
    // this would be wrapped in an async timeout.
    let result = std::fs::metadata(mount_point);
    let latency = start.elapsed();

    match result {
        Ok(_metadata) => {
            let available = latency < timeout;
            if !available {
                tracing::warn!(
                    target: "frankensearch.fsfs.mount",
                    mount = %mount_point.display(),
                    latency_ms = latency.as_millis(),
                    timeout_ms = timeout.as_millis(),
                    "mount probe exceeded timeout — classifying as slow"
                );
            }
            ProbeResult {
                mount_point: mount_point.to_owned(),
                available,
                latency,
                error: if available {
                    None
                } else {
                    Some(format!(
                        "stat latency {}ms exceeds timeout {}ms",
                        latency.as_millis(),
                        timeout.as_millis()
                    ))
                },
            }
        }
        Err(err) => {
            let classification = classify_io_error(&err);
            tracing::warn!(
                target: "frankensearch.fsfs.mount",
                mount = %mount_point.display(),
                error = %err,
                classification = ?classification,
                "mount probe failed"
            );
            ProbeResult {
                mount_point: mount_point.to_owned(),
                available: false,
                latency,
                error: Some(err.to_string()),
            }
        }
    }
}

// ─── Error Classification ────────────────────────────────────────────────────

/// Whether a mount I/O error is transient (may recover) or permanent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Temporarily unavailable — network hiccup, mount busy, etc.
    Transient,
    /// Permanently gone — mount removed, permission denied, etc.
    Permanent,
}

/// Classify an I/O error as transient or permanent.
#[must_use]
pub fn classify_io_error(err: &io::Error) -> ErrorClass {
    match err.kind() {
        // Permanent: the mount point itself is gone or inaccessible.
        io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => ErrorClass::Permanent,

        // Default to transient for unknown errors (safer — don't delete data).
        // This also covers explicit transient cases: TimedOut, ConnectionRefused,
        // ConnectionReset, ConnectionAborted, Interrupted, WouldBlock.
        _ => ErrorClass::Transient,
    }
}

// ─── Linux /proc/mounts Parser ───────────────────────────────────────────────

/// Parse `/proc/mounts` (or any file in the same format) into mount entries.
///
/// Format: `device mount_point fstype options dump_freq pass_num`
#[must_use]
pub fn parse_proc_mounts(content: &str) -> Vec<MountEntry> {
    let mut entries = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        let device = unescape_octal(parts[0]);
        let mount_point_str = unescape_octal(parts[1]);
        let fstype = parts[2].to_owned();
        let options = parts[3].to_owned();

        let category = classify_fstype(&fstype);

        entries.push(MountEntry {
            device,
            mount_point: PathBuf::from(mount_point_str),
            fstype,
            category,
            options,
        });
    }

    entries
}

/// Read and parse the system mount table.
///
/// On Linux, reads `/proc/mounts`. Returns an empty vec on non-Linux
/// platforms or if the mount table is unreadable.
pub fn read_system_mounts() -> Vec<MountEntry> {
    #[cfg(target_os = "linux")]
    {
        match std::fs::read_to_string("/proc/mounts") {
            Ok(content) => parse_proc_mounts(&content),
            Err(err) => {
                tracing::warn!(
                    target: "frankensearch.fsfs.mount",
                    error = %err,
                    "failed to read /proc/mounts; mount detection disabled"
                );
                Vec::new()
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        tracing::debug!(
            target: "frankensearch.fsfs.mount",
            "mount detection not implemented for this platform"
        );
        Vec::new()
    }
}

/// Unescape octal sequences in mount path strings (e.g., `\040` → space).
fn unescape_octal(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Try to read 3 octal digits.
            let mut octal = String::with_capacity(3);
            for _ in 0..3 {
                if let Some(&next) = chars.as_str().as_bytes().first() {
                    if next.is_ascii_digit() && next <= b'7' {
                        octal.push(next as char);
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            if octal.len() == 3 {
                if let Ok(byte) = u8::from_str_radix(&octal, 8) {
                    result.push(byte as char);
                } else {
                    result.push('\\');
                    result.push_str(&octal);
                }
            } else {
                result.push('\\');
                result.push_str(&octal);
            }
        } else {
            result.push(ch);
        }
    }

    result
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_local_filesystems() {
        assert_eq!(classify_fstype("ext4"), FsCategory::Local);
        assert_eq!(classify_fstype("xfs"), FsCategory::Local);
        assert_eq!(classify_fstype("btrfs"), FsCategory::Local);
        assert_eq!(classify_fstype("apfs"), FsCategory::Local);
        assert_eq!(classify_fstype("ntfs"), FsCategory::Local);
        assert_eq!(classify_fstype("ntfs3"), FsCategory::Local);
        assert_eq!(classify_fstype("vfat"), FsCategory::Local);
        assert_eq!(classify_fstype("bcachefs"), FsCategory::Local);
    }

    #[test]
    fn classify_network_filesystems() {
        assert_eq!(classify_fstype("nfs"), FsCategory::Nfs);
        assert_eq!(classify_fstype("nfs4"), FsCategory::Nfs);
        assert_eq!(classify_fstype("cifs"), FsCategory::Cifs);
        assert_eq!(classify_fstype("smb2"), FsCategory::Cifs);
        assert_eq!(classify_fstype("fuse.sshfs"), FsCategory::Sshfs);
    }

    #[test]
    fn classify_fuse_variants() {
        assert_eq!(classify_fstype("fuse"), FsCategory::Fuse);
        assert_eq!(classify_fstype("fuseblk"), FsCategory::Fuse);
        assert_eq!(classify_fstype("fuse.rclone"), FsCategory::Fuse);
        assert_eq!(classify_fstype("fuse.gocryptfs"), FsCategory::Fuse);
        assert_eq!(classify_fstype("fuse.sshfs"), FsCategory::Sshfs);
    }

    #[test]
    fn classify_memory_and_virtual() {
        assert_eq!(classify_fstype("tmpfs"), FsCategory::Memory);
        assert_eq!(classify_fstype("ramfs"), FsCategory::Memory);
        assert_eq!(classify_fstype("proc"), FsCategory::Virtual);
        assert_eq!(classify_fstype("sysfs"), FsCategory::Virtual);
        assert_eq!(classify_fstype("devtmpfs"), FsCategory::Virtual);
        assert_eq!(classify_fstype("cgroup2"), FsCategory::Virtual);
    }

    #[test]
    fn classify_unknown() {
        assert_eq!(classify_fstype("martianfs"), FsCategory::Unknown);
    }

    #[test]
    fn category_traits() {
        assert!(!FsCategory::Local.is_network());
        assert!(!FsCategory::Local.is_virtual());
        assert!(FsCategory::Local.supports_reliable_watch());

        assert!(FsCategory::Nfs.is_network());
        assert!(!FsCategory::Nfs.is_virtual());
        assert!(!FsCategory::Nfs.supports_reliable_watch());

        assert!(FsCategory::Sshfs.is_network());
        assert!(FsCategory::Fuse.is_network());
        assert!(FsCategory::Cifs.is_network());

        assert!(FsCategory::Virtual.is_virtual());
        assert!(!FsCategory::Virtual.is_network());

        assert!(FsCategory::Memory.supports_reliable_watch());
    }

    #[test]
    fn parse_proc_mounts_basic() {
        let content = "\
/dev/sda1 / ext4 rw,relatime 0 1
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
tmpfs /tmp tmpfs rw,nosuid,nodev 0 0
server:/export /mnt/nfs nfs4 rw,relatime,vers=4.2 0 0
user@host:/home /mnt/sshfs fuse.sshfs rw,nosuid,nodev 0 0
//server/share /mnt/smb cifs rw,credentials=/etc/cifs-creds 0 0
";

        let entries = parse_proc_mounts(content);
        assert_eq!(entries.len(), 6);

        assert_eq!(entries[0].mount_point, Path::new("/"));
        assert_eq!(entries[0].category, FsCategory::Local);

        assert_eq!(entries[1].mount_point, Path::new("/proc"));
        assert_eq!(entries[1].category, FsCategory::Virtual);

        assert_eq!(entries[2].mount_point, Path::new("/tmp"));
        assert_eq!(entries[2].category, FsCategory::Memory);

        assert_eq!(entries[3].mount_point, Path::new("/mnt/nfs"));
        assert_eq!(entries[3].category, FsCategory::Nfs);

        assert_eq!(entries[4].mount_point, Path::new("/mnt/sshfs"));
        assert_eq!(entries[4].category, FsCategory::Sshfs);

        assert_eq!(entries[5].mount_point, Path::new("/mnt/smb"));
        assert_eq!(entries[5].category, FsCategory::Cifs);
    }

    #[test]
    fn parse_handles_empty_and_comments() {
        let content = "\
# This is a comment

/dev/sda1 / ext4 rw,relatime 0 1

";
        let entries = parse_proc_mounts(content);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn parse_handles_octal_escaped_paths() {
        // Space in mount path is encoded as \040.
        let content = "/dev/sdb1 /mnt/My\\040Drive ext4 rw 0 0\n";
        let entries = parse_proc_mounts(content);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].mount_point, Path::new("/mnt/My Drive"));
    }

    #[test]
    fn unescape_octal_sequences() {
        assert_eq!(unescape_octal("hello\\040world"), "hello world");
        assert_eq!(unescape_octal("no\\011tab"), "no\ttab");
        assert_eq!(unescape_octal("plain"), "plain");
        assert_eq!(unescape_octal("trail\\040"), "trail ");
    }

    #[test]
    fn default_policies_per_category() {
        let local = MountPolicy::for_category(FsCategory::Local);
        assert_eq!(local.change_detection, ChangeDetectionStrategy::Watch);
        assert!(local.enabled);

        let nfs = MountPolicy::for_category(FsCategory::Nfs);
        assert_eq!(nfs.change_detection, ChangeDetectionStrategy::Poll);
        assert!(nfs.enabled);
        assert!(nfs.max_concurrent_io < local.max_concurrent_io);

        let sshfs = MountPolicy::for_category(FsCategory::Sshfs);
        assert_eq!(sshfs.change_detection, ChangeDetectionStrategy::Poll);
        assert!(sshfs.poll_interval > nfs.poll_interval);

        let virtual_fs = MountPolicy::for_category(FsCategory::Virtual);
        assert!(!virtual_fs.enabled);
    }

    #[test]
    fn mount_override_applies() {
        let base = MountPolicy::nfs_default();
        assert_eq!(base.change_detection, ChangeDetectionStrategy::Poll);
        assert_eq!(base.max_concurrent_io, 8);

        let ovr = MountOverride {
            change_detection: Some(ChangeDetectionStrategy::Watch),
            max_concurrent_io: Some(2),
            enabled: Some(false),
            ..MountOverride::default()
        };

        let applied = ovr.apply(base);
        assert_eq!(applied.change_detection, ChangeDetectionStrategy::Watch);
        assert_eq!(applied.max_concurrent_io, 2);
        assert!(!applied.enabled);
    }

    #[test]
    fn mount_table_lookup_longest_prefix() {
        let entries = vec![
            MountEntry {
                device: "/dev/sda1".into(),
                mount_point: PathBuf::from("/"),
                fstype: "ext4".into(),
                category: FsCategory::Local,
                options: "rw".into(),
            },
            MountEntry {
                device: "server:/export".into(),
                mount_point: PathBuf::from("/mnt/nfs"),
                fstype: "nfs4".into(),
                category: FsCategory::Nfs,
                options: "rw".into(),
            },
        ];

        let table = MountTable::new(entries, &HashMap::new());

        // A file under /mnt/nfs should match the NFS mount.
        let (entry, policy) = table.lookup(Path::new("/mnt/nfs/data/file.txt")).unwrap();
        assert_eq!(entry.category, FsCategory::Nfs);
        assert_eq!(policy.change_detection, ChangeDetectionStrategy::Poll);

        // A file under /home should match the root mount.
        let (entry, _) = table.lookup(Path::new("/home/user/code.rs")).unwrap();
        assert_eq!(entry.category, FsCategory::Local);
    }

    #[test]
    fn mount_table_lookup_respects_path_boundaries() {
        let entries = vec![
            MountEntry {
                device: "/dev/sda1".into(),
                mount_point: PathBuf::from("/"),
                fstype: "ext4".into(),
                category: FsCategory::Local,
                options: "rw".into(),
            },
            MountEntry {
                device: "server:/export".into(),
                mount_point: PathBuf::from("/mnt/nfs"),
                fstype: "nfs4".into(),
                category: FsCategory::Nfs,
                options: "rw".into(),
            },
        ];

        let table = MountTable::new(entries, &HashMap::new());

        // A file whose path only shares the textual prefix (but is outside the mount)
        // should not match the `/mnt/nfs` policy.
        let (entry, _) = table.lookup(Path::new("/mnt/nfs-shared/file.txt")).unwrap();
        assert_eq!(entry.category, FsCategory::Local);
    }

    #[test]
    fn mount_table_with_overrides() {
        let entries = vec![MountEntry {
            device: "server:/export".into(),
            mount_point: PathBuf::from("/mnt/nfs"),
            fstype: "nfs4".into(),
            category: FsCategory::Nfs,
            options: "rw".into(),
        }];

        let mut overrides = HashMap::new();
        overrides.insert(
            "/mnt/nfs".to_owned(),
            MountOverride {
                enabled: Some(false),
                ..MountOverride::default()
            },
        );

        let table = MountTable::new(entries, &overrides);
        let policy = table.policy_for(Path::new("/mnt/nfs")).unwrap();
        assert!(!policy.enabled);
    }

    #[test]
    fn mount_table_category_override_updates_lookup_category_and_policy_defaults() {
        let entries = vec![MountEntry {
            device: "/dev/sda1".into(),
            mount_point: PathBuf::from("/"),
            fstype: "ext4".into(),
            category: FsCategory::Local,
            options: "rw".into(),
        }];

        let mut overrides = HashMap::new();
        overrides.insert(
            "/".to_owned(),
            MountOverride {
                category: Some(FsCategory::Nfs),
                ..MountOverride::default()
            },
        );

        let table = MountTable::new(entries, &overrides);
        let (entry, policy) = table.lookup(Path::new("/tmp/file.txt")).unwrap();
        assert_eq!(
            entry.category,
            FsCategory::Nfs,
            "mount category override should affect lookup results"
        );
        assert_eq!(
            policy.change_detection,
            ChangeDetectionStrategy::Poll,
            "effective category should influence default mount policy"
        );
    }

    #[test]
    fn mount_table_network_mounts() {
        let entries = vec![
            MountEntry {
                device: "/dev/sda1".into(),
                mount_point: PathBuf::from("/"),
                fstype: "ext4".into(),
                category: FsCategory::Local,
                options: "rw".into(),
            },
            MountEntry {
                device: "server:/share".into(),
                mount_point: PathBuf::from("/mnt/nfs"),
                fstype: "nfs4".into(),
                category: FsCategory::Nfs,
                options: "rw".into(),
            },
            MountEntry {
                device: "//host/share".into(),
                mount_point: PathBuf::from("/mnt/smb"),
                fstype: "cifs".into(),
                category: FsCategory::Cifs,
                options: "rw".into(),
            },
        ];

        let table = MountTable::new(entries, &HashMap::new());
        let network = table.network_mounts();
        assert_eq!(network.len(), 2);
    }

    #[test]
    fn mount_table_enabled_mounts_skips_virtual() {
        let entries = vec![
            MountEntry {
                device: "/dev/sda1".into(),
                mount_point: PathBuf::from("/"),
                fstype: "ext4".into(),
                category: FsCategory::Local,
                options: "rw".into(),
            },
            MountEntry {
                device: "proc".into(),
                mount_point: PathBuf::from("/proc"),
                fstype: "proc".into(),
                category: FsCategory::Virtual,
                options: "rw".into(),
            },
        ];

        let table = MountTable::new(entries, &HashMap::new());
        let enabled = table.enabled_mounts();
        // Only the local mount should be enabled (virtual is disabled by default).
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].0.category, FsCategory::Local);
    }

    #[test]
    fn mark_unavailable_and_available() {
        let entries = vec![MountEntry {
            device: "server:/share".into(),
            mount_point: PathBuf::from("/mnt/nfs"),
            fstype: "nfs4".into(),
            category: FsCategory::Nfs,
            options: "rw".into(),
        }];

        let mut table = MountTable::new(entries, &HashMap::new());

        assert!(table.policy_for(Path::new("/mnt/nfs")).unwrap().available);

        assert!(table.mark_unavailable(Path::new("/mnt/nfs")));
        assert!(!table.policy_for(Path::new("/mnt/nfs")).unwrap().available);

        assert!(table.mark_available(Path::new("/mnt/nfs")));
        assert!(table.policy_for(Path::new("/mnt/nfs")).unwrap().available);
    }

    #[test]
    fn mark_unavailable_unknown_mount_returns_false() {
        let mut table = MountTable::new(vec![], &HashMap::new());
        assert!(!table.mark_unavailable(Path::new("/nonexistent")));
    }

    #[test]
    fn error_classification() {
        assert_eq!(
            classify_io_error(&io::Error::new(io::ErrorKind::TimedOut, "timeout")),
            ErrorClass::Transient
        );
        assert_eq!(
            classify_io_error(&io::Error::new(io::ErrorKind::ConnectionRefused, "refused")),
            ErrorClass::Transient
        );
        assert_eq!(
            classify_io_error(&io::Error::new(io::ErrorKind::NotFound, "not found")),
            ErrorClass::Permanent
        );
        assert_eq!(
            classify_io_error(&io::Error::new(io::ErrorKind::PermissionDenied, "denied")),
            ErrorClass::Permanent
        );
        // Unknown errors default to transient (safer).
        assert_eq!(
            classify_io_error(&io::Error::other("unknown")),
            ErrorClass::Transient
        );
    }

    #[test]
    fn probe_mount_on_existing_path() {
        // /tmp should exist and be fast.
        let result = probe_mount(Path::new("/tmp"), Duration::from_secs(5));
        assert!(result.available);
        assert!(result.error.is_none());
        assert!(result.latency < Duration::from_secs(1));
    }

    #[test]
    fn probe_mount_on_nonexistent_path() {
        let result = probe_mount(
            Path::new("/nonexistent_mount_point_xyz_12345"),
            Duration::from_secs(1),
        );
        assert!(!result.available);
        assert!(result.error.is_some());
    }

    #[test]
    fn mount_policy_serde_roundtrip() {
        let policy = MountPolicy::nfs_default();
        let json = serde_json::to_string(&policy).expect("serialize");
        let deserialized: MountPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.change_detection, policy.change_detection);
        assert_eq!(deserialized.max_concurrent_io, policy.max_concurrent_io);
        assert_eq!(deserialized.enabled, policy.enabled);
    }

    #[test]
    fn mount_override_serde_roundtrip() {
        let ovr = MountOverride {
            category: Some(FsCategory::Nfs),
            change_detection: Some(ChangeDetectionStrategy::Poll),
            stat_timeout_ms: Some(1000),
            max_concurrent_io: Some(4),
            poll_interval_secs: Some(60),
            enabled: Some(true),
        };
        let json = serde_json::to_string(&ovr).expect("serialize");
        let deserialized: MountOverride = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.category, ovr.category);
        assert_eq!(deserialized.change_detection, ovr.change_detection);
    }

    #[test]
    fn fs_category_serde_roundtrip() {
        for category in [
            FsCategory::Local,
            FsCategory::Nfs,
            FsCategory::Sshfs,
            FsCategory::Fuse,
            FsCategory::Cifs,
            FsCategory::Memory,
            FsCategory::Virtual,
            FsCategory::Unknown,
        ] {
            let json = serde_json::to_string(&category).expect("serialize");
            let deserialized: FsCategory = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, category);
        }
    }
}
