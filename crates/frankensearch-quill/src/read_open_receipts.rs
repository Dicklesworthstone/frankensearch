//! Opt-in reuse of a successful file-prefix check on immutable local files.
//!
//! A receipt is not a fresh measurement of current bytes. Metadata-preserving
//! media faults and writes through an already-dirty writable mapping remain
//! outside this policy. Strict opens never consult receipts. Section checks
//! remain active even on a hit. SHA-256 detects damaged receipts, not forgery
//! by the index owner, who can already rewrite the MANIFEST witnesses.
//!
//! The caller supplies an existing, owner-private directory outside the index.
//! Reads and atomic writes use its retained descriptor. Unusable storage,
//! malformed proofs, foreign producers and unsupported filesystems fail back
//! to full verification. Hits never renew the original verification time.
//! Expiry uses `CLOCK_BOOTTIME`, not the adjustable wall clock. Boot and namespace
//! identities prevent carrying a proof into a different clock or mount domain.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;
#[cfg(target_os = "linux")]
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"FSQORC03";
const MAX_BYTES: usize = 1 << 20;
const RECORD_BYTES: usize = 12 * 8;
const HEADER_BYTES: usize = 8 + 32 + 4;
const CHECKSUM_BYTES: usize = 32;
#[cfg(target_os = "linux")]
const BOOK: &str = "read-open-receipts-v3";
const NANOS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Policy {
    pub max_age: Duration,
    pub minimum_file_age: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(3600),
            minimum_file_age: Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Clock {
    unix_ns: i128,
    boot_ns: u64,
}

impl Clock {
    #[cfg(target_os = "linux")]
    fn now() -> Option<Self> {
        let unix_ns = i128::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()?
                .as_nanos(),
        )
        .ok()?;
        let boot = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
        let boot_ns = u64::try_from(boot.tv_sec)
            .ok()?
            .checked_mul(NANOS_PER_SECOND)?
            .checked_add(u64::try_from(boot.tv_nsec).ok()?)?;
        Some(Self { unix_ns, boot_ns })
    }

    #[cfg(not(target_os = "linux"))]
    fn now() -> Option<Self> {
        None
    }
}

/// The schema and MANIFEST witness, independent of the path naming the file.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Binding {
    pub schema_id: u64,
    pub segment_id: u64,
    pub file_len: u64,
    pub file_xxh3: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Identity {
    dev: u64,
    ino: u64,
    len: u64,
    mtime_s: i64,
    mtime_ns: i64,
    ctime_s: i64,
    ctime_ns: i64,
}

impl Identity {
    /// fstat the actual descriptor supplied by `open_published_checked`.
    #[cfg(target_os = "linux")]
    pub(crate) fn of_file(file: &File) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().ok()?;
        if !metadata.is_file() {
            return None;
        }
        let identity = Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            len: metadata.len(),
            mtime_s: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
            ctime_s: metadata.ctime(),
            ctime_ns: metadata.ctime_nsec(),
        };
        identity.valid().then_some(identity)
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) fn of_file(_file: &File) -> Option<Self> {
        None
    }

    fn valid(self) -> bool {
        (0..1_000_000_000).contains(&self.mtime_ns) && (0..1_000_000_000).contains(&self.ctime_ns)
    }

    fn old_enough(self, now_ns: i128, minimum: Duration) -> bool {
        let mtime =
            i128::from(self.mtime_s) * i128::from(NANOS_PER_SECOND) + i128::from(self.mtime_ns);
        let ctime =
            i128::from(self.ctime_s) * i128::from(NANOS_PER_SECOND) + i128::from(self.ctime_ns);
        i128::try_from(minimum.as_nanos()).is_ok_and(|minimum| now_ns - mtime.max(ctime) >= minimum)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Receipt {
    binding: Binding,
    identity: Identity,
    verified_boot_ns: u64,
}

pub struct ReceiptBook {
    directory: Option<File>,
    producer: [u8; 32],
    loaded: BTreeMap<Binding, Receipt>,
    kept: BTreeMap<Binding, Receipt>,
    policy: Policy,
    clock: Option<Clock>,
}

impl ReceiptBook {
    pub(crate) fn load(directory: &Path, policy: Policy) -> Self {
        Self::load_at(directory, policy, Clock::now())
    }

    fn load_at(directory: &Path, policy: Policy, clock: Option<Clock>) -> Self {
        let producer = clock.and_then(|_| producer_binding());
        let directory = producer.and_then(|_| open_private_directory(directory).ok());
        let producer = producer.unwrap_or([0; 32]);
        let loaded = directory
            .as_ref()
            .and_then(|dir| read_book(dir).ok())
            .and_then(|bytes| decode(&bytes, producer))
            .unwrap_or_default();
        Self {
            directory,
            producer,
            loaded,
            kept: BTreeMap::new(),
            policy,
            clock,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.directory.is_some() && self.clock.is_some()
    }

    pub(crate) fn admitted(&self, binding: Binding, identity: Identity) -> Option<Receipt> {
        if !self.enabled() || identity.len != binding.file_len {
            return None;
        }
        let clock = self.clock?;
        let receipt = self.loaded.get(&binding)?;
        let age = clock.boot_ns.checked_sub(receipt.verified_boot_ns)?;
        let max_age = u64::try_from(self.policy.max_age.as_nanos()).ok()?;
        (receipt.identity == identity
            && age <= max_age
            && identity.old_enough(clock.unix_ns, self.policy.minimum_file_age))
        .then(|| receipt.clone())
    }

    /// Call only after full prefix verification and before/after fstat agreement.
    pub(crate) fn verified(
        &self,
        binding: Binding,
        before: Identity,
        after: Option<Identity>,
    ) -> Option<Receipt> {
        if !self.enabled() || Some(before) != after || before.len != binding.file_len {
            return None;
        }
        let clock = self.clock?;
        before
            .old_enough(clock.unix_ns, self.policy.minimum_file_age)
            .then_some(Receipt {
                binding,
                identity: before,
                verified_boot_ns: clock.boot_ns,
            })
    }

    pub(crate) fn keep(&mut self, receipt: Receipt) {
        self.kept.insert(receipt.binding, receipt);
    }

    /// Publish only after the entire snapshot and its retained descriptors pass.
    pub(crate) fn persist(self) {
        if self.kept == self.loaded {
            return;
        }
        let Some(directory) = self.directory else {
            return;
        };
        let Some(bytes) = encode(&self.kept, self.producer) else {
            return;
        };
        // Concurrent callers may evict each other's cache entries, not add trust.
        let _ = write_book(&directory, &bytes);
    }
}

/// Unknown, network, FUSE and overlay filesystems retain strict verification.
#[cfg(target_os = "linux")]
pub fn supported(file: &File) -> bool {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    rustix::fs::fstatfs(file).is_ok_and(|stat| {
        matches!(
            stat.f_type as u32,
            0xef53 | 0x5846_5342 | 0x9123_683e | 0xf2f5_2010 | 0x0102_1994
        )
    })
}

#[cfg(not(target_os = "linux"))]
pub fn supported(_file: &File) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn producer_binding() -> Option<[u8; 32]> {
    use std::os::unix::fs::MetadataExt;
    let mut bytes = Vec::new();
    File::open("/proc/sys/kernel/random/boot_id")
        .ok()?
        .take(65)
        .read_to_end(&mut bytes)
        .ok()?;
    let boot = std::str::from_utf8(&bytes).ok()?.trim();
    if boot.len() != 36
        || !boot.bytes().enumerate().all(|(offset, byte)| {
            if [8, 13, 18, 23].contains(&offset) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
    {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(b"quill-read-open-receipts-v3\0");
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    hash.update(b"\0");
    hash.update(boot.as_bytes());
    hash.update(crate::keeper::CURRENT_ENGINE_VERSION.to_le_bytes());
    hash.update(crate::segment::FSLX_FORMAT_VERSION.to_le_bytes());
    for namespace in ["/proc/self/ns/mnt", "/proc/self/ns/time"] {
        let metadata = std::fs::metadata(namespace).ok()?;
        hash.update(metadata.dev().to_le_bytes());
        hash.update(metadata.ino().to_le_bytes());
    }
    Some(hash.finalize().into())
}

#[cfg(not(target_os = "linux"))]
fn producer_binding() -> Option<[u8; 32]> {
    None
}

fn encode(receipts: &BTreeMap<Binding, Receipt>, producer: [u8; 32]) -> Option<Vec<u8>> {
    let size = receipts
        .len()
        .checked_mul(RECORD_BYTES)?
        .checked_add(HEADER_BYTES + CHECKSUM_BYTES)?;
    if size > MAX_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&producer);
    bytes.extend_from_slice(&u32::try_from(receipts.len()).ok()?.to_le_bytes());
    for receipt in receipts.values() {
        let b = receipt.binding;
        let i = receipt.identity;
        for value in [
            b.schema_id,
            b.segment_id,
            b.file_len,
            b.file_xxh3,
            i.dev,
            i.ino,
            i.len,
            u64::from_le_bytes(i.mtime_s.to_le_bytes()),
            u64::from_le_bytes(i.mtime_ns.to_le_bytes()),
            u64::from_le_bytes(i.ctime_s.to_le_bytes()),
            u64::from_le_bytes(i.ctime_ns.to_le_bytes()),
            receipt.verified_boot_ns,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    Some(bytes)
}

fn decode(bytes: &[u8], producer: [u8; 32]) -> Option<BTreeMap<Binding, Receipt>> {
    if !(HEADER_BYTES + CHECKSUM_BYTES..=MAX_BYTES).contains(&bytes.len()) {
        return None;
    }
    let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    if &Sha256::digest(body)[..] != checksum || &body[..8] != MAGIC || body[8..40] != producer {
        return None;
    }
    let count = usize::try_from(u32::from_le_bytes(body[40..44].try_into().ok()?)).ok()?;
    if count.checked_mul(RECORD_BYTES)?.checked_add(HEADER_BYTES)? != body.len() {
        return None;
    }
    let mut receipts = BTreeMap::new();
    // The length check above leaves no partial record or word.
    let (records, _) = body[HEADER_BYTES..].as_chunks::<RECORD_BYTES>();
    for record in records {
        let mut f = [0_u64; 12];
        let (words, _) = record.as_chunks::<8>();
        for (out, word) in f.iter_mut().zip(words) {
            *out = u64::from_le_bytes(*word);
        }
        let binding = Binding {
            schema_id: f[0],
            segment_id: f[1],
            file_len: f[2],
            file_xxh3: f[3],
        };
        let identity = Identity {
            dev: f[4],
            ino: f[5],
            len: f[6],
            mtime_s: i64::from_le_bytes(f[7].to_le_bytes()),
            mtime_ns: i64::from_le_bytes(f[8].to_le_bytes()),
            ctime_s: i64::from_le_bytes(f[9].to_le_bytes()),
            ctime_ns: i64::from_le_bytes(f[10].to_le_bytes()),
        };
        if !identity.valid() || identity.len != binding.file_len {
            return None;
        }
        let receipt = Receipt {
            binding,
            identity,
            verified_boot_ns: f[11],
        };
        if receipts.insert(binding, receipt).is_some() {
            return None;
        }
    }
    Some(receipts)
}

#[cfg(target_os = "linux")]
fn private(metadata: &std::fs::Metadata, directory: bool) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.uid() == rustix::process::geteuid().as_raw()
        && if directory {
            metadata.is_dir() && metadata.mode() & 0o777 == 0o700
        } else {
            metadata.is_file() && metadata.mode() & 0o777 == 0o600 && metadata.nlink() == 1
        }
}

#[cfg(target_os = "linux")]
fn open_private_directory(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let flags =
        rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC;
    let directory = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(i32::try_from(flags.bits()).map_err(io::Error::other)?)
        .open(path)?;
    if !private(&directory.metadata()?, true) {
        return Err(io::Error::other(
            "receipt directory must be owner-controlled and mode 0700",
        ));
    }
    Ok(directory)
}

#[cfg(not(target_os = "linux"))]
fn open_private_directory(_path: &Path) -> io::Result<File> {
    Err(io::Error::other(
        "receipt reuse requires supported Linux storage",
    ))
}

#[cfg(target_os = "linux")]
fn read_book(directory: &File) -> io::Result<Vec<u8>> {
    use rustix::fs::{Mode, OFlags, openat};
    if !private(&directory.metadata()?, true) {
        return Err(io::Error::other("receipt directory permissions changed"));
    }
    let file = File::from(openat(
        directory,
        BOOK,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )?);
    let before = Identity::of_file(&file);
    let metadata = file.metadata()?;
    if !private(&metadata, false) || metadata.len() > u64::try_from(MAX_BYTES).unwrap_or(u64::MAX) {
        return Err(io::Error::other("untrusted or oversized receipt book"));
    }
    let mut bytes = Vec::new();
    (&file)
        .take(u64::try_from(MAX_BYTES + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_BYTES
        || before.is_none()
        || before != Identity::of_file(&file)
        || !private(&directory.metadata()?, true)
    {
        return Err(io::Error::other("receipt state changed during read"));
    }
    Ok(bytes)
}

#[cfg(not(target_os = "linux"))]
fn read_book(_directory: &File) -> io::Result<Vec<u8>> {
    Err(io::Error::other("unsupported"))
}

#[cfg(target_os = "linux")]
fn write_book(directory: &File, bytes: &[u8]) -> io::Result<()> {
    use rustix::fs::{AtFlags, Mode, OFlags, openat, renameat, unlinkat};
    use std::sync::atomic::{AtomicU64, Ordering};
    if !private(&directory.metadata()?, true) {
        return Err(io::Error::other("receipt directory permissions changed"));
    }
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let nonce = Clock::now().map_or(0, |clock| clock.boot_ns);
    let temp = format!(
        ".read-open-{}-{nonce}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let mut file = File::from(openat(
        directory,
        temp.as_str(),
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )?);
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        renameat(directory, temp.as_str(), directory, BOOK)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = unlinkat(directory, temp.as_str(), AtFlags::empty());
    }
    result
}

#[cfg(not(target_os = "linux"))]
fn write_book(_directory: &File, _bytes: &[u8]) -> io::Result<()> {
    Err(io::Error::other("unsupported"))
}

// Explicit, because tests/read_open_receipts.rs includes this file by path,
// where a plain `mod tests;` would resolve to tests/../src/tests.rs.
#[cfg(test)]
#[path = "read_open_receipts/tests.rs"]
mod tests;
