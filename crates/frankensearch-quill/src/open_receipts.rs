//! Open receipts: let a read-only open skip re-hashing a published segment
//! that an earlier open already verified, while the file is provably unchanged.
//!
//! A read-only [`crate::keeper::KeeperSnapshot`] open recomputes the whole-file
//! xxh3 of every live segment. On a multi-gigabyte index that pass dominates a
//! one-shot query: seconds of CPU and a walk over every mapped page, in every
//! process, even when nothing on disk has changed since the last open.
//!
//! A receipt records, for one segment that passed a full check (the
//! whole-file hash and every section checksum), the MANIFEST binding it was
//! checked against (segment id, file length, file xxh3) and the file's
//! identity at the time: device, inode, length, and mtime and ctime to the
//! nanosecond. An open that opts in
//! ([`crate::QuillConfig::read_open_receipts`]) trusts a receipt only when all
//! of these still match and the receipt is younger than its maximum age;
//! anything else takes the usual check. A segment is receipted only if its
//! identity did not change across the check and its last change is older than
//! the racy window (60 s, far longer than a coarse timestamp tick). Writing a
//! published segment through a shared writable mapping is outside the
//! contract here, as it already is for every mapped reader (memmap2's safety
//! contract): stores to an already dirty page do not move the timestamps, so
//! no window can promise to see them, although 60 s outlasts default
//! writeback on disk filesystems, after which the next store faults and does.
//!
//! What still runs on a receipted open: MANIFEST and schema validation, the
//! header CRC, the section-table layout, the trailer CRC, and the binding of
//! segment id, length, trailer xxh3, docid range and doc count against the
//! MANIFEST. What the receipt stands in for: the full-prefix hash and the lazy
//! per-section checksums. Replacing, truncating, extending or rewriting a
//! segment through the filesystem changes its inode, length or ctime, so such
//! a segment is fully verified again. Corruption that bypasses the filesystem
//! (media errors under an unchanged inode) is caught by the first open after
//! the receipt expires (which then verifies in full), or by any strict open
//! (writers, recovery, doctor, readers without receipts); a snapshot already
//! open does not re-verify itself.
//! A rewrite racing the open itself (after the identity check, before first
//! use) is not caught either; published segments are immutable by contract.
//! Receipt age uses the wall clock, so a clock set back can extend it.
//! On Linux, receipts are used only for segment files on known local
//! filesystems (ext2/3/4, XFS, Btrfs, F2FS, ZFS, bcachefs, tmpfs, ramfs);
//! network and FUSE filesystems, where attribute caching or synthesized inode
//! numbers weaken the identity, and anything unknown always verify. Elsewhere
//! on Unix, callers should not enable receipts on such filesystems either.
//!
//! Trust: a book vouches for bytes, so the caller must keep it where only the
//! index's owner can write, as it would the segments and MANIFEST (XXH3 is not
//! a MAC: whoever can write those can already make a strict open accept
//! anything). A book that is not a regular file owned by this user, or is
//! group- or world-writable, is ignored, and books are created mode 0600, so
//! another local user cannot vouch for bytes on a shared read-only index.
//!
//! The receipt book is one advisory file at a path the caller chooses, outside
//! the index directory, so the index directory itself stays untouched by
//! readers. It is a fixed little-endian layout: magic, record count, 88-byte
//! records, and an xxh3-64 of everything before it. A missing, unreadable,
//! oversized, corrupt, foreign or stale book only costs a full verification.
//! Readers replace it with a temp file plus rename, and only receipt-using
//! opens read it.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use xxhash_rust::xxh3::xxh3_64;

const OPEN_RECEIPTS_MAGIC: [u8; 8] = *b"FSQORCP1";
const RECEIPT_RECORD_LEN: usize = 11 * 8;
/// A receipt book larger than this is ignored rather than parsed.
const MAX_OPEN_RECEIPTS_FILE_BYTES: u64 = 1 << 20;
/// Receipts in the future by more than this are treated as foreign.
const CLOCK_SKEW_ALLOWANCE: Duration = Duration::from_secs(60);

/// How long a receipt is trusted, and how old a file's last change must be
/// before it can be receipted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenReceiptPolicy {
    /// A later open does not admit a receipt older than this (by the wall
    /// clock when the book is loaded), and verifies that segment in full.
    pub max_age: Duration,
    /// A file whose mtime or ctime is younger than this is verified but not
    /// receipted, so a rewrite inside one coarse timestamp tick cannot keep
    /// the receipted identity.
    pub racy_window: Duration,
}

impl Default for OpenReceiptPolicy {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(24 * 60 * 60),
            racy_window: Duration::from_secs(60),
        }
    }
}

/// Filesystem identity of one segment file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentFileIdentity {
    dev: u64,
    ino: u64,
    len: u64,
    mtime_s: i64,
    mtime_ns: i64,
    ctime_s: i64,
    ctime_ns: i64,
}

impl SegmentFileIdentity {
    /// Identity of the open regular file (`fstat` on its descriptor).
    #[cfg(unix)]
    pub fn of_file(file: &fs::File) -> Option<Self> {
        Self::from_metadata(&file.metadata().ok()?)
    }

    /// Receipts need a stable inode identity; other platforms always verify.
    #[cfg(not(unix))]
    pub fn of_file(_file: &fs::File) -> Option<Self> {
        None
    }

    #[cfg(unix)]
    fn from_metadata(metadata: &fs::Metadata) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;

        if !metadata.file_type().is_file() {
            return None;
        }
        Some(Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            len: metadata.len(),
            mtime_s: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
            ctime_s: metadata.ctime(),
            ctime_ns: metadata.ctime_nsec(),
        })
    }

    /// Identity of the file at `path` (tests only).
    #[cfg(all(test, unix))]
    pub fn of_path_for_test(path: &Path) -> Option<Self> {
        Self::from_metadata(&fs::metadata(path).ok()?)
    }

    fn last_change_unix_ns(&self) -> i128 {
        let mtime = i128::from(self.mtime_s) * 1_000_000_000 + i128::from(self.mtime_ns);
        let ctime = i128::from(self.ctime_s) * 1_000_000_000 + i128::from(self.ctime_ns);
        mtime.max(ctime)
    }
}

/// One receipt: a MANIFEST binding plus the identity of the file verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenReceipt {
    segment_id: u64,
    file_len: u64,
    file_xxh3: u64,
    identity: SegmentFileIdentity,
    verified_unix_s: i64,
}

/// The MANIFEST binding a receipt is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiptBinding {
    pub segment_id: u64,
    pub file_len: u64,
    pub file_xxh3: u64,
}

/// Receipts loaded for one open, plus the receipts that open will keep.
pub struct OpenReceiptBook {
    policy: OpenReceiptPolicy,
    now: SystemTime,
    loaded: Vec<OpenReceipt>,
    kept: Vec<OpenReceipt>,
}

impl OpenReceiptBook {
    /// Load the receipt book at `book`; any problem yields an empty book.
    pub fn load(book: &Path, policy: OpenReceiptPolicy) -> Self {
        Self::load_at(book, policy, SystemTime::now())
    }

    pub fn load_at(book: &Path, policy: OpenReceiptPolicy, now: SystemTime) -> Self {
        Self {
            policy,
            now,
            loaded: read_receipts(book).unwrap_or_default(),
            kept: Vec::new(),
        }
    }

    fn now_unix_ns(&self) -> i128 {
        match self.now.duration_since(UNIX_EPOCH) {
            Ok(elapsed) => i128::try_from(elapsed.as_nanos()).unwrap_or(i128::MAX),
            Err(before) => -i128::try_from(before.duration().as_nanos()).unwrap_or(i128::MAX),
        }
    }

    fn now_unix_s(&self) -> i64 {
        i64::try_from(self.now_unix_ns().div_euclid(1_000_000_000)).unwrap_or(i64::MAX)
    }

    /// Whether the file with `identity` may skip full verification for
    /// `binding`. On `true` the receipt is kept for the rewritten book.
    #[cfg(test)]
    pub fn admit(&mut self, binding: ReceiptBinding, identity: &SegmentFileIdentity) -> bool {
        self.admitted(binding, identity)
            .map(|receipt| self.keep(receipt))
            .is_some()
    }

    /// The loaded receipt that lets the file with `identity` skip full
    /// verification for `binding`, if there is one (the caller keeps it).
    pub fn admitted(
        &self,
        binding: ReceiptBinding,
        identity: &SegmentFileIdentity,
    ) -> Option<OpenReceipt> {
        let now_s = i128::from(self.now_unix_s());
        let max_age = i128::from(self.policy.max_age.as_secs());
        let skew = i128::from(CLOCK_SKEW_ALLOWANCE.as_secs());
        self.loaded
            .iter()
            .find(|receipt| {
                let age = now_s - i128::from(receipt.verified_unix_s);
                receipt.segment_id == binding.segment_id
                    && receipt.file_len == binding.file_len
                    && receipt.file_xxh3 == binding.file_xxh3
                    && receipt.identity == *identity
                    && identity.len == binding.file_len
                    && age >= -skew
                    && age <= max_age
            })
            .cloned()
    }

    /// Keep a receipt for the rewritten book.
    pub fn keep(&mut self, receipt: OpenReceipt) {
        self.kept.push(receipt);
    }

    /// Whether a file with this identity could be receipted now (its last
    /// change is outside the racy window), so callers skip the extra section
    /// pass for a file that could not be receipted anyway.
    pub fn may_record(&self, identity: &SegmentFileIdentity) -> bool {
        let racy_ns = i128::try_from(self.policy.racy_window.as_nanos()).unwrap_or(i128::MAX);
        self.now_unix_ns() - identity.last_change_unix_ns() >= racy_ns
    }

    /// Receipt a segment that just passed full verification.
    ///
    /// `before` is the identity observed before the file was mapped and
    /// `after` the identity observed once verification finished; a change in
    /// between, or a last change inside the racy window, records nothing.
    #[cfg(test)]
    pub fn record_verified(
        &mut self,
        binding: ReceiptBinding,
        before: &SegmentFileIdentity,
        after: Option<&SegmentFileIdentity>,
    ) {
        if let Some(receipt) = self.receipt_for(binding, before, after) {
            self.keep(receipt);
        }
    }

    /// The receipt a segment that just passed full verification earns, if
    /// its identity is unchanged across the check and outside the racy window.
    pub fn receipt_for(
        &self,
        binding: ReceiptBinding,
        before: &SegmentFileIdentity,
        after: Option<&SegmentFileIdentity>,
    ) -> Option<OpenReceipt> {
        if after != Some(before) || before.len != binding.file_len || !self.may_record(before) {
            return None;
        }
        Some(OpenReceipt {
            segment_id: binding.segment_id,
            file_len: binding.file_len,
            file_xxh3: binding.file_xxh3,
            identity: *before,
            verified_unix_s: self.now_unix_s(),
        })
    }

    /// Replace the receipt book at `book` with the receipts this open kept,
    /// when they differ from what was loaded. Failures are ignored: the only
    /// cost of a missing book is a full verification on the next open.
    pub fn store(self, book: &Path) {
        if self.kept == self.loaded {
            return;
        }
        let _ = write_receipts(book, &self.kept);
    }
}

impl OpenReceipt {
    fn fields(&self) -> [u64; 11] {
        let id = &self.identity;
        [
            self.segment_id,
            self.file_len,
            self.file_xxh3,
            id.dev,
            id.ino,
            id.len,
            id.mtime_s.cast_unsigned(),
            id.mtime_ns.cast_unsigned(),
            id.ctime_s.cast_unsigned(),
            id.ctime_ns.cast_unsigned(),
            self.verified_unix_s.cast_unsigned(),
        ]
    }

    fn from_fields(f: [u64; 11]) -> Self {
        Self {
            segment_id: f[0],
            file_len: f[1],
            file_xxh3: f[2],
            identity: SegmentFileIdentity {
                dev: f[3],
                ino: f[4],
                len: f[5],
                mtime_s: f[6].cast_signed(),
                mtime_ns: f[7].cast_signed(),
                ctime_s: f[8].cast_signed(),
                ctime_ns: f[9].cast_signed(),
            },
            verified_unix_s: f[10].cast_signed(),
        }
    }
}

fn encode_receipts(receipts: &[OpenReceipt]) -> Option<Vec<u8>> {
    let count = u32::try_from(receipts.len()).ok()?;
    let mut bytes = Vec::with_capacity(8 + 4 + receipts.len() * RECEIPT_RECORD_LEN + 8);
    bytes.extend_from_slice(&OPEN_RECEIPTS_MAGIC);
    bytes.extend_from_slice(&count.to_le_bytes());
    for receipt in receipts {
        for field in receipt.fields() {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
    }
    let checksum = xxh3_64(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Some(bytes)
}

fn decode_receipts(bytes: &[u8]) -> Option<Vec<OpenReceipt>> {
    let (body, checksum) = bytes.split_at_checked(bytes.len().checked_sub(8)?)?;
    if xxh3_64(body).to_le_bytes() != checksum {
        return None;
    }
    let rest = body.strip_prefix(&OPEN_RECEIPTS_MAGIC)?;
    let (count, records) = rest.split_at_checked(4)?;
    let count = usize::try_from(u32::from_le_bytes(count.try_into().ok()?)).ok()?;
    if records.len() != count.checked_mul(RECEIPT_RECORD_LEN)? {
        return None;
    }
    let (records, _) = records.as_chunks::<RECEIPT_RECORD_LEN>();
    Some(
        records
            .iter()
            .map(|record| {
                let mut fields = [0_u64; 11];
                for (field, word) in fields.iter_mut().zip(record.as_chunks::<8>().0) {
                    *field = u64::from_le_bytes(*word);
                }
                OpenReceipt::from_fields(fields)
            })
            .collect(),
    )
}

fn read_receipts(path: &Path) -> Option<Vec<OpenReceipt>> {
    use std::io::Read;

    let mut options = fs::OpenOptions::new();
    options.read(true);
    // Never block on a FIFO or follow a symlink put where the book belongs;
    // the handle's own metadata is checked below.
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        &mut options,
        i32::try_from((rustix::fs::OFlags::NONBLOCK | rustix::fs::OFlags::NOFOLLOW).bits()).ok()?,
    );
    let file = options.open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.file_type().is_file()
        || metadata.len() > MAX_OPEN_RECEIPTS_FILE_BYTES
        || !book_is_private(&metadata)
    {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_OPEN_RECEIPTS_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_OPEN_RECEIPTS_FILE_BYTES {
        return None;
    }
    decode_receipts(&bytes)
}

/// A book counts only if this user owns it and nobody else can write it.
#[cfg(unix)]
fn book_is_private(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    metadata.uid() == rustix::process::geteuid().as_raw() && metadata.mode() & 0o022 == 0
}

#[cfg(not(unix))]
fn book_is_private(_metadata: &fs::Metadata) -> bool {
    false
}

fn write_receipts(book: &Path, receipts: &[OpenReceipt]) -> std::io::Result<()> {
    let body = encode_receipts(receipts)
        .filter(|body| {
            u64::try_from(body.len()).is_ok_and(|len| len <= MAX_OPEN_RECEIPTS_FILE_BYTES)
        })
        .ok_or_else(|| std::io::Error::other("too many open receipts"))?;
    static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let sequence = TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut temp = book.as_os_str().to_owned();
    temp.push(format!(".tmp-{}-{nonce}-{sequence}", std::process::id()));
    let temp = std::path::PathBuf::from(temp);
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    // Only a temp file this call created is ever removed.
    let mut file = options.open(&temp)?;
    let result = file.write_all(&body).and_then(|()| fs::rename(&temp, book));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Whether the filesystem holding this segment file gives the identity
/// guarantees a receipt relies on: only known local filesystems do. Network
/// and FUSE filesystems (attribute caching, synthesized inode numbers) and
/// anything unknown always verify.
#[cfg(target_os = "linux")]
pub fn receipts_supported_on(file: &fs::File) -> bool {
    rustix::fs::fstatfs(file).is_ok_and(|stat| filesystem_keeps_identity(stat.f_type))
}

#[cfg(target_os = "linux")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a filesystem magic is the low 32 bits of f_type, whatever its C type"
)]
fn filesystem_keeps_identity<T: TryInto<i64>>(f_type: T) -> bool {
    // An allowlist, so an unknown filesystem fails safe (full verification).
    const LOCAL_IDENTITY_FILESYSTEMS: [u32; 8] = [
        0xef53,      // ext2, ext3, ext4
        0x5846_5342, // XFS
        0x9123_683e, // Btrfs
        0xf2f5_2010, // F2FS
        0x2fc1_2fc1, // ZFS
        0xca45_1a4e, // bcachefs
        0x0102_1994, // tmpfs
        0x8584_58f6, // ramfs
    ];
    f_type
        .try_into()
        .is_ok_and(|kind: i64| LOCAL_IDENTITY_FILESYSTEMS.contains(&(kind as u32)))
}

#[cfg(not(target_os = "linux"))]
pub fn receipts_supported_on(_file: &fs::File) -> bool {
    cfg!(unix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(ino: u64, len: u64, change_s: i64) -> SegmentFileIdentity {
        SegmentFileIdentity {
            dev: 7,
            ino,
            len,
            mtime_s: change_s,
            mtime_ns: 0,
            ctime_s: change_s,
            ctime_ns: 0,
        }
    }

    fn binding(len: u64) -> ReceiptBinding {
        ReceiptBinding {
            segment_id: 0xabc,
            file_len: len,
            file_xxh3: 0x1234,
        }
    }

    fn at(unix_s: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(unix_s)
    }

    fn book_path(directory: &tempfile::TempDir) -> std::path::PathBuf {
        directory.path().join("open-receipts")
    }

    fn policy() -> OpenReceiptPolicy {
        OpenReceiptPolicy {
            max_age: Duration::from_secs(3_600),
            racy_window: Duration::from_secs(2),
        }
    }

    #[test]
    fn receipt_round_trips_and_admits_only_the_exact_file_and_binding() {
        let directory = tempfile::tempdir().expect("tempdir");
        let file = identity(11, 100, 1_000);
        let mut book = OpenReceiptBook::load_at(&book_path(&directory), policy(), at(1_010));
        assert!(
            !book.admit(binding(100), &file),
            "an empty book admits nothing"
        );
        book.record_verified(binding(100), &file, Some(&file));
        book.store(&book_path(&directory));

        let mut book = OpenReceiptBook::load_at(&book_path(&directory), policy(), at(1_020));
        assert!(book.admit(binding(100), &file));
        for (label, other) in [
            ("inode", identity(12, 100, 1_000)),
            ("length", identity(11, 101, 1_000)),
            (
                "ctime",
                SegmentFileIdentity {
                    ctime_ns: 1,
                    ..file
                },
            ),
            (
                "mtime",
                SegmentFileIdentity {
                    mtime_s: 999,
                    ..file
                },
            ),
            ("device", SegmentFileIdentity { dev: 8, ..file }),
        ] {
            assert!(
                !book.admit(binding(100), &other),
                "changed {label} must not be admitted"
            );
        }
        let foreign = ReceiptBinding {
            file_xxh3: 0x9999,
            ..binding(100)
        };
        assert!(
            !book.admit(foreign, &file),
            "a different MANIFEST witness must not be admitted"
        );
        let other_segment = ReceiptBinding {
            segment_id: 0xdef,
            ..binding(100)
        };
        assert!(!book.admit(other_segment, &file));
    }

    #[test]
    fn receipts_expire_and_future_receipts_are_ignored() {
        let directory = tempfile::tempdir().expect("tempdir");
        let file = identity(11, 100, 1_000);
        let mut book = OpenReceiptBook::load_at(&book_path(&directory), policy(), at(10_000));
        book.record_verified(binding(100), &file, Some(&file));
        book.store(&book_path(&directory));

        let mut fresh =
            OpenReceiptBook::load_at(&book_path(&directory), policy(), at(10_000 + 3_600));
        assert!(fresh.admit(binding(100), &file));
        let mut stale =
            OpenReceiptBook::load_at(&book_path(&directory), policy(), at(10_000 + 3_601));
        assert!(
            !stale.admit(binding(100), &file),
            "an expired receipt forces full verification"
        );
        let mut earlier =
            OpenReceiptBook::load_at(&book_path(&directory), policy(), at(10_000 - 61));
        assert!(
            !earlier.admit(binding(100), &file),
            "a receipt from the future is not trusted"
        );
    }

    #[test]
    fn changed_or_racy_files_are_not_receipted() {
        let directory = tempfile::tempdir().expect("tempdir");
        let file = identity(11, 100, 1_000);
        let mut book = OpenReceiptBook::load_at(&book_path(&directory), policy(), at(1_001));
        // (the test policy uses a 2 s racy window; production uses 60 s)
        book.record_verified(binding(100), &file, Some(&file));
        assert!(
            book.kept.is_empty(),
            "a change inside the racy window must not be receipted"
        );

        let mut book = OpenReceiptBook::load_at(&book_path(&directory), policy(), at(1_010));
        book.record_verified(binding(100), &file, Some(&identity(11, 100, 1_005)));
        book.record_verified(binding(100), &file, None);
        book.record_verified(binding(101), &file, Some(&file));
        assert!(
            book.kept.is_empty(),
            "a file that changed during verification is not receipted"
        );
        book.store(&book_path(&directory));
        assert!(
            !book_path(&directory).exists(),
            "an unchanged empty book writes nothing"
        );
    }

    #[test]
    fn corrupt_foreign_or_oversized_books_are_ignored() -> std::io::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = book_path(&directory);
        let file = identity(11, 100, 1_000);
        let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_010));
        book.record_verified(binding(100), &file, Some(&file));
        book.store(&path);
        let good = fs::read(&path)?;
        assert_eq!(good.len(), 8 + 4 + RECEIPT_RECORD_LEN + 8);
        assert!(OpenReceiptBook::load_at(&path, policy(), at(1_020)).admit(binding(100), &file));

        // One flipped bit anywhere (here: in the inode field) breaks the checksum.
        for offset in [0, 8, 12 + 4 * 8, good.len() - 1] {
            let mut edited = good.clone();
            edited[offset] ^= 0x01;
            fs::write(&path, &edited)?;
            let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_020));
            assert!(
                !book.admit(binding(100), &file),
                "flipped byte {offset} admits nothing"
            );
        }
        // A well-formed book with a wrong magic is foreign.
        let mut foreign = good[..good.len() - 8].to_vec();
        foreign[7] = b'9';
        let checksum = xxh3_64(&foreign);
        foreign.extend_from_slice(&checksum.to_le_bytes());
        fs::write(&path, &foreign)?;
        let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_020));
        assert!(
            !book.admit(binding(100), &file),
            "a foreign format admits nothing"
        );

        for garbage in [&b""[..], b"FSQORCP1", &good[..good.len() - 1]] {
            fs::write(&path, garbage)?;
            let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_020));
            assert!(
                !book.admit(binding(100), &file),
                "truncated book admits nothing"
            );
        }
        let oversized = usize::try_from(MAX_OPEN_RECEIPTS_FILE_BYTES + 1).unwrap_or(usize::MAX);
        fs::write(&path, vec![0_u8; oversized])?;
        let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_020));
        assert!(!book.admit(binding(100), &file));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_book_others_can_write_is_ignored() -> std::io::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let path = book_path(&directory);
        let file = identity(11, 100, 1_000);
        let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_010));
        book.record_verified(binding(100), &file, Some(&file));
        book.store(&path);
        assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o600);
        assert!(OpenReceiptBook::load_at(&path, policy(), at(1_020)).admit(binding(100), &file));
        for mode in [0o620, 0o602, 0o666] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))?;
            let mut book = OpenReceiptBook::load_at(&path, policy(), at(1_020));
            assert!(
                !book.admit(binding(100), &file),
                "mode {mode:o} must not be trusted"
            );
        }
        Ok(())
    }
}
