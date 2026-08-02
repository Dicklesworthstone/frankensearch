//! Write-ahead log for incremental FSVI index updates.
//!
//! New vectors are appended to a `.fsvi.wal` sidecar file in batches.
//! Each batch is CRC32-protected so partial writes from crashes are
//! detected and discarded on reload.
//!
//! # File Layout
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │ WAL Header (20 bytes)               │
//! │   magic: b"FWAL" (4)               │
//! │   version: u16 LE                  │
//! │   dimension: u32 LE                │
//! │   quantization: u8                 │
//! │   reserved: [u8; 5]               │
//! │   header_crc32: u32 LE             │
//! ├─────────────────────────────────────┤
//! │ Batch 0                             │
//! │   batch_magic: b"FWB1" (4)         │
//! │   entry_count: u32 LE              │
//! │   entries...                        │
//! │   batch_crc32: u32 LE              │
//! ├─────────────────────────────────────┤
//! │ Batch 1 ...                         │
//! └─────────────────────────────────────┘
//! ```

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;
use frankensearch_core::{
    SearchError, SearchResult,
    generation::{
        ArtifactGenerationIdentityV1, FrozenEmbeddingIdentityBundleV1, QuantizationFormat,
    },
};
use half::f16;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::{
    FsviAdmissionError, FsviSnapshotRejectionReason, Quantization, open_readonly_noatime_nofollow,
    snapshot_parent_or_current, snapshot_rejected, stable_file_identity,
    validate_single_link_regular_file,
};

// ─── Constants ──────────────────────────────────────────────────────────────

const WAL_MAGIC: [u8; 4] = *b"FWAL";
const WAL_VERSION: u16 = 1;
const BATCH_MAGIC: [u8; 4] = *b"FWB1";
/// Minimum WAL file size (header only).
const WAL_HEADER_SIZE: usize = 20;

/// Version of the WAL header that binds an exact FSVI v2 generation and
/// embedding identity.
pub const IDENTITY_BOUND_WAL_VERSION: u16 = 2;
/// Fixed byte length of an identity-bound WAL v2 header.
pub const IDENTITY_BOUND_WAL_HEADER_SIZE: usize = 208;

const IDENTITY_BOUND_WAL_HEADER_SIZE_U16: u16 = 208;
const SHA256_FINGERPRINT_BYTES: usize = 32;
const IDENTITY_BUNDLE_FINGERPRINT_OFFSET: usize = 44;
const SPACE_FINGERPRINT_OFFSET: usize =
    IDENTITY_BUNDLE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const PRODUCER_FINGERPRINT_OFFSET: usize = SPACE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const INPUT_FINGERPRINT_OFFSET: usize = PRODUCER_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const STORAGE_FINGERPRINT_OFFSET: usize = INPUT_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;
const IDENTITY_BOUND_WAL_CRC_OFFSET: usize = STORAGE_FINGERPRINT_OFFSET + SHA256_FINGERPRINT_BYTES;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for WAL-based incremental updates.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum WAL entries before compaction is recommended.
    pub compaction_threshold: usize,
    /// Compaction threshold as fraction of main index size.
    pub compaction_ratio: f64,
    /// Whether to fsync after each batch write.
    pub fsync_on_write: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: 1000,
            compaction_ratio: 0.10,
            fsync_on_write: true,
        }
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// A single WAL entry (in-memory representation).
#[derive(Debug, Clone)]
pub(crate) struct WalEntry {
    pub doc_id: String,
    /// Stored for future dedup/lookup optimizations during compaction.
    #[allow(dead_code)]
    pub doc_id_hash: u64,
    pub embedding: Vec<f32>,
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Records in the main index before compaction.
    pub main_records_before: usize,
    /// Records from the WAL merged in.
    pub wal_records: usize,
    /// Total records in the compacted index.
    pub total_records_after: usize,
    /// Time taken in milliseconds.
    pub elapsed_ms: f64,
}

/// Strict, side-effect-free classification of a WAL pathname.
///
/// Unlike crash-recovery loading, this diagnostic never truncates, removes,
/// repairs, or silently discards a corrupt suffix. Every present byte must form
/// one complete valid header/batch sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrictWalInspection {
    /// No directory entry existed for the WAL path during the stable snapshot.
    Absent,
    /// A zero-length regular file existed. This is staging state only and is
    /// still forbidden beside a published FSVI v2 generation.
    Empty {
        /// SHA-256 of the empty byte image.
        whole_image_sha256: [u8; 32],
    },
    /// A complete legacy WAL v1 byte image.
    LegacyV1(StrictWalImage),
    /// A complete identity-bound WAL v2 byte image.
    IdentityBoundV2 {
        /// Exact validated identity header.
        header: WalIdentityHeaderV2,
        /// Redacted byte/batch witness.
        image: StrictWalImage,
    },
}

/// Redacted strict WAL byte/batch witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictWalImage {
    /// Exact byte length.
    pub byte_len: u64,
    /// SHA-256 of every WAL byte.
    pub whole_image_sha256: [u8; 32],
    /// Stored vector dimension.
    pub dimension: u32,
    /// Stored quantization.
    pub quantization: Quantization,
    /// Number of complete batches.
    pub batch_count: u64,
    /// Total entries across complete batches.
    pub entry_count: u64,
}

/// Parsed, validated identity header for a version-2 WAL.
///
/// The header duplicates the storage dimension and quantization so a reader
/// can reject incompatible payload bytes before parsing batches. Component
/// fingerprints are retained separately from the complete bundle fingerprint
/// to make an exact mismatch diagnosable without trusting display labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalIdentityHeaderV2 {
    generation: ArtifactGenerationIdentityV1,
    dimension: u32,
    quantization: Quantization,
    identity_bundle_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    space_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    producer_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    input_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
    storage_fingerprint: [u8; SHA256_FINGERPRINT_BYTES],
}

impl WalIdentityHeaderV2 {
    /// Decode and structurally validate the fixed header prefix.
    ///
    /// Additional WAL batch bytes after the fixed header are intentionally
    /// ignored. Exact compatibility with an expected artifact is established
    /// by [`WalIdentityBindingV2::validate_header`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] for a truncated header, unknown
    /// version, non-zero reserved byte, invalid generation, unsupported
    /// quantization, zero fingerprint, or CRC mismatch.
    pub fn decode(data: &[u8], path: &Path) -> SearchResult<Self> {
        if data.len() < IDENTITY_BOUND_WAL_HEADER_SIZE {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL header truncated: expected at least \
                     {IDENTITY_BOUND_WAL_HEADER_SIZE} bytes, got {}",
                    data.len()
                ),
            ));
        }
        if data[..4] != WAL_MAGIC {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL has bad magic bytes",
            ));
        }

        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != IDENTITY_BOUND_WAL_VERSION {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL version mismatch: expected \
                     {IDENTITY_BOUND_WAL_VERSION}, got {version}"
                ),
            ));
        }
        let header_size = u16::from_le_bytes([data[6], data[7]]);
        if header_size != IDENTITY_BOUND_WAL_HEADER_SIZE_U16 {
            return Err(wal_corrupted(
                path,
                format!(
                    "identity-bound WAL header size mismatch: expected \
                     {IDENTITY_BOUND_WAL_HEADER_SIZE_U16}, got {header_size}"
                ),
            ));
        }
        if data[13..16] != [0; 3] || data[18..20] != [0; 2] {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL reserved bytes must be zero",
            ));
        }

        let stored_crc = u32::from_le_bytes([
            data[IDENTITY_BOUND_WAL_CRC_OFFSET],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 1],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 2],
            data[IDENTITY_BOUND_WAL_CRC_OFFSET + 3],
        ]);
        let computed_crc = crc32_of(&data[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
        if stored_crc != computed_crc {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL header CRC mismatch",
            ));
        }

        let dimension = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if dimension == 0 {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL dimension must be greater than zero",
            ));
        }
        let quantization = Quantization::from_wire(data[12], path)?;
        let generation = ArtifactGenerationIdentityV1 {
            schema_version: u16::from_le_bytes([data[16], data[17]]),
            sequence: u64::from_le_bytes([
                data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27],
            ]),
            nonce: copy_array::<16>(&data[28..44], path, "artifact generation nonce")?,
        };
        if let Err(error) = generation.validate() {
            return Err(wal_corrupted(
                path,
                format!("identity-bound WAL artifact generation is invalid: {error}"),
            ));
        }

        let header = Self {
            generation,
            dimension,
            quantization,
            identity_bundle_fingerprint: copy_array(
                &data[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET],
                path,
                "embedding identity bundle fingerprint",
            )?,
            space_fingerprint: copy_array(
                &data[SPACE_FINGERPRINT_OFFSET..PRODUCER_FINGERPRINT_OFFSET],
                path,
                "embedding space fingerprint",
            )?,
            producer_fingerprint: copy_array(
                &data[PRODUCER_FINGERPRINT_OFFSET..INPUT_FINGERPRINT_OFFSET],
                path,
                "embedding producer fingerprint",
            )?,
            input_fingerprint: copy_array(
                &data[INPUT_FINGERPRINT_OFFSET..STORAGE_FINGERPRINT_OFFSET],
                path,
                "embedding input fingerprint",
            )?,
            storage_fingerprint: copy_array(
                &data[STORAGE_FINGERPRINT_OFFSET..IDENTITY_BOUND_WAL_CRC_OFFSET],
                path,
                "vector storage fingerprint",
            )?,
        };
        header.validate_fingerprints(path)?;
        Ok(header)
    }

    /// Encode this validated header to its fixed-width representation.
    #[must_use]
    pub fn encode(&self) -> [u8; IDENTITY_BOUND_WAL_HEADER_SIZE] {
        let mut bytes = [0; IDENTITY_BOUND_WAL_HEADER_SIZE];
        bytes[..4].copy_from_slice(&WAL_MAGIC);
        bytes[4..6].copy_from_slice(&IDENTITY_BOUND_WAL_VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&IDENTITY_BOUND_WAL_HEADER_SIZE_U16.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.dimension.to_le_bytes());
        bytes[12] = quantization_to_wire(self.quantization);
        bytes[16..18].copy_from_slice(&self.generation.schema_version.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.generation.sequence.to_le_bytes());
        bytes[28..44].copy_from_slice(&self.generation.nonce);
        bytes[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.identity_bundle_fingerprint);
        bytes[SPACE_FINGERPRINT_OFFSET..PRODUCER_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.space_fingerprint);
        bytes[PRODUCER_FINGERPRINT_OFFSET..INPUT_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.producer_fingerprint);
        bytes[INPUT_FINGERPRINT_OFFSET..STORAGE_FINGERPRINT_OFFSET]
            .copy_from_slice(&self.input_fingerprint);
        bytes[STORAGE_FINGERPRINT_OFFSET..IDENTITY_BOUND_WAL_CRC_OFFSET]
            .copy_from_slice(&self.storage_fingerprint);
        let crc = crc32_of(&bytes[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
        bytes[IDENTITY_BOUND_WAL_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        bytes
    }

    /// Exact immutable generation bound to this WAL.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.generation
    }

    /// Stored vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Stored vector quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// SHA-256 of the complete frozen embedding identity bundle.
    #[must_use]
    pub const fn identity_bundle_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.identity_bundle_fingerprint
    }

    /// SHA-256 of the mathematical embedding-space identity.
    #[must_use]
    pub const fn space_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.space_fingerprint
    }

    /// SHA-256 of the embedding-producer attestation.
    #[must_use]
    pub const fn producer_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.producer_fingerprint
    }

    /// SHA-256 of the outer embedding-input contract.
    #[must_use]
    pub const fn input_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.input_fingerprint
    }

    /// SHA-256 of the physical vector-storage identity.
    #[must_use]
    pub const fn storage_fingerprint(&self) -> &[u8; SHA256_FINGERPRINT_BYTES] {
        &self.storage_fingerprint
    }

    fn validate_fingerprints(&self, path: &Path) -> SearchResult<()> {
        for (field, fingerprint) in [
            (
                "embedding identity bundle fingerprint",
                &self.identity_bundle_fingerprint,
            ),
            ("embedding space fingerprint", &self.space_fingerprint),
            ("embedding producer fingerprint", &self.producer_fingerprint),
            ("embedding input fingerprint", &self.input_fingerprint),
            ("vector storage fingerprint", &self.storage_fingerprint),
        ] {
            if *fingerprint == [0; SHA256_FINGERPRINT_BYTES] {
                return Err(wal_corrupted(
                    path,
                    format!("identity-bound WAL {field} must not be all zero"),
                ));
            }
        }
        Ok(())
    }
}

/// Exact generation and embedding-identity contract expected by an FSVI v2
/// WAL reader or writer.
///
/// Construction validates the full frozen identity bundle and requires the
/// physical storage contract to name `fsvi-v2`. A binding can emit the
/// canonical fixed header and reject any persisted component drift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalIdentityBindingV2 {
    expected: WalIdentityHeaderV2,
}

impl WalIdentityBindingV2 {
    /// Build an exact identity binding for one FSVI v2 artifact generation.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the generation or frozen
    /// identity is invalid, the storage format is not exactly `fsvi-v2`, or
    /// the quantization cannot be represented by the FSVI slab format.
    pub fn new(
        generation: ArtifactGenerationIdentityV1,
        identity: &FrozenEmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        generation.validate()?;
        identity.validate()?;
        if identity.identity.storage.format != "fsvi-v2" {
            return Err(wal_identity_config_error(
                "storage.format",
                "must be exactly fsvi-v2 for an identity-bound WAL",
            ));
        }
        let quantization = match identity.identity.storage.quantization {
            QuantizationFormat::F32 => Quantization::F32,
            QuantizationFormat::F16 => Quantization::F16,
            QuantizationFormat::Int8 | QuantizationFormat::Int4 => {
                return Err(wal_identity_config_error(
                    "storage.quantization",
                    "is not supported by the FSVI v2 slab",
                ));
            }
        };
        let expected = WalIdentityHeaderV2 {
            generation,
            dimension: identity.identity.storage.dimension,
            quantization,
            identity_bundle_fingerprint: decode_sha256_fingerprint(
                "frozen_bundle.fingerprint",
                &identity.fingerprint,
            )?,
            space_fingerprint: decode_sha256_fingerprint(
                "space.fingerprint",
                &identity.identity.space.fingerprint(),
            )?,
            producer_fingerprint: decode_sha256_fingerprint(
                "producer.fingerprint",
                &identity.identity.producer.fingerprint(),
            )?,
            input_fingerprint: decode_sha256_fingerprint(
                "input.fingerprint",
                &identity.identity.input.fingerprint(),
            )?,
            storage_fingerprint: decode_sha256_fingerprint(
                "storage.fingerprint",
                &identity.identity.storage.fingerprint(),
            )?,
        };
        Ok(Self { expected })
    }

    /// Expected parsed header.
    #[must_use]
    pub const fn expected_header(&self) -> &WalIdentityHeaderV2 {
        &self.expected
    }

    /// Canonical fixed-width WAL header for this exact binding.
    #[must_use]
    pub fn encode_header(&self) -> [u8; IDENTITY_BOUND_WAL_HEADER_SIZE] {
        self.expected.encode()
    }

    /// Validate a persisted WAL header against every expected identity field.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] when the header itself is
    /// malformed or any generation, dimension, quantization, or component
    /// fingerprint differs.
    pub fn validate_header(&self, data: &[u8], path: &Path) -> SearchResult<()> {
        let actual = WalIdentityHeaderV2::decode(data, path)?;
        if actual.generation != self.expected.generation {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL artifact generation mismatch",
            ));
        }
        if actual.dimension != self.expected.dimension {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL storage dimension mismatch",
            ));
        }
        if actual.quantization != self.expected.quantization {
            return Err(wal_corrupted(
                path,
                "identity-bound WAL storage quantization mismatch",
            ));
        }
        for (field, actual, expected) in [
            (
                "embedding identity bundle fingerprint",
                actual.identity_bundle_fingerprint,
                self.expected.identity_bundle_fingerprint,
            ),
            (
                "embedding space fingerprint",
                actual.space_fingerprint,
                self.expected.space_fingerprint,
            ),
            (
                "embedding producer fingerprint",
                actual.producer_fingerprint,
                self.expected.producer_fingerprint,
            ),
            (
                "embedding input fingerprint",
                actual.input_fingerprint,
                self.expected.input_fingerprint,
            ),
            (
                "vector storage fingerprint",
                actual.storage_fingerprint,
                self.expected.storage_fingerprint,
            ),
        ] {
            if actual != expected {
                return Err(wal_corrupted(
                    path,
                    format!("identity-bound WAL {field} mismatch"),
                ));
            }
        }
        Ok(())
    }
}

// ─── Index tagging ──────────────────────────────────────────────────────────

/// Sentinel bit marking WAL-sourced entries in the search heap.
/// Uses the MSB of usize so main index positions (always < 2^63) are unaffected.
pub(crate) const WAL_INDEX_BIT: usize = 1_usize << (usize::BITS - 1);

pub(crate) const fn is_wal_index(index: usize) -> bool {
    index & WAL_INDEX_BIT != 0
}

pub(crate) const fn to_wal_index(wal_pos: usize) -> usize {
    wal_pos | WAL_INDEX_BIT
}

pub(crate) const fn from_wal_index(tagged: usize) -> usize {
    tagged & !WAL_INDEX_BIT
}

// ─── Path helpers ───────────────────────────────────────────────────────────

/// Derive the WAL sidecar path from the main FSVI index path.
#[must_use]
pub fn wal_path_for(fsvi_path: &Path) -> PathBuf {
    let mut p = fsvi_path.as_os_str().to_os_string();
    p.push(".wal");
    PathBuf::from(p)
}

/// Strictly inspect one WAL pathname without mutating bytes, timestamps, or
/// directory entries.
///
/// The present-file path uses the same no-atime/no-follow, single-link,
/// pre/post inode and containing-directory checks as immutable FSVI admission.
/// Crash-recovery tolerance is deliberately not applied: a truncated or corrupt
/// final batch is an error rather than an implicitly discarded suffix.
///
/// # Errors
///
/// Returns typed snapshot rejection for unsafe path topology or concurrent
/// mutation and [`SearchError::IndexCorrupted`] for every malformed WAL field.
pub fn inspect_wal_strict(path: &Path) -> Result<StrictWalInspection, FsviAdmissionError> {
    let parent = snapshot_parent_or_current(path);
    let parent_metadata = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
    if !parent_metadata.file_type().is_dir() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
            "the WAL parent must be a real directory, not a symlink or special file",
        ));
    }
    let parent_identity = stable_file_identity(&parent_metadata);
    let path_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let after = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
            if !after.file_type().is_dir() || stable_file_identity(&after) != parent_identity {
                return Err(snapshot_rejected(
                    FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                    "the WAL parent changed while absence was being inspected",
                ));
            }
            return Ok(StrictWalInspection::Absent);
        }
        Err(error) => return Err(SearchError::Io(error).into()),
    };
    let path_identity = validate_single_link_regular_file(&path_metadata)?;
    let mut file = open_readonly_noatime_nofollow(path)?;
    let opened_identity =
        validate_single_link_regular_file(&file.metadata().map_err(SearchError::Io)?)?;
    if opened_identity != path_identity {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL pathname and opened descriptor identify different bytes",
        ));
    }
    let byte_len = usize::try_from(path_identity.size)
        .map_err(|_| wal_corrupted(path, "WAL byte length does not fit in usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(|_| SearchError::InvalidConfig {
            field: "wal_snapshot.byte_len".to_owned(),
            value: byte_len.to_string(),
            reason: "unable to reserve the exact immutable WAL byte image".to_owned(),
        })?;
    bytes.resize(byte_len, 0);
    file.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the WAL inode was truncated while being inspected",
            )
        } else {
            FsviAdmissionError::Index(SearchError::Io(error))
        }
    })?;
    let mut trailing = [0_u8; 1];
    if file.read(&mut trailing).map_err(SearchError::Io)? != 0 {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL inode grew while being inspected",
        ));
    }
    let descriptor_identity =
        validate_single_link_regular_file(&file.metadata().map_err(SearchError::Io)?)?;
    let final_path_metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the WAL pathname disappeared while being inspected",
            )
        } else {
            FsviAdmissionError::Index(SearchError::Io(error))
        }
    })?;
    let final_path_identity = validate_single_link_regular_file(&final_path_metadata)?;
    if descriptor_identity != path_identity || final_path_identity != path_identity {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PathChangedDuringRead,
            "the WAL inode identity, size, mode, links, or timestamps changed during inspection",
        ));
    }
    let final_parent = fs::symlink_metadata(parent).map_err(SearchError::Io)?;
    if !final_parent.file_type().is_dir() || stable_file_identity(&final_parent) != parent_identity
    {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
            "the WAL containing directory changed during inspection",
        ));
    }
    parse_strict_wal_bytes(path, &bytes).map_err(FsviAdmissionError::Index)
}

fn parse_strict_wal_bytes(path: &Path, data: &[u8]) -> SearchResult<StrictWalInspection> {
    if data.is_empty() {
        return Ok(StrictWalInspection::Empty {
            whole_image_sha256: Sha256::digest(data).into(),
        });
    }
    if data.len() < 6 {
        return Err(wal_corrupted(path, "WAL magic/version prefix is truncated"));
    }
    if data[..4] != WAL_MAGIC {
        return Err(wal_corrupted(path, "bad magic bytes"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    let (cursor, dimension, quantization, format) = match version {
        WAL_VERSION => {
            if data.len() < WAL_HEADER_SIZE {
                return Err(wal_corrupted(path, "legacy WAL header is truncated"));
            }
            let dimension = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
            if dimension == 0 {
                return Err(wal_corrupted(
                    path,
                    "legacy WAL dimension must be greater than zero",
                ));
            }
            let quantization = Quantization::from_wire(data[10], path)?;
            if data[12..16] != [0; 4] {
                return Err(wal_corrupted(
                    path,
                    "legacy WAL reserved bytes must be zero",
                ));
            }
            let stored_crc = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
            if stored_crc != crc32_of(&data[..16]) {
                return Err(wal_corrupted(path, "legacy WAL header CRC mismatch"));
            }
            (
                WAL_HEADER_SIZE,
                dimension,
                quantization,
                StrictWalFormat::Legacy,
            )
        }
        IDENTITY_BOUND_WAL_VERSION => {
            let header = WalIdentityHeaderV2::decode(data, path)?;
            let dimension = header.dimension();
            let quantization = header.quantization();
            (
                IDENTITY_BOUND_WAL_HEADER_SIZE,
                dimension,
                quantization,
                StrictWalFormat::IdentityBound(header),
            )
        }
        found => {
            return Err(wal_corrupted(
                path,
                format!("unsupported WAL version {found}"),
            ));
        }
    };
    let dimension_usize = usize::try_from(dimension)
        .map_err(|_| wal_corrupted(path, "WAL dimension does not fit in usize"))?;
    let vector_bytes = dimension_usize
        .checked_mul(quantization.bytes_per_element())
        .ok_or_else(|| wal_corrupted(path, "WAL vector byte length overflow"))?;
    let (batch_count, entry_count) = inspect_complete_batches(path, &data[cursor..], vector_bytes)?;
    let image = StrictWalImage {
        byte_len: u64::try_from(data.len())
            .map_err(|_| wal_corrupted(path, "WAL byte length does not fit in u64"))?,
        whole_image_sha256: Sha256::digest(data).into(),
        dimension,
        quantization,
        batch_count,
        entry_count,
    };
    Ok(match format {
        StrictWalFormat::Legacy => StrictWalInspection::LegacyV1(image),
        StrictWalFormat::IdentityBound(header) => {
            StrictWalInspection::IdentityBoundV2 { header, image }
        }
    })
}

enum StrictWalFormat {
    Legacy,
    IdentityBound(WalIdentityHeaderV2),
}

fn inspect_complete_batches(
    path: &Path,
    data: &[u8],
    vector_bytes: usize,
) -> SearchResult<(u64, u64)> {
    let mut cursor = 0usize;
    let mut batch_count = 0_u64;
    let mut total_entries = 0_u64;
    while cursor < data.len() {
        let remaining = &data[cursor..];
        if remaining.len() < 8 {
            return Err(wal_corrupted(path, "WAL batch header is truncated"));
        }
        if remaining[..4] != BATCH_MAGIC {
            return Err(wal_corrupted(path, "WAL batch magic mismatch"));
        }
        let entry_count =
            u32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
        let mut batch_cursor = 8usize;
        for _ in 0..entry_count {
            let len_end = batch_cursor
                .checked_add(2)
                .ok_or_else(|| wal_corrupted(path, "WAL entry length offset overflow"))?;
            if len_end > remaining.len() {
                return Err(wal_corrupted(path, "WAL entry length is truncated"));
            }
            let doc_id_len = usize::from(u16::from_le_bytes([
                remaining[batch_cursor],
                remaining[batch_cursor + 1],
            ]));
            batch_cursor = len_end;
            let doc_end = batch_cursor
                .checked_add(doc_id_len)
                .ok_or_else(|| wal_corrupted(path, "WAL document id offset overflow"))?;
            let vector_end = doc_end
                .checked_add(vector_bytes)
                .ok_or_else(|| wal_corrupted(path, "WAL vector offset overflow"))?;
            if vector_end > remaining.len() {
                return Err(wal_corrupted(path, "WAL entry payload is truncated"));
            }
            std::str::from_utf8(&remaining[batch_cursor..doc_end]).map_err(|error| {
                wal_corrupted(path, format!("WAL document id is not UTF-8: {error}"))
            })?;
            batch_cursor = vector_end;
        }
        let crc_end = batch_cursor
            .checked_add(4)
            .ok_or_else(|| wal_corrupted(path, "WAL batch CRC offset overflow"))?;
        if crc_end > remaining.len() {
            return Err(wal_corrupted(path, "WAL batch CRC is truncated"));
        }
        let stored_crc = u32::from_le_bytes(
            remaining[batch_cursor..crc_end]
                .try_into()
                .map_err(|_| wal_corrupted(path, "WAL batch CRC is truncated"))?,
        );
        if stored_crc != crc32_of(&remaining[..batch_cursor]) {
            return Err(wal_corrupted(path, "WAL batch CRC mismatch"));
        }
        cursor = cursor
            .checked_add(crc_end)
            .ok_or_else(|| wal_corrupted(path, "WAL batch cursor overflow"))?;
        batch_count = batch_count
            .checked_add(1)
            .ok_or_else(|| wal_corrupted(path, "WAL batch count overflow"))?;
        total_entries = total_entries
            .checked_add(u64::from(entry_count))
            .ok_or_else(|| wal_corrupted(path, "WAL entry count overflow"))?;
    }
    Ok((batch_count, total_entries))
}

// ─── WAL reading ────────────────────────────────────────────────────────────

/// Load all valid WAL entries from disk.
///
/// Partially-written batches (crash recovery) are silently discarded.
/// Returns an empty vec if the WAL file does not exist.
pub(crate) fn read_wal(
    path: &Path,
    expected_dimension: usize,
    quantization: Quantization,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    if !path.exists() {
        return Ok((Vec::new(), 0, 0));
    }
    let data = std::fs::read(path)?;
    if data.len() < WAL_HEADER_SIZE {
        warn!(path = %path.display(), len = data.len(), "WAL file too small, ignoring");
        return Ok((Vec::new(), 0, 0));
    }
    parse_wal_bytes(&data, expected_dimension, quantization, path)
}

fn parse_wal_bytes(
    data: &[u8],
    expected_dimension: usize,
    quantization: Quantization,
    path: &Path,
) -> SearchResult<(Vec<WalEntry>, u8, u64)> {
    // Header validation.
    if data[..4] != WAL_MAGIC {
        return Err(wal_corrupted(path, "bad magic bytes"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != WAL_VERSION {
        return Err(wal_corrupted(
            path,
            format!("version mismatch: expected {WAL_VERSION}, got {version}"),
        ));
    }
    let dimension = u32::from_le_bytes([data[6], data[7], data[8], data[9]]) as usize;
    if dimension != expected_dimension {
        return Err(wal_corrupted(
            path,
            format!("dimension mismatch: expected {expected_dimension}, got {dimension}"),
        ));
    }
    let quant_byte = data[10];
    let wal_quant = Quantization::from_wire(quant_byte, path)?;
    if wal_quant != quantization {
        return Err(wal_corrupted(path, "quantization mismatch"));
    }
    let compaction_gen = data[11];
    // reserved: data[12..16]
    let header_crc_stored = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let header_crc_computed = crc32_of(&data[..16]);
    if header_crc_stored != header_crc_computed {
        return Err(wal_corrupted(path, "header CRC mismatch"));
    }

    // Parse batches.
    let mut entries = Vec::new();
    let mut cursor = WAL_HEADER_SIZE;
    let vector_bytes = dimension * quantization.bytes_per_element();

    while cursor + 8 <= data.len() {
        if let Ok((batch_entries, batch_len)) =
            parse_batch(&data[cursor..], dimension, vector_bytes, quantization)
        {
            entries.extend(batch_entries);
            cursor += batch_len;
        } else {
            warn!(
                path = %path.display(),
                offset = cursor,
                entries_recovered = entries.len(),
                "WAL batch corrupt or truncated, discarding remainder"
            );
            break;
        }
    }

    debug!(path = %path.display(), entries = entries.len(), "loaded WAL entries");
    Ok((entries, compaction_gen, cursor as u64))
}

fn parse_batch(
    data: &[u8],
    dimension: usize,
    vector_bytes: usize,
    quantization: Quantization,
) -> Result<(Vec<WalEntry>, usize), ()> {
    if data.len() < 8 {
        return Err(());
    }
    if data[..4] != BATCH_MAGIC {
        return Err(());
    }
    let entry_count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    // Bound capacity to what the data can actually hold (minimum 2 bytes header
    // + vector_bytes per entry) to prevent OOM from malicious entry_count values.
    let min_entry_bytes = 2_usize.saturating_add(vector_bytes).max(1);
    let max_possible = data.len().saturating_sub(8) / min_entry_bytes;
    let mut cursor: usize = 8;
    let mut entries = Vec::with_capacity(entry_count.min(max_possible));

    for _ in 0..entry_count {
        if cursor.checked_add(2).is_none_or(|end| end > data.len()) {
            return Err(());
        }
        let doc_id_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        let needed = doc_id_len.checked_add(vector_bytes).ok_or(())?;
        if cursor
            .checked_add(needed)
            .is_none_or(|end| end > data.len())
        {
            return Err(());
        }
        let doc_id = std::str::from_utf8(&data[cursor..cursor + doc_id_len])
            .map_err(|_| ())?
            .to_owned();
        cursor += doc_id_len;

        let embedding = decode_vector(
            &data[cursor..cursor + vector_bytes],
            dimension,
            quantization,
        );
        cursor += vector_bytes;

        entries.push(WalEntry {
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            doc_id,
            embedding,
        });
    }

    // Verify batch CRC.
    if cursor + 4 > data.len() {
        return Err(());
    }
    let stored_crc = u32::from_le_bytes([
        data[cursor],
        data[cursor + 1],
        data[cursor + 2],
        data[cursor + 3],
    ]);
    let computed_crc = crc32_of(&data[..cursor]);
    if stored_crc != computed_crc {
        return Err(());
    }
    cursor += 4;

    Ok((entries, cursor))
}

fn decode_vector(bytes: &[u8], dimension: usize, quantization: Quantization) -> Vec<f32> {
    let vec = match quantization {
        Quantization::F16 => {
            // SIMD-widen 8 little-endian f16 per 16-byte block
            // (`widen8_f16_bytes`, the same magic-factor widen the f16 dot
            // kernels use — bit-identical to the scalar `f16::to_f32`), then a
            // scalar tail for the last < 8. The last remaining scalar consumer
            // of the vector_at_f32 SIMD-widen route-next (bd-y3hf).
            let f16_bytes = dimension * 2;
            let source = &bytes[..f16_bytes.min(bytes.len())];
            let mut out = Vec::with_capacity(dimension);
            let (blocks, remainder) = source.as_chunks::<16>();
            for arr in blocks {
                out.extend_from_slice(&crate::simd::widen8_f16_bytes(arr).to_array());
            }
            for chunk in remainder.as_chunks::<2>().0 {
                out.push(f16::from_le_bytes(*chunk).to_f32());
            }
            out.truncate(dimension);
            out
        }
        Quantization::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(dimension)
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
    };
    debug_assert_eq!(
        Vec::<f32>::len(&vec),
        dimension,
        "decode_vector produced {actual} elements, expected {dimension}",
        actual = Vec::<f32>::len(&vec),
    );
    vec
}

// ─── WAL writing ────────────────────────────────────────────────────────────

fn wal_entry_count(entries_len: usize) -> SearchResult<u32> {
    u32::try_from(entries_len).map_err(|_| SearchError::InvalidConfig {
        field: "wal_batch_entries".to_owned(),
        value: entries_len.to_string(),
        reason: "WAL batch entry count exceeds u32 maximum".to_owned(),
    })
}

fn wal_doc_id_len(doc_bytes_len: usize) -> SearchResult<u16> {
    u16::try_from(doc_bytes_len).map_err(|_| SearchError::InvalidConfig {
        field: "doc_id_length".to_owned(),
        value: doc_bytes_len.to_string(),
        reason: format!("document ID length {doc_bytes_len} exceeds WAL u16 limit (65535 bytes)"),
    })
}

fn wal_dimension(dimension: usize) -> SearchResult<u32> {
    u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "vector dimension exceeds u32 maximum for WAL header".to_owned(),
    })
}

/// Append a batch of entries to the WAL file.
///
/// If the WAL file does not exist, a header is written first.
/// The batch is CRC32-protected so partial writes are detectable.
pub(crate) fn append_wal_batch(
    wal_path: &Path,
    entries: &[WalEntry],
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    fsync: bool,
) -> SearchResult<()> {
    // Try to create a new WAL file atomically. If successful, we are the creator
    // and responsible for writing the header.
    let created = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(wal_path);

    let mut created_fresh = false;
    let mut file = match created {
        Ok(mut file) => {
            created_fresh = true;
            write_wal_header(&mut file, dimension, quantization, compaction_gen)?;
            file
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            // File exists — open in write mode (not append) so we can
            // seek and truncate in-place without releasing the handle.
            //
            // We use .create(true) here to handle the TOCTOU race where the file
            // is deleted by another process (e.g. compaction) between our
            // create_new check above and this open. If that happens, we create
            // a new empty file, which `existing_len` check below will catch
            // and initialize with a header.
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(wal_path)?;
            let existing_len = file.metadata()?.len();
            if existing_len < WAL_HEADER_SIZE as u64 {
                // Empty or truncated file — the creator likely crashed before
                // flushing the header. Truncate in-place and write a fresh
                // header. We keep the file handle open to avoid a TOCTOU race
                // where a concurrent writer could land valid data between a
                // drop and a reopen-with-truncate.
                warn!(
                    path = %wal_path.display(),
                    existing_len,
                    "WAL file smaller than header; truncating and writing fresh header"
                );
                file.set_len(0)?;
                file.seek(std::io::SeekFrom::Start(0))?;
                write_wal_header(&mut file, dimension, quantization, compaction_gen)?;
                // `.create(true)` above may have re-created a dirent that a
                // concurrent delete removed after our `create_new` failed, or
                // repaired one a crashed creator never durably linked; either
                // way the dirent needs the same parent sync as a fresh create.
                created_fresh = true;
            } else {
                // Seek to end so the batch is appended after existing data.
                file.seek(std::io::SeekFrom::End(0))?;
            }
            file
        }
        Err(error) => return Err(error.into()),
    };

    // Build batch bytes.
    let mut batch = Vec::new();
    batch.extend_from_slice(&BATCH_MAGIC);
    let entry_count = wal_entry_count(entries.len())?;
    batch.extend_from_slice(&entry_count.to_le_bytes());

    for entry in entries {
        let doc_bytes = entry.doc_id.as_bytes();
        let doc_id_len = wal_doc_id_len(doc_bytes.len())?;
        batch.extend_from_slice(&doc_id_len.to_le_bytes());
        batch.extend_from_slice(doc_bytes);
        encode_vector(&mut batch, &entry.embedding, quantization);
    }

    let crc = crc32_of(&batch);
    batch.extend_from_slice(&crc.to_le_bytes());

    file.write_all(&batch)?;

    if fsync {
        file.sync_all()?;
        if created_fresh {
            // A durable first append also needs the CREATE dirent persisted:
            // file fsync alone does not guarantee the new sidecar survives a
            // crash on every filesystem, and a vanished sidecar silently
            // loses every acknowledged append.
            crate::sync_parent_directory(wal_path)?;
        }
    }

    debug!(
        path = %wal_path.display(),
        batch_entries = entries.len(),
        batch_bytes = batch.len(),
        "appended WAL batch"
    );
    Ok(())
}

fn write_wal_header(
    writer: &mut impl Write,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
) -> SearchResult<()> {
    let mut header = Vec::with_capacity(WAL_HEADER_SIZE);
    header.extend_from_slice(&WAL_MAGIC);
    header.extend_from_slice(&WAL_VERSION.to_le_bytes());
    let dim_u32 = wal_dimension(dimension)?;
    header.extend_from_slice(&dim_u32.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    header.push(quantization as u8);
    header.push(compaction_gen);
    header.extend_from_slice(&[0u8; 4]); // reserved
    let crc = crc32_of(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    writer.write_all(&header)?;
    Ok(())
}

fn encode_vector(buf: &mut Vec<u8>, embedding: &[f32], quantization: Quantization) {
    match quantization {
        Quantization::F16 => {
            for &val in embedding {
                buf.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
        }
        Quantization::F32 => {
            for &val in embedding {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
}

/// Remove the WAL sidecar file.
pub(crate) fn remove_wal(path: &Path) -> SearchResult<()> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn crc32_of(data: &[u8]) -> u32 {
    let mut hasher = Crc32Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

const fn quantization_to_wire(quantization: Quantization) -> u8 {
    match quantization {
        Quantization::F32 => 0,
        Quantization::F16 => 1,
    }
}

fn copy_array<const N: usize>(bytes: &[u8], path: &Path, field: &str) -> SearchResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| wal_corrupted(path, format!("identity-bound WAL {field} is truncated")))
}

fn decode_sha256_fingerprint(field: &str, fingerprint: &str) -> SearchResult<[u8; 32]> {
    if fingerprint.len() != 64 {
        return Err(wal_identity_config_error(
            field,
            "must be a 64-character lowercase SHA-256 fingerprint",
        ));
    }
    let mut decoded = [0; 32];
    for (output, pair) in decoded
        .iter_mut()
        .zip(fingerprint.as_bytes().as_chunks::<2>().0)
    {
        let high = decode_lower_hex_nibble(pair[0]).ok_or_else(|| {
            wal_identity_config_error(
                field,
                "must be a 64-character lowercase SHA-256 fingerprint",
            )
        })?;
        let low = decode_lower_hex_nibble(pair[1]).ok_or_else(|| {
            wal_identity_config_error(
                field,
                "must be a 64-character lowercase SHA-256 fingerprint",
            )
        })?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn wal_identity_config_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("wal_identity_v2.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

fn wal_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::generation::EmbeddingIdentityBundleV1;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wal_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-wal-{name}-{}-{now}.fsvi.wal",
            std::process::id()
        ))
    }

    fn make_entry(doc_id: &str, base: f32, dim: usize) -> WalEntry {
        WalEntry {
            doc_id: doc_id.into(),
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            embedding: vec![base; dim],
        }
    }

    fn identity_binding_v2(
        dimension: u32,
        quantization: QuantizationFormat,
    ) -> WalIdentityBindingV2 {
        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("wal-identity-v2", dimension);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = quantization;
        identity.storage.endianness = "little-endian".to_owned();
        let frozen = identity.freeze().expect("valid FSVI v2 identity");
        let generation =
            ArtifactGenerationIdentityV1::new(u64::MAX, [0xa5; 16]).expect("valid generation");
        WalIdentityBindingV2::new(generation, &frozen).expect("valid WAL identity binding")
    }

    fn refresh_identity_header_crc(bytes: &mut [u8; IDENTITY_BOUND_WAL_HEADER_SIZE]) {
        let crc = crc32_of(&bytes[..IDENTITY_BOUND_WAL_CRC_OFFSET]);
        bytes[IDENTITY_BOUND_WAL_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn encoded_test_batch(entries: &[WalEntry], quantization: Quantization) -> Vec<u8> {
        let mut batch = Vec::new();
        batch.extend_from_slice(&BATCH_MAGIC);
        batch.extend_from_slice(
            &u32::try_from(entries.len())
                .expect("test batch entry count fits")
                .to_le_bytes(),
        );
        for entry in entries {
            let doc_id = entry.doc_id.as_bytes();
            batch.extend_from_slice(
                &u16::try_from(doc_id.len())
                    .expect("test document id fits")
                    .to_le_bytes(),
            );
            batch.extend_from_slice(doc_id);
            encode_vector(&mut batch, &entry.embedding, quantization);
        }
        let crc = crc32_of(&batch);
        batch.extend_from_slice(&crc.to_le_bytes());
        batch
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn directory_entry_names(path: &Path) -> Vec<std::ffi::OsString> {
        let mut names: Vec<_> = fs::read_dir(path)
            .expect("read private WAL fixture directory")
            .map(|entry| entry.expect("read WAL fixture entry").file_name())
            .collect();
        names.sort_unstable();
        names
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn read_noatime(path: &Path) -> Vec<u8> {
        let mut file = open_readonly_noatime_nofollow(path).expect("open test WAL without atime");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .expect("read test WAL without atime");
        bytes
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn freeze_test_timestamps(path: &Path) {
        let timestamp = UNIX_EPOCH + std::time::Duration::from_secs(1_600_000_321);
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .expect("open WAL to freeze timestamps");
        file.set_times(
            std::fs::FileTimes::new()
                .set_accessed(timestamp)
                .set_modified(timestamp),
        )
        .expect("freeze WAL timestamps");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn assert_strict_inspection_preserved_snapshot(
        directory: &Path,
        path: &Path,
        expected_bytes: &[u8],
        file_before: crate::StableFileIdentity,
        parent_before: crate::StableFileIdentity,
        entries_before: &[std::ffi::OsString],
    ) {
        assert_eq!(
            stable_file_identity(&fs::symlink_metadata(path).expect("WAL metadata after")),
            file_before
        );
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(directory).expect("WAL parent metadata after")
            ),
            parent_before
        );
        assert_eq!(directory_entry_names(directory), entries_before);
        assert_eq!(read_noatime(path), expected_bytes);
        assert_eq!(
            stable_file_identity(&fs::symlink_metadata(path).expect("WAL metadata after reread")),
            file_before
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn strict_inspection_classifies_absent_empty_legacy_and_identity_bound_images() {
        let directory = tempfile::tempdir().expect("private strict WAL directory");

        let absent = directory.path().join("absent.fsvi.wal");
        let absent_entries = directory_entry_names(directory.path());
        let absent_parent = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("absent parent before"),
        );
        assert_eq!(
            inspect_wal_strict(&absent).expect("inspect absent WAL"),
            StrictWalInspection::Absent
        );
        assert_eq!(
            stable_file_identity(
                &fs::symlink_metadata(directory.path()).expect("absent parent after")
            ),
            absent_parent
        );
        assert_eq!(directory_entry_names(directory.path()), absent_entries);
        assert!(!absent.exists());

        let empty = directory.path().join("empty.fsvi.wal");
        fs::write(&empty, []).expect("write empty WAL");
        freeze_test_timestamps(&empty);
        let empty_bytes = Vec::new();
        let empty_entries = directory_entry_names(directory.path());
        let empty_file =
            stable_file_identity(&fs::symlink_metadata(&empty).expect("empty metadata before"));
        let empty_parent = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("empty parent before"),
        );
        assert_eq!(
            inspect_wal_strict(&empty).expect("inspect empty WAL"),
            StrictWalInspection::Empty {
                whole_image_sha256: Sha256::digest([]).into(),
            }
        );
        assert_strict_inspection_preserved_snapshot(
            directory.path(),
            &empty,
            &empty_bytes,
            empty_file,
            empty_parent,
            &empty_entries,
        );

        let header_only = directory.path().join("header-only.fsvi.wal");
        let mut header_bytes = Vec::new();
        write_wal_header(&mut header_bytes, 4, Quantization::F32, 7).expect("encode legacy header");
        fs::write(&header_only, &header_bytes).expect("write header-only WAL");
        freeze_test_timestamps(&header_only);
        let header_entries = directory_entry_names(directory.path());
        let header_file = stable_file_identity(
            &fs::symlink_metadata(&header_only).expect("header metadata before"),
        );
        let header_parent = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("header parent before"),
        );
        let StrictWalInspection::LegacyV1(header_image) =
            inspect_wal_strict(&header_only).expect("inspect header-only WAL")
        else {
            panic!("header-only WAL was not classified as legacy v1");
        };
        assert_eq!(header_image.byte_len, WAL_HEADER_SIZE as u64);
        let expected_header_sha256: [u8; 32] = Sha256::digest(&header_bytes).into();
        assert_eq!(header_image.whole_image_sha256, expected_header_sha256);
        assert_eq!(header_image.dimension, 4);
        assert_eq!(header_image.quantization, Quantization::F32);
        assert_eq!(header_image.batch_count, 0);
        assert_eq!(header_image.entry_count, 0);
        assert_strict_inspection_preserved_snapshot(
            directory.path(),
            &header_only,
            &header_bytes,
            header_file,
            header_parent,
            &header_entries,
        );

        let multi_batch = directory.path().join("multi-batch.fsvi.wal");
        append_wal_batch(
            &multi_batch,
            &[make_entry("first", 1.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .expect("write first strict batch");
        append_wal_batch(
            &multi_batch,
            &[make_entry("second", 2.0, 4), make_entry("third", 3.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .expect("write second strict batch");
        let multi_bytes = fs::read(&multi_batch).expect("read multi-batch bytes");
        freeze_test_timestamps(&multi_batch);
        let multi_entries = directory_entry_names(directory.path());
        let multi_file = stable_file_identity(
            &fs::symlink_metadata(&multi_batch).expect("multi metadata before"),
        );
        let multi_parent = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("multi parent before"),
        );
        let StrictWalInspection::LegacyV1(multi_image) =
            inspect_wal_strict(&multi_batch).expect("inspect multi-batch WAL")
        else {
            panic!("multi-batch WAL was not classified as legacy v1");
        };
        assert_eq!(
            multi_image.byte_len,
            u64::try_from(multi_bytes.len()).expect("multi byte length fits")
        );
        let expected_multi_sha256: [u8; 32] = Sha256::digest(&multi_bytes).into();
        assert_eq!(multi_image.whole_image_sha256, expected_multi_sha256);
        assert_eq!(multi_image.dimension, 4);
        assert_eq!(multi_image.quantization, Quantization::F16);
        assert_eq!(multi_image.batch_count, 2);
        assert_eq!(multi_image.entry_count, 3);
        assert_strict_inspection_preserved_snapshot(
            directory.path(),
            &multi_batch,
            &multi_bytes,
            multi_file,
            multi_parent,
            &multi_entries,
        );

        let identity_bound = directory.path().join("identity-bound.fsvi.wal");
        let identity_binding = identity_binding_v2(4, QuantizationFormat::F16);
        let identity_header = identity_binding.encode_header();
        let identity_entries = [make_entry("identity-doc", 4.0, 4)];
        let mut identity_bytes = identity_header.to_vec();
        identity_bytes.extend_from_slice(&encoded_test_batch(&identity_entries, Quantization::F16));
        fs::write(&identity_bound, &identity_bytes).expect("write identity-bound WAL");
        freeze_test_timestamps(&identity_bound);
        let bound_entries = directory_entry_names(directory.path());
        let bound_file = stable_file_identity(
            &fs::symlink_metadata(&identity_bound).expect("identity metadata before"),
        );
        let bound_parent = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("identity parent before"),
        );
        let StrictWalInspection::IdentityBoundV2 { header, image } =
            inspect_wal_strict(&identity_bound).expect("inspect identity-bound WAL")
        else {
            panic!("identity-bound WAL was not classified as v2");
        };
        identity_binding
            .validate_header(&header.encode(), &identity_bound)
            .expect("strict inspector retained exact identity header");
        assert_eq!(
            image.byte_len,
            u64::try_from(identity_bytes.len()).expect("identity byte length fits")
        );
        let expected_identity_sha256: [u8; 32] = Sha256::digest(&identity_bytes).into();
        assert_eq!(image.whole_image_sha256, expected_identity_sha256);
        assert_eq!(image.dimension, 4);
        assert_eq!(image.quantization, Quantization::F16);
        assert_eq!(image.batch_count, 1);
        assert_eq!(image.entry_count, 1);
        assert_strict_inspection_preserved_snapshot(
            directory.path(),
            &identity_bound,
            &identity_bytes,
            bound_file,
            bound_parent,
            &bound_entries,
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn strict_inspection_rejects_corrupt_or_truncated_suffix_without_recovery_mutation() {
        let directory = tempfile::tempdir().expect("private corrupt WAL directory");
        let source = directory.path().join("source.fsvi.wal");
        append_wal_batch(
            &source,
            &[make_entry("complete", 1.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .expect("write complete source WAL");
        let complete = fs::read(&source).expect("read complete source WAL");

        let mut bad_crc = complete.clone();
        *bad_crc.last_mut().expect("batch CRC byte") ^= 1;
        let mut truncated = complete.clone();
        truncated.pop();
        let mut corrupt_suffix = complete;
        corrupt_suffix.extend_from_slice(b"FWB");
        for (name, bytes) in [
            ("bad-crc", bad_crc),
            ("truncated", truncated),
            ("corrupt-suffix", corrupt_suffix),
        ] {
            let path = directory.path().join(format!("{name}.fsvi.wal"));
            fs::write(&path, &bytes).expect("write malformed strict WAL");
            freeze_test_timestamps(&path);
            let entries_before = directory_entry_names(directory.path());
            let file_before = stable_file_identity(
                &fs::symlink_metadata(&path).expect("malformed metadata before"),
            );
            let parent_before = stable_file_identity(
                &fs::symlink_metadata(directory.path()).expect("malformed parent before"),
            );

            assert!(matches!(
                inspect_wal_strict(&path),
                Err(FsviAdmissionError::Index(
                    SearchError::IndexCorrupted { .. }
                ))
            ));
            assert_strict_inspection_preserved_snapshot(
                directory.path(),
                &path,
                &bytes,
                file_before,
                parent_before,
                &entries_before,
            );
        }
    }

    #[test]
    fn identity_bound_header_round_trips_full_width_generation() {
        let path = Path::new("/test/identity-bound.fsvi.wal");
        let binding = identity_binding_v2(37, QuantizationFormat::F16);
        let encoded = binding.encode_header();
        let decoded = WalIdentityHeaderV2::decode(&encoded, path).expect("decode v2 header");

        binding
            .validate_header(&encoded, path)
            .expect("exact identity binding");
        assert_eq!(decoded.generation().sequence, u64::MAX);
        assert_eq!(decoded.generation().nonce, [0xa5; 16]);
        assert_eq!(decoded.dimension(), 37);
        assert_eq!(decoded.quantization(), Quantization::F16);
        assert_ne!(
            *decoded.identity_bundle_fingerprint(),
            [0; SHA256_FINGERPRINT_BYTES]
        );
        assert_ne!(*decoded.space_fingerprint(), [0; SHA256_FINGERPRINT_BYTES]);
        assert_ne!(
            *decoded.producer_fingerprint(),
            [0; SHA256_FINGERPRINT_BYTES]
        );
        assert_ne!(*decoded.input_fingerprint(), [0; SHA256_FINGERPRINT_BYTES]);
        assert_ne!(
            *decoded.storage_fingerprint(),
            [0; SHA256_FINGERPRINT_BYTES]
        );
        assert_eq!(decoded.encode(), encoded);
    }

    #[test]
    fn identity_bound_header_rejects_every_component_fingerprint_drift() {
        let path = Path::new("/test/identity-bound-fingerprint.fsvi.wal");
        let binding = identity_binding_v2(8, QuantizationFormat::F16);
        let original = binding.encode_header();

        for (field, offset) in [
            (
                "embedding identity bundle fingerprint",
                IDENTITY_BUNDLE_FINGERPRINT_OFFSET,
            ),
            ("embedding space fingerprint", SPACE_FINGERPRINT_OFFSET),
            (
                "embedding producer fingerprint",
                PRODUCER_FINGERPRINT_OFFSET,
            ),
            ("embedding input fingerprint", INPUT_FINGERPRINT_OFFSET),
            ("vector storage fingerprint", STORAGE_FINGERPRINT_OFFSET),
        ] {
            let mut drifted = original;
            drifted[offset] ^= 1;
            refresh_identity_header_crc(&mut drifted);
            let error = binding
                .validate_header(&drifted, path)
                .expect_err("fingerprint drift must fail closed");
            assert!(
                error.to_string().contains(field),
                "expected {field} diagnostic, got {error}"
            );
        }
    }

    #[test]
    fn identity_bound_header_rejects_generation_dimension_and_quantization_drift() {
        let path = Path::new("/test/identity-bound-storage.fsvi.wal");
        let binding = identity_binding_v2(8, QuantizationFormat::F16);
        let original = binding.encode_header();

        let mut sequence_drift = original;
        sequence_drift[20] ^= 1;
        refresh_identity_header_crc(&mut sequence_drift);
        assert!(
            binding
                .validate_header(&sequence_drift, path)
                .expect_err("generation sequence drift")
                .to_string()
                .contains("artifact generation mismatch")
        );

        let mut nonce_drift = original;
        nonce_drift[28] ^= 1;
        refresh_identity_header_crc(&mut nonce_drift);
        assert!(
            binding
                .validate_header(&nonce_drift, path)
                .expect_err("generation nonce drift")
                .to_string()
                .contains("artifact generation mismatch")
        );

        let mut schema_drift = original;
        schema_drift[16..18].copy_from_slice(&2_u16.to_le_bytes());
        refresh_identity_header_crc(&mut schema_drift);
        assert!(
            binding
                .validate_header(&schema_drift, path)
                .expect_err("generation schema drift")
                .to_string()
                .contains("artifact generation is invalid")
        );

        let mut dimension_drift = original;
        dimension_drift[8..12].copy_from_slice(&9_u32.to_le_bytes());
        refresh_identity_header_crc(&mut dimension_drift);
        assert!(
            binding
                .validate_header(&dimension_drift, path)
                .expect_err("dimension drift")
                .to_string()
                .contains("storage dimension mismatch")
        );

        let mut quantization_drift = original;
        quantization_drift[12] = quantization_to_wire(Quantization::F32);
        refresh_identity_header_crc(&mut quantization_drift);
        assert!(
            binding
                .validate_header(&quantization_drift, path)
                .expect_err("quantization drift")
                .to_string()
                .contains("storage quantization mismatch")
        );
    }

    #[test]
    fn identity_bound_header_rejects_structural_corruption() {
        let path = Path::new("/test/identity-bound-corrupt.fsvi.wal");
        let binding = identity_binding_v2(8, QuantizationFormat::F16);
        let original = binding.encode_header();

        let mut bad_version = original;
        bad_version[4..6].copy_from_slice(&99_u16.to_le_bytes());
        assert!(WalIdentityHeaderV2::decode(&bad_version, path).is_err());

        let mut bad_size = original;
        bad_size[6..8].copy_from_slice(&207_u16.to_le_bytes());
        assert!(WalIdentityHeaderV2::decode(&bad_size, path).is_err());

        let mut reserved = original;
        reserved[13] = 1;
        refresh_identity_header_crc(&mut reserved);
        assert!(
            WalIdentityHeaderV2::decode(&reserved, path)
                .expect_err("reserved bytes")
                .to_string()
                .contains("reserved")
        );

        let mut bad_crc = original;
        bad_crc[IDENTITY_BOUND_WAL_CRC_OFFSET] ^= 1;
        assert!(
            WalIdentityHeaderV2::decode(&bad_crc, path)
                .expect_err("CRC drift")
                .to_string()
                .contains("CRC")
        );

        let mut zero_fingerprint = original;
        zero_fingerprint[IDENTITY_BUNDLE_FINGERPRINT_OFFSET..SPACE_FINGERPRINT_OFFSET].fill(0);
        refresh_identity_header_crc(&mut zero_fingerprint);
        assert!(
            WalIdentityHeaderV2::decode(&zero_fingerprint, path)
                .expect_err("zero fingerprint")
                .to_string()
                .contains("must not be all zero")
        );

        assert!(
            WalIdentityHeaderV2::decode(&original[..IDENTITY_BOUND_WAL_HEADER_SIZE - 1], path)
                .is_err()
        );
    }

    #[test]
    fn identity_binding_requires_exact_supported_fsvi_v2_storage() {
        let generation = ArtifactGenerationIdentityV1::new(1, [1; 16]).expect("valid generation");
        let mut legacy = EmbeddingIdentityBundleV1::explicit_test_model("wal-legacy", 8);
        legacy.storage.format = "fsvi-v1".to_owned();
        legacy.storage.quantization = QuantizationFormat::F16;
        legacy.storage.endianness = "little-endian".to_owned();
        let legacy = legacy.freeze().expect("valid legacy identity");
        assert!(
            WalIdentityBindingV2::new(generation, &legacy)
                .expect_err("legacy storage must not bind")
                .to_string()
                .contains("storage.format")
        );

        let mut unsupported = EmbeddingIdentityBundleV1::explicit_test_model("wal-unsupported", 8);
        unsupported.storage.format = "fsvi-v2".to_owned();
        unsupported.storage.quantization = QuantizationFormat::Int8;
        unsupported.storage.endianness = "little-endian".to_owned();
        let unsupported = unsupported.freeze().expect("valid unsupported identity");
        assert!(
            WalIdentityBindingV2::new(generation, &unsupported)
                .expect_err("unsupported quantization must not bind")
                .to_string()
                .contains("storage.quantization")
        );
    }

    #[test]
    fn decode_vector_f16_simd_matches_scalar_bit_exact() {
        // Exercise both the SIMD 16-byte block path and the scalar tail: dim
        // 13 = one 8-lane block + a 5-element remainder. Values span normal,
        // subnormal, signed, and non-finite f16 encodings so any magic-factor
        // widen mismatch surfaces as a bit difference (bd-y3hf).
        let f16_values: [f16; 13] = [
            f16::from_f32(0.0),
            f16::from_f32(-0.0),
            f16::from_f32(1.5),
            f16::from_f32(-2.25),
            f16::from_f32(65504.0),
            f16::from_f32(-65504.0),
            f16::from_bits(0x0001),
            f16::from_bits(0x8001),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::NAN,
            f16::from_f32(0.333_251),
            f16::from_f32(-123.4),
        ];
        let mut bytes = Vec::with_capacity(f16_values.len() * 2);
        for value in f16_values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let simd = decode_vector(&bytes, f16_values.len(), Quantization::F16);
        let scalar: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(f16_values.len())
            .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
            .collect();

        assert_eq!(simd.len(), f16_values.len());
        assert_eq!(scalar.len(), f16_values.len());
        for (index, (lhs, rhs)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(
                lhs.to_bits(),
                rhs.to_bits(),
                "SIMD-widen f16 decode diverged from scalar at lane {index}",
            );
        }
    }

    #[test]
    fn roundtrip_single_batch() {
        let path = temp_wal_path("roundtrip-single");
        let dim = 4;
        let entries = vec![make_entry("doc-0", 1.0, dim), make_entry("doc-1", 2.0, dim)];

        append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();

        let (loaded, generation, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(generation, 0);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].doc_id, "doc-0");
        assert_eq!(loaded[1].doc_id, "doc-1");
        assert!((loaded[0].embedding[0] - 1.0).abs() < 0.01);
        assert!((loaded[1].embedding[0] - 2.0).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_multiple_batches() {
        let path = temp_wal_path("roundtrip-multi");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();
        append_wal_batch(
            &path,
            &[make_entry("doc-1", 2.0, dim), make_entry("doc-2", 3.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let (loaded, generation, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(generation, 0);
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].doc_id, "doc-0");
        assert_eq!(loaded[1].doc_id, "doc-1");
        assert_eq!(loaded[2].doc_id, "doc-2");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncated_batch_is_discarded() {
        let path = temp_wal_path("truncated");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        // Append garbage to simulate a partial batch write.
        let mut data = std::fs::read(&path).unwrap();
        data.extend_from_slice(&BATCH_MAGIC);
        data.extend_from_slice(&1_u32.to_le_bytes()); // claims 1 entry
        data.extend_from_slice(&[0xFF; 3]); // truncated entry
        std::fs::write(&path, &data).unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1, "only the good batch should survive");
        assert_eq!(loaded[0].doc_id, "doc-good");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn crc_corrupted_batch_is_discarded() {
        let path = temp_wal_path("crc-corrupt");
        let dim = 4;

        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();
        append_wal_batch(
            &path,
            &[make_entry("doc-bad", 2.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        // Corrupt the CRC of the second batch.
        let mut data = std::fs::read(&path).unwrap();
        let last_byte = data.len() - 1;
        data[last_byte] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(
            loaded.len(),
            1,
            "corrupted second batch should be discarded"
        );
        assert_eq!(loaded[0].doc_id, "doc-good");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn nonexistent_wal_returns_empty() {
        let path = temp_wal_path("nonexistent");
        let (loaded, _, _) = read_wal(&path, 4, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn dimension_mismatch_is_error() {
        let path = temp_wal_path("dim-mismatch");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let result = read_wal(&path, 8, Quantization::F16);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_quantization_roundtrip() {
        let path = temp_wal_path("f32-quant");
        let dim = 4;
        let entries = vec![make_entry("doc-0", 0.123_456, dim)];

        append_wal_batch(&path, &entries, dim, Quantization::F32, 0, false).unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F32).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(
            (loaded[0].embedding[0] - 0.123_456).abs() < f32::EPSILON,
            "f32 should round-trip exactly"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_path_derivation() {
        let fsvi = Path::new("/data/index.fsvi");
        let wal = wal_path_for(fsvi);
        assert_eq!(wal.to_str().unwrap(), "/data/index.fsvi.wal");
    }

    #[test]
    fn remove_wal_is_idempotent() {
        let path = temp_wal_path("remove-idem");
        remove_wal(&path).unwrap(); // doesn't exist — ok
        std::fs::write(&path, b"dummy").unwrap();
        remove_wal(&path).unwrap(); // exists — removed
        assert!(!path.exists());
        remove_wal(&path).unwrap(); // gone again — ok
    }

    #[test]
    fn index_tagging_roundtrip() {
        let pos = 42;
        let tagged = to_wal_index(pos);
        assert!(is_wal_index(tagged));
        assert!(!is_wal_index(pos));
        assert_eq!(from_wal_index(tagged), pos);
    }

    // ─── bd-vbzm tests begin ───

    #[test]
    fn wal_config_default_values() {
        let config = WalConfig::default();
        assert_eq!(config.compaction_threshold, 1000);
        assert!((config.compaction_ratio - 0.10).abs() < f64::EPSILON);
        assert!(config.fsync_on_write);
    }

    #[test]
    fn wal_config_debug_clone() {
        let config = WalConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.compaction_threshold, config.compaction_threshold);
        let debug = format!("{config:?}");
        assert!(debug.contains("WalConfig"));
    }

    #[test]
    fn compaction_stats_debug_clone() {
        let stats = CompactionStats {
            main_records_before: 100,
            wal_records: 10,
            total_records_after: 110,
            elapsed_ms: 42.5,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.main_records_before, 100);
        assert_eq!(cloned.wal_records, 10);
        assert_eq!(cloned.total_records_after, 110);
        assert!((cloned.elapsed_ms - 42.5).abs() < f64::EPSILON);
        let debug = format!("{stats:?}");
        assert!(debug.contains("CompactionStats"));
    }

    #[test]
    fn wal_too_small_returns_empty() {
        let path = temp_wal_path("too-small");
        std::fs::write(&path, [0u8; 10]).unwrap();
        let (loaded, _, _) = read_wal(&path, 4, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_magic_bytes_is_error() {
        let path = temp_wal_path("bad-magic");
        let mut header = vec![0xDE, 0xAD, 0xBE, 0xEF]; // wrong magic
        header.extend_from_slice(&WAL_VERSION.to_le_bytes());
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.push(Quantization::F16 as u8);
        header.push(0); // compaction_gen
        header.extend_from_slice(&[0u8; 4]);
        let crc = crc32_of(&header);
        header.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &header).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("magic"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn version_mismatch_is_error() {
        let path = temp_wal_path("version-mismatch");
        let mut header = Vec::new();
        header.extend_from_slice(&WAL_MAGIC);
        header.extend_from_slice(&99_u16.to_le_bytes()); // wrong version
        header.extend_from_slice(&4_u32.to_le_bytes());
        header.push(Quantization::F16 as u8);
        header.push(0); // compaction_gen
        header.extend_from_slice(&[0u8; 4]);
        let crc = crc32_of(&header);
        header.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &header).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("version"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn quantization_mismatch_is_error() {
        let path = temp_wal_path("quant-mismatch");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let result = read_wal(&path, 4, Quantization::F32);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("quantization"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn header_crc_mismatch_is_error() {
        let path = temp_wal_path("header-crc");
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, 4)],
            4,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Corrupt header CRC (bytes 16-19)
        data[16] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = read_wal(&path, 4, Quantization::F16);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("CRC"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_batch_roundtrip() {
        let path = temp_wal_path("empty-batch");
        let dim = 4;
        append_wal_batch(&path, &[], dim, Quantization::F16, 0, false).unwrap();
        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_entry_doc_id_hash_matches_fnv1a() {
        let entry = make_entry("hello", 1.0, 4);
        assert_eq!(entry.doc_id_hash, crate::fnv1a_hash(b"hello"));
    }

    #[test]
    fn wal_entry_clone() {
        let entry = make_entry("test-doc", 0.5, 8);
        #[allow(clippy::redundant_clone)]
        let cloned = entry.clone();
        assert_eq!(cloned.doc_id, "test-doc");
        assert_eq!(cloned.embedding.len(), 8);
        assert!((cloned.embedding[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn index_tagging_zero() {
        let tagged = to_wal_index(0);
        assert!(is_wal_index(tagged));
        assert_eq!(from_wal_index(tagged), 0);
    }

    #[test]
    fn index_tagging_large_value() {
        let pos = (1_usize << (usize::BITS - 2)) - 1;
        let tagged = to_wal_index(pos);
        assert!(is_wal_index(tagged));
        assert_eq!(from_wal_index(tagged), pos);
    }

    #[test]
    fn is_wal_index_false_for_normal_values() {
        assert!(!is_wal_index(0));
        assert!(!is_wal_index(1));
        assert!(!is_wal_index(usize::MAX >> 1));
    }

    #[test]
    fn crc32_of_empty() {
        let crc = crc32_of(&[]);
        assert_eq!(crc, 0); // CRC32 of empty is 0
    }

    #[test]
    fn crc32_of_known_data() {
        let crc = crc32_of(b"hello");
        assert_ne!(crc, 0);
        // Same input should produce same CRC
        assert_eq!(crc, crc32_of(b"hello"));
    }

    #[test]
    fn oversized_doc_id_returns_error() {
        let path = temp_wal_path("oversized-docid");
        let dim = 4;
        // Create a doc_id that exceeds u16::MAX (65535) bytes.
        let big_id = "x".repeat(70_000);
        let entries = vec![WalEntry {
            doc_id: big_id,
            doc_id_hash: 0,
            embedding: vec![1.0; dim],
        }];
        let result = append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false);
        assert!(result.is_err(), "should reject doc_id exceeding u16 length");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("doc_id_length"),
            "error should mention doc_id_length: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn oversized_batch_entry_count_returns_error() {
        let result = wal_entry_count(usize::MAX);
        assert!(result.is_err(), "should reject entry count exceeding u32");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("wal_batch_entries"),
            "error should mention wal_batch_entries: {err}"
        );
    }

    #[test]
    fn oversized_dimension_returns_error() {
        let path = temp_wal_path("oversized-dimension");
        let result = append_wal_batch(&path, &[], usize::MAX, Quantization::F16, 0, false);
        assert!(result.is_err(), "should reject dimension exceeding u32");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("dimension"),
            "error should mention dimension: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_corrupted_produces_index_corrupted_error() {
        let err = wal_corrupted(Path::new("/test/path.fsvi.wal"), "test detail");
        match err {
            SearchError::IndexCorrupted { path, detail } => {
                assert_eq!(path, Path::new("/test/path.fsvi.wal"));
                assert_eq!(detail, "test detail");
            }
            other => panic!("expected IndexCorrupted, got {other:?}"),
        }
    }

    #[test]
    fn wal_path_for_various_inputs() {
        let wal = wal_path_for(Path::new("index.fsvi"));
        assert_eq!(wal, PathBuf::from("index.fsvi.wal"));

        let wal = wal_path_for(Path::new("/abs/path/data.fsvi"));
        assert_eq!(wal, PathBuf::from("/abs/path/data.fsvi.wal"));
    }

    #[test]
    fn append_with_fsync_succeeds() {
        let path = temp_wal_path("fsync-test");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-sync", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            true,
        )
        .unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-sync");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_preserves_existing_header() {
        let path = temp_wal_path("preserve-header");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-0", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let data_after_first = std::fs::read(&path).unwrap();
        let header_bytes = &data_after_first[..WAL_HEADER_SIZE];

        append_wal_batch(
            &path,
            &[make_entry("doc-1", 2.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let data_after_second = std::fs::read(&path).unwrap();
        // Header should be unchanged
        assert_eq!(&data_after_second[..WAL_HEADER_SIZE], header_bytes);
        // File should be larger
        assert!(data_after_second.len() > data_after_first.len());

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_batch_magic_discards_batch() {
        let path = temp_wal_path("bad-batch-magic");
        let dim = 4;
        append_wal_batch(
            &path,
            &[make_entry("doc-good", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            false,
        )
        .unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Append a batch with wrong magic
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // bad batch magic
        data.extend_from_slice(&1_u32.to_le_bytes());
        std::fs::write(&path, &data).unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-good");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f16_quantization_has_limited_precision() {
        let path = temp_wal_path("f16-precision");
        let dim = 2;
        let entries = vec![WalEntry {
            doc_id: "precise".into(),
            doc_id_hash: crate::fnv1a_hash(b"precise"),
            embedding: vec![0.123_456, -0.987_654],
        }];

        append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();
        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        // F16 has ~3 decimal digits of precision
        assert!((loaded[0].embedding[0] - 0.123_456).abs() < 0.001);
        assert!((loaded[0].embedding[1] + 0.987_654).abs() < 0.001);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_quantization_exact_roundtrip() {
        let path = temp_wal_path("f32-exact");
        let dim = 3;
        let val = std::f32::consts::PI;
        let entries = vec![WalEntry {
            doc_id: "pi-doc".into(),
            doc_id_hash: crate::fnv1a_hash(b"pi-doc"),
            embedding: vec![val, -val, 0.0],
        }];

        append_wal_batch(&path, &entries, dim, Quantization::F32, 0, false).unwrap();
        let (loaded, _, _) = read_wal(&path, dim, Quantization::F32).unwrap();
        assert!((loaded[0].embedding[0] - val).abs() < f32::EPSILON);
        assert!((loaded[0].embedding[1] + val).abs() < f32::EPSILON);
        assert!(loaded[0].embedding[2].abs() < f32::EPSILON);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_to_preexisting_empty_file_writes_header() {
        let path = temp_wal_path("pre-empty");
        let dim = 3;

        // Create an empty file before calling append_wal_batch.
        // This simulates the race window the TOCTOU fix addresses:
        // another process might create the file between our existence
        // check and our open call.
        std::fs::File::create(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);

        let entries = vec![WalEntry {
            doc_id: "doc-0".into(),
            doc_id_hash: crate::fnv1a_hash(b"doc-0"),
            embedding: vec![1.0, 2.0, 3.0],
        }];
        append_wal_batch(&path, &entries, dim, Quantization::F16, 0, false).unwrap();

        let (loaded, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-0");
        std::fs::remove_file(&path).ok();
    }

    // ─── bd-vbzm tests end ───
}
