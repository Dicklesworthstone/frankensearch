//! Vector index storage and loading for frankensearch.
//!
//! This crate implements the FSVI binary format reader/writer plus exact
//! brute-force top-k vector search, with optional HNSW ANN acceleration.
//!
//! # FSVI File Layout
//!
//! All multi-byte integers are little-endian. The vector slab is 64-byte
//! aligned for cache-line / SIMD friendliness.
//!
//! ```text
//! ┌───────────────────────────────────────────┐
//! │ Header (variable length)                  │
//! │   magic: b"FSVI"              (4 bytes)   │
//! │   version: u16                (2 bytes)   │
//! │   embedder_id_len: u16        (2 bytes)   │
//! │   embedder_id: [u8]           (variable)  │
//! │   embedder_revision_len: u16  (2 bytes)   │
//! │   embedder_revision: [u8]     (variable)  │
//! │   dimension: u32              (4 bytes)   │
//! │   quantization: u8            (1 byte)    │
//! │   reserved: [u8; 3]           (3 bytes)   │
//! │   record_count: u64           (8 bytes)   │
//! │   vectors_offset: u64         (8 bytes)   │
//! │   header_crc32: u32           (4 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ Record Table                              │
//! │   record_count × 16 bytes each:           │
//! │     doc_id_hash: u64          (8 bytes)   │
//! │     doc_id_offset: u32        (4 bytes)   │
//! │     doc_id_len: u16           (2 bytes)   │
//! │     flags: u16                (2 bytes)   │
//! ├───────────────────────────────────────────┤
//! │ String Table                              │
//! │   Concatenated UTF-8 doc_id strings       │
//! ├───────────────────────────────────────────┤
//! │ Padding (to 64-byte alignment)            │
//! ├───────────────────────────────────────────┤
//! │ Vector Slab                               │
//! │   record_count × dimension × elem_size    │
//! │   (2 bytes/elem for f16, 4 for f32)       │
//! └───────────────────────────────────────────┘
//! ```

#[cfg(unix)]
pub mod fd_acl;
pub mod file_identity;
#[cfg(feature = "ann")]
pub mod hnsw;
pub mod in_memory;
pub mod mapped_file;
pub mod mrl;
pub mod native_hnsw;
pub mod quantization;
pub mod recall_certificate;
mod repro_soft_delete_rollback;
mod repro_wal_shadow_bug;
mod repro_wal_truncation;
pub mod search;
pub mod simd;
pub mod two_tier;
pub mod wal;
pub mod warmup;

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crc32fast::Hasher as Crc32;
use frankensearch_core::config::ZeroSignalState;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1,
    EmbeddingProjectionV1, EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1,
    FrozenEmbeddingIdentityBundleV1, HashControlProfileV1, QuantizationFormat,
    VECTOR_STORAGE_IDENTITY_SCHEMA_V1, VectorStorageIdentityV1,
};
use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use memmap2::MmapMut;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::debug;

pub use frankensearch_core::config::{ZERO_SIGNAL_SCHEMA_VERSION, ZeroSignalReason};
#[cfg(feature = "ann")]
pub use hnsw::{
    AnnFallbackReason, AnnSearchStats, HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_DEFAULT_EF_SEARCH,
    HNSW_DEFAULT_M, HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex, HnswLoadDisposition,
};
pub use in_memory::{InMemoryTwoTierIndex, InMemoryVectorIndex};
pub use mrl::{MrlConfig, MrlSearchStats};
pub use quantization::ScalarQuantizer;
pub use recall_certificate::{
    CertifiedEf, EfCalibration, calibrate_certified_ef, certified_min_ef, certified_min_ef_mean,
    conformal_recall_lower_bound, mean_recall_lower_bound, mean_recall_lower_bound_bernstein,
};
pub use search::{ClassifiedHits, PARALLEL_CHUNK_SIZE, PARALLEL_THRESHOLD, SearchParams};
pub use simd::{
    PreparedQuery4bit, cosine_similarity_f16, dot_4bit_prepared, dot_4bit_prepared_dynamic,
    dot_4bit_prepared_generic, dot_i8_i8, dot_i8_i8_generic, dot_i8_i8_maddubs,
    dot_i8x4_i8_maddubs, dot_packed_4bit, dot_product_f16_bytes_f32, dot_product_f16_bytes_f32_fma,
    dot_product_f16_bytes_f32_generic, dot_product_f16_f32, dot_product_f16_f32_generic,
    dot_product_f32_bytes_f32, dot_product_f32_bytes_f32_generic, dot_product_f32_f32,
    dot_product_f32_f32_generic, encode_f32_to_f16_extend, encode_f32_to_f16_extend_generic,
    maddubs_query_bias, pack_f16_le_bytes_to_4bit, pack_f16_le_bytes_to_4bit_explicit_round,
    pack_f16_le_bytes_to_4bit_generic, pack_f16_le_bytes_to_4bit_scalar_pack,
    pack_f16_slab_to_4bit, pack_f16_slab_to_4bit_generic, prepare_4bit_query,
    quantize_f16_le_bytes_to_i8, quantize_f16_le_bytes_to_i8_generic, quantize_f16_slab_to_i8,
    quantize_f16_slab_to_i8_generic,
};
pub use two_tier::{
    TwoTierIndex, TwoTierIndexBuilder, TwoTierIndexPaths, VECTOR_INDEX_FALLBACK_FILENAME,
    VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};
#[cfg(feature = "ann")]
pub use two_tier::{VECTOR_ANN_FAST_FILENAME, VECTOR_ANN_QUALITY_FILENAME};
pub use wal::{
    CompactionStats, StrictWalImage, StrictWalInspection, WalConfig, inspect_wal_strict,
    wal_path_for,
};
pub use warmup::{AdaptiveConfig, HeatMap, WarmUpConfig, WarmUpResult, WarmUpStrategy};

/// Magic bytes at the start of every FSVI file.
pub const FSVI_MAGIC: [u8; 4] = *b"FSVI";

/// Supported FSVI format version.
pub const FSVI_VERSION: u16 = 1;

/// Identity-complete immutable FSVI format version.
pub const FSVI_V2_VERSION: u16 = 2;

/// Schema for the identity-binding envelope inside an FSVI v2 header.
pub const FSVI_V2_IDENTITY_BINDING_SCHEMA: u16 = 1;

const FSVI_V2_FIXED_PREFIX_BYTES: usize = 332;
const FSVI_V2_MIN_HEADER_BYTES: usize = FSVI_V2_FIXED_PREFIX_BYTES + 4;
const FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES: usize = 1024 * 1024;
const FSVI_V2_MAX_HEADER_BYTES: usize =
    3 * FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES + FSVI_V2_MIN_HEADER_BYTES;
const FSVI_V1_MAX_HEADER_BYTES: usize = 4 + 2 + 2 + 65_535 + 2 + 65_535 + 4 + 1 + 3 + 8 + 8 + 4;
const SHA256_BYTES: usize = 32;
#[cfg(test)]
const FSVI_V2_BUNDLE_FINGERPRINT_OFFSET: usize = 76;
#[cfg(test)]
const FSVI_V2_SPACE_FINGERPRINT_OFFSET: usize = 108;
#[cfg(test)]
const FSVI_V2_PRODUCER_FINGERPRINT_OFFSET: usize = 140;
#[cfg(test)]
const FSVI_V2_INPUT_FINGERPRINT_OFFSET: usize = 172;
#[cfg(test)]
const FSVI_V2_STORAGE_FINGERPRINT_OFFSET: usize = 204;
#[cfg(test)]
const FSVI_V2_GENERATION_FINGERPRINT_OFFSET: usize = 236;
#[cfg(test)]
const FSVI_V2_DOCSET_DIGEST_OFFSET: usize = 268;
#[cfg(test)]
const FSVI_V2_VECTOR_DIGEST_OFFSET: usize = 300;
const EMBEDDING_BUNDLE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.embedding-bundle.v1";
const EMBEDDING_SPACE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.embedding-space.v1";
const VECTOR_STORAGE_CANONICAL_DOMAIN: &[u8] = b"frankensearch.vector-storage.v1";
const ORDERED_DOCSET_DIGEST_DOMAIN: &[u8] = b"frankensearch.fsvi-v2.ordered-live-docset.v1";
const VECTOR_CONTENT_DIGEST_DOMAIN: &[u8] = b"frankensearch.fsvi-v2.vector-content.v1";
const FSVI_WITNESS_SCHEMA_V1: u16 = 1;

const RECORD_SIZE_BYTES: usize = 16;
const VECTOR_ALIGN_BYTES: u64 = 64;
const RECORD_FLAG_TOMBSTONE: u16 = 0x0001;
const TOMBSTONE_VACUUM_THRESHOLD: f64 = 0.20;

/// Vector element quantization stored in the FSVI slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Quantization {
    /// Full-precision float32.
    F32 = 0,
    /// Half-precision float16.
    F16 = 1,
}

/// Valid record-state bits in an identity-complete FSVI v2 record.
///
/// The all-zero value is a live row. Bit 0 is a retained tombstone. Every
/// other bit is reserved and rejected during admission, so callers never need
/// to guess how an unknown flag should affect search or ANN construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FsviRecordFlags(u16);

impl FsviRecordFlags {
    /// A searchable live row.
    pub const LIVE: Self = Self(0);
    /// A retained, non-searchable tombstone row.
    pub const TOMBSTONE: Self = Self(RECORD_FLAG_TOMBSTONE);

    /// Decode a validated raw FSVI v2 flag word.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] if any reserved bit is set.
    pub fn from_bits(bits: u16) -> SearchResult<Self> {
        if bits & !RECORD_FLAG_TOMBSTONE != 0 {
            return Err(index_corrupted(
                Path::new("<owned-fsvi-v2>"),
                format!("unsupported FSVI v2 record flags {bits:#06x}"),
            ));
        }
        Ok(Self(bits))
    }

    /// Exact on-disk bit representation.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Whether this row is a retained tombstone.
    #[must_use]
    pub const fn is_tombstone(self) -> bool {
        self.0 & RECORD_FLAG_TOMBSTONE != 0
    }

    /// Whether this row participates in exact search and the ordered live
    /// document-set digest.
    #[must_use]
    pub const fn is_live(self) -> bool {
        !self.is_tombstone()
    }
}

impl Quantization {
    pub(crate) fn from_wire(value: u8, path: &Path) -> SearchResult<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            _ => Err(index_corrupted(
                path,
                format!("unsupported quantization byte: {value}"),
            )),
        }
    }

    const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Exact caller-owned identity required to write or admit one FSVI v2 artifact.
///
/// Construction validates the complete frozen embedding identity, requires
/// canonical `fsvi-v2` little-endian storage, and binds a full-width immutable
/// generation. Human-readable model names and vector dimensions are never
/// sufficient to construct this value.
#[derive(Clone, PartialEq, Eq)]
pub struct FsviV2IdentityBinding {
    generation: ArtifactGenerationIdentityV1,
    frozen_identity: FrozenEmbeddingIdentityBundleV1,
    space_canonical_bytes: Vec<u8>,
    storage_canonical_bytes: Vec<u8>,
    bundle_fingerprint: [u8; SHA256_BYTES],
    space_fingerprint: [u8; SHA256_BYTES],
    producer_fingerprint: [u8; SHA256_BYTES],
    input_fingerprint: [u8; SHA256_BYTES],
    storage_fingerprint: [u8; SHA256_BYTES],
    generation_fingerprint: [u8; SHA256_BYTES],
    dimension: usize,
    quantization: Quantization,
}

impl fmt::Debug for FsviV2IdentityBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FsviV2IdentityBinding")
            .field("generation", &self.generation)
            .field("dimension", &self.dimension)
            .field("quantization", &self.quantization)
            .field(
                "bundle_fingerprint",
                &fingerprint_hex(&self.bundle_fingerprint),
            )
            .field(
                "canonical_bundle_bytes",
                &self.frozen_identity.canonical_bytes.len(),
            )
            .field("canonical_space_bytes", &self.space_canonical_bytes.len())
            .field(
                "canonical_storage_bytes",
                &self.storage_canonical_bytes.len(),
            )
            .finish_non_exhaustive()
    }
}

impl FsviV2IdentityBinding {
    /// Validate and freeze an exact FSVI v2 generation/storage binding.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the generation or frozen
    /// identity is invalid, storage is not exactly `fsvi-v2` little-endian,
    /// quantization is not F32/F16, or canonical identity bytes exceed the
    /// bounded header allowance.
    pub fn new(
        generation: ArtifactGenerationIdentityV1,
        frozen_identity: FrozenEmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        generation.validate()?;
        frozen_identity.validate()?;
        let storage = &frozen_identity.identity.storage;
        if storage.format != "fsvi-v2" {
            return Err(fsvi_v2_config_error(
                "storage.format",
                "must be exactly fsvi-v2",
            ));
        }
        if storage.endianness != "little-endian" {
            return Err(fsvi_v2_config_error(
                "storage.endianness",
                "must be exactly little-endian",
            ));
        }
        let quantization = quantization_from_identity(storage.quantization)?;
        let dimension = usize::try_from(storage.dimension)
            .map_err(|_| fsvi_v2_config_error("storage.dimension", "must fit in usize"))?;
        let space_canonical_bytes = frozen_identity.identity.space.canonical_bytes();
        let storage_canonical_bytes = storage.canonical_bytes();
        for (field, bytes) in [
            (
                "identity.canonical_bytes",
                frozen_identity.canonical_bytes.as_slice(),
            ),
            ("space.canonical_bytes", space_canonical_bytes.as_slice()),
            (
                "storage.canonical_bytes",
                storage_canonical_bytes.as_slice(),
            ),
        ] {
            if bytes.is_empty() || bytes.len() > FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES {
                return Err(fsvi_v2_config_error(
                    field,
                    "must be non-empty and no larger than 1 MiB",
                ));
            }
        }

        let bundle_fingerprint =
            decode_sha256_fingerprint("identity.fingerprint", &frozen_identity.fingerprint)?;
        let space_fingerprint = decode_sha256_fingerprint(
            "space.fingerprint",
            &frozen_identity.identity.space.fingerprint(),
        )?;
        let producer_fingerprint = decode_sha256_fingerprint(
            "producer.fingerprint",
            &frozen_identity.identity.producer.fingerprint(),
        )?;
        let input_fingerprint = decode_sha256_fingerprint(
            "input.fingerprint",
            &frozen_identity.identity.input.fingerprint(),
        )?;
        let storage_fingerprint =
            decode_sha256_fingerprint("storage.fingerprint", &storage.fingerprint())?;
        let generation_fingerprint =
            decode_sha256_fingerprint("generation.fingerprint", &generation.fingerprint())?;

        Ok(Self {
            generation,
            frozen_identity,
            space_canonical_bytes,
            storage_canonical_bytes,
            bundle_fingerprint,
            space_fingerprint,
            producer_fingerprint,
            input_fingerprint,
            storage_fingerprint,
            generation_fingerprint,
            dimension,
            quantization,
        })
    }

    /// Immutable artifact generation bound to this writer/reader.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.generation
    }

    /// Complete validated frozen embedding identity.
    #[must_use]
    pub const fn frozen_identity(&self) -> &FrozenEmbeddingIdentityBundleV1 {
        &self.frozen_identity
    }

    /// Stored vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Stored vector quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }
}

/// Identity and content bindings decoded from an FSVI v2 header.
#[derive(Clone, PartialEq, Eq)]
pub struct FsviV2IdentityMetadata {
    /// Full-width immutable artifact generation.
    pub generation: ArtifactGenerationIdentityV1,
    /// Exact canonical bytes of the complete embedding identity bundle.
    pub identity_bundle_canonical_bytes: Vec<u8>,
    /// Exact canonical bytes of the mathematical embedding-space identity.
    pub space_identity_canonical_bytes: Vec<u8>,
    /// Exact canonical bytes of the physical storage identity.
    pub storage_identity_canonical_bytes: Vec<u8>,
    /// SHA-256 of the complete identity bundle canonical bytes.
    pub identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the mathematical embedding space.
    pub space_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the embedding producer attestation.
    pub producer_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the outer embedding input contract.
    pub input_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the physical vector storage identity.
    pub storage_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the canonical full-width generation identity.
    pub generation_fingerprint: [u8; SHA256_BYTES],
    /// SHA-256 of the ordered live document identifiers.
    pub ordered_live_docset_digest: [u8; SHA256_BYTES],
    /// SHA-256 of the exact persisted vector slab bytes and shape.
    pub vector_content_digest: [u8; SHA256_BYTES],
    /// Exact byte length of the complete v2 header, including CRC.
    pub header_size: usize,
}

impl fmt::Debug for FsviV2IdentityMetadata {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FsviV2IdentityMetadata")
            .field("generation", &self.generation)
            .field(
                "identity_bundle_fingerprint",
                &fingerprint_hex(&self.identity_bundle_fingerprint),
            )
            .field(
                "space_fingerprint",
                &fingerprint_hex(&self.space_fingerprint),
            )
            .field(
                "storage_fingerprint",
                &fingerprint_hex(&self.storage_fingerprint),
            )
            .field(
                "ordered_live_docset_digest",
                &fingerprint_hex(&self.ordered_live_docset_digest),
            )
            .field(
                "vector_content_digest",
                &fingerprint_hex(&self.vector_content_digest),
            )
            .field("header_size", &self.header_size)
            .finish_non_exhaustive()
    }
}

/// Serializable, redacted proof of one fully validated immutable FSVI v2 byte
/// image.
///
/// The witness contains no document identifiers or pathnames. Its whole-image
/// SHA-256 covers every persisted byte, while the remaining fields make the
/// semantic, storage, generation, membership, and vector-content bindings
/// directly auditable. Equality is exact and is the only supported way to
/// authorize a reopen of an already witnessed generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FsviV2Witness {
    /// Witness schema version.
    pub schema_version: u16,
    /// FSVI format version covered by this witness.
    pub fsvi_version: u16,
    /// Exact byte length of the owned image.
    pub byte_len: u64,
    /// SHA-256 of every byte in the owned image.
    pub whole_image_sha256: [u8; SHA256_BYTES],
    /// Full-width immutable generation.
    pub generation: ArtifactGenerationIdentityV1,
    /// Complete frozen embedding-bundle fingerprint.
    pub identity_bundle_fingerprint: [u8; SHA256_BYTES],
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: [u8; SHA256_BYTES],
    /// Producer-attestation fingerprint.
    pub producer_fingerprint: [u8; SHA256_BYTES],
    /// Outer embedding-input-contract fingerprint.
    pub input_fingerprint: [u8; SHA256_BYTES],
    /// Physical vector-storage fingerprint.
    pub storage_fingerprint: [u8; SHA256_BYTES],
    /// Full-width generation fingerprint.
    pub generation_fingerprint: [u8; SHA256_BYTES],
    /// Ordered live-document-set digest from the validated header.
    pub ordered_live_docset_digest: [u8; SHA256_BYTES],
    /// Exact vector-slab digest from the validated header.
    pub vector_content_digest: [u8; SHA256_BYTES],
    /// Persisted vector dimension.
    pub dimension: u32,
    /// Persisted vector quantization.
    pub quantization: Quantization,
    /// Number of physical rows, including tombstones.
    pub record_count: u64,
    /// Number of searchable live rows.
    pub live_count: u64,
    /// Number of retained tombstone rows.
    pub tombstone_count: u64,
}

/// Machine-matchable reason an immutable pathname snapshot was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsviSnapshotRejectionReason {
    /// This target cannot provide a safe no-atime open and therefore cannot
    /// prove literal side-effect freedom.
    NoAtimeUnsupported,
    /// The final index path is a symbolic link.
    SymbolicLink,
    /// The final index path is not a regular file.
    NotRegularFile,
    /// The inode has more than one hard link and can be mutated through an
    /// unobserved alias.
    HardLinked,
    /// The file identity or metadata changed while bytes were being owned.
    PathChangedDuringRead,
    /// The containing directory changed while the publication snapshot was
    /// being owned.
    DirectoryChangedDuringRead,
    /// Any WAL directory entry exists beside a purported published FSVI v2
    /// generation, including an empty or otherwise valid sidecar.
    PublishedWalPresent,
    /// A reopen produced a different complete witness.
    WitnessMismatch,
}

/// Typed pathname/publication rejection without document identifiers or
/// unbounded path-derived diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FsviSnapshotRejected {
    /// Stable reason for programmatic recovery routing.
    pub reason: FsviSnapshotRejectionReason,
    /// Bounded redacted diagnostic.
    pub detail: String,
}

/// Why owner-backed ANN is unavailable in this API slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsviAnnDisabledReason {
    /// Existing HNSW APIs accept a mutable/path-opened [`VectorIndex`] and
    /// therefore cannot prove that graph validation consumed this owner's
    /// exact byte image.
    OwnerBoundAdapterUnavailable,
}

/// Typed ANN disposition for one immutable FSVI owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "reason")]
pub enum FsviAnnAdmission {
    /// ANN must not load or rebuild; callers must use the owner's exact-search
    /// path until an owner-bound graph receipt and adapter are available.
    Disabled(FsviAnnDisabledReason),
}

/// Why a recognized artifact must be rebuilt rather than adopted or relabeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsviReindexReason {
    /// FSVI v1 carries no complete embedding/storage/generation identity.
    LegacyUnidentified,
    /// Persisted embedding identity differs from the caller's exact identity.
    IdentityMismatch,
    /// Persisted immutable generation differs from the caller's generation.
    GenerationMismatch,
    /// Persisted physical storage contract differs from the caller's contract.
    StorageMismatch,
}

/// Typed, actionable outcome for a recognized artifact that cannot be admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsviReindexRequired {
    /// Machine-matchable reason.
    pub reason: FsviReindexReason,
    /// Format version found in the artifact.
    pub found_version: u16,
    /// Bounded diagnostic that never suggests relabeling/adoption.
    pub detail: String,
}

/// Typed outcome for an artifact produced by a newer FSVI schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsviUpgradeRequired {
    /// Newer format version found in the artifact.
    pub found_version: u16,
    /// Newest format version this build can inspect and admit.
    pub supported_version: u16,
}

/// Non-mutating format inspection result.
///
/// Structural corruption remains [`SearchError::IndexCorrupted`], so callers
/// cannot confuse damaged bytes with either a source reindex or software
/// upgrade action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsviInspection {
    /// Identity-complete v2 header; full content admission still recomputes
    /// docset/vector digests in [`VectorIndex::open_admitted_v2`].
    V2IdentityComplete(Box<VectorMetadata>),
    /// Recognized legacy bytes that must be rebuilt from source.
    ReindexRequired(FsviReindexRequired),
    /// A newer schema that requires newer reader software.
    UpgradeRequired(FsviUpgradeRequired),
}

/// Error returned by exact FSVI v2 admission.
#[derive(Debug)]
pub enum FsviAdmissionError {
    /// Recognized bytes require a source reindex.
    ReindexRequired(FsviReindexRequired),
    /// Newer bytes require newer reader software.
    UpgradeRequired(FsviUpgradeRequired),
    /// The path/publication snapshot could not be proven immutable and
    /// side-effect-free.
    SnapshotRejected(FsviSnapshotRejected),
    /// I/O, not-found, or actual corruption from the index layer.
    Index(SearchError),
}

impl fmt::Display for FsviAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReindexRequired(required) => write!(
                formatter,
                "FSVI source reindex required ({:?}): {}",
                required.reason, required.detail
            ),
            Self::UpgradeRequired(required) => write!(
                formatter,
                "FSVI reader upgrade required: found v{}, supported through v{}",
                required.found_version, required.supported_version
            ),
            Self::SnapshotRejected(rejected) => write!(
                formatter,
                "FSVI immutable snapshot rejected ({:?}): {}",
                rejected.reason, rejected.detail
            ),
            Self::Index(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FsviAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Index(error) => Some(error),
            Self::ReindexRequired(_) | Self::UpgradeRequired(_) | Self::SnapshotRejected(_) => None,
        }
    }
}

impl From<SearchError> for FsviAdmissionError {
    fn from(error: SearchError) -> Self {
        Self::Index(error)
    }
}

/// Parsed metadata from an FSVI file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorMetadata {
    /// On-disk FSVI format version.
    pub fsvi_version: u16,
    /// Stable embedder id used to build the index.
    pub embedder_id: String,
    /// Model revision identifier (e.g. pinned commit hash).
    pub embedder_revision: String,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Stored quantization.
    pub quantization: Quantization,
    /// Compaction generation counter (0-255) used for stale WAL detection.
    pub compaction_gen: u8,
    /// Number of records in the index.
    pub record_count: usize,
    /// Byte offset to the aligned vector slab.
    pub vectors_offset: u64,
    /// Complete v2 identity/content bindings, absent for legacy v1 bytes.
    pub identity_v2: Option<FsviV2IdentityMetadata>,
}

/// Statistics returned by [`VectorIndex::vacuum`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VacuumStats {
    /// Records in the main index before vacuum.
    pub records_before: usize,
    /// Records in the main index after vacuum.
    pub records_after: usize,
    /// Tombstoned records removed by vacuum.
    pub tombstones_removed: usize,
    /// Approximate number of bytes reclaimed in the main index file.
    pub bytes_reclaimed: usize,
    /// Time taken by the vacuum operation.
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordEntry {
    pub(crate) doc_id_hash: u64,
    pub(crate) doc_id_offset: u32,
    pub(crate) doc_id_len: u16,
    pub(crate) flags: u16,
}

#[derive(Debug)]
pub(crate) enum VectorIndexData {
    Mutable(MmapMut),
    Immutable(Arc<[u8]>),
}

impl VectorIndexData {
    fn write_and_flush(&mut self, offset: usize, bytes: &[u8]) -> SearchResult<()> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| index_corrupted(Path::new("<fsvi>"), "write range overflow"))?;
        match self {
            Self::Mutable(mapping) => {
                if end > mapping.len() {
                    return Err(index_corrupted(
                        Path::new("<fsvi>"),
                        "write range extends beyond mapped data",
                    ));
                }
                mapping[offset..end].copy_from_slice(bytes);
                mapping
                    .flush_range(offset, bytes.len())
                    .map_err(SearchError::Io)
            }
            Self::Immutable(_) => Err(SearchError::InvalidConfig {
                field: "fsvi_v2.mutation".to_owned(),
                value: "immutable-owned-image".to_owned(),
                reason: "the sealed FSVI v2 owner exposes no writable backing store".to_owned(),
            }),
        }
    }
}

impl Deref for VectorIndexData {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mutable(mapping) => mapping,
            Self::Immutable(bytes) => bytes,
        }
    }
}

#[derive(Debug)]
pub struct VectorIndex {
    pub(crate) path: PathBuf,
    pub(crate) data: VectorIndexData,
    pub(crate) metadata: VectorMetadata,
    pub(crate) records_offset: usize,
    pub(crate) strings_offset: usize,
    pub(crate) vectors_offset: usize,
    /// WAL entries for incremental updates (empty if no WAL exists).
    pub(crate) wal_entries: Vec<wal::WalEntry>,
    /// WAL configuration.
    wal_config: WalConfig,
    /// Lazily-built int8 quantization of the (contiguous, F16) main vector region,
    /// for the optional int8 two-pass scan (`search_top_k_int8_two_pass`). Built on
    /// first two-pass use, so exact-only callers never pay the quantization cost or
    /// its footprint. One corpus-wide max-abs scale (see `quantize_f16_bytes_to_i8`).
    pub(crate) vectors_i8: OnceLock<Vec<i8>>,
    /// Lazily-built packed signed-4-bit quantization of the (contiguous, F16) main
    /// vector region — 2 dims/byte, `dim.div_ceil(2)` bytes per vector (half the int8
    /// slab) — for the optional 4-bit two-pass scan (`search_top_k_4bit_two_pass`).
    /// 16 levels stay lossless at mult≈5 on realistic data while halving pass-1
    /// bandwidth vs int8. Built on first 4-bit-two-pass use; other callers never pay.
    pub(crate) vectors_nibbles: OnceLock<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FsviPublicationState {
    OwnedBytesOnly,
    PublishedWalAbsent,
}

/// Sealed owner of one fully validated immutable FSVI v2 byte image.
///
/// Construction copies pathname input into an owned [`Arc`] and then parses,
/// hashes, validates, searches, and serves rows exclusively from that exact
/// allocation. No file mapping or writable slice is retained. Replacing,
/// renaming, linking, or mutating the source pathname after construction cannot
/// alter this owner's search results.
///
/// The owner is safe for concurrent shared reads. A [`ValidatedFsviRowSource`]
/// borrows the owner, so Rust's lifetime rules prevent the row source from
/// outliving or becoming detached from the byte image it describes. The owner
/// intentionally exposes no conversion into a mutable/path-opened
/// [`VectorIndex`].
///
/// ```compile_fail
/// use frankensearch_index::ValidatedFsviBytes;
///
/// fn mutate(owner: &mut ValidatedFsviBytes) {
///     owner.append("forbidden", &[1.0, 0.0]);
/// }
/// ```
pub struct ValidatedFsviBytes {
    bytes: Arc<[u8]>,
    index: VectorIndex,
    witness: FsviV2Witness,
    publication_state: FsviPublicationState,
}

impl fmt::Debug for ValidatedFsviBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviBytes")
            .field("witness", &self.witness)
            .field("publication_state", &self.publication_state)
            .finish_non_exhaustive()
    }
}

/// Sealed read-only row source whose lifetime is tied to one
/// [`ValidatedFsviBytes`] owner.
///
/// ```compile_fail
/// use frankensearch_index::{ValidatedFsviBytes, ValidatedFsviRowSource};
///
/// fn detach<'a>(owner: ValidatedFsviBytes) -> ValidatedFsviRowSource<'a> {
///     owner.row_source()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct ValidatedFsviRowSource<'owner> {
    owner: &'owner ValidatedFsviBytes,
}

impl fmt::Debug for ValidatedFsviRowSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviRowSource")
            .field("witness", self.owner.witness())
            .finish_non_exhaustive()
    }
}

/// One immutable physical FSVI row borrowed from its sealed owner.
pub struct ValidatedFsviRow<'owner> {
    physical_index: usize,
    doc_id: &'owner str,
    vector_bytes: &'owner [u8],
    flags: FsviRecordFlags,
}

impl fmt::Debug for ValidatedFsviRow<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ValidatedFsviRow")
            .field("physical_index", &self.physical_index)
            .field("doc_id", &"<redacted>")
            .field("vector_bytes", &self.vector_bytes.len())
            .field("flags", &self.flags)
            .finish()
    }
}

impl ValidatedFsviRow<'_> {
    /// Physical row position in the exact owned image.
    #[must_use]
    pub const fn physical_index(&self) -> usize {
        self.physical_index
    }

    /// Borrow the validated UTF-8 document identifier.
    #[must_use]
    pub const fn doc_id(&self) -> &str {
        self.doc_id
    }

    /// Borrow the exact persisted vector bytes.
    #[must_use]
    pub const fn vector_bytes(&self) -> &[u8] {
        self.vector_bytes
    }

    /// Validated LIVE/TOMBSTONE state.
    #[must_use]
    pub const fn flags(&self) -> FsviRecordFlags {
        self.flags
    }
}

impl ValidatedFsviRowSource<'_> {
    /// Exact witness of the byte owner backing this row source.
    #[must_use]
    pub const fn witness(&self) -> &FsviV2Witness {
        self.owner.witness()
    }

    /// Persisted vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.owner.dimension()
    }

    /// Persisted storage quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.owner.quantization()
    }

    /// Number of physical rows, including tombstones.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.owner.record_count()
    }

    /// Number of searchable live rows.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.owner.live_count()
    }

    /// Number of retained tombstone rows.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        self.owner.tombstone_count()
    }

    /// Full-width immutable generation.
    #[must_use]
    pub const fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.owner.witness.generation
    }

    /// Ordered live-document-set digest.
    #[must_use]
    pub const fn ordered_live_docset_digest(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.ordered_live_docset_digest
    }

    /// Exact vector-content digest.
    #[must_use]
    pub const fn vector_content_digest(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.vector_content_digest
    }

    /// Mathematical embedding-space fingerprint.
    #[must_use]
    pub const fn space_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.space_fingerprint
    }

    /// Complete frozen embedding-bundle fingerprint.
    #[must_use]
    pub const fn identity_bundle_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.identity_bundle_fingerprint
    }

    /// Embedding producer-attestation fingerprint.
    #[must_use]
    pub const fn producer_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.producer_fingerprint
    }

    /// Outer embedding-input-contract fingerprint.
    #[must_use]
    pub const fn input_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.input_fingerprint
    }

    /// Physical storage fingerprint.
    #[must_use]
    pub const fn storage_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.storage_fingerprint
    }

    /// Full-width generation fingerprint.
    #[must_use]
    pub const fn generation_fingerprint(&self) -> &[u8; SHA256_BYTES] {
        &self.owner.witness.generation_fingerprint
    }

    /// Borrow one validated physical row.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row.
    pub fn row(&self, index: usize) -> SearchResult<ValidatedFsviRow<'_>> {
        self.owner.row(index)
    }
}

impl ValidatedFsviBytes {
    /// Admit an already owned immutable byte image.
    ///
    /// This constructor validates content and identity but does not claim that
    /// a pathname publication had no WAL sidecar. Use [`Self::open_published`]
    /// for published-generation admission.
    ///
    /// # Errors
    ///
    /// Returns typed legacy/upgrade outcomes, exact identity mismatch outcomes,
    /// or structural corruption.
    pub fn from_arc(
        bytes: Arc<[u8]>,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        Self::from_arc_with_state(bytes, expected, FsviPublicationState::OwnedBytesOnly)
    }

    /// Open a published pathname into one immutable owned image.
    ///
    /// Linux and Android use `O_NOATIME | O_NOFOLLOW | O_CLOEXEC`; other
    /// targets fail closed because literal timestamp preservation cannot be
    /// established with the available safe standard-library API. The final
    /// file must be a single-link regular inode. File identity, length, mode,
    /// link count, all timestamps, and containing-directory identity are
    /// compared before and after both the read and full semantic validation.
    ///
    /// # Errors
    ///
    /// Returns a typed snapshot rejection for unsafe path topology, mutation,
    /// unsupported no-atime operation, witness drift, or any adjacent WAL
    /// entry. Identity/version/content errors retain their normal typed forms.
    pub fn open_published(
        path: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        let snapshot = PublishedFsviPathSnapshot::read(path)?;
        let owner = Self::from_arc_with_state(
            Arc::clone(&snapshot.bytes),
            expected,
            FsviPublicationState::PublishedWalAbsent,
        )?;
        snapshot.verify()?;
        Ok(owner)
    }

    /// Reopen a published generation and require byte-for-byte witness equality.
    ///
    /// # Errors
    ///
    /// Returns [`FsviSnapshotRejectionReason::WitnessMismatch`] if the complete
    /// newly validated witness differs in any field.
    pub fn reopen_exact(
        path: &Path,
        expected: &FsviV2IdentityBinding,
        witness: &FsviV2Witness,
    ) -> Result<Self, FsviAdmissionError> {
        let owner = Self::open_published(path, expected)?;
        if owner.witness != *witness {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::WitnessMismatch,
                "reopened bytes differ from the complete expected witness",
            ));
        }
        Ok(owner)
    }

    /// Publish one completed sibling generation from its exact sealed bytes.
    ///
    /// The completed sibling is admitted before the destination is touched and
    /// remains in place as evidence. Publication copies only the admitted
    /// owner's exact bytes into a fresh same-directory file, syncs that file,
    /// atomically replaces the destination, and syncs the directory. Any stale
    /// destination WAL is removed and directory-synced only after the v2 main
    /// file is durable. Therefore a crash cannot leave the old main file with
    /// its committed WAL silently discarded; the only intermediate state is a
    /// v2 main file beside a WAL, which strict admission rejects fail-closed.
    ///
    /// The returned owner is an exact witness-checked reopen of the published
    /// pathname after WAL removal. No mutable [`VectorIndex`] is created or
    /// exposed for the v2 generation.
    ///
    /// # Errors
    ///
    /// Returns a typed admission error when the completed sibling is not an
    /// exact identity-complete v2 generation, the paths are not distinct
    /// siblings, publication or durability fails, a stale WAL cannot be
    /// removed, or the final bytes differ from the pre-publication witness.
    pub fn publish_completed_sibling(
        destination: &Path,
        completed_sibling: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<Self, FsviAdmissionError> {
        Self::publish_completed_sibling_with_hooks(
            destination,
            completed_sibling,
            expected,
            |_| Ok(()),
            || Ok(()),
            || Ok(()),
            || Ok(()),
        )
    }

    fn publish_completed_sibling_with_hooks<B, P, A, R>(
        destination: &Path,
        completed_sibling: &Path,
        expected: &FsviV2IdentityBinding,
        before_temp_admission: B,
        before_replace: P,
        after_main_sync: A,
        before_final_reopen: R,
    ) -> Result<Self, FsviAdmissionError>
    where
        B: FnOnce(&Path) -> SearchResult<()>,
        P: FnOnce() -> SearchResult<()>,
        A: FnOnce() -> SearchResult<()>,
        R: FnOnce() -> SearchResult<()>,
    {
        let destination_parent = snapshot_parent_or_current(destination);
        let completed_parent = snapshot_parent_or_current(completed_sibling);
        let destination_parent_metadata =
            fs::symlink_metadata(destination_parent).map_err(SearchError::Io)?;
        let completed_parent_metadata =
            fs::symlink_metadata(completed_parent).map_err(SearchError::Io)?;
        if !destination_parent_metadata.file_type().is_dir()
            || !completed_parent_metadata.file_type().is_dir()
        {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "FSVI v2 publication requires real source and destination parent directories",
            ));
        }
        #[cfg(unix)]
        let same_parent = {
            let destination_parent_identity = stable_file_identity(&destination_parent_metadata);
            let completed_parent_identity = stable_file_identity(&completed_parent_metadata);
            destination_parent_identity.device == completed_parent_identity.device
                && destination_parent_identity.inode == completed_parent_identity.inode
        };
        #[cfg(not(unix))]
        let same_parent = fs::canonicalize(destination_parent)
            .and_then(|destination| {
                fs::canonicalize(completed_parent).map(|completed| destination == completed)
            })
            .map_err(SearchError::Io)?;
        if !same_parent {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed generation and destination must share one parent directory"
                    .to_owned(),
            }
            .into());
        }
        let destination_identity = match fs::symlink_metadata(destination) {
            Ok(metadata) => Some(validate_single_link_regular_file(&metadata)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(SearchError::Io(error).into()),
        };

        let completed_metadata_before =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_before =
            validate_single_link_regular_file(&completed_metadata_before)?;
        let completed_owner = Self::open_published(completed_sibling, expected)?;
        let completed_metadata_after =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_after =
            validate_single_link_regular_file(&completed_metadata_after)?;
        if completed_identity_before != completed_identity_after {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "completed sibling identity changed across sealed-byte admission",
            ));
        }
        #[cfg(unix)]
        let aliases_destination = destination_identity.is_some_and(|identity| {
            identity.device == completed_identity_after.device
                && identity.inode == completed_identity_after.inode
        });
        #[cfg(not(unix))]
        let aliases_destination = if destination_identity.is_some() {
            fs::canonicalize(destination)
                .and_then(|destination| {
                    fs::canonicalize(completed_sibling).map(|completed| destination == completed)
                })
                .map_err(SearchError::Io)?
        } else {
            false
        };
        if aliases_destination {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed sibling and destination resolve to the same file".to_owned(),
            }
            .into());
        }

        let wal_path = wal::wal_path_for(destination);
        let aliases_destination_wal = match fs::symlink_metadata(&wal_path) {
            Ok(metadata) => {
                #[cfg(unix)]
                {
                    let identity = stable_file_identity(&metadata);
                    identity.device == completed_identity_after.device
                        && identity.inode == completed_identity_after.inode
                }
                #[cfg(not(unix))]
                {
                    !metadata.file_type().is_symlink()
                        && fs::canonicalize(&wal_path)
                            .and_then(|wal| {
                                fs::canonicalize(completed_sibling)
                                    .map(|completed| wal == completed)
                            })
                            .map_err(SearchError::Io)?
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(SearchError::Io(error).into()),
        };
        if aliases_destination_wal {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.completed_sibling".to_owned(),
                value: completed_sibling.display().to_string(),
                reason: "completed sibling resolves to the destination WAL pathname".to_owned(),
            }
            .into());
        }
        let witness = completed_owner.witness.clone();
        let completed_bytes = Arc::clone(&completed_owner.bytes);

        let mut temporary =
            tempfile::NamedTempFile::new_in(destination_parent).map_err(SearchError::Io)?;
        temporary
            .as_file_mut()
            .write_all(completed_bytes.as_ref())
            .map_err(SearchError::Io)?;
        temporary.as_file().sync_all().map_err(SearchError::Io)?;
        before_temp_admission(temporary.path())?;
        let _validated_temporary = Self::reopen_exact(temporary.path(), expected, &witness)?;
        before_replace()?;
        temporary.persist(destination).map_err(|error| {
            SearchError::Io(std::io::Error::new(
                error.error.kind(),
                format!(
                    "failed to atomically publish sealed FSVI v2 bytes at '{}': {}",
                    destination.display(),
                    error.error
                ),
            ))
        })?;
        sync_parent_directory(destination)?;

        after_main_sync()?;

        let completed_metadata_before_wal =
            fs::symlink_metadata(completed_sibling).map_err(SearchError::Io)?;
        let completed_identity_before_wal =
            validate_single_link_regular_file(&completed_metadata_before_wal)?;
        if completed_identity_before_wal != completed_identity_after {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "completed sibling identity changed before destination WAL removal",
            ));
        }
        match fs::symlink_metadata(&wal_path) {
            Ok(_) => fs::remove_file(&wal_path).map_err(SearchError::Io)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(SearchError::Io(error).into()),
        }
        sync_parent_directory(&wal_path)?;

        before_final_reopen()?;
        Self::reopen_exact(destination, expected, &witness)
    }

    /// Redacted, serializable proof of the exact owned image.
    #[must_use]
    pub const fn witness(&self) -> &FsviV2Witness {
        &self.witness
    }

    /// Exact byte length retained by this owner.
    #[must_use]
    pub fn owned_byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether pathname construction proved WAL absence for this published
    /// generation. An owner created with [`Self::from_arc`] returns `false`.
    #[must_use]
    pub const fn published_wal_absent(&self) -> bool {
        matches!(
            self.publication_state,
            FsviPublicationState::PublishedWalAbsent
        )
    }

    /// Borrow a row source that cannot outlive this owner.
    #[must_use]
    pub const fn row_source(&self) -> ValidatedFsviRowSource<'_> {
        ValidatedFsviRowSource { owner: self }
    }

    /// ANN is explicitly disabled until HNSW load/rebuild accepts this sealed
    /// owner and binds its graph receipt to this exact witness.
    #[must_use]
    pub const fn ann_admission(&self) -> FsviAnnAdmission {
        FsviAnnAdmission::Disabled(FsviAnnDisabledReason::OwnerBoundAdapterUnavailable)
    }

    /// Exact top-k search over the owned image.
    ///
    /// This delegates to the normal index search implementation while its
    /// backing store is the owner's exact [`Arc`] allocation, preventing search
    /// and witness logic from drifting onto different bytes.
    ///
    /// # Errors
    ///
    /// Returns the normal exact-search errors for a query dimension mismatch,
    /// non-finite query values, or structurally invalid row data.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn frankensearch_core::filter::SearchFilter>,
    ) -> SearchResult<Vec<frankensearch_core::VectorHit>> {
        self.index.search_top_k(query, limit, filter)
    }

    /// Exact top-k with typed zero-signal classification.
    ///
    /// # Errors
    ///
    /// Returns the normal classified-search errors for a query dimension
    /// mismatch, non-finite query values, or structurally invalid row data.
    pub fn search_top_k_classified(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn frankensearch_core::filter::SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        self.index.search_top_k_classified(query, limit, filter)
    }

    /// Full parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &VectorMetadata {
        self.index.metadata()
    }

    /// Validated logical model id retained for diagnostics only.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        self.index.embedder_id()
    }

    /// Validated immutable model revision retained for diagnostics only.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        self.index.embedder_revision()
    }

    /// Always true for a successfully constructed sealed owner.
    #[must_use]
    pub const fn is_identity_admitted_v2(&self) -> bool {
        true
    }

    /// Complete v2 identity/content metadata.
    #[must_use]
    pub fn identity_v2(&self) -> &FsviV2IdentityMetadata {
        self.index
            .identity_v2()
            .expect("ValidatedFsviBytes construction requires FSVI v2 identity metadata")
    }

    /// Persisted vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.index.dimension()
    }

    /// Persisted storage quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.index.quantization()
    }

    /// Number of physical rows, including tombstones.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.index.record_count()
    }

    /// Number of searchable live rows.
    #[must_use]
    pub fn live_count(&self) -> usize {
        usize::try_from(self.witness.live_count).unwrap_or(usize::MAX)
    }

    /// Number of retained tombstone rows.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        usize::try_from(self.witness.tombstone_count).unwrap_or(usize::MAX)
    }

    /// Resolve a document id from the exact owned image.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or
    /// [`SearchError::IndexCorrupted`] if its validated string range cannot be
    /// decoded.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.index.doc_id_at(index)
    }

    /// Decode a row vector as f32.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or a
    /// decoding error if its persisted vector bytes are structurally invalid.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        self.index.vector_at_f32(index)
    }

    /// Find the first physical row with the requested document hash.
    #[must_use]
    pub fn find_index_by_doc_hash(&self, doc_id_hash: u64) -> Option<usize> {
        self.index.find_index_by_doc_hash(doc_id_hash)
    }

    /// Borrow one validated physical row.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an out-of-range row or
    /// [`SearchError::IndexCorrupted`] if its validated document range cannot
    /// be decoded.
    pub fn row(&self, index: usize) -> SearchResult<ValidatedFsviRow<'_>> {
        let entry = self.index.record_at(index)?;
        Ok(ValidatedFsviRow {
            physical_index: index,
            doc_id: self.index.doc_id_at(index)?,
            vector_bytes: self.index.vector_bytes(index)?,
            flags: FsviRecordFlags(entry.flags),
        })
    }

    #[cfg(test)]
    fn owner_and_search_share_allocation(&self) -> bool {
        matches!(
            &self.index.data,
            VectorIndexData::Immutable(search_bytes) if Arc::ptr_eq(&self.bytes, search_bytes)
        )
    }

    fn from_arc_with_state(
        bytes: Arc<[u8]>,
        expected: &FsviV2IdentityBinding,
        publication_state: FsviPublicationState,
    ) -> Result<Self, FsviAdmissionError> {
        const OWNED_PATH: &str = "<owned-fsvi-v2>";
        let path = Path::new(OWNED_PATH);
        if bytes.len() < 6 {
            return Err(index_corrupted(path, "magic and version are truncated").into());
        }
        if bytes[..4] != FSVI_MAGIC {
            return Err(index_corrupted(path, "invalid FSVI magic bytes").into());
        }
        let version = u16::from_le_bytes(
            bytes[4..6]
                .try_into()
                .expect("owned FSVI version field has fixed width"),
        );
        match version {
            FSVI_VERSION => {
                let _ = parse_header(path, &bytes)?;
                return Err(FsviAdmissionError::ReindexRequired(
                    FsviReindexRequired {
                        reason: FsviReindexReason::LegacyUnidentified,
                        found_version: FSVI_VERSION,
                        detail: "FSVI v1 has no complete embedding-space, storage, content, or full-width generation identity; rebuild from source into a separate v2 generation"
                            .to_owned(),
                    },
                ));
            }
            FSVI_V2_VERSION => {}
            found if found > FSVI_V2_VERSION => {
                return Err(FsviAdmissionError::UpgradeRequired(FsviUpgradeRequired {
                    found_version: found,
                    supported_version: FSVI_V2_VERSION,
                }));
            }
            found => {
                return Err(index_corrupted(
                    path,
                    format!("unsupported historical FSVI schema version {found}"),
                )
                .into());
            }
        }

        let (metadata, header_len) = parse_v2_header(path, &bytes)?;
        validate_expected_v2_binding(path, &metadata, expected)?;
        let (records_offset, strings_offset, vectors_offset) =
            validate_v2_layout_len(path, &metadata, header_len, bytes.len())?;
        let content = validate_v2_records_and_content(
            path,
            &bytes,
            &metadata,
            records_offset,
            strings_offset,
            vectors_offset,
        )?;
        let identity = metadata
            .identity_v2
            .as_ref()
            .ok_or_else(|| index_corrupted(path, "v2 metadata omitted identity bindings"))?;
        let byte_len = u64::try_from(bytes.len())
            .map_err(|_| index_corrupted(path, "owned image length does not fit in u64"))?;
        let dimension = u32::try_from(metadata.dimension)
            .map_err(|_| index_corrupted(path, "dimension does not fit in u32"))?;
        let record_count = u64::try_from(metadata.record_count)
            .map_err(|_| index_corrupted(path, "record count does not fit in u64"))?;
        let witness = FsviV2Witness {
            schema_version: FSVI_WITNESS_SCHEMA_V1,
            fsvi_version: FSVI_V2_VERSION,
            byte_len,
            whole_image_sha256: Sha256::digest(&bytes).into(),
            generation: identity.generation,
            identity_bundle_fingerprint: identity.identity_bundle_fingerprint,
            space_fingerprint: identity.space_fingerprint,
            producer_fingerprint: identity.producer_fingerprint,
            input_fingerprint: identity.input_fingerprint,
            storage_fingerprint: identity.storage_fingerprint,
            generation_fingerprint: identity.generation_fingerprint,
            ordered_live_docset_digest: identity.ordered_live_docset_digest,
            vector_content_digest: identity.vector_content_digest,
            dimension,
            quantization: metadata.quantization,
            record_count,
            live_count: content.live_count,
            tombstone_count: content.tombstone_count,
        };

        let warm_up_config = WarmUpConfig::from_env();
        if !matches!(warm_up_config.strategy, WarmUpStrategy::None) {
            let _ = warmup::warm_up_bytes(&bytes, header_len, &warm_up_config, None);
        }

        let index = VectorIndex {
            path: PathBuf::from(OWNED_PATH),
            data: VectorIndexData::Immutable(Arc::clone(&bytes)),
            metadata,
            records_offset,
            strings_offset,
            vectors_offset,
            wal_entries: Vec::new(),
            wal_config: WalConfig::default(),
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
        };
        Ok(Self {
            bytes,
            index,
            witness,
            publication_state,
        })
    }
}

impl VectorIndex {
    /// Inspect an FSVI header without adopting, relabeling, or mutating it.
    ///
    /// Valid v1 bytes are always reported as
    /// [`FsviInspection::ReindexRequired`] with
    /// [`FsviReindexReason::LegacyUnidentified`]. A future schema is
    /// [`FsviInspection::UpgradeRequired`]. Bad magic, truncation, malformed
    /// identity material, or CRC drift remains actual
    /// [`SearchError::IndexCorrupted`].
    ///
    /// # Errors
    ///
    /// Returns index-not-found/I/O errors or actual structural corruption.
    pub fn inspect(path: &Path) -> SearchResult<FsviInspection> {
        let (version, header) = read_header_for_inspection(path)?;
        match version {
            FSVI_VERSION => {
                let _ = parse_header(path, &header)?;
                Ok(FsviInspection::ReindexRequired(FsviReindexRequired {
                    reason: FsviReindexReason::LegacyUnidentified,
                    found_version: FSVI_VERSION,
                    detail: "FSVI v1 has no complete embedding-space, storage, content, or full-width generation identity; rebuild from source into a separate v2 generation"
                        .to_owned(),
                }))
            }
            FSVI_V2_VERSION => {
                let (metadata, _) = parse_v2_header(path, &header)?;
                Ok(FsviInspection::V2IdentityComplete(Box::new(metadata)))
            }
            found if found > FSVI_V2_VERSION => {
                Ok(FsviInspection::UpgradeRequired(FsviUpgradeRequired {
                    found_version: found,
                    supported_version: FSVI_V2_VERSION,
                }))
            }
            found => Err(index_corrupted(
                path,
                format!("unsupported historical FSVI schema version {found}"),
            )),
        }
    }

    /// Open and fully admit one immutable identity-complete FSVI v2 artifact
    /// into a sealed byte owner.
    ///
    /// The pathname is opened with no-atime and no-follow semantics, copied
    /// once into an [`Arc`], and checked against pre/post inode and directory
    /// identity. Header inspection, complete admission, witness hashing, exact
    /// search, and row access all consume that same allocation. Published
    /// admission rejects every WAL directory entry, including an empty or valid
    /// sidecar.
    ///
    /// # Errors
    ///
    /// Returns typed source-reindex or reader-upgrade outcomes for recognized
    /// incompatible formats. I/O and actual corruption are wrapped in
    /// [`FsviAdmissionError::Index`].
    pub fn open_admitted_v2(
        path: &Path,
        expected: &FsviV2IdentityBinding,
    ) -> Result<ValidatedFsviBytes, FsviAdmissionError> {
        ValidatedFsviBytes::open_published(path, expected)
    }

    /// Open an existing FSVI index from disk.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if the file does not exist and
    /// `SearchError::IndexCorrupted` when header/layout validation fails.
    #[allow(unsafe_code, clippy::too_many_lines)] // MmapMut::map_mut requires unsafe for memory-mapped I/O.
    pub fn open(path: &Path) -> SearchResult<Self> {
        if !path.exists() {
            return Err(SearchError::IndexNotFound {
                path: path.to_path_buf(),
            });
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(SearchError::Io)?;
        let data = unsafe { MmapMut::map_mut(&file).map_err(SearchError::Io)? };
        let (metadata, header_len) = parse_header(path, &data)?;

        let records_bytes = metadata
            .record_count
            .checked_mul(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record table size overflow"))?;
        let records_offset = header_len;
        let strings_offset = records_offset
            .checked_add(records_bytes)
            .ok_or_else(|| index_corrupted(path, "record table offset overflow"))?;
        let vectors_offset = usize::try_from(metadata.vectors_offset)
            .map_err(|_| index_corrupted(path, "vectors_offset does not fit in usize"))?;
        if vectors_offset < strings_offset {
            return Err(index_corrupted(
                path,
                "vectors_offset points inside the record table/string table region",
            ));
        }

        let vector_bytes = metadata
            .record_count
            .checked_mul(metadata.dimension)
            .and_then(|v| v.checked_mul(metadata.quantization.bytes_per_element()))
            .ok_or_else(|| index_corrupted(path, "vector slab size overflow"))?;
        let required_len = vectors_offset
            .checked_add(vector_bytes)
            .ok_or_else(|| index_corrupted(path, "vector slab end overflow"))?;
        if data.len() < required_len {
            return Err(index_corrupted(
                path,
                format!(
                    "truncated file: have {} bytes, need at least {} bytes",
                    data.len(),
                    required_len
                ),
            ));
        }

        let warm_up_config = WarmUpConfig::from_env();
        if !matches!(warm_up_config.strategy, WarmUpStrategy::None) {
            let warm_up = warmup::warm_up_bytes(&data, header_len, &warm_up_config, None);
            debug!(
                target: "frankensearch.warmup",
                path = %path.display(),
                strategy = %warm_up.strategy_name,
                pages_touched = warm_up.pages_touched,
                bytes_touched = warm_up.bytes_touched,
                budget_exhausted = warm_up.budget_exhausted,
                "index warm-up complete"
            );
        }

        // Load WAL entries if a sidecar file exists.
        let wal_path = wal::wal_path_for(path);
        let (wal_entries_raw, wal_compaction_gen, valid_len) =
            wal::read_wal(&wal_path, metadata.dimension, metadata.quantization)?;

        let mut deduped_wal = Vec::with_capacity(wal_entries_raw.len());
        let mut seen_ids = std::collections::HashSet::new();
        for entry in wal_entries_raw.into_iter().rev() {
            if seen_ids.insert(entry.doc_id.clone()) {
                deduped_wal.push(entry);
            }
        }
        deduped_wal.reverse();
        let mut wal_entries = deduped_wal;

        let is_stale = if valid_len > 0 {
            if wal_compaction_gen == 0 {
                metadata.compaction_gen > 0
            } else {
                let expected = next_generation(metadata.compaction_gen);
                wal_compaction_gen != expected
            }
        } else {
            false
        };

        if is_stale {
            tracing::warn!(
                path = %path.display(),
                main_gen = metadata.compaction_gen,
                wal_gen = wal_compaction_gen,
                "discarding stale/mismatched WAL entries and removing file"
            );
            wal_entries.clear();
            if wal_path.exists() {
                let _ = std::fs::remove_file(&wal_path);
            }
        } else if wal_path.exists() {
            let actual_len = std::fs::metadata(&wal_path).map_err(SearchError::Io)?.len();
            if actual_len > valid_len {
                tracing::warn!(
                    path = %wal_path.display(),
                    actual_len,
                    valid_len,
                    "truncating corrupted WAL trailer"
                );
                let file = OpenOptions::new()
                    .write(true)
                    .open(&wal_path)
                    .map_err(SearchError::Io)?;
                file.set_len(valid_len).map_err(SearchError::Io)?;
                file.sync_all().map_err(SearchError::Io)?;
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            data: VectorIndexData::Mutable(data),
            metadata,
            records_offset,
            strings_offset,
            vectors_offset,
            wal_entries,
            wal_config: WalConfig::default(),
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
        })
    }

    /// Create a writer that stores vectors as f16 with an empty revision string.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create(
        path: &Path,
        embedder_id: &str,
        dimension: usize,
    ) -> SearchResult<VectorIndexWriter> {
        Self::create_with_revision(path, embedder_id, "", dimension, Quantization::F16)
    }

    /// Create an immutable identity-complete FSVI v2 writer.
    ///
    /// Unlike the legacy v1 constructors, this API cannot be called with a
    /// display name, empty revision, or dimension-only compatibility claim.
    /// The exact validated identity and full-width generation are required
    /// before any writer is returned.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if the binding is invalid.
    pub fn create_v2(
        path: &Path,
        binding: FsviV2IdentityBinding,
    ) -> SearchResult<VectorIndexWriter> {
        let embedder_id = binding
            .frozen_identity
            .identity
            .space
            .logical_model_id
            .clone();
        let embedder_revision = binding
            .frozen_identity
            .identity
            .space
            .immutable_revision
            .clone();
        Ok(VectorIndexWriter {
            path: path.to_path_buf(),
            embedder_id,
            embedder_revision,
            dimension: binding.dimension,
            quantization: binding.quantization,
            compaction_gen: 0,
            records: Vec::new(),
            identity_v2: Some(binding),
        })
    }

    /// Replace any existing index generation with a durable empty F16 index.
    ///
    /// Unlike calling [`Self::create`] directly, this first removes and
    /// directory-syncs the incremental WAL sidecar. That ordering prevents a
    /// WAL from the replaced generation from being accepted by the new empty
    /// generation after a crash or restart.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the stale WAL cannot be removed durably, or the
    /// same validation and corruption errors as [`Self::create`] and
    /// [`Self::open`].
    pub fn replace_with_empty(
        path: &Path,
        embedder_id: &str,
        dimension: usize,
    ) -> SearchResult<Self> {
        let replacement_path = temporary_output_path(path);
        let writer = Self::create(&replacement_path, embedder_id, dimension)?;
        writer.finish()?;
        Self::install_replacement(path, &replacement_path)
    }

    /// Durably install a fully written FSVI generation over `path`.
    ///
    /// The replacement is validated before the destination is touched. The
    /// destination WAL is then removed and directory-synced before the main
    /// file is atomically replaced, preventing records from the prior
    /// generation from being replayed into the replacement after a crash.
    ///
    /// # Errors
    ///
    /// Returns an error when the replacement is invalid, has a live WAL, is
    /// the destination itself, or cannot be atomically installed and synced.
    pub fn install_replacement(path: &Path, replacement_path: &Path) -> SearchResult<Self> {
        if path == replacement_path {
            return Err(SearchError::InvalidConfig {
                field: "replacement_path".to_owned(),
                value: replacement_path.display().to_string(),
                reason: "replacement path must differ from destination".to_owned(),
            });
        }

        let replacement = Self::open(replacement_path)?;
        if replacement.wal_record_count() != 0 {
            return Err(SearchError::InvalidConfig {
                field: "replacement_path".to_owned(),
                value: replacement_path.display().to_string(),
                reason: "replacement generation must not have a live WAL".to_owned(),
            });
        }
        drop(replacement);

        let wal_path = wal::wal_path_for(path);
        wal::remove_wal(&wal_path)?;
        sync_parent_directory(&wal_path)?;

        let temporary = tempfile::TempPath::try_from_path(replacement_path.to_path_buf())?;
        if let Err(error) = temporary.persist(path) {
            let tempfile::PathPersistError { error, mut path } = error;
            path.disable_cleanup(true);
            return Err(SearchError::Io(error));
        }
        sync_parent_directory(path)?;
        Self::open(path)
    }

    /// Create a writer with explicit embedder revision and quantization.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when arguments are invalid
    /// (for example, zero dimension or oversized header fields).
    pub fn create_with_revision(
        path: &Path,
        embedder_id: &str,
        embedder_revision: &str,
        dimension: usize,
        quantization: Quantization,
    ) -> SearchResult<VectorIndexWriter> {
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        validate_header_string(embedder_id, "embedder_id")?;
        validate_header_string(embedder_revision, "embedder_revision")?;
        let _ = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: "dimension must fit in u32 for FSVI header encoding".to_owned(),
        })?;

        Ok(VectorIndexWriter {
            path: path.to_path_buf(),
            embedder_id: embedder_id.to_owned(),
            embedder_revision: embedder_revision.to_owned(),
            dimension,
            quantization,
            compaction_gen: 1,
            records: Vec::new(),
            identity_v2: None,
        })
    }

    /// Number of vectors in this index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.metadata.record_count
    }

    /// Embedding dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Embedder id stored in the index header.
    #[must_use]
    pub fn embedder_id(&self) -> &str {
        &self.metadata.embedder_id
    }

    /// Embedder revision stored in the index header.
    #[must_use]
    pub fn embedder_revision(&self) -> &str {
        &self.metadata.embedder_revision
    }

    /// Stored quantization.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.metadata.quantization
    }

    /// Full parsed metadata.
    #[must_use]
    pub const fn metadata(&self) -> &VectorMetadata {
        &self.metadata
    }

    /// Complete v2 identity/content metadata, absent for legacy v1 bytes.
    #[must_use]
    pub const fn identity_v2(&self) -> Option<&FsviV2IdentityMetadata> {
        self.metadata.identity_v2.as_ref()
    }

    /// Whether this index was opened through exact FSVI v2 admission.
    #[must_use]
    pub const fn is_identity_admitted_v2(&self) -> bool {
        self.metadata.fsvi_version == FSVI_V2_VERSION && self.metadata.identity_v2.is_some()
    }

    // ─── WAL / Incremental Update API ───────────────────────────────────

    /// Set the WAL configuration for incremental updates.
    pub const fn set_wal_config(&mut self, config: WalConfig) {
        self.wal_config = config;
    }

    /// Number of entries in the write-ahead log (pending compaction).
    #[must_use]
    pub const fn wal_record_count(&self) -> usize {
        self.wal_entries.len()
    }

    /// Return the document ids with live membership in either the compacted
    /// main index or the resident write-ahead log.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::IndexCorrupted`] if a live main-index document
    /// id cannot be decoded.
    pub fn live_doc_ids(&self) -> SearchResult<std::collections::HashSet<String>> {
        let mut ids = std::collections::HashSet::with_capacity(
            self.record_count().saturating_add(self.wal_record_count()),
        );
        for record_index in 0..self.record_count() {
            if !self.is_deleted(record_index) {
                ids.insert(self.doc_id_at(record_index)?.to_owned());
            }
        }
        ids.extend(self.wal_entries.iter().map(|entry| entry.doc_id.clone()));
        Ok(ids)
    }

    /// Iterate the WAL-resident rows as `(doc_id, embedding)` pairs.
    ///
    /// These are acknowledged appends that have not been compacted into the
    /// main slab yet; any full rebuild that merges "the previous index"
    /// must include them or it silently drops durable writes. The resident
    /// set is already deduplicated (last write wins per doc ID).
    pub fn wal_records(&self) -> impl Iterator<Item = (&str, &[f32])> {
        self.wal_entries
            .iter()
            .map(|entry| (entry.doc_id.as_str(), entry.embedding.as_slice()))
    }

    /// Whether the WAL is large enough that compaction is recommended.
    ///
    /// Returns `true` when the WAL exceeds either the absolute threshold
    /// or the ratio threshold relative to the main index size.
    #[must_use]
    pub fn needs_compaction(&self) -> bool {
        if self.wal_entries.is_empty() {
            return false;
        }
        if self.wal_entries.len() >= self.wal_config.compaction_threshold {
            return true;
        }
        if self.record_count() > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.wal_entries.len() as f64 / self.record_count() as f64;
            // NaN compaction_ratio makes >= always false, silently disabling
            // ratio-based compaction. Fall back to the default.
            let threshold = if self.wal_config.compaction_ratio.is_finite() {
                self.wal_config.compaction_ratio
            } else {
                0.10
            };
            if ratio >= threshold {
                return true;
            }
        }
        false
    }

    /// Tombstone (soft-delete) a document by `doc_id`.
    ///
    /// Returns `Ok(true)` when a live record was marked deleted, and `Ok(false)`
    /// when the document does not exist or is already tombstoned.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem write/sync failures and
    /// `SearchError::IndexCorrupted` if the on-disk record table is malformed.
    pub fn soft_delete(&mut self, doc_id: &str) -> SearchResult<bool> {
        self.soft_delete_batch(&[doc_id]).map(|count| count > 0)
    }

    /// Tombstone a batch of document ids.
    ///
    /// Returns the number of records that transitioned from live -> deleted.
    ///
    /// # Errors
    ///
    /// Returns the first IO/corruption error encountered while updating flags.
    pub fn soft_delete_batch(&mut self, doc_ids: &[&str]) -> SearchResult<usize> {
        self.ensure_legacy_mutation_format("soft_delete_batch")?;
        let mut deleted = 0usize;
        let mut wal_changed = false;

        // Track modified main index entries for potential rollback
        let mut modified_main_entries = Vec::new();

        // Use a fast lookup for WAL entries to delete
        let mut to_delete_set = std::collections::HashSet::with_capacity(doc_ids.len());
        for &id in doc_ids {
            to_delete_set.insert(id);
        }

        // 1. Mark all matching records in the main index as tombstoned.
        for &doc_id in doc_ids {
            let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
            if let Some(mut index) = self.find_first_hash_match(doc_id_hash)? {
                while index > 0 {
                    let prev = self.record_at(index - 1)?;
                    if prev.doc_id_hash != doc_id_hash {
                        break;
                    }
                    index -= 1;
                }

                for candidate in index..self.record_count() {
                    let entry = self.record_at(candidate)?;
                    if entry.doc_id_hash != doc_id_hash {
                        break;
                    }
                    if !is_tombstoned_flags(entry.flags) {
                        let candidate_doc_id = self.doc_id_at(candidate)?;
                        if candidate_doc_id == doc_id {
                            let flags = entry.flags | RECORD_FLAG_TOMBSTONE;
                            self.set_record_flags(candidate, flags)?;
                            modified_main_entries.push((candidate, entry.flags));
                            deleted += 1;
                        }
                    }
                }
            }
        }

        // 2. Remove all matching records from WAL entries.
        let original_wal_len = self.wal_entries.len();
        let filtered: Vec<wal::WalEntry> = self
            .wal_entries
            .iter()
            .filter(|entry| !to_delete_set.contains(entry.doc_id.as_str()))
            .cloned()
            .collect();

        let prev_wal = if filtered.len() < original_wal_len {
            deleted += original_wal_len - filtered.len();
            wal_changed = true;
            std::mem::replace(&mut self.wal_entries, filtered)
        } else {
            Vec::new()
        };

        // 3. Rewrite WAL sidecar once if anything was removed.
        if wal_changed {
            if let Err(err) = self.rewrite_wal_sidecar() {
                self.wal_entries = prev_wal;
                // Rollback main index modifications
                for (candidate, original_flags) in modified_main_entries {
                    if let Err(rollback_err) = self.set_record_flags(candidate, original_flags) {
                        tracing::error!(
                            error = %rollback_err,
                            candidate,
                            "failed to rollback main index flag during soft_delete_batch failure"
                        );
                    }
                }
                tracing::error!(
                    error = %err,
                    "failed to rewrite WAL sidecar during batch delete"
                );
                return Err(err);
            }
        }

        Ok(deleted)
    }

    /// Whether the record at `record_index` is tombstoned.
    #[must_use]
    pub fn is_deleted(&self, record_index: usize) -> bool {
        matches!(
            self.record_at(record_index),
            Ok(entry) if is_tombstoned_flags(entry.flags)
        )
    }

    /// Number of tombstoned records in the main index.
    #[must_use]
    pub fn tombstone_count(&self) -> usize {
        (0..self.record_count())
            .filter(|&index| self.is_deleted(index))
            .count()
    }

    /// Number of live (non-tombstoned) records in the main index.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.record_count().saturating_sub(self.tombstone_count())
    }

    /// Whether the stored vector at `record_index` is usable for similarity
    /// ranking: every component finite and the norm non-zero.
    ///
    /// Undecodable records are reported as unusable rather than erroring —
    /// a vector that cannot be read cannot contribute signal.
    #[must_use]
    pub fn is_vector_usable(&self, record_index: usize) -> bool {
        self.vector_at_f32(record_index)
            .is_ok_and(|vector| vector_signal_usable(&vector))
    }

    /// Compute the zero-signal census for this index generation.
    ///
    /// O(n·dim): decodes every live vector once. Callers classify empty
    /// search results with it lazily — an empty result costs one extra pass
    /// comparable to the scan that just ran, and the hot (non-empty) path
    /// pays nothing.
    #[must_use]
    pub fn zero_signal_state(&self) -> ZeroSignalState {
        let record_count = self.record_count();
        let mut tombstone_count = 0usize;
        let mut usable_vector_count = 0usize;
        for index in 0..record_count {
            if self.is_deleted(index) {
                tombstone_count += 1;
            } else if self.is_vector_usable(index) {
                usable_vector_count += 1;
            }
        }
        ZeroSignalState {
            record_count,
            live_count: record_count - tombstone_count,
            tombstone_count,
            wal_count: self.wal_record_count(),
            usable_vector_count,
        }
    }

    /// Fraction of records that are tombstoned (`tombstones / record_count`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tombstone_ratio(&self) -> f64 {
        if self.record_count() == 0 {
            return 0.0;
        }
        self.tombstone_count() as f64 / self.record_count() as f64
    }

    /// Whether the tombstone ratio exceeds the default vacuum threshold.
    #[must_use]
    pub fn needs_vacuum(&self) -> bool {
        self.tombstone_ratio() > TOMBSTONE_VACUUM_THRESHOLD
    }

    /// Rewrite the main index file without tombstoned records.
    ///
    /// WAL entries are preserved and reloaded after the rewrite.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::IndexCorrupted` for malformed data.
    pub fn vacuum(&mut self) -> SearchResult<VacuumStats> {
        self.ensure_legacy_mutation_format("vacuum")?;
        let start = Instant::now();
        let records_before = self.record_count();
        let bytes_before = self.data.len();
        let tombstones_before = self.tombstone_count();

        if records_before == 0 || tombstones_before == 0 {
            return Ok(VacuumStats {
                records_before,
                records_after: records_before,
                tombstones_removed: 0,
                bytes_reclaimed: 0,
                duration: start.elapsed(),
            });
        }

        // Collect live entries from main index.
        let mut sources = Vec::with_capacity(records_before - tombstones_before);
        for index in 0..records_before {
            if !self.is_deleted(index) {
                sources.push(MergeSource::Main(index));
            }
        }

        self.rewrite_index(&sources, self.metadata.compaction_gen)?;

        let records_after = self.record_count();
        let bytes_reclaimed = bytes_before.saturating_sub(self.data.len());
        Ok(VacuumStats {
            records_before,
            records_after,
            tombstones_removed: records_before.saturating_sub(records_after),
            bytes_reclaimed,
            duration: start.elapsed(),
        })
    }

    /// Append a single vector to the index via the WAL.
    ///
    /// The vector is immediately searchable. It is written to the WAL
    /// sidecar file for crash safety.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::Io` for filesystem failures.
    pub fn append(&mut self, doc_id: &str, vector: &[f32]) -> SearchResult<()> {
        self.append_batch(&[(doc_id.to_owned(), vector.to_vec())])
    }

    /// Append a batch of vectors to the index via the WAL.
    ///
    /// All vectors in the batch are written atomically to a single WAL
    /// batch (one CRC covers the whole batch).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths,
    /// `SearchError::InvalidConfig` for invalid values, and
    /// `SearchError::Io` for filesystem failures.
    pub fn append_batch(&mut self, entries: &[(String, Vec<f32>)]) -> SearchResult<()> {
        self.append_batch_impl(entries)
    }

    /// Bench-only alias retained for the wal_append_dedup_ab harness.
    ///
    /// The "skip redundant dedup" candidate arm no longer exists: since the
    /// log-then-supersede reordering, the resident-WAL dedup is load-bearing
    /// (it feeds the sidecar compaction) and can never be skipped, so both
    /// arms measure the same code path.
    ///
    /// # Errors
    ///
    /// Returns the same validation and I/O errors as [`Self::append_batch`].
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn bench_append_batch_skip_redundant_dedup(
        &mut self,
        entries: &[(String, Vec<f32>)],
    ) -> SearchResult<()> {
        self.append_batch_impl(entries)
    }

    fn append_batch_impl(&mut self, entries: &[(String, Vec<f32>)]) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("append_batch")?;
        if entries.is_empty() {
            return Ok(());
        }

        // Validate all entries before writing anything.
        for (doc_id, vector) in entries {
            if vector.len() != self.dimension() {
                return Err(SearchError::DimensionMismatch {
                    expected: self.dimension(),
                    found: vector.len(),
                });
            }
            if vector.iter().any(|v| !v.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "embedding".to_owned(),
                    value: "<contains non-finite values>".to_owned(),
                    reason: "all embedding values must be finite".to_owned(),
                });
            }
            if !vector_signal_usable(vector) {
                return Err(SearchError::InvalidConfig {
                    field: "embedding".to_owned(),
                    value: "<zero-norm vector>".to_owned(),
                    reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
                });
            }
            let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id".to_owned(),
                value: doc_id.clone(),
                reason: "doc_id byte length must fit in u16".to_owned(),
            })?;
        }

        let mut wal_entries: Vec<wal::WalEntry> = Vec::with_capacity(entries.len());
        let mut seen = std::collections::HashSet::new();
        for (doc_id, embedding) in entries.iter().rev() {
            if seen.insert(doc_id) {
                wal_entries.push(wal::WalEntry {
                    doc_id: doc_id.clone(),
                    doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
                    embedding: embedding.clone(),
                });
            }
        }
        wal_entries.reverse();

        // DURABILITY ORDER IS LOAD-BEARING (fleet-review critical): the
        // replacement entries must be durably logged BEFORE anything
        // destroys the old copies. The previous ordering ran
        // `soft_delete_batch` first, which durably tombstoned the main-index
        // rows and durably rewrote the WAL sidecar without the doc — so an
        // append_wal_batch failure (ENOSPC is the canonical one) or a crash
        // in between destroyed the old vector with the new one never
        // written: an update executed as delete-then-log. Log-then-supersede
        // is safe at every cut point because `open` deduplicates duplicate
        // WAL doc IDs with last-wins semantics.
        let wal_path = wal::wal_path_for(&self.path);
        wal::append_wal_batch(
            &wal_path,
            &wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        )?;

        // Supersede older resident copies in memory, then admit the new
        // entries (immediately searchable). This dedup is load-bearing: it
        // produces the exact resident set the sidecar compaction below
        // persists.
        let resident_before = self.wal_entries.len();
        for new_entry in &wal_entries {
            self.wal_entries
                .retain(|existing| existing.doc_id != new_entry.doc_id);
        }
        let superseded_resident = self.wal_entries.len() < resident_before;
        self.wal_entries.extend(wal_entries.clone());

        // BEST-EFFORT: compact superseded duplicates out of the durable
        // sidecar (keeps repeated updates from growing it — the goal the
        // old delete-first ordering pursued unsafely). The atomic
        // tmp+rename rewrite includes the entries appended above, and a
        // failure or crash here merely leaves old+new pairs on disk for
        // `open`'s last-wins dedup — so it must never fail the append.
        if superseded_resident {
            if let Err(err) = self.rewrite_wal_sidecar() {
                tracing::warn!(
                    target: "frankensearch.index",
                    path = %self.path.display(),
                    error = %err,
                    "post-append WAL sidecar compaction failed; superseded \
                     duplicates remain until the next rewrite"
                );
            }
        }

        // BEST-EFFORT: Tombstone the old main index entries so they don't pollute the top-K heap.
        // If this crashes before completing, it's fine; they will be resolved out later (though they might steal a top-K slot temporarily).
        for entry in &wal_entries {
            let hash = entry.doc_id_hash;
            if let Ok(Some(mut index)) = self.find_first_hash_match(hash) {
                while index > 0 {
                    if let Ok(prev) = self.record_at(index - 1) {
                        if prev.doc_id_hash != hash {
                            break;
                        }
                        index -= 1;
                    } else {
                        break;
                    }
                }
                for candidate in index..self.record_count() {
                    if let Ok(rec) = self.record_at(candidate) {
                        if rec.doc_id_hash != hash {
                            break;
                        }
                        if !is_tombstoned_flags(rec.flags) {
                            if let Ok(candidate_doc_id) = self.doc_id_at(candidate) {
                                if candidate_doc_id == entry.doc_id {
                                    let flags = rec.flags | RECORD_FLAG_TOMBSTONE;
                                    if let Err(err) = self.set_record_flags(candidate, flags) {
                                        tracing::warn!(
                                            target: "frankensearch.index",
                                            path = %self.path.display(),
                                            candidate_index = candidate,
                                            doc_id = %entry.doc_id,
                                            error = %err,
                                            "WAL replay: failed to tombstone superseded record; \
                                             duplicate may persist until next compaction"
                                        );
                                    }
                                    break;
                                }
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            batch_size = entries.len(),
            wal_total = self.wal_entries.len(),
            "appended to WAL"
        );
        Ok(())
    }

    /// Compact the WAL into the main index.
    ///
    /// Rewrites the main index file with all main + WAL records merged,
    /// then removes the WAL sidecar. The index is atomically swapped
    /// (write to tmp, rename over original).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` for filesystem failures and
    /// `SearchError::InvalidConfig` for encoding issues.
    #[allow(clippy::cast_precision_loss)]
    pub fn compact(&mut self) -> SearchResult<CompactionStats> {
        self.ensure_legacy_mutation_format("compact")?;
        let start = Instant::now();
        let main_before = self.record_count();
        let wal_count = self.wal_entries.len();

        if wal_count == 0 {
            return Ok(CompactionStats {
                main_records_before: main_before,
                wal_records: 0,
                total_records_after: main_before,
                elapsed_ms: 0.0,
            });
        }

        let deduped_sources = (|| -> SearchResult<Vec<MergeSource>> {
            #[derive(Clone, Copy)]
            struct SortKey<'a> {
                doc_id_hash: u64,
                doc_id: &'a str,
            }

            #[derive(Clone, Copy)]
            struct KeyedSource<'a> {
                key: SortKey<'a>,
                source: MergeSource,
            }

            // Collect all sources with their sort keys.
            let mut keyed_sources = Vec::with_capacity(main_before + wal_count);
            for i in 0..main_before {
                if !self.is_deleted(i) {
                    let entry = self.record_at(i)?;
                    let doc_id = self.doc_id_at(i)?;
                    keyed_sources.push(KeyedSource {
                        key: SortKey {
                            doc_id_hash: entry.doc_id_hash,
                            doc_id,
                        },
                        source: MergeSource::Main(i),
                    });
                }
            }
            for (idx, entry) in self.wal_entries.iter().enumerate() {
                keyed_sources.push(KeyedSource {
                    key: SortKey {
                        doc_id_hash: entry.doc_id_hash,
                        doc_id: &entry.doc_id,
                    },
                    source: MergeSource::Wal(idx),
                });
            }

            // Sort to ensure binary search property.
            keyed_sources.sort_by(|a, b| {
                a.key
                    .doc_id_hash
                    .cmp(&b.key.doc_id_hash)
                    .then(a.key.doc_id.cmp(b.key.doc_id))
            });

            // Deduplicate sources by doc_id, keeping the latest (WAL over Main).
            // Since `keyed_sources` is sorted, duplicates are adjacent and the stable sort
            // ensures that newer sources (WAL) appear after older sources (Main).
            let mut deduped: Vec<KeyedSource<'_>> = Vec::with_capacity(keyed_sources.len());
            for item in keyed_sources {
                if let Some(last) = deduped.last_mut() {
                    if item.key.doc_id_hash == last.key.doc_id_hash
                        && item.key.doc_id == last.key.doc_id
                    {
                        // Overwrite the older entry with the newer one
                        *last = item;
                        continue;
                    }
                }
                deduped.push(item);
            }

            Ok(deduped
                .into_iter()
                .map(|item| item.source)
                .collect::<Vec<_>>())
        })()?;

        // Perform the rewrite.
        self.rewrite_index(
            &deduped_sources,
            next_generation(self.metadata.compaction_gen),
        )?;

        // After rewrite_index succeeds, clear in-memory WAL state immediately
        // (the data is now in the main index). If remove_wal fails, the stale
        // WAL file on disk will be detected and discarded on next open() via
        // the generation counter.
        self.wal_entries.clear();

        // Then try to remove the WAL file (best-effort).
        let wal_path = wal::wal_path_for(&self.path);
        if let Err(e) = wal::remove_wal(&wal_path) {
            tracing::warn!("failed to remove WAL file after compaction: {e}");
        }

        let elapsed = start.elapsed();
        let stats = CompactionStats {
            main_records_before: main_before,
            wal_records: wal_count,
            total_records_after: self.record_count(),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        };

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            main_before,
            wal_count,
            total_after = stats.total_records_after,
            elapsed_ms = format_args!("{:.1}", stats.elapsed_ms),
            "compaction complete"
        );
        Ok(stats)
    }

    fn resolve_sort_key<'a>(&'a self, source: &MergeSource) -> SearchResult<(u64, &'a str)> {
        match source {
            MergeSource::Main(idx) => {
                let entry = self.record_at(*idx)?;
                let id = self.doc_id_at(*idx)?;
                Ok((entry.doc_id_hash, id))
            }
            MergeSource::Wal(idx) => {
                let entry = &self.wal_entries[*idx];
                Ok((entry.doc_id_hash, &entry.doc_id))
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn rewrite_index(&mut self, sources: &[MergeSource], new_gen: u8) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("rewrite_index")?;
        let record_count = sources.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 =
            u64::try_from(records_bytes).map_err(|_| SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size does not fit in u64".to_owned(),
            })?;

        // Pass 1: Build Record Table and calculate layout.
        // We buffer the Record Table in memory (16 bytes * N).
        // 10M records = 160MB, which is acceptable.
        let mut record_table = Vec::with_capacity(records_bytes);
        let mut current_string_offset = 0u32;
        let mut string_table_len = 0u64;

        for source in sources {
            let (doc_id_hash, doc_id) = self.resolve_sort_key(source)?;
            let doc_id_len = doc_id.len();

            // Validation
            let len_u16 = u16::try_from(doc_id_len).map_err(|_| SearchError::InvalidConfig {
                field: "doc_id_len".to_owned(),
                value: doc_id_len.to_string(),
                reason: "doc_id length exceeds u16".to_owned(),
            })?;
            let len_u32 = u32::from(len_u16);
            let len_u64 = u64::from(len_u16);
            if current_string_offset.checked_add(len_u32).is_none() {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: "overflow".to_owned(),
                    reason: "string table offset exceeds u32".to_owned(),
                });
            }

            // Append to record table
            record_table.extend_from_slice(&doc_id_hash.to_le_bytes());
            record_table.extend_from_slice(&current_string_offset.to_le_bytes());
            record_table.extend_from_slice(&len_u16.to_le_bytes());
            record_table.extend_from_slice(&0u16.to_le_bytes()); // Flags cleared (tombstones gone)

            current_string_offset += len_u32;
            string_table_len += len_u64;
        }

        // Calculate layout
        let provisional_header = build_header_prefix(
            &self.metadata.embedder_id,
            &self.metadata.embedder_revision,
            self.dimension(),
            self.quantization(),
            new_gen,
            record_count,
            0,
        )?;
        let header_len = provisional_header.len() + 4; // + CRC
        let header_len_u64 = u64::try_from(header_len).map_err(|_| SearchError::InvalidConfig {
            field: "header".to_owned(),
            value: header_len.to_string(),
            reason: "header length does not fit in u64".to_owned(),
        })?;

        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|v| v.checked_add(string_table_len))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: "overflow".to_owned(),
                reason: "layout offset overflow".to_owned(),
            })?;

        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len = usize::try_from(vectors_offset - pre_vector).map_err(|_| {
            SearchError::InvalidConfig {
                field: "padding_len".to_owned(),
                value: (vectors_offset - pre_vector).to_string(),
                reason: "padding length exceeds usize".to_owned(),
            }
        })?;

        // Open temp file
        let tmp_path = temporary_output_path(&self.path);

        // Helper: perform all I/O into tmp_path, rename atomically, and reload.
        // If anything fails after the temp file is created, we clean it up.
        let result = (|| -> SearchResult<()> {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)?;
            {
                let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

                // Pass 2: Write Header and Record Table
                let mut header_prefix = build_header_prefix(
                    &self.metadata.embedder_id,
                    &self.metadata.embedder_revision,
                    self.dimension(),
                    self.quantization(),
                    new_gen,
                    record_count,
                    vectors_offset,
                )?;
                let header_crc = crc32(&header_prefix);
                header_prefix.extend_from_slice(&header_crc.to_le_bytes());

                writer.write_all(&header_prefix)?;
                writer.write_all(&record_table)?;

                // Pass 3: Write String Table
                for source in sources {
                    let (_, doc_id) = self.resolve_sort_key(source)?;
                    writer.write_all(doc_id.as_bytes())?;
                }

                // Padding
                if padding_len > 0 {
                    writer.write_all(&vec![0u8; padding_len])?;
                }

                // Pass 4: Write Vectors
                match self.quantization() {
                    Quantization::F16 => {
                        for source in sources {
                            match source {
                                MergeSource::Main(idx) => {
                                    // Fast path: copy raw bytes
                                    let start = self.vector_start(*idx)?;
                                    let len = self.dimension() * 2;
                                    let bytes = &self.data[start..start + len];
                                    writer.write_all(bytes)?;
                                }
                                MergeSource::Wal(idx) => {
                                    // Slow path: encode
                                    let entry = &self.wal_entries[*idx];
                                    for &val in &entry.embedding {
                                        writer.write_all(&f16::from_f32(val).to_le_bytes())?;
                                    }
                                }
                            }
                        }
                    }
                    Quantization::F32 => {
                        for source in sources {
                            match source {
                                MergeSource::Main(idx) => {
                                    // Fast path: copy raw bytes
                                    let start = self.vector_start(*idx)?;
                                    let len = self.dimension() * 4;
                                    let bytes = &self.data[start..start + len];
                                    writer.write_all(bytes)?;
                                }
                                MergeSource::Wal(idx) => {
                                    // Slow path: encode
                                    let entry = &self.wal_entries[*idx];
                                    for &val in &entry.embedding {
                                        writer.write_all(&val.to_le_bytes())?;
                                    }
                                }
                            }
                        }
                    }
                }
                writer.flush()?;
            }

            file.sync_all()?;
            fs::rename(&tmp_path, &self.path)?;
            sync_parent_directory(&self.path)?;
            Ok(())
        })();

        if result.is_err() {
            // Clean up the temp file on error (best-effort).
            if tmp_path.exists() {
                if let Err(cleanup_err) = fs::remove_file(&tmp_path) {
                    tracing::warn!(
                        "failed to clean up temp file {} after rewrite error: {cleanup_err}",
                        tmp_path.display()
                    );
                }
            }
        }
        result?;

        // Reload
        let config = self.wal_config.clone();
        let reloaded = Self::open(&self.path)?;
        self.data = reloaded.data;
        self.metadata = reloaded.metadata;
        self.records_offset = reloaded.records_offset;
        self.strings_offset = reloaded.strings_offset;
        self.vectors_offset = reloaded.vectors_offset;
        // WAL entries are cleared by caller if compacting, or preserved if vacuuming
        // But vacuum preserves WAL on disk, so open() loads them.
        // Vacuum caller ignores the reloaded WAL entries? No, vacuum preserves them.
        // self.vacuum() impl:
        //   writer.finish()
        //   Self::open() -> loads WAL entries
        //   self.wal_entries = reloaded.wal_entries
        // So we need to update self.wal_entries from reloaded.
        self.wal_entries = reloaded.wal_entries;
        self.wal_config = config;

        Ok(())
    }

    /// Resolve the document id at `index`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed record/string tables.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.ensure_index(index)?;
        let entry = self.record_at(index)?;
        let doc_id_offset = usize::try_from(entry.doc_id_offset).map_err(|_| {
            index_corrupted(
                &self.path,
                format!("doc_id_offset overflow for record at index {index}"),
            )
        })?;
        let doc_id_len = usize::from(entry.doc_id_len);
        let start = self
            .strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id start offset overflow"))?;
        let end = start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(&self.path, "doc_id end offset overflow"))?;
        if end > self.vectors_offset {
            return Err(index_corrupted(
                &self.path,
                format!(
                    "doc_id range [{start}, {end}) exceeds string table end {}",
                    self.vectors_offset
                ),
            ));
        }
        std::str::from_utf8(&self.data[start..end]).map_err(|error| {
            index_corrupted(
                &self.path,
                format!("invalid UTF-8 in doc_id at index {index}: {error}"),
            )
        })
    }

    /// Decode a vector as f32 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dim = self.dimension();
        match self.quantization() {
            Quantization::F32 => {
                let byte_len = dim.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
                Ok(out)
            }
            Quantization::F16 => {
                let byte_len = dim.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                // SIMD-widen 8 little-endian f16 per 16-byte block (`widen8_f16_bytes`,
                // the same magic-factor widen the f16 dot kernels use — bit-identical
                // to the scalar `f16::to_f32`), then a scalar tail for the last < 8.
                let (blocks, remainder) = self.data[start..end].as_chunks::<16>();
                for arr in blocks {
                    out.extend_from_slice(&crate::simd::widen8_f16_bytes(arr).to_array());
                }
                for chunk in remainder.as_chunks::<2>().0 {
                    out.push(f16::from_le_bytes(*chunk).to_f32());
                }
                Ok(out)
            }
        }
    }

    /// Raw stored slab bytes of the vector at `index` (`dim * 2` for f16,
    /// `dim * 4` for f32) — the same window [`Self::vector_at_f32`] decodes, but
    /// borrowed without materializing an f32 `Vec`. Feeds the fused-kernel scorer.
    fn vector_bytes(&self, index: usize) -> SearchResult<&[u8]> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let byte_len = self
            .dimension()
            .checked_mul(self.quantization().bytes_per_element())
            .ok_or_else(|| index_corrupted(&self.path, "vector byte length overflow"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| index_corrupted(&self.path, "vector byte end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(&self.path, "vector extends past file end"));
        }
        Ok(&self.data[start..end])
    }

    /// Exact dot of the stored vector at `index` against an f32 `query`, using the
    /// fused byte-based SIMD kernel (`dot_product_f16_bytes_f32` / `_f32_bytes_f32`)
    /// the brute-force scan already uses — no per-call f32 `Vec` and a hardware
    /// (`vcvtph2ps`) f16 decode instead of the scalar [`Self::vector_at_f32`] loop.
    ///
    /// For `dim % 32 == 0` (every standard embedding width: 128/256/384/512/768…)
    /// the fused kernel's 4-accumulator grouping and reduction coincide exactly
    /// with `vector_at_f32` + `dot_product_f32_f32` and there is no scalar tail, so
    /// the score is **bit-identical** to the former decode-then-dot path. Other
    /// dims fall back to that path to preserve exact scores.
    ///
    /// # Errors
    ///
    /// Propagates the same index/corruption errors as [`Self::vector_at_f32`], and
    /// `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn dot_query_at(&self, index: usize, query: &[f32]) -> SearchResult<f32> {
        if self.dimension() % 32 != 0 {
            let vector = self.vector_at_f32(index)?;
            return dot_product_f32_f32(&vector, query);
        }
        let bytes = self.vector_bytes(index)?;
        match self.quantization() {
            Quantization::F16 => dot_product_f16_bytes_f32(bytes, query),
            Quantization::F32 => dot_product_f32_bytes_f32(bytes, query),
        }
    }

    /// Decode a vector as f16 values.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for out-of-range indices and
    /// `SearchError::IndexCorrupted` for malformed vector slab data.
    pub fn vector_at_f16(&self, index: usize) -> SearchResult<Vec<f16>> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let dim = self.dimension();
        match self.quantization() {
            Quantization::F16 => {
                let byte_len = dim.checked_mul(2).ok_or_else(|| {
                    index_corrupted(&self.path, "f16 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f16 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f16 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].as_chunks::<2>().0 {
                    out.push(f16::from_le_bytes(*chunk));
                }
                Ok(out)
            }
            Quantization::F32 => {
                let byte_len = dim.checked_mul(4).ok_or_else(|| {
                    index_corrupted(&self.path, "f32 vector byte length overflow")
                })?;
                let end = start
                    .checked_add(byte_len)
                    .ok_or_else(|| index_corrupted(&self.path, "f32 vector end overflow"))?;
                if end > self.data.len() {
                    return Err(index_corrupted(
                        &self.path,
                        "f32 vector extends past file end",
                    ));
                }
                let mut out = Vec::with_capacity(dim);
                for chunk in self.data[start..end].as_chunks::<4>().0 {
                    out.push(f16::from_f32(f32::from_le_bytes(*chunk)));
                }
                Ok(out)
            }
        }
    }

    /// Binary-search the sorted record table by document hash.
    #[must_use]
    pub fn find_index_by_doc_hash(&self, doc_id_hash: u64) -> Option<usize> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid).ok()?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    let mut first = mid;
                    while first > 0 {
                        let prev = self.record_at(first - 1).ok()?;
                        if prev.doc_id_hash != doc_id_hash {
                            break;
                        }
                        first -= 1;
                    }
                    for index in first..self.record_count() {
                        let entry = self.record_at(index).ok()?;
                        if entry.doc_id_hash != doc_id_hash {
                            break;
                        }
                        if !is_tombstoned_flags(entry.flags) {
                            return Some(index);
                        }
                    }
                    return None;
                }
            }
        }
        None
    }

    /// Fetch embeddings for hashed doc ids (f16 values).
    ///
    /// Missing hashes return `None` entries at the same position.
    #[must_use]
    pub fn get_embeddings(&self, doc_id_hashes: &[u64]) -> Vec<Option<Vec<f16>>> {
        doc_id_hashes
            .iter()
            .map(|&hash| {
                for entry in self.wal_entries.iter().rev() {
                    if entry.doc_id_hash == hash {
                        // WAL embeddings are f32, we need to convert them to f16
                        return Some(
                            entry
                                .embedding
                                .iter()
                                .map(|&v| half::f16::from_f32(v))
                                .collect(),
                        );
                    }
                }
                if let Some(index) = self.find_index_by_doc_hash(hash) {
                    if let Ok(vec) = self.vector_at_f16(index) {
                        return Some(vec);
                    }
                }
                None
            })
            .collect()
    }

    fn ensure_index(&self, index: usize) -> SearchResult<()> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index out of range for record_count={}",
                    self.record_count()
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn find_index_by_doc_id(&self, doc_id: &str) -> SearchResult<Option<usize>> {
        let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
        let Some(mut index) = self.find_first_hash_match(doc_id_hash)? else {
            return Ok(None);
        };
        while index > 0 {
            let prev = self.record_at(index - 1)?;
            if prev.doc_id_hash != doc_id_hash {
                break;
            }
            index -= 1;
        }

        for candidate in index..self.record_count() {
            let entry = self.record_at(candidate)?;
            if entry.doc_id_hash != doc_id_hash {
                break;
            }
            if !is_tombstoned_flags(entry.flags) {
                let candidate_doc_id = self.doc_id_at(candidate)?;
                if candidate_doc_id == doc_id {
                    return Ok(Some(candidate));
                }
            }
        }
        Ok(None)
    }

    fn find_first_hash_match(&self, doc_id_hash: u64) -> SearchResult<Option<usize>> {
        let mut low = 0usize;
        let mut high = self.record_count();
        while low < high {
            let mid = low + (high - low) / 2;
            let entry = self.record_at(mid)?;
            match entry.doc_id_hash.cmp(&doc_id_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    fn record_flags_offset(&self, index: usize) -> SearchResult<usize> {
        self.ensure_index(index)?;
        let record_offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        record_offset
            .checked_add(14)
            .ok_or_else(|| index_corrupted(&self.path, "flags offset overflow"))
    }

    fn set_record_flags(&mut self, index: usize, flags: u16) -> SearchResult<()> {
        let flags_offset = self.record_flags_offset(index)?;
        let end = flags_offset
            .checked_add(2)
            .ok_or_else(|| index_corrupted(&self.path, "flags end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "flags offset points beyond mapped data",
            ));
        }

        self.data
            .write_and_flush(flags_offset, &flags.to_le_bytes())
    }

    fn rewrite_wal_sidecar(&self) -> SearchResult<()> {
        self.ensure_legacy_mutation_format("rewrite_wal_sidecar")?;
        let wal_path = wal::wal_path_for(&self.path);
        if self.wal_entries.is_empty() {
            wal::remove_wal(&wal_path)?;
            // Durable removal needs the dirent update persisted: without the
            // parent sync a crash can resurrect the removed sidecar, whose
            // stale entries would replay over the post-delete state.
            sync_parent_directory(&wal_path)?;
            return Ok(());
        }

        let mut tmp = wal_path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp_path = PathBuf::from(tmp);
        let _ = wal::remove_wal(&tmp_path);

        if let Err(e) = wal::append_wal_batch(
            &tmp_path,
            &self.wal_entries,
            self.dimension(),
            self.quantization(),
            next_generation(self.metadata.compaction_gen),
            self.wal_config.fsync_on_write,
        ) {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        match fs::rename(&tmp_path, &wal_path) {
            Ok(()) => sync_parent_directory(&wal_path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                wal::remove_wal(&wal_path)?;
                fs::rename(&tmp_path, &wal_path)?;
                sync_parent_directory(&wal_path)
            }
            Err(error) => {
                let _ = wal::remove_wal(&tmp_path);
                Err(error.into())
            }
        }
    }

    pub(crate) fn record_at(&self, index: usize) -> SearchResult<RecordEntry> {
        self.ensure_index(index)?;
        let offset = self
            .records_offset
            .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                index_corrupted(&self.path, "record offset multiplication overflow")
            })?)
            .ok_or_else(|| index_corrupted(&self.path, "record offset overflow"))?;
        let end = offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(&self.path, "record end overflow"))?;
        if end > self.data.len() {
            return Err(index_corrupted(
                &self.path,
                "record table extends beyond file size",
            ));
        }
        let chunk = &self.data[offset..end];
        Ok(RecordEntry {
            doc_id_hash: u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            doc_id_offset: u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            doc_id_len: u16::from_le_bytes([chunk[12], chunk[13]]),
            flags: u16::from_le_bytes([chunk[14], chunk[15]]),
        })
    }

    fn vector_start(&self, index: usize) -> SearchResult<usize> {
        let stride = self
            .dimension()
            .checked_mul(self.quantization().bytes_per_element())
            .ok_or_else(|| index_corrupted(&self.path, "vector stride overflow"))?;
        self.vectors_offset
            .checked_add(
                index
                    .checked_mul(stride)
                    .ok_or_else(|| index_corrupted(&self.path, "vector index overflow"))?,
            )
            .ok_or_else(|| index_corrupted(&self.path, "vector offset overflow"))
    }

    fn ensure_legacy_mutation_format(&self, operation: &str) -> SearchResult<()> {
        if self.metadata.fsvi_version != FSVI_V2_VERSION {
            return Ok(());
        }
        Err(SearchError::InvalidConfig {
            field: "fsvi_v2.mutation".to_owned(),
            value: operation.to_owned(),
            reason: "identity-complete FSVI v2 artifacts are immutable in this admission slice; rebuild and publish a separate generation rather than mutating content or attaching a legacy WAL"
                .to_owned(),
        })
    }
}

#[derive(Debug, Clone)]
struct PendingRecord {
    doc_id: String,
    doc_id_hash: u64,
    flags: u16,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
enum MergeSource {
    Main(usize),
    Wal(usize),
}

#[derive(Debug)]
pub struct VectorIndexWriter {
    path: PathBuf,
    embedder_id: String,
    embedder_revision: String,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    records: Vec<PendingRecord>,
    identity_v2: Option<FsviV2IdentityBinding>,
}

impl VectorIndexWriter {
    /// Append a single `(doc_id, embedding)` record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::InvalidConfig` for invalid values.
    pub fn write_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        self.write_record_with_flags(doc_id, embedding, FsviRecordFlags::LIVE)
    }

    /// Append one retained tombstone row to an identity-complete FSVI v2
    /// generation.
    ///
    /// Tombstone rows remain part of the physical vector-content digest but do
    /// not participate in exact search or the ordered live-document-set digest.
    /// Legacy v1 writers reject this operation because their identity contract
    /// cannot attest the resulting membership semantics.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::write_record`] or
    /// [`SearchError::InvalidConfig`] when called on a legacy writer.
    pub fn write_tombstone_record(&mut self, doc_id: &str, embedding: &[f32]) -> SearchResult<()> {
        if self.identity_v2.is_none() {
            return Err(SearchError::InvalidConfig {
                field: "fsvi_v2.record_flags".to_owned(),
                value: "tombstone".to_owned(),
                reason: "retained tombstones require an identity-complete FSVI v2 writer"
                    .to_owned(),
            });
        }
        self.write_record_with_flags(doc_id, embedding, FsviRecordFlags::TOMBSTONE)
    }

    fn write_record_with_flags(
        &mut self,
        doc_id: &str,
        embedding: &[f32],
        flags: FsviRecordFlags,
    ) -> SearchResult<()> {
        if embedding.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: embedding.len(),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "all embedding values must be finite".to_owned(),
            });
        }
        if !vector_signal_usable(embedding) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<zero-norm vector>".to_owned(),
                reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
            });
        }
        let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
            field: "doc_id".to_owned(),
            value: doc_id.to_owned(),
            reason: "doc_id byte length must fit in u16".to_owned(),
        })?;
        self.records.push(PendingRecord {
            doc_id: doc_id.into(),
            doc_id_hash: fnv1a_hash(doc_id.as_bytes()),
            flags: flags.bits(),
            embedding: embedding.to_vec(),
        });
        Ok(())
    }

    /// Bench-only access to the owned record handoff used by `TwoTierIndexBuilder`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` for wrong embedding lengths
    /// and `SearchError::InvalidConfig` for invalid values.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn write_record_owned_for_benchmark(
        &mut self,
        doc_id: String,
        embedding: Vec<f32>,
    ) -> SearchResult<()> {
        if embedding.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: embedding.len(),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "all embedding values must be finite".to_owned(),
            });
        }
        if !vector_signal_usable(&embedding) {
            return Err(SearchError::InvalidConfig {
                field: "embedding".to_owned(),
                value: "<zero-norm vector>".to_owned(),
                reason: "embedding norm must be non-zero and finite; a zero vector can never match any query".to_owned(),
            });
        }
        let _ = u16::try_from(doc_id.len()).map_err(|_| SearchError::InvalidConfig {
            field: "doc_id".to_owned(),
            value: doc_id.clone(),
            reason: "doc_id byte length must fit in u16".to_owned(),
        })?;
        let doc_id_hash = fnv1a_hash(doc_id.as_bytes());
        self.records.push(PendingRecord {
            doc_id,
            doc_id_hash,
            flags: 0,
            embedding,
        });
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) const fn with_generation(mut self, generation: u8) -> Self {
        self.compaction_gen = generation;
        self
    }

    /// Persist the index to disk, including fsync of file and parent directory.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for layout/encoding failures and
    /// `SearchError::Io` for filesystem write/sync failures.
    #[allow(clippy::too_many_lines)]
    pub fn finish(mut self) -> SearchResult<()> {
        // STABLE sort is load-bearing here: `self.records` can contain duplicate
        // `doc_id`s (soft-delete then rewrite), so the comparator is NOT a strict
        // total order, and stable ordering preserves last-write-wins among dupes
        // (verified by `soft_delete_wal_restores_state_on_rewrite_failure`). Do not
        // switch to `sort_unstable_by`.
        self.records.sort_by(|left, right| {
            left.doc_id_hash
                .cmp(&right.doc_id_hash)
                .then(left.doc_id.cmp(&right.doc_id))
        });
        if self.identity_v2.is_some() {
            if self.records.iter().any(|record| record.doc_id.is_empty()) {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id".to_owned(),
                    value: "<empty>".to_owned(),
                    reason: "identity-complete FSVI v2 document ids must not be empty".to_owned(),
                });
            }
            if self
                .records
                .windows(2)
                .any(|pair| pair[0].doc_id == pair[1].doc_id)
            {
                return Err(SearchError::InvalidConfig {
                    field: "doc_id".to_owned(),
                    value: "<duplicate>".to_owned(),
                    reason:
                        "identity-complete FSVI v2 requires one unique physical row per document id"
                            .to_owned(),
                });
            }
        }

        let record_count = self.records.len();
        let records_bytes = record_count.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
            SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size overflow".to_owned(),
            }
        })?;
        let records_bytes_u64 =
            u64::try_from(records_bytes).map_err(|_| SearchError::InvalidConfig {
                field: "record_count".to_owned(),
                value: record_count.to_string(),
                reason: "record table size does not fit in u64".to_owned(),
            })?;

        let mut string_table = Vec::<u8>::new();
        let mut record_entries = Vec::<RecordEntry>::with_capacity(record_count);
        for record in &self.records {
            let offset_u32 =
                u32::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_offset".to_owned(),
                    value: string_table.len().to_string(),
                    reason: "string table offset exceeds u32".to_owned(),
                })?;
            let doc_id_bytes = record.doc_id.as_bytes();
            let len_u16 =
                u16::try_from(doc_id_bytes.len()).map_err(|_| SearchError::InvalidConfig {
                    field: "doc_id_len".to_owned(),
                    value: doc_id_bytes.len().to_string(),
                    reason: "doc_id length exceeds u16".to_owned(),
                })?;
            string_table.extend_from_slice(doc_id_bytes);
            record_entries.push(RecordEntry {
                doc_id_hash: record.doc_id_hash,
                doc_id_offset: offset_u32,
                doc_id_len: len_u16,
                flags: record.flags,
            });
        }

        let string_table_len_u64 =
            u64::try_from(string_table.len()).map_err(|_| SearchError::InvalidConfig {
                field: "string_table".to_owned(),
                value: string_table.len().to_string(),
                reason: "string table length does not fit in u64".to_owned(),
            })?;

        let header_len = if let Some(binding) = &self.identity_v2 {
            fsvi_v2_header_len(binding)?
        } else {
            build_header_prefix(
                &self.embedder_id,
                &self.embedder_revision,
                self.dimension,
                self.quantization,
                self.compaction_gen,
                record_count,
                0,
            )?
            .len()
            .checked_add(4)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "header".to_owned(),
                value: "overflow".to_owned(),
                reason: "header length overflow".to_owned(),
            })?
        };
        let header_len_u64 = u64::try_from(header_len).map_err(|_| SearchError::InvalidConfig {
            field: "header".to_owned(),
            value: header_len.to_string(),
            reason: "header length does not fit in u64".to_owned(),
        })?;
        let pre_vector = header_len_u64
            .checked_add(records_bytes_u64)
            .and_then(|value| value.checked_add(string_table_len_u64))
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "layout".to_owned(),
                value: format!("{header_len_u64}+{records_bytes_u64}+{string_table_len_u64}"),
                reason: "layout offset overflow".to_owned(),
            })?;
        let vectors_offset = align_up(pre_vector, VECTOR_ALIGN_BYTES)?;
        let padding_len_u64 =
            vectors_offset
                .checked_sub(pre_vector)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "layout".to_owned(),
                    value: format!("{vectors_offset}-{pre_vector}"),
                    reason: "negative padding detected".to_owned(),
                })?;
        let padding_len =
            usize::try_from(padding_len_u64).map_err(|_| SearchError::InvalidConfig {
                field: "padding".to_owned(),
                value: padding_len_u64.to_string(),
                reason: "padding length does not fit in usize".to_owned(),
            })?;

        let legacy_header = if self.identity_v2.is_none() {
            let mut header = build_header_prefix(
                &self.embedder_id,
                &self.embedder_revision,
                self.dimension,
                self.quantization,
                self.compaction_gen,
                record_count,
                vectors_offset,
            )?;
            let header_crc = crc32(&header);
            header.extend_from_slice(&header_crc.to_le_bytes());
            Some(header)
        } else {
            None
        };
        let ordered_docset_digest = self
            .identity_v2
            .as_ref()
            .map(|_| ordered_docset_digest(&self.records));

        let tmp_path = temporary_output_path(&self.path);
        let result = (|| -> SearchResult<()> {
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)?;
            let vector_content_digest;
            {
                let mut writer = BufWriter::with_capacity(256 * 1024, &mut file);

                if let Some(header) = &legacy_header {
                    writer.write_all(header)?;
                } else {
                    writer.write_all(&vec![0_u8; header_len])?;
                }
                for entry in &record_entries {
                    writer.write_all(&entry.doc_id_hash.to_le_bytes())?;
                    writer.write_all(&entry.doc_id_offset.to_le_bytes())?;
                    writer.write_all(&entry.doc_id_len.to_le_bytes())?;
                    writer.write_all(&entry.flags.to_le_bytes())?;
                }
                writer.write_all(&string_table)?;
                if padding_len > 0 {
                    writer.write_all(&vec![0_u8; padding_len])?;
                }
                vector_content_digest = if self.identity_v2.is_some() {
                    Some(write_vector_slab_v2(
                        &mut writer,
                        &self.records,
                        self.dimension,
                        self.quantization,
                    )?)
                } else {
                    write_vector_slab(&mut writer, &self.records, self.quantization)?;
                    None
                };
                writer.flush()?;
            }

            if let Some(binding) = &self.identity_v2 {
                let docset_digest = ordered_docset_digest.ok_or_else(|| {
                    fsvi_v2_config_error(
                        "ordered_live_docset_digest",
                        "v2 writer did not compute its docset digest",
                    )
                })?;
                let vector_digest = vector_content_digest.ok_or_else(|| {
                    fsvi_v2_config_error(
                        "vector_content_digest",
                        "v2 writer did not compute its vector digest",
                    )
                })?;
                let header = build_v2_header(
                    binding,
                    record_count,
                    vectors_offset,
                    docset_digest,
                    vector_digest,
                )?;
                if header.len() != header_len {
                    return Err(SearchError::InvalidConfig {
                        field: "fsvi_v2.header_size".to_owned(),
                        value: header.len().to_string(),
                        reason: format!("expected precomputed header size {header_len}"),
                    });
                }
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&header)?;
            }
            file.sync_all()?;
            if self.identity_v2.is_some() {
                let wal_path = wal::wal_path_for(&self.path);
                match fs::symlink_metadata(&wal_path) {
                    Ok(_) => {
                        return Err(SearchError::InvalidConfig {
                            field: "fsvi_v2.wal_sidecar".to_owned(),
                            value: wal_path.display().to_string(),
                            reason: "identity-complete v2 publication refuses a target with any existing WAL sidecar"
                                .to_owned(),
                        });
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(SearchError::Io(error)),
                }
            }
            fs::rename(&tmp_path, &self.path)?;
            sync_parent_directory(&self.path)?;
            Ok(())
        })();

        if result.is_err() {
            if tmp_path.exists() {
                if let Err(cleanup_err) = fs::remove_file(&tmp_path) {
                    tracing::warn!(
                        "failed to clean up temp file {} after write error: {cleanup_err}",
                        tmp_path.display()
                    );
                }
            }
        }
        result?;

        debug!(
            target: "frankensearch.index",
            path = %self.path.display(),
            record_count,
            dimension = self.dimension,
            quantization = self.quantization as u8,
            vectors_offset,
            "wrote fsvi index"
        );
        Ok(())
    }
}

fn parse_header(path: &Path, data: &[u8]) -> SearchResult<(VectorMetadata, usize)> {
    let mut cursor = 0usize;
    let magic = read_array::<4>(path, data, &mut cursor, "magic")?;
    if magic != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!("bad magic bytes: expected {FSVI_MAGIC:?}, found {magic:?}"),
        ));
    }

    let version = u16::from_le_bytes(read_array::<2>(path, data, &mut cursor, "version")?);
    if version != FSVI_VERSION {
        return Err(SearchError::IndexVersionMismatch {
            expected: FSVI_VERSION,
            found: version,
        });
    }

    let embedder_id_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_id_len",
    )?));
    let embedder_id_bytes = read_slice(path, data, &mut cursor, embedder_id_len, "embedder_id")?;
    let embedder_id = std::str::from_utf8(embedder_id_bytes)
        .map_err(|error| index_corrupted(path, format!("invalid UTF-8 in embedder_id: {error}")))?
        .to_owned();

    let embedder_revision_len = usize::from(u16::from_le_bytes(read_array::<2>(
        path,
        data,
        &mut cursor,
        "embedder_revision_len",
    )?));
    let embedder_revision_bytes = read_slice(
        path,
        data,
        &mut cursor,
        embedder_revision_len,
        "embedder_revision",
    )?;
    let embedder_revision = std::str::from_utf8(embedder_revision_bytes)
        .map_err(|error| {
            index_corrupted(path, format!("invalid UTF-8 in embedder_revision: {error}"))
        })?
        .to_owned();

    let dimension_u32 = u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "dimension")?);
    let dimension = usize::try_from(dimension_u32)
        .map_err(|_| index_corrupted(path, "dimension does not fit in usize"))?;
    if dimension == 0 {
        return Err(index_corrupted(path, "dimension must be greater than zero"));
    }

    let quantization_byte = read_array::<1>(path, data, &mut cursor, "quantization")?[0];
    let quantization = Quantization::from_wire(quantization_byte, path)?;

    // Use first reserved byte for compaction generation
    let reserved = read_array::<3>(path, data, &mut cursor, "reserved")?;
    let compaction_gen = reserved[0];
    // reserved[1..2] remain unused

    let record_count_u64 =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "record_count")?);
    let record_count = usize::try_from(record_count_u64)
        .map_err(|_| index_corrupted(path, "record_count does not fit in usize"))?;
    let vectors_offset =
        u64::from_le_bytes(read_array::<8>(path, data, &mut cursor, "vectors_offset")?);
    let expected_crc =
        u32::from_le_bytes(read_array::<4>(path, data, &mut cursor, "header_crc32")?);
    let actual_crc = crc32(&data[..cursor - 4]);
    if actual_crc != expected_crc {
        return Err(index_corrupted(
            path,
            format!("header CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"),
        ));
    }

    Ok((
        VectorMetadata {
            fsvi_version: FSVI_VERSION,
            embedder_id,
            embedder_revision,
            dimension,
            quantization,
            compaction_gen,
            record_count,
            vectors_offset,
            identity_v2: None,
        },
        cursor,
    ))
}

fn read_header_for_inspection(path: &Path) -> SearchResult<(u16, Vec<u8>)> {
    if !path.exists() {
        return Err(SearchError::IndexNotFound {
            path: path.to_path_buf(),
        });
    }
    let mut file = File::open(path).map_err(SearchError::Io)?;
    let file_len = file.metadata().map_err(SearchError::Io)?.len();
    let mut prefix = [0_u8; 6];
    read_exact_index_bytes(path, &mut file, &mut prefix, "magic and version")?;
    if prefix[..4] != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!(
                "bad magic bytes: expected {FSVI_MAGIC:?}, found {:?}",
                &prefix[..4]
            ),
        ));
    }
    let version = u16::from_le_bytes([prefix[4], prefix[5]]);
    match version {
        FSVI_VERSION => {
            let bounded_len = usize::try_from(file_len)
                .unwrap_or(usize::MAX)
                .min(FSVI_V1_MAX_HEADER_BYTES);
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = Vec::with_capacity(bounded_len);
            file.take(u64::try_from(bounded_len).unwrap_or(u64::MAX))
                .read_to_end(&mut header)
                .map_err(SearchError::Io)?;
            Ok((version, header))
        }
        FSVI_V2_VERSION => {
            let mut encoded_size = [0_u8; 4];
            read_exact_index_bytes(path, &mut file, &mut encoded_size, "v2 header_size")?;
            let header_size = usize::try_from(u32::from_le_bytes(encoded_size))
                .map_err(|_| index_corrupted(path, "v2 header_size does not fit in usize"))?;
            validate_v2_header_size(path, header_size)?;
            if u64::try_from(header_size).is_ok_and(|size| size > file_len) {
                return Err(index_corrupted(
                    path,
                    format!(
                        "v2 header is truncated: declared {header_size} bytes, file has {file_len}"
                    ),
                ));
            }
            file.seek(SeekFrom::Start(0)).map_err(SearchError::Io)?;
            let mut header = vec![0_u8; header_size];
            read_exact_index_bytes(path, &mut file, &mut header, "v2 header")?;
            Ok((version, header))
        }
        _ => Ok((version, prefix.to_vec())),
    }
}

fn read_exact_index_bytes(
    path: &Path,
    reader: &mut impl Read,
    bytes: &mut [u8],
    field: &str,
) -> SearchResult<()> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            index_corrupted(path, format!("{field} is truncated"))
        } else {
            SearchError::Io(error)
        }
    })
}

fn validate_v2_header_size(path: &Path, header_size: usize) -> SearchResult<()> {
    if !(FSVI_V2_MIN_HEADER_BYTES..=FSVI_V2_MAX_HEADER_BYTES).contains(&header_size) {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header_size {header_size} is outside [{FSVI_V2_MIN_HEADER_BYTES}, {FSVI_V2_MAX_HEADER_BYTES}]"
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn parse_v2_header(path: &Path, data: &[u8]) -> SearchResult<(VectorMetadata, usize)> {
    let mut cursor = 0usize;
    let magic = read_array::<4>(path, data, &mut cursor, "magic")?;
    if magic != FSVI_MAGIC {
        return Err(index_corrupted(
            path,
            format!("bad magic bytes: expected {FSVI_MAGIC:?}, found {magic:?}"),
        ));
    }
    let version = u16::from_le_bytes(read_array::<2>(path, data, &mut cursor, "version")?);
    if version != FSVI_V2_VERSION {
        return Err(SearchError::IndexVersionMismatch {
            expected: FSVI_V2_VERSION,
            found: version,
        });
    }
    let header_size = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        data,
        &mut cursor,
        "header_size",
    )?))
    .map_err(|_| index_corrupted(path, "v2 header_size does not fit in usize"))?;
    validate_v2_header_size(path, header_size)?;
    if header_size > data.len() {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header is truncated: declared {header_size} bytes, have {}",
                data.len()
            ),
        ));
    }
    let header = &data[..header_size];

    let binding_schema = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "identity_binding_schema",
    )?);
    if binding_schema != FSVI_V2_IDENTITY_BINDING_SCHEMA {
        return Err(index_corrupted(
            path,
            format!(
                "unsupported v2 identity binding schema {binding_schema}; expected {FSVI_V2_IDENTITY_BINDING_SCHEMA}"
            ),
        ));
    }
    let quantization = Quantization::from_wire(
        read_array::<1>(path, header, &mut cursor, "quantization")?[0],
        path,
    )?;
    let flags = read_array::<1>(path, header, &mut cursor, "header_flags")?[0];
    let reserved = u16::from_le_bytes(read_array::<2>(path, header, &mut cursor, "reserved")?);
    if flags != 0 || reserved != 0 {
        return Err(index_corrupted(
            path,
            "v2 header flags and reserved fields must be zero",
        ));
    }
    let dimension_u32 =
        u32::from_le_bytes(read_array::<4>(path, header, &mut cursor, "dimension")?);
    let dimension = usize::try_from(dimension_u32)
        .map_err(|_| index_corrupted(path, "dimension does not fit in usize"))?;
    if dimension == 0 {
        return Err(index_corrupted(path, "dimension must be greater than zero"));
    }
    let record_count_u64 =
        u64::from_le_bytes(read_array::<8>(path, header, &mut cursor, "record_count")?);
    let record_count = usize::try_from(record_count_u64)
        .map_err(|_| index_corrupted(path, "record_count does not fit in usize"))?;
    let vectors_offset = u64::from_le_bytes(read_array::<8>(
        path,
        header,
        &mut cursor,
        "vectors_offset",
    )?);

    let generation_schema = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "generation_schema",
    )?);
    let generation_reserved = u16::from_le_bytes(read_array::<2>(
        path,
        header,
        &mut cursor,
        "generation_reserved",
    )?);
    if generation_reserved != 0 {
        return Err(index_corrupted(
            path,
            "v2 generation reserved field must be zero",
        ));
    }
    let generation_sequence = u64::from_le_bytes(read_array::<8>(
        path,
        header,
        &mut cursor,
        "generation_sequence",
    )?);
    let generation_nonce = read_array::<16>(path, header, &mut cursor, "generation_nonce")?;
    let generation = ArtifactGenerationIdentityV1 {
        schema_version: generation_schema,
        sequence: generation_sequence,
        nonce: generation_nonce,
    };
    generation.validate().map_err(|error| {
        index_corrupted(path, format!("v2 artifact generation is invalid: {error}"))
    })?;

    let bundle_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "identity_bundle_len",
    )?))
    .map_err(|_| index_corrupted(path, "identity_bundle_len does not fit in usize"))?;
    let space_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "space_identity_len",
    )?))
    .map_err(|_| index_corrupted(path, "space_identity_len does not fit in usize"))?;
    let storage_len = usize::try_from(u32::from_le_bytes(read_array::<4>(
        path,
        header,
        &mut cursor,
        "storage_identity_len",
    )?))
    .map_err(|_| index_corrupted(path, "storage_identity_len does not fit in usize"))?;
    for (field, len) in [
        ("identity_bundle_len", bundle_len),
        ("space_identity_len", space_len),
        ("storage_identity_len", storage_len),
    ] {
        if len == 0 || len > FSVI_V2_MAX_CANONICAL_IDENTITY_BYTES {
            return Err(index_corrupted(
                path,
                format!("{field} must be non-zero and no larger than 1 MiB"),
            ));
        }
    }

    let identity_bundle_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "identity_bundle_fingerprint")?;
    let space_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "space_fingerprint")?;
    let producer_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "producer_fingerprint")?;
    let input_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "input_fingerprint")?;
    let storage_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "storage_fingerprint")?;
    let generation_fingerprint =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "generation_fingerprint")?;
    let ordered_live_docset_digest =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "ordered_live_docset_digest")?;
    let vector_content_digest =
        read_array::<SHA256_BYTES>(path, header, &mut cursor, "vector_content_digest")?;
    for (field, fingerprint) in [
        ("identity_bundle_fingerprint", identity_bundle_fingerprint),
        ("space_fingerprint", space_fingerprint),
        ("producer_fingerprint", producer_fingerprint),
        ("input_fingerprint", input_fingerprint),
        ("storage_fingerprint", storage_fingerprint),
        ("generation_fingerprint", generation_fingerprint),
        ("ordered_live_docset_digest", ordered_live_docset_digest),
        ("vector_content_digest", vector_content_digest),
    ] {
        if fingerprint == [0; SHA256_BYTES] {
            return Err(index_corrupted(
                path,
                format!("v2 {field} must not be all zero"),
            ));
        }
    }
    if cursor != FSVI_V2_FIXED_PREFIX_BYTES {
        return Err(index_corrupted(
            path,
            "internal v2 fixed header layout disagreement",
        ));
    }

    let identity_bundle_canonical_bytes =
        read_slice(path, header, &mut cursor, bundle_len, "identity_bundle")?.to_vec();
    let space_identity_canonical_bytes =
        read_slice(path, header, &mut cursor, space_len, "space_identity")?.to_vec();
    let storage_identity_canonical_bytes =
        read_slice(path, header, &mut cursor, storage_len, "storage_identity")?.to_vec();
    let crc_offset = header_size
        .checked_sub(4)
        .ok_or_else(|| index_corrupted(path, "v2 CRC offset underflow"))?;
    if cursor != crc_offset {
        return Err(index_corrupted(
            path,
            format!(
                "v2 canonical identity lengths end at byte {cursor}, expected CRC at {crc_offset}"
            ),
        ));
    }
    let expected_crc =
        u32::from_le_bytes(read_array::<4>(path, header, &mut cursor, "header_crc32")?);
    let actual_crc = crc32(&header[..crc_offset]);
    if actual_crc != expected_crc {
        return Err(index_corrupted(
            path,
            format!(
                "v2 header CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
            ),
        ));
    }

    validate_canonical_fingerprint(
        path,
        "identity bundle",
        &identity_bundle_canonical_bytes,
        identity_bundle_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "embedding space",
        &space_identity_canonical_bytes,
        space_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "vector storage",
        &storage_identity_canonical_bytes,
        storage_fingerprint,
    )?;
    validate_canonical_fingerprint(
        path,
        "artifact generation",
        &generation.canonical_bytes(),
        generation_fingerprint,
    )?;
    let bundle_components =
        parse_bundle_component_fingerprints(path, &identity_bundle_canonical_bytes)?;
    if bundle_components
        != [
            space_fingerprint,
            producer_fingerprint,
            input_fingerprint,
            storage_fingerprint,
        ]
    {
        return Err(index_corrupted(
            path,
            "v2 complete identity bundle does not bind the stored component fingerprints",
        ));
    }
    let (embedder_id, embedder_revision, space_normalization) = parse_complete_space_identity(
        path,
        &space_identity_canonical_bytes,
        dimension_u32,
        input_fingerprint,
    )?;
    let storage_normalization = validate_storage_identity_bytes(
        path,
        &storage_identity_canonical_bytes,
        dimension_u32,
        quantization,
    )?;
    if storage_normalization != space_normalization {
        return Err(index_corrupted(
            path,
            "v2 storage normalization disagrees with the embedding-space output normalization",
        ));
    }

    Ok((
        VectorMetadata {
            fsvi_version: FSVI_V2_VERSION,
            embedder_id,
            embedder_revision,
            dimension,
            quantization,
            compaction_gen: 0,
            record_count,
            vectors_offset,
            identity_v2: Some(FsviV2IdentityMetadata {
                generation,
                identity_bundle_canonical_bytes,
                space_identity_canonical_bytes,
                storage_identity_canonical_bytes,
                identity_bundle_fingerprint,
                space_fingerprint,
                producer_fingerprint,
                input_fingerprint,
                storage_fingerprint,
                generation_fingerprint,
                ordered_live_docset_digest,
                vector_content_digest,
                header_size,
            }),
        },
        header_size,
    ))
}

fn validate_canonical_fingerprint(
    path: &Path,
    field: &str,
    canonical_bytes: &[u8],
    expected: [u8; SHA256_BYTES],
) -> SearchResult<()> {
    let actual = sha256_bytes(canonical_bytes);
    if actual != expected {
        return Err(index_corrupted(
            path,
            format!("v2 {field} canonical bytes disagree with their SHA-256 fingerprint"),
        ));
    }
    Ok(())
}

fn parse_bundle_component_fingerprints(
    path: &Path,
    bytes: &[u8],
) -> SearchResult<[[u8; SHA256_BYTES]; 4]> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "bundle.domain")?;
    if domain != EMBEDDING_BUNDLE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 identity bundle canonical domain is invalid",
        ));
    }
    let mut fingerprints = [[0_u8; SHA256_BYTES]; 4];
    for (index, field) in ["space", "producer", "input", "storage"]
        .into_iter()
        .enumerate()
    {
        let value = read_canonical_text(
            path,
            bytes,
            &mut cursor,
            &format!("bundle.{field}_fingerprint"),
        )?;
        fingerprints[index] = decode_sha256_fingerprint_from_index(path, field, value)?;
    }
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 identity bundle canonical bytes contain a trailing suffix",
        ));
    }
    Ok(fingerprints)
}

#[allow(clippy::too_many_lines)]
fn parse_complete_space_identity(
    path: &Path,
    bytes: &[u8],
    expected_dimension: u32,
    expected_input_fingerprint: [u8; SHA256_BYTES],
) -> SearchResult<(String, String, String)> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "space.domain")?;
    if domain != EMBEDDING_SPACE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 embedding-space canonical domain is invalid",
        ));
    }
    let schema = read_canonical_u16(path, bytes, &mut cursor, "space.schema")?;
    if schema != EMBEDDING_SPACE_IDENTITY_SCHEMA_V1 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 embedding-space schema {schema} is unsupported; expected {EMBEDDING_SPACE_IDENTITY_SCHEMA_V1}"
            ),
        ));
    }
    let model_id =
        read_canonical_text(path, bytes, &mut cursor, "space.logical_model_id")?.to_owned();
    let revision =
        read_canonical_text(path, bytes, &mut cursor, "space.immutable_revision")?.to_owned();
    let kind = match read_canonical_u8(path, bytes, &mut cursor, "space.kind")? {
        1 => EmbeddingSpaceKindV1::Semantic,
        2 => EmbeddingSpaceKindV1::HashControl,
        found => {
            return Err(index_corrupted(
                path,
                format!("v2 embedding-space kind tag {found} is invalid"),
            ));
        }
    };
    let artifact_manifest_fingerprint = read_canonical_text(
        path,
        bytes,
        &mut cursor,
        "space.artifact_manifest_fingerprint",
    )?
    .to_owned();
    let artifact_count_u64 = read_canonical_u64(path, bytes, &mut cursor, "space.artifact_count")?;
    let artifact_count = usize::try_from(artifact_count_u64)
        .map_err(|_| index_corrupted(path, "v2 artifact count does not fit in usize"))?;
    let minimum_artifact_bytes = 24usize;
    let maximum_artifact_count = bytes
        .len()
        .saturating_sub(cursor)
        .checked_div(minimum_artifact_bytes)
        .unwrap_or(0);
    if artifact_count > maximum_artifact_count {
        return Err(index_corrupted(
            path,
            format!(
                "v2 artifact count {artifact_count} cannot fit in the remaining canonical bytes"
            ),
        ));
    }
    let mut artifacts = Vec::new();
    for index in 0..artifact_count {
        artifacts.push(EmbeddingArtifactIdentityV1 {
            role: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].role"),
            )?
            .to_owned(),
            sha256: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].sha256"),
            )?
            .to_owned(),
            size: read_canonical_u64(
                path,
                bytes,
                &mut cursor,
                &format!("space.artifacts[{index}].size"),
            )?,
        });
    }

    let tokenizer_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.tokenizer_fingerprint")?.to_owned();
    let vocabulary_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.vocabulary_fingerprint")?.to_owned();
    let model_config_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.model_config_fingerprint")?.to_owned();
    let model_preprocessing =
        read_canonical_text(path, bytes, &mut cursor, "space.model_preprocessing")?.to_owned();
    let sequence_policy =
        read_canonical_text(path, bytes, &mut cursor, "space.sequence_policy")?.to_owned();
    let query_instruction =
        read_canonical_text(path, bytes, &mut cursor, "space.query_instruction")?.to_owned();
    let document_instruction =
        read_canonical_text(path, bytes, &mut cursor, "space.document_instruction")?.to_owned();
    let pooling = read_canonical_text(path, bytes, &mut cursor, "space.pooling")?.to_owned();
    let output_normalization =
        read_canonical_text(path, bytes, &mut cursor, "space.output_normalization")?.to_owned();
    let dimension = read_canonical_u32(path, bytes, &mut cursor, "space.dimension")?;
    let input_contract_fingerprint =
        read_canonical_text(path, bytes, &mut cursor, "space.input_contract_fingerprint")?
            .to_owned();

    let hash_control =
        match read_canonical_u8(path, bytes, &mut cursor, "space.hash_control.option")? {
            0 => None,
            1 => Some(HashControlProfileV1 {
                algorithm: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.algorithm",
                )?
                .to_owned(),
                algorithm_revision: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.algorithm_revision",
                )?
                .to_owned(),
                seed: read_canonical_u64(path, bytes, &mut cursor, "space.hash_control.seed")?,
                feature_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.feature_rules",
                )?
                .to_owned(),
                tokenization_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.tokenization_rules",
                )?
                .to_owned(),
                signing_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.signing_rules",
                )?
                .to_owned(),
                normalization_rules: read_canonical_text(
                    path,
                    bytes,
                    &mut cursor,
                    "space.hash_control.normalization_rules",
                )?
                .to_owned(),
            }),
            found => {
                return Err(index_corrupted(
                    path,
                    format!("v2 hash-control option tag {found} is invalid"),
                ));
            }
        };
    let projection = match read_canonical_u8(path, bytes, &mut cursor, "space.projection.option")? {
        0 => None,
        1 => Some(EmbeddingProjectionV1 {
            parent_space_fingerprint: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.parent_space_fingerprint",
            )?
            .to_owned(),
            source_dimension: read_canonical_u32(
                path,
                bytes,
                &mut cursor,
                "space.projection.source_dimension",
            )?,
            output_dimension: read_canonical_u32(
                path,
                bytes,
                &mut cursor,
                "space.projection.output_dimension",
            )?,
            projection_rule: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.projection_rule",
            )?
            .to_owned(),
            renormalization_rule: read_canonical_text(
                path,
                bytes,
                &mut cursor,
                "space.projection.renormalization_rule",
            )?
            .to_owned(),
        }),
        found => {
            return Err(index_corrupted(
                path,
                format!("v2 projection option tag {found} is invalid"),
            ));
        }
    };
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 embedding-space canonical bytes contain a trailing suffix",
        ));
    }

    let identity = EmbeddingSpaceIdentityV1 {
        schema_version: schema,
        logical_model_id: model_id,
        immutable_revision: revision,
        kind,
        artifact_manifest_fingerprint,
        artifacts,
        tokenizer_fingerprint,
        vocabulary_fingerprint,
        model_config_fingerprint,
        model_preprocessing,
        sequence_policy,
        query_instruction,
        document_instruction,
        pooling,
        output_normalization,
        dimension,
        input_contract_fingerprint,
        hash_control,
        projection,
    };
    identity.validate().map_err(|error| {
        index_corrupted(
            path,
            format!("v2 embedding-space identity is incomplete or invalid: {error}"),
        )
    })?;
    if identity.dimension != expected_dimension {
        return Err(index_corrupted(
            path,
            "v2 embedding-space dimension disagrees with the fixed header",
        ));
    }
    let persisted_input_fingerprint = decode_sha256_fingerprint_from_index(
        path,
        "space.input_contract",
        &identity.input_contract_fingerprint,
    )?;
    if persisted_input_fingerprint != expected_input_fingerprint {
        return Err(index_corrupted(
            path,
            "v2 embedding-space input contract disagrees with the complete identity bundle",
        ));
    }
    if identity.canonical_bytes() != bytes {
        return Err(index_corrupted(
            path,
            "v2 embedding-space bytes are not the exact canonical encoding",
        ));
    }
    let model_id = identity.logical_model_id;
    let revision = identity.immutable_revision;
    let output_normalization = identity.output_normalization;
    Ok((model_id, revision, output_normalization))
}

fn validate_storage_identity_bytes(
    path: &Path,
    bytes: &[u8],
    expected_dimension: u32,
    expected_quantization: Quantization,
) -> SearchResult<String> {
    let mut cursor = 0usize;
    let domain = read_canonical_bytes(path, bytes, &mut cursor, "storage.domain")?;
    if domain != VECTOR_STORAGE_CANONICAL_DOMAIN {
        return Err(index_corrupted(
            path,
            "v2 vector-storage canonical domain is invalid",
        ));
    }
    let schema = read_canonical_u16(path, bytes, &mut cursor, "storage.schema")?;
    if schema != VECTOR_STORAGE_IDENTITY_SCHEMA_V1 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 vector-storage schema {schema} is unsupported; expected {VECTOR_STORAGE_IDENTITY_SCHEMA_V1}"
            ),
        ));
    }
    let format = read_canonical_text(path, bytes, &mut cursor, "storage.format")?;
    if format != "fsvi-v2" {
        return Err(index_corrupted(
            path,
            "v2 storage identity must name exactly fsvi-v2",
        ));
    }
    let quantization = read_canonical_u8(path, bytes, &mut cursor, "storage.quantization")?;
    let (expected_quantization_wire, quantization_format) = match expected_quantization {
        Quantization::F32 => (1, QuantizationFormat::F32),
        Quantization::F16 => (2, QuantizationFormat::F16),
    };
    if quantization != expected_quantization_wire {
        return Err(index_corrupted(
            path,
            "v2 storage identity quantization disagrees with the fixed header",
        ));
    }
    let endianness = read_canonical_text(path, bytes, &mut cursor, "storage.endianness")?;
    if endianness != "little-endian" {
        return Err(index_corrupted(
            path,
            "v2 storage identity must be canonical little-endian",
        ));
    }
    let normalization =
        read_canonical_text(path, bytes, &mut cursor, "storage.vector_normalization")?;
    let normalization = normalization.to_owned();
    let dimension = read_canonical_u32(path, bytes, &mut cursor, "storage.dimension")?;
    if dimension != expected_dimension {
        return Err(index_corrupted(
            path,
            "v2 storage identity dimension disagrees with the fixed header",
        ));
    }
    if cursor != bytes.len() {
        return Err(index_corrupted(
            path,
            "v2 storage identity canonical bytes contain a trailing suffix",
        ));
    }
    let identity = VectorStorageIdentityV1 {
        schema_version: schema,
        format: format.to_owned(),
        quantization: quantization_format,
        endianness: endianness.to_owned(),
        vector_normalization: normalization,
        dimension,
    };
    identity.validate().map_err(|error| {
        index_corrupted(
            path,
            format!("v2 vector-storage identity is incomplete or invalid: {error}"),
        )
    })?;
    if identity.canonical_bytes() != bytes {
        return Err(index_corrupted(
            path,
            "v2 vector-storage bytes are not the exact canonical encoding",
        ));
    }
    Ok(identity.vector_normalization)
}

fn read_canonical_bytes<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<&'a [u8]> {
    let len_u64 = u64::from_be_bytes(read_array::<8>(
        path,
        data,
        cursor,
        &format!("{field}.len"),
    )?);
    let len = usize::try_from(len_u64)
        .map_err(|_| index_corrupted(path, format!("{field} length does not fit in usize")))?;
    read_slice(path, data, cursor, len, field)
}

fn read_canonical_text<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<&'a str> {
    let bytes = read_canonical_bytes(path, data, cursor, field)?;
    std::str::from_utf8(bytes)
        .map_err(|error| index_corrupted(path, format!("{field} is not UTF-8: {error}")))
}

fn read_canonical_u8(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u8> {
    Ok(read_array::<1>(path, data, cursor, field)?[0])
}

fn read_canonical_u16(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u16> {
    Ok(u16::from_be_bytes(read_array::<2>(
        path, data, cursor, field,
    )?))
}

fn read_canonical_u32(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u32> {
    Ok(u32::from_be_bytes(read_array::<4>(
        path, data, cursor, field,
    )?))
}

fn read_canonical_u64(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<u64> {
    Ok(u64::from_be_bytes(read_array::<8>(
        path, data, cursor, field,
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StableFileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    hard_links: u64,
    uid: u32,
    gid: u32,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
    accessed_seconds: i64,
    accessed_nanoseconds: i64,
}

#[cfg(unix)]
fn stable_file_identity(metadata: &fs::Metadata) -> StableFileIdentity {
    use std::os::unix::fs::MetadataExt;

    StableFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        hard_links: metadata.nlink(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
        accessed_seconds: metadata.atime(),
        accessed_nanoseconds: metadata.atime_nsec(),
    }
}

#[cfg(not(unix))]
fn stable_file_identity(metadata: &fs::Metadata) -> StableFileIdentity {
    StableFileIdentity {
        device: 0,
        inode: 0,
        mode: 0,
        hard_links: 1,
        uid: 0,
        gid: 0,
        size: metadata.len(),
        modified_seconds: 0,
        modified_nanoseconds: 0,
        changed_seconds: 0,
        changed_nanoseconds: 0,
        accessed_seconds: 0,
        accessed_nanoseconds: 0,
    }
}

fn snapshot_rejected(
    reason: FsviSnapshotRejectionReason,
    detail: impl Into<String>,
) -> FsviAdmissionError {
    FsviAdmissionError::SnapshotRejected(FsviSnapshotRejected {
        reason,
        detail: detail.into(),
    })
}

fn validate_single_link_regular_file(
    metadata: &fs::Metadata,
) -> Result<StableFileIdentity, FsviAdmissionError> {
    if metadata.file_type().is_symlink() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::SymbolicLink,
            "the final FSVI path must not be a symbolic link",
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::NotRegularFile,
            "the final FSVI path must be a regular file",
        ));
    }
    let identity = stable_file_identity(metadata);
    if identity.hard_links != 1 {
        return Err(snapshot_rejected(
            FsviSnapshotRejectionReason::HardLinked,
            "the FSVI inode must have exactly one hard link",
        ));
    }
    Ok(identity)
}

fn ensure_published_wal_absent(path: &Path) -> Result<(), FsviAdmissionError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(snapshot_rejected(
            FsviSnapshotRejectionReason::PublishedWalPresent,
            "published immutable FSVI v2 generations require the WAL path to be absent",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SearchError::Io(error).into()),
    }
}

pub(crate) fn snapshot_parent_or_current(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_readonly_noatime_nofollow(path: &Path) -> Result<File, FsviAdmissionError> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOATIME | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            let raw_error = error.raw_os_error();
            if raw_error == Some(libc::EPERM)
                || raw_error == Some(libc::EINVAL)
                || raw_error == Some(libc::EOPNOTSUPP)
            {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::NoAtimeUnsupported,
                    "O_NOATIME was denied or unsupported; admission will not weaken timestamp preservation",
                )
            } else if raw_error == Some(libc::ELOOP) {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI path became a symbolic link while it was being opened",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn open_readonly_noatime_nofollow(_: &Path) -> Result<File, FsviAdmissionError> {
    Err(snapshot_rejected(
        FsviSnapshotRejectionReason::NoAtimeUnsupported,
        "this target has no supported safe O_NOATIME pathname-open implementation",
    ))
}

struct PublishedFsviPathSnapshot {
    path: PathBuf,
    wal_path: PathBuf,
    parent: PathBuf,
    opened_file: File,
    file_identity: StableFileIdentity,
    parent_identity: StableFileIdentity,
    bytes: Arc<[u8]>,
}

impl PublishedFsviPathSnapshot {
    fn read(path: &Path) -> Result<Self, FsviAdmissionError> {
        let path_metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(SearchError::IndexNotFound {
                    path: path.to_path_buf(),
                }
                .into());
            }
            Err(error) => return Err(SearchError::Io(error).into()),
        };
        let file_identity = validate_single_link_regular_file(&path_metadata)?;
        let parent = snapshot_parent_or_current(path).to_path_buf();
        let parent_metadata = fs::symlink_metadata(&parent).map_err(SearchError::Io)?;
        if !parent_metadata.file_type().is_dir() {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "the FSVI parent must be a real directory, not a symlink or special file",
            ));
        }
        let parent_identity = stable_file_identity(&parent_metadata);
        let wal_path = wal::wal_path_for(path);
        ensure_published_wal_absent(&wal_path)?;

        let mut opened_file = open_readonly_noatime_nofollow(path)?;
        let opened_identity =
            validate_single_link_regular_file(&opened_file.metadata().map_err(SearchError::Io)?)?;
        if opened_identity != file_identity {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI pathname and opened descriptor identify different immutable bytes",
            ));
        }
        let byte_len = usize::try_from(file_identity.size).map_err(|_| {
            index_corrupted(
                path,
                "FSVI byte length does not fit in this process address space",
            )
        })?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(byte_len)
            .map_err(|_| SearchError::InvalidConfig {
                field: "fsvi_snapshot.byte_len".to_owned(),
                value: byte_len.to_string(),
                reason: "unable to reserve the exact immutable byte image".to_owned(),
            })?;
        owned.resize(byte_len, 0);
        opened_file.read_exact(&mut owned).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI inode was truncated while its byte image was being owned",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })?;
        let mut trailing = [0_u8; 1];
        if opened_file.read(&mut trailing).map_err(SearchError::Io)? != 0 {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI inode grew while its byte image was being owned",
            ));
        }
        let snapshot = Self {
            path: path.to_path_buf(),
            wal_path,
            parent,
            opened_file,
            file_identity,
            parent_identity,
            bytes: Arc::from(owned),
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    fn verify(&self) -> Result<(), FsviAdmissionError> {
        let descriptor_metadata = self.opened_file.metadata().map_err(SearchError::Io)?;
        let descriptor_identity = validate_single_link_regular_file(&descriptor_metadata)?;
        let path_metadata = fs::symlink_metadata(&self.path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                snapshot_rejected(
                    FsviSnapshotRejectionReason::PathChangedDuringRead,
                    "the FSVI pathname disappeared while its byte image was being validated",
                )
            } else {
                FsviAdmissionError::Index(SearchError::Io(error))
            }
        })?;
        let path_identity = validate_single_link_regular_file(&path_metadata)?;
        if descriptor_identity != self.file_identity || path_identity != self.file_identity {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::PathChangedDuringRead,
                "the FSVI inode identity, size, mode, links, or timestamps changed during admission",
            ));
        }
        ensure_published_wal_absent(&self.wal_path)?;
        let parent_metadata = fs::symlink_metadata(&self.parent).map_err(SearchError::Io)?;
        if !parent_metadata.file_type().is_dir()
            || stable_file_identity(&parent_metadata) != self.parent_identity
        {
            return Err(snapshot_rejected(
                FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
                "the FSVI containing directory changed during admission",
            ));
        }
        Ok(())
    }
}

fn validate_expected_v2_binding(
    path: &Path,
    metadata: &VectorMetadata,
    expected: &FsviV2IdentityBinding,
) -> Result<(), FsviAdmissionError> {
    let Some(actual) = metadata.identity_v2.as_ref() else {
        return Err(index_corrupted(path, "v2 metadata omitted identity bindings").into());
    };
    if actual.generation != expected.generation
        || actual.generation_fingerprint != expected.generation_fingerprint
    {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::GenerationMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted full-width artifact generation differs from the caller-owned generation; rebuild or select the exact published generation"
                    .to_owned(),
            },
        ));
    }
    let persisted_storage_bytes = &actual.storage_identity_canonical_bytes;
    let expected_storage_bytes = &expected.storage_canonical_bytes;
    let storage_mismatch = metadata.dimension != expected.dimension
        || metadata.quantization != expected.quantization
        || persisted_storage_bytes != expected_storage_bytes
        || actual.storage_fingerprint != expected.storage_fingerprint;
    if storage_mismatch {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::StorageMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted storage, dimension, or quantization differs from the caller-owned storage identity; source reindex is required"
                    .to_owned(),
            },
        ));
    }
    if actual.identity_bundle_canonical_bytes != expected.frozen_identity.canonical_bytes
        || actual.space_identity_canonical_bytes != expected.space_canonical_bytes
        || actual.identity_bundle_fingerprint != expected.bundle_fingerprint
        || actual.space_fingerprint != expected.space_fingerprint
        || actual.producer_fingerprint != expected.producer_fingerprint
        || actual.input_fingerprint != expected.input_fingerprint
    {
        return Err(FsviAdmissionError::ReindexRequired(
            FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                found_version: FSVI_V2_VERSION,
                detail: "persisted canonical embedding identity differs from the caller-owned identity; same names or dimensions must never be adopted or relabeled"
                    .to_owned(),
            },
        ));
    }
    Ok(())
}

fn validate_v2_layout_len(
    path: &Path,
    metadata: &VectorMetadata,
    header_len: usize,
    actual_len: usize,
) -> SearchResult<(usize, usize, usize)> {
    let records_bytes = metadata
        .record_count
        .checked_mul(RECORD_SIZE_BYTES)
        .ok_or_else(|| index_corrupted(path, "record table size overflow"))?;
    let records_offset = header_len;
    let strings_offset = records_offset
        .checked_add(records_bytes)
        .ok_or_else(|| index_corrupted(path, "record table offset overflow"))?;
    let vectors_offset = usize::try_from(metadata.vectors_offset)
        .map_err(|_| index_corrupted(path, "vectors_offset does not fit in usize"))?;
    if metadata.vectors_offset % VECTOR_ALIGN_BYTES != 0 {
        return Err(index_corrupted(
            path,
            format!(
                "v2 vectors_offset must be {VECTOR_ALIGN_BYTES}-byte aligned, found {}",
                metadata.vectors_offset
            ),
        ));
    }
    if vectors_offset < strings_offset {
        return Err(index_corrupted(
            path,
            "vectors_offset points inside the record table/string table region",
        ));
    }
    let vector_bytes = metadata
        .record_count
        .checked_mul(metadata.dimension)
        .and_then(|count| count.checked_mul(metadata.quantization.bytes_per_element()))
        .ok_or_else(|| index_corrupted(path, "vector slab size overflow"))?;
    let required_len = vectors_offset
        .checked_add(vector_bytes)
        .ok_or_else(|| index_corrupted(path, "vector slab end overflow"))?;
    if actual_len != required_len {
        return Err(index_corrupted(
            path,
            format!(
                "v2 file length must exactly match the bound layout: found {actual_len}, expected {required_len}"
            ),
        ));
    }
    Ok((records_offset, strings_offset, vectors_offset))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValidatedV2ContentStats {
    live_count: u64,
    tombstone_count: u64,
}

fn validate_v2_records_and_content(
    path: &Path,
    data: &[u8],
    metadata: &VectorMetadata,
    records_offset: usize,
    strings_offset: usize,
    vectors_offset: usize,
) -> SearchResult<ValidatedV2ContentStats> {
    let identity = metadata
        .identity_v2
        .as_ref()
        .ok_or_else(|| index_corrupted(path, "v2 metadata omitted identity bindings"))?;
    let mut expected_string_offset = 0usize;
    let mut previous: Option<(u64, &str)> = None;
    let mut live_count = 0_u64;
    let mut tombstone_count = 0_u64;
    for index in 0..metadata.record_count {
        let offset =
            records_offset
                .checked_add(index.checked_mul(RECORD_SIZE_BYTES).ok_or_else(|| {
                    index_corrupted(path, "record offset multiplication overflow")
                })?)
                .ok_or_else(|| index_corrupted(path, "record offset overflow"))?;
        let end = offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record end overflow"))?;
        if end > strings_offset || end > data.len() {
            return Err(index_corrupted(
                path,
                "v2 record table extends beyond its bound region",
            ));
        }
        let record = &data[offset..end];
        let doc_id_hash = u64::from_le_bytes(
            record[..8]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 record hash is truncated"))?,
        );
        let doc_id_offset = usize::try_from(u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 doc_id_offset is truncated"))?,
        ))
        .map_err(|_| index_corrupted(path, "v2 doc_id_offset does not fit in usize"))?;
        let doc_id_len = usize::from(u16::from_le_bytes(
            record[12..14]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 doc_id_len is truncated"))?,
        ));
        let flags = u16::from_le_bytes(
            record[14..16]
                .try_into()
                .map_err(|_| index_corrupted(path, "v2 record flags are truncated"))?,
        );
        if flags & !RECORD_FLAG_TOMBSTONE != 0 {
            return Err(index_corrupted(
                path,
                format!("v2 record {index} uses unsupported flags {flags:#06x}"),
            ));
        }
        if is_tombstoned_flags(flags) {
            tombstone_count = tombstone_count
                .checked_add(1)
                .ok_or_else(|| index_corrupted(path, "tombstone count overflow"))?;
        } else {
            live_count = live_count
                .checked_add(1)
                .ok_or_else(|| index_corrupted(path, "live count overflow"))?;
        }
        if doc_id_offset != expected_string_offset {
            return Err(index_corrupted(
                path,
                "v2 document strings must be contiguous in record order",
            ));
        }
        if doc_id_len == 0 {
            return Err(index_corrupted(path, "v2 document ids must not be empty"));
        }
        let doc_start = strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(path, "v2 document id start overflow"))?;
        let doc_end = doc_start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "v2 document id end overflow"))?;
        if doc_end > vectors_offset || doc_end > data.len() {
            return Err(index_corrupted(
                path,
                "v2 document id extends beyond the bound string table",
            ));
        }
        let doc_id = std::str::from_utf8(&data[doc_start..doc_end]).map_err(|error| {
            index_corrupted(
                path,
                format!("v2 document id at record {index} is not UTF-8: {error}"),
            )
        })?;
        if fnv1a_hash(doc_id.as_bytes()) != doc_id_hash {
            return Err(index_corrupted(
                path,
                format!("v2 document hash mismatch at record {index}"),
            ));
        }
        if previous.is_some_and(|prior| prior >= (doc_id_hash, doc_id)) {
            return Err(index_corrupted(
                path,
                "v2 records must be strictly sorted with unique document ids",
            ));
        }
        previous = Some((doc_id_hash, doc_id));
        expected_string_offset = expected_string_offset
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "v2 string table length overflow"))?;
    }
    let string_end = strings_offset
        .checked_add(expected_string_offset)
        .ok_or_else(|| index_corrupted(path, "v2 string table end overflow"))?;
    if string_end > vectors_offset {
        return Err(index_corrupted(
            path,
            "v2 string table overlaps the vector slab",
        ));
    }
    if data[string_end..vectors_offset]
        .iter()
        .any(|byte| *byte != 0)
    {
        return Err(index_corrupted(
            path,
            "v2 alignment padding must be all zero",
        ));
    }
    let mut docset_hasher = Sha256::new();
    update_digest_domain(&mut docset_hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    docset_hasher.update(live_count.to_be_bytes());
    for index in 0..metadata.record_count {
        let record_offset = records_offset
            .checked_add(
                index
                    .checked_mul(RECORD_SIZE_BYTES)
                    .ok_or_else(|| index_corrupted(path, "record offset overflow"))?,
            )
            .ok_or_else(|| index_corrupted(path, "record offset overflow"))?;
        let record_end = record_offset
            .checked_add(RECORD_SIZE_BYTES)
            .ok_or_else(|| index_corrupted(path, "record end overflow"))?;
        let record = &data[record_offset..record_end];
        let flags = u16::from_le_bytes(
            record[14..16]
                .try_into()
                .map_err(|_| index_corrupted(path, "record flags are truncated"))?,
        );
        if is_tombstoned_flags(flags) {
            continue;
        }
        let doc_id_offset = usize::try_from(u32::from_le_bytes(
            record[8..12]
                .try_into()
                .map_err(|_| index_corrupted(path, "doc_id_offset is truncated"))?,
        ))
        .map_err(|_| index_corrupted(path, "doc_id_offset does not fit in usize"))?;
        let doc_id_len = usize::from(u16::from_le_bytes(
            record[12..14]
                .try_into()
                .map_err(|_| index_corrupted(path, "doc_id_len is truncated"))?,
        ));
        let doc_start = strings_offset
            .checked_add(doc_id_offset)
            .ok_or_else(|| index_corrupted(path, "document id start overflow"))?;
        let doc_end = doc_start
            .checked_add(doc_id_len)
            .ok_or_else(|| index_corrupted(path, "document id end overflow"))?;
        docset_hasher.update(
            u64::try_from(doc_id_len)
                .map_err(|_| index_corrupted(path, "v2 doc_id length does not fit in u64"))?
                .to_be_bytes(),
        );
        docset_hasher.update(&data[doc_start..doc_end]);
    }
    let observed_docset_digest: [u8; SHA256_BYTES] = docset_hasher.finalize().into();
    if observed_docset_digest != identity.ordered_live_docset_digest {
        return Err(index_corrupted(
            path,
            "v2 ordered live-docset digest mismatch",
        ));
    }
    let observed_vector_digest = vector_content_digest_from_bytes(
        metadata.record_count,
        metadata.dimension,
        metadata.quantization,
        &data[vectors_offset..],
    )?;
    if observed_vector_digest != identity.vector_content_digest {
        return Err(index_corrupted(path, "v2 vector-content digest mismatch"));
    }
    Ok(ValidatedV2ContentStats {
        live_count,
        tombstone_count,
    })
}

fn fsvi_v2_header_len(binding: &FsviV2IdentityBinding) -> SearchResult<usize> {
    FSVI_V2_FIXED_PREFIX_BYTES
        .checked_add(binding.frozen_identity.canonical_bytes.len())
        .and_then(|length| length.checked_add(binding.space_canonical_bytes.len()))
        .and_then(|length| length.checked_add(binding.storage_canonical_bytes.len()))
        .and_then(|length| length.checked_add(4))
        .filter(|length| *length <= FSVI_V2_MAX_HEADER_BYTES)
        .ok_or_else(|| {
            fsvi_v2_config_error(
                "header_size",
                "canonical identity payloads overflow the bounded v2 header",
            )
        })
}

fn build_v2_header(
    binding: &FsviV2IdentityBinding,
    record_count: usize,
    vectors_offset: u64,
    ordered_docset_digest: [u8; SHA256_BYTES],
    vector_content_digest: [u8; SHA256_BYTES],
) -> SearchResult<Vec<u8>> {
    let header_len = fsvi_v2_header_len(binding)?;
    let header_size_u32 = u32::try_from(header_len)
        .map_err(|_| fsvi_v2_config_error("header_size", "must fit in the v2 u32 header field"))?;
    let dimension_u32 = u32::try_from(binding.dimension)
        .map_err(|_| fsvi_v2_config_error("dimension", "must fit in u32"))?;
    let record_count_u64 = u64::try_from(record_count)
        .map_err(|_| fsvi_v2_config_error("record_count", "must fit in u64"))?;
    let bundle_len = u32::try_from(binding.frozen_identity.canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("identity_bundle_len", "must fit in u32"))?;
    let space_len = u32::try_from(binding.space_canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("space_identity_len", "must fit in u32"))?;
    let storage_len = u32::try_from(binding.storage_canonical_bytes.len())
        .map_err(|_| fsvi_v2_config_error("storage_identity_len", "must fit in u32"))?;

    let mut header = Vec::with_capacity(header_len);
    header.extend_from_slice(&FSVI_MAGIC);
    header.extend_from_slice(&FSVI_V2_VERSION.to_le_bytes());
    header.extend_from_slice(&header_size_u32.to_le_bytes());
    header.extend_from_slice(&FSVI_V2_IDENTITY_BINDING_SCHEMA.to_le_bytes());
    header.push(binding.quantization as u8);
    header.push(0);
    header.extend_from_slice(&0_u16.to_le_bytes());
    header.extend_from_slice(&dimension_u32.to_le_bytes());
    header.extend_from_slice(&record_count_u64.to_le_bytes());
    header.extend_from_slice(&vectors_offset.to_le_bytes());
    header.extend_from_slice(&binding.generation.schema_version.to_le_bytes());
    header.extend_from_slice(&0_u16.to_le_bytes());
    header.extend_from_slice(&binding.generation.sequence.to_le_bytes());
    header.extend_from_slice(&binding.generation.nonce);
    header.extend_from_slice(&bundle_len.to_le_bytes());
    header.extend_from_slice(&space_len.to_le_bytes());
    header.extend_from_slice(&storage_len.to_le_bytes());
    header.extend_from_slice(&binding.bundle_fingerprint);
    header.extend_from_slice(&binding.space_fingerprint);
    header.extend_from_slice(&binding.producer_fingerprint);
    header.extend_from_slice(&binding.input_fingerprint);
    header.extend_from_slice(&binding.storage_fingerprint);
    header.extend_from_slice(&binding.generation_fingerprint);
    header.extend_from_slice(&ordered_docset_digest);
    header.extend_from_slice(&vector_content_digest);
    header.extend_from_slice(&binding.frozen_identity.canonical_bytes);
    header.extend_from_slice(&binding.space_canonical_bytes);
    header.extend_from_slice(&binding.storage_canonical_bytes);
    let crc = crc32(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    if header.len() != header_len {
        return Err(fsvi_v2_config_error(
            "header_size",
            "encoded bytes disagree with the precomputed v2 header size",
        ));
    }
    Ok(header)
}

fn read_array<const N: usize>(
    path: &Path,
    data: &[u8],
    cursor: &mut usize,
    field: &str,
) -> SearchResult<[u8; N]> {
    let slice = read_slice(path, data, cursor, N, field)?;
    let mut out = [0_u8; N];
    out.copy_from_slice(slice);
    Ok(out)
}

fn read_slice<'a>(
    path: &Path,
    data: &'a [u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> SearchResult<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| index_corrupted(path, format!("{field} offset overflow")))?;
    if end > data.len() {
        return Err(index_corrupted(
            path,
            format!("{field} is truncated (wanted {len} bytes)"),
        ));
    }
    let out = &data[*cursor..end];
    *cursor = end;
    Ok(out)
}

fn build_header_prefix(
    embedder_id: &str,
    embedder_revision: &str,
    dimension: usize,
    quantization: Quantization,
    compaction_gen: u8,
    record_count: usize,
    vectors_offset: u64,
) -> SearchResult<Vec<u8>> {
    validate_header_string(embedder_id, "embedder_id")?;
    validate_header_string(embedder_revision, "embedder_revision")?;
    let dimension_u32 = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension must fit in u32".to_owned(),
    })?;
    let record_count_u64 = u64::try_from(record_count).map_err(|_| SearchError::InvalidConfig {
        field: "record_count".to_owned(),
        value: record_count.to_string(),
        reason: "record_count must fit in u64".to_owned(),
    })?;
    let mut out = Vec::with_capacity(
        4 + 2 + 2 + embedder_id.len() + 2 + embedder_revision.len() + 4 + 1 + 3 + 8 + 8,
    );
    out.extend_from_slice(&FSVI_MAGIC);
    out.extend_from_slice(&FSVI_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_id.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_id".to_owned(),
                value: embedder_id.to_owned(),
                reason: "embedder_id byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_id.as_bytes());
    out.extend_from_slice(
        &u16::try_from(embedder_revision.len())
            .map_err(|_| SearchError::InvalidConfig {
                field: "embedder_revision".to_owned(),
                value: embedder_revision.to_owned(),
                reason: "embedder_revision byte length must fit in u16".to_owned(),
            })?
            .to_le_bytes(),
    );
    out.extend_from_slice(embedder_revision.as_bytes());
    out.extend_from_slice(&dimension_u32.to_le_bytes());
    out.push(quantization as u8);
    out.push(compaction_gen);
    out.extend_from_slice(&[0_u8; 2]);
    out.extend_from_slice(&record_count_u64.to_le_bytes());
    out.extend_from_slice(&vectors_offset.to_le_bytes());
    Ok(out)
}

fn validate_header_string(value: &str, field: &str) -> SearchResult<()> {
    if value.is_empty() && field == "embedder_id" {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: "embedder_id cannot be empty".to_owned(),
        });
    }
    let _ = u16::try_from(value.len()).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: "value length must fit in u16".to_owned(),
    })?;
    Ok(())
}

fn quantization_from_identity(quantization: QuantizationFormat) -> SearchResult<Quantization> {
    match quantization {
        QuantizationFormat::F32 => Ok(Quantization::F32),
        QuantizationFormat::F16 => Ok(Quantization::F16),
        QuantizationFormat::Int8 | QuantizationFormat::Int4 => Err(fsvi_v2_config_error(
            "storage.quantization",
            "FSVI v2 vector slabs currently support only F32 or F16",
        )),
    }
}

fn fsvi_v2_config_error(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("fsvi_v2.{field}"),
        value: "redacted".to_owned(),
        reason: reason.to_owned(),
    }
}

fn sha256_bytes(bytes: &[u8]) -> [u8; SHA256_BYTES] {
    Sha256::digest(bytes).into()
}

fn fingerprint_hex(fingerprint: &[u8; SHA256_BYTES]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(SHA256_BYTES * 2);
    for byte in fingerprint {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn decode_sha256_fingerprint(field: &str, value: &str) -> SearchResult<[u8; SHA256_BYTES]> {
    if value.len() != SHA256_BYTES * 2 {
        return Err(fsvi_v2_config_error(
            field,
            "must be a lowercase 64-character SHA-256 fingerprint",
        ));
    }
    let mut decoded = [0_u8; SHA256_BYTES];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().as_chunks::<2>().0) {
        let high = decode_lower_hex_nibble(pair[0]).ok_or_else(|| {
            fsvi_v2_config_error(
                field,
                "must be a lowercase 64-character SHA-256 fingerprint",
            )
        })?;
        let low = decode_lower_hex_nibble(pair[1]).ok_or_else(|| {
            fsvi_v2_config_error(
                field,
                "must be a lowercase 64-character SHA-256 fingerprint",
            )
        })?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

fn decode_sha256_fingerprint_from_index(
    path: &Path,
    field: &str,
    value: &str,
) -> SearchResult<[u8; SHA256_BYTES]> {
    decode_sha256_fingerprint(field, value).map_err(|_| {
        index_corrupted(
            path,
            format!("v2 {field} fingerprint is not lowercase SHA-256"),
        )
    })
}

const fn decode_lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn update_digest_domain(hasher: &mut Sha256, domain: &[u8]) {
    hasher.update(
        u64::try_from(domain.len())
            .unwrap_or(u64::MAX)
            .to_be_bytes(),
    );
    hasher.update(domain);
}

fn ordered_docset_digest(records: &[PendingRecord]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    update_digest_domain(&mut hasher, ORDERED_DOCSET_DIGEST_DOMAIN);
    let live_count = records
        .iter()
        .filter(|record| !is_tombstoned_flags(record.flags))
        .count();
    hasher.update(u64::try_from(live_count).unwrap_or(u64::MAX).to_be_bytes());
    for record in records
        .iter()
        .filter(|record| !is_tombstoned_flags(record.flags))
    {
        hasher.update(
            u64::try_from(record.doc_id.len())
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        hasher.update(record.doc_id.as_bytes());
    }
    hasher.finalize().into()
}

fn vector_content_hasher(
    record_count: usize,
    dimension: usize,
    quantization: Quantization,
) -> SearchResult<Sha256> {
    let mut hasher = Sha256::new();
    update_digest_domain(&mut hasher, VECTOR_CONTENT_DIGEST_DOMAIN);
    hasher.update(
        u64::try_from(record_count)
            .map_err(|_| fsvi_v2_config_error("record_count", "must fit in u64"))?
            .to_be_bytes(),
    );
    hasher.update(
        u64::try_from(dimension)
            .map_err(|_| fsvi_v2_config_error("dimension", "must fit in u64"))?
            .to_be_bytes(),
    );
    hasher.update([quantization as u8]);
    Ok(hasher)
}

fn write_vector_slab_v2<W: Write>(
    writer: &mut W,
    records: &[PendingRecord],
    dimension: usize,
    quantization: Quantization,
) -> SearchResult<[u8; SHA256_BYTES]> {
    let mut hasher = vector_content_hasher(records.len(), dimension, quantization)?;
    let element_bytes = quantization.bytes_per_element();
    let record_bytes = dimension
        .checked_mul(element_bytes)
        .ok_or_else(|| fsvi_v2_config_error("vector_slab", "record byte length overflow"))?;
    let mut encoded = Vec::with_capacity(record_bytes);
    match quantization {
        Quantization::F16 => {
            let mut scratch = Vec::<f16>::with_capacity(dimension);
            for record in records {
                scratch.clear();
                crate::simd::encode_f32_to_f16_extend(&record.embedding, &mut scratch);
                encoded.clear();
                for value in &scratch {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                hasher.update(&encoded);
                writer.write_all(&encoded)?;
            }
        }
        Quantization::F32 => {
            for record in records {
                encoded.clear();
                for value in &record.embedding {
                    encoded.extend_from_slice(&value.to_le_bytes());
                }
                hasher.update(&encoded);
                writer.write_all(&encoded)?;
            }
        }
    }
    Ok(hasher.finalize().into())
}

fn vector_content_digest_from_bytes(
    record_count: usize,
    dimension: usize,
    quantization: Quantization,
    vector_bytes: &[u8],
) -> SearchResult<[u8; SHA256_BYTES]> {
    let expected_len = record_count
        .checked_mul(dimension)
        .and_then(|count| count.checked_mul(quantization.bytes_per_element()))
        .ok_or_else(|| fsvi_v2_config_error("vector_slab", "byte length overflow"))?;
    if vector_bytes.len() != expected_len {
        return Err(fsvi_v2_config_error(
            "vector_slab",
            "bytes disagree with the declared shape",
        ));
    }
    let mut hasher = vector_content_hasher(record_count, dimension, quantization)?;
    hasher.update(vector_bytes);
    Ok(hasher.finalize().into())
}

fn write_vector_slab<W: Write>(
    writer: &mut W,
    records: &[PendingRecord],
    quantization: Quantization,
) -> SearchResult<()> {
    match quantization {
        Quantization::F16 => {
            // Encode each record's f32 → f16 via the F16C-dispatched kernel and
            // write the whole record's bytes in one `write_all`, instead of a
            // per-element `from_f32` + 2-byte `write_all` (which was ~38M tiny
            // writes for a 100k×384 index). The on-disk format is LE f16.
            let dim = records.first().map_or(0, |r| r.embedding.len());
            let mut scratch: Vec<f16> = Vec::with_capacity(dim);
            for record in records {
                scratch.clear();
                crate::simd::encode_f32_to_f16_extend(&record.embedding, &mut scratch);
                #[cfg(target_endian = "little")]
                {
                    // SAFETY: `half::f16` is `repr(transparent)` over `u16`; on a
                    // little-endian target the native bytes equal `to_le_bytes`, so
                    // the slab is byte-identical to the per-element path.
                    #[allow(unsafe_code)]
                    let bytes = unsafe {
                        core::slice::from_raw_parts(
                            scratch.as_ptr().cast::<u8>(),
                            scratch.len() * 2,
                        )
                    };
                    writer.write_all(bytes)?;
                }
                #[cfg(not(target_endian = "little"))]
                {
                    for &h in &scratch {
                        writer.write_all(&h.to_le_bytes())?;
                    }
                }
            }
        }
        Quantization::F32 => {
            for record in records {
                for value in &record.embedding {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> SearchResult<u64> {
    if alignment == 0 {
        return Ok(value);
    }
    let add = alignment
        .checked_sub(1)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: alignment.to_string(),
            reason: "alignment underflow".to_owned(),
        })?;
    let padded = value
        .checked_add(add)
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "alignment".to_owned(),
            value: format!("{value}+{add}"),
            reason: "alignment overflow".to_owned(),
        })?;
    Ok((padded / alignment) * alignment)
}

fn temporary_output_path(path: &Path) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let mut os = path.as_os_str().to_os_string();
    os.push(format!(".tmp.{pid}.{now}"));
    PathBuf::from(os)
}

pub(crate) fn sync_parent_directory(path: &Path) -> SearchResult<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn index_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Crc32::new();
    hasher.update(data);
    hasher.finalize()
}

pub(crate) fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

/// Whether a vector can contribute retrieval signal: every component is
/// finite and the squared norm is non-zero and finite. An all-zero vector
/// scores 0.0 against everything, so ranking it is arbitrary tie-breaking;
/// a norm that overflows to infinity poisons downstream scores.
pub(crate) fn vector_signal_usable(vector: &[f32]) -> bool {
    let mut norm_sq = 0.0f32;
    for &value in vector {
        if !value.is_finite() {
            return false;
        }
        norm_sq += value * value;
    }
    norm_sq > 0.0 && norm_sq.is_finite()
}

const fn is_tombstoned_flags(flags: u16) -> bool {
    flags & RECORD_FLAG_TOMBSTONE != 0
}

const fn next_generation(current: u8) -> u8 {
    if current == 255 { 1 } else { current + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_index_path(name: &str) -> PathBuf {
        static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let serial = NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "frankensearch-index-{name}-{}-{now}-{serial}",
            std::process::id(),
        ));
        fs::create_dir(&directory).expect("create private index fixture directory");
        directory.join("index.fsvi")
    }

    fn sample_vector(base: f32, dim: usize) -> Vec<f32> {
        vec![base; dim]
    }

    fn fsvi_v2_binding(
        model_id: &str,
        dimension: u32,
        quantization: Quantization,
        sequence: u64,
        nonce_byte: u8,
    ) -> FsviV2IdentityBinding {
        let mut identity =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                model_id, dimension,
            );
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = match quantization {
            Quantization::F32 => QuantizationFormat::F32,
            Quantization::F16 => QuantizationFormat::F16,
        };
        identity.storage.endianness = "little-endian".to_owned();
        let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
            .expect("valid test generation");
        FsviV2IdentityBinding::new(
            generation,
            identity.freeze().expect("valid frozen test identity"),
        )
        .expect("valid FSVI v2 binding")
    }

    fn semantic_fsvi_v2_binding(
        model_id: &str,
        dimension: u32,
        sequence: u64,
        nonce_byte: u8,
    ) -> FsviV2IdentityBinding {
        let mut identity =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                model_id, dimension,
            );
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "1".repeat(64);
        identity.space.artifacts = vec![
            EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "2".repeat(64),
                size: 20,
            },
            EmbeddingArtifactIdentityV1 {
                role: "tokenizer".to_owned(),
                sha256: "3".repeat(64),
                size: 10,
            },
        ];
        identity.space.tokenizer_fingerprint = "3".repeat(64);
        identity.space.vocabulary_fingerprint = "4".repeat(64);
        identity.space.model_config_fingerprint = "5".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        let generation = ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
            .expect("valid semantic test generation");
        FsviV2IdentityBinding::new(
            generation,
            identity.freeze().expect("valid semantic test identity"),
        )
        .expect("valid semantic FSVI v2 binding")
    }

    fn fsvi_v2_binding_with_input_variant(
        model_id: &str,
        dimension: u32,
        sequence: u64,
        nonce_byte: u8,
        chunking: &str,
    ) -> FsviV2IdentityBinding {
        let mut identity =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                model_id, dimension,
            );
        identity.input.chunking = chunking.to_owned();
        identity.space.input_contract_fingerprint = identity.input.fingerprint();
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(sequence, [nonce_byte; 16])
                .expect("valid variant generation"),
            identity.freeze().expect("valid variant frozen identity"),
        )
        .expect("valid variant FSVI v2 binding")
    }

    fn refresh_v2_header_crc(bytes: &mut [u8]) {
        let header_size = usize::try_from(u32::from_le_bytes(
            bytes[6..10].try_into().expect("header-size field"),
        ))
        .expect("header size fits usize");
        let crc_offset = header_size.checked_sub(4).expect("CRC offset");
        let crc = crc32(&bytes[..crc_offset]);
        bytes[crc_offset..header_size].copy_from_slice(&crc.to_le_bytes());
    }

    fn assert_inspection_corrupted(path: &Path) {
        assert!(
            matches!(
                VectorIndex::inspect(path),
                Err(SearchError::IndexCorrupted { .. })
            ),
            "expected typed inspection to reject corrupted bytes at {}",
            path.display()
        );
    }

    fn admit_owned_v2_fixture(
        path: &Path,
        binding: &FsviV2IdentityBinding,
    ) -> Result<ValidatedFsviBytes, FsviAdmissionError> {
        let bytes = fs::read(path).map_err(SearchError::Io)?;
        ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(bytes), binding)
    }

    fn assert_owned_admission_corrupted(path: &Path, binding: &FsviV2IdentityBinding) {
        assert!(
            matches!(
                admit_owned_v2_fixture(path, binding),
                Err(FsviAdmissionError::Index(
                    SearchError::IndexCorrupted { .. }
                ))
            ),
            "expected admitted open to reject corrupted bytes at {}",
            path.display()
        );
    }

    fn assert_snapshot_rejection<T: std::fmt::Debug>(
        result: Result<T, FsviAdmissionError>,
        expected_reason: FsviSnapshotRejectionReason,
    ) {
        match result {
            Err(FsviAdmissionError::SnapshotRejected(rejected)) => {
                assert_eq!(rejected.reason, expected_reason);
                assert!(!rejected.detail.is_empty());
            }
            other => panic!("expected snapshot rejection {expected_reason:?}, observed {other:?}"),
        }
    }

    fn directory_entry_names(path: &Path) -> Vec<std::ffi::OsString> {
        let mut names: Vec<_> = fs::read_dir(path)
            .expect("read private fixture directory")
            .map(|entry| entry.expect("read fixture entry").file_name())
            .collect();
        names.sort_unstable();
        names
    }

    fn write_v2_fixture(
        name: &str,
        binding: FsviV2IdentityBinding,
    ) -> (PathBuf, FsviV2IdentityBinding) {
        let path = temp_index_path(name);
        let expected = binding.clone();
        let mut writer = VectorIndex::create_v2(&path, binding).expect("create v2 writer");
        writer
            .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
            .expect("write beta");
        writer
            .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
            .expect("write alpha");
        writer.finish().expect("finish v2 fixture");
        (path, expected)
    }

    #[test]
    fn validated_fsvi_bytes_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ValidatedFsviBytes>();
    }

    #[test]
    fn snapshot_parent_maps_only_missing_or_empty_parents_to_current_directory() {
        assert_eq!(
            snapshot_parent_or_current(Path::new("index.fsvi")),
            Path::new(".")
        );
        assert_eq!(
            snapshot_parent_or_current(Path::new("./index.fsvi")),
            Path::new(".")
        );
        assert_eq!(
            snapshot_parent_or_current(Path::new("nested/index.fsvi")),
            Path::new("nested")
        );
    }

    #[test]
    fn fsvi_v2_round_trip_inspects_and_admits_exact_identity() {
        let binding = fsvi_v2_binding("v2-round-trip", 4, Quantization::F16, u64::MAX, 0xa5);
        let (path, expected) = write_v2_fixture("v2-round-trip", binding);

        let inspected = VectorIndex::inspect(&path).expect("inspect v2");
        assert!(matches!(&inspected, FsviInspection::V2IdentityComplete(_)));
        let FsviInspection::V2IdentityComplete(metadata) = inspected else {
            return;
        };
        assert_eq!(metadata.fsvi_version, FSVI_V2_VERSION);
        assert_eq!(metadata.dimension, 4);
        assert_eq!(metadata.quantization, Quantization::F16);
        assert_eq!(metadata.record_count, 2);
        assert_eq!(
            metadata
                .identity_v2
                .as_ref()
                .expect("v2 identity metadata")
                .generation,
            expected.generation()
        );
        assert_eq!(expected.generation().sequence, u64::MAX);
        assert_eq!(metadata.vectors_offset % VECTOR_ALIGN_BYTES, 0);

        let mut index =
            admit_owned_v2_fixture(&path, &expected).expect("exact owned admission succeeds");
        assert!(index.is_identity_admitted_v2());
        assert!(!index.published_wal_absent());
        assert!(index.owner_and_search_share_allocation());
        assert_eq!(index.record_count(), 2);
        assert_eq!(index.embedder_id(), "v2-round-trip");
        assert_eq!(index.embedder_revision(), "explicit-test-v1");
        let identity = index.identity_v2();
        assert_ne!(identity.identity_bundle_fingerprint, [0; SHA256_BYTES]);
        assert_ne!(identity.ordered_live_docset_digest, [0; SHA256_BYTES]);
        assert_ne!(identity.vector_content_digest, [0; SHA256_BYTES]);

        let alpha = index
            .find_index_by_doc_hash(fnv1a_hash(b"doc-alpha"))
            .expect("find alpha");
        assert_eq!(index.doc_id_at(alpha).expect("alpha id"), "doc-alpha");
        let vector = index.vector_at_f32(alpha).expect("alpha vector");
        assert!((vector[0] - 1.0).abs() < 0.002);

        let query = [1.0, 0.0, 0.0, 0.0];
        let owner_hits = index
            .search_top_k(&query, 2, None)
            .expect("owner exact search");
        let normal_hits = index
            .index
            .search_top_k(&query, 2, None)
            .expect("normal exact search over same allocation");
        assert_eq!(owner_hits.len(), normal_hits.len());
        for (owner_hit, normal_hit) in owner_hits.iter().zip(&normal_hits) {
            assert_eq!(owner_hit.index, normal_hit.index);
            assert_eq!(owner_hit.doc_id, normal_hit.doc_id);
            assert_eq!(owner_hit.score.to_bits(), normal_hit.score.to_bits());
        }

        let serialized = serde_json::to_vec(index.witness()).expect("serialize witness");
        let round_trip: FsviV2Witness =
            serde_json::from_slice(&serialized).expect("deserialize witness");
        assert_eq!(&round_trip, index.witness());

        let rows = index.row_source();
        assert_eq!(rows.witness(), index.witness());
        assert_eq!(
            rows.identity_bundle_fingerprint(),
            &identity.identity_bundle_fingerprint
        );
        assert_eq!(rows.space_fingerprint(), &identity.space_fingerprint);
        assert_eq!(rows.producer_fingerprint(), &identity.producer_fingerprint);
        assert_eq!(rows.input_fingerprint(), &identity.input_fingerprint);
        assert_eq!(rows.storage_fingerprint(), &identity.storage_fingerprint);
        assert_eq!(
            rows.generation_fingerprint(),
            &identity.generation_fingerprint
        );
        assert_eq!(
            rows.row(alpha).expect("owner row").vector_bytes(),
            index.index.vector_bytes(alpha).expect("normal row bytes")
        );

        let before_bytes = index.bytes.to_vec();
        let before_witness = index.witness().clone();
        let mutation = index
            .index
            .set_record_flags(alpha, FsviRecordFlags::TOMBSTONE.bits());
        assert!(matches!(
            mutation,
            Err(SearchError::InvalidConfig { field, .. }) if field == "fsvi_v2.mutation"
        ));
        assert_eq!(index.bytes.as_ref(), before_bytes.as_slice());
        assert_eq!(index.witness(), &before_witness);

        assert_eq!(
            index.ann_admission(),
            FsviAnnAdmission::Disabled(FsviAnnDisabledReason::OwnerBoundAdapterUnavailable)
        );
        assert!(!wal::wal_path_for(&path).exists());
    }

    #[test]
    fn fsvi_v2_empty_artifact_is_identity_bound_and_admissible() {
        let path = temp_index_path("v2-empty");
        let binding = fsvi_v2_binding("v2-empty", 4, Quantization::F32, 0, 0x11);
        VectorIndex::create_v2(&path, binding.clone())
            .expect("create empty v2")
            .finish()
            .expect("finish empty v2");

        let index = admit_owned_v2_fixture(&path, &binding).expect("admit empty owned v2 artifact");
        assert_eq!(index.record_count(), 0);
        let identity = index.identity_v2();
        assert_ne!(identity.ordered_live_docset_digest, [0; SHA256_BYTES]);
        assert_ne!(identity.vector_content_digest, [0; SHA256_BYTES]);
        assert_eq!(
            usize::try_from(index.metadata().vectors_offset).expect("offset fits"),
            usize::try_from(fs::metadata(&path).expect("empty artifact metadata").len())
                .expect("file length fits")
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn sealed_owner_survives_path_replacement_and_exact_reopen_rejects_new_bytes() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let path = directory.path().join("current.fsvi");
        let replacement_path = directory.path().join("replacement.fsvi");
        let retained_path = directory.path().join("retained-original.fsvi");
        let binding = fsvi_v2_binding(
            "same-display-and-generation",
            4,
            Quantization::F16,
            21,
            0x81,
        );

        let mut original = VectorIndex::create_v2(&path, binding.clone()).expect("original writer");
        original
            .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
            .expect("original alpha");
        original
            .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
            .expect("original beta");
        original.finish().expect("finish original");
        let owner =
            ValidatedFsviBytes::open_published(&path, &binding).expect("admit original owner");
        let expected_witness = owner.witness().clone();
        let expected_hits = owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search original");
        let expected_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
            .map(|index| {
                let row = owner.row(index).expect("original row");
                (
                    row.doc_id().to_owned(),
                    row.vector_bytes().to_vec(),
                    row.flags(),
                )
            })
            .collect();

        let mut replacement =
            VectorIndex::create_v2(&replacement_path, binding.clone()).expect("replacement writer");
        replacement
            .write_record("doc-alpha", &[0.0, 1.0, 0.0, 0.0])
            .expect("replacement alpha");
        replacement
            .write_record("doc-beta", &[1.0, 0.0, 0.0, 0.0])
            .expect("replacement beta");
        replacement.finish().expect("finish replacement");
        fs::rename(&path, &retained_path).expect("retain original inode");
        fs::rename(&replacement_path, &path).expect("publish replacement inode");

        assert_eq!(owner.witness(), &expected_witness);
        let observed_hits = owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search retained owner");
        assert_eq!(observed_hits.len(), expected_hits.len());
        for (observed, expected) in observed_hits.iter().zip(&expected_hits) {
            assert_eq!(observed.index, expected.index);
            assert_eq!(observed.doc_id, expected.doc_id);
            assert_eq!(observed.score.to_bits(), expected.score.to_bits());
        }
        let observed_rows: Vec<(String, Vec<u8>, FsviRecordFlags)> = (0..owner.record_count())
            .map(|index| {
                let row = owner.row(index).expect("retained owner row");
                (
                    row.doc_id().to_owned(),
                    row.vector_bytes().to_vec(),
                    row.flags(),
                )
            })
            .collect();
        assert_eq!(observed_rows, expected_rows);

        assert!(matches!(
            ValidatedFsviBytes::reopen_exact(&path, &binding, &expected_witness),
            Err(FsviAdmissionError::SnapshotRejected(FsviSnapshotRejected {
                reason: FsviSnapshotRejectionReason::WitnessMismatch,
                ..
            }))
        ));
        let replacement_owner =
            ValidatedFsviBytes::open_published(&path, &binding).expect("admit replacement");
        assert_eq!(
            replacement_owner.witness().ordered_live_docset_digest,
            expected_witness.ordered_live_docset_digest
        );
        assert_ne!(
            replacement_owner.witness().vector_content_digest,
            expected_witness.vector_content_digest
        );
        assert_ne!(
            replacement_owner.witness().whole_image_sha256,
            expected_witness.whole_image_sha256
        );
    }

    #[test]
    fn same_display_strings_and_dimension_cannot_cross_open_distinct_identity_bundles() {
        let path_a = temp_index_path("v2-same-display-identity-a");
        let path_b = temp_index_path("v2-same-display-identity-b");
        let binding_a = fsvi_v2_binding_with_input_variant("same-display", 4, 22, 0x82, "none-v1");
        let binding_b = fsvi_v2_binding_with_input_variant("same-display", 4, 22, 0x82, "none-v2");
        for (path, binding) in [(&path_a, &binding_a), (&path_b, &binding_b)] {
            let mut writer =
                VectorIndex::create_v2(path, binding.clone()).expect("identity writer");
            writer
                .write_record("same-doc", &[1.0, 0.0, 0.0, 0.0])
                .expect("identity record");
            writer.finish().expect("identity finish");
        }

        let owner_a = admit_owned_v2_fixture(&path_a, &binding_a).expect("owner a");
        let owner_b = admit_owned_v2_fixture(&path_b, &binding_b).expect("owner b");
        assert_eq!(owner_a.embedder_id(), owner_b.embedder_id());
        assert_eq!(owner_a.embedder_revision(), owner_b.embedder_revision());
        assert_eq!(owner_a.dimension(), owner_b.dimension());
        assert_ne!(
            owner_a.witness().identity_bundle_fingerprint,
            owner_b.witness().identity_bundle_fingerprint
        );
        assert_ne!(owner_a.witness(), owner_b.witness());
        assert!(matches!(
            admit_owned_v2_fixture(&path_a, &binding_b),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                ..
            }))
        ));
        assert!(matches!(
            admit_owned_v2_fixture(&path_b, &binding_a),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                ..
            }))
        ));
    }

    #[test]
    fn tombstones_are_physical_but_absent_from_live_digest_and_exact_search() {
        let path = temp_index_path("v2-tombstone-owner");
        let binding = fsvi_v2_binding("v2-tombstone-owner", 4, Quantization::F32, 23, 0x83);
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("tombstone writer");
        writer
            .write_tombstone_record("best-but-dead", &[1.0, 0.0, 0.0, 0.0])
            .expect("tombstone row");
        writer
            .write_record("live-result", &[0.0, 1.0, 0.0, 0.0])
            .expect("live row");
        writer.finish().expect("finish tombstone fixture");

        let owner = admit_owned_v2_fixture(&path, &binding).expect("tombstone owner");
        assert_eq!(owner.record_count(), 2);
        assert_eq!(owner.live_count(), 1);
        assert_eq!(owner.tombstone_count(), 1);
        let rows = owner.row_source();
        let states: Vec<(String, FsviRecordFlags)> = (0..rows.record_count())
            .map(|index| {
                let row = rows.row(index).expect("validated row");
                (row.doc_id().to_owned(), row.flags())
            })
            .collect();
        assert!(states.contains(&("best-but-dead".to_owned(), FsviRecordFlags::TOMBSTONE)));
        assert!(states.contains(&("live-result".to_owned(), FsviRecordFlags::LIVE)));

        let hits = owner
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("exact tombstone search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "live-result");

        let mut expected_docset = Sha256::new();
        update_digest_domain(&mut expected_docset, ORDERED_DOCSET_DIGEST_DOMAIN);
        expected_docset.update(1_u64.to_be_bytes());
        expected_docset.update(
            u64::try_from("live-result".len())
                .expect("live id length")
                .to_be_bytes(),
        );
        expected_docset.update(b"live-result");
        let expected_docset: [u8; SHA256_BYTES] = expected_docset.finalize().into();
        assert_eq!(owner.witness().ordered_live_docset_digest, expected_docset);
        assert_eq!(
            owner.ann_admission(),
            FsviAnnAdmission::Disabled(FsviAnnDisabledReason::OwnerBoundAdapterUnavailable)
        );
    }

    #[test]
    fn all_live_docset_digest_remains_byte_compatible_with_xomn_formula() {
        let binding = fsvi_v2_binding("v2-all-live-compatible", 4, Quantization::F16, 24, 0x84);
        let (path, expected) = write_v2_fixture("v2-all-live-compatible", binding);
        let owner = admit_owned_v2_fixture(&path, &expected).expect("all-live owner");
        let mut prior_formula = Sha256::new();
        update_digest_domain(&mut prior_formula, ORDERED_DOCSET_DIGEST_DOMAIN);
        prior_formula.update(
            u64::try_from(owner.record_count())
                .expect("record count fits")
                .to_be_bytes(),
        );
        for index in 0..owner.record_count() {
            let doc_id = owner.doc_id_at(index).expect("all-live id");
            prior_formula.update(
                u64::try_from(doc_id.len())
                    .expect("doc id length fits")
                    .to_be_bytes(),
            );
            prior_formula.update(doc_id.as_bytes());
        }
        let prior_formula: [u8; SHA256_BYTES] = prior_formula.finalize().into();
        assert_eq!(owner.witness().ordered_live_docset_digest, prior_formula);
        assert_eq!(owner.live_count(), owner.record_count());
        assert_eq!(owner.tombstone_count(), 0);
    }

    #[test]
    fn v2_owner_accepts_unicode_and_maximum_u16_document_ids() {
        let path = temp_index_path("v2-unicode-boundary-ids");
        let binding = fsvi_v2_binding("v2-unicode-boundary-ids", 4, Quantization::F16, 25, 0x85);
        let boundary_id = "x".repeat(usize::from(u16::MAX));
        let unicode_id = "航海図-🏴‍☠️-café";
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("boundary writer");
        writer
            .write_record(&boundary_id, &[1.0, 0.0, 0.0, 0.0])
            .expect("maximum id");
        writer
            .write_record(unicode_id, &[0.0, 1.0, 0.0, 0.0])
            .expect("unicode id");
        writer.finish().expect("finish boundary fixture");

        let owner = admit_owned_v2_fixture(&path, &binding).expect("boundary owner");
        let ids: std::collections::HashSet<String> = (0..owner.record_count())
            .map(|index| owner.doc_id_at(index).expect("boundary id").to_owned())
            .collect();
        assert!(ids.contains(&boundary_id));
        assert!(ids.contains(unicode_id));
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    #[test]
    fn published_open_fails_closed_when_noatime_is_unsupported() {
        let binding = fsvi_v2_binding("v2-noatime-unsupported", 4, Quantization::F16, 26, 0x86);
        let (path, expected) = write_v2_fixture("v2-noatime-unsupported", binding);

        assert_snapshot_rejection(
            VectorIndex::open_admitted_v2(&path, &expected),
            FsviSnapshotRejectionReason::NoAtimeUnsupported,
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn published_open_preserves_literal_bytes_metadata_directory_and_wal_absence() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let path = directory.path().join("generation.fsvi");
        let binding = fsvi_v2_binding("v2-side-effect-free", 4, Quantization::F16, 26, 0x86);
        let mut writer =
            VectorIndex::create_v2(&path, binding.clone()).expect("side-effect writer");
        writer
            .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("side-effect row");
        writer.finish().expect("finish side-effect fixture");

        let expected_bytes = fs::read(&path).expect("read expected image");
        let timestamp = UNIX_EPOCH + Duration::from_secs(1_600_000_123);
        let file = OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open fixture to freeze timestamps");
        file.set_times(
            std::fs::FileTimes::new()
                .set_accessed(timestamp)
                .set_modified(timestamp),
        )
        .expect("freeze fixture timestamps");
        drop(file);

        let wal_path = wal::wal_path_for(&path);
        assert!(!wal_path.exists());
        let entries_before = directory_entry_names(directory.path());
        let file_before =
            stable_file_identity(&fs::symlink_metadata(&path).expect("metadata before"));
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata before"),
        );

        let owner =
            ValidatedFsviBytes::open_published(&path, &binding).expect("side-effect-free open");

        let file_after =
            stable_file_identity(&fs::symlink_metadata(&path).expect("metadata after"));
        let parent_after = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata after"),
        );
        assert_eq!(file_after, file_before);
        assert_eq!(parent_after, parent_before);
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        assert!(!wal_path.exists());
        assert_eq!(owner.owned_byte_len(), expected_bytes.len());
        assert_eq!(owner.bytes.as_ref(), expected_bytes.as_slice());
        assert!(owner.published_wal_absent());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn published_open_rejects_symlink_hardlink_and_nonregular_paths() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("private topology directory");
        let target = directory.path().join("target.fsvi");
        let symlink_path = directory.path().join("symlink.fsvi");
        let hardlink_path = directory.path().join("hardlink.fsvi");
        let nonregular_path = directory.path().join("directory.fsvi");
        let binding = fsvi_v2_binding("v2-path-topology", 4, Quantization::F16, 27, 0x87);
        VectorIndex::create_v2(&target, binding.clone())
            .expect("topology writer")
            .finish()
            .expect("finish topology fixture");

        symlink(&target, &symlink_path).expect("create final-component symlink");
        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&symlink_path, &binding),
            FsviSnapshotRejectionReason::SymbolicLink,
        );

        fs::hard_link(&target, &hardlink_path).expect("create hardlink alias");
        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&target, &binding),
            FsviSnapshotRejectionReason::HardLinked,
        );
        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&hardlink_path, &binding),
            FsviSnapshotRejectionReason::HardLinked,
        );

        fs::create_dir(&nonregular_path).expect("create nonregular final path");
        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&nonregular_path, &binding),
            FsviSnapshotRejectionReason::NotRegularFile,
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn stable_snapshot_detects_path_and_parent_replacement_after_read() {
        let path_directory = tempfile::tempdir().expect("private path-race directory");
        let current = path_directory.path().join("current.fsvi");
        let replacement = path_directory.path().join("replacement.fsvi");
        let retained = path_directory.path().join("retained.fsvi");
        let binding = fsvi_v2_binding("v2-path-race", 4, Quantization::F16, 28, 0x88);
        for path in [&current, &replacement] {
            VectorIndex::create_v2(path, binding.clone())
                .expect("path-race writer")
                .finish()
                .expect("finish path-race fixture");
        }
        let path_snapshot =
            PublishedFsviPathSnapshot::read(&current).expect("take stable path snapshot");
        fs::rename(&current, &retained).expect("retain read inode");
        fs::rename(&replacement, &current).expect("replace publication pathname");
        assert_snapshot_rejection(
            path_snapshot.verify(),
            FsviSnapshotRejectionReason::PathChangedDuringRead,
        );

        let parent_directory = tempfile::tempdir().expect("private parent-race directory");
        let parent_path = parent_directory.path().join("current.fsvi");
        VectorIndex::create_v2(&parent_path, binding)
            .expect("parent-race writer")
            .finish()
            .expect("finish parent-race fixture");
        let parent_snapshot =
            PublishedFsviPathSnapshot::read(&parent_path).expect("take stable parent snapshot");
        fs::write(
            parent_directory.path().join("publisher-marker"),
            b"published",
        )
        .expect("mutate containing directory");
        assert_snapshot_rejection(
            parent_snapshot.verify(),
            FsviSnapshotRejectionReason::DirectoryChangedDuringRead,
        );
    }

    #[test]
    fn reserved_record_flag_matrix_is_not_constructible_or_admissible() {
        assert_eq!(
            FsviRecordFlags::from_bits(FsviRecordFlags::LIVE.bits()).expect("live flags"),
            FsviRecordFlags::LIVE
        );
        assert_eq!(
            FsviRecordFlags::from_bits(FsviRecordFlags::TOMBSTONE.bits()).expect("tombstone flags"),
            FsviRecordFlags::TOMBSTONE
        );
        for bits in [0x0002_u16, 0x0003, u16::MAX] {
            assert!(matches!(
                FsviRecordFlags::from_bits(bits),
                Err(SearchError::IndexCorrupted { .. })
            ));
        }

        let binding = fsvi_v2_binding("v2-reserved-flags", 4, Quantization::F16, 29, 0x89);
        let (path, expected) = write_v2_fixture("v2-reserved-flags", binding);
        let source = fs::read(path).expect("read flag fixture");
        let header_size = usize::try_from(u32::from_le_bytes(
            source[6..10].try_into().expect("header size"),
        ))
        .expect("header size fits");
        for bits in [0x0002_u16, 0x0003, u16::MAX] {
            let mut mutated = source.clone();
            mutated[header_size + 14..header_size + 16].copy_from_slice(&bits.to_le_bytes());
            assert!(matches!(
                ValidatedFsviBytes::from_arc(Arc::<[u8]>::from(mutated), &expected),
                Err(FsviAdmissionError::Index(
                    SearchError::IndexCorrupted { .. }
                ))
            ));
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn published_open_rejects_every_wal_shape_without_mutating_it() {
        for shape in ["empty", "valid", "corrupt", "truncated"] {
            let directory = tempfile::tempdir().expect("private WAL publication directory");
            let path = directory.path().join(format!("{shape}.fsvi"));
            let binding = fsvi_v2_binding(
                &format!("v2-published-wal-{shape}"),
                4,
                Quantization::F16,
                30,
                0x8a,
            );
            VectorIndex::create_v2(&path, binding.clone())
                .expect("WAL publication writer")
                .finish()
                .expect("finish WAL publication fixture");
            let wal_path = wal::wal_path_for(&path);
            match shape {
                "empty" => fs::write(&wal_path, []).expect("write empty sidecar"),
                "valid" => wal::append_wal_batch(
                    &wal_path,
                    &[wal::WalEntry {
                        doc_id: "wal-doc".to_owned(),
                        doc_id_hash: fnv1a_hash(b"wal-doc"),
                        embedding: vec![1.0, 0.0, 0.0, 0.0],
                    }],
                    4,
                    Quantization::F16,
                    0,
                    false,
                )
                .expect("write valid sidecar"),
                "corrupt" => fs::write(&wal_path, b"not-a-wal").expect("write corrupt sidecar"),
                "truncated" => {
                    wal::append_wal_batch(
                        &wal_path,
                        &[wal::WalEntry {
                            doc_id: "wal-doc".to_owned(),
                            doc_id_hash: fnv1a_hash(b"wal-doc"),
                            embedding: vec![1.0, 0.0, 0.0, 0.0],
                        }],
                        4,
                        Quantization::F16,
                        0,
                        false,
                    )
                    .expect("write complete sidecar before truncation");
                    let mut truncated = fs::read(&wal_path).expect("read complete sidecar");
                    truncated.pop();
                    fs::write(&wal_path, truncated).expect("write truncated sidecar");
                }
                other => panic!("unhandled WAL shape {other}"),
            }

            let wal_bytes_before = fs::read(&wal_path).expect("read sidecar before rejection");
            let index_before =
                stable_file_identity(&fs::symlink_metadata(&path).expect("index metadata before"));
            let wal_before = stable_file_identity(
                &fs::symlink_metadata(&wal_path).expect("WAL metadata before"),
            );
            let entries_before = directory_entry_names(directory.path());
            let parent_before = stable_file_identity(
                &fs::symlink_metadata(directory.path()).expect("parent metadata before"),
            );

            assert_snapshot_rejection(
                ValidatedFsviBytes::open_published(&path, &binding),
                FsviSnapshotRejectionReason::PublishedWalPresent,
            );

            assert_eq!(
                stable_file_identity(&fs::symlink_metadata(&path).expect("index metadata after")),
                index_before
            );
            assert_eq!(
                stable_file_identity(&fs::symlink_metadata(&wal_path).expect("WAL metadata after")),
                wal_before
            );
            assert_eq!(
                stable_file_identity(
                    &fs::symlink_metadata(directory.path()).expect("parent metadata after")
                ),
                parent_before
            );
            assert_eq!(directory_entry_names(directory.path()), entries_before);
            assert_eq!(
                fs::read(&wal_path).expect("read sidecar after rejection"),
                wal_bytes_before
            );
        }
    }

    #[test]
    fn fsvi_v2_inspection_distinguishes_reindex_upgrade_and_corruption() {
        let legacy_path = temp_index_path("v2-inspect-legacy");
        VectorIndex::create(&legacy_path, "legacy-display-name", 4)
            .expect("legacy writer")
            .finish()
            .expect("legacy finish");
        assert!(matches!(
            VectorIndex::inspect(&legacy_path),
            Ok(FsviInspection::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::LegacyUnidentified,
                found_version: FSVI_VERSION,
                ..
            }))
        ));
        let expected = fsvi_v2_binding("v2-inspect", 4, Quantization::F16, 1, 0x22);
        assert!(matches!(
            admit_owned_v2_fixture(&legacy_path, &expected),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::LegacyUnidentified,
                ..
            }))
        ));

        let future_path = temp_index_path("v2-inspect-future");
        let mut future = Vec::from(FSVI_MAGIC);
        future.extend_from_slice(&(FSVI_V2_VERSION + 1).to_le_bytes());
        fs::write(&future_path, future).expect("write future prefix");
        assert!(matches!(
            VectorIndex::inspect(&future_path),
            Ok(FsviInspection::UpgradeRequired(FsviUpgradeRequired {
                found_version,
                supported_version: FSVI_V2_VERSION,
            })) if found_version == FSVI_V2_VERSION + 1
        ));
        assert!(matches!(
            admit_owned_v2_fixture(&future_path, &expected),
            Err(FsviAdmissionError::UpgradeRequired(FsviUpgradeRequired {
                found_version,
                ..
            })) if found_version == FSVI_V2_VERSION + 1
        ));

        let corrupt_path = temp_index_path("v2-inspect-corrupt");
        fs::write(&corrupt_path, b"FSV").expect("write truncated prefix");
        assert_inspection_corrupted(&corrupt_path);
    }

    #[test]
    fn fsvi_v2_admission_requires_exact_generation_storage_and_identity() {
        let binding = fsvi_v2_binding("v2-exact", 4, Quantization::F16, 7, 0x33);
        let (path, expected) = write_v2_fixture("v2-exact", binding);

        let wrong_generation = fsvi_v2_binding("v2-exact", 4, Quantization::F16, 8, 0x34);
        assert!(matches!(
            admit_owned_v2_fixture(&path, &wrong_generation),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::GenerationMismatch,
                ..
            }))
        ));

        let wrong_storage = fsvi_v2_binding("v2-exact", 4, Quantization::F32, 7, 0x33);
        assert!(matches!(
            admit_owned_v2_fixture(&path, &wrong_storage),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::StorageMismatch,
                ..
            }))
        ));

        let wrong_identity = fsvi_v2_binding(
            "same-dimension-different-space",
            4,
            Quantization::F16,
            7,
            0x33,
        );
        assert!(matches!(
            admit_owned_v2_fixture(&path, &wrong_identity),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                ..
            }))
        ));

        admit_owned_v2_fixture(&path, &expected).expect("control admission");
    }

    #[test]
    fn fsvi_v2_header_mutation_matrix_fails_closed() {
        let binding = fsvi_v2_binding("v2-header-matrix", 4, Quantization::F16, 9, 0x44);
        let (source_path, _) = write_v2_fixture("v2-header-matrix-source", binding);
        let source = fs::read(&source_path).expect("read source v2");
        let header_size = usize::try_from(u32::from_le_bytes(
            source[6..10].try_into().expect("header size"),
        ))
        .expect("header size fits");
        assert!(header_size > FSVI_V2_FIXED_PREFIX_BYTES + 4);

        let fixed_mutations = [
            ("binding-schema", 10usize),
            ("quantization", 12),
            ("header-flags", 13),
            ("dimension", 16),
            ("generation-schema", 36),
            ("generation-reserved", 38),
            ("generation-sequence", 40),
            ("generation-nonce", 48),
            ("bundle-fingerprint", FSVI_V2_BUNDLE_FINGERPRINT_OFFSET),
            ("space-fingerprint", FSVI_V2_SPACE_FINGERPRINT_OFFSET),
            ("producer-fingerprint", FSVI_V2_PRODUCER_FINGERPRINT_OFFSET),
            ("input-fingerprint", FSVI_V2_INPUT_FINGERPRINT_OFFSET),
            ("storage-fingerprint", FSVI_V2_STORAGE_FINGERPRINT_OFFSET),
            (
                "generation-fingerprint",
                FSVI_V2_GENERATION_FINGERPRINT_OFFSET,
            ),
        ];
        for (name, offset) in fixed_mutations {
            let path = temp_index_path(&format!("v2-header-{name}"));
            let mut mutated = source.clone();
            mutated[offset] ^= 0x01;
            refresh_v2_header_crc(&mut mutated);
            fs::write(&path, mutated).expect("write header mutation");
            assert_inspection_corrupted(&path);
        }

        let canonical_path = temp_index_path("v2-header-canonical");
        let mut canonical_mutation = source;
        canonical_mutation[FSVI_V2_FIXED_PREFIX_BYTES] ^= 0x01;
        refresh_v2_header_crc(&mut canonical_mutation);
        fs::write(&canonical_path, canonical_mutation).expect("write canonical mutation");
        assert_inspection_corrupted(&canonical_path);
    }

    #[test]
    fn fsvi_v2_complete_space_parser_rejects_suffix_and_dimension_drift() {
        let binding = fsvi_v2_binding("v2-complete-space", 4, Quantization::F16, 9, 0x45);
        let path = temp_index_path("v2-complete-space");

        let mut suffixed = binding.space_canonical_bytes.clone();
        suffixed.push(0);
        assert!(matches!(
            parse_complete_space_identity(&path, &suffixed, 4, binding.input_fingerprint,),
            Err(SearchError::IndexCorrupted { .. })
        ));

        let mut wrong_dimension = binding.frozen_identity.identity.space.clone();
        wrong_dimension.dimension = 5;
        assert!(matches!(
            parse_complete_space_identity(
                &path,
                &wrong_dimension.canonical_bytes(),
                4,
                binding.input_fingerprint,
            ),
            Err(SearchError::IndexCorrupted { .. })
        ));
    }

    #[test]
    fn fsvi_v2_complete_space_parser_round_trips_semantic_and_projection_branches() {
        let path = temp_index_path("v2-complete-space-semantic");
        let semantic = semantic_fsvi_v2_binding("semantic-space", 8, 9, 0x46);
        let parsed = parse_complete_space_identity(
            &path,
            &semantic.space_canonical_bytes,
            8,
            semantic.input_fingerprint,
        )
        .expect("complete semantic identity parses");
        assert_eq!(parsed.0, "semantic-space");
        assert_eq!(parsed.1, "explicit-test-v1");
        assert_eq!(
            parsed.2,
            semantic.frozen_identity.identity.space.output_normalization
        );

        let projected_identity = semantic
            .frozen_identity
            .identity
            .derive_projection(4, "first-four-components", "l2-after-projection")
            .expect("derive semantic projection");
        let projected = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(10, [0x47; 16]).expect("valid projected generation"),
            projected_identity
                .freeze()
                .expect("freeze projected identity"),
        )
        .expect("valid projected FSVI v2 binding");
        let parsed = parse_complete_space_identity(
            &path,
            &projected.space_canonical_bytes,
            4,
            projected.input_fingerprint,
        )
        .expect("complete projected identity parses");
        assert_eq!(parsed.0, "semantic-space");
        assert_eq!(parsed.2, "l2-after-projection");
    }

    #[test]
    fn fsvi_v2_content_mutation_matrix_fails_admission() {
        let binding = fsvi_v2_binding("v2-content-matrix", 4, Quantization::F16, 10, 0x55);
        let (source_path, expected) = write_v2_fixture("v2-content-matrix-source", binding);
        let source = fs::read(&source_path).expect("read source v2");
        let inspected = VectorIndex::inspect(&source_path).expect("inspect source");
        assert!(matches!(&inspected, FsviInspection::V2IdentityComplete(_)));
        let FsviInspection::V2IdentityComplete(metadata) = inspected else {
            return;
        };
        let identity = metadata.identity_v2.as_ref().expect("v2 metadata");
        let header_size = identity.header_size;
        let vectors_offset =
            usize::try_from(metadata.vectors_offset).expect("vector offset fits usize");
        let strings_offset = header_size + metadata.record_count * RECORD_SIZE_BYTES;
        let string_bytes = "doc-alpha".len() + "doc-beta".len();
        let padding_offset = strings_offset + string_bytes;
        assert!(
            padding_offset < vectors_offset,
            "fixture must include padding"
        );

        let content_mutations = [
            ("record-flags", header_size + 14),
            ("document-id", strings_offset),
            ("alignment-padding", padding_offset),
            ("vector-slab", vectors_offset),
        ];
        for (name, offset) in content_mutations {
            let path = temp_index_path(&format!("v2-content-{name}"));
            let mut mutated = source.clone();
            mutated[offset] ^= 0x01;
            fs::write(&path, mutated).expect("write content mutation");
            assert_owned_admission_corrupted(&path, &expected);
        }

        for (name, offset) in [
            ("docset-digest", FSVI_V2_DOCSET_DIGEST_OFFSET),
            ("vector-digest", FSVI_V2_VECTOR_DIGEST_OFFSET),
        ] {
            let path = temp_index_path(&format!("v2-content-{name}"));
            let mut mutated = source.clone();
            mutated[offset] ^= 0x01;
            refresh_v2_header_crc(&mut mutated);
            fs::write(&path, mutated).expect("write digest mutation");
            assert_owned_admission_corrupted(&path, &expected);
        }

        let trailing_path = temp_index_path("v2-content-trailing");
        let mut trailing = source.clone();
        trailing.push(0);
        fs::write(&trailing_path, trailing).expect("write trailing byte");
        assert_owned_admission_corrupted(&trailing_path, &expected);

        let truncated_path = temp_index_path("v2-content-truncated");
        fs::write(&truncated_path, &source[..source.len() - 1]).expect("write truncated vector");
        assert_owned_admission_corrupted(&truncated_path, &expected);

        let unaligned_path = temp_index_path("v2-content-unaligned-vector-slab");
        let mut unaligned = source.clone();
        let unaligned_vectors_offset = vectors_offset
            .checked_sub(1)
            .expect("fixture vector offset has alignment padding");
        assert!(
            unaligned_vectors_offset >= padding_offset,
            "fixture must retain a valid string table when one padding byte is removed"
        );
        unaligned.remove(unaligned_vectors_offset);
        unaligned[28..36].copy_from_slice(
            &u64::try_from(unaligned_vectors_offset)
                .expect("unaligned vector offset fits u64")
                .to_le_bytes(),
        );
        refresh_v2_header_crc(&mut unaligned);
        fs::write(&unaligned_path, unaligned).expect("write unaligned vector slab");
        assert_owned_admission_corrupted(&unaligned_path, &expected);

        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let wal_path = wal::wal_path_for(&source_path);
            fs::write(&wal_path, b"live-sidecar").expect("write live WAL sidecar");
            let sidecar_before = fs::read(&wal_path).expect("read sidecar before rejection");
            assert_snapshot_rejection(
                VectorIndex::open_admitted_v2(&source_path, &expected),
                FsviSnapshotRejectionReason::PublishedWalPresent,
            );
            assert_eq!(
                fs::read(&wal_path).expect("read preserved sidecar"),
                sidecar_before
            );
        }
    }

    #[test]
    fn fsvi_v2_writer_refuses_to_publish_beside_an_existing_wal() {
        let path = temp_index_path("v2-writer-existing-wal");
        let binding = fsvi_v2_binding("v2-writer-existing-wal", 4, Quantization::F16, 11, 0x65);
        let mut writer = VectorIndex::create_v2(&path, binding).expect("create v2 writer");
        writer
            .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("write record");
        let wal_path = wal::wal_path_for(&path);
        fs::write(&wal_path, b"stale-wal").expect("write stale WAL");

        let error = writer
            .finish()
            .expect_err("v2 publication must refuse every existing WAL sidecar");
        assert!(matches!(
            error,
            SearchError::InvalidConfig { ref field, .. } if field == "fsvi_v2.wal_sidecar"
        ));
        assert!(
            !path.exists(),
            "a rejected v2 generation must not publish its target"
        );
        assert!(
            wal_path.exists(),
            "refusal must preserve the pre-existing WAL for operator recovery"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn completed_sibling_publication_uses_sealed_bytes_and_preserves_evidence() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding(
            "completed-sibling-publication",
            4,
            Quantization::F16,
            31,
            0x91,
        );

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"stale destination WAL").expect("write stale WAL");

        let mut completed_writer =
            VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
        completed_writer
            .write_record("doc-alpha", &[1.0, 0.0, 0.0, 0.0])
            .expect("write alpha");
        completed_writer
            .write_record("doc-beta", &[0.0, 1.0, 0.0, 0.0])
            .expect("write beta");
        completed_writer.finish().expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("read completed evidence");
        let completed_owner = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
            .expect("admit completed evidence");
        let expected_witness = completed_owner.witness().clone();

        let published = ValidatedFsviBytes::publish_completed_sibling(
            &destination,
            &completed_sibling,
            &binding,
        )
        .expect("publish exact completed bytes");

        assert_eq!(published.witness(), &expected_witness);
        assert!(published.published_wal_absent());
        assert_eq!(
            fs::read(&destination).expect("read published generation"),
            completed_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains readable"),
            completed_bytes
        );
        assert!(
            !destination_wal.exists(),
            "stale destination WAL must be removed only after main publication"
        );
        let hits = published
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 1, None)
            .expect("search exact published owner");
        assert_eq!(hits[0].doc_id, "doc-alpha");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn completed_sibling_alias_path_is_rejected_without_touching_evidence() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let completed_sibling = directory.path().join("completed.fsvi");
        let destination_alias = directory.path().join(".").join("completed.fsvi");
        let binding = fsvi_v2_binding("completed-sibling-alias", 4, Quantization::F16, 36, 0x96);

        let mut writer =
            VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
        writer
            .write_record("doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("write record");
        writer.finish().expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed evidence");
        let entries_before = directory_entry_names(directory.path());

        let error = ValidatedFsviBytes::publish_completed_sibling(
            &destination_alias,
            &completed_sibling,
            &binding,
        )
        .expect_err("same-inode alias must be rejected");
        assert!(matches!(
            error,
            FsviAdmissionError::Index(SearchError::InvalidConfig {
                ref field,
                ref reason,
                ..
            }) if field == "fsvi_v2.completed_sibling"
                && reason.contains("same file")
        ));
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        let reopened = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
            .expect("completed evidence remains admissible");
        let expected_digest: [u8; SHA256_BYTES] = Sha256::digest(&completed_bytes).into();
        assert_eq!(reopened.witness().whole_image_sha256, expected_digest);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn completed_sibling_cannot_alias_destination_wal_path() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = wal::wal_path_for(&destination);
        let binding = fsvi_v2_binding(
            "completed-sibling-wal-alias",
            4,
            Quantization::F16,
            38,
            0x98,
        );

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");

        let mut writer =
            VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
        writer
            .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("write record");
        writer.finish().expect("finish completed sibling");
        let old_main_bytes = fs::read(&destination).expect("snapshot old main");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed evidence");
        let entries_before = directory_entry_names(directory.path());

        let error = ValidatedFsviBytes::publish_completed_sibling(
            &destination,
            &completed_sibling,
            &binding,
        )
        .expect_err("completed evidence at the WAL pathname must be rejected");
        assert!(matches!(
            error,
            FsviAdmissionError::Index(SearchError::InvalidConfig {
                ref field,
                ref reason,
                ..
            }) if field == "fsvi_v2.completed_sibling"
                && reason.contains("destination WAL")
        ));
        assert_eq!(
            fs::read(&destination).expect("old main remains"),
            old_main_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
        assert_eq!(directory_entry_names(directory.path()), entries_before);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn rejected_completed_sibling_leaves_old_main_and_wal_untouched() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let actual_binding =
            fsvi_v2_binding_with_input_variant("same-model", 4, 41, 0xa1, "actual-chunking");
        let expected_binding =
            fsvi_v2_binding_with_input_variant("same-model", 4, 41, 0xa1, "different-chunking");

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
        let old_main_bytes = fs::read(&destination).expect("snapshot old main");
        let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

        VectorIndex::create_v2(&completed_sibling, actual_binding)
            .expect("v2 writer")
            .finish()
            .expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
        let entries_before = directory_entry_names(directory.path());
        let destination_before = stable_file_identity(
            &fs::symlink_metadata(&destination).expect("destination metadata before rejection"),
        );
        let destination_wal_before = stable_file_identity(
            &fs::symlink_metadata(&destination_wal).expect("WAL metadata before rejection"),
        );
        let completed_before = stable_file_identity(
            &fs::symlink_metadata(&completed_sibling)
                .expect("completed evidence metadata before rejection"),
        );
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata before rejection"),
        );

        assert!(matches!(
            ValidatedFsviBytes::publish_completed_sibling(
                &destination,
                &completed_sibling,
                &expected_binding,
            ),
            Err(FsviAdmissionError::ReindexRequired(FsviReindexRequired {
                reason: FsviReindexReason::IdentityMismatch,
                ..
            }))
        ));
        let destination_after = stable_file_identity(
            &fs::symlink_metadata(&destination).expect("destination metadata after rejection"),
        );
        let destination_wal_after = stable_file_identity(
            &fs::symlink_metadata(&destination_wal).expect("WAL metadata after rejection"),
        );
        let completed_after = stable_file_identity(
            &fs::symlink_metadata(&completed_sibling)
                .expect("completed evidence metadata after rejection"),
        );
        let parent_after = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata after rejection"),
        );
        assert_eq!(destination_after, destination_before);
        assert_eq!(destination_wal_after, destination_wal_before);
        assert_eq!(completed_after, completed_before);
        assert_eq!(parent_after, parent_before);
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        assert_eq!(
            fs::read(&destination).expect("old main remains"),
            old_main_bytes
        );
        assert_eq!(
            fs::read(&destination_wal).expect("old WAL remains"),
            old_wal_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn pre_replace_failure_leaves_old_generation_and_completed_evidence_untouched() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding("pre-replace-failure", 4, Quantization::F16, 44, 0xa4);

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
        let old_main_bytes = fs::read(&destination).expect("snapshot old main");
        let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

        VectorIndex::create_v2(&completed_sibling, binding.clone())
            .expect("v2 writer")
            .finish()
            .expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

        let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
            &destination,
            &completed_sibling,
            &binding,
            |_| Ok(()),
            || {
                Err(SearchError::Cancelled {
                    phase: "before-fsvi-main-replace".to_owned(),
                    reason: "injected publication interruption".to_owned(),
                })
            },
            || Ok(()),
            || Ok(()),
        )
        .expect_err("injected boundary must stop before destination replacement");
        assert!(matches!(
            error,
            FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
                if phase == "before-fsvi-main-replace"
        ));
        assert_eq!(
            fs::read(&destination).expect("old main remains"),
            old_main_bytes
        );
        assert_eq!(
            fs::read(&destination_wal).expect("old WAL remains"),
            old_wal_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn corrupt_durable_temp_is_rejected_before_old_generation_is_touched() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding(
            "pre-publish-temp-validation",
            4,
            Quantization::F16,
            46,
            0xa6,
        );

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
        let old_main_bytes = fs::read(&destination).expect("snapshot old main");
        let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

        VectorIndex::create_v2(&completed_sibling, binding.clone())
            .expect("v2 writer")
            .finish()
            .expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

        assert!(matches!(
            ValidatedFsviBytes::publish_completed_sibling_with_hooks(
                &destination,
                &completed_sibling,
                &binding,
                |temporary_path| {
                    let mut corrupted = OpenOptions::new()
                        .write(true)
                        .truncate(true)
                        .open(temporary_path)?;
                    corrupted.write_all(&FSVI_MAGIC)?;
                    corrupted.sync_all()?;
                    Ok(())
                },
                || Ok(()),
                || Ok(()),
                || Ok(()),
            ),
            Err(FsviAdmissionError::Index(
                SearchError::IndexCorrupted { .. }
            ))
        ));
        assert_eq!(
            fs::read(&destination).expect("old main remains"),
            old_main_bytes
        );
        assert_eq!(
            fs::read(&destination_wal).expect("old WAL remains"),
            old_wal_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn post_main_sync_failure_leaves_v2_plus_wal_fail_closed() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding("post-main-sync-failure", 4, Quantization::F16, 51, 0xb1);

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
        let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

        let mut completed_writer =
            VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
        completed_writer
            .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("new record");
        completed_writer.finish().expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");

        let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
            &destination,
            &completed_sibling,
            &binding,
            |_| Ok(()),
            || Ok(()),
            || {
                Err(SearchError::Cancelled {
                    phase: "after-fsvi-main-sync".to_owned(),
                    reason: "injected publication interruption".to_owned(),
                })
            },
            || Ok(()),
        )
        .expect_err("injected boundary must stop before WAL removal");
        assert!(matches!(
            error,
            FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
                if phase == "after-fsvi-main-sync"
        ));
        assert_eq!(
            fs::read(&destination).expect("new v2 main remains durable"),
            completed_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
        assert_eq!(
            fs::read(&destination_wal).expect("old WAL remains fail-closed"),
            old_wal_bytes
        );
        assert_snapshot_rejection(
            ValidatedFsviBytes::open_published(&destination, &binding),
            FsviSnapshotRejectionReason::PublishedWalPresent,
        );
        assert!(matches!(
            VectorIndex::inspect(&destination),
            Ok(FsviInspection::V2IdentityComplete(_))
        ));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn post_wal_sync_failure_leaves_exact_admissible_v2_generation() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding("post-wal-sync-failure", 4, Quantization::F16, 56, 0xb6);

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");

        let mut completed_writer =
            VectorIndex::create_v2(&completed_sibling, binding.clone()).expect("v2 writer");
        completed_writer
            .write_record("new-doc", &[1.0, 0.0, 0.0, 0.0])
            .expect("new record");
        completed_writer.finish().expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
        let completed_owner = ValidatedFsviBytes::open_published(&completed_sibling, &binding)
            .expect("admit completed evidence");
        let completed_witness = completed_owner.witness().clone();

        let error = ValidatedFsviBytes::publish_completed_sibling_with_hooks(
            &destination,
            &completed_sibling,
            &binding,
            |_| Ok(()),
            || Ok(()),
            || Ok(()),
            || {
                Err(SearchError::Cancelled {
                    phase: "before-fsvi-final-reopen".to_owned(),
                    reason: "injected publication interruption".to_owned(),
                })
            },
        )
        .expect_err("injected boundary must stop before final exact reopen");
        assert!(matches!(
            error,
            FsviAdmissionError::Index(SearchError::Cancelled { ref phase, .. })
                if phase == "before-fsvi-final-reopen"
        ));
        assert_eq!(
            fs::read(&destination).expect("new v2 main remains durable"),
            completed_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
        assert!(
            !destination_wal.exists(),
            "WAL absence must already be durable before the final reopen boundary"
        );
        let reopened = ValidatedFsviBytes::reopen_exact(&destination, &binding, &completed_witness)
            .expect("post-WAL crash state is exact and admissible");
        assert_eq!(reopened.witness(), &completed_witness);
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    #[test]
    fn completed_sibling_publication_is_typed_and_non_mutating_without_noatime() {
        let directory = tempfile::tempdir().expect("private publication directory");
        let destination = directory.path().join("current.fsvi");
        let completed_sibling = directory.path().join("completed.fsvi");
        let binding = fsvi_v2_binding(
            "unsupported-noatime-publication",
            4,
            Quantization::F16,
            61,
            0xc1,
        );

        let mut old_writer =
            VectorIndex::create(&destination, "legacy-space", 4).expect("legacy writer");
        old_writer
            .write_record("old-doc", &[0.0, 0.0, 1.0, 0.0])
            .expect("old record");
        old_writer.finish().expect("finish old generation");
        let destination_wal = wal::wal_path_for(&destination);
        fs::write(&destination_wal, b"committed old-generation WAL").expect("write old WAL");
        let old_main_bytes = fs::read(&destination).expect("snapshot old main");
        let old_wal_bytes = fs::read(&destination_wal).expect("snapshot old WAL");

        VectorIndex::create_v2(&completed_sibling, binding.clone())
            .expect("v2 writer")
            .finish()
            .expect("finish completed sibling");
        let completed_bytes = fs::read(&completed_sibling).expect("snapshot completed sibling");
        let entries_before = directory_entry_names(directory.path());
        let destination_before = stable_file_identity(
            &fs::symlink_metadata(&destination).expect("destination metadata before rejection"),
        );
        let destination_wal_before = stable_file_identity(
            &fs::symlink_metadata(&destination_wal).expect("WAL metadata before rejection"),
        );
        let completed_before = stable_file_identity(
            &fs::symlink_metadata(&completed_sibling)
                .expect("completed evidence metadata before rejection"),
        );
        let parent_before = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata before rejection"),
        );

        assert_snapshot_rejection(
            ValidatedFsviBytes::publish_completed_sibling(
                &destination,
                &completed_sibling,
                &binding,
            ),
            FsviSnapshotRejectionReason::NoAtimeUnsupported,
        );
        let destination_after = stable_file_identity(
            &fs::symlink_metadata(&destination).expect("destination metadata after rejection"),
        );
        let destination_wal_after = stable_file_identity(
            &fs::symlink_metadata(&destination_wal).expect("WAL metadata after rejection"),
        );
        let completed_after = stable_file_identity(
            &fs::symlink_metadata(&completed_sibling)
                .expect("completed evidence metadata after rejection"),
        );
        let parent_after = stable_file_identity(
            &fs::symlink_metadata(directory.path()).expect("parent metadata after rejection"),
        );
        assert_eq!(destination_after, destination_before);
        assert_eq!(destination_wal_after, destination_wal_before);
        assert_eq!(completed_after, completed_before);
        assert_eq!(parent_after, parent_before);
        assert_eq!(directory_entry_names(directory.path()), entries_before);
        assert_eq!(
            fs::read(&destination).expect("old main remains"),
            old_main_bytes
        );
        assert_eq!(
            fs::read(&destination_wal).expect("old WAL remains"),
            old_wal_bytes
        );
        assert_eq!(
            fs::read(&completed_sibling).expect("completed evidence remains"),
            completed_bytes
        );
    }

    #[test]
    fn fsvi_v2_writer_rejects_ambiguous_records_and_invalid_bindings() {
        let duplicate_path = temp_index_path("v2-duplicate");
        let binding = fsvi_v2_binding("v2-writer-guard", 4, Quantization::F16, 11, 0x66);
        let mut duplicate =
            VectorIndex::create_v2(&duplicate_path, binding).expect("duplicate writer");
        duplicate
            .write_record("same", &[1.0, 0.0, 0.0, 0.0])
            .expect("first duplicate");
        duplicate
            .write_record("same", &[0.0, 1.0, 0.0, 0.0])
            .expect("second duplicate");
        assert!(matches!(
            duplicate.finish(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "doc_id"
        ));

        let empty_path = temp_index_path("v2-empty-doc-id");
        let binding = fsvi_v2_binding("v2-writer-guard", 4, Quantization::F16, 12, 0x67);
        let mut empty = VectorIndex::create_v2(&empty_path, binding).expect("empty-id writer");
        empty
            .write_record("", &[1.0, 0.0, 0.0, 0.0])
            .expect("record is rejected atomically at finish");
        assert!(matches!(
            empty.finish(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "doc_id"
        ));

        let mut wrong_format =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                "wrong-format",
                4,
            );
        wrong_format.storage.format = "fsvi-v1".to_owned();
        wrong_format.storage.quantization = QuantizationFormat::F16;
        wrong_format.storage.endianness = "little-endian".to_owned();
        let generation =
            ArtifactGenerationIdentityV1::new(1, [0x70; 16]).expect("valid generation");
        assert!(
            FsviV2IdentityBinding::new(
                generation,
                wrong_format
                    .freeze()
                    .expect("valid non-v2 storage identity")
            )
            .is_err()
        );

        let mut unsupported =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                "unsupported-quantization",
                4,
            );
        unsupported.storage.format = "fsvi-v2".to_owned();
        unsupported.storage.quantization = QuantizationFormat::Int8;
        unsupported.storage.endianness = "little-endian".to_owned();
        assert!(
            FsviV2IdentityBinding::new(
                generation,
                unsupported
                    .freeze()
                    .expect("valid but unsupported storage identity")
            )
            .is_err()
        );

        let mut frozen =
            frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                "noncanonical",
                4,
            );
        frozen.storage.format = "fsvi-v2".to_owned();
        frozen.storage.quantization = QuantizationFormat::F16;
        frozen.storage.endianness = "little-endian".to_owned();
        let mut frozen = frozen.freeze().expect("valid frozen identity");
        frozen.canonical_bytes.push(0);
        assert!(FsviV2IdentityBinding::new(generation, frozen).is_err());

        let invalid_generation = ArtifactGenerationIdentityV1 {
            schema_version: 1,
            sequence: 1,
            nonce: [0; 16],
        };
        let valid_binding = fsvi_v2_binding("v2-generation-guard", 4, Quantization::F16, 1, 0x77);
        assert!(
            FsviV2IdentityBinding::new(invalid_generation, valid_binding.frozen_identity().clone())
                .is_err()
        );
    }

    #[test]
    fn round_trip_f16_with_revision_and_lookup() {
        let path = temp_index_path("round-trip");
        let mut writer =
            VectorIndex::create_with_revision(&path, "fnv1a-384", "rev-123", 8, Quantization::F16)
                .expect("writer");
        writer
            .write_record("doc-b", &sample_vector(1.0, 8))
            .expect("write doc-b");
        writer
            .write_record("doc-a", &sample_vector(2.0, 8))
            .expect("write doc-a");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        assert_eq!(index.record_count(), 2);
        assert_eq!(index.dimension(), 8);
        assert_eq!(index.embedder_id(), "fnv1a-384");
        assert_eq!(index.embedder_revision(), "rev-123");
        assert_eq!(index.quantization(), Quantization::F16);
        assert_eq!(index.metadata().vectors_offset % VECTOR_ALIGN_BYTES, 0);

        let hash_a = fnv1a_hash(b"doc-a");
        let pos_a = index
            .find_index_by_doc_hash(hash_a)
            .expect("hash lookup should find doc-a");
        let doc_id = index.doc_id_at(pos_a).expect("doc id");
        assert_eq!(doc_id, "doc-a");
        let vec_a = index.vector_at_f32(pos_a).expect("vector");
        assert_eq!(vec_a.len(), 8);
        assert!((vec_a[0] - 2.0).abs() < 0.002);
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn owned_record_handoff_matches_borrowed_writer_bytes() {
        let borrowed_path = temp_index_path("borrowed-record-handoff");
        let owned_path = temp_index_path("owned-record-handoff");
        let records = vec![
            ("doc-z".to_owned(), vec![0.25, -0.5, 0.75, 1.0]),
            ("doc-a".to_owned(), vec![-1.0, 0.5, 0.0, 0.125]),
            ("doc-m".to_owned(), vec![0.0, 0.25, -0.25, 0.5]),
        ];

        let mut borrowed = VectorIndex::create(&borrowed_path, "parity", 4).expect("borrowed");
        for (doc_id, embedding) in &records {
            borrowed
                .write_record(doc_id, embedding)
                .expect("write borrowed record");
        }
        borrowed.finish().expect("finish borrowed");

        let mut owned = VectorIndex::create(&owned_path, "parity", 4).expect("owned");
        for (doc_id, embedding) in records {
            owned
                .write_record_owned_for_benchmark(doc_id, embedding)
                .expect("write owned record");
        }
        owned.finish().expect("finish owned");

        assert_eq!(
            fs::read(&borrowed_path).expect("read borrowed bytes"),
            fs::read(&owned_path).expect("read owned bytes")
        );
    }

    #[test]
    fn detects_header_crc_corruption() {
        let path = temp_index_path("crc");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-1", &sample_vector(0.5, 4))
            .expect("write");
        writer.finish().expect("finish");

        let mut bytes = fs::read(&path).expect("read index");
        // Flip a byte in the header payload before crc.
        bytes[6] ^= 0xAA;
        fs::write(&path, bytes).expect("rewrite corrupt index");

        let error = VectorIndex::open(&path).expect_err("corruption should be detected");
        assert!(matches!(error, SearchError::IndexCorrupted { .. }));
    }

    #[test]
    fn write_record_dimension_mismatch_is_error() {
        let path = temp_index_path("dim-mismatch");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 3).expect("writer");
        let error = writer
            .write_record("doc-1", &[1.0, 2.0])
            .expect_err("must reject wrong dimension");
        assert!(matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[test]
    fn empty_index_round_trip() {
        let path = temp_index_path("empty");
        let writer = VectorIndex::create(&path, "fnv1a-384", 16).expect("writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.record_count(), 0);
        assert_eq!(index.dimension(), 16);
    }

    #[test]
    fn get_embeddings_returns_none_for_missing_hashes() {
        let path = temp_index_path("get-embeddings");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-1", &[0.1, 0.2, 0.3, 0.4])
            .expect("write");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        let existing = fnv1a_hash(b"doc-1");
        let missing = fnv1a_hash(b"missing");
        let embeddings = index.get_embeddings(&[existing, missing]);
        assert!(embeddings[0].is_some());
        assert!(embeddings[1].is_none());
        assert_eq!(embeddings[0].as_ref().expect("existing").len(), 4);
    }

    #[test]
    fn soft_delete_marks_record_and_hides_hash_lookup() {
        let path = temp_index_path("soft-delete-main");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write doc-b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.soft_delete("doc-a").expect("soft delete"));
        assert!(!index.soft_delete("doc-a").expect("idempotent soft delete"));

        let hash_a = fnv1a_hash(b"doc-a");
        let hash_b = fnv1a_hash(b"doc-b");
        assert_eq!(index.find_index_by_doc_hash(hash_a), None);
        assert!(index.find_index_by_doc_hash(hash_b).is_some());
        assert_eq!(index.tombstone_count(), 1);

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_missing_returns_false() {
        let path = temp_index_path("soft-delete-missing");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(
            !index
                .soft_delete("missing-doc")
                .expect("missing soft delete")
        );
        assert_eq!(index.tombstone_count(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_batch_counts_only_new_tombstones() {
        let path = temp_index_path("soft-delete-batch");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer
            .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
            .expect("write c");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let deleted = index
            .soft_delete_batch(&["doc-a", "doc-b", "missing", "doc-a"])
            .expect("batch delete");
        assert_eq!(deleted, 2);
        assert_eq!(index.tombstone_count(), 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tombstone_ratio_and_needs_vacuum_threshold() {
        let path = temp_index_path("soft-delete-ratio");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        index.soft_delete("doc-0").expect("delete 0");
        index.soft_delete("doc-1").expect("delete 1");
        assert_eq!(index.tombstone_count(), 2);
        assert!((index.tombstone_ratio() - 0.2).abs() < f64::EPSILON);
        assert!(!index.needs_vacuum(), "threshold is strict greater-than");

        index.soft_delete("doc-2").expect("delete 2");
        assert_eq!(index.tombstone_count(), 3);
        assert!(index.needs_vacuum());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vacuum_removes_tombstones_and_preserves_live_results() {
        let path = temp_index_path("soft-delete-vacuum");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer
            .write_record("doc-c", &[0.0, 0.0, 1.0, 0.0])
            .expect("write c");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index.soft_delete("doc-b").expect("delete b");

        let pre_hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("pre-vacuum search");
        assert_eq!(pre_hits.len(), 2);
        assert!(pre_hits.iter().all(|hit| hit.doc_id != "doc-b"));

        let stats = index.vacuum().expect("vacuum");
        assert_eq!(stats.records_before, 3);
        assert_eq!(stats.records_after, 2);
        assert_eq!(stats.tombstones_removed, 1);
        assert!(stats.bytes_reclaimed > 0);
        assert!(stats.duration >= Duration::ZERO);

        assert_eq!(index.record_count(), 2);
        assert_eq!(index.tombstone_count(), 0);
        assert_eq!(index.find_index_by_doc_hash(fnv1a_hash(b"doc-b")), None);

        let post_hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("post-vacuum search");
        assert_eq!(post_hits.len(), 2);
        assert!(post_hits.iter().all(|hit| hit.doc_id != "doc-b"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_and_search_interleaving_has_no_corruption() {
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};

        let path = temp_index_path("soft-delete-concurrent");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        for i in 0..128 {
            writer
                .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write");
        }
        writer.finish().expect("finish");

        let shared = Arc::new(Mutex::new(VectorIndex::open(&path).expect("open")));
        let deleter = {
            let index = Arc::clone(&shared);
            std::thread::spawn(move || {
                for i in 0..32 {
                    let mut guard = index.lock().expect("lock for delete");
                    let doc_id = format!("doc-{i:03}");
                    let _ = guard.soft_delete(&doc_id).expect("soft delete");
                }
            })
        };

        let query = [1.0, 0.0, 0.0, 0.0];
        let searchers: Vec<_> = (0..4)
            .map(|_| {
                let index = Arc::clone(&shared);
                std::thread::spawn(move || {
                    for _ in 0..32 {
                        let hits = index
                            .lock()
                            .expect("lock for search")
                            .search_top_k(&query, 10, None)
                            .expect("search");
                        assert!(!hits.is_empty());
                    }
                })
            })
            .collect();

        deleter.join().expect("join deleter");
        for handle in searchers {
            handle.join().expect("join searcher");
        }

        let hits = shared
            .lock()
            .expect("lock final")
            .search_top_k(&query, 64, None)
            .expect("final search");
        let deleted_ids: HashSet<String> = (0..32).map(|i| format!("doc-{i:03}")).collect();
        assert!(
            hits.iter()
                .all(|hit| !deleted_ids.contains(hit.doc_id.as_str()))
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_preserves_existing_non_tombstone_flags() {
        let path = temp_index_path("soft-delete-flags");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let hash_a = fnv1a_hash(b"doc-a");
        let record_index = index
            .find_index_by_doc_hash(hash_a)
            .expect("record index for doc-a");

        let custom_flag: u16 = 0x0004;
        index
            .set_record_flags(record_index, custom_flag)
            .expect("seed custom flag");
        assert_eq!(
            index.record_at(record_index).expect("read flags").flags,
            custom_flag
        );

        assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
        let flags_after = index.record_at(record_index).expect("read flags").flags;
        assert_eq!(
            flags_after & RECORD_FLAG_TOMBSTONE,
            RECORD_FLAG_TOMBSTONE,
            "tombstone bit must be set",
        );
        assert_eq!(
            flags_after & custom_flag,
            custom_flag,
            "non-tombstone bits must remain untouched",
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tombstone_flag_persists_after_reopen() {
        let path = temp_index_path("soft-delete-persist");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        {
            let mut index = VectorIndex::open(&path).expect("open for delete");
            assert!(index.soft_delete("doc-a").expect("delete doc-a"));
            assert_eq!(index.tombstone_count(), 1);
        }

        let reopened = VectorIndex::open(&path).expect("reopen");
        assert_eq!(reopened.tombstone_count(), 1);
        assert_eq!(reopened.find_index_by_doc_hash(fnv1a_hash(b"doc-a")), None);
        let hits = reopened
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search after reopen");
        assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delete_vacuum_append_cycle_keeps_expected_live_set() {
        use std::collections::HashSet;

        let path = temp_index_path("soft-delete-reindex-cycle");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        for i in 0..100 {
            writer
                .write_record(&format!("doc-{i:03}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write initial doc");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let delete_ids: Vec<String> = (0..50).map(|i| format!("doc-{i:03}")).collect();
        let delete_refs: Vec<&str> = delete_ids.iter().map(String::as_str).collect();
        let deleted = index.soft_delete_batch(&delete_refs).expect("batch delete");
        assert_eq!(deleted, 50);
        assert_eq!(index.tombstone_count(), 50);

        let vacuum_stats = index.vacuum().expect("vacuum");
        assert_eq!(vacuum_stats.records_before, 100);
        assert_eq!(vacuum_stats.records_after, 50);
        assert_eq!(index.tombstone_count(), 0);
        assert_eq!(index.record_count(), 50);

        let append_entries: Vec<(String, Vec<f32>)> = (100..150)
            .map(|i| (format!("doc-{i:03}"), vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        index.append_batch(&append_entries).expect("append batch");
        assert_eq!(index.wal_record_count(), 50);

        let compact_stats = index.compact().expect("compact");
        assert_eq!(compact_stats.total_records_after, 100);
        assert_eq!(index.record_count(), 100);
        assert_eq!(index.wal_record_count(), 0);

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 150, None)
            .expect("search");
        assert_eq!(hits.len(), 100);
        let ids: HashSet<String> = hits.iter().map(|hit| hit.doc_id.to_string()).collect();

        for i in 0..50 {
            assert!(
                !ids.contains(&format!("doc-{i:03}")),
                "deleted id must not be present",
            );
        }
        for i in 50..150 {
            assert!(
                ids.contains(&format!("doc-{i:03}")),
                "live id must be present",
            );
        }

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn tombstones_remain_excluded_with_wal_and_after_compaction() {
        let path = temp_index_path("soft-delete-wal-integration");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "fnv1a-384", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[1.0, 0.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert!(index.soft_delete("doc-a").expect("delete a"));
        index
            .append("doc-c", &[1.0, 0.0, 0.0, 0.0])
            .expect("append c");
        assert_eq!(index.wal_record_count(), 1);

        let pre_compact = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("pre-compact search");
        assert_eq!(pre_compact.len(), 2);
        assert!(pre_compact.iter().all(|hit| hit.doc_id != "doc-a"));
        assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-b"));
        assert!(pre_compact.iter().any(|hit| hit.doc_id == "doc-c"));

        index.compact().expect("compact");
        let post_compact = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("post-compact search");
        assert_eq!(post_compact.len(), 2);
        assert!(post_compact.iter().all(|hit| hit.doc_id != "doc-a"));
        assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-b"));
        assert!(post_compact.iter().any(|hit| hit.doc_id == "doc-c"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn vacuum_noop_when_no_tombstones() {
        let path = temp_index_path("soft-delete-vacuum-noop");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write a");
        writer
            .write_record("doc-b", &[0.0, 1.0, 0.0, 0.0])
            .expect("write b");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.tombstone_count(), 0);

        let stats = index.vacuum().expect("vacuum with no tombstones");
        assert_eq!(stats.records_before, 2);
        assert_eq!(stats.records_after, 2);
        assert_eq!(stats.tombstones_removed, 0);
        assert_eq!(index.record_count(), 2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn soft_delete_all_records_yields_empty_search() {
        let path = temp_index_path("soft-delete-all");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..5 {
            writer
                .write_record(&format!("doc-{i}"), &sample_vector(0.1, 4))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..5 {
            assert!(index.soft_delete(&format!("doc-{i}")).expect("delete"));
        }
        assert_eq!(index.tombstone_count(), 5);
        assert!((index.tombstone_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(index.needs_vacuum());

        let hits = index
            .search_top_k(&sample_vector(0.1, 4), 10, None)
            .expect("search");
        assert!(
            hits.is_empty(),
            "search with all deleted should return nothing"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn vacuum_after_deleting_all_records_yields_empty_index() {
        let path = temp_index_path("soft-delete-vacuum-all");
        let mut writer = VectorIndex::create(&path, "fnv1a-384", 4).expect("writer");
        for i in 0..3 {
            writer
                .write_record(&format!("doc-{i}"), &[1.0, 0.0, 0.0, 0.0])
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..3 {
            index.soft_delete(&format!("doc-{i}")).expect("delete");
        }

        let stats = index.vacuum().expect("vacuum all deleted");
        assert_eq!(stats.records_before, 3);
        assert_eq!(stats.records_after, 0);
        assert_eq!(stats.tombstones_removed, 3);
        assert_eq!(index.record_count(), 0);
        assert_eq!(index.tombstone_count(), 0);
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(hits.is_empty());

        std::fs::remove_file(&path).ok();
    }

    // ─── WAL integration tests ─────────────────────────────────────────

    /// Fleet-review critical regression pin: an update whose WAL append
    /// FAILS must leave the old vector fully alive. The pre-fix ordering
    /// ran the durably-destructive soft-delete before logging the
    /// replacement, so a failed/interrupted WAL write destroyed the old
    /// vector with the new one never written anywhere.
    #[cfg(unix)]
    #[test]
    fn failed_wal_append_leaves_the_old_vector_alive() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_index_path("wal-append-atomic-update");
        let dim = 4;
        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("doc-x", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");

        // Make WAL-sidecar creation impossible: the parent directory is
        // read-only, so append_wal_batch's create fails with EACCES.
        let dir = path.parent().expect("fixture dir").to_path_buf();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).expect("seal fixture dir");
        let result = index.append("doc-x", &[0.0, 1.0, 0.0, 0.0]);
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("restore fixture dir");
        if result.is_ok() {
            // Root ignores directory permission bits; the scenario cannot
            // be forced, so the ordering property is unobservable here.
            eprintln!("skipping: running as a principal that bypasses directory permissions");
            return;
        }

        // The failed update must not have destroyed the old vector — in
        // the live handle NOR in the durable state a fresh open sees.
        for (label, snapshot) in [
            ("live handle", &mut index),
            ("fresh open", &mut VectorIndex::open(&path).expect("reopen")),
        ] {
            let hits = snapshot
                .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
                .expect("search");
            assert_eq!(
                hits.len(),
                1,
                "{label}: old vector must survive a failed update"
            );
            assert_eq!(hits[0].doc_id, "doc-x", "{label}");
            assert!(
                !snapshot.is_deleted(0),
                "{label}: the main-slab record must not be tombstoned by a failed update"
            );
        }
    }

    #[test]
    fn append_single_vector_is_searchable() {
        let path = temp_index_path("wal-append-single");
        let dim = 4;

        // Build initial index.
        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // Append via WAL.
        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.wal_record_count(), 0);
        index
            .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
            .expect("append");
        assert_eq!(index.wal_record_count(), 1);
        assert_eq!(
            index.live_doc_ids().expect("live document ids"),
            std::collections::HashSet::from(["main-0".to_owned(), "wal-0".to_owned()])
        );

        // Search should find both main and WAL entries.
        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-0", "WAL entry should rank first");

        // Cleanup.
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn replace_with_empty_cannot_resurrect_previous_wal_entries() {
        let path = temp_index_path("wal-replace-empty");
        let mut writer = VectorIndex::create(&path, "old", 4).expect("writer");
        writer
            .write_record("main-old", &[1.0, 0.0, 0.0, 0.0])
            .expect("write old main");
        writer.finish().expect("finish old main");

        let mut old = VectorIndex::open(&path).expect("open old generation");
        old.append("wal-old", &[0.0, 1.0, 0.0, 0.0])
            .expect("append old WAL");
        assert!(wal::wal_path_for(&path).exists());
        drop(old);

        assert!(
            VectorIndex::replace_with_empty(&path, "invalid", 0).is_err(),
            "replacement validation must fail before touching the old generation"
        );
        let intact = VectorIndex::open(&path).expect("reopen generation after rejected replace");
        assert_eq!(intact.record_count(), 1);
        assert_eq!(intact.wal_record_count(), 1);
        drop(intact);

        let replacement =
            VectorIndex::replace_with_empty(&path, "new", 4).expect("replace generation");
        assert_eq!(replacement.embedder_id(), "new");
        assert_eq!(replacement.record_count(), 0);
        assert_eq!(replacement.wal_record_count(), 0);
        assert!(!wal::wal_path_for(&path).exists());
        drop(replacement);

        let reopened = VectorIndex::open(&path).expect("reopen replacement");
        let hits = reopened
            .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search replacement");
        assert!(hits.is_empty(), "old WAL entries must not be resurrected");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_batch_all_searchable() {
        let path = temp_index_path("wal-append-batch");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append_batch(&[
                ("wal-0".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
                ("wal-1".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
                ("wal-2".to_owned(), vec![0.0, 0.0, 0.0, 1.0]),
            ])
            .expect("append batch");
        assert_eq!(index.wal_record_count(), 3);

        let hits = index
            .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 4, "all 4 vectors should be returned");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn compaction_merges_wal_into_main() {
        let path = temp_index_path("wal-compact");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
            .expect("append");
        index
            .append("wal-1", &[0.0, 0.0, 1.0, 0.0])
            .expect("append");

        assert_eq!(index.record_count(), 1);
        assert_eq!(index.wal_record_count(), 2);

        let stats = index.compact().expect("compact");
        assert_eq!(stats.main_records_before, 1);
        assert_eq!(stats.wal_records, 2);
        assert_eq!(stats.total_records_after, 3);
        assert_eq!(index.record_count(), 3);
        assert_eq!(index.wal_record_count(), 0);
        assert!(!wal::wal_path_for(&path).exists(), "WAL should be deleted");

        // All records should still be searchable from main index.
        let hits = index
            .search_top_k(&[1.0, 1.0, 1.0, 1.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 3);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn needs_compaction_threshold() {
        let path = temp_index_path("wal-threshold");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
                .expect("write");
        }
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index.set_wal_config(WalConfig {
            compaction_threshold: 5,
            compaction_ratio: 0.10,
            fsync_on_write: false,
        });

        assert!(!index.needs_compaction());

        // Add 1 entry: ratio = 1/10 = 0.10, hits the ratio threshold.
        index
            .append("wal-0", &sample_vector(0.2, dim))
            .expect("append");
        assert!(index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn wal_survives_reopen() {
        let path = temp_index_path("wal-reopen");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // Append and drop.
        {
            let mut index = VectorIndex::open(&path).expect("open");
            index
                .append("wal-0", &[0.0, 1.0, 0.0, 0.0])
                .expect("append");
        }

        // Reopen — WAL should be loaded automatically.
        let index = VectorIndex::open(&path).expect("reopen");
        assert_eq!(index.wal_record_count(), 1);

        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-0");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_dimension_mismatch_rejected() {
        let path = temp_index_path("wal-dim-mismatch");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let err = index
            .append("bad", &[1.0, 2.0])
            .expect_err("should reject wrong dimension");
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
        assert_eq!(
            index.wal_record_count(),
            0,
            "failed append should not persist"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compact_empty_wal_is_noop() {
        let path = temp_index_path("wal-compact-empty");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let stats = index.compact().expect("compact empty WAL");
        assert_eq!(stats.wal_records, 0);
        assert_eq!(stats.total_records_after, 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wal_entries_rank_correctly_against_main() {
        let path = temp_index_path("wal-ranking");
        let dim = 4;

        // Main index has a mediocre match.
        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-mediocre", &[0.5, 0.5, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        // WAL has a perfect match.
        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-perfect", &[1.0, 0.0, 0.0, 0.0])
            .expect("append");

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-perfect");
        assert!(hits[0].score > hits[1].score);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_duplicate_doc_id_both_searchable() {
        let path = temp_index_path("wal-dup-docid");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        // Append a second entry with the same doc_id but different vector.
        index
            .append("doc-a", &[0.0, 0.0, 0.0, 1.0])
            .expect("append duplicate");
        assert_eq!(index.wal_record_count(), 1);

        // WAL entry shadows the main-index entry (WAL is newer).
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(
            hits.len(),
            1,
            "WAL shadows main — only WAL entry should appear"
        );
        assert_eq!(hits[0].doc_id, "doc-a");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn append_large_batch_100_vectors() {
        let path = temp_index_path("wal-large-batch");
        let dim = 8;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        let batch: Vec<(String, Vec<f32>)> = (0..100)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                // Offset keeps wal-000 above the zero-norm writer gate while
                // staying below main-0's base of 1.0 for the ranking assert.
                let base = (i as f32).mul_add(0.01, 0.005);
                (format!("wal-{i:03}"), sample_vector(base, dim))
            })
            .collect();
        index.append_batch(&batch).expect("large batch");
        assert_eq!(index.wal_record_count(), 100);

        let hits = index
            .search_top_k(&sample_vector(1.0, dim), 5, None)
            .expect("search");
        assert_eq!(hits.len(), 5);
        // The main-0 (base=1.0) should rank near the top with query [1.0, ...].
        assert!(hits.iter().any(|h| h.doc_id == "main-0"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn concurrent_append_and_search() {
        use std::sync::Arc;

        let path = temp_index_path("wal-concurrent");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        for i in 0..10 {
            writer
                .write_record(&format!("main-{i}"), &sample_vector(0.1, dim))
                .expect("write");
        }
        writer.finish().expect("finish");

        // Append sequentially (VectorIndex is not Send+Sync for shared mutation),
        // then search from multiple threads using a snapshot.
        let mut index = VectorIndex::open(&path).expect("open");
        for i in 0..20 {
            index
                .append(&format!("wal-{i}"), &sample_vector(0.5, dim))
                .expect("append");
        }

        let index = Arc::new(index);
        let query = sample_vector(1.0, dim);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let idx = Arc::clone(&index);
                let q = query.clone();
                std::thread::spawn(move || idx.search_top_k(&q, 10, None).expect("search"))
            })
            .collect();

        for handle in handles {
            let hits = handle.join().expect("thread join");
            assert_eq!(hits.len(), 10);
            // All scores should be positive (dot product of positive vectors).
            assert!(hits.iter().all(|h| h.score > 0.0));
        }

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn wal_record_count_across_append_compact_cycles() {
        let path = temp_index_path("wal-count-cycle");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.wal_record_count(), 0);
        assert_eq!(index.record_count(), 1);

        // Append 3 entries.
        index.append("w1", &sample_vector(0.1, dim)).expect("a1");
        index.append("w2", &sample_vector(0.2, dim)).expect("a2");
        index.append("w3", &sample_vector(0.3, dim)).expect("a3");
        assert_eq!(index.wal_record_count(), 3);
        assert_eq!(index.record_count(), 1);

        // Compact.
        index.compact().expect("compact");
        assert_eq!(index.wal_record_count(), 0);
        assert_eq!(index.record_count(), 4);

        // Append 2 more.
        index.append("w4", &sample_vector(0.4, dim)).expect("a4");
        index.append("w5", &sample_vector(0.5, dim)).expect("a5");
        assert_eq!(index.wal_record_count(), 2);
        assert_eq!(index.record_count(), 4);

        // Total searchable = 4 + 2 = 6.
        let hits = index
            .search_top_k(&sample_vector(1.0, dim), 100, None)
            .expect("search");
        assert_eq!(hits.len(), 6);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_removes_wal_only_record_and_persists() {
        let path = temp_index_path("wal-soft-delete-only");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .expect("write");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("wal-only", &[0.0, 1.0, 0.0, 0.0])
            .expect("append wal-only");
        assert_eq!(index.wal_record_count(), 1);

        assert!(index.soft_delete("wal-only").expect("soft delete wal-only"));
        assert_eq!(index.wal_record_count(), 0);
        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(hits.iter().all(|hit| hit.doc_id != "wal-only"));

        drop(index);
        let reopened = VectorIndex::open(&path).expect("reopen");
        assert_eq!(reopened.wal_record_count(), 0);
        let reopened_hits = reopened
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search after reopen");
        assert!(reopened_hits.iter().all(|hit| hit.doc_id != "wal-only"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_clears_pending_wal_updates_for_same_doc_id() {
        let path = temp_index_path("wal-soft-delete-main-and-wal");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("doc-a", &[0.0, 1.0, 0.0, 0.0])
            .expect("append doc-a update");
        index
            .append("doc-b", &[0.0, 0.0, 1.0, 0.0])
            .expect("append doc-b");
        assert_eq!(index.wal_record_count(), 2);

        assert!(index.soft_delete("doc-a").expect("soft delete doc-a"));
        assert_eq!(
            index.wal_record_count(),
            1,
            "doc-a WAL entries should be purged"
        );

        let hits = index
            .search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(
            hits.iter().all(|hit| hit.doc_id != "doc-a"),
            "doc-a should not be searchable from main or WAL"
        );
        assert!(hits.iter().any(|hit| hit.doc_id == "doc-b"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn empty_index_append_only() {
        let path = temp_index_path("wal-empty-append");
        let dim = 4;

        // Create an empty main index.
        let writer = VectorIndex::create(&path, "test", dim).expect("writer");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        assert_eq!(index.record_count(), 0);

        // Append to empty index via WAL.
        index
            .append("first", &[1.0, 0.0, 0.0, 0.0])
            .expect("append");
        assert_eq!(index.wal_record_count(), 1);

        // Should still be searchable.
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "first");

        // Compact from empty main + WAL.
        let stats = index.compact().expect("compact");
        assert_eq!(stats.main_records_before, 0);
        assert_eq!(stats.wal_records, 1);
        assert_eq!(stats.total_records_after, 1);
        assert_eq!(index.record_count(), 1);

        std::fs::remove_file(&path).ok();
    }

    // ─── Quantization edge cases ────────────────────────────────────────

    #[test]
    fn quantization_bytes_per_element() {
        assert_eq!(Quantization::F32.bytes_per_element(), 4);
        assert_eq!(Quantization::F16.bytes_per_element(), 2);
    }

    #[test]
    fn quantization_from_wire_valid() {
        let path = Path::new("test.fsvi");
        assert_eq!(Quantization::from_wire(0, path).unwrap(), Quantization::F32);
        assert_eq!(Quantization::from_wire(1, path).unwrap(), Quantization::F16);
    }

    #[test]
    fn quantization_from_wire_invalid() {
        let path = Path::new("test.fsvi");
        assert!(Quantization::from_wire(2, path).is_err());
        assert!(Quantization::from_wire(255, path).is_err());
    }

    // ─── align_up edge cases ────────────────────────────────────────────

    #[test]
    fn align_up_zero_alignment() {
        assert_eq!(align_up(42, 0).unwrap(), 42);
    }

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(128, 64).unwrap(), 128);
    }

    #[test]
    fn align_up_zero_value() {
        assert_eq!(align_up(0, 64).unwrap(), 0);
    }

    #[test]
    fn align_up_one_over() {
        assert_eq!(align_up(65, 64).unwrap(), 128);
    }

    // ─── fnv1a_hash edge cases ──────────────────────────────────────────

    #[test]
    fn fnv1a_hash_empty_input() {
        let hash = fnv1a_hash(b"");
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_hash_different_inputs_differ() {
        let h1 = fnv1a_hash(b"doc-a");
        let h2 = fnv1a_hash(b"doc-b");
        assert_ne!(h1, h2);
    }

    // ─── is_tombstoned_flags ────────────────────────────────────────────

    #[test]
    fn tombstone_flag_logic() {
        assert!(!is_tombstoned_flags(0x0000));
        assert!(is_tombstoned_flags(RECORD_FLAG_TOMBSTONE));
        assert!(is_tombstoned_flags(0x0003)); // tombstone + custom
        assert!(!is_tombstoned_flags(0x0002)); // only custom
    }

    // ─── validate_header_string ─────────────────────────────────────────

    #[test]
    fn validate_header_string_empty_embedder_id_rejected() {
        let result = validate_header_string("", "embedder_id");
        assert!(result.is_err());
    }

    #[test]
    fn validate_header_string_empty_embedder_revision_ok() {
        let result = validate_header_string("", "embedder_revision");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_header_string_normal_ok() {
        let result = validate_header_string("potion-128M", "embedder_id");
        assert!(result.is_ok());
    }

    // ─── VectorMetadata clone/eq ────────────────────────────────────────

    #[test]
    fn vector_metadata_clone_eq() {
        let meta = VectorMetadata {
            fsvi_version: FSVI_VERSION,
            embedder_id: "test".to_owned(),
            embedder_revision: "v1".to_owned(),
            dimension: 256,
            quantization: Quantization::F16,
            compaction_gen: 0,
            record_count: 100,
            vectors_offset: 1024,
            identity_v2: None,
        };
        let cloned = meta.clone();
        assert_eq!(meta, cloned);
    }

    // ─── VectorIndex::create validation ─────────────────────────────────

    #[test]
    fn create_zero_dimension_rejected() {
        let path = temp_index_path("zero-dim");
        let result = VectorIndex::create(&path, "test", 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn create_empty_embedder_id_rejected() {
        let path = temp_index_path("empty-embedder");
        let result = VectorIndex::create(&path, "", 4);
        assert!(result.is_err());
    }

    #[test]
    fn create_with_revision_empty_revision_ok() {
        let path = temp_index_path("empty-rev");
        let writer =
            VectorIndex::create_with_revision(&path, "test", "", 4, Quantization::F16).unwrap();
        writer.finish().unwrap();
        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.embedder_revision(), "");
        std::fs::remove_file(&path).ok();
    }

    // ─── VectorIndexWriter rejection cases ──────────────────────────────

    #[test]
    fn write_record_nan_embedding_rejected() {
        let path = temp_index_path("nan-embed");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        let result = writer.write_record("doc", &[1.0, f32::NAN, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("non-finite"),
            "expected non-finite error, got: {err}"
        );
    }

    #[test]
    fn write_record_inf_embedding_rejected() {
        let path = temp_index_path("inf-embed");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        let result = writer.write_record("doc", &[1.0, f32::INFINITY, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn write_record_zero_norm_embedding_rejected() {
        // bd-tqhc: an all-zero vector can never match any query, so writers
        // reject it instead of planting a permanently unusable record.
        let path = temp_index_path("zero-norm-embed");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        let result = writer.write_record("doc", &[0.0, 0.0, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("non-zero"),
            "expected zero-norm rejection, got: {err}"
        );
    }

    #[test]
    fn append_zero_norm_embedding_rejected() {
        let path = temp_index_path("zero-norm-append");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        writer.write_record("doc-live", &[1.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let result = index.append("doc-zero", &[0.0, 0.0, 0.0]);
        assert!(result.is_err());
        // The rejection must not have disturbed the live record.
        assert_eq!(index.record_count(), 1);
        assert_eq!(index.wal_record_count(), 0);
    }

    #[test]
    fn zero_signal_census_counts_are_consistent() {
        let path = temp_index_path("zero-signal-census");
        let mut writer = VectorIndex::create(&path, "test", 3).unwrap();
        writer.write_record("doc-a", &[1.0, 0.0, 0.0]).unwrap();
        writer.write_record("doc-b", &[0.0, 1.0, 0.0]).unwrap();
        writer.write_record("doc-c", &[0.0, 0.0, 1.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let healthy = index.zero_signal_state();
        assert_eq!(healthy.record_count, 3);
        assert_eq!(healthy.live_count, 3);
        assert_eq!(healthy.tombstone_count, 0);
        assert_eq!(healthy.wal_count, 0);
        assert_eq!(healthy.usable_vector_count, 3);
        assert_eq!(healthy.state_reason(), None);

        index.soft_delete("doc-b").unwrap();
        let after_delete = index.zero_signal_state();
        assert_eq!(after_delete.live_count, 2);
        assert_eq!(after_delete.tombstone_count, 1);
        assert_eq!(after_delete.usable_vector_count, 2);
        assert_eq!(index.live_count(), 2);
    }

    // ─── VectorIndex::open edge cases ───────────────────────────────────

    #[test]
    fn open_nonexistent_file_returns_index_not_found() {
        let path = temp_index_path("nonexistent-open");
        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::IndexNotFound { .. }
        ));
    }

    #[test]
    fn open_truncated_file_detected() {
        let path = temp_index_path("truncated-open");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc-0", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 4]).unwrap();

        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("truncated") || err.contains("too small") || err.contains("extends"),
            "expected truncation error, got: {err}"
        );

        std::fs::remove_file(&path).ok();
    }

    // ─── FSVI constants ─────────────────────────────────────────────────

    #[test]
    fn fsvi_magic_is_four_bytes() {
        assert_eq!(FSVI_MAGIC.len(), 4);
        assert_eq!(&FSVI_MAGIC, b"FSVI");
    }

    #[test]
    fn fsvi_version_is_one() {
        assert_eq!(FSVI_VERSION, 1);
    }

    #[test]
    fn record_size_is_sixteen() {
        assert_eq!(RECORD_SIZE_BYTES, 16);
    }

    // ─── vector_at_f16 on f16 index ─────────────────────────────────────

    #[test]
    fn vector_at_f16_roundtrip() {
        let path = temp_index_path("f16-at-roundtrip");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
        writer.write_record("doc", &[0.5, -0.5, 1.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let f16_vec = index.vector_at_f16(0).unwrap();
        assert_eq!(f16_vec.len(), 3);
        assert!((f16_vec[0].to_f32() - 0.5).abs() < 0.01);
        assert!((f16_vec[1].to_f32() - (-0.5)).abs() < 0.01);
        assert!((f16_vec[2].to_f32() - 1.0).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    // ─── vector_at_f16 on f32 index (converts) ─────────────────────────

    #[test]
    fn vector_at_f16_from_f32_index() {
        let path = temp_index_path("f16-from-f32");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F32).unwrap();
        writer.write_record("doc", &[0.25, -0.75, 1.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let f16_vec = index.vector_at_f16(0).unwrap();
        assert_eq!(f16_vec.len(), 3);
        assert!((f16_vec[0].to_f32() - 0.25).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    // ─── metadata accessor ──────────────────────────────────────────────

    #[test]
    fn metadata_accessor_returns_consistent_data() {
        let path = temp_index_path("metadata-accessor");
        let mut writer =
            VectorIndex::create_with_revision(&path, "emb-1", "rev-9", 16, Quantization::F32)
                .unwrap();
        writer.write_record("d", &[0.5; 16]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let meta = index.metadata();
        assert_eq!(meta.embedder_id, "emb-1");
        assert_eq!(meta.embedder_revision, "rev-9");
        assert_eq!(meta.dimension, 16);
        assert_eq!(meta.quantization, Quantization::F32);
        assert_eq!(meta.record_count, 1);
        assert_eq!(meta.vectors_offset % 64, 0);

        std::fs::remove_file(&path).ok();
    }

    // ─── is_deleted accessor ────────────────────────────────────────────

    #[test]
    fn is_deleted_false_for_live_record() {
        let path = temp_index_path("is-deleted-live");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(!index.is_deleted(0));

        std::fs::remove_file(&path).ok();
    }

    // ─── tombstone_ratio empty index ────────────────────────────────────

    #[test]
    fn tombstone_ratio_empty_index_is_zero() {
        let path = temp_index_path("tomb-ratio-empty");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(index.tombstone_ratio().abs() < f64::EPSILON);
        assert!(!index.needs_vacuum());

        std::fs::remove_file(&path).ok();
    }

    // ─── WalConfig default ──────────────────────────────────────────────

    #[test]
    fn wal_config_default_values() {
        let cfg = WalConfig::default();
        assert!(cfg.compaction_threshold > 0);
        assert!(cfg.compaction_ratio > 0.0);
    }

    // ─── F32 roundtrip with explicit revision ───────────────────────────

    #[test]
    fn f32_roundtrip_with_revision() {
        let path = temp_index_path("f32-rev-roundtrip");
        let original = vec![std::f32::consts::PI, std::f32::consts::E, 0.0, -1.0];
        let mut writer =
            VectorIndex::create_with_revision(&path, "f32-emb", "rev-42", 4, Quantization::F32)
                .unwrap();
        writer.write_record("doc", &original).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let recovered = index.vector_at_f32(0).unwrap();
        assert_eq!(recovered, original, "f32 must roundtrip exactly");
        assert_eq!(index.embedder_revision(), "rev-42");

        std::fs::remove_file(&path).ok();
    }

    // ─── Header CRC corruption by flipping data byte ────────────────────

    #[test]
    fn header_crc_detects_embedder_id_corruption() {
        let path = temp_index_path("crc-embedder-corrupt");
        let mut writer = VectorIndex::create(&path, "test-embedder-long", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut data = std::fs::read(&path).unwrap();
        // Flip a byte in the embedder_id region (after magic+version+id_len = 8 bytes)
        data[10] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        let result = VectorIndex::open(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("CRC") || err.contains("crc"),
            "expected CRC error, got: {err}"
        );

        std::fs::remove_file(&path).ok();
    }

    // ─── bd-1fh4 tests begin ──────────────────────────────────────────

    #[test]
    fn vacuum_stats_debug_clone_partial_eq() {
        let stats = VacuumStats {
            records_before: 10,
            records_after: 8,
            tombstones_removed: 2,
            bytes_reclaimed: 1024,
            duration: Duration::from_millis(5),
        };
        let debug = format!("{stats:?}");
        assert!(debug.contains("VacuumStats"));
        assert!(debug.contains("records_before: 10"));

        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn quantization_debug_clone_copy_eq() {
        let f16 = Quantization::F16;
        let f32q = Quantization::F32;

        let debug_f16 = format!("{f16:?}");
        assert!(debug_f16.contains("F16"));
        let debug_f32 = format!("{f32q:?}");
        assert!(debug_f32.contains("F32"));

        let f16_copy = f16;
        assert_eq!(f16, f16_copy);
        let f32_copy = f32q;
        assert_eq!(f32q, f32_copy);
        assert_ne!(f16, f32q);
    }

    #[test]
    fn vector_index_debug_includes_path() {
        let path = temp_index_path("debug-fmt");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let debug = format!("{index:?}");
        assert!(debug.contains("VectorIndex"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn set_wal_config_overrides_defaults() {
        let path = temp_index_path("wal-cfg-override");
        let dim = 4;
        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        for i in 0..100 {
            writer
                .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
                .unwrap();
        }
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        // With 100 main records and default config, 1 WAL entry should not trigger.
        index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
        assert!(!index.needs_compaction());

        // Set a low threshold to trigger compaction.
        index.set_wal_config(WalConfig {
            compaction_threshold: 1,
            compaction_ratio: 0.001,
            fsync_on_write: false,
        });
        assert!(index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn find_index_by_doc_hash_empty_index_none() {
        let path = temp_index_path("hash-empty");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert!(index.find_index_by_doc_hash(0xDEAD_BEEF).is_none());
        assert!(index.find_index_by_doc_hash(0).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn get_embeddings_mixed_hit_miss() {
        let path = temp_index_path("emb-mixed");
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 3, Quantization::F16).unwrap();
        writer.write_record("alpha", &[1.0, 0.0, 0.0]).unwrap();
        writer.write_record("beta", &[0.0, 1.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let alpha_hash = fnv1a_hash(b"alpha");
        let beta_hash = fnv1a_hash(b"beta");
        let missing_hash = fnv1a_hash(b"gamma");

        let results = index.get_embeddings(&[alpha_hash, missing_hash, beta_hash]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some(), "alpha should be found");
        assert!(results[1].is_none(), "gamma should be missing");
        assert!(results[2].is_some(), "beta should be found");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_batch_empty_is_noop() {
        let path = temp_index_path("append-empty-batch");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        index.append_batch(&[]).unwrap();
        assert_eq!(index.wal_record_count(), 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn append_nan_embedding_rejected() {
        let path = temp_index_path("append-nan");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let result = index.append("doc", &[1.0, f32::NAN, 0.0, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("finite"), "expected finite error, got: {err}");
    }

    #[test]
    fn append_inf_embedding_rejected() {
        let path = temp_index_path("append-inf");
        let writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let result = index.append("doc", &[1.0, 0.0, f32::INFINITY, 0.0]);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("finite"), "expected finite error, got: {err}");
    }

    #[test]
    fn soft_delete_already_deleted_returns_false() {
        let path = temp_index_path("double-delete");
        let mut writer = VectorIndex::create(&path, "test", 4).unwrap();
        writer.write_record("doc", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        assert!(index.soft_delete("doc").unwrap(), "first delete");
        assert!(!index.soft_delete("doc").unwrap(), "second delete");
        assert!(!index.soft_delete("doc").unwrap(), "third delete");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compact_preserves_wal_config() {
        let path = temp_index_path("compact-cfg");
        let dim = 4;
        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        for i in 0..20 {
            writer
                .write_record(&format!("d{i}"), &sample_vector(0.1, dim))
                .unwrap();
        }
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        let custom = WalConfig {
            compaction_threshold: 99,
            compaction_ratio: 0.90,
            fsync_on_write: false,
        };
        index.set_wal_config(custom);
        index.append("wal-1", &sample_vector(0.5, dim)).unwrap();
        index.compact().unwrap();

        // After compaction, the custom config should be preserved.
        assert_eq!(index.wal_record_count(), 0);
        // Verify config persists: threshold=99 and ratio=0.90,
        // with 21 main records, 1 WAL entry → ratio ~0.048 < 0.90.
        index.append("wal-2", &sample_vector(0.3, dim)).unwrap();
        assert!(!index.needs_compaction());

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn soft_delete_wal_restores_state_on_rewrite_failure() {
        let path = temp_index_path("wal-delete-restore");
        let dim = 4;

        let mut writer = VectorIndex::create(&path, "test", dim).unwrap();
        writer
            .write_record("main-0", &sample_vector(1.0, dim))
            .unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        index.append("wal-a", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.append("wal-b", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(index.wal_record_count(), 2);

        // Make the WAL parent directory read-only to force a rewrite failure.
        let wal_file = wal::wal_path_for(&path);
        let wal_dir = wal_file.parent().unwrap();
        let original_perms = fs::metadata(wal_dir).unwrap().permissions();
        let mut readonly = original_perms.clone();
        readonly.set_readonly(true);
        if fs::set_permissions(wal_dir, readonly).is_err() {
            // Sandboxed environments may not allow permission changes; skip.
            std::fs::remove_file(&path).ok();
            std::fs::remove_file(wal::wal_path_for(&path)).ok();
            return;
        }

        let result = index.soft_delete("wal-a");

        // Restore directory permissions before any assertions so cleanup works.
        fs::set_permissions(wal_dir, original_perms).unwrap();

        // Precondition check: a read-only directory only blocks writes for an
        // unprivileged process. On root / `CAP_DAC_OVERRIDE` environments (some CI
        // and rch remote workers run as root), the rewrite is NOT blocked and
        // `soft_delete` succeeds — the forced-failure scenario this test exercises
        // cannot occur there. Skip gracefully instead of asserting a failure that
        // is physically impossible on that worker (the source of intermittent,
        // worker-dependent flakiness in this test).
        if result.is_ok() {
            std::fs::remove_file(&path).ok();
            std::fs::remove_file(wal::wal_path_for(&path)).ok();
            return;
        }

        // The delete should have failed.
        assert!(result.is_err(), "expected error from read-only directory");

        // In-memory WAL entries must be fully restored.
        assert_eq!(
            index.wal_record_count(),
            2,
            "WAL entries should be restored after rewrite failure"
        );

        // Both entries should still be searchable.
        let hits = index.search_top_k(&[0.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(hits.iter().any(|h| h.doc_id == "wal-a"));
        assert!(hits.iter().any(|h| h.doc_id == "wal-b"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(wal::wal_path_for(&path)).ok();
    }

    // ─── Regression: Duplicate entries on compaction crash ──────────────

    #[test]
    fn repro_duplicate_entries_on_compaction_crash() {
        let path = temp_index_path("compaction-crash");
        let dim = 4;

        // 1. Create initial index with 1 document
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16).unwrap();
        writer.write_record("doc-A", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();

        // 2. Append a document to WAL
        index.append("doc-B", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Check state before "compaction"
        let hits = index.search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);

        // 3. Simulate compaction crash:
        // We want to create a state where "doc-B" is in Main Index AND in WAL.
        // We can do this by running `compact` but preventing the WAL deletion.
        // Since we can't easily interrupt `compact`, we'll simulate the filesystem state.

        // Close index to flush everything
        drop(index);

        // Manually create the "post-compaction" main index that includes both A and B.
        let mut compact_writer =
            VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16)
                .unwrap()
                .with_generation(2); // Simulate correct compaction increment
        compact_writer
            .write_record("doc-A", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        compact_writer
            .write_record("doc-B", &[0.0, 1.0, 0.0, 0.0])
            .unwrap();
        compact_writer.finish().unwrap(); // Overwrites `path` with new index containing A and B.

        // Restore the WAL file (because `finish` doesn't touch it, but we need to ensure it exists and has doc-B)
        // Actually, `finish` overwrites `path`. The WAL file is at `path.wal`.
        // We didn't delete `path.wal`. So `path.wal` still contains "doc-B".

        // 4. Re-open index. It should load Main (A, B) and WAL (B).
        let index_reopened = VectorIndex::open(&path).unwrap();

        // 5. Search. If bug exists, we'll see "doc-B" twice.
        let hits = index_reopened
            .search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None)
            .unwrap();

        // Debug output
        for hit in &hits {
            println!("Hit: {} score={}", hit.doc_id, hit.score);
        }

        // Clean up
        let _ = fs::remove_file(&path);
        let _ = wal::remove_wal(&wal::wal_path_for(&path));

        // Assert failure
        let hit_count = hits.len();
        assert_eq!(
            hit_count, 2,
            "Should have exactly 2 hits (A and B), found {hit_count}"
        );
        let b_count = hits.iter().filter(|h| h.doc_id == "doc-B").count();
        assert_eq!(b_count, 1, "Should have exactly 1 'doc-B', found {b_count}");
    }

    // ─── bd-1fh4 tests end ────────────────────────────────────────────
}
