//! Keeper lifecycle and durability.
//!
//! This module owns the hand-rolled MANIFEST v1 wire format, Q1 range
//! validation, two-slot recovery, cross-process writer ownership, staged
//! durability repair, serialized publication, and writer-admitted garbage
//! collection. It also owns the structural Q1 concat-merge primitive, the
//! bound-consecutive tier planner, and tombstone-folding compaction.

use std::cmp::Ordering;
#[cfg(unix)]
use std::collections::HashSet;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use arc_swap::ArcSwap;
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::{LockError, Mutex, OwnedMutexGuard};
use frankensearch_core::{DocId, SearchError};
#[cfg(feature = "durability")]
use frankensearch_durability::{FileProtector, FileRecoveryOutcome, FileSourceWitness};
use frankensearch_index::mapped_file::ReadOnlyMappedFile;
use thiserror::Error;

use crate::error::QuillError;
use crate::grimoire::{
    ByteSpan, EncodedTermDictionary, OwnedTerm, TermDictionary, TermInput, TermMetadata,
    TermSectionLengths,
};
use crate::quiver::{
    BlockMaxConcatList, DocLenFieldInput, DocLenSection, EncodedBlockMax, EncodedDocLenSection,
    EncodedIdHashSection, EncodedIdMapSection, EncodedNumericSection, EncodedPositionList,
    EncodedPostingList, EncodedStatsSection, EncodedStoredMetaSection, FieldStats,
    IdHashCodecError, IdHashLookupPlan, IdHashSection, IdMapCodecError, IdMapEntry,
    IdMapEntryInput, IdMapSection, NumericEntry, NumericFieldInput, NumericSection, PositionList,
    Posting, PostingList, StatsSection, StoredMetaFieldInput, StoredMetaSection,
    ValidatedTermPruningMetadata, aggregate_field_stats,
};
use crate::schema::{FieldKind, SchemaDescriptor};
use crate::segment::{
    EncodedSegment, PendingSegmentFile, PlannedSection, SectionEntry, SectionKind,
    SegmentAssembler, SegmentHeader, SegmentHeaderInput, SegmentReader,
};

pub use crate::stats::{SegmentStats, SegmentStatsProvider};

/// Eight-byte MANIFEST magic, including its trailing NUL.
pub const MANIFEST_MAGIC: [u8; 8] = *b"FSLXMAN\0";
/// Current durable MANIFEST format version written by this crate.
///
/// v2 adds the generation-level `last_publish_unix_s` freshness witness
/// (registry §6.2). Readers accept v1 images and decode the field as `0`
/// (unknown); v1 readers reject v2 images as an unsupported version.
pub const MANIFEST_FORMAT_VERSION: u32 = 2;
/// First durable MANIFEST format version, still accepted on read.
pub const MANIFEST_FORMAT_VERSION_V1: u32 = 1;
/// Maximum MANIFEST accepted by an eager reader.
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
/// Maximum segment records admitted by one MANIFEST.
///
/// This is deliberately below the byte-format's theoretical limit so hostile
/// inputs cannot amplify a bounded file into millions of heap allocations.
pub const MAX_MANIFEST_SEGMENTS: usize = 65_536;
/// Maximum stats rows. `field_ord` is a `u16`, so no larger set is useful.
pub const MAX_MANIFEST_FIELDS: usize = 65_536;
/// Known MANIFEST flag: a bulk build is in progress.
pub const MANIFEST_FLAG_BULK_MODE_IN_PROGRESS: u32 = 1;
/// All flag bits understood by this reader.
pub const MANIFEST_KNOWN_FLAGS: u32 = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
/// Canonical empty roaring-lite tombstone set (`chunk_count = 0`).
pub const EMPTY_TOMBSTONES: [u8; 4] = [0, 0, 0, 0];
/// Minimum age of an unreachable Quill artifact before writer GC may remove it.
pub const DEFAULT_GARBAGE_GRACE: Duration = Duration::from_secs(300);
/// Eight-byte writer-admission magic, including its trailing NUL.
pub const WRITER_LOCK_MAGIC: [u8; 8] = *b"FSLXLCK\0";
/// Current durable writer-lock record version.
pub const WRITER_LOCK_FORMAT_VERSION: u32 = 1;
/// Exact byte length of a v1 writer-lock record, including its CRC32.
pub const WRITER_LOCK_RECORD_BYTES: usize = 36;

/// Current Quill crate version in the MANIFEST packed-semver representation.
///
/// The build-time assertion in this module's tests intentionally forces this
/// value to change when `Cargo.toml` changes.
pub const CURRENT_ENGINE_VERSION: u32 = pack_engine_version(0, 2, 1);

const MANIFEST_MIN_BYTES: usize = 8 + 4 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4;
/// v2 images carry the additional `last_publish_unix_s` word after `flags`.
const MANIFEST_V2_MIN_BYTES: usize = MANIFEST_MIN_BYTES + 8;
const SEGMENT_FIXED_BYTES: usize = 8 + 8 + 8 + 8 + 8 + 8 + 4;
const FIELD_STATS_BYTES: usize = 2 + 8 + 4;
const MAX_DOCID_EXCLUSIVE: u64 = 4_294_967_296;
const TOMBSTONE_ARRAY_MAX_CARDINALITY: u16 = 4_096;
const TOMBSTONE_BITMAP_MIN_CARDINALITY: u64 = 3_584;
const TOMBSTONE_BITMAP_BYTES: usize = 8_192;
const MAX_TOMBSTONE_CHUNKS: usize = 65_536;

// Construction initializes this once outside the async publication path. The
// asupersync mutex serializes every filesystem mutation within the process;
// every public mutation path additionally requires a retained cross-process
// `KeeperWriter` admission. Its owned guard is moved into blocking work so a
// cancelled async caller cannot release serialization before that work exits.
static PUBLISH_LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();

/// Pack semantic-version components into the durable `engine_version` word.
///
/// Layout: `major:u8 << 24 | minor:u8 << 16 | patch:u16`. Prerelease and
/// build metadata are intentionally not durable.
#[must_use]
pub const fn pack_engine_version(major: u8, minor: u8, patch: u16) -> u32 {
    (major as u32) << 24 | (minor as u32) << 16 | patch as u32
}

/// Decode a packed `engine_version` into `(major, minor, patch)`.
#[must_use]
pub const fn unpack_engine_version(version: u32) -> (u8, u8, u16) {
    (
        version.to_be_bytes()[0],
        version.to_be_bytes()[1],
        u16::from_be_bytes([version.to_be_bytes()[2], version.to_be_bytes()[3]]),
    )
}

/// Typed failures from MANIFEST byte validation.
#[derive(Debug, Error)]
pub enum ManifestCodecError {
    /// An in-memory manifest violates a writer invariant.
    #[error("invalid manifest: {detail}")]
    Invalid {
        /// Violated invariant.
        detail: String,
    },
    /// The byte stream ended before a field was complete.
    #[error("truncated manifest at byte {offset}: need {needed} bytes, only {remaining} remain")]
    Truncated {
        /// Byte offset where the read began.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
        /// Bytes still available.
        remaining: usize,
    },
    /// The magic did not identify a Quill MANIFEST.
    #[error("invalid manifest magic")]
    BadMagic,
    /// The reader does not implement the encoded version.
    #[error("unsupported manifest format version {found}")]
    UnsupportedVersion {
        /// Encoded version.
        found: u32,
    },
    /// The trailing CRC did not authenticate the preceding bytes.
    #[error("manifest CRC mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// CRC stored in the trailer.
        stored: u32,
        /// CRC recomputed by the reader.
        computed: u32,
    },
    /// A count or file length exceeds an eager-reader resource budget.
    #[error("manifest {resource} limit exceeded: {actual} > {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Observed value.
        actual: u64,
        /// Admitted maximum.
        limit: u64,
    },
    /// Structurally parseable bytes violate a canonical or Q1 invariant.
    #[error("non-canonical manifest: {detail}")]
    NonCanonical {
        /// Violated invariant.
        detail: String,
    },
    /// Valid fields were followed by bytes not owned by the format.
    #[error("manifest has {remaining} trailing bytes before its CRC")]
    TrailingBytes {
        /// Unconsumed byte count.
        remaining: usize,
    },
}

/// Policy for folding immutable-segment tombstones into positional holes.
///
/// A segment is eligible only when its tombstone density is strictly greater
/// than `tombstone_density`. Equality is deliberately a no-op, which prevents
/// repeated work at the configured boundary. Compaction always preserves the
/// surviving global document ids and never performs the reserved renumbering
/// deep-compaction operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionPolicy {
    /// Per-segment tombstone density threshold in `(0, 1]`.
    pub tombstone_density: f64,
}

impl CompactionPolicy {
    /// Construct a caller-selected density policy.
    #[must_use]
    pub const fn new(tombstone_density: f64) -> Self {
        Self { tombstone_density }
    }
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self::new(crate::config::DEFAULT_COMPACTION_TOMBSTONE_DENSITY)
    }
}

/// Observable work and byte accounting from one atomic compaction pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompactionReport {
    /// MANIFEST generation inspected by the pass.
    pub generation_before: u64,
    /// Published generation, equal to `generation_before` for a no-op.
    pub generation_after: u64,
    /// Immutable segments examined by the density policy.
    pub examined_segments: usize,
    /// Eligible segments rewritten or removed.
    pub compacted_segments: usize,
    /// Fully deleted segments removed without emitting an empty FSLX file.
    pub removed_segments: usize,
    /// Physical rows converted into positional holes.
    pub dropped_documents: u64,
    /// Sum of immutable source file lengths for eligible segments.
    pub input_bytes: u64,
    /// Sum of replacement FSLX file lengths.
    pub output_bytes: u64,
}

impl CompactionReport {
    /// Whether this pass published a successor generation.
    #[must_use]
    pub const fn changed(self) -> bool {
        self.compacted_segments != 0
    }
}

/// Typed failures from tombstone-folding segment compaction.
#[derive(Debug, Error)]
pub enum CompactionError {
    /// The threshold must match the configuration contract.
    #[error("compaction tombstone density {density:?} is outside (0, 1]")]
    InvalidDensity {
        /// Rejected threshold.
        density: f64,
    },
    /// A source carries an extension whose rewrite semantics are unknown.
    #[error("compaction source {segment_id:#018x} carries unsupported section kind {section_kind}")]
    UnknownSection {
        /// Source segment id.
        segment_id: u64,
        /// Raw durable section kind.
        section_kind: u16,
    },
    /// Checked rewrite arithmetic overflowed.
    #[error("compaction arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Failed counter or offset.
        field: &'static str,
    },
    /// A fallible allocation failed without panicking.
    #[error("unable to reserve {count} entries for compaction {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// One section codec rejected source or rebuilt bytes.
    #[error("compaction {section:?} codec failed: {detail}")]
    SectionCodec {
        /// Affected FSLX section.
        section: SectionKind,
        /// Typed codec's stable diagnosis.
        detail: String,
    },
    /// Existing concat-source validation failed before the rewrite.
    #[error("compaction source validation failed: {detail}")]
    SourceValidation {
        /// Stable source diagnosis.
        detail: String,
    },
    /// A collision-free immutable output identity could not be derived.
    #[error("compaction could not derive a collision-free segment id for {source_id:#018x}")]
    OutputIdExhausted {
        /// Source segment being replaced.
        source_id: u64,
    },
    /// The successor MANIFEST failed canonical validation.
    #[error("compaction successor manifest is invalid: {detail}")]
    InvalidManifest {
        /// Manifest diagnosis.
        detail: String,
    },
    /// The caller cancelled before the visibility boundary.
    #[error("compaction cancelled before MANIFEST publication")]
    Cancelled,
    /// Final FSLX framing or temp-file staging failed.
    #[error("compaction segment operation failed: {source}")]
    Segment {
        /// Segment-layer diagnosis.
        #[source]
        source: QuillError,
    },
}

/// Typed failures from Q1-preserving whole-segment concat merge.
#[derive(Debug, Error)]
pub enum ConcatMergeError {
    /// A merge must replace at least two immutable segments.
    #[error("concat merge requires at least two source segments, got {count}")]
    TooFewSources {
        /// Supplied source count.
        count: usize,
    },
    /// One caller-supplied source id is absent from the current snapshot.
    #[error("concat merge source {segment_id:#018x} at position {position} is not live")]
    SourceNotFound {
        /// Position in caller order.
        position: usize,
        /// Missing immutable segment id.
        segment_id: u64,
    },
    /// Caller order did not name one uninterrupted manifest slice.
    #[error(
        "concat merge source {actual:#018x} at position {position} is not manifest-consecutive; expected {expected:#018x}"
    )]
    NonConsecutiveSources {
        /// Position in caller order.
        position: usize,
        /// Required segment at this manifest position.
        expected: u64,
        /// Caller-supplied segment id.
        actual: u64,
    },
    /// A live id appears after the chosen manifest suffix has ended.
    #[error(
        "concat merge source {actual:#018x} at position {position} wraps past the manifest suffix"
    )]
    SourceRunPastManifest {
        /// Position in caller order.
        position: usize,
        /// Live source id that wrapped to an earlier manifest position.
        actual: u64,
    },
    /// The proposed immutable output id already belongs to a live segment.
    #[error("concat merge output segment id {segment_id:#018x} collides with a live segment")]
    OutputSegmentCollision {
        /// Colliding segment id.
        segment_id: u64,
    },
    /// A source carries an extension whose merge semantics are unknown.
    #[error(
        "concat merge source {segment_id:#018x} carries unsupported section kind {section_kind}"
    )]
    UnknownSection {
        /// Source segment id.
        segment_id: u64,
        /// Raw durable section kind.
        section_kind: u16,
    },
    /// Checked merge arithmetic overflowed.
    #[error("concat merge arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Failed counter or offset.
        field: &'static str,
    },
    /// A fallible allocation failed without panicking.
    #[error("unable to reserve {count} entries for concat merge {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// One section codec rejected source or rebuilt bytes.
    #[error("concat merge {section:?} codec failed: {detail}")]
    SectionCodec {
        /// Affected FSLX section.
        section: SectionKind,
        /// Typed codec's stable diagnosis.
        detail: String,
    },
    /// External-id state admitted more than one live physical row.
    #[error(
        "concat merge document id {document_id:?} has multiple live rows {first_global_docid}/{duplicate_global_docid}"
    )]
    MultipleLiveDocumentIds {
        /// Exact external identifier.
        document_id: String,
        /// First live global document id.
        first_global_docid: u32,
        /// Duplicate live global document id.
        duplicate_global_docid: u32,
    },
    /// The successor MANIFEST failed canonical validation.
    #[error("concat merge successor manifest is invalid: {detail}")]
    InvalidManifest {
        /// Manifest diagnosis.
        detail: String,
    },
    /// The caller cancelled before the visibility boundary.
    #[error("concat merge cancelled before MANIFEST publication")]
    Cancelled,
    /// Final FSLX framing or temp-file staging failed.
    #[error("concat merge segment operation failed: {source}")]
    Segment {
        /// Segment-layer diagnosis.
        #[source]
        source: QuillError,
    },
}

/// Keeper I/O, recovery, locking, and publication failures.
#[derive(Debug, Error)]
pub enum KeeperError {
    /// Neither MANIFEST slot exists.
    #[error("Quill index not found at {directory}")]
    IndexNotFound {
        /// Index directory that was inspected.
        directory: PathBuf,
    },
    /// One durable slot contains invalid bytes.
    #[error("corrupt manifest at {path}: {source}")]
    ManifestCorrupted {
        /// Invalid slot.
        path: PathBuf,
        /// Codec diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// Proposed in-memory state failed writer-side validation.
    #[error("invalid proposed manifest: {source}")]
    InvalidManifest {
        /// Writer invariant diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// A whole-segment concat merge failed before publication completed.
    #[error("keeper concat merge failed: {source}")]
    ConcatMerge {
        /// Merge planning, codec, or cancellation diagnosis.
        #[source]
        source: ConcatMergeError,
    },
    /// A tombstone-folding compaction failed before publication completed.
    #[error("keeper compaction failed: {source}")]
    Compaction {
        /// Policy, rewrite, codec, or cancellation diagnosis.
        #[source]
        source: CompactionError,
    },
    /// Neither existing slot is valid.
    #[error("no valid manifest in {directory}: current={current}; previous={previous}")]
    NoValidManifest {
        /// Index directory that was inspected.
        directory: PathBuf,
        /// Primary-slot diagnosis.
        current: String,
        /// Previous-slot diagnosis.
        previous: String,
    },
    /// Two valid slots do not form an admitted generation pair.
    #[error(
        "invalid manifest generation pair in {directory}: current={current}, previous={previous}"
    )]
    InvalidGenerationPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
    },
    /// Two individually valid slots disagree on durable snapshot identity.
    #[error(
        "invalid manifest pair in {directory}: current generation {current}, previous generation {previous}: {detail}"
    )]
    InvalidManifestPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
        /// Cross-slot invariant that failed.
        detail: String,
    },
    /// The proposed publication is not the next monotone generation.
    #[error("manifest generation conflict: expected {expected}, proposed {proposed}")]
    GenerationConflict {
        /// Required next generation.
        expected: u64,
        /// Proposed generation.
        proposed: u64,
    },
    /// Another process still owns the cross-process writer admission.
    #[error("Quill writer is busy at {path}; owner pid={owner_pid:?}")]
    WriterBusy {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Best-effort owner diagnosis from a complete lock record.
        owner_pid: Option<u32>,
    },
    /// The persistent writer-lock record is malformed and cannot be reclaimed.
    #[error("corrupt Quill writer lock at {path}: {detail}")]
    WriterLockCorrupted {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Failed wire, type, or liveness invariant.
        detail: String,
    },
    /// Cancellation was observed while acquiring or initializing a writer.
    #[error("Quill writer admission cancelled")]
    WriterAdmissionCancelled,
    /// An extant `O_EXCL` artifact already owns the proposed generation.
    #[error("Quill generation {generation} is already claimed at {path}")]
    GenerationClaimConflict {
        /// Existing canonical claim path.
        path: PathBuf,
        /// Claimed MANIFEST generation.
        generation: u64,
    },
    /// The current generation cannot be incremented.
    #[error("manifest generation space exhausted at {current}")]
    GenerationExhausted {
        /// Last representable generation.
        current: u64,
    },
    /// A proposed generation would roll back durable index identity or state.
    #[error("invalid manifest transition: {detail}")]
    InvalidTransition {
        /// Violated monotonicity invariant.
        detail: String,
    },
    /// The caller's expected schema does not match the durable MANIFEST.
    #[error("Quill schema mismatch at {path}: expected {expected:#018x}, found {found:#018x}")]
    SchemaMismatch {
        /// MANIFEST path carrying the durable schema identity.
        path: PathBuf,
        /// Hash of the schema requested by the caller.
        expected: u64,
        /// Hash recorded in the selected MANIFEST.
        found: u64,
    },
    /// The supplied schema descriptor itself is invalid.
    #[error("invalid Quill schema descriptor: {source}")]
    InvalidSchema {
        /// Descriptor validation failure.
        #[source]
        source: QuillError,
    },
    /// A MANIFEST-referenced segment could not be opened structurally.
    #[error("cannot open Quill segment at {path}: {source}")]
    SegmentOpen {
        /// Canonical published segment path.
        path: PathBuf,
        /// Segment reader diagnosis.
        #[source]
        source: QuillError,
    },
    /// Canonical segment bytes could not be staged for an idempotent install.
    #[error("cannot stage Quill segment for publication: {source}")]
    SegmentInstall {
        /// Segment-layer write or validation diagnosis.
        #[source]
        source: QuillError,
    },
    /// A segment disagrees with its immutable MANIFEST witnesses.
    #[error("Quill segment metadata mismatch at {path}: {detail}")]
    SegmentMetadataMismatch {
        /// Canonical published segment path.
        path: PathBuf,
        /// Failed identity, range, length, or checksum-witness comparison.
        detail: String,
    },
    /// A queryable segment omitted one required identity section.
    #[error("Quill segment at {path} is missing required identity section {kind:?}")]
    MissingIdentitySection {
        /// Canonical published segment path.
        path: PathBuf,
        /// Missing IDMAP or IDHASH section.
        kind: SectionKind,
    },
    /// A segment's IDMAP payload failed canonical cross-section validation.
    #[error("corrupt Quill IDMAP at {path}: {source}")]
    IdMapCorrupted {
        /// Canonical published segment path.
        path: PathBuf,
        /// Typed IDMAP diagnosis.
        #[source]
        source: IdMapCodecError,
    },
    /// A segment's IDHASH payload failed canonical IDMAP-bound validation.
    #[error("corrupt Quill IDHASH at {path}: {source}")]
    IdHashCorrupted {
        /// Canonical published segment path.
        path: PathBuf,
        /// Typed IDHASH diagnosis.
        #[source]
        source: IdHashCodecError,
    },
    /// IDMAP presence disagreed with the segment and MANIFEST at-seal count.
    #[error("Quill segment document count mismatch at {path}: manifest={manifest}, IDMAP={id_map}")]
    AtSealDocCountMismatch {
        /// Canonical published segment path.
        path: PathBuf,
        /// Count authenticated by the MANIFEST and segment header.
        manifest: u32,
        /// Present rows validated from IDMAP.
        id_map: usize,
    },
    /// A durable tombstone named a positional IDMAP hole.
    #[error(
        "Quill segment {segment_id:#018x} at {path} tombstones missing IDMAP row {global_docid}"
    )]
    TombstoneReferencesHole {
        /// Canonical published segment path.
        path: PathBuf,
        /// Segment identity carrying the invalid tombstone.
        segment_id: u64,
        /// Tombstoned global document id with no physical row.
        global_docid: u32,
    },
    /// More than one current segment exposed the same live external identifier.
    #[error(
        "Quill upsert invariant violated in {directory}: document identifier resolves to live docids {first_global_docid} and {duplicate_global_docid}"
    )]
    MultipleLiveDocumentIds {
        /// Snapshot directory, or the in-memory sentinel.
        directory: PathBuf,
        /// First live global document id in descending seal order.
        first_global_docid: u32,
        /// Additional live global document id.
        duplicate_global_docid: u32,
    },
    /// A fallback MANIFEST and an extant generation claim disagree.
    #[error(
        "invalid interrupted-publish claim at {path}: recovered generation {recovered}, claimed generation {claimed}"
    )]
    InvalidRecoveryClaim {
        /// Claim path carrying the incompatible generation.
        path: PathBuf,
        /// Generation selected from `MANIFEST.prev`.
        recovered: u64,
        /// Generation named by the claim.
        claimed: u64,
    },
    /// A canonical generation claim is not the required zero-length file.
    #[error("invalid Quill generation claim at {path}: {detail}")]
    InvalidClaimArtifact {
        /// Canonical claim path.
        path: PathBuf,
        /// Failed no-follow type or length invariant.
        detail: String,
    },
    /// A claimed future generation must be resolved before garbage collection.
    #[error(
        "writer recovery required at {path}: current generation {current}, claimed generation {claimed}"
    )]
    ClaimedGenerationPending {
        /// Future claim blocking a garbage sweep.
        path: PathBuf,
        /// Selected durable generation.
        current: u64,
        /// Future claimed generation.
        claimed: u64,
    },
    /// A garbage-collection target was not one safe direct child name.
    #[error("unsafe Quill garbage path rejected: {path}")]
    UnsafeGarbagePath {
        /// Rejected relative or absolute target.
        path: PathBuf,
    },
    /// The supplied index-directory path stopped naming the opened directory.
    #[error("Quill index directory changed during garbage collection: {directory}")]
    GarbageDirectoryChanged {
        /// Directory whose identity was not stable for the sweep.
        directory: PathBuf,
    },
    /// Publishing over a corrupt primary requires the later writer-recovery lane.
    #[error("manifest recovery required before publish at {path}")]
    RecoveryRequired {
        /// Corrupt primary slot.
        path: PathBuf,
    },
    /// A stale temp file exists but is not byte-identical to this proposal.
    #[error("manifest temp conflicts with proposed bytes at {path}")]
    TempConflict {
        /// Temp file that cannot be safely reused or overwritten.
        path: PathBuf,
    },
    /// The cancel-aware in-process publish mutex was not acquired.
    #[error("manifest publish lock failed: {source}")]
    PublishLock {
        /// Asupersync lock failure.
        #[source]
        source: LockError,
    },
    /// Optional repair-sidecar processing failed.
    #[cfg(feature = "durability")]
    #[error("keeper durability {operation} failed at {path}: {source}")]
    Durability {
        /// Durability operation that failed.
        operation: &'static str,
        /// Source or sidecar path being processed.
        path: PathBuf,
        /// Generic durability-layer diagnosis.
        #[source]
        source: SearchError,
    },
    /// A filesystem operation failed with path and operation context.
    #[error("keeper {operation} failed at {path}: {source}")]
    Io {
        /// Operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
}

impl From<ConcatMergeError> for KeeperError {
    fn from(source: ConcatMergeError) -> Self {
        Self::ConcatMerge { source }
    }
}

impl From<CompactionError> for KeeperError {
    fn from(source: CompactionError) -> Self {
        Self::Compaction { source }
    }
}

impl From<KeeperError> for SearchError {
    fn from(error: KeeperError) -> Self {
        match error {
            KeeperError::IndexNotFound { directory } => Self::IndexNotFound { path: directory },
            KeeperError::ManifestCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::NoValidManifest {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!("no valid manifest: current={current}; previous={previous}"),
            },
            KeeperError::InvalidGenerationPair {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest generation pair: current={current}, previous={previous}"
                ),
            },
            KeeperError::InvalidManifestPair {
                directory,
                current,
                previous,
                detail,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest pair at generations {current}/{previous}: {detail}"
                ),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::IndexCorrupted { detail, .. },
            } => Self::IndexCorrupted { path, detail },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::Io(source),
            } if source.kind() == io::ErrorKind::NotFound => Self::IndexCorrupted {
                path,
                detail: "MANIFEST-referenced segment is missing".to_owned(),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::UnknownSchema { schema_id },
            } => Self::IndexCorrupted {
                path,
                detail: format!("MANIFEST-referenced segment has unknown schema {schema_id:#018x}"),
            },
            KeeperError::SegmentOpen { path, source } => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(KeeperError::SegmentOpen { path, source }),
            },
            KeeperError::SegmentMetadataMismatch { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::MissingIdentitySection { path, kind } => Self::IndexCorrupted {
                path,
                detail: format!("MANIFEST-referenced segment is missing {kind:?}"),
            },
            KeeperError::IdMapCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::IdHashCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::AtSealDocCountMismatch {
                path,
                manifest,
                id_map,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "at-seal document count mismatch: manifest={manifest}, IDMAP={id_map}"
                ),
            },
            KeeperError::TombstoneReferencesHole {
                path,
                segment_id,
                global_docid,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "segment {segment_id:#018x} tombstones missing IDMAP row {global_docid}"
                ),
            },
            KeeperError::MultipleLiveDocumentIds {
                directory,
                first_global_docid,
                duplicate_global_docid,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "document identifier has multiple live rows {first_global_docid}/{duplicate_global_docid}"
                ),
            },
            KeeperError::InvalidRecoveryClaim {
                path,
                recovered,
                claimed,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "recovered manifest generation {recovered} conflicts with claim {claimed}"
                ),
            },
            KeeperError::InvalidClaimArtifact { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::WriterAdmissionCancelled => Self::Cancelled {
                phase: "keeper.writer_admission".to_owned(),
                reason: "writer admission cancelled".to_owned(),
            },
            KeeperError::ConcatMerge {
                source: ConcatMergeError::Cancelled,
            } => Self::Cancelled {
                phase: "keeper.concat_merge".to_owned(),
                reason: "concat merge cancelled before MANIFEST publication".to_owned(),
            },
            KeeperError::Compaction {
                source: CompactionError::Cancelled,
            } => Self::Cancelled {
                phase: "keeper.compact".to_owned(),
                reason: "compaction cancelled before MANIFEST publication".to_owned(),
            },
            KeeperError::PublishLock { source } => match source {
                LockError::Cancelled => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: "manifest publish lock cancelled".to_owned(),
                },
                LockError::TimedOut(deadline) => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: format!("manifest publish lock timed out at {deadline:?}"),
                },
                other => Self::SubsystemError {
                    subsystem: "quill",
                    source: Box::new(KeeperError::PublishLock { source: other }),
                },
            },
            other => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(other),
            },
        }
    }
}

/// Owned canonical roaring-lite tombstone set.
///
/// The exact wire bytes remain authoritative so decoding preserves ARRAY or
/// BITMAP representation inside the hysteresis overlap. Cached cardinality
/// makes live-count rollups constant-time, while mutation rewrites only this
/// small delete-path value and copies untouched containers byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TombstoneSet {
    encoded: Vec<u8>,
    cardinality: u64,
}

impl TombstoneSet {
    /// Construct the canonical empty set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoded: EMPTY_TOMBSTONES.to_vec(),
            cardinality: 0,
        }
    }

    /// Build the one history-independent representation for sorted membership.
    ///
    /// Replacement segments use this path so equivalent merge schedules cannot
    /// retain ARRAY/BITMAP hysteresis from their source MANIFEST generations.
    fn from_sorted_docids(docids: &[u32]) -> Result<Self, ManifestCodecError> {
        if docids.is_empty() {
            return Ok(Self::new());
        }
        if let Some([previous, current]) = docids.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(invalid(format!(
                "concat tombstones are not strictly ascending at {}/{}",
                previous, current
            )));
        }

        let mut chunk_count = 0_usize;
        let mut encoded_len = std::mem::size_of::<u32>();
        let mut start = 0_usize;
        while start < docids.len() {
            let chunk_id = split_tombstone_docid(docids[start]).0;
            let end = start
                + docids[start..]
                    .partition_point(|docid| split_tombstone_docid(*docid).0 == chunk_id);
            let cardinality = end - start;
            let payload_len = if cardinality <= usize::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) {
                cardinality.checked_mul(std::mem::size_of::<u16>())
            } else {
                Some(TOMBSTONE_BITMAP_BYTES)
            }
            .ok_or_else(|| invalid("concat tombstone payload length overflow"))?;
            encoded_len = encoded_len
                .checked_add(5)
                .and_then(|length| length.checked_add(payload_len))
                .ok_or_else(|| invalid("concat tombstone wire length overflow"))?;
            chunk_count = chunk_count
                .checked_add(1)
                .ok_or_else(|| invalid("concat tombstone chunk count overflow"))?;
            start = end;
        }
        let chunk_count_u32 = u32::try_from(chunk_count)
            .map_err(|_| invalid("concat tombstone chunk count does not fit u32"))?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(encoded_len)
            .map_err(|error| invalid(format!("concat tombstone allocation failed: {error}")))?;
        put_u32(&mut encoded, chunk_count_u32);
        let mut start = 0_usize;
        while start < docids.len() {
            let chunk_id = split_tombstone_docid(docids[start]).0;
            let end = start
                + docids[start..]
                    .partition_point(|docid| split_tombstone_docid(*docid).0 == chunk_id);
            let cardinality = end - start;
            if cardinality <= usize::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) {
                write_tombstone_container_header(
                    &mut encoded,
                    chunk_id,
                    0,
                    u64::try_from(cardinality).unwrap_or(u64::MAX),
                )?;
                for &docid in &docids[start..end] {
                    put_u16(&mut encoded, split_tombstone_docid(docid).1);
                }
            } else {
                write_tombstone_container_header(
                    &mut encoded,
                    chunk_id,
                    1,
                    u64::try_from(cardinality).unwrap_or(u64::MAX),
                )?;
                let payload_start = encoded.len();
                encoded.resize(payload_start + TOMBSTONE_BITMAP_BYTES, 0);
                for &docid in &docids[start..end] {
                    set_bitmap_value(
                        &mut encoded[payload_start..],
                        split_tombstone_docid(docid).1,
                    );
                }
            }
            start = end;
        }
        let cardinality = u64::try_from(docids.len())
            .map_err(|_| invalid("concat tombstone cardinality does not fit u64"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&encoded, None).ok(),
            Some(cardinality)
        );
        Ok(Self::from_validated_bytes(encoded, cardinality))
    }

    /// Parse canonical roaring-lite bytes without normalizing container kinds.
    ///
    /// # Errors
    ///
    /// Rejects truncation, unknown kinds, noncanonical ordering/cardinality,
    /// resource-limit violations, trailing bytes, or allocation failure.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        let cardinality = validate_tombstone_bytes(bytes, None)?;
        let encoded = copy_bytes(bytes, "tombstone set")?;
        Ok(Self {
            encoded,
            cardinality,
        })
    }

    fn from_validated_bytes(encoded: Vec<u8>, cardinality: u64) -> Self {
        Self {
            encoded,
            cardinality,
        }
    }

    /// Borrow the exact canonical wire representation.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.encoded
    }

    /// Number of tombstoned global document ids.
    #[must_use]
    pub const fn cardinality(&self) -> u64 {
        self.cardinality
    }

    /// Whether the set contains no document ids.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Test one global document id without allocating.
    #[must_use]
    pub fn contains(&self, global_docid: u32) -> bool {
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let Ok(mut containers) = TombstoneContainers::new(&self.encoded) else {
            return false;
        };
        while let Ok(Some(container)) = containers.next_container() {
            match container.chunk_id.cmp(&chunk_id) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => {
                    return match container.kind {
                        0 => array_binary_search(container.payload, low),
                        1 => bitmap_contains(container.payload, low),
                        _ => false,
                    };
                }
            }
        }
        false
    }

    /// Insert one global document id, promoting ARRAY to BITMAP at 4,097.
    ///
    /// # Errors
    ///
    /// Returns an allocation or invariant error without changing the set.
    pub fn insert(&mut self, global_docid: u32) -> Result<bool, ManifestCodecError> {
        if self.contains(global_docid) {
            return Ok(false);
        }
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let chunk_count = tombstone_chunk_count(&self.encoded)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(
                self.encoded
                    .len()
                    .checked_add(7)
                    .ok_or_else(|| invalid("tombstone insertion length overflow"))?,
            )
            .map_err(|error| invalid(format!("tombstone insertion allocation failed: {error}")))?;

        let mut containers = TombstoneContainers::new(&self.encoded)?;
        let mut inserted = false;
        let adds_chunk = !tombstone_chunk_exists(&self.encoded, chunk_id)?;
        let next_chunk_count = chunk_count
            .checked_add(u32::from(adds_chunk))
            .ok_or_else(|| invalid("tombstone chunk count overflow"))?;
        put_u32(&mut output, next_chunk_count);
        while let Some(container) = containers.next_container()? {
            if !inserted && chunk_id < container.chunk_id {
                write_singleton_tombstone_chunk(&mut output, chunk_id, low);
                inserted = true;
            }
            if container.chunk_id == chunk_id {
                write_inserted_tombstone_container(&mut output, container, low)?;
                inserted = true;
            } else {
                output.extend_from_slice(container.encoded);
            }
        }
        if !inserted {
            write_singleton_tombstone_chunk(&mut output, chunk_id, low);
        }
        let next_cardinality = self
            .cardinality
            .checked_add(1)
            .ok_or_else(|| invalid("tombstone cardinality overflow"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&output, None).ok(),
            Some(next_cardinality)
        );
        self.encoded = output;
        self.cardinality = next_cardinality;
        Ok(true)
    }

    /// Remove one id, retaining BITMAP through 3,584 and demoting at 3,583.
    ///
    /// Retained MANIFEST segments prohibit shrinking tombstones; this mutation
    /// is intended for isolated codec work and later replacement-segment
    /// compaction, not an in-place publication transition.
    ///
    /// # Errors
    ///
    /// Returns an allocation or invariant error without changing the set.
    pub fn remove(&mut self, global_docid: u32) -> Result<bool, ManifestCodecError> {
        if !self.contains(global_docid) {
            return Ok(false);
        }
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let chunk_count = tombstone_chunk_count(&self.encoded)?;
        let mut containers = TombstoneContainers::new(&self.encoded)?;
        let mut target_cardinality = None;
        while let Some(container) = containers.next_container()? {
            if container.chunk_id == chunk_id {
                target_cardinality = Some(container.cardinality);
                break;
            }
        }
        let target_cardinality =
            target_cardinality.ok_or_else(|| invalid("tombstone membership/container mismatch"))?;
        let removes_chunk = target_cardinality == 1;
        let next_chunk_count = chunk_count
            .checked_sub(u32::from(removes_chunk))
            .ok_or_else(|| invalid("tombstone chunk count underflow"))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.encoded.len())
            .map_err(|error| invalid(format!("tombstone removal allocation failed: {error}")))?;
        put_u32(&mut output, next_chunk_count);
        let mut containers = TombstoneContainers::new(&self.encoded)?;
        while let Some(container) = containers.next_container()? {
            if container.chunk_id == chunk_id {
                write_removed_tombstone_container(&mut output, container, low)?;
            } else {
                output.extend_from_slice(container.encoded);
            }
        }
        let next_cardinality = self
            .cardinality
            .checked_sub(1)
            .ok_or_else(|| invalid("tombstone cardinality underflow"))?;
        debug_assert_eq!(
            validate_tombstone_bytes(&output, None).ok(),
            Some(next_cardinality)
        );
        self.encoded = output;
        self.cardinality = next_cardinality;
        Ok(true)
    }

    fn validate_range(&self, range: (u64, u64)) -> Result<(), ManifestCodecError> {
        let validated = validate_tombstone_bytes(&self.encoded, Some(range))?;
        if validated != self.cardinality {
            return Err(invalid("cached tombstone cardinality mismatch"));
        }
        Ok(())
    }

    fn is_monotone_superset_of(&self, previous: &Self) -> Result<bool, ManifestCodecError> {
        if previous == self {
            return Ok(true);
        }
        if previous.cardinality > self.cardinality
            || !tombstones_are_subset(&previous.encoded, &self.encoded)?
        {
            return Ok(false);
        }

        // Every chunk follows the mutation state machine across generations:
        // ARRAY promotes only above 4,096 entries, while a retained BITMAP
        // never demotes in this growth-only transition. This also rejects a
        // newly introduced, prematurely promoted BITMAP in the overlap.
        let mut old = TombstoneContainers::new(&previous.encoded)?;
        let mut new = TombstoneContainers::new(&self.encoded)?;
        let mut prior = old.next_container()?;
        while let Some(current) = new.next_container()? {
            if prior.is_some_and(|container| container.chunk_id < current.chunk_id) {
                return Ok(false);
            }
            let previous_kind = prior
                .filter(|container| container.chunk_id == current.chunk_id)
                .map(|container| container.kind);
            let representation_is_valid = match previous_kind {
                Some(1) => current.kind == 1,
                Some(0) | None => {
                    current.kind == 0
                        || (current.kind == 1
                            && current.cardinality > u64::from(TOMBSTONE_ARRAY_MAX_CARDINALITY))
                }
                _ => false,
            };
            if !representation_is_valid {
                return Ok(false);
            }
            if previous_kind.is_some() {
                prior = old.next_container()?;
            }
        }
        Ok(prior.is_none())
    }

    fn encoded_len(&self) -> usize {
        self.encoded.len()
    }
}

impl Default for TombstoneSet {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<&[u8]> for TombstoneSet {
    type Error = ManifestCodecError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

/// One immutable segment referenced by a MANIFEST generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSegment {
    /// Random, collision-checked segment identifier.
    pub segment_id: u64,
    /// Monotone seal order used for newest-first IDHASH probing.
    pub seal_seq: u64,
    /// Exact FSLX file length.
    pub file_len: u64,
    /// xxh3-64 over the FSLX prefix before the trailer.
    pub file_xxh3: u64,
    /// Inclusive lower bound of the segment's Q1 range.
    pub docid_lo: u64,
    /// Exclusive upper bound of the segment's Q1 range.
    pub docid_hi: u64,
    /// Live-at-seal document count.
    pub doc_count: u32,
    /// Canonical roaring-lite tombstone bytes.
    pub tombstones: TombstoneSet,
}

impl ManifestSegment {
    /// Live rows after applying this generation's tombstones.
    #[must_use]
    pub fn live_doc_count(&self) -> u32 {
        self.doc_count
            .saturating_sub(u32::try_from(self.tombstones.cardinality()).unwrap_or(u32::MAX))
    }

    /// Whether a ranged physical candidate survives this segment's tombstones.
    #[must_use]
    pub fn is_live(&self, global_docid: u32) -> bool {
        (self.docid_lo..self.docid_hi).contains(&u64::from(global_docid))
            && !self.tombstones.contains(global_docid)
    }

    fn insert_tombstone(&mut self, global_docid: u32) -> Result<bool, ManifestCodecError> {
        if !(self.docid_lo..self.docid_hi).contains(&u64::from(global_docid)) {
            return Err(invalid(format!(
                "tombstoned docid {global_docid} is outside segment range [{}, {})",
                self.docid_lo, self.docid_hi
            )));
        }
        if self.tombstones.contains(global_docid) {
            return Ok(false);
        }
        if self.tombstones.cardinality() >= u64::from(self.doc_count) {
            return Err(invalid(format!(
                "segment {:#018x} cannot tombstone more than {} physical rows",
                self.segment_id, self.doc_count
            )));
        }
        self.tombstones.insert(global_docid)
    }
}

/// Size class assigned from one segment's Q1 covering-interval width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentSizeTier {
    /// Width is at most [`TierMergePolicy::small_max_docid_width`].
    Small,
    /// Width is above small and at most [`TierMergePolicy::medium_max_docid_width`].
    Medium,
    /// Width exceeds the medium boundary.
    Large,
}

/// Explicit configuration consumed by the bound-consecutive tier planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierMergePolicy {
    /// Same-tier run length that triggers one concat merge.
    pub fanout: usize,
    /// Inclusive upper width of the small tier.
    pub small_max_docid_width: u64,
    /// Inclusive upper width of the medium tier.
    pub medium_max_docid_width: u64,
    /// Maximum fraction of the output hull occupied by inter-segment holes.
    pub max_hole_ratio: f64,
}

impl TierMergePolicy {
    /// Build the Keeper policy from Quill's validated engine configuration.
    #[must_use]
    pub const fn from_config(config: &crate::config::QuillConfig) -> Self {
        Self {
            fanout: config.tier_fanout,
            small_max_docid_width: config.tier_small_max_docid_width,
            medium_max_docid_width: config.tier_medium_max_docid_width,
            max_hole_ratio: config.merge_max_hole_ratio,
        }
    }

    /// Classify one nonempty covering interval by width.
    #[must_use]
    pub const fn classify_width(self, width: u64) -> SegmentSizeTier {
        if width <= self.small_max_docid_width {
            SegmentSizeTier::Small
        } else if width <= self.medium_max_docid_width {
            SegmentSizeTier::Medium
        } else {
            SegmentSizeTier::Large
        }
    }
}

/// One exact manifest slice selected for a Q1-preserving concat merge.
#[derive(Debug, Clone, PartialEq)]
pub struct TierMergePlan {
    /// Common size tier of every source segment.
    pub tier: SegmentSizeTier,
    /// Source identities in current manifest order.
    pub source_segment_ids: Vec<u64>,
    /// Inclusive lower bound of the output hull.
    pub docid_lo: u64,
    /// Exclusive upper bound of the output hull.
    pub docid_hi: u64,
    /// Fraction of the output hull represented by gaps between sources.
    pub hole_ratio: f64,
}

/// Typed failure from allocation-free policy scanning or plan materialization.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TierPolicyError {
    /// A policy must merge at least two segments at a time.
    #[error("tier merge fanout must be at least two, got {fanout}")]
    InvalidFanout {
        /// Rejected fanout.
        fanout: usize,
    },
    /// The tier boundaries must be nonzero and strictly ascending.
    #[error("tier boundaries must satisfy 0 < small < medium, got small={small}, medium={medium}")]
    InvalidBoundaries {
        /// Rejected small boundary.
        small: u64,
        /// Rejected medium boundary.
        medium: u64,
    },
    /// The optional hole-ratio gate must be a finite fraction.
    #[error("tier merge hole ratio must be finite and in [0, 1], got {ratio}")]
    InvalidHoleRatio {
        /// Rejected ratio rendered without float equality in the error type.
        ratio: String,
    },
    /// A standalone segment supplied to the planner has an empty or reversed interval.
    #[error("segment {segment_id} has invalid docid interval [{docid_lo}, {docid_hi})")]
    InvalidSegmentRange {
        /// Rejected segment identity.
        segment_id: u64,
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// Standalone planner inputs must already be sorted and pairwise disjoint.
    #[error("segments {left_segment_id} and {right_segment_id} overlap or are out of docid order")]
    InvalidSegmentOrder {
        /// Earlier segment identity.
        left_segment_id: u64,
        /// Later segment identity.
        right_segment_id: u64,
    },
    /// A selected source-id vector could not be reserved.
    #[error("could not allocate {count} tier merge source ids")]
    Allocation {
        /// Requested source count.
        count: usize,
    },
}

/// Select the first same-tier, bound-consecutive manifest run admitted by the
/// optional hole-ratio gate.
///
/// The planner validates interval ordering and disjointness before selecting an
/// uninterrupted slice. That slice is therefore exactly Q1 R2: the output hull
/// cannot contain a skipped live interval, including when it crosses shards.
///
/// # Errors
///
/// Returns [`TierPolicyError`] for an invalid standalone policy, malformed
/// segment layout, or allocation failure. Engine callers normally construct
/// the policy from a validated [`crate::config::QuillConfig`].
pub fn plan_tier_merge(
    segments: &[ManifestSegment],
    policy: TierMergePolicy,
) -> Result<Option<TierMergePlan>, TierPolicyError> {
    validate_tier_policy(policy)?;
    validate_tier_segments(segments)?;
    if segments.len() < policy.fanout {
        return Ok(None);
    }
    for sources in segments.windows(policy.fanout) {
        let first = &sources[0];
        let tier = policy.classify_width(first.docid_hi - first.docid_lo);
        if sources
            .iter()
            .any(|segment| policy.classify_width(segment.docid_hi - segment.docid_lo) != tier)
        {
            continue;
        }
        let docid_lo = first.docid_lo;
        let docid_hi = sources[sources.len() - 1].docid_hi;
        let hull_width = docid_hi - docid_lo;
        let occupied_width = sources
            .iter()
            .map(|segment| segment.docid_hi - segment.docid_lo)
            .sum::<u64>();
        let hole_width = hull_width.saturating_sub(occupied_width);
        let hole_ratio = if hull_width == 0 {
            0.0
        } else {
            hole_width as f64 / hull_width as f64
        };
        if hole_ratio > policy.max_hole_ratio {
            continue;
        }
        let mut source_segment_ids = Vec::new();
        source_segment_ids
            .try_reserve_exact(sources.len())
            .map_err(|_| TierPolicyError::Allocation {
                count: sources.len(),
            })?;
        source_segment_ids.extend(sources.iter().map(|segment| segment.segment_id));
        return Ok(Some(TierMergePlan {
            tier,
            source_segment_ids,
            docid_lo,
            docid_hi,
            hole_ratio,
        }));
    }
    Ok(None)
}

fn validate_tier_policy(policy: TierMergePolicy) -> Result<(), TierPolicyError> {
    if policy.fanout < 2 {
        return Err(TierPolicyError::InvalidFanout {
            fanout: policy.fanout,
        });
    }
    if policy.small_max_docid_width == 0
        || policy.medium_max_docid_width <= policy.small_max_docid_width
    {
        return Err(TierPolicyError::InvalidBoundaries {
            small: policy.small_max_docid_width,
            medium: policy.medium_max_docid_width,
        });
    }
    if !policy.max_hole_ratio.is_finite() || !(0.0..=1.0).contains(&policy.max_hole_ratio) {
        return Err(TierPolicyError::InvalidHoleRatio {
            ratio: policy.max_hole_ratio.to_string(),
        });
    }
    Ok(())
}

fn validate_tier_segments(segments: &[ManifestSegment]) -> Result<(), TierPolicyError> {
    for segment in segments {
        if segment.docid_lo >= segment.docid_hi {
            return Err(TierPolicyError::InvalidSegmentRange {
                segment_id: segment.segment_id,
                docid_lo: segment.docid_lo,
                docid_hi: segment.docid_hi,
            });
        }
    }
    for pair in segments.windows(2) {
        if pair[0].docid_hi > pair[1].docid_lo {
            return Err(TierPolicyError::InvalidSegmentOrder {
                left_segment_id: pair[0].segment_id,
                right_segment_id: pair[1].segment_id,
            });
        }
    }
    Ok(())
}

/// Snapshot-level field statistics rolled up over referenced segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestFieldStats {
    /// Schema field ordinal.
    pub field_ord: u16,
    /// Sum of at-seal token counts.
    pub total_tokens: u64,
    /// Sum of at-seal document counts for this field.
    pub doc_count: u32,
}

/// Fully owned MANIFEST v2 contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Monotone publication generation. Genesis is generation 1.
    pub generation: u64,
    /// First docid that has never been leased; never rolls back.
    pub docid_high_watermark: u64,
    /// Durable schema descriptor hash.
    pub schema_id: u64,
    /// Informational packed Quill crate version.
    pub engine_version: u32,
    /// Versioned manifest flags.
    pub flags: u32,
    /// Wall-clock seconds when this generation was published; `0` means
    /// unknown (v1 images and hand-built manifests). The publisher stamps a
    /// zero field with the current time, so every keeper-written generation
    /// carries the cross-process freshness witness required by the
    /// visibility contract (bead bd-quill-duel-visibility-contract-9rk3).
    /// Informational only: never validated for monotonicity, because NTP
    /// steps make wall-clock ordering unreliable.
    pub last_publish_unix_s: i64,
    /// Segments in strictly ascending `docid_lo` order.
    pub segments: Vec<ManifestSegment>,
    /// Field statistics in strictly ascending `field_ord` order.
    pub field_stats: Vec<ManifestFieldStats>,
}

impl Manifest {
    /// Construct an empty generation using the current engine version.
    #[must_use]
    pub const fn empty(generation: u64, schema_id: u64, docid_high_watermark: u64) -> Self {
        Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            last_publish_unix_s: 0,
            segments: Vec::new(),
            field_stats: Vec::new(),
        }
    }

    /// Whether the durable bulk-mode flag is set.
    #[must_use]
    pub const fn bulk_mode_in_progress(&self) -> bool {
        self.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS != 0
    }

    /// Validate writer-side canonical form and Q1 range invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant or resource error before any bytes are
    /// written.
    pub fn validate(&self) -> Result<(), ManifestCodecError> {
        validate_manifest(self, ErrorClass::Invalid)
    }

    /// Encode canonical MANIFEST v1 bytes including the trailing CRC32.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid in-memory state, count conversion, or an
    /// output exceeding the eager-reader budget.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ManifestCodecError> {
        validate_manifest_shape(self, ErrorClass::Invalid)?;
        let estimated = manifest_encoded_len(self)?;
        self.validate()?;

        let segment_count =
            u32::try_from(self.segments.len()).map_err(|_| ManifestCodecError::ResourceLimit {
                resource: "segment count",
                actual: usize_to_u64(self.segments.len()),
                limit: u64::from(u32::MAX),
            })?;
        let field_count = u32::try_from(self.field_stats.len()).map_err(|_| {
            ManifestCodecError::ResourceLimit {
                resource: "field count",
                actual: usize_to_u64(self.field_stats.len()),
                limit: u64::from(u32::MAX),
            }
        })?;

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(estimated)
            .map_err(|error| invalid(format!("manifest allocation failed: {error}")))?;
        bytes.extend_from_slice(&MANIFEST_MAGIC);
        put_u32(&mut bytes, MANIFEST_FORMAT_VERSION);
        put_u64(&mut bytes, self.generation);
        put_u64(&mut bytes, self.docid_high_watermark);
        put_u64(&mut bytes, self.schema_id);
        put_u32(&mut bytes, self.engine_version);
        put_u32(&mut bytes, self.flags);
        bytes.extend_from_slice(&self.last_publish_unix_s.to_le_bytes());
        put_u32(&mut bytes, segment_count);
        for segment in &self.segments {
            put_u64(&mut bytes, segment.segment_id);
            put_u64(&mut bytes, segment.seal_seq);
            put_u64(&mut bytes, segment.file_len);
            put_u64(&mut bytes, segment.file_xxh3);
            put_u64(&mut bytes, segment.docid_lo);
            put_u64(&mut bytes, segment.docid_hi);
            put_u32(&mut bytes, segment.doc_count);
            bytes.extend_from_slice(segment.tombstones.as_bytes());
        }
        put_u32(&mut bytes, field_count);
        for stats in &self.field_stats {
            put_u16(&mut bytes, stats.field_ord);
            put_u64(&mut bytes, stats.total_tokens);
            put_u32(&mut bytes, stats.doc_count);
        }
        let checksum = crc32fast::hash(&bytes);
        put_u32(&mut bytes, checksum);
        debug_assert_eq!(bytes.len(), estimated);
        Ok(bytes)
    }

    /// Parse and eagerly validate canonical MANIFEST v1 bytes.
    ///
    /// # Errors
    ///
    /// Rejects unsupported versions, checksum failures, hostile counts,
    /// non-canonical tombstones, Q1 violations, truncation, and trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        check_manifest_byte_limit(bytes.len())?;
        if bytes.len() < MANIFEST_MIN_BYTES {
            return Err(ManifestCodecError::Truncated {
                offset: 0,
                needed: MANIFEST_MIN_BYTES,
                remaining: bytes.len(),
            });
        }

        let body_len = bytes.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            bytes[body_len],
            bytes[body_len + 1],
            bytes[body_len + 2],
            bytes[body_len + 3],
        ]);
        let computed_checksum = crc32fast::hash(&bytes[..body_len]);
        if stored_checksum != computed_checksum {
            return Err(ManifestCodecError::ChecksumMismatch {
                stored: stored_checksum,
                computed: computed_checksum,
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..body_len]);
        if cursor.take(8)? != MANIFEST_MAGIC {
            return Err(ManifestCodecError::BadMagic);
        }
        let version = cursor.u32()?;
        if version != MANIFEST_FORMAT_VERSION && version != MANIFEST_FORMAT_VERSION_V1 {
            return Err(ManifestCodecError::UnsupportedVersion { found: version });
        }
        let generation = cursor.u64()?;
        let docid_high_watermark = cursor.u64()?;
        let schema_id = cursor.u64()?;
        let engine_version = cursor.u32()?;
        let flags = cursor.u32()?;
        let last_publish_unix_s = if version == MANIFEST_FORMAT_VERSION {
            i64::from_le_bytes(cursor.take(8)?.try_into().map_err(|_| {
                ManifestCodecError::Truncated {
                    offset: cursor.position(),
                    needed: 8,
                    remaining: cursor.remaining(),
                }
            })?)
        } else {
            0
        };
        let segment_count = count_to_usize(cursor.u32()?, "segment count", MAX_MANIFEST_SEGMENTS)?;
        let minimum_segment_bytes = segment_count
            .checked_mul(SEGMENT_FIXED_BYTES + EMPTY_TOMBSTONES.len())
            .ok_or_else(|| non_canonical("segment byte count overflow"))?;
        if minimum_segment_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_segment_bytes,
                remaining: cursor.remaining(),
            });
        }

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(segment_count)
            .map_err(|error| non_canonical(format!("segment allocation failed: {error}")))?;
        for _ in 0..segment_count {
            let segment_id = cursor.u64()?;
            let seal_seq = cursor.u64()?;
            let file_len = cursor.u64()?;
            let file_xxh3 = cursor.u64()?;
            let docid_lo = cursor.u64()?;
            let docid_hi = cursor.u64()?;
            let doc_count = cursor.u32()?;
            let tombstone_start = cursor.position();
            let tombstone_count = consume_tombstone_set(&mut cursor, None)?;
            let tombstone_bytes = copy_bytes(
                &bytes[tombstone_start..cursor.position()],
                "tombstone bytes",
            )?;
            let tombstones = TombstoneSet::from_validated_bytes(tombstone_bytes, tombstone_count);
            segments.push(ManifestSegment {
                segment_id,
                seal_seq,
                file_len,
                file_xxh3,
                docid_lo,
                docid_hi,
                doc_count,
                tombstones,
            });
        }

        let field_count = count_to_usize(cursor.u32()?, "field count", MAX_MANIFEST_FIELDS)?;
        let minimum_stats_bytes = field_count
            .checked_mul(FIELD_STATS_BYTES)
            .ok_or_else(|| non_canonical("field-stats byte count overflow"))?;
        if minimum_stats_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_stats_bytes,
                remaining: cursor.remaining(),
            });
        }
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(field_count)
            .map_err(|error| non_canonical(format!("field-stats allocation failed: {error}")))?;
        for _ in 0..field_count {
            field_stats.push(ManifestFieldStats {
                field_ord: cursor.u16()?,
                total_tokens: cursor.u64()?,
                doc_count: cursor.u32()?,
            });
        }
        if cursor.remaining() != 0 {
            return Err(ManifestCodecError::TrailingBytes {
                remaining: cursor.remaining(),
            });
        }

        let manifest = Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version,
            flags,
            last_publish_unix_s,
            segments,
            field_stats,
        };
        validate_manifest(&manifest, ErrorClass::NonCanonical)?;
        Ok(manifest)
    }
}

/// Slot selected by read-only two-slot recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSource {
    /// The primary `MANIFEST` was valid.
    Current,
    /// `MANIFEST` was absent, as in the crash window between the two renames.
    PreviousAfterMissingCurrent,
    /// `MANIFEST` existed but was corrupt, so only `MANIFEST.prev` was usable.
    PreviousAfterCorruptCurrent,
    /// No filesystem is attached; this is an owned-buffer genesis snapshot.
    InMemory,
}

/// A validated MANIFEST together with its recovery provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedManifest {
    /// Validated contents.
    pub manifest: Manifest,
    /// Durable slot that supplied the contents.
    pub source: ManifestSource,
}

#[derive(Clone)]
struct TombstoneIndex {
    chunks: Vec<TombstoneChunkIndex>,
}

#[derive(Clone, Copy)]
struct TombstoneChunkIndex {
    chunk_id: u16,
    kind: u8,
    payload_offset: usize,
    payload_length: usize,
}

impl TombstoneIndex {
    fn build(tombstones: &TombstoneSet) -> Result<Self, ManifestCodecError> {
        let chunk_count =
            usize::try_from(tombstone_chunk_count(tombstones.as_bytes())?).map_err(|_| {
                ManifestCodecError::ResourceLimit {
                    resource: "host tombstone chunk count",
                    actual: u64::from(u32::MAX),
                    limit: usize_to_u64(usize::MAX),
                }
            })?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count)
            .map_err(|error| invalid(format!("tombstone index allocation failed: {error}")))?;
        let mut containers = TombstoneContainers::new(tombstones.as_bytes())?;
        while let Some(container) = containers.next_container()? {
            chunks.push(TombstoneChunkIndex {
                chunk_id: container.chunk_id,
                kind: container.kind,
                payload_offset: container.payload_offset,
                payload_length: container.payload.len(),
            });
        }
        Ok(Self { chunks })
    }

    fn contains(&self, encoded: &[u8], global_docid: u32) -> bool {
        let (chunk_id, low) = split_tombstone_docid(global_docid);
        let Ok(index) = self
            .chunks
            .binary_search_by_key(&chunk_id, |chunk| chunk.chunk_id)
        else {
            return false;
        };
        let chunk = self.chunks[index];
        let Some(payload_end) = chunk.payload_offset.checked_add(chunk.payload_length) else {
            return false;
        };
        let Some(payload) = encoded.get(chunk.payload_offset..payload_end) else {
            return false;
        };
        match chunk.kind {
            0 => array_binary_search(payload, low),
            1 => bitmap_contains(payload, low),
            _ => false,
        }
    }
}

enum RecoveredSegmentBacking {
    Mapped(SegmentReader<ReadOnlyMappedFile>),
    Owned(SegmentReader<Vec<u8>>),
}

impl RecoveredSegmentBacking {
    fn header(&self) -> SegmentHeader {
        match self {
            Self::Mapped(reader) => reader.header(),
            Self::Owned(reader) => reader.header(),
        }
    }

    fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        match self {
            Self::Mapped(reader) => reader.section(kind),
            Self::Owned(reader) => reader.section(kind),
        }
    }

    fn section_entries(&self) -> &[SectionEntry] {
        match self {
            Self::Mapped(reader) => reader.section_entries(),
            Self::Owned(reader) => reader.section_entries(),
        }
    }

    fn verify(&self) -> Result<(), QuillError> {
        match self {
            Self::Mapped(reader) => reader.verify(),
            Self::Owned(reader) => reader.verify(),
        }
    }

    fn source_bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(reader) => reader.source_bytes(),
            Self::Owned(reader) => reader.source_bytes(),
        }
    }

    fn validate_witnesses(
        &self,
        path: &Path,
        manifest: &ManifestSegment,
    ) -> Result<(), KeeperError> {
        match self {
            Self::Mapped(reader) => validate_segment_witnesses(path, manifest, reader),
            Self::Owned(reader) => validate_segment_witnesses(path, manifest, reader),
        }
    }
}

fn required_identity_section<'a>(
    path: &Path,
    reader: &'a RecoveredSegmentBacking,
    kind: SectionKind,
) -> Result<&'a [u8], KeeperError> {
    reader
        .section(kind)
        .map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })?
        .ok_or_else(|| KeeperError::MissingIdentitySection {
            path: path.to_path_buf(),
            kind,
        })
}

fn first_tombstone_hole(
    tombstones: &TombstoneSet,
    id_map: IdMapSection<'_>,
) -> Result<Option<u32>, ManifestCodecError> {
    let mut containers = TombstoneContainers::new(tombstones.as_bytes())?;
    while let Some(container) = containers.next_container()? {
        match container.kind {
            0 => {
                for index in 0..container.payload.len() / 2 {
                    let global_docid = (u32::from(container.chunk_id) << 16)
                        | u32::from(array_value(container.payload, index));
                    if !id_map.contains(u64::from(global_docid)) {
                        return Ok(Some(global_docid));
                    }
                }
            }
            1 => {
                for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                    let mut bits = byte;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        let low = u16::try_from(byte_index * 8 + bit)
                            .map_err(|_| invalid("bitmap tombstone low-bit overflow"))?;
                        let global_docid = (u32::from(container.chunk_id) << 16) | u32::from(low);
                        if !id_map.contains(u64::from(global_docid)) {
                            return Ok(Some(global_docid));
                        }
                        bits &= bits - 1;
                    }
                }
            }
            _ => return Err(invalid("unknown validated tombstone container kind")),
        }
    }
    Ok(None)
}

/// One live external-identifier resolution routed to its immutable segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedDocumentId {
    /// Segment containing the unique live physical row.
    pub segment_id: u64,
    /// Current segment publication order witness.
    pub seal_seq: u64,
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Unseeded xxh3-64 witness over the canonical indexed document.
    pub content_hash: u64,
}

#[derive(Clone)]
struct CachedRankPruningTerm {
    term_ord: u32,
    term_metadata: TermMetadata,
    pruning: Arc<ValidatedTermPruningMetadata>,
}

const MAX_RANK_PRUNING_CACHE_TERMS: usize = 128;
const MAX_RANK_PRUNING_CACHE_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Sparse lock-free cache for immutable per-term rank-pruning metadata.
///
/// Searches overwhelmingly revisit a small vocabulary relative to the full
/// dictionary, so a copy-on-write sorted sidecar avoids allocating one cache
/// cell per durable term. Concurrent first touches may duplicate validation,
/// but compare-and-swap publishes exactly one paired result and readers never
/// block one another.
struct RankPruningCache {
    terms: ArcSwap<Vec<CachedRankPruningTerm>>,
}

impl RankPruningCache {
    fn new() -> Self {
        Self {
            terms: ArcSwap::from_pointee(Vec::new()),
        }
    }

    fn get(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
    ) -> Result<Option<Arc<ValidatedTermPruningMetadata>>, &'static str> {
        let terms = self.terms.load();
        let Ok(index) = terms.binary_search_by_key(&term_ord, |term| term.term_ord) else {
            return Ok(None);
        };
        let cached = &terms[index];
        if cached.term_metadata != term_metadata {
            return Err("cached rank-pruning term metadata disagrees with TERMDICT");
        }
        Ok(Some(Arc::clone(&cached.pruning)))
    }

    fn insert(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
        pruning: Arc<ValidatedTermPruningMetadata>,
    ) -> Result<Arc<ValidatedTermPruningMetadata>, &'static str> {
        loop {
            let current = self.terms.load_full();
            match current.binary_search_by_key(&term_ord, |term| term.term_ord) {
                Ok(index) => {
                    let cached = &current[index];
                    if cached.term_metadata != term_metadata {
                        return Err("concurrent rank-pruning cache entry disagrees with TERMDICT");
                    }
                    return Ok(Arc::clone(&cached.pruning));
                }
                Err(insertion) => {
                    let retained_bytes = current.iter().fold(0_usize, |bytes, term| {
                        bytes.saturating_add(term.pruning.heap_bytes())
                    });
                    if current.len() >= MAX_RANK_PRUNING_CACHE_TERMS
                        || pruning.heap_bytes()
                            > MAX_RANK_PRUNING_CACHE_PAYLOAD_BYTES.saturating_sub(retained_bytes)
                    {
                        return Ok(pruning);
                    }
                    let mut next = Vec::new();
                    next.try_reserve_exact(current.len().saturating_add(1))
                        .map_err(|_| "could not allocate rank-pruning cache sidecar")?;
                    next.extend(current.iter().cloned());
                    next.insert(
                        insertion,
                        CachedRankPruningTerm {
                            term_ord,
                            term_metadata,
                            pruning: Arc::clone(&pruning),
                        },
                    );
                    let next = Arc::new(next);
                    let previous = self.terms.compare_and_swap(&current, next);
                    if Arc::ptr_eq(&current, &previous) {
                        return Ok(pruning);
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.terms.load().len()
    }
}

/// One immutable mapped or owned segment admitted by a recovered snapshot.
///
/// Structural framing, MANIFEST witnesses, and the IDMAP-bound IDHASH identity
/// pair are checked during open or owned publication. All unrelated payload
/// hashes remain lazy and are checked on first access.
#[derive(Clone)]
pub struct RecoveredSegment {
    path: PathBuf,
    manifest: ManifestSegment,
    reader: Arc<RecoveredSegmentBacking>,
    tombstone_index: TombstoneIndex,
    id_lookup: IdHashLookupPlan,
    live_doc_count: u32,
    rank_pruning_cache: Arc<RankPruningCache>,
}

impl RecoveredSegment {
    fn bind(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: SegmentReader<ReadOnlyMappedFile>,
    ) -> Result<Self, KeeperError> {
        Self::bind_backing(path, manifest, RecoveredSegmentBacking::Mapped(reader))
    }

    fn bind_owned(
        path: PathBuf,
        manifest: ManifestSegment,
        encoded: EncodedSegment,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let reader = SegmentReader::from_owned(encoded.into_bytes(), schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        validate_segment_witnesses(&path, &manifest, &reader)?;
        Self::bind_backing(path, manifest, RecoveredSegmentBacking::Owned(reader))
    }

    fn bind_backing(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: RecoveredSegmentBacking,
    ) -> Result<Self, KeeperError> {
        Self::bind_shared(
            path,
            manifest,
            Arc::new(reader),
            Arc::new(RankPruningCache::new()),
        )
    }

    fn bind_shared(
        path: PathBuf,
        manifest: ManifestSegment,
        reader: Arc<RecoveredSegmentBacking>,
        rank_pruning_cache: Arc<RankPruningCache>,
    ) -> Result<Self, KeeperError> {
        let id_map_bytes = required_identity_section(&path, &reader, SectionKind::IDMAP)?;
        let id_map = IdMapSection::parse(id_map_bytes, manifest.docid_lo, manifest.docid_hi)
            .map_err(|source| KeeperError::IdMapCorrupted {
                path: path.clone(),
                source,
            })?;
        if u64::try_from(id_map.present_count()).unwrap_or(u64::MAX)
            != u64::from(manifest.doc_count)
        {
            return Err(KeeperError::AtSealDocCountMismatch {
                path,
                manifest: manifest.doc_count,
                id_map: id_map.present_count(),
            });
        }
        let tombstone_hole =
            first_tombstone_hole(&manifest.tombstones, id_map).map_err(|source| {
                KeeperError::ManifestCorrupted {
                    path: path.clone(),
                    source,
                }
            })?;
        if let Some(global_docid) = tombstone_hole {
            return Err(KeeperError::TombstoneReferencesHole {
                path,
                segment_id: manifest.segment_id,
                global_docid,
            });
        }
        let id_hash_bytes = required_identity_section(&path, &reader, SectionKind::IDHASH)?;
        let id_hash = IdHashSection::parse(id_hash_bytes, id_map).map_err(|source| {
            KeeperError::IdHashCorrupted {
                path: path.clone(),
                source,
            }
        })?;
        let id_lookup = id_hash.lookup_plan();
        let tombstone_index = TombstoneIndex::build(&manifest.tombstones).map_err(|source| {
            KeeperError::ManifestCorrupted {
                path: path.clone(),
                source,
            }
        })?;
        let live_doc_count = manifest.live_doc_count();
        Ok(Self {
            path,
            manifest,
            reader,
            tombstone_index,
            id_lookup,
            live_doc_count,
            rank_pruning_cache,
        })
    }

    fn rebind(&self, manifest: ManifestSegment) -> Result<Self, KeeperError> {
        self.reader.validate_witnesses(&self.path, &manifest)?;
        Self::bind_shared(
            self.path.clone(),
            manifest,
            Arc::clone(&self.reader),
            Arc::clone(&self.rank_pruning_cache),
        )
    }

    pub(crate) fn cached_rank_pruning_metadata(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
    ) -> Result<Option<Arc<ValidatedTermPruningMetadata>>, &'static str> {
        self.rank_pruning_cache.get(term_ord, term_metadata)
    }

    pub(crate) fn cache_rank_pruning_metadata(
        &self,
        term_ord: u32,
        term_metadata: TermMetadata,
        pruning: Arc<ValidatedTermPruningMetadata>,
    ) -> Result<Arc<ValidatedTermPruningMetadata>, &'static str> {
        self.rank_pruning_cache
            .insert(term_ord, term_metadata, pruning)
    }

    #[cfg(test)]
    pub(crate) fn cached_rank_pruning_term_count(&self) -> usize {
        self.rank_pruning_cache.len()
    }

    /// Canonical published path, or a stable synthetic label for owned bytes.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Immutable MANIFEST metadata for this segment.
    #[must_use]
    pub const fn manifest(&self) -> &ManifestSegment {
        &self.manifest
    }

    /// At-seal physical row count used by BM25 statistics.
    #[must_use]
    pub const fn at_seal_doc_count(&self) -> u32 {
        self.manifest.doc_count
    }

    /// Current MANIFEST tombstone cardinality.
    #[must_use]
    pub fn tombstone_count(&self) -> u64 {
        self.manifest.tombstones.cardinality()
    }

    /// Current live row count after tombstones.
    #[must_use]
    pub const fn doc_count(&self) -> u32 {
        self.live_doc_count
    }

    /// Whether this segment's current MANIFEST state hides one document id.
    #[must_use]
    pub fn is_tombstoned(&self, global_docid: u32) -> bool {
        self.tombstone_index
            .contains(self.manifest.tombstones.as_bytes(), global_docid)
    }

    /// Materialize one live external identifier from the cached IDMAP layout.
    ///
    /// The lookup touches only this row's two offsets and identifier bytes. It
    /// returns `None` for an out-of-range id, an IDMAP hole, or a tombstoned row.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        if !(self.manifest.docid_lo..self.manifest.docid_hi).contains(&u64::from(global_docid))
            || self.is_tombstoned(global_docid)
        {
            return None;
        }
        let id_map = self.reader.section(SectionKind::IDMAP).ok().flatten()?;
        self.id_lookup
            .materialize_global_docid(id_map, global_docid)
    }

    fn contains_identity_row(&self, global_docid: u32) -> bool {
        self.reader
            .section(SectionKind::IDMAP)
            .ok()
            .flatten()
            .is_some_and(|id_map| self.id_lookup.contains_global_docid(id_map, global_docid))
    }

    fn lookup_document_id(&self, document_id: &str) -> Result<Option<(u32, u64)>, KeeperError> {
        let id_map = required_identity_section(&self.path, &self.reader, SectionKind::IDMAP)?;
        let id_hash = required_identity_section(&self.path, &self.reader, SectionKind::IDHASH)?;
        self.id_lookup
            .lookup_with_content_hash(id_map, id_hash, document_id)
            .map(|(global_docid, content_hash)| {
                u32::try_from(global_docid)
                    .map(|global_docid| (global_docid, content_hash))
                    .map_err(|_| KeeperError::SegmentMetadataMismatch {
                        path: self.path.clone(),
                        detail: format!("IDHASH returned non-u32 global docid {global_docid}"),
                    })
            })
            .transpose()
    }

    /// Structurally validated FSLX header.
    #[must_use]
    pub fn header(&self) -> SegmentHeader {
        self.reader.header()
    }

    /// Verify and borrow one section payload on first access.
    ///
    /// # Errors
    ///
    /// Returns a typed segment-corruption error when the payload checksum does
    /// not match its section-table witness.
    pub fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        self.reader.section(kind)
    }

    /// Validated section table, including unknown optional extensions.
    #[must_use]
    pub fn section_entries(&self) -> &[SectionEntry] {
        self.reader.section_entries()
    }

    /// Borrow the complete immutable FSLX image for byte-level verification.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        self.reader.source_bytes()
    }

    /// Eagerly recompute every segment checksum for doctor-style verification.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub fn verify(&self) -> Result<(), QuillError> {
        self.reader.verify()
    }
}

impl crate::argus::LiveDocs for RecoveredSegment {
    fn is_live(&self, global_docid: u32) -> bool {
        (self.manifest.docid_lo..self.manifest.docid_hi).contains(&u64::from(global_docid))
            && self.contains_identity_row(global_docid)
            && !self.is_tombstoned(global_docid)
    }
}

/// One retained FSLX artifact omitted from the active MANIFEST after
/// durability recovery proved it unrepairable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantinedSegment {
    /// Stable identifier encoded in the canonical segment file name.
    pub segment_id: u64,
    /// Retained path ending in `.quarantine`.
    pub path: PathBuf,
    /// At-seal row count from the most recent retained MANIFEST witness.
    ///
    /// Older quarantines can outlive the two-slot MANIFEST history, in which
    /// case the estimate is intentionally unknown rather than guessed.
    pub estimated_missing_docs: Option<u64>,
}

/// A read-only, internally consistent Keeper snapshot.
///
/// `open` performs recovery by selecting the admitted MANIFEST slot and
/// validating every referenced segment. It never repairs, renames, or removes
/// filesystem entries. Keeping this value alive retains every immutable mapped
/// or owned byte source as well.
#[derive(Clone)]
pub struct KeeperSnapshot {
    directory: Option<PathBuf>,
    schema: SchemaDescriptor,
    loaded: LoadedManifest,
    segments: Vec<RecoveredSegment>,
    segments_by_seal_desc: Vec<usize>,
    at_seal_doc_count: u64,
    tombstone_count: u64,
    live_doc_count: u64,
    quarantined_segments: Vec<QuarantinedSegment>,
}

impl KeeperSnapshot {
    fn from_parts(
        directory: Option<PathBuf>,
        schema: SchemaDescriptor,
        loaded: LoadedManifest,
        segments: Vec<RecoveredSegment>,
    ) -> Result<Self, KeeperError> {
        let error_path = directory
            .clone()
            .unwrap_or_else(|| PathBuf::from("<in-memory>"));
        let mut segments_by_seal_desc = Vec::new();
        segments_by_seal_desc
            .try_reserve_exact(segments.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate segment probe order",
                path: error_path,
                source: io::Error::other(error.to_string()),
            })?;
        segments_by_seal_desc.extend(0..segments.len());
        segments_by_seal_desc
            .sort_unstable_by_key(|&index| std::cmp::Reverse(segments[index].manifest.seal_seq));

        let at_seal_doc_count = segments
            .iter()
            .map(|segment| u64::from(segment.at_seal_doc_count()))
            .sum();
        let tombstone_count = segments.iter().map(RecoveredSegment::tombstone_count).sum();
        let live_doc_count = segments
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let quarantined_segments = directory
            .as_deref()
            .map(|directory| discover_quarantined_segments(directory, &loaded.manifest))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            directory,
            schema,
            loaded,
            segments,
            segments_by_seal_desc,
            at_seal_doc_count,
            tombstone_count,
            live_doc_count,
            quarantined_segments,
        })
    }

    /// Open one durable snapshot without mutating the index directory.
    ///
    /// # Errors
    ///
    /// Returns typed not-found, manifest, schema, segment, or metadata errors.
    pub fn open(
        directory: impl AsRef<Path>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let directory = directory.as_ref();
        match Self::open_once(directory, schema) {
            Err(error) if recovery_retryable(&error) => Self::open_once(directory, schema),
            result => result,
        }
    }

    fn open_once(directory: &Path, schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        let expected_schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        let loaded = load_manifest_pair(directory)?;
        validate_loaded_schema(directory, expected_schema_id, &loaded)?;
        validate_recovery_claims(directory, &loaded)?;

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(loaded.manifest.segments.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate recovered segment table",
                path: directory.to_path_buf(),
                source: io::Error::other(error.to_string()),
            })?;
        for manifest_segment in &loaded.manifest.segments {
            let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
            let reader = SegmentReader::open_published(&path, schema).map_err(|source| {
                KeeperError::SegmentOpen {
                    path: path.clone(),
                    source,
                }
            })?;
            validate_segment_witnesses(&path, manifest_segment, &reader)?;
            segments.push(RecoveredSegment::bind(
                path,
                manifest_segment.clone(),
                reader,
            )?);
        }

        let snapshot = Self::from_parts(Some(directory.to_path_buf()), schema, loaded, segments)?;
        if !snapshot.quarantined_segments.is_empty() {
            tracing::warn!(
                target: crate::tracing_conventions::TARGET,
                event = "quill.keeper.quarantine_persisted",
                directory = %directory.display(),
                quarantined_segments = snapshot.quarantined_segments.len(),
                estimated_missing_docs = snapshot.estimated_missing_docs(),
                unknown_missing_doc_segments = snapshot
                    .quarantined_segments
                    .iter()
                    .filter(|segment| segment.estimated_missing_docs.is_none())
                    .count(),
                "Quill opened with retained quarantined segments; results must be surfaced as degraded"
            );
        }
        Ok(snapshot)
    }

    /// Create a genesis index or open an existing index with the same schema.
    ///
    /// Creation is a writer operation: it acquires the cross-process admission,
    /// publishes genesis through an `O_EXCL` generation claim, and releases the
    /// writer capability before returning this read-only snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, I/O, schema mismatch, publication, or open
    /// errors. Existing durable state is never rebuilt on mismatch.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let writer = KeeperWriter::create(cx, directory, schema).await?;
        Ok(writer.snapshot.clone())
    }

    /// Acquire the sole cross-process writer capability for this index.
    ///
    /// Reader [`Self::open`] remains lock-free and filesystem-touchless.
    /// Writer open acquires `LOCK` before interrupted-publish recovery and GC.
    ///
    /// # Errors
    ///
    /// Returns a typed busy, corrupt-lock, recovery, schema, or I/O failure.
    pub async fn open_writer(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<KeeperWriter, KeeperError> {
        KeeperWriter::open(cx, directory, schema).await
    }

    /// Construct an owned-buffer genesis snapshot without creating files.
    ///
    /// [`Self::publish_owned_segments`] attaches committed owned FSLX bytes;
    /// recovery and garbage collection remain no-ops for this backend.
    ///
    /// # Errors
    ///
    /// Returns an invalid-schema error when the descriptor is not canonical.
    pub fn in_memory(schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        Self::from_parts(
            None,
            schema,
            LoadedManifest {
                manifest: Manifest::empty(1, schema_id, 0),
                source: ManifestSource::InMemory,
            },
            Vec::new(),
        )
    }

    /// Publish owned FSLX segments into one immutable in-memory successor.
    ///
    /// `proposed` must be the exact next MANIFEST generation. Retained
    /// segments reuse their immutable byte backing while adopting only the
    /// successor's validated tombstone state. Every newly referenced segment
    /// must have exactly one matching `EncodedSegment`; unreferenced or
    /// duplicate inputs are rejected. The current snapshot is never changed,
    /// so encoded bytes remain invisible until this method returns the fully
    /// validated successor.
    ///
    /// # Errors
    ///
    /// Returns a typed backend, MANIFEST-transition, segment-witness, or
    /// identity-section error. Durable snapshots must publish through
    /// [`KeeperWriter`] instead.
    pub fn publish_owned_segments(
        &self,
        proposed: &Manifest,
        encoded_segments: Vec<EncodedSegment>,
    ) -> Result<Self, KeeperError> {
        if self.directory.is_some() || self.loaded.source != ManifestSource::InMemory {
            return Err(KeeperError::InvalidTransition {
                detail: "owned segment publication requires an in-memory Keeper snapshot"
                    .to_owned(),
            });
        }
        proposed
            .validate()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        validate_manifest_successor(&self.loaded.manifest, proposed)?;

        let mut owned_by_id = Vec::new();
        owned_by_id
            .try_reserve_exact(encoded_segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("owned segment inventory allocation failed: {error}"),
            })?;
        for encoded in encoded_segments {
            owned_by_id.push((encoded.header().segment_id, Some(encoded)));
        }
        owned_by_id.sort_unstable_by_key(|entry| entry.0);
        if let Some(duplicate) = owned_by_id
            .windows(2)
            .find(|pair| pair[0].0 == pair[1].0)
            .map(|pair| pair[0].0)
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!("owned segment {duplicate:#018x} was supplied more than once"),
            });
        }

        let mut current_by_id = Vec::new();
        current_by_id
            .try_reserve_exact(self.segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("current segment inventory allocation failed: {error}"),
            })?;
        current_by_id.extend(0..self.segments.len());
        current_by_id.sort_unstable_by_key(|&index| self.segments[index].manifest.segment_id);

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(proposed.segments.len())
            .map_err(|error| KeeperError::InvalidTransition {
                detail: format!("successor segment inventory allocation failed: {error}"),
            })?;
        for manifest_segment in &proposed.segments {
            let segment_id = manifest_segment.segment_id;
            let current = current_by_id
                .binary_search_by_key(&segment_id, |&index| {
                    self.segments[index].manifest.segment_id
                })
                .ok()
                .map(|index| current_by_id[index]);
            let supplied = owned_by_id
                .binary_search_by_key(&segment_id, |entry| entry.0)
                .ok();

            if let Some(current) = current {
                if supplied.is_some() {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "retained segment {segment_id:#018x} also supplied replacement bytes"
                        ),
                    });
                }
                segments.push(self.segments[current].rebind(manifest_segment.clone())?);
                continue;
            }

            let supplied = supplied.ok_or_else(|| KeeperError::InvalidTransition {
                detail: format!(
                    "new manifest segment {segment_id:#018x} has no supplied owned bytes"
                ),
            })?;
            let encoded =
                owned_by_id[supplied]
                    .1
                    .take()
                    .ok_or_else(|| KeeperError::InvalidTransition {
                        detail: format!(
                            "owned segment {segment_id:#018x} was consumed more than once"
                        ),
                    })?;
            let path = PathBuf::from("<in-memory>").join(canonical_segment_name(segment_id));
            segments.push(RecoveredSegment::bind_owned(
                path,
                manifest_segment.clone(),
                encoded,
                self.schema,
            )?);
        }

        if let Some(unreferenced) = owned_by_id
            .iter()
            .find_map(|(segment_id, encoded)| encoded.is_some().then_some(*segment_id))
        {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "supplied owned segment {unreferenced:#018x} is absent from the successor manifest"
                ),
            });
        }

        Self::from_parts(
            None,
            self.schema,
            LoadedManifest {
                manifest: proposed.clone(),
                source: ManifestSource::InMemory,
            },
            segments,
        )
    }

    /// Replace one exact caller-ordered manifest run with a Q1 concat merge.
    ///
    /// This owned-buffer entry point is primarily useful for deterministic
    /// testing and offline assembly. Durable indexes must use
    /// [`KeeperWriter::concat_merge`], which runs construction on the runtime's
    /// blocking lane and preserves writer admission through publication.
    ///
    /// # Errors
    ///
    /// Rejects fewer than two sources, any list that is not one uninterrupted
    /// slice of the current manifest, output-id collision, unknown extension
    /// sections, corrupt source codecs, duplicate-live external identifiers,
    /// or an invalid successor publication.
    pub fn concat_merge_owned(
        &self,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Self, KeeperError> {
        let artifact =
            build_concat_merge(self, source_segment_ids, output_segment_id, created_unix_s)?;
        self.publish_owned_segments(&artifact.manifest, vec![artifact.encoded])
    }

    /// Fold every segment above `policy` into a new immutable snapshot.
    ///
    /// This owned-buffer entry point is useful for deterministic tests and
    /// in-memory indexes. Durable indexes use [`KeeperWriter::compact`] so the
    /// replacement files are synced while still unreachable, followed by one
    /// atomic MANIFEST publication.
    ///
    /// # Errors
    ///
    /// Rejects an invalid policy, corrupt or unsupported source sections,
    /// rewrite/resource failures, or an invalid successor publication.
    pub fn compact_owned(
        &self,
        policy: CompactionPolicy,
        created_unix_s: i64,
    ) -> Result<(Self, CompactionReport), KeeperError> {
        let artifact = build_compaction(self, policy, created_unix_s)?;
        if !artifact.report.changed() {
            return Ok((self.clone(), artifact.report));
        }
        let snapshot = self.publish_owned_segments(&artifact.manifest, artifact.encoded)?;
        Ok((snapshot, artifact.report))
    }

    /// Durable index directory, or `None` for an in-memory snapshot.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Compile-time schema admitted by this snapshot.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Selected MANIFEST and its recovery provenance.
    #[must_use]
    pub const fn loaded_manifest(&self) -> &LoadedManifest {
        &self.loaded
    }

    /// Retained corrupt artifacts omitted from the active MANIFEST.
    #[must_use]
    pub fn quarantined_segments(&self) -> &[QuarantinedSegment] {
        &self.quarantined_segments
    }

    /// Whether this snapshot must be surfaced as degraded.
    #[must_use]
    pub const fn is_degraded(&self) -> bool {
        !self.quarantined_segments.is_empty()
    }

    /// Saturating sum of known at-seal rows from quarantined segments.
    ///
    /// Unknown historical rows are excluded and remain visible through
    /// [`Self::quarantined_segments`].
    #[must_use]
    pub fn estimated_missing_docs(&self) -> u64 {
        self.quarantined_segments
            .iter()
            .filter_map(|segment| segment.estimated_missing_docs)
            .fold(0_u64, u64::saturating_add)
    }

    /// Referenced immutable segments in ascending Q1 range order.
    #[must_use]
    pub fn segments(&self) -> &[RecoveredSegment] {
        &self.segments
    }

    /// Physical at-seal rows retained for BM25 statistics until compaction.
    #[must_use]
    pub const fn at_seal_doc_count(&self) -> u64 {
        self.at_seal_doc_count
    }

    /// Tombstones paired with this immutable snapshot generation.
    #[must_use]
    pub const fn tombstone_count(&self) -> u64 {
        self.tombstone_count
    }

    /// Public live document count, cached during snapshot construction.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.live_doc_count
    }

    /// Whether one global document id is visible in this snapshot.
    #[must_use]
    pub fn is_live(&self, global_docid: u32) -> bool {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.manifest.docid_lo <= global_docid_u64);
        let Some(segment) = insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
        else {
            return false;
        };
        crate::argus::LiveDocs::is_live(segment, global_docid)
    }

    /// Materialize one live winner's external identifier via its IDMAP slice.
    ///
    /// Segment selection is logarithmic in the manifest segment count; the
    /// selected segment performs constant-time checked offset reads without
    /// reparsing the IDMAP span.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.manifest.docid_lo <= global_docid_u64);
        insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
            .and_then(|segment| segment.materialize_document_id(global_docid))
    }

    /// Resolve an external identifier against current sealed segments.
    ///
    /// Probes descend by `seal_seq`, continue after tombstoned hits, and keep
    /// scanning after the first live hit so duplicate-live upsert corruption is
    /// diagnosed rather than silently masked.
    ///
    /// # Errors
    ///
    /// Returns a typed identity-section or multiple-live-row failure.
    pub fn resolve_document_id(
        &self,
        document_id: &str,
    ) -> Result<Option<ResolvedDocumentId>, KeeperError> {
        self.resolve_document_id_in(&self.loaded.manifest, document_id)
    }

    /// Clone exactly one next-generation MANIFEST for staged writer changes.
    ///
    /// # Errors
    ///
    /// Returns a typed generation-exhaustion failure.
    pub fn next_manifest(&self) -> Result<Manifest, KeeperError> {
        let mut manifest = self.loaded.manifest.clone();
        manifest.generation =
            manifest
                .generation
                .checked_add(1)
                .ok_or(KeeperError::GenerationExhausted {
                    current: manifest.generation,
                })?;
        Ok(manifest)
    }

    /// Stage an idempotent sealed-document deletion into one next MANIFEST.
    ///
    /// Missing and already-deleted identifiers return `false` without changing
    /// `proposed`. Multiple calls compose in the same generation; publication
    /// remains the caller's explicit [`KeeperWriter::publish`] operation.
    ///
    /// # Errors
    ///
    /// Rejects a non-successor proposal, corrupt identity state, tombstone-on-
    /// hole attempts, allocation failure, or the multiple-live upsert invariant.
    pub fn delete_document(
        &self,
        proposed: &mut Manifest,
        document_id: &str,
    ) -> Result<bool, KeeperError> {
        validate_staged_manifest(&self.loaded.manifest, proposed)?;
        if let Some(segment) = proposed.segments.iter().find(|segment| {
            self.loaded
                .manifest
                .segments
                .binary_search_by_key(&segment.docid_lo, |current| current.docid_lo)
                .map_or(true, |index| {
                    self.loaded.manifest.segments[index].segment_id != segment.segment_id
                })
        }) {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "stage deletes before adding replacement segment {:#018x}",
                    segment.segment_id
                ),
            });
        }
        let Some(resolved) = self.resolve_document_id_in(proposed, document_id)? else {
            return Ok(false);
        };
        let mut staged = proposed.clone();
        let segment = staged
            .segments
            .iter_mut()
            .find(|segment| segment.segment_id == resolved.segment_id)
            .ok_or_else(|| KeeperError::InvalidTransition {
                detail: format!(
                    "resolved segment {:#018x} is absent from staged manifest",
                    resolved.segment_id
                ),
            })?;
        let before = segment.tombstones.cardinality();
        if !segment
            .insert_tombstone(resolved.global_docid)
            .map_err(|source| KeeperError::InvalidManifest { source })?
        {
            return Ok(false);
        }
        validate_staged_manifest(&self.loaded.manifest, &staged)?;
        tracing::trace!(
            target: crate::tracing_conventions::TARGET,
            phase = "keeper.delete_document",
            generation = staged.generation,
            segment_id = resolved.segment_id,
            seal_seq = resolved.seal_seq,
            tombstones_before = before,
            tombstones_after = before + 1,
            "staged sealed-document tombstone"
        );
        *proposed = staged;
        Ok(true)
    }

    /// Stage a valid empty generation while preserving identity and watermark.
    ///
    /// This does not remove segment files; ordinary two-slot publication,
    /// snapshot lifetime, and writer-locked grace-window GC retain ownership of
    /// physical reclamation.
    ///
    /// # Errors
    ///
    /// Rejects a proposal that is not this snapshot's next generation.
    pub fn delete_all(&self, proposed: &mut Manifest) -> Result<(), KeeperError> {
        validate_staged_manifest(&self.loaded.manifest, proposed)?;
        let mut staged = proposed.clone();
        staged.segments.clear();
        staged.field_stats.clear();
        validate_staged_manifest(&self.loaded.manifest, &staged)?;
        *proposed = staged;
        Ok(())
    }

    fn resolve_document_id_in(
        &self,
        manifest: &Manifest,
        document_id: &str,
    ) -> Result<Option<ResolvedDocumentId>, KeeperError> {
        let mut live: Option<ResolvedDocumentId> = None;
        for &index in &self.segments_by_seal_desc {
            let segment = &self.segments[index];
            let Ok(manifest_index) = manifest
                .segments
                .binary_search_by_key(&segment.manifest.docid_lo, |candidate| candidate.docid_lo)
            else {
                continue;
            };
            let manifest_segment = &manifest.segments[manifest_index];
            if manifest_segment.segment_id != segment.manifest.segment_id {
                return Err(KeeperError::InvalidTransition {
                    detail: format!(
                        "staged segment range [{}, {}) changed identity from {:#018x} to {:#018x}",
                        segment.manifest.docid_lo,
                        segment.manifest.docid_hi,
                        segment.manifest.segment_id,
                        manifest_segment.segment_id
                    ),
                });
            }
            let Some((global_docid, content_hash)) = segment.lookup_document_id(document_id)?
            else {
                tracing::trace!(
                    target: crate::tracing_conventions::TARGET,
                    phase = "keeper.idhash_probe",
                    generation = manifest.generation,
                    segment_id = segment.manifest.segment_id,
                    seal_seq = segment.manifest.seal_seq,
                    query_len = document_id.len(),
                    outcome = "miss",
                    "probed sealed-segment identity"
                );
                continue;
            };
            if manifest_segment.tombstones.contains(global_docid) {
                tracing::trace!(
                    target: crate::tracing_conventions::TARGET,
                    phase = "keeper.idhash_probe",
                    generation = manifest.generation,
                    segment_id = segment.manifest.segment_id,
                    seal_seq = segment.manifest.seal_seq,
                    query_len = document_id.len(),
                    outcome = "tombstoned",
                    "continued after tombstoned identity hit"
                );
                continue;
            }
            let resolved = ResolvedDocumentId {
                segment_id: segment.manifest.segment_id,
                seal_seq: segment.manifest.seal_seq,
                global_docid,
                content_hash,
            };
            if let Some(first) = live {
                return Err(KeeperError::MultipleLiveDocumentIds {
                    directory: self
                        .directory
                        .clone()
                        .unwrap_or_else(|| PathBuf::from("<in-memory>")),
                    first_global_docid: first.global_docid,
                    duplicate_global_docid: global_docid,
                });
            }
            tracing::trace!(
                target: crate::tracing_conventions::TARGET,
                phase = "keeper.idhash_probe",
                generation = manifest.generation,
                segment_id = segment.manifest.segment_id,
                seal_seq = segment.manifest.seal_seq,
                query_len = document_id.len(),
                outcome = "live",
                "resolved live sealed-document identity"
            );
            live = Some(resolved);
        }
        Ok(live)
    }
}

impl crate::argus::LiveDocs for KeeperSnapshot {
    fn is_live(&self, global_docid: u32) -> bool {
        self.is_live(global_docid)
    }
}

impl crate::argus::LiveDocs for Manifest {
    fn is_live(&self, global_docid: u32) -> bool {
        let global_docid_u64 = u64::from(global_docid);
        let insertion = self
            .segments
            .partition_point(|segment| segment.docid_lo <= global_docid_u64);
        insertion
            .checked_sub(1)
            .and_then(|index| self.segments.get(index))
            .is_some_and(|segment| segment.is_live(global_docid))
    }
}

fn validate_staged_manifest(previous: &Manifest, proposed: &Manifest) -> Result<(), KeeperError> {
    proposed
        .validate()
        .map_err(|source| KeeperError::InvalidManifest { source })?;
    validate_manifest_successor(previous, proposed)
}

fn manifest_matches_proposal(installed: &Manifest, proposed: &Manifest) -> bool {
    if proposed.last_publish_unix_s == 0 {
        if installed.last_publish_unix_s <= 0 {
            return false;
        }
        let mut normalized = installed.clone();
        normalized.last_publish_unix_s = 0;
        &normalized == proposed
    } else {
        installed == proposed
    }
}

/// Durability recovery policy for a segment whose source and repair sidecar
/// cannot produce one validated FSLX image.
#[cfg(feature = "durability")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnrepairableSegmentPolicy {
    /// Preserve the existing hard error.
    FailClosed,
    /// Retain the corrupt artifact under a `.quarantine` name and publish a
    /// degraded successor MANIFEST that omits it.
    Quarantine,
}

#[derive(Clone)]
enum WriterProtection {
    Disabled,
    #[cfg(feature = "durability")]
    Enabled {
        protector: FileProtector,
        unrepairable: UnrepairableSegmentPolicy,
    },
}

/// Sole cross-process mutation capability for one Quill index directory.
///
/// This type is intentionally not `Clone`. Its internal admission token may be
/// retained by an in-flight blocking publication so cancellation cannot release
/// the OS lock while filesystem mutation is still running.
///
/// Every process that mutates the index directory must honor Quill's `LOCK`.
/// Crash consistency does not cover an out-of-band process replacing admitted
/// directory entries while a writer operation is in flight.
pub struct KeeperWriter {
    admission: Arc<WriterAdmissionInner>,
    snapshot: KeeperSnapshot,
    garbage_options: GarbageCollectionOptions,
    protection: WriterProtection,
}

impl KeeperWriter {
    /// Open an existing index for mutation with the default GC grace period.
    ///
    /// # Errors
    ///
    /// Returns a typed admission, recovery, schema, segment, or I/O failure.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            false,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    /// Create a genesis index or open an existing one for mutation.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open`] plus directory creation or
    /// genesis-publication failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            true,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    /// Open an existing index and activate writer-only FEC recovery.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open`] plus typed durability
    /// decode, validation, or staged-install failures.
    #[cfg(feature = "durability")]
    pub async fn open_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
    ) -> Result<Self, KeeperError> {
        Self::open_durable_with_policy(
            cx,
            directory,
            schema,
            protector,
            UnrepairableSegmentPolicy::FailClosed,
        )
        .await
    }

    /// Open an existing index with explicit unrepairable-segment policy.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open_durable`]. Quarantine mode
    /// still fails closed when the source is missing without a retained
    /// quarantine witness, is not a regular file, or cannot be published into
    /// a valid successor MANIFEST.
    #[cfg(feature = "durability")]
    pub async fn open_durable_with_policy(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
        unrepairable: UnrepairableSegmentPolicy,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            false,
            GarbageCollectionOptions::default(),
            WriterProtection::Enabled {
                protector,
                unrepairable,
            },
        )
        .await
    }

    /// Create or open an index with writer-only FEC recovery enabled.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open_durable`] plus genesis
    /// directory or publication failures.
    #[cfg(feature = "durability")]
    pub async fn create_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
    ) -> Result<Self, KeeperError> {
        Self::create_durable_with_policy(
            cx,
            directory,
            schema,
            protector,
            UnrepairableSegmentPolicy::FailClosed,
        )
        .await
    }

    /// Create or open an index with explicit unrepairable-segment policy.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::create_durable`].
    #[cfg(feature = "durability")]
    pub async fn create_durable_with_policy(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
        unrepairable: UnrepairableSegmentPolicy,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            true,
            GarbageCollectionOptions::default(),
            WriterProtection::Enabled {
                protector,
                unrepairable,
            },
        )
        .await
    }

    async fn open_inner(
        cx: &Cx,
        directory: PathBuf,
        schema: SchemaDescriptor,
        create: bool,
        garbage_options: GarbageCollectionOptions,
        protection: WriterProtection,
    ) -> Result<Self, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }
        let mut directory = normalize_publish_directory(directory);
        if create {
            let create_path = directory.clone();
            spawn_blocking(move || {
                std::fs::create_dir_all(&create_path).map_err(|source| KeeperError::Io {
                    operation: "create index directory",
                    path: create_path,
                    source,
                })
            })
            .await?;
            directory = normalize_publish_directory(directory);
        }
        let admission_directory = directory.clone();
        let admission =
            spawn_blocking(move || acquire_writer_admission(&admission_directory)).await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }

        let recovery_admission = Arc::clone(&admission);
        let recovery_protection = protection.clone();
        spawn_blocking(move || {
            recover_writer_directory(&recovery_admission, schema, &recovery_protection)
        })
        .await?;

        match open_snapshot_blocking(directory.clone(), schema).await {
            Ok(_) => {}
            Err(KeeperError::IndexNotFound { .. }) if create => {
                let schema_id = schema
                    .schema_id()
                    .map_err(|source| KeeperError::InvalidSchema { source })?;
                let genesis = Manifest::empty(1, schema_id, 0);
                let claim_admission = Arc::clone(&admission);
                let publisher = ManifestPublisher::new(&directory);
                match &protection {
                    WriterProtection::Disabled => {
                        publisher
                            .publish_with_generation_claim(cx, &genesis, move |_, generation| {
                                GenerationClaimGuard::acquire(claim_admission, generation)
                            })
                            .await?;
                    }
                    #[cfg(feature = "durability")]
                    WriterProtection::Enabled { protector, .. } => {
                        publisher
                            .publish_durable_with_generation_claim(
                                cx,
                                &genesis,
                                protector,
                                move |_, generation| {
                                    GenerationClaimGuard::acquire(claim_admission, generation)
                                },
                            )
                            .await?;
                    }
                }
                open_snapshot_blocking(directory.clone(), schema).await?;
            }
            Err(error) => return Err(error),
        }

        let gc_admission = Arc::clone(&admission);
        let gc_directory = directory.clone();
        spawn_blocking(move || {
            gc_admission.ensure_directory_identity()?;
            collect_writer_garbage_under_lock(&gc_directory, schema, garbage_options)
        })
        .await?;
        // GC cannot remove reachable bytes, but reopen once so the writer's
        // snapshot is proven from the post-recovery directory state.
        let snapshot = open_snapshot_blocking(directory, schema).await?;
        Ok(Self {
            admission,
            snapshot,
            garbage_options,
            protection,
        })
    }

    /// Current immutable reader view held by this writer.
    #[must_use]
    pub const fn snapshot(&self) -> &KeeperSnapshot {
        &self.snapshot
    }

    /// Publish exactly the next MANIFEST generation through an `O_EXCL` claim.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, claim, transition, durability, or I/O
    /// failures. An ambiguous prior attempt is reconciled to the exact
    /// installed proposal; a differing installed generation remains an error.
    pub async fn publish(
        &mut self,
        cx: &Cx,
        manifest: &Manifest,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        self.admission.ensure_directory_identity()?;
        validate_manifest_shape(manifest, ErrorClass::Invalid)
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest_encoded_len(manifest).map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest
            .validate()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        validate_manifest_successor(&self.snapshot.loaded_manifest().manifest, manifest)?;
        let directory = self.admission.directory.clone();
        let preflight_directory = directory.clone();
        let preflight_manifest = manifest.clone();
        let preflight_protection = self.protection.clone();
        let schema = self.snapshot.schema();
        spawn_blocking(move || {
            validate_proposed_manifest_segments(
                &preflight_directory,
                &preflight_manifest,
                schema,
                &preflight_protection,
            )
        })
        .await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        let claim_admission = Arc::clone(&self.admission);
        let publisher = ManifestPublisher::new(&directory);
        let publish_result = match &self.protection {
            WriterProtection::Disabled => {
                publisher
                    .publish_with_generation_claim(cx, manifest, move |_, generation| {
                        GenerationClaimGuard::acquire(claim_admission, generation)
                    })
                    .await
            }
            #[cfg(feature = "durability")]
            WriterProtection::Enabled { protector, .. } => {
                publisher
                    .publish_durable_with_generation_claim(
                        cx,
                        manifest,
                        protector,
                        move |_, generation| {
                            GenerationClaimGuard::acquire(claim_admission, generation)
                        },
                    )
                    .await
            }
        };
        if let Err(error) = publish_result {
            if self.reconcile_manifest_proposal(cx, manifest).await? {
                return Ok(&self.snapshot);
            }
            return Err(error);
        }
        self.admission.ensure_directory_identity()?;
        self.snapshot = open_snapshot_blocking(directory, self.snapshot.schema()).await?;
        Ok(&self.snapshot)
    }

    async fn reconcile_manifest_proposal(
        &mut self,
        cx: &Cx,
        proposed: &Manifest,
    ) -> Result<bool, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        let schema = self.snapshot.schema();
        let directory = admission.directory.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            recover_writer_directory(&admission, schema, &protection)?;
            sync_directory(&admission.directory)?;
            admission.ensure_directory_identity()
        })
        .await?;
        let recovered = open_snapshot_blocking(directory, schema).await?;
        if !manifest_matches_proposal(&recovered.loaded_manifest().manifest, proposed) {
            return Ok(false);
        }
        self.snapshot = recovered;
        Ok(true)
    }

    /// Atomically install one synced immutable segment while holding `LOCK`.
    ///
    /// # Errors
    ///
    /// Returns a typed path, collision, sidecar, or fsync failure.
    pub async fn publish_segment(
        &mut self,
        cx: &Cx,
        pending: PendingSegmentFile,
    ) -> Result<PathBuf, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            let parent = pending.path().parent().ok_or_else(|| KeeperError::Io {
                operation: "resolve pending segment directory",
                path: pending.path().to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "pending segment path has no parent directory",
                ),
            })?;
            let parent = std::fs::canonicalize(parent).map_err(|source| KeeperError::Io {
                operation: "canonicalize pending segment directory",
                path: parent.to_path_buf(),
                source,
            })?;
            if parent != admission.directory {
                return Err(KeeperError::Io {
                    operation: "verify pending segment directory",
                    path: pending.path().to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "pending segment belongs to a different index directory",
                    ),
                });
            }
            if let Some(published) = reconcile_published_segment(&admission, &protection, &pending)?
            {
                admission.ensure_directory_identity()?;
                return Ok(published);
            }
            let published = match &protection {
                WriterProtection::Disabled => publish_pending_segment(pending),
                #[cfg(feature = "durability")]
                WriterProtection::Enabled { protector, .. } => {
                    publish_pending_segment_durable(pending, protector)
                }
            }?;
            admission.ensure_directory_identity()?;
            Ok(published)
        })
        .await
    }

    /// Reconcile or install retained canonical bytes under one mutation guard.
    ///
    /// Unlike a caller-created [`PendingSegmentFile`], this operation acquires
    /// the process-wide writer guard before inspecting either the canonical
    /// destination or retry temp. A dropped future may leave its blocking
    /// closure running, but a retry cannot race that closure or manufacture a
    /// redundant temp after the exact canonical file already won.
    ///
    /// # Errors
    ///
    /// Returns a typed lock, byte-conflict, segment-write, sidecar, or fsync
    /// failure. Existing differing artifacts are preserved.
    pub(crate) async fn publish_encoded_segment_retryable(
        &self,
        cx: &Cx,
        encoded: Arc<EncodedSegment>,
    ) -> Result<PathBuf, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            if let Some(published) = reconcile_encoded_segment(&admission, &protection, &encoded)? {
                admission.ensure_directory_identity()?;
                return Ok(published);
            }
            let pending = encoded
                .write_temp_retryable(&admission.directory)
                .map_err(|source| KeeperError::SegmentInstall { source })?;
            let published = match &protection {
                WriterProtection::Disabled => publish_pending_segment(pending),
                #[cfg(feature = "durability")]
                WriterProtection::Enabled { protector, .. } => {
                    publish_pending_segment_durable(pending, protector)
                }
            }?;
            admission.ensure_directory_identity()?;
            Ok(published)
        })
        .await
    }

    /// Build, install, and atomically publish one Q1-preserving concat merge.
    ///
    /// Source ids are interpreted in caller order and must name exactly one
    /// uninterrupted slice of the current manifest. Construction runs on the
    /// asupersync blocking lane. The fully synced immutable output remains
    /// unreachable until the exact successor MANIFEST is published; a
    /// cancellation between those two steps can therefore leave only a safe
    /// garbage-collectable orphan.
    ///
    /// # Errors
    ///
    /// Returns typed merge validation/codec failures plus the ordinary segment
    /// installation and reconciled MANIFEST-publication failures.
    pub async fn concat_merge(
        &mut self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(ConcatMergeError::Cancelled.into());
        }
        let mut source_ids = Vec::new();
        source_ids
            .try_reserve_exact(source_segment_ids.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "source ids",
                count: source_segment_ids.len(),
            })?;
        source_ids.extend_from_slice(source_segment_ids);
        let snapshot = self.snapshot.clone();
        let artifact = spawn_blocking(move || {
            build_concat_merge(&snapshot, &source_ids, output_segment_id, created_unix_s)
        })
        .await?;
        if cx.is_cancel_requested() {
            return Err(ConcatMergeError::Cancelled.into());
        }

        let ConcatMergeArtifact { encoded, manifest } = artifact;
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let pending = spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            encoded
                .write_temp_retryable(&admission.directory)
                .map_err(|source| KeeperError::from(ConcatMergeError::Segment { source }))
        })
        .await?;
        self.publish_segment(cx, pending).await?;
        self.publish(cx, &manifest).await
    }

    /// Rewrite every segment above the selected tombstone-density threshold.
    ///
    /// Replacement files are completely built, validated, and synced before
    /// the successor MANIFEST can reference them. Cancellation or a crash in
    /// that window therefore leaves the old generation authoritative and only
    /// unreferenced, grace-period-GC-eligible output files behind.
    ///
    /// # Errors
    ///
    /// Returns typed policy/rewrite failures plus ordinary segment install and
    /// MANIFEST publication failures.
    pub async fn compact(
        &mut self,
        cx: &Cx,
        policy: CompactionPolicy,
        created_unix_s: i64,
    ) -> Result<CompactionReport, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }
        let snapshot = self.snapshot.clone();
        let artifact =
            spawn_blocking(move || build_compaction(&snapshot, policy, created_unix_s)).await?;
        if !artifact.report.changed() {
            return Ok(artifact.report);
        }
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }

        let CompactionArtifact {
            encoded,
            manifest,
            report,
        } = artifact;
        for segment in encoded {
            self.publish_encoded_segment_retryable(cx, Arc::new(segment))
                .await?;
        }
        if cx.is_cancel_requested() {
            return Err(CompactionError::Cancelled.into());
        }
        self.publish(cx, &manifest).await?;
        Ok(report)
    }

    /// Run one grace-window garbage sweep under the held writer admission.
    ///
    /// # Errors
    ///
    /// Returns before removal on any recovery, identity, metadata, or path
    /// validation failure.
    pub async fn collect_garbage(
        &mut self,
        cx: &Cx,
    ) -> Result<GarbageCollectionReport, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let schema = self.snapshot.schema();
        let options = self.garbage_options;
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            collect_writer_garbage_under_lock(&admission.directory, schema, options)
        })
        .await
    }
}

struct ConcatMergeArtifact {
    encoded: EncodedSegment,
    manifest: Manifest,
}

struct CompactionArtifact {
    encoded: Vec<EncodedSegment>,
    manifest: Manifest,
    report: CompactionReport,
}

struct CompactionReplacement {
    encoded: Option<EncodedSegment>,
    manifest: Option<ManifestSegment>,
    source_stats: Vec<FieldStats>,
    replacement_stats: Vec<FieldStats>,
}

struct CompactedTermSections {
    postings: Vec<u8>,
    positions: Vec<u8>,
    blockmax: Vec<u8>,
    terms: Vec<MergedTermRow>,
    total_tokens: BTreeMap<u16, u64>,
}

struct ConcatSource<'a> {
    segment: &'a RecoveredSegment,
    terms: Vec<OwnedTerm>,
    postings: &'a [u8],
    positions: Option<&'a [u8]>,
    blockmax: &'a [u8],
    doclen: DocLenSection<'a>,
    id_map: IdMapSection<'a>,
    numeric: Option<NumericSection<'a>>,
    stored_meta: Option<StoredMetaSection<'a>>,
    stats: StatsSection,
}

fn build_compaction(
    snapshot: &KeeperSnapshot,
    policy: CompactionPolicy,
    created_unix_s: i64,
) -> Result<CompactionArtifact, CompactionError> {
    validate_compaction_policy(policy)?;
    let generation_before = snapshot.loaded.manifest.generation;
    let mut report = CompactionReport {
        generation_before,
        generation_after: generation_before,
        examined_segments: snapshot.segments.len(),
        ..CompactionReport::default()
    };
    let mut eligible = Vec::new();
    eligible
        .try_reserve_exact(snapshot.segments.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "eligible segment indexes",
            count: snapshot.segments.len(),
        })?;
    for (index, segment) in snapshot.segments.iter().enumerate() {
        let physical = segment.manifest.doc_count;
        let deleted = segment.manifest.tombstones.cardinality();
        if physical != 0
            && deleted != 0
            && deleted as f64 / f64::from(physical) > policy.tombstone_density
        {
            eligible.push(index);
        }
    }
    if eligible.is_empty() {
        return Ok(CompactionArtifact {
            encoded: Vec::new(),
            manifest: snapshot.loaded.manifest.clone(),
            report,
        });
    }

    let generation_after =
        generation_before
            .checked_add(1)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest generation",
            })?;
    let mut next_seal_seq = snapshot
        .segments
        .iter()
        .map(|segment| segment.manifest.seal_seq)
        .max()
        .unwrap_or(0);
    let mut used_ids = BTreeSet::new();
    used_ids.extend(
        snapshot
            .segments
            .iter()
            .map(|segment| segment.manifest.segment_id),
    );
    let mut replacements = BTreeMap::new();
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(eligible.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "replacement segments",
            count: eligible.len(),
        })?;
    let mut manifest = snapshot.loaded.manifest.clone();
    manifest.generation = generation_after;
    manifest.last_publish_unix_s = 0;

    for index in eligible {
        let source = &snapshot.segments[index];
        let deleted = source.manifest.tombstones.cardinality();
        let live = u64::from(source.manifest.live_doc_count());
        let output_id = if live == 0 {
            0
        } else {
            derive_compaction_segment_id(
                snapshot,
                source.manifest.segment_id,
                generation_after,
                created_unix_s,
                &mut used_ids,
            )?
        };
        if live != 0 {
            next_seal_seq =
                next_seal_seq
                    .checked_add(1)
                    .ok_or(CompactionError::ArithmeticOverflow {
                        field: "seal sequence",
                    })?;
        }
        let replacement =
            compact_segment(snapshot, source, output_id, next_seal_seq, created_unix_s)?;
        adjust_compaction_field_stats(
            &mut manifest.field_stats,
            &replacement.source_stats,
            &replacement.replacement_stats,
        )?;
        report.compacted_segments += 1;
        report.dropped_documents = report.dropped_documents.checked_add(deleted).ok_or(
            CompactionError::ArithmeticOverflow {
                field: "dropped document count",
            },
        )?;
        report.input_bytes = report
            .input_bytes
            .checked_add(source.manifest.file_len)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "input bytes",
            })?;
        if let Some(output) = replacement.encoded {
            report.output_bytes = report.output_bytes.checked_add(output.file_len()).ok_or(
                CompactionError::ArithmeticOverflow {
                    field: "output bytes",
                },
            )?;
            encoded.push(output);
        } else {
            report.removed_segments += 1;
        }
        replacements.insert(source.manifest.segment_id, replacement.manifest);
    }

    let mut segments = Vec::new();
    segments
        .try_reserve_exact(manifest.segments.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "successor manifest segments",
            count: manifest.segments.len(),
        })?;
    for segment in &manifest.segments {
        match replacements.remove(&segment.segment_id) {
            Some(Some(replacement)) => segments.push(replacement),
            Some(None) => {}
            None => segments.push(segment.clone()),
        }
    }
    manifest.segments = segments;
    manifest
        .validate()
        .map_err(|error| CompactionError::InvalidManifest {
            detail: error.to_string(),
        })?;
    report.generation_after = generation_after;
    Ok(CompactionArtifact {
        encoded,
        manifest,
        report,
    })
}

fn validate_compaction_policy(policy: CompactionPolicy) -> Result<(), CompactionError> {
    if !policy.tombstone_density.is_finite()
        || policy.tombstone_density <= 0.0
        || policy.tombstone_density > 1.0
    {
        return Err(CompactionError::InvalidDensity {
            density: policy.tombstone_density,
        });
    }
    Ok(())
}

fn derive_compaction_segment_id(
    snapshot: &KeeperSnapshot,
    source_id: u64,
    generation: u64,
    created_unix_s: i64,
    used_ids: &mut BTreeSet<u64>,
) -> Result<u64, CompactionError> {
    for salt in 0_u64..=u64::from(u16::MAX) {
        let mut preimage = [0_u8; 44];
        preimage[..8].copy_from_slice(&snapshot.loaded.manifest.schema_id.to_le_bytes());
        preimage[8..16].copy_from_slice(&generation.to_le_bytes());
        preimage[16..24].copy_from_slice(&source_id.to_le_bytes());
        preimage[24..32].copy_from_slice(&created_unix_s.to_le_bytes());
        preimage[32..36].copy_from_slice(&CURRENT_ENGINE_VERSION.to_le_bytes());
        preimage[36..].copy_from_slice(&salt.to_le_bytes());
        let candidate = xxhash_rust::xxh3::xxh3_64(&preimage);
        if used_ids.insert(candidate) {
            return Ok(candidate);
        }
    }
    Err(CompactionError::OutputIdExhausted { source_id })
}

fn adjust_compaction_field_stats(
    manifest: &mut [ManifestFieldStats],
    source: &[FieldStats],
    replacement: &[FieldStats],
) -> Result<(), CompactionError> {
    if source.len() != replacement.len() || source.len() != manifest.len() {
        return Err(CompactionError::InvalidManifest {
            detail: "compaction field-stat sets differ".to_owned(),
        });
    }
    for ((manifest, source), replacement) in manifest.iter_mut().zip(source).zip(replacement) {
        if manifest.field_ord != source.field_ord || source.field_ord != replacement.field_ord {
            return Err(CompactionError::InvalidManifest {
                detail: "compaction field-stat ordinals differ".to_owned(),
            });
        }
        manifest.total_tokens = manifest
            .total_tokens
            .checked_sub(source.total_tokens)
            .and_then(|value| value.checked_add(replacement.total_tokens))
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest total tokens",
            })?;
        manifest.doc_count = manifest
            .doc_count
            .checked_sub(source.doc_count)
            .and_then(|value| value.checked_add(replacement.doc_count))
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "manifest document count",
            })?;
    }
    Ok(())
}

fn compact_segment(
    snapshot: &KeeperSnapshot,
    segment: &RecoveredSegment,
    output_segment_id: u64,
    seal_seq: u64,
    created_unix_s: i64,
) -> Result<CompactionReplacement, CompactionError> {
    if let Some(entry) = segment.section_entries().iter().find(|entry| {
        !(SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw()).contains(&entry.kind.raw())
    }) {
        return Err(CompactionError::UnknownSection {
            segment_id: segment.manifest.segment_id,
            section_kind: entry.kind.raw(),
        });
    }
    segment
        .verify()
        .map_err(|source| CompactionError::Segment { source })?;
    let schema = snapshot.schema();
    let term_fields =
        concat_term_field_ords(schema).map_err(|error| compaction_source_validation(&error))?;
    let stored_fields =
        concat_stored_field_ords(schema).map_err(|error| compaction_source_validation(&error))?;
    let has_positions = concat_schema_has_positions(schema);
    let has_numeric = concat_schema_has_numeric(schema);
    let source = open_concat_source(
        segment,
        schema,
        &term_fields,
        &stored_fields,
        has_positions,
        has_numeric,
    )
    .map_err(|error| compaction_source_validation(&error))?;
    let source_stats = source.stats.rows().to_vec();
    let doc_count = segment.manifest.live_doc_count();
    if doc_count == 0 {
        let replacement_stats = source_stats
            .iter()
            .map(|row| FieldStats::new(row.field_ord, 0, 0))
            .collect();
        return Ok(CompactionReplacement {
            encoded: None,
            manifest: None,
            source_stats,
            replacement_stats,
        });
    }

    let docid_lo = segment.manifest.docid_lo;
    let docid_hi = segment.manifest.docid_hi;
    let span =
        usize::try_from(docid_hi - docid_lo).map_err(|_| CompactionError::ArithmeticOverflow {
            field: "host document span",
        })?;
    let is_live = |global_docid: u64| {
        source.id_map.contains(global_docid)
            && u32::try_from(global_docid)
                .is_ok_and(|docid| !segment.manifest.tombstones.contains(docid))
    };

    let mut doclen_columns = Vec::new();
    doclen_columns
        .try_reserve_exact(term_fields.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "DOCLEN columns",
            count: term_fields.len(),
        })?;
    for field in source.doclen.fields() {
        let mut lengths = Vec::new();
        lengths
            .try_reserve_exact(span)
            .map_err(|_| CompactionError::Allocation {
                resource: "DOCLEN values",
                count: span,
            })?;
        for ordinal in 0..span {
            let global_docid = docid_lo
                .checked_add(u64::try_from(ordinal).map_err(|_| {
                    CompactionError::ArithmeticOverflow {
                        field: "global document ordinal",
                    }
                })?)
                .ok_or(CompactionError::ArithmeticOverflow {
                    field: "global document id",
                })?;
            let length = if is_live(global_docid) {
                field.decoded_fieldnorm(global_docid)
            } else {
                None
            };
            lengths.push(length);
        }
        doclen_columns.push((field.field_ord(), lengths));
    }
    let doclen_inputs = doclen_columns
        .iter()
        .map(|(field_ord, values)| DocLenFieldInput::new(*field_ord, values))
        .collect::<Vec<_>>();
    let doclen = EncodedDocLenSection::encode(docid_lo, docid_hi, &term_fields, &doclen_inputs)
        .map_err(|error| compaction_codec(SectionKind::DOCLEN, error))?;

    let mut id_map_entries = Vec::new();
    id_map_entries
        .try_reserve_exact(span)
        .map_err(|_| CompactionError::Allocation {
            resource: "IDMAP rows",
            count: span,
        })?;
    for ordinal in 0..span {
        let global_docid = docid_lo
            .checked_add(u64::try_from(ordinal).map_err(|_| {
                CompactionError::ArithmeticOverflow {
                    field: "IDMAP ordinal",
                }
            })?)
            .ok_or(CompactionError::ArithmeticOverflow {
                field: "IDMAP global document id",
            })?;
        let entry = is_live(global_docid)
            .then(|| source.id_map.get(global_docid))
            .flatten()
            .map(|entry| IdMapEntryInput::new(entry.document_id(), entry.content_hash()));
        id_map_entries.push(entry);
    }
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &id_map_entries)
        .map_err(|error| compaction_codec(SectionKind::IDMAP, error))?;
    let id_map_view = id_map
        .section()
        .map_err(|error| compaction_codec(SectionKind::IDMAP, error))?;
    let id_hash = EncodedIdHashSection::encode(id_map_view)
        .map_err(|error| compaction_codec(SectionKind::IDHASH, error))?;

    let CompactedTermSections {
        postings,
        positions,
        blockmax,
        terms: compacted_terms,
        total_tokens,
    } = compact_terms(&source, &segment.manifest.tombstones, has_positions)?;
    let mut replacement_stats = Vec::new();
    replacement_stats
        .try_reserve_exact(source_stats.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "STATS rows",
            count: source_stats.len(),
        })?;
    for row in &source_stats {
        replacement_stats.push(FieldStats::new(
            row.field_ord,
            total_tokens.get(&row.field_ord).copied().unwrap_or(0),
            doc_count,
        ));
    }
    let mut term_inputs = Vec::new();
    term_inputs
        .try_reserve_exact(compacted_terms.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "TERMDICT inputs",
            count: compacted_terms.len(),
        })?;
    for term in &compacted_terms {
        term_inputs.push(TermInput::new(term.field_ord, &term.term, term.metadata));
    }
    let termdict = EncodedTermDictionary::encode_sorted(
        schema,
        TermSectionLengths {
            postings: durable_compaction_len(&postings, "POSTINGS")?,
            positions: has_positions
                .then(|| durable_compaction_len(&positions, "POSITIONS"))
                .transpose()?,
            blockmax: durable_compaction_len(&blockmax, "BLOCKMAX")?,
        },
        &term_inputs,
    )
    .map_err(|error| compaction_codec(SectionKind::TERMDICT, error))?;

    let numeric = if let Some(source_numeric) = &source.numeric {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(source_numeric.field_count())
            .map_err(|_| CompactionError::Allocation {
                resource: "NUMERIC columns",
                count: source_numeric.field_count(),
            })?;
        for field in source_numeric.fields() {
            let mut entries = Vec::<NumericEntry>::new();
            entries
                .try_reserve_exact(field.len())
                .map_err(|_| CompactionError::Allocation {
                    resource: "NUMERIC entries",
                    count: field.len(),
                })?;
            entries.extend(
                field
                    .entries()
                    .filter(|entry| !segment.manifest.tombstones.contains(entry.docid())),
            );
            columns.push((field.field_ord(), entries));
        }
        let inputs = columns
            .iter()
            .map(|(field_ord, entries)| NumericFieldInput::new(*field_ord, entries))
            .collect::<Vec<_>>();
        Some(
            EncodedNumericSection::encode(schema, docid_lo, docid_hi, &inputs)
                .map_err(|error| compaction_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };

    let stored_meta = if let Some(source_stored) = &source.stored_meta {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(source_stored.field_count())
            .map_err(|_| CompactionError::Allocation {
                resource: "STOREDMETA columns",
                count: source_stored.field_count(),
            })?;
        for field in source_stored.fields() {
            let mut values = Vec::new();
            values
                .try_reserve_exact(span)
                .map_err(|_| CompactionError::Allocation {
                    resource: "STOREDMETA values",
                    count: span,
                })?;
            for ordinal in 0..span {
                let global_docid = docid_lo
                    .checked_add(u64::try_from(ordinal).map_err(|_| {
                        CompactionError::ArithmeticOverflow {
                            field: "STOREDMETA ordinal",
                        }
                    })?)
                    .ok_or(CompactionError::ArithmeticOverflow {
                        field: "STOREDMETA global document id",
                    })?;
                values.push(
                    is_live(global_docid)
                        .then(|| field.get(global_docid))
                        .flatten(),
                );
            }
            columns.push((field.field_ord(), values));
        }
        let inputs = columns
            .iter()
            .map(|(field_ord, values)| StoredMetaFieldInput::new(*field_ord, values))
            .collect::<Vec<_>>();
        Some(
            EncodedStoredMetaSection::encode(docid_lo, docid_hi, &stored_fields, &inputs)
                .map_err(|error| compaction_codec(SectionKind::STOREDMETA, error))?,
        )
    } else {
        None
    };
    let stats = EncodedStatsSection::encode(&term_fields, &replacement_stats, doc_count)
        .map_err(|error| compaction_codec(SectionKind::STATS, error))?;

    let mut section_plan = Vec::new();
    section_plan
        .try_reserve_exact(10)
        .map_err(|_| CompactionError::Allocation {
            resource: "FSLX section plan",
            count: 10,
        })?;
    section_plan.push(PlannedSection::new(
        SectionKind::TERMDICT,
        termdict.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(SectionKind::POSTINGS, postings.len()));
    if has_positions {
        section_plan.push(PlannedSection::new(SectionKind::POSITIONS, positions.len()));
    }
    section_plan.push(PlannedSection::new(SectionKind::BLOCKMAX, blockmax.len()));
    section_plan.push(PlannedSection::new(
        SectionKind::DOCLEN,
        doclen.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDMAP,
        id_map.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDHASH,
        id_hash.as_bytes().len(),
    ));
    if let Some(numeric) = &numeric {
        section_plan.push(PlannedSection::new(
            SectionKind::NUMERIC,
            numeric.as_bytes().len(),
        ));
    }
    if let Some(stored_meta) = &stored_meta {
        section_plan.push(PlannedSection::new(
            SectionKind::STOREDMETA,
            stored_meta.as_bytes().len(),
        ));
    }
    section_plan.push(PlannedSection::new(
        SectionKind::STATS,
        stats.as_bytes().len(),
    ));
    let mut assembler = SegmentAssembler::new(
        SegmentHeaderInput {
            segment_id: output_segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s,
            engine_version: CURRENT_ENGINE_VERSION,
        },
        &section_plan,
    )
    .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::TERMDICT, termdict.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::POSTINGS, &postings)
        .map_err(|source| CompactionError::Segment { source })?;
    if has_positions {
        assembler
            .copy_section(SectionKind::POSITIONS, &positions)
            .map_err(|source| CompactionError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::BLOCKMAX, &blockmax)
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::DOCLEN, doclen.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::IDMAP, id_map.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    assembler
        .copy_section(SectionKind::IDHASH, id_hash.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    if let Some(numeric) = &numeric {
        assembler
            .copy_section(SectionKind::NUMERIC, numeric.as_bytes())
            .map_err(|source| CompactionError::Segment { source })?;
    }
    if let Some(stored_meta) = &stored_meta {
        assembler
            .copy_section(SectionKind::STOREDMETA, stored_meta.as_bytes())
            .map_err(|source| CompactionError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::STATS, stats.as_bytes())
        .map_err(|source| CompactionError::Segment { source })?;
    let encoded = assembler
        .finish()
        .map_err(|source| CompactionError::Segment { source })?;
    let reader = SegmentReader::from_bytes(encoded.as_bytes(), schema)
        .map_err(|source| CompactionError::Segment { source })?;
    reader
        .verify()
        .map_err(|source| CompactionError::Segment { source })?;
    let manifest = ManifestSegment {
        segment_id: output_segment_id,
        seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo,
        docid_hi,
        doc_count,
        tombstones: TombstoneSet::new(),
    };
    Ok(CompactionReplacement {
        encoded: Some(encoded),
        manifest: Some(manifest),
        source_stats,
        replacement_stats,
    })
}

fn compact_terms(
    source: &ConcatSource<'_>,
    tombstones: &TombstoneSet,
    has_positions: bool,
) -> Result<CompactedTermSections, CompactionError> {
    let mut postings_output = Vec::new();
    let mut positions_output = Vec::new();
    let mut blockmax_output = Vec::new();
    let mut terms = Vec::new();
    let mut total_tokens = BTreeMap::<u16, u64>::new();
    terms
        .try_reserve_exact(source.terms.len())
        .map_err(|_| CompactionError::Allocation {
            resource: "retained term rows",
            count: source.terms.len(),
        })?;

    for term in &source.terms {
        let posting_bytes = concat_span(
            source.postings,
            term.metadata.postings,
            SectionKind::POSTINGS,
        )
        .map_err(|error| compaction_source_validation(&error))?;
        let posting_list = PostingList::parse(posting_bytes, term.metadata.doc_freq)
            .map_err(|error| compaction_codec(SectionKind::POSTINGS, error))?;
        let decoded = posting_list
            .decode_all_bounded(term.metadata.doc_freq as usize)
            .map_err(|error| compaction_codec(SectionKind::POSTINGS, error))?;
        let position_list = if let Some(span) = term.metadata.positions {
            let section = source
                .positions
                .ok_or_else(|| CompactionError::SectionCodec {
                    section: SectionKind::POSITIONS,
                    detail: "term references POSITIONS but the source section is absent".to_owned(),
                })?;
            let bytes = concat_span(section, span, SectionKind::POSITIONS)
                .map_err(|error| compaction_source_validation(&error))?;
            Some(
                PositionList::parse(bytes, &posting_list)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            None
        };
        let mut retained = Vec::<Posting>::new();
        retained
            .try_reserve_exact(decoded.len())
            .map_err(|_| CompactionError::Allocation {
                resource: "retained postings",
                count: decoded.len(),
            })?;
        let mut retained_positions = Vec::new();
        for (ordinal, posting) in decoded.iter().copied().enumerate() {
            if tombstones.contains(posting.doc_id) {
                continue;
            }
            if let Some(position_list) = &position_list {
                let ordinal =
                    u32::try_from(ordinal).map_err(|_| CompactionError::ArithmeticOverflow {
                        field: "position posting ordinal",
                    })?;
                for position in position_list
                    .positions_for_ordinal(ordinal)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?
                {
                    retained_positions.push(
                        position
                            .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
                    );
                }
            }
            retained.push(posting);
        }
        if retained.is_empty() {
            continue;
        }
        let retained_token_count = retained.iter().try_fold(0_u64, |total, posting| {
            total
                .checked_add(u64::from(posting.freq))
                .ok_or(CompactionError::ArithmeticOverflow {
                    field: "retained posting frequency sum",
                })
        })?;
        let field_total = total_tokens.entry(term.field_ord).or_default();
        *field_total = field_total.checked_add(retained_token_count).ok_or(
            CompactionError::ArithmeticOverflow {
                field: "retained field token total",
            },
        )?;
        let fieldnorms =
            source
                .doclen
                .field(term.field_ord)
                .ok_or_else(|| CompactionError::SectionCodec {
                    section: SectionKind::DOCLEN,
                    detail: format!("missing fieldnorm column for field {}", term.field_ord),
                })?;
        let (encoded_postings, encoded_blockmax) =
            EncodedPostingList::encode_with_block_max(&retained, |docid| {
                fieldnorms.fieldnorm_id(u64::from(docid))
            })
            .map_err(|error| compaction_codec(SectionKind::BLOCKMAX, error))?;
        let encoded_positions = if position_list.is_some() {
            Some(
                EncodedPositionList::encode(&retained, &retained_positions)
                    .map_err(|error| compaction_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            None
        };
        if encoded_positions.is_some() && !has_positions {
            return Err(CompactionError::SectionCodec {
                section: SectionKind::POSITIONS,
                detail: "term carries positions but the schema has no POSITIONS section".to_owned(),
            });
        }

        let postings_offset = durable_compaction_len(&postings_output, "POSTINGS offset")?;
        append_compaction_bytes(
            &mut postings_output,
            encoded_postings.as_bytes(),
            "POSTINGS output",
        )?;
        let blockmax_offset = durable_compaction_len(&blockmax_output, "BLOCKMAX offset")?;
        append_compaction_bytes(
            &mut blockmax_output,
            encoded_blockmax.as_bytes(),
            "BLOCKMAX output",
        )?;
        let positions_span = if let Some(encoded_positions) = encoded_positions {
            let offset = durable_compaction_len(&positions_output, "POSITIONS offset")?;
            let len = durable_compaction_len(encoded_positions.as_bytes(), "POSITIONS length")?;
            append_compaction_bytes(
                &mut positions_output,
                encoded_positions.as_bytes(),
                "POSITIONS output",
            )?;
            Some(ByteSpan::new(offset, len))
        } else {
            None
        };
        let doc_freq = encoded_postings.doc_freq();
        let postings_len = durable_compaction_len(encoded_postings.as_bytes(), "POSTINGS length")?;
        let blockmax_len = durable_compaction_len(encoded_blockmax.as_bytes(), "BLOCKMAX length")?;
        let metadata = positions_span.map_or_else(
            || {
                TermMetadata::without_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
            |span| {
                TermMetadata::with_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    span,
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
        );
        terms.push(MergedTermRow {
            field_ord: term.field_ord,
            term: term.term.clone(),
            metadata,
        });
    }
    Ok(CompactedTermSections {
        postings: postings_output,
        positions: positions_output,
        blockmax: blockmax_output,
        terms,
        total_tokens,
    })
}

fn durable_compaction_len(bytes: &[u8], field: &'static str) -> Result<u64, CompactionError> {
    u64::try_from(bytes.len()).map_err(|_| CompactionError::ArithmeticOverflow { field })
}

fn append_compaction_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    resource: &'static str,
) -> Result<(), CompactionError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| CompactionError::Allocation {
            resource,
            count: output.len().saturating_add(bytes.len()),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn compaction_codec(section: SectionKind, error: impl std::fmt::Display) -> CompactionError {
    CompactionError::SectionCodec {
        section,
        detail: error.to_string(),
    }
}

fn compaction_source_validation(error: &ConcatMergeError) -> CompactionError {
    CompactionError::SourceValidation {
        detail: error.to_string(),
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ConcatTermHeapItem {
    field_ord: u16,
    term: Vec<u8>,
    source_index: usize,
    term_index: usize,
}

impl Ord for ConcatTermHeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .field_ord
            .cmp(&self.field_ord)
            .then_with(|| other.term.cmp(&self.term))
            .then_with(|| other.source_index.cmp(&self.source_index))
            .then_with(|| other.term_index.cmp(&self.term_index))
    }
}

impl PartialOrd for ConcatTermHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct MergedTermRow {
    field_ord: u16,
    term: Vec<u8>,
    metadata: TermMetadata,
}

type MergedTermSections = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MergedTermRow>);

#[derive(Clone, Copy, Debug)]
struct IdentityRepresentative {
    lowest_global_docid: u32,
    live_global_docid: Option<u32>,
}

fn build_concat_merge(
    snapshot: &KeeperSnapshot,
    source_segment_ids: &[u64],
    output_segment_id: u64,
    created_unix_s: i64,
) -> Result<ConcatMergeArtifact, ConcatMergeError> {
    if source_segment_ids.len() < 2 {
        return Err(ConcatMergeError::TooFewSources {
            count: source_segment_ids.len(),
        });
    }
    let segments = snapshot.segments();
    let first_id = source_segment_ids[0];
    let start = segments
        .iter()
        .position(|segment| segment.manifest.segment_id == first_id)
        .ok_or(ConcatMergeError::SourceNotFound {
            position: 0,
            segment_id: first_id,
        })?;
    for (position, &actual) in source_segment_ids.iter().enumerate() {
        let expected_index = start
            .checked_add(position)
            .ok_or(ConcatMergeError::SourceRunPastManifest { position, actual })?;
        let Some(expected) = segments.get(expected_index) else {
            if segments
                .iter()
                .any(|segment| segment.manifest.segment_id == actual)
            {
                return Err(ConcatMergeError::SourceRunPastManifest { position, actual });
            }
            return Err(ConcatMergeError::SourceNotFound {
                position,
                segment_id: actual,
            });
        };
        if expected.manifest.segment_id != actual {
            if segments
                .iter()
                .any(|segment| segment.manifest.segment_id == actual)
            {
                return Err(ConcatMergeError::NonConsecutiveSources {
                    position,
                    expected: expected.manifest.segment_id,
                    actual,
                });
            }
            return Err(ConcatMergeError::SourceNotFound {
                position,
                segment_id: actual,
            });
        }
    }
    if segments
        .iter()
        .any(|segment| segment.manifest.segment_id == output_segment_id)
    {
        return Err(ConcatMergeError::OutputSegmentCollision {
            segment_id: output_segment_id,
        });
    }
    let end = start.checked_add(source_segment_ids.len()).ok_or(
        ConcatMergeError::ArithmeticOverflow {
            field: "source manifest slice end",
        },
    )?;
    let selected =
        segments
            .get(start..end)
            .ok_or_else(|| ConcatMergeError::SourceRunPastManifest {
                position: source_segment_ids.len(),
                actual: *source_segment_ids.last().unwrap_or(&first_id),
            })?;
    for segment in selected {
        if let Some(entry) = segment.section_entries().iter().find(|entry| {
            !(SectionKind::TERMDICT.raw()..=SectionKind::STATS.raw()).contains(&entry.kind.raw())
        }) {
            return Err(ConcatMergeError::UnknownSection {
                segment_id: segment.manifest.segment_id,
                section_kind: entry.kind.raw(),
            });
        }
        segment
            .verify()
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }

    let schema = snapshot.schema();
    let term_fields = concat_term_field_ords(schema)?;
    let stored_fields = concat_stored_field_ords(schema)?;
    let has_positions = concat_schema_has_positions(schema);
    let has_numeric = concat_schema_has_numeric(schema);
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(selected.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "source views",
            count: selected.len(),
        })?;
    for segment in selected {
        sources.push(open_concat_source(
            segment,
            schema,
            &term_fields,
            &stored_fields,
            has_positions,
            has_numeric,
        )?);
    }

    let docid_lo = selected
        .first()
        .map(|segment| segment.manifest.docid_lo)
        .ok_or(ConcatMergeError::TooFewSources { count: 0 })?;
    let docid_hi = selected
        .last()
        .map(|segment| segment.manifest.docid_hi)
        .ok_or(ConcatMergeError::TooFewSources { count: 0 })?;
    let doc_count_u64 = selected.iter().try_fold(0_u64, |count, segment| {
        count
            .checked_add(u64::from(segment.manifest.doc_count))
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "physical document count",
            })
    })?;
    let doc_count =
        u32::try_from(doc_count_u64).map_err(|_| ConcatMergeError::ArithmeticOverflow {
            field: "durable physical document count",
        })?;

    let (postings, positions, blockmax, merged_terms) =
        merge_concat_terms(&sources, has_positions)?;
    let mut term_inputs = Vec::new();
    term_inputs
        .try_reserve_exact(merged_terms.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "TERMDICT inputs",
            count: merged_terms.len(),
        })?;
    for term in &merged_terms {
        term_inputs.push(TermInput::new(term.field_ord, &term.term, term.metadata));
    }
    let termdict = EncodedTermDictionary::encode_sorted(
        schema,
        TermSectionLengths {
            postings: durable_concat_len(&postings, "POSTINGS")?,
            positions: has_positions
                .then(|| durable_concat_len(&positions, "POSITIONS"))
                .transpose()?,
            blockmax: durable_concat_len(&blockmax, "BLOCKMAX")?,
        },
        &term_inputs,
    )
    .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;

    let mut doclen_sections = Vec::new();
    doclen_sections
        .try_reserve_exact(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "DOCLEN source views",
            count: sources.len(),
        })?;
    doclen_sections.extend(sources.iter().map(|source| source.doclen.clone()));
    let doclen = EncodedDocLenSection::concatenate(&doclen_sections, &term_fields)
        .map_err(|error| concat_codec(SectionKind::DOCLEN, error))?;
    let mut id_map_sections = Vec::new();
    id_map_sections
        .try_reserve_exact(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "IDMAP source views",
            count: sources.len(),
        })?;
    id_map_sections.extend(sources.iter().map(|source| source.id_map));
    let id_map_plan = EncodedIdMapSection::plan_concatenate(&id_map_sections)
        .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;

    let tombstone_docids = collect_concat_tombstones(&sources)?;
    let tombstones = TombstoneSet::from_sorted_docids(&tombstone_docids).map_err(|error| {
        ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        }
    })?;
    let representative_ordinals = resolve_concat_representatives_from_sources(
        &id_map_sections,
        id_map_plan.docid_lo(),
        &tombstone_docids,
    )?;
    let id_hash = EncodedIdHashSection::encode_resolved_concat(
        &id_map_sections,
        &id_map_plan,
        &representative_ordinals,
    )
    .map_err(|error| concat_codec(SectionKind::IDHASH, error))?;

    let numeric = if has_numeric {
        let mut numeric_sections = Vec::new();
        numeric_sections
            .try_reserve_exact(sources.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "NUMERIC source views",
                count: sources.len(),
            })?;
        for source in &sources {
            numeric_sections.push(
                source
                    .numeric
                    .clone()
                    .ok_or_else(|| concat_missing_section(source.segment, SectionKind::NUMERIC))?,
            );
        }
        Some(
            EncodedNumericSection::merge_sorted(schema, &numeric_sections)
                .map_err(|error| concat_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };
    let mut stored_sections = Vec::new();
    let stored_meta_plan =
        if stored_fields.is_empty() {
            None
        } else {
            stored_sections
                .try_reserve_exact(sources.len())
                .map_err(|_| ConcatMergeError::Allocation {
                    resource: "STOREDMETA source views",
                    count: sources.len(),
                })?;
            for source in &sources {
                stored_sections.push(source.stored_meta.clone().ok_or_else(|| {
                    concat_missing_section(source.segment, SectionKind::STOREDMETA)
                })?);
            }
            Some(
                EncodedStoredMetaSection::plan_concatenate(&stored_sections, &stored_fields)
                    .map_err(|error| concat_codec(SectionKind::STOREDMETA, error))?,
            )
        };

    let aggregate = aggregate_field_stats(sources.iter().map(|source| &source.stats))
        .map_err(|error| concat_codec(SectionKind::STATS, error))?;
    let mut stats_rows = Vec::new();
    stats_rows
        .try_reserve_exact(aggregate.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "STATS rows",
            count: aggregate.len(),
        })?;
    for row in aggregate {
        if row.doc_count != doc_count_u64 {
            return Err(ConcatMergeError::SectionCodec {
                section: SectionKind::STATS,
                detail: format!(
                    "field {} denominator {} differs from merged physical count {doc_count_u64}",
                    row.field_ord, row.doc_count
                ),
            });
        }
        stats_rows.push(FieldStats::new(row.field_ord, row.total_tokens, doc_count));
    }
    let stats = EncodedStatsSection::encode(&term_fields, &stats_rows, doc_count)
        .map_err(|error| concat_codec(SectionKind::STATS, error))?;

    let mut section_plan = Vec::new();
    section_plan
        .try_reserve_exact(10)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "FSLX section plan",
            count: 10,
        })?;
    section_plan.push(PlannedSection::new(
        SectionKind::TERMDICT,
        termdict.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(SectionKind::POSTINGS, postings.len()));
    if has_positions {
        section_plan.push(PlannedSection::new(SectionKind::POSITIONS, positions.len()));
    }
    section_plan.push(PlannedSection::new(SectionKind::BLOCKMAX, blockmax.len()));
    section_plan.push(PlannedSection::new(
        SectionKind::DOCLEN,
        doclen.as_bytes().len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDMAP,
        id_map_plan.encoded_len(),
    ));
    section_plan.push(PlannedSection::new(
        SectionKind::IDHASH,
        id_hash.as_bytes().len(),
    ));
    if let Some(numeric) = &numeric {
        section_plan.push(PlannedSection::new(
            SectionKind::NUMERIC,
            numeric.as_bytes().len(),
        ));
    }
    if let Some(stored_meta_plan) = &stored_meta_plan {
        section_plan.push(PlannedSection::new(
            SectionKind::STOREDMETA,
            stored_meta_plan.encoded_len(),
        ));
    }
    section_plan.push(PlannedSection::new(
        SectionKind::STATS,
        stats.as_bytes().len(),
    ));

    let mut assembler = SegmentAssembler::new(
        SegmentHeaderInput {
            segment_id: output_segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s,
            engine_version: CURRENT_ENGINE_VERSION,
        },
        &section_plan,
    )
    .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::TERMDICT, termdict.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::POSTINGS, &postings)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    if has_positions {
        assembler
            .copy_section(SectionKind::POSITIONS, &positions)
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::BLOCKMAX, &blockmax)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .copy_section(SectionKind::DOCLEN, doclen.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    assembler
        .write_section(SectionKind::IDMAP, |bytes| {
            id_map_plan.append_to(&id_map_sections, bytes);
        })
        .map_err(|source| ConcatMergeError::Segment { source })?;

    {
        let id_map_bytes = assembler
            .written_section(SectionKind::IDMAP)
            .map_err(|source| ConcatMergeError::Segment { source })?;
        let merged_id_map = IdMapSection::parse(id_map_bytes, docid_lo, docid_hi)
            .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;
        id_hash
            .section(merged_id_map)
            .map_err(|error| concat_codec(SectionKind::IDHASH, error))?;
    }
    assembler
        .copy_section(SectionKind::IDHASH, id_hash.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    if let Some(numeric) = &numeric {
        assembler
            .copy_section(SectionKind::NUMERIC, numeric.as_bytes())
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    if let Some(stored_meta_plan) = &stored_meta_plan {
        assembler
            .write_section(SectionKind::STOREDMETA, |bytes| {
                stored_meta_plan.append_to(&stored_sections, &stored_fields, bytes);
            })
            .map_err(|source| ConcatMergeError::Segment { source })?;
    }
    assembler
        .copy_section(SectionKind::STATS, stats.as_bytes())
        .map_err(|source| ConcatMergeError::Segment { source })?;
    let encoded = assembler
        .finish()
        .map_err(|source| ConcatMergeError::Segment { source })?;
    let reader = SegmentReader::from_bytes(encoded.as_bytes(), schema)
        .map_err(|source| ConcatMergeError::Segment { source })?;
    reader
        .verify()
        .map_err(|source| ConcatMergeError::Segment { source })?;

    let next_seal_seq = snapshot
        .loaded
        .manifest
        .segments
        .iter()
        .map(|segment| segment.seal_seq)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(ConcatMergeError::ArithmeticOverflow {
            field: "seal sequence",
        })?;
    let replacement = ManifestSegment {
        segment_id: output_segment_id,
        seal_seq: next_seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo,
        docid_hi,
        doc_count,
        tombstones,
    };
    let mut manifest = snapshot.loaded.manifest.clone();
    manifest.generation =
        manifest
            .generation
            .checked_add(1)
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "manifest generation",
            })?;
    manifest.last_publish_unix_s = 0;
    manifest.segments.splice(start..end, [replacement]);
    manifest
        .validate()
        .map_err(|error| ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        })?;
    Ok(ConcatMergeArtifact { encoded, manifest })
}

fn open_concat_source<'a>(
    segment: &'a RecoveredSegment,
    schema: SchemaDescriptor,
    term_fields: &[u16],
    stored_fields: &[u16],
    has_positions: bool,
    has_numeric: bool,
) -> Result<ConcatSource<'a>, ConcatMergeError> {
    let postings = required_concat_section(segment, SectionKind::POSTINGS)?;
    let positions = if has_positions {
        Some(required_concat_section(segment, SectionKind::POSITIONS)?)
    } else {
        None
    };
    let blockmax = required_concat_section(segment, SectionKind::BLOCKMAX)?;
    let termdict_bytes = required_concat_section(segment, SectionKind::TERMDICT)?;
    let termdict = TermDictionary::parse(
        termdict_bytes,
        schema,
        TermSectionLengths {
            postings: durable_concat_len(postings, "source POSTINGS")?,
            positions: positions
                .map(|bytes| durable_concat_len(bytes, "source POSITIONS"))
                .transpose()?,
            blockmax: durable_concat_len(blockmax, "source BLOCKMAX")?,
        },
    )
    .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;
    let term_count = usize::try_from(termdict.term_count()).map_err(|_| {
        ConcatMergeError::ArithmeticOverflow {
            field: "source term count",
        }
    })?;
    let terms = termdict
        .cursor()
        .and_then(|cursor| cursor.collect_bounded(term_count))
        .map_err(|error| concat_codec(SectionKind::TERMDICT, error))?;

    let manifest = segment.manifest();
    let doclen = DocLenSection::parse(
        required_concat_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        term_fields,
    )
    .map_err(|error| concat_codec(SectionKind::DOCLEN, error))?;
    let id_map = IdMapSection::parse(
        required_concat_section(segment, SectionKind::IDMAP)?,
        manifest.docid_lo,
        manifest.docid_hi,
    )
    .map_err(|error| concat_codec(SectionKind::IDMAP, error))?;
    let numeric = if has_numeric {
        Some(
            NumericSection::parse(
                required_concat_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(|error| concat_codec(SectionKind::NUMERIC, error))?,
        )
    } else {
        None
    };
    let stored_meta = if stored_fields.is_empty() {
        None
    } else {
        Some(
            StoredMetaSection::parse(
                required_concat_section(segment, SectionKind::STOREDMETA)?,
                manifest.docid_lo,
                manifest.docid_hi,
                stored_fields,
            )
            .map_err(|error| concat_codec(SectionKind::STOREDMETA, error))?,
        )
    };
    let stats = StatsSection::parse(
        required_concat_section(segment, SectionKind::STATS)?,
        term_fields,
        manifest.doc_count,
    )
    .map_err(|error| concat_codec(SectionKind::STATS, error))?;
    Ok(ConcatSource {
        segment,
        terms,
        postings,
        positions,
        blockmax,
        doclen,
        id_map,
        numeric,
        stored_meta,
        stats,
    })
}

fn merge_concat_terms(
    sources: &[ConcatSource<'_>],
    has_positions: bool,
) -> Result<MergedTermSections, ConcatMergeError> {
    let mut heap = BinaryHeap::new();
    heap.try_reserve(sources.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "term merge heap",
            count: sources.len(),
        })?;
    for (source_index, source) in sources.iter().enumerate() {
        push_concat_term(&mut heap, source, source_index, 0)?;
    }
    let mut postings_output = Vec::new();
    let mut positions_output = Vec::new();
    let mut blockmax_output = Vec::new();
    let total_terms = sources.iter().try_fold(0_usize, |count, source| {
        count
            .checked_add(source.terms.len())
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "source term count sum",
            })
    })?;
    let mut merged_terms = Vec::new();
    merged_terms
        .try_reserve(total_terms)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "merged term rows",
            count: total_terms,
        })?;

    while let Some(first) = heap.pop() {
        let field_ord = first.field_ord;
        let term = first.term;
        let mut parts = Vec::new();
        parts
            .try_reserve_exact(sources.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "equal-term source rows",
                count: sources.len(),
            })?;
        parts.push((first.source_index, first.term_index));
        push_concat_term(
            &mut heap,
            &sources[first.source_index],
            first.source_index,
            first.term_index + 1,
        )?;
        while heap
            .peek()
            .is_some_and(|next| next.field_ord == field_ord && next.term == term)
        {
            let next = heap.pop().expect("peeked concat term remains present");
            parts.push((next.source_index, next.term_index));
            push_concat_term(
                &mut heap,
                &sources[next.source_index],
                next.source_index,
                next.term_index + 1,
            )?;
        }
        parts.sort_unstable_by_key(|(source_index, _)| *source_index);

        let mut posting_slices = Vec::new();
        posting_slices.try_reserve_exact(parts.len()).map_err(|_| {
            ConcatMergeError::Allocation {
                resource: "term POSTINGS slices",
                count: parts.len(),
            }
        })?;
        for &(source_index, term_index) in &parts {
            let source = &sources[source_index];
            let metadata = source.terms[term_index].metadata;
            posting_slices.push(concat_span(
                source.postings,
                metadata.postings,
                SectionKind::POSTINGS,
            )?);
        }
        let mut posting_lists = Vec::new();
        posting_lists
            .try_reserve_exact(parts.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "source POSTINGS views",
                count: parts.len(),
            })?;
        for (&bytes, &(source_index, term_index)) in posting_slices.iter().zip(&parts) {
            posting_lists.push(
                PostingList::parse(
                    bytes,
                    sources[source_index].terms[term_index].metadata.doc_freq,
                )
                .map_err(|error| concat_codec(SectionKind::POSTINGS, error))?,
            );
        }
        let mut posting_refs = Vec::new();
        posting_refs
            .try_reserve_exact(posting_lists.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "POSTINGS references",
                count: posting_lists.len(),
            })?;
        posting_refs.extend(posting_lists.iter());
        let merged_postings = EncodedPostingList::concatenate(&posting_refs)
            .map_err(|error| concat_codec(SectionKind::POSTINGS, error))?;
        let doc_freq = merged_postings.doc_freq();

        let mut blockmax_lists = Vec::new();
        blockmax_lists.try_reserve_exact(parts.len()).map_err(|_| {
            ConcatMergeError::Allocation {
                resource: "source BLOCKMAX views",
                count: parts.len(),
            }
        })?;
        for ((source_index, term_index), postings) in parts.iter().copied().zip(&posting_lists) {
            let source = &sources[source_index];
            let bytes = concat_span(
                source.blockmax,
                source.terms[term_index].metadata.blockmax,
                SectionKind::BLOCKMAX,
            )?;
            blockmax_lists.push(
                BlockMaxConcatList::parse(bytes, postings)
                    .map_err(|error| concat_codec(SectionKind::BLOCKMAX, error))?,
            );
        }
        let mut blockmax_refs = Vec::new();
        blockmax_refs
            .try_reserve_exact(blockmax_lists.len())
            .map_err(|_| ConcatMergeError::Allocation {
                resource: "BLOCKMAX references",
                count: blockmax_lists.len(),
            })?;
        blockmax_refs.extend(blockmax_lists.iter());
        let merged_blockmax = EncodedBlockMax::concatenate(&blockmax_refs)
            .map_err(|error| concat_codec(SectionKind::BLOCKMAX, error))?;

        let has_term_positions = sources[parts[0].0].terms[parts[0].1]
            .metadata
            .positions
            .is_some();
        let merged_positions = if has_term_positions {
            let mut position_lists = Vec::new();
            position_lists.try_reserve_exact(parts.len()).map_err(|_| {
                ConcatMergeError::Allocation {
                    resource: "source POSITIONS views",
                    count: parts.len(),
                }
            })?;
            for ((source_index, term_index), postings) in parts.iter().copied().zip(&posting_lists)
            {
                let source = &sources[source_index];
                let span = source.terms[term_index].metadata.positions.ok_or_else(|| {
                    ConcatMergeError::SectionCodec {
                        section: SectionKind::POSITIONS,
                        detail: format!(
                            "field {field_ord} term {:?} has inconsistent positional metadata",
                            String::from_utf8_lossy(&term)
                        ),
                    }
                })?;
                let bytes = concat_span(
                    source.positions.ok_or_else(|| {
                        concat_missing_section(source.segment, SectionKind::POSITIONS)
                    })?,
                    span,
                    SectionKind::POSITIONS,
                )?;
                position_lists.push(
                    PositionList::parse(bytes, postings)
                        .map_err(|error| concat_codec(SectionKind::POSITIONS, error))?,
                );
            }
            let mut refs = Vec::new();
            refs.try_reserve_exact(position_lists.len()).map_err(|_| {
                ConcatMergeError::Allocation {
                    resource: "POSITIONS references",
                    count: position_lists.len(),
                }
            })?;
            refs.extend(position_lists.iter());
            Some(
                EncodedPositionList::concatenate(&refs)
                    .map_err(|error| concat_codec(SectionKind::POSITIONS, error))?,
            )
        } else {
            for &(source_index, term_index) in &parts {
                if sources[source_index].terms[term_index]
                    .metadata
                    .positions
                    .is_some()
                {
                    return Err(ConcatMergeError::SectionCodec {
                        section: SectionKind::POSITIONS,
                        detail: "equal term has inconsistent positional metadata".to_owned(),
                    });
                }
            }
            None
        };
        if has_term_positions && !has_positions {
            return Err(ConcatMergeError::SectionCodec {
                section: SectionKind::POSITIONS,
                detail: "term carries positions but schema has no POSITIONS section".to_owned(),
            });
        }

        let postings_offset = durable_concat_len(&postings_output, "POSTINGS offset")?;
        append_concat_bytes(
            &mut postings_output,
            merged_postings.as_bytes(),
            "POSTINGS output",
        )?;
        let blockmax_offset = durable_concat_len(&blockmax_output, "BLOCKMAX offset")?;
        append_concat_bytes(
            &mut blockmax_output,
            merged_blockmax.as_bytes(),
            "BLOCKMAX output",
        )?;
        let positions_span = if let Some(merged_positions) = merged_positions {
            let offset = durable_concat_len(&positions_output, "POSITIONS offset")?;
            let len = durable_concat_len(merged_positions.as_bytes(), "term POSITIONS length")?;
            append_concat_bytes(
                &mut positions_output,
                merged_positions.as_bytes(),
                "POSITIONS output",
            )?;
            Some(ByteSpan::new(offset, len))
        } else {
            None
        };
        let postings_len = durable_concat_len(merged_postings.as_bytes(), "term POSTINGS length")?;
        let blockmax_len = durable_concat_len(merged_blockmax.as_bytes(), "term BLOCKMAX length")?;
        let metadata = positions_span.map_or_else(
            || {
                TermMetadata::without_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
            |positions_span| {
                TermMetadata::with_positions(
                    doc_freq,
                    ByteSpan::new(postings_offset, postings_len),
                    positions_span,
                    ByteSpan::new(blockmax_offset, blockmax_len),
                )
            },
        );
        merged_terms.push(MergedTermRow {
            field_ord,
            term,
            metadata,
        });
    }
    Ok((
        postings_output,
        positions_output,
        blockmax_output,
        merged_terms,
    ))
}

fn push_concat_term(
    heap: &mut BinaryHeap<ConcatTermHeapItem>,
    source: &ConcatSource<'_>,
    source_index: usize,
    term_index: usize,
) -> Result<(), ConcatMergeError> {
    let Some(term) = source.terms.get(term_index) else {
        return Ok(());
    };
    let mut key = Vec::new();
    key.try_reserve_exact(term.term.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "term heap key",
            count: term.term.len(),
        })?;
    key.extend_from_slice(&term.term);
    heap.push(ConcatTermHeapItem {
        field_ord: term.field_ord,
        term: key,
        source_index,
        term_index,
    });
    Ok(())
}

fn collect_concat_tombstones(sources: &[ConcatSource<'_>]) -> Result<Vec<u32>, ConcatMergeError> {
    let count = sources.iter().try_fold(0_usize, |count, source| {
        let source_count = usize::try_from(source.segment.manifest.tombstones.cardinality())
            .map_err(|_| ConcatMergeError::ArithmeticOverflow {
                field: "host tombstone cardinality",
            })?;
        count
            .checked_add(source_count)
            .ok_or(ConcatMergeError::ArithmeticOverflow {
                field: "merged tombstone cardinality",
            })
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "merged tombstone docids",
            count,
        })?;
    for source in sources {
        let mut containers = TombstoneContainers::new(
            source.segment.manifest.tombstones.as_bytes(),
        )
        .map_err(|error| ConcatMergeError::InvalidManifest {
            detail: error.to_string(),
        })?;
        while let Some(container) =
            containers
                .next_container()
                .map_err(|error| ConcatMergeError::InvalidManifest {
                    detail: error.to_string(),
                })?
        {
            match container.kind {
                0 => {
                    for index in 0..container.payload.len() / 2 {
                        output.push(
                            (u32::from(container.chunk_id) << 16)
                                | u32::from(array_value(container.payload, index)),
                        );
                    }
                }
                1 => {
                    for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                        let mut bits = byte;
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            let low = u16::try_from(byte_index * 8 + bit).map_err(|_| {
                                ConcatMergeError::ArithmeticOverflow {
                                    field: "bitmap tombstone low bits",
                                }
                            })?;
                            output.push((u32::from(container.chunk_id) << 16) | u32::from(low));
                            bits &= bits - 1;
                        }
                    }
                }
                _ => {
                    return Err(ConcatMergeError::InvalidManifest {
                        detail: "unknown validated tombstone container kind".to_owned(),
                    });
                }
            }
        }
    }
    if output.len() != count {
        return Err(ConcatMergeError::InvalidManifest {
            detail: format!(
                "decoded tombstone count {} differs from declared {count}",
                output.len()
            ),
        });
    }
    Ok(output)
}

#[cfg(test)]
fn resolve_concat_representatives(
    id_map: IdMapSection<'_>,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    resolve_concat_representatives_from_entries(
        id_map.docid_lo(),
        id_map.entries(),
        tombstone_docids,
    )
}

fn resolve_concat_representatives_from_sources(
    id_maps: &[IdMapSection<'_>],
    docid_lo: u64,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    resolve_concat_representatives_from_entries(
        docid_lo,
        id_maps.iter().copied().flat_map(|id_map| id_map.entries()),
        tombstone_docids,
    )
}

fn resolve_concat_representatives_from_entries<'a>(
    docid_lo: u64,
    entries: impl IntoIterator<Item = (u64, IdMapEntry<'a>)>,
    tombstone_docids: &[u32],
) -> Result<Vec<u32>, ConcatMergeError> {
    let mut representatives = BTreeMap::<&str, IdentityRepresentative>::new();
    let mut tombstones = tombstone_docids.iter().copied().peekable();
    for (global_docid, entry) in entries {
        let global_docid =
            u32::try_from(global_docid).map_err(|_| ConcatMergeError::ArithmeticOverflow {
                field: "IDHASH representative global docid",
            })?;
        if let Some(&tombstone) = tombstones.peek()
            && tombstone < global_docid
        {
            return Err(ConcatMergeError::InvalidManifest {
                detail: format!(
                    "merged tombstone {tombstone} references an IDMAP hole before row {global_docid}"
                ),
            });
        }
        let is_live = if tombstones.peek().copied() == Some(global_docid) {
            tombstones.next();
            false
        } else {
            true
        };
        match representatives.get_mut(entry.document_id()) {
            Some(representative) => {
                representative.lowest_global_docid =
                    representative.lowest_global_docid.min(global_docid);
                if is_live {
                    if let Some(first_global_docid) = representative.live_global_docid {
                        let mut document_id = String::new();
                        document_id
                            .try_reserve_exact(entry.document_id().len())
                            .map_err(|_| ConcatMergeError::Allocation {
                                resource: "duplicate document id diagnosis",
                                count: entry.document_id().len(),
                            })?;
                        document_id.push_str(entry.document_id());
                        return Err(ConcatMergeError::MultipleLiveDocumentIds {
                            document_id,
                            first_global_docid,
                            duplicate_global_docid: global_docid,
                        });
                    }
                    representative.live_global_docid = Some(global_docid);
                }
            }
            None => {
                representatives.insert(
                    entry.document_id(),
                    IdentityRepresentative {
                        lowest_global_docid: global_docid,
                        live_global_docid: is_live.then_some(global_docid),
                    },
                );
            }
        }
    }
    if let Some(tombstone) = tombstones.next() {
        return Err(ConcatMergeError::InvalidManifest {
            detail: format!(
                "merged tombstone {tombstone} is not represented by a trailing IDMAP row"
            ),
        });
    }
    let mut ordinals = Vec::new();
    ordinals
        .try_reserve_exact(representatives.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "IDHASH representative ordinals",
            count: representatives.len(),
        })?;
    for representative in representatives.values() {
        let global_docid = representative
            .live_global_docid
            .unwrap_or(representative.lowest_global_docid);
        let ordinal = u64::from(global_docid).checked_sub(docid_lo).ok_or(
            ConcatMergeError::ArithmeticOverflow {
                field: "IDHASH representative ordinal",
            },
        )?;
        ordinals.push(u32::try_from(ordinal).map_err(|_| {
            ConcatMergeError::ArithmeticOverflow {
                field: "durable IDHASH representative ordinal",
            }
        })?);
    }
    ordinals.sort_unstable();
    Ok(ordinals)
}

fn required_concat_section(
    segment: &RecoveredSegment,
    kind: SectionKind,
) -> Result<&[u8], ConcatMergeError> {
    segment
        .section(kind)
        .map_err(|source| ConcatMergeError::Segment { source })?
        .ok_or_else(|| concat_missing_section(segment, kind))
}

fn concat_missing_section(segment: &RecoveredSegment, kind: SectionKind) -> ConcatMergeError {
    ConcatMergeError::SectionCodec {
        section: kind,
        detail: format!(
            "source segment {:#018x} is missing the required section",
            segment.manifest.segment_id
        ),
    }
}

fn concat_span(
    bytes: &[u8],
    span: ByteSpan,
    section: SectionKind,
) -> Result<&[u8], ConcatMergeError> {
    let start = usize::try_from(span.offset).map_err(|_| ConcatMergeError::ArithmeticOverflow {
        field: "section span offset",
    })?;
    let len = usize::try_from(span.len).map_err(|_| ConcatMergeError::ArithmeticOverflow {
        field: "section span length",
    })?;
    let end = start
        .checked_add(len)
        .ok_or(ConcatMergeError::ArithmeticOverflow {
            field: "section span end",
        })?;
    bytes
        .get(start..end)
        .ok_or_else(|| ConcatMergeError::SectionCodec {
            section,
            detail: format!(
                "referenced byte span [{start}, {end}) exceeds section length {}",
                bytes.len()
            ),
        })
}

fn durable_concat_len(bytes: &[u8], field: &'static str) -> Result<u64, ConcatMergeError> {
    u64::try_from(bytes.len()).map_err(|_| ConcatMergeError::ArithmeticOverflow { field })
}

fn append_concat_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
    resource: &'static str,
) -> Result<(), ConcatMergeError> {
    output
        .try_reserve(bytes.len())
        .map_err(|_| ConcatMergeError::Allocation {
            resource,
            count: output.len().saturating_add(bytes.len()),
        })?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn concat_codec(section: SectionKind, error: impl std::fmt::Display) -> ConcatMergeError {
    ConcatMergeError::SectionCodec {
        section,
        detail: error.to_string(),
    }
}

fn concat_term_field_ords(schema: SchemaDescriptor) -> Result<Vec<u16>, ConcatMergeError> {
    let count = schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "schema term fields",
            count,
        })?;
    output.extend(schema.fields.iter().filter_map(|field| {
        matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }).then_some(field.id)
    }));
    Ok(output)
}

fn concat_stored_field_ords(schema: SchemaDescriptor) -> Result<Vec<u16>, ConcatMergeError> {
    let count = schema.fields.iter().filter(|field| field.stored).count();
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| ConcatMergeError::Allocation {
            resource: "schema stored fields",
            count,
        })?;
    output.extend(
        schema
            .fields
            .iter()
            .filter_map(|field| field.stored.then_some(field.id)),
    );
    Ok(output)
}

fn concat_schema_has_positions(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    })
}

fn concat_schema_has_numeric(schema: SchemaDescriptor) -> bool {
    schema
        .fields
        .iter()
        .any(|field| field.kind.has_numeric_column())
}

fn reconcile_published_segment(
    admission: &WriterAdmissionInner,
    protection: &WriterProtection,
    pending: &PendingSegmentFile,
) -> Result<Option<PathBuf>, KeeperError> {
    let published = admission
        .directory
        .join(canonical_segment_name(pending.segment_id()));
    let published_metadata = match std::fs::symlink_metadata(&published) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect retry segment destination",
                path: published,
                source,
            });
        }
    };
    let pending_metadata =
        std::fs::symlink_metadata(pending.path()).map_err(|source| KeeperError::Io {
            operation: "inspect retry segment temp",
            path: pending.path().to_path_buf(),
            source,
        })?;
    let expected_file_len = pending.file_len();
    if !published_metadata.file_type().is_file()
        || !pending_metadata.file_type().is_file()
        || published_metadata.len() != expected_file_len
        || pending_metadata.len() != expected_file_len
    {
        return Err(KeeperError::Io {
            operation: "reconcile published segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment and retry temp are not matching regular files",
            ),
        });
    }
    let expected = std::fs::read(pending.path()).map_err(|source| KeeperError::Io {
        operation: "read retry segment temp",
        path: pending.path().to_path_buf(),
        source,
    })?;
    let actual = std::fs::read(&published).map_err(|source| KeeperError::Io {
        operation: "read retry segment destination",
        path: published.clone(),
        source,
    })?;
    if actual != expected {
        return Err(KeeperError::Io {
            operation: "reconcile published segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment differs from the retry temp",
            ),
        });
    }
    File::open(&published)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "sync reconciled segment",
            path: published.clone(),
            source,
        })?;
    sync_directory(&admission.directory)?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled { protector, .. } = protection {
        ensure_matching_durability_sidecar(admission, protector, &published, &actual)?;
        sync_directory(&admission.directory)?;
    }
    #[cfg(not(feature = "durability"))]
    let _ = protection;
    Ok(Some(published))
}

fn reconcile_encoded_segment(
    admission: &WriterAdmissionInner,
    protection: &WriterProtection,
    encoded: &EncodedSegment,
) -> Result<Option<PathBuf>, KeeperError> {
    let published = admission
        .directory
        .join(canonical_segment_name(encoded.header().segment_id));
    let metadata = match std::fs::symlink_metadata(&published) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect retry segment destination",
                path: published,
                source,
            });
        }
    };
    if !metadata.file_type().is_file() || metadata.len() != encoded.file_len() {
        return Err(KeeperError::Io {
            operation: "reconcile encoded segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment is not the exact expected regular file",
            ),
        });
    }
    let actual = std::fs::read(&published).map_err(|source| KeeperError::Io {
        operation: "read retry segment destination",
        path: published.clone(),
        source,
    })?;
    if actual.as_slice() != encoded.as_bytes() {
        return Err(KeeperError::Io {
            operation: "reconcile encoded segment",
            path: published,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment differs from the retained canonical bytes",
            ),
        });
    }
    File::open(&published)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "sync reconciled segment",
            path: published.clone(),
            source,
        })?;
    sync_directory(&admission.directory)?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled { protector, .. } = protection {
        ensure_matching_durability_sidecar(admission, protector, &published, &actual)?;
        sync_directory(&admission.directory)?;
    }
    #[cfg(not(feature = "durability"))]
    let _ = protection;
    Ok(Some(published))
}

async fn writer_mutation_guard(cx: &Cx) -> Result<OwnedMutexGuard<()>, KeeperError> {
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    let guard = OwnedMutexGuard::lock(global_publish_lock(), cx)
        .await
        .map_err(|source| KeeperError::PublishLock { source })?;
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    Ok(guard)
}

fn recovery_retryable(error: &KeeperError) -> bool {
    matches!(
        error,
        KeeperError::SegmentOpen { .. } | KeeperError::SegmentMetadataMismatch { .. }
    )
}

async fn open_snapshot_blocking(
    directory: PathBuf,
    schema: SchemaDescriptor,
) -> Result<KeeperSnapshot, KeeperError> {
    spawn_blocking(move || KeeperSnapshot::open(directory, schema)).await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriterLockRecord {
    pid: u32,
    pid_start_nonce: u64,
    acquired_unix_s: i64,
}

impl WriterLockRecord {
    fn current(path: &Path) -> Result<Self, KeeperError> {
        let pid = std::process::id();
        if i32::try_from(pid).is_err() || pid == 0 {
            return Err(KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: format!("current process id {pid} is outside the POSIX pid range"),
            });
        }
        let acquired_unix_s = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
            .ok_or_else(|| KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: "current wall clock is outside the v1 lock-record range".to_owned(),
            })?;
        Ok(Self {
            pid,
            pid_start_nonce: writer_pid_start_nonce(pid, acquired_unix_s),
            acquired_unix_s,
        })
    }

    fn to_bytes(self) -> [u8; WRITER_LOCK_RECORD_BYTES] {
        let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
        bytes[..8].copy_from_slice(&WRITER_LOCK_MAGIC);
        bytes[8..12].copy_from_slice(&WRITER_LOCK_FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.pid.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.pid_start_nonce.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.acquired_unix_s.to_le_bytes());
        let checksum = crc32fast::hash(&bytes[..32]);
        bytes[32..36].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != WRITER_LOCK_RECORD_BYTES {
            return Err(format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {}",
                bytes.len()
            ));
        }
        if bytes[..8] != WRITER_LOCK_MAGIC {
            return Err("invalid writer-lock magic".to_owned());
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != WRITER_LOCK_FORMAT_VERSION {
            return Err(format!("unsupported writer-lock version {version}"));
        }
        let stored_checksum = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let computed_checksum = crc32fast::hash(&bytes[..32]);
        if stored_checksum != computed_checksum {
            return Err(format!(
                "writer-lock CRC mismatch: stored {stored_checksum:#010x}, computed {computed_checksum:#010x}"
            ));
        }
        let pid = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if pid == 0 || i32::try_from(pid).is_err() {
            return Err(format!(
                "writer-lock pid {pid} is outside the POSIX pid range"
            ));
        }
        let pid_start_nonce = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let acquired_unix_s = i64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        if acquired_unix_s < 0 {
            return Err(format!(
                "writer-lock acquisition timestamp {acquired_unix_s} predates the Unix epoch"
            ));
        }
        Ok(Self {
            pid,
            pid_start_nonce,
            acquired_unix_s,
        })
    }
}

fn writer_pid_start_nonce(pid: u32, acquired_unix_s: i64) -> u64 {
    #[cfg(target_os = "linux")]
    if let Some(start_time) = linux_process_start_time(pid) {
        let mut identity = [0_u8; 12];
        identity[..4].copy_from_slice(&pid.to_le_bytes());
        identity[4..].copy_from_slice(&start_time.to_le_bytes());
        return xxhash_rust::xxh3::xxh3_64(&identity);
    }

    let mut identity = [0_u8; 12];
    identity[..4].copy_from_slice(&pid.to_le_bytes());
    identity[4..].copy_from_slice(&acquired_unix_s.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&identity)
}

#[cfg(target_os = "linux")]
fn linux_process_start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_command = stat.get(stat.rfind(')')?.checked_add(1)?..)?;
    // The first token after the command is field 3 (`state`); starttime is
    // field 22, hence zero-based token 19 in this suffix.
    after_command.split_whitespace().nth(19)?.parse().ok()
}

struct WriterAdmissionInner {
    directory: PathBuf,
    directory_file: File,
    lock_path: PathBuf,
    lock_file: File,
    #[cfg(unix)]
    lock_device: u64,
    #[cfg(unix)]
    lock_inode: u64,
    record: WriterLockRecord,
}

impl WriterAdmissionInner {
    #[cfg(unix)]
    fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        ensure_gc_directory_identity(&self.directory, &self.directory_file)?;
        ensure_writer_lock_identity(self)
    }

    #[cfg(not(unix))]
    fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        Err(KeeperError::Io {
            operation: "verify writer-lock directory identity",
            path: self.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-process Quill writer admission requires Unix directory identity",
            ),
        })
    }
}

impl Drop for WriterAdmissionInner {
    fn drop(&mut self) {
        release_writer_admission(self);
    }
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn acquire_writer_admission(directory: &Path) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, openat};

    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let lock_path = directory.join("LOCK");
    let lock = openat(
        &directory_file,
        "LOCK",
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow writer lock",
        path: lock_path.clone(),
        source,
    })?;
    let mut lock_file = File::from(lock);
    let metadata = lock_file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect writer lock",
        path: lock_path.clone(),
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(KeeperError::WriterLockCorrupted {
            path: lock_path,
            detail: "LOCK must be a no-follow regular file".to_owned(),
        });
    }
    use std::os::unix::fs::MetadataExt;
    let lock_device = metadata.dev();
    let lock_inode = metadata.ino();
    if let Err(source) = flock(&lock_file, FlockOperation::NonBlockingLockExclusive) {
        if source == rustix::io::Errno::AGAIN {
            let owner_pid = read_writer_lock_record(&lock_path, &mut lock_file)
                .ok()
                .flatten()
                .map(|record| record.pid);
            return Err(KeeperError::WriterBusy {
                path: lock_path,
                owner_pid,
            });
        }
        return Err(KeeperError::Io {
            operation: "acquire writer flock",
            path: lock_path,
            source: io::Error::from(source),
        });
    }

    if let Some(previous) = read_writer_lock_record(&lock_path, &mut lock_file)?
        && !writer_pid_is_dead(previous.pid)
    {
        return Err(KeeperError::WriterBusy {
            path: lock_path,
            owner_pid: Some(previous.pid),
        });
    }

    let record = WriterLockRecord::current(&lock_path)?;
    if let Err(error) = write_writer_lock_record(&lock_path, &mut lock_file, record) {
        // The flock proves this descriptor is the only cooperating writer for
        // this inode. Best-effort truncation prevents a short failed write from
        // becoming a permanent corrupt residual record.
        let _ = lock_file.set_len(0);
        let _ = lock_file.sync_all();
        return Err(error);
    }
    let admission = Arc::new(WriterAdmissionInner {
        directory: directory.to_path_buf(),
        directory_file,
        lock_path,
        lock_file,
        lock_device,
        lock_inode,
        record,
    });
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync writer-lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
    admission.ensure_directory_identity()?;
    Ok(admission)
}

#[cfg(not(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
)))]
fn acquire_writer_admission(directory: &Path) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    Err(KeeperError::Io {
        operation: "verify writer-lock support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "cross-process Quill writer admission requires flock semantics",
        ),
    })
}

fn read_writer_lock_record(
    path: &Path,
    file: &mut File,
) -> Result<Option<WriterLockRecord>, KeeperError> {
    let length = file
        .metadata()
        .map_err(|source| KeeperError::Io {
            operation: "inspect writer-lock record",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length == 0 {
        return Ok(None);
    }
    if length != usize_to_u64(WRITER_LOCK_RECORD_BYTES) {
        return Err(KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail: format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {length}"
            ),
        });
    }
    let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|source| KeeperError::Io {
            operation: "read writer-lock record",
            path: path.to_path_buf(),
            source,
        })?;
    WriterLockRecord::from_bytes(&bytes)
        .map(Some)
        .map_err(|detail| KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail,
        })
}

fn write_writer_lock_record(
    path: &Path,
    file: &mut File,
    record: WriterLockRecord,
) -> Result<(), KeeperError> {
    let bytes = record.to_bytes();
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
        .and_then(|()| file.write_all(&bytes))
        .and_then(|()| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist writer-lock record",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(unix)]
fn writer_pid_is_dead(pid: u32) -> bool {
    let Ok(raw_pid) = i32::try_from(pid) else {
        return false;
    };
    let Some(pid) = rustix::process::Pid::from_raw(raw_pid) else {
        return false;
    };
    matches!(
        rustix::process::test_kill_process(pid),
        Err(source) if source == rustix::io::Errno::SRCH
    )
}

#[cfg(not(unix))]
fn writer_pid_is_dead(_: u32) -> bool {
    false
}

#[cfg(unix)]
fn ensure_writer_lock_identity(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let metadata =
        std::fs::symlink_metadata(&admission.lock_path).map_err(|source| KeeperError::Io {
            operation: "verify writer-lock pathname",
            path: admission.lock_path.clone(),
            source,
        })?;
    if !metadata.file_type().is_file()
        || metadata.dev() != admission.lock_device
        || metadata.ino() != admission.lock_inode
    {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.lock_path.clone(),
            detail: "LOCK pathname no longer resolves to the flocked inode".to_owned(),
        });
    }
    Ok(())
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn release_writer_admission(admission: &mut WriterAdmissionInner) {
    use rustix::fs::{FlockOperation, flock};

    if read_writer_lock_record(&admission.lock_path, &mut admission.lock_file)
        .is_ok_and(|record| record == Some(admission.record))
    {
        let _ = admission.lock_file.set_len(0);
        let _ = admission.lock_file.sync_all();
    }
    let _ = flock(&admission.lock_file, FlockOperation::Unlock);
}

#[cfg(not(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
)))]
fn release_writer_admission(_: &mut WriterAdmissionInner) {}

fn recover_writer_directory(
    admission: &Arc<WriterAdmissionInner>,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<(), KeeperError> {
    admission.ensure_directory_identity()?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled {
        protector,
        unrepairable,
    } = protection
    {
        recover_durable_manifest_slots(admission, schema, protector, *unrepairable)?;
    }
    recover_interrupted_generation_claims(admission)?;
    recover_corrupt_primary_slot(admission)?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled {
        protector,
        unrepairable,
    } = protection
    {
        recover_durable_writer_files(admission, schema, protector, *unrepairable)?;
    }
    #[cfg(not(feature = "durability"))]
    {
        let _ = schema;
        let _ = protection;
    }
    admission.ensure_directory_identity()
}

fn recover_interrupted_generation_claims(
    admission: &Arc<WriterAdmissionInner>,
) -> Result<(), KeeperError> {
    let recovered_generation = match load_manifest_pair(&admission.directory) {
        Ok(loaded) => loaded.manifest.generation,
        Err(KeeperError::IndexNotFound { .. }) => 0,
        Err(error) => return Err(error),
    };
    let next_generation = recovered_generation.checked_add(1);
    for (path, generation) in scan_generation_claims(&admission.directory)? {
        if generation > recovered_generation && Some(generation) != next_generation {
            return Err(KeeperError::InvalidRecoveryClaim {
                path,
                recovered: recovered_generation,
                claimed: generation,
            });
        }
        if Some(generation) == next_generation {
            retire_interrupted_manifest_temp(admission, generation)?;
        }
        remove_stale_generation_claim(admission, &path)?;
    }
    Ok(())
}

fn retire_interrupted_manifest_temp(
    admission: &WriterAdmissionInner,
    generation: u64,
) -> Result<(), KeeperError> {
    let source = admission
        .directory
        .join(format!(".tmp-manifest-{generation}"));
    let retired = admission.directory.join(format!(
        ".tmp-abandoned-manifest-{generation}-{:016x}",
        admission.record.pid_start_nonce
    ));
    let changed = retire_regular_artifact(admission, &source, &retired)?;
    let source_sidecar = append_path_suffix(&source, ".fec");
    let retired_sidecar = append_path_suffix(&retired, ".fec");
    let changed = retire_regular_artifact(admission, &source_sidecar, &retired_sidecar)? || changed;
    if changed {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync abandoned MANIFEST temp retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}

fn recover_corrupt_primary_slot(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    let current = admission.directory.join("MANIFEST");
    let previous = admission.directory.join("MANIFEST.prev");
    let current_slot = read_manifest_slot(&current)?;
    let previous_slot = read_manifest_slot(&previous)?;
    if matches!(current_slot, ManifestSlot::Invalid(_))
        && matches!(previous_slot, ManifestSlot::Valid(_))
    {
        let retired = admission.directory.join(format!(
            ".tmp-corrupt-manifest-{:016x}",
            admission.record.pid_start_nonce
        ));
        let changed = retire_regular_artifact(admission, &current, &retired)?;
        let current_sidecar = append_path_suffix(&current, ".fec");
        let retired_sidecar = append_path_suffix(&retired, ".fec");
        let changed =
            retire_regular_artifact(admission, &current_sidecar, &retired_sidecar)? || changed;
        if changed {
            admission
                .directory_file
                .sync_all()
                .map_err(|source| KeeperError::Io {
                    operation: "fsync corrupt MANIFEST retirement",
                    path: admission.directory.clone(),
                    source,
                })?;
        }
        return Ok(());
    }

    // A crash between the two renames above leaves MANIFEST retired while its
    // sidecar remains canonical; complete that interrupted retirement once.
    if matches!(current_slot, ManifestSlot::Missing)
        && matches!(previous_slot, ManifestSlot::Valid(_))
    {
        recover_interrupted_corrupt_retirement(admission)?;
    }
    Ok(())
}

/// Complete an interrupted corrupt-primary retirement (bd-qxce).
///
/// Corrupt-primary retirement performs two renames — `MANIFEST` into its
/// no-replace quarantine, then `MANIFEST.fec` — followed by one directory
/// fsync. A crash between the renames leaves the sidecar at its canonical
/// name while the `.tmp-corrupt-manifest-*` quarantine records the deliberate
/// retirement. Future durable opens would otherwise reconsider that orphan
/// sidecar on every admission, repeating the repair attempt the retiring
/// writer already rejected. When the retirement evidence exists, retire the
/// orphan sidecar once, deterministically, into its own no-replace
/// quarantine. A canonical sidecar without retirement evidence is a
/// legitimate crash survivor and is left untouched.
fn recover_interrupted_corrupt_retirement(
    admission: &WriterAdmissionInner,
) -> Result<(), KeeperError> {
    let sidecar = admission.directory.join("MANIFEST.fec");
    let sidecar_exists = match std::fs::symlink_metadata(&sidecar) {
        Ok(metadata) if metadata.file_type().is_file() => true,
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "inspect interrupted-retirement sidecar",
                path: sidecar.clone(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "interrupted-retirement sidecar is not a regular file",
                ),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect interrupted-retirement sidecar",
                path: sidecar.clone(),
                source: error,
            });
        }
    };
    if !sidecar_exists {
        return Ok(());
    }
    let mut retirement_evidence = false;
    let entries = std::fs::read_dir(&admission.directory).map_err(|source| KeeperError::Io {
        operation: "scan for corrupt-retirement evidence",
        path: admission.directory.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read corrupt-retirement evidence entry",
            path: admission.directory.clone(),
            source,
        })?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(".tmp-corrupt-manifest-"))
        {
            retirement_evidence = true;
            break;
        }
    }
    if !retirement_evidence {
        return Ok(());
    }
    let retired = admission.directory.join(format!(
        ".tmp-corrupt-manifest-fec-{:016x}",
        admission.record.pid_start_nonce
    ));
    if retire_regular_artifact(admission, &sidecar, &retired)? {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync interrupted corrupt-sidecar retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn recover_durable_manifest_slots(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
    unrepairable: UnrepairableSegmentPolicy,
) -> Result<(), KeeperError> {
    let current_path = admission.directory.join("MANIFEST");
    let previous_path = admission.directory.join("MANIFEST.prev");
    let expected_schema = schema
        .schema_id()
        .map_err(|source| KeeperError::InvalidSchema { source })?;

    let (mut current, mut current_repair, current_recovery_error) =
        durable_manifest_candidate(protector, &current_path, expected_schema)?.into_parts();
    let (mut previous, mut previous_repair, previous_recovery_error) =
        durable_manifest_candidate(protector, &previous_path, expected_schema)?.into_parts();
    if current.is_none() && previous.is_none() {
        if let Some(error) = current_recovery_error {
            return Err(*error);
        }
        if let Some(error) = previous_recovery_error {
            return Err(*error);
        }
    }
    if let (Some(current_manifest), Some(previous_manifest)) = (&current, &previous) {
        let pair = validate_manifest_pair(
            &admission.directory,
            current_manifest.clone(),
            previous_manifest,
        );
        if let Err(error) = pair {
            match (current_repair.is_some(), previous_repair.is_some()) {
                // Two pre-existing valid slots disagree: retain the existing
                // fail-closed corruption signal.
                (false, false) => return Err(error),
                // Reconstructed current is not allowed to displace the usable
                // pre-existing fallback when its transition is inconsistent.
                (true, false) => {
                    current = None;
                    current_repair = None;
                }
                // A reconstructed previous is optional while a pre-existing
                // current is usable. Leave the invalid on-disk slot untouched.
                (false, true) => {
                    previous = None;
                    previous_repair = None;
                }
                // If neither slot existed as valid authority, an inconsistent
                // pair of authenticated reconstructions has no safe winner.
                // Stale or swapped sidecars can make either generation look
                // plausible in isolation, so preserve the fail-closed signal.
                (true, true) => return Err(error),
            }
        }
    }
    // A reconstructed MANIFEST is not authority until every referenced
    // segment is readable or has itself been reconstructed and validated.
    // Prefer current, but never let an unusable optional fallback block a
    // usable current or let an unusable current displace a usable previous.
    match (&current, &previous) {
        (Some(current_manifest), _) => {
            match recover_durable_manifest_segments(
                admission,
                schema,
                protector,
                current_manifest,
                unrepairable,
            ) {
                Ok(()) => {
                    // Install a reconstructed fallback only when its own
                    // dependencies are sound. It is optional while current is
                    // usable, so a bad previous-only segment leaves the prior
                    // slot untouched instead of bricking writer admission.
                    if let (Some(bytes), Some(previous_manifest)) =
                        (previous_repair.as_deref(), previous.as_ref())
                        && recover_durable_manifest_segments(
                            admission,
                            schema,
                            protector,
                            previous_manifest,
                            unrepairable,
                        )
                        .is_ok()
                    {
                        install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
                    }
                    if let Some(bytes) = &current_repair {
                        install_recovered_bytes(admission, "MANIFEST", &current_path, bytes)?;
                    }
                }
                Err(current_error) => {
                    // A pre-existing valid current was already reader-visible;
                    // do not mask that corruption here. A reconstructed current
                    // can instead be skipped when previous is usable.
                    if current_repair.is_none() {
                        return Err(current_error);
                    }
                    let Some(previous_manifest) = &previous else {
                        return Err(current_error);
                    };
                    recover_durable_manifest_segments(
                        admission,
                        schema,
                        protector,
                        previous_manifest,
                        unrepairable,
                    )?;
                    if let Some(bytes) = &previous_repair {
                        install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
                    }
                }
            }
        }
        (None, Some(previous_manifest)) => {
            recover_durable_manifest_segments(
                admission,
                schema,
                protector,
                previous_manifest,
                unrepairable,
            )?;
            if let Some(bytes) = &previous_repair {
                install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
            }
        }
        (None, None) => {}
    }

    for path in [&current_path, &previous_path] {
        if let ManifestSlot::Valid(_) = read_manifest_slot(path)? {
            let bytes = std::fs::read(path).map_err(|source| KeeperError::Io {
                operation: "read valid MANIFEST for sidecar witness",
                path: path.clone(),
                source,
            })?;
            ensure_matching_durability_sidecar(admission, protector, path, &bytes)?;
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
enum DurableManifestCandidate {
    Unavailable,
    Existing(Manifest),
    Recovered { manifest: Manifest, bytes: Vec<u8> },
    RecoveryFailed(Box<KeeperError>),
}

#[cfg(feature = "durability")]
impl DurableManifestCandidate {
    fn into_parts(self) -> (Option<Manifest>, Option<Vec<u8>>, Option<Box<KeeperError>>) {
        match self {
            Self::Unavailable => (None, None, None),
            Self::Existing(manifest) => (Some(manifest), None, None),
            Self::Recovered { manifest, bytes } => (Some(manifest), Some(bytes), None),
            Self::RecoveryFailed(error) => (None, None, Some(error)),
        }
    }
}

#[cfg(feature = "durability")]
fn durable_manifest_candidate(
    protector: &FileProtector,
    path: &Path,
    expected_schema: u64,
) -> Result<DurableManifestCandidate, KeeperError> {
    if let ManifestSlot::Valid(manifest) = read_manifest_slot(path)? {
        if manifest.schema_id != expected_schema {
            return Err(KeeperError::SchemaMismatch {
                path: path.to_path_buf(),
                expected: expected_schema,
                found: manifest.schema_id,
            });
        }
        return Ok(DurableManifestCandidate::Existing(manifest));
    }

    let recovered = match recover_manifest_bytes(protector, path) {
        Ok(recovered) => recovered,
        Err(error) => {
            return Ok(DurableManifestCandidate::RecoveryFailed(Box::new(error)));
        }
    };
    let Some((recovered, bytes)) = recovered else {
        return Ok(DurableManifestCandidate::Unavailable);
    };
    if recovered.schema_id != expected_schema {
        return Ok(DurableManifestCandidate::RecoveryFailed(Box::new(
            KeeperError::SchemaMismatch {
                path: path.to_path_buf(),
                expected: expected_schema,
                found: recovered.schema_id,
            },
        )));
    }
    Ok(DurableManifestCandidate::Recovered {
        manifest: recovered,
        bytes,
    })
}

#[cfg(feature = "durability")]
fn recover_manifest_bytes(
    protector: &FileProtector,
    source: &Path,
) -> Result<Option<(Manifest, Vec<u8>)>, KeeperError> {
    let sidecar = FileProtector::sidecar_path(source);
    if !regular_artifact_exists(&sidecar, "inspect MANIFEST repair sidecar")? {
        return Ok(None);
    }
    match protector
        .recover_file_bytes(source, &sidecar)
        .map_err(|source_error| KeeperError::Durability {
            operation: "recover MANIFEST bytes",
            path: source.to_path_buf(),
            source: source_error,
        })? {
        FileRecoveryOutcome::Recovered { bytes, .. } => {
            let witness = FileSourceWitness::from_bytes(&bytes);
            if !protector
                .sidecar_matches_witness(&sidecar, witness)
                .map_err(|source_error| KeeperError::Durability {
                    operation: "validate staged MANIFEST repair witness",
                    path: sidecar.clone(),
                    source: source_error,
                })?
            {
                return Err(KeeperError::Io {
                    operation: "validate staged MANIFEST repair witness",
                    path: source.to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidData,
                        "repaired MANIFEST does not match its complete-source sidecar",
                    ),
                });
            }
            let manifest = Manifest::from_bytes(&bytes).map_err(|source_error| {
                KeeperError::ManifestCorrupted {
                    path: source.to_path_buf(),
                    source: source_error,
                }
            })?;
            Ok(Some((manifest, bytes)))
        }
        FileRecoveryOutcome::NotNeeded | FileRecoveryOutcome::Unrecoverable { .. } => Ok(None),
    }
}

#[cfg(feature = "durability")]
fn recover_durable_writer_files(
    admission: &Arc<WriterAdmissionInner>,
    schema: SchemaDescriptor,
    protector: &FileProtector,
    unrepairable: UnrepairableSegmentPolicy,
) -> Result<(), KeeperError> {
    let expected_schema = schema
        .schema_id()
        .map_err(|source| KeeperError::InvalidSchema { source })?;
    let loaded = match load_manifest_pair(&admission.directory) {
        Ok(loaded) => loaded,
        Err(KeeperError::IndexNotFound { .. }) => return Ok(()),
        Err(error) => return Err(error),
    };
    validate_loaded_schema(&admission.directory, expected_schema, &loaded)?;
    validate_recovery_claims(&admission.directory, &loaded)?;

    let mut quarantined = Vec::new();
    for manifest_segment in &loaded.manifest.segments {
        match recover_durable_segment(admission, schema, protector, manifest_segment)? {
            DurableSegmentRecovery::Healthy => {}
            DurableSegmentRecovery::Unrepairable { error } => {
                if unrepairable == UnrepairableSegmentPolicy::FailClosed
                    || !segment_has_quarantine_source(admission, manifest_segment.segment_id)?
                {
                    return Err(error);
                }
                quarantined.push((manifest_segment.clone(), error.to_string()));
            }
        }
    }
    if quarantined.is_empty() {
        return Ok(());
    }

    let quarantined_ids = quarantined
        .iter()
        .map(|(segment, _)| segment.segment_id)
        .collect::<BTreeSet<_>>();
    let retained = loaded
        .manifest
        .segments
        .iter()
        .filter(|segment| !quarantined_ids.contains(&segment.segment_id))
        .cloned()
        .collect::<Vec<_>>();
    let field_stats = recompute_manifest_field_stats(&admission.directory, schema, &retained)?;

    for (segment, _) in &quarantined {
        quarantine_segment_artifacts(admission, segment.segment_id)?;
    }
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync quarantined segment retirement",
            path: admission.directory.clone(),
            source,
        })?;

    let mut successor = loaded.manifest.clone();
    successor.generation =
        successor
            .generation
            .checked_add(1)
            .ok_or(KeeperError::GenerationExhausted {
                current: successor.generation,
            })?;
    successor.last_publish_unix_s = 0;
    successor.segments = retained;
    successor.field_stats = field_stats;
    let bytes = successor
        .to_bytes()
        .map_err(|source| KeeperError::InvalidManifest { source })?;
    let claim_admission = Arc::clone(admission);
    publish_manifest_durable_choreography(
        admission.directory.clone(),
        &bytes,
        move |_, generation| GenerationClaimGuard::acquire(claim_admission, generation),
        protector,
        |_, _| Ok(()),
    )?;

    for (segment, reason) in quarantined {
        tracing::warn!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.keeper.segment_quarantined",
            directory = %admission.directory.display(),
            segment_id = format_args!("{:#018x}", segment.segment_id),
            estimated_missing_docs = segment.doc_count,
            generation = successor.generation,
            reason,
            freshness_audit_required = true,
            "unrepairable Quill segment retained in quarantine and omitted from the active MANIFEST"
        );
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn recover_durable_manifest_segments(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
    manifest: &Manifest,
    unrepairable: UnrepairableSegmentPolicy,
) -> Result<(), KeeperError> {
    for manifest_segment in &manifest.segments {
        match recover_durable_segment(admission, schema, protector, manifest_segment)? {
            DurableSegmentRecovery::Healthy => {}
            DurableSegmentRecovery::Unrepairable { error } => {
                if unrepairable == UnrepairableSegmentPolicy::Quarantine
                    && segment_has_quarantine_source(admission, manifest_segment.segment_id)?
                {
                    continue;
                }
                return Err(error);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
enum DurableSegmentRecovery {
    Healthy,
    Unrepairable { error: KeeperError },
}

#[cfg(feature = "durability")]
fn recover_durable_segment(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
    manifest_segment: &ManifestSegment,
) -> Result<DurableSegmentRecovery, KeeperError> {
    let path = admission
        .directory
        .join(canonical_segment_name(manifest_segment.segment_id));
    let original_error = match open_verified_segment(&path, manifest_segment, schema) {
        Ok(()) => {
            let bytes = std::fs::read(&path).map_err(|source| KeeperError::Io {
                operation: "read valid segment for sidecar witness",
                path: path.clone(),
                source,
            })?;
            ensure_matching_durability_sidecar(admission, protector, &path, &bytes)?;
            return Ok(DurableSegmentRecovery::Healthy);
        }
        Err(error) => error,
    };

    let sidecar = FileProtector::sidecar_path(&path);
    if !regular_artifact_exists(&sidecar, "inspect segment repair sidecar")? {
        return Ok(DurableSegmentRecovery::Unrepairable {
            error: original_error,
        });
    }
    let recovered = match protector.recover_file_bytes(&path, &sidecar) {
        Ok(FileRecoveryOutcome::Recovered { bytes, .. }) => bytes,
        Ok(FileRecoveryOutcome::NotNeeded | FileRecoveryOutcome::Unrecoverable { .. }) => {
            return Ok(DurableSegmentRecovery::Unrepairable {
                error: original_error,
            });
        }
        Err(source @ SearchError::Io(_)) => {
            return Err(KeeperError::Durability {
                operation: "recover segment bytes",
                path,
                source,
            });
        }
        Err(_) => {
            return Ok(DurableSegmentRecovery::Unrepairable {
                error: original_error,
            });
        }
    };
    let witness = FileSourceWitness::from_bytes(&recovered);
    let sidecar_matches = match protector.sidecar_matches_witness(&sidecar, witness) {
        Ok(matches) => matches,
        Err(source @ SearchError::Io(_)) => {
            return Err(KeeperError::Durability {
                operation: "validate staged segment repair witness",
                path: sidecar,
                source,
            });
        }
        Err(_) => false,
    };
    if !sidecar_matches {
        return Ok(DurableSegmentRecovery::Unrepairable {
            error: original_error,
        });
    }
    let reader = match SegmentReader::from_bytes(&recovered, schema) {
        Ok(reader) => reader,
        Err(source) => {
            return Ok(DurableSegmentRecovery::Unrepairable {
                error: KeeperError::SegmentOpen {
                    path: path.clone(),
                    source,
                },
            });
        }
    };
    if let Err(source) = reader.verify() {
        return Ok(DurableSegmentRecovery::Unrepairable {
            error: KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            },
        });
    }
    if let Err(error) = validate_segment_witnesses(&path, manifest_segment, &reader) {
        return Ok(DurableSegmentRecovery::Unrepairable { error });
    }
    let label = format!("segment-{:016x}", manifest_segment.segment_id);
    install_recovered_bytes(admission, &label, &path, &recovered)?;
    open_verified_segment(&path, manifest_segment, schema)?;
    Ok(DurableSegmentRecovery::Healthy)
}

#[cfg(feature = "durability")]
fn segment_has_quarantine_source(
    admission: &WriterAdmissionInner,
    segment_id: u64,
) -> Result<bool, KeeperError> {
    let path = admission.directory.join(canonical_segment_name(segment_id));
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() => return Ok(true),
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "inspect quarantine candidate",
                path,
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "quarantine candidate is not a regular file",
                ),
            });
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect quarantine candidate",
                path,
                source,
            });
        }
    }
    find_quarantined_segment_path(&admission.directory, segment_id).map(|path| path.is_some())
}

#[cfg(feature = "durability")]
fn find_quarantined_segment_path(
    directory: &Path,
    segment_id: u64,
) -> Result<Option<PathBuf>, KeeperError> {
    for entry in std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan segment quarantine witnesses",
        path: directory.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read segment quarantine witness",
            path: directory.to_path_buf(),
            source,
        })?;
        let matches = entry
            .file_name()
            .to_str()
            .and_then(parse_quarantined_segment_name)
            == Some(segment_id);
        if !matches {
            continue;
        }
        let metadata = entry.metadata().map_err(|source| KeeperError::Io {
            operation: "inspect segment quarantine witness",
            path: entry.path(),
            source,
        })?;
        if !metadata.file_type().is_file() {
            return Err(KeeperError::Io {
                operation: "inspect segment quarantine witness",
                path: entry.path(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "segment quarantine witness is not a regular file",
                ),
            });
        }
        return Ok(Some(entry.path()));
    }
    Ok(None)
}

#[cfg(feature = "durability")]
fn quarantine_segment_artifacts(
    admission: &WriterAdmissionInner,
    segment_id: u64,
) -> Result<PathBuf, KeeperError> {
    let path = admission.directory.join(canonical_segment_name(segment_id));
    let quarantine = append_path_suffix(&path, ".quarantine");
    let retained =
        if let Some(existing) = find_quarantined_segment_path(&admission.directory, segment_id)? {
            existing
        } else {
            if !retire_regular_artifact(admission, &path, &quarantine)? {
                return Err(KeeperError::Io {
                    operation: "quarantine unrepairable segment",
                    path: path.clone(),
                    source: io::Error::new(
                        io::ErrorKind::NotFound,
                        "unrepairable segment disappeared before quarantine",
                    ),
                });
            }
            find_quarantined_segment_path(&admission.directory, segment_id)?.ok_or_else(|| {
                KeeperError::Io {
                    operation: "locate retained segment quarantine",
                    path: quarantine,
                    source: io::Error::new(
                        io::ErrorKind::NotFound,
                        "segment quarantine rename completed without a discoverable destination",
                    ),
                }
            })?
        };
    let sidecar = FileProtector::sidecar_path(&path);
    let sidecar_quarantine = append_path_suffix(&sidecar, ".quarantine");
    let _ = retire_regular_artifact(admission, &sidecar, &sidecar_quarantine)?;
    Ok(retained)
}

#[cfg(feature = "durability")]
fn recompute_manifest_field_stats(
    directory: &Path,
    schema: SchemaDescriptor,
    segments: &[ManifestSegment],
) -> Result<Vec<ManifestFieldStats>, KeeperError> {
    let term_fields =
        concat_term_field_ords(schema).map_err(|source| KeeperError::SegmentMetadataMismatch {
            path: directory.to_path_buf(),
            detail: source.to_string(),
        })?;
    let mut sections = Vec::new();
    sections
        .try_reserve_exact(segments.len())
        .map_err(|source| KeeperError::Io {
            operation: "allocate quarantine field-stat sources",
            path: directory.to_path_buf(),
            source: io::Error::other(source.to_string()),
        })?;
    for segment in segments {
        let path = directory.join(canonical_segment_name(segment.segment_id));
        let reader = SegmentReader::open_published(&path, schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        reader.verify().map_err(|source| KeeperError::SegmentOpen {
            path: path.clone(),
            source,
        })?;
        validate_segment_witnesses(&path, segment, &reader)?;
        let stats = reader
            .section(SectionKind::STATS)
            .map_err(|source| KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            })?
            .ok_or_else(|| KeeperError::MissingIdentitySection {
                path: path.clone(),
                kind: SectionKind::STATS,
            })?;
        sections.push(
            StatsSection::parse(stats, &term_fields, segment.doc_count).map_err(|source| {
                KeeperError::SegmentMetadataMismatch {
                    path: path.clone(),
                    detail: source.to_string(),
                }
            })?,
        );
    }
    aggregate_field_stats(sections.iter())
        .map_err(|source| KeeperError::SegmentMetadataMismatch {
            path: directory.to_path_buf(),
            detail: source.to_string(),
        })?
        .into_iter()
        .map(|stats| {
            let doc_count = u32::try_from(stats.doc_count).map_err(|_| {
                KeeperError::SegmentMetadataMismatch {
                    path: directory.to_path_buf(),
                    detail: format!(
                        "quarantine field {} document count {} exceeds u32",
                        stats.field_ord, stats.doc_count
                    ),
                }
            })?;
            Ok(ManifestFieldStats {
                field_ord: stats.field_ord,
                total_tokens: stats.total_tokens,
                doc_count,
            })
        })
        .collect()
}

#[cfg(feature = "durability")]
fn open_verified_segment(
    path: &Path,
    manifest: &ManifestSegment,
    schema: SchemaDescriptor,
) -> Result<(), KeeperError> {
    let reader =
        SegmentReader::open_published(path, schema).map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })?;
    reader.verify().map_err(|source| KeeperError::SegmentOpen {
        path: path.to_path_buf(),
        source,
    })?;
    validate_segment_witnesses(path, manifest, &reader)
}

#[cfg(feature = "durability")]
fn ensure_matching_durability_sidecar(
    admission: &WriterAdmissionInner,
    protector: &FileProtector,
    source: &Path,
    bytes: &[u8],
) -> Result<(), KeeperError> {
    let witness = FileSourceWitness::from_bytes(bytes);
    let sidecar = FileProtector::sidecar_path(source);
    let sidecar_exists = regular_artifact_exists(&sidecar, "inspect durability sidecar")?;
    let sidecar_matches = sidecar_exists
        && protector
            .sidecar_matches_witness(&sidecar, witness)
            .unwrap_or(false);
    if sidecar_matches {
        return Ok(());
    }
    if sidecar_exists {
        let file_name = source
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("artifact");
        let retired = admission.directory.join(format!(
            ".tmp-stale-fec-{file_name}-{:016x}",
            admission.record.pid_start_nonce
        ));
        if retire_regular_artifact(admission, &sidecar, &retired)? {
            admission
                .directory_file
                .sync_all()
                .map_err(|source_error| KeeperError::Io {
                    operation: "fsync stale sidecar retirement",
                    path: admission.directory.clone(),
                    source: source_error,
                })?;
        }
    }
    protector
        .protect_file_with_witness(source, witness)
        .map_err(|source_error| KeeperError::Durability {
            operation: "regenerate durability sidecar",
            path: source.to_path_buf(),
            source: source_error,
        })?;
    if !protector
        .sidecar_matches_witness(&sidecar, witness)
        .map_err(|source_error| KeeperError::Durability {
            operation: "verify regenerated durability sidecar",
            path: sidecar.clone(),
            source: source_error,
        })?
    {
        return Err(KeeperError::Io {
            operation: "verify regenerated durability sidecar",
            path: sidecar,
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "regenerated sidecar does not match authoritative source bytes",
            ),
        });
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn regular_artifact_exists(path: &Path, operation: &'static str) -> Result<bool, KeeperError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(KeeperError::Io {
            operation,
            path: path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidData, "artifact is not a regular file"),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(KeeperError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(all(
    feature = "durability",
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
fn install_recovered_bytes(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), KeeperError> {
    install_recovered_bytes_with_observer(admission, label, destination, bytes, &mut |_, _| Ok(()))
}

// Supported target matrix for durable Keeper recovery (bd-b188).
//
// - **Tier 1** (`linux`, `android`, `target_vendor = "apple"`): descriptor-
//   bound installation with `renameat2(EXCHANGE)`, inode-verified rollback,
//   and `renameat2(NOREPLACE)` retirement — the strongest no-replace
//   semantics, exercised by the `recovered_byte_install_*` host tests and
//   the CI ubuntu/macos lanes.
// - **Tier 2** (every other unix, Windows, and targets with POSIX/NTFS hard
//   links): the portable `std::fs` choreography in
//   [`install_recovered_bytes_portable`] / [`retire_regular_artifact_portable`]
//   — `create_new` collision probes, atomic no-replace hard-link install,
//   quarantine-first retirement. Always compiled so the `portable_*` host
//   tests pin the exact logic the fallback delegates to.
// - **Tier 3** (filesystems without hard links, e.g. FAT32, and targets
//   without the required file APIs, e.g. wasi): explicit typed
//   `Unsupported` rejection — never a racy check-then-rename or
//   copy-based fallback.

/// Deterministic crash-window checkpoints in recovered-byte installation
/// (bd-qxce). Every durable side effect boundary reports here so the crash
/// matrix can suspend the choreography at each step and prove recovery.
#[cfg(feature = "durability")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveredByteInstallCheckpoint {
    /// The recovered-byte temp file was written and fsynced.
    TempSynced,
    /// The descriptor-bound ready link was created and the directory fsynced.
    StagingSynced,
    /// Immediately before the atomic link/exchange into the canonical name.
    BeforeAtomicInstall,
    /// The canonical name now names the recovered inode.
    AfterAtomicInstall,
    /// A detected substitution was rolled back by the reverse exchange.
    AfterRollback,
    /// The rollback directory fsync completed.
    AfterRollbackSync,
    /// The replaced corrupt authority was retired to its no-replace quarantine.
    AfterCorruptRetirement,
    /// The final installation directory fsync completed.
    FinalSynced,
}

#[cfg(all(
    feature = "durability",
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
fn install_recovered_bytes_with_observer(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
    observe: &mut impl FnMut(RecoveredByteInstallCheckpoint, &Path) -> io::Result<()>,
) -> Result<(), KeeperError> {
    use rustix::fs::{
        AtFlags, CWD, FileType, Mode, OFlags, RenameFlags, linkat, openat, renameat_with, statat,
    };
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    admission.ensure_directory_identity()?;
    if destination.parent() != Some(admission.directory.as_path()) {
        return Err(KeeperError::Io {
            operation: "validate recovered-byte destination",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "repair destination is outside the admitted index directory",
            ),
        });
    }
    let expected_stat_size = i64::try_from(bytes.len()).map_err(|_| KeeperError::Io {
        operation: "validate recovered-byte length",
        path: destination.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidInput,
            "recovered bytes exceed the platform file-size range",
        ),
    })?;
    let destination_name =
        destination
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: destination.to_path_buf(),
            })?;

    let base = format!(
        ".tmp-repair-{label}-{:016x}",
        admission.record.pid_start_nonce
    );
    let mut collision = 0_u64;
    let (temporary_name, mut temporary_file, temporary_metadata) = loop {
        let name = if collision == 0 {
            OsString::from(&base)
        } else {
            OsString::from(format!("{base}.{collision}"))
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        match openat(
            &admission.directory_file,
            &name,
            OFlags::WRONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR | Mode::RGRP | Mode::WGRP | Mode::ROTH | Mode::WOTH,
        ) {
            Ok(file) => {
                let file = File::from(file);
                let metadata = file.metadata().map_err(|source| KeeperError::Io {
                    operation: "inspect owned recovered-byte temp",
                    path: admission.directory.join(&name),
                    source,
                })?;
                break (name, file, metadata);
            }
            Err(source) if source == rustix::io::Errno::EXIST => {
                collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                    operation: "allocate recovered-byte temp",
                    path: admission.directory.join(&base),
                    source: io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "recovered-byte temp suffix space is exhausted",
                    ),
                })?;
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create recovered-byte temp",
                    path: admission.directory.join(&name),
                    source: io::Error::from(source),
                });
            }
        }
    };
    let temporary_path = admission.directory.join(&temporary_name);
    temporary_file
        .write_all(bytes)
        .and_then(|()| temporary_file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist recovered-byte temp",
            path: temporary_path.clone(),
            source,
        })?;
    observe(RecoveredByteInstallCheckpoint::TempSynced, &temporary_path).map_err(|source| {
        KeeperError::Io {
            operation: "run recovered-byte install checkpoint",
            path: temporary_path.clone(),
            source,
        }
    })?;
    if !temporary_metadata.file_type().is_file()
        || temporary_file
            .metadata()
            .map_err(|source| KeeperError::Io {
                operation: "reinspect recovered-byte temp",
                path: temporary_path.clone(),
                source,
            })?
            .len()
            != usize_to_u64(bytes.len())
    {
        return Err(KeeperError::Io {
            operation: "validate recovered-byte temp",
            path: temporary_path,
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "recovered-byte temp is not a regular file with the expected length",
            ),
        });
    }
    let descriptor_root = if cfg!(target_vendor = "apple") {
        "/dev/fd"
    } else {
        "/proc/self/fd"
    };
    let descriptor_path =
        PathBuf::from(format!("{descriptor_root}/{}", temporary_file.as_raw_fd()));
    let ready_base = format!(
        ".tmp-ready-repair-{label}-{:016x}",
        admission.record.pid_start_nonce
    );
    let mut ready_collision = 0_u64;
    let ready_name = loop {
        let name = if ready_collision == 0 {
            OsString::from(&ready_base)
        } else {
            OsString::from(format!("{ready_base}.{ready_collision}"))
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        match linkat(
            CWD,
            &descriptor_path,
            &admission.directory_file,
            &name,
            AtFlags::SYMLINK_FOLLOW,
        ) {
            Ok(()) => break name,
            Err(error) if error == rustix::io::Errno::EXIST => {
                ready_collision =
                    ready_collision
                        .checked_add(1)
                        .ok_or_else(|| KeeperError::Io {
                            operation: "allocate recovered-byte ready link",
                            path: admission.directory.join(&ready_base),
                            source: io::Error::new(
                                io::ErrorKind::AlreadyExists,
                                "recovered-byte ready-link suffix space is exhausted",
                            ),
                        })?;
            }
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "create descriptor-bound recovered-byte ready link",
                    path: admission.directory.join(&name),
                    source: io::Error::from(error),
                });
            }
        }
    };
    let ready_path = admission.directory.join(&ready_name);
    let ready_stat = statat(
        &admission.directory_file,
        &ready_name,
        AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "verify recovered-byte ready link",
        path: ready_path.clone(),
        source,
    })?;
    if ready_stat.st_dev != temporary_metadata.dev()
        || ready_stat.st_ino != temporary_metadata.ino()
        || ready_stat.st_size != expected_stat_size
    {
        return Err(KeeperError::Io {
            operation: "verify recovered-byte ready link",
            path: ready_path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "recovered-byte ready link does not name the owned inode",
            ),
        });
    }
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync recovered-byte staging links",
            path: admission.directory.clone(),
            source,
        })?;
    observe(
        RecoveredByteInstallCheckpoint::StagingSynced,
        &admission.directory,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: admission.directory.clone(),
        source,
    })?;

    let destination_identity = match statat(
        &admission.directory_file,
        destination_name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile => {
            Some((stat.st_dev, stat.st_ino, stat.st_size))
        }
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "inspect recovered-byte destination",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "recovered-byte destination is not a regular file",
                ),
            });
        }
        Err(error) if error == rustix::io::Errno::NOENT => None,
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect recovered-byte destination",
                path: destination.to_path_buf(),
                source: io::Error::from(error),
            });
        }
    };
    observe(
        RecoveredByteInstallCheckpoint::BeforeAtomicInstall,
        &ready_path,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: ready_path.clone(),
        source,
    })?;
    let destination_exists = destination_identity.is_some();
    if destination_exists {
        // Threat model: writer admission excludes cooperating directory
        // mutators. A same-directory process that ignores LOCK can replace the
        // ready pathname; the inode check below detects and rolls that back
        // during an uninterrupted call, but POSIX has no inode-conditioned
        // exchange primitive that also makes such hostile substitution
        // crash-safe. Non-cooperating mutation of an admitted index directory
        // is therefore outside Keeper's crash-consistency contract.
        renameat_with(
            &admission.directory_file,
            &ready_name,
            &admission.directory_file,
            destination_name,
            RenameFlags::EXCHANGE,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "atomically exchange recovered bytes",
            path: destination.to_path_buf(),
            source,
        })?;
    } else {
        // With no canonical destination, link the still-open validated inode
        // directly into place. The ready pathname is deliberately not the
        // source, so substitution of that staging name cannot affect the bytes
        // installed at the canonical name.
        linkat(
            CWD,
            &descriptor_path,
            &admission.directory_file,
            destination_name,
            AtFlags::SYMLINK_FOLLOW,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "link descriptor-bound recovered bytes into place",
            path: destination.to_path_buf(),
            source,
        })?;
    }
    observe(
        RecoveredByteInstallCheckpoint::AfterAtomicInstall,
        destination,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: destination.to_path_buf(),
        source,
    })?;
    let destination_stat = statat(
        &admission.directory_file,
        destination_name,
        AtFlags::SYMLINK_NOFOLLOW,
    );
    let installed_owned_inode = destination_stat.as_ref().is_ok_and(|stat| {
        stat.st_dev == temporary_metadata.dev()
            && stat.st_ino == temporary_metadata.ino()
            && stat.st_size == expected_stat_size
    });
    if !installed_owned_inode {
        if !destination_exists {
            return Err(KeeperError::Io {
                operation: "verify descriptor-bound recovered-byte installation",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "descriptor-bound recovered-byte destination changed after installation",
                ),
            });
        }
        renameat_with(
            &admission.directory_file,
            destination_name,
            &admission.directory_file,
            &ready_name,
            RenameFlags::EXCHANGE,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "roll back substituted recovered-byte installation",
            path: destination.to_path_buf(),
            source,
        })?;
        observe(RecoveredByteInstallCheckpoint::AfterRollback, destination).map_err(|source| {
            KeeperError::Io {
                operation: "run recovered-byte install checkpoint",
                path: destination.to_path_buf(),
                source,
            }
        })?;
        let restored = destination_identity.is_some_and(|(device, inode, size)| {
            statat(
                &admission.directory_file,
                destination_name,
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .is_ok_and(|stat| stat.st_dev == device && stat.st_ino == inode && stat.st_size == size)
        });
        if !restored {
            return Err(KeeperError::Io {
                operation: "verify recovered-byte rollback",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "recovered-byte rollback did not restore the prior authority",
                ),
            });
        }
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync recovered-byte rollback",
                path: admission.directory.clone(),
                source,
            })?;
        observe(
            RecoveredByteInstallCheckpoint::AfterRollbackSync,
            &admission.directory,
        )
        .map_err(|source| KeeperError::Io {
            operation: "run recovered-byte install checkpoint",
            path: admission.directory.clone(),
            source,
        })?;
        return Err(KeeperError::Io {
            operation: "verify installed recovered bytes",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "installed recovered-byte pathname was substituted; prior authority restored",
            ),
        });
    }
    if destination_exists {
        let retired = admission.directory.join(format!(
            ".tmp-corrupt-{label}-{:016x}",
            admission.record.pid_start_nonce
        ));
        retire_regular_artifact(admission, &ready_path, &retired)?;
        observe(
            RecoveredByteInstallCheckpoint::AfterCorruptRetirement,
            &retired,
        )
        .map_err(|source| KeeperError::Io {
            operation: "run recovered-byte install checkpoint",
            path: retired.clone(),
            source,
        })?;
    }
    // Keep the exclusively-created temp name as a second link until ordinary
    // grace-period garbage collection retires it. Conditional unlink is not
    // available on Unix; leaving the temp avoids a stat-then-unlink race that
    // could remove a substituted pathname while consuming no extra data blocks.
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync staged repair installation",
            path: admission.directory.clone(),
            source,
        })?;
    observe(
        RecoveredByteInstallCheckpoint::FinalSynced,
        &admission.directory,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: admission.directory.clone(),
        source,
    })
}

#[cfg(all(
    feature = "durability",
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn install_recovered_bytes(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), KeeperError> {
    install_recovered_bytes_portable(admission, label, destination, bytes, &mut |_, _| Ok(()))
}

/// Tier-2 recovered-byte installation for platforms without descriptor-bound
/// rename exchange (bd-b188), using only `std::fs` primitives. Always
/// compiled so host tests exercise the same choreography the cfg-selected
/// fallback delegates to outside linux/android/apple.
///
/// Choreography (same checkpoint contract as the descriptor-bound tier):
/// 1. `create_new` temp probe — atomically collision-safe, never overwrites
///    an occupied name (NOFOLLOW on unix; the cooperative LOCK contract
///    covers symlink-race hardening elsewhere, matching tier 1's threat
///    model).
/// 2. write + file fsync; hard-link the temp into place with
///    `std::fs::hard_link` — atomic and no-replace on both POSIX and NTFS.
/// 3. An occupied destination is first retired to a probed no-replace
///    quarantine name (the probe claims the name atomically; the rename then
///    replaces only our own probe), keeping the corrupt authority available
///    and never clobbering a foreign quarantine.
/// 4. Directory fsync ordering preserved on unix; on targets without
///    directory fsync the operation is skipped (file-level fsync already
///    landed before the link).
///
/// Crash between retirement and link leaves the canonical name absent with
/// the corrupt copy quarantined and the validated temp present — durable
/// open re-runs recovery from the sidecar, so the choreography stays
/// fail-closed. Filesystems without hard links report `Unsupported`
/// explicitly (Tier 3) rather than degrading to racy copy fallbacks.
#[cfg(feature = "durability")]
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
fn install_recovered_bytes_portable(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
    observe: &mut impl FnMut(RecoveredByteInstallCheckpoint, &Path) -> io::Result<()>,
) -> Result<(), KeeperError> {
    admission.ensure_directory_identity()?;
    if destination.parent() != Some(admission.directory.as_path()) {
        return Err(KeeperError::Io {
            operation: "validate recovered-byte destination",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "repair destination is outside the admitted index directory",
            ),
        });
    }

    // 1. Collision-safe temp probe (atomically claims its name).
    let (temp_path, mut temp_file) = create_probed_file(
        admission,
        &format!(
            ".tmp-repair-{label}-{:016x}",
            admission.record.pid_start_nonce
        ),
    )?;
    temp_file
        .write_all(bytes)
        .and_then(|()| temp_file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist recovered-byte temp",
            path: temp_path.clone(),
            source,
        })?;
    observe(RecoveredByteInstallCheckpoint::TempSynced, &temp_path).map_err(|source| {
        KeeperError::Io {
            operation: "run recovered-byte install checkpoint",
            path: temp_path.clone(),
            source,
        }
    })?;

    // 2. Retire an occupied canonical destination to a no-replace quarantine.
    let destination_exists = match std::fs::symlink_metadata(destination) {
        Ok(metadata) => {
            if !metadata.file_type().is_file() {
                return Err(KeeperError::Io {
                    operation: "inspect recovered-byte destination",
                    path: destination.to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidData,
                        "recovered-byte destination is not a regular file",
                    ),
                });
            }
            true
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect recovered-byte destination",
                path: destination.to_path_buf(),
                source: error,
            });
        }
    };
    observe(
        RecoveredByteInstallCheckpoint::BeforeAtomicInstall,
        &temp_path,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: temp_path.clone(),
        source,
    })?;
    if destination_exists {
        let (quarantine, quarantine_file) = create_probed_file(
            admission,
            &format!(
                ".retired-repair-{label}-{:016x}",
                admission.record.pid_start_nonce
            ),
        )?;
        // The probe claimed the quarantine name atomically; the rename then
        // replaces only our own probe file, never a foreign quarantine.
        quarantine_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "claim no-replace quarantine name",
                path: quarantine.clone(),
                source,
            })?;
        std::fs::rename(destination, &quarantine).map_err(|source| KeeperError::Io {
            operation: "retire corrupt recovered-byte destination",
            path: quarantine.clone(),
            source,
        })?;
        observe(
            RecoveredByteInstallCheckpoint::AfterCorruptRetirement,
            &quarantine,
        )
        .map_err(|source| KeeperError::Io {
            operation: "run recovered-byte install checkpoint",
            path: quarantine.clone(),
            source,
        })?;
    }

    // 3. Atomically hard-link the validated temp into the canonical name
    // (no-replace by construction on POSIX and NTFS).
    if let Err(source) = std::fs::hard_link(&temp_path, destination) {
        if destination_exists {
            // Best-effort rollback: restore the retired corrupt authority so
            // a failed install never loses the pre-existing bytes. The
            // quarantine name is recomputed from the same probe sequence.
            let quarantine_root = format!(
                ".retired-repair-{label}-{:016x}",
                admission.record.pid_start_nonce
            );
            let quarantine = (0..=u16::MAX)
                .map(|suffix| {
                    if suffix == 0 {
                        admission.directory.join(&quarantine_root)
                    } else {
                        admission
                            .directory
                            .join(format!("{quarantine_root}.{suffix}"))
                    }
                })
                .find(|candidate| candidate.exists());
            if let Some(quarantine) = quarantine {
                let _ = std::fs::rename(&quarantine, destination);
                observe(RecoveredByteInstallCheckpoint::AfterRollback, destination).ok();
            }
        }
        let kind = source.kind();
        return Err(KeeperError::Io {
            operation: "link recovered bytes into place",
            path: destination.to_path_buf(),
            source: if kind == io::ErrorKind::Unsupported || kind == io::ErrorKind::CrossesDevices {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "hard links unsupported on this filesystem; explicit Tier-3 rejection (no racy copy fallback)",
                )
            } else {
                source
            },
        });
    }
    let _ = std::fs::remove_file(&temp_path);
    observe(
        RecoveredByteInstallCheckpoint::AfterAtomicInstall,
        destination,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: destination.to_path_buf(),
        source,
    })?;

    // 4. Post-install validation: the canonical name now holds exactly the
    // validated recovered bytes.
    let installed = std::fs::read(destination).map_err(|source| KeeperError::Io {
        operation: "re-read installed recovered bytes",
        path: destination.to_path_buf(),
        source,
    })?;
    if installed != bytes {
        return Err(KeeperError::Io {
            operation: "validate installed recovered bytes",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "installed recovered bytes diverge from the validated payload",
            ),
        });
    }
    sync_directory_best_effort(&admission.directory).map_err(|source| KeeperError::Io {
        operation: "fsync recovered-byte install directory",
        path: admission.directory.clone(),
        source,
    })?;
    observe(
        RecoveredByteInstallCheckpoint::FinalSynced,
        &admission.directory,
    )
    .map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: admission.directory.clone(),
        source,
    })?;
    Ok(())
}

/// Atomically create a fresh collision-free child file inside the admitted
/// directory with `create_new` (bd-b188): each attempt either claims the
/// name or proves it occupied, so no occupied staging name is ever
/// overwritten. Returns the claimed path and its open file.
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
fn create_probed_file(
    admission: &WriterAdmissionInner,
    base: &str,
) -> Result<(PathBuf, File), KeeperError> {
    for collision in 0..=u16::MAX {
        let name = if collision == 0 {
            base.to_owned()
        } else {
            format!("{base}.{collision}")
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        let candidate = admission.directory.join(&name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "create collision-safe staging file",
                    path: candidate,
                    source: error,
                });
            }
        }
    }
    Err(KeeperError::Io {
        operation: "allocate collision-free staging name",
        path: admission.directory.join(base),
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "staging name suffix space is exhausted",
        ),
    })
}

/// fsync a directory where the platform supports it (unix). On targets
/// without directory fsync this is a documented no-op: every file-level
/// fsync in the choreography already landed before the final link.
#[cfg(feature = "durability")]
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
fn sync_directory_best_effort(directory: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(directory)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = directory;
        Ok(())
    }
}

fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn retire_regular_artifact(
    admission: &WriterAdmissionInner,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    admission.ensure_directory_identity()?;
    if source.parent() != Some(admission.directory.as_path())
        || destination.parent() != Some(admission.directory.as_path())
    {
        return Err(KeeperError::Io {
            operation: "validate retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "retirement paths must be direct children of the admitted directory",
            ),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{AtFlags, FileType, RenameFlags, renameat_with, statat};

        let source_name = source
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: source.to_path_buf(),
            })?;
        let source_stat = match statat(
            &admission.directory_file,
            source_name,
            AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "inspect retirement source",
                    path: source.to_path_buf(),
                    source: io::Error::from(error),
                });
            }
        };
        if FileType::from_raw_mode(source_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::Io {
                operation: "retire writer artifact",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retirement source is not a regular file",
                ),
            });
        }

        let destination_name =
            destination
                .file_name()
                .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                    path: destination.to_path_buf(),
                })?;
        let mut collision = 0_u64;
        loop {
            let candidate = if collision == 0 {
                destination_name.to_os_string()
            } else {
                let mut name = destination_name.to_os_string();
                name.push(format!(".{collision}"));
                name
            };
            match renameat_with(
                &admission.directory_file,
                source_name,
                &admission.directory_file,
                &candidate,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(true),
                Err(error) if error == rustix::io::Errno::EXIST => {
                    collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                        operation: "allocate retirement destination",
                        path: destination.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "retirement destination suffix space is exhausted",
                        ),
                    })?;
                }
                Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
                Err(error) => {
                    return Err(KeeperError::Io {
                        operation: "atomically retire writer artifact",
                        path: admission.directory.join(candidate),
                        source: io::Error::from(error),
                    });
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    retire_regular_artifact_portable(admission, source, destination)
}

/// Tier-2 no-replace retirement for platforms without `renameat2(NOREPLACE)`
/// (bd-b188), using only `std::fs`. Always compiled so host tests exercise
/// the same logic the cfg-selected fallback delegates to.
///
/// The destination suffix is claimed atomically with `create_new` (never an
/// occupied quarantine name), then the source is renamed over OUR OWN probe
/// file — portable across POSIX and Windows, and collision-safe by
/// construction. Returns `Ok(false)` when the source is absent.
#[allow(dead_code)] // called only on Tier-2 targets and by host tests
fn retire_regular_artifact_portable(
    admission: &WriterAdmissionInner,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    admission.ensure_directory_identity()?;
    if source.parent() != Some(admission.directory.as_path())
        || destination.parent() != Some(admission.directory.as_path())
    {
        return Err(KeeperError::Io {
            operation: "validate retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "retirement paths must be direct children of the admitted directory",
            ),
        });
    }
    match std::fs::symlink_metadata(source) {
        Ok(metadata) if metadata.file_type().is_file() => {}
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "retire writer artifact",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retirement source is not a regular file",
                ),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect retirement source",
                path: source.to_path_buf(),
                source: error,
            });
        }
    }

    let base = destination
        .file_name()
        .ok_or_else(|| KeeperError::UnsafeGarbagePath {
            path: destination.to_path_buf(),
        })?
        .to_string_lossy()
        .into_owned();
    let (probe, probe_file) = create_probed_file(admission, &base)?;
    drop(probe_file);
    std::fs::rename(source, &probe).map_err(|source_error| KeeperError::Io {
        operation: "retire writer artifact over claimed probe",
        path: probe,
        source: source_error,
    })?;
    Ok(true)
}

#[cfg(unix)]
fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    use rustix::fs::{AtFlags, Mode, OFlags, openat, statat, unlinkat};
    use std::os::unix::fs::MetadataExt;

    admission.ensure_directory_identity()?;
    let name = path
        .file_name()
        .ok_or_else(|| KeeperError::UnsafeGarbagePath {
            path: path.to_path_buf(),
        })?;
    let claim = openat(
        &admission.directory_file,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    let claim_file = File::from(claim);
    let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != 0 {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim must be a zero-length regular file".to_owned(),
        });
    }
    let stat = statat(&admission.directory_file, name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "revalidate stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    if stat.st_dev != metadata.dev() || stat.st_ino != metadata.ino() || stat.st_size != 0 {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim pathname changed during stale recovery".to_owned(),
        });
    }
    unlinkat(&admission.directory_file, Path::new(name), AtFlags::empty())
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "remove stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync stale-claim removal",
            path: admission.directory.clone(),
            source,
        })
}

#[cfg(not(unix))]
fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "remove stale generation claim",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "stale claim recovery is unsupported for {}",
                admission.directory.display()
            ),
        ),
    })
}

/// Writer-owned garbage collection configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarbageCollectionOptions {
    /// Minimum artifact age required before deletion.
    pub grace_period: Duration,
}

impl Default for GarbageCollectionOptions {
    fn default() -> Self {
        Self {
            grace_period: DEFAULT_GARBAGE_GRACE,
        }
    }
}

/// Deterministic report from one writer-locked garbage sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarbageCollectionReport {
    /// Removed direct-child names, sorted bytewise by platform string order.
    pub removed: Vec<PathBuf>,
}

impl GarbageCollectionReport {
    /// Whether the sweep made no filesystem changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty()
    }
}

/// Load the current MANIFEST, falling back to `MANIFEST.prev` without writing.
///
/// A valid current slot wins. When both slots are valid, the previous
/// generation must equal the current generation (idempotent republish) or be
/// exactly one less. Missing current plus valid previous is the expected
/// between-renames crash window and is not `IndexNotFound`.
///
/// # Errors
///
/// Returns a typed not-found, corruption, invalid-pair, or I/O error.
pub fn load_manifest_pair(directory: impl AsRef<Path>) -> Result<LoadedManifest, KeeperError> {
    let directory = directory.as_ref();
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    let current = match read_manifest_slot(&current_path)? {
        ManifestSlot::Valid(current) => {
            return load_valid_current(directory, &current_path, &previous_path, current);
        }
        current => current,
    };

    let previous = read_manifest_slot(&previous_path)?;
    match (current, previous) {
        (ManifestSlot::Missing, ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterMissingCurrent,
        }),
        (ManifestSlot::Invalid(_), ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterCorruptCurrent,
        }),
        (ManifestSlot::Missing, ManifestSlot::Missing) => Err(KeeperError::IndexNotFound {
            directory: directory.to_path_buf(),
        }),
        (ManifestSlot::Invalid(source), ManifestSlot::Missing) => {
            Err(KeeperError::ManifestCorrupted {
                path: current_path,
                source,
            })
        }
        (ManifestSlot::Missing, ManifestSlot::Invalid(source)) => {
            Err(KeeperError::ManifestCorrupted {
                path: previous_path,
                source,
            })
        }
        (ManifestSlot::Invalid(current), ManifestSlot::Invalid(previous)) => {
            Err(KeeperError::NoValidManifest {
                directory: directory.to_path_buf(),
                current: current.to_string(),
                previous: previous.to_string(),
            })
        }
        (ManifestSlot::Valid(_), _) => unreachable!("valid current returned above"),
    }
}

fn load_valid_current(
    directory: &Path,
    current_path: &Path,
    previous_path: &Path,
    current: Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let previous = match read_manifest_slot(previous_path) {
        Ok(ManifestSlot::Valid(previous)) => previous,
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    // Readers intentionally take no writer lock. Re-read the primary after the
    // secondary so a commit that advanced both slots cannot manufacture a
    // false pair error from two different publication instants.
    let stable_current = match read_manifest_slot(current_path) {
        Ok(ManifestSlot::Valid(stable_current)) if stable_current == current => stable_current,
        Ok(ManifestSlot::Valid(newer_current)) => {
            return Ok(LoadedManifest {
                manifest: newer_current,
                source: ManifestSource::Current,
            });
        }
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    validate_manifest_pair(directory, stable_current, &previous)
}

fn validate_manifest_pair(
    directory: &Path,
    current: Manifest,
    previous: &Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let adjacent = previous.generation == current.generation
        || previous
            .generation
            .checked_add(1)
            .is_some_and(|generation| generation == current.generation);
    if !adjacent {
        return Err(KeeperError::InvalidGenerationPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
        });
    }
    if previous.generation == current.generation && previous != &current {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: "equal generations have different canonical contents".to_owned(),
        });
    }
    if previous.schema_id != current.schema_id {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, current.schema_id
            ),
        });
    }
    if current.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, current.docid_high_watermark
            ),
        });
    }
    if previous.generation != current.generation {
        validate_segment_transitions(previous, &current).map_err(|error| {
            KeeperError::InvalidManifestPair {
                directory: directory.to_path_buf(),
                current: current.generation,
                previous: previous.generation,
                detail: error.to_string(),
            }
        })?;
    }
    Ok(LoadedManifest {
        manifest: current,
        source: ManifestSource::Current,
    })
}

fn validate_loaded_schema(
    directory: &Path,
    expected: u64,
    loaded: &LoadedManifest,
) -> Result<(), KeeperError> {
    if loaded.manifest.schema_id == expected {
        return Ok(());
    }
    let slot = match loaded.source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => "<in-memory>",
    };
    Err(KeeperError::SchemaMismatch {
        path: directory.join(slot),
        expected,
        found: loaded.manifest.schema_id,
    })
}

fn validate_recovery_claims(directory: &Path, loaded: &LoadedManifest) -> Result<(), KeeperError> {
    if loaded.source == ManifestSource::Current || loaded.source == ManifestSource::InMemory {
        return Ok(());
    }
    let admitted_claim = loaded.manifest.generation.checked_add(1);
    for (path, claimed) in scan_generation_claims(directory)? {
        if claimed <= loaded.manifest.generation || admitted_claim == Some(claimed) {
            continue;
        }
        return Err(KeeperError::InvalidRecoveryClaim {
            path,
            recovered: loaded.manifest.generation,
            claimed,
        });
    }
    Ok(())
}

fn scan_generation_claims(directory: &Path) -> Result<Vec<(PathBuf, u64)>, KeeperError> {
    let mut claims = Vec::new();
    let entries = std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan recovery claims",
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read recovery claim entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let Some(claimed) = parse_claim_name(&entry.file_name()) else {
            continue;
        };
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| KeeperError::Io {
            operation: "inspect generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: format!(
                    "claim must be a zero-length regular file, found type {:?} and length {}",
                    metadata.file_type(),
                    metadata.len()
                ),
            });
        }
        claims.try_reserve(1).map_err(|error| KeeperError::Io {
            operation: "allocate recovery claim inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
        claims.push((path, claimed));
    }
    claims.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(claims)
}

fn validate_proposed_manifest_segments(
    directory: &Path,
    manifest: &Manifest,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<(), KeeperError> {
    for manifest_segment in &manifest.segments {
        let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
        let reader = SegmentReader::open_published(&path, schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        reader.verify().map_err(|source| KeeperError::SegmentOpen {
            path: path.clone(),
            source,
        })?;
        validate_segment_witnesses(&path, manifest_segment, &reader)?;
        #[cfg(feature = "durability")]
        if let WriterProtection::Enabled { protector, .. } = protection {
            let sidecar = FileProtector::sidecar_path(&path);
            let verification = protector.verify_file(&path, &sidecar).map_err(|source| {
                KeeperError::Durability {
                    operation: "preflight durable segment sidecar",
                    path: path.clone(),
                    source,
                }
            })?;
            if !verification.healthy {
                return Err(KeeperError::SegmentMetadataMismatch {
                    path,
                    detail: "durable segment sidecar does not match the proposed FSLX authority"
                        .to_owned(),
                });
            }
        }
        #[cfg(not(feature = "durability"))]
        let _ = protection;
        let _validated_segment = RecoveredSegment::bind(path, manifest_segment.clone(), reader)?;
    }
    Ok(())
}

fn validate_segment_witnesses(
    path: &Path,
    manifest: &ManifestSegment,
    reader: &SegmentReader<impl AsRef<[u8]>>,
) -> Result<(), KeeperError> {
    let header = reader.header();
    let mismatch = if header.segment_id != manifest.segment_id {
        Some(format!(
            "header segment_id {:#018x} != manifest {:#018x}",
            header.segment_id, manifest.segment_id
        ))
    } else if reader.file_len() != manifest.file_len {
        Some(format!(
            "file length {} != manifest {}",
            reader.file_len(),
            manifest.file_len
        ))
    } else if reader.file_xxh3() != manifest.file_xxh3 {
        Some(format!(
            "trailer file_xxh3 {:#018x} != manifest {:#018x}",
            reader.file_xxh3(),
            manifest.file_xxh3
        ))
    } else if header.docid_lo != manifest.docid_lo || header.docid_hi != manifest.docid_hi {
        Some(format!(
            "header range [{}, {}) != manifest [{}, {})",
            header.docid_lo, header.docid_hi, manifest.docid_lo, manifest.docid_hi
        ))
    } else if header.doc_count != manifest.doc_count {
        Some(format!(
            "header doc_count {} != manifest {}",
            header.doc_count, manifest.doc_count
        ))
    } else {
        None
    };
    if let Some(detail) = mismatch {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail,
        });
    }
    Ok(())
}

/// Remove unreachable Quill artifacts while the crate-internal caller holds
/// the writer lock.
///
/// This seam stays crate-private because callers must already hold the
/// `KeeperWriter` admission that owns the directory handle. The function first
/// opens and validates the selected snapshot, so a failed recovery never
/// performs garbage collection. Reader [`KeeperSnapshot::open`] never calls it.
/// Reachability is the union of every individually valid MANIFEST slot.
///
/// # Errors
///
/// Returns before deletion when recovery, directory enumeration, metadata, or
/// path-safety validation fails. An unlink or final directory fsync failure is
/// reported with the exact affected path.
pub(crate) fn collect_writer_garbage_under_lock(
    directory: impl AsRef<Path>,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at(directory.as_ref(), schema, options, SystemTime::now())
}

fn collect_writer_garbage_at(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at_platform(directory, schema, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    _: SchemaDescriptor,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let snapshot = KeeperSnapshot::open(directory, schema)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let live_segments = live_segment_names_at(&directory_file, directory, &snapshot)?;
    let current_generation = snapshot.loaded_manifest().manifest.generation;
    sweep_garbage_directory(
        &directory_file,
        directory,
        &live_segments,
        current_generation,
        options,
        now,
    )
}

/// Remove abandoned genesis artifacts when both durable slots are absent.
///
/// This is a separate writer-locked state because ordinary snapshot recovery
/// correctly returns [`KeeperError::IndexNotFound`] when neither slot exists.
/// Any canonical generation claim blocks deletion for writer-open recovery to
/// resolve. This helper remains reserved for a future writer-only doctor path.
#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
pub(crate) fn collect_abandoned_genesis_garbage_under_lock(
    directory: impl AsRef<Path>,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at(directory.as_ref(), options, SystemTime::now())
}

#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
fn collect_abandoned_genesis_garbage_at(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at_platform(directory, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        match read_manifest_slot_at(&directory_file, OsStr::new(slot_name), &path)? {
            ManifestSlot::Missing => {}
            ManifestSlot::Valid(_) => return Err(KeeperError::RecoveryRequired { path }),
            ManifestSlot::Invalid(source) => {
                return Err(KeeperError::ManifestCorrupted { path, source });
            }
        }
    }
    ensure_gc_directory_identity(directory, &directory_file)?;
    sweep_garbage_directory(&directory_file, directory, &HashSet::new(), 0, options, now)
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sweep_garbage_directory(
    directory_file: &File,
    directory: &Path,
    live_segments: &HashSet<OsString>,
    current_generation: u64,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    use rustix::fs::{AtFlags, Dir, FileType, statat, unlinkat};
    use std::os::unix::ffi::OsStringExt;

    let mut candidates = Vec::<(OsString, GarbageCandidate, rustix::fs::Stat)>::new();
    let mut entries = Dir::read_from(directory_file)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "scan garbage candidates",
            path: directory.to_path_buf(),
            source,
        })?;
    while let Some(entry) = entries.read() {
        let entry = entry
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "read garbage candidate",
                path: directory.to_path_buf(),
                source,
            })?;
        let name = OsString::from_vec(entry.file_name().to_bytes().to_vec());
        let Some(candidate) = classify_garbage_candidate(&name) else {
            continue;
        };
        let candidate_path = directory.join(&name);
        let stat = match statat(directory_file, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(source) if source == rustix::io::Errno::NOENT => continue,
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "inspect garbage candidate",
                    path: candidate_path,
                    source: io::Error::from(source),
                });
            }
        };
        let is_regular = FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile;
        if let GarbageCandidate::Claim { generation } = &candidate {
            if !is_regular || stat.st_size != 0 {
                return Err(KeeperError::InvalidClaimArtifact {
                    path: candidate_path,
                    detail: format!(
                        "claim must be a zero-length regular file, found type {:?} and length {}",
                        FileType::from_raw_mode(stat.st_mode),
                        stat.st_size
                    ),
                });
            }
            if *generation > current_generation {
                return Err(KeeperError::ClaimedGenerationPending {
                    path: candidate_path,
                    current: current_generation,
                    claimed: *generation,
                });
            }
        }
        if is_regular {
            candidates.try_reserve(1).map_err(|error| KeeperError::Io {
                operation: "allocate garbage candidate inventory",
                path: directory.to_path_buf(),
                source: io::Error::other(error.to_string()),
            })?;
            candidates.push((name, candidate, stat));
        }
    }

    let mut removable_segments = HashSet::<OsString>::new();
    removable_segments
        .try_reserve(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate removable segment inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in &candidates {
        if matches!(candidate, GarbageCandidate::Segment)
            && !live_segments.contains(name)
            && stat_old_enough(stat, now, options.grace_period)
        {
            removable_segments.insert(name.clone());
        }
    }

    let mut removals = Vec::<OsString>::new();
    removals
        .try_reserve_exact(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage removal inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in candidates {
        let old_enough = stat_old_enough(&stat, now, options.grace_period);
        let remove = match candidate {
            GarbageCandidate::Temporary => old_enough,
            GarbageCandidate::Segment => removable_segments.contains(&name),
            GarbageCandidate::Sidecar { base } => {
                old_enough
                    && sidecar_is_orphan_at(directory_file, directory, &removable_segments, &base)?
            }
            GarbageCandidate::Claim { .. } => old_enough,
        };
        if remove {
            removals.push(name);
        }
    }
    removals.sort_unstable();

    let mut removed = Vec::new();
    removed
        .try_reserve_exact(removals.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage report",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for name in removals {
        let relative = Path::new(&name);
        let path = safe_direct_child(directory, relative)?;
        match unlinkat(directory_file, relative, AtFlags::empty()) {
            Ok(()) => removed.push(relative.to_path_buf()),
            Err(source) if source == rustix::io::Errno::NOENT => {}
            Err(source) => {
                if !removed.is_empty() {
                    sync_gc_directory(directory_file, directory)?;
                }
                return Err(KeeperError::Io {
                    operation: "remove garbage candidate",
                    path,
                    source: io::Error::from(source),
                });
            }
        }
    }
    if !removed.is_empty() {
        sync_gc_directory(directory_file, directory)?;
    }
    Ok(GarbageCollectionReport { removed })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn live_segment_names_at(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
) -> Result<HashSet<OsString>, KeeperError> {
    let mut live = HashSet::new();
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        if let ManifestSlot::Valid(manifest) =
            read_manifest_slot_at(directory_file, OsStr::new(slot_name), &path)?
        {
            live.try_reserve(manifest.segments.len())
                .map_err(|error| KeeperError::Io {
                    operation: "allocate live segment set",
                    path: directory.to_path_buf(),
                    source: io::Error::other(error.to_string()),
                })?;
            live.extend(
                manifest
                    .segments
                    .iter()
                    .map(|segment| OsString::from(canonical_segment_name(segment.segment_id))),
            );
        }
    }
    let selected_name = match snapshot.loaded_manifest().source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    };
    let selected_path = directory.join(selected_name);
    match read_manifest_slot_at(directory_file, OsStr::new(selected_name), &selected_path)? {
        ManifestSlot::Valid(manifest) if manifest == snapshot.loaded_manifest().manifest => {}
        ManifestSlot::Missing | ManifestSlot::Invalid(_) | ManifestSlot::Valid(_) => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    }
    Ok(live)
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
#[derive(Debug)]
enum GarbageCandidate {
    Temporary,
    Segment,
    Sidecar { base: OsString },
    Claim { generation: u64 },
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn classify_garbage_candidate(name: &OsStr) -> Option<GarbageCandidate> {
    let name = name.to_str()?;
    if name.starts_with(".tmp-") {
        return Some(GarbageCandidate::Temporary);
    }
    if let Some(base) = name.strip_suffix(".fec.tmp")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Temporary);
    }
    if parse_segment_name(name).is_some() {
        return Some(GarbageCandidate::Segment);
    }
    if let Some(base) = name.strip_suffix(".fec")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Sidecar {
            base: OsString::from(base),
        });
    }
    parse_claim_name(OsStr::new(name)).map(|generation| GarbageCandidate::Claim { generation })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sidecar_is_orphan_at(
    directory_file: &File,
    directory: &Path,
    removable_segments: &HashSet<OsString>,
    base: &OsStr,
) -> Result<bool, KeeperError> {
    use rustix::fs::{AtFlags, statat};

    let base_text = base.to_str();
    if base_text.is_some_and(|name| parse_segment_name(name).is_some())
        && removable_segments.contains(base)
    {
        return Ok(true);
    }
    match statat(directory_file, base, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(false),
        Err(source) if source == rustix::io::Errno::NOENT => Ok(true),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect sidecar base",
            path: directory.join(base),
            source: io::Error::from(source),
        }),
    }
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_old_enough(stat: &rustix::fs::Stat, now: SystemTime, grace_period: Duration) -> bool {
    stat_modified_time(stat).is_some_and(|modified| {
        now.duration_since(modified)
            .is_ok_and(|age| age >= grace_period)
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_modified_time(stat: &rustix::fs::Stat) -> Option<SystemTime> {
    let seconds = stat.st_mtime;
    let nanoseconds = u32::try_from(stat.st_mtime_nsec).ok()?;
    if nanoseconds >= 1_000_000_000 {
        return None;
    }
    if seconds >= 0 {
        return SystemTime::UNIX_EPOCH
            .checked_add(Duration::new(u64::try_from(seconds).ok()?, nanoseconds));
    }
    let magnitude = seconds.unsigned_abs();
    let before_epoch = if nanoseconds == 0 {
        Duration::from_secs(magnitude)
    } else {
        Duration::new(magnitude.checked_sub(1)?, 1_000_000_000 - nanoseconds)
    };
    SystemTime::UNIX_EPOCH.checked_sub(before_epoch)
}

#[cfg(unix)]
fn open_gc_directory(directory: &Path) -> Result<File, KeeperError> {
    use rustix::fs::{Mode, OFlags, open};

    open(
        directory,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow garbage-collection directory",
        path: directory.to_path_buf(),
        source: io::Error::from(source),
    })
}

#[cfg(unix)]
fn ensure_gc_directory_identity(directory: &Path, opened: &File) -> Result<(), KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let path_metadata = std::fs::metadata(directory).map_err(|source| KeeperError::Io {
        operation: "verify garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    let opened_metadata = opened.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect opened garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    if path_metadata.dev() != opened_metadata.dev()
        || path_metadata.ino() != opened_metadata.ino()
        || !opened_metadata.is_dir()
    {
        return Err(KeeperError::GarbageDirectoryChanged {
            directory: directory.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sync_gc_directory(opened: &File, directory: &Path) -> Result<(), KeeperError> {
    opened.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn safe_direct_child(directory: &Path, relative: &Path) -> Result<PathBuf, KeeperError> {
    let mut components = relative.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(KeeperError::UnsafeGarbagePath {
            path: relative.to_path_buf(),
        });
    }
    Ok(directory.join(relative))
}

fn canonical_segment_name(segment_id: u64) -> String {
    format!("seg-{segment_id:016x}.fslx")
}

/// Internal fault-injection points in immutable segment publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentPublishCheckpoint {
    /// The synced temp file was renamed to its canonical published name.
    SegmentRenamed,
    /// The index directory was fsynced after the segment rename.
    DirectorySynced,
    /// The complete-source repair sidecar was emitted and made durable.
    SidecarEmitted,
}

/// Publish one already-synced segment without durability sidecar I/O.
///
/// The first public accumulator-to-seal commit path activates this primitive
/// in the dependent G1a milestone. Until then it remains a tested Keeper
/// checkpoint building block.
#[allow(dead_code, reason = "activated by the dependent G1a commit milestone")]
pub(crate) fn publish_pending_segment(pending: PendingSegmentFile) -> Result<PathBuf, KeeperError> {
    publish_pending_segment_with_observer(pending, |_, _| Ok(()), |_, _| Ok(None), |_, _| Ok(()))
}

/// Publish one already-synced segment and emit its complete-source sidecar.
#[cfg(feature = "durability")]
#[allow(dead_code, reason = "activated by the dependent G1a commit milestone")]
pub(crate) fn publish_pending_segment_durable(
    pending: PendingSegmentFile,
    protector: &FileProtector,
) -> Result<PathBuf, KeeperError> {
    publish_pending_segment_durable_with_observer(pending, protector, |_, _| Ok(()))
}

#[cfg(feature = "durability")]
fn publish_pending_segment_durable_with_observer<O>(
    pending: PendingSegmentFile,
    protector: &FileProtector,
    observe: O,
) -> Result<PathBuf, KeeperError>
where
    O: FnMut(SegmentPublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    publish_pending_segment_with_observer(
        pending,
        |_, published| ensure_durable_segment_sidecar_absent(published),
        |pending, published| {
            let witness = FileSourceWitness::new(pending.file_len(), pending.source_xxh3());
            let result = protector
                .protect_file_with_witness(published, witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect published segment",
                    path: published.to_path_buf(),
                    source,
                })?;
            Ok(Some(result.sidecar_path))
        },
        observe,
    )
}

fn publish_pending_segment_with_observer<B, A, O>(
    pending: PendingSegmentFile,
    mut before_rename: B,
    mut after_directory_sync: A,
    mut observe: O,
) -> Result<PathBuf, KeeperError>
where
    B: FnMut(&PendingSegmentFile, &Path) -> Result<(), KeeperError>,
    A: FnMut(&PendingSegmentFile, &Path) -> Result<Option<PathBuf>, KeeperError>,
    O: FnMut(SegmentPublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let pending_path = pending.path().to_path_buf();
    let expected_temp = format!(".tmp-segment-{:016x}", pending.segment_id());
    if pending_path.file_name() != Some(OsStr::new(&expected_temp)) {
        return Err(KeeperError::Io {
            operation: "validate segment temp name",
            path: pending_path,
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "pending segment does not have its canonical temp name",
            ),
        });
    }
    let metadata = std::fs::symlink_metadata(&pending_path).map_err(|source| KeeperError::Io {
        operation: "inspect segment temp",
        path: pending_path.clone(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != pending.file_len() {
        return Err(KeeperError::Io {
            operation: "validate segment temp",
            path: pending_path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "pending segment is not a regular file with the encoded length",
            ),
        });
    }

    let directory = pending_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let published = directory.join(canonical_segment_name(pending.segment_id()));
    match std::fs::symlink_metadata(&published) {
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect published segment destination",
                path: published,
                source,
            });
        }
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "publish segment",
                path: published,
                source: io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "published segment destination already exists",
                ),
            });
        }
    }

    before_rename(&pending, &published)?;
    std::fs::rename(&pending_path, &published).map_err(|source| KeeperError::Io {
        operation: "rename segment temp to published",
        path: pending_path,
        source,
    })?;
    observe(SegmentPublishCheckpoint::SegmentRenamed, &published)?;
    sync_directory(&directory)?;
    observe(SegmentPublishCheckpoint::DirectorySynced, &directory)?;
    if let Some(sidecar) = after_directory_sync(&pending, &published)? {
        observe(SegmentPublishCheckpoint::SidecarEmitted, &sidecar)?;
    }
    // Publication consumes the temp-file capability: after rename, callers
    // must not be able to reuse the pending handle for a second publish.
    drop(pending);
    Ok(published)
}

#[cfg(feature = "durability")]
fn ensure_durable_segment_sidecar_absent(published: &Path) -> Result<(), KeeperError> {
    let sidecar = FileProtector::sidecar_path(published);
    match std::fs::symlink_metadata(&sidecar) {
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect published segment sidecar destination",
            path: sidecar,
            source,
        }),
        Ok(_) => Err(KeeperError::Io {
            operation: "publish segment sidecar",
            path: sidecar,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment sidecar destination already exists",
            ),
        }),
    }
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn parse_segment_name(name: &str) -> Option<u64> {
    let hex = name.strip_prefix("seg-")?.strip_suffix(".fslx")?;
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

fn parse_quarantined_segment_name(name: &str) -> Option<u64> {
    parse_segment_name(strip_quarantine_suffix(name)?)
}

fn parse_quarantined_sidecar_segment_name(name: &str) -> Option<u64> {
    parse_segment_name(strip_quarantine_suffix(name)?.strip_suffix(".fec")?)
}

fn strip_quarantine_suffix(name: &str) -> Option<&str> {
    if let Some(base) = name.strip_suffix(".quarantine") {
        return Some(base);
    }
    let (base, probe) = name.rsplit_once('.')?;
    if probe.is_empty() || !probe.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    base.strip_suffix(".quarantine")
}

fn discover_quarantined_segments(
    directory: &Path,
    selected: &Manifest,
) -> Result<Vec<QuarantinedSegment>, KeeperError> {
    let mut doc_counts = BTreeMap::new();
    for segment in &selected.segments {
        doc_counts.insert(segment.segment_id, u64::from(segment.doc_count));
    }
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let slot_path = directory.join(slot_name);
        if let ManifestSlot::Valid(manifest) = read_manifest_slot(&slot_path)? {
            for segment in manifest.segments {
                doc_counts
                    .entry(segment.segment_id)
                    .or_insert_with(|| u64::from(segment.doc_count));
            }
        }
    }

    let mut quarantined = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan retained segment quarantines",
        path: directory.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read retained segment quarantine entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(segment_id) = parse_quarantined_segment_name(&name) else {
            continue;
        };
        let file_type = entry.file_type().map_err(|source| KeeperError::Io {
            operation: "inspect retained segment quarantine",
            path: entry.path(),
            source,
        })?;
        if !file_type.is_file() {
            return Err(KeeperError::Io {
                operation: "inspect retained segment quarantine",
                path: entry.path(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "quarantined segment must be a regular file",
                ),
            });
        }
        quarantined.push(QuarantinedSegment {
            segment_id,
            path: entry.path(),
            estimated_missing_docs: doc_counts.get(&segment_id).copied(),
        });
    }
    quarantined.sort_unstable_by_key(|segment| (segment.segment_id, segment.path.clone()));
    Ok(quarantined)
}

fn parse_claim_name(name: &OsStr) -> Option<u64> {
    let name = name.to_str()?;
    let digits = name.strip_prefix("gen-")?.strip_suffix(".claim")?;
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}

struct GenerationClaimGuard {
    admission: Arc<WriterAdmissionInner>,
    name: OsString,
    claim_file: File,
    device: u64,
    inode: u64,
}

impl GenerationClaimGuard {
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    fn acquire(admission: Arc<WriterAdmissionInner>, generation: u64) -> Result<Self, KeeperError> {
        use rustix::fs::{AtFlags, Mode, OFlags, openat, statat};
        use std::os::unix::fs::MetadataExt;

        admission.ensure_directory_identity()?;
        let name = OsString::from(format!("gen-{generation}.claim"));
        let path = admission.directory.join(&name);
        let claim = match openat(
            &admission.directory_file,
            &name,
            OFlags::WRONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(claim) => claim,
            Err(source) if source == rustix::io::Errno::EXIST => {
                let stat = statat(&admission.directory_file, &name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(io::Error::from)
                    .map_err(|source| KeeperError::Io {
                        operation: "inspect conflicting generation claim",
                        path: path.clone(),
                        source,
                    })?;
                let regular = rustix::fs::FileType::from_raw_mode(stat.st_mode)
                    == rustix::fs::FileType::RegularFile;
                if !regular || stat.st_size != 0 {
                    return Err(KeeperError::InvalidClaimArtifact {
                        path,
                        detail: format!(
                            "claim must be a zero-length regular file, found type {:?} and length {}",
                            rustix::fs::FileType::from_raw_mode(stat.st_mode),
                            stat.st_size
                        ),
                    });
                }
                return Err(KeeperError::GenerationClaimConflict { path, generation });
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create O_EXCL generation claim",
                    path,
                    source: io::Error::from(source),
                });
            }
        };
        let claim_file = File::from(claim);
        let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
            operation: "inspect owned generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: "new claim is not a zero-length regular file".to_owned(),
            });
        }
        claim_file.sync_all().map_err(|source| KeeperError::Io {
            operation: "fsync generation claim",
            path: path.clone(),
            source,
        })?;
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync generation-claim directory",
                path: admission.directory.clone(),
                source,
            })?;
        admission.ensure_directory_identity()?;
        Ok(Self {
            admission,
            name,
            device: metadata.dev(),
            inode: metadata.ino(),
            claim_file,
        })
    }

    #[cfg(not(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    )))]
    fn acquire(admission: Arc<WriterAdmissionInner>, _: u64) -> Result<Self, KeeperError> {
        Err(KeeperError::Io {
            operation: "create generation claim",
            path: admission.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "O_EXCL generation claims require Unix filesystem semantics",
            ),
        })
    }
}

impl Drop for GenerationClaimGuard {
    fn drop(&mut self) {
        release_generation_claim(self);
    }
}

#[cfg(unix)]
fn release_generation_claim(claim: &GenerationClaimGuard) {
    use rustix::fs::{AtFlags, statat, unlinkat};

    // Keep the owned descriptor alive through the identity check. If an
    // external actor replaced the pathname, leave that replacement untouched.
    let _ = claim.claim_file.metadata();
    let Ok(stat) = statat(
        &claim.admission.directory_file,
        &claim.name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) else {
        return;
    };
    if stat.st_dev != claim.device || stat.st_ino != claim.inode || stat.st_size != 0 {
        return;
    }
    if unlinkat(
        &claim.admission.directory_file,
        Path::new(&claim.name),
        AtFlags::empty(),
    )
    .is_ok()
    {
        let _ = claim.admission.directory_file.sync_all();
    }
}

#[cfg(not(unix))]
fn release_generation_claim(_: &GenerationClaimGuard) {}

/// In-process serializer for the two-slot MANIFEST publication protocol.
///
/// The owned asupersync guard moves into the blocking I/O closure. If the
/// awaiting task is cancelled after I/O starts, the guard remains held until
/// the atomic publication finishes, so a second writer cannot interleave the
/// two rename steps.
#[derive(Debug, Clone)]
struct ManifestPublisher {
    directory: PathBuf,
    publish_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
enum ManifestProtection {
    Disabled,
    #[cfg(feature = "durability")]
    Enabled(FileProtector),
}

impl ManifestPublisher {
    /// Bind a publisher to one Quill index directory.
    #[must_use]
    fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            publish_lock: global_publish_lock(),
        }
    }

    /// Publish one manifest using only the in-process serializer.
    ///
    /// This is a private substrate-test seam. Public mutation always goes
    /// through [`Self::publish_with_generation_claim`] while `KeeperWriter`
    /// retains `LOCK`. This ordinary path performs no sidecar I/O. If it follows
    /// a durable publication, an existing `MANIFEST.fec` may therefore be stale
    /// and must never be used as repair authority without matching the current
    /// source witness first.
    ///
    /// # Errors
    ///
    /// Returns typed validation, cancellation, transition, recovery, and I/O
    /// errors. A cancelled caller may observe an ambiguous result after the
    /// blocking phase starts; retrying is safe because generation validation
    /// reports whether the prior publication won.
    #[allow(dead_code, reason = "exercised by inline publication substrate tests")]
    async fn publish(&self, cx: &Cx, manifest: &Manifest) -> Result<LoadedManifest, KeeperError> {
        self.publish_with_generation_claim(cx, manifest, |_, _| Ok(()))
            .await
    }

    /// Publish with a caller-supplied cross-process generation claim.
    ///
    /// `claim` runs after the temp file is durable and before either slot is
    /// renamed. Its returned guard stays alive through the directory fsync;
    /// `KeeperWriter` can therefore release its claim from `Drop`.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::publish`] failures, returns any typed error from
    /// the claim callback.
    async fn publish_with_generation_claim<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        self.publish_with_generation_claim_and_protection(
            cx,
            manifest,
            claim,
            ManifestProtection::Disabled,
        )
        .await
    }

    /// Publish a MANIFEST and its complete-file FEC sidecar as one
    /// writer-serialized checkpoint.
    ///
    /// The new sidecar is encoded beside the durable temp MANIFEST before any
    /// slot replacement. The old previous sidecar is retired, then the current
    /// MANIFEST and its matching sidecar move to the previous slot before the
    /// new current slot and sidecar are installed ahead of the final directory
    /// fsync. This prevents a completed checkpoint from pairing a newer
    /// MANIFEST with an older current-slot sidecar. A failure after the
    /// MANIFEST rename but before sidecar installation can leave the new slot
    /// unprotected; the next `open_durable` recovery finishes that state while
    /// retaining its cross-process lock. The ordinary [`Self::publish`] path
    /// remains sidecar-free and private to substrate tests.
    ///
    /// # Errors
    ///
    /// Returns typed publication or durability errors. This method is private
    /// to substrate tests; public callers use the claimed variant below.
    #[cfg(feature = "durability")]
    #[allow(dead_code, reason = "exercised by inline durability substrate tests")]
    async fn publish_durable(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        protector: &FileProtector,
    ) -> Result<LoadedManifest, KeeperError> {
        self.publish_durable_with_generation_claim(cx, manifest, protector, |_, _| Ok(()))
            .await
    }

    /// Durable publication with a caller-supplied cross-process generation
    /// claim.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::publish_durable`] failures, returns any typed
    /// error from the claim callback.
    #[cfg(feature = "durability")]
    async fn publish_durable_with_generation_claim<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        protector: &FileProtector,
        claim: F,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        self.publish_with_generation_claim_and_protection(
            cx,
            manifest,
            claim,
            ManifestProtection::Enabled(protector.clone()),
        )
        .await
    }

    async fn publish_with_generation_claim_and_protection<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
        protection: ManifestProtection,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        let directory = self.directory.clone();
        let publish_lock = Arc::clone(&self.publish_lock);
        let guard = OwnedMutexGuard::lock(publish_lock, cx)
            .await
            .map_err(|source| KeeperError::PublishLock { source })?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        // Encode only after admission to the process-global critical section.
        // Otherwise every waiter could retain its own 64 MiB output buffer.
        //
        // The publisher owns the freshness witness: a caller that left
        // `last_publish_unix_s` zero gets stamped with the current wall clock,
        // so every keeper-written generation carries the cross-process
        // visibility timestamp. An explicit caller stamp is preserved, which
        // keeps deterministic fixtures byte-stable.
        let mut manifest = manifest.clone();
        if manifest.last_publish_unix_s == 0 {
            manifest.last_publish_unix_s = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()
                .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
                .ok_or_else(|| KeeperError::InvalidTransition {
                    detail: "current wall clock is outside the MANIFEST timestamp range".to_owned(),
                })?;
        }
        let bytes = manifest
            .to_bytes()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        spawn_blocking(move || {
            // Filesystem resolution is blocking I/O. Resolve the publication
            // target once inside the blocking phase, then use that same path
            // for inspection, the generation claim, renames, and directory
            // fsync so a late-created alias cannot split the operation.
            let directory = normalize_publish_directory(directory);
            match protection {
                ManifestProtection::Disabled => {
                    publish_manifest_locked(directory, &bytes, guard, claim)
                }
                #[cfg(feature = "durability")]
                ManifestProtection::Enabled(protector) => {
                    publish_manifest_durable_locked(directory, &bytes, guard, claim, &protector)
                }
            }
        })
        .await
    }

    #[cfg(test)]
    fn publish_lock_for_test(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.publish_lock)
    }
}

fn global_publish_lock() -> Arc<Mutex<()>> {
    Arc::clone(
        PUBLISH_LOCK.get_or_init(|| Arc::new(Mutex::with_name("quill.manifest_publish", ()))),
    )
}

fn normalize_publish_directory(directory: PathBuf) -> PathBuf {
    let absolute = if directory.is_absolute() {
        directory
    } else if let Ok(current_directory) = std::env::current_dir() {
        current_directory.join(directory)
    } else {
        directory
    };
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        return canonical;
    }

    // Canonicalize the longest existing prefix so aliases through an existing
    // symlink resolve to one stable I/O target even when the final index
    // directory has not been created yet. Normalize the missing suffix
    // lexically afterwards.
    for ancestor in absolute.ancestors().skip(1) {
        let Ok(mut canonical) = std::fs::canonicalize(ancestor) else {
            continue;
        };
        let Ok(suffix) = absolute.strip_prefix(ancestor) else {
            continue;
        };
        canonical.push(suffix);
        return lexical_normalize(&canonical);
    }
    lexical_normalize(&absolute)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized
                    .file_name()
                    .is_some_and(|part| part != std::ffi::OsStr::new(".."))
                {
                    let _ = normalized.pop();
                } else if !path.is_absolute() {
                    normalized.push("..");
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn publish_manifest_locked<C, F>(
    directory: PathBuf,
    bytes: &[u8],
    _guard: OwnedMutexGuard<()>,
    claim: F,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
{
    #[cfg(test)]
    {
        let observed_directory = directory.clone();
        publish_manifest_choreography(directory, bytes, claim, move |checkpoint, _| {
            observe_manifest_publish_checkpoint_for_test(&observed_directory, checkpoint);
            Ok(())
        })
    }
    #[cfg(not(test))]
    {
        publish_manifest_choreography(directory, bytes, claim, |_, _| Ok(()))
    }
}

#[cfg(feature = "durability")]
fn publish_manifest_durable_locked<C, F>(
    directory: PathBuf,
    bytes: &[u8],
    _guard: OwnedMutexGuard<()>,
    claim: F,
    protector: &FileProtector,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
{
    #[cfg(test)]
    {
        let observed_directory = directory.clone();
        publish_manifest_durable_choreography(
            directory,
            bytes,
            claim,
            protector,
            move |checkpoint, _| {
                observe_manifest_publish_checkpoint_for_test(&observed_directory, checkpoint);
                Ok(())
            },
        )
    }
    #[cfg(not(test))]
    {
        publish_manifest_durable_choreography(directory, bytes, claim, protector, |_, _| Ok(()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublishCheckpoint {
    TempWritten,
    TempSynced,
    GenerationClaimed,
    CurrentMovedToPrevious,
    TempMovedToCurrent,
    DirectorySynced,
}

#[cfg(test)]
#[derive(Clone)]
struct ManifestPublishPause {
    checkpoint: PublishCheckpoint,
    reached: Arc<std::sync::atomic::AtomicBool>,
    released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

/// One-shot control for pausing real MANIFEST choreography in cancellation
/// tests after a named filesystem checkpoint has completed.
#[cfg(test)]
pub(crate) struct ManifestPublishPauseControl {
    directory: PathBuf,
    reached: Arc<std::sync::atomic::AtomicBool>,
    released: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

#[cfg(test)]
static MANIFEST_PUBLISH_PAUSES: OnceLock<
    std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishPause>>,
> = OnceLock::new();

#[cfg(test)]
fn manifest_publish_pauses() -> &'static std::sync::Mutex<BTreeMap<PathBuf, ManifestPublishPause>> {
    MANIFEST_PUBLISH_PAUSES.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}

/// Arm one real MANIFEST publisher checkpoint for a single directory.
#[cfg(test)]
pub(crate) fn pause_manifest_publish_at_checkpoint_for_test(
    directory: &Path,
    checkpoint: PublishCheckpoint,
) -> ManifestPublishPauseControl {
    let directory = normalize_publish_directory(directory.to_path_buf());
    let reached = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let released = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let pause = ManifestPublishPause {
        checkpoint,
        reached: Arc::clone(&reached),
        released: Arc::clone(&released),
    };
    let previous = manifest_publish_pauses()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(directory.clone(), pause);
    assert!(
        previous.is_none(),
        "only one MANIFEST checkpoint pause may be armed per directory"
    );
    ManifestPublishPauseControl {
        directory,
        reached,
        released,
    }
}

#[cfg(test)]
impl ManifestPublishPauseControl {
    /// Whether the blocking publisher has completed the armed checkpoint.
    pub(crate) fn is_reached(&self) -> bool {
        self.reached.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Release the blocked choreography. Calling this more than once is safe.
    pub(crate) fn release(&self) {
        manifest_publish_pauses()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.directory);
        let (released, wake) = self.released.as_ref();
        *released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        wake.notify_all();
    }
}

#[cfg(test)]
impl Drop for ManifestPublishPauseControl {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
fn observe_manifest_publish_checkpoint_for_test(directory: &Path, checkpoint: PublishCheckpoint) {
    let pause = manifest_publish_pauses()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(directory)
        .filter(|pause| pause.checkpoint == checkpoint)
        .cloned();
    let Some(pause) = pause else {
        return;
    };
    pause
        .reached
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let (released, wake) = pause.released.as_ref();
    let mut released = released
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while !*released {
        released = wake
            .wait(released)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    drop(released);
}

fn publish_manifest_choreography<C, F, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    publish_manifest_choreography_with_hooks(
        directory,
        bytes,
        claim,
        |_, _, _| Ok(()),
        |_, _, _, _| Ok(()),
        |_, _| Ok(()),
        observe,
    )
}

#[cfg(feature = "durability")]
fn publish_manifest_durable_choreography<C, F, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    protector: &FileProtector,
    observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let witness = FileSourceWitness::from_bytes(bytes);
    let proposed_generation = Manifest::from_bytes(bytes)
        .map_err(|source| KeeperError::InvalidManifest { source })?
        .generation;
    publish_manifest_choreography_with_hooks(
        directory,
        bytes,
        claim,
        |directory, rename_current, generation| {
            debug_assert_eq!(generation, proposed_generation);
            let temp_path = directory.join(format!(".tmp-manifest-{generation}"));
            protector
                .protect_file_with_witness(&temp_path, witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect temp MANIFEST",
                    path: temp_path,
                    source,
                })?;
            prepare_manifest_sidecar_rotation(directory, rename_current, generation, protector)
        },
        complete_manifest_sidecar_rotation,
        |directory, current_path| {
            install_manifest_sidecar(directory, current_path, proposed_generation)
        },
        observe,
    )
}

fn publish_manifest_choreography_with_hooks<C, F, B, M, A, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    mut before_slot_renames: B,
    mut after_current_move: M,
    mut before_directory_sync: A,
    mut observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    B: FnMut(&Path, bool, u64) -> Result<(), KeeperError>,
    M: FnMut(&Path, &Path, &Path, u64) -> Result<(), KeeperError>,
    A: FnMut(&Path, &Path) -> Result<(), KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    #[cfg(not(unix))]
    ensure_atomic_publish_supported(&directory)?;
    let proposed =
        Manifest::from_bytes(bytes).map_err(|source| KeeperError::InvalidManifest { source })?;
    let metadata = std::fs::metadata(&directory).map_err(|source| KeeperError::Io {
        operation: "inspect directory",
        path: directory.clone(),
        source,
    })?;
    if !metadata.is_dir() {
        return Err(KeeperError::Io {
            operation: "inspect directory",
            path: directory,
            source: io::Error::new(
                io::ErrorKind::NotADirectory,
                "index path is not a directory",
            ),
        });
    }

    let (expected_generation, rename_current, previous_manifest) =
        match load_manifest_pair(&directory) {
            Ok(loaded) => {
                if loaded.source == ManifestSource::PreviousAfterCorruptCurrent {
                    return Err(KeeperError::RecoveryRequired {
                        path: directory.join("MANIFEST"),
                    });
                }
                let expected = loaded.manifest.generation.checked_add(1).ok_or(
                    KeeperError::GenerationExhausted {
                        current: loaded.manifest.generation,
                    },
                )?;
                (
                    expected,
                    loaded.source == ManifestSource::Current,
                    Some(loaded.manifest),
                )
            }
            Err(KeeperError::IndexNotFound { .. }) => (1, false, None),
            Err(error) => return Err(error),
        };
    if proposed.generation != expected_generation {
        return Err(KeeperError::GenerationConflict {
            expected: expected_generation,
            proposed: proposed.generation,
        });
    }
    if let Some(previous) = &previous_manifest {
        if proposed.schema_id != previous.schema_id {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "schema_id changed from {:#018x} to {:#018x}",
                    previous.schema_id, proposed.schema_id
                ),
            });
        }
        if proposed.docid_high_watermark < previous.docid_high_watermark {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "docid_high_watermark rolled back from {} to {}",
                    previous.docid_high_watermark, proposed.docid_high_watermark
                ),
            });
        }
        validate_segment_transitions(previous, &proposed)?;
    }

    let temp_path = directory.join(format!(".tmp-manifest-{}", proposed.generation));
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    prepare_manifest_temp(&temp_path, bytes, &mut observe)?;

    let _claim_guard = claim(&directory, proposed.generation)?;
    observe(PublishCheckpoint::GenerationClaimed, &directory)?;
    before_slot_renames(&directory, rename_current, proposed.generation)?;
    if rename_current {
        std::fs::rename(&current_path, &previous_path).map_err(|source| KeeperError::Io {
            operation: "rename current to previous",
            path: current_path.clone(),
            source,
        })?;
        after_current_move(
            &directory,
            &current_path,
            &previous_path,
            proposed.generation,
        )?;
        observe(PublishCheckpoint::CurrentMovedToPrevious, &previous_path)?;
    }
    std::fs::rename(&temp_path, &current_path).map_err(|source| KeeperError::Io {
        operation: "rename temp to current",
        path: temp_path,
        source,
    })?;
    observe(PublishCheckpoint::TempMovedToCurrent, &current_path)?;
    before_directory_sync(&directory, &current_path)?;
    sync_directory(&directory)?;
    observe(PublishCheckpoint::DirectorySynced, &directory)?;

    Ok(LoadedManifest {
        manifest: proposed,
        source: ManifestSource::Current,
    })
}

#[cfg(feature = "durability")]
fn prepare_manifest_sidecar_rotation(
    directory: &Path,
    rename_current: bool,
    proposed_generation: u64,
    protector: &FileProtector,
) -> Result<(), KeeperError> {
    let current = directory.join("MANIFEST.fec");
    let previous = directory.join("MANIFEST.prev.fec");
    let current_exists = regular_sidecar_exists(&current)?;
    let mut changed = false;

    if rename_current && regular_sidecar_exists(&previous)? {
        let retired = directory.join(format!(".tmp-manifest-previous-fec-{proposed_generation}"));
        changed = retire_manifest_sidecar(directory, &previous, &retired)? || changed;
    } else if !rename_current && current_exists {
        let retired = directory.join(format!(".tmp-manifest-current-fec-{proposed_generation}"));
        changed = retire_manifest_sidecar(directory, &current, &retired)? || changed;
    }

    if changed {
        sync_directory(directory)?;
    }
    if rename_current {
        let current_manifest = directory.join("MANIFEST");
        let current_bytes = std::fs::read(&current_manifest).map_err(|source| KeeperError::Io {
            operation: "read current MANIFEST for sidecar rotation",
            path: current_manifest.clone(),
            source,
        })?;
        let current_witness = FileSourceWitness::from_bytes(&current_bytes);
        let sidecar_matches = current_exists
            && protector
                .sidecar_matches_witness(&current, current_witness)
                .unwrap_or(false);
        if current_exists && !sidecar_matches {
            let retired =
                directory.join(format!(".tmp-manifest-current-fec-{proposed_generation}"));
            if retire_manifest_sidecar(directory, &current, &retired)? {
                sync_directory(directory)?;
            }
        }
        if !sidecar_matches {
            protector
                .protect_file_with_witness(&current_manifest, current_witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect prior MANIFEST before sidecar rotation",
                    path: current_manifest,
                    source,
                })?;
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn complete_manifest_sidecar_rotation(
    directory: &Path,
    current_path: &Path,
    previous_path: &Path,
    _: u64,
) -> Result<(), KeeperError> {
    let current_sidecar = FileProtector::sidecar_path(current_path);
    if !regular_sidecar_exists(&current_sidecar)? {
        return Ok(());
    }
    let previous_sidecar = FileProtector::sidecar_path(previous_path);
    ensure_sidecar_absent(&previous_sidecar)?;
    std::fs::rename(&current_sidecar, &previous_sidecar).map_err(|source| KeeperError::Io {
        operation: "move current MANIFEST sidecar to previous",
        path: current_sidecar,
        source,
    })?;
    sync_directory(directory)
}

#[cfg(feature = "durability")]
fn install_manifest_sidecar(
    directory: &Path,
    current_path: &Path,
    proposed_generation: u64,
) -> Result<(), KeeperError> {
    let temp_path = directory.join(format!(".tmp-manifest-{proposed_generation}"));
    let temp_sidecar = FileProtector::sidecar_path(&temp_path);
    let current_sidecar = FileProtector::sidecar_path(current_path);
    ensure_sidecar_absent(&current_sidecar)?;
    std::fs::rename(&temp_sidecar, &current_sidecar).map_err(|source| KeeperError::Io {
        operation: "install current MANIFEST sidecar",
        path: temp_sidecar,
        source,
    })
}

#[cfg(feature = "durability")]
fn regular_sidecar_exists(path: &Path) -> Result<bool, KeeperError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(KeeperError::Io {
            operation: "inspect MANIFEST sidecar",
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "MANIFEST sidecar is not a regular file",
            ),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect MANIFEST sidecar",
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(feature = "durability")]
fn retire_manifest_sidecar(
    directory: &Path,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    if source.parent() != Some(directory) || destination.parent() != Some(directory) {
        return Err(KeeperError::Io {
            operation: "validate MANIFEST sidecar retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "sidecar retirement paths must be direct children of the publish directory",
            ),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{AtFlags, FileType, RenameFlags, renameat_with, statat};

        let directory_file = File::open(directory).map_err(|source| KeeperError::Io {
            operation: "open MANIFEST publish directory",
            path: directory.to_path_buf(),
            source,
        })?;
        let source_name = source
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: source.to_path_buf(),
            })?;
        let source_stat = match statat(&directory_file, source_name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "inspect MANIFEST sidecar retirement source",
                    path: source.to_path_buf(),
                    source: io::Error::from(error),
                });
            }
        };
        if FileType::from_raw_mode(source_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::Io {
                operation: "retire MANIFEST sidecar",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "MANIFEST sidecar retirement source is not a regular file",
                ),
            });
        }

        let destination_name =
            destination
                .file_name()
                .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                    path: destination.to_path_buf(),
                })?;
        let mut collision = 0_u64;
        loop {
            let candidate = if collision == 0 {
                destination_name.to_os_string()
            } else {
                let mut name = destination_name.to_os_string();
                name.push(format!(".{collision}"));
                name
            };
            match renameat_with(
                &directory_file,
                source_name,
                &directory_file,
                &candidate,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(true),
                Err(error) if error == rustix::io::Errno::EXIST => {
                    collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                        operation: "allocate MANIFEST sidecar retirement destination",
                        path: destination.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "MANIFEST sidecar retirement suffix space is exhausted",
                        ),
                    })?;
                }
                Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
                Err(error) => {
                    return Err(KeeperError::Io {
                        operation: "atomically retire MANIFEST sidecar",
                        path: directory.join(candidate),
                        source: io::Error::from(error),
                    });
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    Err(KeeperError::Io {
        operation: "retire MANIFEST sidecar",
        path: source.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic no-replace MANIFEST sidecar retirement is unsupported on this platform",
        ),
    })
}

#[cfg(feature = "durability")]
fn ensure_sidecar_absent(path: &Path) -> Result<(), KeeperError> {
    if regular_sidecar_exists(path)? {
        return Err(KeeperError::Io {
            operation: "install MANIFEST sidecar",
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "stale MANIFEST sidecar survived its rotation checkpoint",
            ),
        });
    }
    Ok(())
}

fn prepare_manifest_temp<O>(path: &Path, bytes: &[u8], observe: &mut O) -> Result<(), KeeperError>
where
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let mut temp_file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            verify_reusable_manifest_temp(path, bytes)?;
            observe(PublishCheckpoint::TempSynced, path)?;
            return Ok(());
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "create temp",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    temp_file
        .write_all(bytes)
        .map_err(|source| KeeperError::Io {
            operation: "write temp",
            path: path.to_path_buf(),
            source,
        })?;
    observe(PublishCheckpoint::TempWritten, path)?;
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync temp",
        path: path.to_path_buf(),
        source,
    })?;
    observe(PublishCheckpoint::TempSynced, path)
}

fn verify_reusable_manifest_temp(path: &Path, expected: &[u8]) -> Result<(), KeeperError> {
    let mut temp_file = open_existing_manifest_temp(path).map_err(|source| KeeperError::Io {
        operation: "open existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = temp_file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() || metadata.len() != usize_to_u64(expected.len()) {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }

    let mut existing = Vec::new();
    existing
        .try_reserve_exact(expected.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate existing temp buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    Read::by_ref(&mut temp_file)
        .take(usize_to_u64(expected.len()).saturating_add(1))
        .read_to_end(&mut existing)
        .map_err(|source| KeeperError::Io {
            operation: "read existing temp",
            path: path.to_path_buf(),
            source,
        })?;
    if existing.as_slice() != expected {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync existing temp",
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(unix)]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no parent"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no file name"))?;
    let directory = File::open(parent)?;
    let temp_file = openat(
        &directory,
        file_name,
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    Ok(File::from(temp_file))
}

#[cfg(not(unix))]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

pub(crate) fn validate_manifest_successor(
    previous: &Manifest,
    proposed: &Manifest,
) -> Result<(), KeeperError> {
    let expected = previous
        .generation
        .checked_add(1)
        .ok_or(KeeperError::GenerationExhausted {
            current: previous.generation,
        })?;
    if proposed.generation != expected {
        return Err(KeeperError::GenerationConflict {
            expected,
            proposed: proposed.generation,
        });
    }
    if proposed.schema_id != previous.schema_id {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, proposed.schema_id
            ),
        });
    }
    if proposed.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, proposed.docid_high_watermark
            ),
        });
    }
    validate_segment_transitions(previous, proposed)
}

fn validate_segment_transitions(
    previous: &Manifest,
    proposed: &Manifest,
) -> Result<(), KeeperError> {
    let previous_max_seal_seq = previous
        .segments
        .iter()
        .map(|segment| segment.seal_seq)
        .max();
    let mut previous_by_id = Vec::new();
    previous_by_id
        .try_reserve_exact(previous.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("previous segment-index allocation failed: {error}"),
        })?;
    previous_by_id.extend(&previous.segments);
    previous_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut proposed_by_id = Vec::new();
    proposed_by_id
        .try_reserve_exact(proposed.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("proposed segment-index allocation failed: {error}"),
        })?;
    proposed_by_id.extend(&proposed.segments);
    proposed_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut previous_index = 0;
    let mut proposed_index = 0;
    let mut segment_set_changed = false;
    while previous_index < previous_by_id.len() && proposed_index < proposed_by_id.len() {
        let old = previous_by_id[previous_index];
        let new = proposed_by_id[proposed_index];
        match old.segment_id.cmp(&new.segment_id) {
            std::cmp::Ordering::Less => {
                segment_set_changed = true;
                previous_index += 1;
            }
            std::cmp::Ordering::Greater => {
                validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
                segment_set_changed = true;
                proposed_index += 1;
            }
            std::cmp::Ordering::Equal => {
                if old.seal_seq != new.seal_seq
                    || old.file_len != new.file_len
                    || old.file_xxh3 != new.file_xxh3
                    || old.docid_lo != new.docid_lo
                    || old.docid_hi != new.docid_hi
                    || old.doc_count != new.doc_count
                {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "immutable metadata changed for segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                let monotone = new
                    .tombstones
                    .is_monotone_superset_of(&old.tombstones)
                    .map_err(|error| KeeperError::InvalidTransition {
                        detail: format!(
                            "cannot compare tombstones for segment {:#018x}: {error}",
                            old.segment_id
                        ),
                    })?;
                if !monotone {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "tombstones shrank or changed for retained segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                previous_index += 1;
                proposed_index += 1;
            }
        }
    }
    if previous_index != previous_by_id.len() {
        segment_set_changed = true;
    }
    for new in &proposed_by_id[proposed_index..] {
        validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
        segment_set_changed = true;
    }
    if !segment_set_changed && previous.field_stats != proposed.field_stats {
        return Err(KeeperError::InvalidTransition {
            detail: "field-stat rollup changed while the immutable segment set was unchanged"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_new_segment_seal_seq(
    previous_max_seal_seq: Option<u64>,
    segment: &ManifestSegment,
) -> Result<(), KeeperError> {
    if let Some(previous_max) = previous_max_seal_seq
        && segment.seal_seq <= previous_max
    {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "new segment {:#018x} seal_seq {} does not advance prior maximum {}",
                segment.segment_id, segment.seal_seq, previous_max
            ),
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TombstoneContainer<'a> {
    chunk_id: u16,
    kind: u8,
    cardinality: u64,
    payload: &'a [u8],
    payload_offset: usize,
    encoded: &'a [u8],
}

struct TombstoneContainers<'a> {
    cursor: ByteCursor<'a>,
    remaining: usize,
}

impl<'a> TombstoneContainers<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, ManifestCodecError> {
        let mut cursor = ByteCursor::new(bytes);
        let remaining =
            count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
        Ok(Self { cursor, remaining })
    }

    fn next_container(&mut self) -> Result<Option<TombstoneContainer<'a>>, ManifestCodecError> {
        if self.remaining == 0 {
            if self.cursor.remaining() != 0 {
                return Err(non_canonical(format!(
                    "tombstone set has {} trailing bytes",
                    self.cursor.remaining()
                )));
            }
            return Ok(None);
        }

        let container_start = self.cursor.position();
        let chunk_id = self.cursor.u16()?;
        let kind = self.cursor.u8()?;
        let encoded_count = self.cursor.u16()?;
        let (cardinality, payload_length) = match kind {
            0 => (u64::from(encoded_count), usize::from(encoded_count) * 2),
            1 => (
                if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                },
                TOMBSTONE_BITMAP_BYTES,
            ),
            other => {
                return Err(non_canonical(format!(
                    "tombstone container has unknown kind {other}"
                )));
            }
        };
        let payload_offset = self.cursor.position();
        let payload = self.cursor.take(payload_length)?;
        let encoded = &self.cursor.bytes[container_start..self.cursor.position()];
        self.remaining -= 1;
        Ok(Some(TombstoneContainer {
            chunk_id,
            kind,
            cardinality,
            payload,
            payload_offset,
            encoded,
        }))
    }
}

fn tombstone_chunk_count(bytes: &[u8]) -> Result<u32, ManifestCodecError> {
    let mut cursor = ByteCursor::new(bytes);
    cursor.u32()
}

fn tombstone_chunk_exists(bytes: &[u8], chunk_id: u16) -> Result<bool, ManifestCodecError> {
    let mut containers = TombstoneContainers::new(bytes)?;
    while let Some(container) = containers.next_container()? {
        match container.chunk_id.cmp(&chunk_id) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal => return Ok(true),
            std::cmp::Ordering::Greater => return Ok(false),
        }
    }
    Ok(false)
}

fn write_tombstone_container_header(
    output: &mut Vec<u8>,
    chunk_id: u16,
    kind: u8,
    cardinality: u64,
) -> Result<(), ManifestCodecError> {
    put_u16(output, chunk_id);
    output.push(kind);
    let encoded_count = if kind == 1 && cardinality == 65_536 {
        0
    } else {
        u16::try_from(cardinality)
            .map_err(|_| invalid("tombstone container cardinality does not fit wire count"))?
    };
    put_u16(output, encoded_count);
    Ok(())
}

fn write_singleton_tombstone_chunk(output: &mut Vec<u8>, chunk_id: u16, low: u16) {
    put_u16(output, chunk_id);
    output.push(0);
    put_u16(output, 1);
    put_u16(output, low);
}

fn write_inserted_tombstone_container(
    output: &mut Vec<u8>,
    container: TombstoneContainer<'_>,
    low: u16,
) -> Result<(), ManifestCodecError> {
    let next_cardinality = container
        .cardinality
        .checked_add(1)
        .ok_or_else(|| invalid("tombstone container cardinality overflow"))?;
    match container.kind {
        0 if container.cardinality < u64::from(TOMBSTONE_ARRAY_MAX_CARDINALITY) => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            let mut emitted = false;
            for index in 0..container.payload.len() / 2 {
                let current = array_value(container.payload, index);
                if !emitted && low < current {
                    put_u16(output, low);
                    emitted = true;
                }
                put_u16(output, current);
            }
            if !emitted {
                put_u16(output, low);
            }
        }
        0 => {
            write_tombstone_container_header(output, container.chunk_id, 1, next_cardinality)?;
            let payload_start = output.len();
            output.resize(payload_start + TOMBSTONE_BITMAP_BYTES, 0);
            for index in 0..container.payload.len() / 2 {
                set_bitmap_value(
                    &mut output[payload_start..],
                    array_value(container.payload, index),
                );
            }
            set_bitmap_value(&mut output[payload_start..], low);
        }
        1 => {
            let encoded_start = output.len();
            output.extend_from_slice(container.encoded);
            let count_offset = encoded_start + 3;
            let encoded_count = if next_cardinality == 65_536 {
                0
            } else {
                u16::try_from(next_cardinality)
                    .map_err(|_| invalid("bitmap tombstone cardinality does not fit wire count"))?
            };
            output[count_offset..count_offset + 2].copy_from_slice(&encoded_count.to_le_bytes());
            let payload_start = encoded_start + 5;
            set_bitmap_value(
                &mut output[payload_start..payload_start + TOMBSTONE_BITMAP_BYTES],
                low,
            );
        }
        _ => return Err(invalid("unknown validated tombstone container kind")),
    }
    Ok(())
}

fn write_removed_tombstone_container(
    output: &mut Vec<u8>,
    container: TombstoneContainer<'_>,
    low: u16,
) -> Result<(), ManifestCodecError> {
    let next_cardinality = container
        .cardinality
        .checked_sub(1)
        .ok_or_else(|| invalid("tombstone container cardinality underflow"))?;
    match container.kind {
        0 if next_cardinality == 0 => {}
        0 => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            for index in 0..container.payload.len() / 2 {
                let current = array_value(container.payload, index);
                if current != low {
                    put_u16(output, current);
                }
            }
        }
        1 if next_cardinality < TOMBSTONE_BITMAP_MIN_CARDINALITY => {
            write_tombstone_container_header(output, container.chunk_id, 0, next_cardinality)?;
            for (byte_index, byte) in container.payload.iter().copied().enumerate() {
                let mut bits = byte;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let value = u16::try_from(byte_index * 8 + bit)
                        .map_err(|_| invalid("bitmap tombstone low-bit overflow"))?;
                    if value != low {
                        put_u16(output, value);
                    }
                    bits &= bits - 1;
                }
            }
        }
        1 => {
            let encoded_start = output.len();
            output.extend_from_slice(container.encoded);
            let count_offset = encoded_start + 3;
            let encoded_count = if next_cardinality == 65_536 {
                0
            } else {
                u16::try_from(next_cardinality)
                    .map_err(|_| invalid("bitmap tombstone cardinality does not fit wire count"))?
            };
            output[count_offset..count_offset + 2].copy_from_slice(&encoded_count.to_le_bytes());
            let payload_start = encoded_start + 5;
            clear_bitmap_value(
                &mut output[payload_start..payload_start + TOMBSTONE_BITMAP_BYTES],
                low,
            );
        }
        _ => return Err(invalid("unknown validated tombstone container kind")),
    }
    Ok(())
}

fn array_binary_search(bytes: &[u8], value: u16) -> bool {
    let mut lo = 0_usize;
    let mut hi = bytes.len() / 2;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match array_value(bytes, mid).cmp(&value) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    false
}

fn set_bitmap_value(bitmap: &mut [u8], value: u16) {
    let value = usize::from(value);
    bitmap[value / 8] |= 1 << (value % 8);
}

fn clear_bitmap_value(bitmap: &mut [u8], value: u16) {
    let value = usize::from(value);
    bitmap[value / 8] &= !(1 << (value % 8));
}

fn split_tombstone_docid(global_docid: u32) -> (u16, u16) {
    let bytes = global_docid.to_le_bytes();
    (
        u16::from_le_bytes([bytes[2], bytes[3]]),
        u16::from_le_bytes([bytes[0], bytes[1]]),
    )
}

fn tombstones_are_subset(previous: &[u8], proposed: &[u8]) -> Result<bool, ManifestCodecError> {
    if previous == proposed {
        return Ok(true);
    }
    let mut previous = TombstoneContainers::new(previous)?;
    let mut proposed = TombstoneContainers::new(proposed)?;
    let mut old = previous.next_container()?;
    let mut new = proposed.next_container()?;

    while let Some(old_container) = old {
        while new.is_some_and(|container| container.chunk_id < old_container.chunk_id) {
            new = proposed.next_container()?;
        }
        let Some(new_container) = new else {
            return Ok(false);
        };
        if new_container.chunk_id != old_container.chunk_id
            || !tombstone_container_is_subset(old_container, new_container)
        {
            return Ok(false);
        }
        old = previous.next_container()?;
        new = proposed.next_container()?;
    }
    Ok(true)
}

fn tombstone_container_is_subset(
    previous: TombstoneContainer<'_>,
    proposed: TombstoneContainer<'_>,
) -> bool {
    if previous.cardinality > proposed.cardinality {
        return false;
    }
    match (previous.kind, proposed.kind) {
        (0, 0) => array_is_subset_of_array(previous.payload, proposed.payload),
        (0, 1) => array_is_subset_of_bitmap(previous.payload, proposed.payload),
        (1, 0) => bitmap_is_subset_of_array(previous.payload, proposed.payload),
        (1, 1) => previous
            .payload
            .iter()
            .zip(proposed.payload)
            .all(|(old, new)| old & !new == 0),
        _ => false,
    }
}

fn array_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for previous_index in 0..previous.len() / 2 {
        let old = array_value(previous, previous_index);
        while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old {
            proposed_index += 1;
        }
        if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old {
            return false;
        }
        proposed_index += 1;
    }
    true
}

fn array_is_subset_of_bitmap(previous: &[u8], proposed: &[u8]) -> bool {
    (0..previous.len() / 2).all(|index| bitmap_contains(proposed, array_value(previous, index)))
}

fn bitmap_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for (byte_index, byte) in previous.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let Ok(old) = u16::try_from(byte_index * 8 + bit) else {
                return false;
            };
            while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old
            {
                proposed_index += 1;
            }
            if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old
            {
                return false;
            }
            proposed_index += 1;
            bits &= bits - 1;
        }
    }
    true
}

fn array_value(bytes: &[u8], index: usize) -> u16 {
    let offset = index * 2;
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn bitmap_contains(bitmap: &[u8], value: u16) -> bool {
    let value = usize::from(value);
    bitmap[value / 8] & (1 << (value % 8)) != 0
}

enum ManifestSlot {
    Missing,
    Valid(Manifest),
    Invalid(ManifestCodecError),
}

fn read_manifest_slot(path: &Path) -> Result<ManifestSlot, KeeperError> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(ManifestSlot::Missing);
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !path_metadata.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match open_manifest_slot(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(ManifestSlot::Missing),
        #[cfg(unix)]
        Err(error) if error.raw_os_error() == Some(rustix::io::Errno::LOOP.raw_os_error()) => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    read_manifest_file(path, file)
}

fn read_manifest_file(path: &Path, file: File) -> Result<ManifestSlot, KeeperError> {
    let file_length = file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat",
        path: path.to_path_buf(),
        source,
    })?;
    if !file_length.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    if file_length.len() > usize_to_u64(MAX_MANIFEST_BYTES) {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: file_length.len(),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    let reserve = usize::try_from(file_length.len()).unwrap_or(MAX_MANIFEST_BYTES);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reserve)
        .map_err(|error| KeeperError::Io {
            operation: "allocate read buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    let read_limit = usize_to_u64(MAX_MANIFEST_BYTES).saturating_add(1);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| KeeperError::Io {
            operation: "read",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(bytes.len()),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    Ok(match Manifest::from_bytes(&bytes) {
        Ok(manifest) => ManifestSlot::Valid(manifest),
        Err(error) => ManifestSlot::Invalid(error),
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn read_manifest_slot_at(
    directory: &File,
    name: &OsStr,
    path: &Path,
) -> Result<ManifestSlot, KeeperError> {
    use rustix::fs::{AtFlags, FileType, Mode, OFlags, openat, statat};

    let stat = match statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(file) => File::from(file),
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) if source == rustix::io::Errno::LOOP => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    read_manifest_file(path, file)
}

#[cfg(unix)]
fn open_manifest_slot(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "slot path has no file name"))?;
    let directory = File::open(parent)?;
    let slot = openat(
        &directory,
        file_name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(slot))
}

#[cfg(not(unix))]
fn open_manifest_slot(path: &Path) -> io::Result<File> {
    File::open(path)
}

#[cfg(not(unix))]
fn ensure_atomic_publish_supported(directory: &Path) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "verify atomic publish support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "MANIFEST replacement and directory fsync require Unix semantics",
        ),
    })
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "fsync directory",
            path: directory.to_path_buf(),
            source,
        })
}

#[cfg(not(unix))]
fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    ensure_atomic_publish_supported(directory)
}

#[derive(Debug, Clone, Copy)]
enum ErrorClass {
    Invalid,
    NonCanonical,
}

fn validate_manifest_shape(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    if manifest.generation == 0 {
        return Err(reject("generation zero is reserved".to_owned()));
    }
    if manifest.flags & !MANIFEST_KNOWN_FLAGS != 0 {
        return Err(reject(format!(
            "unknown flag bits {:#010x}",
            manifest.flags & !MANIFEST_KNOWN_FLAGS
        )));
    }
    if manifest.segments.len() > MAX_MANIFEST_SEGMENTS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "segment count",
            actual: usize_to_u64(manifest.segments.len()),
            limit: usize_to_u64(MAX_MANIFEST_SEGMENTS),
        });
    }
    if manifest.field_stats.len() > MAX_MANIFEST_FIELDS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "field count",
            actual: usize_to_u64(manifest.field_stats.len()),
            limit: usize_to_u64(MAX_MANIFEST_FIELDS),
        });
    }
    Ok(())
}

fn validate_manifest(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    validate_manifest_shape(manifest, error_class)?;
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    let mut segment_ids = Vec::new();
    segment_ids
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| reject(format!("segment-id validation allocation failed: {error}")))?;
    segment_ids.extend(manifest.segments.iter().map(|segment| segment.segment_id));
    segment_ids.sort_unstable();
    if let Some([duplicate, ..]) = segment_ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate segment_id {duplicate:#018x}")));
    }

    let mut seal_sequences = Vec::new();
    seal_sequences
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| {
            reject(format!(
                "seal-sequence validation allocation failed: {error}"
            ))
        })?;
    seal_sequences.extend(manifest.segments.iter().map(|segment| segment.seal_seq));
    seal_sequences.sort_unstable();
    if let Some([duplicate, ..]) = seal_sequences.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate seal_seq {duplicate}")));
    }

    let mut previous_hi = None;
    let mut maximum_hi = 0_u64;
    let mut total_documents = 0_u64;
    for (index, segment) in manifest.segments.iter().enumerate() {
        if segment.file_len == 0 {
            return Err(reject(format!("segment {index} has zero file_len")));
        }
        if segment.docid_lo >= segment.docid_hi {
            return Err(reject(format!(
                "segment {index} has empty or reversed range [{}, {})",
                segment.docid_lo, segment.docid_hi
            )));
        }
        if segment.docid_hi > MAX_DOCID_EXCLUSIVE {
            return Err(reject(format!(
                "segment {index} range exceeds the u32 payload domain"
            )));
        }
        if previous_hi.is_some_and(|hi| segment.docid_lo < hi) {
            return Err(reject(format!(
                "segment {index} overlaps or is out of docid_lo order"
            )));
        }
        let span = segment.docid_hi - segment.docid_lo;
        if u64::from(segment.doc_count) > span {
            return Err(reject(format!(
                "segment {index} doc_count {} exceeds range span {span}",
                segment.doc_count
            )));
        }
        segment
            .tombstones
            .validate_range((segment.docid_lo, segment.docid_hi))?;
        let tombstone_count = segment.tombstones.cardinality();
        if tombstone_count > u64::from(segment.doc_count) {
            return Err(reject(format!(
                "segment {index} tombstone count {tombstone_count} exceeds doc_count {}",
                segment.doc_count
            )));
        }
        total_documents = total_documents
            .checked_add(u64::from(segment.doc_count))
            .ok_or_else(|| reject("aggregate segment doc_count overflow".to_owned()))?;
        maximum_hi = maximum_hi.max(segment.docid_hi);
        previous_hi = Some(segment.docid_hi);
    }
    if manifest.docid_high_watermark < maximum_hi {
        return Err(reject(format!(
            "docid_high_watermark {} is below maximum docid_hi {maximum_hi}",
            manifest.docid_high_watermark
        )));
    }

    let mut previous_field = None;
    for stats in &manifest.field_stats {
        if previous_field.is_some_and(|field| stats.field_ord <= field) {
            return Err(reject(format!(
                "field stats are not strictly ordered at field_ord {}",
                stats.field_ord
            )));
        }
        if u64::from(stats.doc_count) != total_documents {
            return Err(reject(format!(
                "field {} doc_count {} differs from canonical at-seal denominator {total_documents} (bd-quill-e3-keeper-ndtk.10)",
                stats.field_ord, stats.doc_count
            )));
        }
        previous_field = Some(stats.field_ord);
    }
    Ok(())
}

fn validate_tombstone_bytes(
    bytes: &[u8],
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let mut cursor = ByteCursor::new(bytes);
    let count = consume_tombstone_set(&mut cursor, range)?;
    if cursor.remaining() != 0 {
        return Err(non_canonical(format!(
            "tombstone set has {} trailing bytes",
            cursor.remaining()
        )));
    }
    Ok(count)
}

fn consume_tombstone_set(
    cursor: &mut ByteCursor<'_>,
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let chunk_count = count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
    let mut previous_chunk = None;
    let mut total = 0_u64;
    for chunk_index in 0..chunk_count {
        let chunk_id = cursor.u16()?;
        if previous_chunk.is_some_and(|previous| chunk_id <= previous) {
            return Err(non_canonical(format!(
                "tombstone chunk {chunk_index} is not strictly ordered"
            )));
        }
        previous_chunk = Some(chunk_id);
        let kind = cursor.u8()?;
        let encoded_count = cursor.u16()?;
        let cardinality = match kind {
            0 => {
                if encoded_count == 0 {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} is empty"
                    )));
                }
                if encoded_count > TOMBSTONE_ARRAY_MAX_CARDINALITY {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} cardinality {encoded_count} exceeds promotion threshold {TOMBSTONE_ARRAY_MAX_CARDINALITY}"
                    )));
                }
                let count = usize::from(encoded_count);
                let mut previous_low = None;
                for _ in 0..count {
                    let low = cursor.u16()?;
                    if previous_low.is_some_and(|previous| low <= previous) {
                        return Err(non_canonical(format!(
                            "array tombstone chunk {chunk_index} is not strictly ordered"
                        )));
                    }
                    validate_tombstone_docid(chunk_id, low, range)?;
                    previous_low = Some(low);
                }
                u64::from(encoded_count)
            }
            1 => {
                let bitmap = cursor.take(8_192)?;
                let actual = bitmap
                    .iter()
                    .map(|byte| u64::from(byte.count_ones()))
                    .sum::<u64>();
                let expected = if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                };
                if actual != expected {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} != {expected}"
                    )));
                }
                if actual < TOMBSTONE_BITMAP_MIN_CARDINALITY {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} is below demotion threshold {TOMBSTONE_BITMAP_MIN_CARDINALITY}"
                    )));
                }
                if let Some(bounds) = range {
                    validate_bitmap_range(bitmap, chunk_id, bounds)?;
                }
                actual
            }
            other => {
                return Err(non_canonical(format!(
                    "tombstone chunk {chunk_index} has unknown kind {other}"
                )));
            }
        };
        total = total
            .checked_add(cardinality)
            .ok_or_else(|| non_canonical("tombstone cardinality overflow"))?;
    }
    Ok(total)
}

fn validate_bitmap_range(
    bitmap: &[u8],
    chunk_id: u16,
    bounds: (u64, u64),
) -> Result<(), ManifestCodecError> {
    let (lo, hi) = bounds;
    let chunk_lo = u64::from(chunk_id) << 16;
    let chunk_hi = chunk_lo + 65_536;
    if lo <= chunk_lo && chunk_hi <= hi {
        return Ok(());
    }
    if hi <= chunk_lo || lo >= chunk_hi {
        return Err(non_canonical(format!(
            "bitmap tombstone chunk {chunk_id} is outside segment range [{lo}, {hi})"
        )));
    }

    for (byte_index, byte) in bitmap.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let low = u16::try_from(byte_index * 8 + bit as usize)
                .map_err(|_| non_canonical("bitmap low-bit overflow"))?;
            validate_tombstone_docid(chunk_id, low, Some(bounds))?;
            bits &= bits - 1;
        }
    }
    Ok(())
}

fn validate_tombstone_docid(
    chunk_id: u16,
    low: u16,
    range: Option<(u64, u64)>,
) -> Result<(), ManifestCodecError> {
    if let Some((lo, hi)) = range {
        let docid = (u64::from(chunk_id) << 16) | u64::from(low);
        if !(lo..hi).contains(&docid) {
            return Err(non_canonical(format!(
                "tombstoned docid {docid} is outside segment range [{lo}, {hi})"
            )));
        }
    }
    Ok(())
}

fn check_manifest_byte_limit(length: usize) -> Result<(), ManifestCodecError> {
    if length > MAX_MANIFEST_BYTES {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(length),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        });
    }
    Ok(())
}

fn manifest_encoded_len(manifest: &Manifest) -> Result<usize, ManifestCodecError> {
    let length = MANIFEST_V2_MIN_BYTES
        .checked_add(
            manifest
                .segments
                .len()
                .checked_mul(SEGMENT_FIXED_BYTES)
                .ok_or_else(|| invalid("manifest encoded length overflow"))?,
        )
        .and_then(|length| {
            manifest.segments.iter().try_fold(length, |total, segment| {
                total.checked_add(segment.tombstones.encoded_len())
            })
        })
        .and_then(|length| {
            manifest
                .field_stats
                .len()
                .checked_mul(FIELD_STATS_BYTES)
                .and_then(|stats| length.checked_add(stats))
        })
        .ok_or_else(|| invalid("manifest encoded length overflow"))?;
    check_manifest_byte_limit(length)?;
    Ok(length)
}

fn count_to_usize(
    count: u32,
    resource: &'static str,
    limit: usize,
) -> Result<usize, ManifestCodecError> {
    let count = usize::try_from(count).map_err(|_| ManifestCodecError::ResourceLimit {
        resource,
        actual: u64::from(count),
        limit: usize_to_u64(limit),
    })?;
    if count > limit {
        return Err(ManifestCodecError::ResourceLimit {
            resource,
            actual: usize_to_u64(count),
            limit: usize_to_u64(limit),
        });
    }
    Ok(count)
}

fn copy_bytes(bytes: &[u8], label: &'static str) -> Result<Vec<u8>, ManifestCodecError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|error| non_canonical(format!("{label} allocation failed: {error}")))?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn invalid(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::Invalid {
        detail: detail.into(),
    }
}

fn non_canonical(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::NonCanonical {
        detail: detail.into(),
    }
}

fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

// ============================================================================
// Blue-green engine directories and the CURRENT pointer (registry §7.3, bead
// bd-quill-duel-blue-green-vwf7).
//
// Versioned sibling engine directories (`quill-v1/`, `tantivy/`) sit one level
// below a lexical root; a tiny `CURRENT` pointer file in that root names the
// active engine directory. Publication reuses the MANIFEST temp+rename+dir-
// fsync discipline one level up, so the G3 flip and every rebuild are single
// atomic pointer swaps with rollback being the same swap in reverse. The
// retired engine directory is preserved untouched until a human-approved
// retirement sweep (e9.3), keeping RULE-1 intact by construction.
// ============================================================================

/// Eight-byte CURRENT magic, including its trailing NUL.
pub const CURRENT_MAGIC: [u8; 8] = *b"FSLXCUR\0";
/// Current durable CURRENT-pointer format version.
pub const CURRENT_FORMAT_VERSION: u32 = 1;
/// Canonical CURRENT pointer file name in the lexical root.
pub const CURRENT_FILE_NAME: &str = "CURRENT";
/// Index format version recorded for pre-FSLX (tantivy) engine directories.
pub const TANTIVY_INDEX_FORMAT_VERSION: u32 = 0;
/// Encoded length excluding the variable directory-name bytes.
const CURRENT_FIXED_BYTES: usize = 8 + 4 + 1 + 2 + 4 + 4;
/// Maximum accepted CURRENT pointer length (name bounded by its `u16` field).
const MAX_CURRENT_BYTES: usize = CURRENT_FIXED_BYTES + 65_535;
/// Bound on temp-name collisions tolerated from one publisher.
const CURRENT_TEMP_COLLISION_LIMIT: u32 = 128;

/// Engine family recorded in a CURRENT pointer (registry §7.3 kind codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlueGreenEngine {
    /// Quill FSLX engine directory (kind code 1).
    Quill,
    /// Legacy tantivy engine directory (kind code 2).
    Tantivy,
}

impl BlueGreenEngine {
    /// Stable human label used in errors and telemetry.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Quill => "quill",
            Self::Tantivy => "tantivy",
        }
    }

    /// Registry §7.3 wire code.
    #[must_use]
    pub const fn kind_code(self) -> u8 {
        match self {
            Self::Quill => 1,
            Self::Tantivy => 2,
        }
    }

    fn from_kind_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Quill),
            2 => Some(Self::Tantivy),
            _ => None,
        }
    }

    /// Marker file whose presence identifies an engine directory of this kind
    /// during migration-adoption scans.
    const fn adoption_marker(self) -> &'static str {
        match self {
            Self::Quill => "MANIFEST",
            Self::Tantivy => "meta.json",
        }
    }

    /// Index format version recorded when adopting a directory of this kind.
    const fn adopted_format_version(self) -> u32 {
        match self {
            Self::Quill => crate::segment::FSLX_FORMAT_VERSION,
            Self::Tantivy => TANTIVY_INDEX_FORMAT_VERSION,
        }
    }
}

/// Decoded CURRENT pointer payload (registry §7.3).
///
/// Instances are always validated: construction and decoding reject empty,
/// over-long, non-UTF-8, or path-unsafe directory names, so a pointer can
/// never name anything but a plain direct child of its lexical root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentPointer {
    engine: BlueGreenEngine,
    dir_name: String,
    index_format_version: u32,
}

impl CurrentPointer {
    /// Construct a validated pointer to `dir_name` under a lexical root.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentPointerError`] for an empty, over-long, or path-unsafe
    /// directory name.
    pub fn new(
        engine: BlueGreenEngine,
        dir_name: impl Into<String>,
        index_format_version: u32,
    ) -> Result<Self, CurrentPointerError> {
        let dir_name = dir_name.into();
        validate_current_dir_name(&dir_name)?;
        Ok(Self {
            engine,
            dir_name,
            index_format_version,
        })
    }

    /// Engine family of the active directory.
    #[must_use]
    pub const fn engine(&self) -> BlueGreenEngine {
        self.engine
    }

    /// Plain directory name (never a path) of the active engine directory.
    #[must_use]
    pub fn dir_name(&self) -> &str {
        &self.dir_name
    }

    /// Format version of the named engine directory's index format.
    #[must_use]
    pub const fn index_format_version(&self) -> u32 {
        self.index_format_version
    }

    /// Absolute engine directory for a lexical root.
    #[must_use]
    pub fn engine_dir(&self, lexical_root: &Path) -> PathBuf {
        lexical_root.join(&self.dir_name)
    }

    /// Encode with the registry §7.3 layout: magic, format version, engine
    /// kind, `u16` name length, name bytes, index format version, CRC32.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let name_bytes = self.dir_name.as_bytes();
        let mut bytes = Vec::with_capacity(CURRENT_FIXED_BYTES + name_bytes.len());
        bytes.extend_from_slice(&CURRENT_MAGIC);
        put_u32(&mut bytes, CURRENT_FORMAT_VERSION);
        bytes.push(self.engine.kind_code());
        #[allow(clippy::cast_possible_truncation)]
        put_u16(&mut bytes, name_bytes.len() as u16);
        bytes.extend_from_slice(name_bytes);
        put_u32(&mut bytes, self.index_format_version);
        let checksum = crc32fast::hash(&bytes);
        put_u32(&mut bytes, checksum);
        bytes
    }

    /// Decode and fully validate one CURRENT pointer image.
    ///
    /// The CRC covers every byte before the trailer; content fields are only
    /// interpreted after the checksum verifies, so corruption fails closed.
    ///
    /// # Errors
    ///
    /// Returns [`CurrentPointerError`] for truncation, bad magic, checksum
    /// mismatch, unknown versions or kind codes, trailing bytes, or an
    /// invalid directory name.
    pub fn decode(bytes: &[u8]) -> Result<Self, CurrentPointerError> {
        if bytes.len() < CURRENT_FIXED_BYTES {
            return Err(CurrentPointerError::Truncated {
                actual: bytes.len(),
                minimum: CURRENT_FIXED_BYTES,
            });
        }
        if bytes.len() > MAX_CURRENT_BYTES {
            return Err(CurrentPointerError::LengthMismatch {
                detail: format!(
                    "CURRENT pointer length {} exceeds {MAX_CURRENT_BYTES}",
                    bytes.len()
                ),
            });
        }
        if bytes[..CURRENT_MAGIC.len()] != CURRENT_MAGIC {
            return Err(CurrentPointerError::BadMagic);
        }
        let body_len = bytes.len() - 4;
        let expected_crc = crc32fast::hash(&bytes[..body_len]);
        let actual_crc = u32::from_le_bytes(bytes[body_len..].try_into().map_err(|_| {
            CurrentPointerError::Truncated {
                actual: bytes.len(),
                minimum: CURRENT_FIXED_BYTES,
            }
        })?);
        if actual_crc != expected_crc {
            return Err(CurrentPointerError::CrcMismatch {
                expected_crc,
                actual_crc,
            });
        }
        let mut cursor = ByteCursor::new(&bytes[CURRENT_MAGIC.len()..body_len]);
        let layout_error = |_| CurrentPointerError::LengthMismatch {
            detail: "CURRENT pointer body shorter than its declared layout".to_owned(),
        };
        let format_version = cursor.u32().map_err(layout_error)?;
        if format_version != CURRENT_FORMAT_VERSION {
            return Err(CurrentPointerError::UnsupportedFormatVersion(
                format_version,
            ));
        }
        let kind_code = cursor.u8().map_err(layout_error)?;
        let engine = BlueGreenEngine::from_kind_code(kind_code)
            .ok_or(CurrentPointerError::UnknownEngineKind(kind_code))?;
        let name_len = usize::from(cursor.u16().map_err(layout_error)?);
        if cursor.remaining() != name_len + 4 {
            return Err(CurrentPointerError::LengthMismatch {
                detail: format!(
                    "declared name length {name_len} leaves {} bytes, expected name plus index format version",
                    cursor.remaining()
                ),
            });
        }
        let name_bytes = cursor.take(name_len).map_err(layout_error)?;
        let dir_name = std::str::from_utf8(name_bytes)
            .map_err(|_| CurrentPointerError::DirNameNotUtf8)?
            .to_owned();
        validate_current_dir_name(&dir_name)?;
        let index_format_version = cursor.u32().map_err(layout_error)?;
        Ok(Self {
            engine,
            dir_name,
            index_format_version,
        })
    }
}

/// Validate that `dir_name` names exactly one plain directory entry.
fn validate_current_dir_name(dir_name: &str) -> Result<(), CurrentPointerError> {
    if dir_name.is_empty() {
        return Err(CurrentPointerError::EmptyDirName);
    }
    if dir_name.len() > 65_535 {
        return Err(CurrentPointerError::DirNameTooLong {
            actual: dir_name.len(),
        });
    }
    if dir_name.contains(['\\', '\0']) {
        return Err(CurrentPointerError::UnsafeDirName {
            dir_name: dir_name.to_owned(),
        });
    }
    let mut components = Path::new(dir_name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(CurrentPointerError::UnsafeDirName {
            dir_name: dir_name.to_owned(),
        });
    }
    Ok(())
}

/// Typed CURRENT pointer failures (codec, publication, and resolution).
#[derive(Debug, Error)]
pub enum CurrentPointerError {
    /// Fewer bytes than the fixed pointer layout requires.
    #[error("CURRENT pointer is truncated: {actual} bytes, minimum {minimum}")]
    Truncated {
        /// Bytes actually present.
        actual: usize,
        /// Fixed-layout minimum.
        minimum: usize,
    },
    /// The eight-byte magic did not match `FSLXCUR\0`.
    #[error("CURRENT pointer magic mismatch")]
    BadMagic,
    /// Reader predates the pointer's format version.
    #[error(
        "CURRENT pointer format version {0} is unsupported (reader knows {CURRENT_FORMAT_VERSION})"
    )]
    UnsupportedFormatVersion(u32),
    /// Engine kind byte matches no registry entry.
    #[error("CURRENT pointer engine kind code {0} is unknown")]
    UnknownEngineKind(u8),
    /// Declared and actual field lengths disagree.
    #[error("CURRENT pointer length mismatch: {detail}")]
    LengthMismatch {
        /// Stable mismatch description.
        detail: String,
    },
    /// CRC32 over the body did not match the trailer.
    #[error(
        "CURRENT pointer CRC mismatch: computed {expected_crc:#010x}, stored {actual_crc:#010x}"
    )]
    CrcMismatch {
        /// CRC32 computed over the received body.
        expected_crc: u32,
        /// CRC32 stored in the trailer.
        actual_crc: u32,
    },
    /// Directory name is zero-length.
    #[error("CURRENT pointer directory name is empty")]
    EmptyDirName,
    /// Directory name exceeds its `u16` length field.
    #[error("CURRENT pointer directory name is {actual} bytes, exceeding u16")]
    DirNameTooLong {
        /// Actual name byte length.
        actual: usize,
    },
    /// Directory name bytes are not UTF-8.
    #[error("CURRENT pointer directory name is not valid UTF-8")]
    DirNameNotUtf8,
    /// Directory name is not one plain relative path component.
    #[error("CURRENT pointer directory name {dir_name:?} is not a plain direct child name")]
    UnsafeDirName {
        /// Rejected name.
        dir_name: String,
    },
    /// CURRENT names an engine directory that does not exist.
    #[error(
        "CURRENT pointer names {dir_name:?} under {root}, but that {engine} engine directory is missing; run fsfs doctor"
    )]
    MissingEngineDir {
        /// Lexical root containing the pointer.
        root: PathBuf,
        /// Named engine directory.
        dir_name: String,
        /// Engine family the pointer recorded.
        engine: &'static str,
    },
    /// No CURRENT and several candidate engine directories: refusing to guess.
    #[error(
        "lexical root {root} has no CURRENT pointer and multiple engine directories {candidates:?}; run fsfs doctor to choose one"
    )]
    AmbiguousEngineDirs {
        /// Lexical root scanned.
        root: PathBuf,
        /// Sorted candidate directory names.
        candidates: Vec<String>,
    },
    /// Underlying filesystem failure.
    #[error("CURRENT pointer I/O during {operation} at {path}: {source}")]
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Path being operated on.
        path: PathBuf,
        /// Original error.
        source: io::Error,
    },
}

impl From<CurrentPointerError> for QuillError {
    fn from(error: CurrentPointerError) -> Self {
        match error {
            CurrentPointerError::Io { source, .. } => Self::Io(source),
            other => Self::Invariant {
                detail: other.to_string(),
            },
        }
    }
}

/// Publish `pointer` as the lexical root's CURRENT with the MANIFEST
/// temp+rename+dir-fsync discipline (registry §7.3, §6.2).
///
/// A unique `.tmp-current-<pid>-<n>` sibling is created `O_EXCL`, fully
/// written and fsynced, atomically renamed over `CURRENT`, and sealed with a
/// directory fsync. A crash anywhere before the rename leaves only an inert
/// temp that resolution ignores; the rename itself is the single atomic
/// transition, so flip and rollback are both one swap. Republishing identical
/// content is safe. Stale temps are intentionally left in place: they are
/// tiny, self-describing crash witnesses, and their retirement rides with the
/// flip orchestration layer's human-approved sweep.
///
/// # Errors
///
/// Returns [`CurrentPointerError::Io`] for temp creation, write, fsync,
/// rename, or directory-fsync failures. On non-Unix targets the directory
/// fsync gate fails closed, matching MANIFEST publication.
pub fn publish_current(
    lexical_root: &Path,
    pointer: &CurrentPointer,
) -> Result<(), CurrentPointerError> {
    let encoded = pointer.encode();
    let current_path = lexical_root.join(CURRENT_FILE_NAME);
    let pid = std::process::id();
    let mut attempt = 0_u32;
    loop {
        let temp_path = lexical_root.join(format!(".tmp-current-{pid}-{attempt}"));
        let mut temp_file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                attempt += 1;
                if attempt >= CURRENT_TEMP_COLLISION_LIMIT {
                    return Err(CurrentPointerError::Io {
                        operation: "allocate CURRENT temp",
                        path: temp_path,
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "CURRENT temp collision limit exceeded",
                        ),
                    });
                }
                continue;
            }
            Err(source) => {
                return Err(CurrentPointerError::Io {
                    operation: "create CURRENT temp",
                    path: temp_path,
                    source,
                });
            }
        };
        temp_file
            .write_all(&encoded)
            .and_then(|()| temp_file.sync_all())
            .map_err(|source| CurrentPointerError::Io {
                operation: "persist CURRENT temp",
                path: temp_path.clone(),
                source,
            })?;
        drop(temp_file);
        std::fs::rename(&temp_path, &current_path).map_err(|source| CurrentPointerError::Io {
            operation: "rename CURRENT into place",
            path: current_path.clone(),
            source,
        })?;
        sync_directory(lexical_root).map_err(|source| CurrentPointerError::Io {
            operation: "fsync CURRENT directory",
            path: lexical_root.to_path_buf(),
            source: match source {
                KeeperError::Io { source, .. } => source,
                other => io::Error::other(other.to_string()),
            },
        })?;
        return Ok(());
    }
}

/// Outcome of resolving a lexical root's active engine directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedCurrent {
    /// A valid CURRENT pointer was present.
    Pointer(CurrentPointer),
    /// No CURRENT existed; exactly one engine directory was found and adopted
    /// by writing a CURRENT pointer for it (migration bootstrap, registry
    /// §7.3).
    Adopted(CurrentPointer),
    /// No CURRENT and no engine directories: a fresh root.
    Empty,
}

impl ResolvedCurrent {
    /// The active pointer when one exists (present or adopted).
    #[must_use]
    pub fn pointer(&self) -> Option<&CurrentPointer> {
        match self {
            Self::Pointer(pointer) | Self::Adopted(pointer) => Some(pointer),
            Self::Empty => None,
        }
    }
}

/// Resolve the active engine directory under `lexical_root` (registry §7.3).
///
/// Order of decision: a present CURRENT is decoded, checksum-verified, and
/// its named directory must exist; an absent CURRENT triggers the adoption
/// scan — directories containing `MANIFEST` (quill) or `meta.json` (tantivy)
/// are engine candidates, exactly one candidate is adopted by publishing a
/// pointer for it, zero candidates yield [`ResolvedCurrent::Empty`], and
/// several candidates fail closed demanding doctor. Anything that is not a
/// real directory bearing an engine marker — user files, nested data,
/// `.tmp-*` crash witnesses — is ignored and never modified.
///
/// # Errors
///
/// Returns [`CurrentPointerError`] for corrupt or unreadable CURRENT bytes,
/// a pointer naming a missing directory, ambiguous engine directories, or
/// underlying I/O failure.
pub fn resolve_current(lexical_root: &Path) -> Result<ResolvedCurrent, CurrentPointerError> {
    let current_path = lexical_root.join(CURRENT_FILE_NAME);
    match read_current_file(&current_path)? {
        Some(bytes) => {
            let pointer = CurrentPointer::decode(&bytes)?;
            let engine_dir = pointer.engine_dir(lexical_root);
            if !engine_dir.is_dir() {
                return Err(CurrentPointerError::MissingEngineDir {
                    root: lexical_root.to_path_buf(),
                    dir_name: pointer.dir_name().to_owned(),
                    engine: pointer.engine().label(),
                });
            }
            Ok(ResolvedCurrent::Pointer(pointer))
        }
        None => adopt_or_report_empty(lexical_root),
    }
}

/// Read a CURRENT file with a no-follow open and a bounded buffer.
fn read_current_file(current_path: &Path) -> Result<Option<Vec<u8>>, CurrentPointerError> {
    let mut file = match open_manifest_slot(current_path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(CurrentPointerError::Io {
                operation: "open CURRENT",
                path: current_path.to_path_buf(),
                source,
            });
        }
    };
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(usize_to_u64(MAX_CURRENT_BYTES).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| CurrentPointerError::Io {
            operation: "read CURRENT",
            path: current_path.to_path_buf(),
            source,
        })?;
    Ok(Some(bytes))
}

/// Scan `lexical_root` for engine directories and apply the adoption rule.
fn adopt_or_report_empty(lexical_root: &Path) -> Result<ResolvedCurrent, CurrentPointerError> {
    let entries = std::fs::read_dir(lexical_root).map_err(|source| CurrentPointerError::Io {
        operation: "scan lexical root for engine directories",
        path: lexical_root.to_path_buf(),
        source,
    })?;
    let mut candidates: Vec<(String, BlueGreenEngine)> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| CurrentPointerError::Io {
            operation: "read lexical root entry",
            path: lexical_root.to_path_buf(),
            source,
        })?;
        let file_type = entry
            .file_type()
            .map_err(|source| CurrentPointerError::Io {
                operation: "stat lexical root entry",
                path: entry.path(),
                source,
            })?;
        if !file_type.is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        // MANIFEST wins when both markers exist: quill is the forward format.
        let engine = if entry
            .path()
            .join(BlueGreenEngine::Quill.adoption_marker())
            .is_file()
        {
            Some(BlueGreenEngine::Quill)
        } else if entry
            .path()
            .join(BlueGreenEngine::Tantivy.adoption_marker())
            .is_file()
        {
            Some(BlueGreenEngine::Tantivy)
        } else {
            None
        };
        if let Some(engine) = engine {
            candidates.push((name, engine));
        }
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    match candidates.len() {
        0 => Ok(ResolvedCurrent::Empty),
        1 => {
            let (dir_name, engine) = candidates.remove(0);
            let pointer = CurrentPointer::new(engine, dir_name, engine.adopted_format_version())?;
            publish_current(lexical_root, &pointer)?;
            Ok(ResolvedCurrent::Adopted(pointer))
        }
        _ => Err(CurrentPointerError::AmbiguousEngineDirs {
            root: lexical_root.to_path_buf(),
            candidates: candidates.into_iter().map(|(name, _)| name).collect(),
        }),
    }
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

// ============================================================================
// Segment statistics (visibility contract: published_generation,
// last_publish_unix from the MANIFEST v2 freshness witness, live_writer from
// the D1 LOCK record + POSIX liveness).
// ============================================================================

/// Best-effort total of Quill-managed bytes under `directory`.
///
/// Sums FSLX segments, MANIFEST slots and repair sidecars, and lifecycle
/// metadata (LOCK). Entries that vanish or cannot be stat'ed mid-walk are
/// skipped: this is a status number, never an accounting authority.
// Engine-managed filenames are canonical lowercase by construction; a
// case-insensitive compare would mis-count foreign files the engine never
// wrote.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn managed_disk_bytes(directory: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return 0;
    };
    let mut total = 0_u64;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let managed = (name.starts_with("seg-") && name.ends_with(".fslx"))
            || parse_quarantined_segment_name(name).is_some()
            || parse_quarantined_sidecar_segment_name(name).is_some()
            || matches!(name, "MANIFEST" | "MANIFEST.prev" | "LOCK")
            || name.ends_with(".fec");
        if !managed {
            continue;
        }
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

/// Whether a live writer currently holds `directory`'s LOCK.
///
/// Reads the D1 LOCK record and applies the POSIX `kill(pid, 0)` liveness
/// rule: only a valid record whose pid is demonstrably alive counts. Any
/// read/parse failure conservatively reports no live writer. Non-Unix targets
/// cannot prove liveness, so a valid record alone decides there.
fn detect_live_writer(directory: &Path) -> bool {
    let lock_path = directory.join("LOCK");
    let Ok(mut file) = File::open(&lock_path) else {
        return false;
    };
    let Ok(Some(record)) = read_writer_lock_record(&lock_path, &mut file) else {
        return false;
    };
    !writer_pid_is_dead(record.pid)
}

impl SegmentStatsProvider for KeeperSnapshot {
    fn segment_stats(&self) -> SegmentStats {
        let manifest = &self.loaded.manifest;
        let (managed_disk_bytes, live_writer) =
            self.directory.as_ref().map_or((0, false), |directory| {
                (managed_disk_bytes(directory), detect_live_writer(directory))
            });
        SegmentStats {
            schema_id: manifest.schema_id,
            published_generation: manifest.generation,
            sealed_segments: self.segments.len(),
            // The searchable delta segment is E5; nothing pre-delta exists yet.
            delta_segments: 0,
            live_docs: usize::try_from(self.live_doc_count).unwrap_or(usize::MAX),
            tombstones: usize::try_from(self.tombstone_count).unwrap_or(usize::MAX),
            managed_disk_bytes,
            delta_memory_bytes: 0,
            last_publish_unix: (manifest.last_publish_unix_s != 0)
                .then_some(manifest.last_publish_unix_s),
            live_writer,
            degraded: self.is_degraded(),
            quarantined_segments: self.quarantined_segments.len(),
            estimated_missing_docs: self.estimated_missing_docs(),
            unknown_missing_doc_segments: self
                .quarantined_segments
                .iter()
                .filter(|segment| segment.estimated_missing_docs.is_none())
                .count(),
        }
    }
}

impl SegmentStatsProvider for KeeperWriter {
    fn segment_stats(&self) -> SegmentStats {
        let mut stats = self.snapshot.segment_stats();
        // The writer holds LOCK by construction; no probe is needed.
        stats.live_writer = true;
        stats
    }
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    const fn position(&self) -> usize {
        self.position
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ManifestCodecError> {
        let end =
            self.position
                .checked_add(length)
                .ok_or_else(|| ManifestCodecError::Truncated {
                    offset: self.position,
                    needed: length,
                    remaining: self.remaining(),
                })?;
        let Some(bytes) = self.bytes.get(self.position..end) else {
            return Err(ManifestCodecError::Truncated {
                offset: self.position,
                needed: length,
                remaining: self.remaining(),
            });
        };
        self.position = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ManifestCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ManifestCodecError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ManifestCodecError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ManifestCodecError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use asupersync::runtime::yield_now;
    use asupersync::types::Budget;
    use asupersync::{LabConfig, LabRuntime};
    #[cfg(feature = "durability")]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileHealth};
    use tempfile::tempdir;

    use crate::quiver::{EncodedIdHashSection, EncodedIdMapSection, IdMapEntryInput};
    use crate::schema::{DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA};
    #[cfg(feature = "durability")]
    use crate::segment::SegmentWriteCheckpoint;
    use crate::segment::{EncodedSegment, SectionInput, SegmentHeaderInput};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn tier_test_segment(segment_id: u64, docid_lo: u64, docid_hi: u64) -> ManifestSegment {
        ManifestSegment {
            segment_id,
            seal_seq: segment_id,
            file_len: 1,
            file_xxh3: segment_id,
            docid_lo,
            docid_hi,
            doc_count: u32::try_from(docid_hi - docid_lo).expect("test segment width fits u32"),
            tombstones: TombstoneSet::default(),
        }
    }

    fn apply_tier_test_plan(
        segments: &mut Vec<ManifestSegment>,
        plan: &TierMergePlan,
        output_segment_id: u64,
    ) {
        let start = segments
            .iter()
            .position(|segment| segment.segment_id == plan.source_segment_ids[0])
            .expect("planned first source remains live");
        let end = start + plan.source_segment_ids.len();
        assert_eq!(
            segments[start..end]
                .iter()
                .map(|segment| segment.segment_id)
                .collect::<Vec<_>>(),
            plan.source_segment_ids,
            "policy plan must name one uninterrupted manifest slice",
        );
        let doc_count = segments[start..end]
            .iter()
            .map(|segment| segment.doc_count)
            .sum();
        segments.splice(
            start..end,
            [ManifestSegment {
                segment_id: output_segment_id,
                seal_seq: output_segment_id,
                file_len: 1,
                file_xxh3: output_segment_id,
                docid_lo: plan.docid_lo,
                docid_hi: plan.docid_hi,
                doc_count,
                tombstones: TombstoneSet::default(),
            }],
        );
    }

    fn assert_pairwise_disjoint(segments: &[ManifestSegment]) {
        for pair in segments.windows(2) {
            assert!(
                pair[0].docid_hi <= pair[1].docid_lo,
                "covering intervals overlap: {:?} then {:?}",
                pair[0],
                pair[1],
            );
        }
    }

    #[test]
    fn tier_planner_classifies_widths_and_selects_only_bound_consecutive_runs() {
        let policy = TierMergePolicy {
            fanout: 3,
            small_max_docid_width: 16,
            medium_max_docid_width: 64,
            max_hole_ratio: 0.5,
        };
        assert_eq!(policy.classify_width(16), SegmentSizeTier::Small);
        assert_eq!(policy.classify_width(17), SegmentSizeTier::Medium);
        assert_eq!(policy.classify_width(65), SegmentSizeTier::Large);

        let segments = vec![
            tier_test_segment(1, 0, 4),
            tier_test_segment(2, 6, 10),
            tier_test_segment(3, 12, 16),
            tier_test_segment(4, 80, 112),
        ];
        let plan = plan_tier_merge(&segments, policy)
            .expect("valid policy")
            .expect("three small consecutive sources merge");
        assert_eq!(plan.source_segment_ids, [1, 2, 3]);
        assert_eq!((plan.docid_lo, plan.docid_hi), (0, 16));
        assert!((plan.hole_ratio - 0.25).abs() < f64::EPSILON);

        let rejected = plan_tier_merge(
            &segments,
            TierMergePolicy {
                max_hole_ratio: 0.24,
                ..policy
            },
        )
        .expect("valid restrictive policy");
        assert!(rejected.is_none(), "hole gate must decline the run");

        let mut empty = tier_test_segment(9, 20, 21);
        empty.docid_hi = empty.docid_lo;
        assert!(matches!(
            plan_tier_merge(&[empty], policy),
            Err(TierPolicyError::InvalidSegmentRange { segment_id: 9, .. })
        ));
        assert!(matches!(
            plan_tier_merge(&[segments[1].clone(), segments[0].clone()], policy),
            Err(TierPolicyError::InvalidSegmentOrder {
                left_segment_id: 2,
                right_segment_id: 1,
            })
        ));
    }

    #[test]
    fn tier_policy_bounds_segment_count_under_ten_thousand_watch_batches() {
        let policy = TierMergePolicy {
            fanout: 8,
            small_max_docid_width: 64,
            medium_max_docid_width: 512,
            max_hole_ratio: 0.5,
        };
        let mut segments = Vec::new();
        let mut next_output_id = 10_001_u64;
        for batch in 0..10_000_u64 {
            segments.push(tier_test_segment(batch + 1, batch, batch + 1));
            while let Some(plan) = plan_tier_merge(&segments, policy).expect("valid watch policy") {
                apply_tier_test_plan(&mut segments, &plan, next_output_id);
                next_output_id += 1;
            }
        }
        let tier_count_bound = 3 * (policy.fanout - 1);
        assert!(
            segments.len() <= tier_count_bound,
            "{} live segments exceeded the S/M/L bound {tier_count_bound}",
            segments.len(),
        );
        assert!(
            plan_tier_merge(&segments, policy)
                .expect("valid final policy")
                .is_none(),
            "watch simulation must drain every eligible same-tier run",
        );
        assert_pairwise_disjoint(&segments);
    }

    #[test]
    fn interleaved_multi_shard_random_merge_schedule_preserves_q1_intervals() {
        let policy = TierMergePolicy {
            fanout: 4,
            small_max_docid_width: 65_536,
            medium_max_docid_width: 8 * 65_536,
            max_hole_ratio: 1.0,
        };
        let mut rng = 0x51a7_d15c_0de5_1234_u64;
        let mut next_ord = [0_u64; 4];
        let mut lease_base = [None; 4];
        let mut next_block_base = 0_u64;
        let mut segments = Vec::new();
        let mut next_id = 1_u64;
        for _ in 0..2_000 {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let random = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let shard = usize::try_from(random % 4).expect("shard fits usize");
            let width = 1 + (random >> 8) % 17;
            let base = *lease_base[shard].get_or_insert_with(|| {
                let granted = next_block_base;
                next_block_base += 65_536;
                next_ord[shard] = 0;
                granted
            });
            let lo = base + next_ord[shard];
            let hi = lo + width;
            next_ord[shard] += width;
            segments.push(tier_test_segment(next_id, lo, hi));
            next_id += 1;
            segments.sort_unstable_by_key(|segment| segment.docid_lo);
            assert_pairwise_disjoint(&segments);

            let merge_budget = usize::try_from((random >> 16) % 5).expect("budget fits usize");
            let mut retired_for_merge = false;
            for _ in 0..merge_budget {
                let Some(plan) = plan_tier_merge(&segments, policy).expect("valid random policy")
                else {
                    break;
                };
                apply_tier_test_plan(&mut segments, &plan, next_id);
                next_id += 1;
                if !retired_for_merge {
                    // Production concat retires every live shard session so a
                    // later append cannot land inside the new union hull.
                    lease_base.fill(None);
                    retired_for_merge = true;
                }
                assert_pairwise_disjoint(&segments);
            }
        }
        while let Some(plan) = plan_tier_merge(&segments, policy).expect("valid drain policy") {
            apply_tier_test_plan(&mut segments, &plan, next_id);
            next_id += 1;
            assert_pairwise_disjoint(&segments);
        }
    }

    #[test]
    fn rank_pruning_cache_caps_unique_terms_and_rejects_dictionary_drift() {
        let cache = RankPruningCache::new();
        let term_metadata =
            TermMetadata::without_positions(1, ByteSpan::new(0, 1), ByteSpan::new(0, 1));
        for term_ord in
            0..u32::try_from(MAX_RANK_PRUNING_CACHE_TERMS + 16).expect("cache test count fits u32")
        {
            cache
                .insert(
                    term_ord,
                    term_metadata,
                    Arc::new(ValidatedTermPruningMetadata::empty_for_cache_test()),
                )
                .expect("bounded cache admission");
        }
        assert_eq!(cache.len(), MAX_RANK_PRUNING_CACHE_TERMS);
        assert!(
            cache
                .get(
                    u32::try_from(MAX_RANK_PRUNING_CACHE_TERMS).expect("cache limit fits u32"),
                    term_metadata,
                )
                .expect("bounded cache lookup")
                .is_none(),
            "terms beyond the cap must remain query-local"
        );

        let drifted = TermMetadata::without_positions(1, ByteSpan::new(1, 1), ByteSpan::new(0, 1));
        assert!(cache.get(0, drifted).is_err());
    }

    fn array_tombstone_bytes(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 1);
        put_u16(&mut bytes, chunk_id);
        bytes.push(0);
        put_u16(
            &mut bytes,
            u16::try_from(lows.len()).expect("test tombstone count fits u16"),
        );
        for low in lows {
            put_u16(&mut bytes, *low);
        }
        bytes
    }

    fn array_tombstones(chunk_id: u16, lows: &[u16]) -> TombstoneSet {
        TombstoneSet::from_bytes(&array_tombstone_bytes(chunk_id, lows))
            .expect("test array tombstones are canonical")
    }

    fn bitmap_tombstone_bytes(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 1);
        put_u16(&mut bytes, chunk_id);
        bytes.push(1);
        let encoded_count = if lows.len() == 65_536 {
            0
        } else {
            u16::try_from(lows.len()).expect("test bitmap count fits u16")
        };
        put_u16(&mut bytes, encoded_count);
        bytes.resize(bytes.len() + 8_192, 0);
        let payload_start = bytes.len() - 8_192;
        for low in lows {
            let low = usize::from(*low);
            bytes[payload_start + low / 8] |= 1 << (low % 8);
        }
        bytes
    }

    fn bitmap_tombstones(chunk_id: u16, lows: &[u16]) -> TombstoneSet {
        TombstoneSet::from_bytes(&bitmap_tombstone_bytes(chunk_id, lows))
            .expect("test bitmap tombstones are canonical")
    }

    fn sample_manifest(generation: u64) -> Manifest {
        Manifest {
            generation,
            docid_high_watermark: 70_000 + generation,
            schema_id: 0x1122_3344_5566_7788,
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            last_publish_unix_s: 1_700_000_000,
            segments: vec![
                ManifestSegment {
                    segment_id: 0x1001,
                    seal_seq: 7,
                    file_len: 4_096,
                    file_xxh3: 0xa1a2_a3a4_a5a6_a7a8,
                    docid_lo: 100,
                    docid_hi: 200,
                    doc_count: 3,
                    tombstones: array_tombstones(0, &[101, 150]),
                },
                ManifestSegment {
                    segment_id: 0x1002,
                    seal_seq: 8,
                    file_len: 8_192,
                    file_xxh3: 0xb1b2_b3b4_b5b6_b7b8,
                    docid_lo: 65_536,
                    docid_hi: 65_600,
                    doc_count: 2,
                    tombstones: TombstoneSet::new(),
                },
            ],
            field_stats: vec![
                ManifestFieldStats {
                    field_ord: 1,
                    total_tokens: 12,
                    doc_count: 5,
                },
                ManifestFieldStats {
                    field_ord: 3,
                    total_tokens: 42,
                    doc_count: 5,
                },
            ],
        }
    }

    fn rewrite_crc(bytes: &mut [u8]) {
        let body_len = bytes.len() - 4;
        let checksum = crc32fast::hash(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&checksum.to_le_bytes());
    }

    fn write_manifest(path: &Path, manifest: &Manifest) -> TestResult {
        let bytes = manifest.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    fn encoded_test_segment(
        segment_id: u64,
        docid_lo: u64,
        docid_hi: u64,
        doc_count: u32,
    ) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
        let span = usize::try_from(docid_hi.saturating_sub(docid_lo))?;
        let document_ids = (0..span)
            .map(|ordinal| format!("doc-{segment_id:016x}-{ordinal}"))
            .collect::<Vec<_>>();
        let identifiers = document_ids
            .iter()
            .enumerate()
            .map(|(ordinal, document_id)| {
                (ordinal < usize::try_from(doc_count).unwrap_or(usize::MAX))
                    .then_some(document_id.as_str())
            })
            .collect::<Vec<_>>();
        encoded_identity_test_segment(segment_id, docid_lo, &identifiers)
    }

    fn encoded_identity_test_segment(
        segment_id: u64,
        docid_lo: u64,
        document_ids: &[Option<&str>],
    ) -> Result<EncodedSegment, Box<dyn std::error::Error>> {
        let docid_hi = docid_lo
            .checked_add(u64::try_from(document_ids.len())?)
            .ok_or_else(|| QuillError::Invariant {
                detail: "test IDMAP range overflow".to_owned(),
            })?;
        let entries = document_ids
            .iter()
            .enumerate()
            .map(|(ordinal, document_id)| {
                document_id.map(|document_id| {
                    IdMapEntryInput::new(
                        document_id,
                        u64::try_from(ordinal).unwrap_or(u64::MAX).saturating_add(1),
                    )
                })
            })
            .collect::<Vec<_>>();
        let doc_count = u32::try_from(entries.iter().flatten().count())?;
        let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &entries)?;
        let id_hash = EncodedIdHashSection::encode(id_map.section()?)?;
        let mut numeric_entries = Vec::new();
        for (ordinal, document_id) in document_ids.iter().enumerate() {
            if document_id.is_none() {
                continue;
            }
            let global_docid = docid_lo
                .checked_add(u64::try_from(ordinal)?)
                .ok_or_else(|| QuillError::Invariant {
                    detail: "test numeric docid overflow".to_owned(),
                })?;
            numeric_entries.push(NumericEntry::u64(
                global_docid,
                u32::try_from(global_docid)?,
            ));
        }
        let numeric_inputs = [NumericFieldInput::new(4, &numeric_entries)];
        let numeric =
            EncodedNumericSection::encode(DEFAULT_SCHEMA, docid_lo, docid_hi, &numeric_inputs)?;
        Ok(EncodedSegment::encode(
            SegmentHeaderInput {
                segment_id,
                schema: DEFAULT_SCHEMA,
                docid_lo,
                docid_hi,
                doc_count,
                created_unix_s: 1_700_000_000,
                engine_version: CURRENT_ENGINE_VERSION,
            },
            &[
                SectionInput::new(SectionKind::TERMDICT, b"termdict"),
                SectionInput::new(SectionKind::POSTINGS, b"postings"),
                SectionInput::new(SectionKind::POSITIONS, b"positions"),
                SectionInput::new(SectionKind::BLOCKMAX, b"blockmax"),
                SectionInput::new(SectionKind::DOCLEN, b"doclen"),
                SectionInput::new(SectionKind::IDMAP, id_map.as_bytes()),
                SectionInput::new(SectionKind::IDHASH, id_hash.as_bytes()),
                SectionInput::new(SectionKind::NUMERIC, numeric.as_bytes()),
                SectionInput::new(SectionKind::STOREDMETA, b"storedmeta"),
                SectionInput::new(SectionKind::STATS, b"stats"),
            ],
        )?)
    }

    fn write_identity_test_segment(
        directory: &Path,
        segment_id: u64,
        seal_seq: u64,
        docid_lo: u64,
        document_ids: &[Option<&str>],
    ) -> Result<ManifestSegment, Box<dyn std::error::Error>> {
        let encoded = encoded_identity_test_segment(segment_id, docid_lo, document_ids)?;
        std::fs::write(
            directory.join(canonical_segment_name(segment_id)),
            encoded.as_bytes(),
        )?;
        Ok(manifest_segment(&encoded, seal_seq))
    }

    fn write_test_segment(
        directory: &Path,
        segment_id: u64,
        seal_seq: u64,
        docid_lo: u64,
        docid_hi: u64,
    ) -> Result<ManifestSegment, Box<dyn std::error::Error>> {
        let encoded = encoded_test_segment(segment_id, docid_lo, docid_hi, 1)?;
        std::fs::write(
            directory.join(canonical_segment_name(segment_id)),
            encoded.as_bytes(),
        )?;
        Ok(manifest_segment(&encoded, seal_seq))
    }

    fn manifest_segment(encoded: &EncodedSegment, seal_seq: u64) -> ManifestSegment {
        let header = encoded.header();
        ManifestSegment {
            segment_id: header.segment_id,
            seal_seq,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            tombstones: TombstoneSet::new(),
        }
    }

    fn durable_test_manifest(generation: u64, segments: Vec<ManifestSegment>) -> Manifest {
        Manifest {
            generation,
            docid_high_watermark: segments
                .iter()
                .map(|segment| segment.docid_hi)
                .max()
                .unwrap_or(0),
            schema_id: DEFAULT_SCHEMA.schema_id().expect("valid test schema"),
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            last_publish_unix_s: 1_700_000_000,
            segments,
            field_stats: Vec::new(),
        }
    }

    #[cfg(feature = "durability")]
    fn test_file_protector() -> FileProtector {
        FileProtector::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                repair_overhead: 2.0,
                ..DurabilityConfig::default()
            },
        )
        .expect("valid test durability configuration")
    }

    fn directory_bytes(directory: &Path) -> Result<Vec<(OsString, Vec<u8>)>, io::Error> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                entries.push((entry.file_name(), std::fs::read(entry.path())?));
            }
        }
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        Ok(entries)
    }

    #[test]
    fn packed_engine_version_matches_crate_semver() {
        let major = env!("CARGO_PKG_VERSION_MAJOR")
            .parse::<u8>()
            .expect("crate major fits packed engine version");
        let minor = env!("CARGO_PKG_VERSION_MINOR")
            .parse::<u8>()
            .expect("crate minor fits packed engine version");
        let patch = env!("CARGO_PKG_VERSION_PATCH")
            .parse::<u16>()
            .expect("crate patch fits packed engine version");
        assert_eq!(
            CURRENT_ENGINE_VERSION,
            pack_engine_version(major, minor, patch)
        );
        assert_eq!(
            unpack_engine_version(CURRENT_ENGINE_VERSION),
            (major, minor, patch)
        );
    }

    #[test]
    fn referenced_segment_unknown_schema_maps_to_public_corruption() {
        let path = PathBuf::from("index/seg-0000000000000001.fslx");
        let error: SearchError = KeeperError::SegmentOpen {
            path: path.clone(),
            source: QuillError::UnknownSchema { schema_id: 0x55 },
        }
        .into();
        assert!(matches!(
            error,
            SearchError::IndexCorrupted { path: actual, detail }
                if actual == path && detail.contains("unknown schema")
        ));
    }

    #[test]
    fn empty_manifest_has_stable_wire_golden() -> TestResult {
        let manifest = Manifest::empty(1, 0x1122_3344_5566_7788, 0);
        let bytes = manifest.to_bytes()?;
        let expected = hex_bytes(
            "46534c584d414e0002000000010000000000000000000000000000008877665544332211\
             01000200000000000000000000000000000000000000000087ee6320",
        );
        assert_eq!(bytes, expected);
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn zero_schema_hash_is_a_valid_roundtrip() -> TestResult {
        let manifest = Manifest::empty(1, 0, 0);
        assert_eq!(Manifest::from_bytes(&manifest.to_bytes()?)?, manifest);
        Ok(())
    }

    #[test]
    fn nonempty_manifest_has_stable_wire_golden() -> TestResult {
        let manifest = Manifest {
            generation: 0x0102_0304_0506_0708,
            docid_high_watermark: 20,
            schema_id: 0x1122_3344_5566_7788,
            engine_version: 0x0102_0304,
            flags: MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
            last_publish_unix_s: 1_700_000_000,
            segments: vec![ManifestSegment {
                segment_id: 0x1020_3040_5060_7080,
                seal_seq: 0x1112_1314_1516_1718,
                file_len: 0x2122_2324_2526_2728,
                file_xxh3: 0x3132_3334_3536_3738,
                docid_lo: 10,
                docid_hi: 20,
                doc_count: 2,
                tombstones: array_tombstones(0, &[12, 18]),
            }],
            field_stats: vec![ManifestFieldStats {
                field_ord: 3,
                total_tokens: 0x4142_4344_4546_4748,
                doc_count: 2,
            }],
        };
        let bytes = manifest.to_bytes()?;
        // v2 golden: version 2, `last_publish_unix_s` (1_700_000_000) after
        // flags, then the v1 segment/tombstone/stats layout unchanged.
        let expected = hex_bytes(
            "46534c584d414e0002000000080706050403020114000000000000008877665544332211\
             040302010100000000f15365000000000100000080706050403020101817161514131211\
             282726252423222138373635343332310a00000000000000140000000000000002000000\
             0100000000000002000c001200010000000300484746454443424102000000122be130",
        );
        assert_eq!(
            bytes, expected,
            "nonempty wire golden pins segment, tombstone, and stats field order"
        );
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn nonempty_manifest_roundtrips_exactly() -> TestResult {
        let manifest = sample_manifest(9);
        let bytes = manifest.to_bytes()?;
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn every_truncation_and_checksum_flip_is_rejected_without_panic() -> TestResult {
        let bytes = sample_manifest(3).to_bytes()?;
        for length in 0..bytes.len() {
            let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&bytes[..length]));
            assert!(parsed.is_ok(), "parser panicked at truncation {length}");
            assert!(parsed.expect("catch_unwind result").is_err());
        }
        let body = &bytes[..bytes.len() - 4];
        for length in 0..body.len() {
            let mut valid_crc_prefix = body[..length].to_vec();
            put_u32(&mut valid_crc_prefix, crc32fast::hash(&body[..length]));
            let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&valid_crc_prefix));
            assert!(
                parsed.is_ok(),
                "parser panicked at valid-CRC body truncation {length}"
            );
            assert!(parsed.expect("catch_unwind result").is_err());
        }
        for index in [0, 8, 44, bytes.len() / 2, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 0x80;
            assert!(matches!(
                Manifest::from_bytes(&corrupt),
                Err(ManifestCodecError::ChecksumMismatch { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn valid_crc_cannot_hide_hostile_counts_or_unknown_flags() -> TestResult {
        let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
        bytes[40..44].copy_from_slice(&0x8000_0000_u32.to_le_bytes());
        rewrite_crc(&mut bytes);
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
        let hostile_segment_count =
            u32::try_from(MAX_MANIFEST_SEGMENTS + 1).expect("segment cap fits u32");
        bytes[52..56].copy_from_slice(&hostile_segment_count.to_le_bytes());
        rewrite_crc(&mut bytes);
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(ManifestCodecError::ResourceLimit {
                resource: "segment count",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn q1_validation_rejects_overlap_duplicates_and_watermark_rollback() {
        let mut manifest = sample_manifest(1);
        manifest.segments[1].docid_lo = 150;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.segments[1].segment_id = manifest.segments[0].segment_id;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.segments[1].seal_seq = manifest.segments[0].seal_seq;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.docid_high_watermark = manifest.segments[1].docid_hi - 1;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn tombstones_must_be_canonical_and_inside_the_segment_range() {
        let mut manifest = sample_manifest(1);
        manifest.segments[0].tombstones = array_tombstones(0, &[99]);
        assert!(manifest.validate().is_err());

        assert!(TombstoneSet::from_bytes(&array_tombstone_bytes(0, &[150, 150])).is_err());

        let mut bitmap = vec![1, 0, 0, 0, 0, 0, 1, 1, 0];
        bitmap.extend_from_slice(&[0; 8_192]);
        bitmap[9] = 1;
        assert!(TombstoneSet::from_bytes(&bitmap).is_err());
    }

    #[test]
    fn tombstone_container_thresholds_and_chunk_count_are_bounded() {
        let oversized_lows = (0..=TOMBSTONE_ARRAY_MAX_CARDINALITY).collect::<Vec<_>>();
        assert!(matches!(
            validate_tombstone_bytes(&array_tombstone_bytes(0, &oversized_lows), None),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        assert!(matches!(
            validate_tombstone_bytes(&bitmap_tombstone_bytes(0, &[7]), None),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        let dense_lows = (0..u16::try_from(TOMBSTONE_BITMAP_MIN_CARDINALITY)
            .expect("bitmap threshold fits u16"))
            .collect::<Vec<_>>();
        let dense = bitmap_tombstones(0, &dense_lows);
        assert_eq!(
            validate_tombstone_bytes(dense.as_bytes(), Some((0, 65_536)))
                .expect("dense bitmap is canonical"),
            TOMBSTONE_BITMAP_MIN_CARDINALITY
        );
        assert!(validate_tombstone_bytes(dense.as_bytes(), Some((100, 65_536))).is_err());

        let mut too_many_chunks = Vec::new();
        put_u32(
            &mut too_many_chunks,
            u32::try_from(MAX_TOMBSTONE_CHUNKS + 1).expect("chunk cap fits u32"),
        );
        assert!(matches!(
            validate_tombstone_bytes(&too_many_chunks, None),
            Err(ManifestCodecError::ResourceLimit {
                resource: "tombstone chunk count",
                ..
            })
        ));

        let sparse = array_tombstones(0, &[1, 100]);
        let dense_lows = (0..=TOMBSTONE_ARRAY_MAX_CARDINALITY).collect::<Vec<_>>();
        let dense = bitmap_tombstones(0, &dense_lows);
        assert!(
            dense
                .is_monotone_superset_of(&sparse)
                .expect("compare array to bitmap")
        );
        let missing = array_tombstones(0, &[1, 5_000]);
        assert!(
            !dense
                .is_monotone_superset_of(&missing)
                .expect("detect missing bitmap bit")
        );
    }

    #[test]
    fn tombstone_set_mutation_preserves_hysteresis_and_exact_roundtrip() -> TestResult {
        let sparse_lows = (0..4_095_u16).collect::<Vec<_>>();
        let mut set = array_tombstones(0, &sparse_lows);
        assert_eq!(set.cardinality(), 4_095);
        assert_eq!(set.as_bytes()[6], 0);

        assert!(set.insert(4_095)?);
        assert_eq!(set.cardinality(), 4_096);
        assert_eq!(set.as_bytes()[6], 0, "4,096 remains ARRAY");
        let array_wire = set.as_bytes().to_vec();
        assert_eq!(
            TombstoneSet::from_bytes(&array_wire)?.as_bytes(),
            array_wire
        );

        assert!(set.insert(4_096)?);
        assert_eq!(set.cardinality(), 4_097);
        assert_eq!(set.as_bytes()[6], 1, "4,097 promotes to BITMAP");
        assert!((0..=4_096_u32).all(|docid| set.contains(docid)));
        let bitmap_wire = set.as_bytes().to_vec();
        assert_eq!(
            TombstoneSet::from_bytes(&bitmap_wire)?.as_bytes(),
            bitmap_wire
        );

        let dense_lows = (0..3_585_u16).collect::<Vec<_>>();
        let mut dense = bitmap_tombstones(0, &dense_lows);
        assert!(dense.remove(3_584)?);
        assert_eq!(dense.cardinality(), 3_584);
        assert_eq!(dense.as_bytes()[6], 1, "3,584 retains BITMAP");
        assert!(dense.remove(3_583)?);
        assert_eq!(dense.cardinality(), 3_583);
        assert_eq!(dense.as_bytes()[6], 0, "3,583 demotes to ARRAY");
        Ok(())
    }

    #[test]
    fn tombstone_set_handles_chunk_edges_duplicates_and_full_bitmap() -> TestResult {
        let mut set = TombstoneSet::new();
        for docid in [u32::MAX, 0, 65_536 + 7, 65_535] {
            assert!(set.insert(docid)?);
        }
        assert!(!set.insert(u32::MAX)?);
        assert_eq!(set.cardinality(), 4);
        assert!(
            [u32::MAX, 0, 65_536 + 7, 65_535]
                .into_iter()
                .all(|docid| set.contains(docid))
        );
        assert_eq!(tombstone_chunk_count(set.as_bytes())?, 3);
        assert_eq!(TombstoneSet::from_bytes(set.as_bytes())?, set);

        let all_lows = (0..=u16::MAX).collect::<Vec<_>>();
        let mut full = bitmap_tombstones(u16::MAX, &all_lows);
        assert_eq!(full.cardinality(), 65_536);
        assert_eq!(&full.as_bytes()[7..9], &[0, 0]);
        assert!(full.remove(u32::MAX)?);
        assert_eq!(&full.as_bytes()[7..9], &u16::MAX.to_le_bytes());
        assert!(full.insert(u32::MAX)?);
        assert_eq!(&full.as_bytes()[7..9], &[0, 0]);
        Ok(())
    }

    #[test]
    fn tombstone_successors_preserve_representation_state() -> TestResult {
        let array_4_096 = array_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
        let bitmap_4_097 = bitmap_tombstones(0, &(0..4_097_u16).collect::<Vec<_>>());
        assert!(bitmap_4_097.is_monotone_superset_of(&array_4_096)?);

        let bitmap_3_584 = bitmap_tombstones(0, &(0..3_584_u16).collect::<Vec<_>>());
        let array_3_585 = array_tombstones(0, &(0..3_585_u16).collect::<Vec<_>>());
        assert!(
            !array_3_585.is_monotone_superset_of(&bitmap_3_584)?,
            "retained BITMAP cannot demote in a growth-only generation"
        );

        let array_overlap = array_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
        let bitmap_overlap = bitmap_tombstones(0, &(0..4_096_u16).collect::<Vec<_>>());
        assert!(
            !bitmap_overlap.is_monotone_superset_of(&array_overlap)?,
            "ARRAY cannot promote before its 4,097th member"
        );
        assert!(
            !bitmap_3_584.is_monotone_superset_of(&TombstoneSet::new())?,
            "a new chunk cannot begin as BITMAP inside the hysteresis overlap"
        );
        assert_eq!(
            TombstoneSet::from_bytes(array_overlap.as_bytes())?.as_bytes(),
            array_overlap.as_bytes()
        );
        assert_eq!(
            TombstoneSet::from_bytes(bitmap_overlap.as_bytes())?.as_bytes(),
            bitmap_overlap.as_bytes()
        );
        Ok(())
    }

    #[test]
    fn stats_must_be_ordered_and_bounded_by_snapshot_docs() {
        let mut manifest = sample_manifest(1);
        manifest.field_stats.swap(0, 1);
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.field_stats[0].doc_count = 6;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn stats_denominator_must_equal_the_canonical_at_seal_count() {
        // Every field row uses the same global at-seal denominator
        // (bd-quill-e3-keeper-ndtk.10); tombstones do not change it. Reject
        // deviations in either direction; exact equality round-trips.
        let mut too_large = sample_manifest(1);
        too_large.field_stats[0].doc_count = 6;
        let error = too_large.validate().expect_err("over-count rejected");
        assert!(
            error
                .to_string()
                .contains("differs from canonical at-seal denominator")
        );

        let mut too_small = sample_manifest(1);
        too_small.field_stats[1].doc_count = 4;
        let error = too_small.validate().expect_err("under-count rejected");
        assert!(
            error
                .to_string()
                .contains("differs from canonical at-seal denominator")
        );

        let canonical = sample_manifest(1);
        canonical.validate().expect("exact denominator passes");
        let bytes = canonical.to_bytes().expect("encode canonical");
        assert_eq!(
            Manifest::from_bytes(&bytes).expect("decode canonical"),
            canonical,
            "canonical denominator round-trips"
        );

        // Tombstones never change the denominator: removing the tombstone
        // set from a segment does not relax the equality rule.
        let mut no_tombstones = canonical;
        no_tombstones.segments[0].tombstones = TombstoneSet::new();
        no_tombstones
            .validate()
            .expect("tombstone-free keeps denominator");
        no_tombstones.field_stats[0].doc_count = 2;
        assert!(no_tombstones.validate().is_err());
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut state = 0x5e31_9a72_d0c4_b611_u64;
        for case in 0..1_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let length = usize::try_from(state % 257).expect("small generated length");
            let mut bytes = Vec::with_capacity(length);
            for _ in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bytes.push(state.to_le_bytes()[0]);
            }
            assert!(
                std::panic::catch_unwind(|| Manifest::from_bytes(&bytes)).is_ok(),
                "parser panic in case {case}"
            );
        }
    }

    #[test]
    fn pair_loader_distinguishes_not_found_and_both_fallback_windows() -> TestResult {
        let directory = tempdir()?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::IndexNotFound { .. })
        ));

        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterMissingCurrent);

        std::fs::write(directory.path().join("MANIFEST"), b"corrupt")?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn pair_loader_never_follows_manifest_symlinks_or_special_entries() -> TestResult {
        use std::os::unix::fs::symlink;

        let directory = tempdir()?;
        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let symlink_target = directory.path().join("manifest-target");
        write_manifest(&symlink_target, &sample_manifest(5))?;
        symlink(&symlink_target, directory.path().join("MANIFEST"))?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

        let directory = tempdir()?;
        let previous = sample_manifest(7);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        std::fs::create_dir(directory.path().join("MANIFEST"))?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

        let directory = tempdir()?;
        let current = sample_manifest(9);
        write_manifest(&directory.path().join("MANIFEST"), &current)?;
        let previous_target = directory.path().join("previous-target");
        write_manifest(&previous_target, &sample_manifest(8))?;
        symlink(&previous_target, directory.path().join("MANIFEST.prev"))?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, current);
        assert_eq!(loaded.source, ManifestSource::Current);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pair_loader_treats_fifo_manifest_as_invalid_without_blocking() -> TestResult {
        use rustix::fs::{Mode, mkfifoat};

        let directory = tempdir()?;
        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let directory_file = File::open(directory.path())?;
        mkfifoat(&directory_file, "MANIFEST", Mode::RUSR | Mode::WUSR)?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
        Ok(())
    }

    #[test]
    fn pair_loader_rejects_nonadjacent_valid_generations() -> TestResult {
        let directory = tempdir()?;
        write_manifest(&directory.path().join("MANIFEST"), &sample_manifest(9))?;
        write_manifest(&directory.path().join("MANIFEST.prev"), &sample_manifest(3))?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidGenerationPair {
                current: 9,
                previous: 3,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn pair_loader_rejects_stable_identity_and_watermark_disagreement() -> TestResult {
        let directory = tempdir()?;
        let previous = sample_manifest(8);

        let mut equal_but_different = previous.clone();
        equal_but_different.flags = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        write_manifest(&directory.path().join("MANIFEST"), &equal_but_different)?;
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 8,
                previous: 8,
                ..
            })
        ));

        let mut schema_mismatch = sample_manifest(9);
        schema_mismatch.schema_id ^= 1;
        write_manifest(&directory.path().join("MANIFEST"), &schema_mismatch)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut watermark_rollback = sample_manifest(9);
        watermark_rollback.docid_high_watermark = previous.docid_high_watermark - 1;
        write_manifest(&directory.path().join("MANIFEST"), &watermark_rollback)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut resurrected = sample_manifest(9);
        resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
        write_manifest(&directory.path().join("MANIFEST"), &resurrected)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut changed_stats = sample_manifest(9);
        changed_stats.field_stats[0].total_tokens += 1;
        write_manifest(&directory.path().join("MANIFEST"), &changed_stats)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_current_wins_when_previous_cannot_be_read_as_a_file() -> TestResult {
        let directory = tempdir()?;
        let current = sample_manifest(1);
        write_manifest(&directory.path().join("MANIFEST"), &current)?;
        std::fs::create_dir(directory.path().join("MANIFEST.prev"))?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, current);
        assert_eq!(loaded.source, ManifestSource::Current);
        Ok(())
    }

    #[test]
    fn publish_reopens_monotone_current_and_previous_generations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let mut second = sample_manifest(2);
            second.docid_high_watermark = first.docid_high_watermark + 100;
            publisher
                .publish(&cx, &second)
                .await
                .expect("publish second");

            let loaded = load_manifest_pair(directory.path()).expect("reopen pair");
            assert_eq!(loaded.manifest, second);
            assert_eq!(loaded.source, ManifestSource::Current);
            let previous_bytes =
                std::fs::read(directory.path().join("MANIFEST.prev")).expect("read previous");
            assert_eq!(
                Manifest::from_bytes(&previous_bytes).expect("parse previous"),
                first
            );
        });
    }

    #[test]
    fn publish_rejects_generation_gap_and_watermark_rollback() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(3)).await,
                Err(KeeperError::GenerationConflict {
                    expected: 2,
                    proposed: 3
                })
            ));
            let mut rollback = sample_manifest(2);
            rollback.docid_high_watermark = first.docid_high_watermark - 1;
            assert!(matches!(
                publisher.publish(&cx, &rollback).await,
                Err(KeeperError::InvalidTransition { .. })
            ));
        });
    }

    #[test]
    fn publish_preserves_retained_segment_identity_and_monotone_tombstones() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let mut mutated = sample_manifest(2);
            mutated.segments[0].file_xxh3 ^= 1;
            assert!(matches!(
                publisher.publish(&cx, &mutated).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut resurrected = sample_manifest(2);
            resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
            assert!(matches!(
                publisher.publish(&cx, &resurrected).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut stale_seal = sample_manifest(2);
            stale_seal.segments.push(ManifestSegment {
                segment_id: 0x1003,
                seal_seq: 6,
                file_len: 512,
                file_xxh3: 0xc1c2_c3c4_c5c6_c7c8,
                docid_lo: 70_000,
                docid_hi: 70_001,
                doc_count: 1,
                tombstones: TombstoneSet::new(),
            });
            // Keep the canonical STATS denominator consistent with the new
            // aggregate so the stale seal_seq is the transition being tested.
            for stats in &mut stale_seal.field_stats {
                stats.doc_count = 6;
            }
            assert!(matches!(
                publisher.publish(&cx, &stale_seal).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut changed_stats = sample_manifest(2);
            changed_stats.field_stats[0].total_tokens += 1;
            assert!(matches!(
                publisher.publish(&cx, &changed_stats).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut monotone = sample_manifest(2);
            monotone.segments[0].tombstones = array_tombstones(0, &[101, 150, 175]);
            publisher
                .publish(&cx, &monotone)
                .await
                .expect("monotone tombstone growth publishes");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen monotone publish")
                    .manifest,
                monotone
            );
        });
    }

    #[test]
    fn empty_manifest_starts_a_new_seal_sequence_epoch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let mut empty = Manifest::empty(
                2,
                first.schema_id,
                first.docid_high_watermark.saturating_add(1),
            );
            empty.last_publish_unix_s = 1_700_000_000;
            publisher
                .publish(&cx, &empty)
                .await
                .expect("publish explicit empty generation");

            let mut restarted = Manifest::empty(3, first.schema_id, empty.docid_high_watermark);
            restarted.last_publish_unix_s = 1_700_000_000;
            let mut segment = first.segments[0].clone();
            segment.segment_id = 0x2001;
            segment.seal_seq = 1;
            restarted.segments.push(segment);
            restarted.field_stats = vec![
                ManifestFieldStats {
                    field_ord: 1,
                    total_tokens: 12,
                    doc_count: 3,
                },
                ManifestFieldStats {
                    field_ord: 3,
                    total_tokens: 42,
                    doc_count: 3,
                },
            ];
            publisher
                .publish(&cx, &restarted)
                .await
                .expect("restart seal sequence after empty epoch boundary");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen restarted seal epoch")
                    .manifest,
                restarted
            );
        });
    }

    #[test]
    fn invalid_or_cancelled_proposal_creates_no_manifest_slot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            assert!(matches!(
                publisher.publish(&cx, &Manifest::empty(0, 7, 0)).await,
                Err(KeeperError::InvalidManifest { .. })
            ));

            cx.set_cancel_requested(true);
            assert!(matches!(
                publisher.publish(&cx, &Manifest::empty(1, 7, 0)).await,
                Err(KeeperError::PublishLock {
                    source: LockError::Cancelled
                })
            ));
            assert!(!directory.path().join("MANIFEST").exists());
            assert!(!directory.path().join("MANIFEST.prev").exists());
        });
    }

    #[test]
    fn generation_claim_runs_after_temp_fsync_and_before_renames() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let saw_expected_window = Arc::new(AtomicBool::new(false));
            let witness = Arc::clone(&saw_expected_window);
            publisher
                .publish_with_generation_claim(&cx, &sample_manifest(2), move |path, generation| {
                    let temp = path.join(format!(".tmp-manifest-{generation}"));
                    let current = path.join("MANIFEST");
                    let previous = path.join("MANIFEST.prev");
                    witness.store(
                        temp.is_file() && current.is_file() && !previous.exists(),
                        Ordering::SeqCst,
                    );
                    Ok(())
                })
                .await
                .expect("claimed publish");
            assert!(saw_expected_window.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn ordinary_segment_publish_performs_zero_sidecar_io() -> TestResult {
        let directory = tempdir()?;
        let encoded = encoded_test_segment(0x50, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let published = directory.path().join(canonical_segment_name(0x50));
        let mut sidecar_name = published.as_os_str().to_os_string();
        sidecar_name.push(".fec");
        let sidecar = PathBuf::from(sidecar_name);
        let sentinel = b"opaque feature-off segment sidecar";
        std::fs::write(&sidecar, sentinel)?;

        assert_eq!(publish_pending_segment(pending)?, published);
        assert_eq!(std::fs::read(&published)?, encoded.as_bytes());
        assert_eq!(std::fs::read(sidecar)?, sentinel);
        Ok(())
    }

    #[test]
    fn ordinary_manifest_publish_performs_zero_sidecar_io() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let sidecar = directory.path().join("MANIFEST.fec");
            std::fs::write(&sidecar, b"opaque feature-off artifact").expect("write sentinel");

            publisher
                .publish(&cx, &sample_manifest(2))
                .await
                .expect("publish without durability");

            assert_eq!(
                std::fs::read(&sidecar).expect("sentinel remains readable"),
                b"opaque feature-off artifact"
            );
            assert!(!directory.path().join("MANIFEST.prev.fec").exists());
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_segment_publish_refuses_an_existing_sidecar() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x51, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let pending_path = pending.path().to_path_buf();
        let published = directory.path().join(canonical_segment_name(0x51));
        let sidecar = FileProtector::sidecar_path(&published);
        let sentinel = b"preserved orphan sidecar";
        std::fs::write(&sidecar, sentinel)?;

        assert!(matches!(
            publish_pending_segment_durable(pending, &protector),
            Err(KeeperError::Io {
                operation: "publish segment sidecar",
                ..
            })
        ));
        assert_eq!(std::fs::read(&pending_path)?, encoded.as_bytes());
        assert!(!published.exists());
        assert_eq!(std::fs::read(sidecar)?, sentinel);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_segment_publish_rejects_same_length_temp_mutation() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x52, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let pending_path = pending.path().to_path_buf();
        let published = directory.path().join(canonical_segment_name(0x52));
        let sidecar = FileProtector::sidecar_path(&published);

        let mut mutated = std::fs::read(&pending_path)?;
        let mutation_offset = mutated.len() / 2;
        mutated[mutation_offset] ^= 0x80;
        std::fs::write(&pending_path, &mutated)?;
        assert_eq!(std::fs::metadata(&pending_path)?.len(), pending.file_len());

        assert!(matches!(
            publish_pending_segment_durable(pending, &protector),
            Err(KeeperError::Durability {
                operation: "protect published segment",
                ..
            })
        ));
        assert!(!pending_path.exists());
        assert_eq!(std::fs::read(&published)?, mutated);
        assert!(!sidecar.exists());
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn sidecar_retirement_uses_a_collision_safe_suffix() -> TestResult {
        let directory = tempdir()?;
        let source = directory.path().join("MANIFEST.fec");
        let retired = directory.path().join(".tmp-manifest-current-fec-2");
        let sentinel = b"operator-preserved crash artifact";
        let sidecar = b"active sidecar";
        std::fs::write(&source, sidecar)?;
        std::fs::write(&retired, sentinel)?;

        assert!(retire_manifest_sidecar(
            directory.path(),
            &source,
            &retired
        )?);
        assert!(!source.exists());
        assert_eq!(std::fs::read(&retired)?, sentinel);
        assert_eq!(std::fs::read(append_path_suffix(&retired, ".1"))?, sidecar);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_publish_retries_generation_after_retirement_collision() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let protector = test_file_protector();
            let mut writer = KeeperWriter::create_durable(
                &cx,
                directory.path(),
                DEFAULT_SCHEMA,
                protector.clone(),
            )
            .await
            .expect("create durable writer");
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .expect("publish durable generation two");
            drop(writer);

            let previous = directory.path().join("MANIFEST.prev");
            let previous_sidecar = FileProtector::sidecar_path(&previous);
            let retired = directory.path().join(".tmp-manifest-previous-fec-3");
            let retired_bytes = std::fs::read(&previous_sidecar).expect("read prior sidecar");
            std::fs::rename(&previous_sidecar, &retired)
                .expect("simulate interrupted generation-three retirement");

            let mut writer = KeeperWriter::open_durable(
                &cx,
                directory.path(),
                DEFAULT_SCHEMA,
                protector.clone(),
            )
            .await
            .expect("recover active previous sidecar");
            assert_eq!(
                std::fs::read(&retired).expect("preserve crash artifact"),
                retired_bytes
            );
            assert!(
                protector
                    .verify_file(&previous, &previous_sidecar)
                    .expect("verify regenerated previous sidecar")
                    .healthy
            );

            writer
                .publish(&cx, &durable_test_manifest(3, Vec::new()))
                .await
                .expect("retry durable generation three");
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 3);
            assert_eq!(
                std::fs::read(&retired).expect("retain first retirement artifact"),
                retired_bytes
            );
            assert!(append_path_suffix(&retired, ".1").is_file());
            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                assert!(
                    protector
                        .verify_file(&path, &FileProtector::sidecar_path(&path))
                        .expect("verify retried sidecar")
                        .healthy,
                    "{source}"
                );
            }
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_publish_rotates_matching_sidecars() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let protector = test_file_protector();
            publisher
                .publish_durable(&cx, &sample_manifest(1), &protector)
                .await
                .expect("durable genesis");
            publisher
                .publish_durable(&cx, &sample_manifest(2), &protector)
                .await
                .expect("durable successor");

            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                let sidecar = FileProtector::sidecar_path(&path);
                let verified = protector
                    .verify_file(&path, &sidecar)
                    .expect("verify matching sidecar");
                assert!(verified.healthy, "{source}");
            }
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("load durable pair")
                    .manifest
                    .generation,
                2
            );
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn manifest_sidecar_rotation_has_no_wrong_source_pairing() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        for generation in [1, 2] {
            let bytes = sample_manifest(generation).to_bytes()?;
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &bytes,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;
        }

        let third = sample_manifest(3).to_bytes()?;
        let temp_path = directory.path().join(".tmp-manifest-3");
        std::fs::write(&temp_path, &third)?;
        protector.protect_file_with_witness(&temp_path, FileSourceWitness::from_bytes(&third))?;

        prepare_manifest_sidecar_rotation(directory.path(), true, 3, &protector)?;
        let current = directory.path().join("MANIFEST");
        let previous = directory.path().join("MANIFEST.prev");
        assert!(
            protector
                .verify_file(&current, &FileProtector::sidecar_path(&current))?
                .healthy
        );
        assert!(!FileProtector::sidecar_path(&previous).exists());

        std::fs::rename(&current, &previous)?;
        assert!(!current.exists());
        assert!(FileProtector::sidecar_path(&current).exists());
        assert!(!FileProtector::sidecar_path(&previous).exists());
        assert_eq!(load_manifest_pair(directory.path())?.manifest.generation, 2);

        complete_manifest_sidecar_rotation(directory.path(), &current, &previous, 3)?;
        assert!(!FileProtector::sidecar_path(&current).exists());
        assert!(
            protector
                .verify_file(&previous, &FileProtector::sidecar_path(&previous))?
                .healthy
        );

        std::fs::rename(&temp_path, &current)?;
        assert!(!FileProtector::sidecar_path(&current).exists());
        install_manifest_sidecar(directory.path(), &current, 3)?;
        for source in [&current, &previous] {
            assert!(
                protector
                    .verify_file(source, &FileProtector::sidecar_path(source))?
                    .healthy
            );
        }
        assert_eq!(load_manifest_pair(directory.path())?.manifest.generation, 3);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_publish_replaces_a_stale_mixed_mode_sidecar() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let protector = test_file_protector();
            publisher
                .publish_durable(&cx, &sample_manifest(1), &protector)
                .await
                .expect("durable generation one");
            publisher
                .publish(&cx, &sample_manifest(2))
                .await
                .expect("ordinary generation two");

            let current = directory.path().join("MANIFEST");
            let stale_sidecar = FileProtector::sidecar_path(&current);
            let current_bytes = std::fs::read(&current).expect("read generation two");
            assert!(
                !protector
                    .sidecar_matches_witness(
                        &stale_sidecar,
                        FileSourceWitness::from_bytes(&current_bytes),
                    )
                    .expect("compare stale sidecar")
            );

            publisher
                .publish_durable(&cx, &sample_manifest(3), &protector)
                .await
                .expect("durable generation three");
            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                assert!(
                    protector
                        .verify_file(&path, &FileProtector::sidecar_path(&path))
                        .expect("verify regenerated sidecar")
                        .healthy,
                    "{source}"
                );
            }
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("load mixed-mode pair")
                    .manifest
                    .generation,
                3
            );
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_crash_windows_never_pair_stale_sidecars() -> TestResult {
        const CHECKPOINTS: [PublishCheckpoint; 6] = [
            PublishCheckpoint::TempWritten,
            PublishCheckpoint::TempSynced,
            PublishCheckpoint::GenerationClaimed,
            PublishCheckpoint::CurrentMovedToPrevious,
            PublishCheckpoint::TempMovedToCurrent,
            PublishCheckpoint::DirectorySynced,
        ];

        for fault in CHECKPOINTS {
            let directory = tempdir()?;
            let protector = test_file_protector();
            let first = sample_manifest(1).to_bytes()?;
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &first,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;

            let second = sample_manifest(2).to_bytes()?;
            let result = publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &second,
                |_, _| Ok(()),
                &protector,
                |checkpoint, path| {
                    if checkpoint == fault {
                        return Err(KeeperError::Io {
                            operation: "inject durable manifest crash",
                            path: path.to_path_buf(),
                            source: io::Error::other(format!(
                                "injected durable crash at {checkpoint:?}"
                            )),
                        });
                    }
                    Ok(())
                },
            );
            assert!(matches!(
                result,
                Err(KeeperError::Io {
                    operation: "inject durable manifest crash",
                    ..
                })
            ));

            let loaded = load_manifest_pair(directory.path())?;
            let expected_generation = match fault {
                PublishCheckpoint::TempWritten
                | PublishCheckpoint::TempSynced
                | PublishCheckpoint::GenerationClaimed
                | PublishCheckpoint::CurrentMovedToPrevious => 1,
                PublishCheckpoint::TempMovedToCurrent | PublishCheckpoint::DirectorySynced => 2,
            };
            assert_eq!(loaded.manifest.generation, expected_generation, "{fault:?}");

            let current_path = directory.path().join("MANIFEST");
            let current_sidecar = FileProtector::sidecar_path(&current_path);
            if fault == PublishCheckpoint::TempMovedToCurrent {
                assert!(!current_sidecar.exists());
            } else if fault == PublishCheckpoint::DirectorySynced {
                assert!(current_sidecar.exists());
                assert!(
                    protector
                        .verify_file(&current_path, &current_sidecar)?
                        .healthy
                );
            }

            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                let sidecar = FileProtector::sidecar_path(&path);
                if sidecar.exists() {
                    assert!(
                        path.exists(),
                        "orphan active sidecar at {fault:?}: {source}"
                    );
                    assert!(
                        protector.verify_file(&path, &sidecar)?.healthy,
                        "stale sidecar at {fault:?}: {source}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn real_durable_segment_checkpoints_recover_and_gc_exactly() -> TestResult {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Fault {
            TempWritten,
            TempSynced,
            SegmentRenamed,
            DirectorySynced,
            SidecarEmitted,
        }

        const FAULTS: [Fault; 5] = [
            Fault::TempWritten,
            Fault::TempSynced,
            Fault::SegmentRenamed,
            Fault::DirectorySynced,
            Fault::SidecarEmitted,
        ];

        for fault in FAULTS {
            let directory = tempdir()?;
            let protector = test_file_protector();

            let base_encoded = encoded_test_segment(0x70, 0, 2, 1)?;
            let base_manifest_segment = manifest_segment(&base_encoded, 1);
            let base_pending = base_encoded.write_temp(directory.path())?;
            let base_path = publish_pending_segment_durable(base_pending, &protector)?;
            assert!(
                protector
                    .verify_file(&base_path, &FileProtector::sidecar_path(&base_path))?
                    .healthy
            );
            let base_manifest = durable_test_manifest(1, vec![base_manifest_segment]);
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &base_manifest.to_bytes()?,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;

            let next_encoded = encoded_test_segment(0x71, 2, 4, 1)?;
            let write_result =
                next_encoded.write_temp_with_observer(directory.path(), |checkpoint, path| {
                    let injected = matches!(
                        (fault, checkpoint),
                        (Fault::TempWritten, SegmentWriteCheckpoint::TempWritten)
                            | (Fault::TempSynced, SegmentWriteCheckpoint::TempSynced)
                    );
                    if injected {
                        return Err(QuillError::Io(io::Error::other(format!(
                            "injected segment writer crash at {checkpoint:?}: {}",
                            path.display()
                        ))));
                    }
                    Ok(())
                });

            if matches!(fault, Fault::TempWritten | Fault::TempSynced) {
                assert!(write_result.is_err(), "{fault:?}");
            } else {
                let pending = write_result?;
                assert_ne!(pending.file_xxh3(), pending.source_xxh3());
                let publish_result = publish_pending_segment_durable_with_observer(
                    pending,
                    &protector,
                    |checkpoint, path| {
                        let injected = matches!(
                            (fault, checkpoint),
                            (
                                Fault::SegmentRenamed,
                                SegmentPublishCheckpoint::SegmentRenamed
                            ) | (
                                Fault::DirectorySynced,
                                SegmentPublishCheckpoint::DirectorySynced
                            ) | (
                                Fault::SidecarEmitted,
                                SegmentPublishCheckpoint::SidecarEmitted
                            )
                        );
                        if injected {
                            return Err(KeeperError::Io {
                                operation: "inject segment publish crash",
                                path: path.to_path_buf(),
                                source: io::Error::other(format!(
                                    "injected segment publication crash at {checkpoint:?}"
                                )),
                            });
                        }
                        Ok(())
                    },
                );
                assert!(publish_result.is_err(), "{fault:?}");
            }

            let before_open = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                first.loaded_manifest(),
                second.loaded_manifest(),
                "{fault:?}"
            );
            assert_eq!(first.loaded_manifest().manifest, base_manifest, "{fault:?}");
            assert_eq!(directory_bytes(directory.path())?, before_open, "{fault:?}");

            let segment_name = canonical_segment_name(0x71);
            let segment_path = directory.path().join(&segment_name);
            if fault == Fault::SidecarEmitted {
                assert!(
                    protector
                        .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))?
                        .healthy
                );
            }

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let mut expected_removed = match fault {
                Fault::TempWritten | Fault::TempSynced => {
                    vec![PathBuf::from(".tmp-segment-0000000000000071")]
                }
                Fault::SegmentRenamed | Fault::DirectorySynced => {
                    vec![PathBuf::from(&segment_name)]
                }
                Fault::SidecarEmitted => vec![
                    PathBuf::from(&segment_name),
                    PathBuf::from(format!("{segment_name}.fec")),
                ],
            };
            expected_removed.sort_unstable();
            assert_eq!(report.removed, expected_removed, "{fault:?}");

            let mut expected_inventory = before_open;
            expected_inventory.retain(|(name, _)| {
                !report
                    .removed
                    .iter()
                    .any(|removed| removed.as_os_str() == name)
            });
            assert_eq!(
                directory_bytes(directory.path())?,
                expected_inventory,
                "{fault:?}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{fault:?}"
            );
            assert_eq!(
                collect_writer_garbage_at(
                    directory.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                )?,
                GarbageCollectionReport {
                    removed: Vec::new()
                },
                "{fault:?}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn real_fslx_and_manifest_repair_restore_exact_bytes() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x71, 0, 2, 1)?;
        let segment = manifest_segment(&encoded, 1);
        let pending = encoded.write_temp(directory.path())?;
        assert_eq!(
            pending.source_xxh3(),
            xxhash_rust::xxh3::xxh3_64(encoded.as_bytes())
        );
        let segment_path = publish_pending_segment_durable(pending, &protector)?;
        let manifest = durable_test_manifest(1, vec![segment]);
        let manifest_path = directory.path().join("MANIFEST");
        let manifest_bytes = manifest.to_bytes()?;
        publish_manifest_durable_choreography(
            directory.path().to_path_buf(),
            &manifest_bytes,
            |_, _| Ok(()),
            &protector,
            |_, _| Ok(()),
        )?;

        let segment_bytes = std::fs::read(&segment_path)?;
        assert_eq!(std::fs::read(&manifest_path)?, manifest_bytes);

        let mut corrupted_segment = segment_bytes.clone();
        let corruption_offset = corrupted_segment.len() / 2;
        corrupted_segment[corruption_offset] ^= 0x80;
        std::fs::write(&segment_path, &corrupted_segment)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&segment_path),
            FileSourceWitness::from_bytes(&corrupted_segment),
        )?);
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);

        File::options()
            .write(true)
            .open(&segment_path)?
            .set_len(u64::try_from(segment_bytes.len() / 3)?)?;
        let truncated = std::fs::read(&segment_path)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&segment_path),
            FileSourceWitness::from_bytes(&truncated),
        )?);
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);

        let displaced_segment = directory.path().join(".tmp-missing-segment-fixture");
        std::fs::rename(&segment_path, &displaced_segment)?;
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);
        assert_eq!(std::fs::read(displaced_segment)?, segment_bytes);

        let mut corrupted_manifest = manifest_bytes.clone();
        corrupted_manifest[16] ^= 0x40;
        std::fs::write(&manifest_path, &corrupted_manifest)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&manifest_path),
            FileSourceWitness::from_bytes(&corrupted_manifest),
        )?);
        let manifest_health = protector.verify_and_repair_file(&manifest_path)?;
        assert!(matches!(
            manifest_health.status,
            FileHealth::Repaired { .. }
        ));
        assert_eq!(std::fs::read(&manifest_path)?, manifest_bytes);

        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(reopened.segments().len(), 1);
        reopened.segments()[0].verify()?;
        Ok(())
    }

    #[test]
    fn real_manifest_choreography_recovers_at_every_checkpoint() -> TestResult {
        const CHECKPOINTS: [PublishCheckpoint; 6] = [
            PublishCheckpoint::TempWritten,
            PublishCheckpoint::TempSynced,
            PublishCheckpoint::GenerationClaimed,
            PublishCheckpoint::CurrentMovedToPrevious,
            PublishCheckpoint::TempMovedToCurrent,
            PublishCheckpoint::DirectorySynced,
        ];

        let success_directory = tempdir()?;
        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        write_manifest(
            &success_directory.path().join("MANIFEST"),
            &Manifest::empty(1, schema_id, 0),
        )?;
        let proposed = Manifest::empty(2, schema_id, 0);
        let proposed_bytes = proposed.to_bytes()?;
        let mut observed = Vec::new();
        publish_manifest_choreography(
            success_directory.path().to_path_buf(),
            &proposed_bytes,
            |_, _| Ok(()),
            |checkpoint, _| {
                observed.push(checkpoint);
                Ok(())
            },
        )?;
        assert_eq!(observed, CHECKPOINTS);

        for fault in CHECKPOINTS {
            let directory = tempdir()?;
            let initial = Manifest::empty(1, schema_id, 0);
            write_manifest(&directory.path().join("MANIFEST"), &initial)?;
            let result = publish_manifest_choreography(
                directory.path().to_path_buf(),
                &proposed_bytes,
                |_, _| Ok(()),
                |checkpoint, path| {
                    if checkpoint == fault {
                        return Err(KeeperError::Io {
                            operation: "inject manifest publish crash",
                            path: path.to_path_buf(),
                            source: io::Error::other(format!("injected crash at {checkpoint:?}")),
                        });
                    }
                    Ok(())
                },
            );
            assert!(
                matches!(
                    result,
                    Err(KeeperError::Io {
                        operation: "inject manifest publish crash",
                        ..
                    })
                ),
                "{fault:?}"
            );

            let before_open = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(directory_bytes(directory.path())?, before_open, "{fault:?}");
            assert_eq!(
                first.loaded_manifest(),
                second.loaded_manifest(),
                "{fault:?}"
            );
            let expected_generation = match fault {
                PublishCheckpoint::TempWritten
                | PublishCheckpoint::TempSynced
                | PublishCheckpoint::GenerationClaimed
                | PublishCheckpoint::CurrentMovedToPrevious => 1,
                PublishCheckpoint::TempMovedToCurrent | PublishCheckpoint::DirectorySynced => 2,
            };
            assert_eq!(
                first.loaded_manifest().manifest.generation,
                expected_generation,
                "{fault:?}"
            );
            let expected_source = if fault == PublishCheckpoint::CurrentMovedToPrevious {
                ManifestSource::PreviousAfterMissingCurrent
            } else {
                ManifestSource::Current
            };
            assert_eq!(first.loaded_manifest().source, expected_source, "{fault:?}");

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let temp_survived_fault = matches!(
                fault,
                PublishCheckpoint::TempWritten
                    | PublishCheckpoint::TempSynced
                    | PublishCheckpoint::GenerationClaimed
                    | PublishCheckpoint::CurrentMovedToPrevious
            );
            assert_eq!(
                report.removed,
                if temp_survived_fault {
                    vec![PathBuf::from(".tmp-manifest-2")]
                } else {
                    Vec::new()
                },
                "{fault:?}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{fault:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn retry_reuses_byte_identical_temp_after_claim_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let proposed = sample_manifest(2);
            let expected_bytes = proposed.to_bytes().expect("encode proposal");

            let first_result = publisher
                .publish_with_generation_claim::<(), _>(&cx, &proposed, |path, generation| {
                    Err(KeeperError::Io {
                        operation: "test generation claim",
                        path: path.join(format!("gen-{generation}.claim")),
                        source: io::Error::other("injected claim failure"),
                    })
                })
                .await;
            assert!(matches!(
                first_result,
                Err(KeeperError::Io {
                    operation: "test generation claim",
                    ..
                })
            ));
            let temp_path = directory.path().join(".tmp-manifest-2");
            assert_eq!(
                std::fs::read(&temp_path).expect("read durable temp"),
                expected_bytes
            );

            publisher
                .publish(&cx, &proposed)
                .await
                .expect("retry byte-identical durable temp");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen retried publish")
                    .manifest,
                proposed
            );
        });
    }

    #[test]
    fn mismatched_existing_temp_fails_closed_without_overwrite() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let temp_path = directory.path().join(".tmp-manifest-1");
            let stale_bytes = b"stale temp from another proposal";
            std::fs::write(&temp_path, stale_bytes).expect("write stale temp fixture");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(1)).await,
                Err(KeeperError::TempConflict { .. })
            ));
            assert_eq!(
                std::fs::read(&temp_path).expect("read preserved stale temp"),
                stale_bytes
            );
            assert!(!directory.path().join("MANIFEST").exists());
        });
    }

    #[cfg(unix)]
    #[test]
    fn existing_temp_symlink_fails_closed_without_publication() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let expected_bytes = sample_manifest(1).to_bytes().expect("encode proposal");
            let symlink_target = directory.path().join("byte-identical-target");
            std::fs::write(&symlink_target, expected_bytes).expect("write symlink target");
            let temp_path = directory.path().join(".tmp-manifest-1");
            std::os::unix::fs::symlink(&symlink_target, &temp_path)
                .expect("create temp symlink fixture");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(1)).await,
                Err(KeeperError::Io {
                    operation: "open existing temp",
                    ..
                })
            ));
            assert!(
                std::fs::symlink_metadata(&temp_path)
                    .expect("stat preserved temp symlink")
                    .file_type()
                    .is_symlink()
            );
            assert!(!directory.path().join("MANIFEST").exists());
        });
    }

    #[cfg(unix)]
    #[test]
    fn labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias() {
        let root = tempdir().expect("temp directory");
        let directory = root.path().join("target-index");
        let alias = root.path().join("index-alias");
        std::fs::create_dir(&directory).expect("create target index directory");
        let first_publisher = ManifestPublisher::new(&alias);
        let second_publisher = ManifestPublisher::new(&directory);
        std::os::unix::fs::symlink(&directory, &alias).expect("create alias after first publisher");
        let lock = first_publisher.publish_lock_for_test();
        assert!(Arc::ptr_eq(
            &lock,
            &second_publisher.publish_lock_for_test()
        ));
        let saw_two_waiters = Arc::new(AtomicBool::new(false));
        let successes = Arc::new(AtomicUsize::new(0));
        let conflicts = Arc::new(AtomicUsize::new(0));

        let mut lab = LabRuntime::new(LabConfig::new(0x51a1).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);
        let holder_lock = Arc::clone(&lock);
        let holder_witness = Arc::clone(&saw_two_waiters);
        let (holder, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                let guard = OwnedMutexGuard::lock(Arc::clone(&holder_lock), &cx)
                    .await
                    .expect("holder lock");
                while holder_lock.waiters() < 2 {
                    yield_now().await;
                }
                holder_witness.store(true, Ordering::SeqCst);
                drop(guard);
            })
            .expect("create holder task");

        let first_successes = Arc::clone(&successes);
        let first_conflicts = Arc::clone(&conflicts);
        let (first_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                match first_publisher.publish(&cx, &sample_manifest(1)).await {
                    Ok(_) => {
                        first_successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(KeeperError::GenerationConflict {
                        expected: 2,
                        proposed: 1,
                    }) => {
                        first_conflicts.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(error) => panic!("unexpected first publish result: {error}"),
                }
            })
            .expect("create first publisher");

        let second_successes = Arc::clone(&successes);
        let second_conflicts = Arc::clone(&conflicts);
        let (second_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                match second_publisher.publish(&cx, &sample_manifest(1)).await {
                    Ok(_) => {
                        second_successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(KeeperError::GenerationConflict {
                        expected: 2,
                        proposed: 1,
                    }) => {
                        second_conflicts.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(error) => panic!("unexpected second publish result: {error}"),
                }
            })
            .expect("create second publisher");

        lab.scheduler.lock().schedule(holder, 0);
        lab.step_for_test();
        lab.scheduler.lock().schedule(first_task, 0);
        lab.scheduler.lock().schedule(second_task, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(saw_two_waiters.load(Ordering::SeqCst));
        assert_eq!(successes.load(Ordering::SeqCst), 1);
        assert_eq!(conflicts.load(Ordering::SeqCst), 1);
        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
        assert_eq!(
            load_manifest_pair(&directory)
                .expect("reopen after concurrent publishes")
                .manifest
                .generation,
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn labruntime_writer_admission_blocks_a_second_writer_but_not_a_reader() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let directory = index.path().to_path_buf();
        let holder_ready = Arc::new(AtomicBool::new(false));
        let contender_was_busy = Arc::new(AtomicBool::new(false));
        let reader_was_touchless = Arc::new(AtomicBool::new(false));
        let observers_done = Arc::new(AtomicUsize::new(0));

        let mut lab = LabRuntime::new(LabConfig::new(0x10c6).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let holder_directory = directory.clone();
        let holder_ready_task = Arc::clone(&holder_ready);
        let holder_done = Arc::clone(&observers_done);
        let (holder, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                let writer = KeeperWriter::open(&cx, holder_directory, DEFAULT_SCHEMA)
                    .await
                    .expect("first writer admission");
                holder_ready_task.store(true, Ordering::SeqCst);
                while holder_done.load(Ordering::SeqCst) < 2 {
                    yield_now().await;
                }
                drop(writer);
            })
            .expect("create holder task");

        let contender_directory = directory.clone();
        let contender_ready = Arc::clone(&holder_ready);
        let contender_result = Arc::clone(&contender_was_busy);
        let contender_done = Arc::clone(&observers_done);
        let (contender, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !contender_ready.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                let cx = Cx::for_testing();
                match KeeperWriter::open(&cx, contender_directory, DEFAULT_SCHEMA).await {
                    Err(KeeperError::WriterBusy { .. }) => {
                        contender_result.store(true, Ordering::SeqCst);
                    }
                    Ok(writer) => {
                        drop(writer);
                        panic!("second writer was admitted while the first held LOCK");
                    }
                    Err(error) => panic!("unexpected second-writer result: {error}"),
                }
                contender_done.fetch_add(1, Ordering::SeqCst);
            })
            .expect("create contender task");

        let reader_directory = directory.clone();
        let reader_ready = Arc::clone(&holder_ready);
        let reader_result = Arc::clone(&reader_was_touchless);
        let reader_done = Arc::clone(&observers_done);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !reader_ready.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                let before = directory_bytes(&reader_directory).expect("reader inventory before");
                let snapshot = KeeperSnapshot::open(&reader_directory, DEFAULT_SCHEMA)
                    .expect("reader opens while writer is live");
                assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
                let after = directory_bytes(&reader_directory).expect("reader inventory after");
                assert_eq!(after, before, "reader open must not touch the directory");
                reader_result.store(true, Ordering::SeqCst);
                reader_done.fetch_add(1, Ordering::SeqCst);
            })
            .expect("create reader task");

        lab.scheduler.lock().schedule(holder, 0);
        lab.step_for_test();
        assert!(holder_ready.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(contender, 0);
        lab.scheduler.lock().schedule(reader, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(contender_was_busy.load(Ordering::SeqCst));
        assert!(reader_was_touchless.load(Ordering::SeqCst));
        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
        assert_eq!(std::fs::metadata(directory.join("LOCK"))?.len(), 0);
        Ok(())
    }

    #[test]
    fn keeper_create_is_create_or_open_and_in_memory_creates_no_files() {
        let memory = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("in-memory genesis");
        assert_eq!(memory.directory(), None);
        assert_eq!(memory.loaded_manifest().source, ManifestSource::InMemory);
        assert_eq!(memory.loaded_manifest().manifest.generation, 1);
        assert!(memory.segments().is_empty());

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempdir().expect("temp directory");
            let directory = root.path().join("quill-index");
            let created = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .expect("create genesis");
            assert_eq!(created.directory(), Some(directory.as_path()));
            assert_eq!(created.loaded_manifest().manifest.generation, 1);
            let manifest_bytes = std::fs::read(directory.join("MANIFEST")).expect("manifest");

            let reopened = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .expect("create-or-open existing index");
            assert_eq!(reopened.loaded_manifest().manifest.generation, 1);
            assert_eq!(
                std::fs::read(directory.join("MANIFEST")).expect("manifest after reopen"),
                manifest_bytes
            );
            assert!(matches!(
                KeeperSnapshot::create(&cx, &directory, FSFS_CHUNK_SCHEMA).await,
                Err(KeeperError::SchemaMismatch { .. })
            ));
        });
    }

    #[test]
    fn owned_segment_publication_is_snapshot_isolated_and_uniform() -> TestResult {
        let original = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
        let first = encoded_identity_test_segment(0xb01, 0, &[Some("owned-a")])?;
        let second = encoded_identity_test_segment(0xb02, 65_536, &[None, Some("owned-b")])?;
        let mut proposed = original.next_manifest()?;
        proposed.docid_high_watermark = 131_072;
        proposed.segments = vec![manifest_segment(&first, 10), manifest_segment(&second, 20)];

        assert!(original.segments().is_empty());
        assert_eq!(original.doc_count(), 0);
        assert_eq!(original.materialize_document_id(0), None);

        let published = original.publish_owned_segments(&proposed, vec![first, second])?;
        assert!(
            original.segments().is_empty(),
            "old snapshot stays unchanged"
        );
        assert_eq!(original.doc_count(), 0);
        assert_eq!(published.segments().len(), 2);
        assert_eq!(published.at_seal_doc_count(), 2);
        assert_eq!(published.doc_count(), 2);
        assert_eq!(
            published.materialize_document_id(0),
            Some(DocId::new("owned-a"))
        );
        assert_eq!(published.materialize_document_id(65_536), None);
        assert_eq!(
            published.materialize_document_id(65_537),
            Some(DocId::new("owned-b"))
        );
        assert_eq!(published.segments()[0].header().segment_id, 0xb01);
        assert_eq!(
            published.segments()[1].section(SectionKind::TERMDICT)?,
            Some(b"termdict".as_slice())
        );
        published.segments()[0].verify()?;
        published.segments()[1].verify()?;

        let mut tombstoned = published.next_manifest()?;
        assert!(tombstoned.segments[0].insert_tombstone(0)?);
        let deleted = published.publish_owned_segments(&tombstoned, Vec::new())?;
        assert_eq!(deleted.doc_count(), 1);
        assert_eq!(deleted.materialize_document_id(0), None);
        assert_eq!(
            published.materialize_document_id(0),
            Some(DocId::new("owned-a")),
            "retained owned backing does not change the older snapshot"
        );
        Ok(())
    }

    #[test]
    fn recovered_snapshot_validates_witnesses_and_keeps_section_hashes_lazy() -> TestResult {
        let directory = tempdir()?;
        let encoded = encoded_test_segment(0xabc, 10, 20, 1)?;
        let postings = encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::POSTINGS)
            .expect("postings entry");
        let postings_offset = usize::try_from(postings.offset)?;
        let mut bytes = encoded.as_bytes().to_vec();
        bytes[postings_offset] ^= 0x80;
        std::fs::write(directory.path().join(canonical_segment_name(0xabc)), bytes)?;
        let segment = ManifestSegment {
            segment_id: 0xabc,
            seal_seq: 1,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: 10,
            docid_hi: 20,
            doc_count: 1,
            tombstones: TombstoneSet::new(),
        };
        let manifest = durable_test_manifest(1, vec![segment]);
        write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
        let before = directory_bytes(directory.path())?;

        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(directory.path())?, before);
        assert_eq!(snapshot.segments().len(), 1);
        assert_eq!(
            snapshot.segments()[0].section(SectionKind::TERMDICT)?,
            Some(b"termdict".as_slice())
        );
        assert!(matches!(
            snapshot.segments()[0].section(SectionKind::POSTINGS),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut mismatched = manifest;
        mismatched.segments[0].file_len += 1;
        write_manifest(&directory.path().join("MANIFEST"), &mismatched)?;
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
            Err(KeeperError::SegmentMetadataMismatch { .. })
        ));
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), FSFS_CHUNK_SCHEMA),
            Err(KeeperError::SchemaMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn staged_delete_filters_before_commit_and_persists_after_publish() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let low = write_identity_test_segment(directory.path(), 0x201, 10, 0, &[Some("same")])
                .expect("write low segment");
            let mut high =
                write_identity_test_segment(directory.path(), 0x202, 30, 100, &[Some("same")])
                    .expect("write high segment");
            assert!(
                high.tombstones
                    .insert(100)
                    .expect("tombstone newer representative")
            );
            let mut manifest = durable_test_manifest(1, vec![low, high]);
            manifest.field_stats.push(ManifestFieldStats {
                field_ord: 0,
                total_tokens: 17,
                doc_count: 2,
            });
            write_manifest(&directory.path().join("MANIFEST"), &manifest)
                .expect("write genesis manifest");

            let original = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("open original snapshot");
            assert_eq!(original.at_seal_doc_count(), 2);
            assert_eq!(original.tombstone_count(), 1);
            assert_eq!(original.doc_count(), 1);
            assert!(original.is_live(0));
            assert!(!original.is_live(100));
            assert_eq!(
                original.materialize_document_id(0),
                Some(DocId::new("same"))
            );
            assert_eq!(original.materialize_document_id(100), None);
            assert_eq!(
                original
                    .resolve_document_id("same")
                    .expect("resolve through tombstoned newest segment"),
                Some(ResolvedDocumentId {
                    segment_id: 0x201,
                    seal_seq: 10,
                    global_docid: 0,
                    content_hash: 1,
                })
            );

            let mut proposed = original.next_manifest().expect("next generation");
            assert!(
                original
                    .delete_document(&mut proposed, "same")
                    .expect("stage delete")
            );
            assert!(original.is_live(0), "installed snapshot remains unchanged");
            assert!(!crate::argus::LiveDocs::is_live(&proposed, 0));
            assert_eq!(proposed.field_stats, manifest.field_stats);
            let idempotent = proposed.clone();
            assert!(
                !original
                    .delete_document(&mut proposed, "same")
                    .expect("already-deleted id is a no-op")
            );
            assert_eq!(proposed, idempotent);

            let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
                .await
                .expect("open writer");
            let published = writer
                .publish(&cx, &proposed)
                .await
                .expect("publish tombstone generation");
            assert_eq!(published.at_seal_doc_count(), 2);
            assert_eq!(published.tombstone_count(), 2);
            assert_eq!(published.doc_count(), 0);
            assert_eq!(published.materialize_document_id(0), None);
            assert_eq!(published.materialize_document_id(100), None);
            assert_eq!(
                published.loaded_manifest().manifest.field_stats,
                manifest.field_stats
            );
            assert_eq!(
                published
                    .resolve_document_id("same")
                    .expect("all-dead lookup"),
                None
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("reopen published tombstones");
            assert_eq!(reopened.doc_count(), 0);
            assert!(!reopened.is_live(0));
            assert!(original.is_live(0), "older mmap snapshot stays visible");

            let high_watermark = writer
                .snapshot()
                .loaded_manifest()
                .manifest
                .docid_high_watermark;
            let segment_paths = writer
                .snapshot()
                .segments()
                .iter()
                .map(|segment| segment.path().to_path_buf())
                .collect::<Vec<_>>();
            let mut empty = writer.snapshot().next_manifest().expect("empty successor");
            writer
                .snapshot()
                .delete_all(&mut empty)
                .expect("stage delete_all");
            assert!(empty.segments.is_empty());
            assert!(empty.field_stats.is_empty());
            assert_eq!(empty.docid_high_watermark, high_watermark);
            assert_eq!(empty.schema_id, manifest.schema_id);
            assert_eq!(empty.engine_version, manifest.engine_version);
            let empty_snapshot = writer
                .publish(&cx, &empty)
                .await
                .expect("publish delete_all generation");
            assert_eq!(empty_snapshot.at_seal_doc_count(), 0);
            assert_eq!(empty_snapshot.tombstone_count(), 0);
            assert_eq!(empty_snapshot.doc_count(), 0);
            assert!(
                segment_paths.iter().all(|path| path.exists()),
                "publication never synchronously reclaims segment files"
            );
        });
    }

    #[test]
    fn identity_resolution_handles_repeated_upserts_and_duplicate_live_rows() -> TestResult {
        let directory = tempdir()?;
        let mut segments = Vec::new();
        for (ordinal, (docid_lo, seal_seq)) in [(0_u64, 40_u64), (10, 10), (20, 30), (30, 20)]
            .into_iter()
            .enumerate()
        {
            let segment_id = 0x300 + u64::try_from(ordinal)?;
            let mut segment = write_identity_test_segment(
                directory.path(),
                segment_id,
                seal_seq,
                docid_lo,
                &[Some("repeated")],
            )?;
            if docid_lo != 20 {
                assert!(segment.tombstones.insert(u32::try_from(docid_lo)?)?);
            }
            segments.push(segment);
        }
        let manifest = durable_test_manifest(1, segments);
        write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(snapshot.at_seal_doc_count(), 4);
        assert_eq!(snapshot.tombstone_count(), 3);
        assert_eq!(snapshot.doc_count(), 1);
        assert_eq!(
            snapshot.resolve_document_id("repeated")?,
            Some(ResolvedDocumentId {
                segment_id: 0x302,
                seal_seq: 30,
                global_docid: 20,
                content_hash: 1,
            })
        );

        let mut all_dead = manifest.clone();
        assert!(all_dead.segments[2].tombstones.insert(20)?);
        write_manifest(&directory.path().join("MANIFEST"), &all_dead)?;
        assert_eq!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?
                .resolve_document_id("repeated")?,
            None
        );

        let mut duplicate_live = manifest;
        assert!(duplicate_live.segments[0].tombstones.remove(0)?);
        write_manifest(&directory.path().join("MANIFEST"), &duplicate_live)?;
        let duplicate = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert!(matches!(
            duplicate.resolve_document_id("repeated"),
            Err(KeeperError::MultipleLiveDocumentIds {
                first_global_docid: 0,
                duplicate_global_docid: 20,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn concat_representatives_prefer_the_only_live_duplicate_independent_of_row_age() -> TestResult
    {
        let encoded = EncodedIdMapSection::encode(
            0,
            3,
            &[
                Some(IdMapEntryInput::new("same", 11)),
                Some(IdMapEntryInput::new("other", 22)),
                Some(IdMapEntryInput::new("same", 33)),
            ],
        )?;

        assert_eq!(
            resolve_concat_representatives(encoded.section()?, &[0])?,
            [1, 2],
            "a live later row must replace a tombstoned earlier representative",
        );
        assert_eq!(
            resolve_concat_representatives(encoded.section()?, &[2])?,
            [0, 1],
            "a live earlier row must survive a tombstoned later representative",
        );
        assert_eq!(
            resolve_concat_representatives(encoded.section()?, &[0, 2])?,
            [0, 1],
            "an all-dead identity uses its lowest global docid as canonical hash input",
        );
        assert!(matches!(
            resolve_concat_representatives(encoded.section()?, &[]),
            Err(ConcatMergeError::MultipleLiveDocumentIds {
                first_global_docid: 0,
                duplicate_global_docid: 2,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn snapshot_rejects_tombstone_targeting_idmap_hole() -> TestResult {
        let directory = tempdir()?;
        let segment =
            write_identity_test_segment(directory.path(), 0x401, 1, 50, &[None, Some("present")])?;
        let clean_manifest = durable_test_manifest(1, vec![segment]);
        write_manifest(&directory.path().join("MANIFEST"), &clean_manifest)?;
        let clean = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(clean.doc_count(), 1);
        assert!(!clean.is_live(50));
        assert!(clean.is_live(51));
        assert_eq!(clean.materialize_document_id(50), None);
        assert_eq!(
            clean.materialize_document_id(51),
            Some(DocId::new("present"))
        );
        assert!(!crate::argus::LiveDocs::is_live(&clean.segments()[0], 50));
        assert!(crate::argus::LiveDocs::is_live(&clean.segments()[0], 51));

        let mut manifest = clean_manifest;
        assert!(manifest.segments[0].tombstones.insert(50)?);
        write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
            Err(KeeperError::TombstoneReferencesHole {
                segment_id: 0x401,
                global_docid: 50,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn publish_preflight_rejects_tombstone_hole_without_replacing_manifest() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let segment = write_identity_test_segment(
                directory.path(),
                0x402,
                1,
                50,
                &[None, Some("present")],
            )
            .expect("write identity segment");
            let manifest = durable_test_manifest(1, vec![segment]);
            let manifest_path = directory.path().join("MANIFEST");
            write_manifest(&manifest_path, &manifest).expect("write initial manifest");

            let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
                .await
                .expect("open writer");
            let before = std::fs::read(&manifest_path).expect("read initial manifest");
            let mut proposed = writer.snapshot().next_manifest().expect("next generation");
            assert!(
                proposed.segments[0]
                    .tombstones
                    .insert(50)
                    .expect("stage hole tombstone")
            );

            assert!(matches!(
                writer.publish(&cx, &proposed).await,
                Err(KeeperError::TombstoneReferencesHole {
                    segment_id: 0x402,
                    global_docid: 50,
                    ..
                })
            ));
            assert_eq!(
                std::fs::read(&manifest_path).expect("read manifest after rejection"),
                before,
                "preflight rejection must leave the current MANIFEST untouched"
            );
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 1);
            assert_eq!(
                KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                    .expect("reopen prior snapshot")
                    .loaded_manifest()
                    .manifest
                    .generation,
                1
            );
        });
    }

    #[test]
    fn fallback_ignores_manifest_temp_and_rejects_incompatible_claim() -> TestResult {
        let directory = tempdir()?;
        let segment = write_test_segment(directory.path(), 0x11, 1, 0, 2)?;
        let previous = durable_test_manifest(1, vec![segment]);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let temp = durable_test_manifest(2, Vec::new());
        write_manifest(&directory.path().join(".tmp-manifest-2"), &temp)?;
        let before = directory_bytes(directory.path())?;

        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(
            snapshot.loaded_manifest().source,
            ManifestSource::PreviousAfterMissingCurrent
        );
        assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
        assert_eq!(directory_bytes(directory.path())?, before);

        std::fs::write(directory.path().join("gen-3.claim"), [])?;
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
            Err(KeeperError::InvalidRecoveryClaim {
                recovered: 1,
                claimed: 3,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn deterministic_publish_crash_states_recover_read_only_and_gc_exactly() -> TestResult {
        const CASES: [&str; 10] = [
            "segment temp written",
            "segment temp fsynced",
            "segment renamed",
            "segment directory fsynced",
            "uncommitted sidecar emitted",
            "manifest temp written",
            "manifest temp fsynced",
            "between manifest renames",
            "new manifest current",
            "committed sidecar emitted",
        ];

        for (case_index, label) in CASES.into_iter().enumerate() {
            let directory = tempdir()?;
            let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
            let segment_b_name = canonical_segment_name(0xb);
            let segment_b = if case_index >= 2 {
                Some(write_test_segment(directory.path(), 0xb, 2, 2, 4)?)
            } else {
                let encoded = encoded_test_segment(0xb, 2, 4, 1)?;
                let temp_path = directory.path().join(".tmp-segment-000000000000000b");
                std::fs::write(&temp_path, encoded.as_bytes())?;
                if case_index == 1 {
                    File::open(&temp_path)?.sync_all()?;
                }
                None
            };

            let previous = durable_test_manifest(1, vec![segment_a.clone()]);
            if case_index == 7 {
                write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
            } else if case_index >= 8 {
                let current = durable_test_manifest(
                    2,
                    vec![segment_a.clone(), segment_b.clone().expect("segment B")],
                );
                write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
                write_manifest(&directory.path().join("MANIFEST"), &current)?;
            } else {
                write_manifest(&directory.path().join("MANIFEST"), &previous)?;
            }

            if matches!(case_index, 5..=7) {
                let next = durable_test_manifest(
                    2,
                    vec![segment_a.clone(), segment_b.clone().expect("segment B")],
                );
                let temp_manifest = directory.path().join(".tmp-manifest-2");
                write_manifest(&temp_manifest, &next)?;
                if case_index == 6 {
                    File::open(&temp_manifest)?.sync_all()?;
                }
            }
            if case_index == 3 {
                sync_directory(directory.path())?;
            }
            if case_index == 4 || case_index == 9 {
                std::fs::write(
                    directory.path().join(format!("{segment_b_name}.fec")),
                    b"sidecar",
                )?;
            }

            let before = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: first recovery failed: {error}"));
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: second recovery failed: {error}"));
            assert_eq!(first.loaded_manifest(), second.loaded_manifest(), "{label}");
            assert_eq!(directory_bytes(directory.path())?, before, "{label}");
            let expected_committed = if case_index >= 8 { 2 } else { 1 };
            assert_eq!(first.segments().len(), expected_committed, "{label}");
            for segment in first.segments() {
                assert_eq!(
                    segment.section(SectionKind::TERMDICT)?,
                    Some(b"termdict".as_slice()),
                    "{label}"
                );
            }

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let mut expected_removed = match case_index {
                0 | 1 => vec![PathBuf::from(".tmp-segment-000000000000000b")],
                2 | 3 => vec![PathBuf::from(&segment_b_name)],
                4 => vec![
                    PathBuf::from(&segment_b_name),
                    PathBuf::from(format!("{segment_b_name}.fec")),
                ],
                5..=7 => vec![
                    PathBuf::from(".tmp-manifest-2"),
                    PathBuf::from(&segment_b_name),
                ],
                8 | 9 => Vec::new(),
                _ => unreachable!(),
            };
            expected_removed.sort_unstable();
            assert_eq!(report.removed, expected_removed, "{label}");

            let mut expected_inventory = before;
            expected_inventory.retain(|(name, _)| {
                !report
                    .removed
                    .iter()
                    .any(|removed| removed.as_os_str() == name)
            });
            assert_eq!(
                directory_bytes(directory.path())?,
                expected_inventory,
                "{label}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: post-GC recovery failed: {error}"));
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{label}"
            );
            assert_eq!(
                collect_writer_garbage_at(
                    directory.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                )?,
                GarbageCollectionReport {
                    removed: Vec::new()
                },
                "{label}"
            );
        }
        Ok(())
    }

    #[test]
    fn writer_gc_unions_both_slots_honors_grace_and_is_idempotent() -> TestResult {
        let directory = tempdir()?;
        let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
        let segment_b = write_test_segment(directory.path(), 0xb, 3, 2, 4)?;
        let segment_c = write_test_segment(directory.path(), 0xc, 2, 4, 6)?;
        let segment_e = write_test_segment(directory.path(), 0xe, 4, 6, 8)?;
        let segment_f = write_test_segment(directory.path(), 0x10, 5, 8, 10)?;
        let previous = durable_test_manifest(1, vec![segment_a.clone(), segment_c.clone()]);
        let mut current = durable_test_manifest(2, vec![segment_a, segment_b]);
        current.docid_high_watermark = previous.docid_high_watermark;
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        write_manifest(&directory.path().join("MANIFEST"), &current)?;

        std::fs::write(directory.path().join(".tmp-manifest-3"), b"uncommitted")?;
        std::fs::write(
            directory.path().join(format!(
                "{}.fec",
                canonical_segment_name(segment_e.segment_id)
            )),
            b"orphan sidecar",
        )?;
        let fresh_base = directory
            .path()
            .join(canonical_segment_name(segment_f.segment_id));
        let old_sidecar_with_fresh_base = directory.path().join(format!(
            "{}.fec",
            canonical_segment_name(segment_f.segment_id)
        ));
        std::fs::write(&old_sidecar_with_fresh_base, b"sidecar older than base")?;
        File::options().write(true).open(&fresh_base)?.set_times(
            std::fs::FileTimes::new().set_modified(
                SystemTime::now()
                    .checked_add(Duration::from_secs(3_600))
                    .expect("test clock remains representable"),
            ),
        )?;
        std::fs::write(
            directory.path().join(format!(
                "{}.fec",
                canonical_segment_name(segment_c.segment_id)
            )),
            b"previous sidecar",
        )?;
        std::fs::write(
            directory.path().join("MANIFEST.fec"),
            b"live manifest sidecar",
        )?;
        std::fs::write(directory.path().join("notes.fec"), b"user data")?;
        std::fs::write(directory.path().join(".quarantine"), b"operator data")?;
        let quarantine_name = format!("{}.quarantine", canonical_segment_name(0xf));
        std::fs::write(
            directory.path().join(&quarantine_name),
            b"quarantined operator data",
        )?;
        std::fs::write(
            directory.path().join("SEG-000000000000000f.fslx"),
            b"user data",
        )?;
        std::fs::create_dir(directory.path().join(".tmp-directory"))?;

        #[cfg(unix)]
        let outside_fixture = {
            use std::os::unix::ffi::OsStringExt;

            let outside = tempdir()?;
            let target = outside.path().join("outside-data");
            std::fs::write(&target, b"outside")?;
            std::os::unix::fs::symlink(
                &target,
                directory.path().join("seg-000000000000000f.fslx"),
            )?;
            std::os::unix::fs::symlink(&target, directory.path().join(".tmp-link"))?;
            std::fs::write(
                directory
                    .path()
                    .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec())),
                b"non-UTF-8 user data",
            )?;

            let before_open = directory_bytes(directory.path())?;
            let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(directory_bytes(directory.path())?, before_open);
            assert_eq!(snapshot.segments().len(), 2);
            assert_eq!(std::fs::read(&target)?, b"outside");
            (outside, target)
        };

        let fresh = collect_writer_garbage_under_lock(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
        )?;
        assert!(
            fresh.is_empty(),
            "fresh garbage must survive the grace window"
        );
        assert!(
            directory
                .path()
                .join(canonical_segment_name(segment_e.segment_id))
                .exists()
        );

        std::fs::write(directory.path().join("gen-2.claim"), [])?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?;
        let orphan_name = canonical_segment_name(segment_e.segment_id);
        assert_eq!(
            report.removed,
            vec![
                PathBuf::from(".tmp-manifest-3"),
                PathBuf::from("gen-2.claim"),
                PathBuf::from(&orphan_name),
                PathBuf::from(format!("{orphan_name}.fec")),
            ]
        );
        assert!(directory.path().join(canonical_segment_name(0xc)).exists());
        assert!(fresh_base.exists());
        assert!(old_sidecar_with_fresh_base.exists());
        assert!(
            directory
                .path()
                .join(format!("{}.fec", canonical_segment_name(0xc)))
                .exists()
        );
        assert!(directory.path().join("notes.fec").exists());
        assert!(directory.path().join(".quarantine").exists());
        assert!(directory.path().join(&quarantine_name).exists());
        assert!(directory.path().join("SEG-000000000000000f.fslx").exists());
        assert!(directory.path().join(".tmp-directory").is_dir());
        #[cfg(unix)]
        {
            assert!(directory.path().join(".tmp-link").is_symlink());
            assert!(
                directory
                    .path()
                    .join("seg-000000000000000f.fslx")
                    .is_symlink()
            );
            use std::os::unix::ffi::OsStringExt;
            assert!(
                directory
                    .path()
                    .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec()))
                    .exists()
            );
            assert_eq!(std::fs::read(&outside_fixture.1)?, b"outside");
        }
        assert!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?
            .is_empty()
        );
        Ok(())
    }

    #[test]
    fn future_claim_blocks_gc_before_any_deletion() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        std::fs::write(directory.path().join("gen-3.claim"), [])?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ClaimedGenerationPending {
                current: 2,
                claimed: 3,
                ..
            })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);
        Ok(())
    }

    #[test]
    fn stale_claim_still_honors_the_gc_grace_period() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join("gen-2.claim"), [])?;

        assert!(
            collect_writer_garbage_under_lock(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
            )?
            .is_empty()
        );
        assert!(directory.path().join("gen-2.claim").exists());

        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        assert_eq!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?
            .removed,
            vec![PathBuf::from("gen-2.claim")]
        );
        Ok(())
    }

    #[test]
    fn malformed_claim_blocks_gc_before_any_deletion() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        std::fs::create_dir(directory.path().join("gen-2.claim"))?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::InvalidClaimArtifact { .. })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);
        assert!(directory.path().join("gen-2.claim").is_dir());

        let nonzero = tempdir()?;
        write_manifest(
            &nonzero.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(nonzero.path().join(".tmp-old"), b"must survive")?;
        std::fs::write(nonzero.path().join("gen-2.claim"), b"not empty")?;
        let before = directory_bytes(nonzero.path())?;
        assert!(matches!(
            collect_writer_garbage_at(
                nonzero.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::InvalidClaimArtifact { .. })
        ));
        assert_eq!(directory_bytes(nonzero.path())?, before);

        #[cfg(unix)]
        {
            let symlinked = tempdir()?;
            write_manifest(
                &symlinked.path().join("MANIFEST"),
                &durable_test_manifest(2, Vec::new()),
            )?;
            std::fs::write(symlinked.path().join(".tmp-old"), b"must survive")?;
            let target = symlinked.path().join("claim-target");
            std::fs::write(&target, [])?;
            std::os::unix::fs::symlink(&target, symlinked.path().join("gen-2.claim"))?;
            let before = directory_bytes(symlinked.path())?;
            assert!(matches!(
                collect_writer_garbage_at(
                    symlinked.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                ),
                Err(KeeperError::InvalidClaimArtifact { .. })
            ));
            assert_eq!(directory_bytes(symlinked.path())?, before);
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn gc_rejects_symlinked_directory_alias_without_touching_target() -> TestResult {
        let target = tempdir()?;
        std::fs::write(target.path().join(".tmp-old"), b"must survive")?;
        let alias_root = tempdir()?;
        let alias = alias_root.path().join("index-alias");
        std::os::unix::fs::symlink(target.path(), &alias)?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_abandoned_genesis_garbage_at(&alias, GarbageCollectionOptions::default(), now,),
            Err(KeeperError::Io {
                operation: "open no-follow garbage-collection directory",
                ..
            })
        ));
        assert_eq!(
            std::fs::read(target.path().join(".tmp-old"))?,
            b"must survive"
        );
        Ok(())
    }

    #[test]
    fn genesis_gc_requires_absent_slots_and_no_pending_claim() -> TestResult {
        let directory = tempdir()?;
        let segment = write_test_segment(directory.path(), 0x44, 1, 0, 2)?;
        write_manifest(
            &directory.path().join(".tmp-manifest-1"),
            &durable_test_manifest(1, vec![segment.clone()]),
        )?;
        std::fs::write(directory.path().join(".tmp-write-buffer"), b"uncommitted")?;
        std::fs::write(directory.path().join("MANIFEST.fec"), b"orphan sidecar")?;
        std::fs::write(
            directory.path().join("MANIFEST.fec.tmp"),
            b"interrupted manifest sidecar",
        )?;
        let segment_sidecar_temp = directory.path().join(format!(
            "{}.fec.tmp",
            canonical_segment_name(segment.segment_id)
        ));
        std::fs::write(&segment_sidecar_temp, b"interrupted segment sidecar")?;
        std::fs::write(directory.path().join("notes"), b"operator data")?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        let report = collect_abandoned_genesis_garbage_at(
            directory.path(),
            GarbageCollectionOptions::default(),
            now,
        )?;
        let segment_name = canonical_segment_name(segment.segment_id);
        assert_eq!(
            report.removed,
            vec![
                PathBuf::from(".tmp-manifest-1"),
                PathBuf::from(".tmp-write-buffer"),
                PathBuf::from("MANIFEST.fec"),
                PathBuf::from("MANIFEST.fec.tmp"),
                PathBuf::from(segment_name),
                segment_sidecar_temp
                    .file_name()
                    .expect("sidecar temp has a file name")
                    .into(),
            ]
        );
        assert_eq!(
            std::fs::read(directory.path().join("notes"))?,
            b"operator data"
        );
        assert!(
            collect_abandoned_genesis_garbage_at(
                directory.path(),
                GarbageCollectionOptions::default(),
                now,
            )?
            .is_empty()
        );

        let claimed = tempdir()?;
        std::fs::write(claimed.path().join(".tmp-manifest-1"), b"must survive")?;
        std::fs::write(claimed.path().join("gen-1.claim"), [])?;
        let before = directory_bytes(claimed.path())?;
        assert!(matches!(
            collect_abandoned_genesis_garbage_at(
                claimed.path(),
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ClaimedGenerationPending {
                current: 0,
                claimed: 1,
                ..
            })
        ));
        assert_eq!(directory_bytes(claimed.path())?, before);

        let corrupt = tempdir()?;
        std::fs::write(corrupt.path().join("MANIFEST"), b"corrupt")?;
        std::fs::write(corrupt.path().join(".tmp-old"), b"must survive")?;
        let before = directory_bytes(corrupt.path())?;
        assert!(matches!(
            collect_abandoned_genesis_garbage_at(
                corrupt.path(),
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ManifestCorrupted { .. })
        ));
        assert_eq!(directory_bytes(corrupt.path())?, before);
        Ok(())
    }

    #[test]
    fn failed_writer_recovery_never_collects_and_path_guard_rejects_escape() -> TestResult {
        let directory = tempdir()?;
        let missing = ManifestSegment {
            segment_id: 0xdead,
            seal_seq: 1,
            file_len: 100,
            file_xxh3: 7,
            docid_lo: 0,
            docid_hi: 2,
            doc_count: 1,
            tombstones: TombstoneSet::new(),
        };
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(1, vec![missing]),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::SegmentOpen { .. })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);

        for unsafe_path in [
            Path::new("../outside"),
            Path::new("nested/file"),
            Path::new("/absolute"),
        ] {
            assert!(matches!(
                safe_direct_child(directory.path(), unsafe_path),
                Err(KeeperError::UnsafeGarbagePath { .. })
            ));
        }
        assert_eq!(
            safe_direct_child(directory.path(), Path::new(".tmp-owned"))?,
            directory.path().join(".tmp-owned")
        );
        assert!(classify_garbage_candidate(OsStr::new("notes.fec")).is_none());
        assert!(classify_garbage_candidate(OsStr::new("notes.fec.tmp")).is_none());
        assert!(classify_garbage_candidate(OsStr::new("SEG-0000000000000001.fslx")).is_none());
        assert!(matches!(
            classify_garbage_candidate(OsStr::new("seg-0000000000000001.fslx.fec.tmp")),
            Some(GarbageCandidate::Temporary)
        ));
        assert!(matches!(
            classify_garbage_candidate(OsStr::new("MANIFEST.fec.tmp")),
            Some(GarbageCandidate::Temporary)
        ));
        Ok(())
    }

    fn run_with_test_cx<T, F, Fut>(operation: F) -> T
    where
        T: Send + 'static,
        F: FnOnce(Cx) -> Fut + 'static,
        Fut: std::future::Future<Output = T> + 'static,
    {
        let output = Arc::new(std::sync::Mutex::new(None));
        let task_output = Arc::clone(&output);
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let value = operation(cx).await;
            let mut slot = task_output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *slot = Some(value);
        });
        let output = Arc::try_unwrap(output)
            .unwrap_or_else(|_| panic!("test runtime retained its result slot"));
        output
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("test runtime produced a result")
    }

    #[cfg(unix)]
    struct WriterChild {
        child: Option<std::process::Child>,
    }

    #[cfg(unix)]
    impl WriterChild {
        fn child_mut(&mut self) -> &mut std::process::Child {
            self.child.as_mut().expect("child is still owned")
        }

        fn wait_success(mut self) -> TestResult {
            let status = self.child.take().expect("child is still owned").wait()?;
            if !status.success() {
                return Err(format!("writer child exited with {status}").into());
            }
            Ok(())
        }

        fn kill_and_reap(mut self) -> TestResult {
            let mut child = self.child.take().expect("child is still owned");
            let kill_result = child.kill();
            let wait_result = child.wait();
            kill_result?;
            let status = wait_result?;
            if status.success() {
                return Err("killed writer child unexpectedly exited successfully".into());
            }
            Ok(())
        }
    }

    #[cfg(unix)]
    impl Drop for WriterChild {
        fn drop(&mut self) {
            if let Some(child) = &mut self.child {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    #[cfg(unix)]
    fn spawn_writer_child(
        role: &str,
        directory: &Path,
        control: &Path,
        identifier: &str,
    ) -> Result<WriterChild, Box<dyn std::error::Error>> {
        let child = std::process::Command::new(std::env::current_exe()?)
            .arg("--exact")
            .arg("keeper::tests::writer_lock_child_dispatch")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env("QUILL_WRITER_LOCK_ROLE", role)
            .env("QUILL_WRITER_LOCK_DIRECTORY", directory)
            .env("QUILL_WRITER_LOCK_CONTROL", control)
            .env("QUILL_WRITER_LOCK_ID", identifier)
            .spawn()?;
        Ok(WriterChild { child: Some(child) })
    }

    #[cfg(unix)]
    fn wait_for_child_marker(child: &mut WriterChild, marker: &Path, label: &str) -> TestResult {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if marker.exists() {
                return Ok(());
            }
            if let Some(status) = child.child_mut().try_wait()? {
                return Err(format!("{label} exited before its marker: {status}").into());
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!("timed out waiting for {label}").into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    fn create_test_index(directory: &Path) -> Result<(), String> {
        let directory = directory.to_path_buf();
        run_with_test_cx(move |cx| async move {
            KeeperSnapshot::create(&cx, directory, DEFAULT_SCHEMA)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
    }

    #[cfg(unix)]
    #[test]
    fn writer_lock_child_dispatch() {
        let Ok(role) = std::env::var("QUILL_WRITER_LOCK_ROLE") else {
            return;
        };
        let directory = PathBuf::from(
            std::env::var_os("QUILL_WRITER_LOCK_DIRECTORY")
                .expect("child index directory is configured"),
        );
        let control = PathBuf::from(
            std::env::var_os("QUILL_WRITER_LOCK_CONTROL")
                .expect("child control directory is configured"),
        );
        let identifier = std::env::var("QUILL_WRITER_LOCK_ID").expect("child id is configured");
        let result: Result<(), String> = run_with_test_cx(move |cx| async move {
            if role == "contend" {
                let start = control.join("START");
                while !start.exists() {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            match KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA).await {
                Ok(writer) => {
                    if role == "hold_unpublished" {
                        std::fs::write(
                            directory.join("seg-feedfacefeedface.fslx"),
                            b"sealed but not yet published",
                        )
                        .map_err(|error| error.to_string())?;
                    }
                    let marker = if role == "contend" {
                        control.join(format!("winner-{identifier}"))
                    } else {
                        control.join(format!("ready-{identifier}"))
                    };
                    std::fs::write(&marker, []).map_err(|error| error.to_string())?;
                    let release = control.join("RELEASE");
                    while !release.exists() {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    drop(writer);
                    Ok(())
                }
                Err(KeeperError::WriterBusy { .. }) if role == "contend" => {
                    std::fs::write(control.join(format!("busy-{identifier}")), [])
                        .map_err(|error| error.to_string())
                }
                Err(error) => Err(error.to_string()),
            }
        });
        result.unwrap_or_else(|error| panic!("writer child failed: {error}"));
    }

    #[test]
    fn writer_lock_wire_golden_and_corruption_matrix() {
        let record = WriterLockRecord {
            pid: 42,
            pid_start_nonce: 0x1122_3344_5566_7788,
            acquired_unix_s: 1_700_000_000,
        };
        let golden = hex_bytes(
            "46534c584c434b00010000002a0000008877665544332211\
             00f153650000000044f89139",
        );
        assert_eq!(record.to_bytes().as_slice(), golden);
        assert_eq!(WriterLockRecord::from_bytes(&golden), Ok(record));
        for length in 0..WRITER_LOCK_RECORD_BYTES {
            assert!(WriterLockRecord::from_bytes(&golden[..length]).is_err());
        }
        let mut bad_magic = golden.clone();
        bad_magic[0] ^= 0xff;
        assert!(WriterLockRecord::from_bytes(&bad_magic).is_err());
        let mut bad_version = golden.clone();
        bad_version[8] = 2;
        let checksum = crc32fast::hash(&bad_version[..32]);
        bad_version[32..].copy_from_slice(&checksum.to_le_bytes());
        assert!(WriterLockRecord::from_bytes(&bad_version).is_err());
        let mut zero_pid = golden.clone();
        zero_pid[12..16].copy_from_slice(&0_u32.to_le_bytes());
        let checksum = crc32fast::hash(&zero_pid[..32]);
        zero_pid[32..].copy_from_slice(&checksum.to_le_bytes());
        assert!(WriterLockRecord::from_bytes(&zero_pid).is_err());
        let mut bad_crc = golden;
        bad_crc[20] ^= 1;
        assert!(WriterLockRecord::from_bytes(&bad_crc).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn writer_admission_is_exclusive_reusable_and_corruption_fails_closed() -> TestResult {
        let directory = tempdir()?;
        let first = acquire_writer_admission(directory.path())?;
        let lock_path = directory.path().join("LOCK");
        let active_bytes = std::fs::read(&lock_path)?;
        assert_eq!(active_bytes.len(), WRITER_LOCK_RECORD_BYTES);
        assert!(matches!(
            acquire_writer_admission(directory.path()),
            Err(KeeperError::WriterBusy {
                owner_pid: Some(pid),
                ..
            }) if pid == std::process::id()
        ));
        assert_eq!(std::fs::read(&lock_path)?, active_bytes);
        drop(first);
        assert_eq!(std::fs::metadata(&lock_path)?.len(), 0);

        let second = acquire_writer_admission(directory.path())?;
        drop(second);
        std::fs::write(&lock_path, b"truncated")?;
        let before = std::fs::read(&lock_path)?;
        assert!(matches!(
            acquire_writer_admission(directory.path()),
            Err(KeeperError::WriterLockCorrupted { .. })
        ));
        assert_eq!(std::fs::read(lock_path)?, before);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn old_guards_preserve_replacement_lock_and_claim_names() -> TestResult {
        let directory = tempdir()?;
        let admission = acquire_writer_admission(directory.path())?;
        let original_lock = directory.path().join("LOCK");
        let moved_lock = directory.path().join("LOCK.old");
        std::fs::rename(&original_lock, &moved_lock)?;
        let replacement = WriterLockRecord {
            pid: std::process::id(),
            pid_start_nonce: 0xfeed_face,
            acquired_unix_s: 1_700_000_001,
        }
        .to_bytes();
        std::fs::write(&original_lock, replacement)?;
        assert!(matches!(
            admission.ensure_directory_identity(),
            Err(KeeperError::WriterLockCorrupted { .. })
        ));
        drop(admission);
        assert_eq!(std::fs::read(&original_lock)?, replacement);
        assert_eq!(std::fs::metadata(&moved_lock)?.len(), 0);

        let claims = tempdir()?;
        let admission = acquire_writer_admission(claims.path())?;
        let claim = GenerationClaimGuard::acquire(Arc::clone(&admission), 1)?;
        assert!(matches!(
            GenerationClaimGuard::acquire(Arc::clone(&admission), 1),
            Err(KeeperError::GenerationClaimConflict { generation: 1, .. })
        ));
        let claim_path = claims.path().join("gen-1.claim");
        let moved_claim = claims.path().join("gen-1.claim.old");
        std::fs::rename(&claim_path, moved_claim)?;
        std::fs::write(&claim_path, [])?;
        drop(claim);
        assert!(
            claim_path.exists(),
            "old guard must preserve replacement claim"
        );
        drop(admission);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn sigstop_writer_is_never_reaped_and_readers_remain_touchless() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let control = tempdir()?;
        let ready = control.path().join("ready-holder");
        let mut child =
            spawn_writer_child("hold_unpublished", index.path(), control.path(), "holder")?;
        wait_for_child_marker(&mut child, &ready, "writer holder")?;
        assert!(index.path().join("seg-feedfacefeedface.fslx").exists());

        let lock_path = index.path().join("LOCK");
        let lock_file = OpenOptions::new().write(true).open(&lock_path)?;
        lock_file.set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;
        let before = directory_bytes(index.path())?;
        rustix::process::kill_process(
            rustix::process::Pid::from_child(child.child_mut()),
            rustix::process::Signal::STOP,
        )?;

        let contender_path = index.path().to_path_buf();
        let blocked: Result<(), String> = run_with_test_cx(move |cx| async move {
            match KeeperWriter::open(&cx, contender_path, DEFAULT_SCHEMA).await {
                Err(KeeperError::WriterBusy { .. }) => Ok(()),
                Ok(writer) => {
                    drop(writer);
                    Err("SIGSTOP holder was incorrectly taken over".to_owned())
                }
                Err(error) => Err(format!("unexpected contender failure: {error}")),
            }
        });
        blocked.map_err(io::Error::other)?;
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(index.path())?, before);

        rustix::process::kill_process(
            rustix::process::Pid::from_child(child.child_mut()),
            rustix::process::Signal::CONT,
        )?;
        std::fs::write(control.path().join("RELEASE"), [])?;
        child.wait_success()?;
        let reopen_path = index.path().to_path_buf();
        let reopened: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open(&cx, reopen_path, DEFAULT_SCHEMA)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        reopened.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn two_processes_racing_dead_owner_takeover_admit_exactly_one() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let dead_control = tempdir()?;
        let mut dead = spawn_writer_child("hold", index.path(), dead_control.path(), "dead")?;
        wait_for_child_marker(
            &mut dead,
            &dead_control.path().join("ready-dead"),
            "dead-owner fixture",
        )?;
        dead.kill_and_reap()?;
        assert_eq!(
            std::fs::metadata(index.path().join("LOCK"))?.len(),
            usize_to_u64(WRITER_LOCK_RECORD_BYTES)
        );

        let control = tempdir()?;
        let mut first = spawn_writer_child("contend", index.path(), control.path(), "a")?;
        let mut second = spawn_writer_child("contend", index.path(), control.path(), "b")?;
        std::fs::write(control.path().join("START"), [])?;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let first_done =
                control.path().join("winner-a").exists() || control.path().join("busy-a").exists();
            let second_done =
                control.path().join("winner-b").exists() || control.path().join("busy-b").exists();
            if first_done && second_done {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err("timed out waiting for takeover contenders".into());
            }
            if let Some(status) = first.child_mut().try_wait()?
                && !first_done
            {
                return Err(format!("first contender exited early: {status}").into());
            }
            if let Some(status) = second.child_mut().try_wait()?
                && !second_done
            {
                return Err(format!("second contender exited early: {status}").into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let winners = usize::from(control.path().join("winner-a").exists())
            + usize::from(control.path().join("winner-b").exists());
        let busy = usize::from(control.path().join("busy-a").exists())
            + usize::from(control.path().join("busy-b").exists());
        assert_eq!(winners, 1);
        assert_eq!(busy, 1);
        std::fs::write(control.path().join("RELEASE"), [])?;
        first.wait_success()?;
        second.wait_success()?;
        assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_claims_generations_and_abandons_interrupted_temp() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let first: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(
                std::fs::metadata(directory.join("LOCK"))
                    .map_err(|error| error.to_string())?
                    .len(),
                usize_to_u64(WRITER_LOCK_RECORD_BYTES)
            );
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            assert!(!directory.join("gen-2.claim").exists());
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            drop(writer);
            Ok(())
        });
        first.map_err(io::Error::other)?;
        assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);

        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        let interrupted = Manifest::empty(3, schema_id, 0).to_bytes()?;
        std::fs::write(index.path().join(".tmp-manifest-3"), interrupted)?;
        std::fs::write(index.path().join("gen-3.claim"), [])?;
        let directory = index.path().to_path_buf();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            assert!(!directory.join(".tmp-manifest-3").exists());
            assert!(!directory.join("gen-3.claim").exists());
            writer
                .publish(&cx, &Manifest::empty(3, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 3);
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert!(std::fs::read_dir(index.path())?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.starts_with(".tmp-abandoned-manifest-3-"))
        }));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_adopts_exact_published_segment_and_preserves_retry_temp() -> TestResult {
        let directory = tempdir()?.keep();
        let assertion_directory = directory.clone();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let encoded =
                encoded_test_segment(0xa11d, 0, 2, 1).map_err(|error| error.to_string())?;
            let canonical = directory.join(canonical_segment_name(encoded.header().segment_id));
            std::fs::write(&canonical, encoded.as_bytes()).map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let temp_path = pending.path().to_path_buf();

            let published = writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;

            if published != canonical {
                return Err("reconciled segment returned the wrong canonical path".to_owned());
            }
            if std::fs::read(&canonical).map_err(|error| error.to_string())? != encoded.as_bytes() {
                return Err("reconciled canonical segment bytes changed".to_owned());
            }
            if std::fs::read(&temp_path).map_err(|error| error.to_string())? != encoded.as_bytes() {
                return Err("reconciled retry temp was not preserved exactly".to_owned());
            }
            Ok(())
        });
        outcome.map_err(io::Error::other)?;
        assert!(assertion_directory.is_dir());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_rejects_differing_published_segment_without_overwrite() -> TestResult {
        let directory = tempdir()?.keep();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let encoded =
                encoded_test_segment(0xc011, 0, 2, 1).map_err(|error| error.to_string())?;
            let canonical = directory.join(canonical_segment_name(encoded.header().segment_id));
            let mut differing = encoded.as_bytes().to_vec();
            let changed = differing
                .last_mut()
                .ok_or_else(|| "encoded segment fixture must not be empty".to_owned())?;
            *changed ^= 0x40;
            std::fs::write(&canonical, &differing).map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let temp_path = pending.path().to_path_buf();

            match writer.publish_segment(&cx, pending).await {
                Err(KeeperError::Io {
                    operation: "reconcile published segment",
                    source,
                    ..
                }) if source.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(format!(
                        "unexpected segment reconciliation failure: {error}"
                    ));
                }
                Ok(_) => return Err("differing canonical segment was adopted".to_owned()),
            }
            if std::fs::read(&canonical).map_err(|error| error.to_string())? != differing {
                return Err("differing canonical segment was overwritten".to_owned());
            }
            if std::fs::read(&temp_path).map_err(|error| error.to_string())? != encoded.as_bytes() {
                return Err("retry temp changed after canonical conflict".to_owned());
            }
            Ok(())
        });
        outcome.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_reconciles_exact_installed_manifest_from_stale_snapshot() -> TestResult {
        let directory = tempdir()?.keep();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            let proposed = Manifest::empty(2, schema_id, 0);
            ManifestPublisher::new(&directory)
                .publish(&cx, &proposed)
                .await
                .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot unexpectedly advanced before retry".to_owned());
            }

            writer
                .publish(&cx, &proposed)
                .await
                .map_err(|error| error.to_string())?;

            let installed = &writer.snapshot().loaded_manifest().manifest;
            if installed.generation != 2 {
                return Err("exact installed proposal was not reconciled".to_owned());
            }
            if !manifest_matches_proposal(installed, &proposed) {
                return Err("reconciled MANIFEST differs from the proposal".to_owned());
            }
            Ok(())
        });
        outcome.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_rejects_differing_installed_manifest_from_stale_snapshot() -> TestResult {
        let directory = tempdir()?.keep();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            let mut installed = Manifest::empty(2, schema_id, 0);
            installed.last_publish_unix_s = 1_700_000_000;
            let mut differing = installed.clone();
            differing.last_publish_unix_s = 1_700_000_001;
            ManifestPublisher::new(&directory)
                .publish(&cx, &installed)
                .await
                .map_err(|error| error.to_string())?;

            match writer.publish(&cx, &differing).await {
                Err(KeeperError::GenerationConflict {
                    expected: 3,
                    proposed: 2,
                }) => {}
                Err(error) => {
                    return Err(format!("unexpected differing-proposal failure: {error}"));
                }
                Ok(_) => return Err("differing installed MANIFEST was reconciled".to_owned()),
            }
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot advanced after differing proposal".to_owned());
            }
            let on_disk = load_manifest_pair(&directory)
                .map_err(|error| error.to_string())?
                .manifest;
            if on_disk != installed {
                return Err("differing retry changed the installed MANIFEST".to_owned());
            }
            Ok(())
        });
        outcome.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_preflights_missing_segment_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xdead, 0, 1, 1)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::SegmentOpen { .. }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("missing segment was published as authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("MANIFEST changed before segment preflight failed".to_owned());
            }
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot advanced after rejected publication".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_validates_manifest_before_segment_preflight() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xbad, 0, 1, 1)?;
        let mut proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        proposed.generation = 0;
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::InvalidManifest { .. }) => Ok(()),
                Err(error) => Err(format!("unexpected publish failure: {error}")),
                Ok(_) => Err("invalid manifest reached segment preflight".to_owned()),
            }
        });
        rejected.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_preflights_corrupt_segment_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xcafe, 0, 1, 1)?;
        let section_offset = usize::try_from(encoded.section_entries()[0].offset)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let published = writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            let mut bytes = std::fs::read(&published).map_err(|error| error.to_string())?;
            bytes[section_offset] ^= 0x80;
            std::fs::write(&published, bytes).map_err(|error| error.to_string())?;

            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::SegmentOpen { .. }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("corrupt segment was published as authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("MANIFEST changed before segment preflight failed".to_owned());
            }
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot advanced after rejected publication".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_preflights_segment_sidecar_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xfec, 0, 1, 1)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let protector = test_file_protector();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let published = publish_pending_segment(pending).map_err(|error| error.to_string())?;
            if FileProtector::sidecar_path(&published).exists() {
                return Err(
                    "ordinary segment publication unexpectedly created a sidecar".to_owned(),
                );
            }

            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::Durability {
                    operation: "preflight durable segment sidecar",
                    ..
                }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("unprotected segment became durable authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("durable MANIFEST changed before sidecar preflight failed".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn unrepairable_segment_is_fail_closed_by_default_and_retained_when_opted_in() -> TestResult {
        let index = tempdir()?;
        let segment_id = 0xdeca_fbad;
        let encoded = encoded_test_segment(segment_id, 0, 2, 1)?;
        let expected_doc_count = u64::from(encoded.header().doc_count);
        let manifest_segment = manifest_segment(&encoded, 1);
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, protector.clone())
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let published = writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            let mut next = writer
                .snapshot()
                .next_manifest()
                .map_err(|error| error.to_string())?;
            next.docid_high_watermark = manifest_segment.docid_hi;
            next.segments.push(manifest_segment);
            writer
                .publish(&cx, &next)
                .await
                .map_err(|error| error.to_string())?;
            drop(writer);

            let mut corrupt = std::fs::read(&published).map_err(|error| error.to_string())?;
            let corrupt_offset = corrupt.len() / 2;
            corrupt[corrupt_offset] ^= 0x80;
            std::fs::write(&published, &corrupt).map_err(|error| error.to_string())?;
            let sidecar = FileProtector::sidecar_path(&published);
            std::fs::write(&sidecar, b"invalid repair sidecar")
                .map_err(|error| error.to_string())?;

            match KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, protector.clone())
                .await
            {
                Err(KeeperError::SegmentOpen { .. }) => {}
                Err(error) => return Err(format!("unexpected fail-closed error: {error}")),
                Ok(_) => return Err("default durable open quarantined without opt-in".to_owned()),
            }
            if !published.is_file() {
                return Err("fail-closed open changed the corrupt source".to_owned());
            }

            let writer = KeeperWriter::open_durable_with_policy(
                &cx,
                &directory,
                DEFAULT_SCHEMA,
                protector,
                UnrepairableSegmentPolicy::Quarantine,
            )
            .await
            .map_err(|error| error.to_string())?;
            let snapshot = writer.snapshot();
            if snapshot.loaded_manifest().manifest.generation != 3
                || !snapshot.loaded_manifest().manifest.segments.is_empty()
                || !snapshot.is_degraded()
                || snapshot.quarantined_segments().len() != 1
                || snapshot.estimated_missing_docs() != expected_doc_count
            {
                return Err(format!(
                    "unexpected degraded snapshot: generation={} segments={} quarantine={} missing={}",
                    snapshot.loaded_manifest().manifest.generation,
                    snapshot.loaded_manifest().manifest.segments.len(),
                    snapshot.quarantined_segments().len(),
                    snapshot.estimated_missing_docs()
                ));
            }
            if published.exists() {
                return Err("canonical corrupt segment remained active".to_owned());
            }
            let quarantine = &snapshot.quarantined_segments()[0];
            if quarantine.segment_id != segment_id
                || quarantine.estimated_missing_docs != Some(expected_doc_count)
                || std::fs::read(&quarantine.path).map_err(|error| error.to_string())? != corrupt
            {
                return Err("quarantine witness did not retain the corrupt source".to_owned());
            }
            let sidecar_quarantine = append_path_suffix(&sidecar, ".quarantine");
            if !sidecar_quarantine.is_file() {
                return Err("repair sidecar was not retained beside quarantine".to_owned());
            }
            let retained_bytes = std::fs::metadata(&quarantine.path)
                .map_err(|error| error.to_string())?
                .len()
                .saturating_add(
                    std::fs::metadata(&sidecar_quarantine)
                        .map_err(|error| error.to_string())?
                        .len(),
                );
            if snapshot.segment_stats().managed_disk_bytes < retained_bytes {
                return Err("managed disk accounting omitted quarantine artifacts".to_owned());
            }
            drop(writer);

            let reopened = KeeperSnapshot::open(&directory, DEFAULT_SCHEMA)
                .map_err(|error| error.to_string())?;
            if !reopened.is_degraded()
                || reopened.quarantined_segments().len() != 1
                || reopened.estimated_missing_docs() != expected_doc_count
            {
                return Err("read-only reopen silently lost degraded state".to_owned());
            }
            Ok(())
        });
        outcome.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_skips_stale_temps_and_quarantine_collisions() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        std::fs::write(&destination, b"first corrupt authority")?;
        let base = format!(
            ".tmp-repair-MANIFEST-{:016x}",
            admission.record.pid_start_nonce
        );
        let stale_temp = index.path().join(&base);
        std::fs::write(&stale_temp, b"interrupted prior repair")?;

        install_recovered_bytes(&admission, "MANIFEST", &destination, b"first recovery")?;
        assert_eq!(std::fs::read(&destination)?, b"first recovery");
        assert_eq!(std::fs::read(&stale_temp)?, b"interrupted prior repair");

        std::fs::write(&destination, b"second corrupt authority")?;
        install_recovered_bytes(&admission, "MANIFEST", &destination, b"second recovery")?;
        assert_eq!(std::fs::read(&destination)?, b"second recovery");
        assert_eq!(std::fs::read(&stale_temp)?, b"interrupted prior repair");
        assert_eq!(
            std::fs::read_dir(index.path())?
                .filter_map(Result::ok)
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.starts_with(".tmp-corrupt-MANIFEST-"))
                .count(),
            2
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn portable_install_replaces_occupied_destination_and_quarantines_it() -> TestResult {
        // Tier-2 choreography (bd-b188): occupied destination retires to a
        // probed no-replace quarantine, recovered bytes install atomically,
        // and a second install claims a fresh quarantine without clobbering.
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        std::fs::write(&destination, b"corrupt authority")?;

        install_recovered_bytes_portable(
            &admission,
            "MANIFEST",
            &destination,
            b"first recovery",
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(std::fs::read(&destination)?, b"first recovery");

        std::fs::write(&destination, b"second corrupt authority")?;
        install_recovered_bytes_portable(
            &admission,
            "MANIFEST",
            &destination,
            b"second recovery",
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(std::fs::read(&destination)?, b"second recovery");

        // Both quarantines survive: each install claimed a fresh no-replace
        // name instead of overwriting the prior one.
        let quarantines: Vec<_> = std::fs::read_dir(index.path())?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.starts_with(".retired-repair-MANIFEST-"))
            .collect();
        assert_eq!(quarantines.len(), 2, "expected two probed quarantines");
        let retired: Vec<_> = quarantines
            .iter()
            .map(|name| std::fs::read(index.path().join(name)).expect("read quarantine"))
            .collect();
        assert!(retired.contains(&b"corrupt authority".to_vec()));
        assert!(retired.contains(&b"second corrupt authority".to_vec()));
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn portable_install_into_absent_destination_and_retry_after_interruption() -> TestResult {
        // Absent canonical authority: the temp hard-links straight into
        // place. A stale temp from an interrupted earlier run is probed
        // around, never overwritten (bd-b188 retry-after-interruption).
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        let stale_temp = index.path().join(format!(
            ".tmp-repair-MANIFEST-{:016x}",
            admission.record.pid_start_nonce
        ));
        std::fs::write(&stale_temp, b"interrupted prior repair")?;

        install_recovered_bytes_portable(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(std::fs::read(&destination)?, b"validated recovery");
        assert_eq!(std::fs::read(&stale_temp)?, b"interrupted prior repair");
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn portable_install_reports_checkpoints_in_order() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        std::fs::write(&destination, b"corrupt authority")?;
        let mut checkpoints = Vec::new();
        install_recovered_bytes_portable(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            &mut |checkpoint, _| {
                checkpoints.push(checkpoint);
                Ok(())
            },
        )?;
        assert_eq!(
            checkpoints,
            vec![
                RecoveredByteInstallCheckpoint::TempSynced,
                RecoveredByteInstallCheckpoint::BeforeAtomicInstall,
                RecoveredByteInstallCheckpoint::AfterCorruptRetirement,
                RecoveredByteInstallCheckpoint::AfterAtomicInstall,
                RecoveredByteInstallCheckpoint::FinalSynced,
            ]
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn portable_retire_is_collision_safe_and_never_replaces() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let source = index.path().join("corrupt.bin");
        std::fs::write(&source, b"corrupt payload")?;
        let destination = index.path().join("quarantine.bin");

        // Absent destination: retires onto the base name.
        assert!(retire_regular_artifact_portable(
            &admission,
            &source,
            &destination
        )?);
        assert_eq!(std::fs::read(&destination)?, b"corrupt payload");
        assert!(!source.exists());

        // Occupied destination: the next retry claims a fresh suffixed probe
        // and the occupied quarantine keeps its bytes.
        std::fs::write(&source, b"second corrupt payload")?;
        assert!(retire_regular_artifact_portable(
            &admission,
            &source,
            &destination
        )?);
        assert_eq!(std::fs::read(&destination)?, b"corrupt payload");
        let siblings: Vec<_> = std::fs::read_dir(index.path())?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.starts_with("quarantine.bin."))
            .collect();
        assert_eq!(siblings.len(), 1, "expected one probed sibling quarantine");
        assert_eq!(
            std::fs::read(index.path().join(&siblings[0]))?,
            b"second corrupt payload"
        );

        // Absent source reports no work.
        assert!(!retire_regular_artifact_portable(
            &admission,
            &source,
            &destination
        )?);
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_rolls_back_a_substituted_ready_link() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        let prior = Manifest::empty(1, DEFAULT_SCHEMA.schema_id()?, 0).to_bytes()?;
        std::fs::write(&destination, &prior)?;
        let displaced_ready = index.path().join("displaced-owned-ready-link");

        let error = install_recovered_bytes_with_observer(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            &mut |checkpoint, ready| {
                if checkpoint == RecoveredByteInstallCheckpoint::BeforeAtomicInstall {
                    std::fs::rename(ready, &displaced_ready)?;
                    std::fs::write(ready, b"substituted bytes")?;
                }
                Ok(())
            },
        )
        .expect_err("a substituted ready link must not become authority");
        assert!(matches!(
            error,
            KeeperError::Io {
                operation: "verify installed recovered bytes",
                ..
            }
        ));
        let reopened = std::fs::read(&destination)?;
        assert_eq!(reopened, prior);
        assert_eq!(Manifest::from_bytes(&reopened)?.generation, 1);
        assert_eq!(std::fs::read(&displaced_ready)?, b"validated recovery");
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_ignores_a_substituted_ready_link_when_destination_is_missing()
    -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        let displaced_ready = index.path().join("displaced-owned-ready-link");

        install_recovered_bytes_with_observer(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            &mut |checkpoint, ready| {
                if checkpoint == RecoveredByteInstallCheckpoint::BeforeAtomicInstall {
                    std::fs::rename(ready, &displaced_ready)?;
                    std::fs::write(ready, b"substituted bytes")?;
                }
                Ok(())
            },
        )?;
        assert_eq!(std::fs::read(&destination)?, b"validated recovery");
        assert_eq!(std::fs::read(&displaced_ready)?, b"validated recovery");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_retires_corrupt_primary_and_recovers_from_previous() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;
        std::fs::write(index.path().join("MANIFEST"), b"corrupt primary")?;
        let before_reader = directory_bytes(index.path())?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        assert_eq!(directory_bytes(index.path())?, before_reader);

        let directory = index.path().to_path_buf();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 1);
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        assert!(std::fs::read_dir(index.path())?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.starts_with(".tmp-corrupt-manifest-"))
        }));
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_recovers_both_missing_or_corrupt_manifest_slots() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let previous_path = index.path().join("MANIFEST.prev");
        let expected_current = std::fs::read(&current_path)?;
        let expected_previous = std::fs::read(&previous_path)?;
        let mut corrupt_current = expected_current.clone();
        corrupt_current[16] ^= 0xff;
        let mut corrupt_previous = expected_previous.clone();
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&current_path, corrupt_current)?;
        std::fs::write(&previous_path, corrupt_previous)?;

        let directory = index.path().to_path_buf();
        let corrupt_protector = protector.clone();
        let recovered_corrupt: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, corrupt_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        recovered_corrupt.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&current_path)?, expected_current);
        assert_eq!(std::fs::read(&previous_path)?, expected_previous);

        let saved_current = index.path().join("saved-MANIFEST");
        let saved_previous = index.path().join("saved-MANIFEST.prev");
        std::fs::rename(&current_path, &saved_current)?;
        std::fs::rename(&previous_path, &saved_previous)?;
        let directory = index.path().to_path_buf();
        let missing_protector = protector.clone();
        let recovered_missing: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::create_durable(&cx, directory, DEFAULT_SCHEMA, missing_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        recovered_missing.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&current_path)?, expected_current);
        assert_eq!(std::fs::read(&previous_path)?, expected_previous);
        assert_eq!(std::fs::read(saved_current)?, expected_current);
        assert_eq!(std::fs::read(saved_previous)?, expected_previous);
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_open_rejects_hostile_sidecars_without_exhaustion() -> TestResult {
        // bd-x7l7: hostile .fec sidecars must be typed rejections through the
        // automatic Keeper recovery path — never a memory-exhaustion event.
        // Case A: oversized sidecar. Case B: truncated sidecar. Case C:
        // forged trailer counts. Each index is independent.
        use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileProtector};

        let small_cap_protector = || {
            FileProtector::new(
                Arc::new(DefaultSymbolCodec),
                DurabilityConfig {
                    symbol_size: 256,
                    repair_overhead: 2.0,
                    max_repair_symbols: 64,
                    ..DurabilityConfig::default()
                },
            )
            .expect("small-cap protector")
        };

        // Build one healthy durable index per case.
        let make_index = |label: &str| -> Result<
            (tempfile::TempDir, std::path::PathBuf),
            Box<dyn std::error::Error>,
        > {
            let index = tempdir()?;
            let directory = index.path().to_path_buf();
            let protector = test_file_protector();
            run_with_test_cx(move |cx| async move {
                let mut writer =
                    KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, protector)
                        .await
                        .map_err(|error| io::Error::other(error.to_string()))?;
                writer
                    .publish(&cx, &durable_test_manifest(2, Vec::new()))
                    .await
                    .map_err(|error| io::Error::other(error.to_string()))?;
                Ok::<(), io::Error>(())
            })?;
            let manifest_sidecar = index.path().join(format!("MANIFEST.fec.{label}"));
            Ok((index, manifest_sidecar))
        };

        // Case A: oversized sidecar is rejected at the stat bound.
        let (index, _unused) = make_index("oversized")?;
        // Oversize BOTH manifest sidecars so every recovery path hits the
        // bounded-read rejection (a valid MANIFEST.prev.fec would otherwise
        // recover the previous slot and open successfully).
        for sidecar in ["MANIFEST.fec", "MANIFEST.prev.fec"] {
            std::fs::write(index.path().join(sidecar), vec![0xaa_u8; 64 * 1024])?;
        }
        // Corrupt both manifest slots so the automatic recovery path must
        // engage the sidecar read.
        for slot in ["MANIFEST", "MANIFEST.prev"] {
            let slot_path = index.path().join(slot);
            let mut bytes = std::fs::read(&slot_path)?;
            bytes[16] ^= 0xff;
            std::fs::write(&slot_path, &bytes)?;
        }
        let directory = index.path().to_path_buf();
        let protector = small_cap_protector();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            match KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, protector).await {
                Ok(_) => Err("oversized sidecar must not open".to_owned()),
                Err(error) => {
                    let text = error.to_string();
                    assert!(
                        text.contains("exceeding"),
                        "rejection must name the bound: {text}"
                    );
                    Ok(())
                }
            }
        });
        outcome.map_err(io::Error::other)?;

        // Case B: truncated sidecar is a typed corruption error.
        let (index, _unused) = make_index("truncated")?;
        for sidecar in ["MANIFEST.fec", "MANIFEST.prev.fec"] {
            let sidecar_path = index.path().join(sidecar);
            let mut trailer = std::fs::read(&sidecar_path)?;
            trailer.truncate(trailer.len() / 2);
            std::fs::write(&sidecar_path, &trailer)?;
        }
        // Also corrupt both manifest slots so recovery actually engages.
        for slot in ["MANIFEST", "MANIFEST.prev"] {
            let slot_path = index.path().join(slot);
            let mut bytes = std::fs::read(&slot_path)?;
            bytes[16] ^= 0xff;
            std::fs::write(&slot_path, &bytes)?;
        }
        let directory = index.path().to_path_buf();
        let protector = small_cap_protector();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            match KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, protector).await {
                Ok(_) => Err("truncated sidecar must not silently succeed".to_owned()),
                Err(_) => Ok(()),
            }
        });
        outcome.map_err(io::Error::other)?;

        // Case C: forged trailer counts are rejected before allocation.
        let (index, _unused) = make_index("forged")?;
        for sidecar in ["MANIFEST.fec", "MANIFEST.prev.fec"] {
            let sidecar_path = index.path().join(sidecar);
            let mut trailer = std::fs::read(&sidecar_path)?;
            // V2 layout: repair_symbol_count at offset 34; then fix the trailer CRC.
            trailer[34..38].copy_from_slice(&1_000_000_u32.to_le_bytes());
            let crc = crc32fast::hash(&trailer[..trailer.len() - 4]);
            let end = trailer.len();
            trailer[end - 4..].copy_from_slice(&crc.to_le_bytes());
            std::fs::write(&sidecar_path, &trailer)?;
        }
        for slot in ["MANIFEST", "MANIFEST.prev"] {
            let slot_path = index.path().join(slot);
            let mut bytes = std::fs::read(&slot_path)?;
            bytes[16] ^= 0xff;
            std::fs::write(&slot_path, &bytes)?;
        }
        let directory = index.path().to_path_buf();
        let protector = small_cap_protector();
        let outcome: Result<(), String> = run_with_test_cx(move |cx| async move {
            match KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, protector).await {
                Ok(_) => Err("forged counts must not silently succeed".to_owned()),
                Err(_) => Ok(()),
            }
        });
        outcome.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn unrecoverable_reconstructed_current_preserves_previous_authority() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let encoded = encoded_test_segment(0xfeed, 0, 2, 1)?;
        let manifest_segment = manifest_segment(&encoded, 1);
        let setup_manifest_segment = manifest_segment.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest_segment]))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let segment_path = index
            .path()
            .join(canonical_segment_name(manifest_segment.segment_id));
        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, &corrupt_current)?;
        let mut corrupt_segment = std::fs::read(&segment_path)?;
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;
        let segment_sidecar = FileProtector::sidecar_path(&segment_path);
        let saved_sidecar = index.path().join("saved-unrecoverable-segment.fec");
        std::fs::rename(&segment_sidecar, &saved_sidecar)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("unrecoverable reconstructed current displaced previous".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert!(!current_path.exists());
        let retired_current = std::fs::read_dir(index.path())?
            .filter_map(Result::ok)
            .find(|entry| {
                entry.file_name().into_string().is_ok_and(|name| {
                    name.starts_with(".tmp-corrupt-manifest-")
                        && !Path::new(&name)
                            .extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("fec"))
                })
            })
            .ok_or_else(|| io::Error::other("corrupt current was not quarantined"))?;
        assert_eq!(std::fs::read(retired_current.path())?, corrupt_current);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1,
            "the usable previous generation must remain reader-visible"
        );
        assert!(saved_sidecar.is_file());
        assert!(!segment_sidecar.exists());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn unusable_reconstructed_fallback_does_not_block_healthy_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let encoded = encoded_test_segment(0xfa11, 0, 2, 1)?;
        let manifest_segment = manifest_segment(&encoded, 1);
        let setup_manifest_segment = manifest_segment.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest_segment]))
                .await
                .map_err(|error| error.to_string())?;
            let mut current = durable_test_manifest(3, Vec::new());
            current.docid_high_watermark = 2;
            writer
                .publish(&cx, &current)
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;
        let segment_path = index
            .path()
            .join(canonical_segment_name(manifest_segment.segment_id));
        let mut corrupt_segment = std::fs::read(&segment_path)?;
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;
        let segment_sidecar = FileProtector::sidecar_path(&segment_path);
        let saved_sidecar = index.path().join("saved-fallback-segment.fec");
        std::fs::rename(&segment_sidecar, &saved_sidecar)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 3 {
                return Err("unusable optional fallback displaced healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            3
        );
        assert!(saved_sidecar.is_file());
        assert!(!segment_sidecar.exists());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn stale_reconstructed_fallback_does_not_block_healthy_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let stale = Manifest::empty(50, DEFAULT_SCHEMA.schema_id()?, 0).to_bytes()?;
        std::fs::write(&previous_path, &stale)?;
        protector.protect_file(&previous_path)?;
        let mut corrupt_stale = stale;
        corrupt_stale[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_stale)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 2 {
                return Err("stale optional fallback displaced healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_stale);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn mismatched_dual_reconstructed_manifests_fail_closed() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let previous_path = index.path().join("MANIFEST.prev");
        let current_sidecar = FileProtector::sidecar_path(&current_path);
        let saved_current_sidecar = index.path().join("saved-current-generation-two.fec");
        std::fs::rename(&current_sidecar, &saved_current_sidecar)?;

        let synthetic_current = index.path().join(".tmp-synthetic-current-five");
        let synthetic_bytes = durable_test_manifest(5, Vec::new()).to_bytes()?;
        std::fs::write(&synthetic_current, &synthetic_bytes)?;
        protector.protect_file_with_witness(
            &synthetic_current,
            FileSourceWitness::from_bytes(&synthetic_bytes),
        )?;
        std::fs::rename(
            FileProtector::sidecar_path(&synthetic_current),
            &current_sidecar,
        )?;

        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, &corrupt_current)?;
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;

        let directory = index.path().to_path_buf();
        let open_result: Result<KeeperWriter, KeeperError> =
            run_with_test_cx(move |cx| async move {
                KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, protector).await
            });
        assert!(matches!(
            open_result,
            Err(KeeperError::InvalidGenerationPair {
                current: 5,
                previous: 1,
                ..
            })
        ));
        assert_eq!(std::fs::read(&current_path)?, corrupt_current);
        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert!(saved_current_sidecar.is_file());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn malformed_current_sidecar_falls_back_to_usable_previous() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, corrupt_current)?;
        std::fs::write(
            FileProtector::sidecar_path(&current_path),
            b"malformed current sidecar",
        )?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("malformed current sidecar blocked previous fallback".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn malformed_previous_sidecar_does_not_block_usable_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;
        std::fs::write(
            FileProtector::sidecar_path(&previous_path),
            b"malformed previous sidecar",
        )?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 2 {
                return Err("malformed optional fallback blocked healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_stages_and_validates_manifest_and_segment_repairs() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let protector = test_file_protector();
        let setup_protector = protector.clone();
        let expected = encoded_test_segment(0xabc, 0, 2, 1)?;
        let expected_segment = expected.as_bytes().to_vec();
        let segment_manifest = manifest_segment(&expected, 1);
        let setup_manifest = segment_manifest.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = expected
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest]))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let manifest_path = index.path().join("MANIFEST");
        let expected_manifest = std::fs::read(&manifest_path)?;
        let segment_path = index
            .path()
            .join(canonical_segment_name(segment_manifest.segment_id));
        let mut corrupt_manifest = expected_manifest.clone();
        corrupt_manifest[16] ^= 0xff;
        std::fs::write(&manifest_path, corrupt_manifest)?;
        let mut corrupt_segment = expected_segment.clone();
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let repaired: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        repaired.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, expected_manifest);
        assert_eq!(std::fs::read(&segment_path)?, expected_segment);
        assert!(
            protector
                .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))?
                .healthy
        );
        assert!(
            protector
                .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))?
                .healthy
        );

        let authoritative_manifest = std::fs::read(&manifest_path)?;
        std::fs::write(
            FileProtector::sidecar_path(&manifest_path),
            b"malformed stale sidecar",
        )?;
        let directory = index.path().to_path_buf();
        let stale_protector = protector.clone();
        let regenerated: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, stale_protector)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        regenerated.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, authoritative_manifest);
        assert!(
            protector
                .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))?
                .healthy
        );

        std::fs::write(
            FileProtector::sidecar_path(&manifest_path),
            b"second malformed stale sidecar",
        )?;
        let directory = index.path().to_path_buf();
        let repeated_protector = protector.clone();
        let repeated: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, repeated_protector)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        repeated.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, authoritative_manifest);
        assert_eq!(
            std::fs::read_dir(index.path())?
                .filter_map(Result::ok)
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.starts_with(".tmp-stale-fec-MANIFEST-"))
                .count(),
            2,
            "each stale sidecar must retain a distinct no-replace quarantine"
        );
        Ok(())
    }

    fn hex_bytes(source: &str) -> Vec<u8> {
        let compact = source
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(compact.len() % 2, 0, "hex input must have even length");
        let (pairs, remainder) = compact.as_chunks::<2>();
        assert!(remainder.is_empty());
        pairs
            .iter()
            .map(|&[high, low]| {
                let high = hex_nibble(high);
                let low = hex_nibble(low);
                high << 4 | low
            })
            .collect()
    }

    fn hex_nibble(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => panic!("invalid hex byte"),
        }
    }

    // ==== Blue-green CURRENT pointer (bd-quill-duel-blue-green-vwf7) ====

    fn quill_v1_pointer() -> CurrentPointer {
        CurrentPointer::new(
            BlueGreenEngine::Quill,
            "quill-v1",
            crate::segment::FSLX_FORMAT_VERSION,
        )
        .expect("valid quill-v1 pointer")
    }

    #[test]
    fn current_pointer_codec_roundtrips_both_engines_and_golden_bytes() {
        for (engine, dir_name, version) in [
            (
                BlueGreenEngine::Quill,
                "quill-v1",
                crate::segment::FSLX_FORMAT_VERSION,
            ),
            (
                BlueGreenEngine::Tantivy,
                "tantivy",
                TANTIVY_INDEX_FORMAT_VERSION,
            ),
            (BlueGreenEngine::Quill, "q", 0),
            (BlueGreenEngine::Quill, &"x".repeat(255), u32::MAX),
            (BlueGreenEngine::Tantivy, &"y".repeat(65_535), 7),
        ] {
            let pointer = CurrentPointer::new(engine, dir_name, version).expect("valid pointer");
            let encoded = pointer.encode();
            assert_eq!(encoded.len(), CURRENT_FIXED_BYTES + dir_name.len());
            let decoded = CurrentPointer::decode(&encoded).expect("roundtrip decode");
            assert_eq!(decoded, pointer);
        }

        // Golden image pins the exact registry §7.3 wire layout.
        let encoded = quill_v1_pointer().encode();
        let expected_body: &[u8] = b"FSLXCUR\0";
        assert_eq!(&encoded[..8], expected_body);
        assert_eq!(&encoded[8..12], &1_u32.to_le_bytes());
        assert_eq!(encoded[12], 1);
        assert_eq!(&encoded[13..15], &8_u16.to_le_bytes());
        assert_eq!(&encoded[15..23], b"quill-v1");
        assert_eq!(
            &encoded[23..27],
            &crate::segment::FSLX_FORMAT_VERSION.to_le_bytes()
        );
        let crc = u32::from_le_bytes(encoded[27..31].try_into().expect("crc bytes"));
        assert_eq!(crc, crc32fast::hash(&encoded[..27]));
        assert_eq!(encoded.len(), 31);
    }

    #[test]
    fn current_pointer_decode_fails_closed_on_every_corruption_class() {
        let valid = quill_v1_pointer().encode();

        // Truncation at every byte boundary: typed error, never a panic.
        for cut in 0..valid.len() {
            assert!(
                CurrentPointer::decode(&valid[..cut]).is_err(),
                "truncation at {cut} must fail"
            );
        }
        // Trailing bytes.
        let mut trailed = valid.clone();
        trailed.push(0);
        assert!(matches!(
            CurrentPointer::decode(&trailed),
            Err(CurrentPointerError::CrcMismatch { .. }
                | CurrentPointerError::LengthMismatch { .. })
        ));
        // Single-byte flips across every byte class must fail (CRC or a typed
        // structural rejection), never silently decode or panic.
        for offset in 0..valid.len() {
            let mut flipped = valid.clone();
            flipped[offset] ^= 0x5A;
            assert!(
                CurrentPointer::decode(&flipped).is_err(),
                "flip at {offset} must fail"
            );
        }
        // Bad magic with a repaired CRC reports the magic, not the checksum.
        let mut bad_magic = valid.clone();
        bad_magic[..8].copy_from_slice(b"FSLXNOPE");
        rewrite_crc(&mut bad_magic);
        assert!(matches!(
            CurrentPointer::decode(&bad_magic),
            Err(CurrentPointerError::BadMagic)
        ));
        // Unsupported format version with a repaired CRC.
        let mut future = valid.clone();
        future[8..12].copy_from_slice(&2_u32.to_le_bytes());
        rewrite_crc(&mut future);
        assert!(matches!(
            CurrentPointer::decode(&future),
            Err(CurrentPointerError::UnsupportedFormatVersion(2))
        ));
        // Unknown engine kind with a repaired CRC.
        let mut alien = valid.clone();
        alien[12] = 9;
        rewrite_crc(&mut alien);
        assert!(matches!(
            CurrentPointer::decode(&alien),
            Err(CurrentPointerError::UnknownEngineKind(9))
        ));
        // Non-UTF-8 directory name with a repaired CRC.
        let mut non_utf8 = valid.clone();
        non_utf8[15] = 0xFF;
        rewrite_crc(&mut non_utf8);
        assert!(matches!(
            CurrentPointer::decode(&non_utf8),
            Err(CurrentPointerError::DirNameNotUtf8)
        ));
        // Path-unsafe names with repaired CRCs.
        for hostile in ["..", "a/b", "/abs", "a\\b", "a\0b", ""] {
            assert!(CurrentPointer::new(BlueGreenEngine::Quill, hostile, 1).is_err());
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&CURRENT_MAGIC);
            put_u32(&mut bytes, CURRENT_FORMAT_VERSION);
            bytes.push(1);
            put_u16(
                &mut bytes,
                u16::try_from(hostile.len()).expect("hostile name fits u16"),
            );
            bytes.extend_from_slice(hostile.as_bytes());
            put_u32(&mut bytes, 1);
            let crc = crc32fast::hash(&bytes);
            put_u32(&mut bytes, crc);
            assert!(
                CurrentPointer::decode(&bytes).is_err(),
                "hostile name {hostile:?} must fail"
            );
        }
    }

    #[test]
    fn publish_current_swaps_flip_and_rollback_leave_stale_temps_inert() -> TestResult {
        let root = tempdir()?;
        let quill_dir = root.path().join("quill-v1");
        let tantivy_dir = root.path().join("tantivy");
        std::fs::create_dir(&quill_dir)?;
        std::fs::create_dir(&tantivy_dir)?;

        // A stale crash witness from an interrupted earlier attempt is ignored
        // by resolution and never disturbed by publication.
        let stale_temp = root.path().join(".tmp-current-1-0");
        std::fs::write(&stale_temp, b"abandoned crash witness")?;

        let quill_pointer = quill_v1_pointer();
        publish_current(root.path(), &quill_pointer)?;
        assert_eq!(
            resolve_current(root.path())?,
            ResolvedCurrent::Pointer(quill_pointer.clone())
        );
        assert_eq!(std::fs::read(&stale_temp)?, b"abandoned crash witness");

        // Flip: quill -> tantivy in one pointer swap; old directory retained.
        let tantivy_pointer = CurrentPointer::new(
            BlueGreenEngine::Tantivy,
            "tantivy",
            TANTIVY_INDEX_FORMAT_VERSION,
        )?;
        publish_current(root.path(), &tantivy_pointer)?;
        assert_eq!(
            resolve_current(root.path())?,
            ResolvedCurrent::Pointer(tantivy_pointer.clone())
        );
        assert!(quill_dir.is_dir(), "retired quill directory is retained");

        // Rollback is the identical swap in reverse, zero rebuild.
        publish_current(root.path(), &quill_pointer)?;
        let resolved = resolve_current(root.path())?;
        assert_eq!(resolved, ResolvedCurrent::Pointer(quill_pointer.clone()));
        assert_eq!(
            resolved.pointer().expect("pointer").engine_dir(root.path()),
            quill_dir
        );
        assert!(
            tantivy_dir.is_dir(),
            "retired tantivy directory is retained"
        );
        Ok(())
    }

    #[test]
    fn resolve_current_adopts_single_engine_dir_and_bootstraps_pointer() -> TestResult {
        // Quill directory adopted via its MANIFEST marker.
        let quill_root = tempdir()?;
        let quill_dir = quill_root.path().join("quill-v1");
        std::fs::create_dir(&quill_dir)?;
        std::fs::write(quill_dir.join("MANIFEST"), b"genesis placeholder")?;
        let adopted = resolve_current(quill_root.path())?;
        let expected = CurrentPointer::new(
            BlueGreenEngine::Quill,
            "quill-v1",
            crate::segment::FSLX_FORMAT_VERSION,
        )?;
        assert_eq!(adopted, ResolvedCurrent::Adopted(expected.clone()));
        // Adoption wrote a durable CURRENT; the next resolve reads it.
        assert_eq!(
            resolve_current(quill_root.path())?,
            ResolvedCurrent::Pointer(expected)
        );
        assert_eq!(
            CurrentPointer::decode(&std::fs::read(quill_root.path().join(CURRENT_FILE_NAME))?)?,
            quill_v1_pointer()
        );

        // Tantivy directory adopted via its meta.json marker.
        let tantivy_root = tempdir()?;
        let tantivy_dir = tantivy_root.path().join("tantivy");
        std::fs::create_dir(&tantivy_dir)?;
        std::fs::write(tantivy_dir.join("meta.json"), b"{}")?;
        assert_eq!(
            resolve_current(tantivy_root.path())?,
            ResolvedCurrent::Adopted(CurrentPointer::new(
                BlueGreenEngine::Tantivy,
                "tantivy",
                TANTIVY_INDEX_FORMAT_VERSION,
            )?)
        );

        // A directory bearing both markers adopts as quill (forward format wins).
        let mixed_root = tempdir()?;
        let mixed_dir = mixed_root.path().join("engine");
        std::fs::create_dir(&mixed_dir)?;
        std::fs::write(mixed_dir.join("MANIFEST"), b"x")?;
        std::fs::write(mixed_dir.join("meta.json"), b"{}")?;
        let resolved = resolve_current(mixed_root.path())?;
        assert_eq!(
            resolved.pointer().expect("adopted pointer").engine(),
            BlueGreenEngine::Quill
        );
        Ok(())
    }

    #[test]
    fn resolve_current_reports_empty_ambiguous_and_missing_closed() -> TestResult {
        // Fresh root: Empty, and nothing is created.
        let empty_root = tempdir()?;
        assert_eq!(resolve_current(empty_root.path())?, ResolvedCurrent::Empty);
        assert!(std::fs::read_dir(empty_root.path())?.next().is_none());

        // Multiple engine directories and no CURRENT: fail closed for doctor.
        let ambiguous_root = tempdir()?;
        for name in ["quill-v1", "tantivy"] {
            let dir = ambiguous_root.path().join(name);
            std::fs::create_dir(&dir)?;
            std::fs::write(dir.join("MANIFEST"), b"x")?;
        }
        let error = resolve_current(ambiguous_root.path()).expect_err("must demand doctor");
        match error {
            CurrentPointerError::AmbiguousEngineDirs { candidates, .. } => {
                assert_eq!(
                    candidates,
                    vec!["quill-v1".to_owned(), "tantivy".to_owned()]
                );
            }
            other => panic!("expected AmbiguousEngineDirs, got {other:?}"),
        }
        assert!(
            !ambiguous_root.path().join(CURRENT_FILE_NAME).exists(),
            "ambiguous resolution must not publish a guess"
        );

        // CURRENT names a directory that does not exist: fail closed.
        let missing_root = tempdir()?;
        publish_current(
            missing_root.path(),
            &CurrentPointer::new(BlueGreenEngine::Quill, "quill-v9", 1)?,
        )?;
        assert!(matches!(
            resolve_current(missing_root.path()),
            Err(CurrentPointerError::MissingEngineDir { .. })
        ));

        // Corrupt CURRENT bytes: typed decode failure, never a guess.
        let corrupt_root = tempdir()?;
        std::fs::write(corrupt_root.path().join(CURRENT_FILE_NAME), b"garbage")?;
        assert!(resolve_current(corrupt_root.path()).is_err());
        Ok(())
    }

    #[test]
    fn resolve_current_ignores_user_files_and_tmp_witnesses() -> TestResult {
        let root = tempdir()?;
        std::fs::write(root.path().join("notes.txt"), b"user data")?;
        std::fs::write(root.path().join(".tmp-current-9-3"), b"crash witness")?;
        let random_dir = root.path().join("random");
        std::fs::create_dir(&random_dir)?;
        std::fs::write(random_dir.join("data.bin"), b"not an engine")?;
        assert_eq!(resolve_current(root.path())?, ResolvedCurrent::Empty);
        // Nothing was adopted, published, or modified.
        assert!(!root.path().join(CURRENT_FILE_NAME).exists());
        assert_eq!(std::fs::read(root.path().join("notes.txt"))?, b"user data");
        assert_eq!(
            std::fs::read(root.path().join(".tmp-current-9-3"))?,
            b"crash witness"
        );
        Ok(())
    }

    // ==== Visibility contract (bd-quill-duel-visibility-contract-9rk3) ====

    /// Build a v1 MANIFEST image by removing the v2 timestamp word from a v2
    /// image and re-versioning/re-checksumming it.
    fn downgrade_to_v1_image(v2: &[u8]) -> Vec<u8> {
        // v2 layout: magic(8) version(4) generation(8) watermark(8) schema(8)
        // engine(4) flags(4) timestamp(8) rest.. crc(4)
        assert_eq!(&v2[8..12], &MANIFEST_FORMAT_VERSION.to_le_bytes());
        let mut v1 = Vec::with_capacity(v2.len() - 8);
        v1.extend_from_slice(&v2[..8]);
        v1.extend_from_slice(&MANIFEST_FORMAT_VERSION_V1.to_le_bytes());
        v1.extend_from_slice(&v2[12..44]); // generation..flags
        v1.extend_from_slice(&v2[52..]); // segments/stats..crc
        let body_len = v1.len() - 4;
        let crc = crc32fast::hash(&v1[..body_len]);
        v1[body_len..].copy_from_slice(&crc.to_le_bytes());
        v1
    }

    #[test]
    fn manifest_v1_images_read_with_unknown_timestamp_and_rewrite_as_v2() -> TestResult {
        let v2 = sample_manifest(5).to_bytes()?;
        let v1 = downgrade_to_v1_image(&v2);
        let decoded = Manifest::from_bytes(&v1)?;
        assert_eq!(decoded.last_publish_unix_s, 0, "v1 images carry no witness");
        let mut expected = sample_manifest(5);
        expected.last_publish_unix_s = 0;
        assert_eq!(decoded, expected);
        // Rewriting upgrades the image to v2 while preserving the zero field.
        let rewritten = decoded.to_bytes()?;
        assert_eq!(&rewritten[8..12], &MANIFEST_FORMAT_VERSION.to_le_bytes());
        assert_eq!(Manifest::from_bytes(&rewritten)?, decoded);
        // A v1 image with hostile trailing content still fails closed.
        let mut trailed = v1.clone();
        trailed.push(0);
        assert!(Manifest::from_bytes(&trailed).is_err());
        Ok(())
    }

    #[test]
    fn publisher_stamps_zero_timestamp_and_preserves_explicit_witnesses() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let mut first = sample_manifest(1);
            first.last_publish_unix_s = 0;
            let before = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("wall clock")
                .as_secs();
            let loaded = publisher
                .publish(&cx, &first)
                .await
                .expect("publish zero-timestamp proposal");
            let stamped = loaded.manifest.last_publish_unix_s;
            assert!(
                u64::try_from(stamped).expect("stamped witness is positive")
                    >= before.saturating_sub(2),
                "zero witness is stamped with the publish wall clock"
            );

            // An explicit witness is preserved verbatim (deterministic fixtures).
            let mut second = sample_manifest(2);
            second.last_publish_unix_s = 1_700_000_042;
            let loaded = publisher
                .publish(&cx, &second)
                .await
                .expect("publish explicit-timestamp proposal");
            assert_eq!(loaded.manifest.last_publish_unix_s, 1_700_000_042);

            // Successors built from a loaded manifest keep the stamped chain.
            let mut third = loaded.manifest.clone();
            third.generation = 3;
            let loaded = publisher
                .publish(&cx, &third)
                .await
                .expect("publish successor built from loaded manifest");
            assert_eq!(loaded.manifest.last_publish_unix_s, 1_700_000_042);
        });
    }

    #[test]
    fn segment_stats_surfaces_freshness_fields_and_live_writer() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let writer = KeeperWriter::create(&cx, directory.path(), DEFAULT_SCHEMA)
                .await
                .expect("create index");
            let writer_stats = SegmentStatsProvider::segment_stats(&writer);
            assert_eq!(writer_stats.published_generation, 1);
            assert_eq!(writer_stats.sealed_segments, 0);
            assert_eq!(writer_stats.live_docs, 0);
            assert!(
                writer_stats.last_publish_unix.is_some(),
                "genesis is stamped"
            );
            assert!(
                writer_stats.live_writer,
                "writer holds LOCK by construction"
            );
            assert!(
                writer_stats.managed_disk_bytes > 0,
                "MANIFEST and LOCK are managed bytes"
            );
            assert_eq!(writer_stats.delta_segments, 0);

            // A concurrent read-only snapshot sees the live writer through the
            // D1 LOCK record (same pid, demonstrably alive).
            let snapshot =
                KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("open snapshot");
            let snapshot_stats = SegmentStatsProvider::segment_stats(&snapshot);
            assert_eq!(snapshot_stats.published_generation, 1);
            assert!(snapshot_stats.last_publish_unix.is_some());
            assert!(snapshot_stats.live_writer, "LOCK record names a live pid");

            // After the writer releases admission, the truncated LOCK record
            // reports no live writer.
            drop(writer);
            let snapshot =
                KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA).expect("reopen snapshot");
            let released = SegmentStatsProvider::segment_stats(&snapshot);
            assert!(
                !released.live_writer,
                "released LOCK reports no live writer"
            );
        });
    }

    #[test]
    fn visibility_barrier_truth_table_matches_the_contract() {
        let base = SegmentStats {
            last_publish_unix: Some(1_700_000_000),
            ..SegmentStats::default()
        };
        let one_second = Duration::from_millis(1_000);
        // No pending changes: never due, even with no witness at all.
        assert!(!base.visibility_barrier_due(one_second, false, 1_700_100_000));
        assert!(!SegmentStats::default().visibility_barrier_due(one_second, false, 1_700_100_000));
        // Pending changes with no durable witness: immediately due.
        assert!(SegmentStats::default().visibility_barrier_due(one_second, true, 1_700_100_000));
        // Within the bound: not due; at or beyond: due.
        assert!(!base.visibility_barrier_due(one_second, true, 1_700_000_000));
        assert!(base.visibility_barrier_due(one_second, true, 1_700_000_001));
        assert!(base.visibility_barrier_due(Duration::from_millis(500), true, 1_700_000_001));
        assert!(base.visibility_barrier_due(one_second, true, 1_700_000_100));
        // Clock skew reads as zero lag, never as a negative duration.
        assert!(!base.visibility_barrier_due(one_second, true, 1_699_999_000));
    }

    // ==== Recovered-byte install crash matrix (bd-qxce) ====

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_crash_matrix_recovers_idempotently() -> TestResult {
        use RecoveredByteInstallCheckpoint as Cp;

        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        let prior = Manifest::empty(1, schema_id, 0).to_bytes()?;
        let recovered = Manifest::empty(2, schema_id, 0).to_bytes()?;

        #[derive(Debug)]
        struct Row {
            fault: RecoveredByteInstallCheckpoint,
            destination_present: bool,
            substitute_at_install: bool,
        }
        let mut rows = Vec::new();
        for destination_present in [true, false] {
            for fault in [
                Cp::TempSynced,
                Cp::StagingSynced,
                Cp::BeforeAtomicInstall,
                Cp::AfterAtomicInstall,
                Cp::FinalSynced,
            ] {
                rows.push(Row {
                    fault,
                    destination_present,
                    substitute_at_install: false,
                });
            }
        }
        rows.push(Row {
            fault: Cp::AfterCorruptRetirement,
            destination_present: true,
            substitute_at_install: false,
        });
        rows.push(Row {
            fault: Cp::AfterRollback,
            destination_present: true,
            substitute_at_install: true,
        });
        rows.push(Row {
            fault: Cp::AfterRollbackSync,
            destination_present: true,
            substitute_at_install: true,
        });

        for row in rows {
            let context = format!("{row:?}");
            let index = tempdir()?;
            let admission = acquire_writer_admission(index.path())?;
            let destination = index.path().join("MANIFEST");
            if row.destination_present {
                std::fs::write(&destination, &prior)?;
            }
            let displaced_ready = index.path().join("displaced-ready");
            let error = install_recovered_bytes_with_observer(
                &admission,
                "MANIFEST",
                &destination,
                &recovered,
                &mut |checkpoint, ready| {
                    if row.substitute_at_install && checkpoint == Cp::BeforeAtomicInstall {
                        std::fs::rename(ready, &displaced_ready)?;
                        std::fs::write(ready, b"hostile substitution")?;
                    }
                    if checkpoint == row.fault {
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "injected deterministic crash",
                        ));
                    }
                    Ok(())
                },
            )
            .expect_err(&context);
            assert!(
                matches!(error, KeeperError::Io { .. }),
                "{context}: fault surfaces as typed I/O"
            );

            // Authority after the crash: pre-install faults keep the prior
            // authority (or absence); install faults have already exchanged;
            // rollback faults restored the prior authority.
            let expect_recovered = matches!(
                row.fault,
                Cp::AfterAtomicInstall | Cp::AfterCorruptRetirement | Cp::FinalSynced
            );
            let canonical = std::fs::read(&destination).ok();
            if expect_recovered {
                assert_eq!(
                    canonical.as_deref(),
                    Some(recovered.as_slice()),
                    "{context}: exchanged authority is the recovered generation"
                );
            } else if row.destination_present {
                assert_eq!(
                    canonical.as_deref(),
                    Some(prior.as_slice()),
                    "{context}: prior authority survives"
                );
            } else {
                assert!(canonical.is_none(), "{context}: no authority was installed");
            }

            // Reopen #1: a completed retry converges to the recovered
            // authority regardless of the crash debris left behind.
            drop(admission);
            let admission = acquire_writer_admission(index.path())?;
            install_recovered_bytes(&admission, "MANIFEST", &destination, &recovered)
                .unwrap_or_else(|error| panic!("{context}: retry after crash succeeds: {error}"));
            assert_eq!(
                std::fs::read(&destination)?,
                recovered,
                "{context}: retry installs the recovered generation"
            );
            // Reopen #2: byte-identical authority, proving idempotence.
            drop(admission);
            let admission = acquire_writer_admission(index.path())?;
            assert_eq!(std::fs::read(&destination)?, recovered, "{context}: stable");

            // Grace-expired GC removes every repair orphan without changing
            // canonical bytes; a second sweep is empty.
            let report = collect_writer_garbage_under_lock(
                index.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions {
                    grace_period: Duration::ZERO,
                },
            )?;
            assert_eq!(
                std::fs::read(&destination)?,
                recovered,
                "{context}: GC preserved authority"
            );
            let second = collect_writer_garbage_under_lock(
                index.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions {
                    grace_period: Duration::ZERO,
                },
            )?;
            assert!(
                second.removed.len() <= report.removed.len(),
                "{context}: GC is idempotent"
            );
            assert!(
                second.removed.is_empty(),
                "{context}: second sweep finds nothing: {:?}",
                second.removed
            );
            drop(admission);
        }
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn interrupted_corrupt_retirement_orphan_sidecar_is_retired_once() -> TestResult {
        let index = tempdir()?;
        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        // State: MANIFEST.prev valid (gen 1), MANIFEST deliberately retired
        // (quarantine evidence), orphan MANIFEST.fec left by the interrupted
        // second rename. The sidecar is REAL: produced by the durability
        // protector over the gen-2 bytes, never a mock.
        let previous = Manifest::empty(1, schema_id, 0).to_bytes()?;
        std::fs::write(index.path().join("MANIFEST.prev"), &previous)?;
        let orphan_source = index.path().join("MANIFEST");
        let orphan_bytes = Manifest::empty(2, schema_id, 0).to_bytes()?;
        std::fs::write(&orphan_source, &orphan_bytes)?;
        let protector = test_file_protector();
        protector
            .protect_file_with_witness(&orphan_source, FileSourceWitness::from_bytes(&orphan_bytes))
            .map_err(|error| io::Error::other(error.to_string()))?;
        let orphan_sidecar = FileProtector::sidecar_path(&orphan_source);
        assert!(orphan_sidecar.exists(), "real FEC sidecar was produced");
        std::fs::remove_file(&orphan_source)?; // the interrupted first rename
        std::fs::write(
            index.path().join(".tmp-corrupt-manifest-0000000000000001"),
            b"deliberately retired corrupt authority",
        )?;

        let admission = acquire_writer_admission(index.path())?;
        recover_writer_directory(&admission, DEFAULT_SCHEMA, &WriterProtection::Disabled)?;
        assert!(
            !orphan_sidecar.exists(),
            "orphan sidecar is retired once, deterministically"
        );
        let quarantines = std::fs::read_dir(index.path())?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.starts_with(".tmp-corrupt-manifest-fec-"))
            .count();
        assert_eq!(quarantines, 1, "orphan moved to no-replace quarantine");
        assert_eq!(
            std::fs::read(index.path().join("MANIFEST.prev"))?,
            previous,
            "previous authority is untouched"
        );
        assert!(
            !index.path().join("MANIFEST").exists(),
            "retired authority is not resurrected"
        );

        // A second recovery performs no further work: reopen and compare the
        // whole directory byte-for-byte.
        let before = directory_bytes(index.path())?;
        recover_writer_directory(&admission, DEFAULT_SCHEMA, &WriterProtection::Disabled)?;
        assert_eq!(
            directory_bytes(index.path())?,
            before,
            "recovery is idempotent"
        );
        drop(admission);

        // Negative row: the same orphan sidecar WITHOUT retirement evidence is
        // a legitimate crash survivor and stays put.
        let survivor = tempdir()?;
        std::fs::write(
            survivor.path().join("MANIFEST.prev"),
            &Manifest::empty(1, schema_id, 0).to_bytes()?,
        )?;
        let survivor_source = survivor.path().join("MANIFEST");
        std::fs::write(&survivor_source, &orphan_bytes)?;
        protector
            .protect_file_with_witness(
                &survivor_source,
                FileSourceWitness::from_bytes(&orphan_bytes),
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        std::fs::remove_file(&survivor_source)?;
        let admission = acquire_writer_admission(survivor.path())?;
        recover_writer_directory(&admission, DEFAULT_SCHEMA, &WriterProtection::Disabled)?;
        assert!(
            FileProtector::sidecar_path(&survivor_source).exists(),
            "no evidence: crash-survivor sidecar is preserved"
        );
        drop(admission);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn grace_expired_gc_removes_every_repair_orphan_without_touching_authority() -> TestResult {
        let index = tempdir()?;
        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        let canonical = Manifest::empty(1, schema_id, 0).to_bytes()?;
        std::fs::write(index.path().join("MANIFEST"), &canonical)?;
        let orphans = [
            ".tmp-repair-MANIFEST-00000000000000aa",
            ".tmp-repair-MANIFEST-00000000000000aa.1",
            ".tmp-ready-repair-MANIFEST-00000000000000aa",
            ".tmp-corrupt-MANIFEST-00000000000000aa",
            ".tmp-corrupt-manifest-fec-00000000000000aa",
            ".tmp-corrupt-manifest-00000000000000aa",
        ];
        for (offset, name) in orphans.iter().enumerate() {
            std::fs::write(index.path().join(name), format!("orphan {offset}"))?;
        }
        let report = collect_writer_garbage_under_lock(
            index.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions {
                grace_period: Duration::ZERO,
            },
        )?;
        assert_eq!(
            report.removed.len(),
            orphans.len(),
            "every orphan class swept"
        );
        assert_eq!(
            std::fs::read(index.path().join("MANIFEST"))?,
            canonical,
            "canonical authority bytes are unchanged"
        );
        let second = collect_writer_garbage_under_lock(
            index.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions {
                grace_period: Duration::ZERO,
            },
        )?;
        assert!(second.removed.is_empty(), "second sweep is empty");
        Ok(())
    }
}
