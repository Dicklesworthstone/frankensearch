#![forbid(unsafe_code)]
//! Quill is frankensearch's native, deterministic lexical engine.
//!
//! This crate owns the schema and scoring contracts shared by the ingest,
//! storage, query, and lifecycle subsystems. The current milestone establishes
//! those contracts; later milestones fill in each subsystem behind them.
//!
//! ```
//! use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};
//!
//! let config = QuillConfig::default();
//! assert_eq!(config.tier_fanout, 8);
//! assert_ne!(DEFAULT_SCHEMA.schema_id().unwrap(), 0);
//! ```

pub mod argus;
pub mod config;
pub mod contract;
pub mod delta;
pub mod error;
pub mod grimoire;
pub mod index;
pub mod keeper;
pub mod query;
pub mod quiver;
pub mod schema;
pub mod scribe;
pub mod segment;
pub mod snippet;
pub mod stats;
pub mod tracing_conventions;

pub use config::QuillConfig;
pub use error::{QuillError, map_lock_error};
pub use grimoire::{
    ByteSpan, EncodedTermDictionary, MAX_COMPOSITE_KEY_BYTES, MAX_TERM_BYTES, OwnedTerm,
    TERM_BLOCK_TARGET_BYTES, TERM_RESTART_INTERVAL, TermCursor, TermDictionary,
    TermDictionaryError, TermDictionaryLimits, TermInput, TermMatch, TermMetadata, TermRef,
    TermScratch, TermSectionLengths,
};
pub use index::{
    QuillDocumentWitness, QuillHit, QuillIndex, QuillIndexError, QuillSearchIndex,
    QuillSearchResult, QuillSearchSnapshot, QuillSnippetHit, SnapshotError, SnapshotPublisher,
    indexable_document_content_hash,
};
#[cfg(feature = "durability")]
pub use keeper::UnrepairableSegmentPolicy;
pub use keeper::{
    BlueGreenEngine, CURRENT_ENGINE_VERSION, CURRENT_FILE_NAME, CURRENT_FORMAT_VERSION,
    CompactionError, CompactionPolicy, CompactionReport, ConcatMergeError, CurrentPointer,
    CurrentPointerError, DEFAULT_GARBAGE_GRACE, EMPTY_TOMBSTONES, GarbageCollectionOptions,
    GarbageCollectionReport, KeeperError, KeeperSnapshot, KeeperWriter, LoadedManifest,
    MANIFEST_FORMAT_VERSION, MANIFEST_FORMAT_VERSION_V1, MANIFEST_MAGIC, Manifest,
    ManifestCodecError, ManifestFieldStats, ManifestSegment, ManifestSource, QuarantinedSegment,
    RecoveredSegment, ResolvedCurrent, ResolvedDocumentId, SegmentSizeTier, TierMergePlan,
    TierMergePolicy, TierPolicyError, TombstoneSet, WRITER_LOCK_FORMAT_VERSION, WRITER_LOCK_MAGIC,
    WRITER_LOCK_RECORD_BYTES, load_manifest_pair, pack_engine_version, plan_tier_merge,
    publish_current, resolve_current, unpack_engine_version,
};
pub use query::{
    BooleanClause, BooleanOperator, CassQueryFilters, CassQueryParser, CassQueryParserConfigError,
    CassSourceFilter, CassWildcardPattern, DefaultQueryParser, MAX_QUERY_DEPTH, MAX_QUERY_LENGTH,
    Occur, ParsedQuery, PositionedTerm, Query, QueryCanonicalizationReport, QueryDiagnostic,
    QueryDiagnosticKind, QueryExplanation, QueryField, QueryParserConfigError, QueryValue,
    canonicalize_query, cass_sanitize_query, classify_query, truncate_query,
};
pub use schema::{
    Analyzer, CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor, FieldKind,
    SchemaDescriptor,
};
pub use segment::{
    EncodedSegment, FSLX_FORMAT_VERSION, FSLX_SECTION_ALIGNMENT, FSLX_SEGMENT_MAGIC,
    PendingSegmentFile, SectionEntry, SectionInput, SectionKind, SegmentHeader, SegmentHeaderInput,
    SegmentLimits, SegmentReader,
};
pub use snippet::{DEFAULT_SNIPPET_MAX_CHARS, SnippetConfig, SnippetGenerator, SnippetTerm};
pub use stats::{SegmentStats, SegmentStatsProvider};
