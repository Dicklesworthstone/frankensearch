//! Shipping Quill index orchestration and process-local snapshot publication.
//!
//! This module deliberately joins the already-final Scribe, FSLX, Keeper,
//! parser, cursor, scorer, and collector boundaries. Immutable Delta epochs,
//! their composite statistics, scorer adapters, and seal transaction are
//! assembled here; later mixed-state beads wire them into the public writer loop.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Bound, Deref};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "conformance-internals")]
use std::sync::Mutex as StdMutex;
#[cfg(feature = "conformance-internals")]
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::{ArcSwap, ArcSwapOption};
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::{LockError, Mutex, OwnedMutexGuard, TryLockError};
use frankensearch_core::{
    DocId, IndexableDocument, LexicalCandidateBatch, LexicalHydrationContext, LexicalRead,
    LexicalSearch, LexicalWrite, ScoreSource, ScoredResult, SearchError, SearchFuture,
};
#[cfg(feature = "durability")]
use frankensearch_durability::FileProtector;
use rayon::prelude::*;
#[cfg(feature = "pruning-conformance")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "conformance-internals")]
use sha2::{Digest, Sha256};
use thiserror::Error;
use tracing::Instrument;
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::argus::{
    ArgusError, BMW_MIN_CLAUSES, Bm25FieldSnapshot, CheckpointPostingCursor, DeltaFieldNorms,
    DeltaPostingCursor, DocSetCollector, FieldNormReader, MAX_SCORE_MAX_CLAUSES, PhraseScorer,
    PhraseTerm, PositionsHandle, PositionsReader, PostingCursor, PruningTelemetry,
    QueryWorkCheckpoint, QueryWorkKind, ReferenceScorer, ScorerClause, SealedPostingCursor,
    TermRecordOption, TermScorer, TopDocsCollector,
};
#[cfg(feature = "pruning-conformance")]
use crate::argus::{ConformanceUnionRefill, ConformanceUnionRefillStrategy};
use crate::config::QuillConfig;
use crate::delta::DeltaSnapshot;
use crate::error::QuillError;
use crate::grimoire::{
    ByteSpan, TermDictionary, TermDictionaryError, star_glob_matches, trailing_star_prefix,
    validate_bound_term, validate_query_term,
};
#[cfg(feature = "durability")]
use crate::keeper::UnrepairableSegmentPolicy;
use crate::keeper::{
    CURRENT_ENGINE_VERSION, CompactionPolicy, CompactionReport, KeeperError, KeeperSnapshot,
    KeeperWriter, MANIFEST_FLAG_BULK_MODE_IN_PROGRESS, Manifest, ManifestFieldStats,
    ManifestSegment, RecoveredSegment, TierMergePolicy, TierPolicyError, TombstoneSet,
    plan_tier_merge, validate_manifest_successor,
};
use crate::query::{
    BooleanOperator, DefaultQueryParser, Occur, Query, QueryCapabilityError, QueryDiagnostic,
    QueryExplanation, QueryParserConfigError, QueryValue, canonicalize_query, classify_query,
    validate_index_capabilities,
};
use crate::quiver::{
    BlockMaxError, DocLenCodecError, DocLenField, DocLenSection, EncodedNumericSection,
    NumericEntry, NumericField, NumericFieldInput, NumericSection, NumericValue,
    PositionCodecError, PositionList, Posting, PostingCodecError, PostingList, SnapshotFieldStats,
    StoredMetaCodecError, StoredMetaSection,
};
use crate::schema::{DEFAULT_SCHEMA, FieldKind, SchemaDescriptor};
use crate::scribe::{
    AccumulatorError, ColumnarAccumulator, DOC_ORDS_PER_LEASE, DeltaFlushInput, DocIdAllocator,
    DocIdSpan, FlushDocumentInput, FlushError, FlushMode, FlushSegmentInput, IndexedFieldValue,
    IndexedNumericValue, ShardRouter, StoredFieldValue, flush_accumulator_with_mode,
    flush_delta_snapshot,
};
use crate::segment::{EncodedSegment, SectionKind};
use crate::snippet::{SnippetConfig, SnippetGenerator, SnippetTerm};
use crate::stats::{SegmentStats, SegmentStatsProvider};

const ID_FIELD: u16 = 0;
const CONTENT_FIELD: u16 = 1;
const TITLE_FIELD: u16 = 2;
const METADATA_FIELD: u16 = 3;
const ORD_FIELD: u16 = 4;
const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;
const PARALLEL_INGEST_MIN_DOCS_PER_SHARD: usize = 64;
const CONTENT_HASH_DOMAIN: &[u8] = b"frankensearch.quill.idmap-content.v2\0";

/// Typed failure from the scalar shipping facade.
#[derive(Debug, Error)]
pub enum QuillIndexError {
    /// Engine configuration was rejected before opening resources.
    #[error("invalid Quill index configuration: {0}")]
    Config(#[source] frankensearch_core::SearchError),
    /// Keeper admission, recovery, or publication failed.
    #[error(transparent)]
    Keeper(#[from] KeeperError),
    /// FSLX framing or section access failed.
    #[error(transparent)]
    Quill(#[from] QuillError),
    /// Parser/schema binding failed.
    #[error(transparent)]
    Parser(#[from] QueryParserConfigError),
    /// A parsed or prebuilt operator requires an index capability the schema
    /// does not provide.
    #[error(transparent)]
    QueryCapability(#[from] QueryCapabilityError),
    /// One document could not be accumulated atomically.
    #[error(transparent)]
    Accumulator(#[from] AccumulatorError),
    /// A scalar accumulator could not be sealed.
    #[error(transparent)]
    Flush(#[from] FlushError),
    /// Bound-consecutive tier planning failed.
    #[error(transparent)]
    TierPolicy(#[from] TierPolicyError),
    /// A term dictionary was malformed or incompatible with its sections.
    #[error(transparent)]
    Dictionary(#[from] TermDictionaryError),
    /// A posting list was malformed.
    #[error(transparent)]
    Postings(#[from] PostingCodecError),
    /// A block-max list was malformed or inconsistent with POSTINGS/DOCLEN.
    #[error(transparent)]
    BlockMax(#[from] BlockMaxError),
    /// A positions list was malformed.
    #[error(transparent)]
    Positions(#[from] PositionCodecError),
    /// A field-length section was malformed.
    #[error(transparent)]
    DocLen(#[from] DocLenCodecError),
    /// A stored metadata column was malformed.
    #[error(transparent)]
    StoredMeta(#[from] StoredMetaCodecError),
    /// Exhaustive scorer construction or collection failed.
    #[error(transparent)]
    Argus(ArgusError),
    /// Composite process-local snapshot construction or publication failed.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    /// Canonical metadata serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The requested query shape is outside the scalar G1a surface.
    #[error("unsupported scalar Quill query: {detail}")]
    UnsupportedQuery { detail: String },
    /// An index lifecycle or arithmetic invariant failed.
    #[error("invalid scalar Quill index state: {detail}")]
    InvalidState { detail: String },
    /// Structured cancellation was observed before a state transition.
    #[error("scalar Quill operation cancelled during {phase}")]
    Cancelled { phase: &'static str },
    /// Deterministic coarse query work exceeded the configured budget.
    #[error(
        "scalar Quill query fuel exhausted after {consumed}/{budget} units \
         (segments={segments_touched}, dictionary_blocks={dictionary_blocks}, \
         posting_blocks={posting_blocks}, position_docs={position_docs})"
    )]
    QueryFuelExhausted {
        /// Configured work-unit ceiling.
        budget: u64,
        /// Successfully admitted work before exhaustion.
        consumed: u64,
        /// Segment transitions admitted before exhaustion.
        segments_touched: u64,
        /// Dictionary blocks admitted before exhaustion.
        dictionary_blocks: u64,
        /// Posting blocks admitted before exhaustion.
        posting_blocks: u64,
        /// Phrase candidate documents admitted before exhaustion.
        position_docs: u64,
    },
}

impl From<ArgusError> for QuillIndexError {
    fn from(error: ArgusError) -> Self {
        match error {
            ArgusError::QueryCancelled { phase } => Self::Cancelled { phase },
            ArgusError::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            } => Self::QueryFuelExhausted {
                budget,
                consumed,
                segments_touched,
                dictionary_blocks,
                posting_blocks,
                position_docs,
            },
            other => Self::Argus(other),
        }
    }
}

/// One final lexical winner.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillHit {
    /// Stable external document identifier.
    pub document_id: String,
    /// Snapshot-global Q1 document identifier used as the native tie key.
    pub global_docid: u32,
    /// Exhaustive BM25 score.
    pub score: f32,
}

/// One exact IDHASH/IDMAP identity witness from the published snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuillDocumentWitness {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Unseeded xxh3-64 witness over the canonical indexed document.
    pub content_hash: u64,
}

/// One enriched Quill hit returned by [`QuillIndex::search_with_snippets`].
#[derive(Clone, Debug)]
pub struct QuillSnippetHit {
    /// Stable external document identifier.
    pub document_id: String,
    /// Exhaustive BM25 score.
    pub score: f32,
    /// Zero-based result rank.
    pub rank: usize,
    /// Highlighted content fragment when the active schema stores source text.
    pub snippet: Option<String>,
    /// Stable explanation of the raw query shape.
    pub query_type: QueryExplanation,
    /// Canonical stored metadata.
    pub metadata: Option<Arc<serde_json::Value>>,
}

/// One globally paginated exhaustive result.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillSearchResult {
    /// Page-local hits in `(score desc, global_docid asc)` order.
    pub hits: Vec<QuillHit>,
    /// Exact match count when requested.
    pub total_count: Option<u64>,
    /// Public live-document count for the committed snapshot.
    pub doc_count: u64,
    /// Lenient parser recovery diagnostics.
    pub diagnostics: Vec<QueryDiagnostic>,
}

const RANKED_QUERY_CACHE_SETS: usize = 64;
const RANKED_QUERY_CACHE_WAYS: usize = 4;
const RANKED_QUERY_CACHE_SLOTS: usize = RANKED_QUERY_CACHE_SETS * RANKED_QUERY_CACHE_WAYS;

enum RankedQueryCacheKey {
    Raw(Arc<str>),
    Preparsed(Arc<Query>),
}

struct CachedRankedQuery {
    fingerprint: u64,
    snapshot_epoch: u64,
    limit: usize,
    offset: usize,
    exact_count: bool,
    key: RankedQueryCacheKey,
    result: Arc<QuillSearchResult>,
}

/// Small lock-free memo table for immutable-snapshot searches.
///
/// Exact keys are retained alongside their fingerprints, so collisions only
/// cost a miss. Snapshot epochs make publication the invalidation boundary.
/// The fixed four-way table bounds memory without adding reader coordination.
struct RankedQueryCache {
    slots: [ArcSwapOption<CachedRankedQuery>; RANKED_QUERY_CACHE_SLOTS],
}

impl Default for RankedQueryCache {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| ArcSwapOption::empty()),
        }
    }
}

impl RankedQueryCache {
    fn get_raw(
        &self,
        snapshot_epoch: u64,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Option<QuillSearchResult> {
        let fingerprint =
            raw_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.find(fingerprint, |cached| {
            cached.snapshot_epoch == snapshot_epoch
                && cached.limit == limit
                && cached.offset == offset
                && cached.exact_count == exact_count
                && matches!(&cached.key, RankedQueryCacheKey::Raw(key) if key.as_ref() == query)
        })
    }

    fn get_preparsed(
        &self,
        snapshot_epoch: u64,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Option<QuillSearchResult> {
        let fingerprint =
            preparsed_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.find(fingerprint, |cached| {
            cached.snapshot_epoch == snapshot_epoch
                && cached.limit == limit
                && cached.offset == offset
                && cached.exact_count == exact_count
                && matches!(&cached.key, RankedQueryCacheKey::Preparsed(key) if key.as_ref() == query)
        })
    }

    fn find(
        &self,
        fingerprint: u64,
        exact_match: impl Fn(&CachedRankedQuery) -> bool,
    ) -> Option<QuillSearchResult> {
        let start = ranked_query_cache_set_start(fingerprint);
        let set = self
            .slots
            .get(start..start.checked_add(RANKED_QUERY_CACHE_WAYS)?)?;
        for slot in set {
            let Some(cached) = slot.load_full() else {
                continue;
            };
            if cached.fingerprint == fingerprint && exact_match(cached.as_ref()) {
                return Some(cached.result.as_ref().clone());
            }
        }
        None
    }

    fn insert_raw(
        &self,
        snapshot_epoch: u64,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        result: &QuillSearchResult,
    ) {
        let fingerprint =
            raw_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.insert(Arc::new(CachedRankedQuery {
            fingerprint,
            snapshot_epoch,
            limit,
            offset,
            exact_count,
            key: RankedQueryCacheKey::Raw(Arc::from(query)),
            result: Arc::new(result.clone()),
        }));
    }

    fn insert_preparsed(
        &self,
        snapshot_epoch: u64,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
        result: &QuillSearchResult,
    ) {
        let fingerprint =
            preparsed_query_cache_fingerprint(snapshot_epoch, query, limit, offset, exact_count);
        self.insert(Arc::new(CachedRankedQuery {
            fingerprint,
            snapshot_epoch,
            limit,
            offset,
            exact_count,
            key: RankedQueryCacheKey::Preparsed(Arc::new(query.clone())),
            result: Arc::new(result.clone()),
        }));
    }

    fn insert(&self, cached: Arc<CachedRankedQuery>) {
        let start = ranked_query_cache_set_start(cached.fingerprint);
        let Some(end) = start.checked_add(RANKED_QUERY_CACHE_WAYS) else {
            return;
        };
        let Some(set) = self.slots.get(start..end) else {
            return;
        };
        if let Some(slot) = set.iter().find(|slot| slot.load().is_none()) {
            slot.store(Some(cached));
            return;
        }
        let [_, replacement_byte, ..] = cached.fingerprint.to_le_bytes();
        let replacement = usize::from(replacement_byte) & (RANKED_QUERY_CACHE_WAYS - 1);
        if let Some(slot) = set.get(replacement) {
            slot.store(Some(cached));
        }
    }
}

fn ranked_query_cache_set_start(fingerprint: u64) -> usize {
    let [set_byte, ..] = fingerprint.to_le_bytes();
    let set = usize::from(set_byte) & (RANKED_QUERY_CACHE_SETS - 1);
    set * RANKED_QUERY_CACHE_WAYS
}

fn raw_query_cache_fingerprint(
    snapshot_epoch: u64,
    query: &str,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> u64 {
    let mut hasher = ranked_query_cache_hasher(snapshot_epoch, limit, offset, exact_count);
    hasher.update(&[0]);
    hash_query_cache_bytes(&mut hasher, query.as_bytes());
    hasher.digest()
}

fn preparsed_query_cache_fingerprint(
    snapshot_epoch: u64,
    query: &Query,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> u64 {
    let mut hasher = ranked_query_cache_hasher(snapshot_epoch, limit, offset, exact_count);
    hasher.update(&[1]);
    hash_exact_query(&mut hasher, query);
    hasher.digest()
}

fn ranked_query_cache_hasher(
    snapshot_epoch: u64,
    limit: usize,
    offset: usize,
    exact_count: bool,
) -> Xxh3 {
    let mut hasher = Xxh3::new();
    hasher.update(b"frankensearch.quill.ranked-query-cache.v1\0");
    hasher.update(&snapshot_epoch.to_le_bytes());
    hasher.update(&u64::try_from(limit).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&u64::try_from(offset).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(&[u8::from(exact_count)]);
    hasher
}

fn hash_query_cache_bytes(hasher: &mut Xxh3, bytes: &[u8]) {
    hasher.update(&u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

fn hash_query_cache_len(hasher: &mut Xxh3, len: usize) {
    hasher.update(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

fn hash_query_cache_value(hasher: &mut Xxh3, value: &QueryValue) {
    match value {
        QueryValue::I64(value) => {
            hasher.update(&[0]);
            hasher.update(&value.to_le_bytes());
        }
        QueryValue::U64(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        QueryValue::Str(value) => {
            hasher.update(&[2]);
            hash_query_cache_bytes(hasher, value.as_bytes());
        }
    }
}

fn hash_query_cache_bound(hasher: &mut Xxh3, bound: &Bound<QueryValue>) {
    match bound {
        Bound::Included(value) => {
            hasher.update(&[0]);
            hash_query_cache_value(hasher, value);
        }
        Bound::Excluded(value) => {
            hasher.update(&[1]);
            hash_query_cache_value(hasher, value);
        }
        Bound::Unbounded => hasher.update(&[2]),
    }
}

fn hash_exact_query(hasher: &mut Xxh3, query: &Query) {
    match query {
        Query::Empty => hasher.update(&[0]),
        Query::All => hasher.update(&[1]),
        Query::Term { fields, text } => {
            hasher.update(&[2]);
            hash_query_cache_len(hasher, fields.len());
            for field in fields {
                hasher.update(&field.field_id.to_le_bytes());
                hasher.update(&field.boost.to_bits().to_le_bytes());
            }
            hash_query_cache_bytes(hasher, text.as_bytes());
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            hasher.update(&[3]);
            hash_query_cache_len(hasher, fields.len());
            for field in fields {
                hasher.update(&field.field_id.to_le_bytes());
                hasher.update(&field.boost.to_bits().to_le_bytes());
            }
            hash_query_cache_len(hasher, terms.len());
            for term in terms {
                hasher.update(&term.position.to_le_bytes());
                hash_query_cache_bytes(hasher, term.text.as_bytes());
            }
            hasher.update(&slop.to_le_bytes());
            hasher.update(&[u8::from(*prefix)]);
        }
        Query::Boolean { clauses, operator } => {
            hasher.update(&[4]);
            hasher.update(&[match operator {
                None => 0,
                Some(BooleanOperator::And) => 1,
                Some(BooleanOperator::Or) => 2,
            }]);
            hash_query_cache_len(hasher, clauses.len());
            for clause in clauses {
                hasher.update(&[match clause.occur {
                    Occur::Must => 0,
                    Occur::Should => 1,
                    Occur::MustNot => 2,
                }]);
                hash_exact_query(hasher, &clause.query);
            }
        }
        Query::Range {
            field_id,
            lower,
            upper,
        } => {
            hasher.update(&[5]);
            hasher.update(&field_id.to_le_bytes());
            hash_query_cache_bound(hasher, lower);
            hash_query_cache_bound(hasher, upper);
        }
        Query::Set { field_id, values } => {
            hasher.update(&[6]);
            hasher.update(&field_id.to_le_bytes());
            hash_query_cache_len(hasher, values.len());
            for value in values {
                hash_query_cache_value(hasher, value);
            }
        }
        Query::Glob { field_ids, pattern } => {
            hasher.update(&[7]);
            hash_query_cache_len(hasher, field_ids.len());
            for field_id in field_ids {
                hasher.update(&field_id.to_le_bytes());
            }
            hash_query_cache_bytes(hasher, pattern.as_bytes());
        }
        Query::Boost { query, factor } => {
            hasher.update(&[8]);
            hasher.update(&factor.to_bits().to_le_bytes());
            hash_exact_query(hasher, query);
        }
    }
}

/// Typed failure from process-local Keeper plus Delta snapshot composition.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// A Delta epoch was frozen against a different schema.
    #[error("delta schema {actual} does not match Keeper schema {expected}")]
    SchemaMismatch {
        /// Keeper schema name.
        expected: &'static str,
        /// Delta schema name.
        actual: &'static str,
    },
    /// A Delta epoch was frozen against a different Keeper generation.
    #[error(
        "delta lease starting at {lease_base} belongs to Keeper generation {actual}, expected {expected}"
    )]
    KeeperGenerationMismatch {
        /// Generation carried by the composite Keeper snapshot.
        expected: u64,
        /// Generation witnessed when the Delta epoch was frozen.
        actual: u64,
        /// Delta lease identifying the stale epoch.
        lease_base: u64,
    },
    /// A Delta epoch's physical covering interval intersects a sealed range.
    #[error(
        "delta occupied range [{delta_lo}, {delta_hi}) overlaps Keeper segment {segment_id:#018x} range [{keeper_lo}, {keeper_hi})"
    )]
    KeeperDeltaDocidOverlap {
        /// Inclusive first physically allocated Delta docid.
        delta_lo: u64,
        /// Exclusive Delta covering upper bound.
        delta_hi: u64,
        /// Conflicting sealed segment identity.
        segment_id: u64,
        /// Inclusive sealed covering lower bound.
        keeper_lo: u64,
        /// Exclusive sealed covering upper bound.
        keeper_hi: u64,
    },
    /// Two independently published shards claimed overlapping Q1 leases.
    #[error("delta leases [{first_base}, {first_end}) and [{second_base}, {second_end}) overlap")]
    OverlappingDeltaLeases {
        /// First inclusive lease base.
        first_base: u64,
        /// First exclusive lease end.
        first_end: u64,
        /// Second inclusive lease base.
        second_base: u64,
        /// Second exclusive lease end.
        second_end: u64,
    },
    /// A seal replacement table omitted one independently published shard.
    #[error("delta seal replacement table omitted the epoch in lease [{lease_base}, {lease_end})")]
    MissingDeltaRebind {
        /// Inclusive lease base of the disappearing process-local epoch.
        lease_base: u64,
        /// Exclusive lease end of the disappearing process-local epoch.
        lease_end: u64,
    },
    /// A complete publication attempted to replace a newer Keeper generation.
    #[error("Keeper generation regressed from current {current} to proposed {proposed}")]
    KeeperGenerationRegression {
        /// Currently published durable generation.
        current: u64,
        /// Stale proposed generation.
        proposed: u64,
    },
    /// One generation number named two different durable MANIFEST images.
    #[error("Keeper generation {generation} identifies two different MANIFEST images")]
    KeeperGenerationCollision {
        /// Reused generation number.
        generation: u64,
    },
    /// A forward complete publication was not a valid Keeper successor.
    #[error("invalid Keeper snapshot transition: {detail}")]
    KeeperTransition {
        /// Underlying Keeper transition rejection.
        detail: String,
    },
    /// A non-genesis Keeper snapshot omitted an indexed field's statistics.
    #[error("Keeper snapshot has no field statistics for indexed field {field_ord}")]
    MissingKeeperFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// A schema-compatible Delta failed to expose one indexed field's tokens.
    #[error("delta snapshot has no live token statistics for indexed field {field_ord}")]
    MissingDeltaFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// One checked composite counter exceeded its u64 representation.
    #[error("composite snapshot {counter} overflowed")]
    CounterOverflow {
        /// Stable name of the counter being aggregated.
        counter: &'static str,
    },
    /// A per-shard publication named no current Delta slot.
    #[error("delta shard {shard} is outside the published shard count {shard_count}")]
    ShardOutOfBounds {
        /// Requested zero-based shard slot.
        shard: usize,
        /// Current number of published shard slots.
        shard_count: usize,
    },
    /// The process-local publication epoch cannot advance further.
    #[error("process-local snapshot epoch exhausted")]
    EpochExhausted,
    /// A bounded publication sidecar could not be allocated.
    #[error("unable to allocate {additional} entries for {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested additional element count.
        additional: usize,
    },
}

/// One immutable, internally consistent process-local search view.
///
/// The Keeper component retains sealed segments and MANIFEST tombstones. Each
/// Delta component is an owner-isolated frozen epoch. BM25 statistics use the
/// hybrid lifecycle contract: sealed tombstones remain in Keeper's at-seal
/// counts until compaction, while Delta-local tombstones are excluded
/// immediately because those rows will never be sealed.
pub struct QuillSearchSnapshot {
    snapshot_epoch: u64,
    keeper: Arc<KeeperSnapshot>,
    deltas: Box<[Arc<DeltaSnapshot>]>,
    field_stats: Box<[SnapshotFieldStats]>,
    bm25_doc_count: u64,
    live_doc_count: u64,
}

impl QuillSearchSnapshot {
    fn compose(
        snapshot_epoch: u64,
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let schema = keeper.schema();
        let manifest = &keeper.loaded_manifest().manifest;
        validate_delta_table(schema, manifest, &deltas)?;

        let delta_live_doc_count = delta_live_document_count(&deltas)?;
        let bm25_doc_count = keeper
            .at_seal_doc_count()
            .checked_add(delta_live_doc_count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "BM25 document count",
            })?;
        let live_doc_count = keeper.doc_count().checked_add(delta_live_doc_count).ok_or(
            SnapshotError::CounterOverflow {
                counter: "live document count",
            },
        )?;

        let field_stats = composite_field_stats(
            schema,
            manifest,
            keeper.segments().is_empty(),
            &deltas,
            bm25_doc_count,
        )?;

        Ok(Self {
            snapshot_epoch,
            keeper,
            deltas: deltas.into_boxed_slice(),
            field_stats,
            bm25_doc_count,
            live_doc_count,
        })
    }

    /// Monotone process-local publication epoch.
    #[must_use]
    pub const fn snapshot_epoch(&self) -> u64 {
        self.snapshot_epoch
    }

    /// Durable MANIFEST generation paired with every Delta epoch in this view.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.keeper.loaded_manifest().manifest.generation
    }

    /// Pinned immutable Keeper component.
    #[must_use]
    pub fn keeper_snapshot(&self) -> &KeeperSnapshot {
        &self.keeper
    }

    /// Number of frozen shard epochs in this view.
    #[must_use]
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Immutable shard epochs paired with this Keeper generation.
    #[must_use]
    pub fn delta_snapshots(&self) -> &[Arc<DeltaSnapshot>] {
        &self.deltas
    }

    /// Snapshot-global BM25 `N`: Keeper at-seal rows plus live Delta rows.
    #[must_use]
    pub const fn bm25_doc_count(&self) -> u64 {
        self.bm25_doc_count
    }

    /// Public live rows across Keeper tombstones and Delta-local tombstones.
    #[must_use]
    pub const fn live_doc_count(&self) -> u64 {
        self.live_doc_count
    }

    /// Checked composite BM25 statistics for one indexed string field.
    #[must_use]
    pub fn bm25_field_stats(&self, field_ord: u16) -> Option<SnapshotFieldStats> {
        self.field_stats
            .binary_search_by_key(&field_ord, |row| row.field_ord)
            .ok()
            .and_then(|index| self.field_stats.get(index))
            .copied()
    }

    /// Keeper physical document frequency plus live Delta document frequency.
    ///
    /// # Errors
    ///
    /// Returns typed dictionary, allocation, or checked-counter failures.
    pub fn bm25_doc_freq(&self, field_ord: u16, term: &[u8]) -> Result<u64, QuillIndexError> {
        if self.bm25_field_stats(field_ord).is_none() {
            return Err(invalid_state(format!(
                "snapshot has no BM25 statistics for field {field_ord}"
            )));
        }
        let mut total = snapshot_doc_freq(&self.keeper, self.keeper.schema(), field_ord, term)?;
        for delta in &self.deltas {
            let delta_doc_freq = delta
                .find_term(field_ord, term)
                .map_or(0, |found| found.live_doc_freq());
            total = total
                .checked_add(u64::try_from(delta_doc_freq).map_err(|_| {
                    SnapshotError::CounterOverflow {
                        counter: "live Delta document frequency",
                    }
                })?)
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "snapshot document frequency",
                })?;
        }
        Ok(total)
    }

    /// Materialize one live winner from either immutable residency.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        self.keeper
            .materialize_document_id(global_docid)
            .or_else(|| {
                self.deltas
                    .iter()
                    .find_map(|delta| delta.materialize_document_id(global_docid))
            })
    }

    fn resolve_document_witness(
        &self,
        document_id: &str,
    ) -> Result<Option<QuillDocumentWitness>, QuillIndexError> {
        for delta in self.deltas.iter().rev() {
            if let Some(global_docid) = delta.segment().probe_id(document_id) {
                let content_hash = delta.content_hash(global_docid).ok_or_else(|| {
                    invalid_state(format!(
                        "live Delta document {document_id:?} has no resumable content witness"
                    ))
                })?;
                return Ok(Some(QuillDocumentWitness {
                    global_docid,
                    content_hash,
                }));
            }
        }
        Ok(self
            .keeper
            .resolve_document_id(document_id)?
            .map(|resolved| QuillDocumentWitness {
                global_docid: resolved.global_docid,
                content_hash: resolved.content_hash,
            }))
    }

    fn resolve_document_id(&self, document_id: &str) -> Result<Option<u32>, QuillIndexError> {
        self.resolve_document_witness(document_id)
            .map(|witness| witness.map(|witness| witness.global_docid))
    }

    fn materialize_metadata(
        &self,
        global_docid: u32,
    ) -> Result<Option<Arc<serde_json::Value>>, QuillIndexError> {
        let Some(stored) = self.materialize_stored_value(METADATA_FIELD, global_docid)? else {
            return Ok(None);
        };
        let metadata: serde_json::Value = serde_json::from_slice(&stored)?;
        // Ingest always writes a canonical metadata object, so a document indexed
        // without metadata stores `{}`. Tantivy omits the field entirely in that
        // case, and an empty object carries no information anyway — materialize
        // both as `None` so the public envelope never hands out an
        // information-free `Some({})` (with its per-hit `Arc`) and so
        // `metadata.is_some()` means the same thing on both engines.
        if metadata.as_object().is_some_and(serde_json::Map::is_empty) {
            return Ok(None);
        }
        Ok(Some(Arc::new(metadata)))
    }

    fn materialize_stored_value(
        &self,
        field_ord: u16,
        global_docid: u32,
    ) -> Result<Option<Vec<u8>>, QuillIndexError> {
        for delta in &self.deltas {
            if delta.is_live_document(global_docid) {
                return Ok(delta
                    .stored_value(field_ord, global_docid)
                    .map(<[u8]>::to_vec));
            }
        }

        let Some(segment) = self
            .keeper
            .segments()
            .iter()
            .find(|segment| segment.materialize_document_id(global_docid).is_some())
        else {
            return Ok(None);
        };
        let manifest = segment.manifest();
        let stored_fields = self
            .keeper
            .schema()
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| field.id)
            .collect::<Vec<_>>();
        let stored = StoredMetaSection::parse(
            required_section(segment, SectionKind::STOREDMETA)?,
            manifest.docid_lo,
            manifest.docid_hi,
            &stored_fields,
        )?;
        Ok(stored
            .field(field_ord)
            .and_then(|field| field.get(u64::from(global_docid)))
            .map(<[u8]>::to_vec))
    }

    /// Clone the pinned immutable Keeper component.
    #[must_use]
    pub fn keeper_snapshot_arc(&self) -> Arc<KeeperSnapshot> {
        Arc::clone(&self.keeper)
    }
}

fn validate_delta_table(
    schema: SchemaDescriptor,
    manifest: &Manifest,
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<(), SnapshotError> {
    let keeper_generation = manifest.generation;
    for delta in deltas {
        if delta.schema() != schema {
            return Err(SnapshotError::SchemaMismatch {
                expected: schema.name,
                actual: delta.schema().name,
            });
        }
        if delta.keeper_generation() != keeper_generation {
            return Err(SnapshotError::KeeperGenerationMismatch {
                expected: keeper_generation,
                actual: delta.keeper_generation(),
                lease_base: delta.lease_base(),
            });
        }
        if let Some((delta_lo, delta_hi)) = delta.occupied_docid_range() {
            for segment in &manifest.segments {
                if delta_lo < segment.docid_hi && segment.docid_lo < delta_hi {
                    return Err(SnapshotError::KeeperDeltaDocidOverlap {
                        delta_lo,
                        delta_hi,
                        segment_id: segment.segment_id,
                        keeper_lo: segment.docid_lo,
                        keeper_hi: segment.docid_hi,
                    });
                }
            }
        }
    }
    for (index, first) in deltas.iter().enumerate() {
        for second in &deltas[index + 1..] {
            if first.lease_base() < second.lease_end() && second.lease_base() < first.lease_end() {
                return Err(SnapshotError::OverlappingDeltaLeases {
                    first_base: first.lease_base(),
                    first_end: first.lease_end(),
                    second_base: second.lease_base(),
                    second_end: second.lease_end(),
                });
            }
        }
    }
    Ok(())
}

fn validate_delta_seal_replacements(
    current: &QuillSearchSnapshot,
    sealed: &Arc<DeltaSnapshot>,
    replacements: &[Arc<DeltaSnapshot>],
) -> Result<(), SnapshotError> {
    for survivor in current
        .delta_snapshots()
        .iter()
        .filter(|delta| !Arc::ptr_eq(delta, sealed))
    {
        let lineage = survivor.publication_lineage();
        if !replacements
            .iter()
            .any(|replacement| replacement.publication_lineage() == lineage)
        {
            return Err(SnapshotError::MissingDeltaRebind {
                lease_base: survivor.lease_base(),
                lease_end: survivor.lease_end(),
            });
        }
    }
    Ok(())
}

fn delta_live_document_count(deltas: &[Arc<DeltaSnapshot>]) -> Result<u64, SnapshotError> {
    deltas.iter().try_fold(0_u64, |total, delta| {
        let count = u64::try_from(delta.live_document_count()).map_err(|_| {
            SnapshotError::CounterOverflow {
                counter: "live Delta document count",
            }
        })?;
        total
            .checked_add(count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "live Delta document count",
            })
    })
}

fn composite_field_stats(
    schema: SchemaDescriptor,
    manifest: &Manifest,
    genesis_without_segments: bool,
    deltas: &[Arc<DeltaSnapshot>],
    bm25_doc_count: u64,
) -> Result<Box<[SnapshotFieldStats]>, SnapshotError> {
    let indexed_field_count = schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    let mut field_stats = Vec::new();
    field_stats
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| SnapshotError::Allocation {
            resource: "composite field statistics",
            additional: indexed_field_count,
        })?;
    for field in schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
    {
        let sealed_total_tokens = manifest
            .field_stats
            .iter()
            .find(|row| row.field_ord == field.id)
            .map(|row| row.total_tokens)
            .or_else(|| genesis_without_segments.then_some(0))
            .ok_or(SnapshotError::MissingKeeperFieldStats {
                field_ord: field.id,
            })?;
        let total_tokens = deltas
            .iter()
            .try_fold(sealed_total_tokens, |total, delta| {
                let delta_tokens = delta.live_total_tokens(field.id).ok_or(
                    SnapshotError::MissingDeltaFieldStats {
                        field_ord: field.id,
                    },
                )?;
                total
                    .checked_add(delta_tokens)
                    .ok_or(SnapshotError::CounterOverflow {
                        counter: "field token count",
                    })
            })?;
        field_stats.push(SnapshotFieldStats {
            field_ord: field.id,
            total_tokens,
            doc_count: bm25_doc_count,
        });
    }
    Ok(field_stats.into_boxed_slice())
}

fn manifest_document_counts(manifest: &Manifest) -> Result<(u64, u64), SnapshotError> {
    manifest
        .segments
        .iter()
        .try_fold((0_u64, 0_u64), |(at_seal, live), segment| {
            let at_seal = at_seal.checked_add(u64::from(segment.doc_count)).ok_or(
                SnapshotError::CounterOverflow {
                    counter: "Keeper at-seal document count",
                },
            )?;
            let live = live
                .checked_add(u64::from(segment.live_doc_count()))
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "Keeper live document count",
                })?;
            Ok((at_seal, live))
        })
}

fn validate_complete_keeper_transition(
    current: &Manifest,
    proposed: &Manifest,
) -> Result<(), SnapshotError> {
    if proposed.generation < current.generation {
        return Err(SnapshotError::KeeperGenerationRegression {
            current: current.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.generation == current.generation {
        return if proposed == current {
            Ok(())
        } else {
            Err(SnapshotError::KeeperGenerationCollision {
                generation: proposed.generation,
            })
        };
    }
    validate_manifest_successor(current, proposed).map_err(|error| {
        SnapshotError::KeeperTransition {
            detail: error.to_string(),
        }
    })
}

struct PreparedSealedPublication {
    snapshot_epoch: u64,
    expected_keeper_generation: u64,
    expected_schema_id: u64,
    expected_docid_high_watermark: u64,
    schema: SchemaDescriptor,
    field_stats: Box<[SnapshotFieldStats]>,
    bm25_doc_count: u64,
    live_doc_count: u64,
    delta_live_doc_count: u64,
    deltas: Box<[Arc<DeltaSnapshot>]>,
}

/// Lock-free publisher for one authoritative Keeper plus all-Delta view.
///
/// Readers call [`Self::load`] once and retain that `Arc` for the whole query.
/// Per-shard updates use compare-and-swap so concurrent shard writers cannot
/// lose one another's epochs. A seal transition must use
/// [`Self::publish_complete`] to replace Keeper and every Delta in one swap.
pub struct SnapshotPublisher {
    current: ArcSwap<QuillSearchSnapshot>,
    ranked_query_cache: RankedQueryCache,
}

impl SnapshotPublisher {
    /// Construct the first process-local view at epoch zero.
    ///
    /// # Errors
    ///
    /// Rejects stale generations, schema drift, overlapping leases, missing
    /// field statistics, allocation failure, and checked-counter overflow.
    pub fn new(
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let initial = Arc::new(QuillSearchSnapshot::compose(0, keeper, deltas)?);
        Ok(Self {
            current: ArcSwap::new(initial),
            ranked_query_cache: RankedQueryCache::default(),
        })
    }

    /// Clone the current immutable view without taking a reader lock.
    #[must_use]
    pub fn load(&self) -> Arc<QuillSearchSnapshot> {
        self.current.load_full()
    }

    fn prepare_sealed_manifest(
        &self,
        schema: SchemaDescriptor,
        proposed: &Manifest,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        self.prepare_sealed_manifest_with_deltas(schema, proposed, Vec::new())
    }

    fn prepare_sealed_manifest_with_deltas(
        &self,
        schema: SchemaDescriptor,
        proposed: &Manifest,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        validate_complete_keeper_transition(&current.keeper.loaded_manifest().manifest, proposed)?;
        validate_delta_table(schema, proposed, &deltas)?;
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let (sealed_doc_count, sealed_live_doc_count) = manifest_document_counts(proposed)?;
        let delta_live_doc_count = delta_live_document_count(&deltas)?;
        let bm25_doc_count = sealed_doc_count.checked_add(delta_live_doc_count).ok_or(
            SnapshotError::CounterOverflow {
                counter: "BM25 document count",
            },
        )?;
        let live_doc_count = sealed_live_doc_count
            .checked_add(delta_live_doc_count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "live document count",
            })?;
        let field_stats = composite_field_stats(
            schema,
            proposed,
            proposed.segments.is_empty(),
            &deltas,
            bm25_doc_count,
        )?;
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation: proposed.generation,
            expected_schema_id: proposed.schema_id,
            expected_docid_high_watermark: proposed.docid_high_watermark,
            schema,
            field_stats,
            bm25_doc_count,
            live_doc_count,
            delta_live_doc_count,
            deltas: deltas.into_boxed_slice(),
        })
    }

    fn prepare_equivalent_sealed_successor(
        &self,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        if !current.deltas.is_empty() {
            return Err(SnapshotError::KeeperTransition {
                detail: "concat merge cannot discard active Delta epochs".to_owned(),
            });
        }
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let manifest = &current.keeper.loaded_manifest().manifest;
        let expected_keeper_generation =
            manifest
                .generation
                .checked_add(1)
                .ok_or_else(|| SnapshotError::KeeperTransition {
                    detail: "Keeper MANIFEST generation exhausted".to_owned(),
                })?;
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(current.field_stats.len())
            .map_err(|_| SnapshotError::Allocation {
                resource: "concat-merge field statistics",
                additional: current.field_stats.len(),
            })?;
        field_stats.extend_from_slice(&current.field_stats);
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation,
            expected_schema_id: manifest.schema_id,
            expected_docid_high_watermark: manifest.docid_high_watermark,
            schema: current.keeper.schema(),
            field_stats: field_stats.into_boxed_slice(),
            bm25_doc_count: current.bm25_doc_count,
            live_doc_count: current.live_doc_count,
            delta_live_doc_count: 0,
            deltas: Box::default(),
        })
    }

    fn install_prepared_sealed(
        &self,
        keeper: Arc<KeeperSnapshot>,
        prepared: PreparedSealedPublication,
    ) -> Arc<QuillSearchSnapshot> {
        let manifest = &keeper.loaded_manifest().manifest;
        assert_eq!(
            manifest.generation, prepared.expected_keeper_generation,
            "Keeper installed an unexpected MANIFEST generation after publication"
        );
        assert_eq!(
            manifest.schema_id, prepared.expected_schema_id,
            "Keeper installed an unexpected schema after publication"
        );
        assert_eq!(
            manifest.docid_high_watermark, prepared.expected_docid_high_watermark,
            "Keeper installed an unexpected Q1 watermark after publication"
        );
        assert!(
            keeper.schema() == prepared.schema,
            "Keeper installed a schema descriptor that differs from the prepared publication"
        );
        assert_eq!(
            keeper
                .at_seal_doc_count()
                .checked_add(prepared.delta_live_doc_count),
            Some(prepared.bm25_doc_count),
            "Keeper plus Delta BM25 count differs from the prepared publication"
        );
        assert_eq!(
            keeper
                .doc_count()
                .checked_add(prepared.delta_live_doc_count),
            Some(prepared.live_doc_count),
            "Keeper plus Delta live count differs from the prepared publication"
        );
        for expected in &prepared.field_stats {
            let keeper_stats = manifest
                .field_stats
                .iter()
                .find(|row| row.field_ord == expected.field_ord);
            let keeper_total_tokens = keeper_stats
                .map(|row| row.total_tokens)
                .or_else(|| manifest.segments.is_empty().then_some(0));
            let keeper_doc_count = keeper_stats
                .map(|row| u64::from(row.doc_count))
                .or_else(|| manifest.segments.is_empty().then_some(0));
            let delta_total_tokens = prepared.deltas.iter().fold(0_u64, |total, delta| {
                total
                    .checked_add(
                        delta
                            .live_total_tokens(expected.field_ord)
                            .expect("prepared Delta field statistics must remain complete"),
                    )
                    .expect("prepared composite field statistics must remain in range")
            });
            assert_eq!(
                keeper_total_tokens.and_then(|total| total.checked_add(delta_total_tokens)),
                Some(expected.total_tokens),
                "Keeper plus Delta token statistics differ from the prepared publication"
            );
            assert_eq!(
                keeper_doc_count.and_then(|count| count.checked_add(prepared.delta_live_doc_count)),
                Some(expected.doc_count),
                "Keeper plus Delta field document count differs from the prepared publication"
            );
        }

        let next = Arc::new(QuillSearchSnapshot {
            snapshot_epoch: prepared.snapshot_epoch,
            keeper,
            deltas: prepared.deltas,
            field_stats: prepared.field_stats,
            bm25_doc_count: prepared.bm25_doc_count,
            live_doc_count: prepared.live_doc_count,
        });
        self.current.store(Arc::clone(&next));
        next
    }

    /// Atomically replace Keeper and every shard Delta in one publication.
    ///
    /// This is the seal boundary: callers supply a complete authoritative set,
    /// already quiesced against batch writers. The method retries only to keep
    /// the local epoch monotone if another publication wins the CAS first.
    ///
    /// # Errors
    ///
    /// Returns the same composition failures as [`Self::new`], or epoch
    /// exhaustion.
    pub fn publish_complete(
        &self,
        mut keeper: Arc<KeeperSnapshot>,
        mut deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            validate_complete_keeper_transition(
                &current.keeper.loaded_manifest().manifest,
                &keeper.loaded_manifest().manifest,
            )?;
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let next = Arc::new(QuillSearchSnapshot::compose(epoch, keeper, deltas)?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            keeper = Arc::clone(&next.keeper);
            deltas = clone_delta_arcs(&next.deltas)?;
        }
    }

    /// Atomically replace one shard's frozen Delta epoch.
    ///
    /// Concurrent shard publications are merged through a compare-and-swap
    /// retry; each retry starts from the latest complete composite view.
    ///
    /// # Errors
    ///
    /// Rejects an unknown shard slot and all ordinary composition failures.
    pub fn publish_delta(
        &self,
        shard: usize,
        mut delta: Arc<DeltaSnapshot>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            if shard >= current.deltas.len() {
                return Err(SnapshotError::ShardOutOfBounds {
                    shard,
                    shard_count: current.deltas.len(),
                });
            }
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let mut deltas = clone_delta_arcs(&current.deltas)?;
            deltas[shard] = delta;
            let next = Arc::new(QuillSearchSnapshot::compose(
                epoch,
                Arc::clone(&current.keeper),
                deltas,
            )?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            delta = Arc::clone(&next.deltas[shard]);
        }
    }
}

fn clone_delta_arcs(
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<Vec<Arc<DeltaSnapshot>>, SnapshotError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(deltas.len())
        .map_err(|_| SnapshotError::Allocation {
            resource: "Delta snapshot table",
            additional: deltas.len(),
        })?;
    cloned.extend(deltas.iter().cloned());
    Ok(cloned)
}

#[derive(Debug)]
struct PendingIdentity {
    doc_ord: u32,
    document_id: String,
    content_hash: u64,
}

struct StagedFlush {
    shard: usize,
    encoded: EncodedSegment,
    manifest_segment: ManifestSegment,
    pending_field_stats: BTreeMap<u16, (u64, u32)>,
    next_seal_seq: u64,
}

struct BuiltShardFlush {
    shard: usize,
    encoded: EncodedSegment,
    manifest_segment: ManifestSegment,
    next_seal_seq: u64,
}

struct ShardFlushPlan<'a> {
    shard: usize,
    state: &'a ScribeShardState,
    segment_id: u64,
    lease_docid_base: u64,
    created_unix_s: i64,
    seal_seq: u64,
}

struct ScribeShardState {
    accumulator: ColumnarAccumulator,
    identities: Vec<PendingIdentity>,
    current_lease_base: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct ParallelShardAssignment {
    document_start: usize,
    document_end: usize,
    span: DocIdSpan,
}

#[derive(Debug, Clone, Copy)]
struct ParallelIngestReceipt {
    active_shards: usize,
    worker_count: usize,
    arena_bytes_used_high_water: usize,
    arena_bytes_reserved_high_water: usize,
}

fn parallel_ingest_active_shards(
    document_count: usize,
    configured_shards: usize,
    worker_count: usize,
) -> usize {
    configured_shards
        .min(worker_count)
        .min(document_count / PARALLEL_INGEST_MIN_DOCS_PER_SHARD)
}

#[derive(Default)]
struct QueryFuelState {
    consumed: AtomicU64,
    segments_touched: AtomicU64,
    dictionary_blocks: AtomicU64,
    posting_blocks: AtomicU64,
    position_docs: AtomicU64,
}

/// Refill strategy observed by the exact pruning-conformance witness.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConformancePruningStrategy {
    /// Full window scoring without a competitive cutoff.
    Exhaustive,
    /// Direct-term MaxScore candidate selection.
    MaxScore,
    /// Block-Max WAND candidate selection.
    BlockMaxWand,
}

/// One exact direct-term union refill from a conformance-only search.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformancePruningRefillReceipt {
    ordinal: u64,
    window_start: u32,
    horizon_end: u64,
    cutoff_bits: Option<u32>,
    strategy: ConformancePruningStrategy,
    candidate_docs: u64,
    buffer_empty: bool,
    live_work_remains: bool,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningRefillReceipt {
    /// One-based refill ordinal within the segment scorer.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Inclusive global document identifier that starts this window.
    #[must_use]
    pub const fn window_start(&self) -> u32 {
        self.window_start
    }

    /// Exclusive global document bound for this window.
    #[must_use]
    pub const fn horizon_end(&self) -> u64 {
        self.horizon_end
    }

    /// Exact competitive cutoff bits, or `None` before the heap is full.
    #[must_use]
    pub const fn cutoff_bits(&self) -> Option<u32> {
        self.cutoff_bits
    }

    /// Scoring strategy used for this refill.
    #[must_use]
    pub const fn strategy(&self) -> ConformancePruningStrategy {
        self.strategy
    }

    /// Number of documents admitted by the selected strategy.
    #[must_use]
    pub const fn candidate_docs(&self) -> u64 {
        self.candidate_docs
    }

    /// Whether the refill produced no buffered scored document.
    #[must_use]
    pub const fn buffer_empty(&self) -> bool {
        self.buffer_empty
    }

    /// Whether at least one posting cursor remains beyond this window.
    #[must_use]
    pub const fn live_work_remains(&self) -> bool {
        self.live_work_remains
    }
}

/// Ordered refill receipts for one immutable Quill segment.
#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformanceSegmentPruningReceipt {
    segment_ordinal: u64,
    segment_doc_count: u64,
    refills: Vec<ConformancePruningRefillReceipt>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformanceSegmentPruningReceipt {
    /// Stable ordinal in the committed Quill snapshot.
    #[must_use]
    pub const fn segment_ordinal(&self) -> u64 {
        self.segment_ordinal
    }

    /// Physical document cardinality of the scored segment.
    #[must_use]
    pub const fn segment_doc_count(&self) -> u64 {
        self.segment_doc_count
    }

    /// Ordered refill events observed while collecting this segment.
    #[must_use]
    pub fn refills(&self) -> &[ConformancePruningRefillReceipt] {
        &self.refills
    }
}

#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConformancePruningExecutionMode {
    /// The shipping collector scored sealed segments in snapshot order.
    Serial,
    /// The shipping collector scored sealed segments through Rayon fan-out.
    Rayon,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningExecutionMode {
    const SERIAL_CODE: u8 = 1;
    const RAYON_CODE: u8 = 2;

    const fn code(self) -> u8 {
        match self {
            Self::Serial => Self::SERIAL_CODE,
            Self::Rayon => Self::RAYON_CODE,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            Self::SERIAL_CODE => Some(Self::Serial),
            Self::RAYON_CODE => Some(Self::Rayon),
            _ => None,
        }
    }
}

#[cfg(feature = "pruning-conformance")]
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConformancePruningTraceReceipt {
    execution_mode: ConformancePruningExecutionMode,
    segments: Vec<ConformanceSegmentPruningReceipt>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceReceipt {
    /// Actual shipping collection branch used by this exact search invocation.
    #[must_use]
    pub const fn execution_mode(&self) -> ConformancePruningExecutionMode {
        self.execution_mode
    }

    /// Complete receipts in dense Quill segment order.
    #[must_use]
    pub fn segments(&self) -> &[ConformanceSegmentPruningReceipt] {
        &self.segments
    }
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
enum ConformancePruningTraceState {
    Collecting {
        expected_segments: usize,
        execution_mode: Option<ConformancePruningExecutionMode>,
        receipts: Vec<ConformanceSegmentPruningReceipt>,
    },
    Completed,
    Failed,
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
struct ConformancePruningTraceSession {
    state: StdMutex<ConformancePruningTraceState>,
    cancellation_arm_generation: Option<u64>,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceSession {
    #[cfg(test)]
    fn new(expected_segments: usize) -> Self {
        Self::new_for_cancellation_arm(expected_segments, None)
    }

    fn new_for_cancellation_arm(
        expected_segments: usize,
        cancellation_arm_generation: Option<u64>,
    ) -> Self {
        Self {
            state: StdMutex::new(ConformancePruningTraceState::Collecting {
                expected_segments,
                execution_mode: None,
                receipts: Vec::new(),
            }),
            cancellation_arm_generation,
        }
    }

    const fn cancellation_arm_generation(&self) -> Option<u64> {
        self.cancellation_arm_generation
    }

    fn bind_execution_mode(
        &self,
        execution_mode: ConformancePruningExecutionMode,
    ) -> Result<(), QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        match &mut *state {
            ConformancePruningTraceState::Collecting {
                execution_mode: slot,
                ..
            } if slot.is_none() => {
                *slot = Some(execution_mode);
                Ok(())
            }
            ConformancePruningTraceState::Collecting { .. } => {
                *state = ConformancePruningTraceState::Failed;
                Err(invalid_state(
                    "pruning-conformance execution mode was already bound",
                ))
            }
            ConformancePruningTraceState::Completed => Err(invalid_state(
                "pruning-conformance trace session is already completed",
            )),
            ConformancePruningTraceState::Failed => Err(invalid_state(
                "pruning-conformance trace session has failed",
            )),
        }
    }

    #[cfg(test)]
    fn record(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
    ) -> Result<u64, QuillIndexError> {
        self.record_inner(segment_ordinal, segment_doc_count, refills, None)
    }

    fn record_and_checkpoint(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
        controller: &ConformanceCancellationController,
        cx: &Cx,
    ) -> Result<u64, QuillIndexError> {
        self.record_inner(
            segment_ordinal,
            segment_doc_count,
            refills,
            Some((controller, cx)),
        )
    }

    fn record_inner(
        &self,
        segment_ordinal: u64,
        segment_doc_count: u64,
        refills: &[ConformanceUnionRefill],
        checkpoint: Option<(&ConformanceCancellationController, &Cx)>,
    ) -> Result<u64, QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        let ConformancePruningTraceState::Collecting {
            expected_segments,
            execution_mode,
            receipts,
        } = &mut *state
        else {
            return match &*state {
                ConformancePruningTraceState::Completed => Err(invalid_state(
                    "pruning-conformance trace session is already completed",
                )),
                ConformancePruningTraceState::Failed => Err(invalid_state(
                    "pruning-conformance trace session has failed",
                )),
                ConformancePruningTraceState::Collecting { .. } => {
                    Err(invalid_state("pruning-conformance trace state is invalid"))
                }
            };
        };
        let Some(execution_mode) = *execution_mode else {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(
                "pruning-conformance execution mode was not bound before recording",
            ));
        };
        let expected_segments = u64::try_from(*expected_segments)
            .map_err(|_| invalid_state("pruning-conformance segment count does not fit u64"))?;
        if segment_ordinal >= expected_segments {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(format!(
                "pruning-conformance segment ordinal {segment_ordinal} exceeds expected count \
                 {expected_segments}",
            )));
        }
        if receipts
            .iter()
            .any(|receipt| receipt.segment_ordinal == segment_ordinal)
        {
            *state = ConformancePruningTraceState::Failed;
            return Err(invalid_state(format!(
                "pruning-conformance segment ordinal {segment_ordinal} was recorded twice",
            )));
        }
        let refills = refills
            .iter()
            .map(|refill| ConformancePruningRefillReceipt {
                ordinal: refill.ordinal,
                window_start: refill.window_start,
                horizon_end: refill.horizon_end,
                cutoff_bits: refill.cutoff_bits,
                strategy: match refill.strategy {
                    ConformanceUnionRefillStrategy::Exhaustive => {
                        ConformancePruningStrategy::Exhaustive
                    }
                    ConformanceUnionRefillStrategy::MaxScore => {
                        ConformancePruningStrategy::MaxScore
                    }
                    ConformanceUnionRefillStrategy::BlockMaxWand => {
                        ConformancePruningStrategy::BlockMaxWand
                    }
                },
                candidate_docs: refill.candidate_docs,
                buffer_empty: refill.buffer_empty,
                live_work_remains: refill.live_work_remains,
            })
            .collect();
        receipts.push(ConformanceSegmentPruningReceipt {
            segment_ordinal,
            segment_doc_count,
            refills,
        });
        let recorded_receipts = match u64::try_from(receipts.len()) {
            Ok(recorded_receipts) => recorded_receipts,
            Err(_) => {
                *state = ConformancePruningTraceState::Failed;
                return Err(invalid_state(
                    "pruning-conformance recorded receipt count does not fit u64",
                ));
            }
        };
        if let Some((controller, cx)) = checkpoint
            && let Err(error) = controller.checkpoint_pruning_trace_segment_recorded(
                cx,
                recorded_receipts,
                self.cancellation_arm_generation,
                execution_mode,
            )
        {
            *state = ConformancePruningTraceState::Failed;
            return Err(error);
        }
        Ok(recorded_receipts)
    }

    fn complete(&self) -> Result<ConformancePruningTraceReceipt, QuillIndexError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_state("pruning-conformance receipt lock is poisoned"))?;
        let collecting = std::mem::replace(&mut *state, ConformancePruningTraceState::Failed);
        let ConformancePruningTraceState::Collecting {
            expected_segments,
            execution_mode,
            mut receipts,
        } = collecting
        else {
            return match collecting {
                ConformancePruningTraceState::Completed => Err(invalid_state(
                    "pruning-conformance trace session is already completed",
                )),
                ConformancePruningTraceState::Failed => Err(invalid_state(
                    "pruning-conformance trace session has failed",
                )),
                ConformancePruningTraceState::Collecting { .. } => {
                    Err(invalid_state("pruning-conformance trace state is invalid"))
                }
            };
        };
        let execution_mode = execution_mode
            .ok_or_else(|| invalid_state("pruning-conformance execution mode was never bound"))?;
        receipts.sort_by_key(|receipt| receipt.segment_ordinal);
        if receipts.len() != expected_segments
            || !receipts.iter().enumerate().all(|(ordinal, receipt)| {
                u64::try_from(ordinal).is_ok_and(|ordinal| ordinal == receipt.segment_ordinal)
            })
        {
            return Err(invalid_state(format!(
                "pruning-conformance trace is incomplete: expected {expected_segments} dense \
                 segment receipts, observed {}",
                receipts.len(),
            )));
        }
        *state = ConformancePruningTraceState::Completed;
        Ok(ConformancePruningTraceReceipt {
            execution_mode,
            segments: receipts,
        })
    }

    fn fail(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = ConformancePruningTraceState::Failed;
        }
    }
}

#[cfg(feature = "pruning-conformance")]
#[derive(Debug)]
struct ConformancePruningTraceGuard {
    session: Arc<ConformancePruningTraceSession>,
    controller: Arc<ConformanceCancellationController>,
    finished: bool,
}

#[cfg(feature = "pruning-conformance")]
impl ConformancePruningTraceGuard {
    fn new(expected_segments: usize, controller: Arc<ConformanceCancellationController>) -> Self {
        let cancellation_arm_generation = controller.active_pruning_trace_arm_generation();
        Self {
            session: Arc::new(ConformancePruningTraceSession::new_for_cancellation_arm(
                expected_segments,
                cancellation_arm_generation,
            )),
            controller,
            finished: false,
        }
    }

    fn session(&self) -> &ConformancePruningTraceSession {
        self.session.as_ref()
    }

    fn complete(mut self) -> Result<ConformancePruningTraceReceipt, QuillIndexError> {
        let receipt = self.session.complete()?;
        self.finished = true;
        Ok(receipt)
    }
}

#[cfg(feature = "pruning-conformance")]
impl Drop for ConformancePruningTraceGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.session.fail();
            if let Some(arm_generation) = self.session.cancellation_arm_generation() {
                self.controller
                    .note_discarded_pruning_trace_session(arm_generation);
            }
        }
    }
}

/// Deterministic request-cancellation checkpoints exposed only to the
/// conformance harness.
///
/// Synchronous Quill collection cannot be preempted from an external test
/// task at a reproducible work unit. The conformance controller therefore
/// requests cancellation on the real [`Cx`] from an exact engine checkpoint;
/// the public method must still observe that request through its normal
/// cancellation poll and return its own typed outcome.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConformanceCancellationStage {
    /// One admitted collection work unit.
    QueryCollection = 1,
    /// One deferred-metadata candidate before materialization.
    FusionHydration = 2,
    /// The retained commit transaction immediately before publication.
    CommitPublication = 3,
    /// One sealed-segment pruning receipt has been recorded successfully.
    PruningTraceSegmentRecorded = 4,
}

/// Fixed-size conformance receipt for the complete retained scalar writer
/// transaction.
///
/// The digest binds canonical pending segment bytes and identities, staged
/// flush state, retained MANIFEST proposals, dirty shard document witnesses,
/// allocator leases, and all publication-relevant flags. Summary counters stay
/// explicit so a receipt cannot hide a missing state class behind an opaque
/// digest. This type and the hashing work do not exist in shipping builds.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformancePendingWriterState {
    digest_sha256: [u8; 32],
    shard_count: u64,
    dirty_shard_count: u64,
    pending_identity_count: u64,
    uncommitted_id_count: u64,
    pending_segment_count: u64,
    pending_owned_segment_count: u64,
    flags: u8,
}

#[cfg(feature = "conformance-internals")]
impl ConformancePendingWriterState {
    const STAGED_FLUSH_PRESENT: u8 = 1 << 0;
    const PENDING_MANIFEST_PRESENT: u8 = 1 << 1;
    const PENDING_REPLACEMENT_MANIFEST_PRESENT: u8 = 1 << 2;
    const PENDING_DELTA_SEAL_PRESENT: u8 = 1 << 3;
    const UNPUBLISHED_SINCE_PRESENT: u8 = 1 << 4;
    const INGEST_RETRY_REQUIRED: u8 = 1 << 5;

    const fn has_flag(self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    /// SHA-256 over the canonical complete pending writer transaction.
    #[must_use]
    pub const fn digest_sha256(self) -> [u8; 32] {
        self.digest_sha256
    }

    /// Number of configured scalar writer shards represented by the digest.
    #[must_use]
    pub const fn shard_count(self) -> u64 {
        self.shard_count
    }

    /// Number of shards retaining documents, identities, or an active lease.
    #[must_use]
    pub const fn dirty_shard_count(self) -> u64 {
        self.dirty_shard_count
    }

    /// Canonical document identity rows still resident in dirty shards.
    #[must_use]
    pub const fn pending_identity_count(self) -> u64 {
        self.pending_identity_count
    }

    /// Stable document IDs reserved by all unpublished scalar state.
    #[must_use]
    pub const fn uncommitted_id_count(self) -> u64 {
        self.uncommitted_id_count
    }

    /// Installed but unpublished MANIFEST segment rows.
    #[must_use]
    pub const fn pending_segment_count(self) -> u64 {
        self.pending_segment_count
    }

    /// Canonical in-memory FSLX buffers retained for pending segments.
    #[must_use]
    pub const fn pending_owned_segment_count(self) -> u64 {
        self.pending_owned_segment_count
    }

    /// Whether a flush transaction has been encoded but not fully installed.
    #[must_use]
    pub const fn staged_flush_present(self) -> bool {
        self.has_flag(Self::STAGED_FLUSH_PRESENT)
    }

    /// Whether the exact next ordinary MANIFEST proposal is retained.
    #[must_use]
    pub const fn pending_manifest_present(self) -> bool {
        self.has_flag(Self::PENDING_MANIFEST_PRESENT)
    }

    /// Whether a complete replacement MANIFEST proposal is retained.
    #[must_use]
    pub const fn pending_replacement_manifest_present(self) -> bool {
        self.has_flag(Self::PENDING_REPLACEMENT_MANIFEST_PRESENT)
    }

    /// Whether a Delta-seal publication transaction is retained.
    #[must_use]
    pub const fn pending_delta_seal_present(self) -> bool {
        self.has_flag(Self::PENDING_DELTA_SEAL_PRESENT)
    }

    /// Whether the writer is tracking an unpublished visibility interval.
    #[must_use]
    pub const fn unpublished_since_present(self) -> bool {
        self.has_flag(Self::UNPUBLISHED_SINCE_PRESENT)
    }

    /// Whether scalar ingest requires reconciliation before more mutation.
    #[must_use]
    pub const fn ingest_retry_required(self) -> bool {
        self.has_flag(Self::INGEST_RETRY_REQUIRED)
    }
}

#[cfg(feature = "conformance-internals")]
impl ConformanceCancellationStage {
    const fn code(self) -> u8 {
        match self {
            Self::QueryCollection => 1,
            Self::FusionHydration => 2,
            Self::CommitPublication => 3,
            Self::PruningTraceSegmentRecorded => 4,
        }
    }
}

/// Per-index deterministic cancellation requester for live conformance tests.
///
/// This type, its atomics, and every call site are absent unless the dedicated
/// `conformance-internals` feature is enabled. It cannot fabricate a return
/// value: firing only marks the real request [`Cx`] as cancelled.
#[cfg(feature = "conformance-internals")]
#[derive(Debug, Default)]
pub struct ConformanceCancellationController {
    stage: AtomicU8,
    arm_generation: AtomicU64,
    arm_transition_lock: StdMutex<()>,
    trigger_ordinal: AtomicU64,
    observed_checkpoints: AtomicU64,
    recorded_pruning_receipts_at_fire: AtomicU64,
    #[cfg(feature = "pruning-conformance")]
    pruning_trace_execution_mode_at_fire: AtomicU8,
    discarded_pruning_trace_sessions: AtomicU64,
    fired: AtomicBool,
}

#[cfg(feature = "conformance-internals")]
impl ConformanceCancellationController {
    const DISARMED: u8 = 0;
    const ARMING: u8 = u8::MAX;

    /// Arm one exact nonzero checkpoint ordinal.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error for ordinal zero or when a prior stage
    /// remains armed.
    pub fn arm(
        &self,
        stage: ConformanceCancellationStage,
        trigger_ordinal: u64,
    ) -> Result<(), QuillIndexError> {
        if trigger_ordinal == 0 {
            return Err(invalid_state(
                "conformance cancellation trigger ordinal must be nonzero",
            ));
        }
        let arm_transition = self.arm_transition_lock.lock().map_err(|_| {
            invalid_state("conformance cancellation arm transition lock is poisoned")
        })?;
        self.stage
            .compare_exchange(
                Self::DISARMED,
                Self::ARMING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_state("conformance cancellation controller is already armed"))?;
        let Some(arm_generation) = self.arm_generation.load(Ordering::Relaxed).checked_add(1)
        else {
            self.stage.store(Self::DISARMED, Ordering::Release);
            return Err(invalid_state(
                "conformance cancellation arm generation is exhausted",
            ));
        };
        self.arm_generation.store(arm_generation, Ordering::Relaxed);
        self.trigger_ordinal
            .store(trigger_ordinal, Ordering::Relaxed);
        self.observed_checkpoints.store(0, Ordering::Relaxed);
        self.recorded_pruning_receipts_at_fire
            .store(0, Ordering::Relaxed);
        #[cfg(feature = "pruning-conformance")]
        self.pruning_trace_execution_mode_at_fire
            .store(0, Ordering::Relaxed);
        self.discarded_pruning_trace_sessions
            .store(0, Ordering::Relaxed);
        self.fired.store(false, Ordering::Relaxed);
        self.stage.store(stage.code(), Ordering::Release);
        drop(arm_transition);
        Ok(())
    }

    /// Stop injecting checkpoints without changing request cancellation.
    ///
    /// The caller must clear the real [`Cx`] explicitly. Keeping those actions
    /// separate proves the request context remains the cancellation authority.
    pub fn disarm(&self) {
        let _arm_transition = self.arm_transition_lock.lock().ok();
        self.stage.store(Self::DISARMED, Ordering::Release);
    }

    /// Number of matching stage checkpoints observed by the current arm.
    #[must_use]
    pub fn observed_checkpoints(&self) -> u64 {
        self.observed_checkpoints.load(Ordering::Acquire)
    }

    /// Successfully recorded pruning receipts at the exact firing boundary.
    ///
    /// This stays zero for every non-pruning cancellation stage. A nonzero
    /// value proves the pruning checkpoint ran only after receipt mutation
    /// succeeded; it does not infer how much unrecorded scorer work remained.
    #[must_use]
    pub fn recorded_pruning_receipts_at_fire(&self) -> u64 {
        self.recorded_pruning_receipts_at_fire
            .load(Ordering::Acquire)
    }

    /// Shipping sealed-segment execution branch at the pruning fire boundary.
    #[cfg(feature = "pruning-conformance")]
    #[must_use]
    pub fn pruning_trace_execution_mode_at_fire(&self) -> Option<ConformancePruningExecutionMode> {
        ConformancePruningExecutionMode::from_code(
            self.pruning_trace_execution_mode_at_fire
                .load(Ordering::Acquire),
        )
    }

    /// Invocation-bound pruning trace sessions failed and discarded since arm.
    #[must_use]
    pub fn discarded_pruning_trace_sessions(&self) -> u64 {
        self.discarded_pruning_trace_sessions
            .load(Ordering::Acquire)
    }

    /// Whether the current arm requested cancellation on its real [`Cx`].
    #[must_use]
    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::Acquire)
    }

    fn is_armed(&self) -> bool {
        self.stage.load(Ordering::Acquire) != Self::DISARMED
    }

    fn checkpoint(&self, stage: ConformanceCancellationStage, cx: &Cx) {
        self.checkpoint_with_pruning_receipts(stage, cx, None, None);
    }

    #[cfg(feature = "pruning-conformance")]
    fn checkpoint_pruning_trace_segment_recorded(
        &self,
        cx: &Cx,
        recorded_receipts: u64,
        expected_arm_generation: Option<u64>,
        execution_mode: ConformancePruningExecutionMode,
    ) -> Result<(), QuillIndexError> {
        if recorded_receipts == 0 {
            return Err(invalid_state(
                "pruning-trace cancellation checkpoint requires a successful receipt",
            ));
        }
        let Some(expected_arm_generation) = expected_arm_generation else {
            return Ok(());
        };
        let _arm_transition = self.arm_transition_lock.lock().map_err(|_| {
            invalid_state("conformance cancellation arm transition lock is poisoned")
        })?;
        if self.stage.load(Ordering::Acquire)
            != ConformanceCancellationStage::PruningTraceSegmentRecorded.code()
            || self.arm_generation.load(Ordering::Acquire) != expected_arm_generation
        {
            return Ok(());
        }
        self.checkpoint_with_pruning_receipts(
            ConformanceCancellationStage::PruningTraceSegmentRecorded,
            cx,
            Some(recorded_receipts),
            Some(execution_mode.code()),
        );
        Ok(())
    }

    fn checkpoint_with_pruning_receipts(
        &self,
        stage: ConformanceCancellationStage,
        cx: &Cx,
        recorded_receipts: Option<u64>,
        pruning_execution_mode_code: Option<u8>,
    ) {
        if self.stage.load(Ordering::Acquire) != stage.code() {
            return;
        }
        let ordinal = self
            .observed_checkpoints
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        if ordinal != self.trigger_ordinal.load(Ordering::Acquire) {
            return;
        }
        if let Some(recorded_receipts) = recorded_receipts {
            self.recorded_pruning_receipts_at_fire
                .store(recorded_receipts, Ordering::Release);
        }
        #[cfg(feature = "pruning-conformance")]
        if let Some(execution_mode_code) = pruning_execution_mode_code {
            self.pruning_trace_execution_mode_at_fire
                .store(execution_mode_code, Ordering::Release);
        }
        #[cfg(not(feature = "pruning-conformance"))]
        let _ = pruning_execution_mode_code;
        self.fired.store(true, Ordering::Release);
        tracing::info!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.conformance.cancellation_checkpoint",
            ?stage,
            ordinal,
            "deterministic conformance checkpoint requested cancellation on the real Cx"
        );
        cx.set_cancel_requested(true);
    }

    #[cfg(feature = "pruning-conformance")]
    fn active_pruning_trace_arm_generation(&self) -> Option<u64> {
        let _arm_transition = self.arm_transition_lock.lock().ok()?;
        (self.stage.load(Ordering::Acquire)
            == ConformanceCancellationStage::PruningTraceSegmentRecorded.code())
        .then(|| self.arm_generation.load(Ordering::Acquire))
    }

    #[cfg(feature = "pruning-conformance")]
    fn note_discarded_pruning_trace_session(&self, expected_arm_generation: u64) {
        let Ok(_arm_transition) = self.arm_transition_lock.lock() else {
            return;
        };
        if self.stage.load(Ordering::Acquire)
            == ConformanceCancellationStage::PruningTraceSegmentRecorded.code()
            && self.arm_generation.load(Ordering::Acquire) == expected_arm_generation
        {
            self.discarded_pruning_trace_sessions
                .fetch_add(1, Ordering::AcqRel);
        }
    }
}

struct QueryCheckpoint<'a> {
    cx: &'a Cx,
    phase: &'static str,
    budget: u64,
    metering: bool,
    state: QueryFuelState,
    #[cfg(feature = "conformance-internals")]
    conformance_controller: Arc<ConformanceCancellationController>,
}

impl<'a> QueryCheckpoint<'a> {
    #[cfg(any(not(feature = "conformance-internals"), test))]
    fn new(cx: &'a Cx, phase: &'static str, budget: u64, upper_bound: u64) -> Arc<Self> {
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            #[cfg(feature = "conformance-internals")]
            conformance_controller: Arc::default(),
        })
    }

    #[cfg(feature = "conformance-internals")]
    fn new_with_controller(
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
        conformance_controller: Arc<ConformanceCancellationController>,
    ) -> Arc<Self> {
        Arc::new(Self {
            cx,
            phase,
            budget,
            metering: upper_bound > budget,
            state: QueryFuelState::default(),
            conformance_controller,
        })
    }

    const fn metering(&self) -> bool {
        self.metering
    }

    fn exhausted(&self, consumed: u64) -> ArgusError {
        let segments_touched = self.state.segments_touched.load(Ordering::Acquire);
        let dictionary_blocks = self.state.dictionary_blocks.load(Ordering::Acquire);
        let posting_blocks = self.state.posting_blocks.load(Ordering::Acquire);
        let position_docs = self.state.position_docs.load(Ordering::Acquire);
        tracing::warn!(
            target: crate::tracing_conventions::TARGET,
            event = "quill.argus.query_fuel_exhausted",
            phase = self.phase,
            budget = self.budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
            "Quill query exhausted its deterministic coarse work budget"
        );
        ArgusError::QueryFuelExhausted {
            budget: self.budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
        }
    }
}

impl QueryWorkCheckpoint for QueryCheckpoint<'_> {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        #[cfg(feature = "conformance-internals")]
        self.conformance_controller
            .checkpoint(ConformanceCancellationStage::QueryCollection, self.cx);
        if self.cx.is_cancel_requested() {
            return Err(ArgusError::QueryCancelled { phase: self.phase });
        }
        if units == 0 || !self.metering {
            return Ok(());
        }
        let admitted =
            self.state
                .consumed
                .try_update(Ordering::AcqRel, Ordering::Acquire, |consumed| {
                    consumed
                        .checked_add(units)
                        .filter(|next| *next <= self.budget)
                });
        let previous = match admitted {
            Ok(previous) => previous,
            Err(consumed) => return Err(self.exhausted(consumed)),
        };
        debug_assert!(previous.saturating_add(units) <= self.budget);
        let counter = match kind {
            QueryWorkKind::Segment => &self.state.segments_touched,
            QueryWorkKind::DictionaryBlock => &self.state.dictionary_blocks,
            QueryWorkKind::PostingBlock => &self.state.posting_blocks,
            QueryWorkKind::PositionDocument => &self.state.position_docs,
        };
        let _ = counter.fetch_add(units, Ordering::Relaxed);
        Ok(())
    }
}

type QueryCheckpointHandle<'a> = Arc<dyn QueryWorkCheckpoint + 'a>;

#[derive(Default)]
struct QueryWorkShape {
    posting_streams: u64,
    dictionary_scans: u64,
    phrase_streams: u64,
    phrase_fields: u64,
}

impl QueryWorkShape {
    fn visit(&mut self, query: &Query, glob_expansion_limit: usize) {
        match query {
            Query::Empty | Query::All => {}
            Query::Term { fields, .. } => {
                let fields = u64::try_from(fields.len()).unwrap_or(u64::MAX);
                self.posting_streams = self.posting_streams.saturating_add(fields);
            }
            Query::Phrase { fields, terms, .. } => {
                let fields = u64::try_from(fields.len()).unwrap_or(u64::MAX);
                let terms = u64::try_from(terms.len()).unwrap_or(u64::MAX);
                let streams = fields.saturating_mul(terms);
                self.posting_streams = self.posting_streams.saturating_add(streams);
                self.phrase_streams = self.phrase_streams.saturating_add(streams);
                self.phrase_fields = self.phrase_fields.saturating_add(fields);
            }
            Query::Boolean { clauses, .. } => {
                for clause in clauses {
                    self.visit(&clause.query, glob_expansion_limit);
                }
            }
            Query::Boost { query, .. } => self.visit(query, glob_expansion_limit),
            Query::Range { lower, upper, .. } => {
                let string_range = matches!(
                    lower,
                    Bound::Included(QueryValue::Str(_)) | Bound::Excluded(QueryValue::Str(_))
                ) || matches!(
                    upper,
                    Bound::Included(QueryValue::Str(_)) | Bound::Excluded(QueryValue::Str(_))
                );
                if string_range {
                    self.dictionary_scans = self.dictionary_scans.saturating_add(1);
                    self.posting_streams = self
                        .posting_streams
                        .saturating_add(u64::try_from(glob_expansion_limit).unwrap_or(u64::MAX));
                }
            }
            Query::Glob { field_ids, .. } => {
                let fields = u64::try_from(field_ids.len()).unwrap_or(u64::MAX);
                self.dictionary_scans = self.dictionary_scans.saturating_add(fields);
                self.posting_streams = self.posting_streams.saturating_add(
                    fields.saturating_mul(u64::try_from(glob_expansion_limit).unwrap_or(u64::MAX)),
                );
            }
            Query::Set { values, .. } => {
                let strings = values
                    .iter()
                    .filter(|value| matches!(value, QueryValue::Str(_)))
                    .count();
                let strings = u64::try_from(strings).unwrap_or(u64::MAX);
                self.posting_streams = self.posting_streams.saturating_add(strings);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecycleTrigger {
    ArenaBudget,
    LeaseBoundary,
    VisibilityLag,
    ExplicitFlush,
    BulkCadence,
    BulkFinish,
    TierFanout,
}

impl LifecycleTrigger {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ArenaBudget => "arena_budget",
            Self::LeaseBoundary => "lease_boundary",
            Self::VisibilityLag => "visibility_lag",
            Self::ExplicitFlush => "explicit_flush",
            Self::BulkCadence => "bulk_cadence",
            Self::BulkFinish => "bulk_finish",
            Self::TierFanout => "tier_fanout",
        }
    }
}

struct PendingDeltaSeal {
    encoded: Option<Arc<EncodedSegment>>,
    segment_installed: bool,
    manifest: Manifest,
    prepared: PreparedSealedPublication,
    next_seal_seq: u64,
    successor_watermark: u64,
}

#[derive(Clone)]
struct QuillReader {
    config: QuillConfig,
    schema: SchemaDescriptor,
    parser: Option<DefaultQueryParser>,
    published_snapshot: Arc<SnapshotPublisher>,
    #[cfg(feature = "conformance-internals")]
    conformance_controller: Arc<ConformanceCancellationController>,
}

/// Shared Quill index handle with lock-free readers and one cancel-aware writer.
///
/// The same handle implements [`LexicalSearch`] for progressive fusion and
/// exposes synchronous reader methods for direct and synchronous consumers.
///
/// ```no_run
/// # use asupersync::Cx;
/// # use frankensearch_core::IndexableDocument;
/// # use frankensearch_quill::{
/// #     CompactionPolicy, QuillConfig, QuillIndex, SegmentStatsProvider, SnippetConfig,
/// # };
/// # async fn use_quill(cx: &Cx) -> Result<(), Box<dyn std::error::Error>> {
/// let index = QuillIndex::in_memory(QuillConfig::default())?;
/// index
///     .index_document(cx, &IndexableDocument::new("guide", "native lexical search"))
///     .await?;
/// index.commit(cx).await?;
///
/// assert!(index.path().is_none());
/// assert_eq!(index.doc_count(), 1);
/// let page = index.search_paginated(cx, "lexical", 10, 0, true)?;
/// let full = index.search_results(cx, "lexical", 10)?;
/// let ids = index.search_doc_ids(cx, "lexical", 10)?;
/// let enriched = index.search_with_snippets(cx, "lexical", 10, &SnippetConfig::default())?;
/// let stats = index.segment_stats();
/// assert_eq!(page.total_count, Some(1));
/// assert_eq!(full.len(), ids.len());
/// assert_eq!(enriched.len(), ids.len());
/// assert_eq!(stats.live_docs, 1);
///
/// let _ = index.compact(cx, CompactionPolicy::default()).await?;
/// assert!(index.delete_document(cx, "guide").await?);
/// index.delete_all(cx).await?;
///
/// let path = "target/quill-doc-example";
/// let _created = QuillIndex::create(cx, path, QuillConfig::default()).await?;
/// let _opened = QuillIndex::open(cx, path, QuillConfig::default()).await?;
/// # Ok(())
/// # }
/// ```
pub struct QuillIndex {
    reader: QuillReader,
    writer: Arc<Mutex<QuillWriterState>>,
    directory: Option<PathBuf>,
}

/// Read-only Quill handle for consumers that must coexist with another
/// process holding the durable writer lease.
///
/// Opening this handle pins the currently published MANIFEST without creating
/// or acquiring `LOCK`. A fresh handle therefore observes the latest
/// cross-process publication while a live watcher remains the sole writer.
#[derive(Clone)]
pub struct QuillSearchIndex {
    reader: QuillReader,
    directory: PathBuf,
}

/// Read-only in-memory handle for conformance schemas with their own parser.
///
/// This type intentionally exposes only typed-query execution. It neither
/// constructs Scribe writer shards nor offers string-query or mutation APIs,
/// so a schema whose analyzer and query-language contracts differ from the
/// shipping defaults cannot accidentally enter those paths.
#[cfg(feature = "bench-internals")]
#[derive(Clone)]
pub struct PreparsedQuillIndex {
    reader: QuillReader,
}

/// Scalar Quill writer state guarded by [`QuillIndex::writer`].
struct QuillWriterState {
    reader: QuillReader,
    backend: IndexBackend,
    shards: Vec<ScribeShardState>,
    shard_router: ShardRouter,
    docid_allocator: DocIdAllocator,
    uncommitted_ids: BTreeSet<String>,
    next_lease_base: u64,
    next_seal_seq: u64,
    staged_flush: Option<StagedFlush>,
    pending_segments: Vec<ManifestSegment>,
    pending_owned_segments: Vec<EncodedSegment>,
    pending_field_stats: BTreeMap<u16, (u64, u32)>,
    pending_manifest: Option<Manifest>,
    pending_replacement_manifest: Option<Manifest>,
    pending_delta_seal: Option<PendingDeltaSeal>,
    unpublished_since: Option<Instant>,
    /// A prior scalar ingest returned an error after mutation may have begun.
    ///
    /// Successfully installed segments may span multiple bounded batches, but
    /// an ambiguous partial batch must be reconciled by `commit` before any
    /// further scalar indexing is accepted.
    ingest_retry_required: bool,
}

impl Deref for QuillWriterState {
    type Target = QuillReader;

    fn deref(&self) -> &Self::Target {
        &self.reader
    }
}

enum IndexBackend {
    Durable(KeeperWriter),
    Memory(KeeperSnapshot),
}

impl IndexBackend {
    const fn snapshot(&self) -> &KeeperSnapshot {
        match self {
            Self::Durable(writer) => writer.snapshot(),
            Self::Memory(snapshot) => snapshot,
        }
    }
}

impl QuillWriterState {
    /// Create a shipping-schema index or open an existing compatible index.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Self::create_with_schema(cx, directory, DEFAULT_SCHEMA, config).await
    }

    /// Open an existing shipping-schema index for mutation and search.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer = KeeperWriter::open(cx, directory, DEFAULT_SCHEMA).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Open an existing shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, or durability
    /// failures.
    #[cfg(feature = "durability")]
    pub async fn open_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = true,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let unrepairable = if config.quarantine_on_unrepairable {
                UnrepairableSegmentPolicy::Quarantine
            } else {
                UnrepairableSegmentPolicy::FailClosed
            };
            let writer = KeeperWriter::open_durable_with_policy(
                cx,
                directory,
                DEFAULT_SCHEMA,
                protector,
                unrepairable,
            )
            .await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Create or open a shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, or durability
    /// failures.
    #[cfg(feature = "durability")]
    pub async fn create_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = true,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let unrepairable = if config.quarantine_on_unrepairable {
                UnrepairableSegmentPolicy::Quarantine
            } else {
                UnrepairableSegmentPolicy::FailClosed
            };
            let writer = KeeperWriter::create_durable_with_policy(
                cx,
                directory,
                DEFAULT_SCHEMA,
                protector,
                unrepairable,
            )
            .await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    async fn create_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer = KeeperWriter::create(cx, directory, schema).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), schema, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Construct an owned-buffer genesis index without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    pub fn in_memory(config: QuillConfig) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let _open_entered = open_span.enter();
        let snapshot = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
        let index = Self::from_backend(IndexBackend::Memory(snapshot), DEFAULT_SCHEMA, config)?;
        record_snapshot_fields(&open_span, index.snapshot());
        Ok(index)
    }

    /// Build an owned-buffer index for one explicit compile-time schema.
    ///
    /// This constructor is intentionally limited to conformance and benchmark
    /// builds. It lets the mixed-residency gauntlet exercise the same private
    /// schema binding used by durable opens without expanding the shipping API.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or genesis-snapshot failures.
    #[cfg(feature = "bench-internals")]
    pub fn in_memory_with_schema(
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let snapshot = KeeperSnapshot::in_memory(schema)?;
        Self::from_backend(IndexBackend::Memory(snapshot), schema, config)
    }

    /// Bind an existing owned Keeper snapshot to the private writer facade.
    ///
    /// The conformance gauntlet uses this to replay identical tombstone and
    /// field-stat history before varying only current document residency.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, parser/schema, or snapshot failures.
    #[cfg(feature = "bench-internals")]
    pub fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let schema = snapshot.schema();
        Self::from_backend(IndexBackend::Memory(snapshot), schema, config)
    }

    fn from_backend(
        backend: IndexBackend,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        let parser = Some(DefaultQueryParser::new(schema)?);
        let manifest = &backend.snapshot().loaded_manifest().manifest;
        let next_lease_base = next_lease_boundary(manifest.docid_high_watermark)?;
        let detected_parallelism = std::thread::available_parallelism().map_or(1, usize::from);
        let shard_router = ShardRouter::from_config(&config, detected_parallelism);
        let docid_allocator = DocIdAllocator::open(next_lease_base, shard_router.shard_count())
            .map_err(|error| invalid_state(error.to_string()))?;
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(shard_router.shard_count())
            .map_err(|_| invalid_state("could not allocate Scribe shard table"))?;
        for _ in 0..shard_router.shard_count() {
            shards.push(ScribeShardState {
                accumulator: ColumnarAccumulator::new(schema)?,
                identities: Vec::new(),
                current_lease_base: None,
            });
        }
        let next_seal_seq = manifest
            .segments
            .iter()
            .map(|segment| segment.seal_seq)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        let published_snapshot = Arc::new(SnapshotPublisher::new(
            Arc::new(backend.snapshot().clone()),
            Vec::new(),
        )?);
        Ok(Self {
            reader: QuillReader {
                config,
                schema,
                parser,
                published_snapshot,
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
            backend,
            shards,
            shard_router,
            docid_allocator,
            uncommitted_ids: BTreeSet::new(),
            next_lease_base,
            next_seal_seq,
            staged_flush: None,
            pending_segments: Vec::new(),
            pending_owned_segments: Vec::new(),
            pending_field_stats: BTreeMap::new(),
            pending_manifest: None,
            pending_replacement_manifest: None,
            pending_delta_seal: None,
            unpublished_since: None,
            ingest_retry_required: false,
        })
    }

    /// Current committed immutable snapshot.
    #[must_use]
    pub const fn snapshot(&self) -> &KeeperSnapshot {
        self.backend.snapshot()
    }

    /// Atomically publish the complete process-local Delta table for the
    /// current durable Keeper generation.
    ///
    /// Batch writers call this after freezing every shard at one quiescent
    /// boundary. A same-generation publication changes only process-local
    /// visibility; cross-process readers continue to observe the MANIFEST.
    ///
    /// # Errors
    ///
    /// Rejects schema/generation drift, overlapping leases, Keeper-range
    /// overlap, allocation failure, and process-local epoch exhaustion.
    pub fn publish_delta_table(
        &self,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "resume the retained Delta seal before publishing another Delta table",
            ));
        }
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "Delta publication requires the scalar writer to be fully committed",
            ));
        }
        Ok(self
            .published_snapshot
            .publish_complete(Arc::new(self.backend.snapshot().clone()), deltas)?)
    }

    /// Seal one already-published Delta epoch into Keeper and atomically
    /// replace the complete process-local Delta table.
    ///
    /// The frozen source remains retained through FSLX construction, segment
    /// fsync/install, and MANIFEST publication. The final `ArcSwap` installs the
    /// new Keeper plus all replacement Delta epochs in one non-cancellable
    /// step; only then may this method release its source Arc. Consequently a
    /// reader can observe the document in the old Delta or the new Keeper, but
    /// never a visibility gap between them.
    ///
    /// `replacement_deltas` must already be frozen against the successor
    /// MANIFEST generation. Multi-shard callers must rebind every surviving
    /// shard epoch through [`DeltaSnapshot::rebind_keeper_generation`], not
    /// only the shard being sealed; omission is rejected before FSLX work.
    ///
    /// # Errors
    ///
    /// Rejects scalar pending state, an unpublished source epoch, invalid
    /// replacement epochs, cancellation before durable authority changes, or
    /// any FSLX/Keeper publication failure. An installed-but-unreferenced
    /// segment remains invisible and is safe for exact retry.
    pub async fn seal_delta_snapshot(
        &mut self,
        cx: &Cx,
        sealed: Arc<DeltaSnapshot>,
        replacement_deltas: Vec<Arc<DeltaSnapshot>>,
        input: DeltaFlushInput,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        check_cancel(cx, "Delta seal")?;
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "a prior Delta seal requires resume_pending_delta_seal before new mutation",
            ));
        }
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "Delta seal requires the scalar writer to be fully committed",
            ));
        }
        let current = self.published_snapshot.load();
        if !current
            .delta_snapshots()
            .iter()
            .any(|delta| Arc::ptr_eq(delta, &sealed))
        {
            return Err(invalid_state(
                "Delta seal source is not present in the published process-local epoch",
            ));
        }
        if sealed.keeper_generation() != current.keeper_generation() {
            return Err(invalid_state(
                "Delta seal source is stale relative to the published Keeper",
            ));
        }
        validate_delta_seal_replacements(&current, &sealed, &replacement_deltas)?;

        let build_source = Arc::clone(&sealed);
        let encoded = spawn_blocking(move || flush_delta_snapshot(&build_source, input))
            .await?
            .map(Arc::new);
        check_cancel(cx, "Delta segment install")?;

        let mut manifest = self.backend.snapshot().next_manifest()?;
        // Keeper owns the one authoritative publish timestamp. Retaining a
        // zero here also lets an ambiguous retry compare the exact logical
        // proposal after Keeper normalizes its installed timestamp.
        manifest.last_publish_unix_s = 0;
        let mut next_seal_seq = self.next_seal_seq;
        if let Some(encoded) = &encoded {
            manifest
                .segments
                .push(manifest_segment(encoded, next_seal_seq));
            manifest
                .segments
                .sort_unstable_by_key(|segment| segment.docid_lo);
            next_seal_seq = next_seal_seq
                .checked_add(1)
                .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        }
        let successor_watermark = replacement_deltas.iter().fold(
            manifest.docid_high_watermark.max(sealed.lease_end()),
            |watermark, delta| watermark.max(delta.lease_end()),
        );
        manifest.docid_high_watermark = successor_watermark;
        let live_document_count = u32::try_from(sealed.live_document_count())
            .map_err(|_| invalid_state("Delta live document count does not fit u32"))?;
        let mut pending_field_stats = BTreeMap::new();
        if live_document_count != 0 {
            for field in
                self.schema.fields.iter().filter(|field| {
                    matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. })
                })
            {
                let total_tokens = sealed.live_total_tokens(field.id).ok_or_else(|| {
                    invalid_state(format!(
                        "Delta seal omitted indexed field {} statistics",
                        field.id
                    ))
                })?;
                pending_field_stats.insert(field.id, (total_tokens, live_document_count));
            }
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)?;
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest_with_deltas(self.schema, &manifest, replacement_deltas)?;
        self.pending_delta_seal = Some(PendingDeltaSeal {
            encoded,
            segment_installed: false,
            manifest,
            prepared,
            next_seal_seq,
            successor_watermark,
        });
        drop(sealed);
        self.finish_pending_delta_seal(cx).await
    }

    /// Resume an exact Delta seal proposal retained across cancellation or a
    /// dropped future.
    ///
    /// Until this succeeds, the old process-local Delta epoch remains visible
    /// and every writer mutation is rejected. The retained zero-timestamp
    /// MANIFEST proposal and canonical FSLX bytes make reconciliation exact
    /// even if the prior durable publication won ambiguously.
    ///
    /// # Errors
    ///
    /// Returns a typed error when no seal awaits reconciliation or when segment
    /// installation, MANIFEST publication, or Keeper reopening fails again.
    pub async fn resume_pending_delta_seal(
        &mut self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        if self.pending_delta_seal.is_none() {
            return Err(invalid_state("no Delta seal awaits reconciliation"));
        }
        self.finish_pending_delta_seal(cx).await
    }

    async fn finish_pending_delta_seal(
        &mut self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let (encoded, segment_installed, manifest, memory_owned) = {
            let pending = self
                .pending_delta_seal
                .as_ref()
                .expect("Delta seal completion requires retained state");
            let encoded = pending.encoded.clone();
            let memory_owned = if matches!(&self.backend, IndexBackend::Memory(_)) {
                pending
                    .encoded
                    .iter()
                    .map(|encoded| encoded.as_ref().clone())
                    .collect()
            } else {
                Vec::new()
            };
            (
                encoded,
                pending.segment_installed,
                pending.manifest.clone(),
                memory_owned,
            )
        };

        if let (Some(encoded), IndexBackend::Durable(writer)) = (encoded, &mut self.backend)
            && !segment_installed
        {
            writer
                .publish_encoded_segment_retryable(cx, encoded)
                .await?;
            self.pending_delta_seal
                .as_mut()
                .expect("installed Delta segment retains transaction state")
                .segment_installed = true;
        }
        check_cancel(cx, "Delta MANIFEST publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer.publish(cx, &manifest).await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments(&manifest, memory_owned)?;
            }
        }

        // No await or cancellation checkpoint is permitted after MANIFEST
        // authority changes. Taking the retained state and swapping the local
        // epoch therefore complete in the same poll that observes publication.
        let pending = self
            .pending_delta_seal
            .take()
            .expect("published Delta seal retains its prepared local swap");
        let installed = self
            .published_snapshot
            .install_prepared_sealed(Arc::new(self.backend.snapshot().clone()), pending.prepared);
        self.next_seal_seq = pending.next_seal_seq;
        self.next_lease_base = self.next_lease_base.max(pending.successor_watermark);
        self.docid_allocator =
            DocIdAllocator::open(self.next_lease_base, self.shard_router.shard_count())
                .map_err(|error| invalid_state(error.to_string()))?;
        Ok(installed)
    }

    /// Durable index directory, or `None` for an owned-buffer index.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.backend.snapshot().directory()
    }

    /// Whether the writer holds documents or installed segments not yet visible.
    #[must_use]
    pub fn has_uncommitted_changes(&self) -> bool {
        self.shards
            .iter()
            .any(|shard| shard.accumulator.document_count() != 0)
            || self.staged_flush.is_some()
            || !self.pending_segments.is_empty()
            || self.pending_manifest.is_some()
            || self.pending_replacement_manifest.is_some()
            || self.pending_delta_seal.is_some()
    }

    #[cfg(feature = "conformance-internals")]
    fn conformance_pending_writer_state(
        &self,
    ) -> Result<ConformancePendingWriterState, QuillIndexError> {
        let shard_count = conformance_usize_to_u64(self.shards.len(), "writer shard count")?;
        let dirty_shard_count = conformance_usize_to_u64(
            self.shards
                .iter()
                .filter(|shard| {
                    shard.accumulator.document_count() != 0
                        || !shard.identities.is_empty()
                        || shard.current_lease_base.is_some()
                })
                .count(),
            "dirty writer shard count",
        )?;
        let pending_identity_count = self.shards.iter().try_fold(0_u64, |total, shard| {
            total
                .checked_add(conformance_usize_to_u64(
                    shard.identities.len(),
                    "pending identity count",
                )?)
                .ok_or_else(|| invalid_state("pending identity count overflow"))
        })?;
        let uncommitted_id_count =
            conformance_usize_to_u64(self.uncommitted_ids.len(), "uncommitted id count")?;
        let pending_segment_count =
            conformance_usize_to_u64(self.pending_segments.len(), "pending segment count")?;
        let pending_owned_segment_count = conformance_usize_to_u64(
            self.pending_owned_segments.len(),
            "pending owned segment count",
        )?;
        let staged_flush_present = self.staged_flush.is_some();
        let pending_manifest_present = self.pending_manifest.is_some();
        let pending_replacement_manifest_present = self.pending_replacement_manifest.is_some();
        let pending_delta_seal_present = self.pending_delta_seal.is_some();
        let unpublished_since_present = self.unpublished_since.is_some();
        let mut flags = 0_u8;
        if staged_flush_present {
            flags |= ConformancePendingWriterState::STAGED_FLUSH_PRESENT;
        }
        if pending_manifest_present {
            flags |= ConformancePendingWriterState::PENDING_MANIFEST_PRESENT;
        }
        if pending_replacement_manifest_present {
            flags |= ConformancePendingWriterState::PENDING_REPLACEMENT_MANIFEST_PRESENT;
        }
        if pending_delta_seal_present {
            flags |= ConformancePendingWriterState::PENDING_DELTA_SEAL_PRESENT;
        }
        if unpublished_since_present {
            flags |= ConformancePendingWriterState::UNPUBLISHED_SINCE_PRESENT;
        }
        if self.ingest_retry_required {
            flags |= ConformancePendingWriterState::INGEST_RETRY_REQUIRED;
        }

        let published = self.published_snapshot.load();
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch/quill/pending-writer-state/v2\0");
        conformance_hash_bytes(&mut hasher, &self.schema.canonical_encoding()?);
        conformance_hash_bytes(
            &mut hasher,
            &conformance_manifest_bytes(
                &published.keeper_snapshot().loaded_manifest().manifest,
                "published composite MANIFEST",
            )?,
        );
        hasher.update(published.snapshot_epoch().to_be_bytes());
        hasher.update(published.keeper_generation().to_be_bytes());
        hasher.update(published.bm25_doc_count().to_be_bytes());
        hasher.update(published.live_doc_count().to_be_bytes());
        hasher.update(
            conformance_usize_to_u64(published.delta_count(), "published Delta count")?
                .to_be_bytes(),
        );
        for delta in published.delta_snapshots() {
            conformance_hash_bytes(&mut hasher, &delta.conformance_content_sha256()?);
        }
        conformance_hash_bytes(
            &mut hasher,
            &conformance_manifest_bytes(
                &self.backend.snapshot().loaded_manifest().manifest,
                "writer backend MANIFEST",
            )?,
        );
        hasher.update([u8::from(matches!(&self.backend, IndexBackend::Durable(_)))]);
        hasher.update(self.next_lease_base.to_be_bytes());
        hasher.update(self.next_seal_seq.to_be_bytes());
        hasher.update(self.docid_allocator.watermark().to_be_bytes());
        hasher.update(shard_count.to_be_bytes());
        hasher.update(dirty_shard_count.to_be_bytes());
        hasher.update(pending_identity_count.to_be_bytes());
        hasher.update(uncommitted_id_count.to_be_bytes());
        hasher.update(pending_segment_count.to_be_bytes());
        hasher.update(pending_owned_segment_count.to_be_bytes());
        hasher.update([flags]);

        for (shard_index, shard) in self.shards.iter().enumerate() {
            hasher
                .update(conformance_usize_to_u64(shard_index, "writer shard index")?.to_be_bytes());
            hasher.update(
                conformance_usize_to_u64(
                    shard.accumulator.document_count(),
                    "shard document count",
                )?
                .to_be_bytes(),
            );
            hasher.update(
                conformance_usize_to_u64(shard.accumulator.token_count(), "shard token count")?
                    .to_be_bytes(),
            );
            hasher.update(
                conformance_usize_to_u64(shard.accumulator.bytes_used(), "shard bytes used")?
                    .to_be_bytes(),
            );
            conformance_hash_optional_u64(&mut hasher, shard.current_lease_base);
            conformance_hash_optional_lease(
                &mut hasher,
                self.docid_allocator.live_lease(shard_index),
            );
            hasher.update(
                conformance_usize_to_u64(
                    shard.accumulator.document_ords().len(),
                    "shard document ordinal count",
                )?
                .to_be_bytes(),
            );
            for document_ord in shard.accumulator.document_ords() {
                hasher.update(document_ord.to_be_bytes());
            }
            hasher.update(
                conformance_usize_to_u64(shard.identities.len(), "shard identity count")?
                    .to_be_bytes(),
            );
            for identity in &shard.identities {
                hasher.update(identity.doc_ord.to_be_bytes());
                conformance_hash_bytes(&mut hasher, identity.document_id.as_bytes());
                hasher.update(identity.content_hash.to_be_bytes());
            }
        }

        hasher.update(uncommitted_id_count.to_be_bytes());
        for document_id in &self.uncommitted_ids {
            conformance_hash_bytes(&mut hasher, document_id.as_bytes());
        }

        match &self.staged_flush {
            None => hasher.update([0]),
            Some(staged) => {
                hasher.update([1]);
                hasher.update(
                    conformance_usize_to_u64(staged.shard, "staged flush shard")?.to_be_bytes(),
                );
                conformance_hash_bytes(&mut hasher, staged.encoded.as_bytes());
                conformance_hash_manifest_segment(&mut hasher, &staged.manifest_segment);
                conformance_hash_field_stats(&mut hasher, &staged.pending_field_stats)?;
                hasher.update(staged.next_seal_seq.to_be_bytes());
            }
        }

        hasher.update(pending_segment_count.to_be_bytes());
        for segment in &self.pending_segments {
            conformance_hash_manifest_segment(&mut hasher, segment);
        }
        hasher.update(pending_owned_segment_count.to_be_bytes());
        for segment in &self.pending_owned_segments {
            conformance_hash_bytes(&mut hasher, segment.as_bytes());
        }
        conformance_hash_field_stats(&mut hasher, &self.pending_field_stats)?;
        conformance_hash_optional_manifest(
            &mut hasher,
            self.pending_manifest.as_ref(),
            "pending MANIFEST",
        )?;
        conformance_hash_optional_manifest(
            &mut hasher,
            self.pending_replacement_manifest.as_ref(),
            "pending replacement MANIFEST",
        )?;
        conformance_hash_optional_delta_seal(&mut hasher, self.pending_delta_seal.as_ref())?;

        Ok(ConformancePendingWriterState {
            digest_sha256: hasher.finalize().into(),
            shard_count,
            dirty_shard_count,
            pending_identity_count,
            uncommitted_id_count,
            pending_segment_count,
            pending_owned_segment_count,
            flags,
        })
    }

    fn has_active_deltas(&self) -> bool {
        !self.published_snapshot.load().delta_snapshots().is_empty()
    }

    /// Route and accumulate one bounded batch into the production Scribe shard set.
    ///
    /// A budget or lease boundary seals an immutable segment, but no newly
    /// installed segment becomes searchable until [`Self::commit`] publishes
    /// its successor MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, duplicate-id, accumulation, flush, or
    /// publication failures. A failed prior MANIFEST publish must be retried
    /// with `commit` before more documents are accepted.
    pub async fn index_documents(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let replacement_ids = BTreeSet::new();
        self.index_documents_with_replacements(cx, documents, &replacement_ids, true)
            .await
    }

    async fn try_index_documents_parallel(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
        replacement_ids: &BTreeSet<&str>,
        allow_automatic_publication: bool,
    ) -> Result<Option<ParallelIngestReceipt>, QuillIndexError> {
        let shard_count = self.shards.len();
        let worker_count = rayon::current_num_threads();
        let active_shard_count =
            parallel_ingest_active_shards(documents.len(), shard_count, worker_count);
        if !replacement_ids.is_empty()
            || self.shard_router.is_deterministic()
            || active_shard_count < 2
        {
            return Ok(None);
        }

        let maximum_shard_documents = documents.len().div_ceil(active_shard_count);
        let maximum_shard_documents = u32::try_from(maximum_shard_documents)
            .map_err(|_| invalid_state("parallel shard document count does not fit u32"))?;
        for shard in 0..shard_count {
            let live_lease = self.docid_allocator.live_lease(shard);
            let remaining = live_lease.map_or(DOC_ORDS_PER_LEASE, |(_, next_ord)| {
                DOC_ORDS_PER_LEASE - next_ord
            });
            if maximum_shard_documents > remaining {
                return Ok(None);
            }
            if let (Some(current), Some((live, _))) =
                (self.shards[shard].current_lease_base, live_lease)
                && current != live
            {
                return Ok(None);
            }
        }

        let mut batch_ids = BTreeSet::new();
        for document in documents {
            check_cancel(cx, "parallel index validation")?;
            if document.id.is_empty() {
                return Err(invalid_state("document id must be nonempty"));
            }
            if !batch_ids.insert(document.id.as_str())
                || self.uncommitted_ids.contains(&document.id)
                || self
                    .backend
                    .snapshot()
                    .resolve_document_id(&document.id)?
                    .is_some()
            {
                return Err(invalid_state(format!(
                    "duplicate live document id {:?}",
                    document.id
                )));
            }
        }

        let mut assignments = vec![None; shard_count];
        let documents_per_shard = documents.len() / active_shard_count;
        let remainder = documents.len() % active_shard_count;
        let mut document_start = 0_usize;
        for position in 0..active_shard_count {
            let shard = self.shard_router.route_batch();
            let count = documents_per_shard + usize::from(position < remainder);
            let count_u32 = u32::try_from(count)
                .map_err(|_| invalid_state("parallel shard document count does not fit u32"))?;
            let allocated = self
                .docid_allocator
                .alloc_batch(shard, count_u32)
                .map_err(|error| invalid_state(error.to_string()))?;
            let [span] = allocated.spans() else {
                return Err(invalid_state(
                    "parallel shard allocation unexpectedly crossed a Q1 lease",
                ));
            };
            let document_end = document_start
                .checked_add(count)
                .ok_or_else(|| invalid_state("parallel document range overflow"))?;
            assignments[shard] = Some(ParallelShardAssignment {
                document_start,
                document_end,
                span: *span,
            });
            self.shards[shard].current_lease_base = Some(span.lease_base);
            document_start = document_end;
        }
        debug_assert_eq!(document_start, documents.len());
        self.next_lease_base = self.docid_allocator.watermark();

        self.shards
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(shard, state)| {
                assignments[shard].map_or(Ok(()), |assignment| {
                    Self::accumulate_parallel_shard(state, documents, assignment)
                })
            })?;

        let arena_bytes_used_high_water = self.shards.iter().fold(0_usize, |total, shard| {
            total.saturating_add(shard.accumulator.bytes_used())
        });
        let arena_bytes_reserved_high_water = self.shards.iter().fold(0_usize, |total, shard| {
            total.saturating_add(shard.accumulator.bytes_reserved())
        });

        self.uncommitted_ids
            .extend(documents.iter().map(|document| document.id.clone()));
        self.unpublished_since.get_or_insert_with(Instant::now);

        if self.shards.iter().any(|shard| {
            shard
                .accumulator
                .should_flush(self.config.scribe_shard_budget_bytes)
        }) {
            // All shards reached this point through independent accumulators. Seal
            // their segments as one parallel generation instead of serializing the
            // hot path on the first shard that crosses its arena budget.
            self.flush_all_shards(cx, LifecycleTrigger::ArenaBudget)
                .await?;
            if allow_automatic_publication {
                self.publish_bulk_cadence_if_due(cx).await?;
            }
        }
        let visibility_due = self.unpublished_since.is_some_and(|started| {
            started.elapsed() >= Duration::from_millis(self.config.max_visibility_lag_ms)
        });
        if allow_automatic_publication && visibility_due {
            self.commit_with_trigger(cx, LifecycleTrigger::VisibilityLag)
                .await?;
        }
        Ok(Some(ParallelIngestReceipt {
            active_shards: active_shard_count,
            worker_count,
            arena_bytes_used_high_water,
            arena_bytes_reserved_high_water,
        }))
    }

    fn accumulate_parallel_shard(
        state: &mut ScribeShardState,
        documents: &[IndexableDocument],
        assignment: ParallelShardAssignment,
    ) -> Result<(), QuillIndexError> {
        let shard_documents = documents
            .get(assignment.document_start..assignment.document_end)
            .ok_or_else(|| invalid_state("parallel shard document range is outside the batch"))?;
        if shard_documents.len()
            != usize::try_from(assignment.span.len)
                .map_err(|_| invalid_state("parallel shard span length does not fit usize"))?
        {
            return Err(invalid_state(
                "parallel shard document range differs from its Q1 span",
            ));
        }
        state
            .identities
            .try_reserve(shard_documents.len())
            .map_err(|_| invalid_state("could not reserve parallel shard identities"))?;
        for (offset, document) in shard_documents.iter().enumerate() {
            let offset = u32::try_from(offset)
                .map_err(|_| invalid_state("parallel shard offset does not fit u32"))?;
            let doc_ord = assignment
                .span
                .ord_start
                .checked_add(offset)
                .ok_or_else(|| invalid_state("lease-relative document ordinal overflow"))?;
            let global_docid = assignment
                .span
                .lease_base
                .checked_add(u64::from(doc_ord))
                .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
                .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;
            let metadata = canonical_metadata(&document.metadata)?;
            let title = document.title.as_deref().unwrap_or("");
            let indexed = [
                IndexedFieldValue::new(ID_FIELD, &document.id),
                IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                IndexedFieldValue::new(TITLE_FIELD, title),
            ];
            let numeric = [IndexedNumericValue::u64(ORD_FIELD, global_docid)];
            let stored = [StoredFieldValue::new(METADATA_FIELD, &metadata)];
            state
                .accumulator
                .add_document_with_values(doc_ord, &indexed, &numeric, &stored)?;
            let content_hash = canonical_document_content_hash(document, &metadata)?;
            state.identities.push(PendingIdentity {
                doc_ord,
                document_id: document.id.clone(),
                content_hash,
            });
        }
        Ok(())
    }

    async fn index_documents_with_replacements(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
        replacement_ids: &BTreeSet<&str>,
        allow_automatic_publication: bool,
    ) -> Result<(), QuillIndexError> {
        let ingest_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::SCRIBE_INGEST,
            phase = "ingest",
            doc_count = documents.len(),
            result_count = tracing::field::Empty,
            parallel_active_shards = tracing::field::Empty,
            parallel_worker_count = tracing::field::Empty,
            arena_bytes_used_high_water = tracing::field::Empty,
            arena_bytes_reserved_high_water = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _ingest_timer = crate::tracing_conventions::StageTimer::new(&ingest_span);
        let instrumented = ingest_span.clone();
        async move {
            check_cancel(cx, "index")?;
            if self.pending_delta_seal.is_some() {
                return Err(invalid_state(
                    "resume the retained Delta seal before scalar indexing",
                ));
            }
            if self.has_active_deltas() {
                return Err(invalid_state(
                    "scalar indexing cannot run while process-local Delta epochs are active",
                ));
            }
            if self.ingest_retry_required
                || self.staged_flush.is_some()
                || self.pending_manifest.is_some()
                || (self.pending_replacement_manifest.is_some() && replacement_ids.is_empty())
            {
                return Err(invalid_state(
                    "a prior scalar mutation requires commit retry before indexing",
                ));
            }
            if documents.is_empty() {
                return Ok(());
            }
            // Arm the fail-closed retry guard before allocation or mutation.
            // Only the successful return below (or a successful commit)
            // disarms it.
            self.ingest_retry_required = true;
            if let Some(receipt) = self
                .try_index_documents_parallel(
                    cx,
                    documents,
                    replacement_ids,
                    allow_automatic_publication,
                )
                .await?
            {
                ingest_span.record(
                    "result_count",
                    u64::try_from(documents.len()).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "parallel_active_shards",
                    u64::try_from(receipt.active_shards).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "parallel_worker_count",
                    u64::try_from(receipt.worker_count).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "arena_bytes_used_high_water",
                    u64::try_from(receipt.arena_bytes_used_high_water).unwrap_or(u64::MAX),
                );
                ingest_span.record(
                    "arena_bytes_reserved_high_water",
                    u64::try_from(receipt.arena_bytes_reserved_high_water).unwrap_or(u64::MAX),
                );
                self.ingest_retry_required = false;
                return Ok(());
            }
            let shard_id = self.shard_router.route_batch();
            let document_count = u32::try_from(documents.len())
                .map_err(|_| invalid_state("ingest batch document count does not fit u32"))?;
            let allocated = self
                .docid_allocator
                .alloc_batch(shard_id, document_count)
                .map_err(|error| invalid_state(error.to_string()))?;
            self.next_lease_base = self.docid_allocator.watermark();
            let mut arena_bytes_used_high_water = self
                .shards
                .iter()
                .map(|shard| shard.accumulator.bytes_used())
                .max()
                .unwrap_or(0);
            let mut arena_bytes_reserved_high_water = self
                .shards
                .iter()
                .map(|shard| shard.accumulator.bytes_reserved())
                .max()
                .unwrap_or(0);
            let mut document_index = 0_usize;
            for (span_index, span) in allocated.spans().iter().copied().enumerate() {
                let lease_changed = self.shards[shard_id]
                    .current_lease_base
                    .is_some_and(|base| base != span.lease_base);
                if lease_changed {
                    self.flush_shard(cx, shard_id, LifecycleTrigger::LeaseBoundary)
                        .await?;
                }
                self.shards[shard_id].current_lease_base = Some(span.lease_base);
                for span_offset in 0..span.len {
                    let document = &documents[document_index];
                    document_index += 1;
                    check_cancel(cx, "index")?;
                    if document.id.is_empty() {
                        return Err(invalid_state("document id must be nonempty"));
                    }
                    if self.uncommitted_ids.contains(&document.id)
                        || self
                            .backend
                            .snapshot()
                            .resolve_document_id(&document.id)?
                            .is_some()
                            && !replacement_ids.contains(document.id.as_str())
                    {
                        return Err(invalid_state(format!(
                            "duplicate live document id {:?}",
                            document.id
                        )));
                    }
                    let doc_ord = span
                        .ord_start
                        .checked_add(span_offset)
                        .ok_or_else(|| invalid_state("lease-relative document ordinal overflow"))?;
                    let global_docid = span
                        .lease_base
                        .checked_add(u64::from(doc_ord))
                        .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
                        .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;

                    let metadata = canonical_metadata(&document.metadata)?;
                    let title = document.title.as_deref().unwrap_or("");
                    let indexed = [
                        IndexedFieldValue::new(ID_FIELD, &document.id),
                        IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                        IndexedFieldValue::new(TITLE_FIELD, title),
                    ];
                    let numeric = [IndexedNumericValue::u64(ORD_FIELD, global_docid)];
                    let stored = [StoredFieldValue::new(METADATA_FIELD, &metadata)];
                    let accumulated = self.shards[shard_id]
                        .accumulator
                        .add_document_with_values(doc_ord, &indexed, &numeric, &stored)?;
                    arena_bytes_used_high_water =
                        arena_bytes_used_high_water.max(accumulated.bytes_used);
                    arena_bytes_reserved_high_water =
                        arena_bytes_reserved_high_water.max(accumulated.bytes_reserved);
                    let content_hash = canonical_document_content_hash(document, &metadata)?;
                    self.shards[shard_id].identities.push(PendingIdentity {
                        doc_ord,
                        document_id: document.id.clone(),
                        content_hash,
                    });
                    self.uncommitted_ids.insert(document.id.clone());
                    self.unpublished_since.get_or_insert_with(Instant::now);

                    if self.shards[shard_id]
                        .accumulator
                        .should_flush(self.config.scribe_shard_budget_bytes)
                    {
                        self.flush_shard(cx, shard_id, LifecycleTrigger::ArenaBudget)
                            .await?;
                        self.shards[shard_id].current_lease_base = Some(span.lease_base);
                        if allow_automatic_publication {
                            self.publish_bulk_cadence_if_due(cx).await?;
                        }
                    }
                }
                if span_index + 1 < allocated.spans().len() {
                    self.flush_shard(cx, shard_id, LifecycleTrigger::LeaseBoundary)
                        .await?;
                    if allow_automatic_publication {
                        self.publish_bulk_cadence_if_due(cx).await?;
                    }
                }
            }
            debug_assert_eq!(document_index, documents.len());
            let visibility_due = self.unpublished_since.is_some_and(|started| {
                started.elapsed() >= Duration::from_millis(self.config.max_visibility_lag_ms)
            });
            if allow_automatic_publication && visibility_due {
                self.commit_with_trigger(cx, LifecycleTrigger::VisibilityLag)
                    .await?;
            }
            ingest_span.record(
                "result_count",
                u64::try_from(documents.len()).unwrap_or(u64::MAX),
            );
            ingest_span.record(
                "arena_bytes_used_high_water",
                u64::try_from(arena_bytes_used_high_water).unwrap_or(u64::MAX),
            );
            ingest_span.record(
                "arena_bytes_reserved_high_water",
                u64::try_from(arena_bytes_reserved_high_water).unwrap_or(u64::MAX),
            );
            self.ingest_retry_required = false;
            Ok(())
        }
        .instrument(instrumented)
        .await
    }

    /// Seal the scalar accumulator and atomically publish the next MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed flush, segment-install, manifest-transition, durability,
    /// or cancellation failures. Installed-but-unreferenced segments remain
    /// invisible and are retained for a publication retry.
    pub async fn commit(&mut self, cx: &Cx) -> Result<&KeeperSnapshot, QuillIndexError> {
        self.commit_with_trigger(cx, LifecycleTrigger::ExplicitFlush)
            .await
    }

    async fn commit_with_trigger(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "commit")?;
        if self.pending_delta_seal.is_some() {
            return Err(invalid_state(
                "resume the retained Delta seal before scalar commit",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "scalar commit cannot run while process-local Delta epochs are active",
            ));
        }
        self.flush_all_shards(cx, trigger).await?;
        self.publish_pending_segments(cx, trigger).await?;
        if !self.config.bulk_load_mode {
            self.apply_tier_policy(cx).await?;
        }
        self.ingest_retry_required = false;
        Ok(self.backend.snapshot())
    }

    async fn publish_bulk_cadence_if_due(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        if self.config.bulk_load_mode
            && self.pending_segments.len() >= self.config.bulk_publish_segment_cadence
        {
            self.publish_pending_segments(cx, LifecycleTrigger::BulkCadence)
                .await?;
        }
        Ok(())
    }

    async fn publish_pending_segments(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        check_cancel(cx, "commit publish")?;
        if self.pending_segments.is_empty()
            && self.pending_replacement_manifest.is_none()
            && self.pending_manifest.is_none()
        {
            return Ok(());
        }

        self.prepare_pending_manifest()?;
        let manifest = self
            .pending_manifest
            .as_ref()
            .expect("nonempty pending segments have a retained MANIFEST proposal")
            .clone();
        let prepared_publication = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        let commit_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_LIFECYCLE,
            phase = "publish",
            trigger = trigger.as_str(),
            action = "publish",
            generation = manifest.generation,
            segment_count = manifest.segments.len(),
            doc_count = manifest.field_stats.first().map_or(0, |row| row.doc_count),
            result_count = manifest.segments.len(),
            duration_us = tracing::field::Empty,
        );
        let _commit_timer = crate::tracing_conventions::StageTimer::new(&commit_span);
        let instrumented = commit_span.clone();
        async {
            #[cfg(feature = "conformance-internals")]
            self.reader
                .conformance_controller
                .checkpoint(ConformanceCancellationStage::CommitPublication, cx);
            check_cancel(cx, "commit publish")?;
            let open_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_OPEN,
                phase = "open.committed",
                durability = matches!(&self.backend, IndexBackend::Durable(_)),
                generation = tracing::field::Empty,
                segment_count = tracing::field::Empty,
                doc_count = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
            let instrumented = open_span.clone();
            async {
                match &mut self.backend {
                    IndexBackend::Durable(writer) => {
                        writer.publish(cx, &manifest).await?;
                    }
                    IndexBackend::Memory(snapshot) => {
                        let published = snapshot.publish_owned_segments(
                            &manifest,
                            self.pending_owned_segments.clone(),
                        )?;
                        *snapshot = published;
                    }
                }
                self.published_snapshot.install_prepared_sealed(
                    Arc::new(self.backend.snapshot().clone()),
                    prepared_publication,
                );
                record_snapshot_fields(&open_span, self.backend.snapshot());
                Ok::<(), QuillIndexError>(())
            }
            .instrument(instrumented)
            .await?;
            Ok::<(), QuillIndexError>(())
        }
        .instrument(instrumented)
        .await?;
        self.pending_segments.clear();
        self.pending_owned_segments.clear();
        self.pending_field_stats.clear();
        self.pending_manifest = None;
        self.pending_replacement_manifest = None;
        self.uncommitted_ids.clear();
        for shard in &self.shards {
            self.uncommitted_ids.extend(
                shard
                    .identities
                    .iter()
                    .map(|identity| identity.document_id.clone()),
            );
        }
        if self
            .shards
            .iter()
            .all(|shard| shard.accumulator.document_count() == 0)
        {
            self.unpublished_since = None;
        }
        Ok(())
    }

    async fn apply_tier_policy(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        loop {
            let policy = TierMergePolicy::from_config(&self.config);
            let plan = plan_tier_merge(
                &self.backend.snapshot().loaded_manifest().manifest.segments,
                policy,
            )?;
            let Some(plan) = plan else {
                return Ok(());
            };
            let output_segment_id =
                self.derive_policy_segment_id(&plan.source_segment_ids, b"tier")?;
            let created_unix_s = self.created_unix_s()?;
            let policy_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "merge",
                trigger = LifecycleTrigger::TierFanout.as_str(),
                action = "concat_merge",
                tier = ?plan.tier,
                source_count = plan.source_segment_ids.len(),
                docid_lo = plan.docid_lo,
                docid_hi = plan.docid_hi,
                hole_ratio = plan.hole_ratio,
                segment_id = output_segment_id,
                duration_us = tracing::field::Empty,
            );
            let _policy_timer = crate::tracing_conventions::StageTimer::new(&policy_span);
            self.concat_merge(
                cx,
                &plan.source_segment_ids,
                output_segment_id,
                created_unix_s,
            )
            .instrument(policy_span)
            .await?;
        }
    }

    async fn finish_bulk_load(&mut self, cx: &Cx) -> Result<&KeeperSnapshot, QuillIndexError> {
        if !self.config.bulk_load_mode {
            return Err(invalid_state(
                "finish_bulk_load requires bulk_load_mode in QuillConfig",
            ));
        }
        self.commit_with_trigger(cx, LifecycleTrigger::BulkFinish)
            .await?;
        let source_segment_ids = self
            .backend
            .snapshot()
            .loaded_manifest()
            .manifest
            .segments
            .iter()
            .map(|segment| segment.segment_id)
            .collect::<Vec<_>>();
        if source_segment_ids.len() > 1 {
            let output_segment_id =
                self.derive_policy_segment_id(&source_segment_ids, b"bulk-finish")?;
            let created_unix_s = self.created_unix_s()?;
            let bulk_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "merge",
                trigger = LifecycleTrigger::BulkFinish.as_str(),
                action = "concat_merge",
                source_count = source_segment_ids.len(),
                segment_id = output_segment_id,
                duration_us = tracing::field::Empty,
            );
            let _bulk_timer = crate::tracing_conventions::StageTimer::new(&bulk_span);
            self.concat_merge(cx, &source_segment_ids, output_segment_id, created_unix_s)
                .instrument(bulk_span)
                .await?;
        }
        self.publish_bulk_completion(cx).await?;
        self.reader.config.bulk_load_mode = false;
        Ok(self.backend.snapshot())
    }

    async fn publish_bulk_completion(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        let current = &self.backend.snapshot().loaded_manifest().manifest;
        if current.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS == 0 {
            return Ok(());
        }
        let mut manifest = self.backend.snapshot().next_manifest()?;
        manifest.flags &= !MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        manifest.last_publish_unix_s = 0;
        if matches!(&self.backend, IndexBackend::Durable(_)) {
            manifest.last_publish_unix_s = wall_clock_unix_s()?;
        }
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        check_cancel(cx, "bulk completion publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer.publish(cx, &manifest).await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments(&manifest, Vec::new())?;
            }
        }
        self.published_snapshot
            .install_prepared_sealed(Arc::new(self.backend.snapshot().clone()), prepared);
        Ok(())
    }

    /// Replace one exact committed manifest run with a Q1 concat merge.
    ///
    /// The caller supplies source ids in current manifest order plus a
    /// collision-free output identity. Policy-driven candidate selection and
    /// identity generation remain Keeper E3.7 concerns; this method enforces
    /// that no accumulator or staged publication can race the structural
    /// replacement.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted state, cancellation, a nonconsecutive source run,
    /// codec/invariant failures, or durable publication failure.
    pub async fn concat_merge(
        &mut self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "concat merge")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "concat merge requires a fully committed scalar index",
            ));
        }
        self.retire_ingest_leases()?;
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        let prepared_publication = self
            .published_snapshot
            .prepare_equivalent_sealed_successor()?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .concat_merge(cx, source_segment_ids, output_segment_id, created_unix_s)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                check_cancel(cx, "concat merge")?;
                let source_snapshot = snapshot.clone();
                let mut source_ids = Vec::new();
                source_ids
                    .try_reserve_exact(source_segment_ids.len())
                    .map_err(|_| invalid_state("could not allocate concat-merge source ids"))?;
                source_ids.extend_from_slice(source_segment_ids);
                let successor = spawn_blocking(move || {
                    source_snapshot.concat_merge_owned(
                        &source_ids,
                        output_segment_id,
                        created_unix_s,
                    )
                })
                .await?;
                check_cancel(cx, "concat merge")?;
                *snapshot = successor;
            }
        }
        self.published_snapshot.install_prepared_sealed(
            Arc::new(self.backend.snapshot().clone()),
            prepared_publication,
        );
        self.next_seal_seq = next_seal_seq;
        Ok(self.backend.snapshot())
    }

    fn retire_ingest_leases(&mut self) -> Result<(), QuillIndexError> {
        if self
            .shards
            .iter()
            .any(|shard| shard.accumulator.document_count() != 0)
        {
            return Err(invalid_state(
                "cannot retire ingest leases while a Scribe shard is nonempty",
            ));
        }
        let burn = self.docid_allocator.end_session();
        self.next_lease_base = self.next_lease_base.max(burn.final_watermark);
        self.docid_allocator =
            DocIdAllocator::open(self.next_lease_base, self.shard_router.shard_count())
                .map_err(|error| invalid_state(error.to_string()))?;
        for shard in &mut self.shards {
            shard.current_lease_base = None;
        }
        Ok(())
    }

    /// Fold tombstones into immutable positional holes for eligible segments.
    ///
    /// Every surviving global document id is preserved. The complete set of
    /// replacement files is built and synced before one successor MANIFEST is
    /// published, so interruption before that boundary leaves the old snapshot
    /// authoritative. A policy-boundary equality is a no-op.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted scalar or Delta state, invalid policy values,
    /// cancellation, source/codec failures, and durable publication failures.
    pub async fn compact(
        &mut self,
        cx: &Cx,
        policy: CompactionPolicy,
    ) -> Result<CompactionReport, QuillIndexError> {
        check_cancel(cx, "compaction")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "compaction requires a fully committed scalar index",
            ));
        }
        let created_unix_s = self.created_unix_s()?;
        let compact_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_COMPACT,
            phase = "compact",
            durability = matches!(&self.backend, IndexBackend::Durable(_)),
            generation = self.snapshot().loaded_manifest().manifest.generation,
            segment_count = self.snapshot().segments().len(),
            result_count = tracing::field::Empty,
            output_bytes = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _compact_timer = crate::tracing_conventions::StageTimer::new(&compact_span);
        let instrumented = compact_span.clone();
        let report = async {
            match &mut self.backend {
                IndexBackend::Durable(writer) => Ok::<CompactionReport, QuillIndexError>(
                    writer.compact(cx, policy, created_unix_s).await?,
                ),
                IndexBackend::Memory(snapshot) => {
                    let source = snapshot.clone();
                    let (successor, report) =
                        spawn_blocking(move || source.compact_owned(policy, created_unix_s))
                            .await?;
                    check_cancel(cx, "compaction")?;
                    *snapshot = successor;
                    Ok(report)
                }
            }
        }
        .instrument(instrumented)
        .await?;
        compact_span.record("result_count", report.compacted_segments);
        compact_span.record("output_bytes", report.output_bytes);
        if report.changed() {
            let prepared_publication = self.published_snapshot.prepare_sealed_manifest(
                self.schema,
                &self.backend.snapshot().loaded_manifest().manifest,
            )?;
            self.published_snapshot.install_prepared_sealed(
                Arc::new(self.backend.snapshot().clone()),
                prepared_publication,
            );
            self.next_seal_seq = self
                .snapshot()
                .segments()
                .iter()
                .map(|segment| segment.manifest().seal_seq)
                .max()
                .unwrap_or(0);
        }
        Ok(report)
    }

    async fn upsert_documents(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        check_cancel(cx, "upsert documents")?;
        let mut manifest = self.backend.snapshot().next_manifest()?;
        let mut replacement_ids = BTreeSet::new();
        for document in documents {
            check_cancel(cx, "upsert document batch")?;
            if self
                .backend
                .snapshot()
                .delete_document(&mut manifest, &document.id)?
            {
                replacement_ids.insert(document.id.as_str());
            }
        }
        if replacement_ids.is_empty() {
            return self.index_documents(cx, documents).await;
        }
        if self.has_uncommitted_changes() {
            let message = if self.pending_replacement_manifest.is_some() {
                "a retained replacement MANIFEST awaits commit retry"
            } else {
                "uncommitted changes must be committed before replacing live documents"
            };
            return Err(invalid_state(message));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "upsert documents cannot mutate a frozen process-local Delta epoch",
            ));
        }

        self.pending_replacement_manifest = Some(manifest);
        self.index_documents_with_replacements(cx, documents, &replacement_ids, false)
            .await?;
        self.flush_all_shards(cx, LifecycleTrigger::ExplicitFlush)
            .await?;
        self.prepare_pending_manifest()?;
        self.publish_pending_segments(cx, LifecycleTrigger::ExplicitFlush)
            .await?;
        if !self.config.bulk_load_mode {
            self.apply_tier_policy(cx).await?;
        }
        Ok(())
    }

    async fn delete_document(
        &mut self,
        cx: &Cx,
        document_id: &str,
    ) -> Result<bool, QuillIndexError> {
        Ok(self.delete_documents(cx, &[document_id]).await? != 0)
    }

    async fn delete_documents(
        &mut self,
        cx: &Cx,
        document_ids: &[&str],
    ) -> Result<usize, QuillIndexError> {
        check_cancel(cx, "delete document")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "delete_document requires a fully committed scalar index",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "delete_document cannot mutate a frozen process-local Delta epoch",
            ));
        }
        let mut manifest = self.backend.snapshot().next_manifest()?;
        manifest.last_publish_unix_s = 0;
        let mut deleted = 0_usize;
        for &document_id in document_ids {
            check_cancel(cx, "delete document batch")?;
            if self
                .backend
                .snapshot()
                .delete_document(&mut manifest, document_id)?
            {
                deleted = deleted.saturating_add(1);
            }
        }
        if deleted == 0 {
            return Ok(0);
        }
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        check_cancel(cx, "delete document publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer.publish(cx, &manifest).await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments(&manifest, Vec::new())?;
            }
        }
        self.published_snapshot
            .install_prepared_sealed(Arc::new(self.backend.snapshot().clone()), prepared);
        Ok(deleted)
    }

    async fn delete_all(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        check_cancel(cx, "delete all")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "delete_all requires a fully committed scalar index",
            ));
        }
        if self.has_active_deltas() {
            return Err(invalid_state(
                "delete_all cannot mutate frozen process-local Delta epochs",
            ));
        }
        let mut manifest = self.backend.snapshot().next_manifest()?;
        manifest.last_publish_unix_s = 0;
        self.backend.snapshot().delete_all(&mut manifest)?;
        let prepared = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
        check_cancel(cx, "delete all publish")?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer.publish(cx, &manifest).await?;
            }
            IndexBackend::Memory(snapshot) => {
                *snapshot = snapshot.publish_owned_segments(&manifest, Vec::new())?;
            }
        }
        self.published_snapshot
            .install_prepared_sealed(Arc::new(self.backend.snapshot().clone()), prepared);
        Ok(())
    }

    fn prepare_pending_manifest(&mut self) -> Result<(), QuillIndexError> {
        if self.pending_manifest.is_some() {
            return Ok(());
        }
        let manifest = if let Some(manifest) = &self.pending_replacement_manifest {
            manifest.clone()
        } else {
            self.backend.snapshot().next_manifest()?
        };
        self.prepare_pending_manifest_from(manifest)?;
        self.pending_replacement_manifest = None;
        Ok(())
    }

    fn prepare_pending_manifest_from(
        &mut self,
        mut manifest: Manifest,
    ) -> Result<(), QuillIndexError> {
        if self.pending_manifest.is_some() {
            return Ok(());
        }
        manifest
            .segments
            .extend(self.pending_segments.iter().cloned());
        manifest
            .segments
            .sort_unstable_by_key(|segment| segment.docid_lo);
        manifest.docid_high_watermark = self.next_lease_base;
        if self.config.bulk_load_mode {
            manifest.flags |= MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        } else {
            manifest.flags &= !MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        }
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &self.pending_field_stats)?;
        if matches!(&self.backend, IndexBackend::Durable(_)) {
            // Retain one exact, explicitly stamped image until publication
            // succeeds. A crash-shaped retry may encounter the prior attempt's
            // synced `.tmp-manifest-N`, whose bytes must remain reusable even
            // after the wall clock advances.
            manifest.last_publish_unix_s = wall_clock_unix_s()?;
        }
        self.pending_manifest = Some(manifest);
        Ok(())
    }

    async fn flush_shard(
        &mut self,
        cx: &Cx,
        requested_shard: usize,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        loop {
            let shard = self
                .staged_flush
                .as_ref()
                .map_or(requested_shard, |staged| staged.shard);
            if self.staged_flush.is_none()
                && self.shards[requested_shard].accumulator.document_count() == 0
            {
                return Ok(());
            }
            let state = &self.shards[shard];
            let seal_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "seal",
                trigger = trigger.as_str(),
                action = "seal",
                shard_id = shard,
                segment_id = tracing::field::Empty,
                doc_count = state.accumulator.document_count(),
                token_count = state.accumulator.token_count(),
                result_count = tracing::field::Empty,
                output_bytes = tracing::field::Empty,
                arena_bytes_used_high_water = state.accumulator.bytes_used(),
                arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
                duration_us = tracing::field::Empty,
            );
            let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
            let instrumented = seal_span.clone();
            async {
                check_cancel(cx, "flush")?;
                self.prepare_shard_flush(shard)?;
                if let Some(staged) = self.staged_flush.as_ref() {
                    seal_span.record("segment_id", staged.manifest_segment.segment_id);
                    seal_span.record("result_count", u64::from(staged.manifest_segment.doc_count));
                    seal_span.record("output_bytes", staged.encoded.file_len());
                }
                self.install_staged_flush(cx).await
            }
            .instrument(instrumented)
            .await?;
            if shard == requested_shard {
                return Ok(());
            }
        }
    }

    async fn flush_all_shards(
        &mut self,
        cx: &Cx,
        trigger: LifecycleTrigger,
    ) -> Result<(), QuillIndexError> {
        if let Some(shard) = self.staged_flush.as_ref().map(|staged| staged.shard) {
            self.flush_shard(cx, shard, trigger).await?;
        }

        let nonempty = self
            .shards
            .iter()
            .enumerate()
            .filter_map(|(shard, state)| (state.accumulator.document_count() != 0).then_some(shard))
            .collect::<Vec<_>>();
        if nonempty.len() < 2 || rayon::current_num_threads() < 2 {
            for shard in nonempty {
                self.flush_shard(cx, shard, trigger).await?;
            }
            return Ok(());
        }

        check_cancel(cx, "parallel flush")?;
        let built = {
            let plans = self.plan_parallel_shard_flushes(&nonempty)?;
            plans
                .par_iter()
                .map(Self::build_planned_shard_flush)
                .collect::<Result<Vec<_>, _>>()?
        };

        for built in built {
            check_cancel(cx, "parallel flush install")?;
            let shard = built.shard;
            let state = &self.shards[shard];
            let seal_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_LIFECYCLE,
                phase = "seal",
                trigger = trigger.as_str(),
                action = "install_parallel_build",
                shard_id = shard,
                segment_id = built.manifest_segment.segment_id,
                doc_count = state.accumulator.document_count(),
                token_count = state.accumulator.token_count(),
                result_count = u64::from(built.manifest_segment.doc_count),
                output_bytes = built.encoded.file_len(),
                arena_bytes_used_high_water = state.accumulator.bytes_used(),
                arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
                duration_us = tracing::field::Empty,
            );
            let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
            let instrumented = seal_span.clone();
            async {
                self.stage_built_shard_flush(built)?;
                self.install_staged_flush(cx).await
            }
            .instrument(instrumented)
            .await?;
        }
        Ok(())
    }

    fn plan_parallel_shard_flushes<'a>(
        &'a self,
        shards: &[usize],
    ) -> Result<Vec<ShardFlushPlan<'a>>, QuillIndexError> {
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(shards.len())
            .map_err(|_| invalid_state("could not reserve parallel shard flush plans"))?;
        let mut reserved_segment_ids = BTreeSet::new();
        for (offset, &shard) in shards.iter().enumerate() {
            let state = &self.shards[shard];
            let lease_docid_base = state
                .current_lease_base
                .ok_or_else(|| invalid_state("nonempty accumulator has no Q1 lease"))?;
            let created_unix_s = self.created_unix_s()?;
            let segment_id = self.derive_segment_id_avoiding(
                shard,
                lease_docid_base,
                created_unix_s,
                &reserved_segment_ids,
            )?;
            reserved_segment_ids.insert(segment_id);
            let offset = u64::try_from(offset)
                .map_err(|_| invalid_state("parallel shard count does not fit u64"))?;
            let seal_seq = self
                .next_seal_seq
                .checked_add(offset)
                .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
            plans.push(ShardFlushPlan {
                shard,
                state,
                segment_id,
                lease_docid_base,
                created_unix_s,
                seal_seq,
            });
        }
        Ok(plans)
    }

    fn build_planned_shard_flush(
        plan: &ShardFlushPlan<'_>,
    ) -> Result<BuiltShardFlush, QuillIndexError> {
        let state = plan.state;
        let seal_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_LIFECYCLE,
            phase = "seal",
            trigger = "parallel_batch",
            action = "build_shared_nothing",
            shard_id = plan.shard,
            segment_id = plan.segment_id,
            doc_count = state.accumulator.document_count(),
            token_count = state.accumulator.token_count(),
            result_count = tracing::field::Empty,
            output_bytes = tracing::field::Empty,
            arena_bytes_used_high_water = state.accumulator.bytes_used(),
            arena_bytes_reserved_high_water = state.accumulator.bytes_reserved(),
            duration_us = tracing::field::Empty,
        );
        let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
        let _entered = seal_span.enter();
        let documents = state
            .identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::new(
                    identity.doc_ord,
                    &identity.document_id,
                    identity.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &state.accumulator,
            FlushSegmentInput {
                segment_id: plan.segment_id,
                lease_docid_base: plan.lease_docid_base,
                created_unix_s: plan.created_unix_s,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Scalar,
        )?;
        let manifest_segment = manifest_segment(&encoded, plan.seal_seq);
        let next_seal_seq = plan
            .seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        seal_span.record("result_count", u64::from(manifest_segment.doc_count));
        seal_span.record("output_bytes", encoded.file_len());
        Ok(BuiltShardFlush {
            shard: plan.shard,
            encoded,
            manifest_segment,
            next_seal_seq,
        })
    }

    fn stage_built_shard_flush(&mut self, built: BuiltShardFlush) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_some() {
            return Err(invalid_state(
                "cannot stage a parallel shard build while another flush is retained",
            ));
        }
        let shard = built.shard;
        let document_count = u32::try_from(self.shards[shard].accumulator.document_count())
            .map_err(|_| invalid_state("segment document count does not fit u32"))?;
        let mut pending_field_stats = self.pending_field_stats.clone();
        for field in self.shards[shard].accumulator.fields() {
            let entry = pending_field_stats
                .entry(field.field_ord())
                .or_insert((0, 0));
            entry.0 = entry
                .0
                .checked_add(field.total_tokens())
                .ok_or_else(|| invalid_state("pending field token count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(document_count)
                .ok_or_else(|| invalid_state("pending field document count overflow"))?;
        }
        self.pending_segments
            .try_reserve(1)
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments
                .try_reserve(1)
                .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;
        }
        self.staged_flush = Some(StagedFlush {
            shard,
            encoded: built.encoded,
            manifest_segment: built.manifest_segment,
            pending_field_stats,
            next_seal_seq: built.next_seal_seq,
        });
        Ok(())
    }

    fn prepare_shard_flush(&mut self, shard: usize) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_some() || self.shards[shard].accumulator.document_count() == 0 {
            return Ok(());
        }
        let lease_docid_base = self.shards[shard]
            .current_lease_base
            .ok_or_else(|| invalid_state("nonempty accumulator has no Q1 lease"))?;
        let created_unix_s = self.created_unix_s()?;
        let segment_id = self.derive_segment_id(shard, lease_docid_base, created_unix_s)?;
        let documents = self.shards[shard]
            .identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::new(
                    identity.doc_ord,
                    &identity.document_id,
                    identity.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &self.shards[shard].accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base,
                created_unix_s,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Scalar,
        )?;
        let manifest_segment = manifest_segment(&encoded, self.next_seal_seq);
        let document_count = u32::try_from(self.shards[shard].accumulator.document_count())
            .map_err(|_| invalid_state("segment document count does not fit u32"))?;
        let mut pending_field_stats = self.pending_field_stats.clone();
        for field in self.shards[shard].accumulator.fields() {
            let entry = pending_field_stats
                .entry(field.field_ord())
                .or_insert((0, 0));
            entry.0 = entry
                .0
                .checked_add(field.total_tokens())
                .ok_or_else(|| invalid_state("pending field token count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(document_count)
                .ok_or_else(|| invalid_state("pending field document count overflow"))?;
        }
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        self.pending_segments
            .try_reserve(1)
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments
                .try_reserve(1)
                .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;
        }
        self.staged_flush = Some(StagedFlush {
            shard,
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        });
        Ok(())
    }

    async fn install_staged_flush(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        let Some(staged) = self.staged_flush.as_ref() else {
            return Ok(());
        };
        if let IndexBackend::Durable(writer) = &mut self.backend {
            let directory = writer
                .snapshot()
                .directory()
                .expect("KeeperWriter always owns a durable directory");
            let pending = staged.encoded.write_temp_retryable(directory)?;
            writer.publish_segment(cx, pending).await?;
        }
        let StagedFlush {
            shard,
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        } = self
            .staged_flush
            .take()
            .expect("staged flush remains present until installation succeeds");
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments.push(encoded);
        }
        self.pending_segments.push(manifest_segment);
        self.pending_field_stats = pending_field_stats;
        self.next_seal_seq = next_seal_seq;
        self.shards[shard].accumulator.reset();
        self.shards[shard].identities.clear();
        self.shards[shard].current_lease_base = None;
        Ok(())
    }

    fn derive_segment_id(
        &self,
        shard: usize,
        lease_base: u64,
        created_unix_s: i64,
    ) -> Result<u64, QuillIndexError> {
        self.derive_segment_id_avoiding(shard, lease_base, created_unix_s, &BTreeSet::new())
    }

    fn derive_segment_id_avoiding(
        &self,
        shard: usize,
        lease_base: u64,
        created_unix_s: i64,
        reserved_segment_ids: &BTreeSet<u64>,
    ) -> Result<u64, QuillIndexError> {
        let generation = self
            .backend
            .snapshot()
            .loaded_manifest()
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_state("manifest generation exhausted"))?;
        let schema_id = self.schema.schema_id()?;
        let mut batch_hasher = Xxh3::new();
        for identity in &self.shards[shard].identities {
            batch_hasher.update(&identity.content_hash.to_le_bytes());
        }
        let batch_digest = batch_hasher.digest();
        for salt in 0_u64..=u64::from(u16::MAX) {
            let mut preimage = [0_u8; 52];
            preimage[..8].copy_from_slice(&schema_id.to_le_bytes());
            preimage[8..16].copy_from_slice(&generation.to_le_bytes());
            preimage[16..24].copy_from_slice(&lease_base.to_le_bytes());
            preimage[24..32].copy_from_slice(&created_unix_s.to_le_bytes());
            preimage[32..36].copy_from_slice(&CURRENT_ENGINE_VERSION.to_le_bytes());
            preimage[36..44].copy_from_slice(&batch_digest.to_le_bytes());
            preimage[44..].copy_from_slice(&salt.to_le_bytes());
            let candidate = xxh3_64(&preimage);
            let collision = reserved_segment_ids.contains(&candidate)
                || self
                    .backend
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .segments
                    .iter()
                    .chain(&self.pending_segments)
                    .any(|segment| segment.segment_id == candidate);
            if !collision {
                return Ok(candidate);
            }
        }
        Err(invalid_state(
            "could not derive a collision-free segment id",
        ))
    }

    fn derive_policy_segment_id(
        &self,
        source_segment_ids: &[u64],
        domain: &[u8],
    ) -> Result<u64, QuillIndexError> {
        let generation = self
            .backend
            .snapshot()
            .loaded_manifest()
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_state("manifest generation exhausted"))?;
        let schema_id = self.schema.schema_id()?;
        let mut source_hasher = Xxh3::new();
        source_hasher.update(domain);
        for source in source_segment_ids {
            source_hasher.update(&source.to_le_bytes());
        }
        let source_digest = source_hasher.digest();
        for salt in 0_u64..=u64::from(u16::MAX) {
            let mut preimage = [0_u8; 40];
            preimage[..8].copy_from_slice(&schema_id.to_le_bytes());
            preimage[8..16].copy_from_slice(&generation.to_le_bytes());
            preimage[16..24].copy_from_slice(&source_digest.to_le_bytes());
            preimage[24..32].copy_from_slice(&self.next_seal_seq.to_le_bytes());
            preimage[32..].copy_from_slice(&salt.to_le_bytes());
            let candidate = xxh3_64(&preimage);
            if self
                .backend
                .snapshot()
                .loaded_manifest()
                .manifest
                .segments
                .iter()
                .all(|segment| segment.segment_id != candidate)
            {
                return Ok(candidate);
            }
        }
        Err(invalid_state(
            "could not derive a collision-free policy segment id",
        ))
    }

    fn created_unix_s(&self) -> Result<i64, QuillIndexError> {
        if self.config.deterministic_ingest {
            return Ok(0);
        }
        wall_clock_unix_s()
    }
}

impl QuillReader {
    #[inline]
    fn ranked_query_cache_enabled(&self) -> bool {
        #[cfg(feature = "conformance-internals")]
        {
            !self.conformance_controller.is_armed()
        }
        #[cfg(not(feature = "conformance-internals"))]
        {
            true
        }
    }

    #[cfg(not(feature = "conformance-internals"))]
    #[inline]
    #[allow(
        clippy::unused_self,
        reason = "keeps checkpoint call sites cfg-symmetric with the controller-backed variant"
    )]
    fn query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new(cx, phase, budget, upper_bound)
    }

    #[cfg(feature = "conformance-internals")]
    #[inline]
    fn query_checkpoint<'a>(
        &self,
        cx: &'a Cx,
        phase: &'static str,
        budget: u64,
        upper_bound: u64,
    ) -> Arc<QueryCheckpoint<'a>> {
        QueryCheckpoint::new_with_controller(
            cx,
            phase,
            budget,
            upper_bound,
            Arc::clone(&self.conformance_controller),
        )
    }

    fn default_parser(&self) -> Result<&DefaultQueryParser, QuillIndexError> {
        self.parser.as_ref().ok_or_else(|| {
            invalid_state("string query APIs are unavailable for this preparsed-only index")
        })
    }

    /// Parse and exhaustively execute one query over the published composite
    /// Keeper-plus-Delta snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, section validation, unsupported-query, or
    /// scorer/collector failures. Scalar accumulator state and installed but
    /// MANIFEST-unreferenced bytes are intentionally absent; frozen Delta
    /// epochs published in the process-local view are included.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let published = self.published_snapshot.load();
        self.search_paginated_on(cx, query, limit, offset, exact_count, published.as_ref())
    }

    #[cfg(feature = "pruning-conformance")]
    fn search_paginated_with_conformance_pruning_trace(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<(QuillSearchResult, ConformancePruningTraceReceipt), QuillIndexError> {
        let published = self.published_snapshot.load();
        let expected_segments = published.keeper_snapshot().segments().len();
        let guard = ConformancePruningTraceGuard::new(
            expected_segments,
            Arc::clone(&self.conformance_controller),
        );
        let result = self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            Some(guard.session()),
            published.as_ref(),
        )?;
        let receipt = guard.complete()?;
        Ok((result, receipt))
    }

    fn search_paginated_on(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.search_paginated_on_inner(
            cx,
            query,
            limit,
            offset,
            exact_count,
            #[cfg(feature = "pruning-conformance")]
            None,
            snapshot,
        )
    }

    fn search_paginated_on_inner(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            query_len = query.len(),
            segment_count,
            doc_count = snapshot.live_doc_count(),
            limit,
            offset,
            exact_count,
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        check_cancel(cx, "search")?;
        let cache_enabled = self.ranked_query_cache_enabled();
        #[cfg(feature = "pruning-conformance")]
        let cache_enabled = cache_enabled && pruning_trace.is_none();
        if cache_enabled
            && let Some(result) = self.published_snapshot.ranked_query_cache.get_raw(
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
            )
        {
            query_span.record(
                "result_count",
                u64::try_from(result.hits.len()).unwrap_or(u64::MAX),
            );
            if let Some(total_count) = result.total_count {
                query_span.record("total_count", total_count);
            }
            return Ok(result);
        }
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                query_root = tracing::field::Empty,
                query_shape_hash = tracing::field::Empty,
                query_nodes = tracing::field::Empty,
                query_depth = tracing::field::Empty,
                term_nodes = tracing::field::Empty,
                phrase_nodes = tracing::field::Empty,
                boolean_nodes = tracing::field::Empty,
                predicate_nodes = tracing::field::Empty,
                boost_nodes = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.default_parser()?.parse_lenient(query);
            let report = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            record_query_trace_shape(&parse_span, &parsed.query);
            tracing::debug!(
                target: crate::tracing_conventions::TARGET,
                must_not_duplicates_removed = report.must_not_duplicates_removed,
                filter_duplicates_removed = report.filter_duplicates_removed,
                duplicate_fields_removed = report.duplicate_fields_removed,
                boolean_levels_sorted = report.boolean_levels_sorted,
                glob_fields_canonicalized = report.glob_fields_canonicalized,
                "canonicalized parsed Quill query"
            );
            parsed
        };
        let result = self.execute_ranked_query_inner(
            cx,
            &parsed.query,
            snapshot,
            limit,
            offset,
            exact_count,
            parsed.diagnostics,
            #[cfg(feature = "pruning-conformance")]
            pruning_trace,
        )?;
        let result_count = u64::try_from(result.hits.len()).unwrap_or(u64::MAX);
        query_span.record("result_count", result_count);
        if let Some(total_count) = result.total_count {
            query_span.record("total_count", total_count);
        }
        if cache_enabled {
            self.published_snapshot.ranked_query_cache.insert_raw(
                snapshot.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
                &result,
            );
        }
        Ok(result)
    }

    fn scored_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        hydrate_metadata: bool,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        self.scored_results_pinned(cx, query, limit, hydrate_metadata)
            .map(|(results, _snapshot)| results)
    }

    /// Score on the currently published snapshot and return that exact
    /// snapshot `Arc` alongside the results, so callers can pin hydration to
    /// the generation that produced the scores (bd-8nqz.1).
    fn scored_results_pinned(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        hydrate_metadata: bool,
    ) -> Result<(Vec<ScoredResult>, Arc<QuillSearchSnapshot>), QuillIndexError> {
        let published = self.published_snapshot.load();
        let search = self.search_paginated_on(cx, query, limit, 0, false, published.as_ref())?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(search.hits.len())
            .map_err(|_| invalid_state("could not allocate lexical results"))?;
        for hit in search.hits {
            let metadata = hydrate_metadata
                .then(|| published.materialize_metadata(hit.global_docid))
                .transpose()?
                .flatten();
            results.push(ScoredResult {
                doc_id: hit.document_id.into(),
                score: hit.score,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(hit.score),
                rerank_score: None,
                explanation: None,
                metadata,
            });
        }
        Ok((results, published))
    }

    /// Collect the complete deterministic set of matching global document IDs.
    ///
    /// This is the scoreless collector lane: it lowers the same canonical
    /// default-parser tree as [`Self::search_paginated`], traverses every
    /// Keeper and published process-local Delta leaf without ranking, and
    /// returns sorted unique Q1 IDs. Scalar accumulator documents remain
    /// invisible.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        let published = self.published_snapshot.load();
        let snapshot = published.as_ref();
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            collector = "docset",
            query_len = query.len(),
            segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        check_cancel(cx, "collect_docids")?;
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                query_root = tracing::field::Empty,
                query_shape_hash = tracing::field::Empty,
                query_nodes = tracing::field::Empty,
                query_depth = tracing::field::Empty,
                term_nodes = tracing::field::Empty,
                phrase_nodes = tracing::field::Empty,
                boolean_nodes = tracing::field::Empty,
                predicate_nodes = tracing::field::Empty,
                boost_nodes = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.default_parser()?.parse_lenient(query);
            let _canonicalization = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            record_query_trace_shape(&parse_span, &parsed.query);
            parsed
        };
        let docids = self.execute_docid_query(cx, &parsed.query, snapshot)?;
        let result_count = u64::try_from(docids.len()).unwrap_or(u64::MAX);
        query_span.record("result_count", result_count);
        Ok(docids)
    }

    /// Execute one already-built query tree through the ranked mixed-state path.
    ///
    /// This remains feature-gated because the shipping surface accepts the
    /// language string and owns parser diagnostics. The conformance gauntlet
    /// needs the lower AST boundary to cover query classes that the default
    /// shipping parser cannot construct for every compiled schema.
    ///
    /// # Errors
    ///
    /// Returns the same typed lowering, collection, and cancellation failures
    /// as [`Self::search_paginated`].
    #[cfg(feature = "bench-internals")]
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        check_cancel(cx, "search_preparsed")?;
        let published = self.published_snapshot.load();
        let cache_enabled = self.ranked_query_cache_enabled();
        if cache_enabled
            && let Some(result) = self.published_snapshot.ranked_query_cache.get_preparsed(
                published.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
            )
        {
            return Ok(result);
        }
        let mut canonical = query.clone();
        let _canonicalization = canonicalize_query(&mut canonical);
        let result = self.execute_ranked_query(
            cx,
            &canonical,
            published.as_ref(),
            limit,
            offset,
            exact_count,
            Vec::new(),
        )?;
        if cache_enabled {
            self.published_snapshot.ranked_query_cache.insert_preparsed(
                published.snapshot_epoch(),
                query,
                limit,
                offset,
                exact_count,
                &result,
            );
        }
        Ok(result)
    }

    /// Execute one already-built query tree through the scoreless id-set path.
    ///
    /// # Errors
    ///
    /// Returns the same typed lowering, collection, and cancellation failures
    /// as [`Self::collect_docids`].
    #[cfg(feature = "bench-internals")]
    pub fn collect_preparsed_docids(
        &self,
        cx: &Cx,
        query: &Query,
    ) -> Result<Vec<u32>, QuillIndexError> {
        check_cancel(cx, "collect_preparsed_docids")?;
        let mut canonical = query.clone();
        let _canonicalization = canonicalize_query(&mut canonical);
        let published = self.published_snapshot.load();
        self.execute_docid_query(cx, &canonical, published.as_ref())
    }

    /// Bench-only: run one ranked sealed-segment collection with the fan-out
    /// decision forced, bypassing [`sealed_segment_fanout`]
    /// (bd-quill-e4-argus-3ycz.9 A/B). Returns `(global_docid, score_bits)`
    /// pairs so arms can assert bit-exact parity before timing. Delta
    /// snapshots are intentionally excluded: the lever under measurement is
    /// sealed-segment scoring only.
    ///
    /// `rank_pruning` overrides the shipping gate when `Some`, which is what
    /// lets a single binary time the pruned and exhaustive arms against each
    /// other (`bd-quill-e8-perf-doctrine-x4e4.5.1`). `None` uses the shipping
    /// decision. Overriding it to `true` never makes an ineligible shape prune:
    /// the runtime strategy selection in `argus::competitive_candidates` fails
    /// closed on its own, so this only controls whether pruning metadata is
    /// opened at all.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, collection, and cancellation
    /// failures as [`Self::search_paginated`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_search_sealed_forced(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        fan_out: bool,
        rank_pruning: Option<bool>,
    ) -> Result<Vec<(u32, u32)>, QuillIndexError> {
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        validate_query_lowering(&parsed.query, 1.0, self.schema)?;
        let published = self.published_snapshot.load();
        let snapshot = published.as_ref();
        let topdocs_root = limit != 0;
        let rank_pruning = topdocs_root
            && rank_pruning.unwrap_or_else(|| query_has_prunable_root_union(&parsed.query, 1.0));
        let mut collector = TopDocsCollector::new(limit, 0)?;
        let work_upper_bound =
            query_work_upper_bound(&parsed.query, snapshot, self.config.glob_expansion_limit)?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "search",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        self.collect_sealed_segments(
            cx,
            &checkpoint,
            &mut collector,
            &parsed.query,
            snapshot,
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            None,
            fan_out && !metering,
        )?;
        let collected = collector.finish()?;
        Ok(collected
            .hits
            .iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect())
    }

    #[cfg(any(test, feature = "bench-internals"))]
    fn execute_ranked_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        diagnostics: Vec<QueryDiagnostic>,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.execute_ranked_query_inner(
            cx,
            query,
            snapshot,
            limit,
            offset,
            exact_count,
            diagnostics,
            #[cfg(feature = "pruning-conformance")]
            None,
        )
    }

    fn execute_ranked_query_inner(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        diagnostics: Vec<QueryDiagnostic>,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        validate_query_lowering(query, 1.0, self.schema)?;
        let work_upper_bound =
            query_work_upper_bound(query, snapshot, self.config.glob_expansion_limit)?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "search",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let mut collector = if exact_count {
            TopDocsCollector::with_exact_count(limit, offset)?
        } else {
            TopDocsCollector::new(limit, offset)?
        };
        let topdocs_root = !exact_count && limit != 0;
        let rank_pruning = topdocs_root && query_has_prunable_root_union(query, 1.0);
        let sealed_docs: u64 = keeper
            .segments()
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let fan_out = sealed_segment_fanout(keeper.segments().len(), sealed_docs) && !metering;
        #[cfg(feature = "pruning-conformance")]
        let fan_out = fan_out
            && pruning_trace.is_none_or(|trace| trace.cancellation_arm_generation().is_none());
        self.collect_sealed_segments(
            cx,
            &checkpoint,
            &mut collector,
            query,
            snapshot,
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            pruning_trace,
            fan_out,
        )?;
        for delta in snapshot.delta_snapshots() {
            checkpoint.admit(QueryWorkKind::Segment, 1)?;
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                residency = "delta",
                lease_base = delta.lease_base(),
                doc_count = delta.live_document_count(),
                plan = tracing::field::Empty,
                segments_touched = 1_u64,
                pruning_windows = tracing::field::Empty,
                blocks_skipped = tracing::field::Empty,
                candidate_docs = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer = lower_query(
                cx,
                &checkpoint,
                query,
                1.0,
                QueryLeaf::Delta(delta),
                snapshot,
                self.schema,
                self.config.glob_expansion_limit,
                rank_pruning,
                topdocs_root,
            )?;
            collector.collect(&mut scorer, delta.as_ref())?;
            record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            segment_count,
            segments_touched = segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let collected = collector.finish()?;
        let total_count = collected.total_count;
        let mut hits = Vec::new();
        hits.try_reserve_exact(collected.hits.len())
            .map_err(|_| invalid_state("could not allocate final hit page"))?;
        for hit in collected.hits {
            let document_id = snapshot.materialize_document_id(hit.global_docid).ok_or(
                ArgusError::MissingExternalDocId {
                    global_docid: hit.global_docid,
                },
            )?;
            hits.push(QuillHit {
                document_id: document_id.to_string(),
                global_docid: hit.global_docid,
                score: hit.score,
            });
        }
        let result_count = u64::try_from(hits.len()).unwrap_or(u64::MAX);
        collect_span.record("result_count", result_count);
        if let Some(total_count) = total_count {
            collect_span.record("total_count", total_count);
        }
        Ok(QuillSearchResult {
            hits,
            total_count,
            doc_count: snapshot.live_doc_count(),
            diagnostics,
        })
    }

    /// Score every sealed segment into `collector`, serially or fanned across
    /// rayon (bd-quill-e4-argus-3ycz.9).
    ///
    /// Each fan-out task scores one segment into a private collector built
    /// from `collector`'s shape via [`TopDocsCollector::empty_like`]; the
    /// partials fold back in ascending segment order. The result is exactly
    /// the serial path's: the collector's total order (score `total_cmp`
    /// descending, then ascending global docid) makes the retained top set
    /// unique, so it is independent of both the rayon schedule and the merge
    /// order. Rank pruning stays rank-safe under the weaker per-segment local
    /// cutoffs; a shared cross-task cutoff is an explicit E8 follow-up lever,
    /// not part of this change. Delta snapshots never fan out: the mutable
    /// tail is small and keeps the serial path free of rayon overhead.
    fn collect_sealed_segments(
        &self,
        cx: &Cx,
        checkpoint: &QueryCheckpointHandle<'_>,
        collector: &mut TopDocsCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        rank_pruning: bool,
        topdocs_root: bool,
        #[cfg(feature = "pruning-conformance")] pruning_trace: Option<
            &ConformancePruningTraceSession,
        >,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        #[cfg(feature = "pruning-conformance")]
        if let Some(trace) = pruning_trace {
            trace.bind_execution_mode(if fan_out {
                ConformancePruningExecutionMode::Rayon
            } else {
                ConformancePruningExecutionMode::Serial
            })?;
        }
        if fan_out {
            check_cancel(cx, "search")?;
            let template = collector.empty_like()?;
            let schema = self.schema;
            let glob_expansion_limit = self.config.glob_expansion_limit;
            let segment_count = keeper.segments().len();
            let partials = keeper
                .segments()
                .par_iter()
                .enumerate()
                .map(|(segment_ordinal, segment)| {
                    debug_assert!(segment_ordinal < segment_count);
                    checkpoint.admit(QueryWorkKind::Segment, 1)?;
                    let mut local = template.empty_like()?;
                    let score_span = tracing::info_span!(
                        target: crate::tracing_conventions::TARGET,
                        crate::tracing_conventions::ARGUS_SCORE,
                        phase = "score",
                        segment_id = segment.manifest().segment_id,
                        doc_count = segment.doc_count(),
                        plan = tracing::field::Empty,
                        segments_touched = 1_u64,
                        pruning_windows = tracing::field::Empty,
                        blocks_skipped = tracing::field::Empty,
                        candidate_docs = tracing::field::Empty,
                        duration_us = tracing::field::Empty,
                    );
                    let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                    let _score_entered = score_span.enter();
                    let mut scorer = lower_query(
                        cx,
                        checkpoint,
                        query,
                        1.0,
                        QueryLeaf::Sealed(segment),
                        snapshot,
                        schema,
                        glob_expansion_limit,
                        rank_pruning,
                        topdocs_root,
                    )?;
                    local.collect(&mut scorer, segment)?;
                    record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
                    #[cfg(feature = "pruning-conformance")]
                    if let Some(trace) = pruning_trace {
                        trace.record_and_checkpoint(
                            u64::try_from(segment_ordinal).map_err(|_| {
                                invalid_state("Quill segment ordinal does not fit u64")
                            })?,
                            u64::from(segment.doc_count()),
                            scorer.conformance_union_refills(),
                            &self.conformance_controller,
                            cx,
                        )?;
                        check_cancel(cx, "search")?;
                    }
                    Ok(local)
                })
                .collect::<Result<Vec<_>, QuillIndexError>>()?;
            check_cancel(cx, "search")?;
            for partial in partials {
                collector.merge(partial)?;
            }
        } else {
            let segment_count = keeper.segments().len();
            for (segment_ordinal, segment) in keeper.segments().iter().enumerate() {
                debug_assert!(segment_ordinal < segment_count);
                checkpoint.admit(QueryWorkKind::Segment, 1)?;
                let score_span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    crate::tracing_conventions::ARGUS_SCORE,
                    phase = "score",
                    segment_id = segment.manifest().segment_id,
                    doc_count = segment.doc_count(),
                    plan = tracing::field::Empty,
                    segments_touched = 1_u64,
                    pruning_windows = tracing::field::Empty,
                    blocks_skipped = tracing::field::Empty,
                    candidate_docs = tracing::field::Empty,
                    duration_us = tracing::field::Empty,
                );
                let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                let _score_entered = score_span.enter();
                let mut scorer = lower_query(
                    cx,
                    checkpoint,
                    query,
                    1.0,
                    QueryLeaf::Sealed(segment),
                    snapshot,
                    self.schema,
                    self.config.glob_expansion_limit,
                    rank_pruning,
                    topdocs_root,
                )?;
                collector.collect(&mut scorer, segment)?;
                record_pruning_telemetry(&score_span, scorer.pruning_telemetry());
                #[cfg(feature = "pruning-conformance")]
                if let Some(trace) = pruning_trace {
                    trace.record_and_checkpoint(
                        u64::try_from(segment_ordinal)
                            .map_err(|_| invalid_state("Quill segment ordinal does not fit u64"))?,
                        u64::from(segment.doc_count()),
                        scorer.conformance_union_refills(),
                        &self.conformance_controller,
                        cx,
                    )?;
                    check_cancel(cx, "search")?;
                }
            }
        }
        Ok(())
    }

    /// Walk every sealed segment through the unscored id-set path, serially
    /// or fanned across rayon (bd-quill-e4-argus-3ycz.9 sibling lever).
    ///
    /// Determinism is free here: [`DocSetCollector::finish`] sorts and
    /// dedups, so any partial fold order produces the identical final id
    /// set. Delta snapshots stay on the caller's serial path.
    fn collect_docids_sealed(
        &self,
        cx: &Cx,
        checkpoint: &QueryCheckpointHandle<'_>,
        collector: &mut DocSetCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let keeper = snapshot.keeper_snapshot();
        if fan_out {
            check_cancel(cx, "collect_docids")?;
            let schema = self.schema;
            let glob_expansion_limit = self.config.glob_expansion_limit;
            let partials = keeper
                .segments()
                .par_iter()
                .map(|segment| {
                    checkpoint.admit(QueryWorkKind::Segment, 1)?;
                    let mut local = DocSetCollector::new();
                    let score_span = tracing::info_span!(
                        target: crate::tracing_conventions::TARGET,
                        crate::tracing_conventions::ARGUS_SCORE,
                        phase = "score",
                        collector = "docset",
                        segment_id = segment.manifest().segment_id,
                        doc_count = segment.doc_count(),
                        plan = "unscored",
                        segments_touched = 1_u64,
                        pruning_windows = 0_u64,
                        blocks_skipped = 0_u64,
                        candidate_docs = 0_u64,
                        duration_us = tracing::field::Empty,
                    );
                    let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                    let _score_entered = score_span.enter();
                    let mut scorer = lower_query_unscored(
                        cx,
                        checkpoint,
                        query,
                        1.0,
                        QueryLeaf::Sealed(segment),
                        snapshot,
                        schema,
                        glob_expansion_limit,
                    )?;
                    local.collect(&mut scorer, segment)?;
                    Ok(local)
                })
                .collect::<Result<Vec<_>, QuillIndexError>>()?;
            check_cancel(cx, "collect_docids")?;
            for partial in partials {
                collector.merge(partial)?;
            }
        } else {
            for segment in keeper.segments() {
                checkpoint.admit(QueryWorkKind::Segment, 1)?;
                let score_span = tracing::info_span!(
                    target: crate::tracing_conventions::TARGET,
                    crate::tracing_conventions::ARGUS_SCORE,
                    phase = "score",
                    collector = "docset",
                    segment_id = segment.manifest().segment_id,
                    doc_count = segment.doc_count(),
                    plan = "unscored",
                    segments_touched = 1_u64,
                    pruning_windows = 0_u64,
                    blocks_skipped = 0_u64,
                    candidate_docs = 0_u64,
                    duration_us = tracing::field::Empty,
                );
                let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
                let _score_entered = score_span.enter();
                let mut scorer = lower_query_unscored(
                    cx,
                    checkpoint,
                    query,
                    1.0,
                    QueryLeaf::Sealed(segment),
                    snapshot,
                    self.schema,
                    self.config.glob_expansion_limit,
                )?;
                collector.collect(&mut scorer, segment)?;
            }
        }
        Ok(())
    }

    /// Bench-only: run one unscored id-set collection over the sealed
    /// segments with the fan-out decision forced (bd-quill-e4-argus-3ycz.9
    /// sibling A/B). Delta snapshots are intentionally excluded.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, collection, and cancellation
    /// failures as [`Self::collect_docids`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_collect_docids_forced(
        &self,
        cx: &Cx,
        query: &str,
        fan_out: bool,
    ) -> Result<Vec<u32>, QuillIndexError> {
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        validate_query_lowering(&parsed.query, 1.0, self.schema)?;
        let published = self.published_snapshot.load();
        let snapshot = published.as_ref();
        let mut collector = DocSetCollector::new();
        let work_upper_bound =
            query_work_upper_bound(&parsed.query, snapshot, self.config.glob_expansion_limit)?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "collect_docids",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        self.collect_docids_sealed(
            cx,
            &checkpoint,
            &mut collector,
            &parsed.query,
            snapshot,
            fan_out && !metering,
        )?;
        Ok(collector.finish())
    }

    fn execute_docid_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        validate_query_lowering(query, 1.0, self.schema)?;
        let work_upper_bound =
            query_work_upper_bound(query, snapshot, self.config.glob_expansion_limit)?;
        let concrete_checkpoint = self.query_checkpoint(
            cx,
            "collect_docids",
            self.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        let keeper = snapshot.keeper_snapshot();
        let segment_count = keeper
            .segments()
            .len()
            .saturating_add(snapshot.delta_count());
        let mut collector = DocSetCollector::new();
        let sealed_docs: u64 = keeper
            .segments()
            .iter()
            .map(|segment| u64::from(segment.doc_count()))
            .sum();
        let fan_out = sealed_segment_fanout(keeper.segments().len(), sealed_docs) && !metering;
        self.collect_docids_sealed(cx, &checkpoint, &mut collector, query, snapshot, fan_out)?;
        for delta in snapshot.delta_snapshots() {
            checkpoint.admit(QueryWorkKind::Segment, 1)?;
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                collector = "docset",
                residency = "delta",
                lease_base = delta.lease_base(),
                doc_count = delta.live_document_count(),
                plan = "unscored",
                segments_touched = 1_u64,
                pruning_windows = 0_u64,
                blocks_skipped = 0_u64,
                candidate_docs = 0_u64,
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer = lower_query_unscored(
                cx,
                &checkpoint,
                query,
                1.0,
                QueryLeaf::Delta(delta),
                snapshot,
                self.schema,
                self.config.glob_expansion_limit,
            )?;
            collector.collect(&mut scorer, delta.as_ref())?;
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            collector = "docset",
            segment_count,
            segments_touched = segment_count,
            doc_count = snapshot.live_doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let docids = collector.finish();
        collect_span.record(
            "result_count",
            u64::try_from(docids.len()).unwrap_or(u64::MAX),
        );
        Ok(docids)
    }

    fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<QuillHit>, QuillIndexError> {
        Ok(self.search_paginated(cx, query, limit, 0, false)?.hits)
    }

    fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        let query_type = classify_query(query);
        if query_type == QueryExplanation::Empty {
            return Ok(Vec::new());
        }
        let snapshot = self.published_snapshot.load();
        let search = self.search_paginated_on(cx, query, limit, 0, false, snapshot.as_ref())?;
        let mut parsed = self.default_parser()?.parse_lenient(query);
        let _canonicalization = canonicalize_query(&mut parsed.query);
        let terms = compiled_snippet_terms(
            &parsed.query,
            snapshot.as_ref(),
            self.schema,
            self.config.glob_expansion_limit,
        )?;
        let analyzer = match self.schema.fields.get(usize::from(CONTENT_FIELD)) {
            Some(field) => match field.kind {
                FieldKind::Text { analyzer, .. } => analyzer,
                _ => return Err(invalid_state("content field is not text")),
            },
            None => return Err(invalid_state("schema has no content field")),
        };
        let mut generator = SnippetGenerator::new(analyzer, terms, snippet_config.clone());
        let mut results = Vec::new();
        results
            .try_reserve_exact(search.hits.len())
            .map_err(|_| invalid_state("could not allocate enriched lexical results"))?;
        for (rank, hit) in search.hits.into_iter().enumerate() {
            let snippet = snapshot
                .materialize_stored_value(CONTENT_FIELD, hit.global_docid)?
                .map(|content| {
                    String::from_utf8(content)
                        .map_err(|_| invalid_state("stored content contains non-UTF-8 bytes"))
                })
                .transpose()?
                .as_deref()
                .and_then(|content| generator.snippet(content));
            results.push(QuillSnippetHit {
                document_id: hit.document_id,
                score: hit.score,
                rank,
                snippet,
                query_type,
                metadata: snapshot.materialize_metadata(hit.global_docid)?,
            });
        }
        Ok(results)
    }

    fn segment_stats(&self) -> SegmentStats {
        let snapshot = self.published_snapshot.load();
        let mut stats = snapshot.keeper_snapshot().segment_stats();
        stats.delta_segments = snapshot.delta_count();
        stats.live_docs = usize::try_from(snapshot.live_doc_count()).unwrap_or(usize::MAX);
        stats.delta_memory_bytes = snapshot
            .delta_snapshots()
            .iter()
            .fold(0_u64, |total, delta| {
                total
                    .saturating_add(u64::try_from(delta.segment().bytes_used()).unwrap_or(u64::MAX))
            });
        stats
    }
}

#[cfg(feature = "bench-internals")]
impl PreparsedQuillIndex {
    /// Bind an owned Keeper snapshot for typed query execution.
    ///
    /// # Errors
    ///
    /// Returns typed configuration or snapshot-validation failures.
    pub fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let schema = snapshot.schema();
        let published_snapshot = Arc::new(SnapshotPublisher::new(Arc::new(snapshot), Vec::new())?);
        Ok(Self {
            reader: QuillReader {
                config,
                schema,
                parser: None,
                published_snapshot,
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
        })
    }

    /// Execute one already-built query tree against the pinned snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed lowering, collection, cancellation, or allocation
    /// failures.
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_preparsed_paginated(cx, query, limit, offset, exact_count)
    }

    /// Number of live documents in the pinned snapshot.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.reader.published_snapshot.load().live_doc_count()
    }
}

impl QuillSearchIndex {
    /// Open the latest published shipping-schema snapshot without acquiring the
    /// durable writer lease.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, configuration, recovery, schema, or parser
    /// failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        validate_non_durable_quarantine(&config)?;
        check_cancel(cx, "read-only index open")?;
        let directory = directory.into();
        let open_directory = directory.clone();
        let snapshot =
            spawn_blocking(move || KeeperSnapshot::open(open_directory, DEFAULT_SCHEMA)).await?;
        check_cancel(cx, "read-only index open")?;
        let published_snapshot = Arc::new(SnapshotPublisher::new(Arc::new(snapshot), Vec::new())?);
        Ok(Self {
            reader: QuillReader {
                parser: Some(DefaultQueryParser::new(DEFAULT_SCHEMA)?),
                config,
                schema: DEFAULT_SCHEMA,
                published_snapshot,
                #[cfg(feature = "conformance-internals")]
                conformance_controller: Arc::default(),
            },
            directory,
        })
    }

    /// Durable index directory bound to this published snapshot.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.directory
    }

    /// Number of live documents in this published snapshot.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.reader.published_snapshot.load().live_doc_count()
    }

    /// Durable MANIFEST generation pinned by the currently published snapshot.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.reader.published_snapshot.load().keeper_generation()
    }

    /// Refresh this read-only handle to the latest durable MANIFEST.
    ///
    /// Queries that already loaded the prior snapshot remain pinned to it;
    /// later queries observe the successor after the atomic publication swap.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, recovery, schema, or snapshot-transition
    /// failures. A regressed durable generation is rejected.
    pub async fn refresh(&self, cx: &Cx) -> Result<bool, QuillIndexError> {
        check_cancel(cx, "read-only index refresh")?;
        let directory = self.directory.clone();
        let snapshot =
            spawn_blocking(move || KeeperSnapshot::open(directory, DEFAULT_SCHEMA)).await?;
        check_cancel(cx, "read-only index refresh")?;

        let current_generation = self.reader.published_snapshot.load().keeper_generation();
        let next_generation = snapshot.loaded_manifest().manifest.generation;
        if next_generation == current_generation {
            return Ok(false);
        }
        self.reader
            .published_snapshot
            .publish_complete(Arc::new(snapshot), Vec::new())?;
        Ok(true)
    }

    /// Execute a paginated ranked query against the pinned publication.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, scoring, or collection
    /// failures.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_paginated(cx, query, limit, offset, exact_count)
    }

    /// Search for full lexical results with canonical stored metadata.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, metadata, or allocation
    /// failures.
    pub fn search_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        self.reader.scored_results(cx, query, limit, true)
    }

    /// Search the identifier-only lane.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, scoring, or collection failures.
    pub fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<QuillHit>, QuillIndexError> {
        self.reader.search_doc_ids(cx, query, limit)
    }

    /// Search the pinned publication and generate snippets from stored content.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-content, snippet,
    /// UTF-8, or allocation failures.
    pub fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        self.reader
            .search_with_snippets(cx, query, limit, snippet_config)
    }

    /// Collect every matching global document id from the pinned publication.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parsing, lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.collect_docids(cx, query)
    }
}

impl QuillIndex {
    fn from_writer(writer: QuillWriterState) -> Self {
        let directory = writer.directory().map(Path::to_path_buf);
        let reader = writer.reader.clone();
        Self {
            reader,
            writer: Arc::new(Mutex::with_name("quill.index.writer", writer)),
            directory,
        }
    }

    #[cfg(test)]
    fn from_backend(
        backend: IndexBackend,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::from_backend(
            backend, schema, config,
        )?))
    }

    #[cfg(test)]
    fn writer_mut(&mut self) -> &mut QuillWriterState {
        Arc::get_mut(&mut self.writer)
            .expect("exclusive test index owns its writer mutex")
            .get_mut()
            .expect("exclusive test index cannot have a poisoned writer")
    }

    async fn lock_writer(
        &self,
        cx: &Cx,
        phase: &'static str,
    ) -> Result<OwnedMutexGuard<QuillWriterState>, QuillIndexError> {
        OwnedMutexGuard::lock(Arc::clone(&self.writer), cx)
            .await
            .map_err(|error| map_lock_error_for_index(error, phase))
    }

    #[cfg(test)]
    fn execute_ranked_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        limit: usize,
        offset: usize,
        exact_count: bool,
        diagnostics: Vec<QueryDiagnostic>,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader.execute_ranked_query(
            cx,
            query,
            snapshot,
            limit,
            offset,
            exact_count,
            diagnostics,
        )
    }

    #[cfg(test)]
    fn execute_docid_query(
        &self,
        cx: &Cx,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
    ) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.execute_docid_query(cx, query, snapshot)
    }

    #[cfg(test)]
    fn collect_sealed_segments(
        &self,
        cx: &Cx,
        collector: &mut TopDocsCollector,
        query: &Query,
        snapshot: &QuillSearchSnapshot,
        rank_pruning: bool,
        topdocs_root: bool,
        fan_out: bool,
    ) -> Result<(), QuillIndexError> {
        let work_upper_bound =
            query_work_upper_bound(query, snapshot, self.reader.config.glob_expansion_limit)?;
        let concrete_checkpoint = self.reader.query_checkpoint(
            cx,
            "search",
            self.reader.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = concrete_checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = concrete_checkpoint;
        self.reader.collect_sealed_segments(
            cx,
            &checkpoint,
            collector,
            query,
            snapshot,
            rank_pruning,
            topdocs_root,
            #[cfg(feature = "pruning-conformance")]
            None,
            fan_out && !metering,
        )
    }

    /// Create a shipping-schema index or open an existing compatible index.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::create(cx, directory, config).await?,
        ))
    }

    /// Open an existing shipping-schema index for mutation and search.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::open(cx, directory, config).await?,
        ))
    }

    /// Open an existing shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, or durability
    /// failures.
    #[cfg(feature = "durability")]
    pub async fn open_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::open_durable(cx, directory, config, protector).await?,
        ))
    }

    /// Create or open a shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, creation, or
    /// durability failures.
    #[cfg(feature = "durability")]
    pub async fn create_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::create_durable(cx, directory, config, protector).await?,
        ))
    }

    /// Construct an owned-buffer genesis index without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    pub fn in_memory(config: QuillConfig) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::in_memory(config)?))
    }

    /// Build an owned-buffer index for one explicit compile-time schema.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    #[cfg(feature = "bench-internals")]
    pub fn in_memory_with_schema(
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(QuillWriterState::in_memory_with_schema(
            schema, config,
        )?))
    }

    /// Bind an existing owned Keeper snapshot to the private writer facade.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or snapshot-validation failures.
    #[cfg(feature = "bench-internals")]
    pub fn from_in_memory_snapshot(
        snapshot: KeeperSnapshot,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Ok(Self::from_writer(
            QuillWriterState::from_in_memory_snapshot(snapshot, config)?,
        ))
    }

    /// Current committed immutable snapshot.
    #[must_use]
    pub fn snapshot(&self) -> Arc<KeeperSnapshot> {
        self.reader.published_snapshot.load().keeper_snapshot_arc()
    }

    /// Current process-local Keeper plus Delta snapshot.
    #[must_use]
    pub fn search_snapshot(&self) -> Arc<QuillSearchSnapshot> {
        self.reader.published_snapshot.load()
    }

    /// Clone the deterministic real-`Cx` cancellation requester used by the
    /// method-bound conformance harness.
    ///
    /// This API and its per-index state do not exist in normal builds.
    #[cfg(feature = "conformance-internals")]
    #[must_use]
    pub fn conformance_cancellation_controller(&self) -> Arc<ConformanceCancellationController> {
        Arc::clone(&self.reader.conformance_controller)
    }

    /// Execute one ranked search with an invocation-bound pruning trace.
    ///
    /// # Errors
    ///
    /// Returns the ordinary typed query failure, or a typed conformance error
    /// if the exact invocation does not produce one dense receipt per sealed
    /// segment. Partial receipts are failed and discarded before return.
    #[cfg(feature = "pruning-conformance")]
    #[doc(hidden)]
    pub fn search_paginated_with_conformance_pruning_trace(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<(QuillSearchResult, ConformancePruningTraceReceipt), QuillIndexError> {
        self.reader.search_paginated_with_conformance_pruning_trace(
            cx,
            query,
            limit,
            offset,
            exact_count,
        )
    }

    /// Capture a fixed-size digest receipt for the complete retained scalar
    /// writer transaction.
    ///
    /// This conformance-only surface hashes actual canonical pending FSLX
    /// bytes and identity rows rather than reporting only a dirty boolean.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the writer is busy or any retained MANIFEST
    /// proposal cannot be canonically encoded.
    #[cfg(feature = "conformance-internals")]
    pub fn conformance_pending_writer_state(
        &self,
    ) -> Result<ConformancePendingWriterState, QuillIndexError> {
        let writer = self.writer.try_lock().map_err(map_try_lock_error)?;
        writer.conformance_pending_writer_state()
    }

    /// Resolve one published document through IDHASH and return its IDMAP hash.
    ///
    /// The probe is allocation-free for sealed segments. It returns `None`
    /// when the identifier is absent or tombstoned.
    ///
    /// # Errors
    ///
    /// Returns typed identity corruption or an invalid live-Delta witness.
    pub fn document_witness(
        &self,
        document_id: &str,
    ) -> Result<Option<QuillDocumentWitness>, QuillIndexError> {
        self.search_snapshot().resolve_document_witness(document_id)
    }

    /// Durable index directory, or `None` for an owned-buffer index.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Alias for [`Self::directory`] used by engine-neutral callers.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.directory()
    }

    /// Number of live documents in the published Keeper-plus-Delta view.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.reader.published_snapshot.load().live_doc_count()
    }

    /// Whether the writer holds documents or installed segments not yet visible.
    #[must_use]
    pub fn has_uncommitted_changes(&self) -> bool {
        self.writer
            .try_lock()
            .map_or(true, |writer| writer.has_uncommitted_changes())
    }

    /// Atomically publish the complete process-local Delta table.
    ///
    /// # Errors
    ///
    /// Rejects schema or generation drift, overlapping leases, Keeper-range
    /// overlap, allocation failure, process-local epoch exhaustion, or a busy
    /// writer.
    pub fn publish_delta_table(
        &self,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let writer = self.writer.try_lock().map_err(map_try_lock_error)?;
        writer.publish_delta_table(deltas)
    }

    /// Seal one published Delta epoch into Keeper.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, validation, encoding, or
    /// publication failures.
    pub async fn seal_delta_snapshot(
        &self,
        cx: &Cx,
        sealed: Arc<DeltaSnapshot>,
        replacement_deltas: Vec<Arc<DeltaSnapshot>>,
        input: DeltaFlushInput,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "Delta seal writer lock").await?;
        writer
            .seal_delta_snapshot(cx, sealed, replacement_deltas, input)
            .await
    }

    /// Resume a retained Delta seal proposal.
    ///
    /// # Errors
    ///
    /// Returns a typed error when no seal awaits reconciliation or when the
    /// writer lock, segment installation, or publication fails.
    pub async fn resume_pending_delta_seal(
        &self,
        cx: &Cx,
    ) -> Result<Arc<QuillSearchSnapshot>, QuillIndexError> {
        let mut writer = self
            .lock_writer(cx, "Delta seal resume writer lock")
            .await?;
        writer.resume_pending_delta_seal(cx).await
    }

    /// Accumulate one document into the scalar writer.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, duplicate-id, accumulation,
    /// flush, or publication failures.
    pub async fn index_document(
        &self,
        cx: &Cx,
        document: &IndexableDocument,
    ) -> Result<(), QuillIndexError> {
        self.index_documents(cx, std::slice::from_ref(document))
            .await
    }

    /// Accumulate one bounded batch into the scalar writer.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, writer-lock, duplicate-id, accumulation,
    /// flush, or publication failures.
    pub async fn index_documents(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let mut writer = self.lock_writer(cx, "index writer lock").await?;
        writer.index_documents(cx, documents).await
    }

    /// Seal pending writes and atomically publish the next MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, flush, segment-install, manifest-transition,
    /// durability, or cancellation failures.
    pub async fn commit(&self, cx: &Cx) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "commit writer lock").await?;
        writer.commit(cx).await?;
        drop(writer);
        Ok(self.snapshot())
    }

    /// Finish an explicitly configured bulk build with one final
    /// bound-consecutive concat pass and clear the durable bulk marker.
    ///
    /// Intermediate MANIFEST generations remain crash-resumable according to
    /// [`QuillConfig::bulk_publish_segment_cadence`]. After this succeeds the
    /// shard set has at most one sealed segment and ordinary tier merging is
    /// re-enabled for subsequent commits.
    ///
    /// # Errors
    ///
    /// Rejects a non-bulk writer and propagates writer-lock, seal, concat,
    /// publication, or cancellation failures.
    pub async fn finish_bulk_load(&self, cx: &Cx) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "bulk finish writer lock").await?;
        writer.finish_bulk_load(cx).await?;
        drop(writer);
        Ok(self.snapshot())
    }

    /// Replace one exact committed manifest run with a Q1 concat merge.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted state, cancellation, a
    /// nonconsecutive source run, codec failure, or publication failure.
    pub async fn concat_merge(
        &self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Arc<KeeperSnapshot>, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "concat merge writer lock").await?;
        writer
            .concat_merge(cx, source_segment_ids, output_segment_id, created_unix_s)
            .await?;
        drop(writer);
        Ok(self.snapshot())
    }

    /// Fold tombstones into immutable positional holes for eligible segments.
    ///
    /// # Errors
    ///
    /// Rejects writer-lock failure, uncommitted scalar or Delta state, invalid
    /// policy values, cancellation, source/codec failures, or publication
    /// failures.
    pub async fn compact(
        &self,
        cx: &Cx,
        policy: CompactionPolicy,
    ) -> Result<CompactionReport, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "compaction writer lock").await?;
        writer.compact(cx, policy).await
    }

    /// Delete one live document id and publish the successor snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, identity-resolution, or
    /// successor-publication failures.
    pub async fn delete_document(
        &self,
        cx: &Cx,
        document_id: &str,
    ) -> Result<bool, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete document writer lock").await?;
        writer.delete_document(cx, document_id).await
    }

    /// Delete a bounded batch of live document IDs and publish one successor
    /// snapshot.
    ///
    /// IDs that are already absent are ignored. Publishing once for the batch
    /// keeps bulk tombstone setup from paying one durable MANIFEST transition
    /// per ID while preserving the same atomic visibility boundary.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, identity-resolution, or
    /// successor-publication failures.
    pub async fn delete_documents(
        &self,
        cx: &Cx,
        document_ids: &[&str],
    ) -> Result<usize, QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete documents writer lock").await?;
        writer.delete_documents(cx, document_ids).await
    }

    /// Delete every live document and publish an empty successor snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed writer-lock, cancellation, or successor-publication
    /// failures.
    pub async fn delete_all(&self, cx: &Cx) -> Result<(), QuillIndexError> {
        let mut writer = self.lock_writer(cx, "delete all writer lock").await?;
        writer.delete_all(cx).await
    }

    async fn upsert_documents(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let mut unique = BTreeSet::new();
        for document in documents {
            if !unique.insert(document.id.as_str()) {
                return Err(invalid_state(format!(
                    "duplicate document id {:?} in one upsert batch",
                    document.id
                )));
            }
        }

        let mut writer = self.lock_writer(cx, "upsert writer lock").await?;
        writer.upsert_documents(cx, documents).await
    }

    /// Parse and exhaustively execute one query over the published snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, section-validation,
    /// scorer, collector, or allocation failures.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_paginated(cx, query, limit, offset, exact_count)
    }

    /// Search for full lexical results with canonical stored metadata.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-metadata, or
    /// allocation failures.
    pub fn search_results(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>, QuillIndexError> {
        self.reader.scored_results(cx, query, limit, true)
    }

    /// Search the identifier-only lane used by hot lexical consumers.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, section-validation,
    /// scorer, collector, or allocation failures.
    pub fn search_doc_ids(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> Result<Vec<QuillHit>, QuillIndexError> {
        self.reader.search_doc_ids(cx, query, limit)
    }

    /// Search with the incumbent enriched result shape.
    ///
    /// Snippets use the same pinned snapshot, compiled query terms, global
    /// document frequencies, and stored source content as the ranked search.
    /// Schemas without stored content return `None` for that hit's snippet.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, stored-content,
    /// snippet-term compilation, UTF-8, or allocation failures.
    pub fn search_with_snippets(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        snippet_config: &SnippetConfig,
    ) -> Result<Vec<QuillSnippetHit>, QuillIndexError> {
        self.reader
            .search_with_snippets(cx, query, limit, snippet_config)
    }

    /// Collect the complete deterministic set of matching global document IDs.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.collect_docids(cx, query)
    }

    /// Execute one already-built query tree through the ranked mixed-state path.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, cursor, or allocation failures.
    #[cfg(feature = "bench-internals")]
    pub fn search_preparsed_paginated(
        &self,
        cx: &Cx,
        query: &Query,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        self.reader
            .search_preparsed_paginated(cx, query, limit, offset, exact_count)
    }

    /// Execute one already-built query tree through the scoreless id-set path.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, query execution, or allocation failures.
    #[cfg(feature = "bench-internals")]
    pub fn collect_preparsed_docids(
        &self,
        cx: &Cx,
        query: &Query,
    ) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.collect_preparsed_docids(cx, query)
    }

    /// Bench-only forced sealed fan-out path for scoreless id-set collection.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, collection, and cancellation
    /// failures as [`Self::collect_docids`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_collect_docids_forced(
        &self,
        cx: &Cx,
        query: &str,
        fan_out: bool,
    ) -> Result<Vec<u32>, QuillIndexError> {
        self.reader.bench_collect_docids_forced(cx, query, fan_out)
    }

    /// Bench-only forced sealed fan-out path.
    ///
    /// # Errors
    ///
    /// Returns the same typed parsing, lowering, search, and cancellation
    /// failures as [`Self::search`].
    #[cfg(feature = "bench-internals")]
    pub fn bench_search_sealed_forced(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        fan_out: bool,
        rank_pruning: Option<bool>,
    ) -> Result<Vec<(u32, u32)>, QuillIndexError> {
        self.reader
            .bench_search_sealed_forced(cx, query, limit, fan_out, rank_pruning)
    }
}

impl SegmentStatsProvider for QuillIndex {
    fn segment_stats(&self) -> SegmentStats {
        self.reader.segment_stats()
    }
}

impl SegmentStatsProvider for QuillSearchIndex {
    fn segment_stats(&self) -> SegmentStats {
        self.reader.segment_stats()
    }
}

impl LexicalSearch for QuillIndex {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.reader
                .scored_results(cx, query, limit, true)
                .map_err(SearchError::from)
        })
    }

    fn search_fusion_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.reader
                .scored_results(cx, query, limit, false)
                .map_err(SearchError::from)
        })
    }

    fn fusion_metadata_is_deferred(&self) -> bool {
        true
    }

    fn hydrate_fusion_metadata<'a>(
        &'a self,
        cx: &'a Cx,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            check_cancel(cx, "fusion metadata hydration").map_err(SearchError::from)?;
            let snapshot = self.search_snapshot();
            for result in results
                .iter_mut()
                .filter(|result| result.lexical_score.is_some())
            {
                let Some(global_docid) = snapshot
                    .resolve_document_id(result.doc_id.as_str())
                    .map_err(SearchError::from)?
                else {
                    result.metadata = None;
                    continue;
                };
                result.metadata = snapshot
                    .materialize_metadata(global_docid)
                    .map_err(SearchError::from)?;
            }
            Ok(())
        })
    }

    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, std::slice::from_ref(doc))
                .await
                .map_err(SearchError::from)
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, docs)
                .await
                .map_err(SearchError::from)
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            Self::commit(self, cx)
                .await
                .map(drop)
                .map_err(SearchError::from)
        })
    }

    fn doc_count(&self) -> usize {
        usize::try_from(Self::doc_count(self)).unwrap_or(usize::MAX)
    }
}

/// Backend tag carried by Quill hydration contexts (bd-8nqz.1).
pub const QUILL_LEXICAL_BACKEND: &str = "quill";

/// Feature-only snapshot pin that carries the deterministic hydration
/// checkpoint alongside the immutable scoring generation.
#[cfg(feature = "conformance-internals")]
struct ConformanceHydrationPin {
    snapshot: Arc<QuillSearchSnapshot>,
    controller: Arc<ConformanceCancellationController>,
}

/// Shared generation-pinned candidate search for Quill readers (bd-8nqz.1).
fn quill_search_candidates<'a>(
    reader: &'a QuillReader,
    cx: &'a Cx,
    query: &'a str,
    limit: usize,
) -> SearchFuture<'a, LexicalCandidateBatch> {
    Box::pin(async move {
        let (results, snapshot) = reader
            .scored_results_pinned(cx, query, limit, false)
            .map_err(SearchError::from)?;
        #[cfg(not(feature = "conformance-internals"))]
        let context = LexicalHydrationContext::new(QUILL_LEXICAL_BACKEND, Box::new(snapshot));
        #[cfg(feature = "conformance-internals")]
        let context = LexicalHydrationContext::new(
            QUILL_LEXICAL_BACKEND,
            Box::new(ConformanceHydrationPin {
                snapshot,
                controller: Arc::clone(&reader.conformance_controller),
            }),
        );
        Ok(LexicalCandidateBatch::deferred(results, context))
    })
}

/// Shared pinned hydration for Quill readers (bd-8nqz.1): metadata is
/// materialized from the exact snapshot that scored the batch, never from a
/// newer published generation.
fn quill_hydrate_candidates<'a>(
    cx: &'a Cx,
    context: Option<&'a LexicalHydrationContext>,
    results: &'a mut [ScoredResult],
) -> SearchFuture<'a, ()> {
    Box::pin(async move {
        check_cancel(cx, "fusion metadata hydration").map_err(SearchError::from)?;
        let Some(context) = context else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: "deferred Quill candidates require their batch context; \
                         none was provided"
                    .into(),
            });
        };
        #[cfg(not(feature = "conformance-internals"))]
        let Some(snapshot) = context.downcast_ref::<Arc<QuillSearchSnapshot>>() else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: format!(
                    "hydration context from backend {:?} is not a Quill snapshot pin; \
                     refusing cross-engine hydration",
                    context.backend()
                )
                .into(),
            });
        };
        #[cfg(feature = "conformance-internals")]
        let Some(conformance_pin) = context.downcast_ref::<ConformanceHydrationPin>() else {
            return Err(SearchError::SubsystemError {
                subsystem: "quill.hydration",
                source: format!(
                    "hydration context from backend {:?} is not a Quill snapshot pin; \
                     refusing cross-engine hydration",
                    context.backend()
                )
                .into(),
            });
        };
        #[cfg(feature = "conformance-internals")]
        let snapshot = &conformance_pin.snapshot;
        for result in results
            .iter_mut()
            .filter(|result| result.lexical_score.is_some())
        {
            #[cfg(feature = "conformance-internals")]
            conformance_pin
                .controller
                .checkpoint(ConformanceCancellationStage::FusionHydration, cx);
            check_cancel(cx, "fusion metadata hydration").map_err(SearchError::from)?;
            let Some(global_docid) = snapshot
                .resolve_document_id(result.doc_id.as_str())
                .map_err(SearchError::from)?
            else {
                result.metadata = None;
                continue;
            };
            result.metadata = snapshot
                .materialize_metadata(global_docid)
                .map_err(SearchError::from)?;
        }
        Ok(())
    })
}

impl LexicalRead for QuillIndex {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.reader
                .scored_results(cx, query, limit, true)
                .map_err(SearchError::from)
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        quill_search_candidates(&self.reader, cx, query, limit)
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        quill_hydrate_candidates(cx, context, results)
    }

    fn doc_count(&self) -> usize {
        usize::try_from(Self::doc_count(self)).unwrap_or(usize::MAX)
    }
}

impl LexicalWrite for QuillIndex {
    fn index_document<'a>(
        &'a self,
        cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, std::slice::from_ref(doc))
                .await
                .map_err(SearchError::from)
        })
    }

    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            self.upsert_documents(cx, docs)
                .await
                .map_err(SearchError::from)
        })
    }

    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            Self::commit(self, cx)
                .await
                .map(drop)
                .map_err(SearchError::from)
        })
    }
}

/// The read-only reader implements ONLY [`LexicalRead`] — no writer lease is
/// ever acquired on a read-only open (bd-8nqz.1).
impl LexicalRead for QuillSearchIndex {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            self.reader
                .scored_results(cx, query, limit, true)
                .map_err(SearchError::from)
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        quill_search_candidates(&self.reader, cx, query, limit)
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        quill_hydrate_candidates(cx, context, results)
    }

    fn doc_count(&self) -> usize {
        usize::try_from(Self::doc_count(self)).unwrap_or(usize::MAX)
    }
}

impl From<QuillIndexError> for SearchError {
    fn from(error: QuillIndexError) -> Self {
        match error {
            QuillIndexError::Config(source) => source,
            QuillIndexError::Keeper(source) => source.into(),
            QuillIndexError::Quill(source) => source.into(),
            QuillIndexError::Cancelled { phase } => Self::Cancelled {
                phase: phase.to_owned(),
                reason: "Quill observed request cancellation".to_owned(),
            },
            source => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(source),
            },
        }
    }
}

fn record_pruning_telemetry(span: &tracing::Span, telemetry: PruningTelemetry) {
    span.record("plan", telemetry.plan());
    span.record("pruning_windows", telemetry.pruning_windows());
    span.record("blocks_skipped", telemetry.blocks_skipped());
    span.record("candidate_docs", telemetry.candidate_docs());
}

fn record_snapshot_fields(span: &tracing::Span, snapshot: &KeeperSnapshot) {
    let manifest = &snapshot.loaded_manifest().manifest;
    span.record("generation", manifest.generation);
    span.record(
        "segment_count",
        u64::try_from(manifest.segments.len()).unwrap_or(u64::MAX),
    );
    span.record("doc_count", snapshot.doc_count());
}

fn validate_config(config: &QuillConfig) -> Result<(), QuillIndexError> {
    config.validate().map_err(QuillIndexError::Config)
}

fn validate_non_durable_quarantine(config: &QuillConfig) -> Result<(), QuillIndexError> {
    if config.quarantine_on_unrepairable {
        return Err(QuillIndexError::Config(SearchError::InvalidConfig {
            field: "quarantine_on_unrepairable".to_owned(),
            value: "true".to_owned(),
            reason: "requires open_durable/create_durable and a FileProtector".to_owned(),
        }));
    }
    Ok(())
}

/// Minimum total sealed live-document count before ranked queries fan
/// per-segment scoring across rayon (bd-quill-e4-argus-3ycz.9). Follows the
/// house `PARALLEL_THRESHOLD` philosophy from
/// `frankensearch-index/src/search.rs`: picked by measurement, starting at
/// the same 10k floor. Below the gate, per-task scorer and collector setup
/// costs more than the fan-out returns.
const SEGMENT_FANOUT_THRESHOLD: u64 = 10_000;
/// A fragmented snapshot has enough independent scorer setup and posting work
/// to amortize rayon even below the document-count gate. The 2-segment
/// below-gate control remains serial; watch-mode replacement batches cross
/// this threshold as immutable leaves accumulate.
const SEGMENT_COUNT_FANOUT_THRESHOLD: usize = 8;

/// Decide whether sealed-segment scoring fans across rayon. Single-segment
/// snapshots never fan out (there is nothing to overlap). Larger corpora use
/// the document-count gate, while sufficiently fragmented smaller snapshots
/// use the segment-count gate.
const fn sealed_segment_fanout(segment_count: usize, total_sealed_docs: u64) -> bool {
    segment_count >= 2
        && (total_sealed_docs >= SEGMENT_FANOUT_THRESHOLD
            || segment_count >= SEGMENT_COUNT_FANOUT_THRESHOLD)
}

fn query_work_upper_bound(
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    glob_expansion_limit: usize,
) -> Result<u64, QuillIndexError> {
    let mut shape = QueryWorkShape::default();
    shape.visit(query, glob_expansion_limit);

    let keeper = snapshot.keeper_snapshot();
    let segment_count = keeper
        .segments()
        .len()
        .saturating_add(snapshot.delta_count());
    let segment_count = u64::try_from(segment_count).unwrap_or(u64::MAX);
    let keeper_count = u64::try_from(keeper.segments().len()).unwrap_or(u64::MAX);

    let mut physical_docs = 0_u64;
    let mut posting_blocks_per_stream = 0_u64;
    let mut dictionary_blocks = 0_u64;
    for segment in keeper.segments() {
        let docs = u64::from(segment.doc_count());
        physical_docs = physical_docs.saturating_add(docs);
        posting_blocks_per_stream = posting_blocks_per_stream.saturating_add(docs.div_ceil(128));
        let bytes = required_section(segment, SectionKind::TERMDICT)?;
        let count_bytes = bytes
            .get(..4)
            .ok_or_else(|| invalid_state("TERMDICT is shorter than its block-count header"))?;
        let count = u32::from_le_bytes([
            count_bytes[0],
            count_bytes[1],
            count_bytes[2],
            count_bytes[3],
        ]);
        dictionary_blocks = dictionary_blocks.saturating_add(u64::from(count));
    }
    for delta in snapshot.delta_snapshots() {
        let docs = u64::try_from(delta.live_document_count()).unwrap_or(u64::MAX);
        physical_docs = physical_docs.saturating_add(docs);
        posting_blocks_per_stream = posting_blocks_per_stream.saturating_add(docs.div_ceil(128));
        let terms = u64::try_from(delta.segment().sorted_terms().len()).unwrap_or(u64::MAX);
        dictionary_blocks = dictionary_blocks.saturating_add(terms.div_ceil(16));
    }

    // A rank-pruning fork can traverse a sealed posting stream independently
    // of the scorer-owned cursor. Two full traversals are therefore a safe
    // ceiling for every syntactic stream.
    let posting_blocks = shape
        .posting_streams
        .saturating_mul(posting_blocks_per_stream)
        .saturating_mul(2);
    // Snapshot-level expansions run while each segment scorer is lowered.
    let scanned_dictionary_blocks = shape
        .dictionary_scans
        .saturating_mul(dictionary_blocks)
        .saturating_mul(segment_count);
    // Exact term statistics probe every sealed dictionary for each lowered
    // segment, then the segment-local cursor performs one more indexed probe.
    // Every posting stream is opened through `lower_leaf_term`, including
    // terms materialized by range and glob expansion. Each opening probes all
    // sealed dictionaries for snapshot statistics and then the leaf-local
    // dictionary, so use the complete stream ceiling rather than only the
    // syntactic exact-term count.
    let indexed_dictionary_blocks = shape.posting_streams.saturating_mul(
        segment_count
            .saturating_mul(keeper_count)
            .saturating_add(keeper_count),
    );
    // Existing phrase lowering materializes positioned rows before candidate
    // verification. Bound both the decode and verification passes.
    let position_docs = shape
        .phrase_streams
        .saturating_add(shape.phrase_fields)
        .saturating_mul(physical_docs);

    Ok(segment_count
        .saturating_add(posting_blocks)
        .saturating_add(scanned_dictionary_blocks)
        .saturating_add(indexed_dictionary_blocks)
        .saturating_add(position_docs))
}

fn check_cancel(cx: &Cx, phase: &'static str) -> Result<(), QuillIndexError> {
    if cx.is_cancel_requested() {
        Err(QuillIndexError::Cancelled { phase })
    } else {
        Ok(())
    }
}

fn map_lock_error_for_index(error: LockError, phase: &'static str) -> QuillIndexError {
    match error {
        LockError::Cancelled | LockError::TimedOut(_) => QuillIndexError::Cancelled { phase },
        LockError::Poisoned | LockError::PolledAfterCompletion => {
            invalid_state(format!("Quill writer lock failed during {phase}: {error}"))
        }
    }
}

fn map_try_lock_error(error: TryLockError) -> QuillIndexError {
    invalid_state(format!("Quill writer is unavailable: {error}"))
}

fn invalid_state(detail: impl Into<String>) -> QuillIndexError {
    QuillIndexError::InvalidState {
        detail: detail.into(),
    }
}

fn wall_clock_unix_s() -> Result<i64, QuillIndexError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid_state("system clock precedes Unix epoch"))?
        .as_secs();
    i64::try_from(seconds).map_err(|_| invalid_state("Unix timestamp does not fit i64"))
}

fn next_lease_boundary(watermark: u64) -> Result<u64, QuillIndexError> {
    if watermark > MAX_GLOBAL_DOCID_EXCLUSIVE {
        return Err(invalid_state(
            "manifest document-id watermark exceeds the Q1 address space",
        ));
    }
    let lease_size = u64::from(DOC_ORDS_PER_LEASE);
    let remainder = watermark % lease_size;
    if remainder == 0 {
        return Ok(watermark);
    }
    watermark
        .checked_add(lease_size - remainder)
        .filter(|boundary| *boundary <= MAX_GLOBAL_DOCID_EXCLUSIVE)
        .ok_or_else(|| invalid_state("manifest document-id watermark cannot reach another lease"))
}

fn canonical_metadata(
    metadata: &std::collections::HashMap<String, String>,
) -> Result<Vec<u8>, serde_json::Error> {
    let ordered = metadata.iter().collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&ordered)
}

fn canonical_document_content_hash(
    document: &IndexableDocument,
    metadata: &[u8],
) -> Result<u64, QuillIndexError> {
    let mut hasher = Xxh3::new();
    hasher.update(CONTENT_HASH_DOMAIN);
    for field in [
        document.id.as_bytes(),
        document.content.as_bytes(),
        document.title.as_deref().unwrap_or("").as_bytes(),
        metadata,
    ] {
        let len = u64::try_from(field.len())
            .map_err(|_| invalid_state("canonical document field length does not fit u64"))?;
        hasher.update(&len.to_le_bytes());
        hasher.update(field);
    }
    Ok(hasher.digest())
}

/// Compute the exact IDMAP content witness Quill will persist for a document.
///
/// # Errors
///
/// Returns a JSON error if canonical metadata serialization fails.
pub fn indexable_document_content_hash(
    document: &IndexableDocument,
) -> Result<u64, QuillIndexError> {
    let metadata = canonical_metadata(&document.metadata)?;
    canonical_document_content_hash(document, &metadata)
}

#[cfg(feature = "conformance-internals")]
fn conformance_usize_to_u64(value: usize, label: &'static str) -> Result<u64, QuillIndexError> {
    u64::try_from(value).map_err(|_| invalid_state(format!("{label} does not fit u64")))
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_be_bytes());
    hasher.update(bytes);
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_optional_u64(hasher: &mut Sha256, value: Option<u64>) {
    match value {
        None => hasher.update([0]),
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
    }
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_optional_lease(hasher: &mut Sha256, lease: Option<(u64, u32)>) {
    match lease {
        None => hasher.update([0]),
        Some((base, used)) => {
            hasher.update([1]);
            hasher.update(base.to_be_bytes());
            hasher.update(used.to_be_bytes());
        }
    }
}

#[cfg(feature = "conformance-internals")]
fn conformance_manifest_bytes(
    manifest: &Manifest,
    label: &'static str,
) -> Result<Vec<u8>, QuillIndexError> {
    manifest
        .to_bytes()
        .map_err(|error| invalid_state(format!("could not encode {label} receipt: {error}")))
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_manifest_segment(hasher: &mut Sha256, segment: &ManifestSegment) {
    hasher.update(segment.segment_id.to_be_bytes());
    hasher.update(segment.seal_seq.to_be_bytes());
    hasher.update(segment.file_len.to_be_bytes());
    hasher.update(segment.file_xxh3.to_be_bytes());
    hasher.update(segment.docid_lo.to_be_bytes());
    hasher.update(segment.docid_hi.to_be_bytes());
    hasher.update(segment.doc_count.to_be_bytes());
    conformance_hash_bytes(hasher, segment.tombstones.as_bytes());
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_field_stats(
    hasher: &mut Sha256,
    field_stats: &BTreeMap<u16, (u64, u32)>,
) -> Result<(), QuillIndexError> {
    hasher.update(
        conformance_usize_to_u64(field_stats.len(), "pending field-stat count")?.to_be_bytes(),
    );
    for (&field_ord, &(total_tokens, document_count)) in field_stats {
        hasher.update(field_ord.to_be_bytes());
        hasher.update(total_tokens.to_be_bytes());
        hasher.update(document_count.to_be_bytes());
    }
    Ok(())
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_optional_manifest(
    hasher: &mut Sha256,
    manifest: Option<&Manifest>,
    label: &'static str,
) -> Result<(), QuillIndexError> {
    match manifest {
        None => hasher.update([0]),
        Some(manifest) => {
            hasher.update([1]);
            conformance_hash_bytes(hasher, &conformance_manifest_bytes(manifest, label)?);
        }
    }
    Ok(())
}

#[cfg(feature = "conformance-internals")]
fn conformance_hash_optional_delta_seal(
    hasher: &mut Sha256,
    pending: Option<&PendingDeltaSeal>,
) -> Result<(), QuillIndexError> {
    let Some(pending) = pending else {
        hasher.update([0]);
        return Ok(());
    };
    hasher.update([1]);
    match &pending.encoded {
        None => hasher.update([0]),
        Some(encoded) => {
            hasher.update([1]);
            conformance_hash_bytes(hasher, encoded.as_bytes());
        }
    }
    hasher.update([u8::from(pending.segment_installed)]);
    conformance_hash_bytes(
        hasher,
        &conformance_manifest_bytes(&pending.manifest, "pending Delta MANIFEST")?,
    );
    hasher.update(pending.next_seal_seq.to_be_bytes());
    hasher.update(pending.successor_watermark.to_be_bytes());

    let prepared = &pending.prepared;
    hasher.update(prepared.snapshot_epoch.to_be_bytes());
    hasher.update(prepared.expected_keeper_generation.to_be_bytes());
    hasher.update(prepared.expected_schema_id.to_be_bytes());
    hasher.update(prepared.expected_docid_high_watermark.to_be_bytes());
    conformance_hash_bytes(hasher, &prepared.schema.canonical_encoding()?);
    hasher.update(
        conformance_usize_to_u64(prepared.field_stats.len(), "prepared field-stat count")?
            .to_be_bytes(),
    );
    for field_stats in &prepared.field_stats {
        hasher.update(field_stats.field_ord.to_be_bytes());
        hasher.update(field_stats.total_tokens.to_be_bytes());
        hasher.update(field_stats.doc_count.to_be_bytes());
    }
    hasher.update(prepared.bm25_doc_count.to_be_bytes());
    hasher.update(prepared.live_doc_count.to_be_bytes());
    hasher.update(prepared.delta_live_doc_count.to_be_bytes());
    hasher.update(
        conformance_usize_to_u64(prepared.deltas.len(), "prepared Delta count")?.to_be_bytes(),
    );
    for delta in &prepared.deltas {
        hasher.update(delta.keeper_generation().to_be_bytes());
        let (lineage_id, generation, lease_base, lease_end) = delta.publication_lineage();
        hasher.update(lineage_id.to_be_bytes());
        hasher.update(generation.to_be_bytes());
        hasher.update(lease_base.to_be_bytes());
        hasher.update(lease_end.to_be_bytes());
        hasher.update(
            conformance_usize_to_u64(delta.live_document_count(), "prepared Delta live rows")?
                .to_be_bytes(),
        );
        conformance_hash_bytes(hasher, &delta.conformance_content_sha256()?);
    }
    Ok(())
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

fn merge_field_stats(
    committed: &[ManifestFieldStats],
    pending: &BTreeMap<u16, (u64, u32)>,
) -> Result<Vec<ManifestFieldStats>, QuillIndexError> {
    let mut merged = committed
        .iter()
        .map(|row| (row.field_ord, (row.total_tokens, row.doc_count)))
        .collect::<BTreeMap<_, _>>();
    for (&field_ord, &(tokens, documents)) in pending {
        let row = merged.entry(field_ord).or_insert((0, 0));
        row.0 = row
            .0
            .checked_add(tokens)
            .ok_or_else(|| invalid_state("manifest field token count overflow"))?;
        row.1 = row
            .1
            .checked_add(documents)
            .ok_or_else(|| invalid_state("manifest field document count overflow"))?;
    }
    Ok(merged
        .into_iter()
        .map(
            |(field_ord, (total_tokens, doc_count))| ManifestFieldStats {
                field_ord,
                total_tokens,
                doc_count,
            },
        )
        .collect())
}

#[derive(Debug)]
struct OwnedFieldNorms {
    field_ord: u16,
    docid_lo: u64,
    values: Vec<u8>,
}

impl FieldNormReader for OwnedFieldNorms {
    fn field_ord(&self) -> u16 {
        self.field_ord
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        let offset = u64::from(global_docid).checked_sub(self.docid_lo)?;
        self.values.get(usize::try_from(offset).ok()?).copied()
    }
}

#[derive(Debug)]
struct OwnedPostingCursor {
    postings: Vec<Posting>,
    positions: Option<Vec<Vec<u32>>>,
    cursor: usize,
    segment_num_docs: u32,
}

impl OwnedPostingCursor {
    fn empty(segment_num_docs: u32, positioned: bool) -> Self {
        Self {
            postings: Vec::new(),
            positions: positioned.then(Vec::new),
            cursor: 0,
            segment_num_docs,
        }
    }

    fn current(&self) -> Option<Posting> {
        self.postings.get(self.cursor).copied()
    }
}

impl PostingCursor for OwnedPostingCursor {
    fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current()?;
        self.positions.as_ref()?;
        u32::try_from(self.cursor)
            .ok()
            .map(|ordinal| PositionsHandle::new(self, ordinal))
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.cursor < self.postings.len() {
            self.cursor += 1;
        }
        Ok(self.doc())
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        let relative = self.postings[self.cursor.min(self.postings.len())..]
            .partition_point(|posting| posting.doc_id < target);
        self.cursor = self
            .cursor
            .saturating_add(relative)
            .min(self.postings.len());
        Ok(self.doc())
    }
}

impl PositionsReader for OwnedPostingCursor {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let ordinal = usize::try_from(posting_ordinal)
            .map_err(|_| ArgusError::CursorInvariant("posting ordinal does not fit usize"))?;
        if ordinal != self.cursor {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = self
            .positions
            .as_ref()
            .and_then(|rows| rows.get(ordinal))
            .ok_or(ArgusError::CursorInvariant(
                "positioned cursor has no current position run",
            ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "owned position run",
                count: positions.len(),
            })?;
        output.extend_from_slice(positions);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum QueryLeaf<'a> {
    Sealed(&'a RecoveredSegment),
    Delta(&'a DeltaSnapshot),
}

impl QueryLeaf<'_> {
    fn docid_range(self) -> (u64, u64) {
        match self {
            Self::Sealed(segment) => (segment.manifest().docid_lo, segment.manifest().docid_hi),
            Self::Delta(delta) => delta.occupied_docid_range().unwrap_or_else(|| {
                let lease_base = delta.lease_base();
                (lease_base, lease_base)
            }),
        }
    }

    fn live_document_count(self) -> Result<u32, QuillIndexError> {
        match self {
            Self::Sealed(segment) => Ok(segment.doc_count()),
            Self::Delta(delta) => u32::try_from(delta.live_document_count())
                .map_err(|_| invalid_state("Delta live document count does not fit u32")),
        }
    }

    fn is_live_document(self, global_docid: u32) -> bool {
        match self {
            Self::Sealed(segment) => crate::argus::LiveDocs::is_live(segment, global_docid),
            Self::Delta(delta) => delta.is_live_document(global_docid),
        }
    }
}

fn validate_query_lowering(
    query: &Query,
    inherited_boost: f32,
    schema: SchemaDescriptor,
) -> Result<(), QuillIndexError> {
    match query {
        Query::Empty | Query::All => Ok(()),
        Query::Term { fields, text } => {
            for field in fields {
                validate_query_term(schema, field.field_id, text.as_bytes())?;
                validate_cumulative_boost(inherited_boost, field.boost)?;
            }
            Ok(())
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            if terms.is_empty() {
                return Err(ArgusError::InvalidPhrase {
                    reason: "an exact phrase requires positioned terms",
                }
                .into());
            }
            if terms.len() > 1 {
                // Capability errors take precedence over operator refinements:
                // slop and phrase-prefix queries also consume positions, so a
                // positionless schema must receive the same typed failure.
                validate_index_capabilities(query, schema)?;
                if terms
                    .windows(2)
                    .any(|pair| pair[0].position > pair[1].position)
                {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "phrase positions must be non-decreasing",
                    }
                    .into());
                }
                if terms.first().map(|term| term.position) == terms.last().map(|term| term.position)
                {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "an exact phrase must span at least two positions",
                    }
                    .into());
                }
            }
            if *slop != 0 || *prefix {
                return Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("phrase slop={slop} prefix={prefix}"),
                });
            }
            for term in terms {
                for field in fields {
                    validate_query_term(schema, field.field_id, term.text.as_bytes())?;
                }
            }
            for field in fields {
                validate_cumulative_boost(inherited_boost, field.boost)?;
            }
            Ok(())
        }
        Query::Boolean { clauses, .. } => {
            for clause in clauses {
                validate_query_lowering(&clause.query, inherited_boost, schema)?;
            }
            Ok(())
        }
        Query::Range {
            field_id,
            lower,
            upper,
        } => match query_field_kind(schema, *field_id)? {
            FieldKind::Keyword | FieldKind::Text { .. } => {
                validate_string_query_field(schema, *field_id, "range")?;
                let (lower, upper) = string_query_bounds(*field_id, lower, upper)?;
                validate_bound_term(schema, *field_id, &lower)?;
                validate_bound_term(schema, *field_id, &upper).map_err(QuillIndexError::from)
            }
            FieldKind::I64 { indexed: true, .. }
            | FieldKind::I64 {
                indexed: false,
                fast: true,
            }
            | FieldKind::U64 { indexed: true, .. }
            | FieldKind::U64 {
                indexed: false,
                fast: true,
            } => numeric_query_bounds(schema, *field_id, lower, upper).map(|_| ()),
            FieldKind::I64 {
                indexed: false,
                fast: false,
            }
            | FieldKind::U64 {
                indexed: false,
                fast: false,
            } => Err(QuillIndexError::UnsupportedQuery {
                detail: format!("range names non-indexed numeric field {field_id}"),
            }),
            FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
                detail: format!("range names stored-only field {field_id}"),
            }),
        },
        Query::Set { field_id, values } => match query_field_kind(schema, *field_id)? {
            FieldKind::Keyword | FieldKind::Text { .. } => {
                for value in values {
                    let QueryValue::Str(value) = value else {
                        return Err(QuillIndexError::UnsupportedQuery {
                            detail: format!(
                                "set value type does not match string field {field_id}"
                            ),
                        });
                    };
                    validate_query_term(schema, *field_id, value.as_bytes())?;
                }
                Ok(())
            }
            FieldKind::I64 { indexed: true, .. } | FieldKind::U64 { indexed: true, .. } => {
                numeric_query_values(schema, *field_id, values).map(|_| ())
            }
            FieldKind::I64 { indexed: false, .. } | FieldKind::U64 { indexed: false, .. } => {
                Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("set names non-indexed numeric field {field_id}"),
                })
            }
            FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
                detail: format!("set names stored-only field {field_id}"),
            }),
        },
        Query::Glob { field_ids, .. } => {
            for &field_id in field_ids {
                validate_string_query_field(schema, field_id, "glob")?;
            }
            Ok(())
        }
        Query::Boost { query, factor } => {
            let boost = validate_cumulative_boost(inherited_boost, *factor)?;
            validate_query_lowering(query, boost, schema)
        }
    }
}

fn validate_cumulative_boost(inherited: f32, factor: f32) -> Result<f32, QuillIndexError> {
    let boost = inherited * factor;
    if boost.is_finite() {
        Ok(boost)
    } else {
        Err(QuillIndexError::UnsupportedQuery {
            detail: format!("non-finite cumulative boost bits 0x{:08x}", boost.to_bits()),
        })
    }
}

fn lower_query<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
    rank_pruning: bool,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    lower_query_with_mode(
        cx,
        checkpoint,
        query,
        inherited_boost,
        leaf,
        snapshot,
        schema,
        glob_expansion_limit,
        QueryLoweringMode::Scored,
        rank_pruning,
        topdocs_root,
    )
}

fn lower_query_unscored<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    lower_query_with_mode(
        cx,
        checkpoint,
        query,
        inherited_boost,
        leaf,
        snapshot,
        schema,
        glob_expansion_limit,
        QueryLoweringMode::Unscored,
        false,
        false,
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct QueryTraceShape {
    root_kind: &'static str,
    topology_hash: u64,
    nodes: u64,
    depth: u64,
    terms: u64,
    phrases: u64,
    booleans: u64,
    predicates: u64,
    boosts: u64,
}

impl QueryTraceShape {
    fn from_query(query: &Query) -> Self {
        let mut shape = Self {
            root_kind: query_trace_kind(query),
            topology_hash: query_topology_hash(query),
            ..Self::default()
        };
        shape.visit(query, 1);
        shape
    }

    fn visit(&mut self, query: &Query, depth: u64) {
        self.nodes = self.nodes.saturating_add(1);
        self.depth = self.depth.max(depth);
        match query {
            Query::Empty | Query::All => {}
            Query::Term { .. } => self.terms = self.terms.saturating_add(1),
            Query::Phrase { .. } => self.phrases = self.phrases.saturating_add(1),
            Query::Boolean { clauses, .. } => {
                self.booleans = self.booleans.saturating_add(1);
                for clause in clauses {
                    self.visit(&clause.query, depth.saturating_add(1));
                }
            }
            Query::Range { .. } | Query::Set { .. } | Query::Glob { .. } => {
                self.predicates = self.predicates.saturating_add(1);
            }
            Query::Boost { query, .. } => {
                self.boosts = self.boosts.saturating_add(1);
                self.visit(query, depth.saturating_add(1));
            }
        }
    }
}

const fn query_trace_kind(query: &Query) -> &'static str {
    match query {
        Query::Empty => "empty",
        Query::All => "all",
        Query::Term { .. } => "term",
        Query::Phrase { .. } => "phrase",
        Query::Boolean { .. } => "boolean",
        Query::Range { .. } => "range",
        Query::Set { .. } => "set",
        Query::Glob { .. } => "glob",
        Query::Boost { .. } => "boost",
    }
}

fn record_query_trace_shape(span: &tracing::Span, query: &Query) {
    let shape = QueryTraceShape::from_query(query);
    span.record("query_root", shape.root_kind);
    span.record("query_shape_hash", shape.topology_hash);
    span.record("query_nodes", shape.nodes);
    span.record("query_depth", shape.depth);
    span.record("term_nodes", shape.terms);
    span.record("phrase_nodes", shape.phrases);
    span.record("boolean_nodes", shape.booleans);
    span.record("predicate_nodes", shape.predicates);
    span.record("boost_nodes", shape.boosts);
}

fn query_topology_hash(query: &Query) -> u64 {
    let mut hasher = Xxh3::new();
    hash_query_topology(&mut hasher, query);
    hasher.digest()
}

fn hash_query_topology(hasher: &mut Xxh3, query: &Query) {
    match query {
        Query::Empty => hasher.update(&[0]),
        Query::All => hasher.update(&[1]),
        Query::Term { fields, .. } => {
            hasher.update(&[2]);
            hash_topology_len(hasher, fields.len());
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            hasher.update(&[3]);
            hash_topology_len(hasher, fields.len());
            hash_topology_len(hasher, terms.len());
            hasher.update(&slop.to_le_bytes());
            hasher.update(&[u8::from(*prefix)]);
        }
        Query::Boolean { clauses, operator } => {
            hasher.update(&[4]);
            hash_topology_len(hasher, clauses.len());
            hasher.update(&[match operator {
                None => 0,
                Some(BooleanOperator::And) => 1,
                Some(BooleanOperator::Or) => 2,
            }]);
            for clause in clauses {
                hasher.update(&[match clause.occur {
                    Occur::Must => 0,
                    Occur::Should => 1,
                    Occur::MustNot => 2,
                }]);
                hash_query_topology(hasher, &clause.query);
            }
        }
        Query::Range { lower, upper, .. } => {
            hasher.update(&[5, bound_topology_kind(lower), bound_topology_kind(upper)]);
        }
        Query::Set { values, .. } => {
            hasher.update(&[6]);
            hash_topology_len(hasher, values.len());
        }
        Query::Glob { field_ids, .. } => {
            hasher.update(&[7]);
            hash_topology_len(hasher, field_ids.len());
        }
        Query::Boost { query, .. } => {
            hasher.update(&[8]);
            hash_query_topology(hasher, query);
        }
    }
}

fn hash_topology_len(hasher: &mut Xxh3, len: usize) {
    hasher.update(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

const fn bound_topology_kind(bound: &Bound<QueryValue>) -> u8 {
    match bound {
        Bound::Unbounded => 0,
        Bound::Included(_) => 1,
        Bound::Excluded(_) => 2,
    }
}

/// Gate for deferred grouped `MaxScore` over nested pure-term unions
/// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
///
/// **Currently `false`: the grouped runtime is rank-safe and demonstrably
/// prunes, but it did not earn activation in the paired release A/B.**
///
/// `argus::tests::grouped_max_score_matches_exhaustive_and_prunes` and the
/// drained-essential-group regression both pass. The immutable 100k,
/// same-binary, 41-round campaign recorded pruned/exhaustive medians of 1.0138
/// for two groups, 1.0023 for three, and 1.0151 for four, all inside their A/A
/// median-CI floors; eight groups was a decision-valid 1.0261 regression.
///
/// Flipping this to `true` would open BLOCKMAX for every nested union's terms
/// — i.e. pay the metadata cost on the most common query shape — without a
/// measured query-time win. Full provenance and the retry predicate live in
/// `docs/NEGATIVE_EVIDENCE.md`.
///
/// With this `false`, nested unions fail the rank-pruning gate exactly as they
/// did before this change: the shape classification below is inert, no BLOCKMAX
/// is opened for their terms, and every such query keeps the pre-existing
/// POSTINGS-only exhaustive path. Direct-term `MaxScore`/BMW is unaffected
/// either way.
///
/// **To activate:** first produce a materially different traversal and a
/// same-binary win for at least the two- and four-group 100k cells, then run
/// the full 1M and parity matrix. The gate flip itself remains the only
/// production edit required.
const GROUPED_MAX_SCORE_ENABLED: bool = false;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrunableScorerShape {
    Empty,
    Term,
    Union {
        children: usize,
        kind: UnionChildKind,
    },
}

/// What a prunable union's children lower to, which decides which runtime
/// pruning strategy (if any) can consume the union.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnionChildKind {
    /// Every child lowers to a single direct term cursor, so the union feeds
    /// term-granular `MaxScore` or block-max WAND.
    DirectTerms,
    /// Every child is itself a union of direct terms — a pure-term group, i.e.
    /// the multi-field terms a default query lowers to. Such a union feeds
    /// deferred grouped `MaxScore`, which decides essential vs. non-essential
    /// at *group* granularity (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    TermGroups,
    /// Mixed children or deeper nesting: no runtime strategy consumes it, so
    /// the union stays POSTINGS-only.
    Mixed,
}

/// Mirror the score-tree topology that lowering will build without opening any
/// segment sections. Only pure non-negative term unions can consume `MaxScore`
/// or BMW metadata; every other shape keeps the pre-E4.4 POSTINGS-only path.
fn prunable_scorer_shape(query: &Query, inherited_boost: f32) -> Option<PrunableScorerShape> {
    match query {
        Query::Empty => Some(PrunableScorerShape::Empty),
        Query::Term { fields, .. } => {
            if fields.iter().any(|field| {
                let boost = inherited_boost * field.boost;
                !boost.is_finite() || boost.is_sign_negative()
            }) {
                return None;
            }
            Some(match fields.len() {
                0 => PrunableScorerShape::Empty,
                1 => PrunableScorerShape::Term,
                children => PrunableScorerShape::Union {
                    children,
                    kind: UnionChildKind::DirectTerms,
                },
            })
        }
        Query::Boolean { clauses, .. }
            if clauses
                .iter()
                .all(|clause| clause.occur == crate::query::Occur::Should) =>
        {
            let mut children = 0_usize;
            let mut singleton = PrunableScorerShape::Empty;
            let mut direct_terms = true;
            let mut term_groups = true;
            for clause in clauses {
                let shape = prunable_scorer_shape(&clause.query, inherited_boost)?;
                if shape == PrunableScorerShape::Empty {
                    continue;
                }
                children = children.checked_add(1)?;
                direct_terms &= shape == PrunableScorerShape::Term;
                term_groups &= matches!(
                    shape,
                    PrunableScorerShape::Union {
                        kind: UnionChildKind::DirectTerms,
                        ..
                    }
                );
                singleton = shape;
            }
            Some(match children {
                0 => PrunableScorerShape::Empty,
                1 => singleton,
                _ => PrunableScorerShape::Union {
                    children,
                    // A child is never both a term and a union, so at most one
                    // of these holds once `children >= 1`.
                    kind: if direct_terms {
                        UnionChildKind::DirectTerms
                    } else if term_groups {
                        UnionChildKind::TermGroups
                    } else {
                        UnionChildKind::Mixed
                    },
                },
            })
        }
        Query::Boost { query, factor } => {
            let boost = inherited_boost * *factor;
            (boost.is_finite() && !boost.is_sign_negative())
                .then(|| prunable_scorer_shape(query, boost))?
        }
        Query::All
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. }
        | Query::Boolean { .. } => None,
    }
}

/// Whether the root scorer can actually consume rank-pruning metadata, and so
/// whether opening BLOCKMAX for its terms is worth the setup cost.
///
/// Mirrors the runtime strategy selection in `argus::competitive_candidates`
/// exactly — admitting a shape no strategy consumes would pay for BLOCKMAX and
/// still score exhaustively.
fn query_has_prunable_root_union(query: &Query, inherited_boost: f32) -> bool {
    match prunable_scorer_shape(query, inherited_boost) {
        Some(PrunableScorerShape::Union { children, kind }) => match kind {
            // Direct-term roots take `MaxScore` (2..=8 clauses) or block-max
            // WAND (9+), both term-granular.
            UnionChildKind::DirectTerms => {
                matches!(children, 2..=MAX_SCORE_MAX_CLAUSES | BMW_MIN_CLAUSES..)
            }
            // Grouped roots take deferred grouped `MaxScore` only. Grouped BMW
            // does not exist (a group has no physical block-max list of its
            // own), so wider grouped roots stay POSTINGS-only rather than
            // paying for metadata nothing reads.
            UnionChildKind::TermGroups => {
                GROUPED_MAX_SCORE_ENABLED && matches!(children, 2..=MAX_SCORE_MAX_CLAUSES)
            }
            UnionChildKind::Mixed => false,
        },
        Some(PrunableScorerShape::Empty | PrunableScorerShape::Term) | None => false,
    }
}

#[derive(Clone, Copy)]
enum QueryLoweringMode {
    Scored,
    Unscored,
}

fn lower_boolean(
    clauses: Vec<ScorerClause<'_>>,
    mode: QueryLoweringMode,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'_>, QuillIndexError> {
    match mode {
        QueryLoweringMode::Scored if topdocs_root => ReferenceScorer::boolean_topdocs(clauses),
        QueryLoweringMode::Scored => ReferenceScorer::boolean(clauses),
        QueryLoweringMode::Unscored => ReferenceScorer::boolean_unscored(clauses),
    }
    .map_err(QuillIndexError::from)
}

fn lower_query_with_mode<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    query: &Query,
    inherited_boost: f32,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    glob_expansion_limit: usize,
    mode: QueryLoweringMode,
    rank_pruning: bool,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match query {
        Query::Empty => Ok(ReferenceScorer::empty()),
        Query::All => {
            let (docid_lo, docid_hi) = leaf.docid_range();
            Ok(ReferenceScorer::all_with_boost(
                docid_lo,
                docid_hi,
                leaf.live_document_count()?,
                inherited_boost,
            )?)
        }
        Query::Term { fields, text } => {
            let mut clauses = Vec::new();
            clauses
                .try_reserve_exact(fields.len())
                .map_err(|_| invalid_state("could not allocate expanded term clauses"))?;
            for field in fields {
                clauses.push(ScorerClause::should(lower_leaf_term(
                    leaf,
                    snapshot,
                    schema,
                    field.field_id,
                    text.as_bytes(),
                    inherited_boost * field.boost,
                    rank_pruning,
                    checkpoint,
                )?));
            }
            lower_boolean(clauses, mode, topdocs_root)
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            if *slop != 0 || *prefix {
                return Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("phrase slop={slop} prefix={prefix}"),
                });
            }
            if terms.len() == 1 {
                let term = &terms[0];
                let mut clauses = Vec::new();
                for field in fields {
                    clauses.push(ScorerClause::should(lower_leaf_term(
                        leaf,
                        snapshot,
                        schema,
                        field.field_id,
                        term.text.as_bytes(),
                        inherited_boost * field.boost,
                        false,
                        checkpoint,
                    )?));
                }
                return lower_boolean(clauses, mode, topdocs_root);
            }
            let mut clauses = Vec::new();
            for field in fields {
                let stats = composite_snapshot_field(snapshot, field.field_id)?;
                let mut phrase_terms: Vec<PhraseTerm<'a>> = Vec::new();
                phrase_terms
                    .try_reserve_exact(terms.len())
                    .map_err(|_| invalid_state("could not allocate phrase terms"))?;
                for term in terms {
                    let snapshot_doc_freq = checkpointed_snapshot_doc_freq(
                        checkpoint,
                        snapshot,
                        field.field_id,
                        term.text.as_bytes(),
                    )?;
                    phrase_terms.push(match leaf {
                        QueryLeaf::Sealed(segment) => {
                            let cursor = open_owned_cursor(
                                segment,
                                schema,
                                field.field_id,
                                term.text.as_bytes(),
                                true,
                                Some(checkpoint),
                            )?;
                            PhraseTerm::new(
                                field.field_id,
                                term.position,
                                CheckpointPostingCursor::new(cursor, Arc::clone(checkpoint))?,
                                snapshot_doc_freq,
                            )
                        }
                        QueryLeaf::Delta(delta) => {
                            let cursor = DeltaPostingCursor::new(
                                delta,
                                field.field_id,
                                term.text.as_bytes(),
                            )?;
                            PhraseTerm::new(
                                field.field_id,
                                term.position,
                                CheckpointPostingCursor::new(cursor, Arc::clone(checkpoint))?,
                                snapshot_doc_freq,
                            )
                        }
                    });
                }
                let bm25 = Bm25FieldSnapshot::new(stats)?;
                let boost = inherited_boost * field.boost;
                let scorer = match leaf {
                    QueryLeaf::Sealed(segment) => PhraseScorer::new_with_checkpoint(
                        phrase_terms,
                        owned_fieldnorms(segment, schema, field.field_id)?,
                        bm25,
                        boost,
                        Some(Arc::clone(checkpoint)),
                    )?,
                    QueryLeaf::Delta(delta) => PhraseScorer::new_with_checkpoint(
                        phrase_terms,
                        DeltaFieldNorms::new(delta, field.field_id),
                        bm25,
                        boost,
                        Some(Arc::clone(checkpoint)),
                    )?,
                };
                clauses.push(ScorerClause::should(ReferenceScorer::phrase(scorer)));
            }
            lower_boolean(clauses, mode, topdocs_root)
        }
        Query::Boolean { clauses, .. } => {
            let mut lowered = Vec::new();
            lowered
                .try_reserve_exact(clauses.len())
                .map_err(|_| invalid_state("could not allocate Boolean clauses"))?;
            for clause in clauses {
                lowered.push(ScorerClause::new(
                    clause.occur,
                    lower_query_with_mode(
                        cx,
                        checkpoint,
                        &clause.query,
                        inherited_boost,
                        leaf,
                        snapshot,
                        schema,
                        glob_expansion_limit,
                        mode,
                        rank_pruning,
                        false,
                    )?,
                ));
            }
            lower_boolean(lowered, mode, topdocs_root)
        }
        Query::Boost { query, factor } => {
            let boost = inherited_boost * *factor;
            if !boost.is_finite() {
                return Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("non-finite cumulative boost bits 0x{:08x}", boost.to_bits()),
                });
            }
            lower_query_with_mode(
                cx,
                checkpoint,
                query,
                boost,
                leaf,
                snapshot,
                schema,
                glob_expansion_limit,
                mode,
                rank_pruning,
                topdocs_root,
            )
        }
        Query::Range {
            field_id,
            lower,
            upper,
        } => lower_leaf_range(
            cx,
            checkpoint,
            leaf,
            snapshot,
            schema,
            *field_id,
            lower,
            upper,
            inherited_boost,
            mode,
        ),
        Query::Glob { field_ids, pattern } => lower_leaf_glob(
            checkpoint,
            leaf,
            snapshot,
            schema,
            field_ids,
            pattern.as_bytes(),
            inherited_boost,
            glob_expansion_limit,
            mode,
        ),
        Query::Set { field_id, values } => lower_leaf_set(
            checkpoint,
            leaf,
            snapshot,
            schema,
            *field_id,
            values,
            inherited_boost,
            mode,
        ),
    }
}

fn lower_leaf_range<'a>(
    cx: &Cx,
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        }
        | FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => lower_leaf_numeric_range(cx, leaf, schema, field_ord, lower, upper, boost, mode),
        FieldKind::Keyword | FieldKind::Text { .. } => {
            let (lower, upper) = string_query_bounds(field_ord, lower, upper)?;
            let terms =
                snapshot_string_range_terms(checkpoint, snapshot, schema, field_ord, lower, upper)?;
            lower_leaf_string_predicate(
                checkpoint, leaf, snapshot, schema, field_ord, terms, boost, mode,
            )
        }
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        } => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("range names non-indexed numeric field {field_ord}"),
        }),
        FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("range names stored-only field {field_ord}"),
        }),
    }
}

fn lower_leaf_numeric_range<'a>(
    cx: &Cx,
    leaf: QueryLeaf<'a>,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let (lower, upper) = numeric_query_bounds(schema, field_ord, lower, upper)?;
    let score = match mode {
        QueryLoweringMode::Scored => boost,
        QueryLoweringMode::Unscored => 1.0,
    };
    match leaf {
        QueryLeaf::Sealed(segment) => {
            let manifest = segment.manifest();
            let section = NumericSection::parse(
                required_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("NUMERIC has no indexed-or-fast field {field_ord}"))
            })?;
            materialize_live_numeric_range(
                cx,
                leaf,
                field,
                lower,
                upper,
                segment.at_seal_doc_count(),
                score,
            )
        }
        QueryLeaf::Delta(delta) => {
            let encoded = encode_live_delta_numeric(delta, schema)?;
            let (docid_lo, docid_hi) = leaf.docid_range();
            let section = NumericSection::parse(encoded.as_bytes(), schema, docid_lo, docid_hi)
                .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!(
                    "Delta NUMERIC has no indexed-or-fast field {field_ord}"
                ))
            })?;
            materialize_live_numeric_range(
                cx,
                leaf,
                field,
                lower,
                upper,
                leaf.live_document_count()?,
                score,
            )
        }
    }
}

fn materialize_live_numeric_range<'a>(
    cx: &Cx,
    leaf: QueryLeaf<'a>,
    field: NumericField<'_>,
    lower: Bound<NumericValue>,
    upper: Bound<NumericValue>,
    segment_num_docs: u32,
    score: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    const CANCEL_CHECK_MASK: usize = 1_023;

    check_cancel(cx, "numeric range")?;
    let entries = field
        .range_entries(lower, upper)
        .map_err(ArgusError::from)?;
    let value_count = field.len();
    let covers_every_document = value_count
        == usize::try_from(segment_num_docs).unwrap_or(usize::MAX)
        && entries.len() == value_count;
    let mut docids = Vec::new();
    docids
        .try_reserve_exact(entries.len())
        .map_err(|_| invalid_state("could not allocate live numeric range docids"))?;
    for (index, entry) in entries.enumerate() {
        if index & CANCEL_CHECK_MASK == 0 {
            check_cancel(cx, "numeric range")?;
        }
        if leaf.is_live_document(entry.docid()) {
            docids.push(entry.docid());
        }
    }
    check_cancel(cx, "numeric range")?;
    docids.sort_unstable();
    docids.dedup();
    ReferenceScorer::materialized_numeric_range(
        field.field_ord(),
        docids,
        value_count,
        segment_num_docs,
        covers_every_document,
        score,
    )
    .map_err(QuillIndexError::from)
}

fn numeric_query_bounds(
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: &Bound<QueryValue>,
    upper: &Bound<QueryValue>,
) -> Result<(Bound<NumericValue>, Bound<NumericValue>), QuillIndexError> {
    let kind = query_field_kind(schema, field_ord)?;
    let expected = match kind {
        FieldKind::I64 { indexed: true, .. }
        | FieldKind::I64 {
            indexed: false,
            fast: true,
        } => NumericValueKind::I64,
        FieldKind::U64 { indexed: true, .. }
        | FieldKind::U64 {
            indexed: false,
            fast: true,
        } => NumericValueKind::U64,
        FieldKind::I64 {
            indexed: false,
            fast: false,
        }
        | FieldKind::U64 {
            indexed: false,
            fast: false,
        } => {
            return Err(QuillIndexError::UnsupportedQuery {
                detail: format!("numeric range names non-indexed field {field_ord}"),
            });
        }
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => {
            return Err(QuillIndexError::UnsupportedQuery {
                detail: format!("numeric range names non-numeric field {field_ord}"),
            });
        }
    };
    Ok((
        numeric_query_bound(field_ord, expected, lower)?,
        numeric_query_bound(field_ord, expected, upper)?,
    ))
}

fn query_field_kind(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<FieldKind, QuillIndexError> {
    query_field_descriptor(schema, field_ord).map(|field| field.kind)
}

fn term_record_option(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<TermRecordOption, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword
        | FieldKind::Text {
            positions: false, ..
        } => Ok(TermRecordOption::Basic),
        FieldKind::Text {
            positions: true, ..
        } => Ok(TermRecordOption::WithFreqsAndPositions),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("term scorer names non-string field {field_ord}"),
            })
        }
    }
}

fn query_field_descriptor(
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<crate::schema::FieldDescriptor, QuillIndexError> {
    schema
        .fields
        .get(usize::from(field_ord))
        .filter(|field| field.id == field_ord)
        .copied()
        .ok_or_else(|| QuillIndexError::UnsupportedQuery {
            detail: format!("query names unknown field {field_ord}"),
        })
}

#[derive(Clone, Copy)]
enum NumericValueKind {
    I64,
    U64,
}

fn numeric_query_bound(
    field_ord: u16,
    expected: NumericValueKind,
    bound: &Bound<QueryValue>,
) -> Result<Bound<NumericValue>, QuillIndexError> {
    let convert = |value: &QueryValue| match (expected, value) {
        (NumericValueKind::I64, QueryValue::I64(value)) => Ok(NumericValue::I64(*value)),
        (NumericValueKind::U64, QueryValue::U64(value)) => Ok(NumericValue::U64(*value)),
        _ => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("numeric range bound type does not match field {field_ord}"),
        }),
    };
    match bound {
        Bound::Included(value) => convert(value).map(Bound::Included),
        Bound::Excluded(value) => convert(value).map(Bound::Excluded),
        Bound::Unbounded => Ok(Bound::Unbounded),
    }
}

type StringQueryBounds<'a> = (Bound<&'a [u8]>, Bound<&'a [u8]>);

fn string_query_bounds<'a>(
    field_ord: u16,
    lower: &'a Bound<QueryValue>,
    upper: &'a Bound<QueryValue>,
) -> Result<StringQueryBounds<'a>, QuillIndexError> {
    Ok((
        string_query_bound(field_ord, lower)?,
        string_query_bound(field_ord, upper)?,
    ))
}

fn string_query_bound(
    field_ord: u16,
    bound: &Bound<QueryValue>,
) -> Result<Bound<&[u8]>, QuillIndexError> {
    match bound {
        Bound::Included(QueryValue::Str(value)) => Ok(Bound::Included(value.as_bytes())),
        Bound::Excluded(QueryValue::Str(value)) => Ok(Bound::Excluded(value.as_bytes())),
        Bound::Unbounded => Ok(Bound::Unbounded),
        Bound::Included(QueryValue::I64(_) | QueryValue::U64(_))
        | Bound::Excluded(QueryValue::I64(_) | QueryValue::U64(_)) => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("string range bound type does not match field {field_ord}"),
            })
        }
    }
}

fn snapshot_string_range_terms(
    checkpoint: &QueryCheckpointHandle<'_>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    lower: Bound<&[u8]>,
    upper: Bound<&[u8]>,
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "range")?;
    if string_range_is_reversed(&lower, &upper) {
        return Ok(Vec::new());
    }
    let mut terms = BTreeSet::<Vec<u8>>::new();
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, schema)?;
        let limit = usize::try_from(dictionary.term_count())
            .map_err(|_| invalid_state("dictionary term count does not fit usize"))?;
        let mut cursor = dictionary.range_cursor(field_ord, lower, upper)?;
        let mut admitted_block = None;
        while let Some(current) = cursor.current() {
            let block = cursor.current_block_index();
            if block != admitted_block {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                admitted_block = block;
            }
            if terms.len() >= limit {
                return Err(TermDictionaryError::MaterializationLimitExceeded { limit }.into());
            }
            terms.insert(current.term.to_vec());
            cursor.next()?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        let mut admitted_block = None;
        for (term_index, term) in delta.segment().sorted_terms().iter().enumerate() {
            let block = term_index / 16;
            if Some(block) != admitted_block {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                admitted_block = Some(block);
            }
            if term.field_ord() == field_ord
                && term.live_doc_freq() != 0
                && term_in_string_range(term.term(), &lower, &upper)
            {
                terms.insert(term.term().to_vec());
            }
        }
    }
    Ok(terms.into_iter().collect())
}

fn string_range_is_reversed(lower: &Bound<&[u8]>, upper: &Bound<&[u8]>) -> bool {
    match (lower, upper) {
        (
            Bound::Included(lower) | Bound::Excluded(lower),
            Bound::Included(upper) | Bound::Excluded(upper),
        ) => lower > upper,
        _ => false,
    }
}

fn term_in_string_range(term: &[u8], lower: &Bound<&[u8]>, upper: &Bound<&[u8]>) -> bool {
    let above_lower = match lower {
        Bound::Included(bound) => term >= *bound,
        Bound::Excluded(bound) => term > *bound,
        Bound::Unbounded => true,
    };
    let below_upper = match upper {
        Bound::Included(bound) => term <= *bound,
        Bound::Excluded(bound) => term < *bound,
        Bound::Unbounded => true,
    };
    above_lower && below_upper
}

fn validate_string_query_field(
    schema: SchemaDescriptor,
    field_ord: u16,
    predicate: &'static str,
) -> Result<(), QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword | FieldKind::Text { .. } => Ok(()),
        FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("{predicate} names non-string field {field_ord}"),
            })
        }
    }
}

fn encode_live_delta_numeric(
    delta: &DeltaSnapshot,
    schema: SchemaDescriptor,
) -> Result<EncodedNumericSection, QuillIndexError> {
    let mut owned_fields = Vec::<(u16, Vec<NumericEntry>)>::new();
    for field in schema
        .fields
        .iter()
        .filter(|field| field.kind.has_numeric_column())
    {
        let mut entries = Vec::new();
        for (global_docid, _) in delta.live_documents() {
            let Some(value) = delta.numeric_value(field.id, global_docid) else {
                continue;
            };
            entries.push(match value {
                NumericValue::I64(value) => NumericEntry::i64(value, global_docid),
                NumericValue::U64(value) => NumericEntry::u64(value, global_docid),
            });
        }
        owned_fields.push((field.id, entries));
    }
    let inputs = owned_fields
        .iter()
        .map(|(field_ord, entries)| NumericFieldInput::new(*field_ord, entries))
        .collect::<Vec<_>>();
    let (docid_lo, docid_hi) = QueryLeaf::Delta(delta).docid_range();
    EncodedNumericSection::encode(schema, docid_lo, docid_hi, &inputs)
        .map_err(ArgusError::from)
        .map_err(QuillIndexError::from)
}

fn lower_leaf_set<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[QueryValue],
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::Keyword | FieldKind::Text { .. } => {
            let mut terms = BTreeSet::<Vec<u8>>::new();
            for value in values {
                let QueryValue::Str(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match string field {field_ord}"),
                    });
                };
                terms.insert(value.as_bytes().to_vec());
            }
            lower_leaf_string_predicate(
                checkpoint,
                leaf,
                snapshot,
                schema,
                field_ord,
                terms.into_iter().collect(),
                boost,
                mode,
            )
        }
        FieldKind::I64 { indexed: true, .. } | FieldKind::U64 { indexed: true, .. } => {
            let values = numeric_query_values(schema, field_ord, values)?;
            lower_leaf_numeric_set(leaf, schema, field_ord, &values, boost, mode)
        }
        FieldKind::I64 { indexed: false, .. } | FieldKind::U64 { indexed: false, .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: format!("set names non-indexed numeric field {field_ord}"),
            })
        }
        FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("set names stored-only field {field_ord}"),
        }),
    }
}

fn numeric_query_values(
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[QueryValue],
) -> Result<Vec<NumericValue>, QuillIndexError> {
    match query_field_kind(schema, field_ord)? {
        FieldKind::I64 { indexed: true, .. } => {
            let mut unique = BTreeSet::new();
            for value in values {
                let QueryValue::I64(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match i64 field {field_ord}"),
                    });
                };
                unique.insert(*value);
            }
            Ok(unique.into_iter().map(NumericValue::I64).collect())
        }
        FieldKind::U64 { indexed: true, .. } => {
            let mut unique = BTreeSet::new();
            for value in values {
                let QueryValue::U64(value) = value else {
                    return Err(QuillIndexError::UnsupportedQuery {
                        detail: format!("set value type does not match u64 field {field_ord}"),
                    });
                };
                unique.insert(*value);
            }
            Ok(unique.into_iter().map(NumericValue::U64).collect())
        }
        FieldKind::I64 { indexed: false, .. }
        | FieldKind::U64 { indexed: false, .. }
        | FieldKind::Keyword
        | FieldKind::Text { .. }
        | FieldKind::StoredOnly => Err(QuillIndexError::UnsupportedQuery {
            detail: format!("numeric set names incompatible field {field_ord}"),
        }),
    }
}

fn lower_leaf_numeric_set<'a>(
    leaf: QueryLeaf<'a>,
    schema: SchemaDescriptor,
    field_ord: u16,
    values: &[NumericValue],
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    match leaf {
        QueryLeaf::Sealed(segment) => {
            let manifest = segment.manifest();
            let section = NumericSection::parse(
                required_section(segment, SectionKind::NUMERIC)?,
                schema,
                manifest.docid_lo,
                manifest.docid_hi,
            )
            .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("NUMERIC has no indexed field {field_ord}"))
            })?;
            lower_numeric_field_set(field, values, segment.at_seal_doc_count(), boost, mode)
        }
        QueryLeaf::Delta(delta) => {
            let encoded = encode_live_delta_numeric(delta, schema)?;
            let (docid_lo, docid_hi) = leaf.docid_range();
            let section = NumericSection::parse(encoded.as_bytes(), schema, docid_lo, docid_hi)
                .map_err(ArgusError::from)?;
            let field = section.field(field_ord).ok_or_else(|| {
                invalid_state(format!("Delta NUMERIC has no indexed field {field_ord}"))
            })?;
            lower_numeric_field_set(field, values, leaf.live_document_count()?, boost, mode)
        }
    }
}

fn lower_numeric_field_set<'a>(
    field: NumericField<'_>,
    values: &[NumericValue],
    document_count: u32,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(values.len())
        .map_err(|_| invalid_state("could not allocate numeric set clauses"))?;
    for &value in values {
        clauses.push(ScorerClause::should(ReferenceScorer::numeric_range(
            field,
            Bound::Included(value),
            Bound::Included(value),
            document_count,
        )?));
    }
    let matching = lower_boolean(clauses, QueryLoweringMode::Unscored, false)?;
    match mode {
        QueryLoweringMode::Scored => {
            ReferenceScorer::constant_score(matching, boost).map_err(QuillIndexError::from)
        }
        QueryLoweringMode::Unscored => Ok(matching),
    }
}

fn lower_leaf_string_predicate<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    terms: Vec<Vec<u8>>,
    boost: f32,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut clauses = Vec::new();
    clauses
        .try_reserve_exact(terms.len())
        .map_err(|_| invalid_state("could not allocate string predicate clauses"))?;
    for term in terms {
        clauses.push(ScorerClause::should(lower_leaf_term(
            leaf, snapshot, schema, field_ord, &term, 1.0, false, checkpoint,
        )?));
    }
    let matching = lower_boolean(clauses, QueryLoweringMode::Unscored, false)?;
    match mode {
        QueryLoweringMode::Scored => {
            ReferenceScorer::constant_score(matching, boost).map_err(QuillIndexError::from)
        }
        QueryLoweringMode::Unscored => Ok(matching),
    }
}

fn lower_leaf_glob<'a>(
    checkpoint: &QueryCheckpointHandle<'a>,
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ids: &[u16],
    pattern: &[u8],
    boost: f32,
    expansion_limit: usize,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(field_ids.len())
        .map_err(|_| invalid_state("could not allocate glob field clauses"))?;
    for &field_ord in field_ids {
        let terms = snapshot_glob_terms(
            Some(checkpoint),
            snapshot,
            schema,
            field_ord,
            pattern,
            expansion_limit,
        )?;
        let field_scorer = lower_leaf_string_predicate(
            checkpoint, leaf, snapshot, schema, field_ord, terms, boost, mode,
        )?;
        fields.push(ScorerClause::should(field_scorer));
    }
    lower_boolean(fields, mode, false)
}

fn snapshot_glob_terms(
    checkpoint: Option<&QueryCheckpointHandle<'_>>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    pattern: &[u8],
    expansion_limit: usize,
) -> Result<Vec<Vec<u8>>, QuillIndexError> {
    validate_string_query_field(schema, field_ord, "glob")?;
    let mut terms = BTreeSet::<Vec<u8>>::new();
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, schema)?;
        if !pattern.contains(&b'*') {
            if let Some(checkpoint) = checkpoint {
                checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
            }
            if dictionary.lookup(field_ord, pattern)?.is_some() {
                insert_glob_term(&mut terms, field_ord, pattern.to_vec(), expansion_limit)?;
            }
            continue;
        }
        let trailing_prefix = trailing_star_prefix(pattern);
        let mut cursor = if let Some(prefix) = trailing_prefix {
            dictionary.prefix_cursor(field_ord, prefix)?
        } else {
            dictionary.field_cursor(field_ord)?
        };
        let mut admitted_block = None;
        while let Some(current) = cursor.current() {
            let block = cursor.current_block_index();
            if block != admitted_block {
                if let Some(checkpoint) = checkpoint {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                }
                admitted_block = block;
            }
            if trailing_prefix.is_some() || star_glob_matches(pattern, current.term) {
                insert_glob_term(
                    &mut terms,
                    field_ord,
                    current.term.to_vec(),
                    expansion_limit,
                )?;
            }
            cursor.next()?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        let mut admitted_block = None;
        for (term_index, term) in delta.segment().sorted_terms().iter().enumerate() {
            let block = term_index / 16;
            if Some(block) != admitted_block {
                if let Some(checkpoint) = checkpoint {
                    checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
                }
                admitted_block = Some(block);
            }
            if term.field_ord() == field_ord
                && term.live_doc_freq() != 0
                && star_glob_matches(pattern, term.term())
            {
                insert_glob_term(&mut terms, field_ord, term.term().to_vec(), expansion_limit)?;
            }
        }
    }
    Ok(terms.into_iter().collect())
}

fn compiled_snippet_terms(
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    expansion_limit: usize,
) -> Result<Vec<SnippetTerm>, QuillIndexError> {
    let mut terms = BTreeSet::<Vec<u8>>::new();
    collect_snippet_term_bytes(query, snapshot, schema, expansion_limit, &mut terms)?;
    terms
        .into_iter()
        .map(|term| {
            let document_frequency = snapshot.bm25_doc_freq(CONTENT_FIELD, &term)?;
            let text = String::from_utf8(term)
                .map_err(|_| invalid_state("content term dictionary contains non-UTF-8 bytes"))?;
            Ok(SnippetTerm::new(text, document_frequency))
        })
        .collect()
}

fn collect_snippet_term_bytes(
    query: &Query,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    expansion_limit: usize,
    terms: &mut BTreeSet<Vec<u8>>,
) -> Result<(), QuillIndexError> {
    match query {
        Query::Term { fields, text }
            if fields.iter().any(|field| field.field_id == CONTENT_FIELD) =>
        {
            terms.insert(text.as_bytes().to_vec());
        }
        Query::Phrase {
            fields,
            terms: phrase_terms,
            prefix,
            ..
        } if fields.iter().any(|field| field.field_id == CONTENT_FIELD) => {
            let prefix_index = (*prefix).then_some(phrase_terms.len().saturating_sub(1));
            for (index, term) in phrase_terms.iter().enumerate() {
                if prefix_index == Some(index) {
                    let pattern = format!("{}*", term.text);
                    terms.extend(snapshot_glob_terms(
                        None,
                        snapshot,
                        schema,
                        CONTENT_FIELD,
                        pattern.as_bytes(),
                        expansion_limit,
                    )?);
                } else {
                    terms.insert(term.text.as_bytes().to_vec());
                }
            }
        }
        Query::Boolean { clauses, .. } => {
            for clause in clauses
                .iter()
                .filter(|clause| clause.occur != Occur::MustNot)
            {
                collect_snippet_term_bytes(
                    &clause.query,
                    snapshot,
                    schema,
                    expansion_limit,
                    terms,
                )?;
            }
        }
        Query::Set { field_id, values } if *field_id == CONTENT_FIELD => {
            terms.extend(values.iter().filter_map(|value| match value {
                QueryValue::Str(text) => Some(text.as_bytes().to_vec()),
                QueryValue::I64(_) | QueryValue::U64(_) => None,
            }));
        }
        Query::Glob { field_ids, pattern } if field_ids.contains(&CONTENT_FIELD) => {
            terms.extend(snapshot_glob_terms(
                None,
                snapshot,
                schema,
                CONTENT_FIELD,
                pattern.as_bytes(),
                expansion_limit,
            )?);
        }
        Query::Boost { query, .. } => {
            collect_snippet_term_bytes(query, snapshot, schema, expansion_limit, terms)?;
        }
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => {}
    }
    Ok(())
}

fn insert_glob_term(
    terms: &mut BTreeSet<Vec<u8>>,
    field_ord: u16,
    term: Vec<u8>,
    expansion_limit: usize,
) -> Result<(), QuillIndexError> {
    if terms.insert(term) && terms.len() > expansion_limit {
        return Err(TermDictionaryError::GlobExpansionLimitExceeded {
            field_ord,
            limit: expansion_limit,
            actual: terms.len(),
        }
        .into());
    }
    Ok(())
}

fn lower_leaf_term<'a>(
    leaf: QueryLeaf<'a>,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
    rank_pruning: bool,
    checkpoint: &QueryCheckpointHandle<'a>,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let stats = composite_snapshot_field(snapshot, field_ord)?;
    let doc_freq = checkpointed_snapshot_doc_freq(checkpoint, snapshot, field_ord, term)?;
    let record_option = term_record_option(schema, field_ord)?;
    match leaf {
        QueryLeaf::Sealed(segment) => {
            checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
            let (cursor, fieldnorms) =
                open_sealed_term_cursor(segment, schema, field_ord, term, rank_pruning)?;
            let cursor = CheckpointPostingCursor::new(cursor, Arc::clone(checkpoint))?;
            build_term_scorer(cursor, fieldnorms, stats, doc_freq, record_option, boost)
        }
        QueryLeaf::Delta(delta) => {
            let cursor = DeltaPostingCursor::new(delta, field_ord, term)?;
            let cursor = CheckpointPostingCursor::new(cursor, Arc::clone(checkpoint))?;
            build_term_scorer(
                cursor,
                DeltaFieldNorms::new(delta, field_ord),
                stats,
                doc_freq,
                record_option,
                boost,
            )
        }
    }
}

fn checkpointed_snapshot_doc_freq(
    checkpoint: &QueryCheckpointHandle<'_>,
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
    term: &[u8],
) -> Result<u64, QuillIndexError> {
    if snapshot.bm25_field_stats(field_ord).is_none() {
        return Err(invalid_state(format!(
            "snapshot has no BM25 statistics for field {field_ord}"
        )));
    }
    let mut total = 0_u64;
    for segment in snapshot.keeper_snapshot().segments() {
        let dictionary = open_dictionary(segment, snapshot.keeper_snapshot().schema())?;
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
        if let Some(found) = dictionary.lookup(field_ord, term)? {
            total = total
                .checked_add(u64::from(found.metadata.doc_freq))
                .ok_or_else(|| invalid_state("snapshot document frequency overflow"))?;
        }
    }
    for delta in snapshot.delta_snapshots() {
        let delta_doc_freq = delta
            .find_term(field_ord, term)
            .map_or(0, |found| found.live_doc_freq());
        total = total
            .checked_add(u64::try_from(delta_doc_freq).map_err(|_| {
                SnapshotError::CounterOverflow {
                    counter: "live Delta document frequency",
                }
            })?)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "snapshot document frequency",
            })?;
    }
    Ok(total)
}

fn open_sealed_term_cursor<'a>(
    segment: &'a RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    rank_pruning: bool,
) -> Result<(SealedPostingCursor<'a>, DocLenField<'a>), QuillIndexError> {
    let manifest = segment.manifest();
    let expected = term_field_ords(schema);
    let doclen = DocLenSection::parse(
        required_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        &expected,
    )?;
    let fieldnorms = doclen
        .field(field_ord)
        .ok_or_else(|| invalid_state(format!("DOCLEN has no field {field_ord}")))?;
    let dictionary = open_dictionary(segment, schema)?;
    let Some(found) = dictionary.lookup(field_ord, term)? else {
        let postings = PostingList::parse(&[], 0)?.into_cursor()?;
        let cursor = SealedPostingCursor::from_owned(postings, 0, segment.doc_count());
        return Ok((cursor, fieldnorms));
    };

    let postings_section = required_section(segment, SectionKind::POSTINGS)?;
    let postings_bytes = span(postings_section, found.metadata.postings, "POSTINGS")?;
    if let Some(cached) = segment
        .cached_rank_pruning_metadata(found.term_ord, found.metadata)
        .map_err(invalid_state)?
    {
        let cursor = if rank_pruning {
            SealedPostingCursor::from_validated_pruning(
                postings_bytes,
                cached,
                segment.doc_count(),
            )?
        } else {
            let size_hint = cached.doc_freq();
            SealedPostingCursor::from_owned(
                cached.cursor(postings_bytes)?,
                size_hint,
                segment.doc_count(),
            )
        };
        return Ok((cursor, fieldnorms));
    }
    let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)?;
    if !rank_pruning {
        let size_hint = postings.doc_freq();
        let cursor = postings.into_cursor()?;
        return Ok((
            SealedPostingCursor::from_owned(cursor, size_hint, segment.doc_count()),
            fieldnorms,
        ));
    }
    let blockmax_section = required_section(segment, SectionKind::BLOCKMAX)?;
    let blockmax_bytes = span(blockmax_section, found.metadata.blockmax, "BLOCKMAX")?;
    let pruning = Arc::new(postings.into_pruning_metadata(blockmax_bytes, fieldnorms)?);
    let pruning = segment
        .cache_rank_pruning_metadata(found.term_ord, found.metadata, pruning)
        .map_err(invalid_state)?;
    Ok((
        SealedPostingCursor::from_validated_pruning(postings_bytes, pruning, segment.doc_count())?,
        fieldnorms,
    ))
}

fn composite_snapshot_field(
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
) -> Result<SnapshotFieldStats, QuillIndexError> {
    snapshot
        .bm25_field_stats(field_ord)
        .ok_or_else(|| invalid_state(format!("snapshot has no field statistics for {field_ord}")))
}

#[cfg(test)]
fn lower_term(
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    let stats = snapshot_field(snapshot, field_ord)?;
    let doc_freq = snapshot_doc_freq(snapshot, schema, field_ord, term)?;
    let cursor = open_owned_cursor(segment, schema, field_ord, term, false, None)?;
    let norms = owned_fieldnorms(segment, schema, field_ord)?;
    build_term_scorer(
        cursor,
        norms,
        stats,
        doc_freq,
        term_record_option(schema, field_ord)?,
        boost,
    )
}

fn build_term_scorer<'a, C, F>(
    cursor: C,
    fieldnorms: F,
    stats: SnapshotFieldStats,
    snapshot_doc_freq: u64,
    record_option: TermRecordOption,
    boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError>
where
    C: PostingCursor + 'a,
    F: FieldNormReader + 'a,
{
    Ok(ReferenceScorer::term(TermScorer::new(
        cursor,
        fieldnorms,
        Bm25FieldSnapshot::new(stats)?,
        snapshot_doc_freq,
        record_option,
        boost,
    )?))
}

#[cfg(test)]
fn lower_composite_sealed_term(
    segment: &RecoveredSegment,
    snapshot: &QuillSearchSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    let stats = snapshot.bm25_field_stats(field_ord).ok_or_else(|| {
        invalid_state(format!("snapshot has no field statistics for {field_ord}"))
    })?;
    let doc_freq = snapshot.bm25_doc_freq(field_ord, term)?;
    let cursor = open_owned_cursor(segment, schema, field_ord, term, false, None)?;
    let norms = owned_fieldnorms(segment, schema, field_ord)?;
    build_term_scorer(
        cursor,
        norms,
        stats,
        doc_freq,
        term_record_option(schema, field_ord)?,
        boost,
    )
}

#[cfg(test)]
fn lower_delta_term<'a>(
    delta: &'a DeltaSnapshot,
    snapshot: &QuillSearchSnapshot,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'a>, QuillIndexError> {
    let stats = snapshot.bm25_field_stats(field_ord).ok_or_else(|| {
        invalid_state(format!("snapshot has no field statistics for {field_ord}"))
    })?;
    let doc_freq = snapshot.bm25_doc_freq(field_ord, term)?;
    build_term_scorer(
        DeltaPostingCursor::new(delta, field_ord, term)?,
        DeltaFieldNorms::new(delta, field_ord),
        stats,
        doc_freq,
        term_record_option(delta.schema(), field_ord)?,
        boost,
    )
}

#[cfg(test)]
fn snapshot_field(
    snapshot: &KeeperSnapshot,
    field_ord: u16,
) -> Result<SnapshotFieldStats, QuillIndexError> {
    snapshot
        .loaded_manifest()
        .manifest
        .field_stats
        .iter()
        .find(|row| row.field_ord == field_ord)
        .map(|row| SnapshotFieldStats {
            field_ord,
            total_tokens: row.total_tokens,
            doc_count: u64::from(row.doc_count),
        })
        .ok_or_else(|| invalid_state(format!("manifest has no field statistics for {field_ord}")))
}

fn snapshot_doc_freq(
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
) -> Result<u64, QuillIndexError> {
    let mut total = 0_u64;
    for segment in snapshot.segments() {
        let dictionary = open_dictionary(segment, schema)?;
        if let Some(found) = dictionary.lookup(field_ord, term)? {
            total = total
                .checked_add(u64::from(found.metadata.doc_freq))
                .ok_or_else(|| invalid_state("snapshot document frequency overflow"))?;
        }
    }
    Ok(total)
}

fn open_owned_cursor(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    positioned: bool,
    checkpoint: Option<&QueryCheckpointHandle<'_>>,
) -> Result<OwnedPostingCursor, QuillIndexError> {
    let dictionary = open_dictionary(segment, schema)?;
    if let Some(checkpoint) = checkpoint {
        checkpoint.admit(QueryWorkKind::DictionaryBlock, 1)?;
    }
    let Some(found) = dictionary.lookup(field_ord, term)? else {
        return Ok(OwnedPostingCursor::empty(segment.doc_count(), positioned));
    };
    let postings_section = required_section(segment, SectionKind::POSTINGS)?;
    let postings_bytes = span(postings_section, found.metadata.postings, "POSTINGS")?;
    let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)?;
    let posting_count = usize::try_from(found.metadata.doc_freq)
        .map_err(|_| invalid_state("posting count does not fit usize"))?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(posting_count).map_err(|_| {
        PostingCodecError::MaterializationAllocation {
            count: posting_count,
        }
    })?;
    let mut posting_cursor = postings.cursor()?;
    let mut admitted_block = None;
    while let Some(posting) = posting_cursor.current() {
        let block = posting_cursor
            .block_index()
            .ok_or_else(|| invalid_state("positioned posting cursor has no block index"))?;
        if admitted_block != Some(block) {
            if let Some(checkpoint) = checkpoint {
                checkpoint.admit(QueryWorkKind::PostingBlock, 1)?;
            }
            admitted_block = Some(block);
        }
        rows.push(posting);
        posting_cursor.next()?;
    }
    let positions = if positioned {
        let position_span =
            found
                .metadata
                .positions
                .ok_or_else(|| QuillIndexError::UnsupportedQuery {
                    detail: format!("field {field_ord} has no positions"),
                })?;
        let position_section = required_section(segment, SectionKind::POSITIONS)?;
        let position_bytes = span(position_section, position_span, "POSITIONS")?;
        let positions = PositionList::parse(position_bytes, &postings)?;
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(rows.len())
            .map_err(|_| invalid_state("could not allocate owned position rows"))?;
        for ordinal in 0..found.metadata.doc_freq {
            let row = positions
                .positions_for_ordinal(ordinal)?
                .collect::<Result<Vec<_>, _>>()?;
            decoded.push(row);
        }
        Some(decoded)
    } else {
        None
    };
    Ok(OwnedPostingCursor {
        postings: rows,
        positions,
        cursor: 0,
        segment_num_docs: segment.doc_count(),
    })
}

fn owned_fieldnorms(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<OwnedFieldNorms, QuillIndexError> {
    let expected = term_field_ords(schema);
    let manifest = segment.manifest();
    let section = DocLenSection::parse(
        required_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        &expected,
    )?;
    let field = section
        .field(field_ord)
        .ok_or_else(|| invalid_state(format!("DOCLEN has no field {field_ord}")))?;
    Ok(OwnedFieldNorms {
        field_ord,
        docid_lo: manifest.docid_lo,
        values: field.fieldnorm_ids().to_vec(),
    })
}

fn open_dictionary(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
) -> Result<TermDictionary<'_>, QuillIndexError> {
    Ok(segment.term_dictionary(schema)?)
}

fn required_section(
    segment: &RecoveredSegment,
    kind: SectionKind,
) -> Result<&[u8], QuillIndexError> {
    segment
        .section(kind)?
        .ok_or_else(|| invalid_state(format!("segment is missing section {}", kind.raw())))
}

fn span<'a>(
    section: &'a [u8],
    span: ByteSpan,
    name: &'static str,
) -> Result<&'a [u8], QuillIndexError> {
    let start = usize::try_from(span.offset)
        .map_err(|_| invalid_state(format!("{name} span start does not fit usize")))?;
    let len = usize::try_from(span.len)
        .map_err(|_| invalid_state(format!("{name} span length does not fit usize")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_state(format!("{name} span overflows usize")))?;
    section
        .get(start..end)
        .ok_or_else(|| invalid_state(format!("{name} span lies outside its section")))
}

fn term_field_ords(schema: SchemaDescriptor) -> Vec<u16> {
    schema
        .fields
        .iter()
        .filter_map(|field| match field.kind {
            crate::schema::FieldKind::Keyword | crate::schema::FieldKind::Text { .. } => {
                Some(field.id)
            }
            crate::schema::FieldKind::StoredOnly
            | crate::schema::FieldKind::I64 { .. }
            | crate::schema::FieldKind::U64 { .. } => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use asupersync::runtime::yield_now;
    use asupersync::types::Budget;
    use asupersync::{LabConfig, LabRuntime};
    #[cfg(feature = "durability")]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig};

    use super::*;
    use crate::argus::CollectedTopDocs;
    use crate::contract::fieldnorm_to_id;
    use crate::delta::{
        DeltaFieldNorm, DeltaNumericValue, DeltaSegment, DeltaStoredValue, DeltaTermPosting,
    };
    use crate::keeper::{CompactionError, ConcatMergeError};
    #[cfg(feature = "bench-internals")]
    use crate::query::{BooleanClause, QueryField};
    use crate::quiver::{
        EncodedPositionList, EncodedPostingList, StatsSection, aggregate_field_stats,
    };
    use crate::schema::{Analyzer, FSFS_CHUNK_SCHEMA, FieldDescriptor};

    const CONCAT_MERGE_QUERIES: [&str; 4] =
        ["rust", "python", "rust OR python", "\"rust ownership\""];
    const Q1_OB2A_QUERIES: [&str; 5] = [
        "shared",
        "left",
        "right",
        "left OR right",
        "\"shared left\"",
    ];
    const Q1_OB4_QUERIES: [&str; 5] = [
        "shared",
        "left",
        "shared AND left",
        "\"shared left\"",
        "ord:[0 TO 39]",
    ];

    #[cfg(feature = "pruning-conformance")]
    #[test]
    fn pruning_conformance_session_is_dense_one_shot_and_fail_closed() {
        let completed = ConformancePruningTraceSession::new(2);
        completed
            .bind_execution_mode(ConformancePruningExecutionMode::Rayon)
            .expect("bind exact shipping branch");
        completed
            .record(1, 7, &[])
            .expect("record second segment first");
        completed
            .record(0, 5, &[])
            .expect("record first segment second");
        let receipt = completed.complete().expect("complete dense trace");
        assert_eq!(
            receipt.execution_mode(),
            ConformancePruningExecutionMode::Rayon
        );
        assert_eq!(
            receipt
                .segments()
                .iter()
                .map(ConformanceSegmentPruningReceipt::segment_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1],
        );
        assert_eq!(
            receipt
                .segments()
                .iter()
                .map(ConformanceSegmentPruningReceipt::segment_doc_count)
                .collect::<Vec<_>>(),
            vec![5, 7],
        );
        assert!(matches!(
            completed.complete(),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("already completed")
        ));

        let record_before_bind = ConformancePruningTraceSession::new(1);
        assert!(matches!(
            record_before_bind.record(0, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("was not bound before recording")
        ));
        assert!(matches!(
            record_before_bind.bind_execution_mode(ConformancePruningExecutionMode::Serial),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));

        let double_bind = ConformancePruningTraceSession::new(1);
        double_bind
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind one exact execution mode");
        assert!(matches!(
            double_bind.bind_execution_mode(ConformancePruningExecutionMode::Rayon),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("already bound")
        ));
        assert!(matches!(
            double_bind.record(0, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));

        let out_of_range = ConformancePruningTraceSession::new(1);
        out_of_range
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind out-of-range session");
        assert!(matches!(
            out_of_range.record(1, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("exceeds expected count")
        ));
        assert!(matches!(
            out_of_range.record(0, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));

        let duplicate = ConformancePruningTraceSession::new(1);
        duplicate
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind serial path");
        duplicate.record(0, 1, &[]).expect("first receipt");
        assert!(matches!(
            duplicate.record(0, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("recorded twice")
        ));
        assert!(matches!(
            duplicate.complete(),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));

        let incomplete = ConformancePruningTraceSession::new(2);
        incomplete
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind incomplete serial path");
        incomplete
            .record(0, 1, &[])
            .expect("record partial receipt");
        assert!(matches!(
            incomplete.complete(),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("trace is incomplete")
        ));
        assert!(matches!(
            incomplete.record(1, 1, &[]),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));

        let dropped_controller = Arc::new(ConformanceCancellationController::default());
        dropped_controller
            .arm(
                ConformanceCancellationStage::PruningTraceSegmentRecorded,
                u64::MAX,
            )
            .expect("arm dropped-session accounting");
        let dropped_guard = ConformancePruningTraceGuard::new(1, Arc::clone(&dropped_controller));
        let dropped_session = Arc::clone(&dropped_guard.session);
        dropped_session
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind guard-owned session");
        dropped_session
            .record(0, 1, &[])
            .expect("record partial guard-owned receipt");
        drop(dropped_guard);
        assert!(matches!(
            dropped_session.complete(),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("has failed")
        ));
        assert_eq!(
            dropped_controller.discarded_pruning_trace_sessions(),
            1,
            "dropping an incomplete invocation records one discarded session"
        );

        let generation_controller = Arc::new(ConformanceCancellationController::default());
        generation_controller
            .arm(
                ConformanceCancellationStage::PruningTraceSegmentRecorded,
                u64::MAX,
            )
            .expect("arm stale generation");
        let stale_guard = ConformancePruningTraceGuard::new(1, Arc::clone(&generation_controller));
        stale_guard
            .session()
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind stale guard");
        generation_controller.disarm();
        generation_controller
            .arm(
                ConformanceCancellationStage::PruningTraceSegmentRecorded,
                u64::MAX,
            )
            .expect("arm successor generation");
        drop(stale_guard);
        assert_eq!(
            generation_controller.discarded_pruning_trace_sessions(),
            0,
            "a stale guard must not contaminate a later arm generation"
        );
        let current_guard =
            ConformancePruningTraceGuard::new(1, Arc::clone(&generation_controller));
        current_guard
            .session()
            .bind_execution_mode(ConformancePruningExecutionMode::Serial)
            .expect("bind current guard");
        drop(current_guard);
        assert_eq!(
            generation_controller.discarded_pruning_trace_sessions(),
            1,
            "the current arm generation must retain its own discarded-session count"
        );

        let exhausted_controller = ConformanceCancellationController::default();
        exhausted_controller
            .arm_generation
            .store(u64::MAX, Ordering::Relaxed);
        assert!(matches!(
            exhausted_controller.arm(
                ConformanceCancellationStage::PruningTraceSegmentRecorded,
                1,
            ),
            Err(QuillIndexError::InvalidState { detail })
                if detail.contains("arm generation is exhausted")
        ));
        assert_eq!(
            exhausted_controller.stage.load(Ordering::Acquire),
            ConformanceCancellationController::DISARMED,
            "generation exhaustion must leave the controller disarmed"
        );
    }

    #[cfg(feature = "bench-internals")]
    const POSITIONLESS_QG_FIELDS: [FieldDescriptor; 5] = [
        FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 3,
            name: "metadata_json",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: 4,
            name: "ord",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: true,
        },
    ];
    #[cfg(feature = "bench-internals")]
    const POSITIONLESS_QG_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "qg-positionless-typed-capability-test",
        fields: &POSITIONLESS_QG_FIELDS,
    };
    #[cfg(feature = "bench-internals")]
    const POSITIONED_QG_FIELDS: [FieldDescriptor; 5] = [
        FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 3,
            name: "metadata_json",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: 4,
            name: "ord",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: true,
        },
    ];
    #[cfg(feature = "bench-internals")]
    const POSITIONED_QG_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "qg-positioned-capability-control",
        fields: &POSITIONED_QG_FIELDS,
    };
    const KNOWN_SECTION_KINDS: [SectionKind; 10] = [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::POSITIONS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
        SectionKind::IDMAP,
        SectionKind::IDHASH,
        SectionKind::NUMERIC,
        SectionKind::STOREDMETA,
        SectionKind::STATS,
    ];
    const DELTA_PARITY_FIELDS: [FieldDescriptor; 6] = [
        FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "rank",
            kind: FieldKind::U64 {
                indexed: true,
                fast: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 3,
            name: "opaque",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: 4,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 5,
            name: "fast_rank",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: false,
        },
    ];
    const DELTA_PARITY_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "delta-seal-parity-v1",
        fields: &DELTA_PARITY_FIELDS,
    };

    fn run_with_cx<F, Fut>(test: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(test);
    }

    fn run_with_blocking_cx<F, Fut>(test: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        let cx = Cx::for_testing();
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(1, 4)
            .build()
            .expect("build test runtime with a blocking pool");
        runtime.block_on(test(cx));
    }

    fn deterministic_config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        }
    }

    #[test]
    fn ranked_query_cache_requires_an_exact_key_and_snapshot_epoch() {
        let cache = RankedQueryCache::default();
        let raw_result = QuillSearchResult {
            hits: vec![QuillHit {
                document_id: "raw-hit".to_owned(),
                global_docid: 7,
                score: 3.5,
            }],
            total_count: None,
            doc_count: 11,
            diagnostics: Vec::new(),
        };
        cache.insert_raw(4, "rust", 10, 2, false, &raw_result);

        assert_eq!(cache.get_raw(4, "rust", 10, 2, false), Some(raw_result));
        assert!(cache.get_raw(5, "rust", 10, 2, false).is_none());
        assert!(cache.get_raw(4, "Rust", 10, 2, false).is_none());
        assert!(cache.get_raw(4, "rust", 11, 2, false).is_none());
        assert!(cache.get_raw(4, "rust", 10, 3, false).is_none());
        assert!(cache.get_raw(4, "rust", 10, 2, true).is_none());

        let query = Query::Boost {
            query: Box::new(Query::All),
            factor: 2.0,
        };
        let preparsed_result = QuillSearchResult {
            hits: Vec::new(),
            total_count: Some(11),
            doc_count: 11,
            diagnostics: Vec::new(),
        };
        cache.insert_preparsed(4, &query, 0, 0, true, &preparsed_result);
        assert_eq!(
            cache.get_preparsed(4, &query, 0, 0, true),
            Some(preparsed_result)
        );
        assert!(cache.get_preparsed(5, &query, 0, 0, true).is_none());
        assert!(
            cache
                .get_preparsed(
                    4,
                    &Query::Boost {
                        query: Box::new(Query::All),
                        factor: 2.5,
                    },
                    0,
                    0,
                    true,
                )
                .is_none()
        );
    }

    fn fuel_diagnostics(error: &QuillIndexError) -> (u64, u64, u64, u64, u64, u64) {
        let QuillIndexError::QueryFuelExhausted {
            budget,
            consumed,
            segments_touched,
            dictionary_blocks,
            posting_blocks,
            position_docs,
        } = error
        else {
            panic!("expected typed query fuel exhaustion, got {error:?}");
        };
        (
            *budget,
            *consumed,
            *segments_touched,
            *dictionary_blocks,
            *posting_blocks,
            *position_docs,
        )
    }

    #[test]
    fn query_fuel_checkpoint_diagnostics_and_cancellation_are_deterministic() {
        fn exhaust_once(cx: &Cx) -> (u64, u64, u64, u64, u64, u64) {
            let checkpoint = QueryCheckpoint::new(cx, "fuel_test", 4, 5);
            checkpoint
                .admit(QueryWorkKind::Segment, 1)
                .expect("admit segment");
            checkpoint
                .admit(QueryWorkKind::DictionaryBlock, 1)
                .expect("admit dictionary block");
            checkpoint
                .admit(QueryWorkKind::PostingBlock, 1)
                .expect("admit posting block");
            checkpoint
                .admit(QueryWorkKind::PositionDocument, 1)
                .expect("admit position document");
            let error = checkpoint
                .admit(QueryWorkKind::PostingBlock, 1)
                .expect_err("fifth coarse unit must exhaust a four-unit budget");
            fuel_diagnostics(&QuillIndexError::from(error))
        }

        let cx = Cx::for_testing();
        let first = exhaust_once(&cx);
        assert_eq!(exhaust_once(&cx), first);
        assert_eq!(first, (4, 4, 1, 1, 1, 1));

        for (schedule, kind) in [
            QueryWorkKind::Segment,
            QueryWorkKind::DictionaryBlock,
            QueryWorkKind::PostingBlock,
            QueryWorkKind::PositionDocument,
        ]
        .into_iter()
        .enumerate()
        {
            let seed = 0xf0e1_0000_u64
                .checked_add(u64::try_from(schedule).expect("checkpoint schedule fits u64"))
                .expect("checkpoint schedule seed");
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(10_000));
            let region = lab.state.create_root_region(Budget::INFINITE);
            let cancelled = Arc::new(Cx::for_testing());
            let observed = Arc::new(AtomicBool::new(false));

            let query_cx = Arc::clone(&cancelled);
            let query_observed = Arc::clone(&observed);
            let (query, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while !query_cx.is_cancel_requested() {
                        yield_now().await;
                    }
                    let checkpoint =
                        QueryCheckpoint::new(query_cx.as_ref(), "checkpoint_test", 4, 5);
                    assert!(matches!(
                        checkpoint.admit(kind, 1),
                        Err(ArgusError::QueryCancelled {
                            phase: "checkpoint_test"
                        })
                    ));
                    query_observed.store(true, Ordering::SeqCst);
                })
                .expect("create checkpoint query task");
            let cancel_cx = Arc::clone(&cancelled);
            let (cancel, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    yield_now().await;
                    cancel_cx.set_cancel_requested(true);
                })
                .expect("create checkpoint cancellation task");

            lab.scheduler.lock().schedule(query, 0);
            lab.scheduler.lock().schedule(cancel, 0);
            let report = lab.run_until_quiescent_with_report();
            assert!(
                observed.load(Ordering::SeqCst),
                "checkpoint kind {kind:?} did not observe cancellation"
            );
            assert!(report.quiescent, "checkpoint schedule did not quiesce");
            assert!(
                report.oracle_report.all_passed(),
                "checkpoint schedule oracle failed: {:?}",
                report.oracle_report
            );
            assert!(
                report.invariant_violations.is_empty(),
                "checkpoint schedule invariants failed: {:?}",
                report.invariant_violations
            );
        }
    }

    const E3_9_PINNED_SEEDS: [u64; 4] = [
        0xe3_9000_0000_0001,
        0xe3_9000_0000_001d,
        0xe3_9000_0000_0101,
        0xe3_9000_0000_1009,
    ];
    const E3_9_RANDOM_SEED_COUNT_ENV: &str = "QUILL_E3_9_RANDOM_SEEDS";
    const E6_5_PINNED_SEEDS: [u64; 4] = [
        0xe6_5000_0000_0001,
        0xe6_5000_0000_001d,
        0xe6_5000_0000_0101,
        0xe6_5000_0000_1009,
    ];
    const E6_5_RANDOM_SEED_COUNT_ENV: &str = "QUILL_E6_5_RANDOM_SEEDS";

    fn lab_seed_corpus(pinned: &[u64], random_seed_count_env: &str, salt: u64) -> Vec<u64> {
        let random_seed_count = match std::env::var(random_seed_count_env) {
            Ok(value) => value.parse::<usize>().unwrap_or_else(|error| {
                panic!("{random_seed_count_env}={value:?} is not a seed count: {error}")
            }),
            Err(std::env::VarError::NotPresent) => 0,
            Err(error) => panic!("failed to read {random_seed_count_env}: {error}"),
        };
        let mut seeds = pinned.to_vec();
        if random_seed_count == 0 {
            return seeds;
        }

        let epoch_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("nightly seed clock predates the Unix epoch")
            .as_nanos();
        let mut state = u64::try_from(epoch_nanos & u128::from(u64::MAX))
            .expect("masked epoch nanoseconds fit u64")
            ^ salt;
        for _ in 0..random_seed_count {
            // SplitMix64 gives the scheduled campaign a fresh, well-spread
            // corpus while every assertion still prints the exact replay seed.
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut seed = state;
            seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            seeds.push(seed ^ (seed >> 31));
        }
        seeds
    }

    fn e3_9_seed_corpus() -> Vec<u64> {
        lab_seed_corpus(
            &E3_9_PINNED_SEEDS,
            E3_9_RANDOM_SEED_COUNT_ENV,
            0xe3_9000_9e37_79b9,
        )
    }

    fn e6_5_seed_corpus() -> Vec<u64> {
        lab_seed_corpus(
            &E6_5_PINNED_SEEDS,
            E6_5_RANDOM_SEED_COUNT_ENV,
            0xe6_5000_9e37_79b9,
        )
    }

    fn assert_e3_9_lab_report(scenario: &str, seed: u64, report: &asupersync::lab::LabRunReport) {
        let replay = format!(
            "{scenario}: seed={seed:#018x} fingerprint={:#018x}",
            report.trace_fingerprint
        );
        assert!(report.quiescent, "{replay}: LabRuntime did not quiesce");
        assert!(
            report.oracle_report.all_passed(),
            "{replay}: LabRuntime oracle failure: {:?}",
            report.oracle_report
        );
        assert!(
            report.invariant_violations.is_empty(),
            "{replay}: LabRuntime invariant failures: {:?}",
            report.invariant_violations
        );
    }

    fn assert_pairwise_disjoint_manifest(segments: &[ManifestSegment]) {
        for pair in segments.windows(2) {
            assert!(
                pair[0].docid_hi <= pair[1].docid_lo,
                "manifest intervals overlap: {:?} then {:?}",
                pair[0],
                pair[1],
            );
        }
    }

    fn shipping_content_hash(document_id: &str, content: &str) -> u64 {
        let document = IndexableDocument::new(document_id, content);
        let metadata = canonical_metadata(&document.metadata).expect("canonical fixture metadata");
        canonical_document_content_hash(&document, &metadata)
            .expect("canonical fixture document hash")
    }

    #[test]
    fn content_hash_has_unambiguous_fields_and_canonical_metadata() {
        let first = IndexableDocument::new("ab", "c")
            .with_title("title")
            .with_metadata("zeta", "last")
            .with_metadata("alpha", "first");
        let reordered = IndexableDocument::new("ab", "c")
            .with_title("title")
            .with_metadata("alpha", "first")
            .with_metadata("zeta", "last");
        let boundary_shifted = IndexableDocument::new("a", "bc")
            .with_title("title")
            .with_metadata("alpha", "first")
            .with_metadata("zeta", "last");

        let first_hash = indexable_document_content_hash(&first).expect("hash first document");
        assert_eq!(
            first_hash,
            indexable_document_content_hash(&reordered).expect("hash reordered metadata")
        );
        assert_ne!(
            first_hash,
            indexable_document_content_hash(&boundary_shifted)
                .expect("hash field-boundary-shifted document")
        );
    }

    fn fixture_documents() -> Vec<IndexableDocument> {
        vec![
            IndexableDocument::new("rust-1", "rust ownership prevents data races")
                .with_title("Rust ownership")
                .with_metadata("cluster", "systems"),
            IndexableDocument::new("rust-2", "safe systems programming with rust")
                .with_title("Borrow checker")
                .with_metadata("cluster", "systems"),
            IndexableDocument::new("python-1", "python data science notebooks")
                .with_title("Python guide")
                .with_metadata("cluster", "data"),
        ]
    }

    fn apply_alpha_delta(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        content_frequency: u32,
    ) {
        apply_sealable_delta_document(delta, global_docid, document_id, "alpha", content_frequency);
    }

    fn apply_sealable_delta_document(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        content_term: &str,
        content_frequency: u32,
    ) {
        let positions = (0..content_frequency).collect::<Vec<_>>();
        let content = std::iter::repeat_n(
            content_term,
            usize::try_from(content_frequency).expect("fixture frequency fits usize"),
        )
        .collect::<Vec<_>>()
        .join(" ");
        let ordinal = u64::from(global_docid).to_le_bytes();
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: CONTENT_FIELD,
                raw_length: content_frequency,
                fieldnorm_id: fieldnorm_to_id(content_frequency),
            },
            DeltaFieldNorm {
                field_ord: TITLE_FIELD,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let postings = [
            DeltaTermPosting {
                field_ord: ID_FIELD,
                term: document_id.as_bytes(),
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: CONTENT_FIELD,
                term: content_term.as_bytes(),
                frequency: content_frequency,
                positions: Some(&positions),
            },
        ];
        let stored = [
            DeltaStoredValue::new(ID_FIELD, document_id.as_bytes()),
            DeltaStoredValue::new(CONTENT_FIELD, content.as_bytes()),
            DeltaStoredValue::new(TITLE_FIELD, b""),
            DeltaStoredValue::new(METADATA_FIELD, b"{}"),
            DeltaStoredValue::new(ORD_FIELD, &ordinal),
        ];
        delta
            .apply_document_with_values(
                global_docid,
                frankensearch_core::DocId::from(document_id),
                shipping_content_hash(document_id, &content),
                &fieldnorms,
                &postings,
                &[],
                &stored,
            )
            .expect("apply Delta fixture document");
    }

    fn apply_tokenized_delta_document(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        content: &str,
    ) {
        let content_has_positions = delta
            .schema()
            .fields
            .iter()
            .find(|field| field.id == CONTENT_FIELD)
            .is_some_and(|field| {
                matches!(
                    field.kind,
                    FieldKind::Text {
                        positions: true,
                        ..
                    }
                )
            });
        let mut term_positions = BTreeMap::<&str, Vec<u32>>::new();
        for (position, term) in content.split_ascii_whitespace().enumerate() {
            term_positions.entry(term).or_default().push(
                u32::try_from(position).expect("fixture token position fits the Delta wire type"),
            );
        }
        let token_count = term_positions
            .values()
            .map(Vec::len)
            .try_fold(0_u32, |count, term_count| {
                count.checked_add(
                    u32::try_from(term_count).expect("fixture term frequency fits u32"),
                )
            })
            .expect("fixture token count fits u32");
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: CONTENT_FIELD,
                raw_length: token_count,
                fieldnorm_id: fieldnorm_to_id(token_count),
            },
            DeltaFieldNorm {
                field_ord: TITLE_FIELD,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let mut postings = Vec::with_capacity(term_positions.len() + 1);
        postings.push(DeltaTermPosting {
            field_ord: ID_FIELD,
            term: document_id.as_bytes(),
            frequency: 1,
            positions: None,
        });
        for (term, positions) in &term_positions {
            postings.push(DeltaTermPosting {
                field_ord: CONTENT_FIELD,
                term: term.as_bytes(),
                frequency: u32::try_from(positions.len()).expect("fixture frequency fits u32"),
                positions: content_has_positions.then_some(positions.as_slice()),
            });
        }
        let ordinal = u64::from(global_docid).to_le_bytes();
        let stored = [
            DeltaStoredValue::new(ID_FIELD, document_id.as_bytes()),
            DeltaStoredValue::new(CONTENT_FIELD, content.as_bytes()),
            DeltaStoredValue::new(TITLE_FIELD, b""),
            DeltaStoredValue::new(METADATA_FIELD, b"{}"),
            DeltaStoredValue::new(ORD_FIELD, &ordinal),
        ];
        delta
            .apply_document_with_values(
                global_docid,
                DocId::from(document_id),
                shipping_content_hash(document_id, content),
                &fieldnorms,
                &postings,
                &[],
                &stored,
            )
            .expect("apply tokenized Delta fixture document");
    }

    fn apply_typed_delta_document(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        content: &str,
        rank: u64,
    ) {
        let mut term_positions = BTreeMap::<&str, Vec<u32>>::new();
        for (position, term) in content.split_ascii_whitespace().enumerate() {
            term_positions.entry(term).or_default().push(
                u32::try_from(position).expect("typed fixture position fits the Delta wire type"),
            );
        }
        let token_count = u32::try_from(term_positions.values().map(Vec::len).sum::<usize>())
            .expect("typed fixture token count fits u32");
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: 0,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: 1,
                raw_length: token_count,
                fieldnorm_id: fieldnorm_to_id(token_count),
            },
            DeltaFieldNorm {
                field_ord: 4,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let mut postings = Vec::with_capacity(term_positions.len() + 1);
        postings.push(DeltaTermPosting {
            field_ord: 0,
            term: document_id.as_bytes(),
            frequency: 1,
            positions: None,
        });
        for (term, positions) in &term_positions {
            postings.push(DeltaTermPosting {
                field_ord: 1,
                term: term.as_bytes(),
                frequency: u32::try_from(positions.len())
                    .expect("typed fixture frequency fits u32"),
                positions: Some(positions),
            });
        }
        let stored = [
            DeltaStoredValue::new(0, document_id.as_bytes()),
            DeltaStoredValue::new(1, content.as_bytes()),
            DeltaStoredValue::new(3, b""),
        ];
        delta
            .apply_document_with_values(
                global_docid,
                DocId::from(document_id),
                shipping_content_hash(document_id, content),
                &fieldnorms,
                &postings,
                &[
                    DeltaNumericValue::u64(2, rank),
                    DeltaNumericValue::u64(5, rank),
                ],
                &stored,
            )
            .expect("apply typed lowering fixture document");
    }

    fn typed_delta_snapshot(
        lease_base: u64,
        global_docid: u32,
        document_id: &str,
        content: &str,
        rank: u64,
    ) -> Arc<DeltaSnapshot> {
        let mut delta = DeltaSegment::new(DELTA_PARITY_SCHEMA, lease_base, usize::MAX)
            .expect("typed lowering Delta lease");
        apply_typed_delta_document(&mut delta, global_docid, document_id, content, rank);
        Arc::new(delta.freeze(0))
    }

    fn typed_residency_index(
        sealed: &[Arc<DeltaSnapshot>],
        deltas: &[Arc<DeltaSnapshot>],
        config: QuillConfig,
    ) -> QuillIndex {
        let genesis =
            KeeperSnapshot::in_memory(DELTA_PARITY_SCHEMA).expect("typed lowering genesis Keeper");
        let keeper = if sealed.is_empty() {
            genesis
        } else {
            let mut manifest = genesis
                .next_manifest()
                .expect("typed lowering successor MANIFEST");
            let mut encoded = Vec::new();
            for (index, delta) in sealed.iter().enumerate() {
                let segment = flush_delta_snapshot(
                    delta,
                    DeltaFlushInput {
                        segment_id: 0xe5_5000
                            + u64::try_from(index).expect("typed segment index fits u64"),
                        created_unix_s: 0,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .expect("flush typed lowering Delta")
                .expect("typed lowering Delta is non-empty");
                manifest.segments.push(manifest_segment(
                    &segment,
                    u64::try_from(index + 1).expect("typed seal sequence fits u64"),
                ));
                manifest.docid_high_watermark =
                    manifest.docid_high_watermark.max(delta.lease_end());
                encoded.push(segment);
            }
            manifest.field_stats = DELTA_PARITY_SCHEMA
                .fields
                .iter()
                .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
                .map(|field| ManifestFieldStats {
                    field_ord: field.id,
                    total_tokens: sealed
                        .iter()
                        .map(|delta| {
                            delta
                                .live_total_tokens(field.id)
                                .expect("typed Delta field statistics")
                        })
                        .sum(),
                    doc_count: sealed
                        .iter()
                        .map(|delta| {
                            u32::try_from(delta.live_document_count())
                                .expect("typed Delta count fits u32")
                        })
                        .sum(),
                })
                .collect();
            genesis
                .publish_owned_segments(&manifest, encoded)
                .expect("publish typed lowering Keeper")
        };
        let generation = keeper.loaded_manifest().manifest.generation;
        let rebound = deltas
            .iter()
            .map(|delta| Arc::new(delta.rebind_keeper_generation(generation)))
            .collect::<Vec<_>>();
        let index =
            QuillIndex::from_backend(IndexBackend::Memory(keeper), DELTA_PARITY_SCHEMA, config)
                .expect("bind typed lowering index");
        index
            .publish_delta_table(rebound)
            .expect("publish typed lowering Delta table");
        index
    }

    fn execute_typed_query(
        index: &QuillIndex,
        cx: &Cx,
        query: &Query,
    ) -> (QuillSearchResult, Vec<u32>) {
        let snapshot = index.search_snapshot();
        let ranked = index
            .execute_ranked_query(cx, query, &snapshot, 10, 0, true, Vec::new())
            .expect("execute typed ranked query");
        let docids = index
            .execute_docid_query(cx, query, &snapshot)
            .expect("execute typed doc-set query");
        (ranked, docids)
    }

    fn alpha_snapshot_tuple(snapshot: &QuillSearchSnapshot) -> (u64, u64, u64, u64, u64) {
        let stats = snapshot
            .bm25_field_stats(CONTENT_FIELD)
            .expect("content snapshot statistics");
        (
            snapshot.snapshot_epoch(),
            snapshot.live_doc_count(),
            stats.doc_count,
            stats.total_tokens,
            snapshot
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("alpha document frequency"),
        )
    }

    async fn concat_merge_fixture_index(cx: &Cx) -> QuillIndex {
        let documents = fixture_documents();
        let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        for (batch_index, document) in documents.iter().enumerate() {
            index
                .index_documents(cx, std::slice::from_ref(document))
                .await
                .expect("accumulate concat-merge leaf");
            index.commit(cx).await.expect("commit concat-merge leaf");
            assert_eq!(
                index.snapshot().segments().len(),
                batch_index + 1,
                "each committed fixture batch must seal one leaf segment",
            );
        }
        index
    }

    struct Q1Ob2aSeal {
        encoded: EncodedSegment,
        field_stats: BTreeMap<u16, (u64, u32)>,
    }

    fn q1_ob2a_documents() -> Vec<IndexableDocument> {
        (0_u32..400)
            .map(|ordinal| {
                let cohort = if ordinal < 100 { "left" } else { "right" };
                IndexableDocument::new(
                    format!("q1-ob2a-{ordinal:03}"),
                    format!("shared {cohort} concat parity"),
                )
                .with_title(format!("{cohort} cohort"))
                .with_metadata("cohort", cohort)
            })
            .collect()
    }

    fn q1_ob2a_doc_ord(first_doc_ord: u32, offset: usize) -> u32 {
        first_doc_ord
            .checked_add(u32::try_from(offset).expect("Q1-OB2a offset fits u32"))
            .expect("Q1-OB2a document ordinal stays in one lease")
    }

    fn seal_q1_ob2a_documents(
        documents: &[IndexableDocument],
        first_doc_ord: u32,
        segment_id: u64,
    ) -> Q1Ob2aSeal {
        let mut accumulator =
            ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("Q1-OB2a accumulator");
        let mut content_hashes = Vec::with_capacity(documents.len());
        for (offset, document) in documents.iter().enumerate() {
            let doc_ord = q1_ob2a_doc_ord(first_doc_ord, offset);
            let metadata = canonical_metadata(&document.metadata).expect("canonical metadata");
            let ordinal = u64::from(doc_ord).to_le_bytes();
            let title = document.title.as_deref().unwrap_or("");
            let indexed = [
                IndexedFieldValue::new(ID_FIELD, &document.id),
                IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                IndexedFieldValue::new(TITLE_FIELD, title),
            ];
            let stored = [
                StoredFieldValue::new(METADATA_FIELD, &metadata),
                StoredFieldValue::new(ORD_FIELD, &ordinal),
            ];
            accumulator
                .add_document_with_values(doc_ord, &indexed, &[], &stored)
                .expect("accumulate Q1-OB2a document");
            content_hashes.push(
                canonical_document_content_hash(document, &metadata)
                    .expect("canonical Q1-OB2a content hash"),
            );
        }

        let doc_count = u32::try_from(documents.len()).expect("Q1-OB2a document count fits u32");
        let field_stats = accumulator
            .fields()
            .iter()
            .map(|field| (field.field_ord(), (field.total_tokens(), doc_count)))
            .collect::<BTreeMap<_, _>>();
        let identities = documents
            .iter()
            .zip(&content_hashes)
            .enumerate()
            .map(|(offset, (document, &content_hash))| {
                FlushDocumentInput::new(
                    q1_ob2a_doc_ord(first_doc_ord, offset),
                    &document.id,
                    content_hash,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &identities,
            },
            FlushMode::Scalar,
        )
        .expect("seal Q1-OB2a segment");
        Q1Ob2aSeal {
            encoded,
            field_stats,
        }
    }

    fn q1_ob2a_owned_index(
        encoded_segments: Vec<EncodedSegment>,
        field_stats: Vec<ManifestFieldStats>,
    ) -> QuillIndex {
        let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("Q1-OB2a genesis");
        let mut manifest = genesis.next_manifest().expect("Q1-OB2a manifest");
        manifest.docid_high_watermark = u64::from(DOC_ORDS_PER_LEASE);
        manifest.segments = encoded_segments
            .iter()
            .enumerate()
            .map(|(index, encoded)| {
                manifest_segment(
                    encoded,
                    u64::try_from(index + 1).expect("Q1-OB2a seal sequence fits u64"),
                )
            })
            .collect();
        manifest.field_stats = field_stats;
        let snapshot = genesis
            .publish_owned_segments(&manifest, encoded_segments)
            .expect("publish Q1-OB2a fixture");
        QuillIndex::from_backend(
            IndexBackend::Memory(snapshot),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("bind Q1-OB2a fixture index")
    }

    fn q1_ob2a_fixture_indexes() -> (QuillIndex, QuillIndex) {
        let documents = q1_ob2a_documents();
        let left = seal_q1_ob2a_documents(&documents[..100], 0, 0x0b2a_0001);
        let right = seal_q1_ob2a_documents(&documents[100..], 100, 0x0b2a_0002);
        let monolithic = seal_q1_ob2a_documents(&documents, 0, 0x0b2a_0003);

        let first_segment_stats =
            merge_field_stats(&[], &left.field_stats).expect("left Q1-OB2a stats");
        let combined_segment_stats = merge_field_stats(&first_segment_stats, &right.field_stats)
            .expect("aggregate leaf stats");
        let monolithic_stats =
            merge_field_stats(&[], &monolithic.field_stats).expect("monolithic Q1-OB2a stats");
        assert_eq!(combined_segment_stats, monolithic_stats);

        let leaves = q1_ob2a_owned_index(vec![left.encoded, right.encoded], combined_segment_stats);
        let monolithic = q1_ob2a_owned_index(vec![monolithic.encoded], monolithic_stats);
        (leaves, monolithic)
    }

    fn q1_ob2a_decoded_terms(index: &QuillIndex) -> Vec<(u16, Vec<u8>, u32, Vec<Posting>)> {
        let mut decoded = BTreeMap::<(u16, Vec<u8>), (u32, Vec<Posting>)>::new();
        for segment in index.snapshot().segments() {
            let dictionary = open_dictionary(segment, DEFAULT_SCHEMA).expect("Q1-OB2a TERMDICT");
            let limit = usize::try_from(dictionary.term_count()).expect("term count fits usize");
            let terms = dictionary
                .cursor()
                .expect("Q1-OB2a term cursor")
                .collect_bounded(limit)
                .expect("materialize Q1-OB2a terms");
            for term in terms {
                let postings = open_owned_cursor(
                    segment,
                    DEFAULT_SCHEMA,
                    term.field_ord,
                    &term.term,
                    false,
                    None,
                )
                .expect("decode Q1-OB2a postings")
                .postings;
                let entry = decoded.entry((term.field_ord, term.term)).or_default();
                entry.0 = entry
                    .0
                    .checked_add(term.metadata.doc_freq)
                    .expect("Q1-OB2a aggregate df fits u32");
                entry.1.extend(postings);
            }
        }
        decoded
            .into_iter()
            .map(|((field_ord, term), (doc_freq, postings))| (field_ord, term, doc_freq, postings))
            .collect()
    }

    fn q1_ob2a_decoded_stats(segment: &RecoveredSegment) -> StatsSection {
        let expected_fields = term_field_ords(DEFAULT_SCHEMA);
        StatsSection::parse(
            required_section(segment, SectionKind::STATS).expect("Q1-OB2a STATS bytes"),
            &expected_fields,
            segment.manifest().doc_count,
        )
        .expect("decode Q1-OB2a STATS")
    }

    fn q1_ob2a_query_evidence(index: &QuillIndex, cx: &Cx) -> Vec<(QuillSearchResult, Vec<u32>)> {
        Q1_OB2A_QUERIES
            .iter()
            .map(|query| {
                let ranked = index
                    .search_paginated(cx, query, 500, 0, true)
                    .expect("Q1-OB2a ranked query");
                let docids = index
                    .collect_docids(cx, query)
                    .expect("Q1-OB2a scoreless query");
                (ranked, docids)
            })
            .collect()
    }

    fn q1_ob4_tombstoned_index(document_count: usize, deleted: &[u32]) -> QuillIndex {
        let documents = q1_ob2a_documents();
        let documents = documents
            .get(..document_count)
            .expect("Q1-OB4 fixture document count");
        let sealed = seal_q1_ob2a_documents(documents, 0, 0x0b40_0001);
        let field_stats =
            merge_field_stats(&[], &sealed.field_stats).expect("Q1-OB4 segment statistics");
        let committed = q1_ob2a_owned_index(vec![sealed.encoded], field_stats);
        let mut manifest = committed
            .snapshot()
            .next_manifest()
            .expect("Q1-OB4 tombstone MANIFEST");
        for &global_docid in deleted {
            assert!(
                committed
                    .snapshot()
                    .delete_document(&mut manifest, &format!("q1-ob2a-{global_docid:03}"),)
                    .expect("stage Q1-OB4 tombstone"),
                "Q1-OB4 document {global_docid} must be live before deletion",
            );
        }
        let tombstoned = committed
            .snapshot()
            .publish_owned_segments(&manifest, Vec::new())
            .expect("publish Q1-OB4 tombstone generation");
        QuillIndex::from_backend(
            IndexBackend::Memory(tombstoned),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("bind Q1-OB4 fixture index")
    }

    type Q1Ob4QueryEvidence = (Vec<(String, u32)>, Option<u64>, u64, Vec<u32>);

    fn q1_ob4_query_evidence(index: &QuillIndex, cx: &Cx) -> Vec<Q1Ob4QueryEvidence> {
        Q1_OB4_QUERIES
            .iter()
            .map(|query| {
                let ranked = index
                    .search_paginated(cx, query, 500, 0, true)
                    .expect("Q1-OB4 ranked query");
                let hits = ranked
                    .hits
                    .into_iter()
                    .map(|hit| (hit.document_id, hit.global_docid))
                    .collect();
                let docids = index
                    .collect_docids(cx, query)
                    .expect("Q1-OB4 scoreless query");
                (hits, ranked.total_count, ranked.doc_count, docids)
            })
            .collect()
    }

    const E6_5_QUERIES: [&str; 5] = ["old", "new", "newcomer", "alpha OR beta OR gamma", "epoch"];
    type E6_5QueryArtifact = Vec<(String, QuillSearchResult, Vec<u32>)>;

    fn e6_5_query_artifact(index: &QuillIndex, cx: &Cx) -> E6_5QueryArtifact {
        E6_5_QUERIES
            .iter()
            .map(|query| {
                let ranked = index
                    .search_paginated(cx, query, 100, 0, true)
                    .unwrap_or_else(|error| panic!("E6.5 ranked query {query:?}: {error}"));
                let docids = index
                    .collect_docids(cx, query)
                    .unwrap_or_else(|error| panic!("E6.5 scoreless query {query:?}: {error}"));
                ((*query).to_owned(), ranked, docids)
            })
            .collect()
    }

    async fn e6_5_watch_oracle(cx: &Cx) -> Vec<E6_5QueryArtifact> {
        let index = QuillIndex::in_memory(deterministic_config()).expect("E6.5 watch oracle");
        LexicalSearch::index_documents(
            &index,
            cx,
            &[
                IndexableDocument::new("first", "old alpha epoch"),
                IndexableDocument::new("second", "old beta epoch"),
            ],
        )
        .await
        .expect("seed E6.5 watch oracle");
        LexicalSearch::commit(&index, cx)
            .await
            .expect("publish E6.5 watch oracle");
        let mut artifacts = vec![e6_5_query_artifact(&index, cx)];

        LexicalSearch::index_documents(
            &index,
            cx,
            &[
                IndexableDocument::new("first", "new alpha epoch"),
                IndexableDocument::new("second", "new beta epoch"),
            ],
        )
        .await
        .expect("replace E6.5 watch oracle batch");
        artifacts.push(e6_5_query_artifact(&index, cx));

        assert!(
            index
                .delete_document(cx, "second")
                .await
                .expect("delete E6.5 watch oracle row")
        );
        artifacts.push(e6_5_query_artifact(&index, cx));

        LexicalSearch::index_document(
            &index,
            cx,
            &IndexableDocument::new("third", "newcomer gamma epoch"),
        )
        .await
        .expect("stage E6.5 watch oracle newcomer");
        LexicalSearch::commit(&index, cx)
            .await
            .expect("publish E6.5 watch oracle newcomer");
        artifacts.push(e6_5_query_artifact(&index, cx));
        artifacts
    }

    fn committed_segment_ids(index: &QuillIndex) -> Vec<u64> {
        index
            .snapshot()
            .segments()
            .iter()
            .map(|segment| segment.manifest().segment_id)
            .collect()
    }

    fn fresh_merge_segment_id(index: &QuillIndex, seed: u64) -> u64 {
        let mut candidate = seed;
        loop {
            if index
                .snapshot()
                .segments()
                .iter()
                .all(|segment| segment.manifest().segment_id != candidate)
            {
                return candidate;
            }
            candidate = candidate
                .checked_add(1)
                .expect("concat-merge test segment-id space exhausted");
        }
    }

    fn concat_merge_query_results(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
        CONCAT_MERGE_QUERIES
            .iter()
            .map(|query| {
                index
                    .search_paginated(cx, query, 10, 0, true)
                    .expect("concat-merge fixture query")
            })
            .collect()
    }

    // ==== Segment-parallel query fan-out (bd-quill-e4-argus-3ycz.9) ====

    const SEGMENT_FANOUT_QUERIES: [&str; 5] = [
        "shared",
        "alpha",
        "alpha OR beta OR rare",
        "\"shared shared\"",
        "zeta OR argus",
    ];

    /// Run the sealed-segment collection stage on one explicit path and
    /// return the finished page.
    fn collect_sealed_page(
        index: &QuillIndex,
        cx: &Cx,
        query_text: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
        fan_out: bool,
    ) -> CollectedTopDocs {
        let mut parsed = index
            .reader
            .default_parser()
            .expect("fan-out fixture has a default parser")
            .parse_lenient(query_text);
        let _ = canonicalize_query(&mut parsed.query);
        let snapshot = index.search_snapshot();
        let rank_pruning =
            !exact_count && limit != 0 && query_has_prunable_root_union(&parsed.query, 1.0);
        let topdocs_root = !exact_count && limit != 0;
        let mut collector = if exact_count {
            TopDocsCollector::with_exact_count(limit, offset)
        } else {
            TopDocsCollector::new(limit, offset)
        }
        .expect("ranked collector");
        index
            .collect_sealed_segments(
                cx,
                &mut collector,
                &parsed.query,
                &snapshot,
                rank_pruning,
                topdocs_root,
                fan_out,
            )
            .expect("sealed collection");
        collector.finish().expect("finish sealed collection")
    }

    /// Bit-exact comparison key: global docid plus raw f32 score bits.
    fn ranked_page_key(collected: &CollectedTopDocs) -> (Vec<(u32, u32)>, Option<u64>) {
        (
            collected
                .hits
                .iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect(),
            collected.total_count,
        )
    }

    /// Seeded multi-segment fixture with heavy cross-segment term overlap and
    /// score ties so fan-out ordering mistakes would surface.
    async fn segment_fanout_fixture_index(
        cx: &Cx,
        segments: usize,
        docs_per_segment: usize,
        seed: u64,
    ) -> QuillIndex {
        const VOCABULARY: [&str; 10] = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "shared", "rare", "quill",
            "argus",
        ];
        let tier_fanout = segments
            .checked_add(1)
            .expect("fan-out fixture segment count leaves room for its tier policy");
        let index = QuillIndex::in_memory(QuillConfig {
            deterministic_ingest: true,
            tier_fanout,
            ..QuillConfig::default()
        })
        .expect("memory index");
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for segment in 0..segments {
            let mut batch = Vec::with_capacity(docs_per_segment);
            for ordinal in 0..docs_per_segment {
                let word_count = 3 + usize::try_from(next() % 10).expect("small word count");
                let mut text = String::new();
                for position in 0..word_count {
                    if position != 0 {
                        text.push(' ');
                    }
                    let pick = usize::try_from(next() % 10).expect("vocabulary pick");
                    text.push_str(VOCABULARY[pick]);
                }
                batch.push(IndexableDocument::new(
                    format!("fanout-s{segment:02}-d{ordinal:04}"),
                    text,
                ));
            }
            index
                .index_documents(cx, &batch)
                .await
                .expect("accumulate fan-out fixture batch");
            index
                .commit(cx)
                .await
                .expect("seal fan-out fixture segment");
        }
        assert_eq!(index.snapshot().segments().len(), segments);
        index
    }

    /// Run the sealed-segment unscored id-set stage on one explicit path.
    fn collect_docid_set(index: &QuillIndex, cx: &Cx, query_text: &str, fan_out: bool) -> Vec<u32> {
        let mut parsed = index
            .reader
            .default_parser()
            .expect("fan-out fixture has a default parser")
            .parse_lenient(query_text);
        let _ = canonicalize_query(&mut parsed.query);
        let snapshot = index.search_snapshot();
        let mut collector = DocSetCollector::new();
        let work_upper_bound = query_work_upper_bound(
            &parsed.query,
            &snapshot,
            index.reader.config.glob_expansion_limit,
        )
        .expect("bound sealed docid work");
        let checkpoint = QueryCheckpoint::new(
            cx,
            "collect_docids_test",
            index.reader.config.query_fuel_budget,
            work_upper_bound,
        );
        let metering = checkpoint.metering();
        let checkpoint: QueryCheckpointHandle<'_> = checkpoint;
        index
            .reader
            .collect_docids_sealed(
                cx,
                &checkpoint,
                &mut collector,
                &parsed.query,
                &snapshot,
                fan_out && !metering,
            )
            .expect("sealed docid collection");
        collector.finish()
    }

    #[test]
    fn fanned_docid_collection_matches_serial_exactly() {
        run_with_cx(|cx| async move {
            let concat = concat_merge_fixture_index(&cx).await;
            let seeded = segment_fanout_fixture_index(&cx, 5, 64, 0x0d0c_1d5e_7c01_1ec7).await;
            let fixtures = [
                (&concat, CONCAT_MERGE_QUERIES.as_slice()),
                (&seeded, SEGMENT_FANOUT_QUERIES.as_slice()),
            ];
            for (index, queries) in fixtures {
                for query in queries {
                    let serial = collect_docid_set(index, &cx, query, false);
                    for round in 0..3 {
                        assert_eq!(
                            collect_docid_set(index, &cx, query, true),
                            serial,
                            "docid fan-out diverged from serial: query={query} round={round}",
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn sealed_segment_fanout_gate_thresholds() {
        assert!(!sealed_segment_fanout(0, 0));
        assert!(
            !sealed_segment_fanout(1, SEGMENT_FANOUT_THRESHOLD),
            "single-segment snapshots must never fan out",
        );
        assert!(
            !sealed_segment_fanout(4, SEGMENT_FANOUT_THRESHOLD - 1),
            "small unfragmented corpora stay on the serial path",
        );
        assert!(sealed_segment_fanout(2, SEGMENT_FANOUT_THRESHOLD));
        assert!(sealed_segment_fanout(
            SEGMENT_COUNT_FANOUT_THRESHOLD,
            SEGMENT_FANOUT_THRESHOLD - 1,
        ));
        assert!(!sealed_segment_fanout(
            SEGMENT_COUNT_FANOUT_THRESHOLD - 1,
            SEGMENT_FANOUT_THRESHOLD - 1,
        ));
        assert!(sealed_segment_fanout(16, u64::MAX));
    }

    #[cfg(feature = "pruning-conformance")]
    #[test]
    fn pruning_conformance_traced_search_bypasses_warm_cache_without_mutation() {
        run_with_cx(|cx| async move {
            const QUERY: &str = "shared OR alpha";
            const LIMIT: usize = 20;
            const OFFSET: usize = 0;
            const EXACT_COUNT: bool = false;
            let fixtures = [
                (
                    1,
                    0x5e21_a11c_0000_0001,
                    ConformancePruningExecutionMode::Serial,
                ),
                (
                    SEGMENT_COUNT_FANOUT_THRESHOLD,
                    0x5e21_a11c_0000_0002,
                    ConformancePruningExecutionMode::Rayon,
                ),
            ];

            for (segment_count, seed, expected_mode) in fixtures {
                let index = segment_fanout_fixture_index(&cx, segment_count, 64, seed).await;
                let snapshot = index.search_snapshot();
                let snapshot_epoch = snapshot.snapshot_epoch();
                let expected_doc_counts = index
                    .snapshot()
                    .segments()
                    .iter()
                    .map(|segment| u64::from(segment.doc_count()))
                    .collect::<Vec<_>>();
                let cache = &index.reader.published_snapshot.ranked_query_cache;

                assert!(
                    cache
                        .get_raw(snapshot_epoch, QUERY, LIMIT, OFFSET, EXACT_COUNT)
                        .is_none(),
                    "fixture must begin cold for {expected_mode:?}",
                );
                let warm = index
                    .search_paginated(&cx, QUERY, LIMIT, OFFSET, EXACT_COUNT)
                    .expect("ordinary search warms the ranked query cache");
                assert_eq!(
                    cache.get_raw(snapshot_epoch, QUERY, LIMIT, OFFSET, EXACT_COUNT),
                    Some(warm.clone()),
                    "ordinary search must populate the exact raw-query cache key for {expected_mode:?}",
                );
                let slots_before = cache
                    .slots
                    .iter()
                    .map(ArcSwapOption::load_full)
                    .collect::<Vec<_>>();

                let (traced, receipt) = index
                    .search_paginated_with_conformance_pruning_trace(
                        &cx,
                        QUERY,
                        LIMIT,
                        OFFSET,
                        EXACT_COUNT,
                    )
                    .expect("traced search must execute instead of returning from the warm cache");
                assert_eq!(traced, warm);
                assert_eq!(receipt.execution_mode(), expected_mode);
                assert_eq!(receipt.segments().len(), expected_doc_counts.len());
                assert_eq!(
                    receipt
                        .segments()
                        .iter()
                        .map(ConformanceSegmentPruningReceipt::segment_ordinal)
                        .collect::<Vec<_>>(),
                    (0..u64::try_from(expected_doc_counts.len())
                        .expect("fixture segment count fits u64"))
                        .collect::<Vec<_>>(),
                );
                assert_eq!(
                    receipt
                        .segments()
                        .iter()
                        .map(ConformanceSegmentPruningReceipt::segment_doc_count)
                        .collect::<Vec<_>>(),
                    expected_doc_counts,
                );

                let slots_after = cache
                    .slots
                    .iter()
                    .map(ArcSwapOption::load_full)
                    .collect::<Vec<_>>();
                assert_eq!(slots_after.len(), slots_before.len());
                for (slot, (before, after)) in slots_before.iter().zip(&slots_after).enumerate() {
                    match (before, after) {
                        (Some(before), Some(after)) => assert!(
                            Arc::ptr_eq(before, after),
                            "traced search replaced cache slot {slot} for {expected_mode:?}",
                        ),
                        (None, None) => {}
                        _ => panic!(
                            "traced search changed cache slot occupancy at {slot} for {expected_mode:?}",
                        ),
                    }
                }
            }
        });
    }

    #[cfg(feature = "pruning-conformance")]
    #[test]
    fn pruning_conformance_invocation_bound_traces_do_not_cross_concurrent_searches() {
        run_with_cx(|cx| async move {
            let index =
                segment_fanout_fixture_index(&cx, SEGMENT_COUNT_FANOUT_THRESHOLD, 64, 0x51A7).await;
            let expected_doc_counts = index
                .snapshot()
                .segments()
                .iter()
                .map(|segment| u64::from(segment.doc_count()))
                .collect::<Vec<_>>();
            let query = "shared OR alpha";
            let (first, (second, ordinary)) = rayon::join(
                || {
                    index
                        .search_paginated_with_conformance_pruning_trace(&cx, query, 20, 0, false)
                        .expect("first concurrent traced search")
                },
                || {
                    rayon::join(
                        || {
                            index
                                .search_paginated_with_conformance_pruning_trace(
                                    &cx, query, 20, 0, false,
                                )
                                .expect("second concurrent traced search")
                        },
                        || {
                            index
                                .search_paginated(&cx, query, 20, 0, false)
                                .expect("concurrent ordinary search")
                        },
                    )
                },
            );

            assert_eq!(first.0, second.0);
            assert_eq!(first.0, ordinary);
            for receipt in [&first.1, &second.1] {
                assert_eq!(
                    receipt.execution_mode(),
                    ConformancePruningExecutionMode::Rayon,
                );
                assert_eq!(receipt.segments().len(), expected_doc_counts.len());
                assert_eq!(
                    receipt
                        .segments()
                        .iter()
                        .map(ConformanceSegmentPruningReceipt::segment_ordinal)
                        .collect::<Vec<_>>(),
                    (0..u64::try_from(expected_doc_counts.len())
                        .expect("fan-out segment count fits u64"))
                        .collect::<Vec<_>>(),
                );
                assert_eq!(
                    receipt
                        .segments()
                        .iter()
                        .map(ConformanceSegmentPruningReceipt::segment_doc_count)
                        .collect::<Vec<_>>(),
                    expected_doc_counts,
                );
            }
        });
    }

    #[cfg(feature = "pruning-conformance")]
    #[test]
    fn pruning_conformance_cancellation_forces_high_fanout_trace_to_serial() {
        run_with_cx(|cx| async move {
            let index =
                segment_fanout_fixture_index(&cx, SEGMENT_COUNT_FANOUT_THRESHOLD, 64, 0x5A1E).await;
            let controller = index.conformance_cancellation_controller();
            controller
                .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
                .expect("arm high-fanout pruning cancellation");

            let error = index
                .search_paginated_with_conformance_pruning_trace(
                    &cx,
                    "shared OR alpha",
                    20,
                    0,
                    false,
                )
                .expect_err("armed high-fanout trace must cancel after its first receipt");
            assert!(matches!(
                error,
                QuillIndexError::Cancelled { phase: "search" }
            ));
            assert_eq!(controller.observed_checkpoints(), 1);
            assert_eq!(controller.recorded_pruning_receipts_at_fire(), 1);
            assert_eq!(
                controller.pruning_trace_execution_mode_at_fire(),
                Some(ConformancePruningExecutionMode::Serial),
                "the injected invocation must replace schedule-dependent Rayon receipt order with serial segment order",
            );
            assert_eq!(controller.discarded_pruning_trace_sessions(), 1);

            controller.disarm();
            cx.set_cancel_requested(false);
            let (_, receipt) = index
                .search_paginated_with_conformance_pruning_trace(
                    &cx,
                    "shared OR alpha",
                    20,
                    0,
                    false,
                )
                .expect("unarmed high-fanout replay");
            assert_eq!(
                receipt.execution_mode(),
                ConformancePruningExecutionMode::Rayon,
                "ordinary traced high-fanout searches must retain the shipping Rayon branch",
            );
        });
    }

    #[cfg(feature = "pruning-conformance")]
    #[test]
    fn pruning_conformance_stale_session_cannot_checkpoint_rearmed_controller() {
        run_with_cx(|cx| async move {
            let controller = ConformanceCancellationController::default();
            controller
                .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
                .expect("arm stale-session generation");
            let stale_generation = controller
                .active_pruning_trace_arm_generation()
                .expect("capture stale pruning generation");
            let stale_session =
                ConformancePruningTraceSession::new_for_cancellation_arm(1, Some(stale_generation));
            stale_session
                .bind_execution_mode(ConformancePruningExecutionMode::Serial)
                .expect("bind stale session");

            controller.disarm();
            controller
                .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
                .expect("arm replacement generation");
            assert_eq!(
                stale_session
                    .record_and_checkpoint(0, 1, &[], &controller, &cx)
                    .expect("record stale receipt without crossing generations"),
                1,
            );
            assert_eq!(controller.observed_checkpoints(), 0);
            assert_eq!(controller.recorded_pruning_receipts_at_fire(), 0);
            assert_eq!(controller.pruning_trace_execution_mode_at_fire(), None);
            assert!(!controller.fired());
            assert!(!cx.is_cancel_requested());
            assert_eq!(
                stale_session
                    .complete()
                    .expect("complete generation-isolated stale session")
                    .execution_mode(),
                ConformancePruningExecutionMode::Serial,
            );
        });
    }

    #[test]
    fn fanned_sealed_collection_matches_serial_exactly() {
        run_with_cx(|cx| async move {
            let concat = concat_merge_fixture_index(&cx).await;
            let seeded = segment_fanout_fixture_index(&cx, 5, 64, 0x9e37_79b9_7f4a_7c15).await;
            let pages = [(10_usize, 0_usize), (3, 4), (1, 0), (0, 0)];
            let fixtures = [
                (&concat, CONCAT_MERGE_QUERIES.as_slice()),
                (&seeded, SEGMENT_FANOUT_QUERIES.as_slice()),
            ];
            for (index, queries) in fixtures {
                for query in queries {
                    for (limit, offset) in pages {
                        for exact_count in [false, true] {
                            let serial = ranked_page_key(&collect_sealed_page(
                                index,
                                &cx,
                                query,
                                limit,
                                offset,
                                exact_count,
                                false,
                            ));
                            // Repeat the fanned arm: rayon schedules differ
                            // run to run, and every schedule must reproduce
                            // the serial page bit-exactly.
                            for round in 0..3 {
                                let fanned = ranked_page_key(&collect_sealed_page(
                                    index,
                                    &cx,
                                    query,
                                    limit,
                                    offset,
                                    exact_count,
                                    true,
                                ));
                                assert_eq!(
                                    fanned, serial,
                                    "fan-out diverged from serial: query={query} \
                                     limit={limit} offset={offset} \
                                     exact_count={exact_count} round={round}",
                                );
                            }
                        }
                    }
                }
            }
        });
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

    #[test]
    fn snapshot_publication_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<DeltaSnapshot>();
        assert_send_sync::<QuillSearchSnapshot>();
        assert_send_sync::<SnapshotPublisher>();
        assert_send_sync::<QuillIndex>();
    }

    #[test]
    fn repeated_doc_frequency_cursor_and_search_calls_reuse_validated_termdict_metadata() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("first", "alpha shared"),
                        IndexableDocument::new("second", "beta shared"),
                        IndexableDocument::new("third", "gamma"),
                    ],
                )
                .await
                .expect("index cache fixture");
            index.commit(&cx).await.expect("seal cache fixture");

            let snapshot = index.snapshot();
            assert_eq!(snapshot.segments().len(), 1);
            let segment = &snapshot.segments()[0];
            let (full_validations_before, borrowed_views_before) =
                segment.term_dictionary_cache_counts();
            assert_eq!(full_validations_before, 1);

            assert_eq!(
                snapshot_doc_freq(&snapshot, DEFAULT_SCHEMA, CONTENT_FIELD, b"shared")
                    .expect("cached doc frequency"),
                2
            );
            let term_count = usize::try_from(
                segment
                    .term_dictionary(DEFAULT_SCHEMA)
                    .expect("cached cursor dictionary")
                    .term_count(),
            )
            .expect("fixture term count fits usize");
            let cursor_rows = segment
                .term_dictionary(DEFAULT_SCHEMA)
                .expect("cached cursor dictionary")
                .cursor()
                .and_then(|cursor| cursor.collect_bounded(term_count))
                .expect("collect cached cursor");
            assert!(!cursor_rows.is_empty());

            for _ in 0..32 {
                let hits = index
                    .search_doc_ids(&cx, "shared", 10)
                    .expect("repeat cached search");
                assert_eq!(hits.len(), 2);
            }
            let (full_validations_after, borrowed_views_after) =
                segment.term_dictionary_cache_counts();
            assert_eq!(
                full_validations_after, 1,
                "no query path may repeat complete TERMDICT validation"
            );
            assert_eq!(
                borrowed_views_after,
                borrowed_views_before.saturating_add(7),
                "doc-frequency, two cursor opens, and four field-specific views for the first search must be borrowed; the remaining 31 searches must use the immutable-snapshot result cache"
            );
        });
    }

    #[test]
    fn quill_lexical_contract_is_immediate_hydratable_cancel_safe_and_upserts() {
        run_with_cx(|cx| async move {
            let index =
                QuillIndex::in_memory(deterministic_config()).expect("create lexical backend");
            assert!(index.path().is_none());

            let original = IndexableDocument::new("quill-doc", "native quill ownership backend")
                .with_metadata("path", "src/lib.rs")
                .with_metadata("lang", "rust");
            LexicalSearch::index_document(&index, &cx, &original)
                .await
                .expect("index through lexical trait");
            LexicalSearch::commit(&index, &cx)
                .await
                .expect("commit through lexical trait");
            assert_eq!(LexicalSearch::doc_count(&index), 1);

            let full = LexicalSearch::search(&index, &cx, "ownership", 10)
                .await
                .expect("full lexical search");
            assert_eq!(full.len(), 1);
            let expected_metadata = serde_json::json!({
                "lang": "rust",
                "path": "src/lib.rs",
            });
            assert_eq!(full[0].metadata.as_deref(), Some(&expected_metadata));

            let mut candidate_future =
                LexicalSearch::search_fusion_candidates(&index, &cx, "ownership", 10);
            let candidates = {
                let waker = Waker::noop();
                let mut task_cx = Context::from_waker(waker);
                match candidate_future.as_mut().poll(&mut task_cx) {
                    Poll::Ready(result) => result.expect("fusion candidates"),
                    Poll::Pending => panic!("Quill fusion candidates must be first-poll ready"),
                }
            };
            drop(candidate_future);
            assert!(index.fusion_metadata_is_deferred());
            assert_eq!(candidates.len(), full.len());
            assert_eq!(candidates[0].doc_id, full[0].doc_id);
            assert_eq!(candidates[0].score.to_bits(), full[0].score.to_bits());
            assert!(candidates[0].metadata.is_none());

            let mut hydrated = candidates;
            LexicalSearch::hydrate_fusion_metadata(&index, &cx, &mut hydrated)
                .await
                .expect("hydrate fusion winners");
            assert_eq!(hydrated[0].metadata, full[0].metadata);

            let ids = index
                .search_doc_ids(&cx, "ownership", 10)
                .expect("identifier-only public search");
            assert_eq!(ids[0].document_id, "quill-doc");
            let enriched = index
                .search_with_snippets(&cx, "ownership", 10, &crate::SnippetConfig::default())
                .expect("enriched public search");
            assert_eq!(enriched[0].metadata.as_deref(), Some(&expected_metadata));
            assert!(
                enriched[0]
                    .snippet
                    .as_deref()
                    .is_some_and(|snippet| snippet.contains("<b>ownership</b>"))
            );

            let replacement = IndexableDocument::new("quill-doc", "python replacement backend")
                .with_metadata("lang", "python");
            LexicalSearch::index_document(&index, &cx, &replacement)
                .await
                .expect("upsert replacement through lexical trait");
            assert!(
                LexicalSearch::search(&index, &cx, "ownership", 10)
                    .await
                    .expect("search removed content before explicit commit")
                    .is_empty()
            );
            assert_eq!(
                LexicalSearch::search(&index, &cx, "replacement", 10)
                    .await
                    .expect("search replacement before explicit commit")
                    .len(),
                1
            );
            LexicalSearch::commit(&index, &cx)
                .await
                .expect("no-op commit after atomic replacement");
            assert!(
                LexicalSearch::search(&index, &cx, "ownership", 10)
                    .await
                    .expect("search removed content")
                    .is_empty()
            );
            let replacement_hits = LexicalSearch::search(&index, &cx, "replacement", 10)
                .await
                .expect("search replacement content");
            assert_eq!(replacement_hits.len(), 1);
            assert_eq!(
                replacement_hits[0].metadata.as_deref(),
                Some(&serde_json::json!({"lang": "python"}))
            );
            assert_eq!(index.segment_stats().live_docs, 1);

            assert!(
                index
                    .delete_document(&cx, "quill-doc")
                    .await
                    .expect("delete live document")
            );
            assert!(
                !index
                    .delete_document(&cx, "quill-doc")
                    .await
                    .expect("repeat delete")
            );
            assert_eq!(index.doc_count(), 0);

            LexicalSearch::index_documents(
                &index,
                &cx,
                &[
                    IndexableDocument::new("first", "alpha"),
                    IndexableDocument::new("second", "beta"),
                    IndexableDocument::new("third", "gamma"),
                ],
            )
            .await
            .expect("repopulate through lexical trait");
            LexicalSearch::commit(&index, &cx)
                .await
                .expect("commit repopulated backend");
            assert_eq!(
                index
                    .delete_documents(&cx, &["first", "missing", "second"])
                    .await
                    .expect("delete bounded document batch"),
                2
            );
            assert_eq!(index.doc_count(), 1);
            assert_eq!(
                LexicalSearch::search(&index, &cx, "gamma", 10)
                    .await
                    .expect("search batch-delete survivor")
                    .len(),
                1
            );
            index.delete_all(&cx).await.expect("delete all documents");
            assert_eq!(index.doc_count(), 0);

            let cancelled = cx.clone();
            cancelled.set_cancel_requested(true);
            assert!(matches!(
                LexicalSearch::search(&index, &cancelled, "alpha", 10).await,
                Err(SearchError::Cancelled { ref phase, .. }) if phase == "search"
            ));
            assert!(matches!(
                LexicalSearch::index_document(
                    &index,
                    &cancelled,
                    &IndexableDocument::new("cancelled", "never indexed"),
                )
                .await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(index.doc_count(), 0);
        });
    }

    #[test]
    fn lexical_trait_batch_upsert_publishes_one_atomic_generation() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let original = [
                IndexableDocument::new("first", "old alpha"),
                IndexableDocument::new("second", "old beta"),
                IndexableDocument::new("third", "old gamma"),
            ];
            LexicalSearch::index_documents(&index, &cx, &original)
                .await
                .expect("seed lexical documents");
            LexicalSearch::commit(&index, &cx)
                .await
                .expect("publish seed generation");
            let seed_generation = index.segment_stats().published_generation;

            let replacements = [
                IndexableDocument::new("first", "new alpha"),
                IndexableDocument::new("second", "new beta"),
                IndexableDocument::new("third", "new gamma"),
            ];
            LexicalSearch::index_documents(&index, &cx, &replacements)
                .await
                .expect("publish replacement batch");
            assert_eq!(
                index.segment_stats().published_generation,
                seed_generation.saturating_add(1),
                "one upsert batch must publish tombstones and replacements together"
            );
            assert!(!index.has_uncommitted_changes());
            assert_eq!(
                index
                    .search_doc_ids(&cx, "new", 10)
                    .expect("replacement batch is immediately searchable")
                    .len(),
                3
            );

            LexicalSearch::commit(&index, &cx)
                .await
                .expect("commit after replacement is a no-op");
            assert_eq!(
                index.segment_stats().published_generation,
                seed_generation.saturating_add(1)
            );
            assert_eq!(index.doc_count(), 3);
        });
    }

    #[test]
    fn lexical_trait_disjoint_batches_accumulate_before_commit() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            LexicalSearch::index_documents(
                &index,
                &cx,
                &[
                    IndexableDocument::new("first", "alpha"),
                    IndexableDocument::new("second", "beta"),
                ],
            )
            .await
            .expect("stage first disjoint batch");
            LexicalSearch::index_documents(
                &index,
                &cx,
                &[
                    IndexableDocument::new("third", "gamma"),
                    IndexableDocument::new("fourth", "delta"),
                ],
            )
            .await
            .expect("stage second disjoint batch");

            LexicalSearch::commit(&index, &cx)
                .await
                .expect("publish both disjoint batches");
            assert_eq!(index.doc_count(), 4);
        });
    }

    #[test]
    fn read_only_search_handle_coexists_with_writer_and_pins_publication() {
        run_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("index directory");
            let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable writer");
            LexicalSearch::index_document(
                &writer,
                &cx,
                &IndexableDocument::new("first", "published alpha"),
            )
            .await
            .expect("stage first document");
            LexicalSearch::commit(&writer, &cx)
                .await
                .expect("publish first document");

            let pinned = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
                .await
                .expect("open read-only handle while writer lease is live");
            assert_eq!(pinned.doc_count(), 1);
            assert_eq!(
                pinned
                    .search_doc_ids(&cx, "alpha", 10)
                    .expect("search pinned publication")[0]
                    .document_id,
                "first"
            );

            LexicalSearch::index_document(
                &writer,
                &cx,
                &IndexableDocument::new("second", "published beta"),
            )
            .await
            .expect("stage second document");
            LexicalSearch::commit(&writer, &cx)
                .await
                .expect("publish second document");

            assert!(
                pinned
                    .search_doc_ids(&cx, "beta", 10)
                    .expect("old handle remains pinned")
                    .is_empty()
            );
            assert!(pinned.refresh(&cx).await.expect("refresh publication"));
            assert_eq!(pinned.doc_count(), 2);
            assert_eq!(
                pinned
                    .search_doc_ids(&cx, "beta", 10)
                    .expect("refreshed handle observes successor")[0]
                    .document_id,
                "second"
            );
            assert!(!pinned.refresh(&cx).await.expect("idempotent refresh"));
            assert!(pinned.segment_stats().live_writer);
        });
    }

    #[test]
    fn composite_snapshot_uses_keeper_at_seal_and_live_delta_statistics() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let sealed_documents = [
                IndexableDocument::new("sealed-live", "alpha"),
                IndexableDocument::new("sealed-deleted", "alpha"),
            ];
            index
                .index_documents(&cx, &sealed_documents)
                .await
                .expect("accumulate sealed fixtures");
            index.commit(&cx).await.expect("commit sealed fixtures");

            let committed = index.snapshot().clone();
            let mut tombstoned_manifest = committed.next_manifest().expect("next manifest");
            assert!(
                committed
                    .delete_document(&mut tombstoned_manifest, "sealed-deleted")
                    .expect("stage sealed delete")
            );
            let tombstoned = committed
                .publish_owned_segments(&tombstoned_manifest, Vec::new())
                .expect("publish tombstone-only successor");
            assert_eq!(tombstoned.at_seal_doc_count(), 2);
            assert_eq!(tombstoned.doc_count(), 1);
            assert_eq!(
                snapshot_doc_freq(&tombstoned, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                    .expect("sealed alpha df"),
                2,
                "sealed tombstones remain in BM25 df until compaction"
            );

            let generation = tombstoned.loaded_manifest().manifest.generation;
            let first_base = u64::from(DOC_ORDS_PER_LEASE);
            let second_base = first_base * 2;
            let mut first = DeltaSegment::new(DEFAULT_SCHEMA, first_base, usize::MAX)
                .expect("first Delta shard");
            apply_alpha_delta(
                &mut first,
                u32::try_from(first_base).expect("first Delta docid"),
                "delta-a",
                2,
            );
            let mut second = DeltaSegment::new(DEFAULT_SCHEMA, second_base, usize::MAX)
                .expect("second Delta shard");
            apply_alpha_delta(
                &mut second,
                u32::try_from(second_base).expect("second Delta docid"),
                "delta-b",
                3,
            );

            let snapshot = QuillSearchSnapshot::compose(
                11,
                Arc::new(tombstoned),
                vec![
                    Arc::new(first.freeze(generation)),
                    Arc::new(second.freeze(generation)),
                ],
            )
            .expect("compose Keeper plus Delta snapshot");
            assert_eq!(snapshot.snapshot_epoch(), 11);
            assert_eq!(snapshot.bm25_doc_count(), 4);
            assert_eq!(snapshot.live_doc_count(), 3);
            assert_eq!(
                snapshot
                    .bm25_field_stats(CONTENT_FIELD)
                    .expect("content stats"),
                SnapshotFieldStats {
                    field_ord: CONTENT_FIELD,
                    total_tokens: 7,
                    doc_count: 4,
                }
            );
            assert_eq!(
                snapshot
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .expect("composite alpha df"),
                4
            );
        });
    }

    #[test]
    fn three_delta_upserts_contribute_one_live_bm25_row() {
        run_with_cx(|cx| async move {
            let keeper =
                Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
            let generation = keeper.loaded_manifest().manifest.generation;
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
            apply_alpha_delta(&mut delta, 0, "same-id", 1);
            apply_alpha_delta(&mut delta, 1, "same-id", 2);
            apply_alpha_delta(&mut delta, 2, "same-id", 3);
            assert_eq!(delta.physical_document_count(), 3);
            assert_eq!(delta.live_document_count(), 1);
            assert_eq!(
                delta
                    .find_term(CONTENT_FIELD, b"alpha")
                    .expect("alpha Delta term")
                    .live_doc_freq(),
                1
            );

            let frozen = Arc::new(delta.freeze(generation));
            let snapshot = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
                .expect("compose repeated-upsert snapshot");
            assert_eq!(alpha_snapshot_tuple(&snapshot), (0, 1, 1, 3, 1));
            let pre_stats = snapshot
                .bm25_field_stats(CONTENT_FIELD)
                .expect("pre-seal content stats");
            let pre_df = snapshot
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("pre-seal alpha df");
            let mut pre_scorer = lower_delta_term(&frozen, &snapshot, CONTENT_FIELD, b"alpha", 1.0)
                .expect("pre-seal Delta term scorer");
            assert_eq!(pre_scorer.doc(), Some(2));

            let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
            sealed
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("same-id", "alpha alpha alpha")],
                )
                .await
                .expect("accumulate final logical upsert row");
            sealed
                .commit(&cx)
                .await
                .expect("seal final logical upsert row");
            let keeper = sealed.snapshot();
            let post_stats = snapshot_field(&keeper, CONTENT_FIELD).expect("post-seal stats");
            let post_df = snapshot_doc_freq(&keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                .expect("post-seal alpha df");
            assert_eq!(pre_stats, post_stats);
            assert_eq!(pre_df, post_df);

            let mut post_scorer = lower_term(
                &keeper.segments()[0],
                &keeper,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("post-seal term scorer");
            assert_eq!(
                pre_scorer.score().expect("pre-seal score").to_bits(),
                post_scorer.score().expect("post-seal score").to_bits(),
                "the final live upsert must keep bit-identical BM25 across seal"
            );
        });
    }

    #[test]
    fn delta_cursor_matches_sealed_advance_positions_and_safe_impact_bound() {
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        apply_alpha_delta(&mut delta, 0, "replaced", 300);
        apply_alpha_delta(&mut delta, 1, "kept", 2);
        apply_alpha_delta(&mut delta, 2, "replaced", 3);
        apply_alpha_delta(&mut delta, 3, "deleted", 4);
        assert_eq!(delta.delete_delta_id("deleted"), Some(3));
        let frozen = delta.freeze(0);

        let postings = [Posting::new(1, 2), Posting::new(2, 3)];
        let encoded_postings = EncodedPostingList::encode(&postings).expect("sealed postings");
        let posting_list = encoded_postings
            .posting_list()
            .expect("sealed posting list");
        let encoded_positions =
            EncodedPositionList::encode(&postings, &[0, 1, 0, 1, 2]).expect("sealed positions");
        let position_list = encoded_positions
            .position_list(&posting_list)
            .expect("sealed position list");
        let mut sealed = crate::argus::SealedPostingCursor::with_positions(&position_list, 2)
            .expect("sealed cursor");
        let mut live =
            DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("live Delta cursor");
        let live_norms = DeltaFieldNorms::new(&frozen, CONTENT_FIELD);

        assert_eq!(live.size_hint(), 2);
        assert_eq!(live.cost(), 2);
        assert_eq!(live.segment_num_docs(), 2);
        assert_eq!(live_norms.fieldnorm_id(0), None);
        assert_eq!(live_norms.fieldnorm_id(1), Some(fieldnorm_to_id(2)));
        assert_eq!(live.doc(), sealed.doc());
        assert_eq!(live.freq(), sealed.freq());
        let mut delta_positions = Vec::new();
        let mut sealed_positions = Vec::new();
        live.positions_handle()
            .expect("Delta positions")
            .decode_into(&mut delta_positions)
            .expect("decode Delta positions");
        sealed
            .positions_handle()
            .expect("sealed positions")
            .decode_into(&mut sealed_positions)
            .expect("decode sealed positions");
        assert_eq!(delta_positions, sealed_positions);

        for target in [0, 1, 2, 2] {
            assert_eq!(
                live.advance(target).expect("advance Delta cursor"),
                sealed.advance(target).expect("advance sealed cursor")
            );
            assert_eq!(live.doc(), sealed.doc());
            assert_eq!(live.freq(), sealed.freq());
        }
        live.positions_handle()
            .expect("advanced Delta positions")
            .decode_into(&mut delta_positions)
            .expect("decode advanced Delta positions");
        sealed
            .positions_handle()
            .expect("advanced sealed positions")
            .decode_into(&mut sealed_positions)
            .expect("decode advanced sealed positions");
        assert_eq!(delta_positions, sealed_positions);
        assert_eq!(
            live.next().expect("exhaust Delta cursor"),
            sealed.next().expect("exhaust sealed cursor")
        );
        assert_eq!(live.next().expect("fused Delta cursor"), None);
        assert_eq!(
            live.advance(u32::MAX)
                .expect("advance exhausted Delta cursor"),
            None
        );

        let mut missing = DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"missing")
            .expect("missing-term Delta cursor");
        assert_eq!(missing.doc(), None);
        assert_eq!(missing.freq(), None);
        assert!(missing.positions_handle().is_none());
        assert_eq!(missing.size_hint(), 0);
        assert_eq!(missing.cost(), 0);
        assert_eq!(missing.segment_num_docs(), 2);
        assert_eq!(missing.next().expect("fused missing-term cursor"), None);

        let mut all_dead =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("all-dead Delta shard");
        apply_alpha_delta(&mut all_dead, 0, "dead", 1);
        assert_eq!(all_dead.delete_delta_id("dead"), Some(0));
        let all_dead = all_dead.freeze(0);
        let mut all_dead_cursor = DeltaPostingCursor::new(&all_dead, CONTENT_FIELD, b"alpha")
            .expect("all-dead term cursor");
        assert_eq!(all_dead_cursor.doc(), None);
        assert_eq!(all_dead_cursor.size_hint(), 0);
        assert_eq!(all_dead_cursor.cost(), 0);
        assert_eq!(all_dead_cursor.next().expect("fused all-dead cursor"), None);
        assert!(
            all_dead
                .find_term(CONTENT_FIELD, b"alpha")
                .expect("physical all-dead term")
                .block_max()
                .is_none()
        );

        let impact = frozen
            .find_term(CONTENT_FIELD, b"alpha")
            .expect("alpha term")
            .block_max()
            .expect("live term impact bound");
        assert_eq!(impact.max_frequency(), u32::MAX);
        for average in [0.25_f32, 2.5, 1_000.0] {
            let bound = impact
                .score_upper_bound(average, 1.0)
                .expect("finite non-negative score bound");
            let cache = crate::contract::compute_tf_cache(average);
            for (frequency, fieldnorm_id) in
                [(2_u32, fieldnorm_to_id(2)), (3_u32, fieldnorm_to_id(3))]
            {
                let frequency = frequency as f32;
                let score = frequency / (frequency + cache[usize::from(fieldnorm_id)]);
                assert!(
                    bound >= score,
                    "avgdl={average} bound={bound} live_score={score}"
                );
            }
        }
        assert_eq!(impact.score_upper_bound(2.5, -1.0), None);
    }

    #[test]
    fn delta_advance_matches_sealed_cursor_across_multiblock_tombstone_boundaries() {
        const TOMBSTONED: [u32; 10] = [0, 1, 2, 63, 64, 127, 128, 129, 258, 259];
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        let mut sealed_postings = Vec::new();
        let mut sealed_positions = Vec::new();
        for docid in 0_u32..260 {
            let frequency = docid % 5 + 1;
            let document_id = format!("doc-{docid}");
            apply_alpha_delta(&mut delta, docid, &document_id, frequency);
            if !TOMBSTONED.contains(&docid) {
                sealed_postings.push(Posting::new(docid, frequency));
                sealed_positions.extend(0..frequency);
            }
        }
        for docid in TOMBSTONED {
            assert_eq!(delta.delete_delta_id(&format!("doc-{docid}")), Some(docid));
        }
        let frozen = delta.freeze(0);
        let encoded_postings =
            EncodedPostingList::encode(&sealed_postings).expect("sealed postings");
        let posting_list = encoded_postings
            .posting_list()
            .expect("sealed posting list");
        let encoded_positions = EncodedPositionList::encode(&sealed_postings, &sealed_positions)
            .expect("sealed positions");
        let position_list = encoded_positions
            .position_list(&posting_list)
            .expect("sealed position list");
        let segment_num_docs = u32::try_from(sealed_postings.len()).expect("fixture fits u32");

        for target in (0_u32..=260).chain(std::iter::once(u32::MAX)) {
            let mut live =
                DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("Delta cursor");
            let mut sealed =
                crate::argus::SealedPostingCursor::with_positions(&position_list, segment_num_docs)
                    .expect("sealed cursor");
            assert_eq!(
                live.advance(target).expect("advance Delta cursor"),
                sealed.advance(target).expect("advance sealed cursor"),
                "target={target}"
            );
            assert_eq!(live.freq(), sealed.freq(), "target={target}");
            if live.doc().is_some() {
                let mut live_positions = Vec::new();
                let mut oracle_positions = Vec::new();
                live.positions_handle()
                    .expect("Delta position handle")
                    .decode_into(&mut live_positions)
                    .expect("decode Delta positions");
                sealed
                    .positions_handle()
                    .expect("sealed position handle")
                    .decode_into(&mut oracle_positions)
                    .expect("decode sealed positions");
                assert_eq!(live_positions, oracle_positions, "target={target}");
            }
        }

        let mut live =
            DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha").expect("Delta cursor");
        let mut sealed =
            crate::argus::SealedPostingCursor::with_positions(&position_list, segment_num_docs)
                .expect("sealed cursor");
        loop {
            assert_eq!(live.doc(), sealed.doc());
            assert_eq!(live.freq(), sealed.freq());
            if live.doc().is_none() {
                break;
            }
            assert_eq!(
                live.next().expect("next Delta cursor"),
                sealed.next().expect("next sealed cursor")
            );
        }
    }

    #[test]
    fn snapshot_composition_rejects_stale_schema_and_lease_drift() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;

        let stale = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("stale Delta")
            .freeze(generation + 1);
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(stale)])
        else {
            panic!("stale Delta generation was accepted");
        };
        assert!(matches!(
            error,
            SnapshotError::KeeperGenerationMismatch { .. }
        ));

        let wrong_schema = DeltaSegment::new(FSFS_CHUNK_SCHEMA, 0, usize::MAX)
            .expect("wrong-schema Delta")
            .freeze(generation);
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(wrong_schema)])
        else {
            panic!("wrong-schema Delta was accepted");
        };
        assert!(matches!(error, SnapshotError::SchemaMismatch { .. }));

        let first = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("first overlapping Delta")
                .freeze(generation),
        );
        let second = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("second overlapping Delta")
                .freeze(generation),
        );
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&first), second])
        else {
            panic!("overlapping Delta leases were accepted");
        };
        assert!(matches!(
            error,
            SnapshotError::OverlappingDeltaLeases { .. }
        ));

        let publisher = SnapshotPublisher::new(keeper, vec![first]).expect("snapshot publisher");
        let replacement = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("replacement Delta")
                .freeze(generation),
        );
        assert!(matches!(
            publisher.publish_delta(1, replacement),
            Err(SnapshotError::ShardOutOfBounds {
                shard: 1,
                shard_count: 1
            })
        ));
    }

    #[test]
    fn snapshot_composition_rejects_keeper_delta_occupied_range_overlap() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &[IndexableDocument::new("sealed", "alpha")])
                .await
                .expect("accumulate sealed row");
            index.commit(&cx).await.expect("commit sealed row");

            let keeper = index.snapshot();
            let generation = keeper.loaded_manifest().manifest.generation;
            let segment = &keeper.loaded_manifest().manifest.segments[0];
            assert_eq!((segment.docid_lo, segment.docid_hi), (0, 1));

            let mut overlapping =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("overlapping Delta");
            apply_alpha_delta(&mut overlapping, 0, "duplicate-docid", 1);
            let Err(error) = QuillSearchSnapshot::compose(
                0,
                Arc::clone(&keeper),
                vec![Arc::new(overlapping.freeze(generation))],
            ) else {
                panic!("occupied Keeper/Delta docid overlap was accepted");
            };
            assert_eq!(
                error,
                SnapshotError::KeeperDeltaDocidOverlap {
                    delta_lo: 0,
                    delta_hi: 1,
                    segment_id: segment.segment_id,
                    keeper_lo: 0,
                    keeper_hi: 1,
                }
            );

            let mut continuation =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("same-lease Delta");
            apply_alpha_delta(&mut continuation, 1, "nonoverlapping-docid", 1);
            QuillSearchSnapshot::compose(
                0,
                keeper,
                vec![Arc::new(continuation.freeze(generation))],
            )
            .expect("actual same-lease continuation range does not overlap Keeper");
        });
    }

    #[test]
    fn failed_complete_publications_leave_the_current_epoch_unchanged() {
        run_with_cx(|cx| async move {
            let genesis = Arc::new(
                KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"),
            );
            let genesis_generation = genesis.loaded_manifest().manifest.generation;
            let snapshot_source = SnapshotPublisher::new(Arc::clone(&genesis), Vec::new())
                .expect("genesis publisher");

            let first = QuillIndex::in_memory(deterministic_config()).expect("first index");
            first
                .index_documents(&cx, &[IndexableDocument::new("first", "alpha")])
                .await
                .expect("accumulate first successor");
            first.commit(&cx).await.expect("publish first successor");
            let successor = first.snapshot();

            let before = snapshot_source.load();
            let stale_delta = Arc::new(
                DeltaSegment::new(DEFAULT_SCHEMA, DOC_ORDS_PER_LEASE.into(), usize::MAX)
                    .expect("stale Delta")
                    .freeze(genesis_generation),
            );
            assert!(matches!(
                snapshot_source.publish_complete(Arc::clone(&successor), vec![stale_delta]),
                Err(SnapshotError::KeeperGenerationMismatch { .. })
            ));
            assert!(Arc::ptr_eq(&before, &snapshot_source.load()));
            assert_eq!(alpha_snapshot_tuple(&before), (0, 0, 0, 0, 0));

            let accepted = snapshot_source
                .publish_complete(Arc::clone(&successor), Vec::new())
                .expect("publish valid Keeper successor");
            assert_eq!(alpha_snapshot_tuple(&accepted), (1, 1, 1, 1, 1));

            let Err(rollback) = snapshot_source.publish_complete(genesis, Vec::new()) else {
                panic!("stale Keeper rollback was accepted");
            };
            assert_eq!(
                rollback,
                SnapshotError::KeeperGenerationRegression {
                    current: successor.loaded_manifest().manifest.generation,
                    proposed: genesis_generation,
                }
            );
            assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));

            let second = QuillIndex::in_memory(deterministic_config()).expect("second index");
            second
                .index_documents(&cx, &[IndexableDocument::new("second", "beta")])
                .await
                .expect("accumulate colliding successor");
            second
                .commit(&cx)
                .await
                .expect("publish colliding successor fixture");
            let collision = second.snapshot();
            let Err(collision) = snapshot_source.publish_complete(collision, Vec::new()) else {
                panic!("same-generation divergent MANIFEST was accepted");
            };
            assert_eq!(
                collision,
                SnapshotError::KeeperGenerationCollision {
                    generation: successor.loaded_manifest().manifest.generation,
                }
            );
            assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));
        });
    }

    #[test]
    fn scalar_and_delta_writer_modes_cannot_discard_each_others_pending_state() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &[IndexableDocument::new("scalar", "sealed first")])
                .await
                .expect("accumulate scalar row");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let empty_delta = Arc::new(
                DeltaSegment::new(DEFAULT_SCHEMA, u64::from(DOC_ORDS_PER_LEASE), usize::MAX)
                    .expect("empty Delta")
                    .freeze(generation),
            );
            let Err(error) = index.publish_delta_table(vec![empty_delta]) else {
                panic!("Delta publication overtook scalar pending state");
            };
            assert!(error.to_string().contains("fully committed"));
            index.commit(&cx).await.expect("commit scalar row");

            let lease_base = index
                .snapshot()
                .loaded_manifest()
                .manifest
                .docid_high_watermark;
            let global_docid = u32::try_from(lease_base).expect("Delta docid");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, lease_base, usize::MAX).expect("live Delta");
            apply_alpha_delta(&mut delta, global_docid, "delta", 1);
            index
                .publish_delta_table(vec![Arc::new(delta.freeze(generation))])
                .expect("publish live Delta");
            let before = index.search_snapshot();

            let error = index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("forbidden", "scalar overlap")],
                )
                .await
                .expect_err("scalar ingest must reject an active Delta epoch");
            assert!(error.to_string().contains("Delta epochs are active"));
            let Err(error) = index.commit(&cx).await else {
                panic!("scalar commit accepted an active Delta epoch");
            };
            assert!(error.to_string().contains("Delta epochs are active"));

            let after = index.search_snapshot();
            assert!(Arc::ptr_eq(&before, &after));
            assert_eq!(
                after.materialize_document_id(global_docid).as_deref(),
                Some("delta")
            );
            assert_eq!(after.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(), 1);
        });
    }

    #[test]
    fn publisher_retains_a_held_epoch_until_its_last_reader_drops() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        let publisher = SnapshotPublisher::new(
            Arc::clone(&keeper),
            vec![Arc::new(delta.freeze(generation))],
        )
        .expect("snapshot publisher");
        let held = publisher.load();
        let weak = Arc::downgrade(&held);

        apply_alpha_delta(&mut delta, 0, "new", 1);
        let next_epoch = publisher
            .publish_delta(0, Arc::new(delta.freeze(generation)))
            .expect("publish next Delta epoch");
        assert_eq!(alpha_snapshot_tuple(&held), (0, 0, 0, 0, 0));
        assert_eq!(alpha_snapshot_tuple(&next_epoch), (1, 1, 1, 1, 1));
        assert_eq!(alpha_snapshot_tuple(&publisher.load()), (1, 1, 1, 1, 1));
        assert!(weak.upgrade().is_some());

        drop(held);
        assert!(
            weak.upgrade().is_none(),
            "the old composite must retire after its final reader Arc drops"
        );
    }

    #[test]
    fn labruntime_readers_observe_only_complete_composite_epochs() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        let publisher = Arc::new(
            SnapshotPublisher::new(
                Arc::clone(&keeper),
                vec![Arc::new(delta.freeze(generation))],
            )
            .expect("snapshot publisher"),
        );
        apply_alpha_delta(&mut delta, 0, "new", 1);
        let next_delta = Arc::new(delta.freeze(generation));
        let reader_saw_old = Arc::new(AtomicBool::new(false));
        let writer_done = Arc::new(AtomicBool::new(false));

        let mut lab = LabRuntime::new(LabConfig::new(0xe5_2002).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let reader_publisher = Arc::clone(&publisher);
        let reader_started = Arc::clone(&reader_saw_old);
        let reader_done = Arc::clone(&writer_done);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                assert_eq!(
                    alpha_snapshot_tuple(&reader_publisher.load()),
                    (0, 0, 0, 0, 0)
                );
                reader_started.store(true, Ordering::SeqCst);
                while !reader_done.load(Ordering::SeqCst) {
                    let observed = alpha_snapshot_tuple(&reader_publisher.load());
                    assert!(
                        observed == (0, 0, 0, 0, 0) || observed == (1, 1, 1, 1, 1),
                        "reader observed a torn composite tuple: {observed:?}"
                    );
                    yield_now().await;
                }
                assert_eq!(
                    alpha_snapshot_tuple(&reader_publisher.load()),
                    (1, 1, 1, 1, 1)
                );
            })
            .expect("create snapshot reader task");

        let writer_publisher = Arc::clone(&publisher);
        let writer_started = Arc::clone(&reader_saw_old);
        let writer_finished = Arc::clone(&writer_done);
        let (writer, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !writer_started.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                writer_publisher
                    .publish_delta(0, next_delta)
                    .expect("publish complete Delta epoch");
                writer_finished.store(true, Ordering::SeqCst);
            })
            .expect("create snapshot writer task");

        lab.scheduler.lock().schedule(reader, 0);
        lab.step_for_test();
        assert!(reader_saw_old.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(writer, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn tombstone_folded_delta_seal_is_byte_identical_to_direct_scribe_flush() {
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
        apply_sealable_delta_document(&mut delta, 0, "dead-a", "ghost", 1);
        apply_sealable_delta_document(&mut delta, 1, "live-a", "alpha", 2);
        apply_sealable_delta_document(&mut delta, 2, "dead-b", "ghost", 2);
        apply_sealable_delta_document(&mut delta, 3, "live-b", "beta", 1);
        assert_eq!(delta.delete_delta_id("dead-a"), Some(0));
        assert_eq!(delta.delete_delta_id("dead-b"), Some(2));
        let frozen = delta.freeze(1);

        let metadata = b"{}";
        let ordinal_a = 1_u64.to_le_bytes();
        let ordinal_b = 3_u64.to_le_bytes();
        let mut direct = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("direct accumulator");
        direct
            .add_document_with_values(
                1,
                &[
                    IndexedFieldValue::new(ID_FIELD, "live-a"),
                    IndexedFieldValue::new(CONTENT_FIELD, "alpha alpha"),
                    IndexedFieldValue::new(TITLE_FIELD, ""),
                ],
                &[],
                &[
                    StoredFieldValue::new(METADATA_FIELD, metadata),
                    StoredFieldValue::new(ORD_FIELD, &ordinal_a),
                ],
            )
            .expect("accumulate direct live-a");
        direct
            .add_document_with_values(
                3,
                &[
                    IndexedFieldValue::new(ID_FIELD, "live-b"),
                    IndexedFieldValue::new(CONTENT_FIELD, "beta"),
                    IndexedFieldValue::new(TITLE_FIELD, ""),
                ],
                &[],
                &[
                    StoredFieldValue::new(METADATA_FIELD, metadata),
                    StoredFieldValue::new(ORD_FIELD, &ordinal_b),
                ],
            )
            .expect("accumulate direct live-b");
        let documents = [
            FlushDocumentInput::new(1, "live-a", shipping_content_hash("live-a", "alpha alpha")),
            FlushDocumentInput::new(3, "live-b", shipping_content_hash("live-b", "beta")),
        ];
        let metadata = DeltaFlushInput {
            segment_id: 0xe5_4000,
            created_unix_s: 1_700_000_000,
            engine_version: CURRENT_ENGINE_VERSION,
        };
        let direct_encoded = flush_accumulator_with_mode(
            &direct,
            FlushSegmentInput {
                segment_id: metadata.segment_id,
                lease_docid_base: 0,
                created_unix_s: metadata.created_unix_s,
                engine_version: metadata.engine_version,
                documents: &documents,
            },
            FlushMode::Scalar,
        )
        .expect("flush direct Scribe fixture");
        let delta_encoded = flush_delta_snapshot(&frozen, metadata)
            .expect("flush tombstone-folded Delta")
            .expect("live Delta emits a segment");
        assert_eq!(delta_encoded.as_bytes(), direct_encoded.as_bytes());

        let mut all_dead =
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("all-dead Delta");
        apply_sealable_delta_document(&mut all_dead, 0, "dead-only", "ghost", 1);
        assert_eq!(all_dead.delete_delta_id("dead-only"), Some(0));
        assert!(
            flush_delta_snapshot(&all_dead.freeze(1), metadata)
                .expect("all-dead seal result")
                .is_none(),
            "an all-tombstoned Delta must not emit an empty FSLX"
        );
    }

    #[test]
    fn convenience_delta_row_is_searchable_but_not_sealable_without_content_hash() {
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
        let positions = [0];
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: CONTENT_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: TITLE_FIELD,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let postings = [
            DeltaTermPosting {
                field_ord: ID_FIELD,
                term: b"convenience",
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: CONTENT_FIELD,
                term: b"alpha",
                frequency: 1,
                positions: Some(&positions),
            },
        ];
        delta
            .apply_document(0, DocId::from("convenience"), &fieldnorms, &postings)
            .expect("apply process-local convenience row");
        let frozen = delta.freeze(1);
        assert_eq!(frozen.content_hash(0), None);
        assert_eq!(
            DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"alpha")
                .expect("query convenience row")
                .doc(),
            Some(0)
        );
        assert!(matches!(
            flush_delta_snapshot(
                &frozen,
                DeltaFlushInput {
                    segment_id: 0xe5_4007,
                    created_unix_s: 1_700_000_007,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            ),
            Err(FlushError::MissingDeltaContentHash { global_docid: 0 })
        ));
    }

    #[test]
    fn delta_seal_numeric_and_stored_sidecars_are_byte_identical_to_direct_scribe_flush() {
        let mut delta =
            DeltaSegment::new(DELTA_PARITY_SCHEMA, 0, usize::MAX).expect("Delta parity lease");
        {
            let mut add_delta =
                |global_docid: u32, document_id: &str, frequency: u32, rank: u64, opaque: &[u8]| {
                    let content = std::iter::repeat_n(
                        "alpha",
                        usize::try_from(frequency).expect("fixture frequency fits usize"),
                    )
                    .collect::<Vec<_>>()
                    .join(" ");
                    let positions = (0..frequency).collect::<Vec<_>>();
                    let fieldnorms = [
                        DeltaFieldNorm {
                            field_ord: 0,
                            raw_length: 1,
                            fieldnorm_id: fieldnorm_to_id(1),
                        },
                        DeltaFieldNorm {
                            field_ord: 1,
                            raw_length: frequency,
                            fieldnorm_id: fieldnorm_to_id(frequency),
                        },
                        DeltaFieldNorm {
                            field_ord: 4,
                            raw_length: 0,
                            fieldnorm_id: fieldnorm_to_id(0),
                        },
                    ];
                    let postings = [
                        DeltaTermPosting {
                            field_ord: 0,
                            term: document_id.as_bytes(),
                            frequency: 1,
                            positions: None,
                        },
                        DeltaTermPosting {
                            field_ord: 1,
                            term: b"alpha",
                            frequency,
                            positions: Some(&positions),
                        },
                    ];
                    delta
                        .apply_document_with_values(
                            global_docid,
                            DocId::from(document_id),
                            shipping_content_hash(document_id, &content),
                            &fieldnorms,
                            &postings,
                            &[DeltaNumericValue::u64(2, rank)],
                            &[
                                DeltaStoredValue::new(0, document_id.as_bytes()),
                                DeltaStoredValue::new(1, content.as_bytes()),
                                DeltaStoredValue::new(3, opaque),
                            ],
                        )
                        .expect("apply typed Delta parity document");
                };
            add_delta(0, "dead", 1, 5, b"discarded");
            add_delta(1, "live-a", 2, 11, b"opaque-a");
            add_delta(3, "live-b", 1, u64::MAX, b"");
        }
        assert_eq!(delta.delete_delta_id("dead"), Some(0));
        let frozen = delta.freeze(1);

        let mut direct =
            ColumnarAccumulator::new(DELTA_PARITY_SCHEMA).expect("direct typed accumulator");
        for (doc_ord, document_id, content, rank, opaque) in [
            (1, "live-a", "alpha alpha", 11, b"opaque-a".as_slice()),
            (3, "live-b", "alpha", u64::MAX, b"".as_slice()),
        ] {
            direct
                .add_document_with_values(
                    doc_ord,
                    &[
                        IndexedFieldValue::new(0, document_id),
                        IndexedFieldValue::new(1, content),
                    ],
                    &[IndexedNumericValue::u64(2, rank)],
                    &[StoredFieldValue::new(3, opaque)],
                )
                .expect("accumulate direct typed parity document");
        }
        let documents = [
            FlushDocumentInput::new(1, "live-a", shipping_content_hash("live-a", "alpha alpha")),
            FlushDocumentInput::new(3, "live-b", shipping_content_hash("live-b", "alpha")),
        ];
        let metadata = DeltaFlushInput {
            segment_id: 0xe5_4008,
            created_unix_s: 1_700_000_008,
            engine_version: CURRENT_ENGINE_VERSION,
        };
        let direct_encoded = flush_accumulator_with_mode(
            &direct,
            FlushSegmentInput {
                segment_id: metadata.segment_id,
                lease_docid_base: 0,
                created_unix_s: metadata.created_unix_s,
                engine_version: metadata.engine_version,
                documents: &documents,
            },
            FlushMode::Scalar,
        )
        .expect("flush direct typed parity fixture");
        let delta_encoded = flush_delta_snapshot(&frozen, metadata)
            .expect("flush typed Delta parity fixture")
            .expect("live typed Delta emits FSLX");
        assert_eq!(delta_encoded.as_bytes(), direct_encoded.as_bytes());
    }

    #[test]
    fn delta_seal_transaction_has_no_gap_and_never_crosses_a_lease() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let first_generation = index.snapshot().loaded_manifest().manifest.generation;
            let first_docid = DOC_ORDS_PER_LEASE - 1;
            let mut first_delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("first Delta lease");
            let dead_docid = DOC_ORDS_PER_LEASE - 3;
            apply_sealable_delta_document(&mut first_delta, dead_docid, "dead", "ghost", 1);
            apply_alpha_delta(&mut first_delta, first_docid, "lease-zero", 1);
            assert_eq!(first_delta.delete_delta_id("dead"), Some(dead_docid));
            let first_sealed = Arc::new(first_delta.freeze(first_generation));
            index
                .publish_delta_table(vec![Arc::clone(&first_sealed)])
                .expect("publish first Delta epoch");
            let held_first = index.search_snapshot();
            let first_pre_search = index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("search first Delta epoch");
            let first_pre_all = index
                .search_paginated(&cx, "*", 10, 0, true)
                .expect("match all first Delta epoch");
            let first_pre_docids = index
                .collect_docids(&cx, "alpha")
                .expect("collect first Delta epoch");
            assert_eq!(first_pre_search.hits[0].document_id, "lease-zero");
            assert_eq!(first_pre_all.hits.len(), 1);
            assert_eq!(first_pre_all.total_count, Some(1));
            assert_eq!(first_pre_docids, vec![first_docid]);

            let second_lease = u64::from(DOC_ORDS_PER_LEASE);
            let empty_second = Arc::new(
                DeltaSegment::new(DEFAULT_SCHEMA, second_lease, usize::MAX)
                    .expect("second empty Delta")
                    .freeze(first_generation + 1),
            );
            let installed_first = index
                .seal_delta_snapshot(
                    &cx,
                    Arc::clone(&first_sealed),
                    vec![empty_second],
                    DeltaFlushInput {
                        segment_id: 0xe5_4001,
                        created_unix_s: 1_700_000_001,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("seal first Delta lease");

            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("search sealed first epoch"),
                first_pre_search,
                "public ranking, score bits, ids, and counts must cross the first seal unchanged"
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "*", 10, 0, true)
                    .expect("match all sealed first epoch"),
                first_pre_all,
                "Delta tombstones and lease holes must remain hidden after sealing"
            );
            assert_eq!(
                index
                    .collect_docids(&cx, "alpha")
                    .expect("collect sealed first epoch"),
                first_pre_docids
            );

            assert_eq!(
                held_first.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(),
                1
            );
            assert_eq!(
                installed_first
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .unwrap(),
                1
            );
            assert_eq!(held_first.keeper_snapshot().segments().len(), 0);
            assert_eq!(held_first.delta_snapshots()[0].live_document_count(), 1);
            assert_eq!(installed_first.keeper_snapshot().segments().len(), 1);
            assert_eq!(
                installed_first.delta_snapshots()[0].live_document_count(),
                0
            );
            assert_eq!(
                installed_first
                    .keeper_snapshot()
                    .materialize_document_id(first_docid)
                    .as_deref(),
                Some("lease-zero")
            );

            let second_generation = index.snapshot().loaded_manifest().manifest.generation;
            let second_docid = DOC_ORDS_PER_LEASE;
            let mut second_delta = DeltaSegment::new(DEFAULT_SCHEMA, second_lease, usize::MAX)
                .expect("second Delta lease");
            apply_alpha_delta(&mut second_delta, second_docid, "lease-one", 1);
            let second_sealed = Arc::new(second_delta.freeze(second_generation));
            index
                .publish_delta_table(vec![Arc::clone(&second_sealed)])
                .expect("publish second Delta epoch");
            let held_second = index.search_snapshot();
            let second_pre_search = index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("search second Delta epoch");
            let second_pre_docids = index
                .collect_docids(&cx, "alpha")
                .expect("collect second Delta epoch");

            let third_lease = second_lease + u64::from(DOC_ORDS_PER_LEASE);
            let empty_third = Arc::new(
                DeltaSegment::new(DEFAULT_SCHEMA, third_lease, usize::MAX)
                    .expect("third empty Delta")
                    .freeze(second_generation + 1),
            );
            let installed_second = index
                .seal_delta_snapshot(
                    &cx,
                    Arc::clone(&second_sealed),
                    vec![empty_third],
                    DeltaFlushInput {
                        segment_id: 0xe5_4002,
                        created_unix_s: 1_700_000_002,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("seal second Delta lease");

            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("search sealed second epoch"),
                second_pre_search,
                "global public ranking must cross a mixed Keeper/Delta seal unchanged"
            );
            assert_eq!(
                index
                    .collect_docids(&cx, "alpha")
                    .expect("collect sealed second epoch"),
                second_pre_docids
            );

            assert_eq!(
                held_second.bm25_doc_freq(CONTENT_FIELD, b"alpha").unwrap(),
                2
            );
            assert_eq!(
                installed_second
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .unwrap(),
                2
            );
            let segments = &installed_second
                .keeper_snapshot()
                .loaded_manifest()
                .manifest
                .segments;
            assert_eq!(segments.len(), 2);
            assert_eq!(
                (segments[0].docid_lo, segments[0].docid_hi),
                (65_535, 65_536)
            );
            assert_eq!(
                (segments[1].docid_lo, segments[1].docid_hi),
                (65_536, 65_537)
            );
            for segment in segments {
                assert_eq!(
                    segment.docid_lo / u64::from(DOC_ORDS_PER_LEASE),
                    (segment.docid_hi - 1) / u64::from(DOC_ORDS_PER_LEASE),
                    "one sealed segment must stay inside one Q1 lease"
                );
            }

            let post_survivor_watermark = third_lease + u64::from(DOC_ORDS_PER_LEASE);
            assert_eq!(
                installed_second
                    .keeper_snapshot()
                    .loaded_manifest()
                    .manifest
                    .docid_high_watermark,
                post_survivor_watermark,
                "the durable watermark must cover every allocated replacement lease"
            );
            let mut reopened = QuillIndex::from_backend(
                IndexBackend::Memory(index.snapshot().as_ref().clone()),
                DEFAULT_SCHEMA,
                deterministic_config(),
            )
            .expect("reopen sealed Keeper snapshot");
            assert_eq!(
                reopened.writer_mut().next_lease_base,
                post_survivor_watermark
            );
            reopened
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("after-reopen", "fresh allocation")],
                )
                .await
                .expect("index after reopen");
            reopened.commit(&cx).await.expect("commit after reopen");
            assert_eq!(
                reopened
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .segments
                    .last()
                    .expect("post-reopen segment")
                    .docid_lo,
                post_survivor_watermark,
                "reopen must not reuse any Delta-owned lease"
            );
        });
    }

    #[test]
    fn delta_seal_rejects_an_omitted_surviving_shard_epoch() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut source =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("source Delta shard");
            apply_sealable_delta_document(&mut source, 0, "source", "alpha", 1);
            let source = Arc::new(source.freeze(generation));
            let survivor_base = u64::from(DOC_ORDS_PER_LEASE);
            let survivor_docid = DOC_ORDS_PER_LEASE;
            let mut survivor = DeltaSegment::new(DEFAULT_SCHEMA, survivor_base, usize::MAX)
                .expect("surviving Delta shard");
            apply_sealable_delta_document(&mut survivor, survivor_docid, "survivor", "beta", 1);
            let survivor = Arc::new(survivor.freeze(generation));
            index
                .publish_delta_table(vec![Arc::clone(&source), Arc::clone(&survivor)])
                .expect("publish two-shard Delta table");
            let before = index
                .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                .expect("query both Delta shards");

            let Err(error) = index
                .seal_delta_snapshot(
                    &cx,
                    Arc::clone(&source),
                    Vec::new(),
                    DeltaFlushInput {
                        segment_id: 0xe5_4003,
                        created_unix_s: 1_700_000_003,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
            else {
                panic!("seal must not discard an independently published shard");
            };
            assert!(matches!(
                error,
                QuillIndexError::Snapshot(SnapshotError::MissingDeltaRebind {
                    lease_base,
                    lease_end,
                }) if lease_base == survivor_base
                    && lease_end == survivor_base + u64::from(DOC_ORDS_PER_LEASE)
            ));
            assert!(index.writer_mut().pending_delta_seal.is_none());
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                    .expect("failed seal leaves complete table visible"),
                before
            );

            let rebound = Arc::new(survivor.rebind_keeper_generation(generation + 1));
            index
                .seal_delta_snapshot(
                    &cx,
                    source,
                    vec![rebound],
                    DeltaFlushInput {
                        segment_id: 0xe5_4003,
                        created_unix_s: 1_700_000_003,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("complete successor table seals source shard");
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                    .expect("successful seal retains surviving shard"),
                before
            );
        });
    }

    #[test]
    fn delta_seal_rejects_a_stale_earlier_freeze_of_a_surviving_shard() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut source =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("source Delta shard");
            apply_sealable_delta_document(&mut source, 0, "source", "alpha", 1);
            let source = Arc::new(source.freeze(generation));

            let survivor_base = u64::from(DOC_ORDS_PER_LEASE);
            let mut survivor = DeltaSegment::new(DEFAULT_SCHEMA, survivor_base, usize::MAX)
                .expect("surviving Delta shard");
            apply_sealable_delta_document(
                &mut survivor,
                DOC_ORDS_PER_LEASE,
                "survivor-first",
                "beta",
                1,
            );
            let stale_survivor = Arc::new(survivor.freeze(generation));
            apply_sealable_delta_document(
                &mut survivor,
                DOC_ORDS_PER_LEASE + 1,
                "survivor-latest",
                "gamma",
                1,
            );
            let latest_survivor = Arc::new(survivor.freeze(generation));
            assert_ne!(
                stale_survivor.publication_lineage(),
                latest_survivor.publication_lineage(),
                "successive freezes must name distinct immutable epochs"
            );

            index
                .publish_delta_table(vec![Arc::clone(&source), Arc::clone(&latest_survivor)])
                .expect("publish source and latest surviving epoch");
            let before = index
                .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
                .expect("query complete pre-seal view");
            assert_eq!(before.total_count, Some(3));

            let stale_rebound = Arc::new(stale_survivor.rebind_keeper_generation(generation + 1));
            let Err(error) = index
                .seal_delta_snapshot(
                    &cx,
                    Arc::clone(&source),
                    vec![stale_rebound],
                    DeltaFlushInput {
                        segment_id: 0xe5_4004,
                        created_unix_s: 1_700_000_004,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
            else {
                panic!("seal must reject an older freeze of the surviving shard");
            };
            assert!(matches!(
                error,
                QuillIndexError::Snapshot(SnapshotError::MissingDeltaRebind {
                    lease_base,
                    lease_end,
                }) if lease_base == survivor_base
                    && lease_end == survivor_base + u64::from(DOC_ORDS_PER_LEASE)
            ));
            assert!(index.writer_mut().pending_delta_seal.is_none());
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
                    .expect("rejected stale replacement leaves complete view visible"),
                before
            );

            let latest_rebound = Arc::new(latest_survivor.rebind_keeper_generation(generation + 1));
            index
                .seal_delta_snapshot(
                    &cx,
                    source,
                    vec![latest_rebound],
                    DeltaFlushInput {
                        segment_id: 0xe5_4004,
                        created_unix_s: 1_700_000_004,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("exact surviving epoch rebind seals source shard");
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha OR beta OR gamma", 10, 0, true)
                    .expect("successful seal retains latest surviving epoch"),
                before
            );
        });
    }

    #[test]
    fn labruntime_delta_to_keeper_swap_has_no_visibility_gap() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta lease");
        apply_alpha_delta(&mut delta, 0, "continuous", 1);
        let sealed = Arc::new(delta.freeze(generation));
        let replacement_base = u64::from(DOC_ORDS_PER_LEASE);
        let replacement_docid = u32::try_from(replacement_base).expect("replacement docid");
        let mut current_survivor = DeltaSegment::new(DEFAULT_SCHEMA, replacement_base, usize::MAX)
            .expect("current surviving Delta");
        apply_sealable_delta_document(
            &mut current_survivor,
            replacement_docid,
            "surviving",
            "beta",
            2,
        );
        let current_survivor = Arc::new(current_survivor.freeze(generation));
        let publisher = SnapshotPublisher::new(
            Arc::clone(&keeper),
            vec![Arc::clone(&sealed), Arc::clone(&current_survivor)],
        )
        .expect("snapshot publisher");
        let encoded = flush_delta_snapshot(
            &sealed,
            DeltaFlushInput {
                segment_id: 0xe5_4010,
                created_unix_s: 1_700_000_010,
                engine_version: CURRENT_ENGINE_VERSION,
            },
        )
        .expect("build Delta seal")
        .expect("live Delta emits a segment");
        let mut manifest = keeper.next_manifest().expect("successor manifest");
        manifest.segments.push(manifest_segment(&encoded, 1));
        manifest.docid_high_watermark = sealed.lease_end();
        let mut pending = BTreeMap::new();
        for field in DEFAULT_SCHEMA
            .fields
            .iter()
            .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        {
            pending.insert(
                field.id,
                (
                    sealed
                        .live_total_tokens(field.id)
                        .expect("Delta field stats"),
                    1,
                ),
            );
        }
        manifest.field_stats =
            merge_field_stats(&manifest.field_stats, &pending).expect("merge Delta field stats");
        let successor = Arc::new(
            keeper
                .publish_owned_segments(&manifest, vec![encoded])
                .expect("publish owned successor"),
        );
        let replacement = Arc::new(current_survivor.rebind_keeper_generation(manifest.generation));
        let prepared = publisher
            .prepare_sealed_manifest_with_deltas(DEFAULT_SCHEMA, &manifest, vec![replacement])
            .expect("prepare complete Delta seal publication");
        let publisher = Arc::new(publisher);
        let reader_started = Arc::new(AtomicBool::new(false));
        let writer_done = Arc::new(AtomicBool::new(false));

        let mut lab = LabRuntime::new(LabConfig::new(0xe5_4004).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);
        let reader_publisher = Arc::clone(&publisher);
        let reader_started_flag = Arc::clone(&reader_started);
        let reader_done = Arc::clone(&writer_done);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                reader_started_flag.store(true, Ordering::SeqCst);
                while !reader_done.load(Ordering::SeqCst) {
                    let observed = reader_publisher.load();
                    assert_eq!(
                        observed
                            .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                            .expect("alpha frequency"),
                        1
                    );
                    assert_eq!(
                        observed
                            .bm25_doc_freq(CONTENT_FIELD, b"beta")
                            .expect("beta frequency"),
                        1
                    );
                    let residency = (
                        observed.keeper_snapshot().segments().len(),
                        observed
                            .delta_snapshots()
                            .iter()
                            .map(|delta| delta.live_document_count())
                            .sum::<usize>(),
                    );
                    assert!(
                        residency == (0, 2) || residency == (1, 1),
                        "reader observed a visibility gap or duplicate: {residency:?}"
                    );
                    yield_now().await;
                }
                let observed = reader_publisher.load();
                assert_eq!(observed.keeper_snapshot().segments().len(), 1);
                assert_eq!(observed.delta_snapshots()[0].live_document_count(), 1);
            })
            .expect("create seal reader");

        let writer_publisher = Arc::clone(&publisher);
        let writer_started = Arc::clone(&reader_started);
        let writer_finished = Arc::clone(&writer_done);
        let (writer, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !writer_started.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                writer_publisher.install_prepared_sealed(successor, prepared);
                writer_finished.store(true, Ordering::SeqCst);
            })
            .expect("create seal writer");

        lab.scheduler.lock().schedule(reader, 0);
        lab.step_for_test();
        assert!(reader_started.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(writer, 0);
        let report = lab.run_until_quiescent_with_report();
        assert!(report.quiescent);
        assert!(report.oracle_report.all_passed());
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn e3_9_labruntime_pinned_commit_delete_interleavings_are_atomic() {
        run_with_cx(|cx| async move {
            for seed in e3_9_seed_corpus() {
                let index =
                    Arc::new(QuillIndex::in_memory(deterministic_config()).expect("memory index"));
                let alpha = IndexableDocument::new("alpha", "alpha committed");
                index
                    .index_document(&cx, &alpha)
                    .await
                    .expect("accumulate alpha");
                index.commit(&cx).await.expect("publish alpha");
                let beta = IndexableDocument::new("beta", "beta pending");
                index
                    .index_document(&cx, &beta)
                    .await
                    .expect("accumulate beta");

                let writer = Arc::clone(&index.writer);
                let commit_succeeded = Arc::new(AtomicBool::new(false));
                // 1 = delete followed the commit and succeeded; 2 = delete
                // serialized first and correctly rejected the pending commit.
                let delete_outcome = Arc::new(AtomicU8::new(0));
                let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let holder_writer = Arc::clone(&writer);
                let (holder, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let holder_cx = Cx::for_testing();
                        let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                            .await
                            .expect("acquire interleaving gate");
                        while holder_writer.waiters() < 2 {
                            yield_now().await;
                        }
                        drop(guard);
                    })
                    .expect("create interleaving gate");

                let commit_index = Arc::clone(&index);
                let commit_result = Arc::clone(&commit_succeeded);
                let (commit_task, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let task_cx = Cx::for_testing();
                        commit_index.commit(&task_cx).await.unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: commit failed: {error}")
                        });
                        commit_result.store(true, Ordering::SeqCst);
                    })
                    .expect("create commit task");

                let delete_index = Arc::clone(&index);
                let delete_result = Arc::clone(&delete_outcome);
                let (delete_task, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let task_cx = Cx::for_testing();
                        let outcome = match delete_index.delete_document(&task_cx, "alpha").await {
                            Ok(true) => 1,
                            Err(QuillIndexError::InvalidState { detail })
                                if detail.contains("fully committed") =>
                            {
                                2
                            }
                            Ok(false) => {
                                panic!("seed={seed:#018x}: live alpha was not deleted")
                            }
                            Err(error) => {
                                panic!("seed={seed:#018x}: unexpected delete result: {error}")
                            }
                        };
                        delete_result.store(outcome, Ordering::SeqCst);
                    })
                    .expect("create delete task");

                lab.scheduler.lock().schedule(holder, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(commit_task, 0);
                lab.scheduler.lock().schedule(delete_task, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("commit-delete", seed, &report);

                assert!(
                    commit_succeeded.load(Ordering::SeqCst),
                    "seed={seed:#018x}: commit did not complete"
                );
                let outcome = delete_outcome.load(Ordering::SeqCst);
                assert_ne!(
                    outcome, 0,
                    "seed={seed:#018x}: delete task did not complete"
                );
                let snapshot = index.snapshot();
                assert!(
                    snapshot
                        .resolve_document_id("beta")
                        .expect("resolve beta")
                        .is_some(),
                    "seed={seed:#018x}: committed beta was lost"
                );
                let alpha_is_live = snapshot
                    .resolve_document_id("alpha")
                    .expect("resolve alpha")
                    .is_some();
                if outcome == 1 {
                    assert!(
                        !alpha_is_live,
                        "seed={seed:#018x}: successful delete left alpha live"
                    );
                    assert_eq!(snapshot.doc_count(), 1);
                } else {
                    assert!(
                        alpha_is_live,
                        "seed={seed:#018x}: rejected delete changed alpha visibility"
                    );
                    assert_eq!(snapshot.doc_count(), 2);
                }
            }
        });
    }

    #[test]
    fn e3_9_labruntime_cancelled_ingest_and_merge_waiters_leave_no_partial_state() {
        run_with_cx(|cx| async move {
            for seed in e3_9_seed_corpus() {
                let ingest =
                    Arc::new(QuillIndex::in_memory(deterministic_config()).expect("memory index"));
                let before = ingest.search_snapshot();
                let writer = Arc::clone(&ingest.writer);
                let operation_cx = Arc::new(Cx::for_testing());
                let holder_release = Arc::new(AtomicBool::new(false));
                let operation_cancelled = Arc::new(AtomicBool::new(false));
                let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let holder_writer = Arc::clone(&writer);
                let release = Arc::clone(&holder_release);
                let (holder, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let holder_cx = Cx::for_testing();
                        let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                            .await
                            .expect("acquire ingest cancellation gate");
                        while !release.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        drop(guard);
                    })
                    .expect("create ingest cancellation gate");

                let ingest_index = Arc::clone(&ingest);
                let ingest_cx = Arc::clone(&operation_cx);
                let cancel_observed = Arc::clone(&operation_cancelled);
                let (ingest_task, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let document = IndexableDocument::new("cancelled", "never published");
                        match ingest_index.index_document(&ingest_cx, &document).await {
                            Err(QuillIndexError::Cancelled {
                                phase: "index writer lock",
                            }) => {
                                cancel_observed.store(true, Ordering::SeqCst);
                            }
                            result => {
                                panic!(
                                    "seed={seed:#018x}: unexpected cancelled-ingest result: {result:?}"
                                )
                            }
                        }
                    })
                    .expect("create cancelled ingest");

                let cancel_writer = Arc::clone(&writer);
                let cancel_cx = Arc::clone(&operation_cx);
                let release = Arc::clone(&holder_release);
                let (canceller, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while cancel_writer.waiters() == 0 {
                            yield_now().await;
                        }
                        cancel_cx.set_cancel_requested(true);
                        release.store(true, Ordering::SeqCst);
                    })
                    .expect("create ingest canceller");

                lab.scheduler.lock().schedule(holder, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(ingest_task, 0);
                lab.scheduler.lock().schedule(canceller, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("cancel-ingest", seed, &report);
                assert!(
                    operation_cancelled.load(Ordering::SeqCst),
                    "seed={seed:#018x}: ingest waiter did not observe cancellation"
                );
                assert!(Arc::ptr_eq(&before, &ingest.search_snapshot()));
                assert!(!ingest.has_uncommitted_changes());
                assert_eq!(ingest.doc_count(), 0);

                let merge = Arc::new(concat_merge_fixture_index(&cx).await);
                let before = merge.search_snapshot();
                let before_evidence = q1_ob2a_query_evidence(&merge, &cx);
                let source_ids = committed_segment_ids(&merge);
                let output_segment_id = fresh_merge_segment_id(&merge, seed);
                let writer = Arc::clone(&merge.writer);
                let operation_cx = Arc::new(Cx::for_testing());
                let holder_release = Arc::new(AtomicBool::new(false));
                let operation_cancelled = Arc::new(AtomicBool::new(false));
                let mut lab =
                    LabRuntime::new(LabConfig::new(seed.rotate_left(17)).max_steps(100_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let holder_writer = Arc::clone(&writer);
                let release = Arc::clone(&holder_release);
                let (holder, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let holder_cx = Cx::for_testing();
                        let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                            .await
                            .expect("acquire merge cancellation gate");
                        while !release.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        drop(guard);
                    })
                    .expect("create merge cancellation gate");

                let merge_index = Arc::clone(&merge);
                let merge_cx = Arc::clone(&operation_cx);
                let cancel_observed = Arc::clone(&operation_cancelled);
                let (merge_task, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        match merge_index
                            .concat_merge(&merge_cx, &source_ids, output_segment_id, 1_700_000_000)
                            .await
                        {
                            Err(QuillIndexError::Cancelled {
                                phase: "concat merge writer lock",
                            }) => {
                                cancel_observed.store(true, Ordering::SeqCst);
                            }
                            Ok(_) => {
                                panic!("seed={seed:#018x}: cancelled merge unexpectedly succeeded")
                            }
                            Err(error) => panic!(
                                "seed={seed:#018x}: unexpected cancelled-merge error: {error}"
                            ),
                        }
                    })
                    .expect("create cancelled merge");

                let cancel_writer = Arc::clone(&writer);
                let cancel_cx = Arc::clone(&operation_cx);
                let release = Arc::clone(&holder_release);
                let (canceller, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while cancel_writer.waiters() == 0 {
                            yield_now().await;
                        }
                        cancel_cx.set_cancel_requested(true);
                        release.store(true, Ordering::SeqCst);
                    })
                    .expect("create merge canceller");

                lab.scheduler.lock().schedule(holder, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(merge_task, 0);
                lab.scheduler.lock().schedule(canceller, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("cancel-merge", seed, &report);
                assert!(
                    operation_cancelled.load(Ordering::SeqCst),
                    "seed={seed:#018x}: merge waiter did not observe cancellation"
                );
                assert!(Arc::ptr_eq(&before, &merge.search_snapshot()));
                assert_eq!(q1_ob2a_query_evidence(&merge, &cx), before_evidence);
            }
        });
    }

    #[test]
    fn e3_9_labruntime_compaction_racing_search_keeps_complete_held_snapshots() {
        for seed in e3_9_seed_corpus() {
            let index = Arc::new(q1_ob4_tombstoned_index(20, &[0, 1, 2, 3, 4]));
            let query_cx = Cx::for_testing();
            let expected = q1_ob4_query_evidence(&index, &query_cx);
            let held = index.snapshot();
            let held_generation = held.loaded_manifest().manifest.generation;
            let weak = Arc::downgrade(&held);
            let reader_started = Arc::new(AtomicBool::new(false));
            let writer_done = Arc::new(AtomicBool::new(false));
            let reader_passes = Arc::new(AtomicUsize::new(0));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let reader_index = Arc::clone(&index);
            let reader_expected = expected.clone();
            let reader_started_flag = Arc::clone(&reader_started);
            let reader_done = Arc::clone(&writer_done);
            let passes = Arc::clone(&reader_passes);
            let (reader, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let cx = Cx::for_testing();
                    reader_started_flag.store(true, Ordering::SeqCst);
                    while !reader_done.load(Ordering::SeqCst) {
                        assert_eq!(
                            q1_ob4_query_evidence(&reader_index, &cx),
                            reader_expected,
                            "seed={seed:#018x}: reader observed torn compaction state"
                        );
                        passes.fetch_add(1, Ordering::SeqCst);
                        yield_now().await;
                    }
                    assert_eq!(
                        q1_ob4_query_evidence(&reader_index, &cx),
                        reader_expected,
                        "seed={seed:#018x}: post-compaction query drift"
                    );
                    passes.fetch_add(1, Ordering::SeqCst);
                })
                .expect("create compaction reader");

            let writer_index = Arc::clone(&index);
            let writer_started = Arc::clone(&reader_started);
            let writer_finished = Arc::clone(&writer_done);
            let (writer, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    while !writer_started.load(Ordering::SeqCst) {
                        yield_now().await;
                    }
                    let cx = Cx::for_testing();
                    let report = writer_index
                        .compact(&cx, CompactionPolicy::default())
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: compaction failed: {error}")
                        });
                    assert!(report.changed(), "seed={seed:#018x}: expected compaction");
                    writer_finished.store(true, Ordering::SeqCst);
                })
                .expect("create compaction writer");

            lab.scheduler.lock().schedule(reader, 0);
            lab.step_for_test();
            assert!(reader_started.load(Ordering::SeqCst));
            lab.scheduler.lock().schedule(writer, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("compaction-search", seed, &report);
            assert!(
                reader_passes.load(Ordering::SeqCst) >= 2,
                "seed={seed:#018x}: reader did not span the publication"
            );
            assert_eq!(q1_ob4_query_evidence(&index, &query_cx), expected);
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                held_generation + 1
            );
            assert_eq!(held.loaded_manifest().manifest.generation, held_generation);
            held.segments()[0]
                .verify()
                .unwrap_or_else(|error| panic!("seed={seed:#018x}: held segment failed: {error}"));
            assert!(
                weak.upgrade().is_some(),
                "seed={seed:#018x}: held snapshot retired too early"
            );
            drop(held);
            assert!(
                weak.upgrade().is_none(),
                "seed={seed:#018x}: old snapshot survived its last reader"
            );
        }
    }

    #[test]
    fn e6_5_labruntime_concurrent_ingest_and_search_are_snapshot_atomic() {
        run_with_cx(|cx| async move {
            for seed in e6_5_seed_corpus() {
                let initial = IndexableDocument::new("stable", "old alpha epoch");
                let additions = vec![
                    IndexableDocument::new("next-a", "new beta epoch"),
                    IndexableDocument::new("next-b", "new gamma epoch"),
                ];
                let index =
                    Arc::new(QuillIndex::in_memory(deterministic_config()).expect("E6.5 index"));
                index
                    .index_document(&cx, &initial)
                    .await
                    .expect("seed E6.5 index");
                index.commit(&cx).await.expect("publish E6.5 seed");
                let held_old_snapshot = index.search_snapshot();
                let old_artifact = e6_5_query_artifact(&index, &cx);

                let oracle =
                    QuillIndex::in_memory(deterministic_config()).expect("E6.5 successor oracle");
                oracle
                    .index_document(&cx, &initial)
                    .await
                    .expect("seed E6.5 successor oracle");
                oracle
                    .index_documents(&cx, &additions)
                    .await
                    .expect("extend E6.5 successor oracle");
                oracle
                    .commit(&cx)
                    .await
                    .expect("publish E6.5 successor oracle");
                let new_artifact = e6_5_query_artifact(&oracle, &cx);
                assert_ne!(old_artifact, new_artifact);

                let reader_started = Arc::new(AtomicBool::new(false));
                let writer_done = Arc::new(AtomicBool::new(false));
                let old_seen = Arc::new(AtomicBool::new(false));
                let new_seen = Arc::new(AtomicBool::new(false));
                let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(200_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let reader_index = Arc::clone(&index);
                let reader_started_flag = Arc::clone(&reader_started);
                let reader_done = Arc::clone(&writer_done);
                let reader_old_seen = Arc::clone(&old_seen);
                let reader_new_seen = Arc::clone(&new_seen);
                let reader_old = old_artifact.clone();
                let reader_new = new_artifact.clone();
                let (reader, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let task_cx = Cx::for_testing();
                        reader_started_flag.store(true, Ordering::SeqCst);
                        while !reader_done.load(Ordering::SeqCst) {
                            let observed = e6_5_query_artifact(&reader_index, &task_cx);
                            if observed == reader_old {
                                reader_old_seen.store(true, Ordering::SeqCst);
                            } else if observed == reader_new {
                                reader_new_seen.store(true, Ordering::SeqCst);
                            } else {
                                panic!(
                                    "seed={seed:#018x}: reader observed a torn ingest publication"
                                );
                            }
                            yield_now().await;
                        }
                        assert_eq!(
                            e6_5_query_artifact(&reader_index, &task_cx),
                            reader_new,
                            "seed={seed:#018x}: final reader snapshot is not the successor"
                        );
                        reader_new_seen.store(true, Ordering::SeqCst);
                    })
                    .expect("create E6.5 snapshot reader");

                let writer_index = Arc::clone(&index);
                let writer_started = Arc::clone(&reader_started);
                let writer_finished = Arc::clone(&writer_done);
                let writer_old = old_artifact.clone();
                let (writer, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while !writer_started.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        let task_cx = Cx::for_testing();
                        writer_index
                            .index_documents(&task_cx, &additions)
                            .await
                            .unwrap_or_else(|error| {
                                panic!("seed={seed:#018x}: concurrent ingest failed: {error}")
                            });
                        assert_eq!(
                            e6_5_query_artifact(&writer_index, &task_cx),
                            writer_old,
                            "seed={seed:#018x}: uncommitted ingest leaked into search"
                        );
                        yield_now().await;
                        writer_index.commit(&task_cx).await.unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: concurrent commit failed: {error}")
                        });
                        writer_finished.store(true, Ordering::SeqCst);
                    })
                    .expect("create E6.5 ingest writer");

                lab.scheduler.lock().schedule(reader, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(writer, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("e6.5-ingest-search", seed, &report);
                let replay = format!(
                    "seed={seed:#018x} fingerprint={:#018x}",
                    report.trace_fingerprint
                );
                assert!(
                    old_seen.load(Ordering::SeqCst),
                    "{replay}: reader never observed the old publication"
                );
                assert!(
                    new_seen.load(Ordering::SeqCst),
                    "{replay}: reader never observed the successor publication"
                );
                assert_eq!(
                    held_old_snapshot.live_doc_count(),
                    1,
                    "{replay}: held reader snapshot changed after publication"
                );
                assert_eq!(index.doc_count(), 3, "{replay}: successor document count");
            }
        });
    }

    #[test]
    fn e6_5_labruntime_cancelled_commit_and_seal_waiters_publish_nothing() {
        run_with_cx(|cx| async move {
            for seed in e6_5_seed_corpus() {
                let commit_index =
                    Arc::new(QuillIndex::in_memory(deterministic_config()).expect("commit index"));
                commit_index
                    .index_document(
                        &cx,
                        &IndexableDocument::new("commit-cancelled", "never partly visible"),
                    )
                    .await
                    .expect("stage cancellable commit");
                let commit_before = commit_index.search_snapshot();
                let commit_writer = Arc::clone(&commit_index.writer);
                let commit_cx = Arc::new(Cx::for_testing());
                let commit_release = Arc::new(AtomicBool::new(false));
                let commit_cancelled = Arc::new(AtomicBool::new(false));
                let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(100_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let holder_writer = Arc::clone(&commit_writer);
                let holder_release = Arc::clone(&commit_release);
                let (holder, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let holder_cx = Cx::for_testing();
                        let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                            .await
                            .expect("acquire E6.5 commit cancellation gate");
                        while !holder_release.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        drop(guard);
                    })
                    .expect("create E6.5 commit cancellation gate");

                let operation_index = Arc::clone(&commit_index);
                let operation_cx = Arc::clone(&commit_cx);
                let operation_cancelled = Arc::clone(&commit_cancelled);
                let (operation, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        match operation_index.commit(&operation_cx).await {
                            Err(QuillIndexError::Cancelled {
                                phase: "commit writer lock",
                            }) => {
                                operation_cancelled.store(true, Ordering::SeqCst);
                            }
                            Ok(_) => {
                                panic!("seed={seed:#018x}: cancelled commit unexpectedly succeeded")
                            }
                            Err(error) => panic!(
                                "seed={seed:#018x}: unexpected cancelled-commit error: {error}"
                            ),
                        }
                    })
                    .expect("create E6.5 cancelled commit");

                let cancel_writer = Arc::clone(&commit_writer);
                let cancel_cx = Arc::clone(&commit_cx);
                let cancel_release = Arc::clone(&commit_release);
                let (canceller, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while cancel_writer.waiters() == 0 {
                            yield_now().await;
                        }
                        cancel_cx.set_cancel_requested(true);
                        cancel_release.store(true, Ordering::SeqCst);
                    })
                    .expect("create E6.5 commit canceller");

                lab.scheduler.lock().schedule(holder, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(operation, 0);
                lab.scheduler.lock().schedule(canceller, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("e6.5-cancel-commit", seed, &report);
                let replay = format!(
                    "seed={seed:#018x} fingerprint={:#018x}",
                    report.trace_fingerprint
                );
                assert!(
                    commit_cancelled.load(Ordering::SeqCst),
                    "{replay}: commit waiter missed cancellation"
                );
                assert!(
                    Arc::ptr_eq(&commit_before, &commit_index.search_snapshot()),
                    "{replay}: cancelled commit changed the published snapshot"
                );
                assert!(
                    commit_index.has_uncommitted_changes(),
                    "{replay}: cancelled commit discarded retry state"
                );
                commit_index
                    .commit(&cx)
                    .await
                    .expect("retry cancelled E6.5 commit");
                assert_eq!(commit_index.doc_count(), 1, "{replay}: commit retry");

                let seal_index =
                    Arc::new(QuillIndex::in_memory(deterministic_config()).expect("seal index"));
                let generation = seal_index.snapshot().loaded_manifest().manifest.generation;
                let mut delta =
                    DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("E6.5 seal Delta");
                apply_alpha_delta(&mut delta, 0, "seal-cancelled", 1);
                let sealed = Arc::new(delta.freeze(generation));
                seal_index
                    .publish_delta_table(vec![Arc::clone(&sealed)])
                    .expect("publish E6.5 seal Delta");
                let seal_before = seal_index.search_snapshot();
                let seal_artifact_before = e6_5_query_artifact(&seal_index, &cx);
                let seal_writer = Arc::clone(&seal_index.writer);
                let seal_cx = Arc::new(Cx::for_testing());
                let seal_release = Arc::new(AtomicBool::new(false));
                let seal_cancelled = Arc::new(AtomicBool::new(false));
                let mut lab =
                    LabRuntime::new(LabConfig::new(seed.rotate_left(23)).max_steps(100_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let holder_writer = Arc::clone(&seal_writer);
                let holder_release = Arc::clone(&seal_release);
                let (holder, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let holder_cx = Cx::for_testing();
                        let guard = OwnedMutexGuard::lock(Arc::clone(&holder_writer), &holder_cx)
                            .await
                            .expect("acquire E6.5 seal cancellation gate");
                        while !holder_release.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        drop(guard);
                    })
                    .expect("create E6.5 seal cancellation gate");

                let operation_index = Arc::clone(&seal_index);
                let operation_cx = Arc::clone(&seal_cx);
                let operation_source = Arc::clone(&sealed);
                let operation_cancelled = Arc::clone(&seal_cancelled);
                let (operation, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        match operation_index
                            .seal_delta_snapshot(
                                &operation_cx,
                                operation_source,
                                Vec::new(),
                                DeltaFlushInput {
                                    segment_id: seed ^ 0xe6_5000_5ea1_0000,
                                    created_unix_s: 0,
                                    engine_version: CURRENT_ENGINE_VERSION,
                                },
                            )
                            .await
                        {
                            Err(QuillIndexError::Cancelled {
                                phase: "Delta seal writer lock",
                            }) => {
                                operation_cancelled.store(true, Ordering::SeqCst);
                            }
                            Ok(_) => {
                                panic!("seed={seed:#018x}: cancelled seal unexpectedly succeeded")
                            }
                            Err(error) => panic!(
                                "seed={seed:#018x}: unexpected cancelled-seal error: {error}"
                            ),
                        }
                    })
                    .expect("create E6.5 cancelled seal");

                let cancel_writer = Arc::clone(&seal_writer);
                let cancel_cx = Arc::clone(&seal_cx);
                let cancel_release = Arc::clone(&seal_release);
                let (canceller, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while cancel_writer.waiters() == 0 {
                            yield_now().await;
                        }
                        cancel_cx.set_cancel_requested(true);
                        cancel_release.store(true, Ordering::SeqCst);
                    })
                    .expect("create E6.5 seal canceller");

                lab.scheduler.lock().schedule(holder, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(operation, 0);
                lab.scheduler.lock().schedule(canceller, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("e6.5-cancel-seal", seed, &report);
                let replay = format!(
                    "seed={seed:#018x} fingerprint={:#018x}",
                    report.trace_fingerprint
                );
                assert!(
                    seal_cancelled.load(Ordering::SeqCst),
                    "{replay}: seal waiter missed cancellation"
                );
                assert!(
                    Arc::ptr_eq(&seal_before, &seal_index.search_snapshot()),
                    "{replay}: cancelled seal changed the published snapshot"
                );
                assert_eq!(
                    e6_5_query_artifact(&seal_index, &cx),
                    seal_artifact_before,
                    "{replay}: cancelled seal created a visibility gap"
                );
                assert_eq!(seal_index.search_snapshot().delta_count(), 1);
                seal_index
                    .seal_delta_snapshot(
                        &cx,
                        sealed,
                        Vec::new(),
                        DeltaFlushInput {
                            segment_id: seed ^ 0xe6_5000_5ea1_0000,
                            created_unix_s: 0,
                            engine_version: CURRENT_ENGINE_VERSION,
                        },
                    )
                    .await
                    .expect("retry cancelled E6.5 seal");
                assert_eq!(seal_index.search_snapshot().delta_count(), 0);
                assert_eq!(
                    e6_5_query_artifact(&seal_index, &cx),
                    seal_artifact_before,
                    "{replay}: seal retry changed query evidence"
                );
            }
        });
    }

    #[test]
    fn e6_5_labruntime_watch_stream_is_continuous_and_reopens_exactly() {
        run_with_cx(|cx| async move {
            let allowed = Arc::new(e6_5_watch_oracle(&cx).await);
            assert_eq!(allowed.len(), 4);

            for seed in e6_5_seed_corpus() {
                let directory = tempfile::tempdir().expect("E6.5 watch directory");
                let index = Arc::new(
                    QuillIndex::create(&cx, directory.path(), deterministic_config())
                        .await
                        .expect("create E6.5 watch index"),
                );
                LexicalSearch::index_documents(
                    index.as_ref(),
                    &cx,
                    &[
                        IndexableDocument::new("first", "old alpha epoch"),
                        IndexableDocument::new("second", "old beta epoch"),
                    ],
                )
                .await
                .expect("seed E6.5 watch index");
                LexicalSearch::commit(index.as_ref(), &cx)
                    .await
                    .expect("publish E6.5 watch seed");
                assert_eq!(e6_5_query_artifact(&index, &cx), allowed[0]);

                let reader_started = Arc::new(AtomicBool::new(false));
                let writer_done = Arc::new(AtomicBool::new(false));
                let observed_states = Arc::new(AtomicU8::new(0));
                let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(300_000));
                let region = lab.state.create_root_region(Budget::INFINITE);

                let reader_index = Arc::clone(&index);
                let reader_allowed = Arc::clone(&allowed);
                let reader_started_flag = Arc::clone(&reader_started);
                let reader_done = Arc::clone(&writer_done);
                let reader_observed = Arc::clone(&observed_states);
                let (reader, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        let task_cx = Cx::for_testing();
                        reader_started_flag.store(true, Ordering::SeqCst);
                        loop {
                            let artifact = e6_5_query_artifact(&reader_index, &task_cx);
                            let Some(state) = reader_allowed
                                .iter()
                                .position(|expected| *expected == artifact)
                            else {
                                panic!(
                                    "seed={seed:#018x}: watch reader observed a torn generation"
                                );
                            };
                            reader_observed.fetch_or(1_u8 << state, Ordering::SeqCst);
                            if reader_done.load(Ordering::SeqCst) {
                                break;
                            }
                            yield_now().await;
                        }
                    })
                    .expect("create E6.5 watch reader");

                let writer_index = Arc::clone(&index);
                let writer_started = Arc::clone(&reader_started);
                let writer_done_flag = Arc::clone(&writer_done);
                let writer_observed = Arc::clone(&observed_states);
                let writer_allowed = Arc::clone(&allowed);
                let (writer, _) = lab
                    .state
                    .create_task(region, Budget::INFINITE, async move {
                        while !writer_started.load(Ordering::SeqCst) {
                            yield_now().await;
                        }
                        let task_cx = Cx::for_testing();
                        LexicalSearch::index_documents(
                            writer_index.as_ref(),
                            &task_cx,
                            &[
                                IndexableDocument::new("first", "new alpha epoch"),
                                IndexableDocument::new("second", "new beta epoch"),
                            ],
                        )
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: watch upsert failed: {error}")
                        });
                        while writer_observed.load(Ordering::SeqCst) & (1 << 1) == 0 {
                            yield_now().await;
                        }

                        assert!(
                            writer_index
                                .delete_document(&task_cx, "second")
                                .await
                                .unwrap_or_else(|error| {
                                    panic!("seed={seed:#018x}: watch delete failed: {error}")
                                })
                        );
                        while writer_observed.load(Ordering::SeqCst) & (1 << 2) == 0 {
                            yield_now().await;
                        }

                        LexicalSearch::index_document(
                            writer_index.as_ref(),
                            &task_cx,
                            &IndexableDocument::new("third", "newcomer gamma epoch"),
                        )
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: watch newcomer ingest failed: {error}")
                        });
                        assert_eq!(
                            e6_5_query_artifact(&writer_index, &task_cx),
                            writer_allowed[2],
                            "seed={seed:#018x}: uncommitted watch row became visible"
                        );
                        LexicalSearch::commit(writer_index.as_ref(), &task_cx)
                            .await
                            .unwrap_or_else(|error| {
                                panic!("seed={seed:#018x}: watch commit failed: {error}")
                            });
                        writer_done_flag.store(true, Ordering::SeqCst);
                    })
                    .expect("create E6.5 watch writer");

                lab.scheduler.lock().schedule(reader, 0);
                lab.step_for_test();
                lab.scheduler.lock().schedule(writer, 0);
                let report = lab.run_until_quiescent_with_report();
                assert_e3_9_lab_report("e6.5-watch-stream", seed, &report);
                let replay = format!(
                    "seed={seed:#018x} fingerprint={:#018x}",
                    report.trace_fingerprint
                );
                assert_eq!(
                    observed_states.load(Ordering::SeqCst),
                    0b1111,
                    "{replay}: watch reader did not observe every complete generation"
                );
                assert_eq!(
                    e6_5_query_artifact(&index, &cx),
                    allowed[3],
                    "{replay}: watch writer did not publish the final generation"
                );

                drop(index);
                for recovery_round in 0..2 {
                    let reopened = QuillIndex::open(&cx, directory.path(), deterministic_config())
                        .await
                        .unwrap_or_else(|error| {
                            panic!("{replay}: recovery round {recovery_round} failed: {error}")
                        });
                    assert_eq!(
                        e6_5_query_artifact(&reopened, &cx),
                        allowed[3],
                        "{replay}: recovery round {recovery_round} changed query artifacts"
                    );
                    assert_eq!(
                        reopened.doc_count(),
                        2,
                        "{replay}: recovery round {recovery_round} live count"
                    );
                    drop(reopened);
                }
            }
        });
    }

    #[test]
    fn e6_5_labruntime_same_seed_replays_identical_segments_and_queries() {
        for seed in e6_5_seed_corpus() {
            let documents = (0..12)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("seed-{seed:016x}-{ordinal:02}"),
                        format!(
                            "epoch replay bucket-{} token-{}",
                            (seed ^ ordinal) % 5,
                            ordinal % 3
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let first =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("first replay"));
            let second =
                Arc::new(QuillIndex::in_memory(deterministic_config()).expect("second replay"));
            let mut lab = LabRuntime::new(LabConfig::new(seed).max_steps(200_000));
            let region = lab.state.create_root_region(Budget::INFINITE);

            let first_index = Arc::clone(&first);
            let first_documents = documents.clone();
            let (first_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    first_index
                        .index_documents(&task_cx, &first_documents)
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: first replay ingest failed: {error}")
                        });
                    yield_now().await;
                    first_index.commit(&task_cx).await.unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: first replay commit failed: {error}")
                    });
                })
                .expect("create first E6.5 replay");

            let second_index = Arc::clone(&second);
            let (second_task, _) = lab
                .state
                .create_task(region, Budget::INFINITE, async move {
                    let task_cx = Cx::for_testing();
                    second_index
                        .index_documents(&task_cx, &documents)
                        .await
                        .unwrap_or_else(|error| {
                            panic!("seed={seed:#018x}: second replay ingest failed: {error}")
                        });
                    yield_now().await;
                    second_index.commit(&task_cx).await.unwrap_or_else(|error| {
                        panic!("seed={seed:#018x}: second replay commit failed: {error}")
                    });
                })
                .expect("create second E6.5 replay");

            lab.scheduler.lock().schedule(first_task, 0);
            lab.scheduler.lock().schedule(second_task, 0);
            let report = lab.run_until_quiescent_with_report();
            assert_e3_9_lab_report("e6.5-deterministic-replay", seed, &report);
            let replay = format!(
                "seed={seed:#018x} fingerprint={:#018x}",
                report.trace_fingerprint
            );

            let first_snapshot = first.snapshot();
            let second_snapshot = second.snapshot();
            assert_eq!(
                first_snapshot.loaded_manifest().manifest,
                second_snapshot.loaded_manifest().manifest,
                "{replay}: deterministic MANIFEST mismatch"
            );
            assert_eq!(
                first_snapshot.segments().len(),
                second_snapshot.segments().len(),
                "{replay}: deterministic segment-count mismatch"
            );
            for (ordinal, (first_segment, second_segment)) in first_snapshot
                .segments()
                .iter()
                .zip(second_snapshot.segments())
                .enumerate()
            {
                assert_eq!(
                    first_segment.source_bytes(),
                    second_segment.source_bytes(),
                    "{replay}: deterministic segment {ordinal} bytes differ"
                );
            }
            let query_cx = Cx::for_testing();
            assert_eq!(
                e6_5_query_artifact(&first, &query_cx),
                e6_5_query_artifact(&second, &query_cx),
                "{replay}: deterministic query artifacts differ"
            );
        }
    }

    #[test]
    fn installed_delta_segment_without_manifest_is_not_durable_visibility() {
        run_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("temporary Keeper directory");
            let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("uncommitted Delta");
            apply_alpha_delta(&mut delta, 0, "visible-not-durable", 1);
            let frozen = Arc::new(delta.freeze(generation));
            index
                .publish_delta_table(vec![Arc::clone(&frozen)])
                .expect("publish process-local Delta");
            let encoded = flush_delta_snapshot(
                &frozen,
                DeltaFlushInput {
                    segment_id: 0xe5_4020,
                    created_unix_s: 1_700_000_020,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            )
            .expect("build uncommitted segment")
            .expect("live Delta emits a segment");
            let writer = match &mut index.writer_mut().backend {
                IndexBackend::Durable(writer) => writer,
                IndexBackend::Memory(_) => panic!("fixture must use a durable Keeper"),
            };
            let pending = encoded
                .write_temp_retryable(directory.path())
                .expect("sync retryable segment");
            writer
                .publish_segment(&cx, pending)
                .await
                .expect("install unreferenced segment");

            assert_eq!(
                index
                    .search_snapshot()
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .expect("process-local alpha frequency"),
                1,
                "the writer process sees its published Delta before commit"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("read old authoritative MANIFEST");
            assert_eq!(reopened.doc_count(), 0);
            assert!(reopened.segments().is_empty());
            assert_eq!(reopened.materialize_document_id(0), None);
        });
    }

    #[test]
    fn retained_delta_seal_reconciles_an_ambiguously_installed_segment_without_a_new_temp() {
        run_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("temporary Keeper directory");
            let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("retained Delta source");
            apply_alpha_delta(&mut delta, 0, "retained", 1);
            let sealed = Arc::new(delta.freeze(generation));
            index
                .publish_delta_table(vec![Arc::clone(&sealed)])
                .expect("publish retained Delta source");
            let before = index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query pre-seal Delta");

            let encoded = Arc::new(
                flush_delta_snapshot(
                    &sealed,
                    DeltaFlushInput {
                        segment_id: 0xe5_4030,
                        created_unix_s: 1_700_000_030,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .expect("build retained seal")
                .expect("live Delta emits FSLX"),
            );
            let mut manifest = index
                .snapshot()
                .next_manifest()
                .expect("successor MANIFEST");
            manifest.last_publish_unix_s = 0;
            manifest.segments.push(manifest_segment(&encoded, 1));
            manifest.docid_high_watermark = sealed.lease_end();
            let mut pending_field_stats = BTreeMap::new();
            for field in DEFAULT_SCHEMA
                .fields
                .iter()
                .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
            {
                pending_field_stats.insert(
                    field.id,
                    (
                        sealed
                            .live_total_tokens(field.id)
                            .expect("retained Delta field stats"),
                        1,
                    ),
                );
            }
            manifest.field_stats = merge_field_stats(&manifest.field_stats, &pending_field_stats)
                .expect("merge retained Delta field stats");
            let prepared = index
                .reader
                .published_snapshot
                .prepare_sealed_manifest(DEFAULT_SCHEMA, &manifest)
                .expect("prepare retained local swap");
            index.writer_mut().pending_delta_seal = Some(PendingDeltaSeal {
                encoded: Some(Arc::clone(&encoded)),
                segment_installed: false,
                manifest,
                prepared,
                next_seal_seq: 2,
                successor_watermark: sealed.lease_end(),
            });

            let writer = match &mut index.writer_mut().backend {
                IndexBackend::Durable(writer) => writer,
                IndexBackend::Memory(_) => panic!("fixture must use a durable Keeper"),
            };
            writer
                .publish_encoded_segment_retryable(&cx, Arc::clone(&encoded))
                .await
                .expect("simulate install whose caller future lost completion");
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation,
                "segment installation alone must not advance durable authority"
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("old Delta remains visible while reconciliation is pending"),
                before
            );

            index
                .resume_pending_delta_seal(&cx)
                .await
                .expect("reconcile exact canonical segment and publish retained MANIFEST");
            assert!(index.writer_mut().pending_delta_seal.is_none());
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("query reconciled Keeper"),
                before
            );
            assert!(
                index
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .last_publish_unix_s
                    > 0,
                "Keeper must stamp the retained zero-timestamp proposal exactly once"
            );
            assert!(
                !std::fs::read_dir(directory.path())
                    .expect("inspect retained-seal directory")
                    .filter_map(Result::ok)
                    .any(|entry| entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".tmp-segment-")),
                "exact canonical reconciliation must not manufacture a redundant retry temp"
            );
        });
    }

    #[test]
    fn dropped_delta_seal_after_manifest_install_resumes_exact_generation() {
        // This fixture intentionally pauses inside real blocking filesystem
        // choreography. A configured blocking pool keeps the executor free to
        // observe and release that checkpoint; the lightweight default test
        // runtime otherwise executes spawn_blocking inline by design.
        run_with_blocking_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("temporary Keeper directory");
            let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let next_seal_seq = index.writer_mut().next_seal_seq;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("drop-test Delta source");
            apply_alpha_delta(&mut delta, 0, "drop-survivor", 1);
            let sealed = Arc::new(delta.freeze(generation));
            index
                .publish_delta_table(vec![Arc::clone(&sealed)])
                .expect("publish drop-test Delta");
            let before = index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("query drop-test Delta before seal");

            let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
                directory.path(),
                crate::keeper::PublishCheckpoint::TempMovedToCurrent,
            );
            let mut seal = Box::pin(index.seal_delta_snapshot(
                &cx,
                Arc::clone(&sealed),
                Vec::new(),
                DeltaFlushInput {
                    segment_id: 0xe5_4031,
                    created_unix_s: 1_700_000_031,
                    engine_version: CURRENT_ENGINE_VERSION,
                },
            ));
            std::future::poll_fn(|task_cx| {
                if pause.is_reached() {
                    return Poll::Ready(());
                }
                match seal.as_mut().poll(task_cx) {
                    Poll::Pending => {
                        task_cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                    Poll::Ready(_) => {
                        panic!("Delta seal completed before the armed checkpoint")
                    }
                }
            })
            .await;
            drop(seal);

            assert!(
                index.writer_mut().pending_delta_seal.is_some(),
                "the actual dropped future must leave its complete proposal retained"
            );
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation,
                "KeeperWriter must still hold the pre-publish reader snapshot"
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("old Delta remains visible after the dropped future"),
                before
            );
            let installed = Manifest::from_bytes(
                &std::fs::read(directory.path().join("MANIFEST"))
                    .expect("read renamed successor MANIFEST"),
            )
            .expect("decode renamed successor MANIFEST");
            assert_eq!(
                installed.generation,
                generation + 1,
                "the real blocking publisher must have installed generation N+1"
            );

            pause.release();
            let resumed = index
                .resume_pending_delta_seal(&cx)
                .await
                .expect("reconcile stale writer snapshot and finish the local swap");
            assert!(index.writer_mut().pending_delta_seal.is_none());
            assert_eq!(resumed.keeper_generation(), generation + 1);
            assert_eq!(resumed.delta_count(), 0);
            assert_eq!(index.writer_mut().next_seal_seq, next_seal_seq + 1);
            assert_eq!(index.writer_mut().next_lease_base, sealed.lease_end());
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation + 1,
                "resume must reconcile N+1 rather than publishing N+2"
            );
            assert_eq!(
                KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                    .expect("reopen exactly reconciled Keeper")
                    .loaded_manifest()
                    .manifest
                    .generation,
                generation + 1
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "alpha", 10, 0, true)
                    .expect("query Keeper after exact resume"),
                before
            );
        });
    }

    #[test]
    fn tombstone_fold_keeps_multi_must_score_bits_across_delta_seal() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("score-order Delta");

            for global_docid in 0..3 {
                apply_tokenized_delta_document(
                    &mut delta,
                    global_docid,
                    &format!("dead-a-{global_docid}"),
                    "a",
                );
            }
            apply_tokenized_delta_document(&mut delta, 3, "target", "a b b c c x x");
            apply_tokenized_delta_document(&mut delta, 4, "live-bc", "b c x x");
            apply_tokenized_delta_document(&mut delta, 5, "live-c", "c x x x");
            for document_id in ["dead-a-0", "dead-a-1", "dead-a-2"] {
                assert!(delta.delete_delta_id(document_id).is_some());
            }
            let sealed = Arc::new(delta.freeze(generation));
            let a = sealed
                .find_term(CONTENT_FIELD, b"a")
                .expect("physical a term");
            assert_eq!(a.physical_doc_freq(), 4);
            assert_eq!(a.live_doc_freq(), 1);
            assert_eq!(
                DeltaPostingCursor::new(&sealed, CONTENT_FIELD, b"a")
                    .expect("live a cursor")
                    .cost(),
                1,
                "Delta traversal cost must match the sealed live posting count"
            );
            index
                .publish_delta_table(vec![Arc::clone(&sealed)])
                .expect("publish score-order Delta");
            let query = "content:a AND content:b AND content:c";
            let before = index
                .search_paginated(&cx, query, 10, 0, true)
                .expect("query multi-MUST Delta");
            assert_eq!(before.hits.len(), 1);
            assert_eq!(before.hits[0].document_id, "target");

            index
                .seal_delta_snapshot(
                    &cx,
                    Arc::clone(&sealed),
                    Vec::new(),
                    DeltaFlushInput {
                        segment_id: 0xe5_4032,
                        created_unix_s: 1_700_000_032,
                        engine_version: CURRENT_ENGINE_VERSION,
                    },
                )
                .await
                .expect("seal score-order Delta");
            let after = index
                .search_paginated(&cx, query, 10, 0, true)
                .expect("query multi-MUST Keeper");
            assert_eq!(after.hits.len(), 1);
            assert_eq!(after.hits[0].document_id, "target");
            assert_eq!(
                before.hits[0].score.to_bits(),
                after.hits[0].score.to_bits()
            );
            assert_eq!(before, after);
        });
    }

    #[test]
    fn delta_and_equivalent_sealed_term_have_identical_score_bits() {
        run_with_cx(|cx| async move {
            let keeper = Arc::new(
                KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"),
            );
            let generation = keeper.loaded_manifest().manifest.generation;
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
            apply_alpha_delta(&mut delta, 0, "doc", 1);
            let frozen = Arc::new(delta.freeze(generation));
            let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
                .expect("pre-seal composite snapshot");
            let pre_stats = composite
                .bm25_field_stats(CONTENT_FIELD)
                .expect("pre-seal content stats");
            let pre_df = composite
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("pre-seal alpha df");
            let mut pre_scorer =
                lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
                    .expect("pre-seal Delta term scorer");

            let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
            sealed
                .index_documents(&cx, &[IndexableDocument::new("doc", "alpha")])
                .await
                .expect("accumulate equivalent sealed document");
            sealed.commit(&cx).await.expect("seal equivalent document");
            let keeper = sealed.snapshot();
            let post_stats = snapshot_field(&keeper, CONTENT_FIELD).expect("post-seal stats");
            let post_df = snapshot_doc_freq(&keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                .expect("post-seal alpha df");
            assert_eq!(pre_stats, post_stats);
            assert_eq!(pre_df, post_df);

            let mut post_scorer = lower_term(
                &keeper.segments()[0],
                &keeper,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("post-seal term scorer");
            assert_eq!(
                pre_scorer.score().expect("pre-seal score").to_bits(),
                post_scorer.score().expect("post-seal score").to_bits(),
                "BM25 score must be bit-identical across the seal boundary"
            );
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn basic_record_option_scores_and_bounds_repeated_delta_occurrences_as_presence() {
        let keeper = Arc::new(
            KeeperSnapshot::in_memory(POSITIONLESS_QG_SCHEMA)
                .expect("genesis positionless Keeper snapshot"),
        );
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta =
            DeltaSegment::new(POSITIONLESS_QG_SCHEMA, 0, usize::MAX).expect("Basic Delta shard");
        apply_tokenized_delta_document(&mut delta, 0, "repeat", "token token neutral");
        apply_tokenized_delta_document(&mut delta, 1, "single", "token single neutral");
        let frozen = Arc::new(delta.freeze(generation));
        let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
            .expect("positionless Delta composite snapshot");
        let stats = composite
            .bm25_field_stats(CONTENT_FIELD)
            .expect("positionless Delta content statistics");
        assert_eq!((stats.total_tokens, stats.doc_count), (6, 2));
        assert_eq!(
            composite
                .bm25_doc_freq(CONTENT_FIELD, b"token")
                .expect("positionless Delta token document frequency"),
            2
        );

        let mut cursor = DeltaPostingCursor::new(&frozen, CONTENT_FIELD, b"token")
            .expect("positionless Delta token cursor");
        assert_eq!((cursor.doc(), cursor.freq()), (Some(0), Some(1)));
        assert!(cursor.positions_handle().is_none());
        assert_eq!(cursor.next().expect("advance Basic Delta cursor"), Some(1));
        assert_eq!(cursor.freq(), Some(1));
        assert!(cursor.positions_handle().is_none());

        let average = stats
            .average_field_length()
            .expect("non-empty positionless Delta average");
        let weight = crate::contract::idf(2, 2) * (1.0 + crate::contract::BM25_K1);
        let bound = cursor
            .term_score_upper_bound(average, weight, TermRecordOption::Basic)
            .expect("positionless Delta term bound");
        let mut scorer = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"token", 1.0)
            .expect("positionless Delta term scorer");
        let repeated_score = scorer.score().expect("score repeated Delta occurrence");
        assert_eq!(
            repeated_score.to_bits(),
            bound.to_bits(),
            "Delta Basic term bound must use the scorer's effective tf=1"
        );
        assert_eq!(scorer.next().expect("advance Basic Delta scorer"), Some(1));
        let single_score = scorer.score().expect("score single Delta occurrence");
        assert_eq!(
            repeated_score.to_bits(),
            single_score.to_bits(),
            "Delta Basic postings score document presence, not retained raw frequency"
        );
        assert_eq!(scorer.next().expect("exhaust Basic Delta scorer"), None);
    }

    #[test]
    fn delta_and_sealed_term_cursors_have_rank_exact_corpus_parity() {
        run_with_cx(|cx| async move {
            let keeper = Arc::new(
                KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"),
            );
            let generation = keeper.loaded_manifest().manifest.generation;
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
            for (docid, document_id, frequency) in [(0, "one", 1), (1, "three", 3), (2, "two", 2)] {
                apply_alpha_delta(&mut delta, docid, document_id, frequency);
            }
            let frozen = Arc::new(delta.freeze(generation));
            let composite = QuillSearchSnapshot::compose(0, keeper, vec![Arc::clone(&frozen)])
                .expect("pre-seal composite snapshot");
            let mut pre = lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
                .expect("pre-seal Delta scorer");

            let sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
            sealed
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("one", "alpha"),
                        IndexableDocument::new("three", "alpha alpha alpha"),
                        IndexableDocument::new("two", "alpha alpha"),
                    ],
                )
                .await
                .expect("accumulate equivalent corpus");
            sealed.commit(&cx).await.expect("seal equivalent corpus");
            let sealed_snapshot = sealed.snapshot();
            let mut post = lower_term(
                &sealed_snapshot.segments()[0],
                &sealed_snapshot,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("post-seal scorer");

            let collect_rows = |scorer: &mut ReferenceScorer<'_>| {
                let mut rows = Vec::new();
                while let Some(docid) = scorer.doc() {
                    rows.push((docid, scorer.score().expect("score current row").to_bits()));
                    scorer.next().expect("advance scorer");
                }
                rows
            };
            let pre_rows = collect_rows(&mut pre);
            let post_rows = collect_rows(&mut post);
            assert_eq!(
                pre_rows, post_rows,
                "per-doc score bits must survive sealing"
            );

            let mut pre_rank = pre_rows;
            let mut post_rank = post_rows;
            let best_first = |left: &(u32, u32), right: &(u32, u32)| {
                f32::from_bits(right.1)
                    .total_cmp(&f32::from_bits(left.1))
                    .then_with(|| left.0.cmp(&right.0))
            };
            pre_rank.sort_by(best_first);
            post_rank.sort_by(best_first);
            assert_eq!(pre_rank, post_rank, "RankExact order must survive sealing");
        });
    }

    #[test]
    fn mixed_sealed_and_delta_leaves_share_composite_bm25_statistics() {
        run_with_cx(|cx| async move {
            let mixed = QuillIndex::in_memory(deterministic_config()).expect("mixed index");
            mixed
                .index_documents(&cx, &[IndexableDocument::new("sealed", "alpha")])
                .await
                .expect("accumulate sealed row");
            mixed.commit(&cx).await.expect("commit sealed row");
            let keeper = mixed.snapshot();
            let generation = keeper.loaded_manifest().manifest.generation;
            let delta_base = u64::from(DOC_ORDS_PER_LEASE);
            let delta_docid = u32::try_from(delta_base).expect("second lease fits u32");
            let mut delta =
                DeltaSegment::new(DEFAULT_SCHEMA, delta_base, usize::MAX).expect("Delta shard");
            apply_alpha_delta(&mut delta, delta_docid, "delta", 1);
            let frozen = Arc::new(delta.freeze(generation));
            mixed
                .publish_delta_table(vec![Arc::clone(&frozen)])
                .expect("publish mixed Delta leaf");
            let composite =
                QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&frozen)])
                    .expect("mixed composite snapshot");
            assert_eq!(composite.bm25_doc_count(), 2);
            assert_eq!(
                composite
                    .bm25_field_stats(CONTENT_FIELD)
                    .expect("content stats")
                    .total_tokens,
                2
            );
            assert_eq!(
                composite
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .expect("composite alpha df"),
                2
            );

            let mut mixed_sealed = lower_composite_sealed_term(
                &keeper.segments()[0],
                &composite,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("mixed sealed scorer");
            let mixed_sealed_score = mixed_sealed.score().expect("score sealed leaf");
            let mut mixed_delta =
                lower_delta_term(&frozen, &composite, CONTENT_FIELD, b"alpha", 1.0)
                    .expect("mixed Delta scorer");
            let mixed_delta_score = mixed_delta.score().expect("score Delta leaf");

            let all_sealed =
                QuillIndex::in_memory(deterministic_config()).expect("all-sealed index");
            all_sealed
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("sealed", "alpha"),
                        IndexableDocument::new("delta", "alpha"),
                    ],
                )
                .await
                .expect("accumulate equivalent all-sealed rows");
            all_sealed
                .commit(&cx)
                .await
                .expect("commit equivalent all-sealed rows");
            let all_sealed_snapshot = all_sealed.snapshot();
            let mut oracle = lower_term(
                &all_sealed_snapshot.segments()[0],
                &all_sealed_snapshot,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("all-sealed oracle scorer");
            assert_eq!(
                mixed_sealed_score.to_bits(),
                oracle.score().expect("score first oracle row").to_bits()
            );
            assert_eq!(oracle.next().expect("advance oracle"), Some(1));
            assert_eq!(
                mixed_delta_score.to_bits(),
                oracle.score().expect("score second oracle row").to_bits()
            );
            assert_eq!(mixed_delta_score.to_bits(), mixed_sealed_score.to_bits());

            let mixed_results = mixed
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("execute public mixed query");
            let oracle_results = all_sealed
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("execute public all-sealed oracle");
            let mixed_rows = mixed_results
                .hits
                .iter()
                .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
                .collect::<Vec<_>>();
            let oracle_rows = oracle_results
                .hits
                .iter()
                .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
                .collect::<Vec<_>>();
            assert_eq!(mixed_rows, oracle_rows);
            assert_eq!(mixed_results.total_count, oracle_results.total_count);
            assert_eq!(mixed_results.doc_count, oracle_results.doc_count);
            assert_eq!(
                mixed
                    .search_paginated(&cx, "alpha", 1, 1, true)
                    .expect("paginate mixed tie")
                    .hits[0]
                    .document_id,
                "delta",
                "global docid ascending must break equal-score cross-residency ties before offset"
            );
            assert_eq!(
                mixed.collect_docids(&cx, "alpha").expect("mixed docset"),
                vec![0, delta_docid]
            );
        });
    }

    #[test]
    fn rank_pruning_gate_matches_runtime_union_capabilities() {
        let parser = DefaultQueryParser::new(DEFAULT_SCHEMA).expect("bind shipping parser");
        let direct_two = parser.parse("content:alpha OR content:beta");
        assert!(
            query_has_prunable_root_union(&direct_two.query, 1.0),
            "two direct term children are MaxScore-capable"
        );

        // Nested multi-field unions stay POSTINGS-only while
        // `GROUPED_MAX_SCORE_ENABLED` is false. The runtime candidate is
        // rank-safe and prunes, but its release A/B failed the performance
        // gate (see that constant's docs), so shipping does not pay to open
        // BLOCKMAX for these terms.
        // When `GROUPED_MAX_SCORE_ENABLED` flips to `true`, this assertion is
        // the one to invert — the shape assertions below stay as they are.
        let nested_two = parser.parse("alpha OR beta");
        assert!(
            !query_has_prunable_root_union(&nested_two.query, 1.0),
            "nested field unions must not open BLOCKMAX while grouped MaxScore is gated off"
        );

        let nested_nine = parser
            .parse("alpha OR beta OR gamma OR delta OR epsilon OR zeta OR eta OR theta OR iota");
        assert!(
            !query_has_prunable_root_union(&nested_nine.query, 1.0),
            "nine default multi-field children cannot supply physical BMW blocks, and grouped \
             MaxScore stops at MAX_SCORE_MAX_CLAUSES"
        );

        // Shape classification is pinned directly so it cannot rot while the
        // gate is closed: these are the inputs that decide eligibility the
        // moment `GROUPED_MAX_SCORE_ENABLED` flips.
        assert!(
            matches!(
                prunable_scorer_shape(&nested_two.query, 1.0),
                Some(PrunableScorerShape::Union {
                    children: 2,
                    kind: UnionChildKind::TermGroups,
                })
            ),
            "a default two-word query lowers to two pure-term groups"
        );
        assert!(
            matches!(
                prunable_scorer_shape(&direct_two.query, 1.0),
                Some(PrunableScorerShape::Union {
                    children: 2,
                    kind: UnionChildKind::DirectTerms,
                })
            ),
            "field-scoped children lower to direct terms, not groups"
        );
        // A root mixing a direct term with a multi-field group satisfies neither
        // the all-direct-terms nor the all-groupable runtime branch, so it can
        // never be admitted regardless of the gate.
        let mixed = parser.parse("content:alpha OR beta");
        assert!(
            matches!(
                prunable_scorer_shape(&mixed.query, 1.0),
                Some(PrunableScorerShape::Union {
                    children: 2,
                    kind: UnionChildKind::Mixed,
                })
            ),
            "a mixed direct-term/group root is consumed by no pruning strategy"
        );
        assert!(
            !query_has_prunable_root_union(&mixed.query, 1.0),
            "a mixed root must never open BLOCKMAX"
        );

        let direct_nine = parser.parse(
            "content:alpha OR content:beta OR content:gamma OR content:delta OR \
             content:epsilon OR content:zeta OR content:eta OR content:theta OR content:iota",
        );
        assert!(
            query_has_prunable_root_union(&direct_nine.query, 1.0),
            "nine direct sealed term children are BMW-capable"
        );
    }

    #[test]
    fn mixed_snapshot_disjunction_count_free_matches_exhaustive_at_pinned_k() {
        const DOCS_PER_RESIDENCY: u32 = 5_000;
        run_with_cx(|cx| async move {
            let mixed = QuillIndex::in_memory(deterministic_config()).expect("mixed index");
            let mut sealed_documents = Vec::with_capacity(
                usize::try_from(DOCS_PER_RESIDENCY).expect("fixture count fits usize"),
            );
            for ordinal in 0..DOCS_PER_RESIDENCY {
                let content = if ordinal == 4_352 {
                    std::iter::repeat_n("alpha", 64)
                        .chain(std::iter::once("beta"))
                        .collect::<Vec<_>>()
                        .join(" ")
                } else {
                    "alpha beta".to_owned()
                };
                let title = if ordinal.is_multiple_of(2) {
                    "alpha"
                } else {
                    "unrelated"
                };
                sealed_documents.push(
                    IndexableDocument::new(format!("sealed-{ordinal:05}"), content)
                        .with_title(title),
                );
            }
            mixed
                .index_documents(&cx, &sealed_documents)
                .await
                .expect("accumulate large sealed fixture");
            mixed.commit(&cx).await.expect("seal large fixture");
            let keeper = mixed.snapshot();
            let sealed_snapshot = QuillSearchSnapshot::compose(0, Arc::clone(&keeper), Vec::new())
                .expect("sealed-only statistics snapshot");
            let sealed_stats = sealed_snapshot
                .bm25_field_stats(CONTENT_FIELD)
                .expect("sealed content statistics");

            let generation = keeper.loaded_manifest().manifest.generation;
            let delta_base = u64::from(DOC_ORDS_PER_LEASE);
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, delta_base, usize::MAX)
                .expect("large Delta shard");
            for ordinal in 0..DOCS_PER_RESIDENCY {
                let global_docid = u32::try_from(delta_base + u64::from(ordinal))
                    .expect("Delta fixture docid fits u32");
                let content = if ordinal == 4_352 {
                    "alpha beta beta beta beta beta beta beta beta beta"
                } else {
                    "alpha beta gamma delta epsilon zeta eta theta"
                };
                apply_tokenized_delta_document(
                    &mut delta,
                    global_docid,
                    &format!("delta-{ordinal:05}"),
                    content,
                );
            }
            let delta = Arc::new(delta.freeze(generation));
            mixed
                .publish_delta_table(vec![delta])
                .expect("publish large mixed snapshot");
            let live_snapshot = mixed.search_snapshot();
            let live_stats = live_snapshot
                .bm25_field_stats(CONTENT_FIELD)
                .expect("mixed content statistics");
            assert_ne!(
                sealed_stats
                    .average_field_length()
                    .expect("sealed avgdl")
                    .to_bits(),
                live_stats
                    .average_field_length()
                    .expect("mixed avgdl")
                    .to_bits(),
                "query-time avgdl must differ from the value at block seal time"
            );

            let direct_query = "content:alpha OR content:beta";
            let parsed = DefaultQueryParser::new(DEFAULT_SCHEMA)
                .expect("bind shipping parser")
                .parse(direct_query);
            let segment = live_snapshot
                .keeper_snapshot()
                .segments()
                .first()
                .expect("mixed fixture has one sealed segment");
            assert_eq!(segment.cached_rank_pruning_term_count(), 0);
            let exact_probe = mixed
                .search_paginated(&cx, direct_query, 1, 0, true)
                .expect("execute exact-count cache probe");
            assert_eq!(exact_probe.total_count, Some(10_000));
            assert_eq!(
                segment.cached_rank_pruning_term_count(),
                0,
                "exact counting must not trigger BLOCKMAX validation"
            );
            let direct_checkpoint: QueryCheckpointHandle<'_> =
                QueryCheckpoint::new(&cx, "lower_query_test", u64::MAX, 0);
            let mut direct = lower_query(
                &cx,
                &direct_checkpoint,
                &parsed.query,
                1.0,
                QueryLeaf::Sealed(segment),
                &live_snapshot,
                DEFAULT_SCHEMA,
                1_024,
                true,
                true,
            )
            .expect("lower direct disjunction with pruning metadata");
            let mut direct_collector = TopDocsCollector::new(1, 0).expect("top-one collector");
            direct_collector
                .collect(&mut direct, segment)
                .expect("collect direct disjunction");
            let (maxscore_windows, bmw_windows) = direct
                .pruning_window_counts()
                .expect("direct disjunction remains a top-level union");
            assert!(
                maxscore_windows > 0,
                "direct disjunction silently fell back"
            );
            assert_eq!(bmw_windows, 0);
            assert_eq!(
                segment.cached_rank_pruning_term_count(),
                2,
                "only content alpha/beta should be cached"
            );

            for limit in [1, 10, 100, 1_000] {
                let count_free = mixed
                    .search_paginated(&cx, direct_query, limit, 0, false)
                    .expect("execute count-free disjunction");
                let exhaustive = mixed
                    .search_paginated(&cx, direct_query, limit, 0, true)
                    .expect("execute exhaustive counted disjunction");
                assert_eq!(count_free.total_count, None);
                assert_eq!(
                    exhaustive.total_count,
                    Some(u64::from(DOCS_PER_RESIDENCY) * 2)
                );
                assert_eq!(count_free.hits.len(), exhaustive.hits.len());
                for (candidate, oracle) in count_free.hits.iter().zip(&exhaustive.hits) {
                    assert_eq!(candidate.document_id, oracle.document_id);
                    assert_eq!(candidate.global_docid, oracle.global_docid);
                    assert_eq!(candidate.score.to_bits(), oracle.score.to_bits());
                }
                assert_eq!(segment.cached_rank_pruning_term_count(), 2);
            }

            let nested_count_free = mixed
                .search_paginated(&cx, "alpha OR beta", 10, 0, false)
                .expect("execute nested count-free fallback");
            let nested_exhaustive = mixed
                .search_paginated(&cx, "alpha OR beta", 10, 0, true)
                .expect("execute nested exhaustive oracle");
            assert_eq!(nested_count_free.total_count, None);
            assert_eq!(nested_exhaustive.total_count, Some(10_000));
            assert_eq!(nested_count_free.hits.len(), nested_exhaustive.hits.len());
            for (candidate, oracle) in nested_count_free.hits.iter().zip(&nested_exhaustive.hits) {
                assert_eq!(candidate.document_id, oracle.document_id);
                assert_eq!(candidate.global_docid, oracle.global_docid);
                assert_eq!(candidate.score.to_bits(), oracle.score.to_bits());
            }
            assert_eq!(
                segment.cached_rank_pruning_term_count(),
                2,
                "nested fallback must not validate title BLOCKMAX metadata"
            );
        });
    }

    const E410_ALGEBRA_DOCS: usize = 48;
    const E410_ALGEBRA_VOCAB: [&str; 8] = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    ];

    fn e410_lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    /// Deterministic 48-document corpus over an eight-word vocabulary with
    /// varying term frequencies and document lengths, shared by the E4.10
    /// public-surface property suites (bd-quill-e4-argus-3ycz.10).
    fn e410_algebra_documents() -> Vec<IndexableDocument> {
        let mut state = 0xe410_5eed_u64;
        let mut documents = Vec::with_capacity(E410_ALGEBRA_DOCS);
        for ordinal in 0..E410_ALGEBRA_DOCS {
            let word_count =
                3 + usize::try_from(e410_lcg_next(&mut state) % 8).expect("word count fits usize");
            let mut words = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                words.push(
                    E410_ALGEBRA_VOCAB
                        [usize::try_from(e410_lcg_next(&mut state) % 8).expect("pick fits usize")],
                );
            }
            let title = E410_ALGEBRA_VOCAB
                [usize::try_from(e410_lcg_next(&mut state) % 8).expect("title fits usize")];
            documents.push(
                IndexableDocument::new(format!("algebra-{ordinal:02}"), words.join(" "))
                    .with_title(title),
            );
        }
        documents
    }

    /// E4.10 query-algebra property suite at the preparsed public surface
    /// (bd-quill-e4-argus-3ycz.10): Should-clause commutativity of the result
    /// set and Must-clause associativity hold end-to-end through plan and
    /// collect. Laws compare result sets with epsilon-bounded scores because
    /// clause order feeds non-associative f32 accumulation. Runs behind
    /// `bench-internals` because associativity needs programmatic nesting the
    /// shipping string parser cannot construct.
    #[cfg(feature = "bench-internals")]
    #[test]
    fn e410_public_api_algebra_laws_hold() {
        const DOCS: usize = E410_ALGEBRA_DOCS;
        const VOCAB: [&str; 8] = E410_ALGEBRA_VOCAB;
        const SCORE_EPSILON: f32 = 1e-4;
        fn scored_set(result: &QuillSearchResult) -> Vec<(String, f32)> {
            let mut rows = result
                .hits
                .iter()
                .map(|hit| (hit.document_id.clone(), hit.score))
                .collect::<Vec<_>>();
            rows.sort_by(|left, right| left.0.cmp(&right.0));
            rows
        }
        fn assert_epsilon_equal_sets(label: &str, left: &[(String, f32)], right: &[(String, f32)]) {
            assert_eq!(
                left.iter().map(|(id, _)| id).collect::<Vec<_>>(),
                right.iter().map(|(id, _)| id).collect::<Vec<_>>(),
                "{label}: result document sets diverged",
            );
            for ((document_id, left_score), (_, right_score)) in left.iter().zip(right) {
                let drift = (left_score - right_score).abs()
                    / left_score.abs().max(right_score.abs()).max(1e-12);
                assert!(
                    drift <= SCORE_EPSILON,
                    "{label}: {document_id} score drift {left_score} vs {right_score} \
                     breaches ScoreEpsilon",
                );
            }
        }
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("algebra index");
            index
                .index_documents(&cx, &e410_algebra_documents())
                .await
                .expect("accumulate algebra corpus");
            index.commit(&cx).await.expect("seal algebra corpus");

            let content_term = |text: &str| Query::Term {
                fields: vec![QueryField::new(CONTENT_FIELD, 1.0)],
                text: text.to_owned(),
            };

            let mut nonempty_should = 0_usize;
            let mut nonempty_must = 0_usize;
            let mut seed_state = 0x5eed_e410_u64;
            for seed in 0_u64..16 {
                let mut chosen = Vec::with_capacity(3);
                while chosen.len() < 3 {
                    let pick = usize::try_from(e410_lcg_next(&mut seed_state) % 8)
                        .expect("pick fits usize");
                    if !chosen.contains(&pick) {
                        chosen.push(pick);
                    }
                }
                let terms = [VOCAB[chosen[0]], VOCAB[chosen[1]], VOCAB[chosen[2]]];

                let should = |order: [usize; 3]| Query::Boolean {
                    clauses: order
                        .into_iter()
                        .map(|slot| BooleanClause::new(Occur::Should, content_term(terms[slot])))
                        .collect(),
                    operator: None,
                };
                let baseline = index
                    .search_preparsed_paginated(&cx, &should([0, 1, 2]), DOCS, 0, true)
                    .expect("baseline Should union");
                let baseline_rows = scored_set(&baseline);
                if !baseline_rows.is_empty() {
                    nonempty_should += 1;
                }
                for order in [[2, 0, 1], [1, 2, 0], [2, 1, 0]] {
                    let permuted = index
                        .search_preparsed_paginated(&cx, &should(order), DOCS, 0, true)
                        .expect("permuted Should union");
                    assert_eq!(
                        permuted.total_count, baseline.total_count,
                        "seed {seed}: Should permutation changed the exact count",
                    );
                    assert_epsilon_equal_sets(
                        &format!("seed {seed} Should order {order:?}"),
                        &baseline_rows,
                        &scored_set(&permuted),
                    );
                }

                let must = |query: Query| BooleanClause::new(Occur::Must, query);
                let flat = Query::Boolean {
                    clauses: vec![
                        must(content_term(terms[0])),
                        must(content_term(terms[1])),
                        must(content_term(terms[2])),
                    ],
                    operator: None,
                };
                let right_nested = Query::Boolean {
                    clauses: vec![
                        must(content_term(terms[0])),
                        must(Query::Boolean {
                            clauses: vec![
                                must(content_term(terms[1])),
                                must(content_term(terms[2])),
                            ],
                            operator: None,
                        }),
                    ],
                    operator: None,
                };
                let left_nested = Query::Boolean {
                    clauses: vec![
                        must(Query::Boolean {
                            clauses: vec![
                                must(content_term(terms[0])),
                                must(content_term(terms[1])),
                            ],
                            operator: None,
                        }),
                        must(content_term(terms[2])),
                    ],
                    operator: None,
                };
                let flat_result = index
                    .search_preparsed_paginated(&cx, &flat, DOCS, 0, true)
                    .expect("flat Must intersection");
                let flat_rows = scored_set(&flat_result);
                if !flat_rows.is_empty() {
                    nonempty_must += 1;
                }
                for (shape, query) in [("right-nested", right_nested), ("left-nested", left_nested)]
                {
                    let nested = index
                        .search_preparsed_paginated(&cx, &query, DOCS, 0, true)
                        .expect("nested Must intersection");
                    assert_eq!(
                        nested.total_count, flat_result.total_count,
                        "seed {seed}: {shape} Must association changed the exact count",
                    );
                    assert_epsilon_equal_sets(
                        &format!("seed {seed} Must {shape}"),
                        &flat_rows,
                        &scored_set(&nested),
                    );
                }
            }
            assert!(
                nonempty_should >= 8,
                "Should-commutativity draws were mostly vacuous: {nonempty_should}/16",
            );
            assert!(
                nonempty_must >= 1,
                "every Must-associativity draw was vacuous",
            );
        });
    }

    /// E4.10 pagination metamorphic at the shipping string surface
    /// (bd-quill-e4-argus-3ycz.10): page concatenation reproduces the full
    /// ranking bit-for-bit on both the nested default lowering and the
    /// prunable direct-syntax union, with and without exact counting, and an
    /// offset past the last match yields an empty page.
    #[test]
    fn e410_public_api_pagination_metamorphic_holds() {
        const DOCS: usize = E410_ALGEBRA_DOCS;
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("pagination index");
            index
                .index_documents(&cx, &e410_algebra_documents())
                .await
                .expect("accumulate pagination corpus");
            index.commit(&cx).await.expect("seal pagination corpus");

            for query in ["alpha epsilon", "content:alpha OR content:epsilon"] {
                for exact_count in [true, false] {
                    let full = index
                        .search_paginated(&cx, query, DOCS, 0, exact_count)
                        .expect("full ranking");
                    let full_rows = full
                        .hits
                        .iter()
                        .map(|hit| {
                            (
                                hit.document_id.clone(),
                                hit.global_docid,
                                hit.score.to_bits(),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert!(
                        !full_rows.is_empty(),
                        "pagination fixture query {query:?} must match documents",
                    );
                    for page_size in [1_usize, 7] {
                        let mut paged_rows = Vec::with_capacity(full_rows.len());
                        let mut offset = 0_usize;
                        loop {
                            let page = index
                                .search_paginated(&cx, query, page_size, offset, exact_count)
                                .expect("ranking page");
                            if exact_count {
                                assert_eq!(
                                    page.total_count, full.total_count,
                                    "{query:?}: page at offset {offset} changed the exact count",
                                );
                            } else {
                                assert_eq!(page.total_count, None);
                            }
                            if page.hits.is_empty() {
                                break;
                            }
                            paged_rows.extend(page.hits.iter().map(|hit| {
                                (
                                    hit.document_id.clone(),
                                    hit.global_docid,
                                    hit.score.to_bits(),
                                )
                            }));
                            offset += page_size;
                        }
                        assert_eq!(
                            paged_rows, full_rows,
                            "{query:?} page-size {page_size} exact_count {exact_count}: \
                             page concatenation must reproduce the full ranking bit-for-bit",
                        );
                        let overshoot = index
                            .search_paginated(&cx, query, page_size, full_rows.len(), exact_count)
                            .expect("overshoot page");
                        assert!(
                            overshoot.hits.is_empty(),
                            "{query:?}: offset past the last match must yield an empty page",
                        );
                    }
                }
            }
        });
    }

    /// Scalar ingestion refuses duplicate live document ids, both in-batch
    /// and against the committed snapshot: the Delta lane owns logical
    /// upserts, so the scalar lane pins the reject posture that the lexical
    /// engine's upsert-replaces semantics intentionally diverge from
    /// (bd-quill-e4-argus-3ycz.10).
    #[test]
    fn e410_scalar_duplicate_document_id_is_rejected() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("dup index");
            index
                .index_documents(&cx, &[IndexableDocument::new("dup-doc", "alpha original")])
                .await
                .expect("first ingest");
            let in_batch = index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("dup-doc", "beta replacement")],
                )
                .await
                .expect_err("uncommitted duplicate id must be rejected");
            assert!(
                in_batch.to_string().contains("duplicate live document id"),
                "unexpected in-batch duplicate error: {in_batch}",
            );
            index.commit(&cx).await.expect("seal original document");
            let committed = index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("dup-doc", "gamma replacement")],
                )
                .await
                .expect_err("committed duplicate id must be rejected");
            assert!(
                committed.to_string().contains("duplicate live document id"),
                "unexpected committed duplicate error: {committed}",
            );
            let original = index
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("search the surviving document");
            assert_eq!(original.hits.len(), 1);
            assert_eq!(original.hits[0].document_id, "dup-doc");
            assert_eq!(
                index.doc_count(),
                1,
                "rejected ingests must not change doc_count",
            );
        });
    }

    /// A document indexed without metadata must materialize as absent, not as
    /// an empty JSON object. Ingest always writes a canonical metadata object,
    /// so the empty map round-trips as `{}` unless the materialization
    /// boundary folds it away — while the Tantivy oracle omits the field
    /// entirely. `IndexableDocument::new` starts with an empty map, so leaving
    /// `Some({})` in place flips `metadata.is_some()` on the majority of
    /// documents at the default-engine flip (fusion winner re-attachment,
    /// fsfs JSON output) and allocates a per-hit `Arc` carrying no
    /// information (bd-quill-e4-argus-3ycz.10).
    #[test]
    fn e410_empty_metadata_materializes_as_absent_not_empty_object() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("metadata index");
            index
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("doc-bare", "alpha bare document"),
                        IndexableDocument::new("doc-meta", "alpha annotated document")
                            .with_metadata("lang", "rust"),
                    ],
                )
                .await
                .expect("ingest mixed-metadata batch");
            index.commit(&cx).await.expect("seal mixed-metadata batch");
            let annotated = serde_json::json!({ "lang": "rust" });

            let hydrated = index
                .search_results(&cx, "alpha", 10)
                .expect("search the hydrated envelope");
            assert_eq!(hydrated.len(), 2, "both documents match the shared term");
            for result in &hydrated {
                match result.doc_id.as_str() {
                    "doc-bare" => assert!(
                        result.metadata.is_none(),
                        "empty stored metadata must materialize as absent, got {:?}",
                        result.metadata,
                    ),
                    "doc-meta" => assert_eq!(
                        result.metadata.as_deref(),
                        Some(&annotated),
                        "present metadata must survive the same boundary",
                    ),
                    other => panic!("unexpected document id {other}"),
                }
            }

            // The deferred-fusion lane re-attaches winners' metadata through
            // the same boundary, so hydration must agree hit for hit.
            let mut candidates = LexicalSearch::search_fusion_candidates(&index, &cx, "alpha", 10)
                .await
                .expect("deferred fusion candidates");
            assert!(
                candidates
                    .iter()
                    .all(|candidate| candidate.metadata.is_none()),
                "deferred candidates carry no metadata before hydration",
            );
            LexicalSearch::hydrate_fusion_metadata(&index, &cx, &mut candidates)
                .await
                .expect("hydrate fusion winners");
            for candidate in &candidates {
                match candidate.doc_id.as_str() {
                    "doc-bare" => assert!(
                        candidate.metadata.is_none(),
                        "hydration must not resurrect an empty metadata object",
                    ),
                    "doc-meta" => assert_eq!(
                        candidate.metadata.as_deref(),
                        Some(&annotated),
                        "hydration must restore the annotated metadata verbatim",
                    ),
                    other => panic!("unexpected document id {other}"),
                }
            }
        });
    }

    #[test]
    fn range_set_and_glob_lowering_is_identical_across_residency() {
        run_with_cx(|cx| async move {
            let second_base = u64::from(DOC_ORDS_PER_LEASE);
            let second_docid = u32::try_from(second_base).expect("second fixture lease fits u32");
            let first = typed_delta_snapshot(0, 0, "first", "alpha alpine", 11);
            let second =
                typed_delta_snapshot(second_base, second_docid, "second", "alpine beta", 17);
            let empty = typed_residency_index(&[], &[], deterministic_config());
            let all_delta = typed_residency_index(
                &[],
                &[Arc::clone(&first), Arc::clone(&second)],
                deterministic_config(),
            );
            let mixed = typed_residency_index(
                &[Arc::clone(&first)],
                &[Arc::clone(&second)],
                deterministic_config(),
            );
            let all_sealed = typed_residency_index(
                &[Arc::clone(&first), Arc::clone(&second)],
                &[],
                deterministic_config(),
            );
            assert_eq!(all_delta.doc_count(), 2);
            assert_eq!(empty.doc_count(), 0);
            assert_eq!(mixed.doc_count(), 2, "public count includes the Delta leaf");
            assert_eq!(all_sealed.doc_count(), 2);

            let queries = [
                (
                    Query::Boost {
                        query: Box::new(Query::Range {
                            field_id: 2,
                            lower: Bound::Included(QueryValue::U64(11)),
                            upper: Bound::Included(QueryValue::U64(17)),
                        }),
                        factor: 2.5,
                    },
                    2.5_f32,
                ),
                (
                    Query::Range {
                        field_id: 1,
                        lower: Bound::Included(QueryValue::Str("alpha".to_owned())),
                        upper: Bound::Included(QueryValue::Str("alpine".to_owned())),
                    },
                    1.0,
                ),
                (
                    Query::Range {
                        field_id: 5,
                        lower: Bound::Included(QueryValue::U64(11)),
                        upper: Bound::Included(QueryValue::U64(17)),
                    },
                    1.0,
                ),
                (
                    Query::Set {
                        field_id: 1,
                        values: vec![
                            QueryValue::Str("alpha".to_owned()),
                            QueryValue::Str("alpine".to_owned()),
                            QueryValue::Str("alpha".to_owned()),
                        ],
                    },
                    1.0,
                ),
                (
                    Query::Set {
                        field_id: 2,
                        values: vec![
                            QueryValue::U64(11),
                            QueryValue::U64(17),
                            QueryValue::U64(11),
                        ],
                    },
                    1.0,
                ),
                (
                    Query::Glob {
                        field_ids: vec![1],
                        pattern: "alp*".to_owned(),
                    },
                    1.0,
                ),
                (
                    Query::Glob {
                        field_ids: vec![1, 1],
                        pattern: "alp*".to_owned(),
                    },
                    2.0,
                ),
            ];
            for (query, expected_score) in &queries {
                let delta_evidence = execute_typed_query(&all_delta, &cx, query);
                let mixed_evidence = execute_typed_query(&mixed, &cx, query);
                let sealed_evidence = execute_typed_query(&all_sealed, &cx, query);
                assert_eq!(mixed_evidence, delta_evidence, "mixed query {query:?}");
                assert_eq!(sealed_evidence, delta_evidence, "sealed query {query:?}");
                assert_eq!(delta_evidence.1, vec![0, second_docid]);
                assert_eq!(delta_evidence.0.total_count, Some(2));
                assert!(
                    delta_evidence
                        .0
                        .hits
                        .iter()
                        .all(|hit| { hit.score.to_bits() == expected_score.to_bits() })
                );
            }

            for empty_range in [
                Query::Range {
                    field_id: 1,
                    lower: Bound::Included(QueryValue::Str("zeta".to_owned())),
                    upper: Bound::Included(QueryValue::Str("alpha".to_owned())),
                },
                Query::Range {
                    field_id: 1,
                    lower: Bound::Excluded(QueryValue::Str("alpha".to_owned())),
                    upper: Bound::Included(QueryValue::Str("alpha".to_owned())),
                },
            ] {
                let (ranked, docids) = execute_typed_query(&mixed, &cx, &empty_range);
                assert!(ranked.hits.is_empty());
                assert_eq!(ranked.total_count, Some(0));
                assert!(docids.is_empty());
            }

            let invalid_queries = [
                Query::Range {
                    field_id: 99,
                    lower: Bound::Included(QueryValue::U64(0)),
                    upper: Bound::Included(QueryValue::U64(1)),
                },
                Query::Range {
                    field_id: 2,
                    lower: Bound::Included(QueryValue::Str("wrong".to_owned())),
                    upper: Bound::Unbounded,
                },
                Query::Set {
                    field_id: 5,
                    values: vec![QueryValue::U64(11)],
                },
                Query::Boost {
                    query: Box::new(Query::All),
                    factor: f32::INFINITY,
                },
                Query::Phrase {
                    fields: vec![crate::query::QueryField::new(1, 1.0)],
                    terms: vec![
                        crate::query::PositionedTerm::new(0, "alpha"),
                        crate::query::PositionedTerm::new(1, "beta"),
                    ],
                    slop: 1,
                    prefix: false,
                },
                Query::Range {
                    field_id: 1,
                    lower: Bound::Included(QueryValue::Str(
                        "x".repeat(crate::grimoire::MAX_TERM_BYTES + 1),
                    )),
                    upper: Bound::Unbounded,
                },
            ];
            for query in &invalid_queries {
                let errors = [&empty, &all_delta, &mixed, &all_sealed].map(|index| {
                    let snapshot = index.search_snapshot();
                    let ranked = index
                        .execute_ranked_query(&cx, query, &snapshot, 10, 0, true, Vec::new())
                        .expect_err("invalid ranked query must fail before leaf iteration")
                        .to_string();
                    let docset = index
                        .execute_docid_query(&cx, query, &snapshot)
                        .expect_err("invalid docset query must fail before leaf iteration")
                        .to_string();
                    assert_eq!(ranked, docset, "collector validation parity for {query:?}");
                    ranked
                });
                assert!(
                    errors.windows(2).all(|pair| pair[0] == pair[1]),
                    "residency-independent validation for {query:?}: {errors:?}"
                );
            }

            let mut bounded = deterministic_config();
            bounded.glob_expansion_limit = 1;
            let bounded = typed_residency_index(&[], &[first, second], bounded);
            let snapshot = bounded.search_snapshot();
            let glob = Query::Glob {
                field_ids: vec![1],
                pattern: "alp*".to_owned(),
            };
            assert!(matches!(
                bounded.execute_ranked_query(&cx, &glob, &snapshot, 10, 0, true, Vec::new()),
                Err(QuillIndexError::Dictionary(
                    TermDictionaryError::GlobExpansionLimitExceeded {
                        field_ord: 1,
                        limit: 1,
                        actual: 2,
                    }
                ))
            ));
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn positionless_string_phrase_fails_with_typed_capability_before_execution() {
        let index =
            QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
                .expect("construct the QG positionless fixture");
        let cx = Cx::for_testing();

        let term_result = index
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("position-independent queries remain servable");
        assert!(term_result.hits.is_empty());
        assert_eq!(term_result.total_count, Some(0));

        let assert_positions_error = |error: &QuillIndexError| {
            assert!(matches!(
                error,
                QuillIndexError::QueryCapability(QueryCapabilityError::PositionsRequired {
                    schema: "qg-positionless-typed-capability-test",
                    field,
                    operator: QueryExplanation::Phrase,
                    capability: crate::query::IndexCapability::Positions,
                }) if field == "content"
            ));
        };
        for query in ["\"alpha beta\"", "\"alpha beta\"~2"] {
            let ranked = index
                .search_paginated(&cx, query, 10, 0, true)
                .expect_err("a position-dependent phrase must fail before query execution");
            assert_positions_error(&ranked);
            let docset = index
                .collect_docids(&cx, query)
                .expect_err("the scoreless path must preserve the capability error");
            assert_positions_error(&docset);
            let bench_ranked = index
                .bench_search_sealed_forced(&cx, query, 10, false, None)
                .expect_err("the QG ranked harness must reject before sealed collection");
            assert_positions_error(&bench_ranked);
            let bench_docset = index
                .bench_collect_docids_forced(&cx, query, false)
                .expect_err("the QG scoreless harness must reject before sealed collection");
            assert_positions_error(&bench_docset);
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn positionless_queries_match_positioned_ids_order_and_scores() {
        run_with_cx(|cx| async move {
            let positioned =
                QuillIndex::in_memory_with_schema(POSITIONED_QG_SCHEMA, deterministic_config())
                    .expect("construct the positioned control");
            let positionless =
                QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
                    .expect("construct the positionless fixture");
            let documents = fixture_documents();
            positioned
                .index_documents(&cx, &documents)
                .await
                .expect("index positioned control documents");
            positionless
                .index_documents(&cx, &documents)
                .await
                .expect("index positionless fixture documents");
            positioned
                .commit(&cx)
                .await
                .expect("publish positioned control");
            positionless
                .commit(&cx)
                .await
                .expect("publish positionless fixture");

            for query in [
                "rust",
                "rust OR python",
                "rust AND systems",
                "title:guide",
                "ord:[0 TO 1]",
            ] {
                let control = positioned
                    .search_paginated(&cx, query, 10, 0, true)
                    .unwrap_or_else(|error| {
                        panic!("positioned control failed for {query:?}: {error}")
                    });
                let candidate = positionless
                    .search_paginated(&cx, query, 10, 0, true)
                    .unwrap_or_else(|error| {
                        panic!("positionless fixture failed for {query:?}: {error}")
                    });
                assert_eq!(
                    candidate, control,
                    "positions must not affect ids, order, scores, or exact count for {query:?}"
                );
            }

            let phrase = positioned
                .search_paginated(&cx, "\"rust ownership\"", 10, 0, true)
                .expect("positioned phrase behavior remains available");
            assert!(
                phrase.hits.iter().any(|hit| hit.document_id == "rust-1"),
                "the positioned control must exercise a real phrase hit"
            );
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn positionless_basic_scoring_uses_presence_for_repeated_edge_ngrams() {
        run_with_cx(|cx| async move {
            let positioned =
                QuillIndex::in_memory_with_schema(POSITIONED_QG_SCHEMA, deterministic_config())
                    .expect("construct the positioned control");
            let positionless =
                QuillIndex::in_memory_with_schema(POSITIONLESS_QG_SCHEMA, deterministic_config())
                    .expect("construct the Basic fixture");
            assert!(
                positionless
                    .search_paginated(&cx, "token", 10, 0, true)
                    .expect("search empty Basic fixture")
                    .hits
                    .is_empty(),
                "an empty Basic field must have no term hits"
            );
            let repeated = crate::scribe::cass_generate_edge_ngrams("tokens tokens");
            let single = crate::scribe::cass_generate_edge_ngrams("tokens abcdef");
            assert_eq!(
                repeated.split_whitespace().count(),
                single.split_whitespace().count(),
                "control documents must retain equal field lengths"
            );
            let documents = vec![
                IndexableDocument::new("repeat", repeated).with_title("neutral"),
                IndexableDocument::new("single", single).with_title("neutral"),
            ];
            positioned
                .index_documents(&cx, &documents)
                .await
                .expect("index positioned control documents");
            positionless
                .index_documents(&cx, &documents)
                .await
                .expect("index Basic fixture documents");
            let assert_record_semantics = |positioned: &QuillSearchResult,
                                           positionless: &QuillSearchResult,
                                           phase: &str| {
                let score = |result: &QuillSearchResult, document_id: &str| {
                    result
                        .hits
                        .iter()
                        .find(|hit| hit.document_id == document_id)
                        .unwrap_or_else(|| panic!("{phase}: missing {document_id} hit"))
                        .score
                };
                let basic_repeat = score(positionless, "repeat");
                let basic_single = score(positionless, "single");
                let positioned_repeat = score(positioned, "repeat");
                let positioned_single = score(positioned, "single");

                assert_eq!(
                    basic_repeat.to_bits(),
                    basic_single.to_bits(),
                    "{phase}: Basic postings score document presence, not retained raw frequency"
                );
                assert_eq!(
                    basic_single.to_bits(),
                    positioned_single.to_bits(),
                    "{phase}: a unit frequency is unchanged across record options"
                );
                assert!(
                    positioned_repeat > positioned_single,
                    "{phase}: frequency-bearing postings retain the repeated edge n-gram boost"
                );
            };

            positioned
                .commit(&cx)
                .await
                .expect("publish positioned control");
            positionless
                .commit(&cx)
                .await
                .expect("publish Basic fixture");

            let positioned = positioned
                .search_paginated(&cx, "token", 10, 0, true)
                .expect("search positioned control");
            let positionless = positionless
                .search_paginated(&cx, "token", 10, 0, true)
                .expect("search Basic fixture");
            assert_record_semantics(&positioned, &positionless, "sealed");
        });
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn positionless_basic_scoring_survives_multiple_segments_and_reopen() {
        run_with_blocking_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("temporary Basic scoring directory");
            let writer = QuillWriterState::create_with_schema(
                &cx,
                directory.path(),
                POSITIONLESS_QG_SCHEMA,
                deterministic_config(),
            )
            .await
            .expect("create durable Basic writer");
            let index = QuillIndex::from_writer(writer);
            let repeated = crate::scribe::cass_generate_edge_ngrams("tokens tokens");
            let single = crate::scribe::cass_generate_edge_ngrams("tokens abcdef");

            index
                .index_document(
                    &cx,
                    &IndexableDocument::new("repeat", repeated).with_title("neutral"),
                )
                .await
                .expect("index repeated edge n-gram segment");
            index
                .commit(&cx)
                .await
                .expect("publish first Basic segment");
            index
                .index_document(
                    &cx,
                    &IndexableDocument::new("single", single).with_title("neutral"),
                )
                .await
                .expect("index single edge n-gram segment");
            index
                .commit(&cx)
                .await
                .expect("publish second Basic segment");
            assert_eq!(
                index.snapshot().segments().len(),
                2,
                "fixture must exercise composite scoring across two sealed segments"
            );

            let before_reopen = index
                .search_paginated(&cx, "token", 10, 0, true)
                .expect("search two-segment Basic fixture");
            assert_eq!(before_reopen.hits.len(), 2);
            assert_eq!(
                before_reopen.hits[0].score.to_bits(),
                before_reopen.hits[1].score.to_bits(),
                "Basic tf=1 must remain exact across sealed segments"
            );

            drop(index);
            let writer = KeeperWriter::open(&cx, directory.path(), POSITIONLESS_QG_SCHEMA)
                .await
                .expect("reopen custom-schema Keeper");
            let reopened = QuillIndex::from_backend(
                IndexBackend::Durable(writer),
                POSITIONLESS_QG_SCHEMA,
                deterministic_config(),
            )
            .expect("bind reopened Basic index");
            let after_reopen = reopened
                .search_paginated(&cx, "token", 10, 0, true)
                .expect("search reopened Basic fixture");
            assert_eq!(
                after_reopen, before_reopen,
                "reopen must preserve Basic ids, order, score bits, and exact count"
            );
        });
    }

    #[test]
    fn default_fast_only_ord_range_executes_and_validates_before_leaf_iteration() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let empty_set_error = index
                .search_paginated(&cx, "ord: IN [0 1]", 10, 0, true)
                .expect_err("fast-only set is unsupported before any leaf exists");
            assert!(matches!(
                &empty_set_error,
                QuillIndexError::UnsupportedQuery { detail }
                    if detail == "set names non-indexed numeric field 4"
            ));
            let empty_range = index
                .search_paginated(&cx, "ord:[0 TO 1]", 10, 0, true)
                .expect("fast-only range is valid against an empty snapshot");
            assert!(empty_range.hits.is_empty());
            assert_eq!(empty_range.total_count, Some(0));

            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");
            index.commit(&cx).await.expect("publish fixture");

            let cancelled = Cx::for_testing();
            cancelled.set_cancel_requested(true);
            let snapshot = index.search_snapshot();
            let segment = snapshot
                .keeper_snapshot()
                .segments()
                .first()
                .expect("committed fast-only segment");
            assert!(matches!(
                lower_leaf_numeric_range(
                    &cancelled,
                    QueryLeaf::Sealed(segment),
                    DEFAULT_SCHEMA,
                    ORD_FIELD,
                    &Bound::Included(QueryValue::U64(0)),
                    &Bound::Included(QueryValue::U64(1)),
                    1.0,
                    QueryLoweringMode::Scored,
                ),
                Err(QuillIndexError::Cancelled {
                    phase: "numeric range"
                })
            ));

            let ranked = index
                .search_paginated(&cx, "ord:[0 TO 1]", 10, 0, true)
                .expect("execute default fast-only range");
            assert_eq!(ranked.total_count, Some(2));
            assert_eq!(
                ranked
                    .hits
                    .iter()
                    .map(|hit| (hit.document_id.as_str(), hit.score.to_bits()))
                    .collect::<Vec<_>>(),
                vec![("rust-1", 1.0_f32.to_bits()), ("rust-2", 1.0_f32.to_bits())]
            );
            assert_eq!(
                index
                    .collect_docids(&cx, "ord:[0 TO 1]")
                    .expect("collect fast-only range docids"),
                vec![0, 1]
            );
            assert_eq!(
                index
                    .search_paginated(&cx, "ord:{0 TO 2}", 10, 0, true)
                    .expect("execute exclusive fast-only range")
                    .hits
                    .iter()
                    .map(|hit| hit.document_id.as_str())
                    .collect::<Vec<_>>(),
                vec!["rust-2"]
            );
            assert!(
                index
                    .search_paginated(&cx, "ord:[0 TO 1]^2.5", 10, 0, true)
                    .expect("execute boosted fast-only range")
                    .hits
                    .iter()
                    .all(|hit| hit.score.to_bits() == 2.5_f32.to_bits())
            );
            let populated_set_error = index
                .search_paginated(&cx, "ord: IN [0 1]", 10, 0, true)
                .expect_err("fast-only set remains unsupported with sealed leaves");
            assert_eq!(populated_set_error.to_string(), empty_set_error.to_string());
        });
    }

    #[test]
    fn compaction_preserves_q1_docids_and_live_query_identity() {
        run_with_blocking_cx(|cx| async move {
            let deleted = [1_u32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
            let index = q1_ob4_tombstoned_index(40, &deleted);
            let source = index.snapshot().clone();
            let source_manifest = source.loaded_manifest().manifest.clone();
            let source_segment = &source.segments()[0];
            let source_segment_id = source_segment.manifest().segment_id;
            let query_before = q1_ob4_query_evidence(&index, &cx);
            let live_docids_before = index
                .collect_docids(&cx, "shared")
                .expect("Q1-OB4 live docids before compaction");
            assert_eq!(source_segment.manifest().tombstones.cardinality(), 12);
            assert_eq!(source_segment.manifest().doc_count, 40);

            let report = index
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("Q1-OB4 in-memory compaction");
            assert_eq!(report.generation_before, source_manifest.generation);
            assert_eq!(report.generation_after, source_manifest.generation + 1);
            assert_eq!(report.examined_segments, 1);
            assert_eq!(report.compacted_segments, 1);
            assert_eq!(report.removed_segments, 0);
            assert_eq!(report.dropped_documents, 12);
            assert_eq!(report.input_bytes, source_segment.manifest().file_len);
            assert!(report.output_bytes > 0);

            let replacement_snapshot = index.snapshot();
            let replacement = &replacement_snapshot.segments()[0];
            assert_ne!(replacement.manifest().segment_id, source_segment_id);
            assert_eq!(replacement.manifest().docid_lo, 0);
            assert_eq!(replacement.manifest().docid_hi, 40);
            assert_eq!(replacement.manifest().doc_count, 28);
            assert_eq!(replacement.manifest().tombstones.cardinality(), 0);
            replacement.verify().expect("verify compacted Q1-OB4 FSLX");
            assert_eq!(
                index
                    .collect_docids(&cx, "shared")
                    .expect("Q1-OB4 live docids after compaction"),
                live_docids_before,
                "surviving global docids must not be renumbered",
            );
            assert_eq!(
                q1_ob4_query_evidence(&index, &cx),
                query_before,
                "ranked identities, order, global docids, counts, phrase matches, and fast-column ranges must be query-identical",
            );
            for global_docid in 0_u32..40 {
                let should_live = !deleted.contains(&global_docid);
                assert_eq!(index.snapshot().is_live(global_docid), should_live);
                assert_eq!(
                    index
                        .snapshot()
                        .materialize_document_id(global_docid)
                        .is_some(),
                    should_live,
                    "IDMAP/IDHASH hole mismatch at global docid {global_docid}",
                );
            }
            for kind in KNOWN_SECTION_KINDS {
                assert_eq!(
                    source_segment
                        .section(kind)
                        .expect("read Q1-OB4 source section")
                        .is_some(),
                    replacement
                        .section(kind)
                        .expect("read Q1-OB4 replacement section")
                        .is_some(),
                    "compaction changed section presence for kind {}",
                    kind.raw(),
                );
            }
            assert!(
                index
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .field_stats
                    .iter()
                    .all(|row| row.doc_count == 28),
                "post-compaction statistics must be re-derived over retained documents",
            );
        });
    }

    #[test]
    fn compaction_rederives_exact_token_totals_from_retained_postings() {
        run_with_blocking_cx(|cx| async move {
            let retained_content = std::iter::repeat_n("retained", 4_096)
                .collect::<Vec<_>>()
                .join(" ");
            let deleted_content = std::iter::repeat_n("deleted", 3_001)
                .collect::<Vec<_>>()
                .join(" ");
            let documents = vec![
                IndexableDocument::new("long-retained", retained_content),
                IndexableDocument::new("long-deleted", deleted_content),
            ];
            let sealed = seal_q1_ob2a_documents(&documents, 0, 0x0b40_1001);
            let field_stats =
                merge_field_stats(&[], &sealed.field_stats).expect("long source statistics");
            let committed = q1_ob2a_owned_index(vec![sealed.encoded], field_stats);
            let mut tombstoned = committed
                .snapshot()
                .next_manifest()
                .expect("long tombstone MANIFEST");
            assert!(
                committed
                    .snapshot()
                    .delete_document(&mut tombstoned, "long-deleted")
                    .expect("delete long document")
            );
            let tombstoned = committed
                .snapshot()
                .publish_owned_segments(&tombstoned, Vec::new())
                .expect("publish long tombstones");
            let compacted = QuillIndex::from_backend(
                IndexBackend::Memory(tombstoned),
                DEFAULT_SCHEMA,
                deterministic_config(),
            )
            .expect("bind long compaction fixture");

            let canonical = seal_q1_ob2a_documents(&documents[..1], 0, 0x0b40_1002);
            let canonical_stats = canonical.field_stats;
            let report = compacted
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("compact long fieldnorm fixture");
            assert!(report.changed());
            assert_eq!(
                compacted
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .field_stats
                    .iter()
                    .map(|row| (row.field_ord, (row.total_tokens, row.doc_count)))
                    .collect::<BTreeMap<_, _>>(),
                canonical_stats,
                "STATS must use exact retained posting frequencies, not quantized fieldnorm lengths",
            );
        });
    }

    #[test]
    fn compaction_density_boundary_is_strict_and_successor_is_idempotent() {
        run_with_blocking_cx(|cx| async move {
            let boundary_deleted = [0_u32, 1, 2, 3];
            let boundary = q1_ob4_tombstoned_index(20, &boundary_deleted);
            let boundary_generation = boundary.snapshot().loaded_manifest().manifest.generation;
            let boundary_segment_id = boundary.snapshot().segments()[0].manifest().segment_id;
            let report = boundary
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("20-percent boundary pass");
            assert!(!report.changed(), "density equality must not compact");
            assert_eq!(report.generation_after, boundary_generation);
            assert_eq!(
                boundary.snapshot().segments()[0].manifest().segment_id,
                boundary_segment_id,
            );

            let eligible_deleted = [0_u32, 1, 2, 3, 4];
            let eligible = q1_ob4_tombstoned_index(20, &eligible_deleted);
            let first = eligible
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("25-percent compaction");
            assert!(first.changed());
            let compacted_generation = first.generation_after;
            let compacted_segment_id = eligible.snapshot().segments()[0].manifest().segment_id;
            let second = eligible
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("idempotent successor pass");
            assert!(!second.changed());
            assert_eq!(second.generation_before, compacted_generation);
            assert_eq!(second.generation_after, compacted_generation);
            assert_eq!(
                eligible.snapshot().segments()[0].manifest().segment_id,
                compacted_segment_id,
                "a tombstone-free replacement must not thrash",
            );
        });
    }

    #[test]
    fn compaction_removes_fully_deleted_segments_and_rejects_invalid_policy() {
        run_with_blocking_cx(|cx| async move {
            let deleted = (0_u32..20).collect::<Vec<_>>();
            let index = q1_ob4_tombstoned_index(20, &deleted);
            let generation = index.snapshot().loaded_manifest().manifest.generation;
            let report = index
                .compact(&cx, CompactionPolicy::default())
                .await
                .expect("fully deleted compaction");
            assert_eq!(report.generation_after, generation + 1);
            assert_eq!(report.compacted_segments, 1);
            assert_eq!(report.removed_segments, 1);
            assert_eq!(report.dropped_documents, 20);
            assert_eq!(report.output_bytes, 0);
            assert!(index.snapshot().segments().is_empty());
            assert_eq!(index.snapshot().doc_count(), 0);
            assert!(
                index
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .field_stats
                    .iter()
                    .all(|row| row.doc_count == 0 && row.total_tokens == 0),
            );

            for density in [0.0, -0.1, 1.1, f64::INFINITY, f64::NAN] {
                let Err(error) = index
                    .snapshot()
                    .compact_owned(CompactionPolicy::new(density), 0)
                else {
                    panic!("invalid density {density:?} unexpectedly compacted");
                };
                assert!(matches!(
                    error,
                    KeeperError::Compaction {
                        source: CompactionError::InvalidDensity { .. }
                    }
                ));
            }
        });
    }

    #[test]
    fn durable_compaction_crash_boundary_keeps_old_manifest_authoritative() {
        run_with_blocking_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("temporary Q1-OB4 Keeper directory");
            let documents = q1_ob2a_documents();
            let mut index = QuillIndex::create(&cx, directory.path(), deterministic_config())
                .await
                .expect("create durable Q1-OB4 index");
            index
                .index_documents(&cx, &documents[..20])
                .await
                .expect("accumulate durable Q1-OB4 documents");
            index
                .commit(&cx)
                .await
                .expect("commit durable Q1-OB4 segment");
            let source_segment_id = index.snapshot().segments()[0].manifest().segment_id;
            let source_path = directory
                .path()
                .join(format!("seg-{source_segment_id:016x}.fslx"));
            let mut tombstoned = index
                .snapshot()
                .next_manifest()
                .expect("durable Q1-OB4 tombstone MANIFEST");
            for global_docid in 0_u32..5 {
                assert!(
                    index
                        .snapshot()
                        .delete_document(&mut tombstoned, &format!("q1-ob2a-{global_docid:03}"),)
                        .expect("stage durable Q1-OB4 tombstone"),
                );
            }
            match &mut index.writer_mut().backend {
                IndexBackend::Durable(writer) => {
                    writer
                        .publish(&cx, &tombstoned)
                        .await
                        .expect("publish durable Q1-OB4 tombstones");
                }
                IndexBackend::Memory(_) => panic!("durable fixture must own KeeperWriter"),
            }
            drop(index);
            let index = QuillIndex::open(&cx, directory.path(), deterministic_config())
                .await
                .expect("reopen durable tombstone generation");
            let old_manifest = index.snapshot().loaded_manifest().manifest.clone();
            let query_before = q1_ob4_query_evidence(&index, &cx);
            let pause = crate::keeper::pause_manifest_publish_at_checkpoint_for_test(
                directory.path(),
                crate::keeper::PublishCheckpoint::TempWritten,
            );
            let mut compact = Box::pin(index.compact(&cx, CompactionPolicy::default()));
            std::future::poll_fn(|task_cx| {
                if pause.is_reached() {
                    return Poll::Ready(());
                }
                match compact.as_mut().poll(task_cx) {
                    Poll::Pending => {
                        task_cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                    Poll::Ready(_) => panic!("compaction completed before crash checkpoint"),
                }
            })
            .await;

            let on_disk = Manifest::from_bytes(
                &std::fs::read(directory.path().join("MANIFEST"))
                    .expect("read authoritative pre-compaction MANIFEST"),
            )
            .expect("decode authoritative pre-compaction MANIFEST");
            assert_eq!(on_disk, old_manifest);
            assert!(
                source_path.exists(),
                "old immutable segment must remain intact"
            );
            let installed_segment_count = std::fs::read_dir(directory.path())
                .expect("inspect compaction crash directory")
                .filter_map(Result::ok)
                .filter(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("seg-") && name.ends_with(".fslx")
                })
                .count();
            assert_eq!(
                installed_segment_count, 2,
                "replacement output must be installed but still unreferenced",
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .expect("reopen the old authoritative crash-boundary snapshot");
            assert_eq!(reopened.loaded_manifest().manifest, old_manifest);
            assert_eq!(
                reopened.segments()[0].manifest().segment_id,
                source_segment_id
            );
            assert_eq!(
                reopened.segments()[0].manifest().tombstones.cardinality(),
                5
            );

            pause.release();
            let report = compact.await.expect("finish released durable compaction");
            assert!(report.changed());
            assert_eq!(q1_ob4_query_evidence(&index, &cx), query_before);
            assert!(
                source_path.exists(),
                "compaction does not delete old files inline"
            );
        });
    }

    #[test]
    fn adversarial_glob_and_phrase_queries_exhaust_fuel_deterministically() {
        run_with_cx(|cx| async move {
            let queries = [
                (
                    4,
                    (4, 4, 1, 3, 0, 0),
                    Query::Glob {
                        field_ids: vec![CONTENT_FIELD],
                        pattern: "rust*".to_owned(),
                    },
                ),
                (
                    7,
                    (7, 7, 1, 4, 2, 0),
                    Query::Phrase {
                        fields: vec![crate::query::QueryField::new(CONTENT_FIELD, 1.0)],
                        terms: vec![
                            crate::query::PositionedTerm::new(0, "rust"),
                            crate::query::PositionedTerm::new(1, "ownership"),
                        ],
                        slop: 0,
                        prefix: false,
                    },
                ),
            ];
            for (budget, expected, query) in &queries {
                let limited = QuillIndex::in_memory(QuillConfig {
                    query_fuel_budget: *budget,
                    deterministic_ingest: true,
                    ..QuillConfig::default()
                })
                .expect("memory index with bounded query fuel");
                limited
                    .index_documents(&cx, &fixture_documents())
                    .await
                    .expect("accumulate fixture");
                limited.commit(&cx).await.expect("publish fixture");
                let limited_snapshot = limited.search_snapshot();
                let first = limited
                    .execute_ranked_query(&cx, query, &limited_snapshot, 10, 0, true, Vec::new())
                    .expect_err("adversarial query must exhaust its boundary budget");
                let second = limited
                    .execute_ranked_query(&cx, query, &limited_snapshot, 10, 0, true, Vec::new())
                    .expect_err("repeated adversarial query must exhaust identically");
                let first = fuel_diagnostics(&first);
                assert_eq!(fuel_diagnostics(&second), first, "query={query:?}");
                assert_eq!(first, *expected, "query={query:?}");
            }

            let default = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            default
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate default-budget fixture");
            default
                .commit(&cx)
                .await
                .expect("publish default-budget fixture");
            let default_snapshot = default.search_snapshot();
            for (_, _, query) in &queries {
                default
                    .execute_ranked_query(&cx, query, &default_snapshot, 10, 0, true, Vec::new())
                    .expect("default fuel budget must cover fixture corpus");
                default
                    .execute_docid_query(&cx, query, &default_snapshot)
                    .expect("default fuel budget must cover fixture docset collection");
            }
        });
    }

    #[test]
    fn scalar_memory_commit_is_visibility_boundary_and_queries_end_to_end() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");

            let before = index
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("search old snapshot");
            assert!(before.hits.is_empty(), "uncommitted docs must be invisible");
            assert_eq!(before.total_count, Some(0));
            assert_eq!(before.doc_count, 0);

            index.commit(&cx).await.expect("publish owned snapshot");
            let term = index
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("term query");
            assert_eq!(term.total_count, Some(2));
            assert_eq!(term.doc_count, 3);
            assert_eq!(term.hits.len(), 2);
            assert!(
                term.hits
                    .iter()
                    .all(|hit| hit.document_id.starts_with("rust-"))
            );

            let phrase = index
                .search_paginated(&cx, "\"rust ownership\"", 10, 0, true)
                .expect("phrase query");
            assert_eq!(phrase.total_count, Some(1));
            assert_eq!(phrase.hits[0].document_id, "rust-1");

            let boolean = index
                .search_paginated(&cx, "rust AND systems", 10, 0, true)
                .expect("Boolean query");
            assert_eq!(boolean.total_count, Some(1));
            assert_eq!(boolean.hits[0].document_id, "rust-2");

            let page = index
                .search_paginated(&cx, "rust", 1, 1, true)
                .expect("paginated query");
            assert_eq!(page.total_count, Some(2));
            assert_eq!(page.hits.len(), 1);
        });
    }

    #[test]
    fn production_writer_routes_batches_across_resolved_shards_with_disjoint_leases() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                ..QuillConfig::default()
            };
            let detected = std::thread::available_parallelism().map_or(1, usize::from);
            let expected_shards = config.resolved_ingest_shards(detected);
            let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
            assert_eq!(
                index.writer_mut().shard_router.shard_count(),
                expected_shards
            );

            for batch in 0..expected_shards.max(2) {
                index
                    .index_documents(
                        &cx,
                        &[IndexableDocument::new(
                            format!("shard-{batch}"),
                            "cross shard tier policy fixture",
                        )],
                    )
                    .await
                    .expect("route one production batch");
            }
            index.commit(&cx).await.expect("publish routed batches");
            let snapshot = index.snapshot();
            assert_pairwise_disjoint_manifest(
                snapshot.loaded_manifest().manifest.segments.as_slice(),
            );
            if expected_shards > 1 {
                let ranges = snapshot
                    .segments()
                    .iter()
                    .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                    .collect::<Vec<_>>();
                assert!(
                    ranges
                        .iter()
                        .any(|(lo, _)| *lo >= u64::from(DOC_ORDS_PER_LEASE)),
                    "a resolved multi-shard writer must consume independent Q1 leases",
                );
            }
        });
    }

    #[test]
    fn large_batch_builds_all_available_shard_segments_shared_nothing() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
            let shard_count = index.writer_mut().shard_router.shard_count();
            let documents = (0..shard_count * PARALLEL_INGEST_MIN_DOCS_PER_SHARD)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("parallel-shard-{ordinal:05}"),
                        "parallel shared nothing indexing fixture",
                    )
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("accumulate one large batch");

            let active_shards = parallel_ingest_active_shards(
                documents.len(),
                shard_count,
                rayon::current_num_threads(),
            );
            let expected_segments = active_shards.max(1);
            index.commit(&cx).await.expect("publish parallel batch");
            assert_eq!(index.snapshot().segments().len(), expected_segments);
            assert_eq!(
                index.snapshot().doc_count(),
                u64::try_from(documents.len()).expect("fixture document count fits u64"),
            );
            assert_pairwise_disjoint_manifest(
                index
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .segments
                    .as_slice(),
            );
        });
    }

    #[test]
    fn adaptive_parallel_batch_activates_only_amortized_shards() {
        assert_eq!(parallel_ingest_active_shards(127, 4, 4), 1);
        assert_eq!(parallel_ingest_active_shards(250, 4, 4), 3);
        assert_eq!(parallel_ingest_active_shards(256, 4, 2), 2);
        assert_eq!(parallel_ingest_active_shards(5_000, 128, 128), 78);

        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("multi-shard memory index");
            let shard_count = index.writer_mut().shard_router.shard_count();
            let documents = (0..250)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("adaptive-shard-{ordinal:05}"),
                        "adaptive shared nothing indexing fixture",
                    )
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("accumulate one adaptive batch");

            let active_shards = parallel_ingest_active_shards(
                documents.len(),
                shard_count,
                rayon::current_num_threads(),
            );
            index.commit(&cx).await.expect("publish adaptive batch");
            assert_eq!(index.snapshot().segments().len(), active_shards.max(1));
            assert_eq!(index.snapshot().doc_count(), 250);
            assert_pairwise_disjoint_manifest(
                index
                    .snapshot()
                    .loaded_manifest()
                    .manifest
                    .segments
                    .as_slice(),
            );
        });
    }

    #[test]
    fn deterministic_batch_keeps_adaptive_parallel_route_disabled() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 4,
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let index = QuillIndex::in_memory(config).expect("deterministic memory index");
            let documents = (0..250)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("deterministic-shard-{ordinal:05}"),
                        "deterministic indexing fixture",
                    )
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("accumulate deterministic batch");
            index
                .commit(&cx)
                .await
                .expect("publish deterministic batch");
            assert_eq!(index.snapshot().segments().len(), 1);
            assert_eq!(index.snapshot().doc_count(), 250);
        });
    }

    #[test]
    fn bulk_cadence_publishes_intermediate_manifests_and_finish_leaves_one_segment() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                scribe_shard_budget_bytes: 1,
                deterministic_ingest: true,
                bulk_load_mode: true,
                bulk_publish_segment_cadence: 3,
                tier_fanout: 2,
                merge_max_hole_ratio: 1.0,
                ..QuillConfig::default()
            };
            let index = QuillIndex::in_memory(config).expect("bulk memory index");
            let documents = (0..7)
                .map(|ordinal| {
                    IndexableDocument::new(
                        format!("bulk-{ordinal}"),
                        format!("bulk cadence document {ordinal}"),
                    )
                })
                .collect::<Vec<_>>();
            index
                .index_documents(&cx, &documents)
                .await
                .expect("accumulate bulk fixture");

            let intermediate = index.snapshot();
            assert_eq!(intermediate.segments().len(), 6);
            assert_eq!(intermediate.doc_count(), 6);
            assert_ne!(
                intermediate.loaded_manifest().manifest.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
                0,
            );
            assert!(index.has_uncommitted_changes());

            let finished = index
                .finish_bulk_load(&cx)
                .await
                .expect("finish bulk fixture");
            assert_eq!(finished.segments().len(), 1);
            assert_eq!(finished.doc_count(), 7);
            assert_eq!(
                finished.loaded_manifest().manifest.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
                0,
            );
            assert!(!index.has_uncommitted_changes());
            assert_eq!(
                index
                    .search_paginated(&cx, "bulk", 10, 0, true)
                    .expect("search finished bulk index")
                    .total_count,
                Some(7),
            );
        });
    }

    #[test]
    fn successful_sealed_batches_compose_but_failed_batches_require_commit_retry() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                scribe_shard_budget_bytes: 1,
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let mut index = QuillIndex::in_memory(config).expect("bounded-batch memory index");

            for ordinal in 0..2 {
                index
                    .index_documents(
                        &cx,
                        &[IndexableDocument::new(
                            format!("bounded-{ordinal}"),
                            "bounded batch searchable",
                        )],
                    )
                    .await
                    .expect("successful seal-producing batch");
            }
            assert_eq!(index.doc_count(), 0, "segments remain unpublished");
            assert!(index.has_uncommitted_changes());

            let duplicate_batch = [
                IndexableDocument::new("bounded-2", "bounded batch searchable"),
                IndexableDocument::new("bounded-2", "bounded duplicate"),
            ];
            assert!(
                index.index_documents(&cx, &duplicate_batch).await.is_err(),
                "duplicate batch must fail after accepting its unique prefix"
            );
            assert!(index.writer_mut().ingest_retry_required);
            assert!(
                index
                    .index_documents(
                        &cx,
                        &[IndexableDocument::new(
                            "bounded-3",
                            "bounded batch searchable",
                        )],
                    )
                    .await
                    .is_err(),
                "an ambiguous partial batch must fail closed until commit"
            );

            index.commit(&cx).await.expect("reconcile partial batch");
            assert_eq!(index.doc_count(), 3);
            assert!(!index.writer_mut().ingest_retry_required);

            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "bounded-3",
                        "bounded batch searchable",
                    )],
                )
                .await
                .expect("index after retry commit");
            index.commit(&cx).await.expect("final bounded-batch commit");
            assert_eq!(
                index
                    .search_paginated(&cx, "searchable", 10, 0, true)
                    .expect("search committed bounded batches")
                    .total_count,
                Some(4),
            );
        });
    }

    #[test]
    fn commit_applies_tier_policy_to_cross_shard_bound_consecutive_run() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                max_ingest_shards: 2,
                tier_fanout: 2,
                merge_max_hole_ratio: 1.0,
                ..QuillConfig::default()
            };
            let index = QuillIndex::in_memory(config).expect("tier policy memory index");
            index
                .index_documents(&cx, &[IndexableDocument::new("tier-a", "shared alpha")])
                .await
                .expect("route first tier batch");
            index
                .index_documents(&cx, &[IndexableDocument::new("tier-b", "shared beta")])
                .await
                .expect("route second tier batch");
            index.commit(&cx).await.expect("publish and tier merge");

            let snapshot = index.snapshot();
            assert_eq!(snapshot.segments().len(), 1);
            assert_eq!(snapshot.doc_count(), 2);
            let merged_hi = snapshot.segments()[0].manifest().docid_hi;
            assert_eq!(
                index
                    .search_paginated(&cx, "shared", 10, 0, true)
                    .expect("query tier-merged snapshot")
                    .total_count,
                Some(2),
            );

            index
                .index_documents(&cx, &[IndexableDocument::new("tier-after", "shared gamma")])
                .await
                .expect("append after tier merge");
            index.commit(&cx).await.expect("publish post-merge append");
            let post_merge = index.snapshot();
            assert_pairwise_disjoint_manifest(
                post_merge.loaded_manifest().manifest.segments.as_slice(),
            );
            assert!(
                post_merge
                    .segments()
                    .iter()
                    .any(|segment| segment.manifest().docid_lo >= merged_hi),
                "lease retirement must place future batches above the merged hull",
            );
        });
    }

    #[test]
    fn visibility_lag_uses_the_same_lifecycle_commit_boundary() {
        run_with_cx(|cx| async move {
            let config = QuillConfig {
                deterministic_ingest: true,
                max_visibility_lag_ms: 1,
                ..QuillConfig::default()
            };
            let index = QuillIndex::in_memory(config).expect("visibility memory index");
            index
                .index_documents(&cx, &[IndexableDocument::new("lag-a", "visibility alpha")])
                .await
                .expect("accumulate first visibility batch");
            assert_eq!(index.doc_count(), 0, "sub-bound changes stay unpublished");

            std::thread::sleep(Duration::from_millis(2));
            index
                .index_documents(&cx, &[IndexableDocument::new("lag-b", "visibility beta")])
                .await
                .expect("lag-bound batch forces publication");
            assert_eq!(index.doc_count(), 2);
            assert!(!index.has_uncommitted_changes());
        });
    }

    #[test]
    fn scoreless_docset_collector_is_wired_across_committed_segments() {
        run_with_cx(|cx| async move {
            let documents = fixture_documents();
            let index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &documents[..2])
                .await
                .expect("first segment documents");
            index.commit(&cx).await.expect("first segment commit");
            index
                .index_documents(&cx, &documents[2..])
                .await
                .expect("second segment documents");

            assert_eq!(
                index
                    .collect_docids(&cx, "rust OR python")
                    .expect("old committed doc set")
                    .len(),
                2,
                "uncommitted second-segment document must remain invisible",
            );

            index.commit(&cx).await.expect("second segment commit");
            let docids = index
                .collect_docids(&cx, "rust OR python")
                .expect("Boolean doc set");
            let ranked = index
                .search_paginated(&cx, "rust OR python", 10, 0, true)
                .expect("ranked Boolean search");
            let mut ranked_docids = ranked
                .hits
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>();
            ranked_docids.sort_unstable();
            assert_eq!(docids, ranked_docids);
            assert_eq!(docids.len(), 3);
            assert_eq!(ranked.total_count, Some(3));
        });
    }

    #[test]
    fn in_memory_concat_merge_preserves_query_results_scores_and_docids() {
        run_with_cx(|cx| async move {
            let index = concat_merge_fixture_index(&cx).await;
            let source_ids = committed_segment_ids(&index);
            assert_eq!(source_ids.len(), 3);

            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            let results_before = concat_merge_query_results(&index, &cx);
            let docids_before = index
                .collect_docids(&cx, "rust OR python")
                .expect("collect pre-merge docids");
            let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0001);
            let source_snapshot = index.snapshot();
            let source_dictionary_counts = source_snapshot
                .segments()
                .iter()
                .map(RecoveredSegment::term_dictionary_cache_counts)
                .collect::<Vec<_>>();

            index
                .concat_merge(&cx, &source_ids, output_segment_id, 17)
                .await
                .expect("merge the complete in-memory leaf run");

            for (segment, (full_validations_before, borrowed_views_before)) in source_snapshot
                .segments()
                .iter()
                .zip(source_dictionary_counts)
            {
                let (full_validations_after, borrowed_views_after) =
                    segment.term_dictionary_cache_counts();
                assert_eq!(
                    full_validations_after, full_validations_before,
                    "concat must not repeat complete TERMDICT validation"
                );
                assert_eq!(
                    borrowed_views_after,
                    borrowed_views_before + 1,
                    "concat must open one cached dictionary view per source segment"
                );
            }
            assert_eq!(index.snapshot().segments().len(), 1);
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation_before + 1,
            );
            assert_eq!(
                concat_merge_query_results(&index, &cx),
                results_before,
                "ranked hits, exact scores, global docids, and counts must survive merge",
            );
            assert_eq!(
                index
                    .collect_docids(&cx, "rust OR python")
                    .expect("collect post-merge docids"),
                docids_before,
            );
            let merge_seal_seq = index.snapshot().segments()[0].manifest().seal_seq;

            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "after-merge",
                        "ordinary sealing remains available after concat merge",
                    )],
                )
                .await
                .expect("accumulate after concat merge");
            index
                .commit(&cx)
                .await
                .expect("ordinary commit after concat merge");
            assert_eq!(
                index
                    .snapshot()
                    .segments()
                    .iter()
                    .map(|segment| segment.manifest().seal_seq)
                    .collect::<Vec<_>>(),
                [merge_seal_seq, merge_seal_seq + 1],
                "concat output and the next ordinary seal need distinct monotone sequences",
            );
        });
    }

    #[test]
    fn concat_merge_matches_fresh_monolithic_df_100_plus_300_with_same_docids() {
        run_with_cx(|cx| async move {
            let (leaves, monolithic) = q1_ob2a_fixture_indexes();
            assert_eq!(
                leaves
                    .snapshot()
                    .segments()
                    .iter()
                    .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                    .collect::<Vec<_>>(),
                [(0, 100), (100, 400)],
                "Q1-OB2a leaves must be adjacent inside one lease",
            );
            assert_eq!(monolithic.snapshot().segments()[0].manifest().docid_lo, 0);
            assert_eq!(monolithic.snapshot().segments()[0].manifest().docid_hi, 400);

            let source_shared_dfs = leaves
                .snapshot()
                .segments()
                .iter()
                .map(|segment| {
                    open_dictionary(segment, DEFAULT_SCHEMA)
                        .expect("source Q1-OB2a TERMDICT")
                        .lookup(CONTENT_FIELD, b"shared")
                        .expect("source shared lookup")
                        .expect("source shared term")
                        .metadata
                        .doc_freq
                })
                .collect::<Vec<_>>();
            assert_eq!(source_shared_dfs, [100, 300]);
            assert_eq!(
                snapshot_doc_freq(
                    &monolithic.snapshot(),
                    DEFAULT_SCHEMA,
                    CONTENT_FIELD,
                    b"shared",
                )
                .expect("monolithic shared df"),
                400,
            );

            let source_stats = leaves
                .snapshot()
                .segments()
                .iter()
                .map(q1_ob2a_decoded_stats)
                .collect::<Vec<_>>();
            let source_aggregate =
                aggregate_field_stats(source_stats.iter()).expect("aggregate source STATS");
            let monolithic_stats = q1_ob2a_decoded_stats(&monolithic.snapshot().segments()[0]);
            let monolithic_aggregate = monolithic_stats
                .rows()
                .iter()
                .map(|row| SnapshotFieldStats {
                    field_ord: row.field_ord,
                    total_tokens: row.total_tokens,
                    doc_count: u64::from(row.doc_count),
                })
                .collect::<Vec<_>>();
            assert_eq!(source_aggregate, monolithic_aggregate);
            assert_eq!(
                leaves.snapshot().loaded_manifest().manifest.field_stats,
                monolithic.snapshot().loaded_manifest().manifest.field_stats,
            );

            let monolithic_terms = q1_ob2a_decoded_terms(&monolithic);
            assert_eq!(
                q1_ob2a_decoded_terms(&leaves),
                monolithic_terms,
                "source terms and decoded postings must match fresh monolithic indexing",
            );
            let monolithic_evidence = q1_ob2a_query_evidence(&monolithic, &cx);
            assert_eq!(
                q1_ob2a_query_evidence(&leaves, &cx),
                monolithic_evidence,
                "source leaves and fresh monolithic seal must have exact scores, ids, docids, and counts",
            );
            assert_eq!(
                leaves
                    .collect_docids(&cx, "shared")
                    .expect("source shared docids"),
                (0_u32..400).collect::<Vec<_>>(),
            );

            let source_ids = committed_segment_ids(&leaves);
            leaves
                .concat_merge(&cx, &source_ids, 0x0b2a_0004, 0)
                .await
                .expect("Q1-OB2a concat merge");
            assert_eq!(leaves.snapshot().segments().len(), 1);
            assert_eq!(
                snapshot_doc_freq(&leaves.snapshot(), DEFAULT_SCHEMA, CONTENT_FIELD, b"shared")
                    .expect("merged shared df"),
                400,
            );
            assert_eq!(
                q1_ob2a_decoded_terms(&leaves),
                monolithic_terms,
                "every merged term and decoded posting must match fresh monolithic indexing",
            );
            assert_eq!(
                q1_ob2a_decoded_stats(&leaves.snapshot().segments()[0]),
                monolithic_stats,
                "merged decoded STATS must match fresh monolithic indexing",
            );
            assert_eq!(
                leaves.snapshot().loaded_manifest().manifest.field_stats,
                monolithic.snapshot().loaded_manifest().manifest.field_stats,
            );
            assert_eq!(
                q1_ob2a_query_evidence(&leaves, &cx),
                monolithic_evidence,
                "merged and monolithic query scores, ids, docids, counts, and docsets must match",
            );
        });
    }

    #[test]
    fn concat_merge_rejects_reversed_and_skipped_runs_without_publication() {
        run_with_cx(|cx| async move {
            let index = concat_merge_fixture_index(&cx).await;
            let source_ids = committed_segment_ids(&index);
            assert_eq!(source_ids.len(), 3);
            let manifest_before = index.snapshot().loaded_manifest().manifest.clone();
            let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0010);

            let reversed = [source_ids[1], source_ids[0]];
            let Err(error) = index
                .concat_merge(&cx, &reversed, output_segment_id, 19)
                .await
            else {
                panic!("reversed concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::NonConsecutiveSources { .. }
                })
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "reversed rejection must not advance the snapshot generation",
            );

            let skipped = [source_ids[0], source_ids[2]];
            let Err(error) = index
                .concat_merge(&cx, &skipped, output_segment_id, 23)
                .await
            else {
                panic!("skipped concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::NonConsecutiveSources { .. }
                })
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "skipped-run rejection must leave the exact snapshot unchanged",
            );

            let wrapped = [source_ids[1], source_ids[2], source_ids[0]];
            let Err(error) = index
                .concat_merge(&cx, &wrapped, output_segment_id, 29)
                .await
            else {
                panic!("manifest-wrapping concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::SourceRunPastManifest {
                        position: 2,
                        actual,
                    }
                }) if actual == source_ids[0]
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "wrapped-run rejection must leave the exact snapshot unchanged",
            );
        });
    }

    #[test]
    fn concat_merge_direct_and_associated_schedules_match() {
        run_with_cx(|cx| async move {
            let direct = concat_merge_fixture_index(&cx).await;
            let left_associated = concat_merge_fixture_index(&cx).await;
            let right_associated = concat_merge_fixture_index(&cx).await;
            let leaf_ids = committed_segment_ids(&direct);
            assert_eq!(leaf_ids, committed_segment_ids(&left_associated));
            assert_eq!(leaf_ids, committed_segment_ids(&right_associated));
            assert_eq!(leaf_ids.len(), 3);

            let results_before = concat_merge_query_results(&direct, &cx);
            assert_eq!(
                concat_merge_query_results(&left_associated, &cx),
                results_before,
            );
            assert_eq!(
                concat_merge_query_results(&right_associated, &cx),
                results_before,
            );
            let final_segment_id = fresh_merge_segment_id(&direct, 0xc011_ca7e_0000_0020);
            let intermediate_segment_id =
                fresh_merge_segment_id(&left_associated, 0xc011_ca7e_0000_0030);
            let right_intermediate_segment_id =
                fresh_merge_segment_id(&right_associated, 0xc011_ca7e_0000_0040);

            direct
                .concat_merge(&cx, &leaf_ids, final_segment_id, 31)
                .await
                .expect("direct three-leaf concat merge");
            left_associated
                .concat_merge(&cx, &leaf_ids[..2], intermediate_segment_id, 29)
                .await
                .expect("left-associated first concat merge");
            let left_final_sources = [intermediate_segment_id, leaf_ids[2]];
            left_associated
                .concat_merge(&cx, &left_final_sources, final_segment_id, 31)
                .await
                .expect("left-associated final concat merge");
            right_associated
                .concat_merge(&cx, &leaf_ids[1..], right_intermediate_segment_id, 29)
                .await
                .expect("right-associated first concat merge");
            let right_final_sources = [leaf_ids[0], right_intermediate_segment_id];
            right_associated
                .concat_merge(&cx, &right_final_sources, final_segment_id, 31)
                .await
                .expect("right-associated final concat merge");

            assert_eq!(direct.snapshot().segments().len(), 1);
            assert_eq!(left_associated.snapshot().segments().len(), 1);
            assert_eq!(right_associated.snapshot().segments().len(), 1);
            let direct_snapshot = direct.snapshot();
            let left_snapshot = left_associated.snapshot();
            let right_snapshot = right_associated.snapshot();
            let direct_segment = &direct_snapshot.segments()[0];
            let left_segment = &left_snapshot.segments()[0];
            let right_segment = &right_snapshot.segments()[0];
            assert_eq!(
                direct_segment.source_bytes(),
                left_segment.source_bytes(),
                "the complete merged FSLX image must be schedule-independent",
            );
            assert_eq!(
                direct_segment.source_bytes(),
                right_segment.source_bytes(),
                "right-associated merge must emit the same complete FSLX image",
            );
            for kind in KNOWN_SECTION_KINDS {
                assert_eq!(
                    direct_segment.section(kind).expect("direct merged section"),
                    left_segment
                        .section(kind)
                        .expect("left-associated merged section"),
                    "section {} differs by merge schedule",
                    kind.raw(),
                );
                assert_eq!(
                    direct_segment.section(kind).expect("direct merged section"),
                    right_segment
                        .section(kind)
                        .expect("right-associated merged section"),
                    "section {} differs under right association",
                    kind.raw(),
                );
            }

            let direct_results = concat_merge_query_results(&direct, &cx);
            let left_results = concat_merge_query_results(&left_associated, &cx);
            let right_results = concat_merge_query_results(&right_associated, &cx);
            assert_eq!(direct_results, results_before);
            assert_eq!(left_results, results_before);
            assert_eq!(right_results, results_before);
        });
    }

    #[test]
    fn deterministic_runs_have_identical_segments_and_results() {
        run_with_cx(|cx| async move {
            let documents = fixture_documents();
            let first = QuillIndex::in_memory(deterministic_config()).expect("first index");
            let second = QuillIndex::in_memory(deterministic_config()).expect("second index");
            first
                .index_documents(&cx, &documents)
                .await
                .expect("first ingest");
            second
                .index_documents(&cx, &documents)
                .await
                .expect("second ingest");
            first.commit(&cx).await.expect("first commit");
            second.commit(&cx).await.expect("second commit");

            assert_eq!(
                first.snapshot().loaded_manifest().manifest,
                second.snapshot().loaded_manifest().manifest
            );
            let first_snapshot = first.snapshot();
            let second_snapshot = second.snapshot();
            let first_segment = &first_snapshot.segments()[0];
            let second_segment = &second_snapshot.segments()[0];
            assert_eq!(first_segment.header(), second_segment.header());
            assert_eq!(first_segment.source_bytes(), second_segment.source_bytes());
            for kind in [
                SectionKind::TERMDICT,
                SectionKind::POSTINGS,
                SectionKind::POSITIONS,
                SectionKind::BLOCKMAX,
                SectionKind::DOCLEN,
                SectionKind::IDMAP,
                SectionKind::IDHASH,
                SectionKind::STOREDMETA,
                SectionKind::STATS,
            ] {
                assert_eq!(
                    first_segment.section(kind).expect("first section"),
                    second_segment.section(kind).expect("second section"),
                    "section {}",
                    kind.raw()
                );
            }

            let first_result = first
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("first search");
            let second_result = second
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("second search");
            assert_eq!(first_result, second_result);
        });
    }

    #[test]
    fn unaligned_manifest_watermark_burns_the_partial_lease() {
        let lease_size = u64::from(DOC_ORDS_PER_LEASE);
        assert_eq!(next_lease_boundary(0).expect("genesis"), 0);
        assert_eq!(
            next_lease_boundary(1).expect("partial first lease"),
            lease_size
        );
        assert_eq!(
            next_lease_boundary(lease_size).expect("aligned lease"),
            lease_size
        );
        assert_eq!(
            next_lease_boundary(lease_size + 1).expect("partial second lease"),
            lease_size * 2
        );
        assert!(next_lease_boundary(MAX_GLOBAL_DOCID_EXCLUSIVE - 1).is_ok());
        assert!(next_lease_boundary(MAX_GLOBAL_DOCID_EXCLUSIVE + 1).is_err());
    }

    #[test]
    fn field_stat_overflow_fails_before_segment_installation() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .writer_mut()
                .pending_field_stats
                .insert(ID_FIELD, (u64::MAX, 0));
            index
                .index_documents(&cx, &[IndexableDocument::new("overflow", "one token")])
                .await
                .expect("accumulate document");

            let Err(error) = index.commit(&cx).await else {
                panic!("field stats overflow unexpectedly committed");
            };
            assert!(matches!(error, QuillIndexError::InvalidState { .. }));
            let writer = index.writer_mut();
            assert_eq!(writer.shards[0].accumulator.document_count(), 1);
            assert!(writer.staged_flush.is_none());
            assert!(writer.pending_segments.is_empty());
            assert!(writer.pending_owned_segments.is_empty());
        });
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_manifest_retry_reuses_stamped_bytes_after_clock_advances() {
        let directory = tempfile::tempdir().expect("index tempdir").keep();
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::create(&cx, &directory, deterministic_config())
                .await
                .expect("create filesystem index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");
            index
                .writer_mut()
                .flush_shard(&cx, 0, LifecycleTrigger::ExplicitFlush)
                .await
                .expect("install immutable segment");
            index
                .writer_mut()
                .prepare_pending_manifest()
                .expect("retain exact MANIFEST proposal");
            let proposal = index
                .writer_mut()
                .pending_manifest
                .as_ref()
                .expect("proposal retained")
                .clone();
            assert!(proposal.last_publish_unix_s > 0);
            let proposal_bytes = proposal.to_bytes().expect("encode retained proposal");
            let temp_path = directory.join(format!(".tmp-manifest-{}", proposal.generation));
            std::fs::write(&temp_path, &proposal_bytes).expect("seed synced-attempt image");

            std::thread::sleep(std::time::Duration::from_millis(1_100));
            index
                .commit(&cx)
                .await
                .expect("reuse exact prior-attempt bytes");

            assert_eq!(index.snapshot().loaded_manifest().manifest, proposal);
            assert!(index.writer_mut().pending_manifest.is_none());
        });
    }

    #[cfg(unix)]
    #[test]
    fn restarted_ingest_avoids_differing_abandoned_segment_id() {
        let directory = tempfile::tempdir().expect("index tempdir").keep();
        run_with_cx(|cx| async move {
            let abandoned_segment_id = {
                let mut first = QuillIndex::create(&cx, &directory, deterministic_config())
                    .await
                    .expect("create first writer");
                first
                    .index_documents(
                        &cx,
                        &[IndexableDocument::new("abandoned", "old orphan content")],
                    )
                    .await
                    .expect("accumulate abandoned batch");
                first
                    .writer_mut()
                    .flush_shard(&cx, 0, LifecycleTrigger::ExplicitFlush)
                    .await
                    .expect("install segment without MANIFEST");
                assert_eq!(first.snapshot().doc_count(), 0);
                first.writer_mut().pending_segments[0].segment_id
            };
            let abandoned_path = directory.join(format!("seg-{abandoned_segment_id:016x}.fslx"));
            assert!(
                abandoned_path.exists(),
                "orphan must survive writer restart"
            );

            let restarted = QuillIndex::open(&cx, &directory, deterministic_config())
                .await
                .expect("reopen old committed generation");
            restarted
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "replacement",
                        "fresh replacement content",
                    )],
                )
                .await
                .expect("accumulate replacement batch");
            restarted
                .commit(&cx)
                .await
                .expect("publish replacement batch");

            let restarted_snapshot = restarted.snapshot();
            let committed = &restarted_snapshot.loaded_manifest().manifest;
            assert_eq!(committed.segments.len(), 1);
            assert_ne!(committed.segments[0].segment_id, abandoned_segment_id);
            assert!(
                abandoned_path.exists(),
                "grace-window orphan remains intact"
            );
            let replacement = restarted
                .search_paginated(&cx, "replacement", 10, 0, true)
                .expect("search replacement batch");
            assert_eq!(replacement.total_count, Some(1));
            assert_eq!(replacement.hits[0].document_id, "replacement");
            let abandoned = restarted
                .search_paginated(&cx, "abandoned", 10, 0, true)
                .expect("search committed snapshot only");
            assert_eq!(abandoned.total_count, Some(0));
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_facade_protects_commit_and_reopens_searchable_snapshot() {
        let directory = tempfile::tempdir().expect("durable index tempdir").keep();
        run_with_cx(|cx| async move {
            let protector = test_file_protector();
            let index = QuillIndex::create_durable(
                &cx,
                &directory,
                deterministic_config(),
                protector.clone(),
            )
            .await
            .expect("create durable index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("durable ingest");
            index.commit(&cx).await.expect("durable commit");
            let segment_path = index.snapshot().segments()[0].path().to_path_buf();
            drop(index);

            let manifest_path = directory.join("MANIFEST");
            assert!(
                protector
                    .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))
                    .expect("verify MANIFEST sidecar")
                    .healthy
            );
            assert!(
                protector
                    .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))
                    .expect("verify FSLX sidecar")
                    .healthy
            );

            let reopened =
                QuillIndex::open_durable(&cx, &directory, deterministic_config(), protector)
                    .await
                    .expect("reopen durable index");
            let result = reopened
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("search reopened durable index");
            assert_eq!(result.total_count, Some(2));
            assert_eq!(result.doc_count, 3);
        });
    }

    /// bd-8nqz.1 flagship: a retained candidate batch hydrates from the
    /// EXACT snapshot that scored it, even after a newer generation commits —
    /// while the legacy combined-trait path demonstrably reads the newer
    /// generation (the race the split exists to fix). Also pins the typed
    /// rejection of missing and foreign hydration contexts.
    #[test]
    fn lexical_read_hydration_is_pinned_to_the_scoring_generation() {
        run_with_cx(|cx| async move {
            let index = QuillIndex::in_memory(QuillConfig::default()).expect("in-memory index");
            let doc_v1 =
                IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v1");
            index
                .index_documents(&cx, std::slice::from_ref(&doc_v1))
                .await
                .expect("index generation N");
            index.commit(&cx).await.expect("publish generation N");

            // Score on generation N; candidates defer metadata.
            let batch = LexicalRead::search_candidates(&index, &cx, "pinned", 10)
                .await
                .expect("candidates on generation N");
            assert!(batch.is_deferred(), "Quill candidates must carry a pin");
            assert_eq!(batch.results().len(), 1);
            assert!(
                batch.results()[0].metadata.is_none(),
                "candidate metadata is deferred until hydration"
            );
            let mut legacy_path = batch.results().to_vec();

            // Publish generation N+1 with different metadata for the same doc.
            let doc_v2 =
                IndexableDocument::new("doc-a", "alpha pinned content").with_metadata("rev", "v2");
            LexicalWrite::index_document(&index, &cx, &doc_v2)
                .await
                .expect("upsert generation N+1");
            LexicalWrite::commit(&index, &cx)
                .await
                .expect("publish generation N+1");

            let rev_of = |result: &ScoredResult| -> Option<String> {
                result.metadata.as_ref().and_then(|metadata| {
                    metadata
                        .get("rev")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned)
                })
            };

            // A fresh search observes N+1.
            let fresh = LexicalRead::search(&index, &cx, "pinned", 10)
                .await
                .expect("fresh search on N+1");
            assert_eq!(rev_of(&fresh[0]).as_deref(), Some("v2"));

            // The retained batch hydrates from its PINNED generation N.
            let (mut winners, context) = batch.into_parts();
            LexicalRead::hydrate_candidates(&index, &cx, context.as_ref(), &mut winners)
                .await
                .expect("pinned hydration");
            assert_eq!(
                rev_of(&winners[0]).as_deref(),
                Some("v1"),
                "hydration must read the scoring snapshot, not the newest one"
            );

            // Differential control: the legacy combined-trait hydration reads
            // the CURRENT snapshot and returns v2 for the same retained
            // candidates — the exact generation race.
            LexicalSearch::hydrate_fusion_metadata(&index, &cx, &mut legacy_path)
                .await
                .expect("legacy hydration");
            assert_eq!(
                rev_of(&legacy_path[0]).as_deref(),
                Some("v2"),
                "legacy path demonstrates the race the split fixes"
            );

            // Missing context: typed rejection, never a silent no-op.
            let missing = LexicalRead::hydrate_candidates(&index, &cx, None, &mut winners)
                .await
                .expect_err("deferred hydration without a context must fail");
            assert!(
                matches!(missing, SearchError::SubsystemError { subsystem, .. }
                    if subsystem == "quill.hydration"),
            );

            // Foreign context: typed rejection of cross-engine mixing.
            let foreign = LexicalHydrationContext::new("not-quill", Box::new(7_u32));
            let mixed = LexicalRead::hydrate_candidates(&index, &cx, Some(&foreign), &mut winners)
                .await
                .expect_err("foreign hydration context must fail");
            assert!(
                matches!(mixed, SearchError::SubsystemError { subsystem, .. }
                    if subsystem == "quill.hydration"),
            );
        });
    }
}
