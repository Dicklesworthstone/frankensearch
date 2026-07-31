//! Argus query execution.
//!
//! This module contains the deliberately exhaustive scorer used as the
//! correctness anchor for every later pruning path. It mirrors the pinned
//! Tantivy scorer-tree arithmetic, including buffered `Should` unions, while
//! retaining Quill's explicit `Option<u32>` cursor state so `u32::MAX` remains
//! a searchable global document id.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use frankensearch_core::DocId;
use thiserror::Error;

use crate::contract::{BM25_K1, compute_tf_cache, idf};
use crate::delta::{DeltaBlockMax, DeltaPosting, DeltaPostings, DeltaSnapshot, DeltaTerm};
use crate::quiver::{
    BlockMaxEntry, DocLenField, NumericCodecError, NumericDocIdSet, NumericField, NumericValue,
    PositionCodecError, PostingCodecError, SnapshotFieldStats, ValidatedTermPruningMetadata,
};

pub use crate::query::Occur;

const UNION_HORIZON: usize = 4_096;
const UNION_HORIZON_U64: u64 = 4_096;
pub(crate) const MAX_SCORE_MAX_CLAUSES: usize = 8;
pub(crate) const BMW_MIN_CLAUSES: usize = MAX_SCORE_MAX_CLAUSES + 1;
const BMW_MIN_TOTAL_COST: u64 = 16_384;
const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;

/// Owner-bound decoder for one cursor's compressed position runs.
///
/// Sealed and delta cursors implement this against their own immutable
/// storage. Callers cannot resolve an ordinal against an unrelated segment.
pub trait PositionsReader {
    /// Decode one posting's absolute positions into an already-cleared buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError>;
}

/// Borrowed capability for the current posting's later POSITIONS lookup.
///
/// The immutable borrow prevents its cursor from moving before the handle is
/// resolved. This keeps sealed and future delta position identities local to
/// their owning storage view without a GAT or per-hit allocation.
#[derive(Clone, Copy)]
pub struct PositionsHandle<'a> {
    reader: &'a dyn PositionsReader,
    posting_ordinal: u32,
}

impl fmt::Debug for PositionsHandle<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PositionsHandle")
            .field("posting_ordinal", &self.posting_ordinal)
            .finish_non_exhaustive()
    }
}

impl<'a> PositionsHandle<'a> {
    /// Bind a zero-based posting ordinal to its exact owning reader.
    #[must_use]
    pub fn new(reader: &'a dyn PositionsReader, posting_ordinal: u32) -> Self {
        Self {
            reader,
            posting_ordinal,
        }
    }

    /// Return the zero-based posting ordinal.
    #[must_use]
    pub const fn posting_ordinal(self) -> u32 {
        self.posting_ordinal
    }

    /// Decode this posting into a reusable caller-owned buffer.
    ///
    /// The buffer is cleared before decoding and remains empty on failure.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    pub fn decode_into(self, output: &mut Vec<u32>) -> Result<(), ArgusError> {
        output.clear();
        if let Err(error) = self.reader.decode_positions(self.posting_ordinal, output) {
            output.clear();
            return Err(error);
        }
        Ok(())
    }
}

/// Typed failures from exhaustive query evaluation.
#[derive(Debug, Error)]
pub enum ArgusError {
    /// A sealed posting cursor violated its already-validated storage contract.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
    /// A sealed position cursor violated its validated paired-stream contract.
    #[error(transparent)]
    Position(#[from] PositionCodecError),
    /// A numeric range could not be compiled from its validated field column.
    #[error(transparent)]
    Numeric(#[from] NumericCodecError),
    /// Snapshot counters cannot describe a valid scored field.
    #[error("invalid BM25 snapshot for field {field_ord}: {reason}")]
    InvalidSnapshot {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Failed invariant.
        reason: &'static str,
    },
    /// Snapshot term frequency exceeded the snapshot's BM25 document count.
    #[error("term doc_freq {doc_freq} exceeds snapshot N {doc_count}")]
    InvalidDocFrequency {
        /// Snapshot-level term document frequency.
        doc_freq: u64,
        /// Snapshot-level BM25 document count.
        doc_count: u64,
    },
    /// A scorer bound statistics and field lengths from different fields.
    #[error("BM25 field mismatch: fieldnorm field {fieldnorm_field}, stats field {stats_field}")]
    FieldMismatch {
        /// Field ordinal supplied by the DOCLEN column.
        fieldnorm_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A phrase term cursor was opened for a field other than its snapshot.
    #[error("phrase term field {term_field} does not match snapshot field {stats_field}")]
    PhraseTermFieldMismatch {
        /// Field ordinal supplied when the phrase cursor was opened.
        term_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A programmatic query supplied a non-finite or overflowing field boost.
    #[error("field {field_ord} has invalid boost bits 0x{boost_bits:08x}")]
    InvalidBoost {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
    /// A posting had no corresponding DOCLEN byte in its segment.
    #[error("field {field_ord} has no fieldnorm for global docid {global_docid}")]
    MissingFieldnorm {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id.
        global_docid: u32,
    },
    /// A phrase leaf was opened against a frequency-only posting cursor.
    #[error("field {field_ord} has no positions for global docid {global_docid}")]
    MissingPositions {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id whose phrase positions were required.
        global_docid: u32,
    },
    /// A coarse query-execution checkpoint observed structured cancellation.
    #[error("query execution cancelled during {phase}")]
    QueryCancelled {
        /// Stable query phase.
        phase: &'static str,
    },
    /// Deterministic coarse work exceeded the configured query fuel budget.
    #[error(
        "query fuel exhausted after {consumed}/{budget} units \
         (segments={segments_touched}, dictionary_blocks={dictionary_blocks}, \
         posting_blocks={posting_blocks}, position_docs={position_docs})"
    )]
    QueryFuelExhausted {
        /// Configured work-unit ceiling.
        budget: u64,
        /// Successfully admitted work units before exhaustion.
        consumed: u64,
        /// Segment transitions admitted before exhaustion.
        segments_touched: u64,
        /// Prefix-compressed dictionary blocks admitted before exhaustion.
        dictionary_blocks: u64,
        /// Posting blocks admitted before exhaustion.
        posting_blocks: u64,
        /// Candidate documents admitted for position verification.
        position_docs: u64,
    },
    /// A phrase plan cannot describe an exact positional query.
    #[error("invalid exact phrase: {reason}")]
    InvalidPhrase {
        /// Failed phrase-plan invariant.
        reason: &'static str,
    },
    /// A cursor exposed internally inconsistent accessors.
    #[error("posting cursor invariant failed: {0}")]
    CursorInvariant(&'static str),
    /// Score arithmetic escaped the finite BM25 domain.
    #[error("non-finite score for global docid {global_docid}")]
    NonFiniteScore {
        /// Global Quill document id being scored.
        global_docid: u32,
    },
    /// A bounded internal allocation could not be reserved.
    #[error("could not allocate {count} entries for {resource}")]
    Allocation {
        /// Bounded allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// A winning global document id could not be resolved externally.
    #[error("no external document id for global docid {global_docid}")]
    MissingExternalDocId {
        /// Global Quill document id.
        global_docid: u32,
    },
    /// Offset pagination overflowed the host collection domain.
    #[error("top-k collector window overflows usize: limit {limit}, offset {offset}")]
    CollectorWindowOverflow {
        /// Requested page size.
        limit: usize,
        /// Requested number of leading hits to skip.
        offset: usize,
    },
    /// An exact match count exceeded the durable collection domain.
    #[error("exact match count overflowed u64")]
    MatchCountOverflow,
    /// A movement-only operation received a tree that scores while advancing.
    #[error("movement-only scoring requires a recursively unscored scorer tree")]
    ScoredTreeForUnscoredCollector,
    /// A caller attempted to score a deliberately unscored query tree.
    #[error("an unscored Boolean query has no BM25 score")]
    ScoreUnavailable,
    /// A NUMERIC column cannot contain more values than physical documents.
    #[error(
        "NUMERIC field {field_ord} has {value_count} values for {segment_num_docs} physical documents"
    )]
    InvalidNumericCardinality {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Validated values present in the field column.
        value_count: usize,
        /// Physical at-seal documents supplied by the segment plan.
        segment_num_docs: u32,
    },
    /// A programmatic constant-score predicate supplied a non-finite boost.
    #[error("constant-score predicate has invalid boost bits 0x{boost_bits:08x}")]
    InvalidConstantBoost {
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
    /// A match-all leaf cannot describe the supplied global document domain.
    #[error(
        "invalid All scorer range [{docid_lo}, {docid_hi}) for {segment_num_docs} live documents: {reason}"
    )]
    InvalidAllRange {
        /// Inclusive global document id lower bound.
        docid_lo: u64,
        /// Exclusive global document id upper bound.
        docid_hi: u64,
        /// Live documents reported by the segment.
        segment_num_docs: u32,
        /// Failed range invariant.
        reason: &'static str,
    },
    /// A programmatic match-all query supplied a non-finite boost.
    #[error("All scorer has invalid boost bits 0x{boost_bits:08x}")]
    InvalidAllBoost {
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
}

/// One coarse, deterministic query-work boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryWorkKind {
    /// Entering one sealed or Delta segment.
    Segment,
    /// Entering one prefix-compressed term-dictionary block.
    DictionaryBlock,
    /// Decoding or entering one posting block.
    PostingBlock,
    /// Verifying positions for one phrase candidate document.
    PositionDocument,
}

/// Cancel-aware deterministic query-work admission.
///
/// Implementations must be thread-safe because ordinary, comfortably
/// under-budget queries may still fan sealed segments across Rayon. Fuel
/// exhaustion itself is executed serially by the shipping facade so its
/// progress diagnostics are schedule-independent.
pub trait QueryWorkCheckpoint: Send + Sync {
    /// Admit `units` of one coarse work kind.
    ///
    /// # Errors
    ///
    /// Returns [`ArgusError::QueryCancelled`] or
    /// [`ArgusError::QueryFuelExhausted`] before the guarded work begins.
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError>;
}

/// Forward-only posting access shared by sealed and future delta segments.
///
/// A cursor starts on its first posting. `advance` is an inclusive lower-bound
/// seek and does not move when the current document already satisfies the
/// target. Exhaustion is fused and represented by `None`.
pub trait PostingCursor {
    /// Return the current global document id, or `None` after exhaustion.
    fn doc(&self) -> Option<u32>;

    /// Return the current term frequency, or `None` after exhaustion.
    fn freq(&self) -> Option<u32>;

    /// Return the lazy POSITIONS locator for the current posting, when present.
    fn positions_handle(&self) -> Option<PositionsHandle<'_>>;

    /// Best-effort number of documents matched by this cursor.
    fn size_hint(&self) -> u32;

    /// Relative cost to drive this cursor to exhaustion.
    fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    /// Live document count used by Tantivy's composite size estimators.
    fn segment_num_docs(&self) -> u32;

    /// Move strictly forward by one posting.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn next(&mut self) -> Result<Option<u32>, ArgusError>;

    /// Land on the first posting whose document id is at least `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError>;

    /// Fork the current position into an independent cursor suitable for
    /// competitive candidate generation.
    ///
    /// The default opts out. A cursor must return `Some` only when the fork
    /// retains sound live-snapshot impact metadata for every remaining row.
    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        None
    }

    /// Conservative score ceiling over every remaining block in this term.
    ///
    /// Implementations must apply `record_option` to the cursor's occurrence
    /// count. In particular, [`TermRecordOption::Basic`] bounds use presence
    /// frequency one even when a sealed POSTINGS row retains a raw count.
    fn term_score_upper_bound(
        &self,
        _live_avgdl: f32,
        _weight: f32,
        _record_option: TermRecordOption,
    ) -> Option<f32> {
        None
    }

    /// Whether this cursor exposes real skip-capable posting blocks.
    ///
    /// A whole-term ceiling is sufficient for `MaxScore` but not for BMW. The
    /// default is deliberately false so a custom cursor with only a term bound
    /// falls back to exhaustive collection instead of surfacing an invariant
    /// error when the clause-count threshold selects BMW.
    fn supports_block_max(&self) -> bool {
        false
    }

    /// Conservative score ceiling for the cursor's current posting block.
    fn current_block_score_upper_bound(
        &self,
        _live_avgdl: f32,
        _weight: f32,
        _record_option: TermRecordOption,
    ) -> Option<f32> {
        None
    }

    /// Inclusive validated last docid of the current posting block.
    fn current_block_last_doc(&self) -> Option<u32> {
        None
    }

    /// Stable zero-based work-block ordinal for coarse query checkpoints.
    ///
    /// Sealed cursors report their validated POSTINGS block. Delta cursors
    /// report their logical 128-posting block. Cursors whose postings were
    /// already charged while materializing may retain the default `None`.
    fn current_work_block(&self) -> Option<u64> {
        None
    }

    /// Number of block entries performed since `previous`.
    ///
    /// Sealed skip seeks decode only their destination block, so the default
    /// charges one when the ordinal changes. Sequential Delta cursors override
    /// this to charge every traversed logical block.
    fn work_blocks_since(&self, previous: Option<u64>) -> u64 {
        match (previous, self.current_work_block()) {
            (_, None) => 0,
            (Some(previous), Some(current)) if previous == current => 0,
            _ => 1,
        }
    }
}

impl<T> PostingCursor for Box<T>
where
    T: PostingCursor + ?Sized,
{
    fn doc(&self) -> Option<u32> {
        (**self).doc()
    }

    fn freq(&self) -> Option<u32> {
        (**self).freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        (**self).positions_handle()
    }

    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }

    fn cost(&self) -> u64 {
        (**self).cost()
    }

    fn segment_num_docs(&self) -> u32 {
        (**self).segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        (**self).next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        (**self).advance(target)
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        (**self).fork_for_pruning()
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        (**self).term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        (**self).supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        (**self).current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        (**self).current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        (**self).current_work_block()
    }

    fn work_blocks_since(&self, previous: Option<u64>) -> u64 {
        (**self).work_blocks_since(previous)
    }
}

enum PostingCheckpoint<'a> {
    Shared(Arc<dyn QueryWorkCheckpoint + 'a>),
    Borrowed(&'a dyn QueryWorkCheckpoint),
}

impl PostingCheckpoint<'_> {
    fn admit(&self, kind: QueryWorkKind, units: u64) -> Result<(), ArgusError> {
        match self {
            Self::Shared(checkpoint) => checkpoint.admit(kind, units),
            Self::Borrowed(checkpoint) => checkpoint.admit(kind, units),
        }
    }

    fn as_borrowed(&self) -> &dyn QueryWorkCheckpoint {
        match self {
            Self::Shared(checkpoint) => checkpoint.as_ref(),
            Self::Borrowed(checkpoint) => *checkpoint,
        }
    }
}

/// Posting cursor wrapper that checks cancellation and fuel only when a coarse
/// posting block is entered.
pub struct CheckpointPostingCursor<'a> {
    inner: Box<dyn PostingCursor + 'a>,
    checkpoint: PostingCheckpoint<'a>,
}

impl<'a> CheckpointPostingCursor<'a> {
    /// Wrap a freshly opened cursor and admit its already-decoded first block.
    ///
    /// # Errors
    ///
    /// Returns a typed checkpoint error before scorer construction continues.
    pub fn new<C>(
        inner: C,
        checkpoint: Arc<dyn QueryWorkCheckpoint + 'a>,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
    {
        let cursor = Self {
            inner: Box::new(inner),
            checkpoint: PostingCheckpoint::Shared(checkpoint),
        };
        if cursor.inner.current_work_block().is_some() {
            cursor.checkpoint.admit(QueryWorkKind::PostingBlock, 1)?;
        }
        Ok(cursor)
    }

    fn checkpoint_move(&self, previous: Option<u64>) -> Result<(), ArgusError> {
        let units = self.inner.work_blocks_since(previous);
        self.checkpoint.admit(QueryWorkKind::PostingBlock, units)
    }
}

impl PostingCursor for CheckpointPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.inner.doc()
    }

    fn freq(&self) -> Option<u32> {
        self.inner.freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.inner.positions_handle()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }

    fn cost(&self) -> u64 {
        self.inner.cost()
    }

    fn segment_num_docs(&self) -> u32 {
        self.inner.segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let previous = self.inner.current_work_block();
        let moved = self.inner.next()?;
        self.checkpoint_move(previous)?;
        Ok(moved)
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let previous = self.inner.current_work_block();
        let moved = self.inner.advance(target)?;
        self.checkpoint_move(previous)?;
        Ok(moved)
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        let inner = self.inner.fork_for_pruning()?;
        Some(Box::new(CheckpointPostingCursor {
            inner,
            checkpoint: PostingCheckpoint::Borrowed(self.checkpoint.as_borrowed()),
        }))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.inner
            .term_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn supports_block_max(&self) -> bool {
        self.inner.supports_block_max()
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        self.inner
            .current_block_score_upper_bound(live_avgdl, weight, record_option)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        self.inner.current_block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        self.inner.current_work_block()
    }

    fn work_blocks_since(&self, previous: Option<u64>) -> u64 {
        self.inner.work_blocks_since(previous)
    }
}

enum SealedCursorInner<'a> {
    Docs(crate::quiver::PostingCursor<'a>),
    Positions(crate::quiver::PositionCursor<'a>),
}

#[derive(Clone)]
enum SealedBlockMax {
    ValidatedTerm(Arc<ValidatedTermPruningMetadata>),
    #[cfg(test)]
    Fixture(Arc<[BlockMaxEntry]>),
}

impl SealedBlockMax {
    fn entries(&self) -> &[BlockMaxEntry] {
        match self {
            Self::ValidatedTerm(metadata) => metadata.block_max(),
            #[cfg(test)]
            Self::Fixture(entries) => entries,
        }
    }

    fn term_score_bound_inputs(&self) -> Option<(u32, u8)> {
        match self {
            Self::ValidatedTerm(metadata) => metadata.term_score_bound_inputs(),
            #[cfg(test)]
            Self::Fixture(entries) => {
                let mut maximum_frequency = 0_u32;
                let mut minimum_fieldnorm_id = u8::MAX;
                for entry in entries.iter() {
                    maximum_frequency = maximum_frequency.max(entry.max_frequency());
                    minimum_fieldnorm_id = minimum_fieldnorm_id.min(entry.min_fieldnorm_id());
                }
                (!entries.is_empty()).then_some((maximum_frequency, minimum_fieldnorm_id))
            }
        }
    }
}

/// Owner-safe adapter for one validated sealed posting or position cursor.
pub struct SealedPostingCursor<'a> {
    inner: SealedCursorInner<'a>,
    block_max: Option<SealedBlockMax>,
    size_hint: u32,
    segment_num_docs: u32,
}

impl<'a> SealedPostingCursor<'a> {
    /// Open a position-free sealed posting cursor.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated posting list can no longer open
    /// its cursor.
    pub fn new(
        postings: &'a crate::quiver::PostingList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Docs(postings.cursor()?),
            block_max: None,
            size_hint: postings.doc_freq(),
            segment_num_docs,
        })
    }

    /// Bind an owner-safe compressed posting cursor without pruning metadata.
    ///
    /// This is the exact-count and scoreless production path: POSTINGS remains
    /// lazy and owner-safe, while BLOCKMAX is neither opened nor represented by
    /// an empty sentinel.
    #[must_use]
    pub(crate) fn from_owned(
        postings: crate::quiver::PostingCursor<'a>,
        size_hint: u32,
        segment_num_docs: u32,
    ) -> Self {
        Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: None,
            size_hint,
            segment_num_docs,
        }
    }

    /// Open an owner-safe cursor from one segment-cached, fully validated term.
    ///
    /// POSTINGS block layout and BLOCKMAX rows remain paired behind the same
    /// shared metadata value, so no production caller can bind bounds from a
    /// different term.
    ///
    /// # Errors
    ///
    /// Returns a typed posting invariant error if the selected immutable term
    /// span no longer matches the cached validation witness.
    pub(crate) fn from_validated_pruning(
        postings_bytes: &'a [u8],
        metadata: Arc<ValidatedTermPruningMetadata>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        let size_hint = metadata.doc_freq();
        let postings = metadata.cursor(postings_bytes)?;
        Ok(Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: Some(SealedBlockMax::ValidatedTerm(metadata)),
            size_hint,
            segment_num_docs,
        })
    }

    /// Bind an owner-safe posting cursor to fully validated BLOCKMAX entries.
    ///
    /// The entries must come from the same [`crate::quiver::PostingList`] that
    /// produced `postings`; the storage opener is responsible for validating
    /// that one-to-one relationship before construction.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn with_block_max(
        postings: crate::quiver::PostingCursor<'a>,
        block_max: Arc<[BlockMaxEntry]>,
        size_hint: u32,
        segment_num_docs: u32,
    ) -> Self {
        Self {
            inner: SealedCursorInner::Docs(postings),
            block_max: Some(SealedBlockMax::Fixture(block_max)),
            size_hint,
            segment_num_docs,
        }
    }

    /// Open a sealed cursor already paired with its validated POSITIONS stream.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated paired streams can no longer
    /// open their allocation-free cursor.
    pub fn with_positions(
        positions: &'a crate::quiver::PositionList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Positions(positions.cursor()?),
            block_max: None,
            size_hint: positions.doc_freq(),
            segment_num_docs,
        })
    }
}

impl PostingCursor for SealedPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.doc(),
            SealedCursorInner::Positions(cursor) => cursor.doc(),
        }
    }

    fn freq(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.freq(),
            SealedCursorInner::Positions(cursor) => cursor.freq(),
        }
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        match &self.inner {
            SealedCursorInner::Docs(_) => None,
            SealedCursorInner::Positions(cursor) => cursor
                .posting_ordinal()
                .map(|ordinal| PositionsHandle::new(self, ordinal)),
        }
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => Ok(cursor.next()?.map(|posting| posting.doc_id)),
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.next()?.map(|posting| posting.doc_id))
            }
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
        }
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        self.block_max.as_ref()?;
        Some(Box::new(Self {
            inner: SealedCursorInner::Docs(cursor.clone()),
            block_max: self.block_max.clone(),
            size_hint: self.size_hint,
            segment_num_docs: self.segment_num_docs,
        }))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let (maximum_frequency, minimum_fieldnorm_id) =
            self.block_max.as_ref()?.term_score_bound_inputs()?;
        record_option.score_upper_bound(maximum_frequency, minimum_fieldnorm_id, live_avgdl, weight)
    }

    fn supports_block_max(&self) -> bool {
        matches!(&self.inner, SealedCursorInner::Docs(_))
            && self
                .block_max
                .as_ref()
                .is_some_and(|entries| !entries.entries().is_empty())
    }

    fn current_block_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        let block_index = cursor.block_index()?;
        let entry = self.block_max.as_ref()?.entries().get(block_index)?;
        record_option.score_upper_bound(
            entry.max_frequency(),
            entry.min_fieldnorm_id(),
            live_avgdl,
            weight,
        )
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        let SealedCursorInner::Docs(cursor) = &self.inner else {
            return None;
        };
        cursor.block_last_doc()
    }

    fn current_work_block(&self) -> Option<u64> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.block_index().map(|block| block as u64),
            SealedCursorInner::Positions(cursor) => {
                cursor.posting_block_index().map(|block| block as u64)
            }
        }
    }
}

impl PositionsReader for SealedPostingCursor<'_> {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let SealedCursorInner::Positions(cursor) = &self.inner else {
            return Err(ArgusError::CursorInvariant(
                "position-free sealed cursor cannot resolve positions",
            ));
        };
        if cursor.posting_ordinal() != Some(posting_ordinal) {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = cursor.positions()?.ok_or(ArgusError::CursorInvariant(
            "positioned sealed cursor has no current position run",
        ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "decoded positions",
                count: positions.len(),
            })?;
        for position in positions {
            output.push(position?);
        }
        Ok(())
    }
}

/// Allocation-free cursor over one immutable Delta term chain.
///
/// Physical postings remain append-ordered in the frozen generation. The
/// adapter validates that ordering while skipping rows superseded or deleted
/// by the Delta-local identity overlay. A cursor starts on its first live row,
/// exactly like the sealed adapter.
#[derive(Clone)]
pub struct DeltaPostingCursor<'a> {
    term: Option<DeltaTerm<'a>>,
    block_max: Option<DeltaBlockMax>,
    remaining: Option<DeltaPostings<'a>>,
    current: Option<DeltaPosting<'a>>,
    current_ordinal: Option<u32>,
    next_ordinal: u32,
    last_physical_doc: Option<u32>,
    size_hint: u32,
    cost: u64,
    segment_num_docs: u32,
}

impl<'a> DeltaPostingCursor<'a> {
    /// Open one live-only `(field, term)` cursor in a frozen Delta generation.
    ///
    /// A missing term produces a fused empty cursor with the generation's live
    /// segment cardinality. No posting or position row is materialized.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant failure if a generation exceeds the durable
    /// `u32` cursor domain or its first posting violates ascending order.
    pub fn new(
        delta: &'a DeltaSnapshot,
        field_ord: u16,
        term_bytes: &[u8],
    ) -> Result<Self, ArgusError> {
        let term = delta.find_term(field_ord, term_bytes);
        let (live_doc_freq, block_max) =
            term.map_or((0, None), DeltaTerm::live_doc_freq_and_block_max);
        let size_hint = u32::try_from(live_doc_freq).map_err(|_| {
            ArgusError::CursorInvariant("Delta live document frequency exceeds u32")
        })?;
        // Intersection ordering also fixes the f32 score-accumulation order.
        // Use the same live document frequency emitted by a sealed cursor so
        // folding Delta tombstones cannot perturb score bits across a seal.
        let cost = u64::try_from(live_doc_freq).map_err(|_| {
            ArgusError::CursorInvariant("Delta live document frequency exceeds u64")
        })?;
        let segment_num_docs = u32::try_from(delta.live_document_count())
            .map_err(|_| ArgusError::CursorInvariant("Delta live document count exceeds u32"))?;
        let remaining = term.map(DeltaTerm::postings);
        let mut cursor = Self {
            term,
            block_max,
            remaining,
            current: None,
            current_ordinal: None,
            next_ordinal: 0,
            last_physical_doc: None,
            size_hint,
            cost,
            segment_num_docs,
        };
        cursor.pull_next_live()?;
        Ok(cursor)
    }

    fn pull_next_live(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(term) = self.term else {
            self.current = None;
            self.current_ordinal = None;
            return Ok(None);
        };
        let Some(remaining) = self.remaining.as_mut() else {
            self.current = None;
            self.current_ordinal = None;
            return Ok(None);
        };
        loop {
            let Some(posting) = remaining.next() else {
                self.current = None;
                self.current_ordinal = None;
                return Ok(None);
            };
            if self
                .last_physical_doc
                .is_some_and(|previous| posting.global_docid <= previous)
            {
                return Err(ArgusError::CursorInvariant(
                    "Delta term postings are not strictly docid-ascending",
                ));
            }
            self.last_physical_doc = Some(posting.global_docid);
            let ordinal = self.next_ordinal;
            self.next_ordinal =
                self.next_ordinal
                    .checked_add(1)
                    .ok_or(ArgusError::CursorInvariant(
                        "Delta posting ordinal exceeds u32",
                    ))?;
            if !term.is_live(posting) {
                continue;
            }
            self.current = Some(posting);
            self.current_ordinal = Some(ordinal);
            return Ok(Some(posting.global_docid));
        }
    }
}

impl PostingCursor for DeltaPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.current.map(|posting| posting.global_docid)
    }

    fn freq(&self) -> Option<u32> {
        self.current.map(|posting| posting.frequency)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current.filter(|posting| posting.has_positions())?;
        self.current_ordinal
            .map(|ordinal| PositionsHandle::new(self, ordinal))
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        self.pull_next_live()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_none_or(|doc| doc >= target) {
            return Ok(self.doc());
        }
        loop {
            let moved = self.pull_next_live()?;
            if moved.is_none_or(|doc| doc >= target) {
                return Ok(moved);
            }
        }
    }

    fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
        self.block_max?;
        Some(Box::new(self.clone()))
    }

    fn term_score_upper_bound(
        &self,
        live_avgdl: f32,
        weight: f32,
        record_option: TermRecordOption,
    ) -> Option<f32> {
        let block_max = self.block_max?;
        record_option.score_upper_bound(
            block_max.max_frequency(),
            block_max.min_fieldnorm_id(),
            live_avgdl,
            weight,
        )
    }

    fn current_work_block(&self) -> Option<u64> {
        self.current_ordinal.map(|ordinal| u64::from(ordinal) / 128)
    }

    fn work_blocks_since(&self, previous: Option<u64>) -> u64 {
        match (previous, self.current_work_block()) {
            (_, None) => 0,
            (Some(previous), Some(current)) if current >= previous => current - previous,
            _ => 1,
        }
    }
}

impl PositionsReader for DeltaPostingCursor<'_> {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        if self.current_ordinal != Some(posting_ordinal) {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current Delta posting",
            ));
        }
        let term = self.term.ok_or(ArgusError::CursorInvariant(
            "empty Delta cursor cannot resolve positions",
        ))?;
        let posting = self.current.ok_or(ArgusError::CursorInvariant(
            "exhausted Delta cursor cannot resolve positions",
        ))?;
        let expected = usize::try_from(posting.frequency)
            .map_err(|_| ArgusError::CursorInvariant("Delta position count does not fit usize"))?;
        let positions = term.positions(posting).ok_or(ArgusError::CursorInvariant(
            "positioned Delta cursor has no current position run",
        ))?;
        output
            .try_reserve_exact(expected)
            .map_err(|_| ArgusError::Allocation {
                resource: "decoded Delta positions",
                count: expected,
            })?;
        let initial_len = output.len();
        output.extend(positions);
        if output.len().saturating_sub(initial_len) != expected {
            return Err(ArgusError::CursorInvariant(
                "Delta position run disagrees with posting frequency",
            ));
        }
        Ok(())
    }
}

/// Borrowed fieldnorm view paired with one frozen Delta generation.
#[derive(Clone, Copy)]
pub struct DeltaFieldNorms<'a> {
    delta: &'a DeltaSnapshot,
    field_ord: u16,
}

impl<'a> DeltaFieldNorms<'a> {
    /// Bind one indexed string field in an immutable Delta generation.
    #[must_use]
    pub const fn new(delta: &'a DeltaSnapshot, field_ord: u16) -> Self {
        Self { delta, field_ord }
    }
}

impl FieldNormReader for DeltaFieldNorms<'_> {
    fn field_ord(&self) -> u16 {
        self.field_ord
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        self.delta.live_fieldnorm_id(self.field_ord, global_docid)
    }
}

/// Shared, validated BM25 denominator cache for one `(field, snapshot)` pair.
#[derive(Clone, Debug)]
pub struct Bm25FieldSnapshot {
    stats: SnapshotFieldStats,
    average_field_length: Option<f32>,
    tf_cache: Arc<[f32; 256]>,
}

impl Bm25FieldSnapshot {
    /// Validate snapshot counters and build the field's 256-entry tf cache.
    ///
    /// An empty or all-empty field has no scored postings and therefore keeps
    /// a zero cache. A term scorer rejects any posting that would try to use
    /// that cache.
    ///
    /// # Errors
    ///
    /// Returns a typed error when an empty snapshot reports tokens or the raw
    /// average cannot be represented as a finite f32.
    pub fn new(stats: SnapshotFieldStats) -> Result<Self, ArgusError> {
        if stats.doc_count == 0 && stats.total_tokens != 0 {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "an empty snapshot cannot contain field tokens",
            });
        }
        let average_field_length = stats.average_field_length();
        if average_field_length.is_some_and(|average| !average.is_finite()) {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "raw average field length is not finite",
            });
        }
        let tf_cache = average_field_length
            .filter(|average| *average > 0.0)
            .map_or_else(
                || Arc::new([0.0; 256]),
                |average| Arc::new(compute_tf_cache(average)),
            );
        Ok(Self {
            stats,
            average_field_length,
            tf_cache,
        })
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.stats.field_ord
    }

    /// Snapshot lifecycle population used as BM25 `N`.
    ///
    /// Composite views count Keeper's at-seal rows plus live Delta rows.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.stats.doc_count
    }

    /// Snapshot token numerator under the same lifecycle as [`Self::doc_count`].
    ///
    /// Keeper tombstones remain until compaction; Delta tombstones are excluded
    /// immediately because those rows are never sealed.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.stats.total_tokens
    }

    /// Raw `total_tokens / N` average, or `None` for an empty snapshot.
    #[must_use]
    pub const fn average_field_length(&self) -> Option<f32> {
        self.average_field_length
    }
}

/// Backend-neutral fieldnorm access for sealed and delta segment views.
pub trait FieldNormReader {
    /// Stable schema field ordinal.
    fn field_ord(&self) -> u16;

    /// Stored Tantivy-compatible fieldnorm ID for one present global docid.
    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8>;
}

impl FieldNormReader for DocLenField<'_> {
    fn field_ord(&self) -> u16 {
        DocLenField::field_ord(*self)
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        DocLenField::fieldnorm_id(*self, u64::from(global_docid))
    }
}

/// Posting detail selected by the field schema for term scoring.
///
/// Snapshot field statistics retain the complete analyzed token count
/// independently of posting detail. Tantivy's `Basic` record option scores
/// every matching document with an effective term frequency of one, whether
/// the cursor exposes a canonical one or a raw sealed count. Frequency-bearing
/// options consume the cursor's recorded count unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TermRecordOption {
    /// Document presence only; BM25 uses an effective term frequency of one.
    Basic,
    /// Document presence plus term frequency.
    WithFreqs,
    /// Document presence, term frequency, and positions.
    WithFreqsAndPositions,
}

impl TermRecordOption {
    #[inline]
    const fn effective_frequency(self, raw_frequency: u32) -> u32 {
        match self {
            Self::Basic => 1,
            Self::WithFreqs | Self::WithFreqsAndPositions => raw_frequency,
        }
    }

    fn score_upper_bound(
        self,
        raw_max_frequency: u32,
        min_fieldnorm_id: u8,
        live_avgdl: f32,
        weight: f32,
    ) -> Option<f32> {
        crate::contract::block_max_score(
            crate::contract::block_max_frequency_to_code(
                self.effective_frequency(raw_max_frequency),
            ),
            min_fieldnorm_id,
            live_avgdl,
            weight,
        )
    }
}

/// One BM25 term leaf bound to a segment cursor and fieldnorm view.
pub struct TermScorer<'a> {
    cursor: Box<dyn PostingCursor + 'a>,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    record_option: TermRecordOption,
    weight: f32,
    term_score_upper_bound: Option<f32>,
    cost: u64,
    size_hint: u32,
    segment_num_docs: u32,
}

impl<'a> TermScorer<'a> {
    /// Build a term scorer from snapshot-level statistics and segment-local
    /// cursor state.
    ///
    /// `snapshot_doc_freq` follows the same lifecycle population: Keeper
    /// postings include tombstones until compaction, while Delta postings count
    /// only live rows. Runtime cost, size hint, and segment `num_docs` come from
    /// the cursor so exact required-clause ordering cannot drift from its
    /// storage implementation.
    ///
    /// # Errors
    ///
    /// Rejects field drift, malformed snapshot statistics, inconsistent
    /// cursor accessors, invalid boosts, and `doc_freq > N` without invoking
    /// the vendored `idf` assertion.
    pub fn new<C, F>(
        cursor: C,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        record_option: TermRecordOption,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
        F: FieldNormReader + 'a,
    {
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if snapshot_doc_freq > snapshot.doc_count() {
            return Err(ArgusError::InvalidDocFrequency {
                doc_freq: snapshot_doc_freq,
                doc_count: snapshot.doc_count(),
            });
        }
        let size_hint = cursor.size_hint();
        let cost = cursor.cost();
        let segment_num_docs = cursor.segment_num_docs();
        if u64::from(segment_num_docs) > snapshot.doc_count() {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "segment num_docs exceeds snapshot BM25 document count",
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        if snapshot_doc_freq != 0
            && snapshot
                .average_field_length()
                .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored term requires a positive raw average field length",
            });
        }
        if cursor.doc().is_some() {
            if snapshot_doc_freq == 0 {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "a non-empty cursor cannot have zero snapshot doc_freq",
                });
            }
            if size_hint == 0 || cost == 0 {
                return Err(ArgusError::CursorInvariant(
                    "a non-empty cursor must have non-zero size and runtime cost",
                ));
            }
            validate_cursor_position(&cursor, &fieldnorms)?;
        } else {
            if cursor.freq().is_some() || cursor.positions_handle().is_some() {
                return Err(ArgusError::CursorInvariant(
                    "exhausted cursor retained frequency or positions state",
                ));
            }
            if size_hint != 0 || cost != 0 {
                return Err(ArgusError::CursorInvariant(
                    "an empty cursor must have zero size and runtime cost",
                ));
            }
        }

        let mut weight = idf(snapshot_doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        // Query-time avgdl and idf/boost are immutable for this scorer. Scan a
        // sealed term's validated block table once here instead of once per
        // 4,096-doc competitive window.
        let term_score_upper_bound = snapshot
            .average_field_length()
            .filter(|average| average.is_finite() && *average > 0.0)
            .filter(|_| !weight.is_sign_negative())
            .and_then(|average| cursor.term_score_upper_bound(average, weight, record_option))
            .filter(|bound| bound.is_finite() && !bound.is_sign_negative());
        Ok(Self {
            cursor: Box::new(cursor),
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            record_option,
            weight,
            term_score_upper_bound,
            cost,
            size_hint,
            segment_num_docs,
        })
    }

    fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.next()?;
        let moved =
            validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms.as_ref(), moved)?;
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during next",
            )),
            (Some(before), Some(after)) if after <= before => Err(ArgusError::CursorInvariant(
                "next did not move to a strictly greater document",
            )),
            _ => Ok(moved),
        }
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.advance(target)?;
        let moved =
            validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms.as_ref(), moved)?;
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during advance",
            )),
            (Some(before), Some(after)) if before >= target && after != before => {
                Err(ArgusError::CursorInvariant(
                    "advance moved despite the current document satisfying its target",
                ))
            }
            (Some(before), None) if before >= target => Err(ArgusError::CursorInvariant(
                "advance exhausted despite the current document satisfying its target",
            )),
            (_, Some(after)) if after < target => Err(ArgusError::CursorInvariant(
                "advance landed below its requested target",
            )),
            _ => Ok(moved),
        }
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.cursor.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted cursor",
        ))?;
        let frequency = self.cursor.freq().ok_or(ArgusError::CursorInvariant(
            "current posting has no frequency",
        ))?;
        if frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current posting has zero term frequency",
            ));
        }
        let frequency = self.record_option.effective_frequency(frequency);
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        let tf_factor = frequency / (frequency + norm);
        finite_score(self.weight * tf_factor, doc)
    }

    fn competitive_cursor(&self) -> Option<CompetitiveTermCursor<'_>> {
        let live_avgdl = self.snapshot.average_field_length()?;
        let term_upper_bound = self.term_score_upper_bound?;
        Some(CompetitiveTermCursor {
            cursor: self.cursor.fork_for_pruning()?,
            live_avgdl,
            weight: self.weight,
            record_option: self.record_option,
            term_upper_bound,
        })
    }
}

struct CompetitiveTermCursor<'a> {
    cursor: Box<dyn PostingCursor + 'a>,
    live_avgdl: f32,
    weight: f32,
    record_option: TermRecordOption,
    term_upper_bound: f32,
}

impl CompetitiveTermCursor<'_> {
    fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.cursor.next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        self.cursor.advance(target)
    }

    const fn term_upper_bound(&self) -> f32 {
        self.term_upper_bound
    }

    fn current_block_upper_bound(&self) -> Option<f32> {
        let bound = self.cursor.current_block_score_upper_bound(
            self.live_avgdl,
            self.weight,
            self.record_option,
        )?;
        (bound.is_finite() && !bound.is_sign_negative()).then_some(bound)
    }

    fn current_block_last_doc(&self) -> Option<u32> {
        self.cursor.current_block_last_doc()
    }
}

/// One analyzed phrase term bound to its positioned posting cursor.
///
/// Terms must be supplied in non-decreasing analyzer-position order. Terms at
/// the same position are treated as alternatives for matching while retaining
/// their original order in the fixed phrase-weight calculation. The field
/// ordinal is trusted lowering metadata and must come from the same posting-open
/// operation as the cursor; the opaque cursor contract cannot recover it.
pub struct PhraseTerm<'a> {
    field_ord: u16,
    position: u32,
    cursor: Box<dyn PostingCursor + 'a>,
    snapshot_doc_freq: u64,
}

impl<'a> PhraseTerm<'a> {
    /// Bind one field, query position, and snapshot document frequency to a cursor.
    ///
    /// `field_ord` must be copied from the trusted posting-open path that
    /// produced `cursor`; construction validates it against snapshot STATS.
    #[must_use]
    pub fn new<C>(field_ord: u16, position: u32, cursor: C, snapshot_doc_freq: u64) -> Self
    where
        C: PostingCursor + 'a,
    {
        Self {
            field_ord,
            position,
            cursor: Box::new(cursor),
            snapshot_doc_freq,
        }
    }

    /// Stable schema field ordinal whose postings back this term.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Analyzer position retained from the parsed phrase.
    #[must_use]
    pub const fn position(&self) -> u32 {
        self.position
    }

    /// Snapshot document frequency used in the phrase's fixed BM25 weight.
    #[must_use]
    pub const fn snapshot_doc_freq(&self) -> u64 {
        self.snapshot_doc_freq
    }
}

struct PhraseSlot<'a> {
    position: u32,
    alternatives: Vec<Box<dyn PostingCursor + 'a>>,
    positions: Vec<u32>,
    cost: u64,
    size_hint: u32,
}

impl<'a> PhraseSlot<'a> {
    fn new(position: u32, cursor: Box<dyn PostingCursor + 'a>) -> Self {
        let cost = cursor.cost();
        let size_hint = cursor.size_hint();
        Self {
            position,
            alternatives: vec![cursor],
            positions: Vec::new(),
            cost,
            size_hint,
        }
    }

    fn push_alternative(&mut self, cursor: Box<dyn PostingCursor + 'a>) -> Result<(), ArgusError> {
        self.alternatives
            .try_reserve(1)
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase alternatives",
                count: self.alternatives.len().saturating_add(1),
            })?;
        self.cost = self.cost.saturating_add(cursor.cost());
        self.alternatives.push(cursor);
        Ok(())
    }

    fn doc(&self) -> Option<u32> {
        self.alternatives
            .iter()
            .filter_map(|cursor| cursor.doc())
            .min()
    }

    fn seek(
        &mut self,
        target: u32,
        fieldnorms: &dyn FieldNormReader,
    ) -> Result<Option<u32>, ArgusError> {
        for cursor in &mut self.alternatives {
            let previous = cursor.doc();
            if previous.is_some_and(|doc| doc < target) {
                let moved = cursor.advance(target)?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                match (previous, moved) {
                    (None, Some(_)) => {
                        return Err(ArgusError::CursorInvariant(
                            "exhausted phrase cursor resurrected during advance",
                        ));
                    }
                    (_, Some(after)) if after < target => {
                        return Err(ArgusError::CursorInvariant(
                            "phrase advance landed below its requested target",
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(self.doc())
    }

    fn next(&mut self, fieldnorms: &dyn FieldNormReader) -> Result<Option<u32>, ArgusError> {
        let Some(previous) = self.doc() else {
            return Ok(None);
        };
        for cursor in &mut self.alternatives {
            if cursor.doc() == Some(previous) {
                let moved = cursor.next()?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                if moved.is_some_and(|after| after <= previous) {
                    return Err(ArgusError::CursorInvariant(
                        "phrase next did not move to a strictly greater document",
                    ));
                }
            }
        }
        let moved = self.doc();
        if moved.is_some_and(|after| after <= previous) {
            return Err(ArgusError::CursorInvariant(
                "phrase slot did not move to a strictly greater document",
            ));
        }
        Ok(moved)
    }
}

/// Exact-adjacency phrase scorer over positioned posting cursors.
///
/// Candidate documents are intersected before positions are decoded. The
/// scorer owns and reuses every positional scratch buffer, so advancing after
/// warm-up performs no per-document allocation.
pub struct PhraseScorer<'a> {
    slots: Vec<PhraseSlot<'a>>,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    weight: f32,
    lead_slot: usize,
    cost: u64,
    size_hint: u32,
    segment_num_docs: u32,
    current: Option<u32>,
    current_frequency: u32,
    decode_scratch: Vec<u32>,
    position_indices: Vec<usize>,
    checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
}

impl<'a> PhraseScorer<'a> {
    /// Build and initially align one exact phrase scorer.
    ///
    /// Query terms must be ordered by non-decreasing analyzer position and
    /// span at least two distinct positions. Equal-position terms form one OR
    /// slot. Each term needs its own forward-only positioned cursor, including
    /// repeated terms such as `a a b`.
    ///
    /// # Errors
    ///
    /// Rejects malformed positions, non-positioned cursors, field/domain
    /// drift, invalid snapshot frequencies or boosts, and bounded allocation
    /// failures.
    pub fn new<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        Self::new_with_checkpoint(terms, fieldnorms, snapshot, field_boost, None)
    }

    /// Build one exact phrase scorer with a coarse position-verification
    /// checkpoint.
    ///
    /// The checkpoint runs once per candidate document before position rows
    /// are inspected by the matcher. It is never called inside the tight
    /// position-merge loop.
    ///
    /// # Errors
    ///
    /// Returns the same construction errors as [`Self::new`], plus typed
    /// cancellation or fuel exhaustion from the initial alignment.
    pub fn new_with_checkpoint<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
        checkpoint: Option<Arc<dyn QueryWorkCheckpoint + 'a>>,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        if terms.is_empty() {
            return Err(ArgusError::InvalidPhrase {
                reason: "an exact phrase requires positioned terms",
            });
        }
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let num_terms = u32::try_from(terms.len()).map_err(|_| ArgusError::InvalidPhrase {
            reason: "phrase term count exceeds u32",
        })?;
        let phrase_work_factor = num_terms.checked_mul(10).ok_or(ArgusError::InvalidPhrase {
            reason: "phrase term cost factor exceeds u32",
        })?;

        let mut slots: Vec<PhraseSlot<'a>> = Vec::new();
        slots
            .try_reserve_exact(terms.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position slots",
                count: terms.len(),
            })?;
        let mut segment_num_docs = None;
        let mut idf_sum = 0.0_f32;
        for term in terms {
            if term.field_ord != snapshot.field_ord() {
                return Err(ArgusError::PhraseTermFieldMismatch {
                    term_field: term.field_ord,
                    stats_field: snapshot.field_ord(),
                });
            }
            if term.snapshot_doc_freq > snapshot.doc_count() {
                return Err(ArgusError::InvalidDocFrequency {
                    doc_freq: term.snapshot_doc_freq,
                    doc_count: snapshot.doc_count(),
                });
            }
            let cursor_segment_num_docs = term.cursor.segment_num_docs();
            if u64::from(cursor_segment_num_docs) > snapshot.doc_count() {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "phrase segment num_docs exceeds snapshot BM25 document count",
                });
            }
            if segment_num_docs
                .replace(cursor_segment_num_docs)
                .is_some_and(|previous| previous != cursor_segment_num_docs)
            {
                return Err(ArgusError::CursorInvariant(
                    "phrase cursors belong to different segment domains",
                ));
            }
            validate_phrase_cursor_state(term.cursor.as_ref(), &fieldnorms)?;
            if term.cursor.doc().is_none()
                && (term.cursor.size_hint() != 0 || term.cursor.cost() != 0)
            {
                return Err(ArgusError::CursorInvariant(
                    "an initially empty phrase cursor must have zero size and runtime cost",
                ));
            }
            if term.cursor.doc().is_some() && term.snapshot_doc_freq == 0 {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "a non-empty phrase cursor cannot have zero snapshot doc_freq",
                });
            }
            idf_sum += idf(term.snapshot_doc_freq, snapshot.doc_count());
            if !idf_sum.is_finite() {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "phrase IDF sum is not finite",
                });
            }

            match slots.last_mut() {
                Some(slot) if term.position < slot.position => {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "phrase positions must be non-decreasing",
                    });
                }
                Some(slot) if term.position == slot.position => {
                    slot.push_alternative(term.cursor)?;
                }
                _ => slots.push(PhraseSlot::new(term.position, term.cursor)),
            }
        }
        if slots.len() < 2 {
            return Err(ArgusError::InvalidPhrase {
                reason: "an exact phrase must span at least two positions",
            });
        }
        if slots.iter().any(|slot| {
            slot.alternatives
                .iter()
                .any(|cursor| cursor.doc().is_some())
        }) && snapshot
            .average_field_length()
            .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored phrase requires a positive raw average field length",
            });
        }

        let segment_num_docs = segment_num_docs.unwrap_or(0);
        for slot in &mut slots {
            if slot.alternatives.len() > 1 {
                slot.size_hint = estimate_union(
                    slot.alternatives.iter().map(|cursor| cursor.size_hint()),
                    segment_num_docs,
                );
            }
        }
        let lead_slot = slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.cost)
            .map_or(0, |(index, _)| index);
        let mut cost_ordered_estimates = Vec::new();
        cost_ordered_estimates
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase cost estimates",
                count: slots.len(),
            })?;
        cost_ordered_estimates.extend(slots.iter().map(|slot| (slot.cost, slot.size_hint)));
        cost_ordered_estimates.sort_by_key(|(cost, _)| *cost);
        let candidate_estimate = estimate_intersection(
            cost_ordered_estimates.into_iter().map(|(_, size)| size),
            segment_num_docs,
        );
        let cost = u64::from(candidate_estimate) * u64::from(phrase_work_factor);
        let size_hint = candidate_estimate / phrase_work_factor;
        let mut position_indices = Vec::new();
        position_indices
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position indices",
                count: slots.len(),
            })?;
        position_indices.resize(slots.len(), 0);

        let mut weight = idf_sum * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let mut scorer = Self {
            slots,
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            weight,
            lead_slot,
            cost,
            size_hint,
            segment_num_docs,
            current: None,
            current_frequency: 0,
            decode_scratch: Vec::new(),
            position_indices,
            checkpoint,
        };
        scorer.align()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.current
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].next(fieldnorms)?;
        self.align()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].seek(target, fieldnorms)?;
        self.align()
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        self.current = None;
        self.current_frequency = 0;
        loop {
            let Some(mut target) = self.slots.iter().filter_map(PhraseSlot::doc).max() else {
                return Ok(None);
            };
            loop {
                let mut raised_target = false;
                for slot in &mut self.slots {
                    let Some(doc) = slot.seek(target, self.fieldnorms.as_ref())? else {
                        return Ok(None);
                    };
                    if doc > target {
                        target = doc;
                        raised_target = true;
                    }
                }
                if !raised_target && self.slots.iter().all(|slot| slot.doc() == Some(target)) {
                    break;
                }
            }

            let frequency = self.phrase_frequency(target)?;
            if frequency != 0 {
                self.current = Some(target);
                self.current_frequency = frequency;
                return Ok(self.current);
            }
            let fieldnorms = self.fieldnorms.as_ref();
            self.slots[self.lead_slot].next(fieldnorms)?;
        }
    }

    fn phrase_frequency(&mut self, doc: u32) -> Result<u32, ArgusError> {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint.admit(QueryWorkKind::PositionDocument, 1)?;
        }
        for slot_index in 0..self.slots.len() {
            self.decode_slot_positions(slot_index, doc)?;
        }
        self.position_indices.fill(0);
        let first_position = self.slots[0].position;
        let (slots, position_indices) = (&self.slots, &mut self.position_indices);
        let mut frequency = 0_u32;
        for &base in &slots[0].positions {
            let mut matched = true;
            for slot_index in 1..slots.len() {
                let offset = slots[slot_index].position - first_position;
                let Some(target) = base.checked_add(offset) else {
                    matched = false;
                    break;
                };
                let positions = &slots[slot_index].positions;
                let position_index = &mut position_indices[slot_index];
                while positions
                    .get(*position_index)
                    .is_some_and(|position| *position < target)
                {
                    *position_index += 1;
                }
                if positions.get(*position_index) != Some(&target) {
                    matched = false;
                    break;
                }
            }
            if matched {
                frequency = frequency
                    .checked_add(1)
                    .ok_or(ArgusError::CursorInvariant("phrase frequency exceeds u32"))?;
            }
        }
        Ok(frequency)
    }

    fn decode_slot_positions(&mut self, slot_index: usize, doc: u32) -> Result<(), ArgusError> {
        let (slots, decode_scratch) = (&mut self.slots, &mut self.decode_scratch);
        let slot = &mut slots[slot_index];
        slot.positions.clear();
        for cursor in &slot.alternatives {
            if cursor.doc() != Some(doc) {
                continue;
            }
            let frequency = cursor.freq().ok_or(ArgusError::CursorInvariant(
                "positioned phrase cursor has no frequency",
            ))?;
            let handle = cursor
                .positions_handle()
                .ok_or_else(|| ArgusError::MissingPositions {
                    field_ord: self.snapshot.field_ord(),
                    global_docid: doc,
                })?;
            handle.decode_into(decode_scratch)?;
            let expected = usize::try_from(frequency)
                .map_err(|_| ArgusError::CursorInvariant("phrase frequency does not fit usize"))?;
            if decode_scratch.len() != expected {
                return Err(ArgusError::CursorInvariant(
                    "decoded phrase position count differs from term frequency",
                ));
            }
            let required = slot.positions.len().saturating_add(decode_scratch.len());
            slot.positions
                .try_reserve(decode_scratch.len())
                .map_err(|_| ArgusError::Allocation {
                    resource: "phrase slot positions",
                    count: required,
                })?;
            slot.positions.extend_from_slice(decode_scratch);
        }
        slot.positions.sort_unstable();
        slot.positions.dedup();
        if slot.positions.is_empty() {
            return Err(ArgusError::CursorInvariant(
                "phrase candidate has no decoded positions",
            ));
        }
        Ok(())
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted phrase scorer",
        ))?;
        if self.current_frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current phrase match has zero frequency",
            ));
        }
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = self.current_frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        finite_score(self.weight * (frequency / (frequency + norm)), doc)
    }
}

fn validate_phrase_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<(), ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "phrase movement result disagrees with current document",
        ));
    }
    validate_phrase_cursor_state(cursor, fieldnorms)
}

fn validate_phrase_cursor_state(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError> {
    let Some(doc) = cursor.doc() else {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted phrase cursor retained frequency or positions state",
            ));
        }
        return Ok(());
    };
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned phrase cursor has no positive frequency",
        ));
    }
    if cursor.positions_handle().is_none() {
        return Err(ArgusError::MissingPositions {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    if cursor.size_hint() == 0 || cursor.cost() == 0 {
        return Err(ArgusError::CursorInvariant(
            "a non-empty phrase cursor must have non-zero size and runtime cost",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}

fn validate_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<Option<u32>, ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "movement result disagrees with current document",
        ));
    }
    if moved_doc.is_none() {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted cursor retained frequency or positions state",
            ));
        }
        return Ok(None);
    }
    validate_cursor_position(cursor, fieldnorms)?;
    Ok(moved_doc)
}

fn validate_cursor_position<C>(
    cursor: &C,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError>
where
    C: PostingCursor + ?Sized,
{
    let doc = cursor.doc().ok_or(ArgusError::CursorInvariant(
        "positioned cursor has no document",
    ))?;
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned cursor has no positive frequency",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}

/// One lowered scorer child and its canonical query occurrence.
pub struct ScorerClause<'a> {
    occur: Occur,
    scorer: ReferenceScorer<'a>,
}

impl<'a> ScorerClause<'a> {
    /// Build a lowered scorer child.
    #[must_use]
    pub const fn new(occur: Occur, scorer: ReferenceScorer<'a>) -> Self {
        Self { occur, scorer }
    }

    /// Build a required positive child.
    #[must_use]
    pub const fn must(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Must, scorer)
    }

    /// Build an optional positive child.
    #[must_use]
    pub const fn should(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Should, scorer)
    }

    /// Build a scoreless exclusion child.
    #[must_use]
    pub const fn must_not(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::MustNot, scorer)
    }
}

enum ScorerNode<'a> {
    Empty,
    All(AllScorer),
    Term(TermScorer<'a>),
    Phrase(PhraseScorer<'a>),
    NumericRange(NumericRangeScorer),
    ConstantScore(ConstantScoreScorer),
    Intersection(IntersectionScorer<'a>),
    Union(BufferedUnionScorer<'a>),
    UnscoredUnion(UnscoredUnionScorer<'a>),
    RequiredOptional(RequiredOptionalScorer<'a>),
    Exclude(ExcludeScorer<'a>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BooleanMode {
    Scored,
    Unscored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeekDangerResult {
    Found,
    SeekLowerBound(Option<u32>),
}

fn classify_seek_result(target: u32, result: Option<u32>) -> Result<SeekDangerResult, ArgusError> {
    match result {
        Some(doc) if doc == target => Ok(SeekDangerResult::Found),
        Some(doc) if doc > target => Ok(SeekDangerResult::SeekLowerBound(Some(doc))),
        None => Ok(SeekDangerResult::SeekLowerBound(None)),
        Some(_) => Err(ArgusError::CursorInvariant(
            "danger seek landed below its requested target",
        )),
    }
}

/// Stateful exhaustive scorer tree.
///
/// Construct one tree per segment and consume it with [`Self::top_k`]. The
/// object intentionally retains the unpruned path for differential testing of
/// future WAND and block-max implementations.
pub struct ReferenceScorer<'a> {
    node: ScorerNode<'a>,
}

impl<'a> ReferenceScorer<'a> {
    /// Build a scorer that never matches.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            node: ScorerNode::Empty,
        }
    }

    /// Build a unit-score scorer over one half-open global document-id range.
    ///
    /// The range is the segment's physical document domain. It may include
    /// holes or tombstoned rows; [`LiveDocs`] remains the single visibility
    /// authority during collection. `segment_num_docs` is retained separately
    /// for Tantivy-compatible composite estimates and therefore may be smaller
    /// than the physical range span.
    ///
    /// The bounds use `u64` so `docid_hi == 2^32` can represent a final
    /// searchable `u32::MAX` document. An empty range with zero live documents
    /// lowers to [`Self::empty`].
    ///
    /// # Errors
    ///
    /// Rejects reversed ranges, bounds outside the `u32` payload domain, and a
    /// live count larger than the physical range span.
    pub fn all(docid_lo: u64, docid_hi: u64, segment_num_docs: u32) -> Result<Self, ArgusError> {
        Self::all_with_boost(docid_lo, docid_hi, segment_num_docs, 1.0)
    }

    /// Build a boosted match-all scorer over one half-open global document-id range.
    ///
    /// This is the exact scorer-level lowering for `Boost(All)`: every current
    /// document receives `boost`, while movement and unscored collection never
    /// evaluate it.
    ///
    /// # Errors
    ///
    /// Returns the same range failures as [`Self::all`] and rejects a non-finite
    /// boost before exposing the scorer.
    pub fn all_with_boost(
        docid_lo: u64,
        docid_hi: u64,
        segment_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        let all = AllScorer::new(docid_lo, docid_hi, segment_num_docs, boost)?;
        if all.doc().is_none() {
            Ok(Self::empty())
        } else {
            Ok(Self {
                node: ScorerNode::All(all),
            })
        }
    }

    /// Wrap one BM25 term leaf.
    #[must_use]
    pub fn term(term: TermScorer<'a>) -> Self {
        if term.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Term(term),
            }
        }
    }

    /// Wrap one exact-adjacency phrase leaf.
    #[must_use]
    pub fn phrase(phrase: PhraseScorer<'a>) -> Self {
        if phrase.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Phrase(phrase),
            }
        }
    }

    /// Compile one NUMERIC field range into an owned constant-score doc set.
    ///
    /// `segment_num_docs` is the physical at-seal document count, including
    /// tombstoned rows. Live-document filtering remains a collector concern so
    /// range leaves compose with the same snapshot statistics as term leaves.
    ///
    /// # Errors
    ///
    /// Returns a typed numeric-codec error for a bound whose signedness does
    /// not match the field or if the bounded docid set cannot be allocated.
    /// It also rejects a field value count larger than the caller-supplied
    /// physical segment document count.
    pub fn numeric_range(
        field: NumericField<'_>,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Self::numeric_range_with_boost(field, lower, upper, segment_num_docs, 1.0)
    }

    /// Compile one boosted NUMERIC field range into an owned constant-score doc set.
    ///
    /// Movement, cost, and size estimates are identical to [`Self::numeric_range`].
    /// Each current match scores exactly `boost`, preserving `Boost(Range)` without
    /// introducing a synthetic Boolean child or changing score accumulation order.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::numeric_range`] and rejects a
    /// non-finite boost before exposing the scorer.
    pub fn numeric_range_with_boost(
        field: NumericField<'_>,
        lower: Bound<NumericValue>,
        upper: Bound<NumericValue>,
        segment_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: boost.to_bits(),
            });
        }
        let value_count = field.len();
        if value_count > usize::try_from(segment_num_docs).unwrap_or(usize::MAX) {
            return Err(ArgusError::InvalidNumericCardinality {
                field_ord: field.field_ord(),
                value_count,
                segment_num_docs,
            });
        }
        let docids = field.range_docids(lower, upper)?;
        if docids.is_empty() {
            return Ok(Self::empty());
        }
        Ok(Self {
            node: ScorerNode::NumericRange(NumericRangeScorer::new(
                docids,
                value_count,
                segment_num_docs,
                boost,
            )),
        })
    }

    /// Build one numeric-range scorer from an already materialized sorted doc set.
    ///
    /// The index layer binary-searches the persisted NUMERIC pair run, filters
    /// snapshot-visible documents with cancellation checks, and supplies the
    /// exact matches here. Planning metadata and constant-score behavior remain
    /// identical to the direct NUMERIC range constructor.
    pub(crate) fn materialized_numeric_range(
        field_ord: u16,
        docids: Vec<u32>,
        value_count: usize,
        segment_num_docs: u32,
        covers_every_document: bool,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: boost.to_bits(),
            });
        }
        if value_count > usize::try_from(segment_num_docs).unwrap_or(usize::MAX) {
            return Err(ArgusError::InvalidNumericCardinality {
                field_ord,
                value_count,
                segment_num_docs,
            });
        }
        if docids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ArgusError::CursorInvariant(
                "materialized numeric range docids are not strictly ascending",
            ));
        }
        if docids.is_empty() {
            return Ok(Self::empty());
        }
        Ok(Self {
            node: ScorerNode::NumericRange(NumericRangeScorer::new_materialized(
                docids,
                segment_num_docs,
                covers_every_document,
                boost,
            )),
        })
    }

    /// Replace one movement-only scorer tree's score with a finite constant.
    ///
    /// This is the canonical lowering for predicates such as a bounded glob:
    /// the child is materialized into its exact unique document set, so each
    /// document contributes the constant exactly once and the resulting cost
    /// is the true unique-match cardinality. The child must be safe for
    /// scoreless movement so no discarded BM25 arithmetic can affect traversal
    /// state.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite score or a child tree that requires scoring while
    /// advancing.
    pub fn constant_score(mut inner: Self, score: f32) -> Result<Self, ArgusError> {
        if !score.is_finite() {
            return Err(ArgusError::InvalidConstantBoost {
                boost_bits: score.to_bits(),
            });
        }
        if inner.is_explicit_empty() {
            return Ok(Self::empty());
        }
        if !inner.is_unscored_safe() {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        let segment_num_docs = inner.segment_num_docs();
        let mut docids = Vec::new();
        while let Some(doc) = inner.doc() {
            if docids.last().is_some_and(|previous| *previous >= doc) {
                return Err(ArgusError::CursorInvariant(
                    "constant-score child did not advance monotonically",
                ));
            }
            if docids.len() == docids.capacity() {
                docids.try_reserve(1).map_err(|_| ArgusError::Allocation {
                    resource: "constant-score docids",
                    count: docids.len().saturating_add(1),
                })?;
            }
            docids.push(doc);
            inner.next()?;
        }
        Ok(Self {
            node: ScorerNode::ConstantScore(ConstantScoreScorer {
                docids,
                index: 0,
                segment_num_docs,
                score,
            }),
        })
    }

    fn is_explicit_empty(&self) -> bool {
        matches!(self.node, ScorerNode::Empty)
    }

    fn is_raw_unit_all(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::All(scorer) if scorer.is_raw_unit_score()
        ) || matches!(
            &self.node,
            ScorerNode::NumericRange(scorer) if scorer.is_raw_unit_all()
        )
    }

    fn is_unscored_safe(&self) -> bool {
        match &self.node {
            ScorerNode::Empty
            | ScorerNode::All(_)
            | ScorerNode::Term(_)
            | ScorerNode::Phrase(_)
            | ScorerNode::NumericRange(_) => true,
            ScorerNode::ConstantScore(_) => true,
            ScorerNode::Intersection(scorer) => scorer.scorers.iter().all(Self::is_unscored_safe),
            ScorerNode::Union(_) => false,
            ScorerNode::UnscoredUnion(scorer) => scorer.active.iter().all(Self::is_unscored_safe),
            ScorerNode::RequiredOptional(scorer) => {
                scorer.required.is_unscored_safe() && scorer.optional.is_unscored_safe()
            }
            ScorerNode::Exclude(scorer) => {
                scorer.include.is_unscored_safe()
                    && scorer.excluded.iter().all(Self::is_unscored_safe)
            }
        }
    }

    fn competitive_score_upper_bound(&self) -> Option<f32> {
        match &self.node {
            ScorerNode::Term(term) => term.term_score_upper_bound,
            // A pure-term group (nested multi-field union) reports its ceiling
            // as the sum of its terms' ceilings. Behavior-neutral today: the
            // sole consumer (`competitive_candidates`) only reads this behind
            // the all-direct-terms gate, which a union child never satisfies;
            // the deferred grouped-MaxScore follow-up will consume it directly.
            ScorerNode::Union(union) => union.group_upper_bound(),
            _ => None,
        }
    }

    fn competitive_supports_block_max(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::Term(term)
                if term.term_score_upper_bound.is_some() && term.cursor.supports_block_max()
        )
    }

    fn competitive_is_direct_term(&self) -> bool {
        matches!(&self.node, ScorerNode::Term(_))
    }

    fn is_tantivy_topdocs_frequency_term(&self) -> bool {
        matches!(
            &self.node,
            ScorerNode::Term(term) if term.record_option != TermRecordOption::Basic
        )
    }

    fn competitive_term_cursor(&self) -> Option<CompetitiveTermCursor<'_>> {
        match &self.node {
            ScorerNode::Term(term) => term.competitive_cursor(),
            _ => None,
        }
    }

    /// A pure-term group eligible for grouped `MaxScore`: a union whose active
    /// children are all bounded direct terms, so it exposes a conservative
    /// whole-group ceiling. Deferred grouped-`MaxScore` foundation
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    fn competitive_is_groupable(&self) -> bool {
        matches!(&self.node, ScorerNode::Union(union) if union.group_upper_bound().is_some())
    }

    /// Append this group's competitive documents in `[_, horizon_end)` to
    /// `docs`, enumerated through fresh per-term competitive cursors so the
    /// group's own scan state is left untouched (mirrors how the term
    /// `MaxScore` path uses [`CompetitiveTermCursor`]). Returns `false`
    /// (fail-closed) when the scorer is not a pure-term group or any child
    /// lacks a competitive cursor, so the caller falls back to exhaustive.
    fn group_competitive_docs(
        &self,
        horizon_end: u64,
        docs: &mut Vec<u32>,
    ) -> Result<bool, ArgusError> {
        let ScorerNode::Union(union) = &self.node else {
            return Ok(false);
        };
        // A group's remaining documents are NOT `union.active`. The group is
        // itself a buffered union that has already scored a window and drained
        // the corresponding children, so its remaining set is the union of
        // three disjoint parts. Enumerating only the third silently drops the
        // first two from the root's candidate set (`bd-bt2t`).

        // (1) The document the group is currently positioned on.
        if let Some(doc) = union.current
            && u64::from(doc) < horizon_end
        {
            docs.push(doc);
        }

        // (2) Buffered residue: scored but not yet consumed by `next()`.
        // Covers [group window_start, group horizon).
        if let Some(window_start) = union.window_start {
            for (offset, score) in union
                .score_window
                .iter()
                .enumerate()
                .skip(union.scan_offset)
            {
                if score.is_none() {
                    continue;
                }
                let doc = u64::from(window_start).saturating_add(offset as u64);
                if doc >= horizon_end {
                    break;
                }
                docs.push(u32::try_from(doc).map_err(|_| {
                    ArgusError::CursorInvariant("buffered union residue docid does not fit u32")
                })?);
            }
        }

        // (3) Live children, which the group's own fill already advanced to at
        // least its horizon — so this cannot overlap (2).
        for child in &union.active {
            let Some(mut cursor) = child.competitive_term_cursor() else {
                return Ok(false);
            };
            while let Some(doc) = cursor.doc() {
                if u64::from(doc) >= horizon_end {
                    break;
                }
                docs.push(doc);
                cursor.next()?;
            }
        }
        Ok(true)
    }

    fn union_pruning_stats(&self) -> Option<UnionPruningStats> {
        let ScorerNode::Union(union) = &self.node else {
            return None;
        };
        Some(union.pruning_stats)
    }

    /// Return privacy-safe aggregate evidence for the pruning path that
    /// actually executed while this scorer was collected.
    #[must_use]
    pub(crate) fn pruning_telemetry(&self) -> PruningTelemetry {
        self.union_pruning_stats()
            .map_or_else(PruningTelemetry::default, PruningTelemetry::from)
    }

    #[cfg(test)]
    pub(crate) fn pruning_window_counts(&self) -> Option<(u64, u64)> {
        self.union_pruning_stats()
            .map(|stats| (stats.max_score_windows, stats.block_max_wand_windows))
    }

    /// Lower Boolean clauses into the pinned Tantivy scorer-tree shape.
    ///
    /// `Should` is required only when no positive `Must` exists. A raw
    /// `MustNot`-only node matches nothing.
    ///
    /// # Errors
    ///
    /// Returns a typed error if bounded scorer buffers cannot be allocated or
    /// initial cursor alignment detects malformed state.
    pub fn boolean(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Scored, false)
    }

    /// Lower a ranked root Boolean through Tantivy's frequency-term `TopDocs`
    /// accumulation order when the final scorer shape permits it.
    ///
    /// This is deliberately root-only: nested unions and counted collectors
    /// keep the ordinary exhaustive scorer order. The union constructor also
    /// fails closed unless every surviving child is a frequency-bearing term.
    ///
    /// # Errors
    ///
    /// Returns the same typed allocation, cursor-alignment, and scorer-shape
    /// failures as [`Self::boolean`].
    pub(crate) fn boolean_topdocs(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Scored, true)
    }

    /// Lower Boolean clauses into a genuinely scoreless matching tree.
    ///
    /// The caller must use this constructor recursively for nested Boolean
    /// children. Unlike the parity-pinned buffered scored union, its `Should`
    /// union advances lazily and never invokes BM25 scoring. Optional `Should`
    /// clauses are omitted when a positive `Must` already determines matching.
    ///
    /// # Errors
    ///
    /// Rejects a child tree containing a scored buffered union, or any ordinary
    /// allocation and cursor-alignment failure.
    pub fn boolean_unscored(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Unscored, false)
    }

    fn boolean_with_mode(
        clauses: Vec<ScorerClause<'a>>,
        mode: BooleanMode,
        topdocs_root: bool,
    ) -> Result<Self, ArgusError> {
        if mode == BooleanMode::Unscored
            && clauses
                .iter()
                .any(|clause| !clause.scorer.is_unscored_safe())
        {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        if matches!(
            clauses.as_slice(),
            [clause] if clause.occur == Occur::MustNot
        ) {
            return Ok(Self::empty());
        }
        let must_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Must)
            .count();
        let should_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Should)
            .count();
        let excluded_count = clauses.len() - must_count - should_count;
        let mut must = reserve_scorers("required Boolean clauses", must_count)?;
        let mut should = reserve_scorers("optional Boolean clauses", should_count)?;
        let mut excluded = reserve_scorers("excluded Boolean clauses", excluded_count)?;
        for clause in clauses {
            match clause.occur {
                Occur::Must if clause.scorer.is_explicit_empty() => return Ok(Self::empty()),
                Occur::Must => must.push(clause.scorer),
                Occur::Should if clause.scorer.is_explicit_empty() => {}
                Occur::Should => should.push(clause.scorer),
                Occur::MustNot if clause.scorer.is_explicit_empty() => {}
                Occur::MustNot => excluded.push(clause.scorer),
            }
        }

        // Tantivy removes direct, unboosted AllScorers before composing a
        // complex Boolean tree. A full-cardinality numeric range lowers to
        // that same raw AllScorer, while a boosted one is wrapped and remains
        // score-bearing. Keep one removed scorer per occurrence as the global
        // domain restoration token; duplicate unit contributions collapse.
        let must_all = remove_raw_unit_all_scorers(&mut must);
        let should_all = remove_raw_unit_all_scorers(&mut should);
        if remove_raw_unit_all_scorers(&mut excluded).is_some() {
            return Ok(Self::empty());
        }

        let required = if must.is_empty() {
            must_all
        } else {
            Some(scorer_intersection(must)?)
        };
        let include = match required {
            Some(required) => {
                // A raw optional All is score-neutral in Tantivy once a MUST
                // scorer determines matching. Non-All optional scorers retain
                // their scored RequiredOptional shape.
                if should.is_empty() || mode == BooleanMode::Unscored {
                    required
                } else {
                    let optional = scorer_union(should, false)?;
                    Self {
                        node: ScorerNode::RequiredOptional(RequiredOptionalScorer::new(
                            required, optional,
                        )?),
                    }
                }
            }
            None => match (should_all, mode) {
                (Some(all), BooleanMode::Unscored) => all,
                (Some(all), BooleanMode::Scored) if should.is_empty() => all,
                (Some(all), BooleanMode::Scored) => {
                    // Preserve Tantivy's nested score order: first aggregate
                    // the ordinary SHOULD scorers, then union one AllScorer.
                    let ordinary_should = scorer_union(should, false)?;
                    scorer_union(vec![ordinary_should, all], false)?
                }
                (None, BooleanMode::Scored) => {
                    scorer_union(should, topdocs_root && excluded.is_empty())?
                }
                (None, BooleanMode::Unscored) => scorer_union_unscored(should)?,
            },
        };
        if excluded.is_empty() {
            return Ok(include);
        }
        Ok(Self {
            node: ScorerNode::Exclude(ExcludeScorer::new(include, excluded)?),
        })
    }

    /// Current matching global document id.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        match &self.node {
            ScorerNode::Empty => None,
            ScorerNode::All(scorer) => scorer.doc(),
            ScorerNode::Term(scorer) => scorer.doc(),
            ScorerNode::Phrase(scorer) => scorer.doc(),
            ScorerNode::NumericRange(scorer) => scorer.doc(),
            ScorerNode::ConstantScore(scorer) => scorer.doc(),
            ScorerNode::Intersection(scorer) => scorer.doc(),
            ScorerNode::Union(scorer) => scorer.doc(),
            ScorerNode::UnscoredUnion(scorer) => scorer.doc(),
            ScorerNode::RequiredOptional(scorer) => scorer.doc(),
            ScorerNode::Exclude(scorer) => scorer.doc(),
        }
    }

    /// Runtime cost used for stable required-clause ordering.
    #[must_use]
    pub fn cost(&self) -> u64 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.cost(),
            ScorerNode::Term(scorer) => scorer.cost,
            ScorerNode::Phrase(scorer) => scorer.cost,
            ScorerNode::NumericRange(scorer) => scorer.cost(),
            ScorerNode::ConstantScore(scorer) => scorer.cost(),
            ScorerNode::Intersection(scorer) => scorer.cost(),
            ScorerNode::Union(scorer) => scorer.cost(),
            ScorerNode::UnscoredUnion(scorer) => scorer.cost(),
            ScorerNode::RequiredOptional(scorer) => scorer.cost(),
            ScorerNode::Exclude(scorer) => scorer.cost(),
        }
    }

    /// Tantivy-compatible best-effort match-count estimate.
    #[must_use]
    pub fn size_hint(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.size_hint(),
            ScorerNode::Term(scorer) => scorer.size_hint,
            ScorerNode::Phrase(scorer) => scorer.size_hint,
            ScorerNode::NumericRange(scorer) => scorer.size_hint(),
            ScorerNode::ConstantScore(scorer) => scorer.size_hint(),
            ScorerNode::Intersection(scorer) => scorer.size_hint(),
            ScorerNode::Union(scorer) => scorer.size_hint(),
            ScorerNode::UnscoredUnion(scorer) => scorer.size_hint(),
            ScorerNode::RequiredOptional(scorer) => scorer.size_hint(),
            ScorerNode::Exclude(scorer) => scorer.size_hint(),
        }
    }

    /// Live document count used for Tantivy-compatible composite estimates.
    #[must_use]
    pub fn segment_num_docs(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::All(scorer) => scorer.segment_num_docs,
            ScorerNode::Term(scorer) => scorer.segment_num_docs,
            ScorerNode::Phrase(scorer) => scorer.segment_num_docs,
            ScorerNode::NumericRange(scorer) => scorer.segment_num_docs,
            ScorerNode::ConstantScore(scorer) => scorer.segment_num_docs(),
            ScorerNode::Intersection(scorer) => scorer.segment_num_docs,
            ScorerNode::Union(scorer) => scorer.segment_num_docs,
            ScorerNode::UnscoredUnion(scorer) => scorer.segment_num_docs,
            ScorerNode::RequiredOptional(scorer) => scorer.segment_num_docs,
            ScorerNode::Exclude(scorer) => scorer.segment_num_docs,
        }
    }

    /// Move strictly forward to the next match.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    #[allow(
        clippy::should_implement_trait,
        reason = "a scorer exposes both next-match and inclusive advance operations"
    )]
    pub fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::All(scorer) => Ok(scorer.next()),
            ScorerNode::Term(scorer) => scorer.next(),
            ScorerNode::Phrase(scorer) => scorer.next(),
            ScorerNode::NumericRange(scorer) => Ok(scorer.next()),
            ScorerNode::ConstantScore(scorer) => Ok(scorer.next()),
            ScorerNode::Intersection(scorer) => scorer.next(),
            ScorerNode::Union(scorer) => scorer.next(),
            ScorerNode::UnscoredUnion(scorer) => scorer.next(),
            ScorerNode::RequiredOptional(scorer) => scorer.next(),
            ScorerNode::Exclude(scorer) => scorer.next(),
        }
    }

    /// Seek to the first match at or beyond `target`.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    pub fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::All(scorer) => Ok(scorer.seek(target)),
            ScorerNode::Term(scorer) => scorer.seek(target),
            ScorerNode::Phrase(scorer) => scorer.seek(target),
            ScorerNode::NumericRange(scorer) => Ok(scorer.seek(target)),
            ScorerNode::ConstantScore(scorer) => Ok(scorer.seek(target)),
            ScorerNode::Intersection(scorer) => scorer.seek(target),
            ScorerNode::Union(scorer) => scorer.seek(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek(target),
            ScorerNode::Exclude(scorer) => scorer.seek(target),
        }
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(SeekDangerResult::SeekLowerBound(None)),
            ScorerNode::All(scorer) => classify_seek_result(target, scorer.seek(target)),
            ScorerNode::Term(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::Phrase(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::NumericRange(scorer) => classify_seek_result(target, scorer.seek(target)),
            ScorerNode::ConstantScore(scorer) => scorer.seek_danger(target),
            ScorerNode::Intersection(scorer) => scorer.seek_danger(target),
            ScorerNode::Union(scorer) => scorer.seek_danger(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek_danger(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek_danger(target),
            ScorerNode::Exclude(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
        }
    }

    /// Score the current match in the pinned f32 scorer-tree order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for exhausted or inconsistent cursors, missing
    /// fieldnorms, and non-finite arithmetic.
    pub fn score(&mut self) -> Result<f32, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Err(ArgusError::CursorInvariant("cannot score an empty scorer")),
            ScorerNode::All(scorer) => scorer.score(),
            ScorerNode::Term(scorer) => scorer.score(),
            ScorerNode::Phrase(scorer) => scorer.score(),
            ScorerNode::NumericRange(scorer) => scorer.score(),
            ScorerNode::ConstantScore(scorer) => scorer.score(),
            ScorerNode::Intersection(scorer) => scorer.score(),
            ScorerNode::Union(scorer) => scorer.score(),
            ScorerNode::UnscoredUnion(_) => Err(ArgusError::ScoreUnavailable),
            ScorerNode::RequiredOptional(scorer) => scorer.score(),
            ScorerNode::Exclude(scorer) => scorer.score(),
        }
    }

    /// Exhaustively evaluate this scorer and retain its best `limit` live hits.
    ///
    /// The phase-one heap stores only `(global_docid, score)`. Deleted documents
    /// are filtered here, after physical posting/statistics construction but
    /// before they can affect the heap threshold.
    ///
    /// # Errors
    ///
    /// Propagates typed scorer failures or a bounded heap allocation failure.
    pub fn top_k<L>(&mut self, limit: usize, live_docs: &L) -> Result<Vec<ScoredDoc>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = TopDocsCollector::new(limit, 0)?;
        collector.collect(self, live_docs)?;
        Ok(collector.finish()?.hits)
    }

    /// Exhaust this recursively unscored tree into sorted unique global docids.
    ///
    /// Use [`Self::boolean_unscored`] for every Boolean level. Term and phrase
    /// leaves are intrinsically safe because advancing them never evaluates a
    /// BM25 score.
    ///
    /// # Errors
    ///
    /// Rejects a scored buffered union anywhere in the tree and propagates
    /// cursor or bounded-allocation failures.
    pub fn collect_doc_set<L>(&mut self, live_docs: &L) -> Result<Vec<u32>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = DocSetCollector::new();
        collector.collect(self, live_docs)?;
        Ok(collector.finish())
    }

    fn collect_with<L>(
        &mut self,
        collector: &mut CollectorState<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if matches!(collector, CollectorState::DocSet(_)) && !self.is_unscored_safe() {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        if let CollectorState::TopDocs(top_docs) = collector
            && !top_docs.exact_count
            && let ScorerNode::Union(union) = &mut self.node
        {
            return union.collect_top_docs(top_docs, live_docs);
        }
        while let Some(doc) = self.doc() {
            match collector {
                CollectorState::TopDocs(top_docs) => {
                    // Scoring physical deleted matches before visibility
                    // filtering preserves the buffered union's f32 state.
                    let score = (top_docs.retained != 0).then(|| self.score()).transpose()?;
                    if live_docs.is_live(doc) {
                        top_docs.record_live(doc, score)?;
                    }
                }
                CollectorState::DocSet(doc_set) => {
                    if live_docs.is_live(doc) {
                        doc_set.push(doc)?;
                    }
                }
            }
            self.next()?;
        }
        Ok(())
    }
}

fn reserve_scorers<'a>(
    resource: &'static str,
    count: usize,
) -> Result<Vec<ReferenceScorer<'a>>, ArgusError> {
    let mut scorers = Vec::new();
    scorers
        .try_reserve_exact(count)
        .map_err(|_| ArgusError::Allocation { resource, count })?;
    Ok(scorers)
}

fn remove_raw_unit_all_scorers<'a>(
    scorers: &mut Vec<ReferenceScorer<'a>>,
) -> Option<ReferenceScorer<'a>> {
    let mut retained_all = None;
    let mut index = 0;
    while index < scorers.len() {
        if scorers[index].is_raw_unit_all() {
            let all = scorers.remove(index);
            if retained_all.is_none() {
                retained_all = Some(all);
            }
        } else {
            index += 1;
        }
    }
    retained_all
}

fn shared_segment_num_docs(scorers: &[ReferenceScorer<'_>]) -> Result<u32, ArgusError> {
    let segment_num_docs = scorers.first().map_or(0, ReferenceScorer::segment_num_docs);
    if scorers
        .iter()
        .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
    {
        return Err(ArgusError::CursorInvariant(
            "Boolean children belong to different segment domains",
        ));
    }
    Ok(segment_num_docs)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to u32 input bounds"
)]
fn estimate_intersection(mut sizes: impl Iterator<Item = u32>, segment_num_docs: u32) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let Some(first) = sizes.next() else {
        return 0;
    };
    let mut co_location_factor = 1.3_f64;
    let mut estimate = f64::from(first);
    let mut smallest = estimate;
    for size in sizes {
        co_location_factor = (co_location_factor - 0.1).max(1.0);
        estimate *= (f64::from(size) / f64::from(segment_num_docs)) * co_location_factor;
        smallest = smallest.min(f64::from(size));
    }
    estimate.round().min(smallest) as u32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to segment_num_docs"
)]
fn estimate_union(sizes: impl Iterator<Item = u32>, segment_num_docs: u32) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let mut not_in_any_set_probability = 1.0_f64;
    for size in sizes {
        let probability = (f64::from(size) / f64::from(segment_num_docs)) * 0.8;
        not_in_any_set_probability *= 1.0 - probability;
    }
    let estimate = (f64::from(segment_num_docs) * (1.0 - not_in_any_set_probability)).round();
    estimate.min(f64::from(segment_num_docs)) as u32
}

fn scorer_intersection(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "required scorer count changed during lowering",
        )),
        _ => {
            let intersection = IntersectionScorer::new(scorers)?;
            if intersection.doc().is_none() {
                Ok(ReferenceScorer::empty())
            } else {
                Ok(ReferenceScorer {
                    node: ScorerNode::Intersection(intersection),
                })
            }
        }
    }
}

fn scorer_union(
    mut scorers: Vec<ReferenceScorer<'_>>,
    topdocs_root: bool,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "optional scorer count changed during lowering",
        )),
        _ => Ok(ReferenceScorer {
            node: ScorerNode::Union(BufferedUnionScorer::new(scorers, topdocs_root)?),
        }),
    }
}

fn scorer_union_unscored(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "unscored optional scorer count changed during lowering",
        )),
        _ => Ok(ReferenceScorer {
            node: ScorerNode::UnscoredUnion(UnscoredUnionScorer::new(scorers)?),
        }),
    }
}

/// Allocation-free cursor over one physical segment document-id domain.
///
/// Unlike Tantivy's local-docid `AllScorer`, this cursor retains global bounds
/// and an explicit exhausted state so `u32::MAX` remains a real match.
struct AllScorer {
    current: Option<u32>,
    docid_hi: u64,
    span: u64,
    segment_num_docs: u32,
    score: f32,
}

impl AllScorer {
    fn new(
        docid_lo: u64,
        docid_hi: u64,
        segment_num_docs: u32,
        boost: f32,
    ) -> Result<Self, ArgusError> {
        if !boost.is_finite() {
            return Err(ArgusError::InvalidAllBoost {
                boost_bits: boost.to_bits(),
            });
        }
        let invalid = |reason| ArgusError::InvalidAllRange {
            docid_lo,
            docid_hi,
            segment_num_docs,
            reason,
        };
        if docid_lo > docid_hi {
            return Err(invalid("the lower bound exceeds the upper bound"));
        }
        if docid_hi > MAX_GLOBAL_DOCID_EXCLUSIVE {
            return Err(invalid("the upper bound exceeds the u32 payload domain"));
        }
        let span = docid_hi - docid_lo;
        if u64::from(segment_num_docs) > span {
            return Err(invalid("the live document count exceeds the range span"));
        }
        let current =
            if span == 0 {
                None
            } else {
                Some(u32::try_from(docid_lo).map_err(|_| {
                    invalid("the non-empty lower bound exceeds the u32 payload domain")
                })?)
            };
        Ok(Self {
            current,
            docid_hi,
            span,
            segment_num_docs,
            score: boost,
        })
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    const fn cost(&self) -> u64 {
        self.span
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.span).unwrap_or(u32::MAX)
    }

    fn is_raw_unit_score(&self) -> bool {
        self.score.to_bits() == 1.0_f32.to_bits()
    }

    fn next(&mut self) -> Option<u32> {
        let current = self.current?;
        let next = u64::from(current) + 1;
        self.current = if next < self.docid_hi {
            u32::try_from(next).ok()
        } else {
            None
        };
        self.current
    }

    fn seek(&mut self, target: u32) -> Option<u32> {
        let current = self.current?;
        if current >= target {
            return Some(current);
        }
        self.current = (u64::from(target) < self.docid_hi).then_some(target);
        self.current
    }

    fn score(&self) -> Result<f32, ArgusError> {
        self.current
            .ok_or(ArgusError::CursorInvariant(
                "cannot score an exhausted All scorer",
            ))
            .map(|_| self.score)
    }
}

/// Owned lower-bound cursor over one materialized NUMERIC predicate.
///
/// Movement uses the exact sorted matches, while planning metadata mirrors
/// Tantivy's runtime fast-field range scorer. Keeping those concerns separate
/// preserves both filter-first seeking and the oracle's stable f32 score order.
struct NumericRangeScorer {
    docids: NumericRangeDocIds,
    index: usize,
    cost: u64,
    size_hint: u32,
    segment_num_docs: u32,
    covers_every_document: bool,
    score: f32,
}

impl NumericRangeScorer {
    fn new(docids: NumericDocIdSet, value_count: usize, segment_num_docs: u32, score: f32) -> Self {
        let covers_every_document = value_count
            == usize::try_from(segment_num_docs).unwrap_or(usize::MAX)
            && u32::try_from(docids.len()).unwrap_or(u32::MAX) == segment_num_docs;
        Self::from_docids(
            NumericRangeDocIds::Encoded(docids),
            segment_num_docs,
            covers_every_document,
            score,
        )
    }

    fn new_materialized(
        docids: Vec<u32>,
        segment_num_docs: u32,
        covers_every_document: bool,
        score: f32,
    ) -> Self {
        Self::from_docids(
            NumericRangeDocIds::Materialized(docids),
            segment_num_docs,
            covers_every_document,
            score,
        )
    }

    fn from_docids(
        docids: NumericRangeDocIds,
        segment_num_docs: u32,
        covers_every_document: bool,
        score: f32,
    ) -> Self {
        let (cost, size_hint) = if covers_every_document {
            (u64::from(segment_num_docs), segment_num_docs)
        } else {
            (tantivy_range_cost(segment_num_docs), segment_num_docs / 10)
        };
        Self {
            docids,
            index: 0,
            cost,
            size_hint,
            segment_num_docs,
            covers_every_document,
            score,
        }
    }

    fn doc(&self) -> Option<u32> {
        self.docids.as_slice().get(self.index).copied()
    }

    const fn cost(&self) -> u64 {
        self.cost
    }

    const fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn is_raw_unit_all(&self) -> bool {
        self.covers_every_document && self.score.to_bits() == 1.0_f32.to_bits()
    }

    fn next(&mut self) -> Option<u32> {
        if self.index < self.docids.len() {
            self.index += 1;
        }
        self.doc()
    }

    fn seek(&mut self, target: u32) -> Option<u32> {
        let tail = self.docids.as_slice().get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|&docid| docid < target);
        self.doc()
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted numeric range",
        ))?;
        finite_score(self.score, doc)
    }
}

enum NumericRangeDocIds {
    Encoded(NumericDocIdSet),
    Materialized(Vec<u32>),
}

impl NumericRangeDocIds {
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Encoded(docids) => docids.as_slice(),
            Self::Materialized(docids) => docids,
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "contract parity copies Tantivy RangeDocSet::cost's nonnegative f64-to-u64 truncation"
)]
fn tantivy_range_cost(segment_num_docs: u32) -> u64 {
    (f64::from(segment_num_docs) * 0.8) as u64
}

/// Materialized constant-score cursor over one exact unique document set.
struct ConstantScoreScorer {
    docids: Vec<u32>,
    index: usize,
    segment_num_docs: u32,
    score: f32,
}

impl ConstantScoreScorer {
    fn doc(&self) -> Option<u32> {
        self.docids.get(self.index).copied()
    }

    fn cost(&self) -> u64 {
        u64::try_from(self.docids.len()).unwrap_or(u64::MAX)
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.docids.len()).unwrap_or(u32::MAX)
    }

    const fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Option<u32> {
        if self.index < self.docids.len() {
            self.index += 1;
        }
        self.doc()
    }

    fn seek(&mut self, target: u32) -> Option<u32> {
        let tail = self.docids.get(self.index..).unwrap_or_default();
        self.index += tail.partition_point(|&docid| docid < target);
        self.doc()
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        let result = self.seek(target);
        classify_seek_result(target, result)
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted constant-score predicate",
        ))?;
        finite_score(self.score, doc)
    }
}

/// Lazy scoreless disjunction used only by doc-set collection.
///
/// The scored parity path keeps [`BufferedUnionScorer`] unchanged. This cursor
/// advances all children at the current minimum document together, so it
/// deduplicates matches without evaluating or retaining any BM25 contribution.
struct UnscoredUnionScorer<'a> {
    active: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    segment_num_docs: u32,
}

impl<'a> UnscoredUnionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.retain(|scorer| scorer.doc().is_some());
        let current = scorers.iter().filter_map(ReferenceScorer::doc).min();
        Ok(Self {
            active: scorers,
            current,
            segment_num_docs,
        })
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn refresh_current(&mut self) -> Option<u32> {
        self.active.retain(|scorer| scorer.doc().is_some());
        self.current = self.active.iter().filter_map(ReferenceScorer::doc).min();
        self.current
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        for scorer in &mut self.active {
            if scorer.doc() == Some(current) {
                scorer.next()?;
            }
        }
        Ok(self.refresh_current())
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        for scorer in &mut self.active {
            if scorer.doc().is_some_and(|doc| doc < target) {
                scorer.advance(target)?;
            }
        }
        Ok(self.refresh_current())
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        let result = self.seek(target)?;
        classify_seek_result(target, result)
    }
}

struct IntersectionScorer<'a> {
    scorers: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    segment_num_docs: u32,
}

impl<'a> IntersectionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.sort_by_key(ReferenceScorer::cost);
        let mut scorer = Self {
            scorers,
            current: None,
            segment_num_docs,
        };
        scorer.align()?;
        Ok(scorer)
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.scorers.first().map_or(0, ReferenceScorer::cost)
    }

    fn size_hint(&self) -> u32 {
        estimate_intersection(
            self.scorers.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        let mut candidate = match self.scorers.iter().filter_map(ReferenceScorer::doc).max() {
            Some(candidate) if self.scorers.iter().all(|scorer| scorer.doc().is_some()) => {
                candidate
            }
            _ => {
                self.current = None;
                return Ok(None);
            }
        };
        'outer: loop {
            for scorer in &mut self.scorers {
                let Some(doc) = scorer.advance(candidate)? else {
                    self.current = None;
                    return Ok(None);
                };
                if doc > candidate {
                    candidate = doc;
                    continue 'outer;
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let Some(mut candidate) = current.checked_add(1) else {
            self.scorers[0].next()?;
            self.current = None;
            return Ok(None);
        };

        'outer: loop {
            let Some(driver_doc) = self.scorers[0].advance(candidate)? else {
                self.current = None;
                return Ok(None);
            };
            candidate = driver_doc;
            for scorer in &mut self.scorers[1..] {
                match scorer.seek_danger(candidate)? {
                    SeekDangerResult::Found => {}
                    SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                        if lower_bound <= candidate {
                            return Err(ArgusError::CursorInvariant(
                                "danger seek lower bound did not strictly advance",
                            ));
                        }
                        candidate = lower_bound;
                        continue 'outer;
                    }
                    SeekDangerResult::SeekLowerBound(None) => {
                        self.current = None;
                        return Ok(None);
                    }
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        self.scorers[0].advance(target)?;
        self.align()
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        for scorer in &mut self.scorers {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {}
                lower_bound @ SeekDangerResult::SeekLowerBound(_) => return Ok(lower_bound),
            }
        }
        self.current = Some(target);
        Ok(SeekDangerResult::Found)
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted intersection",
        ))?;
        let (left_slice, rest) = self.scorers.split_at_mut(1);
        let left = left_slice[0].score()?;
        let (right_slice, others) = rest.split_at_mut(1);
        let right = right_slice[0].score()?;
        let mut other_sum = 0.0_f32;
        for scorer in others {
            other_sum += scorer.score()?;
        }
        finite_score(left + right + other_sum, doc)
    }
}

// Rank-safety argument for every pruning comparison:
//
// * each sealed/Delta leaf first recomputes its tf ceiling from conservative
//   `(max_frequency, min_fieldnorm)` metadata using the live snapshot avgdl and
//   the live idf/boost weight;
// * all contributions are non-negative, so the exact f64 sum is inflated by
//   the standard `n * EPSILON` forward-error envelope before rounding outward
//   to f32. The result therefore covers any scorer-vector addition order;
// * callers skip only when this ceiling is strictly below the running kth
//   score. Equality remains competitive because a smaller docid can win a tie.
//
// MaxScore may consequently omit a document that matches only the
// non-essential prefix, while BMW may skip an interval only after summing every
// term whose current doc can occur in that interval. Neither proof depends on
// the avgdl regime under which a sealed block was created.
fn conservative_bound_sum(bounds: impl IntoIterator<Item = f32>) -> Option<f32> {
    conservative_optional_bound_sum(bounds.into_iter().map(Some))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the checked finite f64 envelope is deliberately rounded outward to f32"
)]
fn conservative_optional_bound_sum(bounds: impl IntoIterator<Item = Option<f32>>) -> Option<f32> {
    let mut exact = 0.0_f64;
    let mut count = 0_u32;
    for bound in bounds {
        let bound = bound?;
        if !bound.is_finite() || bound.is_sign_negative() {
            return None;
        }
        exact += f64::from(bound);
        count = count.checked_add(1)?;
    }
    if !exact.is_finite() || exact > f64::from(f32::MAX) {
        return None;
    }
    let rounding_budget = f64::from(count) * f64::from(f32::EPSILON);
    if rounding_budget >= 1.0 {
        return None;
    }
    let inflated = exact / (1.0 - rounding_budget);
    if !inflated.is_finite() || inflated > f64::from(f32::MAX) {
        return None;
    }
    let rounded = inflated as f32;
    if f64::from(rounded) >= inflated {
        Some(rounded)
    } else {
        Some(f32::from_bits(rounded.to_bits().checked_add(1)?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnionPruningStrategy {
    MaxScore,
    BlockMaxWand,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UnionPruningStats {
    max_score_windows: u64,
    block_max_wand_windows: u64,
    blocks_skipped: u64,
    candidate_docs: u64,
}

/// Aggregate scorer evidence suitable for stable tracing fields.
///
/// This deliberately excludes terms, document identifiers, and cutoff scores.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PruningTelemetry {
    max_score_windows: u64,
    block_max_wand_windows: u64,
    blocks_skipped: u64,
    candidate_docs: u64,
}

impl From<UnionPruningStats> for PruningTelemetry {
    fn from(stats: UnionPruningStats) -> Self {
        Self {
            max_score_windows: stats.max_score_windows,
            block_max_wand_windows: stats.block_max_wand_windows,
            blocks_skipped: stats.blocks_skipped,
            candidate_docs: stats.candidate_docs,
        }
    }
}

impl PruningTelemetry {
    /// Stable name for the pruning path observed during collection.
    #[must_use]
    pub(crate) const fn plan(self) -> &'static str {
        match (
            self.max_score_windows != 0,
            self.block_max_wand_windows != 0,
        ) {
            (false, false) => "exhaustive",
            (true, false) => "max_score",
            (false, true) => "block_max_wand",
            (true, true) => "mixed_pruning",
        }
    }

    #[must_use]
    pub(crate) const fn pruning_windows(self) -> u64 {
        self.max_score_windows
            .saturating_add(self.block_max_wand_windows)
    }

    #[must_use]
    pub(crate) const fn blocks_skipped(self) -> u64 {
        self.blocks_skipped
    }

    #[must_use]
    pub(crate) const fn candidate_docs(self) -> u64 {
        self.candidate_docs
    }
}

struct CompetitiveCandidates {
    strategy: UnionPruningStrategy,
    docs: Vec<u32>,
    blocks_skipped: u64,
}

struct BufferedUnionScorer<'a> {
    active: Vec<ReferenceScorer<'a>>,
    /// Ranked `TopDocs` over a direct frequency-term root follows Tantivy's
    /// specialized term-union traversal. That path repeatedly scores the
    /// scorers on the current minimum document, swap-removes exhausted
    /// scorers, and stably re-sorts survivors by their next document. The
    /// resulting f32 addition order is observable in raw score bits.
    tantivy_topdocs_term_union: bool,
    score_window: Vec<Option<f32>>,
    window_start: Option<u32>,
    scan_offset: usize,
    current: Option<u32>,
    current_score: f32,
    segment_num_docs: u32,
    pruning_stats: UnionPruningStats,
    /// Whole-group score ceiling captured once at construction, before the
    /// first refill drains any child into [`Self::score_window`].
    ///
    /// This must not be recomputed from [`Self::active`]: a group whose
    /// postings all land inside one `UNION_HORIZON` window has an empty
    /// `active` immediately after construction, yet its buffered residue can
    /// still realize scores. A sum over *all* original children stays a valid
    /// upper bound for any surviving subset — looser as the group drains,
    /// never lower than a realized score, which is the rank-safety
    /// precondition the grouped candidate path relies on (`bd-bt2t`).
    ///
    /// `None` when any original child was not a bounded direct term, so
    /// callers fail closed for the whole lifetime of the group.
    group_ceiling: Option<f32>,
}

impl<'a> BufferedUnionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>, topdocs_root: bool) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.retain(|scorer| scorer.doc().is_some());
        let tantivy_topdocs_term_union = topdocs_root
            && scorers
                .iter()
                .all(ReferenceScorer::is_tantivy_topdocs_frequency_term);
        let mut score_window = Vec::new();
        score_window
            .try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "buffered union score window",
                count: UNION_HORIZON,
            })?;
        score_window.resize(UNION_HORIZON, None);
        // Capture the whole-group ceiling from the original children, before
        // the refill below drains any of them into the score window.
        let group_ceiling = if scorers
            .iter()
            .all(ReferenceScorer::competitive_is_direct_term)
        {
            conservative_optional_bound_sum(
                scorers
                    .iter()
                    .map(ReferenceScorer::competitive_score_upper_bound),
            )
        } else {
            None
        };
        let mut scorer = Self {
            active: scorers,
            tantivy_topdocs_term_union,
            score_window,
            window_start: None,
            scan_offset: 0,
            current: None,
            current_score: 0.0,
            segment_num_docs,
            pruning_stats: UnionPruningStats::default(),
            group_ceiling,
        };
        if scorer.refill()? {
            scorer.advance_buffered();
        }
        Ok(scorer)
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn clear_buffer(&mut self) {
        self.score_window.fill(None);
        self.scan_offset = 0;
        self.current = None;
        self.current_score = 0.0;
    }

    fn refill(&mut self) -> Result<bool, ArgusError> {
        self.refill_with_cutoff(None)
    }

    /// Whole-group score ceiling for a pure-term group: the conservative sum of
    /// every active child's whole-term ceiling. Returns `None` when the group is
    /// empty or any active child is not a bounded direct term (a nested non-term
    /// child, or a cursor without a ceiling), so callers can fail closed.
    ///
    /// Foundation for the deferred grouped-`MaxScore` lever
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`): a nested pure-term union exposes
    /// this ceiling to a root `MaxScore` over groups. Uses the same
    /// [`conservative_optional_bound_sum`] as the term `MaxScore` path so the
    /// ceiling is rounded outward and never falls below a realized group score
    /// — a rank-safety precondition for the follow-up grouped candidate path.
    ///
    /// Reads the construction-time [`Self::group_ceiling`] rather than summing
    /// over `active`. `active` is NOT the group's remaining-work set: a group
    /// whose postings all fall inside one window drains to empty at
    /// construction while its scores live on in `score_window`, so a live sum
    /// would report `None` for a group that can still realize hits — the
    /// `max_score_windows: 0` regression in `bd-bt2t`.
    ///
    /// Safe to consume only because [`ReferenceScorer::group_competitive_docs`]
    /// now enumerates buffered residue alongside live cursors. Before that fix
    /// this returning `Some` for a drained group meant silently dropping its
    /// documents from the top-k; the two changes must stay together.
    const fn group_upper_bound(&self) -> Option<f32> {
        self.group_ceiling
    }

    fn refill_with_cutoff(&mut self, cutoff: Option<f32>) -> Result<bool, ArgusError> {
        self.clear_buffer();
        let Some(window_start) = self.active.iter().filter_map(ReferenceScorer::doc).min() else {
            self.window_start = None;
            return Ok(false);
        };
        self.window_start = Some(window_start);
        let horizon_end = u64::from(window_start) + UNION_HORIZON_U64;
        if let Some(cutoff) = cutoff
            && cutoff.is_finite()
            && let Some(candidates) =
                self.competitive_candidates(window_start, horizon_end, cutoff)?
        {
            match candidates.strategy {
                UnionPruningStrategy::MaxScore => {
                    self.pruning_stats.max_score_windows =
                        self.pruning_stats.max_score_windows.saturating_add(1);
                }
                UnionPruningStrategy::BlockMaxWand => {
                    self.pruning_stats.block_max_wand_windows =
                        self.pruning_stats.block_max_wand_windows.saturating_add(1);
                }
            }
            self.pruning_stats.blocks_skipped = self
                .pruning_stats
                .blocks_skipped
                .saturating_add(candidates.blocks_skipped);
            self.pruning_stats.candidate_docs = self
                .pruning_stats
                .candidate_docs
                .saturating_add(u64::try_from(candidates.docs.len()).unwrap_or(u64::MAX));
            self.fill_candidate_window(window_start, horizon_end, &candidates.docs)?;
            return Ok(true);
        }
        self.fill_exhaustive_window(window_start, horizon_end)?;
        Ok(true)
    }

    fn fill_exhaustive_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
    ) -> Result<(), ArgusError> {
        if self.tantivy_topdocs_term_union {
            return self.fill_tantivy_topdocs_term_window(window_start, horizon_end);
        }

        let mut index = 0;
        while index < self.active.len() {
            loop {
                let Some(doc) = self.active[index].doc() else {
                    self.active.swap_remove(index);
                    break;
                };
                if u64::from(doc) >= horizon_end {
                    index += 1;
                    break;
                }
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let contribution = self.active[index].score()?;
                let total = self.score_window[offset].unwrap_or(0.0) + contribution;
                self.score_window[offset] = Some(finite_score(total, doc)?);
                self.active[index].next()?;
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Fill one exhaustive window in the same evolving scorer order as
    /// Tantivy's direct frequency-term `BlockWAND` specialization while its
    /// threshold is `Score::MIN`.
    ///
    /// Tantivy initially performs a stable sort by current document. At each
    /// pivot it sums the equal-document prefix, advances that prefix,
    /// swap-removes every exhausted scorer, then stably sorts the survivors by
    /// their new document. Quill buffers documents in windows, but preserving
    /// these mutations across refill boundaries retains the same per-document
    /// f32 accumulation order.
    fn fill_tantivy_topdocs_term_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
    ) -> Result<(), ArgusError> {
        self.active.sort_by_key(ReferenceScorer::doc);
        while let Some(doc) = self.active.first().and_then(ReferenceScorer::doc) {
            if u64::from(doc) >= horizon_end {
                break;
            }
            let pivot_len = self
                .active
                .iter()
                .take_while(|scorer| scorer.doc() == Some(doc))
                .count();
            let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
            let mut score = 0.0_f32;
            for scorer in &mut self.active[..pivot_len] {
                score += scorer.score()?;
            }
            self.score_window[offset] = Some(finite_score(score, doc)?);

            for scorer in &mut self.active[..pivot_len] {
                scorer.next()?;
            }
            let mut index = 0;
            while index < self.active.len() {
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                } else {
                    index += 1;
                }
            }
            self.active.sort_by_key(ReferenceScorer::doc);
        }
        Ok(())
    }

    fn competitive_candidates(
        &self,
        window_start: u32,
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        // Grouped MaxScore: a nested union whose active children are all
        // pure-term groups (multi-field terms) rather than direct terms
        // (`bd-quill-e8-perf-doctrine-x4e4.5.1`). Dormant until the query gate
        // opens BLOCKMAX for group terms — without their whole-term ceilings
        // `competitive_is_groupable` is `false`, so this branch is skipped and
        // nested unions keep scoring exhaustively (behavior-neutral today).
        if (2..=MAX_SCORE_MAX_CLAUSES).contains(&self.active.len())
            && !self
                .active
                .iter()
                .all(ReferenceScorer::competitive_is_direct_term)
            && self
                .active
                .iter()
                .all(ReferenceScorer::competitive_is_groupable)
        {
            let Some(candidates) =
                Self::grouped_max_score_candidates(&self.active, horizon_end, cutoff)?
            else {
                return Ok(None);
            };
            if candidates
                .docs
                .iter()
                .any(|doc| *doc < window_start || u64::from(*doc) >= horizon_end)
            {
                return Err(ArgusError::CursorInvariant(
                    "grouped competitive candidate escaped its union window",
                ));
            }
            return Ok(Some(candidates));
        }
        let strategy = match self.active.len() {
            2..=MAX_SCORE_MAX_CLAUSES
                if self
                    .active
                    .iter()
                    .all(ReferenceScorer::competitive_is_direct_term) =>
            {
                UnionPruningStrategy::MaxScore
            }
            2..=MAX_SCORE_MAX_CLAUSES => return Ok(None),
            BMW_MIN_CLAUSES..
                if self
                    .active
                    .iter()
                    .map(ReferenceScorer::cost)
                    .fold(0_u64, u64::saturating_add)
                    >= BMW_MIN_TOTAL_COST =>
            {
                UnionPruningStrategy::BlockMaxWand
            }
            _ => return Ok(None),
        };
        match strategy {
            UnionPruningStrategy::MaxScore => {
                let mut smallest = f32::INFINITY;
                for scorer in &self.active {
                    let Some(bound) = scorer.competitive_score_upper_bound() else {
                        return Ok(None);
                    };
                    smallest = smallest.min(bound);
                }
                if smallest >= cutoff {
                    return Ok(None);
                }
            }
            UnionPruningStrategy::BlockMaxWand => {
                if self
                    .active
                    .iter()
                    .any(|scorer| !scorer.competitive_supports_block_max())
                {
                    return Ok(None);
                }
            }
        }
        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(self.active.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "competitive term cursors",
                count: self.active.len(),
            })?;
        for scorer in &self.active {
            let Some(cursor) = scorer.competitive_term_cursor() else {
                return Ok(None);
            };
            cursors.push(cursor);
        }
        let candidates = match strategy {
            UnionPruningStrategy::MaxScore => {
                let Some(candidates) =
                    Self::max_score_candidates(&mut cursors, horizon_end, cutoff)?
                else {
                    return Ok(None);
                };
                candidates
            }
            UnionPruningStrategy::BlockMaxWand => {
                Self::block_max_wand_candidates(&mut cursors, horizon_end, cutoff)?
            }
        };
        if candidates
            .docs
            .iter()
            .any(|doc| *doc < window_start || u64::from(*doc) >= horizon_end)
        {
            return Err(ArgusError::CursorInvariant(
                "competitive candidate escaped its union window",
            ));
        }
        Ok(Some(candidates))
    }

    fn max_score_candidates(
        cursors: &mut [CompetitiveTermCursor<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        let mut by_upper_bound = Vec::new();
        by_upper_bound
            .try_reserve_exact(cursors.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "MaxScore bound order",
                count: cursors.len(),
            })?;
        by_upper_bound.extend(0..cursors.len());
        by_upper_bound.sort_unstable_by(|left, right| {
            cursors[*left]
                .term_upper_bound()
                .total_cmp(&cursors[*right].term_upper_bound())
                .then_with(|| left.cmp(right))
        });

        let mut nonessential = 0_usize;
        while nonessential < by_upper_bound.len() {
            let prefix = &by_upper_bound[..=nonessential];
            let bound = conservative_bound_sum(
                prefix
                    .iter()
                    .map(|index| cursors[*index].term_upper_bound()),
            )
            .ok_or(ArgusError::CursorInvariant(
                "MaxScore bound sum was not finite and conservative",
            ))?;
            if bound < cutoff {
                nonessential += 1;
            } else {
                break;
            }
        }
        if nonessential == 0 {
            return Ok(None);
        }

        let maximum_candidates = UNION_HORIZON
            .checked_mul(cursors.len().saturating_sub(nonessential))
            .ok_or(ArgusError::Allocation {
                resource: "MaxScore candidates",
                count: usize::MAX,
            })?;
        let mut docs = Vec::new();
        docs.try_reserve_exact(maximum_candidates)
            .map_err(|_| ArgusError::Allocation {
                resource: "MaxScore candidates",
                count: maximum_candidates,
            })?;
        for &index in &by_upper_bound[nonessential..] {
            while let Some(doc) = cursors[index].doc() {
                if u64::from(doc) >= horizon_end {
                    break;
                }
                docs.push(doc);
                cursors[index].next()?;
            }
        }
        docs.sort_unstable();
        docs.dedup();
        Ok(Some(CompetitiveCandidates {
            strategy: UnionPruningStrategy::MaxScore,
            docs,
            blocks_skipped: 0,
        }))
    }

    /// Grouped `MaxScore` over pure-term groups (nested multi-field unions):
    /// the direct analog of [`Self::max_score_candidates`] with each group's
    /// conservative whole-group ceiling as the partition key and each group's
    /// per-term competitive cursors as the doc source. A candidate document
    /// must appear in an *essential* group; documents confined to the
    /// low-ceiling non-essential groups cannot reach `cutoff` and are skipped.
    /// Fails closed (`Ok(None)` → exhaustive) if any group lacks a ceiling or a
    /// competitive cursor. Rank-safety and f32 score-bit parity are gated by
    /// the exhaustive parity harness that lands with the query gate
    /// (`bd-quill-e8-perf-doctrine-x4e4.5.1`).
    fn grouped_max_score_candidates(
        groups: &[ReferenceScorer<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<Option<CompetitiveCandidates>, ArgusError> {
        let mut ceilings = Vec::new();
        ceilings
            .try_reserve_exact(groups.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "grouped MaxScore ceilings",
                count: groups.len(),
            })?;
        for group in groups {
            let Some(bound) = group.competitive_score_upper_bound() else {
                return Ok(None);
            };
            ceilings.push(bound);
        }

        let mut by_upper_bound = Vec::new();
        by_upper_bound
            .try_reserve_exact(groups.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "grouped MaxScore bound order",
                count: groups.len(),
            })?;
        by_upper_bound.extend(0..groups.len());
        by_upper_bound.sort_unstable_by(|left, right| {
            ceilings[*left]
                .total_cmp(&ceilings[*right])
                .then_with(|| left.cmp(right))
        });

        let mut nonessential = 0_usize;
        while nonessential < by_upper_bound.len() {
            let bound = conservative_bound_sum(
                by_upper_bound[..=nonessential]
                    .iter()
                    .map(|index| ceilings[*index]),
            )
            .ok_or(ArgusError::CursorInvariant(
                "grouped MaxScore bound sum was not finite and conservative",
            ))?;
            if bound < cutoff {
                nonessential += 1;
            } else {
                break;
            }
        }
        if nonessential == 0 {
            return Ok(None);
        }

        let mut docs = Vec::new();
        for &index in &by_upper_bound[nonessential..] {
            if !groups[index].group_competitive_docs(horizon_end, &mut docs)? {
                return Ok(None);
            }
        }
        docs.sort_unstable();
        docs.dedup();
        Ok(Some(CompetitiveCandidates {
            strategy: UnionPruningStrategy::MaxScore,
            docs,
            blocks_skipped: 0,
        }))
    }

    fn block_max_wand_candidates(
        cursors: &mut [CompetitiveTermCursor<'_>],
        horizon_end: u64,
        cutoff: f32,
    ) -> Result<CompetitiveCandidates, ArgusError> {
        let mut docs = Vec::new();
        docs.try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "block-max WAND candidates",
                count: UNION_HORIZON,
            })?;
        let mut order = Vec::new();
        order
            .try_reserve_exact(cursors.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "block-max WAND pivot order",
                count: cursors.len(),
            })?;
        let horizon_last = u32::try_from(horizon_end.saturating_sub(1).min(u64::from(u32::MAX)))
            .map_err(|_| ArgusError::CursorInvariant("union horizon last doc exceeds u32"))?;
        let mut blocks_skipped = 0_u64;

        loop {
            order.clear();
            order.extend(cursors.iter().enumerate().filter_map(|(index, cursor)| {
                cursor
                    .doc()
                    .filter(|doc| u64::from(*doc) < horizon_end)
                    .map(|_| index)
            }));
            if order.is_empty() {
                break;
            }
            order.sort_unstable_by(|left, right| {
                cursors[*left]
                    .doc()
                    .cmp(&cursors[*right].doc())
                    .then_with(|| left.cmp(right))
            });

            let interval_end = order
                .iter()
                .filter_map(|index| cursors[*index].current_block_last_doc())
                .min()
                .ok_or(ArgusError::CursorInvariant(
                    "block-max cursor has no current block end",
                ))?
                .min(horizon_last);
            let relevant_count = order
                .iter()
                .take_while(|index| {
                    cursors[**index]
                        .doc()
                        .is_some_and(|doc| doc <= interval_end)
                })
                .count();
            let relevant = &order[..relevant_count];
            let block_bound = conservative_optional_bound_sum(
                relevant
                    .iter()
                    .map(|index| cursors[*index].current_block_upper_bound()),
            )
            .ok_or(ArgusError::CursorInvariant(
                "block-max bound sum was not finite and conservative",
            ))?;
            if block_bound < cutoff {
                for &index in relevant {
                    if interval_end == u32::MAX {
                        while cursors[index].next()?.is_some() {}
                    } else {
                        cursors[index].advance(interval_end + 1)?;
                    }
                    blocks_skipped = blocks_skipped.saturating_add(1);
                }
                continue;
            }

            let mut pivot = None;
            for pivot_index in 0..order.len() {
                let bound = conservative_bound_sum(
                    order[..=pivot_index]
                        .iter()
                        .map(|index| cursors[*index].term_upper_bound()),
                )
                .ok_or(ArgusError::CursorInvariant(
                    "WAND pivot bound sum was not finite and conservative",
                ))?;
                if bound >= cutoff {
                    pivot = Some(pivot_index);
                    break;
                }
            }
            let Some(pivot_index) = pivot else {
                break;
            };
            let pivot_doc = cursors[order[pivot_index]]
                .doc()
                .ok_or(ArgusError::CursorInvariant("WAND pivot is exhausted"))?;
            let minimum_doc = cursors[order[0]]
                .doc()
                .ok_or(ArgusError::CursorInvariant("WAND minimum is exhausted"))?;
            if minimum_doc < pivot_doc {
                for &index in &order[..pivot_index] {
                    if cursors[index].doc().is_some_and(|doc| doc < pivot_doc) {
                        cursors[index].advance(pivot_doc)?;
                    }
                }
                continue;
            }

            docs.push(pivot_doc);
            for &index in &order {
                if cursors[index].doc() == Some(pivot_doc) {
                    cursors[index].next()?;
                } else {
                    break;
                }
            }
        }

        Ok(CompetitiveCandidates {
            strategy: UnionPruningStrategy::BlockMaxWand,
            docs,
            blocks_skipped,
        })
    }

    fn fill_candidate_window(
        &mut self,
        window_start: u32,
        horizon_end: u64,
        candidates: &[u32],
    ) -> Result<(), ArgusError> {
        // Candidate discovery advances only shadow cursors. Real cursors still
        // visit selected documents in their current scorer-vector order, so
        // pruning cannot regroup ordinary exhaustive contributions.
        let mut index = 0;
        while index < self.active.len() {
            loop {
                let Some(doc) = self.active[index].doc() else {
                    self.active.swap_remove(index);
                    break;
                };
                if u64::from(doc) >= horizon_end {
                    index += 1;
                    break;
                }
                let candidate_index = candidates.partition_point(|candidate| *candidate < doc);
                let Some(&candidate) = candidates.get(candidate_index) else {
                    Self::advance_scorer_to_horizon(&mut self.active[index], horizon_end)?;
                    if self.active[index].doc().is_none() {
                        self.active.swap_remove(index);
                    } else {
                        index += 1;
                    }
                    break;
                };
                if doc < candidate {
                    self.active[index].advance(candidate)?;
                    continue;
                }
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let contribution = self.active[index].score()?;
                let total = self.score_window[offset].unwrap_or(0.0) + contribution;
                self.score_window[offset] = Some(finite_score(total, doc)?);
                self.active[index].next()?;
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                    break;
                }
            }
        }
        Ok(())
    }

    fn advance_scorer_to_horizon(
        scorer: &mut ReferenceScorer<'_>,
        horizon_end: u64,
    ) -> Result<(), ArgusError> {
        if let Ok(target) = u32::try_from(horizon_end) {
            scorer.advance(target)?;
            return Ok(());
        }
        while scorer.next()?.is_some() {}
        Ok(())
    }

    fn advance_buffered(&mut self) -> Option<u32> {
        let start = self.window_start?;
        while self.scan_offset < self.score_window.len() {
            let offset = self.scan_offset;
            self.scan_offset += 1;
            if let Some(score) = self.score_window[offset].take() {
                let doc = u64::from(start) + offset as u64;
                let doc = u32::try_from(doc).ok()?;
                self.current = Some(doc);
                self.current_score = score;
                return self.current;
            }
        }
        None
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        if self.advance_buffered().is_some() {
            return Ok(self.current);
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    fn collect_top_docs<L>(
        &mut self,
        collector: &mut TopDocsCollector,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        while let Some(doc) = self.current {
            let score = self.score()?;
            if live_docs.is_live(doc) {
                collector.record_live(doc, Some(score))?;
            }
            if self.advance_buffered().is_some() {
                continue;
            }
            loop {
                let cutoff = collector.competitive_cutoff_score();
                if !self.refill_with_cutoff(cutoff)? {
                    return Ok(());
                }
                if self.advance_buffered().is_some() {
                    break;
                }
            }
        }
        Ok(())
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON_U64
        }) {
            while self.current.is_some_and(|doc| doc < target) {
                self.next()?;
            }
            return Ok(self.current);
        }

        self.clear_buffer();
        let mut index = 0;
        while index < self.active.len() {
            if self.active[index].doc().is_some_and(|doc| doc < target) {
                self.active[index].advance(target)?;
            }
            if self.active[index].doc().is_none() {
                self.active.swap_remove(index);
            } else {
                index += 1;
            }
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        if self.current.is_none() {
            return Ok(SeekDangerResult::SeekLowerBound(None));
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON_U64
                && target >= start
        }) {
            return classify_seek_result(target, self.seek(target)?);
        }

        let mut minimum_lower_bound = None;
        let mut found = false;
        for scorer in &mut self.active {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {
                    found = true;
                    break;
                }
                SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                    minimum_lower_bound = Some(
                        minimum_lower_bound
                            .map_or(lower_bound, |minimum: u32| minimum.min(lower_bound)),
                    );
                }
                SeekDangerResult::SeekLowerBound(None) => {}
            }
        }
        if found {
            let result = self.seek(target)?;
            if result != Some(target) {
                return Err(ArgusError::CursorInvariant(
                    "union danger seek found a child but not the union target",
                ));
            }
            return Ok(SeekDangerResult::Found);
        }
        Ok(SeekDangerResult::SeekLowerBound(minimum_lower_bound))
    }
    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self
            .current
            .ok_or(ArgusError::CursorInvariant("cannot score exhausted union"))?;
        finite_score(self.current_score, doc)
    }
}

struct RequiredOptionalScorer<'a> {
    required: Box<ReferenceScorer<'a>>,
    optional: Box<ReferenceScorer<'a>>,
    score_cache: Option<(u32, f32)>,
    segment_num_docs: u32,
}

impl<'a> RequiredOptionalScorer<'a> {
    fn new(
        required: ReferenceScorer<'a>,
        optional: ReferenceScorer<'a>,
    ) -> Result<Self, ArgusError> {
        if required.segment_num_docs() != optional.segment_num_docs() {
            return Err(ArgusError::CursorInvariant(
                "required and optional scorers belong to different segment domains",
            ));
        }
        let segment_num_docs = required.segment_num_docs();
        Ok(Self {
            required: Box::new(required),
            optional: Box::new(optional),
            score_cache: None,
            segment_num_docs,
        })
    }

    fn doc(&self) -> Option<u32> {
        self.required.doc()
    }

    fn cost(&self) -> u64 {
        self.required.cost()
    }

    fn size_hint(&self) -> u32 {
        self.required.size_hint()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.score_cache = None;
        self.required.next()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        self.score_cache = None;
        self.required.advance(target)
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        self.score_cache = None;
        self.required.seek_danger(target)
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted required-optional scorer",
        ))?;
        if let Some((cached_doc, score)) = self.score_cache
            && cached_doc == doc
        {
            return Ok(score);
        }
        let mut score = 0.0_f32;
        score += self.required.score()?;
        if self
            .optional
            .doc()
            .is_some_and(|optional_doc| optional_doc <= doc)
            && self.optional.advance(doc)? == Some(doc)
        {
            score += self.optional.score()?;
        }
        let score = finite_score(score, doc)?;
        self.score_cache = Some((doc, score));
        Ok(score)
    }
}

struct ExcludeScorer<'a> {
    include: Box<ReferenceScorer<'a>>,
    excluded: Vec<ReferenceScorer<'a>>,
    segment_num_docs: u32,
}

impl<'a> ExcludeScorer<'a> {
    fn new(
        include: ReferenceScorer<'a>,
        excluded: Vec<ReferenceScorer<'a>>,
    ) -> Result<Self, ArgusError> {
        let segment_num_docs = if include.is_explicit_empty() {
            shared_segment_num_docs(&excluded)?
        } else {
            include.segment_num_docs()
        };
        if !include.is_explicit_empty()
            && excluded
                .iter()
                .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
        {
            return Err(ArgusError::CursorInvariant(
                "include and exclusion scorers belong to different segment domains",
            ));
        }
        let mut scorer = Self {
            include: Box::new(include),
            excluded,
            segment_num_docs,
        };
        scorer.skip_excluded()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.include.doc()
    }

    fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    fn size_hint(&self) -> u32 {
        self.include.size_hint()
    }

    fn contains(&mut self, doc: u32) -> Result<bool, ArgusError> {
        for scorer in &mut self.excluded {
            if scorer.seek_danger(doc)? == SeekDangerResult::Found {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn skip_excluded(&mut self) -> Result<Option<u32>, ArgusError> {
        while let Some(doc) = self.include.doc() {
            if !self.contains(doc)? {
                return Ok(Some(doc));
            }
            self.include.next()?;
        }
        Ok(None)
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_none() {
            return Ok(None);
        }
        self.include.next()?;
        self.skip_excluded()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.include.doc());
        }
        self.include.advance(target)?;
        self.skip_excluded()
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        self.include.score()
    }
}

fn finite_score(score: f32, global_docid: u32) -> Result<f32, ArgusError> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(ArgusError::NonFiniteScore { global_docid })
    }
}

/// Live-document membership used only at result collection.
pub trait LiveDocs {
    /// Return whether a physical posting is visible in the current snapshot.
    fn is_live(&self, global_docid: u32) -> bool;
}

impl<F> LiveDocs for F
where
    F: Fn(u32) -> bool,
{
    fn is_live(&self, global_docid: u32) -> bool {
        self(global_docid)
    }
}

/// Membership policy for snapshots without tombstones.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllLiveDocs;

impl LiveDocs for AllLiveDocs {
    fn is_live(&self, _global_docid: u32) -> bool {
        true
    }
}

/// Phase-one exhaustive result carrying no stored metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredDoc {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Final globally merged top-doc page.
///
/// `total_count` is present only when the collector was created with
/// [`TopDocsCollector::with_exact_count`]. Hits are already offset-sliced and
/// remain in the canonical `(score desc, global_docid asc)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectedTopDocs {
    /// Page-local ranked hits.
    pub hits: Vec<ScoredDoc>,
    /// Exact number of live matches across every collected scorer.
    pub total_count: Option<u64>,
}

/// Cross-segment exhaustive top-doc accumulator.
///
/// Feed every per-segment scorer into one instance, then call [`Self::finish`]
/// exactly once. Offset slicing happens only after the global heap is sorted,
/// preventing per-segment pagination from discarding valid winners.
#[derive(Debug)]
pub struct TopDocsCollector {
    limit: usize,
    offset: usize,
    retained: usize,
    exact_count: bool,
    total_count: u64,
    /// Bounded max-heap of packed selection keys (`bd-y1ab`). Each key folds
    /// the entry's score and docid so the whole heap comparison is one native
    /// `u64` compare instead of an f32 `total_cmp` plus a conditional docid
    /// tiebreak: high 32 bits = `!sortable(score)` (worst score = greatest key
    /// = heap top, evicted first), low 32 bits = the global docid (higher docid
    /// = greater = evicted first on a score tie, so the lower docid is
    /// retained). Order-identical to the former per-entry `total_cmp` heap; see
    /// [`packed_selection_key`].
    heap: BinaryHeap<u64>,
}

/// Monotonic f32 -> u32 bijection: ascending `u32` order equals ascending f32
/// order for every finite value (and both signed zeros). Collector scores are
/// [`finite_score`]-guaranteed finite, so the round-trip through
/// [`score_from_sortable`] is exact.
#[inline]
fn score_to_sortable(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse of [`score_to_sortable`].
#[inline]
fn score_from_sortable(sortable: u32) -> f32 {
    if sortable & 0x8000_0000 != 0 {
        f32::from_bits(sortable & 0x7fff_ffff)
    } else {
        f32::from_bits(!sortable)
    }
}

/// Badness-ordered packed key: greatest for the worst entry (lowest score;
/// highest docid on a score tie), so a max-heap retains the best entries.
#[inline]
fn packed_selection_key(global_docid: u32, score: f32) -> u64 {
    (u64::from(!score_to_sortable(score)) << 32) | u64::from(global_docid)
}

/// Reconstruct the winner encoded by a packed selection key.
#[inline]
fn scored_doc_from_key(key: u64) -> ScoredDoc {
    let global_docid = (key & 0xffff_ffff) as u32;
    let score = score_from_sortable(!((key >> 32) as u32));
    ScoredDoc {
        global_docid,
        score,
    }
}

/// Score of the packed key currently at the heap top (the worst retained
/// entry), used as the competitive pruning cutoff.
#[inline]
fn cutoff_score_from_key(key: u64) -> f32 {
    score_from_sortable(!((key >> 32) as u32))
}

impl TopDocsCollector {
    /// Build a count-free top-k collector with Tantivy-compatible offset.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn new(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, false)
    }

    /// Build a top-k collector that also exhaustively counts every live match.
    ///
    /// `limit == 0` retains no scores but still counts all matches.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn with_exact_count(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, true)
    }

    fn build(limit: usize, offset: usize, exact_count: bool) -> Result<Self, ArgusError> {
        let retained = if limit == 0 {
            0
        } else {
            limit
                .checked_add(offset)
                .ok_or(ArgusError::CollectorWindowOverflow { limit, offset })?
        };
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(retained)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k heap",
                count: retained,
            })?;
        Ok(Self {
            limit,
            offset,
            retained,
            exact_count,
            total_count: 0,
            heap,
        })
    }

    /// Build an empty collector sharing this collector's limit, offset, and
    /// count-mode shape.
    ///
    /// Segment-parallel fan-out uses this to give each rayon task a private
    /// accumulator whose retained window matches the fold target exactly.
    ///
    /// # Errors
    ///
    /// Rejects bounded heap allocation failure.
    pub fn empty_like(&self) -> Result<Self, ArgusError> {
        Self::build(self.limit, self.offset, self.exact_count)
    }

    /// Fold another collector's retained state into this one.
    ///
    /// The packed selection key's total order (score via the monotonic
    /// [`score_to_sortable`] transform, then ascending docid) makes the
    /// retained top-`retained` set unique regardless of insertion order, so
    /// folding per-segment partials in any order yields exactly the set a
    /// single serial collector retains.
    ///
    /// # Errors
    ///
    /// Rejects collectors with mismatched limit/offset/count shape and exact
    /// match-count overflow.
    pub fn merge(&mut self, other: Self) -> Result<(), ArgusError> {
        if self.limit != other.limit
            || self.offset != other.offset
            || self.exact_count != other.exact_count
        {
            return Err(ArgusError::CursorInvariant(
                "merged top-k collectors must share limit, offset, and count mode",
            ));
        }
        if self.exact_count {
            self.total_count = self
                .total_count
                .checked_add(other.total_count)
                .ok_or(ArgusError::MatchCountOverflow)?;
        }
        for key in other.heap {
            if self.heap.len() < self.retained {
                self.heap.push(key);
            } else if self.heap.peek().is_some_and(|&cutoff| key < cutoff) {
                let _ = self.heap.pop();
                self.heap.push(key);
            }
        }
        Ok(())
    }

    /// Consume one segment scorer into this global accumulator.
    ///
    /// # Errors
    ///
    /// Propagates scorer, visibility, count, and allocation failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if self.retained == 0 && !self.exact_count {
            return Ok(());
        }
        let mut state = CollectorState::TopDocs(self);
        scorer.collect_with(&mut state, live_docs)
    }

    fn record_live(&mut self, global_docid: u32, score: Option<f32>) -> Result<(), ArgusError> {
        if self.exact_count {
            self.total_count = self
                .total_count
                .checked_add(1)
                .ok_or(ArgusError::MatchCountOverflow)?;
        }
        if self.retained == 0 {
            return Ok(());
        }
        let score = score.ok_or(ArgusError::CursorInvariant(
            "top-k collector did not score a retained match",
        ))?;
        let key = packed_selection_key(global_docid, score);
        if self.heap.len() < self.retained {
            self.heap.push(key);
        } else if self.heap.peek().is_some_and(|&cutoff| key < cutoff) {
            let _ = self.heap.pop();
            self.heap.push(key);
        }
        Ok(())
    }

    fn competitive_cutoff_score(&self) -> Option<f32> {
        if self.retained != 0 && self.heap.len() == self.retained {
            self.heap.peek().map(|&key| cutoff_score_from_key(key))
        } else {
            None
        }
    }

    /// Sort the global heap, apply offset once, and return page winners.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation failure when the final compact hit vector
    /// cannot be reserved.
    pub fn finish(self) -> Result<CollectedTopDocs, ArgusError> {
        let winner_count = self.heap.len();
        let mut hits = Vec::new();
        hits.try_reserve_exact(winner_count)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k winners",
                count: winner_count,
            })?;
        hits.extend(self.heap.into_iter().map(scored_doc_from_key));
        hits.sort_unstable_by(|left, right| compare_scored_best_first(*left, *right));
        let start = self.offset.min(hits.len());
        let end = start.saturating_add(self.limit).min(hits.len());
        if start != 0 {
            hits.copy_within(start..end, 0);
        }
        hits.truncate(end - start);
        Ok(CollectedTopDocs {
            hits,
            total_count: self.exact_count.then_some(self.total_count),
        })
    }
}

/// Cross-segment scoreless global-document set accumulator.
#[derive(Debug, Default)]
pub struct DocSetCollector {
    docs: Vec<u32>,
}

impl DocSetCollector {
    /// Build an empty scoreless collector.
    #[must_use]
    pub const fn new() -> Self {
        Self { docs: Vec::new() }
    }

    /// Consume one recursively unscored segment scorer.
    ///
    /// # Errors
    ///
    /// Rejects scored buffered unions and propagates cursor or allocation
    /// failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut state = CollectorState::DocSet(self);
        scorer.collect_with(&mut state, live_docs)
    }

    fn push(&mut self, global_docid: u32) -> Result<(), ArgusError> {
        if self.docs.len() == self.docs.capacity() {
            self.docs
                .try_reserve(1)
                .map_err(|_| ArgusError::Allocation {
                    resource: "doc-set matches",
                    count: self.docs.len().saturating_add(1),
                })?;
        }
        self.docs.push(global_docid);
        Ok(())
    }

    /// Fold another collector's raw matches into this one.
    ///
    /// Order does not matter: [`Self::finish`] sorts and dedups, so folding
    /// per-segment partials in any order yields the identical final set.
    ///
    /// # Errors
    ///
    /// Rejects allocation failure while growing the match vector.
    pub fn merge(&mut self, other: Self) -> Result<(), ArgusError> {
        self.docs
            .try_reserve(other.docs.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "doc-set matches",
                count: self.docs.len().saturating_add(other.docs.len()),
            })?;
        self.docs.extend(other.docs);
        Ok(())
    }

    /// Return deterministic sorted unique global docids.
    #[must_use]
    pub fn finish(mut self) -> Vec<u32> {
        self.docs.sort_unstable();
        self.docs.dedup();
        self.docs
    }
}

enum CollectorState<'a> {
    TopDocs(&'a mut TopDocsCollector),
    DocSet(&'a mut DocSetCollector),
}

/// Winner after phase-two external-id materialization.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedHit {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// External document id resolved only after top-k selection.
    pub document_id: DocId,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Resolve external document ids for already-selected winners only.
///
/// # Errors
///
/// Propagates the resolver's typed [`ArgusError`] or a bounded output
/// allocation failure.
pub fn materialize_doc_ids(
    winners: &[ScoredDoc],
    mut resolve: impl FnMut(u32) -> Result<DocId, ArgusError>,
) -> Result<Vec<MaterializedHit>, ArgusError> {
    let mut hits = Vec::new();
    hits.try_reserve_exact(winners.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "materialized hits",
            count: winners.len(),
        })?;
    for winner in winners {
        hits.push(MaterializedHit {
            global_docid: winner.global_docid,
            document_id: resolve(winner.global_docid)?,
            score: winner.score,
        });
    }
    Ok(hits)
}

fn compare_scored_best_first(left: ScoredDoc, right: ScoredDoc) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.global_docid.cmp(&right.global_docid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{BM25_B, fieldnorm_to_id, id_to_fieldnorm};
    use crate::quiver::{
        DocLenFieldInput, EncodedBlockMax, EncodedDocLenSection, EncodedNumericSection,
        EncodedPositionList, EncodedPostingList, NumericEntry, NumericFieldInput, Posting,
    };
    use crate::schema::{FieldDescriptor, FieldKind, SchemaDescriptor};

    const RANGE_TEST_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "created_at",
        kind: FieldKind::I64 {
            indexed: true,
            fast: true,
        },
        stored: false,
    }];
    const RANGE_TEST_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "argus-range-tests",
        fields: &RANGE_TEST_FIELDS,
    };

    #[derive(Clone, Debug)]
    struct VecCursor {
        postings: Vec<Posting>,
        index: usize,
        cost: u64,
        segment_num_docs: u32,
    }

    impl VecCursor {
        const fn new(postings: Vec<Posting>, cost: u64, segment_num_docs: u32) -> Self {
            Self {
                postings,
                index: 0,
                cost,
                segment_num_docs,
            }
        }

        fn current(&self) -> Option<Posting> {
            self.postings.get(self.index).copied()
        }
    }

    impl PostingCursor for VecCursor {
        fn doc(&self) -> Option<u32> {
            self.current().map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current().map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            None
        }

        fn size_hint(&self) -> u32 {
            u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
        }

        fn cost(&self) -> u64 {
            self.cost
        }

        fn segment_num_docs(&self) -> u32 {
            self.segment_num_docs
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            if self.index < self.postings.len() {
                self.index += 1;
            }
            Ok(self.doc())
        }

        fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
            let tail = self.postings.get(self.index..).unwrap_or_default();
            self.index += tail.partition_point(|posting| posting.doc_id < target);
            Ok(self.doc())
        }
    }

    #[derive(Clone, Debug)]
    struct TermBoundOnlyCursor(VecCursor);

    impl PostingCursor for TermBoundOnlyCursor {
        fn doc(&self) -> Option<u32> {
            self.0.doc()
        }

        fn freq(&self) -> Option<u32> {
            self.0.freq()
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            None
        }

        fn size_hint(&self) -> u32 {
            self.0.size_hint()
        }

        fn cost(&self) -> u64 {
            self.0.cost()
        }

        fn segment_num_docs(&self) -> u32 {
            self.0.segment_num_docs()
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            self.0.next()
        }

        fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
            self.0.advance(target)
        }

        fn fork_for_pruning(&self) -> Option<Box<dyn PostingCursor + '_>> {
            Some(Box::new(self.clone()))
        }

        fn term_score_upper_bound(
            &self,
            _live_avgdl: f32,
            weight: f32,
            _record_option: TermRecordOption,
        ) -> Option<f32> {
            (weight.is_finite() && !weight.is_sign_negative()).then_some(weight)
        }
    }

    #[derive(Clone, Debug)]
    struct PositionedVecCursor {
        postings: Vec<Posting>,
        positions: Vec<Vec<u32>>,
        index: usize,
        cost: u64,
        segment_num_docs: u32,
    }

    impl PositionedVecCursor {
        fn new(rows: Vec<(u32, Vec<u32>)>, cost: u64, segment_num_docs: u32) -> Self {
            let mut postings = Vec::with_capacity(rows.len());
            let mut positions = Vec::with_capacity(rows.len());
            for (doc, row_positions) in rows {
                let frequency = u32::try_from(row_positions.len()).expect("fixture frequency fits");
                postings.push(Posting::new(doc, frequency));
                positions.push(row_positions);
            }
            Self {
                postings,
                positions,
                index: 0,
                cost,
                segment_num_docs,
            }
        }

        fn current(&self) -> Option<Posting> {
            self.postings.get(self.index).copied()
        }
    }

    impl PositionsReader for PositionedVecCursor {
        fn decode_positions(
            &self,
            posting_ordinal: u32,
            output: &mut Vec<u32>,
        ) -> Result<(), ArgusError> {
            let ordinal = usize::try_from(posting_ordinal).map_err(|_| {
                ArgusError::CursorInvariant("test position ordinal does not fit usize")
            })?;
            if ordinal != self.index {
                return Err(ArgusError::CursorInvariant(
                    "test position handle is no longer current",
                ));
            }
            let positions = self
                .positions
                .get(ordinal)
                .ok_or(ArgusError::CursorInvariant(
                    "test position ordinal is out of range",
                ))?;
            output
                .try_reserve_exact(positions.len())
                .map_err(|_| ArgusError::Allocation {
                    resource: "test decoded positions",
                    count: positions.len(),
                })?;
            output.extend_from_slice(positions);
            Ok(())
        }
    }

    impl PostingCursor for PositionedVecCursor {
        fn doc(&self) -> Option<u32> {
            self.current().map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current().map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            self.current().and_then(|_| {
                u32::try_from(self.index)
                    .ok()
                    .map(|ordinal| PositionsHandle::new(self, ordinal))
            })
        }

        fn size_hint(&self) -> u32 {
            u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
        }

        fn cost(&self) -> u64 {
            self.cost
        }

        fn segment_num_docs(&self) -> u32 {
            self.segment_num_docs
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            if self.index < self.postings.len() {
                self.index += 1;
            }
            Ok(self.doc())
        }

        fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
            let tail = self.postings.get(self.index..).unwrap_or_default();
            self.index += tail.partition_point(|posting| posting.doc_id < target);
            Ok(self.doc())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum CursorFault {
        StickyNext,
        BackwardNext,
        BelowTargetAdvance,
        MovesDespiteSatisfiedTarget,
        ResurrectsAfterExhaustion,
    }

    #[derive(Clone, Debug)]
    struct FaultCursor {
        current: Option<Posting>,
        fault: CursorFault,
        moves: u8,
    }

    impl FaultCursor {
        const fn new(fault: CursorFault) -> Self {
            let doc_id = match fault {
                CursorFault::BelowTargetAdvance | CursorFault::ResurrectsAfterExhaustion => 1,
                CursorFault::MovesDespiteSatisfiedTarget => 5,
                CursorFault::StickyNext | CursorFault::BackwardNext => 3,
            };
            Self {
                current: Some(Posting::new(doc_id, 1)),
                fault,
                moves: 0,
            }
        }
    }

    impl PostingCursor for FaultCursor {
        fn doc(&self) -> Option<u32> {
            self.current.map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current.map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            None
        }

        fn size_hint(&self) -> u32 {
            if matches!(self.fault, CursorFault::ResurrectsAfterExhaustion) {
                2
            } else {
                1
            }
        }

        fn segment_num_docs(&self) -> u32 {
            8
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            match self.fault {
                CursorFault::StickyNext => {}
                CursorFault::BackwardNext => self.current = Some(Posting::new(2, 1)),
                CursorFault::ResurrectsAfterExhaustion if self.moves == 0 => {
                    self.current = None;
                }
                CursorFault::ResurrectsAfterExhaustion => {
                    self.current = Some(Posting::new(2, 1));
                }
                CursorFault::BelowTargetAdvance | CursorFault::MovesDespiteSatisfiedTarget => {
                    self.current = None;
                }
            }
            self.moves = self.moves.saturating_add(1);
            Ok(self.doc())
        }

        fn advance(&mut self, _target: u32) -> Result<Option<u32>, ArgusError> {
            match self.fault {
                CursorFault::BelowTargetAdvance => self.current = Some(Posting::new(3, 1)),
                CursorFault::MovesDespiteSatisfiedTarget => {
                    self.current = Some(Posting::new(6, 1));
                }
                CursorFault::ResurrectsAfterExhaustion if self.current.is_none() => {
                    self.current = Some(Posting::new(2, 1));
                }
                CursorFault::StickyNext
                | CursorFault::BackwardNext
                | CursorFault::ResurrectsAfterExhaustion => {}
            }
            self.moves = self.moves.saturating_add(1);
            Ok(self.doc())
        }
    }

    fn snapshot(
        field_ord: u16,
        total_tokens: u64,
        doc_count: u64,
    ) -> Result<Bm25FieldSnapshot, ArgusError> {
        Bm25FieldSnapshot::new(SnapshotFieldStats {
            field_ord,
            total_tokens,
            doc_count,
        })
    }

    fn term<'a>(
        postings: Vec<Posting>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        cost: u64,
        boost: f32,
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        let segment_num_docs = u32::try_from(snapshot.doc_count()).unwrap_or(u32::MAX);
        Ok(ReferenceScorer::term(TermScorer::new(
            VecCursor::new(postings, cost, segment_num_docs),
            fieldnorms,
            snapshot.clone(),
            snapshot_doc_freq,
            TermRecordOption::WithFreqs,
            boost,
        )?))
    }

    fn sealed_union<'a>(
        posting_lists: &'a [crate::quiver::PostingList<'_>],
        block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        rows_by_term: &[Vec<Posting>],
        boosts: &[f32],
        segment_num_docs: u32,
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        sealed_union_with_record_option(
            posting_lists,
            block_max,
            fieldnorms,
            snapshot,
            rows_by_term,
            boosts,
            segment_num_docs,
            TermRecordOption::WithFreqs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sealed_union_with_record_option<'a>(
        posting_lists: &'a [crate::quiver::PostingList<'_>],
        block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        rows_by_term: &[Vec<Posting>],
        boosts: &[f32],
        segment_num_docs: u32,
        record_option: TermRecordOption,
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        if posting_lists.len() != rows_by_term.len() || posting_lists.len() != boosts.len() {
            return Err(ArgusError::CursorInvariant(
                "sealed union fixture cardinalities disagree",
            ));
        }
        if block_max.is_some_and(|bounds| bounds.len() != posting_lists.len()) {
            return Err(ArgusError::CursorInvariant(
                "sealed union BLOCKMAX cardinality disagrees",
            ));
        }
        let mut clauses = Vec::new();
        clauses
            .try_reserve_exact(posting_lists.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "sealed union fixture clauses",
                count: posting_lists.len(),
            })?;
        for (index, postings) in posting_lists.iter().enumerate() {
            let cursor = if let Some(bounds) = block_max {
                SealedPostingCursor::with_block_max(
                    postings.cursor()?,
                    Arc::clone(&bounds[index]),
                    postings.doc_freq(),
                    segment_num_docs,
                )
            } else {
                SealedPostingCursor::new(postings, segment_num_docs)?
            };
            let scorer = TermScorer::new(
                cursor,
                fieldnorms,
                snapshot.clone(),
                u64::try_from(rows_by_term[index].len()).map_err(|_| {
                    ArgusError::CursorInvariant("fixture document frequency exceeds u64")
                })?,
                record_option,
                boosts[index],
            )?;
            clauses.push(ScorerClause::should(ReferenceScorer::term(scorer)));
        }
        ReferenceScorer::boolean(clauses)
    }

    /// Build a nested pure-term union: `group_sizes` slices the flat term
    /// arrays into per-group multi-field unions (each a `sealed_union`), wrapped
    /// in an outer `Should` union. With `block_max` present each group's terms
    /// carry ceilings, so the outer union is grouped-`MaxScore` eligible.
    #[allow(clippy::too_many_arguments)]
    fn sealed_grouped_union<'a>(
        posting_lists: &'a [crate::quiver::PostingList<'_>],
        block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        rows_by_term: &[Vec<Posting>],
        boosts: &[f32],
        segment_num_docs: u32,
        group_sizes: &[usize],
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        sealed_grouped_union_with_record_option(
            posting_lists,
            block_max,
            fieldnorms,
            snapshot,
            rows_by_term,
            boosts,
            segment_num_docs,
            group_sizes,
            TermRecordOption::WithFreqs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sealed_grouped_union_with_record_option<'a>(
        posting_lists: &'a [crate::quiver::PostingList<'_>],
        block_max: Option<&'a [Arc<[BlockMaxEntry]>]>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        rows_by_term: &[Vec<Posting>],
        boosts: &[f32],
        segment_num_docs: u32,
        group_sizes: &[usize],
        record_option: TermRecordOption,
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        let mut outer = Vec::new();
        let mut offset = 0_usize;
        for &size in group_sizes {
            let end = offset + size;
            let group = sealed_union_with_record_option(
                &posting_lists[offset..end],
                block_max.map(|bounds| &bounds[offset..end]),
                fieldnorms,
                snapshot,
                &rows_by_term[offset..end],
                &boosts[offset..end],
                segment_num_docs,
                record_option,
            )?;
            outer.push(ScorerClause::should(group));
            offset = end;
        }
        ReferenceScorer::boolean(outer)
    }

    /// The drained-group pin (`bd-bt2t`). An ESSENTIAL group whose postings all
    /// fall inside one `UNION_HORIZON` window drains to an empty `active` and
    /// carries its remaining documents as buffered residue instead. Candidate
    /// discovery must enumerate that residue; enumerating only `active` reports
    /// the group as having no documents in the window and the root silently
    /// drops them from the top-k.
    ///
    /// This case CANNOT be pinned by `grouped_max_score_matches_exhaustive_and_prunes`:
    /// that fixture runs at k=1 where the dropped documents tie the incumbent
    /// at exactly `2e8` (the ~1.0 contribution is below the f32 ULP of ~16 at
    /// that magnitude) and lose the ascending-docid tie-break anyway — so
    /// deleting them changes nothing observable, and a broken fix passes. Here
    /// the drained group's documents must WIN the top-k outright, so losing
    /// them changes both the ranking and the hit count.
    #[test]
    fn grouped_max_score_drained_essential_group_keeps_its_documents()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 8_192;
        let lengths = vec![Some(1); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
        // group0 = terms {A, B, C}, group1 = terms {D, E}.
        //   A holds the early docs 0/1/2 and fills the k=3 heap in window 0,
        //     establishing a cutoff that only group0 can beat.
        //   B and C hold 5000/5001/5002. When the root advances group0 past the
        //     first horizon, group0 refills onto [5000, 9096) where B and C both
        //     exhaust inside the window: active EMPTY, current = 5000, residue
        //     = {5001, 5002}. That is the state under test.
        //   D and E (docs 6000/6001, unit boost) keep group1 non-essential, so
        //     the window genuinely prunes rather than degenerating to exhaustive.
        let rows_by_term = vec![
            vec![Posting::new(0, 1), Posting::new(1, 1), Posting::new(2, 1)],
            vec![
                Posting::new(5_000, 1),
                Posting::new(5_001, 1),
                Posting::new(5_002, 1),
            ],
            vec![
                Posting::new(5_000, 1),
                Posting::new(5_001, 1),
                Posting::new(5_002, 1),
            ],
            vec![Posting::new(6_000, 1)],
            vec![Posting::new(6_001, 1)],
        ];
        let boosts = [1.0e8, 1.0e8, 1.0e8, 1.0, 1.0];
        let group_sizes = [3_usize, 2_usize];
        let fieldnorm_id = fieldnorm_to_id(1);
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

        let mut oracle = sealed_grouped_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
            &group_sizes,
        )?;
        let mut oracle_collector = TopDocsCollector::new(3, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;

        let mut candidate = sealed_grouped_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
            &group_sizes,
        )?;
        let mut candidate_collector = TopDocsCollector::new(3, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;

        // The drained group's documents outscore the heap incumbents 2:1, so
        // losing the residue would change the ranking AND shrink the hit count.
        assert_eq!(
            oracle_hits
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            vec![5_000, 5_001, 5_002],
            "exhaustive top-3 must be the drained group's documents",
        );
        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert!(
            stats.max_score_windows >= 1,
            "grouped MaxScore should prune at least one window (got {stats:?})"
        );
        Ok(())
    }

    #[test]
    fn grouped_max_score_matches_exhaustive_and_prunes() -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 8_192;
        let lengths = vec![Some(32); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS) * 32, u64::from(NUM_DOCS))?;
        // Two multi-field groups of two terms. group0 (high ceiling) is active
        // at doc 0 (window 0) and doc 5_000 (window 1), so it stays *essential*;
        // group1 (low ceiling) holds doc 5_000 and — uniquely — doc 6_000. With
        // k=1 the heap fills on doc 0 (score ~2e8), so in window 1 group1 is
        // non-essential and doc 6_000 (present only in group1) is skipped while
        // the true top-1 (doc 0) is preserved bit-for-bit.
        let rows_by_term = vec![
            vec![Posting::new(0, 7), Posting::new(5_000, 7)],
            vec![Posting::new(0, 11), Posting::new(5_000, 11)],
            vec![Posting::new(5_000, 5)],
            vec![Posting::new(6_000, 17)],
        ];
        let boosts = [1.0e8, 1.0e8, 1.0, 1.0];
        let group_sizes = [2_usize, 2_usize];
        let fieldnorm_id = fieldnorm_to_id(32);
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

        for record_option in [TermRecordOption::WithFreqs, TermRecordOption::Basic] {
            let mut oracle = sealed_grouped_union_with_record_option(
                &posting_lists,
                None,
                field,
                &snapshot,
                &rows_by_term,
                &boosts,
                NUM_DOCS,
                &group_sizes,
                record_option,
            )?;
            let mut oracle_collector = TopDocsCollector::new(1, 0)?;
            oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
            let oracle_hits = oracle_collector.finish()?.hits;

            let mut candidate = sealed_grouped_union_with_record_option(
                &posting_lists,
                Some(&block_max),
                field,
                &snapshot,
                &rows_by_term,
                &boosts,
                NUM_DOCS,
                &group_sizes,
                record_option,
            )?;
            let mut candidate_collector = TopDocsCollector::new(1, 0)?;
            candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
            let stats = candidate
                .union_pruning_stats()
                .expect("top-level union retains pruning stats");
            let candidate_hits = candidate_collector.finish()?.hits;

            assert_hits_bit_exact(&candidate_hits, &oracle_hits);
            assert_eq!(candidate_hits[0].global_docid, 0);
            assert!(
                stats.max_score_windows >= 1,
                "{record_option:?} grouped MaxScore should prune (got {stats:?})"
            );
        }
        Ok(())
    }

    fn assert_hits_bit_exact(actual: &[ScoredDoc], expected: &[ScoredDoc]) {
        assert_eq!(actual.len(), expected.len(), "hit count differs");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.global_docid, expected.global_docid,
                "docid differs at rank {index}"
            );
            assert_eq!(
                actual.score.to_bits(),
                expected.score.to_bits(),
                "score bits differ at rank {index} for doc {}",
                actual.global_docid
            );
        }
    }

    #[cfg(feature = "bench-internals")]
    fn timed_encoded_grouped_union(
        encoded_doclens: &EncodedDocLenSection,
        encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
        cached_metadata: Option<&[Arc<ValidatedTermPruningMetadata>]>,
        snapshot: &Bm25FieldSnapshot,
        rows_by_term: &[Vec<Posting>],
        boosts: &[f32],
        segment_num_docs: u32,
        limit: usize,
        group_size: usize,
        rank_pruning: bool,
    ) -> Result<(u128, Vec<ScoredDoc>, UnionPruningStats), Box<dyn std::error::Error>> {
        if group_size == 0
            || encoded_terms.len() != rows_by_term.len()
            || encoded_terms.len() != boosts.len()
            || !encoded_terms.len().is_multiple_of(group_size)
            || cached_metadata.is_some_and(|metadata| metadata.len() != encoded_terms.len())
        {
            return Err(Box::new(ArgusError::CursorInvariant(
                "encoded grouped-union fixture cardinalities disagree",
            )));
        }
        let started = std::time::Instant::now();
        let doclens = encoded_doclens.section(&[1])?;
        let fieldnorms = doclens.field(1).ok_or(ArgusError::CursorInvariant(
            "encoded grouped-union fixture has no field 1",
        ))?;
        let mut outer = Vec::new();
        outer
            .try_reserve_exact(encoded_terms.len() / group_size)
            .map_err(|_| ArgusError::Allocation {
                resource: "encoded grouped-union outer clauses",
                count: encoded_terms.len() / group_size,
            })?;
        let mut group = Vec::new();
        group
            .try_reserve_exact(group_size)
            .map_err(|_| ArgusError::Allocation {
                resource: "encoded grouped-union inner clauses",
                count: group_size,
            })?;
        for (index, (encoded_postings, encoded_block_max)) in encoded_terms.iter().enumerate() {
            let cached = cached_metadata.map(|metadata| Arc::clone(&metadata[index]));
            let cursor = if let Some(metadata) = cached {
                if rank_pruning {
                    SealedPostingCursor::from_validated_pruning(
                        encoded_postings.as_bytes(),
                        metadata,
                        segment_num_docs,
                    )?
                } else {
                    let size_hint = metadata.doc_freq();
                    SealedPostingCursor::from_owned(
                        metadata.cursor(encoded_postings.as_bytes())?,
                        size_hint,
                        segment_num_docs,
                    )
                }
            } else if rank_pruning {
                let postings = encoded_postings.posting_list()?;
                let metadata = Arc::new(
                    postings.into_pruning_metadata(encoded_block_max.as_bytes(), fieldnorms)?,
                );
                SealedPostingCursor::from_validated_pruning(
                    encoded_postings.as_bytes(),
                    metadata,
                    segment_num_docs,
                )?
            } else {
                let postings = encoded_postings.posting_list()?;
                let size_hint = postings.doc_freq();
                SealedPostingCursor::from_owned(
                    postings.into_cursor()?,
                    size_hint,
                    segment_num_docs,
                )
            };
            let term = ReferenceScorer::term(TermScorer::new(
                cursor,
                fieldnorms,
                snapshot.clone(),
                u64::try_from(rows_by_term[index].len()).map_err(|_| {
                    ArgusError::CursorInvariant("fixture document frequency exceeds u64")
                })?,
                TermRecordOption::WithFreqs,
                boosts[index],
            )?);
            group.push(ScorerClause::should(term));
            if group.len() == group_size {
                let grouped = ReferenceScorer::boolean(std::mem::take(&mut group))?;
                outer.push(ScorerClause::should(grouped));
                group
                    .try_reserve_exact(group_size)
                    .map_err(|_| ArgusError::Allocation {
                        resource: "encoded grouped-union inner clauses",
                        count: group_size,
                    })?;
            }
        }
        let mut scorer = ReferenceScorer::boolean(outer)?;
        let mut collector = TopDocsCollector::new(limit, 0)?;
        collector.collect(&mut scorer, &AllLiveDocs)?;
        let stats = scorer
            .union_pruning_stats()
            .ok_or(ArgusError::CursorInvariant(
                "profile scorer is not a top-level union",
            ))?;
        let hits = collector.finish()?.hits;
        let elapsed_us = started.elapsed().as_micros();
        Ok((elapsed_us, hits, stats))
    }

    #[cfg(feature = "bench-internals")]
    fn validate_encoded_pruning_metadata(
        encoded_doclens: &EncodedDocLenSection,
        encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
    ) -> Result<Vec<Arc<ValidatedTermPruningMetadata>>, Box<dyn std::error::Error>> {
        let doclens = encoded_doclens.section(&[1])?;
        let fieldnorms = doclens.field(1).ok_or(ArgusError::CursorInvariant(
            "encoded grouped-union fixture has no field 1",
        ))?;
        let mut metadata = Vec::new();
        metadata
            .try_reserve_exact(encoded_terms.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "encoded grouped-union cached metadata",
                count: encoded_terms.len(),
            })?;
        for (encoded_postings, encoded_block_max) in encoded_terms {
            let postings = encoded_postings.posting_list()?;
            metadata.push(Arc::new(
                postings.into_pruning_metadata(encoded_block_max.as_bytes(), fieldnorms)?,
            ));
        }
        Ok(metadata)
    }

    fn validated_block_max_entries(
        encoded_terms: &[(EncodedPostingList, EncodedBlockMax)],
        posting_lists: &[crate::quiver::PostingList<'_>],
        fieldnorms: DocLenField<'_>,
    ) -> Result<Vec<Arc<[BlockMaxEntry]>>, crate::quiver::BlockMaxError> {
        encoded_terms
            .iter()
            .zip(posting_lists)
            .map(|((_, block_max), postings)| {
                block_max
                    .block_max_list(postings, fieldnorms)
                    .map(|bounds| Arc::<[BlockMaxEntry]>::from(bounds.entries()))
            })
            .collect()
    }

    fn postings(docs: &[u32]) -> Vec<Posting> {
        docs.iter().map(|&doc| Posting::new(doc, 1)).collect()
    }

    #[test]
    fn all_scorer_is_half_open_unit_scored_and_progresses_monotonically()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut all = ReferenceScorer::all(7, 10, 2)?;
        assert_eq!(all.doc(), Some(7));
        assert_eq!(all.cost(), 3);
        assert_eq!(all.size_hint(), 3);
        assert_eq!(all.segment_num_docs(), 2);
        assert_eq!(all.score()?.to_bits(), 1.0_f32.to_bits());

        assert_eq!(all.advance(6)?, Some(7));
        assert_eq!(all.advance(7)?, Some(7));
        assert_eq!(all.advance(8)?, Some(8));
        assert_eq!(all.next()?, Some(9));
        assert_eq!(all.advance(9)?, Some(9));
        assert_eq!(all.next()?, None);
        assert_eq!(all.next()?, None, "exhaustion is fused");
        assert_eq!(
            all.advance(0)?,
            None,
            "an exhausted scorer cannot resurrect"
        );
        assert!(matches!(
            all.score(),
            Err(ArgusError::CursorInvariant(
                "cannot score an exhausted All scorer"
            ))
        ));

        let visible = |docid| docid != 8;
        let mut top_docs = ReferenceScorer::all(7, 10, 2)?;
        assert_eq!(
            top_docs.top_k(3, &visible)?,
            vec![
                ScoredDoc {
                    global_docid: 7,
                    score: 1.0,
                },
                ScoredDoc {
                    global_docid: 9,
                    score: 1.0,
                },
            ]
        );
        let mut doc_set = ReferenceScorer::all(7, 10, 2)?;
        assert_eq!(doc_set.collect_doc_set(&visible)?, vec![7, 9]);

        let mut boosted = ReferenceScorer::all_with_boost(7, 10, 2, 2.5)?;
        assert_eq!(boosted.score()?.to_bits(), 2.5_f32.to_bits());
        assert!(matches!(
            ReferenceScorer::all_with_boost(7, 10, 2, f32::NAN),
            Err(ArgusError::InvalidAllBoost { boost_bits }) if boost_bits == f32::NAN.to_bits()
        ));
        assert!(matches!(
            ReferenceScorer::all_with_boost(7, 10, 2, f32::INFINITY),
            Err(ArgusError::InvalidAllBoost { boost_bits })
                if boost_bits == f32::INFINITY.to_bits()
        ));
        Ok(())
    }

    #[test]
    fn all_scorer_validates_empty_and_u32_boundary_domains()
    -> Result<(), Box<dyn std::error::Error>> {
        let empty = ReferenceScorer::all(17, 17, 0)?;
        assert_eq!(empty.doc(), None);
        assert_eq!(empty.cost(), 0);
        assert_eq!(empty.size_hint(), 0);
        assert_eq!(empty.segment_num_docs(), 0);

        assert!(matches!(
            ReferenceScorer::all(18, 17, 0),
            Err(ArgusError::InvalidAllRange { .. })
        ));
        assert!(matches!(
            ReferenceScorer::all(0, MAX_GLOBAL_DOCID_EXCLUSIVE + 1, 0),
            Err(ArgusError::InvalidAllRange { .. })
        ));
        assert!(matches!(
            ReferenceScorer::all(17, 18, 2),
            Err(ArgusError::InvalidAllRange { .. })
        ));

        let mut final_doc =
            ReferenceScorer::all(u64::from(u32::MAX), MAX_GLOBAL_DOCID_EXCLUSIVE, 1)?;
        assert_eq!(final_doc.doc(), Some(u32::MAX));
        assert_eq!(final_doc.score()?.to_bits(), 1.0_f32.to_bits());
        assert_eq!(final_doc.advance(u32::MAX)?, Some(u32::MAX));
        assert_eq!(final_doc.next()?, None);

        let mut full_domain = ReferenceScorer::all(0, MAX_GLOBAL_DOCID_EXCLUSIVE, u32::MAX)?;
        assert_eq!(full_domain.cost(), MAX_GLOBAL_DOCID_EXCLUSIVE);
        assert_eq!(full_domain.size_hint(), u32::MAX);
        assert_eq!(full_domain.segment_num_docs(), u32::MAX);
        assert_eq!(full_domain.advance(u32::MAX)?, Some(u32::MAX));
        assert_eq!(full_domain.next()?, None);
        Ok(())
    }

    #[test]
    fn all_anchor_composes_with_exclusion_for_top_docs_and_doc_sets()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 4];
        let encoded =
            EncodedDocLenSection::encode(10, 14, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 4, 4)?;

        let mut scored = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::all(10, 14, 4)?),
            ScorerClause::must_not(term(postings(&[11, 13]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert_eq!(
            scored.top_k(4, &AllLiveDocs)?,
            vec![
                ScoredDoc {
                    global_docid: 10,
                    score: 1.0,
                },
                ScoredDoc {
                    global_docid: 12,
                    score: 1.0,
                },
            ],
            "the All anchor contributes Tantivy's unit score and MustNot only gates matches"
        );

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(ReferenceScorer::all_with_boost(10, 14, 4, 2.5)?),
            ScorerClause::must_not(term(postings(&[11, 13]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![10, 12]);
        Ok(())
    }

    #[test]
    fn boolean_raw_all_does_not_change_required_or_optional_term_score_bits()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 4];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 4, 4)?;
        let expected = expected_term_score(
            &snapshot,
            2,
            field.fieldnorm_id(1).expect("fieldnorm exists"),
            1,
            1.0,
        );

        let mut all_and_term = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
            ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        let hits = all_and_term.top_k(4, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!(
            hits.iter()
                .all(|hit| hit.score.to_bits() == expected.to_bits())
        );

        let mut optional_all = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        let hits = optional_all.top_k(4, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!(
            hits.iter()
                .all(|hit| hit.score.to_bits() == expected.to_bits())
        );

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
            ScorerClause::must(term(postings(&[1, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![1, 3]);
        Ok(())
    }

    #[test]
    fn boolean_raw_should_all_is_restored_once_when_it_supplies_matching()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 4];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 4, 4)?;
        let term_score = expected_term_score(
            &snapshot,
            1,
            field.fieldnorm_id(1).expect("fieldnorm exists"),
            1,
            1.0,
        );
        let expected_overlap = term_score + 1.0;

        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(query.doc(), Some(0));
        assert_eq!(query.score()?.to_bits(), 1.0_f32.to_bits());
        assert_eq!(query.next()?, Some(1));
        assert_eq!(query.score()?.to_bits(), expected_overlap.to_bits());
        Ok(())
    }

    #[test]
    fn boolean_duplicate_raw_all_collapses_to_one_unit_score()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut required = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
            ScorerClause::must(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(required.score()?.to_bits(), 1.0_f32.to_bits());

        let mut optional = ReferenceScorer::boolean(vec![
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(optional.score()?.to_bits(), 1.0_f32.to_bits());

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
            ScorerClause::should(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(unscored.cost(), 4);
        assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn boolean_raw_all_exclusion_lowers_to_empty_in_both_modes()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut scored = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
            ScorerClause::must_not(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(scored.doc(), None);
        assert_eq!(scored.cost(), 0);
        assert_eq!(scored.size_hint(), 0);
        assert_eq!(scored.segment_num_docs(), 0);
        assert!(scored.top_k(4, &AllLiveDocs)?.is_empty());

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
            ScorerClause::must_not(ReferenceScorer::all(0, 4, 4)?),
        ])?;
        assert_eq!(unscored.doc(), None);
        assert_eq!(unscored.cost(), 0);
        assert_eq!(unscored.size_hint(), 0);
        assert_eq!(unscored.segment_num_docs(), 0);
        assert!(unscored.collect_doc_set(&AllLiveDocs)?.is_empty());
        Ok(())
    }

    #[test]
    fn boolean_boosted_all_is_not_elided_from_score_bits() -> Result<(), Box<dyn std::error::Error>>
    {
        let lengths = [Some(1); 4];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 4, 4)?;
        let term_score = expected_term_score(
            &snapshot,
            1,
            field.fieldnorm_id(1).expect("fieldnorm exists"),
            1,
            1.0,
        );

        let mut required = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
            ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let expected_required = term_score + 2.5;
        assert_eq!(required.doc(), Some(1));
        assert_eq!(required.score()?.to_bits(), expected_required.to_bits());

        let mut optional = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
        ])?;
        let mut expected_optional = 0.0_f32;
        expected_optional += term_score;
        expected_optional += 2.5;
        assert_eq!(optional.doc(), Some(1));
        assert_eq!(optional.score()?.to_bits(), expected_optional.to_bits());

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(ReferenceScorer::all_with_boost(0, 4, 4, 2.5)?),
            ScorerClause::must(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![1]);
        Ok(())
    }

    fn expected_term_score(
        snapshot: &Bm25FieldSnapshot,
        doc_freq: u64,
        fieldnorm_id: u8,
        frequency: u32,
        boost: f32,
    ) -> f32 {
        let average = snapshot
            .average_field_length()
            .expect("non-empty scored fixture");
        let decoded = id_to_fieldnorm(fieldnorm_id);
        let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
        let mut weight = idf(doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
        weight *= boost;
        let frequency = frequency as f32;
        weight * (frequency / (frequency + norm))
    }

    fn expected_phrase_score(
        snapshot: &Bm25FieldSnapshot,
        doc_freqs: &[u64],
        fieldnorm_id: u8,
        frequency: u32,
        boost: f32,
    ) -> f32 {
        let average = snapshot
            .average_field_length()
            .expect("non-empty scored fixture");
        let decoded = id_to_fieldnorm(fieldnorm_id);
        let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
        let mut idf_sum = 0.0_f32;
        for &doc_freq in doc_freqs {
            idf_sum += idf(doc_freq, snapshot.doc_count());
        }
        let mut weight = idf_sum * (1.0 + BM25_K1);
        weight *= boost;
        let frequency = frequency as f32;
        weight * (frequency / (frequency + norm))
    }

    #[test]
    fn basic_record_option_clamps_repeated_edge_ngram_frequency()
    -> Result<(), Box<dyn std::error::Error>> {
        // Minimized from the pinned 40-path CASS witness: the target prefix
        // field has 1,775 analyzed tokens, and seven `tokens` source values
        // each emit the `token` edge n-gram. Across the corpus that term has
        // df=16 and the prefix field has 84,416 total tokens.
        let mut lengths = vec![Some(0); 40];
        lengths[39] = Some(1_775);
        let encoded =
            EncodedDocLenSection::encode(0, 40, &[9], &[DocLenFieldInput::new(9, &lengths)])?;
        let section = encoded.section(&[9])?;
        let field = section.field(9).expect("content_prefix field exists");
        let snapshot = snapshot(9, 84_416, 40)?;
        let scorer = |record_option| {
            TermScorer::new(
                VecCursor::new(vec![Posting::new(39, 7)], 1, 40),
                field,
                snapshot.clone(),
                16,
                record_option,
                1.0,
            )
        };

        let basic = scorer(TermRecordOption::Basic)?;
        let with_freqs = scorer(TermRecordOption::WithFreqs)?;
        let with_positions = scorer(TermRecordOption::WithFreqsAndPositions)?;
        let fieldnorm_id = field.fieldnorm_id(39).expect("target fieldnorm exists");
        let expected_basic = expected_term_score(&snapshot, 16, fieldnorm_id, 1, 1.0);
        let expected_frequency = expected_term_score(&snapshot, 16, fieldnorm_id, 7, 1.0);

        assert_eq!(basic.score()?.to_bits(), expected_basic.to_bits());
        assert_eq!(with_freqs.score()?.to_bits(), expected_frequency.to_bits());
        assert_eq!(
            with_positions.score()?.to_bits(),
            expected_frequency.to_bits(),
            "frequency-bearing record options must retain the raw tf"
        );
        assert_eq!(
            (with_freqs.score()? - basic.score()?).to_bits(),
            0x3f41_a57a,
            "the minimized Basic-vs-frequency leaf delta is 0.756431222"
        );

        let maximum = TermScorer::new(
            VecCursor::new(vec![Posting::new(39, u32::MAX)], 1, 40),
            field,
            snapshot.clone(),
            16,
            TermRecordOption::Basic,
            1.0,
        )?;
        assert_eq!(
            maximum.score()?.to_bits(),
            expected_basic.to_bits(),
            "Basic scoring must remain presence-only at maximum stored frequency"
        );

        let bound_rows = vec![Posting::new(39, 7)];
        let (encoded_postings, encoded_block_max) =
            EncodedPostingList::encode_with_block_max(&bound_rows, |doc| {
                field.fieldnorm_id(u64::from(doc))
            })?;
        let posting_list = encoded_postings.posting_list()?;
        let bounds = encoded_block_max.block_max_list(&posting_list, field)?;
        let bounds = Arc::<[BlockMaxEntry]>::from(bounds.entries());
        let bounded = |record_option| {
            TermScorer::new(
                SealedPostingCursor::with_block_max(
                    posting_list.cursor()?,
                    Arc::clone(&bounds),
                    posting_list.doc_freq(),
                    40,
                ),
                field,
                snapshot.clone(),
                16,
                record_option,
                1.0,
            )
        };
        for (record_option, expected) in [
            (TermRecordOption::Basic, expected_basic),
            (TermRecordOption::WithFreqs, expected_frequency),
            (TermRecordOption::WithFreqsAndPositions, expected_frequency),
        ] {
            let scorer = bounded(record_option)?;
            assert_eq!(
                scorer
                    .term_score_upper_bound
                    .expect("sealed term bound")
                    .to_bits(),
                expected.to_bits(),
                "{record_option:?} whole-term bound must use its effective frequency"
            );
            let competitive = scorer.competitive_cursor().expect("competitive cursor");
            assert_eq!(
                competitive
                    .current_block_upper_bound()
                    .expect("current block bound")
                    .to_bits(),
                expected.to_bits(),
                "{record_option:?} current-block bound must use its effective frequency"
            );
        }
        Ok(())
    }

    #[test]
    fn term_scorer_rejects_nonprogress_backward_seek_and_resurrection()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 8];
        let encoded =
            EncodedDocLenSection::encode(0, 8, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 8, 8)?;
        let scorer = |fault| {
            TermScorer::new(
                FaultCursor::new(fault),
                field,
                snapshot.clone(),
                2,
                TermRecordOption::WithFreqs,
                1.0,
            )
        };

        let mut sticky = scorer(CursorFault::StickyNext)?;
        assert!(matches!(sticky.next(), Err(ArgusError::CursorInvariant(_))));
        let mut backward = scorer(CursorFault::BackwardNext)?;
        assert!(matches!(
            backward.next(),
            Err(ArgusError::CursorInvariant(_))
        ));
        let mut below_target = scorer(CursorFault::BelowTargetAdvance)?;
        assert!(matches!(
            below_target.seek(4),
            Err(ArgusError::CursorInvariant(_))
        ));
        let mut moved = scorer(CursorFault::MovesDespiteSatisfiedTarget)?;
        assert!(matches!(moved.seek(4), Err(ArgusError::CursorInvariant(_))));
        let mut resurrecting = scorer(CursorFault::ResurrectsAfterExhaustion)?;
        assert_eq!(resurrecting.next()?, None);
        assert!(matches!(
            resurrecting.next(),
            Err(ArgusError::CursorInvariant(_))
        ));

        assert!(matches!(
            TermScorer::new(
                VecCursor::new(Vec::new(), 1, 8),
                field,
                snapshot,
                2,
                TermRecordOption::WithFreqs,
                1.0,
            ),
            Err(ArgusError::CursorInvariant(_))
        ));
        Ok(())
    }

    #[test]
    fn sealed_cursor_adapter_preserves_option_state_and_max_docid() -> Result<(), ArgusError> {
        let source = [
            Posting::new(3, 2),
            Posting::new(130, 1),
            Posting::new(u32::MAX, 7),
        ];
        let encoded = EncodedPostingList::encode(&source)?;
        let list = encoded.posting_list()?;
        let expected_positions = [1, 4, 9, 0, 1, 2, 3, 4, 5, 6];
        let encoded_positions = EncodedPositionList::encode(&source, &expected_positions)?;
        let positions = encoded_positions.position_list(&list)?;
        let mut cursor = SealedPostingCursor::with_positions(&positions, 3)?;

        assert_eq!(PostingCursor::doc(&cursor), Some(3));
        assert_eq!(PostingCursor::freq(&cursor), Some(2));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 0);
        let mut decoded_positions = vec![u32::MAX];
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![1, 4]);
        assert_eq!(PostingCursor::advance(&mut cursor, 3)?, Some(3));
        assert_eq!(PostingCursor::advance(&mut cursor, 4)?, Some(130));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 1);
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![9]);
        assert_eq!(
            PostingCursor::advance(&mut cursor, u32::MAX)?,
            Some(u32::MAX)
        );
        assert_eq!(PostingCursor::freq(&cursor), Some(7));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 2);
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::doc(&cursor), None);
        assert_eq!(PostingCursor::freq(&cursor), None);
        assert!(PostingCursor::positions_handle(&cursor).is_none());

        let cursor = SealedPostingCursor::new(&list, 3)?;
        assert_eq!(PostingCursor::doc(&cursor), Some(3));
        assert!(PostingCursor::positions_handle(&cursor).is_none());
        Ok(())
    }

    #[test]
    fn phrase_scorer_reads_sealed_postings_and_positions() -> Result<(), Box<dyn std::error::Error>>
    {
        let first_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
        let encoded_first_postings = EncodedPostingList::encode(&first_postings)?;
        let first_posting_list = encoded_first_postings.posting_list()?;
        let encoded_first_positions = EncodedPositionList::encode(&first_postings, &[0, 4, 0, 1])?;
        let first_positions = encoded_first_positions.position_list(&first_posting_list)?;

        let second_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
        let encoded_second_postings = EncodedPostingList::encode(&second_postings)?;
        let second_posting_list = encoded_second_postings.posting_list()?;
        let encoded_second_positions =
            EncodedPositionList::encode(&second_postings, &[1, 5, 2, 2])?;
        let second_positions = encoded_second_positions.position_list(&second_posting_list)?;

        let lengths = [Some(8); 4];
        let encoded_doclens =
            EncodedDocLenSection::encode(0, 4, &[6], &[DocLenFieldInput::new(6, &lengths)])?;
        let doclens = encoded_doclens.section(&[6])?;
        let field = doclens.field(6).expect("field exists");
        let snapshot = snapshot(6, 32, 4)?;
        let mut phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(
                    6,
                    0,
                    SealedPostingCursor::with_positions(&first_positions, 4)?,
                    3,
                ),
                PhraseTerm::new(
                    6,
                    1,
                    SealedPostingCursor::with_positions(&second_positions, 4)?,
                    3,
                ),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;

        assert_eq!(phrase.doc(), Some(1));
        let expected = expected_phrase_score(
            &snapshot,
            &[3, 3],
            field.fieldnorm_id(1).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, Some(3));
        let expected = expected_phrase_score(
            &snapshot,
            &[3, 3],
            field.fieldnorm_id(3).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        Ok(())
    }

    #[test]
    fn phrase_scorer_filters_candidates_counts_frequency_and_seeks()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(8); 8];
        let encoded =
            EncodedDocLenSection::encode(0, 8, &[5], &[DocLenFieldInput::new(5, &lengths)])?;
        let section = encoded.section(&[5])?;
        let field = section.field(5).expect("field exists");
        let snapshot = snapshot(5, 64, 8)?;
        let build = || {
            let first = PositionedVecCursor::new(
                vec![(1, vec![0, 4]), (2, vec![0]), (3, vec![2]), (5, vec![1])],
                4,
                8,
            );
            let second = PositionedVecCursor::new(
                vec![(1, vec![1, 5]), (2, vec![2]), (3, vec![3]), (5, vec![0])],
                4,
                8,
            );
            PhraseScorer::new(
                vec![
                    PhraseTerm::new(5, 0, first, 4),
                    PhraseTerm::new(5, 1, second, 4),
                ],
                field,
                snapshot.clone(),
                1.0,
            )
        };

        let mut phrase = build()?;
        assert_eq!(phrase.doc(), Some(1));
        let candidate_estimate = estimate_intersection([4, 4].into_iter(), 8);
        assert_eq!(phrase.cost, u64::from(candidate_estimate) * 20);
        assert_eq!(phrase.size_hint, candidate_estimate / 20);
        let expected = expected_phrase_score(
            &snapshot,
            &[4, 4],
            field.fieldnorm_id(1).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        let decode_capacity = phrase.decode_scratch.capacity();
        let position_capacities = phrase
            .slots
            .iter()
            .map(|slot| slot.positions.capacity())
            .collect::<Vec<_>>();
        assert_eq!(phrase.next()?, Some(3));
        assert_eq!(phrase.decode_scratch.capacity(), decode_capacity);
        assert_eq!(
            phrase
                .slots
                .iter()
                .map(|slot| slot.positions.capacity())
                .collect::<Vec<_>>(),
            position_capacities
        );
        let expected = expected_phrase_score(
            &snapshot,
            &[4, 4],
            field.fieldnorm_id(3).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        assert_eq!(phrase.next()?, None);

        let mut phrase = build()?;
        assert_eq!(phrase.seek(2)?, Some(3));
        assert_eq!(phrase.seek(3)?, Some(3));
        assert_eq!(phrase.seek(4)?, None);

        let mut scorer = ReferenceScorer::phrase(build()?);
        let winners = scorer.top_k(10, &AllLiveDocs)?;
        assert_eq!(
            winners
                .iter()
                .map(|winner| winner.global_docid)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        Ok(())
    }

    #[test]
    fn phrase_cost_estimate_follows_tantivy_child_cost_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2); 12];
        let encoded =
            EncodedDocLenSection::encode(0, 12, &[4], &[DocLenFieldInput::new(4, &lengths)])?;
        let section = encoded.section(&[4])?;
        let field = section.field(4).expect("field exists");
        let snapshot = snapshot(4, 24, 12)?;
        let broad = PositionedVecCursor::new(
            vec![
                (0, vec![0]),
                (1, vec![0]),
                (2, vec![0]),
                (3, vec![0]),
                (4, vec![0]),
            ],
            5,
            12,
        );
        let narrow = PositionedVecCursor::new(vec![(0, vec![1])], 1, 12);
        let phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(4, 0, broad, 5),
                PhraseTerm::new(4, 1, narrow, 1),
            ],
            field,
            snapshot,
            1.0,
        )?;

        assert_eq!(estimate_intersection([5, 1].into_iter(), 12), 0);
        assert_eq!(estimate_intersection([1, 5].into_iter(), 12), 1);
        let scorer = ReferenceScorer::phrase(phrase);
        assert_eq!(scorer.doc(), Some(0));
        assert_eq!(scorer.cost(), 20);
        assert_eq!(scorer.size_hint(), 0);
        Ok(())
    }

    #[test]
    fn phrase_scorer_handles_repeated_terms_same_position_or_and_query_gaps()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(6); 16];
        let encoded =
            EncodedDocLenSection::encode(0, 16, &[2], &[DocLenFieldInput::new(2, &lengths)])?;
        let section = encoded.section(&[2])?;
        let field = section.field(2).expect("field exists");
        let snapshot = snapshot(2, 96, 16)?;

        let repeated_a = || PositionedVecCursor::new(vec![(7, vec![0, 1, 2])], 1, 16);
        let repeated_b = PositionedVecCursor::new(vec![(7, vec![2, 3])], 1, 16);
        let repeated = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, repeated_a(), 1),
                PhraseTerm::new(2, 1, repeated_a(), 1),
                PhraseTerm::new(2, 2, repeated_b, 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;
        assert_eq!(repeated.doc(), Some(7));
        let expected = expected_phrase_score(
            &snapshot,
            &[1, 1, 1],
            field.fieldnorm_id(7).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(repeated.score()?.to_bits(), expected.to_bits());

        let alternative_x = PositionedVecCursor::new(vec![(8, vec![5]), (10, vec![1])], 2, 16);
        let alternative_y = PositionedVecCursor::new(vec![(9, vec![10]), (10, vec![1])], 2, 16);
        let following =
            PositionedVecCursor::new(vec![(8, vec![6]), (9, vec![11]), (10, vec![2])], 3, 16);
        let mut alternatives = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, alternative_x, 2),
                PhraseTerm::new(2, 0, alternative_y, 2),
                PhraseTerm::new(2, 1, following, 3),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;
        for expected_doc in [8, 9, 10] {
            assert_eq!(alternatives.doc(), Some(expected_doc));
            let expected = expected_phrase_score(
                &snapshot,
                &[2, 2, 3],
                field
                    .fieldnorm_id(u64::from(expected_doc))
                    .expect("fieldnorm"),
                1,
                1.0,
            );
            assert_eq!(alternatives.score()?.to_bits(), expected.to_bits());
            alternatives.next()?;
        }
        assert_eq!(alternatives.doc(), None);

        let gap_start = PositionedVecCursor::new(vec![(6, vec![10]), (11, vec![20])], 2, 16);
        let gap_end = PositionedVecCursor::new(vec![(6, vec![12]), (11, vec![21])], 2, 16);
        let mut gapped = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 4, gap_start, 2),
                PhraseTerm::new(2, 6, gap_end, 2),
            ],
            field,
            snapshot,
            1.0,
        )?;
        assert_eq!(gapped.doc(), Some(6));
        assert_eq!(gapped.next()?, None);
        Ok(())
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn phrase_scorer_matches_tantivy_corpus_docs_ranks_and_score_bits()
    -> Result<(), Box<dyn std::error::Error>> {
        use frankensearch_lexical::tantivy_crate::{
            Index, TantivyDocument, Term,
            collector::{DocSetCollector, TopDocs},
            doc,
            query::PhraseQuery,
            schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
        };

        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(indexing),
        );
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
        let corpus = [
            "a b",
            "a b a b",
            "a x b",
            "a only",
            "b a",
            "a b filler filler filler filler filler filler",
        ];
        for text in corpus {
            writer.add_document(doc!(body => text))?;
        }
        writer.commit()?;
        drop(writer);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let query = PhraseQuery::new(vec![
            Term::from_field_text(body, "a"),
            Term::from_field_text(body, "b"),
        ]);
        let mut oracle_docset = searcher
            .search(&query, &DocSetCollector)?
            .into_iter()
            .map(|address| {
                assert_eq!(address.segment_ord, 0);
                address.doc_id
            })
            .collect::<Vec<_>>();
        oracle_docset.sort_unstable();
        assert_eq!(oracle_docset, vec![0, 1, 5]);
        let oracle_ranked = searcher
            .search(&query, &TopDocs::with_limit(corpus.len()).order_by_score())?
            .into_iter()
            .map(|(score, address)| {
                assert_eq!(address.segment_ord, 0);
                (address.doc_id, score.to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            oracle_ranked
                .iter()
                .map(|(doc, _)| *doc)
                .collect::<Vec<_>>(),
            vec![1, 0, 5]
        );

        let lengths = [Some(2), Some(4), Some(3), Some(2), Some(2), Some(8)];
        let encoded =
            EncodedDocLenSection::encode(0, 6, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
        let section = encoded.section(&[0])?;
        let field = section.field(0).expect("field exists");
        let snapshot = snapshot(0, 21, 6)?;
        let a = PositionedVecCursor::new(
            vec![
                (0, vec![0]),
                (1, vec![0, 2]),
                (2, vec![0]),
                (3, vec![0]),
                (4, vec![1]),
                (5, vec![0]),
            ],
            6,
            6,
        );
        let b = PositionedVecCursor::new(
            vec![
                (0, vec![1]),
                (1, vec![1, 3]),
                (2, vec![2]),
                (4, vec![0]),
                (5, vec![1]),
            ],
            5,
            6,
        );
        let phrase = PhraseScorer::new(
            vec![PhraseTerm::new(0, 0, a, 6), PhraseTerm::new(0, 1, b, 5)],
            field,
            snapshot,
            1.0,
        )?;
        let mut scorer = ReferenceScorer::phrase(phrase);
        let quill_ranked = scorer
            .top_k(corpus.len(), &AllLiveDocs)?
            .into_iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(quill_ranked, oracle_ranked);
        Ok(())
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn phrase_same_position_alternatives_are_reviewed_or_divergence()
    -> Result<(), Box<dyn std::error::Error>> {
        use frankensearch_lexical::tantivy_crate::{
            Index, TantivyDocument, Term,
            collector::DocSetCollector,
            doc,
            query::PhraseQuery,
            schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
            tokenizer::{PreTokenizedString, Token},
        };

        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(indexing),
        );
        let index = Index::create_in_ram(schema_builder.build());
        let token = |text: &str, position: usize, offset_from: usize, offset_to: usize| Token {
            offset_from,
            offset_to,
            position,
            text: text.to_owned(),
            position_length: 1,
        };
        let docs = [
            PreTokenizedString {
                text: "a c".to_owned(),
                tokens: vec![token("a", 0, 0, 1), token("c", 1, 2, 3)],
            },
            PreTokenizedString {
                text: "b c".to_owned(),
                tokens: vec![token("b", 0, 0, 1), token("c", 1, 2, 3)],
            },
            PreTokenizedString {
                text: "a/b c".to_owned(),
                tokens: vec![
                    token("a", 0, 0, 3),
                    token("b", 0, 0, 3),
                    token("c", 1, 4, 5),
                ],
            },
        ];
        let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
        for text in docs {
            writer.add_document(doc!(body => text))?;
        }
        writer.commit()?;
        drop(writer);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let oracle_query = PhraseQuery::new_with_offset(vec![
            (0, Term::from_field_text(body, "a")),
            (0, Term::from_field_text(body, "b")),
            (1, Term::from_field_text(body, "c")),
        ]);
        let mut oracle_docids = searcher
            .search(&oracle_query, &DocSetCollector)?
            .into_iter()
            .map(|address| {
                assert_eq!(address.segment_ord, 0);
                address.doc_id
            })
            .collect::<Vec<_>>();
        oracle_docids.sort_unstable();
        assert_eq!(oracle_docids, vec![2]);

        let lengths = [Some(2), Some(2), Some(3)];
        let encoded =
            EncodedDocLenSection::encode(0, 3, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
        let section = encoded.section(&[0])?;
        let field = section.field(0).expect("field exists");
        let phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(
                    0,
                    0,
                    PositionedVecCursor::new(vec![(0, vec![0]), (2, vec![0])], 2, 3),
                    2,
                ),
                PhraseTerm::new(
                    0,
                    0,
                    PositionedVecCursor::new(vec![(1, vec![0]), (2, vec![0])], 2, 3),
                    2,
                ),
                PhraseTerm::new(
                    0,
                    1,
                    PositionedVecCursor::new(vec![(0, vec![1]), (1, vec![1]), (2, vec![1])], 3, 3),
                    3,
                ),
            ],
            field,
            snapshot(0, 7, 3)?,
            1.0,
        )?;
        let mut scorer = ReferenceScorer::phrase(phrase);
        let mut quill_docids = scorer
            .top_k(3, &AllLiveDocs)?
            .into_iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>();
        quill_docids.sort_unstable();
        assert_eq!(quill_docids, vec![0, 1, 2]);
        assert_ne!(quill_docids, oracle_docids);
        Ok(())
    }

    #[test]
    fn phrase_frequency_exceeds_u8_and_matches_u32_position_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(400); 16];
        let encoded =
            EncodedDocLenSection::encode(0, 16, &[9], &[DocLenFieldInput::new(9, &lengths)])?;
        let section = encoded.section(&[9])?;
        let field = section.field(9).expect("field exists");
        let snapshot = snapshot(9, 6_400, 16)?;
        let first = PositionedVecCursor::new(
            vec![(11, (0..=300).collect()), (12, vec![u32::MAX - 1])],
            2,
            16,
        );
        let second =
            PositionedVecCursor::new(vec![(11, (1..=301).collect()), (12, vec![u32::MAX])], 2, 16);
        let mut phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(9, 0, first, 2),
                PhraseTerm::new(9, 1, second, 2),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;

        assert_eq!(phrase.doc(), Some(11));
        let expected = expected_phrase_score(
            &snapshot,
            &[2, 2],
            field.fieldnorm_id(11).expect("fieldnorm"),
            301,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, Some(12));
        let expected = expected_phrase_score(
            &snapshot,
            &[2, 2],
            field.fieldnorm_id(12).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        Ok(())
    }

    #[test]
    fn phrase_constructor_rejects_missing_positions_and_invalid_slot_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2); 4];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
        let section = encoded.section(&[3])?;
        let field = section.field(3).expect("field exists");
        let snapshot = snapshot(3, 8, 4)?;

        let wrong_field = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(
            wrong_field,
            Err(ArgusError::PhraseTermFieldMismatch {
                term_field: 2,
                stats_field: 3
            })
        ));

        let missing_positions = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 0, VecCursor::new(postings(&[0]), 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(
            missing_positions,
            Err(ArgusError::MissingPositions {
                field_ord: 3,
                global_docid: 0
            })
        ));

        let descending = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 2, PositionedVecCursor::new(vec![(0, vec![2])], 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(descending, Err(ArgusError::InvalidPhrase { .. })));

        let one_slot = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
                PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
            ],
            field,
            snapshot,
            1.0,
        );
        assert!(matches!(one_slot, Err(ArgusError::InvalidPhrase { .. })));
        Ok(())
    }

    #[test]
    fn bm25_score_uses_raw_average_cached_norm_and_boost_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(3)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[7], &[DocLenFieldInput::new(7, &lengths)])?;
        let section = encoded.section(&[7])?;
        let field = section.field(7).expect("field exists");
        let snapshot = snapshot(7, 9, 3)?;
        let mut scorer = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 1.0)?;

        // Spreadsheet row: N=3, n=2, total_tokens=9, dl=3, f=2.
        // idf=0.47000363, norm=1.2, tf=0.625, score=0.646255.
        let score = scorer.score()?;
        assert_eq!(score.to_bits(), 1_059_418_360);
        let expected = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 1.0);
        assert_eq!(score.to_bits(), expected.to_bits());

        let mut boosted = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 2.0)?;
        let boosted_score = boosted.score()?;
        let expected_boosted = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 2.0);
        assert_eq!(boosted_score.to_bits(), expected_boosted.to_bits());
        Ok(())
    }

    #[test]
    fn fieldnorm_boundaries_use_the_shared_snapshot_cache() -> Result<(), Box<dyn std::error::Error>>
    {
        let lengths = [Some(0), Some(40), Some(42), Some(2_013_265_944)];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
        let section = encoded.section(&[3])?;
        let field = section.field(3).expect("field exists");
        let snapshot = snapshot(3, 2_013_266_026, 4)?;
        let mut scorer = term(
            vec![
                Posting::new(0, 1),
                Posting::new(1, 2),
                Posting::new(2, 3),
                Posting::new(3, 4),
            ],
            field,
            &snapshot,
            4,
            4,
            1.0,
        )?;

        for (doc, fieldnorm_id, frequency) in [(0, 0, 1), (1, 40, 2), (2, 41, 3), (3, 255, 4)] {
            assert_eq!(scorer.doc(), Some(doc));
            let score = scorer.score()?;
            let expected = expected_term_score(&snapshot, 4, fieldnorm_id, frequency, 1.0);
            assert_eq!(score.to_bits(), expected.to_bits());
            scorer.next()?;
        }
        assert_eq!(scorer.doc(), None);
        Ok(())
    }

    #[test]
    fn title_boost_wins_and_equal_scores_tie_by_global_docid()
    -> Result<(), Box<dyn std::error::Error>> {
        let content_lengths = [Some(1), Some(1), Some(1)];
        let title_lengths = [Some(1), Some(1), Some(1)];
        let inputs = [
            DocLenFieldInput::new(1, &content_lengths),
            DocLenFieldInput::new(2, &title_lengths),
        ];
        let encoded = EncodedDocLenSection::encode(0, 3, &[1, 2], &inputs)?;
        let section = encoded.section(&[1, 2])?;
        let content = section.field(1).expect("content field");
        let title = section.field(2).expect("title field");
        let content_snapshot = snapshot(1, 3, 3)?;
        let title_snapshot = snapshot(2, 3, 3)?;
        let content_term = term(postings(&[0, 2]), content, &content_snapshot, 2, 2, 1.0)?;
        let title_term = term(postings(&[1]), title, &title_snapshot, 1, 1, 2.0)?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::should(content_term),
            ScorerClause::should(title_term),
        ])?;
        let winners = query.top_k(3, &AllLiveDocs)?;

        assert_eq!(
            winners
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        assert_eq!(winners[1].score.to_bits(), winners[2].score.to_bits());
        assert!(winners[0].score > winners[1].score);
        Ok(())
    }

    #[test]
    fn boolean_occurs_match_tantivy_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 5];
        let encoded =
            EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5, 5)?;

        let mut must_with_optional = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = must_with_optional.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![1]
        );
        let expected_must = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        assert_eq!(hits[0].score.to_bits(), expected_must.to_bits());

        let mut should_with_exclusion = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = should_with_exclusion.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![3]
        );

        let mut negative_only = ReferenceScorer::boolean(vec![ScorerClause::must_not(term(
            postings(&[1, 2]),
            field,
            &snapshot,
            2,
            2,
            1.0,
        )?)])?;
        assert!(negative_only.top_k(5, &AllLiveDocs)?.is_empty());

        let mut multiple_negative_only = ReferenceScorer::boolean(vec![
            ScorerClause::must_not(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(multiple_negative_only.segment_num_docs(), 5);
        assert_eq!(multiple_negative_only.size_hint(), 0);
        assert_eq!(multiple_negative_only.cost(), 0);
        assert!(multiple_negative_only.top_k(5, &AllLiveDocs)?.is_empty());
        Ok(())
    }

    /// Executable Boolean-shape proof for the oversized-query-token class
    /// (`bd-quill-e0-contracts-j53p.8`, register entry DIV-004).
    ///
    /// Quill's symmetric admission rule means an oversized (>65,530-byte)
    /// query token can never name a dictionary entry, so its lowered leaf is
    /// [`ReferenceScorer::empty`]. `boolean_with_mode` then shorts
    /// `Must(empty)` to `MatchNone` and drops empty `Should`/`MustNot`
    /// clauses. The oracle keeps the same unmatchable leaf as an empty
    /// posting list. This proof enumerates every clause list of length 1..=3
    /// over {matchable, oversized} x {`Must`, `Should`, `MustNot`} and asserts the
    /// two shapes return identical hits with bit-identical scores.
    #[test]
    fn oversized_token_boolean_shapes_are_result_equivalent()
    -> Result<(), Box<dyn std::error::Error>> {
        const DOCS: u32 = 5;
        let lengths = [Some(1); DOCS as usize];
        let encoded = EncodedDocLenSection::encode(
            0,
            u64::from(DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(DOCS), u64::from(DOCS))?;
        // The matchable leaf hits docs {1, 2, 3}; the oversized leaf is
        // unmatchable under every semantics.
        let matchable_docs: [u32; 3] = [1, 2, 3];
        let occurs = [Occur::Must, Occur::Should, Occur::MustNot];
        // (use_quill_empty, occur) -> scorer for one clause
        let build = |use_quill_empty: bool,
                     oversized: bool,
                     occur: Occur|
         -> Result<ScorerClause<'_>, ArgusError> {
            let scorer = if oversized {
                if use_quill_empty {
                    ReferenceScorer::empty()
                } else {
                    term(postings(&[]), field, &snapshot, 0, 0, 1.0)?
                }
            } else {
                term(postings(&matchable_docs), field, &snapshot, 3, 3, 1.0)?
            };
            Ok(ScorerClause::new(occur, scorer))
        };
        let mut case_count = 0_u32;
        for len in 1..=3_usize {
            let slots = usize::pow(6, u32::try_from(len)?);
            for combo in 0..slots {
                let mut remaining = combo;
                let mut quill_clauses = Vec::with_capacity(len);
                let mut oracle_clauses = Vec::with_capacity(len);
                for _slot in 0..len {
                    let oversized = remaining % 2 == 1;
                    let occur = occurs[(remaining / 2) % 3];
                    remaining /= 6;
                    quill_clauses.push(build(true, oversized, occur)?);
                    oracle_clauses.push(build(false, oversized, occur)?);
                }
                let mut quill = ReferenceScorer::boolean(quill_clauses)?;
                let mut oracle = ReferenceScorer::boolean(oracle_clauses)?;
                let quill_hits = quill.top_k(usize::try_from(DOCS)?, &AllLiveDocs)?;
                let oracle_hits = oracle.top_k(usize::try_from(DOCS)?, &AllLiveDocs)?;
                assert_eq!(
                    quill_hits
                        .iter()
                        .map(|hit| hit.global_docid)
                        .collect::<Vec<_>>(),
                    oracle_hits
                        .iter()
                        .map(|hit| hit.global_docid)
                        .collect::<Vec<_>>(),
                    "hit sets differ at combo {combo} (len {len})"
                );
                for (quill_hit, oracle_hit) in quill_hits.iter().zip(&oracle_hits) {
                    assert_eq!(
                        quill_hit.score.to_bits(),
                        oracle_hit.score.to_bits(),
                        "scores differ for doc {} at combo {combo} (len {len})",
                        quill_hit.global_docid
                    );
                }
                case_count += 1;
            }
        }
        assert_eq!(case_count, 6 + 36 + 216);
        Ok(())
    }

    /// The doc-set collector path (unscored lowering) must preserve the same
    /// oversized semantics: empty `Should`/`MustNot` drop, `Must(empty)`
    /// shorts to `MatchNone`, and the remaining shapes match identical sets.
    #[test]
    fn oversized_token_unscored_shapes_are_result_equivalent()
    -> Result<(), Box<dyn std::error::Error>> {
        const DOCS: u32 = 5;
        let lengths = [Some(1); DOCS as usize];
        let encoded = EncodedDocLenSection::encode(
            0,
            u64::from(DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(DOCS), u64::from(DOCS))?;
        let matchable_docs: [u32; 3] = [1, 2, 3];
        let occurs = [Occur::Must, Occur::Should, Occur::MustNot];
        for occur in occurs {
            for oversized in [true, false] {
                let make = |use_quill_empty: bool| -> Result<ReferenceScorer<'_>, ArgusError> {
                    let scorer = if oversized {
                        if use_quill_empty {
                            ReferenceScorer::empty()
                        } else {
                            term(postings(&[]), field, &snapshot, 0, 0, 1.0)?
                        }
                    } else {
                        term(postings(&matchable_docs), field, &snapshot, 3, 3, 1.0)?
                    };
                    ReferenceScorer::boolean_unscored(vec![ScorerClause::new(occur, scorer)])
                };
                let mut quill = make(true)?;
                let mut oracle = make(false)?;
                assert_eq!(
                    quill.collect_doc_set(&AllLiveDocs)?,
                    oracle.collect_doc_set(&AllLiveDocs)?,
                    "doc sets differ for occur {occur:?} oversized={oversized}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn required_cost_sort_and_optional_composition_pin_f32_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 1, 1)?;
        let boosts = [1.0_f32, 1.0e8, -1.0e8, 2.0];
        let costs = [900_u64, 3, 200, 200];
        let mut clauses = Vec::new();
        for (&boost, &cost) in boosts.iter().zip(&costs) {
            clauses.push(ScorerClause::must(term(
                postings(&[0]),
                field,
                &snapshot,
                1,
                cost,
                boost,
            )?));
        }
        clauses.push(ScorerClause::should(term(
            postings(&[0]),
            field,
            &snapshot,
            1,
            1,
            3.0,
        )?));
        let mut query = ReferenceScorer::boolean(clauses)?;
        let actual = query.score()?;

        let score = |boost| expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, boost);
        // Required order after stable cost sort: boosts 1e8, -1e8, 2, 1.
        // Tantivy's intersection expression is `(left + right) + sum(others)`;
        // RequiredOptional then adds the optional aggregate.
        let required = score(1.0e8) + score(-1.0e8) + (score(2.0) + score(1.0));
        let mut expected = 0.0_f32;
        expected += required;
        expected += score(3.0);
        assert_eq!(actual.to_bits(), expected.to_bits());
        Ok(())
    }

    #[test]
    fn intersection_danger_seek_preserves_nested_union_score_order()
    -> Result<(), Box<dyn std::error::Error>> {
        const DOC_COUNT: usize = 30_001;
        let lengths = vec![Some(1); DOC_COUNT];
        let encoded = EncodedDocLenSection::encode(
            0,
            DOC_COUNT as u64,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, DOC_COUNT as u64, DOC_COUNT as u64)?;

        let optional = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 16_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                -1.0e8,
            )?),
            ScorerClause::should(term(postings(&[0, 11_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                1.0e8,
            )?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                1.0,
            )?),
        ])?;
        let required_optional = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&[0, 11_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                0.0,
            )?),
            ScorerClause::should(optional),
        ])?;
        let nested_union = ReferenceScorer::boolean(vec![
            ScorerClause::should(required_optional),
            ScorerClause::should(term(postings(&[0, 20_000]), field, &snapshot, 2, 2, 0.0)?),
        ])?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                1,
                0.0,
            )?),
            ScorerClause::must(nested_union),
        ])?;

        assert_eq!(query.doc(), Some(0));
        assert_eq!(query.next()?, Some(20_000));
        let small = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
        assert_ne!(small.to_bits(), 0.0_f32.to_bits());
        assert_eq!(query.score()?.to_bits(), 0.0_f32.to_bits());
        Ok(())
    }

    #[test]
    fn buffered_union_reproduces_oracle_swap_remove_order_across_horizon()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 5_001];
        let encoded =
            EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5_001, 5_001)?;
        let clauses = vec![
            ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0e8)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, -1.0e8)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0)?),
        ];
        let mut query = ReferenceScorer::boolean(clauses)?;
        assert_eq!(query.doc(), Some(0));
        query.next()?;
        assert_eq!(query.doc(), Some(5_000));

        let small = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0);
        let positive = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0e8);
        let negative = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, -1.0e8);
        // The doc-0 scorer exhausts in the first window and `swap_remove`
        // moves the final small scorer ahead of the two large scorers.
        let expected = ((0.0_f32 + small) + positive) + negative;
        let parse_order = ((0.0_f32 + positive) + negative) + small;
        assert_ne!(expected.to_bits(), parse_order.to_bits());
        assert_eq!(query.score()?.to_bits(), expected.to_bits());
        Ok(())
    }

    #[test]
    fn buffered_union_ranked_topdocs_tracks_term_union_order_across_horizon()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 5_001];
        let encoded =
            EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5_001, 5_001)?;
        let clauses = vec![
            ScorerClause::should(term(postings(&[1, 5_000]), field, &snapshot, 2, 1, 1.0e8)?),
            ScorerClause::should(term(postings(&[3, 5_000]), field, &snapshot, 2, 1, -1.0e8)?),
            ScorerClause::should(term(postings(&[0, 5_000]), field, &snapshot, 2, 1, 1.0)?),
            ScorerClause::should(term(postings(&[2, 5_000]), field, &snapshot, 2, 1, 1.0)?),
        ];
        let mut query = ReferenceScorer::boolean_topdocs(clauses)?;
        assert_eq!(query.doc(), Some(0));
        while query.doc().is_some_and(|doc| doc < 5_000) {
            query.next()?;
        }
        assert_eq!(query.doc(), Some(5_000));

        let small = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        let positive = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0e8);
        let negative = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, -1.0e8);
        // The current-document traversal reaches the target in B, C, A, D
        // order. A scorer-major buffered union would retain A, B, C, D.
        let tantivy_term_union = (((0.0_f32 + negative) + small) + positive) + small;
        let scorer_major = (((0.0_f32 + positive) + negative) + small) + small;
        assert_ne!(tantivy_term_union.to_bits(), scorer_major.to_bits());
        assert_eq!(query.score()?.to_bits(), tantivy_term_union.to_bits());
        Ok(())
    }

    #[test]
    fn composite_size_hints_and_costs_match_tantivy_runtime_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 10_000];
        let encoded =
            EncodedDocLenSection::encode(0, 10_000, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 10_000, 10_000)?;
        let first_docs = (0_u32..500).collect::<Vec<_>>();
        let second_docs = (0_u32..1_000).collect::<Vec<_>>();
        let inner = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&first_docs),
                field,
                &snapshot,
                500,
                500,
                1.0,
            )?),
            ScorerClause::must(term(
                postings(&second_docs),
                field,
                &snapshot,
                1_000,
                1_000,
                1.0,
            )?),
        ])?;
        assert_eq!(inner.size_hint(), 60);
        assert_eq!(inner.cost(), 500);

        let nested_exclude = ReferenceScorer::boolean(vec![
            ScorerClause::must(inner),
            ScorerClause::must_not(term(postings(&[9_999]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(nested_exclude.doc(), Some(0));
        assert_eq!(nested_exclude.size_hint(), 60);
        assert_eq!(nested_exclude.cost(), 60);

        let third_docs = (0_u32..100).collect::<Vec<_>>();
        let outer = ReferenceScorer::boolean(vec![
            ScorerClause::must(nested_exclude),
            ScorerClause::must(term(
                postings(&third_docs),
                field,
                &snapshot,
                100,
                100,
                1.0,
            )?),
        ])?;
        assert_eq!(outer.cost(), 60);

        let mut union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[0, 4_096]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert_eq!(union.doc(), Some(0));
        assert_eq!(union.size_hint(), 2);
        assert_eq!(union.cost(), 2);
        assert_eq!(union.next()?, Some(4_096));
        assert_eq!(union.size_hint(), 0);
        assert_eq!(union.cost(), 0);

        // BooleanWeight passes SegmentReader::num_docs(), not physical BM25 N.
        assert_eq!(estimate_intersection([25, 25].into_iter(), 50), 15);
        assert_eq!(estimate_union([25, 25].into_iter(), 50), 32);
        assert_ne!(estimate_intersection([25, 25].into_iter(), 100), 15);
        assert_ne!(estimate_union([25, 25].into_iter(), 100), 32);
        Ok(())
    }

    #[test]
    fn max_global_docid_survives_union_and_intersection_exhaustion()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded = EncodedDocLenSection::encode(
            u64::from(u32::MAX),
            1_u64 << 32,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 1, 1)?;

        let mut union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(union.doc(), Some(u32::MAX));
        assert!(union.score()?.is_finite());
        assert_eq!(union.next()?, None);
        assert_eq!(union.next()?, None);

        let mut intersection = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(intersection.doc(), Some(u32::MAX));
        assert!(intersection.score()?.is_finite());
        assert_eq!(intersection.next()?, None);
        assert_eq!(intersection.next()?, None);
        Ok(())
    }

    #[test]
    fn randomized_boolean_matches_naive_set_oracle() -> Result<(), Box<dyn std::error::Error>> {
        const DOCS: usize = 48;
        let lengths = [Some(1); DOCS];
        let encoded = EncodedDocLenSection::encode(
            0,
            DOCS as u64,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, DOCS as u64, DOCS as u64)?;

        for seed in 1_u64..=24 {
            let mut state = seed;
            let mut membership = [[false; DOCS]; 5];
            for doc in 0..DOCS {
                for term_membership in &mut membership {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    term_membership[doc] = state >> 61 != 0;
                }
            }
            let term_docs = |term_index: usize| {
                membership[term_index]
                    .iter()
                    .enumerate()
                    .filter_map(|(doc, present)| {
                        present.then_some(u32::try_from(doc).expect("DOCS fits in u32"))
                    })
                    .collect::<Vec<_>>()
            };
            let docs_by_term = std::array::from_fn::<_, 5, _>(term_docs);
            let leaf = |term_index: usize| {
                term(
                    postings(&docs_by_term[term_index]),
                    field,
                    &snapshot,
                    docs_by_term[term_index].len() as u64,
                    docs_by_term[term_index].len() as u64,
                    1.0,
                )
            };
            let scored_query = || {
                let either = ReferenceScorer::boolean(vec![
                    ScorerClause::should(leaf(0)?),
                    ScorerClause::should(leaf(1)?),
                ])?;
                ReferenceScorer::boolean(vec![
                    ScorerClause::must(either),
                    ScorerClause::must(leaf(2)?),
                    ScorerClause::must_not(leaf(3)?),
                    ScorerClause::should(leaf(4)?),
                ])
            };
            let unscored_query = || {
                let either = ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::should(leaf(0)?),
                    ScorerClause::should(leaf(1)?),
                ])?;
                ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::must(either),
                    ScorerClause::must(leaf(2)?),
                    ScorerClause::must_not(leaf(3)?),
                    ScorerClause::should(leaf(4)?),
                ])
            };
            let mut query = scored_query()?;
            let ranked = query.top_k(DOCS, &AllLiveDocs)?;
            let mut actual = ranked
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>();
            actual.sort_unstable();
            let expected = (0..DOCS)
                .filter(|&doc| {
                    (membership[0][doc] || membership[1][doc])
                        && membership[2][doc]
                        && !membership[3][doc]
                })
                .map(|doc| u32::try_from(doc).expect("DOCS fits in u32"))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "seed {seed}");

            let offset = usize::try_from(seed % 11).expect("bounded seed offset");
            let limit = 7;
            let mut counted_query = scored_query()?;
            let mut counted_collector = TopDocsCollector::with_exact_count(limit, offset)?;
            counted_collector.collect(&mut counted_query, &AllLiveDocs)?;
            let counted = counted_collector.finish()?;
            let page_start = offset.min(ranked.len());
            let page_end = page_start.saturating_add(limit).min(ranked.len());
            assert_eq!(counted.hits, ranked[page_start..page_end], "seed {seed}");
            assert_eq!(
                counted.total_count,
                Some(expected.len() as u64),
                "seed {seed}"
            );

            let mut unscored = unscored_query()?;
            assert_eq!(
                unscored.collect_doc_set(&AllLiveDocs)?,
                expected,
                "seed {seed}"
            );
        }
        Ok(())
    }

    #[test]
    fn e410_seeded_query_algebra_and_pagination_metamorphics_hold()
    -> Result<(), Box<dyn std::error::Error>> {
        const DOCS: usize = 48;
        const PAGE: usize = 7;
        let lengths = [Some(1); DOCS];
        let encoded = EncodedDocLenSection::encode(
            0,
            DOCS as u64,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, DOCS as u64, DOCS as u64)?;

        for seed in 1_u64..=32 {
            let mut state = seed;
            let mut membership = [[false; DOCS]; 3];
            for doc in 0..DOCS {
                for term_membership in &mut membership {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    term_membership[doc] = state >> 63 != 0;
                }
            }
            let docs_by_term = std::array::from_fn::<_, 3, _>(|term_index| {
                membership[term_index]
                    .iter()
                    .enumerate()
                    .filter_map(|(doc, present)| {
                        present.then_some(u32::try_from(doc).expect("DOCS fits u32"))
                    })
                    .collect::<Vec<_>>()
            });
            let leaf = |term_index: usize| {
                term(
                    postings(&docs_by_term[term_index]),
                    field,
                    &snapshot,
                    docs_by_term[term_index].len() as u64,
                    docs_by_term[term_index].len() as u64,
                    1.0,
                )
            };
            let should_set = |order: [usize; 3]| {
                let mut scorer = ReferenceScorer::boolean_unscored(
                    order
                        .into_iter()
                        .map(|index| leaf(index).map(ScorerClause::should))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                scorer.collect_doc_set(&AllLiveDocs)
            };
            let expected_union = (0..DOCS)
                .filter(|&doc| membership.iter().any(|term| term[doc]))
                .map(|doc| u32::try_from(doc).expect("DOCS fits u32"))
                .collect::<Vec<_>>();
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                assert_eq!(
                    should_set(order)?,
                    expected_union,
                    "Should result-set commutativity failed for seed {seed} order={order:?}",
                );
            }

            let mut left_associated = ReferenceScorer::boolean_unscored(vec![
                ScorerClause::must(ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::must(leaf(0)?),
                    ScorerClause::must(leaf(1)?),
                ])?),
                ScorerClause::must(leaf(2)?),
            ])?;
            let mut right_associated = ReferenceScorer::boolean_unscored(vec![
                ScorerClause::must(leaf(0)?),
                ScorerClause::must(ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::must(leaf(1)?),
                    ScorerClause::must(leaf(2)?),
                ])?),
            ])?;
            let expected_intersection = (0..DOCS)
                .filter(|&doc| membership.iter().all(|term| term[doc]))
                .map(|doc| u32::try_from(doc).expect("DOCS fits u32"))
                .collect::<Vec<_>>();
            assert_eq!(
                left_associated.collect_doc_set(&AllLiveDocs)?,
                expected_intersection,
                "left-associated Must drifted from the set oracle for seed {seed}",
            );
            assert_eq!(
                right_associated.collect_doc_set(&AllLiveDocs)?,
                expected_intersection,
                "right-associated Must drifted from the set oracle for seed {seed}",
            );

            let ranked_should = || {
                ReferenceScorer::boolean(vec![
                    ScorerClause::should(leaf(0)?),
                    ScorerClause::should(leaf(1)?),
                    ScorerClause::should(leaf(2)?),
                ])
            };
            let mut full_scorer = ranked_should()?;
            let full = full_scorer.top_k(DOCS, &AllLiveDocs)?;
            let mut scored_docids = full.iter().map(|hit| hit.global_docid).collect::<Vec<_>>();
            scored_docids.sort_unstable();
            assert_eq!(
                scored_docids, expected_union,
                "scored Should lowering drifted from the set oracle for seed {seed}",
            );
            let mut concatenated = Vec::new();
            for offset in (0..DOCS).step_by(PAGE) {
                let mut page_scorer = ranked_should()?;
                let mut collector = TopDocsCollector::new(PAGE, offset)?;
                collector.collect(&mut page_scorer, &AllLiveDocs)?;
                concatenated.extend(collector.finish()?.hits);
            }
            assert_eq!(
                concatenated, full,
                "page concatenation drifted from the full ranking for seed {seed}",
            );
        }
        Ok(())
    }

    #[test]
    fn global_collectors_page_count_and_materialize_only_winners()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 10];
        let encoded =
            EncodedDocLenSection::encode(0, 10, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 10, 10)?;
        let left_docs = [0, 2, 4, 6, 8];
        let right_docs = [1, 3, 5, 7, 9];

        let collect = |limit: usize,
                       offset: usize,
                       exact_count: bool,
                       live_docs: &dyn LiveDocs|
         -> Result<CollectedTopDocs, ArgusError> {
            let mut collector = if exact_count {
                TopDocsCollector::with_exact_count(limit, offset)?
            } else {
                TopDocsCollector::new(limit, offset)?
            };
            for docs in [&left_docs[..], &right_docs[..]] {
                let mut scorer = term(postings(docs), field, &snapshot, 10, 5, 1.0)?;
                collector.collect(&mut scorer, live_docs)?;
            }
            collector.finish()
        };

        let full = collect(10, 0, false, &AllLiveDocs)?;
        assert_eq!(
            full.hits
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
        assert_eq!(full.total_count, None);

        let page = collect(3, 4, false, &AllLiveDocs)?;
        assert_eq!(page.hits, full.hits[4..7]);
        let past_end = collect(3, 20, false, &AllLiveDocs)?;
        assert!(past_end.hits.is_empty());

        let live = |doc| doc % 3 != 0;
        let live_full = collect(10, 0, false, &live)?;
        let counted = collect(2, 1, true, &live)?;
        assert_eq!(counted.total_count, Some(6));
        assert_eq!(counted.hits, live_full.hits[1..3]);
        let count_only = collect(0, usize::MAX, true, &live)?;
        assert!(count_only.hits.is_empty());
        assert_eq!(count_only.total_count, Some(6));

        let mut unconsumed = term(postings(&left_docs), field, &snapshot, 10, 5, 1.0)?;
        let mut empty_page = TopDocsCollector::new(0, usize::MAX)?;
        empty_page.collect(&mut unconsumed, &AllLiveDocs)?;
        assert_eq!(unconsumed.doc(), Some(0));
        assert!(empty_page.finish()?.hits.is_empty());
        assert!(matches!(
            TopDocsCollector::new(1, usize::MAX),
            Err(ArgusError::CollectorWindowOverflow {
                limit: 1,
                offset: usize::MAX
            })
        ));

        let mut materialized = Vec::new();
        let hits = materialize_doc_ids(&page.hits, |global_docid| {
            materialized.push(global_docid);
            Ok(format!("winner-{global_docid}").into())
        })?;
        assert_eq!(materialized, vec![4, 5, 6]);
        assert_eq!(hits.len(), page.hits.len());
        Ok(())
    }

    #[test]
    fn topdocs_merge_reproduces_single_collector_under_any_partition_and_order()
    -> Result<(), ArgusError> {
        // Deterministic scored stream with heavy ties: seven score buckets
        // over 257 docids so cutoff-boundary ties cross every partition.
        let mut state = 0x5eed_cafe_0000_0001_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let entries: Vec<(u32, f32)> = (0..257_u32)
            .map(|docid| {
                #[allow(clippy::cast_precision_loss)]
                let bucket = (next() % 7) as f32;
                (docid, 0.25_f32.mul_add(bucket, 1.0))
            })
            .collect();

        let build = |limit: usize, offset: usize, exact: bool| {
            if exact {
                TopDocsCollector::with_exact_count(limit, offset)
            } else {
                TopDocsCollector::new(limit, offset)
            }
        };
        let page_key = |collected: CollectedTopDocs| {
            (
                collected
                    .hits
                    .iter()
                    .map(|hit| (hit.global_docid, hit.score.to_bits()))
                    .collect::<Vec<_>>(),
                collected.total_count,
            )
        };

        for (limit, offset, exact) in [
            (10_usize, 0_usize, false),
            (10, 0, true),
            (5, 7, true),
            (3, 1, false),
            (0, 0, true),
        ] {
            let mut baseline = build(limit, offset, exact)?;
            for &(docid, score) in &entries {
                baseline.record_live(docid, Some(score))?;
            }
            let expected = page_key(baseline.finish()?);

            for partitions in [1_usize, 4, 7] {
                for rotation in 0..partitions {
                    let mut partials = Vec::with_capacity(partitions);
                    for _ in 0..partitions {
                        partials.push(build(limit, offset, exact)?);
                    }
                    for (position, &(docid, score)) in entries.iter().enumerate() {
                        partials[position % partitions].record_live(docid, Some(score))?;
                    }
                    let mut fold = build(limit, offset, exact)?;
                    // Merge in a rotated order: the retained set must be
                    // identical no matter which partial folds first.
                    for step in 0..partitions {
                        let partial = partials[(step + rotation) % partitions].empty_like()?;
                        let taken = std::mem::replace(
                            &mut partials[(step + rotation) % partitions],
                            partial,
                        );
                        fold.merge(taken)?;
                    }
                    assert_eq!(
                        page_key(fold.finish()?),
                        expected,
                        "merge diverged: limit={limit} offset={offset} exact={exact} \
                         partitions={partitions} rotation={rotation}",
                    );
                }
            }
        }

        // Shape mismatches fail closed instead of silently mixing windows.
        let mut ten = TopDocsCollector::new(10, 0)?;
        assert!(ten.merge(TopDocsCollector::new(5, 0)?).is_err());
        assert!(ten.merge(TopDocsCollector::new(10, 1)?).is_err());
        assert!(
            ten.merge(TopDocsCollector::with_exact_count(10, 0)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn doc_set_uses_recursive_lazy_unscored_unions() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        #[derive(Clone)]
        struct CountingFieldNorm<'a> {
            field: DocLenField<'a>,
            reads: Arc<AtomicUsize>,
        }

        impl FieldNormReader for CountingFieldNorm<'_> {
            fn field_ord(&self) -> u16 {
                self.field.field_ord()
            }

            fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
                self.reads.fetch_add(1, AtomicOrdering::SeqCst);
                self.field.fieldnorm_id(u64::from(global_docid))
            }
        }

        let lengths = vec![Some(1); 5];
        let encoded =
            EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5, 5)?;
        let reads = Arc::new(AtomicUsize::new(0));
        let leaf = |docs: &[u32]| {
            Ok::<_, ArgusError>(ReferenceScorer::term(TermScorer::new(
                VecCursor::new(postings(docs), docs.len() as u64, 5),
                CountingFieldNorm {
                    field,
                    reads: Arc::clone(&reads),
                },
                snapshot.clone(),
                docs.len() as u64,
                TermRecordOption::WithFreqs,
                1.0,
            )?))
        };
        let union = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::should(leaf(&[0, 2, 4])?),
            ScorerClause::should(leaf(&[1, 2, 3])?),
        ])?;
        let excluded = leaf(&[3])?;
        let mut query = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(union),
            ScorerClause::must_not(excluded),
        ])?;
        assert_eq!(query.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2, 4]);
        assert_eq!(
            reads.load(AtomicOrdering::SeqCst),
            7,
            "each physical posting is fieldnorm-validated once; none is scored"
        );

        let mut scored_union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 2]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert!(matches!(
            scored_union.collect_doc_set(&AllLiveDocs),
            Err(ArgusError::ScoredTreeForUnscoredCollector)
        ));
        Ok(())
    }

    #[test]
    fn numeric_range_leaf_has_constant_score_and_fused_cursor_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        let entries = [
            NumericEntry::i64(-5, 100),
            NumericEntry::i64(0, 101),
            NumericEntry::i64(5, 102),
            NumericEntry::i64(5, 103),
            NumericEntry::i64(10, 104),
            NumericEntry::i64(20, 106),
        ];
        let fields = [NumericFieldInput::new(0, &entries)];
        let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 100, 108, &fields)?;
        let field = encoded
            .section()?
            .field(0)
            .expect("created_at NUMERIC field");

        let mut range = ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(5)),
            Bound::Excluded(NumericValue::I64(10)),
            8,
        )?;
        assert_eq!(range.doc(), Some(102));
        assert_eq!(range.cost(), 6, "Tantivy RangeDocSet uses floor(0.8N)");
        assert_eq!(range.size_hint(), 0, "Tantivy range hint is floor(N/10)");
        assert_eq!(range.score()?.to_bits(), 1.0_f32.to_bits());
        assert_eq!(range.advance(103)?, Some(103));
        assert_eq!(range.score()?.to_bits(), 1.0_f32.to_bits());
        assert_eq!(range.next()?, None);
        assert_eq!(range.next()?, None, "exhaustion is fused");
        assert!(matches!(
            range.score(),
            Err(ArgusError::CursorInvariant(
                "cannot score an exhausted numeric range"
            ))
        ));

        let mut boosted = ReferenceScorer::numeric_range_with_boost(
            field,
            Bound::Included(NumericValue::I64(5)),
            Bound::Excluded(NumericValue::I64(10)),
            8,
            2.5,
        )?;
        assert_eq!(boosted.score()?.to_bits(), 2.5_f32.to_bits());

        let inner = ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(5)),
            Bound::Excluded(NumericValue::I64(10)),
            8,
        )?;
        let mut replaced = ReferenceScorer::constant_score(inner, 3.25)?;
        assert_eq!(replaced.score()?.to_bits(), 3.25_f32.to_bits());
        assert!(matches!(
            ReferenceScorer::constant_score(replaced, f32::NAN),
            Err(ArgusError::InvalidConstantBoost { boost_bits })
                if boost_bits == f32::NAN.to_bits()
        ));

        let overlap = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::should(ReferenceScorer::numeric_range(
                field,
                Bound::Unbounded,
                Bound::Included(NumericValue::I64(5)),
                8,
            )?),
            ScorerClause::should(ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::I64(5)),
                Bound::Included(NumericValue::I64(10)),
                8,
            )?),
        ])?;
        let mut deduplicated = ReferenceScorer::constant_score(overlap, 4.0)?;
        assert_eq!(deduplicated.cost(), 5, "overlapping matches count once");
        assert_eq!(deduplicated.size_hint(), 5);
        assert_eq!(deduplicated.score()?.to_bits(), 4.0_f32.to_bits());
        assert_eq!(
            deduplicated.collect_doc_set(&AllLiveDocs)?,
            vec![100, 101, 102, 103, 104]
        );

        let high = ReferenceScorer::constant_score(
            ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::I64(0)),
                Bound::Included(NumericValue::I64(0)),
                8,
            )?,
            1.0e8,
        )?;
        let low = ReferenceScorer::constant_score(
            ReferenceScorer::numeric_range(
                field,
                Bound::Unbounded,
                Bound::Included(NumericValue::I64(5)),
                8,
            )?,
            -1.0e8,
        )?;
        let unit = ReferenceScorer::constant_score(
            ReferenceScorer::numeric_range(
                field,
                Bound::Unbounded,
                Bound::Included(NumericValue::I64(10)),
                8,
            )?,
            1.0,
        )?;
        let mut cancellation = ReferenceScorer::boolean(vec![
            ScorerClause::must(unit),
            ScorerClause::must(low),
            ScorerClause::must(high),
        ])?;
        assert_eq!(cancellation.doc(), Some(101));
        assert_eq!(
            cancellation.score()?.to_bits(),
            1.0_f32.to_bits(),
            "unique-cardinality costs pin Tantivy's stable Must score order"
        );

        let mut unscored = ReferenceScorer::numeric_range(
            field,
            Bound::Unbounded,
            Bound::Included(NumericValue::I64(5)),
            8,
        )?;
        assert_eq!(
            unscored.collect_doc_set(&AllLiveDocs)?,
            vec![100, 101, 102, 103]
        );

        let empty = ReferenceScorer::numeric_range(
            field,
            Bound::Included(NumericValue::I64(30)),
            Bound::Included(NumericValue::I64(40)),
            8,
        )?;
        assert_eq!(empty.doc(), None);
        assert!(matches!(
            ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::U64(0)),
                Bound::Unbounded,
                8
            ),
            Err(ArgusError::Numeric(NumericCodecError::BoundTypeMismatch {
                field_ord: 0,
                ..
            }))
        ));
        Ok(())
    }

    #[test]
    fn numeric_range_preserves_oracle_cost_order_and_u32_max()
    -> Result<(), Box<dyn std::error::Error>> {
        let sparse_entries = [NumericEntry::i64(7, 0)];
        let sparse_fields = [NumericFieldInput::new(0, &sparse_entries)];
        let numeric = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 10, &sparse_fields)?;
        let numeric_field = numeric
            .section()?
            .field(0)
            .expect("created_at NUMERIC field");
        let range = ReferenceScorer::numeric_range(
            numeric_field,
            Bound::Included(NumericValue::I64(7)),
            Bound::Included(NumericValue::I64(7)),
            10,
        )?;
        assert_eq!(
            range.cost(),
            8,
            "one exact match still has oracle range cost"
        );

        let lengths = [Some(1); 10];
        let encoded_lengths =
            EncodedDocLenSection::encode(0, 10, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let length_section = encoded_lengths.section(&[1])?;
        let fieldnorms = length_section.field(1).expect("fieldnorm column");
        let stats = snapshot(1, 10, 10)?;
        let positive = term(postings(&[0]), fieldnorms, &stats, 1, 1, 1.0e8)?;
        let negative = term(postings(&[0]), fieldnorms, &stats, 1, 2, -1.0e8)?;
        let mut intersection = ReferenceScorer::boolean(vec![
            ScorerClause::must(range),
            ScorerClause::must(positive),
            ScorerClause::must(negative),
        ])?;
        assert_eq!(intersection.doc(), Some(0));
        assert_eq!(intersection.cost(), 1);
        assert_eq!(
            intersection.score()?.to_bits(),
            1.0_f32.to_bits(),
            "terms cancel before the range's constant score is accumulated"
        );

        let high_entries = [
            NumericEntry::i64(0, u32::MAX - 1),
            NumericEntry::i64(1, u32::MAX),
        ];
        let high_fields = [NumericFieldInput::new(0, &high_entries)];
        let high_numeric = EncodedNumericSection::encode(
            RANGE_TEST_SCHEMA,
            u64::from(u32::MAX) - 1,
            u64::from(u32::MAX) + 1,
            &high_fields,
        )?;
        let high_field = high_numeric
            .section()?
            .field(0)
            .expect("high created_at NUMERIC field");
        let mut high =
            ReferenceScorer::numeric_range(high_field, Bound::Unbounded, Bound::Unbounded, 2)?;
        assert_eq!(
            high.cost(),
            2,
            "full coverage lowers like Tantivy AllScorer"
        );
        assert_eq!(high.size_hint(), 2);
        assert_eq!(high.advance(u32::MAX)?, Some(u32::MAX));
        assert_eq!(high.score()?.to_bits(), 1.0_f32.to_bits());
        assert_eq!(high.next()?, None);
        Ok(())
    }

    #[test]
    fn numeric_ranges_preserve_boolean_multiplicity_exclusion_and_live_filtering()
    -> Result<(), Box<dyn std::error::Error>> {
        let entries = [
            NumericEntry::i64(0, 0),
            NumericEntry::i64(1, 1),
            NumericEntry::i64(2, 2),
        ];
        let fields = [NumericFieldInput::new(0, &entries)];
        let encoded = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 3, &fields)?;
        let field = encoded
            .section()?
            .field(0)
            .expect("created_at NUMERIC field");
        let left = || {
            ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::I64(0)),
                Bound::Included(NumericValue::I64(1)),
                3,
            )
        };
        let right = || {
            ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::I64(1)),
                Bound::Included(NumericValue::I64(2)),
                3,
            )
        };
        let all = || ReferenceScorer::numeric_range(field, Bound::Unbounded, Bound::Unbounded, 3);

        let mut required_full_range = ReferenceScorer::boolean(vec![
            ScorerClause::must(all()?),
            ScorerClause::must(left()?),
        ])?;
        let mut required_full_range_hits = required_full_range.top_k(3, &AllLiveDocs)?;
        required_full_range_hits.sort_unstable_by_key(|hit| hit.global_docid);
        assert_eq!(
            required_full_range_hits
                .iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![(0, 1.0_f32.to_bits()), (1, 1.0_f32.to_bits())],
            "Tantivy removes a full-cardinality unit range after it lowers to raw AllScorer"
        );

        let mut required_boosted_full_range = ReferenceScorer::boolean(vec![
            ScorerClause::must(ReferenceScorer::numeric_range_with_boost(
                field,
                Bound::Unbounded,
                Bound::Unbounded,
                3,
                2.5,
            )?),
            ScorerClause::must(left()?),
        ])?;
        let mut required_boosted_hits = required_boosted_full_range.top_k(3, &AllLiveDocs)?;
        required_boosted_hits.sort_unstable_by_key(|hit| hit.global_docid);
        assert_eq!(
            required_boosted_hits
                .iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![(0, 3.5_f32.to_bits()), (1, 3.5_f32.to_bits())],
            "a boosted full-cardinality range remains score-bearing"
        );

        let mut should = ReferenceScorer::boolean(vec![
            ScorerClause::should(left()?),
            ScorerClause::should(right()?),
        ])?;
        let mut hits = should.top_k(3, &AllLiveDocs)?;
        hits.sort_unstable_by_key(|hit| hit.global_docid);
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (0, 1.0_f32.to_bits()),
                (1, 2.0_f32.to_bits()),
                (2, 1.0_f32.to_bits()),
            ],
            "overlapping constant-score ranges retain one contribution per child"
        );

        let mut excluded = ReferenceScorer::boolean(vec![
            ScorerClause::must(all()?),
            ScorerClause::must_not(ReferenceScorer::numeric_range(
                field,
                Bound::Included(NumericValue::I64(1)),
                Bound::Included(NumericValue::I64(1)),
                3,
            )?),
        ])?;
        let excluded_hits = excluded.top_k(3, &AllLiveDocs)?;
        assert_eq!(
            excluded_hits
                .iter()
                .map(|hit| (hit.global_docid, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![(0, 1.0_f32.to_bits()), (2, 1.0_f32.to_bits())],
            "MustNot gates matches without adding a score"
        );

        let mut unscored = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::should(left()?),
            ScorerClause::should(right()?),
        ])?;
        assert_eq!(unscored.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2]);
        let mut live_filtered = all()?;
        assert_eq!(
            live_filtered
                .top_k(3, &|docid| docid != 1)?
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            vec![0, 2],
            "tombstone filtering happens after the physical range cursor"
        );

        let repeated_doc = [NumericEntry::i64(0, 0), NumericEntry::i64(1, 0)];
        let repeated_fields = [NumericFieldInput::new(0, &repeated_doc)];
        let repeated = EncodedNumericSection::encode(RANGE_TEST_SCHEMA, 0, 2, &repeated_fields)?;
        let repeated_field = repeated
            .section()?
            .field(0)
            .expect("multi-valued NUMERIC fixture");
        let ordinary =
            ReferenceScorer::numeric_range(repeated_field, Bound::Unbounded, Bound::Unbounded, 2)?;
        assert_eq!(ordinary.cost(), 1);
        assert_eq!(ordinary.size_hint(), 0);
        assert!(matches!(
            ReferenceScorer::numeric_range(repeated_field, Bound::Unbounded, Bound::Unbounded, 1),
            Err(ArgusError::InvalidNumericCardinality {
                field_ord: 0,
                value_count: 2,
                segment_num_docs: 1,
            })
        ));
        Ok(())
    }

    #[test]
    fn tombstones_keep_physical_stats_but_never_enter_heap_or_materializer()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2), Some(2)];
        let encoded =
            EncodedDocLenSection::encode(0, 2, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let pre_compaction = snapshot(1, 4, 2)?;
        let mut query = term(postings(&[0, 1]), field, &pre_compaction, 2, 2, 1.0)?;
        let live = |doc| doc == 1;
        let winners = query.top_k(10, &live)?;
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].global_docid, 1);
        let expected = expected_term_score(&pre_compaction, 2, fieldnorm_to_id(2), 1, 1.0);
        assert_eq!(winners[0].score.to_bits(), expected.to_bits());

        let mut calls = Vec::new();
        let hits = materialize_doc_ids(&winners, |doc| {
            calls.push(doc);
            Ok(format!("external-{doc}").into())
        })?;
        assert_eq!(calls, vec![1]);
        assert_eq!(hits[0].document_id, "external-1");

        let compacted = snapshot(1, 2, 1)?;
        let compacted_score = expected_term_score(&compacted, 1, fieldnorm_to_id(2), 1, 1.0);
        assert_ne!(expected.to_bits(), compacted_score.to_bits());
        Ok(())
    }

    #[test]
    fn deleted_matches_are_scored_before_live_filtering_to_preserve_union_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 20_001];
        let encoded =
            EncodedDocLenSection::encode(0, 20_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 20_001, 20_001)?;
        let required = term(postings(&[5_000, 10_000]), field, &snapshot, 2, 2, 1.0)?;
        let optional = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 9_500]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                -1.0e8,
            )?),
            ScorerClause::should(term(postings(&[0, 5_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                1.0e8,
            )?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                1.0,
            )?),
        ])?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::must(required),
            ScorerClause::should(optional),
        ])?;
        let winners = query.top_k(1, &|doc| doc == 10_000)?;
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].global_docid, 10_000);

        let required_score = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        let optional_score = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
        let expected = required_score + optional_score;
        assert_eq!(winners[0].score.to_bits(), expected.to_bits());
        assert_ne!(winners[0].score.to_bits(), required_score.to_bits());
        Ok(())
    }

    #[test]
    fn malformed_scoring_inputs_fail_without_reaching_idf_assertion()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let valid = snapshot(1, 1, 1)?;

        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                valid.clone(),
                2,
                TermRecordOption::WithFreqs,
                1.0
            ),
            Err(ArgusError::InvalidDocFrequency {
                doc_freq: 2,
                doc_count: 1
            })
        ));
        let empty_field = snapshot(1, 0, 1)?;
        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                empty_field,
                1,
                TermRecordOption::WithFreqs,
                1.0
            ),
            Err(ArgusError::InvalidSnapshot { .. })
        ));
        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                valid,
                1,
                TermRecordOption::WithFreqs,
                f32::NAN
            ),
            Err(ArgusError::InvalidBoost { .. })
        ));
        Ok(())
    }

    #[test]
    fn e410_zero_document_snapshot_has_no_average_or_tf_division() -> Result<(), ArgusError> {
        let empty = snapshot(1, 0, 0)?;
        assert_eq!(empty.doc_count(), 0);
        assert_eq!(empty.total_tokens(), 0);
        assert_eq!(empty.average_field_length(), None);
        assert!(empty.tf_cache.iter().all(|value| value.to_bits() == 0));
        assert!(matches!(
            snapshot(1, 1, 0),
            Err(ArgusError::InvalidSnapshot {
                field_ord: 1,
                reason: "an empty snapshot cannot contain field tokens",
            })
        ));
        Ok(())
    }

    /// Deliberately broken historical control: after draining the initial
    /// exhaustive buffer, attempt exactly one competitive refill and stop when
    /// that window contains no candidates. Production must keep refilling
    /// because a later horizon can still contain an essential-term winner.
    fn collect_one_competitive_refill_for_negative_control(
        union: &mut BufferedUnionScorer<'_>,
        collector: &mut TopDocsCollector,
    ) -> Result<(), ArgusError> {
        while let Some(doc) = union.doc() {
            collector.record_live(doc, Some(union.score()?))?;
            if union.advance_buffered().is_none() {
                break;
            }
        }
        let cutoff = collector.competitive_cutoff_score();
        if union.refill_with_cutoff(cutoff)? {
            assert!(
                union.advance_buffered().is_none(),
                "negative-control fixture requires an empty first competitive window",
            );
        }
        Ok(())
    }

    #[test]
    fn max_score_empty_competitive_window_refills_again_for_late_winner()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 9_001;
        const FIRST_HORIZON_END: u32 = 4_096;
        const TARGET_DOC: u32 = 9_000;
        let lengths = (0..NUM_DOCS)
            .map(|doc| {
                Some(if doc < FIRST_HORIZON_END {
                    2
                } else if doc == TARGET_DOC {
                    9
                } else {
                    1
                })
            })
            .collect::<Vec<_>>();
        let total_tokens = lengths
            .iter()
            .flatten()
            .map(|length| u64::from(*length))
            .sum::<u64>();
        assert_eq!(total_tokens, 13_105);
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, total_tokens, u64::from(NUM_DOCS))?;
        let rows_by_term = vec![
            (0..NUM_DOCS)
                .map(|doc| Posting::new(doc, 1))
                .collect::<Vec<_>>(),
            (0..FIRST_HORIZON_END)
                .map(|doc| Posting::new(doc, 1))
                .collect::<Vec<_>>(),
            vec![Posting::new(TARGET_DOC, 8)],
        ];
        let boosts = [1.0, 1.0, 1.0];
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| {
                EncodedPostingList::encode_with_block_max(rows, |doc| {
                    field.fieldnorm_id(u64::from(doc))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

        let mut broken = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut broken_collector = TopDocsCollector::new(1, 0)?;
        let ScorerNode::Union(broken_union) = &mut broken.node else {
            panic!("UNION_HORIZON fixture must lower to a buffered union");
        };
        collect_one_competitive_refill_for_negative_control(broken_union, &mut broken_collector)?;
        assert_eq!(broken_union.pruning_stats.max_score_windows, 1);
        assert_eq!(broken_union.pruning_stats.block_max_wand_windows, 0);
        assert_eq!(broken_union.pruning_stats.candidate_docs, 0);
        assert_eq!(broken_union.pruning_stats.blocks_skipped, 0);
        let broken_hits = broken_collector.finish()?.hits;
        assert_eq!(broken_hits[0].global_docid, 0);
        assert_ne!(broken_hits[0].global_docid, TARGET_DOC);

        let mut oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(1, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;

        let mut candidate = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(1, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;
        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, TARGET_DOC);
        assert_eq!(stats.max_score_windows, 2);
        assert_eq!(stats.block_max_wand_windows, 0);
        assert_eq!(stats.candidate_docs, 1);
        assert_eq!(stats.blocks_skipped, 0);
        Ok(())
    }

    #[test]
    fn maxscore_two_pass_preserves_swap_remove_score_bits() -> Result<(), Box<dyn std::error::Error>>
    {
        const NUM_DOCS: u32 = 5_001;
        let lengths = vec![Some(1); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
        let mut rows_by_term = vec![vec![Posting::new(0, 1)], vec![Posting::new(5_000, 1)]];
        rows_by_term.extend((0..6).map(|_| vec![Posting::new(5_000, 1)]));
        let boosts = [5.0e7, 1.0e8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let fieldnorm_id = fieldnorm_to_id(1);
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

        let mut oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(1, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;

        let mut candidate = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(1, 0)?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;
        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, 5_000);
        assert_eq!(stats.max_score_windows, 1);
        assert_eq!(stats.block_max_wand_windows, 0);
        assert_eq!(stats.candidate_docs, 1);
        assert_eq!(stats.blocks_skipped, 0);

        let small = expected_term_score(&snapshot, 1, fieldnorm_id, 1, 2.0);
        let huge = expected_term_score(&snapshot, 1, fieldnorm_id, 1, 1.0e8);
        // Every singleton scorer that exhausts at doc 5_000 is replaced in
        // place with the active vector's last scorer. That cascade scores all
        // six small terms before the huge term, rather than parse order's
        // huge-then-small sequence.
        let mut union_order = 0.0_f32;
        for _ in 0..6 {
            union_order += small;
        }
        union_order += huge;
        let mut parse_order = huge;
        for _ in 0..6 {
            parse_order += small;
        }
        assert_ne!(union_order.to_bits(), parse_order.to_bits());
        assert_eq!(candidate_hits[0].score.to_bits(), union_order.to_bits());
        Ok(())
    }

    #[test]
    fn block_max_wand_skips_low_blocks_and_exact_count_disables_it()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 8_192;
        let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
        let rows_by_term = (0..9)
            .map(|term_index| {
                (0..NUM_DOCS)
                    .map(|doc| {
                        let frequency = if term_index == 0 && doc == 1 {
                            32
                        } else if term_index == 0 && doc == 4_352 {
                            64
                        } else {
                            1
                        };
                        Posting::new(doc, frequency)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let boosts = [4.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let fieldnorm_id = fieldnorm_to_id(8);
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

        for limit in [1, 10, 100, 1_000] {
            let mut oracle = sealed_union(
                &posting_lists,
                None,
                field,
                &snapshot,
                &rows_by_term,
                &boosts,
                NUM_DOCS,
            )?;
            let mut oracle_collector = TopDocsCollector::new(limit, 0)?;
            oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
            let oracle_hits = oracle_collector.finish()?.hits;
            let mut candidate = sealed_union(
                &posting_lists,
                Some(&block_max),
                field,
                &snapshot,
                &rows_by_term,
                &boosts,
                NUM_DOCS,
            )?;
            let mut candidate_collector = TopDocsCollector::new(limit, 0)?;
            candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
            let stats = candidate
                .union_pruning_stats()
                .expect("top-level union retains pruning stats");
            let candidate_hits = candidate_collector.finish()?.hits;
            assert_hits_bit_exact(&candidate_hits, &oracle_hits);
            assert_eq!(candidate_hits[0].global_docid, 4_352);
            assert_eq!(stats.max_score_windows, 0);
            assert_eq!(stats.block_max_wand_windows, 1);
            if limit == 1 {
                assert!(stats.blocks_skipped > 0);
                assert!((1..UNION_HORIZON_U64).contains(&stats.candidate_docs));
            }
        }

        let live = |doc: u32| !doc.is_multiple_of(7);
        let mut live_oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut live_oracle_collector = TopDocsCollector::new(3, 2)?;
        live_oracle_collector.collect(&mut live_oracle, &live)?;
        let live_oracle_hits = live_oracle_collector.finish()?.hits;
        let mut live_candidate = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut live_candidate_collector = TopDocsCollector::new(3, 2)?;
        live_candidate_collector.collect(&mut live_candidate, &live)?;
        let live_stats = live_candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let live_candidate_hits = live_candidate_collector.finish()?.hits;
        assert_hits_bit_exact(&live_candidate_hits, &live_oracle_hits);
        assert!(live_stats.block_max_wand_windows > 0);

        let mut counted_oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::with_exact_count(3, 2)?;
        oracle_collector.collect(&mut counted_oracle, &live)?;
        let oracle_counted = oracle_collector.finish()?;
        let mut counted_candidate = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::with_exact_count(3, 2)?;
        candidate_collector.collect(&mut counted_candidate, &live)?;
        let counted_stats = counted_candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_counted = candidate_collector.finish()?;
        assert_hits_bit_exact(&candidate_counted.hits, &oracle_counted.hits);
        assert_eq!(candidate_counted.total_count, Some(7_021));
        assert_eq!(candidate_counted.total_count, oracle_counted.total_count);
        assert_eq!(counted_stats, UnionPruningStats::default());
        Ok(())
    }

    #[test]
    fn bmw_term_bound_only_cursor_falls_back_without_error()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 8_192;
        let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
        let rows = (0..NUM_DOCS)
            .map(|doc| Posting::new(doc, 1))
            .collect::<Vec<_>>();

        let mut oracle_clauses = Vec::new();
        let mut bounded_clauses = Vec::new();
        for _ in 0..BMW_MIN_CLAUSES {
            oracle_clauses.push(ScorerClause::should(ReferenceScorer::term(
                TermScorer::new(
                    VecCursor::new(rows.clone(), u64::from(NUM_DOCS), NUM_DOCS),
                    field,
                    snapshot.clone(),
                    u64::from(NUM_DOCS),
                    TermRecordOption::WithFreqs,
                    1.0,
                )?,
            )));
            bounded_clauses.push(ScorerClause::should(ReferenceScorer::term(
                TermScorer::new(
                    TermBoundOnlyCursor(VecCursor::new(
                        rows.clone(),
                        u64::from(NUM_DOCS),
                        NUM_DOCS,
                    )),
                    field,
                    snapshot.clone(),
                    u64::from(NUM_DOCS),
                    TermRecordOption::WithFreqs,
                    1.0,
                )?,
            )));
        }
        let mut oracle = ReferenceScorer::boolean(oracle_clauses)?;
        let mut oracle_collector = TopDocsCollector::new(10, 0)?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;
        let mut bounded = ReferenceScorer::boolean(bounded_clauses)?;
        let mut bounded_collector = TopDocsCollector::new(10, 0)?;
        bounded_collector.collect(&mut bounded, &AllLiveDocs)?;
        let stats = bounded
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let bounded_hits = bounded_collector.finish()?.hits;
        assert_hits_bit_exact(&bounded_hits, &oracle_hits);
        assert_eq!(stats, UnionPruningStats::default());
        Ok(())
    }

    #[test]
    fn maxscore_equal_cutoff_keeps_better_docid_and_negative_weight_falls_back()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 5_001;
        let lengths = vec![Some(1); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS), u64::from(NUM_DOCS))?;
        let rows_by_term = vec![
            vec![Posting::new(0, 1), Posting::new(5_000, 16)],
            vec![Posting::new(0, 1), Posting::new(5_000, 16)],
        ];
        let boosts = [1.0, 1.0];
        let fieldnorm_id = fieldnorm_to_id(1);
        let encoded_terms = rows_by_term
            .iter()
            .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
            .collect::<Result<Vec<_>, _>>()?;
        let posting_lists = encoded_terms
            .iter()
            .map(|(postings, _)| postings.posting_list())
            .collect::<Result<Vec<_>, _>>()?;
        let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;
        let term_score = expected_term_score(&snapshot, 2, fieldnorm_id, 16, 1.0);
        let cutoff = term_score + term_score;

        let mut oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(1, 0)?;
        oracle_collector.record_live(10_000, Some(cutoff))?;
        oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;
        let mut candidate = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &boosts,
            NUM_DOCS,
        )?;
        let mut candidate_collector = TopDocsCollector::new(1, 0)?;
        candidate_collector.record_live(10_000, Some(cutoff))?;
        candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
        let stats = candidate
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let candidate_hits = candidate_collector.finish()?.hits;
        assert_hits_bit_exact(&candidate_hits, &oracle_hits);
        assert_eq!(candidate_hits[0].global_docid, 5_000);
        assert_eq!(candidate_hits[0].score.to_bits(), cutoff.to_bits());
        assert_eq!(stats.max_score_windows, 1);
        assert!(stats.candidate_docs >= 1);

        let negative_boosts = [1.0, -0.25];
        let mut fallback_oracle = sealed_union(
            &posting_lists,
            None,
            field,
            &snapshot,
            &rows_by_term,
            &negative_boosts,
            NUM_DOCS,
        )?;
        let mut oracle_collector = TopDocsCollector::new(1, 0)?;
        oracle_collector.collect(&mut fallback_oracle, &AllLiveDocs)?;
        let oracle_hits = oracle_collector.finish()?.hits;
        let mut fallback = sealed_union(
            &posting_lists,
            Some(&block_max),
            field,
            &snapshot,
            &rows_by_term,
            &negative_boosts,
            NUM_DOCS,
        )?;
        let mut fallback_collector = TopDocsCollector::new(1, 0)?;
        fallback_collector.collect(&mut fallback, &AllLiveDocs)?;
        let fallback_stats = fallback
            .union_pruning_stats()
            .expect("top-level union retains pruning stats");
        let fallback_hits = fallback_collector.finish()?.hits;
        assert_hits_bit_exact(&fallback_hits, &oracle_hits);
        assert_eq!(fallback_stats, UnionPruningStats::default());
        Ok(())
    }

    #[test]
    fn randomized_maxscore_matches_exhaustive_for_pinned_k_matrix()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 12_288;
        let lengths = vec![Some(8); usize::try_from(NUM_DOCS).expect("fixture count fits usize")];
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, u64::from(NUM_DOCS) * 8, u64::from(NUM_DOCS))?;
        let fieldnorm_id = fieldnorm_to_id(8);
        let boosts = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

        for seed in 1_u64..=3 {
            let mut state = seed;
            let mut rows_by_term = (0..8).map(|_| Vec::new()).collect::<Vec<_>>();
            for doc in 0..NUM_DOCS {
                for rows in &mut rows_by_term {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    if state >> 62 != 0 {
                        let frequency = 1 + u32::try_from((state >> 54) & 15)
                            .expect("randomized frequency nibble fits u32");
                        rows.push(Posting::new(doc, frequency));
                    }
                }
            }
            let encoded_terms = rows_by_term
                .iter()
                .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
                .collect::<Result<Vec<_>, _>>()?;
            let posting_lists = encoded_terms
                .iter()
                .map(|(postings, _)| postings.posting_list())
                .collect::<Result<Vec<_>, _>>()?;
            let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

            for record_option in [TermRecordOption::WithFreqs, TermRecordOption::Basic] {
                for k in [1, 10, 100, 1_000] {
                    let mut oracle = sealed_union_with_record_option(
                        &posting_lists,
                        None,
                        field,
                        &snapshot,
                        &rows_by_term,
                        &boosts,
                        NUM_DOCS,
                        record_option,
                    )?;
                    let mut oracle_collector = TopDocsCollector::new(k, 0)?;
                    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
                    let oracle_hits = oracle_collector.finish()?.hits;
                    let mut candidate = sealed_union_with_record_option(
                        &posting_lists,
                        Some(&block_max),
                        field,
                        &snapshot,
                        &rows_by_term,
                        &boosts,
                        NUM_DOCS,
                        record_option,
                    )?;
                    let mut candidate_collector = TopDocsCollector::new(k, 0)?;
                    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
                    let pruning_stats = candidate
                        .union_pruning_stats()
                        .expect("top-level union retains pruning stats");
                    let candidate_hits = candidate_collector.finish()?.hits;
                    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
                    assert!(
                        pruning_stats.max_score_windows > 0,
                        "{record_option:?} seed {seed}, k {k} silently fell back"
                    );
                    assert_eq!(pruning_stats.block_max_wand_windows, 0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn randomized_block_max_wand_matches_exhaustive_for_pinned_k_matrix()
    -> Result<(), Box<dyn std::error::Error>> {
        const NUM_DOCS: u32 = 8_192;
        let lengths = (0..NUM_DOCS)
            .map(|doc| Some(if doc.is_multiple_of(2) { 4 } else { 16 }))
            .collect::<Vec<_>>();
        let total_tokens = lengths
            .iter()
            .map(|length| u64::from(length.expect("fixture length is present")))
            .sum();
        let encoded_doclens = EncodedDocLenSection::encode(
            0,
            u64::from(NUM_DOCS),
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let doclens = encoded_doclens.section(&[1])?;
        let field = doclens.field(1).expect("field exists");
        let snapshot = snapshot(1, total_tokens, u64::from(NUM_DOCS))?;
        let boosts = [4.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02];

        for seed in 11_u64..=12 {
            let mut state = seed;
            let mut rows_by_term = (0..9).map(|_| Vec::new()).collect::<Vec<_>>();
            for doc in 0..NUM_DOCS {
                for rows in &mut rows_by_term {
                    state = state
                        .wrapping_mul(2_862_933_555_777_941_757)
                        .wrapping_add(3_037_000_493);
                    if state >> 62 != 0 {
                        let frequency = 1 + u32::try_from((state >> 52) & 31)
                            .expect("randomized frequency bits fit u32");
                        rows.push(Posting::new(doc, frequency));
                    }
                }
            }
            assert!(
                rows_by_term.iter().map(Vec::len).sum::<usize>()
                    >= usize::try_from(BMW_MIN_TOTAL_COST).expect("BMW threshold fits usize")
            );
            let encoded_terms = rows_by_term
                .iter()
                .map(|rows| {
                    EncodedPostingList::encode_with_block_max(rows, |doc| {
                        field.fieldnorm_id(u64::from(doc))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let posting_lists = encoded_terms
                .iter()
                .map(|(postings, _)| postings.posting_list())
                .collect::<Result<Vec<_>, _>>()?;
            let block_max = validated_block_max_entries(&encoded_terms, &posting_lists, field)?;

            for record_option in [TermRecordOption::WithFreqs, TermRecordOption::Basic] {
                for k in [1, 10, 100, 1_000] {
                    let mut oracle = sealed_union_with_record_option(
                        &posting_lists,
                        None,
                        field,
                        &snapshot,
                        &rows_by_term,
                        &boosts,
                        NUM_DOCS,
                        record_option,
                    )?;
                    let mut oracle_collector = TopDocsCollector::new(k, 0)?;
                    oracle_collector.collect(&mut oracle, &AllLiveDocs)?;
                    let oracle_hits = oracle_collector.finish()?.hits;
                    let mut candidate = sealed_union_with_record_option(
                        &posting_lists,
                        Some(&block_max),
                        field,
                        &snapshot,
                        &rows_by_term,
                        &boosts,
                        NUM_DOCS,
                        record_option,
                    )?;
                    let mut candidate_collector = TopDocsCollector::new(k, 0)?;
                    candidate_collector.collect(&mut candidate, &AllLiveDocs)?;
                    let pruning_stats = candidate
                        .union_pruning_stats()
                        .expect("top-level union retains pruning stats");
                    let candidate_hits = candidate_collector.finish()?.hits;
                    assert_hits_bit_exact(&candidate_hits, &oracle_hits);
                    assert_eq!(pruning_stats.max_score_windows, 0);
                    assert!(
                        pruning_stats.block_max_wand_windows > 0,
                        "{record_option:?} seed {seed}, k {k} silently fell back"
                    );
                }
            }
        }
        Ok(())
    }

    // Uses `frankensearch_core::bench_support`, which is feature-gated. This
    // compiled on default features only by accident: an unconditional
    // `frankensearch-lexical` dev-dependency carrying `bench-internals`
    // unified `frankensearch-core/bench-internals` on for every Quill test
    // build. Once that dependency became optional the free ride ended, so the
    // gate is now declared where the dependency actually is (bd-552b).
    #[cfg(feature = "bench-internals")]
    #[test]
    #[ignore = "remote-only E4.4 disjunction performance profile"]
    fn e44_disjunction_profile_100k_and_1m() -> Result<(), Box<dyn std::error::Error>> {
        const PAIRED_ROUNDS: usize = 21;
        const TRIALS: usize = 7;
        for num_docs in [100_000_u32, 1_000_000] {
            let fieldnorm_id = fieldnorm_to_id(8);
            let lengths =
                vec![Some(8); usize::try_from(num_docs).expect("profile count fits usize")];
            let encoded_doclens = EncodedDocLenSection::encode(
                0,
                u64::from(num_docs),
                &[1],
                &[DocLenFieldInput::new(1, &lengths)],
            )?;
            let snapshot = snapshot(1, u64::from(num_docs) * 8, u64::from(num_docs))?;

            let mut rows_by_term = Vec::with_capacity(8);
            let mut rare = Vec::new();
            rare.extend((0..16).map(|doc| Posting::new(doc, 64)));
            rare.extend(
                (8_192..num_docs)
                    .step_by(8_192)
                    .map(|doc| Posting::new(doc, 32)),
            );
            rows_by_term.push(rare);
            for divisor in 2_u32..=9 {
                rows_by_term.push(
                    (0..num_docs)
                        .filter(|doc| doc % divisor == divisor - 1)
                        .map(|doc| Posting::new(doc, 1))
                        .collect(),
                );
            }

            let encoded_terms = rows_by_term
                .iter()
                .map(|rows| EncodedPostingList::encode_with_block_max(rows, |_| Some(fieldnorm_id)))
                .collect::<Result<Vec<_>, _>>()?;
            let boosts = [4.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
            for (strategy, clause_count, group_size) in [
                (UnionPruningStrategy::MaxScore, 8, 1),
                (UnionPruningStrategy::BlockMaxWand, 9, 1),
            ] {
                let encoded_terms = &encoded_terms[..clause_count];
                let rows_by_term = &rows_by_term[..clause_count];
                let boosts = &boosts[..clause_count];
                let cold_exhaustive = timed_encoded_grouped_union(
                    &encoded_doclens,
                    encoded_terms,
                    None,
                    &snapshot,
                    rows_by_term,
                    boosts,
                    num_docs,
                    10,
                    group_size,
                    false,
                )?;
                let cold_pruned = timed_encoded_grouped_union(
                    &encoded_doclens,
                    encoded_terms,
                    None,
                    &snapshot,
                    rows_by_term,
                    boosts,
                    num_docs,
                    10,
                    group_size,
                    true,
                )?;
                assert_hits_bit_exact(&cold_pruned.1, &cold_exhaustive.1);
                let cached_metadata =
                    validate_encoded_pruning_metadata(&encoded_doclens, encoded_terms)?;
                let cache_payload_bytes = cached_metadata.iter().fold(0_usize, |bytes, term| {
                    bytes.saturating_add(term.heap_bytes())
                });

                let null = frankensearch_core::bench_support::paired_median_ratio(
                    PAIRED_ROUNDS,
                    1,
                    || {
                        let _ = std::hint::black_box(
                            timed_encoded_grouped_union(
                                &encoded_doclens,
                                encoded_terms,
                                Some(&cached_metadata),
                                &snapshot,
                                rows_by_term,
                                boosts,
                                num_docs,
                                10,
                                group_size,
                                false,
                            )
                            .expect("run first exhaustive null arm"),
                        );
                    },
                    || {
                        let _ = std::hint::black_box(
                            timed_encoded_grouped_union(
                                &encoded_doclens,
                                encoded_terms,
                                Some(&cached_metadata),
                                &snapshot,
                                rows_by_term,
                                boosts,
                                num_docs,
                                10,
                                group_size,
                                false,
                            )
                            .expect("run second exhaustive null arm"),
                        );
                    },
                );
                let lever = frankensearch_core::bench_support::paired_median_ratio(
                    PAIRED_ROUNDS,
                    1,
                    || {
                        let _ = std::hint::black_box(
                            timed_encoded_grouped_union(
                                &encoded_doclens,
                                encoded_terms,
                                Some(&cached_metadata),
                                &snapshot,
                                rows_by_term,
                                boosts,
                                num_docs,
                                10,
                                group_size,
                                false,
                            )
                            .expect("run paired exhaustive arm"),
                        );
                    },
                    || {
                        let _ = std::hint::black_box(
                            timed_encoded_grouped_union(
                                &encoded_doclens,
                                encoded_terms,
                                Some(&cached_metadata),
                                &snapshot,
                                rows_by_term,
                                boosts,
                                num_docs,
                                10,
                                group_size,
                                true,
                            )
                            .expect("run paired pruned arm"),
                        );
                    },
                );

                // Warm both paths before collecting alternating-order medians.
                let _ = timed_encoded_grouped_union(
                    &encoded_doclens,
                    encoded_terms,
                    Some(&cached_metadata),
                    &snapshot,
                    rows_by_term,
                    boosts,
                    num_docs,
                    10,
                    group_size,
                    false,
                )?;
                let _ = timed_encoded_grouped_union(
                    &encoded_doclens,
                    encoded_terms,
                    Some(&cached_metadata),
                    &snapshot,
                    rows_by_term,
                    boosts,
                    num_docs,
                    10,
                    group_size,
                    true,
                )?;

                let mut exhaustive_us = Vec::with_capacity(TRIALS);
                let mut pruned_us = Vec::with_capacity(TRIALS);
                let mut evidence = None;
                for trial in 0..TRIALS {
                    let run_exhaustive = || {
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            false,
                        )
                    };
                    let run_pruned = || {
                        timed_encoded_grouped_union(
                            &encoded_doclens,
                            encoded_terms,
                            Some(&cached_metadata),
                            &snapshot,
                            rows_by_term,
                            boosts,
                            num_docs,
                            10,
                            group_size,
                            true,
                        )
                    };
                    let (exhaustive, pruned) = if trial % 2 == 0 {
                        (run_exhaustive()?, run_pruned()?)
                    } else {
                        let pruned = run_pruned()?;
                        let exhaustive = run_exhaustive()?;
                        (exhaustive, pruned)
                    };
                    assert_hits_bit_exact(&pruned.1, &exhaustive.1);
                    match strategy {
                        UnionPruningStrategy::MaxScore => {
                            assert!(pruned.2.max_score_windows > 0);
                            assert_eq!(pruned.2.block_max_wand_windows, 0);
                        }
                        UnionPruningStrategy::BlockMaxWand => {
                            assert!(pruned.2.block_max_wand_windows > 0);
                        }
                    }
                    exhaustive_us.push(exhaustive.0);
                    pruned_us.push(pruned.0);
                    let checksum = pruned.1.iter().fold(0_u64, |state, hit| {
                        state
                            .wrapping_mul(16_777_619)
                            .wrapping_add(u64::from(hit.global_docid))
                            .wrapping_add(u64::from(hit.score.to_bits()))
                    });
                    let current_evidence = (pruned.1.len(), checksum);
                    if let Some(expected) = evidence {
                        assert_eq!(current_evidence, expected);
                    } else {
                        evidence = Some(current_evidence);
                    }
                }
                exhaustive_us.sort_unstable();
                pruned_us.sort_unstable();
                let exhaustive_median_us = exhaustive_us[TRIALS / 2];
                let pruned_median_us = pruned_us[TRIALS / 2];
                let (hit_count, checksum) = evidence.expect("profile executes at least one trial");
                let strategy_label = match strategy {
                    UnionPruningStrategy::MaxScore => "maxscore",
                    UnionPruningStrategy::BlockMaxWand => "block_max_wand",
                };
                eprintln!(
                    "e44_profile strategy={strategy_label} docs={num_docs} \
                     cold_exhaustive_us={} cold_pruned_us={} cache_payload_bytes={} \
                     exhaustive_median_us={exhaustive_median_us} \
                     pruned_median_us={pruned_median_us} trials={TRIALS} hits={} \
                     checksum={checksum} null_median={:.6} null_p5={:.6} null_p95={:.6} \
                     lever_median={:.6} lever_p5={:.6} lever_p95={:.6} \
                     paired_rounds={PAIRED_ROUNDS}",
                    cold_exhaustive.0,
                    cold_pruned.0,
                    cache_payload_bytes,
                    hit_count,
                    null.median,
                    null.p5,
                    null.p95,
                    lever.median,
                    lever.p5,
                    lever.p95,
                );
            }
        }
        Ok(())
    }
}
