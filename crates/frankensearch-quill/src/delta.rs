//! Lease-bounded mutable storage for Quill's searchable delta segment.
//!
//! This module owns mutable storage invariants and the immutable freeze
//! boundary. Composite publication lives in `index`, allocation-free cursor
//! adapters live in `argus`, and canonical FSLX sealing lives in `scribe`.

use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::{AHashMap, AHashSet};
use frankensearch_core::DocId;
use thiserror::Error;

use crate::config::DEFAULT_DELTA_BUDGET_BYTES;
use crate::contract::fieldnorm_to_id;
use crate::grimoire::MAX_TERM_BYTES;
use crate::quiver::{NumericValue, POSTINGS_PER_BLOCK};
use crate::schema::{FieldKind, SchemaDescriptor};
#[cfg(test)]
use crate::scribe::{ArenaSpan, FIELD_PREFIX_BYTES, TERM_BUCKET_BYTES_ESTIMATE};
use crate::scribe::{DOC_ORDS_PER_LEASE, TermInterner};

/// First unrolled-chain block size.
pub const DELTA_CHAIN_INITIAL_CAPACITY: usize = 1;
/// Maximum unrolled-chain block size, matching a sealed posting block.
pub const DELTA_CHAIN_MAX_CAPACITY: usize = POSTINGS_PER_BLOCK;

const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;
const TOMBSTONE_WORD_BITS: usize = u64::BITS as usize;
const TOMBSTONE_WORDS_PER_LEASE: usize = DOC_ORDS_PER_LEASE as usize / TOMBSTONE_WORD_BITS;
const HASH_SLOT_ESTIMATE: usize = 2 * size_of::<usize>();
const NEW_TERM_SENTINEL: u32 = u32::MAX;
static NEXT_DELTA_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// Typed delta construction and mutation failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeltaError {
    /// A zero budget would request a seal after every document.
    #[error("delta budget must be greater than zero")]
    ZeroBudget,
    /// Q1 leases begin at fixed 65,536-document boundaries.
    #[error("delta lease base {lease_base} is not aligned to {DOC_ORDS_PER_LEASE} documents")]
    MisalignedLeaseBase { lease_base: u64 },
    /// The lease would exceed the durable u32 posting domain.
    #[error("delta lease [{lease_base}, {lease_end}) exceeds the global u32 docid domain")]
    LeaseOutOfRange { lease_base: u64, lease_end: u64 },
    /// A reset cannot return to an earlier allocator lease.
    #[error("delta lease reset regressed from {current_lease_base} to {next_lease_base}")]
    LeaseRegression {
        current_lease_base: u64,
        next_lease_base: u64,
    },
    /// A document is outside this delta generation's lease.
    #[error("global docid {global_docid} is outside delta lease [{lease_base}, {lease_end})")]
    DocumentOutsideLease {
        global_docid: u32,
        lease_base: u64,
        lease_end: u64,
    },
    /// Same-lease seals retain the first still-admissible global docid.
    #[error("global docid {global_docid} precedes delta continuation floor {floor}")]
    DocumentBeforeFloor { global_docid: u32, floor: u64 },
    /// Physical document rows must be strictly ascending.
    #[error("delta global docid {global_docid} does not follow prior docid {previous}")]
    DocumentOrder { previous: u32, global_docid: u32 },
    /// Durable IDMAP rows cannot carry an empty identifier.
    #[error("delta document id must not be empty")]
    EmptyDocumentId,
    /// Durable IDMAP offsets are u32.
    #[error("delta document id is {bytes} bytes and exceeds the u32 IDMAP domain")]
    DocumentIdTooLarge { bytes: usize },
    /// Every indexed string field needs one schema-ordered norm row.
    #[error("delta fieldnorm row has {actual} fields, expected {expected}")]
    FieldNormCount { expected: usize, actual: usize },
    /// Fieldnorm input order drifted from the schema.
    #[error("delta fieldnorm row {index} names field {actual}, expected {expected}")]
    FieldNormOrder {
        index: usize,
        expected: u16,
        actual: u16,
    },
    /// Quantized and raw field lengths disagree.
    #[error("delta field {field_ord} raw length {raw_length} maps to {expected}, got {actual}")]
    FieldNormMismatch {
        field_ord: u16,
        raw_length: u32,
        expected: u8,
        actual: u8,
    },
    /// Postings may target only Keyword or Text fields.
    #[error("delta posting names non-indexed-string field {field_ord}")]
    UnknownPostingField { field_ord: u16 },
    /// One durable term exceeded the FSLX ceiling.
    #[error("delta term for field {field_ord} is {bytes} bytes, limit {MAX_TERM_BYTES}")]
    TermTooLarge { field_ord: u16, bytes: usize },
    /// Inputs are grouped into one posting per term and document.
    #[error("delta document repeats term {term:?} in field {field_ord}")]
    DuplicateDocumentTerm { field_ord: u16, term: Vec<u8> },
    /// A completed posting always represents an occurrence.
    #[error("delta posting for field {field_ord} term {term:?} has zero frequency")]
    ZeroFrequency { field_ord: u16, term: Vec<u8> },
    /// Positioned Text fields require exact occurrence positions.
    #[error("delta positioned field {field_ord} term {term:?} omitted positions")]
    MissingPositions { field_ord: u16, term: Vec<u8> },
    /// Position-free fields cannot retain position payloads.
    #[error("delta position-free field {field_ord} term {term:?} supplied positions")]
    ForbiddenPositions { field_ord: u16, term: Vec<u8> },
    /// Frequency and position count must match.
    #[error(
        "delta field {field_ord} term {term:?} frequency {frequency} has {positions} positions"
    )]
    PositionCountMismatch {
        field_ord: u16,
        term: Vec<u8>,
        frequency: u32,
        positions: usize,
    },
    /// Positions are nondecreasing; equality is legal.
    #[error("delta field {field_ord} term {term:?} positions descend from {previous} to {current}")]
    DescendingPosition {
        field_ord: u16,
        term: Vec<u8>,
        previous: u32,
        current: u32,
    },
    /// A value named no field in the validated dense descriptor.
    #[error("delta value names unknown schema field {field_ord}")]
    UnknownValueField { field_ord: u16 },
    /// A document may supply a field through only one value input.
    #[error("delta document {global_docid} supplies field {field_ord} more than once")]
    DuplicateValueField { global_docid: u32, field_ord: u16 },
    /// A typed numeric value named a string or stored-only field.
    #[error("delta field {field_ord} ({field_name}) is not numeric")]
    NonNumericField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// A typed numeric value named a field with no NUMERIC column.
    #[error("delta numeric field {field_ord} ({field_name}) is neither indexed nor fast")]
    NonIndexedNumericField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// An in-memory numeric tag disagreed with the schema descriptor.
    #[error("delta numeric field {field_ord} ({field_name}) expects {expected}, got {actual}")]
    NumericTypeMismatch {
        field_ord: u16,
        field_name: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    /// A value named a field whose descriptor does not store original bytes.
    #[error("delta field {field_ord} ({field_name}) is not stored")]
    NonStoredField {
        field_ord: u16,
        field_name: &'static str,
    },
    /// Opaque bytes used as a numeric value must be exactly one scalar wide.
    #[error(
        "delta numeric field {field_ord} ({field_name}) has {actual} stored bytes, expected {expected}"
    )]
    InvalidNumericBytes {
        field_ord: u16,
        field_name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// One stored value cannot be represented by the durable u32 offsets.
    #[error("delta stored field {field_ord} value is {bytes} bytes and exceeds the u32 domain")]
    StoredValueTooLarge { field_ord: u16, bytes: usize },
    /// Appending a value would make one field blob exceed the durable u32 domain.
    #[error(
        "delta stored field {field_ord} blob cannot append {appended} bytes to its current {current} bytes"
    )]
    StoredBlobTooLarge {
        field_ord: u16,
        current: usize,
        appended: usize,
    },
    /// Per-term postings must remain strictly ascending.
    #[error(
        "delta field {field_ord} term {term:?} docid {global_docid} does not follow {previous}"
    )]
    PostingOrder {
        field_ord: u16,
        term: Vec<u8>,
        previous: u32,
        global_docid: u32,
    },
    /// A sidecar or typed arena could not reserve storage.
    #[error("unable to reserve {additional} additional entries for {resource}")]
    Allocation {
        resource: &'static str,
        additional: usize,
    },
    /// A logical locator exceeded its u32 domain.
    #[error("delta {resource} count {count} exceeds the u32 locator domain")]
    CountOverflow {
        resource: &'static str,
        count: usize,
    },
    /// The compile-time schema descriptor is invalid.
    #[error("invalid delta schema: {detail}")]
    InvalidSchema { detail: String },
}

/// Exact per-field document statistics retained by the delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaFieldNorm {
    /// Indexed Keyword/Text field ordinal.
    pub field_ord: u16,
    /// Exact admitted token count for BM25 aggregation.
    pub raw_length: u32,
    /// Tantivy-compatible quantized fieldnorm byte.
    pub fieldnorm_id: u8,
}

/// One schema-typed indexed-or-fast numeric value retained for sealing.
///
/// When the descriptor also sets `stored=true`, the delta derives the exact
/// canonical eight little-endian stored bytes automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNumericValue {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Signed or unsigned value matching the field descriptor.
    pub value: NumericValue,
}

impl DeltaNumericValue {
    /// Construct one signed indexed value.
    #[must_use]
    pub const fn i64(field_ord: u16, value: i64) -> Self {
        Self {
            field_ord,
            value: NumericValue::I64(value),
        }
    }

    /// Construct one unsigned indexed value.
    #[must_use]
    pub const fn u64(field_ord: u16, value: u64) -> Self {
        Self {
            field_ord,
            value: NumericValue::U64(value),
        }
    }
}

/// One opaque schema-stored value retained for sealing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaStoredValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed opaque bytes copied into the delta on apply.
    pub bytes: &'a [u8],
}

impl<'a> DeltaStoredValue<'a> {
    /// Construct one borrowed stored value.
    #[must_use]
    pub const fn new(field_ord: u16, bytes: &'a [u8]) -> Self {
        Self { field_ord, bytes }
    }
}

/// One grouped term posting for a completed document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaTermPosting<'a> {
    /// Indexed Keyword/Text field ordinal.
    pub field_ord: u16,
    /// Exact term bytes without the field prefix.
    pub term: &'a [u8],
    /// Occurrence count in the document.
    pub frequency: u32,
    /// Positions for a position-indexed Text field.
    pub positions: Option<&'a [u32]>,
}

/// Result of applying one complete document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaApply {
    /// Prior live delta docid for this external ID, if any.
    pub replaced_delta_docid: Option<u32>,
    /// Logical current-generation bytes after the apply.
    pub bytes_used: usize,
    /// Whether the accepted document crossed the seal threshold.
    pub seal_required: bool,
}

/// Stable memory and cardinality diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaMemoryStats {
    /// Logical bytes owned by the current generation.
    pub bytes_used: usize,
    /// Retained allocation, including reset-reusable capacity.
    pub bytes_reserved: usize,
    /// Number of distinct `(field, term)` keys.
    pub term_count: usize,
    /// Number of admitted rows, including tombstoned rows.
    pub physical_document_count: usize,
    /// Number of rows visible after applying delta tombstones.
    pub live_document_count: usize,
    /// Active unrolled posting blocks.
    pub posting_blocks: usize,
    /// Active unrolled position blocks.
    pub position_blocks: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PostingRecord {
    global_docid: u32,
    frequency: u32,
    position_start: u32,
    has_positions: bool,
}

/// Immutable posting yielded by a [`DeltaTerm`] view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaPosting<'a> {
    /// Absolute global document identifier.
    pub global_docid: u32,
    /// Occurrence count for this term in the document.
    pub frequency: u32,
    term_index: u32,
    position_start: u32,
    has_positions: bool,
    owner_id: u64,
    generation: u64,
    owner: PhantomData<&'a DeltaSegment>,
}

impl DeltaPosting<'_> {
    /// Whether this posting has an exact position slice.
    #[must_use]
    pub const fn has_positions(self) -> bool {
        self.has_positions
    }
}

/// Conservative whole-term impact envelope for a live Delta cursor.
///
/// Delta postings are append-only within one generation. The envelope keeps
/// the greatest frequency and smallest fieldnorm ever appended for the term.
/// A later upsert or delete may therefore make it looser, but never unsafe for
/// `MaxScore` or block-max pruning over the remaining live rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaBlockMax {
    max_frequency_code: u8,
    min_fieldnorm_id: u8,
}

impl DeltaBlockMax {
    /// Conservatively decoded greatest term frequency.
    #[must_use]
    pub const fn max_frequency(self) -> u32 {
        crate::contract::block_max_frequency_from_code(self.max_frequency_code)
    }

    /// Smallest quantized fieldnorm observed for this term.
    #[must_use]
    pub const fn min_fieldnorm_id(self) -> u8 {
        self.min_fieldnorm_id
    }

    /// Conservative BM25 tf-factor for one live-snapshot average length.
    #[must_use]
    pub fn tf_factor_upper_bound(self, live_avgdl: f32) -> Option<f32> {
        crate::contract::block_max_tf_factor(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
        )
    }

    /// Apply a non-negative `idf * (1 + k1) * boost` term weight.
    #[must_use]
    pub fn score_upper_bound(self, live_avgdl: f32, nonnegative_weight: f32) -> Option<f32> {
        crate::contract::block_max_score(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
            nonnegative_weight,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct BlockMeta {
    start: usize,
    len: usize,
    limit: usize,
    reserved: usize,
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChainState {
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

/// Safe typed bump arena: initialized slots and integer block links only.
#[derive(Debug)]
struct TypedChainArena<T: Copy + Default> {
    resource: &'static str,
    blocks: Vec<BlockMeta>,
    slots: Vec<T>,
    active_blocks: usize,
}

impl<T: Copy + Default> TypedChainArena<T> {
    fn new(resource: &'static str) -> Self {
        Self {
            resource,
            blocks: Vec::new(),
            slots: Vec::new(),
            active_blocks: 0,
        }
    }

    fn allocate_reserved_block(&mut self, limit: usize) -> usize {
        let index = self.active_blocks;
        if index == self.blocks.len() {
            debug_assert!(self.blocks.len() < self.blocks.capacity());
            let start = self.slots.len();
            debug_assert!(start.saturating_add(limit) <= self.slots.capacity());
            self.slots.resize(start + limit, T::default());
            self.blocks.push(BlockMeta {
                start,
                len: 0,
                limit,
                reserved: limit,
                next: None,
            });
        } else {
            if self.blocks[index].reserved < limit {
                let start = self.slots.len();
                debug_assert!(start.saturating_add(limit) <= self.slots.capacity());
                self.slots.resize(start + limit, T::default());
                self.blocks[index].start = start;
                self.blocks[index].reserved = limit;
            }
            self.blocks[index].len = 0;
            self.blocks[index].limit = limit;
            self.blocks[index].next = None;
        }
        self.active_blocks += 1;
        index
    }

    fn append_reserved(&mut self, chain: &mut ChainState, value: T) -> usize {
        let mut added_bytes = 0_usize;
        let mut tail = match chain.tail {
            Some(tail) => tail,
            None => {
                let first = self.allocate_reserved_block(DELTA_CHAIN_INITIAL_CAPACITY);
                added_bytes = added_bytes.saturating_add(
                    size_of::<BlockMeta>()
                        + DELTA_CHAIN_INITIAL_CAPACITY.saturating_mul(size_of::<T>()),
                );
                chain.head = Some(first);
                chain.tail = Some(first);
                first
            }
        };
        if self.blocks[tail].len == self.blocks[tail].limit {
            let next_limit = self.blocks[tail]
                .limit
                .saturating_mul(2)
                .min(DELTA_CHAIN_MAX_CAPACITY);
            let next = self.allocate_reserved_block(next_limit);
            added_bytes = added_bytes
                .saturating_add(size_of::<BlockMeta>() + next_limit.saturating_mul(size_of::<T>()));
            self.blocks[tail].next = Some(next);
            chain.tail = Some(next);
            tail = next;
        }
        let block = &mut self.blocks[tail];
        self.slots[block.start + block.len] = value;
        block.len += 1;
        chain.len = chain.len.checked_add(1).expect("preflighted chain length");
        added_bytes
    }

    fn reserve_appends(&mut self, plans: &[(ChainState, usize)]) -> Result<(), DeltaError> {
        let mut active_blocks = self.active_blocks;
        let mut additional_slots = 0_usize;

        for &(chain, additional) in plans {
            chain
                .len
                .checked_add(additional)
                .ok_or(DeltaError::CountOverflow {
                    resource: self.resource,
                    count: usize::MAX,
                })?;
            if additional == 0 {
                continue;
            }
            let mut remaining = additional;
            let (mut tail_len, mut tail_limit) = match chain.tail {
                Some(tail) => {
                    let block = &self.blocks[tail];
                    (block.len, block.limit)
                }
                None => {
                    self.simulate_block_allocation(
                        &mut active_blocks,
                        DELTA_CHAIN_INITIAL_CAPACITY,
                        &mut additional_slots,
                    )?;
                    (0, DELTA_CHAIN_INITIAL_CAPACITY)
                }
            };
            loop {
                let available = tail_limit.saturating_sub(tail_len);
                let admitted = remaining.min(available);
                remaining -= admitted;
                if remaining == 0 {
                    break;
                }
                let next_limit = tail_limit.saturating_mul(2).min(DELTA_CHAIN_MAX_CAPACITY);
                self.simulate_block_allocation(
                    &mut active_blocks,
                    next_limit,
                    &mut additional_slots,
                )?;
                tail_len = 0;
                tail_limit = next_limit;
            }
        }

        let additional_blocks = active_blocks.saturating_sub(self.blocks.len());
        reserve_vec(&mut self.blocks, additional_blocks, self.resource)?;
        reserve_vec(&mut self.slots, additional_slots, self.resource)
    }

    fn simulate_block_allocation(
        &self,
        active_blocks: &mut usize,
        limit: usize,
        additional_slots: &mut usize,
    ) -> Result<(), DeltaError> {
        let index = *active_blocks;
        if index >= self.blocks.len() || self.blocks[index].reserved < limit {
            *additional_slots =
                additional_slots
                    .checked_add(limit)
                    .ok_or(DeltaError::CountOverflow {
                        resource: self.resource,
                        count: usize::MAX,
                    })?;
        }
        *active_blocks = active_blocks
            .checked_add(1)
            .ok_or(DeltaError::CountOverflow {
                resource: self.resource,
                count: usize::MAX,
            })?;
        Ok(())
    }

    fn iter<'a>(&'a self, chain: &ChainState) -> ChainIter<'a, T> {
        ChainIter {
            arena: self,
            block: chain.head,
            offset: 0,
        }
    }

    fn iter_range<'a>(
        &'a self,
        chain: &ChainState,
        start: usize,
        len: usize,
    ) -> Option<ChainRangeIter<'a, T>> {
        if start.checked_add(len)? > chain.len {
            return None;
        }
        let mut block = chain.head;
        let mut offset = start;
        while let Some(index) = block {
            let meta = &self.blocks[index];
            if offset < meta.len {
                return Some(ChainRangeIter {
                    arena: self,
                    block: Some(index),
                    offset,
                    remaining: len,
                });
            }
            offset -= meta.len;
            block = meta.next;
        }
        (len == 0).then_some(ChainRangeIter {
            arena: self,
            block: None,
            offset: 0,
            remaining: 0,
        })
    }

    /// Copy only the active initialized slots into an immutable epoch arena.
    ///
    /// Mutable arenas retain reset-reusable blocks and relocated slot ranges.
    /// Carrying those inactive allocations into every published snapshot would
    /// keep prior generations' peak footprint alive after a seal. Block indices
    /// remain stable while starts are rebased into one compact owned slot slab.
    fn snapshot_copy(&self) -> Self {
        let mut blocks = Vec::with_capacity(self.active_blocks);
        let active_slots = self.blocks[..self.active_blocks]
            .iter()
            .map(|block| block.len)
            .sum();
        let mut slots = Vec::with_capacity(active_slots);
        for block in &self.blocks[..self.active_blocks] {
            let start = slots.len();
            slots.extend_from_slice(&self.slots[block.start..block.start + block.len]);
            blocks.push(BlockMeta {
                start,
                len: block.len,
                limit: block.len,
                reserved: block.len,
                next: block.next,
            });
        }
        Self {
            resource: self.resource,
            blocks,
            slots,
            active_blocks: self.active_blocks,
        }
    }

    fn reset(&mut self) {
        for block in &mut self.blocks[..self.active_blocks] {
            block.len = 0;
            block.next = None;
        }
        self.active_blocks = 0;
    }

    #[cfg(test)]
    fn bytes_used(&self) -> usize {
        self.blocks[..self.active_blocks]
            .iter()
            .map(|block| size_of::<BlockMeta>() + block.limit.saturating_mul(size_of::<T>()))
            .sum()
    }

    fn bytes_reserved(&self) -> usize {
        self.blocks
            .capacity()
            .saturating_mul(size_of::<BlockMeta>())
            .saturating_add(self.slots.capacity().saturating_mul(size_of::<T>()))
    }

    #[cfg(test)]
    fn block_limits(&self, chain: &ChainState) -> Vec<usize> {
        let mut limits = Vec::new();
        let mut block = chain.head;
        while let Some(index) = block {
            limits.push(self.blocks[index].limit);
            block = self.blocks[index].next;
        }
        limits
    }
}

#[derive(Clone)]
struct ChainIter<'a, T: Copy + Default> {
    arena: &'a TypedChainArena<T>,
    block: Option<usize>,
    offset: usize,
}

impl<'a, T: Copy + Default> Iterator for ChainIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let index = self.block?;
            let block = &self.arena.blocks[index];
            if self.offset < block.len {
                let item = &self.arena.slots[block.start + self.offset];
                self.offset += 1;
                return Some(item);
            }
            self.block = block.next;
            self.offset = 0;
        }
    }
}

struct ChainRangeIter<'a, T: Copy + Default> {
    arena: &'a TypedChainArena<T>,
    block: Option<usize>,
    offset: usize,
    remaining: usize,
}

impl<'a, T: Copy + Default> Iterator for ChainRangeIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        loop {
            let index = self.block?;
            let block = &self.arena.blocks[index];
            if self.offset < block.len {
                let item = &self.arena.slots[block.start + self.offset];
                self.offset += 1;
                self.remaining -= 1;
                return Some(item);
            }
            self.block = block.next;
            self.offset = 0;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FieldLayout {
    field_ord: u16,
    positions: bool,
}

#[derive(Debug, Clone)]
struct FieldNormColumn {
    raw_lengths: Vec<u32>,
    fieldnorm_ids: Vec<u8>,
}

#[derive(Debug, Clone)]
struct NumericColumn {
    field_ord: u16,
    values: Vec<Option<NumericValue>>,
}

#[derive(Debug, Clone)]
struct StoredColumn {
    field_ord: u16,
    values: Vec<Option<Vec<u8>>>,
    payload_bytes: usize,
}

type ResolvedDocumentValues = (Vec<Option<NumericValue>>, Vec<Option<Vec<u8>>>);

#[derive(Debug, Clone, Copy)]
struct TermChain {
    postings: ChainState,
    positions: ChainState,
    last_docid: Option<u32>,
    max_frequency_code: u8,
    min_fieldnorm_id: u8,
}

impl Default for TermChain {
    fn default() -> Self {
        Self {
            postings: ChainState::default(),
            positions: ChainState::default(),
            last_docid: None,
            max_frequency_code: 0,
            min_fieldnorm_id: u8::MAX,
        }
    }
}

/// One mutable, lease-bounded delta generation.
#[derive(Debug)]
pub struct DeltaSegment {
    schema: SchemaDescriptor,
    lease_base: u64,
    lease_end: u64,
    next_docid_floor: u64,
    budget_bytes: usize,
    owner_id: u64,
    generation: u64,
    fields: Vec<FieldLayout>,
    terms: TermInterner,
    chains: Vec<TermChain>,
    posting_arena: TypedChainArena<PostingRecord>,
    position_arena: TypedChainArena<u32>,
    document_docids: Vec<u32>,
    document_ids: Vec<DocId>,
    document_content_hashes: Vec<Option<u64>>,
    document_term_offsets: Vec<u32>,
    document_term_ids: Vec<u32>,
    fieldnorms: Vec<FieldNormColumn>,
    numeric_fields: Vec<NumericColumn>,
    stored_fields: Vec<StoredColumn>,
    live_ids: AHashMap<DocId, u32>,
    tombstone_words: Vec<u64>,
    tombstone_count: usize,
    logical_bytes_used: usize,
}

/// Immutable, owner-isolated copy of one published delta generation.
///
/// Publication deliberately deep-copies the mutable shard state. Readers can
/// then retain this value behind an [`std::sync::Arc`] without observing later
/// appends, tombstones, or arena reuse by the writer. The copy receives a fresh
/// owner identity so a posting handle from either side cannot resolve positions
/// through the other side's arenas.
#[derive(Debug)]
pub struct DeltaSnapshot {
    segment: DeltaSegment,
    keeper_generation: u64,
    lineage_id: u64,
}

impl DeltaSnapshot {
    /// Read-only view of the frozen delta generation.
    #[must_use]
    pub const fn segment(&self) -> &DeltaSegment {
        &self.segment
    }

    /// Keeper MANIFEST generation this delta epoch was derived from.
    #[must_use]
    pub const fn keeper_generation(&self) -> u64 {
        self.keeper_generation
    }

    /// Clone this exact frozen epoch against a successor Keeper generation.
    ///
    /// Sealing one shard advances the MANIFEST generation for every shard.
    /// Surviving immutable epochs use this operation to retain their private
    /// lineage witness while receiving owner-isolated arena storage for the
    /// successor composite publication.
    #[must_use]
    pub fn rebind_keeper_generation(&self, keeper_generation: u64) -> Self {
        let mut rebound = self.segment.freeze(keeper_generation);
        rebound.lineage_id = self.lineage_id;
        rebound
    }

    /// Private shard/epoch/lease identity retained across Keeper rebinding.
    pub(crate) const fn publication_lineage(&self) -> (u64, u64, u64, u64) {
        (
            self.lineage_id,
            self.segment.generation,
            self.segment.lease_base,
            self.segment.lease_end,
        )
    }

    /// Compile-time schema carried by this generation.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.segment.schema()
    }

    /// Inclusive global lease base.
    #[must_use]
    pub const fn lease_base(&self) -> u64 {
        self.segment.lease_base()
    }

    /// Exclusive global lease end.
    #[must_use]
    pub const fn lease_end(&self) -> u64 {
        self.segment.lease_end()
    }

    /// Number of visible rows after applying delta-local tombstones.
    #[must_use]
    pub fn live_document_count(&self) -> usize {
        self.segment.live_document_count()
    }

    /// Whether one global document id names a visible row in this generation.
    #[must_use]
    pub fn is_live_document(&self, global_docid: u32) -> bool {
        self.segment
            .document_docids
            .binary_search(&global_docid)
            .is_ok()
            && !self.segment.is_tombstoned(global_docid)
    }

    /// Materialize one visible row's stable external identifier.
    #[must_use]
    pub fn materialize_document_id(&self, global_docid: u32) -> Option<DocId> {
        let row = self
            .segment
            .document_docids
            .binary_search(&global_docid)
            .ok()?;
        if self.segment.is_tombstoned(global_docid) {
            return None;
        }
        self.segment.document_ids.get(row).cloned()
    }

    /// Exact live token numerator for one indexed string field.
    #[must_use]
    pub fn live_total_tokens(&self, field_ord: u16) -> Option<u64> {
        self.segment.live_total_tokens(field_ord)
    }

    /// Physical fieldnorm byte for one row in this frozen generation.
    #[must_use]
    pub fn fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        self.segment.fieldnorm_id(field_ord, global_docid)
    }

    /// Physical content hash, including a tombstoned row.
    #[must_use]
    pub fn content_hash(&self, global_docid: u32) -> Option<u64> {
        self.segment.content_hash(global_docid)
    }

    /// Physical indexed-or-fast numeric value, including a tombstoned row.
    #[must_use]
    pub fn numeric_value(&self, field_ord: u16, global_docid: u32) -> Option<NumericValue> {
        self.segment.numeric_value(field_ord, global_docid)
    }

    /// Physical stored bytes, including a tombstoned row.
    #[must_use]
    pub fn stored_value(&self, field_ord: u16, global_docid: u32) -> Option<&[u8]> {
        self.segment.stored_value(field_ord, global_docid)
    }

    /// Live-only fieldnorm byte for one visible row in this generation.
    #[must_use]
    pub fn live_fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        if self.segment.is_tombstoned(global_docid) {
            return None;
        }
        self.segment.fieldnorm_id(field_ord, global_docid)
    }

    /// Look up one immutable composite `(field, term)` key.
    #[must_use]
    pub fn find_term(&self, field_ord: u16, term: &[u8]) -> Option<DeltaTerm<'_>> {
        self.segment.find_term(field_ord, term)
    }

    /// Ordered live rows for snapshot composition and later cursor lowering.
    #[must_use]
    pub fn live_documents(&self) -> DeltaLiveDocuments<'_> {
        self.segment.live_documents()
    }

    /// Covering Q1 interval of every physically allocated row in this epoch.
    ///
    /// Tombstoned rows remain part of the interval because their docids were
    /// allocated and therefore cannot collide with a sealed Keeper range.
    pub(crate) fn occupied_docid_range(&self) -> Option<(u64, u64)> {
        let first = u64::from(*self.segment.document_docids.first()?);
        let last = u64::from(*self.segment.document_docids.last()?);
        Some((first, last + 1))
    }
}

impl crate::argus::LiveDocs for DeltaSnapshot {
    fn is_live(&self, global_docid: u32) -> bool {
        self.is_live_document(global_docid)
    }
}

impl DeltaSegment {
    /// Construct a delta with the pinned default 8 MiB budget.
    ///
    /// # Errors
    ///
    /// Rejects an invalid schema or lease range.
    pub fn with_default_budget(
        schema: SchemaDescriptor,
        lease_base: u64,
    ) -> Result<Self, DeltaError> {
        Self::new(schema, lease_base, DEFAULT_DELTA_BUDGET_BYTES)
    }

    /// Construct an empty delta for one aligned Q1 lease.
    ///
    /// # Errors
    ///
    /// Rejects zero budget, invalid schema, misalignment, or a lease extending
    /// past the global u32 posting domain.
    pub fn new(
        schema: SchemaDescriptor,
        lease_base: u64,
        budget_bytes: usize,
    ) -> Result<Self, DeltaError> {
        if budget_bytes == 0 {
            return Err(DeltaError::ZeroBudget);
        }
        schema
            .validate()
            .map_err(|error| DeltaError::InvalidSchema {
                detail: error.to_string(),
            })?;
        let lease_end = validate_lease_base(lease_base)?;
        let fields: Vec<FieldLayout> = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                FieldKind::Keyword => Some(FieldLayout {
                    field_ord: field.id,
                    positions: false,
                }),
                FieldKind::Text { positions, .. } => Some(FieldLayout {
                    field_ord: field.id,
                    positions,
                }),
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => None,
            })
            .collect();
        let fieldnorms = fields
            .iter()
            .map(|_| FieldNormColumn {
                raw_lengths: Vec::new(),
                fieldnorm_ids: Vec::new(),
            })
            .collect();
        let numeric_fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                kind if kind.has_numeric_column() => Some(NumericColumn {
                    field_ord: field.id,
                    values: Vec::new(),
                }),
                _ => None,
            })
            .collect();
        let stored_fields = schema
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| StoredColumn {
                field_ord: field.id,
                values: Vec::new(),
                payload_bytes: 0,
            })
            .collect();
        let mut tombstone_words = Vec::new();
        tombstone_words
            .try_reserve_exact(TOMBSTONE_WORDS_PER_LEASE)
            .map_err(|_| DeltaError::Allocation {
                resource: "delta tombstone bitmap",
                additional: TOMBSTONE_WORDS_PER_LEASE,
            })?;
        Ok(Self {
            schema,
            lease_base,
            lease_end,
            next_docid_floor: lease_base,
            budget_bytes,
            owner_id: NEXT_DELTA_OWNER_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            fields,
            terms: TermInterner::new(),
            chains: Vec::new(),
            posting_arena: TypedChainArena::new("delta posting arena"),
            position_arena: TypedChainArena::new("delta position arena"),
            document_docids: Vec::new(),
            document_ids: Vec::new(),
            document_content_hashes: Vec::new(),
            document_term_offsets: vec![0],
            document_term_ids: Vec::new(),
            fieldnorms,
            numeric_fields,
            stored_fields,
            live_ids: AHashMap::new(),
            tombstone_words,
            tombstone_count: 0,
            logical_bytes_used: 0,
        })
    }

    /// Compile-time schema carried by this generation.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Inclusive global lease base.
    #[must_use]
    pub const fn lease_base(&self) -> u64 {
        self.lease_base
    }

    /// Exclusive global lease end.
    #[must_use]
    pub const fn lease_end(&self) -> u64 {
        self.lease_end
    }

    /// First global docid admissible after any same-lease reset.
    #[must_use]
    pub const fn next_docid_floor(&self) -> u64 {
        self.next_docid_floor
    }

    /// Configured logical-byte seal threshold.
    #[must_use]
    pub const fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Freeze the current generation for lock-free epoch publication.
    ///
    /// The returned snapshot owns independent arenas and sidecars. Subsequent
    /// writes or resets of this mutable segment cannot change the snapshot,
    /// and owner-bound posting handles cannot cross the freeze boundary.
    /// `keeper_generation` binds the epoch to the MANIFEST view current at the
    /// batch boundary; composite publication rejects a stale witness.
    #[must_use]
    pub fn freeze(&self, keeper_generation: u64) -> DeltaSnapshot {
        let snapshot_owner_id = NEXT_DELTA_OWNER_ID.fetch_add(1, Ordering::Relaxed);
        DeltaSnapshot {
            segment: Self {
                schema: self.schema,
                lease_base: self.lease_base,
                lease_end: self.lease_end,
                next_docid_floor: self.next_docid_floor,
                budget_bytes: self.budget_bytes,
                owner_id: snapshot_owner_id,
                generation: self.generation,
                fields: self.fields.clone(),
                terms: self.terms.clone(),
                chains: self.chains.clone(),
                posting_arena: self.posting_arena.snapshot_copy(),
                position_arena: self.position_arena.snapshot_copy(),
                document_docids: self.document_docids.clone(),
                document_ids: self.document_ids.clone(),
                document_content_hashes: self.document_content_hashes.clone(),
                document_term_offsets: self.document_term_offsets.clone(),
                document_term_ids: self.document_term_ids.clone(),
                fieldnorms: self.fieldnorms.clone(),
                numeric_fields: self.numeric_fields.clone(),
                stored_fields: self.stored_fields.clone(),
                live_ids: self.live_ids.clone(),
                tombstone_words: self.tombstone_words.clone(),
                tombstone_count: self.tombstone_count,
                logical_bytes_used: self.logical_bytes_used,
            },
            keeper_generation,
            // Each frozen epoch is a distinct publication candidate, even
            // when two freezes observe the same mutable Delta generation.
            // Rebinding deliberately preserves this witness; a later freeze
            // must never be replaceable by an earlier, shorter snapshot.
            lineage_id: snapshot_owner_id,
        }
    }

    /// Apply one complete document without seal sidecar values.
    ///
    /// The physical row records an absent content hash and absent
    /// numeric/stored values. It remains process-locally searchable, but a
    /// later FSLX seal rejects it rather than fabricating an identity witness.
    /// Use [`Self::apply_document_with_values`] for sealable rows.
    ///
    /// # Errors
    ///
    /// Delegates all validation and allocation failures to
    /// [`Self::apply_document_with_values`].
    pub fn apply_document(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.apply_document_inner(
            global_docid,
            document_id,
            None,
            fieldnorms,
            postings,
            &[],
            &[],
        )
    }

    /// Apply one complete document with every sidecar required for sealing.
    ///
    /// Numeric and stored values may be in any order, but a field may appear in
    /// only one input slice. A stored indexed-or-fast numeric supplied through
    /// `numeric_values` receives canonical little-endian stored bytes. Supplying
    /// its exact eight stored bytes instead also populates the numeric column.
    /// A budget crossing accepts the entire row and reports `seal_required`; a
    /// document is never split.
    ///
    /// # Errors
    ///
    /// Semantic validation and all fallible capacity planning complete before
    /// logical state changes.
    pub fn apply_document_with_values(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        content_hash: u64,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.apply_document_inner(
            global_docid,
            document_id,
            Some(content_hash),
            fieldnorms,
            postings,
            numeric_values,
            stored_values,
        )
    }

    fn apply_document_inner(
        &mut self,
        global_docid: u32,
        document_id: DocId,
        content_hash: Option<u64>,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<DeltaApply, DeltaError> {
        self.validate_document(
            global_docid,
            &document_id,
            fieldnorms,
            postings,
            numeric_values,
            stored_values,
        )?;
        let (resolved_numeric_values, resolved_stored_values) =
            self.resolve_document_values(numeric_values, stored_values)?;
        let mut document_terms = Vec::new();
        reserve_vec(
            &mut document_terms,
            postings.len(),
            "delta document term scratch",
        )?;
        let mut posting_plans = Vec::new();
        let mut position_plans = Vec::new();
        reserve_vec(
            &mut posting_plans,
            postings.len(),
            "delta posting reservation plan",
        )?;
        reserve_vec(
            &mut position_plans,
            postings.len(),
            "delta position reservation plan",
        )?;
        let mut new_term_count = 0_usize;
        for posting in postings {
            let planned_term = self.terms.find(posting.field_ord, posting.term);
            let chain = if let Some(term_index) = planned_term {
                let chain =
                    self.chains[usize::try_from(term_index).expect("u32 term id fits usize")];
                if let Some(previous) = chain.last_docid
                    && global_docid <= previous
                {
                    return Err(DeltaError::PostingOrder {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                        previous,
                        global_docid,
                    });
                }
                chain
            } else {
                new_term_count =
                    new_term_count
                        .checked_add(1)
                        .ok_or(DeltaError::CountOverflow {
                            resource: "term",
                            count: usize::MAX,
                        })?;
                TermChain::default()
            };
            let position_count = posting.positions.map_or(0, <[u32]>::len);
            let final_position_count = chain.positions.len.checked_add(position_count).ok_or(
                DeltaError::CountOverflow {
                    resource: "position",
                    count: usize::MAX,
                },
            )?;
            u32::try_from(final_position_count).map_err(|_| DeltaError::CountOverflow {
                resource: "position",
                count: final_position_count,
            })?;
            posting_plans.push((chain.postings, 1));
            position_plans.push((chain.positions, position_count));
            document_terms.push(planned_term.unwrap_or(NEW_TERM_SENTINEL));
        }
        let final_term_count =
            self.chains
                .len()
                .checked_add(new_term_count)
                .ok_or(DeltaError::CountOverflow {
                    resource: "term",
                    count: usize::MAX,
                })?;
        u32::try_from(final_term_count).map_err(|_| DeltaError::CountOverflow {
            resource: "term",
            count: final_term_count,
        })?;
        let final_document_term_count = self
            .document_term_ids
            .len()
            .checked_add(postings.len())
            .ok_or(DeltaError::CountOverflow {
                resource: "document term",
                count: usize::MAX,
            })?;
        u32::try_from(final_document_term_count).map_err(|_| DeltaError::CountOverflow {
            resource: "document term",
            count: final_document_term_count,
        })?;
        self.reserve_document(postings.len())?;
        self.posting_arena.reserve_appends(&posting_plans)?;
        self.position_arena.reserve_appends(&position_plans)?;
        let document_id_len = document_id.len();
        let overlay_id = document_id.clone();
        let identity_was_live = self.live_ids.contains_key(&overlay_id);
        let stored_value_bytes = resolved_stored_values
            .iter()
            .flatten()
            .map(Vec::len)
            .sum::<usize>();

        let mut added_bytes = 0_usize;
        for (posting, planned_term) in postings.iter().zip(&mut document_terms) {
            let field_index = self
                .field_index(posting.field_ord)
                .expect("validated indexed-string field has a fieldnorm column");
            let fieldnorm_id = fieldnorms[field_index].fieldnorm_id;
            let (term_index, term_bytes) = if *planned_term == NEW_TERM_SENTINEL {
                self.terms.intern_accounted(posting.field_ord, posting.term)
            } else {
                (*planned_term, 0)
            };
            *planned_term = term_index;
            added_bytes = added_bytes.saturating_add(term_bytes);
            let term_offset = usize::try_from(term_index).expect("u32 term id fits usize");
            if term_offset == self.chains.len() {
                self.chains.push(TermChain::default());
                added_bytes = added_bytes.saturating_add(size_of::<TermChain>());
            }
            let chain = &mut self.chains[term_offset];
            let position_start = if let Some(positions) = posting.positions {
                let start = u32::try_from(chain.positions.len)
                    .expect("preflighted position start fits u32");
                for &position in positions {
                    added_bytes = added_bytes.saturating_add(
                        self.position_arena
                            .append_reserved(&mut chain.positions, position),
                    );
                }
                start
            } else {
                0
            };
            added_bytes = added_bytes.saturating_add(self.posting_arena.append_reserved(
                &mut chain.postings,
                PostingRecord {
                    global_docid,
                    frequency: posting.frequency,
                    position_start,
                    has_positions: posting.positions.is_some(),
                },
            ));
            chain.last_docid = Some(global_docid);
            chain.max_frequency_code =
                chain
                    .max_frequency_code
                    .max(crate::contract::block_max_frequency_to_code(
                        posting.frequency,
                    ));
            chain.min_fieldnorm_id = chain.min_fieldnorm_id.min(fieldnorm_id);
        }

        self.document_docids.push(global_docid);
        self.document_ids.push(document_id);
        self.document_content_hashes.push(content_hash);
        self.document_term_ids.extend_from_slice(&document_terms);
        let offset = u32::try_from(self.document_term_ids.len())
            .expect("preflighted document-term offset fits u32");
        self.document_term_offsets.push(offset);
        for (column, input) in self.fieldnorms.iter_mut().zip(fieldnorms) {
            column.raw_lengths.push(input.raw_length);
            column.fieldnorm_ids.push(input.fieldnorm_id);
        }
        for (column, value) in self.numeric_fields.iter_mut().zip(resolved_numeric_values) {
            column.values.push(value);
        }
        for (column, value) in self.stored_fields.iter_mut().zip(resolved_stored_values) {
            column.payload_bytes = column
                .payload_bytes
                .checked_add(value.as_ref().map_or(0, Vec::len))
                .expect("validated stored payload length fits usize");
            column.values.push(value);
        }
        added_bytes = added_bytes
            .saturating_add(size_of::<u32>())
            .saturating_add(size_of::<DocId>() + document_id_len)
            .saturating_add(size_of::<Option<u64>>())
            .saturating_add(size_of::<u32>())
            .saturating_add(postings.len().saturating_mul(size_of::<u32>()))
            .saturating_add(
                self.fieldnorms
                    .len()
                    .saturating_mul(size_of::<u32>() + size_of::<u8>()),
            )
            .saturating_add(
                self.numeric_fields
                    .len()
                    .saturating_mul(size_of::<Option<NumericValue>>()),
            )
            .saturating_add(
                self.stored_fields
                    .len()
                    .saturating_mul(size_of::<Option<Vec<u8>>>()),
            )
            .saturating_add(stored_value_bytes);
        let replaced_delta_docid = self.live_ids.insert(overlay_id, global_docid);
        if !identity_was_live {
            added_bytes = added_bytes.saturating_add(
                size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + document_id_len,
            );
        }
        if let Some(replaced) = replaced_delta_docid {
            added_bytes = added_bytes.saturating_add(self.mark_tombstone(replaced));
        }
        self.logical_bytes_used = self.logical_bytes_used.saturating_add(added_bytes);
        let bytes_used = self.logical_bytes_used;
        Ok(DeltaApply {
            replaced_delta_docid,
            bytes_used,
            seal_required: bytes_used >= self.budget_bytes,
        })
    }

    fn validate_document(
        &self,
        global_docid: u32,
        document_id: &str,
        fieldnorms: &[DeltaFieldNorm],
        postings: &[DeltaTermPosting<'_>],
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<(), DeltaError> {
        if self.lease_ordinal(global_docid).is_none() {
            return Err(DeltaError::DocumentOutsideLease {
                global_docid,
                lease_base: self.lease_base,
                lease_end: self.lease_end,
            });
        }
        if u64::from(global_docid) < self.next_docid_floor {
            return Err(DeltaError::DocumentBeforeFloor {
                global_docid,
                floor: self.next_docid_floor,
            });
        }
        if let Some(&previous) = self.document_docids.last()
            && global_docid <= previous
        {
            return Err(DeltaError::DocumentOrder {
                previous,
                global_docid,
            });
        }
        if document_id.is_empty() {
            return Err(DeltaError::EmptyDocumentId);
        }
        if document_id.len() > u32::MAX as usize {
            return Err(DeltaError::DocumentIdTooLarge {
                bytes: document_id.len(),
            });
        }
        if fieldnorms.len() != self.fields.len() {
            return Err(DeltaError::FieldNormCount {
                expected: self.fields.len(),
                actual: fieldnorms.len(),
            });
        }
        for (index, (field, input)) in self.fields.iter().zip(fieldnorms).enumerate() {
            if input.field_ord != field.field_ord {
                return Err(DeltaError::FieldNormOrder {
                    index,
                    expected: field.field_ord,
                    actual: input.field_ord,
                });
            }
            let expected = fieldnorm_to_id(input.raw_length);
            if input.fieldnorm_id != expected {
                return Err(DeltaError::FieldNormMismatch {
                    field_ord: input.field_ord,
                    raw_length: input.raw_length,
                    expected,
                    actual: input.fieldnorm_id,
                });
            }
        }

        let mut unique: AHashSet<(u16, &[u8])> = AHashSet::new();
        unique
            .try_reserve(postings.len())
            .map_err(|_| DeltaError::Allocation {
                resource: "delta document term validation",
                additional: postings.len(),
            })?;
        for posting in postings {
            let Some(field_index) = self.field_index(posting.field_ord) else {
                return Err(DeltaError::UnknownPostingField {
                    field_ord: posting.field_ord,
                });
            };
            if posting.term.len() > MAX_TERM_BYTES {
                return Err(DeltaError::TermTooLarge {
                    field_ord: posting.field_ord,
                    bytes: posting.term.len(),
                });
            }
            if !unique.insert((posting.field_ord, posting.term)) {
                return Err(DeltaError::DuplicateDocumentTerm {
                    field_ord: posting.field_ord,
                    term: posting.term.to_vec(),
                });
            }
            if posting.frequency == 0 {
                return Err(DeltaError::ZeroFrequency {
                    field_ord: posting.field_ord,
                    term: posting.term.to_vec(),
                });
            }
            match (self.fields[field_index].positions, posting.positions) {
                (true, None) => {
                    return Err(DeltaError::MissingPositions {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                    });
                }
                (false, Some(_)) => {
                    return Err(DeltaError::ForbiddenPositions {
                        field_ord: posting.field_ord,
                        term: posting.term.to_vec(),
                    });
                }
                (true, Some(positions)) => {
                    if usize::try_from(posting.frequency).ok() != Some(positions.len()) {
                        return Err(DeltaError::PositionCountMismatch {
                            field_ord: posting.field_ord,
                            term: posting.term.to_vec(),
                            frequency: posting.frequency,
                            positions: positions.len(),
                        });
                    }
                    if let Some(pair) = positions.windows(2).find(|pair| pair[0] > pair[1]) {
                        return Err(DeltaError::DescendingPosition {
                            field_ord: posting.field_ord,
                            term: posting.term.to_vec(),
                            previous: pair[0],
                            current: pair[1],
                        });
                    }
                }
                (false, None) => {}
            }
        }
        self.validate_document_values(global_docid, numeric_values, stored_values)
    }

    fn validate_document_values(
        &self,
        global_docid: u32,
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<(), DeltaError> {
        let mut seen_fields = Vec::new();
        reserve_vec(
            &mut seen_fields,
            self.schema.fields.len(),
            "delta document value validation",
        )?;
        seen_fields.resize(self.schema.fields.len(), false);

        for value in numeric_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut seen_fields[field_index], true) {
                return Err(DeltaError::DuplicateValueField {
                    global_docid,
                    field_ord: value.field_ord,
                });
            }
            match (field.kind, value.value) {
                (FieldKind::I64 { indexed, fast }, NumericValue::I64(_)) if indexed || fast => {}
                (FieldKind::U64 { indexed, fast }, NumericValue::U64(_)) if indexed || fast => {}
                (
                    FieldKind::I64 {
                        indexed: false,
                        fast: false,
                    }
                    | FieldKind::U64 {
                        indexed: false,
                        fast: false,
                    },
                    _,
                ) => {
                    return Err(DeltaError::NonIndexedNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
                (FieldKind::I64 { .. }, actual) => {
                    return Err(DeltaError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "i64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                (FieldKind::U64 { .. }, actual) => {
                    return Err(DeltaError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "u64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                _ => {
                    return Err(DeltaError::NonNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
        }

        for value in stored_values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(DeltaError::UnknownValueField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut seen_fields[field_index], true) {
                return Err(DeltaError::DuplicateValueField {
                    global_docid,
                    field_ord: value.field_ord,
                });
            }
            if !field.stored {
                return Err(DeltaError::NonStoredField {
                    field_ord: field.id,
                    field_name: field.name,
                });
            }
            if matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. })
                && value.bytes.len() != size_of::<u64>()
            {
                return Err(DeltaError::InvalidNumericBytes {
                    field_ord: field.id,
                    field_name: field.name,
                    expected: size_of::<u64>(),
                    actual: value.bytes.len(),
                });
            }
            if u32::try_from(value.bytes.len()).is_err() {
                return Err(DeltaError::StoredValueTooLarge {
                    field_ord: field.id,
                    bytes: value.bytes.len(),
                });
            }
        }
        for field in &self.stored_fields {
            let appended = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map_or_else(
                    || {
                        stored_values
                            .iter()
                            .find(|value| value.field_ord == field.field_ord)
                            .map_or(0, |value| value.bytes.len())
                    },
                    |_| size_of::<u64>(),
                );
            let final_bytes = field.payload_bytes.checked_add(appended);
            if final_bytes.is_none_or(|bytes| u32::try_from(bytes).is_err()) {
                return Err(DeltaError::StoredBlobTooLarge {
                    field_ord: field.field_ord,
                    current: field.payload_bytes,
                    appended,
                });
            }
        }
        Ok(())
    }

    fn resolve_document_values(
        &self,
        numeric_values: &[DeltaNumericValue],
        stored_values: &[DeltaStoredValue<'_>],
    ) -> Result<ResolvedDocumentValues, DeltaError> {
        let mut resolved_numeric_values = Vec::new();
        reserve_vec(
            &mut resolved_numeric_values,
            self.numeric_fields.len(),
            "delta numeric value scratch",
        )?;
        for field in &self.numeric_fields {
            let value = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.value)
                .or_else(|| {
                    stored_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .and_then(|value| {
                            numeric_value_from_le_bytes(
                                self.schema.fields[usize::from(field.field_ord)].kind,
                                value.bytes,
                            )
                        })
                });
            resolved_numeric_values.push(value);
        }

        let mut resolved_stored_values = Vec::new();
        reserve_vec(
            &mut resolved_stored_values,
            self.stored_fields.len(),
            "delta stored value scratch",
        )?;
        for field in &self.stored_fields {
            let numeric_bytes = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.value.to_le_bytes());
            let bytes = numeric_bytes
                .as_ref()
                .map(|bytes| bytes.as_slice())
                .or_else(|| {
                    stored_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.bytes)
                });
            let value = if let Some(bytes) = bytes {
                let mut owned = Vec::new();
                reserve_vec(&mut owned, bytes.len(), "delta stored value bytes")?;
                owned.extend_from_slice(bytes);
                Some(owned)
            } else {
                None
            };
            resolved_stored_values.push(value);
        }
        Ok((resolved_numeric_values, resolved_stored_values))
    }

    fn reserve_document(&mut self, term_count: usize) -> Result<(), DeltaError> {
        reserve_vec(&mut self.document_docids, 1, "delta document docids")?;
        reserve_vec(&mut self.document_ids, 1, "delta document ids")?;
        reserve_vec(
            &mut self.document_content_hashes,
            1,
            "delta document content hashes",
        )?;
        reserve_vec(
            &mut self.document_term_offsets,
            1,
            "delta document term offsets",
        )?;
        reserve_vec(
            &mut self.document_term_ids,
            term_count,
            "delta document term ids",
        )?;
        reserve_vec(&mut self.chains, term_count, "delta term chains")?;
        self.live_ids
            .try_reserve(1)
            .map_err(|_| DeltaError::Allocation {
                resource: "delta live identity overlay",
                additional: 1,
            })?;
        for column in &mut self.fieldnorms {
            reserve_vec(&mut column.raw_lengths, 1, "delta raw field lengths")?;
            reserve_vec(&mut column.fieldnorm_ids, 1, "delta fieldnorm bytes")?;
        }
        for column in &mut self.numeric_fields {
            reserve_vec(&mut column.values, 1, "delta numeric values")?;
        }
        for column in &mut self.stored_fields {
            reserve_vec(&mut column.values, 1, "delta stored values")?;
        }
        Ok(())
    }

    /// Probe the live delta overlay. A miss directs the caller to sealed IDHASH.
    #[must_use]
    pub fn probe_id(&self, document_id: &str) -> Option<u32> {
        self.live_ids.get(document_id).copied()
    }

    /// Delete a live delta identity. A repeated or sealed-only delete is a miss.
    pub fn delete_delta_id(&mut self, document_id: &str) -> Option<u32> {
        let (removed_id, global_docid) = self.live_ids.remove_entry(document_id)?;
        let identity_bytes =
            size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + removed_id.len();
        self.logical_bytes_used = self
            .logical_bytes_used
            .checked_sub(identity_bytes)
            .expect("live identity bytes are accounted");
        let tombstone_bytes = self.mark_tombstone(global_docid);
        self.logical_bytes_used = self.logical_bytes_used.saturating_add(tombstone_bytes);
        Some(global_docid)
    }

    /// Find a composite term in O(1) expected time with exact collision checks.
    #[must_use]
    pub fn find_term(&self, field_ord: u16, term: &[u8]) -> Option<DeltaTerm<'_>> {
        Some(DeltaTerm {
            delta: self,
            term_index: self.terms.find(field_ord, term)?,
        })
    }

    /// Term views in canonical `(field_ord BE, term bytes)` order.
    #[must_use]
    pub fn sorted_terms(&self) -> Vec<DeltaTerm<'_>> {
        self.terms
            .sorted_ids()
            .into_iter()
            .map(|term_index| DeltaTerm {
                delta: self,
                term_index,
            })
            .collect()
    }

    /// Physical fieldnorm, including a tombstoned row.
    #[must_use]
    pub fn fieldnorm_id(&self, field_ord: u16, global_docid: u32) -> Option<u8> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.field_index(field_ord)?;
        self.fieldnorms[field].fieldnorm_ids.get(row).copied()
    }

    /// Physical content hash, including a tombstoned row.
    #[must_use]
    pub fn content_hash(&self, global_docid: u32) -> Option<u64> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        self.document_content_hashes.get(row).copied().flatten()
    }

    /// Physical indexed-or-fast numeric value, including a tombstoned row.
    #[must_use]
    pub fn numeric_value(&self, field_ord: u16, global_docid: u32) -> Option<NumericValue> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.numeric_field_index(field_ord)?;
        self.numeric_fields[field]
            .values
            .get(row)
            .copied()
            .flatten()
    }

    /// Physical stored bytes, including a tombstoned row.
    #[must_use]
    pub fn stored_value(&self, field_ord: u16, global_docid: u32) -> Option<&[u8]> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.stored_field_index(field_ord)?;
        self.stored_fields[field]
            .values
            .get(row)
            .and_then(Option::as_deref)
    }

    /// Exact raw length for snapshot-level BM25 aggregation.
    #[must_use]
    pub fn raw_field_length(&self, field_ord: u16, global_docid: u32) -> Option<u32> {
        let row = self.document_docids.binary_search(&global_docid).ok()?;
        let field = self.field_index(field_ord)?;
        self.fieldnorms[field].raw_lengths.get(row).copied()
    }

    /// Exact live token numerator for an indexed string field.
    #[must_use]
    pub fn live_total_tokens(&self, field_ord: u16) -> Option<u64> {
        let field = self.field_index(field_ord)?;
        Some(
            self.document_docids
                .iter()
                .zip(&self.fieldnorms[field].raw_lengths)
                .filter(|(docid, _)| !self.is_tombstoned(**docid))
                .map(|(_, length)| u64::from(*length))
                .sum(),
        )
    }

    /// Whether a physical delta row has been superseded or deleted.
    #[must_use]
    pub fn is_tombstoned(&self, global_docid: u32) -> bool {
        let Some(ordinal) = self.lease_ordinal(global_docid) else {
            return false;
        };
        let word = ordinal / TOMBSTONE_WORD_BITS;
        let bit = ordinal % TOMBSTONE_WORD_BITS;
        self.tombstone_words
            .get(word)
            .is_some_and(|value| value & (1_u64 << bit) != 0)
    }

    #[must_use]
    pub fn physical_document_count(&self) -> usize {
        self.document_docids.len()
    }

    #[must_use]
    pub fn live_document_count(&self) -> usize {
        self.document_docids
            .len()
            .saturating_sub(self.tombstone_count)
    }

    /// Ordered live rows for snapshot composition and sealing.
    #[must_use]
    pub fn live_documents(&self) -> DeltaLiveDocuments<'_> {
        DeltaLiveDocuments {
            delta: self,
            index: 0,
        }
    }

    /// Logical current-generation bytes used by the seal trigger.
    ///
    /// Active unrolled slots (including last-block slack) are counted;
    /// reset-retained capacity is not.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.logical_bytes_used
    }

    #[cfg(test)]
    fn recompute_bytes_used(&self) -> usize {
        let identity_bytes = self
            .live_ids
            .keys()
            .map(|id| size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + id.len())
            .sum::<usize>();
        let document_id_bytes = self
            .document_ids
            .iter()
            .map(|id| size_of::<DocId>() + id.len())
            .sum::<usize>();
        let fieldnorm_bytes = self
            .fieldnorms
            .iter()
            .map(|column| {
                column
                    .raw_lengths
                    .len()
                    .saturating_mul(size_of::<u32>())
                    .saturating_add(column.fieldnorm_ids.len())
            })
            .sum::<usize>();
        let numeric_value_bytes = self
            .numeric_fields
            .iter()
            .map(|column| {
                column
                    .values
                    .len()
                    .saturating_mul(size_of::<Option<NumericValue>>())
            })
            .sum::<usize>();
        let stored_value_bytes = self
            .stored_fields
            .iter()
            .map(|column| {
                column
                    .values
                    .len()
                    .saturating_mul(size_of::<Option<Vec<u8>>>())
                    .saturating_add(column.values.iter().flatten().map(Vec::len).sum::<usize>())
            })
            .sum::<usize>();
        self.terms
            .bytes_used()
            .saturating_add(self.chains.len().saturating_mul(size_of::<TermChain>()))
            .saturating_add(self.posting_arena.bytes_used())
            .saturating_add(self.position_arena.bytes_used())
            .saturating_add(self.document_docids.len().saturating_mul(size_of::<u32>()))
            .saturating_add(document_id_bytes)
            .saturating_add(
                self.document_content_hashes
                    .len()
                    .saturating_mul(size_of::<Option<u64>>()),
            )
            .saturating_add(
                self.document_term_offsets
                    .len()
                    .saturating_sub(1)
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.document_term_ids
                    .len()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(fieldnorm_bytes)
            .saturating_add(numeric_value_bytes)
            .saturating_add(stored_value_bytes)
            .saturating_add(identity_bytes)
            .saturating_add(self.tombstone_words.len().saturating_mul(size_of::<u64>()))
    }

    /// Complete retained allocation for reset/RSS diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        let identity_bytes = self
            .live_ids
            .keys()
            .map(|id| id.len())
            .sum::<usize>()
            .saturating_add(
                self.live_ids
                    .capacity()
                    .saturating_mul(size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE),
            );
        let document_id_bytes = self
            .document_ids
            .iter()
            .map(|id| id.len())
            .sum::<usize>()
            .saturating_add(
                self.document_ids
                    .capacity()
                    .saturating_mul(size_of::<DocId>()),
            );
        let fieldnorm_bytes = self
            .fieldnorms
            .iter()
            .map(|column| {
                column
                    .raw_lengths
                    .capacity()
                    .saturating_mul(size_of::<u32>())
                    .saturating_add(column.fieldnorm_ids.capacity())
            })
            .sum::<usize>();
        let numeric_value_bytes = self
            .numeric_fields
            .capacity()
            .saturating_mul(size_of::<NumericColumn>())
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(|column| {
                        column
                            .values
                            .capacity()
                            .saturating_mul(size_of::<Option<NumericValue>>())
                    })
                    .sum::<usize>(),
            );
        let stored_value_bytes = self
            .stored_fields
            .capacity()
            .saturating_mul(size_of::<StoredColumn>())
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(|column| {
                        column
                            .values
                            .capacity()
                            .saturating_mul(size_of::<Option<Vec<u8>>>())
                            .saturating_add(
                                column
                                    .values
                                    .iter()
                                    .flatten()
                                    .map(Vec::capacity)
                                    .sum::<usize>(),
                            )
                    })
                    .sum::<usize>(),
            );
        self.fields
            .capacity()
            .saturating_mul(size_of::<FieldLayout>())
            .saturating_add(self.terms.bytes_reserved())
            .saturating_add(
                self.chains
                    .capacity()
                    .saturating_mul(size_of::<TermChain>()),
            )
            .saturating_add(self.posting_arena.bytes_reserved())
            .saturating_add(self.position_arena.bytes_reserved())
            .saturating_add(
                self.document_docids
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(document_id_bytes)
            .saturating_add(
                self.document_content_hashes
                    .capacity()
                    .saturating_mul(size_of::<Option<u64>>()),
            )
            .saturating_add(
                self.document_term_offsets
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.document_term_ids
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.fieldnorms
                    .capacity()
                    .saturating_mul(size_of::<FieldNormColumn>()),
            )
            .saturating_add(fieldnorm_bytes)
            .saturating_add(numeric_value_bytes)
            .saturating_add(stored_value_bytes)
            .saturating_add(identity_bytes)
            .saturating_add(
                self.tombstone_words
                    .capacity()
                    .saturating_mul(size_of::<u64>()),
            )
    }

    /// Whether a nonempty generation reached its budget.
    #[must_use]
    pub fn should_seal(&self) -> bool {
        !self.document_docids.is_empty() && self.bytes_used() >= self.budget_bytes
    }

    /// Snapshot stable memory/cardinality diagnostics.
    #[must_use]
    pub fn memory_stats(&self) -> DeltaMemoryStats {
        DeltaMemoryStats {
            bytes_used: self.bytes_used(),
            bytes_reserved: self.bytes_reserved(),
            term_count: self.terms.len(),
            physical_document_count: self.physical_document_count(),
            live_document_count: self.live_document_count(),
            posting_blocks: self.posting_arena.active_blocks,
            position_blocks: self.position_arena.active_blocks,
        }
    }

    /// Clear a sealed generation while retaining writer allocations for the
    /// next aligned lease. Rust's borrow rules prevent reset while a term view
    /// lives; published [`DeltaSnapshot`] epochs own independent compact copies.
    ///
    /// # Errors
    ///
    /// Rejects an invalid next lease before changing state.
    pub fn reset_after_seal(&mut self, next_lease_base: u64) -> Result<(), DeltaError> {
        let next_lease_end = validate_lease_base(next_lease_base)?;
        if next_lease_base < self.lease_base {
            return Err(DeltaError::LeaseRegression {
                current_lease_base: self.lease_base,
                next_lease_base,
            });
        }
        let next_docid_floor = if next_lease_base == self.lease_base {
            self.document_docids
                .last()
                .map_or(self.next_docid_floor, |docid| u64::from(*docid) + 1)
        } else {
            next_lease_base
        };
        self.lease_base = next_lease_base;
        self.lease_end = next_lease_end;
        self.next_docid_floor = next_docid_floor;
        self.generation = self.generation.wrapping_add(1);
        self.terms.reset();
        self.chains.clear();
        self.posting_arena.reset();
        self.position_arena.reset();
        self.document_docids.clear();
        self.document_ids.clear();
        self.document_content_hashes.clear();
        self.document_term_offsets.clear();
        self.document_term_offsets.push(0);
        self.document_term_ids.clear();
        for column in &mut self.fieldnorms {
            column.raw_lengths.clear();
            column.fieldnorm_ids.clear();
        }
        for column in &mut self.numeric_fields {
            column.values.clear();
        }
        for column in &mut self.stored_fields {
            column.values.clear();
            column.payload_bytes = 0;
        }
        self.live_ids.clear();
        self.tombstone_words.fill(0);
        self.tombstone_words.clear();
        self.tombstone_count = 0;
        self.logical_bytes_used = 0;
        Ok(())
    }

    fn field_index(&self, field_ord: u16) -> Option<usize> {
        self.fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn numeric_field_index(&self, field_ord: u16) -> Option<usize> {
        self.numeric_fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn stored_field_index(&self, field_ord: u16) -> Option<usize> {
        self.stored_fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()
    }

    fn lease_ordinal(&self, global_docid: u32) -> Option<usize> {
        let ordinal = u64::from(global_docid).checked_sub(self.lease_base)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        (ordinal < DOC_ORDS_PER_LEASE as usize).then_some(ordinal)
    }

    fn mark_tombstone(&mut self, global_docid: u32) -> usize {
        let ordinal = self
            .lease_ordinal(global_docid)
            .expect("live delta identity belongs to this lease");
        let word = ordinal / TOMBSTONE_WORD_BITS;
        let bit = ordinal % TOMBSTONE_WORD_BITS;
        let prior_words = self.tombstone_words.len();
        if prior_words <= word {
            self.tombstone_words.resize(word + 1, 0);
        }
        let mask = 1_u64 << bit;
        if self.tombstone_words[word] & mask == 0 {
            self.tombstone_words[word] |= mask;
            self.tombstone_count += 1;
        }
        self.tombstone_words
            .len()
            .saturating_sub(prior_words)
            .saturating_mul(size_of::<u64>())
    }
}

/// Owner-bound read view for one composite delta term.
#[derive(Debug, Clone, Copy)]
pub struct DeltaTerm<'a> {
    delta: &'a DeltaSegment,
    term_index: u32,
}

impl<'a> DeltaTerm<'a> {
    /// Schema field ordinal carried by this key.
    #[must_use]
    pub fn field_ord(self) -> u16 {
        self.delta.terms.field_and_term(self.term_index).0
    }

    /// Exact term bytes without the field prefix.
    #[must_use]
    pub fn term(self) -> &'a [u8] {
        self.delta.terms.field_and_term(self.term_index).1
    }

    /// Physical posting count, including tombstoned rows.
    #[must_use]
    pub fn physical_doc_freq(self) -> usize {
        self.delta.chains[self.term_index as usize].postings.len
    }

    /// Live posting count after applying delta tombstones.
    #[must_use]
    pub fn live_doc_freq(self) -> usize {
        self.postings()
            .filter(|posting| self.is_live(*posting))
            .count()
    }

    /// Compute the live cardinality and whole-term pruning envelope in one
    /// physical-chain pass.
    #[must_use]
    pub(crate) fn live_doc_freq_and_block_max(self) -> (usize, Option<DeltaBlockMax>) {
        let live_doc_freq = self.live_doc_freq();
        let block_max = (live_doc_freq != 0).then(|| {
            let chain = &self.delta.chains[self.term_index as usize];
            DeltaBlockMax {
                max_frequency_code: chain.max_frequency_code,
                min_fieldnorm_id: chain.min_fieldnorm_id,
            }
        });
        (live_doc_freq, block_max)
    }

    /// Conservative whole-term impact envelope for future pruning paths.
    ///
    /// Returns `None` when every physical posting has been superseded or
    /// deleted. Otherwise the bound may include tombstoned historical maxima,
    /// which only makes it more conservative.
    #[must_use]
    pub fn block_max(self) -> Option<DeltaBlockMax> {
        self.live_doc_freq_and_block_max().1
    }

    /// Whether an owner-bound posting remains visible in this generation.
    pub(crate) fn is_live(self, posting: DeltaPosting<'a>) -> bool {
        posting.term_index == self.term_index
            && posting.owner_id == self.delta.owner_id
            && posting.generation == self.delta.generation
            && !self.delta.is_tombstoned(posting.global_docid)
    }

    /// Iterate physical postings in global-docid order.
    #[must_use]
    pub fn postings(self) -> DeltaPostings<'a> {
        let chain = &self.delta.chains[self.term_index as usize];
        DeltaPostings {
            inner: self.delta.posting_arena.iter(&chain.postings),
            term_index: self.term_index,
            owner_id: self.delta.owner_id,
            generation: self.delta.generation,
        }
    }

    /// Resolve positions for a posting yielded by this same term.
    #[must_use]
    pub fn positions(self, posting: DeltaPosting<'a>) -> Option<DeltaPositions<'a>> {
        if posting.term_index != self.term_index
            || posting.owner_id != self.delta.owner_id
            || posting.generation != self.delta.generation
            || !posting.has_positions
        {
            return None;
        }
        let chain = &self.delta.chains[self.term_index as usize];
        Some(DeltaPositions {
            inner: self.delta.position_arena.iter_range(
                &chain.positions,
                posting.position_start as usize,
                posting.frequency as usize,
            )?,
        })
    }
}

/// Physical posting iterator for one owner-bound term.
#[derive(Clone)]
pub struct DeltaPostings<'a> {
    inner: ChainIter<'a, PostingRecord>,
    term_index: u32,
    owner_id: u64,
    generation: u64,
}

impl<'a> Iterator for DeltaPostings<'a> {
    type Item = DeltaPosting<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let posting = *self.inner.next()?;
        Some(DeltaPosting {
            global_docid: posting.global_docid,
            frequency: posting.frequency,
            term_index: self.term_index,
            position_start: posting.position_start,
            has_positions: posting.has_positions,
            owner_id: self.owner_id,
            generation: self.generation,
            owner: PhantomData,
        })
    }
}

/// Exact position iterator for one posting.
pub struct DeltaPositions<'a> {
    inner: ChainRangeIter<'a, u32>,
}

impl Iterator for DeltaPositions<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied()
    }
}

/// Ordered live document iterator.
pub struct DeltaLiveDocuments<'a> {
    delta: &'a DeltaSegment,
    index: usize,
}

impl<'a> Iterator for DeltaLiveDocuments<'a> {
    type Item = (u32, &'a DocId);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.delta.document_docids.len() {
            let index = self.index;
            self.index += 1;
            let docid = self.delta.document_docids[index];
            if !self.delta.is_tombstoned(docid) {
                return Some((docid, &self.delta.document_ids[index]));
            }
        }
        None
    }
}

fn numeric_value_type_name(value: NumericValue) -> &'static str {
    match value {
        NumericValue::I64(_) => "i64",
        NumericValue::U64(_) => "u64",
    }
}

fn numeric_value_from_le_bytes(kind: FieldKind, bytes: &[u8]) -> Option<NumericValue> {
    let bytes: [u8; 8] = bytes.try_into().ok()?;
    match kind {
        FieldKind::I64 { .. } => Some(NumericValue::I64(i64::from_le_bytes(bytes))),
        FieldKind::U64 { .. } => Some(NumericValue::U64(u64::from_le_bytes(bytes))),
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => None,
    }
}

fn validate_lease_base(lease_base: u64) -> Result<u64, DeltaError> {
    if !lease_base.is_multiple_of(u64::from(DOC_ORDS_PER_LEASE)) {
        return Err(DeltaError::MisalignedLeaseBase { lease_base });
    }
    let lease_end = lease_base.saturating_add(u64::from(DOC_ORDS_PER_LEASE));
    if lease_end > MAX_GLOBAL_DOCID_EXCLUSIVE {
        return Err(DeltaError::LeaseOutOfRange {
            lease_base,
            lease_end,
        });
    }
    Ok(lease_end)
}

fn reserve_vec<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), DeltaError> {
    values
        .try_reserve(additional)
        .map_err(|_| DeltaError::Allocation {
            resource,
            additional,
        })
}

#[cfg(test)]
mod tests {
    use crate::schema::{Analyzer, FieldDescriptor};

    use super::*;

    const TEST_FIELDS: [FieldDescriptor; 4] = [
        FieldDescriptor {
            id: 0,
            name: "keyword",
            kind: FieldKind::Keyword,
            stored: false,
        },
        FieldDescriptor {
            id: 1,
            name: "positioned",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 2,
            name: "plain_text",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 3,
            name: "stored",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
    ];
    const TEST_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "delta-tests",
        fields: &TEST_FIELDS,
    };

    const VALUE_FIELDS: [FieldDescriptor; 5] = [
        FieldDescriptor {
            id: 0,
            name: "keyword",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "signed",
            kind: FieldKind::I64 {
                indexed: true,
                fast: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "unsigned",
            kind: FieldKind::U64 {
                indexed: true,
                fast: true,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 3,
            name: "fast_only",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 4,
            name: "opaque",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
    ];
    const VALUE_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "delta-value-tests",
        fields: &VALUE_FIELDS,
    };

    fn norms(keyword: u32, positioned: u32, plain: u32) -> [DeltaFieldNorm; 3] {
        [
            DeltaFieldNorm {
                field_ord: 0,
                raw_length: keyword,
                fieldnorm_id: fieldnorm_to_id(keyword),
            },
            DeltaFieldNorm {
                field_ord: 1,
                raw_length: positioned,
                fieldnorm_id: fieldnorm_to_id(positioned),
            },
            DeltaFieldNorm {
                field_ord: 2,
                raw_length: plain,
                fieldnorm_id: fieldnorm_to_id(plain),
            },
        ]
    }

    fn value_norms(keyword: u32) -> [DeltaFieldNorm; 1] {
        [DeltaFieldNorm {
            field_ord: 0,
            raw_length: keyword,
            fieldnorm_id: fieldnorm_to_id(keyword),
        }]
    }

    fn apply_positioned(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        term: &[u8],
        positions: &[u32],
    ) -> Result<DeltaApply, DeltaError> {
        let frequency = u32::try_from(positions.len()).expect("test position count fits u32");
        let applied = delta.apply_document(
            global_docid,
            DocId::from(document_id),
            &norms(0, frequency, 0),
            &[DeltaTermPosting {
                field_ord: 1,
                term,
                frequency,
                positions: Some(positions),
            }],
        )?;
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
        Ok(applied)
    }

    fn expected_single_positioned_bytes(term: &[u8], document_id: &str) -> usize {
        let term_table_bytes =
            FIELD_PREFIX_BYTES + term.len() + size_of::<ArenaSpan>() + TERM_BUCKET_BYTES_ESTIMATE;
        let posting_bytes = size_of::<BlockMeta>() + size_of::<PostingRecord>();
        let position_bytes = 2 * size_of::<BlockMeta>() + 3 * size_of::<u32>();
        let document_id_bytes = size_of::<DocId>() + document_id.len();
        let identity_bytes =
            size_of::<DocId>() + size_of::<u32>() + HASH_SLOT_ESTIMATE + document_id.len();
        let fieldnorm_bytes = 3 * (size_of::<u32>() + size_of::<u8>());
        let seal_sidecar_bytes = size_of::<Option<u64>>() + size_of::<Option<Vec<u8>>>();
        term_table_bytes
            + size_of::<TermChain>()
            + posting_bytes
            + position_bytes
            + size_of::<u32>()
            + document_id_bytes
            + size_of::<u32>()
            + size_of::<u32>()
            + fieldnorm_bytes
            + seal_sidecar_bytes
            + identity_bytes
    }

    #[test]
    fn seal_sidecars_survive_tombstones_freeze_and_writer_reset() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(VALUE_SCHEMA, 0, usize::MAX)?;
        let fast_only = 31_u64;
        delta.apply_document_with_values(
            0,
            DocId::from("same-id"),
            0x1122_3344_5566_7788,
            &value_norms(1),
            &[],
            &[
                DeltaNumericValue::i64(1, -7),
                DeltaNumericValue::u64(2, 9),
                DeltaNumericValue::u64(3, fast_only),
            ],
            &[
                DeltaStoredValue::new(0, b"first-keyword"),
                DeltaStoredValue::new(4, b"first-opaque"),
            ],
        )?;

        assert_eq!(delta.content_hash(0), Some(0x1122_3344_5566_7788));
        assert_eq!(delta.numeric_value(1, 0), Some(NumericValue::I64(-7)));
        assert_eq!(delta.numeric_value(2, 0), Some(NumericValue::U64(9)));
        assert_eq!(
            delta.numeric_value(3, 0),
            Some(NumericValue::U64(fast_only))
        );
        assert_eq!(delta.stored_value(0, 0), Some(b"first-keyword".as_slice()));
        assert_eq!(
            delta.stored_value(1, 0),
            Some((-7_i64).to_le_bytes().as_slice())
        );
        assert_eq!(delta.stored_value(2, 0), None);
        assert_eq!(
            delta.stored_value(3, 0),
            Some(fast_only.to_le_bytes().as_slice())
        );
        assert_eq!(delta.stored_value(4, 0), Some(b"first-opaque".as_slice()));
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

        let signed_from_stored = 42_i64.to_le_bytes();
        let next_fast_only = 63_u64;
        let replacement = delta.apply_document_with_values(
            1,
            DocId::from("same-id"),
            0x8877_6655_4433_2211,
            &value_norms(0),
            &[],
            &[
                DeltaNumericValue::u64(2, 17),
                DeltaNumericValue::u64(3, next_fast_only),
            ],
            &[
                DeltaStoredValue::new(1, &signed_from_stored),
                DeltaStoredValue::new(4, b"second-opaque"),
            ],
        )?;
        assert_eq!(replacement.replaced_delta_docid, Some(0));
        assert!(delta.is_tombstoned(0));
        assert_eq!(delta.content_hash(0), Some(0x1122_3344_5566_7788));
        assert_eq!(delta.stored_value(4, 0), Some(b"first-opaque".as_slice()));
        assert_eq!(delta.numeric_value(1, 1), Some(NumericValue::I64(42)));
        assert_eq!(
            delta.stored_value(1, 1),
            Some(signed_from_stored.as_slice())
        );
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

        let frozen = delta.freeze(19);
        assert!(frozen.segment().is_tombstoned(0));
        assert_eq!(frozen.content_hash(0), Some(0x1122_3344_5566_7788));
        assert_eq!(frozen.numeric_value(1, 1), Some(NumericValue::I64(42)));
        assert_eq!(
            frozen.numeric_value(3, 1),
            Some(NumericValue::U64(next_fast_only))
        );
        assert_eq!(frozen.stored_value(4, 1), Some(b"second-opaque".as_slice()));

        delta.reset_after_seal(0)?;
        assert_eq!(delta.content_hash(0), None);
        assert_eq!(delta.numeric_value(1, 1), None);
        assert_eq!(delta.stored_value(4, 1), None);
        assert_eq!(delta.bytes_used(), 0);

        delta.apply_document(2, DocId::from("convenience"), &value_norms(0), &[])?;
        assert_eq!(delta.content_hash(2), None);
        assert_eq!(delta.numeric_value(1, 2), None);
        assert_eq!(delta.stored_value(4, 2), None);
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
        assert_eq!(frozen.content_hash(1), Some(0x8877_6655_4433_2211));
        Ok(())
    }

    #[test]
    fn seal_sidecar_validation_is_typed_and_atomic() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(VALUE_SCHEMA, 0, usize::MAX)?;
        let initial = delta.memory_stats();

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("unknown"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::u64(5, 1)],
                &[],
            ),
            Err(DeltaError::UnknownValueField { field_ord: 5 })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("duplicate"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::i64(1, 1), DeltaNumericValue::i64(1, 2)],
                &[],
            ),
            Err(DeltaError::DuplicateValueField { field_ord: 1, .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("non-numeric"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::u64(0, 1)],
                &[],
            ),
            Err(DeltaError::NonNumericField { field_ord: 0, .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("fast-wrong-type"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::i64(3, -1)],
                &[],
            ),
            Err(DeltaError::NumericTypeMismatch {
                field_ord: 3,
                expected: "u64",
                actual: "i64",
                ..
            })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("wrong-type"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::u64(1, 1)],
                &[],
            ),
            Err(DeltaError::NumericTypeMismatch {
                field_ord: 1,
                expected: "i64",
                actual: "u64",
                ..
            })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("not-stored"),
                1,
                &value_norms(0),
                &[],
                &[],
                &[DeltaStoredValue::new(2, &1_u64.to_le_bytes())],
            ),
            Err(DeltaError::NonStoredField { field_ord: 2, .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("bad-width"),
                1,
                &value_norms(0),
                &[],
                &[],
                &[DeltaStoredValue::new(3, b"short")],
            ),
            Err(DeltaError::InvalidNumericBytes {
                field_ord: 3,
                expected: 8,
                actual: 5,
                ..
            })
        ));
        assert_eq!(delta.memory_stats(), initial);

        let signed = 1_i64.to_le_bytes();
        assert!(matches!(
            delta.apply_document_with_values(
                0,
                DocId::from("cross-slice-duplicate"),
                1,
                &value_norms(0),
                &[],
                &[DeltaNumericValue::i64(1, 1)],
                &[DeltaStoredValue::new(1, &signed)],
            ),
            Err(DeltaError::DuplicateValueField { field_ord: 1, .. })
        ));
        assert_eq!(delta.memory_stats(), initial);
        Ok(())
    }

    #[test]
    fn unrolled_chains_grow_exponentially_then_cap() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        for global_docid in 0_u32..300 {
            let positions = [global_docid * 2, global_docid * 2 + 1];
            apply_positioned(&mut delta, global_docid, "shared-id", b"growth", &positions)?;
        }

        let term_index = delta
            .terms
            .find(1, b"growth")
            .expect("growth term is present");
        let chain = &delta.chains[usize::try_from(term_index).expect("u32 fits usize")];
        assert_eq!(
            delta.posting_arena.block_limits(&chain.postings),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 128]
        );
        assert_eq!(
            delta.position_arena.block_limits(&chain.positions),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128]
        );
        assert_eq!(chain.postings.len, 300);
        assert_eq!(chain.positions.len, 600);
        assert!(chain.postings.tail.is_some());
        assert!(chain.positions.tail.is_some());

        let term = delta.find_term(1, b"growth").expect("growth term view");
        let last = term.postings().last().expect("last posting");
        assert_eq!(last.global_docid, 299);
        assert_eq!(
            term.positions(last)
                .expect("position view")
                .collect::<Vec<_>>(),
            [598, 599]
        );
        Ok(())
    }

    #[test]
    fn arena_preflight_matches_mixed_reuse_replacement_and_growth() -> Result<(), DeltaError> {
        let mut arena = TypedChainArena::<u32>::new("mixed reuse test arena");
        let mut short_chains = [ChainState::default(); 3];
        arena.reserve_appends(&[
            (short_chains[0], 1),
            (short_chains[1], 1),
            (short_chains[2], 1),
        ])?;
        for (value, chain) in short_chains.iter_mut().enumerate() {
            arena.append_reserved(chain, u32::try_from(value).expect("small test value"));
        }
        assert_eq!(arena.blocks.len(), 3);
        assert!(arena.blocks.iter().all(|block| block.reserved == 1));

        arena.reset();
        let mut grown = ChainState::default();
        arena.reserve_appends(&[(grown, 8)])?;
        for value in 0_u32..8 {
            arena.append_reserved(&mut grown, value);
        }

        assert_eq!(arena.block_limits(&grown), vec![1, 2, 4, 8]);
        assert_eq!(
            arena.iter(&grown).copied().collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            arena.blocks.len(),
            4,
            "one descriptor grows past high water"
        );
        assert_eq!(
            arena.blocks[0].reserved, 1,
            "first descriptor reuses in place"
        );
        assert_eq!(arena.blocks[1].reserved, 2, "second descriptor is replaced");
        assert_eq!(arena.blocks[2].reserved, 4, "third descriptor is replaced");
        assert_eq!(arena.blocks[3].reserved, 8, "fourth descriptor is new");
        Ok(())
    }

    #[test]
    fn append_read_interleave_preserves_field_namespace_and_positions() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let first_positions = [0, 0, 4];
        delta.apply_document(
            0,
            DocId::from("first"),
            &norms(1, 3, 2),
            &[
                DeltaTermPosting {
                    field_ord: 0,
                    term: b"same",
                    frequency: 1,
                    positions: None,
                },
                DeltaTermPosting {
                    field_ord: 1,
                    term: b"same",
                    frequency: 3,
                    positions: Some(&first_positions),
                },
                DeltaTermPosting {
                    field_ord: 2,
                    term: b"plain",
                    frequency: 2,
                    positions: None,
                },
            ],
        )?;

        {
            let keyword = delta.find_term(0, b"same").expect("keyword term");
            let positioned = delta.find_term(1, b"same").expect("positioned term");
            let keyword_posting = keyword.postings().next().expect("keyword posting");
            let positioned_posting = positioned.postings().next().expect("positioned posting");
            assert!(!keyword_posting.has_positions());
            assert!(keyword.positions(keyword_posting).is_none());
            assert_eq!(
                positioned
                    .positions(positioned_posting)
                    .expect("positions")
                    .collect::<Vec<_>>(),
                first_positions
            );
        }

        let second_positions = [2, 7];
        apply_positioned(&mut delta, 1, "second", b"same", &second_positions)?;
        let positioned = delta.find_term(1, b"same").expect("positioned term");
        let postings = positioned.postings().collect::<Vec<_>>();
        assert_eq!(
            postings
                .iter()
                .map(|posting| posting.global_docid)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            positioned
                .positions(postings[1])
                .expect("second positions")
                .collect::<Vec<_>>(),
            second_positions
        );
        assert_eq!(delta.raw_field_length(1, 0), Some(3));
        assert_eq!(delta.fieldnorm_id(1, 0), Some(fieldnorm_to_id(3)));
        assert_eq!(delta.live_total_tokens(1), Some(5));
        assert_eq!(delta.raw_field_length(3, 0), None);

        let sorted = delta
            .sorted_terms()
            .into_iter()
            .map(|term| (term.field_ord(), term.term().to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(
            sorted,
            [
                (0, b"same".to_vec()),
                (1, b"same".to_vec()),
                (2, b"plain".to_vec()),
            ]
        );
        Ok(())
    }

    #[test]
    fn interleaved_terms_keep_independent_chain_links_across_growth_boundaries()
    -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        for global_docid in 0_u32..300 {
            let alpha_position = [global_docid * 2];
            let beta_position = [global_docid * 2 + 1];
            delta.apply_document(
                global_docid,
                DocId::from("same-id"),
                &norms(0, 2, 0),
                &[
                    DeltaTermPosting {
                        field_ord: 1,
                        term: b"alpha",
                        frequency: 1,
                        positions: Some(&alpha_position),
                    },
                    DeltaTermPosting {
                        field_ord: 1,
                        term: b"beta",
                        frequency: 1,
                        positions: Some(&beta_position),
                    },
                ],
            )?;
        }

        let expected_limits = [1, 2, 4, 8, 16, 32, 64, 128, 128];
        for (term_bytes, parity) in [(b"alpha".as_slice(), 0), (b"beta".as_slice(), 1)] {
            let term_index = delta.terms.find(1, term_bytes).expect("interleaved term");
            let chain = &delta.chains[usize::try_from(term_index).expect("u32 fits usize")];
            assert_eq!(
                delta.posting_arena.block_limits(&chain.postings),
                expected_limits
            );
            assert_eq!(
                delta.position_arena.block_limits(&chain.positions),
                expected_limits
            );
            let term = delta.find_term(1, term_bytes).expect("term view");
            for posting in term.postings() {
                assert_eq!(
                    term.positions(posting)
                        .expect("one position")
                        .collect::<Vec<_>>(),
                    [posting.global_docid * 2 + parity]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn posting_handles_cannot_cross_delta_owners_or_generations() -> Result<(), DeltaError> {
        let mut left = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let mut right = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        apply_positioned(&mut left, 0, "left", b"term", &[1, 2])?;
        apply_positioned(&mut right, 0, "right", b"term", &[7, 8])?;

        let left_term = left.find_term(1, b"term").expect("left term");
        let left_posting = left_term.postings().next().expect("left posting");
        let right_term = right.find_term(1, b"term").expect("right term");
        assert!(right_term.positions(left_posting).is_none());

        let mut stale_generation = right_term.postings().next().expect("right posting");
        stale_generation.generation = stale_generation.generation.wrapping_add(1);
        assert!(right_term.positions(stale_generation).is_none());
        Ok(())
    }

    #[test]
    fn frozen_snapshot_is_immutable_and_rejects_cross_owner_postings() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        apply_positioned(&mut delta, 0, "first", b"term", &[1, 4])?;
        let frozen = delta.freeze(7);

        assert_eq!(frozen.keeper_generation(), 7);
        assert_ne!(delta.owner_id, frozen.segment.owner_id);
        {
            let mutable_term = delta.find_term(1, b"term").expect("mutable term");
            let mutable_posting = mutable_term.postings().next().expect("mutable posting");
            let frozen_term = frozen.find_term(1, b"term").expect("frozen term");
            let frozen_posting = frozen_term.postings().next().expect("frozen posting");

            assert_eq!(
                mutable_term
                    .positions(mutable_posting)
                    .expect("mutable owner positions")
                    .collect::<Vec<_>>(),
                [1, 4]
            );
            assert_eq!(
                frozen_term
                    .positions(frozen_posting)
                    .expect("frozen owner positions")
                    .collect::<Vec<_>>(),
                [1, 4]
            );
            assert!(frozen_term.positions(mutable_posting).is_none());
            assert!(mutable_term.positions(frozen_posting).is_none());
        }

        apply_positioned(&mut delta, 1, "second", b"term", &[9])?;
        assert_eq!(delta.physical_document_count(), 2);
        assert_eq!(delta.live_document_count(), 2);
        assert_eq!(frozen.segment().physical_document_count(), 1);
        assert_eq!(frozen.live_document_count(), 1);
        assert_eq!(
            frozen
                .find_term(1, b"term")
                .expect("held frozen term")
                .live_doc_freq(),
            1
        );
        assert_eq!(frozen.live_total_tokens(1), Some(2));
        Ok(())
    }

    #[test]
    fn frozen_empty_epoch_drops_reset_retained_chain_storage() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        for docid in 0..260 {
            apply_positioned(
                &mut delta,
                docid,
                &format!("doc-{docid}"),
                b"term",
                &[docid],
            )?;
        }
        assert!(!delta.posting_arena.slots.is_empty());
        assert!(!delta.position_arena.slots.is_empty());

        let populated = delta.freeze(8);
        let populated_term = populated.find_term(1, b"term").expect("frozen term");
        let populated_postings = populated_term.postings().collect::<Vec<_>>();
        assert_eq!(populated_postings.len(), 260);
        for (expected_docid, posting) in (0_u32..260).zip(populated_postings) {
            assert_eq!(posting.global_docid, expected_docid);
            assert_eq!(
                populated_term
                    .positions(posting)
                    .expect("frozen multi-block position")
                    .collect::<Vec<_>>(),
                [expected_docid]
            );
        }

        delta.reset_after_seal(0)?;
        assert_eq!(delta.posting_arena.active_blocks, 0);
        assert_eq!(delta.position_arena.active_blocks, 0);
        assert!(
            !delta.posting_arena.slots.is_empty(),
            "the mutable writer should retain reusable capacity"
        );

        let frozen = delta.freeze(9);
        assert!(frozen.segment.posting_arena.blocks.is_empty());
        assert!(frozen.segment.posting_arena.slots.is_empty());
        assert!(frozen.segment.position_arena.blocks.is_empty());
        assert!(frozen.segment.position_arena.slots.is_empty());
        assert_eq!(frozen.live_document_count(), 0);
        Ok(())
    }

    #[test]
    fn budget_accuracy_has_an_independent_component_oracle() -> Result<(), DeltaError> {
        let term = b"oracle";
        let document_id = "oracle-id";
        let positions = [1, 2, 3];
        let expected = expected_single_positioned_bytes(term, document_id);

        for (budget, expected_seal) in [
            (expected - 1, true),
            (expected, true),
            (expected + 1, false),
        ] {
            let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, budget)?;
            let applied = apply_positioned(&mut delta, 0, document_id, term, &positions)?;
            assert_eq!(applied.bytes_used, expected);
            assert_eq!(applied.seal_required, expected_seal);
            assert_eq!(delta.should_seal(), expected_seal);
        }

        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        apply_positioned(&mut delta, 0, document_id, term, &positions)?;
        apply_positioned(&mut delta, 1, document_id, term, &[4])?;
        let second_row_bytes = size_of::<BlockMeta>()
            + 2 * size_of::<PostingRecord>()
            + size_of::<BlockMeta>()
            + 4 * size_of::<u32>()
            + size_of::<u32>()
            + size_of::<DocId>()
            + document_id.len()
            + size_of::<u32>()
            + size_of::<u32>()
            + 3 * (size_of::<u32>() + size_of::<u8>())
            + size_of::<Option<u64>>()
            + size_of::<Option<Vec<u8>>>()
            + size_of::<u64>();
        assert_eq!(delta.bytes_used(), expected + second_row_bytes);
        Ok(())
    }

    #[test]
    fn budget_accepts_whole_document_and_ignores_retained_reset_capacity() -> Result<(), DeltaError>
    {
        let positions = (0_u32..300).collect::<Vec<_>>();
        let mut probe = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let used = apply_positioned(&mut probe, 0, "large", b"large", &positions)?.bytes_used;

        let mut exact = DeltaSegment::new(TEST_SCHEMA, 0, used)?;
        let exact_apply = apply_positioned(&mut exact, 0, "large", b"large", &positions)?;
        assert_eq!(exact_apply.bytes_used, used);
        assert!(exact_apply.seal_required);
        assert!(exact.should_seal());
        assert_eq!(exact.physical_document_count(), 1);
        let posting = exact
            .find_term(1, b"large")
            .expect("large term")
            .postings()
            .next()
            .expect("large posting");
        assert_eq!(posting.frequency, 300);

        let mut roomy = DeltaSegment::new(TEST_SCHEMA, 0, used + 1)?;
        let roomy_apply = apply_positioned(&mut roomy, 0, "large", b"large", &positions)?;
        assert!(!roomy_apply.seal_required);
        assert!(!roomy.should_seal());

        let reserved = exact.bytes_reserved();
        assert!(reserved >= exact.bytes_used());
        exact.reset_after_seal(u64::from(DOC_ORDS_PER_LEASE))?;
        assert_eq!(exact.bytes_used(), 0);
        assert!(exact.bytes_reserved() <= reserved);
        assert!(exact.bytes_reserved() > 0);
        assert!(!exact.should_seal());
        Ok(())
    }

    #[test]
    fn upsert_delete_and_readd_keep_physical_history_but_filter_live_rows() -> Result<(), DeltaError>
    {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        apply_positioned(&mut delta, 0, "A", b"term", &[0])?;
        let replacement = apply_positioned(&mut delta, 1, "A", b"term", &[1])?;
        assert_eq!(replacement.replaced_delta_docid, Some(0));
        assert_eq!(delta.probe_id("A"), Some(1));
        assert!(delta.is_tombstoned(0));
        assert!(!delta.is_tombstoned(1));

        assert_eq!(delta.delete_delta_id("A"), Some(1));
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
        assert_eq!(delta.delete_delta_id("A"), None);
        assert_eq!(delta.probe_id("A"), None);
        assert!(delta.is_tombstoned(1));
        assert_eq!(delta.live_document_count(), 0);

        let readd = apply_positioned(&mut delta, 2, "A", b"term", &[2])?;
        assert_eq!(readd.replaced_delta_docid, None);
        assert_eq!(delta.probe_id("A"), Some(2));
        assert_eq!(delta.probe_id("sealed-only-miss"), None);
        assert_eq!(delta.physical_document_count(), 3);
        assert_eq!(delta.live_document_count(), 1);
        assert_eq!(delta.live_total_tokens(1), Some(1));

        let term = delta.find_term(1, b"term").expect("term view");
        assert_eq!(term.physical_doc_freq(), 3);
        assert_eq!(term.live_doc_freq(), 1);
        assert_eq!(
            delta
                .live_documents()
                .map(|(docid, id)| (docid, id.as_str()))
                .collect::<Vec<_>>(),
            [(2, "A")]
        );
        Ok(())
    }

    #[test]
    fn tombstones_cover_word_and_lease_boundaries() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        for global_docid in [0, 63, 64, DOC_ORDS_PER_LEASE - 1] {
            apply_positioned(
                &mut delta,
                global_docid,
                "boundary-id",
                b"boundary",
                &[global_docid],
            )?;
        }
        assert!(delta.is_tombstoned(0));
        assert!(delta.is_tombstoned(63));
        assert!(delta.is_tombstoned(64));
        assert!(!delta.is_tombstoned(DOC_ORDS_PER_LEASE - 1));
        assert_eq!(
            delta.delete_delta_id("boundary-id"),
            Some(DOC_ORDS_PER_LEASE - 1)
        );
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
        assert!(delta.is_tombstoned(DOC_ORDS_PER_LEASE - 1));
        assert_eq!(delta.tombstone_words.len(), TOMBSTONE_WORDS_PER_LEASE);
        assert_eq!(delta.live_document_count(), 0);
        Ok(())
    }

    #[test]
    fn semantic_rejections_are_atomic() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let initial = delta.memory_stats();
        let valid_norms = norms(0, 2, 0);
        let valid_positions = [0, 1];
        let valid_posting = DeltaTermPosting {
            field_ord: 1,
            term: b"valid",
            frequency: 2,
            positions: Some(&valid_positions),
        };

        assert!(matches!(
            delta.apply_document(0, DocId::from(""), &valid_norms, &[valid_posting]),
            Err(DeltaError::EmptyDocumentId)
        ));
        assert_eq!(delta.memory_stats(), initial);
        assert!(matches!(
            delta.apply_document(
                DOC_ORDS_PER_LEASE,
                DocId::from("outside"),
                &valid_norms,
                &[valid_posting]
            ),
            Err(DeltaError::DocumentOutsideLease { .. })
        ));
        assert_eq!(delta.memory_stats(), initial);
        assert!(matches!(
            delta.apply_document(
                0,
                DocId::from("few-norms"),
                &valid_norms[..2],
                &[valid_posting]
            ),
            Err(DeltaError::FieldNormCount { .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        let mut bad_norms = valid_norms;
        bad_norms[1].fieldnorm_id = bad_norms[1].fieldnorm_id.wrapping_add(1);
        assert!(matches!(
            delta.apply_document(0, DocId::from("bad-norm"), &bad_norms, &[valid_posting]),
            Err(DeltaError::FieldNormMismatch { .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        let mut wrong_order = valid_norms;
        wrong_order.swap(0, 1);
        assert!(matches!(
            delta.apply_document(
                0,
                DocId::from("wrong-order"),
                &wrong_order,
                &[valid_posting]
            ),
            Err(DeltaError::FieldNormOrder { .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        let descending = [2, 1];
        let invalid_cases = [
            DeltaTermPosting {
                field_ord: 1,
                term: b"missing-positions",
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: 0,
                term: b"forbidden-positions",
                frequency: 2,
                positions: Some(&valid_positions),
            },
            DeltaTermPosting {
                field_ord: 1,
                term: b"count-mismatch",
                frequency: 1,
                positions: Some(&valid_positions),
            },
            DeltaTermPosting {
                field_ord: 1,
                term: b"descending",
                frequency: 2,
                positions: Some(&descending),
            },
            DeltaTermPosting {
                field_ord: 3,
                term: b"stored-only",
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: 0,
                term: b"zero-frequency",
                frequency: 0,
                positions: None,
            },
        ];
        for posting in invalid_cases {
            assert!(
                delta
                    .apply_document(0, DocId::from("invalid"), &valid_norms, &[posting])
                    .is_err()
            );
            assert_eq!(delta.memory_stats(), initial);
        }

        let duplicate = [valid_posting, valid_posting];
        assert!(matches!(
            delta.apply_document(0, DocId::from("duplicate"), &valid_norms, &duplicate),
            Err(DeltaError::DuplicateDocumentTerm { .. })
        ));
        assert_eq!(delta.memory_stats(), initial);

        apply_positioned(&mut delta, 0, "accepted", b"valid", &valid_positions)?;
        let accepted = delta.memory_stats();
        assert!(matches!(
            apply_positioned(&mut delta, 0, "out-of-order", b"valid", &valid_positions),
            Err(DeltaError::DocumentOrder { .. })
        ));
        assert_eq!(delta.memory_stats(), accepted);
        assert!(matches!(
            delta.reset_after_seal(1),
            Err(DeltaError::MisalignedLeaseBase { .. })
        ));
        assert_eq!(delta.memory_stats(), accepted);
        assert_eq!(delta.lease_base(), 0);
        Ok(())
    }

    #[test]
    fn term_length_boundary_matches_the_sealed_contract() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let accepted_term = vec![b'a'; MAX_TERM_BYTES];
        delta.apply_document(
            0,
            DocId::from("max-term"),
            &norms(0, 0, 1),
            &[DeltaTermPosting {
                field_ord: 2,
                term: &accepted_term,
                frequency: 1,
                positions: None,
            }],
        )?;
        assert!(delta.find_term(2, &accepted_term).is_some());
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());

        let before = delta.memory_stats();
        let rejected_term = vec![b'b'; MAX_TERM_BYTES + 1];
        assert!(matches!(
            delta.apply_document(
                1,
                DocId::from("oversized-term"),
                &norms(0, 0, 1),
                &[DeltaTermPosting {
                    field_ord: 2,
                    term: &rejected_term,
                    frequency: 1,
                    positions: None,
                }]
            ),
            Err(DeltaError::TermTooLarge { .. })
        ));
        assert_eq!(delta.memory_stats(), before);
        Ok(())
    }

    #[test]
    fn same_lease_reset_drops_all_logical_state_before_reuse() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        apply_positioned(&mut delta, 7, "old", b"old-term", &[1, 3])?;
        apply_positioned(&mut delta, 8, "old", b"old-term", &[5])?;
        let retained_before = delta.bytes_reserved();
        delta.reset_after_seal(0)?;

        assert_eq!(delta.bytes_used(), 0);
        assert_eq!(delta.next_docid_floor(), 9);
        assert!(delta.bytes_reserved() <= retained_before);
        assert_eq!(delta.probe_id("old"), None);
        assert!(delta.find_term(1, b"old-term").is_none());
        assert!(!delta.is_tombstoned(7));
        assert_eq!(delta.raw_field_length(1, 7), None);
        assert!(matches!(
            apply_positioned(&mut delta, 8, "overlap", b"new-term", &[9]),
            Err(DeltaError::DocumentBeforeFloor { .. })
        ));

        apply_positioned(&mut delta, 9, "new", b"new-term", &[11, 13])?;
        assert_eq!(delta.probe_id("new"), Some(9));
        assert_eq!(delta.live_document_count(), 1);
        let term = delta.find_term(1, b"new-term").expect("new term");
        let posting = term.postings().next().expect("new posting");
        assert_eq!(posting.global_docid, 9);
        assert_eq!(
            term.positions(posting)
                .expect("new positions")
                .collect::<Vec<_>>(),
            [11, 13]
        );
        assert_eq!(delta.raw_field_length(1, 9), Some(2));
        Ok(())
    }

    #[test]
    fn reset_reuses_allocation_across_many_lease_cycles() -> Result<(), DeltaError> {
        let mut delta = DeltaSegment::new(TEST_SCHEMA, 0, usize::MAX)?;
        let mut steady_reserved = None;
        for cycle in 0_u64..8 {
            let lease_base = cycle * u64::from(DOC_ORDS_PER_LEASE);
            assert_eq!(delta.lease_base(), lease_base);
            let lease_base_u32 = u32::try_from(lease_base).expect("test lease remains inside u32");
            for ordinal in 0_u32..128 {
                let global_docid = lease_base_u32 + ordinal;
                apply_positioned(&mut delta, global_docid, "same-id", b"soak", &[ordinal])?;
            }
            assert_eq!(delta.physical_document_count(), 128);
            assert_eq!(delta.live_document_count(), 1);
            assert_eq!(
                delta
                    .find_term(1, b"soak")
                    .expect("soak term")
                    .physical_doc_freq(),
                128
            );
            let stats = delta.memory_stats();
            tracing::info!(
                cycle,
                bytes_used = stats.bytes_used,
                bytes_reserved = stats.bytes_reserved,
                term_count = stats.term_count,
                physical_document_count = stats.physical_document_count,
                live_document_count = stats.live_document_count,
                posting_blocks = stats.posting_blocks,
                position_blocks = stats.position_blocks,
                "delta arena reset/reuse soak cycle"
            );
            let reserved = delta.bytes_reserved();
            if let Some(expected) = steady_reserved {
                assert_eq!(
                    reserved, expected,
                    "retained bytes drifted in cycle {cycle}"
                );
            } else {
                steady_reserved = Some(reserved);
            }

            let next_lease = lease_base + u64::from(DOC_ORDS_PER_LEASE);
            delta.reset_after_seal(next_lease)?;
            assert_eq!(delta.bytes_used(), 0);
            assert!(delta.bytes_reserved() <= reserved);
            assert_eq!(delta.physical_document_count(), 0);
            assert_eq!(delta.live_document_count(), 0);
            assert!(delta.find_term(1, b"soak").is_none());
            assert!(!delta.should_seal());
        }
        Ok(())
    }

    #[test]
    fn constructor_and_lease_boundaries_are_explicit() -> Result<(), DeltaError> {
        assert!(matches!(
            DeltaSegment::new(TEST_SCHEMA, 0, 0),
            Err(DeltaError::ZeroBudget)
        ));
        assert!(matches!(
            DeltaSegment::new(TEST_SCHEMA, 1, 1),
            Err(DeltaError::MisalignedLeaseBase { .. })
        ));
        assert!(matches!(
            DeltaSegment::new(TEST_SCHEMA, MAX_GLOBAL_DOCID_EXCLUSIVE, 1),
            Err(DeltaError::LeaseOutOfRange { .. })
        ));

        let final_lease = MAX_GLOBAL_DOCID_EXCLUSIVE - u64::from(DOC_ORDS_PER_LEASE);
        let mut delta = DeltaSegment::with_default_budget(TEST_SCHEMA, final_lease)?;
        assert_eq!(delta.budget_bytes(), DEFAULT_DELTA_BUDGET_BYTES);
        assert_eq!(delta.lease_end(), MAX_GLOBAL_DOCID_EXCLUSIVE);
        delta.apply_document(u32::MAX, DocId::from("last"), &norms(0, 0, 0), &[])?;
        assert_eq!(delta.bytes_used(), delta.recompute_bytes_used());
        assert_eq!(delta.probe_id("last"), Some(u32::MAX));
        assert!(matches!(
            delta.reset_after_seal(0),
            Err(DeltaError::LeaseRegression { .. })
        ));
        Ok(())
    }

    #[test]
    fn shared_term_interner_reports_exact_delta_accounting() {
        let mut terms = TermInterner::new();
        let before = terms.bytes_used();
        let (alpha, alpha_bytes) = terms.intern_accounted(0, b"alpha");
        assert_eq!(terms.bytes_used() - before, alpha_bytes);

        let (alpha_again, duplicate_bytes) = terms.intern_accounted(0, b"alpha");
        assert_eq!(alpha_again, alpha);
        assert_eq!(duplicate_bytes, 0);

        let before_other_field = terms.bytes_used();
        let (other_field, other_field_bytes) = terms.intern_accounted(1, b"alpha");
        assert_ne!(other_field, alpha);
        assert_eq!(terms.bytes_used() - before_other_field, other_field_bytes);
        assert_eq!(terms.find(0, b"alpha"), Some(alpha));
        assert_eq!(terms.find(1, b"alpha"), Some(other_field));
    }
}
