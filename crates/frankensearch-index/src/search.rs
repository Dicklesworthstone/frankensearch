//! Brute-force top-k vector search over an opened [`crate::VectorIndex`].

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use ahash::AHashSet;

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::filter::{DocIdHashSet, ExcludeDocIdsFilter, SearchFilter, fnv1a_hash};
use frankensearch_core::{Cx, SearchError, SearchResult, VectorHit};
use rayon::prelude::*;

use crate::simd::{dot_i8x4_i8, quantize_f16_le_bytes_to_i8_controlled};
use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{
    PreparedQuery4bit, Quantization, VectorIndex, dot_4bit_prepared, dot_i8_i8, dot_i8_i8_maddubs,
    dot_product_f16_bytes_f32, dot_product_f32_bytes_f32, dot_product_f32_f32, maddubs_query_bias,
    pack_f16_le_bytes_to_4bit, prepare_4bit_query,
};

/// Record-count threshold where search switches from sequential to Rayon.
pub const PARALLEL_THRESHOLD: usize = 10_000;
/// Chunk size per Rayon task in the parallel scan path.
pub const PARALLEL_CHUNK_SIZE: usize = 1_024;
const INT8_PARALLEL_CHUNK_SIZE: usize = PARALLEL_CHUNK_SIZE * 4;
/// Selectivity threshold for the file-backed gather fast-path. A hash-addressable
/// filter must be smaller than `record_count / GATHER_SELECTIVITY_DIVISOR` before
/// we invert the loop and binary-search/gather the allowed hash ranges. The FSVI
/// crossover is lower than the in-memory gather because each allowed hash pays
/// `log2(N)` record-table probes; the short `filtered_gather` sweep keeps clear of
/// the measured 5% regression.
const GATHER_SELECTIVITY_DIVISOR: usize = 50;
/// Bound work between cooperative checks independently of Rayon chunk tuning.
pub(crate) const SCAN_CHECKPOINT_INTERVAL: usize = 256;

#[cfg(test)]
thread_local! {
    pub(super) static GATHER_SCAN_CHUNKS: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// The raw numeric API keeps its existing noncancellable kernel. Retained-tier
/// callers instantiate the same traversal with their context, so cancellation
/// cannot accidentally select a different scorer or visibility policy.
pub(crate) trait ScanCancellation: Copy + Send + Sync {
    fn checkpoint(self) -> SearchResult<()>;

    #[inline]
    fn checkpoint_at(self, offset: usize) -> SearchResult<()> {
        if offset.is_multiple_of(SCAN_CHECKPOINT_INTERVAL) {
            self.checkpoint()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct UncancelledScan;

impl ScanCancellation for UncancelledScan {
    #[inline]
    fn checkpoint(self) -> SearchResult<()> {
        Ok(())
    }

    // Keep the interval test out of the noncancellable numeric hot loops.
    #[inline]
    fn checkpoint_at(self, _offset: usize) -> SearchResult<()> {
        Ok(())
    }
}

impl ScanCancellation for &Cx {
    fn checkpoint(self) -> SearchResult<()> {
        if Cx::checkpoint(self).is_err() || self.is_cancel_requested() {
            return Err(SearchError::Cancelled {
                phase: "search_activation.exact".to_owned(),
                reason: "operation cancelled".to_owned(),
            });
        }
        Ok(())
    }
}

/// Release initializer ownership on success, cancellation, or unwinding.
struct Int8SlabBuildGuard<'a>(&'a AtomicBool);

impl Drop for Int8SlabBuildGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, AtomicOrdering::Release);
    }
}

/// Configurable parameters for vector search parallelism.
///
/// Controls when and how the brute-force scan switches from sequential
/// to Rayon-parallel execution. Use [`SearchParams::default()`] for the
/// standard settings (threshold = 10,000, chunk size = 1,024, parallel
/// enabled via `FRANKENSEARCH_PARALLEL_SEARCH` env var).
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Minimum record count to trigger parallel scanning.
    /// Below this threshold, search runs sequentially.
    pub parallel_threshold: usize,
    /// Number of records processed per Rayon chunk in parallel mode.
    pub parallel_chunk_size: usize,
    /// Whether parallel scanning is allowed at all. When `false`, search
    /// always runs sequentially regardless of record count.
    pub parallel_enabled: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            parallel_threshold: PARALLEL_THRESHOLD,
            parallel_chunk_size: PARALLEL_CHUNK_SIZE,
            parallel_enabled: parallel_search_enabled(),
        }
    }
}

/// A top-k result plus a typed classification when it is empty.
///
/// The invariant `zero_signal.is_some() == hits.is_empty()` lets callers
/// distinguish a legitimately empty answer (benign request/state outcome)
/// from an unusable semantic lane (availability failure) without inferring
/// anything from bare emptiness.
#[derive(Debug, Clone)]
pub struct ClassifiedHits {
    /// Ranked hits, best first. May be empty.
    pub hits: Vec<VectorHit>,
    /// `Some(reason)` if and only if `hits` is empty.
    pub zero_signal: Option<ZeroSignalReason>,
}

impl ClassifiedHits {
    /// An empty result with its typed reason.
    #[must_use]
    pub const fn empty(reason: ZeroSignalReason) -> Self {
        Self {
            hits: Vec::new(),
            zero_signal: Some(reason),
        }
    }
}

/// Classify request-scoped reasons after the caller has checked the dimension.
pub(crate) fn query_zero_signal(
    query: &[f32],
    limit: usize,
) -> SearchResult<Option<ZeroSignalReason>> {
    if limit == 0 {
        return Ok(Some(ZeroSignalReason::CallerRequestedZeroK));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(SearchError::InvalidConfig {
            field: "query".to_owned(),
            value: "<contains non-finite values>".to_owned(),
            reason: "query vector must be finite".to_owned(),
        });
    }
    Ok(query
        .iter()
        .all(|&value| value == 0.0)
        .then_some(ZeroSignalReason::ZeroNormQuery))
}

/// Validate winners without changing the raw numeric kernels. Only corrupt
/// winners require a generation census and wider retrieval; valid negative
/// scores must not stay hidden behind unusable zero-score rows.
pub(crate) fn classified_candidates(
    limit: usize,
    candidate_count: usize,
    search: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    usable: impl FnMut(&VectorHit) -> bool,
    mut census: impl FnMut() -> (bool, ZeroSignalReason),
) -> SearchResult<ClassifiedHits> {
    let mut classified = classified_candidates_controlled(
        limit,
        candidate_count,
        search,
        usable,
        || Ok(census()),
        UncancelledScan,
    )?;
    classified.hits = deduplicate_ranked_hits(classified.hits);
    Ok(classified)
}

// The cutoff is in physical rows. Keep the best surviving row per ID after
// ranking, validating its vector, and applying the requested limit.
pub(super) fn deduplicate_ranked_hits(mut hits: Vec<VectorHit>) -> Vec<VectorHit> {
    if hits.len() > 1 {
        let mut seen = AHashSet::with_capacity(hits.len());
        hits.retain(|hit| seen.insert(hit.doc_id.clone()));
    }
    hits
}

pub(super) fn classified_candidates_controlled(
    limit: usize,
    candidate_count: usize,
    mut search: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    mut usable: impl FnMut(&VectorHit) -> bool,
    mut census: impl FnMut() -> SearchResult<(bool, ZeroSignalReason)>,
    cancel: impl ScanCancellation,
) -> SearchResult<ClassifiedHits> {
    let mut candidate_limit = limit.min(candidate_count);
    let mut repairing = false;
    let mut signal_state = None;
    loop {
        cancel.checkpoint()?;
        let mut hits = search(candidate_limit)?;
        let before = hits.len();
        let mut offset = 0;
        let mut checkpoint = Ok(());
        hits.retain(|hit| {
            if checkpoint.is_ok() {
                checkpoint = cancel.checkpoint_at(offset);
                offset += 1;
            }
            checkpoint.is_ok() && hit.score.is_finite() && usable(hit)
        });
        checkpoint?;
        repairing |= hits.len() != before;
        if signal_state.is_none() && (repairing || hits.is_empty()) {
            signal_state = Some(census()?);
        }
        if let Some((false, reason)) = signal_state {
            return Ok(ClassifiedHits::empty(reason));
        }
        if !repairing || hits.len() >= limit || candidate_limit >= candidate_count {
            hits.truncate(limit);
            let zero_signal = signal_state
                .filter(|_| hits.is_empty())
                .map(|(_, reason)| reason);
            return Ok(ClassifiedHits { hits, zero_signal });
        }
        candidate_limit = candidate_limit
            .saturating_mul(2)
            .max(candidate_limit.saturating_add(1))
            .min(candidate_count);
    }
}

static PARALLEL_SEARCH_ENABLED_CACHE: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    index: usize,
    score: f32,
}

impl HeapEntry {
    const fn new(index: usize, score: f32) -> Self {
        Self { index, score }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap keeps the largest element at the top.
        // We define "largest" == "worst" so peek() returns the current cutoff.
        match score_key(self.score).total_cmp(&score_key(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

/// Largest dimension whose entire `[-127 * 127 * dim, 127 * 127 * dim]` integer
/// dot range is exactly representable as f32. The common 384-dimensional path can
/// therefore rank raw i32 values while preserving the shipped i32-to-f32 order.
const MAX_EXACT_I8_DOT_DIM: usize = 1_040;

/// Packed pass-1 ordering key for the int8 two-pass scan. Larger means worse, so
/// `BinaryHeap::peek` remains the cutoff. The high word reverses score order
/// (higher scores become smaller keys); the low word preserves the full `usize`
/// index as the deterministic tiebreak (lower indices are better).
type Int8HeapKey = u128;

#[allow(clippy::inline_always)]
#[inline(always)]
const fn int8_heap_key(index: usize, score: i32) -> Int8HeapKey {
    let ascending_score = score.cast_unsigned() ^ 0x8000_0000;
    let descending_score = !ascending_score;
    ((descending_score as u128) << usize::BITS) | index as u128
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn int8_heap_key_from_f32(index: usize, score: i32) -> Int8HeapKey {
    let score = score as f32;
    let bits = score.to_bits();
    let sign_mask = (bits >> 31).wrapping_neg() | 0x8000_0000;
    let ascending_score = bits ^ sign_mask;
    let descending_score = !ascending_score;
    (u128::from(descending_score) << usize::BITS) | index as u128
}

const fn int8_heap_index(key: Int8HeapKey) -> usize {
    key as usize
}

#[inline]
fn retain_int8_candidate(
    heap: &mut BinaryHeap<Int8HeapKey>,
    cutoff: &mut Int8HeapKey,
    candidate: Int8HeapKey,
    limit: usize,
) {
    if heap.len() < limit {
        heap.push(candidate);
        if heap.len() == limit {
            *cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
        }
    } else if candidate < *cutoff {
        let _ = heap.pop();
        heap.push(candidate);
        *cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
    }
}

impl VectorIndex {
    /// Brute-force cosine-similarity top-k search over all records.
    ///
    /// The query is expected to already be normalized for cosine similarity.
    /// The result is sorted by descending score with NaN-safe semantics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed vector slab contents.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_internal(
            query,
            limit,
            filter,
            PARALLEL_THRESHOLD,
            PARALLEL_CHUNK_SIZE,
            parallel_search_enabled(),
        )
    }

    /// Exact retrieval below an admitted tier, with bounded cooperative checks.
    /// No partial candidate or result set escapes on cancellation.
    pub(crate) fn search_top_k_cancellable(
        &self,
        cx: &Cx,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_with_params_cancellable(cx, query, limit, filter, SearchParams::default())
    }

    pub(crate) fn search_top_k_with_params_cancellable(
        &self,
        cx: &Cx,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        let result = self.search_top_k_controlled::<true>(query, limit, filter, params, cx);
        // Cancellation wins over a decode/filter error occurring at the same
        // boundary, and no success can escape after the final filter callback.
        ScanCancellation::checkpoint(cx)?;
        result
    }

    /// Classify a search over a retained, admitted v2 owner. Admission has
    /// already checked the query and the owner cannot have a resident WAL.
    pub(crate) fn classify_owner_search(
        &self,
        cx: &Cx,
        query: &[f32],
        limit: usize,
        native_ann: bool,
        search: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    ) -> SearchResult<ClassifiedHits> {
        ScanCancellation::checkpoint(cx)?;
        if limit == 0 {
            return Ok(ClassifiedHits::empty(
                ZeroSignalReason::CallerRequestedZeroK,
            ));
        }
        if query.iter().all(|&component| component == 0.0) {
            return Ok(ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery));
        }
        let result = classified_candidates_controlled(
            limit,
            self.record_count(),
            search,
            |hit| self.classified_hit_is_usable(hit),
            || self.owner_search_census(cx, native_ann),
            cx,
        );
        ScanCancellation::checkpoint(cx)?;
        result
    }

    fn owner_search_census(
        &self,
        cancel: impl ScanCancellation,
        native_ann: bool,
    ) -> SearchResult<(bool, ZeroSignalReason)> {
        let mut reason = if self.record_count() == 0 {
            ZeroSignalReason::NewlyCreatedEmpty
        } else {
            ZeroSignalReason::AllTombstoned
        };
        for row in 0..self.record_count() {
            cancel.checkpoint_at(row)?;
            if !self.is_deleted(row) {
                reason = ZeroSignalReason::NoUsableVectors;
                if self.is_vector_usable(row) {
                    return Ok((
                        true,
                        if native_ann {
                            ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors
                        } else {
                            reason
                        },
                    ));
                }
            }
        }
        Ok((false, reason))
    }

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Behaves like [`Self::search_top_k`] with three fail-closed differences
    /// that align the exact path with the ANN path:
    /// - a query containing NaN or infinite components is rejected with
    ///   [`SearchError::InvalidConfig`] instead of silently scoring garbage
    ///   (parity with `HnswIndex`);
    /// - unusable stored vectors cannot contribute ranks or hide usable
    ///   neighbors;
    /// - an empty result always carries a typed
    ///   [`ZeroSignalReason`], so a legitimate empty answer is
    ///   distinguishable from an unusable semantic lane.
    ///
    /// Winning vectors are checked for a finite, non-zero norm. Corrupt
    /// winners trigger bounded widening so they cannot hide valid neighbors;
    /// a generation census is needed only for empty or corrupt results.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_top_k`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_top_k_classified(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        self.search_top_k_classified_with_params(query, limit, filter, SearchParams::default())
    }

    /// Internal classified scan with an admitted caller's parallelism settings.
    pub(crate) fn search_top_k_classified_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<ClassifiedHits> {
        self.ensure_query_dimension(query)?;
        if let Some(reason) = query_zero_signal(query, limit)? {
            return Ok(ClassifiedHits::empty(reason));
        }
        classified_candidates(
            limit,
            self.record_count().saturating_add(self.wal_entries.len()),
            |candidate_limit| self.search_top_k_rows(query, candidate_limit, filter, params),
            |hit| self.classified_hit_is_usable(hit),
            || self.classified_search_census(filter.is_some()),
        )
    }

    /// Classified exact scoring for an already admitted query, retaining
    /// physical copies until unusable rows have been removed. Cancellation
    /// never returns a partial candidate batch.
    pub(crate) fn search_top_k_classified_cancellable(
        &self,
        cancel: impl ScanCancellation,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<ClassifiedHits> {
        cancel.checkpoint()?;
        self.ensure_query_dimension(query)?;
        if let Some(reason) = query_zero_signal(query, limit)? {
            return Ok(ClassifiedHits::empty(reason));
        }
        let result = classified_candidates_controlled(
            limit,
            self.record_count().saturating_add(self.wal_entries.len()),
            |candidate_limit| {
                self.search_top_k_controlled::<false>(
                    query,
                    candidate_limit,
                    filter,
                    params,
                    cancel,
                )
            },
            |hit| self.classified_hit_is_usable(hit),
            || self.classified_search_census_cancellable(cancel, filter.is_some()),
            cancel,
        );
        cancel.checkpoint()?;
        let mut classified = result?;
        classified.hits = deduplicate_ranked_hits(classified.hits);
        cancel.checkpoint()?;
        Ok(classified)
    }

    pub(super) fn classified_search_census_cancellable(
        &self,
        cancel: impl ScanCancellation,
        had_filter: bool,
    ) -> SearchResult<(bool, ZeroSignalReason)> {
        if self.wal_entries.is_empty() {
            let (usable, reason) = self.owner_search_census(cancel, false)?;
            return Ok((
                usable,
                if usable && had_filter {
                    ZeroSignalReason::FilterEliminatedAll
                } else {
                    reason
                },
            ));
        }
        let mut replaced = AHashSet::with_capacity(self.wal_entries.len());
        let mut usable_wal = false;
        for (offset, entry) in self.wal_entries.iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            replaced.insert(entry.doc_id.as_str());
            usable_wal = usable_wal || crate::vector_signal_usable(&entry.embedding);
            if usable_wal && had_filter {
                cancel.checkpoint()?;
                return Ok((true, ZeroSignalReason::FilterEliminatedAll));
            }
        }
        let mut has_usable = usable_wal;
        let mut all_main_shadowed = true;
        for row in 0..self.record_count() {
            cancel.checkpoint_at(row)?;
            if self.is_deleted(row) {
                continue;
            }
            match self.doc_id_at(row) {
                Ok(id) if replaced.contains(id) => {}
                Ok(_) => {
                    all_main_shadowed = false;
                    has_usable = has_usable || self.is_vector_usable(row);
                }
                Err(_) => all_main_shadowed = false,
            }
            if has_usable && !all_main_shadowed {
                cancel.checkpoint()?;
                return Ok((
                    true,
                    if had_filter {
                        ZeroSignalReason::FilterEliminatedAll
                    } else {
                        ZeroSignalReason::NoUsableVectors
                    },
                ));
            }
        }
        cancel.checkpoint()?;
        let reason = if has_usable && had_filter {
            ZeroSignalReason::FilterEliminatedAll
        } else if usable_wal && all_main_shadowed {
            ZeroSignalReason::WalOnlyNoLiveRecords
        } else {
            ZeroSignalReason::NoUsableVectors
        };
        Ok((has_usable, reason))
    }

    /// Rank visible physical copies so their caller can validate them before collapsing IDs.
    pub(super) fn search_top_k_rows(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_controlled::<false>(query, limit, filter, params, UncancelledScan)
    }

    pub(crate) fn classified_hit_is_usable(&self, hit: &VectorHit) -> bool {
        usize::try_from(hit.index).is_ok_and(|index| {
            // Resolved hits use a contiguous public index space;
            // only the internal heap uses the WAL sentinel bit.
            if index >= self.record_count() {
                self.wal_entries
                    .get(index - self.record_count())
                    .is_some_and(|entry| crate::vector_signal_usable(&entry.embedding))
            } else {
                self.is_vector_usable(index)
            }
        })
    }

    pub(crate) fn classified_search_census(&self, had_filter: bool) -> (bool, ZeroSignalReason) {
        if self.wal_entries.is_empty() {
            let state = self.zero_signal_state();
            return (
                state.usable_vector_count > 0,
                state.empty_result_reason(had_filter),
            );
        }
        // A durable replacement can precede the old main row's tombstone.
        // Neither that stale vector nor main-only counts may attest the
        // availability of the current WAL-resident generation.
        let replaced: AHashSet<_> = self
            .wal_entries
            .iter()
            .map(|entry| entry.doc_id.as_str())
            .collect();
        let usable_wal = self
            .wal_entries
            .iter()
            .any(|entry| crate::vector_signal_usable(&entry.embedding));
        let has_usable = usable_wal
            || (0..self.record_count()).any(|row| {
                !self.is_deleted(row)
                    && self.doc_id_at(row).is_ok_and(|id| !replaced.contains(id))
                    && self.is_vector_usable(row)
            });
        let reason = if has_usable && had_filter {
            ZeroSignalReason::FilterEliminatedAll
        } else if usable_wal
            && (0..self.record_count()).all(|row| {
                self.is_deleted(row) || self.doc_id_at(row).is_ok_and(|id| replaced.contains(id))
            })
        {
            ZeroSignalReason::WalOnlyNoLiveRecords
        } else {
            ZeroSignalReason::NoUsableVectors
        };
        (has_usable, reason)
    }

    #[cfg(feature = "ann")]
    pub(crate) fn classified_ann_census(&self) -> (bool, ZeroSignalReason) {
        let (has_usable, reason) = self.classified_search_census(false);
        // A WAL-only miss comes from exact WAL scoring, outside the graph.
        if has_usable && reason == ZeroSignalReason::NoUsableVectors {
            (true, ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors)
        } else {
            (has_usable, reason)
        }
    }

    /// Exact top-k over the persisted main slab only.
    ///
    /// HNSW uses this crate-private lane to repair a native underfill with the
    /// same quantization decoder, dot-product implementation, tombstone
    /// semantics, score ordering, physical row identities, and post-top-k
    /// document-ID deduplication as canonical `VectorIndex` search. Resident
    /// WAL entries and their supersession rules are deliberately excluded:
    /// `TwoTierIndex` merges them exactly once after ANN candidate retrieval.
    #[cfg(feature = "ann")]
    pub(crate) fn search_main_top_k(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_main_top_k_raw(query, limit)
            .map(deduplicate_ranked_hits)
    }

    /// Raw physical top-k over the persisted main slab only.
    ///
    /// Unlike [`Self::search_main_top_k`], this does not apply document-ID
    /// deduplication. Neither main-only lane applies resident-WAL supersession.
    /// `TwoTierIndex` uses this raw lane only when an ANN underfill must be
    /// repaired before main and WAL candidates are ranked together through the
    /// canonical result resolver.
    #[cfg(feature = "ann")]
    pub(crate) fn search_main_top_k_raw(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let heap = self.scan_main_top_k_heap(query, limit)?;
        let mut winners = heap.into_vec();
        winners.sort_unstable_by(compare_best_first);
        winners
            .into_iter()
            .map(|winner| {
                let index =
                    u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: winner.index.to_string(),
                        reason: "winner index exceeds u32 range for VectorHit".to_owned(),
                    })?;
                Ok(VectorHit {
                    index,
                    score: winner.score,
                    doc_id: self.doc_id_at(winner.index)?.into(),
                })
            })
            .collect()
    }

    #[cfg(feature = "ann")]
    fn scan_main_top_k_heap(
        &self,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        self.ensure_query_dimension(query)?;
        if limit == 0 || self.record_count() == 0 {
            return Ok(BinaryHeap::new());
        }
        if parallel_search_enabled() && self.record_count() >= PARALLEL_THRESHOLD {
            self.scan_parallel(query, limit, None, PARALLEL_CHUNK_SIZE, UncancelledScan)
        } else {
            self.scan_sequential(query, limit, None, UncancelledScan)
        }
    }

    /// Brute-force cosine-similarity top-k search with configurable parallelism.
    ///
    /// Behaves identically to [`search_top_k`](Self::search_top_k) but uses the
    /// caller-supplied [`SearchParams`] instead of the compiled-in defaults.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed vector slab contents.
    pub fn search_top_k_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_internal(
            query,
            limit,
            filter,
            params.parallel_threshold,
            params.parallel_chunk_size,
            params.parallel_enabled,
        )
    }

    /// Bench-only: force the old per-document filtered scan, bypassing the
    /// selective-filter gather fast-path.
    #[doc(hidden)]
    pub fn bench_scan_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.ensure_query_dimension(query)?;
        let has_main = self.record_count() > 0;
        let has_wal = !self.wal_entries.is_empty();
        if limit == 0 || (!has_main && !has_wal) {
            return Ok(Vec::new());
        }
        let use_parallel = parallel_search_enabled() && self.record_count() >= PARALLEL_THRESHOLD;
        let mut heap = if has_main {
            if use_parallel {
                self.scan_parallel(query, limit, filter, PARALLEL_CHUNK_SIZE, UncancelledScan)?
            } else {
                self.scan_sequential(query, limit, filter, UncancelledScan)?
            }
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if has_wal {
            heap = self.merge_wal_top_k(
                query,
                heap,
                limit,
                filter,
                SearchParams::default(),
                UncancelledScan,
            )?;
        }
        self.resolve_hits::<true>(heap, UncancelledScan)
    }

    /// Bench-only: force the selective-filter gather path, ignoring the production
    /// selectivity gate.
    #[doc(hidden)]
    pub fn bench_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<Vec<VectorHit>> {
        self.ensure_query_dimension(query)?;
        if limit == 0 || (self.record_count() == 0 && self.wal_entries.is_empty()) {
            return Ok(Vec::new());
        }
        let mut heap = if self.record_count() > 0 {
            if let Some(ids) = filter.candidate_doc_ids() {
                self.scan_gather_doc_ids(
                    ids,
                    filter,
                    query,
                    limit,
                    Some(PARALLEL_CHUNK_SIZE),
                    UncancelledScan,
                )?
            } else if let Some(allowed) = filter.candidate_hashes() {
                self.scan_gather_hashes(
                    allowed,
                    query,
                    limit,
                    Some(PARALLEL_CHUNK_SIZE),
                    UncancelledScan,
                )?
            } else {
                return self.bench_scan_filtered(query, limit, Some(filter));
            }
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if !self.wal_entries.is_empty() {
            heap = self.merge_wal_top_k(
                query,
                heap,
                limit,
                Some(filter),
                SearchParams::default(),
                UncancelledScan,
            )?;
        }
        self.resolve_hits::<true>(heap, UncancelledScan)
    }

    // The input heap contains main-slab candidates only. Ordinary appends
    // tombstone superseded rows, but a durable WAL can survive a crash before
    // that tombstone reaches the main file. Dropping stale winners only in
    // resolve_sorted_entries is too late: they already displaced live rows.
    fn merge_wal_top_k(
        &self,
        query: &[f32],
        mut heap: BinaryHeap<HeapEntry>,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        let shadowed_winner = {
            let mut wal_hashes = AHashSet::with_capacity(self.wal_entries.len());
            let mut wal_ids = AHashSet::with_capacity(self.wal_entries.len());
            for (offset, entry) in self.wal_entries.iter().enumerate() {
                cancel.checkpoint_at(offset)?;
                wal_hashes.insert(entry.doc_id_hash);
                wal_ids.insert(entry.doc_id.as_str());
            }
            let mut shadowed = false;
            for (offset, winner) in heap.iter().enumerate() {
                cancel.checkpoint_at(offset)?;
                let hash = self.record_at(winner.index)?.doc_id_hash;
                if wal_hashes.contains(&hash) && wal_ids.contains(self.doc_id_at(winner.index)?) {
                    shadowed = true;
                    break;
                }
            }
            // Release these temporary sets before allocating the exclusion view.
            shadowed
        };
        if shadowed_winner {
            // The exclusion view builds hash sets, so its input must also
            // stop promptly when a large WAL is cancelled during preparation.
            let mut cancelled = None;
            let exclusions = self
                .wal_entries
                .iter()
                .enumerate()
                .map_while(|(offset, entry)| match cancel.checkpoint_at(offset) {
                    Ok(()) => Some(entry.doc_id.as_str()),
                    Err(error) => {
                        cancelled = Some(error);
                        None
                    }
                });
            let visible_main = ExcludeDocIdsFilter::new(filter, exclusions);
            if let Some(error) = cancelled {
                return Err(error);
            }
            cancel.checkpoint()?;
            // Only an actual superseded winner takes the extra scan. Normal
            // WAL appends preserve the existing selective gather and SIMD path.
            // Exact-ID candidate bounds survive exclusions: their gather still
            // checks the full predicate. Hash-only allow-lists cannot safely
            // represent this view when a collision occurs.
            heap = if let Some(gathered) =
                self.try_gather_filtered(query, limit, Some(&visible_main), params, cancel)?
            {
                gathered
            } else if params.parallel_enabled && self.record_count() >= params.parallel_threshold {
                self.scan_parallel(
                    query,
                    limit,
                    Some(&visible_main),
                    params.parallel_chunk_size.max(1),
                    cancel,
                )?
            } else {
                self.scan_sequential(query, limit, Some(&visible_main), cancel)?
            };
        }
        // Apply the caller's original filter to the live WAL, NOT visible_main,
        // which would exclude the very replacements that must remain searchable.
        self.scan_wal(query, &mut heap, limit, filter, cancel)?;
        cancel.checkpoint()?;
        Ok(heap)
    }

    fn search_top_k_internal(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        parallel_threshold: usize,
        parallel_chunk_size: usize,
        parallel_enabled: bool,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_controlled::<true>(
            query,
            limit,
            filter,
            SearchParams {
                parallel_threshold,
                parallel_chunk_size,
                parallel_enabled,
            },
            UncancelledScan,
        )
    }

    pub(super) fn search_top_k_controlled<const DEDUPLICATE: bool>(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        cancel.checkpoint()?;
        self.ensure_query_dimension(query)?;
        let has_main = self.record_count() > 0;
        let has_wal = !self.wal_entries.is_empty();
        if limit == 0 || (!has_main && !has_wal) {
            return Ok(Vec::new());
        }
        let chunk_size = params.parallel_chunk_size.max(1);
        let use_parallel =
            params.parallel_enabled && self.record_count() >= params.parallel_threshold;
        let total_candidate_upper_bound =
            self.record_count().saturating_add(self.wal_entries.len());

        // Full-recall requests should avoid top-k heap churn.
        // When the caller asks for all available candidates (`k >= total`),
        // collect-and-sort is measurably faster than maintaining a size-k heap.
        if filter.is_none() && limit >= total_candidate_upper_bound {
            let mut winners = if has_main {
                if use_parallel {
                    self.scan_parallel_collect_all(query, chunk_size, cancel)?
                } else {
                    self.scan_range_collect_all(0, self.record_count(), query, cancel)?
                }
            } else {
                Vec::new()
            };
            if has_wal {
                self.scan_wal_collect_all(query, &mut winners, cancel)?;
            }
            // `limit_all` scan-all path: `winners` can hold every match. Above a
            // threshold the final sort dominates, and a parallel sort pays
            // (measured ~2.81× at 50k winners, `winners_sort` bench); below it the
            // rayon overhead is not worth it, so stay serial. Bit-identical either
            // way — `compare_best_first` is a strict total order.
            cancel.checkpoint()?;
            if use_parallel && winners.len() >= PAR_SORT_THRESHOLD {
                winners.par_sort_unstable_by(compare_best_first);
            } else {
                winners.sort_unstable_by(compare_best_first);
            }
            cancel.checkpoint()?;
            return self.resolve_sorted_entries::<DEDUPLICATE>(winners, cancel);
        }

        let mut heap = if has_main {
            if let Some(gathered) =
                self.try_gather_filtered(query, limit, filter, params, cancel)?
            {
                gathered
            } else if use_parallel {
                self.scan_parallel(query, limit, filter, chunk_size, cancel)?
            } else {
                self.scan_sequential(query, limit, filter, cancel)?
            }
        } else {
            let max_wal = self.wal_entries.len();
            BinaryHeap::with_capacity(limit.min(max_wal).saturating_add(1))
        };

        // Resolve main/WAL visibility before the final bounded selection.
        if has_wal {
            heap = self.merge_wal_top_k(query, heap, limit, filter, params, cancel)?;
        }

        self.resolve_hits::<DEDUPLICATE>(heap, cancel)
    }

    /// Select up to `k * candidate_multiplier.max(1)` physical rows by their
    /// int8 dot products, then rank them by their exact f16 scores. Rounding
    /// can tie or reorder candidates near the cutoff and exclude a true winner.
    /// Use [`Self::search_top_k`] for exhaustive selection. Indexes with a
    /// resident WAL or F32 vectors already use that exact scan.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed slab data.
    pub fn search_top_k_int8_two_pass(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_rows(query, k, candidate_multiplier)
            .map(deduplicate_ranked_hits)
    }

    pub(super) fn search_top_k_int8_rows(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_rows_controlled(query, k, candidate_multiplier, UncancelledScan)
    }

    pub(super) fn search_top_k_int8_rows_controlled(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        // Production keeps the EXACT-int8 pass-1. The `vpmaddubs` kernel (bd-b5wl) is 1.23× faster
        // in isolation (decidable) and recall-exact, but its **scan-level** win is only ~1.02–1.11×
        // (Amdahl-shrunk) and is NOT robustly decidable under fleet contention: two null-controlled
        // runs on `hetzner1` disagreed — 0.9023 (median below null p5, a clear win) then 0.9821
        // (inside the null floor). Shipping the approximate kernel as default on a marginal,
        // contention-dependent effect fails the gate, so it stays behind
        // `bench_search_top_k_int8_two_pass_maddubs`. Retry = worker isolation (same as cod's int8
        // micro-opt block). See docs/NEGATIVE_EVIDENCE.md 2026-07-10.
        self.search_top_k_int8_two_pass_impl::<false, false>(query, k, candidate_multiplier, cancel)
    }

    /// Exact pre-row-block implementation retained only for same-binary
    /// performance comparisons. Production callers should use
    /// [`Self::search_top_k_int8_two_pass`].
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_orig(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<false, false>(
            query,
            k,
            candidate_multiplier,
            UncancelledScan,
        )
        .map(deduplicate_ranked_hits)
    }

    /// Four-row query-decode-reuse candidate retained only so the null-controlled
    /// negative measurement stays reproducible. Production callers use
    /// [`Self::search_top_k_int8_two_pass`].
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_row_block_candidate(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<true, false>(
            query,
            k,
            candidate_multiplier,
            UncancelledScan,
        )
        .map(deduplicate_ranked_hits)
    }

    /// `vpmaddubs` pass-1 kernel candidate (bd-b5wl), retained for the same-binary A/B.
    /// Bit-identical *ranking* to [`Self::search_top_k_int8_two_pass`] on realistic quantized data
    /// (proven recall in `simd::tests::maddubs_pass1_preserves_f32_recall_under_real_saturation`);
    /// the pass-1 int8 dot is the approximate `dot_i8_i8_maddubs` (see its saturation caveat).
    #[doc(hidden)]
    pub fn bench_search_top_k_int8_two_pass_maddubs(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_impl::<false, true>(
            query,
            k,
            candidate_multiplier,
            UncancelledScan,
        )
        .map(deduplicate_ranked_hits)
    }

    fn search_top_k_int8_two_pass_impl<const ROW_BLOCKED: bool, const MADDUBS: bool>(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        cancel.checkpoint()?;
        let count = self.record_count();
        // Fall back to the exact scan for anything this fast path does not cover.
        if k == 0
            || count == 0
            || !self.wal_entries.is_empty()
            || self.quantization() != Quantization::F16
        {
            return self.search_top_k_controlled::<false>(
                query,
                k,
                None,
                SearchParams::default(),
                cancel,
            );
        }
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }

        let dim = self.dimension();
        let candidate_count = k
            .saturating_mul(candidate_multiplier.max(1))
            .min(count)
            .max(k.min(count));
        let query_i8 = quantize_i8_query(query, cancel)?;
        // Per-query bias `128·Σq` for the `MADDUBS` pass-1 kernel; unused (0) otherwise.
        let q_bias128 = if MADDUBS {
            maddubs_query_bias(&query_i8, dim)
        } else {
            0
        };
        let slab = self.int8_slab(cancel)?;

        // Pass 1: bounded-heap int8 scan keeping the top `candidate_count`.
        // The int8 dot is cheap enough that exact-scan sized chunks overproduce
        // local top-N heaps; larger chunks keep enough Rayon tasks while shrinking
        // the post-scan merge fan-in.
        let candidate_heap = if count < PARALLEL_THRESHOLD {
            if ROW_BLOCKED {
                self.int8_scan_range(slab, &query_i8, 0, count, candidate_count, cancel)?
            } else {
                self.int8_scan_range_orig::<MADDUBS>(
                    slab,
                    &query_i8,
                    q_bias128,
                    0,
                    count,
                    candidate_count,
                    cancel,
                )?
            }
        } else {
            let chunk_count = count.div_ceil(INT8_PARALLEL_CHUNK_SIZE);
            let partials: SearchResult<Vec<BinaryHeap<Int8HeapKey>>> = (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * INT8_PARALLEL_CHUNK_SIZE;
                    let end = (start + INT8_PARALLEL_CHUNK_SIZE).min(count);
                    if ROW_BLOCKED {
                        self.int8_scan_range(slab, &query_i8, start, end, candidate_count, cancel)
                    } else {
                        self.int8_scan_range_orig::<MADDUBS>(
                            slab,
                            &query_i8,
                            q_bias128,
                            start,
                            end,
                            candidate_count,
                            cancel,
                        )
                    }
                })
                .collect();
            merge_int8_partial_heaps(partials?, candidate_count, cancel)?
        };

        self.rescore_int8_candidates(query, k, candidate_heap, cancel)
    }

    fn rescore_int8_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidate_heap: BinaryHeap<Int8HeapKey>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        cancel.checkpoint()?;
        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as `search_top_k`, so the final order is identical
        // whenever pass-1 retained the true top-k.
        let stride = self.dimension() * 2;
        let mut heap = BinaryHeap::with_capacity(k.min(candidate_heap.len()));
        for (offset, candidate) in candidate_heap.into_iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            let index = int8_heap_index(candidate);
            let vector_offset = self.vectors_offset + index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(index, score), k);
        }
        cancel.checkpoint()?;
        self.resolve_hits::<false>(heap, cancel)
    }

    fn int8_scan_range_orig<const MADDUBS: bool>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        q_bias128: i32,
        start: usize,
        end: usize,
        limit: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<Int8HeapKey>> {
        cancel.checkpoint()?;
        if limit == 0 {
            return Ok(BinaryHeap::new());
        }
        if self.dimension() <= MAX_EXACT_I8_DOT_DIM {
            self.int8_scan_range_orig_with_key::<MADDUBS, _>(
                slab,
                query_i8,
                q_bias128,
                start,
                end,
                limit,
                int8_heap_key,
                cancel,
            )
        } else {
            self.int8_scan_range_orig_with_key::<MADDUBS, _>(
                slab,
                query_i8,
                q_bias128,
                start,
                end,
                limit,
                int8_heap_key_from_f32,
                cancel,
            )
        }
    }

    /// Exact `1948a65` per-row scan retained for the in-binary ORIGINAL arm. `MADDUBS` swaps the
    /// pass-1 int8 dot to the approximate `vpmaddubs` kernel (bd-b5wl); `q_bias128 = 128·Σq` is then
    /// live, else ignored. Production is `MADDUBS = false` → byte-identical to the shipped scan.
    fn int8_scan_range_orig_with_key<const MADDUBS: bool, F>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        q_bias128: i32,
        start: usize,
        end: usize,
        limit: usize,
        make_key: F,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<Int8HeapKey>>
    where
        F: Fn(usize, i32) -> Int8HeapKey,
    {
        let dim = self.dimension();
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = Int8HeapKey::MAX;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * dim;

        for index in start..end {
            cancel.checkpoint_at(index - start)?;
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + dim];
                let dot = if MADDUBS {
                    dot_i8_i8_maddubs(stored, query_i8, q_bias128)
                } else {
                    dot_i8_i8(stored, query_i8)
                };
                let candidate = make_key(index, dot);
                if heap.len() < limit {
                    heap.push(candidate);
                    if heap.len() == limit {
                        cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
                    }
                } else if candidate < cutoff {
                    let _ = heap.pop();
                    heap.push(candidate);
                    cutoff = heap.peek().copied().unwrap_or(Int8HeapKey::MAX);
                }
            }
            flags_offset += 16;
            slab_offset += dim;
        }
        cancel.checkpoint()?;
        Ok(heap)
    }

    /// Bounded-heap int8 scan of records `[start, end)` over the int8 `slab`
    /// (index-aligned with the record table), skipping tombstoned records via the
    /// same flag check + cutoff fast-path as the exact `scan_range_chunk`.
    fn int8_scan_range(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        start: usize,
        end: usize,
        limit: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<Int8HeapKey>> {
        cancel.checkpoint()?;
        if limit == 0 {
            return Ok(BinaryHeap::new());
        }
        if self.dimension() <= MAX_EXACT_I8_DOT_DIM {
            self.int8_scan_range_with_key(slab, query_i8, start, end, limit, int8_heap_key, cancel)
        } else {
            self.int8_scan_range_with_key(
                slab,
                query_i8,
                start,
                end,
                limit,
                int8_heap_key_from_f32,
                cancel,
            )
        }
    }

    fn int8_scan_range_with_key<F>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        start: usize,
        end: usize,
        limit: usize,
        make_key: F,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<Int8HeapKey>>
    where
        F: Fn(usize, i32) -> Int8HeapKey,
    {
        let dim = self.dimension();
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = Int8HeapKey::MAX;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * dim;

        let mut index = start;
        while index + 4 <= end {
            cancel.checkpoint_at(index - start)?;
            let flags0 = u16::from_le_bytes([self.data[flags_offset], self.data[flags_offset + 1]]);
            let flags1 =
                u16::from_le_bytes([self.data[flags_offset + 16], self.data[flags_offset + 17]]);
            let flags2 =
                u16::from_le_bytes([self.data[flags_offset + 32], self.data[flags_offset + 33]]);
            let flags3 =
                u16::from_le_bytes([self.data[flags_offset + 48], self.data[flags_offset + 49]]);
            let flags = [flags0, flags1, flags2, flags3];

            if ((flags0 | flags1 | flags2 | flags3) & 0x0001) == 0 {
                let stored_rows = &slab[slab_offset..slab_offset + 4 * dim];
                let scores = dot_i8x4_i8(stored_rows, query_i8);
                for (lane, score) in scores.into_iter().enumerate() {
                    retain_int8_candidate(
                        &mut heap,
                        &mut cutoff,
                        make_key(index + lane, score),
                        limit,
                    );
                }
            } else {
                for (lane, flags) in flags.into_iter().enumerate() {
                    if (flags & 0x0001) == 0 {
                        let row_offset = slab_offset + lane * dim;
                        let stored = &slab[row_offset..row_offset + dim];
                        retain_int8_candidate(
                            &mut heap,
                            &mut cutoff,
                            make_key(index + lane, dot_i8_i8(stored, query_i8)),
                            limit,
                        );
                    }
                }
            }

            index += 4;
            flags_offset += 64;
            slab_offset += 4 * dim;
        }

        while index < end {
            cancel.checkpoint_at(index - start)?;
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + dim];
                retain_int8_candidate(
                    &mut heap,
                    &mut cutoff,
                    make_key(index, dot_i8_i8(stored, query_i8)),
                    limit,
                );
            }
            index += 1;
            flags_offset += 16;
            slab_offset += dim;
        }
        cancel.checkpoint()?;
        Ok(heap)
    }

    /// Lazily quantize the F16 main slab. All callers share one builder, while
    /// contenders can cancel without waiting for that builder's whole corpus.
    /// Cancellation drops private output and releases the claim for a retry.
    fn int8_slab(&self, cancel: impl ScanCancellation) -> SearchResult<&[i8]> {
        let mut yielded = false;
        loop {
            cancel.checkpoint()?;
            if let Some(slab) = self.vectors_i8.get() {
                return Ok(slab);
            }
            if self
                .vectors_i8_building
                .compare_exchange(
                    false,
                    true,
                    AtomicOrdering::Acquire,
                    AtomicOrdering::Relaxed,
                )
                .is_ok()
            {
                // Declare the guard before the private Vec, so a failed build
                // releases its allocation before another caller can acquire.
                let _building = Int8SlabBuildGuard(&self.vectors_i8_building);
                if let Some(slab) = self.vectors_i8.get() {
                    cancel.checkpoint()?;
                    return Ok(slab);
                }
                cancel.checkpoint()?;
                let byte_len = self.record_count() * self.dimension() * 2;
                let slab = quantize_f16_le_bytes_to_i8_controlled(
                    &self.data[self.vectors_offset..self.vectors_offset + byte_len],
                    |offset| cancel.checkpoint_at(offset),
                )?;
                cancel.checkpoint()?;
                // Initialization only moves a complete Vec; no traversal runs
                // inside OnceLock, and the claim excludes other slab builders.
                let slab = self.vectors_i8.get_or_init(|| slab);
                cancel.checkpoint()?;
                return Ok(slab);
            }
            if yielded {
                // This CPU-bound API is synchronous. A short sleep prevents
                // contenders from busy-spinning while keeping checks bounded.
                std::thread::sleep(std::time::Duration::from_millis(1));
            } else {
                std::thread::yield_now();
                yielded = true;
            }
        }
    }

    /// Select candidates using packed signed 4-bit vectors, then rank them by
    /// their exact f16 scores. As with [`Self::search_top_k_int8_two_pass`],
    /// the multiplier controls how many physical rows survive approximate
    /// selection. Indexes with a resident WAL or F32 vectors use the exact scan.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, and `SearchError::IndexCorrupted` for
    /// malformed slab data.
    pub fn search_top_k_4bit_two_pass(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        if k == 0
            || count == 0
            || !self.wal_entries.is_empty()
            || self.quantization() != Quantization::F16
        {
            return self.search_top_k(query, k, None);
        }
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }

        let dim = self.dimension();
        let bytes_per_vector = dim.div_ceil(2);
        let candidate_count = k
            .saturating_mul(candidate_multiplier.max(1))
            .min(count)
            .max(k.min(count));
        let query_packed = pack_4bit_query(query);
        let query_prepared = prepare_4bit_query(&query_packed);
        let slab = self.nibbles_slab();

        let candidate_heap = if count < PARALLEL_THRESHOLD {
            self.nibble_scan_range(
                slab,
                &query_prepared,
                bytes_per_vector,
                0,
                count,
                candidate_count,
            )
        } else {
            let chunk_count = count.div_ceil(PARALLEL_CHUNK_SIZE);
            let partials: Vec<BinaryHeap<HeapEntry>> = (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * PARALLEL_CHUNK_SIZE;
                    let end = (start + PARALLEL_CHUNK_SIZE).min(count);
                    self.nibble_scan_range(
                        slab,
                        &query_prepared,
                        bytes_per_vector,
                        start,
                        end,
                        candidate_count,
                    )
                })
                .collect();
            merge_partial_heaps(partials, candidate_count, UncancelledScan)?
        };

        // Pass 2: exact f16 rescore (same bounded-heap selection + tie-break).
        let stride = dim * 2;
        let mut heap = BinaryHeap::with_capacity(k.min(candidate_heap.len()));
        for candidate in candidate_heap {
            let vector_offset = self.vectors_offset + candidate.index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), k);
        }
        self.resolve_hits::<true>(heap, UncancelledScan)
    }

    /// Bounded-heap 4-bit scan of records `[start, end)` over the packed nibble
    /// `slab` (index-aligned with the record table), skipping tombstoned records,
    /// with the same cutoff fast-path as the exact scan.
    fn nibble_scan_range(
        &self,
        slab: &[u8],
        query_prepared: &PreparedQuery4bit,
        bytes_per_vector: usize,
        start: usize,
        end: usize,
        limit: usize,
    ) -> BinaryHeap<HeapEntry> {
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = f32::NEG_INFINITY;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * bytes_per_vector;

        for index in start..end {
            let flags_bytes = &self.data[flags_offset..flags_offset + 2];
            let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);
            if (flags & 0x0001) == 0 {
                let stored = &slab[slab_offset..slab_offset + bytes_per_vector];
                let score = dot_4bit_prepared(stored, query_prepared) as f32;
                if heap.len() < limit || score_key(score) >= cutoff {
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                    if heap.len() >= limit
                        && let Some(&worst) = heap.peek()
                    {
                        cutoff = score_key(worst.score);
                    }
                }
            }
            flags_offset += 16;
            slab_offset += bytes_per_vector;
        }
        heap
    }

    /// Lazily build (once) the packed signed-4-bit quantization of the contiguous
    /// F16 main-vector region. Only called after the F16/no-WAL gate.
    fn nibbles_slab(&self) -> &[u8] {
        self.vectors_nibbles.get_or_init(|| {
            let count = self.record_count();
            let dim = self.dimension();
            let byte_len = count * dim * 2;
            pack_f16_le_bytes_to_4bit(
                &self.data[self.vectors_offset..self.vectors_offset + byte_len],
                dim,
            )
        })
    }

    fn scan_sequential(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        // Re-use the parallel chunk logic for sequential scan to benefit from optimizations.
        filter.map_or_else(
            || self.scan_range_chunk(0, self.record_count(), query, limit, cancel),
            |filter| {
                self.scan_range_chunk_filtered(0, self.record_count(), query, limit, filter, cancel)
            },
        )
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                filter.map_or_else(
                    || self.scan_range_chunk(start, end, query, limit, cancel),
                    |active_filter| {
                        self.scan_range_chunk_filtered(
                            start,
                            end,
                            query,
                            limit,
                            active_filter,
                            cancel,
                        )
                    },
                )
            })
            .collect();

        merge_partial_heaps(partial_heaps?, limit, cancel)
    }

    fn scan_parallel_collect_all(
        &self,
        query: &[f32],
        chunk_size: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<HeapEntry>> {
        cancel.checkpoint()?;
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial: SearchResult<Vec<Vec<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                self.scan_range_collect_all(start, end, query, cancel)
            })
            .collect();

        let partial = partial?;
        let mut total = 0;
        for (offset, chunk) in partial.iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            total += chunk.len();
        }
        let mut merged = Vec::with_capacity(total);
        for mut chunk in partial {
            cancel.checkpoint()?;
            merged.append(&mut chunk);
        }
        cancel.checkpoint()?;
        Ok(merged)
    }

    fn scan_range_collect_all(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<HeapEntry>> {
        cancel.checkpoint()?;
        let mut winners = Vec::with_capacity(end.saturating_sub(start));
        let dim = self.dimension();

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                        winners.push(HeapEntry::new(index, score));
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                        winners.push(HeapEntry::new(index, score));
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
        }
        cancel.checkpoint()?;
        Ok(winners)
    }

    fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Option<BinaryHeap<HeapEntry>>> {
        cancel.checkpoint()?;
        let Some(active_filter) = filter else {
            return Ok(None);
        };
        let count = self.record_count();
        let parallel_chunk_size = (params.parallel_enabled && count >= params.parallel_threshold)
            .then_some(params.parallel_chunk_size.max(1));
        if let Some(ids) = active_filter.candidate_doc_ids()
            && ids.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) < count
        {
            return self
                .scan_gather_doc_ids(
                    ids,
                    active_filter,
                    query,
                    limit,
                    parallel_chunk_size,
                    cancel,
                )
                .map(Some);
        }
        let Some(allowed) = active_filter.candidate_hashes() else {
            return Ok(None);
        };
        if count == 0 || allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        self.scan_gather_hashes(allowed, query, limit, parallel_chunk_size, cancel)
            .map(Some)
    }

    fn scan_gather_hashes(
        &self,
        allowed: &DocIdHashSet,
        query: &[f32],
        limit: usize,
        parallel_chunk_size: Option<usize>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let positions = self.gather_positions_for_hashes(allowed, cancel)?;
        self.scan_gather_positions(&positions, query, limit, parallel_chunk_size, cancel)
    }

    fn scan_gather_doc_ids(
        &self,
        ids: &HashSet<String>,
        filter: &dyn SearchFilter,
        query: &[f32],
        limit: usize,
        parallel_chunk_size: Option<usize>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        let mut positions = Vec::with_capacity(ids.len());
        for (offset, id) in ids.iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            let Some((start, end)) = self.hash_range(fnv1a_hash(id.as_bytes()))? else {
                continue;
            };
            // All physical copies must participate in top-k. A hash collision
            // is only a lookup candidate and never establishes ID membership.
            for index in start..end {
                cancel.checkpoint_at(index - start)?;
                if !self.is_deleted(index)
                    && self.doc_id_at(index)? == id
                    && filter.matches(id, None)
                {
                    positions.push(index);
                }
            }
        }
        cancel.checkpoint()?;
        positions.sort_unstable();
        cancel.checkpoint()?;
        self.scan_gather_positions(&positions, query, limit, parallel_chunk_size, cancel)
    }

    fn gather_positions_for_hashes(
        &self,
        allowed: &DocIdHashSet,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<usize>> {
        cancel.checkpoint()?;
        let mut positions = Vec::with_capacity(allowed.len());
        for (offset, &hash) in allowed.iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            let Some((start, end)) = self.hash_range(hash)? else {
                continue;
            };
            for index in start..end {
                cancel.checkpoint_at(index - start)?;
                let entry = self.record_at(index)?;
                if (entry.flags & 0x0001) == 0 {
                    positions.push(index);
                }
            }
        }
        cancel.checkpoint()?;
        positions.sort_unstable();
        cancel.checkpoint()?;
        Ok(positions)
    }

    pub(crate) fn hash_range(&self, hash: u64) -> SearchResult<Option<(usize, usize)>> {
        let count = self.record_count();
        let start = self.lower_bound_hash(hash, 0, count)?;
        if start == count || self.record_at(start)?.doc_id_hash != hash {
            return Ok(None);
        }
        let end = self.upper_bound_hash(hash, start + 1, count)?;
        Ok(Some((start, end)))
    }

    fn lower_bound_hash(&self, hash: u64, mut low: usize, mut high: usize) -> SearchResult<usize> {
        while low < high {
            let mid = low + (high - low) / 2;
            if self.record_at(mid)?.doc_id_hash < hash {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(low)
    }

    fn upper_bound_hash(&self, hash: u64, mut low: usize, mut high: usize) -> SearchResult<usize> {
        while low < high {
            let mid = low + (high - low) / 2;
            if self.record_at(mid)?.doc_id_hash <= hash {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(low)
    }

    fn scan_gather_positions(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
        parallel_chunk_size: Option<usize>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        if let Some(chunk_size) = parallel_chunk_size
            && positions.len() > chunk_size
        {
            let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = positions
                .par_chunks(chunk_size)
                .map(|chunk| self.gather_range(chunk, query, limit, cancel))
                .collect();
            return merge_partial_heaps(partials?, limit, cancel);
        }
        self.gather_range(positions, query, limit, cancel)
    }

    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        #[cfg(test)]
        GATHER_SCAN_CHUNKS.with(|observation| {
            if let Some(chunks) = observation.borrow_mut().as_mut() {
                chunks.push(positions.len());
            }
        });
        let mut heap = BinaryHeap::with_capacity(limit.min(positions.len()).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                for (offset, &index) in positions.iter().enumerate() {
                    cancel.checkpoint_at(offset)?;
                    let vector_offset = self.vectors_offset + index * stride;
                    let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                    let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                    if heap.len() < limit || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                        if heap.len() >= limit
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                for (offset, &index) in positions.iter().enumerate() {
                    cancel.checkpoint_at(offset)?;
                    let vector_offset = self.vectors_offset + index * stride;
                    let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                    let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                    if heap.len() < limit || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                        if heap.len() >= limit
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
            }
        }
        cancel.checkpoint()?;
        Ok(heap)
    }

    fn scan_range_chunk(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        let max_elements = end.saturating_sub(start);
        let mut heap = BinaryHeap::with_capacity(limit.min(max_elements).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                // Flags are at offset 14 in 16-byte record
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    // Check flags directly from mapped memory
                    // SAFETY: offset arithmetic is bounded by record_count checks in open()
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    let flags_bytes = &self.data[flags_offset..flags_offset + 2];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) == 0 {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    flags_offset += 16;
                    vector_offset += stride;
                }
            }
        }
        cancel.checkpoint()?;
        Ok(heap)
    }

    #[allow(clippy::too_many_arguments)] // One context accompanies the existing exact scan inputs.
    fn scan_range_chunk_filtered(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
        cancel: impl ScanCancellation,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        cancel.checkpoint()?;
        let max_elements = end.saturating_sub(start);
        let mut heap = BinaryHeap::with_capacity(limit.min(max_elements).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                let mut record_offset = self.records_offset + start * 16;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let hash_bytes = &self.data[record_offset..record_offset + 8];
                    let hash = u64::from_le_bytes([
                        hash_bytes[0],
                        hash_bytes[1],
                        hash_bytes[2],
                        hash_bytes[3],
                        hash_bytes[4],
                        hash_bytes[5],
                        hash_bytes[6],
                        hash_bytes[7],
                    ]);

                    let passed = if let Some(matches) = filter.matches_doc_id_hash(hash, None) {
                        matches
                    } else {
                        let doc_id = self.doc_id_at(index)?;
                        filter.matches(doc_id, None)
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
            Quantization::F32 => {
                let stride = dim * 4;
                let mut record_offset = self.records_offset + start * 16;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
                    cancel.checkpoint_at(index - start)?;
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let hash_bytes = &self.data[record_offset..record_offset + 8];
                    let hash = u64::from_le_bytes([
                        hash_bytes[0],
                        hash_bytes[1],
                        hash_bytes[2],
                        hash_bytes[3],
                        hash_bytes[4],
                        hash_bytes[5],
                        hash_bytes[6],
                        hash_bytes[7],
                    ]);

                    let passed = if let Some(matches) = filter.matches_doc_id_hash(hash, None) {
                        matches
                    } else {
                        let doc_id = self.doc_id_at(index)?;
                        filter.matches(doc_id, None)
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + stride];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query)?;
                        if heap.len() < limit || score_key(score) >= cutoff {
                            insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                            if heap.len() >= limit
                                && let Some(&worst) = heap.peek()
                            {
                                cutoff = score_key(worst.score);
                            }
                        }
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
        }
        cancel.checkpoint()?;
        Ok(heap)
    }

    fn scan_wal(
        &self,
        query: &[f32],
        heap: &mut BinaryHeap<HeapEntry>,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<()> {
        for (idx, entry) in self.wal_entries.iter().enumerate() {
            cancel.checkpoint_at(idx)?;
            if let Some(f) = filter {
                if let Some(matches) = f.matches_doc_id_hash(entry.doc_id_hash, None) {
                    if !matches {
                        continue;
                    }
                } else if !f.matches(&entry.doc_id, None) {
                    continue;
                }
            }
            let score = dot_product_f32_f32(&entry.embedding, query)?;
            // Guard: corrupt WAL embeddings can produce NaN/Inf scores that
            // poison the top-k sort. Skip them (matches two_tier ANN path).
            if !score.is_finite() {
                continue;
            }
            insert_candidate(heap, HeapEntry::new(to_wal_index(idx), score), limit);
        }
        cancel.checkpoint()?;
        Ok(())
    }

    fn scan_wal_collect_all(
        &self,
        query: &[f32],
        winners: &mut Vec<HeapEntry>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<()> {
        cancel.checkpoint()?;
        winners.reserve(self.wal_entries.len());
        for (idx, entry) in self.wal_entries.iter().enumerate() {
            cancel.checkpoint_at(idx)?;
            let score = dot_product_f32_f32(&entry.embedding, query)?;
            if !score.is_finite() {
                continue;
            }
            winners.push(HeapEntry::new(to_wal_index(idx), score));
        }
        cancel.checkpoint()?;
        Ok(())
    }

    fn resolve_hits<const DEDUPLICATE: bool>(
        &self,
        heap: BinaryHeap<HeapEntry>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        cancel.checkpoint()?;
        if heap.is_empty() {
            return Ok(Vec::new());
        }

        let mut winners = heap.into_vec();
        winners.sort_unstable_by(compare_best_first);
        cancel.checkpoint()?;
        self.resolve_sorted_entries::<DEDUPLICATE>(winners, cancel)
    }

    fn resolve_sorted_entries<const DEDUPLICATE: bool>(
        &self,
        winners: Vec<HeapEntry>,
        cancel: impl ScanCancellation,
    ) -> SearchResult<Vec<VectorHit>> {
        cancel.checkpoint()?;
        // Pre-build a hash set of WAL doc_id hashes for O(1) pre-screening
        // instead of O(W) linear scan per main-index winner. On hash match,
        // falls back to string verification for correctness.
        let mut wal_hashes = AHashSet::with_capacity(self.wal_entries.len());
        for (offset, entry) in self.wal_entries.iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            wal_hashes.insert(entry.doc_id_hash);
        }

        // Classified retrieval needs the physical copies until their vectors
        // have been checked. The ordinary numeric API deduplicates here.
        let mut seen = DEDUPLICATE.then(|| AHashSet::<String>::with_capacity(winners.len()));
        let mut hits = Vec::with_capacity(winners.len());
        for (offset, winner) in winners.into_iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            if is_wal_index(winner.index) {
                let wal_idx = from_wal_index(winner.index);
                let doc_id = &self.wal_entries[wal_idx].doc_id;
                // Skip WAL-vs-WAL duplicates (keep the first, i.e. highest-scored).
                if seen.as_mut().is_some_and(|ids| !ids.insert(doc_id.clone())) {
                    continue;
                }
                hits.push(self.resolve_wal_hit(&winner)?);
            } else {
                // Main index entry.
                if self.is_deleted(winner.index) {
                    continue;
                }
                let doc_id = self.doc_id_at(winner.index)?.to_owned();
                // Read pre-computed hash from record table instead of recomputing.
                let record = self.record_at(winner.index)?;
                let doc_id_hash = record.doc_id_hash;
                // O(1) hash pre-screen; only linear-scan on hash match.
                if wal_hashes.contains(&doc_id_hash) {
                    let mut has_wal_entry = false;
                    for (wal_offset, entry) in self.wal_entries.iter().enumerate() {
                        cancel.checkpoint_at(wal_offset)?;
                        if entry.doc_id_hash == doc_id_hash && entry.doc_id == doc_id {
                            has_wal_entry = true;
                            break;
                        }
                    }
                    if has_wal_entry {
                        continue;
                    }
                }
                // Skip main-vs-main duplicates (String-based for correctness).
                if seen.as_mut().is_some_and(|ids| !ids.insert(doc_id.clone())) {
                    continue;
                }
                let index_u32 =
                    u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: winner.index.to_string(),
                        reason: "winner index exceeds u32 range for VectorHit".to_owned(),
                    })?;
                hits.push(VectorHit {
                    index: index_u32,
                    score: winner.score,
                    doc_id: doc_id.into(),
                });
            }
        }

        cancel.checkpoint()?;
        Ok(hits)
    }

    fn resolve_wal_hit(&self, winner: &HeapEntry) -> SearchResult<VectorHit> {
        if !is_wal_index(winner.index) {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: winner.index.to_string(),
                reason: "winner index is not WAL-encoded".to_owned(),
            });
        }

        let wal_idx = from_wal_index(winner.index);
        let entry = self
            .wal_entries
            .get(wal_idx)
            .ok_or_else(|| SearchError::IndexCorrupted {
                path: self.path.clone(),
                detail: format!(
                    "WAL index {} out of bounds (wal_entries.len() = {})",
                    wal_idx,
                    self.wal_entries.len()
                ),
            })?;
        let virtual_index =
            self.record_count()
                .checked_add(wal_idx)
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "index".to_owned(),
                    value: wal_idx.to_string(),
                    reason: "WAL virtual index overflow".to_owned(),
                })?;
        let index_u32 = u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
            field: "index".to_owned(),
            value: virtual_index.to_string(),
            reason: "WAL entry index exceeds u32 range".to_owned(),
        })?;
        Ok(VectorHit {
            index: index_u32,
            score: winner.score,
            doc_id: entry.doc_id.as_str().into(),
        })
    }

    #[allow(clippy::missing_const_for_fn)]
    fn ensure_query_dimension(&self, query: &[f32]) -> SearchResult<()> {
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }
        Ok(())
    }
}

/// Quantize an f32 query to int8 using its own max-abs scale (a per-query constant
/// that does not change the dot-product ranking).
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_query(query: &[f32], cancel: impl ScanCancellation) -> SearchResult<Vec<i8>> {
    let mut max_abs = 0.0_f32;
    for (offset, value) in query.iter().enumerate() {
        cancel.checkpoint_at(offset)?;
        max_abs = max_abs.max(value.abs());
    }
    let scale = 127.0 / max_abs;
    let mut values = Vec::with_capacity(query.len());
    for (offset, &value) in query.iter().enumerate() {
        cancel.checkpoint_at(offset)?;
        values.push(if max_abs <= 0.0 {
            0
        } else {
            (value * scale).round().clamp(-127.0, 127.0) as i8
        });
    }
    cancel.checkpoint()?;
    Ok(values)
}

/// Quantize one component to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's
/// complement in the low 4 bits) given a scale.
#[inline]
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the cast
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack an f32 query into signed 4-bit nibbles, 2 dims/byte (low = even dim, high =
/// odd dim), using the query's own max-abs scale (a per-query constant that does not
/// change the dot-product ranking). Matches `pack_4bit_f16_bytes`.
fn pack_4bit_query(query: &[f32]) -> Vec<u8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let mut packed = vec![0_u8; query.len().div_ceil(2)];
    for (d, &x) in query.iter().enumerate() {
        let nib = nibble_of(x, scale);
        if d % 2 == 0 {
            packed[d / 2] |= nib;
        } else {
            packed[d / 2] |= nib << 4;
        }
    }
    packed
}

pub(crate) const fn score_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// Winners-count threshold above which the `limit_all` final sort uses a parallel
/// `par_sort_unstable_by` instead of the serial sort. Below it, rayon's spawn/merge
/// overhead is not amortized (the per-element comparison is cheap); at 50k winners
/// the parallel sort is ~2.81× faster (`winners_sort` bench). Bit-identical output.
const PAR_SORT_THRESHOLD: usize = 16_384;

// Strict total order: `score_key.total_cmp` then a unique-`index` tiebreak. Because
// no two distinct entries compare Equal, the `winners` sorts that use this run as
// `sort_unstable_by` (pdqsort, no scratch alloc) with output identical to a stable
// sort — a ~1.16× (top-k) to ~1.47× (limit_all, 50k winners) win on the final order.
fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
    }
}

fn candidate_is_better(left: HeapEntry, right: HeapEntry) -> bool {
    match score_key(left.score).total_cmp(&score_key(right.score)) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => left.index < right.index,
    }
}

fn insert_candidate(heap: &mut BinaryHeap<HeapEntry>, candidate: HeapEntry, limit: usize) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(candidate);
        return;
    }
    if let Some(&worst) = heap.peek()
        && candidate_is_better(candidate, worst)
    {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn merge_partial_heaps(
    partial_heaps: Vec<BinaryHeap<HeapEntry>>,
    limit: usize,
    cancel: impl ScanCancellation,
) -> SearchResult<BinaryHeap<HeapEntry>> {
    cancel.checkpoint()?;
    let mut total_elements = 0_usize;
    for (offset, heap) in partial_heaps.iter().enumerate() {
        cancel.checkpoint_at(offset)?;
        total_elements = total_elements.saturating_add(heap.len());
    }
    let capacity = limit.min(total_elements).saturating_add(1);
    let mut merged = BinaryHeap::with_capacity(capacity);
    for heap in partial_heaps {
        cancel.checkpoint()?;
        for (offset, entry) in heap.into_iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            insert_candidate(&mut merged, entry, limit);
        }
    }
    cancel.checkpoint()?;
    Ok(merged)
}

fn merge_int8_partial_heaps(
    partial_heaps: Vec<BinaryHeap<Int8HeapKey>>,
    limit: usize,
    cancel: impl ScanCancellation,
) -> SearchResult<BinaryHeap<Int8HeapKey>> {
    cancel.checkpoint()?;
    let mut total_elements = 0_usize;
    for (offset, heap) in partial_heaps.iter().enumerate() {
        cancel.checkpoint_at(offset)?;
        total_elements = total_elements.saturating_add(heap.len());
    }
    let mut merged = BinaryHeap::with_capacity(limit.min(total_elements).saturating_add(1));
    for heap in partial_heaps {
        cancel.checkpoint()?;
        for (offset, candidate) in heap.into_iter().enumerate() {
            cancel.checkpoint_at(offset)?;
            if merged.len() < limit {
                merged.push(candidate);
            } else if merged.peek().is_some_and(|&worst| candidate < worst) {
                let _ = merged.pop();
                merged.push(candidate);
            }
        }
    }
    cancel.checkpoint()?;
    Ok(merged)
}

fn parallel_search_enabled() -> bool {
    *PARALLEL_SEARCH_ENABLED_CACHE.get_or_init(|| {
        let value = std::env::var("FRANKENSEARCH_PARALLEL_SEARCH").ok();
        parse_parallel_search_env(value.as_deref())
    })
}

fn parse_parallel_search_env(value: Option<&str>) -> bool {
    value.is_none_or(|raw| {
        let normalized = raw.trim();
        !normalized.eq_ignore_ascii_case("0")
            && !normalized.eq_ignore_ascii_case("false")
            && !normalized.eq_ignore_ascii_case("no")
            && !normalized.eq_ignore_ascii_case("off")
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::test_fixtures::TempFixturePath;
    use crate::{Quantization, VectorIndex};
    use frankensearch_core::PredicateFilter;
    use proptest::prelude::*;

    fn temp_index_path(name: &str) -> TempFixturePath {
        TempFixturePath::new(&format!("search-{name}"), "index.fsvi")
    }

    fn write_index(path: &std::path::Path, rows: &[(&str, Vec<f32>)]) -> SearchResult<()> {
        let dimension =
            rows.first()
                .map(|(_, vec)| vec.len())
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "rows".to_owned(),
                    value: "[]".to_owned(),
                    reason: "rows must not be empty".to_owned(),
                })?;
        let mut writer =
            VectorIndex::create_with_revision(path, "hash", "test", dimension, Quantization::F16)?;
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector)?;
        }
        writer.finish()
    }

    fn create_rows(vectors: &[Vec<f32>]) -> Vec<(String, Vec<f32>)> {
        vectors
            .iter()
            .enumerate()
            .map(|(idx, vector)| (format!("doc-{idx:03}"), vector.clone()))
            .collect()
    }

    fn hit_ids(hits: &[VectorHit]) -> Vec<String> {
        hits.iter().map(|hit| hit.doc_id.to_string()).collect()
    }

    const CANCEL_AFTER_ROWS: usize = 17;
    const CANCELLATION_TEST_ROWS: usize = 4_096;

    fn cancellation_index(
        name: &str,
        quantization: Quantization,
        count: usize,
    ) -> (TempFixturePath, VectorIndex) {
        let path = temp_index_path(name);
        let mut writer = VectorIndex::create_with_revision(&path, "hash", "test", 2, quantization)
            .expect("cancellation fixture writer");
        for row in 0..count {
            let score = f32::from(u16::try_from(row % 31).expect("bounded score")) / 31.0;
            writer
                .write_record(&format!("doc-{row:05}"), &[score, 1.0 - score])
                .expect("write cancellation fixture");
        }
        writer.finish().expect("finish cancellation fixture");
        let index = VectorIndex::open(&path).expect("open cancellation fixture");
        (path, index)
    }

    fn assert_same_hits(actual: &[VectorHit], expected: &[VectorHit]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.doc_id, expected.doc_id);
            assert_eq!(actual.index, expected.index);
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        }
    }

    fn assert_bounded_cancellation<T>(result: &SearchResult<T>, visits: usize, workers: usize) {
        assert!(matches!(
            result,
            Err(SearchError::Cancelled { phase, .. }) if phase == "search_activation.exact"
        ));
        assert!(
            visits >= CANCEL_AFTER_ROWS,
            "cancellation followed real row work"
        );
        assert!(
            visits <= CANCEL_AFTER_ROWS + workers * SCAN_CHECKPOINT_INTERVAL,
            "cancelled scan kept traversing rows: {visits} visits with {workers} workers"
        );
    }

    /// Inject cancellation after actual traversal work in kernels without a
    /// filter callback. The interval and context check remain the production
    /// implementation; only the trigger and work counter are test-owned.
    #[derive(Clone, Copy)]
    struct CancelDuringRows<'a> {
        cx: &'a Cx,
        visits: &'a Mutex<usize>,
    }

    impl ScanCancellation for CancelDuringRows<'_> {
        fn checkpoint(self) -> SearchResult<()> {
            ScanCancellation::checkpoint(self.cx)
        }

        fn checkpoint_at(self, offset: usize) -> SearchResult<()> {
            {
                let mut visits = self.visits.lock().expect("row work counter");
                *visits += 1;
                if *visits == CANCEL_AFTER_ROWS {
                    self.cx.set_cancel_requested(true);
                }
            }
            ScanCancellation::checkpoint_at(self.cx, offset)
        }
    }

    #[test]
    fn cancellable_exact_stops_filter_work_and_retry_keeps_exact_results() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-worker scan pool");
        for quantization in [Quantization::F16, Quantization::F32] {
            let (_path, index) =
                cancellation_index("cancel-filter-work", quantization, CANCELLATION_TEST_ROWS);
            let query = [1.0, 0.0];
            let expected = index.search_top_k(&query, 23, None).expect("raw control");
            for parallel_enabled in [false, true] {
                // Both tiny chunks and a single large chunk must respect the
                // row bound; tuning may not stretch cancellation latency.
                for parallel_chunk_size in [37, CANCELLATION_TEST_ROWS] {
                    let cx = Cx::for_testing();
                    let visits = Arc::new(Mutex::new(0));
                    let filter_cx = cx.clone();
                    let filter_visits = Arc::clone(&visits);
                    let filter = PredicateFilter::new("cancel-after-work", move |_| {
                        {
                            let mut visits = filter_visits.lock().expect("filter work counter");
                            *visits += 1;
                            if *visits == CANCEL_AFTER_ROWS {
                                filter_cx.set_cancel_requested(true);
                            }
                        }
                        true
                    });
                    let params = SearchParams {
                        parallel_enabled,
                        parallel_threshold: 1,
                        parallel_chunk_size,
                    };
                    let result = pool.install(|| {
                        index.search_top_k_with_params_cancellable(
                            &cx,
                            &query,
                            23,
                            Some(&filter),
                            params,
                        )
                    });
                    assert_bounded_cancellation(
                        &result,
                        *visits.lock().expect("filter work counter"),
                        if parallel_enabled { 2 } else { 1 },
                    );

                    cx.set_cancel_requested(false);
                    let retry = pool
                        .install(|| {
                            index.search_top_k_with_params_cancellable(
                                &cx,
                                &query,
                                23,
                                Some(&filter),
                                params,
                            )
                        })
                        .expect("same-owner retry");
                    assert_same_hits(&retry, &expected);
                }
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn selective_gather_obeys_parallel_policy_and_preserves_exact_results() {
        use frankensearch_core::{BitsetFilter, IncludeDocIdsFilter};

        fn observe_chunks(
            pool: &rayon::ThreadPool,
            search: impl FnOnce() -> Vec<VectorHit> + Send,
        ) -> (Vec<VectorHit>, Vec<usize>) {
            pool.broadcast(|_| {
                GATHER_SCAN_CHUNKS.with(|chunks| *chunks.borrow_mut() = Some(Vec::new()));
            });
            let hits = pool.install(search);
            let chunks = pool
                .broadcast(|_| {
                    GATHER_SCAN_CHUNKS.with(|chunks| chunks.borrow_mut().take().unwrap())
                })
                .into_iter()
                .flatten()
                .collect();
            (hits, chunks)
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let selected = 64;
        let query = [1.0, 0.0];
        for quantization in [Quantization::F16, Quantization::F32] {
            let (_path, index) =
                cancellation_index("gather-policy", quantization, CANCELLATION_TEST_ROWS);
            let allowed: HashSet<_> = (0..selected).map(|i| format!("doc-{i:05}")).collect();
            let exact = IncludeDocIdsFilter::new(allowed.iter().cloned());
            let hashed = BitsetFilter::from_doc_ids(allowed.iter().cloned());
            let predicate =
                PredicateFilter::new("selected-row-reference", move |id| allowed.contains(id));
            let sequential = SearchParams {
                parallel_enabled: false,
                parallel_threshold: 0,
                parallel_chunk_size: 7,
            };
            let cx = Cx::for_testing();
            let (reference, chunks) = observe_chunks(&pool, || {
                index
                    .search_top_k_classified_cancellable(
                        &cx,
                        &query,
                        23,
                        Some(&predicate),
                        sequential,
                    )
                    .unwrap()
                    .hits
            });
            assert!(
                chunks.is_empty(),
                "the reference uses the full predicate scan"
            );
            for filter in [&exact as &dyn SearchFilter, &hashed as &dyn SearchFilter] {
                for (params, expected_chunk_size) in [
                    (sequential, None),
                    (
                        SearchParams {
                            parallel_enabled: true,
                            parallel_threshold: usize::MAX,
                            ..sequential
                        },
                        None,
                    ),
                    (
                        SearchParams {
                            parallel_enabled: true,
                            ..sequential
                        },
                        Some(7),
                    ),
                    (
                        SearchParams {
                            parallel_enabled: true,
                            parallel_chunk_size: 0,
                            ..sequential
                        },
                        Some(1),
                    ),
                    (
                        SearchParams {
                            parallel_enabled: true,
                            parallel_chunk_size: selected + 1,
                            ..sequential
                        },
                        None,
                    ),
                ] {
                    let (actual, chunks) = observe_chunks(&pool, || {
                        index
                            .search_top_k_classified_cancellable(
                                &cx,
                                &query,
                                23,
                                Some(filter),
                                params,
                            )
                            .unwrap()
                            .hits
                    });
                    assert_same_hits(&actual, &reference);
                    assert_eq!(chunks.iter().sum::<usize>(), selected);
                    if let Some(chunk_size) = expected_chunk_size {
                        assert_eq!(chunks.len(), selected.div_ceil(chunk_size));
                        assert!(chunks.iter().all(|&size| size > 0 && size <= chunk_size));
                    } else {
                        assert_eq!(chunks, [selected]);
                    }
                }
            }

            // A gathered pool larger than the old fixed 1024-row threshold
            // must still remain one sequential batch when parallelism is off.
            let positions: Vec<_> = (0..index.record_count()).collect();
            let reference = index.search_top_k(&query, 23, None).unwrap();
            let (actual, chunks) = observe_chunks(&pool, || {
                let heap = index
                    .scan_gather_positions(&positions, &query, 23, None, &cx)
                    .unwrap();
                index.resolve_hits::<true>(heap, &cx).unwrap()
            });
            assert_same_hits(&actual, &reference);
            assert_eq!(chunks, [positions.len()]);
        }
    }

    #[test]
    fn cancellable_exact_checks_unfiltered_collect_all_and_gather_rows() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-worker scan pool");
        for quantization in [Quantization::F16, Quantization::F32] {
            let (_path, index) =
                cancellation_index("cancel-exact-kernels", quantization, CANCELLATION_TEST_ROWS);
            let positions: Vec<usize> = (0..index.record_count()).collect();
            let hashes: DocIdHashSet = positions
                .iter()
                .map(|&row| index.record_at(row).expect("record").doc_id_hash)
                .collect();
            for lane in [
                "top-k",
                "collect-all",
                "parallel-top-k",
                "parallel-collect-all",
                "gather",
                "parallel-gather",
                "gather-positions",
            ] {
                let cx = Cx::for_testing();
                let visits = Mutex::new(0);
                let cancel = CancelDuringRows {
                    cx: &cx,
                    visits: &visits,
                };
                let result = pool.install(|| match lane {
                    "top-k" => index
                        .scan_range_chunk(0, index.record_count(), &[1.0, 0.0], 23, cancel)
                        .map(|_| ()),
                    "collect-all" => index
                        .scan_range_collect_all(0, index.record_count(), &[1.0, 0.0], cancel)
                        .map(|_| ()),
                    "parallel-top-k" => index
                        .scan_parallel(&[1.0, 0.0], 23, None, 1_024, cancel)
                        .map(|_| ()),
                    "parallel-collect-all" => index
                        .scan_parallel_collect_all(&[1.0, 0.0], 1_024, cancel)
                        .map(|_| ()),
                    "gather" => index
                        .gather_range(&positions, &[1.0, 0.0], 23, cancel)
                        .map(|_| ()),
                    "parallel-gather" => index
                        .scan_gather_positions(
                            &positions,
                            &[1.0, 0.0],
                            23,
                            Some(PARALLEL_CHUNK_SIZE),
                            cancel,
                        )
                        .map(|_| ()),
                    "gather-positions" => index
                        .gather_positions_for_hashes(&hashes, cancel)
                        .map(|_| ()),
                    _ => unreachable!("fixed lane list"),
                });
                assert_bounded_cancellation(
                    &result,
                    *visits.lock().expect("row work counter"),
                    if lane.starts_with("parallel") { 2 } else { 1 },
                );
            }
        }
    }

    #[test]
    fn cancellable_exact_checks_wal_work_and_does_not_mutate_visibility() {
        let (_path, mut index) = cancellation_index("cancel-wal", Quantization::F16, 0);
        let entries: Vec<_> = (0..CANCELLATION_TEST_ROWS)
            .map(|row| (format!("wal-{row:05}"), vec![1.0, 0.0]))
            .collect();
        index.append_batch(&entries).expect("append WAL fixture");
        let expected = index
            .search_top_k(&[1.0, 0.0], 23, None)
            .expect("WAL control");
        let cx = Cx::for_testing();
        let visits = Arc::new(Mutex::new(0));
        let filter_cx = cx.clone();
        let filter_visits = Arc::clone(&visits);
        let filter = PredicateFilter::new("cancel-WAL-after-work", move |_| {
            {
                let mut visits = filter_visits.lock().expect("WAL work counter");
                *visits += 1;
                if *visits == CANCEL_AFTER_ROWS {
                    filter_cx.set_cancel_requested(true);
                }
            }
            true
        });
        let result = index.search_top_k_cancellable(&cx, &[1.0, 0.0], 23, Some(&filter));
        assert_bounded_cancellation(&result, *visits.lock().expect("WAL work counter"), 1);
        cx.set_cancel_requested(false);
        let retry = index
            .search_top_k_cancellable(&cx, &[1.0, 0.0], 23, None)
            .expect("WAL retry");
        assert_same_hits(&retry, &expected);

        let visits = Mutex::new(0);
        let cancel = CancelDuringRows {
            cx: &cx,
            visits: &visits,
        };
        let mut winners = Vec::new();
        let result = index.scan_wal_collect_all(&[1.0, 0.0], &mut winners, cancel);
        assert_bounded_cancellation(&result, *visits.lock().expect("WAL work counter"), 1);
    }

    #[test]
    fn cancellable_exact_stops_result_materialization_and_heap_merging() {
        let (_path, index) =
            cancellation_index("cancel-results", Quantization::F16, CANCELLATION_TEST_ROWS);
        let cx = Cx::for_testing();
        let visits = Mutex::new(0);
        let winners: Vec<_> = (0..index.record_count())
            .map(|row| HeapEntry::new(row, 1.0))
            .collect();
        let result = index.resolve_sorted_entries::<true>(
            winners,
            CancelDuringRows {
                cx: &cx,
                visits: &visits,
            },
        );
        assert_bounded_cancellation(&result, *visits.lock().expect("row work counter"), 1);

        cx.set_cancel_requested(false);
        *visits.lock().expect("row work counter") = 0;
        let partials: Vec<_> = (0..4)
            .map(|chunk| {
                (0..1_024)
                    .map(|row| HeapEntry::new(chunk * 1_024 + row, 1.0))
                    .collect()
            })
            .collect();
        let result = merge_partial_heaps(
            partials,
            23,
            CancelDuringRows {
                cx: &cx,
                visits: &visits,
            },
        );
        assert_bounded_cancellation(&result, *visits.lock().expect("row work counter"), 1);
    }

    #[test]
    fn cancelled_exact_refuses_empty_fast_paths_and_final_filter_cancellation() {
        let (_empty_path, empty) = cancellation_index("cancel-empty", Quantization::F16, 0);
        let cx = Cx::for_testing();
        cx.set_cancel_requested(true);
        for limit in [0, 1] {
            assert!(matches!(
                empty.search_top_k_cancellable(&cx, &[1.0, 0.0], limit, None),
                Err(SearchError::Cancelled { .. })
            ));
        }

        let (_path, index) = cancellation_index("cancel-final-row", Quantization::F16, 1);
        cx.set_cancel_requested(false);
        let filter_cx = cx.clone();
        let filter = PredicateFilter::new("cancel-final-row", move |_| {
            filter_cx.set_cancel_requested(true);
            true
        });
        assert!(matches!(
            index.search_top_k_cancellable(&cx, &[1.0, 0.0], 1, Some(&filter)),
            Err(SearchError::Cancelled { .. })
        ));
    }

    #[test]
    fn cancellable_int8_cold_build_discards_both_partial_passes_and_retries() {
        #[derive(Clone, Copy)]
        struct CancelQuantizationPass<'a> {
            cx: &'a Cx,
            state: &'a Mutex<(usize, usize, bool)>,
            target_pass: usize,
        }

        impl ScanCancellation for CancelQuantizationPass<'_> {
            fn checkpoint(self) -> SearchResult<()> {
                ScanCancellation::checkpoint(self.cx)
            }

            fn checkpoint_at(self, offset: usize) -> SearchResult<()> {
                {
                    let mut state = self.state.lock().unwrap();
                    if offset == 0 && state.0 > 0 {
                        state.1 += 1;
                    }
                    state.0 = offset;
                    if state.1 == self.target_pass && offset > 0 {
                        state.2 = true;
                        drop(state);
                        self.cx.set_cancel_requested(true);
                    }
                }
                ScanCancellation::checkpoint_at(self.cx, offset)
            }
        }

        for target_pass in [0, 1] {
            let (_path, index) = cancellation_index(
                "cancel-int8-cold-build",
                Quantization::F16,
                CANCELLATION_TEST_ROWS,
            );
            let cx = Cx::for_testing();
            let state = Mutex::new((0, 0, false));
            let result = index.int8_slab(CancelQuantizationPass {
                cx: &cx,
                state: &state,
                target_pass,
            });
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(state.lock().unwrap().2, "cancel inside the selected pass");
            assert!(
                index.vectors_i8.get().is_none(),
                "never publish partial bytes"
            );
            assert!(!index.vectors_i8_building.load(AtomicOrdering::Acquire));

            cx.set_cancel_requested(false);
            let bytes = &index.data[index.vectors_offset
                ..index.vectors_offset + index.record_count() * index.dimension() * 2];
            let expected = crate::quantize_f16_le_bytes_to_i8(bytes);
            let actual = index
                .int8_slab(&cx)
                .expect("retry completes the cold build");
            assert_eq!(actual, expected);
            let published = actual.as_ptr();
            cx.set_cancel_requested(true);
            assert!(matches!(
                index.int8_slab(&cx),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(index.vectors_i8.get().unwrap().as_ptr(), published);
            assert_eq!(index.vectors_i8.get().unwrap().as_slice(), expected);
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn cancellable_int8_waiter_does_not_allocate_a_second_cold_slab() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Barrier, mpsc};
        use std::time::Duration;

        #[derive(Clone, Copy)]
        struct PausedBuilder<'a> {
            entered: &'a Barrier,
            release: &'a Barrier,
            paused: &'a AtomicBool,
        }

        impl ScanCancellation for PausedBuilder<'_> {
            fn checkpoint(self) -> SearchResult<()> {
                Ok(())
            }

            fn checkpoint_at(self, _offset: usize) -> SearchResult<()> {
                if !self.paused.swap(true, AtomicOrdering::AcqRel) {
                    self.entered.wait();
                    self.release.wait();
                }
                Ok(())
            }
        }

        #[derive(Clone, Copy)]
        struct CancelWaiting<'a> {
            cx: &'a Cx,
            checks: &'a AtomicUsize,
            quantizer_visits: &'a AtomicUsize,
        }

        impl ScanCancellation for CancelWaiting<'_> {
            fn checkpoint(self) -> SearchResult<()> {
                if self.checks.fetch_add(1, AtomicOrdering::Relaxed) == 3 {
                    self.cx.set_cancel_requested(true);
                }
                ScanCancellation::checkpoint(self.cx)
            }

            fn checkpoint_at(self, offset: usize) -> SearchResult<()> {
                self.quantizer_visits.fetch_add(1, AtomicOrdering::Relaxed);
                ScanCancellation::checkpoint_at(self.cx, offset)
            }
        }

        let (_path, index) = cancellation_index(
            "cancel-int8-contender",
            Quantization::F16,
            CANCELLATION_TEST_ROWS,
        );
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        let paused = AtomicBool::new(false);
        let checks = AtomicUsize::new(0);
        let quantizer_visits = AtomicUsize::new(0);
        let cx = Cx::for_testing();
        let (sender, receiver) = mpsc::channel();
        std::thread::scope(|scope| {
            let builder = scope.spawn(|| {
                index.int8_slab(PausedBuilder {
                    entered: &entered,
                    release: &release,
                    paused: &paused,
                })
            });
            entered.wait();
            assert!(index.vectors_i8.get().is_none());
            let waiter = scope.spawn(|| {
                let result = index
                    .int8_slab(CancelWaiting {
                        cx: &cx,
                        checks: &checks,
                        quantizer_visits: &quantizer_visits,
                    })
                    .map(<[i8]>::len);
                sender.send(result).unwrap();
            });
            // Always release the builder before asserting, so a broken waiter
            // fails the test rather than leaving a blocked scoped thread.
            let result = receiver.recv_timeout(Duration::from_secs(2));
            release.wait();
            let published = builder.join().unwrap().unwrap();
            waiter.join().unwrap();
            assert!(matches!(result, Ok(Err(SearchError::Cancelled { .. }))));
            assert_eq!(quantizer_visits.load(AtomicOrdering::Relaxed), 0);
            assert!(!index.vectors_i8_building.load(AtomicOrdering::Acquire));
            let bytes = &index.data[index.vectors_offset
                ..index.vectors_offset + index.record_count() * index.dimension() * 2];
            assert_eq!(published, crate::quantize_f16_le_bytes_to_i8(bytes));
            cx.set_cancel_requested(false);
            assert_eq!(index.int8_slab(&cx).unwrap().as_ptr(), published.as_ptr());
        });
    }

    #[test]
    fn cancellable_int8_candidate_scan_keeps_cached_bytes_and_exact_retry_bits() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for count in [CANCELLATION_TEST_ROWS, PARALLEL_THRESHOLD + 3] {
            let (_path, mut index) =
                cancellation_index("cancel-int8-pass1", Quantization::F16, count);
            index
                .soft_delete_batch(&["doc-00001", "doc-00030"])
                .unwrap();
            let query = [1.0, 0.0];
            let expected = index.search_top_k_int8_two_pass(&query, 23, 3).unwrap();
            let cached = index.vectors_i8.get().unwrap().clone();
            let cached_address = index.vectors_i8.get().unwrap().as_ptr();
            let cx = Cx::for_testing();
            let visits = Mutex::new(0);
            let result = pool.install(|| {
                index.search_top_k_int8_rows_controlled(
                    &query,
                    23,
                    3,
                    CancelDuringRows {
                        cx: &cx,
                        visits: &visits,
                    },
                )
            });
            assert_bounded_cancellation(
                &result,
                *visits.lock().unwrap(),
                if count >= PARALLEL_THRESHOLD { 2 } else { 1 },
            );
            assert_eq!(index.vectors_i8.get().unwrap(), &cached);
            assert_eq!(index.vectors_i8.get().unwrap().as_ptr(), cached_address);
            cx.set_cancel_requested(false);
            let retry = pool
                .install(|| index.search_top_k_int8_rows_controlled(&query, 23, 3, &cx))
                .unwrap();
            assert_same_hits(&retry, &expected);
        }
    }

    #[test]
    fn cancellable_int8_rescore_and_parallel_merge_stop_after_real_work() {
        let (_path, index) = cancellation_index(
            "cancel-int8-rescore",
            Quantization::F16,
            CANCELLATION_TEST_ROWS,
        );
        let candidates: BinaryHeap<_> = (0..index.record_count())
            .map(|row| int8_heap_key(row, 100))
            .collect();
        let query = [1.0, 0.0];
        let expected = index
            .rescore_int8_candidates(&query, 23, candidates.clone(), UncancelledScan)
            .unwrap();
        let cx = Cx::for_testing();
        let visits = Mutex::new(0);
        let result = index.rescore_int8_candidates(
            &query,
            23,
            candidates.clone(),
            CancelDuringRows {
                cx: &cx,
                visits: &visits,
            },
        );
        assert_bounded_cancellation(&result, *visits.lock().unwrap(), 1);
        cx.set_cancel_requested(false);
        assert_same_hits(
            &index
                .rescore_int8_candidates(&query, 23, candidates, &cx)
                .unwrap(),
            &expected,
        );

        let partials: Vec<BinaryHeap<_>> = (0..4)
            .map(|chunk| {
                (0..1024)
                    .map(|row| int8_heap_key(chunk * 1024 + row, 100))
                    .collect()
            })
            .collect();
        let expected = merge_int8_partial_heaps(partials.clone(), 23, UncancelledScan)
            .unwrap()
            .into_sorted_vec();
        *visits.lock().unwrap() = 0;
        let result = merge_int8_partial_heaps(
            partials.clone(),
            23,
            CancelDuringRows {
                cx: &cx,
                visits: &visits,
            },
        );
        assert_bounded_cancellation(&result, *visits.lock().unwrap(), 1);
        cx.set_cancel_requested(false);
        assert_eq!(
            merge_int8_partial_heaps(partials, 23, &cx)
                .unwrap()
                .into_sorted_vec(),
            expected,
        );
    }

    #[test]
    fn cancellable_int8_fallbacks_keep_f32_and_wal_visibility() {
        for (quantization, wal) in [
            (Quantization::F32, false),
            (Quantization::F16, true),
            (Quantization::F32, true),
        ] {
            let (_path, mut index) = cancellation_index(
                "cancel-int8-exact-fallback",
                quantization,
                if wal { 1 } else { CANCELLATION_TEST_ROWS },
            );
            if wal {
                let rows: Vec<_> = (0..CANCELLATION_TEST_ROWS)
                    .map(|row| (format!("wal-{row:05}"), vec![1.0, 0.5]))
                    .collect();
                index.append_batch(&rows).unwrap();
                // The old main row scores 1.0; its live WAL replacement scores
                // zero and must not conceal any of the new 0.5-score rows.
                index.append("doc-00000", &[-1.0, 0.0]).unwrap();
            }
            let query = [0.0, 1.0];
            let expected = index.search_top_k(&query, 23, None).unwrap();
            if wal {
                assert!(expected.iter().all(|hit| hit.doc_id != "doc-00000"));
            }
            let cx = Cx::for_testing();
            let visits = Mutex::new(0);
            let result = index.search_top_k_int8_rows_controlled(
                &query,
                23,
                3,
                CancelDuringRows {
                    cx: &cx,
                    visits: &visits,
                },
            );
            assert_bounded_cancellation(&result, *visits.lock().unwrap(), 1);
            assert!(index.vectors_i8.get().is_none());
            cx.set_cancel_requested(false);
            let retry = index
                .search_top_k_int8_rows_controlled(&query, 23, 3, &cx)
                .unwrap();
            assert_same_hits(&retry, &expected);
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
    fn int8_two_pass_keep_all_matches_exact() {
        // With a multiplier large enough to retain every record, pass-1 keeps all
        // main vectors, so the exact f16 rescore must reproduce `search_top_k`
        // bit-for-bit — verifying the byte offsets, tombstone flag check, rescore,
        // and resolve are correct, independent of int8 selection quality.
        let path = temp_index_path("int8-two-pass-keepall");
        let dim = 8;
        let count = 300;
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut s = (i as u64).wrapping_mul(2_654_435_761)
                            ^ (j as u64).wrapping_mul(40_503);
                        s ^= s >> 13;
                        ((s & 0xffff) as f32 / 65_535.0) - 0.5
                    })
                    .collect()
            })
            .collect();
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        write_index(&path, &row_refs).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        for qi in 0..8_usize {
            let query: Vec<f32> = (0..dim)
                .map(|j| (((qi * 7 + j * 3) % 11) as f32 / 11.0) - 0.5)
                .collect();
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            // mult=50 → candidate_count clamps to `count` → pass-1 retains all.
            let approx = index
                .search_top_k_int8_two_pass(&query, 10, 50)
                .expect("int8 two-pass");
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let approx_ids: Vec<&str> = approx.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                exact_ids, approx_ids,
                "int8 two-pass (keep-all) must match exact search_top_k for query {qi}"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
    fn int8_row_block_matches_orig_with_tombstones_and_tail() {
        // Cross the parallel threshold and leave a three-row tail in the final
        // chunk. Tombstones in two different four-row blocks force the exact
        // per-row fallback while the remaining blocks use the fused x4 kernel.
        let path = temp_index_path("int8-row-block-orig-parity");
        let dim = 33;
        let count = 10_003;
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut s = (i as u64).wrapping_mul(2_654_435_761)
                            ^ (j as u64).wrapping_mul(40_503);
                        s ^= s >> 13;
                        ((s & 0xffff) as f32 / 65_535.0) - 0.5
                    })
                    .collect()
            })
            .collect();
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        write_index(&path, &row_refs).expect("write index");
        let mut index = VectorIndex::open(&path).expect("open index");
        index
            .soft_delete_batch(&["doc-001", "doc-004", "doc-4097", "doc-10002"])
            .expect("soft delete mixed rows");

        for qi in 0..3_usize {
            let query: Vec<f32> = (0..dim)
                .map(|j| (((qi * 11 + j * 7) % 29) as f32 / 29.0) - 0.5)
                .collect();
            for mult in [2, 3, 5, 10] {
                let orig = index
                    .bench_search_top_k_int8_two_pass_orig(&query, 10, mult)
                    .expect("original int8 two-pass");
                let candidate = index
                    .bench_search_top_k_int8_two_pass_row_block_candidate(&query, 10, mult)
                    .expect("row-blocked int8 two-pass");
                assert_eq!(candidate.len(), orig.len());
                for (candidate_hit, orig_hit) in candidate.iter().zip(&orig) {
                    assert_eq!(candidate_hit.index, orig_hit.index, "qi={qi} mult={mult}");
                    assert_eq!(candidate_hit.doc_id, orig_hit.doc_id, "qi={qi} mult={mult}");
                    assert_eq!(
                        candidate_hit.score.to_bits(),
                        orig_hit.score.to_bits(),
                        "qi={qi} mult={mult}"
                    );
                }
            }
        }
    }

    /// CI RECALL GUARD for the shipped maddubs pass-1 kernel (bd-b5wl). Unlike the row-block
    /// candidate, the `vpmaddubs` pass-1 is APPROXIMATE (saturates on the quantizer's ±127 tail), so
    /// it is *not* bit-identical to the exact int8 scan. What must hold — and what makes it safe to
    /// ship as the default — is that it does not lose recall vs the exact-flat f32 top-k: the pass-1
    /// still retains the true top-k into the candidate set, and the f16 rescore then orders them
    /// exactly. Asserts maddubs recall@10 ≥ orig recall@10 and both are perfect on a realistic
    /// (normalized, clustered) corpus that exercises the saturation.
    #[test]
    #[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
    fn int8_two_pass_maddubs_preserves_recall_vs_flat() {
        let path = temp_index_path("int8-maddubs-recall");
        let dim = 384;
        let count = 4_000;
        let normalize = |mut v: Vec<f32>| -> Vec<f32> {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut v {
                *x /= n;
            }
            v
        };
        // 16 clusters + jitter → realistic ANN corpus (tight top-k, real quantized magnitudes).
        let centroids: Vec<Vec<f32>> = (0..16)
            .map(|c| {
                normalize(
                    (0..dim)
                        .map(|j| {
                            let mut s = (c as u64 + 1).wrapping_mul(0x9e37)
                                ^ (j as u64).wrapping_mul(40_503);
                            s ^= s >> 13;
                            ((s & 0xffff) as f32 / 65_535.0) - 0.5
                        })
                        .collect(),
                )
            })
            .collect();
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                let c = &centroids[i % 16];
                normalize(
                    (0..dim)
                        .map(|j| {
                            let mut s = (i as u64 + 1).wrapping_mul(2_654_435_761)
                                ^ (j as u64).wrapping_mul(7);
                            s ^= s >> 13;
                            c[j] + 0.15 * (((s & 0xffff) as f32 / 65_535.0) - 0.5)
                        })
                        .collect(),
                )
            })
            .collect();
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        write_index(&path, &row_refs).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        let ids = |hits: Vec<VectorHit>| -> Vec<String> {
            hits.into_iter().map(|h| h.doc_id.to_string()).collect()
        };
        let recall = |exact: &[String], approx: &[String]| -> f64 {
            let hit = approx.iter().filter(|id| exact.contains(id)).count();
            hit as f64 / exact.len().max(1) as f64
        };

        for (c, centroid) in centroids.iter().take(4).enumerate() {
            let query = normalize(centroid.clone());
            let exact = ids(index.search_top_k(&query, 10, None).expect("flat"));
            for mult in [3usize, 5] {
                let orig = ids(index
                    .bench_search_top_k_int8_two_pass_orig(&query, 10, mult)
                    .expect("orig"));
                let maddubs = ids(index
                    .bench_search_top_k_int8_two_pass_maddubs(&query, 10, mult)
                    .expect("maddubs"));
                let orig_r = recall(&exact, &orig);
                let maddubs_r = recall(&exact, &maddubs);
                assert!(
                    maddubs_r >= orig_r - 1e-9,
                    "c={c} mult={mult}: maddubs recall {maddubs_r} < orig {orig_r}"
                );
                assert!(
                    (maddubs_r - 1.0).abs() < 1e-9,
                    "c={c} mult={mult}: maddubs recall {maddubs_r} != 1.0"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_sign_loss)] // deterministic non-negative fixture mixing
    fn four_bit_two_pass_keep_all_matches_exact() {
        // With a multiplier large enough to retain every record, the exact f16
        // rescore must reproduce `search_top_k` bit-for-bit — verifying the nibble
        // pack/unpack offsets, tombstone flag check, rescore, and resolve.
        let path = temp_index_path("4bit-two-pass-keepall");
        let dim = 70; // odd-ish, > 64, exercises packing + a partial last byte
        let count = 300;
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut s = (i as u64).wrapping_mul(2_654_435_761)
                            ^ (j as u64).wrapping_mul(40_503);
                        s ^= s >> 13;
                        ((s & 0xffff) as f32 / 65_535.0) - 0.5
                    })
                    .collect()
            })
            .collect();
        let rows = create_rows(&vectors);
        let row_refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        write_index(&path, &row_refs).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        for qi in 0..8_usize {
            let query: Vec<f32> = (0..dim)
                .map(|j| (((qi * 7 + j * 3) % 11) as f32 / 11.0) - 0.5)
                .collect();
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let approx = index
                .search_top_k_4bit_two_pass(&query, 10, 50)
                .expect("4bit two-pass");
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let approx_ids: Vec<&str> = approx.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                exact_ids, approx_ids,
                "4bit two-pass (keep-all) must match exact search_top_k for query {qi}"
            );
        }
    }

    proptest! {
        #[test]
        fn property_top_k_invariants_hold(
            vectors in prop::collection::vec(prop::collection::vec(-1.0_f32..1.0_f32, 4), 1..20),
            query in prop::collection::vec(-1.0_f32..1.0_f32, 4),
            limit in 1_usize..20,
        ) {
            let path = temp_index_path("prop-top-k");
            let rows = create_rows(&vectors);
            let row_refs: Vec<(&str, Vec<f32>)> = rows
                .iter()
                .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
                .collect();
            // Skip test case if temp dir is unwritable (macOS CI runners can
            // hit PermissionDenied under heavy concurrent test load).
            prop_assume!(write_index(&path, &row_refs).is_ok());

            let index = VectorIndex::open(&path).expect("open index");
            let hits = index.search_top_k(&query, limit, None).expect("search");

            let expected_len = limit.min(vectors.len());
            prop_assert_eq!(hits.len(), expected_len);
            let mut seen_indices = HashSet::new();
            for hit in &hits {
                prop_assert!(seen_indices.insert(hit.index));
            }
        }

        #[test]
        fn property_parallel_and_sequential_paths_match(
            vectors in prop::collection::vec(prop::collection::vec(-1.0_f32..1.0_f32, 4), 8..40),
            query in prop::collection::vec(-1.0_f32..1.0_f32, 4),
            limit in 1_usize..20,
        ) {
            let path = temp_index_path("prop-parallel");
            let rows = create_rows(&vectors);
            let row_refs: Vec<(&str, Vec<f32>)> = rows
                .iter()
                .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
                .collect();
            prop_assume!(write_index(&path, &row_refs).is_ok());

            let index = VectorIndex::open(&path).expect("open index");
            let sequential = index
                .search_top_k_internal(&query, limit, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
                .expect("sequential search");
            let parallel = index
                .search_top_k_internal(&query, limit, None, 1, 4, true)
                .expect("parallel search");

            prop_assert_eq!(sequential.len(), parallel.len());
            for (left, right) in sequential.iter().zip(parallel.iter()) {
                prop_assert_eq!(&left.doc_id, &right.doc_id);
                prop_assert_eq!(left.index, right.index);
                prop_assert!((left.score - right.score).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn top_k_orders_by_score_descending() {
        let path = temp_index_path("top-k-order");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(hits[1].doc_id, "doc-b");
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn filter_excludes_matching_doc_id() {
        let path = temp_index_path("filter");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let filter = PredicateFilter::new("exclude-a", |doc_id| doc_id != "doc-a");
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 2, Some(&filter))
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));
    }

    #[test]
    fn tombstoned_records_are_excluded_from_search() {
        let path = temp_index_path("tombstone-excluded");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        assert!(index.soft_delete("doc-a").expect("delete doc-a"));

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.doc_id != "doc-a"));
    }

    #[test]
    fn parallel_and_sequential_ignore_tombstones() {
        let path = temp_index_path("tombstone-parallel");
        let mut rows = Vec::new();
        for i in 0..96 {
            let score = f32::from(u16::try_from(96 - i).expect("test index must fit in u16"));
            rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        let deleted = index
            .soft_delete_batch(&["doc-000", "doc-001", "doc-002", "doc-003"])
            .expect("batch delete");
        assert_eq!(deleted, 4);

        let query = [1.0, 0.0, 0.0, 0.0];
        let sequential = index
            .search_top_k_internal(&query, 10, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
            .expect("sequential");
        let parallel = index
            .search_top_k_internal(&query, 10, None, 1, 8, true)
            .expect("parallel");

        assert_eq!(sequential.len(), parallel.len());
        let deleted_ids = ["doc-000", "doc-001", "doc-002", "doc-003"];
        assert!(
            sequential
                .iter()
                .all(|hit| !deleted_ids.contains(&hit.doc_id.as_str()))
        );
        assert!(
            parallel
                .iter()
                .all(|hit| !deleted_ids.contains(&hit.doc_id.as_str()))
        );
        for (left, right) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(left.doc_id, right.doc_id);
            assert!((left.score - right.score).abs() < 1e-6);
        }
    }

    #[test]
    fn parallel_and_sequential_paths_match() {
        let path = temp_index_path("parallel-match");
        let mut rows = Vec::new();
        for i in 0..64 {
            // Start at 1: a zero-norm row would be rejected by the writer gate.
            let rank = f32::from(u16::try_from(i + 1).expect("test index must fit in u16"));
            rows.push((
                format!("doc-{i:03}"),
                vec![rank, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sequential = index
            .search_top_k_internal(&query, 10, None, usize::MAX, PARALLEL_CHUNK_SIZE, true)
            .expect("sequential search");
        let parallel = index
            .search_top_k_internal(&query, 10, None, 1, 4, true)
            .expect("parallel search");

        assert_eq!(sequential.len(), parallel.len());
        for (left, right) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(left.doc_id, right.doc_id);
            assert!((left.score - right.score).abs() < 1e-6);
        }
    }

    #[test]
    fn parallel_and_sequential_paths_match_with_filter() {
        let path = temp_index_path("parallel-match-filter");
        let mut rows = Vec::new();
        for i in 0..96 {
            // Start at 1: a zero-norm row would be rejected by the writer gate.
            let rank = f32::from(u16::try_from(i + 1).expect("test index must fit in u16"));
            rows.push((
                format!("doc-{i:03}"),
                vec![rank, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let filter = PredicateFilter::new("even-docs", |doc_id| {
            let suffix = doc_id.strip_prefix("doc-").unwrap_or_default();
            suffix.parse::<u32>().is_ok_and(|v| v % 2 == 0)
        });

        let sequential = index
            .search_top_k_internal(
                &query,
                15,
                Some(&filter),
                usize::MAX,
                PARALLEL_CHUNK_SIZE,
                true,
            )
            .expect("sequential search");
        let parallel = index
            .search_top_k_internal(&query, 15, Some(&filter), 1, 8, true)
            .expect("parallel search");

        assert_eq!(sequential.len(), parallel.len());
        for (left, right) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(left.doc_id, right.doc_id);
            assert!((left.score - right.score).abs() < 1e-6);
        }
    }

    #[test]
    fn resolves_doc_ids_only_for_winners() {
        let path = temp_index_path("two-phase");
        write_index(
            &path,
            &[("winner", vec![1.0, 0.0]), ("loser", vec![0.0, 1.0])],
        )
        .expect("write index");

        let inspect = VectorIndex::open(&path).expect("open index");
        let loser_idx = inspect
            .find_index_by_doc_hash(super::super::fnv1a_hash(b"loser"))
            .expect("loser index");
        let entry = inspect.record_at(loser_idx).expect("record");
        let loser_offset =
            inspect.strings_offset + usize::try_from(entry.doc_id_offset).unwrap_or(0);
        drop(inspect);

        let mut bytes = fs::read(&path).expect("read bytes");
        bytes[loser_offset] = 0xFF;
        fs::write(&path, bytes).expect("write corrupt bytes");

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index
            .search_top_k(&[1.0, 0.0], 1, None)
            .expect("search should only resolve winner doc_id");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "winner");
    }

    #[test]
    fn bitset_filter_skips_doc_id_decode_for_non_matching_records() {
        let path = temp_index_path("bitset-hash-fast-path");
        write_index(
            &path,
            &[("doc-a", vec![1.0, 0.0]), ("doc-b", vec![0.0, 1.0])],
        )
        .expect("write index");

        let inspect = VectorIndex::open(&path).expect("open index");
        let bad_idx = inspect
            .find_index_by_doc_hash(super::super::fnv1a_hash(b"doc-b"))
            .expect("doc-b index");
        let record = inspect.record_at(bad_idx).expect("record");
        let bad_offset =
            inspect.strings_offset + usize::try_from(record.doc_id_offset).expect("offset");
        drop(inspect);

        let mut bytes = fs::read(&path).expect("read bytes");
        bytes[bad_offset] = 0xFF;
        fs::write(&path, bytes).expect("write corrupt bytes");

        let index = VectorIndex::open(&path).expect("open index");
        let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a"]);
        let hits = index
            .search_top_k(&[1.0, 0.0], 10, Some(&filter))
            .expect("search should ignore corrupted filtered-out doc_id");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[test]
    fn selective_exact_ids_gather_matches_scan_and_bounds_predicate_work() {
        use frankensearch_core::IncludeDocIdsFilter;
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        struct Candidates {
            ids: IncludeDocIdsFilter,
            visits: AtomicUsize,
        }
        impl SearchFilter for Candidates {
            fn matches(&self, id: &str, metadata: Option<&serde_json::Value>) -> bool {
                self.visits.fetch_add(1, AtomicOrdering::Relaxed);
                self.ids.matches(id, metadata) && id != "doc-097"
            }
            fn candidate_doc_ids(&self) -> Option<&HashSet<String>> {
                self.ids.candidate_doc_ids()
            }
            fn name(&self) -> &'static str {
                "counted-exact-candidates"
            }
        }

        for quantization in [Quantization::F16, Quantization::F32] {
            let path = temp_index_path("exact-id-gather");
            let mut writer =
                VectorIndex::create_with_revision(&path, "hash", "test", 2, quantization).unwrap();
            for i in 0_u16..512 {
                writer
                    .write_record(&format!("doc-{i:03}"), &[f32::from(i) + 1.0, 1.0])
                    .unwrap();
            }
            writer.finish().unwrap();
            let index = VectorIndex::open(&path).unwrap();
            let filter = Candidates {
                ids: IncludeDocIdsFilter::new(["doc-003", "doc-097", "doc-203", "absent"]),
                visits: AtomicUsize::new(0),
            };
            let query = [1.0, 0.0];
            let gathered = index
                .search_top_k(&query, usize::MAX, Some(&filter))
                .unwrap();
            assert_eq!(filter.visits.load(AtomicOrdering::Relaxed), 3);
            assert_eq!(hit_ids(&gathered), ["doc-203", "doc-003"]);
            filter.visits.store(0, AtomicOrdering::Relaxed);
            let scanned = index
                .bench_scan_filtered(&query, usize::MAX, Some(&filter))
                .unwrap();
            assert_eq!(filter.visits.load(AtomicOrdering::Relaxed), 512);
            let fingerprint = |hits: &[VectorHit]| {
                hits.iter()
                    .map(|hit| (hit.doc_id.clone(), hit.index, hit.score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(fingerprint(&gathered), fingerprint(&scanned));
        }
    }

    #[test]
    fn selective_bitset_filter_uses_file_backed_gather() {
        let path = temp_index_path("fsvi-selective-gather");
        let rows: Vec<(String, Vec<f32>)> = (0..256)
            .map(|i| {
                let x = f32::from(u16::try_from(i % 31).expect("small test value")) / 31.0;
                let y = f32::from(u16::try_from(i / 31).expect("small test value")) / 10.0;
                (format!("doc-{i:03}"), vec![x, y, 0.25, 0.5])
            })
            .collect();
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vector)| (doc_id.as_str(), vector.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let filter =
            frankensearch_core::BitsetFilter::from_doc_ids(["doc-003", "doc-097", "doc-203"]);
        let query = [0.7, 0.3, 0.0, 0.0];
        let scan = index
            .bench_scan_filtered(&query, 3, Some(&filter))
            .expect("forced scan");
        let gather = index
            .bench_gather_filtered(&query, 3, &filter)
            .expect("forced gather");
        let public = index
            .search_top_k(&query, 3, Some(&filter))
            .expect("public search");

        assert_eq!(hit_ids(&gather), hit_ids(&scan));
        assert_eq!(hit_ids(&public), hit_ids(&scan));
    }

    #[test]
    fn limit_zero_or_empty_index_returns_no_hits() {
        let path = temp_index_path("limit-zero");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
            .expect("writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        let zero_limit = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .expect("search");
        let empty_index = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search");

        assert!(zero_limit.is_empty());
        assert!(empty_index.is_empty());
    }

    // ── search_top_k_classified (bd-tqhc) ───────────────────────────────

    #[test]
    fn classified_distinguishes_k_zero_from_newly_created_empty() {
        let path = temp_index_path("classified-kzero-vs-empty");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
            .expect("writer");
        writer.finish().expect("finish");
        let index = VectorIndex::open(&path).expect("open index");

        let k_zero = index
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .expect("k=0 search");
        assert!(k_zero.hits.is_empty());
        assert_eq!(
            k_zero.zero_signal,
            Some(ZeroSignalReason::CallerRequestedZeroK)
        );

        let empty = index
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("empty-index search");
        assert!(empty.hits.is_empty());
        assert_eq!(
            empty.zero_signal,
            Some(ZeroSignalReason::NewlyCreatedEmpty),
            "an index that never held a record must classify as newly created, \
             not collapse into the k=0 shape"
        );
    }

    #[test]
    fn classified_rejects_non_finite_query_like_ann_does() {
        let path = temp_index_path("classified-nonfinite-query");
        write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        let err = index
            .search_top_k_classified(&[f32::NAN, 0.0, 0.0, 0.0], 5, None)
            .expect_err("non-finite query must fail closed");
        assert!(
            matches!(err, SearchError::InvalidConfig { ref field, .. } if field == "query"),
            "expected InvalidConfig on query, got: {err:?}"
        );

        // The unclassified lane keeps its legacy behavior (no error), which is
        // exactly why production flows through the classified lane.
        index
            .search_top_k(&[f32::NAN, 0.0, 0.0, 0.0], 5, None)
            .expect("legacy lane is unchanged");
    }

    #[test]
    fn classified_zero_norm_query_is_typed_not_silent() {
        let path = temp_index_path("classified-zero-norm");
        write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        let classified = index
            .search_top_k_classified(&[0.0, 0.0, 0.0, 0.0], 5, None)
            .expect("zero-norm search");
        assert!(classified.hits.is_empty());
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::ZeroNormQuery)
        );
    }

    #[test]
    fn classified_filter_eliminating_all_is_distinct_from_empty_index() {
        let path = temp_index_path("classified-filter-all");
        write_index(
            &path,
            &[
                ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        let reject_all = PredicateFilter::new("reject-all", |_| false);
        let classified = index
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
            .expect("filtered search");
        assert!(classified.hits.is_empty());
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
    }

    #[test]
    fn classified_all_tombstoned_is_distinct_from_newly_created() {
        let path = temp_index_path("classified-all-tombstoned");
        write_index(
            &path,
            &[
                ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");
        let mut index = VectorIndex::open(&path).expect("open index");
        index.soft_delete("doc-a").expect("tombstone doc-a");
        index.soft_delete("doc-b").expect("tombstone doc-b");

        let classified = index
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search over tombstoned index");
        assert!(classified.hits.is_empty());
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::AllTombstoned)
        );
    }

    #[test]
    fn classified_nonempty_result_carries_no_reason() {
        let path = temp_index_path("classified-nonempty");
        write_index(&path, &[("doc-a", vec![0.1, 0.0, 0.0, 0.0])]).expect("write index");
        let index = VectorIndex::open(&path).expect("open index");

        let classified = index
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search");
        assert_eq!(classified.hits.len(), 1);
        assert_eq!(classified.zero_signal, None);
    }

    fn corrupt_classified_index(
        name: &str,
        quantization: Quantization,
        all_unusable: bool,
    ) -> (TempFixturePath, VectorIndex) {
        let path = temp_index_path(name);
        let mut writer = VectorIndex::create_with_revision(&path, "hash", "test", 2, quantization)
            .expect("corruption fixture writer");
        for (id, vector) in [
            ("bad-zero", [1.0, 0.0]),
            ("bad-nan", [1.0, 0.0]),
            ("negative-high", [-0.5, 0.5]),
            ("negative-low", [-1.0, 0.0]),
        ] {
            writer
                .write_record(id, &vector)
                .expect("valid source vector");
        }
        writer.finish().expect("finish valid fixture");
        let index = VectorIndex::open_read_only(&path).expect("open valid fixture");
        let offset = usize::try_from(index.metadata().vectors_offset).unwrap();
        let stride = match quantization {
            Quantization::F16 => 4,
            Quantization::F32 => 8,
        };
        let mut bytes = fs::read(&path).expect("read persisted fixture");
        for row in 0..index.record_count() {
            let id = index.doc_id_at(row).unwrap();
            if all_unusable || id.starts_with("bad-") {
                // Mutate only the slab after the writer has admitted valid
                // vectors. All-one bytes encode NaNs in both IEEE formats.
                bytes[offset + row * stride..offset + (row + 1) * stride]
                    .fill(if id == "bad-nan" { 0xff } else { 0 });
            }
        }
        drop(index);
        fs::write(&path, bytes).expect("persist vector corruption");
        let index = VectorIndex::open_read_only(&path).expect("open corrupt vector slab");
        (path, index)
    }

    #[test]
    fn classified_corrupt_vectors_do_not_hide_valid_negative_scores() {
        for quantization in [Quantization::F16, Quantization::F32] {
            let (path, index) =
                corrupt_classified_index("classified-corrupt-mixed", quantization, false);
            let memory = crate::InMemoryVectorIndex::from_fsvi(&path).unwrap();
            let two_tier = crate::TwoTierIndex::open_with_paths(
                &crate::TwoTierIndexPaths::new(path.to_path_buf()),
                frankensearch_core::TwoTierConfig::default(),
            )
            .unwrap();
            let raw = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();
            assert_eq!(hit_ids(&raw), ["bad-zero"], "raw numeric contract");
            for limit in [1, 2, usize::MAX] {
                let classified = index
                    .search_top_k_classified(&[1.0, 0.0], limit, None)
                    .unwrap();
                let expected = if limit == 1 {
                    vec!["negative-high"]
                } else {
                    vec!["negative-high", "negative-low"]
                };
                assert_eq!(
                    hit_ids(&classified.hits),
                    expected,
                    "{quantization:?} k={limit}"
                );
                assert_eq!(classified.zero_signal, None);
                for search_dims in [1, 2] {
                    let config = crate::MrlConfig {
                        search_dims,
                        rescore_top_k: limit,
                        ..crate::MrlConfig::default()
                    };
                    let (hits, stats) = index
                        .mrl_search_with_stats(&[1.0, 0.0], limit, &config, None)
                        .unwrap();
                    assert_eq!(hit_ids(&hits), expected, "MRL dims={search_dims} k={limit}");
                    assert_eq!(stats.zero_signal, None);
                }
                let in_memory = memory
                    .search_top_k_classified(&[1.0, 0.0], limit, None)
                    .unwrap();
                assert_eq!(
                    hit_ids(&in_memory.hits),
                    expected,
                    "in-memory {quantization:?}"
                );
                assert_eq!(in_memory.zero_signal, None);
                let fast = two_tier.search_fast_classified(&[1.0, 0.0], limit).unwrap();
                assert_eq!(hit_ids(&fast.hits), expected, "fast tier {quantization:?}");
                assert_eq!(fast.zero_signal, None);
            }
            let low_only = PredicateFilter::new("negative-low-and-corrupt", |id| {
                id == "negative-low" || id.starts_with("bad-")
            });
            for classified in [
                index
                    .search_top_k_classified(&[1.0, 0.0], 1, Some(&low_only))
                    .unwrap(),
                memory
                    .search_top_k_classified(&[1.0, 0.0], 1, Some(&low_only))
                    .unwrap(),
            ] {
                assert_eq!(hit_ids(&classified.hits), ["negative-low"]);
                assert_eq!(classified.zero_signal, None);
            }
        }
    }

    #[test]
    fn classified_unusable_persisted_vectors_report_availability_failure() {
        for quantization in [Quantization::F16, Quantization::F32] {
            let (path, index) =
                corrupt_classified_index("classified-corrupt-all", quantization, true);
            assert_eq!(index.zero_signal_state().usable_vector_count, 0);
            let classified = index.search_top_k_classified(&[1.0, 0.0], 2, None).unwrap();
            assert!(classified.hits.is_empty(), "{quantization:?}");
            assert_eq!(
                classified.zero_signal,
                Some(ZeroSignalReason::NoUsableVectors)
            );
            for search_dims in [1, 2] {
                let config = crate::MrlConfig {
                    search_dims,
                    ..crate::MrlConfig::default()
                };
                let (hits, stats) = index
                    .mrl_search_with_stats(&[1.0, 0.0], 2, &config, None)
                    .unwrap();
                assert!(hits.is_empty());
                assert_eq!(stats.zero_signal, Some(ZeroSignalReason::NoUsableVectors));
            }
            let two_tier = crate::TwoTierIndex::open_with_paths(
                &crate::TwoTierIndexPaths::new(path.to_path_buf()),
                frankensearch_core::TwoTierConfig::default(),
            )
            .unwrap();
            let fast = two_tier.search_fast_classified(&[1.0, 0.0], 2).unwrap();
            assert!(fast.hits.is_empty(), "fast tier {quantization:?}");
            assert_eq!(fast.zero_signal, Some(ZeroSignalReason::NoUsableVectors));
            let memory = crate::InMemoryVectorIndex::from_fsvi(&path).unwrap();
            let classified = memory
                .search_top_k_classified(&[1.0, 0.0], 2, None)
                .unwrap();
            assert!(classified.hits.is_empty());
            assert_eq!(
                classified.zero_signal,
                Some(ZeroSignalReason::NoUsableVectors)
            );
        }
    }

    #[test]
    fn classified_orthogonal_vectors_remain_usable() {
        let path = temp_index_path("classified-orthogonal");
        write_index(&path, &[("orthogonal", vec![0.0, 1.0])]).unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let classified = index.search_top_k_classified(&[1.0, 0.0], 1, None).unwrap();
        assert_eq!(hit_ids(&classified.hits), ["orthogonal"]);
        assert_eq!(classified.hits[0].score, 0.0);
        assert_eq!(classified.zero_signal, None);
        let memory = crate::InMemoryVectorIndex::from_fsvi(&path).unwrap();
        let classified = memory
            .search_top_k_classified(&[1.0, 0.0], 1, None)
            .unwrap();
        assert_eq!(hit_ids(&classified.hits), ["orthogonal"]);
        assert_eq!(classified.hits[0].score.to_bits(), 0.0_f32.to_bits());
        assert_eq!(classified.zero_signal, None);
        let two_tier = crate::TwoTierIndex::open_with_paths(
            &crate::TwoTierIndexPaths::new(path.to_path_buf()),
            frankensearch_core::TwoTierConfig::default(),
        )
        .unwrap();
        let fast = two_tier.search_fast_classified(&[1.0, 0.0], 1).unwrap();
        assert_eq!(hit_ids(&fast.hits), ["orthogonal"]);
        assert_eq!(fast.hits[0].score.to_bits(), 0.0_f32.to_bits());
        assert_eq!(fast.zero_signal, None);
    }

    #[test]
    fn classified_unusable_wal_replacement_does_not_resurrect_main() {
        let path = temp_index_path("classified-corrupt-wal");
        write_index(
            &path,
            &[("replaced", vec![1.0, 0.0]), ("negative", vec![-1.0, 0.0])],
        )
        .unwrap();
        let mut index = VectorIndex::open(&path).unwrap();
        index.append("replaced", &[0.0, 1.0]).unwrap();
        index.wal_entries[0].embedding.fill(0.0);
        let classified = index.search_top_k_classified(&[1.0, 0.0], 1, None).unwrap();
        assert_eq!(hit_ids(&classified.hits), ["negative"]);
        assert_eq!(classified.zero_signal, None);
        let cancellable = index
            .search_top_k_classified_cancellable(
                &Cx::for_testing(),
                &[1.0, 0.0],
                1,
                None,
                SearchParams::default(),
            )
            .unwrap();
        assert_eq!(cancellable.hits, classified.hits);
        assert_eq!(cancellable.zero_signal, classified.zero_signal);
    }

    #[test]
    fn classified_finite_but_unusable_norms_are_not_ranked() {
        let (path, index) =
            corrupt_classified_index("classified-corrupt-norms", Quantization::F32, false);
        let offset = usize::try_from(index.metadata().vectors_offset).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        for row in 0..index.record_count() {
            let value: f32 = match index.doc_id_at(row).unwrap() {
                "bad-nan" => 1.0e30,
                "bad-zero" => 1.0e-30,
                _ => continue,
            };
            bytes[offset + row * 8..offset + row * 8 + 4].copy_from_slice(&value.to_le_bytes());
            bytes[offset + row * 8 + 4..offset + (row + 1) * 8].fill(0);
        }
        drop(index);
        fs::write(&path, bytes).unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let memory = crate::InMemoryVectorIndex::from_fsvi(&path).unwrap();
        let two_tier = crate::TwoTierIndex::open_with_paths(
            &crate::TwoTierIndexPaths::new(path.to_path_buf()),
            frankensearch_core::TwoTierConfig::default(),
        )
        .unwrap();
        for classified in [
            index.search_top_k_classified(&[1.0, 0.0], 2, None).unwrap(),
            memory
                .search_top_k_classified(&[1.0, 0.0], 2, None)
                .unwrap(),
            two_tier.search_fast_classified(&[1.0, 0.0], 2).unwrap(),
        ] {
            assert_eq!(hit_ids(&classified.hits), ["negative-high", "negative-low"]);
            assert_eq!(classified.zero_signal, None);
        }
    }

    #[test]
    fn classified_healthy_wal_survives_an_unusable_main_slab() {
        let (path, index) =
            corrupt_classified_index("classified-healthy-wal", Quantization::F16, true);
        drop(index);
        let mut index = VectorIndex::open(&path).unwrap();
        index.append("live-wal", &[-0.25, 0.0]).unwrap();
        let classified = index.search_top_k_classified(&[1.0, 0.0], 1, None).unwrap();
        assert_eq!(hit_ids(&classified.hits), ["live-wal"]);
        assert_eq!(classified.zero_signal, None);
        let reject_all = PredicateFilter::new("reject-healthy-wal", |_| false);
        let classified = index
            .search_top_k_classified(&[1.0, 0.0], 1, Some(&reject_all))
            .unwrap();
        assert!(classified.hits.is_empty());
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
        // The writer retains the exclusive mapping lock until it is dropped.
        // Reopen the persisted WAL through the wrapper only after releasing it.
        drop(index);
        let two_tier = crate::TwoTierIndex::open_with_paths(
            &crate::TwoTierIndexPaths::new(path.to_path_buf()),
            frankensearch_core::TwoTierConfig::default(),
        )
        .unwrap();
        let fast = two_tier.search_fast_classified(&[1.0, 0.0], 1).unwrap();
        assert_eq!(hit_ids(&fast.hits), ["live-wal"]);
        assert_eq!(fast.zero_signal, None);
    }

    #[test]
    fn classified_in_memory_parity_on_request_scoped_states() {
        // The bead requires equivalent states to classify identically across
        // backends. Request-scoped states are representable in both; index
        // states diverge only where a state is structurally unrepresentable
        // in memory (tombstones compact away at load).
        let path = temp_index_path("classified-parity");
        write_index(
            &path,
            &[
                ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");
        let file_backed = VectorIndex::open_read_only(&path).expect("open index");
        let in_memory = crate::InMemoryVectorIndex::from_fsvi(&path).expect("load in-memory copy");

        // k = 0.
        let file_k0 = file_backed
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .expect("file k=0");
        let mem_k0 = in_memory
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .expect("mem k=0");
        assert_eq!(file_k0.zero_signal, mem_k0.zero_signal);

        // Zero-norm query.
        let file_zero = file_backed
            .search_top_k_classified(&[0.0; 4], 5, None)
            .expect("file zero-norm");
        let mem_zero = in_memory
            .search_top_k_classified(&[0.0; 4], 5, None)
            .expect("mem zero-norm");
        assert_eq!(file_zero.zero_signal, mem_zero.zero_signal);

        // Non-finite query errors on both.
        assert!(
            file_backed
                .search_top_k_classified(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, None)
                .is_err()
        );
        assert!(
            in_memory
                .search_top_k_classified(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, None)
                .is_err()
        );

        // Filter excludes all.
        let reject_all = PredicateFilter::new("reject-all", |_| false);
        let file_filtered = file_backed
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
            .expect("file filter-all");
        let mem_filtered = in_memory
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, Some(&reject_all))
            .expect("mem filter-all");
        assert_eq!(file_filtered.zero_signal, mem_filtered.zero_signal);

        // Non-empty carries no reason on either.
        let file_hits = file_backed
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("file search");
        let mem_hits = in_memory
            .search_top_k_classified(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("mem search");
        assert_eq!(file_hits.zero_signal, None);
        assert_eq!(mem_hits.zero_signal, None);
    }

    #[test]
    fn k_above_record_count_returns_all_hits() {
        let path = temp_index_path("k-above-count");
        write_index(
            &path,
            &[
                ("doc-a", vec![0.1, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 20, None)
            .expect("search");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn full_recall_collect_all_matches_heap_prefix_main_only() {
        let path = temp_index_path("collect-all-main");
        let mut rows = Vec::new();
        for i in 0..80 {
            let score = f32::from(u16::try_from(80 - i).expect("test index must fit in u16"));
            rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0];
        let total = index.record_count();
        let heap_limit = total.saturating_sub(7);

        let collect_all = index
            .search_top_k_internal(&query, total, None, 1, 8, true)
            .expect("collect-all");
        let heap_top = index
            .search_top_k_internal(
                &query,
                heap_limit,
                None,
                usize::MAX,
                PARALLEL_CHUNK_SIZE,
                true,
            )
            .expect("heap-top");

        assert_eq!(collect_all.len(), total);
        assert_eq!(heap_top.len(), heap_limit);
        for (heap_hit, full_hit) in heap_top.iter().zip(collect_all.iter()) {
            assert_eq!(heap_hit.doc_id, full_hit.doc_id);
            assert_eq!(heap_hit.index, full_hit.index);
            assert!((heap_hit.score - full_hit.score).abs() < 1e-6);
        }
    }

    #[test]
    fn full_recall_collect_all_matches_heap_prefix_with_wal() {
        let path = temp_index_path("collect-all-wal");
        let mut rows = Vec::new();
        for i in 0..48 {
            let score = f32::from(u16::try_from(48 - i).expect("test index must fit in u16"));
            rows.push((format!("doc-{i:03}"), vec![score, 0.0, 0.0, 0.0]));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        index
            .append_batch(&[
                ("wal-top".to_owned(), vec![200.0, 0.0, 0.0, 0.0]),
                ("wal-mid".to_owned(), vec![24.5, 0.0, 0.0, 0.0]),
                ("wal-tail".to_owned(), vec![-1.0, 0.0, 0.0, 0.0]),
            ])
            .expect("append wal batch");

        let query = [1.0, 0.0, 0.0, 0.0];
        let total = index
            .record_count()
            .saturating_add(index.wal_record_count());
        let heap_limit = total.saturating_sub(5);

        let collect_all = index
            .search_top_k_internal(&query, total.saturating_add(10), None, 1, 8, true)
            .expect("collect-all with wal");
        let heap_top = index
            .search_top_k_internal(
                &query,
                heap_limit,
                None,
                usize::MAX,
                PARALLEL_CHUNK_SIZE,
                true,
            )
            .expect("heap-top with wal");

        assert_eq!(collect_all.len(), total);
        assert_eq!(collect_all[0].doc_id, "wal-top");
        assert_eq!(heap_top.len(), heap_limit);
        for (heap_hit, full_hit) in heap_top.iter().zip(collect_all.iter()) {
            assert_eq!(heap_hit.doc_id, full_hit.doc_id);
            assert_eq!(heap_hit.index, full_hit.index);
            assert!((heap_hit.score - full_hit.score).abs() < 1e-6);
        }
    }

    #[test]
    fn ties_are_broken_by_index() {
        let path = temp_index_path("tie-break");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-c", vec![1.0, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 3, None)
            .expect("search");

        let mut indexes: Vec<u32> = hits.iter().map(|hit| hit.index).collect();
        let mut sorted = indexes.clone();
        sorted.sort_unstable();
        assert_eq!(indexes, sorted);
        assert_eq!(hits.len(), 3);
        indexes.clear();
    }

    #[test]
    fn nan_scores_do_not_panic_and_sort_last() {
        let path = temp_index_path("nan-safe");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.5, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let hits = index
            .search_top_k(&[f32::NAN, 0.0, 0.0, 0.0], 3, None)
            .expect("search");

        assert_eq!(hits.len(), 3);
        assert!(hits.iter().all(|hit| hit.score.is_nan()));
        assert!(hits.windows(2).all(|pair| pair[0].index <= pair[1].index));
    }

    #[test]
    fn parse_parallel_search_env_values() {
        assert!(parse_parallel_search_env(None));
        assert!(parse_parallel_search_env(Some("1")));
        assert!(parse_parallel_search_env(Some("true")));
        assert!(parse_parallel_search_env(Some("yes")));
        assert!(!parse_parallel_search_env(Some("0")));
        assert!(!parse_parallel_search_env(Some("false")));
        assert!(!parse_parallel_search_env(Some("no")));
        assert!(!parse_parallel_search_env(Some("off")));
    }

    // --- SearchFilter integration tests ---

    #[test]
    fn strict_metadata_filter_rejects_both_fsvi_and_wal_rows() {
        use frankensearch_core::{DocTypeFilter, MissingMetadataPolicy};

        let path = temp_index_path("strict-metadata");
        let query = [1.0, 0.0];
        write_index(&path, &[("base", query.to_vec())]).expect("write index");
        let mut index = VectorIndex::open(&path).expect("open index");
        index.append("wal", &query).expect("append to WAL");

        let filter = DocTypeFilter::new(["active"]);
        let hits = index.search_top_k(&query, 2, Some(&filter)).unwrap();
        assert_eq!(hits.len(), 2);

        let strict = filter.on_missing_metadata(MissingMetadataPolicy::Reject);
        let result = index
            .search_top_k_classified(&query, 2, Some(&strict))
            .unwrap();
        assert!(result.hits.is_empty());
        assert_eq!(
            result.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
    }

    #[test]
    fn strict_metadata_filters_reject_metadata_less_exact_and_wal_searches() {
        use frankensearch_core::{
            BitsetFilter, DateRangeFilter, DocTypeFilter, FilterChain, FilterMode,
            IncludeDocIdsFilter, MissingMetadataPolicy,
        };

        let strict_type =
            DocTypeFilter::new(["active"]).on_missing_metadata(MissingMetadataPolicy::Reject);
        let strict_date =
            DateRangeFilter::between(100, 200).on_missing_metadata(MissingMetadataPolicy::Reject);
        // The exact-ID operand is small enough to select the gather route.
        // Neither gathering nor hash compilation may bypass metadata policy.
        let exact_scope = FilterChain::new(FilterMode::All)
            .with(Box::new(IncludeDocIdsFilter::new(["doc-0"])))
            .with(Box::new(strict_type.clone()));
        let hashed_scope = FilterChain::new(FilterMode::All)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-0"])))
            .with(Box::new(strict_date));
        let filters: [&dyn SearchFilter; 4] =
            [&strict_type, &strict_date, &exact_scope, &hashed_scope];
        let query = [1.0, 0.0, 0.0, 0.0];
        let mrl = crate::MrlConfig {
            search_dims: 2,
            ..crate::MrlConfig::default()
        };
        for quantization in [Quantization::F16, Quantization::F32] {
            let path = temp_index_path("strict-metadata");
            let mut writer =
                VectorIndex::create_with_revision(&path, "hash", "test", query.len(), quantization)
                    .expect("create index");
            for row in 0..128 {
                writer
                    .write_record(&format!("doc-{row}"), &query)
                    .expect("write row");
            }
            writer.finish().expect("finish index");
            let mut index = VectorIndex::open(&path).expect("open index");
            for with_wal in [false, true] {
                if with_wal {
                    index.append("wal-only", &query).expect("append new ID");
                    index.append("doc-0", &query).expect("replace main row");
                }
                assert_eq!(
                    index
                        .search_top_k(&query, 3, Some(&DocTypeFilter::new(["active"])))
                        .expect("permissive metadata policy")
                        .len(),
                    3
                );
                for filter in filters {
                    for parallel_enabled in [false, true] {
                        let params = SearchParams {
                            parallel_enabled,
                            parallel_threshold: 0,
                            parallel_chunk_size: 7,
                        };
                        assert!(
                            index
                                .search_top_k_with_params(&query, 3, Some(filter), params)
                                .expect("strict exact search")
                                .is_empty(),
                            "{} admitted a metadata-less row ({quantization:?}, WAL={with_wal}, parallel={parallel_enabled})",
                            filter.name()
                        );
                    }
                    let classified = index
                        .search_top_k_classified(&query, 3, Some(filter))
                        .expect("strict classified search");
                    assert!(classified.hits.is_empty());
                    assert_eq!(
                        classified.zero_signal,
                        Some(ZeroSignalReason::FilterEliminatedAll)
                    );
                    assert!(
                        index
                            .mrl_search(&query, 3, &mrl, Some(filter))
                            .expect("strict truncated search")
                            .is_empty()
                    );
                }
            }
        }
    }

    #[test]
    fn bitset_filter_during_vector_search() {
        let path = temp_index_path("bitset-filter");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a", "doc-c"]);
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(hits[1].doc_id, "doc-c");
    }

    #[test]
    fn filter_chain_and_semantics_in_search() {
        let path = temp_index_path("filter-chain-and");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.6, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let chain = frankensearch_core::FilterChain::new(frankensearch_core::FilterMode::All)
            .with(Box::new(PredicateFilter::new("not-c", |id| id != "doc-c")))
            .with(Box::new(PredicateFilter::new("not-a", |id| id != "doc-a")));

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&chain))
            .expect("search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");
    }

    #[test]
    fn filter_chain_or_semantics_in_search() {
        let path = temp_index_path("filter-chain-or");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.6, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let chain = frankensearch_core::FilterChain::new(frankensearch_core::FilterMode::Any)
            .with(Box::new(PredicateFilter::new("is-a", |id| id == "doc-a")))
            .with(Box::new(PredicateFilter::new("is-c", |id| id == "doc-c")));

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&chain))
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(hits[1].doc_id, "doc-c");
    }

    #[test]
    fn filter_rejects_all_returns_empty() {
        let path = temp_index_path("filter-all-rejected");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let filter = PredicateFilter::new("reject-all", |_| false);
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
            .expect("search");

        assert!(hits.is_empty());
    }

    #[test]
    fn filter_applies_to_wal_entries() {
        let path = temp_index_path("filter-wal");
        write_index(&path, &[("doc-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        index
            .append("doc-b", &[0.9, 0.0, 0.0, 0.0])
            .expect("append doc-b");
        index
            .append("doc-c", &[0.8, 0.0, 0.0, 0.0])
            .expect("append doc-c");

        // Filter to only include doc-b (WAL entry).
        let filter = PredicateFilter::new("only-b", |id| id == "doc-b");
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
            .expect("search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");
    }

    #[test]
    fn filter_works_with_wal_and_main_combined() {
        let path = temp_index_path("filter-wal-main");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.5, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        index
            .append("doc-c", &[0.9, 0.0, 0.0, 0.0])
            .expect("append doc-c");

        // Filter includes doc-a (main) and doc-c (WAL), excludes doc-b (main).
        let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a", "doc-c"]);
        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filter))
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(hits[1].doc_id, "doc-c");
    }

    #[test]
    fn all_records_soft_deleted_returns_empty() {
        let path = temp_index_path("all-deleted");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
                ("doc-c", vec![0.5, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        let deleted = index
            .soft_delete_batch(&["doc-a", "doc-b", "doc-c"])
            .expect("batch delete");
        assert_eq!(deleted, 3);

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .expect("search");
        assert!(
            hits.is_empty(),
            "search over fully-deleted index should return empty"
        );
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let path = temp_index_path("dim-mismatch");
        write_index(&path, &[("doc-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let result = index.search_top_k(&[1.0, 0.0], 5, None);
        assert!(matches!(
            result,
            Err(SearchError::DimensionMismatch {
                expected: 4,
                found: 2
            })
        ));
    }

    #[test]
    fn wal_only_search_returns_wal_entries() {
        let path = temp_index_path("wal-only");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
            .expect("writer");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open index");
        assert_eq!(index.record_count(), 0);

        index
            .append("wal-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("append wal-a");
        index
            .append("wal-b", &[0.5, 0.0, 0.0, 0.0])
            .expect("append wal-b");

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .expect("search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "wal-a");
        assert_eq!(hits[1].doc_id, "wal-b");
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn wal_entries_can_outrank_main_index() {
        let path = temp_index_path("wal-outranks-main");
        write_index(
            &path,
            &[
                ("main-a", vec![0.3, 0.0, 0.0, 0.0]),
                ("main-b", vec![0.2, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let mut index = VectorIndex::open(&path).expect("open index");
        index
            .append("wal-top", &[1.0, 0.0, 0.0, 0.0])
            .expect("append wal-top");

        let hits = index
            .search_top_k(&[1.0, 0.0, 0.0, 0.0], 3, None)
            .expect("search");

        assert_eq!(hits.len(), 3);
        assert_eq!(
            hits[0].doc_id, "wal-top",
            "WAL entry with highest score should rank first"
        );
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
    }

    #[test]
    fn stale_main_entry_shadowed_by_wal() {
        let path = temp_index_path("stale-shadow");
        // Create main index with [1.0, 0.0]
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 2, Quantization::F32).unwrap();
        writer.write_record("doc-a", &[1.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        // Append doc-a with [0.0, 1.0] to WAL
        index.append("doc-a", &[0.0, 1.0]).unwrap();

        // Search for [1.0, 0.0]. The WAL entry scores 0.0, the Main entry scores 1.0.
        // If the bug exists, the Main entry will be returned instead of the WAL entry,
        // because the WAL entry (score 0.0) might not make it into the top-K heap
        // if there are other candidates, or it just gets omitted if K is small.
        let hits = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();

        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].score.abs() < f32::EPSILON,
            "Expected score 0.0 from WAL entry, but got leaked score {}",
            hits[0].score
        );
    }

    #[test]
    fn wal_index_marker_out_of_bounds_returns_error() {
        let path = temp_index_path("wal-oob-index-marker");
        write_index(&path, &[("main-a", vec![1.0, 0.0, 0.0, 0.0])]).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let fabricated = HeapEntry::new(to_wal_index(42), 1.0);
        let err = index
            .resolve_wal_hit(&fabricated)
            .expect_err("fabricated WAL marker should fail bounds check");
        assert!(matches!(err, SearchError::IndexCorrupted { .. }));
    }

    #[test]
    fn heap_entry_nan_sorted_below_finite_scores() {
        let nan_entry = HeapEntry::new(0, f32::NAN);
        let finite_entry = HeapEntry::new(1, 0.5);
        // candidate_is_better: finite should beat NaN
        assert!(candidate_is_better(finite_entry, nan_entry));
        assert!(!candidate_is_better(nan_entry, finite_entry));
    }

    #[test]
    fn heap_entry_equal_scores_tiebreak_by_index() {
        let left = HeapEntry::new(3, 0.5);
        let right = HeapEntry::new(7, 0.5);
        // Lower index wins the tiebreak
        assert!(candidate_is_better(left, right));
        assert!(!candidate_is_better(right, left));
    }

    #[test]
    fn int8_heap_keys_match_legacy_score_and_index_order() {
        let exact_scores = [-16_774_160, -1, 0, 1, 16_774_160];
        let indices = [0, 17, usize::MAX];
        for &left_score in &exact_scores {
            for &left_index in &indices {
                for &right_score in &exact_scores {
                    for &right_index in &indices {
                        let packed = int8_heap_key(left_index, left_score)
                            .cmp(&int8_heap_key(right_index, right_score));
                        let legacy = HeapEntry::new(left_index, left_score as f32)
                            .cmp(&HeapEntry::new(right_index, right_score as f32));
                        assert_eq!(packed, legacy);
                    }
                }
                assert_eq!(
                    int8_heap_index(int8_heap_key(left_index, left_score)),
                    left_index
                );
            }
        }

        // Above the consecutive-integer f32 range, distinct i32 dots can collapse
        // to the same float. The fallback must preserve that legacy tie behavior.
        let fallback_scores = [i32::MIN, -16_777_217, 16_777_216, 16_777_217, i32::MAX];
        for &left_score in &fallback_scores {
            for &right_score in &fallback_scores {
                let packed = int8_heap_key_from_f32(11, left_score)
                    .cmp(&int8_heap_key_from_f32(29, right_score));
                let legacy = HeapEntry::new(11, left_score as f32)
                    .cmp(&HeapEntry::new(29, right_score as f32));
                assert_eq!(packed, legacy);
            }
        }
    }

    #[test]
    fn packed_int8_merge_matches_legacy_heap_merge() {
        let partitions = [
            [(0_usize, 90_i32), (1, 40), (2, 40)],
            [(3, 100), (4, -5), (5, 40)],
            [(6, 70), (7, 40), (8, i32::MIN)],
        ];
        for limit in [0, 1, 3, 9] {
            let legacy_parts = partitions
                .iter()
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&(index, score)| HeapEntry::new(index, score as f32))
                        .collect::<BinaryHeap<_>>()
                })
                .collect();
            let packed_parts = partitions
                .iter()
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&(index, score)| int8_heap_key(index, score))
                        .collect::<BinaryHeap<_>>()
                })
                .collect();

            let mut legacy_indices = merge_partial_heaps(legacy_parts, limit, UncancelledScan)
                .expect("uncancelled merge")
                .into_iter()
                .map(|entry| entry.index)
                .collect::<Vec<_>>();
            let mut packed_indices = merge_int8_partial_heaps(packed_parts, limit, UncancelledScan)
                .expect("uncancelled int8 merge")
                .into_iter()
                .map(int8_heap_index)
                .collect::<Vec<_>>();
            legacy_indices.sort_unstable();
            packed_indices.sort_unstable();
            assert_eq!(packed_indices, legacy_indices);
        }
    }

    #[test]
    fn insert_candidate_with_limit_zero_is_noop() {
        let mut heap = BinaryHeap::new();
        insert_candidate(&mut heap, HeapEntry::new(0, 0.9), 0);
        assert!(heap.is_empty());
    }

    #[test]
    fn insert_candidate_evicts_worst_when_full() {
        let mut heap = BinaryHeap::new();
        insert_candidate(&mut heap, HeapEntry::new(0, 0.1), 2);
        insert_candidate(&mut heap, HeapEntry::new(1, 0.5), 2);
        // Heap is full (limit=2). Insert better candidate.
        insert_candidate(&mut heap, HeapEntry::new(2, 0.9), 2);

        let entries: Vec<HeapEntry> = heap.into_vec();
        assert_eq!(entries.len(), 2);
        let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
        // Top 3: 0.9, 0.7, 0.5 — the 0.1 should be evicted
        assert!(scores.contains(&0.9));
        assert!(scores.contains(&0.5));
        assert!(!scores.contains(&0.1));
    }

    #[test]
    fn insert_candidate_rejects_worse_when_full() {
        let mut heap = BinaryHeap::new();
        insert_candidate(&mut heap, HeapEntry::new(0, 0.5), 2);
        insert_candidate(&mut heap, HeapEntry::new(1, 0.9), 2);
        // Heap is full. Insert worse candidate — should be rejected.
        insert_candidate(&mut heap, HeapEntry::new(2, 0.1), 2);

        let entries: Vec<HeapEntry> = heap.into_vec();
        assert_eq!(entries.len(), 2);
        let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
        assert!(scores.contains(&0.9));
        assert!(scores.contains(&0.5));
        assert!(!scores.contains(&0.1));
    }

    #[test]
    fn merge_partial_heaps_preserves_top_k() {
        let mut heap_a = BinaryHeap::new();
        insert_candidate(&mut heap_a, HeapEntry::new(0, 0.9), 3);
        insert_candidate(&mut heap_a, HeapEntry::new(1, 0.1), 3);

        let mut heap_b = BinaryHeap::new();
        insert_candidate(&mut heap_b, HeapEntry::new(2, 0.7), 3);
        insert_candidate(&mut heap_b, HeapEntry::new(3, 0.5), 3);

        let merged = merge_partial_heaps(vec![heap_a, heap_b], 3, UncancelledScan)
            .expect("uncancelled merge");
        let entries: Vec<HeapEntry> = merged.into_vec();
        assert_eq!(entries.len(), 3);
        let scores: Vec<f32> = entries.iter().map(|e| e.score).collect();
        // Top 3: 0.9, 0.7, 0.5 — the 0.1 should be evicted
        assert!(scores.contains(&0.9));
        assert!(scores.contains(&0.7));
        assert!(scores.contains(&0.5));
        assert!(!scores.contains(&0.1));
    }

    #[test]
    fn parse_parallel_search_env_case_insensitive() {
        assert!(!parse_parallel_search_env(Some("OFF")));
        assert!(!parse_parallel_search_env(Some("False")));
        assert!(!parse_parallel_search_env(Some("NO")));
        assert!(!parse_parallel_search_env(Some("  off  ")));
    }

    #[test]
    fn parse_parallel_search_env_empty_string_enables() {
        // Empty string is not any of the disable values, so parallel stays enabled.
        assert!(parse_parallel_search_env(Some("")));
        assert!(parse_parallel_search_env(Some("  ")));
    }

    #[test]
    fn parallel_filter_path_propagates_doc_id_errors() {
        let path = temp_index_path("parallel-filter-errors");
        write_index(
            &path,
            &[("doc-a", vec![1.0, 0.0]), ("doc-b", vec![0.0, 1.0])],
        )
        .expect("write index");

        let inspect = VectorIndex::open(&path).expect("open index");
        let bad_idx = inspect
            .find_index_by_doc_hash(super::super::fnv1a_hash(b"doc-b"))
            .expect("doc-b index");
        let record = inspect.record_at(bad_idx).expect("record");
        let bad_offset =
            inspect.strings_offset + usize::try_from(record.doc_id_offset).unwrap_or(0);
        drop(inspect);

        let mut bytes = fs::read(&path).expect("read index bytes");
        bytes[bad_offset] = 0xFF;
        fs::write(&path, bytes).expect("write corrupt bytes");

        let index = VectorIndex::open(&path).expect("reopen index");
        let filter = PredicateFilter::new("allow-all", |_| true);
        let query = [1.0, 0.0];

        let sequential = index.search_top_k_internal(&query, 1, Some(&filter), usize::MAX, 2, true);
        let parallel = index.search_top_k_internal(&query, 1, Some(&filter), 1, 2, true);

        assert!(
            sequential.is_err(),
            "sequential path should surface doc_id errors"
        );
        assert!(
            parallel.is_err(),
            "parallel path should surface doc_id errors"
        );
    }

    // --- SearchParams tests ---

    #[test]
    fn search_params_default_matches_constants() {
        let params = SearchParams::default();
        assert_eq!(params.parallel_threshold, PARALLEL_THRESHOLD);
        assert_eq!(params.parallel_chunk_size, PARALLEL_CHUNK_SIZE);
    }

    #[test]
    fn search_top_k_with_params_matches_default_search() {
        let path = temp_index_path("with-params-default");
        let mut rows = Vec::new();
        for i in 0..32 {
            // Start at 1: a zero-norm row would be rejected by the writer gate.
            let rank = f32::from(u16::try_from(i + 1).expect("fits u16"));
            rows.push((format!("doc-{i:03}"), vec![rank, 0.0, 0.0, 0.0]));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0];

        let default_hits = index.search_top_k(&query, 5, None).expect("default search");
        let params_hits = index
            .search_top_k_with_params(&query, 5, None, SearchParams::default())
            .expect("params search");

        assert_eq!(default_hits.len(), params_hits.len());
        for (left, right) in default_hits.iter().zip(params_hits.iter()) {
            assert_eq!(left.doc_id, right.doc_id);
            assert_eq!(left.index, right.index);
            assert!((left.score - right.score).abs() < 1e-6);
        }
    }

    #[test]
    fn search_top_k_with_params_custom_threshold() {
        let path = temp_index_path("with-params-custom");
        let mut rows = Vec::new();
        for i in 0..64 {
            // Start at 1: a zero-norm row would be rejected by the writer gate.
            let rank = f32::from(u16::try_from(i + 1).expect("fits u16"));
            rows.push((
                format!("doc-{i:03}"),
                vec![rank, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let filter = PredicateFilter::new("even-docs", |doc_id| {
            let suffix = doc_id.strip_prefix("doc-").unwrap_or_default();
            suffix.parse::<u32>().is_ok_and(|v| v % 2 == 0)
        });

        let sequential = index
            .search_top_k_internal(
                &query,
                10,
                Some(&filter),
                usize::MAX,
                PARALLEL_CHUNK_SIZE,
                true,
            )
            .expect("sequential search");
        let parallel = index
            .search_top_k_internal(&query, 10, Some(&filter), 1, 8, true)
            .expect("parallel search");

        assert_eq!(sequential.len(), parallel.len());
        for (left, right) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(left.doc_id, right.doc_id);
            assert!((left.score - right.score).abs() < 1e-6);
        }
    }

    #[test]
    fn search_top_k_with_params_disabled_parallel() {
        let path = temp_index_path("with-params-disabled");
        write_index(
            &path,
            &[
                ("doc-a", vec![1.0, 0.0, 0.0, 0.0]),
                ("doc-b", vec![0.8, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0];

        let params = SearchParams {
            parallel_threshold: 1, // would trigger parallel...
            parallel_chunk_size: 1,
            parallel_enabled: false, // ...but disabled
        };
        let hits = index
            .search_top_k_with_params(&query, 2, None, params)
            .expect("search with disabled parallel");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(hits[1].doc_id, "doc-b");
    }

    // ─── bd-2k3d tests begin ──────────────────────────────────────────

    #[test]
    fn search_params_debug_clone_copy() {
        let params = SearchParams {
            parallel_threshold: 100,
            parallel_chunk_size: 32,
            parallel_enabled: true,
        };
        let debug = format!("{params:?}");
        assert!(debug.contains("SearchParams"));
        assert!(debug.contains("100"));

        let copied: SearchParams = params;
        assert_eq!(copied.parallel_threshold, 100);
        assert_eq!(copied.parallel_chunk_size, 32);

        let cloned = params;
        assert!(cloned.parallel_enabled);
    }

    #[test]
    fn compare_best_first_higher_score_wins() {
        let a = HeapEntry::new(0, 0.9);
        let b = HeapEntry::new(1, 0.5);
        assert_eq!(compare_best_first(&a, &b), Ordering::Less);
        assert_eq!(compare_best_first(&b, &a), Ordering::Greater);
    }

    #[test]
    fn compare_best_first_equal_scores_tiebreak_by_index() {
        let a = HeapEntry::new(2, 0.7);
        let b = HeapEntry::new(5, 0.7);
        assert_eq!(compare_best_first(&a, &b), Ordering::Less);
    }

    #[test]
    fn candidate_is_better_with_nan() {
        let good = HeapEntry::new(0, 0.5);
        let nan_entry = HeapEntry::new(1, f32::NAN);
        assert!(candidate_is_better(good, nan_entry));
        assert!(!candidate_is_better(nan_entry, good));
    }

    #[test]
    fn candidate_is_better_equal_scores_lower_index_wins() {
        let a = HeapEntry::new(3, 0.8);
        let b = HeapEntry::new(7, 0.8);
        assert!(candidate_is_better(a, b));
        assert!(!candidate_is_better(b, a));
    }

    #[test]
    fn score_key_maps_nan_to_neg_infinity() {
        assert_eq!(score_key(f32::NAN).to_bits(), f32::NEG_INFINITY.to_bits());
        assert!((score_key(0.5) - 0.5).abs() < f32::EPSILON);
        assert!((score_key(-0.3) - (-0.3)).abs() < f32::EPSILON);
        assert!((score_key(0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn merge_partial_heaps_empty_list() {
        let merged = merge_partial_heaps(vec![], 10, UncancelledScan).expect("uncancelled merge");
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_partial_heaps_single_heap() {
        let mut h = BinaryHeap::new();
        h.push(HeapEntry::new(0, 0.9));
        h.push(HeapEntry::new(1, 0.5));
        let merged = merge_partial_heaps(vec![h], 10, UncancelledScan).expect("uncancelled merge");
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn search_f32_quantization_index() {
        let path = temp_index_path("f32-quant-search");
        let dim = 4;
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", dim, Quantization::F32).unwrap();
        writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.write_record("doc-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        writer.write_record("doc-c", &[0.5, 0.5, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        assert_eq!(index.quantization(), Quantization::F32);

        let query = [1.0, 0.0, 0.0, 0.0];
        let hits = index.search_top_k(&query, 2, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[test]
    fn search_f32_with_filter() {
        let path = temp_index_path("f32-filter");
        let dim = 4;
        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", dim, Quantization::F32).unwrap();
        writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.write_record("doc-b", &[0.9, 0.1, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let index = VectorIndex::open(&path).unwrap();
        let filter = PredicateFilter::new("only-b", |doc_id: &str| doc_id == "doc-b");
        let query = [1.0, 0.0, 0.0, 0.0];
        let hits = index.search_top_k(&query, 10, Some(&filter)).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");
    }

    // ─── bd-2k3d tests end ────────────────────────────────────────────
}
