//! Brute-force top-k vector search over an opened [`crate::VectorIndex`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use ahash::AHashSet;

use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::filter::{DocIdHashSet, SearchFilter};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use rayon::prelude::*;

use crate::simd::dot_i8x4_i8;
use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{
    PreparedQuery4bit, Quantization, VectorIndex, dot_4bit_prepared, dot_i8_i8, dot_i8_i8_maddubs,
    dot_product_f16_bytes_f32, dot_product_f32_bytes_f32, dot_product_f32_f32, maddubs_query_bias,
    pack_f16_le_bytes_to_4bit, prepare_4bit_query, quantize_f16_le_bytes_to_i8,
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

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Behaves like [`Self::search_top_k`] with two fail-closed differences
    /// that align the exact path with the ANN path:
    /// - a query containing NaN or infinite components is rejected with
    ///   [`SearchError::InvalidConfig`] instead of silently scoring garbage
    ///   (parity with `HnswIndex`);
    /// - an empty result always carries a typed
    ///   [`ZeroSignalReason`], so a legitimate empty answer is
    ///   distinguishable from an unusable semantic lane.
    ///
    /// Classification is lazy: the non-empty path costs nothing extra, and
    /// an empty result pays one census pass comparable to the scan that
    /// just ran.
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
        self.ensure_query_dimension(query)?;
        if limit == 0 {
            return Ok(ClassifiedHits::empty(
                ZeroSignalReason::CallerRequestedZeroK,
            ));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "<contains non-finite values>".to_owned(),
                reason: "query vector must be finite".to_owned(),
            });
        }
        if query.iter().all(|&value| value == 0.0) {
            return Ok(ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery));
        }
        let hits = self.search_top_k(query, limit, filter)?;
        if hits.is_empty() {
            let reason = self.classify_empty_result(filter.is_some());
            return Ok(ClassifiedHits {
                hits,
                zero_signal: Some(reason),
            });
        }
        Ok(ClassifiedHits {
            hits,
            zero_signal: None,
        })
    }

    /// Classify why a well-formed search (k > 0, finite non-zero query)
    /// returned nothing, following the precedence documented on
    /// [`ZeroSignalReason`].
    pub(crate) fn classify_empty_result(&self, had_filter: bool) -> ZeroSignalReason {
        self.zero_signal_state().empty_result_reason(had_filter)
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
        let mut hits = self.search_main_top_k_raw(query, limit)?;
        let mut seen = AHashSet::with_capacity(hits.len());
        hits.retain(|hit| seen.insert(hit.doc_id.clone()));
        Ok(hits)
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
            self.scan_parallel(query, limit, None, PARALLEL_CHUNK_SIZE)
        } else {
            self.scan_sequential(query, limit, None)
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
                self.scan_parallel(query, limit, filter, PARALLEL_CHUNK_SIZE)?
            } else {
                self.scan_sequential(query, limit, filter)?
            }
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if has_wal {
            self.scan_wal(query, &mut heap, limit, filter)?;
        }
        self.resolve_hits(heap)
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
        let Some(allowed) = filter.candidate_hashes() else {
            return self.bench_scan_filtered(query, limit, Some(filter));
        };
        let mut heap = if self.record_count() > 0 {
            self.scan_gather_hashes(allowed, query, limit)?
        } else {
            BinaryHeap::with_capacity(limit.min(self.wal_entries.len()).saturating_add(1))
        };
        if !self.wal_entries.is_empty() {
            self.scan_wal(query, &mut heap, limit, Some(filter))?;
        }
        self.resolve_hits(heap)
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
        self.ensure_query_dimension(query)?;
        let has_main = self.record_count() > 0;
        let has_wal = !self.wal_entries.is_empty();
        if limit == 0 || (!has_main && !has_wal) {
            return Ok(Vec::new());
        }
        let chunk_size = parallel_chunk_size.max(1);
        let use_parallel = parallel_enabled && self.record_count() >= parallel_threshold;
        let total_candidate_upper_bound =
            self.record_count().saturating_add(self.wal_entries.len());

        // Full-recall requests should avoid top-k heap churn.
        // When the caller asks for all available candidates (`k >= total`),
        // collect-and-sort is measurably faster than maintaining a size-k heap.
        if filter.is_none() && limit >= total_candidate_upper_bound {
            let mut winners = if has_main {
                if use_parallel {
                    self.scan_parallel_collect_all(query, chunk_size)?
                } else {
                    self.scan_range_collect_all(0, self.record_count(), query)?
                }
            } else {
                Vec::new()
            };
            if has_wal {
                self.scan_wal_collect_all(query, &mut winners)?;
            }
            // `limit_all` scan-all path: `winners` can hold every match. Above a
            // threshold the final sort dominates, and a parallel sort pays
            // (measured ~2.81× at 50k winners, `winners_sort` bench); below it the
            // rayon overhead is not worth it, so stay serial. Bit-identical either
            // way — `compare_best_first` is a strict total order.
            if winners.len() >= PAR_SORT_THRESHOLD {
                winners.par_sort_unstable_by(compare_best_first);
            } else {
                winners.sort_unstable_by(compare_best_first);
            }
            return self.resolve_sorted_entries(winners);
        }

        let mut heap = if has_main {
            if let Some(gathered) = self.try_gather_filtered(query, limit, filter)? {
                gathered
            } else if use_parallel {
                self.scan_parallel(query, limit, filter, chunk_size)?
            } else {
                self.scan_sequential(query, limit, filter)?
            }
        } else {
            let max_wal = self.wal_entries.len();
            BinaryHeap::with_capacity(limit.min(max_wal).saturating_add(1))
        };

        // Merge WAL entries into the same heap.
        if has_wal {
            self.scan_wal(query, &mut heap, limit, filter)?;
        }

        self.resolve_hits(heap)
    }

    /// int8 ADC two-pass exact top-k for **standalone** large-N vector search:
    /// a fast parallel int8 pass-1 over all main records keeps the top
    /// `k·candidate_multiplier` by approximate score, then an exact f16 rescore of
    /// just those candidates selects the final top-k. Lossless (recall=1.0) whenever
    /// pass-1 retains the true top-k — validated on the in-memory twin; the int8
    /// dot is monotonic with the true dot under one corpus max-abs scale.
    ///
    /// Covers the contiguous F16 main-vector region only; falls back to the exact
    /// [`VectorIndex::search_top_k`] when a WAL is present or quantization is not F16
    /// (so results are always correct, never silently degraded). Not wired into the
    /// BOLD hybrid (that gap is not vector-bound — see `docs/NEGATIVE_EVIDENCE.md`);
    /// this targets pure vector-search latency at large N.
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
        // Production keeps the EXACT-int8 pass-1. The `vpmaddubs` kernel (bd-b5wl) is 1.23× faster
        // in isolation (decidable) and recall-exact, but its **scan-level** win is only ~1.02–1.11×
        // (Amdahl-shrunk) and is NOT robustly decidable under fleet contention: two null-controlled
        // runs on `hetzner1` disagreed — 0.9023 (median below null p5, a clear win) then 0.9821
        // (inside the null floor). Shipping the approximate kernel as default on a marginal,
        // contention-dependent effect fails the gate, so it stays behind
        // `bench_search_top_k_int8_two_pass_maddubs`. Retry = worker isolation (same as cod's int8
        // micro-opt block). See docs/NEGATIVE_EVIDENCE.md 2026-07-10.
        self.search_top_k_int8_two_pass_impl::<false, false>(query, k, candidate_multiplier)
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
        self.search_top_k_int8_two_pass_impl::<false, false>(query, k, candidate_multiplier)
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
        self.search_top_k_int8_two_pass_impl::<true, false>(query, k, candidate_multiplier)
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
        self.search_top_k_int8_two_pass_impl::<false, true>(query, k, candidate_multiplier)
    }

    fn search_top_k_int8_two_pass_impl<const ROW_BLOCKED: bool, const MADDUBS: bool>(
        &self,
        query: &[f32],
        k: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        // Fall back to the exact scan for anything this fast path does not cover.
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
        let candidate_count = k
            .saturating_mul(candidate_multiplier.max(1))
            .min(count)
            .max(k.min(count));
        let query_i8 = quantize_i8_query(query);
        // Per-query bias `128·Σq` for the `MADDUBS` pass-1 kernel; unused (0) otherwise.
        let q_bias128 = if MADDUBS {
            maddubs_query_bias(&query_i8, dim)
        } else {
            0
        };
        let slab = self.int8_slab();

        // Pass 1: bounded-heap int8 scan keeping the top `candidate_count`.
        // The int8 dot is cheap enough that exact-scan sized chunks overproduce
        // local top-N heaps; larger chunks keep enough Rayon tasks while shrinking
        // the post-scan merge fan-in.
        let candidate_heap = if count < PARALLEL_THRESHOLD {
            if ROW_BLOCKED {
                self.int8_scan_range(slab, &query_i8, 0, count, candidate_count)
            } else {
                self.int8_scan_range_orig::<MADDUBS>(
                    slab,
                    &query_i8,
                    q_bias128,
                    0,
                    count,
                    candidate_count,
                )
            }
        } else {
            let chunk_count = count.div_ceil(INT8_PARALLEL_CHUNK_SIZE);
            let partials: Vec<BinaryHeap<Int8HeapKey>> = (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * INT8_PARALLEL_CHUNK_SIZE;
                    let end = (start + INT8_PARALLEL_CHUNK_SIZE).min(count);
                    if ROW_BLOCKED {
                        self.int8_scan_range(slab, &query_i8, start, end, candidate_count)
                    } else {
                        self.int8_scan_range_orig::<MADDUBS>(
                            slab,
                            &query_i8,
                            q_bias128,
                            start,
                            end,
                            candidate_count,
                        )
                    }
                })
                .collect();
            merge_int8_partial_heaps(partials, candidate_count)
        };

        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as `search_top_k`, so the final order is identical
        // whenever pass-1 retained the true top-k.
        let stride = dim * 2;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        for candidate in candidate_heap {
            let index = int8_heap_index(candidate);
            let vector_offset = self.vectors_offset + index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(index, score), k);
        }
        self.resolve_hits(heap)
    }

    fn int8_scan_range_orig<const MADDUBS: bool>(
        &self,
        slab: &[i8],
        query_i8: &[i8],
        q_bias128: i32,
        start: usize,
        end: usize,
        limit: usize,
    ) -> BinaryHeap<Int8HeapKey> {
        if limit == 0 {
            return BinaryHeap::new();
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
    ) -> BinaryHeap<Int8HeapKey>
    where
        F: Fn(usize, i32) -> Int8HeapKey,
    {
        let dim = self.dimension();
        let mut heap = BinaryHeap::with_capacity(limit.min(end - start).saturating_add(1));
        let mut cutoff = Int8HeapKey::MAX;
        let mut flags_offset = self.records_offset + start * 16 + 14;
        let mut slab_offset = start * dim;

        for index in start..end {
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
        heap
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
    ) -> BinaryHeap<Int8HeapKey> {
        if limit == 0 {
            return BinaryHeap::new();
        }
        if self.dimension() <= MAX_EXACT_I8_DOT_DIM {
            self.int8_scan_range_with_key(slab, query_i8, start, end, limit, int8_heap_key)
        } else {
            self.int8_scan_range_with_key(slab, query_i8, start, end, limit, int8_heap_key_from_f32)
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
    ) -> BinaryHeap<Int8HeapKey>
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
        heap
    }

    /// Lazily build (once) the int8 quantization of the contiguous F16 main-vector
    /// region. Only called after the F16/no-WAL gate in `search_top_k_int8_two_pass`.
    fn int8_slab(&self) -> &[i8] {
        self.vectors_i8.get_or_init(|| {
            let count = self.record_count();
            let dim = self.dimension();
            let byte_len = count * dim * 2;
            quantize_f16_le_bytes_to_i8(
                &self.data[self.vectors_offset..self.vectors_offset + byte_len],
            )
        })
    }

    /// 4-bit (16-level) two-pass exact top-k for standalone large-N vector search.
    /// A fast parallel pass-1 over a packed signed-4-bit slab (`dim/2` bytes/vector —
    /// half the int8 slab, so the bandwidth-bound pass-1 is faster) keeps the top
    /// `k·candidate_multiplier` by approximate score (`dot_4bit_prepared`), then
    /// an exact f16 rescore of just those candidates selects the final top-k. 16
    /// levels stay lossless at mult≈5 on realistic clustered data (see
    /// `fsvi_4bit_two_pass` bench); recall rises with `candidate_multiplier`.
    /// Falls back to the exact `search_top_k` for WAL/non-F16 indexes. Not wired
    /// into the BOLD hybrid.
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
            merge_partial_heaps(partials, candidate_count)
        };

        // Pass 2: exact f16 rescore (same bounded-heap selection + tie-break).
        let stride = dim * 2;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        for candidate in candidate_heap {
            let vector_offset = self.vectors_offset + candidate.index * stride;
            let vector_bytes = &self.data[vector_offset..vector_offset + stride];
            let score = dot_product_f16_bytes_f32(vector_bytes, query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), k);
        }
        self.resolve_hits(heap)
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
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        // Re-use the parallel chunk logic for sequential scan to benefit from optimizations.
        filter.map_or_else(
            || self.scan_range_chunk(0, self.record_count(), query, limit),
            |filter| self.scan_range_chunk_filtered(0, self.record_count(), query, limit, filter),
        )
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                filter.map_or_else(
                    || self.scan_range_chunk(start, end, query, limit),
                    |active_filter| {
                        self.scan_range_chunk_filtered(start, end, query, limit, active_filter)
                    },
                )
            })
            .collect();

        Ok(merge_partial_heaps(partial_heaps?, limit))
    }

    fn scan_parallel_collect_all(
        &self,
        query: &[f32],
        chunk_size: usize,
    ) -> SearchResult<Vec<HeapEntry>> {
        let chunk_count = self.record_count().div_ceil(chunk_size);
        let partial: SearchResult<Vec<Vec<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(self.record_count());
                self.scan_range_collect_all(start, end, query)
            })
            .collect();

        let partial = partial?;
        let total = partial.iter().map(std::vec::Vec::len).sum();
        let mut merged = Vec::with_capacity(total);
        for mut chunk in partial {
            merged.append(&mut chunk);
        }
        Ok(merged)
    }

    fn scan_range_collect_all(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
    ) -> SearchResult<Vec<HeapEntry>> {
        let mut winners = Vec::with_capacity(end.saturating_sub(start));
        let dim = self.dimension();

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                let mut flags_offset = self.records_offset + start * 16 + 14;
                let mut vector_offset = self.vectors_offset + start * stride;

                for index in start..end {
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
        Ok(winners)
    }

    fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Option<BinaryHeap<HeapEntry>>> {
        let Some(active_filter) = filter else {
            return Ok(None);
        };
        let Some(allowed) = active_filter.candidate_hashes() else {
            return Ok(None);
        };
        let count = self.record_count();
        if count == 0 || allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        self.scan_gather_hashes(allowed, query, limit).map(Some)
    }

    fn scan_gather_hashes(
        &self,
        allowed: &DocIdHashSet,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let positions = self.gather_positions_for_hashes(allowed)?;
        self.scan_gather_positions(&positions, query, limit)
    }

    fn gather_positions_for_hashes(&self, allowed: &DocIdHashSet) -> SearchResult<Vec<usize>> {
        let mut positions = Vec::with_capacity(allowed.len());
        for &hash in allowed {
            let Some((start, end)) = self.hash_range(hash)? else {
                continue;
            };
            for index in start..end {
                let entry = self.record_at(index)?;
                if (entry.flags & 0x0001) == 0 {
                    positions.push(index);
                }
            }
        }
        positions.sort_unstable();
        Ok(positions)
    }

    fn hash_range(&self, hash: u64) -> SearchResult<Option<(usize, usize)>> {
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
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        if positions.len() > PARALLEL_CHUNK_SIZE {
            let partials: SearchResult<Vec<BinaryHeap<HeapEntry>>> = positions
                .par_chunks(PARALLEL_CHUNK_SIZE)
                .map(|chunk| self.gather_range(chunk, query, limit))
                .collect();
            return Ok(merge_partial_heaps(partials?, limit));
        }
        self.gather_range(positions, query, limit)
    }

    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.min(positions.len()).saturating_add(1));
        let dim = self.dimension();
        let mut cutoff = f32::NEG_INFINITY;

        match self.quantization() {
            Quantization::F16 => {
                let stride = dim * 2;
                for &index in positions {
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
                for &index in positions {
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
        Ok(heap)
    }

    fn scan_range_chunk(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
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
        Ok(heap)
    }

    fn scan_range_chunk_filtered(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
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
        Ok(heap)
    }

    fn scan_wal(
        &self,
        query: &[f32],
        heap: &mut BinaryHeap<HeapEntry>,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<()> {
        for (idx, entry) in self.wal_entries.iter().enumerate() {
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
        Ok(())
    }

    fn scan_wal_collect_all(
        &self,
        query: &[f32],
        winners: &mut Vec<HeapEntry>,
    ) -> SearchResult<()> {
        winners.reserve(self.wal_entries.len());
        for (idx, entry) in self.wal_entries.iter().enumerate() {
            let score = dot_product_f32_f32(&entry.embedding, query)?;
            if !score.is_finite() {
                continue;
            }
            winners.push(HeapEntry::new(to_wal_index(idx), score));
        }
        Ok(())
    }

    fn resolve_hits(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }

        let mut winners = heap.into_vec();
        winners.sort_unstable_by(compare_best_first);
        self.resolve_sorted_entries(winners)
    }

    fn resolve_sorted_entries(&self, winners: Vec<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        // Pre-build a hash set of WAL doc_id hashes for O(1) pre-screening
        // instead of O(W) linear scan per main-index winner. On hash match,
        // falls back to string verification for correctness.
        let wal_hashes: AHashSet<u64> = self.wal_entries.iter().map(|e| e.doc_id_hash).collect();

        let mut seen: AHashSet<String> = AHashSet::with_capacity(winners.len());
        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            if is_wal_index(winner.index) {
                let wal_idx = from_wal_index(winner.index);
                let doc_id = &self.wal_entries[wal_idx].doc_id;
                // Skip WAL-vs-WAL duplicates (keep the first, i.e. highest-scored).
                if !seen.insert(doc_id.clone()) {
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
                    let has_wal_entry = self
                        .wal_entries
                        .iter()
                        .any(|e| e.doc_id_hash == doc_id_hash && e.doc_id == doc_id);
                    if has_wal_entry {
                        continue;
                    }
                }
                // Skip main-vs-main duplicates (String-based for correctness).
                if !seen.insert(doc_id.clone()) {
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
fn quantize_i8_query(query: &[f32]) -> Vec<i8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0; query.len()];
    }
    let scale = 127.0 / max_abs;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
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
) -> BinaryHeap<HeapEntry> {
    let mut total_elements = 0_usize;
    for heap in &partial_heaps {
        total_elements = total_elements.saturating_add(heap.len());
    }
    let capacity = limit.min(total_elements).saturating_add(1);
    let mut merged = BinaryHeap::with_capacity(capacity);
    for heap in partial_heaps {
        for entry in heap {
            insert_candidate(&mut merged, entry, limit);
        }
    }
    merged
}

fn merge_int8_partial_heaps(
    partial_heaps: Vec<BinaryHeap<Int8HeapKey>>,
    limit: usize,
) -> BinaryHeap<Int8HeapKey> {
    let total_elements = partial_heaps
        .iter()
        .map(BinaryHeap::len)
        .fold(0_usize, usize::saturating_add);
    let mut merged = BinaryHeap::with_capacity(limit.min(total_elements).saturating_add(1));
    for heap in partial_heaps {
        for candidate in heap {
            if merged.len() < limit {
                merged.push(candidate);
            } else if merged.peek().is_some_and(|&worst| candidate < worst) {
                let _ = merged.pop();
                merged.push(candidate);
            }
        }
    }
    merged
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
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::{Quantization, VectorIndex};
    use frankensearch_core::PredicateFilter;
    use proptest::prelude::*;

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-index-search-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
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
            let _ = fs::remove_file(&path);
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
            let _ = fs::remove_file(&path);
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

            let mut legacy_indices = merge_partial_heaps(legacy_parts, limit)
                .into_iter()
                .map(|entry| entry.index)
                .collect::<Vec<_>>();
            let mut packed_indices = merge_int8_partial_heaps(packed_parts, limit)
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

        let merged = merge_partial_heaps(vec![heap_a, heap_b], 3);
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
        let merged = merge_partial_heaps(vec![], 10);
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_partial_heaps_single_heap() {
        let mut h = BinaryHeap::new();
        h.push(HeapEntry::new(0, 0.9));
        h.push(HeapEntry::new(1, 0.5));
        let merged = merge_partial_heaps(vec![h], 10);
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

        fs::remove_file(&path).ok();
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

        fs::remove_file(&path).ok();
    }

    // ─── bd-2k3d tests end ────────────────────────────────────────────
}
