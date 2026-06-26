//! In-memory vector index for zero-latency search.
//!
//! Unlike the file-backed [`crate::VectorIndex`] (memory-mapped FSVI), this
//! module stores all vectors in heap-allocated memory, guaranteeing no page
//! faults on access. Vectors are stored as f16 for 50% memory savings.
//!
//! # Usage
//!
//! ```rust,ignore
//! use frankensearch_index::in_memory::InMemoryVectorIndex;
//!
//! // From pre-computed f32 vectors
//! let index = InMemoryVectorIndex::from_vectors(
//!     doc_ids,
//!     vectors,
//!     256,
//! ).unwrap();
//!
//! let hits = index.search_top_k(&query, 10, None).unwrap();
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::OnceLock;

use frankensearch_core::filter::{SearchFilter, fnv1a_hash};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::prelude::*;

use crate::VectorIndex;
use crate::search::{PARALLEL_CHUNK_SIZE, SearchParams};
use crate::simd::{dot_i8_i8, dot_product_f16_f32};

/// Fully-resident in-memory vector index with f16 quantization.
///
/// All vectors are stored in a contiguous `Vec<f16>` in row-major order,
/// eliminating memory-map page faults for deterministic sub-millisecond search.
#[derive(Debug, Clone)]
pub struct InMemoryVectorIndex {
    /// Document IDs, indexed by position.
    doc_ids: Vec<String>,
    /// Flat f16 vector slab: `doc_ids.len() * dimension` elements.
    vectors: Vec<f16>,
    /// Lazily-built flat int8 vector slab (same row-major layout) for the int8 ADC
    /// pass-1 of [`InMemoryVectorIndex::search_top_k_int8_two_pass`]. Quantized with
    /// a single corpus-wide max-abs scale, which preserves the dot-product ranking
    /// (the scale is a per-query constant). Built on first two-pass use so exact-only
    /// callers pay neither the quantization work nor its `N·d`-byte footprint.
    vectors_i8: OnceLock<Vec<i8>>,
    /// Lazily-built FNV-1a hashes of `doc_ids` (same `frankensearch_core::filter`
    /// hash that `BitsetFilter` uses). Lets the filtered scan call
    /// `SearchFilter::matches_doc_id_hash` with a precomputed hash instead of
    /// re-hashing each `doc_id` string per vector (the FSVI `search.rs` scan already
    /// does this). Built on first *filtered* search, so unfiltered callers pay
    /// neither the hashing nor the `8·N`-byte footprint.
    doc_id_hashes: OnceLock<Vec<u64>>,
    /// Vector dimensionality.
    dimension: usize,
}

/// Quantize an f16 slab to int8 using one corpus-wide max-abs scale.
///
/// Symmetric int8: `q = round(x / max_abs * 127)`, clamped to `[-127, 127]`. A
/// single global scale keeps `Σ q_a·q_b` monotonic with the true dot for a fixed
/// query, so pass-1 ranking is preserved; the exact f16 rescore restores values.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_slab(vectors_f16: &[f16]) -> Vec<i8> {
    let max_abs = vectors_f16
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; vectors_f16.len()];
    }
    let scale = 127.0 / max_abs;
    vectors_f16
        .iter()
        .map(|x| (x.to_f32() * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Quantize an f32 query to int8 using its own max-abs scale (the scale is a
/// per-query constant and does not affect ranking).
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the f32->i8 cast
fn quantize_i8_query(query: &[f32]) -> Vec<i8> {
    let max_abs = query.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; query.len()];
    }
    let scale = 127.0 / max_abs;
    query
        .iter()
        .map(|&x| (x * scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

impl InMemoryVectorIndex {
    /// Build from pre-computed f32 vectors, quantizing to f16.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if any vector's length does not
    /// match `dimension`.
    pub fn from_vectors(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
    ) -> SearchResult<Self> {
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: format!("doc_ids={}, vectors={}", doc_ids.len(), vectors.len()),
                reason: "doc_ids and vectors must have the same length".to_owned(),
            });
        }
        let count = doc_ids.len();
        let mut flat = Vec::with_capacity(count * dimension);
        for (i, vec) in vectors.into_iter().enumerate() {
            if vec.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vec.len(),
                });
            }
            // Validate finite values
            for val in &vec {
                if !val.is_finite() {
                    return Err(SearchError::InvalidConfig {
                        field: "vectors".to_owned(),
                        value: format!("vector[{i}] contains non-finite value"),
                        reason: "all vector elements must be finite".to_owned(),
                    });
                }
            }
            flat.extend(vec.into_iter().map(f16::from_f32));
        }
        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            dimension,
        })
    }

    /// Load from an existing FSVI file, reading all data into memory.
    ///
    /// This reads the entire file-backed index into heap memory, eliminating
    /// page-fault latency on subsequent searches.
    ///
    /// # Errors
    ///
    /// Returns errors from [`VectorIndex::open`] or vector decoding failures.
    pub fn from_fsvi(path: &Path) -> SearchResult<Self> {
        let index = VectorIndex::open(path)?;
        let count = index.record_count();
        let dimension = index.dimension();
        let mut doc_ids = Vec::with_capacity(count);
        let mut flat = Vec::with_capacity(count * dimension);

        for i in 0..count {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            let f16_vec = index.vector_at_f16(i)?;
            flat.extend_from_slice(&f16_vec);
        }

        for entry in &index.wal_entries {
            doc_ids.push(entry.doc_id.clone());
            let f16_vec: Vec<half::f16> = entry
                .embedding
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            flat.extend_from_slice(&f16_vec);
        }

        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            dimension,
        })
    }

    /// Number of vectors in the index.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the document ID at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn doc_id_at(&self, index: usize) -> SearchResult<&str> {
        self.doc_ids
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.doc_ids.len()
                ),
            })
    }

    /// Get the f16 vector slice at position `index`.
    fn vector_slice(&self, index: usize) -> &[f16] {
        let start = index * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Brute-force cosine-similarity top-k search.
    ///
    /// Query must be pre-normalized. Uses f16→f32 SIMD dot product.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_with_params(query, limit, filter, SearchParams::default())
    }

    /// Brute-force top-k search with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_with_params(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        params: SearchParams,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }

        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;
        let chunk_size = params.parallel_chunk_size.max(1);

        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, chunk_size)?
        } else {
            self.scan_sequential(query, limit, filter)?
        };

        self.resolve_heap(heap)
    }

    /// Approximate top-k via an **int8 ADC two-pass** (`bd-b5wl`): an int8 pass-1
    /// over all vectors keeps the top `limit * candidate_multiplier` candidates,
    /// then an exact f16 rescore with the same deterministic selection as
    /// [`Self::search_top_k`] produces the final ranking.
    ///
    /// Results are **bit-identical** to [`Self::search_top_k`] whenever pass-1
    /// retains the true top-k (recall = 1). Measured ~1.4–1.5× faster than the
    /// parallel exact path across 10k–100k at `candidate_multiplier = 5` (int8 is
    /// half the bytes + an integer `mul_widen` MAC — see `docs/PERF_LEDGER.md`), at
    /// the cost of *approximate* recall.
    ///
    /// **Tuning `candidate_multiplier`:** recall@10 = 1.0 held down to `mult = 2`
    /// for well-separated (random) vectors, so `mult = 5` is a good default; the
    /// candidate budget (`limit * mult`) is the selection-overhead knob — smaller is
    /// faster. Clustered real embeddings have closer neighbours, so re-measure recall
    /// on a representative corpus and raise `mult` if needed.
    ///
    /// Falls back to the exact path when the int8 slab is unavailable.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_int8_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_int8_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// int8 ADC two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same `matches_doc_id_hash` path
    /// the exact scan uses), so **filtered** large-N searches get the int8 speedup
    /// instead of falling back to the exact scan. The result matches the exact
    /// filtered top-k whenever pass-1 retains the true filtered top-k.
    pub fn search_top_k_int8_two_pass_filtered(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let query_i8 = quantize_i8_query(query);
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Build the int8 slab once, on first use — exact-only callers never pay the
        // O(N·d) quantization or its `N·d`-byte footprint at construction time.
        let vectors_i8 = self
            .vectors_i8
            .get_or_init(|| quantize_i8_slab(&self.vectors));

        // Pass 1: parallel **bounded-heap** int8 scan — each chunk keeps only its
        // top `candidate_count` (never materializing all N scores, unlike a
        // collect-all + select), then merge. This mirrors the exact path's
        // structure so the 3× int8 dot win is not eaten by selection overhead.
        // int8 dots peak well below 2^24 for realistic dims, so `i32 as f32` is
        // exact and preserves the candidate ranking + deterministic index tie-break.
        // Match the exact path's chunking so pass-1 uses all cores, not ~2.
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        // When filtering, build the precomputed doc_id-hash slab once here (before
        // the parallel section) so pass-1 pre-screens by a hash lookup instead of
        // re-hashing each doc_id string per vector.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: Vec<BinaryHeap<HeapEntry>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = BinaryHeap::with_capacity(candidate_count.min(end - start) + 1);
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = match doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                        {
                            Some(decided) => decided,
                            None => f.matches(&self.doc_ids[index], None),
                        };
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * self.dimension;
                    let stored = &vectors_i8[offset..offset + self.dimension];
                    let score = dot_i8_i8(stored, &query_i8) as f32;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                }
                heap
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials, candidate_count);

        // Pass 2: exact f16 rescore of the candidates through the SAME bounded-heap
        // selection + tie-break as the exact path, so the final order matches
        // `search_top_k` exactly whenever pass-1 retained the true top-k.
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        for candidate in candidate_heap {
            let score = dot_product_f16_f32(self.vector_slice(candidate.index), query)?;
            insert_candidate(&mut heap, HeapEntry::new(candidate.index, score), limit);
        }
        self.resolve_heap(heap)
    }

    fn scan_sequential(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        self.scan_range(0, self.record_count(), query, limit, filter)
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let count = self.record_count();
        let chunk_count = count.div_ceil(chunk_size);
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                self.scan_range(start, end, query, limit, filter)
            })
            .collect();

        Ok(merge_partial_heaps(partial_heaps?, limit))
    }

    /// Lazily-built FNV-1a hashes of every `doc_id` (matches the hash
    /// `BitsetFilter` computes), so the filtered scan can pre-screen by hash via
    /// `SearchFilter::matches_doc_id_hash` instead of re-hashing each `doc_id`
    /// string per vector. Built once on first filtered search.
    fn doc_id_hashes(&self) -> &[u64] {
        self.doc_id_hashes.get_or_init(|| {
            self.doc_ids
                .iter()
                .map(|id| fnv1a_hash(id.as_bytes()))
                .collect()
        })
    }

    fn scan_range(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let max_elements = end.saturating_sub(start);
        let mut heap = BinaryHeap::with_capacity(limit.min(max_elements).saturating_add(1));
        let mut cutoff = f32::NEG_INFINITY;

        // When filtering, pre-screen by precomputed doc_id hash (one HashSet lookup)
        // instead of re-hashing the doc_id string per vector; fall back to the
        // string/metadata path only when the filter can't decide by hash.
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());

        for index in start..end {
            if let Some(f) = filter {
                let passed = match doc_id_hashes.and_then(|h| f.matches_doc_id_hash(h[index], None))
                {
                    Some(decided) => decided,
                    None => f.matches(&self.doc_ids[index], None),
                };
                if !passed {
                    continue;
                }
            }
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit
                    && let Some(&worst) = heap.peek()
                {
                    cutoff = score_key(worst.score);
                }
            }
        }
        Ok(heap)
    }

    fn resolve_heap(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }
        let mut winners = heap.into_vec();
        winners.sort_by(compare_best_first);
        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            let index_u32 =
                u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                    field: "index".to_owned(),
                    value: winner.index.to_string(),
                    reason: "index exceeds u32 range for VectorHit".to_owned(),
                })?;
            hits.push(VectorHit {
                index: index_u32,
                score: winner.score,
                doc_id: self.doc_ids[winner.index].clone(),
            });
        }
        Ok(hits)
    }

    /// Iterate over all document IDs.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.doc_ids.iter().map(String::as_str)
    }

    /// Get the f32 vector at position `index`.
    ///
    /// # Errors
    ///
    /// Returns error if index is out of bounds.
    pub fn vector_at_f32(&self, index: usize) -> SearchResult<Vec<f32>> {
        if index >= self.record_count() {
            return Err(SearchError::InvalidConfig {
                field: "index".to_owned(),
                value: index.to_string(),
                reason: format!(
                    "index {} out of bounds (record_count = {})",
                    index,
                    self.record_count()
                ),
            });
        }
        let stored = self.vector_slice(index);
        Ok(stored.iter().map(|v| v.to_f32()).collect())
    }

    /// Compute dot products between a query and specific hit positions.
    ///
    /// Used for quality scoring when this index serves as the quality tier.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn scores_for_hits(&self, query: &[f32], hits: &[VectorHit]) -> SearchResult<Vec<f32>> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            // Try to find by doc_id
            let score = self
                .doc_ids
                .iter()
                .position(|id| id == &hit.doc_id)
                .map(|idx| {
                    let stored = self.vector_slice(idx);
                    dot_product_f16_f32(stored, query)
                })
                .transpose()?
                .unwrap_or(0.0);
            scores.push(score);
        }
        Ok(scores)
    }
}

/// In-memory two-tier index wrapping fast and optional quality `InMemoryVectorIndex`.
///
/// Provides the same `search_fast()` / `quality_scores_for_hits()` API as
/// [`crate::TwoTierIndex`] but with fully-resident memory for deterministic latency.
#[derive(Debug, Clone)]
pub struct InMemoryTwoTierIndex {
    fast_index: InMemoryVectorIndex,
    quality_index: Option<InMemoryVectorIndex>,
}

impl InMemoryTwoTierIndex {
    /// Create from two pre-built in-memory indices.
    #[must_use]
    pub const fn new(
        fast_index: InMemoryVectorIndex,
        quality_index: Option<InMemoryVectorIndex>,
    ) -> Self {
        Self {
            fast_index,
            quality_index,
        }
    }

    /// Load from an existing two-tier index directory, reading all data into memory.
    ///
    /// Looks for `vector.fast.idx` (required) and `vector.quality.idx` (optional).
    /// Falls back to `vector.idx` if the fast filename doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns errors from FSVI parsing or vector loading.
    pub fn from_dir(dir: &Path) -> SearchResult<Self> {
        let fast_path = dir.join(crate::two_tier::VECTOR_INDEX_FAST_FILENAME);
        let fast_path = if fast_path.exists() {
            fast_path
        } else {
            let fallback = dir.join(crate::two_tier::VECTOR_INDEX_FALLBACK_FILENAME);
            if !fallback.exists() {
                return Err(SearchError::IndexNotFound { path: fast_path });
            }
            fallback
        };
        let fast_index = InMemoryVectorIndex::from_fsvi(&fast_path)?;

        let quality_path = dir.join(crate::two_tier::VECTOR_INDEX_QUALITY_FILENAME);
        let quality_index = if quality_path.exists() {
            Some(InMemoryVectorIndex::from_fsvi(&quality_path)?)
        } else {
            None
        };

        Ok(Self {
            fast_index,
            quality_index,
        })
    }

    /// Search the fast tier.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k`].
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        self.fast_index.search_top_k(query_vec, k, None)
    }

    /// Search the fast tier with configurable parallelism.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`InMemoryVectorIndex::search_top_k_with_params`].
    pub fn search_fast_with_params(
        &self,
        query_vec: &[f32],
        k: usize,
        params: Option<SearchParams>,
    ) -> SearchResult<Vec<VectorHit>> {
        let params = params.unwrap_or_default();
        self.fast_index
            .search_top_k_with_params(query_vec, k, None, params)
    }

    /// Compute quality-tier scores for fast-index hits.
    ///
    /// Missing quality entries produce `0.0`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` doesn't match
    /// the quality index dimensionality.
    pub fn quality_scores_for_hits(
        &self,
        query_vec: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<Option<f32>>> {
        let Some(quality) = &self.quality_index else {
            return Ok(vec![None; hits.len()]);
        };
        if query_vec.len() != quality.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: quality.dimension,
                found: query_vec.len(),
            });
        }
        let mut scores = Vec::with_capacity(hits.len());
        for hit in hits {
            let score = quality
                .doc_ids
                .iter()
                .position(|id| id == &hit.doc_id)
                .map(|idx| dot_product_f16_f32(quality.vector_slice(idx), query_vec))
                .transpose()?;
            scores.push(score);
        }
        Ok(scores)
    }

    /// Whether a quality index is loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_index.is_some()
    }

    /// Number of documents in the fast tier.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.fast_index.record_count()
    }

    /// Iterate over all document IDs in fast-tier order.
    pub fn iter_doc_ids(&self) -> impl Iterator<Item = &str> {
        self.fast_index.iter_doc_ids()
    }

    /// Get a reference to the fast index.
    #[must_use]
    pub const fn fast_index(&self) -> &InMemoryVectorIndex {
        &self.fast_index
    }

    /// Get a reference to the quality index (if present).
    #[must_use]
    pub const fn quality_index(&self) -> Option<&InMemoryVectorIndex> {
        self.quality_index.as_ref()
    }
}

// ─── Heap helpers (mirrors search.rs internals) ─────────────────────────────

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
        // Min-heap: "largest" == worst score, so peek() returns cutoff.
        match score_key(self.score).total_cmp(&score_key(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

const fn score_key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

fn compare_best_first(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    match score_key(right.score).total_cmp(&score_key(left.score)) {
        Ordering::Equal => left.index.cmp(&right.index),
        other => other,
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
        && match score_key(candidate.score).total_cmp(&score_key(worst.score)) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => candidate.index < worst.index,
        }
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
    for partial in partial_heaps {
        for entry in partial {
            insert_candidate(&mut merged, entry, limit);
        }
    }
    merged
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::items_after_statements,
        clippy::redundant_clone,
        clippy::suboptimal_flops,
        clippy::unnecessary_literal_bound
    )]

    use super::*;
    use crate::Quantization;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    fn temp_index_path(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nonce = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join("frankensearch_in_memory_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir.join(format!("{name}-{nonce}.fsvi"))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("fsvi.wal"));
    }

    fn make_normalized_vec(dim: usize, seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (seed + i as f32 * 0.1).sin()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn from_vectors_basic() {
        let dim = 8;
        let doc_ids = vec!["a".into(), "b".into(), "c".into()];
        let vectors = vec![
            make_normalized_vec(dim, 1.0),
            make_normalized_vec(dim, 2.0),
            make_normalized_vec(dim, 3.0),
        ];
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        assert_eq!(index.record_count(), 3);
        assert_eq!(index.dimension(), 8);
        assert_eq!(index.doc_id_at(0).unwrap(), "a");
        assert_eq!(index.doc_id_at(2).unwrap(), "c");
    }

    #[test]
    fn from_vectors_dimension_mismatch() {
        let doc_ids = vec!["a".into()];
        let vectors = vec![vec![1.0, 2.0, 3.0]]; // dim 3 != expected 4
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 4);
        assert!(result.is_err());
    }

    #[test]
    fn from_vectors_count_mismatch() {
        let doc_ids = vec!["a".into(), "b".into()];
        let vectors = vec![vec![1.0, 2.0]]; // 1 vector, 2 doc_ids
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
        assert!(result.is_err());
    }

    #[test]
    fn from_vectors_non_finite_rejected() {
        let doc_ids = vec!["a".into()];
        let vectors = vec![vec![1.0, f32::NAN]];
        let result = InMemoryVectorIndex::from_vectors(doc_ids, vectors, 2);
        assert!(result.is_err());
    }

    #[test]
    fn from_fsvi_matches_file_backed_search() {
        let path = temp_index_path("from_fsvi");
        cleanup(&path);

        let dim = 32;
        let docs = 64usize;
        let mut writer = crate::VectorIndex::create_with_revision(
            &path,
            "test-embedder",
            "rev-a",
            dim,
            Quantization::F16,
        )
        .unwrap();

        for i in 0..docs {
            let vector = make_normalized_vec(dim, i as f32 * 0.73);
            writer.write_record(&format!("doc-{i}"), &vector).unwrap();
        }
        writer.finish().unwrap();

        let file_index = crate::VectorIndex::open(&path).unwrap();
        let memory_index = InMemoryVectorIndex::from_fsvi(&path).unwrap();
        assert_eq!(memory_index.record_count(), docs);
        assert_eq!(memory_index.dimension(), dim);

        let query = make_normalized_vec(dim, 12.4);
        let file_hits = file_index.search_top_k(&query, 10, None).unwrap();
        let memory_hits = memory_index.search_top_k(&query, 10, None).unwrap();
        assert_eq!(file_hits.len(), memory_hits.len());

        for (file, memory) in file_hits.iter().zip(memory_hits.iter()) {
            assert_eq!(file.doc_id, memory.doc_id);
            assert!(
                (file.score - memory.score).abs() < 0.001,
                "score mismatch for {}: file={} memory={}",
                file.doc_id,
                file.score,
                memory.score
            );
        }

        // Verify vectors were loaded in quantized form and still round-trip.
        let recovered = memory_index.vector_at_f32(0).unwrap();
        assert_eq!(recovered.len(), dim);

        cleanup(&path);
    }

    #[test]
    fn search_top_k_correctness() {
        let dim = 16;
        let n = 50;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.7))
            .collect();
        let query = make_normalized_vec(dim, 0.7); // should match doc-1 best

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let hits = index.search_top_k(&query, 5, None).unwrap();

        assert_eq!(hits.len(), 5);
        // Scores should be descending
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "scores not descending");
        }
        // Top hit should be doc-1 (same seed as query)
        assert_eq!(hits[0].doc_id, "doc-1");
    }

    #[test]
    fn int8_two_pass_matches_exact_topk() {
        let dim = 32;
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
            .collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
            let query = make_normalized_vec(dim, qseed);
            let exact = index.search_top_k(&query, 10, None).unwrap();
            // mult=10 -> 100 candidates of 200; pass-1 recall is 1 here, so the
            // two-pass result must be bit-identical to the exact top-k.
            let two_pass = index.search_top_k_int8_two_pass(&query, 10, 10).unwrap();

            assert_eq!(two_pass.len(), exact.len(), "qseed={qseed}");
            for w in two_pass.windows(2) {
                assert!(w[0].score >= w[1].score, "two-pass not descending");
            }
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                tp_ids, exact_ids,
                "int8 two-pass should match exact top-k at mult=10 (qseed={qseed})"
            );
            for (a, b) in two_pass.iter().zip(exact.iter()) {
                assert!((a.score - b.score).abs() < 1e-6, "scores differ");
            }
        }
    }

    #[test]
    fn int8_two_pass_dimension_mismatch() {
        let dim = 8;
        let doc_ids: Vec<String> = (0..4).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..4).map(|i| make_normalized_vec(dim, i as f32)).collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let err = index
            .search_top_k_int8_two_pass(&[1.0; 7], 3, 4)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_top_k_with_filter() {
        let dim = 8;
        let doc_ids: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let query = make_normalized_vec(dim, 0.0); // matches doc-0

        struct OddFilter;
        impl SearchFilter for OddFilter {
            fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
                // Only allow odd-numbered docs
                doc_id
                    .strip_prefix("doc-")
                    .and_then(|n| n.parse::<usize>().ok())
                    .is_some_and(|n| n % 2 == 1)
            }
            fn name(&self) -> &str {
                "odd"
            }
        }

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let hits = index.search_top_k(&query, 5, Some(&OddFilter)).unwrap();
        assert_eq!(hits.len(), 5);
        for hit in &hits {
            let num: usize = hit.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
            assert!(num % 2 == 1, "filter should exclude even docs");
        }
    }

    #[test]
    fn search_with_bitset_filter_uses_precomputed_hash_path() {
        // BitsetFilter resolves via matches_doc_id_hash (the precomputed-hash
        // prescreen). Result must equal the allowed doc-id set's top-k — i.e. the
        // precomputed hashes match BitsetFilter's own hashing.
        use frankensearch_core::filter::BitsetFilter;
        let dim = 8;
        let doc_ids: Vec<String> = (0..20).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let allowed: Vec<String> = doc_ids.iter().step_by(3).cloned().collect(); // doc-0,3,6,...
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());

        let query = make_normalized_vec(dim, 6.0);
        let hits = index.search_top_k(&query, 20, Some(&filter)).unwrap();

        assert!(!hits.is_empty());
        for hit in &hits {
            assert!(
                allowed.contains(&hit.doc_id),
                "bitset filter must only return allowed doc-ids; got {}",
                hit.doc_id
            );
        }
        // Every allowed doc should be returned (limit 20 ≥ allowed count).
        assert_eq!(hits.len(), allowed.len());
    }

    #[test]
    fn int8_two_pass_filtered_matches_exact_filtered() {
        // The filtered int8 two-pass must return the same top-k as the exact
        // filtered scan (pass-1 pre-screens by the same doc_id hash; lossless when
        // pass-1 retains the true filtered top-k at a generous multiplier).
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let doc_ids: Vec<String> = (0..200).map(|i| format!("doc-{i:04}")).collect();
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [3.0_f32, 17.0, 88.0] {
            let query = make_normalized_vec(dim, qseed);
            let exact: Vec<String> = index
                .search_top_k(&query, 10, Some(&filter))
                .unwrap()
                .into_iter()
                .map(|h| h.doc_id)
                .collect();
            let two_pass: Vec<String> = index
                .search_top_k_int8_two_pass_filtered(&query, 10, 10, Some(&filter))
                .unwrap()
                .into_iter()
                .map(|h| h.doc_id)
                .collect();
            // Only allowed docs, and identical to the exact filtered top-k.
            for id in &two_pass {
                assert!(allowed.contains(id), "two-pass returned filtered-out {id}");
            }
            assert_eq!(
                two_pass, exact,
                "filtered two-pass != exact (qseed={qseed})"
            );
        }
    }

    #[test]
    fn search_empty_index() {
        let index = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 4).unwrap();
        let hits = index.search_top_k(&[0.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_dimension_mismatch() {
        let index = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![make_normalized_vec(4, 1.0)],
            4,
        )
        .unwrap();
        let result = index.search_top_k(&[1.0, 0.0], 10, None); // dim 2 != 4
        assert!(result.is_err());
    }

    #[test]
    fn f16_precision_tolerance() {
        let dim = 256;
        let v = make_normalized_vec(dim, 42.0);
        let index =
            InMemoryVectorIndex::from_vectors(vec!["test".into()], vec![v.clone()], dim).unwrap();

        // Self-similarity should be ~1.0 (within f16 precision)
        let hits = index.search_top_k(&v, 1, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() < 0.001,
            "f16 self-similarity should be within 0.001 of 1.0, got {}",
            hits[0].score
        );
    }

    #[test]
    fn vector_at_f32_roundtrip() {
        let dim = 8;
        let original = make_normalized_vec(dim, 5.0);
        let index =
            InMemoryVectorIndex::from_vectors(vec!["a".into()], vec![original.clone()], dim)
                .unwrap();
        let recovered = index.vector_at_f32(0).unwrap();
        assert_eq!(recovered.len(), dim);
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.002, "f16 round-trip error too large");
        }
    }

    #[test]
    fn two_tier_search_fast() {
        let dim = 8;
        let n = 20;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
        let query = make_normalized_vec(dim, 5.0);

        let fast = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let two_tier = InMemoryTwoTierIndex::new(fast, None);

        assert!(!two_tier.has_quality_index());
        assert_eq!(two_tier.doc_count(), 20);

        let hits = two_tier.search_fast(&query, 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].doc_id, "doc-5");
    }

    #[test]
    fn two_tier_quality_scores() {
        let dim_fast = 8;
        let dim_quality = 16;
        let n = 10;

        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let fast_vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim_fast, i as f32))
            .collect();
        let quality_vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim_quality, i as f32 * 0.5))
            .collect();

        let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vecs, dim_fast).unwrap();
        let quality =
            InMemoryVectorIndex::from_vectors(doc_ids, quality_vecs, dim_quality).unwrap();

        let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
        assert!(two_tier.has_quality_index());

        let fast_query = make_normalized_vec(dim_fast, 3.0);
        let hits = two_tier.search_fast(&fast_query, 5).unwrap();

        let quality_query = make_normalized_vec(dim_quality, 1.5);
        let scores = two_tier
            .quality_scores_for_hits(&quality_query, &hits)
            .unwrap();
        assert_eq!(scores.len(), 5);
        // All scores should be Some and finite
        for s in &scores {
            assert!(
                s.is_some_and(|v| v.is_finite()),
                "quality score should be Some and finite"
            );
        }
    }

    #[test]
    fn two_tier_no_quality_returns_nones() {
        let dim = 4;
        let fast = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![make_normalized_vec(dim, 1.0)],
            dim,
        )
        .unwrap();
        let two_tier = InMemoryTwoTierIndex::new(fast, None);

        let hits = two_tier
            .search_fast(&make_normalized_vec(dim, 1.0), 1)
            .unwrap();
        let scores = two_tier
            .quality_scores_for_hits(&make_normalized_vec(dim, 1.0), &hits)
            .unwrap();
        assert_eq!(scores, vec![None]);
    }

    #[test]
    fn parallel_search_matches_sequential() {
        let dim = 16;
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.3))
            .collect();
        let query = make_normalized_vec(dim, 7.0);

        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        let seq_params = SearchParams {
            parallel_enabled: false,
            parallel_threshold: 1,
            parallel_chunk_size: 32,
        };
        let par_params = SearchParams {
            parallel_enabled: true,
            parallel_threshold: 1, // force parallel even for small index
            parallel_chunk_size: 32,
        };

        let seq_hits = index
            .search_top_k_with_params(&query, 10, None, seq_params)
            .unwrap();
        let par_hits = index
            .search_top_k_with_params(&query, 10, None, par_params)
            .unwrap();

        assert_eq!(seq_hits.len(), par_hits.len());
        for (s, p) in seq_hits.iter().zip(par_hits.iter()) {
            assert_eq!(s.doc_id, p.doc_id);
            assert!(
                (s.score - p.score).abs() < 1e-6,
                "parallel vs sequential score mismatch"
            );
        }
    }
}
