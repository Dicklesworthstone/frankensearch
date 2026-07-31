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
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use std::sync::OnceLock;

use ahash::AHashMap;
use frankensearch_core::config::{ZeroSignalReason, ZeroSignalState};
use frankensearch_core::filter::{BuildIdentityHasherU64, SearchFilter, fnv1a_hash};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::prelude::*;

use crate::VectorIndex;
use crate::search::{ClassifiedHits, PARALLEL_CHUNK_SIZE, SearchParams};
use crate::simd::{dot_4bit_prepared, dot_i8_i8, dot_product_f16_f32, prepare_4bit_query};

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
    /// Lazily-built packed signed-4-bit quantization (2 dims/byte, `dim.div_ceil(2)`
    /// bytes/vector — half the int8 slab) for the optional 4-bit two-pass scan
    /// (`search_top_k_4bit_two_pass`). Built on first 4-bit-two-pass use.
    vectors_nibbles: OnceLock<Vec<u8>>,
    /// Lazily-built FNV-1a hashes of `doc_ids` (same `frankensearch_core::filter`
    /// hash that `BitsetFilter` uses). Lets the filtered scan call
    /// `SearchFilter::matches_doc_id_hash` with a precomputed hash instead of
    /// re-hashing each `doc_id` string per vector (the FSVI `search.rs` scan already
    /// does this). Built on first *filtered* search, so unfiltered callers pay
    /// neither the hashing nor the `8·N`-byte footprint.
    doc_id_hashes: OnceLock<Vec<u64>>,
    /// Lazily-built `doc_id → position` map for O(1) lookup, replacing the O(N)
    /// linear `doc_ids.iter().position(...)` scan in the per-hit quality-rerank
    /// path (`quality_scores_for_hits`), which was O(hits·N). Built on first
    /// doc-id lookup, so search-only callers pay nothing.
    doc_id_index: OnceLock<AHashMap<String, usize>>,
    /// Lazily-built `doc_id_hash → position` map (identity-hashed, same FNV-1a key
    /// space as `BitsetFilter`). Lets a *selective* filtered search gather the
    /// allow-set's positions directly — `O(|allow-set|)` exact dots instead of one
    /// filter probe per corpus document (`scan_gather` vs `scan_range`). Stored as
    /// `Option`: `None` means two `doc_ids` collide to the same hash, so the map is
    /// not a bijection and the gather fast-path is disabled (the per-document scan
    /// stays correct). Built on first selective-filter search; other callers pay
    /// neither the build nor its footprint.
    hash_to_pos: OnceLock<Option<HashMap<u64, usize, BuildIdentityHasherU64>>>,
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
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::quantize_f16_slab_to_i8(vectors_f16)
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

/// Quantize one component to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's
/// complement in the low 4 bits) given a scale.
#[allow(clippy::cast_possible_truncation)] // round()+clamp() bounds the cast
fn nibble_of(value: f32, scale: f32) -> u8 {
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    q.cast_unsigned() & 0x0F
}

/// Pack an f32 query into signed 4-bit nibbles, 2 dims/byte (low = even dim, high =
/// odd dim), using the query's own max-abs scale (a per-query constant that does not
/// change the dot ranking). Matches `pack_4bit_slab`'s layout for `dot_packed_4bit`.
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

/// Pack a contiguous f16 vector slab (`count·dim`) into signed 4-bit nibbles
/// (`dim.div_ceil(2)` bytes/vector) with one corpus-wide max-abs scale (a constant
/// factor, so the dot ranking is preserved).
fn pack_4bit_slab(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    // Runtime-dispatched (AVX2+F16C when available); see `simd` for the kernel.
    crate::simd::pack_f16_slab_to_4bit(vectors_f16, dim)
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
            crate::simd::encode_f32_to_f16_extend(&vec, &mut flat);
        }
        Ok(Self {
            doc_ids,
            vectors: flat,
            vectors_i8: OnceLock::new(),
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
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
            vectors_nibbles: OnceLock::new(),
            doc_id_hashes: OnceLock::new(),
            doc_id_index: OnceLock::new(),
            hash_to_pos: OnceLock::new(),
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

    /// Compute the zero-signal census for this in-memory index.
    ///
    /// The in-memory index has no tombstones and no WAL, so `live_count`
    /// equals `record_count` and both WAL-derived states are structurally
    /// unreachable. O(n·dim); intended for lazy classification of empty
    /// results, not the hot path.
    #[must_use]
    pub fn zero_signal_state(&self) -> ZeroSignalState {
        let record_count = self.record_count();
        let mut usable_vector_count = 0usize;
        for row in 0..record_count {
            let start = row * self.dimension;
            let row_slice = &self.vectors[start..start + self.dimension];
            let mut norm_sq = 0.0_f32;
            let mut finite = true;
            for &value in row_slice {
                let value = value.to_f32();
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                norm_sq += value * value;
            }
            if finite && norm_sq > 0.0 && norm_sq.is_finite() {
                usable_vector_count += 1;
            }
        }
        ZeroSignalState {
            record_count,
            live_count: record_count,
            tombstone_count: 0,
            wal_count: 0,
            usable_vector_count,
        }
    }

    /// Brute-force top-k with typed zero-signal classification.
    ///
    /// Mirrors [`VectorIndex::search_top_k_classified`] exactly (bd-tqhc
    /// requires the in-memory and file-backed paths to classify equivalent
    /// states identically): non-finite queries are rejected fail-closed and
    /// an empty result always carries a typed [`ZeroSignalReason`].
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
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
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
        let zero_signal = hits.is_empty().then(|| {
            self.zero_signal_state()
                .empty_result_reason(filter.is_some())
        });
        Ok(ClassifiedHits { hits, zero_signal })
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

        // Selective hash-addressable filter → gather the allow-set and exact-scan
        // only those positions (work ∝ |allow-set|, not corpus N). Bit-identical to
        // the per-document scan below.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
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
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
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
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut: when the candidate budget already covers every
        // vector (`limit·mult ≥ count`, e.g. `limit_all` or a large limit on a small
        // corpus), pass-1 prunes NOTHING — the int8 scan, the size-N candidate heap,
        // the query quantize, and the slab build are all pure overhead, and pass-2
        // would rescore all N regardless. Delegate to the exact f16 single pass.
        // **Bit-identical by construction:** the two-pass equals `search_top_k`
        // whenever pass-1 retains the true top-k (its own doc-contract), and here it
        // retains *every* candidate. `limit.min(count)` avoids a `usize::MAX`-sized
        // heap and returns the same `min(limit, count)` hits.
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set. The
        // gather scans only `|allow-set|` vectors (vs all N for the int8 pass-1) and
        // is *exact* f16, so it is strictly more accurate than this approximate
        // two-pass while doing far less work. Falls through when the filter is not a
        // selective allow-list.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        let query_i8 = quantize_i8_query(query);
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
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * self.dimension;
                    let stored = &vectors_i8[offset..offset + self.dimension];
                    let score = dot_i8_i8(stored, &query_i8) as f32;
                    // Skip the insert_candidate call for scores that cannot enter the
                    // full bounded heap — the same cutoff fast-path scan_range uses,
                    // which the int8 pass-1 previously lacked. Result is unchanged
                    // (a sub-cutoff score never makes the bounded heap anyway).
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
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

    /// 4-bit (16-level) two-pass exact top-k — the in-memory twin of the FSVI
    /// `search_top_k_4bit_two_pass`. A parallel pass-1 over a packed signed-4-bit
    /// slab (`dim/2` bytes/vector — half the int8 slab) via the fused
    /// `dot_packed_4bit` kernel keeps the top `k·mult`, then an exact f16 rescore
    /// selects the final top-k. 16 levels are lossless at mult≈5 on realistic
    /// clustered data; the result matches the exact top-k whenever pass-1 retains it.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass(
        &self,
        query: &[f32],
        limit: usize,
        candidate_multiplier: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.search_top_k_4bit_two_pass_filtered(query, limit, candidate_multiplier, None)
    }

    /// 4-bit two-pass with an optional [`SearchFilter`]. Pass-1 pre-screens each
    /// vector by its precomputed `doc_id` hash (the same path the int8 two-pass and
    /// exact scan use), so **filtered** searches keep the 4-bit speedup. Result
    /// matches the exact filtered top-k whenever pass-1 retains the true filtered top-k.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len() != dimension`.
    pub fn search_top_k_4bit_two_pass_filtered(
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
        let candidate_count = limit.saturating_mul(candidate_multiplier.max(1)).min(count);
        // Full-recall short-cut (see the int8 two-pass for the full rationale): when
        // `limit·mult ≥ count`, pass-1 prunes nothing, so the 4-bit scan, size-N
        // heap, query nibble-prep, and slab build are pure overhead — delegate to the
        // exact f16 pass. Bit-identical by construction (pass-1 retains every
        // candidate ⇒ two-pass == `search_top_k`).
        if candidate_count >= count {
            return self.search_top_k(query, limit.min(count), filter);
        }
        // Selective hash-addressable filter → exact gather of the allow-set (see the
        // int8 two-pass for the rationale): scans only `|allow-set|` vectors and is
        // exact f16, so it is strictly more accurate than this approximate 4-bit
        // two-pass while doing far less work.
        if let Some(hits) = self.try_gather_filtered(query, limit, filter, count)? {
            return Ok(hits);
        }
        // Decode the (loop-invariant) query nibbles once, not per stored vector.
        let query_prepared = prepare_4bit_query(&pack_4bit_query(query));
        let bytes_per_vector = self.dimension.div_ceil(2);
        let nibbles = self
            .vectors_nibbles
            .get_or_init(|| pack_4bit_slab(&self.vectors, self.dimension));

        // Pass 1: parallel bounded-heap 4-bit scan (same chunking + cutoff fast-path
        // + filter pre-screen as the int8 two-pass).
        let chunk_size = PARALLEL_CHUNK_SIZE;
        let chunk_count = count.div_ceil(chunk_size);
        let doc_id_hashes = filter.map(|_| self.doc_id_hashes());
        let partials: Vec<BinaryHeap<HeapEntry>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(count);
                let mut heap = BinaryHeap::with_capacity(candidate_count.min(end - start) + 1);
                let mut cutoff = f32::NEG_INFINITY;
                for index in start..end {
                    if let Some(f) = filter {
                        let passed = doc_id_hashes
                            .and_then(|h| f.matches_doc_id_hash(h[index], None))
                            .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
                        if !passed {
                            continue;
                        }
                    }
                    let offset = index * bytes_per_vector;
                    let stored = &nibbles[offset..offset + bytes_per_vector];
                    let score = dot_4bit_prepared(stored, &query_prepared) as f32;
                    if heap.len() < candidate_count || score_key(score) >= cutoff {
                        insert_candidate(&mut heap, HeapEntry::new(index, score), candidate_count);
                        if heap.len() >= candidate_count
                            && let Some(&worst) = heap.peek()
                        {
                            cutoff = score_key(worst.score);
                        }
                    }
                }
                heap
            })
            .collect();
        let candidate_heap = merge_partial_heaps(partials, candidate_count);

        // Pass 2: exact f16 rescore of the candidates (same selection + tie-break).
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

    /// Lazily-built `doc_id_hash → position` map for the selective-filter gather
    /// fast-path. Returns `None` when two `doc_ids` hash to the same value (the map
    /// would not be a bijection, so a gather could miss a colliding position the
    /// per-document scan would visit) — callers then fall back to the full scan,
    /// preserving exact results. Built once on first selective-filter search.
    fn hash_to_pos(&self) -> Option<&HashMap<u64, usize, BuildIdentityHasherU64>> {
        self.hash_to_pos
            .get_or_init(|| {
                let hashes = self.doc_id_hashes();
                let mut map =
                    HashMap::with_capacity_and_hasher(hashes.len(), BuildIdentityHasherU64);
                for (pos, &h) in hashes.iter().enumerate() {
                    if map.insert(h, pos).is_some() {
                        // Hash collision (or duplicate doc_id): the map can hold
                        // only one position per hash, so disable the fast path.
                        return None;
                    }
                }
                Some(map)
            })
            .as_ref()
    }

    /// Exact f16 top-k over an explicit list of candidate positions (the
    /// selective-filter gather fast-path). Parallel above `PARALLEL_CHUNK_SIZE`
    /// positions, serial below (tiny allow-sets don't amortize rayon overhead). The
    /// parallel path scans disjoint position chunks into per-chunk bounded heaps and
    /// merges by the `(score, index)` total order — order-independent, exactly like
    /// [`Self::scan_parallel`] — so it is bit-identical to the serial gather, to the
    /// per-document scan, and across thread counts.
    fn scan_gather(
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

    /// Bounded-heap exact f16 scan over one slice of candidate positions. Mirrors
    /// [`Self::scan_range`]'s cutoff fast-path, minus the per-document filter probe
    /// (membership was already decided when the positions were gathered).
    fn gather_range(
        &self,
        positions: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.min(positions.len()).saturating_add(1));
        let mut cutoff = f32::NEG_INFINITY;
        for &index in positions {
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

    /// Try the selective-filter gather fast-path: when `filter` is a
    /// hash-addressable allow-list whose size is below
    /// `count / GATHER_SELECTIVITY_DIVISOR`, gather the allowed positions and exact
    /// f16-scan only those. Returns `Some(hits)` when the fast-path applied,
    /// `None` to fall through to the per-document scan. Bit-identical: the gathered
    /// passing set equals `{ pos : doc_id_hash[pos] ∈ allow-set }`, the same set the
    /// scan keeps, and both rank by the `(score, index)` total order.
    fn try_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        count: usize,
    ) -> SearchResult<Option<Vec<VectorHit>>> {
        let Some(f) = filter else {
            return Ok(None);
        };
        let Some(allowed) = f.candidate_hashes() else {
            return Ok(None);
        };
        // Selectivity gate: gather pays a map lookup + sort per allowed hash, so it
        // only wins when the allow-set is a small fraction of the corpus. Above the
        // crossover the per-document scan (which skips the gather's setup) is faster.
        if allowed.len().saturating_mul(GATHER_SELECTIVITY_DIVISOR) >= count {
            return Ok(None);
        }
        let Some(map) = self.hash_to_pos() else {
            return Ok(None);
        };
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        // Ascending position order → sequential slab access (cache-friendly); not
        // required for correctness (the heap's total order is position-independent).
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        Ok(Some(self.resolve_heap(heap)?))
    }

    /// Bench-only: forced per-document filtered scan (the gather-fast-path baseline).
    /// Bypasses [`Self::try_gather_filtered`] so the A/B measures the old behavior.
    #[doc(hidden)]
    pub fn bench_scan_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        let count = self.record_count();
        if limit == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let params = SearchParams::default();
        let use_parallel = params.parallel_enabled && count >= params.parallel_threshold;
        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, params.parallel_chunk_size.max(1))?
        } else {
            self.scan_sequential(query, limit, filter)?
        };
        self.resolve_heap(heap)
    }

    /// Bench-only: forced gather over a hash-addressable allow-set, ignoring the
    /// selectivity gate (so the crossover can be measured directly).
    #[doc(hidden)]
    pub fn bench_gather_filtered(
        &self,
        query: &[f32],
        limit: usize,
        filter: &dyn SearchFilter,
    ) -> SearchResult<Vec<VectorHit>> {
        let allowed = filter
            .candidate_hashes()
            .expect("bench_gather_filtered requires a hash-addressable allow-set");
        let map = self
            .hash_to_pos()
            .expect("bench_gather_filtered requires a bijective hash→pos map");
        let mut positions: Vec<usize> =
            allowed.iter().filter_map(|h| map.get(h).copied()).collect();
        positions.sort_unstable();
        let heap = self.scan_gather(&positions, query, limit)?;
        self.resolve_heap(heap)
    }

    /// O(1) `doc_id → position` lookup via a lazily-built map, replacing the O(N)
    /// `doc_ids.iter().position(...)` linear scan. First-insert-wins, matching
    /// `position`'s first-match semantics for any (non-canonical) duplicate ids.
    /// Built once on first lookup (the per-hit quality-rerank path); search-only
    /// callers never pay the `O(N)` build or its footprint. The map hashes with
    /// `ahash` (not SipHash): every quality query pays one hash of each hit's
    /// `doc_id` here, and `ahash` cuts the rescore loop ~1.27–1.48× at 128–300
    /// candidates (see `quality_rescore_hasher_ab`). Semantics are unchanged.
    fn index_of_doc_id(&self, doc_id: &str) -> Option<usize> {
        self.doc_id_index
            .get_or_init(|| {
                let mut map = AHashMap::with_capacity(self.doc_ids.len());
                for (i, id) in self.doc_ids.iter().enumerate() {
                    map.entry(id.clone()).or_insert(i);
                }
                map
            })
            .get(doc_id)
            .copied()
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
                let passed = doc_id_hashes
                    .and_then(|h| f.matches_doc_id_hash(h[index], None))
                    .unwrap_or_else(|| f.matches(&self.doc_ids[index], None));
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
        // In-memory limit_all (`limit >= count`) builds a count-sized heap, so
        // `winners` can hold every record. Above a threshold the final sort
        // dominates and a parallel sort pays (~2.81× at 50k, `winners_sort` bench);
        // below it the rayon overhead is not worth it. Bit-identical — the same
        // gated lever as `search.rs:183` (`compare_best_first` is a strict total
        // order, so the parallel sort yields the same unique order).
        if winners.len() >= PAR_SORT_THRESHOLD {
            winners.par_sort_unstable_by(compare_best_first);
        } else {
            winners.sort_unstable_by(compare_best_first);
        }
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
                doc_id: self.doc_ids[winner.index].as_str().into(),
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
        // SIMD-widen 8 f16 per block (`widen8_f16_slice`, the same magic-factor widen
        // the f16 dot kernels use — bit-identical to the scalar `f16::to_f32`), then a
        // scalar tail for the last < 8. Mirrors the FSVI `VectorIndex::vector_at_f32`.
        let mut out = Vec::with_capacity(stored.len());
        let (blocks, remainder) = stored.as_chunks::<8>();
        for arr in blocks {
            out.extend_from_slice(&crate::simd::widen8_f16_slice(arr).to_array());
        }
        for v in remainder {
            out.push(v.to_f32());
        }
        Ok(out)
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
                .index_of_doc_id(&hit.doc_id)
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

    /// Search the fast tier with typed zero-signal classification.
    ///
    /// Mirrors [`crate::TwoTierIndex::search_fast_classified`] so in-memory
    /// and persistent two-tier paths classify equivalent states identically
    /// (bd-tqhc). The in-memory variant performs no transition logging: its
    /// indexes are ephemeral, and the once-per-generation logging bound is
    /// owned by the persistent path.
    ///
    /// # Errors
    ///
    /// Everything [`Self::search_fast`] returns, plus
    /// [`SearchError::InvalidConfig`] for non-finite query vectors.
    pub fn search_fast_classified(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<ClassifiedHits> {
        self.fast_index.search_top_k_classified(query_vec, k, None)
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
                .index_of_doc_id(&hit.doc_id)
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

/// Selectivity threshold for the gather fast-path: take it only when the filter's
/// allow-set is smaller than `corpus / GATHER_SELECTIVITY_DIVISOR`. Below this the
/// gather (exact f16 dots over the allow-set, parallel above `PARALLEL_CHUNK_SIZE`)
/// beats the parallel per-document scan; above it the scan wins because the gather's
/// serial setup (allow-set collect + position sort) grows with the allow-set. The
/// `filtered_gather` selectivity-sweep bench (N=50k, dim 384) measured, with the
/// parallel gather: 14× at 0.5%, 8.6× at 1%, 2.1× at 5%, 1.3× at 10%, then a loss by
/// 25% (crossover ~13%). Gate at **N/10 (10%)** — inside the winning region
/// (≥1.3× at the boundary) with margin for a machine whose core count shifts the
/// crossover. (Tiny allow-sets stay serial and still hit 6.9–50× — see the ledger.)
const GATHER_SELECTIVITY_DIVISOR: usize = 10;

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

/// Winners-count threshold above which the `limit_all` final sort uses a parallel
/// `par_sort_unstable_by` (mirrors `search::PAR_SORT_THRESHOLD`). Below it, rayon's
/// spawn/merge overhead is not amortized for the cheap `compare_best_first`.
const PAR_SORT_THRESHOLD: usize = 16_384;

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
    fn four_bit_two_pass_keep_all_matches_exact() {
        // With a multiplier large enough to retain every record, the exact f16
        // rescore must reproduce `search_top_k` bit-for-bit — verifying the nibble
        // pack/unpack, offsets, rescore, and resolve.
        let dim = 34; // odd-ish, > 32, exercises a partial last packed byte
        let n = 200;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.31))
            .collect();
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        for qseed in [0.31_f32, 3.0, 17.5, 99.9] {
            let query = make_normalized_vec(dim, qseed);
            let exact = index.search_top_k(&query, 10, None).unwrap();
            // mult=20 → candidate_count clamps to n → pass-1 retains all → identical.
            let two_pass = index.search_top_k_4bit_two_pass(&query, 10, 20).unwrap();
            let exact_ids: Vec<&str> = exact.iter().map(|h| h.doc_id.as_str()).collect();
            let tp_ids: Vec<&str> = two_pass.iter().map(|h| h.doc_id.as_str()).collect();
            assert_eq!(
                tp_ids, exact_ids,
                "4bit two-pass (keep-all) should match exact top-k (qseed={qseed})"
            );
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
                allowed.iter().any(|a| a.as_str() == hit.doc_id.as_str()),
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
                .map(|h| h.doc_id.to_string())
                .collect();
            let two_pass: Vec<String> = index
                .search_top_k_int8_two_pass_filtered(&query, 10, 10, Some(&filter))
                .unwrap()
                .into_iter()
                .map(|h| h.doc_id.to_string())
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
    fn selective_filter_gather_matches_scan() {
        // A selective hash-addressable filter takes the gather fast-path through
        // `search_top_k`/the two-pass filtered fns; it must be bit-identical to the
        // forced per-document filtered scan (same passing set, `(score,index)` order).
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let doc_ids: Vec<String> = (0..500).map(|i| format!("doc-{i:04}")).collect();
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        // ~5% allow-set (well under the selectivity gate) → gather path is taken.
        let allowed: Vec<String> = doc_ids.iter().step_by(20).cloned().collect();
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

        let ids = |hits: Vec<VectorHit>| -> Vec<String> {
            hits.into_iter().map(|h| h.doc_id.to_string()).collect()
        };
        for qseed in [1.0_f32, 42.0, 313.0] {
            let query = make_normalized_vec(dim, qseed);
            let scan = ids(index
                .bench_scan_filtered(&query, 10, Some(&filter))
                .unwrap());
            let gather = ids(index.bench_gather_filtered(&query, 10, &filter).unwrap());
            let public = ids(index.search_top_k(&query, 10, Some(&filter)).unwrap());
            let int8 = ids(index
                .search_top_k_int8_two_pass_filtered(&query, 10, 3, Some(&filter))
                .unwrap());
            let fourbit = ids(index
                .search_top_k_4bit_two_pass_filtered(&query, 10, 3, Some(&filter))
                .unwrap());
            for id in &gather {
                assert!(allowed.contains(id), "gather returned filtered-out {id}");
            }
            assert_eq!(gather, scan, "gather != scan (qseed={qseed})");
            assert_eq!(public, scan, "search_top_k gather != scan (qseed={qseed})");
            assert_eq!(
                int8, scan,
                "int8 two-pass gather != exact scan (qseed={qseed})"
            );
            assert_eq!(
                fourbit, scan,
                "4bit two-pass gather != exact scan (qseed={qseed})"
            );
        }
    }

    #[test]
    fn parallel_gather_matches_scan() {
        // Allow-set larger than PARALLEL_CHUNK_SIZE → the gather runs its parallel
        // per-chunk-heap + merge path, which must stay bit-identical to the scan.
        use frankensearch_core::filter::BitsetFilter;
        let dim = 16;
        let n = 3000; // > PARALLEL_CHUNK_SIZE allow-set below forces the parallel path
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:05}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vec(dim, i as f32)).collect();
        // ~half the corpus (1500 > 1024 chunk size) → parallel gather.
        let allowed: Vec<String> = doc_ids.iter().step_by(2).cloned().collect();
        assert!(
            allowed.len() > PARALLEL_CHUNK_SIZE,
            "must exceed chunk size"
        );
        let filter = BitsetFilter::from_doc_ids(allowed.iter().cloned());
        let index = InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
        let ids = |hits: Vec<VectorHit>| -> Vec<String> {
            hits.into_iter().map(|h| h.doc_id.to_string()).collect()
        };
        for qseed in [2.0_f32, 99.0, 1234.0] {
            let query = make_normalized_vec(dim, qseed);
            let scan = ids(index
                .bench_scan_filtered(&query, 25, Some(&filter))
                .unwrap());
            let gather = ids(index.bench_gather_filtered(&query, 25, &filter).unwrap());
            assert_eq!(gather, scan, "parallel gather != scan (qseed={qseed})");
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
