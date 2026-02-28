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

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::prelude::*;

use crate::simd::dot_product_f16_f32;
use crate::search::{PARALLEL_CHUNK_SIZE, PARALLEL_THRESHOLD, SearchParams};
use crate::VectorIndex;

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
    /// Vector dimensionality.
    dimension: usize,
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
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vec.len(),
                });
            }
            // Validate finite values
            for &val in vec {
                if !val.is_finite() {
                    return Err(SearchError::InvalidConfig {
                        field: "vectors".to_owned(),
                        value: format!("vector[{i}] contains non-finite value"),
                        reason: "all vector elements must be finite".to_owned(),
                    });
                }
            }
            flat.extend(vec.iter().map(|&v| f16::from_f32(v)));
        }
        Ok(Self {
            doc_ids,
            vectors: flat,
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

        Ok(Self {
            doc_ids,
            vectors: flat,
            dimension,
        })
    }

    /// Number of vectors in the index.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
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

        let use_parallel =
            params.parallel_enabled && count >= params.parallel_threshold;
        let chunk_size = params.parallel_chunk_size.max(1);

        let heap = if use_parallel {
            self.scan_parallel(query, limit, filter, chunk_size)?
        } else {
            self.scan_sequential(query, limit, filter)?
        };

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

        for index in start..end {
            if let Some(f) = filter {
                if !f.matches(&self.doc_ids[index], None) {
                    continue;
                }
            }
            let stored = self.vector_slice(index);
            let score = dot_product_f16_f32(stored, query)?;
            if heap.len() < limit || score_key(score) >= cutoff {
                insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                if heap.len() >= limit {
                    if let Some(&worst) = heap.peek() {
                        cutoff = score_key(worst.score);
                    }
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
    pub fn scores_for_hits(
        &self,
        query: &[f32],
        hits: &[VectorHit],
    ) -> SearchResult<Vec<f32>> {
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
    pub fn new(
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
                return Err(SearchError::IndexNotFound {
                    path: fast_path,
                });
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
    ) -> SearchResult<Vec<f32>> {
        let Some(quality) = &self.quality_index else {
            return Ok(vec![0.0; hits.len()]);
        };
        quality.scores_for_hits(query_vec, hits)
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
    pub fn fast_index(&self) -> &InMemoryVectorIndex {
        &self.fast_index
    }

    /// Get a reference to the quality index (if present).
    #[must_use]
    pub fn quality_index(&self) -> Option<&InMemoryVectorIndex> {
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
    if let Some(&worst) = heap.peek() {
        if match score_key(candidate.score).total_cmp(&score_key(worst.score)) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => candidate.index < worst.index,
        } {
            let _ = heap.pop();
            heap.push(candidate);
        }
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
    use super::*;

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
    fn search_top_k_correctness() {
        let dim = 16;
        let n = 50;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32 * 0.7))
            .collect();
        let query = make_normalized_vec(dim, 0.7); // should match doc-1 best

        let index =
            InMemoryVectorIndex::from_vectors(doc_ids, vectors.clone(), dim).unwrap();
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
    fn search_empty_index() {
        let index =
            InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 4).unwrap();
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
        let index = InMemoryVectorIndex::from_vectors(
            vec!["test".into()],
            vec![v.clone()],
            dim,
        )
        .unwrap();

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
        let index = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![original.clone()],
            dim,
        )
        .unwrap();
        let recovered = index.vector_at_f32(0).unwrap();
        assert_eq!(recovered.len(), dim);
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.002,
                "f16 round-trip error too large"
            );
        }
    }

    #[test]
    fn two_tier_search_fast() {
        let dim = 8;
        let n = 20;
        let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| make_normalized_vec(dim, i as f32))
            .collect();
        let query = make_normalized_vec(dim, 5.0);

        let fast =
            InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();
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

        let fast = InMemoryVectorIndex::from_vectors(
            doc_ids.clone(),
            fast_vecs,
            dim_fast,
        )
        .unwrap();
        let quality = InMemoryVectorIndex::from_vectors(
            doc_ids,
            quality_vecs,
            dim_quality,
        )
        .unwrap();

        let two_tier = InMemoryTwoTierIndex::new(fast, Some(quality));
        assert!(two_tier.has_quality_index());

        let fast_query = make_normalized_vec(dim_fast, 3.0);
        let hits = two_tier.search_fast(&fast_query, 5).unwrap();

        let quality_query = make_normalized_vec(dim_quality, 1.5);
        let scores = two_tier.quality_scores_for_hits(&quality_query, &hits).unwrap();
        assert_eq!(scores.len(), 5);
        // All scores should be finite
        for &s in &scores {
            assert!(s.is_finite(), "quality score should be finite");
        }
    }

    #[test]
    fn two_tier_no_quality_returns_zeros() {
        let dim = 4;
        let fast = InMemoryVectorIndex::from_vectors(
            vec!["a".into()],
            vec![make_normalized_vec(dim, 1.0)],
            dim,
        )
        .unwrap();
        let two_tier = InMemoryTwoTierIndex::new(fast, None);

        let hits = two_tier.search_fast(&make_normalized_vec(dim, 1.0), 1).unwrap();
        let scores = two_tier
            .quality_scores_for_hits(&make_normalized_vec(dim, 1.0), &hits)
            .unwrap();
        assert_eq!(scores, vec![0.0]);
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

        let index =
            InMemoryVectorIndex::from_vectors(doc_ids, vectors, dim).unwrap();

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
