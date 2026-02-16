//! Brute-force top-k vector search over an opened [`crate::VectorIndex`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::OnceLock;

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use rayon::prelude::*;

use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{
    Quantization, VectorIndex, dot_product_f16_bytes_f32, dot_product_f32_bytes_f32,
    dot_product_f32_f32,
};

/// Record-count threshold where search switches from sequential to Rayon.
pub const PARALLEL_THRESHOLD: usize = 10_000;
/// Chunk size per Rayon task in the parallel scan path.
pub const PARALLEL_CHUNK_SIZE: usize = 1_024;

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

        let mut heap = if has_main {
            if use_parallel {
                self.scan_parallel(query, limit, filter, chunk_size)?
            } else {
                self.scan_sequential(query, limit, filter)?
            }
        } else {
            BinaryHeap::with_capacity(limit.saturating_add(1))
        };

        // Merge WAL entries into the same heap.
        if has_wal {
            self.scan_wal(query, &mut heap, limit, filter)?;
        }

        self.resolve_hits(heap)
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

    fn scan_range_chunk(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        let dim = self.dimension();

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
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
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
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
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
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        let dim = self.dimension();

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
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
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
                        insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
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
            // Guard: corrupt WAL embeddings (e.g. from crash recovery) can
            // produce NaN/Inf scores. Skip them rather than polluting results.
            if !score.is_finite() {
                continue;
            }
            insert_candidate(heap, HeapEntry::new(to_wal_index(idx), score), limit);
        }
        Ok(())
    }

    fn resolve_hits(&self, heap: BinaryHeap<HeapEntry>) -> SearchResult<Vec<VectorHit>> {
        if heap.is_empty() {
            return Ok(Vec::new());
        }

        let mut winners = heap.into_vec();
        winners.sort_by(compare_best_first);

        let mut hits = Vec::with_capacity(winners.len());
        for winner in winners {
            if is_wal_index(winner.index) {
                hits.push(self.resolve_wal_hit(&winner)?);
            } else {
                // Main index entry.
                if self.is_deleted(winner.index) {
                    continue;
                }
                let index_u32 =
                    u32::try_from(winner.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: winner.index.to_string(),
                        reason: "winner index exceeds u32 range for VectorHit".to_owned(),
                    })?;
                let doc_id = self.doc_id_at(winner.index)?.to_owned();
                hits.push(VectorHit {
                    index: index_u32,
                    score: winner.score,
                    doc_id,
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
                path: PathBuf::new(),
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
            doc_id: entry.doc_id.clone(),
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
    let mut merged = BinaryHeap::with_capacity(limit.saturating_add(1));
    for heap in partial_heaps {
        for entry in heap {
            insert_candidate(&mut merged, entry, limit);
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
            write_index(&path, &row_refs).expect("write index");

            let index = VectorIndex::open(&path).expect("open index");
            let hits = index.search_top_k(&query, limit, None).expect("search");

            let expected_len = limit.min(vectors.len());
            prop_assert_eq!(hits.len(), expected_len);
            let mut seen_indices = HashSet::new();
            for hit in &hits {
                prop_assert!(seen_indices.insert(hit.index));
            }
            for pair in hits.windows(2) {
                let left = &pair[0];
                let right = &pair[1];
                let ordered = match left.score.total_cmp(&right.score) {
                    Ordering::Greater => true,
                    Ordering::Equal => left.index <= right.index,
                    Ordering::Less => false,
                };
                prop_assert!(ordered, "hits must be score-descending with index tie-breaks");
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
            write_index(&path, &row_refs).expect("write index");

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
            let rank = f32::from(u16::try_from(i).expect("test index must fit in u16"));
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
            let rank = f32::from(u16::try_from(i).expect("test index must fit in u16"));
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
            let rank = f32::from(u16::try_from(32 - i).expect("fits u16"));
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
            let rank = f32::from(u16::try_from(64 - i).expect("fits u16"));
            rows.push((format!("doc-{i:03}"), vec![rank, 0.0, 0.0, 0.0]));
        }
        let refs: Vec<(&str, Vec<f32>)> = rows
            .iter()
            .map(|(doc_id, vec)| (doc_id.as_str(), vec.clone()))
            .collect();
        write_index(&path, &refs).expect("write index");

        let index = VectorIndex::open(&path).expect("open index");
        let query = [1.0, 0.0, 0.0, 0.0];

        // Force parallel with low threshold
        let parallel_params = SearchParams {
            parallel_threshold: 1,
            parallel_chunk_size: 8,
            parallel_enabled: true,
        };
        // Force sequential
        let sequential_params = SearchParams {
            parallel_threshold: usize::MAX,
            parallel_chunk_size: PARALLEL_CHUNK_SIZE,
            parallel_enabled: true,
        };

        let parallel = index
            .search_top_k_with_params(&query, 10, None, parallel_params)
            .expect("parallel search");
        let sequential = index
            .search_top_k_with_params(&query, 10, None, sequential_params)
            .expect("sequential search");

        assert_eq!(parallel.len(), sequential.len());
        for (left, right) in parallel.iter().zip(sequential.iter()) {
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
