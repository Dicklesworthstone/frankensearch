//! Brute-force top-k vector search over an opened [`crate::VectorIndex`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use half::f16;
use rayon::prelude::*;

use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{Quantization, VectorIndex, dot_product_f16_f32, dot_product_f32_f32};

/// Record-count threshold where search switches from sequential to Rayon.
pub const PARALLEL_THRESHOLD: usize = 10_000;
/// Chunk size per Rayon task in the parallel scan path.
pub const PARALLEL_CHUNK_SIZE: usize = 1_024;

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
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        match self.quantization() {
            Quantization::F16 => {
                let mut scratch = vec![f16::from_f32(0.0); self.dimension()];
                for index in 0..self.record_count() {
                    if !self.passes_search_filter(filter, index)? {
                        continue;
                    }
                    let score = self.score_f16(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
            Quantization::F32 => {
                let mut scratch = vec![0.0_f32; self.dimension()];
                for index in 0..self.record_count() {
                    if !self.passes_search_filter(filter, index)? {
                        continue;
                    }
                    let score = self.score_f32(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
        }
        Ok(heap)
    }

    fn scan_parallel(
        &self,
        query: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        chunk_size: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let partial_heaps: SearchResult<Vec<BinaryHeap<HeapEntry>>> = if filter.is_some() {
            // Pre-filter indices on the main thread (doc_id resolution is cheap),
            // then scatter the surviving indices across Rayon tasks.
            let filtered_indices = self.collect_filtered_indices(filter)?;
            if filtered_indices.is_empty() {
                return Ok(BinaryHeap::new());
            }
            filtered_indices
                .par_chunks(chunk_size)
                .map(|chunk| self.scan_indices_chunk(chunk, query, limit))
                .collect()
        } else {
            let chunk_count = self.record_count().div_ceil(chunk_size);
            (0..chunk_count)
                .into_par_iter()
                .map(|chunk_index| {
                    let start = chunk_index * chunk_size;
                    let end = (start + chunk_size).min(self.record_count());
                    self.scan_range_chunk(start, end, query, limit)
                })
                .collect()
        };

        Ok(merge_partial_heaps(partial_heaps?, limit))
    }

    fn collect_filtered_indices(
        &self,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<usize>> {
        let mut indices = Vec::new();
        for index in 0..self.record_count() {
            if self.passes_search_filter(filter, index)? {
                indices.push(index);
            }
        }
        Ok(indices)
    }

    fn scan_range_chunk(
        &self,
        start: usize,
        end: usize,
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        match self.quantization() {
            Quantization::F16 => {
                let mut scratch = vec![f16::from_f32(0.0); self.dimension()];
                for index in start..end {
                    if self.is_deleted(index) {
                        continue;
                    }
                    let score = self.score_f16(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
            Quantization::F32 => {
                let mut scratch = vec![0.0_f32; self.dimension()];
                for index in start..end {
                    if self.is_deleted(index) {
                        continue;
                    }
                    let score = self.score_f32(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
        }
        Ok(heap)
    }

    fn scan_indices_chunk(
        &self,
        indices: &[usize],
        query: &[f32],
        limit: usize,
    ) -> SearchResult<BinaryHeap<HeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        match self.quantization() {
            Quantization::F16 => {
                let mut scratch = vec![f16::from_f32(0.0); self.dimension()];
                for &index in indices {
                    if self.is_deleted(index) {
                        continue;
                    }
                    let score = self.score_f16(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
                }
            }
            Quantization::F32 => {
                let mut scratch = vec![0.0_f32; self.dimension()];
                for &index in indices {
                    if self.is_deleted(index) {
                        continue;
                    }
                    let score = self.score_f32(index, query, &mut scratch)?;
                    insert_candidate(&mut heap, HeapEntry::new(index, score), limit);
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
                // WAL entry — resolve doc_id from in-memory WAL state.
                let wal_idx = from_wal_index(winner.index);
                let entry = &self.wal_entries[wal_idx];
                let virtual_index = self.record_count().saturating_add(wal_idx);
                let index_u32 =
                    u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".to_owned(),
                        value: virtual_index.to_string(),
                        reason: "WAL entry index exceeds u32 range".to_owned(),
                    })?;
                hits.push(VectorHit {
                    index: index_u32,
                    score: winner.score,
                    doc_id: entry.doc_id.clone(),
                });
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

        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });
        Ok(hits)
    }

    fn score_f16(&self, index: usize, query: &[f32], scratch: &mut [f16]) -> SearchResult<f32> {
        self.decode_f16_into(index, scratch)?;
        dot_product_f16_f32(scratch, query)
    }

    fn score_f32(&self, index: usize, query: &[f32], scratch: &mut [f32]) -> SearchResult<f32> {
        self.decode_f32_into(index, scratch)?;
        dot_product_f32_f32(scratch, query)
    }

    fn decode_f16_into(&self, index: usize, output: &mut [f16]) -> SearchResult<()> {
        if output.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: output.len(),
            });
        }
        let expected_len = self
            .dimension()
            .checked_mul(2)
            .ok_or_else(|| super::index_corrupted(&self.path, "f16 vector byte length overflow"))?;
        let bytes = self.raw_vector_bytes(index)?;
        if bytes.len() != expected_len {
            return Err(super::index_corrupted(
                &self.path,
                format!(
                    "f16 vector byte length mismatch: expected {expected_len}, found {}",
                    bytes.len()
                ),
            ));
        }
        for (slot, chunk) in output.iter_mut().zip(bytes.chunks_exact(2)) {
            *slot = f16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Ok(())
    }

    fn decode_f32_into(&self, index: usize, output: &mut [f32]) -> SearchResult<()> {
        if output.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: output.len(),
            });
        }
        let expected_len = self
            .dimension()
            .checked_mul(4)
            .ok_or_else(|| super::index_corrupted(&self.path, "f32 vector byte length overflow"))?;
        let bytes = self.raw_vector_bytes(index)?;
        if bytes.len() != expected_len {
            return Err(super::index_corrupted(
                &self.path,
                format!(
                    "f32 vector byte length mismatch: expected {expected_len}, found {}",
                    bytes.len()
                ),
            ));
        }
        for (slot, chunk) in output.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    /// Check whether a main-index record passes a `SearchFilter`.
    ///
    /// Uses `SearchFilter::matches_doc_id_hash` when available to avoid
    /// decoding `doc_id` strings in the hot scan loop.
    /// Falls back to `filter.matches(doc_id, ...)` when hash-only matching
    /// is not possible.
    fn passes_search_filter(
        &self,
        filter: Option<&dyn SearchFilter>,
        index: usize,
    ) -> SearchResult<bool> {
        let entry = self.record_at(index)?;
        if super::is_tombstoned_flags(entry.flags) {
            return Ok(false);
        }
        let Some(f) = filter else {
            return Ok(true);
        };
        if let Some(matches) = f.matches_doc_id_hash(entry.doc_id_hash, None) {
            return Ok(matches);
        }
        let doc_id = self.doc_id_at(index)?;
        Ok(f.matches(doc_id, None))
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
    let value = std::env::var("FRANKENSEARCH_PARALLEL_SEARCH").ok();
    parse_parallel_search_env(value.as_deref())
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
    fn compare_best_first_orders_descending_by_score() {
        let high = HeapEntry::new(0, 0.9);
        let low = HeapEntry::new(1, 0.1);
        assert_eq!(compare_best_first(&high, &low), Ordering::Less);
        assert_eq!(compare_best_first(&low, &high), Ordering::Greater);
    }

    #[test]
    fn compare_best_first_nan_ranks_after_finite() {
        let finite = HeapEntry::new(0, 0.5);
        let nan = HeapEntry::new(1, f32::NAN);
        // NaN maps to NEG_INFINITY, so finite ranks before NaN
        assert_eq!(compare_best_first(&finite, &nan), Ordering::Less);
        assert_eq!(compare_best_first(&nan, &finite), Ordering::Greater);
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
}
