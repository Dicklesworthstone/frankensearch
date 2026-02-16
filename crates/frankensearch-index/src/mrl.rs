//! Matryoshka Representation Learning (MRL) adaptive dimensionality at search time.
//!
//! MRL-trained embedding models (including potion-128M and many modern sentence
//! transformers) produce vectors where the first N dimensions carry the most
//! information. [`MrlConfig`] enables a two-phase search that exploits this:
//!
//! 1. **Truncated scan**: compute dot products using only the first
//!    `search_dims` dimensions. This is 2-6x faster than a full-dimension scan
//!    for large indices.
//! 2. **Full-dimension rescore**: re-score the top `rescore_top_k` candidates
//!    using the full stored dimensionality for maximum accuracy.
//!
//! # Performance model
//!
//! - Standard search (384 dims): 384 multiply-accumulate per vector.
//! - MRL search (64 dims + rescore 30): 64*N + 384*30 = 64N + 11520 ops.
//! - Break-even at ~36 vectors. For 10K vectors: 640K vs 3.84M ops = **6x**
//!   speedup on the initial scan.
//!
//! # SIMD alignment
//!
//! For best SIMD throughput, `search_dims` should be a multiple of 8 (one
//! `f32x8` operation). Common choices: 64, 128, 192, 256. Non-aligned values
//! work correctly but with a scalar remainder tail.
//!
//! # Index format
//!
//! No FSVI changes are needed. Full-dimension vectors are stored as-is, and
//! truncation is a runtime operation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use serde::{Deserialize, Serialize};

use crate::wal::{from_wal_index, is_wal_index, to_wal_index};
use crate::{
    VectorIndex, dot_product_f16_bytes_f32, dot_product_f32_bytes_f32, dot_product_f32_f32,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for MRL-accelerated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrlConfig {
    /// Number of dimensions for the initial truncated scan.
    ///
    /// For best SIMD alignment, use a multiple of 8 (e.g., 64, 128).
    /// Must be at least 1 and at most the index dimension.
    /// Default: 64.
    pub search_dims: usize,

    /// Number of dimensions for re-scoring top candidates.
    ///
    /// Set to 0 to use the full index dimension (recommended).
    /// Default: 0 (full dimension).
    pub rescore_dims: usize,

    /// Number of top candidates to re-score with full dimensions.
    ///
    /// Set to 0 to use `3 * limit`.
    /// Default: 0 (auto = 3x limit).
    pub rescore_top_k: usize,
}

impl Default for MrlConfig {
    fn default() -> Self {
        Self {
            search_dims: 64,
            rescore_dims: 0,
            rescore_top_k: 0,
        }
    }
}

impl MrlConfig {
    /// Resolve `rescore_dims` to the effective value given the index dimension.
    ///
    /// Always returns at least `self.search_dims` so the rescore phase never
    /// uses fewer dimensions than the initial truncated scan.
    const fn effective_rescore_dims(&self, index_dim: usize) -> usize {
        let dims = if self.rescore_dims == 0 || self.rescore_dims > index_dim {
            index_dim
        } else {
            self.rescore_dims
        };
        // Rescore must use at least as many dims as the initial scan,
        // otherwise Phase 2 is strictly worse than Phase 1.
        if dims < self.search_dims {
            self.search_dims
        } else {
            dims
        }
    }

    /// Resolve `rescore_top_k` to the effective value given the search limit.
    const fn effective_rescore_top_k(&self, limit: usize) -> usize {
        if self.rescore_top_k == 0 {
            limit.saturating_mul(3)
        } else {
            self.rescore_top_k
        }
    }
}

// ---------------------------------------------------------------------------
// MRL search stats
// ---------------------------------------------------------------------------

/// Diagnostic statistics from an MRL search execution.
#[derive(Debug, Clone, Default)]
pub struct MrlSearchStats {
    /// Dimensions used for the initial truncated scan.
    pub scan_dims: usize,
    /// Dimensions used for re-scoring.
    pub rescore_dims: usize,
    /// Number of candidates passed to the rescore phase.
    pub candidates_rescored: usize,
    /// Total records scanned in the initial phase.
    pub records_scanned: usize,
    /// Whether the search fell back to standard full-dimension scan.
    pub fell_back_to_full: bool,
}

// ---------------------------------------------------------------------------
// Heap entry (reuse search.rs pattern)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct MrlHeapEntry {
    index: usize,
    score: f32,
}

impl PartialEq for MrlHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for MrlHeapEntry {}

impl PartialOrd for MrlHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MrlHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: worst score at the top for efficient pruning.
        match nan_safe(self.score).total_cmp(&nan_safe(other.score)) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.index.cmp(&other.index),
        }
    }
}

const fn nan_safe(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

fn insert_mrl_candidate(
    heap: &mut BinaryHeap<MrlHeapEntry>,
    candidate: MrlHeapEntry,
    limit: usize,
) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(candidate);
        return;
    }
    if let Some(&worst) = heap.peek() {
        let better = match nan_safe(candidate.score).total_cmp(&nan_safe(worst.score)) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => candidate.index < worst.index,
        };
        if better {
            let _ = heap.pop();
            heap.push(candidate);
        }
    }
}

// ---------------------------------------------------------------------------
// VectorIndex extension
// ---------------------------------------------------------------------------

impl VectorIndex {
    /// Search using MRL-accelerated truncated scan with full-dimension rescore.
    ///
    /// If `config.search_dims >= self.dimension()`, this falls back to the
    /// standard `search_top_k` (no truncation benefit).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` when `query.len()` does not
    /// match index dimensionality, `SearchError::InvalidConfig` for invalid
    /// config values, and `SearchError::IndexCorrupted` for malformed data.
    pub fn mrl_search(
        &self,
        query: &[f32],
        limit: usize,
        config: &MrlConfig,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        self.mrl_search_with_stats(query, limit, config, filter)
            .map(|(hits, _stats)| hits)
    }

    /// Like [`mrl_search`](Self::mrl_search) but also returns diagnostic stats.
    ///
    /// # Errors
    ///
    /// Same error conditions as [`mrl_search`](Self::mrl_search).
    pub fn mrl_search_with_stats(
        &self,
        query: &[f32],
        limit: usize,
        config: &MrlConfig,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<VectorHit>, MrlSearchStats)> {
        // Validate query dimension.
        if query.len() != self.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension(),
                found: query.len(),
            });
        }

        // Validate config.
        if config.search_dims == 0 {
            return Err(SearchError::InvalidConfig {
                field: "search_dims".into(),
                value: "0".into(),
                reason: "search_dims must be at least 1".into(),
            });
        }

        let dim = self.dimension();

        // Fall back to standard search if truncation wouldn't help.
        if config.search_dims >= dim {
            let hits = self.search_top_k(query, limit, filter)?;
            let stats = MrlSearchStats {
                scan_dims: dim,
                rescore_dims: dim,
                candidates_rescored: 0,
                records_scanned: self.record_count() + self.wal_entries.len(),
                fell_back_to_full: true,
            };
            return Ok((hits, stats));
        }

        if limit == 0 || (self.record_count() == 0 && self.wal_entries.is_empty()) {
            return Ok((Vec::new(), MrlSearchStats::default()));
        }

        let search_dims = config.search_dims;
        let rescore_dims = config.effective_rescore_dims(dim);
        let rescore_top_k = config.effective_rescore_top_k(limit);

        // Phase 1: truncated scan.
        let query_truncated = &query[..search_dims];
        let mut heap =
            self.mrl_truncated_scan(query_truncated, rescore_top_k, search_dims, filter)?;

        // Also scan WAL entries.
        self.mrl_scan_wal_truncated(
            query_truncated,
            &mut heap,
            rescore_top_k,
            search_dims,
            filter,
        )?;

        let candidates: Vec<MrlHeapEntry> = heap.into_vec();
        let records_scanned = self.record_count() + self.wal_entries.len();
        let candidates_rescored = candidates.len();

        // Phase 2: rescore candidates with full (or rescore_dims) dimensions.
        let query_rescore = &query[..rescore_dims];
        let mut rescored = Vec::with_capacity(candidates.len());

        for candidate in &candidates {
            let full_score = self.mrl_rescore(candidate.index, query_rescore, rescore_dims)?;
            rescored.push(MrlHeapEntry {
                index: candidate.index,
                score: full_score,
            });
        }

        // Select top `limit` from rescored candidates.
        rescored.sort_by(|a, b| {
            nan_safe(b.score)
                .total_cmp(&nan_safe(a.score))
                .then_with(|| a.index.cmp(&b.index))
        });
        rescored.truncate(limit);

        // Resolve doc_ids.
        let hits = self.resolve_mrl_hits(&rescored)?;

        let stats = MrlSearchStats {
            scan_dims: search_dims,
            rescore_dims,
            candidates_rescored,
            records_scanned,
            fell_back_to_full: false,
        };

        Ok((hits, stats))
    }

    // ── Internal: truncated scan ─────────────────────────────────────

    fn mrl_truncated_scan(
        &self,
        query_truncated: &[f32],
        limit: usize,
        search_dims: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<BinaryHeap<MrlHeapEntry>> {
        let mut heap = BinaryHeap::with_capacity(limit.saturating_add(1));
        let stride = match self.quantization() {
            crate::Quantization::F16 => self.dimension() * 2,
            crate::Quantization::F32 => self.dimension() * 4,
        };

        match self.quantization() {
            crate::Quantization::F16 => {
                let partial_bytes = search_dims * 2;
                let mut record_offset = self.records_offset;
                let mut vector_offset = self.vectors_offset;

                for index in 0..self.record_count() {
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let passed = if let Some(f) = filter {
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
                        if let Some(matches) = f.matches_doc_id_hash(hash, None) {
                            matches
                        } else {
                            let doc_id = self.doc_id_at(index)?;
                            f.matches(doc_id, None)
                        }
                    } else {
                        true
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + partial_bytes];
                        let score = dot_product_f16_bytes_f32(vector_bytes, query_truncated)?;
                        insert_mrl_candidate(&mut heap, MrlHeapEntry { index, score }, limit);
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
            crate::Quantization::F32 => {
                let partial_bytes = search_dims * 4;
                let mut record_offset = self.records_offset;
                let mut vector_offset = self.vectors_offset;

                for index in 0..self.record_count() {
                    let flags_bytes = &self.data[record_offset + 14..record_offset + 16];
                    let flags = u16::from_le_bytes([flags_bytes[0], flags_bytes[1]]);

                    if (flags & 0x0001) != 0 {
                        record_offset += 16;
                        vector_offset += stride;
                        continue;
                    }

                    let passed = if let Some(f) = filter {
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
                        if let Some(matches) = f.matches_doc_id_hash(hash, None) {
                            matches
                        } else {
                            let doc_id = self.doc_id_at(index)?;
                            f.matches(doc_id, None)
                        }
                    } else {
                        true
                    };

                    if passed {
                        let vector_bytes = &self.data[vector_offset..vector_offset + partial_bytes];
                        let score = dot_product_f32_bytes_f32(vector_bytes, query_truncated)?;
                        insert_mrl_candidate(&mut heap, MrlHeapEntry { index, score }, limit);
                    }

                    record_offset += 16;
                    vector_offset += stride;
                }
            }
        }

        Ok(heap)
    }

    fn mrl_scan_wal_truncated(
        &self,
        query_truncated: &[f32],
        heap: &mut BinaryHeap<MrlHeapEntry>,
        limit: usize,
        search_dims: usize,
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
            // WAL embeddings are f32 in memory — truncate to search_dims.
            let truncated_emb = &entry.embedding[..search_dims.min(entry.embedding.len())];
            let truncated_query = &query_truncated[..truncated_emb.len()];
            let score = dot_product_f32_f32(truncated_emb, truncated_query)?;
            insert_mrl_candidate(
                heap,
                MrlHeapEntry {
                    index: to_wal_index(idx),
                    score,
                },
                limit,
            );
        }
        Ok(())
    }

    // ── Internal: rescore ────────────────────────────────────────────

    fn mrl_rescore(
        &self,
        index: usize,
        query_rescore: &[f32],
        rescore_dims: usize,
    ) -> SearchResult<f32> {
        if is_wal_index(index) {
            let wal_idx = from_wal_index(index);
            let entry = &self.wal_entries[wal_idx];
            let emb_slice = &entry.embedding[..rescore_dims.min(entry.embedding.len())];
            let q_slice = &query_rescore[..emb_slice.len()];
            return dot_product_f32_f32(emb_slice, q_slice);
        }

        match self.quantization() {
            crate::Quantization::F16 => {
                let byte_count = rescore_dims.checked_mul(2).ok_or_else(|| {
                    crate::index_corrupted(&self.path, "f16 truncated byte length overflow")
                })?;
                let bytes = self.raw_vector_bytes_partial(index, byte_count)?;
                dot_product_f16_bytes_f32(bytes, query_rescore)
            }
            crate::Quantization::F32 => {
                let byte_count = rescore_dims.checked_mul(4).ok_or_else(|| {
                    crate::index_corrupted(&self.path, "f32 truncated byte length overflow")
                })?;
                let bytes = self.raw_vector_bytes_partial(index, byte_count)?;
                dot_product_f32_bytes_f32(bytes, query_rescore)
            }
        }
    }

    /// Read the first `byte_count` bytes of a stored vector (without reading
    /// the full stride).
    fn raw_vector_bytes_partial(&self, index: usize, byte_count: usize) -> SearchResult<&[u8]> {
        self.ensure_index(index)?;
        let start = self.vector_start(index)?;
        let end = start
            .checked_add(byte_count)
            .ok_or_else(|| crate::index_corrupted(&self.path, "partial vector end overflow"))?;
        if end > self.data.len() {
            return Err(crate::index_corrupted(
                &self.path,
                "partial vector extends past file end",
            ));
        }
        Ok(&self.data[start..end])
    }

    fn resolve_mrl_hits(&self, entries: &[MrlHeapEntry]) -> SearchResult<Vec<VectorHit>> {
        let mut hits = Vec::with_capacity(entries.len());
        for entry in entries {
            if is_wal_index(entry.index) {
                let wal_idx = from_wal_index(entry.index);
                let wal_entry = &self.wal_entries[wal_idx];
                let virtual_index = self.record_count().saturating_add(wal_idx);
                let index_u32 =
                    u32::try_from(virtual_index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".into(),
                        value: virtual_index.to_string(),
                        reason: "WAL entry index exceeds u32 range".into(),
                    })?;
                hits.push(VectorHit {
                    index: index_u32,
                    score: entry.score,
                    doc_id: wal_entry.doc_id.clone(),
                });
            } else {
                if self.is_deleted(entry.index) {
                    continue;
                }
                let index_u32 =
                    u32::try_from(entry.index).map_err(|_| SearchError::InvalidConfig {
                        field: "index".into(),
                        value: entry.index.to_string(),
                        reason: "index exceeds u32 range".into(),
                    })?;
                let doc_id = self.doc_id_at(entry.index)?.to_owned();
                hits.push(VectorHit {
                    index: index_u32,
                    score: entry.score,
                    doc_id,
                });
            }
        }
        Ok(hits)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_core::PredicateFilter;

    use super::*;
    use crate::{Quantization, VectorIndex};

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-index-mrl-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    fn write_index(path: &std::path::Path, rows: &[(&str, Vec<f32>)]) -> SearchResult<()> {
        let dimension =
            rows.first()
                .map(|(_, vec)| vec.len())
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "rows".into(),
                    value: "[]".into(),
                    reason: "rows must not be empty".into(),
                })?;
        let mut writer = VectorIndex::create_with_revision(
            path,
            "test",
            "mrl-test",
            dimension,
            Quantization::F16,
        )?;
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector)?;
        }
        writer.finish()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    /// Build a vector with a strong signal in the first `signal_dims` dimensions.
    fn signal_vector(dim: usize, signal_dims: usize, signal: f32) -> Vec<f32> {
        let mut v = vec![0.01; dim];
        for d in v.iter_mut().take(signal_dims) {
            *d = signal;
        }
        normalize(&v)
    }

    // ── Basic MRL search ─────────────────────────────────────────────

    #[test]
    fn mrl_search_returns_correct_top_1() {
        let dim = 16;
        let path = temp_index_path("basic-top1");

        // Use directionally distinct vectors (not normalized, so magnitudes
        // in the first 8 dims differ clearly and survive f16 quantization).
        let rows = [
            ("doc-a", vec![1.0; dim]), // first 8 dot = 8.0
            ("doc-b", vec![0.5; dim]), // first 8 dot = 4.0
            ("doc-c", vec![0.1; dim]), // first 8 dot = 0.8
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 0,
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 1, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(stats.scan_dims, 8);
        assert!(!stats.fell_back_to_full);
        assert!(stats.candidates_rescored > 0);

        std::fs::remove_file(&path).ok();
    }

    // ── Fallback to full search when search_dims >= dimension ────────

    #[test]
    fn mrl_search_falls_back_when_search_dims_equals_dimension() {
        let dim = 8;
        let path = temp_index_path("fallback-full");

        let rows = [
            ("doc-a", signal_vector(dim, 4, 1.0)),
            ("doc-b", signal_vector(dim, 4, 0.5)),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 4, 1.0);

        let config = MrlConfig {
            search_dims: 8, // equals dimension
            ..MrlConfig::default()
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert!(stats.fell_back_to_full);
        assert_eq!(stats.scan_dims, dim);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mrl_search_falls_back_when_search_dims_exceeds_dimension() {
        let dim = 8;
        let path = temp_index_path("fallback-exceed");

        let rows = [("doc-a", signal_vector(dim, 4, 1.0))];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 4, 1.0);

        let config = MrlConfig {
            search_dims: 100, // exceeds dimension
            ..MrlConfig::default()
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 1, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 1);
        assert!(stats.fell_back_to_full);

        std::fs::remove_file(&path).ok();
    }

    // ── search_dims = 0 → error ──────────────────────────────────────

    #[test]
    fn mrl_search_rejects_zero_search_dims() {
        let dim = 8;
        let path = temp_index_path("zero-dims");

        let rows = [("doc-a", signal_vector(dim, 4, 1.0))];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 4, 1.0);

        let config = MrlConfig {
            search_dims: 0,
            ..MrlConfig::default()
        };

        let err = index
            .mrl_search(&query, 1, &config, None)
            .expect_err("should reject search_dims=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));

        std::fs::remove_file(&path).ok();
    }

    // ── Empty index ──────────────────────────────────────────────────

    #[test]
    fn mrl_search_empty_index() {
        let dim = 8;
        let path = temp_index_path("empty-index");

        let writer =
            VectorIndex::create_with_revision(&path, "test", "mrl-test", dim, Quantization::F16)
                .expect("writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 4, 1.0);

        let config = MrlConfig {
            search_dims: 4,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 10, &config, None)
            .expect("mrl search");
        assert!(hits.is_empty());

        std::fs::remove_file(&path).ok();
    }

    // ── Single vector ────────────────────────────────────────────────

    #[test]
    fn mrl_search_single_vector() {
        let dim = 16;
        let path = temp_index_path("single-vector");

        let rows = [("sole-doc", signal_vector(dim, 8, 1.0))];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 8, 1.0);

        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 5, &config, None)
            .expect("mrl search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "sole-doc");

        std::fs::remove_file(&path).ok();
    }

    // ── Dimension mismatch ───────────────────────────────────────────

    #[test]
    fn mrl_search_dimension_mismatch() {
        let dim = 8;
        let path = temp_index_path("dim-mismatch");

        let rows = [("doc-a", signal_vector(dim, 4, 1.0))];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let bad_query = vec![1.0; 4]; // wrong dimension

        let config = MrlConfig {
            search_dims: 4,
            ..MrlConfig::default()
        };

        let err = index
            .mrl_search(&bad_query, 1, &config, None)
            .expect_err("should reject wrong dimension");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 4
            }
        ));

        std::fs::remove_file(&path).ok();
    }

    // ── MRL search matches standard search on same top-1 ────────────

    #[test]
    fn mrl_search_agrees_with_standard_on_top_1() {
        let dim = 16;
        let path = temp_index_path("agrees-standard");

        // Vectors with distinguishable signals in first 8 dims.
        let rows = [
            ("doc-best", signal_vector(dim, 8, 1.0)),
            ("doc-mid", signal_vector(dim, 8, 0.6)),
            ("doc-weak", signal_vector(dim, 8, 0.2)),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 8, 1.0);

        let standard = index
            .search_top_k(&query, 1, None)
            .expect("standard search");

        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 0,
        };
        let mrl = index
            .mrl_search(&query, 1, &config, None)
            .expect("mrl search");

        assert_eq!(standard[0].doc_id, mrl[0].doc_id);

        std::fs::remove_file(&path).ok();
    }

    // ── SIMD-aligned dims (multiple of 8) ────────────────────────────

    #[test]
    fn mrl_search_simd_aligned_dims() {
        let dim = 64;
        let path = temp_index_path("simd-aligned");

        let rows = [("doc-a", vec![1.0; dim]), ("doc-b", vec![0.5; dim])];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        // search_dims = 8 (perfect SIMD alignment)
        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 2, &config, None)
            .expect("mrl search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");

        std::fs::remove_file(&path).ok();
    }

    // ── Non-aligned search_dims (remainder handling) ─────────────────

    #[test]
    fn mrl_search_non_aligned_dims() {
        let dim = 16;
        let path = temp_index_path("non-aligned");

        let rows = [
            ("doc-a", signal_vector(dim, 5, 1.0)),
            ("doc-b", signal_vector(dim, 5, 0.5)),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 5, 1.0);

        // search_dims = 5 (not a multiple of 8)
        let config = MrlConfig {
            search_dims: 5,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 2, &config, None)
            .expect("mrl search");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");

        std::fs::remove_file(&path).ok();
    }

    // ── Filter integration ───────────────────────────────────────────

    #[test]
    fn mrl_search_with_filter() {
        let dim = 16;
        let path = temp_index_path("filter");

        let rows = [
            ("doc-a", vec![1.0; dim]),
            ("doc-b", vec![0.8; dim]),
            ("doc-c", vec![0.5; dim]),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let filter = PredicateFilter::new("no-a", |id| id != "doc-a");
        let hits = index
            .mrl_search(&query, 2, &config, Some(&filter))
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.doc_id != "doc-a"));
        assert_eq!(hits[0].doc_id, "doc-b");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mrl_bitset_filter_skips_doc_id_decode_for_non_matching_records() {
        let dim = 16;
        let path = temp_index_path("bitset-hash-fast-path");

        let rows = [("doc-a", vec![1.0; dim]), ("doc-b", vec![0.2; dim])];
        write_index(&path, &rows).expect("write index");

        let inspect = VectorIndex::open(&path).expect("open");
        let bad_idx = inspect
            .find_index_by_doc_hash(super::super::fnv1a_hash(b"doc-b"))
            .expect("doc-b index");
        let record = inspect.record_at(bad_idx).expect("record");
        let bad_offset =
            inspect.strings_offset + usize::try_from(record.doc_id_offset).expect("offset");
        drop(inspect);

        let mut bytes = std::fs::read(&path).expect("read bytes");
        bytes[bad_offset] = 0xFF;
        std::fs::write(&path, bytes).expect("write corrupt bytes");

        let index = VectorIndex::open(&path).expect("open");
        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };
        let filter = frankensearch_core::BitsetFilter::from_doc_ids(["doc-a"]);
        let hits = index
            .mrl_search(&vec![1.0; dim], 10, &config, Some(&filter))
            .expect("mrl search should ignore corrupted filtered-out doc_id");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");

        std::fs::remove_file(&path).ok();
    }

    // ── Tombstoned records excluded ──────────────────────────────────

    #[test]
    fn mrl_search_excludes_tombstoned() {
        let dim = 16;
        let path = temp_index_path("tombstone");

        let rows = [
            ("doc-a", signal_vector(dim, 8, 1.0)),
            ("doc-b", signal_vector(dim, 8, 0.8)),
        ];
        write_index(&path, &rows).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open");
        index.soft_delete("doc-a").expect("delete doc-a");

        let query = signal_vector(dim, 8, 1.0);
        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 10, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-b");

        std::fs::remove_file(&path).ok();
    }

    // ── WAL entries participate in MRL search ────────────────────────

    #[test]
    fn mrl_search_includes_wal_entries() {
        let dim = 16;
        let path = temp_index_path("wal");

        let rows = [("doc-main", vec![0.5; dim])];
        write_index(&path, &rows).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open");
        index.append("doc-wal", &vec![1.0; dim]).expect("append");

        let query = vec![1.0; dim];
        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-wal");

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(crate::wal::wal_path_for(&path)).ok();
    }

    // ── rescore_top_k = 0 → defaults to 3x limit ────────────────────

    #[test]
    fn mrl_rescore_top_k_defaults_to_3x() {
        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 0,
        };
        assert_eq!(config.effective_rescore_top_k(5), 15);
        assert_eq!(config.effective_rescore_top_k(0), 0);
        assert_eq!(config.effective_rescore_top_k(10), 30);
    }

    // ── rescore_dims = 0 → full dimension ────────────────────────────

    #[test]
    fn mrl_rescore_dims_defaults_to_full() {
        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 0,
        };
        assert_eq!(config.effective_rescore_dims(384), 384);
        assert_eq!(config.effective_rescore_dims(256), 256);
    }

    #[test]
    fn mrl_rescore_dims_clamped_to_index_dim() {
        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 1000,
            rescore_top_k: 0,
        };
        // rescore_dims > index_dim → use index_dim
        assert_eq!(config.effective_rescore_dims(384), 384);
    }

    // ── Config serde roundtrip ───────────────────────────────────────

    #[test]
    fn mrl_config_serde_roundtrip() {
        let config = MrlConfig {
            search_dims: 128,
            rescore_dims: 256,
            rescore_top_k: 50,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: MrlConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.search_dims, 128);
        assert_eq!(decoded.rescore_dims, 256);
        assert_eq!(decoded.rescore_top_k, 50);
    }

    // ── Limit zero returns empty ─────────────────────────────────────

    #[test]
    fn mrl_search_limit_zero() {
        let dim = 8;
        let path = temp_index_path("limit-zero");

        let rows = [("doc-a", signal_vector(dim, 4, 1.0))];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 4, 1.0);

        let config = MrlConfig {
            search_dims: 4,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 0, &config, None)
            .expect("mrl search");
        assert!(hits.is_empty());

        std::fs::remove_file(&path).ok();
    }

    // ── Verify truncated scan uses only search_dims ──────────────────

    #[test]
    fn truncated_scan_uses_only_search_dims() {
        // doc-a: strong signal in dims 0-3, noise in dims 4-15
        // doc-b: weak in dims 0-3, strong in dims 4-15
        // With search_dims=4, doc-a should rank higher in truncated scan.
        // With full rescore, doc-a should still win because first dims carry
        // most info in MRL-style embeddings.
        let dim = 16;
        let path = temp_index_path("truncated-only");

        let mut a = vec![0.01; dim];
        for d in a.iter_mut().take(4) {
            *d = 1.0;
        }
        let a = normalize(&a);

        let mut b = vec![0.01; dim];
        for d in b.iter_mut().skip(4).take(12) {
            *d = 1.0;
        }
        let b = normalize(&b);

        let rows = [("doc-a", a.clone()), ("doc-b", b)];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");

        let config = MrlConfig {
            search_dims: 4,
            rescore_dims: 0,
            rescore_top_k: 10,
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&a, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert_eq!(stats.scan_dims, 4);
        assert!(!stats.fell_back_to_full);

        std::fs::remove_file(&path).ok();
    }

    // ── Multiple results ordered by rescore ───────────────────────────

    #[test]
    fn mrl_results_ordered_by_rescore() {
        let dim = 16;
        let path = temp_index_path("rescore-order");

        let rows = [
            ("doc-a", signal_vector(dim, 8, 1.0)),
            ("doc-b", signal_vector(dim, 8, 0.7)),
            ("doc-c", signal_vector(dim, 8, 0.3)),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = signal_vector(dim, 8, 1.0);

        let config = MrlConfig {
            search_dims: 4,
            ..MrlConfig::default()
        };

        let hits = index
            .mrl_search(&query, 3, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 3);
        // Scores should be in descending order
        for pair in hits.windows(2) {
            assert!(
                pair[0].score >= pair[1].score,
                "results should be descending: {} >= {}",
                pair[0].score,
                pair[1].score
            );
        }

        std::fs::remove_file(&path).ok();
    }

    // ─── bd-2c7e tests begin ───

    #[test]
    fn mrl_config_default_values() {
        let config = MrlConfig::default();
        assert_eq!(config.search_dims, 64);
        assert_eq!(config.rescore_dims, 0);
        assert_eq!(config.rescore_top_k, 0);
    }

    #[test]
    fn mrl_config_debug_clone() {
        let config = MrlConfig {
            search_dims: 32,
            rescore_dims: 128,
            rescore_top_k: 20,
        };
        let cloned = config.clone();
        assert_eq!(cloned.search_dims, 32);
        assert_eq!(cloned.rescore_dims, 128);
        assert_eq!(cloned.rescore_top_k, 20);
        let dbg = format!("{config:?}");
        assert!(dbg.contains("MrlConfig"));
        assert!(dbg.contains("32"));
    }

    #[test]
    fn mrl_search_stats_default_values() {
        let stats = MrlSearchStats::default();
        assert_eq!(stats.scan_dims, 0);
        assert_eq!(stats.rescore_dims, 0);
        assert_eq!(stats.candidates_rescored, 0);
        assert_eq!(stats.records_scanned, 0);
        assert!(!stats.fell_back_to_full);
    }

    #[test]
    fn mrl_search_stats_debug_clone() {
        let stats = MrlSearchStats {
            scan_dims: 64,
            rescore_dims: 256,
            candidates_rescored: 30,
            records_scanned: 1000,
            fell_back_to_full: true,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.scan_dims, 64);
        assert_eq!(cloned.rescore_dims, 256);
        assert_eq!(cloned.candidates_rescored, 30);
        assert_eq!(cloned.records_scanned, 1000);
        assert!(cloned.fell_back_to_full);
        let dbg = format!("{stats:?}");
        assert!(dbg.contains("MrlSearchStats"));
    }

    #[test]
    fn nan_safe_replaces_nan_with_neg_infinity() {
        assert!((nan_safe(1.0) - 1.0).abs() < f32::EPSILON);
        assert!(nan_safe(0.0).abs() < f32::EPSILON);
        assert!((nan_safe(-1.0) + 1.0).abs() < f32::EPSILON);
        assert!(nan_safe(f32::NEG_INFINITY) == f32::NEG_INFINITY);
        assert!(nan_safe(f32::INFINITY) == f32::INFINITY);
        assert!(nan_safe(f32::NAN) == f32::NEG_INFINITY);
    }

    #[test]
    fn mrl_heap_entry_eq_same_values() {
        let a = MrlHeapEntry {
            index: 5,
            score: 0.9,
        };
        let b = MrlHeapEntry {
            index: 5,
            score: 0.9,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn mrl_heap_entry_ne_different_index() {
        let a = MrlHeapEntry {
            index: 5,
            score: 0.9,
        };
        let b = MrlHeapEntry {
            index: 6,
            score: 0.9,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn mrl_heap_entry_ne_different_score() {
        let a = MrlHeapEntry {
            index: 5,
            score: 0.9,
        };
        let b = MrlHeapEntry {
            index: 5,
            score: 0.8,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn mrl_heap_entry_ordering_min_heap() {
        // Min-heap: worse (lower) score should be "Greater" so it comes to top.
        let low = MrlHeapEntry {
            index: 0,
            score: 0.1,
        };
        let high = MrlHeapEntry {
            index: 1,
            score: 0.9,
        };
        // In std BinaryHeap (max-heap), the "greatest" element is popped first.
        // Our Ord reversal makes the lowest-score entry the "greatest" → it gets popped first.
        assert_eq!(low.cmp(&high), Ordering::Greater);
        assert_eq!(high.cmp(&low), Ordering::Less);
    }

    #[test]
    fn mrl_heap_entry_ordering_nan_treated_as_worst() {
        let nan_entry = MrlHeapEntry {
            index: 0,
            score: f32::NAN,
        };
        let normal = MrlHeapEntry {
            index: 1,
            score: 0.5,
        };
        // NaN → NEG_INFINITY → worst score → should be "Greater" (popped first from min-heap).
        assert_eq!(nan_entry.cmp(&normal), Ordering::Greater);
    }

    #[test]
    fn mrl_heap_entry_ordering_tie_breaks_on_index() {
        let a = MrlHeapEntry {
            index: 3,
            score: 0.5,
        };
        let b = MrlHeapEntry {
            index: 7,
            score: 0.5,
        };
        // Same score, lower index is "stable" (comes after in min-heap → Less).
        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn mrl_heap_entry_partial_ord_consistent() {
        let a = MrlHeapEntry {
            index: 0,
            score: 0.3,
        };
        let b = MrlHeapEntry {
            index: 1,
            score: 0.7,
        };
        assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
    }

    #[test]
    fn insert_mrl_candidate_limit_zero_noop() {
        let mut heap = BinaryHeap::new();
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 0,
                score: 1.0,
            },
            0,
        );
        assert!(heap.is_empty());
    }

    #[test]
    fn insert_mrl_candidate_fills_heap() {
        let mut heap = BinaryHeap::new();
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 0,
                score: 0.5,
            },
            3,
        );
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 1,
                score: 0.8,
            },
            3,
        );
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 2,
                score: 0.3,
            },
            3,
        );
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn insert_mrl_candidate_replaces_worst_when_better() {
        let mut heap = BinaryHeap::new();
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 0,
                score: 0.1,
            },
            2,
        );
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 1,
                score: 0.2,
            },
            2,
        );
        // Insert better candidate — should replace the worst (0.1).
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 2,
                score: 0.9,
            },
            2,
        );
        assert_eq!(heap.len(), 2);
        let entries: Vec<MrlHeapEntry> = heap.into_vec();
        assert!(entries.iter().all(|e| e.index != 0));
    }

    #[test]
    fn insert_mrl_candidate_keeps_worst_when_candidate_worse() {
        let mut heap = BinaryHeap::new();
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 0,
                score: 0.5,
            },
            1,
        );
        // Insert worse candidate — heap should not change.
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 1,
                score: 0.1,
            },
            1,
        );
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek().unwrap().index, 0);
    }

    #[test]
    fn insert_mrl_candidate_tie_prefers_lower_index() {
        let mut heap = BinaryHeap::new();
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 5,
                score: 0.5,
            },
            1,
        );
        // Same score, lower index → should replace.
        insert_mrl_candidate(
            &mut heap,
            MrlHeapEntry {
                index: 2,
                score: 0.5,
            },
            1,
        );
        assert_eq!(heap.peek().unwrap().index, 2);
    }

    #[test]
    fn effective_rescore_dims_uses_explicit_value() {
        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 128,
            rescore_top_k: 0,
        };
        assert_eq!(config.effective_rescore_dims(384), 128);
    }

    #[test]
    fn effective_rescore_top_k_uses_explicit_value() {
        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 42,
        };
        assert_eq!(config.effective_rescore_top_k(10), 42);
        assert_eq!(config.effective_rescore_top_k(1), 42);
    }

    #[test]
    fn mrl_search_f32_quantization() {
        let dim = 16;
        let path = temp_index_path("f32-quant");

        let dimension = dim;
        let mut writer = VectorIndex::create_with_revision(
            &path,
            "test",
            "mrl-test",
            dimension,
            Quantization::F32,
        )
        .expect("writer");
        let v_a = vec![1.0_f32; dim];
        let v_b = vec![0.5_f32; dim];
        writer.write_record("doc-a", &v_a).expect("write a");
        writer.write_record("doc-b", &v_b).expect("write b");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 0,
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, "doc-a");
        assert!(!stats.fell_back_to_full);
        assert_eq!(stats.scan_dims, 8);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mrl_search_wal_entries_with_filter() {
        let dim = 16;
        let path = temp_index_path("wal-filter");

        let rows = [("doc-main", vec![0.3; dim])];
        write_index(&path, &rows).expect("write index");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("doc-wal-keep", &vec![1.0; dim])
            .expect("append keep");
        index
            .append("doc-wal-skip", &vec![0.9; dim])
            .expect("append skip");

        let query = vec![1.0; dim];
        let config = MrlConfig {
            search_dims: 8,
            ..MrlConfig::default()
        };

        let filter = PredicateFilter::new("keep-only", |id| id != "doc-wal-skip");
        let hits = index
            .mrl_search(&query, 10, &config, Some(&filter))
            .expect("mrl search");

        assert!(hits.iter().all(|h| h.doc_id != "doc-wal-skip"));
        assert!(hits.iter().any(|h| h.doc_id == "doc-wal-keep"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(crate::wal::wal_path_for(&path)).ok();
    }

    #[test]
    fn mrl_search_explicit_rescore_top_k() {
        let dim = 16;
        let path = temp_index_path("explicit-rescore-k");

        let rows = [
            ("doc-a", vec![1.0; dim]),
            ("doc-b", vec![0.8; dim]),
            ("doc-c", vec![0.5; dim]),
            ("doc-d", vec![0.3; dim]),
        ];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        let config = MrlConfig {
            search_dims: 8,
            rescore_dims: 0,
            rescore_top_k: 2, // Only rescore top 2 candidates.
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert!(stats.candidates_rescored <= 2);
        assert!(!stats.fell_back_to_full);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mrl_search_explicit_rescore_dims() {
        let dim = 16;
        let path = temp_index_path("explicit-rescore-dims");

        let rows = [("doc-a", vec![1.0; dim]), ("doc-b", vec![0.5; dim])];
        write_index(&path, &rows).expect("write index");

        let index = VectorIndex::open(&path).expect("open");
        let query = vec![1.0; dim];

        let config = MrlConfig {
            search_dims: 4,
            rescore_dims: 12, // Rescore with 12 dims (not full 16).
            rescore_top_k: 0,
        };

        let (hits, stats) = index
            .mrl_search_with_stats(&query, 2, &config, None)
            .expect("mrl search");

        assert_eq!(hits.len(), 2);
        assert_eq!(stats.rescore_dims, 12);
        assert_eq!(stats.scan_dims, 4);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mrl_heap_entry_debug() {
        let entry = MrlHeapEntry {
            index: 42,
            score: 0.75,
        };
        let dbg = format!("{entry:?}");
        assert!(dbg.contains("MrlHeapEntry"));
        assert!(dbg.contains("42"));
    }

    #[test]
    fn mrl_heap_entry_copy() {
        let a = MrlHeapEntry {
            index: 1,
            score: 0.5,
        };
        let b = a; // Copy
        assert_eq!(a.index, b.index);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
    }

    // ─── bd-2c7e tests end ───
}
