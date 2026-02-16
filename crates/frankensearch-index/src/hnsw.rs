//! Optional HNSW approximate nearest-neighbor index (`ann` feature).
//!
//! This module wraps `hnsw_rs` behind a frankensearch-native API.
//!
//! # Persistence
//!
//! Persistence uses one metadata file plus `hnsw_rs` sidecars:
//! 1. `vector.fast.hnsw` stores metadata (`doc_ids`, `HnswConfig`, dimension).
//! 2. `vector.fast.hnsw.graph` stores graph topology.
//! 3. `vector.fast.hnsw.data` stores vector payload for reloading.

use std::path::Path;
use std::time::Instant;

use frankensearch_core::{SearchError, SearchResult, VectorHit};
use hnsw_rs::prelude::{AnnT, DistDot, Hnsw, HnswIo};
use serde::{Deserialize, Serialize};

use crate::VectorIndex;

/// Default HNSW `M` (max connections per node).
pub const HNSW_DEFAULT_M: usize = 16;
/// Default HNSW `ef_construction` (build-time beam width).
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Default HNSW `ef_search` (query-time beam width).
pub const HNSW_DEFAULT_EF_SEARCH: usize = 100;
/// Default HNSW max layer depth.
pub const HNSW_DEFAULT_MAX_LAYER: usize = 16;
const DIST_DOT_SHRINK: f32 = 0.999_999;

/// ANN construction/runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// HNSW `M` (max connections per node).
    pub m: usize,
    /// HNSW `ef_construction` (build-time beam width).
    pub ef_construction: usize,
    /// Default HNSW `ef_search` (query-time beam width).
    pub ef_search: usize,
    /// Maximum HNSW layer depth.
    pub max_layer: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: HNSW_DEFAULT_M,
            ef_construction: HNSW_DEFAULT_EF_CONSTRUCTION,
            ef_search: HNSW_DEFAULT_EF_SEARCH,
            max_layer: HNSW_DEFAULT_MAX_LAYER,
        }
    }
}

/// On-disk metadata for the HNSW index.
#[derive(Debug, Serialize, Deserialize)]
struct HnswMeta {
    doc_ids: Vec<String>,
    config: HnswConfig,
    dimension: usize,
}

/// Diagnostics for one ANN query.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnSearchStats {
    /// Number of vectors indexed.
    pub index_size: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Effective ef used for this query.
    pub ef_search: usize,
    /// Requested `k`.
    pub k_requested: usize,
    /// Returned result count.
    pub k_returned: usize,
    /// Query latency in microseconds.
    pub search_time_us: u64,
    /// Whether this path is approximate ANN.
    pub is_approximate: bool,
    /// Estimated recall@k from ef/k ratio.
    pub estimated_recall: f64,
}

/// HNSW ANN index over vectors aligned to `VectorIndex` row order.
pub struct HnswIndex {
    hnsw: Hnsw<'static, f32, DistDot>,
    doc_ids: Vec<String>,
    dimension: usize,
    config: HnswConfig,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("points", &self.hnsw.get_nb_point())
            .field("doc_ids", &self.doc_ids.len())
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HnswIndex {
    /// Build a new HNSW index from an opened `VectorIndex`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `SearchError::InvalidConfig` for invalid HNSW params
    /// - `SearchError::IndexCorrupted` if `vectors/doc_ids` cannot be decoded
    pub fn build_from_vector_index(index: &VectorIndex, config: HnswConfig) -> SearchResult<Self> {
        let dimension = index.dimension();
        let mut doc_ids = Vec::with_capacity(index.record_count());
        let mut vectors = Vec::with_capacity(index.record_count());
        for i in 0..index.record_count() {
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            vectors.push(index.vector_at_f32(i)?);
        }
        Self::build_from_parts(doc_ids, vectors, dimension, config)
    }

    /// Load an ANN index from disk.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` if metadata or graph files are missing/malformed.
    pub fn load(path: &Path) -> SearchResult<Self> {
        // 1. Load metadata (JSON)
        let metadata_bytes = std::fs::read(path).map_err(SearchError::Io)?;
        let meta: HnswMeta = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| ann_corrupted(path, format!("failed to parse HNSW metadata: {e}")))?;

        // 2. Load graph + vectors from hnsw_rs sidecar files.
        let (directory, basename) = dump_location(path)?;
        let mut reloader = HnswIo::new(&directory, &basename);
        let loaded = reloader.load_hnsw::<f32, DistDot>().map_err(|e| {
            ann_corrupted(path, format!("failed to load HNSW graph sidecar pair: {e}"))
        })?;

        if loaded.get_nb_point() != meta.doc_ids.len() {
            return Err(ann_corrupted(
                path,
                format!(
                    "graph size ({}) mismatch with doc_ids count ({})",
                    loaded.get_nb_point(),
                    meta.doc_ids.len()
                ),
            ));
        }

        // Convert loaded points back to row-ordered vectors (origin id order),
        // then rebuild a `'static` HNSW so we are not tied to the reloader lifetime.
        let mut vectors_by_origin: Vec<Option<Vec<f32>>> = vec![None; meta.doc_ids.len()];
        for point in loaded.get_point_indexation().get_layer_iterator(0) {
            let origin_id = point.get_origin_id();
            if origin_id >= vectors_by_origin.len() {
                return Err(ann_corrupted(
                    path,
                    format!(
                        "loaded origin id {origin_id} exceeds doc_id count {}",
                        vectors_by_origin.len()
                    ),
                ));
            }
            vectors_by_origin[origin_id] = Some(point.get_v().to_vec());
        }
        let mut vectors = Vec::with_capacity(vectors_by_origin.len());
        for (origin_id, maybe_vector) in vectors_by_origin.into_iter().enumerate() {
            let vector = maybe_vector.ok_or_else(|| {
                ann_corrupted(
                    path,
                    format!("missing loaded vector for origin id {origin_id}"),
                )
            })?;
            vectors.push(vector);
        }

        Self::build_from_parts(meta.doc_ids, vectors, meta.dimension, meta.config)
    }

    /// Persist ANN index to disk.
    ///
    /// Writes metadata to `path` and graph/vector payloads via `hnsw_rs`
    /// sidecar files in the same directory.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` on write failure.
    pub fn save(&self, path: &Path) -> SearchResult<()> {
        let (directory, basename) = dump_location(path)?;
        std::fs::create_dir_all(&directory)?;

        // 1. Save graph + vectors through hnsw_rs.
        let dumped_basename = self
            .hnsw
            .file_dump(&directory, &basename)
            .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))?;
        if dumped_basename != basename {
            return Err(SearchError::Io(std::io::Error::other(format!(
                "unexpected dump basename '{dumped_basename}', expected '{basename}'"
            ))));
        }

        // 2. Save metadata
        let meta = HnswMeta {
            doc_ids: self.doc_ids.clone(),
            config: self.config,
            dimension: self.dimension,
        };
        let metadata_bytes = serde_json::to_vec(&meta)
            .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))?;
        std::fs::write(path, metadata_bytes).map_err(SearchError::Io)?;

        Ok(())
    }

    /// Run ANN query and return hits plus query diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }

        if k == 0 || self.doc_ids.is_empty() {
            let stats = AnnSearchStats {
                index_size: self.len(),
                dimension: self.dimension,
                ef_search,
                k_requested: k,
                k_returned: 0,
                search_time_us: 0,
                is_approximate: true,
                estimated_recall: 1.0,
            };
            return Ok((Vec::new(), stats));
        }

        let effective_k = k.min(self.doc_ids.len());
        let effective_ef = ef_search.max(effective_k).max(1);
        let normalized_query = normalize_for_dist_dot(query.to_vec());

        let start = Instant::now();
        let neighbors = self
            .hnsw
            .search(&normalized_query, effective_k, effective_ef);
        let elapsed = start.elapsed();
        let search_time_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);

        let mut hits = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let doc_id =
                self.doc_ids
                    .get(neighbor.d_id)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "neighbor_id".to_owned(),
                        value: neighbor.d_id.to_string(),
                        reason: "neighbor id exceeds doc_id table".to_owned(),
                    })?;
            let index = u32::try_from(neighbor.d_id).map_err(|_| SearchError::InvalidConfig {
                field: "neighbor_id".to_owned(),
                value: neighbor.d_id.to_string(),
                reason: "neighbor id exceeds u32 range for VectorHit".to_owned(),
            })?;
            hits.push(VectorHit {
                index,
                score: 1.0 - neighbor.distance,
                doc_id: doc_id.clone(),
            });
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });

        let stats = AnnSearchStats {
            index_size: self.len(),
            dimension: self.dimension,
            ef_search: effective_ef,
            k_requested: k,
            k_returned: hits.len(),
            search_time_us,
            is_approximate: true,
            estimated_recall: estimate_recall(effective_ef, effective_k),
        };
        Ok((hits, stats))
    }

    /// Run ANN query and return only the hits.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats(query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Returns true when this ANN index matches row order and shape of a `VectorIndex`.
    ///
    /// # Errors
    ///
    /// Propagates decoding errors from `VectorIndex::doc_id_at`.
    pub fn matches_vector_index(&self, index: &VectorIndex) -> SearchResult<bool> {
        if self.dimension != index.dimension() || self.len() != index.record_count() {
            return Ok(false);
        }
        for i in 0..self.doc_ids.len() {
            if self.doc_ids[i] != index.doc_id_at(i)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Number of indexed vectors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether ANN index has zero vectors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// ANN configuration used to build this index.
    #[must_use]
    pub const fn config(&self) -> HnswConfig {
        self.config
    }

    fn build_from_parts(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        config: HnswConfig,
    ) -> SearchResult<Self> {
        validate_config(config)?;
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: vectors.len().to_string(),
                reason: format!("doc_id count {} must match vector count", doc_ids.len()),
            });
        }
        let mut normalized_vectors = Vec::with_capacity(vectors.len());
        for (idx, vector) in vectors.into_iter().enumerate() {
            if vector.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "vector".to_owned(),
                    value: idx.to_string(),
                    reason: "all vector values must be finite".to_owned(),
                });
            }
            normalized_vectors.push(normalize_for_dist_dot(vector));
        }

        let hnsw = Hnsw::new(
            config.m,
            doc_ids.len().max(1),
            config.max_layer,
            config.ef_construction,
            DistDot,
        );
        if !normalized_vectors.is_empty() {
            let vectors_with_ids: Vec<(&Vec<f32>, usize)> = normalized_vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| (vector, index))
                .collect();
            hnsw.parallel_insert(&vectors_with_ids);
        }

        Ok(Self {
            hnsw,
            doc_ids,
            // vectors field removed to save RAM
            dimension,
            config,
        })
    }
}

fn validate_config(config: HnswConfig) -> SearchResult<()> {
    if config.m == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_m must be greater than zero".to_owned(),
        });
    }
    if config.m > 256 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: config.m.to_string(),
            reason: "hnsw_m must be <= 256".to_owned(),
        });
    }
    if config.ef_construction == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_construction".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_construction must be greater than zero".to_owned(),
        });
    }
    if config.ef_search == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_search".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_search must be greater than zero".to_owned(),
        });
    }
    if config.max_layer == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_max_layer must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn ann_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

fn dump_location(path: &Path) -> SearchResult<(std::path::PathBuf, String)> {
    let directory = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let basename = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| ann_corrupted(path, "metadata path has no valid UTF-8 basename"))?
        .to_owned();
    Ok((directory, basename))
}

fn normalize_for_dist_dot(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv_norm = DIST_DOT_SHRINK / norm;
        for value in &mut vector {
            *value *= inv_norm;
        }
    }
    vector
}

fn estimate_recall(ef_search: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let numerator = f64::from(u32::try_from(ef_search.max(1)).unwrap_or(u32::MAX));
    let denominator = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let ratio = numerator / denominator;
    0.1_f64.mul_add(ratio.log2(), 0.9_f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::Quantization;

    fn temp_path(label: &str, extension: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-hnsw-{label}-{}-{now}.{extension}",
            std::process::id()
        ))
    }

    fn lcg_next(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        u32::try_from((*state >> 32) & u64::from(u32::MAX)).unwrap_or(u32::MAX)
    }

    fn normalized_vector(seed: usize, dimension: usize) -> Vec<f32> {
        let mut state = u64::try_from(seed).unwrap_or(0).wrapping_add(1);
        let mut out = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let random = lcg_next(&mut state);
            let upper = u16::try_from((random >> 16) & u32::from(u16::MAX)).unwrap_or(u16::MAX);
            let raw = f32::from(upper) / f32::from(u16::MAX);
            out.push((raw * 2.0_f32) - 1.0_f32);
        }
        let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut out {
                *value /= norm;
            }
        }
        out
    }

    fn write_index(path: &Path, vectors: &[Vec<f32>]) -> SearchResult<VectorIndex> {
        let dimension = vectors.first().map_or(8, Vec::len);
        let mut writer =
            VectorIndex::create_with_revision(path, "hash", "test", dimension, Quantization::F32)?;
        for (idx, vector) in vectors.iter().enumerate() {
            writer.write_record(&format!("doc-{idx:04}"), vector)?;
        }
        writer.finish()?;
        VectorIndex::open(path)
    }

    fn recall_at_k(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
        if exact.is_empty() {
            return 1.0;
        }
        let exact_ids: HashSet<&str> = exact.iter().map(|hit| hit.doc_id.as_str()).collect();
        let overlap = approx
            .iter()
            .filter(|hit| exact_ids.contains(hit.doc_id.as_str()))
            .count();
        f64::from(u32::try_from(overlap).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(exact.len()).unwrap_or(u32::MAX))
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let path = temp_path("empty", "fsvi");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 8, Quantization::F16)
            .expect("create writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(7, 8), 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
    }

    #[test]
    fn single_vector_round_trip() {
        let path = temp_path("single", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 32)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let hits = ann
            .knn_search(&normalized_vector(1, 32), 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-0000");
    }

    #[test]
    fn higher_ef_improves_or_matches_recall() {
        let fsvi_path = temp_path("ef", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..256).map(|i| normalized_vector(i, 96)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut low_total = 0.0_f64;
        let mut high_total = 0.0_f64;
        let mut count = 0_u32;
        for query_seed in (0..128).step_by(16) {
            let query = normalized_vector(query_seed, 96);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let low = ann.knn_search(&query, 10, 10).expect("low ef");
            let high = ann.knn_search(&query, 10, 100).expect("high ef");
            low_total += recall_at_k(&low, &exact);
            high_total += recall_at_k(&high, &exact);
            count += 1;
        }

        let count_f = f64::from(count);
        assert!((high_total / count_f) >= (low_total / count_f));
    }

    #[test]
    fn recall_against_bruteforce_is_high() {
        let fsvi_path = temp_path("recall", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..1_000).map(|i| normalized_vector(i, 384)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut total_recall = 0.0_f64;
        let mut query_count = 0_u32;
        for query_seed in (0..1_000).step_by(40) {
            let query = normalized_vector(query_seed, 384);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let approx = ann
                .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
                .expect("approx");
            total_recall += recall_at_k(&approx, &exact);
            query_count += 1;
        }

        let avg_recall = total_recall / f64::from(query_count);
        assert!(
            avg_recall >= 0.95,
            "expected avg recall >= 0.95, got {avg_recall:.4}"
        );
    }

    // ── Config validation edge cases ──────────────────────────────────

    #[test]
    fn validate_config_rejects_m_zero() {
        let config = HnswConfig {
            m: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_m_over_256() {
        let config = HnswConfig {
            m: 257,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_construction_zero() {
        let config = HnswConfig {
            ef_construction: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_construction"),
            "expected InvalidConfig for ef_construction, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_search_zero() {
        let config = HnswConfig {
            ef_search: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_search"),
            "expected InvalidConfig for ef_search, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_max_layer_zero() {
        let config = HnswConfig {
            max_layer: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"),
            "expected InvalidConfig for max_layer, got {error:?}"
        );
    }

    #[test]
    fn validate_config_accepts_m_256_boundary() {
        let config = HnswConfig {
            m: 256,
            ..HnswConfig::default()
        };
        assert!(validate_config(config).is_ok());
    }

    // ── build_from_parts error paths ────────────────────────────────────

    #[test]
    fn build_rejects_dimension_zero() {
        let error =
            HnswIndex::build_from_parts(vec![], vec![], 0, HnswConfig::default()).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension"),
            "expected InvalidConfig for dimension, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_doc_id_vector_count_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned(), "b".to_owned()],
            vec![vec![1.0, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vectors"),
            "expected InvalidConfig for vectors, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_vector_dimension_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, 0.0, 0.0]], // 3D but declared 2D
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 2,
                    found: 3
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_nan_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, f32::NAN]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                     if field == "vector" && reason.contains("finite")),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_infinity_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![f32::INFINITY, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vector"),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    // ── knn_search_with_stats boundary conditions ───────────────────────

    #[test]
    fn search_with_k_zero_returns_empty_with_stats() {
        let path = temp_path("k0", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(1, 16), 0, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
        assert_eq!(stats.k_requested, 0);
        assert_eq!(stats.k_returned, 0);
        assert!(stats.is_approximate);
        assert!((stats.estimated_recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn search_dimension_mismatch_returns_error() {
        let path = temp_path("dimmis", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let error = ann
            .knn_search_with_stats(&normalized_vector(1, 8), 5, HNSW_DEFAULT_EF_SEARCH)
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 16,
                    found: 8
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn search_stats_fields_are_populated() {
        let path = temp_path("stats", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..50).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(999, 32), 5, 64)
            .expect("search");
        assert_eq!(stats.index_size, 50);
        assert_eq!(stats.dimension, 32);
        assert_eq!(stats.ef_search, 64);
        assert_eq!(stats.k_requested, 5);
        assert_eq!(stats.k_returned, hits.len());
        assert!(stats.is_approximate);
        assert!(stats.estimated_recall > 0.0);
        assert!(stats.estimated_recall <= 1.0);
    }

    #[test]
    fn search_k_larger_than_index_returns_all() {
        let path = temp_path("klarge", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| normalized_vector(i, 16)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(999, 16), 100, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 5);
    }

    // ── matches_vector_index edge cases ─────────────────────────────────

    #[test]
    fn matches_returns_false_for_dimension_mismatch() {
        let path_a = temp_path("match-a", "fsvi");
        let path_b = temp_path("match-b", "fsvi");
        let index_a = write_index(&path_a, &[normalized_vector(1, 16)]).expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 32)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn matches_returns_false_for_record_count_mismatch() {
        let path_a = temp_path("match-rc-a", "fsvi");
        let path_b = temp_path("match-rc-b", "fsvi");
        let index_a = write_index(
            &path_a,
            &[normalized_vector(1, 16), normalized_vector(2, 16)],
        )
        .expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 16)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    // ── normalize_for_dist_dot ──────────────────────────────────────────

    #[test]
    fn normalize_zero_vector_unchanged() {
        let zero = vec![0.0_f32; 8];
        let result = normalize_for_dist_dot(zero.clone());
        assert_eq!(
            result, zero,
            "zero vector should remain zero after normalize"
        );
    }

    // ── estimate_recall ─────────────────────────────────────────────────

    #[test]
    fn estimate_recall_k_zero_returns_one() {
        assert!((estimate_recall(100, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_recall_clamped_between_zero_and_one() {
        // Very low ef relative to k
        let low = estimate_recall(1, 1000);
        assert!((0.0..=1.0).contains(&low), "low recall: {low}");

        // Very high ef relative to k
        let high = estimate_recall(10_000, 1);
        assert!((0.0..=1.0).contains(&high), "high recall: {high}");
    }

    #[test]
    fn estimate_recall_increases_with_ef() {
        let r_low = estimate_recall(10, 10);
        let r_high = estimate_recall(100, 10);
        assert!(
            r_high >= r_low,
            "recall should increase with ef: {r_low} vs {r_high}"
        );
    }

    // ── len / is_empty / dimension / config accessors ───────────────────

    #[test]
    fn accessors_report_correct_values() {
        let path = temp_path("accessors", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| normalized_vector(i, 24)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let config = HnswConfig {
            m: 8,
            ..HnswConfig::default()
        };
        let ann = HnswIndex::build_from_vector_index(&index, config).expect("ann");
        assert_eq!(ann.len(), 10);
        assert!(!ann.is_empty());
        assert_eq!(ann.dimension(), 24);
        assert_eq!(ann.config().m, 8);
    }

    // ── Debug impl ──────────────────────────────────────────────────────

    #[test]
    fn debug_impl_does_not_panic() {
        let path = temp_path("debug", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 8)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let debug_str = format!("{ann:?}");
        assert!(debug_str.contains("HnswIndex"));
        assert!(debug_str.contains("dimension: 8"));
    }

    // ── Original tests ──────────────────────────────────────────────────

    #[test]
    fn scores_are_consistent_with_exact_top_hit() {
        let fsvi_path = temp_path("score", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..128).map(|i| normalized_vector(i, 64)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let query = normalized_vector(7, 64);
        let exact = index.search_top_k(&query, 1, None).expect("exact");
        let approx = ann
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("approx");

        assert_eq!(exact[0].doc_id, approx[0].doc_id);
        assert!((exact[0].score - approx[0].score).abs() < 1e-3);
    }

    #[test]
    fn persistence_round_trip() {
        let fsvi_path = temp_path("persist", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist", "hnsw");
        ann.save(&save_path).expect("save");

        let loaded = HnswIndex::load(&save_path).expect("load");
        assert_eq!(loaded.len(), 64);
        assert_eq!(loaded.dimension(), 32);

        let query = normalized_vector(10, 32);
        let hits = loaded
            .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0010");
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }
}
