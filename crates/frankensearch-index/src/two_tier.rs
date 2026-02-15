//! Two-tier index wrapper for fast and quality vector indices.
//!
//! `TwoTierIndex` provides a single object that coordinates:
//! - fast-tier retrieval from `vector.fast.idx` (or `vector.idx` fallback)
//! - optional quality-tier rescoring from `vector.quality.idx`
//! - doc-id alignment between both tiers

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::{SearchError, SearchResult, TwoTierConfig, VectorHit};
use tracing::{debug, warn};

#[cfg(feature = "ann")]
use crate::{HNSW_DEFAULT_MAX_LAYER, HnswConfig, HnswIndex};
use crate::{VectorIndex, dot_product_f32_f32};

/// Preferred fast-tier index filename.
pub const VECTOR_INDEX_FAST_FILENAME: &str = "vector.fast.idx";
/// Optional quality-tier index filename.
pub const VECTOR_INDEX_QUALITY_FILENAME: &str = "vector.quality.idx";
/// Fallback single-tier index filename used as the fast tier when no dedicated fast file exists.
pub const VECTOR_INDEX_FALLBACK_FILENAME: &str = "vector.idx";
/// Serialized fast-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_FAST_FILENAME: &str = "vector.fast.hnsw";
/// Serialized quality-tier ANN sidecar.
#[cfg(feature = "ann")]
pub const VECTOR_ANN_QUALITY_FILENAME: &str = "vector.quality.hnsw";

/// Dual-index container used by progressive search orchestration.
#[derive(Debug)]
pub struct TwoTierIndex {
    fast_index: VectorIndex,
    quality_index: Option<VectorIndex>,
    #[cfg(feature = "ann")]
    fast_ann: Option<HnswIndex>,
    #[cfg(feature = "ann")]
    quality_ann: Option<HnswIndex>,
    doc_ids: Vec<String>,
    doc_id_to_fast_index: HashMap<String, usize>,
    has_quality: Vec<bool>,
    quality_lookup: Vec<Option<usize>>,
    config: TwoTierConfig,
}

impl TwoTierIndex {
    /// Open a two-tier index from a directory.
    ///
    /// Fast index lookup order:
    /// 1. `{dir}/vector.fast.idx`
    /// 2. `{dir}/vector.idx` (fallback)
    ///
    /// Quality index (optional):
    /// - `{dir}/vector.quality.idx`
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexNotFound` if neither fast-tier file exists,
    /// and propagates index parse/corruption errors from `VectorIndex::open`.
    pub fn open(dir: &Path, config: TwoTierConfig) -> SearchResult<Self> {
        let fast_path = resolve_fast_path(dir)?;
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        let fast_index = VectorIndex::open(&fast_path)?;
        let doc_ids = load_doc_ids(&fast_index)?;
        let doc_id_to_fast_index: HashMap<String, usize> = doc_ids
            .iter()
            .enumerate()
            .map(|(index, doc_id)| (doc_id.clone(), index))
            .collect();
        let mut has_quality = vec![false; doc_ids.len()];
        let mut quality_lookup = vec![None; doc_ids.len()];

        let quality_index = if quality_path.exists() {
            let quality = VectorIndex::open(&quality_path)?;

            if quality.record_count() != fast_index.record_count() {
                warn!(
                    fast_records = fast_index.record_count(),
                    quality_records = quality.record_count(),
                    "fast and quality index record counts differ; using doc-id alignment"
                );
            }

            let fast_positions: HashMap<&str, usize> = doc_ids
                .iter()
                .enumerate()
                .map(|(index, doc_id)| (doc_id.as_str(), index))
                .collect();

            let mut unmatched_quality_docs = 0usize;
            for quality_idx in 0..quality.record_count() {
                let quality_doc_id = quality.doc_id_at(quality_idx)?;
                if let Some(&fast_idx) = fast_positions.get(quality_doc_id) {
                    has_quality[fast_idx] = true;
                    quality_lookup[fast_idx] = Some(quality_idx);
                } else {
                    unmatched_quality_docs += 1;
                }
            }

            if unmatched_quality_docs > 0 {
                warn!(
                    unmatched_quality_docs,
                    "quality index contains doc_ids that are not present in fast index"
                );
            }

            Some(quality)
        } else {
            None
        };

        #[cfg(feature = "ann")]
        let fast_ann = maybe_load_or_build_ann(
            &fast_index,
            &dir.join(VECTOR_ANN_FAST_FILENAME),
            config.hnsw_threshold,
            &config,
            "fast",
        );

        #[cfg(feature = "ann")]
        let quality_ann = quality_index.as_ref().and_then(|quality_index| {
            maybe_load_or_build_ann(
                quality_index,
                &dir.join(VECTOR_ANN_QUALITY_FILENAME),
                config.hnsw_threshold,
                &config,
                "quality",
            )
        });

        #[cfg(feature = "ann")]
        debug!(
            fast_path = %fast_path.display(),
            quality_path = %quality_path.display(),
            quality_available = quality_index.is_some(),
            fast_ann = fast_ann.is_some(),
            quality_ann = quality_ann.is_some(),
            doc_count = doc_ids.len(),
            "opened two-tier index"
        );

        #[cfg(not(feature = "ann"))]
        debug!(
            fast_path = %fast_path.display(),
            quality_path = %quality_path.display(),
            quality_available = quality_index.is_some(),
            doc_count = doc_ids.len(),
            "opened two-tier index"
        );

        Ok(Self {
            fast_index,
            quality_index,
            #[cfg(feature = "ann")]
            fast_ann,
            #[cfg(feature = "ann")]
            quality_ann,
            doc_ids,
            doc_id_to_fast_index,
            has_quality,
            quality_lookup,
            config,
        })
    }

    /// Create a builder for a new two-tier index directory.
    ///
    /// The builder buffers added vectors and writes FSVI files on `finish()`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` if the directory cannot be created.
    pub fn create(dir: &Path, config: TwoTierConfig) -> SearchResult<TwoTierIndexBuilder> {
        fs::create_dir_all(dir)?;
        Ok(TwoTierIndexBuilder::new(dir.to_path_buf(), config))
    }

    /// Search the fast tier only.
    ///
    /// # Errors
    ///
    /// Propagates errors from `HnswIndex::knn_search` (when ANN is selected)
    /// or `VectorIndex::search_top_k` (brute-force fallback).
    pub fn search_fast(&self, query_vec: &[f32], k: usize) -> SearchResult<Vec<VectorHit>> {
        #[cfg(feature = "ann")]
        if let Some(ann) = &self.fast_ann {
            return ann.knn_search(query_vec, k, self.config.hnsw_ef_search);
        }
        self.fast_index.search_top_k(query_vec, k, None)
    }

    /// Compute quality-tier scores for fast-index document positions.
    ///
    /// Missing quality entries produce `0.0`, preserving index alignment for
    /// downstream blending.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if `query_vec` does not match
    /// the quality index dimensionality (when a quality index is present), and
    /// propagates decode/corruption errors from the quality index.
    pub fn quality_scores_for_indices(
        &self,
        query_vec: &[f32],
        indices: &[usize],
    ) -> SearchResult<Vec<f32>> {
        let Some(quality_index) = &self.quality_index else {
            return Ok(vec![0.0; indices.len()]);
        };

        if query_vec.len() != quality_index.dimension() {
            return Err(SearchError::DimensionMismatch {
                expected: quality_index.dimension(),
                found: query_vec.len(),
            });
        }

        let mut scores = Vec::with_capacity(indices.len());
        for &fast_idx in indices {
            let score = self.score_quality_for_fast_index(quality_index, query_vec, fast_idx)?;
            scores.push(score);
        }
        Ok(scores)
    }

    /// Returns true when a quality index was loaded.
    #[must_use]
    pub const fn has_quality_index(&self) -> bool {
        self.quality_index.is_some()
    }

    /// Returns true when fast-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_fast_ann(&self) -> bool {
        self.fast_ann.is_some()
    }

    /// Returns true when quality-tier ANN is loaded/enabled.
    #[cfg(feature = "ann")]
    #[must_use]
    pub const fn has_quality_ann(&self) -> bool {
        self.quality_ann.is_some()
    }

    /// Number of documents in the fast tier (canonical document count).
    #[must_use]
    pub const fn doc_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Document IDs in fast-tier order.
    #[must_use]
    pub fn doc_ids(&self) -> &[String] {
        &self.doc_ids
    }

    /// Fast-tier index position for a given document id.
    #[must_use]
    pub fn fast_index_for_doc_id(&self, doc_id: &str) -> Option<usize> {
        self.doc_id_to_fast_index.get(doc_id).copied()
    }

    /// Whether the fast-tier document at `index` has a quality-tier vector.
    #[must_use]
    pub fn has_quality_for_index(&self, index: usize) -> bool {
        self.has_quality.get(index).copied().unwrap_or(false)
    }

    /// Accessor for the configuration used to open this index.
    #[must_use]
    pub const fn config(&self) -> &TwoTierConfig {
        &self.config
    }

    fn score_quality_for_fast_index(
        &self,
        quality_index: &VectorIndex,
        query_vec: &[f32],
        fast_idx: usize,
    ) -> SearchResult<f32> {
        let Some(Some(quality_idx)) = self.quality_lookup.get(fast_idx).copied() else {
            return Ok(0.0);
        };
        let quality_vector = quality_index.vector_at_f32(quality_idx)?;
        dot_product_f32_f32(&quality_vector, query_vec)
    }
}

/// Builder for writing fast and optional quality FSVI indices.
#[derive(Debug)]
pub struct TwoTierIndexBuilder {
    dir: PathBuf,
    config: TwoTierConfig,
    fast_embedder_id: String,
    quality_embedder_id: String,
    fast_dimension: Option<usize>,
    quality_dimension: Option<usize>,
    fast_records: Vec<(String, Vec<f32>)>,
    quality_records: Vec<(String, Vec<f32>)>,
}

impl TwoTierIndexBuilder {
    fn new(dir: PathBuf, config: TwoTierConfig) -> Self {
        Self {
            dir,
            config,
            fast_embedder_id: "fast-tier".to_owned(),
            quality_embedder_id: "quality-tier".to_owned(),
            fast_dimension: None,
            quality_dimension: None,
            fast_records: Vec::new(),
            quality_records: Vec::new(),
        }
    }

    /// Override the embedder id written to the fast-tier index header.
    pub fn set_fast_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.fast_embedder_id = embedder_id.into();
        self
    }

    /// Override the embedder id written to the quality-tier index header.
    pub fn set_quality_embedder_id(&mut self, embedder_id: impl Into<String>) -> &mut Self {
        self.quality_embedder_id = embedder_id.into();
        self
    }

    /// Add a fast-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added fast-tier vectors.
    pub fn add_fast_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.fast_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        self.fast_records.push((doc_id.into(), embedding.to_vec()));
        Ok(())
    }

    /// Add a quality-tier vector record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if this record dimension differs
    /// from previously added quality-tier vectors.
    pub fn add_quality_record(
        &mut self,
        doc_id: impl Into<String>,
        embedding: &[f32],
    ) -> SearchResult<()> {
        let dimension = embedding.len();
        let expected = self.quality_dimension.get_or_insert(dimension);
        if *expected != dimension {
            return Err(SearchError::DimensionMismatch {
                expected: *expected,
                found: dimension,
            });
        }
        self.quality_records
            .push((doc_id.into(), embedding.to_vec()));
        Ok(())
    }

    /// Add a fast record and optionally a matching quality record.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if either tier dimension is inconsistent.
    pub fn add_record(
        &mut self,
        doc_id: impl Into<String>,
        fast_embedding: &[f32],
        quality_embedding: Option<&[f32]>,
    ) -> SearchResult<()> {
        let doc_id = doc_id.into();
        self.add_fast_record(doc_id.clone(), fast_embedding)?;
        if let Some(quality_embedding) = quality_embedding {
            self.add_quality_record(doc_id, quality_embedding)?;
        }
        Ok(())
    }

    /// Write all buffered records and open the resulting `TwoTierIndex`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no fast-tier records were added,
    /// and propagates writer/open errors from `VectorIndex`.
    pub fn finish(self) -> SearchResult<TwoTierIndex> {
        let fast_dimension = self
            .fast_dimension
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "fast_records".to_owned(),
                value: "0".to_owned(),
                reason: "at least one fast-tier record is required".to_owned(),
            })?;

        let fast_path = self.dir.join(VECTOR_INDEX_FAST_FILENAME);
        let mut fast_writer =
            VectorIndex::create(&fast_path, &self.fast_embedder_id, fast_dimension)?;
        for (doc_id, embedding) in &self.fast_records {
            fast_writer.write_record(doc_id, embedding)?;
        }
        fast_writer.finish()?;

        if let Some(quality_dimension) = self.quality_dimension {
            let quality_path = self.dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let mut quality_writer =
                VectorIndex::create(&quality_path, &self.quality_embedder_id, quality_dimension)?;
            for (doc_id, embedding) in &self.quality_records {
                quality_writer.write_record(doc_id, embedding)?;
            }
            quality_writer.finish()?;
        }

        TwoTierIndex::open(&self.dir, self.config)
    }
}

fn resolve_fast_path(dir: &Path) -> SearchResult<PathBuf> {
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    if fast_path.exists() {
        return Ok(fast_path);
    }

    let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    if fallback_path.exists() {
        return Ok(fallback_path);
    }

    Err(SearchError::IndexNotFound { path: fast_path })
}

fn load_doc_ids(index: &VectorIndex) -> SearchResult<Vec<String>> {
    let mut doc_ids = Vec::with_capacity(index.record_count());
    for record_index in 0..index.record_count() {
        doc_ids.push(index.doc_id_at(record_index)?.to_owned());
    }
    Ok(doc_ids)
}

#[cfg(feature = "ann")]
fn maybe_load_or_build_ann(
    vector_index: &VectorIndex,
    ann_path: &Path,
    threshold: usize,
    config: &TwoTierConfig,
    tier: &str,
) -> Option<HnswIndex> {
    if vector_index.record_count() < threshold {
        return None;
    }

    let ann_config = HnswConfig {
        m: config.hnsw_m,
        ef_construction: config.hnsw_ef_construction,
        ef_search: config.hnsw_ef_search,
        max_layer: HNSW_DEFAULT_MAX_LAYER,
    };

    if ann_path.exists() {
        match HnswIndex::load(ann_path) {
            Ok(ann) => match ann.matches_vector_index(vector_index) {
                Ok(true) => {
                    let loaded_config = ann.config();
                    if loaded_config == ann_config {
                        return Some(ann);
                    }
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?loaded_config,
                        ?ann_config,
                        "ANN sidecar config differs from requested config; rebuilding"
                    );
                }
                Ok(false) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        "ANN sidecar exists but does not match vector index; rebuilding"
                    );
                }
                Err(error) => {
                    warn!(
                        tier,
                        ann_path = %ann_path.display(),
                        ?error,
                        "failed to validate ANN sidecar; rebuilding"
                    );
                }
            },
            Err(error) => {
                warn!(
                    tier,
                    ann_path = %ann_path.display(),
                    ?error,
                    "failed to load ANN sidecar; rebuilding"
                );
            }
        }
    }

    let ann = match HnswIndex::build_from_vector_index(vector_index, ann_config) {
        Ok(ann) => ann,
        Err(error) => {
            warn!(
                tier,
                ?error,
                "failed to build ANN index; using brute-force fallback"
            );
            return None;
        }
    };

    if let Err(error) = ann.save(ann_path) {
        warn!(
            tier,
            ann_path = %ann_path.display(),
            ?error,
            "failed to persist ANN sidecar; ANN stays in-memory for this process"
        );
    }
    Some(ann)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_index_dir(label: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-two-tier-{label}-{}-{timestamp}",
            std::process::id()
        ))
    }

    fn write_index_file(path: &Path, rows: &[(&str, &[f32])]) -> SearchResult<()> {
        let dimension = rows
            .first()
            .map(|(_, vector)| vector.len())
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "rows".to_owned(),
                value: "[]".to_owned(),
                reason: "rows must not be empty".to_owned(),
            })?;
        let mut writer = VectorIndex::create(path, "test", dimension)?;
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector)?;
        }
        writer.finish()
    }

    #[test]
    fn opens_with_fallback_fast_index() {
        let dir = temp_index_dir("fallback");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fallback = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);

        write_index_file(
            &fallback,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
            ],
        )
        .expect("write fallback index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
        assert_eq!(index.doc_ids(), &["doc-a".to_owned(), "doc-b".to_owned()]);
        assert_eq!(index.fast_index_for_doc_id("doc-a"), Some(0));
        assert_eq!(index.fast_index_for_doc_id("doc-b"), Some(1));
        assert_eq!(index.fast_index_for_doc_id("missing"), None);

        let hits = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect("fast search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[test]
    fn quality_alignment_handles_partial_coverage() {
        let dir = temp_index_dir("alignment");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        // Quality tier intentionally omits doc-b and uses different order.
        write_index_file(
            &quality_path,
            &[
                ("doc-c", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
            ],
        )
        .expect("write quality index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open two-tier");
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
        assert!(index.has_quality_for_index(2));

        let scores = index
            .quality_scores_for_indices(&[1.0, 0.0, 0.0, 0.0], &[0, 1, 2])
            .expect("quality scores");
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!(scores[1].abs() < 1e-6);
        assert!(scores[2].abs() < 1e-6);
    }

    #[test]
    fn quality_scores_are_zero_without_quality_index() {
        let dir = temp_index_dir("no-quality");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let scores = index
            .quality_scores_for_indices(&[1.0, 0.0], &[0, 1, 99])
            .expect("scores");
        assert_eq!(scores, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn builder_round_trips_fast_and_quality_records() {
        let dir = temp_index_dir("builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .set_fast_embedder_id("fast-test")
            .set_quality_embedder_id("quality-test");
        builder
            .add_record("doc-a", &[1.0, 0.0, 0.0], Some(&[1.0, 0.0, 0.0]))
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0, 0.0], None)
            .expect("add doc-b");

        let index = builder.finish().expect("finish builder");
        assert_eq!(index.doc_count(), 2);
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
    }

    #[test]
    fn builder_rejects_inconsistent_fast_dimension() {
        let dir = temp_index_dir("bad-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("first record");

        let err = builder
            .add_fast_record("doc-b", &[1.0, 0.0])
            .expect_err("must reject dimension mismatch");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_sidecar_is_created_when_threshold_is_met() {
        let dir = temp_index_dir("ann-enabled");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_ef_search: 32,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open with ann");
        assert!(index.has_fast_ann());
        assert!(dir.join(VECTOR_ANN_FAST_FILENAME).exists());

        let hits = index
            .search_fast(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect("ann search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-a");
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_is_skipped_below_threshold() {
        let dir = temp_index_dir("ann-disabled");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 10_000,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open");
        assert!(!index.has_fast_ann());
        assert!(!dir.join(VECTOR_ANN_FAST_FILENAME).exists());
    }

    // ── Error paths ──────────────────────────────────────────────────

    #[test]
    fn open_returns_index_not_found_when_no_fast_or_fallback() {
        let dir = temp_index_dir("missing");
        fs::create_dir_all(&dir).expect("create temp dir");
        let error = TwoTierIndex::open(&dir, TwoTierConfig::default()).unwrap_err();
        assert!(
            matches!(error, SearchError::IndexNotFound { .. }),
            "expected IndexNotFound, got {error:?}"
        );
    }

    #[test]
    fn quality_scores_dimension_mismatch() {
        let dir = temp_index_dir("quality-dim");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0])])
            .expect("write fast index");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
            .expect("write quality index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_index());

        // Query dimension (4) doesn't match quality dimension (6)
        let error = index
            .quality_scores_for_indices(&[1.0, 0.0, 0.0, 0.0], &[0])
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 6,
                    found: 4
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn has_quality_for_index_out_of_bounds_returns_false() {
        let dir = temp_index_dir("oob");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_for_index(999));
    }

    // ── Builder error paths ─────────────────────────────────────────

    #[test]
    fn builder_finish_rejects_empty_fast_records() {
        let dir = temp_index_dir("empty-builder");
        let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let error = builder.finish().unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "fast_records"),
            "expected InvalidConfig for fast_records, got {error:?}"
        );
    }

    #[test]
    fn builder_rejects_inconsistent_quality_dimension() {
        let dir = temp_index_dir("bad-quality-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("fast record");
        builder
            .add_quality_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("first quality");
        let error = builder
            .add_quality_record("doc-b", &[1.0, 0.0])
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 4,
                    found: 2
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    // ── Fast-tier with explicit fast.idx (not fallback) ─────────────

    #[test]
    fn opens_with_explicit_fast_index() {
        let dir = temp_index_dir("explicit-fast");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-x", &[0.0, 1.0, 0.0]), ("doc-y", &[0.0, 0.0, 1.0])],
        )
        .expect("write fast index");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.doc_count(), 2);
        assert_eq!(index.doc_ids(), &["doc-x".to_owned(), "doc-y".to_owned()]);

        let hits = index.search_fast(&[0.0, 0.0, 1.0], 1).expect("fast search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-y");
    }

    // ── Accessors ───────────────────────────────────────────────────

    #[test]
    fn config_accessor_returns_construction_config() {
        let dir = temp_index_dir("config-acc");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast index");

        let config = TwoTierConfig {
            hnsw_threshold: 42,
            ..TwoTierConfig::default()
        };
        let index = TwoTierIndex::open(&dir, config).expect("open");
        assert_eq!(index.config().hnsw_threshold, 42);
    }

    // ── Quality alignment: unmatched doc_ids ────────────────────────

    #[test]
    fn quality_index_with_extra_doc_ids_still_opens() {
        let dir = temp_index_dir("quality-extra");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
        // Quality has a doc_id not in fast — should trigger warning but still open
        write_index_file(
            &quality_path,
            &[
                ("doc-a", &[1.0, 0.0]),
                ("doc-z", &[0.0, 1.0]), // extra
            ],
        )
        .expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0)); // doc-a matched
        assert_eq!(index.doc_count(), 1); // only fast-tier docs counted
    }

    // ── Builder: fast+quality via add_record convenience ────────────

    #[test]
    fn builder_add_record_with_none_quality_skips_quality_tier() {
        let dir = temp_index_dir("add-record-no-quality");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], None)
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0], None)
            .expect("add doc-b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
    }

    #[cfg(feature = "ann")]
    #[test]
    fn ann_sidecar_rebuilds_when_config_changes() {
        let dir = temp_index_dir("ann-rebuild-config");
        fs::create_dir_all(&dir).expect("create temp dir");

        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0, 0.0, 0.0]),
                ("doc-b", &[0.0, 1.0, 0.0, 0.0]),
                ("doc-c", &[0.0, 0.0, 1.0, 0.0]),
            ],
        )
        .expect("write fast index");

        let initial = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_m: 8,
            hnsw_ef_construction: 64,
            hnsw_ef_search: 16,
            ..TwoTierConfig::default()
        };
        let first_open = TwoTierIndex::open(&dir, initial).expect("open with initial ann config");
        assert!(first_open.has_fast_ann());

        let ann_path = dir.join(VECTOR_ANN_FAST_FILENAME);
        let before = HnswIndex::load(&ann_path).expect("load initial ann sidecar");
        let before_config = before.config();
        assert_eq!(before_config.m, 8);
        assert_eq!(before_config.ef_construction, 64);
        assert_eq!(before_config.ef_search, 16);

        let updated = TwoTierConfig {
            hnsw_threshold: 1,
            hnsw_m: 24,
            hnsw_ef_construction: 96,
            hnsw_ef_search: 48,
            ..TwoTierConfig::default()
        };
        let second_open = TwoTierIndex::open(&dir, updated).expect("open with updated ann config");
        assert!(second_open.has_fast_ann());

        let after = HnswIndex::load(&ann_path).expect("load rebuilt ann sidecar");
        let after_config = after.config();
        assert_eq!(after_config.m, 24);
        assert_eq!(after_config.ef_construction, 96);
        assert_eq!(after_config.ef_search, 48);
    }

    // ─── bd-3szp tests begin ───

    #[test]
    fn filename_constants_are_correct() {
        assert_eq!(VECTOR_INDEX_FAST_FILENAME, "vector.fast.idx");
        assert_eq!(VECTOR_INDEX_QUALITY_FILENAME, "vector.quality.idx");
        assert_eq!(VECTOR_INDEX_FALLBACK_FILENAME, "vector.idx");
    }

    #[test]
    fn two_tier_index_implements_debug() {
        let dir = temp_index_dir("debug-index");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let debug_str = format!("{index:?}");
        assert!(debug_str.contains("TwoTierIndex"));
    }

    #[test]
    fn two_tier_index_builder_implements_debug() {
        let dir = temp_index_dir("debug-builder");
        let builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("TwoTierIndexBuilder"));
    }

    #[test]
    fn fast_index_preferred_over_fallback_when_both_exist() {
        let dir = temp_index_dir("prefer-fast");
        fs::create_dir_all(&dir).expect("create temp dir");

        // Write fallback with doc-fallback
        let fallback_path = dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        write_index_file(&fallback_path, &[("doc-fallback", &[0.0, 1.0])]).expect("write fallback");

        // Write explicit fast with doc-fast
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-fast", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.doc_count(), 1);
        assert_eq!(index.doc_ids(), &["doc-fast".to_owned()]);
    }

    #[test]
    fn search_fast_returns_sorted_by_score_descending() {
        let dir = temp_index_dir("sort-order");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("doc-low", &[0.1, 0.0, 0.0]),
                ("doc-mid", &[0.5, 0.5, 0.0]),
                ("doc-high", &[1.0, 0.0, 0.0]),
            ],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0, 0.0], 3).expect("search");
        assert_eq!(hits.len(), 3);
        // Scores should be in descending order
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
        assert_eq!(hits[0].doc_id, "doc-high");
    }

    #[test]
    fn search_fast_k_zero_returns_empty() {
        let dir = temp_index_dir("k-zero");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0], 0).expect("search k=0");
        assert!(hits.is_empty());
    }

    #[test]
    fn search_fast_k_larger_than_doc_count_returns_all() {
        let dir = temp_index_dir("k-large");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.0, 1.0])],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let hits = index.search_fast(&[1.0, 0.0], 100).expect("search k=100");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn quality_scores_empty_indices_returns_empty() {
        let dir = temp_index_dir("empty-indices");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");
        write_index_file(&quality_path, &[("doc-a", &[1.0, 0.0])]).expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        let scores = index
            .quality_scores_for_indices(&[1.0, 0.0], &[])
            .expect("empty indices");
        assert!(scores.is_empty());
    }

    #[test]
    fn quality_scores_full_coverage() {
        let dir = temp_index_dir("full-quality");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[("doc-a", &[1.0, 0.0, 0.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write fast");

        write_index_file(
            &quality_path,
            &[("doc-a", &[0.0, 0.0, 1.0]), ("doc-b", &[0.0, 1.0, 0.0])],
        )
        .expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(index.has_quality_for_index(0));
        assert!(index.has_quality_for_index(1));

        let scores = index
            .quality_scores_for_indices(&[0.0, 1.0, 0.0], &[0, 1])
            .expect("quality scores");
        assert_eq!(scores.len(), 2);
        // doc-a quality = [0,0,1] dot [0,1,0] = 0.0
        assert!(scores[0].abs() < 1e-6);
        // doc-b quality = [0,1,0] dot [0,1,0] = 1.0
        assert!((scores[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn builder_embedder_id_chaining() {
        let dir = temp_index_dir("embedder-chain");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");

        // Verify chaining returns &mut Self
        let _same_ref = builder
            .set_fast_embedder_id("custom-fast")
            .set_quality_embedder_id("custom-quality");

        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.0, 1.0]))
            .expect("add record");
        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 1);
    }

    #[test]
    fn builder_fast_only_no_quality_index_created() {
        let dir = temp_index_dir("fast-only-builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_fast_record("doc-a", &[1.0, 0.0, 0.0])
            .expect("fast a");
        builder
            .add_fast_record("doc-b", &[0.0, 1.0, 0.0])
            .expect("fast b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(!index.has_quality_index());
        assert!(!dir.join(VECTOR_INDEX_QUALITY_FILENAME).exists());
    }

    #[test]
    fn builder_add_record_with_quality_creates_both_tiers() {
        let dir = temp_index_dir("both-tiers-builder");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[0.5, 0.5, 0.0]))
            .expect("add doc-a");
        builder
            .add_record("doc-b", &[0.0, 1.0], Some(&[0.0, 0.5, 0.5]))
            .expect("add doc-b");

        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 2);
        assert!(index.has_quality_index());
        assert!(index.has_quality_for_index(0));
        assert!(index.has_quality_for_index(1));
    }

    #[test]
    fn builder_preserves_all_doc_ids() {
        let dir = temp_index_dir("all-docids");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        let names = ["zebra", "apple", "mango", "banana"];
        for name in &names {
            builder
                .add_fast_record(*name, &[1.0, 0.0])
                .expect("add record");
        }
        let index = builder.finish().expect("finish");
        assert_eq!(index.doc_count(), 4);
        let mut actual: Vec<&str> = index.doc_ids().iter().map(String::as_str).collect();
        actual.sort_unstable();
        let mut expected = names.to_vec();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn fast_index_for_doc_id_empty_string_returns_none() {
        let dir = temp_index_dir("empty-docid-lookup");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert_eq!(index.fast_index_for_doc_id(""), None);
    }

    #[test]
    fn has_quality_for_index_boundary_last_valid() {
        let dir = temp_index_dir("boundary-last");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);

        write_index_file(
            &fast_path,
            &[
                ("doc-a", &[1.0, 0.0]),
                ("doc-b", &[0.0, 1.0]),
                ("doc-c", &[0.5, 0.5]),
            ],
        )
        .expect("write fast");

        // Quality only has last doc
        write_index_file(&quality_path, &[("doc-c", &[0.5, 0.5])]).expect("write quality");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_for_index(0));
        assert!(!index.has_quality_for_index(1));
        assert!(index.has_quality_for_index(2)); // last valid index
        assert!(!index.has_quality_for_index(3)); // out of bounds
    }

    #[test]
    fn quality_scores_no_quality_index_ignores_query_dimension() {
        // When there's no quality index, any query dimension is accepted (returns all 0.0)
        let dir = temp_index_dir("no-quality-any-dim");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(&fast_path, &[("doc-a", &[1.0, 0.0])]).expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");
        assert!(!index.has_quality_index());

        // Use a completely different dimension query — should still return 0s
        let scores = index
            .quality_scores_for_indices(&[1.0, 2.0, 3.0, 4.0, 5.0], &[0])
            .expect("any dim accepted");
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn search_fast_returns_correct_doc_ids() {
        let dir = temp_index_dir("correct-docids");
        fs::create_dir_all(&dir).expect("create temp dir");
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        write_index_file(
            &fast_path,
            &[
                ("alpha", &[1.0, 0.0, 0.0]),
                ("beta", &[0.0, 1.0, 0.0]),
                ("gamma", &[0.0, 0.0, 1.0]),
            ],
        )
        .expect("write fast");

        let index = TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open");

        // Query aligned with beta
        let hits = index
            .search_fast(&[0.0, 1.0, 0.0], 1)
            .expect("search for beta");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "beta");

        // Query aligned with gamma
        let hits = index
            .search_fast(&[0.0, 0.0, 1.0], 1)
            .expect("search for gamma");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "gamma");
    }

    #[test]
    fn builder_add_record_dimension_mismatch_in_quality() {
        let dir = temp_index_dir("record-quality-dim");
        let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("builder");
        builder
            .add_record("doc-a", &[1.0, 0.0], Some(&[1.0, 0.0, 0.0]))
            .expect("first record ok");

        // Second record has different quality dimension
        let err = builder
            .add_record("doc-b", &[0.0, 1.0], Some(&[1.0, 0.0]))
            .expect_err("quality dim mismatch");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                found: 2
            }
        ));
    }

    #[test]
    fn open_nonexistent_directory_returns_error() {
        let dir = temp_index_dir("nonexistent-subdir");
        // Don't create the directory
        let result = TwoTierIndex::open(&dir, TwoTierConfig::default());
        assert!(result.is_err());
    }

    // ─── bd-3szp tests end ───
}
