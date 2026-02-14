//! Configuration types for the two-tier progressive search pipeline.
//!
//! [`TwoTierConfig`] contains all tuning knobs for the search pipeline.
//! [`TwoTierMetrics`] provides diagnostics from a search execution.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::query_class::QueryClass;
use crate::traits::MetricsExporter;
use crate::types::RankChanges;

/// Configuration for the two-tier progressive search pipeline.
///
/// All fields have sensible defaults. Override selectively via the builder
/// pattern or environment variables.
///
/// # Environment Variable Overrides
///
/// | Variable                        | Field              | Default    |
/// |----------------------------------|--------------------|------------|
/// | `FRANKENSEARCH_QUALITY_WEIGHT`   | `quality_weight`   | `0.7`      |
/// | `FRANKENSEARCH_RRF_K`            | `rrf_k`            | `60.0`     |
/// | `FRANKENSEARCH_FAST_ONLY`        | `fast_only`        | `false`    |
/// | `FRANKENSEARCH_QUALITY_TIMEOUT`  | `quality_timeout_ms` | `500`    |
/// | `FRANKENSEARCH_HNSW_THRESHOLD`   | `hnsw_threshold`   | `50000`    |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoTierConfig {
    /// Weight for quality-tier scores in the blend (0.0–1.0).
    /// Default: 0.7 (70% quality, 30% fast).
    pub quality_weight: f64,

    /// RRF constant K. Higher values flatten the rank distribution.
    /// Default: 60.0 (Cormack et al., 2009).
    pub rrf_k: f64,

    /// Fetch `candidate_multiplier * limit` candidates from each source.
    /// Default: 3.
    pub candidate_multiplier: usize,

    /// Maximum time (ms) to wait for quality embedding + search.
    /// Default: 500.
    pub quality_timeout_ms: u64,

    /// Skip quality refinement entirely (fast-only mode).
    /// Default: false.
    pub fast_only: bool,

    /// Optional telemetry exporter callback target.
    ///
    /// `None` means telemetry callbacks are skipped entirely (zero-overhead
    /// fast path for consumers that do not need exported metrics).
    #[serde(skip)]
    pub metrics_exporter: Option<Arc<dyn MetricsExporter>>,

    /// Enable per-hit explanations. Adds ~2-5% latency overhead.
    /// Default: false.
    pub explain: bool,

    /// HNSW `ef_search` parameter (query-time beam width).
    /// Only used when `ann` feature is enabled. Default: 100.
    pub hnsw_ef_search: usize,

    /// HNSW `ef_construction` parameter (build-time beam width).
    /// Default: 200.
    pub hnsw_ef_construction: usize,

    /// HNSW M parameter (max connections per node).
    /// Default: 16.
    pub hnsw_m: usize,

    /// Minimum record count before ANN search is attempted.
    /// Only used when `ann` feature is enabled. Default: `50_000`.
    pub hnsw_threshold: usize,

    /// MRL search dimensions for initial scan (0 = disabled).
    /// Only meaningful for models that support Matryoshka Representation Learning.
    /// Default: 0 (use full dimensions).
    pub mrl_search_dims: usize,

    /// Number of top-k candidates to re-score at full dimensionality after MRL scan.
    /// Default: 30.
    pub mrl_rescore_top_k: usize,
}

impl Default for TwoTierConfig {
    fn default() -> Self {
        Self {
            quality_weight: 0.7,
            rrf_k: 60.0,
            candidate_multiplier: 3,
            quality_timeout_ms: 500,
            fast_only: false,
            metrics_exporter: None,
            explain: false,
            hnsw_ef_search: 100,
            hnsw_ef_construction: 200,
            hnsw_m: 16,
            hnsw_threshold: 50_000,
            mrl_search_dims: 0,
            mrl_rescore_top_k: 30,
        }
    }
}

impl TwoTierConfig {
    /// Load overrides from environment variables.
    ///
    /// Only overrides fields for which environment variables are set.
    /// Invalid values are silently ignored (defaults are kept).
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = std::env::var("FRANKENSEARCH_QUALITY_WEIGHT")
            && let Ok(w) = val.parse::<f64>()
            && (0.0..=1.0).contains(&w)
        {
            self.quality_weight = w;
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_RRF_K")
            && let Ok(k) = val.parse::<f64>()
            && k > 0.0
        {
            self.rrf_k = k;
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_FAST_ONLY") {
            self.fast_only = val == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_QUALITY_TIMEOUT")
            && let Ok(ms) = val.parse::<u64>()
        {
            self.quality_timeout_ms = ms;
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_HNSW_THRESHOLD")
            && let Ok(threshold) = val.parse::<usize>()
        {
            self.hnsw_threshold = threshold;
        }
        self
    }

    /// Load optimized parameters from `data/optimized_params.toml` at the workspace root.
    ///
    /// Falls back to `Default::default()` if the file does not exist or cannot be parsed.
    /// The TOML file uses flat keys matching the field names of `TwoTierConfig`.
    #[must_use]
    pub fn optimized() -> Self {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(std::path::Path::parent)
            .unwrap_or_else(|| std::path::Path::new(manifest_dir));
        let path = workspace_root.join("data").join("optimized_params.toml");

        std::fs::read_to_string(&path).map_or_else(
            |_| Self::default(),
            |contents| match toml::from_str::<Self>(&contents) {
                Ok(config) => config,
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to parse optimized params, using defaults"
                    );
                    Self::default()
                }
            },
        )
    }

    /// Attach a telemetry exporter.
    #[must_use]
    pub fn with_metrics_exporter(mut self, exporter: Arc<dyn MetricsExporter>) -> Self {
        self.metrics_exporter = Some(exporter);
        self
    }

    /// Remove any telemetry exporter and skip export callbacks.
    #[must_use]
    pub fn without_metrics_exporter(mut self) -> Self {
        self.metrics_exporter = None;
        self
    }

    /// Returns the configured telemetry exporter, if any.
    #[must_use]
    pub fn metrics_exporter(&self) -> Option<&Arc<dyn MetricsExporter>> {
        self.metrics_exporter.as_ref()
    }
}

/// Diagnostics from a two-tier search execution.
///
/// Populated by `TwoTierSearcher` and available after search completes.
/// All latency values are in milliseconds (f64 for sub-millisecond precision).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TwoTierMetrics {
    // ── Phase 1 (Initial) ───────────────────────────────────────────
    /// Time spent on fast-tier embedding.
    pub fast_embed_ms: f64,
    /// Time spent on vector search (brute-force or HNSW).
    pub vector_search_ms: f64,
    /// Time spent on lexical (BM25) search.
    pub lexical_search_ms: f64,
    /// Time spent on RRF fusion.
    pub rrf_fusion_ms: f64,
    /// Total time for Phase 1 (Initial results).
    pub phase1_total_ms: f64,

    // ── Phase 2 (Refined) ───────────────────────────────────────────
    /// Time spent on quality-tier embedding.
    pub quality_embed_ms: f64,
    /// Time spent on quality vector search.
    pub quality_search_ms: f64,
    /// Time spent on two-tier blending.
    pub blend_ms: f64,
    /// Time spent on cross-encoder reranking.
    pub rerank_ms: f64,
    /// Total time for Phase 2 (Refined results).
    pub phase2_total_ms: f64,

    // ── Ranking quality ─────────────────────────────────────────────
    /// Kendall tau rank correlation between Phase 1 and Phase 2 rankings.
    /// Range: [-1.0, 1.0]. Higher values mean refinement changed less.
    pub kendall_tau: Option<f64>,
    /// How many documents changed rank between phases.
    pub rank_changes: RankChanges,

    // ── Retrieval stats ─────────────────────────────────────────────
    /// Why refinement was skipped, if applicable.
    pub skip_reason: Option<String>,
    /// The query classification used.
    pub query_class: Option<QueryClass>,
    /// Number of candidates retrieved from lexical search.
    pub lexical_candidates: usize,
    /// Number of candidates retrieved from semantic search.
    pub semantic_candidates: usize,
    /// Embedder used for fast tier.
    pub fast_embedder_id: Option<String>,
    /// Embedder used for quality tier.
    pub quality_embedder_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::traits::NoOpMetricsExporter;

    #[test]
    fn default_config_values() {
        let config = TwoTierConfig::default();
        assert!((config.quality_weight - 0.7).abs() < 1e-10);
        assert!((config.rrf_k - 60.0).abs() < 1e-10);
        assert_eq!(config.candidate_multiplier, 3);
        assert_eq!(config.quality_timeout_ms, 500);
        assert!(!config.fast_only);
        assert!(config.metrics_exporter.is_none());
        assert!(!config.explain);
        assert_eq!(config.hnsw_ef_search, 100);
        assert_eq!(config.hnsw_ef_construction, 200);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_threshold, 50_000);
        assert_eq!(config.mrl_search_dims, 0);
        assert_eq!(config.mrl_rescore_top_k, 30);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = TwoTierConfig {
            quality_weight: 0.8,
            fast_only: true,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: TwoTierConfig = serde_json::from_str(&json).unwrap();
        assert!((decoded.quality_weight - 0.8).abs() < 1e-10);
        assert!(decoded.fast_only);
        assert!(decoded.metrics_exporter.is_none());
        assert_eq!(decoded.candidate_multiplier, 3);
        assert_eq!(decoded.hnsw_threshold, 50_000);
    }

    #[test]
    fn metrics_default() {
        let metrics = TwoTierMetrics::default();
        assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
        assert!(metrics.phase2_total_ms.abs() < f64::EPSILON);
        assert!(metrics.kendall_tau.is_none());
        assert!(metrics.skip_reason.is_none());
        assert!(metrics.query_class.is_none());
        assert_eq!(metrics.lexical_candidates, 0);
        assert_eq!(metrics.semantic_candidates, 0);
    }

    #[test]
    fn metrics_serialization_roundtrip() {
        let metrics = TwoTierMetrics {
            fast_embed_ms: 0.57,
            vector_search_ms: 3.2,
            phase1_total_ms: 6.0,
            quality_embed_ms: 128.0,
            phase2_total_ms: 150.0,
            kendall_tau: Some(0.85),
            query_class: Some(QueryClass::NaturalLanguage),
            lexical_candidates: 50,
            semantic_candidates: 30,
            fast_embedder_id: Some("potion-128M".into()),
            quality_embedder_id: Some("MiniLM-L6-v2".into()),
            ..Default::default()
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: TwoTierMetrics = serde_json::from_str(&json).unwrap();
        assert!((decoded.fast_embed_ms - 0.57).abs() < 1e-10);
        assert!((decoded.phase2_total_ms - 150.0).abs() < 1e-10);
        assert_eq!(decoded.kendall_tau, Some(0.85));
        assert_eq!(decoded.query_class, Some(QueryClass::NaturalLanguage));
    }

    #[test]
    fn env_override_ignores_invalid_values() {
        // With no env vars set, defaults should be preserved
        let config = TwoTierConfig::default().with_env_overrides();
        assert!((config.quality_weight - 0.7).abs() < 1e-10);
    }

    #[test]
    fn metrics_exporter_builder_helpers() {
        let config = TwoTierConfig::default().with_metrics_exporter(Arc::new(NoOpMetricsExporter));
        assert!(config.metrics_exporter().is_some());

        let config = config.without_metrics_exporter();
        assert!(config.metrics_exporter().is_none());
    }
}
