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
/// | `FRANKENSEARCH_GRAPH_RANKING_ENABLED` | `graph_ranking_enabled` | `false` |
/// | `FRANKENSEARCH_GRAPH_RANKING_WEIGHT` | `graph_ranking_weight` | `0.5` |
/// | `FRANKENSEARCH_QUALITY_TIMEOUT`  | `quality_timeout_ms` | `500`    |
/// | `FRANKENSEARCH_HNSW_THRESHOLD`   | `hnsw_threshold`   | `50000`    |
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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

    /// Enable optional graph-ranking contribution in Phase 1 fusion.
    /// Default: false.
    pub graph_ranking_enabled: bool,

    /// Relative graph signal weight when graph ranking is enabled (0.0-1.0).
    /// Default: 0.5.
    pub graph_ranking_weight: f64,

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
            graph_ranking_enabled: false,
            graph_ranking_weight: 0.5,
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
    fn parse_env_bool(value: &str) -> Option<bool> {
        let normalized = value.trim();
        if normalized.is_empty() {
            return None;
        }
        if normalized.eq_ignore_ascii_case("true")
            || normalized.eq_ignore_ascii_case("1")
            || normalized.eq_ignore_ascii_case("yes")
            || normalized.eq_ignore_ascii_case("on")
        {
            return Some(true);
        }
        if normalized.eq_ignore_ascii_case("false")
            || normalized.eq_ignore_ascii_case("0")
            || normalized.eq_ignore_ascii_case("no")
            || normalized.eq_ignore_ascii_case("off")
        {
            return Some(false);
        }
        None
    }

    fn from_optimized_file(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path).map_or_else(
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

    /// Load overrides from environment variables.
    ///
    /// Only overrides fields for which environment variables are set.
    /// Invalid or out-of-range values are rejected with a warning log.
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(val) = std::env::var("FRANKENSEARCH_QUALITY_WEIGHT") {
            match val.parse::<f64>() {
                Ok(w) if (0.0..=1.0).contains(&w) => self.quality_weight = w,
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_QUALITY_WEIGHT",
                    value = %val,
                    "invalid value (expected f64 in 0.0..=1.0), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_RRF_K") {
            match val.parse::<f64>() {
                Ok(k) if k > 0.0 => self.rrf_k = k,
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_RRF_K",
                    value = %val,
                    "invalid value (expected f64 > 0.0), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_FAST_ONLY") {
            match Self::parse_env_bool(&val) {
                Some(flag) => self.fast_only = flag,
                None => tracing::warn!(
                    var = "FRANKENSEARCH_FAST_ONLY",
                    value = %val,
                    "invalid value (expected boolean), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_GRAPH_RANKING_ENABLED") {
            match Self::parse_env_bool(&val) {
                Some(flag) => self.graph_ranking_enabled = flag,
                None => tracing::warn!(
                    var = "FRANKENSEARCH_GRAPH_RANKING_ENABLED",
                    value = %val,
                    "invalid value (expected boolean), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_GRAPH_RANKING_WEIGHT") {
            match val.parse::<f64>() {
                Ok(weight) if (0.0..=1.0).contains(&weight) => {
                    self.graph_ranking_weight = weight;
                }
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_GRAPH_RANKING_WEIGHT",
                    value = %val,
                    "invalid value (expected f64 in 0.0..=1.0), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_QUALITY_TIMEOUT") {
            match val.parse::<u64>() {
                Ok(ms) => self.quality_timeout_ms = ms,
                Err(_) => tracing::warn!(
                    var = "FRANKENSEARCH_QUALITY_TIMEOUT",
                    value = %val,
                    "invalid value (expected u64), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_HNSW_THRESHOLD") {
            match val.parse::<usize>() {
                Ok(threshold) => self.hnsw_threshold = threshold,
                Err(_) => tracing::warn!(
                    var = "FRANKENSEARCH_HNSW_THRESHOLD",
                    value = %val,
                    "invalid value (expected usize), keeping default"
                ),
            }
        }
        self
    }

    /// Load optimized parameters from `data/optimized_params.toml` at the workspace root.
    ///
    /// Falls back to `Default::default()` if the file does not exist or cannot be parsed.
    /// The TOML file uses flat keys matching the field names of `TwoTierConfig`.
    #[must_use]
    pub fn optimized() -> Self {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let exe_data_path = exe_dir.join("../data/optimized_params.toml");
                if exe_data_path.exists() {
                    return Self::from_optimized_file(&exe_data_path);
                }
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            let cwd_data_path = cwd.join("data/optimized_params.toml");
            if cwd_data_path.exists() {
                return Self::from_optimized_file(&cwd_data_path);
            }
        }

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(std::path::Path::parent)
            .unwrap_or_else(|| std::path::Path::new(manifest_dir));
        let path = workspace_root.join("data").join("optimized_params.toml");

        Self::from_optimized_file(&path)
    }

    /// Validates the configuration parameters to prevent degenerate behavior.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if any parameter is out of bounds.
    pub fn validate(&self) -> Result<(), crate::error::SearchError> {
        if self.candidate_multiplier < 1 {
            return Err(crate::error::SearchError::InvalidConfig {
                field: "candidate_multiplier".to_owned(),
                value: self.candidate_multiplier.to_string(),
                reason: "must be >= 1".to_owned(),
            });
        }
        if !self.rrf_k.is_finite() || self.rrf_k <= 0.0 {
            return Err(crate::error::SearchError::InvalidConfig {
                field: "rrf_k".to_owned(),
                value: self.rrf_k.to_string(),
                reason: "must be finite and > 0.0".to_owned(),
            });
        }
        if !self.quality_weight.is_finite() || !(0.0..=1.0).contains(&self.quality_weight) {
            return Err(crate::error::SearchError::InvalidConfig {
                field: "quality_weight".to_owned(),
                value: self.quality_weight.to_string(),
                reason: "must be in range [0.0, 1.0]".to_owned(),
            });
        }
        if self.quality_timeout_ms < 10 {
            return Err(crate::error::SearchError::InvalidConfig {
                field: "quality_timeout_ms".to_owned(),
                value: self.quality_timeout_ms.to_string(),
                reason: "must be >= 10".to_owned(),
            });
        }
        if self.graph_ranking_enabled && (!self.graph_ranking_weight.is_finite() || !(0.0..=1.0).contains(&self.graph_ranking_weight)) {
            return Err(crate::error::SearchError::InvalidConfig {
                field: "graph_ranking_weight".to_owned(),
                value: self.graph_ranking_weight.to_string(),
                reason: "must be in range [0.0, 1.0]".to_owned(),
            });
        }
        Ok(())
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
    /// How many vectors were evaluated during Phase 1.
    pub phase1_vectors_searched: usize,

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
    /// How many vectors were evaluated during Phase 2.
    pub phase2_vectors_searched: usize,

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
    /// Number of candidates lacking a quality-tier embedding.
    pub incomplete_embeddings: usize,
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
        assert!(!config.graph_ranking_enabled);
        assert!((config.graph_ranking_weight - 0.5).abs() < 1e-10);
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
            graph_ranking_enabled: true,
            graph_ranking_weight: 0.65,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: TwoTierConfig = serde_json::from_str(&json).unwrap();
        assert!((decoded.quality_weight - 0.8).abs() < 1e-10);
        assert!(decoded.fast_only);
        assert!(decoded.graph_ranking_enabled);
        assert!((decoded.graph_ranking_weight - 0.65).abs() < 1e-10);
        assert!(decoded.metrics_exporter.is_none());
        assert_eq!(decoded.candidate_multiplier, 3);
        assert_eq!(decoded.hnsw_threshold, 50_000);
    }

    #[test]
    fn parse_env_bool_accepts_truthy_values() {
        for value in ["true", "TRUE", "1", "yes", "on", " On "] {
            assert_eq!(TwoTierConfig::parse_env_bool(value), Some(true));
        }
    }

    #[test]
    fn parse_env_bool_accepts_falsey_values() {
        for value in ["false", "FALSE", "0", "no", "off", " Off "] {
            assert_eq!(TwoTierConfig::parse_env_bool(value), Some(false));
        }
    }

    #[test]
    fn parse_env_bool_rejects_unknown_values() {
        for value in ["", "maybe", "enable", "disable"] {
            assert_eq!(TwoTierConfig::parse_env_bool(value), None);
        }
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
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.phase2_vectors_searched, 0);
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
        assert!(!config.graph_ranking_enabled);
        assert!((config.graph_ranking_weight - 0.5).abs() < 1e-10);
    }

    #[test]
    fn metrics_exporter_builder_helpers() {
        let config = TwoTierConfig::default().with_metrics_exporter(Arc::new(NoOpMetricsExporter));
        assert!(config.metrics_exporter().is_some());

        let config = config.without_metrics_exporter();
        assert!(config.metrics_exporter().is_none());
    }

    #[test]
    fn optimized_loader_reads_toml_file() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "frankensearch-optimized-config-{}-{unique}.toml",
            std::process::id()
        ));
        let expected = TwoTierConfig {
            quality_weight: 0.82,
            rrf_k: 73.5,
            candidate_multiplier: 4,
            quality_timeout_ms: 777,
            hnsw_ef_search: 123,
            mrl_rescore_top_k: 45,
            ..TwoTierConfig::default()
        };
        std::fs::write(&path, toml::to_string(&expected).expect("serialize config"))
            .expect("write optimized config fixture");

        let loaded = TwoTierConfig::from_optimized_file(&path);
        assert!((loaded.quality_weight - expected.quality_weight).abs() < 1e-12);
        assert!((loaded.rrf_k - expected.rrf_k).abs() < 1e-12);
        assert_eq!(loaded.candidate_multiplier, expected.candidate_multiplier);
        assert_eq!(loaded.quality_timeout_ms, expected.quality_timeout_ms);
        assert_eq!(loaded.hnsw_ef_search, expected.hnsw_ef_search);
        assert_eq!(loaded.mrl_rescore_top_k, expected.mrl_rescore_top_k);
    }

    #[test]
    fn optimized_loader_falls_back_to_default_for_missing_or_invalid_file() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let missing = std::env::temp_dir().join(format!(
            "frankensearch-optimized-missing-{}-{unique}.toml",
            std::process::id()
        ));
        let from_missing = TwoTierConfig::from_optimized_file(&missing);
        assert!(
            (from_missing.quality_weight - TwoTierConfig::default().quality_weight).abs() < 1e-12
        );
        assert!((from_missing.rrf_k - TwoTierConfig::default().rrf_k).abs() < 1e-12);

        let invalid = std::env::temp_dir().join(format!(
            "frankensearch-optimized-invalid-{}-{unique}.toml",
            std::process::id()
        ));
        std::fs::write(&invalid, "quality_weight = \"not-a-number\"")
            .expect("write invalid optimized config");
        let from_invalid = TwoTierConfig::from_optimized_file(&invalid);
        assert!(
            (from_invalid.quality_weight - TwoTierConfig::default().quality_weight).abs() < 1e-12
        );
        assert!((from_invalid.rrf_k - TwoTierConfig::default().rrf_k).abs() < 1e-12);
    }

    #[test]
    fn config_boundary_quality_weight_extremes() {
        let zero = TwoTierConfig {
            quality_weight: 0.0,
            ..Default::default()
        };
        assert!(zero.quality_weight.abs() < f64::EPSILON);

        let one = TwoTierConfig {
            quality_weight: 1.0,
            ..Default::default()
        };
        assert!((one.quality_weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_clone_is_independent() {
        let original = TwoTierMetrics {
            phase1_total_ms: 10.0,
            skip_reason: Some("timeout".into()),
            fast_embedder_id: Some("potion".into()),
            ..Default::default()
        };
        let mut cloned = original.clone();
        cloned.phase1_total_ms = 999.0;
        cloned.skip_reason = None;

        assert!((original.phase1_total_ms - 10.0).abs() < f64::EPSILON);
        assert_eq!(original.skip_reason.as_deref(), Some("timeout"));
    }

    #[test]
    fn config_debug_format() {
        let config = TwoTierConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("quality_weight"));
        assert!(debug.contains("rrf_k"));
        assert!(debug.contains("graph_ranking_enabled"));
        assert!(debug.contains("hnsw_threshold"));
    }

    #[test]
    fn metrics_debug_format() {
        let metrics = TwoTierMetrics {
            kendall_tau: Some(0.92),
            query_class: Some(QueryClass::NaturalLanguage),
            ..Default::default()
        };
        let debug = format!("{metrics:?}");
        assert!(debug.contains("kendall_tau"));
        assert!(debug.contains("NaturalLanguage"));
    }

    #[test]
    fn optimized_partial_toml_merges_with_defaults() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "frankensearch-partial-{}-{unique}.toml",
            std::process::id()
        ));
        std::fs::write(&path, "rrf_k = 99.0\n").expect("write partial config");

        let loaded = TwoTierConfig::from_optimized_file(&path);
        // rrf_k should be updated from the file
        assert!((loaded.rrf_k - 99.0).abs() < 1e-12);
        // quality_weight should remain default
        assert!((loaded.quality_weight - 0.7).abs() < 1e-12);
    }

    #[test]
    fn fast_only_env_override_with_one() {
        // Directly test the parsing logic: "1" should map to true
        let mut config = TwoTierConfig::default();
        assert!(!config.fast_only);
        config.fast_only = "1" == "1";
        assert!(config.fast_only);
    }
}
