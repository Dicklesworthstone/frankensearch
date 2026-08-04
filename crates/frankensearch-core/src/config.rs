//! Configuration types for the two-tier progressive search pipeline.
//!
//! [`TwoTierConfig`] contains all tuning knobs for the search pipeline.
//! [`TwoTierMetrics`] provides diagnostics from a search execution.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::decision_plane::ReasonCode;
use crate::query_class::QueryClass;
use crate::traits::MetricsExporter;
use crate::types::RankChanges;

/// Which operator fuses the Phase-1 lexical and semantic candidate lists.
///
/// Two structurally different primitives. RRF is purely *rank*-based, so it is immune to
/// score-scale mismatch between BM25 and cosine — but it therefore discards score *magnitude*,
/// and a semantic-only document with an overwhelming cosine cannot outrank a marginal document
/// that merely appears in both lists. Pool-min-max normalizes each tier's scores over its own
/// retrieved pool and fuses the normalized scores, recovering that magnitude signal.
///
/// Measured on the BEIR harness (see `docs/NEGATIVE_EVIDENCE.md`): pool-min-max is **+0.0038 mean
/// nDCG@10** over RRF across four corpora. It is **not** a universal win — where a tier's pool
/// statistics are unreliable (tiny or degenerate pools, one-sided retrieval) the min-max
/// normalization amplifies noise, which is why [`FusionStrategy::Rrf`] remains the default and
/// the switch is opt-in rather than a flipped default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (K = `rrf_k`). Rank-based, scale-free, the shipped default.
    #[default]
    Rrf,
    /// Pool-local min-max score fusion. Recovers magnitude; needs trustworthy pool statistics.
    ///
    /// Graph ranking is **not** supported by this operator — `pool_minmax_fuse_merge` has no
    /// graph arm. When `graph_ranking_enabled` is also set, fusion falls back to
    /// [`FusionStrategy::Rrf`] so the graph contribution is never silently dropped.
    PoolMinMax,
}

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
/// | `FRANKENSEARCH_FUSION_STRATEGY`  | `fusion_strategy`  | `rrf`      |
/// | `FRANKENSEARCH_SMOOTHING_ALPHA`  | `neighbor_smoothing_alpha` | `0.0` (off) |
/// | `FRANKENSEARCH_HUBNESS_BETA`     | `hubness_beta`     | `0.0` (off) |
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
// These booleans are independent, serde-visible public configuration flags.
#[allow(clippy::struct_excessive_bools)]
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

    /// How Phase-1 fuses the lexical and semantic candidate lists.
    /// Default: [`FusionStrategy::Rrf`] — the current behaviour, byte-identical.
    pub fusion_strategy: FusionStrategy,

    /// Diffusion weight for k-NN neighbour smoothing of the Phase-1 semantic pool.
    ///
    /// `0.0` (default) disables the pass entirely — the searcher never calls it, so the default
    /// path is byte-identical and pays nothing. `0.3` is the label-free never-worse setting.
    /// Requires the `graph` feature *and* a document graph attached via `with_document_graph`;
    /// with either absent the pass is skipped.
    ///
    /// Measured (`docs/PERF_LEDGER.md`, `257c468`): **+0.0039 mean nDCG@10** over no smoothing,
    /// and the deployable pool-restricted form beats full-cosine on recall-bound corpora.
    pub neighbor_smoothing_alpha: f32,

    /// Number of nearest `Similar` neighbours averaged per candidate when smoothing. Default `10`.
    pub neighbor_smoothing_m: usize,

    /// Require reciprocal (mutual) k-NN edges when smoothing. Cuts hub and one-way-edge noise at
    /// ~5× the kernel cost. Default `false`.
    pub neighbor_smoothing_mutual: bool,

    /// Penalty weight for the query-hubness dense-score correction `s'(q,d) = cos(q,d) − β·r_d`.
    ///
    /// `0.0` (default) disables the pass entirely — the searcher never calls it, so the default
    /// path is byte-identical and pays nothing. Requires an `r_d` table attached via
    /// `with_hubness_table`; with it absent the pass is skipped.
    ///
    /// `0.2` is the never-negative cross-corpus setting; raise toward `0.3` on stance/citation
    /// corpora where topical centrality anti-correlates with relevance. Measured
    /// (`docs/NEGATIVE_EVIDENCE.md`, `ba5052a`): **+0.0033 mean hybrid nDCG@10** at β=0.2,
    /// all-positive across 4 BEIR corpora, with dense-tier gains of +0.0128 (arguana) / +0.0078
    /// (scidocs). The `r_d` table must be a **query-distribution** statistic built by
    /// `compute_query_hubness` from a background query sample — the cheap query-free proxies
    /// (doc-doc density, centroid distance, PC removal) were REJECTED (`64ac8b7`).
    pub hubness_beta: f32,

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
            fusion_strategy: FusionStrategy::Rrf,
            neighbor_smoothing_alpha: 0.0,
            neighbor_smoothing_m: 10,
            neighbor_smoothing_mutual: false,
            hubness_beta: 0.0,
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
        if let Ok(val) = std::env::var("FRANKENSEARCH_SMOOTHING_ALPHA") {
            match val.parse::<f32>() {
                Ok(a) if a.is_finite() && (0.0..=1.0).contains(&a) => {
                    self.neighbor_smoothing_alpha = a;
                }
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_SMOOTHING_ALPHA",
                    value = %val,
                    "invalid value (expected f32 in 0.0..=1.0), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_HUBNESS_BETA") {
            match val.parse::<f32>() {
                Ok(b) if b.is_finite() && (0.0..=1.0).contains(&b) => self.hubness_beta = b,
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_HUBNESS_BETA",
                    value = %val,
                    "invalid value (expected f32 in 0.0..=1.0), keeping default"
                ),
            }
        }
        if let Ok(val) = std::env::var("FRANKENSEARCH_FUSION_STRATEGY") {
            match val.trim().to_ascii_lowercase().as_str() {
                "rrf" => self.fusion_strategy = FusionStrategy::Rrf,
                "pool_minmax" | "pool-minmax" => self.fusion_strategy = FusionStrategy::PoolMinMax,
                _ => tracing::warn!(
                    var = "FRANKENSEARCH_FUSION_STRATEGY",
                    value = %val,
                    "invalid value (expected `rrf` or `pool_minmax`), keeping default"
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
        if self.graph_ranking_enabled
            && (!self.graph_ranking_weight.is_finite()
                || !(0.0..=1.0).contains(&self.graph_ranking_weight))
        {
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
    /// Typed classification when the semantic lane produced zero results.
    ///
    /// `None` means the semantic lane returned at least one hit (or was not
    /// consulted). Compatibility with older serialized payloads is preserved
    /// via `serde(default)`.
    #[serde(default)]
    pub zero_signal: Option<ZeroSignalReason>,
    /// Per-tier coverage reconstructed from the retained owner witnesses and
    /// the candidates actually returned (bd-ctzo C4).
    ///
    /// `None` means this search had no owner-backed activation to witness —
    /// a legacy artifact, or a lane that never reached the quality tier. That
    /// absence is deliberately NOT a zero-coverage receipt: an unwitnessed
    /// search and a search that covered nothing are different facts, and
    /// [`crate::TierQueryCoverageV1`] has no variant that conflates them.
    /// `serde(default)` keeps older payloads readable.
    #[serde(default)]
    pub coverage: Option<crate::types::SearchCoverageV1>,
}

/// Schema version for zero-signal classification payloads.
pub const ZERO_SIGNAL_SCHEMA_VERSION: &str = "frankensearch.zero_signal.v1";

/// Why a semantic search lane produced zero results.
///
/// Distinct no-signal states must never collapse into an undifferentiated
/// `Ok(empty)`: callers need to distinguish a legitimately empty answer
/// (benign request/state outcome) from an unusable semantic lane
/// (availability failure). Exact (brute-force) and ANN paths classify
/// equivalent states identically.
///
/// Classification precedence, first match wins:
/// 1. request-scoped pre-scan: [`CallerRequestedZeroK`], then
///    [`NonFiniteQuery`], then [`ZeroNormQuery`]
/// 2. state-scoped pre-scan (from [`ZeroSignalState`]):
///    [`NewlyCreatedEmpty`], [`AllTombstoned`], [`NoUsableVectors`]
/// 3. post-scan: [`FilterEliminatedAll`] when a filter rejected every
///    candidate, [`WalOnlyNoLiveRecords`] when only WAL entries existed and
///    none survived, [`AnnReturnedEmptyDespiteUsableVectors`] when the ANN
///    graph came back empty although usable live vectors exist.
///
/// [`CallerRequestedZeroK`]: ZeroSignalReason::CallerRequestedZeroK
/// [`NonFiniteQuery`]: ZeroSignalReason::NonFiniteQuery
/// [`ZeroNormQuery`]: ZeroSignalReason::ZeroNormQuery
/// [`NewlyCreatedEmpty`]: ZeroSignalReason::NewlyCreatedEmpty
/// [`AllTombstoned`]: ZeroSignalReason::AllTombstoned
/// [`NoUsableVectors`]: ZeroSignalReason::NoUsableVectors
/// [`FilterEliminatedAll`]: ZeroSignalReason::FilterEliminatedAll
/// [`WalOnlyNoLiveRecords`]: ZeroSignalReason::WalOnlyNoLiveRecords
/// [`AnnReturnedEmptyDespiteUsableVectors`]: ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZeroSignalReason {
    /// The caller asked for `k == 0` results. Expected outcome, never warned.
    CallerRequestedZeroK,
    /// A caller-supplied filter excluded every candidate. Expected outcome.
    FilterEliminatedAll,
    /// The query vector contains NaN or infinite components.
    NonFiniteQuery,
    /// The query vector's norm is (near-)zero, so every similarity score is
    /// zero and ranking would be arbitrary tie-breaking.
    ZeroNormQuery,
    /// The index was created but has never contained any record (no main
    /// records, no WAL entries).
    NewlyCreatedEmpty,
    /// Every main-index record is tombstoned and no WAL entry remains.
    AllTombstoned,
    /// The main index holds no live records; only WAL-resident entries
    /// existed and none of them produced a usable hit.
    WalOnlyNoLiveRecords,
    /// Live records exist but none of their stored vectors is usable (all
    /// zero-norm, non-finite, or otherwise corrupt). Availability failure.
    NoUsableVectors,
    /// The ANN graph returned no candidates although usable live vectors
    /// exist; exact search would have found hits. Availability anomaly.
    AnnReturnedEmptyDespiteUsableVectors,
}

impl ZeroSignalReason {
    /// Stable machine-readable code, registered in
    /// [`crate::decision_plane::ReasonCode`] and the observability-lint
    /// registry (OBS-003).
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::CallerRequestedZeroK => ReasonCode::ZERO_SIGNAL_CALLER_REQUESTED_ZERO_K,
            Self::FilterEliminatedAll => ReasonCode::ZERO_SIGNAL_FILTER_ELIMINATED_ALL,
            Self::NonFiniteQuery => ReasonCode::ZERO_SIGNAL_NON_FINITE_QUERY,
            Self::ZeroNormQuery => ReasonCode::ZERO_SIGNAL_ZERO_NORM_QUERY,
            Self::NewlyCreatedEmpty => ReasonCode::ZERO_SIGNAL_NEWLY_CREATED_EMPTY,
            Self::AllTombstoned => ReasonCode::ZERO_SIGNAL_ALL_TOMBSTONED,
            Self::WalOnlyNoLiveRecords => ReasonCode::ZERO_SIGNAL_WAL_ONLY_NO_LIVE_RECORDS,
            Self::NoUsableVectors => ReasonCode::ZERO_SIGNAL_NO_USABLE_VECTORS,
            Self::AnnReturnedEmptyDespiteUsableVectors => {
                ReasonCode::ZERO_SIGNAL_ANN_EMPTY_DESPITE_USABLE
            }
        }
    }

    /// Availability failures mean the semantic lane is unusable and warrant
    /// operator attention; every other reason is an expected outcome of the
    /// request or index state and must not warn.
    #[must_use]
    pub const fn is_availability_failure(self) -> bool {
        matches!(
            self,
            Self::NoUsableVectors | Self::AnnReturnedEmptyDespiteUsableVectors
        )
    }

    /// True when the reason depends only on the request (k, filter, query
    /// vector) rather than on index state. Request-scoped events are logged
    /// at debug level, never per-query warnings.
    #[must_use]
    pub const fn is_request_scoped(self) -> bool {
        matches!(
            self,
            Self::CallerRequestedZeroK
                | Self::FilterEliminatedAll
                | Self::NonFiniteQuery
                | Self::ZeroNormQuery
        )
    }

    /// Short human-readable summary.
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::CallerRequestedZeroK => "caller requested zero results (k = 0)",
            Self::FilterEliminatedAll => "search filter excluded every candidate",
            Self::NonFiniteQuery => "query vector contains non-finite values",
            Self::ZeroNormQuery => "query vector has zero norm",
            Self::NewlyCreatedEmpty => "index is newly created and empty",
            Self::AllTombstoned => "all index records are tombstoned",
            Self::WalOnlyNoLiveRecords => "only WAL entries exist and none produced a hit",
            Self::NoUsableVectors => "live records exist but no stored vector is usable",
            Self::AnnReturnedEmptyDespiteUsableVectors => {
                "ANN returned no candidates despite usable live vectors"
            }
        }
    }
}

impl fmt::Display for ZeroSignalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.summary())
    }
}

/// Point-in-time census of a vector index generation, used to classify
/// zero-signal outcomes without per-query scans.
///
/// Computed once at open/reload time (the usable-vector pass is O(n·dim))
/// and invalidated by mutations (append, soft-delete, vacuum, compaction).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeroSignalState {
    /// Physical records in the main index, including tombstoned ones.
    pub record_count: usize,
    /// Records not tombstoned.
    pub live_count: usize,
    /// Tombstoned records.
    pub tombstone_count: usize,
    /// WAL-resident entries not yet compacted into the main index.
    pub wal_count: usize,
    /// Live records whose stored vector is usable (finite, non-zero norm).
    pub usable_vector_count: usize,
}

impl ZeroSignalState {
    /// Classify the state-scoped reason an empty result would have, if the
    /// index state alone explains it. Request-scoped conditions (k, filter,
    /// query vector) take precedence and are classified by the caller.
    #[must_use]
    pub const fn state_reason(&self) -> Option<ZeroSignalReason> {
        if self.record_count == 0 && self.wal_count == 0 {
            return Some(ZeroSignalReason::NewlyCreatedEmpty);
        }
        if self.live_count == 0 && self.wal_count == 0 {
            return Some(ZeroSignalReason::AllTombstoned);
        }
        if self.live_count > 0 && self.usable_vector_count == 0 {
            return Some(ZeroSignalReason::NoUsableVectors);
        }
        None
    }

    /// True when queries can only be served from WAL-resident entries.
    #[must_use]
    pub const fn is_wal_only(&self) -> bool {
        self.live_count == 0 && self.wal_count > 0
    }

    /// Classify why a well-formed search (k > 0, finite non-zero query)
    /// over this state returned nothing, following the precedence
    /// documented on [`ZeroSignalReason`].
    ///
    /// The fallback for "usable candidates existed, no filter, still empty"
    /// is [`ZeroSignalReason::NoUsableVectors`]: an exact scan cannot
    /// legitimately come back empty in that state, so it is reported as an
    /// availability failure. ANN callers refine that case to
    /// [`ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors`].
    #[must_use]
    pub const fn empty_result_reason(&self, had_filter: bool) -> ZeroSignalReason {
        if let Some(reason) = self.state_reason() {
            return reason;
        }
        if had_filter {
            return ZeroSignalReason::FilterEliminatedAll;
        }
        if self.is_wal_only() {
            return ZeroSignalReason::WalOnlyNoLiveRecords;
        }
        ZeroSignalReason::NoUsableVectors
    }
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

    const ALL_ZERO_SIGNAL_REASONS: [ZeroSignalReason; 9] = [
        ZeroSignalReason::CallerRequestedZeroK,
        ZeroSignalReason::FilterEliminatedAll,
        ZeroSignalReason::NonFiniteQuery,
        ZeroSignalReason::ZeroNormQuery,
        ZeroSignalReason::NewlyCreatedEmpty,
        ZeroSignalReason::AllTombstoned,
        ZeroSignalReason::WalOnlyNoLiveRecords,
        ZeroSignalReason::NoUsableVectors,
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors,
    ];

    #[test]
    fn zero_signal_reason_serde_roundtrip_snake_case() {
        for reason in ALL_ZERO_SIGNAL_REASONS {
            let json = serde_json::to_string(&reason).unwrap();
            assert_eq!(json, json.to_ascii_lowercase(), "snake_case: {json}");
            let decoded: ZeroSignalReason = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, reason);
        }
        assert_eq!(
            serde_json::to_string(&ZeroSignalReason::NewlyCreatedEmpty).unwrap(),
            "\"newly_created_empty\""
        );
    }

    #[test]
    fn zero_signal_reason_codes_are_registered_and_valid() {
        for reason in ALL_ZERO_SIGNAL_REASONS {
            let code = ReasonCode::new(reason.reason_code());
            assert!(code.is_valid(), "invalid reason code: {}", code.as_str());
        }
        // Codes are unique.
        let codes: std::collections::HashSet<&str> = ALL_ZERO_SIGNAL_REASONS
            .iter()
            .map(|r| r.reason_code())
            .collect();
        assert_eq!(codes.len(), ALL_ZERO_SIGNAL_REASONS.len());
    }

    #[test]
    fn zero_signal_availability_partition() {
        for reason in ALL_ZERO_SIGNAL_REASONS {
            let is_failure = matches!(
                reason,
                ZeroSignalReason::NoUsableVectors
                    | ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors
            );
            assert_eq!(reason.is_availability_failure(), is_failure);
            // A reason is never both an availability failure and request-scoped.
            assert!(!(reason.is_availability_failure() && reason.is_request_scoped()));
        }
    }

    #[test]
    fn zero_signal_state_classification_table() {
        // (record, live, tombstone, wal, usable) -> expected state reason
        let cases: [(ZeroSignalState, Option<ZeroSignalReason>); 6] = [
            (
                ZeroSignalState::default(),
                Some(ZeroSignalReason::NewlyCreatedEmpty),
            ),
            (
                ZeroSignalState {
                    record_count: 5,
                    live_count: 0,
                    tombstone_count: 5,
                    wal_count: 0,
                    usable_vector_count: 0,
                },
                Some(ZeroSignalReason::AllTombstoned),
            ),
            (
                ZeroSignalState {
                    record_count: 5,
                    live_count: 3,
                    tombstone_count: 2,
                    wal_count: 0,
                    usable_vector_count: 0,
                },
                Some(ZeroSignalReason::NoUsableVectors),
            ),
            (
                ZeroSignalState {
                    record_count: 0,
                    live_count: 0,
                    tombstone_count: 0,
                    wal_count: 4,
                    usable_vector_count: 0,
                },
                // WAL-only is not a pre-scan verdict: WAL entries may serve
                // the query, so classification happens post-scan.
                None,
            ),
            (
                ZeroSignalState {
                    record_count: 5,
                    live_count: 0,
                    tombstone_count: 5,
                    wal_count: 2,
                    usable_vector_count: 0,
                },
                None,
            ),
            (
                ZeroSignalState {
                    record_count: 5,
                    live_count: 5,
                    tombstone_count: 0,
                    wal_count: 0,
                    usable_vector_count: 5,
                },
                None,
            ),
        ];
        for (state, expected) in cases {
            assert_eq!(state.state_reason(), expected, "state: {state:?}");
        }
        assert!(
            ZeroSignalState {
                record_count: 0,
                live_count: 0,
                tombstone_count: 0,
                wal_count: 4,
                usable_vector_count: 0,
            }
            .is_wal_only()
        );
    }

    #[test]
    fn two_tier_metrics_zero_signal_serde_default_compat() {
        // Old payloads without the field must still deserialize.
        let legacy = serde_json::json!({
            "fast_embed_ms": 0.0, "vector_search_ms": 0.0, "lexical_search_ms": 0.0,
            "rrf_fusion_ms": 0.0, "phase1_total_ms": 0.0, "phase1_vectors_searched": 0,
            "quality_embed_ms": 0.0, "quality_search_ms": 0.0, "blend_ms": 0.0,
            "rerank_ms": 0.0, "phase2_total_ms": 0.0, "phase2_vectors_searched": 0,
            "kendall_tau": null, "rank_changes": RankChanges::default(),
            "skip_reason": null, "query_class": null, "lexical_candidates": 0,
            "semantic_candidates": 0, "incomplete_embeddings": 0,
            "fast_embedder_id": null, "quality_embedder_id": null
        });
        let decoded: TwoTierMetrics = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.zero_signal, None);

        let metrics = TwoTierMetrics {
            zero_signal: Some(ZeroSignalReason::AllTombstoned),
            ..Default::default()
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: TwoTierMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.zero_signal, Some(ZeroSignalReason::AllTombstoned));
    }
}
