//! RRF fusion, score blending, and two-tier progressive search for frankensearch.
//!
//! This crate provides:
//! - **RRF**: Reciprocal Rank Fusion (K=60) with 4-level tie-breaking.
//! - **Blending**: Two-tier score blending (0.7 quality / 0.3 fast).
//! - **`TwoTierSearcher`**: Progressive iterator orchestrator yielding `SearchPhase` results.
//! - **Query classification**: Adaptive candidate budgets per query class.
//! - **Score calibration**: Platt/isotonic/temperature scaling for raw scores.
//! - **Circuit breaker**: Quality-tier health monitoring with automatic skip.
//! - **Phase gate**: Anytime-valid sequential testing for phase transition decisions.
//! - **Feedback**: Implicit relevance feedback with exponentially-decaying boost map.

pub mod adaptive;
pub mod blend;
pub mod cache;
pub mod calibration;
pub mod circuit_breaker;
pub mod conformal;
pub mod federated;
pub mod feedback;
#[cfg(feature = "graph")]
pub mod graph_rank;
pub mod incremental;
pub mod interaction_lanes;
pub mod interaction_oracles;
pub mod mmr;
pub mod normalize;
pub mod ope;
pub mod phase_gate;
pub mod prf;
pub mod queue;
pub mod refresh;
pub mod rrf;
pub mod searcher;

pub use adaptive::{
    AdaptiveConfig, AdaptiveFusion, AdaptiveSnapshot, BlendPosterior, EvidenceEvent, KPosterior,
    SignalSource,
};
pub use blend::{
    blend_two_tier, build_borrowed_rank_map, compute_rank_changes, compute_rank_changes_with_maps,
    kendall_tau, kendall_tau_with_refined_rank,
};
pub use cache::{
    IndexCache, IndexSentinel, IndexStaleness, SENTINEL_FILENAME, SENTINEL_VERSION,
    SentinelFileDetector, StalenessDetector,
};
pub use calibration::{
    CalibratorConfig, Identity, IsotonicRegression, PlattScaling, ScoreCalibrator,
    TemperatureScaling, compute_brier_score, compute_ece,
};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitMetrics, QualityOutcome};
pub use conformal::{
    AdaptiveConformalState, AdaptiveConformalUpdate, ConformalSearchCalibration,
    MondrianConformalCalibration,
};
pub use federated::{FederatedConfig, FederatedFusion, FederatedHit, FederatedSearcher};
pub use feedback::{FeedbackCollector, FeedbackConfig, FeedbackSignal, SignalWeights};
#[cfg(feature = "graph")]
pub use graph_rank::GraphRanker;
pub use incremental::{IncrementalConfig, IncrementalSearcher, SearchPlan, SearchStrategy};
pub use interaction_lanes::{
    CalibratorChoice, CorpusSlice, ExpectedPhase, FeatureToggles, FixtureQuery, InteractionLane,
    QuerySlice, RiskLevel, derive_query_seed, fixture_queries, lane_by_id, lane_catalog, lane_ids,
    lanes_at_risk, queries_for_lane,
};
pub use interaction_oracles::{
    InvariantCategory, InvariantGroup, LaneOracleMapping, LaneOracleTemplate, LaneTestReport,
    OracleDescriptor, OracleOutcome, OracleRequirements, OracleVerdict, RequiredFeature,
    all_oracles, compute_lane_oracle_mappings, lane_oracle_templates, oracle_applicable,
    oracle_template_for_lane, oracles_for_lane,
};
pub use mmr::{MmrConfig, mmr_rerank};
pub use normalize::{
    NormalizationMethod, min_max_normalize, normalize_in_place, normalize_scores,
    normalize_scores_with_method, z_score_normalize,
};
pub use ope::{
    LoggedObservation, OpeConfig, OpeResult, dr_estimate, effective_sample_size, ips_estimate,
};
pub use phase_gate::{PhaseDecision, PhaseGate, PhaseGateConfig, PhaseObservation};
pub use prf::{PrfConfig, prf_expand};
pub use queue::{
    EmbeddingJob, EmbeddingQueue, EmbeddingQueueConfig, EmbeddingRequest, JobOutcome, QueueMetrics,
};
pub use refresh::{RefreshMetrics, RefreshMetricsSnapshot, RefreshWorker, RefreshWorkerConfig};
pub use rrf::{RrfConfig, candidate_count, rrf_fuse, rrf_fuse_with_graph};
pub use searcher::TwoTierSearcher;
