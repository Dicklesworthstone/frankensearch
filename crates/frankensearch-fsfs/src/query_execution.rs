//! Multi-stage query execution orchestration for fsfs runtime lanes.
//!
//! This module specifies:
//! - stage sequencing across lexical/semantic/fusion/rerank phases
//! - cancellation semantics with partial-result correctness
//! - deterministic rank fusion with explicit tie-break policy
//! - degraded-mode compatibility driven by pressure state

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::config::PressureProfile;
use crate::orchestration::BackpressureMode;
use crate::pressure::PressureState;
use crate::query_planning::{QueryFallbackPath, QueryIntentDecision, RetrievalBudget};

/// Query execution stages in deterministic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalStage {
    Canonicalize,
    ClassifyIntent,
    MetadataFallbackRetrieve,
    LexicalRetrieve,
    FastSemanticRetrieve,
    Fuse,
    QualitySemanticRetrieve,
    Rerank,
}

/// Stage-level execution contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    pub stage: RetrievalStage,
    pub timeout_ms: u64,
    pub fanout: usize,
    pub required: bool,
    pub reason_code: &'static str,
}

/// Runtime mode for degraded compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedRetrievalMode {
    Normal,
    EmbedDeferred,
    LexicalOnly,
    MetadataOnly,
    Paused,
}

impl DegradedRetrievalMode {
    const fn severity(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::EmbedDeferred => 1,
            Self::LexicalOnly => 2,
            Self::MetadataOnly => 3,
            Self::Paused => 4,
        }
    }

    const fn next_more_restrictive(self) -> Self {
        match self {
            Self::Normal => Self::EmbedDeferred,
            Self::EmbedDeferred => Self::LexicalOnly,
            Self::LexicalOnly => Self::MetadataOnly,
            Self::MetadataOnly | Self::Paused => Self::Paused,
        }
    }

    const fn next_less_restrictive(self) -> Self {
        match self {
            Self::Paused => Self::MetadataOnly,
            Self::MetadataOnly => Self::LexicalOnly,
            Self::LexicalOnly => Self::EmbedDeferred,
            Self::EmbedDeferred | Self::Normal => Self::Normal,
        }
    }
}

/// Operator override for degradation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationOverride {
    Auto,
    ForceNormal,
    ForceEmbedDeferred,
    ForceLexicalOnly,
    ForceMetadataOnly,
    ForcePause,
}

impl DegradationOverride {
    const fn forced_mode(self) -> Option<DegradedRetrievalMode> {
        match self {
            Self::Auto => None,
            Self::ForceNormal => Some(DegradedRetrievalMode::Normal),
            Self::ForceEmbedDeferred => Some(DegradedRetrievalMode::EmbedDeferred),
            Self::ForceLexicalOnly => Some(DegradedRetrievalMode::LexicalOnly),
            Self::ForceMetadataOnly => Some(DegradedRetrievalMode::MetadataOnly),
            Self::ForcePause => Some(DegradedRetrievalMode::Paused),
        }
    }
}

/// Signals that drive degradation decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationSignals {
    pub pressure_state: PressureState,
    pub backpressure_mode: BackpressureMode,
    pub quality_circuit_open: bool,
}

impl Default for DegradationSignals {
    fn default() -> Self {
        Self {
            pressure_state: PressureState::Normal,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: false,
        }
    }
}

/// Transition thresholds for the degradation state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationPolicy {
    pub escalate_after: u8,
    pub recover_after: u8,
}

impl DegradationPolicy {
    #[must_use]
    pub const fn for_profile(profile: PressureProfile) -> Self {
        match profile {
            PressureProfile::Strict => Self {
                escalate_after: 1,
                recover_after: 3,
            },
            PressureProfile::Performance => Self {
                escalate_after: 2,
                recover_after: 3,
            },
            PressureProfile::Degraded => Self {
                escalate_after: 2,
                recover_after: 5,
            },
        }
    }

    #[must_use]
    pub const fn normalized(self) -> Self {
        Self {
            escalate_after: if self.escalate_after == 0 {
                1
            } else {
                self.escalate_after
            },
            recover_after: if self.recover_after == 0 {
                1
            } else {
                self.recover_after
            },
        }
    }
}

impl Default for DegradationPolicy {
    fn default() -> Self {
        Self::for_profile(PressureProfile::Performance)
    }
}

/// User-facing degradation banner and control hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationStatus {
    pub banner: &'static str,
    pub controls_hint: &'static str,
}

/// State-machine transition summary with audit reason code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationTransition {
    pub from: DegradedRetrievalMode,
    pub to: DegradedRetrievalMode,
    pub changed: bool,
    pub reason_code: &'static str,
    pub status: DegradationStatus,
    pub override_mode: DegradationOverride,
}

/// Ladder-based graceful degradation controller for query execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationStateMachine {
    policy: DegradationPolicy,
    mode: DegradedRetrievalMode,
    override_mode: DegradationOverride,
    pending_target: Option<DegradedRetrievalMode>,
    pending_count: u8,
}

impl Default for DegradationStateMachine {
    fn default() -> Self {
        Self::new(DegradationPolicy::default())
    }
}

impl DegradationStateMachine {
    #[must_use]
    pub const fn new(policy: DegradationPolicy) -> Self {
        Self {
            policy: policy.normalized(),
            mode: DegradedRetrievalMode::Normal,
            override_mode: DegradationOverride::Auto,
            pending_target: None,
            pending_count: 0,
        }
    }

    #[must_use]
    pub const fn for_profile(profile: PressureProfile) -> Self {
        Self::new(DegradationPolicy::for_profile(profile))
    }

    #[must_use]
    pub const fn mode(&self) -> DegradedRetrievalMode {
        self.mode
    }

    #[must_use]
    pub const fn override_mode(&self) -> DegradationOverride {
        self.override_mode
    }

    pub const fn set_override(&mut self, override_mode: DegradationOverride) {
        self.override_mode = override_mode;
        self.pending_target = None;
        self.pending_count = 0;
    }

    #[must_use]
    pub fn observe(&mut self, signals: DegradationSignals) -> DegradationTransition {
        let from = self.mode;
        let forced_mode = self.override_mode.forced_mode();
        let desired = forced_mode.unwrap_or_else(|| target_mode(signals));
        if forced_mode.is_some() && self.mode != desired {
            self.mode = desired;
            return DegradationTransition {
                from,
                to: self.mode,
                changed: true,
                reason_code: "degrade.override.applied",
                status: status_for_mode(self.mode),
                override_mode: self.override_mode,
            };
        }
        if forced_mode.is_some() {
            return DegradationTransition {
                from,
                to: self.mode,
                changed: false,
                reason_code: "degrade.override.stable",
                status: status_for_mode(self.mode),
                override_mode: self.override_mode,
            };
        }

        let candidate = match desired.severity().cmp(&self.mode.severity()) {
            Ordering::Greater => self.mode.next_more_restrictive(),
            Ordering::Less => self.mode.next_less_restrictive(),
            Ordering::Equal => self.mode,
        };

        if candidate == self.mode {
            self.pending_target = None;
            self.pending_count = 0;
            return DegradationTransition {
                from,
                to: self.mode,
                changed: false,
                reason_code: "degrade.transition.stable",
                status: status_for_mode(self.mode),
                override_mode: self.override_mode,
            };
        }

        if self.pending_target == Some(candidate) {
            self.pending_count = self.pending_count.saturating_add(1);
        } else {
            self.pending_target = Some(candidate);
            self.pending_count = 1;
        }

        let threshold = if candidate.severity() > self.mode.severity() {
            self.policy.escalate_after
        } else {
            self.policy.recover_after
        };
        if self.pending_count < threshold {
            return DegradationTransition {
                from,
                to: self.mode,
                changed: false,
                reason_code: "degrade.transition.hysteresis_hold",
                status: status_for_mode(self.mode),
                override_mode: self.override_mode,
            };
        }

        self.mode = candidate;
        self.pending_target = None;
        self.pending_count = 0;
        DegradationTransition {
            from,
            to: self.mode,
            changed: true,
            reason_code: transition_reason_code(self.mode),
            status: status_for_mode(self.mode),
            override_mode: self.override_mode,
        }
    }
}

/// Cancellation point used to resolve deterministic cancellation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationPoint {
    BeforeInitialCandidates,
    DuringInitialFusion,
    AfterInitialResults,
    DuringRefinement,
}

/// Action taken when cancellation is observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationAction {
    PropagateCancelledError,
    EmitPartiallyFusedResults,
    EmitInitialResults,
}

/// Cancellation directive emitted from a cancellation point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancellationDirective {
    pub action: CancellationAction,
    pub reason_code: &'static str,
}

/// Plan assembled for one query execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryExecutionPlan {
    pub mode: DegradedRetrievalMode,
    pub stages: Vec<StagePlan>,
    pub reason_code: &'static str,
}

/// Lexical candidate sorted by descending lexical relevance.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalCandidate {
    pub doc_id: String,
    pub score: f32,
}

impl LexicalCandidate {
    #[must_use]
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
        }
    }
}

/// Semantic candidate sorted by descending similarity.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticCandidate {
    pub doc_id: String,
    pub score: f32,
}

impl SemanticCandidate {
    #[must_use]
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
        }
    }
}

/// Fused ranking row with deterministic tie-break metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedCandidate {
    pub doc_id: String,
    pub fused_score: f64,
    pub prior_boost: f64,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub in_both_sources: bool,
}

/// Optional prior signals attached to a candidate.
///
/// All signals are expected in the normalized range `[0.0, 1.0]`.
/// Non-finite values are sanitized to `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RankingPriorSignals {
    pub recency: Option<f64>,
    pub path: Option<f64>,
    pub project: Option<f64>,
}

impl RankingPriorSignals {
    #[must_use]
    pub const fn new(recency: Option<f64>, path: Option<f64>, project: Option<f64>) -> Self {
        Self {
            recency,
            path,
            project,
        }
    }
}

/// Weights for prior families used to post-adjust fused scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankingPriorWeights {
    pub recency: f64,
    pub path: f64,
    pub project: f64,
}

impl RankingPriorWeights {
    #[must_use]
    pub const fn new(recency: f64, path: f64, project: f64) -> Self {
        Self {
            recency,
            path,
            project,
        }
    }
}

impl Default for RankingPriorWeights {
    fn default() -> Self {
        Self::new(0.12, 0.08, 0.05)
    }
}

/// Deterministic tuning controls for recency/path/project priors.
///
/// `max_total_boost` caps the total additive prior contribution so priors
/// cannot dominate base retrieval signals. This preserves reproducibility and
/// stable tie-break behavior under profile changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankingPriorTuning {
    pub enabled: bool,
    pub weights: RankingPriorWeights,
    pub max_total_boost: f64,
}

impl RankingPriorTuning {
    #[must_use]
    pub const fn for_profile(profile: PressureProfile) -> Self {
        match profile {
            PressureProfile::Strict => Self {
                enabled: true,
                weights: RankingPriorWeights::new(0.03, 0.02, 0.01),
                max_total_boost: 0.002,
            },
            PressureProfile::Performance => Self {
                enabled: true,
                weights: RankingPriorWeights::new(0.12, 0.08, 0.05),
                max_total_boost: 0.01,
            },
            PressureProfile::Degraded => Self {
                enabled: true,
                weights: RankingPriorWeights::new(0.06, 0.03, 0.02),
                max_total_boost: 0.004,
            },
        }
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let max_total_boost = sanitize_max_prior_boost(self.max_total_boost);
        let raw_weights = self.weights;
        let mut recency = sanitize_non_negative(raw_weights.recency);
        let mut path = sanitize_non_negative(raw_weights.path);
        let mut project = sanitize_non_negative(raw_weights.project);
        let sum = recency + path + project;
        if sum > max_total_boost && sum > 0.0 {
            let scale = max_total_boost / sum;
            recency *= scale;
            path *= scale;
            project *= scale;
        }
        Self {
            enabled: self.enabled,
            weights: RankingPriorWeights::new(recency, path, project),
            max_total_boost,
        }
    }

    #[must_use]
    fn boost_for(self, signals: RankingPriorSignals) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let recency = sanitize_signal(signals.recency);
        let path = sanitize_signal(signals.path);
        let project = sanitize_signal(signals.project);
        let boost = self.weights.recency.mul_add(
            recency,
            self.weights
                .path
                .mul_add(path, self.weights.project * project),
        );
        boost.clamp(0.0, self.max_total_boost)
    }
}

impl Default for RankingPriorTuning {
    fn default() -> Self {
        Self::for_profile(PressureProfile::Performance)
    }
}

/// Deterministic fusion policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionPolicy {
    pub rrf_k: f64,
}

impl Default for FusionPolicy {
    fn default() -> Self {
        Self { rrf_k: 60.0 }
    }
}

impl FusionPolicy {
    #[must_use]
    fn effective_k(self) -> f64 {
        if self.rrf_k.is_finite() && self.rrf_k > 0.0 {
            self.rrf_k
        } else {
            60.0
        }
    }
}

/// Stateless orchestrator for planning and deterministic fusion.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QueryExecutionOrchestrator {
    pub fusion_policy: FusionPolicy,
}

impl QueryExecutionOrchestrator {
    #[must_use]
    pub const fn new(fusion_policy: FusionPolicy) -> Self {
        Self { fusion_policy }
    }

    /// Build multi-stage retrieval plan for one query.
    #[must_use]
    pub fn plan(
        &self,
        intent: &QueryIntentDecision,
        budget: &RetrievalBudget,
        pressure_state: PressureState,
    ) -> QueryExecutionPlan {
        self.plan_with_mode(intent, budget, mode_from_pressure(pressure_state))
    }

    /// Build retrieval plan for an explicit degradation mode.
    #[must_use]
    pub fn plan_with_mode(
        &self,
        intent: &QueryIntentDecision,
        budget: &RetrievalBudget,
        mode: DegradedRetrievalMode,
    ) -> QueryExecutionPlan {
        let mut stages = vec![
            StagePlan {
                stage: RetrievalStage::Canonicalize,
                timeout_ms: 5,
                fanout: 0,
                required: true,
                reason_code: "query.stage.canonicalize",
            },
            StagePlan {
                stage: RetrievalStage::ClassifyIntent,
                timeout_ms: 5,
                fanout: 0,
                required: true,
                reason_code: "query.stage.classify_intent",
            },
        ];

        if matches!(mode, DegradedRetrievalMode::MetadataOnly) {
            stages.push(StagePlan {
                stage: RetrievalStage::MetadataFallbackRetrieve,
                timeout_ms: stage_timeout(budget.latency_budget_ms, 1, 1, 10, 60),
                fanout: budget.limit,
                required: true,
                reason_code: "query.stage.metadata_fallback_retrieve",
            });
        } else if !matches!(mode, DegradedRetrievalMode::Paused) && budget.lexical_fanout > 0 {
            stages.push(StagePlan {
                stage: RetrievalStage::LexicalRetrieve,
                timeout_ms: stage_timeout(budget.latency_budget_ms, 1, 3, 20, 120),
                fanout: budget.lexical_fanout,
                required: true,
                reason_code: "query.stage.lexical_retrieve",
            });
        }

        if matches!(
            mode,
            DegradedRetrievalMode::Normal | DegradedRetrievalMode::EmbedDeferred
        ) && budget.semantic_fanout > 0
            && !matches!(intent.fallback, QueryFallbackPath::MalformedLexicalOnly)
        {
            stages.push(StagePlan {
                stage: RetrievalStage::FastSemanticRetrieve,
                timeout_ms: stage_timeout(budget.latency_budget_ms, 1, 2, 25, 160),
                fanout: budget.semantic_fanout,
                required: true,
                reason_code: "query.stage.fast_semantic_retrieve",
            });
        }

        if !matches!(
            mode,
            DegradedRetrievalMode::MetadataOnly | DegradedRetrievalMode::Paused
        ) {
            stages.push(StagePlan {
                stage: RetrievalStage::Fuse,
                timeout_ms: 10,
                fanout: budget.limit,
                required: true,
                reason_code: "query.stage.fuse",
            });
        }

        let allow_refinement = budget.quality_enabled
            && matches!(mode, DegradedRetrievalMode::Normal)
            && !matches!(intent.fallback, QueryFallbackPath::MalformedLexicalOnly);

        if allow_refinement {
            stages.push(StagePlan {
                stage: RetrievalStage::QualitySemanticRetrieve,
                timeout_ms: budget.latency_budget_ms.max(50),
                fanout: budget.semantic_fanout.min(budget.limit.saturating_mul(4)),
                required: false,
                reason_code: "query.stage.quality_semantic_retrieve",
            });
        }

        if allow_refinement && budget.rerank_depth > 0 {
            stages.push(StagePlan {
                stage: RetrievalStage::Rerank,
                timeout_ms: stage_timeout(budget.latency_budget_ms, 1, 2, 25, 250),
                fanout: budget.rerank_depth,
                required: false,
                reason_code: "query.stage.rerank",
            });
        }

        let reason_code = match mode {
            DegradedRetrievalMode::Normal => {
                if matches!(intent.fallback, QueryFallbackPath::None) {
                    "query.execution.normal"
                } else {
                    "query.execution.fallback"
                }
            }
            DegradedRetrievalMode::EmbedDeferred => "query.execution.embed_deferred",
            DegradedRetrievalMode::LexicalOnly => "query.execution.lexical_only",
            DegradedRetrievalMode::MetadataOnly => "query.execution.metadata_only",
            DegradedRetrievalMode::Paused => "query.execution.paused",
        };

        QueryExecutionPlan {
            mode,
            stages,
            reason_code,
        }
    }

    /// Resolve cancellation behavior while preserving partial-result correctness.
    #[must_use]
    pub const fn cancellation_directive(&self, point: CancellationPoint) -> CancellationDirective {
        match point {
            CancellationPoint::BeforeInitialCandidates => CancellationDirective {
                action: CancellationAction::PropagateCancelledError,
                reason_code: "query.cancel.before_initial_candidates",
            },
            CancellationPoint::DuringInitialFusion => CancellationDirective {
                action: CancellationAction::EmitPartiallyFusedResults,
                reason_code: "query.cancel.during_initial_fusion",
            },
            CancellationPoint::AfterInitialResults => CancellationDirective {
                action: CancellationAction::EmitInitialResults,
                reason_code: "query.cancel.after_initial_results",
            },
            CancellationPoint::DuringRefinement => CancellationDirective {
                action: CancellationAction::EmitInitialResults,
                reason_code: "query.cancel.during_refinement",
            },
        }
    }

    /// Fuse lexical + semantic rankings with deterministic tie-break ordering.
    #[must_use]
    pub fn fuse_rankings(
        &self,
        lexical: &[LexicalCandidate],
        semantic: &[SemanticCandidate],
        limit: usize,
        offset: usize,
    ) -> Vec<FusedCandidate> {
        self.fuse_rankings_with_priors(
            lexical,
            semantic,
            limit,
            offset,
            &HashMap::new(),
            RankingPriorTuning::default(),
        )
    }

    /// Fuse lexical + semantic rankings and apply optional prior boosts.
    ///
    /// Prior boosts are additive and bounded by `tuning.max_total_boost`.
    /// Deterministic tie-break ordering is always preserved.
    #[must_use]
    pub fn fuse_rankings_with_priors(
        &self,
        lexical: &[LexicalCandidate],
        semantic: &[SemanticCandidate],
        limit: usize,
        offset: usize,
        prior_signals: &HashMap<String, RankingPriorSignals>,
        tuning: RankingPriorTuning,
    ) -> Vec<FusedCandidate> {
        let k = self.fusion_policy.effective_k();
        let tuning = tuning.normalized();
        let mut merged: HashMap<String, FusedCandidate> =
            HashMap::with_capacity(lexical.len() + semantic.len());

        let mut lexical_ranked: Vec<&LexicalCandidate> = lexical.iter().collect();
        lexical_ranked.sort_by(|left, right| {
            sanitize_score(right.score)
                .total_cmp(&sanitize_score(left.score))
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });

        for (rank, candidate) in lexical_ranked.iter().enumerate() {
            let contribution = rrf_contribution(k, rank);
            let lexical_score = sanitize_score(candidate.score);
            merged
                .entry(candidate.doc_id.clone())
                .and_modify(|hit| {
                    hit.fused_score += contribution;
                    hit.lexical_rank = Some(rank);
                    hit.lexical_score = Some(lexical_score);
                    hit.in_both_sources = true;
                })
                .or_insert_with(|| FusedCandidate {
                    doc_id: candidate.doc_id.clone(),
                    fused_score: contribution,
                    prior_boost: 0.0,
                    lexical_rank: Some(rank),
                    semantic_rank: None,
                    lexical_score: Some(lexical_score),
                    semantic_score: None,
                    in_both_sources: false,
                });
        }

        let mut semantic_ranked: Vec<&SemanticCandidate> = semantic.iter().collect();
        semantic_ranked.sort_by(|left, right| {
            sanitize_score(right.score)
                .total_cmp(&sanitize_score(left.score))
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });

        for (rank, candidate) in semantic_ranked.iter().enumerate() {
            let contribution = rrf_contribution(k, rank);
            let semantic_score = sanitize_score(candidate.score);
            merged
                .entry(candidate.doc_id.clone())
                .and_modify(|hit| {
                    hit.fused_score += contribution;
                    hit.semantic_rank = Some(rank);
                    hit.semantic_score = Some(semantic_score);
                    hit.in_both_sources = true;
                })
                .or_insert_with(|| FusedCandidate {
                    doc_id: candidate.doc_id.clone(),
                    fused_score: contribution,
                    prior_boost: 0.0,
                    lexical_rank: None,
                    semantic_rank: Some(rank),
                    lexical_score: None,
                    semantic_score: Some(semantic_score),
                    in_both_sources: false,
                });
        }

        let mut fused: Vec<FusedCandidate> = merged
            .into_values()
            .map(|mut candidate| {
                let signals = prior_signals
                    .get(&candidate.doc_id)
                    .copied()
                    .unwrap_or_default();
                let prior_boost = tuning.boost_for(signals);
                candidate.prior_boost = prior_boost;
                candidate.fused_score += prior_boost;
                candidate
            })
            .collect();
        fused.sort_by(fused_cmp);

        fused.into_iter().skip(offset).take(limit).collect()
    }
}

#[must_use]
const fn mode_from_pressure(pressure_state: PressureState) -> DegradedRetrievalMode {
    match pressure_state {
        PressureState::Normal => DegradedRetrievalMode::Normal,
        PressureState::Constrained => DegradedRetrievalMode::EmbedDeferred,
        PressureState::Degraded => DegradedRetrievalMode::LexicalOnly,
        PressureState::Emergency => DegradedRetrievalMode::MetadataOnly,
    }
}

#[must_use]
const fn target_mode(signals: DegradationSignals) -> DegradedRetrievalMode {
    match signals.pressure_state {
        PressureState::Emergency => {
            if matches!(signals.backpressure_mode, BackpressureMode::Saturated) {
                DegradedRetrievalMode::Paused
            } else {
                DegradedRetrievalMode::MetadataOnly
            }
        }
        PressureState::Degraded => {
            if matches!(signals.backpressure_mode, BackpressureMode::Saturated)
                || signals.quality_circuit_open
            {
                DegradedRetrievalMode::MetadataOnly
            } else {
                DegradedRetrievalMode::LexicalOnly
            }
        }
        PressureState::Constrained => {
            if matches!(signals.backpressure_mode, BackpressureMode::Saturated)
                || signals.quality_circuit_open
            {
                DegradedRetrievalMode::LexicalOnly
            } else {
                DegradedRetrievalMode::EmbedDeferred
            }
        }
        PressureState::Normal => {
            if matches!(signals.backpressure_mode, BackpressureMode::Saturated) {
                DegradedRetrievalMode::LexicalOnly
            } else if matches!(signals.backpressure_mode, BackpressureMode::HighWatermark)
                || signals.quality_circuit_open
            {
                DegradedRetrievalMode::EmbedDeferred
            } else {
                DegradedRetrievalMode::Normal
            }
        }
    }
}

#[must_use]
const fn transition_reason_code(mode: DegradedRetrievalMode) -> &'static str {
    match mode {
        DegradedRetrievalMode::Normal => "degrade.transition.recovered",
        DegradedRetrievalMode::EmbedDeferred => "degrade.transition.embed_deferred",
        DegradedRetrievalMode::LexicalOnly => "degrade.transition.lexical_only",
        DegradedRetrievalMode::MetadataOnly => "degrade.transition.metadata_only",
        DegradedRetrievalMode::Paused => "degrade.transition.pause",
    }
}

#[must_use]
const fn status_for_mode(mode: DegradedRetrievalMode) -> DegradationStatus {
    match mode {
        DegradedRetrievalMode::Normal => DegradationStatus {
            banner: "Normal operation",
            controls_hint: "override:auto|embed_deferred|lexical_only|metadata_only|pause",
        },
        DegradedRetrievalMode::EmbedDeferred => DegradationStatus {
            banner: "Constrained: quality embedding deferred",
            controls_hint: "override:auto|lexical_only|metadata_only|pause|normal",
        },
        DegradedRetrievalMode::LexicalOnly => DegradationStatus {
            banner: "Degraded: lexical-only retrieval active",
            controls_hint: "override:auto|metadata_only|pause|normal",
        },
        DegradedRetrievalMode::MetadataOnly => DegradationStatus {
            banner: "Emergency: metadata-only fallback active",
            controls_hint: "override:auto|pause|normal",
        },
        DegradedRetrievalMode::Paused => DegradationStatus {
            banner: "Paused: query pipeline writes paused",
            controls_hint: "override:auto|normal",
        },
    }
}

#[must_use]
const fn stage_timeout(
    total_budget_ms: u64,
    numerator: u64,
    denominator: u64,
    min_ms: u64,
    max_ms: u64,
) -> u64 {
    let scaled = total_budget_ms
        .saturating_mul(numerator)
        .saturating_add(denominator.saturating_sub(1))
        / denominator;
    let at_least_min = if scaled < min_ms { min_ms } else { scaled };
    if at_least_min > max_ms {
        max_ms
    } else {
        at_least_min
    }
}

#[must_use]
fn rrf_contribution(k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (k + f64::from(rank_u32) + 1.0)
}

#[must_use]
const fn sanitize_score(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

#[must_use]
fn option_score(score: Option<f32>) -> f32 {
    score.map_or(f32::NEG_INFINITY, sanitize_score)
}

#[must_use]
fn sanitize_non_negative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

#[must_use]
fn sanitize_signal(value: Option<f64>) -> f64 {
    value.map_or(0.0, |v| sanitize_non_negative(v).clamp(0.0, 1.0))
}

#[must_use]
fn sanitize_max_prior_boost(value: f64) -> f64 {
    let sanitized = sanitize_non_negative(value);
    if sanitized > 0.0 {
        sanitized.min(0.05)
    } else {
        0.01
    }
}

fn fused_cmp(left: &FusedCandidate, right: &FusedCandidate) -> Ordering {
    right
        .fused_score
        .total_cmp(&left.fused_score)
        .then_with(|| right.in_both_sources.cmp(&left.in_both_sources))
        .then_with(|| {
            option_score(right.lexical_score).total_cmp(&option_score(left.lexical_score))
        })
        .then_with(|| {
            option_score(right.semantic_score).total_cmp(&option_score(left.semantic_score))
        })
        .then_with(|| left.doc_id.cmp(&right.doc_id))
}

#[cfg(test)]
mod tests {
    use super::{
        CancellationAction, CancellationPoint, DegradationOverride, DegradationPolicy,
        DegradationSignals, DegradationStateMachine, DegradedRetrievalMode, LexicalCandidate,
        QueryExecutionOrchestrator, RankingPriorSignals, RankingPriorTuning, RetrievalStage,
        SemanticCandidate,
    };
    use crate::config::{FsfsConfig, PressureProfile};
    use crate::orchestration::BackpressureMode;
    use crate::pressure::PressureState;
    use crate::query_planning::QueryPlanner;
    use std::collections::HashMap;

    fn stage_names(plan: &super::QueryExecutionPlan) -> Vec<RetrievalStage> {
        plan.stages.iter().map(|stage| stage.stage).collect()
    }

    #[test]
    fn plan_includes_quality_and_rerank_for_normal_mode() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let intent = planner.classify_intent("how does fusion ranking work");
        let budget = planner.budget_for_decision(&intent, Some(10));
        let orchestrator = QueryExecutionOrchestrator::default();

        let plan = orchestrator.plan(&intent, &budget, PressureState::Normal);
        let names = stage_names(&plan);

        assert_eq!(plan.mode, DegradedRetrievalMode::Normal);
        assert!(names.contains(&RetrievalStage::QualitySemanticRetrieve));
        assert!(names.contains(&RetrievalStage::Rerank));
        assert_eq!(plan.reason_code, "query.execution.normal");
    }

    #[test]
    fn plan_skips_refinement_when_lexical_only() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let intent = planner.classify_intent("how does fusion ranking work");
        let budget = planner.budget_for_decision(&intent, Some(10));
        let orchestrator = QueryExecutionOrchestrator::default();

        let plan = orchestrator.plan(&intent, &budget, PressureState::Degraded);
        let names = stage_names(&plan);

        assert_eq!(plan.mode, DegradedRetrievalMode::LexicalOnly);
        assert!(!names.contains(&RetrievalStage::QualitySemanticRetrieve));
        assert!(!names.contains(&RetrievalStage::Rerank));
        assert_eq!(plan.reason_code, "query.execution.lexical_only");
    }

    #[test]
    fn plan_metadata_only_and_paused_modes_have_expected_stages() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let intent = planner.classify_intent("status");
        let budget = planner.budget_for_decision(&intent, Some(10));
        let orchestrator = QueryExecutionOrchestrator::default();

        let metadata_only =
            orchestrator.plan_with_mode(&intent, &budget, DegradedRetrievalMode::MetadataOnly);
        let metadata_stages = stage_names(&metadata_only);
        assert!(metadata_stages.contains(&RetrievalStage::MetadataFallbackRetrieve));
        assert!(!metadata_stages.contains(&RetrievalStage::LexicalRetrieve));
        assert_eq!(metadata_only.reason_code, "query.execution.metadata_only");

        let paused = orchestrator.plan_with_mode(&intent, &budget, DegradedRetrievalMode::Paused);
        let paused_stages = stage_names(&paused);
        assert_eq!(
            paused_stages,
            vec![RetrievalStage::Canonicalize, RetrievalStage::ClassifyIntent]
        );
        assert_eq!(paused.reason_code, "query.execution.paused");
    }

    #[test]
    fn malformed_query_uses_lexical_only_plan() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let intent = planner.classify_intent("find\u{0007}secret");
        let budget = planner.budget_for_decision(&intent, Some(8));
        let orchestrator = QueryExecutionOrchestrator::default();

        let plan = orchestrator.plan(&intent, &budget, PressureState::Normal);
        let names = stage_names(&plan);

        assert!(names.contains(&RetrievalStage::LexicalRetrieve));
        assert!(!names.contains(&RetrievalStage::FastSemanticRetrieve));
        assert!(!names.contains(&RetrievalStage::QualitySemanticRetrieve));
        assert_eq!(plan.reason_code, "query.execution.fallback");
    }

    #[test]
    fn cancellation_directives_preserve_partial_result_correctness() {
        let orchestrator = QueryExecutionOrchestrator::default();

        let before_initial =
            orchestrator.cancellation_directive(CancellationPoint::BeforeInitialCandidates);
        assert_eq!(
            before_initial.action,
            CancellationAction::PropagateCancelledError
        );

        let during_fusion =
            orchestrator.cancellation_directive(CancellationPoint::DuringInitialFusion);
        assert_eq!(
            during_fusion.action,
            CancellationAction::EmitPartiallyFusedResults
        );

        let during_refinement =
            orchestrator.cancellation_directive(CancellationPoint::DuringRefinement);
        assert_eq!(
            during_refinement.action,
            CancellationAction::EmitInitialResults
        );
    }

    #[test]
    fn fusion_tie_breaker_is_deterministic() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", 10.0),
            LexicalCandidate::new("doc-b", 9.5),
        ];
        let semantic = vec![
            SemanticCandidate::new("doc-b", 0.8),
            SemanticCandidate::new("doc-c", 0.8),
        ];

        let fused = orchestrator.fuse_rankings(&lexical, &semantic, 10, 0);
        let ids: Vec<&str> = fused.iter().map(|hit| hit.doc_id.as_str()).collect();

        assert_eq!(ids, vec!["doc-b", "doc-a", "doc-c"]);
    }

    #[test]
    fn fusion_priors_promote_candidates_without_breaking_determinism() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", 10.0),
            LexicalCandidate::new("doc-b", 10.0),
        ];
        let semantic = vec![
            SemanticCandidate::new("doc-a", 0.8),
            SemanticCandidate::new("doc-b", 0.8),
        ];
        let mut priors = HashMap::new();
        priors.insert(
            "doc-b".to_owned(),
            RankingPriorSignals::new(Some(1.0), Some(0.5), Some(0.25)),
        );
        let tuning = RankingPriorTuning::for_profile(PressureProfile::Performance);

        let fused =
            orchestrator.fuse_rankings_with_priors(&lexical, &semantic, 10, 0, &priors, tuning);
        let ids: Vec<&str> = fused.iter().map(|hit| hit.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["doc-b", "doc-a"]);
        assert!(fused[0].prior_boost > 0.0);
        assert!(fused[1].prior_boost.abs() < f64::EPSILON);
    }

    #[test]
    fn prior_boost_is_clamped_and_sanitized() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![LexicalCandidate::new("doc-a", 1.0)];
        let semantic = vec![SemanticCandidate::new("doc-a", 1.0)];
        let mut priors = HashMap::new();
        priors.insert(
            "doc-a".to_owned(),
            RankingPriorSignals::new(Some(f64::NAN), Some(99.0), Some(99.0)),
        );
        let tuning = RankingPriorTuning {
            enabled: true,
            max_total_boost: 0.002,
            ..RankingPriorTuning::for_profile(PressureProfile::Performance)
        };

        let fused =
            orchestrator.fuse_rankings_with_priors(&lexical, &semantic, 10, 0, &priors, tuning);
        assert_eq!(fused.len(), 1);
        assert!(fused[0].prior_boost.is_finite());
        assert!(fused[0].prior_boost > 0.0);
        assert!(fused[0].prior_boost <= 0.002 + 1e-12);
    }

    #[test]
    fn prior_profile_defaults_are_compatible_with_pressure_modes() {
        let strict = RankingPriorTuning::for_profile(PressureProfile::Strict).normalized();
        let performance =
            RankingPriorTuning::for_profile(PressureProfile::Performance).normalized();
        let degraded = RankingPriorTuning::for_profile(PressureProfile::Degraded).normalized();

        assert!(strict.max_total_boost < performance.max_total_boost);
        assert!(degraded.max_total_boost <= performance.max_total_boost);
        assert!(strict.weights.recency <= performance.weights.recency);
        assert!(degraded.weights.path <= performance.weights.path);
    }

    #[test]
    fn fusion_handles_nan_scores_without_panicking() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", f32::NAN),
            LexicalCandidate::new("doc-b", 1.0),
        ];
        let semantic = vec![];

        let fused = orchestrator.fuse_rankings(&lexical, &semantic, 10, 0);
        let ids: Vec<&str> = fused.iter().map(|hit| hit.doc_id.as_str()).collect();

        assert_eq!(ids, vec!["doc-b", "doc-a"]);
    }

    #[test]
    fn degradation_ladder_escalates_stepwise_to_pause() {
        let mut machine = DegradationStateMachine::new(DegradationPolicy {
            escalate_after: 2,
            recover_after: 2,
        });
        let emergency = DegradationSignals {
            pressure_state: PressureState::Emergency,
            backpressure_mode: BackpressureMode::Saturated,
            quality_circuit_open: true,
        };

        for _ in 0..2 {
            let _ = machine.observe(emergency);
        }
        assert_eq!(machine.mode(), DegradedRetrievalMode::EmbedDeferred);

        for _ in 0..2 {
            let _ = machine.observe(emergency);
        }
        assert_eq!(machine.mode(), DegradedRetrievalMode::LexicalOnly);

        for _ in 0..2 {
            let _ = machine.observe(emergency);
        }
        assert_eq!(machine.mode(), DegradedRetrievalMode::MetadataOnly);

        for _ in 0..2 {
            let _ = machine.observe(emergency);
        }
        assert_eq!(machine.mode(), DegradedRetrievalMode::Paused);
    }

    #[test]
    fn degradation_recovery_uses_exit_threshold_and_override_controls() {
        let mut machine = DegradationStateMachine::new(DegradationPolicy {
            escalate_after: 1,
            recover_after: 2,
        });
        machine.set_override(DegradationOverride::ForcePause);
        let forced = machine.observe(DegradationSignals::default());
        assert!(forced.changed);
        assert_eq!(forced.to, DegradedRetrievalMode::Paused);
        assert_eq!(forced.reason_code, "degrade.override.applied");
        assert_eq!(forced.status.banner, "Paused: query pipeline writes paused");

        machine.set_override(DegradationOverride::Auto);
        let normal = DegradationSignals::default();

        let hold = machine.observe(normal);
        assert!(!hold.changed);
        assert_eq!(hold.reason_code, "degrade.transition.hysteresis_hold");
        let recovered = machine.observe(normal);
        assert!(recovered.changed);
        assert_eq!(recovered.to, DegradedRetrievalMode::MetadataOnly);

        let _ = machine.observe(normal);
        let second_recovery = machine.observe(normal);
        assert!(second_recovery.changed);
        assert_eq!(second_recovery.to, DegradedRetrievalMode::LexicalOnly);
    }

    #[test]
    fn profile_policies_have_distinct_recovery_requirements() {
        let strict = DegradationPolicy::for_profile(PressureProfile::Strict);
        let performance = DegradationPolicy::for_profile(PressureProfile::Performance);
        let degraded = DegradationPolicy::for_profile(PressureProfile::Degraded);

        assert!(strict.escalate_after <= performance.escalate_after);
        assert!(degraded.recover_after > performance.recover_after);
    }
}
