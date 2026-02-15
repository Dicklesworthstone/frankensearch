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
use crate::output_schema::{SearchHitPayload, SearchOutputPhase, SearchPayload};
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

impl DegradationTransition {
    #[must_use]
    pub const fn transition_context(self) -> &'static str {
        transition_context_for_transition(self)
    }

    #[must_use]
    pub const fn manual_intervention(self) -> bool {
        !matches!(self.override_mode, DegradationOverride::Auto)
    }

    #[must_use]
    pub const fn override_guardrail(self) -> &'static str {
        override_guardrail_for_mode(self.to)
    }
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

    /// Build search output payload rows from fused candidate rankings.
    #[must_use]
    pub fn build_search_payload(
        &self,
        query: &str,
        phase: SearchOutputPhase,
        fused: &[FusedCandidate],
        snippets_by_doc: &HashMap<String, String>,
    ) -> SearchPayload {
        let hits = fused
            .iter()
            .enumerate()
            .map(|(idx, candidate)| SearchHitPayload {
                rank: idx.saturating_add(1),
                path: candidate.doc_id.clone(),
                score: sanitize_fused_score(candidate.fused_score),
                snippet: snippets_by_doc
                    .get(&candidate.doc_id)
                    .cloned()
                    .filter(|snippet| !snippet.trim().is_empty()),
                lexical_rank: candidate.lexical_rank,
                semantic_rank: candidate.semantic_rank,
                in_both_sources: candidate.in_both_sources,
            })
            .collect();

        SearchPayload::new(query, phase, fused.len(), hits)
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
const fn transition_context_for_transition(transition: DegradationTransition) -> &'static str {
    if !transition.changed {
        if matches!(transition.override_mode, DegradationOverride::Auto) {
            "state_stable"
        } else {
            "manual_override_hold"
        }
    } else if !matches!(transition.override_mode, DegradationOverride::Auto) {
        "manual_override_transition"
    } else if transition.to.severity() > transition.from.severity() {
        "pressure_escalation"
    } else {
        "pressure_recovery"
    }
}

#[must_use]
const fn override_guardrail_for_mode(mode: DegradedRetrievalMode) -> &'static str {
    match mode {
        DegradedRetrievalMode::Normal => {
            "manual overrides are optional; controller can still escalate under pressure"
        }
        DegradedRetrievalMode::EmbedDeferred => {
            "resume to normal only when pressure recovers or operator forces normal"
        }
        DegradedRetrievalMode::LexicalOnly => {
            "metadata-only and pause remain available as emergency guardrails"
        }
        DegradedRetrievalMode::MetadataOnly => {
            "only pause or normal-recovery overrides are allowed at this severity"
        }
        DegradedRetrievalMode::Paused => {
            "writes remain paused until operator clears override or forces recovery"
        }
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
const fn sanitize_fused_score(score: f64) -> f64 {
    if score.is_finite() { score } else { 0.0 }
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
        DegradationSignals, DegradationStateMachine, DegradationTransition, DegradedRetrievalMode,
        LexicalCandidate, QueryExecutionOrchestrator, RankingPriorSignals, RankingPriorTuning,
        RetrievalStage, SemanticCandidate, status_for_mode,
    };
    use crate::config::{FsfsConfig, PressureProfile};
    use crate::orchestration::BackpressureMode;
    use crate::output_schema::SearchOutputPhase;
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
    fn build_search_payload_maps_ranks_scores_and_snippets() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let fused = vec![
            super::FusedCandidate {
                doc_id: "src/auth.rs".to_owned(),
                fused_score: 0.923,
                prior_boost: 0.0,
                lexical_rank: Some(0),
                semantic_rank: Some(1),
                lexical_score: Some(1.0),
                semantic_score: Some(0.9),
                in_both_sources: true,
            },
            super::FusedCandidate {
                doc_id: "docs/auth.md".to_owned(),
                fused_score: f64::NAN,
                prior_boost: 0.0,
                lexical_rank: Some(3),
                semantic_rank: None,
                lexical_score: Some(0.4),
                semantic_score: None,
                in_both_sources: false,
            },
        ];
        let snippets = HashMap::from([
            (
                "src/auth.rs".to_owned(),
                "fn authenticate(token: &str) -> bool".to_owned(),
            ),
            ("docs/auth.md".to_owned(), "   ".to_owned()),
        ]);

        let payload = orchestrator.build_search_payload(
            "how auth works",
            SearchOutputPhase::Initial,
            &fused,
            &snippets,
        );
        assert_eq!(payload.query, "how auth works");
        assert_eq!(payload.phase, SearchOutputPhase::Initial);
        assert_eq!(payload.total_candidates, 2);
        assert_eq!(payload.returned_hits, 2);
        assert_eq!(payload.hits[0].rank, 1);
        assert_eq!(payload.hits[0].path, "src/auth.rs");
        assert!((payload.hits[0].score - 0.923).abs() < f64::EPSILON);
        assert_eq!(
            payload.hits[0].snippet.as_deref(),
            Some("fn authenticate(token: &str) -> bool")
        );
        assert_eq!(payload.hits[1].rank, 2);
        assert!(payload.hits[1].score.abs() < f64::EPSILON);
        assert!(payload.hits[1].snippet.is_none());
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

    #[test]
    fn transition_context_distinguishes_escalation_recovery_and_override() {
        let escalation = DegradationTransition {
            from: DegradedRetrievalMode::Normal,
            to: DegradedRetrievalMode::EmbedDeferred,
            changed: true,
            reason_code: "degrade.transition.embed_deferred",
            status: status_for_mode(DegradedRetrievalMode::EmbedDeferred),
            override_mode: DegradationOverride::Auto,
        };
        assert_eq!(escalation.transition_context(), "pressure_escalation");
        assert!(!escalation.manual_intervention());

        let recovery = DegradationTransition {
            from: DegradedRetrievalMode::LexicalOnly,
            to: DegradedRetrievalMode::EmbedDeferred,
            changed: true,
            reason_code: "degrade.transition.recovered",
            status: status_for_mode(DegradedRetrievalMode::EmbedDeferred),
            override_mode: DegradationOverride::Auto,
        };
        assert_eq!(recovery.transition_context(), "pressure_recovery");
        assert!(!recovery.manual_intervention());

        let manual = DegradationTransition {
            from: DegradedRetrievalMode::LexicalOnly,
            to: DegradedRetrievalMode::Paused,
            changed: true,
            reason_code: "degrade.override.applied",
            status: status_for_mode(DegradedRetrievalMode::Paused),
            override_mode: DegradationOverride::ForcePause,
        };
        assert_eq!(manual.transition_context(), "manual_override_transition");
        assert!(manual.manual_intervention());
    }

    #[test]
    fn override_guardrail_hint_tracks_target_mode() {
        let paused = DegradationTransition {
            from: DegradedRetrievalMode::MetadataOnly,
            to: DegradedRetrievalMode::Paused,
            changed: true,
            reason_code: "degrade.transition.pause",
            status: status_for_mode(DegradedRetrievalMode::Paused),
            override_mode: DegradationOverride::Auto,
        };
        assert!(
            paused.override_guardrail().contains("writes remain paused"),
            "guardrail should explain paused safety behavior"
        );

        let normal = DegradationTransition {
            from: DegradedRetrievalMode::EmbedDeferred,
            to: DegradedRetrievalMode::Normal,
            changed: true,
            reason_code: "degrade.transition.recovered",
            status: status_for_mode(DegradedRetrievalMode::Normal),
            override_mode: DegradationOverride::Auto,
        };
        assert!(
            normal
                .override_guardrail()
                .contains("controller can still escalate"),
            "normal mode guardrail should preserve safety semantics"
        );
    }

    // ─── bd-olv6 tests begin ───

    // --- DegradedRetrievalMode severity ---

    #[test]
    fn degraded_mode_severity_monotonic() {
        use super::DegradedRetrievalMode::*;
        let modes = [Normal, EmbedDeferred, LexicalOnly, MetadataOnly, Paused];
        for pair in modes.windows(2) {
            assert!(
                pair[0].severity() < pair[1].severity(),
                "{:?} should be less severe than {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn degraded_mode_next_more_restrictive() {
        use super::DegradedRetrievalMode::*;
        assert_eq!(Normal.next_more_restrictive(), EmbedDeferred);
        assert_eq!(EmbedDeferred.next_more_restrictive(), LexicalOnly);
        assert_eq!(LexicalOnly.next_more_restrictive(), MetadataOnly);
        assert_eq!(MetadataOnly.next_more_restrictive(), Paused);
        assert_eq!(Paused.next_more_restrictive(), Paused);
    }

    #[test]
    fn degraded_mode_next_less_restrictive() {
        use super::DegradedRetrievalMode::*;
        assert_eq!(Paused.next_less_restrictive(), MetadataOnly);
        assert_eq!(MetadataOnly.next_less_restrictive(), LexicalOnly);
        assert_eq!(LexicalOnly.next_less_restrictive(), EmbedDeferred);
        assert_eq!(EmbedDeferred.next_less_restrictive(), Normal);
        assert_eq!(Normal.next_less_restrictive(), Normal);
    }

    // --- DegradationOverride forced_mode ---

    #[test]
    fn degradation_override_auto_returns_none() {
        assert_eq!(DegradationOverride::Auto.forced_mode(), None);
    }

    #[test]
    fn degradation_override_force_variants() {
        assert_eq!(
            DegradationOverride::ForceNormal.forced_mode(),
            Some(DegradedRetrievalMode::Normal)
        );
        assert_eq!(
            DegradationOverride::ForceEmbedDeferred.forced_mode(),
            Some(DegradedRetrievalMode::EmbedDeferred)
        );
        assert_eq!(
            DegradationOverride::ForceLexicalOnly.forced_mode(),
            Some(DegradedRetrievalMode::LexicalOnly)
        );
        assert_eq!(
            DegradationOverride::ForceMetadataOnly.forced_mode(),
            Some(DegradedRetrievalMode::MetadataOnly)
        );
        assert_eq!(
            DegradationOverride::ForcePause.forced_mode(),
            Some(DegradedRetrievalMode::Paused)
        );
    }

    // --- DegradationSignals ---

    #[test]
    fn degradation_signals_default() {
        let signals = DegradationSignals::default();
        assert_eq!(signals.pressure_state, PressureState::Normal);
        assert_eq!(signals.backpressure_mode, BackpressureMode::Normal);
        assert!(!signals.quality_circuit_open);
    }

    // --- DegradationPolicy ---

    #[test]
    fn degradation_policy_default_matches_performance() {
        let policy = super::DegradationPolicy::default();
        let perf = super::DegradationPolicy::for_profile(PressureProfile::Performance);
        assert_eq!(policy, perf);
    }

    #[test]
    fn degradation_policy_normalized_clamps_zero() {
        let policy = DegradationPolicy {
            escalate_after: 0,
            recover_after: 0,
        };
        let normalized = policy.normalized();
        assert_eq!(normalized.escalate_after, 1);
        assert_eq!(normalized.recover_after, 1);
    }

    #[test]
    fn degradation_policy_normalized_preserves_nonzero() {
        let policy = DegradationPolicy {
            escalate_after: 5,
            recover_after: 3,
        };
        let normalized = policy.normalized();
        assert_eq!(normalized.escalate_after, 5);
        assert_eq!(normalized.recover_after, 3);
    }

    // --- mode_from_pressure ---

    #[test]
    fn mode_from_pressure_mapping() {
        assert_eq!(
            super::mode_from_pressure(PressureState::Normal),
            DegradedRetrievalMode::Normal
        );
        assert_eq!(
            super::mode_from_pressure(PressureState::Constrained),
            DegradedRetrievalMode::EmbedDeferred
        );
        assert_eq!(
            super::mode_from_pressure(PressureState::Degraded),
            DegradedRetrievalMode::LexicalOnly
        );
        assert_eq!(
            super::mode_from_pressure(PressureState::Emergency),
            DegradedRetrievalMode::MetadataOnly
        );
    }

    // --- target_mode ---

    #[test]
    fn target_mode_normal_no_backpressure() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Normal,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: false,
        };
        assert_eq!(super::target_mode(signals), DegradedRetrievalMode::Normal);
    }

    #[test]
    fn target_mode_normal_saturated() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Normal,
            backpressure_mode: BackpressureMode::Saturated,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::LexicalOnly
        );
    }

    #[test]
    fn target_mode_normal_high_watermark() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Normal,
            backpressure_mode: BackpressureMode::HighWatermark,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::EmbedDeferred
        );
    }

    #[test]
    fn target_mode_normal_circuit_open() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Normal,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: true,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::EmbedDeferred
        );
    }

    #[test]
    fn target_mode_constrained_normal_backpressure() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Constrained,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::EmbedDeferred
        );
    }

    #[test]
    fn target_mode_constrained_saturated() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Constrained,
            backpressure_mode: BackpressureMode::Saturated,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::LexicalOnly
        );
    }

    #[test]
    fn target_mode_degraded_normal_backpressure() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Degraded,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::LexicalOnly
        );
    }

    #[test]
    fn target_mode_degraded_circuit_open() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Degraded,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: true,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::MetadataOnly
        );
    }

    #[test]
    fn target_mode_emergency_saturated() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Emergency,
            backpressure_mode: BackpressureMode::Saturated,
            quality_circuit_open: false,
        };
        assert_eq!(super::target_mode(signals), DegradedRetrievalMode::Paused);
    }

    #[test]
    fn target_mode_emergency_not_saturated() {
        let signals = DegradationSignals {
            pressure_state: PressureState::Emergency,
            backpressure_mode: BackpressureMode::Normal,
            quality_circuit_open: false,
        };
        assert_eq!(
            super::target_mode(signals),
            DegradedRetrievalMode::MetadataOnly
        );
    }

    // --- transition_reason_code ---

    #[test]
    fn transition_reason_code_all_modes() {
        assert_eq!(
            super::transition_reason_code(DegradedRetrievalMode::Normal),
            "degrade.transition.recovered"
        );
        assert_eq!(
            super::transition_reason_code(DegradedRetrievalMode::EmbedDeferred),
            "degrade.transition.embed_deferred"
        );
        assert_eq!(
            super::transition_reason_code(DegradedRetrievalMode::LexicalOnly),
            "degrade.transition.lexical_only"
        );
        assert_eq!(
            super::transition_reason_code(DegradedRetrievalMode::MetadataOnly),
            "degrade.transition.metadata_only"
        );
        assert_eq!(
            super::transition_reason_code(DegradedRetrievalMode::Paused),
            "degrade.transition.pause"
        );
    }

    // --- status_for_mode ---

    #[test]
    fn status_for_mode_all_variants() {
        let normal = status_for_mode(DegradedRetrievalMode::Normal);
        assert_eq!(normal.banner, "Normal operation");

        let embed = status_for_mode(DegradedRetrievalMode::EmbedDeferred);
        assert!(embed.banner.contains("quality embedding deferred"));

        let lexical = status_for_mode(DegradedRetrievalMode::LexicalOnly);
        assert!(lexical.banner.contains("lexical-only"));

        let metadata = status_for_mode(DegradedRetrievalMode::MetadataOnly);
        assert!(metadata.banner.contains("metadata-only"));

        let paused = status_for_mode(DegradedRetrievalMode::Paused);
        assert!(paused.banner.contains("Paused"));
    }

    // --- override_guardrail_for_mode all variants ---

    #[test]
    fn override_guardrail_all_modes() {
        use super::override_guardrail_for_mode;
        assert!(override_guardrail_for_mode(DegradedRetrievalMode::Normal).contains("optional"));
        assert!(
            override_guardrail_for_mode(DegradedRetrievalMode::EmbedDeferred).contains("resume")
        );
        assert!(
            override_guardrail_for_mode(DegradedRetrievalMode::LexicalOnly).contains("emergency")
        );
        assert!(override_guardrail_for_mode(DegradedRetrievalMode::MetadataOnly).contains("pause"));
        assert!(
            override_guardrail_for_mode(DegradedRetrievalMode::Paused).contains("writes remain")
        );
    }

    // --- stage_timeout ---

    #[test]
    fn stage_timeout_clamps_to_min() {
        // 10 * 1/3 = ~3, but min is 20
        assert_eq!(super::stage_timeout(10, 1, 3, 20, 120), 20);
    }

    #[test]
    fn stage_timeout_clamps_to_max() {
        // 1000 * 1/1 = 1000, but max is 120
        assert_eq!(super::stage_timeout(1000, 1, 1, 20, 120), 120);
    }

    #[test]
    fn stage_timeout_within_bounds() {
        // 300 * 1/3 = 100, within [20, 120]
        assert_eq!(super::stage_timeout(300, 1, 3, 20, 120), 100);
    }

    // --- rrf_contribution ---

    #[test]
    fn rrf_contribution_rank_0() {
        let contrib = super::rrf_contribution(60.0, 0);
        // 1 / (60 + 0 + 1) = 1/61
        assert!((contrib - 1.0 / 61.0).abs() < 1e-12);
    }

    #[test]
    fn rrf_contribution_rank_1() {
        let contrib = super::rrf_contribution(60.0, 1);
        // 1 / (60 + 1 + 1) = 1/62
        assert!((contrib - 1.0 / 62.0).abs() < 1e-12);
    }

    #[test]
    fn rrf_contribution_decreases_with_rank() {
        let c0 = super::rrf_contribution(60.0, 0);
        let c1 = super::rrf_contribution(60.0, 1);
        let c10 = super::rrf_contribution(60.0, 10);
        assert!(c0 > c1);
        assert!(c1 > c10);
    }

    // --- sanitize_score ---

    #[test]
    fn sanitize_score_nan_becomes_neg_infinity() {
        let value = super::sanitize_score(f32::NAN);
        assert!(value.is_infinite() && value.is_sign_negative());
    }

    #[test]
    fn sanitize_score_normal_value_unchanged() {
        assert!((super::sanitize_score(1.5) - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn sanitize_score_infinity_unchanged() {
        let value = super::sanitize_score(f32::INFINITY);
        assert!(value.is_infinite() && value.is_sign_positive());
    }

    // --- sanitize_fused_score ---

    #[test]
    fn sanitize_fused_score_nan_becomes_zero() {
        assert!(super::sanitize_fused_score(f64::NAN).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_fused_score_infinity_becomes_zero() {
        assert!(super::sanitize_fused_score(f64::INFINITY).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_fused_score_finite_unchanged() {
        assert!((super::sanitize_fused_score(0.5) - 0.5).abs() < f64::EPSILON);
    }

    // --- sanitize_non_negative ---

    #[test]
    fn sanitize_non_negative_negative_becomes_zero() {
        assert!(super::sanitize_non_negative(-1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_non_negative_nan_becomes_zero() {
        assert!(super::sanitize_non_negative(f64::NAN).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_non_negative_positive_unchanged() {
        assert!((super::sanitize_non_negative(0.5) - 0.5).abs() < f64::EPSILON);
    }

    // --- sanitize_signal ---

    #[test]
    fn sanitize_signal_none_becomes_zero() {
        assert!(super::sanitize_signal(None).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_signal_nan_becomes_zero() {
        assert!(super::sanitize_signal(Some(f64::NAN)).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_signal_clamped_above_1() {
        assert!((super::sanitize_signal(Some(2.0)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_signal_clamped_below_0() {
        assert!(super::sanitize_signal(Some(-0.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_signal_valid_value() {
        assert!((super::sanitize_signal(Some(0.75)) - 0.75).abs() < f64::EPSILON);
    }

    // --- sanitize_max_prior_boost ---

    #[test]
    fn sanitize_max_prior_boost_zero_defaults_to_001() {
        assert!((super::sanitize_max_prior_boost(0.0) - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_max_prior_boost_negative_defaults_to_001() {
        assert!((super::sanitize_max_prior_boost(-1.0) - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_max_prior_boost_capped_at_005() {
        assert!((super::sanitize_max_prior_boost(1.0) - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_max_prior_boost_valid_value() {
        assert!((super::sanitize_max_prior_boost(0.02) - 0.02).abs() < f64::EPSILON);
    }

    // --- option_score ---

    #[test]
    fn option_score_none_is_neg_infinity() {
        let value = super::option_score(None);
        assert!(value.is_infinite() && value.is_sign_negative());
    }

    #[test]
    fn option_score_some_nan_is_neg_infinity() {
        let value = super::option_score(Some(f32::NAN));
        assert!(value.is_infinite() && value.is_sign_negative());
    }

    #[test]
    fn option_score_some_value() {
        assert!((super::option_score(Some(1.5)) - 1.5).abs() < f32::EPSILON);
    }

    // --- FusionPolicy ---

    #[test]
    fn fusion_policy_default_rrf_k() {
        let policy = super::FusionPolicy::default();
        assert!((policy.rrf_k - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fusion_policy_effective_k_valid() {
        let policy = super::FusionPolicy { rrf_k: 42.0 };
        assert!((policy.effective_k() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fusion_policy_effective_k_nan_fallback() {
        let policy = super::FusionPolicy { rrf_k: f64::NAN };
        assert!((policy.effective_k() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fusion_policy_effective_k_zero_fallback() {
        let policy = super::FusionPolicy { rrf_k: 0.0 };
        assert!((policy.effective_k() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fusion_policy_effective_k_negative_fallback() {
        let policy = super::FusionPolicy { rrf_k: -1.0 };
        assert!((policy.effective_k() - 60.0).abs() < f64::EPSILON);
    }

    // --- RankingPriorWeights ---

    #[test]
    fn ranking_prior_weights_default() {
        let weights = super::RankingPriorWeights::default();
        assert!((weights.recency - 0.12).abs() < f64::EPSILON);
        assert!((weights.path - 0.08).abs() < f64::EPSILON);
        assert!((weights.project - 0.05).abs() < f64::EPSILON);
    }

    // --- RankingPriorTuning ---

    #[test]
    fn ranking_prior_tuning_default_matches_performance() {
        let default = RankingPriorTuning::default();
        let perf = RankingPriorTuning::for_profile(PressureProfile::Performance);
        assert_eq!(default.enabled, perf.enabled);
        assert!((default.max_total_boost - perf.max_total_boost).abs() < f64::EPSILON);
    }

    #[test]
    fn ranking_prior_tuning_boost_disabled_returns_zero() {
        let tuning = RankingPriorTuning {
            enabled: false,
            ..RankingPriorTuning::default()
        };
        let boost = tuning.boost_for(RankingPriorSignals::new(Some(1.0), Some(1.0), Some(1.0)));
        assert!(boost.abs() < f64::EPSILON);
    }

    #[test]
    fn ranking_prior_tuning_normalized_scales_weights_if_sum_exceeds_max() {
        let tuning = RankingPriorTuning {
            enabled: true,
            weights: super::RankingPriorWeights::new(1.0, 1.0, 1.0),
            max_total_boost: 0.01,
        };
        let normalized = tuning.normalized();
        let sum = normalized.weights.recency + normalized.weights.path + normalized.weights.project;
        assert!(sum <= normalized.max_total_boost + 1e-12);
    }

    #[test]
    fn ranking_prior_tuning_boost_clamped() {
        let tuning = RankingPriorTuning::for_profile(PressureProfile::Performance).normalized();
        let boost = tuning.boost_for(RankingPriorSignals::new(Some(1.0), Some(1.0), Some(1.0)));
        assert!(boost >= 0.0);
        assert!(boost <= tuning.max_total_boost + 1e-12);
    }

    // --- RankingPriorSignals ---

    #[test]
    fn ranking_prior_signals_default() {
        let signals = RankingPriorSignals::default();
        assert!(signals.recency.is_none());
        assert!(signals.path.is_none());
        assert!(signals.project.is_none());
    }

    #[test]
    fn ranking_prior_signals_new() {
        let signals = RankingPriorSignals::new(Some(0.5), Some(0.3), None);
        assert_eq!(signals.recency, Some(0.5));
        assert_eq!(signals.path, Some(0.3));
        assert!(signals.project.is_none());
    }

    // --- Candidate constructors ---

    #[test]
    fn lexical_candidate_new() {
        let c = LexicalCandidate::new("doc-x", 2.5);
        assert_eq!(c.doc_id, "doc-x");
        assert!((c.score - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn semantic_candidate_new() {
        let c = SemanticCandidate::new("doc-y", 0.9);
        assert_eq!(c.doc_id, "doc-y");
        assert!((c.score - 0.9).abs() < f32::EPSILON);
    }

    // --- QueryExecutionOrchestrator ---

    #[test]
    fn orchestrator_default() {
        let o = QueryExecutionOrchestrator::default();
        assert!((o.fusion_policy.rrf_k - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn orchestrator_new_custom_k() {
        let o = QueryExecutionOrchestrator::new(super::FusionPolicy { rrf_k: 42.0 });
        assert!((o.fusion_policy.rrf_k - 42.0).abs() < f64::EPSILON);
    }

    // --- fuse_rankings offset ---

    #[test]
    fn fuse_rankings_offset_skips_top_results() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", 10.0),
            LexicalCandidate::new("doc-b", 9.0),
            LexicalCandidate::new("doc-c", 8.0),
        ];
        let semantic = vec![];

        let all = orchestrator.fuse_rankings(&lexical, &semantic, 10, 0);
        let offset = orchestrator.fuse_rankings(&lexical, &semantic, 10, 1);

        assert_eq!(all.len(), 3);
        assert_eq!(offset.len(), 2);
        assert_eq!(offset[0].doc_id, all[1].doc_id);
    }

    #[test]
    fn fuse_rankings_limit_caps_results() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let lexical = vec![
            LexicalCandidate::new("doc-a", 10.0),
            LexicalCandidate::new("doc-b", 9.0),
            LexicalCandidate::new("doc-c", 8.0),
        ];
        let fused = orchestrator.fuse_rankings(&lexical, &[], 2, 0);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_rankings_empty_inputs() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let fused = orchestrator.fuse_rankings(&[], &[], 10, 0);
        assert!(fused.is_empty());
    }

    // --- fused_cmp ---

    #[test]
    fn fused_cmp_score_descending() {
        let a = super::FusedCandidate {
            doc_id: "a".to_owned(),
            fused_score: 0.9,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        let b = super::FusedCandidate {
            doc_id: "b".to_owned(),
            fused_score: 0.8,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        assert_eq!(super::fused_cmp(&a, &b), std::cmp::Ordering::Less);
    }

    #[test]
    fn fused_cmp_in_both_tiebreak() {
        let a = super::FusedCandidate {
            doc_id: "a".to_owned(),
            fused_score: 0.5,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: true,
        };
        let b = super::FusedCandidate {
            doc_id: "b".to_owned(),
            fused_score: 0.5,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        // a in_both=true should sort first (compare right.in_both > left.in_both → Less)
        assert_eq!(super::fused_cmp(&a, &b), std::cmp::Ordering::Less);
    }

    #[test]
    fn fused_cmp_doc_id_tiebreak() {
        let a = super::FusedCandidate {
            doc_id: "alpha".to_owned(),
            fused_score: 0.5,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        let b = super::FusedCandidate {
            doc_id: "beta".to_owned(),
            fused_score: 0.5,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        // "alpha" < "beta" alphabetically → a sorts first
        assert_eq!(super::fused_cmp(&a, &b), std::cmp::Ordering::Less);
    }

    // --- DegradationStateMachine ---

    #[test]
    fn degradation_state_machine_for_profile() {
        let machine = DegradationStateMachine::for_profile(PressureProfile::Strict);
        assert_eq!(machine.mode(), DegradedRetrievalMode::Normal);
        assert_eq!(machine.override_mode(), DegradationOverride::Auto);
    }

    #[test]
    fn degradation_state_machine_stable_transition() {
        let mut machine = DegradationStateMachine::default();
        let t = machine.observe(DegradationSignals::default());
        assert!(!t.changed);
        assert_eq!(t.reason_code, "degrade.transition.stable");
        assert_eq!(t.to, DegradedRetrievalMode::Normal);
    }

    #[test]
    fn degradation_state_machine_override_stable() {
        let mut machine = DegradationStateMachine::default();
        machine.set_override(DegradationOverride::ForceNormal);
        let t = machine.observe(DegradationSignals::default());
        assert!(!t.changed);
        assert_eq!(t.reason_code, "degrade.override.stable");
    }

    #[test]
    fn degradation_state_machine_override_applied_then_stable() {
        let mut machine = DegradationStateMachine::default();
        machine.set_override(DegradationOverride::ForceLexicalOnly);
        let t1 = machine.observe(DegradationSignals::default());
        assert!(t1.changed);
        assert_eq!(t1.reason_code, "degrade.override.applied");
        assert_eq!(t1.to, DegradedRetrievalMode::LexicalOnly);

        let t2 = machine.observe(DegradationSignals::default());
        assert!(!t2.changed);
        assert_eq!(t2.reason_code, "degrade.override.stable");
    }

    #[test]
    fn degradation_transition_context_state_stable() {
        let t = DegradationTransition {
            from: DegradedRetrievalMode::Normal,
            to: DegradedRetrievalMode::Normal,
            changed: false,
            reason_code: "degrade.transition.stable",
            status: status_for_mode(DegradedRetrievalMode::Normal),
            override_mode: DegradationOverride::Auto,
        };
        assert_eq!(t.transition_context(), "state_stable");
    }

    #[test]
    fn degradation_transition_context_manual_override_hold() {
        let t = DegradationTransition {
            from: DegradedRetrievalMode::Paused,
            to: DegradedRetrievalMode::Paused,
            changed: false,
            reason_code: "degrade.override.stable",
            status: status_for_mode(DegradedRetrievalMode::Paused),
            override_mode: DegradationOverride::ForcePause,
        };
        assert_eq!(t.transition_context(), "manual_override_hold");
    }

    // --- cancellation directive reason codes ---

    #[test]
    fn cancellation_directive_after_initial_results() {
        let orchestrator = QueryExecutionOrchestrator::default();
        let d = orchestrator.cancellation_directive(CancellationPoint::AfterInitialResults);
        assert_eq!(d.action, CancellationAction::EmitInitialResults);
        assert_eq!(d.reason_code, "query.cancel.after_initial_results");
    }

    // --- embed_deferred plan ---

    #[test]
    fn plan_embed_deferred_includes_semantic() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let intent = planner.classify_intent("how does fusion work");
        let budget = planner.budget_for_decision(&intent, Some(10));
        let orchestrator = QueryExecutionOrchestrator::default();

        let plan =
            orchestrator.plan_with_mode(&intent, &budget, DegradedRetrievalMode::EmbedDeferred);
        let names = stage_names(&plan);

        assert!(names.contains(&RetrievalStage::FastSemanticRetrieve));
        assert!(!names.contains(&RetrievalStage::QualitySemanticRetrieve));
        assert_eq!(plan.reason_code, "query.execution.embed_deferred");
    }

    // ─── bd-olv6 tests end ───
}
