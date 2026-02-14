//! fsfs explanation payload schema for ranking and policy decisions.
//!
//! This module bridges library-level per-hit explanations into fsfs-facing
//! payloads that are compatible with:
//! - CLI JSON output (`to_cli_json`)
//! - CLI TOON text output (`to_toon`)
//! - TUI explainability panels (`to_tui_panel`)

use std::collections::BTreeMap;
use std::io;

use frankensearch_core::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent, SearchError,
    SearchResult,
};
use serde::{Deserialize, Serialize};

use crate::config::DiscoveryDecision;
use crate::evidence::TraceLink;
use crate::query_execution::{DegradationTransition, FusedCandidate, QueryExecutionPlan};
use crate::query_planning::{QueryBudgetProfile, QueryIntentDecision, RetrievalBudget};

pub const EXPLANATION_PAYLOAD_SCHEMA_VERSION: &str = "fsfs.explanation.payload.v1";
const SUBSYSTEM: &str = "fsfs_explanation";

/// Top-level fsfs explanation payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FsfsExplanationPayload {
    pub schema_version: String,
    pub query: String,
    pub trace: Option<TraceLink>,
    pub ranking: RankingExplanation,
    pub policy_decisions: Vec<PolicyDecisionExplanation>,
}

impl FsfsExplanationPayload {
    #[must_use]
    pub fn new(query: impl Into<String>, ranking: RankingExplanation) -> Self {
        Self {
            schema_version: EXPLANATION_PAYLOAD_SCHEMA_VERSION.to_owned(),
            query: query.into(),
            trace: None,
            ranking,
            policy_decisions: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_trace(mut self, trace: TraceLink) -> Self {
        self.trace = Some(trace);
        self
    }

    #[must_use]
    pub fn with_policy_decision(mut self, decision: PolicyDecisionExplanation) -> Self {
        self.policy_decisions.push(decision);
        self
    }

    /// Serialize payload to CLI JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization fails.
    pub fn to_cli_json(&self) -> SearchResult<String> {
        serde_json::to_string_pretty(self).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "failed to serialize explanation payload: {source}"
            ))),
        })
    }

    /// Render payload to deterministic TOON text.
    #[must_use]
    pub fn to_toon(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("schema_version: {}", self.schema_version));
        lines.push(format!("query: {}", self.query));
        if let Some(trace) = &self.trace {
            lines.push(format!("trace_id: {}", trace.trace_id));
            lines.push(format!("event_id: {}", trace.event_id));
        }
        lines.push(format!(
            "ranking: doc={} score={:.6} phase={} reason={} confidence_per_mille={}",
            self.ranking.doc_id,
            self.ranking.final_score,
            phase_token(self.ranking.phase),
            self.ranking.reason_code,
            self.ranking.confidence_per_mille
        ));

        for component in &self.ranking.components {
            lines.push(format!(
                "  component: source={} raw={:.6} normalized={:.6} rrf={:.6} weight={:.3} confidence_per_mille={} summary={}",
                component.source,
                component.raw_score,
                component.normalized_score,
                component.rrf_contribution,
                component.weight,
                component.confidence_per_mille,
                component.summary
            ));
        }

        for decision in &self.policy_decisions {
            lines.push(format!(
                "  policy: domain={} decision={} reason={} confidence_per_mille={} summary={}",
                decision.domain,
                decision.decision,
                decision.reason_code,
                decision.confidence_per_mille,
                decision.summary
            ));
        }

        lines.join("\n")
    }

    /// Build a TUI-ready panel representation.
    #[must_use]
    pub fn to_tui_panel(&self) -> TuiExplanationPanel {
        let mut lines = Vec::new();
        lines.push(format!("Query: {}", self.query));
        lines.push(format!(
            "Result: {} ({}) score {:.6}",
            self.ranking.doc_id, self.ranking.phase, self.ranking.final_score
        ));
        lines.push(format!(
            "Reason: {} (confidence {}‰)",
            self.ranking.reason_code, self.ranking.confidence_per_mille
        ));

        for component in &self.ranking.components {
            lines.push(format!(
                "- {}: {:.3} (w {:.2}) {}",
                component.source, component.normalized_score, component.weight, component.summary
            ));
        }

        for decision in &self.policy_decisions {
            lines.push(format!(
                "* {}: {} [{} {}‰]",
                decision.domain,
                decision.decision,
                decision.reason_code,
                decision.confidence_per_mille
            ));
        }

        TuiExplanationPanel {
            title: "Explainability".to_owned(),
            subtitle: format!("schema {}", self.schema_version),
            lines,
        }
    }
}

/// TUI explainability panel projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuiExplanationPanel {
    pub title: String,
    pub subtitle: String,
    pub lines: Vec<String>,
}

/// Ranking explanation schema with component-level breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingExplanation {
    pub doc_id: String,
    pub final_score: f64,
    pub phase: ExplanationPhase,
    pub reason_code: String,
    pub confidence_per_mille: u16,
    pub rank_movement: Option<RankMovementSnapshot>,
    pub fusion: Option<FusionContext>,
    pub components: Vec<ScoreComponentBreakdown>,
}

impl RankingExplanation {
    #[must_use]
    pub fn from_hit_explanation(
        doc_id: impl Into<String>,
        explanation: &HitExplanation,
        reason_code: impl Into<String>,
        confidence_per_mille: u16,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            final_score: explanation.final_score,
            phase: explanation.phase,
            reason_code: reason_code.into(),
            confidence_per_mille: normalize_confidence(confidence_per_mille),
            rank_movement: explanation
                .rank_movement
                .as_ref()
                .map(RankMovementSnapshot::from),
            fusion: None,
            components: explanation
                .components
                .iter()
                .map(ScoreComponentBreakdown::from)
                .collect(),
        }
    }

    #[must_use]
    pub fn with_fusion_context(mut self, fused: &FusedCandidate) -> Self {
        self.fusion = Some(FusionContext::from(fused));
        self
    }
}

/// Rank movement snapshot (phase 1 -> phase 2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankMovementSnapshot {
    pub initial_rank: usize,
    pub refined_rank: usize,
    pub delta: i32,
    pub reason: String,
}

impl From<&RankMovement> for RankMovementSnapshot {
    fn from(value: &RankMovement) -> Self {
        Self {
            initial_rank: value.initial_rank,
            refined_rank: value.refined_rank,
            delta: value.delta,
            reason: value.reason.clone(),
        }
    }
}

/// Fused ranking metadata for explainability surfaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionContext {
    pub fused_score: f64,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub in_both_sources: bool,
}

impl From<&FusedCandidate> for FusionContext {
    fn from(value: &FusedCandidate) -> Self {
        Self {
            fused_score: value.fused_score,
            lexical_rank: value.lexical_rank,
            semantic_rank: value.semantic_rank,
            lexical_score: value.lexical_score,
            semantic_score: value.semantic_score,
            in_both_sources: value.in_both_sources,
        }
    }
}

/// Component source for stable schema serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreComponentSource {
    LexicalBm25,
    SemanticFast,
    SemanticQuality,
    Rerank,
}

impl std::fmt::Display for ScoreComponentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::LexicalBm25 => "lexical_bm25",
            Self::SemanticFast => "semantic_fast",
            Self::SemanticQuality => "semantic_quality",
            Self::Rerank => "rerank",
        };
        write!(f, "{value}")
    }
}

/// One component-level score breakdown row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreComponentBreakdown {
    pub source: ScoreComponentSource,
    pub summary: String,
    pub raw_score: f64,
    pub normalized_score: f64,
    pub rrf_contribution: f64,
    pub weight: f64,
    pub confidence_per_mille: u16,
}

impl From<&ScoreComponent> for ScoreComponentBreakdown {
    fn from(value: &ScoreComponent) -> Self {
        Self {
            source: source_from_explained(&value.source),
            summary: value.source.to_string(),
            raw_score: value.raw_score,
            normalized_score: value.normalized_score,
            rrf_contribution: value.rrf_contribution,
            weight: value.weight,
            confidence_per_mille: component_confidence_per_mille(value),
        }
    }
}

/// Policy decision domain represented in explanation payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDomain {
    QueryIntent,
    RetrievalBudget,
    QueryExecution,
    Degradation,
    Discovery,
}

impl std::fmt::Display for PolicyDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::QueryIntent => "query_intent",
            Self::RetrievalBudget => "retrieval_budget",
            Self::QueryExecution => "query_execution",
            Self::Degradation => "degradation",
            Self::Discovery => "discovery",
        };
        write!(f, "{value}")
    }
}

/// Policy decision explanation row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecisionExplanation {
    pub domain: PolicyDomain,
    pub decision: String,
    pub reason_code: String,
    pub confidence_per_mille: u16,
    pub summary: String,
    pub metadata: BTreeMap<String, String>,
}

impl PolicyDecisionExplanation {
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl From<&QueryIntentDecision> for PolicyDecisionExplanation {
    fn from(value: &QueryIntentDecision) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "normalized_query".to_owned(),
            value.normalized_query.clone(),
        );
        metadata.insert("intent".to_owned(), format!("{:?}", value.intent));
        metadata.insert("fallback".to_owned(), format!("{:?}", value.fallback));
        if let Some(base_class) = value.base_class {
            metadata.insert("base_class".to_owned(), format!("{base_class:?}"));
        }

        Self {
            domain: PolicyDomain::QueryIntent,
            decision: format!("{:?}", value.intent),
            reason_code: value.reason_code.to_owned(),
            confidence_per_mille: normalize_confidence(value.confidence_per_mille),
            summary: format!("query intent classified as {:?}", value.intent),
            metadata,
        }
    }
}

impl From<&RetrievalBudget> for PolicyDecisionExplanation {
    fn from(value: &RetrievalBudget) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("profile".to_owned(), format!("{:?}", value.profile));
        metadata.insert("limit".to_owned(), value.limit.to_string());
        metadata.insert(
            "latency_budget_ms".to_owned(),
            value.latency_budget_ms.to_string(),
        );
        metadata.insert(
            "lexical_fanout".to_owned(),
            value.lexical_fanout.to_string(),
        );
        metadata.insert(
            "semantic_fanout".to_owned(),
            value.semantic_fanout.to_string(),
        );
        metadata.insert("rerank_depth".to_owned(), value.rerank_depth.to_string());
        metadata.insert(
            "quality_enabled".to_owned(),
            value.quality_enabled.to_string(),
        );

        Self {
            domain: PolicyDomain::RetrievalBudget,
            decision: format!("{:?}", value.profile),
            reason_code: value.reason_code.to_owned(),
            confidence_per_mille: budget_profile_confidence(value.profile),
            summary: format!(
                "budget limit {} with lexical fanout {} and semantic fanout {}",
                value.limit, value.lexical_fanout, value.semantic_fanout
            ),
            metadata,
        }
    }
}

impl From<&QueryExecutionPlan> for PolicyDecisionExplanation {
    fn from(value: &QueryExecutionPlan) -> Self {
        let stage_codes = value
            .stages
            .iter()
            .map(|stage| stage.reason_code)
            .collect::<Vec<_>>()
            .join(",");

        let mut metadata = BTreeMap::new();
        metadata.insert("mode".to_owned(), format!("{:?}", value.mode));
        metadata.insert("stage_count".to_owned(), value.stages.len().to_string());
        metadata.insert("stage_reason_codes".to_owned(), stage_codes);

        Self {
            domain: PolicyDomain::QueryExecution,
            decision: format!("{:?}", value.mode),
            reason_code: value.reason_code.to_owned(),
            confidence_per_mille: execution_plan_confidence(value),
            summary: format!(
                "execution mode {:?} with {} stages",
                value.mode,
                value.stages.len()
            ),
            metadata,
        }
    }
}

impl From<&DegradationTransition> for PolicyDecisionExplanation {
    fn from(value: &DegradationTransition) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("from".to_owned(), format!("{:?}", value.from));
        metadata.insert("to".to_owned(), format!("{:?}", value.to));
        metadata.insert("changed".to_owned(), value.changed.to_string());
        metadata.insert("banner".to_owned(), value.status.banner.to_owned());
        metadata.insert(
            "controls_hint".to_owned(),
            value.status.controls_hint.to_owned(),
        );
        metadata.insert(
            "override_mode".to_owned(),
            format!("{:?}", value.override_mode),
        );

        Self {
            domain: PolicyDomain::Degradation,
            decision: format!("{:?}", value.to),
            reason_code: value.reason_code.to_owned(),
            confidence_per_mille: degradation_confidence(value),
            summary: format!(
                "degradation transition {:?} -> {:?} (changed={})",
                value.from, value.to, value.changed
            ),
            metadata,
        }
    }
}

impl From<&DiscoveryDecision> for PolicyDecisionExplanation {
    fn from(value: &DiscoveryDecision) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("scope".to_owned(), format!("{:?}", value.scope));
        metadata.insert(
            "ingestion_class".to_owned(),
            format!("{:?}", value.ingestion_class),
        );
        metadata.insert("utility_score".to_owned(), value.utility_score.to_string());
        metadata.insert("reason_codes".to_owned(), value.reason_codes.join(","));

        let reason_code = value
            .reason_codes
            .first()
            .cloned()
            .unwrap_or_else(|| "discovery.reason.unknown".to_owned());

        Self {
            domain: PolicyDomain::Discovery,
            decision: format!("{:?}", value.ingestion_class),
            reason_code,
            confidence_per_mille: utility_confidence(value.utility_score),
            summary: format!(
                "discovery scope {:?} with utility {}",
                value.scope, value.utility_score
            ),
            metadata,
        }
    }
}

const fn normalize_confidence(value: u16) -> u16 {
    if value > 1_000 { 1_000 } else { value }
}

const fn budget_profile_confidence(profile: QueryBudgetProfile) -> u16 {
    match profile {
        QueryBudgetProfile::Empty => 1_000,
        QueryBudgetProfile::IdentifierFocused => 900,
        QueryBudgetProfile::Balanced => 850,
        QueryBudgetProfile::SemanticFocused => 800,
        QueryBudgetProfile::SafeFallback => 700,
    }
}

fn execution_plan_confidence(plan: &QueryExecutionPlan) -> u16 {
    let mode_bonus = if matches!(
        plan.mode,
        crate::query_execution::DegradedRetrievalMode::Normal
    ) {
        150
    } else {
        0
    };
    let stage_signal = u16::try_from(plan.stages.len().min(8))
        .ok()
        .and_then(|count| count.checked_mul(100))
        .unwrap_or(800);
    normalize_confidence(500 + mode_bonus + stage_signal)
}

const fn degradation_confidence(transition: &DegradationTransition) -> u16 {
    if matches!(
        transition.override_mode,
        crate::query_execution::DegradationOverride::Auto
    ) && transition.changed
    {
        850
    } else if transition.changed {
        900
    } else {
        700
    }
}

fn utility_confidence(utility_score: i32) -> u16 {
    let bounded = utility_score.clamp(-100, 100);
    let scaled = (bounded + 100) * 5;
    u16::try_from(scaled).unwrap_or_default()
}

fn component_confidence_per_mille(component: &ScoreComponent) -> u16 {
    let normalized_signal = bounded_signal(component.normalized_score.abs(), 500);
    let weight_signal = bounded_signal(component.weight.abs(), 300);
    let raw_signal = bounded_signal((component.raw_score.abs() / 10.0).min(1.0), 200);
    normalize_confidence(
        normalized_signal
            .saturating_add(weight_signal)
            .saturating_add(raw_signal),
    )
}

const fn source_from_explained(source: &ExplainedSource) -> ScoreComponentSource {
    match source {
        ExplainedSource::LexicalBm25 { .. } => ScoreComponentSource::LexicalBm25,
        ExplainedSource::SemanticFast { .. } => ScoreComponentSource::SemanticFast,
        ExplainedSource::SemanticQuality { .. } => ScoreComponentSource::SemanticQuality,
        ExplainedSource::Rerank { .. } => ScoreComponentSource::Rerank,
    }
}

const fn phase_token(phase: ExplanationPhase) -> &'static str {
    match phase {
        ExplanationPhase::Initial => "initial",
        ExplanationPhase::Refined => "refined",
    }
}

fn bounded_signal(value: f64, max: u16) -> u16 {
    let bounded = value.clamp(0.0, 1.0);
    let mut signal = 0_u16;
    while signal < max && (f64::from(signal) + 0.5) / f64::from(max) <= bounded {
        signal = signal.saturating_add(1);
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::{
        FsfsExplanationPayload, PolicyDecisionExplanation, PolicyDomain, RankingExplanation,
        ScoreComponentBreakdown, ScoreComponentSource,
    };
    use crate::config::{DiscoveryDecision, DiscoveryScopeDecision, IngestionClass};
    use crate::query_execution::{
        CancellationAction, DegradationOverride, DegradationStatus, DegradationTransition,
        DegradedRetrievalMode, QueryExecutionPlan, RetrievalStage, StagePlan,
    };
    use crate::query_planning::{
        QueryBudgetProfile, QueryFallbackPath, QueryIntentClass, QueryIntentDecision,
        RetrievalBudget,
    };
    use frankensearch_core::{
        ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
    };

    #[test]
    fn ranking_schema_maps_core_hit_explanation_components() {
        let hit = HitExplanation {
            final_score: 0.42,
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 3,
                refined_rank: 1,
                delta: -2,
                reason: "quality semantic uplift".to_owned(),
            }),
            components: vec![
                ScoreComponent {
                    source: ExplainedSource::LexicalBm25 {
                        matched_terms: vec!["rust".to_owned()],
                        tf: 1.0,
                        idf: 2.0,
                    },
                    raw_score: 12.0,
                    normalized_score: 0.9,
                    rrf_contribution: 0.015,
                    weight: 0.3,
                },
                ScoreComponent {
                    source: ExplainedSource::SemanticFast {
                        embedder: "potion-128M".to_owned(),
                        cosine_sim: 0.77,
                    },
                    raw_score: 0.77,
                    normalized_score: 0.8,
                    rrf_contribution: 0.014,
                    weight: 0.7,
                },
            ],
        };

        let ranking =
            RankingExplanation::from_hit_explanation("doc-1", &hit, "query.explain.attached", 940);
        assert_eq!(ranking.doc_id, "doc-1");
        assert_eq!(ranking.reason_code, "query.explain.attached");
        assert_eq!(ranking.confidence_per_mille, 940);
        assert_eq!(ranking.components.len(), 2);
        assert_eq!(
            ranking.components[0].source,
            ScoreComponentSource::LexicalBm25
        );
        assert_eq!(
            ranking.components[1].source,
            ScoreComponentSource::SemanticFast
        );
        assert!(ranking.rank_movement.is_some());
    }

    #[test]
    fn policy_conversions_capture_reason_codes_and_confidence() {
        let intent = QueryIntentDecision {
            normalized_query: "rust async".to_owned(),
            intent: QueryIntentClass::NaturalLanguage,
            base_class: None,
            confidence_per_mille: 920,
            fallback: QueryFallbackPath::None,
            reason_code: "query.intent.natural_language",
        };
        let budget = RetrievalBudget {
            profile: QueryBudgetProfile::Balanced,
            limit: 20,
            latency_budget_ms: 120,
            lexical_fanout: 100,
            semantic_fanout: 80,
            rerank_depth: 25,
            quality_enabled: true,
            fallback: QueryFallbackPath::None,
            reason_code: "query.budget.balanced",
        };
        let plan = QueryExecutionPlan {
            mode: DegradedRetrievalMode::Normal,
            reason_code: "query.execution.normal",
            stages: vec![StagePlan {
                stage: RetrievalStage::LexicalRetrieve,
                timeout_ms: 50,
                fanout: 100,
                required: true,
                reason_code: "query.stage.lexical_retrieve",
            }],
        };
        let discovery = DiscoveryDecision {
            scope: DiscoveryScopeDecision::Include,
            ingestion_class: IngestionClass::FullSemanticLexical,
            utility_score: 80,
            reason_codes: vec!["discovery.file.included".to_owned()],
        };

        let intent_policy = PolicyDecisionExplanation::from(&intent);
        let budget_policy = PolicyDecisionExplanation::from(&budget);
        let execution_policy = PolicyDecisionExplanation::from(&plan);
        let discovery_policy = PolicyDecisionExplanation::from(&discovery);

        assert_eq!(intent_policy.domain, PolicyDomain::QueryIntent);
        assert_eq!(intent_policy.reason_code, "query.intent.natural_language");
        assert_eq!(intent_policy.confidence_per_mille, 920);

        assert_eq!(budget_policy.domain, PolicyDomain::RetrievalBudget);
        assert_eq!(budget_policy.reason_code, "query.budget.balanced");
        assert!(budget_policy.confidence_per_mille >= 800);

        assert_eq!(execution_policy.domain, PolicyDomain::QueryExecution);
        assert_eq!(execution_policy.reason_code, "query.execution.normal");
        assert!(execution_policy.confidence_per_mille > 500);

        assert_eq!(discovery_policy.domain, PolicyDomain::Discovery);
        assert_eq!(discovery_policy.reason_code, "discovery.file.included");
        assert!(discovery_policy.confidence_per_mille > 800);
    }

    #[test]
    fn degradation_policy_conversion_preserves_banner_and_reason() {
        let transition = DegradationTransition {
            from: DegradedRetrievalMode::EmbedDeferred,
            to: DegradedRetrievalMode::LexicalOnly,
            changed: true,
            reason_code: "degrade.transition.applied",
            status: DegradationStatus {
                banner: "Constrained mode",
                controls_hint: "reduce fanout",
            },
            override_mode: DegradationOverride::Auto,
        };

        let policy = PolicyDecisionExplanation::from(&transition);
        assert_eq!(policy.domain, PolicyDomain::Degradation);
        assert_eq!(policy.reason_code, "degrade.transition.applied");
        assert_eq!(
            policy.metadata.get("banner"),
            Some(&"Constrained mode".to_owned())
        );
    }

    #[test]
    fn payload_json_toon_and_tui_outputs_include_schema_fields() {
        let hit = HitExplanation {
            final_score: 0.75,
            phase: ExplanationPhase::Initial,
            rank_movement: None,
            components: vec![ScoreComponent {
                source: ExplainedSource::Rerank {
                    model: "cross-encoder".to_owned(),
                    logit: 2.0,
                    sigmoid: 0.88,
                },
                raw_score: 0.88,
                normalized_score: 0.88,
                rrf_contribution: 0.012,
                weight: 1.0,
            }],
        };
        let ranking =
            RankingExplanation::from_hit_explanation("doc-99", &hit, "query.explain.attached", 970);
        let policy = PolicyDecisionExplanation {
            domain: PolicyDomain::QueryExecution,
            decision: format!("{:?}", DegradedRetrievalMode::Normal),
            reason_code: "query.execution.normal".to_owned(),
            confidence_per_mille: 900,
            summary: "normal execution".to_owned(),
            metadata: std::collections::BTreeMap::new(),
        };

        let payload =
            FsfsExplanationPayload::new("how does rrf work", ranking).with_policy_decision(policy);

        let json = payload.to_cli_json().expect("json serialization");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        assert_eq!(
            value["schema_version"].as_str(),
            Some("fsfs.explanation.payload.v1")
        );
        assert_eq!(
            value["ranking"]["reason_code"].as_str(),
            Some("query.explain.attached")
        );
        assert_eq!(
            value["policy_decisions"][0]["confidence_per_mille"].as_u64(),
            Some(900)
        );

        let toon = payload.to_toon();
        assert!(toon.contains("schema_version: fsfs.explanation.payload.v1"));
        assert!(toon.contains("reason=query.explain.attached"));
        assert!(toon.contains("policy: domain=query_execution"));

        let panel = payload.to_tui_panel();
        assert_eq!(panel.title, "Explainability");
        assert!(panel.lines.iter().any(|line| line.contains("Reason:")));
    }

    #[test]
    fn score_component_breakdown_retains_source_specific_summary() {
        let component = ScoreComponent {
            source: ExplainedSource::SemanticQuality {
                embedder: "all-MiniLM-L6-v2".to_owned(),
                cosine_sim: 0.91,
            },
            raw_score: 0.91,
            normalized_score: 0.89,
            rrf_contribution: 0.016,
            weight: 0.7,
        };

        let breakdown = ScoreComponentBreakdown::from(&component);
        assert_eq!(breakdown.source, ScoreComponentSource::SemanticQuality);
        assert!(breakdown.summary.contains("QualitySemantic"));
        assert!(
            breakdown.confidence_per_mille > 600,
            "expected confidence > 600 for quality semantic with 0.91 raw score, got {}",
            breakdown.confidence_per_mille,
        );
    }

    #[test]
    fn extra_enums_used_in_schema_tests() {
        let _ = CancellationAction::EmitInitialResults;
    }
}
