//! Explainability screen contracts for the fsfs deluxe TUI.
//!
//! Defines the data model and rendering contracts for the "galaxy-brain"
//! explainability view: ranking decision cards, policy decision cards,
//! evidence-trace drilldowns, and adaptive detail levels from novice to expert.
//!
//! This module builds on [`explanation_payload`](crate::explanation_payload) for
//! data and [`interaction_primitives`](crate::interaction_primitives) for layout
//! and navigation contracts.

use std::fmt;

use crate::evidence::TraceLink;
use crate::explanation_payload::{
    FusionContext, PolicyDecisionExplanation, PolicyDomain, RankMovementSnapshot,
    RankingExplanation, ScoreComponentBreakdown, ScoreComponentSource,
};
use crate::interaction_primitives::VirtualizedListState;

// ─── Schema Version ──────────────────────────────────────────────────────────

/// Schema version for explainability screen contracts.
pub const EXPLAINABILITY_SCREEN_SCHEMA_VERSION: u32 = 1;

// ─── Explainability Level ───────────────────────────────────────────────────

/// Controls the depth of explanation detail shown in the TUI.
///
/// Users cycle through levels via keyboard shortcut. Each level adds more
/// technical detail while keeping the screen readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ExplainabilityLevel {
    /// Plain-language summary only. Good for quick triage.
    #[default]
    Novice,
    /// Adds score breakdowns and policy reason codes.
    Intermediate,
    /// Full detail: equations, substituted values, trace links.
    Expert,
}

impl ExplainabilityLevel {
    /// Cycle to the next level (wraps from `Expert` to `Novice`).
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Novice => Self::Intermediate,
            Self::Intermediate => Self::Expert,
            Self::Expert => Self::Novice,
        }
    }

    /// Human label for status bar display.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Novice => "Novice",
            Self::Intermediate => "Intermediate",
            Self::Expert => "Expert",
        }
    }

    /// Whether score component tables should be shown.
    #[must_use]
    pub const fn show_components(self) -> bool {
        matches!(self, Self::Intermediate | Self::Expert)
    }

    /// Whether equation strings and raw values should be shown.
    #[must_use]
    pub const fn show_equations(self) -> bool {
        matches!(self, Self::Expert)
    }

    /// Whether trace-link IDs should be shown inline.
    #[must_use]
    pub const fn show_trace_links(self) -> bool {
        matches!(self, Self::Expert)
    }

    /// Maximum number of policy decision cards to show.
    #[must_use]
    pub const fn max_policy_cards(self) -> usize {
        match self {
            Self::Novice => 2,
            Self::Intermediate => 5,
            Self::Expert => usize::MAX,
        }
    }
}

impl fmt::Display for ExplainabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── Ranking Decision Card ──────────────────────────────────────────────────

/// A rendered ranking decision card for the explainability panel.
///
/// Contains all the information needed to render one result's ranking
/// explanation, pre-filtered to the current [`ExplainabilityLevel`].
#[derive(Debug, Clone, PartialEq)]
pub struct RankingDecisionCard {
    /// Document ID.
    pub doc_id: String,
    /// Final fused score.
    pub final_score: f64,
    /// Phase indicator (e.g., "initial" or "refined").
    pub phase: String,
    /// Plain-language summary line.
    pub summary: String,
    /// Confidence indicator (0..=1000 per-mille).
    pub confidence_per_mille: u16,
    /// Score component rows (empty at novice level).
    pub components: Vec<ComponentRow>,
    /// Rank movement summary (if refined).
    pub rank_movement: Option<RankMovementRow>,
    /// Fusion context (lexical rank, semantic rank, overlap).
    pub fusion: Option<FusionRow>,
    /// Equation string for expert mode.
    pub equation: Option<String>,
    /// Trace link for expert mode.
    pub trace: Option<TraceLink>,
}

/// One row in the score component table.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentRow {
    /// Source label (e.g., "BM25", "`FastSemantic`").
    pub source_label: String,
    /// Normalized score (0..1).
    pub normalized_score: f64,
    /// RRF contribution.
    pub rrf_contribution: f64,
    /// Weight applied.
    pub weight: f64,
    /// Confidence badge (per-mille).
    pub confidence_per_mille: u16,
    /// Component-specific summary text.
    pub summary: String,
    /// Raw score (shown at expert level only).
    pub raw_score: Option<f64>,
}

/// Rank movement summary row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankMovementRow {
    /// Rank before refinement.
    pub initial_rank: usize,
    /// Rank after refinement.
    pub refined_rank: usize,
    /// Delta (negative = improved).
    pub delta: i32,
    /// Movement reason.
    pub reason: String,
    /// Human-readable summary.
    pub direction_label: String,
}

/// Fusion context row.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionRow {
    /// Fused score.
    pub fused_score: f64,
    /// Lexical rank (if present).
    pub lexical_rank: Option<usize>,
    /// Semantic rank (if present).
    pub semantic_rank: Option<usize>,
    /// Whether the doc appeared in both lexical and semantic sources.
    pub in_both_sources: bool,
    /// Human-readable overlap label.
    pub overlap_label: String,
}

// ─── Policy Decision Card ───────────────────────────────────────────────────

/// A rendered policy decision card for the explainability panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecisionCard {
    /// Policy domain label (e.g., "Query Intent", "Degradation").
    pub domain_label: String,
    /// Decision text.
    pub decision: String,
    /// Reason code.
    pub reason_code: String,
    /// Confidence badge (per-mille).
    pub confidence_per_mille: u16,
    /// Summary text.
    pub summary: String,
    /// Metadata key-value pairs (shown at intermediate+ levels).
    pub metadata: Vec<(String, String)>,
}

// ─── Evidence Trace Node ────────────────────────────────────────────────────

/// One node in an evidence-trace drilldown.
///
/// The drilldown shows the causal chain of events that led to the
/// current ranking or policy decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceNode {
    /// Event ID (ULID).
    pub event_id: String,
    /// Parent event ID (for tree rendering).
    pub parent_event_id: Option<String>,
    /// Event type label.
    pub event_type: String,
    /// Reason code.
    pub reason_code: String,
    /// Human-readable summary.
    pub summary: String,
    /// Timestamp string.
    pub timestamp: String,
    /// Nesting depth in the trace tree.
    pub depth: usize,
}

// ─── Card Builders ──────────────────────────────────────────────────────────

/// Build a [`RankingDecisionCard`] from a [`RankingExplanation`] at the given level.
#[must_use]
pub fn build_ranking_card(
    ranking: &RankingExplanation,
    level: ExplainabilityLevel,
    trace: Option<&TraceLink>,
) -> RankingDecisionCard {
    let summary = build_ranking_summary(ranking);

    let components = if level.show_components() {
        ranking
            .components
            .iter()
            .map(|c| build_component_row(c, level))
            .collect()
    } else {
        Vec::new()
    };

    let rank_movement = ranking.rank_movement.as_ref().map(build_rank_movement_row);

    let fusion = ranking.fusion.as_ref().map(build_fusion_row);

    let equation = if level.show_equations() {
        Some(build_ranking_equation(ranking))
    } else {
        None
    };

    let trace_link = if level.show_trace_links() {
        trace.cloned()
    } else {
        None
    };

    RankingDecisionCard {
        doc_id: ranking.doc_id.clone(),
        final_score: ranking.final_score,
        phase: format!("{:?}", ranking.phase),
        summary,
        confidence_per_mille: ranking.confidence_per_mille,
        components,
        rank_movement,
        fusion,
        equation,
        trace: trace_link,
    }
}

/// Build a [`PolicyDecisionCard`] from a [`PolicyDecisionExplanation`] at the given level.
#[must_use]
pub fn build_policy_card(
    decision: &PolicyDecisionExplanation,
    level: ExplainabilityLevel,
) -> PolicyDecisionCard {
    let metadata = if level.show_components() {
        decision
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    } else {
        Vec::new()
    };

    PolicyDecisionCard {
        domain_label: domain_label(decision.domain),
        decision: decision.decision.clone(),
        reason_code: decision.reason_code.clone(),
        confidence_per_mille: decision.confidence_per_mille,
        summary: decision.summary.clone(),
        metadata,
    }
}

// ─── Explainability Screen State ────────────────────────────────────────────

/// Complete state for the explainability screen.
///
/// Combines the explanation payload data with interaction state (selection,
/// scroll, level). Screen implementations render from this state.
#[derive(Debug, Clone)]
pub struct ExplainabilityScreenState {
    /// Current detail level.
    pub level: ExplainabilityLevel,
    /// Ranking cards for each explained result.
    pub ranking_cards: Vec<RankingDecisionCard>,
    /// Policy decision cards.
    pub policy_cards: Vec<PolicyDecisionCard>,
    /// Evidence trace nodes for the selected result.
    pub trace_nodes: Vec<TraceNode>,
    /// List state for ranking cards (primary panel).
    pub ranking_list: VirtualizedListState,
    /// List state for policy cards (if visible).
    pub policy_list: VirtualizedListState,
    /// List state for trace nodes (if visible).
    pub trace_list: VirtualizedListState,
    /// Whether the trace drilldown panel is expanded.
    pub trace_expanded: bool,
    /// Active query text (carried from search screen).
    pub query_text: String,
}

impl ExplainabilityScreenState {
    /// Create a new empty state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: ExplainabilityLevel::default(),
            ranking_cards: Vec::new(),
            policy_cards: Vec::new(),
            trace_nodes: Vec::new(),
            ranking_list: VirtualizedListState::empty(),
            policy_list: VirtualizedListState::empty(),
            trace_list: VirtualizedListState::empty(),
            trace_expanded: false,
            query_text: String::new(),
        }
    }

    /// Cycle the explainability level and rebuild visible cards.
    pub const fn cycle_level(&mut self) {
        self.level = self.level.next();
    }

    /// Update ranking cards from explanation data.
    pub fn set_ranking_explanations(
        &mut self,
        rankings: &[RankingExplanation],
        trace: Option<&TraceLink>,
    ) {
        self.ranking_cards = rankings
            .iter()
            .map(|r| build_ranking_card(r, self.level, trace))
            .collect();
        self.ranking_list.set_total_items(self.ranking_cards.len());
    }

    /// Update policy cards from explanation data.
    pub fn set_policy_decisions(&mut self, decisions: &[PolicyDecisionExplanation]) {
        let max = self.level.max_policy_cards();
        self.policy_cards = decisions
            .iter()
            .take(max)
            .map(|d| build_policy_card(d, self.level))
            .collect();
        self.policy_list.set_total_items(self.policy_cards.len());
    }

    /// Set trace nodes for the drilldown panel.
    pub fn set_trace_nodes(&mut self, nodes: Vec<TraceNode>) {
        let count = nodes.len();
        self.trace_nodes = nodes;
        self.trace_list.set_total_items(count);
    }

    /// Currently selected ranking card, if any.
    #[must_use]
    pub fn selected_ranking_card(&self) -> Option<&RankingDecisionCard> {
        self.ranking_cards.get(self.ranking_list.selected)
    }

    /// Currently selected policy card, if any.
    #[must_use]
    pub fn selected_policy_card(&self) -> Option<&PolicyDecisionCard> {
        self.policy_cards.get(self.policy_list.selected)
    }

    /// Rebuild all cards at the current level from stored data sources.
    ///
    /// Call this after `cycle_level()` to refresh card content without
    /// re-fetching data from the search pipeline.
    pub fn rebuild_cards(
        &mut self,
        rankings: &[RankingExplanation],
        decisions: &[PolicyDecisionExplanation],
        trace: Option<&TraceLink>,
    ) {
        self.set_ranking_explanations(rankings, trace);
        self.set_policy_decisions(decisions);
    }
}

impl Default for ExplainabilityScreenState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Confidence Badge ───────────────────────────────────────────────────────

/// Human-readable confidence badge derived from per-mille values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceBadge {
    /// >= 900 per-mille.
    High,
    /// 600..900 per-mille.
    Medium,
    /// < 600 per-mille.
    Low,
}

impl ConfidenceBadge {
    /// Classify a per-mille confidence value.
    #[must_use]
    pub const fn from_per_mille(value: u16) -> Self {
        if value >= 900 {
            Self::High
        } else if value >= 600 {
            Self::Medium
        } else {
            Self::Low
        }
    }

    /// Label string for rendering.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::High => "HIGH",
            Self::Medium => "MED",
            Self::Low => "LOW",
        }
    }

    /// Suggested color token (maps to theme accents).
    #[must_use]
    pub const fn color_token(self) -> &'static str {
        match self {
            Self::High => "success",
            Self::Medium => "warning",
            Self::Low => "error",
        }
    }
}

impl fmt::Display for ConfidenceBadge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

fn build_ranking_summary(ranking: &RankingExplanation) -> String {
    let badge = ConfidenceBadge::from_per_mille(ranking.confidence_per_mille);
    format!(
        "{} scored {:.4} ({}) [{}]",
        ranking.doc_id, ranking.final_score, badge, ranking.reason_code,
    )
}

fn build_component_row(
    component: &ScoreComponentBreakdown,
    level: ExplainabilityLevel,
) -> ComponentRow {
    ComponentRow {
        source_label: source_label(component.source),
        normalized_score: component.normalized_score,
        rrf_contribution: component.rrf_contribution,
        weight: component.weight,
        confidence_per_mille: component.confidence_per_mille,
        summary: component.summary.clone(),
        raw_score: if level.show_equations() {
            Some(component.raw_score)
        } else {
            None
        },
    }
}

fn build_rank_movement_row(movement: &RankMovementSnapshot) -> RankMovementRow {
    let direction_label = match movement.delta.cmp(&0) {
        std::cmp::Ordering::Less => {
            format!("improved by {} positions", movement.delta.unsigned_abs())
        }
        std::cmp::Ordering::Greater => {
            format!("dropped by {} positions", movement.delta)
        }
        std::cmp::Ordering::Equal => "unchanged".to_string(),
    };

    RankMovementRow {
        initial_rank: movement.initial_rank,
        refined_rank: movement.refined_rank,
        delta: movement.delta,
        reason: movement.reason.clone(),
        direction_label,
    }
}

fn build_fusion_row(fusion: &FusionContext) -> FusionRow {
    let overlap_label = if fusion.vector_generation_is_hash {
        if fusion.in_both_sources {
            "overlapping (lexical + hash control)".to_string()
        } else {
            match (fusion.lexical_rank, fusion.semantic_rank) {
                (Some(_), None) => "lexical-only".to_string(),
                (None, Some(_)) => "hash-control-only".to_string(),
                _ => "unknown source".to_string(),
            }
        }
    } else if fusion.in_both_sources {
        "overlapping (both lexical + semantic)".to_string()
    } else {
        match (fusion.lexical_rank, fusion.semantic_rank) {
            (Some(_), None) => "lexical-only".to_string(),
            (None, Some(_)) => "semantic-only".to_string(),
            _ => "unknown source".to_string(),
        }
    };

    FusionRow {
        fused_score: fusion.fused_score,
        lexical_rank: fusion.lexical_rank,
        semantic_rank: fusion.semantic_rank,
        in_both_sources: fusion.in_both_sources,
        overlap_label,
    }
}

/// Build the RRF equation string with substituted values for expert mode.
fn build_ranking_equation(ranking: &RankingExplanation) -> String {
    let mut parts = Vec::new();
    for c in &ranking.components {
        parts.push(format!(
            "1/(60+{rank}) [{src}]",
            rank = format_rank_placeholder(c.source),
            src = source_label(c.source),
        ));
    }
    if parts.is_empty() {
        return format!("score = {:.6}", ranking.final_score);
    }
    format!("score = {} = {:.6}", parts.join(" + "), ranking.final_score)
}

const fn format_rank_placeholder(source: ScoreComponentSource) -> &'static str {
    match source {
        ScoreComponentSource::LexicalBm25 => "rank_lex",
        ScoreComponentSource::SemanticFast => "rank_fast",
        ScoreComponentSource::SemanticQuality => "rank_qual",
        ScoreComponentSource::Rerank => "rank_rerank",
        ScoreComponentSource::HashControl => "rank_hash",
    }
}

fn source_label(source: ScoreComponentSource) -> String {
    match source {
        ScoreComponentSource::LexicalBm25 => "BM25".to_string(),
        ScoreComponentSource::SemanticFast => "FastSemantic".to_string(),
        ScoreComponentSource::SemanticQuality => "QualitySemantic".to_string(),
        ScoreComponentSource::Rerank => "Rerank".to_string(),
        ScoreComponentSource::HashControl => "HashControl".to_string(),
    }
}

fn domain_label(domain: PolicyDomain) -> String {
    match domain {
        PolicyDomain::QueryIntent => "Query Intent".to_string(),
        PolicyDomain::RetrievalBudget => "Retrieval Budget".to_string(),
        PolicyDomain::QueryExecution => "Execution Plan".to_string(),
        PolicyDomain::Degradation => "Degradation".to_string(),
        PolicyDomain::Discovery => "Discovery".to_string(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::explanation_payload::{
        FusionContext, PolicyDecisionExplanation, PolicyDomain, RankMovementSnapshot,
        RankingExplanation, ScoreComponentBreakdown, ScoreComponentSource,
    };
    use frankensearch_core::ExplanationPhase;

    fn sample_ranking() -> RankingExplanation {
        RankingExplanation {
            doc_id: "doc-42".to_string(),
            final_score: 0.0234,
            phase: ExplanationPhase::Refined,
            reason_code: "query.explain.attached".to_string(),
            confidence_per_mille: 920,
            rank_movement: Some(RankMovementSnapshot {
                initial_rank: 5,
                refined_rank: 2,
                delta: -3,
                reason: "quality semantic uplift".to_string(),
            }),
            fusion: Some(FusionContext {
                fused_score: 0.0234,
                lexical_rank: Some(3),
                semantic_rank: Some(7),
                lexical_score: Some(12.5),
                semantic_score: Some(0.77),
                in_both_sources: true,
                vector_generation_is_hash: false,
            }),
            components: vec![
                ScoreComponentBreakdown {
                    source: ScoreComponentSource::LexicalBm25,
                    summary: "LexicalBm25: matched 'rust'".to_string(),
                    raw_score: 12.5,
                    normalized_score: 0.9,
                    rrf_contribution: 0.0159,
                    weight: 0.3,
                    confidence_per_mille: 850,
                },
                ScoreComponentBreakdown {
                    source: ScoreComponentSource::SemanticFast,
                    summary: "FastSemantic: potion-128M cos=0.77".to_string(),
                    raw_score: 0.77,
                    normalized_score: 0.8,
                    rrf_contribution: 0.0149,
                    weight: 0.7,
                    confidence_per_mille: 800,
                },
            ],
        }
    }

    fn sample_policy() -> PolicyDecisionExplanation {
        let mut metadata = BTreeMap::new();
        metadata.insert("profile".to_string(), "Balanced".to_string());
        metadata.insert("limit".to_string(), "20".to_string());

        PolicyDecisionExplanation {
            domain: PolicyDomain::RetrievalBudget,
            decision: "Balanced".to_string(),
            reason_code: "query.budget.balanced".to_string(),
            confidence_per_mille: 850,
            summary: "budget limit 20 with balanced fanout".to_string(),
            metadata,
        }
    }

    fn sample_trace() -> TraceLink {
        TraceLink {
            trace_id: "01HQXYZREQUEST00000000000A".to_string(),
            event_id: "01HQXYZ1234567890ABCDEFGHIJ".to_string(),
            parent_event_id: None,
            claim_id: None,
            policy_id: None,
        }
    }

    // -- ExplainabilityLevel --

    #[test]
    fn level_cycles_through_all() {
        let mut level = ExplainabilityLevel::Novice;
        level = level.next();
        assert_eq!(level, ExplainabilityLevel::Intermediate);
        level = level.next();
        assert_eq!(level, ExplainabilityLevel::Expert);
        level = level.next();
        assert_eq!(level, ExplainabilityLevel::Novice);
    }

    #[test]
    fn level_feature_flags() {
        let novice = ExplainabilityLevel::Novice;
        assert!(!novice.show_components());
        assert!(!novice.show_equations());
        assert!(!novice.show_trace_links());
        assert_eq!(novice.max_policy_cards(), 2);

        let intermediate = ExplainabilityLevel::Intermediate;
        assert!(intermediate.show_components());
        assert!(!intermediate.show_equations());
        assert_eq!(intermediate.max_policy_cards(), 5);

        let expert = ExplainabilityLevel::Expert;
        assert!(expert.show_components());
        assert!(expert.show_equations());
        assert!(expert.show_trace_links());
        assert_eq!(expert.max_policy_cards(), usize::MAX);
    }

    #[test]
    fn level_display() {
        assert_eq!(ExplainabilityLevel::Novice.to_string(), "Novice");
        assert_eq!(ExplainabilityLevel::Expert.to_string(), "Expert");
    }

    // -- ConfidenceBadge --

    #[test]
    fn confidence_badge_classification() {
        assert_eq!(ConfidenceBadge::from_per_mille(950), ConfidenceBadge::High);
        assert_eq!(ConfidenceBadge::from_per_mille(900), ConfidenceBadge::High);
        assert_eq!(
            ConfidenceBadge::from_per_mille(899),
            ConfidenceBadge::Medium
        );
        assert_eq!(
            ConfidenceBadge::from_per_mille(600),
            ConfidenceBadge::Medium
        );
        assert_eq!(ConfidenceBadge::from_per_mille(599), ConfidenceBadge::Low);
        assert_eq!(ConfidenceBadge::from_per_mille(0), ConfidenceBadge::Low);
    }

    #[test]
    fn confidence_badge_display_and_color() {
        let high = ConfidenceBadge::High;
        assert_eq!(high.to_string(), "HIGH");
        assert_eq!(high.color_token(), "success");

        let low = ConfidenceBadge::Low;
        assert_eq!(low.to_string(), "LOW");
        assert_eq!(low.color_token(), "error");
    }

    // -- Ranking Card Builder --

    #[test]
    fn ranking_card_novice_has_no_components() {
        let ranking = sample_ranking();
        let card = build_ranking_card(&ranking, ExplainabilityLevel::Novice, None);

        assert_eq!(card.doc_id, "doc-42");
        assert!(card.components.is_empty());
        assert!(card.equation.is_none());
        assert!(card.trace.is_none());
        assert!(card.summary.contains("doc-42"));
        assert!(card.summary.contains("HIGH"));
        assert!(card.rank_movement.is_some()); // Always shown.
        assert!(card.fusion.is_some()); // Always shown.
    }

    #[test]
    fn ranking_card_intermediate_has_components_no_equations() {
        let ranking = sample_ranking();
        let card = build_ranking_card(&ranking, ExplainabilityLevel::Intermediate, None);

        assert_eq!(card.components.len(), 2);
        assert_eq!(card.components[0].source_label, "BM25");
        assert!(card.components[0].raw_score.is_none()); // No raw at intermediate.
        assert!(card.equation.is_none());
    }

    #[test]
    fn ranking_card_expert_has_everything() {
        let ranking = sample_ranking();
        let trace = sample_trace();
        let card = build_ranking_card(&ranking, ExplainabilityLevel::Expert, Some(&trace));

        assert_eq!(card.components.len(), 2);
        assert!(card.components[0].raw_score.is_some()); // Raw at expert.
        assert!(card.equation.is_some());
        assert!(card.equation.as_ref().unwrap().contains("rank_lex"));
        assert!(card.trace.is_some());
        assert_eq!(card.trace.as_ref().unwrap().trace_id, trace.trace_id);
    }

    // -- Policy Card Builder --

    #[test]
    fn policy_card_novice_has_no_metadata() {
        let policy = sample_policy();
        let card = build_policy_card(&policy, ExplainabilityLevel::Novice);

        assert_eq!(card.domain_label, "Retrieval Budget");
        assert_eq!(card.reason_code, "query.budget.balanced");
        assert!(card.metadata.is_empty());
    }

    #[test]
    fn policy_card_intermediate_has_metadata() {
        let policy = sample_policy();
        let card = build_policy_card(&policy, ExplainabilityLevel::Intermediate);

        assert!(!card.metadata.is_empty());
        assert!(card.metadata.iter().any(|(k, _)| k == "profile"));
    }

    // -- Rank Movement Row --

    #[test]
    fn rank_movement_improved() {
        let movement = RankMovementSnapshot {
            initial_rank: 5,
            refined_rank: 2,
            delta: -3,
            reason: "quality uplift".to_string(),
        };
        let row = build_rank_movement_row(&movement);
        assert!(row.direction_label.contains("improved"));
        assert!(row.direction_label.contains('3'));
    }

    #[test]
    fn rank_movement_dropped() {
        let movement = RankMovementSnapshot {
            initial_rank: 2,
            refined_rank: 5,
            delta: 3,
            reason: "rerank penalty".to_string(),
        };
        let row = build_rank_movement_row(&movement);
        assert!(row.direction_label.contains("dropped"));
    }

    #[test]
    fn rank_movement_unchanged() {
        let movement = RankMovementSnapshot {
            initial_rank: 3,
            refined_rank: 3,
            delta: 0,
            reason: "stable".to_string(),
        };
        let row = build_rank_movement_row(&movement);
        assert_eq!(row.direction_label, "unchanged");
    }

    // -- Fusion Row --

    #[test]
    fn fusion_row_both_sources() {
        let fusion = FusionContext {
            fused_score: 0.03,
            lexical_rank: Some(3),
            semantic_rank: Some(7),
            lexical_score: None,
            semantic_score: None,
            in_both_sources: true,
            vector_generation_is_hash: false,
        };
        let row = build_fusion_row(&fusion);
        assert!(row.overlap_label.contains("overlapping"));
        assert!(row.in_both_sources);
    }

    #[test]
    fn fusion_row_lexical_only() {
        let fusion = FusionContext {
            fused_score: 0.02,
            lexical_rank: Some(5),
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
            vector_generation_is_hash: false,
        };
        let row = build_fusion_row(&fusion);
        assert_eq!(row.overlap_label, "lexical-only");
    }

    #[test]
    fn fusion_row_semantic_only() {
        let fusion = FusionContext {
            fused_score: 0.01,
            lexical_rank: None,
            semantic_rank: Some(10),
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
            vector_generation_is_hash: false,
        };
        let row = build_fusion_row(&fusion);
        assert_eq!(row.overlap_label, "semantic-only");
    }

    #[test]
    fn fusion_row_hash_control_is_not_semantic() {
        let hash_only = FusionContext {
            fused_score: 0.01,
            lexical_rank: None,
            semantic_rank: Some(10),
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
            vector_generation_is_hash: true,
        };
        assert_eq!(
            build_fusion_row(&hash_only).overlap_label,
            "hash-control-only"
        );

        let both = FusionContext {
            fused_score: 0.03,
            lexical_rank: Some(3),
            semantic_rank: Some(7),
            lexical_score: None,
            semantic_score: None,
            in_both_sources: true,
            vector_generation_is_hash: true,
        };
        assert_eq!(
            build_fusion_row(&both).overlap_label,
            "overlapping (lexical + hash control)"
        );
        assert!(!build_fusion_row(&both).overlap_label.contains("semantic"));
    }

    // -- Ranking Equation --

    #[test]
    fn ranking_equation_includes_component_sources() {
        let ranking = sample_ranking();
        let equation = build_ranking_equation(&ranking);
        assert!(equation.contains("rank_lex"));
        assert!(equation.contains("rank_fast"));
        assert!(equation.contains("BM25"));
        assert!(equation.contains("FastSemantic"));
    }

    #[test]
    fn ranking_equation_empty_components() {
        let mut ranking = sample_ranking();
        ranking.components.clear();
        let equation = build_ranking_equation(&ranking);
        assert!(equation.starts_with("score ="));
        assert!(equation.contains("0.0234"));
    }

    // -- ExplainabilityScreenState --

    #[test]
    fn screen_state_default_is_novice() {
        let state = ExplainabilityScreenState::new();
        assert_eq!(state.level, ExplainabilityLevel::Novice);
        assert!(state.ranking_cards.is_empty());
        assert!(state.policy_cards.is_empty());
        assert!(!state.trace_expanded);
    }

    #[test]
    fn screen_state_cycle_level() {
        let mut state = ExplainabilityScreenState::new();
        state.cycle_level();
        assert_eq!(state.level, ExplainabilityLevel::Intermediate);
        state.cycle_level();
        assert_eq!(state.level, ExplainabilityLevel::Expert);
        state.cycle_level();
        assert_eq!(state.level, ExplainabilityLevel::Novice);
    }

    #[test]
    fn screen_state_set_rankings() {
        let mut state = ExplainabilityScreenState::new();
        let rankings = vec![sample_ranking()];
        state.set_ranking_explanations(&rankings, None);

        assert_eq!(state.ranking_cards.len(), 1);
        assert_eq!(state.ranking_list.total_items, 1);
        assert!(state.selected_ranking_card().is_some());
    }

    #[test]
    fn screen_state_set_policies_respects_level_limit() {
        let mut state = ExplainabilityScreenState::new();
        let policies: Vec<PolicyDecisionExplanation> = (0..5)
            .map(|i| PolicyDecisionExplanation {
                domain: PolicyDomain::QueryIntent,
                decision: format!("decision-{i}"),
                reason_code: format!("code.{i}"),
                confidence_per_mille: 800,
                summary: format!("summary {i}"),
                metadata: BTreeMap::new(),
            })
            .collect();

        // Novice: max 2.
        state.set_policy_decisions(&policies);
        assert_eq!(state.policy_cards.len(), 2);

        // Intermediate: max 5.
        state.level = ExplainabilityLevel::Intermediate;
        state.set_policy_decisions(&policies);
        assert_eq!(state.policy_cards.len(), 5);
    }

    #[test]
    fn screen_state_rebuild_updates_all_cards() {
        let mut state = ExplainabilityScreenState::new();
        let rankings = vec![sample_ranking()];
        let policies = vec![sample_policy()];
        let trace = sample_trace();

        state.rebuild_cards(&rankings, &policies, Some(&trace));
        assert_eq!(state.ranking_cards.len(), 1);
        assert_eq!(state.policy_cards.len(), 1);
        // Novice: no trace in card.
        assert!(state.ranking_cards[0].trace.is_none());

        // Switch to expert and rebuild.
        state.level = ExplainabilityLevel::Expert;
        state.rebuild_cards(&rankings, &policies, Some(&trace));
        assert!(state.ranking_cards[0].trace.is_some());
        assert!(state.ranking_cards[0].equation.is_some());
    }

    #[test]
    fn screen_state_trace_nodes() {
        let mut state = ExplainabilityScreenState::new();
        let nodes = vec![
            TraceNode {
                event_id: "evt-1".to_string(),
                parent_event_id: None,
                event_type: "decision".to_string(),
                reason_code: "query.phase.initial".to_string(),
                summary: "Initial search phase".to_string(),
                timestamp: "2026-02-14T12:00:00Z".to_string(),
                depth: 0,
            },
            TraceNode {
                event_id: "evt-2".to_string(),
                parent_event_id: Some("evt-1".to_string()),
                event_type: "transition".to_string(),
                reason_code: "query.phase.refined".to_string(),
                summary: "Refined with quality embedder".to_string(),
                timestamp: "2026-02-14T12:00:01Z".to_string(),
                depth: 1,
            },
        ];

        state.set_trace_nodes(nodes);
        assert_eq!(state.trace_nodes.len(), 2);
        assert_eq!(state.trace_list.total_items, 2);
    }

    // -- Domain Labels --

    #[test]
    fn domain_labels_are_human_readable() {
        assert_eq!(domain_label(PolicyDomain::QueryIntent), "Query Intent");
        assert_eq!(domain_label(PolicyDomain::Degradation), "Degradation");
        assert_eq!(domain_label(PolicyDomain::Discovery), "Discovery");
    }

    // -- Source Labels --

    #[test]
    fn source_labels_are_concise() {
        assert_eq!(source_label(ScoreComponentSource::LexicalBm25), "BM25");
        assert_eq!(
            source_label(ScoreComponentSource::SemanticFast),
            "FastSemantic"
        );
        assert_eq!(source_label(ScoreComponentSource::Rerank), "Rerank");
        assert_eq!(
            source_label(ScoreComponentSource::HashControl),
            "HashControl"
        );
    }
}
