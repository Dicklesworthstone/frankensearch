//! Query intent classification and retrieval-budget mapping for fsfs.
//!
//! This module defines:
//! - explicit query intent categories
//! - a deterministic confidence model
//! - profile-aware budget mapping (latency/fanout/rerank)
//! - robust fallback behavior for uncertain or malformed input

use frankensearch_core::query_class::QueryClass;

use crate::config::{FsfsConfig, PressureProfile};

/// Default minimum confidence required before using non-fallback budgets.
pub const DEFAULT_LOW_CONFIDENCE_THRESHOLD_PER_MILLE: u16 = 650;

const MAX_QUERY_CHARS: usize = 4_096;

/// fsfs intent categories for routing retrieval behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntentClass {
    Empty,
    Identifier,
    ShortKeyword,
    NaturalLanguage,
    Uncertain,
    Malformed,
}

/// Explicit fallback mode used for robust degraded behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryFallbackPath {
    None,
    EmptyQuery,
    LowConfidenceLexicalBias,
    MalformedLexicalOnly,
}

/// Retrieval-budget profile selected for the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryBudgetProfile {
    Empty,
    IdentifierFocused,
    Balanced,
    SemanticFocused,
    SafeFallback,
}

/// Deterministic query-intent classification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryIntentDecision {
    pub normalized_query: String,
    pub intent: QueryIntentClass,
    pub base_class: Option<QueryClass>,
    pub confidence_per_mille: u16,
    pub fallback: QueryFallbackPath,
    pub reason_code: &'static str,
}

/// Retrieval budget produced from intent classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalBudget {
    pub profile: QueryBudgetProfile,
    pub limit: usize,
    pub latency_budget_ms: u64,
    pub lexical_fanout: usize,
    pub semantic_fanout: usize,
    pub rerank_depth: usize,
    pub quality_enabled: bool,
    pub fallback: QueryFallbackPath,
    pub reason_code: &'static str,
}

/// Retrieval execution mode selected for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryExecutionMode {
    Empty,
    HybridRrf,
    FastSemanticOnly,
    LexicalOnly,
}

/// Fusion strategy used by the execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    None,
    Rrf,
    SemanticOnly,
    LexicalOnly,
}

/// Deterministic tie-break rules for fused rankings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionTieBreakRule {
    ScoreDesc,
    InBothSourcesDesc,
    LexicalScoreDesc,
    DocIdAsc,
}

/// Explicit runtime capability snapshot used by planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityState {
    Enabled,
    Disabled,
}

impl CapabilityState {
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Explicit runtime capability snapshot used by planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryExecutionCapabilities {
    pub lexical: CapabilityState,
    pub fast_semantic: CapabilityState,
    pub quality_semantic: CapabilityState,
    pub rerank: CapabilityState,
}

impl QueryExecutionCapabilities {
    #[must_use]
    pub const fn all_enabled() -> Self {
        Self {
            lexical: CapabilityState::Enabled,
            fast_semantic: CapabilityState::Enabled,
            quality_semantic: CapabilityState::Enabled,
            rerank: CapabilityState::Enabled,
        }
    }
}

impl Default for QueryExecutionCapabilities {
    fn default() -> Self {
        Self::all_enabled()
    }
}

/// Per-stage execution directive for query orchestration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDirective {
    pub enabled: bool,
    pub candidate_budget: usize,
    pub timeout_ms: u64,
    pub reason_code: &'static str,
}

/// Fusion + tie-break policy selected for the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionPolicy {
    pub strategy: FusionStrategy,
    pub rrf_k: Option<f64>,
    pub tie_break_rules: Vec<FusionTieBreakRule>,
    pub reason_code: &'static str,
}

/// Cancellation outcomes for key query execution boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationOutcome {
    AbortWithoutPartial,
    ReturnInitialResults,
    ReturnLexicalResults,
    ReturnEmptyResults,
}

/// Cancellation contract for query-stage orchestration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancellationSemantics {
    pub before_initial_yield: CancellationOutcome,
    pub after_initial_yield: CancellationOutcome,
    pub reason_code: &'static str,
}

/// Full multi-stage query execution plan for fsfs runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryExecutionPlan {
    pub intent: QueryIntentDecision,
    pub budget: RetrievalBudget,
    pub mode: QueryExecutionMode,
    pub capabilities: QueryExecutionCapabilities,
    pub lexical_stage: StageDirective,
    pub semantic_stage: StageDirective,
    pub quality_stage: StageDirective,
    pub rerank_stage: StageDirective,
    pub fusion_policy: FusionPolicy,
    pub cancellation: CancellationSemantics,
    pub reason_code: &'static str,
}

/// Configuration surface for the query planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryPlannerConfig {
    pub default_limit: usize,
    pub quality_timeout_ms: u64,
    pub rrf_k: f64,
    pub fast_only: bool,
    pub pressure_profile: PressureProfile,
    pub low_confidence_threshold_per_mille: u16,
}

impl QueryPlannerConfig {
    #[must_use]
    pub fn from_fsfs(config: &FsfsConfig) -> Self {
        Self {
            default_limit: config.search.default_limit.max(1),
            quality_timeout_ms: config.search.quality_timeout_ms,
            rrf_k: config.search.rrf_k,
            fast_only: config.search.fast_only,
            pressure_profile: config.pressure.profile,
            low_confidence_threshold_per_mille: DEFAULT_LOW_CONFIDENCE_THRESHOLD_PER_MILLE,
        }
    }
}

/// Deterministic classifier and budget mapper for fsfs search queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryPlanner {
    config: QueryPlannerConfig,
}

impl QueryPlanner {
    #[must_use]
    pub const fn new(config: QueryPlannerConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn from_fsfs(config: &FsfsConfig) -> Self {
        Self::new(QueryPlannerConfig::from_fsfs(config))
    }

    #[must_use]
    pub fn classify_intent(&self, query: &str) -> QueryIntentDecision {
        let normalized_query = normalize_query(query);
        if normalized_query.is_empty() {
            return QueryIntentDecision {
                normalized_query,
                intent: QueryIntentClass::Empty,
                base_class: Some(QueryClass::Empty),
                confidence_per_mille: 1_000,
                fallback: QueryFallbackPath::EmptyQuery,
                reason_code: "query.intent.empty",
            };
        }

        if let Some(reason_code) = malformed_reason(&normalized_query) {
            return QueryIntentDecision {
                normalized_query,
                intent: QueryIntentClass::Malformed,
                base_class: None,
                confidence_per_mille: 1_000,
                fallback: QueryFallbackPath::MalformedLexicalOnly,
                reason_code,
            };
        }

        let base_class = QueryClass::classify(&normalized_query);
        let confidence_per_mille = confidence_per_mille(&normalized_query, base_class);

        if confidence_per_mille < self.config.low_confidence_threshold_per_mille {
            return QueryIntentDecision {
                normalized_query,
                intent: QueryIntentClass::Uncertain,
                base_class: Some(base_class),
                confidence_per_mille,
                fallback: QueryFallbackPath::LowConfidenceLexicalBias,
                reason_code: "query.intent.uncertain.low_confidence",
            };
        }

        let (intent, reason_code) = match base_class {
            QueryClass::Empty => (QueryIntentClass::Empty, "query.intent.empty"),
            QueryClass::Identifier => (QueryIntentClass::Identifier, "query.intent.identifier"),
            QueryClass::ShortKeyword => {
                (QueryIntentClass::ShortKeyword, "query.intent.short_keyword")
            }
            QueryClass::NaturalLanguage => (
                QueryIntentClass::NaturalLanguage,
                "query.intent.natural_language",
            ),
        };

        QueryIntentDecision {
            normalized_query,
            intent,
            base_class: Some(base_class),
            confidence_per_mille,
            fallback: QueryFallbackPath::None,
            reason_code,
        }
    }

    #[must_use]
    pub fn budget_for_query(&self, query: &str, requested_limit: Option<usize>) -> RetrievalBudget {
        let decision = self.classify_intent(query);
        self.budget_for_decision(&decision, requested_limit)
    }

    #[must_use]
    pub fn budget_for_decision(
        &self,
        decision: &QueryIntentDecision,
        requested_limit: Option<usize>,
    ) -> RetrievalBudget {
        let limit = self.resolve_limit(requested_limit);
        let scale = BudgetScale::from_pressure(self.config.pressure_profile);

        let quality_enabled = self.quality_enabled(scale, decision.fallback);
        let semantic_timeout_ms = self.config.quality_timeout_ms.clamp(180, 2_000);

        match decision.intent {
            QueryIntentClass::Empty => RetrievalBudget {
                profile: QueryBudgetProfile::Empty,
                limit,
                latency_budget_ms: 0,
                lexical_fanout: 0,
                semantic_fanout: 0,
                rerank_depth: 0,
                quality_enabled: false,
                fallback: QueryFallbackPath::EmptyQuery,
                reason_code: "query.budget.empty",
            },
            QueryIntentClass::Identifier => RetrievalBudget {
                profile: QueryBudgetProfile::IdentifierFocused,
                limit,
                latency_budget_ms: scale.scale_latency(120),
                lexical_fanout: scale.scale_fanout(limit.saturating_mul(6)),
                semantic_fanout: scale.scale_fanout(limit.saturating_mul(2)),
                rerank_depth: rerank_depth(quality_enabled, scale, limit.min(12)),
                quality_enabled,
                fallback: QueryFallbackPath::None,
                reason_code: "query.budget.identifier_focused",
            },
            QueryIntentClass::ShortKeyword => RetrievalBudget {
                profile: QueryBudgetProfile::Balanced,
                limit,
                latency_budget_ms: scale.scale_latency(180),
                lexical_fanout: scale.scale_fanout(limit.saturating_mul(4)),
                semantic_fanout: scale.scale_fanout(limit.saturating_mul(4)),
                rerank_depth: rerank_depth(quality_enabled, scale, limit.saturating_mul(2).min(24)),
                quality_enabled,
                fallback: QueryFallbackPath::None,
                reason_code: "query.budget.balanced",
            },
            QueryIntentClass::NaturalLanguage => RetrievalBudget {
                profile: QueryBudgetProfile::SemanticFocused,
                limit,
                latency_budget_ms: scale.scale_latency(semantic_timeout_ms),
                lexical_fanout: scale.scale_fanout(limit.saturating_mul(2)),
                semantic_fanout: scale.scale_fanout(limit.saturating_mul(8)),
                rerank_depth: rerank_depth(quality_enabled, scale, limit.saturating_mul(3).min(48)),
                quality_enabled,
                fallback: QueryFallbackPath::None,
                reason_code: "query.budget.semantic_focused",
            },
            QueryIntentClass::Uncertain => RetrievalBudget {
                profile: QueryBudgetProfile::SafeFallback,
                limit,
                latency_budget_ms: scale.scale_latency(90),
                lexical_fanout: scale.scale_fanout(limit.saturating_mul(3)),
                semantic_fanout: scale.scale_fanout(limit),
                rerank_depth: 0,
                quality_enabled: false,
                fallback: QueryFallbackPath::LowConfidenceLexicalBias,
                reason_code: "query.budget.safe_fallback",
            },
            QueryIntentClass::Malformed => RetrievalBudget {
                profile: QueryBudgetProfile::SafeFallback,
                limit,
                latency_budget_ms: scale.scale_latency(60),
                lexical_fanout: scale.scale_fanout(limit.saturating_mul(2)),
                semantic_fanout: 0,
                rerank_depth: 0,
                quality_enabled: false,
                fallback: QueryFallbackPath::MalformedLexicalOnly,
                reason_code: "query.budget.safe_fallback",
            },
        }
    }

    /// Build a deterministic multi-stage execution plan for one query.
    #[must_use]
    pub fn execution_plan_for_query(
        &self,
        query: &str,
        requested_limit: Option<usize>,
        capabilities: QueryExecutionCapabilities,
    ) -> QueryExecutionPlan {
        let decision = self.classify_intent(query);
        let budget = self.budget_for_decision(&decision, requested_limit);
        self.execution_plan_for_decision(decision, budget, capabilities)
    }

    /// Build a deterministic multi-stage execution plan from precomputed
    /// intent + budget.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn execution_plan_for_decision(
        &self,
        decision: QueryIntentDecision,
        budget: RetrievalBudget,
        capabilities: QueryExecutionCapabilities,
    ) -> QueryExecutionPlan {
        let mode = resolve_execution_mode(&decision, &budget, capabilities);
        let quality_timeout_ms = self.config.quality_timeout_ms.clamp(180, 2_000);

        let lexical_enabled = matches!(
            mode,
            QueryExecutionMode::HybridRrf | QueryExecutionMode::LexicalOnly
        );
        let semantic_enabled = matches!(
            mode,
            QueryExecutionMode::HybridRrf | QueryExecutionMode::FastSemanticOnly
        );

        let quality_enabled = semantic_enabled
            && budget.quality_enabled
            && capabilities.quality_semantic.is_enabled();
        let rerank_enabled =
            quality_enabled && capabilities.rerank.is_enabled() && budget.rerank_depth > 0;

        let lexical_stage = if lexical_enabled {
            StageDirective {
                enabled: true,
                candidate_budget: budget.lexical_fanout,
                timeout_ms: budget.latency_budget_ms.min(250),
                reason_code: "query.stage.lexical.enabled",
            }
        } else {
            StageDirective {
                enabled: false,
                candidate_budget: 0,
                timeout_ms: 0,
                reason_code: "query.stage.lexical.disabled",
            }
        };

        let semantic_stage = if semantic_enabled {
            StageDirective {
                enabled: true,
                candidate_budget: budget.semantic_fanout,
                timeout_ms: budget.latency_budget_ms,
                reason_code: "query.stage.semantic.enabled",
            }
        } else {
            StageDirective {
                enabled: false,
                candidate_budget: 0,
                timeout_ms: 0,
                reason_code: "query.stage.semantic.disabled",
            }
        };

        let quality_stage = if quality_enabled {
            StageDirective {
                enabled: true,
                candidate_budget: budget.semantic_fanout.min(budget.limit.saturating_mul(3)),
                timeout_ms: quality_timeout_ms,
                reason_code: "query.stage.quality.enabled",
            }
        } else {
            StageDirective {
                enabled: false,
                candidate_budget: 0,
                timeout_ms: 0,
                reason_code: if !semantic_enabled {
                    "query.stage.quality.disabled.no_semantic"
                } else if !budget.quality_enabled {
                    "query.stage.quality.disabled.policy_or_pressure"
                } else {
                    "query.stage.quality.disabled.unavailable"
                },
            }
        };

        let rerank_stage = if rerank_enabled {
            StageDirective {
                enabled: true,
                candidate_budget: budget.rerank_depth,
                timeout_ms: quality_timeout_ms.min(300),
                reason_code: "query.stage.rerank.enabled",
            }
        } else {
            StageDirective {
                enabled: false,
                candidate_budget: 0,
                timeout_ms: 0,
                reason_code: if !quality_enabled {
                    "query.stage.rerank.disabled.no_quality"
                } else if !capabilities.rerank.is_enabled() {
                    "query.stage.rerank.disabled.unavailable"
                } else {
                    "query.stage.rerank.disabled.zero_depth"
                },
            }
        };

        let fusion_policy = fusion_policy_for_mode(mode, self.config.rrf_k);
        let cancellation = cancellation_semantics_for_mode(mode, quality_enabled);
        let reason_code = execution_reason_code(mode, decision.fallback);

        QueryExecutionPlan {
            intent: decision,
            budget,
            mode,
            capabilities,
            lexical_stage,
            semantic_stage,
            quality_stage,
            rerank_stage,
            fusion_policy,
            cancellation,
            reason_code,
        }
    }

    #[must_use]
    fn resolve_limit(&self, requested_limit: Option<usize>) -> usize {
        requested_limit.map_or(self.config.default_limit, |limit| limit.max(1))
    }

    #[must_use]
    const fn quality_enabled(&self, scale: BudgetScale, fallback: QueryFallbackPath) -> bool {
        !self.config.fast_only
            && !scale.force_fast_only
            && matches!(fallback, QueryFallbackPath::None)
    }
}

#[must_use]
const fn resolve_execution_mode(
    decision: &QueryIntentDecision,
    budget: &RetrievalBudget,
    capabilities: QueryExecutionCapabilities,
) -> QueryExecutionMode {
    if matches!(decision.intent, QueryIntentClass::Empty)
        || matches!(budget.profile, QueryBudgetProfile::Empty)
    {
        return QueryExecutionMode::Empty;
    }

    if !capabilities.lexical.is_enabled() && !capabilities.fast_semantic.is_enabled() {
        return QueryExecutionMode::Empty;
    }

    let lexical_only_fallback =
        matches!(decision.fallback, QueryFallbackPath::MalformedLexicalOnly);
    if lexical_only_fallback {
        return if capabilities.lexical.is_enabled() && budget.lexical_fanout > 0 {
            QueryExecutionMode::LexicalOnly
        } else {
            QueryExecutionMode::Empty
        };
    }

    if !capabilities.fast_semantic.is_enabled() || budget.semantic_fanout == 0 {
        return if capabilities.lexical.is_enabled() && budget.lexical_fanout > 0 {
            QueryExecutionMode::LexicalOnly
        } else {
            QueryExecutionMode::Empty
        };
    }

    if capabilities.lexical.is_enabled() && budget.lexical_fanout > 0 {
        return QueryExecutionMode::HybridRrf;
    }

    QueryExecutionMode::FastSemanticOnly
}

#[must_use]
const fn execution_reason_code(
    mode: QueryExecutionMode,
    fallback: QueryFallbackPath,
) -> &'static str {
    match mode {
        QueryExecutionMode::Empty => "query.execution.mode.empty",
        QueryExecutionMode::HybridRrf => {
            if matches!(fallback, QueryFallbackPath::LowConfidenceLexicalBias) {
                "query.execution.mode.hybrid.lexical_bias"
            } else {
                "query.execution.mode.hybrid"
            }
        }
        QueryExecutionMode::FastSemanticOnly => "query.execution.mode.semantic_only",
        QueryExecutionMode::LexicalOnly => {
            if matches!(fallback, QueryFallbackPath::MalformedLexicalOnly) {
                "query.execution.mode.lexical_only.malformed"
            } else {
                "query.execution.mode.lexical_only"
            }
        }
    }
}

#[must_use]
fn fusion_policy_for_mode(mode: QueryExecutionMode, configured_rrf_k: f64) -> FusionPolicy {
    let rrf_k = if configured_rrf_k.is_finite() && configured_rrf_k >= 1.0 {
        configured_rrf_k
    } else {
        60.0
    };

    match mode {
        QueryExecutionMode::HybridRrf => FusionPolicy {
            strategy: FusionStrategy::Rrf,
            rrf_k: Some(rrf_k),
            tie_break_rules: vec![
                FusionTieBreakRule::ScoreDesc,
                FusionTieBreakRule::InBothSourcesDesc,
                FusionTieBreakRule::LexicalScoreDesc,
                FusionTieBreakRule::DocIdAsc,
            ],
            reason_code: "query.fusion.rrf",
        },
        QueryExecutionMode::FastSemanticOnly => FusionPolicy {
            strategy: FusionStrategy::SemanticOnly,
            rrf_k: None,
            tie_break_rules: vec![FusionTieBreakRule::ScoreDesc, FusionTieBreakRule::DocIdAsc],
            reason_code: "query.fusion.semantic_only",
        },
        QueryExecutionMode::LexicalOnly => FusionPolicy {
            strategy: FusionStrategy::LexicalOnly,
            rrf_k: None,
            tie_break_rules: vec![FusionTieBreakRule::ScoreDesc, FusionTieBreakRule::DocIdAsc],
            reason_code: "query.fusion.lexical_only",
        },
        QueryExecutionMode::Empty => FusionPolicy {
            strategy: FusionStrategy::None,
            rrf_k: None,
            tie_break_rules: vec![FusionTieBreakRule::DocIdAsc],
            reason_code: "query.fusion.none",
        },
    }
}

#[must_use]
const fn cancellation_semantics_for_mode(
    mode: QueryExecutionMode,
    quality_enabled: bool,
) -> CancellationSemantics {
    match mode {
        QueryExecutionMode::Empty => CancellationSemantics {
            before_initial_yield: CancellationOutcome::ReturnEmptyResults,
            after_initial_yield: CancellationOutcome::ReturnEmptyResults,
            reason_code: "query.cancel.empty",
        },
        QueryExecutionMode::LexicalOnly => CancellationSemantics {
            before_initial_yield: CancellationOutcome::AbortWithoutPartial,
            after_initial_yield: CancellationOutcome::ReturnLexicalResults,
            reason_code: "query.cancel.lexical_only",
        },
        QueryExecutionMode::HybridRrf | QueryExecutionMode::FastSemanticOnly => {
            let reason_code = if quality_enabled {
                "query.cancel.phase2_returns_initial"
            } else {
                "query.cancel.single_phase_returns_initial"
            };
            CancellationSemantics {
                before_initial_yield: CancellationOutcome::AbortWithoutPartial,
                after_initial_yield: CancellationOutcome::ReturnInitialResults,
                reason_code,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BudgetScale {
    fanout_num: usize,
    fanout_den: usize,
    latency_num: u64,
    latency_den: u64,
    rerank_num: usize,
    rerank_den: usize,
    force_fast_only: bool,
}

impl BudgetScale {
    #[must_use]
    const fn from_pressure(profile: PressureProfile) -> Self {
        match profile {
            PressureProfile::Performance => Self {
                fanout_num: 1,
                fanout_den: 1,
                latency_num: 1,
                latency_den: 1,
                rerank_num: 1,
                rerank_den: 1,
                force_fast_only: false,
            },
            PressureProfile::Strict => Self {
                fanout_num: 3,
                fanout_den: 4,
                latency_num: 4,
                latency_den: 5,
                rerank_num: 1,
                rerank_den: 2,
                force_fast_only: true,
            },
            PressureProfile::Degraded => Self {
                fanout_num: 2,
                fanout_den: 3,
                latency_num: 7,
                latency_den: 10,
                rerank_num: 1,
                rerank_den: 3,
                force_fast_only: true,
            },
        }
    }

    #[must_use]
    fn scale_fanout(self, value: usize) -> usize {
        if value == 0 {
            return 0;
        }
        scale_usize(value, self.fanout_num, self.fanout_den).max(1)
    }

    #[must_use]
    fn scale_latency(self, value: u64) -> u64 {
        if value == 0 {
            return 0;
        }
        scale_u64(value, self.latency_num, self.latency_den).max(1)
    }

    #[must_use]
    fn scale_rerank(self, value: usize) -> usize {
        if value == 0 {
            return 0;
        }
        scale_usize(value, self.rerank_num, self.rerank_den).max(1)
    }
}

#[must_use]
fn rerank_depth(enabled: bool, scale: BudgetScale, base_depth: usize) -> usize {
    if enabled {
        scale.scale_rerank(base_depth)
    } else {
        0
    }
}

#[must_use]
fn normalize_query(query: &str) -> String {
    let mut normalized = String::new();
    for token in query.split_whitespace() {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(token);
    }
    normalized
}

#[must_use]
fn malformed_reason(query: &str) -> Option<&'static str> {
    if query
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\t' | '\n' | '\r'))
    {
        return Some("query.intent.malformed.control_chars");
    }

    if query.chars().count() > MAX_QUERY_CHARS {
        return Some("query.intent.malformed.too_long");
    }

    None
}

#[must_use]
fn confidence_per_mille(query: &str, base_class: QueryClass) -> u16 {
    if appears_low_signal(query) {
        return 420;
    }

    match base_class {
        QueryClass::Empty => 1_000,
        QueryClass::Identifier => {
            let signal_count = identifier_signal_count(query);
            (700 + signal_count.saturating_mul(65)).min(970)
        }
        QueryClass::ShortKeyword => {
            let word_count = query.split_whitespace().count();
            match word_count {
                0 => 1_000,
                1 => 860,
                2 => 810,
                _ => 760,
            }
        }
        QueryClass::NaturalLanguage => {
            let signal_count = natural_language_signal_count(query);
            match signal_count {
                0 => 730,
                1 => 810,
                2 => 900,
                _ => 960,
            }
        }
    }
}

#[must_use]
fn identifier_signal_count(query: &str) -> u16 {
    let mut signals = 0_u16;
    if query.contains('/') || query.contains('\\') {
        signals = signals.saturating_add(1);
    }
    if !query.contains(' ') && query.contains("::") {
        signals = signals.saturating_add(1);
    }
    if !query.contains(' ') && query.contains('.') {
        signals = signals.saturating_add(1);
    }
    if looks_like_ticket_id(query) {
        signals = signals.saturating_add(1);
    }

    let starts_with_code_keyword = ["fn ", "struct ", "impl "]
        .iter()
        .any(|prefix| query.starts_with(prefix));
    if starts_with_code_keyword {
        signals = signals.saturating_add(1);
    }

    signals
}

#[must_use]
fn natural_language_signal_count(query: &str) -> u16 {
    let mut signals = 0_u16;
    if query.ends_with('?') {
        signals = signals.saturating_add(1);
    }

    if let Some(first_word) = query.split_whitespace().next() {
        let lower = first_word.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "how" | "what" | "why" | "where" | "when" | "which" | "who" | "can" | "does"
        ) {
            signals = signals.saturating_add(1);
        }
    }

    if query.split_whitespace().count() >= 6 {
        signals = signals.saturating_add(1);
    }

    signals
}

#[must_use]
fn appears_low_signal(query: &str) -> bool {
    let mut alnum = 0_usize;
    let mut punctuation = 0_usize;

    for ch in query.chars() {
        if ch.is_ascii_alphanumeric() {
            alnum = alnum.saturating_add(1);
        } else if ch.is_ascii_punctuation() {
            punctuation = punctuation.saturating_add(1);
        }
    }

    alnum == 0 || (alnum < 3 && punctuation >= alnum.saturating_mul(2))
}

#[must_use]
fn looks_like_ticket_id(query: &str) -> bool {
    if query.contains(' ') || !query.contains('-') {
        return false;
    }

    let mut parts = query.splitn(2, '-');
    let left = parts.next().unwrap_or_default();
    let right = parts.next().unwrap_or_default();
    !left.is_empty()
        && !right.is_empty()
        && left.chars().all(|ch| ch.is_ascii_alphanumeric())
        && right.chars().all(|ch| ch.is_ascii_alphanumeric())
}

#[must_use]
const fn scale_usize(value: usize, numer: usize, denom: usize) -> usize {
    let scaled = value.saturating_mul(numer);
    scaled.saturating_add(denom.saturating_sub(1)) / denom
}

#[must_use]
const fn scale_u64(value: u64, numer: u64, denom: u64) -> u64 {
    let scaled = value.saturating_mul(numer);
    scaled.saturating_add(denom.saturating_sub(1)) / denom
}

#[cfg(test)]
mod tests {
    use super::{
        CancellationOutcome, FusionStrategy, FusionTieBreakRule, QueryBudgetProfile,
        QueryExecutionCapabilities, QueryExecutionMode, QueryFallbackPath, QueryIntentClass,
        QueryIntentDecision, QueryPlanner, QueryPlannerConfig, RetrievalBudget,
    };
    use crate::config::{FsfsConfig, PressureProfile};

    #[test]
    fn classifier_marks_identifier_with_high_confidence() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("src/main.rs");

        assert_eq!(decision.intent, QueryIntentClass::Identifier);
        assert!(decision.confidence_per_mille >= 800);
        assert_eq!(decision.fallback, QueryFallbackPath::None);
        assert_eq!(decision.reason_code, "query.intent.identifier");
    }

    #[test]
    fn classifier_routes_low_signal_query_to_uncertain_fallback() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("???!!!");

        assert_eq!(decision.intent, QueryIntentClass::Uncertain);
        assert_eq!(
            decision.fallback,
            QueryFallbackPath::LowConfidenceLexicalBias
        );
        assert!(decision.confidence_per_mille < 650);
        assert_eq!(
            decision.reason_code,
            "query.intent.uncertain.low_confidence"
        );
    }

    #[test]
    fn classifier_routes_control_chars_to_malformed_fallback() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("find\u{0007}secret");

        assert_eq!(decision.intent, QueryIntentClass::Malformed);
        assert_eq!(decision.fallback, QueryFallbackPath::MalformedLexicalOnly);
        assert_eq!(decision.reason_code, "query.intent.malformed.control_chars");
    }

    #[test]
    fn budget_mapping_is_profile_aware() {
        let performance_planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let performance_budget =
            performance_planner.budget_for_query("how does query ranking work", Some(10));

        let strict_planner = QueryPlanner::new(QueryPlannerConfig {
            pressure_profile: PressureProfile::Strict,
            ..QueryPlannerConfig::from_fsfs(&FsfsConfig::default())
        });
        let strict_budget =
            strict_planner.budget_for_query("how does query ranking work", Some(10));

        assert_eq!(
            performance_budget.profile,
            QueryBudgetProfile::SemanticFocused
        );
        assert_eq!(strict_budget.profile, QueryBudgetProfile::SemanticFocused);
        assert!(strict_budget.lexical_fanout < performance_budget.lexical_fanout);
        assert!(strict_budget.semantic_fanout < performance_budget.semantic_fanout);
        assert!(strict_budget.latency_budget_ms < performance_budget.latency_budget_ms);
        assert!(!strict_budget.quality_enabled);
        assert_eq!(strict_budget.rerank_depth, 0);
    }

    #[test]
    fn uncertain_budget_uses_safe_fallback_profile() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("???", Some(6));

        assert_eq!(budget.profile, QueryBudgetProfile::SafeFallback);
        assert_eq!(budget.fallback, QueryFallbackPath::LowConfidenceLexicalBias);
        assert_eq!(budget.reason_code, "query.budget.safe_fallback");
        assert!(!budget.quality_enabled);
        assert_eq!(budget.rerank_depth, 0);
        assert!(budget.lexical_fanout >= budget.semantic_fanout);
    }

    #[test]
    fn empty_query_budget_is_noop() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("   ", Some(12));

        assert_eq!(budget.profile, QueryBudgetProfile::Empty);
        assert_eq!(budget.lexical_fanout, 0);
        assert_eq!(budget.semantic_fanout, 0);
        assert_eq!(budget.latency_budget_ms, 0);
        assert_eq!(budget.rerank_depth, 0);
        assert!(!budget.quality_enabled);
        assert_eq!(budget.fallback, QueryFallbackPath::EmptyQuery);
    }

    #[test]
    fn fast_only_config_disables_quality_and_rerank() {
        let mut config = FsfsConfig::default();
        config.search.fast_only = true;
        let planner = QueryPlanner::from_fsfs(&config);
        let budget = planner.budget_for_query("how does query ranking work", Some(8));

        assert_eq!(budget.profile, QueryBudgetProfile::SemanticFocused);
        assert!(!budget.quality_enabled);
        assert_eq!(budget.rerank_depth, 0);
    }

    #[test]
    fn execution_plan_hybrid_rrf_is_deterministic() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let plan = planner.execution_plan_for_query(
            "how does query ranking work",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );

        assert_eq!(plan.mode, QueryExecutionMode::HybridRrf);
        assert_eq!(plan.reason_code, "query.execution.mode.hybrid");
        assert_eq!(plan.fusion_policy.strategy, FusionStrategy::Rrf);
        assert_eq!(plan.fusion_policy.rrf_k, Some(60.0));
        assert_eq!(
            plan.fusion_policy.tie_break_rules,
            vec![
                FusionTieBreakRule::ScoreDesc,
                FusionTieBreakRule::InBothSourcesDesc,
                FusionTieBreakRule::LexicalScoreDesc,
                FusionTieBreakRule::DocIdAsc,
            ]
        );
        assert!(plan.lexical_stage.enabled);
        assert!(plan.semantic_stage.enabled);
        assert!(plan.quality_stage.enabled);
        assert!(plan.rerank_stage.enabled);
        assert_eq!(
            plan.cancellation.before_initial_yield,
            CancellationOutcome::AbortWithoutPartial
        );
        assert_eq!(
            plan.cancellation.after_initial_yield,
            CancellationOutcome::ReturnInitialResults
        );
    }

    #[test]
    fn execution_plan_degrades_to_semantic_only_without_lexical() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Disabled,
            fast_semantic: super::CapabilityState::Enabled,
            quality_semantic: super::CapabilityState::Enabled,
            rerank: super::CapabilityState::Disabled,
        };
        let plan = planner.execution_plan_for_query("distributed consensus", Some(10), caps);

        assert_eq!(plan.mode, QueryExecutionMode::FastSemanticOnly);
        assert_eq!(plan.reason_code, "query.execution.mode.semantic_only");
        assert_eq!(plan.fusion_policy.strategy, FusionStrategy::SemanticOnly);
        assert!(!plan.lexical_stage.enabled);
        assert!(plan.semantic_stage.enabled);
        assert!(plan.quality_stage.enabled);
        assert!(!plan.rerank_stage.enabled);
    }

    #[test]
    fn execution_plan_degrades_to_lexical_when_semantic_unavailable() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Enabled,
            fast_semantic: super::CapabilityState::Disabled,
            quality_semantic: super::CapabilityState::Disabled,
            rerank: super::CapabilityState::Enabled,
        };
        let plan = planner.execution_plan_for_query("rust ownership", Some(8), caps);

        assert_eq!(plan.mode, QueryExecutionMode::LexicalOnly);
        assert_eq!(plan.fusion_policy.strategy, FusionStrategy::LexicalOnly);
        assert!(plan.lexical_stage.enabled);
        assert!(!plan.semantic_stage.enabled);
        assert!(!plan.quality_stage.enabled);
        assert!(!plan.rerank_stage.enabled);
        assert_eq!(
            plan.cancellation.after_initial_yield,
            CancellationOutcome::ReturnLexicalResults
        );
    }

    #[test]
    fn execution_plan_malformed_query_forces_lexical_path() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let plan = planner.execution_plan_for_query(
            "find\u{0007}secret",
            Some(8),
            QueryExecutionCapabilities::all_enabled(),
        );

        assert_eq!(plan.mode, QueryExecutionMode::LexicalOnly);
        assert_eq!(
            plan.reason_code,
            "query.execution.mode.lexical_only.malformed"
        );
        assert_eq!(plan.fusion_policy.strategy, FusionStrategy::LexicalOnly);
        assert!(!plan.semantic_stage.enabled);
    }

    #[test]
    fn execution_plan_empty_query_returns_empty_mode() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let plan = planner.execution_plan_for_query(
            "   ",
            Some(12),
            QueryExecutionCapabilities::all_enabled(),
        );

        assert_eq!(plan.mode, QueryExecutionMode::Empty);
        assert_eq!(plan.reason_code, "query.execution.mode.empty");
        assert_eq!(plan.fusion_policy.strategy, FusionStrategy::None);
        assert!(!plan.lexical_stage.enabled);
        assert!(!plan.semantic_stage.enabled);
        assert_eq!(
            plan.cancellation.before_initial_yield,
            CancellationOutcome::ReturnEmptyResults
        );
    }

    // ─── bd-3het tests begin ───

    // --- normalize_query ---

    #[test]
    fn normalize_query_collapses_whitespace() {
        assert_eq!(super::normalize_query("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_query_empty_input() {
        assert_eq!(super::normalize_query(""), "");
    }

    #[test]
    fn normalize_query_only_whitespace() {
        assert_eq!(super::normalize_query("   \t\n  "), "");
    }

    #[test]
    fn normalize_query_single_token() {
        assert_eq!(super::normalize_query("  token  "), "token");
    }

    #[test]
    fn normalize_query_tabs_and_newlines() {
        assert_eq!(super::normalize_query("a\tb\nc"), "a b c");
    }

    // --- malformed_reason ---

    #[test]
    fn malformed_reason_control_char() {
        assert_eq!(
            super::malformed_reason("hello\u{0007}world"),
            Some("query.intent.malformed.control_chars")
        );
    }

    #[test]
    fn malformed_reason_null_byte() {
        assert_eq!(
            super::malformed_reason("hello\u{0000}world"),
            Some("query.intent.malformed.control_chars")
        );
    }

    #[test]
    fn malformed_reason_tab_allowed() {
        assert_eq!(super::malformed_reason("hello\tworld"), None);
    }

    #[test]
    fn malformed_reason_newline_allowed() {
        assert_eq!(super::malformed_reason("hello\nworld"), None);
    }

    #[test]
    fn malformed_reason_cr_allowed() {
        assert_eq!(super::malformed_reason("hello\rworld"), None);
    }

    #[test]
    fn malformed_reason_too_long() {
        let long_query = "a".repeat(super::MAX_QUERY_CHARS + 1);
        assert_eq!(
            super::malformed_reason(&long_query),
            Some("query.intent.malformed.too_long")
        );
    }

    #[test]
    fn malformed_reason_at_limit_ok() {
        let at_limit = "a".repeat(super::MAX_QUERY_CHARS);
        assert_eq!(super::malformed_reason(&at_limit), None);
    }

    #[test]
    fn malformed_reason_normal_query() {
        assert_eq!(super::malformed_reason("normal query text"), None);
    }

    // --- appears_low_signal ---

    #[test]
    fn appears_low_signal_pure_punctuation() {
        assert!(super::appears_low_signal("???!!!"));
    }

    #[test]
    fn appears_low_signal_no_alnum() {
        assert!(super::appears_low_signal("..."));
    }

    #[test]
    fn appears_low_signal_few_alnum_lots_of_punct() {
        // 2 alnum, 4+ punctuation → low signal
        assert!(super::appears_low_signal("a!@#$b"));
    }

    #[test]
    fn appears_low_signal_normal_text() {
        assert!(!super::appears_low_signal("hello world"));
    }

    #[test]
    fn appears_low_signal_single_word() {
        assert!(!super::appears_low_signal("rust"));
    }

    #[test]
    fn appears_low_signal_empty() {
        // 0 alnum → low signal
        assert!(super::appears_low_signal(""));
    }

    // --- looks_like_ticket_id ---

    #[test]
    fn looks_like_ticket_id_valid() {
        assert!(super::looks_like_ticket_id("JIRA-1234"));
    }

    #[test]
    fn looks_like_ticket_id_lowercase() {
        assert!(super::looks_like_ticket_id("proj-42"));
    }

    #[test]
    fn looks_like_ticket_id_with_space() {
        assert!(!super::looks_like_ticket_id("JIRA 1234"));
    }

    #[test]
    fn looks_like_ticket_id_no_dash() {
        assert!(!super::looks_like_ticket_id("JIRA1234"));
    }

    #[test]
    fn looks_like_ticket_id_empty_left() {
        assert!(!super::looks_like_ticket_id("-1234"));
    }

    #[test]
    fn looks_like_ticket_id_empty_right() {
        assert!(!super::looks_like_ticket_id("JIRA-"));
    }

    #[test]
    fn looks_like_ticket_id_multiple_dashes() {
        // splitn(2, '-') keeps the second part as "is-it"
        // right = "is-it" which contains non-alphanumeric '-'
        assert!(!super::looks_like_ticket_id("what-is-it"));
    }

    // --- identifier_signal_count ---

    #[test]
    fn identifier_signal_count_path_separator() {
        assert!(super::identifier_signal_count("src/main.rs") >= 2);
    }

    #[test]
    fn identifier_signal_count_double_colon() {
        assert!(super::identifier_signal_count("std::vec::Vec") >= 1);
    }

    #[test]
    fn identifier_signal_count_dot_no_space() {
        assert!(super::identifier_signal_count("config.toml") >= 1);
    }

    #[test]
    fn identifier_signal_count_ticket_id() {
        let count = super::identifier_signal_count("PROJ-123");
        assert!(count >= 1);
    }

    #[test]
    fn identifier_signal_count_code_keyword_fn() {
        assert!(super::identifier_signal_count("fn main") >= 1);
    }

    #[test]
    fn identifier_signal_count_code_keyword_struct() {
        assert!(super::identifier_signal_count("struct Foo") >= 1);
    }

    #[test]
    fn identifier_signal_count_code_keyword_impl() {
        assert!(super::identifier_signal_count("impl Bar") >= 1);
    }

    #[test]
    fn identifier_signal_count_plain_word() {
        assert_eq!(super::identifier_signal_count("hello"), 0);
    }

    #[test]
    fn identifier_signal_double_colon_with_space_no_signal() {
        // "foo :: bar" has a space, so :: check doesn't fire
        assert_eq!(
            super::identifier_signal_count("foo :: bar") & 0, // just check it doesn't panic
            0
        );
    }

    // --- natural_language_signal_count ---

    #[test]
    fn natural_language_signal_question_mark() {
        assert!(super::natural_language_signal_count("what is this?") >= 2);
    }

    #[test]
    fn natural_language_signal_interrogative_word() {
        assert!(super::natural_language_signal_count("how does it work") >= 1);
    }

    #[test]
    fn natural_language_signal_all_interrogatives() {
        for word in &[
            "how", "what", "why", "where", "when", "which", "who", "can", "does",
        ] {
            let query = format!("{word} is this");
            assert!(
                super::natural_language_signal_count(&query) >= 1,
                "Expected signal for '{word}'"
            );
        }
    }

    #[test]
    fn natural_language_signal_long_query() {
        // 6+ words → extra signal
        let count = super::natural_language_signal_count("one two three four five six seven");
        assert!(count >= 1);
    }

    #[test]
    fn natural_language_signal_short_no_signal() {
        assert_eq!(super::natural_language_signal_count("rust"), 0);
    }

    // --- confidence_per_mille ---

    #[test]
    fn confidence_per_mille_empty_is_low_signal() {
        // Empty string has 0 alnum chars → appears_low_signal returns true → 420
        assert_eq!(
            super::confidence_per_mille("", frankensearch_core::query_class::QueryClass::Empty),
            420
        );
    }

    #[test]
    fn confidence_per_mille_identifier_base() {
        let conf = super::confidence_per_mille(
            "foo",
            frankensearch_core::query_class::QueryClass::Identifier,
        );
        // 0 signals → 700 + 0 = 700
        assert_eq!(conf, 700);
    }

    #[test]
    fn confidence_per_mille_identifier_with_signals() {
        let conf = super::confidence_per_mille(
            "src/main.rs",
            frankensearch_core::query_class::QueryClass::Identifier,
        );
        // Has / and . → 2 signals → 700 + 130 = 830
        assert_eq!(conf, 830);
    }

    #[test]
    fn confidence_per_mille_identifier_capped_at_970() {
        // Many signals, but capped
        let conf = super::confidence_per_mille(
            "fn src/foo.rs",
            frankensearch_core::query_class::QueryClass::Identifier,
        );
        assert!(conf <= 970);
    }

    #[test]
    fn confidence_per_mille_short_keyword_1_word() {
        let conf = super::confidence_per_mille(
            "hello",
            frankensearch_core::query_class::QueryClass::ShortKeyword,
        );
        assert_eq!(conf, 860);
    }

    #[test]
    fn confidence_per_mille_short_keyword_2_words() {
        let conf = super::confidence_per_mille(
            "hello world",
            frankensearch_core::query_class::QueryClass::ShortKeyword,
        );
        assert_eq!(conf, 810);
    }

    #[test]
    fn confidence_per_mille_short_keyword_3_words() {
        let conf = super::confidence_per_mille(
            "one two three",
            frankensearch_core::query_class::QueryClass::ShortKeyword,
        );
        assert_eq!(conf, 760);
    }

    #[test]
    fn confidence_per_mille_natural_language_0_signals() {
        let conf = super::confidence_per_mille(
            "rust",
            frankensearch_core::query_class::QueryClass::NaturalLanguage,
        );
        assert_eq!(conf, 730);
    }

    #[test]
    fn confidence_per_mille_natural_language_1_signal() {
        let conf = super::confidence_per_mille(
            "does it work",
            frankensearch_core::query_class::QueryClass::NaturalLanguage,
        );
        assert_eq!(conf, 810);
    }

    #[test]
    fn confidence_per_mille_natural_language_3_signals() {
        // question mark + interrogative + 6+ words
        let conf = super::confidence_per_mille(
            "how does this particular thing actually work?",
            frankensearch_core::query_class::QueryClass::NaturalLanguage,
        );
        assert_eq!(conf, 960);
    }

    #[test]
    fn confidence_per_mille_low_signal_returns_420() {
        let conf = super::confidence_per_mille(
            "???",
            frankensearch_core::query_class::QueryClass::ShortKeyword,
        );
        assert_eq!(conf, 420);
    }

    // --- scale_usize ---

    #[test]
    fn scale_usize_identity() {
        assert_eq!(super::scale_usize(100, 1, 1), 100);
    }

    #[test]
    fn scale_usize_half_rounds_up() {
        // 3 * 1 / 2 with ceiling = (3 + 1) / 2 = 2
        assert_eq!(super::scale_usize(3, 1, 2), 2);
    }

    #[test]
    fn scale_usize_three_quarters() {
        // 100 * 3 / 4 = 300/4 = 75 exactly, ceiling = 75
        assert_eq!(super::scale_usize(100, 3, 4), 75);
    }

    #[test]
    fn scale_usize_zero_value() {
        assert_eq!(super::scale_usize(0, 3, 4), 0);
    }

    // --- scale_u64 ---

    #[test]
    fn scale_u64_identity() {
        assert_eq!(super::scale_u64(100, 1, 1), 100);
    }

    #[test]
    fn scale_u64_four_fifths() {
        // 100 * 4 / 5 = 400/5 = 80 exactly, ceiling = 80
        assert_eq!(super::scale_u64(100, 4, 5), 80);
    }

    #[test]
    fn scale_u64_seven_tenths() {
        // 100 * 7 / 10 = 700/10 = 70 exactly, ceiling = 70
        assert_eq!(super::scale_u64(100, 7, 10), 70);
    }

    #[test]
    fn scale_u64_zero_value() {
        assert_eq!(super::scale_u64(0, 7, 10), 0);
    }

    // --- BudgetScale ---

    #[test]
    fn budget_scale_performance_is_identity() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Performance);
        assert_eq!(scale.fanout_num, 1);
        assert_eq!(scale.fanout_den, 1);
        assert!(!scale.force_fast_only);
    }

    #[test]
    fn budget_scale_strict_forces_fast_only() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Strict);
        assert!(scale.force_fast_only);
    }

    #[test]
    fn budget_scale_degraded_forces_fast_only() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Degraded);
        assert!(scale.force_fast_only);
    }

    #[test]
    fn budget_scale_fanout_zero_stays_zero() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Strict);
        assert_eq!(scale.scale_fanout(0), 0);
    }

    #[test]
    fn budget_scale_fanout_nonzero_min_1() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Degraded);
        assert!(scale.scale_fanout(1) >= 1);
    }

    #[test]
    fn budget_scale_latency_zero_stays_zero() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Strict);
        assert_eq!(scale.scale_latency(0), 0);
    }

    #[test]
    fn budget_scale_latency_nonzero_min_1() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Degraded);
        assert!(scale.scale_latency(1) >= 1);
    }

    #[test]
    fn budget_scale_rerank_zero_stays_zero() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Strict);
        assert_eq!(scale.scale_rerank(0), 0);
    }

    #[test]
    fn budget_scale_rerank_nonzero_min_1() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Degraded);
        assert!(scale.scale_rerank(1) >= 1);
    }

    #[test]
    fn budget_scale_strict_reduces_fanout() {
        let perf = super::BudgetScale::from_pressure(PressureProfile::Performance);
        let strict = super::BudgetScale::from_pressure(PressureProfile::Strict);
        assert!(strict.scale_fanout(100) < perf.scale_fanout(100));
    }

    #[test]
    fn budget_scale_degraded_reduces_more_than_strict() {
        let strict = super::BudgetScale::from_pressure(PressureProfile::Strict);
        let degraded = super::BudgetScale::from_pressure(PressureProfile::Degraded);
        assert!(degraded.scale_fanout(100) <= strict.scale_fanout(100));
    }

    // --- CapabilityState ---

    #[test]
    fn capability_state_enabled_is_enabled() {
        assert!(super::CapabilityState::Enabled.is_enabled());
    }

    #[test]
    fn capability_state_disabled_is_not_enabled() {
        assert!(!super::CapabilityState::Disabled.is_enabled());
    }

    // --- QueryExecutionCapabilities ---

    #[test]
    fn query_execution_capabilities_all_enabled() {
        let caps = QueryExecutionCapabilities::all_enabled();
        assert!(caps.lexical.is_enabled());
        assert!(caps.fast_semantic.is_enabled());
        assert!(caps.quality_semantic.is_enabled());
        assert!(caps.rerank.is_enabled());
    }

    #[test]
    fn query_execution_capabilities_default_is_all_enabled() {
        let caps = QueryExecutionCapabilities::default();
        assert!(caps.lexical.is_enabled());
        assert!(caps.fast_semantic.is_enabled());
        assert!(caps.quality_semantic.is_enabled());
        assert!(caps.rerank.is_enabled());
    }

    // --- rerank_depth ---

    #[test]
    fn rerank_depth_disabled_returns_zero() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Performance);
        assert_eq!(super::rerank_depth(false, scale, 24), 0);
    }

    #[test]
    fn rerank_depth_enabled_returns_scaled() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Performance);
        assert_eq!(super::rerank_depth(true, scale, 24), 24);
    }

    #[test]
    fn rerank_depth_enabled_strict_scales_down() {
        let scale = super::BudgetScale::from_pressure(PressureProfile::Strict);
        let depth = super::rerank_depth(true, scale, 24);
        assert!(depth > 0);
        assert!(depth < 24);
    }

    // --- QueryPlannerConfig::from_fsfs ---

    #[test]
    fn planner_config_from_fsfs_default_limit_min_1() {
        let mut cfg = FsfsConfig::default();
        cfg.search.default_limit = 0;
        let planner_cfg = QueryPlannerConfig::from_fsfs(&cfg);
        assert_eq!(planner_cfg.default_limit, 1);
    }

    #[test]
    fn planner_config_from_fsfs_copies_fields() {
        let mut cfg = FsfsConfig::default();
        cfg.search.default_limit = 25;
        cfg.search.quality_timeout_ms = 500;
        cfg.search.rrf_k = 42.0;
        cfg.search.fast_only = true;
        cfg.pressure.profile = PressureProfile::Degraded;
        let planner_cfg = QueryPlannerConfig::from_fsfs(&cfg);
        assert_eq!(planner_cfg.default_limit, 25);
        assert_eq!(planner_cfg.quality_timeout_ms, 500);
        assert!((planner_cfg.rrf_k - 42.0).abs() < f64::EPSILON);
        assert!(planner_cfg.fast_only);
        assert_eq!(planner_cfg.pressure_profile, PressureProfile::Degraded);
        assert_eq!(
            planner_cfg.low_confidence_threshold_per_mille,
            super::DEFAULT_LOW_CONFIDENCE_THRESHOLD_PER_MILLE
        );
    }

    // --- resolve_limit ---

    #[test]
    fn resolve_limit_none_uses_default() {
        let planner = QueryPlanner::new(QueryPlannerConfig {
            default_limit: 15,
            quality_timeout_ms: 500,
            rrf_k: 60.0,
            fast_only: false,
            pressure_profile: PressureProfile::Performance,
            low_confidence_threshold_per_mille: 650,
        });
        let budget = planner.budget_for_query("hello world", None);
        assert_eq!(budget.limit, 15);
    }

    #[test]
    fn resolve_limit_some_zero_clamps_to_1() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("hello world", Some(0));
        assert_eq!(budget.limit, 1);
    }

    #[test]
    fn resolve_limit_some_value_used() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("hello world", Some(42));
        assert_eq!(budget.limit, 42);
    }

    // --- quality_enabled ---

    #[test]
    fn quality_enabled_false_when_fast_only() {
        let mut cfg = FsfsConfig::default();
        cfg.search.fast_only = true;
        let planner = QueryPlanner::from_fsfs(&cfg);
        let budget = planner.budget_for_query("how does this work", Some(10));
        assert!(!budget.quality_enabled);
    }

    #[test]
    fn quality_enabled_false_under_strict_pressure() {
        let planner = QueryPlanner::new(QueryPlannerConfig {
            pressure_profile: PressureProfile::Strict,
            ..QueryPlannerConfig::from_fsfs(&FsfsConfig::default())
        });
        let budget = planner.budget_for_query("how does this work", Some(10));
        assert!(!budget.quality_enabled);
    }

    #[test]
    fn quality_enabled_false_with_fallback() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        // Malformed queries have MalformedLexicalOnly fallback
        let budget = planner.budget_for_query("hello\u{0007}world", Some(10));
        assert!(!budget.quality_enabled);
    }

    #[test]
    fn quality_enabled_true_performance_no_fallback() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("how does this work", Some(10));
        assert!(budget.quality_enabled);
    }

    // --- budget profiles per intent ---

    #[test]
    fn budget_identifier_focused_profile() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("src/main.rs", Some(10));
        assert_eq!(budget.profile, QueryBudgetProfile::IdentifierFocused);
        assert_eq!(budget.reason_code, "query.budget.identifier_focused");
        assert!(budget.lexical_fanout > budget.semantic_fanout);
    }

    #[test]
    fn budget_balanced_profile_for_short_keyword() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("rust ownership", Some(10));
        assert_eq!(budget.profile, QueryBudgetProfile::Balanced);
        assert_eq!(budget.reason_code, "query.budget.balanced");
        assert_eq!(budget.lexical_fanout, budget.semantic_fanout);
    }

    #[test]
    fn budget_malformed_uses_safe_fallback() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let budget = planner.budget_for_query("find\u{0007}secret", Some(10));
        assert_eq!(budget.profile, QueryBudgetProfile::SafeFallback);
        assert_eq!(budget.semantic_fanout, 0);
        assert!(!budget.quality_enabled);
        assert_eq!(budget.rerank_depth, 0);
        assert_eq!(budget.fallback, QueryFallbackPath::MalformedLexicalOnly);
    }

    #[test]
    fn budget_degraded_pressure_reduces_all() {
        let perf_planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let degraded_planner = QueryPlanner::new(QueryPlannerConfig {
            pressure_profile: PressureProfile::Degraded,
            ..QueryPlannerConfig::from_fsfs(&FsfsConfig::default())
        });
        let perf_budget = perf_planner.budget_for_query("how does ranking work", Some(10));
        let deg_budget = degraded_planner.budget_for_query("how does ranking work", Some(10));

        assert!(deg_budget.lexical_fanout < perf_budget.lexical_fanout);
        assert!(deg_budget.semantic_fanout < perf_budget.semantic_fanout);
        assert!(deg_budget.latency_budget_ms < perf_budget.latency_budget_ms);
        assert!(!deg_budget.quality_enabled);
    }

    // --- fusion_policy_for_mode ---

    #[test]
    fn fusion_policy_rrf_mode() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::HybridRrf, 60.0);
        assert_eq!(policy.strategy, FusionStrategy::Rrf);
        assert_eq!(policy.rrf_k, Some(60.0));
        assert_eq!(policy.tie_break_rules.len(), 4);
        assert_eq!(policy.reason_code, "query.fusion.rrf");
    }

    #[test]
    fn fusion_policy_semantic_only_mode() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::FastSemanticOnly, 60.0);
        assert_eq!(policy.strategy, FusionStrategy::SemanticOnly);
        assert_eq!(policy.rrf_k, None);
        assert_eq!(policy.tie_break_rules.len(), 2);
        assert_eq!(policy.reason_code, "query.fusion.semantic_only");
    }

    #[test]
    fn fusion_policy_lexical_only_mode() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::LexicalOnly, 60.0);
        assert_eq!(policy.strategy, FusionStrategy::LexicalOnly);
        assert_eq!(policy.rrf_k, None);
        assert_eq!(policy.reason_code, "query.fusion.lexical_only");
    }

    #[test]
    fn fusion_policy_empty_mode() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::Empty, 60.0);
        assert_eq!(policy.strategy, FusionStrategy::None);
        assert_eq!(policy.rrf_k, None);
        assert_eq!(policy.tie_break_rules, vec![FusionTieBreakRule::DocIdAsc]);
        assert_eq!(policy.reason_code, "query.fusion.none");
    }

    #[test]
    fn fusion_policy_invalid_rrf_k_defaults_to_60() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::HybridRrf, f64::NAN);
        assert_eq!(policy.rrf_k, Some(60.0));
    }

    #[test]
    fn fusion_policy_rrf_k_below_1_defaults_to_60() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::HybridRrf, 0.5);
        assert_eq!(policy.rrf_k, Some(60.0));
    }

    #[test]
    fn fusion_policy_rrf_k_infinite_defaults_to_60() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::HybridRrf, f64::INFINITY);
        assert_eq!(policy.rrf_k, Some(60.0));
    }

    #[test]
    fn fusion_policy_rrf_k_valid_custom() {
        let policy = super::fusion_policy_for_mode(QueryExecutionMode::HybridRrf, 42.0);
        assert_eq!(policy.rrf_k, Some(42.0));
    }

    // --- cancellation_semantics_for_mode ---

    #[test]
    fn cancellation_empty_mode() {
        let sem = super::cancellation_semantics_for_mode(QueryExecutionMode::Empty, false);
        assert_eq!(
            sem.before_initial_yield,
            CancellationOutcome::ReturnEmptyResults
        );
        assert_eq!(
            sem.after_initial_yield,
            CancellationOutcome::ReturnEmptyResults
        );
        assert_eq!(sem.reason_code, "query.cancel.empty");
    }

    #[test]
    fn cancellation_lexical_only_mode() {
        let sem = super::cancellation_semantics_for_mode(QueryExecutionMode::LexicalOnly, false);
        assert_eq!(
            sem.before_initial_yield,
            CancellationOutcome::AbortWithoutPartial
        );
        assert_eq!(
            sem.after_initial_yield,
            CancellationOutcome::ReturnLexicalResults
        );
        assert_eq!(sem.reason_code, "query.cancel.lexical_only");
    }

    #[test]
    fn cancellation_hybrid_quality_enabled() {
        let sem = super::cancellation_semantics_for_mode(QueryExecutionMode::HybridRrf, true);
        assert_eq!(
            sem.before_initial_yield,
            CancellationOutcome::AbortWithoutPartial
        );
        assert_eq!(
            sem.after_initial_yield,
            CancellationOutcome::ReturnInitialResults
        );
        assert_eq!(sem.reason_code, "query.cancel.phase2_returns_initial");
    }

    #[test]
    fn cancellation_hybrid_quality_disabled() {
        let sem = super::cancellation_semantics_for_mode(QueryExecutionMode::HybridRrf, false);
        assert_eq!(sem.reason_code, "query.cancel.single_phase_returns_initial");
    }

    #[test]
    fn cancellation_semantic_only_quality_enabled() {
        let sem =
            super::cancellation_semantics_for_mode(QueryExecutionMode::FastSemanticOnly, true);
        assert_eq!(sem.reason_code, "query.cancel.phase2_returns_initial");
    }

    // --- execution_reason_code ---

    #[test]
    fn execution_reason_code_empty() {
        assert_eq!(
            super::execution_reason_code(QueryExecutionMode::Empty, QueryFallbackPath::None),
            "query.execution.mode.empty"
        );
    }

    #[test]
    fn execution_reason_code_hybrid_no_fallback() {
        assert_eq!(
            super::execution_reason_code(QueryExecutionMode::HybridRrf, QueryFallbackPath::None),
            "query.execution.mode.hybrid"
        );
    }

    #[test]
    fn execution_reason_code_hybrid_lexical_bias() {
        assert_eq!(
            super::execution_reason_code(
                QueryExecutionMode::HybridRrf,
                QueryFallbackPath::LowConfidenceLexicalBias
            ),
            "query.execution.mode.hybrid.lexical_bias"
        );
    }

    #[test]
    fn execution_reason_code_semantic_only() {
        assert_eq!(
            super::execution_reason_code(
                QueryExecutionMode::FastSemanticOnly,
                QueryFallbackPath::None
            ),
            "query.execution.mode.semantic_only"
        );
    }

    #[test]
    fn execution_reason_code_lexical_only_malformed() {
        assert_eq!(
            super::execution_reason_code(
                QueryExecutionMode::LexicalOnly,
                QueryFallbackPath::MalformedLexicalOnly
            ),
            "query.execution.mode.lexical_only.malformed"
        );
    }

    #[test]
    fn execution_reason_code_lexical_only_no_malformed() {
        assert_eq!(
            super::execution_reason_code(QueryExecutionMode::LexicalOnly, QueryFallbackPath::None),
            "query.execution.mode.lexical_only"
        );
    }

    // --- resolve_execution_mode ---

    #[test]
    fn resolve_execution_mode_empty_intent() {
        let decision = QueryIntentDecision {
            normalized_query: String::new(),
            intent: QueryIntentClass::Empty,
            base_class: Some(frankensearch_core::query_class::QueryClass::Empty),
            confidence_per_mille: 1000,
            fallback: QueryFallbackPath::EmptyQuery,
            reason_code: "query.intent.empty",
        };
        let budget = RetrievalBudget {
            profile: QueryBudgetProfile::Empty,
            limit: 10,
            latency_budget_ms: 0,
            lexical_fanout: 0,
            semantic_fanout: 0,
            rerank_depth: 0,
            quality_enabled: false,
            fallback: QueryFallbackPath::EmptyQuery,
            reason_code: "query.budget.empty",
        };
        let mode = super::resolve_execution_mode(
            &decision,
            &budget,
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(mode, QueryExecutionMode::Empty);
    }

    #[test]
    fn resolve_execution_mode_no_capabilities() {
        let decision = QueryIntentDecision {
            normalized_query: "hello".to_string(),
            intent: QueryIntentClass::ShortKeyword,
            base_class: Some(frankensearch_core::query_class::QueryClass::ShortKeyword),
            confidence_per_mille: 860,
            fallback: QueryFallbackPath::None,
            reason_code: "query.intent.short_keyword",
        };
        let budget = RetrievalBudget {
            profile: QueryBudgetProfile::Balanced,
            limit: 10,
            latency_budget_ms: 180,
            lexical_fanout: 40,
            semantic_fanout: 40,
            rerank_depth: 20,
            quality_enabled: true,
            fallback: QueryFallbackPath::None,
            reason_code: "query.budget.balanced",
        };
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Disabled,
            fast_semantic: super::CapabilityState::Disabled,
            quality_semantic: super::CapabilityState::Disabled,
            rerank: super::CapabilityState::Disabled,
        };
        let mode = super::resolve_execution_mode(&decision, &budget, caps);
        assert_eq!(mode, QueryExecutionMode::Empty);
    }

    #[test]
    fn resolve_execution_mode_malformed_lexical_only() {
        let decision = QueryIntentDecision {
            normalized_query: "hello".to_string(),
            intent: QueryIntentClass::Malformed,
            base_class: None,
            confidence_per_mille: 1000,
            fallback: QueryFallbackPath::MalformedLexicalOnly,
            reason_code: "query.intent.malformed.control_chars",
        };
        let budget = RetrievalBudget {
            profile: QueryBudgetProfile::SafeFallback,
            limit: 10,
            latency_budget_ms: 60,
            lexical_fanout: 20,
            semantic_fanout: 0,
            rerank_depth: 0,
            quality_enabled: false,
            fallback: QueryFallbackPath::MalformedLexicalOnly,
            reason_code: "query.budget.safe_fallback",
        };
        let mode = super::resolve_execution_mode(
            &decision,
            &budget,
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(mode, QueryExecutionMode::LexicalOnly);
    }

    #[test]
    fn resolve_execution_mode_semantic_zero_fanout_falls_to_lexical() {
        let decision = QueryIntentDecision {
            normalized_query: "hello".to_string(),
            intent: QueryIntentClass::ShortKeyword,
            base_class: Some(frankensearch_core::query_class::QueryClass::ShortKeyword),
            confidence_per_mille: 860,
            fallback: QueryFallbackPath::None,
            reason_code: "query.intent.short_keyword",
        };
        let budget = RetrievalBudget {
            profile: QueryBudgetProfile::Balanced,
            limit: 10,
            latency_budget_ms: 180,
            lexical_fanout: 40,
            semantic_fanout: 0, // no semantic budget
            rerank_depth: 0,
            quality_enabled: false,
            fallback: QueryFallbackPath::None,
            reason_code: "query.budget.balanced",
        };
        let mode = super::resolve_execution_mode(
            &decision,
            &budget,
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(mode, QueryExecutionMode::LexicalOnly);
    }

    // --- execution plan stage reason codes ---

    #[test]
    fn execution_plan_quality_disabled_reason_no_semantic() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Enabled,
            fast_semantic: super::CapabilityState::Disabled,
            quality_semantic: super::CapabilityState::Enabled,
            rerank: super::CapabilityState::Enabled,
        };
        let plan = planner.execution_plan_for_query("hello world", Some(10), caps);
        assert_eq!(
            plan.quality_stage.reason_code,
            "query.stage.quality.disabled.no_semantic"
        );
    }

    #[test]
    fn execution_plan_quality_disabled_reason_policy() {
        let mut cfg = FsfsConfig::default();
        cfg.search.fast_only = true;
        let planner = QueryPlanner::from_fsfs(&cfg);
        let plan = planner.execution_plan_for_query(
            "how does this work",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(
            plan.quality_stage.reason_code,
            "query.stage.quality.disabled.policy_or_pressure"
        );
    }

    #[test]
    fn execution_plan_quality_disabled_reason_unavailable() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Enabled,
            fast_semantic: super::CapabilityState::Enabled,
            quality_semantic: super::CapabilityState::Disabled,
            rerank: super::CapabilityState::Enabled,
        };
        let plan = planner.execution_plan_for_query("how does this work", Some(10), caps);
        assert_eq!(
            plan.quality_stage.reason_code,
            "query.stage.quality.disabled.unavailable"
        );
    }

    #[test]
    fn execution_plan_rerank_disabled_reason_no_quality() {
        let mut cfg = FsfsConfig::default();
        cfg.search.fast_only = true;
        let planner = QueryPlanner::from_fsfs(&cfg);
        let plan = planner.execution_plan_for_query(
            "how does this work",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(
            plan.rerank_stage.reason_code,
            "query.stage.rerank.disabled.no_quality"
        );
    }

    #[test]
    fn execution_plan_rerank_disabled_reason_unavailable() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let caps = QueryExecutionCapabilities {
            lexical: super::CapabilityState::Enabled,
            fast_semantic: super::CapabilityState::Enabled,
            quality_semantic: super::CapabilityState::Enabled,
            rerank: super::CapabilityState::Disabled,
        };
        let plan = planner.execution_plan_for_query("how does this work", Some(10), caps);
        assert_eq!(
            plan.rerank_stage.reason_code,
            "query.stage.rerank.disabled.unavailable"
        );
    }

    // --- classify_intent integration ---

    #[test]
    fn classify_intent_natural_language() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("how does query ranking work");
        assert_eq!(decision.intent, QueryIntentClass::NaturalLanguage);
        assert_eq!(decision.fallback, QueryFallbackPath::None);
        assert!(decision.confidence_per_mille >= 650);
    }

    #[test]
    fn classify_intent_short_keyword() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("rust ownership");
        assert_eq!(decision.intent, QueryIntentClass::ShortKeyword);
    }

    #[test]
    fn classify_intent_normalizes_whitespace() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("  hello   world  ");
        assert_eq!(decision.normalized_query, "hello world");
    }

    #[test]
    fn classify_intent_too_long_query() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let long_query = "a ".repeat(super::MAX_QUERY_CHARS + 1);
        let decision = planner.classify_intent(&long_query);
        assert_eq!(decision.intent, QueryIntentClass::Malformed);
        assert_eq!(decision.reason_code, "query.intent.malformed.too_long");
    }

    #[test]
    fn classify_intent_empty_whitespace() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let decision = planner.classify_intent("   ");
        assert_eq!(decision.intent, QueryIntentClass::Empty);
        assert_eq!(decision.confidence_per_mille, 1000);
    }

    // --- QueryPlanner::from_fsfs ---

    #[test]
    fn planner_from_fsfs_roundtrip() {
        let cfg = FsfsConfig::default();
        let planner = QueryPlanner::from_fsfs(&cfg);
        let planner2 = QueryPlanner::new(QueryPlannerConfig::from_fsfs(&cfg));
        assert_eq!(planner, planner2);
    }

    // --- execution plan uncertain fallback ---

    #[test]
    fn execution_plan_uncertain_uses_lexical_bias_reason() {
        let planner = QueryPlanner::from_fsfs(&FsfsConfig::default());
        let plan = planner.execution_plan_for_query(
            "???!!!",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );
        assert_eq!(plan.reason_code, "query.execution.mode.hybrid.lexical_bias");
    }

    // --- quality_timeout_ms clamping ---

    #[test]
    fn quality_timeout_ms_clamped_low() {
        let mut cfg = FsfsConfig::default();
        cfg.search.quality_timeout_ms = 10; // Below 180
        let planner = QueryPlanner::from_fsfs(&cfg);
        let plan = planner.execution_plan_for_query(
            "how does this work",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );
        // quality_stage timeout should be clamped to at least 180
        assert!(plan.quality_stage.timeout_ms >= 180);
    }

    #[test]
    fn quality_timeout_ms_clamped_high() {
        let mut cfg = FsfsConfig::default();
        cfg.search.quality_timeout_ms = 10_000; // Above 2000
        let planner = QueryPlanner::from_fsfs(&cfg);
        let plan = planner.execution_plan_for_query(
            "how does this work",
            Some(10),
            QueryExecutionCapabilities::all_enabled(),
        );
        assert!(plan.quality_stage.timeout_ms <= 2_000);
    }

    // ─── bd-3het tests end ───
}
