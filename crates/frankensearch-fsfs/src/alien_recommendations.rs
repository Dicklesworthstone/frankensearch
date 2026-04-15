use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Subsystem {
    IngestionPolicy,
    DegradationScheduler,
    RankingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetedMode {
    pub latency_budget_ms: u32,
    pub memory_budget_mb: u32,
    pub retry_budget: u32,
    pub on_exhaustion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FallbackTrigger {
    pub condition: String,
    pub fallback_action: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IsomorphismProofPlan {
    pub invariants: Vec<String>,
    pub baseline_harness: String,
    pub replay_checks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestField {
    Seed,
    ConfigHash,
    Subsystem,
    PolicyVersion,
    ScenarioId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproArtifacts {
    pub manifest_fields: Vec<ManifestField>,
    pub artifact_outputs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RollbackPlan {
    pub rollback_command: String,
    pub abort_conditions: Vec<String>,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendationCard {
    pub kind: String, // "fsfs_alien_recommendation_card"
    pub v: u32,       // 1
    pub subsystem: Subsystem,
    pub ev_score: f64,
    pub priority_tier: String, // "A", "B", "C"
    pub adoption_wedge: String,
    pub budgeted_mode: BudgetedMode,
    pub fallback_trigger: FallbackTrigger,
    pub baseline_comparator: String,
    pub isomorphism_proof_plan: IsomorphismProofPlan,
    pub repro_artifacts: ReproArtifacts,
    pub rollback_plan: RollbackPlan,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendationBundle {
    pub kind: String, // "fsfs_alien_recommendation_bundle"
    pub v: u32,       // 1
    pub cards: Vec<RecommendationCard>,
}
