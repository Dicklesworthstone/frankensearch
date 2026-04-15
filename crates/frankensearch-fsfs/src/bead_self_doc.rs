use serde::{Deserialize, Serialize};
use serde_json::Number;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WaveAssignment {
    #[serde(rename = "wave_1")]
    Wave1,
    #[serde(rename = "wave_2")]
    Wave2,
    #[serde(rename = "wave_3")]
    Wave3,
    #[serde(rename = "exception_register")]
    ExceptionRegister,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DebtItem {
    pub id: String,
    pub title: String,
    pub issue_type: String,
    pub status: String,
    pub priority: u32,
    pub class: String, // "gate", "program", "exploratory", "implementation"
    pub severity: String, // "error", "warning", "info"
    pub has_rationale: bool,
    pub has_evidence: bool,
    pub has_exception: bool,
    pub suggested_owner: String,
    pub owner_suggestion_basis: String,
    pub pagerank: Number,
    pub betweenness: Number,
    pub critical_path: u32,
    pub risk_score: u32,
    pub recommended_wave: WaveAssignment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoringMetadata {
    pub risk_score: String,
    pub owner_suggestion: String,
    pub wave_assignment: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InventorySource {
    pub issues_jsonl: String,
    pub insights_data_hash: String,
    pub insights_generated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelfDocInventory {
    pub generated_at: String,
    pub inventory_date: String,
    pub source: InventorySource,
    pub scoring: ScoringMetadata,
    pub items: Vec<DebtItem>,
}
