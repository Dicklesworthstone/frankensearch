use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossContractDefinition {
    pub kind: String, // "fsfs_expected_loss_contract_definition"
    pub v: u32,       // 1
    pub action_families: HashMap<String, Vec<String>>,
    pub cost_asymmetry_definitions: HashMap<String, String>,
    pub required_decision_fields: Vec<String>,
    pub fallback_policy: FallbackPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FallbackPolicy {
    pub required_for_high_risk: bool,
    pub required_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossDecisionEvent {
    pub kind: String, // "fsfs_expected_loss_decision_event"
    pub v: u32,       // 1
    pub decision_id: String,
    pub seed: u64,
    pub config_hash: String,
    pub family: String,
    pub state_id: String,
    pub chosen_action: String,
    pub evaluated_actions: Vec<ExpectedLossActionEvaluation>,
    pub selected_reason_code: String,
    pub fallback_invoked: bool,
    pub fallback_reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossMatrix {
    pub kind: String, // "fsfs_expected_loss_matrix"
    pub v: u32,       // 1
    pub family: String,
    pub state_space: Vec<String>,
    pub action_space: Vec<String>,
    pub loss_rows: Vec<ExpectedLossMatrixRow>,
    pub fallback_triggers: Vec<ExpectedLossFallbackTrigger>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossActionEvaluation {
    pub action: String,
    pub expected_loss: f64,
    pub false_include_cost: f64,
    pub false_exclude_cost: f64,
    pub latency_cost: f64,
    pub quality_cost: f64,
    pub compute_cost: f64,
    pub risk_level: String, // "low", "medium", "high", "critical"
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossMatrixRow {
    pub state_id: String,
    pub action_losses: Vec<ExpectedLossActionEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectedLossFallbackTrigger {
    pub condition: String,
    pub fallback_action: String,
    pub reason_code: String,
    pub trip_threshold: String,
    pub applies_to_actions: Vec<String>,
}
