use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlacementStatus {
    Resolved,
    Conflict,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlockingImpact {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlacementRule {
    pub rule_id: String,
    pub id_pattern: String,
    pub placement_status: PlacementStatus,
    pub target_paths: Vec<String>,
    pub owner: String,
    pub rationale: String,
    pub integration_boundaries: Option<Vec<String>>,
    pub resolution_owner: Option<String>,
    pub resolution_deadline: Option<String>,
    pub blocking_impact: Option<BlockingImpact>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChangeManagement {
    pub owner: String,
    pub update_workflow: Vec<String>,
    pub lint_commands: Vec<String>,
    pub diagnostic_output: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CratePlacementRegistry {
    pub version: String,
    pub generated_from: String,
    pub last_updated: String,
    pub owners: HashMap<String, String>,
    pub rules: Vec<PlacementRule>,
    pub change_management: ChangeManagement,
}
