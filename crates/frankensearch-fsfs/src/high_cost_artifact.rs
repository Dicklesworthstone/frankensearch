use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GiantLogDetector {
    pub max_size_mb: u32,
    pub churn_window_minutes: u32,
    pub redundancy_ratio_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VendorGeneratedDetector {
    pub vendor_path_patterns: Vec<String>,
    pub generated_markers: Vec<String>,
    pub library_tree_depth_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchiveTransientDetector {
    pub archive_extensions: Vec<String>,
    pub transient_directories: Vec<String>,
    pub build_artifact_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverridePolicy {
    pub allow_user_force_include: bool,
    pub requires_reason: bool,
    pub max_override_ttl_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DefaultAction {
    IndexMetadataOnly,
    Skip,
    IndexFull,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DefaultActions {
    pub giant_log: DefaultAction,
    pub vendor_tree: DefaultAction,
    pub generated_file: DefaultAction,
    pub archive_container: DefaultAction,
    pub transient_build_artifact: DefaultAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighCostArtifactContractDefinition {
    pub kind: String, // "fsfs_high_cost_artifact_contract_definition"
    pub v: u32,       // 1
    pub giant_log_detector: GiantLogDetector,
    pub vendor_generated_detector: VendorGeneratedDetector,
    pub archive_transient_detector: ArchiveTransientDetector,
    pub override_policy: OverridePolicy,
    pub default_actions: DefaultActions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Evidence {
    pub size_mb: u32,
    pub churn_rate_per_hour: u32,
    pub redundancy_ratio: f64,
    pub path_depth: u32,
    pub extension: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighCostArtifactDecision {
    pub kind: String, // "fsfs_high_cost_artifact_decision"
    pub v: u32,       // 1
    pub path: String,
    pub detectors_fired: Vec<String>,
    pub evidence: Evidence,
    pub final_action: DefaultAction,
    pub reason_code: String,
    pub cost_score: f64,
    pub override_applied: bool,
    pub needs_manual_review: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighCostOverrideEvent {
    pub kind: String, // "fsfs_high_cost_override_event"
    pub v: u32,       // 1
    pub path: String,
    pub requested_action: DefaultAction,
    pub approved: bool,
    pub expires_at: String,
    pub reason: String,
    pub reason_code: String,
}
