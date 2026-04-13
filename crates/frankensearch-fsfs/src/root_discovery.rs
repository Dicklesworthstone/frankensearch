use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StrictModeConfig {
    pub deny_on_ambiguity: bool,
    pub cross_mount_default: bool,
    pub follow_symlink_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermissiveModeConfig {
    pub allow_explicit_include_override: bool,
    pub audit_reason_codes_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverrideModes {
    pub strict: StrictModeConfig,
    pub permissive: PermissiveModeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraversalSafety {
    pub symlink_policy: String, // "follow_bounded", etc.
    pub max_symlink_depth: u32,
    pub detect_loops: bool,
    pub mount_boundary_policy: String, // "stay_on_device", etc.
    pub max_mount_hops: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RootDiscoveryContractDefinition {
    pub kind: String, // "fsfs_root_discovery_contract_definition"
    pub v: u32,       // 1
    pub default_roots: Vec<String>,
    pub override_modes: OverrideModes,
    pub precedence_order: Vec<String>,
    pub traversal_safety: TraversalSafety,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuleEvaluation {
    pub source: String,
    pub matched: bool,
    pub effect: String, // "noop", "include", "exclude"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RootDiscoveryDecision {
    pub kind: String, // "fsfs_root_discovery_decision"
    pub v: u32,       // 1
    pub path: String,
    pub override_mode: String,
    pub rules_evaluated: Vec<RuleEvaluation>,
    pub final_decision: String,
    pub reason_code: String,
    pub symlink_detected: bool,
    pub mount_crossing: bool,
    pub loop_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RootTraversalGuardEvent {
    pub kind: String, // "fsfs_root_traversal_guard_event"
    pub v: u32,       // 1
    pub path: String,
    pub guard_type: String,
    pub action_taken: String,
    pub reason_code: String,
}
