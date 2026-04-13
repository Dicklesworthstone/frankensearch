use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScopeDefaults {
    pub default_roots: Vec<String>,
    pub requires_explicit_opt_in_outside_defaults: bool,
    pub opt_out_globs_supported: bool,
    pub precedence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathPolicies {
    pub deny_always_globs: Vec<String>,
    pub allow_with_opt_in_globs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedactionConfig {
    pub logs_default_action: String,
    pub explain_default_action: String,
    pub replay_default_action: String,
    pub deterministic_profile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThreatModel {
    pub local_multi_user_assumed: bool,
    pub same_host_read_risk: bool,
    pub mitigations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryEmissionRules {
    pub raw_content_allowed: bool,
    pub reason_code_required: bool,
    pub redaction_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScopePrivacyContractDefinition {
    pub kind: String, // "fsfs_scope_privacy_contract_definition"
    pub v: u32,       // 1
    pub scope_defaults: ScopeDefaults,
    pub sensitive_classes: Vec<String>,
    pub path_policies: PathPolicies,
    pub redaction: RedactionConfig,
    pub threat_model: ThreatModel,
    pub telemetry_emission_rules: TelemetryEmissionRules,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedactedArtifact {
    pub kind: String, // "fsfs_scope_redacted_artifact"
    pub v: u32,       // 1
    pub artifact_type: String,
    pub path: String,
    pub reason_code: String,
    pub redaction_applied: bool,
    pub raw_content_present: bool,
    pub redaction_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScopeScanDecision {
    pub kind: String, // "fsfs_scope_scan_decision"
    pub v: u32,       // 1
    pub path: String,
    pub decision: String,
    pub reason_code: String,
    pub sensitive_classes: Vec<String>,
    pub persist_allowed: bool,
    pub emit_allowed: bool,
    pub display_allowed: bool,
    pub redaction_profile: String,
}
