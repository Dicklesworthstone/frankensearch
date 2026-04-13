use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseTarget {
    pub target_triple: String,
    pub os_family: String,
    pub build_tool: String,
    pub archive_format: String,
    pub binary_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactNaming {
    pub archive_template: String,
    pub checksum_suffix: String,
    pub metadata_suffix: String,
    pub signature_suffix: String,
    pub certificate_suffix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntegrityPolicy {
    pub checksum_algorithm: String,
    pub checksum_required: bool,
    pub signature_strategy: String,
    pub installer_verify_modes: Vec<String>,
    pub transparency_log_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstallPolicy {
    pub default_entrypoint: String,
    pub developer_entrypoint: String,
    pub preflight_checks: Vec<String>,
    pub required_flags: Vec<String>,
    pub non_root_default: bool,
    pub upgrade_entrypoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollbackBehavior {
    pub on_verification_failure: String,
    pub on_post_install_failure: String,
    pub downgrade_support: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpgradePolicy {
    pub version_resolution_order: Vec<String>,
    pub supported_upgrade_paths: Vec<String>,
    pub rollback_behavior: RollbackBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CiJobs {
    pub release_build_job: String,
    pub release_publish_job: String,
    pub crates_publish_job: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackagingContractDefinition {
    pub kind: String, // "fsfs_packaging_release_install_contract"
    pub schema_version: u32,
    pub release_matrix: Vec<ReleaseTarget>,
    pub artifact_naming: ArtifactNaming,
    pub integrity_policy: IntegrityPolicy,
    pub install_policy: InstallPolicy,
    pub upgrade_policy: UpgradePolicy,
    pub ci_jobs: CiJobs,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactSignature {
    pub signature_path: String,
    pub certificate_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseArtifact {
    pub target_triple: String,
    pub archive_path: String,
    pub checksum: String,
    pub metadata_path: String,
    pub build_tool: String,
    pub signature: ArtifactSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationSummary {
    pub checksum_verified: bool,
    pub signatures_present: bool,
    pub transparency_log_checked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseManifest {
    pub kind: String, // "fsfs_release_manifest"
    pub schema_version: u32,
    pub tag: String,
    pub generated_at: String,
    pub install_script_url: String,
    pub distribution_channels: Vec<String>,
    pub artifacts: Vec<ReleaseArtifact>,
    pub verification_summary: VerificationSummary,
    pub compatibility_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpgradeStep {
    pub stage: String,
    pub action: String,
    pub on_failure_reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollbackConfig {
    pub trigger_reason_codes: Vec<String>,
    pub procedure: String,
    pub user_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpgradePlan {
    pub kind: String, // "fsfs_upgrade_plan"
    pub schema_version: u32,
    pub from_version: String,
    pub to_version: String,
    pub selected_channel: String,
    pub steps: Vec<UpgradeStep>,
    pub rollback: RollbackConfig,
    pub telemetry_fields: Vec<String>,
}
