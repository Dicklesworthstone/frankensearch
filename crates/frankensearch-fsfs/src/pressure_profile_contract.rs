//! Pressure profile contract types for fsfs.
//!
//! This module defines the data structures for the fsfs pressure profiles
//! contract v1, which specifies:
//! - Profile configurations (strict, performance, degraded)
//! - Override policies and field locking
//! - Resolution decisions with safety clamps
//! - Migration policy for schema versioning

use serde::{Deserialize, Serialize};
use serde_json::Value;

// --- Kind Constants ---

pub const KIND_CONTRACT_DEFINITION: &str = "fsfs_pressure_profiles_contract_definition";
pub const KIND_PROFILE_RESOLUTION: &str = "fsfs_pressure_profile_resolution";
pub const CONTRACT_VERSION: u32 = 1;

// --- Reason Codes ---

pub const REASON_OVERRIDE_APPLIED_CLI: &str = "override.applied.cli_field";
pub const REASON_OVERRIDE_APPLIED_ENV: &str = "override.applied.env_field";
pub const REASON_OVERRIDE_APPLIED_CONFIG: &str = "override.applied.config_field";
pub const REASON_OVERRIDE_REJECTED_LOCKED: &str = "override.rejected.locked_field";
pub const REASON_OVERRIDE_REJECTED_INVALID: &str = "override.rejected.invalid_value";
pub const REASON_SAFETY_CLAMP_PREFIX: &str = "safety.clamp";
pub const REASON_PROFILE_RESOLUTION_OK: &str = "profile.resolution.ok";
pub const REASON_PROFILE_RESOLUTION_CONFLICT: &str = "profile.resolution.conflict";

// --- Precedence Order Constants ---

pub const PRECEDENCE_HARD_SAFETY_GUARDS: &str = "hard_safety_guards";
pub const PRECEDENCE_CLI_OVERRIDE: &str = "cli_override";
pub const PRECEDENCE_ENV_OVERRIDE: &str = "env_override";
pub const PRECEDENCE_CONFIG_OVERRIDE: &str = "config_override";
pub const PRECEDENCE_PROFILE_DEFAULT: &str = "profile_default";

/// Default precedence order for profile resolution.
pub const DEFAULT_PRECEDENCE_ORDER: &[&str] = &[
    PRECEDENCE_HARD_SAFETY_GUARDS,
    PRECEDENCE_CLI_OVERRIDE,
    PRECEDENCE_ENV_OVERRIDE,
    PRECEDENCE_CONFIG_OVERRIDE,
    PRECEDENCE_PROFILE_DEFAULT,
];

// --- Enums ---

/// Profile identifier for pressure profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileId {
    Strict,
    Performance,
    Degraded,
}

/// Scheduler mode for pressure profile contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractSchedulerMode {
    FairShare,
    LatencySensitive,
}

/// Field names that can be overridden or locked in pressure profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileField {
    SchedulerMode,
    MaxEmbedConcurrency,
    MaxIndexConcurrency,
    QualityEnabled,
    AllowBackgroundIndexing,
}

/// Override source for profile field overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverrideSource {
    Cli,
    Env,
    Config,
}

/// Drift protection mode for migration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftProtection {
    ExplicitMigrationRequired,
    CompatLayerWithReasonCode,
}

// --- Profile Configuration ---

/// Override policy specifying which fields can be overridden and which are locked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverridePolicy {
    pub overridable_fields: Vec<ProfileField>,
    pub locked_fields: Vec<ProfileField>,
}

/// Configuration for a single pressure profile.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub scheduler_mode: ContractSchedulerMode,
    pub max_embed_concurrency: u8,
    pub max_index_concurrency: u8,
    pub quality_enabled: bool,
    pub allow_background_indexing: bool,
    pub pressure_enter_threshold: f64,
    pub pressure_exit_threshold: f64,
    pub override_policy: OverridePolicy,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            scheduler_mode: ContractSchedulerMode::FairShare,
            max_embed_concurrency: 2,
            max_index_concurrency: 2,
            quality_enabled: false,
            allow_background_indexing: false,
            pressure_enter_threshold: 0.35,
            pressure_exit_threshold: 0.20,
            override_policy: OverridePolicy::default(),
        }
    }
}

/// All profile configurations indexed by profile ID.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileSet {
    pub strict: ProfileConfig,
    pub performance: ProfileConfig,
    pub degraded: ProfileConfig,
}

impl Default for ProfileSet {
    fn default() -> Self {
        Self {
            strict: ProfileConfig {
                scheduler_mode: ContractSchedulerMode::FairShare,
                max_embed_concurrency: 2,
                max_index_concurrency: 2,
                quality_enabled: false,
                allow_background_indexing: false,
                pressure_enter_threshold: 0.35,
                pressure_exit_threshold: 0.20,
                override_policy: OverridePolicy {
                    overridable_fields: vec![
                        ProfileField::SchedulerMode,
                        ProfileField::MaxIndexConcurrency,
                    ],
                    locked_fields: vec![
                        ProfileField::QualityEnabled,
                        ProfileField::AllowBackgroundIndexing,
                        ProfileField::MaxEmbedConcurrency,
                    ],
                },
            },
            performance: ProfileConfig {
                scheduler_mode: ContractSchedulerMode::LatencySensitive,
                max_embed_concurrency: 6,
                max_index_concurrency: 8,
                quality_enabled: true,
                allow_background_indexing: true,
                pressure_enter_threshold: 0.65,
                pressure_exit_threshold: 0.45,
                override_policy: OverridePolicy {
                    overridable_fields: vec![
                        ProfileField::SchedulerMode,
                        ProfileField::MaxEmbedConcurrency,
                        ProfileField::MaxIndexConcurrency,
                        ProfileField::AllowBackgroundIndexing,
                    ],
                    locked_fields: vec![ProfileField::QualityEnabled],
                },
            },
            degraded: ProfileConfig {
                scheduler_mode: ContractSchedulerMode::FairShare,
                max_embed_concurrency: 1,
                max_index_concurrency: 1,
                quality_enabled: false,
                allow_background_indexing: false,
                pressure_enter_threshold: 0.15,
                pressure_exit_threshold: 0.10,
                override_policy: OverridePolicy {
                    overridable_fields: Vec::new(),
                    locked_fields: vec![
                        ProfileField::SchedulerMode,
                        ProfileField::MaxEmbedConcurrency,
                        ProfileField::MaxIndexConcurrency,
                        ProfileField::QualityEnabled,
                        ProfileField::AllowBackgroundIndexing,
                    ],
                },
            },
        }
    }
}

// --- Migration Policy ---

/// Migration policy for schema versioning and drift protection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationPolicy {
    pub profile_version: u32,
    pub requires_revision_bump_on_semantic_change: bool,
    pub drift_protection: DriftProtection,
    #[serde(default)]
    pub deprecated_fields: Vec<String>,
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        Self {
            profile_version: 1,
            requires_revision_bump_on_semantic_change: true,
            drift_protection: DriftProtection::ExplicitMigrationRequired,
            deprecated_fields: Vec::new(),
        }
    }
}

// --- Contract Definition ---

/// Contract definition for pressure profiles.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureProfilesContractDefinition {
    pub kind: String,
    pub v: u32,
    pub profiles: ProfileSet,
    pub precedence_order: Vec<String>,
    pub migration_policy: MigrationPolicy,
}

impl Default for PressureProfilesContractDefinition {
    fn default() -> Self {
        Self {
            kind: KIND_CONTRACT_DEFINITION.to_owned(),
            v: CONTRACT_VERSION,
            profiles: ProfileSet::default(),
            precedence_order: DEFAULT_PRECEDENCE_ORDER
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            migration_policy: MigrationPolicy::default(),
        }
    }
}

impl PressureProfilesContractDefinition {
    /// Create a new contract definition with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the profile configuration for a given profile ID.
    #[must_use]
    pub fn get_profile(&self, id: ProfileId) -> &ProfileConfig {
        match id {
            ProfileId::Strict => &self.profiles.strict,
            ProfileId::Performance => &self.profiles.performance,
            ProfileId::Degraded => &self.profiles.degraded,
        }
    }

    /// Check if a field is locked for a given profile.
    #[must_use]
    pub fn is_field_locked(&self, id: ProfileId, field: ProfileField) -> bool {
        self.get_profile(id)
            .override_policy
            .locked_fields
            .contains(&field)
    }

    /// Check if a field is overridable for a given profile.
    #[must_use]
    pub fn is_field_overridable(&self, id: ProfileId, field: ProfileField) -> bool {
        self.get_profile(id)
            .override_policy
            .overridable_fields
            .contains(&field)
    }
}

// --- Override Decision ---

/// A single override decision for a profile field.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverrideDecision {
    pub field: ProfileField,
    pub source: OverrideSource,
    pub requested_value: Value,
    pub applied: bool,
    pub reason_code: String,
}

impl OverrideDecision {
    /// Create a new override decision.
    #[must_use]
    pub fn new(
        field: ProfileField,
        source: OverrideSource,
        requested_value: Value,
        applied: bool,
        reason_code: String,
    ) -> Self {
        Self {
            field,
            source,
            requested_value,
            applied,
            reason_code,
        }
    }

    /// Create an applied override decision.
    #[must_use]
    pub fn applied(field: ProfileField, source: OverrideSource, requested_value: Value) -> Self {
        let reason_code = match source {
            OverrideSource::Cli => REASON_OVERRIDE_APPLIED_CLI,
            OverrideSource::Env => REASON_OVERRIDE_APPLIED_ENV,
            OverrideSource::Config => REASON_OVERRIDE_APPLIED_CONFIG,
        };
        Self::new(field, source, requested_value, true, reason_code.to_owned())
    }

    /// Create a rejected override decision.
    #[must_use]
    pub fn rejected_locked(
        field: ProfileField,
        source: OverrideSource,
        requested_value: Value,
    ) -> Self {
        Self::new(
            field,
            source,
            requested_value,
            false,
            REASON_OVERRIDE_REJECTED_LOCKED.to_owned(),
        )
    }

    /// Returns true if the override was applied.
    #[must_use]
    pub fn was_applied(&self) -> bool {
        self.applied
    }
}

// --- Safety Clamp ---

/// A safety clamp applied to a field value.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyClamp {
    pub field: ProfileField,
    pub clamped_to: Value,
    pub reason_code: String,
}

impl SafetyClamp {
    /// Create a new safety clamp.
    #[must_use]
    pub fn new(field: ProfileField, clamped_to: Value, reason_code: String) -> Self {
        Self {
            field,
            clamped_to,
            reason_code,
        }
    }

    /// Create a safety clamp with auto-generated reason code.
    #[must_use]
    pub fn for_field(field: ProfileField, clamped_to: Value) -> Self {
        let field_name = serde_json::to_string(&field)
            .unwrap_or_default()
            .trim_matches('"')
            .to_owned();
        let reason_code = format!("{REASON_SAFETY_CLAMP_PREFIX}.{field_name}");
        Self::new(field, clamped_to, reason_code)
    }
}

// --- Resolution Diagnostics ---

/// Diagnostics for profile resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionDiagnostics {
    pub event: String,
    pub reason_code: String,
    pub precedence_chain: Vec<String>,
    pub effective_profile_version: u32,
}

impl Default for ResolutionDiagnostics {
    fn default() -> Self {
        Self {
            event: "profile_resolution_completed".to_owned(),
            reason_code: REASON_PROFILE_RESOLUTION_OK.to_owned(),
            precedence_chain: DEFAULT_PRECEDENCE_ORDER
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            effective_profile_version: CONTRACT_VERSION,
        }
    }
}

// --- Profile Resolution ---

/// Resolution decision for pressure profile selection and overrides.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureProfileResolution {
    pub kind: String,
    pub v: u32,
    pub trace_id: String,
    pub selected_profile: ProfileId,
    pub overrides: Vec<OverrideDecision>,
    pub effective: ProfileConfig,
    pub safety_clamps: Vec<SafetyClamp>,
    pub conflict_detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflict_reason_code: Option<String>,
    pub diagnostics: ResolutionDiagnostics,
}

impl PressureProfileResolution {
    /// Create a new profile resolution.
    #[must_use]
    pub fn new(trace_id: String, selected_profile: ProfileId, effective: ProfileConfig) -> Self {
        Self {
            kind: KIND_PROFILE_RESOLUTION.to_owned(),
            v: CONTRACT_VERSION,
            trace_id,
            selected_profile,
            overrides: Vec::new(),
            effective,
            safety_clamps: Vec::new(),
            conflict_detected: false,
            conflict_reason_code: None,
            diagnostics: ResolutionDiagnostics::default(),
        }
    }

    /// Add an override decision to the resolution.
    pub fn add_override(&mut self, decision: OverrideDecision) {
        self.overrides.push(decision);
    }

    /// Add a safety clamp to the resolution.
    pub fn add_safety_clamp(&mut self, clamp: SafetyClamp) {
        self.safety_clamps.push(clamp);
    }

    /// Mark the resolution as having a conflict.
    pub fn set_conflict(&mut self, reason_code: String) {
        self.conflict_detected = true;
        self.conflict_reason_code = Some(reason_code);
        REASON_PROFILE_RESOLUTION_CONFLICT.clone_into(&mut self.diagnostics.reason_code);
    }

    /// Returns true if any overrides were applied.
    #[must_use]
    pub fn has_applied_overrides(&self) -> bool {
        self.overrides.iter().any(|o| o.applied)
    }

    /// Returns true if any safety clamps were applied.
    #[must_use]
    pub fn has_safety_clamps(&self) -> bool {
        !self.safety_clamps.is_empty()
    }

    /// Returns true if a conflict was detected.
    #[must_use]
    pub fn has_conflict(&self) -> bool {
        self.conflict_detected
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_has_correct_kind_and_version() {
        let contract = PressureProfilesContractDefinition::default();
        assert_eq!(contract.kind, KIND_CONTRACT_DEFINITION);
        assert_eq!(contract.v, CONTRACT_VERSION);
    }

    #[test]
    fn default_profiles_have_expected_values() {
        let contract = PressureProfilesContractDefinition::default();

        // Strict profile
        assert_eq!(
            contract.profiles.strict.scheduler_mode,
            ContractSchedulerMode::FairShare
        );
        assert_eq!(contract.profiles.strict.max_embed_concurrency, 2);
        assert!(!contract.profiles.strict.quality_enabled);

        // Performance profile
        assert_eq!(
            contract.profiles.performance.scheduler_mode,
            ContractSchedulerMode::LatencySensitive
        );
        assert_eq!(contract.profiles.performance.max_embed_concurrency, 6);
        assert!(contract.profiles.performance.quality_enabled);

        // Degraded profile
        assert_eq!(contract.profiles.degraded.max_embed_concurrency, 1);
        assert!(!contract.profiles.degraded.allow_background_indexing);
    }

    #[test]
    fn profile_field_locking_works() {
        let contract = PressureProfilesContractDefinition::default();

        // Strict profile has quality_enabled locked
        assert!(contract.is_field_locked(ProfileId::Strict, ProfileField::QualityEnabled));
        assert!(!contract.is_field_locked(ProfileId::Strict, ProfileField::SchedulerMode));

        // Performance profile has quality_enabled locked
        assert!(contract.is_field_locked(ProfileId::Performance, ProfileField::QualityEnabled));
        assert!(
            !contract.is_field_locked(ProfileId::Performance, ProfileField::MaxEmbedConcurrency)
        );

        // Degraded profile has everything locked
        assert!(contract.is_field_locked(ProfileId::Degraded, ProfileField::SchedulerMode));
        assert!(contract.is_field_locked(ProfileId::Degraded, ProfileField::QualityEnabled));
    }

    #[test]
    fn profile_field_overridable_works() {
        let contract = PressureProfilesContractDefinition::default();

        // Strict profile allows scheduler_mode override
        assert!(contract.is_field_overridable(ProfileId::Strict, ProfileField::SchedulerMode));
        assert!(!contract.is_field_overridable(ProfileId::Strict, ProfileField::QualityEnabled));

        // Degraded profile allows nothing
        assert!(!contract.is_field_overridable(ProfileId::Degraded, ProfileField::SchedulerMode));
    }

    #[test]
    fn override_decision_applied_creates_correct_reason_code() {
        let decision = OverrideDecision::applied(
            ProfileField::MaxIndexConcurrency,
            OverrideSource::Cli,
            serde_json::json!(10),
        );
        assert!(decision.was_applied());
        assert_eq!(decision.reason_code, REASON_OVERRIDE_APPLIED_CLI);
    }

    #[test]
    fn override_decision_rejected_creates_correct_reason_code() {
        let decision = OverrideDecision::rejected_locked(
            ProfileField::QualityEnabled,
            OverrideSource::Env,
            serde_json::json!(false),
        );
        assert!(!decision.was_applied());
        assert_eq!(decision.reason_code, REASON_OVERRIDE_REJECTED_LOCKED);
    }

    #[test]
    fn safety_clamp_generates_reason_code() {
        let clamp =
            SafetyClamp::for_field(ProfileField::MaxIndexConcurrency, serde_json::json!(10));
        assert!(clamp.reason_code.starts_with(REASON_SAFETY_CLAMP_PREFIX));
    }

    #[test]
    fn profile_resolution_new_has_correct_defaults() {
        let resolution = PressureProfileResolution::new(
            "trace-001".to_owned(),
            ProfileId::Performance,
            ProfileConfig::default(),
        );
        assert_eq!(resolution.kind, KIND_PROFILE_RESOLUTION);
        assert_eq!(resolution.v, CONTRACT_VERSION);
        assert!(!resolution.has_conflict());
        assert!(!resolution.has_applied_overrides());
        assert!(!resolution.has_safety_clamps());
    }

    #[test]
    fn profile_resolution_conflict_sets_fields() {
        let mut resolution = PressureProfileResolution::new(
            "trace-002".to_owned(),
            ProfileId::Strict,
            ProfileConfig::default(),
        );
        resolution.set_conflict("profile.conflict.test".to_owned());

        assert!(resolution.has_conflict());
        assert_eq!(
            resolution.conflict_reason_code,
            Some("profile.conflict.test".to_owned())
        );
        assert_eq!(
            resolution.diagnostics.reason_code,
            REASON_PROFILE_RESOLUTION_CONFLICT
        );
    }

    #[test]
    fn contract_roundtrip_serialization() {
        let contract = PressureProfilesContractDefinition::default();
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: PressureProfilesContractDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(contract, parsed);
    }

    #[test]
    fn resolution_roundtrip_serialization() {
        let mut resolution = PressureProfileResolution::new(
            "trace-003".to_owned(),
            ProfileId::Performance,
            ProfileConfig::default(),
        );
        resolution.add_override(OverrideDecision::applied(
            ProfileField::MaxIndexConcurrency,
            OverrideSource::Cli,
            serde_json::json!(10),
        ));
        resolution.add_safety_clamp(SafetyClamp::for_field(
            ProfileField::MaxIndexConcurrency,
            serde_json::json!(10),
        ));

        let json = serde_json::to_string(&resolution).unwrap();
        let parsed: PressureProfileResolution = serde_json::from_str(&json).unwrap();
        assert_eq!(resolution, parsed);
    }

    #[test]
    fn golden_contract_roundtrip() {
        // Verify we can parse the golden contract definition
        let golden = r#"{
          "kind": "fsfs_pressure_profiles_contract_definition",
          "v": 1,
          "profiles": {
            "strict": {
              "scheduler_mode": "fair_share",
              "max_embed_concurrency": 2,
              "max_index_concurrency": 2,
              "quality_enabled": false,
              "allow_background_indexing": false,
              "pressure_enter_threshold": 0.35,
              "pressure_exit_threshold": 0.20,
              "override_policy": {
                "overridable_fields": ["scheduler_mode", "max_index_concurrency"],
                "locked_fields": ["quality_enabled", "allow_background_indexing", "max_embed_concurrency"]
              }
            },
            "performance": {
              "scheduler_mode": "latency_sensitive",
              "max_embed_concurrency": 6,
              "max_index_concurrency": 8,
              "quality_enabled": true,
              "allow_background_indexing": true,
              "pressure_enter_threshold": 0.65,
              "pressure_exit_threshold": 0.45,
              "override_policy": {
                "overridable_fields": ["scheduler_mode", "max_embed_concurrency", "max_index_concurrency", "allow_background_indexing"],
                "locked_fields": ["quality_enabled"]
              }
            },
            "degraded": {
              "scheduler_mode": "fair_share",
              "max_embed_concurrency": 1,
              "max_index_concurrency": 1,
              "quality_enabled": false,
              "allow_background_indexing": false,
              "pressure_enter_threshold": 0.15,
              "pressure_exit_threshold": 0.10,
              "override_policy": {
                "overridable_fields": [],
                "locked_fields": ["scheduler_mode", "max_embed_concurrency", "max_index_concurrency", "quality_enabled", "allow_background_indexing"]
              }
            }
          },
          "precedence_order": ["hard_safety_guards", "cli_override", "env_override", "config_override", "profile_default"],
          "migration_policy": {
            "profile_version": 1,
            "requires_revision_bump_on_semantic_change": true,
            "drift_protection": "explicit_migration_required",
            "deprecated_fields": []
          }
        }"#;

        let parsed: PressureProfilesContractDefinition = serde_json::from_str(golden).unwrap();
        assert_eq!(parsed.kind, KIND_CONTRACT_DEFINITION);
        assert_eq!(parsed.v, 1);

        // Roundtrip
        let serialized = serde_json::to_string(&parsed).unwrap();
        let reparsed: PressureProfilesContractDefinition =
            serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn golden_resolution_roundtrip() {
        // Verify we can parse the golden resolution
        let golden = r#"{
          "kind": "fsfs_pressure_profile_resolution",
          "v": 1,
          "trace_id": "trace-prof-001",
          "selected_profile": "performance",
          "overrides": [
            {
              "field": "max_index_concurrency",
              "source": "cli",
              "requested_value": 10,
              "applied": true,
              "reason_code": "override.applied.cli_field"
            },
            {
              "field": "quality_enabled",
              "source": "env",
              "requested_value": false,
              "applied": false,
              "reason_code": "override.rejected.locked_field"
            }
          ],
          "effective": {
            "scheduler_mode": "latency_sensitive",
            "max_embed_concurrency": 6,
            "max_index_concurrency": 10,
            "quality_enabled": true,
            "allow_background_indexing": true,
            "pressure_enter_threshold": 0.65,
            "pressure_exit_threshold": 0.45,
            "override_policy": {
              "overridable_fields": ["scheduler_mode", "max_embed_concurrency", "max_index_concurrency", "allow_background_indexing"],
              "locked_fields": ["quality_enabled"]
            }
          },
          "safety_clamps": [
            {
              "field": "max_index_concurrency",
              "clamped_to": 10,
              "reason_code": "safety.clamp.max_index_concurrency"
            }
          ],
          "conflict_detected": false,
          "diagnostics": {
            "event": "profile_resolution_completed",
            "reason_code": "profile.resolution.ok",
            "precedence_chain": ["hard_safety_guards", "cli_override", "env_override", "config_override", "profile_default"],
            "effective_profile_version": 1
          }
        }"#;

        let parsed: PressureProfileResolution = serde_json::from_str(golden).unwrap();
        assert_eq!(parsed.kind, KIND_PROFILE_RESOLUTION);
        assert_eq!(parsed.v, 1);
        assert_eq!(parsed.selected_profile, ProfileId::Performance);
        assert_eq!(parsed.overrides.len(), 2);
        assert!(parsed.overrides[0].applied);
        assert!(!parsed.overrides[1].applied);

        // Roundtrip
        let serialized = serde_json::to_string(&parsed).unwrap();
        let reparsed: PressureProfileResolution = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed, reparsed);
    }
}
