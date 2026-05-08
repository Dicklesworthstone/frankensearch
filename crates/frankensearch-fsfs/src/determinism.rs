use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchemaVersion1;

impl Serialize for SchemaVersion1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(1)
    }
}

impl<'de> Deserialize<'de> for SchemaVersion1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        if value == 1 {
            Ok(Self)
        } else {
            Err(de::Error::invalid_value(
                de::Unexpected::Unsigned(u64::from(value)),
                &"schema version 1",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeterminismContractDefinitionKind {
    #[serde(rename = "fsfs_determinism_contract_definition")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReproManifestKind {
    #[serde(rename = "fsfs_reproducibility_manifest")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeterminismCheckResultKind {
    #[serde(rename = "fsfs_determinism_check_result")]
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismTier {
    Tier1,
    Tier2,
    Tier3,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    BitExact,
    SemanticEquivalence,
    StatisticalTolerance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TierMatrixEntry {
    pub tier: DeterminismTier,
    pub comparison_mode: ComparisonMode,
    pub required_surfaces: Vec<String>,
    pub guarantee: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NondeterminismSource {
    FloatArithmetic,
    ThreadScheduling,
    FilesystemOrdering,
    ClockSource,
    RandomSampling,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NondeterminismMitigation {
    pub source: NondeterminismSource,
    pub mitigation: String,
    pub requirement_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TestContract {
    pub unit_replay_count_min: u32,
    pub integration_replay_count_min: u32,
    pub e2e_replay_count_min: u32,
    pub required_checks: Vec<String>,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoggingRequirements {
    pub seed_in_every_log: bool,
    pub config_hash_in_every_log: bool,
    pub tier_in_every_log: bool,
    pub mismatch_reason_codes_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeterminismContractDefinition {
    pub kind: DeterminismContractDefinitionKind,
    pub v: SchemaVersion1,
    pub tier_matrix: Vec<TierMatrixEntry>,
    pub nondeterminism_mitigations: Vec<NondeterminismMitigation>,
    pub repro_manifest_required_fields: Vec<String>,
    pub test_contract: TestContract,
    pub logging_requirements: LoggingRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelVersion {
    pub name: String,
    pub version: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub rustc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FloatPolicy {
    pub mode: String,
    pub max_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QueryFingerprint {
    pub query_hash: String,
    pub canonicalizer_version: String,
    pub corpus_snapshot_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConfigSignature {
    pub schema_version: String,
    pub config_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBundle {
    pub manifest_hash: String,
    pub artifact_paths: Vec<String>,
    pub config_signature: ConfigSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReproManifest {
    pub kind: ReproManifestKind,
    pub v: SchemaVersion1,
    pub run_id: String,
    pub determinism_tier: DeterminismTier,
    pub seed: u64,
    pub config_hash: String,
    pub index_version: String,
    pub model_versions: Vec<ModelVersion>,
    pub platform: PlatformInfo,
    pub clock_mode: String,
    pub tie_break_policy: String,
    pub float_policy: FloatPolicy,
    pub query_fingerprint: QueryFingerprint,
    pub evidence_bundle: EvidenceBundle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TolerancePolicy {
    pub metric: String,
    pub max_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MismatchDiagnostic {
    pub reason_code: String,
    pub field_path: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeterminismCheckResult {
    pub kind: DeterminismCheckResultKind,
    pub v: SchemaVersion1,
    pub scenario_id: String,
    pub determinism_tier: DeterminismTier,
    pub comparison_mode: ComparisonMode,
    pub run_count: u32,
    pub pass: bool,
    pub manifest_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance_policy: Option<TolerancePolicy>,
    pub mismatch_diagnostics: Vec<MismatchDiagnostic>,
}

impl DeterminismCheckResult {
    fn validate(&self) -> Result<(), String> {
        let expected_mode = match self.determinism_tier {
            DeterminismTier::Tier1 => ComparisonMode::BitExact,
            DeterminismTier::Tier2 => ComparisonMode::SemanticEquivalence,
            DeterminismTier::Tier3 => ComparisonMode::StatisticalTolerance,
        };

        if self.comparison_mode != expected_mode {
            return Err(format!(
                "{:?} determinism check result requires {:?} comparison mode",
                self.determinism_tier, expected_mode
            ));
        }

        if matches!(self.determinism_tier, DeterminismTier::Tier3)
            && self.tolerance_policy.is_none()
        {
            return Err("tier3 determinism check result requires tolerance_policy".to_owned());
        }

        if !self.pass && self.mismatch_diagnostics.is_empty() {
            return Err(
                "failed determinism check result requires at least one mismatch diagnostic"
                    .to_owned(),
            );
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for DeterminismCheckResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawDeterminismCheckResult {
            kind: DeterminismCheckResultKind,
            v: SchemaVersion1,
            scenario_id: String,
            determinism_tier: DeterminismTier,
            comparison_mode: ComparisonMode,
            run_count: u32,
            pass: bool,
            manifest_ref: String,
            tolerance_policy: Option<TolerancePolicy>,
            mismatch_diagnostics: Vec<MismatchDiagnostic>,
        }

        let raw = RawDeterminismCheckResult::deserialize(deserializer)?;
        let result = Self {
            kind: raw.kind,
            v: raw.v,
            scenario_id: raw.scenario_id,
            determinism_tier: raw.determinism_tier,
            comparison_mode: raw.comparison_mode,
            run_count: raw.run_count,
            pass: raw.pass,
            manifest_ref: raw.manifest_ref,
            tolerance_policy: raw.tolerance_policy,
            mismatch_diagnostics: raw.mismatch_diagnostics,
        };
        result.validate().map_err(de::Error::custom)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DeterminismCheckResult, DeterminismContractDefinition};

    fn valid_contract_definition() -> Value {
        json!({
            "kind": "fsfs_determinism_contract_definition",
            "v": 1,
            "tier_matrix": [
                {
                    "tier": "tier1",
                    "comparison_mode": "bit_exact",
                    "required_surfaces": ["ranked_output"],
                    "guarantee": "Identical inputs and state produce bit-identical outputs."
                }
            ],
            "nondeterminism_mitigations": [
                {
                    "source": "float_arithmetic",
                    "mitigation": "Canonical rounding policy at score comparison boundaries.",
                    "requirement_id": "DET-FLOAT_ROUNDING"
                }
            ],
            "repro_manifest_required_fields": ["seed"],
            "test_contract": {
                "unit_replay_count_min": 2,
                "integration_replay_count_min": 2,
                "e2e_replay_count_min": 2,
                "required_checks": ["ranking_output_stability"]
            },
            "logging_requirements": {
                "seed_in_every_log": true,
                "config_hash_in_every_log": true,
                "tier_in_every_log": true,
                "mismatch_reason_codes_required": true
            }
        })
    }

    fn valid_check_result() -> Value {
        json!({
            "kind": "fsfs_determinism_check_result",
            "v": 1,
            "scenario_id": "tier1-ranked-output-replay",
            "determinism_tier": "tier1",
            "comparison_mode": "bit_exact",
            "run_count": 2,
            "pass": true,
            "manifest_ref": "run-fsfs-tier1-0001",
            "mismatch_diagnostics": []
        })
    }

    #[test]
    fn contract_definition_rejects_wrong_kind() {
        let mut value = valid_contract_definition();
        value["kind"] = json!("wrong_kind");

        let error = serde_json::from_value::<DeterminismContractDefinition>(value)
            .expect_err("reject bad kind");

        assert!(
            error
                .to_string()
                .contains("fsfs_determinism_contract_definition")
        );
    }

    #[test]
    fn contract_definition_rejects_wrong_version() {
        let mut value = valid_contract_definition();
        value["v"] = json!(2);

        let error = serde_json::from_value::<DeterminismContractDefinition>(value)
            .expect_err("reject bad version");

        assert!(error.to_string().contains("schema version 1"));
    }

    #[test]
    fn check_result_rejects_unknown_fields() {
        let mut value = valid_check_result();
        value["extra"] = json!(true);

        let error = serde_json::from_value::<DeterminismCheckResult>(value)
            .expect_err("reject extra field");

        assert!(error.to_string().contains("unknown field `extra`"));
    }

    #[test]
    fn check_result_rejects_tier_comparison_mismatch() {
        let mut value = valid_check_result();
        value["comparison_mode"] = json!("semantic_equivalence");

        let error = serde_json::from_value::<DeterminismCheckResult>(value)
            .expect_err("reject tier1 semantic mode");

        assert!(error.to_string().contains("BitExact"));
    }

    #[test]
    fn check_result_rejects_failed_result_without_diagnostics() {
        let mut value = valid_check_result();
        value["pass"] = json!(false);

        let error = serde_json::from_value::<DeterminismCheckResult>(value)
            .expect_err("reject failed result without diagnostics");

        assert!(error.to_string().contains("mismatch diagnostic"));
    }
}
