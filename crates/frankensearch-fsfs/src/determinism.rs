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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
}
