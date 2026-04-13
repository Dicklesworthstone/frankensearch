use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NondeterminismMitigation {
    pub source: NondeterminismSource,
    pub mitigation: String,
    pub requirement_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestContract {
    pub unit_replay_count_min: u32,
    pub integration_replay_count_min: u32,
    pub e2e_replay_count_min: u32,
    pub required_checks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingRequirements {
    pub seed_in_every_log: bool,
    pub config_hash_in_every_log: bool,
    pub tier_in_every_log: bool,
    pub mismatch_reason_codes_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeterminismContractDefinition {
    pub kind: String, // "fsfs_determinism_contract_definition"
    pub v: u32,       // 1
    pub tier_matrix: Vec<TierMatrixEntry>,
    pub nondeterminism_mitigations: Vec<NondeterminismMitigation>,
    pub repro_manifest_required_fields: Vec<String>,
    pub test_contract: TestContract,
    pub logging_requirements: LoggingRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelVersion {
    pub name: String,
    pub version: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub rustc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FloatPolicy {
    pub mode: String,
    pub max_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryFingerprint {
    pub query_hash: String,
    pub canonicalizer_version: String,
    pub corpus_snapshot_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigSignature {
    pub schema_version: String,
    pub config_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceBundle {
    pub manifest_hash: String,
    pub artifact_paths: Vec<String>,
    pub config_signature: ConfigSignature,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReproManifest {
    pub kind: String, // "fsfs_reproducibility_manifest"
    pub v: u32,       // 1
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
pub struct TolerancePolicy {
    pub metric: String,
    pub max_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MismatchDiagnostic {
    pub reason_code: String,
    pub field_path: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeterminismCheckResult {
    pub kind: String, // "fsfs_determinism_check_result"
    pub v: u32,       // 1
    pub scenario_id: String,
    pub determinism_tier: DeterminismTier,
    pub comparison_mode: ComparisonMode,
    pub run_count: u32,
    pub pass: bool,
    pub manifest_ref: String,
    pub tolerance_policy: Option<TolerancePolicy>,
    pub mismatch_diagnostics: Vec<MismatchDiagnostic>,
}
