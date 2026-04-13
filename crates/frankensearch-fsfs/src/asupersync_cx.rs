use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CxRules {
    pub async_requires_cx_first: bool,
    pub sync_forbids_cx: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsupersyncDomain {
    IoBound,
    StructuredConcurrency,
    TimeoutOrchestration,
    CancellationControl,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RayonDomain {
    CpuParallel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SyncDomain {
    PureTransform,
    DeterministicMath,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionBoundaries {
    pub asupersync_domain: Vec<AsupersyncDomain>,
    pub rayon_domain: Vec<RayonDomain>,
    pub sync_domain: Vec<SyncDomain>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AsyncPattern {
    TimeoutBounded,
    StructuredWorkerPool,
    TwoPhaseChannel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsyncTestMatrix {
    pub unit: Vec<String>,
    pub integration: Vec<String>,
    pub e2e: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsupersyncCxContractDefinition {
    pub kind: String, // "asupersync_cx_contract_definition"
    pub v: u32,       // 1
    pub cx_rules: CxRules,
    pub execution_boundaries: ExecutionBoundaries,
    pub patterns: Vec<AsyncPattern>,
    pub test_matrix: AsyncTestMatrix,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FunctionKind {
    AsyncPublic,
    SyncPublic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CxPosition {
    First,
    NotFirst,
    Absent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiSignatureCase {
    pub kind: String, // "asupersync_cx_api_signature_case"
    pub v: u32,       // 1
    pub function_name: String,
    pub function_kind: FunctionKind,
    pub cx_position: CxPosition,
    pub valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutOutcome {
    Ok,
    Err,
    Cancelled,
    Panicked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabRuntimeResult {
    pub kind: String, // "asupersync_cx_labruntime_result"
    pub v: u32,       // 1
    pub seed: u64,
    pub scenario: String,
    pub deterministic_match: bool,
    pub quiescence_ok: bool,
    pub obligation_leak_ok: bool,
    pub timeout_outcome: TimeoutOutcome,
}
