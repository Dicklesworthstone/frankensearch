//! Strict machine-class registry and runner-receipt admission.
//!
//! The normative registry is compiled into this crate and bound by both its
//! reviewed Git blob identity and its exact file SHA-256. Admission never
//! trusts a caller-supplied machine label: it derives the class and execution
//! identity from a duplicate-key-rejecting, unknown-field-rejecting runner
//! receipt, then compares an optional caller expectation.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path};

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Reviewed commit containing the normative registry.
pub const MACHINE_CLASS_REGISTRY_SPEC_COMMIT: &str = "c3b581e286d04979a9954567a5d6c318d061f6cc";
/// Exact Git blob of the normative registry.
pub const MACHINE_CLASS_REGISTRY_GIT_BLOB: &str = "83a904721c81a4e5d19c8164d4ffe3924f3afddd";
/// SHA-256 of the exact normative registry file bytes.
pub const MACHINE_CLASS_REGISTRY_SHA256: &str =
    "ee047b288dbabc620a1eb5d4c4b619a8655ab11011d81adab0916375d495ab48";
/// Registry schema accepted by this consumer.
pub const MACHINE_CLASS_REGISTRY_SCHEMA_VERSION: &str =
    "frankensearch.quill-machine-class-registry.v1";

const REGISTRY_BYTES: &[u8] = include_bytes!("../../../docs/contracts/quill-machine-classes.json");
const TRJ_PROVENANCE_BYTES: &[u8] =
    include_bytes!("../../../docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/provenance.json");
const TRJ_LSCPU_BYTES: &[u8] =
    include_bytes!("../../../docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/lscpu.txt");
const TRJ_NUMACTL_BYTES: &[u8] =
    include_bytes!("../../../docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/numactl-H.txt");
const M4_PROVENANCE_BYTES: &[u8] =
    include_bytes!("../../../docs/evidence/e8h/fingerprints/m4-macos-20260728/provenance.json");
const M4_SYSCTL_BYTES: &[u8] =
    include_bytes!("../../../docs/evidence/e8h/fingerprints/m4-macos-20260728/sysctl.txt");

/// Stable admission reason defined by the normative registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MachineClassReason {
    /// Receipt and external admission context are fully admitted.
    Admitted,
    /// The named class has not yet proven homogeneous hardware.
    ClassHomogeneityUnproven,
    /// The named class has no registered reachable hardware.
    ClassUnavailable,
    /// No class rule matches the requested ID.
    UnknownClassId,
    /// The requested ID is an explicitly obsolete alias.
    ObsoleteClassId,
    /// More than one class rule matches an ID.
    AmbiguousClassId,
    /// JSON repeats an object key.
    DuplicateKey,
    /// JSON contains a field outside the strict schema.
    UnknownField,
    /// JSON omits a required field.
    MissingField,
    /// Operating system does not match the class.
    HardwareOsMismatch,
    /// Architecture does not match the class.
    HardwareArchMismatch,
    /// CPU vendor does not match the class.
    HardwareCpuVendorMismatch,
    /// CPU family does not match the class.
    HardwareCpuFamilyMismatch,
    /// CPU model does not match the class.
    HardwareCpuModelMismatch,
    /// CPU stepping does not match the class.
    HardwareCpuSteppingMismatch,
    /// CPU or chip name does not match the class.
    HardwareCpuNameMismatch,
    /// Physical or logical topology does not match the class.
    HardwareTopologyMismatch,
    /// NUMA hardware does not match the class.
    HardwareNumaMismatch,
    /// Memory size does not match the class.
    HardwareMemoryMismatch,
    /// Page size does not match the class.
    HardwarePageSizeMismatch,
    /// Apple performance-core count does not match the class.
    HardwarePerformanceCoreMismatch,
    /// Apple efficiency-core count does not match the class.
    HardwareEfficiencyCoreMismatch,
    /// Requested or observed execution width is inconsistent.
    ExecutionWidthMismatch,
    /// CPU assignment is malformed, unobservable when required, or unequal.
    ExecutionCpusetInvalid,
    /// Thread budget is zero or exceeds the admitted CPU pool.
    ExecutionThreadBudgetInvalid,
    /// SMT suffix, threads-per-core, and SMT state disagree.
    ExecutionSmtMismatch,
    /// NUMA execution policy does not match the class.
    ExecutionNumaMismatch,
    /// CPU governor does not match the class.
    ExecutionGovernorMismatch,
    /// Apple execution mode is invalid or inconsistent.
    ExecutionModeMismatch,
    /// Thermal pressure makes timed evidence inadmissible.
    ThermalPressure,
    /// The exclusive measurement lease is absent.
    ExclusiveLeaseMissing,
    /// The benchmark executed through an offloaded worker.
    ExecutionOffloaded,
    /// The measured source tree was dirty.
    SourceDirty,
    /// A source, executable, command, or fingerprint identity is invalid.
    SourceIdentityInvalid,
    /// Durability-adjacent arms used different sync treatment.
    DurabilityAsymmetric,
    /// Completion or artifact digests were not verified.
    CompletionUnverified,
    /// The benchmark command exited unsuccessfully.
    CompletionFailed,
    /// Start and end execution snapshots differ.
    PrePostIdentityDrift,
    /// Requested and derived class IDs differ.
    ReceiptClassMismatch,
    /// Receipt and exact registry digest differ.
    RegistryHashMismatch,
    /// A recomputable hardware, cpuset, snapshot, or execution hash differs.
    DerivedHashMismatch,
    /// Destination latest key does not name the derived class.
    DestinationIdentityMismatch,
}

impl MachineClassReason {
    /// Stable kebab-case reason code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::ClassHomogeneityUnproven => "class-homogeneity-unproven",
            Self::ClassUnavailable => "class-unavailable",
            Self::UnknownClassId => "unknown-class-id",
            Self::ObsoleteClassId => "obsolete-class-id",
            Self::AmbiguousClassId => "ambiguous-class-id",
            Self::DuplicateKey => "duplicate-key",
            Self::UnknownField => "unknown-field",
            Self::MissingField => "missing-field",
            Self::HardwareOsMismatch => "hardware-os-mismatch",
            Self::HardwareArchMismatch => "hardware-arch-mismatch",
            Self::HardwareCpuVendorMismatch => "hardware-cpu-vendor-mismatch",
            Self::HardwareCpuFamilyMismatch => "hardware-cpu-family-mismatch",
            Self::HardwareCpuModelMismatch => "hardware-cpu-model-mismatch",
            Self::HardwareCpuSteppingMismatch => "hardware-cpu-stepping-mismatch",
            Self::HardwareCpuNameMismatch => "hardware-cpu-name-mismatch",
            Self::HardwareTopologyMismatch => "hardware-topology-mismatch",
            Self::HardwareNumaMismatch => "hardware-numa-mismatch",
            Self::HardwareMemoryMismatch => "hardware-memory-mismatch",
            Self::HardwarePageSizeMismatch => "hardware-page-size-mismatch",
            Self::HardwarePerformanceCoreMismatch => "hardware-performance-core-mismatch",
            Self::HardwareEfficiencyCoreMismatch => "hardware-efficiency-core-mismatch",
            Self::ExecutionWidthMismatch => "execution-width-mismatch",
            Self::ExecutionCpusetInvalid => "execution-cpuset-invalid",
            Self::ExecutionThreadBudgetInvalid => "execution-thread-budget-invalid",
            Self::ExecutionSmtMismatch => "execution-smt-mismatch",
            Self::ExecutionNumaMismatch => "execution-numa-mismatch",
            Self::ExecutionGovernorMismatch => "execution-governor-mismatch",
            Self::ExecutionModeMismatch => "execution-mode-mismatch",
            Self::ThermalPressure => "thermal-pressure",
            Self::ExclusiveLeaseMissing => "exclusive-lease-missing",
            Self::ExecutionOffloaded => "execution-offloaded",
            Self::SourceDirty => "source-dirty",
            Self::SourceIdentityInvalid => "source-identity-invalid",
            Self::DurabilityAsymmetric => "durability-asymmetric",
            Self::CompletionUnverified => "completion-unverified",
            Self::CompletionFailed => "completion-failed",
            Self::PrePostIdentityDrift => "pre-post-identity-drift",
            Self::ReceiptClassMismatch => "receipt-class-mismatch",
            Self::RegistryHashMismatch => "registry-hash-mismatch",
            Self::DerivedHashMismatch => "derived-hash-mismatch",
            Self::DestinationIdentityMismatch => "destination-identity-mismatch",
        }
    }
}

impl fmt::Display for MachineClassReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Typed fail-closed registry or receipt rejection.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{reason}: {detail}")]
pub struct MachineClassError {
    /// Stable reason code.
    pub reason: MachineClassReason,
    /// Bounded diagnostic detail.
    pub detail: String,
}

impl MachineClassError {
    fn new(reason: MachineClassReason, detail: impl Into<String>) -> Self {
        let mut detail = detail.into();
        detail.truncate(240);
        Self { reason, detail }
    }
}

/// Terminal class-lookup decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineClassDecision {
    /// Class is registered and eligible for receipt admission.
    Allow,
    /// Class can be identified but cannot support promotion.
    DiagnosticOnly,
    /// Receipt admission failed.
    Reject,
    /// Registry loading failed.
    RejectRegistry,
}

/// Result of resolving one requested class ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineClassLookup {
    /// Terminal lookup decision.
    pub decision: MachineClassDecision,
    /// Canonical class ID when one class was identified.
    pub class_id: Option<String>,
    /// Stable reason code.
    pub reason: MachineClassReason,
}

/// External ratchet context that cannot relabel a receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineClassAdmissionContext {
    /// Canonical gate label such as `QG-2`.
    pub gate: String,
    /// Exact destination basename proposed for the latest pointer.
    pub destination_basename: String,
}

/// Recomputed hashes defined by the registry canonicalization contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineClassDerivedHashes {
    /// Canonical hardware-facts hash.
    pub hardware: String,
    /// Canonical start cpuset hash.
    pub start_cpuset: String,
    /// Canonical start snapshot hash.
    pub start_snapshot: String,
    /// Canonical end snapshot hash.
    pub end_snapshot: String,
    /// Canonical stable execution-identity hash.
    pub identity: String,
}

/// Frozen registry and canonical-JSON contract bound into an admitted identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineClassCanonicalizationBinding {
    registry_schema_version: String,
    registry_sha256: String,
    registry_git_blob: String,
    canonical_hash_contract_sha256: String,
}

impl MachineClassCanonicalizationBinding {
    /// Registry schema interpreted by the consumer.
    #[must_use]
    pub fn registry_schema_version(&self) -> &str {
        &self.registry_schema_version
    }

    /// SHA-256 of the exact registry file bytes.
    #[must_use]
    pub fn registry_sha256(&self) -> &str {
        &self.registry_sha256
    }

    /// Reviewed Git blob that supplied the registry.
    #[must_use]
    pub fn registry_git_blob(&self) -> &str {
        &self.registry_git_blob
    }

    /// SHA-256 of the registry's canonical-hash contract object.
    #[must_use]
    pub fn canonical_hash_contract_sha256(&self) -> &str {
        &self.canonical_hash_contract_sha256
    }
}

/// A runner identity admitted from exact receipt bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedRunnerIdentity {
    receipt_json: String,
    receipt_sha256: String,
    admission_context: MachineClassAdmissionContext,
    canonicalization: MachineClassCanonicalizationBinding,
    canonical_class_id: String,
    hardware: Value,
    execution_request: Value,
    execution_start: Value,
    execution_end: Value,
    build: Value,
    durability: Value,
    completion: Value,
    derived_sha256: MachineClassDerivedHashes,
}

impl VerifiedRunnerIdentity {
    /// Canonical class derived from strict receipt facts.
    #[must_use]
    pub fn class_id(&self) -> &str {
        &self.canonical_class_id
    }

    /// SHA-256 of the exact sealed runner receipt bytes.
    #[must_use]
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    /// Exact strict JSON receipt bytes represented as UTF-8 text.
    #[must_use]
    pub fn receipt_json(&self) -> &str {
        &self.receipt_json
    }

    /// External gate and destination context used during admission.
    #[must_use]
    pub const fn admission_context(&self) -> &MachineClassAdmissionContext {
        &self.admission_context
    }

    /// Frozen registry and canonicalization identity used for admission.
    #[must_use]
    pub const fn canonicalization(&self) -> &MachineClassCanonicalizationBinding {
        &self.canonicalization
    }

    /// Immutable hardware facts parsed from the strict receipt.
    #[must_use]
    pub const fn hardware(&self) -> &Value {
        &self.hardware
    }

    /// Explicit execution request parsed from the strict receipt.
    #[must_use]
    pub const fn execution_request(&self) -> &Value {
        &self.execution_request
    }

    /// Explicit start snapshot parsed from the strict receipt.
    #[must_use]
    pub const fn execution_start(&self) -> &Value {
        &self.execution_start
    }

    /// Explicit end snapshot parsed from the strict receipt.
    #[must_use]
    pub const fn execution_end(&self) -> &Value {
        &self.execution_end
    }

    /// Clean source, Cargo.lock, executable, and command identity.
    #[must_use]
    pub const fn build(&self) -> &Value {
        &self.build
    }

    /// Durability-symmetry facts parsed from the strict receipt.
    #[must_use]
    pub const fn durability(&self) -> &Value {
        &self.durability
    }

    /// Sealed runner-completion facts parsed from the strict receipt.
    #[must_use]
    pub const fn completion(&self) -> &Value {
        &self.completion
    }

    /// Recomputed canonical identity hashes.
    #[must_use]
    pub const fn derived_sha256(&self) -> &MachineClassDerivedHashes {
        &self.derived_sha256
    }

    /// Re-admit the stored exact receipt and compare every stored field.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection if the binding was mutated, stale, or was
    /// never admitted by the frozen registry.
    pub fn verify(&self) -> Result<(), MachineClassError> {
        let registry = MachineClassRegistry::frozen()?;
        let recomputed = registry.admit(self.receipt_json.as_bytes(), &self.admission_context)?;
        if recomputed == *self {
            Ok(())
        } else {
            Err(MachineClassError::new(
                MachineClassReason::DerivedHashMismatch,
                "stored runner binding does not equal exact re-admission",
            ))
        }
    }

    /// Whether two receipts name the same registry, hardware, class, and
    /// stable execution identity. Exact receipt digests may differ across
    /// independent runs.
    #[must_use]
    pub fn same_execution_identity(&self, other: &Self) -> bool {
        self.canonicalization == other.canonicalization
            && self.canonical_class_id == other.canonical_class_id
            && self.derived_sha256.hardware == other.derived_sha256.hardware
            && self.derived_sha256.identity == other.derived_sha256.identity
            && self.durability == other.durability
    }
}

/// Explicit machine binding carried by current evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MachineClassEvidenceBinding {
    /// Exact runner bytes were admitted and remain self-verifiable.
    Verified {
        /// Strict verified identity.
        identity: Box<VerifiedRunnerIdentity>,
    },
    /// Evidence is durable for diagnosis but cannot promote.
    Unverified {
        /// Stable bounded explanation.
        reason: String,
    },
}

impl MachineClassEvidenceBinding {
    /// Construct an explicit nonpromotable binding.
    #[must_use]
    pub fn unverified(reason: impl Into<String>) -> Self {
        let mut reason = reason.into();
        reason.truncate(240);
        Self::Unverified { reason }
    }

    /// Construct a verified binding.
    #[must_use]
    pub fn verified(identity: VerifiedRunnerIdentity) -> Self {
        Self::Verified {
            identity: Box::new(identity),
        }
    }

    /// Return the verified identity, if present.
    #[must_use]
    pub fn identity(&self) -> Option<&VerifiedRunnerIdentity> {
        match self {
            Self::Verified { identity } => Some(identity.as_ref()),
            Self::Unverified { .. } => None,
        }
    }

    /// Validate the explicit binding.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection for a stale verified identity or an empty
    /// unverified reason.
    pub fn validate(&self) -> Result<(), MachineClassError> {
        match self {
            Self::Verified { identity } => identity.verify(),
            Self::Unverified { reason } if reason.trim().is_empty() => Err(MachineClassError::new(
                MachineClassReason::MissingField,
                "unverified machine binding requires a reason",
            )),
            Self::Unverified { .. } => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFingerprint {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryClassRule {
    family: String,
    id_kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    id_pattern: Option<String>,
    #[serde(default)]
    width_capture: Option<u64>,
    admission_state: String,
    admission_reason: MachineClassReason,
    hardware_predicates: Map<String, Value>,
    #[serde(default)]
    execution_predicates: Option<Map<String, Value>>,
    source_fingerprints: Vec<SourceFingerprint>,
    #[serde(default, rename = "notes")]
    _notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerHardware {
    os: String,
    arch: String,
    cpu_vendor: String,
    cpu_family: Option<u64>,
    cpu_model: Option<u64>,
    cpu_stepping: Option<u64>,
    cpu_model_name: String,
    physical_cores: u64,
    logical_cpus: u64,
    numa_nodes: u64,
    memory_bytes: u64,
    page_size_bytes: u64,
    performance_cores: Option<u64>,
    efficiency_cores: Option<u64>,
    topology_sha256: String,
    fingerprint_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerExecutionRequest {
    requested_logical_cpu_ids: Vec<u64>,
    requested_physical_core_width: u64,
    thread_budget: u64,
    apple_execution_mode: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerExecutionSnapshot {
    observed_logical_cpu_ids: Vec<u64>,
    effective_physical_core_ids: Vec<String>,
    cpu_assignment_observability: String,
    effective_cpuset_sha256: String,
    threads_per_core: u64,
    smt_state: String,
    numa_node_ids: Vec<u64>,
    numa_policy: String,
    governor: String,
    thermal_pressure: bool,
    exclusive_lease: bool,
    exclusive_lease_id: String,
    local_execution: bool,
    observed_hardware_fingerprint_sha256: String,
    snapshot_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerExecution {
    request: RunnerExecutionRequest,
    start: RunnerExecutionSnapshot,
    end: RunnerExecutionSnapshot,
    identity_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerBuild {
    git_revision: String,
    git_dirty: bool,
    worktree_state_sha256: Option<String>,
    cargo_lock_sha256: String,
    executable_sha256: String,
    command_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerDurability {
    adjacent: bool,
    control_treatment: String,
    candidate_treatment: String,
    symmetric: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerCompletion {
    verified: bool,
    exit_status: i64,
    run_log_sha256: String,
    artifact_manifest_sha256: String,
    artifact_digests_verified: bool,
    started_at_utc: String,
    finished_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerReceipt {
    requested_class_id: String,
    derived_class_id: String,
    registry_sha256: String,
    hardware: RunnerHardware,
    execution: RunnerExecution,
    build: RunnerBuild,
    durability: RunnerDurability,
    completion: RunnerCompletion,
}

#[derive(Debug, Clone)]
struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(|number| StrictValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let StrictValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

pub fn parse_strict_json(bytes: &[u8]) -> Result<Value, MachineClassError> {
    serde_json::from_slice::<StrictValue>(bytes)
        .map(|strict| strict.0)
        .map_err(|error| {
            let detail = error.to_string();
            let reason = if detail.contains("duplicate key") {
                MachineClassReason::DuplicateKey
            } else {
                MachineClassReason::MissingField
            };
            MachineClassError::new(reason, detail)
        })
}

fn required_object<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a Map<String, Value>, MachineClassError> {
    value.get(field).and_then(Value::as_object).ok_or_else(|| {
        MachineClassError::new(
            MachineClassReason::MissingField,
            format!("registry field {field:?} must be an object"),
        )
    })
}

fn required_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], MachineClassError> {
    value.get(field).and_then(Value::as_array).map_or_else(
        || {
            Err(MachineClassError::new(
                MachineClassReason::MissingField,
                format!("registry field {field:?} must be an array"),
            ))
        },
        |values| Ok(values.as_slice()),
    )
}

fn validate_unknown_fields(
    candidate: &Value,
    schemas: &[&Value],
    path: &str,
) -> Result<(), MachineClassError> {
    match candidate {
        Value::Object(object) => {
            let schema_objects = schemas
                .iter()
                .filter_map(|schema| schema.as_object())
                .collect::<Vec<_>>();
            for (key, child) in object {
                let child_schemas = schema_objects
                    .iter()
                    .filter_map(|schema| schema.get(key))
                    .collect::<Vec<_>>();
                if child_schemas.is_empty() {
                    return Err(MachineClassError::new(
                        MachineClassReason::UnknownField,
                        format!("{path}.{key} is not in the strict schema"),
                    ));
                }
                validate_unknown_fields(child, &child_schemas, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(values) => {
            let child_schemas = schemas
                .iter()
                .filter_map(|schema| schema.as_array())
                .flat_map(|values| values.iter())
                .collect::<Vec<_>>();
            for (index, child) in values.iter().enumerate() {
                validate_unknown_fields(child, &child_schemas, &format!("{path}.{index}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn embedded_source(path: &str) -> Option<&'static [u8]> {
    match path {
        "docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/provenance.json" => {
            Some(TRJ_PROVENANCE_BYTES)
        }
        "docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/lscpu.txt" => Some(TRJ_LSCPU_BYTES),
        "docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/numactl-H.txt" => {
            Some(TRJ_NUMACTL_BYTES)
        }
        "docs/evidence/e8h/fingerprints/m4-macos-20260728/provenance.json" => {
            Some(M4_PROVENANCE_BYTES)
        }
        "docs/evidence/e8h/fingerprints/m4-macos-20260728/sysctl.txt" => Some(M4_SYSCTL_BYTES),
        _ => None,
    }
}

fn safe_relative_source_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Loaded, self-consistent exact machine-class registry.
#[derive(Debug, Clone)]
pub struct MachineClassRegistry {
    #[cfg(test)]
    raw: Value,
    classes: Vec<RegistryClassRule>,
    receipt_shapes: Vec<Value>,
    canonical_hash_contract_sha256: String,
}

impl MachineClassRegistry {
    /// Load and validate the compiled-in reviewed registry.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection for syntax, schema, collision, provenance,
    /// or exact-byte identity failure.
    pub fn frozen() -> Result<Self, MachineClassError> {
        Self::load_candidate(REGISTRY_BYTES, embedded_source)
    }

    fn load_candidate(
        bytes: &[u8],
        source: impl Fn(&str) -> Option<&'static [u8]>,
    ) -> Result<Self, MachineClassError> {
        let raw = parse_strict_json(bytes)?;
        let schema = parse_strict_json(REGISTRY_BYTES)?;
        validate_unknown_fields(&raw, &[&schema], "$")?;

        let root = raw.as_object().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::MissingField,
                "registry root must be an object",
            )
        })?;
        for field in [
            "schema_version",
            "owner_bead",
            "contract_status",
            "unknown_field_policy",
            "duplicate_key_policy",
            "class_id_semantics",
            "reason_codes",
            "validation_precedence",
            "canonical_hash_contract",
            "receipt_contract",
            "classes",
            "fact_templates",
            "requirements",
            "class_lookup_vectors",
            "test_vector_mutation_format",
            "test_vector_expected_contract",
            "test_vectors",
            "registry_test_vector_mutation_format",
            "registry_test_vectors",
        ] {
            if !root.contains_key(field) {
                return Err(MachineClassError::new(
                    MachineClassReason::MissingField,
                    format!("registry is missing required field {field:?}"),
                ));
            }
        }
        let found_schema = raw
            .get("schema_version")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if found_schema != MACHINE_CLASS_REGISTRY_SCHEMA_VERSION {
            return Err(MachineClassError::new(
                MachineClassReason::SourceIdentityInvalid,
                format!("unsupported registry schema {found_schema:?}"),
            ));
        }

        let classes = required_array(&raw, "classes")?
            .iter()
            .map(|value| {
                serde_json::from_value::<RegistryClassRule>(value.clone()).map_err(|error| {
                    let reason = if error.to_string().contains("unknown field") {
                        MachineClassReason::UnknownField
                    } else {
                        MachineClassReason::MissingField
                    };
                    MachineClassError::new(reason, error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::validate_class_rules(&classes, &source)?;
        let receipt_shapes = required_object(&raw, "fact_templates")?
            .values()
            .cloned()
            .collect::<Vec<_>>();
        if receipt_shapes.is_empty() {
            return Err(MachineClassError::new(
                MachineClassReason::MissingField,
                "registry requires at least one complete receipt template",
            ));
        }
        let canonical_hash_contract_sha256 =
            hash_value(root.get("canonical_hash_contract").ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::MissingField,
                    "registry is missing canonical_hash_contract",
                )
            })?)?;

        if sha256_hex(bytes) != MACHINE_CLASS_REGISTRY_SHA256 {
            return Err(MachineClassError::new(
                MachineClassReason::RegistryHashMismatch,
                "registry bytes do not match the reviewed SHA-256",
            ));
        }
        Ok(Self {
            #[cfg(test)]
            raw,
            classes,
            receipt_shapes,
            canonical_hash_contract_sha256,
        })
    }

    fn validate_class_rules(
        classes: &[RegistryClassRule],
        source: &impl Fn(&str) -> Option<&'static [u8]>,
    ) -> Result<(), MachineClassError> {
        if classes.is_empty() {
            return Err(MachineClassError::new(
                MachineClassReason::MissingField,
                "registry has no class rules",
            ));
        }
        for class in classes {
            match class.id_kind.as_str() {
                "exact" if class.id.is_some() && class.id_pattern.is_none() => {}
                "pattern"
                    if class.id.is_none()
                        && class.id_pattern.as_deref()
                            == Some("^trj-zen3-([1-9]|[1-5][0-9]|6[0-4])c(?:-smt2)?$")
                        && class.width_capture == Some(1) => {}
                _ => {
                    return Err(MachineClassError::new(
                        MachineClassReason::SourceIdentityInvalid,
                        format!("class family {:?} has an invalid ID rule", class.family),
                    ));
                }
            }
            if class.admission_state == "registered" && class.source_fingerprints.is_empty() {
                return Err(MachineClassError::new(
                    MachineClassReason::SourceIdentityInvalid,
                    format!("registered class {:?} has no provenance", class.family),
                ));
            }
            for fingerprint in &class.source_fingerprints {
                if !safe_relative_source_path(&fingerprint.path) || !is_sha256(&fingerprint.sha256)
                {
                    return Err(MachineClassError::new(
                        MachineClassReason::SourceIdentityInvalid,
                        format!("invalid source fingerprint {:?}", fingerprint.path),
                    ));
                }
                let bytes = source(&fingerprint.path).ok_or_else(|| {
                    MachineClassError::new(
                        MachineClassReason::SourceIdentityInvalid,
                        format!("unbound source fingerprint {:?}", fingerprint.path),
                    )
                })?;
                if sha256_hex(bytes) != fingerprint.sha256 {
                    return Err(MachineClassError::new(
                        MachineClassReason::SourceIdentityInvalid,
                        format!("source fingerprint content mismatch {:?}", fingerprint.path),
                    ));
                }
            }
        }

        let mut representatives = classes
            .iter()
            .filter_map(|class| class.id.as_deref())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for width in 1..=64 {
            representatives.push(format!("trj-zen3-{width}c"));
            representatives.push(format!("trj-zen3-{width}c-smt2"));
        }
        for representative in representatives {
            let matches = classes
                .iter()
                .filter(|class| class_matches_id(class, &representative))
                .count();
            if matches > 1 {
                return Err(MachineClassError::new(
                    MachineClassReason::AmbiguousClassId,
                    format!("class ID {representative:?} matches {matches} rules"),
                ));
            }
        }
        Ok(())
    }

    /// Resolve a class without admitting a runner receipt.
    #[must_use]
    pub fn lookup(&self, requested_class_id: &str) -> MachineClassLookup {
        match self.resolve(requested_class_id) {
            Ok(resolved) => {
                if resolved.rule.admission_state == "registered" {
                    MachineClassLookup {
                        decision: MachineClassDecision::Allow,
                        class_id: Some(requested_class_id.to_owned()),
                        reason: MachineClassReason::Admitted,
                    }
                } else {
                    MachineClassLookup {
                        decision: MachineClassDecision::DiagnosticOnly,
                        class_id: Some(requested_class_id.to_owned()),
                        reason: resolved.rule.admission_reason,
                    }
                }
            }
            Err(error) => MachineClassLookup {
                decision: MachineClassDecision::Reject,
                class_id: None,
                reason: error.reason,
            },
        }
    }

    fn resolve(&self, requested_class_id: &str) -> Result<ResolvedClass<'_>, MachineClassError> {
        if requested_class_id == "trj-zen-128c" {
            return Err(MachineClassError::new(
                MachineClassReason::ObsoleteClassId,
                "trj-zen-128c is historical fingerprint provenance only",
            ));
        }
        let matches = self
            .classes
            .iter()
            .filter(|class| class_matches_id(class, requested_class_id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(MachineClassError::new(
                MachineClassReason::UnknownClassId,
                format!("no class rule matches {requested_class_id:?}"),
            )),
            [rule] => {
                let trj = parse_trj_class_id(requested_class_id);
                Ok(ResolvedClass { rule, trj })
            }
            _ => Err(MachineClassError::new(
                MachineClassReason::AmbiguousClassId,
                format!("multiple class rules match {requested_class_id:?}"),
            )),
        }
    }

    /// Strictly admit exact runner receipt bytes.
    ///
    /// # Errors
    ///
    /// Returns exactly one stable rejection reason using the registry-defined
    /// precedence. Diagnostic-only classes are rejected for promotion.
    pub fn admit(
        &self,
        receipt_bytes: &[u8],
        context: &MachineClassAdmissionContext,
    ) -> Result<VerifiedRunnerIdentity, MachineClassError> {
        let receipt_value = parse_strict_json(receipt_bytes)?;
        let receipt_schema_refs = self.receipt_shapes.iter().collect::<Vec<_>>();
        validate_unknown_fields(&receipt_value, &receipt_schema_refs, "$receipt")?;
        prevalidate_cpuset_types(&receipt_value)?;
        let receipt = serde_json::from_value::<RunnerReceipt>(receipt_value).map_err(|error| {
            let detail = error.to_string();
            let reason = if detail.contains("unknown field") {
                MachineClassReason::UnknownField
            } else if detail.contains("missing field") {
                MachineClassReason::MissingField
            } else {
                MachineClassReason::SourceIdentityInvalid
            };
            MachineClassError::new(reason, detail)
        })?;

        if receipt.registry_sha256 != MACHINE_CLASS_REGISTRY_SHA256 {
            return Err(MachineClassError::new(
                MachineClassReason::RegistryHashMismatch,
                "runner receipt names a different registry SHA-256",
            ));
        }
        let derived = derive_hashes(&receipt)?;
        if receipt.hardware.fingerprint_sha256 != derived.hardware
            || receipt.execution.start.observed_hardware_fingerprint_sha256 != derived.hardware
            || receipt.execution.end.observed_hardware_fingerprint_sha256 != derived.hardware
            || receipt.execution.start.effective_cpuset_sha256 != derived.start_cpuset
            || receipt.execution.end.effective_cpuset_sha256
                != derive_cpuset_hash(&receipt.execution.end)?
            || receipt.execution.start.snapshot_sha256 != derived.start_snapshot
            || receipt.execution.end.snapshot_sha256 != derived.end_snapshot
            || receipt.execution.identity_sha256 != derived.identity
        {
            return Err(MachineClassError::new(
                MachineClassReason::DerivedHashMismatch,
                "one or more runner-derived hashes do not recompute",
            ));
        }

        let resolved = self.resolve(&receipt.requested_class_id)?;
        if resolved.rule.admission_state != "registered" {
            return Err(MachineClassError::new(
                resolved.rule.admission_reason,
                format!(
                    "class {:?} is {}",
                    receipt.requested_class_id, resolved.rule.admission_state
                ),
            ));
        }
        if receipt.derived_class_id != receipt.requested_class_id {
            return Err(MachineClassError::new(
                MachineClassReason::ReceiptClassMismatch,
                "requested and derived class IDs differ",
            ));
        }

        validate_hardware(&receipt.hardware, resolved.rule)?;
        validate_execution(&receipt, &resolved)?;
        if receipt.execution.start != receipt.execution.end {
            return Err(MachineClassError::new(
                MachineClassReason::PrePostIdentityDrift,
                "start and end execution snapshots differ",
            ));
        }
        if !receipt.execution.start.exclusive_lease
            || receipt.execution.start.exclusive_lease_id.trim().is_empty()
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExclusiveLeaseMissing,
                "timed evidence requires a named exclusive lease",
            ));
        }
        if !receipt.execution.start.local_execution {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionOffloaded,
                "timed evidence must execute locally",
            ));
        }
        validate_source_identity(&receipt)?;
        validate_durability(&receipt.durability)?;
        validate_completion(&receipt.completion)?;
        validate_destination(context, &receipt.derived_class_id)?;

        let receipt_json = std::str::from_utf8(receipt_bytes)
            .map_err(|error| {
                MachineClassError::new(MachineClassReason::MissingField, error.to_string())
            })?
            .to_owned();
        Ok(VerifiedRunnerIdentity {
            receipt_json,
            receipt_sha256: sha256_hex(receipt_bytes),
            admission_context: context.clone(),
            canonicalization: MachineClassCanonicalizationBinding {
                registry_schema_version: MACHINE_CLASS_REGISTRY_SCHEMA_VERSION.to_owned(),
                registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
                registry_git_blob: MACHINE_CLASS_REGISTRY_GIT_BLOB.to_owned(),
                canonical_hash_contract_sha256: self.canonical_hash_contract_sha256.clone(),
            },
            canonical_class_id: receipt.derived_class_id.clone(),
            hardware: serde_json::to_value(&receipt.hardware).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            execution_request: serde_json::to_value(&receipt.execution.request).map_err(
                |error| {
                    MachineClassError::new(
                        MachineClassReason::DerivedHashMismatch,
                        error.to_string(),
                    )
                },
            )?,
            execution_start: serde_json::to_value(&receipt.execution.start).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            execution_end: serde_json::to_value(&receipt.execution.end).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            build: serde_json::to_value(&receipt.build).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            durability: serde_json::to_value(&receipt.durability).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            completion: serde_json::to_value(&receipt.completion).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
            derived_sha256: derived,
        })
    }

    /// Admit first, then invoke a mutation callback only on success.
    ///
    /// # Errors
    ///
    /// Returns the exact admission rejection without invoking `on_allow`.
    pub fn admit_then<T>(
        &self,
        receipt_bytes: &[u8],
        context: &MachineClassAdmissionContext,
        on_allow: impl FnOnce(&VerifiedRunnerIdentity) -> T,
    ) -> Result<T, MachineClassError> {
        let identity = self.admit(receipt_bytes, context)?;
        Ok(on_allow(&identity))
    }

    #[cfg(test)]
    fn raw(&self) -> &Value {
        &self.raw
    }
}

struct ResolvedClass<'a> {
    rule: &'a RegistryClassRule,
    trj: Option<TrjClass>,
}

#[derive(Debug, Clone, Copy)]
struct TrjClass {
    width: u64,
    threads_per_core: u64,
}

fn parse_trj_class_id(value: &str) -> Option<TrjClass> {
    let suffix = value.strip_prefix("trj-zen3-")?;
    let (width, threads_per_core) = suffix.strip_suffix("c-smt2").map_or_else(
        || suffix.strip_suffix('c').map(|width| (width, 1)),
        |width| Some((width, 2)),
    )?;
    if width.starts_with('0') {
        return None;
    }
    let width = width.parse::<u64>().ok()?;
    (1..=64).contains(&width).then_some(TrjClass {
        width,
        threads_per_core,
    })
}

fn class_matches_id(class: &RegistryClassRule, value: &str) -> bool {
    match class.id_kind.as_str() {
        "exact" => class.id.as_deref() == Some(value),
        "pattern" => class.id_pattern.is_some() && parse_trj_class_id(value).is_some(),
        _ => false,
    }
}

fn prevalidate_cpuset_types(value: &Value) -> Result<(), MachineClassError> {
    for path in [
        ["execution", "request", "requested_logical_cpu_ids"],
        ["execution", "start", "observed_logical_cpu_ids"],
        ["execution", "end", "observed_logical_cpu_ids"],
    ] {
        let mut cursor = value;
        let mut missing = false;
        for component in path {
            if let Some(next) = cursor.get(component) {
                cursor = next;
            } else {
                missing = true;
                break;
            }
        }
        if !missing
            && !cursor
                .as_array()
                .is_some_and(|values| values.iter().all(|value| value.as_u64().is_some()))
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionCpusetInvalid,
                format!("{} must be an array of unsigned CPU IDs", path.join(".")),
            ));
        }
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, MachineClassError> {
    fn check_ascii(value: &Value) -> bool {
        match value {
            Value::String(value) => value.is_ascii(),
            Value::Array(values) => values.iter().all(check_ascii),
            Value::Object(values) => {
                values.keys().all(|key| key.is_ascii()) && values.values().all(check_ascii)
            }
            _ => true,
        }
    }
    if !check_ascii(value) {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "canonical machine identity JSON must contain only ASCII strings",
        ));
    }
    serde_json::to_vec(value).map_err(|error| {
        MachineClassError::new(MachineClassReason::SourceIdentityInvalid, error.to_string())
    })
}

fn hash_value(value: &Value) -> Result<String, MachineClassError> {
    Ok(sha256_hex(&canonical_json_bytes(value)?))
}

fn derive_hashes(receipt: &RunnerReceipt) -> Result<MachineClassDerivedHashes, MachineClassError> {
    let mut hardware = serde_json::to_value(&receipt.hardware).map_err(|error| {
        MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
    })?;
    hardware
        .as_object_mut()
        .expect("serialized hardware is an object")
        .remove("fingerprint_sha256");

    let start_cpuset = derive_cpuset_hash(&receipt.execution.start)?;
    let start_snapshot = derive_snapshot_hash(&receipt.execution.start)?;
    let end_snapshot = derive_snapshot_hash(&receipt.execution.end)?;
    let identity = derive_execution_identity_hash(receipt)?;
    Ok(MachineClassDerivedHashes {
        hardware: hash_value(&hardware)?,
        start_cpuset,
        start_snapshot,
        end_snapshot,
        identity,
    })
}

fn derive_cpuset_hash(snapshot: &RunnerExecutionSnapshot) -> Result<String, MachineClassError> {
    let value = serde_json::json!({
        "observed_logical_cpu_ids": snapshot.observed_logical_cpu_ids,
        "effective_physical_core_ids": snapshot.effective_physical_core_ids,
        "cpu_assignment_observability": snapshot.cpu_assignment_observability,
        "threads_per_core": snapshot.threads_per_core,
        "smt_state": snapshot.smt_state,
    });
    hash_value(&value)
}

fn derive_snapshot_hash(snapshot: &RunnerExecutionSnapshot) -> Result<String, MachineClassError> {
    let mut value = serde_json::to_value(snapshot).map_err(|error| {
        MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
    })?;
    value
        .as_object_mut()
        .expect("serialized snapshot is an object")
        .remove("snapshot_sha256");
    hash_value(&value)
}

fn derive_execution_identity_hash(receipt: &RunnerReceipt) -> Result<String, MachineClassError> {
    let mut stable_start = serde_json::to_value(&receipt.execution.start).map_err(|error| {
        MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
    })?;
    let stable_start = stable_start
        .as_object_mut()
        .expect("serialized snapshot is an object");
    stable_start.remove("thermal_pressure");
    stable_start.remove("snapshot_sha256");
    let value = serde_json::json!({
        "requested_class_id": receipt.requested_class_id,
        "derived_class_id": receipt.derived_class_id,
        "request": receipt.execution.request,
        "stable_execution": stable_start,
    });
    hash_value(&value)
}

fn predicate_matches(
    actual: &Value,
    predicates: &Map<String, Value>,
    field: &str,
    reason: MachineClassReason,
) -> Result<(), MachineClassError> {
    if let Some(expected) = predicates.get(field)
        && actual != expected
    {
        return Err(MachineClassError::new(
            reason,
            format!("{field} observed {actual}, expected {expected}"),
        ));
    }
    Ok(())
}

fn validate_hardware(
    hardware: &RunnerHardware,
    class: &RegistryClassRule,
) -> Result<(), MachineClassError> {
    let value = serde_json::to_value(hardware).map_err(|error| {
        MachineClassError::new(MachineClassReason::SourceIdentityInvalid, error.to_string())
    })?;
    let value = value
        .as_object()
        .expect("serialized runner hardware is an object");
    for (field, reason) in [
        ("os", MachineClassReason::HardwareOsMismatch),
        ("arch", MachineClassReason::HardwareArchMismatch),
        ("cpu_vendor", MachineClassReason::HardwareCpuVendorMismatch),
        ("cpu_family", MachineClassReason::HardwareCpuFamilyMismatch),
        ("cpu_model", MachineClassReason::HardwareCpuModelMismatch),
        (
            "cpu_stepping",
            MachineClassReason::HardwareCpuSteppingMismatch,
        ),
        (
            "cpu_model_name",
            MachineClassReason::HardwareCpuNameMismatch,
        ),
        (
            "physical_cores",
            MachineClassReason::HardwareTopologyMismatch,
        ),
        ("logical_cpus", MachineClassReason::HardwareTopologyMismatch),
        ("numa_nodes", MachineClassReason::HardwareNumaMismatch),
        ("memory_bytes", MachineClassReason::HardwareMemoryMismatch),
        (
            "page_size_bytes",
            MachineClassReason::HardwarePageSizeMismatch,
        ),
        (
            "performance_cores",
            MachineClassReason::HardwarePerformanceCoreMismatch,
        ),
        (
            "efficiency_cores",
            MachineClassReason::HardwareEfficiencyCoreMismatch,
        ),
    ] {
        predicate_matches(&value[field], &class.hardware_predicates, field, reason)?;
    }
    if !is_sha256(&hardware.topology_sha256) {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "topology_sha256 is not lowercase SHA-256",
        ));
    }
    Ok(())
}

fn validate_execution(
    receipt: &RunnerReceipt,
    resolved: &ResolvedClass<'_>,
) -> Result<(), MachineClassError> {
    let request = &receipt.execution.request;
    let start = &receipt.execution.start;
    let end = &receipt.execution.end;
    if let Some(trj) = resolved.trj {
        if start.threads_per_core != trj.threads_per_core
            || end.threads_per_core != trj.threads_per_core
            || start.smt_state
                != if trj.threads_per_core == 2 {
                    "on"
                } else {
                    "off"
                }
            || end.smt_state != start.smt_state
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionSmtMismatch,
                "Threadripper class suffix, SMT state, and threads-per-core disagree",
            ));
        }
        validate_cpu_ids(
            &request.requested_logical_cpu_ids,
            receipt.hardware.logical_cpus,
        )?;
        validate_cpu_ids(
            &start.observed_logical_cpu_ids,
            receipt.hardware.logical_cpus,
        )?;
        validate_cpu_ids(&end.observed_logical_cpu_ids, receipt.hardware.logical_cpus)?;
        let expected_logical = trj.width * trj.threads_per_core;
        let physical_count = unique_string_count(&start.effective_physical_core_ids);
        if request.requested_physical_core_width != trj.width
            || request.requested_logical_cpu_ids.len()
                != usize::try_from(expected_logical).unwrap_or(usize::MAX)
            || start.observed_logical_cpu_ids.len()
                != usize::try_from(expected_logical).unwrap_or(usize::MAX)
            || physical_count != usize::try_from(trj.width).unwrap_or(usize::MAX)
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionWidthMismatch,
                "Threadripper class width does not match requested and observed CPUs",
            ));
        }
        if start.cpu_assignment_observability != "affinity-enforced"
            || end.cpu_assignment_observability != "affinity-enforced"
            || request.requested_logical_cpu_ids != start.observed_logical_cpu_ids
            || request.requested_logical_cpu_ids != end.observed_logical_cpu_ids
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionCpusetInvalid,
                "Threadripper requested and observed affinity must be explicit and equal",
            ));
        }
        if request.thread_budget == 0 || request.thread_budget > expected_logical {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionThreadBudgetInvalid,
                "thread budget is outside the admitted CPU pool",
            ));
        }
        if request.apple_execution_mode != "not-applicable" {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionModeMismatch,
                "Threadripper receipt must use not-applicable Apple mode",
            ));
        }
        if start.numa_node_ids != [0]
            || end.numa_node_ids != [0]
            || start.numa_policy != "bind:0"
            || end.numa_policy != "bind:0"
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionNumaMismatch,
                "Threadripper evidence requires NUMA bind:0",
            ));
        }
        if start.governor != "performance" || end.governor != "performance" {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionGovernorMismatch,
                "Threadripper evidence requires the performance governor",
            ));
        }
    } else if resolved.rule.family == "m4-macos" {
        let max_width = match request.apple_execution_mode.as_str() {
            "p-only" => 10,
            "p-plus-e" => 14,
            _ => {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionModeMismatch,
                    "M4 execution mode must be p-only or p-plus-e",
                ));
            }
        };
        if request.requested_physical_core_width != max_width {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionWidthMismatch,
                "M4 requested width does not match execution mode",
            ));
        }
        if request.thread_budget == 0 || request.thread_budget > max_width {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionThreadBudgetInvalid,
                "M4 thread budget is outside the requested pool",
            ));
        }
        if start.threads_per_core != 1
            || end.threads_per_core != 1
            || start.smt_state != "not-applicable"
            || end.smt_state != "not-applicable"
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionSmtMismatch,
                "M4 receipts cannot claim SMT",
            ));
        }
        if !["unavailable", "scheduler-observed", "affinity-enforced"]
            .contains(&start.cpu_assignment_observability.as_str())
            || start.cpu_assignment_observability != end.cpu_assignment_observability
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionCpusetInvalid,
                "M4 CPU assignment observability is invalid or drifted",
            ));
        }
        if start.numa_node_ids != [0]
            || end.numa_node_ids != [0]
            || start.numa_policy != "system"
            || end.numa_policy != "system"
        {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionNumaMismatch,
                "M4 evidence requires system NUMA policy",
            ));
        }
        if start.governor != "not-applicable" || end.governor != "not-applicable" {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionGovernorMismatch,
                "M4 receipts must use not-applicable governor",
            ));
        }
        if start.thermal_pressure || end.thermal_pressure {
            return Err(MachineClassError::new(
                MachineClassReason::ThermalPressure,
                "thermal pressure invalidates timed M4 evidence",
            ));
        }
    }
    let _ = &resolved.rule.execution_predicates;
    Ok(())
}

fn validate_cpu_ids(ids: &[u64], logical_cpus: u64) -> Result<(), MachineClassError> {
    let unique = ids.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != ids.len() || ids.iter().any(|cpu| *cpu >= logical_cpus) {
        return Err(MachineClassError::new(
            MachineClassReason::ExecutionCpusetInvalid,
            "CPU IDs must be unique and inside the registered logical CPU range",
        ));
    }
    Ok(())
}

fn unique_string_count(values: &[String]) -> usize {
    values.iter().collect::<BTreeSet<_>>().len()
}

fn validate_source_identity(receipt: &RunnerReceipt) -> Result<(), MachineClassError> {
    if receipt.build.git_dirty {
        return Err(MachineClassError::new(
            MachineClassReason::SourceDirty,
            "promotion requires an exact clean source tree",
        ));
    }
    if receipt.build.worktree_state_sha256.is_some()
        || receipt.build.git_revision.trim().is_empty()
        || !is_sha256(&receipt.build.cargo_lock_sha256)
        || !is_sha256(&receipt.build.executable_sha256)
        || !is_sha256(&receipt.build.command_sha256)
    {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "build source, Cargo.lock, executable, or command identity is invalid",
        ));
    }
    Ok(())
}

fn validate_durability(durability: &RunnerDurability) -> Result<(), MachineClassError> {
    let symmetric = durability.symmetric
        && (!durability.adjacent || durability.control_treatment == durability.candidate_treatment);
    if !symmetric {
        return Err(MachineClassError::new(
            MachineClassReason::DurabilityAsymmetric,
            "durability-adjacent arms must use identical sync treatment",
        ));
    }
    Ok(())
}

fn validate_completion(completion: &RunnerCompletion) -> Result<(), MachineClassError> {
    if !completion.verified
        || !completion.artifact_digests_verified
        || !is_sha256(&completion.run_log_sha256)
        || !is_sha256(&completion.artifact_manifest_sha256)
    {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "runner completion and artifact digests must be verified",
        ));
    }
    if completion.exit_status != 0 {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionFailed,
            "runner completion exit status is nonzero",
        ));
    }
    if completion.started_at_utc.trim().is_empty() || completion.finished_at_utc.trim().is_empty() {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "runner completion timestamps are missing",
        ));
    }
    Ok(())
}

fn validate_destination(
    context: &MachineClassAdmissionContext,
    class_id: &str,
) -> Result<(), MachineClassError> {
    let expected = format!("{}.{}.latest.json", context.gate, class_id);
    if context.destination_basename != expected {
        return Err(MachineClassError::new(
            MachineClassReason::DestinationIdentityMismatch,
            format!(
                "destination {:?} does not equal {expected:?}",
                context.destination_basename
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub fn admitted_test_identity_for(
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
) -> VerifiedRunnerIdentity {
    admitted_test_identity_for_run(
        gate,
        git_revision,
        cargo_lock_sha256,
        executable_sha256,
        command_sha256,
        "fixture",
    )
}

#[cfg(test)]
pub fn admitted_test_identity_for_run(
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    run_label: &str,
) -> VerifiedRunnerIdentity {
    admitted_test_identity_from_vector_for_run(
        "MCV-001-trj-registered",
        gate,
        git_revision,
        cargo_lock_sha256,
        executable_sha256,
        command_sha256,
        run_label,
    )
}

#[cfg(test)]
pub fn admitted_test_identity_from_vector_for_run(
    vector_id: &str,
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    run_label: &str,
) -> VerifiedRunnerIdentity {
    let registry = MachineClassRegistry::frozen().expect("frozen registry");
    let vector = registry
        .raw()
        .get("test_vectors")
        .and_then(Value::as_array)
        .and_then(|vectors| {
            vectors
                .iter()
                .find(|vector| vector.get("id").and_then(Value::as_str) == Some(vector_id))
        })
        .expect("registered test vector");
    let (bytes, mut context) = materialize_receipt_vector(registry.raw(), vector);
    let mut receipt: Value = serde_json::from_slice(&bytes).expect("materialized fixture JSON");
    set_path(
        &mut receipt,
        "build.git_revision",
        Value::String(git_revision.to_owned()),
    );
    set_path(
        &mut receipt,
        "build.cargo_lock_sha256",
        Value::String(cargo_lock_sha256.to_owned()),
    );
    set_path(
        &mut receipt,
        "build.executable_sha256",
        Value::String(executable_sha256.to_owned()),
    );
    set_path(
        &mut receipt,
        "build.command_sha256",
        Value::String(command_sha256.to_owned()),
    );
    set_path(
        &mut receipt,
        "completion.run_log_sha256",
        Value::String(sha256_hex(format!("runner-log:{run_label}").as_bytes())),
    );
    set_path(
        &mut receipt,
        "completion.artifact_manifest_sha256",
        Value::String(sha256_hex(
            format!("artifact-manifest:{run_label}").as_bytes(),
        )),
    );
    let class_id = receipt["derived_class_id"]
        .as_str()
        .expect("fixture class")
        .to_owned();
    context.gate = gate.to_owned();
    context.destination_basename = format!("{gate}.{class_id}.latest.json");
    let bytes = serde_json::to_vec(&receipt).expect("serialize customized fixture");
    registry.admit(&bytes, &context).expect("admitted fixture")
}

#[cfg(test)]
fn value_at_mut<'a>(value: &'a mut Value, path: &[&str]) -> Option<&'a mut Value> {
    let mut cursor = value;
    for component in path {
        cursor = if let Ok(index) = component.parse::<usize>() {
            cursor.as_array_mut()?.get_mut(index)?
        } else {
            cursor.as_object_mut()?.get_mut(*component)?
        };
    }
    Some(cursor)
}

#[cfg(test)]
fn set_path(value: &mut Value, path: &str, replacement: Value) {
    let components = path.split('.').collect::<Vec<_>>();
    let (last, parents) = components.split_last().expect("nonempty mutation path");
    let parent = value_at_mut(value, parents).expect("mutation parent exists");
    if let Ok(index) = last.parse::<usize>() {
        parent.as_array_mut().expect("array parent")[index] = replacement;
    } else {
        parent
            .as_object_mut()
            .expect("object parent")
            .insert((*last).to_owned(), replacement);
    }
}

#[cfg(test)]
fn remove_path(value: &mut Value, path: &str) {
    let components = path.split('.').collect::<Vec<_>>();
    let (last, parents) = components.split_last().expect("nonempty mutation path");
    let parent = value_at_mut(value, parents).expect("mutation parent exists");
    if let Ok(index) = last.parse::<usize>() {
        parent.as_array_mut().expect("array parent").remove(index);
    } else {
        parent.as_object_mut().expect("object parent").remove(*last);
    }
}

#[cfg(test)]
fn substitute_registry_sha256(value: &mut Value) {
    match value {
        Value::String(text) if text == "$REGISTRY_SHA256" => {
            *text = MACHINE_CLASS_REGISTRY_SHA256.to_owned();
        }
        Value::Array(values) => {
            for value in values {
                substitute_registry_sha256(value);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                substitute_registry_sha256(value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
fn derive_receipt_placeholders(value: &mut Value) {
    fn dynamic_hash_without(value: &Value, omitted: &str) -> String {
        let mut value = value.clone();
        value
            .as_object_mut()
            .expect("hash input object")
            .remove(omitted);
        hash_value(&value).expect("canonical hash")
    }
    let hardware = value.get("hardware").expect("hardware").clone();
    let hardware_hash = dynamic_hash_without(&hardware, "fingerprint_sha256");
    if value["hardware"]["fingerprint_sha256"] == "$DERIVE_HARDWARE_FINGERPRINT_SHA256" {
        value["hardware"]["fingerprint_sha256"] = Value::String(hardware_hash.clone());
    }
    for side in ["start", "end"] {
        if value["execution"][side]["observed_hardware_fingerprint_sha256"]
            == "$DERIVE_HARDWARE_FINGERPRINT_SHA256"
        {
            value["execution"][side]["observed_hardware_fingerprint_sha256"] =
                Value::String(hardware_hash.clone());
        }
        let snapshot = value["execution"][side].clone();
        let cpuset = serde_json::json!({
            "observed_logical_cpu_ids": snapshot["observed_logical_cpu_ids"],
            "effective_physical_core_ids": snapshot["effective_physical_core_ids"],
            "cpu_assignment_observability": snapshot["cpu_assignment_observability"],
            "threads_per_core": snapshot["threads_per_core"],
            "smt_state": snapshot["smt_state"],
        });
        let cpuset_hash = hash_value(&cpuset).expect("cpuset hash");
        if value["execution"][side]["effective_cpuset_sha256"]
            == if side == "start" {
                "$DERIVE_START_CPUSET_SHA256"
            } else {
                "$DERIVE_END_CPUSET_SHA256"
            }
        {
            value["execution"][side]["effective_cpuset_sha256"] = Value::String(cpuset_hash);
        }
        let snapshot = value["execution"][side].clone();
        let snapshot_hash = dynamic_hash_without(&snapshot, "snapshot_sha256");
        if value["execution"][side]["snapshot_sha256"]
            == if side == "start" {
                "$DERIVE_START_SNAPSHOT_SHA256"
            } else {
                "$DERIVE_END_SNAPSHOT_SHA256"
            }
        {
            value["execution"][side]["snapshot_sha256"] = Value::String(snapshot_hash);
        }
    }
    if value["execution"]["identity_sha256"] == "$DERIVE_EXECUTION_IDENTITY_SHA256" {
        let mut stable_start = value["execution"]["start"].clone();
        stable_start
            .as_object_mut()
            .expect("start object")
            .remove("thermal_pressure");
        stable_start
            .as_object_mut()
            .expect("start object")
            .remove("snapshot_sha256");
        let identity = serde_json::json!({
            "requested_class_id": value["requested_class_id"],
            "derived_class_id": value["derived_class_id"],
            "request": value["execution"]["request"],
            "stable_execution": stable_start,
        });
        value["execution"]["identity_sha256"] =
            Value::String(hash_value(&identity).expect("identity hash"));
    }
}

#[cfg(test)]
fn materialize_receipt_vector(
    registry: &Value,
    vector: &Value,
) -> (Vec<u8>, MachineClassAdmissionContext) {
    if let Some(raw) = vector.get("raw_json").and_then(Value::as_str) {
        return (
            raw.as_bytes().to_vec(),
            MachineClassAdmissionContext {
                gate: "QG-2".to_owned(),
                destination_basename: "QG-2.trj-zen3-16c.latest.json".to_owned(),
            },
        );
    }
    let template = vector
        .get("template")
        .and_then(Value::as_str)
        .expect("named template");
    let mut receipt = registry["fact_templates"][template].clone();
    substitute_registry_sha256(&mut receipt);
    for (path, replacement) in vector["set"].as_object().expect("set map") {
        set_path(&mut receipt, path, replacement.clone());
    }
    for path in vector["remove"].as_array().expect("remove array") {
        remove_path(&mut receipt, path.as_str().expect("remove path"));
    }
    for (path, replacement) in vector["add_unknown"].as_object().expect("unknown map") {
        set_path(&mut receipt, path, replacement.clone());
    }
    derive_receipt_placeholders(&mut receipt);
    let context = vector.get("admission_context").map_or_else(
        || MachineClassAdmissionContext {
            gate: "QG-2".to_owned(),
            destination_basename: format!(
                "QG-2.{}.latest.json",
                receipt["derived_class_id"].as_str().unwrap_or("unknown")
            ),
        },
        |value| serde_json::from_value(value.clone()).expect("admission context"),
    );
    (serde_json::to_vec(&receipt).expect("receipt JSON"), context)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn expected_reason(value: &Value) -> MachineClassReason {
        serde_json::from_value(value.clone()).expect("known expected reason")
    }

    #[test]
    fn frozen_registry_binds_exact_reviewed_bytes_and_corpus_counts() {
        assert_eq!(sha256_hex(REGISTRY_BYTES), MACHINE_CLASS_REGISTRY_SHA256);
        assert_eq!(
            MACHINE_CLASS_REGISTRY_GIT_BLOB,
            "83a904721c81a4e5d19c8164d4ffe3924f3afddd"
        );
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        assert_eq!(registry.raw()["classes"].as_array().unwrap().len(), 4);
        assert_eq!(
            registry.raw()["class_lookup_vectors"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(registry.raw()["test_vectors"].as_array().unwrap().len(), 51);
        assert_eq!(
            registry.raw()["registry_test_vectors"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn self_hosts_every_lookup_and_receipt_vector_with_zero_denial_writes() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        for vector in registry.raw()["class_lookup_vectors"]
            .as_array()
            .expect("lookup vectors")
        {
            let lookup = registry.lookup(
                vector["requested_class_id"]
                    .as_str()
                    .expect("requested class"),
            );
            assert_eq!(
                serde_json::to_value(lookup.decision).unwrap(),
                vector["expected"]["decision"],
                "{} decision",
                vector["id"]
            );
            assert_eq!(
                serde_json::to_value(lookup.class_id).unwrap(),
                vector["expected"]["class_id"],
                "{} class",
                vector["id"]
            );
            assert_eq!(
                lookup.reason,
                expected_reason(&vector["expected"]["reason"]),
                "{} reason",
                vector["id"]
            );
        }

        let mut allow_count = 0;
        let mut reject_count = 0;
        for vector in registry.raw()["test_vectors"]
            .as_array()
            .expect("receipt vectors")
        {
            let (bytes, context) = materialize_receipt_vector(registry.raw(), vector);
            let write_count = Cell::new(0_u64);
            let result = registry.admit_then(&bytes, &context, |identity| {
                write_count.set(write_count.get() + 1);
                identity.clone()
            });
            let expected = &vector["expected"];
            match expected["decision"].as_str().expect("decision") {
                "allow" => {
                    allow_count += 1;
                    let identity = result.unwrap_or_else(|error| {
                        panic!("{} unexpectedly rejected: {error}", vector["id"])
                    });
                    assert_eq!(write_count.get(), 1, "{} write count", vector["id"]);
                    assert_eq!(
                        Some(identity.class_id()),
                        expected["class_id"].as_str(),
                        "{} class",
                        vector["id"]
                    );
                    let golden: MachineClassDerivedHashes =
                        serde_json::from_value(expected["derived_sha256"].clone())
                            .expect("golden hashes");
                    assert_eq!(
                        identity.derived_sha256(),
                        &golden,
                        "{} golden hashes",
                        vector["id"]
                    );
                    identity.verify().expect("stored identity re-verifies");
                }
                "reject" => {
                    reject_count += 1;
                    let error = result.expect_err("negative vector must reject");
                    assert_eq!(
                        error.reason,
                        expected_reason(&expected["reason"]),
                        "{} exact reason",
                        vector["id"]
                    );
                    assert_eq!(write_count.get(), 0, "{} wrote on denial", vector["id"]);
                }
                other => panic!("unexpected receipt decision {other}"),
            }
        }
        assert_eq!(allow_count, 3);
        assert_eq!(reject_count, 48);
    }

    #[test]
    fn self_hosts_all_registry_rejection_vectors() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        for vector in registry.raw()["registry_test_vectors"]
            .as_array()
            .expect("registry vectors")
        {
            let bytes = vector.get("raw_json").and_then(Value::as_str).map_or_else(
                || {
                    let mut candidate = registry.raw().clone();
                    if let Some(class) = vector.get("add_class").filter(|value| !value.is_null()) {
                        candidate["classes"]
                            .as_array_mut()
                            .expect("classes")
                            .push(class.clone());
                    }
                    for (path, replacement) in vector["set"].as_object().expect("set map") {
                        set_path(&mut candidate, path, replacement.clone());
                    }
                    for path in vector["remove"].as_array().expect("remove array") {
                        remove_path(&mut candidate, path.as_str().expect("remove path"));
                    }
                    for (path, replacement) in
                        vector["add_unknown"].as_object().expect("unknown map")
                    {
                        set_path(&mut candidate, path, replacement.clone());
                    }
                    serde_json::to_vec(&candidate).expect("candidate registry")
                },
                |raw| raw.as_bytes().to_vec(),
            );
            let error = MachineClassRegistry::load_candidate(&bytes, embedded_source)
                .expect_err("registry vector must reject");
            assert_eq!(
                error.reason,
                expected_reason(&vector["expected"]["reason"]),
                "{} exact reason",
                vector["id"]
            );
        }
    }

    #[test]
    fn explicit_unverified_binding_is_durable_but_not_verified() {
        let binding = MachineClassEvidenceBinding::unverified("runner receipt missing");
        binding.validate().expect("bounded explicit reason");
        assert!(binding.identity().is_none());
        let json = serde_json::to_string(&binding).expect("serialize binding");
        let roundtrip: MachineClassEvidenceBinding =
            serde_json::from_str(&json).expect("deserialize binding");
        assert_eq!(roundtrip, binding);
    }
}
