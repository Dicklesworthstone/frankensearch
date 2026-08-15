//! Strict machine-class registry and runner-receipt admission.
//!
//! The normative registry is compiled into this crate and bound by both its
//! reviewed Git blob identity and its exact file SHA-256. Admission never
//! trusts a caller-supplied machine label: it derives the class and execution
//! identity from a duplicate-key-rejecting, unknown-field-rejecting runner
//! receipt, then compares an optional caller expectation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::perf::{
    PerfApplicabilityPlan, PerfApplicabilityPlanBinding, PerfGate, PerfMatrixSpec, PerfRawSample,
    PerfSampleArm, PerfSamplePhase,
};
use crate::perf_evidence::{EvidenceCellBody, PerfEvidenceArtifact};

/// Reviewed commit containing the normative registry.
pub const MACHINE_CLASS_REGISTRY_SPEC_COMMIT: &str = "d251ddea584519aba64922d9720e02d941a9385d";
/// Exact Git blob of the normative registry.
pub const MACHINE_CLASS_REGISTRY_GIT_BLOB: &str = "fe68e97c8e66accd0abaa9a4e3146134c271e964";
/// SHA-256 of the exact normative registry file bytes.
pub const MACHINE_CLASS_REGISTRY_SHA256: &str =
    "798338985ea28fc9b726bd2d8a260294777f5a701e012be91348465e5483b86c";
/// Registry schema accepted by this consumer.
pub const MACHINE_CLASS_REGISTRY_SCHEMA_VERSION: &str =
    "frankensearch.quill-machine-class-registry.v2";
/// Schema for the exact post-exit artifact manifest bound into one runner
/// completion receipt.
pub const RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION: &str =
    "frankensearch.perf-runner-artifact-manifest.v3";
/// Strict schema carried by every typed runner-completion receipt.
pub const RUNNER_RECEIPT_SCHEMA_VERSION: &str = "frankensearch.perf-runner-completion.v6";
/// Build-time and executing-ELF identity required from the typed local
/// performance producer.
pub const LOCAL_PERF_PRODUCER_CONTRACT_VERSION: &str = "frankensearch.quill-local-perf-producer.v4";

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
    /// No execution profile matches the requested hardware class and profile ID.
    UnknownExecutionProfile,
    /// The named profile belongs to a different immutable hardware class.
    ExecutionProfileClassMismatch,
    /// The named profile is registered as unavailable.
    ExecutionProfileUnavailable,
    /// The execution-profile contract is internally inconsistent.
    ExecutionProfileContractInvalid,
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
    /// Runtime-detected CPU ISA features are malformed, forbidden, or differ.
    HardwareIsaMismatch,
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
            Self::UnknownExecutionProfile => "unknown-execution-profile",
            Self::ExecutionProfileClassMismatch => "execution-profile-class-mismatch",
            Self::ExecutionProfileUnavailable => "execution-profile-unavailable",
            Self::ExecutionProfileContractInvalid => "execution-profile-contract-invalid",
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
            Self::HardwareIsaMismatch => "hardware-isa-mismatch",
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
        let mut limit = detail.len().min(240);
        while !detail.is_char_boundary(limit) {
            limit -= 1;
        }
        detail.truncate(limit);
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
    /// Canonical immutable hardware-class ID when one class was identified.
    pub hardware_class_id: Option<String>,
    /// Stable reason code.
    pub reason: MachineClassReason,
}

/// Meaning of one profile's admitted execution capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionCapacitySemantics {
    /// One admitted worker per physical core with one hardware thread per core.
    PhysicalCores,
    /// Admitted logical hardware threads, including explicit SMT siblings.
    LogicalThreads,
    /// Scheduler-managed worker capacity without an affinity or residency claim.
    SchedulerWorkers,
    /// Deliberately bounded diagnostic worker budget on a heterogeneous class.
    DiagnosticWorkerBudget,
}

/// Closed immutable hardware identities admitted by the normative registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HardwareClassId {
    /// Heterogeneous x86 VPS diagnostic workers.
    #[serde(rename = "x86-vps-ovh")]
    X86VpsOvh,
    /// AMD Ryzen Threadripper PRO 5995WX host.
    #[serde(rename = "trj-zen3-5995wx")]
    TrjZen35995wx,
    /// Apple M4 Pro host.
    #[serde(rename = "m4-macos")]
    M4Macos,
    /// Reserved Apple M5 host identity.
    #[serde(rename = "m5-macos")]
    M5Macos,
}

impl HardwareClassId {
    /// Stable registry spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X86VpsOvh => "x86-vps-ovh",
            Self::TrjZen35995wx => "trj-zen3-5995wx",
            Self::M4Macos => "m4-macos",
            Self::M5Macos => "m5-macos",
        }
    }
}

/// Closed execution-profile identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExecutionProfileId {
    /// Runtime-observed single-worker diagnostic lane.
    #[serde(rename = "x86-diagnostic")]
    X86Diagnostic,
    /// One worker per physical Threadripper core.
    #[serde(rename = "physical-64")]
    Physical64,
    /// Two sibling workers per physical Threadripper core.
    #[serde(rename = "smt2-128")]
    Smt2_128,
    /// Scheduler-managed ten-worker Apple M4 lane.
    #[serde(rename = "scheduler-10")]
    Scheduler10,
    /// Reserved scheduler-managed fourteen-worker Apple M5 lane.
    #[serde(rename = "scheduler-14")]
    Scheduler14,
}

impl ExecutionProfileId {
    /// Stable registry spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X86Diagnostic => "x86-diagnostic",
            Self::Physical64 => "physical-64",
            Self::Smt2_128 => "smt2-128",
            Self::Scheduler10 => "scheduler-10",
            Self::Scheduler14 => "scheduler-14",
        }
    }
}

/// One collision-free hardware/profile execution identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct MachineProfileKey {
    hardware_class_id: HardwareClassId,
    execution_profile_id: ExecutionProfileId,
}

impl<'de> Deserialize<'de> for MachineProfileKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawMachineProfileKey {
            hardware_class_id: HardwareClassId,
            execution_profile_id: ExecutionProfileId,
        }

        let raw = RawMachineProfileKey::deserialize(deserializer)?;
        Self::new(raw.hardware_class_id, raw.execution_profile_id).map_err(de::Error::custom)
    }
}

impl MachineProfileKey {
    /// Construct a canonical typed profile key.
    ///
    /// # Errors
    ///
    /// Rejects cross-hardware profile substitutions.
    pub fn new(
        hardware_class_id: HardwareClassId,
        execution_profile_id: ExecutionProfileId,
    ) -> Result<Self, MachineClassError> {
        let key = Self {
            hardware_class_id,
            execution_profile_id,
        };
        if matches!(
            key,
            Self {
                hardware_class_id: HardwareClassId::X86VpsOvh,
                execution_profile_id: ExecutionProfileId::X86Diagnostic,
            } | Self {
                hardware_class_id: HardwareClassId::TrjZen35995wx,
                execution_profile_id: ExecutionProfileId::Physical64 | ExecutionProfileId::Smt2_128,
            } | Self {
                hardware_class_id: HardwareClassId::M4Macos,
                execution_profile_id: ExecutionProfileId::Scheduler10,
            } | Self {
                hardware_class_id: HardwareClassId::M5Macos,
                execution_profile_id: ExecutionProfileId::Scheduler14,
            }
        ) {
            Ok(key)
        } else {
            Err(MachineClassError::new(
                MachineClassReason::ExecutionProfileClassMismatch,
                format!(
                    "execution profile {:?} is not registered for hardware class {:?}",
                    execution_profile_id.as_str(),
                    hardware_class_id.as_str()
                ),
            ))
        }
    }

    /// Immutable hardware identity.
    #[must_use]
    pub const fn hardware_class_id(self) -> HardwareClassId {
        self.hardware_class_id
    }

    /// Immutable execution-profile identity.
    #[must_use]
    pub const fn execution_profile_id(self) -> ExecutionProfileId {
        self.execution_profile_id
    }

    /// Collision-free profile-qualified latest-pointer basename.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassReason::DestinationIdentityMismatch`] when `gate`
    /// is not one of the ten frozen Quill performance gates.
    pub fn latest_basename(self, gate: &str) -> Result<String, MachineClassError> {
        if !matches!(
            gate,
            "QG-1"
                | "QG-2"
                | "QG-3"
                | "QG-4"
                | "QG-5"
                | "QG-6"
                | "QG-7"
                | "QG-8"
                | "QG-9"
                | "QG-10"
        ) {
            return Err(MachineClassError::new(
                MachineClassReason::DestinationIdentityMismatch,
                format!("unsupported performance gate {gate:?}"),
            ));
        }
        Ok(format!(
            "{gate}.{}.{}.latest.json",
            self.hardware_class_id.as_str(),
            self.execution_profile_id.as_str()
        ))
    }
}

/// Release requirement independent of whether the hardware is currently available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultFlipDisposition {
    /// This profile remains mandatory even when its hardware is unavailable.
    RequiredForDefaultFlip,
    /// The profile may produce diagnostics but cannot satisfy a release gate.
    DiagnosticOnly,
}

/// Availability of the execution-profile identity itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineProfileAvailability {
    /// The hardware/profile identity is registered.
    Registered,
    /// No admissible real identity exists yet.
    Unavailable,
}

/// Frozen release policy for one profile and one performance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineProfileGatePolicy {
    default_flip_disposition: DefaultFlipDisposition,
    max_exercised_cell_width: Option<u64>,
}

impl MachineProfileGatePolicy {
    /// Whether the profile is mandatory or diagnostic for this gate.
    #[must_use]
    pub const fn default_flip_disposition(self) -> DefaultFlipDisposition {
        self.default_flip_disposition
    }

    /// Widest canonical cell this profile may exercise for this gate.
    #[must_use]
    pub const fn max_exercised_cell_width(self) -> Option<u64> {
        self.max_exercised_cell_width
    }
}

/// Immutable hardware class plus an independently registered execution profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineExecutionProfile {
    key: MachineProfileKey,
    availability: MachineProfileAvailability,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: Option<u64>,
    required_scheduler_or_affinity_facts: Vec<String>,
    forbidden_claims: Vec<String>,
    gate_policies: BTreeMap<String, MachineProfileGatePolicy>,
    contract_sha256: String,
}

impl MachineExecutionProfile {
    /// Immutable hardware/profile key.
    #[must_use]
    pub const fn key(&self) -> MachineProfileKey {
        self.key
    }

    /// Whether the profile has an admissible real identity.
    #[must_use]
    pub const fn availability(&self) -> MachineProfileAvailability {
        self.availability
    }

    /// Meaning of `execution_capacity`.
    #[must_use]
    pub const fn capacity_semantics(&self) -> ExecutionCapacitySemantics {
        self.capacity_semantics
    }

    /// Maximum admitted execution capacity, absent for runtime-derived or
    /// unavailable profiles.
    #[must_use]
    pub const fn execution_capacity(&self) -> Option<u64> {
        self.execution_capacity
    }

    /// Scheduler or affinity facts that every receipt must prove.
    #[must_use]
    pub fn required_scheduler_or_affinity_facts(&self) -> &[String] {
        &self.required_scheduler_or_affinity_facts
    }

    /// Claims this profile can never derive from its observable facts.
    #[must_use]
    pub fn forbidden_claims(&self) -> &[String] {
        &self.forbidden_claims
    }

    /// Frozen per-gate release and maximum-width policy.
    #[must_use]
    pub fn gate_policy(&self, gate: &str) -> Option<MachineProfileGatePolicy> {
        self.gate_policies.get(gate).copied()
    }

    /// Domain-separated hash of the exact profile contract object.
    #[must_use]
    pub fn contract_sha256(&self) -> &str {
        &self.contract_sha256
    }
}

/// External ratchet context that cannot relabel a receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineClassAdmissionContext {
    /// Canonical gate label such as `QG-2`.
    pub gate: String,
    /// Expected profile identity checked against, never copied into, a receipt.
    pub expected_profile: MachineProfileKey,
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
    profile: MachineProfileKey,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    hardware: Value,
    execution_request: Value,
    execution_start: Value,
    execution_end: Value,
    build: Value,
    durability: Value,
    completion: Value,
    artifact_manifest: Option<RunnerArtifactManifestBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    qg5_durability_witnesses: Option<Qg5DurabilityWitnessSet>,
    #[serde(skip_serializing_if = "Option::is_none")]
    qg5_durability_witness_scope: Option<Qg5DurabilityWitnessScope>,
    derived_sha256: MachineClassDerivedHashes,
}

/// Registry admission facts proven before the measured child is allowed to
/// spawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreSpawnAdmission {
    admission_context: MachineClassAdmissionContext,
    profile: MachineProfileKey,
    hardware_sha256: String,
    execution_identity_sha256: String,
    durability: Value,
}

impl PreSpawnAdmission {
    /// Prove that terminal receipt admission retained the exact pre-spawn
    /// hardware, execution, durability, gate, and class identity.
    pub fn verify_final(&self, identity: &VerifiedRunnerIdentity) -> Result<(), MachineClassError> {
        let expected_pending_qg5_durability = serde_json::to_value(qg5_pending_runner_durability())
            .map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?;
        let durability_matches = if self.admission_context.gate == "QG-5" {
            self.durability == expected_pending_qg5_durability
                && identity
                    .qg5_durability_witnesses
                    .as_ref()
                    .is_some_and(|witnesses| {
                        witnesses.to_json_bytes().is_ok_and(|bytes| {
                            serde_json::from_value::<RunnerDurability>(identity.durability.clone())
                                .is_ok_and(|durability| {
                                    durability == qg5_post_exit_runner_durability(&bytes)
                                })
                        })
                    })
        } else {
            identity.durability == self.durability
        };
        if identity.admission_context != self.admission_context
            || identity.profile != self.profile
            || identity.derived_sha256.hardware != self.hardware_sha256
            || identity.derived_sha256.identity != self.execution_identity_sha256
            || !durability_matches
        {
            return Err(MachineClassError::new(
                MachineClassReason::PrePostIdentityDrift,
                "final runner admission differs from the pre-spawn registry admission",
            ));
        }
        Ok(())
    }
}

/// Strict post-exit manifest naming the exact artifacts emitted by one run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerArtifactManifest {
    schema_version: String,
    gate: String,
    profile: MachineProfileKey,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    applicability_plan: PerfApplicabilityPlanBinding,
    run_id: String,
    run_window: String,
    run_log_sha256: String,
    threshold_artifact_sha256: String,
    prebinding_evidence_artifact_sha256: String,
}

impl RunnerArtifactManifest {
    /// Construct a canonical manifest for a completed invocation.
    ///
    /// The returned object still has to be serialized and named by the sealed
    /// runner receipt before it can be admitted.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the frozen registry cannot be
    /// admitted, the applicability plan does not reconstruct from that
    /// registry, or any required bounded identity is absent or invalid.
    pub fn from_artifacts(
        applicability_plan: &PerfApplicabilityPlan,
        run_id: impl Into<String>,
        run_window: impl Into<String>,
        run_log_bytes: &[u8],
        threshold_artifact_bytes: &[u8],
        evidence_artifact_bytes: &[u8],
    ) -> Result<Self, MachineClassError> {
        let registry = MachineClassRegistry::frozen()?;
        applicability_plan
            .verify_against(&PerfMatrixSpec::complete(), &registry)
            .map_err(|error| {
                MachineClassError::new(
                    MachineClassReason::CompletionUnverified,
                    format!("artifact manifest applicability plan is invalid: {error}"),
                )
            })?;
        let execution_capacity = applicability_plan.execution_capacity.ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "artifact manifest applicability plan has no bounded execution capacity",
            )
        })?;
        let max_exercised_cell_width =
            applicability_plan.max_exercised_cell_width.ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::CompletionUnverified,
                    "artifact manifest applicability plan has no bounded maximum cell width",
                )
            })?;
        Ok(Self {
            schema_version: RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION.to_owned(),
            gate: applicability_plan.binding.gate.label().to_owned(),
            profile: applicability_plan.binding.profile,
            capacity_semantics: applicability_plan.capacity_semantics,
            execution_capacity,
            max_exercised_cell_width,
            applicability_plan: applicability_plan.binding.clone(),
            run_id: run_id.into(),
            run_window: run_window.into(),
            run_log_sha256: sha256_hex(run_log_bytes),
            threshold_artifact_sha256: sha256_hex(threshold_artifact_bytes),
            prebinding_evidence_artifact_sha256: sha256_hex(evidence_artifact_bytes),
        })
    }

    /// Canonical compact JSON bytes used by the runner completion digest.
    ///
    /// # Errors
    ///
    /// Returns a JSON serialization error only if the schema stops being
    /// representable.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Gate sealed by this manifest.
    #[must_use]
    pub fn gate(&self) -> &str {
        &self.gate
    }

    /// Hardware/profile identity sealed by this manifest.
    #[must_use]
    pub const fn profile(&self) -> MachineProfileKey {
        self.profile
    }

    /// Meaning of the execution capacity sealed by this manifest.
    #[must_use]
    pub const fn capacity_semantics(&self) -> ExecutionCapacitySemantics {
        self.capacity_semantics
    }

    /// Exact registry-derived execution capacity sealed by this manifest.
    #[must_use]
    pub const fn execution_capacity(&self) -> u64 {
        self.execution_capacity
    }

    /// Widest canonical cell admitted for this profile and gate.
    #[must_use]
    pub const fn max_exercised_cell_width(&self) -> u64 {
        self.max_exercised_cell_width
    }

    /// Exact matrix, manifest, registry, profile, and plan identity.
    #[must_use]
    pub const fn applicability_plan(&self) -> &PerfApplicabilityPlanBinding {
        &self.applicability_plan
    }

    /// Producer run ID sealed by this manifest.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Producer run window sealed by this manifest.
    #[must_use]
    pub fn run_window(&self) -> &str {
        &self.run_window
    }
}

/// Exact manifest bytes verified against a sealed completion receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerArtifactManifestBinding {
    manifest_json: String,
    manifest_sha256: String,
    manifest: RunnerArtifactManifest,
}

impl RunnerArtifactManifestBinding {
    /// Parsed strict manifest facts.
    #[must_use]
    pub const fn manifest(&self) -> &RunnerArtifactManifest {
        &self.manifest
    }

    /// SHA-256 of the exact strict manifest bytes.
    #[must_use]
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

impl VerifiedRunnerIdentity {
    /// Canonical hardware/profile identity derived from strict receipt facts.
    #[must_use]
    pub const fn profile(&self) -> MachineProfileKey {
        self.profile
    }

    /// Meaning of the bound execution capacity.
    #[must_use]
    pub const fn capacity_semantics(&self) -> ExecutionCapacitySemantics {
        self.capacity_semantics
    }

    /// Exact admitted execution capacity.
    #[must_use]
    pub const fn execution_capacity(&self) -> u64 {
        self.execution_capacity
    }

    /// Widest canonical cell admitted for this profile and gate.
    #[must_use]
    pub const fn max_exercised_cell_width(&self) -> u64 {
        self.max_exercised_cell_width
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

    /// Exact artifact manifest admitted after the measured process exited.
    #[must_use]
    pub const fn artifact_manifest(&self) -> Option<&RunnerArtifactManifestBinding> {
        self.artifact_manifest.as_ref()
    }

    /// Bind exact manifest bytes and prove they name this role's threshold and
    /// pre-binding evidence artifacts.
    ///
    /// # Errors
    ///
    /// Returns a typed completion rejection if the manifest is malformed,
    /// does not hash to the digest sealed by the receipt, names another role,
    /// or names different artifact bytes.
    pub fn bind_artifact_manifest(
        mut self,
        manifest_bytes: &[u8],
        run_log_bytes: &[u8],
        threshold_artifact_bytes: &[u8],
        evidence_artifact_bytes: &[u8],
    ) -> Result<Self, MachineClassError> {
        self.verify()?;
        let binding = parse_artifact_manifest_binding(manifest_bytes)?;
        if let Some(existing) = &self.artifact_manifest
            && existing != &binding
        {
            return Err(MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "runner identity already carries a different artifact manifest",
            ));
        }
        self.artifact_manifest = Some(binding);
        self.verify_artifact_inputs(
            run_log_bytes,
            threshold_artifact_bytes,
            evidence_artifact_bytes,
        )?;
        self.verify()?;
        Ok(self)
    }

    /// Recheck exact role and artifact bytes against the bound manifest.
    ///
    /// # Errors
    ///
    /// Returns a typed completion rejection when no manifest is bound or any
    /// exact digest differs.
    pub fn verify_artifact_inputs(
        &self,
        run_log_bytes: &[u8],
        threshold_artifact_bytes: &[u8],
        evidence_artifact_bytes: &[u8],
    ) -> Result<(), MachineClassError> {
        let binding = self.artifact_manifest.as_ref().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "runner receipt has no exact artifact-manifest binding",
            )
        })?;
        validate_artifact_manifest_binding(self, binding)?;
        let manifest = &binding.manifest;
        if manifest.run_log_sha256 != sha256_hex(run_log_bytes)
            || manifest.threshold_artifact_sha256 != sha256_hex(threshold_artifact_bytes)
            || manifest.prebinding_evidence_artifact_sha256 != sha256_hex(evidence_artifact_bytes)
        {
            return Err(MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "artifact manifest does not name the supplied run-log/threshold/evidence bytes",
            ));
        }
        Ok(())
    }

    /// Recheck the exact post-exit run log against the bound manifest.
    ///
    /// # Errors
    ///
    /// Returns a typed completion rejection if the exact bytes differ.
    pub fn verify_run_log(&self, run_log_bytes: &[u8]) -> Result<(), MachineClassError> {
        let binding = self.artifact_manifest.as_ref().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "runner receipt has no exact artifact-manifest binding",
            )
        })?;
        validate_artifact_manifest_binding(self, binding)?;
        if binding.manifest.run_log_sha256 != sha256_hex(run_log_bytes) {
            return Err(MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "artifact manifest does not name the supplied run-log bytes",
            ));
        }
        Ok(())
    }

    /// Recheck only the exact threshold artifact for a ratchet role.
    ///
    /// # Errors
    ///
    /// Returns a typed completion rejection if the role or exact bytes differ.
    pub fn verify_threshold_artifact(
        &self,
        threshold_artifact_bytes: &[u8],
    ) -> Result<(), MachineClassError> {
        let binding = self.artifact_manifest.as_ref().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "runner receipt has no exact artifact-manifest binding",
            )
        })?;
        validate_artifact_manifest_binding(self, binding)?;
        if binding.manifest.threshold_artifact_sha256 != sha256_hex(threshold_artifact_bytes) {
            return Err(MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "artifact manifest does not name the supplied threshold bytes",
            ));
        }
        Ok(())
    }

    /// Recheck only the exact pre-binding evidence artifact.
    ///
    /// # Errors
    ///
    /// Returns a typed completion rejection if the exact bytes differ.
    pub fn verify_evidence_artifact(
        &self,
        evidence_artifact_bytes: &[u8],
    ) -> Result<(), MachineClassError> {
        let binding = self.artifact_manifest.as_ref().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "runner receipt has no exact artifact-manifest binding",
            )
        })?;
        validate_artifact_manifest_binding(self, binding)?;
        if binding.manifest.prebinding_evidence_artifact_sha256
            != sha256_hex(evidence_artifact_bytes)
        {
            return Err(MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                "artifact manifest does not name the supplied pre-binding evidence bytes",
            ));
        }
        Ok(())
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
        let recomputed = match (
            &self.qg5_durability_witnesses,
            &self.qg5_durability_witness_scope,
        ) {
            (Some(witnesses), Some(scope)) => {
                let bytes = witnesses.to_json_bytes()?;
                registry.admit_qg5_post_exit_with_scope(
                    self.receipt_json.as_bytes(),
                    &self.admission_context,
                    &bytes,
                    scope,
                )?
            }
            (None, None) => {
                registry.admit(self.receipt_json.as_bytes(), &self.admission_context)?
            }
            _ => {
                return Err(MachineClassError::new(
                    MachineClassReason::DerivedHashMismatch,
                    "stored QG-5 witness and its independently retained admission scope disagree",
                ));
            }
        };
        let mut receipt_only = self.clone();
        receipt_only.artifact_manifest = None;
        if recomputed != receipt_only {
            return Err(MachineClassError::new(
                MachineClassReason::DerivedHashMismatch,
                "stored runner binding does not equal exact re-admission",
            ));
        }
        if let Some(binding) = &self.artifact_manifest {
            validate_artifact_manifest_binding(self, binding)?;
        }
        Ok(())
    }

    /// Whether two receipts name the same registry, hardware/profile,
    /// capacity, durability, and stable execution identity. Exact receipt
    /// digests may differ across independent runs.
    #[must_use]
    pub fn same_execution_identity(&self, other: &Self) -> bool {
        self.canonicalization == other.canonicalization
            && self.profile == other.profile
            && self.capacity_semantics == other.capacity_semantics
            && self.execution_capacity == other.execution_capacity
            && self.max_exercised_cell_width == other.max_exercised_cell_width
            && self.derived_sha256.hardware == other.derived_sha256.hardware
            && self.derived_sha256.identity == other.derived_sha256.identity
            && self.durability == other.durability
    }
}

/// Parent-retained QG-5 run scope used to recheck persisted witnesses.
///
/// The witness bytes are sealed by the benchmark child, but the run ID and
/// canonical selected-cell census originate outside that child at the parent
/// admission boundary. Retaining them separately prevents a resealed witness
/// from another run or a different cell census from becoming self-authenticating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qg5DurabilityWitnessScope {
    run_id: String,
    selected_cell_ids: Vec<String>,
    expected_census_sha256: String,
}

impl Qg5DurabilityWitnessScope {
    fn new(
        run_id: &str,
        selected_cell_ids: &[String],
        expected_census: &Qg5ExpectedDurabilityCensus,
    ) -> Result<Self, MachineClassError> {
        let scope = Self {
            run_id: run_id.to_owned(),
            selected_cell_ids: selected_cell_ids.to_vec(),
            expected_census_sha256: expected_census.binding_sha256()?,
        };
        if !valid_qg5_identity(&scope.run_id)
            || scope.selected_cell_ids.is_empty()
            || scope.selected_cell_ids.len()
                != scope
                    .selected_cell_ids
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
            || scope
                .selected_cell_ids
                .iter()
                .any(|cell| !valid_qg5_cell(cell))
            || !is_sha256(&scope.expected_census_sha256)
            || expected_census.run_id != scope.run_id
            || expected_census.cell_ids() != scope.selected_cell_ids.iter().cloned().collect()
        {
            return Err(qg5_witness_error(
                "QG-5 witness admission scope has an invalid run ID, selected cell census, or evidence binding",
            ));
        }
        Ok(scope)
    }
}

fn parse_artifact_manifest_binding(
    manifest_bytes: &[u8],
) -> Result<RunnerArtifactManifestBinding, MachineClassError> {
    let manifest_value = parse_strict_json(manifest_bytes)?;
    let manifest =
        serde_json::from_value::<RunnerArtifactManifest>(manifest_value).map_err(|error| {
            let detail = error.to_string();
            let reason = if detail.contains("unknown field") {
                MachineClassReason::UnknownField
            } else if detail.contains("missing field") {
                MachineClassReason::MissingField
            } else {
                MachineClassReason::CompletionUnverified
            };
            MachineClassError::new(reason, detail)
        })?;
    let canonical = manifest.to_json_bytes().map_err(|error| {
        MachineClassError::new(MachineClassReason::CompletionUnverified, error.to_string())
    })?;
    if canonical != manifest_bytes {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "artifact manifest is not exact canonical compact JSON",
        ));
    }
    let manifest_json = std::str::from_utf8(manifest_bytes)
        .map_err(|error| {
            MachineClassError::new(MachineClassReason::CompletionUnverified, error.to_string())
        })?
        .to_owned();
    Ok(RunnerArtifactManifestBinding {
        manifest_json,
        manifest_sha256: sha256_hex(manifest_bytes),
        manifest,
    })
}

fn validate_artifact_manifest_binding(
    identity: &VerifiedRunnerIdentity,
    binding: &RunnerArtifactManifestBinding,
) -> Result<(), MachineClassError> {
    let reparsed = parse_artifact_manifest_binding(binding.manifest_json.as_bytes())?;
    if reparsed != *binding {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "stored artifact manifest differs from strict reparsing",
        ));
    }
    let manifest = &binding.manifest;
    if manifest.schema_version != RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION
        || manifest.gate.trim().is_empty()
        || manifest.execution_capacity == 0
        || manifest.max_exercised_cell_width == 0
        || manifest.max_exercised_cell_width > manifest.execution_capacity
        || manifest.run_id.trim().is_empty()
        || manifest.run_window.trim().is_empty()
        || !is_sha256(&manifest.run_log_sha256)
        || !is_sha256(&manifest.threshold_artifact_sha256)
        || !is_sha256(&manifest.prebinding_evidence_artifact_sha256)
    {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "artifact manifest schema, run identity, or digests are invalid",
        ));
    }
    let registry = MachineClassRegistry::frozen()?;
    let expected_plan = PerfMatrixSpec::complete()
        .applicability_plan(
            &registry,
            manifest.applicability_plan.profile,
            manifest.applicability_plan.gate,
        )
        .map_err(|error| {
            MachineClassError::new(
                MachineClassReason::CompletionUnverified,
                format!("artifact manifest applicability plan does not reconstruct: {error}"),
            )
        })?;
    if manifest.applicability_plan != expected_plan.binding
        || manifest.gate != expected_plan.binding.gate.label()
        || manifest.profile != expected_plan.binding.profile
        || manifest.capacity_semantics != expected_plan.capacity_semantics
        || expected_plan.execution_capacity != Some(manifest.execution_capacity)
        || expected_plan.max_exercised_cell_width != Some(manifest.max_exercised_cell_width)
        || manifest.applicability_plan.registry_schema_version
            != identity.canonicalization.registry_schema_version
        || manifest.applicability_plan.registry_sha256 != identity.canonicalization.registry_sha256
    {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "artifact manifest applicability identity does not equal the canonical plan",
        ));
    }
    let completion = identity.completion.as_object().ok_or_else(|| {
        MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "verified runner completion is not an object",
        )
    })?;
    let completion_string = |field: &str| {
        completion
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::CompletionUnverified,
                    format!("verified runner completion field {field:?} is not a string"),
                )
            })
    };
    if binding.manifest_sha256 != completion_string("artifact_manifest_sha256")?
        || manifest.run_log_sha256 != completion_string("run_log_sha256")?
        || manifest.gate != identity.admission_context.gate
        || manifest.profile != identity.profile
        || manifest.capacity_semantics != identity.capacity_semantics
        || manifest.execution_capacity != identity.execution_capacity
        || manifest.max_exercised_cell_width != identity.max_exercised_cell_width
    {
        return Err(MachineClassError::new(
            MachineClassReason::CompletionUnverified,
            "artifact manifest does not match the receipt completion or admission gate",
        ));
    }
    if let Some(scope) = &identity.qg5_durability_witness_scope
        && manifest.run_id != scope.run_id
    {
        return Err(qg5_witness_error(
            "QG-5 artifact manifest run ID differs from the retained witness admission scope",
        ));
    }
    Ok(())
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
struct RegistryHardwareClassRule {
    hardware_class_id: HardwareClassId,
    family: String,
    availability: MachineProfileAvailability,
    admission_reason: MachineClassReason,
    hardware_predicates: Map<String, Value>,
    source_fingerprints: Vec<SourceFingerprint>,
    #[serde(default, rename = "notes")]
    _notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryExecutionProfileRule {
    key: MachineProfileKey,
    availability: MachineProfileAvailability,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: RequiredNullableU64,
    required_scheduler_or_affinity_facts: Vec<String>,
    forbidden_claims: Vec<String>,
    gate_policies: BTreeMap<String, RegistryProfileGatePolicy>,
    #[serde(default, rename = "notes")]
    _notes: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
struct RequiredNullableU64(Option<u64>);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryProfileGatePolicy {
    default_flip_disposition: DefaultFlipDisposition,
    max_exercised_cell_width: RequiredNullableU64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryArtifactManifestContract {
    schema_version: String,
    unknown_field_policy: String,
    duplicate_key_policy: String,
    canonical_encoding: String,
    required_fields: Vec<String>,
    binding_law: String,
    history_law: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerHardware {
    pub(crate) os: String,
    pub(crate) arch: String,
    pub(crate) cpu_vendor: String,
    pub(crate) cpu_family: Option<u64>,
    pub(crate) cpu_model: Option<u64>,
    pub(crate) cpu_stepping: Option<u64>,
    pub(crate) cpu_model_name: String,
    pub(crate) physical_cores: u64,
    pub(crate) logical_cpus: u64,
    pub(crate) numa_nodes: u64,
    pub(crate) memory_bytes: u64,
    pub(crate) page_size_bytes: u64,
    pub(crate) performance_cores: Option<u64>,
    pub(crate) efficiency_cores: Option<u64>,
    pub(crate) runtime_detected_isa: Vec<String>,
    pub(crate) topology_sha256: String,
    pub(crate) fingerprint_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerExecutionRequest {
    pub(crate) capacity_semantics: ExecutionCapacitySemantics,
    pub(crate) execution_capacity: u64,
    pub(crate) max_exercised_cell_width: u64,
    pub(crate) requested_logical_cpu_ids: Vec<u64>,
    pub(crate) requested_physical_core_width: Option<u64>,
    pub(crate) requested_worker_pool_width: u64,
    pub(crate) requested_qos: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerExecutionSnapshot {
    pub(crate) observed_logical_cpu_ids: Vec<u64>,
    pub(crate) effective_physical_core_ids: Vec<String>,
    pub(crate) cpu_assignment_observability: String,
    pub(crate) effective_cpuset_sha256: String,
    pub(crate) threads_per_core: u64,
    pub(crate) smt_state: String,
    pub(crate) numa_node_ids: Vec<u64>,
    pub(crate) numa_policy: String,
    pub(crate) governor: String,
    pub(crate) thermal_pressure: bool,
    pub(crate) exclusive_lease: bool,
    pub(crate) exclusive_lease_id: String,
    pub(crate) local_execution: bool,
    pub(crate) observed_hardware_fingerprint_sha256: String,
    pub(crate) snapshot_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerExecution {
    pub(crate) request: RunnerExecutionRequest,
    pub(crate) start: RunnerExecutionSnapshot,
    pub(crate) end: RunnerExecutionSnapshot,
    pub(crate) identity_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerProducer {
    pub(crate) contract_version: String,
    pub(crate) source_git_revision: String,
    pub(crate) source_git_dirty: bool,
    pub(crate) cargo_lock_sha256: String,
    pub(crate) executable_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerBuild {
    pub(crate) git_revision: String,
    pub(crate) git_dirty: bool,
    pub(crate) worktree_state_sha256: Option<String>,
    pub(crate) cargo_lock_sha256: String,
    pub(crate) executable_sha256: String,
    pub(crate) command_sha256: String,
    pub(crate) environment_sha256: String,
    pub(crate) producer: RunnerProducer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerDurability {
    pub(crate) adjacent: bool,
    pub(crate) control_treatment: String,
    pub(crate) candidate_treatment: String,
    pub(crate) symmetric: bool,
}

const QG5_PENDING_DURABILITY_TREATMENT: &str = "qg5-post-exit-witness-required";
const QG5_POST_EXIT_DURABILITY_PREFIX: &str = "qg5-post-exit-witness-set-v2:";
const QG5_RAW_SAMPLE_HASH_DOMAIN: &[u8] = b"frankensearch.quill.qg5-raw-sample.v2\0";
const QG5_CENSUS_HASH_DOMAIN: &[u8] = b"frankensearch.quill.qg5-sample-census.v2\0";
const QG5_MAX_SAMPLES_PER_CELL: usize = 400;
const QG5_MAX_PROBE_DOCUMENT_ID_BYTES: usize = 128;
/// Exact canonical child artifact required before QG-5 may bind promotion
/// evidence. The child writes this file below `QUILL_PERF_OUTPUT_DIR`.
pub const QG5_DURABILITY_WITNESS_FILE_NAME: &str = "QG-5.durability-witnesses.json";
/// Strict schema for the sealed QG-5 post-exit measured-sample census.
pub const QG5_DURABILITY_WITNESS_SCHEMA_VERSION: &str = "frankensearch.qg5-durability-witnesses.v2";

/// Engine named by a QG-5 durability witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg5DurabilityEngine {
    /// The Quill compaction arm.
    Quill,
    /// The Tantivy force-merge arm.
    Tantivy,
}

/// Retained paired stream containing one measured QG-5 sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg5StreamRole {
    /// Tantivy control versus Quill treatment.
    Effect,
    /// Tantivy control versus Tantivy treatment.
    OracleNull,
}

/// Typed facts observed after deletes are durably published and before the
/// maintenance timer starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5DeletePublicationObservation {
    /// Documents present before deletion.
    pub source_document_count: u64,
    /// Exact number of requested deletions.
    pub requested_delete_count: u64,
    /// Authoritative live count after publication.
    pub published_live_document_count: u64,
    /// Authoritative segment count presented to maintenance.
    pub published_segment_count: u64,
    /// Exact deleted document used by the visibility probe.
    pub deleted_probe_document_id: String,
    /// Matches for the deleted probe after publication; policy requires zero.
    pub deleted_probe_match_count: u64,
    /// Exact retained document used by the visibility probe.
    pub live_probe_document_id: String,
    /// Matches for the live probe after publication; policy requires one.
    pub live_probe_match_count: u64,
}

/// Engine-specific timed maintenance facts. The elapsed interval is the exact
/// duration converted into the parent [`PerfRawSample::observed_value`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Qg5TimedMaintenanceObservation {
    /// Quill compacted the published segment set.
    QuillCompaction {
        /// Exact inner maintenance interval.
        elapsed_ns: u64,
        /// MANIFEST generation inspected by the pass.
        generation_before: u64,
        /// Successor generation published by the changed pass.
        generation_after: u64,
        /// Immutable segments examined by the density policy.
        examined_segments: u64,
        /// Eligible segments rewritten or removed.
        compacted_segments: u64,
        /// Fully deleted segments removed without replacement files.
        removed_segments: u64,
        /// Physical rows folded out of rewritten segments.
        dropped_documents: u64,
        /// Source bytes read for eligible segments.
        input_bytes: u64,
        /// Replacement bytes emitted for surviving compacted segments.
        output_bytes: u64,
        /// Segments visible immediately before compaction.
        input_segment_count: u64,
        /// Segments visible immediately after compaction.
        output_segment_count: u64,
    },
    /// Tantivy force-merged the published segment set.
    TantivyForceMerge {
        /// Exact inner maintenance interval.
        elapsed_ns: u64,
        /// Segments visible immediately before force-merge.
        input_segment_count: u64,
        /// Segments visible immediately after force-merge.
        output_segment_count: u64,
    },
}

impl Qg5TimedMaintenanceObservation {
    fn engine(&self) -> Qg5DurabilityEngine {
        match self {
            Self::QuillCompaction { .. } => Qg5DurabilityEngine::Quill,
            Self::TantivyForceMerge { .. } => Qg5DurabilityEngine::Tantivy,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        match self {
            Self::QuillCompaction { elapsed_ns, .. }
            | Self::TantivyForceMerge { elapsed_ns, .. } => *elapsed_ns,
        }
    }
}

/// Typed facts observed from a fresh post-maintenance reopen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5ReopenValidationObservation {
    /// Authoritative live count from the reopened index.
    pub reopened_live_document_count: u64,
    /// Authoritative segment count from the reopened index.
    pub reopened_segment_count: u64,
    /// The same deleted probe used before maintenance.
    pub deleted_probe_document_id: String,
    /// Reopened-index matches for the deleted probe; policy requires zero.
    pub deleted_probe_match_count: u64,
    /// The same live probe used before maintenance.
    pub live_probe_document_id: String,
    /// Reopened-index matches for the live probe; policy requires one.
    pub live_probe_match_count: u64,
}

/// Policy-checkable three-stage observation for one measured QG-5 operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5DurabilityObservation {
    /// Durable delete-publication facts captured before timing.
    pub delete_publication: Qg5DeletePublicationObservation,
    /// Exact engine-specific maintenance interval and topology transition.
    pub timed_maintenance: Qg5TimedMaintenanceObservation,
    /// Fresh-reopen count, topology, and visibility facts.
    pub reopen_validation: Qg5ReopenValidationObservation,
}

impl Qg5DurabilityObservation {
    /// Construct and validate one typed three-stage observation.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the publication, timed maintenance,
    /// and reopened-index facts do not form a valid engine-specific QG-5
    /// observation.
    pub fn new(
        delete_publication: Qg5DeletePublicationObservation,
        timed_maintenance: Qg5TimedMaintenanceObservation,
        reopen_validation: Qg5ReopenValidationObservation,
    ) -> Result<Self, MachineClassError> {
        let observation = Self {
            delete_publication,
            timed_maintenance,
            reopen_validation,
        };
        observation.validate(observation.timed_maintenance.engine())?;
        Ok(observation)
    }

    fn validate(&self, engine: Qg5DurabilityEngine) -> Result<(), MachineClassError> {
        let delete = &self.delete_publication;
        let reopen = &self.reopen_validation;
        let valid_probe = |value: &str| {
            !value.is_empty()
                && value.len() <= QG5_MAX_PROBE_DOCUMENT_ID_BYTES
                && value.trim() == value
                && value.bytes().all(|byte| byte.is_ascii_graphic())
        };
        if self.timed_maintenance.engine() != engine
            || self.timed_maintenance.elapsed_ns() == 0
            || delete.source_document_count < 2
            || delete.requested_delete_count == 0
            || delete.requested_delete_count >= delete.source_document_count
            || delete.published_live_document_count
                != delete.source_document_count - delete.requested_delete_count
            || delete.published_segment_count < 2
            || reopen.reopened_live_document_count != delete.published_live_document_count
            || !valid_probe(&delete.deleted_probe_document_id)
            || !valid_probe(&delete.live_probe_document_id)
            || delete.deleted_probe_document_id == delete.live_probe_document_id
            || delete.deleted_probe_match_count != 0
            || delete.live_probe_match_count != 1
            || reopen.deleted_probe_document_id != delete.deleted_probe_document_id
            || reopen.live_probe_document_id != delete.live_probe_document_id
            || reopen.deleted_probe_match_count != 0
            || reopen.live_probe_match_count != 1
        {
            return Err(qg5_witness_error(
                "QG-5 observation does not prove published deletes, engine-specific maintenance, and a consistent fresh reopen",
            ));
        }
        let output_segment_count = match &self.timed_maintenance {
            Qg5TimedMaintenanceObservation::QuillCompaction {
                generation_before,
                generation_after,
                examined_segments,
                compacted_segments,
                removed_segments,
                dropped_documents,
                input_bytes,
                output_bytes,
                input_segment_count,
                output_segment_count,
                ..
            } => {
                let expected_output = input_segment_count.checked_sub(*removed_segments);
                let surviving_compactions = compacted_segments.saturating_sub(*removed_segments);
                if *input_segment_count != delete.published_segment_count
                    || examined_segments != input_segment_count
                    || *compacted_segments == 0
                    || compacted_segments != examined_segments
                    || removed_segments > compacted_segments
                    || generation_after <= generation_before
                    || *dropped_documents == 0
                    || *dropped_documents > delete.requested_delete_count
                    || *input_bytes == 0
                    || (surviving_compactions == 0) != (*output_bytes == 0)
                    || expected_output != Some(*output_segment_count)
                    || *output_segment_count == 0
                {
                    return Err(qg5_witness_error(
                        "QG-5 Quill observation is not a changed, internally consistent CompactionReport",
                    ));
                }
                *output_segment_count
            }
            Qg5TimedMaintenanceObservation::TantivyForceMerge {
                input_segment_count,
                output_segment_count,
                ..
            } => {
                if *input_segment_count != delete.published_segment_count
                    || *output_segment_count == 0
                    || output_segment_count >= input_segment_count
                {
                    return Err(qg5_witness_error(
                        "QG-5 Tantivy force-merge did not reduce the published segment topology",
                    ));
                }
                *output_segment_count
            }
        };
        if reopen.reopened_segment_count != output_segment_count {
            return Err(qg5_witness_error(
                "QG-5 fresh reopen does not preserve the post-maintenance topology",
            ));
        }
        Ok(())
    }

    fn validate_against_expected(
        &self,
        expected: &Qg5ExpectedSample,
    ) -> Result<(), MachineClassError> {
        self.validate(expected.binding.engine)?;
        let elapsed_ns = self.timed_maintenance.elapsed_ns();
        let observed_ms = std::time::Duration::from_nanos(elapsed_ns).as_secs_f64() * 1_000.0;
        if elapsed_ns > expected.sample_interval_ns
            || observed_ms.to_bits() != expected.observed_value_bits
        {
            return Err(qg5_witness_error(
                "QG-5 maintenance interval does not equal the exact retained PerfRawSample value",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct Qg5SampleBinding {
    stream: Qg5StreamRole,
    block_id: u64,
    sample_id: u64,
    engine: Qg5DurabilityEngine,
    raw_sample_sha256: String,
}

impl Qg5SampleBinding {
    const fn key(&self) -> (Qg5StreamRole, u64, u64) {
        (self.stream, self.block_id, self.sample_id)
    }
}

/// One measured-sample durability witness emitted after the sample timer has
/// closed. The enclosing witness-set seal authenticates the complete census.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5SampleDurabilityWitness {
    stream: Qg5StreamRole,
    block_id: u64,
    sample_id: u64,
    engine: Qg5DurabilityEngine,
    raw_sample_sha256: String,
    observation: Qg5DurabilityObservation,
}

impl Qg5SampleDurabilityWitness {
    /// Bind one completed typed engine observation to the exact raw sample.
    ///
    /// Callers must invoke this only after capturing the sample's terminal
    /// timestamp, so serialization and hashing never enter the timed interval.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the raw sample is not a valid
    /// measured row, the engine disagrees with its stream and arm, the typed
    /// observation is invalid, or its elapsed latency differs from the sample.
    pub fn seal(
        stream: Qg5StreamRole,
        engine: Qg5DurabilityEngine,
        raw_sample: &PerfRawSample,
        observation: Qg5DurabilityObservation,
    ) -> Result<Self, MachineClassError> {
        let expected = qg5_expected_sample(stream, raw_sample)?;
        if engine != expected.binding.engine {
            return Err(qg5_witness_error(
                "QG-5 sample engine does not match its stream role and measured arm",
            ));
        }
        observation.validate_against_expected(&expected)?;
        let witness = Self {
            stream,
            block_id: raw_sample.block_id,
            sample_id: raw_sample.sample_id,
            engine,
            raw_sample_sha256: expected.binding.raw_sample_sha256,
            observation,
        };
        witness.verify()?;
        Ok(witness)
    }

    fn binding(&self) -> Qg5SampleBinding {
        Qg5SampleBinding {
            stream: self.stream,
            block_id: self.block_id,
            sample_id: self.sample_id,
            engine: self.engine,
            raw_sample_sha256: self.raw_sample_sha256.clone(),
        }
    }

    fn key(&self) -> (Qg5StreamRole, u64, u64) {
        (self.stream, self.block_id, self.sample_id)
    }

    fn verify(&self) -> Result<(), MachineClassError> {
        if !is_sha256(&self.raw_sample_sha256) {
            return Err(qg5_witness_error(
                "QG-5 measured-sample witness has an invalid raw-sample binding",
            ));
        }
        self.observation.validate(self.engine)
    }
}

/// Canonical measured-sample durability census for one QG-5 cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5CellDurabilityWitness {
    cell: String,
    samples: Vec<Qg5SampleDurabilityWitness>,
}

impl Qg5CellDurabilityWitness {
    /// Sort and validate the complete measured-sample census for one cell.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the cell identity is invalid, the
    /// census is empty or oversized, an exact sample key is duplicated, or a
    /// retained sample witness is invalid.
    pub fn new(
        cell: impl Into<String>,
        mut samples: Vec<Qg5SampleDurabilityWitness>,
    ) -> Result<Self, MachineClassError> {
        samples.sort_by_key(Qg5SampleDurabilityWitness::key);
        let witness = Self {
            cell: cell.into(),
            samples,
        };
        witness.verify()?;
        Ok(witness)
    }

    fn verify(&self) -> Result<(), MachineClassError> {
        if !valid_qg5_cell(&self.cell)
            || self.samples.is_empty()
            || self.samples.len() > QG5_MAX_SAMPLES_PER_CELL
            || self
                .samples
                .windows(2)
                .any(|pair| pair[0].key() >= pair[1].key())
        {
            return Err(qg5_witness_error(
                "QG-5 cell sample census is empty, oversized, unsorted, or contains duplicate keys",
            ));
        }
        for sample in &self.samples {
            sample.verify()?;
        }
        Ok(())
    }

    fn bindings(&self) -> Vec<Qg5SampleBinding> {
        self.samples
            .iter()
            .map(Qg5SampleDurabilityWitness::binding)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qg5ExpectedSample {
    binding: Qg5SampleBinding,
    observed_value_bits: u64,
    sample_interval_ns: u64,
}

/// Parent-derived projection of the exact retained QG-5 raw samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg5ExpectedDurabilityCensus {
    run_id: String,
    cells: BTreeMap<String, Vec<Qg5ExpectedSample>>,
}

impl Qg5ExpectedDurabilityCensus {
    /// Derive the independent expected census from already verified canonical
    /// evidence. No child witness fields participate in this construction.
    pub(super) fn from_evidence(
        run_id: &str,
        selected_cell_ids: &[String],
        evidence: &PerfEvidenceArtifact,
    ) -> Result<Self, MachineClassError> {
        let selected = selected_cell_ids.iter().cloned().collect::<BTreeSet<_>>();
        if evidence.gate != PerfGate::Qg5
            || evidence.provenance.run_id != run_id
            || !valid_qg5_identity(run_id)
            || selected.is_empty()
            || selected.len() != selected_cell_ids.len()
            || selected.iter().any(|cell| !valid_qg5_cell(cell))
        {
            return Err(qg5_witness_error(
                "QG-5 expected census does not name the verified run and selected cells",
            ));
        }

        let mut cells = BTreeMap::new();
        for cell in &evidence.cells {
            if !selected.contains(&cell.cell_id) || cells.contains_key(&cell.cell_id) {
                return Err(qg5_witness_error(
                    "QG-5 evidence cell census contains an extra or duplicate cell",
                ));
            }
            let EvidenceCellBody::Paired { paired, .. } = &cell.body else {
                return Err(qg5_witness_error(
                    "QG-5 evidence cell does not retain paired effect and oracle-null samples",
                ));
            };
            cells.insert(
                cell.cell_id.clone(),
                qg5_expected_cell_samples(run_id, &paired.effect_samples, &paired.null_samples)?,
            );
        }
        let census = Self {
            run_id: run_id.to_owned(),
            cells,
        };
        if census.cell_ids() != selected {
            return Err(qg5_witness_error(
                "QG-5 evidence is missing one or more selected cells",
            ));
        }
        let _ = census.binding_sha256()?;
        Ok(census)
    }

    fn cell_ids(&self) -> BTreeSet<String> {
        self.cells.keys().cloned().collect()
    }

    fn binding_cells(&self) -> BTreeMap<String, Vec<Qg5SampleBinding>> {
        self.cells
            .iter()
            .map(|(cell, samples)| {
                (
                    cell.clone(),
                    samples
                        .iter()
                        .map(|sample| sample.binding.clone())
                        .collect(),
                )
            })
            .collect()
    }

    fn binding_sha256(&self) -> Result<String, MachineClassError> {
        qg5_census_binding_sha256(&self.run_id, &self.binding_cells())
    }
}

fn qg5_expected_engine(stream: Qg5StreamRole, arm: PerfSampleArm) -> Qg5DurabilityEngine {
    match (stream, arm) {
        (Qg5StreamRole::Effect, PerfSampleArm::Treatment) => Qg5DurabilityEngine::Quill,
        (Qg5StreamRole::Effect, PerfSampleArm::Control) | (Qg5StreamRole::OracleNull, _) => {
            Qg5DurabilityEngine::Tantivy
        }
    }
}

fn qg5_raw_sample_sha256(sample: &PerfRawSample) -> Result<String, MachineClassError> {
    let encoded = serde_json::to_vec(sample).map_err(|error| {
        qg5_witness_error(format!("QG-5 raw sample serialization failed: {error}"))
    })?;
    qg5_domain_sha256(QG5_RAW_SAMPLE_HASH_DOMAIN, &encoded)
}

fn qg5_expected_sample(
    stream: Qg5StreamRole,
    sample: &PerfRawSample,
) -> Result<Qg5ExpectedSample, MachineClassError> {
    let observed_value = sample.observed_value.ok_or_else(|| {
        qg5_witness_error("QG-5 raw sample is missing its measured maintenance value")
    })?;
    let sample_interval_ns = sample
        .ended_ns
        .checked_sub(sample.started_ns)
        .ok_or_else(|| qg5_witness_error("QG-5 raw sample has a reversed monotonic interval"))?;
    if sample.phase != PerfSamplePhase::Measurement
        || sample_interval_ns == 0
        || !observed_value.is_finite()
        || observed_value <= 0.0
    {
        return Err(qg5_witness_error(
            "QG-5 census may contain only finite positive decision samples",
        ));
    }
    Ok(Qg5ExpectedSample {
        binding: Qg5SampleBinding {
            stream,
            block_id: sample.block_id,
            sample_id: sample.sample_id,
            engine: qg5_expected_engine(stream, sample.arm),
            raw_sample_sha256: qg5_raw_sample_sha256(sample)?,
        },
        observed_value_bits: observed_value.to_bits(),
        sample_interval_ns,
    })
}

fn qg5_expected_cell_samples(
    run_id: &str,
    effect_samples: &[PerfRawSample],
    null_samples: &[PerfRawSample],
) -> Result<Vec<Qg5ExpectedSample>, MachineClassError> {
    if effect_samples.is_empty()
        || effect_samples.len() % 2 != 0
        || effect_samples.len() != null_samples.len()
        || effect_samples.len() + null_samples.len() > QG5_MAX_SAMPLES_PER_CELL
    {
        return Err(qg5_witness_error(
            "QG-5 evidence must retain equally sized non-empty T/Q and T/T paired streams",
        ));
    }

    let stream_blocks = |samples: &[PerfRawSample]| {
        let mut blocks = BTreeMap::<u64, u8>::new();
        for sample in samples {
            let arm_bit = match sample.arm {
                PerfSampleArm::Control => 0b01,
                PerfSampleArm::Treatment => 0b10,
            };
            let observed_arms = blocks.entry(sample.block_id).or_default();
            if *observed_arms & arm_bit != 0 {
                return Err(qg5_witness_error(
                    "QG-5 evidence stream contains a duplicate arm in one measured block",
                ));
            }
            *observed_arms |= arm_bit;
        }
        if blocks.values().any(|observed_arms| *observed_arms != 0b11) {
            return Err(qg5_witness_error(
                "QG-5 evidence stream does not contain one control and one treatment in every measured block",
            ));
        }
        Ok(blocks.keys().copied().collect::<BTreeSet<_>>())
    };
    let effect_blocks = stream_blocks(effect_samples)?;
    let null_blocks = stream_blocks(null_samples)?;
    if effect_blocks != null_blocks || effect_blocks.len() != effect_samples.len() / 2 {
        return Err(qg5_witness_error(
            "QG-5 effect and oracle-null streams do not retain the same exact round census",
        ));
    }

    let mut sample_ids = BTreeSet::new();
    let mut sample_keys = BTreeSet::new();
    let mut expected = Vec::with_capacity(effect_samples.len() + null_samples.len());
    for (stream, samples) in [
        (Qg5StreamRole::Effect, effect_samples),
        (Qg5StreamRole::OracleNull, null_samples),
    ] {
        for sample in samples {
            if sample.provenance.run_id != run_id
                || !sample_ids.insert(sample.sample_id)
                || !sample_keys.insert((stream, sample.block_id, sample.sample_id))
            {
                return Err(qg5_witness_error(
                    "QG-5 raw samples contain a foreign run, duplicate sample ID, or duplicate exact key",
                ));
            }
            expected.push(qg5_expected_sample(stream, sample)?);
        }
    }
    expected.sort_by_key(|sample| sample.binding.key());
    if expected
        .windows(2)
        .any(|pair| pair[0].binding.key() >= pair[1].binding.key())
    {
        return Err(qg5_witness_error(
            "QG-5 expected sample census contains a duplicate key",
        ));
    }
    Ok(expected)
}

/// Sealed census of every measured QG-5 sample emitted by one benchmark child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg5DurabilityWitnessSet {
    schema_version: String,
    gate: String,
    run_id: String,
    cells: BTreeMap<String, Qg5CellDurabilityWitness>,
    seal_sha256: String,
}

impl Qg5DurabilityWitnessSet {
    /// Seal the full selected-cell measured-sample census emitted by a QG-5 child.
    ///
    /// The benchmark writes [`Self::to_json_bytes`] to
    /// [`QG5_DURABILITY_WITNESS_FILE_NAME`] only after all selected cells have
    /// completed their real delete-publication, timed maintenance, and reopen
    /// validation events.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the run identity or cell census is
    /// invalid, a retained sample witness fails validation, or the canonical
    /// seal cannot be produced.
    pub fn seal(
        run_id: impl Into<String>,
        cells: BTreeMap<String, Qg5CellDurabilityWitness>,
    ) -> Result<Self, MachineClassError> {
        let mut set = Self {
            schema_version: QG5_DURABILITY_WITNESS_SCHEMA_VERSION.to_owned(),
            gate: "QG-5".to_owned(),
            run_id: run_id.into(),
            cells,
            seal_sha256: String::new(),
        };
        set.refresh_seal()?;
        set.verify()?;
        Ok(set)
    }

    /// Return the exact canonical bytes the child must publish.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the witness set no longer verifies
    /// or cannot be serialized into canonical JSON.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, MachineClassError> {
        self.verify()?;
        qg5_canonical_bytes(self)
    }

    /// Strictly parse and verify exact child-published witness bytes.
    ///
    /// # Errors
    ///
    /// Returns [`MachineClassError`] when the bytes are malformed, contain
    /// unknown fields, are not exact canonical compact JSON, or describe an
    /// invalid or incorrectly sealed witness set.
    pub fn from_verified_slice(bytes: &[u8]) -> Result<Self, MachineClassError> {
        let value = parse_strict_json(bytes).map_err(|error| {
            qg5_witness_error(format!("QG-5 witness set is not strict JSON: {error}"))
        })?;
        let set = serde_json::from_value::<Self>(value).map_err(|error| {
            qg5_witness_error(format!("QG-5 witness set is malformed: {error}"))
        })?;
        let canonical = qg5_canonical_bytes(&set)?;
        if bytes != canonical {
            return Err(qg5_witness_error(
                "QG-5 witness set bytes are not exact canonical compact JSON",
            ));
        }
        set.verify()?;
        Ok(set)
    }

    fn verify_for_run(
        &self,
        run_id: &str,
        selected_cell_ids: &[String],
    ) -> Result<(), MachineClassError> {
        self.verify()?;
        let selected = selected_cell_ids.iter().cloned().collect::<BTreeSet<_>>();
        if !valid_qg5_identity(run_id)
            || self.run_id != run_id
            || selected.is_empty()
            || selected.len() != selected_cell_ids.len()
            || selected.iter().any(|cell| !valid_qg5_cell(cell))
            || self.cells.keys().cloned().collect::<BTreeSet<_>>() != selected
        {
            return Err(qg5_witness_error(
                "QG-5 witness set does not bind the exact runner ID and selected cell census",
            ));
        }
        Ok(())
    }

    /// Verify exact one-to-one binding against the parent-derived raw-sample
    /// census retained in canonical evidence.
    pub(crate) fn verify_for_run_and_census(
        &self,
        run_id: &str,
        selected_cell_ids: &[String],
        expected: &Qg5ExpectedDurabilityCensus,
    ) -> Result<(), MachineClassError> {
        self.verify_for_run(run_id, selected_cell_ids)?;
        if expected.run_id != run_id
            || expected.cell_ids() != selected_cell_ids.iter().cloned().collect()
            || self.census_binding_sha256()? != expected.binding_sha256()?
        {
            return Err(qg5_witness_error(
                "QG-5 witness sample keys do not exactly equal the evidence-derived census",
            ));
        }
        for (cell, witness) in &self.cells {
            let expected_samples = expected.cells.get(cell).ok_or_else(|| {
                qg5_witness_error("QG-5 witness contains a cell absent from canonical evidence")
            })?;
            if witness.samples.len() != expected_samples.len() {
                return Err(qg5_witness_error(
                    "QG-5 witness has a missing or extra measured sample",
                ));
            }
            for (observed, expected_sample) in witness.samples.iter().zip(expected_samples.iter()) {
                if observed.binding() != expected_sample.binding {
                    return Err(qg5_witness_error(
                        "QG-5 witness has a duplicate, missing, extra, or stream-swapped sample",
                    ));
                }
                observed
                    .observation
                    .validate_against_expected(expected_sample)?;
            }
        }
        Ok(())
    }

    fn verify_for_scope(&self, scope: &Qg5DurabilityWitnessScope) -> Result<(), MachineClassError> {
        self.verify_for_run(&scope.run_id, &scope.selected_cell_ids)?;
        if self.census_binding_sha256()? != scope.expected_census_sha256 {
            return Err(qg5_witness_error(
                "QG-5 witness sample census differs from the parent-retained evidence binding",
            ));
        }
        Ok(())
    }

    fn census_binding_sha256(&self) -> Result<String, MachineClassError> {
        let cells = self
            .cells
            .iter()
            .map(|(cell, witness)| (cell.clone(), witness.bindings()))
            .collect();
        qg5_census_binding_sha256(&self.run_id, &cells)
    }

    fn seal_preimage(&self) -> Result<Vec<u8>, MachineClassError> {
        let mut unsealed = self.clone();
        unsealed.seal_sha256.clear();
        qg5_canonical_bytes(&unsealed)
    }

    fn refresh_seal(&mut self) -> Result<(), MachineClassError> {
        self.seal_sha256 = sha256_hex(&self.seal_preimage()?);
        Ok(())
    }

    fn verify(&self) -> Result<(), MachineClassError> {
        if self.schema_version != QG5_DURABILITY_WITNESS_SCHEMA_VERSION
            || self.gate != "QG-5"
            || !valid_qg5_identity(&self.run_id)
            || self.cells.is_empty()
            || !is_sha256(&self.seal_sha256)
            || self.seal_sha256 != sha256_hex(&self.seal_preimage()?)
        {
            return Err(qg5_witness_error(
                "QG-5 witness set is malformed, unsealed, or has no arm witnesses",
            ));
        }
        for (cell, witness) in &self.cells {
            if witness.cell != *cell {
                return Err(qg5_witness_error(
                    "QG-5 witness set has a cell identity mismatch",
                ));
            }
            witness.verify()?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct Qg5CensusBinding<'a> {
    schema_version: &'static str,
    run_id: &'a str,
    cells: &'a BTreeMap<String, Vec<Qg5SampleBinding>>,
}

fn qg5_census_binding_sha256(
    run_id: &str,
    cells: &BTreeMap<String, Vec<Qg5SampleBinding>>,
) -> Result<String, MachineClassError> {
    let binding = Qg5CensusBinding {
        schema_version: QG5_DURABILITY_WITNESS_SCHEMA_VERSION,
        run_id,
        cells,
    };
    let bytes = qg5_canonical_bytes(&binding)?;
    qg5_domain_sha256(QG5_CENSUS_HASH_DOMAIN, &bytes)
}

fn qg5_domain_sha256(domain: &[u8], payload: &[u8]) -> Result<String, MachineClassError> {
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| qg5_witness_error("QG-5 hash payload length does not fit u64"))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(payload_len.to_le_bytes());
    hasher.update(payload);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    Ok(output)
}

fn qg5_canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, MachineClassError> {
    let value = serde_json::to_value(value).map_err(|error| {
        qg5_witness_error(format!("QG-5 witness serialization failed: {error}"))
    })?;
    canonical_json_bytes(&value).map_err(|error| {
        qg5_witness_error(format!("QG-5 witness canonicalization failed: {error}"))
    })
}

fn qg5_witness_error(detail: impl Into<String>) -> MachineClassError {
    MachineClassError::new(MachineClassReason::DurabilityAsymmetric, detail)
}

fn valid_qg5_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.trim() == value
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn valid_qg5_cell(value: &str) -> bool {
    value.starts_with("QG-5/") && valid_qg5_identity(value) && value.len() <= 240
}

pub fn qg5_pending_runner_durability() -> RunnerDurability {
    RunnerDurability {
        adjacent: true,
        control_treatment: QG5_PENDING_DURABILITY_TREATMENT.to_owned(),
        candidate_treatment: QG5_PENDING_DURABILITY_TREATMENT.to_owned(),
        symmetric: false,
    }
}

pub fn qg5_post_exit_runner_durability(witness_bytes: &[u8]) -> RunnerDurability {
    let treatment = format!(
        "{QG5_POST_EXIT_DURABILITY_PREFIX}{}",
        sha256_hex(witness_bytes)
    );
    RunnerDurability {
        adjacent: true,
        control_treatment: treatment.clone(),
        candidate_treatment: treatment,
        symmetric: true,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerCompletion {
    pub(crate) verified: bool,
    pub(crate) exit_status: i64,
    pub(crate) run_log_sha256: String,
    pub(crate) artifact_manifest_sha256: String,
    pub(crate) artifact_digests_verified: bool,
    pub(crate) started_at_utc: String,
    pub(crate) finished_at_utc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerReceipt {
    pub(crate) schema_version: String,
    pub(crate) requested_profile: MachineProfileKey,
    pub(crate) derived_profile: MachineProfileKey,
    pub(crate) registry_sha256: String,
    pub(crate) hardware: RunnerHardware,
    pub(crate) execution: RunnerExecution,
    pub(crate) build: RunnerBuild,
    pub(crate) durability: RunnerDurability,
    pub(crate) completion: RunnerCompletion,
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

pub fn sha256_hex(bytes: &[u8]) -> String {
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

fn is_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
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

fn string_set_equals(actual: &[String], expected: &[&str]) -> bool {
    actual.len() == expected.len()
        && actual.iter().collect::<BTreeSet<_>>().len() == actual.len()
        && actual.iter().map(String::as_str).collect::<BTreeSet<_>>()
            == expected.iter().copied().collect::<BTreeSet<_>>()
}

fn profile_facts_are(profile: &MachineExecutionProfile, expected: &[&str]) -> bool {
    string_set_equals(&profile.required_scheduler_or_affinity_facts, expected)
}

fn profile_forbidden_are(profile: &MachineExecutionProfile, expected: &[&str]) -> bool {
    string_set_equals(&profile.forbidden_claims, expected)
}

fn profile_all_gates_have_disposition(
    profile: &MachineExecutionProfile,
    expected: DefaultFlipDisposition,
) -> bool {
    profile
        .gate_policies
        .values()
        .all(|policy| policy.default_flip_disposition == expected)
}

fn profile_gate_widths_match(
    profile: &MachineExecutionProfile,
    expected_widths: [Option<u64>; 10],
) -> bool {
    expected_widths
        .into_iter()
        .enumerate()
        .all(|(index, expected)| {
            let gate = index + 1;
            profile
                .gate_policies
                .get(&format!("QG-{gate}"))
                .is_some_and(|policy| policy.max_exercised_cell_width == expected)
        })
}

fn validate_canonical_execution_profile(
    profile: &MachineExecutionProfile,
) -> Result<(), MachineClassError> {
    let valid = match profile.key {
        MachineProfileKey {
            hardware_class_id: HardwareClassId::X86VpsOvh,
            execution_profile_id: ExecutionProfileId::X86Diagnostic,
        } => {
            profile.availability == MachineProfileAvailability::Registered
                && profile.capacity_semantics == ExecutionCapacitySemantics::DiagnosticWorkerBudget
                && profile.execution_capacity.is_none()
                && profile_all_gates_have_disposition(
                    profile,
                    DefaultFlipDisposition::DiagnosticOnly,
                )
                && profile_gate_widths_match(profile, [None; 10])
                && profile_facts_are(
                    profile,
                    &[
                        "requested-worker-budget",
                        "observable-worker-activity",
                        "local-execution",
                        "exclusive-lease",
                    ],
                )
                && profile_forbidden_are(
                    profile,
                    &[
                        "hardware-homogeneity",
                        "default-flip-authority",
                        "cross-worker-pooling",
                    ],
                )
        }
        MachineProfileKey {
            hardware_class_id: HardwareClassId::TrjZen35995wx,
            execution_profile_id: ExecutionProfileId::Physical64,
        } => {
            profile.availability == MachineProfileAvailability::Registered
                && profile.capacity_semantics == ExecutionCapacitySemantics::PhysicalCores
                && profile.execution_capacity == Some(64)
                && profile_all_gates_have_disposition(
                    profile,
                    DefaultFlipDisposition::RequiredForDefaultFlip,
                )
                && profile_gate_widths_match(
                    profile,
                    [
                        Some(64),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(8),
                        Some(32),
                        Some(1),
                        Some(1),
                    ],
                )
                && profile_facts_are(
                    profile,
                    &[
                        "effective-cpuset",
                        "physical-core-sibling-map",
                        "one-thread-per-core",
                        "smt-state",
                        "numa-policy",
                        "governor",
                        "observable-worker-activity",
                        "local-execution",
                        "exclusive-lease",
                    ],
                )
                && profile_forbidden_are(
                    profile,
                    &[
                        "logical-thread-capacity-128",
                        "cross-profile-evidence-reuse",
                    ],
                )
        }
        MachineProfileKey {
            hardware_class_id: HardwareClassId::TrjZen35995wx,
            execution_profile_id: ExecutionProfileId::Smt2_128,
        } => {
            profile.availability == MachineProfileAvailability::Registered
                && profile.capacity_semantics == ExecutionCapacitySemantics::LogicalThreads
                && profile.execution_capacity == Some(128)
                && profile_all_gates_have_disposition(
                    profile,
                    DefaultFlipDisposition::RequiredForDefaultFlip,
                )
                && profile_gate_widths_match(
                    profile,
                    [
                        Some(128),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(8),
                        Some(32),
                        Some(1),
                        Some(1),
                    ],
                )
                && profile_facts_are(
                    profile,
                    &[
                        "effective-cpuset",
                        "physical-core-sibling-map",
                        "two-threads-per-core",
                        "smt-state",
                        "numa-policy",
                        "governor",
                        "observable-worker-activity",
                        "local-execution",
                        "exclusive-lease",
                    ],
                )
                && profile_forbidden_are(
                    profile,
                    &["physical-only-residency", "cross-profile-evidence-reuse"],
                )
        }
        MachineProfileKey {
            hardware_class_id: HardwareClassId::M4Macos,
            execution_profile_id: ExecutionProfileId::Scheduler10,
        } => {
            profile.availability == MachineProfileAvailability::Registered
                && profile.capacity_semantics == ExecutionCapacitySemantics::SchedulerWorkers
                && profile.execution_capacity == Some(10)
                && profile_all_gates_have_disposition(
                    profile,
                    DefaultFlipDisposition::RequiredForDefaultFlip,
                )
                && profile_gate_widths_match(
                    profile,
                    [
                        Some(8),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(1),
                        Some(8),
                        Some(8),
                        Some(1),
                        Some(1),
                    ],
                )
                && profile_facts_are(
                    profile,
                    &[
                        "requested-pool-width",
                        "requested-qos",
                        "observable-worker-activity",
                        "thermal-pressure",
                        "page-size",
                        "local-execution",
                        "exclusive-lease",
                        "executing-image-attestation",
                    ],
                )
                && profile_forbidden_are(
                    profile,
                    &[
                        "p-core-affinity",
                        "e-core-affinity",
                        "p-core-residency",
                        "e-core-residency",
                        "invented-width-10-cell",
                    ],
                )
        }
        MachineProfileKey {
            hardware_class_id: HardwareClassId::M5Macos,
            execution_profile_id: ExecutionProfileId::Scheduler14,
        } => {
            profile.availability == MachineProfileAvailability::Unavailable
                && profile.capacity_semantics == ExecutionCapacitySemantics::SchedulerWorkers
                && profile.execution_capacity.is_none()
                && profile_all_gates_have_disposition(
                    profile,
                    DefaultFlipDisposition::RequiredForDefaultFlip,
                )
                && profile_gate_widths_match(profile, [None; 10])
                && profile_facts_are(profile, &[])
                && profile_forbidden_are(
                    profile,
                    &[
                        "m4-substitution",
                        "fabricated-hardware-fingerprint",
                        "fabricated-capacity",
                        "all-not-applicable-plan",
                    ],
                )
        }
        _ => false,
    };
    if !valid
        || !is_sha256(&profile.contract_sha256)
        || unique_string_count(&profile.forbidden_claims) != profile.forbidden_claims.len()
    {
        return Err(MachineClassError::new(
            MachineClassReason::ExecutionProfileContractInvalid,
            format!(
                "canonical execution profile {}.{} violates its frozen contract",
                profile.key.hardware_class_id.as_str(),
                profile.key.execution_profile_id.as_str()
            ),
        ));
    }
    Ok(())
}

/// Loaded, self-consistent exact machine-class registry.
#[derive(Debug, Clone)]
pub struct MachineClassRegistry {
    #[cfg(test)]
    raw: Value,
    hardware_classes: Vec<RegistryHardwareClassRule>,
    execution_profiles: Vec<MachineExecutionProfile>,
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
            "artifact_manifest_contract",
            "hardware_classes",
            "execution_profiles",
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
        validate_registry_receipt_contract(&raw)?;
        validate_registry_artifact_manifest_contract(&raw)?;

        let hardware_classes = required_array(&raw, "hardware_classes")?
            .iter()
            .map(|value| {
                serde_json::from_value::<RegistryHardwareClassRule>(value.clone()).map_err(
                    |error| {
                        let reason = if error.to_string().contains("unknown field") {
                            MachineClassReason::UnknownField
                        } else {
                            MachineClassReason::MissingField
                        };
                        MachineClassError::new(reason, error.to_string())
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::validate_hardware_class_rules(&hardware_classes, &source)?;
        let execution_profiles = required_array(&raw, "execution_profiles")?
            .iter()
            .map(|value| {
                let profile = value.as_object().ok_or_else(|| {
                    MachineClassError::new(
                        MachineClassReason::MissingField,
                        "each execution profile must be an object",
                    )
                })?;
                if !profile.contains_key("execution_capacity") {
                    return Err(MachineClassError::new(
                        MachineClassReason::MissingField,
                        "execution profile is missing required field \"execution_capacity\"",
                    ));
                }
                let gate_policies = profile
                    .get("gate_policies")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        MachineClassError::new(
                            MachineClassReason::MissingField,
                            "execution profile field \"gate_policies\" must be an object",
                        )
                    })?;
                for (gate, policy) in gate_policies {
                    let policy = policy.as_object().ok_or_else(|| {
                        MachineClassError::new(
                            MachineClassReason::MissingField,
                            format!("execution profile gate policy {gate:?} must be an object"),
                        )
                    })?;
                    if !policy.contains_key("max_exercised_cell_width") {
                        return Err(MachineClassError::new(
                            MachineClassReason::MissingField,
                            format!(
                                "execution profile gate policy {gate:?} is missing required field \
                                 \"max_exercised_cell_width\""
                            ),
                        ));
                    }
                }
                let rule = serde_json::from_value::<RegistryExecutionProfileRule>(value.clone())
                    .map_err(|error| {
                        let detail = error.to_string();
                        let reason = if detail.contains("unknown field") {
                            MachineClassReason::UnknownField
                        } else if detail.contains("missing field") {
                            MachineClassReason::MissingField
                        } else if detail
                            .starts_with(MachineClassReason::ExecutionProfileClassMismatch.as_str())
                        {
                            MachineClassReason::ExecutionProfileClassMismatch
                        } else {
                            MachineClassReason::ExecutionProfileContractInvalid
                        };
                        MachineClassError::new(reason, detail)
                    })?;
                Ok(MachineExecutionProfile {
                    key: rule.key,
                    availability: rule.availability,
                    capacity_semantics: rule.capacity_semantics,
                    execution_capacity: rule.execution_capacity.0,
                    required_scheduler_or_affinity_facts: rule.required_scheduler_or_affinity_facts,
                    forbidden_claims: rule.forbidden_claims,
                    gate_policies: rule
                        .gate_policies
                        .into_iter()
                        .map(|(gate, policy)| {
                            (
                                gate,
                                MachineProfileGatePolicy {
                                    default_flip_disposition: policy.default_flip_disposition,
                                    max_exercised_cell_width: policy.max_exercised_cell_width.0,
                                },
                            )
                        })
                        .collect(),
                    contract_sha256: hash_profile_contract(value)?,
                })
            })
            .collect::<Result<Vec<_>, MachineClassError>>()?;
        Self::validate_execution_profiles(&hardware_classes, &execution_profiles)?;
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
            hardware_classes,
            execution_profiles,
            receipt_shapes,
            canonical_hash_contract_sha256,
        })
    }

    fn validate_hardware_class_rules(
        hardware_classes: &[RegistryHardwareClassRule],
        source: &impl Fn(&str) -> Option<&'static [u8]>,
    ) -> Result<(), MachineClassError> {
        if hardware_classes.is_empty() {
            return Err(MachineClassError::new(
                MachineClassReason::MissingField,
                "registry has no hardware-class rules",
            ));
        }
        let expected = [
            HardwareClassId::X86VpsOvh,
            HardwareClassId::TrjZen35995wx,
            HardwareClassId::M4Macos,
            HardwareClassId::M5Macos,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut identities = BTreeSet::new();
        for class in hardware_classes {
            if !identities.insert(class.hardware_class_id) {
                return Err(MachineClassError::new(
                    MachineClassReason::AmbiguousClassId,
                    format!(
                        "duplicate hardware class {:?}",
                        class.hardware_class_id.as_str()
                    ),
                ));
            }
            validate_hardware_predicate_contract(class)?;
            if class.availability == MachineProfileAvailability::Registered
                && !matches!(class.hardware_class_id, HardwareClassId::X86VpsOvh)
                && class.source_fingerprints.is_empty()
            {
                return Err(MachineClassError::new(
                    MachineClassReason::SourceIdentityInvalid,
                    format!(
                        "registered hardware class {:?} has no provenance",
                        class.hardware_class_id.as_str()
                    ),
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
        if identities != expected {
            return Err(MachineClassError::new(
                MachineClassReason::SourceIdentityInvalid,
                "registry must contain exactly the four canonical hardware classes",
            ));
        }
        Ok(())
    }

    fn validate_execution_profiles(
        hardware_classes: &[RegistryHardwareClassRule],
        profiles: &[MachineExecutionProfile],
    ) -> Result<(), MachineClassError> {
        if profiles.is_empty() {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionProfileContractInvalid,
                "registry has no execution profiles",
            ));
        }

        let expected_gates = (1..=10)
            .map(|gate| format!("QG-{gate}"))
            .collect::<BTreeSet<_>>();
        let expected_identities = [
            MachineProfileKey {
                hardware_class_id: HardwareClassId::X86VpsOvh,
                execution_profile_id: ExecutionProfileId::X86Diagnostic,
            },
            MachineProfileKey {
                hardware_class_id: HardwareClassId::TrjZen35995wx,
                execution_profile_id: ExecutionProfileId::Physical64,
            },
            MachineProfileKey {
                hardware_class_id: HardwareClassId::TrjZen35995wx,
                execution_profile_id: ExecutionProfileId::Smt2_128,
            },
            MachineProfileKey {
                hardware_class_id: HardwareClassId::M4Macos,
                execution_profile_id: ExecutionProfileId::Scheduler10,
            },
            MachineProfileKey {
                hardware_class_id: HardwareClassId::M5Macos,
                execution_profile_id: ExecutionProfileId::Scheduler14,
            },
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut identities = BTreeSet::new();
        for profile in profiles {
            if !identities.insert(profile.key) {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionProfileContractInvalid,
                    format!(
                        "duplicate execution-profile identity {}.{}",
                        profile.key.hardware_class_id.as_str(),
                        profile.key.execution_profile_id.as_str()
                    ),
                ));
            }

            let matching_classes = hardware_classes
                .iter()
                .filter(|class| class.hardware_class_id == profile.key.hardware_class_id)
                .collect::<Vec<_>>();
            let [hardware_class] = matching_classes.as_slice() else {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionProfileClassMismatch,
                    format!(
                        "execution profile {}.{} does not resolve to exactly one hardware class",
                        profile.key.hardware_class_id.as_str(),
                        profile.key.execution_profile_id.as_str()
                    ),
                ));
            };
            let profile_gates = profile
                .gate_policies
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            if profile_gates != expected_gates {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionProfileContractInvalid,
                    format!(
                        "execution profile {}.{} must classify exactly QG-1 through QG-10",
                        profile.key.hardware_class_id.as_str(),
                        profile.key.execution_profile_id.as_str()
                    ),
                ));
            }

            match profile.availability {
                MachineProfileAvailability::Registered => {
                    if profile.execution_capacity == Some(0)
                        || hardware_class.availability == MachineProfileAvailability::Unavailable
                        || profile.required_scheduler_or_affinity_facts.is_empty()
                        || unique_string_count(&profile.required_scheduler_or_affinity_facts)
                            != profile.required_scheduler_or_affinity_facts.len()
                        || profile.gate_policies.values().any(|policy| {
                            policy.max_exercised_cell_width == Some(0)
                                || profile.execution_capacity.is_some_and(|capacity| {
                                    policy
                                        .max_exercised_cell_width
                                        .is_some_and(|width| width > capacity)
                                })
                        })
                    {
                        return Err(MachineClassError::new(
                            MachineClassReason::ExecutionProfileContractInvalid,
                            format!(
                                "registered execution profile {}.{} has inconsistent capacity, facts, class, or gate policy",
                                profile.key.hardware_class_id.as_str(),
                                profile.key.execution_profile_id.as_str()
                            ),
                        ));
                    }
                }
                MachineProfileAvailability::Unavailable => {
                    if profile.execution_capacity.is_some()
                        || profile.gate_policies.values().any(|policy| {
                            policy.max_exercised_cell_width.is_some()
                                || policy.default_flip_disposition
                                    != DefaultFlipDisposition::RequiredForDefaultFlip
                        })
                    {
                        return Err(MachineClassError::new(
                            MachineClassReason::ExecutionProfileContractInvalid,
                            format!(
                                "unavailable execution profile {}.{} must remain required without fabricating capacity or gate widths",
                                profile.key.hardware_class_id.as_str(),
                                profile.key.execution_profile_id.as_str()
                            ),
                        ));
                    }
                }
            }
            validate_canonical_execution_profile(profile)?;
        }
        if identities != expected_identities {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionProfileContractInvalid,
                "registry must contain exactly the five canonical execution profiles",
            ));
        }
        Ok(())
    }

    /// Resolve a class without admitting a runner receipt.
    #[must_use]
    pub fn lookup(&self, requested_hardware_class_id: &str) -> MachineClassLookup {
        match parse_hardware_class_id(requested_hardware_class_id)
            .and_then(|class_id| self.resolve_hardware(class_id))
        {
            Ok(rule) if rule.availability == MachineProfileAvailability::Registered => {
                MachineClassLookup {
                    decision: MachineClassDecision::Allow,
                    hardware_class_id: Some(requested_hardware_class_id.to_owned()),
                    reason: MachineClassReason::Admitted,
                }
            }
            Ok(rule) => MachineClassLookup {
                decision: MachineClassDecision::DiagnosticOnly,
                hardware_class_id: Some(requested_hardware_class_id.to_owned()),
                reason: rule.admission_reason,
            },
            Err(error) => MachineClassLookup {
                decision: MachineClassDecision::Reject,
                hardware_class_id: None,
                reason: error.reason,
            },
        }
    }

    /// Resolve one immutable hardware class and execution-profile identity.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection for an unknown profile or a profile attached
    /// to another hardware class. Unavailable profiles remain resolvable so a
    /// caller can report their exact typed disposition without inventing a
    /// capacity or applicability plan.
    pub fn execution_profile(
        &self,
        key: MachineProfileKey,
    ) -> Result<&MachineExecutionProfile, MachineClassError> {
        let matches = self
            .execution_profiles
            .iter()
            .filter(|profile| profile.key == key)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [profile] => Ok(profile),
            [] if self
                .execution_profiles
                .iter()
                .any(|profile| profile.key.execution_profile_id == key.execution_profile_id) =>
            {
                Err(MachineClassError::new(
                    MachineClassReason::ExecutionProfileClassMismatch,
                    format!(
                        "execution profile {:?} is not registered for hardware class {:?}",
                        key.execution_profile_id.as_str(),
                        key.hardware_class_id.as_str()
                    ),
                ))
            }
            [] => Err(MachineClassError::new(
                MachineClassReason::UnknownExecutionProfile,
                format!(
                    "no execution profile {:?}.{:?} is registered",
                    key.hardware_class_id.as_str(),
                    key.execution_profile_id.as_str()
                ),
            )),
            _ => Err(MachineClassError::new(
                MachineClassReason::ExecutionProfileContractInvalid,
                format!(
                    "multiple execution profiles match {:?}.{:?}",
                    key.hardware_class_id.as_str(),
                    key.execution_profile_id.as_str()
                ),
            )),
        }
    }

    fn resolve_hardware(
        &self,
        hardware_class_id: HardwareClassId,
    ) -> Result<&RegistryHardwareClassRule, MachineClassError> {
        let matches = self
            .hardware_classes
            .iter()
            .filter(|class| class.hardware_class_id == hardware_class_id)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(MachineClassError::new(
                MachineClassReason::UnknownClassId,
                format!(
                    "no hardware-class rule matches {:?}",
                    hardware_class_id.as_str()
                ),
            )),
            [rule] => Ok(rule),
            _ => Err(MachineClassError::new(
                MachineClassReason::AmbiguousClassId,
                format!(
                    "multiple hardware-class rules match {:?}",
                    hardware_class_id.as_str()
                ),
            )),
        }
    }

    fn resolve_profile(
        &self,
        key: MachineProfileKey,
    ) -> Result<ResolvedProfile<'_>, MachineClassError> {
        Ok(ResolvedProfile {
            hardware: self.resolve_hardware(key.hardware_class_id)?,
            profile: self.execution_profile(key)?,
        })
    }

    /// Validate the exact hardware/execution envelope before a measured child
    /// can spawn.
    ///
    /// This uses the same registry predicates and precedence as terminal
    /// receipt admission. Build, completion, and artifact facts intentionally
    /// remain outside this pre-spawn token because they are validated at the
    /// final commit boundary.
    pub(crate) fn preflight(
        &self,
        requested_profile: MachineProfileKey,
        hardware: RunnerHardware,
        request: RunnerExecutionRequest,
        snapshot: RunnerExecutionSnapshot,
        durability: RunnerDurability,
        context: &MachineClassAdmissionContext,
    ) -> Result<PreSpawnAdmission, MachineClassError> {
        let receipt = RunnerReceipt {
            schema_version: RUNNER_RECEIPT_SCHEMA_VERSION.to_owned(),
            requested_profile,
            derived_profile: requested_profile,
            registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
            hardware,
            execution: RunnerExecution {
                request,
                start: snapshot.clone(),
                end: snapshot,
                identity_sha256: String::new(),
            },
            build: RunnerBuild {
                git_revision: "0".repeat(40),
                git_dirty: false,
                worktree_state_sha256: None,
                cargo_lock_sha256: "0".repeat(64),
                executable_sha256: "0".repeat(64),
                command_sha256: "0".repeat(64),
                environment_sha256: "0".repeat(64),
                producer: RunnerProducer {
                    contract_version: LOCAL_PERF_PRODUCER_CONTRACT_VERSION.to_owned(),
                    source_git_revision: "0".repeat(40),
                    source_git_dirty: false,
                    cargo_lock_sha256: "0".repeat(64),
                    executable_sha256: "0".repeat(64),
                },
            },
            durability,
            completion: RunnerCompletion {
                verified: true,
                exit_status: 0,
                run_log_sha256: "0".repeat(64),
                artifact_manifest_sha256: "0".repeat(64),
                artifact_digests_verified: true,
                started_at_utc: "pre-spawn".to_owned(),
                finished_at_utc: "pre-spawn".to_owned(),
            },
        };
        let sealed = seal_runner_receipt(receipt)?;
        let receipt = serde_json::from_slice::<RunnerReceipt>(&sealed).map_err(|error| {
            MachineClassError::new(MachineClassReason::SourceIdentityInvalid, error.to_string())
        })?;
        let resolved = self.resolve_profile(receipt.requested_profile)?;
        validate_admission_envelope(&receipt, &resolved, context)?;
        validate_pre_spawn_durability(&receipt.durability, context)?;
        validate_pre_spawn_gate_profile_policy(&resolved, context)?;
        validate_destination(context, receipt.derived_profile)?;
        let derived = derive_hashes(&receipt)?;
        Ok(PreSpawnAdmission {
            admission_context: context.clone(),
            profile: receipt.derived_profile,
            hardware_sha256: derived.hardware,
            execution_identity_sha256: derived.identity,
            durability: serde_json::to_value(receipt.durability).map_err(|error| {
                MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
            })?,
        })
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
        self.admit_inner(receipt_bytes, context, None)
    }

    /// Admit QG-5 evidence only after the child has exited and published a
    /// complete sealed durability-witness census for its exact selected cells.
    pub(crate) fn admit_qg5_post_exit(
        &self,
        receipt_bytes: &[u8],
        context: &MachineClassAdmissionContext,
        run_id: &str,
        selected_cell_ids: &[String],
        witness_bytes: &[u8],
        expected_census: &Qg5ExpectedDurabilityCensus,
    ) -> Result<VerifiedRunnerIdentity, MachineClassError> {
        if context.gate != "QG-5" {
            return Err(qg5_witness_error(
                "post-exit QG-5 durability admission cannot authenticate another gate",
            ));
        }
        let witnesses = Qg5DurabilityWitnessSet::from_verified_slice(witness_bytes)?;
        witnesses.verify_for_run_and_census(run_id, selected_cell_ids, expected_census)?;
        let scope = Qg5DurabilityWitnessScope::new(run_id, selected_cell_ids, expected_census)?;
        self.admit_inner(
            receipt_bytes,
            context,
            Some((&witnesses, witness_bytes, &scope)),
        )
    }

    fn admit_qg5_post_exit_with_scope(
        &self,
        receipt_bytes: &[u8],
        context: &MachineClassAdmissionContext,
        witness_bytes: &[u8],
        scope: &Qg5DurabilityWitnessScope,
    ) -> Result<VerifiedRunnerIdentity, MachineClassError> {
        if context.gate != "QG-5" {
            return Err(qg5_witness_error(
                "post-exit QG-5 durability admission cannot authenticate another gate",
            ));
        }
        let witnesses = Qg5DurabilityWitnessSet::from_verified_slice(witness_bytes)?;
        witnesses.verify_for_scope(scope)?;
        self.admit_inner(
            receipt_bytes,
            context,
            Some((&witnesses, witness_bytes, scope)),
        )
    }

    fn admit_inner(
        &self,
        receipt_bytes: &[u8],
        context: &MachineClassAdmissionContext,
        qg5_witness: Option<(&Qg5DurabilityWitnessSet, &[u8], &Qg5DurabilityWitnessScope)>,
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

        if receipt.schema_version != RUNNER_RECEIPT_SCHEMA_VERSION {
            return Err(MachineClassError::new(
                MachineClassReason::SourceIdentityInvalid,
                "runner receipt names an unsupported schema version",
            ));
        }
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

        let resolved = self.resolve_profile(receipt.requested_profile)?;
        validate_admission_envelope(&receipt, &resolved, context)?;
        validate_source_identity(&receipt)?;
        if context.gate == "QG-5" {
            let (witnesses, witness_bytes, scope) = qg5_witness.ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::ClassUnavailable,
                    "QG-5 is not promotion-admissible until both arms emit a non-declarative symmetric durability-treatment witness",
                )
            })?;
            witnesses.verify_for_scope(scope)?;
            validate_qg5_post_exit_durability(&receipt.durability, witnesses, witness_bytes)?;
        } else {
            if qg5_witness.is_some() {
                return Err(qg5_witness_error(
                    "post-exit QG-5 durability witnesses cannot authenticate another gate",
                ));
            }
            validate_durability(&receipt.durability)?;
        }
        validate_completion(&receipt.completion)?;
        validate_gate_profile_policy(&resolved, context, qg5_witness.is_some())?;
        validate_destination(context, receipt.derived_profile)?;

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
            profile: receipt.derived_profile,
            capacity_semantics: receipt.execution.request.capacity_semantics,
            execution_capacity: receipt.execution.request.execution_capacity,
            max_exercised_cell_width: receipt.execution.request.max_exercised_cell_width,
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
            artifact_manifest: None,
            qg5_durability_witnesses: qg5_witness.map(|(witnesses, _, _)| witnesses.clone()),
            qg5_durability_witness_scope: qg5_witness.map(|(_, _, scope)| scope.clone()),
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

fn validate_registry_receipt_contract(registry: &Value) -> Result<(), MachineClassError> {
    let contract = registry
        .get("receipt_contract")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::MissingField,
                "registry receipt_contract must be an object",
            )
        })?;
    let schema_version = contract
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let required_sections = contract_string_array(contract, "required_sections")?;
    let required_hardware_fields = contract_string_array(contract, "required_hardware_fields")?;
    let required_execution_fields = contract_string_array(contract, "required_execution_fields")?;
    let required_execution_request_fields =
        contract_string_array(contract, "required_execution_request_fields")?;
    let required_execution_snapshot_fields =
        contract_string_array(contract, "required_execution_snapshot_fields")?;
    let required_build_fields = contract_string_array(contract, "required_build_fields")?;
    let required_producer_fields = contract_string_array(contract, "required_producer_fields")?;
    let required_durability_fields = contract_string_array(contract, "required_durability_fields")?;
    let required_completion_fields = contract_string_array(contract, "required_completion_fields")?;
    let producer_identity = contract
        .get("producer_identity")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let producer_requirement_present = registry
        .get("requirements")
        .and_then(Value::as_array)
        .is_some_and(|requirements| {
            requirements.iter().any(|requirement| {
                requirement.get("id").and_then(Value::as_str) == Some("MC-MUST-022")
                    && requirement.get("level").and_then(Value::as_str) == Some("MUST")
                    && requirement
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| {
                            text.contains("Every fresh candidate and rerun")
                                && text.contains("typed producer contract v4")
                                && text.contains("build-time Git revision and dirty posture")
                                && text.contains("build-time Cargo.lock SHA-256")
                                && text.contains("independently attested executing finalizer image")
                                && text.contains(
                                    "Producer revision and lock equal the benchmark build",
                                )
                        })
            })
        });
    if schema_version != RUNNER_RECEIPT_SCHEMA_VERSION
        || required_sections
            != [
                "schema_version",
                "requested_profile",
                "derived_profile",
                "registry_sha256",
                "hardware",
                "execution",
                "build",
                "durability",
                "completion",
            ]
            .map(str::to_owned)
        || required_hardware_fields
            != [
                "os",
                "arch",
                "cpu_vendor",
                "cpu_family",
                "cpu_model",
                "cpu_stepping",
                "cpu_model_name",
                "physical_cores",
                "logical_cpus",
                "numa_nodes",
                "memory_bytes",
                "page_size_bytes",
                "performance_cores",
                "efficiency_cores",
                "runtime_detected_isa",
                "topology_sha256",
                "fingerprint_sha256",
            ]
            .map(str::to_owned)
        || required_execution_fields
            != ["request", "start", "end", "identity_sha256"].map(str::to_owned)
        || required_execution_request_fields
            != [
                "capacity_semantics",
                "execution_capacity",
                "max_exercised_cell_width",
                "requested_logical_cpu_ids",
                "requested_physical_core_width",
                "requested_worker_pool_width",
                "requested_qos",
            ]
            .map(str::to_owned)
        || required_execution_snapshot_fields
            != [
                "observed_logical_cpu_ids",
                "effective_physical_core_ids",
                "cpu_assignment_observability",
                "effective_cpuset_sha256",
                "threads_per_core",
                "smt_state",
                "numa_node_ids",
                "numa_policy",
                "governor",
                "thermal_pressure",
                "exclusive_lease",
                "exclusive_lease_id",
                "local_execution",
                "observed_hardware_fingerprint_sha256",
                "snapshot_sha256",
            ]
            .map(str::to_owned)
        || required_build_fields
            != [
                "git_revision",
                "git_dirty",
                "worktree_state_sha256",
                "cargo_lock_sha256",
                "executable_sha256",
                "command_sha256",
                "environment_sha256",
                "producer",
            ]
            .map(str::to_owned)
        || required_producer_fields
            != [
                "contract_version",
                "source_git_revision",
                "source_git_dirty",
                "cargo_lock_sha256",
                "executable_sha256",
            ]
            .map(str::to_owned)
        || required_durability_fields
            != [
                "adjacent",
                "control_treatment",
                "candidate_treatment",
                "symmetric",
            ]
            .map(str::to_owned)
        || required_completion_fields
            != [
                "verified",
                "exit_status",
                "run_log_sha256",
                "artifact_manifest_sha256",
                "artifact_digests_verified",
                "started_at_utc",
                "finished_at_utc",
            ]
            .map(str::to_owned)
        || producer_identity.trim().is_empty()
        || !producer_requirement_present
    {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "runner receipt contract differs from the compiled strict schema",
        ));
    }
    Ok(())
}

fn contract_string_array(
    contract: &Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, MachineClassError> {
    contract
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::MissingField,
                format!("receipt_contract.{field} must be an array"),
            )
        })?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::SourceIdentityInvalid,
                    format!("receipt_contract.{field} must contain only strings"),
                )
            })
        })
        .collect()
}

fn validate_registry_artifact_manifest_contract(registry: &Value) -> Result<(), MachineClassError> {
    const MANIFEST_PRECEDENCE: &str = "exact canonical artifact manifest and actual run-log, threshold, and pre-binding-evidence binding";
    let value = registry.get("artifact_manifest_contract").ok_or_else(|| {
        MachineClassError::new(
            MachineClassReason::MissingField,
            "registry is missing artifact_manifest_contract",
        )
    })?;
    let contract = serde_json::from_value::<RegistryArtifactManifestContract>(value.clone())
        .map_err(|error| {
            let detail = error.to_string();
            let reason = if detail.contains("unknown field") {
                MachineClassReason::UnknownField
            } else {
                MachineClassReason::MissingField
            };
            MachineClassError::new(reason, detail)
        })?;
    let expected_fields = [
        "schema_version",
        "gate",
        "profile",
        "capacity_semantics",
        "execution_capacity",
        "max_exercised_cell_width",
        "applicability_plan",
        "run_id",
        "run_window",
        "run_log_sha256",
        "threshold_artifact_sha256",
        "prebinding_evidence_artifact_sha256",
    ];
    let canonical_encoding = contract.canonical_encoding.to_ascii_lowercase();
    let manifest_precedence_present = registry
        .get("validation_precedence")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.as_str() == Some(MANIFEST_PRECEDENCE))
        });
    let manifest_requirement_present = registry
        .get("requirements")
        .and_then(Value::as_array)
        .is_some_and(|requirements| {
            requirements.iter().any(|requirement| {
                requirement.get("id").and_then(Value::as_str) == Some("MC-MUST-021")
                    && requirement.get("level").and_then(Value::as_str) == Some("MUST")
                    && requirement
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| {
                            text.contains("Every fresh candidate and rerun")
                                && text.contains("exact v3 artifact manifest")
                                && text.contains("profile key")
                                && text.contains("capacity semantics")
                                && text.contains("execution capacity")
                                && text.contains("maximum exercised width")
                                && text.contains("applicability-plan binding")
                                && text.contains("actual run log")
                                && text.contains("canonical threshold bytes")
                                && text.contains("exact pre-binding evidence bytes")
                                && text.contains("gate")
                                && text.contains("run ID")
                                && text.contains("run window")
                                && text.contains("before history opens")
                        })
            })
        });
    if contract.schema_version != RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION
        || contract.unknown_field_policy != "reject"
        || contract.duplicate_key_policy != "reject"
        || contract.required_fields != expected_fields.map(str::to_owned).to_vec()
        || !canonical_encoding.contains("compact json")
        || !canonical_encoding.contains("exact")
        || contract.binding_law.trim().is_empty()
        || contract.history_law.trim().is_empty()
        || !manifest_precedence_present
        || !manifest_requirement_present
    {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "artifact manifest contract differs from the compiled strict schema",
        ));
    }
    Ok(())
}

struct ResolvedProfile<'a> {
    hardware: &'a RegistryHardwareClassRule,
    profile: &'a MachineExecutionProfile,
}

fn parse_hardware_class_id(value: &str) -> Result<HardwareClassId, MachineClassError> {
    match value {
        "x86-vps-ovh" => Ok(HardwareClassId::X86VpsOvh),
        "trj-zen3-5995wx" => Ok(HardwareClassId::TrjZen35995wx),
        "m4-macos" => Ok(HardwareClassId::M4Macos),
        "m5-macos" => Ok(HardwareClassId::M5Macos),
        "p-plus-e" | "trj-zen-128c" => Err(MachineClassError::new(
            MachineClassReason::ObsoleteClassId,
            format!("{value:?} is a legacy execution label, not a hardware class"),
        )),
        _ => {
            let legacy_width = value.strip_prefix("trj-zen3-").and_then(|suffix| {
                suffix
                    .strip_suffix("c-smt2")
                    .or_else(|| suffix.strip_suffix('c'))
                    .filter(|width| !width.starts_with('0'))
                    .and_then(|width| width.parse::<u64>().ok())
                    .filter(|width| (1..=64).contains(width))
            });
            if legacy_width.is_some() {
                Err(MachineClassError::new(
                    MachineClassReason::ObsoleteClassId,
                    format!("{value:?} is a width-encoded legacy execution label"),
                ))
            } else {
                Err(MachineClassError::new(
                    MachineClassReason::UnknownClassId,
                    format!("no hardware-class rule matches {value:?}"),
                ))
            }
        }
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
    fn check_canonical_subset(value: &Value) -> bool {
        match value {
            Value::String(value) => value.is_ascii(),
            Value::Number(value) => value.is_i64() || value.is_u64(),
            Value::Array(values) => values.iter().all(check_canonical_subset),
            Value::Object(values) => {
                values.keys().all(|key| key.is_ascii())
                    && values.values().all(check_canonical_subset)
            }
            _ => true,
        }
    }
    if !check_canonical_subset(value) {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "canonical machine identity JSON must contain only ASCII strings and integer numbers",
        ));
    }

    fn write_value(value: &Value, output: &mut Vec<u8>) -> Result<(), serde_json::Error> {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(value) => {
                output.extend_from_slice(if *value { b"true" } else { b"false" });
            }
            Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
            Value::String(value) => serde_json::to_writer(output, value)?,
            Value::Array(values) => {
                output.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    write_value(value, output)?;
                }
                output.push(b']');
            }
            Value::Object(values) => {
                output.push(b'{');
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    serde_json::to_writer(&mut *output, key)?;
                    output.push(b':');
                    write_value(&values[key], output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    write_value(value, &mut output).map_err(|error| {
        MachineClassError::new(MachineClassReason::SourceIdentityInvalid, error.to_string())
    })?;
    Ok(output)
}

fn hash_value(value: &Value) -> Result<String, MachineClassError> {
    Ok(sha256_hex(&canonical_json_bytes(value)?))
}

fn hash_profile_contract(value: &Value) -> Result<String, MachineClassError> {
    let mut preimage = b"frankensearch.machine-execution-profile.v1\0".to_vec();
    preimage.extend(canonical_json_bytes(value)?);
    Ok(sha256_hex(&preimage))
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

pub fn seal_runner_receipt(mut receipt: RunnerReceipt) -> Result<Vec<u8>, MachineClassError> {
    receipt.hardware.fingerprint_sha256.clear();
    receipt
        .execution
        .start
        .observed_hardware_fingerprint_sha256
        .clear();
    receipt
        .execution
        .end
        .observed_hardware_fingerprint_sha256
        .clear();
    receipt.execution.start.effective_cpuset_sha256.clear();
    receipt.execution.end.effective_cpuset_sha256.clear();
    receipt.execution.start.snapshot_sha256.clear();
    receipt.execution.end.snapshot_sha256.clear();
    receipt.execution.identity_sha256.clear();

    let mut hardware = serde_json::to_value(&receipt.hardware).map_err(|error| {
        MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
    })?;
    hardware
        .as_object_mut()
        .expect("serialized runner hardware is an object")
        .remove("fingerprint_sha256");
    let hardware_sha256 = hash_value(&hardware)?;
    receipt.hardware.fingerprint_sha256 = hardware_sha256.clone();
    receipt
        .execution
        .start
        .observed_hardware_fingerprint_sha256
        .clone_from(&hardware_sha256);
    receipt.execution.end.observed_hardware_fingerprint_sha256 = hardware_sha256;
    receipt.execution.start.effective_cpuset_sha256 = derive_cpuset_hash(&receipt.execution.start)?;
    receipt.execution.end.effective_cpuset_sha256 = derive_cpuset_hash(&receipt.execution.end)?;
    receipt.execution.start.snapshot_sha256 = derive_snapshot_hash(&receipt.execution.start)?;
    receipt.execution.end.snapshot_sha256 = derive_snapshot_hash(&receipt.execution.end)?;
    receipt.execution.identity_sha256 = derive_execution_identity_hash(&receipt)?;

    serde_json::to_vec(&receipt).map_err(|error| {
        MachineClassError::new(MachineClassReason::DerivedHashMismatch, error.to_string())
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
        "requested_profile": receipt.requested_profile,
        "derived_profile": receipt.derived_profile,
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
    class: &RegistryHardwareClassRule,
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
        (
            "runtime_detected_isa",
            MachineClassReason::HardwareIsaMismatch,
        ),
    ] {
        predicate_matches(&value[field], &class.hardware_predicates, field, reason)?;
    }
    validate_runtime_isa(&hardware.runtime_detected_isa)?;
    if let Some(forbidden) = class.hardware_predicates.get("forbidden_runtime_isa") {
        let forbidden = forbidden.as_array().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::SourceIdentityInvalid,
                "forbidden_runtime_isa must be an array",
            )
        })?;
        for feature in forbidden {
            let feature = feature.as_str().ok_or_else(|| {
                MachineClassError::new(
                    MachineClassReason::SourceIdentityInvalid,
                    "forbidden_runtime_isa entries must be strings",
                )
            })?;
            if hardware
                .runtime_detected_isa
                .binary_search_by(|candidate| candidate.as_str().cmp(feature))
                .is_ok()
            {
                return Err(MachineClassError::new(
                    MachineClassReason::HardwareIsaMismatch,
                    format!("runtime-detected ISA includes forbidden feature {feature:?}"),
                ));
            }
        }
    }
    if !is_sha256(&hardware.topology_sha256) {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "topology_sha256 is not lowercase SHA-256",
        ));
    }
    Ok(())
}

fn validate_hardware_predicate_contract(
    class: &RegistryHardwareClassRule,
) -> Result<(), MachineClassError> {
    const CONSUMED_FIELDS: &[&str] = &[
        "os",
        "arch",
        "cpu_vendor",
        "cpu_family",
        "cpu_model",
        "cpu_stepping",
        "cpu_model_name",
        "physical_cores",
        "logical_cpus",
        "numa_nodes",
        "memory_bytes",
        "page_size_bytes",
        "performance_cores",
        "efficiency_cores",
        "runtime_detected_isa",
        "forbidden_runtime_isa",
    ];
    if let Some(field) = class
        .hardware_predicates
        .keys()
        .find(|field| !CONSUMED_FIELDS.contains(&field.as_str()))
    {
        return Err(MachineClassError::new(
            MachineClassReason::UnknownField,
            format!(
                "class family {:?} has unconsumed hardware predicate {field:?}",
                class.family
            ),
        ));
    }
    if let Some(forbidden) = class.hardware_predicates.get("forbidden_runtime_isa") {
        let forbidden = forbidden.as_array().ok_or_else(|| {
            MachineClassError::new(
                MachineClassReason::SourceIdentityInvalid,
                "forbidden_runtime_isa must be an array",
            )
        })?;
        let forbidden = forbidden
            .iter()
            .map(|feature| {
                feature.as_str().map(str::to_owned).ok_or_else(|| {
                    MachineClassError::new(
                        MachineClassReason::SourceIdentityInvalid,
                        "forbidden_runtime_isa entries must be strings",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_runtime_isa(&forbidden)?;
    }
    Ok(())
}

fn validate_runtime_isa(features: &[String]) -> Result<(), MachineClassError> {
    const REPORTABLE_RUNTIME_ISA: &[&str] = &[
        "aes", "asimd", "avx2", "avx512f", "bmi2", "fma", "neon", "scalar", "sha2", "vaes",
    ];
    let valid_token = |feature: &str| {
        !feature.is_empty()
            && feature.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'_' | b'.' | b'-')
            })
    };
    if features.is_empty()
        || features.iter().any(|feature| !valid_token(feature))
        || features
            .iter()
            .any(|feature| !REPORTABLE_RUNTIME_ISA.contains(&feature.as_str()))
        || features.windows(2).any(|pair| pair[0] >= pair[1])
        || (features.len() > 1 && features.iter().any(|feature| feature == "scalar"))
    {
        return Err(MachineClassError::new(
            MachineClassReason::HardwareIsaMismatch,
            "runtime-detected ISA must be a nonempty, strictly sorted, duplicate-free token list",
        ));
    }
    Ok(())
}

fn validate_execution(
    receipt: &RunnerReceipt,
    resolved: &ResolvedProfile<'_>,
    context: &MachineClassAdmissionContext,
) -> Result<(), MachineClassError> {
    let request = &receipt.execution.request;
    let start = &receipt.execution.start;
    let end = &receipt.execution.end;
    let profile = resolved.profile;
    let gate_policy = profile.gate_policy(&context.gate).ok_or_else(|| {
        MachineClassError::new(
            MachineClassReason::ExecutionProfileContractInvalid,
            format!("profile has no policy for {:?}", context.gate),
        )
    })?;
    if start.thermal_pressure || end.thermal_pressure {
        return Err(MachineClassError::new(
            MachineClassReason::ThermalPressure,
            "observed thermal pressure invalidates timed evidence",
        ));
    }
    if request.capacity_semantics != profile.capacity_semantics
        || request.execution_capacity == 0
        || request.requested_worker_pool_width != request.execution_capacity
        || request.max_exercised_cell_width == 0
        || request.max_exercised_cell_width > request.execution_capacity
        || profile
            .execution_capacity
            .is_some_and(|capacity| capacity != request.execution_capacity)
        || gate_policy
            .max_exercised_cell_width
            .is_some_and(|width| width != request.max_exercised_cell_width)
        || (gate_policy.max_exercised_cell_width.is_none()
            && gate_policy.default_flip_disposition
                == DefaultFlipDisposition::RequiredForDefaultFlip)
    {
        return Err(MachineClassError::new(
            MachineClassReason::ExecutionThreadBudgetInvalid,
            "receipt capacity semantics, execution capacity, worker pool, or gate maximum differs from the registered profile",
        ));
    }
    let expected_lease_id = match resolved.hardware.family.as_str() {
        "trj-zen3" => "trj-zen3-exclusive",
        "m4-macos" => "m4-macos-exclusive",
        "x86-vps-ovh" => "x86-vps-ovh-exclusive",
        _ => {
            return Err(MachineClassError::new(
                MachineClassReason::ExclusiveLeaseMissing,
                "registered hardware has no canonical host-family lease identity",
            ));
        }
    };
    if !start.exclusive_lease
        || !end.exclusive_lease
        || start.exclusive_lease_id != expected_lease_id
        || end.exclusive_lease_id != expected_lease_id
    {
        return Err(MachineClassError::new(
            MachineClassReason::ExclusiveLeaseMissing,
            format!(
                "timed evidence for {} requires canonical lease identity {expected_lease_id:?}",
                resolved.hardware.family
            ),
        ));
    }
    match profile.key.execution_profile_id {
        ExecutionProfileId::Physical64 | ExecutionProfileId::Smt2_128 => {
            let threads_per_core =
                if profile.key.execution_profile_id == ExecutionProfileId::Smt2_128 {
                    2
                } else {
                    1
                };
            let logical_width = 64 * threads_per_core;
            if start.threads_per_core != threads_per_core
                || end.threads_per_core != threads_per_core
                || start.smt_state != if threads_per_core == 2 { "on" } else { "off" }
                || end.smt_state != start.smt_state
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionSmtMismatch,
                    "Threadripper profile, SMT state, and threads-per-core disagree",
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
            if request.requested_physical_core_width != Some(64)
                || request.requested_logical_cpu_ids.len()
                    != usize::try_from(logical_width).unwrap_or(usize::MAX)
                || start.observed_logical_cpu_ids.len()
                    != usize::try_from(logical_width).unwrap_or(usize::MAX)
                || unique_string_count(&start.effective_physical_core_ids) != 64
                || request.requested_qos != "not-applicable"
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionWidthMismatch,
                    "Threadripper profile does not match its exact logical and physical execution width",
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
        }
        ExecutionProfileId::Scheduler10 => {
            if request.requested_physical_core_width.is_some()
                || !request.requested_logical_cpu_ids.is_empty()
                || request.requested_qos != "inherit-process-default"
                || !start.effective_physical_core_ids.is_empty()
                || !end.effective_physical_core_ids.is_empty()
                || start.cpu_assignment_observability == "affinity-enforced"
                || end.cpu_assignment_observability == "affinity-enforced"
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionModeMismatch,
                    "M4 scheduler profile cannot claim logical-CPU affinity, physical-core width, or P/E residency",
                ));
            }
            if start.threads_per_core != 1
                || end.threads_per_core != 1
                || start.smt_state != "not-applicable"
                || end.smt_state != "not-applicable"
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionSmtMismatch,
                    "M4 scheduler receipts cannot claim SMT",
                ));
            }
            if !["unavailable", "scheduler-observed"]
                .contains(&start.cpu_assignment_observability.as_str())
                || start.cpu_assignment_observability != end.cpu_assignment_observability
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionCpusetInvalid,
                    "M4 assignment observability is invalid, drifted, or fabricates affinity",
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
        }
        ExecutionProfileId::X86Diagnostic => {
            if request.requested_physical_core_width.is_some()
                || request.requested_qos != "not-applicable"
            {
                return Err(MachineClassError::new(
                    MachineClassReason::ExecutionModeMismatch,
                    "x86 diagnostic profile uses runtime worker capacity without topology claims",
                ));
            }
        }
        ExecutionProfileId::Scheduler14 => {
            return Err(MachineClassError::new(
                MachineClassReason::ExecutionProfileUnavailable,
                "M5 scheduler-14 has no registered real hardware identity",
            ));
        }
    }
    Ok(())
}

fn validate_admission_envelope(
    receipt: &RunnerReceipt,
    resolved: &ResolvedProfile<'_>,
    context: &MachineClassAdmissionContext,
) -> Result<(), MachineClassError> {
    if resolved.hardware.availability != MachineProfileAvailability::Registered
        || resolved.profile.availability != MachineProfileAvailability::Registered
    {
        return Err(MachineClassError::new(
            MachineClassReason::ExecutionProfileUnavailable,
            "hardware/profile identity is unavailable",
        ));
    }
    if receipt.requested_profile != receipt.derived_profile {
        return Err(MachineClassError::new(
            MachineClassReason::ReceiptClassMismatch,
            "requested, derived, and externally expected profile identities differ",
        ));
    }
    if receipt.derived_profile != context.expected_profile {
        return Err(MachineClassError::new(
            MachineClassReason::ReceiptClassMismatch,
            "requested, derived, and externally expected profile identities differ",
        ));
    }
    validate_hardware(&receipt.hardware, resolved.hardware)?;
    validate_execution(receipt, resolved, context)?;
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
    Ok(())
}

fn validate_pre_spawn_gate_profile_policy(
    resolved: &ResolvedProfile<'_>,
    context: &MachineClassAdmissionContext,
) -> Result<(), MachineClassError> {
    validate_gate_profile_policy_base(resolved, context)?;
    if matches!(context.gate.as_str(), "QG-3" | "QG-4") {
        return Err(MachineClassError::new(
            MachineClassReason::ClassUnavailable,
            format!(
                "{} is not promotion-admissible until both arms emit a non-declarative symmetric durability-treatment witness",
                context.gate
            ),
        ));
    }
    Ok(())
}

fn validate_gate_profile_policy(
    resolved: &ResolvedProfile<'_>,
    context: &MachineClassAdmissionContext,
    qg5_witness_verified: bool,
) -> Result<(), MachineClassError> {
    validate_gate_profile_policy_base(resolved, context)?;
    if matches!(context.gate.as_str(), "QG-3" | "QG-4")
        || (context.gate == "QG-5" && !qg5_witness_verified)
    {
        return Err(MachineClassError::new(
            MachineClassReason::ClassUnavailable,
            format!(
                "{} is not promotion-admissible until both arms emit a non-declarative symmetric durability-treatment witness",
                context.gate
            ),
        ));
    }
    Ok(())
}

fn validate_gate_profile_policy_base(
    resolved: &ResolvedProfile<'_>,
    context: &MachineClassAdmissionContext,
) -> Result<(), MachineClassError> {
    let policy = resolved.profile.gate_policy(&context.gate).ok_or_else(|| {
        MachineClassError::new(
            MachineClassReason::ExecutionProfileContractInvalid,
            format!("profile has no policy for {:?}", context.gate),
        )
    })?;
    if resolved.profile.availability != MachineProfileAvailability::Registered
        || resolved.hardware.availability != MachineProfileAvailability::Registered
    {
        return Err(MachineClassError::new(
            MachineClassReason::ExecutionProfileUnavailable,
            format!(
                "{}.{} is required but unavailable for {}",
                resolved.profile.key.hardware_class_id.as_str(),
                resolved.profile.key.execution_profile_id.as_str(),
                context.gate
            ),
        ));
    }
    if policy.default_flip_disposition == DefaultFlipDisposition::DiagnosticOnly {
        return Err(MachineClassError::new(
            MachineClassReason::ClassHomogeneityUnproven,
            "diagnostic-only execution profiles cannot promote history",
        ));
    }
    if resolved.profile.key.hardware_class_id != HardwareClassId::M4Macos {
        return Ok(());
    }
    Err(MachineClassError::new(
        MachineClassReason::ClassUnavailable,
        format!(
            "{} is not promotion-admissible on M4 until the producer attests the actual executing image",
            context.gate,
        ),
    ))
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
    if receipt.build.git_dirty || receipt.build.producer.source_git_dirty {
        return Err(MachineClassError::new(
            MachineClassReason::SourceDirty,
            "promotion requires clean benchmark and typed-producer source identities",
        ));
    }
    if receipt.build.worktree_state_sha256.is_some()
        || !is_git_revision(&receipt.build.git_revision)
        || !is_sha256(&receipt.build.cargo_lock_sha256)
        || !is_sha256(&receipt.build.executable_sha256)
        || !is_sha256(&receipt.build.command_sha256)
        || !is_sha256(&receipt.build.environment_sha256)
        || receipt.build.producer.contract_version != LOCAL_PERF_PRODUCER_CONTRACT_VERSION
        || !is_git_revision(&receipt.build.producer.source_git_revision)
        || receipt.build.producer.source_git_revision != receipt.build.git_revision
        || receipt.build.producer.cargo_lock_sha256 != receipt.build.cargo_lock_sha256
        || !is_sha256(&receipt.build.producer.executable_sha256)
    {
        return Err(MachineClassError::new(
            MachineClassReason::SourceIdentityInvalid,
            "benchmark and typed-producer source, Cargo.lock, executable, command, or controlled-environment identities are invalid or disagree",
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

fn validate_pre_spawn_durability(
    durability: &RunnerDurability,
    context: &MachineClassAdmissionContext,
) -> Result<(), MachineClassError> {
    if context.gate != "QG-5" {
        return validate_durability(durability);
    }
    let pending = qg5_pending_runner_durability();
    if durability != &pending {
        return Err(qg5_witness_error(
            "QG-5 pre-spawn admission requires the exact pending post-exit witness token",
        ));
    }
    Ok(())
}

fn validate_qg5_post_exit_durability(
    durability: &RunnerDurability,
    witnesses: &Qg5DurabilityWitnessSet,
    witness_bytes: &[u8],
) -> Result<(), MachineClassError> {
    witnesses.verify()?;
    let expected = qg5_post_exit_runner_durability(witness_bytes);
    if durability != &expected {
        return Err(qg5_witness_error(
            "QG-5 final receipt does not bind the exact sealed post-exit witness set",
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
    profile: MachineProfileKey,
) -> Result<(), MachineClassError> {
    let expected = profile.latest_basename(&context.gate)?;
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
pub fn admitted_test_identity_for_run(
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    environment_sha256: &str,
    run_label: &str,
) -> VerifiedRunnerIdentity {
    admitted_test_identity_from_vector_for_run(
        "MCV-001-trj-physical-64-admitted",
        gate,
        git_revision,
        cargo_lock_sha256,
        executable_sha256,
        command_sha256,
        environment_sha256,
        run_label,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn admitted_test_identity_for_artifacts(
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    environment_sha256: &str,
    run_label: &str,
    run_id: &str,
    run_window: &str,
    threshold_artifact_bytes: &[u8],
    evidence_artifact_bytes: &[u8],
) -> VerifiedRunnerIdentity {
    let bare = admitted_test_identity_for_run(
        gate,
        git_revision,
        cargo_lock_sha256,
        executable_sha256,
        command_sha256,
        environment_sha256,
        run_label,
    );
    bind_test_identity_to_artifacts(
        &bare,
        gate,
        run_id,
        run_window,
        format!("runner-log:{run_label}").as_bytes(),
        threshold_artifact_bytes,
        evidence_artifact_bytes,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn admitted_test_identity_for_artifacts_with_producer(
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    environment_sha256: &str,
    producer_executable_sha256: &str,
    run_label: &str,
    run_id: &str,
    run_window: &str,
    threshold_artifact_bytes: &[u8],
    evidence_artifact_bytes: &[u8],
) -> VerifiedRunnerIdentity {
    let bare = admitted_test_identity_for_run(
        gate,
        git_revision,
        cargo_lock_sha256,
        executable_sha256,
        command_sha256,
        environment_sha256,
        run_label,
    );
    let mut receipt =
        serde_json::from_str::<Value>(bare.receipt_json()).expect("test runner receipt JSON");
    set_path(
        &mut receipt,
        "build.producer.executable_sha256",
        Value::String(producer_executable_sha256.to_owned()),
    );
    let receipt_bytes = serde_json::to_vec(&receipt).expect("test runner receipt bytes");
    let registry = MachineClassRegistry::frozen().expect("frozen registry");
    let producer_bound = registry
        .admit(&receipt_bytes, bare.admission_context())
        .expect("producer-specific test runner receipt admission");
    bind_test_identity_to_artifacts(
        &producer_bound,
        gate,
        run_id,
        run_window,
        format!("runner-log:{run_label}").as_bytes(),
        threshold_artifact_bytes,
        evidence_artifact_bytes,
    )
}

#[cfg(test)]
fn bind_test_identity_to_artifacts(
    bare: &VerifiedRunnerIdentity,
    gate: &str,
    run_id: &str,
    run_window: &str,
    run_log_bytes: &[u8],
    threshold_artifact_bytes: &[u8],
    evidence_artifact_bytes: &[u8],
) -> VerifiedRunnerIdentity {
    let receipt_run_log_sha256 = bare
        .completion()
        .get("run_log_sha256")
        .and_then(Value::as_str)
        .expect("test receipt run-log digest");
    assert_eq!(receipt_run_log_sha256, sha256_hex(run_log_bytes));
    let manifest = RunnerArtifactManifest::from_artifacts(
        &PerfMatrixSpec::complete()
            .applicability_plan(
                &MachineClassRegistry::frozen().expect("frozen registry"),
                bare.profile(),
                gate.parse().expect("normative gate"),
            )
            .expect("test applicability plan"),
        run_id,
        run_window,
        run_log_bytes,
        threshold_artifact_bytes,
        evidence_artifact_bytes,
    )
    .expect("test artifact manifest");
    let manifest_bytes = manifest.to_json_bytes().expect("test artifact manifest");
    let mut receipt =
        serde_json::from_str::<Value>(bare.receipt_json()).expect("test runner receipt JSON");
    set_path(
        &mut receipt,
        "completion.artifact_manifest_sha256",
        Value::String(sha256_hex(&manifest_bytes)),
    );
    let receipt_bytes = serde_json::to_vec(&receipt).expect("test runner receipt bytes");
    let registry = MachineClassRegistry::frozen().expect("frozen registry");
    registry
        .admit(&receipt_bytes, bare.admission_context())
        .expect("test runner receipt admission")
        .bind_artifact_manifest(
            &manifest_bytes,
            run_log_bytes,
            threshold_artifact_bytes,
            evidence_artifact_bytes,
        )
        .expect("test artifact-manifest binding")
}

#[cfg(test)]
pub fn admitted_test_identity_from_vector_for_run(
    vector_id: &str,
    gate: &str,
    git_revision: &str,
    cargo_lock_sha256: &str,
    executable_sha256: &str,
    command_sha256: &str,
    environment_sha256: &str,
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
        "build.producer.source_git_revision",
        Value::String(git_revision.to_owned()),
    );
    set_path(
        &mut receipt,
        "build.producer.cargo_lock_sha256",
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
        "build.environment_sha256",
        Value::String(environment_sha256.to_owned()),
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
    let profile = serde_json::from_value::<MachineProfileKey>(receipt["derived_profile"].clone())
        .expect("fixture profile");
    let max_exercised_cell_width = registry
        .execution_profile(profile)
        .expect("fixture execution profile")
        .gate_policy(gate)
        .and_then(MachineProfileGatePolicy::max_exercised_cell_width)
        .expect("fixture gate maximum");
    set_path(
        &mut receipt,
        "execution.request.max_exercised_cell_width",
        Value::from(max_exercised_cell_width),
    );
    set_path(
        &mut receipt,
        "execution.identity_sha256",
        Value::String("$DERIVE_EXECUTION_IDENTITY_SHA256".to_owned()),
    );
    derive_receipt_placeholders(&mut receipt);
    context.gate = gate.to_owned();
    context.expected_profile = profile;
    context.destination_basename = profile.latest_basename(gate).expect("fixture destination");
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
            "requested_profile": value["requested_profile"],
            "derived_profile": value["derived_profile"],
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
    let context = serde_json::from_value(
        vector
            .get("admission_context")
            .expect("receipt vector admission context")
            .clone(),
    )
    .expect("admission context");
    if let Some(raw) = vector.get("raw_json").and_then(Value::as_str) {
        return (raw.as_bytes().to_vec(), context);
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
    (serde_json::to_vec(&receipt).expect("receipt JSON"), context)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use super::*;
    use crate::perf::{
        PerfMetricSemantics, PerfOperationScope, PerfSampleOrder, PerfSampleProvenance,
    };

    fn expected_reason(value: &Value) -> MachineClassReason {
        serde_json::from_value(value.clone()).expect("known expected reason")
    }

    fn qg5_test_raw_sample(
        run_id: &str,
        block_id: u64,
        sample_id: u64,
        arm: PerfSampleArm,
        order: PerfSampleOrder,
        elapsed_ns: u64,
    ) -> PerfRawSample {
        let started_ns = sample_id
            .checked_mul(20_000_000)
            .expect("test timestamp fits u64");
        PerfRawSample {
            block_id,
            sample_id,
            arm,
            order,
            phase: PerfSamplePhase::Measurement,
            scope: PerfOperationScope {
                operation_id: "qg5.compaction".to_owned(),
                version: 1,
                semantics: PerfMetricSemantics::GaugeLowerIsBetter,
                unit: "ms".to_owned(),
            },
            provenance: PerfSampleProvenance {
                run_id: run_id.to_owned(),
                executable_sha256: "a".repeat(64),
                corpus_sha256: "b".repeat(64),
                input_identity: None,
                worker_id: "qg5-test-worker".to_owned(),
                build_profile: "release".to_owned(),
            },
            started_ns,
            ended_ns: started_ns + 10_000_000,
            work_units: None,
            byte_count: None,
            observed_value: Some(Duration::from_nanos(elapsed_ns).as_secs_f64() * 1_000.0),
            group_id: None,
            qg6_sample_binding: None,
            qg1_sample_binding: None,
            tantivy_config_sha256: None,
        }
    }

    fn qg5_test_observation(
        engine: Qg5DurabilityEngine,
        elapsed_ns: u64,
    ) -> Qg5DurabilityObservation {
        let (timed_maintenance, reopened_segment_count) = match engine {
            Qg5DurabilityEngine::Quill => (
                Qg5TimedMaintenanceObservation::QuillCompaction {
                    elapsed_ns,
                    generation_before: 7,
                    generation_after: 8,
                    examined_segments: 4,
                    compacted_segments: 4,
                    removed_segments: 0,
                    dropped_documents: 20,
                    input_bytes: 10_000,
                    output_bytes: 8_000,
                    input_segment_count: 4,
                    output_segment_count: 4,
                },
                4,
            ),
            Qg5DurabilityEngine::Tantivy => (
                Qg5TimedMaintenanceObservation::TantivyForceMerge {
                    elapsed_ns,
                    input_segment_count: 4,
                    output_segment_count: 1,
                },
                1,
            ),
        };
        Qg5DurabilityObservation::new(
            Qg5DeletePublicationObservation {
                source_document_count: 100,
                requested_delete_count: 20,
                published_live_document_count: 80,
                published_segment_count: 4,
                deleted_probe_document_id: "qg5-deleted-probe".to_owned(),
                deleted_probe_match_count: 0,
                live_probe_document_id: "qg5-live-probe".to_owned(),
                live_probe_match_count: 1,
            },
            timed_maintenance,
            Qg5ReopenValidationObservation {
                reopened_live_document_count: 80,
                reopened_segment_count,
                deleted_probe_document_id: "qg5-deleted-probe".to_owned(),
                deleted_probe_match_count: 0,
                live_probe_document_id: "qg5-live-probe".to_owned(),
                live_probe_match_count: 1,
            },
        )
        .expect("valid typed QG-5 observation")
    }

    fn qg5_test_raw_streams(run_id: &str, rounds: u64) -> (Vec<PerfRawSample>, Vec<PerfRawSample>) {
        let elapsed_ns = 2_000_000;
        let mut effect = Vec::new();
        let mut oracle_null = Vec::new();
        for round in 0..rounds {
            let control_order = if round % 2 == 0 {
                PerfSampleOrder::First
            } else {
                PerfSampleOrder::Second
            };
            let treatment_order = if control_order == PerfSampleOrder::First {
                PerfSampleOrder::Second
            } else {
                PerfSampleOrder::First
            };
            let effect_control_id = round * 2;
            effect.push(qg5_test_raw_sample(
                run_id,
                round,
                effect_control_id,
                PerfSampleArm::Control,
                control_order,
                elapsed_ns,
            ));
            effect.push(qg5_test_raw_sample(
                run_id,
                round,
                effect_control_id + 1,
                PerfSampleArm::Treatment,
                treatment_order,
                elapsed_ns,
            ));
            let null_control_id = 1_000_000 + round * 2;
            oracle_null.push(qg5_test_raw_sample(
                run_id,
                round,
                null_control_id,
                PerfSampleArm::Control,
                control_order,
                elapsed_ns,
            ));
            oracle_null.push(qg5_test_raw_sample(
                run_id,
                round,
                null_control_id + 1,
                PerfSampleArm::Treatment,
                treatment_order,
                elapsed_ns,
            ));
        }
        (effect, oracle_null)
    }

    fn qg5_test_fixture_for_cells(
        run_id: &str,
        cell_ids: &[&str],
        rounds: u64,
    ) -> (Qg5DurabilityWitnessSet, Qg5ExpectedDurabilityCensus) {
        let mut witness_cells = BTreeMap::new();
        let mut expected_cells = BTreeMap::new();
        for cell in cell_ids {
            let (effect, oracle_null) = qg5_test_raw_streams(run_id, rounds);
            let mut sample_witnesses = Vec::new();
            for (stream, samples) in [
                (Qg5StreamRole::Effect, effect.as_slice()),
                (Qg5StreamRole::OracleNull, oracle_null.as_slice()),
            ] {
                for sample in samples {
                    let engine = qg5_expected_engine(stream, sample.arm);
                    sample_witnesses.push(
                        Qg5SampleDurabilityWitness::seal(
                            stream,
                            engine,
                            sample,
                            qg5_test_observation(engine, 2_000_000),
                        )
                        .expect("seal measured QG-5 sample witness"),
                    );
                }
            }
            witness_cells.insert(
                (*cell).to_owned(),
                Qg5CellDurabilityWitness::new(*cell, sample_witnesses)
                    .expect("canonical QG-5 sample census"),
            );
            expected_cells.insert(
                (*cell).to_owned(),
                qg5_expected_cell_samples(run_id, &effect, &oracle_null)
                    .expect("expected QG-5 raw-sample census"),
            );
        }
        let witnesses =
            Qg5DurabilityWitnessSet::seal(run_id, witness_cells).expect("seal QG-5 witness set");
        let expected = Qg5ExpectedDurabilityCensus {
            run_id: run_id.to_owned(),
            cells: expected_cells,
        };
        (witnesses, expected)
    }

    fn qg5_test_fixture(
        run_id: &str,
        cell: &str,
    ) -> (Qg5DurabilityWitnessSet, Qg5ExpectedDurabilityCensus) {
        qg5_test_fixture_for_cells(run_id, &[cell], 2)
    }

    fn qg5_post_exit_receipt(witness_bytes: &[u8]) -> (Vec<u8>, MachineClassAdmissionContext) {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let vector = registry.raw()["test_vectors"]
            .as_array()
            .expect("receipt vectors")
            .iter()
            .find(|vector| vector["id"].as_str() == Some("MCV-001-trj-physical-64-admitted"))
            .expect("registered Threadripper vector");
        let (bytes, mut context) = materialize_receipt_vector(registry.raw(), vector);
        let mut receipt: Value = serde_json::from_slice(&bytes).expect("materialized receipt");
        let profile =
            serde_json::from_value::<MachineProfileKey>(receipt["derived_profile"].clone())
                .expect("registered profile");
        let max_width = registry
            .execution_profile(profile)
            .expect("registered execution profile")
            .gate_policy("QG-5")
            .and_then(MachineProfileGatePolicy::max_exercised_cell_width)
            .expect("QG-5 maximum width");
        set_path(
            &mut receipt,
            "execution.request.max_exercised_cell_width",
            Value::from(max_width),
        );
        set_path(
            &mut receipt,
            "durability",
            serde_json::to_value(qg5_post_exit_runner_durability(witness_bytes))
                .expect("serialize QG-5 durability binding"),
        );
        set_path(
            &mut receipt,
            "execution.identity_sha256",
            Value::String("$DERIVE_EXECUTION_IDENTITY_SHA256".to_owned()),
        );
        derive_receipt_placeholders(&mut receipt);
        context.gate = "QG-5".to_owned();
        context.expected_profile = profile;
        context.destination_basename = profile.latest_basename("QG-5").expect("QG-5 destination");
        (
            serde_json::to_vec(&receipt).expect("serialize QG-5 receipt"),
            context,
        )
    }

    #[test]
    fn frozen_registry_binds_exact_reviewed_bytes_and_corpus_counts() {
        assert_eq!(sha256_hex(REGISTRY_BYTES), MACHINE_CLASS_REGISTRY_SHA256);
        assert_eq!(
            MACHINE_CLASS_REGISTRY_GIT_BLOB,
            "fe68e97c8e66accd0abaa9a4e3146134c271e964"
        );
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        assert_eq!(
            registry.raw()["hardware_classes"].as_array().unwrap().len(),
            4
        );
        assert_eq!(
            registry.raw()["execution_profiles"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            registry.raw()["class_lookup_vectors"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(registry.raw()["test_vectors"].as_array().unwrap().len(), 28);
        assert_eq!(
            registry.raw()["registry_test_vectors"]
                .as_array()
                .unwrap()
                .len(),
            14
        );
        assert_eq!(
            registry.raw()["fact_templates"].as_object().unwrap().len(),
            2
        );
    }

    #[test]
    fn registry_rejects_artifact_manifest_and_producer_requirement_drift() {
        let mutate_and_reject = |requirement_id: &str, field: &str, replacement: Value| {
            let mut raw = parse_strict_json(REGISTRY_BYTES).expect("reviewed registry JSON");
            let requirement = raw["requirements"]
                .as_array_mut()
                .expect("requirements array")
                .iter_mut()
                .find(|requirement| {
                    requirement.get("id").and_then(Value::as_str) == Some(requirement_id)
                })
                .expect("reviewed requirement");
            requirement[field] = replacement;
            let bytes = serde_json::to_vec(&raw).expect("candidate registry");
            let error = MachineClassRegistry::load_candidate(&bytes, embedded_source)
                .expect_err("binding requirement drift must reject");
            assert_eq!(
                error.reason,
                MachineClassReason::SourceIdentityInvalid,
                "{requirement_id}.{field} drift rejected for wrong reason: {error}"
            );
        };

        for requirement_id in ["MC-MUST-021", "MC-MUST-022"] {
            mutate_and_reject(
                requirement_id,
                "id",
                Value::String(format!("{requirement_id}-renamed")),
            );
            mutate_and_reject(
                requirement_id,
                "text",
                Value::String("weakened binding requirement".to_owned()),
            );

            let mut raw = parse_strict_json(REGISTRY_BYTES).expect("reviewed registry JSON");
            raw["requirements"]
                .as_array_mut()
                .expect("requirements array")
                .retain(|requirement| {
                    requirement.get("id").and_then(Value::as_str) != Some(requirement_id)
                });
            let bytes = serde_json::to_vec(&raw).expect("candidate registry");
            let error = MachineClassRegistry::load_candidate(&bytes, embedded_source)
                .expect_err("missing binding requirement must reject");
            assert_eq!(
                error.reason,
                MachineClassReason::SourceIdentityInvalid,
                "missing {requirement_id} rejected for wrong reason: {error}"
            );
        }
    }

    #[test]
    fn frozen_execution_profiles_separate_hardware_capacity_and_release_disposition() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");

        let diagnostic_key = MachineProfileKey::new(
            HardwareClassId::X86VpsOvh,
            ExecutionProfileId::X86Diagnostic,
        )
        .expect("canonical x86 diagnostic key");
        let diagnostic = registry
            .execution_profile(diagnostic_key)
            .expect("x86 diagnostic profile");
        assert_eq!(
            diagnostic.capacity_semantics(),
            ExecutionCapacitySemantics::DiagnosticWorkerBudget
        );
        assert_eq!(diagnostic.execution_capacity(), None);
        assert_eq!(
            diagnostic
                .gate_policy("QG-1")
                .expect("diagnostic QG-1 policy")
                .default_flip_disposition(),
            DefaultFlipDisposition::DiagnosticOnly
        );
        assert_eq!(
            diagnostic
                .gate_policy("QG-1")
                .expect("diagnostic QG-1 policy")
                .max_exercised_cell_width(),
            None
        );

        let physical_key = MachineProfileKey::new(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        )
        .expect("canonical Threadripper physical key");
        let physical = registry
            .execution_profile(physical_key)
            .expect("Threadripper physical profile");
        assert_eq!(
            physical.capacity_semantics(),
            ExecutionCapacitySemantics::PhysicalCores
        );
        assert_eq!(physical.execution_capacity(), Some(64));
        assert_eq!(
            physical
                .gate_policy("QG-1")
                .expect("physical QG-1 policy")
                .max_exercised_cell_width(),
            Some(64)
        );
        assert_eq!(
            physical_key
                .latest_basename("QG-1")
                .expect("physical destination"),
            "QG-1.trj-zen3-5995wx.physical-64.latest.json"
        );

        let smt_key =
            MachineProfileKey::new(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128)
                .expect("canonical Threadripper SMT key");
        let smt = registry
            .execution_profile(smt_key)
            .expect("Threadripper SMT profile");
        assert_eq!(
            smt.capacity_semantics(),
            ExecutionCapacitySemantics::LogicalThreads
        );
        assert_eq!(smt.execution_capacity(), Some(128));
        assert_eq!(
            smt.gate_policy("QG-1")
                .expect("SMT QG-1 policy")
                .max_exercised_cell_width(),
            Some(128)
        );

        let m4_key =
            MachineProfileKey::new(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10)
                .expect("canonical M4 scheduler key");
        let m4 = registry
            .execution_profile(m4_key)
            .expect("M4 scheduler profile");
        assert_eq!(
            m4.capacity_semantics(),
            ExecutionCapacitySemantics::SchedulerWorkers
        );
        assert_eq!(m4.execution_capacity(), Some(10));
        assert_eq!(
            m4.gate_policy("QG-1")
                .expect("M4 QG-1 policy")
                .max_exercised_cell_width(),
            Some(8)
        );
        assert!(
            m4.forbidden_claims()
                .iter()
                .any(|claim| claim == "p-core-residency")
        );
        assert_eq!(
            m4_key.latest_basename("QG-1").expect("M4 destination"),
            "QG-1.m4-macos.scheduler-10.latest.json"
        );

        let m5_key =
            MachineProfileKey::new(HardwareClassId::M5Macos, ExecutionProfileId::Scheduler14)
                .expect("canonical M5 scheduler key");
        let m5 = registry
            .execution_profile(m5_key)
            .expect("typed unavailable M5 profile");
        assert_eq!(m5.availability(), MachineProfileAvailability::Unavailable);
        assert_eq!(m5.execution_capacity(), None);
        assert_eq!(
            m5.gate_policy("QG-1")
                .expect("M5 QG-1 policy")
                .default_flip_disposition(),
            DefaultFlipDisposition::RequiredForDefaultFlip
        );
        assert_eq!(
            m5.gate_policy("QG-1")
                .expect("M5 QG-1 policy")
                .max_exercised_cell_width(),
            None
        );
        assert_eq!(
            m5_key
                .latest_basename("QG-1")
                .expect("typed M5 destination"),
            "QG-1.m5-macos.scheduler-14.latest.json"
        );

        let profile_hashes = registry
            .execution_profiles
            .iter()
            .map(|profile| profile.contract_sha256())
            .collect::<BTreeSet<_>>();
        assert_eq!(profile_hashes.len(), registry.execution_profiles.len());
    }

    #[test]
    fn every_profile_and_gate_has_a_unique_latest_destination() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        assert_eq!(
            registry.execution_profiles.len(),
            5,
            "the frozen registry proof must cover exactly five execution profiles"
        );
        let mut destinations = BTreeSet::new();
        let mut normalized_destinations = BTreeSet::new();
        for profile in &registry.execution_profiles {
            for gate in 1..=10 {
                let gate = format!("QG-{gate}");
                let destination = profile
                    .key()
                    .latest_basename(&gate)
                    .expect("registered profile and normative gate destination");
                assert!(
                    destinations.insert(destination.clone()),
                    "profile-qualified destination collision: {destination}"
                );
                assert!(
                    destination.is_ascii(),
                    "latest destinations must remain ASCII: {destination}"
                );
                assert!(
                    normalized_destinations.insert(destination.to_ascii_lowercase()),
                    "ASCII-normalized destination collision: {destination}"
                );
            }
        }
        assert_eq!(
            destinations.len(),
            50,
            "five profiles times ten normative gates must produce fifty destinations"
        );
        assert_eq!(
            normalized_destinations.len(),
            50,
            "all fifty destinations must also remain unique after ASCII normalization"
        );
    }

    #[test]
    fn execution_profile_resolution_rejects_unknown_and_cross_class_substitution() {
        assert_eq!(
            MachineProfileKey::new(HardwareClassId::M5Macos, ExecutionProfileId::Scheduler10)
                .expect_err("M4 profile cannot relabel M5")
                .reason,
            MachineClassReason::ExecutionProfileClassMismatch
        );
        assert_eq!(
            MachineProfileKey::new(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler14)
                .expect_err("M5 profile cannot relabel M4")
                .reason,
            MachineClassReason::ExecutionProfileClassMismatch
        );
        let unknown = br#"{
            "hardware_class_id":"m4-macos",
            "execution_profile_id":"scheduler-128"
        }"#;
        assert!(
            serde_json::from_slice::<MachineProfileKey>(unknown).is_err(),
            "unknown profile spelling must fail typed deserialization"
        );
    }

    #[test]
    fn registry_rejects_profile_capacity_shrink_fabrication_and_all_unavailable_relabeling() {
        let mutate_and_reject = |path: &str, replacement: Value| {
            let mut raw = MachineClassRegistry::frozen()
                .expect("frozen registry")
                .raw()
                .clone();
            set_path(&mut raw, path, replacement);
            let bytes = serde_json::to_vec(&raw).expect("candidate registry");
            let error = MachineClassRegistry::load_candidate(&bytes, embedded_source)
                .expect_err("profile mutation must reject");
            assert_eq!(
                error.reason,
                MachineClassReason::ExecutionProfileContractInvalid,
                "mutation at {path} rejected for wrong reason: {error}"
            );
        };

        mutate_and_reject("execution_profiles.3.execution_capacity", Value::from(128));
        mutate_and_reject("execution_profiles.1.execution_capacity", Value::from(32));
        mutate_and_reject("execution_profiles.4.execution_capacity", Value::from(14));
        mutate_and_reject(
            "execution_profiles.4.gate_policies.QG-1.max_exercised_cell_width",
            Value::from(8),
        );
        for gate in 1..=10 {
            mutate_and_reject(
                &format!("execution_profiles.3.gate_policies.QG-{gate}.default_flip_disposition"),
                Value::String("diagnostic_only".to_owned()),
            );
        }
    }

    #[test]
    fn registry_requires_explicit_nullable_capacity_fields() {
        for path in [
            "execution_profiles.0.execution_capacity",
            "execution_profiles.4.execution_capacity",
            "execution_profiles.0.gate_policies.QG-1.max_exercised_cell_width",
            "execution_profiles.4.gate_policies.QG-1.max_exercised_cell_width",
        ] {
            let mut raw = MachineClassRegistry::frozen()
                .expect("frozen registry")
                .raw()
                .clone();
            remove_path(&mut raw, path);
            let bytes = serde_json::to_vec(&raw).expect("candidate registry");
            let error = MachineClassRegistry::load_candidate(&bytes, embedded_source)
                .expect_err("omitted nullable capacity field must reject");
            assert_eq!(
                error.reason,
                MachineClassReason::MissingField,
                "omission at {path} rejected for wrong reason: {error}"
            );
        }
    }

    #[test]
    fn canonical_machine_identity_sorts_every_object_independent_of_insertion_order() {
        let mut nested_forward = serde_json::Map::new();
        nested_forward.insert("d".to_owned(), Value::from(4));
        nested_forward.insert("c".to_owned(), Value::from(3));
        let mut forward = serde_json::Map::new();
        forward.insert("b".to_owned(), Value::from(2));
        forward.insert("a".to_owned(), Value::Object(nested_forward));

        let mut nested_reverse = serde_json::Map::new();
        nested_reverse.insert("c".to_owned(), Value::from(3));
        nested_reverse.insert("d".to_owned(), Value::from(4));
        let mut reverse = serde_json::Map::new();
        reverse.insert("a".to_owned(), Value::Object(nested_reverse));
        reverse.insert("b".to_owned(), Value::from(2));

        let expected = br#"{"a":{"c":3,"d":4},"b":2}"#;
        assert_eq!(
            canonical_json_bytes(&Value::Object(forward)).expect("forward encoding"),
            expected
        );
        assert_eq!(
            canonical_json_bytes(&Value::Object(reverse)).expect("reverse encoding"),
            expected
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
                vector["requested_hardware_class_id"]
                    .as_str()
                    .expect("requested hardware class"),
            );
            assert_eq!(
                serde_json::to_value(lookup.decision).unwrap(),
                vector["expected"]["decision"],
                "{} decision",
                vector["id"]
            );
            assert_eq!(
                serde_json::to_value(lookup.hardware_class_id).unwrap(),
                vector["expected"]["hardware_class_id"],
                "{} hardware class",
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
                    let expected_profile =
                        serde_json::from_value::<MachineProfileKey>(expected["profile"].clone())
                            .expect("expected admitted profile");
                    assert_eq!(
                        identity.profile(),
                        expected_profile,
                        "{} profile",
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
        assert_eq!(allow_count, 1);
        assert_eq!(reject_count, 27);
    }

    #[test]
    fn registry_preflight_rejects_hardware_before_spawn_and_binds_final_identity() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let vector = registry.raw()["test_vectors"]
            .as_array()
            .expect("receipt vectors")
            .iter()
            .find(|vector| vector["id"].as_str() == Some("MCV-001-trj-physical-64-admitted"))
            .expect("registered TRJ vector");
        let (bytes, context) = materialize_receipt_vector(registry.raw(), vector);
        let receipt =
            serde_json::from_slice::<RunnerReceipt>(&bytes).expect("materialized runner receipt");
        let preflight = registry
            .preflight(
                receipt.requested_profile,
                receipt.hardware.clone(),
                receipt.execution.request.clone(),
                receipt.execution.start.clone(),
                receipt.durability.clone(),
                &context,
            )
            .expect("registered pre-spawn envelope");
        let final_identity = registry
            .admit(&bytes, &context)
            .expect("terminal identity admission");
        preflight
            .verify_final(&final_identity)
            .expect("pre-spawn and terminal identities agree");

        let mut wrong_hardware = receipt.hardware;
        wrong_hardware.cpu_vendor = "GenuineIntel".to_owned();
        let error = registry
            .preflight(
                receipt.requested_profile,
                wrong_hardware,
                receipt.execution.request,
                receipt.execution.start,
                receipt.durability,
                &context,
            )
            .expect_err("wrong hardware must reject before a child can spawn");
        assert_eq!(error.reason, MachineClassReason::HardwareCpuVendorMismatch);
    }

    #[test]
    fn qg5_schema_v2_admits_exact_eight_row_census_for_two_rounds() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let run_id = "qg5-post-exit-test";
        let cell = "QG-5/compaction/xlarge/20pct/wall_clock_ms";
        let (witnesses, expected) = qg5_test_fixture(run_id, cell);
        assert_eq!(
            witnesses.cells[cell].samples.len(),
            8,
            "two rounds retain effect control/treatment plus oracle-null control/treatment"
        );
        let witness_bytes = witnesses.to_json_bytes().expect("canonical QG-5 witnesses");
        let (receipt_bytes, context) = qg5_post_exit_receipt(&witness_bytes);
        let receipt =
            serde_json::from_slice::<RunnerReceipt>(&receipt_bytes).expect("QG-5 runner receipt");

        let pre_spawn = registry
            .preflight(
                receipt.requested_profile,
                receipt.hardware.clone(),
                receipt.execution.request.clone(),
                receipt.execution.start.clone(),
                qg5_pending_runner_durability(),
                &context,
            )
            .expect("QG-5 pre-spawn envelope may only defer witness admission");
        let ordinary_error = registry
            .admit(&receipt_bytes, &context)
            .expect_err("QG-5 must not admit before parent verifies child witnesses");
        assert_eq!(ordinary_error.reason, MachineClassReason::ClassUnavailable);

        let identity = registry
            .admit_qg5_post_exit(
                &receipt_bytes,
                &context,
                run_id,
                &[cell.to_owned()],
                &witness_bytes,
                &expected,
            )
            .expect("matching observed Quill/Tantivy QG-5 witnesses admit");
        pre_spawn
            .verify_final(&identity)
            .expect("post-exit QG-5 admission retains the pre-spawn host envelope");
        identity
            .verify()
            .expect("stored QG-5 identity re-verifies its sealed witnesses");
    }

    #[test]
    fn qg5_parent_rejects_missing_duplicate_and_stream_swapped_exact_keys() {
        let run_id = "qg5-census-hostiles";
        let cell = "QG-5/compaction/xlarge/20pct/wall_clock_ms";
        let selected = [cell.to_owned()];
        let (witnesses, expected) = qg5_test_fixture(run_id, cell);

        let mut missing_effect_treatment = witnesses.clone();
        missing_effect_treatment
            .cells
            .get_mut(cell)
            .expect("fixture cell")
            .samples
            .retain(|sample| {
                !(sample.stream == Qg5StreamRole::Effect
                    && sample.block_id == 0
                    && sample.sample_id == 1)
            });
        missing_effect_treatment
            .refresh_seal()
            .expect("reseal missing-row hostile");
        assert!(
            missing_effect_treatment
                .verify_for_run_and_census(run_id, &selected, &expected)
                .is_err(),
            "a missing effect-treatment row must reject"
        );

        let mut duplicate_exact_key = witnesses.clone();
        let duplicate = duplicate_exact_key.cells[cell].samples[0].clone();
        let duplicate_cell = duplicate_exact_key
            .cells
            .get_mut(cell)
            .expect("fixture cell");
        duplicate_cell.samples.push(duplicate);
        duplicate_cell
            .samples
            .sort_by_key(Qg5SampleDurabilityWitness::key);
        duplicate_exact_key
            .refresh_seal()
            .expect("reseal duplicate-key hostile");
        assert!(
            duplicate_exact_key
                .verify_for_run_and_census(run_id, &selected, &expected)
                .is_err(),
            "a duplicate exact stream/block/sample key must reject"
        );

        let mut swapped_controls = witnesses;
        let swapped_cell = swapped_controls.cells.get_mut(cell).expect("fixture cell");
        let effect_control = swapped_cell
            .samples
            .iter()
            .position(|sample| {
                sample.stream == Qg5StreamRole::Effect
                    && sample.block_id == 0
                    && sample.sample_id == 0
            })
            .expect("effect control row");
        let oracle_null_control = swapped_cell
            .samples
            .iter()
            .position(|sample| {
                sample.stream == Qg5StreamRole::OracleNull
                    && sample.block_id == 0
                    && sample.sample_id == 1_000_000
            })
            .expect("oracle-null control row");
        swapped_cell.samples[effect_control].stream = Qg5StreamRole::OracleNull;
        swapped_cell.samples[oracle_null_control].stream = Qg5StreamRole::Effect;
        swapped_cell
            .samples
            .sort_by_key(Qg5SampleDurabilityWitness::key);
        swapped_controls
            .refresh_seal()
            .expect("reseal stream-swap hostile");
        assert!(
            swapped_controls
                .verify_for_run_and_census(run_id, &selected, &expected)
                .is_err(),
            "Effect/Control/Tantivy cannot swap with OracleNull/Control/Tantivy"
        );
    }

    #[test]
    fn qg5_parent_rejects_warmup_rows_and_exact_latency_bit_drift() {
        let run_id = "qg5-latency-hostiles";
        let cell = "QG-5/compaction/xlarge/20pct/wall_clock_ms";
        let selected = [cell.to_owned()];
        let (mut effect, oracle_null) = qg5_test_raw_streams(run_id, 2);
        effect[0].phase = PerfSamplePhase::Warmup;
        assert!(
            qg5_expected_cell_samples(run_id, &effect, &oracle_null).is_err(),
            "warmup rows must never enter the expected durability census"
        );

        let (mut witnesses, expected) = qg5_test_fixture(run_id, cell);
        let sample = witnesses
            .cells
            .get_mut(cell)
            .expect("fixture cell")
            .samples
            .first_mut()
            .expect("fixture sample");
        match &mut sample.observation.timed_maintenance {
            Qg5TimedMaintenanceObservation::QuillCompaction { elapsed_ns, .. }
            | Qg5TimedMaintenanceObservation::TantivyForceMerge { elapsed_ns, .. } => {
                *elapsed_ns += 1;
            }
        }
        witnesses
            .refresh_seal()
            .expect("reseal latency-bit hostile");
        assert!(
            witnesses
                .verify_for_run_and_census(run_id, &selected, &expected)
                .is_err(),
            "typed elapsed latency must equal the exact parent raw-sample f64 bits"
        );
    }

    #[test]
    fn qg5_typed_maintenance_accepts_quill_tombstone_fold_same_count_only() {
        let same_count = qg5_test_observation(Qg5DurabilityEngine::Quill, 2_000_000);
        same_count
            .validate(Qg5DurabilityEngine::Quill)
            .expect("Quill may rewrite tombstones without removing a segment");

        let mut false_quill_topology = same_count.clone();
        let Qg5TimedMaintenanceObservation::QuillCompaction {
            output_segment_count,
            ..
        } = &mut false_quill_topology.timed_maintenance
        else {
            panic!("Quill fixture must carry a compaction report");
        };
        *output_segment_count -= 1;
        assert!(
            false_quill_topology
                .validate(Qg5DurabilityEngine::Quill)
                .is_err(),
            "Quill output count must equal input count minus removed segments"
        );

        let mut unchanged_generation = same_count.clone();
        let Qg5TimedMaintenanceObservation::QuillCompaction {
            generation_before,
            generation_after,
            ..
        } = &mut unchanged_generation.timed_maintenance
        else {
            panic!("Quill fixture must carry a compaction report");
        };
        *generation_after = *generation_before;
        assert!(
            unchanged_generation
                .validate(Qg5DurabilityEngine::Quill)
                .is_err(),
            "changed Quill compaction must advance generation"
        );

        let mut partial_quill_work = same_count.clone();
        let Qg5TimedMaintenanceObservation::QuillCompaction {
            compacted_segments, ..
        } = &mut partial_quill_work.timed_maintenance
        else {
            panic!("Quill fixture must carry a compaction report");
        };
        *compacted_segments -= 1;
        assert!(
            partial_quill_work
                .validate(Qg5DurabilityEngine::Quill)
                .is_err(),
            "Quill must compact every examined segment before comparison with Tantivy force-merge"
        );

        let mut unchanged_tantivy = qg5_test_observation(Qg5DurabilityEngine::Tantivy, 2_000_000);
        let Qg5TimedMaintenanceObservation::TantivyForceMerge {
            input_segment_count,
            output_segment_count,
            ..
        } = &mut unchanged_tantivy.timed_maintenance
        else {
            panic!("Tantivy fixture must carry a force-merge observation");
        };
        *output_segment_count = *input_segment_count;
        assert!(
            unchanged_tantivy
                .validate(Qg5DurabilityEngine::Tantivy)
                .is_err(),
            "Tantivy force merge must strictly reduce segment count"
        );
    }

    #[test]
    fn registered_classes_reject_forged_family_lease_identities() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        for vector_id in [
            "MCV-001-trj-physical-64-admitted",
            "MCV-002-m4-scheduler-10-runtime-unavailable",
        ] {
            let vector = registry.raw()["test_vectors"]
                .as_array()
                .expect("receipt vectors")
                .iter()
                .find(|vector| vector["id"].as_str() == Some(vector_id))
                .expect("registered vector");
            let (bytes, context) = materialize_receipt_vector(registry.raw(), vector);
            let mut receipt =
                serde_json::from_slice::<Value>(&bytes).expect("materialized receipt");
            for side in ["start", "end"] {
                set_path(
                    &mut receipt,
                    &format!("execution.{side}.exclusive_lease_id"),
                    Value::String("operator-selected-lease".to_owned()),
                );
                set_path(
                    &mut receipt,
                    &format!("execution.{side}.snapshot_sha256"),
                    Value::String(format!(
                        "$DERIVE_{}_SNAPSHOT_SHA256",
                        side.to_ascii_uppercase()
                    )),
                );
            }
            set_path(
                &mut receipt,
                "execution.identity_sha256",
                Value::String("$DERIVE_EXECUTION_IDENTITY_SHA256".to_owned()),
            );
            derive_receipt_placeholders(&mut receipt);
            let forged = serde_json::to_vec(&receipt).expect("forged receipt JSON");
            let error = registry
                .admit(&forged, &context)
                .expect_err("noncanonical family lease must reject");
            assert_eq!(
                error.reason,
                MachineClassReason::ExclusiveLeaseMissing,
                "{vector_id}"
            );
        }
    }

    #[test]
    fn receipt_admission_rejects_stale_or_substituted_producer_identity_without_writes() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let vector = registry.raw()["test_vectors"]
            .as_array()
            .expect("receipt vectors")
            .iter()
            .find(|vector| vector["id"].as_str() == Some("MCV-001-trj-physical-64-admitted"))
            .expect("registered vector");
        let cases = [
            (
                "build.producer.contract_version",
                Value::String("frankensearch.quill-local-perf-producer.v2".to_owned()),
                MachineClassReason::SourceIdentityInvalid,
            ),
            (
                "build.producer.source_git_revision",
                Value::String("a".repeat(40)),
                MachineClassReason::SourceIdentityInvalid,
            ),
            (
                "build.producer.source_git_dirty",
                Value::Bool(true),
                MachineClassReason::SourceDirty,
            ),
            (
                "build.producer.cargo_lock_sha256",
                Value::String("b".repeat(64)),
                MachineClassReason::SourceIdentityInvalid,
            ),
            (
                "build.producer.executable_sha256",
                Value::String("arbitrary-executable".to_owned()),
                MachineClassReason::SourceIdentityInvalid,
            ),
        ];
        for (path, replacement, expected_reason) in cases {
            let (bytes, context) = materialize_receipt_vector(registry.raw(), vector);
            let mut receipt =
                serde_json::from_slice::<Value>(&bytes).expect("materialized receipt");
            set_path(&mut receipt, path, replacement);
            let forged = serde_json::to_vec(&receipt).expect("forged receipt JSON");
            let write_count = Cell::new(0_u64);
            let error = registry
                .admit_then(&forged, &context, |_| {
                    write_count.set(write_count.get() + 1);
                })
                .expect_err("forged producer identity must reject");
            assert_eq!(error.reason, expected_reason, "{path}");
            assert_eq!(write_count.get(), 0, "{path} wrote on denial");
        }

        let (bytes, context) = materialize_receipt_vector(registry.raw(), vector);
        let mut receipt = serde_json::from_slice::<Value>(&bytes).expect("materialized receipt");
        remove_path(&mut receipt, "build.producer");
        let missing = serde_json::to_vec(&receipt).expect("receipt without producer");
        let error = registry
            .admit(&missing, &context)
            .expect_err("missing producer identity must reject");
        assert_eq!(error.reason, MachineClassReason::MissingField);
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

    #[test]
    fn artifact_manifest_requires_exact_compact_typed_bytes() {
        let profile = MachineProfileKey::new(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        )
        .expect("canonical test profile");
        let manifest = RunnerArtifactManifest::from_artifacts(
            &PerfMatrixSpec::complete()
                .applicability_plan(
                    &MachineClassRegistry::frozen().expect("frozen registry"),
                    profile,
                    "QG-2".parse().expect("normative gate"),
                )
                .expect("test applicability plan"),
            "candidate-a",
            "window-a",
            b"exact run log",
            b"exact threshold",
            b"exact pre-binding evidence",
        )
        .expect("test artifact manifest");
        let canonical = manifest.to_json_bytes().expect("canonical manifest");
        parse_artifact_manifest_binding(&canonical).expect("canonical manifest admission");

        let pretty = serde_json::to_vec_pretty(&manifest).expect("pretty manifest");
        let mut trailing_lf = canonical.clone();
        trailing_lf.push(b'\n');
        let canonical_text = std::str::from_utf8(&canonical).expect("manifest UTF-8");
        let duplicate = canonical_text.replacen("\"gate\":", "\"gate\":\"QG-2\",\"gate\":", 1);
        let unknown = canonical_text.replacen('{', "{\"unreviewed\":true,", 1);
        let mut missing = serde_json::to_value(&manifest).expect("manifest value");
        missing
            .as_object_mut()
            .expect("manifest object")
            .remove("run_window");
        let missing = serde_json::to_vec(&missing).expect("missing-field manifest");

        for rejected in [
            pretty.as_slice(),
            trailing_lf.as_slice(),
            duplicate.as_bytes(),
            unknown.as_bytes(),
            missing.as_slice(),
        ] {
            assert!(
                parse_artifact_manifest_binding(rejected).is_err(),
                "noncanonical or structurally invalid manifest was admitted"
            );
        }
    }

    #[test]
    fn artifact_manifest_rejects_plan_and_capacity_substitution_after_receipt_rehash() {
        let gate = "QG-2";
        let run_label = "manifest-envelope";
        let run_log = b"runner-log:manifest-envelope";
        let threshold = b"exact threshold";
        let evidence = b"exact pre-binding evidence";
        let bare = admitted_test_identity_for_run(
            gate,
            &"d".repeat(40),
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"2".repeat(64),
            run_label,
        );
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let plan = PerfMatrixSpec::complete()
            .applicability_plan(
                &registry,
                bare.profile(),
                gate.parse().expect("normative gate"),
            )
            .expect("test applicability plan");
        let original = RunnerArtifactManifest::from_artifacts(
            &plan,
            "candidate-a",
            "window-a",
            run_log,
            threshold,
            evidence,
        )
        .expect("test artifact manifest");
        assert_eq!(original.profile(), plan.binding.profile);
        assert_eq!(original.capacity_semantics(), plan.capacity_semantics);
        assert_eq!(Some(original.execution_capacity()), plan.execution_capacity);
        assert_eq!(
            Some(original.max_exercised_cell_width()),
            plan.max_exercised_cell_width
        );
        assert_eq!(original.applicability_plan(), plan.binding());

        let mut manifest = original.clone();
        manifest.gate = "QG-1".to_owned();
        let mut mutations = vec![("gate", manifest)];
        let mut manifest = original.clone();
        manifest.profile =
            MachineProfileKey::new(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128)
                .expect("alternate profile");
        mutations.push(("profile", manifest));
        let mut manifest = original.clone();
        manifest.capacity_semantics = ExecutionCapacitySemantics::LogicalThreads;
        mutations.push(("capacity_semantics", manifest));
        let mut manifest = original.clone();
        manifest.execution_capacity += 1;
        mutations.push(("execution_capacity", manifest));
        let mut manifest = original.clone();
        manifest.max_exercised_cell_width += 1;
        mutations.push(("max_exercised_cell_width", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.profile =
            MachineProfileKey::new(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128)
                .expect("alternate plan profile");
        mutations.push(("applicability_plan.profile", manifest));
        let mut manifest = original.clone();
        manifest
            .applicability_plan
            .registry_schema_version
            .push_str("-stale");
        mutations.push(("applicability_plan.registry_schema_version", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.registry_sha256 = "0".repeat(64);
        mutations.push(("applicability_plan.registry_sha256", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.profile_contract_sha256 = "0".repeat(64);
        mutations.push(("applicability_plan.profile_contract_sha256", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.gate = "QG-1".parse().expect("normative gate");
        mutations.push(("applicability_plan.gate", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.normalized_perf_manifest_sha256 = "0".repeat(64);
        mutations.push((
            "applicability_plan.normalized_perf_manifest_sha256",
            manifest,
        ));
        let mut manifest = original.clone();
        manifest.applicability_plan.primary_target_cell_width = Some(1);
        mutations.push(("applicability_plan.primary_target_cell_width", manifest));
        let mut manifest = original.clone();
        manifest
            .applicability_plan
            .matrix_contract_schema_version
            .push_str("-stale");
        mutations.push((
            "applicability_plan.matrix_contract_schema_version",
            manifest,
        ));
        let mut manifest = original.clone();
        manifest.applicability_plan.gate_matrix_contract_sha256 = "0".repeat(64);
        mutations.push(("applicability_plan.gate_matrix_contract_sha256", manifest));
        let mut manifest = original.clone();
        manifest.applicability_plan.applicability_plan_sha256 = "0".repeat(64);
        mutations.push(("applicability_plan.applicability_plan_sha256", manifest));

        for (field, manifest) in mutations {
            let manifest_bytes = manifest.to_json_bytes().expect("mutated manifest bytes");
            let mut receipt = serde_json::from_str::<Value>(bare.receipt_json())
                .expect("test runner receipt JSON");
            set_path(
                &mut receipt,
                "completion.artifact_manifest_sha256",
                Value::String(sha256_hex(&manifest_bytes)),
            );
            let receipt_bytes = serde_json::to_vec(&receipt).expect("test runner receipt bytes");
            let identity = registry
                .admit(&receipt_bytes, bare.admission_context())
                .expect("receipt with rehashed manifest");
            assert!(
                identity
                    .bind_artifact_manifest(&manifest_bytes, run_log, threshold, evidence,)
                    .is_err(),
                "manifest mutation {field} survived strict binding"
            );
        }
    }

    #[test]
    fn bound_manifest_rejects_actual_artifact_tamper() {
        let threshold = b"exact threshold";
        let evidence = b"exact pre-binding evidence";
        let identity = admitted_test_identity_for_artifacts(
            "QG-2",
            &"d".repeat(40),
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"2".repeat(64),
            "manifest-tamper",
            "candidate-a",
            "window-a",
            threshold,
            evidence,
        );

        identity
            .verify_artifact_inputs(b"runner-log:manifest-tamper", threshold, evidence)
            .expect("exact artifact inputs");
        assert!(
            identity
                .verify_run_log(b"runner-log:manifest-tampeR")
                .is_err()
        );
        assert!(
            identity
                .verify_threshold_artifact(b"exact thresholD")
                .is_err()
        );
        assert!(
            identity
                .verify_evidence_artifact(b"exact pre-binding evidencE")
                .is_err()
        );
    }
}
