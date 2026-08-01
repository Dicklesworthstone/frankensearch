//! Machine-readable performance-matrix contracts for Quill QG-1 through QG-10.
//!
//! The Criterion entry point owns engine execution. This module owns the
//! deterministic matrix, statistics, artifact schema, RSS probe, and human
//! rendering so the evidence format is unit-tested without running a benchmark.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::GauntletError;
use crate::machine_class_registry::{
    DefaultFlipDisposition, ExecutionCapacitySemantics, MACHINE_CLASS_REGISTRY_SCHEMA_VERSION,
    MACHINE_CLASS_REGISTRY_SHA256, MachineClassError, MachineClassRegistry,
    MachineExecutionProfile, MachineProfileAvailability, MachineProfileKey,
};

/// Version of the JSON emitted by the QG matrix harness.
pub const PERF_ARTIFACT_SCHEMA_VERSION: &str = "quill-perf-artifact-v7";
/// Read-only schema identifier for historical gate artifacts that lack
/// auditable host topology and effective-thread provenance.
pub const LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3: &str = "quill-perf-artifact-v3";
/// Minimum independent samples required by the standing statistical law.
pub const PERF_MIN_RUNS: usize = 10;
/// Deterministic bootstrap resamples used for the 95% confidence interval on
/// each sample median.
pub const PERF_BOOTSTRAP_RESAMPLES: usize = 2_000;
/// Required margin between a claimed paired effect and its same-invocation
/// A/A null floor.
pub const PERF_NULL_MARGIN_MULTIPLIER: f64 = 2.0;
/// Legacy display reference retained for artifact consumers.
///
/// `cv_pct` is provenance only. Neither the harness nor the ratchet uses this
/// value as an admission threshold.
pub const PERF_MAX_CV_PCT: f64 = 5.0;
/// Oracle writer heap pinned for all same-binary comparisons (50 MiB).
pub const PERF_WRITER_HEAP_BYTES: usize = 50_000_000;
/// Tantivy's pinned minimum arena per writer thread. Multi-thread cells raise
/// both engines' equal total budget rather than silently reducing thread count.
pub const PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES: usize = 15_000_000;
/// Version of the metric-specific paired estimator contract.
pub const PAIRED_ESTIMATOR_SCHEMA_VERSION: &str = "quill-paired-estimator-v1";
/// Exact ordered query groups required by every normative QG-6 class cell.
///
/// One group is one independent frozen query identity. Leaves collected for a
/// query are repeated measurements of that group, never additional queries.
pub const QG6_QUERY_GROUPS: usize = 16;
/// Canonical QG-6 group IDs. Prepared queries are indexed in manifest order.
pub const QG6_QUERY_GROUP_IDS: [u64; QG6_QUERY_GROUPS] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Exact normative performance manifest compiled into every applicability plan.
const NORMATIVE_PERF_MANIFEST: &str = include_str!("../../../docs/contracts/quill-perf-gates.toml");

/// Hash the normative performance contract without binding administrative
/// activation state into measurement identity.
///
/// A gate is necessarily measured before its `activated` flag can be flipped.
/// Canonicalizing every exact `activated = true` assignment to `false` keeps
/// that review-only transition from invalidating the evidence it activates,
/// while every fixture, target, estimator, and provenance change still moves
/// the digest.
#[must_use]
pub fn perf_manifest_contract_sha256(manifest: &str) -> String {
    let mut normalized = String::with_capacity(manifest.len());
    for line in manifest.split_inclusive('\n') {
        match line {
            "activated = true\n" => normalized.push_str("activated = false\n"),
            "activated = true\r\n" => normalized.push_str("activated = false\r\n"),
            "activated = true" => normalized.push_str("activated = false"),
            _ => normalized.push_str(line),
        }
    }
    lower_sha256_hex(normalized.as_bytes())
}

fn lower_sha256_hex(bytes: &[u8]) -> String {
    finish_sha256_hex(Sha256::new_with_prefix(bytes))
}

fn finish_sha256_hex(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn update_length_framed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_string().as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
}

/// Equal total heap budget for one thread-count cell.
#[must_use]
pub const fn perf_writer_heap_bytes(threads: usize) -> usize {
    let per_thread = PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES.saturating_mul(threads);
    if per_thread > PERF_WRITER_HEAP_BYTES {
        per_thread
    } else {
        PERF_WRITER_HEAP_BYTES
    }
}

/// One normative Quill performance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerfGate {
    Qg1,
    Qg2,
    Qg3,
    Qg4,
    Qg5,
    Qg6,
    Qg7,
    Qg8,
    Qg9,
    Qg10,
}

impl PerfGate {
    /// Gates in the normative manifest order.
    pub const ALL: [Self; 10] = [
        Self::Qg1,
        Self::Qg2,
        Self::Qg3,
        Self::Qg4,
        Self::Qg5,
        Self::Qg6,
        Self::Qg7,
        Self::Qg8,
        Self::Qg9,
        Self::Qg10,
    ];

    /// Stable manifest label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Qg1 => "QG-1",
            Self::Qg2 => "QG-2",
            Self::Qg3 => "QG-3",
            Self::Qg4 => "QG-4",
            Self::Qg5 => "QG-5",
            Self::Qg6 => "QG-6",
            Self::Qg7 => "QG-7",
            Self::Qg8 => "QG-8",
            Self::Qg9 => "QG-9",
            Self::Qg10 => "QG-10",
        }
    }
}

impl fmt::Display for PerfGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

impl FromStr for PerfGate {
    type Err = GauntletError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_uppercase().replace('_', "-");
        Self::ALL
            .into_iter()
            .find(|gate| gate.label() == normalized)
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: format!("unknown Quill performance gate {value:?}"),
            })
    }
}

/// Pinned corpus sizes from the FSFS golden fixtures plus the E6 xlarge recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfCorpus {
    Tiny,
    Small,
    Medium,
    Xlarge,
}

impl PerfCorpus {
    /// Number of documents in the committed profile.
    #[must_use]
    pub const fn document_count(self) -> u64 {
        match self {
            Self::Tiny => 500,
            Self::Small => 5_000,
            Self::Medium => 50_000,
            Self::Xlarge => 1_000_000,
        }
    }

    /// Stable fixture label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Xlarge => "xlarge",
        }
    }
}

/// Whether text fields retain exact token positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionMode {
    On,
    Off,
}

impl PositionMode {
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::On => "positions_on",
            Self::Off => "positions_off",
        }
    }
}

/// Visibility topology required by QG-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfTopology {
    InProcess,
    FreshProcess,
}

/// Query families pinned by QG-6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfQueryClass {
    Identifier,
    ShortKeyword,
    NaturalLanguage,
    Phrase,
    Boolean,
}

impl PerfQueryClass {
    pub const ALL: [Self; 5] = [
        Self::Identifier,
        Self::ShortKeyword,
        Self::NaturalLanguage,
        Self::Phrase,
        Self::Boolean,
    ];

    /// Stable manifest label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Identifier => "identifier",
            Self::ShortKeyword => "short_keyword",
            Self::NaturalLanguage => "natural_language",
            Self::Phrase => "phrase",
            Self::Boolean => "boolean",
        }
    }
}

/// One fully pinned matrix cell before it is measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfCellSpec {
    pub gate: PerfGate,
    pub fixture: String,
    pub metric: String,
    pub corpus: Option<PerfCorpus>,
    pub document_count: Option<u64>,
    pub threads: Option<usize>,
    pub writer_heap_bytes: Option<usize>,
    pub positions: Option<PositionMode>,
    pub tombstone_density_pct: Option<u8>,
    pub query_class: Option<PerfQueryClass>,
    pub k: Option<usize>,
    pub topology: Option<PerfTopology>,
}

impl PerfCellSpec {
    fn new(gate: PerfGate, fixture: impl Into<String>, metric: impl Into<String>) -> Self {
        Self {
            gate,
            fixture: fixture.into(),
            metric: metric.into(),
            corpus: None,
            document_count: None,
            threads: None,
            writer_heap_bytes: None,
            positions: None,
            tombstone_density_pct: None,
            query_class: None,
            k: None,
            topology: None,
        }
    }

    /// Domain-separated identity of every serialized cell-contract field.
    ///
    /// The JSON payload is length-framed so string contents cannot alias field
    /// boundaries. Struct field order and enum spellings are deliberately part
    /// of this versioned contract: adding, removing, reordering, or changing a
    /// field changes the digest.
    ///
    /// # Errors
    ///
    /// Returns a JSON error if the cell cannot be serialized.
    pub fn contract_sha256(&self) -> Result<String, GauntletError> {
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.perf-cell-contract.v1\0");
        update_length_framed(&mut hasher, encoded.as_slice());
        Ok(finish_sha256_hex(hasher))
    }
}

/// Wire schema for a profile-qualified projection of one canonical gate matrix.
pub const PERF_APPLICABILITY_PLAN_SCHEMA_VERSION: &str =
    "frankensearch.quill-perf-applicability-plan.v2";

/// Whether one canonical cell is required, diagnostic, or impossible for one
/// immutable execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfCellApplicability {
    /// The cell must be measured to satisfy this profile's default-flip gate.
    Required,
    /// The cell may be measured but can never satisfy a release requirement.
    Diagnostic,
    /// The cell is outside the profile's frozen execution envelope.
    NotApplicable,
}

impl PerfCellApplicability {
    /// Whether the benchmark may execute this cell under the profile.
    #[must_use]
    pub const fn is_runnable(self) -> bool {
        !matches!(self, Self::NotApplicable)
    }

    const fn contract_label(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::Diagnostic => "diagnostic",
            Self::NotApplicable => "not_applicable",
        }
    }
}

/// Stable reason for one profile-specific cell classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PerfCellApplicabilityReason {
    /// The profile and gate require this ordinary measurement cell.
    RequiredForDefaultFlip,
    /// The complete profile is diagnostic-only for this gate.
    DiagnosticProfile,
    /// The canonical matrix defines this cell as diagnostic for every profile.
    DiagnosticCell,
    /// The cell's configured width exceeds the profile's frozen gate maximum.
    ExceedsProfileMaximum {
        /// Immutable hardware and execution profile that makes the cell N/A.
        profile: MachineProfileKey,
        /// Meaning of the admitted execution-capacity value.
        capacity_semantics: ExecutionCapacitySemantics,
        /// Frozen capacity admitted for this exact profile.
        execution_capacity: u64,
        /// Canonical cell width that the profile cannot exercise.
        required_cell_width: u64,
        /// Widest canonical cell the profile may exercise for this gate.
        max_exercised_cell_width: u64,
    },
}

impl PerfCellApplicabilityReason {
    const fn contract_label(self) -> &'static str {
        match self {
            Self::RequiredForDefaultFlip => "required_for_default_flip",
            Self::DiagnosticProfile => "diagnostic_profile",
            Self::DiagnosticCell => "diagnostic_cell",
            Self::ExceedsProfileMaximum { .. } => "exceeds_profile_maximum",
        }
    }

    fn update_contract_hash(self, hasher: &mut Sha256) {
        update_length_framed(hasher, self.contract_label().as_bytes());
        if let Self::ExceedsProfileMaximum {
            profile,
            capacity_semantics,
            execution_capacity,
            required_cell_width,
            max_exercised_cell_width,
        } = self
        {
            update_length_framed(hasher, profile.hardware_class_id().as_str().as_bytes());
            update_length_framed(hasher, profile.execution_profile_id().as_str().as_bytes());
            update_length_framed(
                hasher,
                capacity_semantics_contract_label(capacity_semantics).as_bytes(),
            );
            update_length_framed(hasher, execution_capacity.to_string().as_bytes());
            update_length_framed(hasher, required_cell_width.to_string().as_bytes());
            update_length_framed(hasher, max_exercised_cell_width.to_string().as_bytes());
        }
    }
}

/// One ordered classification inside a profile applicability plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfCellApplicabilityEntry {
    /// Zero-based ordinal in the unchanged canonical gate matrix.
    pub ordinal: usize,
    /// Domain-separated contract hash of the exact canonical cell.
    pub cell_contract_sha256: String,
    /// Configured engine width frozen in that cell.
    pub configured_threads: usize,
    /// Profile-specific classification.
    pub applicability: PerfCellApplicability,
    /// Stable explanation for the classification.
    pub reason: PerfCellApplicabilityReason,
}

/// Compact identity downstream artifacts carry to name one exact plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfApplicabilityPlanBinding {
    /// Applicability-plan wire schema.
    pub schema_version: String,
    /// Immutable hardware and execution-profile identity.
    pub profile: MachineProfileKey,
    /// Frozen machine-registry schema interpreted by the planner.
    pub registry_schema_version: String,
    /// SHA-256 of the exact frozen machine-registry bytes.
    pub registry_sha256: String,
    /// Domain-separated hash of the exact profile contract object.
    pub profile_contract_sha256: String,
    /// Gate whose complete canonical slice is classified.
    pub gate: PerfGate,
    /// SHA-256 of the independently normalized normative performance manifest.
    pub normalized_perf_manifest_sha256: String,
    /// Mandatory ordinary-cell width declared by this gate, when one is frozen.
    pub primary_target_cell_width: Option<u64>,
    /// Schema of the canonical gate-matrix hash contract.
    pub matrix_contract_schema_version: String,
    /// Ordered hash of every canonical cell in this gate.
    pub gate_matrix_contract_sha256: String,
    /// Ordered hash of the complete profile-specific plan preimage.
    pub applicability_plan_sha256: String,
}

/// Exhaustive, immutable classification of one canonical gate for one profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfApplicabilityPlan {
    /// Content-addressed profile, registry, matrix, and plan identity.
    pub binding: PerfApplicabilityPlanBinding,
    /// Meaning of this profile's admitted execution capacity.
    pub capacity_semantics: ExecutionCapacitySemantics,
    /// Maximum admitted capacity, if the registered profile freezes one.
    pub execution_capacity: Option<u64>,
    /// Whether this profile is required or diagnostic for the gate.
    pub default_flip_disposition: DefaultFlipDisposition,
    /// Widest canonical cell this profile may exercise for the gate.
    pub max_exercised_cell_width: Option<u64>,
    /// Every canonical gate cell, in unchanged matrix order.
    pub cells: Vec<PerfCellApplicabilityEntry>,
}

impl PerfApplicabilityPlan {
    /// Borrow the compact identity suitable for threshold/evidence artifacts.
    #[must_use]
    pub const fn binding(&self) -> &PerfApplicabilityPlanBinding {
        &self.binding
    }

    /// Count cells with one exact applicability classification.
    #[must_use]
    pub fn cell_count(&self, applicability: PerfCellApplicability) -> usize {
        self.cells
            .iter()
            .filter(|cell| cell.applicability == applicability)
            .count()
    }

    /// Widest runnable canonical cell in the plan.
    #[must_use]
    pub fn max_runnable_cell_width(&self) -> Option<usize> {
        self.cells
            .iter()
            .filter(|cell| cell.applicability.is_runnable())
            .map(|cell| cell.configured_threads)
            .max()
    }

    /// Rebuild this plan from the frozen registry and canonical matrix.
    ///
    /// # Errors
    ///
    /// Returns a typed planning error if the profile is unavailable, the
    /// matrix or registry changed, or any stored plan field was modified.
    pub fn verify_against(
        &self,
        matrix: &PerfMatrixSpec,
        registry: &MachineClassRegistry,
    ) -> Result<(), PerfApplicabilityPlanError> {
        let expected =
            matrix.applicability_plan(registry, self.binding.profile, self.binding.gate)?;
        if *self != expected {
            return Err(PerfApplicabilityPlanError::PlanMismatch {
                profile: self.binding.profile,
                gate: self.binding.gate,
            });
        }
        Ok(())
    }

    fn contract_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.perf-applicability-plan.v2\0");
        update_length_framed(&mut hasher, self.binding.schema_version.as_bytes());
        update_length_framed(
            &mut hasher,
            self.binding.profile.hardware_class_id().as_str().as_bytes(),
        );
        update_length_framed(
            &mut hasher,
            self.binding
                .profile
                .execution_profile_id()
                .as_str()
                .as_bytes(),
        );
        update_length_framed(&mut hasher, self.binding.registry_schema_version.as_bytes());
        update_length_framed(&mut hasher, self.binding.registry_sha256.as_bytes());
        update_length_framed(&mut hasher, self.binding.profile_contract_sha256.as_bytes());
        update_length_framed(&mut hasher, self.binding.gate.label().as_bytes());
        update_length_framed(
            &mut hasher,
            self.binding.normalized_perf_manifest_sha256.as_bytes(),
        );
        update_optional_u64(&mut hasher, self.binding.primary_target_cell_width);
        update_length_framed(
            &mut hasher,
            self.binding.matrix_contract_schema_version.as_bytes(),
        );
        update_length_framed(
            &mut hasher,
            self.binding.gate_matrix_contract_sha256.as_bytes(),
        );
        update_length_framed(
            &mut hasher,
            capacity_semantics_contract_label(self.capacity_semantics).as_bytes(),
        );
        update_optional_u64(&mut hasher, self.execution_capacity);
        update_length_framed(
            &mut hasher,
            default_flip_disposition_contract_label(self.default_flip_disposition).as_bytes(),
        );
        update_optional_u64(&mut hasher, self.max_exercised_cell_width);
        update_length_framed(&mut hasher, self.cells.len().to_string().as_bytes());
        for cell in &self.cells {
            update_length_framed(&mut hasher, cell.ordinal.to_string().as_bytes());
            update_length_framed(&mut hasher, cell.cell_contract_sha256.as_bytes());
            update_length_framed(&mut hasher, cell.configured_threads.to_string().as_bytes());
            update_length_framed(&mut hasher, cell.applicability.contract_label().as_bytes());
            cell.reason.update_contract_hash(&mut hasher);
        }
        finish_sha256_hex(hasher)
    }
}

/// Fail-closed applicability-plan construction or verification error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PerfApplicabilityPlanError {
    /// The frozen registry rejected the requested profile.
    #[error(transparent)]
    Registry(#[from] MachineClassError),
    /// Only the exact complete matrix may be projected.
    #[error("applicability plans require the exact canonical performance matrix")]
    NonCanonicalMatrix,
    /// An unavailable required profile cannot manufacture an all-NA plan.
    #[error("execution profile {profile:?} is unavailable for {gate}")]
    ProfileUnavailable {
        /// Unavailable profile key.
        profile: MachineProfileKey,
        /// Gate that cannot be planned.
        gate: PerfGate,
    },
    /// The profile omitted the selected gate policy.
    #[error("execution profile {profile:?} has no policy for {gate}")]
    MissingGatePolicy {
        /// Profile with the incomplete contract.
        profile: MachineProfileKey,
        /// Missing gate.
        gate: PerfGate,
    },
    /// A required profile omitted its maximum runnable cell width.
    #[error("required execution profile {profile:?} has no maximum width for {gate}")]
    RequiredProfileWithoutMaximum {
        /// Profile with the incomplete contract.
        profile: MachineProfileKey,
        /// Gate with no maximum.
        gate: PerfGate,
    },
    /// A diagnostic profile cannot make runnable claims without both bounds.
    #[error(
        "diagnostic execution profile {profile:?} cannot plan {gate} without a verified bounded \
         capacity envelope (capacity {execution_capacity:?}, maximum \
         {max_exercised_cell_width:?})"
    )]
    UnboundedDiagnosticProfile {
        /// Diagnostic profile whose execution envelope is incomplete.
        profile: MachineProfileKey,
        /// Gate that cannot be planned.
        gate: PerfGate,
        /// Hash-bound execution capacity, when one exists.
        execution_capacity: Option<u64>,
        /// Hash-bound maximum runnable canonical width, when one exists.
        max_exercised_cell_width: Option<u64>,
    },
    /// Profile capacity and per-gate maximum contradict one another.
    #[error("execution profile {profile:?} has an invalid capacity envelope for {gate}")]
    InvalidCapacityEnvelope {
        /// Profile with the invalid envelope.
        profile: MachineProfileKey,
        /// Gate whose maximum is invalid.
        gate: PerfGate,
    },
    /// A required profile cannot exercise the gate's primary target width.
    #[error(
        "required execution profile {profile:?} cannot exercise {gate} primary target width \
         {primary_target_cell_width} with capacity {execution_capacity:?} and maximum \
         {max_exercised_cell_width:?}"
    )]
    RequiredProfileBelowPrimaryTarget {
        /// Profile whose execution envelope is too narrow.
        profile: MachineProfileKey,
        /// Gate whose primary target would be omitted.
        gate: PerfGate,
        /// Manifest-declared mandatory target width.
        primary_target_cell_width: u64,
        /// Frozen execution capacity.
        execution_capacity: Option<u64>,
        /// Frozen maximum runnable canonical width.
        max_exercised_cell_width: Option<u64>,
    },
    /// The normative manifest cannot provide a bounded gate identity.
    #[error("invalid normative performance manifest contract for {gate}: {detail}")]
    ManifestContract {
        /// Gate whose manifest identity is invalid.
        gate: PerfGate,
        /// Bounded parse or shape failure.
        detail: String,
    },
    /// A canonical cell lacks a representable positive configured width.
    #[error("canonical {gate} cell ordinal {ordinal} has no representable positive width")]
    InvalidCellWidth {
        /// Gate containing the cell.
        gate: PerfGate,
        /// Gate-local canonical ordinal.
        ordinal: usize,
    },
    /// A matrix or cell identity could not be computed.
    #[error("cannot hash the canonical applicability input: {detail}")]
    ContractIdentity {
        /// Underlying bounded contract error.
        detail: String,
    },
    /// Stored plan contents differ from a fresh reconstruction.
    #[error("stored applicability plan for {profile:?} {gate} does not recompute")]
    PlanMismatch {
        /// Profile named by the stale plan.
        profile: MachineProfileKey,
        /// Gate named by the stale plan.
        gate: PerfGate,
    },
}

const fn capacity_semantics_contract_label(value: ExecutionCapacitySemantics) -> &'static str {
    match value {
        ExecutionCapacitySemantics::PhysicalCores => "physical_cores",
        ExecutionCapacitySemantics::LogicalThreads => "logical_threads",
        ExecutionCapacitySemantics::SchedulerWorkers => "scheduler_workers",
        ExecutionCapacitySemantics::DiagnosticWorkerBudget => "diagnostic_worker_budget",
    }
}

const fn default_flip_disposition_contract_label(value: DefaultFlipDisposition) -> &'static str {
    match value {
        DefaultFlipDisposition::RequiredForDefaultFlip => "required_for_default_flip",
        DefaultFlipDisposition::DiagnosticOnly => "diagnostic_only",
    }
}

fn update_optional_u64(hasher: &mut Sha256, value: Option<u64>) {
    match value {
        Some(value) => {
            update_length_framed(hasher, b"some");
            update_length_framed(hasher, value.to_string().as_bytes());
        }
        None => update_length_framed(hasher, b"none"),
    }
}

/// Complete bounded facts used to classify cells for one admitted profile.
///
/// Production construction occurs only after the registry's profile contract
/// supplies both values. A future receipt-bound diagnostic path must bind the
/// same facts before constructing this envelope; `None` never means unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundedProfileApplicabilityEnvelope {
    profile: MachineProfileKey,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    disposition: DefaultFlipDisposition,
    max_exercised_cell_width: u64,
}

impl BoundedProfileApplicabilityEnvelope {
    fn classify_cell(
        self,
        configured_width: u64,
        canonical_diagnostic: bool,
    ) -> (PerfCellApplicability, PerfCellApplicabilityReason) {
        if configured_width > self.max_exercised_cell_width {
            (
                PerfCellApplicability::NotApplicable,
                PerfCellApplicabilityReason::ExceedsProfileMaximum {
                    profile: self.profile,
                    capacity_semantics: self.capacity_semantics,
                    execution_capacity: self.execution_capacity,
                    required_cell_width: configured_width,
                    max_exercised_cell_width: self.max_exercised_cell_width,
                },
            )
        } else if self.disposition == DefaultFlipDisposition::DiagnosticOnly {
            (
                PerfCellApplicability::Diagnostic,
                PerfCellApplicabilityReason::DiagnosticProfile,
            )
        } else if canonical_diagnostic {
            (
                PerfCellApplicability::Diagnostic,
                PerfCellApplicabilityReason::DiagnosticCell,
            )
        } else {
            (
                PerfCellApplicability::Required,
                PerfCellApplicabilityReason::RequiredForDefaultFlip,
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PerfGateManifestIdentity {
    normalized_perf_manifest_sha256: String,
    primary_target_cell_width: Option<u64>,
}

fn perf_gate_manifest_identity(
    manifest: &str,
    gate: PerfGate,
) -> Result<PerfGateManifestIdentity, PerfApplicabilityPlanError> {
    let parsed = toml::from_str::<toml::Value>(manifest).map_err(|error| {
        PerfApplicabilityPlanError::ManifestContract {
            gate,
            detail: format!("manifest is not valid TOML: {error}"),
        }
    })?;
    let schema = parsed
        .get("schemas")
        .and_then(toml::Value::as_table)
        .and_then(|schemas| schemas.get("applicability_plan"))
        .and_then(toml::Value::as_str)
        .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
            gate,
            detail: "schemas.applicability_plan is missing or not a string".to_owned(),
        })?;
    if schema != PERF_APPLICABILITY_PLAN_SCHEMA_VERSION {
        return Err(PerfApplicabilityPlanError::ManifestContract {
            gate,
            detail: format!(
                "schemas.applicability_plan is {schema:?}, expected \
                 {PERF_APPLICABILITY_PLAN_SCHEMA_VERSION:?}"
            ),
        });
    }
    let gate_contract = parsed
        .get("gate")
        .and_then(toml::Value::as_table)
        .and_then(|gates| gates.get(gate.label()))
        .and_then(toml::Value::as_table)
        .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
            gate,
            detail: "gate table is missing or not a table".to_owned(),
        })?;
    let primary_target_cell_width = gate_contract
        .get("primary_target_cell_width")
        .map(|value| {
            let width =
                value
                    .as_integer()
                    .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
                        gate,
                        detail: "primary_target_cell_width is not an integer".to_owned(),
                    })?;
            u64::try_from(width)
                .ok()
                .filter(|width| *width > 0)
                .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
                    gate,
                    detail: "primary_target_cell_width is not a positive u64".to_owned(),
                })
        })
        .transpose()?;
    if gate == PerfGate::Qg1 && primary_target_cell_width.is_none() {
        return Err(PerfApplicabilityPlanError::ManifestContract {
            gate,
            detail: "QG-1 requires primary_target_cell_width".to_owned(),
        });
    }

    Ok(PerfGateManifestIdentity {
        normalized_perf_manifest_sha256: perf_manifest_contract_sha256(manifest),
        primary_target_cell_width,
    })
}

/// Complete, deterministic QG-1..QG-10 execution matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfMatrixSpec {
    pub manifest: String,
    pub cells: Vec<PerfCellSpec>,
}

impl PerfMatrixSpec {
    /// Schema bound into canonical gate-matrix identities.
    pub const CONTRACT_SCHEMA_VERSION: &str = "quill-perf-gate-matrix-contract-v1";

    /// Reviewed identity of the unchanged canonical 74-cell QG-1 universe.
    ///
    /// Per-machine applicability plans project this universe; they must never
    /// replace it with a smaller caller-selected matrix.
    pub const QG1_CANONICAL_SHA256: &str =
        "1b0080f0ebb444a9e653161e4989c6df19971235037d44b45e2506c2cbf2a3a7";

    /// Build every normative cell. Runtime slice filters may select a gate or
    /// fixture, but they never redefine the matrix.
    #[must_use]
    pub fn complete() -> Self {
        let mut cells = Vec::new();
        let corpora = [
            PerfCorpus::Tiny,
            PerfCorpus::Small,
            PerfCorpus::Medium,
            PerfCorpus::Xlarge,
        ];
        for corpus in corpora {
            for threads in [1, 2, 4, 8, 16, 32, 64, 96, 128] {
                for positions in [PositionMode::On, PositionMode::Off] {
                    let mut cell = PerfCellSpec::new(
                        PerfGate::Qg1,
                        format!("bulk/{}/{threads}/{}", corpus.label(), positions.label()),
                        "docs_per_second",
                    );
                    cell.corpus = Some(corpus);
                    cell.document_count = Some(corpus.document_count());
                    cell.threads = Some(threads);
                    cell.writer_heap_bytes = Some(perf_writer_heap_bytes(threads));
                    cell.positions = Some(positions);
                    cells.push(cell);
                }
            }
        }
        for corpus in [PerfCorpus::Medium, PerfCorpus::Xlarge] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg1,
                format!("tokenize_only/{}", corpus.label()),
                "tokenize_docs_per_second",
            );
            cell.corpus = Some(corpus);
            cell.document_count = Some(corpus.document_count());
            cell.threads = Some(1);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
            cells.push(cell);
        }

        let mut single = PerfCellSpec::new(
            PerfGate::Qg2,
            "bulk/medium/1/positions_on",
            "docs_per_second",
        );
        single.corpus = Some(PerfCorpus::Medium);
        single.document_count = Some(PerfCorpus::Medium.document_count());
        single.threads = Some(1);
        single.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        single.positions = Some(PositionMode::On);
        cells.push(single);

        let mut initial =
            PerfCellSpec::new(PerfGate::Qg3, "watch/medium/initial", "docs_per_second");
        initial.corpus = Some(PerfCorpus::Medium);
        initial.document_count = Some(PerfCorpus::Medium.document_count());
        initial.threads = Some(1);
        initial.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        initial.positions = Some(PositionMode::On);
        cells.push(initial);

        for topology in [PerfTopology::InProcess, PerfTopology::FreshProcess] {
            for metric in ["updates_per_second", "update_to_searchable_ms"] {
                let mut cell = PerfCellSpec::new(
                    PerfGate::Qg3,
                    format!("watch/medium/5000/{topology:?}").to_ascii_lowercase(),
                    metric,
                );
                cell.corpus = Some(PerfCorpus::Medium);
                cell.document_count = Some(5_000);
                cell.threads = Some(1);
                cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
                cell.positions = Some(PositionMode::On);
                cell.topology = Some(topology);
                cells.push(cell);
            }
        }

        let mut commit =
            PerfCellSpec::new(PerfGate::Qg4, "commit/100000/warm", "commit_latency_ms");
        commit.document_count = Some(100_000);
        commit.positions = Some(PositionMode::On);
        commit.threads = Some(1);
        commit.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        cells.push(commit);

        for density in [5, 20, 50] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg5,
                format!("compaction/xlarge/{density}pct"),
                "wall_clock_ms",
            );
            cell.corpus = Some(PerfCorpus::Xlarge);
            cell.document_count = Some(PerfCorpus::Xlarge.document_count());
            cell.positions = Some(PositionMode::On);
            cell.threads = Some(1);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
            cell.tombstone_density_pct = Some(density);
            cells.push(cell);
        }

        for query_class in PerfQueryClass::ALL {
            for k in [10, 100] {
                for (label, document_count) in [("100k", 100_000), ("1m", 1_000_000)] {
                    let mut cell = PerfCellSpec::new(
                        PerfGate::Qg6,
                        format!("query/{}/k{k}/{label}", query_class.label()),
                        "latency_ms",
                    );
                    cell.document_count = Some(document_count);
                    cell.positions = Some(PositionMode::On);
                    cell.threads = Some(1);
                    cell.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
                    cell.query_class = Some(query_class);
                    cell.k = Some(k);
                    cells.push(cell);
                }
            }
        }

        for corpus in [PerfCorpus::Medium, PerfCorpus::Xlarge] {
            for positions in [PositionMode::On, PositionMode::Off] {
                let mut rss = PerfCellSpec::new(
                    PerfGate::Qg7,
                    format!("memory/{}/{}", corpus.label(), positions.label()),
                    "peak_rss_bytes",
                );
                rss.corpus = Some(corpus);
                rss.document_count = Some(corpus.document_count());
                rss.threads = Some(8);
                rss.writer_heap_bytes = Some(perf_writer_heap_bytes(8));
                rss.positions = Some(positions);
                cells.push(rss);

                let mut bytes = PerfCellSpec::new(
                    PerfGate::Qg7,
                    format!("size/{}/{}", corpus.label(), positions.label()),
                    "index_bytes_per_document",
                );
                bytes.corpus = Some(corpus);
                bytes.document_count = Some(corpus.document_count());
                bytes.threads = Some(8);
                bytes.writer_heap_bytes = Some(perf_writer_heap_bytes(8));
                bytes.positions = Some(positions);
                cells.push(bytes);
            }
        }

        for threads in [1, 2, 4, 8, 16, 32] {
            let mut cell = PerfCellSpec::new(
                PerfGate::Qg8,
                format!("scaling/xlarge/{threads}/positions_on"),
                "docs_per_second",
            );
            cell.corpus = Some(PerfCorpus::Xlarge);
            cell.document_count = Some(PerfCorpus::Xlarge.document_count());
            cell.threads = Some(threads);
            cell.writer_heap_bytes = Some(perf_writer_heap_bytes(threads));
            cell.positions = Some(PositionMode::On);
            cells.push(cell);
        }

        let mut cold =
            PerfCellSpec::new(PerfGate::Qg9, "cold_open/xlarge/default", "open_latency_ms");
        cold.corpus = Some(PerfCorpus::Xlarge);
        cold.document_count = Some(PerfCorpus::Xlarge.document_count());
        cold.positions = Some(PositionMode::On);
        cold.threads = Some(1);
        cold.writer_heap_bytes = Some(perf_writer_heap_bytes(1));
        cells.push(cold);

        let mut dependencies = PerfCellSpec::new(
            PerfGate::Qg10,
            "dependency_surface/default_lexical",
            "tantivy_nodes",
        );
        dependencies.threads = Some(1);
        cells.push(dependencies);

        Self {
            manifest: "docs/contracts/quill-perf-gates.toml".to_owned(),
            cells,
        }
    }

    /// Select an immutable matrix slice without changing its pins.
    #[must_use]
    pub fn for_gate(&self, gate: PerfGate) -> Vec<&PerfCellSpec> {
        self.cells.iter().filter(|cell| cell.gate == gate).collect()
    }

    /// Maximum engine width required by a gate's complete frozen matrix.
    #[must_use]
    pub fn max_thread_width(&self, gate: PerfGate) -> Option<usize> {
        self.cells
            .iter()
            .filter(|cell| cell.gate == gate)
            .filter_map(|cell| cell.threads)
            .max()
    }

    /// Hash one gate's ordered projection from the global immutable matrix.
    ///
    /// This binds the matrix schema, gate, cell count, cell order, and every
    /// domain-separated cell-contract hash. The normalized semantic manifest
    /// hash remains a separate identity so later plans can reconcile both
    /// independently. Applicability is intentionally absent: a later
    /// machine-bound plan classifies these exact cells without redefining them.
    ///
    /// # Errors
    ///
    /// Returns an error when the gate has no cells or a cell cannot be
    /// serialized.
    pub fn gate_contract_sha256(&self, gate: PerfGate) -> Result<String, GauntletError> {
        let cells = self.for_gate(gate);
        if cells.is_empty() {
            return Err(GauntletError::InvalidCampaign {
                reason: format!("{} has no cells to hash", gate),
            });
        }

        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.perf-gate-matrix-contract.v1\0");
        update_length_framed(&mut hasher, Self::CONTRACT_SCHEMA_VERSION.as_bytes());
        update_length_framed(&mut hasher, gate.label().as_bytes());
        update_length_framed(&mut hasher, cells.len().to_string().as_bytes());
        for (ordinal, cell) in cells.into_iter().enumerate() {
            update_length_framed(&mut hasher, ordinal.to_string().as_bytes());
            let cell_sha256 = cell.contract_sha256()?;
            update_length_framed(&mut hasher, cell_sha256.as_bytes());
        }
        Ok(finish_sha256_hex(hasher))
    }

    /// Project one complete canonical gate through an immutable execution
    /// profile.
    ///
    /// This method deliberately accepts neither a caller-selected cell slice
    /// nor an ad hoc width. The frozen registry supplies release disposition
    /// and maximum width, while the unchanged complete matrix supplies every
    /// ordered cell and its contract identity.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a noncanonical matrix, unknown or unavailable
    /// profile, incomplete profile policy, contradictory capacity envelope, or
    /// non-hashable cell contract.
    pub fn applicability_plan(
        &self,
        registry: &MachineClassRegistry,
        profile_key: MachineProfileKey,
        gate: PerfGate,
    ) -> Result<PerfApplicabilityPlan, PerfApplicabilityPlanError> {
        if self != &Self::complete() {
            return Err(PerfApplicabilityPlanError::NonCanonicalMatrix);
        }
        validate_matrix(self).map_err(|error| PerfApplicabilityPlanError::ContractIdentity {
            detail: error.to_string(),
        })?;
        let profile = registry.execution_profile(profile_key)?;
        let manifest_identity = perf_gate_manifest_identity(NORMATIVE_PERF_MANIFEST, gate)?;
        self.applicability_plan_for_profile(profile, gate, &manifest_identity)
    }

    /// Build one exhaustive plan for every normative gate.
    ///
    /// Concatenating the returned plans' cell entries classifies the complete
    /// unchanged [`PerfMatrixSpec`] universe exactly once.
    ///
    /// # Errors
    ///
    /// Returns the first typed planning error in normative gate order.
    pub fn applicability_plans(
        &self,
        registry: &MachineClassRegistry,
        profile_key: MachineProfileKey,
    ) -> Result<Vec<PerfApplicabilityPlan>, PerfApplicabilityPlanError> {
        PerfGate::ALL
            .into_iter()
            .map(|gate| self.applicability_plan(registry, profile_key, gate))
            .collect()
    }

    fn applicability_plan_for_profile(
        &self,
        profile: &MachineExecutionProfile,
        gate: PerfGate,
        manifest_identity: &PerfGateManifestIdentity,
    ) -> Result<PerfApplicabilityPlan, PerfApplicabilityPlanError> {
        let profile_key = profile.key();
        if profile.availability() == MachineProfileAvailability::Unavailable {
            return Err(PerfApplicabilityPlanError::ProfileUnavailable {
                profile: profile_key,
                gate,
            });
        }
        let policy = profile.gate_policy(gate.label()).ok_or(
            PerfApplicabilityPlanError::MissingGatePolicy {
                profile: profile_key,
                gate,
            },
        )?;
        let disposition = policy.default_flip_disposition();
        let max_exercised_cell_width = policy.max_exercised_cell_width();
        let execution_capacity = profile.execution_capacity();
        if disposition == DefaultFlipDisposition::RequiredForDefaultFlip
            && max_exercised_cell_width.is_none()
        {
            return Err(PerfApplicabilityPlanError::RequiredProfileWithoutMaximum {
                profile: profile_key,
                gate,
            });
        }
        if disposition == DefaultFlipDisposition::DiagnosticOnly
            && (execution_capacity.is_none() || max_exercised_cell_width.is_none())
        {
            return Err(PerfApplicabilityPlanError::UnboundedDiagnosticProfile {
                profile: profile_key,
                gate,
                execution_capacity,
                max_exercised_cell_width,
            });
        }
        if execution_capacity == Some(0)
            || max_exercised_cell_width == Some(0)
            || (execution_capacity.is_none() && max_exercised_cell_width.is_some())
            || execution_capacity.is_some_and(|capacity| {
                max_exercised_cell_width.is_some_and(|maximum| maximum > capacity)
            })
        {
            return Err(PerfApplicabilityPlanError::InvalidCapacityEnvelope {
                profile: profile_key,
                gate,
            });
        }
        if let Some(primary_target_cell_width) = manifest_identity.primary_target_cell_width {
            let target_exists = self.for_gate(gate).into_iter().any(|cell| {
                cell.threads.and_then(|width| u64::try_from(width).ok())
                    == Some(primary_target_cell_width)
                    && !canonical_cell_is_diagnostic(cell)
            });
            if !target_exists {
                return Err(PerfApplicabilityPlanError::ManifestContract {
                    gate,
                    detail: format!(
                        "primary target width {primary_target_cell_width} has no ordinary \
                         canonical cell"
                    ),
                });
            }
            if disposition == DefaultFlipDisposition::RequiredForDefaultFlip
                && (execution_capacity.is_none_or(|capacity| capacity < primary_target_cell_width)
                    || max_exercised_cell_width
                        .is_none_or(|maximum| maximum < primary_target_cell_width))
            {
                return Err(
                    PerfApplicabilityPlanError::RequiredProfileBelowPrimaryTarget {
                        profile: profile_key,
                        gate,
                        primary_target_cell_width,
                        execution_capacity,
                        max_exercised_cell_width,
                    },
                );
            }
        }
        let (Some(execution_capacity), Some(max_exercised_cell_width)) =
            (execution_capacity, max_exercised_cell_width)
        else {
            return Err(PerfApplicabilityPlanError::InvalidCapacityEnvelope {
                profile: profile_key,
                gate,
            });
        };
        let envelope = BoundedProfileApplicabilityEnvelope {
            profile: profile_key,
            capacity_semantics: profile.capacity_semantics(),
            execution_capacity,
            disposition,
            max_exercised_cell_width,
        };

        let mut entries = Vec::new();
        for (ordinal, cell) in self.for_gate(gate).into_iter().enumerate() {
            let configured_threads = cell
                .threads
                .filter(|threads| *threads > 0)
                .ok_or(PerfApplicabilityPlanError::InvalidCellWidth { gate, ordinal })?;
            let configured_width = u64::try_from(configured_threads)
                .map_err(|_| PerfApplicabilityPlanError::InvalidCellWidth { gate, ordinal })?;
            let (applicability, reason) =
                envelope.classify_cell(configured_width, canonical_cell_is_diagnostic(cell));
            entries.push(PerfCellApplicabilityEntry {
                ordinal,
                cell_contract_sha256: cell.contract_sha256().map_err(|error| {
                    PerfApplicabilityPlanError::ContractIdentity {
                        detail: error.to_string(),
                    }
                })?,
                configured_threads,
                applicability,
                reason,
            });
        }
        if entries.is_empty() {
            return Err(PerfApplicabilityPlanError::ContractIdentity {
                detail: format!("{gate} has no canonical cells"),
            });
        }
        if disposition == DefaultFlipDisposition::RequiredForDefaultFlip {
            if let Some(primary_target_cell_width) = manifest_identity.primary_target_cell_width {
                if !entries.iter().any(|entry| {
                    u64::try_from(entry.configured_threads) == Ok(primary_target_cell_width)
                        && entry.applicability == PerfCellApplicability::Required
                }) {
                    return Err(
                        PerfApplicabilityPlanError::RequiredProfileBelowPrimaryTarget {
                            profile: profile_key,
                            gate,
                            primary_target_cell_width,
                            execution_capacity: Some(execution_capacity),
                            max_exercised_cell_width: Some(max_exercised_cell_width),
                        },
                    );
                }
            }
        }

        let gate_matrix_contract_sha256 = self.gate_contract_sha256(gate).map_err(|error| {
            PerfApplicabilityPlanError::ContractIdentity {
                detail: error.to_string(),
            }
        })?;
        let mut plan = PerfApplicabilityPlan {
            binding: PerfApplicabilityPlanBinding {
                schema_version: PERF_APPLICABILITY_PLAN_SCHEMA_VERSION.to_owned(),
                profile: profile_key,
                registry_schema_version: MACHINE_CLASS_REGISTRY_SCHEMA_VERSION.to_owned(),
                registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
                profile_contract_sha256: profile.contract_sha256().to_owned(),
                gate,
                normalized_perf_manifest_sha256: manifest_identity
                    .normalized_perf_manifest_sha256
                    .clone(),
                primary_target_cell_width: manifest_identity.primary_target_cell_width,
                matrix_contract_schema_version: Self::CONTRACT_SCHEMA_VERSION.to_owned(),
                gate_matrix_contract_sha256,
                applicability_plan_sha256: String::new(),
            },
            capacity_semantics: profile.capacity_semantics(),
            execution_capacity: Some(execution_capacity),
            default_flip_disposition: disposition,
            max_exercised_cell_width: Some(max_exercised_cell_width),
            cells: entries,
        };
        plan.binding.applicability_plan_sha256 = plan.contract_sha256();
        Ok(plan)
    }
}

fn canonical_cell_is_diagnostic(cell: &PerfCellSpec) -> bool {
    cell.gate == PerfGate::Qg10 || cell.metric == "tokenize_docs_per_second"
}

/// Distribution summary required for every timed cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub value: f64,
    pub p50: f64,
    pub median_ci95_low: f64,
    pub median_ci95_high: f64,
    pub p95: f64,
    pub p99: f64,
    pub mad: f64,
    /// Provenance only. Activation decisions are based on the median CI and
    /// paired A/A null floor, never on this coefficient of variation.
    pub cv_pct: f64,
    pub runs: usize,
}

/// How a raw measurement becomes the value compared between paired arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfMetricSemantics {
    /// Work completed per elapsed second. Both arms must report equal work.
    Throughput,
    /// Elapsed nanoseconds. Smaller treatment/control ratios are faster.
    Duration,
    /// A positive directly observed value where larger is better.
    GaugeHigherIsBetter,
    /// A positive directly observed value where smaller is better.
    GaugeLowerIsBetter,
}

impl PerfMetricSemantics {
    /// Whether a larger treatment/control ratio is favorable.
    #[must_use]
    pub const fn higher_is_better(self) -> bool {
        matches!(self, Self::Throughput | Self::GaugeHigherIsBetter)
    }
}

/// Versioned identity for the exact operation inside one timed scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfOperationScope {
    /// Stable operation identifier, such as `qg1.bulk_index_publish`.
    pub operation_id: String,
    /// Positive schema version for this exact timing boundary.
    pub version: u32,
    /// Metric-specific conversion applied to every raw record.
    pub semantics: PerfMetricSemantics,
    /// Human-readable unit for the derived absolute summaries.
    pub unit: String,
}

impl PerfOperationScope {
    /// Stable display identity used by diagnostics and artifact consumers.
    #[must_use]
    pub fn stable_id(&self) -> String {
        format!("{}@{}", self.operation_id, self.version)
    }

    fn validate(&self) -> Result<(), PairedEstimatorError> {
        if self.operation_id.trim().is_empty()
            || self.operation_id.len() > 128
            || self.unit.trim().is_empty()
            || self.unit.len() > 32
            || self.version == 0
        {
            return Err(PairedEstimatorError::InvalidScope {
                reason: format!(
                    "operation scope must have a bounded non-empty ID/unit and positive version: \
                     {:?}",
                    self
                ),
            });
        }
        Ok(())
    }
}

/// Exact prepared-corpus, query, and semantic-configuration identity.
///
/// This cell-level identity deliberately remains separate from
/// [`PerfSampleProvenance::corpus_sha256`], which names the invocation-wide
/// selection manifest. One invocation can prepare multiple exact corpus
/// subsets, ordered query manifests, and configuration contracts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfInputIdentity {
    /// SHA-256 of the exact ordered corpus prepared for this cell.
    pub prepared_corpus_sha256: String,
    /// SHA-256 of the exact ordered query manifest.
    pub query_manifest_sha256: String,
    /// SHA-256 of the semantic configuration shared by both engines.
    pub config_contract_sha256: String,
    /// SHA-256 of the cell-local sealed QG-6 semantic receipt contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_contract_sha256: Option<String>,
    /// Number of ordered query groups represented by the manifest.
    pub query_group_count: usize,
    /// Exact ordered group IDs emitted into both A/B and A/A raw streams.
    pub query_group_ids: Vec<u64>,
}

impl PerfInputIdentity {
    pub(crate) fn validate(&self) -> Result<(), PairedEstimatorError> {
        if !is_lower_hex_digest(&self.prepared_corpus_sha256)
            || !is_lower_hex_digest(&self.query_manifest_sha256)
            || !is_lower_hex_digest(&self.config_contract_sha256)
            || self
                .semantic_contract_sha256
                .as_deref()
                .is_none_or(|digest| !is_lower_hex_digest(digest))
            || self.query_group_count != QG6_QUERY_GROUPS
            || self.query_group_ids.as_slice() != QG6_QUERY_GROUP_IDS.as_slice()
        {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "prepared-input identity requires separate lowercase SHA-256 prepared \
                         corpus, ordered query, configuration, and semantic-contract hashes plus \
                         the exact sixteen canonical query-group IDs"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Domain-separated SHA-256 of all exact prepared inputs for telemetry.
    ///
    /// The three component hashes remain serialized separately for diagnosis
    /// and independent verification. This digest is only a concise log key.
    #[must_use]
    pub fn fingerprint_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.perf-input-identity.v1\0");
        hasher.update(b"prepared-corpus-sha256\0");
        hasher.update(self.prepared_corpus_sha256.as_bytes());
        hasher.update(b"ordered-query-manifest-sha256\0");
        hasher.update(self.query_manifest_sha256.as_bytes());
        hasher.update(b"config-contract-sha256\0");
        hasher.update(self.config_contract_sha256.as_bytes());
        hasher.update(b"semantic-contract-sha256\0");
        match self.semantic_contract_sha256.as_deref() {
            Some(digest) => {
                hasher.update([1]);
                hasher.update(digest.as_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(b"query-group-count\0");
        hasher.update(self.query_group_count.to_string().as_bytes());
        hasher.update(b"\0query-group-ids\0");
        for group_id in &self.query_group_ids {
            hasher.update(group_id.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        encoded
    }
}

/// Immutable execution context shared by every record in one paired run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfSampleProvenance {
    /// Unique identifier for this process-level measurement run.
    pub run_id: String,
    /// SHA-256 reported by the executing benchmark binary.
    pub executable_sha256: String,
    /// SHA-256 of the invocation-wide corpus-selection manifest.
    pub corpus_sha256: String,
    /// Exact cell-level prepared-input identity, separate from the
    /// invocation-wide corpus manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_identity: Option<PerfInputIdentity>,
    /// Stable worker or machine identity.
    pub worker_id: String,
    /// Exact Cargo profile label.
    pub build_profile: String,
}

impl PerfSampleProvenance {
    fn validate(&self) -> Result<(), PairedEstimatorError> {
        if self.run_id.trim().is_empty()
            || self.worker_id.trim().is_empty()
            || self.build_profile.trim().is_empty()
            || !is_lower_hex_digest(&self.executable_sha256)
            || !is_lower_hex_digest(&self.corpus_sha256)
        {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "paired samples require a run ID, worker, profile, and two lowercase \
                         SHA-256 values"
                    .to_owned(),
            });
        }
        if let Some(identity) = self.input_identity.as_ref() {
            identity.validate()?;
        }
        Ok(())
    }

    pub(crate) fn same_reproduction_context(&self, other: &Self) -> bool {
        self.executable_sha256 == other.executable_sha256
            && self.corpus_sha256 == other.corpus_sha256
            && self.input_identity == other.input_identity
            && self.worker_id == other.worker_id
            && self.build_profile == other.build_profile
    }
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Logical arm carried by one raw sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfSampleArm {
    /// Baseline or oracle arm.
    Control,
    /// Candidate or subject arm.
    Treatment,
}

/// Execution order inside one paired block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfSampleOrder {
    /// This sample executed first.
    First,
    /// This sample executed second.
    Second,
}

/// Whether a record belongs to warmup or to the decision sample set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfSamplePhase {
    /// Untimed-for-decision warmup record retained only for diagnostics.
    Warmup,
    /// Record admitted to the estimator.
    Measurement,
}

/// Compact per-row binding into the cell-local QG-6 semantic contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6SampleBinding {
    /// Stable redacted query ID resolved through `group_id`.
    pub query_id: String,
    /// Domain-separated digest of the validated role receipt sequence.
    pub result_sequence_sha256: String,
}

impl Qg6SampleBinding {
    fn validate(&self) -> bool {
        !self.query_id.is_empty()
            && self.query_id.len() <= 256
            && is_lower_hex_digest(&self.result_sequence_sha256)
    }
}

/// One bounded raw record emitted by the timing harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfRawSample {
    /// Stable pair identifier. Exactly one control and treatment must share it.
    pub block_id: u64,
    /// Globally unique sample identifier within the experiment.
    pub sample_id: u64,
    /// Baseline or candidate arm.
    pub arm: PerfSampleArm,
    /// First or second execution inside the block.
    pub order: PerfSampleOrder,
    /// Warmup or decision phase.
    pub phase: PerfSamplePhase,
    /// Exact versioned timing scope.
    pub scope: PerfOperationScope,
    /// Immutable process/corpus/worker provenance.
    pub provenance: PerfSampleProvenance,
    /// Monotonic timestamp relative to process start.
    pub started_ns: u64,
    /// Monotonic timestamp relative to process start.
    pub ended_ns: u64,
    /// Equal per-arm work denominator for throughput operations.
    pub work_units: Option<u64>,
    /// Equal per-arm byte denominator when applicable.
    pub byte_count: Option<u64>,
    /// Positive directly observed value for gauge operations.
    pub observed_value: Option<f64>,
    /// First-stage resampling unit for hierarchical estimands, such as the
    /// identity of one query inside a per-query latency cell. Flat estimands
    /// leave this unset; hierarchical estimation requires it on every sample.
    #[serde(default)]
    pub group_id: Option<u64>,
    /// Compact semantic binding for QG-6 only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qg6_sample_binding: Option<Qg6SampleBinding>,
}

impl PerfRawSample {
    fn validate_and_value(&self) -> Result<f64, PairedEstimatorError> {
        self.scope.validate()?;
        self.provenance.validate()?;
        match (
            self.provenance.input_identity.is_some(),
            self.qg6_sample_binding.as_ref(),
        ) {
            (true, Some(binding)) if binding.validate() => {}
            (true, _) => {
                return Err(PairedEstimatorError::InvalidProvenance {
                    reason: "prepared-input samples require one valid compact QG-6 result binding"
                        .to_owned(),
                });
            }
            (false, None) => {}
            (false, Some(_)) => {
                return Err(PairedEstimatorError::InvalidProvenance {
                    reason: "non-QG-6 samples cannot carry QG-6 result bindings".to_owned(),
                });
            }
        }
        if self.phase != PerfSamplePhase::Measurement {
            return Err(PairedEstimatorError::WarmupInDecisionSet {
                sample_id: self.sample_id,
            });
        }
        let elapsed_ns = self
            .ended_ns
            .checked_sub(self.started_ns)
            .filter(|value| *value > 0)
            .ok_or(PairedEstimatorError::InvalidTimestamp {
                sample_id: self.sample_id,
            })?;
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ns = elapsed_ns as f64;
        let value = match self.scope.semantics {
            PerfMetricSemantics::Throughput => {
                let work_units = self.work_units.filter(|value| *value > 0).ok_or_else(|| {
                    PairedEstimatorError::InvalidValue {
                        sample_id: self.sample_id,
                        reason: "throughput samples require positive work_units".to_owned(),
                    }
                })?;
                #[allow(clippy::cast_precision_loss)]
                let work_units = work_units as f64;
                work_units * 1_000_000_000.0 / elapsed_ns
            }
            PerfMetricSemantics::Duration => elapsed_ns,
            PerfMetricSemantics::GaugeHigherIsBetter | PerfMetricSemantics::GaugeLowerIsBetter => {
                self.observed_value
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| PairedEstimatorError::InvalidValue {
                        sample_id: self.sample_id,
                        reason: "gauge samples require a finite positive observed_value".to_owned(),
                    })?
            }
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(PairedEstimatorError::InvalidValue {
                sample_id: self.sample_id,
                reason: "derived sample value must be finite and positive".to_owned(),
            });
        }
        Ok(value)
    }
}

/// Predeclared validity thresholds for one paired estimator invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEstimatorConfig {
    /// Seed used only for deterministic paired bootstrap resampling.
    pub bootstrap_seed: u64,
    /// Number of paired bootstrap resamples.
    pub bootstrap_resamples: usize,
    /// Minimum complete measurement blocks.
    pub min_pairs: usize,
    /// Largest permitted first-arm count imbalance.
    pub max_order_imbalance: usize,
    /// Maximum absolute A/A robust center on the log-ratio scale.
    pub max_null_center_log: f64,
    /// Maximum A/A confidence-bound distance from zero on the log scale.
    pub max_null_ci_half_width_log: f64,
    /// Maximum robust A/A dispersion on the paired log-ratio scale.
    pub max_null_log_mad: f64,
    /// Maximum difference between control-first and treatment-first A/A centers.
    pub max_null_order_effect_log: f64,
    /// Maximum first-half versus second-half A/A drift.
    pub max_null_drift_log: f64,
    /// Small log-scale dead band used only when comparing summary directions.
    pub summary_direction_dead_band_log: f64,
    /// Maximum effect delta admitted between stable process invocations.
    pub max_reproduction_delta_log: f64,
}

impl PairedEstimatorConfig {
    /// Predeclared gate thresholds for harness-emitted QG evidence.
    ///
    /// These bounds are the bd-tqi3 estimator-repair defaults: at least
    /// [`PERF_MIN_RUNS`] complete pairs, [`PERF_BOOTSTRAP_RESAMPLES`]
    /// deterministic resamples, a 5% A/A center/dispersion budget, and a 10%
    /// A/A confidence half-width budget, all on the log-ratio scale. The
    /// order-imbalance allowance is one block because
    /// [`seeded_balanced_pair_order`] balances first arms to within one.
    #[must_use]
    pub fn predeclared(bootstrap_seed: u64) -> Self {
        Self {
            bootstrap_seed,
            bootstrap_resamples: PERF_BOOTSTRAP_RESAMPLES,
            min_pairs: PERF_MIN_RUNS,
            max_order_imbalance: 1,
            max_null_center_log: 1.05_f64.ln(),
            max_null_ci_half_width_log: 1.10_f64.ln(),
            max_null_log_mad: 1.05_f64.ln(),
            max_null_order_effect_log: 1.05_f64.ln(),
            max_null_drift_log: 1.05_f64.ln(),
            summary_direction_dead_band_log: 1.000_001_f64.ln(),
            max_reproduction_delta_log: 1.02_f64.ln(),
        }
    }

    /// Validate that every threshold was fixed to a finite, usable value.
    ///
    /// # Errors
    ///
    /// Returns a typed configuration error for undersampling or invalid bounds.
    pub fn validate(&self) -> Result<(), PairedEstimatorError> {
        let finite_non_negative = [
            self.max_null_center_log,
            self.max_null_ci_half_width_log,
            self.max_null_log_mad,
            self.max_null_order_effect_log,
            self.max_null_drift_log,
            self.summary_direction_dead_band_log,
            self.max_reproduction_delta_log,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0);
        if self.bootstrap_resamples < 100 || self.min_pairs < 4 || !finite_non_negative {
            return Err(PairedEstimatorError::InvalidConfig {
                reason: "paired estimation requires >=100 resamples, >=4 pairs, and finite \
                         non-negative thresholds"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Typed fail-closed input and verification errors for paired estimation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PairedEstimatorError {
    #[error("invalid paired estimator configuration: {reason}")]
    InvalidConfig { reason: String },
    #[error("invalid performance operation scope: {reason}")]
    InvalidScope { reason: String },
    #[error("invalid performance sample provenance: {reason}")]
    InvalidProvenance { reason: String },
    #[error("duplicate paired sample ID {sample_id}")]
    DuplicateSampleId { sample_id: u64 },
    #[error("warmup sample {sample_id} was passed to the decision estimator")]
    WarmupInDecisionSet { sample_id: u64 },
    #[error("sample {sample_id} has an invalid monotonic timestamp interval")]
    InvalidTimestamp { sample_id: u64 },
    #[error("sample {sample_id} has an invalid value: {reason}")]
    InvalidValue { sample_id: u64, reason: String },
    #[error(
        "paired block {block_id} is incomplete: controls={control_count}, \
         treatments={treatment_count}"
    )]
    IncompleteBlock {
        block_id: u64,
        control_count: usize,
        treatment_count: usize,
    },
    #[error("paired block {block_id} repeats the {arm:?} arm")]
    DuplicateArm { block_id: u64, arm: PerfSampleArm },
    #[error("paired block {block_id} does not contain one first and one second sample")]
    InvalidOrder { block_id: u64 },
    #[error("paired block {block_id} contains overlapping or reversed executions")]
    OverlappingSamples { block_id: u64 },
    #[error("paired block {block_id} mixes operation scopes")]
    ScopeMismatch { block_id: u64 },
    #[error("paired block {block_id} mixes execution provenance")]
    ProvenanceMismatch { block_id: u64 },
    #[error("sample {sample_id} requires a hierarchical group ID")]
    MissingGroupId { sample_id: u64 },
    #[error("paired block {block_id} mixes hierarchical group IDs")]
    GroupMismatch { block_id: u64 },
    #[error("paired block {block_id} mixes QG-6 query bindings")]
    Qg6BindingMismatch { block_id: u64 },
    #[error("paired block {block_id} compares different work or byte denominators")]
    WorkMismatch { block_id: u64 },
    #[error("paired experiment has only {actual} complete blocks; require {required}")]
    InsufficientPairs { actual: usize, required: usize },
    #[error("hierarchical experiment has only {actual} groups; require {required}")]
    InsufficientGroups { actual: usize, required: usize },
    #[error("hierarchical group {group_id} has only {actual} complete blocks; require {required}")]
    InsufficientGroupPairs {
        group_id: u64,
        actual: usize,
        required: usize,
    },
    #[error("A/B and A/A streams disagree on {field}")]
    CrossExperimentMismatch { field: &'static str },
    #[error("paired result no longer recomputes from its raw samples")]
    InconsistentSummary,
    #[error("paired reproduction context is incompatible: {field}")]
    ReproductionMismatch { field: &'static str },
    #[error("paired reproduction requires two distinct run IDs")]
    ReusedRunId,
    #[error("paired result has no decision-eligible effect")]
    NoDecision,
}

/// Robust treatment/control estimate reconstructed from complete paired blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEffectEstimate {
    /// Median paired log(treatment/control), the primary robust estimand.
    pub median_log_ratio: f64,
    /// Exponentiated primary estimand.
    pub treatment_over_control: f64,
    /// Lower 95% paired-bootstrap bound on the log scale.
    pub ci95_low_log: f64,
    /// Upper 95% paired-bootstrap bound on the log scale.
    pub ci95_high_log: f64,
    /// Exponentiated lower confidence bound.
    pub ci95_low_ratio: f64,
    /// Exponentiated upper confidence bound.
    pub ci95_high_ratio: f64,
    /// Median absolute deviation of paired log ratios.
    pub log_mad: f64,
    /// Arithmetic mean of paired log ratios, retained for algebraic checks.
    pub mean_log_ratio: f64,
    /// Ratio of marginal arm medians, diagnostic only.
    pub ratio_of_arm_medians: f64,
    /// Difference between mean paired logs and mean arm logs.
    pub algebraic_reconciliation_error: f64,
    /// Absolute value distribution for the control arm.
    pub control: DistributionSummary,
    /// Absolute value distribution for the treatment arm.
    pub treatment: DistributionSummary,
    /// Number of complete paired blocks.
    pub pair_count: usize,
    /// Blocks in which the control arm executed first.
    pub control_first_blocks: usize,
    /// Blocks in which the treatment arm executed first.
    pub treatment_first_blocks: usize,
    /// Difference between treatment-first and control-first log centers.
    pub order_effect_log: Option<f64>,
    /// Difference between second-half and first-half log centers.
    pub drift_log: f64,
}

/// Whether the estimator produced admissible diagnostic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairedEvidenceStatus {
    /// Pairing, null, and summary checks all passed.
    Valid,
    /// A/A noise, drift, carryover, or order balance invalidated the run.
    InvalidNull,
    /// The A/B stream itself violated a predeclared design check.
    InvalidExperiment,
    /// Paired and marginal summaries point in opposite directions.
    ContradictorySummaries,
}

/// Claim eligibility is deliberately separate from measured diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairedClaimState {
    /// A downstream gate may apply its predeclared Allow/Block threshold.
    EligibleForDecision,
    /// Persist diagnostics, but emit no performance claim.
    NoDecision,
}

/// Stable reason emitted by a paired estimator decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedEstimatorReason {
    /// Machine-readable reason code.
    pub code: String,
    /// Bounded operator-facing explanation.
    pub message: String,
}

/// Complete replayable paired estimator output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedExperimentResult {
    /// Estimator schema identifier.
    pub schema_version: String,
    /// Exact operation shared by A/B and A/A.
    pub scope: PerfOperationScope,
    /// Immutable execution context.
    pub provenance: PerfSampleProvenance,
    /// Predeclared estimator thresholds.
    pub config: PairedEstimatorConfig,
    /// Candidate-versus-control estimate.
    pub effect: PairedEffectEstimate,
    /// Same-operation A/A estimate.
    pub null: PairedEffectEstimate,
    /// Diagnostic validity.
    pub status: PairedEvidenceStatus,
    /// Whether a downstream decision is permitted.
    pub claim_state: PairedClaimState,
    /// Stable reasons explaining invalid or contradictory evidence.
    pub reasons: Vec<PairedEstimatorReason>,
    /// Bounded raw A/B records from which `effect` recomputes.
    pub effect_samples: Vec<PerfRawSample>,
    /// Bounded raw A/A records from which `null` recomputes.
    pub null_samples: Vec<PerfRawSample>,
}

impl PairedExperimentResult {
    /// Recompute every estimate and decision from the retained raw records.
    ///
    /// # Errors
    ///
    /// Returns [`PairedEstimatorError::InconsistentSummary`] on any mismatch.
    pub fn verify_recomputed(&self) -> Result<(), PairedEstimatorError> {
        if self.config != PairedEstimatorConfig::predeclared(self.config.bootstrap_seed) {
            return Err(PairedEstimatorError::InvalidConfig {
                reason: "persisted evidence must use the exact predeclared estimator thresholds; \
                         only the bootstrap seed may vary"
                    .to_owned(),
            });
        }
        let recomputed =
            estimate_paired_experiment(&self.effect_samples, &self.null_samples, &self.config)?;
        if recomputed == *self {
            Ok(())
        } else {
            Err(PairedEstimatorError::InconsistentSummary)
        }
    }

    /// Absolute log-effect delta against an independent process invocation.
    ///
    /// # Errors
    ///
    /// Rejects reused run IDs, different scopes/configs, incompatible
    /// executable/corpus/worker/profile provenance, or a `NoDecision` input.
    pub fn reproduction_delta_log(&self, other: &Self) -> Result<f64, PairedEstimatorError> {
        if self.provenance.run_id == other.provenance.run_id {
            return Err(PairedEstimatorError::ReusedRunId);
        }
        if self.scope != other.scope {
            return Err(PairedEstimatorError::ReproductionMismatch {
                field: "operation scope",
            });
        }
        if self.config != other.config {
            return Err(PairedEstimatorError::ReproductionMismatch {
                field: "estimator configuration",
            });
        }
        if !self.provenance.same_reproduction_context(&other.provenance) {
            return Err(PairedEstimatorError::ReproductionMismatch {
                field: "execution context",
            });
        }
        if self.claim_state != PairedClaimState::EligibleForDecision
            || other.claim_state != PairedClaimState::EligibleForDecision
        {
            return Err(PairedEstimatorError::NoDecision);
        }
        Ok((self.effect.median_log_ratio - other.effect.median_log_ratio).abs())
    }

    /// Whether an independent process replay meets the predeclared tolerance.
    ///
    /// # Errors
    ///
    /// Propagates incompatibility or `NoDecision` errors from
    /// [`Self::reproduction_delta_log`].
    pub fn reproduces_within(&self, other: &Self) -> Result<bool, PairedEstimatorError> {
        Ok(self.reproduction_delta_log(other)? <= self.config.max_reproduction_delta_log)
    }
}

#[derive(Debug)]
pub struct ValidatedPair {
    pub block_id: u64,
    pub group_id: Option<u64>,
    pub control_value: f64,
    pub treatment_value: f64,
    pub log_ratio: f64,
    pub control_first: bool,
}

type PairedStream = (
    PerfOperationScope,
    PerfSampleProvenance,
    Vec<ValidatedPair>,
    Vec<PerfRawSample>,
);

type PairedBlocks = (
    Option<PerfOperationScope>,
    Option<PerfSampleProvenance>,
    Vec<ValidatedPair>,
    Vec<PerfRawSample>,
);

/// Structural validation shared by the flat and hierarchical estimators.
///
/// Enforces every per-sample and per-block law except the stream-level
/// minimum pair count, which each caller owns.
pub fn validate_paired_blocks(
    samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
) -> Result<PairedBlocks, PairedEstimatorError> {
    config.validate()?;
    let mut sample_ids = BTreeSet::new();
    let mut blocks = BTreeMap::<u64, (Option<&PerfRawSample>, Option<&PerfRawSample>)>::new();
    let mut stream_scope: Option<&PerfOperationScope> = None;
    let mut stream_provenance: Option<&PerfSampleProvenance> = None;

    for sample in samples {
        let _ = sample.validate_and_value()?;
        if !sample_ids.insert(sample.sample_id) {
            return Err(PairedEstimatorError::DuplicateSampleId {
                sample_id: sample.sample_id,
            });
        }
        if let Some(scope) = stream_scope {
            if scope != &sample.scope {
                return Err(PairedEstimatorError::CrossExperimentMismatch {
                    field: "operation scope within one stream",
                });
            }
        } else {
            stream_scope = Some(&sample.scope);
        }
        if let Some(provenance) = stream_provenance {
            if provenance != &sample.provenance {
                return Err(PairedEstimatorError::CrossExperimentMismatch {
                    field: "provenance within one stream",
                });
            }
        } else {
            stream_provenance = Some(&sample.provenance);
        }

        let entry = blocks.entry(sample.block_id).or_default();
        let slot = match sample.arm {
            PerfSampleArm::Control => &mut entry.0,
            PerfSampleArm::Treatment => &mut entry.1,
        };
        if slot.replace(sample).is_some() {
            return Err(PairedEstimatorError::DuplicateArm {
                block_id: sample.block_id,
                arm: sample.arm,
            });
        }
    }

    let mut pairs = Vec::with_capacity(blocks.len());
    for (block_id, (control, treatment)) in blocks {
        let control_count = usize::from(control.is_some());
        let treatment_count = usize::from(treatment.is_some());
        let (Some(control), Some(treatment)) = (control, treatment) else {
            return Err(PairedEstimatorError::IncompleteBlock {
                block_id,
                control_count,
                treatment_count,
            });
        };
        if control.scope != treatment.scope {
            return Err(PairedEstimatorError::ScopeMismatch { block_id });
        }
        if control.provenance != treatment.provenance {
            return Err(PairedEstimatorError::ProvenanceMismatch { block_id });
        }
        if control.work_units != treatment.work_units || control.byte_count != treatment.byte_count
        {
            return Err(PairedEstimatorError::WorkMismatch { block_id });
        }
        if control.group_id != treatment.group_id {
            return Err(PairedEstimatorError::GroupMismatch { block_id });
        }
        if control
            .qg6_sample_binding
            .as_ref()
            .map(|binding| binding.query_id.as_str())
            != treatment
                .qg6_sample_binding
                .as_ref()
                .map(|binding| binding.query_id.as_str())
        {
            return Err(PairedEstimatorError::Qg6BindingMismatch { block_id });
        }
        if control.order == treatment.order {
            return Err(PairedEstimatorError::InvalidOrder { block_id });
        }
        let control_first = control.order == PerfSampleOrder::First;
        let sequential = if control_first {
            control.ended_ns <= treatment.started_ns
        } else {
            treatment.ended_ns <= control.started_ns
        };
        if !sequential {
            return Err(PairedEstimatorError::OverlappingSamples { block_id });
        }
        let control_value = control.validate_and_value()?;
        let treatment_value = treatment.validate_and_value()?;
        let log_ratio = (treatment_value / control_value).ln();
        if !log_ratio.is_finite() {
            return Err(PairedEstimatorError::InvalidValue {
                sample_id: treatment.sample_id,
                reason: "paired log ratio must be finite".to_owned(),
            });
        }
        pairs.push(ValidatedPair {
            block_id,
            group_id: control.group_id,
            control_value,
            treatment_value,
            log_ratio,
            control_first,
        });
    }

    let mut raw = samples.to_vec();
    raw.sort_by_key(|sample| {
        let order = match sample.order {
            PerfSampleOrder::First => 0_u8,
            PerfSampleOrder::Second => 1_u8,
        };
        (sample.block_id, order, sample.arm, sample.sample_id)
    });
    Ok((
        stream_scope.cloned(),
        stream_provenance.cloned(),
        pairs,
        raw,
    ))
}

fn validate_paired_stream(
    samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
) -> Result<PairedStream, PairedEstimatorError> {
    let (scope, provenance, pairs, raw) = validate_paired_blocks(samples, config)?;
    if pairs.len() < config.min_pairs {
        return Err(PairedEstimatorError::InsufficientPairs {
            actual: pairs.len(),
            required: config.min_pairs,
        });
    }
    let scope = scope.ok_or(PairedEstimatorError::InsufficientPairs {
        actual: 0,
        required: config.min_pairs,
    })?;
    let provenance = provenance.ok_or(PairedEstimatorError::InsufficientPairs {
        actual: 0,
        required: config.min_pairs,
    })?;
    Ok((scope, provenance, pairs, raw))
}

fn summarize_pairs(
    pairs: &[ValidatedPair],
    config: &PairedEstimatorConfig,
    seed_domain: u64,
) -> Result<PairedEffectEstimate, PairedEstimatorError> {
    let control_values = pairs
        .iter()
        .map(|pair| pair.control_value)
        .collect::<Vec<_>>();
    let treatment_values = pairs
        .iter()
        .map(|pair| pair.treatment_value)
        .collect::<Vec<_>>();
    let log_ratios = pairs.iter().map(|pair| pair.log_ratio).collect::<Vec<_>>();
    let mut sorted_logs = log_ratios.clone();
    sorted_logs.sort_unstable_by(f64::total_cmp);
    let median_log_ratio = median_sorted(&sorted_logs);
    let (ci95_low_log, ci95_high_log) = bootstrap_log_median_ci95(
        &log_ratios,
        config.bootstrap_seed ^ seed_domain,
        config.bootstrap_resamples,
    );
    let mut log_deviations = log_ratios
        .iter()
        .map(|value| (value - median_log_ratio).abs())
        .collect::<Vec<_>>();
    log_deviations.sort_unstable_by(f64::total_cmp);
    let control = DistributionSummary::from_samples(&control_values).map_err(|error| {
        PairedEstimatorError::InvalidValue {
            sample_id: 0,
            reason: error.to_string(),
        }
    })?;
    let treatment = DistributionSummary::from_samples(&treatment_values).map_err(|error| {
        PairedEstimatorError::InvalidValue {
            sample_id: 0,
            reason: error.to_string(),
        }
    })?;
    #[allow(clippy::cast_precision_loss)]
    let pair_count = pairs.len() as f64;
    let mean_log_ratio = log_ratios.iter().sum::<f64>() / pair_count;
    let mean_control_log = control_values.iter().map(|value| value.ln()).sum::<f64>() / pair_count;
    let mean_treatment_log =
        treatment_values.iter().map(|value| value.ln()).sum::<f64>() / pair_count;
    let control_first_logs = pairs
        .iter()
        .filter(|pair| pair.control_first)
        .map(|pair| pair.log_ratio)
        .collect::<Vec<_>>();
    let treatment_first_logs = pairs
        .iter()
        .filter(|pair| !pair.control_first)
        .map(|pair| pair.log_ratio)
        .collect::<Vec<_>>();
    let order_effect_log = median_of(&treatment_first_logs)
        .zip(median_of(&control_first_logs))
        .map(|(treatment_first, control_first)| treatment_first - control_first);
    let midpoint = pairs.len() / 2;
    let drift_log = median_of(
        &pairs[midpoint..]
            .iter()
            .map(|pair| pair.log_ratio)
            .collect::<Vec<_>>(),
    )
    .zip(median_of(
        &pairs[..midpoint]
            .iter()
            .map(|pair| pair.log_ratio)
            .collect::<Vec<_>>(),
    ))
    .map_or(0.0, |(second, first)| second - first);

    Ok(PairedEffectEstimate {
        median_log_ratio,
        treatment_over_control: median_log_ratio.exp(),
        ci95_low_log,
        ci95_high_log,
        ci95_low_ratio: ci95_low_log.exp(),
        ci95_high_ratio: ci95_high_log.exp(),
        log_mad: median_sorted(&log_deviations),
        mean_log_ratio,
        ratio_of_arm_medians: treatment.p50 / control.p50,
        algebraic_reconciliation_error: mean_log_ratio - (mean_treatment_log - mean_control_log),
        control,
        treatment,
        pair_count: pairs.len(),
        control_first_blocks: control_first_logs.len(),
        treatment_first_blocks: treatment_first_logs.len(),
        order_effect_log,
        drift_log,
    })
}

fn median_of(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    Some(median_sorted(&sorted))
}

fn bootstrap_log_median_ci95(samples: &[f64], mut seed: u64, resamples: usize) -> (f64, f64) {
    debug_assert!(!samples.is_empty());
    let sample_count = u64::try_from(samples.len()).expect("sample count fits u64");
    for sample in samples {
        seed = splitmix64(seed ^ sample.to_bits());
    }
    let mut scratch = Vec::with_capacity(samples.len());
    let mut medians = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        scratch.clear();
        for _ in 0..samples.len() {
            seed = splitmix64(seed);
            let index = usize::try_from(seed % sample_count).expect("sample modulus fits usize");
            scratch.push(samples[index]);
        }
        scratch.sort_unstable_by(f64::total_cmp);
        medians.push(median_sorted(&scratch));
    }
    medians.sort_unstable_by(f64::total_cmp);
    (percentile(&medians, 0.025), percentile(&medians, 0.975))
}

fn direction_conflicts(effect: &PairedEffectEstimate, dead_band: f64) -> bool {
    let marginal_log_ratio = effect.ratio_of_arm_medians.ln();
    effect.median_log_ratio.abs() > dead_band
        && marginal_log_ratio.abs() > dead_band
        && effect.median_log_ratio.is_sign_positive() != marginal_log_ratio.is_sign_positive()
}

fn push_reason(reasons: &mut Vec<PairedEstimatorReason>, code: &str, message: impl Into<String>) {
    reasons.push(PairedEstimatorReason {
        code: code.to_owned(),
        message: message.into(),
    });
}

/// Estimate one candidate/control stream beside its same-operation A/A null.
///
/// Both streams retain raw samples even when the null is invalid. Structural
/// defects such as missing IDs or mixed scopes return an error; statistical
/// invalidity returns a replayable [`PairedExperimentResult`] whose
/// [`PairedClaimState`] is [`PairedClaimState::NoDecision`].
///
/// # Errors
///
/// Returns a typed fail-closed error for malformed pairs, mixed scopes or
/// provenance, undersampling, and invalid raw values.
pub fn estimate_paired_experiment(
    effect_samples: &[PerfRawSample],
    null_samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
) -> Result<PairedExperimentResult, PairedEstimatorError> {
    let (scope, provenance, effect_pairs, effect_raw) =
        validate_paired_stream(effect_samples, config)?;
    let (null_scope, null_provenance, null_pairs, null_raw) =
        validate_paired_stream(null_samples, config)?;
    if scope != null_scope {
        return Err(PairedEstimatorError::CrossExperimentMismatch {
            field: "operation scope",
        });
    }
    if provenance != null_provenance {
        return Err(PairedEstimatorError::CrossExperimentMismatch {
            field: "provenance",
        });
    }
    let mut global_ids = BTreeSet::new();
    for sample in effect_raw.iter().chain(&null_raw) {
        if !global_ids.insert(sample.sample_id) {
            return Err(PairedEstimatorError::DuplicateSampleId {
                sample_id: sample.sample_id,
            });
        }
    }

    let effect = summarize_pairs(&effect_pairs, config, 0x4142_5f45_4646_4543)?;
    let null = summarize_pairs(&null_pairs, config, 0x4141_5f4e_554c_4c00)?;
    let mut null_invalid = false;
    let mut experiment_invalid = false;
    let mut contradictory = false;
    let mut reasons = Vec::new();
    let null_ci_half_width = null.ci95_low_log.abs().max(null.ci95_high_log.abs());
    if !(null.ci95_low_log <= 0.0 && 0.0 <= null.ci95_high_log)
        || null.median_log_ratio.abs() > config.max_null_center_log
    {
        null_invalid = true;
        push_reason(
            &mut reasons,
            "paired.null_center_invalid",
            format!(
                "A/A center {:.6} with CI [{:.6}, {:.6}] exceeds the predeclared null center",
                null.median_log_ratio, null.ci95_low_log, null.ci95_high_log
            ),
        );
    }
    if null_ci_half_width > config.max_null_ci_half_width_log {
        null_invalid = true;
        push_reason(
            &mut reasons,
            "paired.null_too_wide",
            format!(
                "A/A log-CI half width {null_ci_half_width:.6} exceeds {:.6}",
                config.max_null_ci_half_width_log
            ),
        );
    }
    if null.log_mad > config.max_null_log_mad {
        null_invalid = true;
        push_reason(
            &mut reasons,
            "paired.null_dispersion",
            format!(
                "A/A log-MAD {:.6} exceeds {:.6}",
                null.log_mad, config.max_null_log_mad
            ),
        );
    }
    let null_order_imbalance = null
        .control_first_blocks
        .abs_diff(null.treatment_first_blocks);
    if null_order_imbalance > config.max_order_imbalance {
        null_invalid = true;
        push_reason(
            &mut reasons,
            "paired.null_order_imbalance",
            format!(
                "A/A first-arm imbalance {null_order_imbalance} exceeds {}",
                config.max_order_imbalance
            ),
        );
    }
    match null.order_effect_log {
        Some(effect) if effect.abs() <= config.max_null_order_effect_log => {}
        Some(effect) => {
            null_invalid = true;
            push_reason(
                &mut reasons,
                "paired.null_order_effect",
                format!(
                    "A/A order effect {effect:.6} exceeds {:.6}",
                    config.max_null_order_effect_log
                ),
            );
        }
        None => {
            null_invalid = true;
            push_reason(
                &mut reasons,
                "paired.null_order_unobserved",
                "A/A stream did not execute both randomized orders",
            );
        }
    }
    if null.drift_log.abs() > config.max_null_drift_log {
        null_invalid = true;
        push_reason(
            &mut reasons,
            "paired.null_drift",
            format!(
                "A/A first/second-half drift {:.6} exceeds {:.6}",
                null.drift_log, config.max_null_drift_log
            ),
        );
    }
    let effect_order_imbalance = effect
        .control_first_blocks
        .abs_diff(effect.treatment_first_blocks);
    if effect_order_imbalance > config.max_order_imbalance {
        experiment_invalid = true;
        push_reason(
            &mut reasons,
            "paired.effect_order_imbalance",
            format!(
                "A/B first-arm imbalance {effect_order_imbalance} exceeds {}",
                config.max_order_imbalance
            ),
        );
    }
    if direction_conflicts(&effect, config.summary_direction_dead_band_log) {
        contradictory = true;
        push_reason(
            &mut reasons,
            "paired.absolute_direction_conflict",
            format!(
                "paired ratio {:.6} and ratio-of-arm-medians {:.6} point in opposite directions",
                effect.treatment_over_control, effect.ratio_of_arm_medians
            ),
        );
    }
    if effect.algebraic_reconciliation_error.abs() > 1.0e-12
        || null.algebraic_reconciliation_error.abs() > 1.0e-12
    {
        experiment_invalid = true;
        push_reason(
            &mut reasons,
            "paired.algebraic_reconciliation_failed",
            "mean paired log effect does not reconcile with the same raw arm values",
        );
    }

    let status = if null_invalid {
        PairedEvidenceStatus::InvalidNull
    } else if experiment_invalid {
        PairedEvidenceStatus::InvalidExperiment
    } else if contradictory {
        PairedEvidenceStatus::ContradictorySummaries
    } else {
        PairedEvidenceStatus::Valid
    };
    let claim_state = if status == PairedEvidenceStatus::Valid {
        PairedClaimState::EligibleForDecision
    } else {
        PairedClaimState::NoDecision
    };
    Ok(PairedExperimentResult {
        schema_version: PAIRED_ESTIMATOR_SCHEMA_VERSION.to_owned(),
        scope,
        provenance,
        config: config.clone(),
        effect,
        null,
        status,
        claim_state,
        reasons,
        effect_samples: effect_raw,
        null_samples: null_raw,
    })
}

/// Produce a deterministic, balanced randomized first-arm schedule.
///
/// # Errors
///
/// Requires at least two blocks and a count representable by the deterministic
/// shuffle.
pub fn seeded_balanced_pair_order(
    pair_count: usize,
    mut seed: u64,
) -> Result<Vec<PerfSampleArm>, PairedEstimatorError> {
    if pair_count < 2 {
        return Err(PairedEstimatorError::InvalidConfig {
            reason: "paired order schedule requires at least two blocks".to_owned(),
        });
    }
    let mut first_arms = (0..pair_count)
        .map(|index| {
            if index < pair_count / 2 {
                PerfSampleArm::Control
            } else {
                PerfSampleArm::Treatment
            }
        })
        .collect::<Vec<_>>();
    for index in (1..pair_count).rev() {
        seed = splitmix64(seed);
        let modulus =
            u64::try_from(index + 1).map_err(|_| PairedEstimatorError::InvalidConfig {
                reason: "pair count does not fit deterministic shuffle modulus".to_owned(),
            })?;
        let swap_index =
            usize::try_from(seed % modulus).map_err(|_| PairedEstimatorError::InvalidConfig {
                reason: "shuffle index does not fit usize".to_owned(),
            })?;
        first_arms.swap(index, swap_index);
    }
    Ok(first_arms)
}

impl DistributionSummary {
    /// Summarize finite non-negative samples. The result records fewer than ten
    /// runs but cannot be gate-activated until [`Self::sampled_for_activation`]
    /// is true.
    ///
    /// # Errors
    ///
    /// Rejects an empty set, NaN/infinite values, and negative durations or
    /// counters.
    pub fn from_samples(samples: &[f64]) -> Result<Self, GauntletError> {
        if samples.is_empty()
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || *sample < 0.0)
        {
            return Err(GauntletError::InvalidCampaign {
                reason: "performance samples must be finite, non-negative, and non-empty"
                    .to_owned(),
            });
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let p50 = median_sorted(&sorted);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let variance = sorted
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / sorted.len() as f64;
        let cv_pct = if mean == 0.0 {
            0.0
        } else {
            variance.sqrt() / mean * 100.0
        };
        let mut deviations = sorted
            .iter()
            .map(|sample| (sample - p50).abs())
            .collect::<Vec<_>>();
        deviations.sort_unstable_by(f64::total_cmp);
        let (median_ci95_low, median_ci95_high) = bootstrap_median_ci95(samples);
        Ok(Self {
            value: p50,
            p50,
            median_ci95_low,
            median_ci95_high,
            p95: percentile(&sorted, 0.95),
            p99: percentile(&sorted, 0.99),
            mad: median_sorted(&deviations),
            cv_pct,
            runs: sorted.len(),
        })
    }

    /// Whether this distribution has the minimum independent sample count.
    ///
    /// This deliberately ignores `cv_pct`: the ratchet decides paired claims
    /// from the bootstrap median CI and the same-invocation A/A null floor.
    #[must_use]
    pub fn sampled_for_activation(&self) -> bool {
        self.runs >= PERF_MIN_RUNS
            && self.median_ci95_low.is_finite()
            && self.median_ci95_high.is_finite()
            && self.median_ci95_low <= self.p50
            && self.p50 <= self.median_ci95_high
    }
}

fn bootstrap_median_ci95(samples: &[f64]) -> (f64, f64) {
    debug_assert!(!samples.is_empty());
    let sample_count = u64::try_from(samples.len()).expect("sample count fits u64");
    let mut seed = 0x6a09_e667_f3bc_c909_u64 ^ sample_count;
    for sample in samples {
        seed = splitmix64(seed ^ sample.to_bits());
    }

    let mut resample = Vec::with_capacity(samples.len());
    let mut medians = Vec::with_capacity(PERF_BOOTSTRAP_RESAMPLES);
    for _ in 0..PERF_BOOTSTRAP_RESAMPLES {
        resample.clear();
        for _ in 0..samples.len() {
            seed = splitmix64(seed);
            let index = usize::try_from(seed % sample_count).expect("sample modulus fits usize");
            resample.push(samples[index]);
        }
        resample.sort_unstable_by(f64::total_cmp);
        medians.push(median_sorted(&resample));
    }
    medians.sort_unstable_by(f64::total_cmp);
    (percentile(&medians, 0.025), percentile(&medians, 0.975))
}

pub const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let scaled = (sorted.len() - 1) as f64 * quantile;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = scaled.round() as usize;
    sorted[index]
}

pub fn median_sorted(sorted: &[f64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let midpoint = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        sorted[midpoint - 1] / 2.0 + sorted[midpoint] / 2.0
    } else {
        sorted[midpoint]
    }
}

/// Closed producer operating-system identity for persisted evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerfProducerOs {
    /// Linux producer, where the effective CPU allow-list is mandatory.
    Linux,
    /// macOS producer, where Linux affinity evidence is inapplicable.
    Macos,
}

impl PerfProducerOs {
    /// Stable serialized operating-system label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Macos => "macos",
        }
    }

    fn current() -> Option<Self> {
        match std::env::consts::OS {
            "linux" => Some(Self::Linux),
            "macos" => Some(Self::Macos),
            _ => None,
        }
    }
}

/// Auditable host topology and effective execution width for one benchmark
/// artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfExecutionProvenance {
    /// Benchmark-reported host name used only as a diagnostic correlation
    /// label. Receipt-admitted hardware, topology, and execution facts—not
    /// this hostname string—authorize a machine profile.
    pub host_identity: String,
    /// Operating system of the producer that captured this persisted
    /// provenance. Validation must never depend on the reader's host OS.
    pub producer_os: PerfProducerOs,
    /// Host-wide physical core count.
    pub physical_cores: usize,
    /// Host-wide logical hardware-thread count.
    pub logical_threads: usize,
    /// Concurrency available to the benchmark process after scheduler and
    /// cgroup constraints.
    pub process_available_threads: usize,
    /// Registry-admitted execution capacity for the immutable profile.
    ///
    /// This is the profile's verified worker envelope, not the widest cell
    /// selected by this invocation.
    pub execution_capacity: u64,
    /// Widest canonical cell the applicability plan permits this profile to
    /// exercise for the gate.
    ///
    /// A scheduler profile may intentionally have a larger execution capacity
    /// than this literal matrix width (for example capacity 10 with maximum
    /// exercised width 8).
    pub max_exercised_cell_width: u64,
    /// Exact engine thread-width knobs configured by the selected cells.
    ///
    /// This is configuration provenance, never a claim that every configured
    /// worker performed useful work.
    pub configured_engine_thread_widths: Vec<usize>,
    /// ISA features detected at runtime on the executing host.
    pub runtime_detected_isa: Vec<String>,
    /// Effective Linux `Cpus_allowed_list`, when the platform exposes it.
    pub cpu_affinity_allowed_list: Option<String>,
    /// Effective affinity, cpuset, or scheduler cap when narrower than the
    /// host-wide logical topology.
    pub affinity_or_cpuset_cap: Option<String>,
}

impl PerfExecutionProvenance {
    /// Capture host-wide topology beside the exact widths selected by this
    /// invocation.
    #[must_use]
    pub fn capture(
        execution_capacity: u64,
        max_exercised_cell_width: u64,
        configured_engine_thread_widths: impl IntoIterator<Item = usize>,
    ) -> Self {
        let (physical_cores, logical_threads) = host_cpu_topology()
            .expect("performance evidence requires host physical/logical CPU topology");
        let process_available_threads = std::thread::available_parallelism().map_or(1, usize::from);
        let cpu_affinity_allowed_list = linux_cpu_allowed_list();
        let allowed_threads = cpu_affinity_allowed_list
            .as_deref()
            .and_then(parse_cpu_list_count);
        let affinity_or_cpuset_cap = if allowed_threads.is_some_and(|count| count < logical_threads)
        {
            Some(format!(
                "Cpus_allowed_list={} ({} of {} host logical threads)",
                cpu_affinity_allowed_list
                    .as_deref()
                    .expect("allowed-thread count came from an affinity list"),
                allowed_threads.expect("narrow affinity has a parsed count"),
                logical_threads,
            ))
        } else if process_available_threads < logical_threads {
            Some(format!(
                "available_parallelism={process_available_threads} of \
                     {logical_threads} host logical threads"
            ))
        } else {
            None
        };
        let mut configured_engine_thread_widths = configured_engine_thread_widths
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        configured_engine_thread_widths.retain(|threads| *threads > 0);
        let provenance = Self {
            host_identity: host_identity()
                .expect("performance evidence requires a non-empty host identity"),
            producer_os: PerfProducerOs::current()
                .expect("performance evidence supports only Linux and macOS producers"),
            physical_cores,
            logical_threads,
            process_available_threads,
            execution_capacity,
            max_exercised_cell_width,
            configured_engine_thread_widths,
            runtime_detected_isa: runtime_detected_isa(),
            cpu_affinity_allowed_list,
            affinity_or_cpuset_cap,
        };
        assert!(
            provenance.is_complete(),
            "performance execution provenance is incomplete: {provenance:?}"
        );
        provenance
    }

    /// Whether all fields required to interpret a scaling result are present.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.host_identity.trim().is_empty()
            && self.physical_cores > 0
            && self.logical_threads >= self.physical_cores
            && self.process_available_threads > 0
            && self.process_available_threads <= self.logical_threads
            && self.execution_capacity > 0
            && self.max_exercised_cell_width > 0
            && self.max_exercised_cell_width <= self.execution_capacity
            && u64::try_from(self.process_available_threads)
                .is_ok_and(|available| available >= self.execution_capacity)
            && !self.configured_engine_thread_widths.is_empty()
            && self.configured_engine_thread_widths.iter().all(|threads| {
                *threads > 0
                    && u64::try_from(*threads)
                        .is_ok_and(|width| width <= self.max_exercised_cell_width)
            })
            && self
                .configured_engine_thread_widths
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && runtime_isa_is_normalized(&self.runtime_detected_isa)
            && match self.producer_os {
                PerfProducerOs::Linux => self
                    .cpu_affinity_allowed_list
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                PerfProducerOs::Macos => true,
            }
    }

    /// Whether process-available concurrency represents the registry's
    /// capacity semantics without collapsing scheduler workers into host CPUs.
    #[must_use]
    pub fn matches_capacity_semantics(&self, semantics: ExecutionCapacitySemantics) -> bool {
        u64::try_from(self.process_available_threads).is_ok_and(|available| match semantics {
            ExecutionCapacitySemantics::SchedulerWorkers => available >= self.execution_capacity,
            ExecutionCapacitySemantics::PhysicalCores
            | ExecutionCapacitySemantics::LogicalThreads
            | ExecutionCapacitySemantics::DiagnosticWorkerBudget => {
                available == self.execution_capacity
            }
        })
    }
}

fn host_identity() -> Option<String> {
    fs::read_to_string("/etc/hostname")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("HOSTNAME")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            Command::new("hostname")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
}

fn host_cpu_topology() -> Option<(usize, usize)> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| parse_linux_cpu_topology(&contents))
    }
    #[cfg(target_os = "macos")]
    {
        let physical = sysctl_usize("hw.physicalcpu")?;
        let logical = sysctl_usize("hw.logicalcpu")?;
        Some((physical, logical))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn parse_linux_cpu_topology(cpuinfo: &str) -> Option<(usize, usize)> {
    let mut logical_threads = 0_usize;
    let mut cores = BTreeSet::new();
    for record in cpuinfo.split("\n\n") {
        let mut has_processor = false;
        let mut physical_id = None;
        let mut core_id = None;
        for line in record.lines() {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            match name.trim() {
                "processor" => has_processor = value.trim().parse::<usize>().is_ok(),
                "physical id" => physical_id = value.trim().parse::<usize>().ok(),
                "core id" => core_id = value.trim().parse::<usize>().ok(),
                _ => {}
            }
        }
        if has_processor {
            logical_threads = logical_threads.saturating_add(1);
            if let (Some(package), Some(core)) = (physical_id, core_id) {
                cores.insert((package, core));
            }
        }
    }
    (!cores.is_empty() && logical_threads >= cores.len()).then_some((cores.len(), logical_threads))
}

#[cfg(target_os = "macos")]
fn sysctl_usize(name: &str) -> Option<usize> {
    Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn linux_cpu_allowed_list() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    line.strip_prefix("Cpus_allowed_list:")
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                })
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn parse_cpu_list_count(value: &str) -> Option<usize> {
    parse_cpu_list_ids(value).map(|ids| ids.len())
}

/// Parse a Linux CPU-list projection into its exact unique logical CPU IDs.
pub fn parse_cpu_list_ids(value: &str) -> Option<BTreeSet<u64>> {
    let mut ids = BTreeSet::new();
    for component in value.split(',') {
        let component = component.trim();
        let (first, last) = component
            .split_once('-')
            .map_or((component, component), |(first, last)| (first, last));
        let first = first.parse::<u64>().ok()?;
        let last = last.parse::<u64>().ok()?;
        if last < first {
            return None;
        }
        let width = last.checked_sub(first)?.checked_add(1)?;
        if width > 1_048_576 || u64::try_from(ids.len()).ok()?.checked_add(width)? > 1_048_576 {
            return None;
        }
        for id in first..=last {
            if !ids.insert(id) {
                return None;
            }
        }
    }
    (!ids.is_empty()).then_some(ids)
}

pub fn runtime_detected_isa() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(target_os = "linux")]
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let flags = cpuinfo.lines().find_map(|line| {
            let (name, values) = line.split_once(':')?;
            matches!(name.trim(), "flags" | "Features").then_some(values)
        });
        if let Some(flags) = flags {
            let flags = flags.split_ascii_whitespace().collect::<BTreeSet<_>>();
            for feature in [
                "avx2", "fma", "bmi2", "aes", "vaes", "avx512f", "neon", "asimd",
            ] {
                if flags.contains(feature) {
                    features.push(feature.to_owned());
                }
            }
        }
    }
    #[cfg(all(
        not(target_os = "linux"),
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        for (name, detected) in [
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("bmi2", std::is_x86_feature_detected!("bmi2")),
            ("aes", std::is_x86_feature_detected!("aes")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
        ] {
            if detected {
                features.push(name.to_owned());
            }
        }
    }
    #[cfg(all(not(target_os = "linux"), target_arch = "aarch64"))]
    {
        for (name, detected) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("aes", std::arch::is_aarch64_feature_detected!("aes")),
            ("sha2", std::arch::is_aarch64_feature_detected!("sha2")),
        ] {
            if detected {
                features.push(name.to_owned());
            }
        }
    }
    if features.is_empty() {
        features.push("scalar".to_owned());
    }
    features.sort_unstable();
    features.dedup();
    features
}

fn runtime_isa_is_normalized(features: &[String]) -> bool {
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
    !features.is_empty()
        && features.iter().all(|feature| valid_token(feature))
        && features
            .iter()
            .all(|feature| REPORTABLE_RUNTIME_ISA.contains(&feature.as_str()))
        && features.windows(2).all(|pair| pair[0] < pair[1])
        && (features.len() == 1 || features.iter().all(|feature| feature != "scalar"))
}

/// One engine or comparison row in a gate artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfCellResult {
    pub fixture: String,
    pub metric: String,
    pub engine: String,
    pub unit: String,
    #[serde(flatten)]
    pub distribution: DistributionSummary,
}

type PerfCellResultContract = (String, String, String, String);

/// Canonical display/storage unit for one normative performance metric.
#[must_use]
pub fn perf_metric_unit(metric: &str) -> &'static str {
    match metric {
        "docs_per_second" | "updates_per_second" | "tokenize_docs_per_second" => "docs/s",
        "commit_latency_ms"
        | "latency_ms"
        | "open_latency_ms"
        | "update_to_searchable_ms"
        | "wall_clock_ms" => "ms",
        "peak_rss_bytes" => "bytes",
        "index_bytes_per_document" => "bytes/doc",
        "tantivy_nodes" => "nodes",
        _ => "ratio",
    }
}

/// Reconstruct the exact operation scope emitted for one canonical cell.
#[must_use]
pub fn perf_operation_scope(gate: PerfGate, fixture: &str, metric: &str) -> PerfOperationScope {
    let semantics = match metric {
        "docs_per_second" | "tokenize_docs_per_second" | "updates_per_second" => {
            PerfMetricSemantics::GaugeHigherIsBetter
        }
        _ => PerfMetricSemantics::GaugeLowerIsBetter,
    };
    PerfOperationScope {
        operation_id: format!("{gate}.{fixture}.{metric}"),
        version: 1,
        semantics,
        unit: perf_metric_unit(metric).to_owned(),
    }
}

fn result_contract(cell: &PerfCellResult) -> PerfCellResultContract {
    (
        cell.fixture.clone(),
        cell.metric.clone(),
        cell.engine.clone(),
        cell.unit.clone(),
    )
}

fn expected_result_contracts(gate: PerfGate, spec: &PerfCellSpec) -> Vec<PerfCellResultContract> {
    if gate == PerfGate::Qg10 {
        return vec![(
            spec.fixture.clone(),
            spec.metric.clone(),
            "default_feature_graph".to_owned(),
            "nodes".to_owned(),
        )];
    }
    let absolute_engine = if spec.metric == "tokenize_docs_per_second" {
        "quill_tokenizer"
    } else {
        "quill"
    };
    let oracle_engine = if spec.metric == "tokenize_docs_per_second" {
        "quill_tokenizer_null"
    } else {
        "tantivy"
    };
    let mut contracts = vec![
        (
            spec.fixture.clone(),
            spec.metric.clone(),
            absolute_engine.to_owned(),
            perf_metric_unit(&spec.metric).to_owned(),
        ),
        (
            spec.fixture.clone(),
            spec.metric.clone(),
            oracle_engine.to_owned(),
            perf_metric_unit(&spec.metric).to_owned(),
        ),
        (
            spec.fixture.clone(),
            format!("{}_quill_over_tantivy", spec.metric),
            "paired_ab".to_owned(),
            "ratio".to_owned(),
        ),
        (
            spec.fixture.clone(),
            format!("{}_tantivy_over_tantivy", spec.metric),
            "paired_null".to_owned(),
            "ratio".to_owned(),
        ),
    ];
    if gate == PerfGate::Qg1 {
        contracts.push((
            spec.fixture.clone(),
            format!("{}_quill_over_quill", spec.metric),
            "paired_null_quill".to_owned(),
            "ratio".to_owned(),
        ));
    }
    contracts
}

/// Per-gate JSON artifact matching the committed E0.6 schema contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfGateArtifact {
    pub schema_version: String,
    pub gate: PerfGate,
    /// Required on every measured v7 artifact. `None` exists only for the
    /// exact unmeasured v7 sentinel and explicit read-only legacy loaders.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applicability_plan: Option<PerfApplicabilityPlanBinding>,
    /// SHA-256 emitted by the benchmark process for its own executing ELF.
    pub bench_elf_sha256: String,
    pub machine_fingerprint: String,
    /// Required on measured v7 artifacts. `None` exists only for the exact
    /// unmeasured v7 sentinel and the explicit read-only v3 loader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<PerfExecutionProvenance>,
    pub git_rev: String,
    /// Shared identifier for the bounded candidate/rerun measurement window.
    pub run_window: String,
    /// Unique identifier for one pass inside the measurement window.
    pub run_id: String,
    pub corpus_manifest_hash: String,
    pub manifest_sha256: String,
    pub cells: Vec<PerfCellResult>,
    pub laws_attested: bool,
}

impl PerfGateArtifact {
    fn validate_selected_cells(
        &self,
        plan: &PerfApplicabilityPlan,
    ) -> Result<(BTreeSet<usize>, bool), GauntletError> {
        let invalid = |reason: String| GauntletError::InvalidPreparedArtifact { reason };
        let matrix = PerfMatrixSpec::complete();
        let canonical_cells = matrix.for_gate(self.gate);
        if canonical_cells.len() != plan.cells.len() {
            return Err(invalid(
                "threshold plan does not classify the complete canonical gate".to_owned(),
            ));
        }

        let mut expected_by_ordinal = BTreeMap::new();
        let mut contract_to_ordinal = BTreeMap::new();
        for (spec, classification) in canonical_cells.into_iter().zip(&plan.cells) {
            if !classification.applicability.is_runnable() {
                continue;
            }
            let contracts = expected_result_contracts(self.gate, spec)
                .into_iter()
                .collect::<BTreeSet<_>>();
            for contract in &contracts {
                if contract_to_ordinal
                    .insert(contract.clone(), classification.ordinal)
                    .is_some()
                {
                    return Err(invalid(
                        "canonical threshold row contract is ambiguous".to_owned(),
                    ));
                }
            }
            expected_by_ordinal.insert(classification.ordinal, contracts);
        }

        let mut seen = BTreeSet::new();
        let mut selected_by_ordinal = BTreeMap::<usize, BTreeSet<_>>::new();
        for cell in &self.cells {
            let contract = result_contract(cell);
            if !seen.insert(contract.clone()) {
                return Err(invalid(format!(
                    "threshold repeats cell row {}/{}/{}/{}",
                    contract.0, contract.1, contract.2, contract.3
                )));
            }
            let ordinal = contract_to_ordinal.get(&contract).ok_or_else(|| {
                invalid(format!(
                    "threshold row {}/{}/{}/{} is not part of the runnable canonical plan",
                    contract.0, contract.1, contract.2, contract.3
                ))
            })?;
            selected_by_ordinal
                .entry(*ordinal)
                .or_default()
                .insert(contract);
        }
        if selected_by_ordinal.is_empty() {
            return Err(invalid(
                "a measured threshold requires at least one complete canonical cell".to_owned(),
            ));
        }

        let mut selected_widths = BTreeSet::new();
        for (ordinal, selected) in &selected_by_ordinal {
            let expected = expected_by_ordinal.get(ordinal).ok_or_else(|| {
                invalid(format!(
                    "threshold selected non-runnable canonical cell ordinal {ordinal}"
                ))
            })?;
            if selected != expected {
                return Err(invalid(format!(
                    "threshold canonical cell ordinal {ordinal} has missing, extra, or altered engine rows"
                )));
            }
            let configured_threads = plan
                .cells
                .get(*ordinal)
                .ok_or_else(|| invalid(format!("threshold plan omits cell ordinal {ordinal}")))?
                .configured_threads;
            selected_widths.insert(configured_threads);
        }
        Ok((
            selected_widths,
            selected_by_ordinal.len() == expected_by_ordinal.len(),
        ))
    }

    /// Parse and verify one current measured threshold artifact.
    ///
    /// This loader intentionally rejects unmeasured sentinels and historical
    /// schemas. It reconstructs the applicability plan from the compiled
    /// matrix, normative performance manifest, and frozen machine registry;
    /// artifact-provided hashes never establish their own validity.
    ///
    /// # Errors
    ///
    /// Returns a strict JSON or contract error when the bytes are
    /// noncanonical, stale, incomplete, or do not reconstruct exactly.
    pub fn from_verified_measured_slice(bytes: &[u8]) -> Result<Self, GauntletError> {
        let artifact = serde_json::from_slice::<Self>(bytes)?;
        if serde_json::to_vec_pretty(&artifact)? != bytes {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "threshold artifact is not exact canonical pretty JSON".to_owned(),
            });
        }
        artifact.verify_current_measured_contract()?;
        Ok(artifact)
    }

    /// Reconstruct the exact plan and verify this current measured artifact's
    /// matrix, manifest, registry, profile, capacity, and plan envelope.
    ///
    /// # Errors
    ///
    /// Returns a bounded contract error for sentinels, legacy schemas, or any
    /// field that differs from fresh canonical reconstruction.
    pub fn verify_current_measured_contract(&self) -> Result<PerfApplicabilityPlan, GauntletError> {
        let invalid = |reason: String| GauntletError::InvalidPreparedArtifact { reason };
        if self.schema_version != PERF_ARTIFACT_SCHEMA_VERSION {
            return Err(invalid(format!(
                "measured threshold schema is {:?}, expected {:?}",
                self.schema_version, PERF_ARTIFACT_SCHEMA_VERSION
            )));
        }
        let binding = self.applicability_plan.as_ref().ok_or_else(|| {
            invalid("current measured threshold has no applicability-plan binding".to_owned())
        })?;
        if binding.gate != self.gate {
            return Err(invalid(
                "threshold gate differs from its applicability-plan gate".to_owned(),
            ));
        }
        let registry = MachineClassRegistry::frozen().map_err(|error| {
            invalid(format!(
                "frozen machine registry rejected threshold verification: {error}"
            ))
        })?;
        let plan = PerfMatrixSpec::complete()
            .applicability_plan(&registry, binding.profile, binding.gate)
            .map_err(|error| {
                invalid(format!(
                    "threshold applicability plan does not reconstruct: {error}"
                ))
            })?;
        if plan.binding != *binding {
            return Err(invalid(
                "threshold applicability-plan binding differs from canonical reconstruction"
                    .to_owned(),
            ));
        }
        if self.manifest_sha256 != binding.normalized_perf_manifest_sha256 {
            return Err(invalid(
                "threshold manifest digest differs from its reconstructed plan".to_owned(),
            ));
        }
        let execution = self.execution.as_ref().ok_or_else(|| {
            invalid("current measured threshold has no execution provenance".to_owned())
        })?;
        if !execution.is_complete() {
            return Err(invalid(
                "current measured threshold has incomplete execution provenance".to_owned(),
            ));
        }
        if !execution.matches_capacity_semantics(plan.capacity_semantics) {
            return Err(invalid(
                "threshold process availability contradicts its capacity semantics".to_owned(),
            ));
        }
        if plan.execution_capacity != Some(execution.execution_capacity)
            || plan.max_exercised_cell_width != Some(execution.max_exercised_cell_width)
        {
            return Err(invalid(
                "threshold execution capacity/maximum differs from its reconstructed plan"
                    .to_owned(),
            ));
        }
        let (selected_widths, complete_selection) = self.validate_selected_cells(&plan)?;
        if execution.configured_engine_thread_widths
            != selected_widths.iter().copied().collect::<Vec<_>>()
        {
            return Err(invalid(
                "threshold configured engine widths differ from its actual selected canonical cells"
                    .to_owned(),
            ));
        }
        if self.laws_attested != complete_selection {
            return Err(invalid(
                "threshold laws_attested must equal exact full runnable-plan coverage".to_owned(),
            ));
        }
        let measured_label =
            |value: &str| !value.trim().is_empty() && !value.eq_ignore_ascii_case("unmeasured");
        let git_revision = matches!(self.git_rev.len(), 40 | 64)
            && self
                .git_rev
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if !measured_label(&self.machine_fingerprint)
            || !measured_label(&self.run_window)
            || !measured_label(&self.run_id)
            || !git_revision
        {
            return Err(invalid(
                "measured threshold requires a concrete machine, run/window, and lowercase Git revision identity"
                    .to_owned(),
            ));
        }
        if !is_lower_hex_digest(&self.bench_elf_sha256)
            || !is_lower_hex_digest(&self.corpus_manifest_hash)
            || !is_lower_hex_digest(&self.manifest_sha256)
        {
            return Err(invalid(
                "threshold identity digests must be lowercase SHA-256".to_owned(),
            ));
        }
        Ok(plan)
    }

    /// Encode canonical pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns a serde error when a non-finite number slipped past validation.
    pub fn to_json_pretty(&self) -> Result<String, GauntletError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Render the compact operator table printed beside JSON.
    #[must_use]
    pub fn human_table(&self) -> String {
        let mut table = String::new();
        if let Some(execution) = &self.execution {
            let _ = writeln!(
                table,
                "host={} | physical_cores={} | logical_threads={} | \
                 process_available_threads={} | execution_capacity={} | \
                 max_exercised_cell_width={} | configured_engine_thread_widths={:?} | \
                 runtime_detected_isa={:?} | cpu_affinity_allowed_list={} | \
                 affinity_or_cpuset_cap={}",
                execution.host_identity,
                execution.physical_cores,
                execution.logical_threads,
                execution.process_available_threads,
                execution.execution_capacity,
                execution.max_exercised_cell_width,
                execution.configured_engine_thread_widths,
                execution.runtime_detected_isa,
                execution
                    .cpu_affinity_allowed_list
                    .as_deref()
                    .unwrap_or("unavailable"),
                execution
                    .affinity_or_cpuset_cap
                    .as_deref()
                    .unwrap_or("none"),
            );
        }
        table.push_str(
            "fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission\n",
        );
        table.push_str("--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---\n");
        for cell in &self.cells {
            let admission = if cell.distribution.sampled_for_activation() {
                "sampled"
            } else {
                "under-sampled"
            };
            let _ = writeln!(
                table,
                "{} | {} | {} ({}) | {:.6} | [{:.6}, {:.6}] | {:.6} | {:.6} | {:.3} | {} | {}",
                cell.fixture,
                cell.engine,
                cell.metric,
                cell.unit,
                cell.distribution.p50,
                cell.distribution.median_ci95_low,
                cell.distribution.median_ci95_high,
                cell.distribution.p95,
                cell.distribution.p99,
                cell.distribution.cv_pct,
                cell.distribution.runs,
                admission,
            );
        }
        table
    }

    /// Write JSON and Markdown artifacts for one gate.
    ///
    /// # Errors
    ///
    /// Returns typed serialization or filesystem errors.
    pub fn write_to(&self, output_dir: &Path) -> Result<(PathBuf, PathBuf), GauntletError> {
        fs::create_dir_all(output_dir)?;
        let stem = self.gate.label();
        let json_path = output_dir.join(format!("{stem}.json"));
        let table_path = output_dir.join(format!("{stem}.md"));
        fs::write(&json_path, self.to_json_pretty()?)?;
        fs::write(&table_path, self.human_table())?;
        Ok((json_path, table_path))
    }
}

/// Deterministic benchmark-reported machine label for diagnostic correlation.
///
/// Receipt-admitted hardware/profile facts remain authoritative for ratchet
/// comparisons; this label alone never authorizes a cross-machine decision.
#[must_use]
pub fn machine_fingerprint() -> String {
    let logical_threads = host_cpu_topology()
        .map(|(_, logical_threads)| logical_threads)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let host = host_identity().unwrap_or_else(|| "unknown-host".to_owned());
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents
                .lines()
                .find_map(|line| {
                    line.strip_prefix("model name\t:")
                        .or_else(|| line.strip_prefix("Hardware\t:"))
                        .map(str::trim)
                })
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown-cpu".to_owned());
    format!(
        "{}-{}-{host}-{logical_threads}thread-{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cpu.replace(['/', ' '], "_")
    )
}

/// Linux peak resident set size in bytes from `VmHWM`.
///
/// Other operating systems return `None`. The isolated macOS benchmark child
/// is wrapped in `/usr/bin/time -l` and parsed by
/// [`parse_macos_time_max_rss_bytes`] instead of fabricating an in-process
/// value.
#[must_use]
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        parse_linux_vmhwm_bytes(&status)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse the byte-valued peak RSS row emitted by macOS `/usr/bin/time -l`.
///
/// The parser requires the complete four-word label so unrelated counters such
/// as `peak memory footprint` cannot be mistaken for resident-set evidence.
#[must_use]
pub fn parse_macos_time_max_rss_bytes(report: &str) -> Option<u64> {
    report.lines().find_map(|line| {
        let mut fields = line.split_ascii_whitespace();
        let bytes = fields.next()?.parse::<u64>().ok()?;
        (fields.next() == Some("maximum")
            && fields.next() == Some("resident")
            && fields.next() == Some("set")
            && fields.next() == Some("size")
            && fields.next().is_none())
        .then_some(bytes)
    })
}

fn parse_linux_vmhwm_bytes(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let mut fields = line.split_ascii_whitespace();
    let _label = fields.next()?;
    let kib = fields.next()?.parse::<u64>().ok()?;
    match fields.next() {
        Some("kB") => kib.checked_mul(1024),
        _ => None,
    }
}

/// Assert that a matrix contains every gate and no dishonest zero-density QG-5
/// cell.
///
/// # Errors
///
/// Returns a typed campaign error when coverage is incomplete.
pub fn validate_matrix(matrix: &PerfMatrixSpec) -> Result<(), GauntletError> {
    let gates = matrix
        .cells
        .iter()
        .map(|cell| cell.gate)
        .collect::<BTreeSet<_>>();
    if gates != PerfGate::ALL.into_iter().collect() {
        return Err(GauntletError::InvalidCampaign {
            reason: "performance matrix does not cover QG-1 through QG-10".to_owned(),
        });
    }
    if matrix.cells.iter().any(|cell| {
        cell.gate == PerfGate::Qg5
            && cell
                .tombstone_density_pct
                .is_none_or(|density| density == 0)
    }) {
        return Err(GauntletError::InvalidCampaign {
            reason: "QG-5 requires a nonzero tombstone density".to_owned(),
        });
    }
    if matrix
        .cells
        .iter()
        .any(|cell| cell.threads.is_none_or(|threads| threads == 0))
    {
        return Err(GauntletError::InvalidCampaign {
            reason: "every performance cell requires a positive configured thread width".to_owned(),
        });
    }
    validate_qg6_matrix(matrix)?;
    Ok(())
}

fn validate_qg6_matrix(matrix: &PerfMatrixSpec) -> Result<(), GauntletError> {
    let qg6_cells = matrix
        .cells
        .iter()
        .filter(|cell| cell.gate == PerfGate::Qg6)
        .collect::<Vec<_>>();
    if qg6_cells.len() != PerfQueryClass::ALL.len() * 2 * 2 {
        return Err(GauntletError::InvalidCampaign {
            reason: "QG-6 requires exactly 20 warm total-search cells".to_owned(),
        });
    }

    let mut observed = BTreeSet::new();
    for cell in qg6_cells {
        let Some(query_class) = cell.query_class else {
            return Err(GauntletError::InvalidCampaign {
                reason: "every QG-6 cell requires a named query class".to_owned(),
            });
        };
        let Some(k @ (10 | 100)) = cell.k else {
            return Err(GauntletError::InvalidCampaign {
                reason: "every QG-6 cell requires k=10 or k=100".to_owned(),
            });
        };
        let Some(document_count @ (100_000 | 1_000_000)) = cell.document_count else {
            return Err(GauntletError::InvalidCampaign {
                reason: "every QG-6 cell requires the 100k or 1M corpus".to_owned(),
            });
        };
        let corpus_label = if document_count == 100_000 {
            "100k"
        } else {
            "1m"
        };
        let expected_fixture = format!("query/{}/k{k}/{corpus_label}", query_class.label());
        if cell.fixture != expected_fixture
            || cell.metric != "latency_ms"
            || cell.positions != Some(PositionMode::On)
            || cell.threads != Some(1)
            || cell.writer_heap_bytes != Some(perf_writer_heap_bytes(1))
            || cell.topology.is_some()
            || cell.tombstone_density_pct.is_some()
        {
            return Err(GauntletError::InvalidCampaign {
                reason: format!(
                    "QG-6 cell {expected_fixture} must be the warm total-search latency lane with \
                     positions, one thread, and the canonical heap budget"
                ),
            });
        }
        if !observed.insert((query_class, k, document_count)) {
            return Err(GauntletError::InvalidCampaign {
                reason: format!("duplicate QG-6 cell {expected_fixture}"),
            });
        }
    }

    let expected = PerfQueryClass::ALL
        .into_iter()
        .flat_map(|query_class| {
            [10, 100].into_iter().flat_map(move |k| {
                [100_000, 1_000_000]
                    .into_iter()
                    .map(move |document_count| (query_class, k, document_count))
            })
        })
        .collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(GauntletError::InvalidCampaign {
            reason: "QG-6 matrix has a missing or reclassified class/k/corpus cell".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine_class_registry::{ExecutionProfileId, HardwareClassId};

    const PERF_MANIFEST: &str = include_str!("../../../docs/contracts/quill-perf-gates.toml");

    fn profile_key(
        hardware_class_id: HardwareClassId,
        execution_profile_id: ExecutionProfileId,
    ) -> MachineProfileKey {
        MachineProfileKey::new(hardware_class_id, execution_profile_id)
            .expect("canonical profile key")
    }

    fn qg1_plan(
        registry: &MachineClassRegistry,
        execution_profile_id: ExecutionProfileId,
    ) -> PerfApplicabilityPlan {
        let hardware_class_id = match execution_profile_id {
            ExecutionProfileId::Physical64 | ExecutionProfileId::Smt2_128 => {
                HardwareClassId::TrjZen35995wx
            }
            ExecutionProfileId::Scheduler10 => HardwareClassId::M4Macos,
            ExecutionProfileId::X86Diagnostic => HardwareClassId::X86VpsOvh,
            ExecutionProfileId::Scheduler14 => HardwareClassId::M5Macos,
        };
        PerfMatrixSpec::complete()
            .applicability_plan(
                registry,
                profile_key(hardware_class_id, execution_profile_id),
                PerfGate::Qg1,
            )
            .expect("canonical QG-1 applicability plan")
    }

    fn threshold_rows(gate: PerfGate, specs: &[&PerfCellSpec]) -> Vec<PerfCellResult> {
        let distribution = DistributionSummary::from_samples(&[1.0; PERF_MIN_RUNS])
            .expect("constant threshold distribution");
        specs
            .iter()
            .flat_map(|spec| expected_result_contracts(gate, spec))
            .map(|(fixture, metric, engine, unit)| PerfCellResult {
                fixture,
                metric,
                engine,
                unit,
                distribution: distribution.clone(),
            })
            .collect()
    }

    #[test]
    fn prepared_input_fingerprint_binds_each_component_independently() {
        let identity = PerfInputIdentity {
            prepared_corpus_sha256: "a".repeat(64),
            query_manifest_sha256: "b".repeat(64),
            config_contract_sha256: "c".repeat(64),
            semantic_contract_sha256: Some("d".repeat(64)),
            query_group_count: QG6_QUERY_GROUPS,
            query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
        };
        let fingerprint = identity.fingerprint_sha256();
        assert!(is_lower_hex_digest(&fingerprint));

        for field in [
            "prepared_corpus_sha256",
            "query_manifest_sha256",
            "config_contract_sha256",
            "semantic_contract_sha256",
        ] {
            let mut mutated = identity.clone();
            match field {
                "prepared_corpus_sha256" => mutated.prepared_corpus_sha256 = "d".repeat(64),
                "query_manifest_sha256" => mutated.query_manifest_sha256 = "e".repeat(64),
                "config_contract_sha256" => mutated.config_contract_sha256 = "f".repeat(64),
                "semantic_contract_sha256" => {
                    mutated.semantic_contract_sha256 = Some("0".repeat(64));
                }
                _ => unreachable!("enumerated identity field"),
            }
            assert_ne!(
                fingerprint,
                mutated.fingerprint_sha256(),
                "{field} is not bound by the telemetry fingerprint"
            );
        }

        let mut wrong_count = identity.clone();
        wrong_count.query_group_count -= 1;
        assert_ne!(fingerprint, wrong_count.fingerprint_sha256());
        assert!(wrong_count.validate().is_err());

        let mut wrong_ids = identity;
        wrong_ids.query_group_ids.swap(0, 1);
        assert_ne!(fingerprint, wrong_ids.fingerprint_sha256());
        assert!(wrong_ids.validate().is_err());
    }

    #[test]
    fn absent_semantic_contract_has_a_total_fingerprint_and_omits_the_wire_field() {
        let identity = PerfInputIdentity {
            prepared_corpus_sha256: "a".repeat(64),
            query_manifest_sha256: "b".repeat(64),
            config_contract_sha256: "c".repeat(64),
            semantic_contract_sha256: None,
            query_group_count: QG6_QUERY_GROUPS,
            query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
        };
        let fingerprint = identity.fingerprint_sha256();
        assert!(is_lower_hex_digest(&fingerprint));
        let json = serde_json::to_value(&identity).expect("identity JSON");
        assert!(json.get("semantic_contract_sha256").is_none());

        let mut present = identity;
        present.semantic_contract_sha256 = Some("d".repeat(64));
        assert_ne!(fingerprint, present.fingerprint_sha256());
    }

    fn estimator_config() -> PairedEstimatorConfig {
        PairedEstimatorConfig::predeclared(0x5eed_1234_5678_9abc)
    }

    fn operation_scope(semantics: PerfMetricSemantics) -> PerfOperationScope {
        PerfOperationScope {
            operation_id: "qg.synthetic_operation".to_owned(),
            version: 1,
            semantics,
            unit: match semantics {
                PerfMetricSemantics::Throughput => "work/s",
                PerfMetricSemantics::Duration => "ns",
                PerfMetricSemantics::GaugeHigherIsBetter
                | PerfMetricSemantics::GaugeLowerIsBetter => "units",
            }
            .to_owned(),
        }
    }

    fn provenance(run_id: &str) -> PerfSampleProvenance {
        PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            input_identity: None,
            worker_id: "synthetic-worker".to_owned(),
            build_profile: "release-perf".to_owned(),
        }
    }

    fn duration_stream(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        control_durations: &[u64],
        treatment_durations: &[u64],
        sample_id_base: u64,
    ) -> Vec<PerfRawSample> {
        assert_eq!(control_durations.len(), treatment_durations.len());
        let first_arms =
            seeded_balanced_pair_order(control_durations.len(), 0x00dd_5eed).expect("pair order");
        let mut samples = Vec::with_capacity(control_durations.len() * 2);
        for (index, ((control_duration, treatment_duration), first_arm)) in control_durations
            .iter()
            .zip(treatment_durations)
            .zip(first_arms)
            .enumerate()
        {
            let block_id = u64::try_from(index).expect("test block ID");
            let base = block_id.saturating_mul(100_000_000);
            let control_first = first_arm == PerfSampleArm::Control;
            let (control_start, treatment_start) = if control_first {
                (base, base + control_duration + 1_000)
            } else {
                (base + treatment_duration + 1_000, base)
            };
            let control_order = if control_first {
                PerfSampleOrder::First
            } else {
                PerfSampleOrder::Second
            };
            let treatment_order = if control_first {
                PerfSampleOrder::Second
            } else {
                PerfSampleOrder::First
            };
            let index = u64::try_from(index).expect("test sample index");
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2,
                arm: PerfSampleArm::Control,
                order: control_order,
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: control_start,
                ended_ns: control_start + control_duration,
                work_units: Some(1_000),
                byte_count: Some(64_000),
                observed_value: None,
                group_id: None,
                qg6_sample_binding: None,
            });
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2 + 1,
                arm: PerfSampleArm::Treatment,
                order: treatment_order,
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: treatment_start,
                ended_ns: treatment_start + treatment_duration,
                work_units: Some(1_000),
                byte_count: Some(64_000),
                observed_value: None,
                group_id: None,
                qg6_sample_binding: None,
            });
        }
        samples
    }

    fn gauge_stream(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        controls: &[f64],
        treatments: &[f64],
        sample_id_base: u64,
    ) -> Vec<PerfRawSample> {
        assert_eq!(controls.len(), treatments.len());
        let first_arms =
            seeded_balanced_pair_order(controls.len(), 0x00c0_ffee).expect("pair order");
        let mut samples = Vec::with_capacity(controls.len() * 2);
        for (index, ((control, treatment), first_arm)) in
            controls.iter().zip(treatments).zip(first_arms).enumerate()
        {
            let block_id = u64::try_from(index).expect("test block ID");
            let base = block_id.saturating_mul(1_000);
            let control_first = first_arm == PerfSampleArm::Control;
            let (control_start, treatment_start) = if control_first {
                (base, base + 200)
            } else {
                (base + 200, base)
            };
            let index = u64::try_from(index).expect("test sample index");
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2,
                arm: PerfSampleArm::Control,
                order: if control_first {
                    PerfSampleOrder::First
                } else {
                    PerfSampleOrder::Second
                },
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: control_start,
                ended_ns: control_start + 100,
                work_units: None,
                byte_count: None,
                observed_value: Some(*control),
                group_id: None,
                qg6_sample_binding: None,
            });
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2 + 1,
                arm: PerfSampleArm::Treatment,
                order: if control_first {
                    PerfSampleOrder::Second
                } else {
                    PerfSampleOrder::First
                },
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: treatment_start,
                ended_ns: treatment_start + 100,
                work_units: None,
                byte_count: None,
                observed_value: Some(*treatment),
                group_id: None,
                qg6_sample_binding: None,
            });
        }
        samples
    }

    fn stable_null(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) -> Vec<PerfRawSample> {
        let durations = [
            1_000_000, 1_200_000, 900_000, 1_500_000, 800_000, 1_100_000, 1_300_000, 950_000,
            1_050_000, 1_400_000,
        ];
        duration_stream(scope, provenance, &durations, &durations, 10_000)
    }

    #[test]
    fn paired_samples_reject_misaligned_blocks_and_duplicate_sample_ids() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("misaligned");
        let durations = [1_000_000; PERF_MIN_RUNS];
        let mut effect = duration_stream(&scope, &provenance, &durations, &durations, 0);
        effect
            .iter_mut()
            .find(|sample| sample.arm == PerfSampleArm::Treatment && sample.block_id == 7)
            .expect("target treatment sample")
            .block_id = 77;
        let error = estimate_paired_experiment(
            &effect,
            &stable_null(&scope, &provenance),
            &estimator_config(),
        )
        .expect_err("misaligned block IDs must fail closed");
        assert!(
            matches!(error, PairedEstimatorError::IncompleteBlock { .. }),
            "unexpected pairing error: {error}"
        );

        let mut duplicate_ids = duration_stream(&scope, &provenance, &durations, &durations, 0);
        duplicate_ids[1].sample_id = duplicate_ids[0].sample_id;
        assert!(matches!(
            estimate_paired_experiment(
                &duplicate_ids,
                &stable_null(&scope, &provenance),
                &estimator_config()
            ),
            Err(PairedEstimatorError::DuplicateSampleId { .. })
        ));
    }

    #[test]
    fn paired_log_estimator_recovers_known_effect_with_heteroskedastic_outlier() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("known-effect");
        let controls = [
            1_000_000, 9_000_000, 2_000_000, 20_000_000, 3_000_000, 30_000_000, 4_000_000,
            40_000_000, 5_000_000, 50_000_000,
        ];
        let mut treatments = controls.map(|duration| duration / 2);
        treatments[3] = controls[3] * 10;
        let result = estimate_paired_experiment(
            &duration_stream(&scope, &provenance, &controls, &treatments, 0),
            &stable_null(&scope, &provenance),
            &estimator_config(),
        )
        .expect("known paired effect");
        assert_eq!(result.status, PairedEvidenceStatus::Valid);
        assert_eq!(result.claim_state, PairedClaimState::EligibleForDecision);
        assert!((result.effect.treatment_over_control - 2.0).abs() < 1.0e-12);
        assert!(result.effect.ci95_low_ratio > 1.0);
        assert!(result.effect.algebraic_reconciliation_error.abs() < 1.0e-12);
        result.verify_recomputed().expect("raw samples recompute");
    }

    #[test]
    fn noisy_order_dependent_null_is_retained_as_no_decision() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("invalid-null");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let treatments = [500_000; PERF_MIN_RUNS];
        let effect = duration_stream(&scope, &provenance, &controls, &treatments, 0);
        let mut null = stable_null(&scope, &provenance);
        for sample in &mut null {
            if sample.arm == PerfSampleArm::Treatment {
                let duration = sample.ended_ns - sample.started_ns;
                let biased = if sample.order == PerfSampleOrder::Second {
                    duration / 4
                } else {
                    duration * 4
                };
                sample.ended_ns = sample.started_ns + biased;
            }
        }
        for block_id in 0..u64::try_from(PERF_MIN_RUNS).expect("test block count") {
            let first_end = null
                .iter()
                .find(|sample| {
                    sample.block_id == block_id && sample.order == PerfSampleOrder::First
                })
                .expect("first sample")
                .ended_ns;
            let second = null
                .iter_mut()
                .find(|sample| {
                    sample.block_id == block_id && sample.order == PerfSampleOrder::Second
                })
                .expect("second sample");
            let duration = second.ended_ns - second.started_ns;
            second.started_ns = first_end + 1_000;
            second.ended_ns = second.started_ns + duration;
        }
        let result =
            estimate_paired_experiment(&effect, &null, &estimator_config()).expect("diagnostic");
        assert_eq!(result.status, PairedEvidenceStatus::InvalidNull);
        assert_eq!(result.claim_state, PairedClaimState::NoDecision);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "paired.null_order_effect")
        );
        assert_eq!(result.effect_samples.len(), PERF_MIN_RUNS * 2);
        assert_eq!(result.null_samples.len(), PERF_MIN_RUNS * 2);
        result
            .verify_recomputed()
            .expect("invalid diagnostics still recompute");
    }

    #[test]
    fn drifting_null_is_invalid_even_when_other_null_limits_are_wide() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("drifting-null");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let effect_treatments = [500_000; PERF_MIN_RUNS];
        let mut null_treatments = controls;
        null_treatments[PERF_MIN_RUNS / 2..].fill(1_200_000);
        let effect = duration_stream(&scope, &provenance, &controls, &effect_treatments, 0);
        let null = duration_stream(&scope, &provenance, &controls, &null_treatments, 10_000);
        let mut config = estimator_config();
        config.max_null_center_log = 1.5_f64.ln();
        config.max_null_ci_half_width_log = 1.5_f64.ln();
        config.max_null_log_mad = 1.5_f64.ln();
        config.max_null_order_effect_log = 1.5_f64.ln();
        let result = estimate_paired_experiment(&effect, &null, &config).expect("drift diagnostic");
        assert_eq!(result.status, PairedEvidenceStatus::InvalidNull);
        assert_eq!(result.claim_state, PairedClaimState::NoDecision);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "paired.null_drift")
        );
    }

    #[test]
    fn same_seed_bootstrap_and_json_round_trip_are_exact() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("deterministic");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let treatments = [625_000; PERF_MIN_RUNS];
        let effect = duration_stream(&scope, &provenance, &controls, &treatments, 0);
        let null = stable_null(&scope, &provenance);
        let first = estimate_paired_experiment(&effect, &null, &estimator_config()).expect("first");
        let second =
            estimate_paired_experiment(&effect, &null, &estimator_config()).expect("second");
        assert_eq!(first, second);
        let encoded = serde_json::to_vec(&first).expect("encode paired result");
        let value: serde_json::Value =
            serde_json::from_slice(&encoded).expect("decode paired schema");
        let top_level_keys = value
            .as_object()
            .expect("paired result object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            top_level_keys,
            BTreeSet::from([
                "claim_state",
                "config",
                "effect",
                "effect_samples",
                "null",
                "null_samples",
                "provenance",
                "reasons",
                "schema_version",
                "scope",
                "status",
            ])
        );
        let raw_sample_keys = value["effect_samples"][0]
            .as_object()
            .expect("raw paired sample")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            raw_sample_keys,
            BTreeSet::from([
                "arm",
                "block_id",
                "byte_count",
                "ended_ns",
                "group_id",
                "observed_value",
                "order",
                "phase",
                "provenance",
                "sample_id",
                "scope",
                "started_ns",
                "work_units",
            ])
        );
        let decoded: PairedExperimentResult =
            serde_json::from_slice(&encoded).expect("decode paired result");
        assert_eq!(
            serde_json::to_vec(&decoded).expect("re-encode paired result"),
            encoded
        );
        decoded.verify_recomputed().expect("round-trip recomputes");
    }

    #[test]
    fn scope_mismatch_and_warmup_leak_fail_closed() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("scope-mismatch");
        let durations = [1_000_000; PERF_MIN_RUNS];
        let effect = duration_stream(&scope, &provenance, &durations, &durations, 0);
        let mut other_scope = scope.clone();
        other_scope.version = 2;
        let null = stable_null(&other_scope, &provenance);
        assert!(matches!(
            estimate_paired_experiment(&effect, &null, &estimator_config()),
            Err(PairedEstimatorError::CrossExperimentMismatch {
                field: "operation scope"
            })
        ));

        let mut warmup = effect;
        warmup[0].phase = PerfSamplePhase::Warmup;
        assert!(matches!(
            estimate_paired_experiment(
                &warmup,
                &stable_null(&scope, &provenance),
                &estimator_config()
            ),
            Err(PairedEstimatorError::WarmupInDecisionSet { .. })
        ));
    }

    #[test]
    fn contradictory_paired_and_marginal_directions_yield_no_decision() {
        let scope = operation_scope(PerfMetricSemantics::GaugeHigherIsBetter);
        let provenance = provenance("contradictory");
        let controls = [100.0, 1_000.0, 10_000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let treatments = [
            99.0, 999.0, 9_999.0, 100_000.0, 200_000.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ];
        let effect = gauge_stream(&scope, &provenance, &controls, &treatments, 0);
        let null = gauge_stream(&scope, &provenance, &controls, &controls, 10_000);
        let result =
            estimate_paired_experiment(&effect, &null, &estimator_config()).expect("diagnostic");
        assert_eq!(result.status, PairedEvidenceStatus::ContradictorySummaries);
        assert_eq!(result.claim_state, PairedClaimState::NoDecision);
        assert!(result.effect.median_log_ratio.is_sign_negative());
        assert!(result.effect.ratio_of_arm_medians > 1.0);
    }

    #[test]
    fn independent_run_ids_reproduce_within_predeclared_tolerance() {
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let first_provenance = provenance("process-one");
        let second_provenance = provenance("process-two");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let treatments = [500_000; PERF_MIN_RUNS];
        let first = estimate_paired_experiment(
            &duration_stream(&scope, &first_provenance, &controls, &treatments, 0),
            &stable_null(&scope, &first_provenance),
            &estimator_config(),
        )
        .expect("first process");
        let second = estimate_paired_experiment(
            &duration_stream(&scope, &second_provenance, &controls, &treatments, 0),
            &stable_null(&scope, &second_provenance),
            &estimator_config(),
        )
        .expect("second process");
        assert!(first.reproduces_within(&second).expect("compatible replay"));
        assert!(matches!(
            first.reproduction_delta_log(&first),
            Err(PairedEstimatorError::ReusedRunId)
        ));
    }

    #[test]
    fn seeded_order_is_deterministic_randomized_and_balanced() {
        let first = seeded_balanced_pair_order(PERF_MIN_RUNS, 42).expect("first schedule");
        let second = seeded_balanced_pair_order(PERF_MIN_RUNS, 42).expect("second schedule");
        assert_eq!(first, second);
        assert_ne!(
            first,
            vec![
                PerfSampleArm::Control,
                PerfSampleArm::Treatment,
                PerfSampleArm::Control,
                PerfSampleArm::Treatment,
                PerfSampleArm::Control,
                PerfSampleArm::Treatment,
                PerfSampleArm::Control,
                PerfSampleArm::Treatment,
                PerfSampleArm::Control,
                PerfSampleArm::Treatment,
            ],
            "the seed must randomize more than fixed alternation"
        );
        assert_eq!(
            first
                .iter()
                .filter(|arm| **arm == PerfSampleArm::Control)
                .count(),
            PERF_MIN_RUNS / 2
        );
    }

    #[test]
    fn complete_matrix_covers_every_gate_and_required_cross_products() {
        let matrix = PerfMatrixSpec::complete();
        validate_matrix(&matrix).expect("complete matrix");
        assert_eq!(
            matrix
                .for_gate(PerfGate::Qg1)
                .into_iter()
                .filter(|cell| cell.metric == "docs_per_second")
                .count(),
            4 * 9 * 2
        );
        assert_eq!(matrix.for_gate(PerfGate::Qg1).len(), 4 * 9 * 2 + 2);
        assert_eq!(matrix.for_gate(PerfGate::Qg3).len(), 5);
        assert_eq!(matrix.for_gate(PerfGate::Qg5).len(), 3);
        assert_eq!(matrix.for_gate(PerfGate::Qg6).len(), 5 * 2 * 2);
        assert_eq!(matrix.for_gate(PerfGate::Qg8).len(), 6);
        let qg10 = matrix.for_gate(PerfGate::Qg10);
        assert_eq!(qg10.len(), 1);
        assert_eq!(qg10[0].threads, Some(1));
    }

    #[test]
    fn qg1_matrix_identity_is_golden_ordered_and_roundtrips() {
        let matrix = PerfMatrixSpec::complete();
        let qg1_cells = matrix.for_gate(PerfGate::Qg1);
        assert_eq!(qg1_cells.len(), 74);

        let qg1_sha256 = matrix
            .gate_contract_sha256(PerfGate::Qg1)
            .expect("hash canonical QG-1 matrix");
        assert_eq!(qg1_sha256, PerfMatrixSpec::QG1_CANONICAL_SHA256);
        assert_eq!(qg1_sha256.len(), 64);
        assert!(
            qg1_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );

        let cell_hashes = qg1_cells
            .iter()
            .map(|cell| cell.contract_sha256().expect("hash QG-1 cell"))
            .collect::<BTreeSet<_>>();
        assert_eq!(cell_hashes.len(), qg1_cells.len());

        let encoded = serde_json::to_vec(&matrix).expect("encode matrix");
        let decoded: PerfMatrixSpec = serde_json::from_slice(&encoded).expect("decode matrix");
        assert_eq!(
            decoded
                .gate_contract_sha256(PerfGate::Qg1)
                .expect("hash decoded QG-1 matrix"),
            qg1_sha256
        );

        let mut reordered = matrix.clone();
        let qg1_indices = reordered
            .cells
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| (cell.gate == PerfGate::Qg1).then_some(index))
            .collect::<Vec<_>>();
        reordered.cells.swap(qg1_indices[0], qg1_indices[1]);
        assert_ne!(
            reordered
                .gate_contract_sha256(PerfGate::Qg1)
                .expect("hash reordered QG-1 matrix"),
            qg1_sha256
        );

        let mut changed_cell = matrix.clone();
        changed_cell.cells[qg1_indices[0]]
            .fixture
            .push_str("/mutated");
        assert_ne!(
            changed_cell
                .gate_contract_sha256(PerfGate::Qg1)
                .expect("hash cell-mutated QG-1 matrix"),
            qg1_sha256
        );
    }

    #[test]
    fn qg1_profile_plans_have_frozen_exhaustive_applicability_counts() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let cases = [
            (
                ExecutionProfileId::Physical64,
                56,
                2,
                16,
                Some(64),
                Some(64),
                "2b9d450b21a3b051a5b05a479709dbbca482d7d27f1221ddd77adaed2934f5b3",
            ),
            (
                ExecutionProfileId::Smt2_128,
                72,
                2,
                0,
                Some(128),
                Some(128),
                "bb66e93d76bee49640023abbc56738361dc6d9729eb4f8739af228df9785a56e",
            ),
            (
                ExecutionProfileId::Scheduler10,
                32,
                2,
                40,
                Some(10),
                Some(8),
                "24787b922c8ce58fd6166a7475def35e498986500b7fa27e88a05a566a0bbdc5",
            ),
        ];
        let mut plan_hashes = BTreeSet::new();
        let mut observed_plan_hashes = Vec::new();
        let mut expected_plan_hashes = Vec::new();
        for (
            profile,
            required,
            diagnostic,
            not_applicable,
            execution_capacity,
            max_exercised_cell_width,
            expected_plan_sha256,
        ) in cases
        {
            let plan = qg1_plan(&registry, profile);
            assert_eq!(plan.cells.len(), 74);
            assert_eq!(plan.cell_count(PerfCellApplicability::Required), required);
            assert_eq!(
                plan.cell_count(PerfCellApplicability::Diagnostic),
                diagnostic
            );
            assert_eq!(
                plan.cell_count(PerfCellApplicability::NotApplicable),
                not_applicable
            );
            assert_eq!(plan.execution_capacity, execution_capacity);
            assert_eq!(plan.max_exercised_cell_width, max_exercised_cell_width);
            assert_eq!(
                plan.binding.gate_matrix_contract_sha256,
                PerfMatrixSpec::QG1_CANONICAL_SHA256
            );
            assert_eq!(
                plan.binding.normalized_perf_manifest_sha256,
                perf_manifest_contract_sha256(PERF_MANIFEST)
            );
            assert_eq!(plan.binding.primary_target_cell_width, Some(8));
            assert_eq!(
                plan.binding.registry_schema_version,
                MACHINE_CLASS_REGISTRY_SCHEMA_VERSION
            );
            assert_eq!(plan.binding.registry_sha256, MACHINE_CLASS_REGISTRY_SHA256);
            assert!(is_lower_hex_digest(&plan.binding.profile_contract_sha256));
            observed_plan_hashes.push((profile, plan.binding.applicability_plan_sha256.clone()));
            expected_plan_hashes.push((profile, expected_plan_sha256.to_owned()));
            assert!(
                plan.cells
                    .iter()
                    .enumerate()
                    .all(|(ordinal, cell)| cell.ordinal == ordinal
                        && is_lower_hex_digest(&cell.cell_contract_sha256))
            );
            assert!(
                plan.cells.iter().any(|cell| {
                    cell.configured_threads == 8
                        && cell.applicability == PerfCellApplicability::Required
                }),
                "{profile:?} must retain the QG-1 primary target width as Required"
            );
            plan.verify_against(&PerfMatrixSpec::complete(), &registry)
                .expect("recompute canonical plan");
            assert!(
                plan_hashes.insert(plan.binding.applicability_plan_sha256),
                "each execution profile requires a distinct plan hash"
            );
        }
        assert_eq!(
            observed_plan_hashes, expected_plan_hashes,
            "all profile plan hashes must remain frozen"
        );
    }

    #[test]
    fn qg1_plan_identity_binds_manifest_and_primary_target_independently() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let matrix = PerfMatrixSpec::complete();
        let trj_key = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        let trj = registry
            .execution_profile(trj_key)
            .expect("physical Threadripper profile");
        let normative_identity = perf_gate_manifest_identity(PERF_MANIFEST, PerfGate::Qg1)
            .expect("normative QG-1 manifest identity");
        let normative = matrix
            .applicability_plan_for_profile(trj, PerfGate::Qg1, &normative_identity)
            .expect("normative physical QG-1 plan");

        let manifest_only_change = PERF_MANIFEST.replacen(
            "Quill performance gate manifests",
            "Quill performance gate contract manifests",
            1,
        );
        let manifest_only_identity =
            perf_gate_manifest_identity(&manifest_only_change, PerfGate::Qg1)
                .expect("manifest-only QG-1 identity");
        assert_eq!(
            manifest_only_identity.primary_target_cell_width,
            normative_identity.primary_target_cell_width
        );
        assert_ne!(
            manifest_only_identity.normalized_perf_manifest_sha256,
            normative_identity.normalized_perf_manifest_sha256
        );
        let manifest_only_plan = matrix
            .applicability_plan_for_profile(trj, PerfGate::Qg1, &manifest_only_identity)
            .expect("manifest-only physical QG-1 plan");
        assert_eq!(manifest_only_plan.cells, normative.cells);
        assert_ne!(
            manifest_only_plan.binding.applicability_plan_sha256,
            normative.binding.applicability_plan_sha256
        );

        let activation_only_change =
            PERF_MANIFEST.replacen("activated = false", "activated = true", 1);
        let activation_only_identity =
            perf_gate_manifest_identity(&activation_only_change, PerfGate::Qg1)
                .expect("activation-only QG-1 identity");
        let activation_only_plan = matrix
            .applicability_plan_for_profile(trj, PerfGate::Qg1, &activation_only_identity)
            .expect("activation-only physical QG-1 plan");
        assert_eq!(activation_only_identity, normative_identity);
        assert_eq!(
            activation_only_plan.binding.applicability_plan_sha256,
            normative.binding.applicability_plan_sha256
        );

        let mut primary_only_identity = normative_identity.clone();
        primary_only_identity.primary_target_cell_width = Some(16);
        let primary_only_plan = matrix
            .applicability_plan_for_profile(trj, PerfGate::Qg1, &primary_only_identity)
            .expect("physical Threadripper supports the mutated target");
        assert_eq!(
            primary_only_plan.binding.normalized_perf_manifest_sha256,
            normative.binding.normalized_perf_manifest_sha256
        );
        assert_ne!(
            primary_only_plan.binding.applicability_plan_sha256,
            normative.binding.applicability_plan_sha256
        );

        let m4_key = profile_key(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10);
        let m4 = registry
            .execution_profile(m4_key)
            .expect("M4 scheduler profile");
        assert!(matches!(
            matrix.applicability_plan_for_profile(
                m4,
                PerfGate::Qg1,
                &primary_only_identity
            ),
            Err(PerfApplicabilityPlanError::RequiredProfileBelowPrimaryTarget {
                profile,
                gate: PerfGate::Qg1,
                primary_target_cell_width: 16,
                execution_capacity: Some(10),
                max_exercised_cell_width: Some(8),
            }) if profile == m4_key
        ));

        let mut stored_manifest_mutation = normative.clone();
        stored_manifest_mutation
            .binding
            .normalized_perf_manifest_sha256 = "0".repeat(64);
        stored_manifest_mutation.binding.applicability_plan_sha256 =
            stored_manifest_mutation.contract_sha256();
        assert!(matches!(
            stored_manifest_mutation.verify_against(&matrix, &registry),
            Err(PerfApplicabilityPlanError::PlanMismatch { .. })
        ));

        let mut stored_primary_mutation = normative;
        stored_primary_mutation.binding.primary_target_cell_width = Some(16);
        stored_primary_mutation.binding.applicability_plan_sha256 =
            stored_primary_mutation.contract_sha256();
        assert!(matches!(
            stored_primary_mutation.verify_against(&matrix, &registry),
            Err(PerfApplicabilityPlanError::PlanMismatch { .. })
        ));
    }

    #[test]
    fn qg1_manifest_contract_rejects_missing_or_unbounded_primary_target() {
        let missing = PERF_MANIFEST.replacen("primary_target_cell_width = 8\n", "", 1);
        assert!(matches!(
            perf_gate_manifest_identity(&missing, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::ManifestContract {
                gate: PerfGate::Qg1,
                ..
            })
        ));

        for invalid in ["0", "-1", "\"eight\""] {
            let mutated = PERF_MANIFEST.replacen(
                "primary_target_cell_width = 8",
                &format!("primary_target_cell_width = {invalid}"),
                1,
            );
            assert!(
                matches!(
                    perf_gate_manifest_identity(&mutated, PerfGate::Qg1),
                    Err(PerfApplicabilityPlanError::ManifestContract {
                        gate: PerfGate::Qg1,
                        ..
                    })
                ),
                "invalid primary target {invalid} must fail closed"
            );
        }

        let stale_schema = PERF_MANIFEST.replacen(
            "applicability_plan = \"frankensearch.quill-perf-applicability-plan.v2\"",
            "applicability_plan = \"frankensearch.quill-perf-applicability-plan.v1\"",
            1,
        );
        assert!(matches!(
            perf_gate_manifest_identity(&stale_schema, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::ManifestContract {
                gate: PerfGate::Qg1,
                ..
            })
        ));
    }

    #[test]
    fn qg1_required_profiles_retain_all_primary_width_cells() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        for profile in [
            ExecutionProfileId::Physical64,
            ExecutionProfileId::Smt2_128,
            ExecutionProfileId::Scheduler10,
        ] {
            let plan = qg1_plan(&registry, profile);
            let required_primary_cells = plan
                .cells
                .iter()
                .filter(|cell| {
                    cell.configured_threads == 8
                        && cell.applicability == PerfCellApplicability::Required
                })
                .count();
            assert_eq!(
                required_primary_cells, 8,
                "{profile:?} must retain all eight ordinary QG-1 width-8 cells"
            );
        }
    }

    #[test]
    fn qg1_not_applicable_reason_facts_are_hash_bound_and_reconstructed() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let matrix = PerfMatrixSpec::complete();
        let plan = qg1_plan(&registry, ExecutionProfileId::Scheduler10);
        let entry_index = plan
            .cells
            .iter()
            .position(|cell| cell.applicability == PerfCellApplicability::NotApplicable)
            .expect("M4 QG-1 has N/A cells");
        let reason = plan.cells[entry_index].reason;
        let profile = plan.binding.profile;
        let capacity_semantics = ExecutionCapacitySemantics::SchedulerWorkers;
        let execution_capacity = 10;
        let required_cell_width = 16;
        let max_exercised_cell_width = 8;
        assert_eq!(
            reason,
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile,
                capacity_semantics,
                execution_capacity,
                required_cell_width,
                max_exercised_cell_width,
            },
            "M4 N/A cell must carry the exact typed execution-envelope facts"
        );
        assert_eq!(profile, plan.binding.profile);
        assert_eq!(
            capacity_semantics,
            ExecutionCapacitySemantics::SchedulerWorkers
        );
        assert_eq!(execution_capacity, 10);
        assert_eq!(required_cell_width, 16);
        assert_eq!(max_exercised_cell_width, 8);

        let baseline_hash = plan.binding.applicability_plan_sha256.clone();
        let assert_mutation_rejected = |mutated_reason: PerfCellApplicabilityReason,
                                        field: &str| {
            let mut mutated = plan.clone();
            mutated.cells[entry_index].reason = mutated_reason;
            mutated.binding.applicability_plan_sha256 = mutated.contract_sha256();
            assert_ne!(
                mutated.binding.applicability_plan_sha256, baseline_hash,
                "{field} must participate in the plan hash"
            );
            assert!(
                matches!(
                    mutated.verify_against(&matrix, &registry),
                    Err(PerfApplicabilityPlanError::PlanMismatch { .. })
                ),
                "{field} mutation must fail reconstruction"
            );
        };
        let physical_key = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        assert_mutation_rejected(
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile: physical_key,
                capacity_semantics,
                execution_capacity,
                required_cell_width,
                max_exercised_cell_width,
            },
            "profile",
        );
        assert_mutation_rejected(
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile,
                capacity_semantics: ExecutionCapacitySemantics::PhysicalCores,
                execution_capacity,
                required_cell_width,
                max_exercised_cell_width,
            },
            "capacity_semantics",
        );
        assert_mutation_rejected(
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile,
                capacity_semantics,
                execution_capacity: execution_capacity + 1,
                required_cell_width,
                max_exercised_cell_width,
            },
            "execution_capacity",
        );
        assert_mutation_rejected(
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile,
                capacity_semantics,
                execution_capacity,
                required_cell_width: required_cell_width + 1,
                max_exercised_cell_width,
            },
            "required_cell_width",
        );
        assert_mutation_rejected(
            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                profile,
                capacity_semantics,
                execution_capacity,
                required_cell_width,
                max_exercised_cell_width: max_exercised_cell_width + 1,
            },
            "max_exercised_cell_width",
        );

        let encoded = serde_json::to_vec(&reason).expect("serialize typed N/A reason");
        let decoded: PerfCellApplicabilityReason =
            serde_json::from_slice(&encoded).expect("deserialize typed N/A reason");
        assert_eq!(decoded, reason);
    }

    #[test]
    fn m4_scheduler_capacity_does_not_invent_a_width_ten_cell() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let plan = qg1_plan(&registry, ExecutionProfileId::Scheduler10);
        assert_eq!(plan.execution_capacity, Some(10));
        assert_eq!(plan.max_exercised_cell_width, Some(8));
        assert_eq!(plan.max_runnable_cell_width(), Some(8));
        assert!(plan.cells.iter().all(|cell| cell.configured_threads != 10));
        assert!(
            plan.cells
                .iter()
                .filter(|cell| cell.configured_threads > 8)
                .all(|cell| {
                    cell.applicability == PerfCellApplicability::NotApplicable
                        && matches!(
                            cell.reason,
                            PerfCellApplicabilityReason::ExceedsProfileMaximum {
                                profile,
                                capacity_semantics:
                                    ExecutionCapacitySemantics::SchedulerWorkers,
                                execution_capacity: 10,
                                required_cell_width,
                                max_exercised_cell_width: 8,
                            } if profile == plan.binding.profile
                                && required_cell_width
                                    == u64::try_from(cell.configured_threads)
                                        .expect("canonical width fits u64")
                        )
                })
        );
    }

    #[test]
    fn unbounded_x86_diagnostic_profile_returns_typed_no_claim_error() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let key = profile_key(
            HardwareClassId::X86VpsOvh,
            ExecutionProfileId::X86Diagnostic,
        );
        let profile = registry
            .execution_profile(key)
            .expect("frozen x86 diagnostic profile");
        let policy = profile
            .gate_policy(PerfGate::Qg1.label())
            .expect("x86 QG-1 policy");
        assert_eq!(
            policy.default_flip_disposition(),
            DefaultFlipDisposition::DiagnosticOnly
        );
        assert_eq!(profile.execution_capacity(), None);
        assert_eq!(policy.max_exercised_cell_width(), None);
        assert!(matches!(
            PerfMatrixSpec::complete().applicability_plan(&registry, key, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::UnboundedDiagnosticProfile {
                profile,
                gate: PerfGate::Qg1,
                execution_capacity: None,
                max_exercised_cell_width: None,
            }) if profile == key
        ));
    }

    #[test]
    fn bounded_diagnostic_envelope_marks_wider_cells_na_and_never_required() {
        let profile = profile_key(
            HardwareClassId::X86VpsOvh,
            ExecutionProfileId::X86Diagnostic,
        );
        let envelope = BoundedProfileApplicabilityEnvelope {
            profile,
            capacity_semantics: ExecutionCapacitySemantics::DiagnosticWorkerBudget,
            execution_capacity: 8,
            disposition: DefaultFlipDisposition::DiagnosticOnly,
            max_exercised_cell_width: 8,
        };
        let classifications = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .map(|cell| {
                let configured_width = u64::try_from(
                    cell.threads
                        .expect("canonical QG-1 cell has a configured width"),
                )
                .expect("canonical QG-1 width fits u64");
                (
                    configured_width,
                    envelope.classify_cell(configured_width, canonical_cell_is_diagnostic(cell)),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(classifications.len(), 74);
        assert_eq!(
            classifications
                .iter()
                .filter(|(_, (applicability, _))| {
                    *applicability == PerfCellApplicability::Diagnostic
                })
                .count(),
            34
        );
        assert_eq!(
            classifications
                .iter()
                .filter(|(_, (applicability, _))| {
                    *applicability == PerfCellApplicability::NotApplicable
                })
                .count(),
            40
        );
        let any_required = classifications
            .iter()
            .any(|(_, (applicability, _))| *applicability == PerfCellApplicability::Required);
        assert!(
            !any_required,
            "a diagnostic-only envelope must force PerfEvidenceArtifact::fold onto its \
             evidence.gate_without_required_cells NoDecision path, never an Allow path"
        );
        assert!(
            classifications
                .iter()
                .all(|(configured_width, (applicability, reason))| {
                    if *configured_width > 8 {
                        *applicability == PerfCellApplicability::NotApplicable
                            && matches!(
                                reason,
                                PerfCellApplicabilityReason::ExceedsProfileMaximum {
                                    profile: reason_profile,
                                    capacity_semantics:
                                        ExecutionCapacitySemantics::DiagnosticWorkerBudget,
                                    execution_capacity: 8,
                                    required_cell_width,
                                    max_exercised_cell_width: 8,
                                } if *reason_profile == profile
                                    && *required_cell_width == *configured_width
                            )
                    } else {
                        *applicability == PerfCellApplicability::Diagnostic
                            && *reason == PerfCellApplicabilityReason::DiagnosticProfile
                    }
                })
        );
    }

    #[test]
    fn required_but_unavailable_m5_returns_typed_error_not_all_na_plan() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let key = profile_key(HardwareClassId::M5Macos, ExecutionProfileId::Scheduler14);
        let profile = registry
            .execution_profile(key)
            .expect("reserved M5 profile resolves");
        assert_eq!(
            profile.availability(),
            MachineProfileAvailability::Unavailable
        );
        assert_eq!(
            profile
                .gate_policy(PerfGate::Qg1.label())
                .expect("M5 QG-1 policy")
                .default_flip_disposition(),
            DefaultFlipDisposition::RequiredForDefaultFlip
        );
        assert!(matches!(
            PerfMatrixSpec::complete().applicability_plan(&registry, key, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::ProfileUnavailable {
                profile,
                gate: PerfGate::Qg1,
            }) if profile == key
        ));
    }

    #[test]
    fn profile_plans_cover_the_complete_matrix_and_reject_plan_or_matrix_drift() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let matrix = PerfMatrixSpec::complete();
        let key = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        let plans = matrix
            .applicability_plans(&registry, key)
            .expect("all physical-core plans");
        assert_eq!(plans.len(), PerfGate::ALL.len());
        assert_eq!(
            plans.iter().map(|plan| plan.cells.len()).sum::<usize>(),
            matrix.cells.len()
        );
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.binding.gate)
                .collect::<Vec<_>>(),
            PerfGate::ALL
        );

        let encoded = serde_json::to_vec(&plans[0]).expect("serialize applicability plan");
        let decoded: PerfApplicabilityPlan =
            serde_json::from_slice(&encoded).expect("deserialize applicability plan");
        assert_eq!(decoded, plans[0]);
        decoded
            .verify_against(&matrix, &registry)
            .expect("round-tripped plan recomputes");

        let mut reordered = plans[0].clone();
        reordered.cells.swap(0, 1);
        assert!(matches!(
            reordered.verify_against(&matrix, &registry),
            Err(PerfApplicabilityPlanError::PlanMismatch { .. })
        ));

        let mut incomplete = matrix.clone();
        incomplete.cells.remove(0);
        assert!(matches!(
            incomplete.applicability_plan(&registry, key, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::NonCanonicalMatrix)
        ));
    }

    #[test]
    fn cell_identity_binds_every_serialized_contract_field() {
        let cell = PerfCellSpec {
            gate: PerfGate::Qg6,
            fixture: "query/identifier/k10/100k".to_owned(),
            metric: "latency_ms".to_owned(),
            corpus: Some(PerfCorpus::Medium),
            document_count: Some(100_000),
            threads: Some(8),
            writer_heap_bytes: Some(perf_writer_heap_bytes(8)),
            positions: Some(PositionMode::On),
            tombstone_density_pct: Some(10),
            query_class: Some(PerfQueryClass::Identifier),
            k: Some(10),
            topology: Some(PerfTopology::InProcess),
        };
        let baseline = cell.contract_sha256().expect("hash complete cell");
        let assert_changed = |mutated: PerfCellSpec, field: &str| {
            assert_ne!(
                mutated.contract_sha256().expect("hash mutated cell"),
                baseline,
                "{field} must participate in the cell-contract identity"
            );
        };

        let mut mutated = cell.clone();
        mutated.gate = PerfGate::Qg1;
        assert_changed(mutated, "gate");
        let mut mutated = cell.clone();
        mutated.fixture.push_str("/mutated");
        assert_changed(mutated, "fixture");
        let mut mutated = cell.clone();
        mutated.metric.push_str("_mutated");
        assert_changed(mutated, "metric");
        let mut mutated = cell.clone();
        mutated.corpus = Some(PerfCorpus::Xlarge);
        assert_changed(mutated, "corpus");
        let mut mutated = cell.clone();
        mutated.document_count = Some(1_000_000);
        assert_changed(mutated, "document_count");
        let mut mutated = cell.clone();
        mutated.threads = Some(16);
        assert_changed(mutated, "threads");
        let mut mutated = cell.clone();
        mutated.writer_heap_bytes = Some(perf_writer_heap_bytes(16));
        assert_changed(mutated, "writer_heap_bytes");
        let mut mutated = cell.clone();
        mutated.positions = Some(PositionMode::Off);
        assert_changed(mutated, "positions");
        let mut mutated = cell.clone();
        mutated.tombstone_density_pct = Some(20);
        assert_changed(mutated, "tombstone_density_pct");
        let mut mutated = cell.clone();
        mutated.query_class = Some(PerfQueryClass::Boolean);
        assert_changed(mutated, "query_class");
        let mut mutated = cell.clone();
        mutated.k = Some(100);
        assert_changed(mutated, "k");
        let mut mutated = cell;
        mutated.topology = Some(PerfTopology::FreshProcess);
        assert_changed(mutated, "topology");
    }

    #[test]
    fn qg6_matrix_rejects_missing_duplicate_reclassified_and_non_warm_cells() {
        let complete = PerfMatrixSpec::complete();
        let first_qg6 = complete
            .cells
            .iter()
            .position(|cell| cell.gate == PerfGate::Qg6)
            .expect("QG-6 cell");

        let mut missing = complete.clone();
        missing.cells.remove(first_qg6);
        assert!(validate_matrix(&missing).is_err());

        let mut duplicate = complete.clone();
        duplicate.cells.push(duplicate.cells[first_qg6].clone());
        assert!(validate_matrix(&duplicate).is_err());

        let mut reclassified = complete.clone();
        reclassified.cells[first_qg6].query_class = Some(PerfQueryClass::Boolean);
        assert!(validate_matrix(&reclassified).is_err());

        for substituted_metric in ["cold_open_latency_ms", "stage_parse_latency_ms"] {
            let mut substituted = complete.clone();
            substituted.cells[first_qg6].metric = substituted_metric.to_owned();
            assert!(
                validate_matrix(&substituted).is_err(),
                "{substituted_metric} must not substitute for warm total search"
            );
        }
    }

    #[test]
    fn manifest_contract_hash_ignores_only_activation_state() {
        let manifest = PERF_MANIFEST;
        assert_eq!(manifest.matches("activated = false").count(), 10);
        assert_eq!(
            perf_manifest_contract_sha256(manifest),
            "404c0e24c9f3f4919e3b9c3213e722c77bcdb89ea2f991d0a66dc67eafd0fc89",
            "the normalized all-inactive manifest digest must remain frozen"
        );
        assert_eq!(
            perf_manifest_contract_sha256(manifest),
            lower_sha256_hex(manifest.as_bytes()),
            "the all-inactive manifest's normalized digest equals its raw-file digest"
        );

        let activated = manifest.replacen("activated = false", "activated = true", 1);
        assert_eq!(
            perf_manifest_contract_sha256(manifest),
            perf_manifest_contract_sha256(&activated),
            "administrative activation must not invalidate pre-flip evidence"
        );

        let changed_comment =
            manifest.replacen("PROVISIONAL until `activated = true`", "measured", 1);
        assert_ne!(
            perf_manifest_contract_sha256(manifest),
            perf_manifest_contract_sha256(&changed_comment),
            "only assignment lines are administrative; comments remain hash-bound"
        );
        assert_ne!(
            perf_manifest_contract_sha256("activated = true # explanatory suffix\n"),
            perf_manifest_contract_sha256("activated = false # explanatory suffix\n"),
            "assignment-like prose with a suffix remains hash-bound"
        );

        let changed_target = manifest.replacen("docs_per_sec >= 3.0x", "docs_per_sec >= 3.1x", 1);
        assert_ne!(
            perf_manifest_contract_sha256(manifest),
            perf_manifest_contract_sha256(&changed_target),
            "a performance-contract change must invalidate old evidence"
        );
    }

    #[test]
    fn normative_manifest_names_the_repaired_paired_estimator() {
        let manifest: toml::Value = toml::from_str(include_str!(
            "../../../docs/contracts/quill-perf-gates.toml"
        ))
        .expect("parse normative performance manifest");
        let defaults = manifest
            .get("defaults")
            .and_then(toml::Value::as_table)
            .expect("defaults table");
        assert!(
            defaults["stat_rule"]
                .as_str()
                .expect("statistical rule")
                .contains(PAIRED_ESTIMATOR_SCHEMA_VERSION)
        );
        assert_eq!(
            defaults["decision_precedence"].as_str(),
            Some("Fatal > Block > Quarantine > NoDecision/Provisional > Allow")
        );
    }

    #[test]
    fn distribution_reports_median_ci_and_cv_provenance() {
        let samples = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0];
        let summary = DistributionSummary::from_samples(&samples).expect("summary");
        assert!((summary.p50 - 10.0).abs() < f64::EPSILON);
        assert!(summary.median_ci95_low <= summary.p50);
        assert!(summary.p50 <= summary.median_ci95_high);
        assert_eq!(summary.runs, PERF_MIN_RUNS);
        assert!(summary.mad <= 0.1);
        assert!(summary.cv_pct < 2.0);
        assert!(summary.sampled_for_activation());
        let repeated =
            DistributionSummary::from_samples(&samples).expect("deterministic repeated summary");
        assert_eq!(
            summary.median_ci95_low.to_bits(),
            repeated.median_ci95_low.to_bits()
        );
        assert_eq!(
            summary.median_ci95_high.to_bits(),
            repeated.median_ci95_high.to_bits()
        );
        let high_cv = DistributionSummary::from_samples(&[
            1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0,
        ])
        .expect("high-CV provenance");
        assert!(high_cv.cv_pct > 5.0);
        assert!(
            high_cv.sampled_for_activation(),
            "CV is provenance and must not be an activation gate"
        );
        let even =
            DistributionSummary::from_samples(&[1.0, 2.0, 100.0, 200.0]).expect("even summary");
        assert!((even.p50 - 51.0).abs() < f64::EPSILON);
        assert!(DistributionSummary::from_samples(&[]).is_err());
        assert!(DistributionSummary::from_samples(&[f64::NAN]).is_err());
        assert!(DistributionSummary::from_samples(&[-1.0]).is_err());
    }

    #[test]
    fn artifact_v7_json_roundtrips_and_binds_profile_plan_and_capacity_identity() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let plan = qg1_plan(&registry, ExecutionProfileId::Smt2_128);
        let applicability_plan = plan.binding().clone();
        let matrix = PerfMatrixSpec::complete();
        let selected_specs = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .zip(&plan.cells)
            .filter_map(|(spec, classification)| {
                classification.applicability.is_runnable().then_some(spec)
            })
            .collect::<Vec<_>>();
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            applicability_plan: Some(applicability_plan.clone()),
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            execution: Some(PerfExecutionProvenance {
                host_identity: "test-host".to_owned(),
                producer_os: PerfProducerOs::Linux,
                physical_cores: 64,
                logical_threads: 128,
                process_available_threads: 128,
                execution_capacity: 128,
                max_exercised_cell_width: 128,
                configured_engine_thread_widths: vec![1, 2, 4, 8, 16, 32, 64, 96, 128],
                runtime_detected_isa: vec!["avx2".to_owned(), "bmi2".to_owned(), "fma".to_owned()],
                cpu_affinity_allowed_list: Some("0-127".to_owned()),
                affinity_or_cpuset_cap: None,
            }),
            git_rev: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            run_window: "test-window".to_owned(),
            run_id: "candidate".to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: applicability_plan.normalized_perf_manifest_sha256.clone(),
            cells: threshold_rows(PerfGate::Qg1, &selected_specs),
            laws_attested: true,
        };
        let json = artifact.to_json_pretty().expect("artifact JSON");
        let value: serde_json::Value = serde_json::from_str(&json).expect("decode artifact");
        assert_eq!(PERF_ARTIFACT_SCHEMA_VERSION, "quill-perf-artifact-v7");
        assert_eq!(
            value["schema_version"].as_str(),
            Some(PERF_ARTIFACT_SCHEMA_VERSION)
        );
        for key in [
            "schema_version",
            "gate",
            "applicability_plan",
            "bench_elf_sha256",
            "machine_fingerprint",
            "execution",
            "git_rev",
            "run_window",
            "run_id",
            "corpus_manifest_hash",
            "manifest_sha256",
            "cells",
            "laws_attested",
        ] {
            assert!(value.get(key).is_some(), "missing required field {key}");
        }
        let decoded = PerfGateArtifact::from_verified_measured_slice(json.as_bytes())
            .expect("round-trip verified typed v7 artifact");
        assert_eq!(decoded, artifact);
        assert_eq!(
            decoded.applicability_plan.as_ref(),
            Some(&applicability_plan)
        );

        let artifact_sha256 = lower_sha256_hex(json.as_bytes());
        let mut profile_drift = artifact.clone();
        profile_drift
            .applicability_plan
            .as_mut()
            .expect("measured v7 binding")
            .profile = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        let profile_drift_json = profile_drift
            .to_json_pretty()
            .expect("serialize profile-drifted artifact");
        assert_ne!(profile_drift_json, json);
        assert_ne!(
            lower_sha256_hex(profile_drift_json.as_bytes()),
            artifact_sha256
        );

        let mut plan_drift = artifact.clone();
        plan_drift
            .applicability_plan
            .as_mut()
            .expect("measured v7 binding")
            .applicability_plan_sha256 = "d".repeat(64);
        let plan_drift_json = plan_drift
            .to_json_pretty()
            .expect("serialize plan-drifted artifact");
        assert_ne!(plan_drift_json, json);
        assert_ne!(
            lower_sha256_hex(plan_drift_json.as_bytes()),
            artifact_sha256
        );

        let table = artifact.human_table();
        assert!(table.contains("cv_pct"));
        assert!(table.contains("median_ci95"));
        assert!(table.contains("bulk/tiny/1/positions_on"));
        assert!(table.contains("sampled"));
        assert!(
            table.contains("configured_engine_thread_widths=[1, 2, 4, 8, 16, 32, 64, 96, 128]")
        );
        assert!(table.contains("execution_capacity=128"));
        assert!(table.contains("max_exercised_cell_width=128"));

        assert_eq!(
            artifact
                .verify_current_measured_contract()
                .expect("verified measured threshold"),
            plan
        );

        let mut capacity_drift = artifact.clone();
        capacity_drift
            .execution
            .as_mut()
            .expect("measured v7 execution provenance")
            .execution_capacity = 64;
        assert!(
            !capacity_drift
                .execution
                .as_ref()
                .expect("mutated execution provenance")
                .is_complete(),
            "capacity below the process and selected width must fail closed"
        );

        let mut maximum_drift = artifact.clone();
        maximum_drift
            .execution
            .as_mut()
            .expect("measured v7 execution provenance")
            .max_exercised_cell_width = 64;
        assert!(
            !maximum_drift
                .execution
                .as_ref()
                .expect("mutated execution provenance")
                .is_complete(),
            "a selected width above the plan maximum must fail closed"
        );

        let mut strict_execution = serde_json::to_value(
            artifact
                .execution
                .as_ref()
                .expect("measured v7 execution provenance"),
        )
        .expect("serialize strict execution provenance");
        strict_execution
            .as_object_mut()
            .expect("execution provenance is an object")
            .insert("caller_thread_budget".to_owned(), serde_json::json!(128));
        assert!(
            serde_json::from_value::<PerfExecutionProvenance>(strict_execution).is_err(),
            "unknown caller-authoritative capacity fields must be rejected"
        );
    }

    #[test]
    fn measured_threshold_verified_reload_reconstructs_every_gate_and_rejects_drift() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let profile = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        for gate in PerfGate::ALL {
            let matrix = PerfMatrixSpec::complete();
            let plan = matrix
                .applicability_plan(&registry, profile, gate)
                .expect("canonical per-gate plan");
            let execution_capacity = plan.execution_capacity.expect("bounded profile capacity");
            let max_exercised_cell_width = plan
                .max_exercised_cell_width
                .expect("bounded per-gate maximum");
            let runnable = matrix
                .for_gate(gate)
                .into_iter()
                .zip(&plan.cells)
                .filter(|(_, classification)| classification.applicability.is_runnable())
                .collect::<Vec<_>>();
            let (selected_spec, selected_classification) = runnable[0];
            let artifact = PerfGateArtifact {
                schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
                gate,
                applicability_plan: Some(plan.binding().clone()),
                bench_elf_sha256: "c".repeat(64),
                machine_fingerprint: "linux-x86_64-test".to_owned(),
                execution: Some(PerfExecutionProvenance {
                    host_identity: "test-host".to_owned(),
                    producer_os: PerfProducerOs::Linux,
                    physical_cores: 64,
                    logical_threads: 128,
                    process_available_threads: usize::try_from(execution_capacity)
                        .expect("test capacity fits usize"),
                    execution_capacity,
                    max_exercised_cell_width,
                    configured_engine_thread_widths: vec![
                        selected_classification.configured_threads,
                    ],
                    runtime_detected_isa: vec![
                        "avx2".to_owned(),
                        "bmi2".to_owned(),
                        "fma".to_owned(),
                    ],
                    cpu_affinity_allowed_list: Some("0-63".to_owned()),
                    affinity_or_cpuset_cap: Some(
                        "Cpus_allowed_list=0-63 (64 of 128 host logical threads)".to_owned(),
                    ),
                }),
                git_rev: "0".repeat(40),
                run_window: "test-window".to_owned(),
                run_id: format!("{}-verified-reload", gate.label()),
                corpus_manifest_hash: "a".repeat(64),
                manifest_sha256: plan.binding.normalized_perf_manifest_sha256.clone(),
                cells: threshold_rows(gate, &[selected_spec]),
                laws_attested: runnable.len() == 1,
            };
            let bytes = serde_json::to_vec_pretty(&artifact).expect("canonical threshold bytes");
            let reloaded = PerfGateArtifact::from_verified_measured_slice(&bytes)
                .expect("strict measured threshold reload");
            assert_eq!(reloaded, artifact);

            let mut noncanonical = bytes.clone();
            noncanonical.push(b'\n');
            assert!(PerfGateArtifact::from_verified_measured_slice(&noncanonical).is_err());

            let mut missing_engine_row = artifact.clone();
            missing_engine_row.cells.pop();
            assert!(
                missing_engine_row
                    .verify_current_measured_contract()
                    .is_err(),
                "a partial canonical engine-row group must fail strict reload for {gate}"
            );

            let mut duplicate_engine_row = artifact.clone();
            duplicate_engine_row.cells.push(artifact.cells[0].clone());
            assert!(
                duplicate_engine_row
                    .verify_current_measured_contract()
                    .is_err(),
                "a duplicate engine row must fail strict reload for {gate}"
            );

            let mut altered_cell_spec = artifact.clone();
            altered_cell_spec.cells[0].unit.push_str("-tampered");
            assert!(
                altered_cell_spec
                    .verify_current_measured_contract()
                    .is_err(),
                "an altered cell fixture/metric/engine/unit contract must fail for {gate}"
            );

            let mut width_projection_drift = artifact.clone();
            width_projection_drift
                .execution
                .as_mut()
                .expect("measured execution")
                .configured_engine_thread_widths =
                vec![selected_classification.configured_threads.saturating_add(1)];
            assert!(
                width_projection_drift
                    .verify_current_measured_contract()
                    .is_err(),
                "configured widths must reconstruct from actual selected cells for {gate}"
            );

            let mut law_scope_drift = artifact.clone();
            law_scope_drift.laws_attested = !law_scope_drift.laws_attested;
            assert!(
                law_scope_drift.verify_current_measured_contract().is_err(),
                "laws_attested must exactly describe full runnable coverage for {gate}"
            );

            for identity_field in ["machine_fingerprint", "run_window", "run_id"] {
                let mut identity_drift = artifact.clone();
                match identity_field {
                    "machine_fingerprint" => identity_drift.machine_fingerprint.clear(),
                    "run_window" => identity_drift.run_window = "unmeasured".to_owned(),
                    "run_id" => identity_drift.run_id.clear(),
                    _ => unreachable!("enumerated threshold identity field"),
                }
                assert!(
                    identity_drift.verify_current_measured_contract().is_err(),
                    "invalid measured identity field {identity_field} must fail for {gate}"
                );
            }
            let mut git_identity_drift = artifact.clone();
            git_identity_drift.git_rev = "not-a-git-revision".to_owned();
            assert!(
                git_identity_drift
                    .verify_current_measured_contract()
                    .is_err()
            );

            let mut manifest_drift = artifact.clone();
            manifest_drift.manifest_sha256 = "0".repeat(64);
            assert!(manifest_drift.verify_current_measured_contract().is_err());

            let mut capacity_drift = artifact.clone();
            capacity_drift
                .execution
                .as_mut()
                .expect("measured execution")
                .execution_capacity += 1;
            assert!(capacity_drift.verify_current_measured_contract().is_err());

            let mut incomplete_execution = artifact.clone();
            incomplete_execution
                .execution
                .as_mut()
                .expect("measured execution")
                .host_identity
                .clear();
            assert!(
                incomplete_execution
                    .verify_current_measured_contract()
                    .is_err(),
                "empty host identity must fail the shared strict loader"
            );

            let mut zero_process_width = artifact.clone();
            zero_process_width
                .execution
                .as_mut()
                .expect("measured execution")
                .process_available_threads = 0;
            assert!(
                zero_process_width
                    .verify_current_measured_contract()
                    .is_err(),
                "zero process width must fail the shared strict loader"
            );

            let mut unsorted_widths = artifact.clone();
            unsorted_widths
                .execution
                .as_mut()
                .expect("measured execution")
                .configured_engine_thread_widths = vec![2, 1];
            assert!(
                unsorted_widths.verify_current_measured_contract().is_err(),
                "noncanonical configured widths must fail the shared strict loader"
            );

            let mut plan_drift = artifact.clone();
            plan_drift
                .applicability_plan
                .as_mut()
                .expect("measured plan")
                .gate_matrix_contract_sha256 = "0".repeat(64);
            assert!(plan_drift.verify_current_measured_contract().is_err());

            let mut schema_drift = artifact;
            schema_drift.schema_version = "quill-perf-artifact-v6".to_owned();
            assert!(schema_drift.verify_current_measured_contract().is_err());
        }
    }

    #[test]
    fn measured_threshold_rejects_a_complete_not_applicable_cell_group() {
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let matrix = PerfMatrixSpec::complete();
        let profile = profile_key(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        let plan = matrix
            .applicability_plan(&registry, profile, PerfGate::Qg1)
            .expect("physical QG-1 plan");
        let classified = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .zip(&plan.cells)
            .collect::<Vec<_>>();
        let selected = classified
            .iter()
            .find(|(_, entry)| entry.applicability.is_runnable())
            .expect("runnable physical QG-1 cell");
        let not_applicable = classified
            .iter()
            .find(|(_, entry)| entry.applicability == PerfCellApplicability::NotApplicable)
            .expect("non-applicable physical QG-1 cell");
        let mut artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            applicability_plan: Some(plan.binding().clone()),
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            execution: Some(PerfExecutionProvenance {
                host_identity: "test-host".to_owned(),
                producer_os: PerfProducerOs::Linux,
                physical_cores: 64,
                logical_threads: 128,
                process_available_threads: 64,
                execution_capacity: 64,
                max_exercised_cell_width: 64,
                configured_engine_thread_widths: vec![selected.1.configured_threads],
                runtime_detected_isa: vec!["avx2".to_owned()],
                cpu_affinity_allowed_list: Some("0-63".to_owned()),
                affinity_or_cpuset_cap: Some(
                    "Cpus_allowed_list=0-63 (64 of 128 host logical threads)".to_owned(),
                ),
            }),
            git_rev: "0".repeat(40),
            run_window: "test-window".to_owned(),
            run_id: "qg1-not-applicable-hostile".to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: plan.binding.normalized_perf_manifest_sha256.clone(),
            cells: threshold_rows(PerfGate::Qg1, &[selected.0]),
            laws_attested: false,
        };
        artifact
            .verify_current_measured_contract()
            .expect("valid partial QG-1 threshold");
        artifact
            .cells
            .extend(threshold_rows(PerfGate::Qg1, &[not_applicable.0]));
        assert!(
            artifact.verify_current_measured_contract().is_err(),
            "a fully populated non-applicable cell group must fail strict reload"
        );
    }

    #[test]
    fn scheduler_capacity_is_distinct_from_host_process_availability() {
        let mut execution = PerfExecutionProvenance {
            host_identity: "m4-test-host".to_owned(),
            producer_os: PerfProducerOs::Macos,
            physical_cores: 14,
            logical_threads: 14,
            process_available_threads: 14,
            execution_capacity: 10,
            max_exercised_cell_width: 8,
            configured_engine_thread_widths: vec![1, 2, 4, 8],
            runtime_detected_isa: vec!["aes".to_owned(), "neon".to_owned(), "sha2".to_owned()],
            cpu_affinity_allowed_list: None,
            affinity_or_cpuset_cap: None,
        };
        assert!(
            execution.is_complete(),
            "a scheduler-managed ten-worker pool must remain representable on a fourteen-CPU M4 host"
        );
        assert!(execution.matches_capacity_semantics(ExecutionCapacitySemantics::SchedulerWorkers));
        assert!(!execution.matches_capacity_semantics(ExecutionCapacitySemantics::PhysicalCores));

        execution.process_available_threads = 9;
        assert!(
            !execution.is_complete(),
            "process availability below the registered scheduler capacity must fail closed"
        );
        assert!(
            !execution.matches_capacity_semantics(ExecutionCapacitySemantics::SchedulerWorkers)
        );
        execution.process_available_threads = 15;
        assert!(
            !execution.is_complete(),
            "process availability cannot exceed the attested host topology"
        );
    }

    #[test]
    fn cpu_topology_and_affinity_parsers_preserve_host_wide_width() {
        let cpuinfo = "\
processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 0

processor : 2
physical id : 0
core id : 1

processor : 3
physical id : 0
core id : 1
";
        assert_eq!(parse_linux_cpu_topology(cpuinfo), Some((2, 4)));
        assert_eq!(parse_cpu_list_count("0-127"), Some(128));
        assert_eq!(parse_cpu_list_count("0-15,32-47,63"), Some(33));
        assert_eq!(parse_cpu_list_count("4-2"), None);
    }

    #[test]
    fn linux_vmhwm_parser_requires_the_documented_unit() {
        assert_eq!(
            parse_linux_vmhwm_bytes("Name:\tbench\nVmHWM:\t   1234 kB\n"),
            Some(1_263_616)
        );
        assert_eq!(parse_linux_vmhwm_bytes("VmHWM: 12 MB"), None);
        assert_eq!(parse_linux_vmhwm_bytes("VmRSS: 12 kB"), None);
    }

    #[test]
    fn macos_time_parser_requires_the_exact_peak_rss_label() {
        let report = "\
        0.12 real 0.10 user 0.02 sys
        1212416  maximum resident set size
        9122317  instructions retired
        917696  peak memory footprint
";
        assert_eq!(parse_macos_time_max_rss_bytes(report), Some(1_212_416));
        assert_eq!(
            parse_macos_time_max_rss_bytes("917696 peak memory footprint"),
            None
        );
        assert_eq!(
            parse_macos_time_max_rss_bytes("1212416 maximum resident set size bytes"),
            None
        );
    }

    #[test]
    fn gate_parser_accepts_only_normative_labels() {
        assert_eq!("qg-1".parse::<PerfGate>().expect("QG-1"), PerfGate::Qg1);
        assert_eq!("QG_10".parse::<PerfGate>().expect("QG-10"), PerfGate::Qg10);
        assert!("QG-0".parse::<PerfGate>().is_err());
    }
}
