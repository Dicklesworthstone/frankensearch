//! Machine-readable performance-matrix contracts for Quill QG-1 through QG-10.
//!
//! The Criterion entry point owns engine execution. This module owns the
//! deterministic matrix, statistics, artifact schema, RSS probe, and human
//! rendering so the evidence format is unit-tested without running a benchmark.

use std::cell::RefCell;
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
use crate::local_perf_runner::{
    LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION, LOCAL_PERF_BOOKING_RECEIPT_SCHEMA_VERSION,
    LOCAL_PERF_LEASE_RELEASE_RECEIPT_SCHEMA_VERSION, PERF_RUN_PRECOMMIT_SCHEMA_VERSION,
};
use crate::machine_class_registry::{
    DefaultFlipDisposition, ExecutionCapacitySemantics, LOCAL_PERF_PRODUCER_CONTRACT_VERSION,
    MACHINE_CLASS_REGISTRY_SCHEMA_VERSION, MACHINE_CLASS_REGISTRY_SHA256, MachineClassError,
    MachineClassRegistry, MachineExecutionProfile, MachineProfileAvailability, MachineProfileKey,
    RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION, RUNNER_RECEIPT_SCHEMA_VERSION,
};
use crate::perf_assembly::PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION;
use crate::perf_evidence::PERF_EVIDENCE_SCHEMA_VERSION;
use crate::perf_ratchet::PERF_HISTORY_POINTER_SCHEMA_VERSION;

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
/// Schema for one QG-1 raw lifecycle receipt.  It is intentionally independent
/// of the outer artifact schema: raw rows are replayable evidence and must fail
/// closed when an older binding cannot name its receipt fields.
const QG1_LIFECYCLE_AUTHORITY_SCHEMA_VERSION: &str =
    "frankensearch.quill.qg1-lifecycle-authority.v3";
const QG1_LIFECYCLE_BINDING_SCHEMA_VERSION: &str = "frankensearch.quill.qg1-lifecycle-binding.v6";
const QG1_STREAM_ROLE_EFFECT: &str = "qg1.effect.tantivy_vs_quill.v1";
const QG1_STREAM_ROLE_TANTIVY_NULL: &str = "qg1.null.tantivy.v1";
const QG1_STREAM_ROLE_QUILL_NULL: &str = "qg1.null.quill.v1";
const QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT: &str = "qg1.pilot.tantivy_effect.v1";
const QG1_STREAM_ROLE_TANTIVY_PILOT_NULL: &str = "qg1.pilot.tantivy_null.v1";
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

/// Tantivy version whose writer construction semantics the QG-1 incumbent
/// screen is allowed to compare.
///
/// This is intentionally a protocol pin rather than a Cargo-version guess:
/// changing Tantivy can change shipping-auto worker selection and therefore
/// requires a new incumbent screen.
pub const QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION: &str = "0.26.1";
/// Wire schema for the provisional QG-1 Tantivy incumbent screen.
pub const QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION: &str =
    "quill-qg1-tantivy-incumbent-screen-v2";

/// Tantivy writer construction admitted to the QG-1 incumbent screen.
///
/// `ShippingAuto` means the unmodified Tantivy `Index::writer(heap)` path.
/// `Fixed` means Tantivy's public `writer_with_num_threads` path.  Both retain
/// the same heap, corpus, schema, analyzer, merge, visibility, terminal
/// searchability, and durability contract recorded beside the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg1TantivyWriterMode {
    /// Shipping Tantivy writer selection under the pinned writer heap.
    ShippingAuto,
    /// A preregistered explicit Tantivy writer width.
    Fixed {
        /// Number of internal Tantivy writer workers requested from Tantivy.
        writer_threads: usize,
    },
}

impl Qg1TantivyWriterMode {
    /// Stable, human-readable mode label used in evidence and tie breaks.
    #[must_use]
    pub fn stable_id(self) -> String {
        match self {
            Self::ShippingAuto => "shipping_auto".to_owned(),
            Self::Fixed { writer_threads } => format!("fixed_{writer_threads}"),
        }
    }
}

/// Exact non-writer semantics that must remain identical across a QG-1
/// Tantivy incumbent screen.
///
/// The runner derives these receipts from the live index setup.  Keeping them
/// explicit means a quicker arm cannot be selected after changing merge,
/// searchable-terminal, or durability behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivySemanticContract {
    /// Tantivy release that supplied the writer implementation.
    pub tantivy_version: String,
    /// Exact Tantivy schema receipt.
    pub schema_sha256: String,
    /// Exact analyzer receipt.
    pub analyzer_sha256: String,
    /// Exact indexed-field receipt.
    pub indexed_fields_sha256: String,
    /// Exact merge-policy receipt.
    pub merge_policy_sha256: String,
    /// Exact commit-cadence and visibility receipt.
    pub visibility_sha256: String,
    /// Exact terminal searchable-scope receipt.
    pub searchable_terminal_scope_sha256: String,
    /// Exact durability receipt.
    pub durability_sha256: String,
    /// Exact Quill backend/configuration receipt used in the same-invocation
    /// T/Quill and Quill/Quill streams.
    pub quill_config_sha256: String,
}

impl Qg1TantivySemanticContract {
    fn validate(&self) -> Result<(), Qg1TantivyIncumbentError> {
        if self.tantivy_version != QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION
            || [
                &self.schema_sha256,
                &self.analyzer_sha256,
                &self.indexed_fields_sha256,
                &self.merge_policy_sha256,
                &self.visibility_sha256,
                &self.searchable_terminal_scope_sha256,
                &self.durability_sha256,
                &self.quill_config_sha256,
            ]
            .into_iter()
            .any(|digest| !is_lower_hex_digest(digest))
        {
            return Err(Qg1TantivyIncumbentError::InvalidSemanticContract);
        }
        Ok(())
    }

    /// Domain-separated identity of all non-writer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when the semantic contract is not fully pinned.
    pub fn contract_sha256(&self) -> Result<String, Qg1TantivyIncumbentError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|_| Qg1TantivyIncumbentError::InvalidSemanticContract)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-tantivy-semantics.v1\0");
        update_length_framed(&mut hasher, encoded.as_slice());
        Ok(finish_sha256_hex(hasher))
    }
}

/// Immutable context for one independent machine-class/execution-mode screen.
///
/// The external CPU budget is deliberately independent from Tantivy's internal
/// writer widths.  It remains identical for every candidate and control, while
/// the preregistered fixed widths may be above or below that budget when the
/// pinned heap can support them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyIncumbentScreenPlan {
    /// Candidate-screen schema version.
    pub schema_version: String,
    /// Frozen hardware class and execution mode this screen belongs to.
    pub machine_profile: MachineProfileKey,
    /// Externally admitted CPU budget for the whole invocation.
    pub external_cpu_budget: usize,
    /// Fixed Tantivy widths preregistered before pilots execute.
    pub preregistered_writer_widths: Vec<usize>,
    /// Exact QG-1 bulk work denominator, derived from the canonical cell.
    pub work_units: u64,
    /// Exact prepared-content byte denominator emitted by the live producer.
    pub content_bytes: u64,
    /// Domain-separated receipt binding the canonical cell and both denominators.
    pub work_contract_sha256: String,
    /// Domain-separated identity of this immutable screen plan.
    pub plan_sha256: String,
}

impl Qg1TantivyIncumbentScreenPlan {
    /// Construct one machine-profile-specific screen plan.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid bulk cell, empty denominator, empty CPU
    /// budget, or non-canonical width list.
    pub fn new(
        machine_profile: MachineProfileKey,
        external_cpu_budget: usize,
        preregistered_writer_widths: Vec<usize>,
        cell: &PerfCellSpec,
        content_bytes: u64,
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        qg1_bulk_cell_resources(cell)?;
        let work_units = cell
            .document_count
            .filter(|count| *count > 0)
            .ok_or(Qg1TantivyIncumbentError::InvalidBulkCell)?;
        if content_bytes == 0 {
            return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
        }
        let mut plan = Self {
            schema_version: QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION.to_owned(),
            machine_profile,
            external_cpu_budget,
            preregistered_writer_widths,
            work_units,
            content_bytes,
            work_contract_sha256: qg1_bulk_work_contract_sha256(cell, work_units, content_bytes)?,
            plan_sha256: String::new(),
        };
        plan.validate_shape()?;
        plan.plan_sha256 = plan.recomputed_plan_sha256()?;
        Ok(plan)
    }

    fn validate_shape(&self) -> Result<(), Qg1TantivyIncumbentError> {
        if self.schema_version != QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION
            || self.external_cpu_budget == 0
            || self.work_units == 0
            || self.content_bytes == 0
            || !is_lower_hex_digest(&self.work_contract_sha256)
            || self.preregistered_writer_widths.is_empty()
            || self
                .preregistered_writer_widths
                .windows(2)
                .any(|widths| widths[0] >= widths[1])
            || self
                .preregistered_writer_widths
                .iter()
                .any(|width| *width == 0)
        {
            return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
        }
        Ok(())
    }

    fn recomputed_plan_sha256(&self) -> Result<String, Qg1TantivyIncumbentError> {
        self.validate_shape()?;
        let machine_profile = serde_json::to_vec(&self.machine_profile)
            .map_err(|_| Qg1TantivyIncumbentError::InvalidScreenPlan)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-tantivy-screen-plan.v1\0");
        update_length_framed(&mut hasher, self.schema_version.as_bytes());
        update_length_framed(&mut hasher, machine_profile.as_slice());
        update_length_framed(&mut hasher, self.external_cpu_budget.to_string().as_bytes());
        update_length_framed(&mut hasher, self.work_units.to_string().as_bytes());
        update_length_framed(&mut hasher, self.content_bytes.to_string().as_bytes());
        update_length_framed(&mut hasher, self.work_contract_sha256.as_bytes());
        for width in &self.preregistered_writer_widths {
            update_length_framed(&mut hasher, width.to_string().as_bytes());
        }
        Ok(finish_sha256_hex(hasher))
    }

    fn validate_for_cell(&self, cell: &PerfCellSpec) -> Result<(), Qg1TantivyIncumbentError> {
        self.validate_shape()?;
        if self.plan_sha256 != self.recomputed_plan_sha256()? {
            return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
        }
        let (_, writer_heap_bytes) = qg1_bulk_cell_resources(cell)?;
        let expected_work_units = cell
            .document_count
            .filter(|count| *count > 0)
            .ok_or(Qg1TantivyIncumbentError::InvalidBulkCell)?;
        if self.work_units != expected_work_units
            || self.work_contract_sha256
                != qg1_bulk_work_contract_sha256(cell, self.work_units, self.content_bytes)?
        {
            return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
        }
        if self.preregistered_writer_widths.iter().any(|width| {
            PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES.saturating_mul(*width) > writer_heap_bytes
        }) {
            return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
        }
        Ok(())
    }
}

/// One preregistered QG-1 Tantivy configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyIncumbentCandidate {
    /// Candidate-screen schema version.
    pub schema_version: String,
    /// Screen-plan identity, including machine class and execution mode.
    pub screen_plan_sha256: String,
    /// Canonical QG-1 cell contract shared by every candidate.
    pub cell_contract_sha256: String,
    /// Non-writer semantic contract shared by every candidate.
    pub semantic_contract: Qg1TantivySemanticContract,
    /// Writer heap shared by every candidate and the Quill comparison.
    pub writer_heap_bytes: usize,
    /// Only permitted writer configuration difference between candidates.
    pub writer_mode: Qg1TantivyWriterMode,
    /// Domain-separated digest of this exact candidate configuration.
    pub config_sha256: String,
}

impl Qg1TantivyIncumbentCandidate {
    fn new(
        cell: &PerfCellSpec,
        screen_plan: &Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
        writer_mode: Qg1TantivyWriterMode,
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        let (_, writer_heap_bytes) = qg1_bulk_cell_resources(cell)?;
        screen_plan.validate_for_cell(cell)?;
        if let Qg1TantivyWriterMode::Fixed { writer_threads } = writer_mode {
            if !screen_plan
                .preregistered_writer_widths
                .contains(&writer_threads)
            {
                return Err(Qg1TantivyIncumbentError::InfeasibleWriterMode {
                    writer_mode: writer_mode.stable_id(),
                });
            }
        }
        semantic_contract.validate()?;
        let mut candidate = Self {
            schema_version: QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION.to_owned(),
            screen_plan_sha256: screen_plan.plan_sha256.clone(),
            cell_contract_sha256: cell
                .contract_sha256()
                .map_err(|_| Qg1TantivyIncumbentError::InvalidBulkCell)?,
            semantic_contract: semantic_contract.clone(),
            writer_heap_bytes,
            writer_mode,
            config_sha256: String::new(),
        };
        candidate.config_sha256 = candidate.recomputed_config_sha256()?;
        Ok(candidate)
    }

    fn recomputed_config_sha256(&self) -> Result<String, Qg1TantivyIncumbentError> {
        self.semantic_contract.validate()?;
        if self.schema_version != QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION
            || !is_lower_hex_digest(&self.screen_plan_sha256)
            || !is_lower_hex_digest(&self.cell_contract_sha256)
            || self.writer_heap_bytes == 0
        {
            return Err(Qg1TantivyIncumbentError::InvalidCandidate);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-tantivy-incumbent-candidate.v1\0");
        update_length_framed(&mut hasher, self.schema_version.as_bytes());
        update_length_framed(&mut hasher, self.screen_plan_sha256.as_bytes());
        update_length_framed(&mut hasher, self.cell_contract_sha256.as_bytes());
        update_length_framed(
            &mut hasher,
            self.semantic_contract.contract_sha256()?.as_bytes(),
        );
        update_length_framed(&mut hasher, self.writer_heap_bytes.to_string().as_bytes());
        update_length_framed(&mut hasher, self.writer_mode.stable_id().as_bytes());
        Ok(finish_sha256_hex(hasher))
    }

    fn validate_against(
        &self,
        cell: &PerfCellSpec,
        screen_plan: &Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
    ) -> Result<(), Qg1TantivyIncumbentError> {
        let (_, writer_heap_bytes) = qg1_bulk_cell_resources(cell)?;
        screen_plan.validate_for_cell(cell)?;
        if self.screen_plan_sha256 != screen_plan.plan_sha256
            || self.cell_contract_sha256
                != cell
                    .contract_sha256()
                    .map_err(|_| Qg1TantivyIncumbentError::InvalidBulkCell)?
            || self.writer_heap_bytes != writer_heap_bytes
            || &self.semantic_contract != semantic_contract
            || self.config_sha256 != self.recomputed_config_sha256()?
        {
            return Err(Qg1TantivyIncumbentError::CandidateContractMismatch);
        }
        if let Qg1TantivyWriterMode::Fixed { writer_threads } = self.writer_mode {
            if !screen_plan
                .preregistered_writer_widths
                .contains(&writer_threads)
            {
                return Err(Qg1TantivyIncumbentError::InfeasibleWriterMode {
                    writer_mode: self.writer_mode.stable_id(),
                });
            }
        }
        Ok(())
    }
}

/// Immutable binding between one raw QG-1 observation and the engine,
/// denominator, and producer identity that emitted it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1RawObservationBinding {
    raw_sample_sha256: String,
    observation_id_sha256: String,
    engine_id: String,
    engine_config_sha256: String,
    work_units: u64,
    content_bytes: u64,
}

/// Pilot result for one preregistered Tantivy candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyIncumbentPilot {
    /// Candidate measured as the treatment arm.
    pub candidate: Qg1TantivyIncumbentCandidate,
    /// Receipt identity for this complete pilot stream; unique per invocation.
    pub stream_receipt_sha256: String,
    /// Live Tantivy worker count materialized by this candidate.
    pub observed_writer_threads: usize,
    /// Raw paired candidate/control and candidate A/A evidence.
    pub experiment: PairedExperimentResult,
    /// Sealed bindings for the candidate/control effect records.
    effect_observations: Vec<Qg1RawObservationBinding>,
    /// Sealed bindings for the candidate/candidate A/A records.
    null_observations: Vec<Qg1RawObservationBinding>,
}

impl Qg1TantivyIncumbentPilot {
    fn recomputed_stream_receipt_sha256(&self) -> Result<String, Qg1TantivyIncumbentError> {
        let experiment = serde_json::to_vec(&self.experiment)
            .map_err(|_| Qg1TantivyIncumbentError::StreamReceiptMismatch)?;
        let observations =
            serde_json::to_vec(&(&self.effect_observations, &self.null_observations))
                .map_err(|_| Qg1TantivyIncumbentError::StreamReceiptMismatch)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-tantivy-pilot-stream.v1\0");
        update_length_framed(&mut hasher, self.candidate.config_sha256.as_bytes());
        update_length_framed(
            &mut hasher,
            self.observed_writer_threads.to_string().as_bytes(),
        );
        update_length_framed(&mut hasher, experiment.as_slice());
        update_length_framed(&mut hasher, observations.as_slice());
        Ok(finish_sha256_hex(hasher))
    }
}

/// Identity of one configuration-bound raw QG-1 stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg1TantivyDecisionStreamKind {
    /// The Tantivy-versus-Quill comparison stream.
    TantivyVsQuill,
    /// The independent Tantivy-versus-Tantivy A/A stream.
    TantivyNull,
    /// The independent Quill-versus-Quill A/A stream.
    QuillNull,
}

impl Qg1TantivyDecisionStreamKind {
    const fn stable_id(self) -> &'static str {
        match self {
            Self::TantivyVsQuill => "tantivy_vs_quill",
            Self::TantivyNull => "tantivy_null",
            Self::QuillNull => "quill_null",
        }
    }
}

/// One configuration-bound raw QG-1 stream.
///
/// This intentionally stores only one paired stream.  The T/Q stream is used
/// as the effect in two recomputable decisions, once against the real T/T
/// null and once against the real Q/Q null.  It can therefore never masquerade
/// as its own A/A null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyBoundStream {
    /// Which independent stream this record represents.
    pub kind: Qg1TantivyDecisionStreamKind,
    /// Explicit engine identity bound to the control arm.
    pub control_engine_id: String,
    /// Exact configuration receipt bound to the control arm.
    pub control_engine_config_sha256: String,
    /// Explicit engine identity bound to the treatment arm.
    pub treatment_engine_id: String,
    /// Exact configuration receipt bound to the treatment arm.
    pub treatment_engine_config_sha256: String,
    /// Raw paired samples for exactly this stream.
    pub samples: Vec<PerfRawSample>,
    /// Sealed raw-observation bindings parallel to [`Self::samples`].
    observations: Vec<Qg1RawObservationBinding>,
    /// Domain-separated receipt over kind, configurations, run, and raw samples.
    pub stream_receipt_sha256: String,
}

/// Provisional result of a complete QG-1 Tantivy incumbent pilot screen.
///
/// This is deliberately not a freeze or a Quill claim.  It is scoped to one
/// machine class and execution mode, and only a subsequent fully-bound
/// T/Quill, T/T, Q/Q decision may consume its unique candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyIncumbentScreen {
    /// Candidate-screen schema version.
    pub schema_version: String,
    /// Machine-profile-specific plan used to preregister this screen.
    pub screen_plan: Qg1TantivyIncumbentScreenPlan,
    /// Canonical candidate universe, in preregistered order.
    pub candidates: Vec<Qg1TantivyIncumbentCandidate>,
    /// Pilot evidence, in the same order as `candidates`.
    pub pilots: Vec<Qg1TantivyIncumbentPilot>,
    /// Process-level run identity shared by every pilot.
    pub run_id: String,
    /// Candidates statistically tied for fastest at the predeclared 95% median CI.
    pub tied_fastest_candidates: Vec<Qg1TantivyIncumbentCandidate>,
    /// Provisional fastest configuration only when the confidence intervals
    /// distinguish it from every other candidate.
    pub selected_candidate: Option<Qg1TantivyIncumbentCandidate>,
    /// Stable explanation when a complete provisional selection is unavailable.
    pub no_decision_reason: Option<String>,
}

/// Same-invocation T/Quill, T/T, and Q/Q raw streams bound to one selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1TantivyIncumbentDecision {
    /// Shared predeclared estimator configuration used for both recomputations.
    pub estimator_config: PairedEstimatorConfig,
    /// T/Quill comparison with Tantivy as the control arm, never an A/A null.
    pub tantivy_vs_quill: Qg1TantivyBoundStream,
    /// Independent Tantivy/Tantivy A/A control.
    pub tantivy_null: Qg1TantivyBoundStream,
    /// Independent Quill/Quill A/A control.
    pub quill_null: Qg1TantivyBoundStream,
}

/// Typed failures for the QG-1 incumbent model.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Qg1TantivyIncumbentError {
    #[error("QG-1 incumbent candidates require a canonical bulk docs-per-second cell")]
    InvalidBulkCell,
    #[error("QG-1 incumbent semantics are incomplete or use a different Tantivy version")]
    InvalidSemanticContract,
    #[error("QG-1 incumbent screen plan is malformed or infeasible for the cell")]
    InvalidScreenPlan,
    #[error("QG-1 incumbent candidate is malformed")]
    InvalidCandidate,
    #[error(
        "QG-1 incumbent candidate differs from the canonical cell, screen plan, or semantic contract"
    )]
    CandidateContractMismatch,
    #[error("QG-1 incumbent writer mode {writer_mode} is not preregistered or feasible")]
    InfeasibleWriterMode { writer_mode: String },
    #[error("QG-1 incumbent candidate screen has no unique provisional selection")]
    NoProvisionalSelection,
    #[error("QG-1 incumbent screen fields do not recompute from its pilots")]
    ScreenSelectionMismatch,
    #[error(
        "QG-1 incumbent decision streams must use one process invocation and exact scope/provenance"
    )]
    DecisionInvocationMismatch,
    #[error("QG-1 incumbent decision lacks valid eligible throughput evidence")]
    DecisionEvidenceInvalid,
    #[error("QG-1 incumbent pilots and decision streams must use one estimator configuration")]
    EstimatorConfigMismatch,
    #[error(
        "QG-1 incumbent decision stream does not bind every raw arm to its expected configuration"
    )]
    DecisionCandidateMismatch,
    #[error(
        "QG-1 incumbent raw-observation binding is malformed or no longer matches its raw sample"
    )]
    ObservationBindingMismatch,
    #[error("QG-1 incumbent raw observation is reused across pilots or decision streams")]
    ObservationReuse,
    #[error("QG-1 incumbent stream receipt is malformed, stale, or reused")]
    StreamReceiptMismatch,
}

fn qg1_bulk_cell_resources(
    cell: &PerfCellSpec,
) -> Result<(usize, usize), Qg1TantivyIncumbentError> {
    let (
        Some(corpus),
        Some(document_count),
        Some(threads),
        Some(writer_heap_bytes),
        Some(positions),
    ) = (
        cell.corpus,
        cell.document_count,
        cell.threads,
        cell.writer_heap_bytes,
        cell.positions,
    )
    else {
        return Err(Qg1TantivyIncumbentError::InvalidBulkCell);
    };
    if cell.gate != PerfGate::Qg1
        || cell.metric != "docs_per_second"
        || cell.fixture != format!("bulk/{}/{threads}/{}", corpus.label(), positions.label())
        || document_count != corpus.document_count()
        || threads == 0
        || writer_heap_bytes != perf_writer_heap_bytes(threads)
    {
        return Err(Qg1TantivyIncumbentError::InvalidBulkCell);
    }
    Ok((threads, writer_heap_bytes))
}

fn qg1_bulk_work_contract_sha256(
    cell: &PerfCellSpec,
    work_units: u64,
    content_bytes: u64,
) -> Result<String, Qg1TantivyIncumbentError> {
    if work_units == 0 || content_bytes == 0 {
        return Err(Qg1TantivyIncumbentError::InvalidScreenPlan);
    }
    let cell_contract = cell
        .contract_sha256()
        .map_err(|_| Qg1TantivyIncumbentError::InvalidBulkCell)?;
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.quill.qg1-bulk-work-contract.v1\0");
    update_length_framed(&mut hasher, cell_contract.as_bytes());
    update_length_framed(&mut hasher, work_units.to_string().as_bytes());
    update_length_framed(&mut hasher, content_bytes.to_string().as_bytes());
    Ok(finish_sha256_hex(hasher))
}

/// Reconstruct the sole timing scope admitted by the QG-1 incumbent screen.
///
/// The public matrix scope supplies the cell-derived operation identity,
/// version, and unit.  QG-1 engine indexing is intentionally stricter than
/// the generic matrix default: its rate must be recomputable from one
/// continuous work/time interval, so this screen requires native throughput
/// semantics for that exact scope.
fn qg1_expected_throughput_scope(
    cell: &PerfCellSpec,
) -> Result<PerfOperationScope, Qg1TantivyIncumbentError> {
    qg1_bulk_cell_resources(cell)?;
    let mut scope = perf_operation_scope(cell.gate, &cell.fixture, &cell.metric);
    scope.semantics = PerfMetricSemantics::Throughput;
    Ok(scope)
}

const QG1_TANTIVY_ENGINE_ID: &str = "tantivy";
const QG1_QUILL_ENGINE_ID: &str = "quill";

fn qg1_valid_engine_identity(engine_id: &str, config_sha256: &str) -> bool {
    matches!(engine_id, QG1_TANTIVY_ENGINE_ID | QG1_QUILL_ENGINE_ID)
        && is_lower_hex_digest(config_sha256)
}

fn qg1_raw_sample_sha256(sample: &PerfRawSample) -> Result<String, Qg1TantivyIncumbentError> {
    let encoded = serde_json::to_vec(sample)
        .map_err(|_| Qg1TantivyIncumbentError::ObservationBindingMismatch)?;
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.quill.qg1-raw-observation.v1\0");
    update_length_framed(&mut hasher, encoded.as_slice());
    Ok(finish_sha256_hex(hasher))
}

fn qg1_bind_raw_observations(
    samples: &[PerfRawSample],
    control_engine_id: &str,
    control_engine_config_sha256: &str,
    treatment_engine_id: &str,
    treatment_engine_config_sha256: &str,
    observation_ids: Vec<String>,
) -> Result<Vec<Qg1RawObservationBinding>, Qg1TantivyIncumbentError> {
    if samples.len() != observation_ids.len()
        || !qg1_valid_engine_identity(control_engine_id, control_engine_config_sha256)
        || !qg1_valid_engine_identity(treatment_engine_id, treatment_engine_config_sha256)
    {
        return Err(Qg1TantivyIncumbentError::ObservationBindingMismatch);
    }
    samples
        .iter()
        .zip(observation_ids)
        .map(|(sample, observation_id_sha256)| {
            let (engine_id, engine_config_sha256) = match sample.arm {
                PerfSampleArm::Control => (control_engine_id, control_engine_config_sha256),
                PerfSampleArm::Treatment => (treatment_engine_id, treatment_engine_config_sha256),
            };
            let (Some(work_units), Some(content_bytes)) = (sample.work_units, sample.byte_count)
            else {
                return Err(Qg1TantivyIncumbentError::ObservationBindingMismatch);
            };
            if !is_lower_hex_digest(&observation_id_sha256)
                || sample
                    .qg1_sample_binding
                    .as_ref()
                    .is_none_or(|binding| binding.engine_id() != engine_id)
            {
                return Err(Qg1TantivyIncumbentError::ObservationBindingMismatch);
            }
            Ok(Qg1RawObservationBinding {
                raw_sample_sha256: qg1_raw_sample_sha256(sample)?,
                observation_id_sha256,
                engine_id: engine_id.to_owned(),
                engine_config_sha256: engine_config_sha256.to_owned(),
                work_units,
                content_bytes,
            })
        })
        .collect()
}

fn qg1_validate_raw_observations(
    samples: &[PerfRawSample],
    observations: &[Qg1RawObservationBinding],
    control_engine_id: &str,
    control_engine_config_sha256: &str,
    treatment_engine_id: &str,
    treatment_engine_config_sha256: &str,
    expected_work_units: u64,
    expected_content_bytes: u64,
) -> Result<(), Qg1TantivyIncumbentError> {
    if samples.len() != observations.len()
        || !qg1_valid_engine_identity(control_engine_id, control_engine_config_sha256)
        || !qg1_valid_engine_identity(treatment_engine_id, treatment_engine_config_sha256)
    {
        return Err(Qg1TantivyIncumbentError::ObservationBindingMismatch);
    }
    for (sample, observation) in samples.iter().zip(observations) {
        let (engine_id, engine_config_sha256) = match sample.arm {
            PerfSampleArm::Control => (control_engine_id, control_engine_config_sha256),
            PerfSampleArm::Treatment => (treatment_engine_id, treatment_engine_config_sha256),
        };
        if observation.raw_sample_sha256 != qg1_raw_sample_sha256(sample)?
            || !is_lower_hex_digest(&observation.observation_id_sha256)
            || observation.engine_id != engine_id
            || observation.engine_config_sha256 != engine_config_sha256
            || observation.work_units != expected_work_units
            || observation.content_bytes != expected_content_bytes
            || sample.work_units != Some(expected_work_units)
            || sample.byte_count != Some(expected_content_bytes)
            || sample
                .qg1_sample_binding
                .as_ref()
                .is_none_or(|binding| binding.engine_id() != engine_id)
        {
            return Err(Qg1TantivyIncumbentError::ObservationBindingMismatch);
        }
    }
    Ok(())
}

fn qg1_insert_observation_ids<'a>(
    observations: impl IntoIterator<Item = &'a Qg1RawObservationBinding>,
    seen: &mut BTreeSet<String>,
) -> Result<(), Qg1TantivyIncumbentError> {
    for observation in observations {
        if !seen.insert(observation.observation_id_sha256.clone()) {
            return Err(Qg1TantivyIncumbentError::ObservationReuse);
        }
    }
    Ok(())
}

impl Qg1TantivyIncumbentPilot {
    /// Seal one live QG-1 Tantivy pilot without exposing receipt hashing to the
    /// producer. The caller supplies immutable observation IDs emitted by the
    /// runner; engine/config and raw-content bindings are derived here.
    pub fn from_experiment(
        candidate: Qg1TantivyIncumbentCandidate,
        observed_writer_threads: usize,
        shipping_auto_config_sha256: String,
        experiment: PairedExperimentResult,
        effect_observation_ids: Vec<String>,
        null_observation_ids: Vec<String>,
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        let effect_observations = qg1_bind_raw_observations(
            &experiment.effect_samples,
            QG1_TANTIVY_ENGINE_ID,
            &shipping_auto_config_sha256,
            QG1_TANTIVY_ENGINE_ID,
            &candidate.config_sha256,
            effect_observation_ids,
        )?;
        let null_observations = qg1_bind_raw_observations(
            &experiment.null_samples,
            QG1_TANTIVY_ENGINE_ID,
            &candidate.config_sha256,
            QG1_TANTIVY_ENGINE_ID,
            &candidate.config_sha256,
            null_observation_ids,
        )?;
        let mut pilot = Self {
            candidate,
            stream_receipt_sha256: String::new(),
            observed_writer_threads,
            experiment,
            effect_observations,
            null_observations,
        };
        pilot.stream_receipt_sha256 = pilot.recomputed_stream_receipt_sha256()?;
        Ok(pilot)
    }
}

impl Qg1TantivyBoundStream {
    /// Seal one raw decision stream with explicit engine/config identity and
    /// immutable producer observation IDs. This is the live-producer boundary;
    /// callers cannot supply an arbitrary stream receipt.
    pub fn from_raw_samples(
        kind: Qg1TantivyDecisionStreamKind,
        control_engine_id: String,
        control_engine_config_sha256: String,
        treatment_engine_id: String,
        treatment_engine_config_sha256: String,
        samples: Vec<PerfRawSample>,
        observation_ids: Vec<String>,
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        let observations = qg1_bind_raw_observations(
            &samples,
            &control_engine_id,
            &control_engine_config_sha256,
            &treatment_engine_id,
            &treatment_engine_config_sha256,
            observation_ids,
        )?;
        let mut stream = Self {
            kind,
            control_engine_id,
            control_engine_config_sha256,
            treatment_engine_id,
            treatment_engine_config_sha256,
            samples,
            observations,
            stream_receipt_sha256: String::new(),
        };
        stream.stream_receipt_sha256 = stream.recomputed_stream_receipt_sha256()?;
        Ok(stream)
    }
}

/// Build the complete, machine-profile-qualified Tantivy candidate universe for
/// one QG-1 bulk cell.
///
/// The supplied fixed widths are preregistered by the screen plan, not derived
/// from the external CPU budget or the cell's configured width.
///
/// # Errors
///
/// Returns an error when the cell, plan, or shared semantic contract is not exact.
pub fn preregister_qg1_tantivy_incumbents(
    cell: &PerfCellSpec,
    screen_plan: &Qg1TantivyIncumbentScreenPlan,
    semantic_contract: &Qg1TantivySemanticContract,
) -> Result<Vec<Qg1TantivyIncumbentCandidate>, Qg1TantivyIncumbentError> {
    screen_plan.validate_for_cell(cell)?;
    let mut candidates = Vec::with_capacity(screen_plan.preregistered_writer_widths.len() + 1);
    candidates.push(Qg1TantivyIncumbentCandidate::new(
        cell,
        screen_plan,
        semantic_contract,
        Qg1TantivyWriterMode::ShippingAuto,
    )?);
    for writer_threads in &screen_plan.preregistered_writer_widths {
        candidates.push(Qg1TantivyIncumbentCandidate::new(
            cell,
            screen_plan,
            semantic_contract,
            Qg1TantivyWriterMode::Fixed {
                writer_threads: *writer_threads,
            },
        )?);
    }
    Ok(candidates)
}

fn qg1_valid_throughput_experiment(
    experiment: &PairedExperimentResult,
    external_qg1_authority: Option<&Qg1ExpectedAuthority>,
    expected_scope: &PerfOperationScope,
    expected_provenance: Option<&PerfSampleProvenance>,
    expected_work_units: u64,
    expected_content_bytes: u64,
) -> bool {
    experiment.recomputes_against_qg1_authority(external_qg1_authority)
        && experiment.status == PairedEvidenceStatus::Valid
        && experiment.claim_state == PairedClaimState::EligibleForDecision
        && experiment.scope == *expected_scope
        && expected_provenance.is_none_or(|provenance| experiment.provenance == *provenance)
        && experiment
            .effect_samples
            .iter()
            .chain(&experiment.null_samples)
            .all(|sample| {
                sample.work_units == Some(expected_work_units)
                    && sample.byte_count == Some(expected_content_bytes)
            })
}

fn qg1_validate_pilot_observations(
    pilot: &Qg1TantivyIncumbentPilot,
    shipping_auto_config_sha256: &str,
    expected_work_units: u64,
    expected_content_bytes: u64,
) -> Result<(), Qg1TantivyIncumbentError> {
    qg1_validate_raw_observations(
        &pilot.experiment.effect_samples,
        &pilot.effect_observations,
        QG1_TANTIVY_ENGINE_ID,
        shipping_auto_config_sha256,
        QG1_TANTIVY_ENGINE_ID,
        &pilot.candidate.config_sha256,
        expected_work_units,
        expected_content_bytes,
    )?;
    qg1_validate_raw_observations(
        &pilot.experiment.null_samples,
        &pilot.null_observations,
        QG1_TANTIVY_ENGINE_ID,
        &pilot.candidate.config_sha256,
        QG1_TANTIVY_ENGINE_ID,
        &pilot.candidate.config_sha256,
        expected_work_units,
        expected_content_bytes,
    )
}

impl Qg1TantivyIncumbentScreen {
    /// Validate a complete pilot set and report the fastest configuration or an
    /// explicit confidence-interval tie/no-decision result.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed cells, plans, or non-equivalent candidates.
    pub fn screen(
        cell: &PerfCellSpec,
        screen_plan: Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
        pilots: Vec<Qg1TantivyIncumbentPilot>,
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        Self::screen_against_qg1_authorities(cell, screen_plan, semantic_contract, pilots, &[])
    }

    /// Screen pilots that no longer carry their live producer, using the
    /// expectations their consumer retained.
    ///
    /// Reloaded pilots lose the never-serialized producer expectation, so this
    /// is the only entry that can admit persisted QG-1 screen evidence. Each
    /// pilot selects the retained expectation that issued its own sealed
    /// authority; live callers keep passing an empty set and remain bound by
    /// the expectation the producer installed in their configuration.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::screen`].
    pub fn screen_against_qg1_authorities(
        cell: &PerfCellSpec,
        screen_plan: Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
        pilots: Vec<Qg1TantivyIncumbentPilot>,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Self, Qg1TantivyIncumbentError> {
        let candidates = preregister_qg1_tantivy_incumbents(cell, &screen_plan, semantic_contract)?;
        let expected_scope = qg1_expected_throughput_scope(cell)?;
        let no_decision = |reason: impl Into<String>| Self {
            schema_version: QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION.to_owned(),
            screen_plan: screen_plan.clone(),
            candidates: candidates.clone(),
            pilots: pilots.clone(),
            run_id: String::new(),
            tied_fastest_candidates: Vec::new(),
            selected_candidate: None,
            no_decision_reason: Some(reason.into()),
        };
        if pilots.len() != candidates.len() {
            return Ok(no_decision(format!(
                "expected {} preregistered pilots, observed {}",
                candidates.len(),
                pilots.len()
            )));
        }
        let shipping_auto = &candidates[0];
        let mut seen_stream_receipts = BTreeSet::new();
        let mut seen_observation_ids = BTreeSet::new();
        let mut run_id = None;
        let mut scope = None;
        let mut provenance = None;
        let mut estimator_config = None;
        for (expected, pilot) in candidates.iter().zip(&pilots) {
            pilot
                .candidate
                .validate_against(cell, &screen_plan, semantic_contract)?;
            if &pilot.candidate != expected
                || pilot.stream_receipt_sha256 != pilot.recomputed_stream_receipt_sha256()?
                || !seen_stream_receipts.insert(pilot.stream_receipt_sha256.clone())
            {
                return Ok(no_decision(
                    "pilot candidates or stream receipts do not match the preregistered screen",
                ));
            }
            let (_, writer_heap_bytes) = qg1_bulk_cell_resources(cell)?;
            let width_matches = match pilot.candidate.writer_mode {
                Qg1TantivyWriterMode::ShippingAuto => pilot.observed_writer_threads > 0,
                Qg1TantivyWriterMode::Fixed { writer_threads } => {
                    pilot.observed_writer_threads == writer_threads
                }
            };
            if !width_matches
                || PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES
                    .saturating_mul(pilot.observed_writer_threads)
                    > writer_heap_bytes
            {
                return Ok(no_decision(
                    "candidate materialized an infeasible writer width",
                ));
            }
            if !qg1_valid_throughput_experiment(
                &pilot.experiment,
                select_qg1_expected_authority(external_qg1_authorities, &pilot.experiment.config),
                &expected_scope,
                None,
                screen_plan.work_units,
                screen_plan.content_bytes,
            ) {
                return Ok(no_decision(
                    "candidate pilot lacks valid configuration-bound throughput evidence",
                ));
            }
            qg1_validate_pilot_observations(
                pilot,
                &shipping_auto.config_sha256,
                screen_plan.work_units,
                screen_plan.content_bytes,
            )?;
            qg1_insert_observation_ids(
                pilot
                    .effect_observations
                    .iter()
                    .chain(&pilot.null_observations),
                &mut seen_observation_ids,
            )?;
            match &estimator_config {
                Some(expected_config) if expected_config != &pilot.experiment.config => {
                    return Err(Qg1TantivyIncumbentError::EstimatorConfigMismatch);
                }
                Some(_) => {}
                None => estimator_config = Some(pilot.experiment.config.clone()),
            }
            match (&run_id, &scope, &provenance) {
                (Some(expected_run_id), Some(expected_scope), Some(expected_provenance))
                    if expected_run_id != &pilot.experiment.provenance.run_id
                        || expected_scope != &pilot.experiment.scope
                        || expected_provenance != &pilot.experiment.provenance =>
                {
                    return Ok(no_decision(
                        "candidate pilots used different process invocations or semantic identities",
                    ));
                }
                (None, None, None) => {
                    run_id = Some(pilot.experiment.provenance.run_id.clone());
                    scope = Some(pilot.experiment.scope.clone());
                    provenance = Some(pilot.experiment.provenance.clone());
                }
                _ => return Ok(no_decision("candidate pilot identity state is incomplete")),
            }
        }
        let fastest = pilots
            .iter()
            .max_by(|left, right| {
                left.experiment
                    .effect
                    .treatment
                    .p50
                    .total_cmp(&right.experiment.effect.treatment.p50)
            })
            .expect("complete preregistered candidate set");
        let tied_fastest_candidates = pilots
            .iter()
            .filter(|pilot| {
                pilot.experiment.effect.treatment.median_ci95_high
                    >= fastest.experiment.effect.treatment.median_ci95_low
            })
            .map(|pilot| pilot.candidate.clone())
            .collect::<Vec<_>>();
        let selected_candidate =
            (tied_fastest_candidates.len() == 1).then(|| tied_fastest_candidates[0].clone());
        let no_decision_reason = selected_candidate.is_none().then_some(
            "fastest candidate is tied within the predeclared 95% median confidence intervals"
                .to_owned(),
        );
        Ok(Self {
            schema_version: QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION.to_owned(),
            screen_plan,
            candidates,
            pilots,
            run_id: run_id.expect("complete candidate set has a run ID"),
            tied_fastest_candidates,
            selected_candidate,
            no_decision_reason,
        })
    }

    /// Validate the same-invocation T/Quill, T/T, and Q/Q streams against this
    /// provisional selection. Passing this method remains evidence only; it
    /// does not freeze a gate or claim that Quill won.
    pub fn validate_decision(
        &self,
        cell: &PerfCellSpec,
        semantic_contract: &Qg1TantivySemanticContract,
        decision: &Qg1TantivyIncumbentDecision,
    ) -> Result<(), Qg1TantivyIncumbentError> {
        self.validate_decision_against_qg1_authorities(cell, semantic_contract, decision, &[])
    }

    /// Validate a persisted screen and decision against the QG-1 expectations
    /// their consumer retained outside the artifact.
    ///
    /// Pilot streams and decision streams are issued by separate producers, so
    /// the retained set is matched per stream configuration rather than
    /// assumed to be one authority.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::validate_decision`].
    pub fn validate_decision_against_qg1_authorities(
        &self,
        cell: &PerfCellSpec,
        semantic_contract: &Qg1TantivySemanticContract,
        decision: &Qg1TantivyIncumbentDecision,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), Qg1TantivyIncumbentError> {
        let recomputed_screen = Self::screen_against_qg1_authorities(
            cell,
            self.screen_plan.clone(),
            semantic_contract,
            self.pilots.clone(),
            external_qg1_authorities,
        )?;
        if &recomputed_screen != self {
            return Err(Qg1TantivyIncumbentError::ScreenSelectionMismatch);
        }
        let selected_candidate = recomputed_screen
            .selected_candidate
            .as_ref()
            .ok_or(Qg1TantivyIncumbentError::NoProvisionalSelection)?;
        let expected_pilot = recomputed_screen
            .pilots
            .first()
            .ok_or(Qg1TantivyIncumbentError::ScreenSelectionMismatch)?;
        let expected_scope = &expected_pilot.experiment.scope;
        let expected_provenance = &expected_pilot.experiment.provenance;
        let expected_estimator_config = &expected_pilot.experiment.config;
        let pilot_receipts = self
            .pilots
            .iter()
            .map(|pilot| pilot.stream_receipt_sha256.as_str())
            .collect::<BTreeSet<_>>();
        if [
            &decision.tantivy_vs_quill.stream_receipt_sha256,
            &decision.tantivy_null.stream_receipt_sha256,
            &decision.quill_null.stream_receipt_sha256,
        ]
        .into_iter()
        .any(|receipt| {
            pilot_receipts
                .iter()
                .any(|pilot_receipt| *pilot_receipt == receipt.as_str())
        }) {
            return Err(Qg1TantivyIncumbentError::StreamReceiptMismatch);
        }
        let mut seen_observation_ids = BTreeSet::new();
        for pilot in &recomputed_screen.pilots {
            qg1_insert_observation_ids(
                pilot
                    .effect_observations
                    .iter()
                    .chain(&pilot.null_observations),
                &mut seen_observation_ids,
            )?;
        }
        decision.recompute_against(
            selected_candidate,
            semantic_contract,
            expected_scope,
            expected_provenance,
            expected_estimator_config,
            select_qg1_expected_authority(external_qg1_authorities, &decision.estimator_config),
            recomputed_screen.screen_plan.work_units,
            recomputed_screen.screen_plan.content_bytes,
            &mut seen_observation_ids,
        )?;
        Ok(())
    }
}

impl Qg1TantivyBoundStream {
    fn recomputed_stream_receipt_sha256(&self) -> Result<String, Qg1TantivyIncumbentError> {
        let samples = serde_json::to_vec(&self.samples)
            .map_err(|_| Qg1TantivyIncumbentError::StreamReceiptMismatch)?;
        let observations = serde_json::to_vec(&self.observations)
            .map_err(|_| Qg1TantivyIncumbentError::StreamReceiptMismatch)?;
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-tantivy-decision-stream.v1\0");
        update_length_framed(&mut hasher, self.kind.stable_id().as_bytes());
        update_length_framed(&mut hasher, self.control_engine_id.as_bytes());
        update_length_framed(&mut hasher, self.control_engine_config_sha256.as_bytes());
        update_length_framed(&mut hasher, self.treatment_engine_id.as_bytes());
        update_length_framed(&mut hasher, self.treatment_engine_config_sha256.as_bytes());
        update_length_framed(&mut hasher, samples.as_slice());
        update_length_framed(&mut hasher, observations.as_slice());
        Ok(finish_sha256_hex(hasher))
    }

    fn validate_against(
        &self,
        kind: Qg1TantivyDecisionStreamKind,
        control_engine_id: &str,
        control_engine_config_sha256: &str,
        treatment_engine_id: &str,
        treatment_engine_config_sha256: &str,
        estimator_config: &PairedEstimatorConfig,
        expected_scope: &PerfOperationScope,
        expected_provenance: &PerfSampleProvenance,
        expected_work_units: u64,
        expected_content_bytes: u64,
    ) -> Result<(), Qg1TantivyIncumbentError> {
        if self.kind != kind
            || self.control_engine_id != control_engine_id
            || self.control_engine_config_sha256 != control_engine_config_sha256
            || self.treatment_engine_id != treatment_engine_id
            || self.treatment_engine_config_sha256 != treatment_engine_config_sha256
        {
            return Err(Qg1TantivyIncumbentError::DecisionCandidateMismatch);
        }
        if self.samples.iter().any(|sample| {
            sample.scope != *expected_scope || sample.provenance != *expected_provenance
        }) {
            return Err(Qg1TantivyIncumbentError::DecisionInvocationMismatch);
        }
        if self.stream_receipt_sha256 != self.recomputed_stream_receipt_sha256()? {
            return Err(Qg1TantivyIncumbentError::StreamReceiptMismatch);
        }
        qg1_validate_raw_observations(
            &self.samples,
            &self.observations,
            control_engine_id,
            control_engine_config_sha256,
            treatment_engine_id,
            treatment_engine_config_sha256,
            expected_work_units,
            expected_content_bytes,
        )?;
        validate_paired_stream(&self.samples, estimator_config)
            .map_err(|_| Qg1TantivyIncumbentError::DecisionEvidenceInvalid)?;
        Ok(())
    }
}

impl Qg1TantivyIncumbentDecision {
    fn recompute_against(
        &self,
        selected_candidate: &Qg1TantivyIncumbentCandidate,
        semantic_contract: &Qg1TantivySemanticContract,
        expected_scope: &PerfOperationScope,
        expected_provenance: &PerfSampleProvenance,
        expected_estimator_config: &PairedEstimatorConfig,
        external_qg1_authority: Option<&Qg1ExpectedAuthority>,
        expected_work_units: u64,
        expected_content_bytes: u64,
        seen_observation_ids: &mut BTreeSet<String>,
    ) -> Result<(PairedExperimentResult, PairedExperimentResult), Qg1TantivyIncumbentError> {
        if &self.estimator_config != expected_estimator_config {
            return Err(Qg1TantivyIncumbentError::EstimatorConfigMismatch);
        }
        let expected_qg1_authority =
            external_qg1_authority.or(self.estimator_config.qg1_expected_authority.as_ref());
        let selected_config = selected_candidate.config_sha256.as_str();
        let streams = [
            (
                &self.tantivy_vs_quill,
                Qg1TantivyDecisionStreamKind::TantivyVsQuill,
                QG1_TANTIVY_ENGINE_ID,
                selected_config,
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.as_str(),
            ),
            (
                &self.tantivy_null,
                Qg1TantivyDecisionStreamKind::TantivyNull,
                QG1_TANTIVY_ENGINE_ID,
                selected_config,
                QG1_TANTIVY_ENGINE_ID,
                selected_config,
            ),
            (
                &self.quill_null,
                Qg1TantivyDecisionStreamKind::QuillNull,
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.as_str(),
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.as_str(),
            ),
        ];
        let mut receipts = BTreeSet::new();
        for (
            stream,
            kind,
            control_engine_id,
            control_engine_config_sha256,
            treatment_engine_id,
            treatment_engine_config_sha256,
        ) in streams
        {
            stream.validate_against(
                kind,
                control_engine_id,
                control_engine_config_sha256,
                treatment_engine_id,
                treatment_engine_config_sha256,
                &self.estimator_config,
                expected_scope,
                expected_provenance,
                expected_work_units,
                expected_content_bytes,
            )?;
            if !receipts.insert(stream.stream_receipt_sha256.clone()) {
                return Err(Qg1TantivyIncumbentError::StreamReceiptMismatch);
            }
            qg1_insert_observation_ids(stream.observations.iter(), seen_observation_ids)?;
        }
        // Both decisions are QG-1 by construction, so they are estimated
        // through the authority-bearing entry: the generic estimator refuses
        // canonical QG-1 scopes precisely so a resealed row cannot reach here.
        let tantivy_decision = estimate_paired_experiment_against_qg1_authority(
            &self.tantivy_vs_quill.samples,
            &self.tantivy_null.samples,
            &self.estimator_config,
            expected_qg1_authority,
        )
        .map_err(|_| Qg1TantivyIncumbentError::DecisionEvidenceInvalid)?;
        let quill_decision = estimate_paired_experiment_against_qg1_authority(
            &self.tantivy_vs_quill.samples,
            &self.quill_null.samples,
            &self.estimator_config,
            expected_qg1_authority,
        )
        .map_err(|_| Qg1TantivyIncumbentError::DecisionEvidenceInvalid)?;
        if !qg1_valid_throughput_experiment(
            &tantivy_decision,
            external_qg1_authority,
            expected_scope,
            Some(expected_provenance),
            expected_work_units,
            expected_content_bytes,
        ) || !qg1_valid_throughput_experiment(
            &quill_decision,
            external_qg1_authority,
            expected_scope,
            Some(expected_provenance),
            expected_work_units,
            expected_content_bytes,
        ) {
            return Err(Qg1TantivyIncumbentError::DecisionEvidenceInvalid);
        }
        Ok((tantivy_decision, quill_decision))
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

fn validate_perf_manifest_gate_set(
    parsed: &toml::Value,
    requested_gate: PerfGate,
) -> Result<(), PerfApplicabilityPlanError> {
    let gates = parsed
        .get("gate")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
            gate: requested_gate,
            detail: "manifest does not define a [gate] table".to_owned(),
        })?;

    for gate in PerfGate::ALL {
        let label = gate.label();
        let policy = gates
            .get(label)
            .and_then(toml::Value::as_table)
            .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest gate.{label} is missing or not a table"),
            })?;
        for field in ["name", "fixture", "target"] {
            if policy
                .get(field)
                .and_then(toml::Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(PerfApplicabilityPlanError::ManifestContract {
                    gate: requested_gate,
                    detail: format!("manifest gate.{label}.{field} is missing or empty"),
                });
            }
        }
        if policy
            .get("activated")
            .and_then(toml::Value::as_bool)
            .is_none()
        {
            return Err(PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest gate.{label}.activated is missing or not boolean"),
            });
        }
        if gate == PerfGate::Qg6 {
            let query_groups = policy
                .get("queries_per_class")
                .and_then(toml::Value::as_integer)
                .and_then(|count| usize::try_from(count).ok());
            if query_groups != Some(QG6_QUERY_GROUPS) {
                return Err(PerfApplicabilityPlanError::ManifestContract {
                    gate: requested_gate,
                    detail: format!(
                        "manifest gate.{label}.queries_per_class must equal the frozen QG-6 group count {QG6_QUERY_GROUPS}"
                    ),
                });
            }
        }
        let allowed_fields: &[&str] = match gate {
            PerfGate::Qg1 => [
                "name",
                "fixture",
                "target",
                "primary_target_cell_width",
                "activated",
            ]
            .as_slice(),
            PerfGate::Qg2 => ["name", "fixture", "target", "activated", "qg2_contract"].as_slice(),
            PerfGate::Qg6 => [
                "name",
                "fixture",
                "queries_per_class",
                "target",
                "activated",
            ]
            .as_slice(),
            _ => ["name", "fixture", "target", "activated"].as_slice(),
        };
        for field in policy.keys() {
            if !allowed_fields.contains(&field.as_str()) {
                return Err(PerfApplicabilityPlanError::ManifestContract {
                    gate: requested_gate,
                    detail: format!("manifest gate.{label} defines unexpected field {field}"),
                });
            }
        }
        if gate == PerfGate::Qg2 {
            validate_qg2_comparator_contract_table(policy, requested_gate)?;
        }
    }

    let expected_labels = PerfGate::ALL
        .iter()
        .map(|gate| gate.label())
        .collect::<BTreeSet<_>>();
    for label in gates.keys() {
        if !expected_labels.contains(label.as_str()) {
            return Err(PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest defines unexpected gate.{label}"),
            });
        }
    }
    Ok(())
}

/// Admit the one nested table the gate manifest may carry, and admit it only
/// as the exact canonical QG-2 comparator contract.
///
/// Absence is the protected bootstrap state and stays admissible, so this
/// never forces the contract to exist. Presence is admitted only when the
/// table deserializes to the closed typed contract *and* equals the canonical
/// value field for field: an unknown key, a missing key, a reordered exclusion
/// list, or one altered enum spelling is a manifest-contract error. Widening
/// the allowlist without this check would turn `qg2_contract` into an
/// unvalidated free-text hole in an otherwise closed manifest.
fn validate_qg2_comparator_contract_table(
    policy: &toml::Table,
    requested_gate: PerfGate,
) -> Result<(), PerfApplicabilityPlanError> {
    let Some(value) = policy.get("qg2_contract") else {
        return Ok(());
    };
    let observed = value
        .clone()
        .try_into::<crate::qg2_contract::Qg2ComparatorContract>()
        .map_err(|error| PerfApplicabilityPlanError::ManifestContract {
            gate: requested_gate,
            detail: format!(
                "manifest gate.QG-2.qg2_contract is not the closed typed comparator contract: {error}"
            ),
        })?;
    if observed == crate::qg2_contract::Qg2ComparatorContract::canonical() {
        Ok(())
    } else {
        Err(PerfApplicabilityPlanError::ManifestContract {
            gate: requested_gate,
            detail: "manifest gate.QG-2.qg2_contract is not the canonical Q2C comparator contract"
                .to_owned(),
        })
    }
}

fn validate_perf_manifest_schema_bindings(
    parsed: &toml::Value,
    requested_gate: PerfGate,
) -> Result<(), PerfApplicabilityPlanError> {
    let schemas = parsed
        .get("schemas")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
            gate: requested_gate,
            detail: "manifest does not define a [schemas] table".to_owned(),
        })?;
    for (field, expected) in [
        ("threshold_artifact", PERF_ARTIFACT_SCHEMA_VERSION),
        ("evidence_artifact", PERF_EVIDENCE_SCHEMA_VERSION),
        ("evidence_assembly", PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION),
        ("history_pointer", PERF_HISTORY_POINTER_SCHEMA_VERSION),
        ("machine_registry", MACHINE_CLASS_REGISTRY_SCHEMA_VERSION),
        ("applicability_plan", PERF_APPLICABILITY_PLAN_SCHEMA_VERSION),
        ("runner_completion_receipt", RUNNER_RECEIPT_SCHEMA_VERSION),
        (
            "runner_artifact_manifest",
            RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        ),
        (
            "local_producer_contract",
            LOCAL_PERF_PRODUCER_CONTRACT_VERSION,
        ),
        (
            "runner_attempt_receipt",
            LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION,
        ),
        (
            "runner_lease_release_receipt",
            LOCAL_PERF_LEASE_RELEASE_RECEIPT_SCHEMA_VERSION,
        ),
        (
            "runner_booking_receipt",
            LOCAL_PERF_BOOKING_RECEIPT_SCHEMA_VERSION,
        ),
        ("precommit_inventory", PERF_RUN_PRECOMMIT_SCHEMA_VERSION),
    ] {
        let found = schemas
            .get(field)
            .and_then(toml::Value::as_str)
            .ok_or_else(|| PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest schemas.{field} is missing or not a string"),
            })?;
        if found != expected {
            return Err(PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest schemas.{field} is {found:?}, expected {expected:?}"),
            });
        }
    }
    for field in schemas.keys() {
        if ![
            "threshold_artifact",
            "evidence_artifact",
            "evidence_assembly",
            "machine_registry",
            "applicability_plan",
            "runner_completion_receipt",
            "runner_artifact_manifest",
            "local_producer_contract",
            "history_pointer",
            "runner_attempt_receipt",
            "runner_lease_release_receipt",
            "runner_booking_receipt",
            "precommit_inventory",
        ]
        .contains(&field.as_str())
        {
            return Err(PerfApplicabilityPlanError::ManifestContract {
                gate: requested_gate,
                detail: format!("manifest schemas.{field} is unreviewed"),
            });
        }
    }
    Ok(())
}

/// Run the live manifest admission the planner runs, for one gate.
///
/// The Q2C contract readers must agree with *this* function, not with a
/// restatement of it. Restating the rules has now drifted from them twice —
/// missing gate presence, field placement, the positive QG-1 primary target
/// width, and the exact schema set — so the readers call it instead.
pub(crate) fn validate_normative_manifest(
    manifest: &str,
    gate: PerfGate,
) -> Result<(), PerfApplicabilityPlanError> {
    perf_gate_manifest_identity(manifest, gate).map(|_| ())
}

/// Run that admission for **every** normative gate.
///
/// `perf_gate_manifest_identity` reads gate-specific identity — notably QG-1's
/// positive `primary_target_cell_width` — only from the *requested* gate's own
/// table, so admitting a manifest for one gate says nothing about the other
/// nine. A contract reader asking "would planning accept this file?" must ask
/// for all ten, which is what this does. Still no restatement: every rule comes
/// from the live entry point.
pub(crate) fn validate_normative_manifest_all_gates(
    manifest: &str,
) -> Result<(), PerfApplicabilityPlanError> {
    for gate in PerfGate::ALL {
        validate_normative_manifest(manifest, gate)?;
    }
    Ok(())
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
    validate_perf_manifest_schema_bindings(&parsed, gate)?;
    validate_perf_manifest_gate_set(&parsed, gate)?;
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

        // QG-5 re-baselines on `xlarge` now that the deterministic e6.1
        // generator has landed. Its fixture must remain synchronized with the
        // ratchet pin: a pin with no emitted cell can otherwise score nothing.
        // The tombstone densities and the >=5x force-merge target are unchanged.
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

const fn qg1_arm_id(arm: PerfSampleArm) -> &'static str {
    match arm {
        PerfSampleArm::Control => "control",
        PerfSampleArm::Treatment => "treatment",
    }
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

const fn qg1_order_id(order: PerfSampleOrder) -> &'static str {
    match order {
        PerfSampleOrder::First => "first",
        PerfSampleOrder::Second => "second",
    }
}

fn qg1_stream_role_is_known(role: &str) -> bool {
    matches!(
        role,
        QG1_STREAM_ROLE_EFFECT
            | QG1_STREAM_ROLE_TANTIVY_NULL
            | QG1_STREAM_ROLE_QUILL_NULL
            | QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT
            | QG1_STREAM_ROLE_TANTIVY_PILOT_NULL
    )
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

/// One contiguous prepared-input interval consumed by a QG-1 engine sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1BatchCoverage {
    /// Zero-based offset of the first prepared document in this batch.
    pub document_start: u64,
    /// Number of prepared documents consumed by this batch.
    pub document_count: u64,
}

/// The one terminal lifecycle that authenticated a QG-1 engine arm.
///
/// This is deliberately an enum rather than a collection of optional fields:
/// one raw sample must name exactly one engine-specific witness. A Quill
/// publication receipt cannot be relabelled as a Tantivy writer-join receipt,
/// and vice versa.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case", deny_unknown_fields)]
pub enum Qg1LifecycleWitness {
    /// Quill published the final commit and the retained reader found the
    /// prepared tail document.
    Quill {
        /// Positive publication-generation change caused by the terminal
        /// publishing commit.
        publication_generation_delta: u64,
    },
    /// Tantivy completed one non-rearming writer join before the retained
    /// reader found the prepared tail document.
    Tantivy {
        /// Searchable segment count immediately before the terminal join.
        searchable_segments_before: usize,
        /// Searchable segment count after every terminal worker joined.
        searchable_segments_after: usize,
        /// Time spent in Tantivy's terminal worker join.
        join_elapsed_ns: u64,
        /// Whether the terminal join constructed a replacement writer.
        writer_rearmed: bool,
    },
}

impl Qg1LifecycleWitness {
    fn validate(&self) -> bool {
        match self {
            Self::Quill {
                publication_generation_delta,
            } => *publication_generation_delta > 0,
            Self::Tantivy {
                searchable_segments_before,
                searchable_segments_after,
                join_elapsed_ns,
                writer_rearmed,
                ..
            } => {
                *searchable_segments_before > 0
                    && *searchable_segments_after > 0
                    && *join_elapsed_ns > 0
                    && !writer_rearmed
            }
        }
    }

    fn engine_id(&self) -> &'static str {
        match self {
            Self::Quill { .. } => QG1_QUILL_ENGINE_ID,
            Self::Tantivy { .. } => QG1_TANTIVY_ENGINE_ID,
        }
    }
}

/// One exact raw-row slot issued before timing begins. The authority owns this
/// transcript; raw evidence can only consume a listed slot, never mint one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qg1IssuedRow {
    stream_role: String,
    stream_sequence: u64,
    block_id: u64,
    sample_id: u64,
    arm: PerfSampleArm,
    order: PerfSampleOrder,
    /// Commitment to an entropy-backed producer capability. The capability
    /// preimage exists only in the live producer and is consumed exactly once.
    producer_capability_sha256: String,
}

/// Pre-timing authority for every headline-eligible raw row in one QG-1
/// cell. It is retained by the paired-estimator configuration, outside the raw
/// rows it governs, so resealing mutable row fields cannot replace the planned
/// corpus, schedule, or stream cardinality after measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qg1LifecycleAuthority {
    /// Versioned authority schema; legacy authorities never gain admission.
    pub schema_version: String,
    /// Digest of this complete external authority.
    pub authority_sha256: String,
    /// Exact canonical operation this authority permits.
    pub scope: PerfOperationScope,
    /// Invocation-wide corpus provenance the prepared cell belongs to.
    pub provenance_corpus_sha256: String,
    /// Digest of the authority's complete prepared input constituents.
    pub prepared_input_sha256: String,
    /// Exact normalized manifest used to prepare the input.
    pub prepared_manifest_sha256: String,
    /// Digest of the content supplied to both engines.
    pub indexed_content_sha256: String,
    /// Pre-timing prepared document count and throughput work denominator.
    pub document_count: u64,
    /// Pre-timing prepared content-byte denominator.
    pub content_bytes: u64,
    /// Number of planned feed batches.
    pub prepared_batch_count: usize,
    /// Exact planned batch schedule, in feed order.
    pub batch_coverage: Vec<Qg1BatchCoverage>,
    /// Exact prepared tail required at the terminal endpoint.
    pub tail_document_id: String,
    /// Exact raw-row count every authorized stream must retain.
    pub expected_stream_row_count: u64,
    /// Exact complete-pair count every authorized stream must retain.
    pub expected_pair_count: u64,
    /// Canonically ordered roles issued for this one cell invocation.
    pub stream_roles: Vec<String>,
    /// Exact role-specific raw rows issued before timing, canonically ordered
    /// by role then stream sequence.
    pub issued_rows: Vec<Qg1IssuedRow>,
}

impl Qg1LifecycleAuthority {
    /// Construct and seal one authority before warmup or measurement begins.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied prepared cell is not complete,
    /// canonical, or internally consistent.
    fn new(
        scope: PerfOperationScope,
        provenance_corpus_sha256: String,
        prepared_manifest_sha256: String,
        indexed_content_sha256: String,
        document_count: u64,
        content_bytes: u64,
        prepared_batch_count: usize,
        batch_coverage: Vec<Qg1BatchCoverage>,
        tail_document_id: String,
        expected_pair_count: u64,
        mut issued_rows: Vec<Qg1IssuedRow>,
    ) -> Result<Self, &'static str> {
        issued_rows.sort_by(|left, right| {
            (left.stream_role.as_str(), left.stream_sequence)
                .cmp(&(right.stream_role.as_str(), right.stream_sequence))
        });
        let stream_roles = issued_rows
            .iter()
            .map(|row| row.stream_role.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let expected_stream_row_count = expected_pair_count
            .checked_mul(2)
            .ok_or("QG-1 authority stream row count overflowed")?;
        let mut authority = Self {
            schema_version: QG1_LIFECYCLE_AUTHORITY_SCHEMA_VERSION.to_owned(),
            authority_sha256: String::new(),
            scope,
            provenance_corpus_sha256,
            prepared_input_sha256: String::new(),
            prepared_manifest_sha256,
            indexed_content_sha256,
            document_count,
            content_bytes,
            prepared_batch_count,
            batch_coverage,
            tail_document_id,
            expected_stream_row_count,
            expected_pair_count,
            stream_roles,
            issued_rows,
        };
        authority.prepared_input_sha256 = authority.recomputed_prepared_input_sha256();
        authority.authority_sha256 = authority.recomputed_authority_sha256();
        authority.validate()?;
        Ok(authority)
    }

    fn issue(
        scope: PerfOperationScope,
        provenance_corpus_sha256: String,
        prepared_manifest_sha256: String,
        indexed_content_sha256: String,
        document_count: u64,
        content_bytes: u64,
        prepared_batch_count: usize,
        batch_coverage: Vec<Qg1BatchCoverage>,
        tail_document_id: String,
        expected_pair_count: u64,
        mut issued_rows: Vec<Qg1IssuedRow>,
    ) -> Result<(Self, BTreeMap<String, [u8; 32]>), PairedEstimatorError> {
        let mut capabilities = BTreeMap::new();
        for row in &mut issued_rows {
            let mut capability = [0_u8; 32];
            getrandom::getrandom(&mut capability).map_err(|_| {
                PairedEstimatorError::InvalidConfig {
                    reason: "QG-1 producer capability entropy is unavailable".to_owned(),
                }
            })?;
            row.producer_capability_sha256 = lower_sha256_hex(&capability);
            if capabilities
                .insert(qg1_issued_row_key(row), capability)
                .is_some()
            {
                return Err(PairedEstimatorError::InvalidConfig {
                    reason: "QG-1 producer capability coordinates are duplicated".to_owned(),
                });
            }
        }
        let authority = Self::new(
            scope,
            provenance_corpus_sha256,
            prepared_manifest_sha256,
            indexed_content_sha256,
            document_count,
            content_bytes,
            prepared_batch_count,
            batch_coverage,
            tail_document_id,
            expected_pair_count,
            issued_rows,
        )
        .map_err(|reason| PairedEstimatorError::InvalidConfig {
            reason: reason.to_owned(),
        })?;
        Ok((authority, capabilities))
    }

    /// Derive the role-specific identity this authority assigned to a raw row.
    #[must_use]
    fn stream_role_identity_sha256(&self, role: &str) -> Option<String> {
        self.stream_roles
            .binary_search_by(|candidate| candidate.as_str().cmp(role))
            .ok()
            .map(|_| {
                let mut hasher = Sha256::new();
                hasher.update(b"frankensearch.quill.qg1-authority-role.v1\0");
                update_length_framed(&mut hasher, self.authority_sha256.as_bytes());
                update_length_framed(&mut hasher, role.as_bytes());
                finish_sha256_hex(hasher)
            })
    }

    fn recomputed_prepared_input_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-authority-prepared-input.v1\0");
        for value in [
            self.provenance_corpus_sha256.as_bytes(),
            self.prepared_manifest_sha256.as_bytes(),
            self.indexed_content_sha256.as_bytes(),
            self.tail_document_id.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &self.document_count.to_le_bytes());
        update_length_framed(&mut hasher, &self.content_bytes.to_le_bytes());
        update_length_framed(
            &mut hasher,
            &u64::try_from(self.prepared_batch_count)
                .expect("QG-1 authority batch count fits u64")
                .to_le_bytes(),
        );
        for batch in &self.batch_coverage {
            update_length_framed(&mut hasher, &batch.document_start.to_le_bytes());
            update_length_framed(&mut hasher, &batch.document_count.to_le_bytes());
        }
        finish_sha256_hex(hasher)
    }

    fn recomputed_authority_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-lifecycle-authority.v1\0");
        for value in [
            self.schema_version.as_bytes(),
            self.scope.operation_id.as_bytes(),
            self.scope.unit.as_bytes(),
            self.provenance_corpus_sha256.as_bytes(),
            self.prepared_input_sha256.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &self.scope.version.to_le_bytes());
        update_length_framed(
            &mut hasher,
            match self.scope.semantics {
                PerfMetricSemantics::Throughput => b"throughput",
                PerfMetricSemantics::Duration => b"duration",
                PerfMetricSemantics::GaugeHigherIsBetter => b"gauge_higher_is_better",
                PerfMetricSemantics::GaugeLowerIsBetter => b"gauge_lower_is_better",
            },
        );
        update_length_framed(&mut hasher, &self.expected_stream_row_count.to_le_bytes());
        update_length_framed(&mut hasher, &self.expected_pair_count.to_le_bytes());
        for role in &self.stream_roles {
            update_length_framed(&mut hasher, role.as_bytes());
        }
        for row in &self.issued_rows {
            for value in [
                row.stream_role.as_bytes(),
                qg1_arm_id(row.arm).as_bytes(),
                qg1_order_id(row.order).as_bytes(),
            ] {
                update_length_framed(&mut hasher, value);
            }
            update_length_framed(&mut hasher, &row.stream_sequence.to_le_bytes());
            update_length_framed(&mut hasher, &row.block_id.to_le_bytes());
            update_length_framed(&mut hasher, &row.sample_id.to_le_bytes());
            update_length_framed(&mut hasher, row.producer_capability_sha256.as_bytes());
        }
        finish_sha256_hex(hasher)
    }

    /// Verify the authority's schema, digest, prepared constituents, and
    /// canonical stream plan.
    fn validate(&self) -> Result<(), &'static str> {
        let expected_tail_document_id = self
            .document_count
            .checked_sub(1)
            .map(|ordinal| format!("synthetic-{ordinal:08}"));
        if self.schema_version != QG1_LIFECYCLE_AUTHORITY_SCHEMA_VERSION
            || self.scope.validate().is_err()
            || !is_lower_hex_digest(&self.authority_sha256)
            || !is_lower_hex_digest(&self.provenance_corpus_sha256)
            || !is_lower_hex_digest(&self.prepared_input_sha256)
            || !is_lower_hex_digest(&self.prepared_manifest_sha256)
            || !is_lower_hex_digest(&self.indexed_content_sha256)
            || self.authority_sha256 != self.recomputed_authority_sha256()
            || self.prepared_input_sha256 != self.recomputed_prepared_input_sha256()
            || self.document_count == 0
            || self.content_bytes == 0
            || expected_tail_document_id.as_deref() != Some(&self.tail_document_id)
            || self.prepared_batch_count == 0
            || self.prepared_batch_count != self.batch_coverage.len()
            || self.expected_pair_count == 0
            || self.expected_pair_count.checked_mul(2) != Some(self.expected_stream_row_count)
            || self.stream_roles.is_empty()
            || self
                .stream_roles
                .windows(2)
                .any(|roles| roles[0] >= roles[1])
            || self
                .stream_roles
                .iter()
                .any(|role| !qg1_stream_role_is_known(role))
        {
            return Err("QG-1 lifecycle authority is not a complete canonical prepared-cell plan");
        }
        let mut next_document = 0_u64;
        for batch in &self.batch_coverage {
            if batch.document_start != next_document || batch.document_count == 0 {
                return Err(
                    "QG-1 lifecycle authority batch schedule is not contiguous and positive",
                );
            }
            next_document = next_document
                .checked_add(batch.document_count)
                .ok_or("QG-1 lifecycle authority batch schedule overflowed")?;
        }
        if next_document != self.document_count {
            return Err("QG-1 lifecycle authority batch schedule does not cover its input");
        }
        let mut issued_by_role = BTreeMap::<&str, Vec<&Qg1IssuedRow>>::new();
        let mut capability_commitments = BTreeSet::new();
        for row in &self.issued_rows {
            if !is_lower_hex_digest(&row.producer_capability_sha256)
                || !capability_commitments.insert(row.producer_capability_sha256.as_str())
            {
                return Err(
                    "QG-1 lifecycle authority has invalid or duplicate producer capabilities",
                );
            }
            issued_by_role
                .entry(row.stream_role.as_str())
                .or_default()
                .push(row);
        }
        if issued_by_role.len() != self.stream_roles.len()
            || issued_by_role
                .keys()
                .copied()
                .ne(self.stream_roles.iter().map(String::as_str))
        {
            return Err("QG-1 lifecycle authority transcript roles do not match its role plan");
        }
        for (role, rows) in issued_by_role {
            if !qg1_stream_role_is_known(role)
                || u64::try_from(rows.len()).ok() != Some(self.expected_stream_row_count)
            {
                return Err("QG-1 lifecycle authority transcript row count is not exact");
            }
            let mut seen_coordinates = BTreeSet::<(u64, u64)>::new();
            let mut blocks = BTreeMap::<u64, Vec<&Qg1IssuedRow>>::new();
            for (expected_sequence, row) in rows.into_iter().enumerate() {
                if row.stream_sequence
                    != u64::try_from(expected_sequence)
                        .map_err(|_| "QG-1 transcript sequence does not fit u64")?
                    || !seen_coordinates.insert((row.block_id, row.sample_id))
                {
                    return Err("QG-1 lifecycle authority transcript has duplicate or gapped rows");
                }
                blocks.entry(row.block_id).or_default().push(row);
            }
            if u64::try_from(blocks.len()).ok() != Some(self.expected_pair_count) {
                return Err("QG-1 lifecycle authority transcript pair count is not exact");
            }
            for rows in blocks.into_values() {
                if rows.len() != 2
                    || !rows.iter().any(|row| row.arm == PerfSampleArm::Control)
                    || !rows.iter().any(|row| row.arm == PerfSampleArm::Treatment)
                    || !rows.iter().any(|row| row.order == PerfSampleOrder::First)
                    || !rows.iter().any(|row| row.order == PerfSampleOrder::Second)
                {
                    return Err(
                        "QG-1 lifecycle authority transcript does not issue one randomized pair",
                    );
                }
            }
        }
        Ok(())
    }

    fn issued_row_matches(
        &self,
        stream_role: &str,
        stream_sequence: u64,
        block_id: u64,
        sample_id: u64,
        arm: PerfSampleArm,
        order: PerfSampleOrder,
    ) -> bool {
        self.issued_rows
            .binary_search_by(|row| {
                (row.stream_role.as_str(), row.stream_sequence).cmp(&(stream_role, stream_sequence))
            })
            .ok()
            .is_some_and(|index| {
                let row = &self.issued_rows[index];
                row.block_id == block_id
                    && row.sample_id == sample_id
                    && row.arm == arm
                    && row.order == order
            })
    }

    fn issued_row_for(
        &self,
        stream_role: &str,
        stream_sequence: u64,
        block_id: u64,
        sample_id: u64,
        arm: PerfSampleArm,
        order: PerfSampleOrder,
    ) -> Option<&Qg1IssuedRow> {
        self.issued_rows
            .binary_search_by(|row| {
                (row.stream_role.as_str(), row.stream_sequence).cmp(&(stream_role, stream_sequence))
            })
            .ok()
            .and_then(|index| {
                let row = &self.issued_rows[index];
                (row.block_id == block_id
                    && row.sample_id == sample_id
                    && row.arm == arm
                    && row.order == order)
                    .then_some(row)
            })
    }

    fn issued_row_count(&self, stream_role: &str) -> usize {
        self.issued_rows
            .iter()
            .filter(|row| row.stream_role == stream_role)
            .count()
    }
}

fn qg1_issued_row_key(row: &Qg1IssuedRow) -> String {
    format!(
        "{}:{:020}:{:020}:{:020}:{}:{}",
        row.stream_role,
        row.stream_sequence,
        row.block_id,
        row.sample_id,
        qg1_arm_id(row.arm),
        qg1_order_id(row.order),
    )
}

fn qg1_producer_capability_tag_sha256(
    capability: &[u8; 32],
    binding: &Qg1SampleBinding,
    scope: &PerfOperationScope,
    provenance: &PerfSampleProvenance,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.quill.qg1-producer-capability-tag.v1\0");
    update_length_framed(&mut hasher, capability);
    for value in [
        binding.stream_id_sha256.as_bytes(),
        binding.stream_role.as_bytes(),
        binding.lifecycle_authority_sha256.as_bytes(),
        binding.stream_role_identity_sha256.as_bytes(),
        binding.producer_capability_sha256.as_bytes(),
        binding.prepared_input_sha256.as_bytes(),
        scope.operation_id.as_bytes(),
        scope.unit.as_bytes(),
        provenance.run_id.as_bytes(),
        provenance.executable_sha256.as_bytes(),
        provenance.corpus_sha256.as_bytes(),
        provenance.worker_id.as_bytes(),
        provenance.build_profile.as_bytes(),
    ] {
        update_length_framed(&mut hasher, value);
    }
    update_length_framed(&mut hasher, &binding.stream_sequence.to_le_bytes());
    update_length_framed(&mut hasher, &binding.raw_sample_id.to_le_bytes());
    update_length_framed(&mut hasher, &binding.raw_block_id.to_le_bytes());
    update_length_framed(&mut hasher, qg1_arm_id(binding.raw_arm).as_bytes());
    update_length_framed(&mut hasher, qg1_order_id(binding.raw_order).as_bytes());
    update_length_framed(&mut hasher, &binding.terminal_endpoint_ns.to_le_bytes());
    let witness = serde_json::to_vec(&binding.lifecycle_witness)
        .expect("QG-1 lifecycle witness serializes for its producer capability tag");
    update_length_framed(&mut hasher, &witness);
    finish_sha256_hex(hasher)
}

/// Immutable pre-timing authority retained independently from mutable samples.
/// It deliberately has no serialization implementation: persisted evidence must
/// receive it from the producer's retained authority store, never from itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg1ExpectedAuthority {
    authority: Qg1LifecycleAuthority,
    capability_preimages: BTreeMap<String, [u8; 32]>,
}

impl Qg1ExpectedAuthority {
    /// Stable identity used in externally retained evidence indexes.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.authority.authority_sha256
    }

    /// Whether this retained expectation is the one that issued `config`'s
    /// sealed authority.
    ///
    /// Evidence replay uses this to select, per cell, the single retained
    /// expectation a persisted QG-1 cell was measured under. A non-QG-1
    /// configuration never matches, so a mixed artifact keeps its exact
    /// generic behavior.
    pub(crate) fn matches_config(&self, config: &PairedEstimatorConfig) -> bool {
        config.qg1_lifecycle_authority.as_ref() == Some(&self.authority)
    }

    fn binding_matches_capability(
        &self,
        binding: &Qg1SampleBinding,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) -> bool {
        let Some(row) = self.authority.issued_row_for(
            &binding.stream_role,
            binding.stream_sequence,
            binding.raw_block_id,
            binding.raw_sample_id,
            binding.raw_arm,
            binding.raw_order,
        ) else {
            return false;
        };
        let Some(capability) = self.capability_preimages.get(&qg1_issued_row_key(row)) else {
            return false;
        };
        lower_sha256_hex(capability) == row.producer_capability_sha256
            && binding.producer_capability_tag_sha256
                == qg1_producer_capability_tag_sha256(capability, binding, scope, provenance)
    }

    fn samples_match_capabilities(
        &self,
        samples: impl IntoIterator<Item = &PerfRawSample>,
    ) -> bool {
        samples.into_iter().all(|sample| {
            sample.qg1_sample_binding.as_ref().is_some_and(|binding| {
                self.binding_matches_capability(binding, &sample.scope, &sample.provenance)
            })
        })
    }
}

/// The only live producer able to attach a QG-1 lifecycle receipt. It owns
/// opaque capability preimages and removes one before returning each binding.
#[derive(Debug)]
pub struct Qg1LifecycleProducer {
    expected_authority: Qg1ExpectedAuthority,
    unconsumed_capabilities: RefCell<BTreeMap<String, [u8; 32]>>,
}

impl Qg1LifecycleProducer {
    fn new(authority: Qg1LifecycleAuthority, capabilities: BTreeMap<String, [u8; 32]>) -> Self {
        Self {
            expected_authority: Qg1ExpectedAuthority {
                authority,
                capability_preimages: capabilities.clone(),
            },
            unconsumed_capabilities: RefCell::new(capabilities),
        }
    }

    /// Return the producer-retained expectation required for later replay.
    #[must_use]
    pub fn expected_authority(&self) -> &Qg1ExpectedAuthority {
        &self.expected_authority
    }

    /// Consume one pre-issued capability and attach its receipt to a live row.
    /// No public resealing API exists; copied or coordinate-mutated rows retain
    /// the source commitment and fail the independently retained authority.
    #[must_use]
    pub fn consume_lifecycle_receipt(
        &self,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        mut binding: Qg1SampleBinding,
    ) -> Option<Qg1SampleBinding> {
        let authority = &self.expected_authority.authority;
        if authority.validate().is_err()
            || authority.scope != *scope
            || authority.provenance_corpus_sha256 != provenance.corpus_sha256
        {
            return None;
        }
        let row = authority.issued_row_for(
            &binding.stream_role,
            binding.stream_sequence,
            binding.raw_block_id,
            binding.raw_sample_id,
            binding.raw_arm,
            binding.raw_order,
        )?;
        let capability = self
            .unconsumed_capabilities
            .borrow_mut()
            .remove(&qg1_issued_row_key(row))?;
        if lower_sha256_hex(&capability) != row.producer_capability_sha256 {
            return None;
        }
        binding.lifecycle_authority_sha256 = authority.authority_sha256.clone();
        binding.stream_role_identity_sha256 =
            authority.stream_role_identity_sha256(&binding.stream_role)?;
        binding.producer_capability_sha256 = row.producer_capability_sha256.clone();
        binding.seal_lifecycle_receipt(scope, provenance);
        binding.producer_capability_tag_sha256 =
            qg1_producer_capability_tag_sha256(&capability, &binding, scope, provenance);
        binding.seal_lifecycle_receipt(scope, provenance);
        (binding.matches_authority(authority, scope, provenance)
            && self
                .expected_authority
                .binding_matches_capability(&binding, scope, provenance))
        .then_some(binding)
    }
}

/// Typed lifecycle binding retained with each headline-eligible QG-1 sample.
///
/// It joins the exact prepared corpus identities, complete batch schedule,
/// prepared tail proof, one arm-specific lifecycle witness, and the endpoint
/// that supplied the throughput denominator. `NoClaim` diagnostics have no
/// value of this type and therefore cannot enter the paired estimator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1SampleBinding {
    /// Versioned nested receipt schema.  Missing, unknown, or older receipts
    /// never silently gain headline eligibility.
    pub schema_version: String,
    /// Stable role of this stream in a QG-1 invocation.  The estimator maps
    /// this exact value to the required engine identities; it is not a label
    /// an arm may freely relabel.
    pub stream_role: String,
    /// Domain-separated identity for the one role stream in this invocation.
    pub stream_id_sha256: String,
    /// Exact zero-based position of this raw row in canonical raw-order
    /// enumeration within `stream_id_sha256`.
    pub stream_sequence: u64,
    /// Raw-row coordinates duplicated into the receipt so a lifecycle proof
    /// cannot be copied to another raw sample, block, or arm.
    pub raw_sample_id: u64,
    /// Block coordinate duplicated into the receipt.
    pub raw_block_id: u64,
    /// Arm coordinate duplicated into the receipt.
    pub raw_arm: PerfSampleArm,
    /// Execution order duplicated into the receipt.  Sample IDs are arm
    /// coordinates, not chronological coordinates, so this prevents a valid
    /// receipt from being replayed after a first/second order swap.
    pub raw_order: PerfSampleOrder,
    /// Digest of the immutable pre-timing lifecycle authority held by the
    /// paired-estimator configuration, never synthesized from this raw row.
    pub lifecycle_authority_sha256: String,
    /// Authority-issued identity for this exact stream role.
    pub stream_role_identity_sha256: String,
    /// Commitment to the consumed one-shot producer capability for this exact
    /// raw transcript slot. The preimage is never serialized with the row.
    pub producer_capability_sha256: String,
    /// Capability-preimage tag over this exact receipt payload. Only the
    /// independently retained authority can verify it after serialization.
    pub producer_capability_tag_sha256: String,
    /// Deterministic identity of this unique lifecycle receipt.
    pub lifecycle_receipt_id_sha256: String,
    /// Digest over the receipt identity, frozen prepared input, terminal
    /// endpoint, and exactly-one arm-specific witness.
    pub lifecycle_receipt_sha256: String,
    /// Invocation-wide QG-1 corpus identity copied from raw provenance.
    pub prepared_corpus_sha256: String,
    /// Digest of the complete frozen prepared input, including its batch
    /// schedule.  Every effect and null stream must carry the same value.
    pub prepared_input_sha256: String,
    /// Exact normalized gate-manifest identity used while preparing the input.
    pub prepared_manifest_sha256: String,
    /// Digest of exactly the indexed prepared content.
    pub indexed_content_sha256: String,
    /// Number of documents in the prepared input.
    pub document_count: u64,
    /// Total UTF-8 content bytes in that prepared input.
    pub content_bytes: u64,
    /// Number of batches fixed before timing began.
    pub prepared_batch_count: usize,
    /// Number of batch feed calls that completed during this sample.
    pub recorded_batch_count: usize,
    /// Complete contiguous coverage of the prepared input.
    pub batch_coverage: Vec<Qg1BatchCoverage>,
    /// Exact prepared tail document proved searchable at the endpoint.
    pub tail_document_id: String,
    /// Measured terminal searchable-and-quiescent offset from interval start.
    pub terminal_endpoint_ns: u64,
    /// Exactly one engine-specific lifecycle witness for this arm.
    pub lifecycle_witness: Qg1LifecycleWitness,
}

impl Qg1SampleBinding {
    /// Current fail-closed schema identifier for QG-1 lifecycle receipts.
    pub const SCHEMA_VERSION: &'static str = QG1_LIFECYCLE_BINDING_SCHEMA_VERSION;

    /// Seal the deterministic receipt fields after the harness has recorded
    /// this row's immutable coordinates and prepared input.  The paired
    /// estimator always recomputes these values; this is a construction helper,
    /// not a trust bypass.
    fn seal_lifecycle_receipt(
        &mut self,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) {
        self.prepared_input_sha256 = self.recomputed_prepared_input_sha256();
        self.stream_id_sha256 = self.recomputed_stream_id_sha256(scope, provenance);
        self.lifecycle_receipt_id_sha256 = self.recomputed_lifecycle_receipt_id_sha256();
        self.lifecycle_receipt_sha256 = self.recomputed_lifecycle_receipt_sha256();
    }

    fn stream_role_is_known(&self) -> bool {
        qg1_stream_role_is_known(&self.stream_role)
    }

    fn recomputed_prepared_input_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-authority-prepared-input.v1\0");
        for value in [
            self.prepared_corpus_sha256.as_bytes(),
            self.prepared_manifest_sha256.as_bytes(),
            self.indexed_content_sha256.as_bytes(),
            self.tail_document_id.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &self.document_count.to_le_bytes());
        update_length_framed(&mut hasher, &self.content_bytes.to_le_bytes());
        update_length_framed(
            &mut hasher,
            &u64::try_from(self.prepared_batch_count)
                .expect("QG-1 batch count fits u64")
                .to_le_bytes(),
        );
        for batch in &self.batch_coverage {
            update_length_framed(&mut hasher, &batch.document_start.to_le_bytes());
            update_length_framed(&mut hasher, &batch.document_count.to_le_bytes());
        }
        finish_sha256_hex(hasher)
    }

    fn recomputed_stream_id_sha256(
        &self,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-lifecycle-stream.v4\0");
        for value in [
            scope.operation_id.as_bytes(),
            scope.unit.as_bytes(),
            provenance.run_id.as_bytes(),
            provenance.executable_sha256.as_bytes(),
            provenance.corpus_sha256.as_bytes(),
            provenance.worker_id.as_bytes(),
            provenance.build_profile.as_bytes(),
            self.stream_role.as_bytes(),
            self.prepared_input_sha256.as_bytes(),
            self.lifecycle_authority_sha256.as_bytes(),
            self.stream_role_identity_sha256.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &scope.version.to_le_bytes());
        update_length_framed(
            &mut hasher,
            match scope.semantics {
                PerfMetricSemantics::Throughput => b"throughput",
                PerfMetricSemantics::Duration => b"duration",
                PerfMetricSemantics::GaugeHigherIsBetter => b"gauge_higher_is_better",
                PerfMetricSemantics::GaugeLowerIsBetter => b"gauge_lower_is_better",
            },
        );
        finish_sha256_hex(hasher)
    }

    fn recomputed_lifecycle_receipt_id_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-lifecycle-receipt-id.v4\0");
        for value in [
            self.schema_version.as_bytes(),
            self.stream_id_sha256.as_bytes(),
            self.stream_role.as_bytes(),
            qg1_arm_id(self.raw_arm).as_bytes(),
            qg1_order_id(self.raw_order).as_bytes(),
            self.lifecycle_authority_sha256.as_bytes(),
            self.stream_role_identity_sha256.as_bytes(),
            self.producer_capability_sha256.as_bytes(),
            self.producer_capability_tag_sha256.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &self.stream_sequence.to_le_bytes());
        update_length_framed(&mut hasher, &self.raw_sample_id.to_le_bytes());
        update_length_framed(&mut hasher, &self.raw_block_id.to_le_bytes());
        finish_sha256_hex(hasher)
    }

    fn recomputed_lifecycle_receipt_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.quill.qg1-lifecycle-receipt.v4\0");
        for value in [
            self.schema_version.as_bytes(),
            self.lifecycle_receipt_id_sha256.as_bytes(),
            self.stream_id_sha256.as_bytes(),
            self.stream_role.as_bytes(),
            qg1_arm_id(self.raw_arm).as_bytes(),
            qg1_order_id(self.raw_order).as_bytes(),
            self.prepared_input_sha256.as_bytes(),
            self.lifecycle_authority_sha256.as_bytes(),
            self.stream_role_identity_sha256.as_bytes(),
            self.producer_capability_sha256.as_bytes(),
        ] {
            update_length_framed(&mut hasher, value);
        }
        update_length_framed(&mut hasher, &self.stream_sequence.to_le_bytes());
        update_length_framed(&mut hasher, &self.raw_sample_id.to_le_bytes());
        update_length_framed(&mut hasher, &self.raw_block_id.to_le_bytes());
        update_length_framed(&mut hasher, &self.terminal_endpoint_ns.to_le_bytes());
        let witness = serde_json::to_vec(&self.lifecycle_witness)
            .expect("QG-1 lifecycle witness serializes without allocation failure");
        update_length_framed(&mut hasher, &witness);
        finish_sha256_hex(hasher)
    }

    fn validate_for_raw(
        &self,
        raw: &PerfRawSample,
        elapsed_ns: u64,
        work_units: u64,
        byte_count: Option<u64>,
    ) -> Result<(), &'static str> {
        let expected_tail_document_id = self
            .document_count
            .checked_sub(1)
            .map(|ordinal| format!("synthetic-{ordinal:08}"));
        if self.schema_version != Self::SCHEMA_VERSION
            || !self.stream_role_is_known()
            || !is_lower_hex_digest(&self.stream_id_sha256)
            || !is_lower_hex_digest(&self.lifecycle_receipt_id_sha256)
            || !is_lower_hex_digest(&self.lifecycle_receipt_sha256)
            || !is_lower_hex_digest(&self.prepared_corpus_sha256)
            || !is_lower_hex_digest(&self.prepared_input_sha256)
            || !is_lower_hex_digest(&self.lifecycle_authority_sha256)
            || !is_lower_hex_digest(&self.stream_role_identity_sha256)
            || !is_lower_hex_digest(&self.producer_capability_sha256)
            || !is_lower_hex_digest(&self.producer_capability_tag_sha256)
            || self.raw_sample_id != raw.sample_id
            || self.raw_block_id != raw.block_id
            || self.raw_arm != raw.arm
            || self.raw_order != raw.order
        {
            return Err("QG-1 lifecycle receipt is not uniquely bound to its raw row");
        }
        if self.prepared_corpus_sha256 != raw.provenance.corpus_sha256
            || self.prepared_input_sha256 != self.recomputed_prepared_input_sha256()
            || self.stream_id_sha256
                != self.recomputed_stream_id_sha256(&raw.scope, &raw.provenance)
            || self.lifecycle_receipt_id_sha256 != self.recomputed_lifecycle_receipt_id_sha256()
            || self.lifecycle_receipt_sha256 != self.recomputed_lifecycle_receipt_sha256()
        {
            return Err("QG-1 lifecycle receipt digest is not authenticated to raw provenance");
        }
        if !is_lower_hex_digest(&self.prepared_manifest_sha256)
            || !is_lower_hex_digest(&self.indexed_content_sha256)
            || self.document_count == 0
            || self.content_bytes == 0
            || self.tail_document_id.trim().is_empty()
            || self.tail_document_id.len() > 256
            || expected_tail_document_id.as_deref() != Some(&self.tail_document_id)
        {
            return Err("QG-1 lifecycle binding has invalid prepared input identity");
        }
        if self.prepared_batch_count == 0
            || self.prepared_batch_count != self.recorded_batch_count
            || self.prepared_batch_count != self.batch_coverage.len()
        {
            return Err(
                "QG-1 lifecycle binding does not retain one complete prepared batch schedule",
            );
        }
        let mut next_document = 0_u64;
        for batch in &self.batch_coverage {
            if batch.document_start != next_document || batch.document_count == 0 {
                return Err("QG-1 lifecycle binding batch coverage is not contiguous and positive");
            }
            next_document = next_document
                .checked_add(batch.document_count)
                .ok_or("QG-1 lifecycle binding batch coverage overflowed")?;
        }
        if next_document != self.document_count {
            return Err("QG-1 lifecycle binding batch coverage differs from prepared input");
        }
        if work_units != self.document_count || byte_count != Some(self.content_bytes) {
            return Err("QG-1 raw denominator differs from its lifecycle-bound prepared input");
        }
        if self.terminal_endpoint_ns == 0 || self.terminal_endpoint_ns != elapsed_ns {
            return Err("QG-1 raw interval does not end at its lifecycle-bound terminal endpoint");
        }
        if !self.lifecycle_witness.validate() {
            return Err("QG-1 lifecycle binding has an invalid arm-specific witness");
        }
        Ok(())
    }

    fn same_prepared_input(&self, other: &Self) -> bool {
        self.prepared_corpus_sha256 == other.prepared_corpus_sha256
            && self.prepared_input_sha256 == other.prepared_input_sha256
            && self.lifecycle_authority_sha256 == other.lifecycle_authority_sha256
            && self.prepared_manifest_sha256 == other.prepared_manifest_sha256
            && self.indexed_content_sha256 == other.indexed_content_sha256
            && self.document_count == other.document_count
            && self.content_bytes == other.content_bytes
            && self.prepared_batch_count == other.prepared_batch_count
            && self.recorded_batch_count == other.recorded_batch_count
            && self.batch_coverage == other.batch_coverage
            && self.tail_document_id == other.tail_document_id
    }

    /// Verify this raw receipt's public commitment against the issued authority.
    /// The capability preimage is checked separately by the retained expected
    /// authority, so callers cannot reseal a replacement receipt themselves.
    #[must_use]
    fn matches_authority(
        &self,
        authority: &Qg1LifecycleAuthority,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) -> bool {
        authority.validate().is_ok()
            && authority.scope == *scope
            && authority.provenance_corpus_sha256 == provenance.corpus_sha256
            && self.lifecycle_authority_sha256 == authority.authority_sha256
            && authority.stream_role_identity_sha256(&self.stream_role)
                == Some(self.stream_role_identity_sha256.clone())
            && authority
                .issued_row_for(
                    &self.stream_role,
                    self.stream_sequence,
                    self.raw_block_id,
                    self.raw_sample_id,
                    self.raw_arm,
                    self.raw_order,
                )
                .is_some_and(|row| {
                    row.producer_capability_sha256 == self.producer_capability_sha256
                })
            && authority.issued_row_matches(
                &self.stream_role,
                self.stream_sequence,
                self.raw_block_id,
                self.raw_sample_id,
                self.raw_arm,
                self.raw_order,
            )
            && self.prepared_corpus_sha256 == authority.provenance_corpus_sha256
            && self.prepared_input_sha256 == authority.prepared_input_sha256
            && self.prepared_manifest_sha256 == authority.prepared_manifest_sha256
            && self.indexed_content_sha256 == authority.indexed_content_sha256
            && self.document_count == authority.document_count
            && self.content_bytes == authority.content_bytes
            && self.prepared_batch_count == authority.prepared_batch_count
            && self.recorded_batch_count == authority.prepared_batch_count
            && self.batch_coverage == authority.batch_coverage
            && self.tail_document_id == authority.tail_document_id
    }

    fn engine_id(&self) -> &'static str {
        self.lifecycle_witness.engine_id()
    }
}

/// One bounded raw record emitted by the timing harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Complete prepared-input and terminal-lifecycle binding for a QG-1
    /// throughput sample. Required for every headline-eligible QG-1 rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qg1_sample_binding: Option<Qg1SampleBinding>,
    /// Exact QG-1 Tantivy candidate configuration for this arm, when the arm
    /// runs Tantivy. Non-Tantivy arms carry no value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tantivy_config_sha256: Option<String>,
}

impl PerfRawSample {
    fn validate_and_value(&self) -> Result<f64, PairedEstimatorError> {
        self.scope.validate()?;
        self.provenance.validate()?;
        if self
            .tantivy_config_sha256
            .as_deref()
            .is_some_and(|digest| !is_lower_hex_digest(digest))
        {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "Tantivy-bound samples require a lowercase SHA-256 configuration ID"
                    .to_owned(),
            });
        }
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
        let qg1_scope = is_canonical_qg1_throughput_scope(&self.scope);
        if qg1_scope {
            let work_units = self.work_units.filter(|value| *value > 0).ok_or_else(|| {
                PairedEstimatorError::InvalidValue {
                    sample_id: self.sample_id,
                    reason: "throughput samples require positive work_units".to_owned(),
                }
            })?;
            let binding = self.qg1_sample_binding.as_ref().ok_or_else(|| {
                PairedEstimatorError::InvalidProvenance {
                    reason: "QG-1 throughput samples require one typed prepared-input and lifecycle binding"
                        .to_owned(),
                }
            })?;
            binding
                .validate_for_raw(self, elapsed_ns, work_units, self.byte_count)
                .map_err(|reason| PairedEstimatorError::InvalidProvenance {
                    reason: reason.to_owned(),
                })?;
        } else if self.qg1_sample_binding.is_some() {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "non-QG-1-throughput samples cannot carry QG-1 lifecycle bindings"
                    .to_owned(),
            });
        }
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
    /// Immutable pre-timing QG-1 lifecycle authority. Canonical QG-1
    /// throughput rows are rejected unless this authority verifies every
    /// receipt, while every non-QG-1 estimator invocation leaves it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    qg1_lifecycle_authority: Option<Qg1LifecycleAuthority>,
    /// Live-only authority preimages retained outside mutable raw samples.
    /// They are deliberately excluded from persisted artifacts, which must
    /// supply an independently retained expectation during replay.
    #[serde(skip)]
    qg1_expected_authority: Option<Qg1ExpectedAuthority>,
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
            qg1_lifecycle_authority: None,
            qg1_expected_authority: None,
        }
    }

    fn has_predeclared_thresholds(&self) -> bool {
        let expected = Self::predeclared(self.bootstrap_seed);
        self.bootstrap_resamples == expected.bootstrap_resamples
            && self.min_pairs == expected.min_pairs
            && self.max_order_imbalance == expected.max_order_imbalance
            && self.max_null_center_log == expected.max_null_center_log
            && self.max_null_ci_half_width_log == expected.max_null_ci_half_width_log
            && self.max_null_log_mad == expected.max_null_log_mad
            && self.max_null_order_effect_log == expected.max_null_order_effect_log
            && self.max_null_drift_log == expected.max_null_drift_log
            && self.summary_direction_dead_band_log == expected.summary_direction_dead_band_log
            && self.max_reproduction_delta_log == expected.max_reproduction_delta_log
    }

    /// Freeze the complete QG-1 prepared-cell authority before any warmup or
    /// timing sample is produced.
    ///
    /// The authority remains inside this estimator configuration rather than
    /// being reconstructed from serializable raw rows during replay.
    ///
    /// # Errors
    ///
    /// `issued_streams` supplies `(role, block_id_base, sample_id_base,
    /// first_arms)`. Each `first_arms` entry expands to the exact control and
    /// treatment rows for one randomized pair, then becomes immutable inside
    /// the authority before the runner starts.
    ///
    /// Returns an invalid-configuration error when the proposed QG-1 plan is
    /// not one complete canonical authority, or when an authority was already
    /// frozen into this configuration.
    pub fn install_qg1_lifecycle_authority(
        &mut self,
        scope: PerfOperationScope,
        provenance_corpus_sha256: String,
        prepared_manifest_sha256: String,
        indexed_content_sha256: String,
        document_count: u64,
        content_bytes: u64,
        prepared_batch_count: usize,
        batch_coverage: Vec<Qg1BatchCoverage>,
        tail_document_id: String,
        expected_pair_count: u64,
        issued_streams: Vec<(String, u64, u64, Vec<PerfSampleArm>)>,
    ) -> Result<Qg1LifecycleProducer, PairedEstimatorError> {
        if self.qg1_lifecycle_authority.is_some() {
            return Err(PairedEstimatorError::InvalidConfig {
                reason: "QG-1 lifecycle authority is single-assignment once timing is planned"
                    .to_owned(),
            });
        }
        let mut issued_rows = Vec::new();
        for (stream_role, block_id_base, sample_id_base, first_arms) in issued_streams {
            if u64::try_from(first_arms.len()).ok() != Some(expected_pair_count) {
                return Err(PairedEstimatorError::InvalidConfig {
                    reason: "QG-1 issued transcript does not contain the exact planned pair count"
                        .to_owned(),
                });
            }
            for (round, first_arm) in first_arms.into_iter().enumerate() {
                let round =
                    u64::try_from(round).map_err(|_| PairedEstimatorError::InvalidConfig {
                        reason: "QG-1 issued transcript round does not fit u64".to_owned(),
                    })?;
                let block_id = block_id_base.checked_add(round).ok_or_else(|| {
                    PairedEstimatorError::InvalidConfig {
                        reason: "QG-1 issued transcript block ID overflowed".to_owned(),
                    }
                })?;
                let control_sample_id = sample_id_base
                    .checked_add(round.checked_mul(2).ok_or_else(|| {
                        PairedEstimatorError::InvalidConfig {
                            reason: "QG-1 issued transcript sample ID overflowed".to_owned(),
                        }
                    })?)
                    .ok_or_else(|| PairedEstimatorError::InvalidConfig {
                        reason: "QG-1 issued transcript sample ID overflowed".to_owned(),
                    })?;
                let treatment_sample_id = control_sample_id.checked_add(1).ok_or_else(|| {
                    PairedEstimatorError::InvalidConfig {
                        reason: "QG-1 issued transcript sample ID overflowed".to_owned(),
                    }
                })?;
                let stream_sequence =
                    round
                        .checked_mul(2)
                        .ok_or_else(|| PairedEstimatorError::InvalidConfig {
                            reason: "QG-1 issued transcript sequence overflowed".to_owned(),
                        })?;
                let second_arm = match first_arm {
                    PerfSampleArm::Control => PerfSampleArm::Treatment,
                    PerfSampleArm::Treatment => PerfSampleArm::Control,
                };
                for (offset, arm, order, sample_id) in [
                    (
                        0_u64,
                        first_arm,
                        PerfSampleOrder::First,
                        if first_arm == PerfSampleArm::Control {
                            control_sample_id
                        } else {
                            treatment_sample_id
                        },
                    ),
                    (
                        1_u64,
                        second_arm,
                        PerfSampleOrder::Second,
                        if second_arm == PerfSampleArm::Control {
                            control_sample_id
                        } else {
                            treatment_sample_id
                        },
                    ),
                ] {
                    issued_rows.push(Qg1IssuedRow {
                        stream_role: stream_role.clone(),
                        stream_sequence: stream_sequence.checked_add(offset).ok_or_else(|| {
                            PairedEstimatorError::InvalidConfig {
                                reason: "QG-1 issued transcript sequence overflowed".to_owned(),
                            }
                        })?,
                        block_id,
                        sample_id,
                        arm,
                        order,
                        producer_capability_sha256: String::new(),
                    });
                }
            }
        }
        let (authority, capabilities) = Qg1LifecycleAuthority::issue(
            scope,
            provenance_corpus_sha256,
            prepared_manifest_sha256,
            indexed_content_sha256,
            document_count,
            content_bytes,
            prepared_batch_count,
            batch_coverage,
            tail_document_id,
            expected_pair_count,
            issued_rows,
        )?;
        self.qg1_lifecycle_authority = Some(authority.clone());
        let producer = Qg1LifecycleProducer::new(authority, capabilities);
        self.qg1_expected_authority = Some(producer.expected_authority().clone());
        Ok(producer)
    }

    /// Compare one QG-1 raw receipt against this configuration's authority.
    #[must_use]
    pub fn qg1_binding_matches_lifecycle_authority(
        &self,
        binding: &Qg1SampleBinding,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
    ) -> bool {
        self.qg1_lifecycle_authority
            .as_ref()
            .is_some_and(|authority| binding.matches_authority(authority, scope, provenance))
    }

    /// Confirm that a live producer is the retained authority for this exact
    /// QG-1 invocation before it consumes a one-shot capability.
    #[must_use]
    pub fn qg1_expected_authority_matches(&self, expected: &Qg1ExpectedAuthority) -> bool {
        self.qg1_expected_authority.as_ref() == Some(expected)
    }

    /// Return the exact number of pre-issued rows for one QG-1 stream role.
    #[must_use]
    pub fn qg1_issued_stream_row_count(
        &self,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        stream_role: &str,
    ) -> Option<usize> {
        let authority = self.qg1_lifecycle_authority.as_ref()?;
        if authority.validate().is_err()
            || authority.scope != *scope
            || authority.provenance_corpus_sha256 != provenance.corpus_sha256
        {
            return None;
        }
        Some(authority.issued_row_count(stream_role))
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
        if self.bootstrap_resamples < 100
            || self.min_pairs < 4
            || !finite_non_negative
            || self
                .qg1_lifecycle_authority
                .as_ref()
                .is_some_and(|authority| authority.validate().is_err())
            || self
                .qg1_expected_authority
                .as_ref()
                .is_some_and(|expected| !expected.matches_config(self))
        {
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
    #[error("paired block {block_id} mixes QG-1 prepared-input lifecycle bindings")]
    Qg1BindingMismatch { block_id: u64 },
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
    /// Recompute a non-QG-1 persisted estimate from retained raw records.
    ///
    /// Persisted QG-1 evidence must instead call
    /// [`Self::verify_recomputed_against_qg1_authority`] with the independently
    /// retained authority frozen before measurement. The artifact's serialized
    /// configuration is evidence to compare, never its own replay authority.
    ///
    /// # Errors
    ///
    /// Returns [`PairedEstimatorError::InconsistentSummary`] on any mismatch.
    pub fn verify_recomputed(&self) -> Result<(), PairedEstimatorError> {
        self.verify_recomputed_against_qg1_authority(None)
    }

    /// Recompute retained evidence against an authority supplied outside the
    /// serialized result.
    ///
    /// The caller must retain this authority from the pre-timing cell plan. A
    /// QG-1 artifact cannot satisfy this requirement by cloning or presenting
    /// the authority embedded in its own configuration.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error unless the optional expectation
    /// exactly matches the artifact's QG-1 authority presence and complete
    /// expected authority.
    pub fn verify_recomputed_against_qg1_authority(
        &self,
        expected_qg1_authority: Option<&Qg1ExpectedAuthority>,
    ) -> Result<(), PairedEstimatorError> {
        if !self.config.has_predeclared_thresholds() || self.config.validate().is_err() {
            return Err(PairedEstimatorError::InvalidConfig {
                reason: "persisted evidence must use the exact predeclared estimator thresholds"
                    .to_owned(),
            });
        }
        let authority_matches = match (
            self.config.qg1_lifecycle_authority.as_ref(),
            expected_qg1_authority,
        ) {
            (None, None) => true,
            (Some(_), Some(expected)) => expected.matches_config(&self.config),
            _ => false,
        };
        if !authority_matches {
            return Err(PairedEstimatorError::InvalidConfig {
                reason:
                    "persisted QG-1 evidence requires an independently supplied expected authority"
                        .to_owned(),
            });
        }
        if let Some(expected) = expected_qg1_authority {
            if !expected
                .samples_match_capabilities(self.effect_samples.iter().chain(&self.null_samples))
            {
                return Err(PairedEstimatorError::InvalidProvenance {
                    reason: "persisted QG-1 evidence does not match consumed producer capabilities"
                        .to_owned(),
                });
            }
        }
        let recomputed = estimate_paired_experiment_inner(
            &self.effect_samples,
            &self.null_samples,
            &self.config,
        )?;
        if recomputed == *self {
            Ok(())
        } else {
            Err(PairedEstimatorError::InconsistentSummary)
        }
    }

    /// Whether this result recomputes against a proven QG-1 expectation.
    ///
    /// A live producer supplies it through the configuration it installed; a
    /// reloaded artifact cannot, so its consumer passes the retained
    /// expectation as `external`. Canonical QG-1 evidence with neither is
    /// refused by the authority-bearing estimator, never silently admitted.
    fn recomputes_against_qg1_authority(
        &self,
        external_qg1_authority: Option<&Qg1ExpectedAuthority>,
    ) -> bool {
        // A live producer supplies the expectation through the configuration
        // it installed; a consumer supplies the one it retained. Neither can
        // be absent for canonical QG-1 evidence without failing closed below.
        let expected = external_qg1_authority.or(self.config.qg1_expected_authority.as_ref());
        self.config.has_predeclared_thresholds()
            && self.config.validate().is_ok()
            && estimate_paired_experiment_against_qg1_authority(
                &self.effect_samples,
                &self.null_samples,
                &self.config,
                expected,
            )
            .is_ok_and(|recomputed| recomputed == *self)
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qg1ValidatedStream {
    role: String,
    lifecycle_authority_sha256: String,
}

fn qg1_expected_engine_for_role(role: &str, arm: PerfSampleArm) -> Option<&'static str> {
    match (role, arm) {
        (QG1_STREAM_ROLE_EFFECT, PerfSampleArm::Control) => Some(QG1_TANTIVY_ENGINE_ID),
        (QG1_STREAM_ROLE_EFFECT, PerfSampleArm::Treatment) => Some(QG1_QUILL_ENGINE_ID),
        (QG1_STREAM_ROLE_TANTIVY_NULL, _) => Some(QG1_TANTIVY_ENGINE_ID),
        (QG1_STREAM_ROLE_QUILL_NULL, _) => Some(QG1_QUILL_ENGINE_ID),
        (QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT, _) => Some(QG1_TANTIVY_ENGINE_ID),
        (QG1_STREAM_ROLE_TANTIVY_PILOT_NULL, _) => Some(QG1_TANTIVY_ENGINE_ID),
        _ => None,
    }
}

fn qg1_validate_stream(
    samples: &[PerfRawSample],
    scope: &PerfOperationScope,
    provenance: &PerfSampleProvenance,
    actual_pair_count: usize,
    authority: Option<&Qg1LifecycleAuthority>,
) -> Result<Option<Qg1ValidatedStream>, PairedEstimatorError> {
    if !is_canonical_qg1_throughput_scope(scope) {
        return Ok(None);
    }
    let first = samples
        .first()
        .and_then(|sample| sample.qg1_sample_binding.as_ref())
        .ok_or_else(|| PairedEstimatorError::InvalidProvenance {
            reason: "canonical QG-1 throughput streams require lifecycle receipts".to_owned(),
        })?;
    let authority = authority.ok_or_else(|| PairedEstimatorError::InvalidProvenance {
        reason: "canonical QG-1 throughput streams require one pre-timing lifecycle authority"
            .to_owned(),
    })?;
    if authority.validate().is_err()
        || authority.scope != *scope
        || authority.provenance_corpus_sha256 != provenance.corpus_sha256
    {
        return Err(PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 lifecycle authority does not match the estimator scope and provenance"
                .to_owned(),
        });
    }
    let expected_stream_id = first.recomputed_stream_id_sha256(scope, provenance);
    let mut receipt_ids = BTreeSet::new();
    let actual_stream_row_count =
        u64::try_from(samples.len()).map_err(|_| PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 raw stream row count does not fit its receipt schema".to_owned(),
        })?;
    let actual_pair_count =
        u64::try_from(actual_pair_count).map_err(|_| PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 raw stream pair count does not fit its receipt schema".to_owned(),
        })?;
    if authority.expected_stream_row_count != actual_stream_row_count
        || authority.expected_pair_count != actual_pair_count
    {
        return Err(PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 lifecycle receipt expected stream row/pair count differs from raw stream"
                .to_owned(),
        });
    }
    let mut canonical = samples.iter().collect::<Vec<_>>();
    canonical.sort_by_key(|sample| {
        let order = match sample.order {
            PerfSampleOrder::First => 0_u8,
            PerfSampleOrder::Second => 1_u8,
        };
        (sample.block_id, order, sample.arm, sample.sample_id)
    });
    for (expected_sequence, sample) in canonical.into_iter().enumerate() {
        let expected_sequence = u64::try_from(expected_sequence).map_err(|_| {
            PairedEstimatorError::InvalidProvenance {
                reason: "QG-1 raw-order sequence does not fit its receipt schema".to_owned(),
            }
        })?;
        let binding = sample.qg1_sample_binding.as_ref().ok_or_else(|| {
            PairedEstimatorError::InvalidProvenance {
                reason: "canonical QG-1 throughput stream omitted a lifecycle receipt".to_owned(),
            }
        })?;
        let expected_engine = qg1_expected_engine_for_role(&first.stream_role, sample.arm)
            .ok_or_else(|| PairedEstimatorError::InvalidProvenance {
                reason: "QG-1 lifecycle receipt names an unknown stream role".to_owned(),
            })?;
        if binding.stream_role != first.stream_role
            || binding.stream_id_sha256 != expected_stream_id
            || binding.lifecycle_authority_sha256 != first.lifecycle_authority_sha256
            || !binding.same_prepared_input(first)
            || !binding.matches_authority(authority, scope, provenance)
            || binding.engine_id() != expected_engine
        {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "QG-1 stream mixes prepared input, stream identity, or engine role"
                    .to_owned(),
            });
        }
        if binding.stream_sequence != expected_sequence
            || !receipt_ids.insert(binding.lifecycle_receipt_id_sha256.clone())
        {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "QG-1 stream receipt is reused or does not match canonical raw-order enumeration"
                    .to_owned(),
            });
        }
    }
    Ok(Some(Qg1ValidatedStream {
        role: first.stream_role.clone(),
        lifecycle_authority_sha256: first.lifecycle_authority_sha256.clone(),
    }))
}

fn qg1_validate_experiment_streams(
    effect: Option<Qg1ValidatedStream>,
    null: Option<Qg1ValidatedStream>,
) -> Result<(), PairedEstimatorError> {
    match (effect, null) {
        (None, None) => Ok(()),
        (Some(effect), Some(null))
            if effect.lifecycle_authority_sha256 == null.lifecycle_authority_sha256
                && matches!(
                    (effect.role.as_str(), null.role.as_str()),
                    (QG1_STREAM_ROLE_EFFECT, QG1_STREAM_ROLE_TANTIVY_NULL)
                        | (QG1_STREAM_ROLE_EFFECT, QG1_STREAM_ROLE_QUILL_NULL)
                        | (
                            QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT,
                            QG1_STREAM_ROLE_TANTIVY_PILOT_NULL
                        )
                ) =>
        {
            Ok(())
        }
        (Some(_), Some(_)) => Err(PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 effect/null streams must share one pre-timing lifecycle authority and canonical engine roles"
                .to_owned(),
        }),
        _ => Err(PairedEstimatorError::InvalidProvenance {
            reason: "QG-1 lifecycle evidence cannot be paired with a non-QG-1 stream".to_owned(),
        }),
    }
}

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
        match (
            control.qg1_sample_binding.as_ref(),
            treatment.qg1_sample_binding.as_ref(),
        ) {
            (Some(control_binding), Some(treatment_binding))
                if control_binding.same_prepared_input(treatment_binding) => {}
            (None, None) => {}
            _ => return Err(PairedEstimatorError::Qg1BindingMismatch { block_id }),
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
    if let (Some(scope), Some(provenance)) = (stream_scope, stream_provenance) {
        let _ = qg1_validate_stream(
            &raw,
            scope,
            provenance,
            pairs.len(),
            config.qg1_lifecycle_authority.as_ref(),
        )?;
    }
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
/// Canonical QG-1 throughput evidence is refused here. Its receipts are only
/// authenticated by a producer-retained capability, so admitting it through an
/// authority-free entry would let a fully resealed row estimate itself; such
/// callers must use [`estimate_paired_experiment_against_qg1_authority`].
///
/// # Errors
///
/// Returns a typed fail-closed error for malformed pairs, mixed scopes or
/// provenance, undersampling, invalid raw values, and any canonical QG-1
/// throughput sample.
pub fn estimate_paired_experiment(
    effect_samples: &[PerfRawSample],
    null_samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
) -> Result<PairedExperimentResult, PairedEstimatorError> {
    if effect_samples
        .iter()
        .chain(null_samples.iter())
        .any(|sample| is_canonical_qg1_throughput_scope(&sample.scope))
    {
        return Err(PairedEstimatorError::InvalidProvenance {
            reason: "canonical QG-1 throughput evidence must be estimated against an \
                     independently retained expected authority"
                .to_owned(),
        });
    }
    estimate_paired_experiment_inner(effect_samples, null_samples, config)
}

/// Shared estimator body reached only after the caller's QG-1 admission
/// decision. It performs no authority check of its own by design: every public
/// entry above it either refuses canonical QG-1 scopes outright or has already
/// matched the retained producer capabilities.
fn estimate_paired_experiment_inner(
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
    qg1_validate_experiment_streams(
        qg1_validate_stream(
            &effect_raw,
            &scope,
            &provenance,
            effect_pairs.len(),
            config.qg1_lifecycle_authority.as_ref(),
        )?,
        qg1_validate_stream(
            &null_raw,
            &null_scope,
            &null_provenance,
            null_pairs.len(),
            config.qg1_lifecycle_authority.as_ref(),
        )?,
    )?;
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
    // bd-yo5by: drift and order-effect were computed for the effect stream but
    // never bounded, so a quiet null phase followed by a drifting or
    // carryover-asymmetric effect phase was admissible. Gate both with the same
    // predeclared bounds the null already answers to.
    match effect.order_effect_log {
        Some(order_effect) if order_effect.abs() <= config.max_null_order_effect_log => {}
        Some(order_effect) => {
            experiment_invalid = true;
            push_reason(
                &mut reasons,
                "paired.effect_order_effect",
                format!(
                    "A/B order effect {order_effect:.6} exceeds {:.6}",
                    config.max_null_order_effect_log
                ),
            );
        }
        None => {
            experiment_invalid = true;
            push_reason(
                &mut reasons,
                "paired.effect_order_unobserved",
                "A/B stream did not execute both randomized orders",
            );
        }
    }
    if effect.drift_log.abs() > config.max_null_drift_log {
        experiment_invalid = true;
        push_reason(
            &mut reasons,
            "paired.effect_drift",
            format!(
                "A/B first/second-half drift {:.6} exceeds {:.6}",
                effect.drift_log, config.max_null_drift_log
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

/// Estimate a live or replayed QG-1 experiment against authority retained
/// outside its mutable raw samples. Generic callers retain the exact previous
/// contract by supplying `None`; an authority-bearing QG-1 invocation must
/// supply the producer-retained expectation.
///
/// # Errors
///
/// Rejects missing, foreign, or self-substituted authority before estimating.
pub fn estimate_paired_experiment_against_qg1_authority(
    effect_samples: &[PerfRawSample],
    null_samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
    expected_qg1_authority: Option<&Qg1ExpectedAuthority>,
) -> Result<PairedExperimentResult, PairedEstimatorError> {
    let qg1_samples = effect_samples
        .iter()
        .chain(null_samples.iter())
        .any(|sample| is_canonical_qg1_throughput_scope(&sample.scope));
    match (qg1_samples, expected_qg1_authority) {
        (true, Some(expected))
            if expected.matches_config(config)
                && expected.samples_match_capabilities(
                    effect_samples.iter().chain(null_samples.iter()),
                ) => {}
        (true, _) => {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "QG-1 estimation requires an independently retained expected authority"
                    .to_owned(),
            });
        }
        (false, None) => {}
        (false, Some(_)) => {
            return Err(PairedEstimatorError::InvalidProvenance {
                reason: "non-QG-1 estimation cannot accept a QG-1 expected authority".to_owned(),
            });
        }
    }
    estimate_paired_experiment_inner(effect_samples, null_samples, config)
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

#[cfg(any(target_os = "linux", test))]
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

/// True only for a normative QG-1 bulk-indexing cell with the exact native
/// continuous-throughput operation contract.  This is deliberately rebuilt
/// from the canonical matrix instead of treating a `QG-1.*` string prefix as
/// an admission authority.
fn is_canonical_qg1_throughput_scope(scope: &PerfOperationScope) -> bool {
    PerfMatrixSpec::complete()
        .for_gate(PerfGate::Qg1)
        .into_iter()
        .filter(|cell| cell.metric == "docs_per_second")
        .any(|cell| {
            scope
                == &PerfOperationScope {
                    operation_id: format!("{}.{}.{}", PerfGate::Qg1, cell.fixture, cell.metric),
                    version: 1,
                    semantics: PerfMetricSemantics::Throughput,
                    unit: "docs/s".to_owned(),
                }
        })
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

#[cfg(any(target_os = "linux", test))]
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

// ─── Human run plan generation (bd-quill-e8-perf-doctrine-x4e4.15) ──────────
//
// The normative TOML is the machine-readable source of truth; the human run
// plan is DERIVED from it (and from the compiled matrix), never hand-written.
// `render_perf_run_plan_markdown` renders the complete plan; the drift test
// in this module fails when the committed document differs from a fresh
// render, so docs and harness cannot disagree silently.

/// Repository path of the generated run-plan document.
pub const PERF_RUN_PLAN_DOC_PATH: &str = "docs/contracts/quill-perf-gates.run-plan.md";

/// Render the human operator run plan from the normative manifest and the
/// canonical cell matrix.
///
/// The document carries the manifest contract hash it was rendered from; a
/// reader can prove freshness by comparing hashes, and the drift test proves
/// it on every build.
///
/// # Errors
///
/// Returns an error when the manifest is malformed or lacks the defaults the
/// command template and machine table are rendered from.
pub fn render_perf_run_plan_markdown() -> Result<String, GauntletError> {
    use std::fmt::Write as _;

    let manifest: toml::Value = toml::from_str(NORMATIVE_PERF_MANIFEST).map_err(|error| {
        GauntletError::InvalidCampaign {
            reason: format!("normative perf manifest does not parse: {error}"),
        }
    })?;
    let defaults = manifest
        .get("defaults")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| GauntletError::InvalidCampaign {
            reason: "normative perf manifest lacks [defaults]".to_owned(),
        })?;
    let host_command = defaults
        .get("registered_host_command")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| GauntletError::InvalidCampaign {
            reason: "normative perf manifest lacks defaults.registered_host_command".to_owned(),
        })?;
    let machines = defaults
        .get("machines")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| GauntletError::InvalidCampaign {
            reason: "normative perf manifest lacks defaults.machines".to_owned(),
        })?;
    let gates = manifest
        .get("gate")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| GauntletError::InvalidCampaign {
            reason: "normative perf manifest lacks [gate] tables".to_owned(),
        })?;
    let matrix = PerfMatrixSpec::complete();
    validate_matrix(&matrix)?;

    let mut out = String::new();
    out.push_str("# Quill Performance Gates — Generated Operator Run Plan\n\n");
    out.push_str(
        "**GENERATED FILE — do not edit.** Rendered from `quill-perf-gates.toml` and the\n\
         compiled `PerfMatrixSpec` by `render_perf_run_plan_markdown`; the gauntlet test\n\
         `perf_run_plan_document_matches_the_manifest` fails closed on any drift. Regenerate\n\
         deliberately with `QUILL_PERF_RUN_PLAN_UPDATE=1`.\n\n",
    );
    let _ = writeln!(
        &mut out,
        "- manifest contract SHA-256: `{}`",
        perf_manifest_contract_sha256(NORMATIVE_PERF_MANIFEST)
    );
    let _ = writeln!(
        &mut out,
        "- canonical matrix cells: {}\n",
        matrix.cells.len()
    );

    out.push_str("## Registered machines\n\n");
    out.push_str("| hardware-class | execution-profile | registry status |\n");
    out.push_str("|---|---|---|\n");
    let mut machine_keys = Vec::new();
    for machine in machines {
        let machine = machine
            .as_str()
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: "defaults.machines entries must be strings".to_owned(),
            })?;
        let (key, status) = machine.split_once(' ').unwrap_or((machine, ""));
        let status = status
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let (hardware_class, profile) =
            key.split_once('/')
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: format!(
                        "machine entry {key} is not <hardware-class>/<execution-profile>"
                    ),
                })?;
        let _ = writeln!(&mut out, "| {hardware_class} | {profile} | {status} |");
        machine_keys.push((hardware_class.to_owned(), profile.to_owned()));
    }
    out.push('\n');

    out.push_str("## Gates\n\n");
    for gate in PerfGate::ALL {
        let table = gates
            .get(gate.label())
            .and_then(toml::Value::as_table)
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: format!("manifest lacks [gate.{}]", gate.label()),
            })?;
        let name = table
            .get("name")
            .and_then(toml::Value::as_str)
            .unwrap_or("<unnamed>");
        let fixture = table
            .get("fixture")
            .and_then(toml::Value::as_str)
            .unwrap_or("<no fixture>");
        let target = table
            .get("target")
            .and_then(toml::Value::as_str)
            .unwrap_or("<no target>");
        let activated = table
            .get("activated")
            .and_then(toml::Value::as_bool)
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: format!("gate {} lacks a boolean activated flag", gate.label()),
            })?;
        let _ = write!(&mut out, "### {} — {}\n\n", gate.label(), name);
        let _ = writeln!(&mut out, "- activated: `{activated}`");
        let _ = writeln!(&mut out, "- fixture: {fixture}");
        let _ = writeln!(&mut out, "- target: {target}");
        if let Some(width) = table.get("primary_target_cell_width") {
            let _ = writeln!(&mut out, "- primary_target_cell_width: {width}");
        }
        if let Some(qpc) = table.get("queries_per_class") {
            let _ = writeln!(&mut out, "- queries_per_class: {qpc}");
        }
        out.push('\n');

        let cells = matrix.for_gate(gate);
        let _ = write!(&mut out, "**Canonical cells ({}):**\n\n", cells.len());
        out.push_str("| fixture | metric | corpus | threads | positions | extras |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for cell in cells {
            let corpus = cell
                .corpus
                .map_or_else(|| "-".to_owned(), |c| c.label().to_owned());
            let threads = cell
                .threads
                .map_or_else(|| "-".to_owned(), |t| t.to_string());
            let positions = cell
                .positions
                .map_or_else(|| "-".to_owned(), |p| p.label().to_owned());
            let mut extras = Vec::new();
            if let Some(density) = cell.tombstone_density_pct {
                extras.push(format!("tombstones={density}%"));
            }
            if let Some(class) = cell.query_class {
                extras.push(format!("class={}", class.label()));
            }
            if let Some(k) = cell.k {
                extras.push(format!("k={k}"));
            }
            if let Some(topology) = cell.topology {
                extras.push(format!("topology={topology:?}"));
            }
            let _ = writeln!(
                &mut out,
                "| {} | {} | {} | {} | {} | {} |",
                cell.fixture,
                cell.metric,
                corpus,
                threads,
                positions,
                extras.join(", ")
            );
        }
        out.push('\n');

        out.push_str("**Registered-host commands:**\n\n```text\n");
        for (hardware_class, profile) in &machine_keys {
            let command = host_command
                .replace("<QG-N>", gate.label())
                .replace("<hardware-class>", hardware_class)
                .replace("<execution-profile>", profile);
            out.push_str(&command);
            out.push('\n');
        }
        out.push_str("```\n\n");
    }
    Ok(out)
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

    fn qg1_bulk_cell(threads: usize) -> PerfCellSpec {
        PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|cell| {
                cell.metric == "docs_per_second"
                    && cell.corpus == Some(PerfCorpus::Tiny)
                    && cell.threads == Some(threads)
                    && cell.positions == Some(PositionMode::On)
            })
            .expect("canonical QG-1 tiny bulk cell")
            .clone()
    }

    fn qg1_semantic_contract() -> Qg1TantivySemanticContract {
        Qg1TantivySemanticContract {
            tantivy_version: QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION.to_owned(),
            schema_sha256: "a".repeat(64),
            analyzer_sha256: "b".repeat(64),
            indexed_fields_sha256: "c".repeat(64),
            merge_policy_sha256: "d".repeat(64),
            visibility_sha256: "e".repeat(64),
            searchable_terminal_scope_sha256: "f".repeat(64),
            durability_sha256: "0".repeat(64),
            quill_config_sha256: "1".repeat(64),
        }
    }

    fn qg1_screen_plan(cell: &PerfCellSpec, widths: Vec<usize>) -> Qg1TantivyIncumbentScreenPlan {
        Qg1TantivyIncumbentScreenPlan::new(
            profile_key(
                HardwareClassId::TrjZen35995wx,
                ExecutionProfileId::Physical64,
            ),
            2,
            widths,
            cell,
            64_000,
        )
        .expect("screen plan")
    }

    fn qg1_throughput_scope(cell: &PerfCellSpec) -> PerfOperationScope {
        qg1_expected_throughput_scope(cell).expect("canonical QG-1 throughput scope")
    }

    fn qg1_test_capability(stream_role: &str, round: u64, offset: u64, sample_id: u64) -> [u8; 32] {
        Sha256::digest(
            format!("qg1-test-capability:{stream_role}:{round}:{offset}:{sample_id}").as_bytes(),
        )
        .into()
    }

    fn qg1_test_expected_authority(authority: &Qg1LifecycleAuthority) -> Qg1ExpectedAuthority {
        let capability_preimages = authority
            .issued_rows
            .iter()
            .map(|row| {
                (
                    qg1_issued_row_key(row),
                    qg1_test_capability(
                        &row.stream_role,
                        row.block_id,
                        row.stream_sequence % 2,
                        row.sample_id,
                    ),
                )
            })
            .collect();
        Qg1ExpectedAuthority {
            authority: authority.clone(),
            capability_preimages,
        }
    }

    fn qg1_test_authority(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        work_units: u64,
        content_bytes: u64,
        expected_pair_count: u64,
        issued_streams: &[(&str, u64)],
    ) -> Qg1LifecycleAuthority {
        let first_arms = seeded_balanced_pair_order(
            usize::try_from(expected_pair_count).expect("QG-1 test pair count fits usize"),
            0x00dd_5eed,
        )
        .expect("QG-1 test issued order");
        let mut issued_rows = Vec::new();
        for (stream_role, sample_id_base) in issued_streams {
            for (round, first_arm) in first_arms.iter().copied().enumerate() {
                let round = u64::try_from(round).expect("QG-1 test round fits u64");
                let control_sample_id = sample_id_base + round * 2;
                let treatment_sample_id = control_sample_id + 1;
                let second_arm = match first_arm {
                    PerfSampleArm::Control => PerfSampleArm::Treatment,
                    PerfSampleArm::Treatment => PerfSampleArm::Control,
                };
                for (offset, arm, order, sample_id) in [
                    (
                        0_u64,
                        first_arm,
                        PerfSampleOrder::First,
                        if first_arm == PerfSampleArm::Control {
                            control_sample_id
                        } else {
                            treatment_sample_id
                        },
                    ),
                    (
                        1_u64,
                        second_arm,
                        PerfSampleOrder::Second,
                        if second_arm == PerfSampleArm::Control {
                            control_sample_id
                        } else {
                            treatment_sample_id
                        },
                    ),
                ] {
                    issued_rows.push(Qg1IssuedRow {
                        stream_role: (*stream_role).to_owned(),
                        stream_sequence: round * 2 + offset,
                        block_id: round,
                        sample_id,
                        arm,
                        order,
                        producer_capability_sha256: lower_sha256_hex(&qg1_test_capability(
                            stream_role,
                            round,
                            offset,
                            sample_id,
                        )),
                    });
                }
            }
        }
        Qg1LifecycleAuthority::new(
            scope.clone(),
            provenance.corpus_sha256.clone(),
            "a".repeat(64),
            "b".repeat(64),
            work_units,
            content_bytes,
            1,
            vec![Qg1BatchCoverage {
                document_start: 0,
                document_count: work_units,
            }],
            format!("synthetic-{:08}", work_units.saturating_sub(1)),
            expected_pair_count,
            issued_rows,
        )
        .expect("freeze QG-1 test lifecycle authority")
    }

    fn qg1_test_sample_binding(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        authority: &Qg1LifecycleAuthority,
        work_units: u64,
        content_bytes: u64,
        elapsed_ns: u64,
        engine_id: &str,
        stream_role: &str,
        stream_sequence: u64,
        sample_id: u64,
        block_id: u64,
        arm: PerfSampleArm,
        order: PerfSampleOrder,
    ) -> Qg1SampleBinding {
        let lifecycle_witness = match engine_id {
            QG1_QUILL_ENGINE_ID => Qg1LifecycleWitness::Quill {
                publication_generation_delta: 1,
            },
            QG1_TANTIVY_ENGINE_ID => Qg1LifecycleWitness::Tantivy {
                searchable_segments_before: 1,
                searchable_segments_after: 1,
                join_elapsed_ns: 1,
                writer_rearmed: false,
            },
            _ => panic!("QG-1 test binding requires a known engine ID"),
        };
        let mut binding = Qg1SampleBinding {
            schema_version: Qg1SampleBinding::SCHEMA_VERSION.to_owned(),
            stream_role: stream_role.to_owned(),
            stream_id_sha256: String::new(),
            stream_sequence,
            raw_sample_id: sample_id,
            raw_block_id: block_id,
            raw_arm: arm,
            raw_order: order,
            lifecycle_authority_sha256: authority.authority_sha256.clone(),
            stream_role_identity_sha256: authority
                .stream_role_identity_sha256(stream_role)
                .expect("QG-1 test authority permits its stream role"),
            producer_capability_sha256: authority
                .issued_row_for(
                    stream_role,
                    stream_sequence,
                    block_id,
                    sample_id,
                    arm,
                    order,
                )
                .expect("QG-1 test authority issues its raw row")
                .producer_capability_sha256
                .clone(),
            producer_capability_tag_sha256: String::new(),
            lifecycle_receipt_id_sha256: String::new(),
            lifecycle_receipt_sha256: String::new(),
            prepared_corpus_sha256: provenance.corpus_sha256.clone(),
            prepared_input_sha256: String::new(),
            prepared_manifest_sha256: "a".repeat(64),
            indexed_content_sha256: "b".repeat(64),
            document_count: work_units,
            content_bytes,
            prepared_batch_count: 1,
            recorded_batch_count: 1,
            batch_coverage: vec![Qg1BatchCoverage {
                document_start: 0,
                document_count: work_units,
            }],
            tail_document_id: format!("synthetic-{:08}", work_units.saturating_sub(1)),
            terminal_endpoint_ns: elapsed_ns,
            lifecycle_witness,
        };
        binding.seal_lifecycle_receipt(scope, provenance);
        let row = authority
            .issued_row_for(
                stream_role,
                stream_sequence,
                block_id,
                sample_id,
                arm,
                order,
            )
            .expect("QG-1 test authority issues the receipt slot");
        binding.producer_capability_tag_sha256 = qg1_producer_capability_tag_sha256(
            &qg1_test_capability(
                &row.stream_role,
                row.block_id,
                row.stream_sequence % 2,
                row.sample_id,
            ),
            &binding,
            scope,
            provenance,
        );
        binding.seal_lifecycle_receipt(scope, provenance);
        binding
    }

    fn qg1_observation_ids(label: &str, samples: &[PerfRawSample]) -> Vec<String> {
        samples
            .iter()
            .map(|sample| {
                lower_sha256_hex(
                    format!("{label}:{}:{}", sample.block_id, sample.sample_id).as_bytes(),
                )
            })
            .collect()
    }

    fn qg1_duration_stream(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        control_durations: &[u64],
        treatment_durations: &[u64],
        sample_id_base: u64,
        work_units: u64,
        content_bytes: u64,
        authority: &Qg1LifecycleAuthority,
        control_engine_id: &str,
        treatment_engine_id: &str,
        stream_role: &str,
    ) -> Vec<PerfRawSample> {
        let mut samples = duration_stream(
            scope,
            provenance,
            control_durations,
            treatment_durations,
            sample_id_base,
        );
        let expected_stream_row_count =
            u64::try_from(samples.len()).expect("QG-1 test stream row count fits u64");
        assert_eq!(
            expected_stream_row_count % 2,
            0,
            "QG-1 test stream has an even raw row count"
        );
        let expected_pair_count = expected_stream_row_count / 2;
        assert_eq!(
            authority.expected_stream_row_count, expected_stream_row_count,
            "QG-1 test authority owns the exact raw stream row count"
        );
        assert_eq!(
            authority.expected_pair_count, expected_pair_count,
            "QG-1 test authority owns the exact complete-pair count"
        );
        let mut canonical_indices = (0..samples.len()).collect::<Vec<_>>();
        canonical_indices.sort_by_key(|index| {
            let sample = &samples[*index];
            let order = match sample.order {
                PerfSampleOrder::First => 0_u8,
                PerfSampleOrder::Second => 1_u8,
            };
            (sample.block_id, order, sample.arm, sample.sample_id)
        });
        for (sequence, index) in canonical_indices.into_iter().enumerate() {
            let sample = &mut samples[index];
            sample.work_units = Some(work_units);
            sample.byte_count = Some(content_bytes);
            sample.qg1_sample_binding = Some(qg1_test_sample_binding(
                scope,
                provenance,
                authority,
                work_units,
                content_bytes,
                sample.ended_ns - sample.started_ns,
                match sample.arm {
                    PerfSampleArm::Control => control_engine_id,
                    PerfSampleArm::Treatment => treatment_engine_id,
                },
                stream_role,
                u64::try_from(sequence).expect("QG-1 test stream sequence fits u64"),
                sample.sample_id,
                sample.block_id,
                sample.arm,
                sample.order,
            ));
        }
        samples
    }

    fn qg1_pilot(
        candidate: Qg1TantivyIncumbentCandidate,
        shipping_auto: &Qg1TantivyIncumbentCandidate,
        scope: &PerfOperationScope,
        work_units: u64,
        content_bytes: u64,
        run_id: &str,
        treatment_duration: u64,
    ) -> Qg1TantivyIncumbentPilot {
        let provenance = provenance(run_id);
        let control_durations = [1_000_000; PERF_MIN_RUNS];
        let treatment_durations = [treatment_duration; PERF_MIN_RUNS];
        let authority = qg1_test_authority(
            scope,
            &provenance,
            work_units,
            content_bytes,
            u64::try_from(PERF_MIN_RUNS).expect("QG-1 pilot pair count fits u64"),
            &[
                (QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT, 0),
                (QG1_STREAM_ROLE_TANTIVY_PILOT_NULL, 10_000),
            ],
        );
        let effect = qg1_duration_stream(
            scope,
            &provenance,
            &control_durations,
            &treatment_durations,
            0,
            work_units,
            content_bytes,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_TANTIVY_ENGINE_ID,
            QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT,
        );
        let null = qg1_duration_stream(
            scope,
            &provenance,
            &control_durations,
            &control_durations,
            10_000,
            work_units,
            content_bytes,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_TANTIVY_ENGINE_ID,
            QG1_STREAM_ROLE_TANTIVY_PILOT_NULL,
        );
        let mut estimator_config = estimator_config();
        estimator_config.qg1_lifecycle_authority = Some(authority.clone());
        estimator_config.qg1_expected_authority = Some(qg1_test_expected_authority(&authority));
        let experiment = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &null,
            &estimator_config,
            estimator_config.qg1_expected_authority.as_ref(),
        )
        .expect("valid QG-1 candidate pilot");
        let observed_writer_threads = match candidate.writer_mode {
            Qg1TantivyWriterMode::ShippingAuto => 1,
            Qg1TantivyWriterMode::Fixed { writer_threads } => writer_threads,
        };
        let effect_observation_label = format!("pilot-effect:{}", candidate.config_sha256);
        let null_observation_label = format!("pilot-null:{}", candidate.config_sha256);
        Qg1TantivyIncumbentPilot::from_experiment(
            candidate,
            observed_writer_threads,
            shipping_auto.config_sha256.clone(),
            experiment,
            qg1_observation_ids(&effect_observation_label, &effect),
            qg1_observation_ids(&null_observation_label, &null),
        )
        .expect("seal pilot")
    }

    fn qg1_complete_pilots(
        cell: &PerfCellSpec,
        screen_plan: &Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
        run_id: &str,
    ) -> Vec<Qg1TantivyIncumbentPilot> {
        let candidates = preregister_qg1_tantivy_incumbents(cell, screen_plan, semantic_contract)
            .expect("preregister");
        let shipping_auto = &candidates[0];
        let scope = qg1_throughput_scope(cell);
        candidates
            .iter()
            .cloned()
            .map(|candidate| {
                let treatment_duration = match candidate.writer_mode {
                    Qg1TantivyWriterMode::ShippingAuto => 1_000_000,
                    Qg1TantivyWriterMode::Fixed { writer_threads } => {
                        1_000_000 / u64::try_from(writer_threads).expect("positive writer width")
                    }
                };
                qg1_pilot(
                    candidate,
                    shipping_auto,
                    &scope,
                    screen_plan.work_units,
                    screen_plan.content_bytes,
                    run_id,
                    treatment_duration,
                )
            })
            .collect()
    }

    fn qg1_bound_stream(
        cell: &PerfCellSpec,
        kind: Qg1TantivyDecisionStreamKind,
        run_id: &str,
        control_engine_id: &str,
        control_engine_config_sha256: &str,
        treatment_engine_id: &str,
        treatment_engine_config_sha256: &str,
        treatment_duration: u64,
        work_units: u64,
        content_bytes: u64,
        authority: &Qg1LifecycleAuthority,
    ) -> Qg1TantivyBoundStream {
        let scope = qg1_throughput_scope(cell);
        let provenance = provenance(run_id);
        let durations = [1_000_000; PERF_MIN_RUNS];
        let sample_id_base = match kind {
            Qg1TantivyDecisionStreamKind::TantivyVsQuill => 100_000,
            Qg1TantivyDecisionStreamKind::TantivyNull => 200_000,
            Qg1TantivyDecisionStreamKind::QuillNull => 300_000,
        };
        let effect = qg1_duration_stream(
            &scope,
            &provenance,
            &durations,
            &[treatment_duration; PERF_MIN_RUNS],
            sample_id_base,
            work_units,
            content_bytes,
            authority,
            control_engine_id,
            treatment_engine_id,
            match kind {
                Qg1TantivyDecisionStreamKind::TantivyVsQuill => QG1_STREAM_ROLE_EFFECT,
                Qg1TantivyDecisionStreamKind::TantivyNull => QG1_STREAM_ROLE_TANTIVY_NULL,
                Qg1TantivyDecisionStreamKind::QuillNull => QG1_STREAM_ROLE_QUILL_NULL,
            },
        );
        let observation_ids = qg1_observation_ids(kind.stable_id(), &effect);
        Qg1TantivyBoundStream::from_raw_samples(
            kind,
            control_engine_id.to_owned(),
            control_engine_config_sha256.to_owned(),
            treatment_engine_id.to_owned(),
            treatment_engine_config_sha256.to_owned(),
            effect,
            observation_ids,
        )
        .expect("seal bound stream")
    }

    fn qg1_decision(
        cell: &PerfCellSpec,
        screen_plan: &Qg1TantivyIncumbentScreenPlan,
        semantic_contract: &Qg1TantivySemanticContract,
        selected: &Qg1TantivyIncumbentCandidate,
        run_id: &str,
    ) -> Qg1TantivyIncumbentDecision {
        let scope = qg1_throughput_scope(cell);
        let provenance = provenance(run_id);
        let authority = qg1_test_authority(
            &scope,
            &provenance,
            screen_plan.work_units,
            screen_plan.content_bytes,
            u64::try_from(PERF_MIN_RUNS).expect("QG-1 decision pair count fits u64"),
            &[
                (QG1_STREAM_ROLE_EFFECT, 100_000),
                (QG1_STREAM_ROLE_TANTIVY_NULL, 200_000),
                (QG1_STREAM_ROLE_QUILL_NULL, 300_000),
            ],
        );
        let mut estimator_config = estimator_config();
        estimator_config.qg1_lifecycle_authority = Some(authority.clone());
        estimator_config.qg1_expected_authority = Some(qg1_test_expected_authority(&authority));
        Qg1TantivyIncumbentDecision {
            estimator_config,
            tantivy_vs_quill: qg1_bound_stream(
                cell,
                Qg1TantivyDecisionStreamKind::TantivyVsQuill,
                run_id,
                QG1_TANTIVY_ENGINE_ID,
                &selected.config_sha256,
                QG1_QUILL_ENGINE_ID,
                &semantic_contract.quill_config_sha256,
                500_000,
                screen_plan.work_units,
                screen_plan.content_bytes,
                &authority,
            ),
            tantivy_null: qg1_bound_stream(
                cell,
                Qg1TantivyDecisionStreamKind::TantivyNull,
                run_id,
                QG1_TANTIVY_ENGINE_ID,
                &selected.config_sha256,
                QG1_TANTIVY_ENGINE_ID,
                &selected.config_sha256,
                1_000_000,
                screen_plan.work_units,
                screen_plan.content_bytes,
                &authority,
            ),
            quill_null: qg1_bound_stream(
                cell,
                Qg1TantivyDecisionStreamKind::QuillNull,
                run_id,
                QG1_QUILL_ENGINE_ID,
                &semantic_contract.quill_config_sha256,
                QG1_QUILL_ENGINE_ID,
                &semantic_contract.quill_config_sha256,
                1_000_000,
                screen_plan.work_units,
                screen_plan.content_bytes,
                &authority,
            ),
        }
    }

    fn qg1_rebind_stream_scope(stream: &mut Qg1TantivyBoundStream, scope: &PerfOperationScope) {
        for sample in &mut stream.samples {
            sample.scope = scope.clone();
        }
        let observation_ids = stream
            .observations
            .iter()
            .map(|observation| observation.observation_id_sha256.clone())
            .collect();
        stream.observations = qg1_bind_raw_observations(
            &stream.samples,
            &stream.control_engine_id,
            &stream.control_engine_config_sha256,
            &stream.treatment_engine_id,
            &stream.treatment_engine_config_sha256,
            observation_ids,
        )
        .expect("scope-bound raw observations");
        stream.stream_receipt_sha256 = stream
            .recomputed_stream_receipt_sha256()
            .expect("scope-bound stream receipt");
    }

    #[test]
    fn qg1_lifecycle_binding_hostile_mutations_reach_the_live_estimator() {
        let cell = qg1_bulk_cell(4);
        let scope = qg1_throughput_scope(&cell);
        let provenance = provenance("qg1-binding-hostile");
        let durations = [1_000_000; PERF_MIN_RUNS];
        let mut treatment_durations = [900_000; PERF_MIN_RUNS];
        treatment_durations[PERF_MIN_RUNS - 1] = 940_000;
        let authority = qg1_test_authority(
            &scope,
            &provenance,
            500,
            64_000,
            u64::try_from(PERF_MIN_RUNS).expect("QG-1 hostile pair count fits u64"),
            &[
                (QG1_STREAM_ROLE_EFFECT, 0),
                (QG1_STREAM_ROLE_TANTIVY_NULL, 10_000),
            ],
        );
        let mut config = estimator_config();
        config.qg1_lifecycle_authority = Some(authority.clone());
        let effect = qg1_duration_stream(
            &scope,
            &provenance,
            &durations,
            &treatment_durations,
            0,
            500,
            64_000,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_QUILL_ENGINE_ID,
            QG1_STREAM_ROLE_EFFECT,
        );
        let null = qg1_duration_stream(
            &scope,
            &provenance,
            &durations,
            &durations,
            10_000,
            500,
            64_000,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_TANTIVY_ENGINE_ID,
            QG1_STREAM_ROLE_TANTIVY_NULL,
        );
        let expected_authority = qg1_test_expected_authority(&authority);
        // The public authority-free estimator refuses canonical QG-1 scopes
        // outright, so every hostile case below is exercised through the
        // authority-bearing entry and cannot pass for the trivial reason.
        assert!(
            matches!(
                estimate_paired_experiment(&effect, &null, &config),
                Err(PairedEstimatorError::InvalidProvenance { .. })
            ),
            "the authority-free estimator must refuse canonical QG-1 throughput evidence"
        );
        let intact_experiment = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &null,
            &config,
            Some(&expected_authority),
        )
        .expect("the intact QG-1 lifecycle binding must reach the authority-bearing estimator");
        assert!(
            intact_experiment.verify_recomputed().is_err(),
            "a persisted QG-1 artifact must not authenticate itself from its embedded authority"
        );
        intact_experiment
            .verify_recomputed_against_qg1_authority(Some(&expected_authority))
            .expect(
                "persisted QG-1 evidence recomputes against the independently retained authority",
            );
        let mut live_config = config.clone();
        live_config.qg1_expected_authority = Some(expected_authority.clone());
        let live_intact_experiment = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &null,
            &live_config,
            live_config.qg1_expected_authority.as_ref(),
        )
        .expect("the intact QG-1 lifecycle binding reaches the screen and decision guard");
        assert!(
            qg1_valid_throughput_experiment(
                &live_intact_experiment,
                None,
                &scope,
                Some(&provenance),
                500,
                64_000,
            ),
            "the shared QG-1 screen and decision guard admits live producer-authenticated evidence"
        );
        assert!(
            qg1_valid_throughput_experiment(
                &intact_experiment,
                Some(&expected_authority),
                &scope,
                Some(&provenance),
                500,
                64_000,
            ),
            "a reloaded QG-1 experiment is admitted by the externally supplied retained authority"
        );
        assert!(
            !qg1_valid_throughput_experiment(
                &intact_experiment,
                None,
                &scope,
                Some(&provenance),
                500,
                64_000,
            ),
            "removing the external authority must fail the guard closed, never admit or panic"
        );

        let assert_rejected = |effect: Vec<PerfRawSample>, label: &str| {
            assert!(
                estimate_paired_experiment_against_qg1_authority(
                    &effect,
                    &null,
                    &config,
                    Some(&expected_authority),
                )
                .is_err(),
                "the live estimator accepted hostile QG-1 lifecycle mutation: {label}"
            );
        };
        let assert_streams_rejected =
            |effect: Vec<PerfRawSample>, null: Vec<PerfRawSample>, label: &str| {
                assert!(
                    estimate_paired_experiment_against_qg1_authority(
                        &effect,
                        &null,
                        &config,
                        Some(&expected_authority),
                    )
                    .is_err(),
                    "the live estimator accepted hostile QG-1 stream mutation: {label}"
                );
            };

        let mut no_claim = effect.clone();
        no_claim[0].qg1_sample_binding = None;
        assert_rejected(no_claim, "NoClaim/missing lifecycle binding");

        let mut replayed_receipt = effect.clone();
        replayed_receipt[1].qg1_sample_binding = replayed_receipt[0].qg1_sample_binding.clone();
        assert_rejected(
            replayed_receipt,
            "one lifecycle receipt replayed to another raw row",
        );

        let mut duplicated_receipt = effect.clone();
        let original_receipt_id = duplicated_receipt[0]
            .qg1_sample_binding
            .as_ref()
            .expect("binding")
            .lifecycle_receipt_id_sha256
            .clone();
        duplicated_receipt[1]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .lifecycle_receipt_id_sha256 = original_receipt_id;
        assert_rejected(duplicated_receipt, "duplicate lifecycle receipt identity");

        let mut cloned_fast_pair = effect.clone();
        let fast_pair = cloned_fast_pair
            .iter()
            .filter(|sample| sample.block_id == 0)
            .cloned()
            .collect::<Vec<_>>();
        let suffix_block =
            u64::try_from(PERF_MIN_RUNS - 1).expect("QG-1 hostile suffix block fits u64");
        for sample in cloned_fast_pair
            .iter_mut()
            .filter(|sample| sample.block_id == suffix_block)
        {
            let replacement = fast_pair
                .iter()
                .find(|candidate| candidate.arm == sample.arm)
                .expect("fast source pair has the matching arm");
            sample.started_ns = replacement.started_ns;
            sample.ended_ns = replacement.ended_ns;
            sample.work_units = replacement.work_units;
            sample.byte_count = replacement.byte_count;
            sample.observed_value = replacement.observed_value;
            let target_sequence = sample
                .qg1_sample_binding
                .as_ref()
                .expect("suffix binding")
                .stream_sequence;
            let mut forged_binding = replacement
                .qg1_sample_binding
                .as_ref()
                .expect("fast source binding")
                .clone();
            forged_binding.stream_sequence = target_sequence;
            forged_binding.raw_sample_id = sample.sample_id;
            forged_binding.raw_block_id = sample.block_id;
            forged_binding.raw_arm = sample.arm;
            forged_binding.raw_order = sample.order;
            forged_binding.producer_capability_sha256 = authority
                .issued_row_for(
                    &forged_binding.stream_role,
                    forged_binding.stream_sequence,
                    forged_binding.raw_block_id,
                    forged_binding.raw_sample_id,
                    forged_binding.raw_arm,
                    forged_binding.raw_order,
                )
                .expect("destination suffix slot is issued")
                .producer_capability_sha256
                .clone();
            forged_binding.seal_lifecycle_receipt(&scope, &provenance);
            sample.qg1_sample_binding = Some(forged_binding);
        }
        assert!(
            matches!(
                estimate_paired_experiment(&cloned_fast_pair, &null, &config),
                Err(PairedEstimatorError::InvalidProvenance { .. })
            ),
            "a fully resealed destination-ID clone must not reach any authority-free public estimator"
        );
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &cloned_fast_pair,
                &null,
                &config,
                Some(&expected_authority),
            )
            .is_err(),
            "independent authority must reject exact suffix slots copied from fast rows after full resealing"
        );
        // The forged rows stay publicly self-consistent, so the guard is
        // proved against a result the private estimator body will still
        // produce. Building it here deliberately bypasses the public
        // admission that already refused it above: the screen and decision
        // guard is the second, independent line of defence.
        let forged_live_experiment =
            estimate_paired_experiment_inner(&cloned_fast_pair, &null, &live_config)
                .expect("the public receipt fields remain self-consistent after full resealing");
        assert!(
            !qg1_valid_throughput_experiment(
                &forged_live_experiment,
                None,
                &scope,
                Some(&provenance),
                500,
                64_000,
            ),
            "the shared QG-1 screen and decision guard must consume retained authority, not self-sealed receipt fields"
        );
        assert!(
            !qg1_valid_throughput_experiment(
                &forged_live_experiment,
                Some(&expected_authority),
                &scope,
                Some(&provenance),
                500,
                64_000,
            ),
            "an externally supplied retained authority must reject the resealed clone at the guard too"
        );
        assert!(
            forged_live_experiment
                .verify_recomputed_against_qg1_authority(Some(&expected_authority))
                .is_err(),
            "persisted replay must reject an exact fast-suffix substitution even after complete public resealing"
        );

        let lowered_authority = qg1_test_authority(
            &scope,
            &provenance,
            500,
            64_000,
            9,
            &[
                (QG1_STREAM_ROLE_EFFECT, 0),
                (QG1_STREAM_ROLE_TANTIVY_NULL, 10_000),
            ],
        );
        let mut coordinated_suffix_effect = effect.clone();
        let mut coordinated_suffix_null = null.clone();
        coordinated_suffix_effect.truncate(coordinated_suffix_effect.len() - 2);
        coordinated_suffix_null.truncate(coordinated_suffix_null.len() - 2);
        for sample in coordinated_suffix_effect
            .iter_mut()
            .chain(coordinated_suffix_null.iter_mut())
        {
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.lifecycle_authority_sha256 = lowered_authority.authority_sha256.clone();
            binding.stream_role_identity_sha256 = lowered_authority
                .stream_role_identity_sha256(&binding.stream_role)
                .expect("lowered authority preserves the known role");
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        let mut lowered_min_pairs_config = config.clone();
        lowered_min_pairs_config.min_pairs = 4;
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &coordinated_suffix_effect,
                &coordinated_suffix_null,
                &lowered_min_pairs_config,
                Some(&expected_authority),
            )
            .is_err(),
            "the live estimator accepted both-arm suffix truncation with lowered/resealed row authority counts"
        );

        let mut swapped_sequences = effect.clone();
        let first_sequence = swapped_sequences[0]
            .qg1_sample_binding
            .as_ref()
            .expect("binding")
            .stream_sequence;
        let second_sequence = swapped_sequences[1]
            .qg1_sample_binding
            .as_ref()
            .expect("binding")
            .stream_sequence;
        let (first_row, remaining_rows) = swapped_sequences.split_at_mut(1);
        for (sample, stream_sequence) in [
            (&mut first_row[0], second_sequence),
            (&mut remaining_rows[0], first_sequence),
        ] {
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.stream_sequence = stream_sequence;
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_rejected(
            swapped_sequences,
            "receipt sequences swapped away from canonical raw-order enumeration",
        );

        let mut swapped_raw_orders = effect.clone();
        let first_order = swapped_raw_orders[0]
            .qg1_sample_binding
            .as_ref()
            .expect("binding")
            .raw_order;
        let second_order = swapped_raw_orders[1]
            .qg1_sample_binding
            .as_ref()
            .expect("binding")
            .raw_order;
        let (first_row, remaining_rows) = swapped_raw_orders.split_at_mut(1);
        for (sample, raw_order) in [
            (&mut first_row[0], second_order),
            (&mut remaining_rows[0], first_order),
        ] {
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.raw_order = raw_order;
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_rejected(
            swapped_raw_orders,
            "receipt raw-order coordinates swapped between rows",
        );

        let mut resealed_lowered_count = effect.clone();
        for sample in &mut resealed_lowered_count {
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.lifecycle_authority_sha256 = lowered_authority.authority_sha256.clone();
            binding.stream_role_identity_sha256 = lowered_authority
                .stream_role_identity_sha256(&binding.stream_role)
                .expect("lowered authority preserves the known role");
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_rejected(
            resealed_lowered_count,
            "resealed row-local lowered count authority differs from the fixed estimator authority",
        );

        let mut different_content = effect.clone();
        different_content[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .indexed_content_sha256 = "c".repeat(64);
        assert_rejected(different_content, "prepared content identity");

        let mut incomplete_batches = effect.clone();
        incomplete_batches[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .recorded_batch_count = 2;
        assert_rejected(incomplete_batches, "recorded batch count");

        let mut different_tail = effect.clone();
        different_tail[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .tail_document_id = "synthetic-hostile-tail".to_owned();
        assert_rejected(different_tail, "exact prepared tail");

        let mut detached_endpoint = effect.clone();
        detached_endpoint[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .terminal_endpoint_ns = 1;
        assert_rejected(detached_endpoint, "terminal endpoint");

        let mut rearmed_tantivy = effect.clone();
        rearmed_tantivy[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .lifecycle_witness = Qg1LifecycleWitness::Tantivy {
            searchable_segments_before: 1,
            searchable_segments_after: 1,
            join_elapsed_ns: 1,
            writer_rearmed: true,
        };
        assert_rejected(rearmed_tantivy, "rearmed Tantivy terminal witness");

        let mut zero_searchable_before = effect.clone();
        let binding = zero_searchable_before[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding");
        binding.lifecycle_witness = Qg1LifecycleWitness::Tantivy {
            searchable_segments_before: 0,
            searchable_segments_after: 1,
            join_elapsed_ns: 1,
            writer_rearmed: false,
        };
        binding.seal_lifecycle_receipt(&scope, &provenance);
        assert_rejected(
            zero_searchable_before,
            "zero Tantivy searchable-segments-before witness",
        );

        let mut zero_join_elapsed = effect.clone();
        let binding = zero_join_elapsed[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding");
        binding.lifecycle_witness = Qg1LifecycleWitness::Tantivy {
            searchable_segments_before: 1,
            searchable_segments_after: 1,
            join_elapsed_ns: 0,
            writer_rearmed: false,
        };
        binding.seal_lifecycle_receipt(&scope, &provenance);
        assert_rejected(
            zero_join_elapsed,
            "zero Tantivy terminal-join elapsed witness",
        );

        let mut valid_engine_relabel = effect.clone();
        let relabeled = valid_engine_relabel
            .iter_mut()
            .find(|sample| sample.arm == PerfSampleArm::Treatment)
            .expect("Quill treatment sample");
        let binding = relabeled.qg1_sample_binding.as_mut().expect("binding");
        binding.lifecycle_witness = Qg1LifecycleWitness::Tantivy {
            searchable_segments_before: 1,
            searchable_segments_after: 1,
            join_elapsed_ns: 1,
            writer_rearmed: false,
        };
        binding.seal_lifecycle_receipt(&scope, &provenance);
        assert_rejected(
            valid_engine_relabel,
            "locally valid Tantivy witness relabelled into Quill treatment arm",
        );

        let mut zero_quill_delta = effect.clone();
        let quill_binding = zero_quill_delta
            .iter_mut()
            .find(|sample| sample.arm == PerfSampleArm::Treatment)
            .expect("Quill treatment sample")
            .qg1_sample_binding
            .as_mut()
            .expect("binding");
        quill_binding.lifecycle_witness = Qg1LifecycleWitness::Quill {
            publication_generation_delta: 0,
        };
        quill_binding.seal_lifecycle_receipt(&scope, &provenance);
        assert_rejected(
            zero_quill_delta,
            "zero Quill terminal publication generation delta",
        );

        let mut coordinated_cross_block = effect.clone();
        for sample in &mut coordinated_cross_block {
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.prepared_manifest_sha256 = "c".repeat(64);
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_rejected(
            coordinated_cross_block,
            "coordinated cross-block prepared manifest substitution",
        );

        let mut coordinated_effect = effect.clone();
        let mut coordinated_null = null.clone();
        for sample in coordinated_effect
            .iter_mut()
            .chain(coordinated_null.iter_mut())
        {
            sample.work_units = Some(501);
            sample.byte_count = Some(65_000);
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.prepared_manifest_sha256 = "c".repeat(64);
            binding.indexed_content_sha256 = "d".repeat(64);
            binding.document_count = 501;
            binding.content_bytes = 65_000;
            binding.prepared_batch_count = 1;
            binding.recorded_batch_count = 1;
            binding.batch_coverage = vec![Qg1BatchCoverage {
                document_start: 0,
                document_count: 501,
            }];
            binding.tail_document_id = "synthetic-00000500".to_owned();
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_streams_rejected(
            coordinated_effect,
            coordinated_null,
            "coordinated whole-stream prepared manifest/content/tail/schedule substitution",
        );

        let mut substituted_authority = authority.clone();
        for issued_row in &mut substituted_authority.issued_rows {
            issued_row.block_id = issued_row
                .block_id
                .checked_add(1_000_000)
                .expect("hostile transcript block shift fits u64");
        }
        substituted_authority.authority_sha256 =
            substituted_authority.recomputed_authority_sha256();
        substituted_authority
            .validate()
            .expect("coordinated hostile authority is internally consistent");
        let mut substituted_effect = effect.clone();
        let mut substituted_null = null.clone();
        for sample in substituted_effect
            .iter_mut()
            .chain(substituted_null.iter_mut())
        {
            sample.block_id = sample
                .block_id
                .checked_add(1_000_000)
                .expect("hostile raw block shift fits u64");
            let binding = sample.qg1_sample_binding.as_mut().expect("binding");
            binding.raw_block_id = sample.block_id;
            binding.lifecycle_authority_sha256 = substituted_authority.authority_sha256.clone();
            binding.stream_role_identity_sha256 = substituted_authority
                .stream_role_identity_sha256(&binding.stream_role)
                .expect("substituted authority preserves the known role");
            binding.producer_capability_sha256 = substituted_authority
                .issued_row_for(
                    &binding.stream_role,
                    binding.stream_sequence,
                    binding.raw_block_id,
                    binding.raw_sample_id,
                    binding.raw_arm,
                    binding.raw_order,
                )
                .expect("substituted authority issues the relabelled raw row")
                .producer_capability_sha256
                .clone();
            binding.seal_lifecycle_receipt(&scope, &provenance);
        }
        assert_streams_rejected(
            substituted_effect.clone(),
            substituted_null.clone(),
            "fixed live authority rejects coordinated embedded authority and row transcript substitution",
        );
        let mut substituted_config = config.clone();
        substituted_config.qg1_lifecycle_authority = Some(substituted_authority);
        // Built through the private body on purpose: a coordinated substitution
        // is self-consistent, so the point of the assertions below is that the
        // independently retained authority rejects it anyway.
        let substituted_experiment = estimate_paired_experiment_inner(
            &substituted_effect,
            &substituted_null,
            &substituted_config,
        )
        .expect("coordinated artifact substitution remains self-consistent");
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &substituted_effect,
                &substituted_null,
                &substituted_config,
                Some(&expected_authority),
            )
            .is_err(),
            "the live estimator must reject a self-consistent replacement authority"
        );
        assert!(
            substituted_experiment
                .verify_recomputed_against_qg1_authority(Some(&expected_authority))
                .is_err(),
            "persisted QG-1 verification must compare artifact evidence to its independently retained authority digest"
        );

        let mut noncanonical_effect = effect.clone();
        let mut noncanonical_null = null.clone();
        let noncanonical_scope = PerfOperationScope {
            operation_id: "QG-1.untrusted-prefix.docs_per_second".to_owned(),
            version: 1,
            semantics: PerfMetricSemantics::Throughput,
            unit: "docs/s".to_owned(),
        };
        for sample in noncanonical_effect
            .iter_mut()
            .chain(noncanonical_null.iter_mut())
        {
            sample.scope = noncanonical_scope.clone();
            sample
                .qg1_sample_binding
                .as_mut()
                .expect("binding")
                .seal_lifecycle_receipt(&noncanonical_scope, &provenance);
        }
        assert_streams_rejected(
            noncanonical_effect,
            noncanonical_null,
            "noncanonical QG-1 textual prefix",
        );

        let mut backward_schema = effect.clone();
        let binding = backward_schema[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding");
        binding.schema_version = "frankensearch.quill.qg1-lifecycle-binding.v1".to_owned();
        binding.seal_lifecycle_receipt(&scope, &provenance);
        assert_rejected(backward_schema, "backward QG-1 lifecycle binding schema");

        let mut serialized =
            serde_json::to_value(effect[0].qg1_sample_binding.as_ref().expect("binding"))
                .expect("serialize QG-1 binding");
        serialized
            .as_object_mut()
            .expect("binding encodes as object")
            .insert(
                "unknown_lifecycle_field".to_owned(),
                serde_json::Value::Null,
            );
        assert!(
            serde_json::from_value::<Qg1SampleBinding>(serialized).is_err(),
            "unknown QG-1 lifecycle schema fields must fail before the live estimator"
        );
        let mut raw_with_unknown_field =
            serde_json::to_value(&effect[0]).expect("serialize outer raw sample");
        raw_with_unknown_field
            .as_object_mut()
            .expect("raw sample encodes as object")
            .insert("future_outer_raw_field".to_owned(), serde_json::Value::Null);
        assert!(
            serde_json::from_value::<PerfRawSample>(raw_with_unknown_field).is_err(),
            "outer raw samples must reject unknown fields without a versioned envelope"
        );
        let mut witness = serde_json::to_value(Qg1LifecycleWitness::Quill {
            publication_generation_delta: 1,
        })
        .expect("serialize one typed Quill witness");
        witness
            .as_object_mut()
            .expect("witness encodes as object")
            .insert("join_elapsed_ns".to_owned(), serde_json::Value::from(1_u64));
        assert!(
            serde_json::from_value::<Qg1LifecycleWitness>(witness).is_err(),
            "one QG-1 arm cannot deserialize multiple lifecycle witness variants"
        );

        let mut unsearchable_tantivy = effect;
        unsearchable_tantivy[0]
            .qg1_sample_binding
            .as_mut()
            .expect("binding")
            .lifecycle_witness = Qg1LifecycleWitness::Tantivy {
            searchable_segments_before: 1,
            searchable_segments_after: 0,
            join_elapsed_ns: 1,
            writer_rearmed: false,
        };
        assert_rejected(
            unsearchable_tantivy,
            "unsearchable Tantivy terminal witness",
        );
    }

    /// Traverse the seams a live QG-1 cell actually uses: one producer issues
    /// the effect stream, the oracle null, and the treatment-arm null; the
    /// cell attaches that null, is persisted, reloaded, and replayed against
    /// the expectation its consumer retained. The producer expectation is
    /// never serialized, so replay must receive it from outside the artifact.
    #[test]
    fn qg1_live_cell_attaches_persists_and_replays_against_retained_authority() {
        use crate::perf_evidence::{EvidenceCell, EvidenceCellSpec, EvidencePolicy, EvidenceRole};

        let cell = qg1_bulk_cell(4);
        let scope = qg1_throughput_scope(&cell);
        let provenance = provenance("qg1-live-cell-replay");
        let control = [1_000_000; PERF_MIN_RUNS];
        let mut treatment = [900_000; PERF_MIN_RUNS];
        treatment[PERF_MIN_RUNS - 1] = 940_000;
        let authority = qg1_test_authority(
            &scope,
            &provenance,
            500,
            64_000,
            u64::try_from(PERF_MIN_RUNS).expect("QG-1 cell pair count fits u64"),
            &[
                (QG1_STREAM_ROLE_EFFECT, 0),
                (QG1_STREAM_ROLE_TANTIVY_NULL, 10_000),
                (QG1_STREAM_ROLE_QUILL_NULL, 20_000),
            ],
        );
        let expected_authority = qg1_test_expected_authority(&authority);
        let mut config = estimator_config();
        config.qg1_lifecycle_authority = Some(authority.clone());
        config.qg1_expected_authority = Some(expected_authority.clone());

        let effect = qg1_duration_stream(
            &scope,
            &provenance,
            &control,
            &treatment,
            0,
            500,
            64_000,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_QUILL_ENGINE_ID,
            QG1_STREAM_ROLE_EFFECT,
        );
        let oracle_null = qg1_duration_stream(
            &scope,
            &provenance,
            &control,
            &control,
            10_000,
            500,
            64_000,
            &authority,
            QG1_TANTIVY_ENGINE_ID,
            QG1_TANTIVY_ENGINE_ID,
            QG1_STREAM_ROLE_TANTIVY_NULL,
        );
        let treatment_null = qg1_duration_stream(
            &scope,
            &provenance,
            &treatment,
            &treatment,
            20_000,
            500,
            64_000,
            &authority,
            QG1_QUILL_ENGINE_ID,
            QG1_QUILL_ENGINE_ID,
            QG1_STREAM_ROLE_QUILL_NULL,
        );

        let live_expected = config.qg1_expected_authority.clone();
        let experiment = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &oracle_null,
            &config,
            live_expected.as_ref(),
        )
        .expect("the live cell's A/B stream estimates under its producer authority");
        let treatment_null_experiment = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &treatment_null,
            &config,
            live_expected.as_ref(),
        )
        .expect("the treatment-arm null shares the same producer authority");

        let policy = EvidencePolicy::predeclared();
        // Diagnostic role: a required QG-1 cell additionally demands a
        // concurrency witness, which is orthogonal to the authority seams.
        let spec = EvidenceCellSpec {
            gate: PerfGate::Qg1,
            fixture: cell.fixture.clone(),
            metric: cell.metric.clone(),
            unit: "docs/s".to_owned(),
            role: EvidenceRole::Diagnostic,
            input_identity: None,
            qg6_semantic_contract: None,
            cold_cache: None,
            concurrency_witness: None,
        };
        let mut live_cell = EvidenceCell::evaluate(spec, experiment, &policy)
            .expect("a QG-1 paired cell evaluates from authority-bearing evidence");

        // Planted negative: the authority-free attach is a typed reject, and
        // the real cell no longer panics when the producer hands its retained
        // expectation to the same seam.
        assert!(
            live_cell
                .clone()
                .attach_treatment_arm_null(treatment_null_experiment.clone(), &policy)
                .is_err(),
            "authority-free attachment must keep refusing an embedded QG-1 authority"
        );
        live_cell
            .attach_treatment_arm_null_against_qg1_authority(
                treatment_null_experiment.clone(),
                &policy,
                Some(&expected_authority),
            )
            .expect("the live QG-1 treatment-arm null attaches under its retained authority");
        live_cell
            .verify_recomputed_against_qg1_authorities(&policy, &[&expected_authority])
            .expect("the live cell recomputes against the retained authority");
        assert!(
            live_cell.verify_recomputed(&policy).is_err(),
            "a QG-1 cell must never authenticate itself without an external authority"
        );

        let encoded = serde_json::to_string(&live_cell).expect("QG-1 evidence cell serializes");
        assert!(
            !encoded.contains("capability_preimage") && !encoded.contains("expected_authority"),
            "producer capability preimages must never reach a persisted artifact"
        );
        let reloaded: EvidenceCell =
            serde_json::from_str(&encoded).expect("QG-1 evidence cell reloads");
        reloaded
            .verify_recomputed_against_qg1_authorities(&policy, &[&expected_authority])
            .expect("reloaded QG-1 evidence replays against the externally retained authority");
        assert!(
            reloaded
                .verify_recomputed_against_qg1_authorities(&policy, &[])
                .is_err(),
            "removing the external authority must be a typed reject, never a green replay"
        );

        // A destination-ID resealed suffix row is rejected at every seam that
        // consumes the retained authority: estimate, attach, and replay.
        let mut forged_null = treatment_null.clone();
        let source_binding = forged_null[0]
            .qg1_sample_binding
            .clone()
            .expect("the source row retains its issued binding");
        let destination = forged_null
            .last()
            .expect("the treatment-arm null retains a suffix row")
            .clone();
        let mut forged_binding = source_binding;
        forged_binding.stream_sequence = destination
            .qg1_sample_binding
            .as_ref()
            .expect("the destination row retains its issued binding")
            .stream_sequence;
        forged_binding.raw_sample_id = destination.sample_id;
        forged_binding.raw_block_id = destination.block_id;
        forged_binding.raw_arm = destination.arm;
        forged_binding.raw_order = destination.order;
        forged_binding.producer_capability_sha256 = authority
            .issued_row_for(
                &forged_binding.stream_role,
                forged_binding.stream_sequence,
                forged_binding.raw_block_id,
                forged_binding.raw_sample_id,
                forged_binding.raw_arm,
                forged_binding.raw_order,
            )
            .expect("the destination suffix slot is issued")
            .producer_capability_sha256
            .clone();
        forged_binding.seal_lifecycle_receipt(&scope, &provenance);
        let forged_index = forged_null.len() - 1;
        forged_null[forged_index].qg1_sample_binding = Some(forged_binding);
        let forged_experiment = estimate_paired_experiment_inner(&effect, &forged_null, &config)
            .expect("the forged null stays publicly self-consistent after resealing");
        assert!(
            live_cell
                .clone()
                .attach_treatment_arm_null_against_qg1_authority(
                    forged_experiment.clone(),
                    &policy,
                    Some(&expected_authority),
                )
                .is_err(),
            "the attach seam must reject a destination-ID resealed suffix row"
        );
        assert!(
            forged_experiment
                .verify_recomputed_against_qg1_authority(Some(&expected_authority))
                .is_err(),
            "the replay seam must reject a destination-ID resealed suffix row"
        );
    }

    #[test]
    fn qg1_lifecycle_authority_installation_is_single_assignment() {
        let cell = qg1_bulk_cell(4);
        let scope = qg1_throughput_scope(&cell);
        let provenance = provenance("qg1-single-assignment");
        let schedule =
            seeded_balanced_pair_order(PERF_MIN_RUNS, 0x00dd_5eed).expect("QG-1 frozen schedule");
        let issued_streams = || {
            vec![
                (QG1_STREAM_ROLE_EFFECT.to_owned(), 0, 0, schedule.clone()),
                (
                    QG1_STREAM_ROLE_TANTIVY_NULL.to_owned(),
                    0,
                    10_000,
                    schedule.clone(),
                ),
            ]
        };
        let mut config = estimator_config();
        let producer = config
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                "a".repeat(64),
                "b".repeat(64),
                500,
                64_000,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: 500,
                }],
                "synthetic-00000499".to_owned(),
                u64::try_from(PERF_MIN_RUNS).expect("QG-1 pair count fits u64"),
                issued_streams(),
            )
            .expect("first pre-timing lifecycle authority install");
        let frozen_digest = producer.expected_authority().digest().to_owned();
        assert!(matches!(
            config.install_qg1_lifecycle_authority(
                scope,
                provenance.corpus_sha256.clone(),
                "a".repeat(64),
                "b".repeat(64),
                500,
                64_000,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: 500,
                }],
                "synthetic-00000499".to_owned(),
                u64::try_from(PERF_MIN_RUNS).expect("QG-1 pair count fits u64"),
                issued_streams(),
            ),
            Err(PairedEstimatorError::InvalidConfig { .. })
        ));
        assert_eq!(
            config
                .qg1_lifecycle_authority
                .as_ref()
                .map(|authority| authority.authority_sha256.as_str()),
            Some(frozen_digest.as_str()),
            "a rejected overwrite must leave the original pre-timing authority frozen"
        );
    }

    #[test]
    fn qg1_incumbent_preregistration_binds_machine_mode_and_keeps_budget_separate_from_widths() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let candidates =
            preregister_qg1_tantivy_incumbents(&cell, &screen_plan, &semantic_contract)
                .expect("candidates");

        assert_eq!(screen_plan.external_cpu_budget, 2);
        assert_eq!(
            screen_plan.work_units,
            cell.document_count.expect("bulk docs")
        );
        assert_eq!(screen_plan.content_bytes, 64_000);
        assert_eq!(
            screen_plan.work_contract_sha256,
            qg1_bulk_work_contract_sha256(
                &cell,
                screen_plan.work_units,
                screen_plan.content_bytes,
            )
            .expect("work contract")
        );
        assert_eq!(candidates.len(), 4);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.writer_mode.stable_id())
                .collect::<Vec<_>>(),
            vec!["shipping_auto", "fixed_1", "fixed_2", "fixed_4"]
        );
        assert!(candidates.iter().all(|candidate| {
            candidate.screen_plan_sha256 == screen_plan.plan_sha256
                && candidate.cell_contract_sha256 == cell.contract_sha256().expect("cell hash")
                && candidate.semantic_contract == semantic_contract
                && candidate.config_sha256
                    == candidate.recomputed_config_sha256().expect("config hash")
        }));
    }

    /// A reloaded screen and decision have lost the never-serialized producer
    /// expectation their live configuration carried. They must therefore fail
    /// closed on the authority-free entries and be admitted only through the
    /// entries their consumer supplies the retained expectation to.
    #[test]
    fn qg1_reloaded_screen_and_decision_require_an_externally_supplied_authority() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let mut pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let pilot_authority = qg1_test_expected_authority(
            pilots[0]
                .experiment
                .config
                .qg1_lifecycle_authority
                .as_ref()
                .expect("live pilots carry their sealed lifecycle authority"),
        );
        let live_screen = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan.clone(),
            &semantic_contract,
            pilots.clone(),
        )
        .expect("live pilots screen from their own producer expectation");
        let selected = live_screen
            .selected_candidate
            .as_ref()
            .expect("CI-distinct fastest candidate");
        let mut decision = qg1_decision(
            &cell,
            &live_screen.screen_plan,
            &semantic_contract,
            selected,
            "one-live-invocation",
        );

        // Pilot and decision streams are issued by separate producers, so the
        // consumer retains both expectations and each seam selects its own.
        let decision_authority = qg1_test_expected_authority(
            decision
                .estimator_config
                .qg1_lifecycle_authority
                .as_ref()
                .expect("the live decision carries its sealed lifecycle authority"),
        );
        let retained = [&pilot_authority, &decision_authority];

        // Reload drops exactly the field serde never wrote.
        for pilot in &mut pilots {
            pilot.experiment.config.qg1_expected_authority = None;
        }
        decision.estimator_config.qg1_expected_authority = None;
        for stream in [
            &decision.tantivy_vs_quill,
            &decision.tantivy_null,
            &decision.quill_null,
        ] {
            assert!(
                stream
                    .samples
                    .iter()
                    .all(|sample| sample.qg1_sample_binding.is_some()),
                "every reloaded decision row keeps its serialized lifecycle receipt"
            );
        }

        let reloaded_without_authority = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan.clone(),
            &semantic_contract,
            pilots.clone(),
        )
        .expect("a reloaded screen is a typed no-decision receipt, never a panic");
        assert!(
            reloaded_without_authority.selected_candidate.is_none(),
            "reloaded QG-1 pilots must not select an arm without their retained authority"
        );
        let reloaded_screen = Qg1TantivyIncumbentScreen::screen_against_qg1_authorities(
            &cell,
            screen_plan,
            &semantic_contract,
            pilots,
            &retained,
        )
        .expect("reloaded pilots screen against the externally supplied authority");
        assert!(
            reloaded_screen.selected_candidate.is_some(),
            "the externally supplied retained authority admits reloaded QG-1 pilots"
        );
        assert!(
            reloaded_screen
                .validate_decision(&cell, &semantic_contract, &decision)
                .is_err(),
            "the authority-free decision entry must keep refusing reloaded QG-1 evidence"
        );
        reloaded_screen
            .validate_decision_against_qg1_authorities(
                &cell,
                &semantic_contract,
                &decision,
                &retained,
            )
            .expect("reloaded decision validates against the externally supplied authority");
    }

    #[test]
    fn qg1_incumbent_screen_selects_a_ci_distinct_arm_and_binds_all_three_decision_controls() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let screen =
            Qg1TantivyIncumbentScreen::screen(&cell, screen_plan, &semantic_contract, pilots)
                .expect("valid incumbent screen");
        let selected = screen
            .selected_candidate
            .as_ref()
            .expect("CI-distinct fastest candidate");
        assert_eq!(
            selected.writer_mode,
            Qg1TantivyWriterMode::Fixed { writer_threads: 4 }
        );
        assert_eq!(screen.tied_fastest_candidates, vec![selected.clone()]);

        let decision = qg1_decision(
            &cell,
            &screen.screen_plan,
            &semantic_contract,
            selected,
            "one-live-invocation",
        );
        screen
            .validate_decision(&cell, &semantic_contract, &decision)
            .expect("same-invocation T/Quill, T/T, and Q/Q streams bind selected candidate");

        let mut relabeled = decision.clone();
        relabeled.tantivy_vs_quill.control_engine_id = QG1_QUILL_ENGINE_ID.to_owned();
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &relabeled),
            Err(Qg1TantivyIncumbentError::StreamReceiptMismatch)
        );

        relabeled.tantivy_vs_quill.stream_receipt_sha256 = relabeled
            .tantivy_vs_quill
            .recomputed_stream_receipt_sha256()
            .expect("relabeled receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &relabeled),
            Err(Qg1TantivyIncumbentError::DecisionCandidateMismatch)
        );

        let mut wrong_quill_config = decision.clone();
        wrong_quill_config
            .tantivy_vs_quill
            .treatment_engine_config_sha256 = "2".repeat(64);
        wrong_quill_config.tantivy_vs_quill.stream_receipt_sha256 = wrong_quill_config
            .tantivy_vs_quill
            .recomputed_stream_receipt_sha256()
            .expect("wrong-Quill-config receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &wrong_quill_config),
            Err(Qg1TantivyIncumbentError::DecisionCandidateMismatch)
        );

        let mut mislabeled_null = decision.clone();
        mislabeled_null.tantivy_vs_quill.kind = Qg1TantivyDecisionStreamKind::TantivyNull;
        mislabeled_null.tantivy_vs_quill.stream_receipt_sha256 = mislabeled_null
            .tantivy_vs_quill
            .recomputed_stream_receipt_sha256()
            .expect("mislabeled receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &mislabeled_null),
            Err(Qg1TantivyIncumbentError::DecisionCandidateMismatch)
        );

        let mut unequal_tantivy_null = decision.clone();
        for sample in &mut unequal_tantivy_null.tantivy_null.samples {
            if sample.arm == PerfSampleArm::Treatment {
                sample.ended_ns = sample.started_ns + 500_000;
            }
        }
        unequal_tantivy_null.tantivy_null.stream_receipt_sha256 = unequal_tantivy_null
            .tantivy_null
            .recomputed_stream_receipt_sha256()
            .expect("unequal A/A receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &unequal_tantivy_null),
            Err(Qg1TantivyIncumbentError::ObservationBindingMismatch)
        );

        let mut forged_screen = screen.clone();
        forged_screen.selected_candidate = Some(forged_screen.candidates[0].clone());
        assert_eq!(
            forged_screen.validate_decision(&cell, &semantic_contract, &decision),
            Err(Qg1TantivyIncumbentError::ScreenSelectionMismatch)
        );
    }

    #[test]
    fn qg1_incumbent_screen_rejects_cross_invocation_and_unrelated_throughput_scopes() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let screen = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan.clone(),
            &semantic_contract,
            pilots,
        )
        .expect("valid incumbent screen");
        let selected = screen
            .selected_candidate
            .as_ref()
            .expect("CI-distinct fastest candidate");
        let decision = qg1_decision(
            &cell,
            &screen.screen_plan,
            &semantic_contract,
            selected,
            "one-live-invocation",
        );

        let mut cross_invocation = decision.clone();
        for sample in &mut cross_invocation.tantivy_vs_quill.samples {
            sample.provenance = provenance("later-process-invocation");
        }
        cross_invocation.tantivy_vs_quill.stream_receipt_sha256 = cross_invocation
            .tantivy_vs_quill
            .recomputed_stream_receipt_sha256()
            .expect("cross-invocation receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &cross_invocation),
            Err(Qg1TantivyIncumbentError::DecisionInvocationMismatch)
        );

        let mut wrong_scope = qg1_throughput_scope(&cell);
        wrong_scope.operation_id = "qg1.unrelated_throughput".to_owned();
        let mut unrelated_tantivy_vs_quill = decision.clone();
        qg1_rebind_stream_scope(
            &mut unrelated_tantivy_vs_quill.tantivy_vs_quill,
            &wrong_scope,
        );
        let mut unrelated_tantivy_null = decision.clone();
        qg1_rebind_stream_scope(&mut unrelated_tantivy_null.tantivy_null, &wrong_scope);
        let mut unrelated_quill_null = decision.clone();
        qg1_rebind_stream_scope(&mut unrelated_quill_null.quill_null, &wrong_scope);
        for unrelated_scope in [
            unrelated_tantivy_vs_quill,
            unrelated_tantivy_null,
            unrelated_quill_null,
        ] {
            assert_eq!(
                screen.validate_decision(&cell, &semantic_contract, &unrelated_scope),
                Err(Qg1TantivyIncumbentError::DecisionInvocationMismatch)
            );
        }

        let mut wrong_scope_pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let pilot = &mut wrong_scope_pilots[0];
        let (effect_samples, null_samples) = (
            &mut pilot.experiment.effect_samples,
            &mut pilot.experiment.null_samples,
        );
        for sample in effect_samples.iter_mut().chain(null_samples.iter_mut()) {
            sample.scope = wrong_scope.clone();
        }
        pilot.experiment = estimate_paired_experiment(
            &pilot.experiment.effect_samples,
            &pilot.experiment.null_samples,
            &estimator_config(),
        )
        .expect("unrelated scope remains generically valid throughput evidence");
        pilot.stream_receipt_sha256 = pilot
            .recomputed_stream_receipt_sha256()
            .expect("unrelated-scope pilot receipt");
        let rejected_pilots = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan,
            &semantic_contract,
            wrong_scope_pilots,
        )
        .expect("wrong-scope pilot is a fail-closed no-decision receipt");
        assert!(rejected_pilots.selected_candidate.is_none());
        assert_eq!(
            rejected_pilots.no_decision_reason.as_deref(),
            Some("candidate pilot lacks valid configuration-bound throughput evidence")
        );
    }

    #[test]
    fn qg1_incumbent_screen_binds_denominators_estimator_and_immutable_observations() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );

        let mut wrong_work_pilots = pilots.clone();
        let wrong_work_pilot = &mut wrong_work_pilots[0];
        let (effect_samples, null_samples) = (
            &mut wrong_work_pilot.experiment.effect_samples,
            &mut wrong_work_pilot.experiment.null_samples,
        );
        for sample in effect_samples.iter_mut().chain(null_samples.iter_mut()) {
            sample.work_units = Some(screen_plan.work_units - 1);
        }
        // Deliberately degraded fixture: the samples keep their canonical QG-1
        // scope, which the public estimator now refuses outright, so the
        // private body builds the pilot the screen must still reject.
        wrong_work_pilot.experiment = estimate_paired_experiment_inner(
            &wrong_work_pilot.experiment.effect_samples,
            &wrong_work_pilot.experiment.null_samples,
            &estimator_config(),
        )
        .expect("wrong denominator remains generically estimable");
        wrong_work_pilot.stream_receipt_sha256 = wrong_work_pilot
            .recomputed_stream_receipt_sha256()
            .expect("wrong-work pilot receipt");
        let wrong_work = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan.clone(),
            &semantic_contract,
            wrong_work_pilots,
        )
        .expect("wrong QG-1 denominator is a fail-closed no-decision receipt");
        assert!(wrong_work.selected_candidate.is_none());

        let mut mixed_estimator_pilots = pilots.clone();
        let alternate_config = PairedEstimatorConfig::predeclared(0xa11c_e55e);
        mixed_estimator_pilots[1].experiment = estimate_paired_experiment_inner(
            &mixed_estimator_pilots[1].experiment.effect_samples,
            &mixed_estimator_pilots[1].experiment.null_samples,
            &alternate_config,
        )
        .expect("alternate predeclared estimator remains structurally valid");
        mixed_estimator_pilots[1].stream_receipt_sha256 = mixed_estimator_pilots[1]
            .recomputed_stream_receipt_sha256()
            .expect("alternate-estimator pilot receipt");
        assert_eq!(
            Qg1TantivyIncumbentScreen::screen(
                &cell,
                screen_plan.clone(),
                &semantic_contract,
                mixed_estimator_pilots,
            ),
            Err(Qg1TantivyIncumbentError::EstimatorConfigMismatch)
        );

        let mut reused_pilot_observation = pilots.clone();
        let reused_pilot_id = reused_pilot_observation[0].effect_observations[0]
            .observation_id_sha256
            .clone();
        reused_pilot_observation[1].effect_observations[0].observation_id_sha256 = reused_pilot_id;
        reused_pilot_observation[1].stream_receipt_sha256 = reused_pilot_observation[1]
            .recomputed_stream_receipt_sha256()
            .expect("reused-observation pilot receipt");
        assert_eq!(
            Qg1TantivyIncumbentScreen::screen(
                &cell,
                screen_plan.clone(),
                &semantic_contract,
                reused_pilot_observation,
            ),
            Err(Qg1TantivyIncumbentError::ObservationReuse)
        );

        let screen =
            Qg1TantivyIncumbentScreen::screen(&cell, screen_plan, &semantic_contract, pilots)
                .expect("valid incumbent screen");
        let selected = screen
            .selected_candidate
            .as_ref()
            .expect("CI-distinct fastest candidate");
        let decision = qg1_decision(
            &cell,
            &screen.screen_plan,
            &semantic_contract,
            selected,
            "one-live-invocation",
        );

        let mut wrong_bytes = decision.clone();
        for sample in &mut wrong_bytes.tantivy_null.samples {
            sample.byte_count = Some(screen.screen_plan.content_bytes - 1);
        }
        let observation_ids = wrong_bytes
            .tantivy_null
            .observations
            .iter()
            .map(|observation| observation.observation_id_sha256.clone())
            .collect();
        wrong_bytes.tantivy_null.observations = qg1_bind_raw_observations(
            &wrong_bytes.tantivy_null.samples,
            &wrong_bytes.tantivy_null.control_engine_id,
            &wrong_bytes.tantivy_null.control_engine_config_sha256,
            &wrong_bytes.tantivy_null.treatment_engine_id,
            &wrong_bytes.tantivy_null.treatment_engine_config_sha256,
            observation_ids,
        )
        .expect("reseal wrong-byte observations");
        wrong_bytes.tantivy_null.stream_receipt_sha256 = wrong_bytes
            .tantivy_null
            .recomputed_stream_receipt_sha256()
            .expect("wrong-byte stream receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &wrong_bytes),
            Err(Qg1TantivyIncumbentError::ObservationBindingMismatch)
        );

        let mut reused_decision_observation = decision;
        reused_decision_observation.quill_null.observations[0].observation_id_sha256 =
            screen.pilots[0].effect_observations[0]
                .observation_id_sha256
                .clone();
        reused_decision_observation.quill_null.stream_receipt_sha256 = reused_decision_observation
            .quill_null
            .recomputed_stream_receipt_sha256()
            .expect("reused decision-observation receipt");
        assert_eq!(
            screen.validate_decision(&cell, &semantic_contract, &reused_decision_observation),
            Err(Qg1TantivyIncumbentError::ObservationReuse)
        );
    }

    #[test]
    fn qg1_incumbent_screen_reports_ci_ties_without_a_unique_winner() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let mut pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let shipping_auto = pilots[0].candidate.clone();
        let scope = qg1_throughput_scope(&cell);
        pilots[3] = qg1_pilot(
            pilots[3].candidate.clone(),
            &shipping_auto,
            &scope,
            screen_plan.work_units,
            screen_plan.content_bytes,
            "one-live-invocation",
            500_000,
        );
        pilots[2] = qg1_pilot(
            pilots[2].candidate.clone(),
            &shipping_auto,
            &scope,
            screen_plan.work_units,
            screen_plan.content_bytes,
            "one-live-invocation",
            500_000,
        );
        let screen =
            Qg1TantivyIncumbentScreen::screen(&cell, screen_plan, &semantic_contract, pilots)
                .expect("tied screen remains replayable");
        assert!(screen.selected_candidate.is_none());
        assert_eq!(screen.tied_fastest_candidates.len(), 2);
        assert_eq!(
            screen.no_decision_reason.as_deref(),
            Some(
                "fastest candidate is tied within the predeclared 95% median confidence intervals"
            )
        );
    }

    #[test]
    fn qg1_incumbent_screen_fails_closed_for_changed_semantics_or_infeasible_materialization() {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let mut different_merge_semantics = semantic_contract.clone();
        different_merge_semantics.merge_policy_sha256 = "1".repeat(64);
        let mismatch_pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &different_merge_semantics,
            "one-live-invocation",
        );
        assert_eq!(
            Qg1TantivyIncumbentScreen::screen(
                &cell,
                screen_plan.clone(),
                &semantic_contract,
                mismatch_pilots,
            ),
            Err(Qg1TantivyIncumbentError::CandidateContractMismatch)
        );

        let mut infeasible_pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        infeasible_pilots[0].observed_writer_threads = 5;
        let screen = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan,
            &semantic_contract,
            infeasible_pilots,
        )
        .expect("infeasible materialization is a no-decision receipt");
        assert!(screen.selected_candidate.is_none());
        assert_eq!(
            screen.no_decision_reason.as_deref(),
            Some("candidate materialized an infeasible writer width")
        );
    }

    #[test]
    fn qg1_incumbent_screen_rejects_incomplete_and_cross_invocation_pilots_without_cross_stream_sample_ids()
     {
        let cell = qg1_bulk_cell(4);
        let semantic_contract = qg1_semantic_contract();
        let screen_plan = qg1_screen_plan(&cell, vec![1, 2, 4]);
        let mut pilots = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        pilots.pop();
        let incomplete = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan.clone(),
            &semantic_contract,
            pilots,
        )
        .expect("missing candidate is an explicit no-decision");
        assert!(incomplete.selected_candidate.is_none());
        assert!(
            incomplete
                .no_decision_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("expected 4 preregistered pilots"))
        );

        let mut split_invocation = qg1_complete_pilots(
            &cell,
            &screen_plan,
            &semantic_contract,
            "one-live-invocation",
        );
        let candidates =
            preregister_qg1_tantivy_incumbents(&cell, &screen_plan, &semantic_contract)
                .expect("candidates");
        let scope = qg1_throughput_scope(&cell);
        split_invocation[1] = qg1_pilot(
            candidates[1].clone(),
            &candidates[0],
            &scope,
            screen_plan.work_units,
            screen_plan.content_bytes,
            "different-invocation",
            500_000,
        );
        let screen = Qg1TantivyIncumbentScreen::screen(
            &cell,
            screen_plan,
            &semantic_contract,
            split_invocation,
        )
        .expect("cross-invocation pilot is an explicit no-decision");
        assert!(screen.selected_candidate.is_none());
        assert_eq!(
            screen.no_decision_reason.as_deref(),
            Some("candidate pilots used different process invocations or semantic identities")
        );
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
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
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
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
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
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
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
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
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
    fn drifting_effect_stream_is_invalid_even_when_null_is_quiet() {
        // bd-yo5by: a quiet null must not bless a drifting effect stream.
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("drifting-effect");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let mut effect_treatments = [500_000; PERF_MIN_RUNS];
        effect_treatments[PERF_MIN_RUNS / 2..].fill(900_000);
        let effect = duration_stream(&scope, &provenance, &controls, &effect_treatments, 0);
        let null = stable_null(&scope, &provenance);
        let result =
            estimate_paired_experiment(&effect, &null, &estimator_config()).expect("diagnostic");
        assert_eq!(result.status, PairedEvidenceStatus::InvalidExperiment);
        assert_eq!(result.claim_state, PairedClaimState::NoDecision);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "paired.effect_drift")
        );
        result
            .verify_recomputed()
            .expect("invalid diagnostics still recompute");
    }

    #[test]
    fn effect_order_effect_beyond_bound_is_invalid() {
        // bd-yo5by: carryover asymmetry in the A/B stream is gated with the
        // same predeclared bound as the A/A stream.
        let scope = operation_scope(PerfMetricSemantics::Throughput);
        let provenance = provenance("effect-order-effect");
        let controls = [1_000_000; PERF_MIN_RUNS];
        let treatments = [800_000; PERF_MIN_RUNS];
        let mut effect = duration_stream(&scope, &provenance, &controls, &treatments, 0);
        // Bias every sample by execution order: first-position samples run 4x
        // faster than second-position samples, an order effect far beyond
        // ln(1.05) while per-arm medians stay balanced across orders.
        for sample in &mut effect {
            let duration = sample.ended_ns - sample.started_ns;
            let biased = if sample.order == PerfSampleOrder::First {
                duration / 4
            } else {
                duration * 4
            };
            sample.ended_ns = sample.started_ns + biased;
        }
        let null = stable_null(&scope, &provenance);
        let result =
            estimate_paired_experiment(&effect, &null, &estimator_config()).expect("diagnostic");
        assert_eq!(result.status, PairedEvidenceStatus::InvalidExperiment);
        assert_eq!(result.claim_state, PairedClaimState::NoDecision);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "paired.effect_order_effect")
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
        // Eight blocks at an identical 0.9 per-block ratio plus two extreme
        // blocks (tiny control, enormous treatment). The paired median stays
        // 0.9 (negative log) while the marginal arm-median ratio flips above
        // 1.0 — the direction conflict this test exists for. Because a median
        // over any 5-block half or order subset contains at most two extreme
        // blocks, every drift/order-effect median stays at 0.9, keeping the
        // bd-yo5by effect-stream drift and order-effect gates quiet (the
        // previous fixture drifted between halves and now correctly
        // classifies as InvalidExperiment before the contradiction check).
        let controls = [
            100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 1.0, 2.0,
        ];
        let treatments = [
            90.0, 99.0, 108.0, 117.0, 126.0, 135.0, 144.0, 153.0, 100_000.0, 200_000.0,
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
    fn qg5_matrix_is_repinned_to_xlarge_after_generator_landing() {
        let matrix = PerfMatrixSpec::complete();
        let qg5 = matrix.for_gate(PerfGate::Qg5);
        assert_eq!(qg5.len(), 3);

        for (cell, density) in qg5.into_iter().zip([5, 20, 50]) {
            assert_eq!(cell.fixture, format!("compaction/xlarge/{density}pct"));
            assert_eq!(cell.corpus, Some(PerfCorpus::Xlarge));
            assert_eq!(
                cell.document_count,
                Some(PerfCorpus::Xlarge.document_count())
            );
        }
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
        // GOLDEN-CHANGE (QG-5 xlarge re-pin, 02b5ec25): a plan identity binds
        // the normalized normative manifest, so re-pinning QG-5 from medium to
        // xlarge advances all three profile plan hashes. Every structural
        // assertion below is unchanged and still passes -- 74 cells, the same
        // Required/Diagnostic/NotApplicable split per profile, the same
        // capacities and primary target width -- which is what distinguishes a
        // manifest-identity advance from a matrix change. That commit did not
        // re-freeze here, and this gate stayed red on the trunk until bd-916qm.
        // The three hashes are the values the frozen registry and manifest now
        // recompute to; they are read from the plan, never chosen.
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let cases = [
            (
                ExecutionProfileId::Physical64,
                56,
                2,
                16,
                Some(64),
                Some(64),
                "e59deedec99d6d7d8ce7d7c53d2627997fefece864b500a5362c6b301d7a14c3",
            ),
            (
                ExecutionProfileId::Smt2_128,
                72,
                2,
                0,
                Some(128),
                Some(128),
                "13ad902a710627eaee8cf8a1a4fb3674e73322c431c7fd7c8848059c2024c89b",
            ),
            (
                ExecutionProfileId::Scheduler10,
                32,
                2,
                40,
                Some(10),
                Some(8),
                "cb90d7d60f5f9a25bd36510121828d934a5a61cc5dc438057b7685a349079ae4",
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
    fn the_applied_qg2_contract_block_is_admitted_by_its_live_manifest_consumer() {
        // The correction is only real if the live consumer accepts the applied
        // manifest. Before this admission the projected block made
        // perf_gate_manifest_identity fail with "unexpected field qg2_contract",
        // so applying the contract would have broken planning for every gate.
        let bootstrap = PERF_MANIFEST;
        assert_eq!(
            bootstrap
                .matches(crate::qg2_contract::QG2_MANIFEST_BLOCK_PRE_REGION)
                .count(),
            1,
            "the live manifest must still carry the exact protected QG-2 block"
        );
        let applied = bootstrap.replacen(
            crate::qg2_contract::QG2_MANIFEST_BLOCK_PRE_REGION,
            crate::qg2_contract::QG2_MANIFEST_BLOCK_POST_REGION,
            1,
        );

        for gate in [PerfGate::Qg1, PerfGate::Qg2] {
            perf_gate_manifest_identity(&applied, gate)
                .expect("the applied QG-2 contract block must be admitted by the live consumer");
        }
        assert_ne!(
            perf_manifest_contract_sha256(bootstrap),
            perf_manifest_contract_sha256(&applied),
            "applying the contract must move the normalized manifest digest"
        );
    }

    #[test]
    fn the_qg2_contract_table_is_admitted_only_as_the_exact_canonical_contract() {
        let applied = PERF_MANIFEST.replacen(
            crate::qg2_contract::QG2_MANIFEST_BLOCK_PRE_REGION,
            crate::qg2_contract::QG2_MANIFEST_BLOCK_POST_REGION,
            1,
        );

        // Altered prose still parses as the typed contract, so only value
        // equality catches it. Widening the allowlist without this check would
        // let any string ride into an otherwise closed manifest.
        let altered = applied.replacen(
            "BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 compares",
            "BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 sometimes compares",
            1,
        );
        assert_ne!(altered, applied, "the alteration mutation must apply");
        let altered = perf_gate_manifest_identity(&altered, PerfGate::Qg2)
            .expect_err("an altered contract clause must be rejected");
        assert!(
            altered
                .to_string()
                .contains("is not the canonical Q2C comparator contract"),
            "unexpected altered-contract error: {altered}"
        );

        // An unknown key inside the table is a closed-shape violation.
        let unknown_key = applied.replacen(
            "storage_topology = \"symmetric_in_memory\"",
            "storage_topology = \"symmetric_in_memory\"\nextra_scope = \"durable\"",
            1,
        );
        let unknown_key = perf_gate_manifest_identity(&unknown_key, PerfGate::Qg2)
            .expect_err("an unknown contract field must be rejected");
        assert!(
            unknown_key
                .to_string()
                .contains("is not the closed typed comparator contract"),
            "unexpected unknown-field error: {unknown_key}"
        );

        // The admission is QG-2 only: the same table under another gate stays
        // an unexpected field.
        let foreign_gate =
            applied.replacen("[gate.QG-2.qg2_contract]", "[gate.QG-3.qg2_contract]", 1);
        let foreign_gate = perf_gate_manifest_identity(&foreign_gate, PerfGate::Qg1)
            .expect_err("the contract table must not be admitted under another gate");
        assert!(
            foreign_gate
                .to_string()
                .contains("manifest gate.QG-3 defines unexpected field qg2_contract"),
            "unexpected foreign-gate error: {foreign_gate}"
        );
    }

    #[test]
    fn qg1_manifest_contract_rejects_missing_or_unbounded_primary_target() {
        let missing_unrelated_gate = PERF_MANIFEST.replacen("[gate.QG-10]", "[omitted.QG-10]", 1);
        let missing_unrelated_gate =
            perf_gate_manifest_identity(&missing_unrelated_gate, PerfGate::Qg1).expect_err(
                "QG-1 planning must reject a manifest missing an unrelated normative gate",
            );
        assert!(
            missing_unrelated_gate
                .to_string()
                .contains("manifest gate.QG-10 is missing or not a table"),
            "unexpected missing-gate error: {missing_unrelated_gate}"
        );

        let extra_gate = format!("{PERF_MANIFEST}\n[gate.QG-11]\nactivated = false\n");
        let extra_gate = perf_gate_manifest_identity(&extra_gate, PerfGate::Qg1)
            .expect_err("QG-1 planning must reject an unexpected normative gate");
        assert!(
            extra_gate
                .to_string()
                .contains("manifest defines unexpected gate.QG-11"),
            "unexpected extra-gate error: {extra_gate}"
        );

        let missing_unrelated_target = PERF_MANIFEST.replacen(
            "target = \"open() <= 50ms (manifest + lazy sections) vs oracle reader open\"",
            "target = \"\"",
            1,
        );
        let missing_unrelated_target =
            perf_gate_manifest_identity(&missing_unrelated_target, PerfGate::Qg1).expect_err(
                "QG-1 planning must reject an unrelated gate with an empty required field",
            );
        assert!(
            missing_unrelated_target
                .to_string()
                .contains("manifest gate.QG-9.target is missing or empty"),
            "unexpected empty-target error: {missing_unrelated_target}"
        );

        for (field, expected) in [
            ("threshold_artifact", PERF_ARTIFACT_SCHEMA_VERSION),
            ("evidence_artifact", PERF_EVIDENCE_SCHEMA_VERSION),
            ("evidence_assembly", PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION),
            ("history_pointer", PERF_HISTORY_POINTER_SCHEMA_VERSION),
            ("machine_registry", MACHINE_CLASS_REGISTRY_SCHEMA_VERSION),
            ("runner_completion_receipt", RUNNER_RECEIPT_SCHEMA_VERSION),
            (
                "runner_artifact_manifest",
                RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            ),
            (
                "local_producer_contract",
                LOCAL_PERF_PRODUCER_CONTRACT_VERSION,
            ),
            (
                "runner_attempt_receipt",
                LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION,
            ),
            (
                "runner_lease_release_receipt",
                LOCAL_PERF_LEASE_RELEASE_RECEIPT_SCHEMA_VERSION,
            ),
            (
                "runner_booking_receipt",
                LOCAL_PERF_BOOKING_RECEIPT_SCHEMA_VERSION,
            ),
            ("precommit_inventory", PERF_RUN_PRECOMMIT_SCHEMA_VERSION),
        ] {
            let stale_schema = PERF_MANIFEST.replacen(
                &format!("{field} = \"{expected}\""),
                &format!("{field} = \"stale-schema\""),
                1,
            );
            let stale_schema = perf_gate_manifest_identity(&stale_schema, PerfGate::Qg1)
                .expect_err("every artifact schema declaration must match the shared validator");
            assert!(
                stale_schema
                    .to_string()
                    .contains(&format!("manifest schemas.{field} is \"stale-schema\"")),
                "unexpected stale {field} schema error: {stale_schema}"
            );
        }

        let unreviewed_schema = PERF_MANIFEST.replacen(
            "precommit_inventory = \"frankensearch.perf-run-precommit.v5\"",
            "precommit_inventory = \"frankensearch.perf-run-precommit.v5\"\nunreviewed_schema = \"unreviewed.v1\"",
            1,
        );
        let unreviewed_schema = perf_gate_manifest_identity(&unreviewed_schema, PerfGate::Qg1)
            .expect_err("every manifest schema key must be reviewed");
        assert!(
            unreviewed_schema
                .to_string()
                .contains("manifest schemas.unreviewed_schema is unreviewed"),
            "unexpected unreviewed schema error: {unreviewed_schema}"
        );

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

        for invalid in ["0", "15", "17", "\"sixteen\""] {
            let mutated = PERF_MANIFEST.replacen(
                "queries_per_class = 16",
                &format!("queries_per_class = {invalid}"),
                1,
            );
            assert!(matches!(
                perf_gate_manifest_identity(&mutated, PerfGate::Qg1),
                Err(PerfApplicabilityPlanError::ManifestContract {
                    gate: PerfGate::Qg1,
                    ..
                })
            ));
        }

        let missing_qg6_query_groups = PERF_MANIFEST.replacen("queries_per_class = 16\n", "", 1);
        assert!(matches!(
            perf_gate_manifest_identity(&missing_qg6_query_groups, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::ManifestContract {
                gate: PerfGate::Qg1,
                ..
            })
        ));

        let unexpected_qg6_field = PERF_MANIFEST.replacen(
            "queries_per_class = 16",
            "queries_per_class = 16\nunreviewed_query_groups = 32",
            1,
        );
        assert!(matches!(
            perf_gate_manifest_identity(&unexpected_qg6_field, PerfGate::Qg1),
            Err(PerfApplicabilityPlanError::ManifestContract {
                gate: PerfGate::Qg1,
                ..
            })
        ));

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
        // GOLDEN-CHANGE (QG-5 xlarge re-pin, 02b5ec25): the QG-5 fixture line
        // moved from `50k docs (medium)` to `1M docs (xlarge)` once the e6.1
        // xlarge generator landed, so the manifest a producer must bind is a
        // different document and every measurement taken against the medium
        // pin is correctly invalidated. That commit changed
        // docs/contracts/quill-perf-gates.toml without re-freezing here, which
        // is what left this gate red on the trunk (bd-916qm); the digest is
        // re-frozen to the committed file, NOT the gate loosened -- the
        // assertions below still bind every non-administrative byte.
        // Activation is still the sole administrative normalization exception.
        assert_eq!(
            perf_manifest_contract_sha256(manifest),
            "6b23048474bf8812bfc3527c7eb6f28f70bbdc9b25618b2734ee709c9f7da048",
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

    /// bd-quill-e8-perf-doctrine-x4e4.15: the human run plan is generated
    /// from the manifest and the compiled matrix. This test re-renders it and
    /// compares bytes, so the document can never drift from the harness.
    /// Deliberate regeneration is `QUILL_PERF_RUN_PLAN_UPDATE=1` with a
    /// reviewed diff — never to force green.
    #[test]
    fn perf_run_plan_document_matches_the_manifest() {
        let rendered = render_perf_run_plan_markdown().expect("render run plan");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PERF_RUN_PLAN_DOC_PATH);
        if std::env::var_os("QUILL_PERF_RUN_PLAN_UPDATE").is_some() {
            std::fs::write(&path, &rendered).expect("update run plan document");
            return;
        }
        let committed = std::fs::read_to_string(&path).expect("read committed run plan document");
        assert_eq!(
            rendered, committed,
            "the generated run plan drifted from the manifest/matrix; regenerate with \
             QUILL_PERF_RUN_PLAN_UPDATE=1 and review the diff"
        );
    }
}
