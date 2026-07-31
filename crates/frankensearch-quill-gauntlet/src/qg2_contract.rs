//! Executable validation for the QG-2 symmetric in-memory comparator contract.
//!
//! The validator is intentionally read-only. It binds the six logical
//! contract surfaces, their seven physical locators, the typed TOML topology,
//! and the ten canonical unmeasured sentinels into one fresh-process receipt.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{PerfGate, PerfGateArtifact, is_explicit_bootstrap, perf_manifest_contract_sha256};

/// Version of the fresh-process QG-2 contract validation report.
pub const QG2_CONTRACT_REPORT_SCHEMA_VERSION: &str = "frankensearch.quill-qg2-contract-report.v1";
/// Exact normative QG-2 clause shared by every authoritative physical locator.
pub const QG2_CANONICAL_CONTRACT: &str = "BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 compares both arms symmetrically in memory with no durable storage. Continuous timing begins at the first document feed and ends only after terminal searchable visibility plus complete worker, merge, and queue quiescence. Commit is the searchable-visibility boundary, not durable publication. QG-2 excludes fsync, F_FULLFSYNC, crash recovery, durable publication, and on-disk-byte measurements. Durable gates and production-source durability nonregression remain mandatory outside QG-2.";
/// Number of independent normative QG-2 contract surfaces.
pub const QG2_LOGICAL_SURFACE_COUNT: usize = 6;
/// Number of concrete locators occupied by the six logical surfaces.
pub const QG2_PHYSICAL_LOCATOR_COUNT: usize = 7;
/// Number of canonical unmeasured gate sentinels.
pub const QG2_SENTINEL_COUNT: usize = 10;

const PERF_GATES_DOC_PATH: &str = "docs/contracts/quill-perf-gates.md";
const COMPREHENSIVE_PLAN_PATH: &str = "COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md";
const PERF_MANIFEST_PATH: &str = "docs/contracts/quill-perf-gates.toml";
const HYPEROPT_DOC_PATH: &str = "docs/contracts/quill-hyperopt-campaign.md";
const TRACKER_PATH: &str = ".beads/issues.jsonl";
const HISTORY_DIRECTORY: &str = ".bench-history";
const MAX_DIVERGENCES: usize = 32;
const MAX_DIAGNOSTIC_BYTES: usize = 512;
const MAX_RETRY_BYTES: usize = 256;
const STALE_PLAN_ISSUE_ID: &str = "bd-quill-e8-hyperopt-nyps.1";
const STALE_PLAN_PHRASE: &str =
    "Integrate the already-admissible QG-2 baseline as first-class campaign input.";
const STALE_SUPERSESSION_PREFIX: &str = "BINDING SUPERSESSION 2026-07-30: the phrase below saying the QG-2 baseline was already admissible is false and retained only as historical plan text.";
const STALE_PRESERVED_VALUES: [&str; 2] = ["0.349775", "0.345546"];
const SURFACE_RETRY: &str =
    "Restore the exact canonical Q2C clause at the named locator, then rerun quill-qg2-contract.";
const MANIFEST_RETRY: &str = "Restore the typed gate.QG-2.qg2_contract table, settle final TOML bytes, then rerun quill-qg2-contract.";
const TRACKER_RETRY: &str = "Restore the binding active tracker note without rewriting historical comments, then rerun quill-qg2-contract.";
const STALE_RETRY: &str = "Keep the stale phrase append-only behind its binding supersession; do not promote it into an active field.";
const SENTINEL_RETRY: &str = "After final TOML bytes settle, recompute the normalized manifest SHA-256 and change only manifest_sha256 in all ten unmeasured sentinels.";

/// Storage topology admitted by the QG-2 comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2StorageTopology {
    /// Both the Quill and Tantivy arms are memory-backed.
    SymmetricInMemory,
}

/// Durability scope admitted inside the QG-2 timed operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2DurabilityScope {
    /// The comparator proves visibility, not durable publication.
    NonDurable,
}

/// Start boundary for continuous QG-2 timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2TimingStart {
    /// Start immediately before the first document is fed to either arm.
    FirstDocumentFeed,
}

/// Terminal boundary for continuous QG-2 timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2TimingEnd {
    /// Stop only after visibility and every worker, merge, and queue is quiescent.
    TerminalSearchableVisibilityAndCompleteWorkerMergeQueueQuiescence,
}

/// Meaning of commit inside the QG-2 comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2CommitBoundary {
    /// Commit is a searchable-visibility boundary, not durable publication.
    SearchableVisibilityNotDurablePublication,
}

/// Operations and metrics intentionally excluded from QG-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qg2ExcludedOperation {
    /// POSIX-style file synchronization.
    #[serde(rename = "fsync")]
    Fsync,
    /// macOS full media synchronization.
    #[serde(rename = "F_FULLFSYNC")]
    FFullfsync,
    /// Recovery behavior after a process or machine crash.
    #[serde(rename = "crash_recovery")]
    CrashRecovery,
    /// Publication to a durable on-disk generation.
    #[serde(rename = "durable_publication")]
    DurablePublication,
    /// On-disk storage-footprint measurements.
    #[serde(rename = "on_disk_bytes")]
    OnDiskBytes,
}

/// Production-source obligation that remains outside the QG-2 timing scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2SourceNonregression {
    /// Durable gates and production durability remain mandatory.
    DurableGatesAndProductionSourceDurabilityRemainMandatory,
}

/// Typed machine-readable QG-2 comparator contract from the TOML manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg2ComparatorContract {
    /// Exact prose clause shared with all other authoritative surfaces.
    pub contract: String,
    /// Symmetric storage topology for the two comparator arms.
    pub storage_topology: Qg2StorageTopology,
    /// Durability boundary for this gate.
    pub durability_scope: Qg2DurabilityScope,
    /// Continuous timing start.
    pub timing_start: Qg2TimingStart,
    /// Continuous timing end.
    pub timing_end: Qg2TimingEnd,
    /// Meaning of commit in this gate.
    pub commit_boundary: Qg2CommitBoundary,
    /// Explicitly excluded durable operations and storage metrics.
    pub excluded_operations: Vec<Qg2ExcludedOperation>,
    /// Obligation retained for production code and durable gates.
    pub source_nonregression: Qg2SourceNonregression,
}

impl Qg2ComparatorContract {
    /// Return the one canonical typed contract.
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            contract: QG2_CANONICAL_CONTRACT.to_owned(),
            storage_topology: Qg2StorageTopology::SymmetricInMemory,
            durability_scope: Qg2DurabilityScope::NonDurable,
            timing_start: Qg2TimingStart::FirstDocumentFeed,
            timing_end:
                Qg2TimingEnd::TerminalSearchableVisibilityAndCompleteWorkerMergeQueueQuiescence,
            commit_boundary: Qg2CommitBoundary::SearchableVisibilityNotDurablePublication,
            excluded_operations: vec![
                Qg2ExcludedOperation::Fsync,
                Qg2ExcludedOperation::FFullfsync,
                Qg2ExcludedOperation::CrashRecovery,
                Qg2ExcludedOperation::DurablePublication,
                Qg2ExcludedOperation::OnDiskBytes,
            ],
            source_nonregression:
                Qg2SourceNonregression::DurableGatesAndProductionSourceDurabilityRemainMandatory,
        }
    }
}

/// Overall outcome of QG-2 contract validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2ContractStatus {
    /// Every contract surface and sentinel is exact.
    Pass,
    /// At least one fail-closed divergence was found.
    Divergence,
}

/// Receipt for one expected physical contract locator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2SurfaceReceipt {
    /// Logical surface identity. One logical surface has two physical locators.
    pub logical_surface: String,
    /// Unique physical locator identity.
    pub locator: String,
    /// Project-relative source path.
    pub path: String,
    /// Tracker issue identity when this is a Beads locator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue_id: Option<String>,
    /// Whether the expected physical source or issue object was present.
    pub discovered: bool,
    /// Number of exact canonical clauses in the locator's active region.
    pub marker_count: usize,
    /// SHA-256 of the bounded active region or tracker note.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_sha256: Option<String>,
    /// Whether this individual locator validated.
    pub valid: bool,
}

/// Exact logical and physical topology summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2TopologySummary {
    /// Required logical surface count.
    pub expected_logical_surfaces: usize,
    /// Logical surface identities actually represented.
    pub discovered_logical_surfaces: usize,
    /// Required physical locator count.
    pub expected_physical_locators: usize,
    /// Physical locator identities actually represented.
    pub discovered_physical_locators: usize,
    /// Physical locators that individually validated.
    pub validated_physical_locators: usize,
}

/// Exact unmeasured-sentinel topology summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2SentinelSummary {
    /// Required number of canonical unmeasured sentinels.
    pub expected: usize,
    /// Matching filenames discovered in `.bench-history`.
    pub discovered: usize,
    /// Sentinels with canonical bytes, shape, gate, and manifest binding.
    pub validated: usize,
}

/// Disposition assigned to the retained stale QG-2 planning sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2StaleHistoryDisposition {
    /// The stale claim is retained only as append-only superseded diagnostics.
    AppendOnlySupersededDiagnostic,
}

/// Cardinality receipt for one historically material retained value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2PreservedValueReceipt {
    /// Exact historical value.
    pub value: String,
    /// Required singleton occurrence count in the stale plan description.
    pub count: usize,
}

/// Explicit receipt distinguishing retained stale history from active scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2StaleHistoryReceipt {
    /// Historical issue that retains the superseded sentence.
    pub issue_id: String,
    /// Number of tracker objects found for the historical issue identity.
    pub issue_count: usize,
    /// Its only admissible interpretation.
    pub disposition: Qg2StaleHistoryDisposition,
    /// Number of binding supersession prefixes.
    pub supersession_count: usize,
    /// Number of retained stale "already-admissible" phrases.
    pub stale_phrase_count: usize,
    /// Counts for the two retained historical QG-2 ratios.
    pub preserved_values: Vec<Qg2PreservedValueReceipt>,
    /// Whether the singleton supersession strictly precedes the stale phrase.
    pub supersession_precedes_stale: bool,
    /// Whether cardinality, ordering, and historical values are exact.
    pub valid: bool,
}

/// One bounded, machine-readable contract divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2ContractDivergence {
    /// Stable diagnostic code.
    pub code: String,
    /// Exact project-relative path and optional locator fragment.
    pub path: String,
    /// Bounded expected value.
    pub expected: String,
    /// Bounded observed value, absent when the source could not be read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
    /// SHA-256 of the unbounded expected value, or the expected contract hash.
    pub expected_sha256: String,
    /// SHA-256 of the unbounded observed value, or the observed contract hash.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_sha256: Option<String>,
    /// Bounded concrete retry guidance.
    pub retry: String,
}

/// Fresh-process QG-2 contract validation report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2ContractReport {
    /// Report wire schema.
    pub schema_version: String,
    /// Pass or fail-closed divergence.
    pub status: Qg2ContractStatus,
    /// Canonical typed comparator contract.
    pub contract: Qg2ComparatorContract,
    /// Explicit acceptance receipt for append-only superseded historical text.
    pub stale_history: Qg2StaleHistoryReceipt,
    /// Exact six-logical/seven-physical topology summary.
    pub topology: Qg2TopologySummary,
    /// Normalized performance-manifest SHA-256.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<String>,
    /// Exact ten-sentinel summary.
    pub sentinels: Qg2SentinelSummary,
    /// Ordered receipts for all seven expected physical locators.
    pub surfaces: Vec<Qg2SurfaceReceipt>,
    /// Ordered, bounded divergences.
    pub divergences: Vec<Qg2ContractDivergence>,
    /// Number of additional divergences suppressed by the report bound.
    pub dropped_divergences: usize,
}

impl Qg2ContractReport {
    /// Whether every authoritative surface and sentinel validated.
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self.status, Qg2ContractStatus::Pass)
    }
}

#[derive(Debug, Clone, Copy)]
struct TextSurfaceSpec {
    logical_surface: &'static str,
    locator: &'static str,
    path: &'static str,
    region_start: &'static str,
    region_end: &'static str,
}

const PERF_GATES_SURFACE: TextSurfaceSpec = TextSurfaceSpec {
    logical_surface: "performance_gate_law_1",
    locator: "perf_gate_law_1",
    path: PERF_GATES_DOC_PATH,
    region_start: "1. **No benchmark-only semantics; comparator scope is explicit.**",
    region_end: "2. **Distributions, not averages.**",
};
const PLAN_SURFACE: TextSurfaceSpec = TextSurfaceSpec {
    logical_surface: "comprehensive_plan_qg2",
    locator: "comprehensive_plan_qg2_row",
    path: COMPREHENSIVE_PLAN_PATH,
    region_start: "| **QG-2 Bulk indexing, single-thread**",
    region_end: "| **QG-3 Watch-mode incremental**",
};
const HYPEROPT_SURFACE: TextSurfaceSpec = TextSurfaceSpec {
    logical_surface: "hyperopt_law_7_and_epic",
    locator: "hyperopt_law_7",
    path: HYPEROPT_DOC_PATH,
    region_start: "7. **QG-2 comparator scope and platform durability.**",
    region_end: "## 2. Hardware/profile matrix",
};

#[derive(Debug, Deserialize)]
struct ManifestDocument {
    gate: BTreeMap<String, ManifestGate>,
}

#[derive(Debug, Deserialize)]
struct ManifestGate {
    #[serde(default)]
    qg2_contract: Option<Qg2ComparatorContract>,
}

#[derive(Debug, Clone, Deserialize)]
struct TrackerIssue {
    id: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    acceptance_criteria: Option<String>,
    #[serde(default)]
    design: Option<String>,
    #[serde(default)]
    notes: Option<String>,
}

impl TrackerIssue {
    fn active_marker_count(&self) -> usize {
        [
            self.description.as_deref(),
            self.acceptance_criteria.as_deref(),
            self.design.as_deref(),
            self.notes.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(|value| value.matches(QG2_CANONICAL_CONTRACT).count())
        .sum()
    }
}

struct ReportBuilder {
    manifest_sha256: Option<String>,
    sentinels: Qg2SentinelSummary,
    stale_history: Qg2StaleHistoryReceipt,
    surfaces: Vec<Qg2SurfaceReceipt>,
    divergences: Vec<Qg2ContractDivergence>,
    dropped_divergences: usize,
}

impl ReportBuilder {
    fn new() -> Self {
        Self {
            manifest_sha256: None,
            sentinels: Qg2SentinelSummary {
                expected: QG2_SENTINEL_COUNT,
                discovered: 0,
                validated: 0,
            },
            stale_history: Qg2StaleHistoryReceipt {
                issue_id: STALE_PLAN_ISSUE_ID.to_owned(),
                issue_count: 0,
                disposition: Qg2StaleHistoryDisposition::AppendOnlySupersededDiagnostic,
                supersession_count: 0,
                stale_phrase_count: 0,
                preserved_values: STALE_PRESERVED_VALUES
                    .into_iter()
                    .map(|value| Qg2PreservedValueReceipt {
                        value: value.to_owned(),
                        count: 0,
                    })
                    .collect(),
                supersession_precedes_stale: false,
                valid: false,
            },
            surfaces: Vec::with_capacity(QG2_PHYSICAL_LOCATOR_COUNT),
            divergences: Vec::new(),
            dropped_divergences: 0,
        }
    }

    fn divergence(
        &mut self,
        code: &str,
        path: &str,
        expected: &str,
        observed: Option<&str>,
        retry: &str,
    ) {
        let expected_hash = sha256_hex(expected.as_bytes());
        let observed_hash = observed.map(|value| sha256_hex(value.as_bytes()));
        self.divergence_with_hashes(
            code,
            path,
            expected,
            observed,
            &expected_hash,
            observed_hash.as_deref(),
            retry,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn divergence_with_hashes(
        &mut self,
        code: &str,
        path: &str,
        expected: &str,
        observed: Option<&str>,
        expected_sha256: &str,
        observed_sha256: Option<&str>,
        retry: &str,
    ) {
        if self.divergences.len() == MAX_DIVERGENCES {
            self.dropped_divergences = self.dropped_divergences.saturating_add(1);
            return;
        }
        self.divergences.push(Qg2ContractDivergence {
            code: bounded(code, MAX_DIAGNOSTIC_BYTES),
            path: bounded(path, MAX_DIAGNOSTIC_BYTES),
            expected: bounded(expected, MAX_DIAGNOSTIC_BYTES),
            observed: observed.map(|value| bounded(value, MAX_DIAGNOSTIC_BYTES)),
            expected_sha256: bounded(expected_sha256, 64),
            observed_sha256: observed_sha256.map(|value| bounded(value, 64)),
            retry: bounded(retry, MAX_RETRY_BYTES),
        });
    }

    fn finish(mut self) -> Qg2ContractReport {
        let configured_logical = self
            .surfaces
            .iter()
            .map(|receipt| receipt.logical_surface.clone())
            .collect::<BTreeSet<_>>();
        let configured_locators = self
            .surfaces
            .iter()
            .map(|receipt| receipt.locator.clone())
            .collect::<BTreeSet<_>>();
        if configured_logical.len() != QG2_LOGICAL_SURFACE_COUNT
            || configured_locators.len() != QG2_PHYSICAL_LOCATOR_COUNT
            || self.surfaces.len() != QG2_PHYSICAL_LOCATOR_COUNT
        {
            self.divergence(
                "qg2.topology.validator_map",
                "qg2://validator-map",
                "six configured logical surfaces mapped to seven unique physical locators",
                Some(&format!(
                    "{} logical surfaces, {} locators, {} receipts",
                    configured_logical.len(),
                    configured_locators.len(),
                    self.surfaces.len()
                )),
                "Restore the validator's fixed six-logical/seven-physical topology map.",
            );
        }
        let discovered_logical = self
            .surfaces
            .iter()
            .filter(|receipt| receipt.discovered)
            .map(|receipt| receipt.logical_surface.clone())
            .collect::<BTreeSet<_>>();
        let discovered_physical_locators = self
            .surfaces
            .iter()
            .filter(|receipt| receipt.discovered)
            .count();
        if discovered_logical.len() != QG2_LOGICAL_SURFACE_COUNT {
            self.divergence(
                "qg2.topology.logical_surface_count",
                "qg2://topology",
                &QG2_LOGICAL_SURFACE_COUNT.to_string(),
                Some(&discovered_logical.len().to_string()),
                "Restore every expected logical contract surface at its fixed locator.",
            );
        }
        if discovered_physical_locators != QG2_PHYSICAL_LOCATOR_COUNT {
            self.divergence(
                "qg2.topology.physical_locator_count",
                "qg2://topology",
                &QG2_PHYSICAL_LOCATOR_COUNT.to_string(),
                Some(&discovered_physical_locators.to_string()),
                "Restore every expected physical contract locator; do not add substitute surfaces.",
            );
        }
        let validated_physical_locators =
            self.surfaces.iter().filter(|receipt| receipt.valid).count();
        let status = if self.divergences.is_empty() {
            Qg2ContractStatus::Pass
        } else {
            Qg2ContractStatus::Divergence
        };
        Qg2ContractReport {
            schema_version: QG2_CONTRACT_REPORT_SCHEMA_VERSION.to_owned(),
            status,
            contract: Qg2ComparatorContract::canonical(),
            stale_history: self.stale_history,
            topology: Qg2TopologySummary {
                expected_logical_surfaces: QG2_LOGICAL_SURFACE_COUNT,
                discovered_logical_surfaces: discovered_logical.len(),
                expected_physical_locators: QG2_PHYSICAL_LOCATOR_COUNT,
                discovered_physical_locators,
                validated_physical_locators,
            },
            manifest_sha256: self.manifest_sha256,
            sentinels: self.sentinels,
            surfaces: self.surfaces,
            divergences: self.divergences,
            dropped_divergences: self.dropped_divergences,
        }
    }
}

/// Validate the complete QG-2 contract topology beneath one repository root.
///
/// This function performs only read operations. Every source failure becomes a
/// bounded divergence rather than an early return, so one invocation reports
/// all independently actionable repair paths.
#[must_use]
pub fn validate_qg2_contract(repo_root: &Path) -> Qg2ContractReport {
    let mut report = ReportBuilder::new();
    validate_text_surface(repo_root, PERF_GATES_SURFACE, &mut report);
    validate_text_surface(repo_root, PLAN_SURFACE, &mut report);
    validate_manifest_surface(repo_root, &mut report);
    validate_text_surface(repo_root, HYPEROPT_SURFACE, &mut report);
    validate_tracker_surfaces(repo_root, &mut report);
    validate_sentinels(repo_root, &mut report);
    report.finish()
}

fn validate_text_surface(repo_root: &Path, spec: TextSurfaceSpec, report: &mut ReportBuilder) {
    let source = match fs::read_to_string(repo_root.join(spec.path)) {
        Ok(source) => source,
        Err(error) => {
            report.divergence(
                "qg2.surface.read",
                spec.path,
                QG2_CANONICAL_CONTRACT,
                Some(&error.to_string()),
                SURFACE_RETRY,
            );
            report
                .surfaces
                .push(surface_receipt(spec, false, 0, None, false));
            return;
        }
    };
    let marker_count = source.matches(QG2_CANONICAL_CONTRACT).count();
    let region = unique_region(&source, spec.region_start, spec.region_end);
    let mut valid = marker_count == 1;
    if marker_count != 1 {
        report.divergence(
            "qg2.surface.marker_count",
            spec.path,
            "exactly one canonical Q2C clause in the authoritative file",
            Some(&format!("{marker_count} canonical clauses")),
            SURFACE_RETRY,
        );
    }
    let content_sha256 = match region {
        Ok(region) => {
            if region.matches(QG2_CANONICAL_CONTRACT).count() != 1 {
                valid = false;
                report.divergence(
                    "qg2.surface.marker_scope",
                    spec.path,
                    "canonical Q2C clause inside the named authoritative region",
                    Some(region),
                    SURFACE_RETRY,
                );
            }
            Some(sha256_hex(region.as_bytes()))
        }
        Err(error) => {
            valid = false;
            report.divergence(
                "qg2.surface.region",
                spec.path,
                &format!(
                    "one region from {:?} through {:?}",
                    spec.region_start, spec.region_end
                ),
                Some(&error),
                SURFACE_RETRY,
            );
            Some(sha256_hex(source.as_bytes()))
        }
    };
    report.surfaces.push(surface_receipt(
        spec,
        true,
        marker_count,
        content_sha256,
        valid,
    ));
}

fn surface_receipt(
    spec: TextSurfaceSpec,
    discovered: bool,
    marker_count: usize,
    content_sha256: Option<String>,
    valid: bool,
) -> Qg2SurfaceReceipt {
    Qg2SurfaceReceipt {
        logical_surface: spec.logical_surface.to_owned(),
        locator: spec.locator.to_owned(),
        path: spec.path.to_owned(),
        issue_id: None,
        discovered,
        marker_count,
        content_sha256,
        valid,
    }
}

fn validate_manifest_surface(repo_root: &Path, report: &mut ReportBuilder) {
    let source = match fs::read_to_string(repo_root.join(PERF_MANIFEST_PATH)) {
        Ok(source) => source,
        Err(error) => {
            report.divergence(
                "qg2.manifest.read",
                PERF_MANIFEST_PATH,
                "readable typed QG-2 manifest",
                Some(&error.to_string()),
                MANIFEST_RETRY,
            );
            report.surfaces.push(Qg2SurfaceReceipt {
                logical_surface: "machine_manifest_qg2".to_owned(),
                locator: "perf_manifest_qg2_contract".to_owned(),
                path: PERF_MANIFEST_PATH.to_owned(),
                issue_id: None,
                discovered: false,
                marker_count: 0,
                content_sha256: None,
                valid: false,
            });
            return;
        }
    };

    let manifest_sha256 = perf_manifest_contract_sha256(&source);
    report.manifest_sha256 = Some(manifest_sha256);
    let marker_count = source.matches(QG2_CANONICAL_CONTRACT).count();
    let canonical = Qg2ComparatorContract::canonical();
    let parsed = toml::from_str::<ManifestDocument>(&source);
    let mut valid = marker_count == 1;
    let mut content_sha256 = Some(sha256_hex(source.as_bytes()));
    if marker_count != 1 {
        report.divergence(
            "qg2.manifest.marker_count",
            "docs/contracts/quill-perf-gates.toml#gate.QG-2.qg2_contract",
            "exactly one canonical Q2C clause",
            Some(&format!("{marker_count} canonical clauses")),
            MANIFEST_RETRY,
        );
    }
    match parsed {
        Ok(document) => match document
            .gate
            .get("QG-2")
            .and_then(|gate| gate.qg2_contract.as_ref())
        {
            Some(observed) => {
                let observed_json = contract_json(observed);
                let expected_json = contract_json(&canonical);
                content_sha256 = Some(sha256_hex(observed_json.as_bytes()));
                if observed != &canonical {
                    valid = false;
                    report.divergence(
                        "qg2.manifest.typed_contract",
                        "docs/contracts/quill-perf-gates.toml#gate.QG-2.qg2_contract",
                        &expected_json,
                        Some(&observed_json),
                        MANIFEST_RETRY,
                    );
                }
            }
            None => {
                valid = false;
                report.divergence(
                    "qg2.manifest.missing_typed_contract",
                    "docs/contracts/quill-perf-gates.toml#gate.QG-2.qg2_contract",
                    &contract_json(&canonical),
                    None,
                    MANIFEST_RETRY,
                );
            }
        },
        Err(error) => {
            valid = false;
            report.divergence(
                "qg2.manifest.parse",
                PERF_MANIFEST_PATH,
                "valid TOML with the closed typed QG-2 contract",
                Some(&error.to_string()),
                MANIFEST_RETRY,
            );
        }
    }
    report.surfaces.push(Qg2SurfaceReceipt {
        logical_surface: "machine_manifest_qg2".to_owned(),
        locator: "perf_manifest_qg2_contract".to_owned(),
        path: PERF_MANIFEST_PATH.to_owned(),
        issue_id: None,
        discovered: true,
        marker_count,
        content_sha256,
        valid,
    });
}

fn validate_tracker_surfaces(repo_root: &Path, report: &mut ReportBuilder) {
    let expected = [
        (
            "hyperopt_law_7_and_epic",
            "hyperopt_epic_active_contract",
            "bd-quill-e8-hyperopt-nyps",
        ),
        (
            "qg2_r1_quarantine",
            "qg2_r1_active_contract",
            "bd-quill-e8-perf-doctrine-x4e4.5.5",
        ),
        (
            "gate_activation_scope",
            "gate_activation_active_contract",
            "bd-h6eh",
        ),
    ];
    let source = match fs::read_to_string(repo_root.join(TRACKER_PATH)) {
        Ok(source) => source,
        Err(error) => {
            report.divergence(
                "qg2.tracker.read",
                TRACKER_PATH,
                "readable JSONL with three active contract notes",
                Some(&error.to_string()),
                TRACKER_RETRY,
            );
            for (logical_surface, locator, issue_id) in expected {
                report.surfaces.push(tracker_receipt(
                    logical_surface,
                    locator,
                    issue_id,
                    false,
                    0,
                    None,
                    false,
                ));
            }
            return;
        }
    };

    let expected_ids = expected
        .iter()
        .map(|(_, _, issue_id)| *issue_id)
        .collect::<BTreeSet<_>>();
    let mut selected = BTreeMap::<String, Vec<TrackerIssue>>::new();
    for (line_index, line) in source.lines().enumerate() {
        match serde_json::from_str::<TrackerIssue>(line) {
            Ok(issue) => {
                let marker_count = issue.active_marker_count();
                if marker_count > 0 && !expected_ids.contains(issue.id.as_str()) {
                    report.divergence(
                        "qg2.tracker.extra_active_surface",
                        &format!("{TRACKER_PATH}#{}", issue.id),
                        "canonical Q2C clause only in the three expected active tracker issues",
                        Some(&format!("{marker_count} active canonical clauses")),
                        TRACKER_RETRY,
                    );
                }
                if expected_ids.contains(issue.id.as_str()) || issue.id == STALE_PLAN_ISSUE_ID {
                    selected.entry(issue.id.clone()).or_default().push(issue);
                }
            }
            Err(error) => report.divergence(
                "qg2.tracker.jsonl",
                &format!("{TRACKER_PATH}:{}", line_index + 1),
                "one valid tracker JSON object",
                Some(&error.to_string()),
                TRACKER_RETRY,
            ),
        }
    }

    for (logical_surface, locator, issue_id) in expected {
        let matches = selected.get(issue_id).map_or(&[][..], Vec::as_slice);
        if matches.len() != 1 {
            report.divergence(
                "qg2.tracker.issue_cardinality",
                &format!("{TRACKER_PATH}#{issue_id}"),
                "exactly one active issue object",
                Some(&format!("{} issue objects", matches.len())),
                TRACKER_RETRY,
            );
            report.surfaces.push(tracker_receipt(
                logical_surface,
                locator,
                issue_id,
                !matches.is_empty(),
                0,
                None,
                false,
            ));
            continue;
        }
        let issue = &matches[0];
        let marker_count = issue.active_marker_count();
        let note = issue.notes.as_deref();
        let valid = note == Some(QG2_CANONICAL_CONTRACT) && marker_count == 1;
        if !valid {
            report.divergence(
                "qg2.tracker.active_contract",
                &format!("{TRACKER_PATH}#{issue_id}.notes"),
                QG2_CANONICAL_CONTRACT,
                note,
                TRACKER_RETRY,
            );
        }
        report.surfaces.push(tracker_receipt(
            logical_surface,
            locator,
            issue_id,
            true,
            marker_count,
            note.map(|value| sha256_hex(value.as_bytes())),
            valid,
        ));
    }
    validate_stale_plan_supersession(&selected, report);
}

fn tracker_receipt(
    logical_surface: &str,
    locator: &str,
    issue_id: &str,
    discovered: bool,
    marker_count: usize,
    content_sha256: Option<String>,
    valid: bool,
) -> Qg2SurfaceReceipt {
    Qg2SurfaceReceipt {
        logical_surface: logical_surface.to_owned(),
        locator: locator.to_owned(),
        path: TRACKER_PATH.to_owned(),
        issue_id: Some(issue_id.to_owned()),
        discovered,
        marker_count,
        content_sha256,
        valid,
    }
}

fn validate_stale_plan_supersession(
    selected: &BTreeMap<String, Vec<TrackerIssue>>,
    report: &mut ReportBuilder,
) {
    let matches = selected
        .get(STALE_PLAN_ISSUE_ID)
        .map_or(&[][..], Vec::as_slice);
    report.stale_history.issue_count = matches.len();
    if matches.len() != 1 {
        report.divergence(
            "qg2.tracker.stale_plan_cardinality",
            &format!("{TRACKER_PATH}#{STALE_PLAN_ISSUE_ID}"),
            "exactly one append-only stale plan issue",
            Some(&format!("{} issue objects", matches.len())),
            STALE_RETRY,
        );
        return;
    }
    let description = matches[0].description.as_deref().unwrap_or_default();
    let supersession_count = description.matches(STALE_SUPERSESSION_PREFIX).count();
    let stale_phrase_count = description.matches(STALE_PLAN_PHRASE).count();
    let preserved_value_counts =
        STALE_PRESERVED_VALUES.map(|value| (value, description.matches(value).count()));
    let supersession_position = description.find(STALE_SUPERSESSION_PREFIX);
    let stale_position = description.find(STALE_PLAN_PHRASE);
    let supersession_precedes_stale = supersession_position
        .zip(stale_position)
        .is_some_and(|(supersession, stale)| supersession < stale);
    let valid = supersession_count == 1
        && stale_phrase_count == 1
        && preserved_value_counts.iter().all(|(_, count)| *count == 1)
        && supersession_position == Some(0)
        && stale_position.is_some_and(|position| position > STALE_SUPERSESSION_PREFIX.len());
    report.stale_history = Qg2StaleHistoryReceipt {
        issue_id: STALE_PLAN_ISSUE_ID.to_owned(),
        issue_count: matches.len(),
        disposition: Qg2StaleHistoryDisposition::AppendOnlySupersededDiagnostic,
        supersession_count,
        stale_phrase_count,
        preserved_values: preserved_value_counts
            .into_iter()
            .map(|(value, count)| Qg2PreservedValueReceipt {
                value: value.to_owned(),
                count,
            })
            .collect(),
        supersession_precedes_stale,
        valid,
    };
    if !valid {
        report.divergence(
            "qg2.tracker.stale_scope_unsuperseded",
            &format!("{TRACKER_PATH}#{STALE_PLAN_ISSUE_ID}.description"),
            &format!("{STALE_SUPERSESSION_PREFIX} ... {STALE_PLAN_PHRASE}"),
            Some(&format!(
                "supersession_count={supersession_count}; stale_phrase_count={stale_phrase_count}; preserved_value_counts={preserved_value_counts:?}; description={description}"
            )),
            STALE_RETRY,
        );
    }
}

fn validate_sentinels(repo_root: &Path, report: &mut ReportBuilder) {
    let expected = PerfGate::ALL
        .into_iter()
        .map(|gate| (format!("{}.unmeasured.latest.json", gate.label()), gate))
        .collect::<BTreeMap<_, _>>();
    let directory = repo_root.join(HISTORY_DIRECTORY);
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) => {
            report.divergence(
                "qg2.sentinel.directory",
                HISTORY_DIRECTORY,
                "readable .bench-history directory with exactly ten unmeasured sentinels",
                Some(&error.to_string()),
                SENTINEL_RETRY,
            );
            return;
        }
    };
    let mut discovered = BTreeSet::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.divergence(
                    "qg2.sentinel.enumeration",
                    HISTORY_DIRECTORY,
                    "readable directory entry",
                    Some(&error.to_string()),
                    SENTINEL_RETRY,
                );
                continue;
            }
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("QG-") && name.ends_with(".unmeasured.latest.json") {
            discovered.insert(name.into_owned());
        }
    }
    report.sentinels.discovered = discovered.len();

    for missing in expected
        .keys()
        .filter(|name| !discovered.contains(name.as_str()))
    {
        report.divergence(
            "qg2.sentinel.missing",
            &format!("{HISTORY_DIRECTORY}/{missing}"),
            "canonical unmeasured sentinel",
            None,
            SENTINEL_RETRY,
        );
    }
    for extra in discovered
        .iter()
        .filter(|name| !expected.contains_key(name.as_str()))
    {
        report.divergence(
            "qg2.sentinel.extra",
            &format!("{HISTORY_DIRECTORY}/{extra}"),
            "no unmeasured sentinel outside QG-1 through QG-10",
            Some("unexpected sentinel filename"),
            SENTINEL_RETRY,
        );
    }

    let expected_manifest_sha256 = report.manifest_sha256.clone();
    for (name, gate) in expected {
        if !discovered.contains(&name) {
            continue;
        }
        let path = format!("{HISTORY_DIRECTORY}/{name}");
        if validate_one_sentinel(
            repo_root,
            &path,
            gate,
            expected_manifest_sha256.as_deref(),
            report,
        ) {
            report.sentinels.validated += 1;
        }
    }
}

fn validate_one_sentinel(
    repo_root: &Path,
    path: &str,
    gate: PerfGate,
    expected_manifest_sha256: Option<&str>,
    report: &mut ReportBuilder,
) -> bool {
    let bytes = match fs::read(repo_root.join(path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            report.divergence(
                "qg2.sentinel.read",
                path,
                "readable canonical sentinel JSON",
                Some(&error.to_string()),
                SENTINEL_RETRY,
            );
            return false;
        }
    };
    let artifact = match serde_json::from_slice::<PerfGateArtifact>(&bytes) {
        Ok(artifact) => artifact,
        Err(error) => {
            report.divergence(
                "qg2.sentinel.parse",
                path,
                "current-schema sentinel JSON",
                Some(&error.to_string()),
                SENTINEL_RETRY,
            );
            return false;
        }
    };
    let mut valid = true;
    let canonical = match serde_json::to_vec_pretty(&artifact) {
        Ok(canonical) => canonical,
        Err(error) => {
            report.divergence(
                "qg2.sentinel.serialize",
                path,
                "canonically serializable sentinel JSON",
                Some(&error.to_string()),
                SENTINEL_RETRY,
            );
            return false;
        }
    };
    if canonical != bytes {
        valid = false;
        report.divergence_with_hashes(
            "qg2.sentinel.canonical_bytes",
            path,
            "canonical pretty JSON with no trailing bytes",
            Some("noncanonical JSON bytes"),
            &sha256_hex(&canonical),
            Some(&sha256_hex(&bytes)),
            SENTINEL_RETRY,
        );
    }
    if !is_explicit_bootstrap(&artifact) || artifact.gate != gate {
        valid = false;
        report.divergence(
            "qg2.sentinel.shape",
            path,
            &format!("exact current-schema {} unmeasured sentinel", gate.label()),
            Some(&format!(
                "gate={:?}, schema={}, cells={}, laws_attested={}",
                artifact.gate,
                artifact.schema_version,
                artifact.cells.len(),
                artifact.laws_attested
            )),
            SENTINEL_RETRY,
        );
    }
    match expected_manifest_sha256 {
        Some(expected_hash) if !artifact.manifest_sha256.eq(expected_hash) => {
            valid = false;
            report.divergence_with_hashes(
                "qg2.sentinel.manifest_hash",
                path,
                expected_hash,
                Some(&artifact.manifest_sha256),
                expected_hash,
                Some(&artifact.manifest_sha256),
                SENTINEL_RETRY,
            );
        }
        None => {
            valid = false;
            report.divergence(
                "qg2.sentinel.manifest_unavailable",
                path,
                "manifest hash derived from readable final TOML bytes",
                None,
                SENTINEL_RETRY,
            );
        }
        Some(_) => {}
    }
    valid
}

fn unique_region<'a>(source: &'a str, start: &str, end: &str) -> Result<&'a str, String> {
    let start_count = source.matches(start).count();
    if start_count != 1 {
        return Err(format!("start anchor count was {start_count}, expected 1"));
    }
    let start_offset = source
        .find(start)
        .ok_or_else(|| "start anchor disappeared after counting".to_owned())?;
    let tail = &source[start_offset..];
    let end_count = tail.matches(end).count();
    if end_count != 1 {
        return Err(format!(
            "end anchor count after start was {end_count}, expected 1"
        ));
    }
    let end_offset = tail
        .find(end)
        .ok_or_else(|| "end anchor disappeared after counting".to_owned())?;
    Ok(&tail[..end_offset])
}

fn contract_json(contract: &Qg2ComparatorContract) -> String {
    serde_json::to_string(contract)
        .unwrap_or_else(|error| format!("unserializable canonical contract: {error}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn bounded(value: &str, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value.to_owned();
    }
    let suffix = "...";
    let mut end = maximum_bytes.saturating_sub(suffix.len());
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    let mut bounded = String::with_capacity(maximum_bytes);
    bounded.push_str(&value[..end]);
    bounded.push_str(suffix);
    bounded
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    struct Fixture {
        directory: TempDir,
        manifest: String,
    }

    impl Fixture {
        fn root(&self) -> &Path {
            self.directory.path()
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.root().join(relative)
        }
    }

    fn complete_fixture() -> Fixture {
        let directory = tempfile::tempdir().expect("temporary repository");
        let root = directory.path();
        fs::create_dir_all(root.join("docs/contracts")).expect("contract directory");
        fs::create_dir_all(root.join(".beads")).expect("tracker directory");
        fs::create_dir_all(root.join(HISTORY_DIRECTORY)).expect("history directory");

        fs::write(
            root.join(PERF_GATES_DOC_PATH),
            format!(
                "# Laws\n1. **No benchmark-only semantics; comparator scope is explicit.** {QG2_CANONICAL_CONTRACT}\n2. **Distributions, not averages.** next\n"
            ),
        )
        .expect("performance gate fixture");
        fs::write(
            root.join(COMPREHENSIVE_PLAN_PATH),
            format!(
                "| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |\n| **QG-3 Watch-mode incremental** | next |\n"
            ),
        )
        .expect("plan fixture");
        fs::write(
            root.join(HYPEROPT_DOC_PATH),
            format!(
                "7. **QG-2 comparator scope and platform durability.** {QG2_CANONICAL_CONTRACT}\n## 2. Hardware/profile matrix\n"
            ),
        )
        .expect("hyperopt fixture");

        let manifest = format!(
            "[gate.QG-2]\nname = \"bulk indexing, single-thread\"\nactivated = false\n\n[gate.QG-2.qg2_contract]\ncontract = {contract:?}\nstorage_topology = \"symmetric_in_memory\"\ndurability_scope = \"non_durable\"\ntiming_start = \"first_document_feed\"\ntiming_end = \"terminal_searchable_visibility_and_complete_worker_merge_queue_quiescence\"\ncommit_boundary = \"searchable_visibility_not_durable_publication\"\nexcluded_operations = [\"fsync\", \"F_FULLFSYNC\", \"crash_recovery\", \"durable_publication\", \"on_disk_bytes\"]\nsource_nonregression = \"durable_gates_and_production_source_durability_remain_mandatory\"\n",
            contract = QG2_CANONICAL_CONTRACT
        );
        fs::write(root.join(PERF_MANIFEST_PATH), &manifest).expect("manifest fixture");

        let mut tracker = String::new();
        for issue_id in [
            "bd-quill-e8-hyperopt-nyps",
            "bd-quill-e8-perf-doctrine-x4e4.5.5",
            "bd-h6eh",
        ] {
            tracker.push_str(
                &serde_json::to_string(&json!({
                    "id": issue_id,
                    "notes": QG2_CANONICAL_CONTRACT
                }))
                .expect("tracker issue"),
            );
            tracker.push('\n');
        }
        tracker.push_str(
            &serde_json::to_string(&json!({
                "id": STALE_PLAN_ISSUE_ID,
                "description": format!(
                    "{STALE_SUPERSESSION_PREFIX} The 0.349775 candidate and 0.345546 rerun remain immutable diagnostics.\n\nHistorical body. {STALE_PLAN_PHRASE}"
                )
            }))
            .expect("stale tracker issue"),
        );
        tracker.push('\n');
        fs::write(root.join(TRACKER_PATH), tracker).expect("tracker fixture");

        let manifest_sha256 = perf_manifest_contract_sha256(&manifest);
        let template = serde_json::from_str::<PerfGateArtifact>(include_str!(
            "../../../.bench-history/QG-1.unmeasured.latest.json"
        ))
        .expect("sentinel template");
        for gate in PerfGate::ALL {
            let mut artifact = template.clone();
            artifact.gate = gate;
            artifact.manifest_sha256.clone_from(&manifest_sha256);
            fs::write(
                root.join(format!(
                    "{HISTORY_DIRECTORY}/{}.unmeasured.latest.json",
                    gate.label()
                )),
                serde_json::to_vec_pretty(&artifact).expect("sentinel JSON"),
            )
            .expect("sentinel fixture");
        }
        Fixture {
            directory,
            manifest,
        }
    }

    #[test]
    fn complete_fixture_has_exact_six_by_seven_topology_and_ten_sentinels() {
        let fixture = complete_fixture();
        let report = validate_qg2_contract(fixture.root());
        assert!(report.is_pass(), "{:#?}", report.divergences);
        assert_eq!(report.topology.expected_logical_surfaces, 6);
        assert_eq!(report.topology.discovered_logical_surfaces, 6);
        assert_eq!(report.topology.expected_physical_locators, 7);
        assert_eq!(report.topology.discovered_physical_locators, 7);
        assert_eq!(report.topology.validated_physical_locators, 7);
        assert_eq!(report.sentinels.expected, 10);
        assert_eq!(report.sentinels.discovered, 10);
        assert_eq!(report.sentinels.validated, 10);
        assert_eq!(report.stale_history.issue_count, 1);
        assert_eq!(report.stale_history.supersession_count, 1);
        assert_eq!(report.stale_history.stale_phrase_count, 1);
        assert_eq!(
            report
                .stale_history
                .preserved_values
                .iter()
                .map(|receipt| (receipt.value.as_str(), receipt.count))
                .collect::<Vec<_>>(),
            vec![("0.349775", 1), ("0.345546", 1)]
        );
        assert!(report.stale_history.supersession_precedes_stale);
        assert!(report.stale_history.valid);
    }

    #[test]
    fn missing_and_extra_contract_markers_fail_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            "| **QG-2 Bulk indexing, single-thread** | missing |\n| **QG-3 Watch-mode incremental** | next |\n",
        )
        .expect("missing marker mutation");
        let missing = validate_qg2_contract(fixture.root());
        assert!(missing.divergences.iter().any(|divergence| {
            divergence.code == "qg2.surface.marker_count"
                && divergence.path == COMPREHENSIVE_PLAN_PATH
        }));

        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            format!(
                "| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} {QG2_CANONICAL_CONTRACT} |\n| **QG-3 Watch-mode incremental** | next |\n"
            ),
        )
        .expect("extra marker mutation");
        let extra = validate_qg2_contract(fixture.root());
        assert!(extra.divergences.iter().any(|divergence| {
            divergence.code == "qg2.surface.marker_count"
                && divergence.path == COMPREHENSIVE_PLAN_PATH
        }));
    }

    #[test]
    fn durability_contamination_is_rejected_by_typed_manifest() {
        let fixture = complete_fixture();
        let contaminated = fixture.manifest.replace(
            "durability_scope = \"non_durable\"",
            "durability_scope = \"durable\"",
        );
        fs::write(fixture.path(PERF_MANIFEST_PATH), contaminated)
            .expect("durability contamination mutation");
        let report = validate_qg2_contract(fixture.root());
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code.eq("qg2.manifest.parse"))
        );
    }

    #[test]
    fn every_durable_operation_exclusion_is_required() {
        let mutations = [
            ("\"fsync\", ", "fsync"),
            ("\"F_FULLFSYNC\", ", "F_FULLFSYNC"),
            ("\"crash_recovery\", ", "crash_recovery"),
            ("\"durable_publication\", ", "durable_publication"),
            (", \"on_disk_bytes\"", "on_disk_bytes"),
        ];
        for (needle, operation) in mutations {
            let fixture = complete_fixture();
            let contaminated = fixture.manifest.replace(needle, "");
            assert_ne!(
                contaminated, fixture.manifest,
                "test mutation did not remove {operation}"
            );
            fs::write(fixture.path(PERF_MANIFEST_PATH), contaminated)
                .expect("excluded-operation mutation");
            let report = validate_qg2_contract(fixture.root());
            assert!(
                report
                    .divergences
                    .iter()
                    .any(|divergence| divergence.code.eq("qg2.manifest.typed_contract")),
                "removing {operation} unexpectedly passed: {:#?}",
                report.divergences
            );
        }
    }

    #[test]
    fn tracker_topology_rejects_missing_and_extra_active_locators() {
        let missing_fixture = complete_fixture();
        let original =
            fs::read_to_string(missing_fixture.path(TRACKER_PATH)).expect("tracker fixture");
        let missing = original
            .lines()
            .filter(|line| !line.contains("\"id\":\"bd-h6eh\""))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(missing_fixture.path(TRACKER_PATH), format!("{missing}\n"))
            .expect("missing tracker locator mutation");
        let missing_report = validate_qg2_contract(missing_fixture.root());
        assert!(missing_report.divergences.iter().any(|divergence| {
            divergence.code == "qg2.tracker.issue_cardinality"
                && divergence.path.ends_with("#bd-h6eh")
        }));
        assert_eq!(
            missing_report.topology.discovered_physical_locators,
            QG2_PHYSICAL_LOCATOR_COUNT - 1
        );

        let extra_fixture = complete_fixture();
        let mut extra =
            fs::read_to_string(extra_fixture.path(TRACKER_PATH)).expect("tracker fixture");
        extra.push_str(
            &serde_json::to_string(&json!({
                "id": "bd-unexpected-qg2-contract",
                "notes": QG2_CANONICAL_CONTRACT
            }))
            .expect("extra tracker locator"),
        );
        extra.push('\n');
        fs::write(extra_fixture.path(TRACKER_PATH), extra).expect("extra tracker locator mutation");
        let extra_report = validate_qg2_contract(extra_fixture.root());
        assert!(extra_report.divergences.iter().any(|divergence| {
            divergence.code == "qg2.tracker.extra_active_surface"
                && divergence.path.ends_with("#bd-unexpected-qg2-contract")
        }));
    }

    #[test]
    fn stale_already_admissible_phrase_requires_leading_supersession() {
        let fixture = complete_fixture();
        let source = fs::read_to_string(fixture.path(TRACKER_PATH)).expect("tracker fixture");
        let unsuperseded = source.replace(STALE_SUPERSESSION_PREFIX, "Historical plan text:");
        fs::write(fixture.path(TRACKER_PATH), unsuperseded).expect("stale supersession mutation");
        let report = validate_qg2_contract(fixture.root());
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| { divergence.code == "qg2.tracker.stale_scope_unsuperseded" })
        );
    }

    #[test]
    fn stale_history_requires_exact_phrase_and_preserved_values() {
        for needle in [
            STALE_PLAN_PHRASE,
            STALE_PRESERVED_VALUES[0],
            STALE_PRESERVED_VALUES[1],
        ] {
            let fixture = complete_fixture();
            let source = fs::read_to_string(fixture.path(TRACKER_PATH)).expect("tracker fixture");
            let mutated = source.replacen(needle, "removed-historical-value", 1);
            fs::write(fixture.path(TRACKER_PATH), mutated)
                .expect("historical preservation mutation");
            let report = validate_qg2_contract(fixture.root());
            assert!(
                report.divergences.iter().any(|divergence| {
                    divergence.code == "qg2.tracker.stale_scope_unsuperseded"
                }),
                "removing {needle} unexpectedly passed"
            );
        }

        let fixture = complete_fixture();
        let source = fs::read_to_string(fixture.path(TRACKER_PATH)).expect("tracker fixture");
        let duplicated = source.replace(
            STALE_PLAN_PHRASE,
            &format!("{STALE_PLAN_PHRASE} {STALE_PLAN_PHRASE}"),
        );
        fs::write(fixture.path(TRACKER_PATH), duplicated).expect("duplicate stale phrase mutation");
        let report = validate_qg2_contract(fixture.root());
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.tracker.stale_scope_unsuperseded")
        );
    }

    #[test]
    fn sentinel_enumeration_rejects_missing_and_extra_gate() {
        let missing_fixture = complete_fixture();
        fs::rename(
            missing_fixture.path(".bench-history/QG-10.unmeasured.latest.json"),
            missing_fixture.path(".bench-history/QG-10.held-for-mutation.json"),
        )
        .expect("missing sentinel mutation");
        let missing_report = validate_qg2_contract(missing_fixture.root());
        assert!(missing_report.divergences.iter().any(|divergence| {
            divergence.code == "qg2.sentinel.missing"
                && divergence.path.ends_with("QG-10.unmeasured.latest.json")
        }));
        assert_eq!(missing_report.sentinels.discovered, QG2_SENTINEL_COUNT - 1);

        let extra_fixture = complete_fixture();
        let qg10 = fs::read(extra_fixture.path(".bench-history/QG-10.unmeasured.latest.json"))
            .expect("QG-10 sentinel");
        fs::write(
            extra_fixture.path(".bench-history/QG-11.unmeasured.latest.json"),
            qg10,
        )
        .expect("extra sentinel mutation");
        let extra_report = validate_qg2_contract(extra_fixture.root());
        assert!(
            extra_report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.sentinel.extra")
        );
    }

    #[test]
    fn sentinel_validation_rejects_stale_manifest_hash() {
        let fixture = complete_fixture();
        let qg4_path = fixture.path(".bench-history/QG-4.unmeasured.latest.json");
        let mut qg4 =
            serde_json::from_slice::<PerfGateArtifact>(&fs::read(&qg4_path).expect("QG-4 bytes"))
                .expect("QG-4 JSON");
        qg4.manifest_sha256 = "0".repeat(64);
        fs::write(
            qg4_path,
            serde_json::to_vec_pretty(&qg4).expect("mutated QG-4"),
        )
        .expect("stale hash mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(report.divergences.iter().any(|divergence| {
            divergence.code == "qg2.sentinel.manifest_hash"
                && divergence.path.ends_with("QG-4.unmeasured.latest.json")
        }));
    }

    #[test]
    fn sentinel_validation_rejects_collateral_final_newline() {
        let fixture = complete_fixture();
        let path = fixture.path(".bench-history/QG-5.unmeasured.latest.json");
        let mut bytes = fs::read(&path).expect("QG-5 sentinel");
        assert!(!bytes.ends_with(b"\n"));
        bytes.push(b'\n');
        fs::write(path, bytes).expect("sentinel newline mutation");
        let report = validate_qg2_contract(fixture.root());
        assert!(report.divergences.iter().any(|divergence| {
            divergence.code == "qg2.sentinel.canonical_bytes"
                && divergence.path.ends_with("QG-5.unmeasured.latest.json")
        }));
    }

    #[test]
    fn manifest_hash_ignores_activation_only_but_binds_topology() {
        let fixture = complete_fixture();
        let activated = fixture
            .manifest
            .replacen("activated = false", "activated = true", 1);
        assert_eq!(
            perf_manifest_contract_sha256(&fixture.manifest),
            perf_manifest_contract_sha256(&activated)
        );
        let changed_topology = fixture.manifest.replace(
            "storage_topology = \"symmetric_in_memory\"",
            "storage_topology = \"disk_backed\"",
        );
        assert_ne!(
            perf_manifest_contract_sha256(&fixture.manifest),
            perf_manifest_contract_sha256(&changed_topology)
        );
    }

    #[test]
    fn diagnostic_values_and_count_are_bounded() {
        let mut builder = ReportBuilder::new();
        for _ in 0..(MAX_DIVERGENCES + 5) {
            builder.divergence(
                &"x".repeat(MAX_DIAGNOSTIC_BYTES * 2),
                &"p".repeat(MAX_DIAGNOSTIC_BYTES * 2),
                &"e".repeat(MAX_DIAGNOSTIC_BYTES * 2),
                Some(&"o".repeat(MAX_DIAGNOSTIC_BYTES * 2)),
                &"r".repeat(MAX_RETRY_BYTES * 2),
            );
        }
        assert_eq!(builder.divergences.len(), MAX_DIVERGENCES);
        assert_eq!(builder.dropped_divergences, 5);
        assert!(builder.divergences.iter().all(|divergence| {
            divergence.code.len() <= MAX_DIAGNOSTIC_BYTES
                && divergence.path.len() <= MAX_DIAGNOSTIC_BYTES
                && divergence.expected.len() <= MAX_DIAGNOSTIC_BYTES
                && divergence
                    .observed
                    .as_ref()
                    .is_none_or(|value| value.len() <= MAX_DIAGNOSTIC_BYTES)
                && divergence.retry.len() <= MAX_RETRY_BYTES
        }));
    }
}
