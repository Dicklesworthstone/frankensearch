//! Executable validation for the QG-2 symmetric in-memory comparator contract.
//!
//! The validator is intentionally read-only. It binds the six logical
//! contract surfaces, their nine physical locators, the typed TOML topology,
//! and the ten canonical unmeasured sentinels into one fresh-process receipt.
//!
//! Two of the six logical surfaces are one-to-many — the comprehensive-plan
//! surface holds two locators and the hyperopt surface three — and two files
//! host two locators each. Locator identity is therefore bounded by region, not
//! by file: a file carries exactly as many canonical clauses as it hosts
//! locators, every declared region carries exactly one, and regions declared
//! for the same file must stay disjoint and in declared document order.
//!
//! The file-wide census is deliberately whole-file: a stray clause parked
//! outside every bounded region invalidates *every* locator in that file, since
//! nothing in the file is trustworthy until the census is exact again. Per
//! locator, the divergence still names the one region that must be repaired.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    PerfGate, PerfGateArtifact, is_explicit_bootstrap, is_explicit_bootstrap_for,
    perf_manifest_contract_sha256,
};

/// Version of the fresh-process QG-2 contract validation report.
pub const QG2_CONTRACT_REPORT_SCHEMA_VERSION: &str = "frankensearch.quill-qg2-contract-report.v1";
/// Version of the fresh-process QG-2 bootstrap preflight report.
pub const QG2_PREFLIGHT_REPORT_SCHEMA_VERSION: &str = "frankensearch.quill-qg2-preflight-report.v1";
/// Fixed no-claim status bound into every preflight receipt.
pub const QG2_NO_CLAIM: &str = "Q2C binds contract identity only. It admits no performance evidence and authorizes no gate activation, target satisfaction, or speed claim.";
/// Exact normative QG-2 clause shared by every authoritative physical locator.
pub const QG2_CANONICAL_CONTRACT: &str = "BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 compares both arms symmetrically in memory with no durable storage. Continuous timing begins at the first document feed and ends only after terminal searchable visibility plus complete worker, merge, and queue quiescence. Commit is the searchable-visibility boundary, not durable publication. QG-2 excludes fsync, F_FULLFSYNC, crash recovery, durable publication, and on-disk-byte measurements. Durable gates and production-source durability nonregression remain mandatory outside QG-2.";
/// Number of independent normative QG-2 contract surfaces.
pub const QG2_LOGICAL_SURFACE_COUNT: usize = 6;
/// Number of concrete locators occupied by the six logical surfaces.
pub const QG2_PHYSICAL_LOCATOR_COUNT: usize = 9;
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
const PREFLIGHT_DRIFT_RETRY: &str = "Restore the protected bootstrap state at the named selector, or apply the canonical contract to every selector; a tree split across both fails closed.";
const PREFLIGHT_RENDER_RETRY: &str = "Fix the rendered QG-2 contract table so it parses back to the canonical typed contract, then rerun the preflight.";

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
    /// Logical surface identity. Two of the six logical surfaces are
    /// one-to-many: the comprehensive-plan surface holds two locators and the
    /// hyperopt surface three. The other four hold exactly one each.
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
    /// Exact six-logical/nine-physical topology summary.
    pub topology: Qg2TopologySummary,
    /// Normalized performance-manifest SHA-256.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<String>,
    /// Exact ten-sentinel summary.
    pub sentinels: Qg2SentinelSummary,
    /// Ordered receipts for all nine expected physical locators.
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
    /// Start anchor of the bounded region at the protected bootstrap base.
    ///
    /// Two selectors rename their law heading as part of the correction, so
    /// their PRE and POST start anchors are different, mutually exclusive
    /// strings. For the other three the anchor survives the correction and
    /// this equals `region_start`; those selectors are told apart purely by
    /// whether the bounded region carries the canonical clause.
    pre_region_start: &'static str,
    /// Start anchor of the bounded region once the correction is applied.
    region_start: &'static str,
    region_end: &'static str,
}

impl TextSurfaceSpec {
    /// Whether applying the correction also renames this region's start anchor.
    fn anchors_differ(self) -> bool {
        self.pre_region_start != self.region_start
    }
}

/// Sole locator of logical surface 1, in the performance-gate law list.
const PERF_GATES_GROUP: [TextSurfaceSpec; 1] = [TextSurfaceSpec {
    logical_surface: "performance_gate_law_1",
    locator: "perf_gate_law_1",
    path: PERF_GATES_DOC_PATH,
    pre_region_start: "1. **No benchmark-only semantics.**",
    region_start: "1. **No benchmark-only semantics; comparator scope is explicit.**",
    region_end: "2. **Distributions, not averages.**",
}];
/// Both locators of logical surface 2, in declared document order: the QG-2
/// gate row and the Method clause whose law 1 otherwise reads durable.
const PLAN_GROUP: [TextSurfaceSpec; 2] = [
    TextSurfaceSpec {
        logical_surface: "comprehensive_plan_qg2",
        locator: "comprehensive_plan_qg2_row",
        path: COMPREHENSIVE_PLAN_PATH,
        pre_region_start: "| **QG-2 Bulk indexing, single-thread**",
        region_start: "| **QG-2 Bulk indexing, single-thread**",
        region_end: "| **QG-3 Watch-mode incremental**",
    },
    TextSurfaceSpec {
        logical_surface: "comprehensive_plan_qg2",
        locator: "comprehensive_plan_method_law_1",
        path: COMPREHENSIVE_PLAN_PATH,
        pre_region_start: "Method: the five standing laws \u{2014}",
        region_start: "Method: the five standing laws \u{2014}",
        region_end: "## 15. The Conformance Gauntlet (Bet Q5)",
    },
];
/// The two document locators of logical surface 4, in declared document order:
/// campaign law 7 and the W2 commit-path fsync lever row. The third locator of
/// this surface is the epic's tracker note, validated with the other trackers.
const HYPEROPT_GROUP: [TextSurfaceSpec; 2] = [
    TextSurfaceSpec {
        logical_surface: "hyperopt_law_7_and_epic",
        locator: "hyperopt_law_7",
        path: HYPEROPT_DOC_PATH,
        pre_region_start: "7. **Platform-symmetric durability.**",
        region_start: "7. **QG-2 comparator scope and platform durability.**",
        region_end: "## 2. Hardware/profile matrix",
    },
    TextSurfaceSpec {
        logical_surface: "hyperopt_law_7_and_epic",
        locator: "hyperopt_w2_fsync_row",
        path: HYPEROPT_DOC_PATH,
        pre_region_start: "| Commit-path fsync count |",
        region_start: "| Commit-path fsync count |",
        region_end: "### W3 \u{2014} Parallel scale-out",
    },
];

/// Exact protected `[gate.QG-2]` block at the bootstrap base, verbatim from
/// protected commit `3f86ea57`.
///
/// Binding the whole block by bytes — not just the absence of a nested table —
/// is what stops a coordinated rewrite of `name`, `fixture`, `target`, or
/// `activated` from passing as the protected base while the typed contract
/// still parses.
pub const QG2_MANIFEST_BLOCK_PRE_REGION: &str = r#"[gate.QG-2]
name = "bulk indexing, single-thread"
fixture = "medium; positions ON; threads = 1; commit included"
target = "docs_per_sec >= 1.5x oracle"
activated = false

"#;

/// Exact protected `[gate.QG-2]` block once the correction is applied,
/// verbatim from the frozen candidate `4e136ac8`.
///
/// The projection is **not** only the nested table: the `fixture` string is
/// rewritten from "commit included" to the continuous first-feed-through-
/// quiescence scope, because a QG-2 fixture that still advertises commit-
/// inclusive durable framing contradicts the contract in the same block.
/// Rendering the table alone would leave that contradiction in place.
pub const QG2_MANIFEST_BLOCK_POST_REGION: &str = r#"[gate.QG-2]
name = "bulk indexing, single-thread"
fixture = "medium; positions ON; threads = 1; continuous first-feed through terminal searchable visibility and complete worker/merge/queue quiescence"
target = "docs_per_sec >= 1.5x oracle"
activated = false

[gate.QG-2.qg2_contract]
contract = "BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 compares both arms symmetrically in memory with no durable storage. Continuous timing begins at the first document feed and ends only after terminal searchable visibility plus complete worker, merge, and queue quiescence. Commit is the searchable-visibility boundary, not durable publication. QG-2 excludes fsync, F_FULLFSYNC, crash recovery, durable publication, and on-disk-byte measurements. Durable gates and production-source durability nonregression remain mandatory outside QG-2."
storage_topology = "symmetric_in_memory"
durability_scope = "non_durable"
timing_start = "first_document_feed"
timing_end = "terminal_searchable_visibility_and_complete_worker_merge_queue_quiescence"
commit_boundary = "searchable_visibility_not_durable_publication"
excluded_operations = ["fsync", "F_FULLFSYNC", "crash_recovery", "durable_publication", "on_disk_bytes"]
source_nonregression = "durable_gates_and_production_source_durability_remain_mandatory"

"#;

/// The three tracker selectors, as `(logical surface, locator, issue id)`.
///
/// Single source of truth for both the applied-state validator and the
/// bootstrap preflight, so the two can never bind different issue identities.
const TRACKER_SELECTORS: [(&str, &str, &str); 3] = [
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

/// Exact one-to-many grouping of the nine physical locators over the six
/// logical surfaces, in the sorted order the validator compares against.
const EXPECTED_LOGICAL_GROUPING: [(&str, usize); QG2_LOGICAL_SURFACE_COUNT] = [
    ("comprehensive_plan_qg2", 2),
    ("gate_activation_scope", 1),
    ("hyperopt_law_7_and_epic", 3),
    ("machine_manifest_qg2", 1),
    ("performance_gate_law_1", 1),
    ("qg2_r1_quarantine", 1),
];

#[derive(Debug, Deserialize)]
struct ManifestDocument {
    gate: BTreeMap<String, ManifestGate>,
}

/// One gate policy, closed against unknown fields.
///
/// A partial model would let an attacker park an unmodelled key inside
/// `[gate.QG-2]` that the preflight never sees but the live consumer does, so
/// the two could disagree about the same bytes. Every field the normative
/// manifest may legitimately carry is declared here and checked below.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestGate {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    fixture: Option<String>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    activated: Option<bool>,
    #[serde(default)]
    primary_target_cell_width: Option<u64>,
    #[serde(default)]
    queries_per_class: Option<u64>,
    #[serde(default)]
    qg2_contract: Option<Qg2ComparatorContract>,
}

impl ManifestGate {
    /// Field names this gate actually declares, for placement checking.
    fn declared_fields(&self) -> Vec<&'static str> {
        let mut declared = Vec::with_capacity(7);
        for (field, present) in [
            ("name", self.name.is_some()),
            ("fixture", self.fixture.is_some()),
            ("target", self.target.is_some()),
            ("activated", self.activated.is_some()),
            (
                "primary_target_cell_width",
                self.primary_target_cell_width.is_some(),
            ),
            ("queries_per_class", self.queries_per_class.is_some()),
            ("qg2_contract", self.qg2_contract.is_some()),
        ] {
            if present {
                declared.push(field);
            }
        }
        declared
    }

    /// Every scalar a normative gate must declare, regardless of gate.
    fn declares_required_scalars(&self) -> bool {
        self.name.is_some() && self.target.is_some() && self.activated.is_some()
    }

    /// Whether the typed view agrees with the byte-determined QG-2 state.
    ///
    /// This is the cross-check that keeps the byte layer and the typed layer
    /// from drifting apart: the parsed `fixture` must be the exact string the
    /// matched protected block carries, and the nested contract must be present
    /// exactly when that block is the applied one.
    fn agrees_with_qg2_block(&self, block: &str, applied: bool) -> bool {
        if !self.declares_required_scalars() {
            return false;
        }
        let fixture_agrees = self
            .fixture
            .as_deref()
            .is_some_and(|fixture| block.contains(&format!("fixture = \"{fixture}\"")));
        let contract_agrees = if applied {
            self.qg2_contract.as_ref() == Some(&Qg2ComparatorContract::canonical())
        } else {
            self.qg2_contract.is_none()
        };
        // QG-2 owns neither of these two gate-specific knobs.
        let scope_agrees =
            self.primary_target_cell_width.is_none() && self.queries_per_class.is_none();
        fixture_agrees && contract_agrees && scope_agrees
    }
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
        self.divergences.push(bounded_divergence(
            code,
            path,
            expected,
            observed,
            expected_sha256,
            observed_sha256,
            retry,
        ));
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
                "six configured logical surfaces mapped to nine unique physical locators",
                Some(&format!(
                    "{} logical surfaces, {} locators, {} receipts",
                    configured_logical.len(),
                    configured_locators.len(),
                    self.surfaces.len()
                )),
                "Restore the validator's fixed six-logical/nine-physical topology map.",
            );
        }
        // Keys are owned: `divergence` below reborrows `self` mutably, so the
        // grouping cannot keep borrowing `self.surfaces`.
        let configured_grouping =
            self.surfaces
                .iter()
                .fold(BTreeMap::<String, usize>::new(), |mut grouping, receipt| {
                    *grouping.entry(receipt.logical_surface.clone()).or_default() += 1;
                    grouping
                });
        let expected_grouping = EXPECTED_LOGICAL_GROUPING
            .into_iter()
            .map(|(surface, locators)| (surface.to_owned(), locators))
            .collect::<BTreeMap<String, usize>>();
        if configured_grouping != expected_grouping {
            self.divergence(
                "qg2.topology.logical_grouping",
                "qg2://validator-map",
                &format!("{expected_grouping:?}"),
                Some(&format!("{configured_grouping:?}")),
                "Restore the exact one-to-many grouping of the nine locators over six surfaces.",
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
    validate_text_group(repo_root, &PERF_GATES_GROUP, &mut report);
    validate_text_group(repo_root, &PLAN_GROUP, &mut report);
    validate_manifest_surface(repo_root, &mut report);
    validate_text_group(repo_root, &HYPEROPT_GROUP, &mut report);
    validate_tracker_surfaces(repo_root, &mut report);
    validate_sentinels(repo_root, &mut report);
    report.finish()
}

/// Validate every locator declared for one authoritative document.
///
/// The whole file must carry exactly as many canonical clauses as it hosts
/// locators, and each declared region exactly one, so a clause parked outside
/// every bounded region is a divergence rather than a silent pass. Regions are
/// additionally required to be disjoint and to appear in declared order, which
/// is what makes two locators in one file independently repairable.
fn validate_text_group(repo_root: &Path, group: &[TextSurfaceSpec], report: &mut ReportBuilder) {
    let Some(path) = group.first().map(|spec| spec.path) else {
        return;
    };
    let source = match fs::read_to_string(repo_root.join(path)) {
        Ok(source) => source,
        Err(error) => {
            report.divergence(
                "qg2.surface.read",
                path,
                QG2_CANONICAL_CONTRACT,
                Some(&error.to_string()),
                SURFACE_RETRY,
            );
            for spec in group {
                report
                    .surfaces
                    .push(surface_receipt(*spec, false, 0, None, false));
            }
            return;
        }
    };
    let file_marker_count = source.matches(QG2_CANONICAL_CONTRACT).count();
    let expected_marker_count = group.len();
    let file_markers_exact = file_marker_count == expected_marker_count;
    if !file_markers_exact {
        report.divergence(
            "qg2.surface.marker_count",
            path,
            &format!(
                "exactly {expected_marker_count} canonical Q2C clauses in the authoritative file"
            ),
            Some(&format!("{file_marker_count} canonical clauses")),
            SURFACE_RETRY,
        );
    }
    let mut preceding: Option<(usize, &'static str)> = None;
    for spec in group {
        let locator_path = format!("{path}#{}", spec.locator);
        let mut valid = file_markers_exact;
        let mut marker_count = 0;
        let content_sha256 = match unique_region(&source, spec.region_start, spec.region_end) {
            Ok(region) => {
                marker_count = region.text.matches(QG2_CANONICAL_CONTRACT).count();
                if marker_count != 1 {
                    valid = false;
                    report.divergence(
                        "qg2.surface.marker_scope",
                        &locator_path,
                        "exactly one canonical Q2C clause inside the named bounded region",
                        Some(&format!("{marker_count} canonical clauses in the region")),
                        SURFACE_RETRY,
                    );
                }
                if let Some((preceding_end, preceding_locator)) = preceding
                    && region.start < preceding_end
                {
                    valid = false;
                    report.divergence(
                        "qg2.surface.region_overlap",
                        &locator_path,
                        &format!("region starting at or after the end of {preceding_locator}"),
                        Some(&format!(
                            "region starts at byte {} but {preceding_locator} ends at byte {preceding_end}",
                            region.start
                        )),
                        SURFACE_RETRY,
                    );
                }
                preceding = Some((region.end, spec.locator));
                Some(sha256_hex(region.text.as_bytes()))
            }
            Err(error) => {
                valid = false;
                report.divergence(
                    "qg2.surface.region",
                    &locator_path,
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
            *spec,
            true,
            marker_count,
            content_sha256,
            valid,
        ));
    }
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

/// Byte, topology, and typed agreement every manifest must satisfy for the
/// byte-matched QG-2 block, shared by the applied validator and the preflight.
///
/// `applied` selects which protected block is authoritative. Beyond that block
/// this also closes two gaps the typed contract cannot see on its own: a gate
/// label the normative set does not define, and a `qg2_contract` table hung
/// under some *other* gate, where the live consumer would reject it as an
/// unexpected field while a QG-2-only check would never look.
fn manifest_block_agreement(source: &str, applied: bool) -> Result<(), String> {
    let (expected, other, expected_topology) = if applied {
        (
            QG2_MANIFEST_BLOCK_POST_REGION,
            QG2_MANIFEST_BLOCK_PRE_REGION,
            Qg2BlockTopology::Applied,
        )
    } else {
        (
            QG2_MANIFEST_BLOCK_PRE_REGION,
            QG2_MANIFEST_BLOCK_POST_REGION,
            Qg2BlockTopology::Bootstrap,
        )
    };
    let expected_count = source.matches(expected).count();
    let other_count = source.matches(other).count();
    if expected_count != 1 || other_count != 0 {
        return Err(format!(
            "expected exactly one protected block on the {} side; found {expected_count} expected \
             and {other_count} opposite blocks",
            if applied { "applied" } else { "bootstrap" }
        ));
    }
    match qg2_block_topology(source) {
        Ok(topology) if topology == expected_topology => {}
        Ok(topology) => return Err(format!("table ordering reports {topology:?}")),
        Err(error) => return Err(error),
    }

    let document = toml::from_str::<ManifestDocument>(source).map_err(|error| {
        format!("the manifest does not parse under the closed gate model: {error}")
    })?;
    manifest_topology_parity(&document)?;
    if document
        .gate
        .get("QG-2")
        .is_some_and(|gate| gate.agrees_with_qg2_block(expected, applied))
    {
        Ok(())
    } else {
        Err("the typed gate.QG-2 view disagrees with the byte-matched protected block".to_owned())
    }
}

/// Agreement for the applied side, used by the applied-state validator.
fn applied_manifest_block_agreement(source: &str) -> Result<(), String> {
    manifest_block_agreement(source, true)
}

/// Fields each normative gate may declare, mirroring the live consumer's
/// per-gate allowlist exactly.
///
/// The typed model is necessarily a *union* of every gate's fields, so without
/// this placement table a QG-1-only knob on QG-3 — or a QG-6 count on QG-2 —
/// parses cleanly here while the consumer rejects it as an unexpected field.
const fn gate_field_placement(gate: PerfGate) -> &'static [&'static str] {
    match gate {
        PerfGate::Qg1 => &[
            "name",
            "fixture",
            "target",
            "primary_target_cell_width",
            "activated",
        ],
        PerfGate::Qg2 => &["name", "fixture", "target", "activated", "qg2_contract"],
        PerfGate::Qg6 => &[
            "name",
            "fixture",
            "queries_per_class",
            "target",
            "activated",
        ],
        _ => &["name", "fixture", "target", "activated"],
    }
}

/// Require the same gate topology and field placement the live consumer does.
///
/// Byte-binding the QG-2 block says nothing about the other nine gates, so a
/// manifest could satisfy the Q2C selector and still be refused by planning.
/// These are the consumer's rules restated over the typed model: every
/// normative gate present, `name`/`fixture`/`target` non-empty, `activated`
/// boolean, the frozen QG-6 group count, no field outside its gate's
/// allowlist, and no gate label the normative set does not define.
fn manifest_topology_parity(document: &ManifestDocument) -> Result<(), String> {
    for gate in PerfGate::ALL {
        let label = gate.label();
        let Some(policy) = document.gate.get(label) else {
            return Err(format!("manifest gate.{label} is missing or not a table"));
        };
        for (field, value) in [
            ("name", &policy.name),
            ("fixture", &policy.fixture),
            ("target", &policy.target),
        ] {
            if !value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            {
                return Err(format!("manifest gate.{label}.{field} is missing or empty"));
            }
        }
        if policy.activated.is_none() {
            return Err(format!(
                "manifest gate.{label}.activated is missing or not boolean"
            ));
        }
        if gate == PerfGate::Qg6
            && policy
                .queries_per_class
                .and_then(|count| usize::try_from(count).ok())
                != Some(crate::QG6_QUERY_GROUPS)
        {
            return Err(format!(
                "manifest gate.{label}.queries_per_class must equal the frozen QG-6 group count {}",
                crate::QG6_QUERY_GROUPS
            ));
        }
        let allowed = gate_field_placement(gate);
        for field in policy.declared_fields() {
            if !allowed.contains(&field) {
                return Err(format!(
                    "manifest gate.{label} defines unexpected field {field}"
                ));
            }
        }
    }
    for label in document.gate.keys() {
        if !PerfGate::ALL.iter().any(|gate| gate.label() == label) {
            return Err(format!("manifest defines unexpected gate.{label}"));
        }
    }
    Ok(())
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

    // The typed contract alone cannot see a coordinated rewrite of `name`,
    // `fixture`, `target`, or `activated` sitting beside it, so the applied
    // state is bound to the exact protected projected block by bytes, by table
    // ordering, and by typed agreement — the same three checks the bootstrap
    // preflight applies, so the two can never disagree about one manifest.
    if let Err(detail) = applied_manifest_block_agreement(&source) {
        valid = false;
        report.divergence(
            "qg2.manifest.projected_block",
            "docs/contracts/quill-perf-gates.toml#gate.QG-2",
            "the exact protected projected [gate.QG-2] block",
            Some(&detail),
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
    let expected = TRACKER_SELECTORS;
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

// ---------------------------------------------------------------------------
// Bootstrap preflight: typed PRE contracts for the same nine selectors.
//
// `validate_qg2_contract` answers "is the correction applied and exact?". It
// cannot answer "is this tree the protected bootstrap base, ready to mutate?",
// because at the base every locator is legitimately clause-free and the nested
// TOML table is legitimately absent — which the applied-state validator can
// only report as divergence. The preflight below is the other half: expected
// bootstrap absence is a PASS, and only ambiguity, unexpected content, or a
// tree split across both states fails closed.
// ---------------------------------------------------------------------------

/// State of one physical selector relative to the Q2C correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2SelectorState {
    /// Exact protected bootstrap state. Expected absence, never an error.
    Bootstrap,
    /// Already carries the exact canonical contract.
    Applied,
    /// Neither exact bootstrap nor exact applied. Fails closed.
    Drift,
}

/// Terminal state of a whole-tree bootstrap preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg2PreflightState {
    /// Every selector is at its exact bootstrap contract; mutation may proceed.
    BootstrapReady,
    /// Every selector already carries the canonical contract; mutation is a
    /// no-op and re-running it would be the only way to change bytes.
    AlreadyApplied,
    /// Fail-closed: drift at a selector, or a tree split across both states.
    Drift,
}

/// Preflight receipt for one physical selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2SelectorReceipt {
    /// Logical surface identity, shared with the applied-state report.
    pub logical_surface: String,
    /// Unique physical locator identity.
    pub locator: String,
    /// Project-relative source path.
    pub path: String,
    /// Tracker issue identity when this is a Beads selector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue_id: Option<String>,
    /// Which side of the correction this selector is on.
    pub state: Qg2SelectorState,
    /// Digest of the bounded bootstrap region, when the selector is at PRE.
    ///
    /// Absent for a tracker selector at PRE: its bootstrap state is an *absent*
    /// note, which has no bytes to bind. Absence is the receipt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_sha256: Option<String>,
    /// Digest of the bounded applied region or note, when the selector is POST.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_sha256: Option<String>,
}

/// Manifest-digest rebinding one bootstrap sentinel would require.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2SentinelRebind {
    /// Project-relative sentinel path.
    pub path: String,
    /// Gate identity the sentinel must match.
    pub gate: PerfGate,
    /// Normalized manifest digest the sentinel carries now.
    pub bound_manifest_sha256: String,
    /// Normalized manifest digest of the rendered poststate manifest.
    pub rebound_manifest_sha256: String,
    /// Whether applying the correction changes this sentinel's one mutable
    /// field. False when the tree is already applied.
    pub rebind_required: bool,
}

/// Fresh-process bootstrap preflight report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qg2PreflightReport {
    /// Report wire schema.
    pub schema_version: String,
    /// Terminal state of the whole tree.
    pub state: Qg2PreflightState,
    /// Canonical typed comparator contract this preflight would render.
    pub contract: Qg2ComparatorContract,
    /// Fixed no-claim status; this receipt admits no performance evidence.
    pub no_claim: String,
    /// Ordered receipts for all nine expected physical selectors.
    pub selectors: Vec<Qg2SelectorReceipt>,
    /// Normalized manifest digest before the correction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_sha256_pre: Option<String>,
    /// Normalized manifest digest of the rendered poststate manifest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest_sha256_post: Option<String>,
    /// Per-sentinel manifest rebinding the correction implies.
    pub sentinel_rebinds: Vec<Qg2SentinelRebind>,
    /// Ordered, bounded divergences.
    pub divergences: Vec<Qg2ContractDivergence>,
    /// Number of additional divergences suppressed by the report bound.
    pub dropped_divergences: usize,
}

impl Qg2PreflightReport {
    /// Whether every selector is at its exact protected bootstrap contract.
    #[must_use]
    pub const fn is_bootstrap_ready(&self) -> bool {
        matches!(self.state, Qg2PreflightState::BootstrapReady)
    }
}

struct PreflightBuilder {
    selectors: Vec<Qg2SelectorReceipt>,
    manifest_sha256_pre: Option<String>,
    manifest_sha256_post: Option<String>,
    sentinel_rebinds: Vec<Qg2SentinelRebind>,
    divergences: Vec<Qg2ContractDivergence>,
    dropped_divergences: usize,
}

impl PreflightBuilder {
    fn new() -> Self {
        Self {
            selectors: Vec::with_capacity(QG2_PHYSICAL_LOCATOR_COUNT),
            manifest_sha256_pre: None,
            manifest_sha256_post: None,
            sentinel_rebinds: Vec::with_capacity(QG2_SENTINEL_COUNT),
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
        if self.divergences.len() == MAX_DIVERGENCES {
            self.dropped_divergences = self.dropped_divergences.saturating_add(1);
            return;
        }
        let expected_sha256 = sha256_hex(expected.as_bytes());
        let observed_sha256 = observed.map(|value| sha256_hex(value.as_bytes()));
        self.divergences.push(bounded_divergence(
            code,
            path,
            expected,
            observed,
            &expected_sha256,
            observed_sha256.as_deref(),
            retry,
        ));
    }

    fn finish(mut self) -> Qg2PreflightReport {
        let bootstrap = self
            .selectors
            .iter()
            .filter(|receipt| receipt.state == Qg2SelectorState::Bootstrap)
            .count();
        let applied = self
            .selectors
            .iter()
            .filter(|receipt| receipt.state == Qg2SelectorState::Applied)
            .count();
        if self.selectors.len() != QG2_PHYSICAL_LOCATOR_COUNT {
            self.divergence(
                "qg2.preflight.selector_count",
                "qg2://preflight",
                &QG2_PHYSICAL_LOCATOR_COUNT.to_string(),
                Some(&self.selectors.len().to_string()),
                PREFLIGHT_DRIFT_RETRY,
            );
        } else if self.divergences.is_empty()
            && bootstrap != QG2_PHYSICAL_LOCATOR_COUNT
            && applied != QG2_PHYSICAL_LOCATOR_COUNT
        {
            // Every selector is individually exact, but they disagree about
            // which side of the correction the tree is on. A half-applied tree
            // is precisely the state the mutation must never leave behind, so
            // it fails closed even though no single selector is at fault.
            self.divergence(
                "qg2.preflight.split_state",
                "qg2://preflight",
                &format!(
                    "all {QG2_PHYSICAL_LOCATOR_COUNT} selectors on one side of the correction"
                ),
                Some(&format!(
                    "{bootstrap} bootstrap, {applied} applied selectors"
                )),
                PREFLIGHT_DRIFT_RETRY,
            );
        }
        let state = if !self.divergences.is_empty() {
            Qg2PreflightState::Drift
        } else if bootstrap == QG2_PHYSICAL_LOCATOR_COUNT {
            Qg2PreflightState::BootstrapReady
        } else {
            Qg2PreflightState::AlreadyApplied
        };
        Qg2PreflightReport {
            schema_version: QG2_PREFLIGHT_REPORT_SCHEMA_VERSION.to_owned(),
            state,
            contract: Qg2ComparatorContract::canonical(),
            no_claim: QG2_NO_CLAIM.to_owned(),
            selectors: self.selectors,
            manifest_sha256_pre: self.manifest_sha256_pre,
            manifest_sha256_post: self.manifest_sha256_post,
            sentinel_rebinds: self.sentinel_rebinds,
            divergences: self.divergences,
            dropped_divergences: self.dropped_divergences,
        }
    }
}

/// Classify one repository against the typed bootstrap contract of all six
/// logical surfaces and nine physical selectors.
///
/// This function performs only read operations, and never mutates the tree,
/// the tracker, or the sentinels. It renders the poststate manifest in memory
/// and verifies that render round-trips back to the canonical typed contract,
/// so a rendering defect is caught before any mutation could consume it.
#[must_use]
pub fn validate_qg2_preflight(repo_root: &Path) -> Qg2PreflightReport {
    let mut builder = PreflightBuilder::new();
    preflight_text_group(repo_root, &PERF_GATES_GROUP, &mut builder);
    preflight_text_group(repo_root, &PLAN_GROUP, &mut builder);
    preflight_manifest(repo_root, &mut builder);
    preflight_text_group(repo_root, &HYPEROPT_GROUP, &mut builder);
    preflight_tracker(repo_root, &mut builder);
    preflight_sentinels(repo_root, &mut builder);
    builder.finish()
}

struct TextSelectorOutcome {
    state: Qg2SelectorState,
    pre_sha256: Option<String>,
    post_sha256: Option<String>,
    detail: Option<String>,
}

impl TextSelectorOutcome {
    fn drift(detail: String) -> Self {
        Self {
            state: Qg2SelectorState::Drift,
            pre_sha256: None,
            post_sha256: None,
            detail: Some(detail),
        }
    }

    fn bootstrap(region: BoundedRegion<'_>) -> Self {
        let count = region.text.matches(QG2_CANONICAL_CONTRACT).count();
        if count == 0 {
            Self {
                state: Qg2SelectorState::Bootstrap,
                pre_sha256: Some(sha256_hex(region.text.as_bytes())),
                post_sha256: None,
                detail: None,
            }
        } else {
            Self::drift(format!(
                "the bootstrap region already carries {count} canonical clauses; expected none"
            ))
        }
    }

    fn applied(region: BoundedRegion<'_>) -> Self {
        let count = region.text.matches(QG2_CANONICAL_CONTRACT).count();
        if count == 1 {
            Self {
                state: Qg2SelectorState::Applied,
                pre_sha256: None,
                post_sha256: Some(sha256_hex(region.text.as_bytes())),
                detail: None,
            }
        } else {
            Self::drift(format!(
                "the applied region carries {count} canonical clauses; expected exactly one"
            ))
        }
    }
}

fn classify_text_selector(source: &str, spec: TextSurfaceSpec) -> TextSelectorOutcome {
    if spec.anchors_differ() {
        // The correction renames this region's heading, so exactly one of the
        // two anchors may resolve. Both resolving is ambiguity, not progress.
        return match (
            unique_region(source, spec.pre_region_start, spec.region_end),
            unique_region(source, spec.region_start, spec.region_end),
        ) {
            (Ok(_), Ok(_)) => TextSelectorOutcome::drift(
                "both the bootstrap and applied start anchors resolve; the region is ambiguous"
                    .to_owned(),
            ),
            (Ok(region), Err(_)) => TextSelectorOutcome::bootstrap(region),
            (Err(_), Ok(region)) => TextSelectorOutcome::applied(region),
            (Err(bootstrap_error), Err(applied_error)) => TextSelectorOutcome::drift(format!(
                "neither anchor resolves: bootstrap ({bootstrap_error}); applied ({applied_error})"
            )),
        };
    }
    // The anchor survives the correction, so only the clause tells the states
    // apart: none is the protected base, exactly one is applied.
    match unique_region(source, spec.region_start, spec.region_end) {
        Ok(region) => match region.text.matches(QG2_CANONICAL_CONTRACT).count() {
            0 => TextSelectorOutcome::bootstrap(region),
            1 => TextSelectorOutcome::applied(region),
            count => TextSelectorOutcome::drift(format!(
                "the bounded region carries {count} canonical clauses; expected none or one"
            )),
        },
        Err(error) => TextSelectorOutcome::drift(error),
    }
}

fn preflight_text_group(
    repo_root: &Path,
    group: &[TextSurfaceSpec],
    builder: &mut PreflightBuilder,
) {
    let Some(path) = group.first().map(|spec| spec.path) else {
        return;
    };
    let source = match fs::read_to_string(repo_root.join(path)) {
        Ok(source) => source,
        Err(error) => {
            builder.divergence(
                "qg2.preflight.read",
                path,
                "readable authoritative document",
                Some(&error.to_string()),
                PREFLIGHT_DRIFT_RETRY,
            );
            for spec in group {
                builder.selectors.push(Qg2SelectorReceipt {
                    logical_surface: spec.logical_surface.to_owned(),
                    locator: spec.locator.to_owned(),
                    path: path.to_owned(),
                    issue_id: None,
                    state: Qg2SelectorState::Drift,
                    pre_sha256: None,
                    post_sha256: None,
                });
            }
            return;
        }
    };
    let outcomes = group
        .iter()
        .map(|spec| classify_text_selector(&source, *spec))
        .collect::<Vec<_>>();

    // The file may carry exactly one canonical clause per *applied* selector
    // and no others. At the protected base that means zero clauses anywhere,
    // so a clause parked outside every bounded region cannot slip through as
    // "expected bootstrap absence".
    let applied_in_file = outcomes
        .iter()
        .filter(|outcome| outcome.state == Qg2SelectorState::Applied)
        .count();
    let file_clause_count = source.matches(QG2_CANONICAL_CONTRACT).count();
    if file_clause_count != applied_in_file {
        builder.divergence(
            "qg2.preflight.file_census",
            path,
            &format!("exactly {applied_in_file} canonical Q2C clauses in the whole file"),
            Some(&format!("{file_clause_count} canonical clauses")),
            PREFLIGHT_DRIFT_RETRY,
        );
    }

    for (spec, outcome) in group.iter().zip(outcomes) {
        if let Some(detail) = outcome.detail.as_deref() {
            builder.divergence(
                "qg2.preflight.selector_drift",
                &format!("{path}#{}", spec.locator),
                "a clause-free bootstrap region, or an applied region with exactly one clause",
                Some(detail),
                PREFLIGHT_DRIFT_RETRY,
            );
        }
        builder.selectors.push(Qg2SelectorReceipt {
            logical_surface: spec.logical_surface.to_owned(),
            locator: spec.locator.to_owned(),
            path: path.to_owned(),
            issue_id: None,
            state: outcome.state,
            pre_sha256: outcome.pre_sha256,
            post_sha256: outcome.post_sha256,
        });
    }
}

/// Where the typed QG-2 table sits relative to the `[gate.QG-2]` block.
///
/// Byte equality against the two protected blocks already decides the state.
/// This is the independent structural corroboration: the exact-byte match must
/// also agree with the document's table ordering, so a protected block pasted
/// into the wrong position — or duplicated under a foreign header — cannot be
/// admitted on byte equality alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qg2BlockTopology {
    /// `[gate.QG-2]` is followed immediately by `[gate.QG-3]`: the base.
    Bootstrap,
    /// `[gate.QG-2]` is followed immediately by its nested contract table.
    Applied,
}

fn qg2_block_topology(source: &str) -> Result<Qg2BlockTopology, String> {
    const QG2_HEADER: &str = "[gate.QG-2]";
    const QG3_HEADER: &str = "[gate.QG-3]";
    const CONTRACT_HEADER: &str = "[gate.QG-2.qg2_contract]";

    let mut seen_qg2 = false;
    let mut following: Option<String> = None;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == QG2_HEADER {
            if seen_qg2 {
                return Err(format!("{QG2_HEADER} appears more than once"));
            }
            seen_qg2 = true;
        } else if seen_qg2 && following.is_none() && trimmed.starts_with('[') {
            following = Some(trimmed.to_owned());
        }
    }
    if !seen_qg2 {
        return Err(format!("{QG2_HEADER} is absent"));
    }
    match following {
        Some(header) if header == QG3_HEADER => Ok(Qg2BlockTopology::Bootstrap),
        Some(header) if header == CONTRACT_HEADER => Ok(Qg2BlockTopology::Applied),
        Some(header) => Err(format!(
            "{QG2_HEADER} is followed by {header}, not {QG3_HEADER}"
        )),
        None => Err(format!("{QG2_HEADER} is followed by no table header")),
    }
}

fn preflight_manifest(repo_root: &Path, builder: &mut PreflightBuilder) {
    let source = match fs::read_to_string(repo_root.join(PERF_MANIFEST_PATH)) {
        Ok(source) => source,
        Err(error) => {
            builder.divergence(
                "qg2.preflight.read",
                PERF_MANIFEST_PATH,
                "readable typed QG-2 manifest",
                Some(&error.to_string()),
                PREFLIGHT_DRIFT_RETRY,
            );
            builder.selectors.push(Qg2SelectorReceipt {
                logical_surface: "machine_manifest_qg2".to_owned(),
                locator: "perf_manifest_qg2_contract".to_owned(),
                path: PERF_MANIFEST_PATH.to_owned(),
                issue_id: None,
                state: Qg2SelectorState::Drift,
                pre_sha256: None,
                post_sha256: None,
            });
            return;
        }
    };
    builder.manifest_sha256_pre = Some(perf_manifest_contract_sha256(&source));

    let canonical = Qg2ComparatorContract::canonical();
    let mut state = Qg2SelectorState::Drift;
    let mut pre_sha256 = None;
    let mut post_sha256 = None;
    let mut rendered: Option<String> = None;
    let mut drift: Option<(&'static str, String)> = None;

    // Byte identity, not inference. The block is the selector, so `name`,
    // `fixture`, `target`, and `activated` are all bound; nothing about the
    // block can be rewritten while still classifying as a protected base.
    let bootstrap_blocks = source.matches(QG2_MANIFEST_BLOCK_PRE_REGION).count();
    let applied_blocks = source.matches(QG2_MANIFEST_BLOCK_POST_REGION).count();
    match (bootstrap_blocks, applied_blocks) {
        (1, 0) if qg2_block_topology(&source) == Ok(Qg2BlockTopology::Bootstrap) => {
            state = Qg2SelectorState::Bootstrap;
            pre_sha256 = Some(sha256_hex(QG2_MANIFEST_BLOCK_PRE_REGION.as_bytes()));
            rendered = Some(source.replacen(
                QG2_MANIFEST_BLOCK_PRE_REGION,
                QG2_MANIFEST_BLOCK_POST_REGION,
                1,
            ));
        }
        (0, 1) if qg2_block_topology(&source) == Ok(Qg2BlockTopology::Applied) => {
            state = Qg2SelectorState::Applied;
            post_sha256 = Some(sha256_hex(QG2_MANIFEST_BLOCK_POST_REGION.as_bytes()));
            rendered = Some(source.clone());
        }
        (bootstrap, applied) => {
            drift = Some((
                "qg2.preflight.manifest_conflict",
                format!(
                    "expected exactly one protected [gate.QG-2] block on one side of the \
                     correction; found {bootstrap} bootstrap and {applied} applied blocks"
                ),
            ));
        }
    }

    // Exactly the agreement the applied validator applies, so one manifest can
    // never be a pass to one reader and drift to the other.
    if drift.is_none()
        && let Err(detail) = manifest_block_agreement(&source, state == Qg2SelectorState::Applied)
    {
        state = Qg2SelectorState::Drift;
        drift = Some(("qg2.preflight.manifest_typed_disagreement", detail));
    }

    // Never hand a rendered poststate to a mutation without proving it parses
    // back to the exact typed contract it claims to encode.
    if let Some(candidate) = rendered.as_deref() {
        match toml::from_str::<ManifestDocument>(candidate) {
            Ok(document)
                if document
                    .gate
                    .get("QG-2")
                    .and_then(|gate| gate.qg2_contract.as_ref())
                    == Some(&canonical) =>
            {
                builder.manifest_sha256_post = Some(perf_manifest_contract_sha256(candidate));
            }
            Ok(_) => {
                state = Qg2SelectorState::Drift;
                drift = Some((
                    "qg2.preflight.render_roundtrip",
                    "the rendered manifest does not parse back to the canonical typed contract"
                        .to_owned(),
                ));
            }
            Err(error) => {
                state = Qg2SelectorState::Drift;
                drift = Some((
                    "qg2.preflight.render_roundtrip",
                    format!("the rendered manifest does not parse: {error}"),
                ));
            }
        }
    }

    if let Some((code, detail)) = drift {
        let retry = if code == "qg2.preflight.render" || code == "qg2.preflight.render_roundtrip" {
            PREFLIGHT_RENDER_RETRY
        } else {
            PREFLIGHT_DRIFT_RETRY
        };
        builder.divergence(
            code,
            "docs/contracts/quill-perf-gates.toml#gate.QG-2",
            "either the protected [gate.QG-2] block followed immediately by [gate.QG-3], or the exact canonical nested table",
            Some(&detail),
            retry,
        );
        state = Qg2SelectorState::Drift;
    }

    builder.selectors.push(Qg2SelectorReceipt {
        logical_surface: "machine_manifest_qg2".to_owned(),
        locator: "perf_manifest_qg2_contract".to_owned(),
        path: PERF_MANIFEST_PATH.to_owned(),
        issue_id: None,
        state,
        pre_sha256,
        post_sha256,
    });
}

fn preflight_tracker(repo_root: &Path, builder: &mut PreflightBuilder) {
    let source = match fs::read_to_string(repo_root.join(TRACKER_PATH)) {
        Ok(source) => source,
        Err(error) => {
            builder.divergence(
                "qg2.preflight.read",
                TRACKER_PATH,
                "readable tracker JSONL",
                Some(&error.to_string()),
                PREFLIGHT_DRIFT_RETRY,
            );
            for (logical_surface, locator, issue_id) in TRACKER_SELECTORS {
                builder.selectors.push(Qg2SelectorReceipt {
                    logical_surface: logical_surface.to_owned(),
                    locator: locator.to_owned(),
                    path: TRACKER_PATH.to_owned(),
                    issue_id: Some(issue_id.to_owned()),
                    state: Qg2SelectorState::Drift,
                    pre_sha256: None,
                    post_sha256: None,
                });
            }
            return;
        }
    };

    let expected_ids = TRACKER_SELECTORS
        .iter()
        .map(|(_, _, issue_id)| *issue_id)
        .collect::<BTreeSet<_>>();
    let mut selected = BTreeMap::<String, Vec<TrackerIssue>>::new();
    for (line_index, line) in source.lines().enumerate() {
        match serde_json::from_str::<TrackerIssue>(line) {
            Ok(issue) => {
                let marker_count = issue.active_marker_count();
                if marker_count > 0 && !expected_ids.contains(issue.id.as_str()) {
                    builder.divergence(
                        "qg2.preflight.tracker_extra_surface",
                        &format!("{TRACKER_PATH}#{}", issue.id),
                        "the canonical clause only in the three protected tracker selectors",
                        Some(&format!("{marker_count} canonical clauses")),
                        PREFLIGHT_DRIFT_RETRY,
                    );
                }
                if expected_ids.contains(issue.id.as_str()) {
                    selected.entry(issue.id.clone()).or_default().push(issue);
                }
            }
            Err(error) => builder.divergence(
                "qg2.preflight.tracker_jsonl",
                &format!("{TRACKER_PATH}:{}", line_index + 1),
                "one valid tracker JSON object",
                Some(&error.to_string()),
                PREFLIGHT_DRIFT_RETRY,
            ),
        }
    }

    for (logical_surface, locator, issue_id) in TRACKER_SELECTORS {
        let matches = selected.get(issue_id).map_or(&[][..], Vec::as_slice);
        let (state, post_sha256, detail) = if matches.len() == 1 {
            match matches[0].notes.as_deref() {
                // Absent or empty notes are the protected base, exactly as the
                // bootstrap contract specifies. This is not an error.
                None | Some("") => (Qg2SelectorState::Bootstrap, None, None),
                Some(note) if note == QG2_CANONICAL_CONTRACT => (
                    Qg2SelectorState::Applied,
                    Some(sha256_hex(note.as_bytes())),
                    None,
                ),
                Some(note) => (
                    Qg2SelectorState::Drift,
                    None,
                    Some(format!(
                        "unexpected nonempty notes of {} bytes at the protected base",
                        note.len()
                    )),
                ),
            }
        } else {
            (
                Qg2SelectorState::Drift,
                None,
                Some(format!(
                    "{} tracker objects; expected exactly one",
                    matches.len()
                )),
            )
        };
        if let Some(detail) = detail.as_deref() {
            builder.divergence(
                "qg2.preflight.selector_drift",
                &format!("{TRACKER_PATH}#{issue_id}.notes"),
                "exactly one issue whose notes are absent, or exactly the canonical contract",
                Some(detail),
                PREFLIGHT_DRIFT_RETRY,
            );
        }
        builder.selectors.push(Qg2SelectorReceipt {
            logical_surface: logical_surface.to_owned(),
            locator: locator.to_owned(),
            path: TRACKER_PATH.to_owned(),
            issue_id: Some(issue_id.to_owned()),
            state,
            pre_sha256: None,
            post_sha256,
        });
    }
}

fn preflight_sentinels(repo_root: &Path, builder: &mut PreflightBuilder) {
    let (Some(bound), Some(rebound)) = (
        builder.manifest_sha256_pre.clone(),
        builder.manifest_sha256_post.clone(),
    ) else {
        builder.divergence(
            "qg2.preflight.sentinel_manifest_unavailable",
            HISTORY_DIRECTORY,
            "both manifest digests from a readable, renderable manifest",
            None,
            PREFLIGHT_RENDER_RETRY,
        );
        return;
    };
    for gate in PerfGate::ALL {
        let path = format!(
            "{HISTORY_DIRECTORY}/{}.unmeasured.latest.json",
            gate.label()
        );
        let bytes = match fs::read(repo_root.join(&path)) {
            Ok(bytes) => bytes,
            Err(error) => {
                builder.divergence(
                    "qg2.preflight.sentinel_read",
                    &path,
                    "readable canonical bootstrap sentinel",
                    Some(&error.to_string()),
                    SENTINEL_RETRY,
                );
                continue;
            }
        };
        let artifact = match serde_json::from_slice::<PerfGateArtifact>(&bytes) {
            Ok(artifact) => artifact,
            Err(error) => {
                builder.divergence(
                    "qg2.preflight.sentinel_parse",
                    &path,
                    "current-schema bootstrap sentinel JSON",
                    Some(&error.to_string()),
                    SENTINEL_RETRY,
                );
                continue;
            }
        };
        if !is_explicit_bootstrap_for(&artifact, gate, &bound) {
            builder.divergence(
                "qg2.preflight.sentinel_binding",
                &path,
                &format!(
                    "an exact {} bootstrap sentinel bound to {bound}",
                    gate.label()
                ),
                Some(&format!(
                    "gate={:?} bound to {}",
                    artifact.gate, artifact.manifest_sha256
                )),
                SENTINEL_RETRY,
            );
        }
        let rebind_required = artifact.manifest_sha256 != rebound;
        builder.sentinel_rebinds.push(Qg2SentinelRebind {
            path,
            gate,
            bound_manifest_sha256: artifact.manifest_sha256,
            rebound_manifest_sha256: rebound.clone(),
            rebind_required,
        });
    }
}

/// One bounded locator region resolved inside its authoritative document.
#[derive(Debug, Clone, Copy)]
struct BoundedRegion<'a> {
    /// Region bytes, from the start anchor up to the end anchor.
    text: &'a str,
    /// Byte offset of the start anchor inside the whole document.
    start: usize,
    /// Byte offset immediately past the region, where the end anchor begins.
    end: usize,
}

fn unique_region<'a>(source: &'a str, start: &str, end: &str) -> Result<BoundedRegion<'a>, String> {
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
    Ok(BoundedRegion {
        text: &tail[..end_offset],
        start: start_offset,
        end: start_offset.saturating_add(end_offset),
    })
}

/// Build one divergence with every field clamped to its diagnostic bound.
#[allow(clippy::too_many_arguments)]
fn bounded_divergence(
    code: &str,
    path: &str,
    expected: &str,
    observed: Option<&str>,
    expected_sha256: &str,
    observed_sha256: Option<&str>,
    retry: &str,
) -> Qg2ContractDivergence {
    Qg2ContractDivergence {
        code: bounded(code, MAX_DIAGNOSTIC_BYTES),
        path: bounded(path, MAX_DIAGNOSTIC_BYTES),
        expected: bounded(expected, MAX_DIAGNOSTIC_BYTES),
        observed: observed.map(|value| bounded(value, MAX_DIAGNOSTIC_BYTES)),
        expected_sha256: bounded(expected_sha256, 64),
        observed_sha256: observed_sha256.map(|value| bounded(value, 64)),
        retry: bounded(retry, MAX_RETRY_BYTES),
    }
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

    /// Plan document carrying both declared locators of logical surface 2, in
    /// declared document order.
    fn plan_document() -> String {
        format!(
            "| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |\n\
             | **QG-3 Watch-mode incremental** | next |\n\
             \n\
             Method: the five standing laws \u{2014} (1) no benchmark-only semantics. \
             {QG2_CANONICAL_CONTRACT}\n\
             \n\
             ## 15. The Conformance Gauntlet (Bet Q5)\n"
        )
    }

    /// Hyperopt document carrying both of that surface's document locators.
    ///
    /// Law 7's bounded region is byte-identical to the single-locator fixture
    /// it replaces, so adding the W2 row cannot perturb law 7's receipt hash.
    fn hyperopt_document() -> String {
        format!(
            "7. **QG-2 comparator scope and platform durability.** {QG2_CANONICAL_CONTRACT}\n\
             ## 2. Hardware/profile matrix\n\
             \n\
             ### W2 \u{2014} Bulk-index single-thread cost (QG-1 and QG-2)\n\
             \n\
             | Commit-path fsync count | batch directory syncs | {QG2_CANONICAL_CONTRACT} |\n\
             \n\
             ### W3 \u{2014} Parallel scale-out\n"
        )
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
        fs::write(root.join(COMPREHENSIVE_PLAN_PATH), plan_document()).expect("plan fixture");
        fs::write(root.join(HYPEROPT_DOC_PATH), hyperopt_document()).expect("hyperopt fixture");

        // The applied fixture carries the *protected* projected block verbatim,
        // so the applied-state tests bind the same bytes the live tree must
        // reach rather than a hand-written lookalike.
        let manifest = applied_manifest();
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
            "../../../.bench-history/QG-1.v7.unmeasured.latest.json"
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
    fn complete_fixture_has_exact_six_by_nine_topology_and_ten_sentinels() {
        let fixture = complete_fixture();
        let report = validate_qg2_contract(fixture.root());
        assert!(report.is_pass(), "{:#?}", report.divergences);
        assert_eq!(report.topology.expected_logical_surfaces, 6);
        assert_eq!(report.topology.discovered_logical_surfaces, 6);
        assert_eq!(report.topology.expected_physical_locators, 9);
        assert_eq!(report.topology.discovered_physical_locators, 9);
        assert_eq!(report.topology.validated_physical_locators, 9);
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

    fn has_divergence(report: &Qg2ContractReport, code: &str, path_suffix: &str) -> bool {
        report
            .divergences
            .iter()
            .any(|divergence| divergence.code == code && divergence.path.ends_with(path_suffix))
    }

    fn locator_receipt<'a>(report: &'a Qg2ContractReport, locator: &str) -> &'a Qg2SurfaceReceipt {
        let missing = format!("report must carry a receipt for {locator}");
        report
            .surfaces
            .iter()
            .find(|receipt| receipt.locator == locator)
            .expect(&missing)
    }

    #[test]
    fn report_binds_the_nine_protected_locators_in_declared_order() {
        let fixture = complete_fixture();
        let report = validate_qg2_contract(fixture.root());
        assert!(report.is_pass(), "{:#?}", report.divergences);
        assert_eq!(
            report
                .surfaces
                .iter()
                .map(|receipt| (receipt.logical_surface.as_str(), receipt.locator.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("performance_gate_law_1", "perf_gate_law_1"),
                ("comprehensive_plan_qg2", "comprehensive_plan_qg2_row"),
                ("comprehensive_plan_qg2", "comprehensive_plan_method_law_1"),
                ("machine_manifest_qg2", "perf_manifest_qg2_contract"),
                ("hyperopt_law_7_and_epic", "hyperopt_law_7"),
                ("hyperopt_law_7_and_epic", "hyperopt_w2_fsync_row"),
                ("hyperopt_law_7_and_epic", "hyperopt_epic_active_contract"),
                ("qg2_r1_quarantine", "qg2_r1_active_contract"),
                ("gate_activation_scope", "gate_activation_active_contract"),
            ]
        );
        let grouping =
            report
                .surfaces
                .iter()
                .fold(BTreeMap::<&str, usize>::new(), |mut grouping, receipt| {
                    *grouping
                        .entry(receipt.logical_surface.as_str())
                        .or_default() += 1;
                    grouping
                });
        assert_eq!(
            grouping,
            BTreeMap::from([
                ("comprehensive_plan_qg2", 2),
                ("gate_activation_scope", 1),
                ("hyperopt_law_7_and_epic", 3),
                ("machine_manifest_qg2", 1),
                ("performance_gate_law_1", 1),
                ("qg2_r1_quarantine", 1),
            ])
        );
        assert!(
            report
                .surfaces
                .iter()
                .all(|receipt| receipt.discovered && receipt.valid && receipt.marker_count == 1)
        );
    }

    #[test]
    fn missing_and_extra_contract_markers_fail_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            plan_document().replacen(QG2_CANONICAL_CONTRACT, "missing", 1),
        )
        .expect("missing marker mutation");
        let missing = validate_qg2_contract(fixture.root());
        assert!(missing.divergences.iter().any(|divergence| {
            divergence.code == "qg2.surface.marker_count"
                && divergence.path == COMPREHENSIVE_PLAN_PATH
        }));
        assert!(has_divergence(
            &missing,
            "qg2.surface.marker_scope",
            "#comprehensive_plan_qg2_row"
        ));

        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            plan_document().replacen(
                QG2_CANONICAL_CONTRACT,
                &format!("{QG2_CANONICAL_CONTRACT} {QG2_CANONICAL_CONTRACT}"),
                1,
            ),
        )
        .expect("extra marker mutation");
        let extra = validate_qg2_contract(fixture.root());
        assert!(extra.divergences.iter().any(|divergence| {
            divergence.code == "qg2.surface.marker_count"
                && divergence.path == COMPREHENSIVE_PLAN_PATH
        }));
        assert!(has_divergence(
            &extra,
            "qg2.surface.marker_scope",
            "#comprehensive_plan_qg2_row"
        ));
    }

    #[test]
    fn plan_method_law_locator_is_mandatory() {
        let fixture = complete_fixture();
        let method_clause_removed = plan_document()
            .rsplit_once(QG2_CANONICAL_CONTRACT)
            .map(|(head, tail)| {
                format!("{head}durability settings identical to shipped defaults{tail}")
            })
            .expect("plan fixture carries the method clause");
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            &method_clause_removed,
        )
        .expect("method clause mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(!report.is_pass());
        assert!(has_divergence(
            &report,
            "qg2.surface.marker_scope",
            "#comprehensive_plan_method_law_1"
        ));
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.surface.marker_count"
                    && divergence.path == COMPREHENSIVE_PLAN_PATH)
        );
        assert!(!locator_receipt(&report, "comprehensive_plan_method_law_1").valid);
        assert_eq!(
            locator_receipt(&report, "comprehensive_plan_method_law_1").marker_count,
            0
        );

        // A failed file-wide census invalidates BOTH locators hosted by that
        // file, by design: until the census is exact again, nothing in the file
        // is trustworthy, so `validated` drops by two rather than by one. The
        // per-locator diagnostics stay precise underneath that — only the
        // Method region is named — which is what keeps the repair targeted
        // without letting the neighbour's receipt claim validity meanwhile.
        assert!(!locator_receipt(&report, "comprehensive_plan_qg2_row").valid);
        assert_eq!(report.topology.validated_physical_locators, 7);
        assert!(
            !has_divergence(
                &report,
                "qg2.surface.marker_scope",
                "#comprehensive_plan_qg2_row"
            ),
            "the region diagnostic must name the Method locator, not its file neighbour"
        );
        assert_eq!(
            locator_receipt(&report, "comprehensive_plan_qg2_row").marker_count,
            1
        );
        assert!(locator_receipt(&report, "perf_gate_law_1").valid);
    }

    #[test]
    fn hyperopt_w2_fsync_row_locator_is_mandatory() {
        let fixture = complete_fixture();
        let row_clause_removed = hyperopt_document()
            .rsplit_once(QG2_CANONICAL_CONTRACT)
            .map(|(head, tail)| format!("{head}Law 7 on macOS.{tail}"))
            .expect("hyperopt fixture carries the W2 row clause");
        fs::write(fixture.path(HYPEROPT_DOC_PATH), &row_clause_removed)
            .expect("W2 fsync row mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(!report.is_pass());
        assert!(has_divergence(
            &report,
            "qg2.surface.marker_scope",
            "#hyperopt_w2_fsync_row"
        ));
        assert!(!locator_receipt(&report, "hyperopt_w2_fsync_row").valid);

        // Same file-wide census semantics as the plan surface: both document
        // locators of this file go invalid, while the third locator of the same
        // *logical* surface — the epic's tracker note, which lives in another
        // file — is untouched. That is the line between file census and
        // logical-surface grouping.
        assert!(!locator_receipt(&report, "hyperopt_law_7").valid);
        assert!(locator_receipt(&report, "hyperopt_epic_active_contract").valid);
        assert_eq!(report.topology.validated_physical_locators, 7);
    }

    #[test]
    fn clause_outside_every_bounded_region_fails_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            format!("{}\nAppendix. {QG2_CANONICAL_CONTRACT}\n", plan_document()),
        )
        .expect("out-of-region clause mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(!report.is_pass());
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.surface.marker_count"
                    && divergence.path == COMPREHENSIVE_PLAN_PATH
                    && divergence.observed.as_deref() == Some("3 canonical clauses"))
        );
        assert!(
            !report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.surface.marker_scope"),
            "both bounded regions still hold exactly one clause; only the file census can catch a stray"
        );
    }

    #[test]
    fn duplicate_clause_inside_one_region_fails_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(HYPEROPT_DOC_PATH),
            hyperopt_document().replace(
                &format!("| {QG2_CANONICAL_CONTRACT} |"),
                &format!("| {QG2_CANONICAL_CONTRACT} {QG2_CANONICAL_CONTRACT} |"),
            ),
        )
        .expect("duplicate region clause mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(has_divergence(
            &report,
            "qg2.surface.marker_scope",
            "#hyperopt_w2_fsync_row"
        ));
        assert_eq!(
            locator_receipt(&report, "hyperopt_w2_fsync_row").marker_count,
            2
        );
        assert!(
            !has_divergence(&report, "qg2.surface.marker_scope", "#hyperopt_law_7"),
            "the duplicate must be charged to the W2 row, not to law 7"
        );
        assert_eq!(locator_receipt(&report, "hyperopt_law_7").marker_count, 1);
    }

    #[test]
    fn reordered_plan_regions_fail_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            format!(
                "Method: the five standing laws \u{2014} (1) no benchmark-only semantics. \
                 {QG2_CANONICAL_CONTRACT}\n\
                 \n\
                 | **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |\n\
                 | **QG-3 Watch-mode incremental** | next |\n\
                 \n\
                 ## 15. The Conformance Gauntlet (Bet Q5)\n"
            ),
        )
        .expect("reordered region mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(!report.is_pass());
        assert!(has_divergence(
            &report,
            "qg2.surface.region_overlap",
            "#comprehensive_plan_method_law_1"
        ));
        assert!(!locator_receipt(&report, "comprehensive_plan_method_law_1").valid);
    }

    #[test]
    fn missing_region_anchor_fails_closed() {
        let fixture = complete_fixture();
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            plan_document().replace(
                "## 15. The Conformance Gauntlet (Bet Q5)",
                "## 15. Gauntlet",
            ),
        )
        .expect("missing end anchor mutation");

        let report = validate_qg2_contract(fixture.root());
        assert!(!report.is_pass());
        assert!(has_divergence(
            &report,
            "qg2.surface.region",
            "#comprehensive_plan_method_law_1"
        ));
        assert!(locator_receipt(&report, "comprehensive_plan_qg2_row").valid);
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

    /// Manifest at the protected base: `[gate.QG-2]` followed immediately by
    /// `[gate.QG-3]`, with no nested contract table.
    /// A manifest carrying every normative gate, with the given protected block
    /// verbatim in the QG-2 position.
    ///
    /// The fixture must satisfy the same topology the live consumer requires —
    /// all ten gates, non-empty scalars, the frozen QG-6 count — or the tests
    /// would be exercising a manifest planning would refuse.
    fn manifest_with_qg2_block(block: &str) -> String {
        let mut manifest = String::new();
        for gate in PerfGate::ALL {
            if gate == PerfGate::Qg2 {
                manifest.push_str(block);
                continue;
            }
            let label = gate.label();
            let _ = write!(
                &mut manifest,
                "[gate.{label}]\nname = \"{label} gate\"\nfixture = \"{label} fixture\"\n\
                 target = \"{label} target\"\n"
            );
            if gate == PerfGate::Qg6 {
                let _ = writeln!(
                    &mut manifest,
                    "queries_per_class = {}",
                    crate::QG6_QUERY_GROUPS
                );
            }
            manifest.push_str("activated = false\n\n");
        }
        manifest
    }

    /// Manifest at the protected base. The QG-2 block is the protected block
    /// verbatim, not a lookalike, so the fixture exercises the same byte
    /// identity the live tree must have.
    fn bootstrap_manifest() -> String {
        manifest_with_qg2_block(QG2_MANIFEST_BLOCK_PRE_REGION)
    }

    /// The same manifest once the protected projection is applied.
    fn applied_manifest() -> String {
        manifest_with_qg2_block(QG2_MANIFEST_BLOCK_POST_REGION)
    }

    /// The three documents at the protected base: both renamed law headings
    /// still carry their bootstrap spelling and no region carries a clause.
    fn bootstrap_fixture() -> Fixture {
        let directory = tempfile::tempdir().expect("temporary repository");
        let root = directory.path();
        fs::create_dir_all(root.join("docs/contracts")).expect("contract directory");
        fs::create_dir_all(root.join(".beads")).expect("tracker directory");
        fs::create_dir_all(root.join(HISTORY_DIRECTORY)).expect("history directory");

        fs::write(
            root.join(PERF_GATES_DOC_PATH),
            concat!(
                "# Laws\n",
                "1. **No benchmark-only semantics.** Durability settings, commits, and result \
                 consumption match shipped defaults.\n",
                "2. **Distributions, not averages.** next\n",
            ),
        )
        .expect("bootstrap performance gate fixture");
        fs::write(
            root.join(COMPREHENSIVE_PLAN_PATH),
            concat!(
                "| **QG-2 Bulk indexing, single-thread** | >= 1.5x tantivy |\n",
                "| **QG-3 Watch-mode incremental** | next |\n",
                "\n",
                "Method: the five standing laws \u{2014} (1) no benchmark-only semantics.\n",
                "\n",
                "## 15. The Conformance Gauntlet (Bet Q5)\n",
            ),
        )
        .expect("bootstrap plan fixture");
        fs::write(
            root.join(HYPEROPT_DOC_PATH),
            concat!(
                "7. **Platform-symmetric durability.** On macOS a commit number is admissible \
                 only with F_FULLFSYNC attested symmetric.\n",
                "## 2. Hardware/profile matrix\n",
                "\n",
                "### W2 \u{2014} Bulk-index single-thread cost (QG-1 and QG-2)\n",
                "\n",
                "| Commit-path fsync count | batch directory syncs | census first. |\n",
                "\n",
                "### W3 \u{2014} Parallel scale-out\n",
            ),
        )
        .expect("bootstrap hyperopt fixture");

        let manifest = bootstrap_manifest();
        fs::write(root.join(PERF_MANIFEST_PATH), &manifest).expect("bootstrap manifest fixture");

        let mut tracker = String::new();
        for (_, _, issue_id) in TRACKER_SELECTORS {
            tracker.push_str(
                &serde_json::to_string(&json!({ "id": issue_id })).expect("tracker issue"),
            );
            tracker.push('\n');
        }
        fs::write(root.join(TRACKER_PATH), tracker).expect("bootstrap tracker fixture");

        let manifest_sha256 = perf_manifest_contract_sha256(&manifest);
        let template = serde_json::from_str::<PerfGateArtifact>(include_str!(
            "../../../.bench-history/QG-1.v7.unmeasured.latest.json"
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
            .expect("bootstrap sentinel fixture");
        }
        Fixture {
            directory,
            manifest,
        }
    }

    fn preflight_state(report: &Qg2PreflightReport, locator: &str) -> Qg2SelectorState {
        let missing = format!("preflight must carry a receipt for {locator}");
        report
            .selectors
            .iter()
            .find(|receipt| receipt.locator == locator)
            .expect(&missing)
            .state
    }

    fn has_preflight_divergence(
        report: &Qg2PreflightReport,
        code: &str,
        path_suffix: &str,
    ) -> bool {
        report
            .divergences
            .iter()
            .any(|divergence| divergence.code == code && divergence.path.ends_with(path_suffix))
    }

    #[test]
    fn protected_base_preflights_as_bootstrap_ready() {
        let fixture = bootstrap_fixture();
        let report = validate_qg2_preflight(fixture.root());
        assert!(report.is_bootstrap_ready(), "{:#?}", report.divergences);
        assert_eq!(report.selectors.len(), QG2_PHYSICAL_LOCATOR_COUNT);
        assert!(
            report
                .selectors
                .iter()
                .all(|receipt| receipt.state == Qg2SelectorState::Bootstrap)
        );
        assert_eq!(
            report
                .selectors
                .iter()
                .map(|receipt| receipt.locator.as_str())
                .collect::<Vec<_>>(),
            vec![
                "perf_gate_law_1",
                "comprehensive_plan_qg2_row",
                "comprehensive_plan_method_law_1",
                "perf_manifest_qg2_contract",
                "hyperopt_law_7",
                "hyperopt_w2_fsync_row",
                "hyperopt_epic_active_contract",
                "qg2_r1_active_contract",
                "gate_activation_active_contract",
            ]
        );
        assert_eq!(report.no_claim, QG2_NO_CLAIM);
    }

    #[test]
    fn expected_bootstrap_absence_is_not_a_divergence() {
        // The whole point of the preflight: on the very same tree the
        // applied-state validator must report divergence (the correction is
        // not applied yet) while the preflight reports a clean ready state.
        let fixture = bootstrap_fixture();
        let applied_view = validate_qg2_contract(fixture.root());
        assert!(!applied_view.is_pass());
        assert!(!applied_view.divergences.is_empty());

        let preflight = validate_qg2_preflight(fixture.root());
        assert_eq!(preflight.state, Qg2PreflightState::BootstrapReady);
        assert!(preflight.divergences.is_empty(), "{preflight:#?}");
    }

    #[test]
    fn applied_tree_preflights_as_already_applied_with_no_rebind() {
        let fixture = complete_fixture();
        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(
            report.state,
            Qg2PreflightState::AlreadyApplied,
            "{:#?}",
            report.divergences
        );
        assert!(
            report
                .selectors
                .iter()
                .all(|receipt| receipt.state == Qg2SelectorState::Applied)
        );
        assert_eq!(report.manifest_sha256_pre, report.manifest_sha256_post);
        assert_eq!(report.sentinel_rebinds.len(), QG2_SENTINEL_COUNT);
        assert!(
            report
                .sentinel_rebinds
                .iter()
                .all(|rebind| !rebind.rebind_required)
        );
    }

    #[test]
    fn bootstrap_preflight_binds_the_full_sentinel_rebind_cascade() {
        let fixture = bootstrap_fixture();
        let report = validate_qg2_preflight(fixture.root());
        let pre = report
            .manifest_sha256_pre
            .as_deref()
            .expect("bootstrap manifest digest");
        let post = report
            .manifest_sha256_post
            .as_deref()
            .expect("rendered manifest digest");
        assert_ne!(
            pre, post,
            "inserting the typed contract must move the normalized manifest digest"
        );
        assert_eq!(report.sentinel_rebinds.len(), QG2_SENTINEL_COUNT);
        assert!(report.sentinel_rebinds.iter().all(|rebind| {
            rebind.bound_manifest_sha256 == pre
                && rebind.rebound_manifest_sha256 == post
                && rebind.rebind_required
        }));
        assert_eq!(
            report
                .sentinel_rebinds
                .iter()
                .map(|rebind| rebind.gate)
                .collect::<Vec<_>>(),
            PerfGate::ALL.to_vec()
        );
    }

    #[test]
    fn the_applied_validator_binds_the_exact_projected_block() {
        // Before the block binding these three mutations all false-greened: the
        // typed contract was still canonical and the file still held exactly one
        // canonical clause, so a coordinated rewrite beside the contract passed.
        for (label, mutated) in [
            (
                "fixture rewritten beside the contract",
                applied_manifest().replacen(
                    "fixture = \"medium; positions ON; threads = 1; continuous",
                    "fixture = \"medium; positions ON; threads = 1; commit included; continuous",
                    1,
                ),
            ),
            (
                "activated flipped beside the contract",
                applied_manifest().replacen(
                    "target = \"docs_per_sec >= 1.5x oracle\"\nactivated = false",
                    "target = \"docs_per_sec >= 1.5x oracle\"\nactivated = true",
                    1,
                ),
            ),
            (
                "name rewritten beside the contract",
                applied_manifest().replacen(
                    "name = \"bulk indexing, single-thread\"",
                    "name = \"bulk indexing, relaxed\"",
                    1,
                ),
            ),
        ] {
            assert_ne!(mutated, applied_manifest(), "{label} mutation must apply");
            let fixture = complete_fixture();
            fs::write(fixture.path(PERF_MANIFEST_PATH), &mutated).expect("block mutation");
            let report = validate_qg2_contract(fixture.root());
            assert!(
                report
                    .divergences
                    .iter()
                    .any(|divergence| divergence.code == "qg2.manifest.projected_block"),
                "{label} unexpectedly passed: {:#?}",
                report.divergences
            );
        }
    }

    #[test]
    fn manifest_topology_and_field_placement_match_the_live_consumer() {
        // Byte-binding the QG-2 block says nothing about the other nine gates,
        // so each of these satisfied the Q2C selector while planning would have
        // refused the same file. Parity means both readers reject all of them.
        let applied = applied_manifest();
        for (label, mutated) in [
            (
                "a missing normative gate",
                applied.replacen("[gate.QG-10]", "[omitted.QG-10]", 1),
            ),
            (
                "an empty required scalar on a non-QG-2 gate",
                applied.replacen("target = \"QG-5 target\"", "target = \"   \"", 1),
            ),
            (
                "a missing required scalar on a non-QG-2 gate",
                applied.replacen("fixture = \"QG-4 fixture\"\n", "", 1),
            ),
            (
                "a missing activated flag",
                applied.replacen(
                    "target = \"QG-7 target\"\nactivated = false",
                    "target = \"QG-7 target\"",
                    1,
                ),
            ),
            (
                "a wrong frozen QG-6 group count",
                applied.replacen(
                    &format!("queries_per_class = {}", crate::QG6_QUERY_GROUPS),
                    "queries_per_class = 4",
                    1,
                ),
            ),
            (
                "a QG-1-only field placed on another gate",
                applied.replacen(
                    "target = \"QG-9 target\"\n",
                    "target = \"QG-9 target\"\nprimary_target_cell_width = 8\n",
                    1,
                ),
            ),
            (
                "a QG-6-only field placed on another gate",
                applied.replacen(
                    "target = \"QG-8 target\"\n",
                    "target = \"QG-8 target\"\nqueries_per_class = 16\n",
                    1,
                ),
            ),
        ] {
            assert_ne!(mutated, applied, "{label} mutation must apply");
            assert!(
                manifest_block_agreement(&mutated, true).is_err(),
                "{label} must fail the shared agreement"
            );
            let fixture = complete_fixture();
            fs::write(fixture.path(PERF_MANIFEST_PATH), &mutated).expect("parity mutation");
            assert!(
                validate_qg2_contract(fixture.root())
                    .divergences
                    .iter()
                    .any(|divergence| divergence.code == "qg2.manifest.projected_block"),
                "{label} unexpectedly passed the applied validator"
            );
            assert_eq!(
                validate_qg2_preflight(fixture.root()).state,
                Qg2PreflightState::Drift,
                "{label} unexpectedly passed the preflight"
            );
        }
    }

    #[test]
    fn a_wrong_gate_contract_table_or_extra_label_is_rejected_by_both_readers() {
        // QG-2 keeps its exact projected block; a *second* canonical table is
        // hung under QG-3, which a QG-2-only check would never look at.
        let table = QG2_MANIFEST_BLOCK_POST_REGION
            .split_once("[gate.QG-2.qg2_contract]\n")
            .map(|(_, table)| table)
            .expect("the projected block carries the contract table");
        let foreign_table = format!("{}[gate.QG-3.qg2_contract]\n{table}", applied_manifest());
        let extra_label = format!("{}\n[gate.QG-11]\nactivated = false\n", applied_manifest());

        for (label, mutated) in [
            ("a contract table under another gate", foreign_table),
            ("an undefined gate label", extra_label),
        ] {
            // The live consumer rejects both. Both readers must agree, or the
            // preflight blesses a manifest that planning will refuse.
            assert!(
                manifest_block_agreement(&mutated, true).is_err(),
                "{label} must fail the applied agreement"
            );
            let fixture = complete_fixture();
            fs::write(fixture.path(PERF_MANIFEST_PATH), &mutated).expect("agreement mutation");
            let applied_view = validate_qg2_contract(fixture.root());
            assert!(
                applied_view
                    .divergences
                    .iter()
                    .any(|divergence| divergence.code == "qg2.manifest.projected_block"),
                "{label} unexpectedly passed the applied validator"
            );
            let preflight = validate_qg2_preflight(fixture.root());
            assert_eq!(
                preflight.state,
                Qg2PreflightState::Drift,
                "{label} unexpectedly passed the preflight"
            );
        }
    }

    #[test]
    fn the_protected_projection_rewrites_the_fixture_and_not_only_the_table() {
        // The correction is not "insert a table". The protected projection also
        // retires the "commit included" fixture string, because a QG-2 fixture
        // that still advertises commit-inclusive durable framing contradicts the
        // contract sitting in the same block. A projection that only inserted
        // the table would leave that contradiction standing.
        assert!(QG2_MANIFEST_BLOCK_PRE_REGION.contains("threads = 1; commit included"));
        assert!(!QG2_MANIFEST_BLOCK_POST_REGION.contains("commit included"));
        assert!(QG2_MANIFEST_BLOCK_POST_REGION.contains(
            "continuous first-feed through terminal searchable visibility and complete \
             worker/merge/queue quiescence"
        ));
        assert!(QG2_MANIFEST_BLOCK_POST_REGION.contains("[gate.QG-2.qg2_contract]\n"));

        for block in [
            QG2_MANIFEST_BLOCK_PRE_REGION,
            QG2_MANIFEST_BLOCK_POST_REGION,
        ] {
            assert!(block.starts_with("[gate.QG-2]\n"));
            assert!(
                block.ends_with("\n\n"),
                "a block must not swallow its successor"
            );
            assert!(!block.contains("[gate.QG-3]"));
        }
    }

    #[test]
    fn the_projected_manifest_parses_back_to_the_canonical_typed_contract() {
        let source = bootstrap_manifest();
        assert_eq!(
            source.matches(QG2_MANIFEST_BLOCK_PRE_REGION).count(),
            1,
            "the bootstrap fixture must carry the exact protected block"
        );
        assert_eq!(qg2_block_topology(&source), Ok(Qg2BlockTopology::Bootstrap));

        let applied = source.replacen(
            QG2_MANIFEST_BLOCK_PRE_REGION,
            QG2_MANIFEST_BLOCK_POST_REGION,
            1,
        );
        let document =
            toml::from_str::<ManifestDocument>(&applied).expect("projected manifest must parse");
        assert_eq!(
            document
                .gate
                .get("QG-2")
                .and_then(|gate| gate.qg2_contract.as_ref()),
            Some(&Qg2ComparatorContract::canonical())
        );
        assert_eq!(qg2_block_topology(&applied), Ok(Qg2BlockTopology::Applied));
        assert_ne!(
            perf_manifest_contract_sha256(&source),
            perf_manifest_contract_sha256(&applied)
        );
    }

    #[test]
    fn a_tree_split_across_both_states_fails_closed() {
        let fixture = bootstrap_fixture();
        let plan = fs::read_to_string(fixture.path(COMPREHENSIVE_PLAN_PATH)).expect("plan fixture");
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            plan.replacen(
                "| **QG-2 Bulk indexing, single-thread** | >= 1.5x tantivy |",
                &format!("| **QG-2 Bulk indexing, single-thread** | {QG2_CANONICAL_CONTRACT} |"),
                1,
            ),
        )
        .expect("half-applied mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert_eq!(
            preflight_state(&report, "comprehensive_plan_qg2_row"),
            Qg2SelectorState::Applied
        );
        assert_eq!(
            preflight_state(&report, "comprehensive_plan_method_law_1"),
            Qg2SelectorState::Bootstrap
        );
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.preflight.split_state"),
            "{:#?}",
            report.divergences
        );
    }

    #[test]
    fn a_clause_outside_every_region_is_not_expected_bootstrap_absence() {
        let fixture = bootstrap_fixture();
        let plan = fs::read_to_string(fixture.path(COMPREHENSIVE_PLAN_PATH)).expect("plan fixture");
        fs::write(
            fixture.path(COMPREHENSIVE_PLAN_PATH),
            format!("{plan}\nAppendix. {QG2_CANONICAL_CONTRACT}\n"),
        )
        .expect("out-of-region clause mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert!(has_preflight_divergence(
            &report,
            "qg2.preflight.file_census",
            COMPREHENSIVE_PLAN_PATH
        ));
    }

    #[test]
    fn unexpected_tracker_notes_at_the_protected_base_are_drift() {
        let fixture = bootstrap_fixture();
        let mut tracker = String::new();
        for (index, (_, _, issue_id)) in TRACKER_SELECTORS.into_iter().enumerate() {
            let issue = if index == 2 {
                json!({ "id": issue_id, "notes": "a peer's unrelated active note" })
            } else {
                json!({ "id": issue_id })
            };
            tracker.push_str(&serde_json::to_string(&issue).expect("tracker issue"));
            tracker.push('\n');
        }
        fs::write(fixture.path(TRACKER_PATH), tracker).expect("unexpected notes mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert_eq!(
            preflight_state(&report, "gate_activation_active_contract"),
            Qg2SelectorState::Drift
        );
        assert_eq!(
            preflight_state(&report, "hyperopt_epic_active_contract"),
            Qg2SelectorState::Bootstrap
        );
        assert!(has_preflight_divergence(
            &report,
            "qg2.preflight.selector_drift",
            "#bd-h6eh.notes"
        ));
    }

    #[test]
    fn a_conflicting_nested_contract_table_is_drift() {
        let fixture = bootstrap_fixture();
        let conflicting = fixture.manifest.replace(
            "[gate.QG-3]",
            concat!(
                "[gate.QG-2.qg2_contract]\n",
                "contract = \"an earlier draft of the comparator contract\"\n",
                "storage_topology = \"symmetric_in_memory\"\n",
                "durability_scope = \"non_durable\"\n",
                "timing_start = \"first_document_feed\"\n",
                "timing_end = \"terminal_searchable_visibility_and_complete_worker_merge_queue_quiescence\"\n",
                "commit_boundary = \"searchable_visibility_not_durable_publication\"\n",
                "excluded_operations = [\"fsync\"]\n",
                "source_nonregression = \"durable_gates_and_production_source_durability_remain_mandatory\"\n",
                "\n",
                "[gate.QG-3]",
            ),
        );
        fs::write(fixture.path(PERF_MANIFEST_PATH), conflicting)
            .expect("conflicting table mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert_eq!(
            preflight_state(&report, "perf_manifest_qg2_contract"),
            Qg2SelectorState::Drift
        );
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.code == "qg2.preflight.manifest_conflict"),
            "{:#?}",
            report.divergences
        );
    }

    #[test]
    fn an_ambiguous_renamed_anchor_is_drift() {
        let fixture = bootstrap_fixture();
        let laws = fs::read_to_string(fixture.path(PERF_GATES_DOC_PATH)).expect("laws fixture");
        fs::write(
            fixture.path(PERF_GATES_DOC_PATH),
            laws.replace(
                "2. **Distributions, not averages.**",
                concat!(
                    "1. **No benchmark-only semantics; comparator scope is explicit.** draft\n",
                    "2. **Distributions, not averages.**",
                ),
            ),
        )
        .expect("ambiguous anchor mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert_eq!(
            preflight_state(&report, "perf_gate_law_1"),
            Qg2SelectorState::Drift
        );
        assert!(has_preflight_divergence(
            &report,
            "qg2.preflight.selector_drift",
            "#perf_gate_law_1"
        ));
    }

    #[test]
    fn a_sentinel_bound_to_a_foreign_digest_is_drift() {
        let fixture = bootstrap_fixture();
        let path = fixture.path(".bench-history/QG-7.unmeasured.latest.json");
        let mut artifact =
            serde_json::from_slice::<PerfGateArtifact>(&fs::read(&path).expect("QG-7 bytes"))
                .expect("QG-7 JSON");
        artifact.manifest_sha256 = "b".repeat(64);
        fs::write(
            path,
            serde_json::to_vec_pretty(&artifact).expect("mutated QG-7"),
        )
        .expect("foreign digest mutation");

        let report = validate_qg2_preflight(fixture.root());
        assert_eq!(report.state, Qg2PreflightState::Drift);
        assert!(has_preflight_divergence(
            &report,
            "qg2.preflight.sentinel_binding",
            "QG-7.unmeasured.latest.json"
        ));
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
