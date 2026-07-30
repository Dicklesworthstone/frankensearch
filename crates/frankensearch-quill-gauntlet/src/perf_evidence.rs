//! Versioned QG evidence artifacts: both-engine absolutes, paired effects,
//! validity diagnostics, executable provenance, and atomic persistence.
//!
//! This module is the decision-grade evidence layer above the repaired paired
//! estimator in [`crate::perf`]. Ratios alone hide which engine moved, so an
//! evidence cell always carries the absolute distributions for BOTH engines
//! from the same paired blocks, the paired relative effect with its seeded
//! bootstrap interval, the same-invocation A/A diagnostics, and the bounded
//! raw samples every summary recomputes from.
//!
//! Estimands are metric-specific by construction: flat paired log ratios for
//! throughput-style cells, two-stage hierarchical resampling for per-query
//! latency, and explicit process-level RSS / cold-open / dependency-facts
//! declarations that fail closed when their preconditions are unproven. One
//! generic bootstrap is never silently applied to the wrong metric.
//!
//! Invalid runs are durable: they persist as [`EvidenceDecisionStatus::InvalidNull`]
//! or [`EvidenceDecisionStatus::NoDecision`] artifacts with their raw samples,
//! and [`PerfEvidenceArtifact::ratchet_admissible`] refuses to let them
//! establish or move any baseline.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::perf::{
    DistributionSummary, LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3, PairedClaimState,
    PairedEstimatorConfig, PairedEstimatorError, PairedEvidenceStatus, PairedExperimentResult,
    PerfExecutionProvenance, PerfGate, PerfGateArtifact, PerfInputIdentity, PerfRawSample,
    PerfSampleArm, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, median_sorted, percentile, splitmix64,
    validate_paired_blocks,
};
use crate::qg6_prepared::{
    Qg6ArmRole, Qg6QueryIdentityReceipt, Qg6QuerySpec, Qg6SemanticContract,
    qg6_result_sequence_sha256,
};
use crate::{MachineClassEvidenceBinding, VerifiedRunnerIdentity};

/// Version of the evidence artifact emitted by this module.
///
/// Old artifacts never masquerade as current: loading any other version
/// through [`PerfEvidenceArtifact::load_verified`] is a typed
/// [`EvidenceArtifactError::SchemaMismatch`], and legacy v3 gate artifacts are
/// only readable through the explicit, read-only
/// [`load_legacy_gate_artifact_v3`].
pub const PERF_EVIDENCE_SCHEMA_VERSION: &str = "quill-perf-evidence-v3";
/// Version of the hierarchical latency estimate carried by latency cells.
pub const HIERARCHICAL_LATENCY_SCHEMA_VERSION: &str = "quill-hierarchical-latency-v1";
/// Upper bound on retained reasons per artifact or cell.
pub const EVIDENCE_MAX_REASONS: usize = 64;
/// Upper bound on one bounded reason message, in bytes.
pub const EVIDENCE_MAX_REASON_MESSAGE_BYTES: usize = 240;

/// Hash exact operating-system argv bytes with one NUL separator and a final
/// NUL terminator, as required by the runner receipt contract.
///
/// Operating-system argv entries cannot themselves contain NUL. Accepting
/// byte slices keeps this helper lossless for non-UTF-8 Unix arguments.
#[must_use]
pub fn command_sha256_from_argv<'a>(arguments: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    for argument in arguments {
        hasher.update(argument);
        hasher.update([0]);
    }
    lower_hex(&hasher.finalize())
}

/// Deterministic severity precedence: `Fatal > Block > Quarantine > NoClaim > Allow`.
///
/// Folding any reason set takes the maximum, so a single fatal reason always
/// dominates and an allow can never mask an invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSeverity {
    /// No objection; carries no decision weight on its own.
    Allow,
    /// Evidence exists but must not support any performance claim.
    NoClaim,
    /// Claim admitted previously but now requires quarantine review.
    Quarantine,
    /// Predeclared threshold violation that blocks promotion.
    Block,
    /// Structural corruption; the artifact must be rejected outright.
    Fatal,
}

impl fmt::Display for EvidenceSeverity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allow => "allow",
            Self::NoClaim => "no_claim",
            Self::Quarantine => "quarantine",
            Self::Block => "block",
            Self::Fatal => "fatal",
        })
    }
}

/// Whether a cell participates in the gate decision or is informational.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRole {
    /// The gate decision folds this cell in.
    Required,
    /// Persisted for operators; never gates.
    Diagnostic,
}

/// One bounded, structured reason. Messages never contain query or document
/// text; they are truncated to [`EVIDENCE_MAX_REASON_MESSAGE_BYTES`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReason {
    /// Stable machine-readable code, such as `evidence.undersampled`.
    pub code: String,
    /// Bounded operator-facing explanation.
    pub message: String,
    /// Severity folded by the deterministic precedence.
    pub severity: EvidenceSeverity,
}

impl EvidenceReason {
    /// Build a bounded reason, truncating the message on a char boundary.
    #[must_use]
    pub fn new(code: &str, message: impl Into<String>, severity: EvidenceSeverity) -> Self {
        let mut message = message.into();
        if message.len() > EVIDENCE_MAX_REASON_MESSAGE_BYTES {
            let mut cut = EVIDENCE_MAX_REASON_MESSAGE_BYTES;
            while !message.is_char_boundary(cut) {
                cut -= 1;
            }
            message.truncate(cut);
        }
        Self {
            code: code.to_owned(),
            message,
            severity,
        }
    }
}

/// Decision state persisted with every gate artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDecisionStatus {
    /// Predeclared threshold check passed and a claim is admitted.
    Allow,
    /// Valid measurement exists; no promotion decision has been applied yet.
    MeasuredProvisional,
    /// The same-invocation A/A null invalidated the run. Durable, no claim.
    InvalidNull,
    /// Measurement exists but cannot support any claim. Durable, no claim.
    NoDecision,
    /// Evidence requires human review before any further claim.
    Quarantine,
    /// Predeclared threshold violation.
    Block,
}

impl fmt::Display for EvidenceDecisionStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allow => "allow",
            Self::MeasuredProvisional => "measured_provisional",
            Self::InvalidNull => "invalid_null",
            Self::NoDecision => "no_decision",
            Self::Quarantine => "quarantine",
            Self::Block => "block",
        })
    }
}

/// Metric-specific estimand.
///
/// Each cell declares exactly how its raw records become a comparable
/// effect, so a generic bootstrap cannot be silently applied to a metric
/// whose sampling structure it does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEstimand {
    /// Flat paired log ratio over complete blocks (throughput, commit and
    /// update latency, merge and scaling cells).
    PairedLogRatio,
    /// Two-stage per-query resampling: queries first, then blocks within each
    /// query, so between-query variance is not laundered into precision.
    HierarchicalLatency,
    /// Process-level peak RSS measured in an isolated child process.
    ProcessRss,
    /// Cold index open; requires explicit cache-state proof.
    ColdOpen,
    /// Dependency or build-size facts outside noisy timing A/A.
    DependencyFacts,
}

impl fmt::Display for EvidenceEstimand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PairedLogRatio => "paired_log_ratio",
            Self::HierarchicalLatency => "hierarchical_latency",
            Self::ProcessRss => "process_rss",
            Self::ColdOpen => "cold_open",
            Self::DependencyFacts => "dependency_facts",
        })
    }
}

/// The estimand each gate's metrics must use.
#[must_use]
pub const fn required_estimand(gate: PerfGate) -> EvidenceEstimand {
    match gate {
        PerfGate::Qg6 => EvidenceEstimand::HierarchicalLatency,
        PerfGate::Qg7 => EvidenceEstimand::ProcessRss,
        PerfGate::Qg9 => EvidenceEstimand::ColdOpen,
        PerfGate::Qg10 => EvidenceEstimand::DependencyFacts,
        _ => EvidenceEstimand::PairedLogRatio,
    }
}

/// Predeclared evidence-layer thresholds, persisted with every artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencePolicy {
    /// Log-scale dead band below which reconciliation directions are noise.
    pub reconciliation_dead_band_log: f64,
    /// Maximum admitted |log paired-ratio − log marginal-ratio| divergence.
    pub reconciliation_tolerance_log: f64,
    /// Minimum first-stage query groups for a hierarchical latency claim.
    pub min_hierarchical_groups: usize,
    /// Minimum complete blocks inside each hierarchical group.
    pub min_group_pairs: usize,
    /// Warmup rounds executed and excluded from every decision set.
    pub warmup_rounds: usize,
    /// Upper bound on raw samples retained per cell stream.
    pub max_raw_samples: usize,
}

impl EvidencePolicy {
    /// Predeclared defaults for harness-emitted evidence.
    ///
    /// The reconciliation tolerance is deliberately loose (25% on the ratio
    /// scale) because the paired median-of-ratios and the marginal
    /// ratio-of-medians are different estimands that legitimately diverge on
    /// skewed distributions; the hard contradiction is opposite direction.
    #[must_use]
    pub fn predeclared() -> Self {
        Self {
            reconciliation_dead_band_log: 1.005_f64.ln(),
            reconciliation_tolerance_log: 1.25_f64.ln(),
            min_hierarchical_groups: 2,
            min_group_pairs: 2,
            warmup_rounds: 1,
            max_raw_samples: 20_000,
        }
    }

    /// Validate that every threshold is finite and usable.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::InvalidPolicy`] for non-finite or
    /// degenerate bounds.
    pub fn validate(&self) -> Result<(), EvidenceArtifactError> {
        let finite = self.reconciliation_dead_band_log.is_finite()
            && self.reconciliation_dead_band_log >= 0.0
            && self.reconciliation_tolerance_log.is_finite()
            && self.reconciliation_tolerance_log > 0.0;
        if !finite
            || self.min_hierarchical_groups < 2
            || self.min_group_pairs < 2
            || self.max_raw_samples == 0
        {
            return Err(EvidenceArtifactError::InvalidPolicy {
                reason: "evidence policy requires finite bounds, >=2 groups, >=2 pairs per \
                         group, and a positive raw-sample cap"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Exact build identity of the process that produced the evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildIdentity {
    /// SHA-256 the running benchmark binary computed over its own ELF.
    pub executable_sha256: String,
    /// Git revision of the measured tree.
    pub git_revision: String,
    /// Whether the working tree differed from the committed revision.
    pub git_dirty: bool,
    /// SHA-256 over the dirty diff; required whenever `git_dirty` is true.
    pub worktree_state_sha256: Option<String>,
    /// SHA-256 of the exact `Cargo.lock`, when resolvable.
    pub cargo_lock_sha256: Option<String>,
    /// SHA-256 of exact NUL-separated, NUL-terminated process argv bytes.
    pub command_sha256: String,
    /// SHA-256 of the typed producer's canonical build and workload environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_sha256: Option<String>,
    /// `rustc --version` of the toolchain that built the binary.
    pub rustc_version: String,
    /// Compilation target triple.
    pub target_triple: String,
    /// Cargo profile label.
    pub build_profile: String,
    /// Cargo features active in the measuring binary.
    pub cargo_features: Vec<String>,
}

impl BuildIdentity {
    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        let hex_ok = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if !hex_ok(&self.executable_sha256)
            || !hex_ok(&self.command_sha256)
            || self
                .environment_sha256
                .as_deref()
                .is_some_and(|value| !hex_ok(value))
            || self.git_revision.trim().is_empty()
            || self.rustc_version.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.build_profile.trim().is_empty()
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "build identity requires executable and command SHA-256 values, a git \
                         revision, rustc version, target triple, and profile"
                    .to_owned(),
            });
        }
        if self.git_dirty && !self.worktree_state_sha256.as_deref().is_some_and(hex_ok) {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "a dirty tree requires a worktree-state SHA-256".to_owned(),
            });
        }
        if self
            .cargo_lock_sha256
            .as_deref()
            .is_some_and(|v| !hex_ok(v))
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "cargo lock hash must be lowercase SHA-256 hex".to_owned(),
            });
        }
        Ok(())
    }
}

/// Machine identity captured at run start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MachineIdentity {
    /// Deterministic machine label from [`crate::perf::machine_fingerprint`].
    pub fingerprint: String,
    /// Operating system constant.
    pub os: String,
    /// Architecture constant.
    pub arch: String,
    /// Logical CPU count observed by the process.
    pub logical_cpus: usize,
    /// Host topology, ISA, affinity, and exact configured engine widths for
    /// the invocation.
    pub execution: PerfExecutionProvenance,
    /// CPU frequency governor, when the platform exposes one.
    pub cpu_governor: Option<String>,
    /// One-minute load average at run start, when readable.
    pub load_average_start: Option<f64>,
    /// One-minute load average at run end, when readable.
    pub load_average_end: Option<f64>,
}

impl MachineIdentity {
    /// Capture the current machine identity. Unavailable probes report
    /// `None` rather than fabricating zeros.
    #[must_use]
    pub fn capture(configured_engine_thread_widths: impl IntoIterator<Item = usize>) -> Self {
        Self {
            fingerprint: crate::perf::machine_fingerprint(),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
            execution: PerfExecutionProvenance::capture(configured_engine_thread_widths),
            cpu_governor: fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
            )
            .ok()
            .map(|value| value.trim().to_owned()),
            load_average_start: read_load_average(),
            load_average_end: None,
        }
    }

    /// Record the end-of-run load average.
    pub fn finish(&mut self) {
        self.load_average_end = read_load_average();
    }

    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        if self.fingerprint.trim().is_empty()
            || self.os.trim().is_empty()
            || self.arch.trim().is_empty()
            || self.logical_cpus == 0
            || !self.execution.is_complete()
            || self.execution.producer_os.as_str() != self.os
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "machine identity requires fingerprint, os, arch, CPUs, and matching \
                         serialized producer OS"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

fn read_load_average() -> Option<f64> {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|contents| {
            contents
                .split_ascii_whitespace()
                .next()
                .and_then(|field| field.parse::<f64>().ok())
        })
        .filter(|value| value.is_finite())
}

/// Peak RSS evidence with its measurement method. Unsupported platforms
/// report `method = "unsupported"` with no bytes, never a fabricated zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakRssEvidence {
    /// One of `linux_vmhwm`, `macos_time_l`, or `unsupported`.
    pub method: String,
    /// Peak resident set size in bytes when the method supports it.
    pub bytes: Option<u64>,
}

impl PeakRssEvidence {
    /// Capture peak RSS for the current process.
    #[must_use]
    pub fn capture() -> Self {
        crate::perf::peak_rss_bytes().map_or_else(
            || Self {
                method: "unsupported".to_owned(),
                bytes: None,
            },
            |bytes| Self {
                method: "linux_vmhwm".to_owned(),
                bytes: Some(bytes),
            },
        )
    }

    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        let valid = match self.method.as_str() {
            "unsupported" => self.bytes.is_none(),
            "linux_vmhwm" | "macos_time_l" => self.bytes.is_some_and(|bytes| bytes > 0),
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "peak RSS method {:?} is inconsistent with its byte value; unsupported \
                     probes must report no bytes rather than zero",
                    self.method
                ),
            })
        }
    }
}

/// Explicit cache-state proof for cold-open cells.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColdCacheEvidence {
    /// Bounded description of the cache-state procedure that was executed.
    pub procedure: String,
    /// Whether the procedure actually proves a cold cache.
    pub verified: bool,
}

/// Identity of the measured corpus and query set. Content never rides along;
/// only hashes, sizes, and generator coordinates do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusIdentity {
    /// SHA-256 over the corpus manifest.
    pub corpus_sha256: String,
    /// SHA-256 over the invocation-wide query universe, when the gate uses
    /// one. A prepared cell's exact ordered manifest is separately sealed in
    /// [`PerfInputIdentity::query_manifest_sha256`].
    pub query_set_sha256: Option<String>,
    /// SHA-256 over relevance judgments, when the gate uses them.
    pub qrels_sha256: Option<String>,
    /// Number of documents measured.
    pub document_count: u64,
    /// Total corpus content bytes, when computed.
    pub content_bytes: Option<u64>,
    /// Deterministic generator seed.
    pub generator_seed: u64,
    /// Generator recipe revision label.
    pub generator_revision: String,
}

impl CorpusIdentity {
    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        let hex_ok = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if !hex_ok(&self.corpus_sha256)
            || self.generator_revision.trim().is_empty()
            || self.query_set_sha256.as_deref().is_some_and(|v| !hex_ok(v))
            || self.qrels_sha256.as_deref().is_some_and(|v| !hex_ok(v))
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "corpus identity requires SHA-256 hashes and a generator revision"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Complete run-level provenance persisted with every evidence artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceProvenance {
    /// Unique identifier for one pass inside the measurement window.
    pub run_id: String,
    /// Shared identifier for the bounded measurement window.
    pub run_window: String,
    /// SHA-256 of the committed gate manifest.
    pub manifest_sha256: String,
    /// Build identity of the measuring binary.
    pub build: BuildIdentity,
    /// Machine identity at run start/end.
    pub machine: MachineIdentity,
    /// Peak RSS of the harness process.
    pub peak_rss: PeakRssEvidence,
    /// Measured corpus identity.
    pub corpus: CorpusIdentity,
}

impl EvidenceProvenance {
    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        if self.run_id.trim().is_empty() || self.run_window.trim().is_empty() {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "evidence requires a run ID and run window".to_owned(),
            });
        }
        if self.manifest_sha256.len() != 64 {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "manifest hash must be SHA-256 hex".to_owned(),
            });
        }
        self.build.validate()?;
        self.machine.validate()?;
        self.peak_rss.validate()?;
        self.corpus.validate()
    }
}

/// Same-scope reconciliation between the paired effect and the marginal
/// absolute summaries computed from the identical raw arm values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbsoluteRelativeReconciliation {
    /// Exponentiated paired median log ratio.
    pub paired_median_ratio: f64,
    /// Ratio of the marginal arm medians from the same blocks.
    pub marginal_median_ratio: f64,
    /// Absolute divergence between the two ratios on the log scale.
    pub abs_log_delta: f64,
    /// Whether both ratios agree in direction outside the dead band.
    pub direction_agrees: bool,
    /// Whether the divergence sits inside the predeclared tolerance.
    pub within_tolerance: bool,
}

impl AbsoluteRelativeReconciliation {
    /// Reconcile a paired effect against its own marginal summaries.
    #[must_use]
    pub fn from_effect(
        effect: &crate::perf::PairedEffectEstimate,
        policy: &EvidencePolicy,
    ) -> Self {
        let paired_log = effect.median_log_ratio;
        let marginal_log = effect.ratio_of_arm_medians.ln();
        let abs_log_delta = (paired_log - marginal_log).abs();
        let direction_agrees = !(paired_log.abs() > policy.reconciliation_dead_band_log
            && marginal_log.abs() > policy.reconciliation_dead_band_log
            && paired_log.is_sign_positive() != marginal_log.is_sign_positive());
        Self {
            paired_median_ratio: effect.treatment_over_control,
            marginal_median_ratio: effect.ratio_of_arm_medians,
            abs_log_delta,
            direction_agrees,
            within_tolerance: abs_log_delta <= policy.reconciliation_tolerance_log,
        }
    }
}

/// Per-group summary inside a hierarchical latency estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalGroupSummary {
    /// First-stage unit identity (one query).
    pub group_id: u64,
    /// Complete paired blocks measured for this group.
    pub pair_count: usize,
    /// Median paired log ratio inside this group.
    pub median_log_ratio: f64,
}

/// Two-stage hierarchical latency estimate.
///
/// Queries are resampled first, then blocks within each selected query, so
/// the interval reflects between-query variance instead of pretending all
/// blocks are exchangeable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalLatencyEstimate {
    /// Version of this estimate's contract.
    pub schema_version: String,
    /// Number of first-stage groups.
    pub group_count: usize,
    /// Total complete pairs across all groups.
    pub total_pairs: usize,
    /// Median of per-group median log ratios: the primary estimand.
    pub median_of_group_medians_log: f64,
    /// Exponentiated primary estimand.
    pub treatment_over_control: f64,
    /// Lower 95% two-stage bootstrap bound on the log scale.
    pub ci95_low_log: f64,
    /// Upper 95% two-stage bootstrap bound on the log scale.
    pub ci95_high_log: f64,
    /// Exponentiated lower bound.
    pub ci95_low_ratio: f64,
    /// Exponentiated upper bound.
    pub ci95_high_ratio: f64,
    /// Deterministic per-group summaries ordered by group ID.
    pub groups: Vec<HierarchicalGroupSummary>,
}

/// Estimate a per-query latency effect with two-stage resampling.
///
/// Every sample must carry a `group_id`; blocks never mix groups. Each group
/// needs at least `policy.min_group_pairs` complete blocks and the stream
/// needs at least `policy.min_hierarchical_groups` groups.
///
/// # Errors
///
/// Returns typed fail-closed errors for missing or mixed group IDs,
/// undersampled groups, and every structural defect the flat estimator
/// rejects.
pub fn estimate_hierarchical_latency(
    samples: &[PerfRawSample],
    config: &PairedEstimatorConfig,
    policy: &EvidencePolicy,
) -> Result<HierarchicalLatencyEstimate, PairedEstimatorError> {
    for sample in samples {
        if sample.group_id.is_none() {
            return Err(PairedEstimatorError::MissingGroupId {
                sample_id: sample.sample_id,
            });
        }
    }
    let (_, _, pairs, _) = validate_paired_blocks(samples, config)?;
    let mut groups: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for pair in &pairs {
        let group_id = pair.group_id.ok_or(PairedEstimatorError::MissingGroupId {
            sample_id: pair.block_id,
        })?;
        groups.entry(group_id).or_default().push(pair.log_ratio);
    }
    if groups.len() < policy.min_hierarchical_groups {
        return Err(PairedEstimatorError::InsufficientGroups {
            actual: groups.len(),
            required: policy.min_hierarchical_groups,
        });
    }
    for (group_id, ratios) in &groups {
        if ratios.len() < policy.min_group_pairs {
            return Err(PairedEstimatorError::InsufficientGroupPairs {
                group_id: *group_id,
                actual: ratios.len(),
                required: policy.min_group_pairs,
            });
        }
    }

    let group_ids = groups.keys().copied().collect::<Vec<_>>();
    let group_values = groups.values().cloned().collect::<Vec<_>>();
    let mut group_medians = Vec::with_capacity(group_values.len());
    let mut summaries = Vec::with_capacity(group_values.len());
    for (group_id, values) in group_ids.iter().zip(&group_values) {
        let mut sorted = values.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let median = median_sorted(&sorted);
        group_medians.push(median);
        summaries.push(HierarchicalGroupSummary {
            group_id: *group_id,
            pair_count: values.len(),
            median_log_ratio: median,
        });
    }
    let mut sorted_medians = group_medians.clone();
    sorted_medians.sort_unstable_by(f64::total_cmp);
    let point = median_sorted(&sorted_medians);

    let mut seed = config.bootstrap_seed ^ 0x4849_4552_4152_4348;
    for median in &group_medians {
        seed = splitmix64(seed ^ median.to_bits());
    }
    let group_count = group_values.len();
    let group_count_u64 =
        u64::try_from(group_count).map_err(|_| PairedEstimatorError::InvalidConfig {
            reason: "hierarchical group count does not fit u64".to_owned(),
        })?;
    let mut statistics = Vec::with_capacity(config.bootstrap_resamples);
    let mut chosen_medians = Vec::with_capacity(group_count);
    let mut scratch = Vec::new();
    for _ in 0..config.bootstrap_resamples {
        chosen_medians.clear();
        for _ in 0..group_count {
            seed = splitmix64(seed);
            let group_index = usize::try_from(seed % group_count_u64).map_err(|_| {
                PairedEstimatorError::InvalidConfig {
                    reason: "hierarchical group index does not fit usize".to_owned(),
                }
            })?;
            let values = &group_values[group_index];
            let value_count =
                u64::try_from(values.len()).map_err(|_| PairedEstimatorError::InvalidConfig {
                    reason: "hierarchical group size does not fit u64".to_owned(),
                })?;
            scratch.clear();
            for _ in 0..values.len() {
                seed = splitmix64(seed);
                let index = usize::try_from(seed % value_count).map_err(|_| {
                    PairedEstimatorError::InvalidConfig {
                        reason: "hierarchical sample index does not fit usize".to_owned(),
                    }
                })?;
                scratch.push(values[index]);
            }
            scratch.sort_unstable_by(f64::total_cmp);
            chosen_medians.push(median_sorted(&scratch));
        }
        chosen_medians.sort_unstable_by(f64::total_cmp);
        statistics.push(median_sorted(&chosen_medians));
    }
    statistics.sort_unstable_by(f64::total_cmp);
    let ci95_low_log = percentile(&statistics, 0.025);
    let ci95_high_log = percentile(&statistics, 0.975);

    Ok(HierarchicalLatencyEstimate {
        schema_version: HIERARCHICAL_LATENCY_SCHEMA_VERSION.to_owned(),
        group_count,
        total_pairs: pairs.len(),
        median_of_group_medians_log: point,
        treatment_over_control: point.exp(),
        ci95_low_log,
        ci95_high_log,
        ci95_low_ratio: ci95_low_log.exp(),
        ci95_high_ratio: ci95_high_log.exp(),
        groups: summaries,
    })
}

/// Engine arm covered by a materialized-pool observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfConcurrencyEngine {
    /// Quill candidate arm.
    Quill,
    /// Pinned Tantivy incumbent arm.
    Tantivy,
}

/// Independent mechanism that witnessed a materialized engine pool width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfConcurrencyObserver {
    /// `rayon::current_num_threads()` executed inside the exact Quill pool.
    RayonCurrentPoolWidth,
    /// Tantivy successfully constructed the exact configured writer pool.
    TantivyWriterConstruction,
}

/// Repeated observations for one engine in one scaling cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConcurrencyObservation {
    /// Engine whose materialized pool was observed.
    pub engine: PerfConcurrencyEngine,
    /// Independent observation mechanism.
    pub observer: PerfConcurrencyObserver,
    /// Number of measured invocations covered by this observation.
    pub observation_count: usize,
    /// Minimum materialized pool width over all observations.
    pub min_observed_worker_pool_threads: usize,
    /// Maximum materialized pool width over all observations.
    pub max_observed_worker_pool_threads: usize,
}

/// Exact per-cell witness that configured scaling knobs materialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfConcurrencyWitness {
    /// Thread-width knob declared by the normative matrix cell.
    pub configured_threads: usize,
    /// Exactly one Quill and one Tantivy observation, in engine order.
    pub observations: Vec<EngineConcurrencyObservation>,
}

impl PerfConcurrencyWitness {
    fn validate(&self) -> Result<(), EvidenceArtifactError> {
        let expected = [
            (
                PerfConcurrencyEngine::Quill,
                PerfConcurrencyObserver::RayonCurrentPoolWidth,
            ),
            (
                PerfConcurrencyEngine::Tantivy,
                PerfConcurrencyObserver::TantivyWriterConstruction,
            ),
        ];
        if self.configured_threads == 0 || self.observations.len() != expected.len() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason:
                    "scaling concurrency witness requires one positive-width observation per engine"
                        .to_owned(),
            });
        }
        for (observation, (engine, observer)) in self.observations.iter().zip(expected) {
            if observation.engine != engine
                || observation.observer != observer
                || observation.observation_count == 0
                || observation.min_observed_worker_pool_threads != self.configured_threads
                || observation.max_observed_worker_pool_threads != self.configured_threads
            {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason:
                        "scaling concurrency witness is missing, duplicated, or disagrees with \
                         the configured pool width"
                            .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Identity of one evidence cell before measurement is attached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCellSpec {
    /// Gate this cell belongs to.
    pub gate: PerfGate,
    /// Fixture label matching the QG matrix cell.
    pub fixture: String,
    /// Metric label matching the QG matrix cell.
    pub metric: String,
    /// Human-readable unit of the absolute summaries.
    pub unit: String,
    /// Whether the gate decision folds this cell in.
    pub role: EvidenceRole,
    /// Separate exact prepared-corpus, ordered-query, and semantic-configuration
    /// hashes. QG-6 requires this identity; flat gates leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_identity: Option<PerfInputIdentity>,
    /// Full redacted semantic receipt table, required only for QG-6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qg6_semantic_contract: Option<Qg6SemanticContract>,
    /// Cache-state proof, required for cold-open cells.
    pub cold_cache: Option<ColdCacheEvidence>,
    /// Per-engine materialized-pool witness for QG-1/QG-8 scaling cells.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency_witness: Option<PerfConcurrencyWitness>,
}

/// Measured body of one evidence cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvidenceCellBody {
    /// Paired candidate/control measurement with its A/A null.
    Paired {
        /// Complete replayable paired result, including both-engine absolute
        /// distributions and the bounded raw samples they recompute from.
        paired: Box<PairedExperimentResult>,
        /// The identical A/B stream adjudicated against a same-invocation
        /// treatment/treatment null. QG-1 uses Tantivy as control and Quill as
        /// treatment, so this carries the replayable Quill/Quill null while
        /// `paired.null` carries Tantivy/Tantivy.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        treatment_arm_null: Option<Box<PairedExperimentResult>>,
        /// Two-stage A/B effect estimate for hierarchical latency cells.
        hierarchical: Option<HierarchicalLatencyEstimate>,
        /// Two-stage same-invocation A/A null estimate for hierarchical latency
        /// cells. A hierarchical effect can never borrow a flat null inference.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hierarchical_null: Option<HierarchicalLatencyEstimate>,
        /// Same-scope absolute-versus-paired reconciliation. This is a
        /// diagnostic projection for QG-6, whose inference is hierarchical.
        reconciliation: AbsoluteRelativeReconciliation,
    },
    /// Direct facts outside noisy timing A/A, such as dependency counts.
    Facts {
        /// Raw observed values.
        raw_values: Vec<f64>,
        /// Summary recomputable from `raw_values`.
        summary: DistributionSummary,
    },
}

/// One decision-grade evidence cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCell {
    /// Stable cell identity: `gate/fixture/metric`.
    pub cell_id: String,
    /// Cell identity and role.
    pub spec: EvidenceCellSpec,
    /// Declared metric-specific estimand.
    pub estimand: EvidenceEstimand,
    /// Measured body.
    pub body: EvidenceCellBody,
    /// Deterministic decision status for this cell.
    pub status: EvidenceDecisionStatus,
    /// Bounded structured reasons in fixed check order.
    pub reasons: Vec<EvidenceReason>,
}

/// Flat-estimator findings that are diagnostics only for QG-6.
///
/// QG-6's inferential unit is the query, so its effect and null center,
/// interval, and dispersion come from the two-stage hierarchy. Everything
/// outside this explicit allowlist remains a fail-closed paired-stream
/// structural/design finding.
fn qg6_flat_inference_only(code: &str) -> bool {
    matches!(
        code,
        "paired.null_center_invalid"
            | "paired.null_too_wide"
            | "paired.null_dispersion"
            | "paired.absolute_direction_conflict"
    )
}

fn hierarchical_groups_match_input(
    estimate: &HierarchicalLatencyEstimate,
    identity: &PerfInputIdentity,
) -> bool {
    estimate.group_count == identity.query_group_count
        && estimate.groups.len() == identity.query_group_count
        && estimate
            .groups
            .iter()
            .map(|group| group.group_id)
            .eq(identity.query_group_ids.iter().copied())
}

fn qg6_role(effect_stream: bool, arm: PerfSampleArm) -> Qg6ArmRole {
    match (effect_stream, arm) {
        (false, PerfSampleArm::Control) => Qg6ArmRole::NullLeft,
        (false, PerfSampleArm::Treatment) => Qg6ArmRole::NullRight,
        (true, PerfSampleArm::Control) => Qg6ArmRole::EffectControl,
        (true, PerfSampleArm::Treatment) => Qg6ArmRole::EffectTreatment,
    }
}

fn validate_qg6_sample_stream(
    samples: &[PerfRawSample],
    effect_stream: bool,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
    represented: &mut BTreeMap<(u64, Qg6ArmRole), usize>,
    row_keys: &mut BTreeSet<(bool, u64, Qg6ArmRole)>,
) -> Result<(), EvidenceArtifactError> {
    for sample in samples {
        if sample.provenance.input_identity.as_ref() != Some(identity) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row does not carry the exact cell input identity".to_owned(),
            });
        }
        let group_id =
            sample
                .group_id
                .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 raw row is missing its canonical query group".to_owned(),
                })?;
        let group_index =
            usize::try_from(group_id).map_err(|_| EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row query group does not fit the platform".to_owned(),
            })?;
        let group = contract.groups.get(group_index).ok_or_else(|| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row query group is outside the semantic contract".to_owned(),
            }
        })?;
        if group.group_id != group_id {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row query group does not resolve canonically".to_owned(),
            });
        }
        let binding = sample.qg6_sample_binding.as_ref().ok_or_else(|| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row is missing its compact semantic binding".to_owned(),
            }
        })?;
        if binding.query_id != group.query.query_id {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row query ID does not match its semantic-contract group"
                    .to_owned(),
            });
        }
        let work_units = sample
            .work_units
            .filter(|count| *count > 0)
            .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row requires a positive explicit result-sequence length"
                    .to_owned(),
            })?;
        let role = qg6_role(effect_stream, sample.arm);
        if !row_keys.insert((effect_stream, sample.block_id, role)) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 semantic stream repeats one logical role in a paired block"
                    .to_owned(),
            });
        }
        let expected_sequence = qg6_result_sequence_sha256(group.roles.get(role), work_units)
            .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
                reason: format!("QG-6 raw row result sequence cannot recompute: {error}"),
            })?;
        if binding.result_sequence_sha256 != expected_sequence {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 raw row result sequence diverges from its full role receipt"
                    .to_owned(),
            });
        }
        let count = represented.entry((group_id, role)).or_default();
        *count =
            count
                .checked_add(1)
                .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 semantic role multiplicity overflowed".to_owned(),
                })?;
    }
    Ok(())
}

fn evaluate_qg6(
    spec: &EvidenceCellSpec,
    paired: &PairedExperimentResult,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
) -> Result<Vec<EvidenceReason>, EvidenceArtifactError> {
    identity.validate()?;
    contract
        .verify()
        .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
            reason: format!("QG-6 semantic contract failed verification: {error}"),
        })?;
    if contract.groups.len() != QG6_QUERY_GROUPS
        || contract
            .groups
            .iter()
            .map(|group| group.group_id)
            .ne(QG6_QUERY_GROUP_IDS)
        || identity.prepared_corpus_sha256 != contract.prepared_corpus_sha256
        || identity.query_manifest_sha256 != contract.query_manifest_sha256
        || identity.config_contract_sha256 != contract.config_contract_sha256
        || identity.semantic_contract_sha256.as_deref() != Some(contract.contract_sha256.as_str())
        || identity.query_group_count != contract.groups.len()
        || identity.query_group_ids.as_slice() != QG6_QUERY_GROUP_IDS.as_slice()
        || paired.provenance.input_identity.as_ref() != Some(identity)
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 input identity, query universe, and semantic contract do not cross-bind"
                .to_owned(),
        });
    }
    let Some(query_class) = contract.groups.first().map(|group| group.query.class) else {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 semantic contract has no query groups".to_owned(),
        });
    };
    let corpus_label = match contract.document_count {
        100_000 => "100k",
        1_000_000 => "1m",
        _ => {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 semantic contract has a non-normative corpus cardinality".to_owned(),
            });
        }
    };
    let expected_fixture = format!(
        "query/{}/k{}/{corpus_label}",
        query_class.label(),
        contract.k
    );
    if contract
        .groups
        .iter()
        .any(|group| group.query.class != query_class)
        || spec.fixture != expected_fixture
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 semantic contract class, cutoff, or corpus does not match the cell \
                     fixture"
                .to_owned(),
        });
    }
    let mut frozen_queries = Qg6QuerySpec::normative_for_class(query_class)
        .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
            reason: format!("QG-6 frozen query slice is unavailable: {error}"),
        })?
        .iter()
        .map(Qg6QueryIdentityReceipt::from_query)
        .collect::<Vec<_>>();
    frozen_queries.sort_unstable_by(|left, right| left.query_id.cmp(&right.query_id));
    if contract
        .groups
        .iter()
        .map(|group| &group.query)
        .ne(frozen_queries.iter())
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 semantic contract does not match the frozen normative query slice"
                .to_owned(),
        });
    }

    let mut represented = BTreeMap::new();
    let mut row_keys = BTreeSet::new();
    validate_qg6_sample_stream(
        &paired.effect_samples,
        true,
        identity,
        contract,
        &mut represented,
        &mut row_keys,
    )?;
    validate_qg6_sample_stream(
        &paired.null_samples,
        false,
        identity,
        contract,
        &mut represented,
        &mut row_keys,
    )?;
    let expected = QG6_QUERY_GROUP_IDS
        .into_iter()
        .flat_map(|group_id| {
            Qg6ArmRole::ALL
                .into_iter()
                .map(move |role| (group_id, role))
        })
        .collect::<BTreeSet<_>>();
    if represented.keys().copied().collect::<BTreeSet<_>>() != expected
        || represented.values().copied().collect::<BTreeSet<_>>().len() != 1
        || represented.values().any(|count| *count == 0)
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 raw rows require one common positive multiplicity across every query \
                     and logical role"
                .to_owned(),
        });
    }

    Ok(vec![EvidenceReason::new(
        "qg6.tail_protocol_not_implemented",
        "QG-6 semantic receipts are verified, but true-leaf p50/p99 resampling and tail-validity \
         evidence are not implemented",
        EvidenceSeverity::NoClaim,
    )])
}

impl EvidenceCell {
    /// Evaluate one paired measurement into a decision-grade cell.
    ///
    /// The status derivation is deterministic and ordered. Flat gates map an
    /// invalid A/A null to [`EvidenceDecisionStatus::InvalidNull`] and retain
    /// their paired-estimator inference. QG-6 instead derives inference from
    /// its hierarchical A/B and A/A estimates while preserving paired-stream
    /// structural/design failures. Any remaining invalidity, estimand
    /// precondition failure, reconciliation conflict, or undersampling yields
    /// [`EvidenceDecisionStatus::NoDecision`]; otherwise the cell is
    /// [`EvidenceDecisionStatus::MeasuredProvisional`].
    ///
    /// # Errors
    ///
    /// Returns typed errors for unbounded raw sample sets and invalid policy.
    pub fn evaluate(
        spec: EvidenceCellSpec,
        paired: PairedExperimentResult,
        policy: &EvidencePolicy,
    ) -> Result<Self, EvidenceArtifactError> {
        policy.validate()?;
        let requires_concurrency_witness = spec.gate == PerfGate::Qg8
            || (spec.gate == PerfGate::Qg1 && spec.role == EvidenceRole::Required);
        match (
            requires_concurrency_witness,
            spec.concurrency_witness.as_ref(),
        ) {
            (true, Some(witness)) => witness.validate()?,
            (true, None) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-1/QG-8 scaling evidence requires a per-engine concurrency witness"
                        .to_owned(),
                });
            }
            (false, Some(_)) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "only required QG-1/QG-8 scaling cells can carry concurrency witnesses"
                        .to_owned(),
                });
            }
            (false, None) => {}
        }
        let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
        let retained = paired.effect_samples.len() + paired.null_samples.len();
        if retained > policy.max_raw_samples {
            return Err(EvidenceArtifactError::UnboundedRawSamples {
                cell_id,
                count: retained,
                max: policy.max_raw_samples,
            });
        }
        let estimand = required_estimand(spec.gate);
        let mut reasons = Vec::new();

        match (
            spec.gate,
            spec.input_identity.as_ref(),
            spec.qg6_semantic_contract.as_ref(),
        ) {
            (PerfGate::Qg6, Some(identity), Some(contract)) => {
                reasons.extend(evaluate_qg6(&spec, &paired, identity, contract)?);
            }
            (PerfGate::Qg6, _, _) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 evidence requires both a compact input identity and its full \
                             semantic contract"
                        .to_owned(),
                });
            }
            (_, None, None)
                if paired.provenance.input_identity.is_none()
                    && paired
                        .effect_samples
                        .iter()
                        .chain(&paired.null_samples)
                        .all(|sample| sample.qg6_sample_binding.is_none()) => {}
            (_, _, _) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "only QG-6 evidence can carry prepared-input identity, semantic \
                             contracts, or compact result bindings"
                        .to_owned(),
                });
            }
        }

        let hierarchical_gate = estimand == EvidenceEstimand::HierarchicalLatency;
        if !hierarchical_gate && paired.status != PairedEvidenceStatus::Valid {
            reasons.push(EvidenceReason::new(
                "evidence.paired_invalid",
                format!(
                    "paired estimator status {:?} with {} reason(s)",
                    paired.status,
                    paired.reasons.len()
                ),
                EvidenceSeverity::NoClaim,
            ));
        }
        if hierarchical_gate {
            for diagnostic in paired
                .reasons
                .iter()
                .filter(|reason| !qg6_flat_inference_only(reason.code.as_str()))
            {
                reasons.push(EvidenceReason::new(
                    "evidence.qg6_paired_design_invalid",
                    format!(
                        "QG-6 paired stream failed structural/design check {}: {}",
                        diagnostic.code, diagnostic.message
                    ),
                    EvidenceSeverity::NoClaim,
                ));
            }
        }

        let (hierarchical, hierarchical_null) = if hierarchical_gate {
            let effect =
                match estimate_hierarchical_latency(&paired.effect_samples, &paired.config, policy)
                {
                    Ok(estimate) => Some(estimate),
                    Err(error) => {
                        reasons.push(EvidenceReason::new(
                            "evidence.hierarchical_effect_unavailable",
                            format!("hierarchical A/B latency estimand failed: {error}"),
                            EvidenceSeverity::NoClaim,
                        ));
                        None
                    }
                };
            let null =
                match estimate_hierarchical_latency(&paired.null_samples, &paired.config, policy) {
                    Ok(estimate) => Some(estimate),
                    Err(error) => {
                        reasons.push(EvidenceReason::new(
                            "evidence.hierarchical_null_unavailable",
                            format!("hierarchical A/A latency estimand failed: {error}"),
                            EvidenceSeverity::NoClaim,
                        ));
                        None
                    }
                };
            if let Some(null) = null.as_ref() {
                let max_ci_distance = null.ci95_low_log.abs().max(null.ci95_high_log.abs());
                if !(null.ci95_low_log <= 0.0 && 0.0 <= null.ci95_high_log)
                    || null.median_of_group_medians_log.abs() > paired.config.max_null_center_log
                    || max_ci_distance > paired.config.max_null_ci_half_width_log
                {
                    reasons.push(EvidenceReason::new(
                        "evidence.hierarchical_null_invalid",
                        format!(
                            "hierarchical A/A center {:.6} with log-CI [{:.6}, {:.6}] exceeds \
                                 the predeclared null bounds",
                            null.median_of_group_medians_log, null.ci95_low_log, null.ci95_high_log
                        ),
                        EvidenceSeverity::NoClaim,
                    ));
                }
            }
            if let Some(identity) = spec.input_identity.as_ref() {
                for (label, estimate) in
                    [("A/B effect", effect.as_ref()), ("A/A null", null.as_ref())]
                {
                    if estimate.is_some_and(|estimate| {
                        !hierarchical_groups_match_input(estimate, identity)
                    }) {
                        reasons.push(EvidenceReason::new(
                            "evidence.qg6_query_groups_incomplete",
                            format!(
                                "hierarchical {label} groups do not exactly match the prepared \
                                 ordered query-group identity"
                            ),
                            EvidenceSeverity::NoClaim,
                        ));
                    }
                }
            }
            (effect, null)
        } else {
            (None, None)
        };

        if estimand == EvidenceEstimand::ColdOpen
            && !spec
                .cold_cache
                .as_ref()
                .is_some_and(|evidence| evidence.verified)
        {
            reasons.push(EvidenceReason::new(
                "evidence.cold_cache_unproven",
                "cold-open estimand requires verified cache-state evidence",
                EvidenceSeverity::NoClaim,
            ));
        }

        let reconciliation = AbsoluteRelativeReconciliation::from_effect(&paired.effect, policy);
        if !hierarchical_gate {
            if !reconciliation.direction_agrees {
                reasons.push(EvidenceReason::new(
                    "evidence.absolute_relative_direction_conflict",
                    format!(
                        "paired ratio {:.6} and marginal ratio {:.6} disagree in direction",
                        reconciliation.paired_median_ratio, reconciliation.marginal_median_ratio
                    ),
                    EvidenceSeverity::NoClaim,
                ));
            } else if !reconciliation.within_tolerance {
                reasons.push(EvidenceReason::new(
                    "evidence.absolute_relative_magnitude_divergence",
                    format!(
                        "paired/marginal log divergence {:.6} exceeds tolerance {:.6}",
                        reconciliation.abs_log_delta, policy.reconciliation_tolerance_log
                    ),
                    EvidenceSeverity::NoClaim,
                ));
            }
        }

        if !(paired.effect.control.sampled_for_activation()
            && paired.effect.treatment.sampled_for_activation())
        {
            reasons.push(EvidenceReason::new(
                "evidence.undersampled",
                format!(
                    "arms report {}/{} runs; activation requires the standing minimum",
                    paired.effect.control.runs, paired.effect.treatment.runs
                ),
                EvidenceSeverity::NoClaim,
            ));
        }

        reasons.truncate(EVIDENCE_MAX_REASONS);
        let status = if !hierarchical_gate && paired.status == PairedEvidenceStatus::InvalidNull {
            EvidenceDecisionStatus::InvalidNull
        } else if (!hierarchical_gate && paired.status != PairedEvidenceStatus::Valid)
            || reasons
                .iter()
                .any(|reason| reason.severity >= EvidenceSeverity::NoClaim)
        {
            EvidenceDecisionStatus::NoDecision
        } else {
            EvidenceDecisionStatus::MeasuredProvisional
        };

        Ok(Self {
            cell_id,
            spec,
            estimand,
            body: EvidenceCellBody::Paired {
                paired: Box::new(paired),
                treatment_arm_null: None,
                hierarchical,
                hierarchical_null,
                reconciliation,
            },
            status,
            reasons,
        })
    }

    /// Attach the treatment arm's independently measured same-invocation A/A
    /// null to an already evaluated A/B cell.
    ///
    /// # Errors
    ///
    /// Rejects a mismatched A/B stream, malformed replay, unbounded retained
    /// samples, or use on a non-paired cell.
    pub fn attach_treatment_arm_null(
        &mut self,
        treatment_arm_null: PairedExperimentResult,
        policy: &EvidencePolicy,
    ) -> Result<(), EvidenceArtifactError> {
        policy.validate()?;
        treatment_arm_null.verify_recomputed()?;
        let EvidenceCellBody::Paired {
            paired,
            treatment_arm_null: slot,
            ..
        } = &mut self.body
        else {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "treatment-arm A/A null requires a paired evidence cell".to_owned(),
            });
        };
        if treatment_arm_null.scope != paired.scope
            || treatment_arm_null.provenance != paired.provenance
            || treatment_arm_null.config != paired.config
            || treatment_arm_null.effect != paired.effect
            || treatment_arm_null.effect_samples != paired.effect_samples
        {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "treatment-arm A/A null does not share the cell's exact A/B stream"
                    .to_owned(),
            });
        }
        let retained = paired.effect_samples.len()
            + paired.null_samples.len()
            + treatment_arm_null.effect_samples.len()
            + treatment_arm_null.null_samples.len();
        if retained > policy.max_raw_samples {
            return Err(EvidenceArtifactError::UnboundedRawSamples {
                cell_id: self.cell_id.clone(),
                count: retained,
                max: policy.max_raw_samples,
            });
        }
        if treatment_arm_null.status != PairedEvidenceStatus::Valid {
            self.reasons.push(EvidenceReason::new(
                "evidence.treatment_arm_null_invalid",
                format!(
                    "treatment/treatment estimator status {:?} with {} reason(s)",
                    treatment_arm_null.status,
                    treatment_arm_null.reasons.len()
                ),
                EvidenceSeverity::NoClaim,
            ));
            self.status = EvidenceDecisionStatus::InvalidNull;
        }
        *slot = Some(Box::new(treatment_arm_null));
        self.reasons.truncate(EVIDENCE_MAX_REASONS);
        Ok(())
    }

    /// Build a facts cell for measurements outside noisy timing A/A.
    ///
    /// # Errors
    ///
    /// Rejects non-diagnostic roles, gates whose estimand is not
    /// [`EvidenceEstimand::DependencyFacts`], and invalid raw values.
    pub fn facts(
        spec: EvidenceCellSpec,
        raw_values: Vec<f64>,
        policy: &EvidencePolicy,
    ) -> Result<Self, EvidenceArtifactError> {
        policy.validate()?;
        let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
        let estimand = required_estimand(spec.gate);
        if estimand != EvidenceEstimand::DependencyFacts {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!("gate {} does not admit a facts cell", spec.gate),
            });
        }
        if spec.input_identity.is_some() || spec.qg6_semantic_contract.is_some() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "dependency facts cannot carry QG-6 identity or semantic contracts"
                    .to_owned(),
            });
        }
        if spec.role != EvidenceRole::Diagnostic && raw_values.len() < 2 {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "required facts cells need at least two observations".to_owned(),
            });
        }
        if raw_values.len() > policy.max_raw_samples {
            return Err(EvidenceArtifactError::UnboundedRawSamples {
                cell_id,
                count: raw_values.len(),
                max: policy.max_raw_samples,
            });
        }
        let summary = DistributionSummary::from_samples(&raw_values).map_err(|error| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: format!("facts summary rejected raw values: {error}"),
            }
        })?;
        Ok(Self {
            cell_id,
            spec,
            estimand,
            body: EvidenceCellBody::Facts {
                raw_values,
                summary,
            },
            status: EvidenceDecisionStatus::MeasuredProvisional,
            reasons: Vec::new(),
        })
    }

    /// Whether this cell can feed a downstream promotion decision.
    #[must_use]
    pub fn claim_eligible(&self) -> bool {
        match &self.body {
            EvidenceCellBody::Paired {
                paired: _,
                hierarchical,
                hierarchical_null,
                ..
            } if self.spec.gate == PerfGate::Qg6 => {
                self.status == EvidenceDecisionStatus::MeasuredProvisional
                    && hierarchical.is_some()
                    && hierarchical_null.is_some()
            }
            EvidenceCellBody::Paired {
                paired,
                treatment_arm_null,
                ..
            } => {
                self.status == EvidenceDecisionStatus::MeasuredProvisional
                    && paired.claim_state == PairedClaimState::EligibleForDecision
                    && (self.spec.gate != PerfGate::Qg1
                        || treatment_arm_null.as_ref().is_some_and(|null| {
                            null.status == PairedEvidenceStatus::Valid
                                && null.claim_state == PairedClaimState::EligibleForDecision
                        }))
            }
            EvidenceCellBody::Facts { .. } => false,
        }
    }

    /// Recompute this cell from its own raw contents and compare.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::InconsistentArtifact`] on any
    /// mismatch between stored summaries and their raw sources.
    pub fn verify_recomputed(&self, policy: &EvidencePolicy) -> Result<(), EvidenceArtifactError> {
        match &self.body {
            EvidenceCellBody::Paired {
                paired,
                treatment_arm_null,
                ..
            } => {
                let mut rebuilt =
                    Self::evaluate(self.spec.clone(), paired.as_ref().clone(), policy)?;
                // QG-6 semantic bindings are cell-level evidence. Validate
                // them before the generic paired estimator so hostile row
                // mutations receive the semantic fail-closed classification
                // instead of being obscured by a lower-level pair mismatch.
                paired.verify_recomputed()?;
                if let Some(treatment_arm_null) = treatment_arm_null {
                    rebuilt
                        .attach_treatment_arm_null(treatment_arm_null.as_ref().clone(), policy)?;
                }
                if rebuilt == *self {
                    Ok(())
                } else {
                    Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: format!("cell {} does not recompute", self.cell_id),
                    })
                }
            }
            EvidenceCellBody::Facts {
                raw_values,
                summary,
            } => {
                let recomputed =
                    DistributionSummary::from_samples(raw_values).map_err(|error| {
                        EvidenceArtifactError::InconsistentArtifact {
                            reason: format!("facts raw values no longer summarize: {error}"),
                        }
                    })?;
                if recomputed == *summary
                    && self.status == EvidenceDecisionStatus::MeasuredProvisional
                {
                    Ok(())
                } else {
                    Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: format!("facts cell {} does not recompute", self.cell_id),
                    })
                }
            }
        }
    }
}

/// Paths written by one atomic artifact persistence pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceArtifactPaths {
    /// Authoritative JSON artifact.
    pub json: PathBuf,
    /// Human table derived from the JSON.
    pub table: PathBuf,
}

/// Versioned, hash-sealed, decision-grade gate evidence artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceArtifact {
    /// Always [`PERF_EVIDENCE_SCHEMA_VERSION`] for current artifacts.
    pub schema_version: String,
    /// Gate this artifact certifies.
    pub gate: PerfGate,
    /// Predeclared evidence-layer thresholds used for every cell.
    pub policy: EvidencePolicy,
    /// Complete run provenance.
    pub provenance: EvidenceProvenance,
    /// Strict runner-receipt machine identity. An explicit unverified binding
    /// is durable for diagnosis but can never promote.
    pub machine_class: MachineClassEvidenceBinding,
    /// Decision-grade cells.
    pub cells: Vec<EvidenceCell>,
    /// Deterministic fold of required-cell statuses.
    pub gate_status: EvidenceDecisionStatus,
    /// Promotion decision recorded by a downstream validator, if any.
    pub gate_decision: Option<EvidenceDecisionStatus>,
    /// Invocation-level reason this otherwise valid evidence must not support
    /// a claim, such as selecting only part of a normative gate.
    ///
    /// This is stored separately from [`Self::reasons`] because `reasons` is a
    /// derived fold. Keeping the input explicit makes a partial-run artifact
    /// recomputable by [`Self::load_verified`] instead of manufacturing a gate
    /// status that cannot be derived from its persisted sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_no_claim: Option<EvidenceReason>,
    /// Gate-level reasons in fixed fold order.
    pub reasons: Vec<EvidenceReason>,
    /// SHA-256 over the canonical JSON with this field empty.
    pub artifact_sha256: String,
}

impl PerfEvidenceArtifact {
    fn validate_cell_set(
        gate: PerfGate,
        cells: &[EvidenceCell],
    ) -> Result<(), EvidenceArtifactError> {
        if cells.is_empty() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "an evidence artifact requires at least one cell".to_owned(),
            });
        }
        let mut cell_ids = BTreeSet::new();
        for cell in cells {
            if cell.spec.gate != gate {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "cell {} belongs to gate {}, not {gate}",
                        cell.cell_id, cell.spec.gate
                    ),
                });
            }
            let expected_id = format!("{gate}/{}/{}", cell.spec.fixture, cell.spec.metric);
            if cell.cell_id != expected_id {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "cell ID {:?} does not match its canonical identity {:?}",
                        cell.cell_id, expected_id
                    ),
                });
            }
            if !cell_ids.insert(cell.cell_id.as_str()) {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!("evidence artifact repeats cell {}", cell.cell_id),
                });
            }
        }
        Ok(())
    }

    /// Assemble and fold one gate's evidence.
    ///
    /// # Errors
    ///
    /// Returns typed errors for invalid policy or provenance, an empty cell
    /// set, or a cell belonging to a different gate.
    pub fn assemble(
        gate: PerfGate,
        policy: EvidencePolicy,
        provenance: EvidenceProvenance,
        cells: Vec<EvidenceCell>,
    ) -> Result<Self, EvidenceArtifactError> {
        policy.validate()?;
        provenance.validate()?;
        Self::validate_cell_set(gate, &cells)?;
        let admission_no_claim = None;
        let (gate_status, reasons) = Self::fold(&cells, admission_no_claim.as_ref());
        Ok(Self {
            schema_version: PERF_EVIDENCE_SCHEMA_VERSION.to_owned(),
            gate,
            policy,
            provenance,
            machine_class: MachineClassEvidenceBinding::unverified(
                "sealed runner receipt has not been bound",
            ),
            cells,
            gate_status,
            gate_decision: None,
            admission_no_claim,
            reasons,
            artifact_sha256: String::new(),
        })
    }

    /// Deterministic severity-precedence fold of required cells.
    fn fold(
        cells: &[EvidenceCell],
        admission_no_claim: Option<&EvidenceReason>,
    ) -> (EvidenceDecisionStatus, Vec<EvidenceReason>) {
        let mut reasons = Vec::new();
        let mut any_invalid_null = false;
        let mut any_no_decision = false;
        let mut any_required = false;
        for cell in cells {
            if cell.spec.role != EvidenceRole::Required {
                continue;
            }
            any_required = true;
            match cell.status {
                EvidenceDecisionStatus::InvalidNull => {
                    any_invalid_null = true;
                    reasons.push(EvidenceReason::new(
                        "evidence.gate_cell_invalid_null",
                        format!("required cell {} has an invalid A/A null", cell.cell_id),
                        EvidenceSeverity::NoClaim,
                    ));
                }
                EvidenceDecisionStatus::NoDecision => {
                    any_no_decision = true;
                    reasons.push(EvidenceReason::new(
                        "evidence.gate_cell_no_decision",
                        format!("required cell {} supports no claim", cell.cell_id),
                        EvidenceSeverity::NoClaim,
                    ));
                }
                _ => {}
            }
        }
        if let Some(reason) = admission_no_claim {
            any_no_decision = true;
            reasons.push(reason.clone());
        }
        if !any_required {
            reasons.push(EvidenceReason::new(
                "evidence.gate_without_required_cells",
                "every cell is diagnostic; the gate cannot claim anything",
                EvidenceSeverity::NoClaim,
            ));
        }
        reasons.truncate(EVIDENCE_MAX_REASONS);
        let status = if any_invalid_null {
            EvidenceDecisionStatus::InvalidNull
        } else if any_no_decision || !any_required {
            EvidenceDecisionStatus::NoDecision
        } else {
            EvidenceDecisionStatus::MeasuredProvisional
        };
        (status, reasons)
    }

    /// Whether this artifact may establish or move a ratchet baseline.
    ///
    /// Invalid runs remain durable but can never ratchet.
    #[must_use]
    pub fn ratchet_admissible(&self) -> bool {
        self.gate_status == EvidenceDecisionStatus::MeasuredProvisional
            && self
                .machine_class
                .identity()
                .is_some_and(|identity| identity.artifact_manifest().is_some())
            && self
                .cells
                .iter()
                .filter(|cell| cell.spec.role == EvidenceRole::Required)
                .all(EvidenceCell::claim_eligible)
    }

    /// Bind the exact registry-admitted runner identity before sealing.
    ///
    /// Binding invalidates any prior artifact seal and downstream gate
    /// decision. The stored receipt remains independently re-admitted by
    /// [`Self::verify_integrity`] on every load.
    ///
    /// # Errors
    ///
    /// Returns a provenance error if the supplied identity no longer
    /// re-admits against the frozen registry.
    pub fn bind_machine_class_identity(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
    ) -> Result<(), EvidenceArtifactError> {
        let source = Self::from_verified_slice(prebinding_evidence_bytes)?;
        if source != *self {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "in-memory evidence differs from the exact pre-binding source bytes"
                    .to_owned(),
            });
        }
        identity
            .verify()
            .map_err(|error| EvidenceArtifactError::InvalidProvenance {
                reason: format!("machine-class binding rejected: {error}"),
            })?;
        identity
            .verify_threshold_artifact(threshold_artifact_bytes)
            .and_then(|()| identity.verify_evidence_artifact(prebinding_evidence_bytes))
            .map_err(|error| EvidenceArtifactError::InvalidProvenance {
                reason: format!("runner artifact-manifest binding rejected: {error}"),
            })?;
        let artifact_manifest = identity
            .artifact_manifest()
            .expect("exact artifact inputs require a manifest")
            .manifest();
        if artifact_manifest.gate() != self.gate.label()
            || artifact_manifest.run_id() != self.provenance.run_id
            || artifact_manifest.run_window() != self.provenance.run_window
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "runner artifact manifest names a different gate, run ID, or run window"
                    .to_owned(),
            });
        }
        if let Some(existing) = self.machine_class.identity()
            && existing != &identity
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "evidence already carries a different verified runner receipt; verified \
                         bindings are immutable"
                    .to_owned(),
            });
        }
        let context = identity.admission_context();
        let gate_label = self.gate.label();
        let expected_destination = format!("{gate_label}.{}.latest.json", identity.class_id());
        if context.gate != gate_label || context.destination_basename != expected_destination {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "machine-class receipt was admitted for gate/destination {}/{} instead of \
                     {gate_label}/{expected_destination}",
                    context.gate, context.destination_basename
                ),
            });
        }
        let build = identity.build().as_object().ok_or_else(|| {
            EvidenceArtifactError::InvalidProvenance {
                reason: "verified runner build facts are not an object".to_owned(),
            }
        })?;
        let runner_string = |field: &str| {
            build
                .get(field)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| EvidenceArtifactError::InvalidProvenance {
                    reason: format!("verified runner build field {field:?} is not a string"),
                })
        };
        let runner_git_dirty = build
            .get("git_dirty")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| EvidenceArtifactError::InvalidProvenance {
                reason: "verified runner build field \"git_dirty\" is not a boolean".to_owned(),
            })?;
        let runner_worktree_state = match build.get("worktree_state_sha256") {
            Some(serde_json::Value::String(value)) => Some(value.as_str()),
            Some(serde_json::Value::Null) => None,
            _ => {
                return Err(EvidenceArtifactError::InvalidProvenance {
                    reason: "verified runner worktree-state identity is malformed".to_owned(),
                });
            }
        };
        let runner_cargo_lock = runner_string("cargo_lock_sha256")?;
        let evidence_build = &self.provenance.build;
        let build_matches = evidence_build.git_revision == runner_string("git_revision")?
            && evidence_build.git_dirty == runner_git_dirty
            && evidence_build.worktree_state_sha256.as_deref() == runner_worktree_state
            && evidence_build.cargo_lock_sha256.as_deref() == Some(runner_cargo_lock)
            && evidence_build.executable_sha256 == runner_string("executable_sha256")?
            && evidence_build.command_sha256 == runner_string("command_sha256")?
            && evidence_build.environment_sha256.as_deref()
                == Some(runner_string("environment_sha256")?);
        if !build_matches {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "evidence build identity differs from the verified runner receipt"
                    .to_owned(),
            });
        }
        let hardware = identity.hardware().as_object().ok_or_else(|| {
            EvidenceArtifactError::InvalidProvenance {
                reason: "verified runner hardware facts are not an object".to_owned(),
            }
        })?;
        let hardware_string = |field: &str| {
            hardware
                .get(field)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| EvidenceArtifactError::InvalidProvenance {
                    reason: format!("verified runner hardware field {field:?} is not a string"),
                })
        };
        let hardware_os = hardware_string("os")?;
        let hardware_arch = hardware_string("arch")?;
        let target_matches = match (hardware_os, hardware_arch) {
            ("linux", "x86_64") => {
                evidence_build.target_triple.starts_with("x86_64-")
                    && evidence_build.target_triple.contains("linux")
            }
            ("macos", "aarch64") => {
                evidence_build.target_triple.starts_with("aarch64-")
                    && evidence_build.target_triple.contains("apple-darwin")
            }
            _ => false,
        };
        if self.provenance.machine.os != hardware_os
            || self.provenance.machine.arch != hardware_arch
            || self.provenance.machine.execution.producer_os.as_str() != hardware_os
            || !target_matches
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "evidence OS/architecture/target differs from verified runner hardware"
                    .to_owned(),
            });
        }
        self.machine_class = MachineClassEvidenceBinding::verified(identity);
        self.gate_decision = None;
        self.artifact_sha256.clear();
        Ok(())
    }

    /// Bind one post-exit runner receipt and return the newly sealed artifact.
    ///
    /// The benchmark process can persist an explicit unverified diagnostic
    /// artifact while it is still running, but a completion receipt cannot be
    /// sealed until that process exits. The ratchet finalization lane uses this
    /// method to join those two exact objects in memory before evaluating or
    /// opening any promotion-history destination. The returned bytes are the
    /// only evidence bytes a successful promotion may persist.
    ///
    /// # Errors
    ///
    /// Returns the same provenance errors as
    /// [`Self::bind_machine_class_identity`], plus serialization or invariant
    /// errors if the newly bound artifact cannot be verified exactly.
    pub fn bind_machine_class_identity_and_seal(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
    ) -> Result<Vec<u8>, EvidenceArtifactError> {
        let mut bound = self.clone();
        bound.bind_machine_class_identity(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
        )?;
        bound.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&bound)?;
        bound.artifact_sha256 = lower_hex(&Sha256::digest(unsealed.as_bytes()));
        let sealed = serde_json::to_vec_pretty(&bound)?;
        bound.verify_integrity()?;
        *self = bound;
        Ok(sealed)
    }

    /// Fail closed when an invocation selected only part of a normative gate.
    ///
    /// The measured cells and raw samples remain durable, but the artifact
    /// cannot establish a ratchet or accept a downstream gate decision. If a
    /// runner receipt was already bound, this pre-binding-content mutation
    /// explicitly discards it: a fresh receipt must bind the changed bytes.
    pub fn force_no_claim(&mut self, code: &str, message: impl Into<String>) {
        if self.machine_class.identity().is_some() {
            self.machine_class = MachineClassEvidenceBinding::unverified(
                "evidence changed after runner binding; a fresh receipt is required",
            );
        }
        self.admission_no_claim = Some(EvidenceReason::new(
            code,
            message,
            EvidenceSeverity::NoClaim,
        ));
        self.gate_decision = None;
        (self.gate_status, self.reasons) =
            Self::fold(&self.cells, self.admission_no_claim.as_ref());
        self.artifact_sha256.clear();
    }

    /// Record a downstream promotion decision.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::NotClaimEligible`] unless the folded
    /// evidence is ratchet-admissible, and rejects non-decision statuses.
    pub fn apply_gate_decision(
        &mut self,
        decision: EvidenceDecisionStatus,
    ) -> Result<(), EvidenceArtifactError> {
        if !matches!(
            decision,
            EvidenceDecisionStatus::Allow
                | EvidenceDecisionStatus::Quarantine
                | EvidenceDecisionStatus::Block
        ) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!("{decision} is not a promotion decision"),
            });
        }
        if !self.ratchet_admissible() {
            return Err(EvidenceArtifactError::NotClaimEligible);
        }
        self.gate_decision = Some(decision);
        Ok(())
    }

    /// Canonical pretty JSON with the currently stored seal.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if a non-finite value slipped through.
    pub fn canonical_json(&self) -> Result<String, EvidenceArtifactError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    fn sealed_json(&self) -> Result<String, EvidenceArtifactError> {
        let mut unsealed = self.clone();
        unsealed.artifact_sha256 = String::new();
        let bytes = serde_json::to_string_pretty(&unsealed)?;
        let digest = Sha256::digest(bytes.as_bytes());
        let mut sealed = unsealed;
        sealed.artifact_sha256 = lower_hex(&digest);
        Ok(serde_json::to_string_pretty(&sealed)?)
    }

    fn reconstructed_prebinding_bytes(&self) -> Result<Vec<u8>, EvidenceArtifactError> {
        let mut source = self.clone();
        source.machine_class =
            MachineClassEvidenceBinding::unverified("sealed runner receipt has not been bound");
        source.gate_decision = None;
        source.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&source)?;
        source.artifact_sha256 = lower_hex(&Sha256::digest(unsealed.as_bytes()));
        Ok(serde_json::to_vec_pretty(&source)?)
    }

    /// Verify this in-memory artifact's seal and every derived invariant.
    ///
    /// This is the in-memory counterpart to [`Self::load_verified`]. Public
    /// consumers that retain a loaded artifact and later pass it to another
    /// decision API must be able to detect post-load mutation instead of
    /// trusting the syntax of a stale `artifact_sha256` field.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for a stale schema,
    /// broken content seal, invalid policy or provenance, malformed cell set,
    /// non-recomputable cell or gate fold, or inadmissible recorded decision.
    pub fn verify_integrity(&self) -> Result<(), EvidenceArtifactError> {
        if self.schema_version != PERF_EVIDENCE_SCHEMA_VERSION {
            return Err(EvidenceArtifactError::SchemaMismatch {
                found: self.schema_version.clone(),
            });
        }
        let mut unsealed = self.clone();
        unsealed.artifact_sha256 = String::new();
        let recomputed = lower_hex(&Sha256::digest(
            serde_json::to_string_pretty(&unsealed)?.as_bytes(),
        ));
        if recomputed != self.artifact_sha256 {
            return Err(EvidenceArtifactError::HashMismatch);
        }
        self.policy.validate()?;
        self.provenance.validate()?;
        self.machine_class.validate().map_err(|error| {
            EvidenceArtifactError::InvalidProvenance {
                reason: format!("machine-class binding rejected: {error}"),
            }
        })?;
        if let Some(identity) = self.machine_class.identity() {
            let prebinding_bytes = self.reconstructed_prebinding_bytes()?;
            identity
                .verify_evidence_artifact(&prebinding_bytes)
                .map_err(|error| EvidenceArtifactError::InvalidProvenance {
                    reason: format!("runner evidence-artifact binding rejected: {error}"),
                })?;
            let manifest = identity
                .artifact_manifest()
                .expect("verified evidence binding requires an artifact manifest")
                .manifest();
            if manifest.gate() != self.gate.label()
                || manifest.run_id() != self.provenance.run_id
                || manifest.run_window() != self.provenance.run_window
            {
                return Err(EvidenceArtifactError::InvalidProvenance {
                    reason: "bound artifact manifest names a different gate, run ID, or run window"
                        .to_owned(),
                });
            }
        }
        Self::validate_cell_set(self.gate, &self.cells)?;
        if let Some(reason) = self.admission_no_claim.as_ref()
            && (reason.severity != EvidenceSeverity::NoClaim
                || reason.code.trim().is_empty()
                || reason.code.len() > EVIDENCE_MAX_REASON_MESSAGE_BYTES
                || reason.message.len() > EVIDENCE_MAX_REASON_MESSAGE_BYTES)
        {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "admission no-claim input must be bounded and have no-claim severity"
                    .to_owned(),
            });
        }
        for cell in &self.cells {
            cell.verify_recomputed(&self.policy)?;
        }
        let (expected_status, expected_reasons) =
            Self::fold(&self.cells, self.admission_no_claim.as_ref());
        if expected_status != self.gate_status || expected_reasons != self.reasons {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "gate fold does not recompute from the stored cells".to_owned(),
            });
        }
        if let Some(decision) = self.gate_decision {
            if !matches!(
                decision,
                EvidenceDecisionStatus::Allow
                    | EvidenceDecisionStatus::Quarantine
                    | EvidenceDecisionStatus::Block
            ) {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!("{decision} is not a promotion decision"),
                });
            }
            if !self.ratchet_admissible() {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "a promotion decision is recorded on non-eligible evidence".to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Render the operator table for this artifact.
    #[must_use]
    pub fn human_table(&self) -> String {
        let mut table = String::new();
        let _ = writeln!(
            table,
            "gate {} | status {} | decision {} | run {} | window {}",
            self.gate,
            self.gate_status,
            self.gate_decision
                .map_or_else(|| "none".to_owned(), |decision| decision.to_string()),
            self.provenance.run_id,
            self.provenance.run_window,
        );
        table.push_str(
            "cell | role | estimand | status | control_p50 | treatment_p50 | ratio | \
             ci95_ratio | pairs | reasons\n",
        );
        table.push_str("--- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---:\n");
        for cell in &self.cells {
            let role = match cell.spec.role {
                EvidenceRole::Required => "required",
                EvidenceRole::Diagnostic => "diagnostic",
            };
            match &cell.body {
                EvidenceCellBody::Paired {
                    paired,
                    hierarchical,
                    ..
                } => {
                    let (ratio, ci_low, ci_high) = hierarchical.as_ref().map_or(
                        (
                            paired.effect.treatment_over_control,
                            paired.effect.ci95_low_ratio,
                            paired.effect.ci95_high_ratio,
                        ),
                        |estimate| {
                            (
                                estimate.treatment_over_control,
                                estimate.ci95_low_ratio,
                                estimate.ci95_high_ratio,
                            )
                        },
                    );
                    let _ = writeln!(
                        table,
                        "{} | {} | {} | {} | {:.6} | {:.6} | {:.6} | [{:.6}, {:.6}] | {} | {}",
                        cell.cell_id,
                        role,
                        cell.estimand,
                        cell.status,
                        paired.effect.control.p50,
                        paired.effect.treatment.p50,
                        ratio,
                        ci_low,
                        ci_high,
                        paired.effect.pair_count,
                        cell.reasons.len(),
                    );
                }
                EvidenceCellBody::Facts { summary, .. } => {
                    let _ = writeln!(
                        table,
                        "{} | {} | {} | {} | {:.6} | - | - | - | {} | {}",
                        cell.cell_id,
                        role,
                        cell.estimand,
                        cell.status,
                        summary.p50,
                        summary.runs,
                        cell.reasons.len(),
                    );
                }
            }
        }
        table
    }

    /// Atomically persist the sealed JSON artifact plus its derived table.
    ///
    /// Each file is written to a temporary sibling, flushed with `fsync`,
    /// renamed into place, and the directory itself is synced. The Markdown
    /// table is derived from the exact JSON string that was persisted, which
    /// proves the human view is a pure function of the authoritative JSON.
    ///
    /// # Errors
    ///
    /// Returns typed serialization and I/O errors.
    pub fn write_atomic(
        &self,
        output_dir: &Path,
    ) -> Result<EvidenceArtifactPaths, EvidenceArtifactError> {
        fs::create_dir_all(output_dir)?;
        let stem = format!("{}.evidence", self.gate.label());
        let json = self.sealed_json()?;
        let json_path = output_dir.join(format!("{stem}.json"));
        write_file_atomic(output_dir, &json_path, json.as_bytes())?;
        let table = human_table_from_json(&json)?;
        let table_path = output_dir.join(format!("{stem}.md"));
        write_file_atomic(output_dir, &table_path, table.as_bytes())?;
        Ok(EvidenceArtifactPaths {
            json: json_path,
            table: table_path,
        })
    }

    /// Parse exact artifact bytes and verify seal, schema, and recomputability.
    ///
    /// This byte-oriented entry point lets callers hash and retain the same
    /// object they verified, without a second filesystem read that could race
    /// a replacement.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each defect class.
    pub fn from_verified_slice(contents: &[u8]) -> Result<Self, EvidenceArtifactError> {
        let probe =
            crate::machine_class_registry::parse_strict_json(contents).map_err(|error| {
                EvidenceArtifactError::Malformed {
                    reason: format!("artifact is not strict JSON: {error}"),
                }
            })?;
        let found = probe
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");
        if found != PERF_EVIDENCE_SCHEMA_VERSION {
            return Err(EvidenceArtifactError::SchemaMismatch {
                found: found.to_owned(),
            });
        }
        let artifact: Self = serde_json::from_value(probe.clone()).map_err(|error| {
            EvidenceArtifactError::Malformed {
                reason: format!("artifact does not decode as the current schema: {error}"),
            }
        })?;
        if probe != serde_json::to_value(&artifact)? {
            return Err(EvidenceArtifactError::Malformed {
                reason: "artifact contains unknown fields or a noncanonical persisted shape"
                    .to_owned(),
            });
        }
        artifact.verify_integrity()?;
        Ok(artifact)
    }

    /// Load one artifact and verify seal, schema, and recomputability.
    ///
    /// A truncated or otherwise malformed file, a stale schema version, a
    /// broken hash seal, or any summary that no longer recomputes from its
    /// raw samples is a typed error, never a silent acceptance.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each defect class.
    pub fn load_verified(path: &Path) -> Result<Self, EvidenceArtifactError> {
        let contents = fs::read(path)?;
        Self::from_verified_slice(&contents)
    }
}

/// Derive the operator table from an authoritative JSON string.
///
/// # Errors
///
/// Returns [`EvidenceArtifactError::Malformed`] when the JSON does not parse
/// into a current-schema artifact.
pub fn human_table_from_json(json: &str) -> Result<String, EvidenceArtifactError> {
    let artifact: PerfEvidenceArtifact =
        serde_json::from_str(json).map_err(|error| EvidenceArtifactError::Malformed {
            reason: format!("cannot derive table: {error}"),
        })?;
    Ok(artifact.human_table())
}

/// Explicit read-only loader for legacy v3 gate artifacts.
///
/// This is the only sanctioned path for consuming old artifacts. It never
/// converts them into current evidence and never seals or re-persists them.
///
/// # Errors
///
/// Returns [`EvidenceArtifactError::SchemaMismatch`] when the file is not a
/// v3 artifact and [`EvidenceArtifactError::Malformed`] when it does not
/// parse.
pub fn load_legacy_gate_artifact_v3(
    path: &Path,
) -> Result<PerfGateArtifact, EvidenceArtifactError> {
    let contents = fs::read_to_string(path)?;
    let probe: serde_json::Value =
        serde_json::from_str(&contents).map_err(|error| EvidenceArtifactError::Malformed {
            reason: format!("legacy artifact is not valid JSON: {error}"),
        })?;
    let found = probe
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<missing>");
    if found != LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3 {
        return Err(EvidenceArtifactError::SchemaMismatch {
            found: found.to_owned(),
        });
    }
    Ok(serde_json::from_str(&contents)?)
}

fn write_file_atomic(
    dir: &Path,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), EvidenceArtifactError> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| EvidenceArtifactError::Malformed {
            reason: "artifact destination has no file name".to_owned(),
        })?;
    let temp_path = dir.join(format!("{file_name}.tmp"));
    let mut file = File::create(&temp_path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp_path, destination)?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

/// Typed fail-closed errors for evidence assembly, persistence, and loading.
#[derive(Debug, Error)]
pub enum EvidenceArtifactError {
    /// The artifact bytes are not a parseable artifact at all.
    #[error("evidence artifact is malformed: {reason}")]
    Malformed {
        /// Bounded parse failure description.
        reason: String,
    },
    /// The artifact carries a non-current schema version.
    #[error("evidence artifact schema is {found}; current is quill-perf-evidence-v3")]
    SchemaMismatch {
        /// The version string found in the file.
        found: String,
    },
    /// The embedded hash seal does not match the artifact contents.
    #[error("evidence artifact hash seal does not match its contents")]
    HashMismatch,
    /// Predeclared evidence policy is unusable.
    #[error("evidence policy is invalid: {reason}")]
    InvalidPolicy {
        /// Bounded description.
        reason: String,
    },
    /// Provenance failed validation.
    #[error("evidence provenance is invalid: {reason}")]
    InvalidProvenance {
        /// Bounded description.
        reason: String,
    },
    /// A cell retained more raw samples than the predeclared bound.
    #[error("evidence cell {cell_id} retains {count} raw samples; bound is {max}")]
    UnboundedRawSamples {
        /// Offending cell.
        cell_id: String,
        /// Retained sample count.
        count: usize,
        /// Predeclared bound.
        max: usize,
    },
    /// A promotion decision was requested on non-eligible evidence.
    #[error("gate evidence is not claim-eligible")]
    NotClaimEligible,
    /// Stored summaries and raw contents disagree.
    #[error("evidence artifact does not recompute: {reason}")]
    InconsistentArtifact {
        /// Bounded description.
        reason: String,
    },
    /// Paired estimator rejected the raw streams.
    #[error(transparent)]
    Estimator(#[from] PairedEstimatorError),
    /// Filesystem failure during atomic persistence or loading.
    #[error("evidence artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization failure.
    #[error("evidence artifact serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
}

#[cfg(test)]
pub mod qg6_test_fixture {
    use super::*;
    use crate::perf::Qg6SampleBinding;
    use crate::qg6_prepared::{
        Qg6ExperimentIdentity, Qg6FourArmResultReceipts, Qg6RankedHitReceipt, Qg6ResultReceipt,
        query_manifest_sha256,
    };

    pub fn contract(
        query_class: crate::PerfQueryClass,
    ) -> (PerfInputIdentity, Qg6SemanticContract) {
        contract_for(query_class, 100_000, 10)
    }

    pub fn contract_for(
        query_class: crate::PerfQueryClass,
        document_count: u64,
        k: usize,
    ) -> (PerfInputIdentity, Qg6SemanticContract) {
        let queries =
            Qg6QuerySpec::normative_for_class(query_class).expect("frozen QG-6 query slice");
        let receipts = queries
            .iter()
            .map(|query| {
                let hit = Qg6RankedHitReceipt {
                    document_id_sha256: lower_hex(&Sha256::digest(
                        format!("{}-document", query.id()).as_bytes(),
                    )),
                    score_bits: 1.0_f32.to_bits(),
                };
                let receipt = Qg6ResultReceipt::from_redacted_hits(vec![hit], 1, document_count, k)
                    .expect("sealed QG-6 result receipt");
                Qg6FourArmResultReceipts {
                    null_left: receipt.clone(),
                    null_right: receipt.clone(),
                    effect_control: receipt.clone(),
                    effect_treatment: receipt,
                }
            })
            .collect::<Vec<_>>();
        let experiment_identity = Qg6ExperimentIdentity {
            corpus_sha256: "a".repeat(64),
            query_manifest_sha256: query_manifest_sha256(&queries),
            config_contract_sha256: "f".repeat(64),
            document_count,
            k,
        };
        let contract =
            Qg6SemanticContract::from_receipts(&experiment_identity, &queries, &receipts)
                .expect("sealed QG-6 semantic contract");
        let identity = PerfInputIdentity {
            prepared_corpus_sha256: contract.prepared_corpus_sha256.clone(),
            query_manifest_sha256: contract.query_manifest_sha256.clone(),
            config_contract_sha256: contract.config_contract_sha256.clone(),
            semantic_contract_sha256: Some(contract.contract_sha256.clone()),
            query_group_count: QG6_QUERY_GROUPS,
            query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
        };
        (identity, contract)
    }

    pub fn attach_stream(
        samples: &mut [PerfRawSample],
        effect_stream: bool,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
    ) {
        for sample in samples {
            let group_id = sample.group_id.expect("QG-6 fixture group");
            let group_index = usize::try_from(group_id).expect("QG-6 group index");
            let group = &contract.groups[group_index];
            let role = qg6_role(effect_stream, sample.arm);
            sample.work_units = Some(1);
            sample.provenance.input_identity = Some(identity.clone());
            sample.qg6_sample_binding = Some(Qg6SampleBinding {
                query_id: group.query.query_id.clone(),
                result_sequence_sha256: qg6_result_sequence_sha256(group.roles.get(role), 1)
                    .expect("QG-6 sequence digest"),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{
        PerfCellResult, PerfMetricSemantics, PerfOperationScope, PerfSampleArm, PerfSampleOrder,
        PerfSamplePhase, PerfSampleProvenance, QG6_QUERY_GROUP_IDS, estimate_paired_experiment,
        seeded_balanced_pair_order,
    };
    use crate::qg6_prepared::Qg6ResultReceipt;

    const CANARY: &str = "CANARY_DOCUMENT_TEXT_MUST_NEVER_PERSIST";

    fn scope() -> PerfOperationScope {
        PerfOperationScope {
            operation_id: "qg.synthetic_gauge".to_owned(),
            version: 1,
            semantics: PerfMetricSemantics::GaugeLowerIsBetter,
            unit: "ms".to_owned(),
        }
    }

    fn sample_provenance(run_id: &str) -> PerfSampleProvenance {
        PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            input_identity: None,
            worker_id: "test-worker".to_owned(),
            build_profile: "test".to_owned(),
        }
    }

    fn config() -> PairedEstimatorConfig {
        PairedEstimatorConfig::predeclared(0x5eed_0001)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_gauge_block(
        samples: &mut Vec<PerfRawSample>,
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        block_id: u64,
        sample_id_base: u64,
        control_value: f64,
        treatment_value: f64,
        control_first: bool,
        group_id: Option<u64>,
    ) {
        let base = block_id * 10_000;
        let (control_start, treatment_start) = if control_first {
            (base, base + 200)
        } else {
            (base + 200, base)
        };
        samples.push(PerfRawSample {
            block_id,
            sample_id: sample_id_base,
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
            observed_value: Some(control_value),
            group_id,
            qg6_sample_binding: None,
        });
        samples.push(PerfRawSample {
            block_id,
            sample_id: sample_id_base + 1,
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
            observed_value: Some(treatment_value),
            group_id,
            qg6_sample_binding: None,
        });
    }

    fn gauge_stream(
        pairs: &[(f64, f64)],
        sample_id_base: u64,
        block_id_base: u64,
        group_id: Option<u64>,
    ) -> Vec<PerfRawSample> {
        let scope = scope();
        let provenance = sample_provenance("run-a");
        let order = seeded_balanced_pair_order(pairs.len(), 0x00c0_ffee).expect("order");
        let mut samples = Vec::with_capacity(pairs.len() * 2);
        for (index, ((control, treatment), first_arm)) in pairs.iter().zip(order).enumerate() {
            let index = u64::try_from(index).expect("index");
            push_gauge_block(
                &mut samples,
                &scope,
                &provenance,
                block_id_base + index,
                sample_id_base + index * 2,
                *control,
                *treatment,
                first_arm == PerfSampleArm::Control,
                group_id,
            );
        }
        samples
    }

    fn grouped_gauge_stream(
        pairs: &[(u64, f64, f64)],
        sample_id_base: u64,
        force_control_first: Option<bool>,
    ) -> Vec<PerfRawSample> {
        let scope = scope();
        let provenance = sample_provenance("run-a");
        let order = seeded_balanced_pair_order(pairs.len(), 0x00c0_ffee).expect("order");
        let mut samples = Vec::with_capacity(pairs.len() * 2);
        for (index, ((group_id, control, treatment), first_arm)) in
            pairs.iter().zip(order).enumerate()
        {
            let block_id = u64::try_from(index).expect("block index");
            push_gauge_block(
                &mut samples,
                &scope,
                &provenance,
                block_id,
                sample_id_base + block_id * 2,
                *control,
                *treatment,
                force_control_first.unwrap_or(first_arm == PerfSampleArm::Control),
                Some(*group_id),
            );
        }
        samples
    }

    fn quiet_null_pairs(count: usize) -> Vec<(f64, f64)> {
        (0..count)
            .map(|index| {
                let epsilon = if index % 2 == 0 { 0.002 } else { -0.002 };
                (100.0, 100.0 * (1.0 + epsilon))
            })
            .collect()
    }

    fn effect_pairs(count: usize, ratio: f64) -> Vec<(f64, f64)> {
        (0..count).map(|_| (100.0, 100.0 * ratio)).collect()
    }

    fn valid_experiment(ratio: f64) -> PairedExperimentResult {
        let effect = gauge_stream(&effect_pairs(12, ratio), 0, 0, None);
        let null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        estimate_paired_experiment(&effect, &null, &config()).expect("valid experiment")
    }

    fn valid_treatment_arm_null_experiment(ratio: f64) -> PairedExperimentResult {
        let effect = gauge_stream(&effect_pairs(12, ratio), 0, 0, None);
        let null = gauge_stream(&quiet_null_pairs(12), 20_000, 20_000, None);
        estimate_paired_experiment(&effect, &null, &config())
            .expect("valid treatment-arm null experiment")
    }

    fn policy() -> EvidencePolicy {
        EvidencePolicy::predeclared()
    }

    fn build_identity() -> BuildIdentity {
        BuildIdentity {
            executable_sha256: "a".repeat(64),
            git_revision: "d".repeat(40),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: Some("c".repeat(64)),
            command_sha256: "f".repeat(64),
            environment_sha256: Some("e".repeat(64)),
            rustc_version: "rustc 1.91.0-nightly".to_owned(),
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            build_profile: "test".to_owned(),
            cargo_features: vec!["perf-harness".to_owned()],
        }
    }

    fn evidence_provenance() -> EvidenceProvenance {
        EvidenceProvenance {
            run_id: "run-a".to_owned(),
            run_window: "window-1".to_owned(),
            manifest_sha256: "e".repeat(64),
            build: build_identity(),
            machine: MachineIdentity {
                fingerprint: "test-machine".to_owned(),
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                logical_cpus: 8,
                execution: PerfExecutionProvenance {
                    host_identity: "test-machine".to_owned(),
                    producer_os: crate::PerfProducerOs::Linux,
                    physical_cores: 4,
                    logical_threads: 8,
                    process_available_threads: 8,
                    configured_engine_thread_widths: vec![1],
                    runtime_detected_isa: vec!["avx2".to_owned()],
                    cpu_affinity_allowed_list: Some("0-7".to_owned()),
                    affinity_or_cpuset_cap: None,
                },
                cpu_governor: None,
                load_average_start: Some(0.5),
                load_average_end: Some(0.6),
            },
            peak_rss: PeakRssEvidence {
                method: "unsupported".to_owned(),
                bytes: None,
            },
            corpus: CorpusIdentity {
                corpus_sha256: "b".repeat(64),
                query_set_sha256: Some("d".repeat(64)),
                qrels_sha256: None,
                document_count: 500,
                content_bytes: Some(1_000_000),
                generator_seed: 42,
                generator_revision: "zipf-s11-v1".to_owned(),
            },
        }
    }

    fn seal_unbound_artifact(artifact: &mut PerfEvidenceArtifact) -> Vec<u8> {
        let bytes = artifact
            .sealed_json()
            .expect("seal unbound test evidence")
            .into_bytes();
        *artifact = PerfEvidenceArtifact::from_verified_slice(&bytes)
            .expect("reload unbound test evidence");
        bytes
    }

    fn admitted_identity(
        gate: PerfGate,
        threshold_artifact_bytes: &[u8],
        evidence_artifact_bytes: &[u8],
        run_label: &str,
    ) -> VerifiedRunnerIdentity {
        crate::machine_class_registry::admitted_test_identity_for_artifacts(
            gate.label(),
            &"d".repeat(40),
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"e".repeat(64),
            run_label,
            "run-a",
            "window-1",
            threshold_artifact_bytes,
            evidence_artifact_bytes,
        )
    }

    fn bind_test_identity(
        artifact: &mut PerfEvidenceArtifact,
        gate: PerfGate,
        threshold_artifact_bytes: &[u8],
        run_label: &str,
    ) -> Vec<u8> {
        let source = seal_unbound_artifact(artifact);
        let identity = admitted_identity(gate, threshold_artifact_bytes, &source, run_label);
        artifact
            .bind_machine_class_identity(identity, threshold_artifact_bytes, &source)
            .expect("bind admitted test identity");
        source
    }

    fn unbind_test_artifact(artifact: &mut PerfEvidenceArtifact) {
        artifact.machine_class =
            MachineClassEvidenceBinding::unverified("sealed runner receipt has not been bound");
        artifact.gate_decision = None;
        artifact.artifact_sha256.clear();
    }

    fn cell_spec(gate: PerfGate, role: EvidenceRole) -> EvidenceCellSpec {
        let (input_identity, qg6_semantic_contract, fixture) = if gate == PerfGate::Qg6 {
            let (identity, contract) =
                qg6_test_fixture::contract(crate::PerfQueryClass::Identifier);
            (
                Some(identity),
                Some(contract),
                "query/identifier/k10/100k".to_owned(),
            )
        } else {
            (None, None, "bulk/synthetic/1".to_owned())
        };
        let concurrency_witness = (gate == PerfGate::Qg8
            || (gate == PerfGate::Qg1 && role == EvidenceRole::Required))
            .then(|| PerfConcurrencyWitness {
                configured_threads: 1,
                observations: vec![
                    EngineConcurrencyObservation {
                        engine: PerfConcurrencyEngine::Quill,
                        observer: PerfConcurrencyObserver::RayonCurrentPoolWidth,
                        observation_count: 12,
                        min_observed_worker_pool_threads: 1,
                        max_observed_worker_pool_threads: 1,
                    },
                    EngineConcurrencyObservation {
                        engine: PerfConcurrencyEngine::Tantivy,
                        observer: PerfConcurrencyObserver::TantivyWriterConstruction,
                        observation_count: 12,
                        min_observed_worker_pool_threads: 1,
                        max_observed_worker_pool_threads: 1,
                    },
                ],
            });
        EvidenceCellSpec {
            gate,
            fixture,
            metric: "latency_ms".to_owned(),
            unit: "ms".to_owned(),
            role,
            input_identity,
            qg6_semantic_contract,
            cold_cache: None,
            concurrency_witness,
        }
    }

    fn provisional_cell() -> EvidenceCell {
        let mut cell = EvidenceCell::evaluate(
            cell_spec(PerfGate::Qg1, EvidenceRole::Required),
            valid_experiment(1.10),
            &policy(),
        )
        .expect("provisional cell");
        cell.attach_treatment_arm_null(valid_treatment_arm_null_experiment(1.10), &policy())
            .expect("attach QG-1 treatment-arm null");
        cell
    }

    fn provisional_artifact() -> PerfEvidenceArtifact {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![provisional_cell()],
        )
        .expect("provisional artifact");
        bind_test_identity(
            &mut artifact,
            PerfGate::Qg1,
            b"qg1-threshold",
            "qg1-primary",
        );
        artifact
    }

    fn qg6_artifact() -> PerfEvidenceArtifact {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let input_identity = spec.input_identity.as_ref().expect("QG-6 input identity");
        let semantic_contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect = qg6_hierarchical_stream_with_ratio(1.02, 0);
        let mut null = qg6_hierarchical_stream_with_ratio(1.0, 10_000);
        qg6_test_fixture::attach_stream(&mut effect, true, input_identity, semantic_contract);
        qg6_test_fixture::attach_stream(&mut null, false, input_identity, semantic_contract);
        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        let cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        let mut provenance = evidence_provenance();
        provenance.corpus.query_set_sha256 = Some("d".repeat(64));
        let mut artifact =
            PerfEvidenceArtifact::assemble(PerfGate::Qg6, policy(), provenance, vec![cell])
                .expect("QG-6 artifact");
        bind_test_identity(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-primary",
        );
        artifact
    }

    fn fully_reseal_qg6_query_mutation(
        artifact: &mut PerfEvidenceArtifact,
        mutate: impl FnOnce(&mut Qg6QueryIdentityReceipt),
    ) {
        unbind_test_artifact(artifact);
        let cell = &mut artifact.cells[0];
        let contract = cell
            .spec
            .qg6_semantic_contract
            .as_mut()
            .expect("QG-6 semantic contract");
        mutate(&mut contract.groups[0].query);
        contract.groups[0].query.query_identity_sha256 =
            contract.groups[0].query.canonical_sha256();
        contract.query_manifest_sha256 = crate::qg6_prepared::query_identity_manifest_sha256(
            contract.groups.iter().map(|group| &group.query),
        );
        contract.contract_sha256 = contract.canonical_sha256();
        let query_ids = contract
            .groups
            .iter()
            .map(|group| group.query.query_id.clone())
            .collect::<Vec<_>>();

        let identity = cell
            .spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity");
        identity.query_manifest_sha256 = contract.query_manifest_sha256.clone();
        identity.semantic_contract_sha256 = Some(contract.contract_sha256.clone());
        let identity = identity.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut cell.body else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.input_identity = Some(identity.clone());
        for sample in paired
            .effect_samples
            .iter_mut()
            .chain(&mut paired.null_samples)
        {
            sample.provenance.input_identity = Some(identity.clone());
            let group_index =
                usize::try_from(sample.group_id.expect("QG-6 sample group")).expect("group index");
            sample
                .qg6_sample_binding
                .as_mut()
                .expect("QG-6 compact binding")
                .query_id = query_ids[group_index].clone();
        }
    }

    fn fully_reseal_qg6_result_receipt_mutation(
        artifact: &mut PerfEvidenceArtifact,
        mutate: impl FnOnce(&mut Qg6ResultReceipt),
    ) {
        unbind_test_artifact(artifact);
        let cell = &mut artifact.cells[0];
        let contract = cell
            .spec
            .qg6_semantic_contract
            .as_mut()
            .expect("QG-6 semantic contract");
        let receipt = &mut contract.groups[0].roles.effect_treatment;
        mutate(receipt);
        receipt.reseal_for_test();
        contract.contract_sha256 = contract.canonical_sha256();
        let contract = contract.clone();

        let identity = cell
            .spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity");
        identity.semantic_contract_sha256 = Some(contract.contract_sha256.clone());
        let identity = identity.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut cell.body else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.input_identity = Some(identity.clone());
        let rebind_stream = |samples: &mut [PerfRawSample], effect_stream: bool| {
            for sample in samples {
                sample.provenance.input_identity = Some(identity.clone());
                let group_index = usize::try_from(sample.group_id.expect("QG-6 sample group"))
                    .expect("group index");
                let group = &contract.groups[group_index];
                let work_units = sample.work_units.expect("QG-6 sample work units");
                let role = qg6_role(effect_stream, sample.arm);
                let binding = sample
                    .qg6_sample_binding
                    .as_mut()
                    .expect("QG-6 compact binding");
                binding.query_id.clone_from(&group.query.query_id);
                binding.result_sequence_sha256 =
                    qg6_result_sequence_sha256(group.roles.get(role), work_units)
                        .expect("fully resealed result sequence");
            }
        };
        rebind_stream(&mut paired.effect_samples, true);
        rebind_stream(&mut paired.null_samples, false);
    }

    #[test]
    fn severity_precedence_is_total_and_fatal_dominates() {
        let mut severities = [
            EvidenceSeverity::Fatal,
            EvidenceSeverity::Allow,
            EvidenceSeverity::Block,
            EvidenceSeverity::NoClaim,
            EvidenceSeverity::Quarantine,
        ];
        severities.sort();
        assert_eq!(
            severities,
            [
                EvidenceSeverity::Allow,
                EvidenceSeverity::NoClaim,
                EvidenceSeverity::Quarantine,
                EvidenceSeverity::Block,
                EvidenceSeverity::Fatal,
            ]
        );
        let folded = severities.iter().copied().max().expect("non-empty");
        assert_eq!(folded, EvidenceSeverity::Fatal);
    }

    #[test]
    fn required_estimand_is_metric_specific() {
        assert_eq!(
            required_estimand(PerfGate::Qg1),
            EvidenceEstimand::PairedLogRatio
        );
        assert_eq!(
            required_estimand(PerfGate::Qg6),
            EvidenceEstimand::HierarchicalLatency
        );
        assert_eq!(
            required_estimand(PerfGate::Qg7),
            EvidenceEstimand::ProcessRss
        );
        assert_eq!(required_estimand(PerfGate::Qg9), EvidenceEstimand::ColdOpen);
        assert_eq!(
            required_estimand(PerfGate::Qg10),
            EvidenceEstimand::DependencyFacts
        );
    }

    #[test]
    fn known_answer_effect_recovers_ten_percent() {
        let experiment = valid_experiment(1.10);
        assert_eq!(experiment.status, PairedEvidenceStatus::Valid);
        assert!((experiment.effect.median_log_ratio - 1.10_f64.ln()).abs() < 1.0e-9);
        assert!((experiment.effect.treatment_over_control - 1.10).abs() < 1.0e-9);
        let cell = provisional_cell();
        assert_eq!(cell.status, EvidenceDecisionStatus::MeasuredProvisional);
        assert!(cell.claim_eligible());
        assert!(cell.reasons.is_empty());
        match &cell.body {
            EvidenceCellBody::Paired {
                treatment_arm_null,
                reconciliation,
                ..
            } => {
                assert!(
                    treatment_arm_null
                        .as_ref()
                        .is_some_and(|null| null.status == PairedEvidenceStatus::Valid)
                );
                assert!(reconciliation.direction_agrees);
                assert!(reconciliation.within_tolerance);
            }
            EvidenceCellBody::Facts { .. } => panic!("expected a paired body"),
        }
    }

    #[test]
    fn invalid_null_stays_durable_and_cannot_claim_or_ratchet() {
        let noisy_null = (0..12)
            .map(|index| {
                let epsilon: f64 = if index % 2 == 0 { 0.35 } else { -0.30 };
                (100.0, 100.0 * (1.0 + epsilon))
            })
            .collect::<Vec<_>>();
        let effect = gauge_stream(&effect_pairs(12, 1.10), 0, 0, None);
        let null = gauge_stream(&noisy_null, 10_000, 0, None);
        let experiment = estimate_paired_experiment(&effect, &null, &config()).expect("estimate");
        assert_eq!(experiment.status, PairedEvidenceStatus::InvalidNull);
        assert_eq!(experiment.claim_state, PairedClaimState::NoDecision);

        let cell = EvidenceCell::evaluate(
            cell_spec(PerfGate::Qg1, EvidenceRole::Required),
            experiment,
            &policy(),
        )
        .expect("cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::InvalidNull);
        assert!(!cell.claim_eligible());

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("artifact");
        assert_eq!(artifact.gate_status, EvidenceDecisionStatus::InvalidNull);
        assert!(!artifact.ratchet_admissible());
        assert!(matches!(
            artifact.apply_gate_decision(EvidenceDecisionStatus::Allow),
            Err(EvidenceArtifactError::NotClaimEligible)
        ));

        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact
            .write_atomic(dir.path())
            .expect("durable invalid run");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("reload");
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::InvalidNull);
    }

    #[test]
    fn unverified_machine_binding_is_explicit_durable_and_nonpromotable() {
        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![provisional_cell()],
        )
        .expect("unverified evidence artifact");
        assert!(matches!(
            &artifact.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(!artifact.ratchet_admissible());

        let directory = tempfile::tempdir().expect("unverified evidence directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("persist explicit unverified evidence");
        let reloaded =
            PerfEvidenceArtifact::load_verified(&paths.json).expect("reload unverified evidence");
        assert!(matches!(
            &reloaded.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(!reloaded.ratchet_admissible());
    }

    #[test]
    fn post_exit_binding_returns_exact_verified_receipt_bound_bytes() {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![provisional_cell()],
        )
        .expect("unverified producer evidence");
        let threshold_bytes = b"qg1-threshold";
        let source = seal_unbound_artifact(&mut artifact);
        let identity =
            admitted_identity(PerfGate::Qg1, threshold_bytes, &source, "post-exit-primary");
        let bytes = artifact
            .bind_machine_class_identity_and_seal(identity, threshold_bytes, &source)
            .expect("post-exit receipt binding");

        assert_eq!(
            bytes,
            serde_json::to_vec_pretty(&artifact).expect("bound artifact JSON")
        );
        artifact.verify_integrity().expect("bound artifact seal");
        assert!(artifact.ratchet_admissible());
        assert!(artifact.machine_class.identity().is_some_and(|identity| {
            bytes
                .windows(identity.receipt_sha256().len())
                .any(|window| window == identity.receipt_sha256().as_bytes())
        }));
    }

    #[test]
    fn nul_delimited_argv_digest_drift_rejects_post_exit_binding() {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![provisional_cell()],
        )
        .expect("unverified producer evidence");
        let threshold_bytes = b"qg1-threshold";
        let source = seal_unbound_artifact(&mut artifact);
        let drifted_argv_identity =
            crate::machine_class_registry::admitted_test_identity_for_artifacts(
                PerfGate::Qg1.label(),
                &"d".repeat(40),
                &"c".repeat(64),
                &"a".repeat(64),
                &"0".repeat(64),
                &"e".repeat(64),
                "drifted-command",
                "run-a",
                "window-1",
                threshold_bytes,
                &source,
            );

        assert!(matches!(
            artifact.bind_machine_class_identity_and_seal(
                drifted_argv_identity,
                threshold_bytes,
                &source,
            ),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("build identity differs")
        ));
        assert!(matches!(
            artifact.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(!artifact.artifact_sha256.is_empty());
    }

    #[test]
    fn verified_runner_binding_cannot_be_reassigned_to_another_receipt() {
        let mut artifact = provisional_artifact();
        let original = artifact.machine_class.clone();
        let source = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstruct pre-binding source");
        let different_receipt = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg1.label(),
            &"d".repeat(40),
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"e".repeat(64),
            "different-completion",
            "run-a",
            "window-1",
            b"qg1-threshold",
            &source,
        );

        assert!(matches!(
            artifact.bind_machine_class_identity_and_seal(
                different_receipt,
                b"qg1-threshold",
                &source,
            ),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("in-memory evidence differs")
        ));
        assert_eq!(artifact.machine_class, original);
        assert!(artifact.artifact_sha256.is_empty());
    }

    #[test]
    fn command_digest_uses_unambiguous_nul_separation_and_final_terminator() {
        let digest = command_sha256_from_argv([b"cargo".as_slice(), b"bench".as_slice()]);
        assert_eq!(digest, lower_hex(&Sha256::digest(b"cargo\0bench\0")));
        assert_ne!(
            command_sha256_from_argv([b"ab".as_slice(), b"c".as_slice()]),
            command_sha256_from_argv([b"a".as_slice(), b"bc".as_slice()])
        );
        assert_ne!(
            command_sha256_from_argv([b"cargo".as_slice(), b"bench".as_slice()]),
            lower_hex(&Sha256::digest(b"cargo bench"))
        );
    }

    #[test]
    fn outer_reseal_cannot_hide_tampered_embedded_runner_receipt() {
        let mut artifact = provisional_artifact();
        let mut binding = serde_json::to_value(&artifact.machine_class).expect("binding JSON");
        let receipt_json = binding["identity"]["receipt_json"]
            .as_str()
            .expect("embedded receipt JSON");
        let mut receipt: serde_json::Value =
            serde_json::from_str(receipt_json).expect("parse embedded receipt");
        receipt["build"]["git_revision"] = serde_json::Value::String("tampered".to_owned());
        binding["identity"]["receipt_json"] = serde_json::Value::String(
            serde_json::to_string(&receipt).expect("serialize tampered receipt"),
        );
        artifact.machine_class =
            serde_json::from_value(binding).expect("deserialize tampered binding");

        let directory = tempfile::tempdir().expect("tampered binding directory");
        let path = directory.path().join("tampered-binding.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("reseal artifact around tampered binding"),
        )
        .expect("persist tampered binding");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));
    }

    #[test]
    fn nested_unknown_field_cannot_bypass_the_outer_artifact_seal() {
        let artifact = provisional_artifact();
        let directory = tempfile::tempdir().expect("unknown-field directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("write artifact");
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.json).expect("artifact bytes"))
                .expect("artifact JSON");
        value["provenance"]["build"]["unreviewed_identity"] =
            serde_json::Value::String("must-not-be-ignored".to_owned());
        fs::write(
            &paths.json,
            serde_json::to_vec_pretty(&value).expect("unknown-field JSON"),
        )
        .expect("persist unknown-field artifact");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&paths.json),
            Err(EvidenceArtifactError::Malformed { reason })
                if reason.contains("unknown fields")
        ));
    }

    #[test]
    fn absolute_relative_contradiction_is_no_decision() {
        let mut pairs = Vec::new();
        for _ in 0..6 {
            pairs.push((1.0, 1.2));
        }
        for _ in 0..6 {
            pairs.push((100.0, 90.0));
        }
        let effect = gauge_stream(&pairs, 0, 0, None);
        let null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        let experiment = estimate_paired_experiment(&effect, &null, &config()).expect("estimate");
        assert_eq!(
            experiment.status,
            PairedEvidenceStatus::ContradictorySummaries
        );
        let cell = EvidenceCell::evaluate(
            cell_spec(PerfGate::Qg1, EvidenceRole::Required),
            experiment,
            &policy(),
        )
        .expect("cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(
            cell.reasons
                .iter()
                .any(|reason| reason.code == "evidence.absolute_relative_direction_conflict")
        );
    }

    fn hierarchical_stream() -> Vec<PerfRawSample> {
        hierarchical_stream_with_ratio(1.10, 0)
    }

    fn hierarchical_stream_with_ratio(ratio: f64, sample_id_base: u64) -> Vec<PerfRawSample> {
        let mut samples = Vec::new();
        let group_scales = [1.0, 10.0, 100.0, 1_000.0];
        let mut sample_id = sample_id_base;
        for (group_index, scale) in group_scales.iter().enumerate() {
            let group = u64::try_from(group_index).expect("group index");
            let jitter = 0.005 * (group_index as f64 + 1.0);
            let pairs = (0..5)
                .map(|block| {
                    let wobble = if block % 2 == 0 { jitter } else { -jitter };
                    let control = scale * (1.0 + wobble);
                    (control, control * ratio)
                })
                .collect::<Vec<_>>();
            let block_base = group * 100;
            samples.extend(gauge_stream(&pairs, sample_id, block_base, Some(group)));
            sample_id += u64::try_from(pairs.len() * 2).expect("sample count");
        }
        samples
    }

    fn qg6_hierarchical_stream_with_ratio(ratio: f64, sample_id_base: u64) -> Vec<PerfRawSample> {
        let mut samples = Vec::new();
        let group_scales = [1.0, 10.0, 100.0, 1_000.0];
        let mut sample_id = sample_id_base;
        for (group_index, group_id) in QG6_QUERY_GROUP_IDS.into_iter().enumerate() {
            let scale = group_scales[group_index % group_scales.len()];
            let jitter = [0.005, 0.010, 0.015, 0.020][group_index % 4];
            let pairs = (0..5)
                .map(|block| {
                    let wobble = if block % 2 == 0 { jitter } else { -jitter };
                    let control = scale * (1.0 + wobble);
                    (control, control * ratio)
                })
                .collect::<Vec<_>>();
            samples.extend(gauge_stream(
                &pairs,
                sample_id,
                group_id * 100,
                Some(group_id),
            ));
            sample_id += u64::try_from(pairs.len() * 2).expect("sample count");
        }
        samples
    }

    #[test]
    fn hierarchical_known_answer_covers_heteroskedastic_groups() {
        let estimate = estimate_hierarchical_latency(&hierarchical_stream(), &config(), &policy())
            .expect("hierarchical estimate");
        assert_eq!(estimate.group_count, 4);
        assert_eq!(estimate.total_pairs, 20);
        let truth = 1.10_f64.ln();
        assert!((estimate.median_of_group_medians_log - truth).abs() < 0.02);
        assert!(estimate.ci95_low_log <= truth && truth <= estimate.ci95_high_log);
        assert_eq!(estimate.groups.len(), 4);
        for group in &estimate.groups {
            assert_eq!(group.pair_count, 5);
            assert!((group.median_log_ratio - truth).abs() < 0.02);
        }
    }

    #[test]
    fn hierarchical_misaligned_ids_fail_closed() {
        let mut missing = hierarchical_stream();
        missing[0].group_id = None;
        assert!(matches!(
            estimate_hierarchical_latency(&missing, &config(), &policy()),
            Err(PairedEstimatorError::MissingGroupId { .. })
        ));

        let mut mixed = hierarchical_stream();
        mixed[1].group_id = Some(999);
        assert!(matches!(
            estimate_hierarchical_latency(&mixed, &config(), &policy()),
            Err(PairedEstimatorError::GroupMismatch { .. })
        ));

        let single_group = gauge_stream(&effect_pairs(12, 1.10), 0, 0, Some(7));
        assert!(matches!(
            estimate_hierarchical_latency(&single_group, &config(), &policy()),
            Err(PairedEstimatorError::InsufficientGroups { .. })
        ));
    }

    #[test]
    fn qg6_rows_without_groups_or_semantic_bindings_fail_closed() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let effect = gauge_stream(&effect_pairs(12, 1.10), 0, 0, None);
        let null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        let experiment =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 estimate");
        assert!(matches!(
            EvidenceCell::evaluate(spec, experiment, &policy()),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn qg6_rejects_each_prepared_input_identity_mismatch_independently() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let expected_identity = spec.input_identity.as_ref().expect("QG-6 input identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect = qg6_hierarchical_stream_with_ratio(1.02, 0);
        let mut null = qg6_hierarchical_stream_with_ratio(1.0, 10_000);
        qg6_test_fixture::attach_stream(&mut effect, true, expected_identity, contract);
        qg6_test_fixture::attach_stream(&mut null, false, expected_identity, contract);
        let experiment =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 estimate");

        for field in [
            "prepared_corpus_sha256",
            "query_manifest_sha256",
            "config_contract_sha256",
            "semantic_contract_sha256",
        ] {
            let mut corrupted = experiment.clone();
            let identity = corrupted
                .provenance
                .input_identity
                .as_mut()
                .expect("QG-6 prepared-input identity");
            match field {
                "prepared_corpus_sha256" => identity.prepared_corpus_sha256 = "0".repeat(64),
                "query_manifest_sha256" => identity.query_manifest_sha256 = "1".repeat(64),
                "config_contract_sha256" => identity.config_contract_sha256 = "2".repeat(64),
                "semantic_contract_sha256" => {
                    identity.semantic_contract_sha256 = Some("3".repeat(64));
                }
                _ => unreachable!("enumerated identity field"),
            }
            let error = EvidenceCell::evaluate(spec.clone(), corrupted, &policy())
                .expect_err("one independently changed prepared input must fail closed");
            assert!(
                matches!(
                    error,
                    EvidenceArtifactError::InconsistentArtifact { ref reason }
                        if reason.contains("input identity")
                ),
                "{field} mismatch returned {error:?}"
            );
        }
    }

    #[test]
    fn qg6_verified_roundtrip_distinguishes_query_universe_from_exact_cell_manifest() {
        let artifact = qg6_artifact();
        let exact_query_manifest = artifact.cells[0]
            .spec
            .input_identity
            .as_ref()
            .expect("QG-6 input identity")
            .query_manifest_sha256
            .clone();
        let selection_query_universe = artifact
            .provenance
            .corpus
            .query_set_sha256
            .clone()
            .expect("selection query universe");
        assert_ne!(
            selection_query_universe, exact_query_manifest,
            "selection-wide query-universe provenance must not masquerade as one cell's ordered \
             query manifest"
        );

        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("persist QG-6 artifact");
        let verified =
            PerfEvidenceArtifact::load_verified(&paths.json).expect("verify QG-6 artifact");
        assert_eq!(
            verified.cells[0].spec.input_identity,
            match &verified.cells[0].body {
                EvidenceCellBody::Paired { paired, .. } => {
                    paired.provenance.input_identity.clone()
                }
                EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
            }
        );
    }

    #[test]
    fn current_schema_qg6_missing_new_semantic_fields_deserializes_but_fails_closed() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let cell = &mut artifact.cells[0];
        cell.spec.qg6_semantic_contract = None;
        cell.spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity")
            .semantic_contract_sha256 = None;
        let EvidenceCellBody::Paired { paired, .. } = &mut cell.body else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.input_identity = cell.spec.input_identity.clone();
        for sample in paired
            .effect_samples
            .iter_mut()
            .chain(&mut paired.null_samples)
        {
            sample.provenance.input_identity = cell.spec.input_identity.clone();
            sample.qg6_sample_binding = None;
        }
        let json = artifact
            .sealed_json()
            .expect("seal current-schema legacy-shaped QG-6");
        assert!(!json.contains("\"qg6_semantic_contract\""));
        assert!(!json.contains("\"semantic_contract_sha256\""));
        assert!(!json.contains("\"qg6_sample_binding\""));
        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-current-schema-old-shape.json");
        fs::write(&path, json).expect("persist old-shaped current schema");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn non_qg6_artifact_wire_shape_omits_all_semantic_receipt_fields() {
        let mut artifact = provisional_artifact();
        unbind_test_artifact(&mut artifact);
        let json = artifact.sealed_json().expect("seal non-QG-6 artifact");
        assert!(!json.contains("\"qg6_semantic_contract\""));
        assert!(!json.contains("\"semantic_contract_sha256\""));
        assert!(!json.contains("\"qg6_sample_binding\""));
        PerfEvidenceArtifact::from_verified_slice(json.as_bytes())
            .expect("unchanged non-QG-6 artifact roundtrip");
    }

    #[test]
    fn qg6_verified_load_rejects_sealed_cell_identity_divergence() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        artifact.cells[0]
            .spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity")
            .query_manifest_sha256 = "9".repeat(64);
        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-sealed-mismatch.json");
        fs::write(
            &path,
            artifact.sealed_json().expect("seal mismatched artifact"),
        )
        .expect("persist sealed mismatch");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn qg6_fully_resealed_query_mutations_cannot_escape_the_frozen_slice_anchor() {
        for mutation in [
            "query_id",
            "normalized_text",
            "parsed_ast",
            "coverage_row",
            "coverage_column",
            "support_contract",
            "query_generator_revision",
            "corpus_generator_revision",
        ] {
            let mut artifact = qg6_artifact();
            fully_reseal_qg6_query_mutation(&mut artifact, |query| match mutation {
                "query_id" => query.query_id = "identifier-000".to_owned(),
                "normalized_text" => query.normalized_text_sha256 = "9".repeat(64),
                "parsed_ast" => query.parsed_ast_sha256 = "8".repeat(64),
                "coverage_row" => query.coverage_row = 3,
                "coverage_column" => query.coverage_column = 3,
                "support_contract" => {
                    query.support_divergence =
                        crate::qg6_prepared::Qg6SupportDivergence::SupportedWithReviewedDivergence {
                            register_id: "hostile-resealed-fixture".to_owned(),
                            contract_sha256: "7".repeat(64),
                        };
                }
                "query_generator_revision" => {
                    query.query_generator_revision = "hostile-query-generator-v1".to_owned();
                }
                "corpus_generator_revision" => {
                    query.corpus_generator_revision = "hostile-corpus-generator-v1".to_owned();
                }
                _ => unreachable!("enumerated mutation"),
            });
            let directory = tempfile::tempdir().expect("QG-6 artifact directory");
            let path = directory
                .path()
                .join(format!("qg6-resealed-{mutation}.json"));
            fs::write(
                &path,
                artifact
                    .sealed_json()
                    .expect("outer-reseal hostile artifact"),
            )
            .expect("persist hostile artifact");

            assert!(
                matches!(
                    PerfEvidenceArtifact::load_verified(&path),
                    Err(EvidenceArtifactError::InconsistentArtifact { ref reason })
                        if reason.contains("frozen normative query slice")
                ),
                "fully re-sealed mutation {mutation} escaped the frozen anchor"
            );
        }
    }

    #[test]
    fn qg6_outer_reseal_rejects_invalid_class_supported_k_and_query_self_seal() {
        for mutation in ["class", "supported_k", "query_identity_sha256"] {
            let mut artifact = qg6_artifact();
            if mutation == "query_identity_sha256" {
                unbind_test_artifact(&mut artifact);
                artifact.cells[0]
                    .spec
                    .qg6_semantic_contract
                    .as_mut()
                    .expect("QG-6 semantic contract")
                    .groups[0]
                    .query
                    .query_identity_sha256 = "6".repeat(64);
            } else {
                fully_reseal_qg6_query_mutation(&mut artifact, |query| match mutation {
                    "class" => query.class = crate::PerfQueryClass::Boolean,
                    "supported_k" => query.supported_k = [10, 99],
                    _ => unreachable!("enumerated mutation"),
                });
            }
            let directory = tempfile::tempdir().expect("QG-6 artifact directory");
            let path = directory
                .path()
                .join(format!("qg6-invalid-query-{mutation}.json"));
            fs::write(
                &path,
                artifact
                    .sealed_json()
                    .expect("outer-reseal invalid query identity"),
            )
            .expect("persist invalid query identity");
            assert!(
                matches!(
                    PerfEvidenceArtifact::load_verified(&path),
                    Err(EvidenceArtifactError::InconsistentArtifact { .. })
                ),
                "invalid query identity field {mutation} escaped verification"
            );
        }
    }

    #[test]
    fn qg6_verified_load_rejects_fully_resealed_unsupported_contract_cutoff() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let cell = &mut artifact.cells[0];
        let contract = cell
            .spec
            .qg6_semantic_contract
            .as_mut()
            .expect("QG-6 semantic contract");
        contract.k = 99;
        contract.contract_sha256 = contract.canonical_sha256();
        let semantic_contract_sha256 = contract.contract_sha256.clone();

        cell.spec.fixture = "query/identifier/k99/100k".to_owned();
        cell.cell_id = format!(
            "{}/{}/{}",
            cell.spec.gate, cell.spec.fixture, cell.spec.metric
        );
        let identity = cell
            .spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity");
        identity.semantic_contract_sha256 = Some(semantic_contract_sha256);
        let identity = identity.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut cell.body else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.input_identity = Some(identity.clone());
        for sample in paired
            .effect_samples
            .iter_mut()
            .chain(&mut paired.null_samples)
        {
            sample.provenance.input_identity = Some(identity.clone());
        }

        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-resealed-unsupported-k99.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("outer-reseal unsupported cutoff"),
        )
        .expect("persist fully resealed unsupported cutoff");
        let error = PerfEvidenceArtifact::load_verified(&path)
            .expect_err("fully resealed unsupported cutoff must fail verified reload");
        assert!(
            matches!(
                error,
                EvidenceArtifactError::InconsistentArtifact { ref reason }
                    if reason.contains("semantic contract failed verification")
            ),
            "unsupported cutoff failed outside the semantic-contract boundary: {error}"
        );
    }

    #[test]
    fn qg6_outer_reseal_cannot_hide_any_nested_result_receipt_mutation() {
        for mutation in [
            "returned_count",
            "document_id",
            "score_bits",
            "total_count",
            "doc_count",
            "receipt_sha256",
        ] {
            let mut artifact = qg6_artifact();
            unbind_test_artifact(&mut artifact);
            let receipt = &mut artifact.cells[0]
                .spec
                .qg6_semantic_contract
                .as_mut()
                .expect("QG-6 semantic contract")
                .groups[0]
                .roles
                .effect_treatment;
            match mutation {
                "returned_count" => receipt.returned_count += 1,
                "document_id" => receipt.ordered_hits[0].document_id_sha256 = "6".repeat(64),
                "score_bits" => receipt.ordered_hits[0].score_bits = 2.0_f32.to_bits(),
                "total_count" => receipt.total_count += 1,
                "doc_count" => receipt.doc_count += 1,
                "receipt_sha256" => receipt.receipt_sha256 = "5".repeat(64),
                _ => unreachable!("enumerated mutation"),
            }
            let directory = tempfile::tempdir().expect("QG-6 artifact directory");
            let path = directory
                .path()
                .join(format!("qg6-receipt-{mutation}.json"));
            fs::write(
                &path,
                artifact
                    .sealed_json()
                    .expect("outer-reseal mutated receipt"),
            )
            .expect("persist mutated receipt");
            assert!(
                matches!(
                    PerfEvidenceArtifact::load_verified(&path),
                    Err(EvidenceArtifactError::InconsistentArtifact { .. })
                ),
                "outer seal hid nested receipt mutation {mutation}"
            );
        }
    }

    #[test]
    fn qg6_reloaded_artifact_rejects_fully_resealed_invalid_ranked_hits() {
        for mutation in [
            "underfilled_top_k",
            "empty_document_id",
            "duplicate_document_id",
            "non_finite_score",
        ] {
            let mut artifact = qg6_artifact();
            fully_reseal_qg6_result_receipt_mutation(&mut artifact, |receipt| match mutation {
                "underfilled_top_k" => receipt.total_count = 2,
                "empty_document_id" => {
                    receipt.ordered_hits[0].document_id_sha256 = lower_hex(&Sha256::digest(b""));
                }
                "duplicate_document_id" => {
                    receipt.ordered_hits.push(receipt.ordered_hits[0].clone());
                    receipt.returned_count = 2;
                    receipt.total_count = 2;
                }
                "non_finite_score" => {
                    receipt.ordered_hits[0].score_bits = f32::NAN.to_bits();
                }
                _ => unreachable!("enumerated mutation"),
            });
            let directory = tempfile::tempdir().expect("QG-6 artifact directory");
            let path = directory
                .path()
                .join(format!("qg6-resealed-invalid-hit-{mutation}.json"));
            fs::write(
                &path,
                artifact
                    .sealed_json()
                    .expect("outer-reseal invalid ranked hit"),
            )
            .expect("persist fully resealed invalid ranked hit");

            let error = PerfEvidenceArtifact::load_verified(&path)
                .expect_err("fully resealed invalid ranked hit must fail verified reload");
            assert!(
                matches!(&error, EvidenceArtifactError::InconsistentArtifact { .. }),
                "unexpected reload error for {mutation}: {error}"
            );
            assert!(
                error
                    .to_string()
                    .contains("QG-6 result receipt is malformed"),
                "invalid ranked hit {mutation} bypassed receipt verification: {error}"
            );
        }
    }

    #[test]
    fn qg6_outer_reseal_rejects_both_compact_binding_field_mutations() {
        for mutation in ["query_id", "result_sequence_sha256"] {
            let mut artifact = qg6_artifact();
            unbind_test_artifact(&mut artifact);
            let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body else {
                unreachable!("QG-6 must be paired");
            };
            let binding = paired.effect_samples[0]
                .qg6_sample_binding
                .as_mut()
                .expect("compact QG-6 binding");
            match mutation {
                "query_id" => binding.query_id = "identifier-hostile-binding".to_owned(),
                "result_sequence_sha256" => {
                    binding.result_sequence_sha256 = "4".repeat(64);
                }
                _ => unreachable!("enumerated mutation"),
            }
            let directory = tempfile::tempdir().expect("QG-6 artifact directory");
            let path = directory
                .path()
                .join(format!("qg6-binding-{mutation}.json"));
            fs::write(
                &path,
                artifact
                    .sealed_json()
                    .expect("outer-reseal binding mutation"),
            )
            .expect("persist binding mutation");
            assert!(
                matches!(
                    PerfEvidenceArtifact::load_verified(&path),
                    Err(EvidenceArtifactError::InconsistentArtifact { .. })
                ),
                "compact binding field {mutation} escaped reload verification"
            );
        }
    }

    #[test]
    fn qg6_sealed_reload_rejects_one_groups_extra_balanced_pair() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body else {
            unreachable!("QG-6 must be paired");
        };
        let source_block = paired.effect_samples[0].block_id;
        let mut extra = paired
            .effect_samples
            .iter()
            .filter(|sample| sample.block_id == source_block)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(extra.len(), 2);
        let block_id = paired
            .effect_samples
            .iter()
            .map(|sample| sample.block_id)
            .max()
            .expect("effect block")
            + 1;
        let sample_id = paired
            .effect_samples
            .iter()
            .chain(&paired.null_samples)
            .map(|sample| sample.sample_id)
            .max()
            .expect("sample ID")
            + 1;
        for (offset, sample) in extra.iter_mut().enumerate() {
            sample.block_id = block_id;
            sample.sample_id = sample_id + u64::try_from(offset).expect("sample offset");
            sample.started_ns += 1_000_000_000;
            sample.ended_ns += 1_000_000_000;
        }
        paired.effect_samples.extend(extra);
        let recomputed = estimate_paired_experiment(
            &paired.effect_samples,
            &paired.null_samples,
            &paired.config,
        )
        .expect("recompute structurally valid extra-pair stream");
        **paired = recomputed;

        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-extra-balanced-pair.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("outer-reseal extra balanced pair"),
        )
        .expect("persist extra balanced pair");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InconsistentArtifact { ref reason })
                if reason.contains("common positive multiplicity")
        ));
    }

    #[test]
    fn qg6_sealed_reload_rejects_a_missing_named_role_receipt() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let mut value = serde_json::to_value(&artifact).expect("QG-6 artifact value");
        value["cells"][0]["spec"]["qg6_semantic_contract"]["groups"][0]["roles"]
            .as_object_mut()
            .expect("named role table")
            .remove("effect_treatment")
            .expect("effect-treatment role");
        value["artifact_sha256"] = serde_json::Value::String(String::new());
        let unsealed = serde_json::to_string_pretty(&value).expect("unsealed role-missing JSON");
        value["artifact_sha256"] =
            serde_json::Value::String(lower_hex(&Sha256::digest(unsealed.as_bytes())));
        let sealed = serde_json::to_string_pretty(&value).expect("sealed role-missing JSON");
        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-missing-role.json");
        fs::write(&path, sealed).expect("persist role-missing artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::Malformed { .. })
        ));
    }

    #[test]
    fn qg6_verified_load_recomputes_hierarchical_null_from_raw_groups() {
        let mut artifact = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let EvidenceCellBody::Paired {
            hierarchical_null: Some(null),
            ..
        } = &mut artifact.cells[0].body
        else {
            unreachable!("QG-6 must carry hierarchical A/A evidence");
        };
        null.median_of_group_medians_log = 1.25_f64.ln();
        null.treatment_over_control = 1.25;
        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-sealed-null-mismatch.json");
        fs::write(
            &path,
            artifact.sealed_json().expect("seal mismatched artifact"),
        )
        .expect("persist sealed mismatch");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn qg6_partial_query_group_removal_is_rejected_as_inconsistent() {
        let original = qg6_artifact();
        let original_cell = &original.cells[0];
        let EvidenceCellBody::Paired { paired, .. } = &original_cell.body else {
            unreachable!("QG-6 must carry paired evidence");
        };
        let retain_first_half = |sample: &&PerfRawSample| sample.group_id.is_some_and(|id| id < 2);
        let effect = paired
            .effect_samples
            .iter()
            .filter(retain_first_half)
            .cloned()
            .collect::<Vec<_>>();
        let null = paired
            .null_samples
            .iter()
            .filter(retain_first_half)
            .cloned()
            .collect::<Vec<_>>();
        let partial_pair = estimate_paired_experiment(&effect, &null, &paired.config)
            .expect("two retained groups still satisfy the generic hierarchical minimum");
        assert!(matches!(
            EvidenceCell::evaluate(original_cell.spec.clone(), partial_pair, &original.policy),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn qg6_hierarchical_null_is_retained_but_tail_protocol_forces_no_claim() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.as_ref().expect("QG-6 identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let effect_pairs = QG6_QUERY_GROUP_IDS
            .into_iter()
            .flat_map(|group_id| [(group_id, 100.0, 98.0); 100])
            .collect::<Vec<_>>();
        let mut null_pairs = Vec::new();
        for group_id in QG6_QUERY_GROUP_IDS {
            null_pairs.extend((0..25).map(|_| (group_id, 100.0, 80.0)));
            null_pairs.extend((0..49).map(|_| (group_id, 100.0, 100.0)));
            null_pairs.extend((0..26).map(|_| (group_id, 100.0, 125.0)));
        }
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        qg6_test_fixture::attach_stream(&mut effect, true, identity, contract);
        qg6_test_fixture::attach_stream(&mut null, false, identity, contract);

        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        assert_eq!(paired.status, PairedEvidenceStatus::InvalidNull);
        assert_eq!(paired.claim_state, PairedClaimState::NoDecision);
        assert!(
            paired
                .reasons
                .iter()
                .all(|reason| qg6_flat_inference_only(&reason.code)),
            "fixture accidentally carries a structural/design failure: {:?}",
            paired.reasons
        );

        let cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        let EvidenceCellBody::Paired {
            hierarchical_null: Some(null),
            ..
        } = &cell.body
        else {
            unreachable!("QG-6 must carry hierarchical null evidence");
        };
        assert!((null.treatment_over_control - 1.0).abs() < 1.0e-12);
        assert!(null.ci95_low_ratio >= 0.95 && null.ci95_high_ratio <= 1.05);
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(!cell.claim_eligible());
        assert!(cell.reasons.iter().any(|reason| {
            reason.code == "qg6.tail_protocol_not_implemented"
                && reason.severity == EvidenceSeverity::NoClaim
        }));

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("QG-6 artifact");
        bind_test_identity(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-hierarchy-primary",
        );
        assert!(!artifact.ratchet_admissible());
        let directory = tempfile::tempdir().expect("QG-6 hierarchy-native artifact directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("seal hierarchy-native QG-6 artifact");
        let verified = PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("hierarchy-native QG-6 artifact must recompute");
        assert!(!verified.ratchet_admissible());
    }

    #[test]
    fn qg6_hierarchy_remains_diagnostic_despite_flat_marginal_direction_conflict() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.as_ref().expect("QG-6 identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect_pairs = Vec::new();
        let mut null_pairs = Vec::new();
        for group_id in QG6_QUERY_GROUP_IDS {
            effect_pairs.extend((0..31).map(|_| (group_id, 1_000.0, 950.0)));
            effect_pairs.extend((0..31).map(|_| (group_id, 10_000.0, 9_500.0)));
            effect_pairs.extend((0..38).map(|_| (group_id, 1.0, 100_000.0)));
            null_pairs.extend([(group_id, 100.0, 100.0); 100]);
        }
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        qg6_test_fixture::attach_stream(&mut effect, true, identity, contract);
        qg6_test_fixture::attach_stream(&mut null, false, identity, contract);

        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        assert_eq!(paired.status, PairedEvidenceStatus::ContradictorySummaries);
        assert_eq!(paired.claim_state, PairedClaimState::NoDecision);
        assert!(
            paired
                .reasons
                .iter()
                .all(|reason| qg6_flat_inference_only(&reason.code))
        );

        let cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        let EvidenceCellBody::Paired {
            hierarchical: Some(effect),
            reconciliation,
            ..
        } = &cell.body
        else {
            unreachable!("QG-6 must carry hierarchical effect evidence");
        };
        assert!(effect.ci95_low_ratio >= 0.90 && effect.ci95_high_ratio <= 1.10);
        assert!(
            !reconciliation.direction_agrees,
            "fixture must preserve the flat marginal conflict as a diagnostic"
        );
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(!cell.claim_eligible());
        assert!(!cell.reasons.iter().any(|reason| {
            reason.code == "evidence.absolute_relative_direction_conflict"
                || reason.code == "evidence.paired_invalid"
        }));

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("QG-6 artifact");
        bind_test_identity(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-conflict-primary",
        );
        assert!(!artifact.ratchet_admissible());
    }

    #[test]
    fn qg6_paired_stream_structural_invalidity_still_blocks_admission() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.as_ref().expect("QG-6 identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let effect_pairs = QG6_QUERY_GROUP_IDS
            .into_iter()
            .flat_map(|group_id| [(group_id, 100.0, 98.0); 3])
            .collect::<Vec<_>>();
        let null_pairs = QG6_QUERY_GROUP_IDS
            .into_iter()
            .flat_map(|group_id| [(group_id, 100.0, 100.0); 3])
            .collect::<Vec<_>>();
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, Some(true));
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, Some(true));
        qg6_test_fixture::attach_stream(&mut effect, true, identity, contract);
        qg6_test_fixture::attach_stream(&mut null, false, identity, contract);

        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        assert!(
            paired
                .reasons
                .iter()
                .any(|reason| !qg6_flat_inference_only(&reason.code))
        );
        let cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(!cell.claim_eligible());
        assert!(cell.reasons.iter().any(|reason| {
            reason.code == "evidence.qg6_paired_design_invalid"
                && (reason.message.contains("paired.null_order_imbalance")
                    || reason.message.contains("paired.null_order_unobserved")
                    || reason.message.contains("paired.effect_order_imbalance"))
        }));
    }

    #[test]
    fn cold_open_requires_verified_cache_proof() {
        let mut unproven = cell_spec(PerfGate::Qg9, EvidenceRole::Required);
        unproven.metric = "open_latency_ms".to_owned();
        unproven.cold_cache = Some(ColdCacheEvidence {
            procedure: "same-process reopen; OS page cache not dropped".to_owned(),
            verified: false,
        });
        let cell =
            EvidenceCell::evaluate(unproven, valid_experiment(1.10), &policy()).expect("cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(
            cell.reasons
                .iter()
                .any(|reason| reason.code == "evidence.cold_cache_unproven")
        );

        let mut proven = cell_spec(PerfGate::Qg9, EvidenceRole::Required);
        proven.metric = "open_latency_ms".to_owned();
        proven.cold_cache = Some(ColdCacheEvidence {
            procedure: "echo 3 > drop_caches before reopen".to_owned(),
            verified: true,
        });
        let cell = EvidenceCell::evaluate(proven, valid_experiment(1.10), &policy()).expect("cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::MeasuredProvisional);
    }

    #[test]
    fn facts_cells_are_diagnostic_and_never_gate_alone() {
        let mut spec = cell_spec(PerfGate::Qg10, EvidenceRole::Diagnostic);
        spec.metric = "tantivy_nodes".to_owned();
        spec.unit = "nodes".to_owned();
        let cell = EvidenceCell::facts(spec, vec![76.0; 10], &policy()).expect("facts cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::MeasuredProvisional);
        assert!(!cell.claim_eligible());

        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg10,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("artifact");
        assert_eq!(artifact.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(
            artifact
                .reasons
                .iter()
                .any(|reason| reason.code == "evidence.gate_without_required_cells")
        );
        assert!(!artifact.ratchet_admissible());
    }

    #[test]
    fn qg6_incomplete_gate_selection_is_durable_but_forced_to_no_claim() {
        let mut artifact = provisional_artifact();
        assert!(artifact.ratchet_admissible());

        artifact.force_no_claim(
            "evidence.incomplete_gate_selection",
            "fixture-filtered pre-admission run",
        );

        assert_eq!(artifact.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(!artifact.ratchet_admissible());
        assert!(matches!(
            artifact.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(
            artifact
                .reasons
                .iter()
                .any(|reason| reason.code == "evidence.incomplete_gate_selection")
        );
        assert!(matches!(
            artifact.apply_gate_decision(EvidenceDecisionStatus::Allow),
            Err(EvidenceArtifactError::NotClaimEligible)
        ));

        let directory = tempfile::tempdir().expect("partial evidence directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("persist partial no-claim evidence");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("partial no-claim evidence must remain structurally verifiable");
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(!reloaded.ratchet_admissible());
        assert_eq!(
            reloaded
                .admission_no_claim
                .as_ref()
                .map(|reason| reason.code.as_str()),
            Some("evidence.incomplete_gate_selection")
        );

        let contents = fs::read_to_string(&paths.json).expect("read partial evidence");
        let tampered = contents.replace(
            "fixture-filtered pre-admission run",
            "full-gate admission run",
        );
        assert_ne!(contents, tampered, "selection-scope tamper target");
        fs::write(&paths.json, tampered).expect("tamper selection scope");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&paths.json),
            Err(EvidenceArtifactError::HashMismatch)
        ));
    }

    #[test]
    fn artifact_roundtrip_write_load_verify_and_table_derives_from_json() {
        let artifact = provisional_artifact();
        assert!(artifact.ratchet_admissible());
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        assert!(paths.json.exists());
        assert!(paths.table.exists());
        let leftovers = fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "atomic write must not leave temp files");

        let written_bytes = fs::read(&paths.json).expect("read exact json bytes");
        let written = std::str::from_utf8(&written_bytes).expect("written evidence is UTF-8");
        let reloaded_from_bytes =
            PerfEvidenceArtifact::from_verified_slice(&written_bytes).expect("verified byte load");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("verified load");
        assert_eq!(reloaded_from_bytes, reloaded);
        assert_eq!(reloaded.cells, artifact.cells);
        assert_eq!(reloaded.gate_status, artifact.gate_status);
        assert!(!reloaded.artifact_sha256.is_empty());

        let table_on_disk = fs::read_to_string(&paths.table).expect("read table");
        assert_eq!(
            table_on_disk,
            human_table_from_json(written).expect("derive")
        );
        assert_eq!(table_on_disk, reloaded.human_table());
    }

    #[test]
    fn stable_seeded_replay_produces_identical_artifacts() {
        let first = provisional_artifact();
        let second = provisional_artifact();
        let dir = tempfile::tempdir().expect("tempdir");
        let first_paths = first.write_atomic(&dir.path().join("a")).expect("write a");
        let second_paths = second.write_atomic(&dir.path().join("b")).expect("write b");
        let first_bytes = fs::read(&first_paths.json).expect("read a");
        let second_bytes = fs::read(&second_paths.json).expect("read b");
        assert_eq!(first_bytes, second_bytes);
    }

    #[test]
    fn truncated_artifact_is_rejected() {
        let artifact = provisional_artifact();
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        let full = fs::read_to_string(&paths.json).expect("read");
        fs::write(&paths.json, &full[..full.len() / 2]).expect("truncate");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&paths.json),
            Err(EvidenceArtifactError::Malformed { .. })
        ));
    }

    #[test]
    fn tampered_artifact_fails_the_hash_seal() {
        let artifact = provisional_artifact();
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        let contents = fs::read_to_string(&paths.json).expect("read");
        let tampered = contents.replace(
            "\"run_window\": \"window-1\"",
            "\"run_window\": \"window-2\"",
        );
        assert_ne!(contents, tampered, "tamper target must exist");
        fs::write(&paths.json, tampered).expect("tamper");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&paths.json),
            Err(EvidenceArtifactError::HashMismatch)
        ));
    }

    #[test]
    fn verified_load_reapplies_cell_set_invariants_after_resealing() {
        let directory = tempfile::tempdir().expect("cell-set artifact directory");

        let mut wrong_gate = provisional_artifact();
        unbind_test_artifact(&mut wrong_gate);
        wrong_gate.gate = PerfGate::Qg2;
        let wrong_gate_path = directory.path().join("wrong-gate.json");
        fs::write(
            &wrong_gate_path,
            wrong_gate
                .sealed_json()
                .expect("reseal wrong-gate artifact"),
        )
        .expect("persist wrong-gate artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&wrong_gate_path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));

        let mut empty = provisional_artifact();
        unbind_test_artifact(&mut empty);
        empty.cells.clear();
        let empty_path = directory.path().join("empty.json");
        fs::write(
            &empty_path,
            empty.sealed_json().expect("reseal empty artifact"),
        )
        .expect("persist empty artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&empty_path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));

        let mut duplicate = provisional_artifact();
        unbind_test_artifact(&mut duplicate);
        duplicate.cells.push(duplicate.cells[0].clone());
        let duplicate_path = directory.path().join("duplicate.json");
        fs::write(
            &duplicate_path,
            duplicate.sealed_json().expect("reseal duplicate artifact"),
        )
        .expect("persist duplicate artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&duplicate_path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));

        let mut noncanonical = provisional_artifact();
        unbind_test_artifact(&mut noncanonical);
        noncanonical.cells[0].cell_id = "forged/cell/id".to_owned();
        let noncanonical_path = directory.path().join("noncanonical-id.json");
        fs::write(
            &noncanonical_path,
            noncanonical
                .sealed_json()
                .expect("reseal noncanonical-cell artifact"),
        )
        .expect("persist noncanonical-cell artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&noncanonical_path),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn schema_rejects_missing_required_fields() {
        let artifact = provisional_artifact();
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        let contents = fs::read_to_string(&paths.json).expect("read");

        for field in [
            "provenance",
            "machine_class",
            "cells",
            "policy",
            "gate_status",
        ] {
            let mut value: serde_json::Value = serde_json::from_str(&contents).expect("parse");
            value
                .as_object_mut()
                .expect("object")
                .remove(field)
                .expect("field present");
            let mutilated = serde_json::to_string_pretty(&value).expect("serialize");
            fs::write(&paths.json, &mutilated).expect("write mutilated");
            assert!(
                PerfEvidenceArtifact::load_verified(&paths.json).is_err(),
                "artifact missing {field} must be rejected"
            );
        }

        let mut value: serde_json::Value = serde_json::from_str(&contents).expect("parse");
        let paired = value["cells"][0]["body"]
            .as_object_mut()
            .expect("cell body");
        paired.remove("paired").expect("paired present");
        let mutilated = serde_json::to_string_pretty(&value).expect("serialize");
        fs::write(&paths.json, &mutilated).expect("write mutilated");
        assert!(PerfEvidenceArtifact::load_verified(&paths.json).is_err());
    }

    #[test]
    fn old_schema_never_masquerades_and_legacy_load_is_explicit() {
        let legacy = PerfGateArtifact {
            schema_version: LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3.to_owned(),
            gate: PerfGate::Qg1,
            bench_elf_sha256: "a".repeat(64),
            machine_fingerprint: "legacy-machine".to_owned(),
            execution: None,
            git_rev: "deadbeef".to_owned(),
            run_window: "legacy-window".to_owned(),
            run_id: "legacy-run".to_owned(),
            corpus_manifest_hash: "b".repeat(64),
            manifest_sha256: "e".repeat(64),
            cells: vec![PerfCellResult {
                fixture: "bulk/synthetic/1".to_owned(),
                metric: "docs_per_second".to_owned(),
                engine: "quill".to_owned(),
                unit: "docs/s".to_owned(),
                distribution: DistributionSummary::from_samples(&[1.0; 10]).expect("summary"),
            }],
            laws_attested: false,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let legacy_path = dir.path().join("legacy.json");
        fs::write(
            &legacy_path,
            serde_json::to_string_pretty(&legacy).expect("serialize legacy"),
        )
        .expect("write legacy");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&legacy_path),
            Err(EvidenceArtifactError::SchemaMismatch { found })
                if found == LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3
        ));
        let loaded = load_legacy_gate_artifact_v3(&legacy_path).expect("legacy read-only load");
        assert_eq!(loaded, legacy);

        let current = provisional_artifact();
        let current_paths = current.write_atomic(dir.path()).expect("write current");
        assert!(matches!(
            load_legacy_gate_artifact_v3(&current_paths.json),
            Err(EvidenceArtifactError::SchemaMismatch { found }) if found == PERF_EVIDENCE_SCHEMA_VERSION
        ));
    }

    #[test]
    fn v1_evidence_is_legacy_nonpromotable_and_has_no_upgrade_path() {
        let mut legacy_v1 = provisional_artifact();
        legacy_v1.schema_version = "quill-perf-evidence-v1".to_owned();
        let directory = tempfile::tempdir().expect("legacy v1 directory");
        let path = directory.path().join("legacy-v1.evidence.json");
        fs::write(
            &path,
            legacy_v1
                .sealed_json()
                .expect("seal exact legacy-v1-shaped evidence"),
        )
        .expect("persist legacy v1 evidence");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::SchemaMismatch { found })
                if found == "quill-perf-evidence-v1"
        ));
        assert!(matches!(
            load_legacy_gate_artifact_v3(&path),
            Err(EvidenceArtifactError::SchemaMismatch { found })
                if found == "quill-perf-evidence-v1"
        ));
    }

    #[test]
    fn normative_manifest_is_bound_to_the_current_evidence_schema() {
        let manifest: toml::Value = toml::from_str(include_str!(
            "../../../docs/contracts/quill-perf-gates.toml"
        ))
        .expect("parse normative performance manifest");
        assert_eq!(
            manifest
                .get("evidence")
                .and_then(|evidence| evidence.get("schema"))
                .and_then(toml::Value::as_str),
            Some(PERF_EVIDENCE_SCHEMA_VERSION)
        );
    }

    #[test]
    fn two_clean_process_replays_reproduce_within_tolerance() {
        let first = valid_experiment(1.10);
        let second_effect = {
            let scope = scope();
            let provenance = sample_provenance("run-b");
            let order = seeded_balanced_pair_order(12, 0x00c0_ffee).expect("order");
            let mut samples = Vec::new();
            for (index, first_arm) in order.into_iter().enumerate() {
                let index = u64::try_from(index).expect("index");
                push_gauge_block(
                    &mut samples,
                    &scope,
                    &provenance,
                    index,
                    index * 2,
                    100.0,
                    100.0 * 1.104,
                    first_arm == PerfSampleArm::Control,
                    None,
                );
            }
            samples
        };
        let second_null = {
            let scope = scope();
            let provenance = sample_provenance("run-b");
            let order = seeded_balanced_pair_order(12, 0x00c0_ffee).expect("order");
            let mut samples = Vec::new();
            for (index, first_arm) in order.into_iter().enumerate() {
                let epsilon: f64 = if index % 2 == 0 { 0.002 } else { -0.002 };
                let index = u64::try_from(index).expect("index");
                push_gauge_block(
                    &mut samples,
                    &scope,
                    &provenance,
                    index,
                    10_000 + index * 2,
                    100.0,
                    100.0 * (1.0 + epsilon),
                    first_arm == PerfSampleArm::Control,
                    None,
                );
            }
            samples
        };
        let second =
            estimate_paired_experiment(&second_effect, &second_null, &config()).expect("second");
        assert!(
            first
                .reproduces_within(&second)
                .expect("comparable replays")
        );
    }

    #[test]
    fn reasons_are_bounded_and_artifacts_carry_no_content() {
        let oversized = EvidenceReason::new(
            "evidence.test",
            CANARY.repeat(40),
            EvidenceSeverity::NoClaim,
        );
        assert!(oversized.message.len() <= EVIDENCE_MAX_REASON_MESSAGE_BYTES);

        let artifact = provisional_artifact();
        let json = artifact.canonical_json().expect("json");
        assert!(!json.contains(CANARY));
        assert!(!json.to_lowercase().contains("term00001"));
    }

    #[test]
    fn unsupported_rss_probe_must_not_fabricate_zero() {
        let mut provenance = evidence_provenance();
        provenance.peak_rss = PeakRssEvidence {
            method: "unsupported".to_owned(),
            bytes: Some(0),
        };
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg1,
                policy(),
                provenance,
                vec![provisional_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));

        let mut zeroed = evidence_provenance();
        zeroed.peak_rss = PeakRssEvidence {
            method: "linux_vmhwm".to_owned(),
            bytes: Some(0),
        };
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg1,
                policy(),
                zeroed,
                vec![provisional_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));
    }

    #[test]
    fn gate_decisions_only_apply_to_eligible_evidence() {
        let mut artifact = provisional_artifact();
        artifact
            .apply_gate_decision(EvidenceDecisionStatus::Allow)
            .expect("eligible promotion");
        assert_eq!(artifact.gate_decision, Some(EvidenceDecisionStatus::Allow));

        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("reload");
        assert_eq!(reloaded.gate_decision, Some(EvidenceDecisionStatus::Allow));

        let mut artifact = provisional_artifact();
        assert!(matches!(
            artifact.apply_gate_decision(EvidenceDecisionStatus::MeasuredProvisional),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }
}
