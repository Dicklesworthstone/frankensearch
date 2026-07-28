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
    DistributionSummary, PERF_ARTIFACT_SCHEMA_VERSION, PairedClaimState, PairedEstimatorConfig,
    PairedEstimatorError, PairedEvidenceStatus, PairedExperimentResult, PerfGate, PerfGateArtifact,
    PerfInputIdentity, PerfRawSample, median_sorted, percentile, splitmix64,
    validate_paired_blocks,
};

/// Version of the evidence artifact emitted by this module.
///
/// Old artifacts never masquerade as current: loading any other version
/// through [`PerfEvidenceArtifact::load_verified`] is a typed
/// [`EvidenceArtifactError::SchemaMismatch`], and legacy v3 gate artifacts are
/// only readable through the explicit, read-only
/// [`load_legacy_gate_artifact_v3`].
pub const PERF_EVIDENCE_SCHEMA_VERSION: &str = "quill-perf-evidence-v1";
/// Version of the hierarchical latency estimate carried by latency cells.
pub const HIERARCHICAL_LATENCY_SCHEMA_VERSION: &str = "quill-hierarchical-latency-v1";
/// Upper bound on retained reasons per artifact or cell.
pub const EVIDENCE_MAX_REASONS: usize = 64;
/// Upper bound on one bounded reason message, in bytes.
pub const EVIDENCE_MAX_REASON_MESSAGE_BYTES: usize = 240;

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
            || self.git_revision.trim().is_empty()
            || self.rustc_version.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.build_profile.trim().is_empty()
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "build identity requires an executable SHA-256, git revision, rustc \
                         version, target triple, and profile"
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
    pub fn capture() -> Self {
        Self {
            fingerprint: crate::perf::machine_fingerprint(),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
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
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "machine identity requires fingerprint, os, arch, and CPUs".to_owned(),
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
    /// Cache-state proof, required for cold-open cells.
    pub cold_cache: Option<ColdCacheEvidence>,
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

        match (spec.gate, spec.input_identity.as_ref()) {
            (PerfGate::Qg6, Some(identity)) => {
                identity.validate()?;
                if paired.provenance.input_identity.as_ref() != Some(identity) {
                    return Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: "QG-6 cell input identity does not match its raw-sample provenance"
                            .to_owned(),
                    });
                }
            }
            (PerfGate::Qg6, None) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason:
                        "QG-6 evidence requires separate exact prepared-corpus, ordered-query, \
                             and configuration identity"
                            .to_owned(),
                });
            }
            (_, Some(_)) => {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "exact prepared-input identity is only valid for QG-6".to_owned(),
                });
            }
            (_, None) => {}
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
                hierarchical,
                hierarchical_null,
                reconciliation,
            },
            status,
            reasons,
        })
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
        if spec.input_identity.is_some() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "dependency facts cannot carry exact prepared-input identity".to_owned(),
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
            EvidenceCellBody::Paired { paired, .. } => {
                self.status == EvidenceDecisionStatus::MeasuredProvisional
                    && paired.claim_state == PairedClaimState::EligibleForDecision
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
            EvidenceCellBody::Paired { paired, .. } => {
                paired.verify_recomputed()?;
                let rebuilt = Self::evaluate(self.spec.clone(), paired.as_ref().clone(), policy)?;
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
pub struct PerfEvidenceArtifact {
    /// Always [`PERF_EVIDENCE_SCHEMA_VERSION`] for current artifacts.
    pub schema_version: String,
    /// Gate this artifact certifies.
    pub gate: PerfGate,
    /// Predeclared evidence-layer thresholds used for every cell.
    pub policy: EvidencePolicy,
    /// Complete run provenance.
    pub provenance: EvidenceProvenance,
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
                .cells
                .iter()
                .filter(|cell| cell.spec.role == EvidenceRole::Required)
                .all(EvidenceCell::claim_eligible)
    }

    /// Fail closed when an invocation selected only part of a normative gate.
    ///
    /// The measured cells and raw samples remain durable, but the artifact
    /// cannot establish a ratchet or accept a downstream gate decision.
    pub fn force_no_claim(&mut self, code: &str, message: impl Into<String>) {
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
        let contents = fs::read_to_string(path)?;
        let probe: serde_json::Value =
            serde_json::from_str(&contents).map_err(|error| EvidenceArtifactError::Malformed {
                reason: format!("artifact is not valid JSON: {error}"),
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
        let artifact: Self = serde_json::from_str(&contents)?;
        artifact.verify_integrity()?;
        Ok(artifact)
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
    if found != PERF_ARTIFACT_SCHEMA_VERSION {
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
    #[error("evidence artifact schema is {found}; current is quill-perf-evidence-v1")]
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
mod tests {
    use super::*;
    use crate::perf::{
        PerfCellResult, PerfMetricSemantics, PerfOperationScope, PerfSampleArm, PerfSampleOrder,
        PerfSamplePhase, PerfSampleProvenance, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS,
        estimate_paired_experiment, seeded_balanced_pair_order,
    };

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

    fn attach_input_identity(samples: &mut [PerfRawSample], identity: Option<&PerfInputIdentity>) {
        for sample in samples {
            sample.provenance.input_identity = identity.cloned();
        }
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

    fn policy() -> EvidencePolicy {
        EvidencePolicy::predeclared()
    }

    fn build_identity() -> BuildIdentity {
        BuildIdentity {
            executable_sha256: "a".repeat(64),
            git_revision: "deadbeef".to_owned(),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: Some("c".repeat(64)),
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

    fn cell_spec(gate: PerfGate, role: EvidenceRole) -> EvidenceCellSpec {
        EvidenceCellSpec {
            gate,
            fixture: "bulk/synthetic/1".to_owned(),
            metric: "latency_ms".to_owned(),
            unit: "ms".to_owned(),
            role,
            input_identity: (gate == PerfGate::Qg6).then(|| PerfInputIdentity {
                prepared_corpus_sha256: "a".repeat(64),
                query_manifest_sha256: "c".repeat(64),
                config_contract_sha256: "f".repeat(64),
                query_group_count: QG6_QUERY_GROUPS,
                query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
            }),
            cold_cache: None,
        }
    }

    fn provisional_cell() -> EvidenceCell {
        EvidenceCell::evaluate(
            cell_spec(PerfGate::Qg1, EvidenceRole::Required),
            valid_experiment(1.10),
            &policy(),
        )
        .expect("provisional cell")
    }

    fn provisional_artifact() -> PerfEvidenceArtifact {
        PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            policy(),
            evidence_provenance(),
            vec![provisional_cell()],
        )
        .expect("provisional artifact")
    }

    fn qg6_artifact() -> PerfEvidenceArtifact {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let input_identity = spec.input_identity.clone();
        let mut effect = hierarchical_stream_with_ratio(1.02, 0);
        let mut null = hierarchical_stream_with_ratio(1.0, 10_000);
        for sample in effect.iter_mut().chain(&mut null) {
            sample.provenance.input_identity = input_identity.clone();
        }
        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        let cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        let mut provenance = evidence_provenance();
        provenance.corpus.query_set_sha256 = Some("d".repeat(64));
        PerfEvidenceArtifact::assemble(PerfGate::Qg6, policy(), provenance, vec![cell])
            .expect("QG-6 artifact")
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
            EvidenceCellBody::Paired { reconciliation, .. } => {
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
    fn latency_gate_without_groups_is_no_decision() {
        let mut spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        spec.metric = "latency_ms".to_owned();
        let input_identity = spec.input_identity.clone();
        let mut effect = gauge_stream(&effect_pairs(12, 1.10), 0, 0, None);
        let mut null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        for sample in effect.iter_mut().chain(&mut null) {
            sample.provenance.input_identity = input_identity.clone();
        }
        let experiment =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 estimate");
        let cell = EvidenceCell::evaluate(spec, experiment, &policy()).expect("cell");
        assert_eq!(cell.estimand, EvidenceEstimand::HierarchicalLatency);
        assert_eq!(cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(
            cell.reasons
                .iter()
                .any(|reason| reason.code == "evidence.hierarchical_effect_unavailable")
        );
        assert!(
            cell.reasons
                .iter()
                .any(|reason| reason.code == "evidence.hierarchical_null_unavailable")
        );
    }

    #[test]
    fn qg6_rejects_each_prepared_input_identity_mismatch_independently() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let expected_identity = spec.input_identity.clone();
        let mut effect = gauge_stream(&effect_pairs(12, 1.02), 0, 0, None);
        let mut null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        for sample in effect.iter_mut().chain(&mut null) {
            sample.provenance.input_identity = expected_identity.clone();
        }
        let experiment =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 estimate");

        for field in [
            "prepared_corpus_sha256",
            "query_manifest_sha256",
            "config_contract_sha256",
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
    fn qg6_verified_load_rejects_sealed_cell_identity_divergence() {
        let mut artifact = qg6_artifact();
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
    fn qg6_verified_load_recomputes_hierarchical_null_from_raw_groups() {
        let mut artifact = qg6_artifact();
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
    fn qg6_partial_query_group_removal_cannot_remain_claim_eligible() {
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
        let partial_cell =
            EvidenceCell::evaluate(original_cell.spec.clone(), partial_pair, &original.policy)
                .expect("partial-group QG-6 cell remains durable");
        assert_eq!(partial_cell.status, EvidenceDecisionStatus::NoDecision);
        assert!(!partial_cell.claim_eligible());
        assert!(partial_cell.reasons.iter().any(|reason| {
            reason.code == "evidence.qg6_query_groups_incomplete"
                && reason
                    .message
                    .contains("prepared ordered query-group identity")
        }));

        let partial = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            original.policy,
            original.provenance,
            vec![partial_cell],
        )
        .expect("partial-group artifact remains durable");
        let directory = tempfile::tempdir().expect("partial-group artifact directory");
        let paths = partial
            .write_atomic(directory.path())
            .expect("seal partial-group artifact");
        let verified = PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("partial-group artifact must recompute");
        assert!(!verified.ratchet_admissible());
    }

    #[test]
    fn qg6_hierarchical_null_can_admit_when_flat_null_inference_is_invalid() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.clone();
        let effect_pairs = (0_u64..4)
            .flat_map(|group_id| [(group_id, 100.0, 98.0); 3])
            .collect::<Vec<_>>();
        let mut null_pairs = Vec::new();
        for group_id in 0_u64..4 {
            null_pairs.extend((0..25).map(|_| (group_id, 100.0, 80.0)));
            null_pairs.extend((0..49).map(|_| (group_id, 100.0, 100.0)));
            null_pairs.extend((0..26).map(|_| (group_id, 100.0, 125.0)));
        }
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        attach_input_identity(&mut effect, identity.as_ref());
        attach_input_identity(&mut null, identity.as_ref());

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
        assert_eq!(cell.status, EvidenceDecisionStatus::MeasuredProvisional);
        assert!(cell.claim_eligible());

        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("QG-6 artifact");
        assert!(artifact.ratchet_admissible());
        let directory = tempfile::tempdir().expect("QG-6 hierarchy-native artifact directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("seal hierarchy-native QG-6 artifact");
        let verified = PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("hierarchy-native QG-6 artifact must recompute");
        assert!(verified.ratchet_admissible());
    }

    #[test]
    fn qg6_hierarchy_can_admit_despite_flat_marginal_direction_conflict() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.clone();
        let mut effect_pairs = Vec::new();
        let mut null_pairs = Vec::new();
        for group_id in 0_u64..4 {
            effect_pairs.extend((0..31).map(|_| (group_id, 1_000.0, 950.0)));
            effect_pairs.extend((0..31).map(|_| (group_id, 10_000.0, 9_500.0)));
            effect_pairs.extend((0..38).map(|_| (group_id, 1.0, 100_000.0)));
            null_pairs.extend([(group_id, 100.0, 100.0); 3]);
        }
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        attach_input_identity(&mut effect, identity.as_ref());
        attach_input_identity(&mut null, identity.as_ref());

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
        assert_eq!(cell.status, EvidenceDecisionStatus::MeasuredProvisional);
        assert!(cell.claim_eligible());
        assert!(!cell.reasons.iter().any(|reason| {
            reason.code == "evidence.absolute_relative_direction_conflict"
                || reason.code == "evidence.paired_invalid"
        }));

        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            policy(),
            evidence_provenance(),
            vec![cell],
        )
        .expect("QG-6 artifact");
        assert!(artifact.ratchet_admissible());
    }

    #[test]
    fn qg6_paired_stream_structural_invalidity_still_blocks_admission() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.clone();
        let effect_pairs = (0_u64..4)
            .flat_map(|group_id| [(group_id, 100.0, 98.0); 3])
            .collect::<Vec<_>>();
        let null_pairs = (0_u64..4)
            .flat_map(|group_id| [(group_id, 100.0, 100.0); 3])
            .collect::<Vec<_>>();
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, Some(true));
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, Some(true));
        attach_input_identity(&mut effect, identity.as_ref());
        attach_input_identity(&mut null, identity.as_ref());

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

        let written = fs::read_to_string(&paths.json).expect("read json");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("verified load");
        assert_eq!(reloaded.cells, artifact.cells);
        assert_eq!(reloaded.gate_status, artifact.gate_status);
        assert!(!reloaded.artifact_sha256.is_empty());

        let table_on_disk = fs::read_to_string(&paths.table).expect("read table");
        assert_eq!(
            table_on_disk,
            human_table_from_json(&written).expect("derive")
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

        for field in ["provenance", "cells", "policy", "gate_status"] {
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
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            bench_elf_sha256: "a".repeat(64),
            machine_fingerprint: "legacy-machine".to_owned(),
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
            Err(EvidenceArtifactError::SchemaMismatch { found }) if found == PERF_ARTIFACT_SCHEMA_VERSION
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
