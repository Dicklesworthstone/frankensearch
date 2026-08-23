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
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::perf::{
    DistributionSummary, LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3, PERF_NULL_MARGIN_MULTIPLIER,
    PairedClaimState, PairedEstimatorConfig, PairedEstimatorError, PairedEvidenceStatus,
    PairedExperimentResult, PerfApplicabilityPlan, PerfApplicabilityPlanBinding,
    PerfCellApplicability, PerfCellSpec, PerfExecutionProvenance, PerfGate, PerfGateArtifact,
    PerfInputIdentity, PerfMatrixSpec, PerfRawSample, PerfSampleArm, QG6_QUERY_GROUP_IDS,
    QG6_QUERY_GROUPS, Qg1ExpectedAuthority, Qg1TantivyIncumbentDecision, Qg1TantivyIncumbentError,
    Qg1TantivyIncumbentScreen, Qg1TantivySemanticContract, Qg1TantivyWriterMode, median_sorted,
    parse_cpu_list_ids, percentile, perf_metric_unit, perf_operation_scope,
    resolve_qg1_expected_authority_for_replay, splitmix64, validate_paired_blocks,
};
use crate::qg6_prepared::{
    Qg6ArmRole, Qg6Comparison, Qg6QueryIdentityReceipt, Qg6QuerySpec, Qg6SampleOrder,
    Qg6ScheduleAuthority, Qg6SemanticContract, qg6_result_sequence_sha256,
};
use crate::{MachineClassEvidenceBinding, MachineClassRegistry, VerifiedRunnerIdentity};

/// Version of the evidence artifact emitted by this module.
///
/// Old artifacts never masquerade as current: loading any other version
/// through [`PerfEvidenceArtifact::load_verified`] is a typed
/// [`EvidenceArtifactError::SchemaMismatch`], and legacy v3 gate artifacts are
/// only readable through the explicit, read-only
/// [`load_legacy_gate_artifact_v3`].
pub const PERF_EVIDENCE_SCHEMA_VERSION: &str = "quill-perf-evidence-v7";
/// Version of the hierarchical latency estimate carried by latency cells.
pub const HIERARCHICAL_LATENCY_SCHEMA_VERSION: &str = "quill-hierarchical-latency-v1";
/// Version of the joint six-arm QG-6 p50/p99 estimate.
pub const QG6_JOINT_TAIL_SCHEMA_VERSION: &str = "quill-qg6-joint-tail-v2";

/// Normative floor on joint-tail bootstrap replicates (bd-quill-e8-perf-doctrine-x4e4.9.3).
pub const QG6_JOINT_TAIL_MIN_BOOTSTRAP_REPLICATES: usize = 50_000;
/// Hard ceiling; exceeding it without boundary stability marks the estimate
/// `monte_carlo_stable = false` (fail-closed admissibility, not an abort).
pub const QG6_JOINT_TAIL_MAX_BOOTSTRAP_REPLICATES: usize = 400_000;
pub const QG6_PER_CELL_ALPHA: f64 = 0.0025;
/// p50 equivalence window: the equal-tailed (alpha/2 per side) percentile
/// interval must lie wholly inside this ratio window.
pub const QG6_P50_TOST_WINDOW_RATIO: (f64, f64) = (0.90, 1.10);
/// p99 noninferiority limit: the one-sided alpha-level upper bound must not exceed it.
pub const QG6_P99_UCB_LIMIT_RATIO: f64 = 1.0;
/// Batch count for batch-means Monte Carlo standard errors.
const QG6_MC_BATCH_COUNT: usize = 20;
/// Safety multiple: a boundary is only stable when its margin exceeds Z * SE.
const QG6_MC_SAFETY_Z: f64 = 3.0;
pub const QG6_NULL_EFFECT_MARGIN: f64 = 0.10;
/// Maximum serialized evidence artifact admitted by the public loader.
pub const PERF_EVIDENCE_MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
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
        let mut expected = Self::predeclared();
        expected.warmup_rounds = self.warmup_rounds;
        if !finite
            || self.min_hierarchical_groups < 2
            || self.min_group_pairs < 2
            || self.warmup_rounds == 0
            || self.max_raw_samples == 0
            || *self != expected
        {
            return Err(EvidenceArtifactError::InvalidPolicy {
                reason: "evidence policy must equal the predeclared thresholds with only a \
                         positive warmup-round count varying"
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
    /// Benchmark-reported `rustc --version` of the toolchain that built the
    /// binary. The receipt independently binds the controlled compiler path,
    /// digest, environment policy, source, and resulting ELF; it does not
    /// repeat this display string as a separate attested fact.
    pub rustc_version: String,
    /// Benchmark-reported compilation target triple. Receipt projection
    /// independently requires this triple to be compatible with the admitted
    /// hardware OS and architecture, but does not attest the exact string.
    pub target_triple: String,
    /// Benchmark-reported Cargo profile label, sealed into the exact evidence
    /// bytes and attributable to the receipt-bound source and ELF.
    pub build_profile: String,
    /// Benchmark-reported Cargo features active in the measuring binary,
    /// sealed into the exact evidence bytes and attributable to the
    /// receipt-bound source and ELF.
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
    /// Benchmark-reported deterministic machine label from
    /// [`crate::perf::machine_fingerprint`]. This is a diagnostic correlation
    /// label, not an independently receipt-attested hostname; registry class,
    /// hardware, topology, and execution facts remain authoritative.
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
    pub fn capture(
        execution_capacity: u64,
        max_exercised_cell_width: u64,
        configured_engine_thread_widths: impl IntoIterator<Item = usize>,
    ) -> Self {
        Self {
            fingerprint: crate::perf::machine_fingerprint(),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
            execution: PerfExecutionProvenance::capture(
                execution_capacity,
                max_exercised_cell_width,
                configured_engine_thread_widths,
            ),
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
            || self.logical_cpus != self.execution.process_available_threads
            || !self.execution.is_complete()
            || self.execution.producer_os.as_str() != self.os
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "machine identity requires fingerprint, os, arch, process-available \
                         CPUs, and matching serialized producer OS"
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

/// One jointly bootstrapped treatment/control latency contrast.
///
/// QG-6 keeps p50 and p99 in the same resample so a favorable point estimate
/// cannot borrow a confidence interval from another bootstrap population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6JointTailContrast {
    /// Point p50 treatment/control ratio.
    pub p50_ratio: f64,
    /// Lower 95% p50 ratio bound.
    pub p50_ci95_low_ratio: f64,
    /// Upper 95% p50 ratio bound.
    pub p50_ci95_high_ratio: f64,
    /// Point p99 treatment/control ratio.
    pub p99_ratio: f64,
    /// Lower 95% p99 ratio bound.
    pub p99_ci95_low_ratio: f64,
    /// Upper 95% p99 ratio bound.
    pub p99_ci95_high_ratio: f64,
    /// Lower equal-tailed alpha-level p50 TOST bound (ratio scale).
    pub p50_tost_low_ratio: f64,
    /// Upper equal-tailed alpha-level p50 TOST bound (ratio scale).
    pub p50_tost_high_ratio: f64,
    /// One-sided alpha-level p99 upper confidence bound (ratio scale).
    pub p99_ucb_ratio: f64,
    /// Monte Carlo standard error of the p50 TOST bounds (log scale).
    pub mc_se_p50_log: f64,
    /// Monte Carlo standard error of the p99 UCB (log scale).
    pub mc_se_p99_log: f64,
    /// p50 equivalence: the TOST interval lies wholly inside the frozen window.
    pub p50_equivalent: bool,
    /// p99 noninferiority: the one-sided upper bound does not exceed the limit.
    pub p99_noninferior: bool,
}

/// Joint query-first, unit-cluster QG-6 p50/p99 estimate.
///
/// Every bootstrap draw selects queries first and then complete query-round
/// units. One unit carries T/T, Q/Q, and T/Q together, so all six logical arms
/// share the exact same resample. Individual timing leaves are observations
/// inside a selected unit, never independent bootstrap units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6JointTailEstimate {
    /// Version of the joint estimator contract.
    pub schema_version: String,
    /// Canonical query groups resampled at stage one.
    pub query_count: usize,
    /// Complete three-comparison units per query.
    pub units_per_query: usize,
    /// Individually timed leaves per arm in every unit.
    pub leaves_per_arm_per_unit: usize,
    /// Bootstrap draws from the frozen paired-estimator configuration.
    pub bootstrap_resamples: usize,
    /// Replicates actually drawn after normative Monte Carlo escalation.
    pub replicates_used: usize,
    /// False when the replicate ceiling was reached before every effect-arm
    /// decision boundary stabilized; consumers must refuse PASS claims.
    pub monte_carlo_stable: bool,
    /// How many times the driver doubled replicates for boundary stability.
    pub escalations: u32,
    /// Tantivy/Tantivy null contrast.
    pub tantivy_null: Qg6JointTailContrast,
    /// Quill/Quill null contrast.
    pub quill_null: Qg6JointTailContrast,
    /// Quill/Tantivy effect contrast.
    pub effect: Qg6JointTailContrast,
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

#[derive(Debug)]
struct Qg6TailUnit {
    query_index: usize,
    role_leaves_ms: [Vec<f64>; 6],
}

const fn qg6_tail_role_index(role: Qg6ArmRole) -> usize {
    match role {
        Qg6ArmRole::TantivyNullLeft => 0,
        Qg6ArmRole::TantivyNullRight => 1,
        Qg6ArmRole::QuillNullLeft => 2,
        Qg6ArmRole::QuillNullRight => 3,
        Qg6ArmRole::EffectControl => 4,
        Qg6ArmRole::EffectTreatment => 5,
    }
}

fn qg6_tail_arm_quantiles(role_leaves_ms: &mut [Vec<f64>; 6]) -> [(f64, f64); 6] {
    std::array::from_fn(|role_index| {
        let leaves = &mut role_leaves_ms[role_index];
        leaves.sort_unstable_by(f64::total_cmp);
        (median_sorted(leaves), percentile(leaves, 0.99))
    })
}

fn qg6_tail_contrast_logs(quantiles: &[(f64, f64); 6]) -> [f64; 6] {
    let log_ratio = |left: usize, right: usize, quantile: usize| {
        let control = if quantile == 0 {
            quantiles[left].0
        } else {
            quantiles[left].1
        };
        let treatment = if quantile == 0 {
            quantiles[right].0
        } else {
            quantiles[right].1
        };
        (treatment / control).ln()
    };
    [
        log_ratio(0, 1, 0),
        log_ratio(0, 1, 1),
        log_ratio(2, 3, 0),
        log_ratio(2, 3, 1),
        log_ratio(4, 5, 0),
        log_ratio(4, 5, 1),
    ]
}

/// Frozen QG-6 empirical quantile: nearest-rank ceil(level*N), clamped to 1..=N
/// (bd-quill-e8-perf-doctrine-x4e4.9.3 freezes this definition for every
/// normative endpoint; the generic round-based helper is untouched elsewhere).
fn qg6_nearest_rank_quantile(sorted: &[f64], level: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&level));
    #[allow(clippy::cast_precision_loss)]
    let scaled = level * sorted.len() as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank = scaled.ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn qg6_joint_tail_contrast(
    point_p50_log: f64,
    point_p99_log: f64,
    p50_bootstrap: &mut [f64],
    p99_bootstrap: &mut [f64],
    mc_se_p50_log: f64,
    mc_se_p99_log: f64,
) -> Qg6JointTailContrast {
    p50_bootstrap.sort_unstable_by(f64::total_cmp);
    p99_bootstrap.sort_unstable_by(f64::total_cmp);
    let tost_low = qg6_nearest_rank_quantile(p50_bootstrap, QG6_PER_CELL_ALPHA).exp();
    let tost_high = qg6_nearest_rank_quantile(p50_bootstrap, 1.0 - QG6_PER_CELL_ALPHA).exp();
    let ucb = qg6_nearest_rank_quantile(p99_bootstrap, 1.0 - QG6_PER_CELL_ALPHA).exp();
    Qg6JointTailContrast {
        p50_ratio: point_p50_log.exp(),
        p50_ci95_low_ratio: percentile(p50_bootstrap, 0.025).exp(),
        p50_ci95_high_ratio: percentile(p50_bootstrap, 0.975).exp(),
        p99_ratio: point_p99_log.exp(),
        p99_ci95_low_ratio: percentile(p99_bootstrap, 0.025).exp(),
        p99_ci95_high_ratio: percentile(p99_bootstrap, 0.975).exp(),
        mc_se_p50_log,
        mc_se_p99_log,
        p50_tost_low_ratio: tost_low,
        p50_tost_high_ratio: tost_high,
        p99_ucb_ratio: ucb,
        p50_equivalent: tost_low >= QG6_P50_TOST_WINDOW_RATIO.0
            && tost_high <= QG6_P50_TOST_WINDOW_RATIO.1,
        p99_noninferior: ucb <= QG6_P99_UCB_LIMIT_RATIO,
    }
}

/// Monte Carlo standard error of one bootstrap endpoint via the order
/// statistic asymptotic form `sqrt(level*(1-level)/R) / f_hat`, where the
/// replicate density at the endpoint is estimated from a fixed-count local
/// window of the sorted replicates around the frozen nearest-rank quantile.
/// Batch means on extreme levels are hopelessly noisy (a within-batch
/// alpha=0.0025 quantile of a small chunk degenerates to its minimum); the
/// density form converges as replicates grow and escalates only genuinely
/// borderline decisions.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn qg6_mc_standard_error(draw_order: &[f64], level: f64) -> f64 {
    let replicate_count = draw_order.len();
    if replicate_count < QG6_MC_BATCH_COUNT {
        return f64::INFINITY;
    }
    let mut sorted = draw_order.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss)]
    let rank = ((level * replicate_count as f64).ceil() as usize).clamp(1, replicate_count);
    // Fixed-count symmetric window around the rank; at least 100 points per
    // side so far-tail levels still see enough neighbors for a stable slope.
    let half_window = 100_usize.max(replicate_count / 1_000);
    let lo = rank.saturating_sub(half_window);
    let hi = (rank + half_window).min(sorted.len());
    if hi - lo < 2 {
        return f64::INFINITY;
    }
    let span = sorted[hi - 1] - sorted[lo];
    if !span.is_finite() || span <= 0.0 {
        // Degenerate spike: every neighbor identical, density unbounded, so
        // the endpoint carries no sampling noise at this resolution.
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let density = (hi - lo - 1) as f64 / span;
    if !density.is_finite() || density <= 0.0 {
        return f64::INFINITY;
    }
    ((level * (1.0 - level)) / replicate_count as f64).sqrt() / density
}

/// Worst boundary-margin-to-SE ratio across the three effect-arm release
/// decision boundaries (p50 TOST low/high, p99 UCB). Only the Quill/Tantivy
/// effect boundaries gate escalation: a true A/A null sits ON its identity
/// boundary by construction, so null contrasts carry serialized standard
/// errors for audit but can never stabilize against a zero margin they are
/// expected to straddle.
///
/// Endpoint vectors: `tantivy_p50`, `tantivy_p99`, `quill_p50`, `quill_p99`,
/// `effect_p50`, `effect_p99`; the release decision consumes only the effect
/// pair (index 4/5).
fn qg6_worst_stability_ratio(bootstrap: &[Vec<f64>; 6]) -> f64 {
    let ratio = |vector: &[f64], boundary_log: f64, level: f64| {
        let se = qg6_mc_standard_error(vector, level);
        let mut sorted = vector.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let bound = qg6_nearest_rank_quantile(&sorted, level);
        ((bound - boundary_log).abs()) / se
    };
    let effect_p50 = &bootstrap[4];
    let effect_p99 = &bootstrap[5];
    let candidates = [
        ratio(
            effect_p50,
            QG6_P50_TOST_WINDOW_RATIO.0.ln(),
            QG6_PER_CELL_ALPHA,
        ),
        ratio(
            effect_p50,
            QG6_P50_TOST_WINDOW_RATIO.1.ln(),
            1.0 - QG6_PER_CELL_ALPHA,
        ),
        ratio(
            effect_p99,
            QG6_P99_UCB_LIMIT_RATIO.ln(),
            1.0 - QG6_PER_CELL_ALPHA,
        ),
    ];
    candidates.iter().copied().fold(f64::INFINITY, f64::min)
}
/// Admissibility poison only. Effect-arm TOST/noninferiority verdicts are
/// deliberately NOT emitted here: a decisively failed equivalence is valid,
/// claim-eligible evidence that the ratchet must consume into a Block (via
/// the serialized `p50_equivalent` / `p99_noninferior` fields), not a
/// `NoDecision` that would hide the measurement from the gate.
fn qg6_joint_tail_decision_reasons(estimate: &Qg6JointTailEstimate) -> Vec<EvidenceReason> {
    let mut reasons = Vec::new();
    for (engine, null) in [
        ("Tantivy/Tantivy", &estimate.tantivy_null),
        ("Quill/Quill", &estimate.quill_null),
    ] {
        for (quantile, low, high) in [
            ("p50", null.p50_ci95_low_ratio, null.p50_ci95_high_ratio),
            ("p99", null.p99_ci95_low_ratio, null.p99_ci95_high_ratio),
        ] {
            let null_floor = (low - 1.0).abs().max((high - 1.0).abs());
            if low > 1.0
                || high < 1.0
                || PERF_NULL_MARGIN_MULTIPLIER * null_floor > QG6_NULL_EFFECT_MARGIN
            {
                reasons.push(EvidenceReason::new(
                    "qg6.joint_tail_null_invalid",
                    format!(
                        "QG-6 {engine} {quantile} null CI [{low:.6}, {high:.6}] fails identity \
                         containment or the required {PERF_NULL_MARGIN_MULTIPLIER:.1}x margin"
                    ),
                    EvidenceSeverity::NoClaim,
                ));
            }
        }
    }
    if !estimate.monte_carlo_stable {
        reasons.push(EvidenceReason::new(
            "qg6.joint_tail_monte_carlo_unstable",
            format!(
                "QG-6 joint tail used {} replicates without stabilizing the effect-arm \
                 decision boundaries; no PASS claim is admissible",
                estimate.replicates_used
            ),
            EvidenceSeverity::NoClaim,
        ));
    }
    reasons
}

fn for_each_qg6_joint_tail_bootstrap_unit(
    seed: &mut u64,
    query_count: usize,
    units_per_query: usize,
    mut visit: impl FnMut(usize, usize),
) -> Result<(), EvidenceArtifactError> {
    let query_count_u64 =
        u64::try_from(query_count).map_err(|_| EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 joint tail query count does not fit u64".to_owned(),
        })?;
    let units_per_query_u64 = u64::try_from(units_per_query).map_err(|_| {
        EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 joint tail unit count does not fit u64".to_owned(),
        }
    })?;
    if query_count_u64 == 0 || units_per_query_u64 == 0 {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 joint tail bootstrap requires positive query and unit counts".to_owned(),
        });
    }
    for _ in 0..query_count {
        *seed = splitmix64(*seed);
        let query_index = usize::try_from(*seed % query_count_u64).map_err(|_| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 joint tail query draw does not fit usize".to_owned(),
            }
        })?;
        for _ in 0..units_per_query {
            *seed = splitmix64(*seed);
            let unit_index = usize::try_from(*seed % units_per_query_u64).map_err(|_| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail unit draw does not fit usize".to_owned(),
                }
            })?;
            visit(query_index, unit_index);
        }
    }
    Ok(())
}

/// Estimate the formal six-arm QG-6 p50 and p99 contrasts jointly.
///
/// The function first replays the complete external schedule authority and
/// semantic contract. It then resamples queries, followed by whole
/// query-round units. All six roles from a selected unit travel together;
/// timing leaves are never resampled as if they were independent queries.
///
/// # Errors
///
/// Returns a fail-closed evidence error for any authority, schedule, semantic,
/// cardinality, or finite-positive-latency defect.
pub fn estimate_qg6_joint_tail(
    paired: &PairedExperimentResult,
    protocol: &Qg6FormalProtocolEvidence,
    external_authority: &Qg6ScheduleAuthority,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
) -> Result<Qg6JointTailEstimate, EvidenceArtifactError> {
    validate_qg6_formal_protocol(paired, protocol, external_authority, identity, contract)?;
    let estimate = estimate_qg6_joint_tail_from_validated_rows(
        paired,
        &protocol.quill_null_samples,
        external_authority,
    )?;
    Ok(estimate)
}

fn estimate_qg6_joint_tail_from_validated_rows(
    paired: &PairedExperimentResult,
    quill_null_samples: &[PerfRawSample],
    external_authority: &Qg6ScheduleAuthority,
) -> Result<Qg6JointTailEstimate, EvidenceArtifactError> {
    estimate_qg6_joint_tail_with_budget(
        paired,
        quill_null_samples,
        external_authority,
        QG6_JOINT_TAIL_MIN_BOOTSTRAP_REPLICATES,
        QG6_JOINT_TAIL_MAX_BOOTSTRAP_REPLICATES,
    )
}

/// Test-only reduced-budget driver: identical mechanics, smaller replicate
/// floor/ceiling so decision-layer unit tests do not pay normative Monte
/// Carlo scale. Production paths always use the frozen constants.
#[cfg(test)]
fn estimate_qg6_joint_tail_fixture(
    paired: &PairedExperimentResult,
    quill_null_samples: &[PerfRawSample],
    external_authority: &Qg6ScheduleAuthority,
) -> Result<Qg6JointTailEstimate, EvidenceArtifactError> {
    estimate_qg6_joint_tail_with_budget(
        paired,
        quill_null_samples,
        external_authority,
        2_000,
        8_000,
    )
}

fn estimate_qg6_joint_tail_with_budget(
    paired: &PairedExperimentResult,
    quill_null_samples: &[PerfRawSample],
    external_authority: &Qg6ScheduleAuthority,
    replicate_floor: usize,
    replicate_ceiling: usize,
) -> Result<Qg6JointTailEstimate, EvidenceArtifactError> {
    if paired.config.bootstrap_resamples == 0 {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 joint tail estimator requires positive bootstrap resamples".to_owned(),
        });
    }

    let mut units = BTreeMap::<u64, Qg6TailUnit>::new();
    for (samples, stream) in [
        (&paired.null_samples[..], Qg6FormalStream::TantivyNull),
        (quill_null_samples, Qg6FormalStream::QuillNull),
        (&paired.effect_samples[..], Qg6FormalStream::Effect),
    ] {
        for sample in samples {
            let binding = sample.qg6_sample_binding.as_ref().ok_or_else(|| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail row is missing its timed-sample binding".to_owned(),
                }
            })?;
            let block_index = usize::try_from(sample.block_id).map_err(|_| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail block ID does not fit the platform".to_owned(),
                }
            })?;
            let block = external_authority
                .schedule
                .get(block_index)
                .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail row is outside the external schedule".to_owned(),
                })?;
            let role = qg6_role(stream, sample.arm);
            let unit = units.entry(block.unit_id).or_insert_with(|| Qg6TailUnit {
                query_index: block.query_index,
                role_leaves_ms: std::array::from_fn(|_| Vec::new()),
            });
            if unit.query_index != block.query_index {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail unit mixes query groups".to_owned(),
                });
            }
            let leaves = &mut unit.role_leaves_ms[qg6_tail_role_index(role)];
            if !leaves.is_empty() {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 joint tail unit repeats one logical role".to_owned(),
                });
            }
            for leaf in &binding.timed_sample.timing_leaves {
                let latency_ms =
                    Duration::from_nanos(leaf.observed_latency_ns()).as_secs_f64() * 1_000.0;
                if !latency_ms.is_finite() || latency_ms <= 0.0 {
                    return Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: "QG-6 joint tail leaf is not finite and positive".to_owned(),
                    });
                }
                leaves.push(latency_ms);
            }
        }
    }

    let mut queries = (0..external_authority.query_count)
        .map(|_| Vec::<Qg6TailUnit>::new())
        .collect::<Vec<_>>();
    for unit in units.into_values() {
        let query = queries.get_mut(unit.query_index).ok_or_else(|| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 joint tail unit names an unknown query".to_owned(),
            }
        })?;
        if unit
            .role_leaves_ms
            .iter()
            .any(|leaves| leaves.len() != external_authority.searches_per_sample)
        {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 joint tail unit lacks exact equal leaf cardinality".to_owned(),
            });
        }
        query.push(unit);
    }
    if queries
        .iter()
        .any(|units| units.len() != external_authority.rounds_per_query)
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 joint tail queries lack exact equal unit cardinality".to_owned(),
        });
    }

    let append_unit = |target: &mut [Vec<f64>; 6], unit: &Qg6TailUnit| {
        for (target, source) in target.iter_mut().zip(&unit.role_leaves_ms) {
            target.extend_from_slice(source);
        }
    };
    let mut point_leaves = std::array::from_fn(|_| Vec::new());
    for query in &queries {
        for unit in query {
            append_unit(&mut point_leaves, unit);
        }
    }
    let point_logs = qg6_tail_contrast_logs(&qg6_tail_arm_quantiles(&mut point_leaves));

    let mut authority_seed = Sha256::new();
    authority_seed.update(external_authority.authority_sha256.as_bytes());
    authority_seed.update(paired.config.bootstrap_seed.to_le_bytes());
    let authority_seed = authority_seed.finalize();
    let authority_seed_bytes: [u8; 32] = authority_seed.into();
    let mut chain = u64::from_le_bytes(
        authority_seed_bytes[..std::mem::size_of::<u64>()]
            .try_into()
            .map_err(|_| EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 joint tail seed digest is malformed".to_owned(),
            })?,
    );
    // Replicate seeds reproduce the historical single sequential splitmix
    // chain EXACTLY: replicate i starts from the chain state after i full
    // replicates (visits_per_replicate advances each). The per-replicate
    // states are precomputed once so parallel workers can index them without
    // sharing mutable state, and draw order equals index order for batch
    // statistics.
    let visits_per_replicate = queries
        .len()
        .saturating_mul(external_authority.rounds_per_query);
    let mut replicate_seeds: Vec<u64> = Vec::new();
    let extend_seeds = |seeds: &mut Vec<u64>, chain: &mut u64, up_to: usize| {
        while seeds.len() < up_to {
            seeds.push(*chain);
            for _ in 0..visits_per_replicate {
                *chain = splitmix64(*chain);
            }
        }
    };
    extend_seeds(&mut replicate_seeds, &mut chain, replicate_floor);
    // Normative driver (bd-quill-e8-perf-doctrine-x4e4.9.3): start at the
    // frozen replicate floor regardless of the generic paired config, then
    // double while the effect-arm decision boundaries keep moving away from
    // flip risk (measured as the worst boundary-margin-to-SE ratio). Two
    // consecutive escalations without a >=1.5x improvement mark the data
    // genuinely borderline: fail closed via `monte_carlo_stable = false`
    // instead of burning the ceiling. Replicates are drawn in parallel and
    // stored by index, so the experiment stays deterministic for a given
    // (authority, seed) pair.
    let mut drawn = 0_usize;
    let mut escalations = 0_u32;
    let mut monte_carlo_stable = false;
    let mut stalled_escalations = 0_u32;
    let mut last_stability_ratio = f64::INFINITY;
    let mut bootstrap: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(replicate_floor));
    loop {
        let target = if drawn == 0 {
            replicate_floor
        } else {
            drawn.saturating_mul(2).min(replicate_ceiling)
        };
        if target <= drawn {
            // Ceiling reached without boundary stability: fail closed by
            // flag, never by aborting the estimate.
            monte_carlo_stable = false;
            break;
        }
        extend_seeds(&mut replicate_seeds, &mut chain, target);
        let batch: Vec<[f64; 6]> = (drawn..target)
            .into_par_iter()
            .map(|index| {
                let mut seed = replicate_seeds[index];
                let mut role_leaves = std::array::from_fn(|_| Vec::new());
                for_each_qg6_joint_tail_bootstrap_unit(
                    &mut seed,
                    queries.len(),
                    external_authority.rounds_per_query,
                    |query_index, unit_index| {
                        append_unit(&mut role_leaves, &queries[query_index][unit_index]);
                    },
                )?;
                Ok(qg6_tail_contrast_logs(&qg6_tail_arm_quantiles(
                    &mut role_leaves,
                )))
            })
            .collect::<Result<Vec<_>, EvidenceArtifactError>>()?;
        for row in batch {
            for (values, value) in bootstrap.iter_mut().zip(row) {
                values.push(value);
            }
        }
        drawn = target;
        if drawn > replicate_floor {
            escalations = escalations.saturating_add(1);
        }
        let stability_ratio = qg6_worst_stability_ratio(&bootstrap);
        if stability_ratio >= QG6_MC_SAFETY_Z {
            monte_carlo_stable = true;
            break;
        }
        if drawn > replicate_floor && stability_ratio < 1.5 * last_stability_ratio.max(1.0) {
            stalled_escalations = stalled_escalations.saturating_add(1);
            if stalled_escalations >= 2 {
                break;
            }
        } else {
            stalled_escalations = 0;
        }
        last_stability_ratio = stability_ratio;
    }
    let [
        mut tantivy_p50,
        mut tantivy_p99,
        mut quill_p50,
        mut quill_p99,
        mut effect_p50,
        mut effect_p99,
    ] = bootstrap;
    // Batch-means errors must be measured on draw order; the contrast builder
    // sorts its inputs in place, so every SE is captured first. Non-finite
    // densities saturate to MAX (never NaN/Inf: they must survive JSON
    // sealing) which correctly forces the unstable flag downstream.
    let se = |vector: &[f64], levels: [f64; 2]| {
        levels
            .into_iter()
            .map(|level| {
                let value = qg6_mc_standard_error(vector, level);
                if value.is_finite() {
                    value
                } else if value.is_nan() {
                    0.0
                } else {
                    f64::MAX
                }
            })
            .fold(f64::NAN, f64::max)
    };
    let se_tantivy_p50 = se(&tantivy_p50, [QG6_PER_CELL_ALPHA, 1.0 - QG6_PER_CELL_ALPHA]);
    let se_quill_p50 = se(&quill_p50, [QG6_PER_CELL_ALPHA, 1.0 - QG6_PER_CELL_ALPHA]);
    let se_effect_p50 = se(&effect_p50, [QG6_PER_CELL_ALPHA, 1.0 - QG6_PER_CELL_ALPHA]);
    let se_tantivy_p99 = se(&tantivy_p99, [1.0 - QG6_PER_CELL_ALPHA, 1.0]);
    let se_quill_p99 = se(&quill_p99, [1.0 - QG6_PER_CELL_ALPHA, 1.0]);
    let se_effect_p99 = se(&effect_p99, [1.0 - QG6_PER_CELL_ALPHA, 1.0]);
    let tantivy_null = qg6_joint_tail_contrast(
        point_logs[0],
        point_logs[1],
        &mut tantivy_p50,
        &mut tantivy_p99,
        se_tantivy_p50,
        se_tantivy_p99,
    );
    let quill_null = qg6_joint_tail_contrast(
        point_logs[2],
        point_logs[3],
        &mut quill_p50,
        &mut quill_p99,
        se_quill_p50,
        se_quill_p99,
    );
    let effect = qg6_joint_tail_contrast(
        point_logs[4],
        point_logs[5],
        &mut effect_p50,
        &mut effect_p99,
        se_effect_p50,
        se_effect_p99,
    );
    Ok(Qg6JointTailEstimate {
        schema_version: QG6_JOINT_TAIL_SCHEMA_VERSION.to_owned(),
        query_count: external_authority.query_count,
        units_per_query: external_authority.rounds_per_query,
        leaves_per_arm_per_unit: external_authority.searches_per_sample,
        bootstrap_resamples: paired.config.bootstrap_resamples,
        replicates_used: drawn,
        monte_carlo_stable,
        escalations,
        tantivy_null,
        quill_null,
        effect,
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
///
/// For QG-8 both engines materialize the matrix width. For QG-1 the Quill arm
/// materializes the matrix width while the Tantivy arm materializes whichever
/// writer width its incumbent screen froze, so the two observations there are
/// not required to be equal to each other — see [`Self::validate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfConcurrencyWitness {
    /// Thread-width knob declared by the normative matrix cell. Always the
    /// Quill width; also the Tantivy width outside QG-1.
    pub configured_threads: usize,
    /// Exactly one Quill and one Tantivy observation, in engine order.
    pub observations: Vec<EngineConcurrencyObservation>,
}

impl PerfConcurrencyWitness {
    /// Validate the witness against the contract its gate actually has.
    ///
    /// QG-8 is a scaling gate: both engines run at the matrix width, and both
    /// observations must equal `configured_threads` exactly. This is unchanged.
    ///
    /// QG-1 screens a Tantivy incumbent, and the frozen arm may legitimately be
    /// a different writer width from the Quill width the cell was configured
    /// with. Quill therefore still binds to `configured_threads`, while Tantivy
    /// is required here only to be positive and stable across the run; its
    /// exact value is bound to the screen's selected fixed width by
    /// [`Qg1IncumbentScreenEvidence::verify_selected_width_witness`]. That is a
    /// division of labour, not a relaxation: QG-1 evidence must screen every
    /// required engine cell to be admissible at all, so a witness that only
    /// reached the positive-and-stable bar can never ratchet unscreened.
    fn validate(&self, gate: PerfGate) -> Result<(), EvidenceArtifactError> {
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
            // The screened Tantivy arm is the one width this witness does not
            // pin here; every other property, including stability across the
            // run, is required of both engines exactly as before.
            let binds_configured_width =
                engine == PerfConcurrencyEngine::Quill || gate != PerfGate::Qg1;
            let width = observation.min_observed_worker_pool_threads;
            if observation.engine != engine
                || observation.observer != observer
                || observation.observation_count == 0
                || width == 0
                || observation.max_observed_worker_pool_threads != width
                || (binds_configured_width && width != self.configured_threads)
            {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: "scaling concurrency witness is missing, duplicated, unstable, or \
                         disagrees with the configured pool width"
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
        /// Formal six-arm QG-6 protocol evidence. This is deliberately
        /// separate from the QG-1 treatment-arm null: it carries the
        /// independently retained pre-timing schedule authority plus the
        /// Quill/Quill raw stream needed to complete T/T, Q/Q, and T/Q.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        qg6_protocol: Option<Box<Qg6FormalProtocolEvidence>>,
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

/// QG-6-only evidence that completes the formal six-arm measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6FormalProtocolEvidence {
    /// Exact schedule frozen and retained before any timed search.
    pub schedule_authority: Qg6ScheduleAuthority,
    /// Same-invocation Quill/Quill null rows, including authenticated leaves.
    pub quill_null_samples: Vec<PerfRawSample>,
    /// Joint query-first p50/p99 estimate recomputed from all six raw arms.
    pub joint_tail: Qg6JointTailEstimate,
}

impl Qg6FormalProtocolEvidence {
    /// Build formal protocol evidence from raw Quill/Quill rows and an
    /// independently retained pre-timing authority.
    ///
    /// # Errors
    ///
    /// Rejects any invalid six-arm row, authority, semantic receipt, or joint
    /// tail estimate input.
    pub fn new_against_authority(
        paired: &PairedExperimentResult,
        quill_null_samples: Vec<PerfRawSample>,
        external_authority: &Qg6ScheduleAuthority,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
    ) -> Result<Self, EvidenceArtifactError> {
        validate_qg6_formal_protocol_rows(
            paired,
            &quill_null_samples,
            external_authority,
            identity,
            contract,
        )?;
        let joint_tail = estimate_qg6_joint_tail_from_validated_rows(
            paired,
            &quill_null_samples,
            external_authority,
        )?;
        Ok(Self {
            schedule_authority: external_authority.clone(),
            quill_null_samples,
            joint_tail,
        })
    }

    /// Test-only twin of [`Self::new_against_authority`] using the reduced
    /// fixture replicate budget; production code must use the normative
    /// constructor.
    #[cfg(test)]
    pub(crate) fn new_against_authority_fixture(
        paired: &PairedExperimentResult,
        quill_null_samples: Vec<PerfRawSample>,
        external_authority: &Qg6ScheduleAuthority,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
    ) -> Result<Self, EvidenceArtifactError> {
        validate_qg6_formal_protocol_rows(
            paired,
            &quill_null_samples,
            external_authority,
            identity,
            contract,
        )?;
        let joint_tail =
            estimate_qg6_joint_tail_fixture(paired, &quill_null_samples, external_authority)?;
        Ok(Self {
            schedule_authority: external_authority.clone(),
            quill_null_samples,
            joint_tail,
        })
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Qg6FormalStream {
    TantivyNull,
    QuillNull,
    Effect,
}

fn qg6_role(stream: Qg6FormalStream, arm: PerfSampleArm) -> Qg6ArmRole {
    match (stream, arm) {
        (Qg6FormalStream::TantivyNull, PerfSampleArm::Control) => Qg6ArmRole::TantivyNullLeft,
        (Qg6FormalStream::TantivyNull, PerfSampleArm::Treatment) => Qg6ArmRole::TantivyNullRight,
        (Qg6FormalStream::QuillNull, PerfSampleArm::Control) => Qg6ArmRole::QuillNullLeft,
        (Qg6FormalStream::QuillNull, PerfSampleArm::Treatment) => Qg6ArmRole::QuillNullRight,
        (Qg6FormalStream::Effect, PerfSampleArm::Control) => Qg6ArmRole::EffectControl,
        (Qg6FormalStream::Effect, PerfSampleArm::Treatment) => Qg6ArmRole::EffectTreatment,
    }
}

fn validate_qg6_sample_stream(
    samples: &[PerfRawSample],
    stream: Qg6FormalStream,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
    represented: &mut BTreeMap<(u64, Qg6ArmRole), usize>,
    row_keys: &mut BTreeSet<(Qg6FormalStream, u64, Qg6ArmRole)>,
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
        let role = qg6_role(stream, sample.arm);
        if !row_keys.insert((stream, sample.block_id, role)) {
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
        Qg6FormalStream::Effect,
        identity,
        contract,
        &mut represented,
        &mut row_keys,
    )?;
    validate_qg6_sample_stream(
        &paired.null_samples,
        Qg6FormalStream::TantivyNull,
        identity,
        contract,
        &mut represented,
        &mut row_keys,
    )?;
    let expected = QG6_QUERY_GROUP_IDS
        .into_iter()
        .flat_map(|group_id| {
            [
                Qg6ArmRole::TantivyNullLeft,
                Qg6ArmRole::TantivyNullRight,
                Qg6ArmRole::EffectControl,
                Qg6ArmRole::EffectTreatment,
            ]
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

fn validate_qg6_formal_protocol_rows(
    paired: &PairedExperimentResult,
    quill_null_samples: &[PerfRawSample],
    external_authority: &Qg6ScheduleAuthority,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
) -> Result<(), EvidenceArtifactError> {
    paired
        .verify_recomputed()
        .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
            reason: format!("QG-6 paired T/T and T/Q rows failed authenticated replay: {error}"),
        })?;
    let authority = external_authority;
    authority
        .verify()
        .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
            reason: format!("QG-6 schedule authority failed verification: {error}"),
        })?;
    if authority.identity.corpus_sha256 != identity.prepared_corpus_sha256
        || authority.identity.query_manifest_sha256 != identity.query_manifest_sha256
        || authority.identity.config_contract_sha256 != identity.config_contract_sha256
        || authority.identity.document_count != contract.document_count
        || authority.identity.k != contract.k
        || authority.query_count != contract.groups.len()
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 schedule authority does not bind the exact prepared cell identity"
                .to_owned(),
        });
    }
    validate_paired_blocks(quill_null_samples, &paired.config).map_err(|error| {
        EvidenceArtifactError::InconsistentArtifact {
            reason: format!("QG-6 Quill/Quill null stream is invalid: {error}"),
        }
    })?;
    let mut represented = BTreeMap::new();
    let mut row_keys = BTreeSet::new();
    for (samples, stream) in [
        (&paired.null_samples[..], Qg6FormalStream::TantivyNull),
        (quill_null_samples, Qg6FormalStream::QuillNull),
        (&paired.effect_samples[..], Qg6FormalStream::Effect),
    ] {
        validate_qg6_sample_stream(
            samples,
            stream,
            identity,
            contract,
            &mut represented,
            &mut row_keys,
        )?;
    }
    let expected_roles = QG6_QUERY_GROUP_IDS
        .into_iter()
        .flat_map(|group_id| {
            Qg6ArmRole::ALL
                .into_iter()
                .map(move |role| (group_id, role))
        })
        .collect::<BTreeSet<_>>();
    if represented.keys().copied().collect::<BTreeSet<_>>() != expected_roles
        || represented
            .values()
            .any(|count| *count != authority.rounds_per_query)
    {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason:
                "QG-6 formal rows do not provide the exact schedule multiplicity for all six roles"
                    .to_owned(),
        });
    }

    let mut observed_schedule_rows = BTreeSet::new();
    for (samples, stream) in [
        (&paired.null_samples[..], Qg6FormalStream::TantivyNull),
        (quill_null_samples, Qg6FormalStream::QuillNull),
        (&paired.effect_samples[..], Qg6FormalStream::Effect),
    ] {
        let expected_comparison = match stream {
            Qg6FormalStream::TantivyNull => Qg6Comparison::TantivyNull,
            Qg6FormalStream::QuillNull => Qg6Comparison::QuillNull,
            Qg6FormalStream::Effect => Qg6Comparison::Effect,
        };
        for sample in samples {
            if sample.scope != paired.scope || sample.provenance != paired.provenance {
                return Err(EvidenceArtifactError::InvalidProvenance {
                    reason:
                        "QG-6 formal rows do not share the exact effect-stream scope and provenance"
                            .to_owned(),
                });
            }
            let binding = sample.qg6_sample_binding.as_ref().ok_or_else(|| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 formal row is missing its authenticated timed sample".to_owned(),
                }
            })?;
            let block_index = usize::try_from(sample.block_id).map_err(|_| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 formal block ID does not fit the platform".to_owned(),
                }
            })?;
            let block = authority.schedule.get(block_index).ok_or_else(|| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 formal row names a block outside the retained authority"
                        .to_owned(),
                }
            })?;
            let expected_role = qg6_role(stream, sample.arm);
            let scheduled_role = match binding.timed_sample.order {
                Qg6SampleOrder::First => block.first,
                Qg6SampleOrder::Second => block.second,
            };
            let expected_sample_id = block
                .block_id
                .checked_mul(2)
                .and_then(|base| {
                    base.checked_add(u64::from(
                        binding.timed_sample.order == Qg6SampleOrder::Second,
                    ))
                })
                .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                    reason: "QG-6 formal sample ID overflowed".to_owned(),
                })?;
            if block.block_id != sample.block_id
                || block.query_index != binding.timed_sample.query_index
                || block.comparison != expected_comparison
                || binding.timed_sample.comparison != expected_comparison
                || binding.timed_sample.arm != expected_role
                || scheduled_role != expected_role
                || sample.sample_id != expected_sample_id
                || binding.timed_sample.timing_leaves.len() != authority.searches_per_sample
                || !observed_schedule_rows.insert((block.block_id, expected_role))
            {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason:
                        "QG-6 formal row does not match the externally retained pre-timing schedule"
                            .to_owned(),
                });
            }
        }
    }
    let expected_schedule_rows = authority
        .schedule
        .iter()
        .flat_map(|block| {
            [
                (block.block_id, block.first),
                (block.block_id, block.second),
            ]
        })
        .collect::<BTreeSet<_>>();
    if observed_schedule_rows != expected_schedule_rows {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "QG-6 formal evidence is missing or repeats a retained schedule row".to_owned(),
        });
    }
    Ok(())
}

fn validate_qg6_formal_protocol(
    paired: &PairedExperimentResult,
    protocol: &Qg6FormalProtocolEvidence,
    external_authority: &Qg6ScheduleAuthority,
    identity: &PerfInputIdentity,
    contract: &Qg6SemanticContract,
) -> Result<(), EvidenceArtifactError> {
    if &protocol.schedule_authority != external_authority {
        return Err(EvidenceArtifactError::InvalidProvenance {
            reason: "persisted QG-6 schedule authority does not match the independently retained authority"
                .to_owned(),
        });
    }
    validate_qg6_formal_protocol_rows(
        paired,
        &protocol.quill_null_samples,
        external_authority,
        identity,
        contract,
    )?;
    // Verification must reproduce the persisted estimate under ITS OWN
    // replicate budget. Production artifacts always carry the normative
    // floor/ceiling; test fixtures declare a reduced budget and the stored
    // `replicates_used` is the exact deterministic draw count to replay.
    #[cfg(test)]
    let recomputed = {
        let stored = &protocol.joint_tail;
        let floor = stored.replicates_used.max(1);
        estimate_qg6_joint_tail_with_budget(
            paired,
            &protocol.quill_null_samples,
            external_authority,
            floor,
            floor,
        )?
    };
    #[cfg(not(test))]
    let recomputed = estimate_qg6_joint_tail_from_validated_rows(
        paired,
        &protocol.quill_null_samples,
        external_authority,
    )?;
    if recomputed != protocol.joint_tail {
        return Err(EvidenceArtifactError::InconsistentArtifact {
            reason: "persisted QG-6 joint tail estimate does not recompute from raw leaves"
                .to_owned(),
        });
    }
    Ok(())
}

fn resolve_qg6_schedule_authority_for_replay<'authority>(
    external_authorities: &[&'authority Qg6ScheduleAuthority],
    protocol: &Qg6FormalProtocolEvidence,
) -> Result<&'authority Qg6ScheduleAuthority, EvidenceArtifactError> {
    let mut matching = external_authorities.iter().copied().filter(|authority| {
        authority.authority_sha256 == protocol.schedule_authority.authority_sha256
    });
    let selected = matching
        .next()
        .ok_or_else(|| EvidenceArtifactError::InvalidProvenance {
            reason: "QG-6 evidence has no independently retained schedule authority".to_owned(),
        })?;
    if matching.next().is_some() {
        return Err(EvidenceArtifactError::InvalidProvenance {
            reason: "QG-6 evidence matches more than one retained schedule authority".to_owned(),
        });
    }
    selected
        .verify()
        .map_err(|error| EvidenceArtifactError::InvalidProvenance {
            reason: format!("retained QG-6 schedule authority failed verification: {error}"),
        })?;
    if selected != &protocol.schedule_authority {
        return Err(EvidenceArtifactError::InvalidProvenance {
            reason: "QG-6 evidence substituted schedule bytes under a retained authority digest"
                .to_owned(),
        });
    }
    Ok(selected)
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
            (true, Some(witness)) => witness.validate(spec.gate)?,
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
                qg6_protocol: None,
                hierarchical,
                hierarchical_null,
                reconciliation,
            },
            status,
            reasons,
        })
    }

    /// Attach the externally authorized Quill/Quill stream that completes a
    /// formal six-arm QG-6 measurement.
    ///
    /// A successfully replayed protocol replaces the temporary tail hold with
    /// the joint query-first p50/p99 measurement. Invalid T/T or Q/Q nulls
    /// remain durable no-decisions; the effect intervals stay measured so the
    /// ratchet can apply the standing p50 and p99 targets without re-estimating.
    ///
    /// # Errors
    ///
    /// Rejects non-QG-6 cells, duplicate attachment, unbounded raw rows,
    /// malformed Q/Q pairs, authority substitution, and any row that does not
    /// exactly match the schedule frozen before timing.
    pub fn attach_qg6_formal_protocol_against_authority(
        &mut self,
        protocol: Qg6FormalProtocolEvidence,
        policy: &EvidencePolicy,
        external_authority: &Qg6ScheduleAuthority,
    ) -> Result<(), EvidenceArtifactError> {
        if self.spec.gate != PerfGate::Qg6 {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "only QG-6 cells may carry formal six-arm protocol evidence".to_owned(),
            });
        }
        let identity = self.spec.input_identity.clone().ok_or_else(|| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 formal protocol is missing the prepared input identity".to_owned(),
            }
        })?;
        let contract = self.spec.qg6_semantic_contract.clone().ok_or_else(|| {
            EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 formal protocol is missing the semantic contract".to_owned(),
            }
        })?;
        let EvidenceCellBody::Paired {
            paired,
            treatment_arm_null,
            qg6_protocol,
            ..
        } = &mut self.body
        else {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 formal protocol requires a paired evidence cell".to_owned(),
            });
        };
        if treatment_arm_null.is_some() || qg6_protocol.is_some() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-6 formal protocol is duplicated or collides with a QG-1 null"
                    .to_owned(),
            });
        }
        let retained = paired
            .effect_samples
            .len()
            .checked_add(paired.null_samples.len())
            .and_then(|count| count.checked_add(protocol.quill_null_samples.len()))
            .ok_or_else(|| EvidenceArtifactError::UnboundedRawSamples {
                cell_id: self.cell_id.clone(),
                count: usize::MAX,
                max: policy.max_raw_samples,
            })?;
        if retained > policy.max_raw_samples {
            return Err(EvidenceArtifactError::UnboundedRawSamples {
                cell_id: self.cell_id.clone(),
                count: retained,
                max: policy.max_raw_samples,
            });
        }
        validate_qg6_formal_protocol(paired, &protocol, external_authority, &identity, &contract)?;
        let tail_reasons = qg6_joint_tail_decision_reasons(&protocol.joint_tail);
        self.reasons
            .retain(|reason| reason.code != "qg6.tail_protocol_not_implemented");
        let has_no_claim = self
            .reasons
            .iter()
            .chain(&tail_reasons)
            .any(|reason| reason.severity >= EvidenceSeverity::NoClaim);
        *qg6_protocol = Some(Box::new(protocol));
        self.reasons.extend(tail_reasons);
        self.reasons.truncate(EVIDENCE_MAX_REASONS);
        self.status = if has_no_claim {
            EvidenceDecisionStatus::NoDecision
        } else {
            EvidenceDecisionStatus::MeasuredProvisional
        };
        Ok(())
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
        self.attach_treatment_arm_null_against_qg1_authority(treatment_arm_null, policy, None)
    }

    /// Attach a treatment-arm A/A null measured under a QG-1 producer, using
    /// the expectation that producer retained.
    ///
    /// A QG-1 null carries a sealed lifecycle authority in its configuration,
    /// and authority-free verification refuses to authenticate such evidence
    /// from its own artifact. The live matrix therefore hands the producer's
    /// retained expectation to this entry; replay hands the one its consumer
    /// kept. Passing `None` keeps the exact generic contract, including that
    /// refusal.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::attach_treatment_arm_null`].
    pub fn attach_treatment_arm_null_against_qg1_authority(
        &mut self,
        treatment_arm_null: PairedExperimentResult,
        policy: &EvidencePolicy,
        external_qg1_authority: Option<&Qg1ExpectedAuthority>,
    ) -> Result<(), EvidenceArtifactError> {
        policy.validate()?;
        treatment_arm_null.verify_recomputed_against_qg1_authority(external_qg1_authority)?;
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
                qg6_protocol,
                ..
            } if self.spec.gate == PerfGate::Qg6 => {
                self.status == EvidenceDecisionStatus::MeasuredProvisional
                    && hierarchical.is_some()
                    && hierarchical_null.is_some()
                    && qg6_protocol.is_some()
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
        self.verify_recomputed_against_authorities(policy, &[], &[])
    }

    /// Recompute this cell, selecting the retained QG-1 expectation it was
    /// measured under from the consumer's independently held set.
    ///
    /// A reloaded QG-1 cell can never carry its own expectation: the producer
    /// capability preimages are not serialized and cannot be reconstructed
    /// from the artifact. The consumer therefore supplies the expectations it
    /// retained, and exactly the one that issued this cell's sealed authority
    /// is used. An empty set reproduces the generic contract, under which a
    /// QG-1 cell still fails closed rather than authenticating itself.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::verify_recomputed`].
    pub fn verify_recomputed_against_qg1_authorities(
        &self,
        policy: &EvidencePolicy,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        self.verify_recomputed_against_authorities(policy, external_qg1_authorities, &[])
    }

    /// Recompute this cell against every independently retained authority it
    /// requires. QG-1 selects its producer expectation; QG-6 selects the exact
    /// pre-timing schedule authority. Neither gate may authenticate itself
    /// from authority bytes serialized inside the cell.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::verify_recomputed`], plus a typed
    /// provenance refusal for an absent, duplicate, malformed, or substituted
    /// external authority.
    pub fn verify_recomputed_against_authorities(
        &self,
        policy: &EvidencePolicy,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        match &self.body {
            EvidenceCellBody::Paired {
                paired,
                treatment_arm_null,
                qg6_protocol,
                ..
            } => {
                if self.spec.gate == PerfGate::Qg6 && qg6_protocol.is_none() {
                    return Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: format!(
                            "QG-6 cell {} is missing formal six-arm protocol evidence",
                            self.cell_id
                        ),
                    });
                }
                if self.spec.gate != PerfGate::Qg6 && qg6_protocol.is_some() {
                    return Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: format!(
                            "non-QG-6 cell {} carries formal six-arm protocol evidence",
                            self.cell_id
                        ),
                    });
                }
                // Replay context: only an externally retained expectation that
                // names this cell's producer can authenticate it. An empty set
                // is not permission to fall back to the expectation a live
                // configuration still carries — that would let a QG-1 cell
                // authenticate itself, which is the whole defect this guards.
                let expected = resolve_qg1_expected_authority_for_replay(
                    external_qg1_authorities,
                    &paired.config,
                );
                let mut rebuilt =
                    Self::evaluate(self.spec.clone(), paired.as_ref().clone(), policy)?;
                // QG-6 semantic bindings are cell-level evidence. Validate
                // them before the generic paired estimator so hostile row
                // mutations receive the semantic fail-closed classification
                // instead of being obscured by a lower-level pair mismatch.
                paired.verify_recomputed_against_qg1_authority(expected)?;
                if let Some(treatment_arm_null) = treatment_arm_null {
                    rebuilt.attach_treatment_arm_null_against_qg1_authority(
                        treatment_arm_null.as_ref().clone(),
                        policy,
                        expected,
                    )?;
                }
                if let Some(qg6_protocol) = qg6_protocol {
                    let external_authority = resolve_qg6_schedule_authority_for_replay(
                        external_qg6_authorities,
                        qg6_protocol,
                    )?;
                    rebuilt.attach_qg6_formal_protocol_against_authority(
                        qg6_protocol.as_ref().clone(),
                        policy,
                        external_authority,
                    )?;
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

/// Durable QG-1 fastest-incumbent screen outcome and the decision it bound.
///
/// The screen is the part of QG-1 that decides *which* Tantivy arm a headline
/// may be measured against, so an artifact that omits it cannot be audited for
/// weaker-headline substitution afterwards. Persisting it here puts the
/// selection, its full preregistered candidate universe, and the same-invocation
/// decision streams inside the same hash-sealed object as the cells.
///
/// Replay is authority-bearing exactly like a QG-1 cell: the never-serialized
/// producer expectations are supplied by the consumer that retained them, and
/// every component — each pilot stream and the decision — must be named by that
/// retained set exactly once. A screen that cannot re-derive under the supplied
/// expectations fails closed rather than being admitted on its own say-so.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg1IncumbentScreenEvidence {
    /// Canonical `gate/fixture/metric` identity of the screened QG-1 bulk cell.
    pub cell_id: String,
    /// Exact non-writer semantics held constant across every candidate.
    pub semantic_contract: Qg1TantivySemanticContract,
    /// Screen outcome: one uniquely fastest candidate, or an explicit
    /// `NoDecision` carrying its stable reason.
    pub screen: Qg1TantivyIncumbentScreen,
    /// Same-invocation T/Quill, T/T, and Q/Q decision. Required exactly when
    /// the screen selected a candidate; forbidden when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<Qg1TantivyIncumbentDecision>,
}

impl Qg1IncumbentScreenEvidence {
    /// Whether this screen froze one uniquely fastest incumbent arm.
    ///
    /// An incomplete screen is not a failure of the run — it is a valid
    /// `NoDecision` — but it can never headline, which is what the gate fold
    /// consumes this for.
    #[must_use]
    pub fn has_selection(&self) -> bool {
        self.screen.selected_candidate.is_some()
    }

    /// Require the decision streams to be the very evidence the artifact's
    /// named required cell was computed from.
    ///
    /// Without this a screen and a decision could merely *coexist* with an
    /// unrelated cell measured in some other invocation, which is precisely the
    /// substitution the incumbent screen exists to prevent. The named cell is
    /// therefore rebuilt from the decision's own raw streams — T/Quill against
    /// the real T/T null, and the same T/Quill against the real Q/Q null for
    /// the treatment-arm null — under the artifact's own policy, and must equal
    /// the stored cell exactly. An absent named cell rejects.
    fn verify_decision_binds_named_cell(
        &self,
        decision: &Qg1TantivyIncumbentDecision,
        decision_authority: &Qg1ExpectedAuthority,
        cells: &[EvidenceCell],
        policy: &EvidencePolicy,
    ) -> Result<(), EvidenceArtifactError> {
        let Some(named) = cells.iter().find(|candidate| {
            candidate.cell_id == self.cell_id
                && candidate.spec.gate == PerfGate::Qg1
                && candidate.spec.role == EvidenceRole::Required
        }) else {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent decision names required cell {:?}, which this artifact does \
                     not carry",
                    self.cell_id
                ),
            });
        };
        let estimator = |effect: &[PerfRawSample], null: &[PerfRawSample], role: &str| {
            crate::perf::estimate_paired_experiment_against_qg1_authority(
                effect,
                null,
                &decision.estimator_config,
                Some(decision_authority),
            )
            .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent decision {role} stream does not estimate under its retained \
                     authority: {error}"
                ),
            })
        };
        // The frozen incumbent width must be the width this cell actually
        // materialized, proven per engine, before its streams are allowed to
        // stand in for the cell's result.
        self.verify_selected_width_witness(named)?;
        let effect = estimator(
            &decision.tantivy_vs_quill.samples,
            &decision.tantivy_null.samples,
            "effect",
        )?;
        let treatment_arm_null = estimator(
            &decision.tantivy_vs_quill.samples,
            &decision.quill_null.samples,
            "treatment-arm null",
        )?;
        let mut rebuilt = EvidenceCell::evaluate(named.spec.clone(), effect, policy)?;
        rebuilt.attach_treatment_arm_null_against_qg1_authority(
            treatment_arm_null,
            policy,
            Some(decision_authority),
        )?;
        if rebuilt != *named {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "required cell {:?} is not the cell this QG-1 incumbent decision measured",
                    self.cell_id
                ),
            });
        }
        Ok(())
    }

    /// Resolve the canonical cell this screen names from the frozen matrix.
    fn resolved_cell(&self) -> Result<PerfCellSpec, EvidenceArtifactError> {
        PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|cell| {
                format!("{}/{}/{}", PerfGate::Qg1, cell.fixture, cell.metric) == self.cell_id
            })
            .cloned()
            .ok_or_else(|| EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent screen names cell {:?}, which is not in the frozen matrix",
                    self.cell_id
                ),
            })
    }

    /// The frozen Tantivy writer width this screen selected, when it selected
    /// one at all.
    ///
    /// A `ShippingAuto` selection is refused here rather than reported: its
    /// materialized width is chosen by Tantivy at runtime and is typed
    /// unobservable, so it can never satisfy the frozen observed-width
    /// requirement. Such a screen must be recorded as `NoDecision` — accepting it
    /// with a relaxed or absent witness is exactly the weakening this refuses.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::InconsistentArtifact`] when the
    /// selection is `ShippingAuto`.
    fn selected_writer_threads(&self) -> Result<Option<usize>, EvidenceArtifactError> {
        let Some(selected) = self.screen.selected_candidate.as_ref() else {
            return Ok(None);
        };
        match selected.writer_mode {
            Qg1TantivyWriterMode::Fixed { writer_threads } => Ok(Some(writer_threads)),
            Qg1TantivyWriterMode::ShippingAuto => {
                Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "QG-1 incumbent screen for {} selected the shipping-auto arm, whose \
                     materialized writer width is unobservable; an unobservable width is a \
                     NoDecision, never a relaxed witness",
                        self.cell_id
                    ),
                })
            }
        }
    }

    /// Require the named cell's per-engine witness to prove the exact widths
    /// this screen froze.
    ///
    /// Quill must have materialized the matrix width, and Tantivy must have
    /// materialized the selected candidate's fixed writer width. This is the
    /// only place the screened Tantivy width is pinned to an exact value:
    /// [`PerfConcurrencyWitness::validate`] deliberately requires only a
    /// positive, stable Tantivy width for QG-1, because the width that arm
    /// legitimately ran at is the frozen one, which the witness alone cannot
    /// know. QG-8 keeps its both-equal contract there and never reaches here.
    fn verify_selected_width_witness(
        &self,
        named: &EvidenceCell,
    ) -> Result<(), EvidenceArtifactError> {
        let Some(writer_threads) = self.selected_writer_threads()? else {
            return Ok(());
        };
        let Some(witness) = named.spec.concurrency_witness.as_ref() else {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "screen-bound QG-1 cell {} has no concurrency witness to prove its frozen \
                     incumbent width",
                    self.cell_id
                ),
            });
        };
        let observed = |engine: PerfConcurrencyEngine| {
            witness
                .observations
                .iter()
                .find(|observation| observation.engine == engine)
        };
        let proves = |engine: PerfConcurrencyEngine, expected: usize| {
            observed(engine).is_some_and(|observation| {
                observation.min_observed_worker_pool_threads == expected
                    && observation.max_observed_worker_pool_threads == expected
            })
        };
        if !proves(PerfConcurrencyEngine::Quill, witness.configured_threads) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "screen-bound QG-1 cell {} did not materialize the configured Quill width {}",
                    self.cell_id, witness.configured_threads
                ),
            });
        }
        if !proves(PerfConcurrencyEngine::Tantivy, writer_threads) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "screen-bound QG-1 cell {} froze Tantivy writer width {writer_threads}, which \
                     its concurrency witness does not prove was materialized",
                    self.cell_id
                ),
            });
        }
        Ok(())
    }

    /// Reject a screen whose outcome is neither a selection nor a valid
    /// `NoDecision`, before any authority work is attempted.
    fn validate_shape(&self) -> Result<(), EvidenceArtifactError> {
        let inconsistent = |reason: String| EvidenceArtifactError::InconsistentArtifact { reason };
        // Refuse an unobservable selected width before anything else, so a
        // shipping-auto selection is rejected on its own terms rather than
        // incidentally through a downstream witness or decision rule.
        self.selected_writer_threads()?;
        self.semantic_contract.contract_sha256().map_err(|error| {
            inconsistent(format!(
                "QG-1 incumbent semantic contract is not fully pinned: {error}"
            ))
        })?;
        match (
            self.screen.selected_candidate.as_ref(),
            self.screen.no_decision_reason.as_ref(),
            self.decision.as_ref(),
        ) {
            (Some(_), Some(_), _) => Err(inconsistent(
                "QG-1 incumbent screen cannot both select a candidate and declare NoDecision"
                    .to_owned(),
            )),
            (Some(_), None, None) => Err(inconsistent(
                "a selected QG-1 incumbent screen must carry its same-invocation decision evidence"
                    .to_owned(),
            )),
            (Some(_), None, Some(_)) => Ok(()),
            (None, Some(_), Some(_)) => Err(inconsistent(
                "a NoDecision QG-1 incumbent screen must not carry decision evidence".to_owned(),
            )),
            (None, Some(reason), None) => {
                if reason.trim().is_empty() || reason.len() > EVIDENCE_MAX_REASON_MESSAGE_BYTES {
                    return Err(inconsistent(
                        "QG-1 incumbent NoDecision reason must be non-empty and bounded".to_owned(),
                    ));
                }
                Ok(())
            }
            (None, None, _) => Err(inconsistent(
                "QG-1 incumbent screen has neither a selected candidate nor a NoDecision reason"
                    .to_owned(),
            )),
        }
    }

    /// Re-derive this screen and decision under the expectations their consumer
    /// retained outside the artifact.
    ///
    /// Every QG-1 component is authenticated separately because pilots and the
    /// decision are issued by separate producer invocations: a set that names
    /// only some of them is an incomplete retention, not a partial success.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::InvalidProvenance`] when any single
    /// component is missing from the retained set, named by a foreign
    /// expectation, or named more than once, and
    /// [`EvidenceArtifactError::InconsistentArtifact`] when the screen or
    /// decision no longer re-derives from its persisted pilots and streams.
    pub fn verify_against_qg1_authorities(
        &self,
        cells: &[EvidenceCell],
        policy: &EvidencePolicy,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        self.validate_shape()?;
        let cell = self.resolved_cell()?;
        let missing = |component: String| EvidenceArtifactError::InvalidProvenance {
            reason: format!(
                "QG-1 incumbent {component} is not named exactly once by the retained \
                 expectation set; replay cannot authenticate it"
            ),
        };
        let rederive =
            |error: Qg1TantivyIncumbentError| EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent screen does not re-derive from its evidence: {error}"
                ),
            };
        for (index, pilot) in self.screen.pilots.iter().enumerate() {
            if resolve_qg1_expected_authority_for_replay(
                external_qg1_authorities,
                &pilot.experiment.config,
            )
            .is_none()
            {
                return Err(missing(format!("pilot stream {index}")));
            }
        }
        if let Some(decision) = self.decision.as_ref() {
            let Some(decision_authority) = resolve_qg1_expected_authority_for_replay(
                external_qg1_authorities,
                &decision.estimator_config,
            ) else {
                return Err(missing("decision stream set".to_owned()));
            };
            self.screen
                .validate_decision_against_qg1_authorities(
                    &cell,
                    &self.semantic_contract,
                    decision,
                    external_qg1_authorities,
                )
                .map_err(rederive)?;
            return self.verify_decision_binds_named_cell(
                decision,
                decision_authority,
                cells,
                policy,
            );
        }
        let recomputed = Qg1TantivyIncumbentScreen::screen_against_qg1_authorities(
            &cell,
            self.screen.screen_plan.clone(),
            &self.semantic_contract,
            self.screen.pilots.clone(),
            external_qg1_authorities,
        )
        .map_err(rederive)?;
        if recomputed != self.screen {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "QG-1 incumbent screen outcome does not recompute from its pilots"
                    .to_owned(),
            });
        }
        Ok(())
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
    /// Exact profile-qualified projection of the complete canonical gate
    /// matrix. Current evidence cannot be assembled or verified from a
    /// class-only identity.
    pub applicability_plan: PerfApplicabilityPlanBinding,
    /// Predeclared evidence-layer thresholds used for every cell.
    pub policy: EvidencePolicy,
    /// Complete run provenance.
    pub provenance: EvidenceProvenance,
    /// Strict runner-receipt machine identity. An explicit unverified binding
    /// is durable for diagnosis but can never promote.
    pub machine_class: MachineClassEvidenceBinding,
    /// Decision-grade cells.
    pub cells: Vec<EvidenceCell>,
    /// Durable QG-1 fastest-incumbent screens, one per required engine cell.
    ///
    /// This is the complete matrix projection, not a single optional screen:
    /// every runnable required QG-1 engine-indexing-lifecycle cell in
    /// [`Self::cells`] must be screened exactly once, tokenizer diagnostic cells
    /// are never screened, and the vector is held in strictly ascending
    /// `cell_id` order so duplicates and a noncanonical persisted shape are
    /// rejected rather than tolerated.
    ///
    /// Empty on every non-QG-1 artifact and skipped on serialization there, so
    /// those artifacts keep their exact persisted bytes and seal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qg1_incumbent_screens: Vec<Qg1IncumbentScreenEvidence>,
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
    fn reconstruct_applicability_plan(
        gate: PerfGate,
        binding: &PerfApplicabilityPlanBinding,
    ) -> Result<(PerfMatrixSpec, PerfApplicabilityPlan), EvidenceArtifactError> {
        if binding.gate != gate {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "applicability plan names gate {} instead of artifact gate {gate}",
                    binding.gate
                ),
            });
        }
        let registry = MachineClassRegistry::frozen().map_err(|error| {
            EvidenceArtifactError::InvalidProvenance {
                reason: format!("frozen machine registry rejected applicability planning: {error}"),
            }
        })?;
        let matrix = PerfMatrixSpec::complete();
        let plan = matrix
            .applicability_plan(&registry, binding.profile, gate)
            .map_err(|error| EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "cannot reconstruct applicability plan for {:?} {gate}: {error}",
                    binding.profile
                ),
            })?;
        if plan.binding != *binding {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "stored applicability-plan binding for {:?} {gate} does not equal the frozen \
                     registry and canonical matrix projection",
                    binding.profile
                ),
            });
        }
        Ok((matrix, plan))
    }

    fn validate_cell_set(
        gate: PerfGate,
        cells: &[EvidenceCell],
        matrix: &PerfMatrixSpec,
        plan: &PerfApplicabilityPlan,
    ) -> Result<BTreeSet<usize>, EvidenceArtifactError> {
        if cells.is_empty() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "an evidence artifact requires at least one cell".to_owned(),
            });
        }
        if plan.binding.gate != gate {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "applicability plan and evidence artifact name different gates".to_owned(),
            });
        }
        let canonical_cells = matrix.for_gate(gate);
        if canonical_cells.len() != plan.cells.len() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: "applicability plan does not classify the complete canonical gate"
                    .to_owned(),
            });
        }
        let mut cell_ids = BTreeSet::new();
        let mut selected_widths = BTreeSet::new();
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

            let mut matching = canonical_cells.iter().enumerate().filter(|(_, canonical)| {
                canonical.fixture == cell.spec.fixture && canonical.metric == cell.spec.metric
            });
            let Some((ordinal, canonical)) = matching.next() else {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "measured cell {} is not in the complete canonical {gate} matrix",
                        cell.cell_id
                    ),
                });
            };
            if matching.next().is_some() {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "measured cell {} resolves ambiguously in the canonical {gate} matrix",
                        cell.cell_id
                    ),
                });
            }
            let classification = plan.cells.get(ordinal).ok_or_else(|| {
                EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "applicability plan omits canonical {gate} cell ordinal {ordinal}"
                    ),
                }
            })?;
            let expected_role = match classification.applicability {
                PerfCellApplicability::Required => EvidenceRole::Required,
                PerfCellApplicability::Diagnostic => EvidenceRole::Diagnostic,
                PerfCellApplicability::NotApplicable => {
                    return Err(EvidenceArtifactError::InconsistentArtifact {
                        reason: format!(
                            "measured cell {} is not applicable to profile {:?}",
                            cell.cell_id, plan.binding.profile
                        ),
                    });
                }
            };
            if cell.spec.role != expected_role {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "measured cell {} has role {:?}; applicability plan requires {:?}",
                        cell.cell_id, cell.spec.role, expected_role
                    ),
                });
            }
            if cell.spec.unit != perf_metric_unit(&canonical.metric) {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "measured cell {} has unit {:?}; canonical metric {} requires {:?}",
                        cell.cell_id,
                        cell.spec.unit,
                        canonical.metric,
                        perf_metric_unit(&canonical.metric)
                    ),
                });
            }
            let requires_concurrency_witness = gate == PerfGate::Qg8
                || (gate == PerfGate::Qg1 && expected_role == EvidenceRole::Required);
            if requires_concurrency_witness
                && cell
                    .spec
                    .concurrency_witness
                    .as_ref()
                    .is_none_or(|witness| {
                        witness.configured_threads != classification.configured_threads
                    })
            {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "measured cell {} concurrency witness does not equal canonical width {}",
                        cell.cell_id, classification.configured_threads
                    ),
                });
            }
            selected_widths.insert(classification.configured_threads);
        }
        Ok(selected_widths)
    }

    fn verify_cell_provenance(&self, cell: &EvidenceCell) -> Result<(), EvidenceArtifactError> {
        let EvidenceCellBody::Paired {
            paired,
            treatment_arm_null,
            qg6_protocol,
            ..
        } = &cell.body
        else {
            return Ok(());
        };
        let expected_scope =
            perf_operation_scope(cell.spec.gate, &cell.spec.fixture, &cell.spec.metric);
        let expected_provenance = crate::PerfSampleProvenance {
            run_id: self.provenance.run_id.clone(),
            executable_sha256: self.provenance.build.executable_sha256.clone(),
            corpus_sha256: self.provenance.corpus.corpus_sha256.clone(),
            input_identity: cell.spec.input_identity.clone(),
            worker_id: self.provenance.machine.fingerprint.clone(),
            build_profile: self.provenance.build.build_profile.clone(),
        };
        if paired.scope != expected_scope || paired.provenance != expected_provenance {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "cell {} operation scope or sample provenance differs from its canonical cell and top-level evidence",
                    cell.cell_id
                ),
            });
        }
        // A QG-1 cell's configuration carries the sealed lifecycle authority
        // its producer issued, which no predeclared template can equal. Its
        // estimator policy must still be exactly predeclared; the authority
        // itself is authenticated separately, against the expectation the
        // consumer retained. Every other gate keeps full equality.
        let predeclared = crate::PairedEstimatorConfig::predeclared(paired.config.bootstrap_seed);
        let policy_matches = if cell.spec.gate == PerfGate::Qg1 {
            paired.config.matches_estimator_policy(&predeclared)
        } else {
            paired.config == predeclared
        };
        if !policy_matches {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "cell {} does not use the exact predeclared estimator configuration",
                    cell.cell_id
                ),
            });
        }
        if cell.spec.gate != PerfGate::Qg1 && treatment_arm_null.is_some() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "only QG-1 cells may carry a treatment-arm A/A null: {}",
                    cell.cell_id
                ),
            });
        }
        if cell.spec.gate == PerfGate::Qg6 && qg6_protocol.is_none() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-6 cell {} is missing formal six-arm protocol evidence",
                    cell.cell_id
                ),
            });
        }
        if cell.spec.gate != PerfGate::Qg6 && qg6_protocol.is_some() {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "only QG-6 cells may carry formal six-arm protocol evidence: {}",
                    cell.cell_id
                ),
            });
        }
        if let Some(treatment_arm_null) = treatment_arm_null {
            if treatment_arm_null.scope != expected_scope
                || treatment_arm_null.provenance != expected_provenance
                || treatment_arm_null.config != paired.config
                || treatment_arm_null.effect != paired.effect
                || treatment_arm_null.effect_samples != paired.effect_samples
            {
                return Err(EvidenceArtifactError::InvalidProvenance {
                    reason: format!(
                        "cell {} treatment-arm null does not share the exact canonical scope, provenance, configuration, and A/B stream",
                        cell.cell_id
                    ),
                });
            }
        }
        if let Some(qg6_protocol) = qg6_protocol {
            validate_qg6_formal_protocol(
                paired,
                qg6_protocol,
                &qg6_protocol.schedule_authority,
                cell.spec.input_identity.as_ref().ok_or_else(|| {
                    EvidenceArtifactError::InvalidProvenance {
                        reason: format!(
                            "cell {} has formal QG-6 rows without an input identity",
                            cell.cell_id
                        ),
                    }
                })?,
                cell.spec.qg6_semantic_contract.as_ref().ok_or_else(|| {
                    EvidenceArtifactError::InvalidProvenance {
                        reason: format!(
                            "cell {} has formal QG-6 rows without a semantic contract",
                            cell.cell_id
                        ),
                    }
                })?,
            )?;
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
        applicability_plan: PerfApplicabilityPlanBinding,
        policy: EvidencePolicy,
        provenance: EvidenceProvenance,
        cells: Vec<EvidenceCell>,
    ) -> Result<Self, EvidenceArtifactError> {
        policy.validate()?;
        provenance.validate()?;
        let (matrix, reconstructed_plan) =
            Self::reconstruct_applicability_plan(gate, &applicability_plan)?;
        let selected_widths = Self::validate_cell_set(gate, &cells, &matrix, &reconstructed_plan)?;
        Self::verify_execution_plan_envelope(
            &provenance.machine.execution,
            &reconstructed_plan,
            &selected_widths,
        )?;
        let admission_no_claim = None;
        let (gate_status, reasons) = Self::fold(gate, &cells, admission_no_claim.as_ref(), &[]);
        Ok(Self {
            schema_version: PERF_EVIDENCE_SCHEMA_VERSION.to_owned(),
            gate,
            applicability_plan,
            policy,
            provenance,
            machine_class: MachineClassEvidenceBinding::unverified(
                "sealed runner receipt has not been bound",
            ),
            cells,
            qg1_incumbent_screens: Vec::new(),
            gate_status,
            gate_decision: None,
            admission_no_claim,
            reasons,
            artifact_sha256: String::new(),
        })
    }

    /// Canonical `cell_id` set this artifact's QG-1 screens must cover exactly.
    ///
    /// Only required engine-indexing-lifecycle cells are screened. Tokenizer
    /// and every other diagnostic cell never freeze an incumbent, so a screen
    /// naming one is an extra screen, not a bonus.
    fn required_qg1_screen_cell_ids(cells: &[EvidenceCell]) -> BTreeSet<String> {
        cells
            .iter()
            .filter(|cell| {
                cell.spec.gate == PerfGate::Qg1 && cell.spec.role == EvidenceRole::Required
            })
            .map(|cell| cell.cell_id.clone())
            .collect()
    }

    /// Whether the screens cover the required engine cells exactly once each,
    /// in canonical order.
    fn qg1_screen_coverage_is_exact(
        cells: &[EvidenceCell],
        screens: &[Qg1IncumbentScreenEvidence],
    ) -> bool {
        screens
            .windows(2)
            .all(|pair| pair[0].cell_id < pair[1].cell_id)
            && screens
                .iter()
                .map(|screen| screen.cell_id.clone())
                .collect::<BTreeSet<_>>()
                == Self::required_qg1_screen_cell_ids(cells)
    }

    /// Attach the complete durable QG-1 incumbent screen projection.
    ///
    /// The screens are pre-binding content: attaching them changes the bytes a
    /// runner receipt would have to cover, so any existing verified binding and
    /// gate decision are discarded exactly as [`Self::force_no_claim`] does, and
    /// the seal is cleared for a fresh one.
    ///
    /// Coverage must be exact — one screen per required engine cell, none for
    /// diagnostic cells — because a partial projection is precisely how a
    /// screened headline could be paired with an unscreened one.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::InconsistentArtifact`] for a non-QG-1
    /// artifact, a screen whose outcome is neither a unique selection nor a
    /// valid `NoDecision`, or a projection that is missing, extra, duplicated, or
    /// out of canonical `cell_id` order.
    pub fn attach_qg1_incumbent_screens(
        &mut self,
        screens: Vec<Qg1IncumbentScreenEvidence>,
    ) -> Result<(), EvidenceArtifactError> {
        if self.gate != PerfGate::Qg1 {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent screens cannot be attached to {} evidence",
                    self.gate
                ),
            });
        }
        for screen in &screens {
            // Width first: an unobservable or unmaterialized frozen width is a
            // defect in its own right, and must be reported as one rather than
            // as whatever downstream shape rule happens to trip next.
            if let Some(named) = self
                .cells
                .iter()
                .find(|candidate| candidate.cell_id == screen.cell_id)
            {
                screen.verify_selected_width_witness(named)?;
            }
            screen.validate_shape()?;
        }
        if !Self::qg1_screen_coverage_is_exact(&self.cells, &screens) {
            return Err(EvidenceArtifactError::InconsistentArtifact {
                reason: format!(
                    "QG-1 incumbent screens must cover every required engine cell exactly once in \
                     canonical order; expected {:?}, observed {:?}",
                    Self::required_qg1_screen_cell_ids(&self.cells),
                    screens
                        .iter()
                        .map(|screen| screen.cell_id.as_str())
                        .collect::<Vec<_>>()
                ),
            });
        }
        if self.machine_class.identity().is_some() {
            self.machine_class = MachineClassEvidenceBinding::unverified(
                "evidence changed after runner binding; a fresh receipt is required",
            );
        }
        self.qg1_incumbent_screens = screens;
        self.gate_decision = None;
        (self.gate_status, self.reasons) = Self::fold(
            self.gate,
            &self.cells,
            self.admission_no_claim.as_ref(),
            &self.qg1_incumbent_screens,
        );
        self.artifact_sha256.clear();
        Ok(())
    }

    /// Deterministic severity-precedence fold of required cells.
    fn fold(
        gate: PerfGate,
        cells: &[EvidenceCell],
        admission_no_claim: Option<&EvidenceReason>,
        qg1_incumbent_screens: &[Qg1IncumbentScreenEvidence],
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
        // Incomplete coverage is the same refusal as an incomplete screen, and
        // must be: omitting a screen — or screening only some required engine
        // cells — would otherwise be the cheapest way to ratchet without ever
        // naming an incumbent. Other gates never screen and are untouched.
        if gate == PerfGate::Qg1
            && any_required
            && !Self::qg1_screen_coverage_is_exact(cells, qg1_incumbent_screens)
        {
            any_no_decision = true;
            reasons.push(EvidenceReason::new(
                "evidence.qg1_incumbent_screen_missing",
                "QG-1 evidence does not screen every required engine cell exactly once",
                EvidenceSeverity::NoClaim,
            ));
        }
        // One incomplete screen folds the whole gate to no-claim: the cells are
        // a single projection, so a headline anywhere in it needs every engine
        // cell to have frozen its incumbent.
        for screen in qg1_incumbent_screens
            .iter()
            .filter(|screen| !screen.has_selection())
        {
            any_no_decision = true;
            reasons.push(EvidenceReason::new(
                "evidence.qg1_incumbent_screen_no_decision",
                screen.screen.no_decision_reason.as_deref().map_or_else(
                    || {
                        format!(
                            "QG-1 incumbent screen for {} selected no candidate",
                            screen.cell_id
                        )
                    },
                    |reason| {
                        format!(
                            "QG-1 incumbent screen for {} made no selection: {reason}",
                            screen.cell_id
                        )
                    },
                ),
                EvidenceSeverity::NoClaim,
            ));
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

    fn has_exact_runnable_plan_coverage(&self) -> bool {
        let Ok((matrix, plan)) =
            Self::reconstruct_applicability_plan(self.gate, &self.applicability_plan)
        else {
            return false;
        };
        if Self::validate_cell_set(self.gate, &self.cells, &matrix, &plan).is_err() {
            return false;
        }
        let expected = matrix
            .for_gate(self.gate)
            .into_iter()
            .zip(&plan.cells)
            .filter(|(_, classification)| classification.applicability.is_runnable())
            .map(|(cell, _)| format!("{}/{}/{}", self.gate, cell.fixture, cell.metric))
            .collect::<BTreeSet<_>>();
        let measured = self
            .cells
            .iter()
            .map(|cell| cell.cell_id.clone())
            .collect::<BTreeSet<_>>();
        self.cells.len() == measured.len() && measured == expected
    }

    fn verify_runner_plan_envelope(
        identity: &VerifiedRunnerIdentity,
        plan: &PerfApplicabilityPlan,
    ) -> Result<(), EvidenceArtifactError> {
        if identity.profile() != plan.binding.profile
            || identity.capacity_semantics() != plan.capacity_semantics
            || plan.execution_capacity != Some(identity.execution_capacity())
            || plan.max_exercised_cell_width != Some(identity.max_exercised_cell_width())
            || identity
                .artifact_manifest()
                .is_some_and(|binding| binding.manifest().applicability_plan() != plan.binding())
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "verified runner profile/capacity/maximum envelope does not equal \
                     applicability plan {:?} {}",
                    plan.binding.profile, plan.binding.gate
                ),
            });
        }
        Ok(())
    }

    fn verify_execution_plan_envelope(
        execution: &PerfExecutionProvenance,
        plan: &PerfApplicabilityPlan,
        selected_widths: &BTreeSet<usize>,
    ) -> Result<(), EvidenceArtifactError> {
        if plan.execution_capacity != Some(execution.execution_capacity)
            || plan.max_exercised_cell_width != Some(execution.max_exercised_cell_width)
            || !execution.matches_capacity_semantics(plan.capacity_semantics)
            || execution.configured_engine_thread_widths
                != selected_widths.iter().copied().collect::<Vec<_>>()
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "execution provenance capacity/maximum/selected-width envelope does not \
                     equal applicability plan {:?} {}",
                    plan.binding.profile, plan.binding.gate
                ),
            });
        }
        Ok(())
    }

    /// Whether this artifact may establish or move a ratchet baseline.
    ///
    /// Invalid runs remain durable but can never ratchet.
    #[must_use]
    pub fn ratchet_admissible(&self) -> bool {
        self.gate_status == EvidenceDecisionStatus::MeasuredProvisional
            // For QG-1 every required engine cell must have frozen an incumbent
            // and carry its decision; anything less is not admissible evidence
            // to ratchet on. Every other gate keeps its exact prior behaviour.
            && (self.gate != PerfGate::Qg1
                || (Self::qg1_screen_coverage_is_exact(&self.cells, &self.qg1_incumbent_screens)
                    && self
                        .qg1_incumbent_screens
                        .iter()
                        .all(|screen| screen.has_selection() && screen.decision.is_some())))
            && self.has_exact_runnable_plan_coverage()
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

    fn verify_runner_identity_projection(
        &self,
        identity: &VerifiedRunnerIdentity,
    ) -> Result<(), EvidenceArtifactError> {
        let invalid = |reason: String| EvidenceArtifactError::InvalidProvenance { reason };
        let build = identity
            .build()
            .as_object()
            .ok_or_else(|| invalid("verified runner build facts are not an object".to_owned()))?;
        let runner_string = |field: &str| {
            build
                .get(field)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    invalid(format!(
                        "verified runner build field {field:?} is not a string"
                    ))
                })
        };
        let runner_git_dirty = build
            .get("git_dirty")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                invalid("verified runner build field \"git_dirty\" is not a boolean".to_owned())
            })?;
        let runner_worktree_state = match build.get("worktree_state_sha256") {
            Some(serde_json::Value::String(value)) => Some(value.as_str()),
            Some(serde_json::Value::Null) => None,
            _ => {
                return Err(invalid(
                    "verified runner worktree-state identity is malformed".to_owned(),
                ));
            }
        };
        let evidence_build = &self.provenance.build;
        // These are the build facts the typed runner captures independently.
        // The remaining BuildIdentity display/context fields stay sealed and
        // attributable to this exact source/ELF, but are not duplicated as
        // direct receipt facts.
        let build_matches = evidence_build.git_revision == runner_string("git_revision")?
            && evidence_build.git_dirty == runner_git_dirty
            && evidence_build.worktree_state_sha256.as_deref() == runner_worktree_state
            && evidence_build.cargo_lock_sha256.as_deref()
                == Some(runner_string("cargo_lock_sha256")?)
            && evidence_build.executable_sha256 == runner_string("executable_sha256")?
            && evidence_build.command_sha256 == runner_string("command_sha256")?
            && evidence_build.environment_sha256.as_deref()
                == Some(runner_string("environment_sha256")?);
        if !build_matches {
            return Err(invalid(
                "evidence build identity differs from the verified runner receipt".to_owned(),
            ));
        }

        let hardware = identity.hardware().as_object().ok_or_else(|| {
            invalid("verified runner hardware facts are not an object".to_owned())
        })?;
        let hardware_string = |field: &str| {
            hardware
                .get(field)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    invalid(format!(
                        "verified runner hardware field {field:?} is not a string"
                    ))
                })
        };
        let hardware_usize = |field: &str| {
            hardware
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    invalid(format!(
                        "verified runner hardware field {field:?} is not a host-sized integer"
                    ))
                })
        };
        let hardware_os = hardware_string("os")?;
        let hardware_arch = hardware_string("arch")?;
        let hardware_physical = hardware_usize("physical_cores")?;
        let hardware_logical = hardware_usize("logical_cpus")?;
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
        let machine = &self.provenance.machine;
        let execution = &machine.execution;
        let serialized_isa = serde_json::to_value(&execution.runtime_detected_isa)?;
        if machine.os != hardware_os
            || machine.arch != hardware_arch
            || execution.producer_os.as_str() != hardware_os
            || execution.physical_cores != hardware_physical
            || execution.logical_threads != hardware_logical
            || hardware.get("runtime_detected_isa") != Some(&serialized_isa)
            || !target_matches
        {
            return Err(invalid(
                "evidence OS, architecture, topology, runtime ISA, or target differs from the verified runner hardware"
                    .to_owned(),
            ));
        }

        let observed_cpu_ids = identity
            .execution_start()
            .get("observed_logical_cpu_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                invalid("verified runner execution start omits observed logical CPU IDs".to_owned())
            })?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    invalid("verified runner observed CPU ID is not an integer".to_owned())
                })
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let expected_process_threads = if observed_cpu_ids.is_empty() {
            hardware_logical
        } else {
            observed_cpu_ids.len()
        };
        if execution.process_available_threads != expected_process_threads {
            return Err(invalid(
                "evidence process-available topology differs from the verified runner execution snapshot"
                    .to_owned(),
            ));
        }

        let allowed_threads = match hardware_os {
            "linux" => {
                let projected = execution
                    .cpu_affinity_allowed_list
                    .as_deref()
                    .and_then(parse_cpu_list_ids)
                    .ok_or_else(|| {
                        invalid(
                            "Linux evidence requires a valid exact CPU-affinity projection"
                                .to_owned(),
                        )
                    })?;
                if projected != observed_cpu_ids {
                    return Err(invalid(
                        "evidence CPU-affinity projection differs from the verified runner execution snapshot"
                            .to_owned(),
                    ));
                }
                let runner_governor = identity
                    .execution_start()
                    .get("governor")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        invalid("verified runner governor is not a string".to_owned())
                    })?;
                if machine.cpu_governor.as_deref() != Some(runner_governor) {
                    return Err(invalid(
                        "evidence CPU governor differs from the verified runner execution snapshot"
                            .to_owned(),
                    ));
                }
                Some(projected.len())
            }
            "macos" => {
                if execution.cpu_affinity_allowed_list.is_some()
                    || machine.cpu_governor.is_some()
                    || !observed_cpu_ids.is_empty()
                {
                    return Err(invalid(
                        "macOS scheduler evidence cannot fabricate Linux affinity or governor projections"
                            .to_owned(),
                    ));
                }
                None
            }
            unsupported => {
                return Err(invalid(format!(
                    "verified runner OS {unsupported:?} is not supported for execution-topology evidence"
                )));
            }
        };
        let expected_cap = if let Some(allowed_threads) =
            allowed_threads.filter(|count| *count < hardware_logical)
        {
            let allowed_list = execution
                .cpu_affinity_allowed_list
                .as_deref()
                .ok_or_else(|| {
                    invalid(
                        "verified Linux affinity count is missing its serialized CPU list"
                            .to_owned(),
                    )
                })?;
            Some(format!(
                "Cpus_allowed_list={} ({} of {} host logical threads)",
                allowed_list, allowed_threads, hardware_logical,
            ))
        } else if execution.process_available_threads < hardware_logical {
            Some(format!(
                "available_parallelism={} of {} host logical threads",
                execution.process_available_threads, hardware_logical,
            ))
        } else {
            None
        };
        if execution.affinity_or_cpuset_cap != expected_cap {
            return Err(invalid(
                "evidence affinity/cpuset cap text is not the deterministic projection of verified topology"
                    .to_owned(),
            ));
        }
        let cpu_label = if hardware_os == "linux" {
            hardware_string("cpu_model_name")?.replace(['/', ' '], "_")
        } else {
            "unknown-cpu".to_owned()
        };
        // The hostname component is benchmark-reported rather than an
        // independent receipt fact. This equality is therefore an internal
        // fingerprint-consistency check; promotion authority comes from the
        // receipt-projected class, hardware, topology, ISA, and execution
        // envelope checked above.
        let expected_fingerprint = format!(
            "{hardware_os}-{hardware_arch}-{}-{hardware_logical}thread-{cpu_label}",
            execution.host_identity
        );
        if machine.fingerprint != expected_fingerprint {
            return Err(invalid(
                "evidence machine fingerprint is not the deterministic projection of verified hardware and host identity"
                    .to_owned(),
            ));
        }
        Ok(())
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
        self.bind_machine_class_identity_against_authorities(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
            &[],
            &[],
        )
    }

    /// Bind a runner identity to evidence whose QG-1 components are
    /// authenticated against the expectations their consumer retained.
    ///
    /// Binding re-parses the exact pre-binding bytes, and that parse is a
    /// replay: QG-1 evidence cannot be re-admitted there without the retained
    /// set, so honest QG-1 evidence can only be bound through this entry.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::bind_machine_class_identity`],
    /// including the fail-closed refusal of an incomplete retained set.
    pub fn bind_machine_class_identity_against_qg1_authorities(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        self.bind_machine_class_identity_against_authorities(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
            external_qg1_authorities,
            &[],
        )
    }

    /// Bind the exact runner identity after replaying all QG-1 and QG-6
    /// components against independently retained authority sets.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::bind_machine_class_identity`],
    /// including fail-closed authority selection failures.
    pub fn bind_machine_class_identity_against_authorities(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        let source = Self::from_verified_slice_against_authorities(
            prebinding_evidence_bytes,
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
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
        let (_, reconstructed_plan) =
            Self::reconstruct_applicability_plan(self.gate, &self.applicability_plan)?;
        Self::verify_runner_plan_envelope(&identity, &reconstructed_plan)?;
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
            || artifact_manifest.applicability_plan() != &self.applicability_plan
            || self.provenance.manifest_sha256
                != self.applicability_plan.normalized_perf_manifest_sha256
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "runner artifact manifest names a different gate, run, manifest, or applicability-plan identity"
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
        let expected_destination =
            identity
                .profile()
                .latest_basename(gate_label)
                .map_err(|error| EvidenceArtifactError::InvalidProvenance {
                    reason: format!("machine-profile destination rejected: {error}"),
                })?;
        if context.gate != gate_label || context.destination_basename != expected_destination {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: format!(
                    "machine-class receipt was admitted for gate/destination {}/{} instead of \
                     {gate_label}/{expected_destination}",
                    context.gate, context.destination_basename
                ),
            });
        }
        self.verify_runner_identity_projection(&identity)?;
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
        self.bind_machine_class_identity_and_seal_against_authorities(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
            &[],
            &[],
        )
    }

    /// Bind and seal evidence whose QG-1 components are authenticated against
    /// the expectations their consumer retained.
    ///
    /// The newly sealed bytes are re-verified under the same retained set
    /// before they are returned, so a seal can never be handed back for an
    /// object that its own replay entry would refuse.
    ///
    /// # Errors
    ///
    /// Returns the same failures as
    /// [`Self::bind_machine_class_identity_and_seal`].
    pub fn bind_machine_class_identity_and_seal_against_qg1_authorities(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Vec<u8>, EvidenceArtifactError> {
        self.bind_machine_class_identity_and_seal_against_authorities(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
            external_qg1_authorities,
            &[],
        )
    }

    /// Bind and seal evidence after replaying every authority-bound component
    /// against independently retained QG-1 and QG-6 authority sets.
    ///
    /// # Errors
    ///
    /// Returns the same failures as
    /// [`Self::bind_machine_class_identity_and_seal`].
    pub fn bind_machine_class_identity_and_seal_against_authorities(
        &mut self,
        identity: VerifiedRunnerIdentity,
        threshold_artifact_bytes: &[u8],
        prebinding_evidence_bytes: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<Vec<u8>, EvidenceArtifactError> {
        let mut bound = self.clone();
        bound.bind_machine_class_identity_against_authorities(
            identity,
            threshold_artifact_bytes,
            prebinding_evidence_bytes,
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
        bound.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&bound)?;
        bound.artifact_sha256 = lower_hex(&Sha256::digest(unsealed.as_bytes()));
        let sealed = serde_json::to_vec_pretty(&bound)?;
        bound.verify_integrity_against_authorities(
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
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
        (self.gate_status, self.reasons) = Self::fold(
            self.gate,
            &self.cells,
            self.admission_no_claim.as_ref(),
            &self.qg1_incumbent_screens,
        );
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

    pub(crate) fn reconstructed_prebinding_bytes(&self) -> Result<Vec<u8>, EvidenceArtifactError> {
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
        self.verify_integrity_against_authorities(&[], &[])
    }

    /// Verify this artifact against the QG-1 expectations its consumer
    /// retained outside it.
    ///
    /// This is the replay entry for artifacts containing headline QG-1 cells.
    /// Each paired cell selects the single retained expectation that issued
    /// its sealed authority; cells that name no QG-1 authority are unaffected,
    /// and an empty set is exactly [`Self::verify_integrity`].
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::verify_integrity`], including the
    /// fail-closed refusal of a QG-1 cell whose retained expectation is absent.
    pub fn verify_integrity_against_qg1_authorities(
        &self,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        self.verify_integrity_against_authorities(external_qg1_authorities, &[])
    }

    /// Verify this artifact against every independently retained authority
    /// required by its cells.
    ///
    /// QG-1 authorities prove producer-issued throughput expectations. QG-6
    /// authorities prove schedules frozen before timing. Serialized authority
    /// receipts remain replay inputs, never self-authenticating capabilities.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::verify_integrity`], including the
    /// fail-closed refusal of any authority-bound cell whose unique external
    /// authority is absent or substituted.
    pub fn verify_integrity_against_authorities(
        &self,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<(), EvidenceArtifactError> {
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
        let (matrix, reconstructed_plan) =
            Self::reconstruct_applicability_plan(self.gate, &self.applicability_plan)?;
        if self.provenance.manifest_sha256
            != reconstructed_plan.binding.normalized_perf_manifest_sha256
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason:
                    "evidence manifest digest differs from its reconstructed applicability plan"
                        .to_owned(),
            });
        }
        let selected_widths =
            Self::validate_cell_set(self.gate, &self.cells, &matrix, &reconstructed_plan)?;
        Self::verify_execution_plan_envelope(
            &self.provenance.machine.execution,
            &reconstructed_plan,
            &selected_widths,
        )?;
        self.machine_class.validate().map_err(|error| {
            EvidenceArtifactError::InvalidProvenance {
                reason: format!("machine-class binding rejected: {error}"),
            }
        })?;
        if let Some(identity) = self.machine_class.identity() {
            Self::verify_runner_plan_envelope(identity, &reconstructed_plan)?;
            self.verify_runner_identity_projection(identity)?;
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
                || manifest.applicability_plan() != reconstructed_plan.binding()
            {
                return Err(EvidenceArtifactError::InvalidProvenance {
                    reason: "bound artifact manifest names a different gate, run, or applicability-plan identity"
                        .to_owned(),
                });
            }
        }
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
            self.verify_cell_provenance(cell)?;
            cell.verify_recomputed_against_authorities(
                &self.policy,
                external_qg1_authorities,
                external_qg6_authorities,
            )?;
        }
        if !self.qg1_incumbent_screens.is_empty() {
            if self.gate != PerfGate::Qg1 {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "{} evidence carries QG-1 incumbent screens it can never have measured",
                        self.gate
                    ),
                });
            }
            // Coverage is verified here as well as at attachment, because these
            // bytes may have been produced anywhere: a persisted artifact that
            // dropped or duplicated one screen must not verify.
            if !Self::qg1_screen_coverage_is_exact(&self.cells, &self.qg1_incumbent_screens) {
                return Err(EvidenceArtifactError::InconsistentArtifact {
                    reason: format!(
                        "persisted QG-1 incumbent screens do not cover every required engine cell \
                         exactly once in canonical order; expected {:?}",
                        Self::required_qg1_screen_cell_ids(&self.cells)
                    ),
                });
            }
            for screen in &self.qg1_incumbent_screens {
                screen.verify_against_qg1_authorities(
                    &self.cells,
                    &self.policy,
                    external_qg1_authorities,
                )?;
            }
        }
        let (expected_status, expected_reasons) = Self::fold(
            self.gate,
            &self.cells,
            self.admission_no_claim.as_ref(),
            &self.qg1_incumbent_screens,
        );
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
        if !self.qg1_incumbent_screens.is_empty()
            || self.cells.iter().any(|cell| {
                matches!(
                    &cell.body,
                    EvidenceCellBody::Paired {
                        qg6_protocol: Some(_),
                        ..
                    }
                )
            })
        {
            return Err(EvidenceArtifactError::InvalidProvenance {
                reason: "authority-bound evidence may only be persisted through \
                         write_atomic_against_authorities with its complete retained sets"
                    .to_owned(),
            });
        }
        self.write_atomic_unchecked(output_dir)
    }

    /// Persist evidence whose QG-1 components are proven against the
    /// expectations their consumer retained.
    ///
    /// Writing is where evidence stops being a live object and becomes
    /// something a later process must replay, so the complete retained set is
    /// required here rather than only at load: a screen-bearing artifact that
    /// no retained set can authenticate must never reach disk looking intact.
    ///
    /// # Errors
    ///
    /// Returns every failure [`Self::verify_integrity_against_qg1_authorities`]
    /// can, plus typed serialization and I/O errors.
    pub fn write_atomic_against_qg1_authorities(
        &self,
        output_dir: &Path,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<EvidenceArtifactPaths, EvidenceArtifactError> {
        self.write_atomic_against_authorities(output_dir, external_qg1_authorities, &[])
    }

    /// Persist authority-bound evidence after replaying every QG-1 and QG-6
    /// component against independently retained authority sets.
    ///
    /// # Errors
    ///
    /// Returns every failure [`Self::verify_integrity_against_authorities`]
    /// can, plus typed serialization and I/O errors.
    pub fn write_atomic_against_authorities(
        &self,
        output_dir: &Path,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<EvidenceArtifactPaths, EvidenceArtifactError> {
        // Verify the sealed form, which is exactly what reaches disk. Checking
        // the in-memory copy instead would refuse an artifact whose seal is
        // legitimately pending, and would prove nothing about the bytes.
        let mut sealed = self.clone();
        sealed.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&sealed)?;
        sealed.artifact_sha256 = lower_hex(&Sha256::digest(unsealed.as_bytes()));
        sealed.verify_integrity_against_authorities(
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
        sealed.write_atomic_unchecked(output_dir)
    }

    fn write_atomic_unchecked(
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
        Self::from_verified_slice_against_authorities(contents, &[], &[])
    }

    /// Parse exact artifact bytes and verify them against the QG-1
    /// expectations their consumer retained outside the artifact.
    ///
    /// Strict canonical parsing is identical; only the integrity pass differs,
    /// so a persisted QG-1 artifact is verifiable exactly once its retained
    /// authority set is supplied. An empty set is [`Self::from_verified_slice`],
    /// under which a QG-1 cell still fails closed and non-QG-1 evidence is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each defect class.
    pub fn from_verified_slice_against_qg1_authorities(
        contents: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
        Self::from_verified_slice_against_authorities(contents, external_qg1_authorities, &[])
    }

    /// Parse exact artifact bytes and verify all QG-1 and QG-6 authority-bound
    /// components against independently retained authority sets.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each syntax, seal,
    /// replay, or authority mismatch.
    pub fn from_verified_slice_against_authorities(
        contents: &[u8],
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
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
        if serde_json::to_vec_pretty(&artifact)? != contents {
            return Err(EvidenceArtifactError::Malformed {
                reason: "artifact bytes are not exact canonical pretty JSON".to_owned(),
            });
        }
        artifact.verify_integrity_against_authorities(
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
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
        Self::load_verified_against_authorities(path, &[], &[])
    }

    /// Load one artifact and verify it against the QG-1 expectations its
    /// consumer retained outside the artifact.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each defect class.
    pub fn load_verified_against_qg1_authorities(
        path: &Path,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
        Self::load_verified_against_authorities(path, external_qg1_authorities, &[])
    }

    /// Load one artifact and verify all authority-bound components against
    /// independently retained QG-1 and QG-6 authority sets.
    ///
    /// # Errors
    ///
    /// Returns the specific [`EvidenceArtifactError`] for each defect class.
    pub fn load_verified_against_authorities(
        path: &Path,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
        let contents = read_evidence_artifact_bounded(path)?;
        Self::from_verified_slice_against_authorities(
            &contents,
            external_qg1_authorities,
            external_qg6_authorities,
        )
    }

    /// Load one artifact as ADMISSIBLE EVIDENCE: verified exactly as
    /// [`Self::load_verified`] does, then screened against the append-only
    /// quarantine register.
    ///
    /// [`Self::load_verified`] answers "are these bytes an intact artifact?".
    /// That is a strictly weaker question than "may this artifact support a
    /// claim?", and the difference is the entire point of a quarantine: the
    /// structurally invalid sweep is intact, seals correctly, and recomputes
    /// from its own samples. Every consumer that treats an artifact as evidence
    /// must come through here; [`Self::load_verified`] remains available for
    /// diagnosis and for reading history, which stays readable forever.
    ///
    /// # Errors
    ///
    /// Returns every [`EvidenceArtifactError`] [`Self::load_verified`] can, plus
    /// [`EvidenceArtifactError::QuarantinedRevision`] when the measured
    /// revision is registered as structurally invalid.
    pub fn load_admissible_evidence(
        path: &Path,
        register: &PerfQuarantineRegister,
    ) -> Result<Self, EvidenceArtifactError> {
        Self::load_admissible_evidence_against_authorities(path, register, &[], &[])
    }

    /// Load admissible evidence whose QG-1 cells are authenticated against the
    /// expectations their consumer retained.
    ///
    /// # Errors
    ///
    /// Returns every error [`Self::load_admissible_evidence`] can.
    pub fn load_admissible_evidence_against_qg1_authorities(
        path: &Path,
        register: &PerfQuarantineRegister,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
        Self::load_admissible_evidence_against_authorities(
            path,
            register,
            external_qg1_authorities,
            &[],
        )
    }

    /// Load admissible evidence after replaying every QG-1 and QG-6 component
    /// against independently retained authority sets.
    ///
    /// # Errors
    ///
    /// Returns every error [`Self::load_admissible_evidence`] can.
    pub fn load_admissible_evidence_against_authorities(
        path: &Path,
        register: &PerfQuarantineRegister,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<Self, EvidenceArtifactError> {
        let artifact = Self::load_verified_against_authorities(
            path,
            external_qg1_authorities,
            external_qg6_authorities,
        )?;
        register.screen(&artifact)?;
        Ok(artifact)
    }
}

/// Schema tag carried by every append-only quarantine record.
pub const PERF_QUARANTINE_SCHEMA_VERSION: &str = "quill-perf-quarantine-v1";

/// File name of the append-only quarantine register inside `.bench-history`.
pub const PERF_QUARANTINE_FILE_NAME: &str = "QUARANTINE.jsonl";

/// Shortest revision prefix a quarantine record may carry.
///
/// A prefix shorter than this could collide with an unrelated revision and
/// quarantine evidence that was never in the invalid sweep, so it is rejected
/// when the register is parsed rather than silently over-matching at screen
/// time.
const MIN_QUARANTINE_REVISION_PREFIX: usize = 7;

/// One append-only record marking a measured revision structurally invalid.
///
/// Records are never edited or removed. Retracting a quarantine means appending
/// a later record through whatever supersession process the ledger defines, not
/// rewriting this file — the same append-only discipline the evidence ledgers
/// themselves follow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfQuarantineRecord {
    /// Always [`PERF_QUARANTINE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Lowercase hex git-revision prefix identifying the quarantined sweep.
    pub git_revision_prefix: String,
    /// Why the sweep cannot support a claim.
    pub reason: String,
    /// Tracker identifier that recorded the quarantine.
    pub recorded_by: String,
}

/// Append-only register of structurally invalid measured revisions.
///
/// The register is deliberately OUT OF BAND. `PerfEvidenceArtifact` already
/// carries an in-band [`PerfEvidenceArtifact::admission_no_claim`], but setting
/// it on a historical artifact would mean rewriting sealed evidence that has
/// already been published — precisely the deletion-and-reinterpretation this
/// correction forbids. Keying the quarantine on immutable identity instead lets
/// history stay byte-identical and still stop supporting claims.
#[derive(Clone, Debug, Default)]
pub struct PerfQuarantineRegister {
    records: Vec<PerfQuarantineRecord>,
}

impl PerfQuarantineRegister {
    /// Parse an append-only JSONL register.
    ///
    /// Blank lines and `#` comment lines are ignored so the file can carry
    /// human context; every other line must be a complete record.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::Malformed`] for a line that is not a
    /// record, carries a foreign schema tag, or names a revision prefix shorter
    /// than [`MIN_QUARANTINE_REVISION_PREFIX`] or containing non-hex bytes.
    pub fn from_jsonl(contents: &str) -> Result<Self, EvidenceArtifactError> {
        let mut records = Vec::new();
        for (index, line) in contents.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let record: PerfQuarantineRecord = serde_json::from_str(trimmed).map_err(|error| {
                EvidenceArtifactError::Malformed {
                    reason: format!(
                        "quarantine register line {} is not a record: {error}",
                        index + 1
                    ),
                }
            })?;
            if record.schema_version != PERF_QUARANTINE_SCHEMA_VERSION {
                return Err(EvidenceArtifactError::Malformed {
                    reason: format!(
                        "quarantine register line {} carries schema {}; current is {PERF_QUARANTINE_SCHEMA_VERSION}",
                        index + 1,
                        record.schema_version
                    ),
                });
            }
            let prefix = record.git_revision_prefix.trim().to_ascii_lowercase();
            if prefix.len() < MIN_QUARANTINE_REVISION_PREFIX
                || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(EvidenceArtifactError::Malformed {
                    reason: format!(
                        "quarantine register line {} names revision prefix {:?}, which must be at \
                         least {MIN_QUARANTINE_REVISION_PREFIX} lowercase hex characters",
                        index + 1,
                        record.git_revision_prefix
                    ),
                });
            }
            records.push(PerfQuarantineRecord {
                git_revision_prefix: prefix,
                ..record
            });
        }
        Ok(Self { records })
    }

    /// Load the append-only register from a path.
    ///
    /// A MISSING register is an empty register, not an error: a checkout with
    /// no quarantined revisions is a legitimate state. A present-but-unreadable
    /// or malformed register is an error, so a corrupted file can never be
    /// mistaken for "nothing is quarantined".
    ///
    /// # Errors
    ///
    /// Returns the I/O or parse error for a register that exists but cannot be
    /// read as a valid append-only register.
    pub fn load(path: &Path) -> Result<Self, EvidenceArtifactError> {
        match fs::read_to_string(path) {
            Ok(contents) => Self::from_jsonl(&contents),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(EvidenceArtifactError::Io(error)),
        }
    }

    /// Load the register that belongs to a `.bench-history`-style directory.
    ///
    /// Callers own the history directory, not the register's file name, so this
    /// is the call the promotion path wants: it keeps
    /// [`PERF_QUARANTINE_FILE_NAME`] an implementation detail of this module
    /// instead of a string every call site has to repeat and keep in sync.
    ///
    /// # Errors
    ///
    /// Same contract as [`Self::load`]: an absent register is empty, a present
    /// but unreadable or malformed one is an error.
    pub fn load_from_history_dir(history_dir: &Path) -> Result<Self, EvidenceArtifactError> {
        Self::load(&history_dir.join(PERF_QUARANTINE_FILE_NAME))
    }

    /// The record quarantining `git_revision`, when one exists.
    #[must_use]
    pub fn quarantine_of(&self, git_revision: &str) -> Option<&PerfQuarantineRecord> {
        let revision = git_revision.trim().to_ascii_lowercase();
        self.records
            .iter()
            .find(|record| revision.starts_with(&record.git_revision_prefix))
    }

    /// Every record in append order.
    #[must_use]
    pub fn records(&self) -> &[PerfQuarantineRecord] {
        &self.records
    }

    /// Refuse an artifact whose measured revision is quarantined.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceArtifactError::QuarantinedRevision`] naming the
    /// artifact revision, the matched record, and its reason.
    pub fn screen(&self, artifact: &PerfEvidenceArtifact) -> Result<(), EvidenceArtifactError> {
        let revision = &artifact.provenance.build.git_revision;
        if let Some(record) = self.quarantine_of(revision) {
            return Err(EvidenceArtifactError::QuarantinedRevision {
                git_revision: revision.clone(),
                git_revision_prefix: record.git_revision_prefix.clone(),
                reason: record.reason.clone(),
                recorded_by: record.recorded_by.clone(),
            });
        }
        Ok(())
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
fn read_evidence_artifact_bounded(path: &Path) -> Result<Vec<u8>, EvidenceArtifactError> {
    let file = File::open(path)?;
    let declared_len = file.metadata()?.len();
    if declared_len
        > u64::try_from(PERF_EVIDENCE_MAX_ARTIFACT_BYTES).expect("evidence byte cap fits u64")
    {
        return Err(EvidenceArtifactError::UnboundedArtifactBytes {
            count: declared_len,
            max: PERF_EVIDENCE_MAX_ARTIFACT_BYTES,
        });
    }

    let mut contents = Vec::new();
    contents
        .try_reserve_exact(
            usize::try_from(declared_len).unwrap_or(PERF_EVIDENCE_MAX_ARTIFACT_BYTES),
        )
        .map_err(|error| EvidenceArtifactError::Malformed {
            reason: format!("cannot reserve bounded evidence artifact buffer: {error}"),
        })?;
    file.take(
        u64::try_from(PERF_EVIDENCE_MAX_ARTIFACT_BYTES).expect("evidence byte cap fits u64") + 1,
    )
    .read_to_end(&mut contents)?;
    if contents.len() > PERF_EVIDENCE_MAX_ARTIFACT_BYTES {
        return Err(EvidenceArtifactError::UnboundedArtifactBytes {
            count: u64::try_from(contents.len()).unwrap_or(u64::MAX),
            max: PERF_EVIDENCE_MAX_ARTIFACT_BYTES,
        });
    }
    Ok(contents)
}

#[derive(Debug, Error)]
pub enum EvidenceArtifactError {
    /// The artifact bytes are not a parseable artifact at all.
    #[error("evidence artifact is malformed: {reason}")]
    Malformed {
        /// Bounded parse failure description.
        reason: String,
    },
    /// The artifact carries a non-current schema version.
    #[error("evidence artifact schema is {found}; current is quill-perf-evidence-v7")]
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
    /// The serialized evidence file exceeds the public loader's memory bound.
    #[error("evidence artifact retains {count} bytes; bound is {max}")]
    UnboundedArtifactBytes {
        /// Observed file or bounded-read byte count.
        count: u64,
        /// Maximum admitted serialized byte count.
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
    /// The artifact is intact but its measured revision is registered as
    /// structurally invalid, so it can never support a claim.
    ///
    /// This is deliberately distinct from every "broken artifact" variant: the
    /// bytes verify, the seal matches, and the summaries recompute. What fails
    /// is admissibility, not integrity, and conflating the two would let an
    /// operator read this as corruption and "fix" it by re-measuring the same
    /// invalid shape.
    #[error(
        "evidence artifact revision {git_revision} is quarantined by {recorded_by} \
         (prefix {git_revision_prefix}): {reason}"
    )]
    QuarantinedRevision {
        /// Measured revision recorded in the artifact.
        git_revision: String,
        /// Register prefix that matched it.
        git_revision_prefix: String,
        /// Why the sweep cannot support a claim.
        reason: String,
        /// Tracker identifier that recorded the quarantine.
        recorded_by: String,
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
    use crate::perf::{PerfSampleOrder, Qg6SampleBinding};
    use crate::qg6_prepared::{
        Qg6Comparison, Qg6ExperimentIdentity, Qg6RankedHitReceipt, Qg6ResultReceipt,
        Qg6SampleOrder, Qg6SearchTimingLeafReceipt, Qg6SixArmResultReceipts, Qg6TimedSample,
        query_manifest_sha256,
    };

    pub fn verified_terminal_artifact() -> (PerfEvidenceArtifact, Vec<u8>, Qg6ScheduleAuthority) {
        let (artifact, authority) = super::tests::qg6_artifact();
        let bytes = artifact
            .sealed_json()
            .expect("sealed canonical terminal QG-6 artifact")
            .into_bytes();
        let verified = PerfEvidenceArtifact::from_verified_slice_against_authorities(
            &bytes,
            &[],
            &[&authority],
        )
        .expect("authority-verified terminal QG-6 artifact");
        (verified, bytes, authority)
    }

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
        contract_for_hit_count(query_class, document_count, k, 1)
    }

    pub fn contract_for_full_top_k(
        query_class: crate::PerfQueryClass,
        document_count: u64,
        k: usize,
    ) -> (PerfInputIdentity, Qg6SemanticContract) {
        let hit_count = k.min(usize::try_from(document_count).unwrap_or(usize::MAX));
        contract_for_hit_count(query_class, document_count, k, hit_count)
    }

    fn contract_for_hit_count(
        query_class: crate::PerfQueryClass,
        document_count: u64,
        k: usize,
        hit_count: usize,
    ) -> (PerfInputIdentity, Qg6SemanticContract) {
        let queries =
            Qg6QuerySpec::normative_for_class(query_class).expect("frozen QG-6 query slice");
        let receipts = queries
            .iter()
            .map(|query| {
                let hits = (0..hit_count)
                    .map(|rank| Qg6RankedHitReceipt {
                        document_id_sha256: lower_hex(&Sha256::digest(
                            format!("{}-document-{rank}", query.id()).as_bytes(),
                        )),
                        score_bits: 1.0_f32.to_bits(),
                    })
                    .collect::<Vec<_>>();
                let receipt = Qg6ResultReceipt::from_redacted_hits(
                    hits,
                    u64::try_from(hit_count).expect("QG-6 fixture hit count"),
                    document_count,
                    k,
                )
                .expect("sealed QG-6 result receipt");
                Qg6SixArmResultReceipts {
                    tantivy_null_left: receipt.clone(),
                    tantivy_null_right: receipt.clone(),
                    quill_null_left: receipt.clone(),
                    quill_null_right: receipt.clone(),
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
        attach_stream_with_leaf_latencies(
            samples,
            effect_stream,
            identity,
            contract,
            |_, parent_latency_ns| vec![parent_latency_ns],
        );
    }

    pub fn attach_stream_with_leaf_latencies(
        samples: &mut [PerfRawSample],
        effect_stream: bool,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
        leaf_latencies: impl FnMut(&PerfRawSample, u64) -> Vec<u64>,
    ) {
        let stream = if effect_stream {
            Qg6FormalStream::Effect
        } else {
            Qg6FormalStream::TantivyNull
        };
        attach_formal_stream_with_leaf_latencies(
            samples,
            stream,
            identity,
            contract,
            leaf_latencies,
        );
    }

    pub fn attach_stream_against_schedule_authority(
        samples: &mut [PerfRawSample],
        comparison: Qg6Comparison,
        authority: &Qg6ScheduleAuthority,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
    ) {
        attach_stream_against_schedule_authority_with_leaf_latencies(
            samples,
            comparison,
            authority,
            identity,
            contract,
            |_, parent_latency_ns| vec![parent_latency_ns],
        );
    }

    pub fn attach_stream_against_schedule_authority_with_leaf_latencies(
        samples: &mut [PerfRawSample],
        comparison: Qg6Comparison,
        authority: &Qg6ScheduleAuthority,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
        leaf_latencies: impl FnMut(&PerfRawSample, u64) -> Vec<u64>,
    ) {
        let stream = match comparison {
            Qg6Comparison::TantivyNull => Qg6FormalStream::TantivyNull,
            Qg6Comparison::QuillNull => Qg6FormalStream::QuillNull,
            Qg6Comparison::Effect => Qg6FormalStream::Effect,
        };
        let mut query_rounds = [0_usize; QG6_QUERY_GROUPS];
        let (pairs, remainder) = samples.as_chunks_mut::<2>();
        assert!(remainder.is_empty(), "paired QG-6 fixture");
        for pair in pairs {
            assert_eq!(pair[0].group_id, pair[1].group_id, "QG-6 fixture group");
            let query_index = usize::try_from(pair[0].group_id.expect("QG-6 fixture group"))
                .expect("bounded QG-6 query index");
            let round = query_rounds[query_index];
            query_rounds[query_index] += 1;
            let block = authority
                .schedule
                .iter()
                .filter(|block| block.query_index == query_index && block.comparison == comparison)
                .nth(round)
                .expect("authority schedules every QG-6 fixture pair");
            for sample in pair {
                let role = qg6_role(stream, sample.arm);
                let order = if role == block.first {
                    PerfSampleOrder::First
                } else {
                    assert_eq!(role, block.second, "authority schedules both fixture arms");
                    PerfSampleOrder::Second
                };
                sample.order = order;
                sample.block_id = block.block_id;
                sample.sample_id = block.block_id * 2 + u64::from(order == PerfSampleOrder::Second);
            }
        }
        assert!(
            query_rounds
                .iter()
                .all(|rounds| *rounds == authority.rounds_per_query),
            "fixture provides the authority's exact per-query unit count"
        );
        attach_formal_stream_with_leaf_latencies(
            samples,
            stream,
            identity,
            contract,
            leaf_latencies,
        );
    }

    fn attach_formal_stream_with_leaf_latencies(
        samples: &mut [PerfRawSample],
        stream: Qg6FormalStream,
        identity: &PerfInputIdentity,
        contract: &Qg6SemanticContract,
        mut leaf_latencies: impl FnMut(&PerfRawSample, u64) -> Vec<u64>,
    ) {
        let mut timeline_ns = 0_u64;
        let mut sample_leaf_latencies = BTreeMap::<u64, Vec<u64>>::new();
        let (pairs, remainder) = samples.as_chunks_mut::<2>();
        for pair in pairs {
            assert_eq!(pair[0].block_id, pair[1].block_id, "paired QG-6 fixture");
            let (left, right) = pair.split_at_mut(1);
            let ordered = if left[0].order == PerfSampleOrder::First {
                [&mut left[0], &mut right[0]]
            } else {
                [&mut right[0], &mut left[0]]
            };
            for sample in ordered {
                let observed_ms = sample.observed_value.expect("QG-6 fixture gauge");
                let elapsed = std::time::Duration::try_from_secs_f64(observed_ms / 1_000.0)
                    .expect("finite positive QG-6 fixture latency");
                let elapsed_ns =
                    u64::try_from(elapsed.as_nanos()).expect("bounded QG-6 fixture latency");
                assert!(elapsed_ns > 0, "positive QG-6 fixture latency");
                let latencies = leaf_latencies(sample, elapsed_ns);
                assert!(!latencies.is_empty(), "QG-6 fixture has timing leaves");
                assert!(
                    latencies.iter().all(|latency| *latency > 0),
                    "QG-6 fixture timing leaves are positive"
                );
                let mut sorted = latencies.clone();
                sorted.sort_unstable();
                assert_eq!(
                    sorted[sorted.len() / 2],
                    elapsed_ns,
                    "QG-6 fixture leaves reproduce the parent median"
                );
                let total_elapsed_ns = latencies
                    .iter()
                    .try_fold(0_u64, |total, latency| total.checked_add(*latency))
                    .expect("bounded QG-6 fixture leaf interval");
                sample.started_ns = timeline_ns;
                sample.ended_ns = timeline_ns + total_elapsed_ns;
                sample.observed_value = Some(elapsed_ns as f64 / 1_000_000.0);
                timeline_ns = sample.ended_ns + 1_000;
                assert!(
                    sample_leaf_latencies
                        .insert(sample.sample_id, latencies)
                        .is_none(),
                    "QG-6 fixture sample IDs are unique"
                );
            }
        }
        assert!(remainder.is_empty(), "paired QG-6 fixture");
        for sample in samples {
            let group_id = sample.group_id.expect("QG-6 fixture group");
            let group_index = usize::try_from(group_id).expect("QG-6 group index");
            let group = &contract.groups[group_index];
            let role = qg6_role(stream, sample.arm);
            sample.provenance.input_identity = Some(identity.clone());
            let comparison = match stream {
                Qg6FormalStream::TantivyNull => Qg6Comparison::TantivyNull,
                Qg6FormalStream::QuillNull => Qg6Comparison::QuillNull,
                Qg6FormalStream::Effect => Qg6Comparison::Effect,
            };
            let order = match sample.order {
                PerfSampleOrder::First => Qg6SampleOrder::First,
                PerfSampleOrder::Second => Qg6SampleOrder::Second,
            };
            let latencies = sample_leaf_latencies
                .remove(&sample.sample_id)
                .expect("prepared QG-6 fixture leaf latencies");
            let mut leaf_started_ns = sample.started_ns;
            let timing_leaves = latencies
                .iter()
                .copied()
                .map(|latency_ns| {
                    let leaf_ended_ns = leaf_started_ns + latency_ns;
                    let leaf =
                        Qg6SearchTimingLeafReceipt::from_interval(leaf_started_ns, leaf_ended_ns)
                            .expect("sealed QG-6 timing leaf");
                    leaf_started_ns = leaf_ended_ns;
                    leaf
                })
                .collect::<Vec<_>>();
            assert_eq!(
                leaf_started_ns, sample.ended_ns,
                "QG-6 fixture leaves fill their parent interval"
            );
            let subsample_count =
                u64::try_from(timing_leaves.len()).expect("bounded QG-6 fixture leaf cardinality");
            let mut sorted_latencies = latencies.clone();
            sorted_latencies.sort_unstable();
            sample.work_units = Some(subsample_count);
            let mut timed_sample = Qg6TimedSample {
                block_id: sample.block_id,
                sample_id: sample.sample_id,
                query_id: group.query.query_id.clone(),
                query_index: group_index,
                comparison,
                arm: role,
                order,
                started_ns: sample.started_ns,
                ended_ns: sample.ended_ns,
                observed_latency_ns: sorted_latencies[sorted_latencies.len() / 2],
                subsample_count,
                result_receipt_sha256: group.roles.get(role).receipt_sha256.clone(),
                result_sha256: qg6_result_sequence_sha256(group.roles.get(role), subsample_count)
                    .expect("QG-6 sequence digest"),
                timing_leaves,
                timing_leaves_sha256: String::new(),
            };
            timed_sample.timing_leaves_sha256 = timed_sample
                .recomputed_timing_leaves_sha256()
                .expect("QG-6 timing leaf seal");
            timed_sample
                .verify_timing_leaves()
                .expect("valid QG-6 timing fixture");
            sample.qg6_sample_binding = Some(Qg6SampleBinding {
                query_id: group.query.query_id.clone(),
                result_sequence_sha256: timed_sample.result_sha256.clone(),
                timed_sample,
            });
        }
        assert!(
            sample_leaf_latencies.is_empty(),
            "every QG-6 fixture leaf set is consumed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{
        PerfCellResult, PerfMetricSemantics, PerfOperationScope, PerfSampleArm, PerfSampleOrder,
        PerfSamplePhase, PerfSampleProvenance, QG6_QUERY_GROUP_IDS, Qg1BatchCoverage,
        Qg1LifecycleProducer, Qg1LifecycleWitness, Qg1SampleBinding, estimate_paired_experiment,
        estimate_paired_experiment_against_qg1_authority, seeded_balanced_pair_order,
    };
    use crate::qg6_prepared::{Qg6ExperimentIdentity, Qg6ResultReceipt};

    const CANARY: &str = "CANARY_DOCUMENT_TEXT_MUST_NEVER_PERSIST";
    const TEST_MACHINE_FINGERPRINT: &str =
        "linux-x86_64-test-machine-128thread-AMD_Ryzen_Threadripper_PRO_5995WX_64-Cores";

    #[test]
    fn public_evidence_loader_refuses_artifact_above_memory_bound() {
        let directory = tempfile::tempdir().expect("bounded evidence directory");
        let path = directory.path().join("oversized-evidence.json");
        let file = File::create(&path).expect("create sparse oversized evidence file");
        file.set_len(
            u64::try_from(PERF_EVIDENCE_MAX_ARTIFACT_BYTES).expect("evidence byte cap fits u64")
                + 1,
        )
        .expect("size sparse oversized evidence file");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::UnboundedArtifactBytes {
                count,
                max: PERF_EVIDENCE_MAX_ARTIFACT_BYTES,
            }) if count == u64::try_from(PERF_EVIDENCE_MAX_ARTIFACT_BYTES)
                .expect("evidence byte cap fits u64") + 1
        ));
    }

    fn gauge_scope() -> PerfOperationScope {
        perf_operation_scope(
            PerfGate::Qg2,
            "bulk/medium/1/positions_on",
            "docs_per_second",
        )
    }

    fn latency_scope() -> PerfOperationScope {
        perf_operation_scope(PerfGate::Qg6, "query/identifier/k10/100k", "latency_ms")
    }

    fn sample_provenance(run_id: &str) -> PerfSampleProvenance {
        PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            input_identity: None,
            worker_id: TEST_MACHINE_FINGERPRINT.to_owned(),
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
            qg1_sample_binding: None,
            tantivy_config_sha256: None,
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
            qg1_sample_binding: None,
            tantivy_config_sha256: None,
        });
    }

    fn gauge_stream_for_scope(
        scope: &PerfOperationScope,
        pairs: &[(f64, f64)],
        sample_id_base: u64,
        block_id_base: u64,
        group_id: Option<u64>,
    ) -> Vec<PerfRawSample> {
        let provenance = sample_provenance("run-a");
        let order = seeded_balanced_pair_order(pairs.len(), 0x00c0_ffee).expect("order");
        let mut samples = Vec::with_capacity(pairs.len() * 2);
        for (index, ((control, treatment), first_arm)) in pairs.iter().zip(order).enumerate() {
            let index = u64::try_from(index).expect("index");
            push_gauge_block(
                &mut samples,
                scope,
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

    fn gauge_stream(
        pairs: &[(f64, f64)],
        sample_id_base: u64,
        block_id_base: u64,
        group_id: Option<u64>,
    ) -> Vec<PerfRawSample> {
        gauge_stream_for_scope(
            &gauge_scope(),
            pairs,
            sample_id_base,
            block_id_base,
            group_id,
        )
    }

    fn grouped_gauge_stream(
        pairs: &[(u64, f64, f64)],
        sample_id_base: u64,
        force_control_first: Option<bool>,
    ) -> Vec<PerfRawSample> {
        let scope = gauge_scope();
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

    #[allow(clippy::too_many_arguments)]
    fn authority_bound_qg1_throughput_stream(
        scope: &PerfOperationScope,
        provenance: &PerfSampleProvenance,
        producer: &Qg1LifecycleProducer,
        stream_role: &str,
        first_arms: &[PerfSampleArm],
        elapsed_pairs_ns: &[(u64, u64)],
        sample_id_base: u64,
        work_units: u64,
        content_bytes: u64,
    ) -> Vec<PerfRawSample> {
        assert_eq!(
            first_arms.len(),
            elapsed_pairs_ns.len(),
            "every issued QG-1 pair needs exact control/treatment timings"
        );
        let mut samples = Vec::with_capacity(first_arms.len() * 2);
        for (index, (first_arm, (control_elapsed_ns, treatment_elapsed_ns))) in first_arms
            .iter()
            .copied()
            .zip(elapsed_pairs_ns.iter().copied())
            .enumerate()
        {
            let block_id = u64::try_from(index).expect("QG-1 test block ID");
            let base = block_id * 1_000_000;
            let control_first = first_arm == PerfSampleArm::Control;
            let (control_start, treatment_start) = if control_first {
                (base, base + control_elapsed_ns + 100)
            } else {
                (base + treatment_elapsed_ns + 100, base)
            };
            let sample_id = sample_id_base + block_id * 2;
            samples.push(PerfRawSample {
                block_id,
                sample_id,
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
                ended_ns: control_start + control_elapsed_ns,
                work_units: Some(work_units),
                byte_count: Some(content_bytes),
                observed_value: None,
                group_id: None,
                qg6_sample_binding: None,
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
            });
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id + 1,
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
                ended_ns: treatment_start + treatment_elapsed_ns,
                work_units: Some(work_units),
                byte_count: Some(content_bytes),
                observed_value: None,
                group_id: None,
                qg6_sample_binding: None,
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
            });
        }

        for sample in &mut samples {
            let stream_sequence =
                sample.block_id * 2 + u64::from(sample.order == PerfSampleOrder::Second);
            let tantivy_witness = stream_role == crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL
                || (stream_role == crate::perf::QG1_STREAM_ROLE_EFFECT
                    && sample.arm == PerfSampleArm::Control);
            let lifecycle_witness = if tantivy_witness {
                Qg1LifecycleWitness::Tantivy {
                    searchable_segments_before: 1,
                    searchable_segments_after: 1,
                    join_elapsed_ns: 1,
                    writer_rearmed: false,
                }
            } else {
                Qg1LifecycleWitness::Quill {
                    publication_generation_delta: 1,
                }
            };
            let binding = Qg1SampleBinding {
                schema_version: Qg1SampleBinding::SCHEMA_VERSION.to_owned(),
                stream_role: stream_role.to_owned(),
                stream_id_sha256: String::new(),
                stream_sequence,
                raw_sample_id: sample.sample_id,
                raw_block_id: sample.block_id,
                raw_arm: sample.arm,
                raw_order: sample.order,
                lifecycle_authority_sha256: String::new(),
                stream_role_identity_sha256: String::new(),
                producer_capability_sha256: String::new(),
                producer_capability_tag_sha256: String::new(),
                lifecycle_receipt_id_sha256: String::new(),
                lifecycle_receipt_sha256: String::new(),
                tantivy_writer_witness_sha256: tantivy_witness.then(|| {
                    let mut hasher = Sha256::new();
                    hasher.update(b"fixture-tantivy-witness-v1\0");
                    hasher.update(stream_role.as_bytes());
                    hasher.update(sample.block_id.to_le_bytes());
                    hasher.update(sample.sample_id.to_le_bytes());
                    hasher.update(format!("{:?}", sample.order).as_bytes());
                    lower_hex(&hasher.finalize())
                }),
                prepared_corpus_sha256: sample.provenance.corpus_sha256.clone(),
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
                terminal_endpoint_ns: sample.ended_ns - sample.started_ns,
                lifecycle_witness,
            };
            sample.qg1_sample_binding = Some(
                producer
                    .consume_lifecycle_receipt(&sample.scope, &sample.provenance, binding)
                    .expect("producer consumes one QG-1 lifecycle receipt per raw row"),
            );
        }
        samples
    }

    struct Qg1ExperimentFixture {
        paired: PairedExperimentResult,
        treatment_arm_null: PairedExperimentResult,
        expected_authority: Qg1ExpectedAuthority,
    }

    fn qg1_experiment_for_spec(
        spec: &EvidenceCellSpec,
        ratio: f64,
        tantivy_null_control_elapsed_ns: &[u64],
    ) -> Qg1ExperimentFixture {
        const PAIRS: usize = 12;
        const CONTENT_BYTES: u64 = 64_000;
        const BASE_ELAPSED_NS: u64 = 100_000;

        assert_eq!(
            tantivy_null_control_elapsed_ns.len(),
            PAIRS,
            "the QG-1 null fixture must fill every pre-issued pair"
        );

        let matrix = PerfMatrixSpec::complete();
        let canonical = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|cell| cell.fixture == spec.fixture && cell.metric == spec.metric)
            .expect("QG-1 fixture uses one canonical matrix cell");
        let work_units = canonical
            .document_count
            .expect("canonical QG-1 throughput cell has work units");
        let scope = perf_operation_scope(spec.gate, &spec.fixture, &spec.metric);
        let mut provenance = sample_provenance("run-a");
        provenance.input_identity = spec.input_identity.clone();
        let schedule =
            seeded_balanced_pair_order(PAIRS, 0x00c0_ffee).expect("QG-1 authority schedule");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let control_elapsed_ns = (ratio * 100_000.0).round() as u64;
        let effect_elapsed_ns = vec![(control_elapsed_ns, BASE_ELAPSED_NS); PAIRS];
        let tantivy_null_elapsed_ns = tantivy_null_control_elapsed_ns
            .iter()
            .copied()
            .map(|control| (control, BASE_ELAPSED_NS))
            .collect::<Vec<_>>();
        let quill_null_elapsed_ns = vec![(BASE_ELAPSED_NS, BASE_ELAPSED_NS); PAIRS];
        let mut estimator = config();
        let producer = estimator
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                "a".repeat(64),
                "b".repeat(64),
                work_units,
                CONTENT_BYTES,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: work_units,
                }],
                format!("synthetic-{:08}", work_units.saturating_sub(1)),
                u64::try_from(PAIRS).expect("QG-1 pair count fits u64"),
                vec![
                    (
                        crate::perf::QG1_STREAM_ROLE_EFFECT.to_owned(),
                        0,
                        0,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL.to_owned(),
                        0,
                        10_000,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_QUILL_NULL.to_owned(),
                        0,
                        20_000,
                        schedule.clone(),
                    ),
                ],
            )
            .expect("mint QG-1 authority before the first raw row");
        let expected_authority = producer.expected_authority().clone();
        let effect = authority_bound_qg1_throughput_stream(
            &scope,
            &provenance,
            &producer,
            crate::perf::QG1_STREAM_ROLE_EFFECT,
            &schedule,
            &effect_elapsed_ns,
            0,
            work_units,
            CONTENT_BYTES,
        );
        let tantivy_null = authority_bound_qg1_throughput_stream(
            &scope,
            &provenance,
            &producer,
            crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL,
            &schedule,
            &tantivy_null_elapsed_ns,
            10_000,
            work_units,
            CONTENT_BYTES,
        );
        let quill_null = authority_bound_qg1_throughput_stream(
            &scope,
            &provenance,
            &producer,
            crate::perf::QG1_STREAM_ROLE_QUILL_NULL,
            &schedule,
            &quill_null_elapsed_ns,
            20_000,
            work_units,
            CONTENT_BYTES,
        );
        let paired = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &tantivy_null,
            &estimator,
            Some(&expected_authority),
        )
        .expect("authority-bound QG-1 effect estimate");
        let treatment_arm_null = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &quill_null,
            &estimator,
            Some(&expected_authority),
        )
        .expect("authority-bound QG-1 treatment-arm null estimate");
        Qg1ExperimentFixture {
            paired,
            treatment_arm_null,
            expected_authority,
        }
    }

    fn valid_qg1_experiment_for_spec(spec: &EvidenceCellSpec, ratio: f64) -> Qg1ExperimentFixture {
        qg1_experiment_for_spec(spec, ratio, &[100_000; 12])
    }

    fn bind_samples_to_spec(samples: &mut [PerfRawSample], spec: &EvidenceCellSpec) {
        let scope = perf_operation_scope(spec.gate, &spec.fixture, &spec.metric);
        let mut provenance = sample_provenance("run-a");
        provenance.input_identity = spec.input_identity.clone();
        for sample in samples {
            sample.scope = scope.clone();
            sample.provenance = provenance.clone();
        }
    }

    fn valid_experiment_for_spec(spec: &EvidenceCellSpec, ratio: f64) -> PairedExperimentResult {
        if spec.gate == PerfGate::Qg1 && spec.metric == "docs_per_second" {
            return valid_qg1_experiment_for_spec(spec, ratio).paired;
        }
        let mut effect = gauge_stream(&effect_pairs(12, ratio), 0, 0, None);
        let mut null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        bind_samples_to_spec(&mut effect, spec);
        bind_samples_to_spec(&mut null, spec);
        estimate_paired_experiment(&effect, &null, &config()).expect("valid non-QG experiment")
    }

    fn policy() -> EvidencePolicy {
        EvidencePolicy::predeclared()
    }

    fn test_profile() -> crate::MachineProfileKey {
        crate::MachineProfileKey::new(
            crate::HardwareClassId::TrjZen35995wx,
            crate::ExecutionProfileId::Physical64,
        )
        .expect("canonical test profile")
    }

    fn applicability_plan(gate: PerfGate) -> PerfApplicabilityPlan {
        PerfMatrixSpec::complete()
            .applicability_plan(
                &MachineClassRegistry::frozen().expect("frozen machine registry"),
                test_profile(),
                gate,
            )
            .expect("canonical test applicability plan")
    }

    fn plan_binding(gate: PerfGate) -> PerfApplicabilityPlanBinding {
        applicability_plan(gate).binding
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

    fn evidence_provenance(gate: PerfGate) -> EvidenceProvenance {
        let plan = applicability_plan(gate);
        EvidenceProvenance {
            run_id: "run-a".to_owned(),
            run_window: "window-1".to_owned(),
            manifest_sha256: plan.binding.normalized_perf_manifest_sha256.clone(),
            build: build_identity(),
            machine: MachineIdentity {
                fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                logical_cpus: 64,
                execution: PerfExecutionProvenance {
                    host_identity: "test-machine".to_owned(),
                    producer_os: crate::PerfProducerOs::Linux,
                    physical_cores: 64,
                    logical_threads: 128,
                    process_available_threads: 64,
                    execution_capacity: plan
                        .execution_capacity
                        .expect("test profile has a frozen execution capacity"),
                    max_exercised_cell_width: plan
                        .max_exercised_cell_width
                        .expect("test profile has a frozen gate maximum"),
                    configured_engine_thread_widths: vec![1],
                    runtime_detected_isa: ["aes", "avx2", "bmi2", "fma", "vaes"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                    cpu_affinity_allowed_list: Some("0-63".to_owned()),
                    affinity_or_cpuset_cap: Some(
                        "Cpus_allowed_list=0-63 (64 of 128 host logical threads)".to_owned(),
                    ),
                },
                cpu_governor: Some("performance".to_owned()),
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
        seal_unbound_artifact_against_qg1_authorities(artifact, &[])
    }

    fn seal_unbound_artifact_against_qg1_authorities(
        artifact: &mut PerfEvidenceArtifact,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Vec<u8> {
        seal_unbound_artifact_against_authorities(artifact, external_qg1_authorities, &[])
    }

    fn seal_unbound_artifact_against_authorities(
        artifact: &mut PerfEvidenceArtifact,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Vec<u8> {
        let bytes = artifact
            .sealed_json()
            .expect("seal unbound test evidence")
            .into_bytes();
        *artifact = PerfEvidenceArtifact::from_verified_slice_against_authorities(
            &bytes,
            external_qg1_authorities,
            external_qg6_authorities,
        )
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

    fn bind_test_identity_against_authorities(
        artifact: &mut PerfEvidenceArtifact,
        gate: PerfGate,
        threshold_artifact_bytes: &[u8],
        run_label: &str,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
        external_qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Vec<u8> {
        let source = seal_unbound_artifact_against_authorities(
            artifact,
            external_qg1_authorities,
            external_qg6_authorities,
        );
        let identity = admitted_identity(gate, threshold_artifact_bytes, &source, run_label);
        artifact
            .bind_machine_class_identity_against_authorities(
                identity,
                threshold_artifact_bytes,
                &source,
                external_qg1_authorities,
                external_qg6_authorities,
            )
            .expect("bind admitted test identity against retained authorities");
        source
    }

    fn unbind_test_artifact(artifact: &mut PerfEvidenceArtifact) {
        artifact.machine_class =
            MachineClassEvidenceBinding::unverified("sealed runner receipt has not been bound");
        artifact.gate_decision = None;
        artifact.artifact_sha256.clear();
    }

    fn reseal_json_value(mut value: serde_json::Value) -> Vec<u8> {
        value["artifact_sha256"] = serde_json::Value::String(String::new());
        let unsealed = serde_json::to_string_pretty(&value).expect("serialize unsealed JSON");
        value["artifact_sha256"] =
            serde_json::Value::String(lower_hex(&Sha256::digest(unsealed.as_bytes())));
        serde_json::to_vec_pretty(&value).expect("serialize sealed JSON")
    }

    /// Build a fully coherent hostile object: the mutated pre-binding bytes
    /// receive a fresh artifact manifest and admitted completion receipt, then
    /// the enclosing evidence is resealed. Strict reload must therefore reject
    /// the semantic join itself rather than merely noticing a stale digest.
    fn coherently_bind_and_reseal(
        mut artifact: PerfEvidenceArtifact,
        threshold_artifact_bytes: &[u8],
        run_label: &str,
    ) -> Vec<u8> {
        let gate = artifact.gate;
        unbind_test_artifact(&mut artifact);
        let prebinding_bytes = artifact
            .sealed_json()
            .expect("seal hostile pre-binding evidence")
            .into_bytes();
        let mut bound: PerfEvidenceArtifact = serde_json::from_slice(&prebinding_bytes)
            .expect("decode hostile pre-binding evidence without admitting it");
        let identity =
            admitted_identity(gate, threshold_artifact_bytes, &prebinding_bytes, run_label);
        bound.machine_class = MachineClassEvidenceBinding::verified(identity);
        bound.artifact_sha256.clear();
        bound
            .sealed_json()
            .expect("seal hostile receipt-bound evidence")
            .into_bytes()
    }

    fn cell_spec(gate: PerfGate, role: EvidenceRole) -> EvidenceCellSpec {
        let (input_identity, qg6_semantic_contract, fixture, metric) = match gate {
            PerfGate::Qg1 => (
                None,
                None,
                "bulk/tiny/1/positions_on".to_owned(),
                "docs_per_second".to_owned(),
            ),
            PerfGate::Qg2 => (
                None,
                None,
                "bulk/medium/1/positions_on".to_owned(),
                "docs_per_second".to_owned(),
            ),
            PerfGate::Qg6 => {
                let (identity, contract) =
                    qg6_test_fixture::contract(crate::PerfQueryClass::Identifier);
                (
                    Some(identity),
                    Some(contract),
                    "query/identifier/k10/100k".to_owned(),
                    "latency_ms".to_owned(),
                )
            }
            PerfGate::Qg8 => (
                None,
                None,
                "scaling/xlarge/1/positions_on".to_owned(),
                "docs_per_second".to_owned(),
            ),
            PerfGate::Qg9 => (
                None,
                None,
                "cold_open/xlarge/default".to_owned(),
                "open_latency_ms".to_owned(),
            ),
            PerfGate::Qg10 => (
                None,
                None,
                "dependency_surface/default_lexical".to_owned(),
                "tantivy_nodes".to_owned(),
            ),
            _ => {
                let matrix = PerfMatrixSpec::complete();
                let canonical = matrix
                    .for_gate(gate)
                    .into_iter()
                    .next()
                    .expect("canonical gate cell");
                (
                    None,
                    None,
                    canonical.fixture.clone(),
                    canonical.metric.clone(),
                )
            }
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
            unit: perf_metric_unit(&metric).to_owned(),
            metric,
            role,
            input_identity,
            qg6_semantic_contract,
            cold_cache: None,
            concurrency_witness,
        }
    }

    fn provisional_cell_with_authority() -> (EvidenceCell, Qg1ExpectedAuthority) {
        let spec = cell_spec(PerfGate::Qg1, EvidenceRole::Required);
        let fixture = valid_qg1_experiment_for_spec(&spec, 1.10);
        let mut cell = EvidenceCell::evaluate(spec.clone(), fixture.paired, &policy())
            .expect("provisional cell");
        cell.attach_treatment_arm_null_against_qg1_authority(
            fixture.treatment_arm_null,
            &policy(),
            Some(&fixture.expected_authority),
        )
        .expect("attach authority-bound QG-1 treatment-arm null");
        (cell, fixture.expected_authority)
    }

    fn provisional_cell() -> EvidenceCell {
        provisional_cell_with_authority().0
    }

    fn provisional_qg2_cell() -> EvidenceCell {
        let spec = cell_spec(PerfGate::Qg2, EvidenceRole::Required);
        EvidenceCell::evaluate(
            spec.clone(),
            valid_experiment_for_spec(&spec, 1.10),
            &policy(),
        )
        .expect("provisional QG-2 cell")
    }

    fn provisional_artifact() -> PerfEvidenceArtifact {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg2,
            plan_binding(PerfGate::Qg2),
            policy(),
            evidence_provenance(PerfGate::Qg2),
            vec![provisional_qg2_cell()],
        )
        .expect("provisional artifact");
        bind_test_identity(
            &mut artifact,
            PerfGate::Qg2,
            b"qg2-threshold",
            "qg2-primary",
        );
        artifact
    }

    pub(super) fn qg6_artifact() -> (PerfEvidenceArtifact, Qg6ScheduleAuthority) {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let input_identity = spec.input_identity.as_ref().expect("QG-6 input identity");
        let semantic_contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect = qg6_hierarchical_stream_with_ratio(1.02, 0);
        let mut null = qg6_hierarchical_stream_with_ratio(1.0, 10_000);
        let mut quill_null = qg6_hierarchical_stream_with_ratio(1.0, 20_000);
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
        bind_samples_to_spec(&mut quill_null, &spec);
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: input_identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: input_identity.query_manifest_sha256.clone(),
                config_contract_sha256: input_identity.config_contract_sha256.clone(),
                document_count: semantic_contract.document_count,
                k: semantic_contract.k,
            },
            semantic_contract.groups.len(),
            5,
            1,
            config().bootstrap_seed,
        )
        .expect("retained QG-6 schedule authority");
        qg6_test_fixture::attach_stream_against_schedule_authority(
            &mut effect,
            Qg6Comparison::Effect,
            &authority,
            input_identity,
            semantic_contract,
        );
        qg6_test_fixture::attach_stream_against_schedule_authority(
            &mut null,
            Qg6Comparison::TantivyNull,
            &authority,
            input_identity,
            semantic_contract,
        );
        qg6_test_fixture::attach_stream_against_schedule_authority(
            &mut quill_null,
            Qg6Comparison::QuillNull,
            &authority,
            input_identity,
            semantic_contract,
        );
        let paired =
            estimate_paired_experiment(&effect, &null, &config()).expect("QG-6 paired estimate");
        let protocol = Qg6FormalProtocolEvidence::new_against_authority_fixture(
            &paired,
            quill_null,
            &authority,
            input_identity,
            semantic_contract,
        )
        .expect("construct retained QG-6 formal protocol");
        let mut cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        cell.attach_qg6_formal_protocol_against_authority(protocol, &policy(), &authority)
            .expect("attach retained QG-6 formal protocol");
        let mut provenance = evidence_provenance(PerfGate::Qg6);
        provenance.corpus.query_set_sha256 = Some("d".repeat(64));
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            plan_binding(PerfGate::Qg6),
            policy(),
            provenance,
            vec![cell],
        )
        .expect("QG-6 artifact");
        bind_test_identity_against_authorities(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-primary",
            &[],
            &[&authority],
        );
        (artifact, authority)
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
            let binding = sample
                .qg6_sample_binding
                .as_mut()
                .expect("QG-6 authenticated binding");
            binding.query_id = query_ids[group_index].clone();
            binding.timed_sample.query_id.clone_from(&binding.query_id);
            binding.timed_sample.timing_leaves_sha256 = binding
                .timed_sample
                .recomputed_timing_leaves_sha256()
                .expect("re-seal query-bound timing leaves");
        }
    }

    fn reauthorize_qg6_test_artifact(
        artifact: &mut PerfEvidenceArtifact,
        previous_authority: &Qg6ScheduleAuthority,
    ) -> Qg6ScheduleAuthority {
        let cell = &mut artifact.cells[0];
        let contract = cell
            .spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract")
            .clone();
        let identity = cell
            .spec
            .input_identity
            .as_ref()
            .expect("QG-6 input identity")
            .clone();
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: identity.query_manifest_sha256.clone(),
                config_contract_sha256: identity.config_contract_sha256.clone(),
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            previous_authority.rounds_per_query,
            previous_authority.searches_per_sample,
            previous_authority.schedule_seed,
        )
        .expect("re-authorize fully resealed QG-6 test artifact");
        let query_ids = contract
            .groups
            .iter()
            .map(|group| group.query.query_id.clone())
            .collect::<Vec<_>>();
        let EvidenceCellBody::Paired {
            paired,
            qg6_protocol,
            ..
        } = &mut cell.body
        else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.input_identity = Some(identity.clone());
        let protocol = qg6_protocol
            .as_mut()
            .expect("QG-6 formal protocol evidence");
        for sample in paired
            .effect_samples
            .iter_mut()
            .chain(&mut paired.null_samples)
            .chain(&mut protocol.quill_null_samples)
        {
            sample.provenance.input_identity = Some(identity.clone());
            let group_index =
                usize::try_from(sample.group_id.expect("QG-6 sample group")).expect("group index");
            let binding = sample
                .qg6_sample_binding
                .as_mut()
                .expect("QG-6 authenticated binding");
            binding.query_id = query_ids[group_index].clone();
            binding.timed_sample.query_id.clone_from(&binding.query_id);
            binding.timed_sample.timing_leaves_sha256 = binding
                .timed_sample
                .recomputed_timing_leaves_sha256()
                .expect("re-seal query-bound timing leaves");
        }
        protocol.schedule_authority = authority.clone();
        protocol.joint_tail = estimate_qg6_joint_tail_from_validated_rows(
            paired,
            &protocol.quill_null_samples,
            &authority,
        )
        .expect("recompute fully resealed QG-6 joint tail");
        authority
    }

    fn fully_reseal_qg6_result_receipt_mutation(
        artifact: &mut PerfEvidenceArtifact,
        authority: &Qg6ScheduleAuthority,
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
                let stream = if effect_stream {
                    Qg6FormalStream::Effect
                } else {
                    Qg6FormalStream::TantivyNull
                };
                let role = qg6_role(stream, sample.arm);
                let binding = sample
                    .qg6_sample_binding
                    .as_mut()
                    .expect("QG-6 authenticated binding");
                binding.query_id.clone_from(&group.query.query_id);
                let receipt = group.roles.get(role);
                let result_sequence_sha256 = qg6_result_sequence_sha256(receipt, work_units)
                    .expect("fully resealed result sequence");
                binding
                    .result_sequence_sha256
                    .clone_from(&result_sequence_sha256);
                binding
                    .timed_sample
                    .result_receipt_sha256
                    .clone_from(&receipt.receipt_sha256);
                binding
                    .timed_sample
                    .result_sha256
                    .clone_from(&result_sequence_sha256);
                binding.timed_sample.timing_leaves_sha256 = binding
                    .timed_sample
                    .recomputed_timing_leaves_sha256()
                    .expect("fully resealed timing-leaf sequence");
            }
        };
        rebind_stream(&mut paired.effect_samples, true);
        rebind_stream(&mut paired.null_samples, false);
        reauthorize_qg6_test_artifact(artifact, authority);
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
        let spec = cell_spec(PerfGate::Qg1, EvidenceRole::Required);
        let noisy_null_control_elapsed_ns = (0..12)
            .map(|index| if index % 2 == 0 { 135_000 } else { 70_000 })
            .collect::<Vec<_>>();
        let Qg1ExperimentFixture {
            paired: experiment,
            expected_authority,
            ..
        } = qg1_experiment_for_spec(&spec, 1.10, &noisy_null_control_elapsed_ns);
        assert_eq!(experiment.status, PairedEvidenceStatus::InvalidNull);
        assert_eq!(experiment.claim_state, PairedClaimState::NoDecision);

        let cell = EvidenceCell::evaluate(spec, experiment, &policy()).expect("cell");
        assert_eq!(cell.status, EvidenceDecisionStatus::InvalidNull);
        assert!(!cell.claim_eligible());

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
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
            .write_atomic_against_qg1_authorities(dir.path(), &[&expected_authority])
            .expect("durable invalid run");
        let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
            &paths.json,
            &[&expected_authority],
        )
        .expect("reload");
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::InvalidNull);
    }

    #[test]
    fn unverified_machine_binding_is_explicit_durable_and_nonpromotable() {
        let (cell, expected_authority) = provisional_cell_with_authority();
        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![cell],
        )
        .expect("unverified evidence artifact");
        assert!(matches!(
            &artifact.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(!artifact.ratchet_admissible());

        let directory = tempfile::tempdir().expect("unverified evidence directory");
        let paths = artifact
            .write_atomic_against_qg1_authorities(directory.path(), &[&expected_authority])
            .expect("persist explicit unverified evidence");
        let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
            &paths.json,
            &[&expected_authority],
        )
        .expect("reload unverified evidence");
        assert!(matches!(
            &reloaded.machine_class,
            MachineClassEvidenceBinding::Unverified { .. }
        ));
        assert!(!reloaded.ratchet_admissible());
    }

    #[test]
    fn post_exit_binding_returns_exact_verified_receipt_bound_bytes() {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg2,
            plan_binding(PerfGate::Qg2),
            policy(),
            evidence_provenance(PerfGate::Qg2),
            vec![provisional_qg2_cell()],
        )
        .expect("unverified producer evidence");
        let threshold_bytes = b"qg2-threshold";
        let source = seal_unbound_artifact(&mut artifact);
        let identity =
            admitted_identity(PerfGate::Qg2, threshold_bytes, &source, "post-exit-primary");
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
            PerfGate::Qg2,
            plan_binding(PerfGate::Qg2),
            policy(),
            evidence_provenance(PerfGate::Qg2),
            vec![provisional_qg2_cell()],
        )
        .expect("unverified producer evidence");
        let threshold_bytes = b"qg2-threshold";
        let source = seal_unbound_artifact(&mut artifact);
        let drifted_argv_identity =
            crate::machine_class_registry::admitted_test_identity_for_artifacts(
                PerfGate::Qg2.label(),
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
    fn runner_binding_and_integrity_reject_plan_maximum_envelope_drift() {
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg2,
            plan_binding(PerfGate::Qg2),
            policy(),
            evidence_provenance(PerfGate::Qg2),
            vec![provisional_qg2_cell()],
        )
        .expect("unverified QG-2 producer evidence");
        let source = seal_unbound_artifact(&mut artifact);
        let wrong_maximum_identity = admitted_identity(
            PerfGate::Qg1,
            b"qg1-envelope-threshold",
            &source,
            "wrong-maximum",
        );

        assert!(matches!(
            artifact.bind_machine_class_identity(
                wrong_maximum_identity.clone(),
                b"qg1-envelope-threshold",
                &source,
            ),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("profile/capacity/maximum envelope")
        ));

        artifact.machine_class = MachineClassEvidenceBinding::verified(wrong_maximum_identity);
        let directory = tempfile::tempdir().expect("runner envelope directory");
        let path = directory.path().join("wrong-runner-envelope.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("reseal wrong-runner-envelope artifact"),
        )
        .expect("persist wrong-runner-envelope artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("profile/capacity/maximum envelope")
        ));
    }

    #[test]
    fn verified_runner_binding_cannot_be_reassigned_to_another_receipt() {
        let mut artifact = provisional_artifact();
        let original = artifact.machine_class.clone();
        let source = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstruct pre-binding source");
        let different_receipt = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg2.label(),
            &"d".repeat(40),
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"e".repeat(64),
            "different-completion",
            "run-a",
            "window-1",
            b"qg2-threshold",
            &source,
        );

        assert!(matches!(
            artifact.bind_machine_class_identity_and_seal(
                different_receipt,
                b"qg2-threshold",
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

    /// The shipped append-only register is the one the repository actually
    /// carries, so a malformed edit to it fails the build rather than silently
    /// disarming the quarantine.
    const SHIPPED_QUARANTINE_REGISTER: &str =
        include_str!("../../../.bench-history/QUARANTINE.jsonl");

    /// Build an intact, sealed artifact whose measured revision is `revision`.
    ///
    /// The runner receipt has to be minted at the SAME revision as the
    /// provenance: `bind_machine_class_identity` cross-checks them and rejects
    /// a mismatch with `InvalidProvenance`. That check is doing its job, so the
    /// fixture binds a receipt at `revision` rather than reusing the default
    /// `admitted_identity` helper, which hardcodes `"d".repeat(40)`. The point
    /// of these tests is an artifact that is flawless everywhere EXCEPT that
    /// its revision is quarantined; an artifact with a desynchronized receipt
    /// would be refused for the wrong reason and prove nothing.
    fn artifact_measured_at(revision: &str) -> PerfEvidenceArtifact {
        let mut provenance = evidence_provenance(PerfGate::Qg2);
        provenance.build.git_revision = revision.to_owned();
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg2,
            plan_binding(PerfGate::Qg2),
            policy(),
            provenance,
            vec![provisional_qg2_cell()],
        )
        .expect("artifact at the requested revision");
        let source = seal_unbound_artifact(&mut artifact);
        let identity = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg2.label(),
            revision,
            &"c".repeat(64),
            &"a".repeat(64),
            &"f".repeat(64),
            &"e".repeat(64),
            "qg2-primary",
            "run-a",
            "window-1",
            b"qg2-threshold",
            &source,
        );
        artifact
            .bind_machine_class_identity(identity, b"qg2-threshold", &source)
            .expect("bind an admitted receipt minted at the same revision");
        artifact
    }

    /// PLANTED NEGATIVE for the QG-1 invalid-sweep quarantine.
    ///
    /// The artifact here is not broken in any way an integrity check can see:
    /// current schema, exact canonical pretty JSON, intact seal, summaries that
    /// recompute from their own raw samples. Only its measured revision is
    /// quarantined. Both halves are asserted deliberately — that the naive
    /// loader ADMITS it is the defect being closed, and asserting only the
    /// refusal would leave a test that still passes if the quarantine stopped
    /// discriminating and simply refused everything.
    #[test]
    fn quarantined_revision_is_refused_although_the_naive_loader_admits_it() {
        let artifact = artifact_measured_at("193d2e3fa1b2c3d4e5f60718293a4b5c6d7e8f90");
        let directory = tempfile::tempdir().expect("quarantine directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("write quarantined artifact");

        PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("an intact artifact from a quarantined sweep still passes integrity checks");

        let register = PerfQuarantineRegister::from_jsonl(SHIPPED_QUARANTINE_REGISTER)
            .expect("the shipped register parses");
        let error = PerfEvidenceArtifact::load_admissible_evidence(&paths.json, &register)
            .expect_err("a quarantined revision must never load as evidence");
        let EvidenceArtifactError::QuarantinedRevision {
            git_revision,
            git_revision_prefix,
            recorded_by,
            ..
        } = error
        else {
            panic!("expected a quarantine refusal, got {error:?}");
        };
        assert_eq!(git_revision, "193d2e3fa1b2c3d4e5f60718293a4b5c6d7e8f90");
        assert_eq!(git_revision_prefix, "193d2e3f");
        assert_eq!(recorded_by, "bd-qg1-invalid-sweep-quarantine-h4sqj");
    }

    /// The quarantine must discriminate: an artifact from any other revision is
    /// still admissible through the same loader.
    #[test]
    fn an_unquarantined_revision_still_loads_as_evidence() {
        let artifact = artifact_measured_at("0f1e2d3c4b5a69788796a5b4c3d2e1f009182736");
        let directory = tempfile::tempdir().expect("admissible directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("write admissible artifact");
        let register = PerfQuarantineRegister::from_jsonl(SHIPPED_QUARANTINE_REGISTER)
            .expect("the shipped register parses");

        PerfEvidenceArtifact::load_admissible_evidence(&paths.json, &register)
            .expect("an unquarantined revision remains admissible evidence");
    }

    /// The shipped register must actually cover the three sweep revisions this
    /// correction names, so the file cannot drift into a decorative comment.
    #[test]
    fn the_shipped_register_quarantines_every_named_sweep_revision() {
        let register = PerfQuarantineRegister::from_jsonl(SHIPPED_QUARANTINE_REGISTER)
            .expect("the shipped register parses");
        assert_eq!(register.records().len(), 3);
        for revision in [
            "193d2e3fa1b2c3d4e5f60718293a4b5c6d7e8f90",
            "544ffeb0112233445566778899aabbccddeeff00",
            "e0dc6ba3ffeeddccbbaa99887766554433221100",
        ] {
            assert!(
                register.quarantine_of(revision).is_some(),
                "{revision} must be quarantined by the shipped register"
            );
        }
    }

    /// A register that exists but cannot be parsed must be an error, never an
    /// empty register: a corrupted file must not read as "nothing is
    /// quarantined". A genuinely absent register is still empty and fine.
    #[test]
    fn a_corrupt_register_fails_closed_but_an_absent_one_is_empty() {
        assert!(matches!(
            PerfQuarantineRegister::from_jsonl(
                r#"{"schema_version":"quill-perf-quarantine-v1","git_revision_prefix":"193d","reason":"too short","recorded_by":"t"}"#
            ),
            Err(EvidenceArtifactError::Malformed { .. })
        ));
        assert!(matches!(
            PerfQuarantineRegister::from_jsonl(
                r#"{"schema_version":"quill-perf-quarantine-v0","git_revision_prefix":"193d2e3f","reason":"foreign schema","recorded_by":"t"}"#
            ),
            Err(EvidenceArtifactError::Malformed { .. })
        ));

        let directory = tempfile::tempdir().expect("absent register directory");
        let absent = PerfQuarantineRegister::load_from_history_dir(directory.path())
            .expect("an absent register is an empty register");
        assert!(absent.records().is_empty());
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
        // Eight blocks at an identical 0.9 per-block ratio plus two extreme
        // blocks (tiny control, enormous treatment): the paired median stays
        // 0.9 while the marginal arm-median ratio flips above 1.0. Any
        // half/order-subset median contains at most two extreme blocks and
        // stays at 0.9, so the bd-yo5by effect drift/order-effect gates stay
        // quiet (the previous half-split fixture drifted between halves and
        // now correctly classifies as InvalidExperiment first).
        let pairs = vec![
            (100.0, 90.0),
            (110.0, 99.0),
            (120.0, 108.0),
            (130.0, 117.0),
            (140.0, 126.0),
            (150.0, 135.0),
            (160.0, 144.0),
            (170.0, 153.0),
            (1.0, 100_000.0),
            (2.0, 200_000.0),
        ];
        let effect = gauge_stream(&pairs, 0, 0, None);
        let null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        let experiment = estimate_paired_experiment(&effect, &null, &config()).expect("estimate");
        assert_eq!(
            experiment.status,
            PairedEvidenceStatus::ContradictorySummaries
        );
        let cell = EvidenceCell::evaluate(
            cell_spec(PerfGate::Qg2, EvidenceRole::Required),
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
        let scope = latency_scope();
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
            samples.extend(gauge_stream_for_scope(
                &scope,
                &pairs,
                sample_id,
                block_base,
                Some(group),
            ));
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
    fn qg6_joint_tail_recovers_p99_hidden_by_every_parent_median() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.as_ref().expect("QG-6 input identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect = qg6_hierarchical_stream_with_ratio(1.0, 0);
        let mut tantivy_null = qg6_hierarchical_stream_with_ratio(1.0, 10_000);
        let mut quill_null = qg6_hierarchical_stream_with_ratio(1.0, 20_000);
        for samples in [&mut effect, &mut tantivy_null, &mut quill_null] {
            for sample in samples {
                sample.observed_value = Some(1.0);
            }
        }
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut tantivy_null, &spec);
        bind_samples_to_spec(&mut quill_null, &spec);
        let estimator_config = config();
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: identity.query_manifest_sha256.clone(),
                config_contract_sha256: identity.config_contract_sha256.clone(),
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            5,
            101,
            estimator_config.bootstrap_seed,
        )
        .expect("tail authority");
        qg6_test_fixture::attach_stream_against_schedule_authority_with_leaf_latencies(
            &mut effect,
            Qg6Comparison::Effect,
            &authority,
            identity,
            contract,
            |sample, parent| {
                let mut leaves = vec![parent; 101];
                if sample.arm == PerfSampleArm::Treatment {
                    // Nine heavy leaves in EVERY treatment unit: the mass sits
                    // below the median (hidden from every parent median) yet
                    // dominates the pooled p99 in every possible unit mixture,
                    // keeping the replicate distribution unimodal.
                    for leaf in &mut leaves[92..] {
                        *leaf = parent * 100;
                    }
                }
                leaves
            },
        );
        for (samples, comparison) in [
            (&mut tantivy_null[..], Qg6Comparison::TantivyNull),
            (&mut quill_null[..], Qg6Comparison::QuillNull),
        ] {
            qg6_test_fixture::attach_stream_against_schedule_authority_with_leaf_latencies(
                samples,
                comparison,
                &authority,
                identity,
                contract,
                |_, parent| vec![parent; 101],
            );
        }
        let paired = estimate_paired_experiment(&effect, &tantivy_null, &estimator_config)
            .expect("parent-median stream remains exactly null");
        assert!((paired.effect.treatment_over_control - 1.0).abs() < 1.0e-12);
        let mut mutated_paired = paired.clone();
        let mutated_leaf = mutated_paired.effect_samples[0]
            .qg6_sample_binding
            .as_mut()
            .expect("timed QG-6 binding")
            .timed_sample
            .timing_leaves
            .first_mut()
            .expect("timing leaf");
        mutated_leaf.ended_ns = mutated_leaf.ended_ns.saturating_add(1);
        assert!(matches!(
            Qg6FormalProtocolEvidence::new_against_authority_fixture(
                &mutated_paired,
                quill_null.clone(),
                &authority,
                identity,
                contract,
            ),
            Err(EvidenceArtifactError::InconsistentArtifact { ref reason })
                if reason.contains("failed authenticated replay")
        ));
        let protocol = Qg6FormalProtocolEvidence::new_against_authority_fixture(
            &paired, quill_null, &authority, identity, contract,
        )
        .expect("construct formal tail protocol");
        let tail = estimate_qg6_joint_tail(&paired, &protocol, &authority, identity, contract)
            .expect("joint tail estimate");
        assert!((tail.tantivy_null.p50_ratio - 1.0).abs() < 1.0e-12);
        assert!((tail.tantivy_null.p99_ratio - 1.0).abs() < 1.0e-12);
        assert!((tail.quill_null.p50_ratio - 1.0).abs() < 1.0e-12);
        assert!((tail.quill_null.p99_ratio - 1.0).abs() < 1.0e-12);
        assert!((tail.effect.p50_ratio - 1.0).abs() < 1.0e-12);
        assert!(
            tail.effect.p99_ratio > 50.0,
            "true-leaf p99 must expose the hidden treatment tail: {tail:?}"
        );
        // Normative decisions: the hidden tail sits below the median, so p50
        // equivalence legitimately holds; p99 noninferiority must fail, and
        // the tampered recompute check above proves every field here is bound
        // to the raw leaves.
        assert!(tail.effect.p99_ucb_ratio > 1.0);
        assert!(!tail.effect.p99_noninferior);
    }

    #[test]
    fn qg6_joint_tail_bootstrap_draws_queries_then_whole_units() {
        let mut seed = 7;
        let mut draws = Vec::new();
        for_each_qg6_joint_tail_bootstrap_unit(&mut seed, 3, 2, |query, unit| {
            draws.push((query, unit));
        })
        .expect("bounded query-first draw schedule");
        assert_eq!(draws, [(0, 1), (0, 0), (2, 0), (2, 0), (2, 0), (2, 1)]);
    }

    #[test]
    fn qg6_joint_tail_bootstrap_carries_all_six_roles_in_one_cluster_draw() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let identity = spec.input_identity.as_ref().expect("QG-6 input identity");
        let contract = spec
            .qg6_semantic_contract
            .as_ref()
            .expect("QG-6 semantic contract");
        let mut effect = qg6_hierarchical_stream_with_ratio(0.8, 0);
        let mut tantivy_null = qg6_hierarchical_stream_with_ratio(1.0, 10_000);
        let mut quill_null = qg6_hierarchical_stream_with_ratio(1.0, 20_000);
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut tantivy_null, &spec);
        bind_samples_to_spec(&mut quill_null, &spec);
        let estimator_config = config();
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: identity.query_manifest_sha256.clone(),
                config_contract_sha256: identity.config_contract_sha256.clone(),
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            5,
            3,
            estimator_config.bootstrap_seed,
        )
        .expect("shared-draw authority");
        for (samples, comparison) in [
            (&mut effect[..], Qg6Comparison::Effect),
            (&mut tantivy_null[..], Qg6Comparison::TantivyNull),
            (&mut quill_null[..], Qg6Comparison::QuillNull),
        ] {
            qg6_test_fixture::attach_stream_against_schedule_authority_with_leaf_latencies(
                samples,
                comparison,
                &authority,
                identity,
                contract,
                |_, parent| vec![parent, parent, parent],
            );
        }
        let paired = estimate_paired_experiment(&effect, &tantivy_null, &estimator_config)
            .expect("paired six-role stream");
        let protocol = Qg6FormalProtocolEvidence::new_against_authority_fixture(
            &paired, quill_null, &authority, identity, contract,
        )
        .expect("formal six-role protocol");
        let tail = estimate_qg6_joint_tail(&paired, &protocol, &authority, identity, contract)
            .expect("joint clustered estimate");
        for ratio in [
            tail.tantivy_null.p50_ci95_low_ratio,
            tail.tantivy_null.p50_ci95_high_ratio,
            tail.tantivy_null.p99_ci95_low_ratio,
            tail.tantivy_null.p99_ci95_high_ratio,
            tail.quill_null.p50_ci95_low_ratio,
            tail.quill_null.p50_ci95_high_ratio,
            tail.quill_null.p99_ci95_low_ratio,
            tail.quill_null.p99_ci95_high_ratio,
        ] {
            assert!(
                (ratio - 1.0).abs() < 1.0e-12,
                "null draw split roles: {tail:?}"
            );
        }
        for ratio in [
            tail.effect.p50_ratio,
            tail.effect.p50_ci95_low_ratio,
            tail.effect.p50_ci95_high_ratio,
            tail.effect.p99_ratio,
            tail.effect.p99_ci95_low_ratio,
            tail.effect.p99_ci95_high_ratio,
        ] {
            assert!(
                (ratio - 0.8).abs() < 1.0e-12,
                "effect draw split roles: {tail:?}"
            );
        }
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

        let single_group =
            gauge_stream_for_scope(&latency_scope(), &effect_pairs(12, 1.10), 0, 0, Some(7));
        assert!(matches!(
            estimate_hierarchical_latency(&single_group, &config(), &policy()),
            Err(PairedEstimatorError::InsufficientGroups { .. })
        ));
    }

    #[test]
    fn qg6_rows_without_groups_or_semantic_bindings_fail_closed() {
        let spec = cell_spec(PerfGate::Qg6, EvidenceRole::Required);
        let mut effect = gauge_stream(&effect_pairs(12, 1.10), 0, 0, None);
        let mut null = gauge_stream(&quiet_null_pairs(12), 10_000, 0, None);
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
        assert!(matches!(
            estimate_paired_experiment(&effect, &null, &config()),
            Err(PairedEstimatorError::InvalidProvenance { .. })
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
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
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
        let (artifact, authority) = qg6_artifact();
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
            .write_atomic_against_authorities(directory.path(), &[], &[&authority])
            .expect("persist QG-6 artifact");
        let verified = PerfEvidenceArtifact::load_verified_against_authorities(
            &paths.json,
            &[],
            &[&authority],
        )
        .expect("verify QG-6 artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&paths.json),
            Err(EvidenceArtifactError::InvalidProvenance { ref reason })
                if reason.contains("no independently retained schedule authority")
        ));
        let wrong_authority = Qg6ScheduleAuthority::for_experiment(
            authority.identity.clone(),
            authority.query_count,
            authority.rounds_per_query,
            authority.searches_per_sample,
            authority.schedule_seed.wrapping_add(1),
        )
        .expect("different valid QG-6 authority");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified_against_authorities(
                &paths.json,
                &[],
                &[&wrong_authority],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { ref reason })
                if reason.contains("no independently retained schedule authority")
        ));
        assert!(matches!(
            PerfEvidenceArtifact::load_verified_against_authorities(
                &paths.json,
                &[],
                &[&authority, &authority],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { ref reason })
                if reason.contains("more than one retained schedule authority")
        ));
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
        let (mut artifact, authority) = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let cell = &mut artifact.cells[0];
        cell.spec.qg6_semantic_contract = None;
        cell.spec
            .input_identity
            .as_mut()
            .expect("QG-6 input identity")
            .semantic_contract_sha256 = None;
        let EvidenceCellBody::Paired {
            paired,
            qg6_protocol,
            ..
        } = &mut cell.body
        else {
            unreachable!("QG-6 must be paired");
        };
        *qg6_protocol = None;
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
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority]),
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
        let (mut artifact, authority) = qg6_artifact();
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
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority]),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
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
            let (mut artifact, previous_authority) = qg6_artifact();
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
            let authority = reauthorize_qg6_test_artifact(&mut artifact, &previous_authority);
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

            let error =
                PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority])
                    .expect_err("fully re-sealed query mutation must fail verified reload");
            assert!(
                matches!(
                    error,
                    EvidenceArtifactError::InconsistentArtifact { ref reason }
                        if reason.contains("frozen normative query slice")
                ),
                "fully re-sealed mutation {mutation} failed outside the frozen anchor: {error}"
            );
        }
    }

    #[test]
    fn qg6_outer_reseal_rejects_invalid_class_supported_k_and_query_self_seal() {
        for mutation in ["class", "supported_k", "query_identity_sha256"] {
            let (mut artifact, authority) = qg6_artifact();
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
                    PerfEvidenceArtifact::load_verified_against_authorities(
                        &path,
                        &[],
                        &[&authority],
                    ),
                    Err(EvidenceArtifactError::InconsistentArtifact { .. })
                ),
                "invalid query identity field {mutation} escaped verification"
            );
        }
    }

    #[test]
    fn qg6_verified_load_rejects_fully_resealed_unsupported_contract_cutoff() {
        let (mut artifact, previous_authority) = qg6_artifact();
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
        let authority = reauthorize_qg6_test_artifact(&mut artifact, &previous_authority);

        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-resealed-unsupported-k99.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("outer-reseal unsupported cutoff"),
        )
        .expect("persist fully resealed unsupported cutoff");
        let error =
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority])
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
            let (mut artifact, authority) = qg6_artifact();
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
                    PerfEvidenceArtifact::load_verified_against_authorities(
                        &path,
                        &[],
                        &[&authority],
                    ),
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
            let (mut artifact, authority) = qg6_artifact();
            fully_reseal_qg6_result_receipt_mutation(&mut artifact, &authority, |receipt| {
                match mutation {
                    "underfilled_top_k" => receipt.total_count = 2,
                    "empty_document_id" => {
                        receipt.ordered_hits[0].document_id_sha256 =
                            lower_hex(&Sha256::digest(b""));
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
                }
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

            let error =
                PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority])
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
            let (mut artifact, authority) = qg6_artifact();
            unbind_test_artifact(&mut artifact);
            let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body else {
                unreachable!("QG-6 must be paired");
            };
            let binding = paired.effect_samples[0]
                .qg6_sample_binding
                .as_mut()
                .expect("authenticated QG-6 binding");
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
                    PerfEvidenceArtifact::load_verified_against_authorities(
                        &path,
                        &[],
                        &[&authority],
                    ),
                    Err(EvidenceArtifactError::InconsistentArtifact { .. })
                ),
                "authenticated binding field {mutation} escaped reload verification"
            );
        }
    }

    #[test]
    fn qg6_outer_reseal_rejects_authenticated_timing_leaf_mutation() {
        let (mut artifact, authority) = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body else {
            unreachable!("QG-6 must be paired");
        };
        let leaf = &mut paired.effect_samples[0]
            .qg6_sample_binding
            .as_mut()
            .expect("authenticated QG-6 binding")
            .timed_sample
            .timing_leaves[0];
        leaf.ended_ns = leaf
            .ended_ns
            .checked_add(1)
            .expect("bounded hostile latency");

        let directory = tempfile::tempdir().expect("QG-6 artifact directory");
        let path = directory.path().join("qg6-timing-leaf-mutation.json");
        fs::write(
            &path,
            artifact
                .sealed_json()
                .expect("outer-reseal timing leaf mutation"),
        )
        .expect("persist timing leaf mutation");
        let error =
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority])
                .expect_err("mutated authenticated QG-6 timing leaf must fail closed");
        assert!(
            matches!(
                error,
                EvidenceArtifactError::Malformed { ref reason }
                    if reason.contains(
                        "QG-6 timing leaves do not have the exact parent order and interval"
                    )
            ),
            "unexpected timing-leaf rejection: {error:?}"
        );
    }

    #[test]
    fn qg6_sealed_reload_rejects_one_groups_extra_balanced_pair() {
        let (mut artifact, authority) = qg6_artifact();
        unbind_test_artifact(&mut artifact);
        let identity = artifact.cells[0]
            .spec
            .input_identity
            .clone()
            .expect("QG-6 input identity");
        let contract = artifact.cells[0]
            .spec
            .qg6_semantic_contract
            .clone()
            .expect("QG-6 semantic contract");
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
        qg6_test_fixture::attach_stream(&mut extra, true, &identity, &contract);
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
        let error =
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority])
                .expect_err("extra QG-6 pair must fail verified reload");
        assert!(
            matches!(
                error,
                EvidenceArtifactError::InconsistentArtifact { ref reason }
                    if reason.contains("exact schedule multiplicity for all six roles")
            ),
            "unexpected extra-pair rejection: {error}"
        );
    }

    #[test]
    fn qg6_sealed_reload_rejects_a_missing_named_role_receipt() {
        let (mut artifact, authority) = qg6_artifact();
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
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority]),
            Err(EvidenceArtifactError::Malformed { .. })
        ));
    }

    #[test]
    fn qg6_verified_load_recomputes_hierarchical_null_from_raw_groups() {
        let (mut artifact, authority) = qg6_artifact();
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
            PerfEvidenceArtifact::load_verified_against_authorities(&path, &[], &[&authority]),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn qg6_partial_query_group_removal_is_rejected_as_inconsistent() {
        let (original, _authority) = qg6_artifact();
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
    fn qg6_hierarchical_null_is_retained_by_formal_tail_protocol() {
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
        // Ninety-five exact-identity pairs plus five tail pairs per query. The
        // flat paired estimator sees a null whose center is exactly 1.0 and
        // whose dispersion is small enough to stay Valid/Eligible, so nothing
        // in the ordinary inference can reject this fixture. Only the formal
        // true-leaf p99 can: five percent of the leaves sit at 125, which is
        // above the 99th percentile boundary, so the joint-tail null is the
        // sole rejection. The tail pairs are spread across the block order
        // rather than appended, so the flat null carries no order or drift
        // signal that could reject it for the wrong reason.
        let mut null_pairs = Vec::new();
        for group_id in QG6_QUERY_GROUP_IDS {
            for index in 0..100 {
                let treatment = if index % 20 == 19 { 125.0 } else { 100.0 };
                null_pairs.push((group_id, 100.0, treatment));
            }
        }
        let quill_null_pairs = QG6_QUERY_GROUP_IDS
            .into_iter()
            .flat_map(|group_id| [(group_id, 100.0, 100.0); 100])
            .collect::<Vec<_>>();
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        let mut quill_null = grouped_gauge_stream(&quill_null_pairs, 20_000, None);
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
        bind_samples_to_spec(&mut quill_null, &spec);
        let estimator_config = config();
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: identity.query_manifest_sha256.clone(),
                config_contract_sha256: identity.config_contract_sha256.clone(),
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            100,
            1,
            estimator_config.bootstrap_seed,
        )
        .expect("retained QG-6 schedule authority");
        for (samples, comparison) in [
            (&mut effect[..], Qg6Comparison::Effect),
            (&mut null[..], Qg6Comparison::TantivyNull),
            (&mut quill_null[..], Qg6Comparison::QuillNull),
        ] {
            qg6_test_fixture::attach_stream_against_schedule_authority(
                samples, comparison, &authority, identity, contract,
            );
        }

        let paired = estimate_paired_experiment(&effect, &null, &estimator_config)
            .expect("QG-6 paired estimate");
        assert_eq!(paired.status, PairedEvidenceStatus::Valid);
        assert_eq!(paired.claim_state, PairedClaimState::EligibleForDecision);
        assert!(paired.reasons.is_empty());

        let protocol = Qg6FormalProtocolEvidence::new_against_authority_fixture(
            &paired, quill_null, &authority, identity, contract,
        )
        .expect("construct retained QG-6 formal protocol");
        let mut cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        cell.attach_qg6_formal_protocol_against_authority(protocol, &policy(), &authority)
            .expect("attach retained QG-6 formal protocol");
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
        let no_claim = cell
            .reasons
            .iter()
            .filter(|reason| reason.severity == EvidenceSeverity::NoClaim)
            .collect::<Vec<_>>();
        assert!(
            !no_claim.is_empty(),
            "the formal joint-tail protocol must contribute NoClaim reasons: {:?}",
            cell.reasons
        );
        assert!(
            no_claim
                .iter()
                .all(|reason| reason.code.starts_with("qg6.joint_tail_")),
            "every NoClaim must originate from the formal joint-tail checks: {:?}",
            no_claim
        );
        assert!(
            no_claim
                .iter()
                .any(|reason| reason.code == "qg6.joint_tail_null_invalid"
                    && reason.message.contains("Tantivy/Tantivy")
                    && reason.message.contains("p99")),
            "the engineered Tantivy/Tantivy true-leaf p99 rejection must be present: {:?}",
            no_claim
        );
        assert!(
            !cell
                .reasons
                .iter()
                .any(|reason| reason.code == "qg6.tail_protocol_not_implemented")
        );

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            plan_binding(PerfGate::Qg6),
            policy(),
            evidence_provenance(PerfGate::Qg6),
            vec![cell],
        )
        .expect("QG-6 artifact");
        bind_test_identity_against_authorities(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-hierarchy-primary",
            &[],
            &[&authority],
        );
        assert!(!artifact.ratchet_admissible());
        let directory = tempfile::tempdir().expect("QG-6 hierarchy-native artifact directory");
        let paths = artifact
            .write_atomic_against_authorities(directory.path(), &[], &[&authority])
            .expect("seal hierarchy-native QG-6 artifact");
        let verified = PerfEvidenceArtifact::load_verified_against_authorities(
            &paths.json,
            &[],
            &[&authority],
        )
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
            for round in 0..100 {
                let (control, treatment) = match round % 10 {
                    0 | 3 | 6 => (1_000.0, 950.0),
                    1 | 4 | 7 => (10_000.0, 9_500.0),
                    _ => (1.0, 100_000.0),
                };
                effect_pairs.push((group_id, control, treatment));
            }
            null_pairs.extend([(group_id, 100.0, 100.0); 100]);
        }
        let quill_null_pairs = QG6_QUERY_GROUP_IDS
            .into_iter()
            .flat_map(|group_id| [(group_id, 100.0, 100.0); 100])
            .collect::<Vec<_>>();
        let mut effect = grouped_gauge_stream(&effect_pairs, 0, None);
        let mut null = grouped_gauge_stream(&null_pairs, 10_000, None);
        let mut quill_null = grouped_gauge_stream(&quill_null_pairs, 20_000, None);
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
        bind_samples_to_spec(&mut quill_null, &spec);
        let estimator_config = config();
        let authority = Qg6ScheduleAuthority::for_experiment(
            Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256.clone(),
                query_manifest_sha256: identity.query_manifest_sha256.clone(),
                config_contract_sha256: identity.config_contract_sha256.clone(),
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            100,
            1,
            estimator_config.bootstrap_seed,
        )
        .expect("retained QG-6 schedule authority");
        for (samples, comparison) in [
            (&mut effect[..], Qg6Comparison::Effect),
            (&mut null[..], Qg6Comparison::TantivyNull),
            (&mut quill_null[..], Qg6Comparison::QuillNull),
        ] {
            qg6_test_fixture::attach_stream_against_schedule_authority(
                samples, comparison, &authority, identity, contract,
            );
        }

        let paired = estimate_paired_experiment(&effect, &null, &estimator_config)
            .expect("QG-6 paired estimate");
        assert!(
            matches!(
                paired.status,
                PairedEvidenceStatus::InvalidExperiment
                    | PairedEvidenceStatus::ContradictorySummaries
            ),
            "fixture must remain non-admissible only through flat diagnostics: {:?}",
            paired.reasons
        );
        assert_eq!(paired.claim_state, PairedClaimState::NoDecision);
        assert!(
            paired
                .reasons
                .iter()
                .all(|reason| qg6_flat_inference_only(&reason.code)),
            "fixture accidentally carries a structural experiment failure: {:?}",
            paired.reasons
        );

        let protocol = Qg6FormalProtocolEvidence::new_against_authority_fixture(
            &paired, quill_null, &authority, identity, contract,
        )
        .expect("construct retained QG-6 formal protocol");
        let mut cell = EvidenceCell::evaluate(spec, paired, &policy()).expect("QG-6 evidence cell");
        cell.attach_qg6_formal_protocol_against_authority(protocol, &policy(), &authority)
            .expect("attach retained QG-6 formal protocol");
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

        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            plan_binding(PerfGate::Qg6),
            policy(),
            evidence_provenance(PerfGate::Qg6),
            vec![cell],
        )
        .expect("QG-6 artifact");
        bind_test_identity_against_authorities(
            &mut artifact,
            PerfGate::Qg6,
            b"qg6-threshold",
            "qg6-conflict-primary",
            &[],
            &[&authority],
        );
        assert_eq!(
            artifact.gate_status,
            EvidenceDecisionStatus::MeasuredProvisional
        );
        assert!(!artifact.has_exact_runnable_plan_coverage());
        assert!(
            !artifact.ratchet_admissible(),
            "one-cell fixture cannot satisfy the complete runnable matrix even when its cell is eligible"
        );
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
        bind_samples_to_spec(&mut effect, &spec);
        bind_samples_to_spec(&mut null, &spec);
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
        let experiment = valid_experiment_for_spec(&unproven, 1.10);
        let cell = EvidenceCell::evaluate(unproven, experiment, &policy()).expect("cell");
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
        let experiment = valid_experiment_for_spec(&proven, 1.10);
        let cell = EvidenceCell::evaluate(proven, experiment, &policy()).expect("cell");
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
            plan_binding(PerfGate::Qg10),
            policy(),
            evidence_provenance(PerfGate::Qg10),
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
        artifact
            .apply_gate_decision(EvidenceDecisionStatus::Quarantine)
            .expect("eligible evidence accepts a terminal decision before its scope changes");
        assert_eq!(
            artifact.gate_decision,
            Some(EvidenceDecisionStatus::Quarantine)
        );

        artifact.force_no_claim(
            "evidence.incomplete_gate_selection",
            "fixture-filtered pre-admission run",
        );

        assert_eq!(artifact.gate_status, EvidenceDecisionStatus::NoDecision);
        assert_eq!(artifact.gate_decision, None);
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
        wrong_gate.gate = PerfGate::Qg1;
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
    fn v5_rejects_resealed_applicability_binding_profile_plan_hash_and_gate_mutations() {
        let directory = tempfile::tempdir().expect("applicability mutation directory");

        let mut wrong_profile = provisional_artifact();
        unbind_test_artifact(&mut wrong_profile);
        wrong_profile.applicability_plan.profile = crate::MachineProfileKey::new(
            crate::HardwareClassId::TrjZen35995wx,
            crate::ExecutionProfileId::Smt2_128,
        )
        .expect("canonical alternate profile");
        let wrong_profile_path = directory.path().join("wrong-profile.json");
        fs::write(
            &wrong_profile_path,
            wrong_profile
                .sealed_json()
                .expect("reseal wrong-profile artifact"),
        )
        .expect("persist wrong-profile artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&wrong_profile_path),
            Err(EvidenceArtifactError::InconsistentArtifact { reason })
                if reason.contains("does not equal the frozen registry")
        ));

        let mut wrong_plan_hash = provisional_artifact();
        unbind_test_artifact(&mut wrong_plan_hash);
        wrong_plan_hash.applicability_plan.applicability_plan_sha256 = "0".repeat(64);
        let wrong_plan_hash_path = directory.path().join("wrong-plan-hash.json");
        fs::write(
            &wrong_plan_hash_path,
            wrong_plan_hash
                .sealed_json()
                .expect("reseal wrong-plan-hash artifact"),
        )
        .expect("persist wrong-plan-hash artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&wrong_plan_hash_path),
            Err(EvidenceArtifactError::InconsistentArtifact { reason })
                if reason.contains("does not equal the frozen registry")
        ));

        let mut wrong_gate = provisional_artifact();
        unbind_test_artifact(&mut wrong_gate);
        wrong_gate.applicability_plan.gate = PerfGate::Qg1;
        let wrong_gate_path = directory.path().join("wrong-plan-gate.json");
        fs::write(
            &wrong_gate_path,
            wrong_gate
                .sealed_json()
                .expect("reseal wrong-plan-gate artifact"),
        )
        .expect("persist wrong-plan-gate artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&wrong_gate_path),
            Err(EvidenceArtifactError::InconsistentArtifact { reason })
                if reason.contains("instead of artifact gate")
        ));
    }

    #[test]
    fn v5_rejects_resealed_role_and_not_applicable_cell_mutations() {
        let directory = tempfile::tempdir().expect("applicability cell mutation directory");

        let mut wrong_role = provisional_artifact();
        unbind_test_artifact(&mut wrong_role);
        wrong_role.cells[0].spec.role = EvidenceRole::Diagnostic;
        let wrong_role_path = directory.path().join("wrong-role.json");
        fs::write(
            &wrong_role_path,
            wrong_role
                .sealed_json()
                .expect("reseal wrong-role artifact"),
        )
        .expect("persist wrong-role artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&wrong_role_path),
            Err(EvidenceArtifactError::InconsistentArtifact { reason })
                if reason.contains("applicability plan requires Required")
        ));

        let (cell, expected_authority) = provisional_cell_with_authority();
        let mut not_applicable = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![cell],
        )
        .expect("applicable QG-1 artifact");
        not_applicable.cells[0].spec.fixture = "bulk/tiny/96/positions_on".to_owned();
        not_applicable.cells[0].cell_id = format!(
            "{}/{}/{}",
            PerfGate::Qg1,
            not_applicable.cells[0].spec.fixture,
            not_applicable.cells[0].spec.metric
        );
        let not_applicable_path = directory.path().join("not-applicable-cell.json");
        fs::write(
            &not_applicable_path,
            not_applicable
                .sealed_json()
                .expect("reseal not-applicable artifact"),
        )
        .expect("persist not-applicable artifact");
        assert!(matches!(
            PerfEvidenceArtifact::load_verified_against_qg1_authorities(
                &not_applicable_path,
                &[&expected_authority],
            ),
            Err(EvidenceArtifactError::InconsistentArtifact { reason })
                if reason.contains("is not applicable to profile")
        ));
    }

    #[test]
    fn partial_plan_evidence_is_durable_but_never_ratchet_admissible() {
        let (cell, expected_authority) = provisional_cell_with_authority();
        let mut partial_required = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![cell],
        )
        .expect("partial required QG-1 evidence");
        let threshold_bytes = b"qg1-partial-threshold";
        let prebinding_bytes = seal_unbound_artifact_against_qg1_authorities(
            &mut partial_required,
            &[&expected_authority],
        );
        let identity = admitted_identity(
            PerfGate::Qg1,
            threshold_bytes,
            &prebinding_bytes,
            "qg1-partial-primary",
        );
        partial_required
            .bind_machine_class_identity_and_seal_against_qg1_authorities(
                identity,
                threshold_bytes,
                &prebinding_bytes,
                &[&expected_authority],
            )
            .expect("bind and seal partial required evidence");
        partial_required
            .verify_integrity_against_qg1_authorities(&[&expected_authority])
            .expect("partial required evidence remains durable");
        assert_eq!(
            partial_required.gate_status,
            EvidenceDecisionStatus::NoDecision,
            "the QG-1 fold refuses a required cell with no exact incumbent-screen coverage"
        );
        assert!(
            partial_required.admission_no_claim.is_none(),
            "the refusal comes from the production fold, not an injected no-claim override"
        );
        assert!(
            partial_required
                .reasons
                .iter()
                .any(|reason| reason.code == "evidence.qg1_incumbent_screen_missing"),
            "the fold must name the missing QG-1 incumbent-screen coverage: {:?}",
            partial_required.reasons
        );
        assert!(!partial_required.ratchet_admissible());
        assert!(matches!(
            partial_required.apply_gate_decision(EvidenceDecisionStatus::Allow),
            Err(EvidenceArtifactError::NotClaimEligible)
        ));

        let mut diagnostic_spec = cell_spec(PerfGate::Qg1, EvidenceRole::Diagnostic);
        diagnostic_spec.fixture = "tokenize_only/medium".to_owned();
        diagnostic_spec.metric = "tokenize_docs_per_second".to_owned();
        diagnostic_spec.unit = perf_metric_unit(&diagnostic_spec.metric).to_owned();
        let diagnostic_cell = EvidenceCell::evaluate(
            diagnostic_spec.clone(),
            valid_experiment_for_spec(&diagnostic_spec, 1.10),
            &policy(),
        )
        .expect("partial diagnostic cell");
        let partial_diagnostic = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![diagnostic_cell],
        )
        .expect("partial diagnostic QG-1 evidence");
        assert_eq!(
            partial_diagnostic.gate_status,
            EvidenceDecisionStatus::NoDecision
        );
        assert!(!partial_diagnostic.ratchet_admissible());

        let directory = tempfile::tempdir().expect("partial evidence directory");
        let paths = partial_required
            .write_atomic_against_qg1_authorities(directory.path(), &[&expected_authority])
            .expect("persist partial required no-claim evidence");
        let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
            &paths.json,
            &[&expected_authority],
        )
        .expect("partial required no-claim evidence remains loadable");
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(!reloaded.ratchet_admissible());
        assert_eq!(
            reloaded.cells, partial_required.cells,
            "the no-claim artifact retains its measured raw-cell evidence exactly"
        );
    }

    #[test]
    fn schema_rejects_missing_required_fields() {
        let artifact = provisional_artifact();
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = artifact.write_atomic(dir.path()).expect("write");
        let contents = fs::read_to_string(&paths.json).expect("read");

        for field in [
            "applicability_plan",
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
    fn v5_rejects_class_only_applicability_identity_even_when_outer_seal_is_valid() {
        let artifact = provisional_artifact();
        let mut value = serde_json::to_value(artifact).expect("artifact JSON");
        value["applicability_plan"] = serde_json::json!({
            "hardware_class_id": "trj-zen3-5995wx"
        });
        let bytes = reseal_json_value(value);
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&bytes),
            Err(EvidenceArtifactError::Malformed { .. })
        ));
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
            applicability_plan: None,
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
    fn prior_v3_evidence_is_read_only_and_has_no_current_upgrade_path() {
        let mut legacy_v3 = provisional_artifact();
        legacy_v3.schema_version = "quill-perf-evidence-v3".to_owned();
        let directory = tempfile::tempdir().expect("legacy v3 directory");
        let path = directory.path().join("legacy-v3.evidence.json");
        fs::write(
            &path,
            legacy_v3
                .sealed_json()
                .expect("seal exact legacy-v3-shaped evidence"),
        )
        .expect("persist legacy v3 evidence");

        assert!(matches!(
            PerfEvidenceArtifact::load_verified(&path),
            Err(EvidenceArtifactError::SchemaMismatch { found })
                if found == "quill-perf-evidence-v3"
        ));
        assert!(matches!(
            load_legacy_gate_artifact_v3(&path),
            Err(EvidenceArtifactError::SchemaMismatch { found })
                if found == "quill-perf-evidence-v3"
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
            let scope = gauge_scope();
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
            let scope = gauge_scope();
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
        let mut provenance = evidence_provenance(PerfGate::Qg1);
        provenance.peak_rss = PeakRssEvidence {
            method: "unsupported".to_owned(),
            bytes: Some(0),
        };
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg1,
                plan_binding(PerfGate::Qg1),
                policy(),
                provenance,
                vec![provisional_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));

        let mut zeroed = evidence_provenance(PerfGate::Qg1);
        zeroed.peak_rss = PeakRssEvidence {
            method: "linux_vmhwm".to_owned(),
            bytes: Some(0),
        };
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg1,
                plan_binding(PerfGate::Qg1),
                policy(),
                zeroed,
                vec![provisional_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));
    }

    #[test]
    fn execution_provenance_must_equal_the_reconstructed_plan_envelope() {
        let mut capacity_drift = evidence_provenance(PerfGate::Qg1);
        capacity_drift.machine.execution.execution_capacity = 63;
        capacity_drift.machine.execution.max_exercised_cell_width = 63;
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg1,
                plan_binding(PerfGate::Qg1),
                policy(),
                capacity_drift,
                vec![provisional_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("capacity/maximum/selected-width envelope")
        ));

        let mut maximum_drift = evidence_provenance(PerfGate::Qg2);
        maximum_drift.machine.execution.max_exercised_cell_width = 2;
        assert!(matches!(
            PerfEvidenceArtifact::assemble(
                PerfGate::Qg2,
                plan_binding(PerfGate::Qg2),
                policy(),
                maximum_drift,
                vec![provisional_qg2_cell()],
            ),
            Err(EvidenceArtifactError::InvalidProvenance { reason })
                if reason.contains("capacity/maximum/selected-width envelope")
        ));
    }

    #[test]
    fn verified_loader_requires_the_exact_canonical_pretty_bytes() {
        let artifact = provisional_artifact();
        let canonical = artifact
            .sealed_json()
            .expect("canonical sealed evidence bytes");
        let value: serde_json::Value =
            serde_json::from_str(&canonical).expect("canonical evidence JSON");
        let compact = serde_json::to_vec(&value).expect("compact equivalent evidence JSON");
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&compact),
            Err(EvidenceArtifactError::Malformed { reason })
                if reason.contains("exact canonical pretty JSON")
        ));
    }

    #[test]
    fn coherent_receipt_reseals_cannot_forge_runner_build_or_machine_projection() {
        for mutation in [
            "git_revision",
            "git_dirty_worktree",
            "cargo_lock",
            "executable",
            "command",
            "environment",
            "runtime_isa",
            "affinity",
            "governor",
            "capacity_text",
            "machine_fingerprint",
        ] {
            let mut artifact = provisional_artifact();
            match mutation {
                "git_revision" => artifact.provenance.build.git_revision = "9".repeat(40),
                "git_dirty_worktree" => {
                    artifact.provenance.build.git_dirty = true;
                    artifact.provenance.build.worktree_state_sha256 = Some("9".repeat(64));
                }
                "cargo_lock" => {
                    artifact.provenance.build.cargo_lock_sha256 = Some("9".repeat(64));
                }
                "executable" => {
                    let forged = "9".repeat(64);
                    artifact
                        .provenance
                        .build
                        .executable_sha256
                        .clone_from(&forged);
                    let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body
                    else {
                        unreachable!("QG-2 fixture must be paired");
                    };
                    paired.provenance.executable_sha256.clone_from(&forged);
                    for sample in paired
                        .effect_samples
                        .iter_mut()
                        .chain(&mut paired.null_samples)
                    {
                        sample.provenance.executable_sha256.clone_from(&forged);
                    }
                }
                "command" => artifact.provenance.build.command_sha256 = "9".repeat(64),
                "environment" => {
                    artifact.provenance.build.environment_sha256 = Some("9".repeat(64));
                }
                "runtime_isa" => {
                    artifact
                        .provenance
                        .machine
                        .execution
                        .runtime_detected_isa
                        .remove(0);
                }
                "affinity" => {
                    artifact
                        .provenance
                        .machine
                        .execution
                        .cpu_affinity_allowed_list = Some("1-64".to_owned());
                    artifact.provenance.machine.execution.affinity_or_cpuset_cap =
                        Some("Cpus_allowed_list=1-64 (64 of 128 host logical threads)".to_owned());
                }
                "governor" => {
                    artifact.provenance.machine.cpu_governor = Some("powersave".to_owned());
                }
                "capacity_text" => {
                    artifact.provenance.machine.execution.affinity_or_cpuset_cap =
                        Some("available_parallelism=64 of 128 host logical threads".to_owned());
                }
                "machine_fingerprint" => {
                    let forged = format!("{TEST_MACHINE_FINGERPRINT}-forged");
                    artifact.provenance.machine.fingerprint.clone_from(&forged);
                    let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body
                    else {
                        unreachable!("QG-2 fixture must be paired");
                    };
                    paired.provenance.worker_id.clone_from(&forged);
                    for sample in paired
                        .effect_samples
                        .iter_mut()
                        .chain(&mut paired.null_samples)
                    {
                        sample.provenance.worker_id.clone_from(&forged);
                    }
                }
                _ => unreachable!("bounded mutation table"),
            }
            let bytes = coherently_bind_and_reseal(
                artifact,
                b"qg2-threshold",
                &format!("qg2-hostile-{mutation}"),
            );
            let result = PerfEvidenceArtifact::from_verified_slice(&bytes);
            assert!(
                matches!(
                    &result,
                    Err(EvidenceArtifactError::InvalidProvenance { .. })
                ),
                "coherently receipt-bound {mutation} mutation was not rejected: {result:?}"
            );
            if mutation == "executable" {
                assert!(
                    matches!(
                        &result,
                        Err(EvidenceArtifactError::InvalidProvenance { reason })
                            if reason.contains(
                                "evidence build identity differs from the verified runner receipt"
                            )
                    ),
                    "coherent executable mutation did not reach the receipt projection: {result:?}"
                );
            }
        }
    }

    #[test]
    fn coherent_receipt_reseals_cannot_forge_cell_scope_provenance_or_estimator() {
        for mutation in [
            "scope_id",
            "scope_version",
            "scope_semantics",
            "scope_unit",
            "run_id",
            "executable",
            "corpus",
            "worker",
            "build_profile",
            "estimator_config",
        ] {
            let mut artifact = provisional_artifact();
            let EvidenceCellBody::Paired { paired, .. } = &mut artifact.cells[0].body else {
                unreachable!("QG-2 fixture must be paired");
            };
            match mutation {
                "scope_id" => paired.scope.operation_id.push_str(".forged"),
                "scope_version" => paired.scope.version += 1,
                "scope_semantics" => {
                    paired.scope.semantics = PerfMetricSemantics::GaugeLowerIsBetter;
                }
                "scope_unit" => paired.scope.unit = "docs/second".to_owned(),
                "run_id" => paired.provenance.run_id = "forged-run".to_owned(),
                "executable" => paired.provenance.executable_sha256 = "9".repeat(64),
                "corpus" => paired.provenance.corpus_sha256 = "9".repeat(64),
                "worker" => paired.provenance.worker_id = "forged-worker".to_owned(),
                "build_profile" => paired.provenance.build_profile = "forged".to_owned(),
                "estimator_config" => paired.config.bootstrap_resamples += 1,
                _ => unreachable!("bounded mutation table"),
            }
            let scope = paired.scope.clone();
            let provenance = paired.provenance.clone();
            for sample in paired
                .effect_samples
                .iter_mut()
                .chain(&mut paired.null_samples)
            {
                sample.scope.clone_from(&scope);
                sample.provenance.clone_from(&provenance);
            }
            let bytes = coherently_bind_and_reseal(
                artifact,
                b"qg2-threshold",
                &format!("qg2-cell-hostile-{mutation}"),
            );
            let result = PerfEvidenceArtifact::from_verified_slice(&bytes);
            assert!(
                matches!(
                    &result,
                    Err(EvidenceArtifactError::InvalidProvenance { .. })
                ),
                "coherently receipt-bound {mutation} mutation was not rejected: {result:?}"
            );
        }
    }

    #[test]
    fn coherent_reseals_cannot_change_policy_unit_or_configured_widths() {
        let mut policy_drift = provisional_artifact();
        policy_drift.policy.reconciliation_tolerance_log = 1.30_f64.ln();
        let bytes =
            coherently_bind_and_reseal(policy_drift, b"qg2-threshold", "qg2-hostile-policy");
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&bytes),
            Err(EvidenceArtifactError::InvalidPolicy { .. })
        ));

        let mut unit_drift = provisional_artifact();
        unit_drift.cells[0].spec.unit = "docs/second".to_owned();
        let bytes = coherently_bind_and_reseal(unit_drift, b"qg2-threshold", "qg2-hostile-unit");
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&bytes),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));

        let mut width_drift = provisional_artifact();
        width_drift
            .provenance
            .machine
            .execution
            .configured_engine_thread_widths = vec![2];
        let bytes =
            coherently_bind_and_reseal(width_drift, b"qg2-threshold", "qg2-hostile-width-envelope");
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&bytes),
            Err(EvidenceArtifactError::InvalidProvenance { .. })
        ));

        let (cell, expected_authority) = provisional_cell_with_authority();
        let mut witness_drift = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![cell],
        )
        .expect("QG-1 witness mutation fixture");
        let witness = witness_drift.cells[0]
            .spec
            .concurrency_witness
            .as_mut()
            .expect("QG-1 concurrency witness");
        witness.configured_threads = 2;
        for observation in &mut witness.observations {
            observation.min_observed_worker_pool_threads = 2;
            observation.max_observed_worker_pool_threads = 2;
        }
        witness_drift
            .provenance
            .machine
            .execution
            .configured_engine_thread_widths = vec![2];
        let bytes = coherently_bind_and_reseal(
            witness_drift,
            b"qg1-threshold",
            "qg1-hostile-concurrency-witness",
        );
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
                &bytes,
                &[&expected_authority],
            ),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    #[test]
    fn gate_decisions_only_apply_to_eligible_evidence() {
        for decision in [
            EvidenceDecisionStatus::Allow,
            EvidenceDecisionStatus::Quarantine,
            EvidenceDecisionStatus::Block,
        ] {
            let mut artifact = provisional_artifact();
            artifact
                .apply_gate_decision(decision)
                .expect("eligible terminal decision");
            assert_eq!(artifact.gate_decision, Some(decision));

            let dir = tempfile::tempdir().expect("tempdir");
            let paths = artifact.write_atomic(dir.path()).expect("write");
            let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("reload");
            assert_eq!(reloaded.gate_decision, Some(decision));
        }

        let mut artifact = provisional_artifact();
        assert!(matches!(
            artifact.apply_gate_decision(EvidenceDecisionStatus::MeasuredProvisional),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));
    }

    fn qg1_screen_cell_for(cell_id: &str) -> PerfCellSpec {
        PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|cell| format!("{}/{}/{}", PerfGate::Qg1, cell.fixture, cell.metric) == cell_id)
            .cloned()
            .expect("the screened cell is a frozen QG-1 matrix cell")
    }

    fn qg1_semantic_contract() -> Qg1TantivySemanticContract {
        Qg1TantivySemanticContract {
            tantivy_version: crate::perf::QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION.to_owned(),
            schema_sha256: "1".repeat(64),
            analyzer_sha256: "2".repeat(64),
            indexed_fields_sha256: "3".repeat(64),
            merge_policy_sha256: "4".repeat(64),
            visibility_sha256: "5".repeat(64),
            searchable_terminal_scope_sha256: "6".repeat(64),
            durability_sha256: "7".repeat(64),
            quill_config_sha256: "8".repeat(64),
        }
    }

    /// A screen that preregistered its candidate universe and retained no
    /// pilots at all. This is the incomplete-screen outcome the H3 contract
    /// names, and it carries no authority-bearing component, so it is the one
    /// screen shape whose evidence is constructible without a live producer.
    fn qg1_no_decision_screen_evidence(cell_id: &str) -> Qg1IncumbentScreenEvidence {
        let cell = qg1_screen_cell_for(cell_id);
        let semantic_contract = qg1_semantic_contract();
        let plan = crate::perf::Qg1TantivyIncumbentScreenPlan::new(
            test_profile(),
            1,
            vec![1],
            &cell,
            64_000,
        )
        .expect("QG-1 incumbent screen plan");
        let screen = Qg1TantivyIncumbentScreen::screen(&cell, plan, &semantic_contract, Vec::new())
            .expect("QG-1 incumbent screen");
        assert!(
            screen.selected_candidate.is_none() && screen.no_decision_reason.is_some(),
            "a screen with no retained pilots is a valid NoDecision"
        );
        Qg1IncumbentScreenEvidence {
            cell_id: cell_id.to_owned(),
            semantic_contract,
            screen,
            decision: None,
        }
    }

    /// The exact projection an artifact's required engine cells demand: one
    /// screen each, in canonical `cell_id` order.
    fn qg1_screen_projection(artifact: &PerfEvidenceArtifact) -> Vec<Qg1IncumbentScreenEvidence> {
        let mut cell_ids = artifact
            .cells
            .iter()
            .filter(|cell| {
                cell.spec.gate == PerfGate::Qg1 && cell.spec.role == EvidenceRole::Required
            })
            .map(|cell| cell.cell_id.clone())
            .collect::<Vec<_>>();
        cell_ids.sort();
        assert!(
            !cell_ids.is_empty(),
            "the screen projection fixture needs at least one required engine cell"
        );
        cell_ids
            .iter()
            .map(|cell_id| qg1_no_decision_screen_evidence(cell_id))
            .collect()
    }

    fn qg1_screen_artifact_with_authority() -> (PerfEvidenceArtifact, Qg1ExpectedAuthority) {
        let (cell, expected_authority) = provisional_cell_with_authority();
        let artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            plan_binding(PerfGate::Qg1),
            policy(),
            evidence_provenance(PerfGate::Qg1),
            vec![cell],
        )
        .expect("QG-1 screen-bearing artifact");
        (artifact, expected_authority)
    }

    fn qg1_screen_artifact() -> PerfEvidenceArtifact {
        qg1_screen_artifact_with_authority().0
    }

    /// Planted omission negative: QG-1 evidence that does not screen every
    /// required engine cell is inadmissible, and omitting the projection is not
    /// a way around the screen. Non-QG-1 evidence keeps its exact prior
    /// admissibility.
    #[test]
    fn omitted_qg1_incumbent_screen_is_no_decision_and_never_ratchets() {
        let (mut omitted, expected_authority) = qg1_screen_artifact_with_authority();
        assert!(omitted.qg1_incumbent_screens.is_empty());
        assert_eq!(
            omitted.gate_status,
            EvidenceDecisionStatus::NoDecision,
            "QG-1 evidence without an incumbent screen can support no claim"
        );
        assert!(
            omitted
                .reasons
                .iter()
                .any(|reason| reason.code == "evidence.qg1_incumbent_screen_missing"),
            "the fold names the omission: {:?}",
            omitted.reasons
        );
        assert!(!omitted.ratchet_admissible());
        assert!(matches!(
            omitted.apply_gate_decision(EvidenceDecisionStatus::Allow),
            Err(EvidenceArtifactError::NotClaimEligible)
        ));

        let directory = tempfile::tempdir().expect("omitted-screen evidence directory");
        let paths = omitted
            .write_atomic_against_qg1_authorities(directory.path(), &[&expected_authority])
            .expect("evidence with no screen still persists durably");
        let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
            &paths.json,
            &[&expected_authority],
        )
        .expect("reload evidence with no screen");
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(!reloaded.ratchet_admissible());

        assert!(
            provisional_artifact().ratchet_admissible(),
            "the QG-1 omission rule must not change any other gate"
        );
    }

    /// The attached projection is durable and is *consumed*: one incomplete
    /// screen folds the whole gate to no-claim, which is what stops a
    /// convenient Tantivy arm from headlining before every engine cell has
    /// frozen its incumbent.
    #[test]
    fn attached_qg1_incumbent_screen_is_durable_and_forces_no_decision() {
        let (mut artifact, expected_authority) = qg1_screen_artifact_with_authority();
        let screens = qg1_screen_projection(&artifact);
        artifact
            .attach_qg1_incumbent_screens(screens.clone())
            .expect("attach the QG-1 incumbent screen projection");
        assert_eq!(
            artifact.gate_status,
            EvidenceDecisionStatus::NoDecision,
            "an incomplete incumbent screen yields NoDecision"
        );
        assert!(
            artifact
                .reasons
                .iter()
                .any(|reason| reason.code == "evidence.qg1_incumbent_screen_no_decision"),
            "the fold records why the screen refused: {:?}",
            artifact.reasons
        );
        assert!(!artifact.ratchet_admissible());
        assert!(matches!(
            artifact.apply_gate_decision(EvidenceDecisionStatus::Allow),
            Err(EvidenceArtifactError::NotClaimEligible)
        ));

        let directory = tempfile::tempdir().expect("QG-1 screen evidence directory");
        assert!(
            matches!(
                artifact.write_atomic(directory.path()),
                Err(EvidenceArtifactError::InvalidProvenance { .. })
            ),
            "screen-bearing evidence may not be persisted through the authority-free writer"
        );

        let paths = artifact
            .write_atomic_against_qg1_authorities(directory.path(), &[&expected_authority])
            .expect("persist the screen-bearing artifact");
        let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
            &paths.json,
            &[&expected_authority],
        )
        .expect("reload the screen-bearing artifact");
        assert_eq!(
            reloaded.qg1_incumbent_screens, screens,
            "the complete screen projection survives write and reload exactly"
        );
        assert_eq!(reloaded.gate_status, EvidenceDecisionStatus::NoDecision);
        assert!(!reloaded.ratchet_admissible());
    }

    /// Planted width negatives. A shipping-auto selection has no observable
    /// materialized writer width, so it is a `NoDecision` and never an excuse for
    /// a relaxed witness; and a frozen fixed width the cell's witness does not
    /// prove was materialized is rejected rather than accepted.
    #[test]
    fn qg1_screen_selected_width_must_be_observable_and_materialized() {
        let baseline = qg1_screen_artifact();
        let screen = qg1_screen_projection(&baseline)
            .into_iter()
            .next()
            .expect("the projection screens one required cell");
        let shipping_auto = screen
            .screen
            .candidates
            .first()
            .expect("the preregistered universe leads with the shipping-auto arm")
            .clone();
        assert_eq!(
            shipping_auto.writer_mode,
            Qg1TantivyWriterMode::ShippingAuto,
            "the first preregistered candidate is the shipping-auto arm"
        );
        let fixed = screen
            .screen
            .candidates
            .iter()
            .find(|candidate| matches!(candidate.writer_mode, Qg1TantivyWriterMode::Fixed { .. }))
            .expect("the preregistered universe carries a fixed-width arm")
            .clone();
        let Qg1TantivyWriterMode::Fixed {
            writer_threads: frozen_width,
        } = fixed.writer_mode
        else {
            unreachable!("the fixed arm was matched above");
        };

        let mut unobservable = screen.clone();
        unobservable.screen.selected_candidate = Some(shipping_auto);
        unobservable.screen.no_decision_reason = None;
        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(vec![unobservable]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a shipping-auto selection has an unobservable width and must be NoDecision"
        );

        let mut selected = screen;
        selected.screen.selected_candidate = Some(fixed);
        selected.screen.no_decision_reason = None;
        let mut mismatched = qg1_screen_artifact();
        let witness = mismatched
            .cells
            .first_mut()
            .expect("the QG-1 artifact carries its required engine cell")
            .spec
            .concurrency_witness
            .as_mut()
            .expect("required QG-1 cells carry a concurrency witness");
        let unmaterialized = frozen_width.saturating_add(1);
        for observation in &mut witness.observations {
            if observation.engine == PerfConcurrencyEngine::Tantivy {
                observation.min_observed_worker_pool_threads = unmaterialized;
                observation.max_observed_worker_pool_threads = unmaterialized;
            }
        }
        assert!(
            matches!(
                mismatched.attach_qg1_incumbent_screens(vec![selected]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a frozen Tantivy width the witness never materialized must reject"
        );
    }

    /// The screened Tantivy arm may legitimately run at a different width from
    /// the Quill width its cell was configured with. The witness admits that
    /// for QG-1, and the screen is what pins it to the exact frozen value —
    /// which is why a differing width constructs here and a wrong one still
    /// rejects. This drives the binding directly because a selected screen
    /// otherwise requires a decision object that only a live producer can seal.
    #[test]
    fn qg1_screen_binds_a_tantivy_width_that_differs_from_the_configured_quill_width() {
        let mut spec = cell_spec(PerfGate::Qg1, EvidenceRole::Required);
        let witness = spec
            .concurrency_witness
            .as_mut()
            .expect("required QG-1 cells carry a concurrency witness");
        let configured = witness.configured_threads;
        let screened_width = configured.saturating_add(2);
        for observation in &mut witness.observations {
            if observation.engine == PerfConcurrencyEngine::Tantivy {
                observation.min_observed_worker_pool_threads = screened_width;
                observation.max_observed_worker_pool_threads = screened_width;
            }
        }
        let cell = EvidenceCell::evaluate(
            spec.clone(),
            valid_experiment_for_spec(&spec, 1.10),
            &policy(),
        )
        .expect("QG-1 admits a screened Tantivy width that differs from the Quill width");

        let screen = qg1_screen_projection(&qg1_screen_artifact())
            .into_iter()
            .next()
            .expect("the projection screens one required cell");
        let mut candidate = screen
            .screen
            .candidates
            .iter()
            .find(|candidate| matches!(candidate.writer_mode, Qg1TantivyWriterMode::Fixed { .. }))
            .expect("the preregistered universe carries a fixed-width arm")
            .clone();
        candidate.writer_mode = Qg1TantivyWriterMode::Fixed {
            writer_threads: screened_width,
        };
        let mut selected = screen;
        selected.screen.selected_candidate = Some(candidate.clone());
        selected.screen.no_decision_reason = None;
        selected
            .verify_selected_width_witness(&cell)
            .expect("the screen binds Tantivy to exactly the width it froze");

        candidate.writer_mode = Qg1TantivyWriterMode::Fixed {
            writer_threads: configured,
        };
        let mut mismatched = selected;
        mismatched.screen.selected_candidate = Some(candidate);
        assert!(
            matches!(
                mismatched.verify_selected_width_witness(&cell),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a frozen width the witness never materialized still rejects"
        );
    }

    /// QG-8 keeps its exact prior both-equal witness contract: the QG-1
    /// division of labour never reaches it.
    #[test]
    fn qg8_concurrency_witness_contract_is_unchanged() {
        let spec = cell_spec(PerfGate::Qg8, EvidenceRole::Required);
        let witness = spec
            .concurrency_witness
            .as_ref()
            .expect("QG-8 scaling cells carry a concurrency witness");
        assert!(
            witness.observations.iter().all(|observation| {
                observation.min_observed_worker_pool_threads == witness.configured_threads
                    && observation.max_observed_worker_pool_threads == witness.configured_threads
            }),
            "QG-8 requires both engines at the configured width"
        );
        let mut relaxed = spec.clone();
        let relaxed_witness = relaxed
            .concurrency_witness
            .as_mut()
            .expect("QG-8 witness is present");
        for observation in &mut relaxed_witness.observations {
            if observation.engine == PerfConcurrencyEngine::Tantivy {
                observation.min_observed_worker_pool_threads =
                    relaxed_witness.configured_threads.saturating_add(1);
                observation.max_observed_worker_pool_threads =
                    relaxed_witness.configured_threads.saturating_add(1);
            }
        }
        assert!(
            matches!(
                EvidenceCell::evaluate(
                    relaxed.clone(),
                    valid_experiment_for_spec(&relaxed, 1.10),
                    &policy()
                ),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "QG-8 must still reject a Tantivy width that differs from the configured width"
        );
    }

    /// Planted coverage negatives: the projection must name every required
    /// engine cell exactly once. A duplicate, an extra screen for a cell this
    /// artifact never measured, and an empty projection all reject, and a
    /// persisted artifact whose projection was edited fails verification.
    #[test]
    fn qg1_incumbent_screen_coverage_must_be_exact() {
        let (artifact, expected_authority) = qg1_screen_artifact_with_authority();
        let screens = qg1_screen_projection(&artifact);

        let mut duplicated = screens.clone();
        duplicated.push(
            screens
                .first()
                .expect("the projection screens one required cell")
                .clone(),
        );
        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(duplicated),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a duplicated cell_id is not canonical coverage"
        );

        let mut extra = screens.clone();
        let unscreened = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .map(|cell| format!("{}/{}/{}", PerfGate::Qg1, cell.fixture, cell.metric))
            .find(|cell_id| screens.iter().all(|screen| &screen.cell_id != cell_id))
            .expect("the frozen QG-1 matrix has more than one cell");
        extra.push(qg1_no_decision_screen_evidence(&unscreened));
        extra.sort_by(|left, right| left.cell_id.cmp(&right.cell_id));
        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(extra),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "screening a cell this artifact never measured is extra coverage, not bonus coverage"
        );

        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(Vec::new()),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "an empty projection cannot cover a required engine cell"
        );

        let mut attached = artifact;
        attached
            .attach_qg1_incumbent_screens(screens)
            .expect("attach the canonical projection");
        let mut dropped = attached;
        dropped.qg1_incumbent_screens.clear();
        // Attaching deliberately clears the content seal, and dropping the
        // projection changes the bytes again, so an unsealed object stops at
        // the hash gate before the fold recomputation can adjudicate it.
        // Resealing hash-consistently is what makes the assertion below
        // exercise the coverage/refold rule it names rather than the seal.
        let dropped: PerfEvidenceArtifact = serde_json::from_str(
            &dropped
                .sealed_json()
                .expect("reseal the dropped projection"),
        )
        .expect("the resealed dropped projection re-parses");
        assert!(
            matches!(
                dropped.verify_integrity_against_qg1_authorities(&[&expected_authority]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "an artifact whose projection was dropped after folding must not verify"
        );
    }

    /// Planted negatives on the attachment boundary: a screen that is neither a
    /// selection nor a valid `NoDecision`, one that claims both, one that selects
    /// without its decision, and one attached to evidence that never measured
    /// a QG-1 incumbent at all.
    #[test]
    fn incomplete_or_foreign_qg1_incumbent_screens_fail_closed() {
        let screen = qg1_screen_projection(&qg1_screen_artifact())
            .into_iter()
            .next()
            .expect("the projection screens one required cell");

        let mut foreign = provisional_artifact();
        assert_eq!(foreign.gate, PerfGate::Qg2);
        assert!(
            matches!(
                foreign.attach_qg1_incumbent_screens(vec![screen.clone()]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "non-QG-1 evidence can never carry a QG-1 incumbent screen"
        );
        assert!(foreign.qg1_incumbent_screens.is_empty());

        let mut neither = screen.clone();
        neither.screen.no_decision_reason = None;
        assert!(matches!(
            qg1_screen_artifact().attach_qg1_incumbent_screens(vec![neither]),
            Err(EvidenceArtifactError::InconsistentArtifact { .. })
        ));

        let mut both = screen.clone();
        both.screen.selected_candidate = Some(
            both.screen
                .candidates
                .first()
                .expect("preregistered candidate universe")
                .clone(),
        );
        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(vec![both.clone()]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a screen cannot both select a candidate and declare NoDecision"
        );

        let mut selected_without_decision = both;
        selected_without_decision.screen.no_decision_reason = None;
        assert!(
            matches!(
                qg1_screen_artifact().attach_qg1_incumbent_screens(vec![selected_without_decision]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a selected incumbent must carry its same-invocation decision evidence"
        );

        let mut unknown_cell = screen;
        unknown_cell.cell_id = "qg1/bulk/nonexistent/1/positions_on/docs_per_second".to_owned();
        assert!(
            matches!(
                unknown_cell.verify_against_qg1_authorities(&[], &policy(), &[]),
                Err(EvidenceArtifactError::InconsistentArtifact { .. })
            ),
            "a screen naming a cell outside the frozen matrix cannot be verified"
        );
    }

    /// A non-QG-1 artifact is byte-identical to what it was before the field
    /// existed: the absent screen is skipped on serialization, so its seal and
    /// its canonical bytes are unchanged.
    #[test]
    fn absent_qg1_incumbent_screen_leaves_non_qg_artifacts_exact() {
        let artifact = provisional_artifact();
        assert!(artifact.qg1_incumbent_screens.is_empty());
        let json = artifact.canonical_json().expect("canonical JSON");
        assert!(
            !json.contains("qg1_incumbent_screen"),
            "an absent screen must not appear in persisted bytes"
        );

        let directory = tempfile::tempdir().expect("non-QG evidence directory");
        let paths = artifact
            .write_atomic(directory.path())
            .expect("the authority-free writer still serves non-QG evidence");
        // `provisional_artifact` binds a runner identity, which intentionally
        // clears the content seal, while the writer seals a clone. What lands
        // on disk is therefore the sealed projection of this fixture, not the
        // unsealed fixture object, so the exactness this test pins is against
        // that sealed form and the bytes that carry it.
        let sealed = artifact.sealed_json().expect("canonical sealed bytes");
        let expected: PerfEvidenceArtifact =
            serde_json::from_str(&sealed).expect("the canonical sealed bytes re-parse");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json).expect("reload");
        assert_eq!(reloaded, expected);
        assert_eq!(
            std::fs::read_to_string(&paths.json).expect("persisted evidence bytes"),
            sealed,
            "the writer persists exactly the canonical sealed bytes"
        );
    }
}
