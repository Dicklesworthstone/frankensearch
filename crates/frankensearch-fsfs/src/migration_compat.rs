//! Upgrade and migration compatibility verification for fsfs.
//!
//! Implements the normative "Upgrade and Migration Compatibility Verification
//! Strategy" of `docs/fsfs-packaging-release-install-contract.md` and emits the
//! five `upgrade.migration.*` reason codes that section requires.
//!
//! This is deliberately NOT installer surface: `install.sh` owns the
//! `install.*` and `upgrade.apply.*` codes. This module evaluates a completed
//! migration-compatibility RUN and decides whether the release candidate may
//! progress.
//!
//! Every code here is emitted from a detected condition. The conditions that
//! matter most are the ones a naive emitter gets wrong, so each is called out
//! at its detection site:
//!
//! - a matrix whose reported paths all passed but which is missing a required
//!   path entirely,
//! - an invariant set that reports success while a repeated migration is not
//!   idempotent, or while a deprecated config key was silently reinterpreted
//!   without the mandated warning,
//! - a quality delta that moved by more than the threshold in the FAVOURABLE
//!   direction (the contract bounds drift, not decline) and the exact
//!   boundary value that a `<=` comparison would admit,
//! - a cycle where rollback was never attempted, as opposed to attempted and
//!   failed,
//! - a soak that finished inside its wall-clock budget while exceeding its
//!   memory budget, or that ran on a corpus below the required multi-GB floor.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

// --- Kind constants ---

pub const KIND_MATRIX_REPORT: &str = "fsfs_migration_matrix_report";
pub const KIND_INVARIANTS_REPORT: &str = "fsfs_migration_invariants_report";
pub const KIND_QUALITY_REGRESSION_REPORT: &str = "fsfs_migration_quality_regression";
pub const KIND_SOAK_METRICS: &str = "fsfs_migration_soak_metrics";
pub const MIGRATION_SCHEMA_VERSION: u32 = 1;

// --- Reason codes (contract: Required Reason Codes) ---

pub const REASON_MATRIX_FAILED: &str = "upgrade.migration.matrix_failed";
pub const REASON_INVARIANT_VIOLATION: &str = "upgrade.migration.invariant_violation";
pub const REASON_QUALITY_REGRESSION: &str = "upgrade.migration.quality_regression";
pub const REASON_ROLLBACK_VERIFICATION_FAILED: &str =
    "upgrade.migration.rollback_verification_failed";
pub const REASON_SOAK_BUDGET_EXCEEDED: &str = "upgrade.migration.soak_budget_exceeded";

// --- Required artifact names (contract: Required migration artifacts) ---

pub const ARTIFACT_MATRIX_REPORT: &str = "migration_matrix_report.json";
pub const ARTIFACT_INVARIANTS_REPORT: &str = "migration_invariants_report.json";
pub const ARTIFACT_QUALITY_REGRESSION: &str = "migration_quality_regression.json";
pub const ARTIFACT_SOAK_METRICS: &str = "migration_soak_metrics.json";
pub const ARTIFACT_REPLAY_COMMAND: &str = "migration_replay_command.txt";

/// Maximum acceptable NDCG drift on a migration path.
///
/// The contract states the delta MUST be `< 0.01`, so the boundary value
/// itself is a failure and the bound is on the magnitude of the change.
pub const NDCG_DRIFT_THRESHOLD: f64 = 0.01;

/// Corpus floor for the soak lane: "at least one multi-GB corpus migration
/// soak run per release cycle".
pub const SOAK_MIN_CORPUS_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// A version path in the required matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionPath {
    /// `N-2 -> N`: automatic migration or deterministic hard-fail.
    TwoBackToCurrent,
    /// `N-1 -> N`: automatic migration with no data loss.
    OneBackToCurrent,
    /// `N -> N`: fresh install, no migration required.
    CurrentToCurrent,
    /// `N -> N-1`: rollback.
    CurrentToOneBack,
}

impl VersionPath {
    /// Every path the contract's version-path matrix requires.
    pub const REQUIRED: &'static [Self] = &[
        Self::TwoBackToCurrent,
        Self::OneBackToCurrent,
        Self::CurrentToCurrent,
        Self::CurrentToOneBack,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TwoBackToCurrent => "N-2->N",
            Self::OneBackToCurrent => "N-1->N",
            Self::CurrentToCurrent => "N->N",
            Self::CurrentToOneBack => "N->N-1",
        }
    }

    /// Paths that carry the NDCG result-stability gate.
    #[must_use]
    pub const fn is_quality_gated(self) -> bool {
        matches!(self, Self::TwoBackToCurrent | Self::OneBackToCurrent)
    }
}

/// Outcome of one executed version path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathOutcome {
    /// Migration completed automatically.
    Migrated,
    /// No migration was required (fresh install baseline).
    NotRequired,
    /// Migration refused, but deterministically and with recovery guidance.
    DeterministicHardFail,
    /// Migration failed in a way the contract does not admit.
    Failed,
}

/// One executed row of the version-path matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathResult {
    pub path: VersionPath,
    pub outcome: PathOutcome,
    /// Set when a hard fail carried the explicit recovery guidance the
    /// contract requires for `N-2 -> N`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recovery_guidance: Option<String>,
}

/// A post-migration invariant check over one subsystem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantCheck {
    /// `fsvi`, `frankensqlite`, `tantivy`, or `config`.
    pub subsystem: String,
    pub holds: bool,
    /// Digest of the artifact after the first migration run.
    pub post_migration_digest: String,
    /// Digest after running the same migration a second time. The contract
    /// requires repeated invocation to be idempotent, so this must equal
    /// `post_migration_digest`.
    pub repeat_migration_digest: String,
    /// Deprecated configuration keys observed in the source config.
    #[serde(default)]
    pub deprecated_keys_observed: Vec<String>,
    /// Deprecated keys for which a warning reason code was actually emitted.
    #[serde(default)]
    pub deprecated_keys_warned: Vec<String>,
}

/// NDCG on the fixed golden query set, before and after a migration path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualitySample {
    pub path: VersionPath,
    pub golden_query_set: String,
    pub ndcg_before: f64,
    pub ndcg_after: f64,
}

impl QualitySample {
    /// Signed drift; positive means the migrated build scored higher.
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.ndcg_after - self.ndcg_before
    }
}

/// What the cycle actually did about rollback validation.
///
/// "Not attempted" is a distinct state from "attempted and unsupported": the
/// contract requires every cycle to ATTEMPT validation, and only the second
/// state is admissible when it carries operator guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollbackAttempt {
    /// No rollback validation ran for this cycle.
    NotAttempted,
    /// Rollback ran and full rollback is supported.
    Completed,
    /// Rollback ran, but full rollback is unsupported on this path.
    UnsupportedByDesign,
}

/// Rollback validation for one migration cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackValidation {
    pub cycle: String,
    pub attempt: RollbackAttempt,
    pub runtime_started_in_safe_mode: bool,
    pub migrated_artifacts_intact: bool,
    /// Operator guidance, required when full rollback is unsupported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_guidance: Option<String>,
}

/// Measurements from the large-corpus soak lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoakRun {
    pub corpus_bytes: u64,
    pub duration_secs: u64,
    pub duration_budget_secs: u64,
    pub peak_memory_bytes: u64,
    pub memory_budget_bytes: u64,
    pub post_migration_checks_passed: bool,
}

/// A complete migration-compatibility run awaiting adjudication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationRun {
    pub from_version: String,
    pub to_version: String,
    pub replay_command: String,
    pub paths: Vec<PathResult>,
    pub invariants: Vec<InvariantCheck>,
    pub quality: Vec<QualitySample>,
    pub rollback: Vec<RollbackValidation>,
    /// Absent when the soak lane did not run this cycle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soak: Option<SoakRun>,
}

/// One emitted reason code with the evidence that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationFinding {
    pub reason_code: String,
    pub subject: String,
    pub detail: String,
}

impl MigrationFinding {
    fn new(reason_code: &str, subject: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.to_owned(),
            subject: subject.into(),
            detail: detail.into(),
        }
    }
}

/// Adjudicated result of a migration-compatibility run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationVerdict {
    pub findings: Vec<MigrationFinding>,
}

impl MigrationVerdict {
    /// True when no reason code was emitted and rollout may progress.
    #[must_use]
    pub fn is_clear(&self) -> bool {
        self.findings.is_empty()
    }

    /// Every distinct reason code emitted, in contract order.
    #[must_use]
    pub fn reason_codes(&self) -> Vec<&str> {
        let mut codes: Vec<&str> = self
            .findings
            .iter()
            .map(|finding| finding.reason_code.as_str())
            .collect();
        codes.sort_unstable();
        codes.dedup();
        codes
    }

    #[must_use]
    pub fn emitted(&self, reason_code: &str) -> bool {
        self.findings
            .iter()
            .any(|finding| finding.reason_code == reason_code)
    }
}

/// Adjudicate a completed migration-compatibility run.
#[must_use]
pub fn evaluate(run: &MigrationRun) -> MigrationVerdict {
    let mut findings = Vec::new();
    evaluate_matrix(run, &mut findings);
    evaluate_invariants(run, &mut findings);
    evaluate_quality(run, &mut findings);
    evaluate_rollback(run, &mut findings);
    evaluate_soak(run, &mut findings);
    MigrationVerdict { findings }
}

fn evaluate_matrix(run: &MigrationRun, findings: &mut Vec<MigrationFinding>) {
    // A matrix is not judged by its reported rows alone. Checking only
    // "did any executed path fail?" passes a run that never executed a
    // required path at all, which is the failure mode that lets an
    // unverified upgrade path reach a release candidate.
    let executed: BTreeSet<VersionPath> = run.paths.iter().map(|result| result.path).collect();
    for required in VersionPath::REQUIRED {
        if !executed.contains(required) {
            findings.push(MigrationFinding::new(
                REASON_MATRIX_FAILED,
                required.label(),
                "required version path is absent from the matrix report",
            ));
        }
    }
    if run.paths.len() > executed.len() {
        findings.push(MigrationFinding::new(
            REASON_MATRIX_FAILED,
            "matrix",
            "the matrix report contains duplicate rows for a version path",
        ));
    }

    for result in &run.paths {
        match (result.path, result.outcome) {
            (_, PathOutcome::Failed) => findings.push(MigrationFinding::new(
                REASON_MATRIX_FAILED,
                result.path.label(),
                "version path failed",
            )),
            // A deterministic hard fail is admissible only for N-2 -> N, and
            // only when it carried explicit recovery guidance.
            (VersionPath::TwoBackToCurrent, PathOutcome::DeterministicHardFail) => {
                if result
                    .recovery_guidance
                    .as_ref()
                    .is_none_or(|guidance| guidance.trim().is_empty())
                {
                    findings.push(MigrationFinding::new(
                        REASON_MATRIX_FAILED,
                        result.path.label(),
                        "hard fail is admissible only with explicit recovery guidance",
                    ));
                }
            }
            (path, PathOutcome::DeterministicHardFail) => findings.push(MigrationFinding::new(
                REASON_MATRIX_FAILED,
                path.label(),
                "this path must migrate automatically; a hard fail is not admissible",
            )),
            (VersionPath::CurrentToCurrent, PathOutcome::Migrated) => {
                findings.push(MigrationFinding::new(
                    REASON_MATRIX_FAILED,
                    result.path.label(),
                    "a fresh install must not require migration",
                ));
            }
            _ => (),
        }
    }
}

fn evaluate_invariants(run: &MigrationRun, findings: &mut Vec<MigrationFinding>) {
    for check in &run.invariants {
        if !check.holds {
            findings.push(MigrationFinding::new(
                REASON_INVARIANT_VIOLATION,
                &check.subsystem,
                "post-migration invariant does not hold",
            ));
        }
        // Idempotence is a separate fact from the invariant flag: a migration
        // can leave a valid artifact and still mutate it again on the second
        // run, which the contract forbids and a top-level ok flag hides.
        if check.post_migration_digest != check.repeat_migration_digest {
            findings.push(MigrationFinding::new(
                REASON_INVARIANT_VIOLATION,
                &check.subsystem,
                format!(
                    "repeated migration is not idempotent: {} then {}",
                    check.post_migration_digest, check.repeat_migration_digest
                ),
            ));
        }
        // Deprecated keys must keep working WITH warnings. A key that was
        // honoured silently is a silent semantic reinterpretation, which is
        // exactly what the contract forbids, and it looks like success.
        let warned: BTreeSet<&str> = check
            .deprecated_keys_warned
            .iter()
            .map(String::as_str)
            .collect();
        for key in &check.deprecated_keys_observed {
            if !warned.contains(key.as_str()) {
                findings.push(MigrationFinding::new(
                    REASON_INVARIANT_VIOLATION,
                    &check.subsystem,
                    format!("deprecated configuration key {key} was accepted without a warning"),
                ));
            }
        }
    }
}

fn evaluate_quality(run: &MigrationRun, findings: &mut Vec<MigrationFinding>) {
    let sampled: BTreeSet<VersionPath> = run.quality.iter().map(|sample| sample.path).collect();
    for path in VersionPath::REQUIRED
        .iter()
        .filter(|path| path.is_quality_gated())
    {
        if !sampled.contains(path) {
            findings.push(MigrationFinding::new(
                REASON_QUALITY_REGRESSION,
                path.label(),
                "quality-gated path has no golden query set measurement",
            ));
        }
    }

    for sample in &run.quality {
        if !sample.is_quality_gated_path() {
            continue;
        }
        let delta = sample.delta();
        // The contract bounds DRIFT, not decline. A large favourable swing on
        // a fixed golden query set means the migration changed ranking
        // materially, so a one-sided `after < before - threshold` test would
        // wave it through. The bound is also strict, so the threshold value
        // itself fails.
        if delta.abs() >= NDCG_DRIFT_THRESHOLD {
            findings.push(MigrationFinding::new(
                REASON_QUALITY_REGRESSION,
                sample.path.label(),
                format!(
                    "NDCG drift {delta:+.6} on {} is not below the {NDCG_DRIFT_THRESHOLD} threshold",
                    sample.golden_query_set
                ),
            ));
        }
    }
}

impl QualitySample {
    fn is_quality_gated_path(&self) -> bool {
        self.path.is_quality_gated()
    }
}

fn evaluate_rollback(run: &MigrationRun, findings: &mut Vec<MigrationFinding>) {
    // "Every migration test cycle MUST attempt rollback validation." A cycle
    // with no rollback record at all is the case a "did rollback fail?" check
    // reports as clean.
    if run.rollback.is_empty() {
        findings.push(MigrationFinding::new(
            REASON_ROLLBACK_VERIFICATION_FAILED,
            "cycle",
            "no rollback validation was attempted for this migration cycle",
        ));
    }
    for validation in &run.rollback {
        if validation.attempt == RollbackAttempt::NotAttempted {
            findings.push(MigrationFinding::new(
                REASON_ROLLBACK_VERIFICATION_FAILED,
                &validation.cycle,
                "rollback validation was recorded but never attempted",
            ));
            continue;
        }
        if !validation.runtime_started_in_safe_mode {
            findings.push(MigrationFinding::new(
                REASON_ROLLBACK_VERIFICATION_FAILED,
                &validation.cycle,
                "runtime did not start in safe mode after rollback",
            ));
        }
        if !validation.migrated_artifacts_intact {
            findings.push(MigrationFinding::new(
                REASON_ROLLBACK_VERIFICATION_FAILED,
                &validation.cycle,
                "rollback silently corrupted migrated artifacts",
            ));
        }
        // An unsupported rollback is acceptable only when it is deterministic
        // for the operator, so missing guidance is itself the failure.
        if validation.attempt == RollbackAttempt::UnsupportedByDesign
            && validation
                .operator_guidance
                .as_ref()
                .is_none_or(|guidance| guidance.trim().is_empty())
        {
            findings.push(MigrationFinding::new(
                REASON_ROLLBACK_VERIFICATION_FAILED,
                &validation.cycle,
                "unsupported rollback must carry deterministic operator guidance",
            ));
        }
    }
}

fn evaluate_soak(run: &MigrationRun, findings: &mut Vec<MigrationFinding>) {
    let Some(soak) = run.soak.as_ref() else {
        return;
    };
    if soak.duration_secs > soak.duration_budget_secs {
        findings.push(MigrationFinding::new(
            REASON_SOAK_BUDGET_EXCEEDED,
            "duration",
            format!(
                "soak ran {}s against a {}s budget",
                soak.duration_secs, soak.duration_budget_secs
            ),
        ));
    }
    // Memory is a budget too. A soak that finishes early while peaking over
    // its memory ceiling passes any duration-only check.
    if soak.peak_memory_bytes > soak.memory_budget_bytes {
        findings.push(MigrationFinding::new(
            REASON_SOAK_BUDGET_EXCEEDED,
            "peak_memory",
            format!(
                "soak peaked at {} bytes against a {} byte budget",
                soak.peak_memory_bytes, soak.memory_budget_bytes
            ),
        ));
    }
    // A fast, small soak is not a soak. Without this floor the lane can be
    // satisfied by a corpus that never exercises multi-GB migration at all.
    if soak.corpus_bytes < SOAK_MIN_CORPUS_BYTES {
        findings.push(MigrationFinding::new(
            REASON_SOAK_BUDGET_EXCEEDED,
            "corpus_bytes",
            format!(
                "soak corpus {} bytes is below the required {SOAK_MIN_CORPUS_BYTES} byte floor",
                soak.corpus_bytes
            ),
        ));
    }
    if !soak.post_migration_checks_passed {
        findings.push(MigrationFinding::new(
            REASON_SOAK_BUDGET_EXCEEDED,
            "post_migration_checks",
            "soak completed but its post-migration correctness checks failed",
        ));
    }
}

/// The five artifacts a migration-compatibility run must publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationArtifacts {
    pub matrix_report: String,
    pub invariants_report: String,
    pub quality_regression: String,
    /// Present only when the soak lane ran this cycle.
    pub soak_metrics: Option<String>,
    pub replay_command: String,
}

impl MigrationArtifacts {
    /// File names actually published, for manifest checks.
    #[must_use]
    pub fn published_names(&self) -> Vec<&'static str> {
        let mut names = vec![
            ARTIFACT_MATRIX_REPORT,
            ARTIFACT_INVARIANTS_REPORT,
            ARTIFACT_QUALITY_REGRESSION,
        ];
        if self.soak_metrics.is_some() {
            names.push(ARTIFACT_SOAK_METRICS);
        }
        names.push(ARTIFACT_REPLAY_COMMAND);
        names
    }
}

/// Render the contract-mandated artifacts for a run and its verdict.
///
/// # Errors
///
/// Returns a `serde_json` error when a report cannot be serialized.
pub fn render_artifacts(
    run: &MigrationRun,
    verdict: &MigrationVerdict,
) -> Result<MigrationArtifacts, serde_json::Error> {
    let findings_for = |code: &str| -> Vec<&MigrationFinding> {
        verdict
            .findings
            .iter()
            .filter(|finding| finding.reason_code == code)
            .collect()
    };

    let matrix_report = serde_json::to_string_pretty(&serde_json::json!({
        "kind": KIND_MATRIX_REPORT,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "from_version": run.from_version,
        "to_version": run.to_version,
        "paths": run.paths,
        "reason_codes": findings_for(REASON_MATRIX_FAILED),
    }))?;
    let invariants_report = serde_json::to_string_pretty(&serde_json::json!({
        "kind": KIND_INVARIANTS_REPORT,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "invariants": run.invariants,
        "reason_codes": findings_for(REASON_INVARIANT_VIOLATION),
    }))?;
    let quality_regression = serde_json::to_string_pretty(&serde_json::json!({
        "kind": KIND_QUALITY_REGRESSION_REPORT,
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "ndcg_drift_threshold": NDCG_DRIFT_THRESHOLD,
        "samples": run.quality,
        "reason_codes": findings_for(REASON_QUALITY_REGRESSION),
    }))?;
    let soak_metrics = run
        .soak
        .as_ref()
        .map(|soak| {
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": KIND_SOAK_METRICS,
                "schema_version": MIGRATION_SCHEMA_VERSION,
                "soak": soak,
                "reason_codes": findings_for(REASON_SOAK_BUDGET_EXCEEDED),
            }))
        })
        .transpose()?;

    Ok(MigrationArtifacts {
        matrix_report,
        invariants_report,
        quality_regression,
        soak_metrics,
        replay_command: format!("{}\n", run.replay_command.trim_end()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_invariant(subsystem: &str) -> InvariantCheck {
        InvariantCheck {
            subsystem: subsystem.to_owned(),
            holds: true,
            post_migration_digest: format!("{subsystem}-digest"),
            repeat_migration_digest: format!("{subsystem}-digest"),
            deprecated_keys_observed: Vec::new(),
            deprecated_keys_warned: Vec::new(),
        }
    }

    fn clean_run() -> MigrationRun {
        MigrationRun {
            from_version: "0.1.0".to_owned(),
            to_version: "0.3.0".to_owned(),
            replay_command: "fsfs migration replay --cycle rc1".to_owned(),
            paths: vec![
                PathResult {
                    path: VersionPath::TwoBackToCurrent,
                    outcome: PathOutcome::Migrated,
                    recovery_guidance: None,
                },
                PathResult {
                    path: VersionPath::OneBackToCurrent,
                    outcome: PathOutcome::Migrated,
                    recovery_guidance: None,
                },
                PathResult {
                    path: VersionPath::CurrentToCurrent,
                    outcome: PathOutcome::NotRequired,
                    recovery_guidance: None,
                },
                PathResult {
                    path: VersionPath::CurrentToOneBack,
                    outcome: PathOutcome::Migrated,
                    recovery_guidance: None,
                },
            ],
            invariants: vec![
                clean_invariant("fsvi"),
                clean_invariant("frankensqlite"),
                clean_invariant("tantivy"),
                clean_invariant("config"),
            ],
            quality: vec![
                QualitySample {
                    path: VersionPath::TwoBackToCurrent,
                    golden_query_set: "golden-v1".to_owned(),
                    ndcg_before: 0.8100,
                    ndcg_after: 0.8105,
                },
                QualitySample {
                    path: VersionPath::OneBackToCurrent,
                    golden_query_set: "golden-v1".to_owned(),
                    ndcg_before: 0.8100,
                    ndcg_after: 0.8098,
                },
            ],
            rollback: vec![RollbackValidation {
                cycle: "rc1".to_owned(),
                attempt: RollbackAttempt::Completed,
                runtime_started_in_safe_mode: true,
                migrated_artifacts_intact: true,
                operator_guidance: None,
            }],
            soak: Some(SoakRun {
                corpus_bytes: 4 * 1024 * 1024 * 1024,
                duration_secs: 900,
                duration_budget_secs: 1_800,
                peak_memory_bytes: 6 * 1024 * 1024 * 1024,
                memory_budget_bytes: 8 * 1024 * 1024 * 1024,
                post_migration_checks_passed: true,
            }),
        }
    }

    #[test]
    fn a_conformant_run_emits_no_reason_code() {
        let verdict = evaluate(&clean_run());
        assert!(
            verdict.is_clear(),
            "the control run must emit nothing, observed {:?}",
            verdict.reason_codes()
        );
    }

    /// Planted negative: every REPORTED path passes, so a "did any path fail?"
    /// emitter sees a clean matrix — but `N-2 -> N` was never executed.
    #[test]
    fn matrix_failed_catches_a_required_path_that_was_never_executed() {
        let mut run = clean_run();
        run.paths
            .retain(|result| result.path != VersionPath::TwoBackToCurrent);
        assert!(
            run.paths
                .iter()
                .all(|result| result.outcome != PathOutcome::Failed),
            "the planted negative must contain no failed row at all"
        );

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_MATRIX_FAILED),
            "an absent required path must emit {REASON_MATRIX_FAILED}, observed {:?}",
            verdict.reason_codes()
        );
        assert!(
            verdict
                .findings
                .iter()
                .any(|finding| finding.subject == "N-2->N" && finding.detail.contains("absent")),
            "the finding must name the missing path"
        );
    }

    #[test]
    fn matrix_failed_rejects_a_hard_fail_without_recovery_guidance() {
        let mut run = clean_run();
        for result in &mut run.paths {
            if result.path == VersionPath::TwoBackToCurrent {
                result.outcome = PathOutcome::DeterministicHardFail;
                result.recovery_guidance = None;
            }
        }
        assert!(evaluate(&run).emitted(REASON_MATRIX_FAILED));

        // The same hard fail WITH guidance is admissible for this path.
        for result in &mut run.paths {
            if result.path == VersionPath::TwoBackToCurrent {
                result.recovery_guidance = Some("rebuild the index with fsfs reindex".to_owned());
            }
        }
        assert!(
            evaluate(&run).is_clear(),
            "a deterministic hard fail with guidance is admissible on N-2->N"
        );
    }

    /// Planted negative: the invariant flag says the artifact is fine, and it
    /// is — but running the same migration again changes it, which the
    /// contract forbids and a `holds` check cannot see.
    #[test]
    fn invariant_violation_catches_a_non_idempotent_repeat_migration() {
        let mut run = clean_run();
        for check in &mut run.invariants {
            if check.subsystem == "frankensqlite" {
                assert!(check.holds, "the planted negative must still report holds");
                check.repeat_migration_digest = "frankensqlite-digest-second-run".to_owned();
            }
        }

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_INVARIANT_VIOLATION),
            "a non-idempotent repeat must emit {REASON_INVARIANT_VIOLATION}, observed {:?}",
            verdict.reason_codes()
        );
        assert!(
            verdict
                .findings
                .iter()
                .any(|finding| finding.detail.contains("not idempotent"))
        );
    }

    /// Planted negative: a deprecated key kept working — which the contract
    /// requires — but no warning was emitted, so the semantics were
    /// reinterpreted silently while every boolean still reads healthy.
    #[test]
    fn invariant_violation_catches_a_silently_honoured_deprecated_key() {
        let mut run = clean_run();
        for check in &mut run.invariants {
            if check.subsystem == "config" {
                check.deprecated_keys_observed = vec!["index.max_threads".to_owned()];
                check.deprecated_keys_warned = Vec::new();
            }
        }
        assert!(
            run.invariants.iter().all(|check| check.holds),
            "the planted negative must keep every invariant flag healthy"
        );

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_INVARIANT_VIOLATION),
            "a silently honoured deprecated key must emit {REASON_INVARIANT_VIOLATION}"
        );

        // Warning on the same key clears it: the key working is not the defect.
        for check in &mut run.invariants {
            if check.subsystem == "config" {
                check.deprecated_keys_warned = vec!["index.max_threads".to_owned()];
            }
        }
        assert!(evaluate(&run).is_clear());
    }

    /// Planted negative: quality got BETTER by more than the threshold. A
    /// one-sided `after < before - threshold` emitter passes it, but the
    /// contract bounds drift on a fixed golden query set in both directions.
    #[test]
    fn quality_regression_catches_a_large_favourable_ndcg_swing() {
        let mut run = clean_run();
        for sample in &mut run.quality {
            if sample.path == VersionPath::OneBackToCurrent {
                sample.ndcg_before = 0.8100;
                sample.ndcg_after = 0.8400;
            }
        }
        let improved = run
            .quality
            .iter()
            .find(|sample| sample.path == VersionPath::OneBackToCurrent)
            .expect("planted sample");
        assert!(
            improved.delta() > 0.0,
            "the planted negative must be an improvement, not a decline"
        );

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_QUALITY_REGRESSION),
            "a large favourable swing must emit {REASON_QUALITY_REGRESSION}, observed {:?}",
            verdict.reason_codes()
        );
    }

    /// Planted negative: the drift is EXACTLY the threshold. The contract says
    /// the delta must be `< 0.01`, so a `<=` comparison admits a failing run.
    #[test]
    fn quality_regression_fails_at_exactly_the_threshold() {
        let mut run = clean_run();
        for sample in &mut run.quality {
            if sample.path == VersionPath::TwoBackToCurrent {
                sample.ndcg_before = 0.5;
                sample.ndcg_after = 0.5 - NDCG_DRIFT_THRESHOLD;
            }
        }
        assert!(
            evaluate(&run).emitted(REASON_QUALITY_REGRESSION),
            "drift equal to the threshold must fail the strict bound"
        );
    }

    #[test]
    fn quality_regression_catches_an_unmeasured_gated_path() {
        let mut run = clean_run();
        run.quality
            .retain(|sample| sample.path != VersionPath::TwoBackToCurrent);
        assert!(evaluate(&run).emitted(REASON_QUALITY_REGRESSION));
    }

    /// Planted negative: rollback was never attempted. Nothing failed, because
    /// nothing ran — the exact shape a "did rollback fail?" emitter clears.
    #[test]
    fn rollback_verification_failed_catches_a_cycle_that_never_attempted_rollback() {
        let mut run = clean_run();
        for validation in &mut run.rollback {
            validation.attempt = RollbackAttempt::NotAttempted;
            // Everything else still reads healthy.
            validation.runtime_started_in_safe_mode = true;
            validation.migrated_artifacts_intact = true;
        }

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_ROLLBACK_VERIFICATION_FAILED),
            "an unattempted rollback must emit {REASON_ROLLBACK_VERIFICATION_FAILED}, observed {:?}",
            verdict.reason_codes()
        );

        // A cycle with no rollback record at all is the same defect.
        run.rollback.clear();
        assert!(evaluate(&run).emitted(REASON_ROLLBACK_VERIFICATION_FAILED));
    }

    #[test]
    fn rollback_verification_failed_requires_guidance_when_rollback_is_unsupported() {
        let mut run = clean_run();
        for validation in &mut run.rollback {
            validation.attempt = RollbackAttempt::UnsupportedByDesign;
            validation.operator_guidance = None;
        }
        assert!(evaluate(&run).emitted(REASON_ROLLBACK_VERIFICATION_FAILED));

        for validation in &mut run.rollback {
            validation.operator_guidance =
                Some("restore the pre-upgrade snapshot from /var/backups".to_owned());
        }
        assert!(evaluate(&run).is_clear());
    }

    /// Planted negative: the soak finished well inside its wall-clock budget,
    /// so a duration-only emitter clears it — while peaking over its memory
    /// ceiling.
    #[test]
    fn soak_budget_exceeded_catches_memory_overrun_inside_the_time_budget() {
        let mut run = clean_run();
        if let Some(soak) = run.soak.as_mut() {
            soak.duration_secs = 60;
            soak.duration_budget_secs = 1_800;
            soak.peak_memory_bytes = 9 * 1024 * 1024 * 1024;
            soak.memory_budget_bytes = 8 * 1024 * 1024 * 1024;
        }
        let soak = run.soak.as_ref().expect("planted soak");
        assert!(
            soak.duration_secs < soak.duration_budget_secs,
            "the planted negative must sit inside the duration budget"
        );

        let verdict = evaluate(&run);
        assert!(
            verdict.emitted(REASON_SOAK_BUDGET_EXCEEDED),
            "a memory overrun must emit {REASON_SOAK_BUDGET_EXCEEDED}, observed {:?}",
            verdict.reason_codes()
        );
        assert!(
            verdict
                .findings
                .iter()
                .any(|finding| finding.subject == "peak_memory")
        );
    }

    /// Planted negative: a soak that is fast and cheap because the corpus is
    /// far below the multi-GB floor the strategy requires.
    #[test]
    fn soak_budget_exceeded_catches_a_corpus_below_the_multi_gb_floor() {
        let mut run = clean_run();
        if let Some(soak) = run.soak.as_mut() {
            soak.corpus_bytes = 64 * 1024 * 1024;
            soak.duration_secs = 12;
            soak.peak_memory_bytes = 1024 * 1024 * 1024;
        }
        let verdict = evaluate(&run);
        assert!(verdict.emitted(REASON_SOAK_BUDGET_EXCEEDED));
        assert!(
            verdict
                .findings
                .iter()
                .any(|finding| finding.subject == "corpus_bytes")
        );
    }

    #[test]
    fn a_skipped_soak_lane_emits_nothing() {
        let mut run = clean_run();
        run.soak = None;
        assert!(
            evaluate(&run).is_clear(),
            "the soak lane is conditional; skipping it is not a budget failure"
        );
    }

    #[test]
    fn artifacts_cover_every_contract_named_file_and_carry_their_reason_codes() {
        let mut run = clean_run();
        run.rollback.clear();
        if let Some(soak) = run.soak.as_mut() {
            soak.post_migration_checks_passed = false;
        }
        let verdict = evaluate(&run);
        let artifacts = render_artifacts(&run, &verdict).expect("render artifacts");

        assert_eq!(
            artifacts.published_names(),
            vec![
                ARTIFACT_MATRIX_REPORT,
                ARTIFACT_INVARIANTS_REPORT,
                ARTIFACT_QUALITY_REGRESSION,
                ARTIFACT_SOAK_METRICS,
                ARTIFACT_REPLAY_COMMAND,
            ]
        );
        assert!(
            artifacts
                .soak_metrics
                .as_ref()
                .expect("soak metrics present")
                .contains(REASON_SOAK_BUDGET_EXCEEDED),
            "the soak artifact must carry the code its own lane emitted"
        );
        assert!(
            artifacts
                .replay_command
                .starts_with("fsfs migration replay")
        );
        assert!(artifacts.replay_command.ends_with('\n'));

        // A skipped soak lane publishes four artifacts, not a stub fifth.
        run.soak = None;
        let without_soak = render_artifacts(&run, &evaluate(&run)).expect("render without soak");
        assert!(without_soak.soak_metrics.is_none());
        assert!(
            !without_soak
                .published_names()
                .contains(&ARTIFACT_SOAK_METRICS)
        );
    }

    /// The contract's Required Reason Codes list is the source of truth; this
    /// pins the exact strings so a rename in either direction is caught.
    #[test]
    fn every_contract_named_migration_code_is_declared_verbatim() {
        assert_eq!(REASON_MATRIX_FAILED, "upgrade.migration.matrix_failed");
        assert_eq!(
            REASON_INVARIANT_VIOLATION,
            "upgrade.migration.invariant_violation"
        );
        assert_eq!(
            REASON_QUALITY_REGRESSION,
            "upgrade.migration.quality_regression"
        );
        assert_eq!(
            REASON_ROLLBACK_VERIFICATION_FAILED,
            "upgrade.migration.rollback_verification_failed"
        );
        assert_eq!(
            REASON_SOAK_BUDGET_EXCEEDED,
            "upgrade.migration.soak_budget_exceeded"
        );
    }
}
