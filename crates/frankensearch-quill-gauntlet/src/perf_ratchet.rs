//! Pass-over-pass evaluation for committed Quill performance artifacts.
//!
//! The benchmark harness emits measurements; this module decides whether a
//! result may advance the committed `.bench-history` baseline. It deliberately
//! keeps noisy results in quarantine and requires a same-revision rerun before
//! a performance result can be promoted.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::perf::PERF_NULL_MARGIN_MULTIPLIER;
use crate::{
    DistributionSummary, EvidenceCellBody, EvidenceRole, PERF_ARTIFACT_SCHEMA_VERSION,
    PERF_MIN_RUNS, PerfCellResult, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    PerfMatrixSpec,
};

/// Version of the machine-readable ratchet decision artifact.
pub const PERF_RATCHET_SCHEMA_VERSION: &str = "quill-perf-ratchet-v2";
/// Maximum directional pass-over-pass regression admitted for a cell.
pub const PERF_MAX_REGRESSION_PCT: f64 = 5.0;
/// Maximum disagreement admitted between same-revision candidate reruns.
pub const PERF_MAX_REPRODUCTION_DELTA_PCT: f64 = 5.0;
/// Robust-z value retained as diagnostic provenance beside CI-gated decisions.
pub const PERF_REGRESSION_ROBUST_Z: f64 = 3.0;

const MAD_SCALE: f64 = 1.4826;
const MAD_EPSILON: f64 = 1.0e-12;

/// Evaluation purpose. Promotion is stricter than the PR regression alarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfRatchetMode {
    /// Require a complete gate, attested laws, and a same-revision rerun.
    Promotion,
    /// Compare a fast matrix slice with the committed baseline.
    RegressionAlarm,
}

/// Final operator decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfGateDecision {
    /// The measured result is eligible for the requested operation.
    Allow,
    /// A reproducible regression or activated-gate target failure blocks it.
    Block,
    /// Evidence is noisy, incomplete, incompatible, or still provisional.
    Quarantine,
}

impl fmt::Display for PerfGateDecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Allow => "Allow",
            Self::Block => "Block",
            Self::Quarantine => "Quarantine",
        })
    }
}

/// One content-addressed input or output named by the evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfEvidenceFile {
    /// Stable role such as `baseline`, `candidate`, or `manifest`.
    pub role: String,
    /// Repository-relative or operator-supplied path.
    pub path: String,
    /// Lowercase SHA-256 of the exact file bytes.
    pub sha256: String,
}

/// One structured reason contributing to the final decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfRatchetReason {
    /// Stable reason code suitable for CI and dashboards.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
}

/// One median+MAD pass-over-pass comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfCellComparison {
    /// Fixture label from the QG matrix.
    pub fixture: String,
    /// Metric label from the QG matrix.
    pub metric: String,
    /// Engine or paired-comparison arm.
    pub engine: String,
    /// Prior committed median.
    pub baseline_value: f64,
    /// Candidate median.
    pub candidate_value: f64,
    /// Positive values are regressions in the metric's declared direction.
    pub regression_pct: f64,
    /// Robust z-score using the larger candidate/baseline MAD.
    pub robust_z: f64,
    /// Whether the directional 5% pass-over-pass threshold was exceeded.
    pub threshold_exceeded: bool,
}

/// Complete machine-readable ratchet decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfRatchetEvaluation {
    /// Schema identifier.
    pub schema_version: String,
    /// Gate being evaluated.
    pub gate: PerfGate,
    /// Requested evaluation mode.
    pub mode: PerfRatchetMode,
    /// Whether the normative gate manifest marks this gate active.
    pub gate_activated: bool,
    /// Final decision.
    pub decision: PerfGateDecision,
    /// Stable structured reasons.
    pub reasons: Vec<PerfRatchetReason>,
    /// Median+MAD comparisons against the committed baseline.
    pub comparisons: Vec<PerfCellComparison>,
    /// Content-addressed evidence inputs.
    pub evidence: Vec<PerfEvidenceFile>,
    /// History files written after an Allow decision.
    pub history_updates: Vec<PerfEvidenceFile>,
}

/// Inputs to one ratchet evaluation.
pub struct PerfRatchetRequest<'a> {
    /// Prior committed history artifact, if one exists.
    pub baseline: Option<&'a PerfGateArtifact>,
    /// First candidate measurement.
    pub candidate: &'a PerfGateArtifact,
    /// Same-revision candidate rerun. Required in promotion mode.
    pub rerun: Option<&'a PerfGateArtifact>,
    /// Hash-sealed, raw-recomputable evidence for the candidate artifact.
    pub candidate_evidence: Option<&'a PerfEvidenceArtifact>,
    /// Hash-sealed, raw-recomputable evidence for the candidate rerun.
    pub rerun_evidence: Option<&'a PerfEvidenceArtifact>,
    /// Whether promotion must carry current-schema evidence in addition to the
    /// legacy threshold projection.
    pub require_current_evidence: bool,
    /// Whether the normative gate manifest marks the gate active.
    pub gate_activated: bool,
    /// Evaluation purpose.
    pub mode: PerfRatchetMode,
    /// SHA-256 of the normative TOML manifest.
    pub expected_manifest_sha256: &'a str,
    /// Content-addressed evidence paths.
    pub evidence: Vec<PerfEvidenceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    fixture: String,
    metric: String,
    engine: String,
    unit: String,
}

impl From<&PerfCellResult> for CellKey {
    fn from(cell: &PerfCellResult) -> Self {
        Self {
            fixture: cell.fixture.clone(),
            metric: cell.metric.clone(),
            engine: cell.engine.clone(),
            unit: cell.unit.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EvidenceCellKey {
    fixture: String,
    metric: String,
    unit: String,
}

impl EvidenceCellKey {
    fn from_parts(fixture: &str, metric: &str, unit: &str) -> Self {
        Self {
            fixture: fixture.to_owned(),
            metric: metric.to_owned(),
            unit: unit.to_owned(),
        }
    }
}

#[derive(Default)]
struct DecisionState {
    fatal: bool,
    blocked: bool,
    quarantined: bool,
    reasons: Vec<PerfRatchetReason>,
}

impl DecisionState {
    fn fatal(&mut self, code: &str, message: impl Into<String>) {
        self.fatal = true;
        self.reasons.push(PerfRatchetReason {
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn block(&mut self, code: &str, message: impl Into<String>) {
        self.blocked = true;
        self.reasons.push(PerfRatchetReason {
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn quarantine(&mut self, code: &str, message: impl Into<String>) {
        self.quarantined = true;
        self.reasons.push(PerfRatchetReason {
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn note(&mut self, code: &str, message: impl Into<String>) {
        self.reasons.push(PerfRatchetReason {
            code: code.to_owned(),
            message: message.into(),
        });
    }

    const fn decision(&self) -> PerfGateDecision {
        if self.fatal || self.blocked {
            PerfGateDecision::Block
        } else if self.quarantined {
            PerfGateDecision::Quarantine
        } else {
            PerfGateDecision::Allow
        }
    }
}

fn finish_evaluation(
    gate: PerfGate,
    mode: PerfRatchetMode,
    gate_activated: bool,
    state: DecisionState,
    comparisons: Vec<PerfCellComparison>,
    evidence: Vec<PerfEvidenceFile>,
) -> PerfRatchetEvaluation {
    PerfRatchetEvaluation {
        schema_version: PERF_RATCHET_SCHEMA_VERSION.to_owned(),
        gate,
        mode,
        gate_activated,
        decision: state.decision(),
        reasons: state.reasons,
        comparisons,
        evidence,
        history_updates: Vec::new(),
    }
}

/// Evaluate a candidate against the committed pass-over-pass baseline.
#[must_use]
pub fn evaluate_perf_ratchet(request: PerfRatchetRequest<'_>) -> PerfRatchetEvaluation {
    let gate = request.candidate.gate;
    let mut state = DecisionState::default();
    let candidate_cells = validate_artifact(
        request.candidate,
        gate,
        request.expected_manifest_sha256,
        "candidate",
        &mut state,
    );
    validate_paired_evidence(gate, &candidate_cells, "candidate", &mut state);
    let candidate_evidence = match request.candidate_evidence {
        Some(evidence) => {
            if !validate_current_evidence(
                evidence,
                request.candidate,
                &candidate_cells,
                request.expected_manifest_sha256,
                "candidate",
                &mut state,
            ) {
                return finish_evaluation(
                    gate,
                    request.mode,
                    request.gate_activated,
                    state,
                    Vec::new(),
                    request.evidence,
                );
            }
            Some(evidence)
        }
        None if request.require_current_evidence => {
            state.quarantine(
                "perf.ratchet.missing_current_candidate_evidence",
                "promotion requires a hash-sealed current-schema candidate evidence artifact",
            );
            None
        }
        None => None,
    };

    if request.mode == PerfRatchetMode::Promotion {
        validate_complete_gate(gate, &candidate_cells, request.gate_activated, &mut state);
        if !request.candidate.laws_attested {
            state.quarantine(
                "perf.ratchet.laws_not_attested",
                "promotion requires a full release-perf run with every standing law attested",
            );
        }
    }

    let mut comparisons = Vec::new();
    if let Some(baseline) = request.baseline {
        let baseline_cells = validate_artifact(
            baseline,
            gate,
            request.expected_manifest_sha256,
            "baseline",
            &mut state,
        );
        compare_baseline(
            baseline,
            request.candidate,
            &baseline_cells,
            &candidate_cells,
            request.mode,
            &mut comparisons,
            &mut state,
        );
    } else {
        state.quarantine(
            "perf.ratchet.missing_baseline",
            "no committed baseline exists for this gate and machine class",
        );
    }

    match (request.mode, request.rerun) {
        (PerfRatchetMode::Promotion, Some(rerun)) => {
            let rerun_cells = validate_artifact(
                rerun,
                gate,
                request.expected_manifest_sha256,
                "rerun",
                &mut state,
            );
            validate_paired_evidence(gate, &rerun_cells, "rerun", &mut state);
            if !rerun.laws_attested {
                state.quarantine(
                    "perf.ratchet.rerun_laws_not_attested",
                    "promotion requires the same-revision rerun to attest every standing law",
                );
            }
            let rerun_evidence = match request.rerun_evidence {
                Some(evidence) => {
                    if !validate_current_evidence(
                        evidence,
                        rerun,
                        &rerun_cells,
                        request.expected_manifest_sha256,
                        "rerun",
                        &mut state,
                    ) {
                        return finish_evaluation(
                            gate,
                            request.mode,
                            request.gate_activated,
                            state,
                            comparisons,
                            request.evidence,
                        );
                    }
                    Some(evidence)
                }
                None if request.require_current_evidence => {
                    state.quarantine(
                        "perf.ratchet.missing_current_rerun_evidence",
                        "promotion requires a hash-sealed current-schema rerun evidence artifact",
                    );
                    None
                }
                None => None,
            };
            if candidate_is_complete(gate, &rerun_cells) {
                // Promotion requires both independent passes to satisfy every
                // gate target. Reproduction tolerance cannot substitute for
                // independently clearing a threshold; QG-6 additionally
                // consumes the rerun's hierarchical CI/null-margin evidence.
                evaluate_gate_targets(
                    rerun,
                    &rerun_cells,
                    rerun_evidence,
                    request.gate_activated,
                    &mut state,
                );
            }
            if let (Some(candidate_evidence), Some(rerun_evidence)) =
                (candidate_evidence, rerun_evidence)
            {
                compare_current_evidence_reproduction(
                    candidate_evidence,
                    rerun_evidence,
                    &mut state,
                );
            }
            compare_reproduction(
                request.candidate,
                rerun,
                &candidate_cells,
                &rerun_cells,
                &mut state,
            );
        }
        (PerfRatchetMode::Promotion, None) => state.quarantine(
            "perf.ratchet.missing_rerun",
            "promotion requires a second measurement from the same revision and machine",
        ),
        (PerfRatchetMode::RegressionAlarm, _) => {}
    }

    if request.rerun.is_none() && request.rerun_evidence.is_some() {
        state.quarantine(
            "perf.ratchet.orphan_current_rerun_evidence",
            "current-schema rerun evidence has no matching threshold artifact",
        );
    }

    if request.mode == PerfRatchetMode::Promotion && candidate_is_complete(gate, &candidate_cells) {
        evaluate_gate_targets(
            request.candidate,
            &candidate_cells,
            candidate_evidence,
            request.gate_activated,
            &mut state,
        );
        if !request.gate_activated {
            state.quarantine(
                "perf.ratchet.gate_inactive",
                format!(
                    "{} remains provisional because the normative manifest has activated=false",
                    gate.label()
                ),
            );
        }
    }

    finish_evaluation(
        gate,
        request.mode,
        request.gate_activated,
        state,
        comparisons,
        request.evidence,
    )
}

fn validate_artifact<'a>(
    artifact: &'a PerfGateArtifact,
    gate: PerfGate,
    expected_manifest_sha256: &str,
    role: &str,
    state: &mut DecisionState,
) -> BTreeMap<CellKey, &'a PerfCellResult> {
    if artifact.schema_version != PERF_ARTIFACT_SCHEMA_VERSION {
        state.fatal(
            "perf.ratchet.invalid_schema",
            format!(
                "{role} uses schema {:?}, expected {PERF_ARTIFACT_SCHEMA_VERSION:?}",
                artifact.schema_version
            ),
        );
    }
    if artifact.gate != gate {
        state.fatal(
            "perf.ratchet.gate_mismatch",
            format!("{role} is for {}, expected {}", artifact.gate, gate),
        );
    }
    if artifact.manifest_sha256 != expected_manifest_sha256 {
        state.fatal(
            "perf.ratchet.manifest_hash_mismatch",
            format!(
                "{role} records manifest hash {}, expected {expected_manifest_sha256}",
                artifact.manifest_sha256
            ),
        );
    }
    if artifact.run_window.trim().is_empty() || artifact.run_id.trim().is_empty() {
        state.fatal(
            "perf.ratchet.missing_run_identity",
            format!("{role} must record non-empty run_window and run_id values"),
        );
    }
    let explicit_bootstrap = artifact.cells.is_empty()
        && artifact.machine_fingerprint == "unmeasured"
        && artifact.bench_elf_sha256 == "unmeasured";
    if !explicit_bootstrap && !is_lower_hex_sha256(&artifact.bench_elf_sha256) {
        state.fatal(
            "perf.ratchet.invalid_bench_elf_sha256",
            format!(
                "{role} must carry the 64-character lowercase SHA-256 self-reported by its benchmark ELF"
            ),
        );
    }

    let mut cells = BTreeMap::new();
    for cell in &artifact.cells {
        let key = CellKey::from(cell);
        if cells.insert(key.clone(), cell).is_some() {
            state.fatal(
                "perf.ratchet.duplicate_cell",
                format!(
                    "{role} repeats {}/{}/{}",
                    key.fixture, key.metric, key.engine
                ),
            );
        }
        if cell.distribution.runs < PERF_MIN_RUNS {
            state.quarantine(
                "perf.ratchet.insufficient_samples",
                format!(
                    "{role} {}/{}/{} has runs={}; require runs>={}",
                    cell.fixture, cell.metric, cell.engine, cell.distribution.runs, PERF_MIN_RUNS,
                ),
            );
        }
        let distribution = &cell.distribution;
        if !distribution.median_ci95_low.is_finite()
            || !distribution.median_ci95_high.is_finite()
            || distribution.median_ci95_low > distribution.p50
            || distribution.p50 > distribution.median_ci95_high
        {
            state.fatal(
                "perf.ratchet.invalid_median_ci",
                format!(
                    "{role} {}/{}/{} has invalid median CI [{:.6}, {:.6}] around {:.6}",
                    cell.fixture,
                    cell.metric,
                    cell.engine,
                    distribution.median_ci95_low,
                    distribution.median_ci95_high,
                    distribution.p50,
                ),
            );
        }
    }
    cells
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn expected_evidence_cells(gate: PerfGate) -> BTreeMap<EvidenceCellKey, EvidenceRole> {
    PerfMatrixSpec::complete()
        .for_gate(gate)
        .into_iter()
        .map(|spec| {
            let role = if gate == PerfGate::Qg10 || spec.metric == "tokenize_docs_per_second" {
                EvidenceRole::Diagnostic
            } else {
                EvidenceRole::Required
            };
            (
                EvidenceCellKey::from_parts(&spec.fixture, &spec.metric, metric_unit(&spec.metric)),
                role,
            )
        })
        .collect()
}

fn validate_current_evidence(
    evidence: &PerfEvidenceArtifact,
    legacy: &PerfGateArtifact,
    legacy_cells: &BTreeMap<CellKey, &PerfCellResult>,
    expected_manifest_sha256: &str,
    role: &str,
    state: &mut DecisionState,
) -> bool {
    if let Err(error) = evidence.verify_integrity() {
        let (code, message) = if is_lower_hex_sha256(&evidence.artifact_sha256) {
            (
                "perf.ratchet.current_evidence_integrity_failed",
                format!(
                    "{role} current-schema evidence failed in-memory integrity verification: \
                     {error}"
                ),
            )
        } else {
            (
                "perf.ratchet.current_evidence_unsealed",
                format!(
                    "{role} current-schema evidence lacks a valid lowercase SHA-256 content seal: \
                     {error}"
                ),
            )
        };
        state.fatal(code, message);
        return false;
    }
    if evidence.gate != legacy.gate {
        state.fatal(
            "perf.ratchet.current_evidence_gate_mismatch",
            format!(
                "{role} current-schema evidence is for {}, threshold artifact is for {}",
                evidence.gate, legacy.gate
            ),
        );
    }
    if evidence.provenance.manifest_sha256 != expected_manifest_sha256 {
        state.fatal(
            "perf.ratchet.current_evidence_manifest_mismatch",
            format!(
                "{role} current-schema evidence records manifest {}, expected \
                 {expected_manifest_sha256}",
                evidence.provenance.manifest_sha256
            ),
        );
    }
    let identity_mismatches = [
        (
            evidence.provenance.run_id != legacy.run_id,
            "run ID",
            evidence.provenance.run_id.as_str(),
            legacy.run_id.as_str(),
        ),
        (
            evidence.provenance.run_window != legacy.run_window,
            "run window",
            evidence.provenance.run_window.as_str(),
            legacy.run_window.as_str(),
        ),
        (
            evidence.provenance.build.executable_sha256 != legacy.bench_elf_sha256,
            "executing ELF SHA-256",
            evidence.provenance.build.executable_sha256.as_str(),
            legacy.bench_elf_sha256.as_str(),
        ),
        (
            evidence.provenance.build.git_revision != legacy.git_rev,
            "git revision",
            evidence.provenance.build.git_revision.as_str(),
            legacy.git_rev.as_str(),
        ),
        (
            evidence.provenance.machine.fingerprint != legacy.machine_fingerprint,
            "machine fingerprint",
            evidence.provenance.machine.fingerprint.as_str(),
            legacy.machine_fingerprint.as_str(),
        ),
        (
            evidence.provenance.corpus.corpus_sha256 != legacy.corpus_manifest_hash,
            "corpus manifest SHA-256",
            evidence.provenance.corpus.corpus_sha256.as_str(),
            legacy.corpus_manifest_hash.as_str(),
        ),
    ];
    for (mismatched, field, current, projected) in identity_mismatches {
        if mismatched {
            state.quarantine(
                "perf.ratchet.current_evidence_identity_mismatch",
                format!(
                    "{role} {field} differs between current evidence ({current:?}) and its \
                     threshold projection ({projected:?})"
                ),
            );
        }
    }
    if evidence.gate_decision.is_some() {
        state.quarantine(
            "perf.ratchet.current_evidence_predecided",
            format!(
                "{role} evidence already carries a gate decision; ratchet inputs must be provisional"
            ),
        );
    }

    let expected = expected_evidence_cells(legacy.gate);
    let mut actual = BTreeMap::new();
    for cell in &evidence.cells {
        let key =
            EvidenceCellKey::from_parts(&cell.spec.fixture, &cell.spec.metric, &cell.spec.unit);
        if actual.insert(key.clone(), cell.spec.role).is_some() {
            state.fatal(
                "perf.ratchet.current_evidence_duplicate_cell",
                format!(
                    "{role} current evidence repeats {}/{}/{}",
                    key.fixture, key.metric, key.unit
                ),
            );
        }
        if cell.spec.gate != legacy.gate {
            state.fatal(
                "perf.ratchet.current_evidence_cell_gate_mismatch",
                format!(
                    "{role} cell {} belongs to {}, expected {}",
                    cell.cell_id, cell.spec.gate, legacy.gate
                ),
            );
        }
        let normative = expected.get(&key).copied();
        if let Some(expected_role) = normative
            && cell.spec.role != expected_role
        {
            state.fatal(
                "perf.ratchet.current_evidence_role_mismatch",
                format!(
                    "{role} {}/{}/{} has role {:?}, expected {:?}",
                    key.fixture, key.metric, key.unit, cell.spec.role, expected_role
                ),
            );
        }
        validate_current_evidence_cell(
            cell,
            evidence,
            legacy_cells,
            normative.is_some(),
            role,
            state,
        );
    }
    let missing = expected
        .iter()
        .filter(|(key, expected_role)| actual.get(*key) != Some(*expected_role))
        .count();
    let extra_required = actual
        .iter()
        .filter(|(key, actual_role)| {
            **actual_role == EvidenceRole::Required && expected.get(*key) != Some(*actual_role)
        })
        .count();
    if missing != 0 || extra_required != 0 {
        state.fatal(
            "perf.ratchet.current_evidence_matrix_mismatch",
            format!(
                "{role} current evidence is not the normative {} matrix: {missing} missing or \
                 wrong-role cells, {extra_required} unexpected required cells",
                legacy.gate
            ),
        );
    }

    if legacy.gate == PerfGate::Qg10 {
        state.quarantine(
            "perf.ratchet.qg10_structural_evidence_not_decision_capable",
            "QG-10 dependency facts remain diagnostic until typed structural facts become \
             decision-capable",
        );
    } else if !evidence.ratchet_admissible() {
        state.quarantine(
            "perf.ratchet.current_evidence_not_admissible",
            format!(
                "{role} current evidence has gate status {} and cannot update history",
                evidence.gate_status
            ),
        );
    }
    true
}

fn validate_current_evidence_cell(
    cell: &crate::EvidenceCell,
    evidence: &PerfEvidenceArtifact,
    legacy_cells: &BTreeMap<CellKey, &PerfCellResult>,
    normative: bool,
    role: &str,
    state: &mut DecisionState,
) {
    match &cell.body {
        EvidenceCellBody::Paired {
            paired,
            hierarchical,
            hierarchical_null,
            ..
        } => {
            let provenance = &paired.provenance;
            let scope_matches = paired.scope.unit == cell.spec.unit;
            let provenance_matches = provenance.run_id == evidence.provenance.run_id
                && provenance.executable_sha256 == evidence.provenance.build.executable_sha256
                && provenance.corpus_sha256 == evidence.provenance.corpus.corpus_sha256
                && provenance.input_identity == cell.spec.input_identity
                && provenance.worker_id == evidence.provenance.machine.fingerprint
                && provenance.build_profile == evidence.provenance.build.build_profile;
            if !scope_matches || !provenance_matches {
                state.fatal(
                    "perf.ratchet.current_evidence_scope_mismatch",
                    format!(
                        "{role} cell {} does not share the artifact's operation unit and sealed \
                         execution provenance",
                        cell.cell_id
                    ),
                );
            }
            if cell.spec.gate == PerfGate::Qg6
                && (hierarchical.is_none() || hierarchical_null.is_none())
            {
                state.fatal(
                    "perf.ratchet.qg6_hierarchical_evidence_missing",
                    format!(
                        "{role} QG-6 cell {} lacks a verified hierarchical effect or null estimate",
                        cell.cell_id
                    ),
                );
            }
            if normative {
                reconcile_current_cell_with_projection(cell, paired, legacy_cells, role, state);
            }
        }
        EvidenceCellBody::Facts {
            raw_values: _,
            summary,
        } => {
            if cell.spec.gate != PerfGate::Qg10 {
                state.fatal(
                    "perf.ratchet.current_evidence_unexpected_facts",
                    format!(
                        "{role} non-QG-10 cell {} uses structural facts",
                        cell.cell_id
                    ),
                );
            }
            if normative {
                let key = CellKey {
                    fixture: cell.spec.fixture.clone(),
                    metric: cell.spec.metric.clone(),
                    engine: "default_feature_graph".to_owned(),
                    unit: cell.spec.unit.clone(),
                };
                match legacy_cells.get(&key) {
                    Some(projected) if projected.distribution == *summary => {}
                    Some(_) => state.fatal(
                        "perf.ratchet.current_evidence_projection_mismatch",
                        format!(
                            "{role} QG-10 facts summary does not match its threshold projection"
                        ),
                    ),
                    None => state.fatal(
                        "perf.ratchet.current_evidence_projection_missing",
                        format!("{role} QG-10 facts cell has no matching threshold projection"),
                    ),
                }
            }
        }
    }
}

fn reconcile_current_cell_with_projection(
    cell: &crate::EvidenceCell,
    paired: &crate::PairedExperimentResult,
    legacy_cells: &BTreeMap<CellKey, &PerfCellResult>,
    role: &str,
    state: &mut DecisionState,
) {
    let (treatment_engine, control_engine) = if cell.spec.metric == "tokenize_docs_per_second" {
        ("quill_tokenizer", "quill_tokenizer_null")
    } else {
        ("quill", "tantivy")
    };
    let absolute = |engine: &str| CellKey {
        fixture: cell.spec.fixture.clone(),
        metric: cell.spec.metric.clone(),
        engine: engine.to_owned(),
        unit: cell.spec.unit.clone(),
    };
    let ratio = |suffix: &str, engine: &str| CellKey {
        fixture: cell.spec.fixture.clone(),
        metric: format!("{}_{}", cell.spec.metric, suffix),
        engine: engine.to_owned(),
        unit: "ratio".to_owned(),
    };
    let treatment = legacy_cells.get(&absolute(treatment_engine));
    let control = legacy_cells.get(&absolute(control_engine));
    let effect = legacy_cells.get(&ratio("quill_over_tantivy", "paired_ab"));
    let null = legacy_cells.get(&ratio("tantivy_over_tantivy", "paired_null"));
    let projected_effect = projected_ratio_distribution(&paired.effect_samples);
    let projected_null = projected_ratio_distribution(&paired.null_samples);
    let aligned = treatment
        .is_some_and(|projected| projected.distribution == paired.effect.treatment)
        && control.is_some_and(|projected| projected.distribution == paired.effect.control)
        && effect.is_some_and(|projected| {
            projected_effect
                .as_ref()
                .is_some_and(|summary| projected.distribution == *summary)
        })
        && null.is_some_and(|projected| {
            projected_null
                .as_ref()
                .is_some_and(|summary| projected.distribution == *summary)
        });
    if !aligned {
        state.fatal(
            "perf.ratchet.current_evidence_projection_mismatch",
            format!(
                "{role} cell {} does not reproduce both absolute arms plus A/B and A/A medians \
                 in its legacy threshold projection",
                cell.cell_id
            ),
        );
    }
}

fn projected_ratio_distribution(
    samples: &[crate::PerfRawSample],
) -> Option<crate::DistributionSummary> {
    let mut blocks = BTreeMap::<u64, (Option<f64>, Option<f64>)>::new();
    for sample in samples {
        let elapsed_ns = sample.ended_ns.checked_sub(sample.started_ns)?;
        if elapsed_ns == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ns = elapsed_ns as f64;
        let value = match sample.scope.semantics {
            crate::PerfMetricSemantics::Throughput => {
                #[allow(clippy::cast_precision_loss)]
                let work_units = sample.work_units? as f64;
                work_units * 1_000_000_000.0 / elapsed_ns
            }
            crate::PerfMetricSemantics::Duration => elapsed_ns,
            crate::PerfMetricSemantics::GaugeHigherIsBetter
            | crate::PerfMetricSemantics::GaugeLowerIsBetter => sample.observed_value?,
        };
        if !value.is_finite() || value <= 0.0 {
            return None;
        }
        let block = blocks.entry(sample.block_id).or_default();
        match sample.arm {
            crate::PerfSampleArm::Control => {
                if block.0.replace(value).is_some() {
                    return None;
                }
            }
            crate::PerfSampleArm::Treatment => {
                if block.1.replace(value).is_some() {
                    return None;
                }
            }
        }
    }
    let ratios = blocks
        .into_values()
        .map(|(control, treatment)| Some(treatment? / control?))
        .collect::<Option<Vec<_>>>()?;
    crate::DistributionSummary::from_samples(&ratios).ok()
}

fn compare_current_evidence_reproduction(
    candidate: &PerfEvidenceArtifact,
    rerun: &PerfEvidenceArtifact,
    state: &mut DecisionState,
) {
    if candidate.gate != rerun.gate {
        state.quarantine(
            "perf.ratchet.current_rerun_gate_mismatch",
            "candidate and rerun current evidence belong to different gates",
        );
        return;
    }
    if candidate.provenance.run_window != rerun.provenance.run_window {
        state.quarantine(
            "perf.ratchet.current_rerun_window_mismatch",
            "candidate and rerun current evidence must share one bounded measurement window",
        );
    }
    if candidate.provenance.run_id == rerun.provenance.run_id {
        state.quarantine(
            "perf.ratchet.current_rerun_identity_reused",
            "candidate and rerun current evidence must be distinct passes",
        );
    }
    if candidate.provenance.build != rerun.provenance.build {
        state.quarantine(
            "perf.ratchet.current_rerun_build_mismatch",
            "candidate and rerun current evidence must share the exact sealed build identity",
        );
    }
    if candidate.provenance.machine.fingerprint != rerun.provenance.machine.fingerprint {
        state.quarantine(
            "perf.ratchet.current_rerun_machine_mismatch",
            "candidate and rerun current evidence must share a machine fingerprint",
        );
    }
    if candidate.provenance.corpus != rerun.provenance.corpus {
        state.quarantine(
            "perf.ratchet.current_rerun_corpus_mismatch",
            "candidate and rerun current evidence must share invocation-level corpus provenance",
        );
    }

    let rerun_cells = rerun
        .cells
        .iter()
        .map(|cell| {
            (
                EvidenceCellKey::from_parts(&cell.spec.fixture, &cell.spec.metric, &cell.spec.unit),
                cell,
            )
        })
        .collect::<BTreeMap<_, _>>();
    for cell in candidate
        .cells
        .iter()
        .filter(|cell| cell.spec.role == EvidenceRole::Required)
    {
        let key =
            EvidenceCellKey::from_parts(&cell.spec.fixture, &cell.spec.metric, &cell.spec.unit);
        let Some(other) = rerun_cells.get(&key).copied() else {
            state.quarantine(
                "perf.ratchet.current_rerun_missing_cell",
                format!("rerun current evidence is missing {}", cell.cell_id),
            );
            continue;
        };
        let (
            EvidenceCellBody::Paired {
                paired: candidate_pair,
                ..
            },
            EvidenceCellBody::Paired {
                paired: rerun_pair, ..
            },
        ) = (&cell.body, &other.body)
        else {
            state.quarantine(
                "perf.ratchet.current_rerun_body_mismatch",
                format!("current evidence cell {} changed body kind", cell.cell_id),
            );
            continue;
        };
        if candidate.gate == PerfGate::Qg6 {
            compare_qg6_hierarchical_reproduction(cell, other, candidate_pair, rerun_pair, state);
            continue;
        }
        match candidate_pair.reproduces_within(rerun_pair) {
            Ok(true) => {}
            Ok(false) => state.quarantine(
                "perf.ratchet.current_reproduction_failed",
                format!(
                    "current-schema candidate and rerun disagree beyond the predeclared \
                     tolerance for {}",
                    cell.cell_id
                ),
            ),
            Err(error) => state.quarantine(
                "perf.ratchet.current_reproduction_incompatible",
                format!(
                    "current-schema candidate and rerun are incompatible for {}: {error}",
                    cell.cell_id
                ),
            ),
        }
    }
}

fn compare_qg6_hierarchical_reproduction(
    candidate_cell: &crate::EvidenceCell,
    rerun_cell: &crate::EvidenceCell,
    candidate_pair: &crate::PairedExperimentResult,
    rerun_pair: &crate::PairedExperimentResult,
    state: &mut DecisionState,
) {
    let (
        EvidenceCellBody::Paired {
            hierarchical: Some(candidate),
            hierarchical_null: Some(candidate_null),
            ..
        },
        EvidenceCellBody::Paired {
            hierarchical: Some(rerun),
            hierarchical_null: Some(rerun_null),
            ..
        },
    ) = (&candidate_cell.body, &rerun_cell.body)
    else {
        state.quarantine(
            "perf.ratchet.qg6_hierarchical_reproduction_missing",
            format!(
                "QG-6 reproduction requires hierarchical estimates for {}",
                candidate_cell.cell_id
            ),
        );
        return;
    };
    let same_grouping = |left: &crate::HierarchicalLatencyEstimate,
                         right: &crate::HierarchicalLatencyEstimate| {
        left.schema_version == right.schema_version
            && left.group_count == right.group_count
            && left
                .groups
                .iter()
                .map(|group| (group.group_id, group.pair_count))
                .eq(right
                    .groups
                    .iter()
                    .map(|group| (group.group_id, group.pair_count)))
    };
    let compatible = same_grouping(candidate, rerun)
        && same_grouping(candidate_null, rerun_null)
        && candidate_cell.cell_id == rerun_cell.cell_id
        && candidate_cell.spec == rerun_cell.spec
        && candidate_pair.config == rerun_pair.config
        && candidate_pair.scope == rerun_pair.scope
        && candidate_pair
            .provenance
            .same_reproduction_context(&rerun_pair.provenance);
    if !compatible {
        state.quarantine(
            "perf.ratchet.qg6_hierarchical_reproduction_incompatible",
            format!(
                "QG-6 candidate and rerun hierarchical inputs differ for {}",
                candidate_cell.cell_id
            ),
        );
        return;
    }
    for (label, candidate, rerun) in [
        ("A/B effect", candidate, rerun),
        ("A/A null", candidate_null, rerun_null),
    ] {
        let delta =
            (candidate.median_of_group_medians_log - rerun.median_of_group_medians_log).abs();
        if delta > candidate_pair.config.max_reproduction_delta_log {
            state.quarantine(
                "perf.ratchet.qg6_hierarchical_reproduction_failed",
                format!(
                    "QG-6 hierarchical {label} candidate and rerun differ by {delta:.6} \
                     log-ratio for {}",
                    candidate_cell.cell_id
                ),
            );
        }
    }
}

fn validate_paired_evidence(
    gate: PerfGate,
    cells: &BTreeMap<CellKey, &PerfCellResult>,
    role: &str,
    state: &mut DecisionState,
) {
    let paired = cells
        .iter()
        .filter(|(key, _)| key.engine == "paired_ab")
        .map(|(key, cell)| (key.clone(), *cell))
        .collect::<Vec<_>>();
    for (key, claim) in paired {
        let Some(metric_stem) = key.metric.strip_suffix("_quill_over_tantivy") else {
            state.fatal(
                "perf.ratchet.invalid_paired_metric",
                format!(
                    "{role} paired A/B row {}/{}/{} has no canonical metric suffix",
                    key.fixture, key.metric, key.engine
                ),
            );
            continue;
        };
        let null_key = CellKey {
            fixture: key.fixture.clone(),
            metric: format!("{metric_stem}_tantivy_over_tantivy"),
            engine: "paired_null".to_owned(),
            unit: "ratio".to_owned(),
        };
        let Some(null) = cells.get(&null_key).copied() else {
            state.quarantine(
                "perf.ratchet.missing_null_control",
                format!(
                    "{role} paired claim {}/{} has no same-invocation A/A null row",
                    key.fixture, key.metric
                ),
            );
            continue;
        };
        if gate == PerfGate::Qg6 {
            // QG-6 is a two-stage per-query estimand. Its flat legacy rows are
            // compatibility projections only; both A/B and A/A admission are
            // taken from the recomputable current-schema hierarchy.
            continue;
        }
        if !validate_null_control(null, role, state) {
            continue;
        }

        let null_floor = (null.distribution.median_ci95_low - 1.0)
            .abs()
            .max((null.distribution.median_ci95_high - 1.0).abs());

        let effect = (claim.distribution.p50 - 1.0).abs();
        let outside_null = claim.distribution.p50 < null.distribution.median_ci95_low
            || claim.distribution.p50 > null.distribution.median_ci95_high;
        if !outside_null || effect < PERF_NULL_MARGIN_MULTIPLIER * null_floor {
            state.quarantine(
                "perf.ratchet.inconclusive_paired_claim",
                format!(
                    "{role} {}/{} median {:.6} does not clear A/A median CI \
                     [{:.6}, {:.6}] with the required {:.1}x margin",
                    key.fixture,
                    key.metric,
                    claim.distribution.p50,
                    null.distribution.median_ci95_low,
                    null.distribution.median_ci95_high,
                    PERF_NULL_MARGIN_MULTIPLIER,
                ),
            );
        }
    }
}

fn expected_gate_keys(gate: PerfGate) -> BTreeSet<CellKey> {
    PerfMatrixSpec::complete()
        .for_gate(gate)
        .into_iter()
        .flat_map(|spec| {
            if gate == PerfGate::Qg10 {
                return vec![CellKey {
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    engine: "default_feature_graph".to_owned(),
                    unit: "nodes".to_owned(),
                }];
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
            vec![
                CellKey {
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    engine: absolute_engine.to_owned(),
                    unit: metric_unit(&spec.metric).to_owned(),
                },
                CellKey {
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    engine: oracle_engine.to_owned(),
                    unit: metric_unit(&spec.metric).to_owned(),
                },
                CellKey {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_quill_over_tantivy", spec.metric),
                    engine: "paired_ab".to_owned(),
                    unit: "ratio".to_owned(),
                },
                CellKey {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_tantivy_over_tantivy", spec.metric),
                    engine: "paired_null".to_owned(),
                    unit: "ratio".to_owned(),
                },
            ]
        })
        .collect()
}

fn metric_unit(metric: &str) -> &'static str {
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

fn candidate_is_complete(
    gate: PerfGate,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
) -> bool {
    candidate_cells.keys().cloned().collect::<BTreeSet<_>>() == expected_gate_keys(gate)
}

fn validate_complete_gate(
    gate: PerfGate,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
    activated: bool,
    state: &mut DecisionState,
) {
    if candidate_is_complete(gate, candidate_cells) {
        return;
    }
    let expected = expected_gate_keys(gate);
    let actual = candidate_cells.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected.difference(&actual).count();
    let extra = actual.difference(&expected).count();
    let message = format!(
        "{} promotion artifact is incomplete: {missing} missing cell rows, {extra} unexpected",
        gate.label()
    );
    if activated {
        state.block("perf.ratchet.incomplete_matrix", message);
    } else {
        state.quarantine("perf.ratchet.incomplete_matrix", message);
    }
}

fn compare_baseline(
    baseline: &PerfGateArtifact,
    candidate: &PerfGateArtifact,
    baseline_cells: &BTreeMap<CellKey, &PerfCellResult>,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
    mode: PerfRatchetMode,
    comparisons: &mut Vec<PerfCellComparison>,
    state: &mut DecisionState,
) {
    let explicit_bootstrap = baseline.cells.is_empty()
        && baseline.machine_fingerprint == "unmeasured"
        && baseline.bench_elf_sha256 == "unmeasured"
        && baseline.git_rev == "unmeasured"
        && baseline.run_window == "unmeasured"
        && baseline.run_id == "unmeasured";
    if explicit_bootstrap {
        if mode == PerfRatchetMode::RegressionAlarm {
            state.quarantine(
                "perf.ratchet.bootstrap_baseline",
                "the committed baseline is an explicit unmeasured bootstrap placeholder",
            );
        } else {
            state.note(
                "perf.ratchet.bootstrap_promotion",
                "an otherwise-Allow promotion will establish the first measured baseline",
            );
        }
        return;
    }
    if baseline.cells.is_empty() || baseline.machine_fingerprint == "unmeasured" {
        state.fatal(
            "perf.ratchet.invalid_bootstrap_baseline",
            "an empty or unmeasured baseline must use the complete explicit bootstrap identity",
        );
        return;
    }
    validate_paired_evidence(baseline.gate, baseline_cells, "baseline", state);
    if baseline.machine_fingerprint != candidate.machine_fingerprint {
        state.quarantine(
            "perf.ratchet.machine_mismatch",
            format!(
                "baseline machine {:?} differs from candidate machine {:?}",
                baseline.machine_fingerprint, candidate.machine_fingerprint
            ),
        );
        return;
    }
    if baseline.corpus_manifest_hash != candidate.corpus_manifest_hash {
        state.quarantine(
            "perf.ratchet.corpus_mismatch",
            "baseline and candidate corpus manifest hashes differ",
        );
        return;
    }

    for (key, current) in candidate_cells {
        let Some(previous) = baseline_cells.get(key) else {
            state.quarantine(
                "perf.ratchet.cell_without_baseline",
                format!(
                    "candidate {}/{}/{} has no committed baseline row",
                    key.fixture, key.metric, key.engine
                ),
            );
            continue;
        };
        if candidate.gate == PerfGate::Qg6
            && matches!(key.engine.as_str(), "paired_ab" | "paired_null")
        {
            // QG-6's flat paired rows are compatibility projections. Current
            // hierarchical evidence owns effect and null inference.
            continue;
        }
        if key.engine == "paired_null" {
            let _ = validate_null_control(current, "candidate", state);
            continue;
        }

        let regression_pct = directional_regression_pct(
            previous.distribution.p50,
            current.distribution.p50,
            higher_is_better(&key.metric),
        );
        let robust_z = robust_z(previous, current);
        let threshold_exceeded = regression_pct > PERF_MAX_REGRESSION_PCT;
        comparisons.push(PerfCellComparison {
            fixture: key.fixture.clone(),
            metric: key.metric.clone(),
            engine: key.engine.clone(),
            baseline_value: previous.distribution.p50,
            candidate_value: current.distribution.p50,
            regression_pct,
            robust_z,
            threshold_exceeded,
        });

        if matches!(key.engine.as_str(), "tantivy" | "quill_tokenizer_null") {
            if relative_delta_pct(previous.distribution.p50, current.distribution.p50)
                > PERF_MAX_REGRESSION_PCT
                && confidence_intervals_show_shift(previous, current)
            {
                state.quarantine(
                    "perf.ratchet.oracle_drift",
                    format!(
                        "oracle row {}/{}/{} moved more than {:.1}%; rerun on a quiet same-class host",
                        key.fixture, key.metric, key.engine, PERF_MAX_REGRESSION_PCT
                    ),
                );
            }
            continue;
        }

        if threshold_exceeded {
            let message = format!(
                "{}/{}/{} regressed {:.3}% (baseline median CI [{:.6}, {:.6}], \
                 candidate median CI [{:.6}, {:.6}], robust_z provenance={robust_z:.3})",
                key.fixture,
                key.metric,
                key.engine,
                regression_pct,
                previous.distribution.median_ci95_low,
                previous.distribution.median_ci95_high,
                current.distribution.median_ci95_low,
                current.distribution.median_ci95_high,
            );
            if confidence_interval_confirms_regression(
                previous,
                current,
                higher_is_better(&key.metric),
            ) {
                state.block("perf.ratchet.regression_detected", message);
            } else {
                state.quarantine("perf.ratchet.inconclusive_regression", message);
            }
        }
    }
}

fn compare_reproduction(
    candidate: &PerfGateArtifact,
    rerun: &PerfGateArtifact,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
    rerun_cells: &BTreeMap<CellKey, &PerfCellResult>,
    state: &mut DecisionState,
) {
    if candidate.git_rev != rerun.git_rev {
        state.quarantine(
            "perf.ratchet.rerun_revision_mismatch",
            "candidate and rerun must come from the same git revision",
        );
    }
    if candidate.bench_elf_sha256 != rerun.bench_elf_sha256 {
        state.quarantine(
            "perf.ratchet.rerun_elf_mismatch",
            "candidate and rerun must self-report the same benchmark ELF SHA-256",
        );
    }
    if candidate.run_window != rerun.run_window {
        state.quarantine(
            "perf.ratchet.rerun_window_mismatch",
            "candidate and rerun must come from the same bounded measurement window",
        );
    }
    if candidate.run_id == rerun.run_id {
        state.quarantine(
            "perf.ratchet.rerun_identity_reused",
            "candidate and rerun must be distinct passes, not the same artifact reused twice",
        );
    }
    if candidate.machine_fingerprint != rerun.machine_fingerprint {
        state.quarantine(
            "perf.ratchet.rerun_machine_mismatch",
            "candidate and rerun must come from the same machine fingerprint",
        );
    }
    if candidate.corpus_manifest_hash != rerun.corpus_manifest_hash {
        state.quarantine(
            "perf.ratchet.rerun_corpus_mismatch",
            "candidate and rerun corpus manifest hashes differ",
        );
    }
    if candidate_cells.len() != rerun_cells.len() {
        state.quarantine(
            "perf.ratchet.rerun_shape_mismatch",
            "candidate and rerun contain different cell counts",
        );
    }
    for (key, first) in candidate_cells {
        if candidate.gate == PerfGate::Qg6
            && matches!(key.engine.as_str(), "paired_ab" | "paired_null")
        {
            // The verified current-schema two-stage estimates are the only
            // QG-6 effect and null reproduction inputs. Flat compatibility
            // projections are retained for legacy consumers, never for this
            // decision.
            continue;
        }
        let Some(second) = rerun_cells.get(key) else {
            state.quarantine(
                "perf.ratchet.rerun_missing_cell",
                format!(
                    "rerun is missing {}/{}/{}",
                    key.fixture, key.metric, key.engine
                ),
            );
            continue;
        };
        let delta = relative_delta_pct(first.distribution.p50, second.distribution.p50);
        if delta > PERF_MAX_REPRODUCTION_DELTA_PCT {
            state.quarantine(
                "perf.ratchet.reproduction_failed",
                format!(
                    "{}/{}/{} candidate and rerun medians differ by {delta:.3}%",
                    key.fixture, key.metric, key.engine
                ),
            );
        }
    }
}

fn validate_null_control(cell: &PerfCellResult, role: &str, state: &mut DecisionState) -> bool {
    let contains_identity =
        cell.distribution.median_ci95_low <= 1.0 && 1.0 <= cell.distribution.median_ci95_high;
    if !contains_identity {
        state.quarantine(
            "perf.ratchet.invalid_null_control",
            format!(
                "{role} {}/{}/{} A/A median CI [{:.6}, {:.6}] does not contain 1.0 \
                 (cv_pct={:.3} is provenance only)",
                cell.fixture,
                cell.metric,
                cell.engine,
                cell.distribution.median_ci95_low,
                cell.distribution.median_ci95_high,
                cell.distribution.cv_pct,
            ),
        );
    }
    contains_identity
}

fn confidence_interval_confirms_regression(
    baseline: &PerfCellResult,
    candidate: &PerfCellResult,
    higher_is_better: bool,
) -> bool {
    let threshold = PERF_MAX_REGRESSION_PCT / 100.0;
    if higher_is_better {
        candidate.distribution.median_ci95_high
            < baseline.distribution.median_ci95_low * (1.0 - threshold)
    } else {
        candidate.distribution.median_ci95_low
            > baseline.distribution.median_ci95_high * (1.0 + threshold)
    }
}

fn confidence_intervals_show_shift(baseline: &PerfCellResult, candidate: &PerfCellResult) -> bool {
    let threshold = PERF_MAX_REGRESSION_PCT / 100.0;
    candidate.distribution.median_ci95_low
        > baseline.distribution.median_ci95_high * (1.0 + threshold)
        || candidate.distribution.median_ci95_high
            < baseline.distribution.median_ci95_low * (1.0 - threshold)
}

fn higher_is_better(metric: &str) -> bool {
    metric.contains("docs_per_second")
        || metric.contains("updates_per_second")
        || metric.contains("tokenize_docs_per_second")
}

fn directional_regression_pct(baseline: f64, candidate: f64, higher_is_better: bool) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        return if candidate.abs() <= f64::EPSILON {
            0.0
        } else {
            100.0
        };
    }
    let signed = if higher_is_better {
        baseline - candidate
    } else {
        candidate - baseline
    };
    signed / baseline.abs() * 100.0
}

fn relative_delta_pct(left: f64, right: f64) -> f64 {
    if left.abs() <= f64::EPSILON {
        if right.abs() <= f64::EPSILON {
            0.0
        } else {
            100.0
        }
    } else {
        (right - left).abs() / left.abs() * 100.0
    }
}

fn robust_z(baseline: &PerfCellResult, candidate: &PerfCellResult) -> f64 {
    let mad = baseline
        .distribution
        .mad
        .max(candidate.distribution.mad)
        .max(baseline.distribution.p50.abs() * 0.001)
        .max(MAD_EPSILON);
    (candidate.distribution.p50 - baseline.distribution.p50).abs() / (MAD_SCALE * mad)
}

struct GateTargetEvaluator<'a, 'b> {
    artifact: &'a PerfGateArtifact,
    cells: &'b BTreeMap<CellKey, &'a PerfCellResult>,
    activated: bool,
    state: &'b mut DecisionState,
}

impl GateTargetEvaluator<'_, '_> {
    fn summary(
        &mut self,
        fixture: &str,
        metric: &str,
        engine: &str,
    ) -> Option<DistributionSummary> {
        let key = self
            .cells
            .keys()
            .find(|key| key.fixture == fixture && key.metric == metric && key.engine == engine)
            .cloned();
        let Some(key) = key else {
            self.state.quarantine(
                "perf.ratchet.target_cell_missing",
                format!(
                    "{} target requires {fixture}/{metric}/{engine}",
                    self.artifact.gate
                ),
            );
            return None;
        };
        self.cells.get(&key).map(|cell| cell.distribution.clone())
    }

    fn value(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
        self.summary(fixture, metric, engine)
            .map(|summary| summary.p50)
    }

    fn median_ci95(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<(f64, f64)> {
        self.summary(fixture, metric, engine)
            .map(|summary| (summary.median_ci95_low, summary.median_ci95_high))
    }

    fn p95(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
        self.summary(fixture, metric, engine)
            .map(|summary| summary.p95)
    }

    fn p99(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
        self.summary(fixture, metric, engine)
            .map(|summary| summary.p99)
    }

    fn target(&mut self, passed: bool, message: impl Into<String>) {
        if passed {
            return;
        }
        if self.activated {
            self.state.block("perf.ratchet.gate_target_missed", message);
        } else {
            self.state
                .quarantine("perf.ratchet.provisional_target_missed", message);
        }
    }

    fn target_higher_ci(
        &mut self,
        ci_low: f64,
        ci_high: f64,
        threshold: f64,
        message: impl Into<String>,
    ) {
        if ci_low >= threshold {
            return;
        }
        let message = message.into();
        if ci_high < threshold {
            self.target(false, message);
        } else {
            self.state.quarantine(
                "perf.ratchet.gate_target_ci_inconclusive",
                format!("{message}; the median CI crosses {threshold:.6}"),
            );
        }
    }

    fn target_lower_ci(
        &mut self,
        ci_low: f64,
        ci_high: f64,
        threshold: f64,
        message: impl Into<String>,
    ) {
        if ci_high <= threshold {
            return;
        }
        let message = message.into();
        if ci_low > threshold {
            self.target(false, message);
        } else {
            self.state.quarantine(
                "perf.ratchet.gate_target_ci_inconclusive",
                format!("{message}; the median CI crosses {threshold:.6}"),
            );
        }
    }

    fn target_interval_ci(
        &mut self,
        ci_low: f64,
        ci_high: f64,
        allowed_low: f64,
        allowed_high: f64,
        message: impl Into<String>,
    ) {
        if ci_low >= allowed_low && ci_high <= allowed_high {
            return;
        }
        let message = message.into();
        if ci_high < allowed_low || ci_low > allowed_high {
            self.target(false, message);
        } else {
            self.state.quarantine(
                "perf.ratchet.gate_target_ci_inconclusive",
                format!(
                    "{message}; the median CI overlaps but is not contained in \
                     [{allowed_low:.6}, {allowed_high:.6}]"
                ),
            );
        }
    }
}

fn evaluate_gate_targets(
    artifact: &PerfGateArtifact,
    cells: &BTreeMap<CellKey, &PerfCellResult>,
    current_evidence: Option<&PerfEvidenceArtifact>,
    activated: bool,
    state: &mut DecisionState,
) {
    let mut target = GateTargetEvaluator {
        artifact,
        cells,
        activated,
        state,
    };
    match artifact.gate {
        PerfGate::Qg1 => evaluate_qg1(&mut target),
        PerfGate::Qg2 => evaluate_qg2(&mut target),
        PerfGate::Qg3 => evaluate_qg3(&mut target),
        PerfGate::Qg4 => evaluate_qg4(&mut target),
        PerfGate::Qg5 => evaluate_qg5(&mut target),
        PerfGate::Qg6 => evaluate_qg6(&mut target, current_evidence),
        PerfGate::Qg7 => evaluate_qg7(&mut target),
        PerfGate::Qg8 => evaluate_qg8(&mut target),
        PerfGate::Qg9 => evaluate_qg9(&mut target),
        PerfGate::Qg10 => evaluate_qg10(&mut target),
    }
}

fn evaluate_qg1(target: &mut GateTargetEvaluator<'_, '_>) {
    for corpus in ["medium", "xlarge"] {
        let fixture = format!("bulk/{corpus}/8/positions_on");
        if let Some((ci_low, ci_high)) =
            target.median_ci95(&fixture, "docs_per_second_quill_over_tantivy", "paired_ab")
        {
            target.target_higher_ci(
                ci_low,
                ci_high,
                3.0,
                format!(
                    "QG-1 {fixture} median-ratio CI [{ci_low:.6}, {ci_high:.6}] does not clear 3.0"
                ),
            );
        }
        let tokenize_fixture = format!("tokenize_only/{corpus}");
        if let (Some((index_low, index_high)), Some((tokenize_low, tokenize_high))) = (
            target.median_ci95(&fixture, "docs_per_second", "quill"),
            target.median_ci95(
                &tokenize_fixture,
                "tokenize_docs_per_second",
                "quill_tokenizer",
            ),
        ) {
            let ceiling_low = index_low / tokenize_high.max(f64::MIN_POSITIVE);
            let ceiling_high = index_high / tokenize_low.max(f64::MIN_POSITIVE);
            target.target_higher_ci(
                ceiling_low,
                ceiling_high,
                0.60,
                format!(
                    "QG-1 {corpus} indexing/tokenize ratio CI \
                     [{ceiling_low:.6}, {ceiling_high:.6}] does not clear 0.60"
                ),
            );
        }
    }
}

fn evaluate_qg2(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some((ci_low, ci_high)) = target.median_ci95(
        "bulk/medium/1/positions_on",
        "docs_per_second_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target_higher_ci(
            ci_low,
            ci_high,
            1.5,
            format!(
                "QG-2 single-thread median-ratio CI [{ci_low:.6}, {ci_high:.6}] does not clear 1.5"
            ),
        );
    }
}

fn evaluate_qg3(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some((initial_low, initial_high)) =
        target.median_ci95("watch/medium/initial", "docs_per_second", "quill")
    {
        target.target_higher_ci(
            initial_low,
            initial_high,
            20_000.0,
            format!(
                "QG-3 initial throughput median CI [{initial_low:.3}, {initial_high:.3}] docs/s \
                 does not clear 20000"
            ),
        );
    }
    for topology in ["inprocess", "freshprocess"] {
        let fixture = format!("watch/medium/5000/{topology}");
        if let Some((updates_low, updates_high)) =
            target.median_ci95(&fixture, "updates_per_second", "quill")
        {
            target.target_higher_ci(
                updates_low,
                updates_high,
                5_000.0,
                format!(
                    "QG-3 {topology} throughput median CI \
                     [{updates_low:.3}, {updates_high:.3}] updates/s does not clear 5000"
                ),
            );
        }
        if let Some(p95) = target.p95(&fixture, "update_to_searchable_ms", "quill") {
            target.target(
                p95 <= 25.0,
                format!("QG-3 {topology} update-to-searchable p95 {p95:.3}ms exceeds 25ms"),
            );
        }
    }
    if let Some((ci_low, ci_high)) = target.median_ci95(
        "watch/medium/5000/inprocess",
        "update_to_searchable_ms_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target_lower_ci(
            ci_low,
            ci_high,
            0.25,
            format!(
                "QG-3 in-process visibility median-ratio CI [{ci_low:.6}, {ci_high:.6}] does not \
                 clear the <=0.25 target"
            ),
        );
    }
}

fn evaluate_qg4(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some(p99) = target.p99("commit/100000/warm", "commit_latency_ms", "quill") {
        target.target(
            p99 <= 50.0,
            format!("QG-4 sealed commit p99 {p99:.3}ms exceeds 50ms"),
        );
    }
}

fn evaluate_qg5(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some((ci_low, ci_high)) = target.median_ci95(
        "compaction/xlarge/20pct",
        "wall_clock_ms_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target_lower_ci(
            ci_low,
            ci_high,
            0.20,
            format!(
                "QG-5 20% compaction median-ratio CI [{ci_low:.6}, {ci_high:.6}] does not clear \
                 the <=0.20 target"
            ),
        );
    }
}

fn evaluate_qg6(
    target: &mut GateTargetEvaluator<'_, '_>,
    current_evidence: Option<&PerfEvidenceArtifact>,
) {
    let fixtures = target
        .cells
        .keys()
        .filter(|key| key.engine == "paired_ab" && key.metric == "latency_ms_quill_over_tantivy")
        .map(|key| key.fixture.clone())
        .collect::<Vec<_>>();
    for fixture in fixtures {
        let hierarchical =
            current_evidence.and_then(|artifact| exact_qg6_hierarchical_cell(artifact, &fixture));
        if let Some((effect, null)) = hierarchical {
            let ci_low = effect.ci95_low_ratio;
            let ci_high = effect.ci95_high_ratio;
            target.target_interval_ci(
                ci_low,
                ci_high,
                0.90,
                1.10,
                format!(
                    "QG-6 {fixture} hierarchical median-ratio CI \
                    [{ci_low:.6}, {ci_high:.6}] is not contained in [0.90, 1.10]"
                ),
            );
            let null_floor = (null.ci95_low_ratio - 1.0)
                .abs()
                .max((null.ci95_high_ratio - 1.0).abs());
            let null_contains_identity = null.ci95_low_ratio <= 1.0 && 1.0 <= null.ci95_high_ratio;
            if !null_contains_identity || PERF_NULL_MARGIN_MULTIPLIER * null_floor > 0.10 {
                target.state.quarantine(
                    "perf.ratchet.inconclusive_equivalence",
                    format!(
                        "QG-6 {fixture} cannot establish ±10% equivalence: hierarchical A/A \
                         median-ratio CI [{:.6}, {:.6}] fails identity containment or the required \
                         {:.1}x null margin",
                        null.ci95_low_ratio, null.ci95_high_ratio, PERF_NULL_MARGIN_MULTIPLIER,
                    ),
                );
            }
        } else {
            target.state.quarantine(
                "perf.ratchet.qg6_hierarchical_evidence_missing",
                format!(
                    "QG-6 {fixture} requires a verified two-stage hierarchical estimate; flat \
                     threshold projections are not decision inputs"
                ),
            );
        }
        if let (Some(quill_p99), Some(oracle_p99)) = (
            target.p99(&fixture, "latency_ms", "quill"),
            target.p99(&fixture, "latency_ms", "tantivy"),
        ) {
            target.target(
                quill_p99 <= oracle_p99,
                format!(
                    "QG-6 {fixture} Quill p99 {quill_p99:.6}ms exceeds oracle {oracle_p99:.6}ms"
                ),
            );
        }
    }
}

fn exact_qg6_hierarchical_cell<'a>(
    artifact: &'a PerfEvidenceArtifact,
    fixture: &str,
) -> Option<(
    &'a crate::HierarchicalLatencyEstimate,
    &'a crate::HierarchicalLatencyEstimate,
)> {
    let expected_cell_id = format!("{}/{fixture}/latency_ms", PerfGate::Qg6);
    let mut exact = artifact.cells.iter().filter(|cell| {
        cell.cell_id == expected_cell_id
            && cell.spec.gate == PerfGate::Qg6
            && cell.spec.fixture == fixture
            && cell.spec.metric == "latency_ms"
            && cell.spec.unit == "ms"
            && cell.spec.role == EvidenceRole::Required
            && cell
                .spec
                .input_identity
                .as_ref()
                .is_some_and(|identity| identity.validate().is_ok())
    });
    let cell = exact.next()?;
    if exact.next().is_some() {
        return None;
    }
    match &cell.body {
        EvidenceCellBody::Paired {
            paired,
            hierarchical: Some(estimate),
            hierarchical_null: Some(null),
            ..
        } if paired.provenance.corpus_sha256 == artifact.provenance.corpus.corpus_sha256
            && paired.provenance.input_identity == cell.spec.input_identity =>
        {
            Some((estimate, null))
        }
        _ => None,
    }
}

fn evaluate_qg7(target: &mut GateTargetEvaluator<'_, '_>) {
    for corpus in ["medium", "xlarge"] {
        let memory_fixture = format!("memory/{corpus}/positions_on");
        if let Some(ratio) = target.value(
            &memory_fixture,
            "peak_rss_bytes_quill_over_tantivy",
            "paired_ab",
        ) {
            target.target(
                ratio <= 1.0,
                format!("QG-7 {memory_fixture} RSS ratio {ratio:.6} exceeds 1.0"),
            );
        }
        let on_fixture = format!("size/{corpus}/positions_on");
        if let Some(ratio) = target.value(
            &on_fixture,
            "index_bytes_per_document_quill_over_tantivy",
            "paired_ab",
        ) {
            target.target(
                ratio <= 1.15,
                format!("QG-7 {on_fixture} bytes/doc ratio {ratio:.6} exceeds 1.15"),
            );
        }
        let off_fixture = format!("size/{corpus}/positions_off");
        if let (Some(quill_off), Some(oracle_on)) = (
            target.value(&off_fixture, "index_bytes_per_document", "quill"),
            target.value(&on_fixture, "index_bytes_per_document", "tantivy"),
        ) {
            let ratio = quill_off / oracle_on.max(f64::MIN_POSITIVE);
            target.target(
                ratio <= 0.80,
                format!("QG-7 {corpus} positions-off/default-oracle ratio {ratio:.6} exceeds 0.80"),
            );
        }
    }
}

fn evaluate_qg8(target: &mut GateTargetEvaluator<'_, '_>) {
    if let (Some(four), Some(sixteen)) = (
        target.value("scaling/xlarge/4/positions_on", "docs_per_second", "quill"),
        target.value("scaling/xlarge/16/positions_on", "docs_per_second", "quill"),
    ) {
        let ratio = sixteen / four.max(f64::MIN_POSITIVE);
        target.target(
            ratio >= 1.8,
            format!("QG-8 16-thread/4-thread scaling {ratio:.6} is below 1.8"),
        );
    }
}

fn evaluate_qg9(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some(open) = target.value("cold_open/xlarge/default", "open_latency_ms", "quill") {
        target.target(
            open <= 50.0,
            format!("QG-9 cold-open median {open:.3}ms exceeds 50ms"),
        );
    }
}

fn evaluate_qg10(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some(nodes) = target.value(
        "dependency_surface/default_lexical",
        "tantivy_nodes",
        "default_feature_graph",
    ) {
        target.target(
            nodes == 0.0,
            format!("QG-10 default feature graph still contains {nodes:.0} Tantivy nodes"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::PerfInputIdentity;
    use crate::{
        BuildIdentity, CorpusIdentity, DistributionSummary, EvidenceCell, EvidenceCellSpec,
        EvidencePolicy, EvidenceProvenance, MachineIdentity, PairedEstimatorConfig,
        PeakRssEvidence, PerfCellResult, PerfMetricSemantics, PerfOperationScope, PerfRawSample,
        PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
        estimate_paired_experiment, seeded_balanced_pair_order,
    };
    use sha2::{Digest, Sha256};

    fn seal_evidence(evidence: &mut PerfEvidenceArtifact) {
        const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut unsealed = evidence.clone();
        unsealed.artifact_sha256.clear();
        let canonical =
            serde_json::to_string_pretty(&unsealed).expect("serialize unsealed evidence");
        let digest = Sha256::digest(canonical.as_bytes());
        let mut seal = String::with_capacity(digest.len() * 2);
        for byte in digest {
            seal.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            seal.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
        evidence.artifact_sha256 = seal;
        evidence
            .verify_integrity()
            .expect("test evidence must be integrity-valid after sealing");
    }

    fn distribution(value: f64) -> DistributionSummary {
        DistributionSummary {
            value,
            p50: value,
            median_ci95_low: value,
            median_ci95_high: value,
            p95: value,
            p99: value,
            mad: value.abs() * 0.002,
            cv_pct: 1.0,
            runs: PERF_MIN_RUNS,
        }
    }

    fn qg2_artifact(revision: &str, quill: f64, oracle: f64) -> PerfGateArtifact {
        let ratio = quill / oracle;
        PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg2,
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            git_rev: revision.to_owned(),
            run_window: "test-window".to_owned(),
            run_id: format!("{revision}-{quill}-{oracle}"),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: "b".repeat(64),
            cells: vec![
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second".to_owned(),
                    engine: "quill".to_owned(),
                    unit: "docs/s".to_owned(),
                    distribution: distribution(quill),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second".to_owned(),
                    engine: "tantivy".to_owned(),
                    unit: "docs/s".to_owned(),
                    distribution: distribution(oracle),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second_quill_over_tantivy".to_owned(),
                    engine: "paired_ab".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: distribution(ratio),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second_tantivy_over_tantivy".to_owned(),
                    engine: "paired_null".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: distribution(1.0),
                },
            ],
            laws_attested: true,
        }
    }

    fn qg2_current_pair(
        revision: &str,
        run_id: &str,
        quill: f64,
        oracle: f64,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let scope = PerfOperationScope {
            operation_id: "qg2.bulk_index".to_owned(),
            version: 1,
            semantics: PerfMetricSemantics::GaugeHigherIsBetter,
            unit: "docs/s".to_owned(),
        };
        let sample_provenance = PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "c".repeat(64),
            corpus_sha256: "a".repeat(64),
            input_identity: None,
            worker_id: "linux-x86_64-test".to_owned(),
            build_profile: "release-perf".to_owned(),
        };
        let order = seeded_balanced_pair_order(PERF_MIN_RUNS, 0x5152_0002).expect("balanced order");
        let stream = |control: f64, treatment: f64, sample_base: u64| {
            let mut samples = Vec::with_capacity(PERF_MIN_RUNS * 2);
            for (index, first) in order.iter().copied().enumerate() {
                let block_id = u64::try_from(index).expect("block");
                let first_start = block_id * 1_000;
                let second_start = first_start + 200;
                let control_first = first == PerfSampleArm::Control;
                for (offset, arm, value, is_first) in [
                    (0_u64, PerfSampleArm::Control, control, control_first),
                    (1_u64, PerfSampleArm::Treatment, treatment, !control_first),
                ] {
                    let started_ns = if is_first { first_start } else { second_start };
                    samples.push(PerfRawSample {
                        block_id,
                        sample_id: sample_base + block_id * 2 + offset,
                        arm,
                        order: if is_first {
                            PerfSampleOrder::First
                        } else {
                            PerfSampleOrder::Second
                        },
                        phase: PerfSamplePhase::Measurement,
                        scope: scope.clone(),
                        provenance: sample_provenance.clone(),
                        started_ns,
                        ended_ns: started_ns + 100,
                        work_units: None,
                        byte_count: None,
                        observed_value: Some(value),
                        group_id: None,
                    });
                }
            }
            samples
        };
        let effect_samples = stream(oracle, quill, 0);
        let null_samples = stream(oracle, oracle, 100_000);
        let paired = estimate_paired_experiment(
            &effect_samples,
            &null_samples,
            &PairedEstimatorConfig::predeclared(0x5152_0002),
        )
        .expect("paired evidence");
        let cell = EvidenceCell::evaluate(
            EvidenceCellSpec {
                gate: PerfGate::Qg2,
                fixture: "bulk/medium/1/positions_on".to_owned(),
                metric: "docs_per_second".to_owned(),
                unit: "docs/s".to_owned(),
                role: EvidenceRole::Required,
                input_identity: None,
                cold_cache: None,
            },
            paired,
            &EvidencePolicy::predeclared(),
        )
        .expect("evidence cell");
        let paired = match &cell.body {
            EvidenceCellBody::Paired { paired, .. } => Some(paired),
            EvidenceCellBody::Facts { .. } => None,
        }
        .expect("QG-2 must be paired");
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg2,
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            git_rev: revision.to_owned(),
            run_window: "test-window".to_owned(),
            run_id: run_id.to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: "b".repeat(64),
            cells: vec![
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second".to_owned(),
                    engine: "quill".to_owned(),
                    unit: "docs/s".to_owned(),
                    distribution: paired.effect.treatment.clone(),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second".to_owned(),
                    engine: "tantivy".to_owned(),
                    unit: "docs/s".to_owned(),
                    distribution: paired.effect.control.clone(),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second_quill_over_tantivy".to_owned(),
                    engine: "paired_ab".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: projected_ratio_distribution(&paired.effect_samples)
                        .expect("effect projection"),
                },
                PerfCellResult {
                    fixture: "bulk/medium/1/positions_on".to_owned(),
                    metric: "docs_per_second_tantivy_over_tantivy".to_owned(),
                    engine: "paired_null".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: projected_ratio_distribution(&paired.null_samples)
                        .expect("null projection"),
                },
            ],
            laws_attested: true,
        };
        let mut evidence = PerfEvidenceArtifact::assemble(
            PerfGate::Qg2,
            EvidencePolicy::predeclared(),
            EvidenceProvenance {
                run_id: run_id.to_owned(),
                run_window: "test-window".to_owned(),
                manifest_sha256: "b".repeat(64),
                build: BuildIdentity {
                    executable_sha256: "c".repeat(64),
                    git_revision: revision.to_owned(),
                    git_dirty: false,
                    worktree_state_sha256: None,
                    cargo_lock_sha256: Some("e".repeat(64)),
                    rustc_version: "rustc test".to_owned(),
                    target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                    build_profile: "release-perf".to_owned(),
                    cargo_features: vec!["perf-harness".to_owned()],
                },
                machine: MachineIdentity {
                    fingerprint: "linux-x86_64-test".to_owned(),
                    os: "linux".to_owned(),
                    arch: "x86_64".to_owned(),
                    logical_cpus: 8,
                    cpu_governor: None,
                    load_average_start: None,
                    load_average_end: None,
                },
                peak_rss: PeakRssEvidence {
                    method: "unsupported".to_owned(),
                    bytes: None,
                },
                corpus: CorpusIdentity {
                    corpus_sha256: "a".repeat(64),
                    query_set_sha256: None,
                    qrels_sha256: None,
                    document_count: 100_000,
                    content_bytes: None,
                    generator_seed: 42,
                    generator_revision: "test-v1".to_owned(),
                },
            },
            vec![cell],
        )
        .expect("evidence artifact");
        seal_evidence(&mut evidence);
        (artifact, evidence)
    }

    fn qg6_current_pair(
        run_id: &str,
        group_ratios: [[f64; 3]; 4],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_current_pair_with_null(run_id, group_ratios, [[1.0; 3]; 4])
    }

    fn qg6_current_pair_with_null(
        run_id: &str,
        effect_group_ratios: [[f64; 3]; 4],
        null_group_ratios: [[f64; 3]; 4],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let fixture = "query/identifier/k10/medium";
        let scope = PerfOperationScope {
            operation_id: "qg6.prepared_query".to_owned(),
            version: 1,
            semantics: PerfMetricSemantics::GaugeLowerIsBetter,
            unit: "ms".to_owned(),
        };
        let input_identity = PerfInputIdentity {
            prepared_corpus_sha256: "0".repeat(64),
            query_manifest_sha256: "1".repeat(64),
            config_contract_sha256: "2".repeat(64),
            query_group_count: crate::QG6_QUERY_GROUPS,
            query_group_ids: crate::QG6_QUERY_GROUP_IDS.to_vec(),
        };
        let sample_provenance = PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "c".repeat(64),
            corpus_sha256: "a".repeat(64),
            input_identity: Some(input_identity.clone()),
            worker_id: "linux-x86_64-test".to_owned(),
            build_profile: "release-perf".to_owned(),
        };
        let order = seeded_balanced_pair_order(12, 0x5156_0006).expect("balanced QG-6 pair order");
        let stream = |group_ratios: &[[f64; 3]; 4], sample_base: u64| {
            let mut samples = Vec::with_capacity(24);
            let mut ordinal = 0_usize;
            for (group_index, ratios) in group_ratios.iter().enumerate() {
                for ratio in ratios {
                    let block_id = u64::try_from(ordinal).expect("QG-6 block");
                    let first_start = block_id * 1_000;
                    let second_start = first_start + 200;
                    let control_first = order[ordinal] == PerfSampleArm::Control;
                    let control = if *ratio > 1.15 { 1.0 } else { 100.0 };
                    let treatment = control * *ratio;
                    for (offset, arm, value, is_first) in [
                        (0_u64, PerfSampleArm::Control, control, control_first),
                        (1_u64, PerfSampleArm::Treatment, treatment, !control_first),
                    ] {
                        let started_ns = if is_first { first_start } else { second_start };
                        samples.push(PerfRawSample {
                            block_id,
                            sample_id: sample_base + block_id * 2 + offset,
                            arm,
                            order: if is_first {
                                PerfSampleOrder::First
                            } else {
                                PerfSampleOrder::Second
                            },
                            phase: PerfSamplePhase::Measurement,
                            scope: scope.clone(),
                            provenance: sample_provenance.clone(),
                            started_ns,
                            ended_ns: started_ns + 100,
                            work_units: None,
                            byte_count: None,
                            observed_value: Some(value),
                            group_id: Some(u64::try_from(group_index).expect("QG-6 group")),
                        });
                    }
                    ordinal += 1;
                }
            }
            samples
        };
        let effect_samples = stream(&effect_group_ratios, 0);
        let null_samples = stream(&null_group_ratios, 100_000);
        let paired = estimate_paired_experiment(
            &effect_samples,
            &null_samples,
            &PairedEstimatorConfig::predeclared(0x5156_0006),
        )
        .expect("QG-6 paired evidence");
        let cell = EvidenceCell::evaluate(
            EvidenceCellSpec {
                gate: PerfGate::Qg6,
                fixture: fixture.to_owned(),
                metric: "latency_ms".to_owned(),
                unit: "ms".to_owned(),
                role: EvidenceRole::Required,
                input_identity: Some(input_identity),
                cold_cache: None,
            },
            paired,
            &EvidencePolicy::predeclared(),
        )
        .expect("QG-6 evidence cell");
        let paired = match &cell.body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg6,
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: "linux-x86_64-test".to_owned(),
            git_rev: "new".to_owned(),
            run_window: "test-window".to_owned(),
            run_id: run_id.to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: "b".repeat(64),
            cells: vec![
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms".to_owned(),
                    engine: "quill".to_owned(),
                    unit: "ms".to_owned(),
                    distribution: paired.effect.treatment.clone(),
                },
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms".to_owned(),
                    engine: "tantivy".to_owned(),
                    unit: "ms".to_owned(),
                    distribution: paired.effect.control.clone(),
                },
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms_quill_over_tantivy".to_owned(),
                    engine: "paired_ab".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: projected_ratio_distribution(&paired.effect_samples)
                        .expect("QG-6 effect projection"),
                },
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms_tantivy_over_tantivy".to_owned(),
                    engine: "paired_null".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: projected_ratio_distribution(&paired.null_samples)
                        .expect("QG-6 null projection"),
                },
            ],
            laws_attested: true,
        };
        let mut evidence = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            EvidencePolicy::predeclared(),
            EvidenceProvenance {
                run_id: run_id.to_owned(),
                run_window: "test-window".to_owned(),
                manifest_sha256: "b".repeat(64),
                build: BuildIdentity {
                    executable_sha256: "c".repeat(64),
                    git_revision: "new".to_owned(),
                    git_dirty: false,
                    worktree_state_sha256: None,
                    cargo_lock_sha256: Some("e".repeat(64)),
                    rustc_version: "rustc test".to_owned(),
                    target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                    build_profile: "release-perf".to_owned(),
                    cargo_features: vec!["perf-harness".to_owned()],
                },
                machine: MachineIdentity {
                    fingerprint: "linux-x86_64-test".to_owned(),
                    os: "linux".to_owned(),
                    arch: "x86_64".to_owned(),
                    logical_cpus: 8,
                    cpu_governor: None,
                    load_average_start: None,
                    load_average_end: None,
                },
                peak_rss: PeakRssEvidence {
                    method: "unsupported".to_owned(),
                    bytes: None,
                },
                corpus: CorpusIdentity {
                    corpus_sha256: "a".repeat(64),
                    query_set_sha256: Some("1".repeat(64)),
                    qrels_sha256: None,
                    document_count: 100_000,
                    content_bytes: None,
                    generator_seed: 42,
                    generator_revision: "test-v1".to_owned(),
                },
            },
            vec![cell],
        )
        .expect("QG-6 evidence artifact");
        seal_evidence(&mut evidence);
        (artifact, evidence)
    }

    fn qg6_complete_pair(
        run_id: &str,
        group_ratios: [[f64; 3]; 4],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let (mut artifact, template_evidence) = qg6_current_pair(run_id, group_ratios);
        let template_rows = artifact.cells.clone();
        let template_cell = template_evidence.cells[0].clone();
        let mut rows = Vec::new();
        let mut cells = Vec::new();
        for spec in PerfMatrixSpec::complete().for_gate(PerfGate::Qg6) {
            for template in &template_rows {
                let mut row = template.clone();
                row.fixture.clone_from(&spec.fixture);
                rows.push(row);
            }
            let mut cell = template_cell.clone();
            cell.spec.fixture.clone_from(&spec.fixture);
            cell.cell_id = format!("{}/{}/{}", PerfGate::Qg6, spec.fixture, spec.metric);
            cells.push(cell);
        }
        artifact.cells = rows;
        let mut evidence = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            template_evidence.policy,
            template_evidence.provenance,
            cells,
        )
        .expect("complete QG-6 evidence");
        seal_evidence(&mut evidence);
        (artifact, evidence)
    }

    fn mutate_qg6_prepared_input(
        evidence: &mut PerfEvidenceArtifact,
        field: &str,
        replacement: &str,
    ) {
        let cell = evidence.cells.first().expect("QG-6 evidence cell");
        let EvidenceCellBody::Paired { paired, .. } = &cell.body else {
            unreachable!("QG-6 must be paired");
        };
        let mut identity = cell
            .spec
            .input_identity
            .clone()
            .expect("QG-6 prepared-input identity");
        match field {
            "prepared_corpus_sha256" => {
                identity.prepared_corpus_sha256 = replacement.to_owned();
            }
            "query_manifest_sha256" => {
                identity.query_manifest_sha256 = replacement.to_owned();
            }
            "config_contract_sha256" => {
                identity.config_contract_sha256 = replacement.to_owned();
            }
            _ => unreachable!("enumerated identity field"),
        }
        let mut effect_samples = paired.effect_samples.clone();
        let mut null_samples = paired.null_samples.clone();
        for sample in effect_samples.iter_mut().chain(&mut null_samples) {
            sample.provenance.input_identity = Some(identity.clone());
        }
        let rebuilt_pair =
            estimate_paired_experiment(&effect_samples, &null_samples, &paired.config)
                .expect("coherent mutated QG-6 paired evidence");
        let mut rebuilt_spec = cell.spec.clone();
        rebuilt_spec.input_identity = Some(identity);
        evidence.cells[0] = EvidenceCell::evaluate(rebuilt_spec, rebuilt_pair, &evidence.policy)
            .expect("coherent mutated QG-6 cell");
    }

    fn evaluate(
        baseline: &PerfGateArtifact,
        candidate: &PerfGateArtifact,
        rerun: Option<&PerfGateArtifact>,
        activated: bool,
        mode: PerfRatchetMode,
    ) -> PerfRatchetEvaluation {
        evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(baseline),
            candidate,
            rerun,
            candidate_evidence: None,
            rerun_evidence: None,
            require_current_evidence: false,
            gate_activated: activated,
            mode,
            expected_manifest_sha256: &"b".repeat(64),
            evidence: Vec::new(),
        })
    }

    fn evaluate_with_current(
        baseline: &PerfGateArtifact,
        candidate: &PerfGateArtifact,
        rerun: Option<&PerfGateArtifact>,
        candidate_evidence: Option<&PerfEvidenceArtifact>,
        rerun_evidence: Option<&PerfEvidenceArtifact>,
    ) -> PerfRatchetEvaluation {
        evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(baseline),
            candidate,
            rerun,
            candidate_evidence,
            rerun_evidence,
            require_current_evidence: true,
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &"b".repeat(64),
            evidence: Vec::new(),
        })
    }

    #[test]
    fn decision_severity_is_fatal_then_block_then_quarantine_then_allow() {
        let mut state = DecisionState::default();
        assert_eq!(state.decision(), PerfGateDecision::Allow);

        state.quarantine("test.quarantine", "inconclusive evidence");
        assert_eq!(state.decision(), PerfGateDecision::Quarantine);

        state.block("test.block", "decisive regression");
        assert_eq!(state.decision(), PerfGateDecision::Block);

        state.fatal("test.fatal", "invalid artifact");
        assert_eq!(state.decision(), PerfGateDecision::Block);
    }

    #[test]
    fn clean_activated_same_revision_rerun_allows_promotion() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let rerun = qg2_artifact("new", 160.5, 100.0);
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Allow);
    }

    #[test]
    fn promotion_requires_law_attestation_on_the_rerun() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let mut rerun = qg2_artifact("new", 160.5, 100.0);
        assert!(candidate.laws_attested);
        rerun.laws_attested = false;

        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.rerun_laws_not_attested"
                && reason.message.contains("same-revision rerun")
        }));
    }

    #[test]
    fn current_evidence_reconciles_with_threshold_projection_and_rerun() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 161.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 161.0, 100.0);
        let result = evaluate_with_current(
            &baseline,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
        );
        assert_eq!(result.decision, PerfGateDecision::Allow);
    }

    #[test]
    fn post_seal_current_evidence_mutation_blocks_before_target_evaluation() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let (candidate, mut candidate_evidence) =
            qg2_current_pair("new", "candidate", 161.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 161.0, 100.0);
        let retained_seal = candidate_evidence.artifact_sha256.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut candidate_evidence.cells[0].body else {
            unreachable!("QG-2 evidence must be paired");
        };
        paired.effect.treatment.p50 += 1.0;
        assert_eq!(candidate_evidence.artifact_sha256, retained_seal);

        let result = evaluate_with_current(
            &baseline,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.current_evidence_integrity_failed"
                && reason.message.contains("hash seal")
        }));
        assert!(
            result.comparisons.is_empty(),
            "integrity failure must short-circuit before baseline or target evaluation"
        );
        assert!(
            !result
                .reasons
                .iter()
                .any(|reason| reason.code.starts_with("perf.ratchet.gate_target_")),
            "integrity failure must not reach gate target evaluation"
        );
    }

    #[test]
    fn qg2_promotion_requires_rerun_target_to_pass_independently() {
        let baseline = qg2_artifact("old", 152.0, 100.0);
        let candidate = qg2_artifact("new", 152.0, 100.0);
        let rerun = qg2_artifact("new", 149.0, 100.0);

        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("QG-2")
                && reason.message.contains("does not clear 1.5")
        }));
        assert!(
            !result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.reproduction_failed"),
            "the adversarial rerun must remain inside generic reproduction tolerance: {:?}",
            result.reasons
        );
    }

    #[test]
    fn required_current_evidence_cannot_be_omitted() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();
        let result = evaluate_with_current(&baseline, &candidate, Some(&rerun), None, None);
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.missing_current_candidate_evidence" })
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.missing_current_rerun_evidence")
        );
    }

    #[test]
    fn current_evidence_elf_projection_mismatch_quarantines() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let (mut candidate, candidate_evidence) =
            qg2_current_pair("new", "candidate", 161.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 161.0, 100.0);
        candidate.bench_elf_sha256 = "f".repeat(64);
        let result = evaluate_with_current(
            &baseline,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.current_evidence_identity_mismatch")
        );
    }

    #[test]
    fn qg6_query_config_identity_stays_separate_from_shared_corpus_provenance() {
        let ratios = [[0.95; 3]; 4];
        let (artifact, evidence) = qg6_current_pair("candidate", ratios);
        let mut state = DecisionState::default();
        let legacy_cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        validate_current_evidence_cell(
            &evidence.cells[0],
            &evidence,
            &legacy_cells,
            true,
            "candidate",
            &mut state,
        );
        assert!(!state.fatal);
        assert!(
            !state
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.current_evidence_scope_mismatch")
        );
        let paired = match &evidence.cells[0].body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        assert_eq!(
            paired.provenance.corpus_sha256,
            evidence.provenance.corpus.corpus_sha256
        );
        assert_eq!(
            paired.provenance.input_identity,
            evidence.cells[0].spec.input_identity
        );

        let mut corrupted = evidence.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut corrupted.cells[0].body else {
            unreachable!("QG-6 must be paired");
        };
        paired.provenance.corpus_sha256 = "9".repeat(64);
        let mut corrupted_state = DecisionState::default();
        validate_current_evidence_cell(
            &corrupted.cells[0],
            &corrupted,
            &legacy_cells,
            true,
            "candidate",
            &mut corrupted_state,
        );
        assert!(corrupted_state.fatal);
        assert!(
            corrupted_state
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.current_evidence_scope_mismatch" })
        );
    }

    #[test]
    fn qg6_reproduction_rejects_each_prepared_input_identity_change_independently() {
        let ratios = [[0.95; 3]; 4];
        let (_, candidate) = qg6_current_pair("candidate", ratios);
        for (field, replacement) in [
            ("prepared_corpus_sha256", "3".repeat(64)),
            ("query_manifest_sha256", "4".repeat(64)),
            ("config_contract_sha256", "5".repeat(64)),
        ] {
            let (_, mut rerun) = qg6_current_pair("rerun", ratios);
            mutate_qg6_prepared_input(&mut rerun, field, &replacement);
            let mut state = DecisionState::default();
            compare_current_evidence_reproduction(&candidate, &rerun, &mut state);
            assert!(
                state.reasons.iter().any(|reason| {
                    reason.code == "perf.ratchet.qg6_hierarchical_reproduction_incompatible"
                }),
                "{field} changed independently without invalidating hierarchical reproduction: {:?}",
                state.reasons
            );
        }
    }

    #[test]
    fn qg6_promotion_requires_rerun_hierarchical_ci_to_pass_independently() {
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.0; 3]; 4]);
        let (rerun, rerun_evidence) =
            qg6_complete_pair("rerun", [[0.80; 3], [1.0; 3], [1.0; 3], [1.25; 3]]);
        let mut baseline = candidate.clone();
        baseline.run_id = "baseline".to_owned();
        baseline.git_rev = "old".to_owned();

        let candidate_hierarchy =
            exact_qg6_hierarchical_cell(&candidate_evidence, "query/identifier/k10/100k")
                .expect("candidate hierarchy")
                .0;
        let rerun_hierarchy =
            exact_qg6_hierarchical_cell(&rerun_evidence, "query/identifier/k10/100k")
                .expect("rerun hierarchy")
                .0;
        assert!(
            (candidate_hierarchy.median_of_group_medians_log
                - rerun_hierarchy.median_of_group_medians_log)
                .abs()
                < 1.0e-12,
            "fixture must reproduce at the hierarchical point estimate"
        );
        assert!(
            rerun_hierarchy.ci95_low_ratio < 0.90 || rerun_hierarchy.ci95_high_ratio > 1.10,
            "rerun fixture must violate the hierarchical target only through interval width"
        );

        let result = evaluate_with_current(
            &baseline,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_ci_inconclusive"
                && reason.message.contains("hierarchical")
        }));
        assert!(!result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                && reason.message.contains("A/B effect")
        }));
    }

    #[test]
    fn qg6_promotion_requires_rerun_absolute_p99_to_pass_independently() {
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.0; 3]; 4]);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", [[1.01; 3]; 4]);
        let mut baseline = candidate.clone();
        baseline.run_id = "baseline".to_owned();
        baseline.git_rev = "old".to_owned();

        let result = evaluate_with_current(
            &baseline,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
        );
        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("Quill p99")
                && reason.message.contains("exceeds oracle")
        }));
        assert!(!result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
        }));
    }

    #[test]
    fn qg6_gate_uses_hierarchical_ci_when_flat_projection_would_false_pass() {
        let ratios = [[0.91; 3], [0.91; 3], [0.91; 3], [1.20; 3]];
        let (artifact, evidence) = qg6_current_pair("candidate", ratios);
        let flat = artifact
            .cells
            .iter()
            .find(|cell| cell.engine == "paired_ab")
            .expect("flat compatibility projection");
        assert!(
            flat.distribution.median_ci95_low >= 0.90 && flat.distribution.median_ci95_high <= 1.10,
            "fixture must demonstrate the former flat-CI false pass: {:?}",
            flat.distribution
        );
        let EvidenceCellBody::Paired {
            hierarchical: Some(hierarchical),
            ..
        } = &evidence.cells[0].body
        else {
            unreachable!("QG-6 must carry a hierarchical estimate");
        };
        assert!(
            hierarchical.ci95_high_ratio > 1.10,
            "hierarchical CI must expose between-query uncertainty"
        );

        let mut state = DecisionState::default();
        let cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            state: &mut state,
        };
        evaluate_qg6(&mut target, Some(&evidence));
        assert!(state.quarantined);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_ci_inconclusive"
                && reason.message.contains("hierarchical")
        }));
    }

    #[test]
    fn qg6_hierarchical_equivalence_is_not_vetoed_by_flat_effect_claim_logic() {
        let ratios = [[1.0; 3]; 4];
        let (artifact, evidence) = qg6_current_pair("candidate", ratios);
        let flat_effect = artifact
            .cells
            .iter()
            .find(|cell| cell.engine == "paired_ab")
            .expect("flat A/B compatibility projection");
        let flat_null = artifact
            .cells
            .iter()
            .find(|cell| cell.engine == "paired_null")
            .expect("flat A/A compatibility projection");
        assert!(
            flat_null.distribution.median_ci95_low <= flat_effect.distribution.p50
                && flat_effect.distribution.p50 <= flat_null.distribution.median_ci95_high,
            "the generic flat effect-claim rule must be inconclusive for this equivalence fixture"
        );

        let mut state = DecisionState::default();
        let cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        validate_paired_evidence(PerfGate::Qg6, &cells, "candidate", &mut state);
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            state: &mut state,
        };
        evaluate_qg6(&mut target, Some(&evidence));
        assert!(
            !state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.inconclusive_paired_claim"
                    || reason.code == "perf.ratchet.gate_target_missed"
                    || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
            }),
            "flat claim logic vetoed a decision-valid hierarchical equivalence result: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg6_null_admission_uses_hierarchy_when_flat_projection_would_false_pass() {
        let effect_ratios = [[0.95; 3]; 4];
        let null_ratios = [
            [1.0, 1.0, 1.0],
            [1.08, 1.08, 1.0],
            [1.08, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let (artifact, evidence) =
            qg6_current_pair_with_null("candidate", effect_ratios, null_ratios);
        let flat = artifact
            .cells
            .iter()
            .find(|cell| cell.engine == "paired_null")
            .expect("flat A/A compatibility projection");
        assert!(
            flat.distribution.median_ci95_low >= 0.95
                && flat.distribution.median_ci95_high <= 1.05
                && flat.distribution.median_ci95_low <= 1.0
                && 1.0 <= flat.distribution.median_ci95_high,
            "fixture must demonstrate the former flat A/A false pass: {:?}",
            flat.distribution
        );
        let EvidenceCellBody::Paired {
            hierarchical_null: Some(hierarchical_null),
            ..
        } = &evidence.cells[0].body
        else {
            unreachable!("QG-6 must carry a hierarchical A/A estimate");
        };
        assert!(
            (hierarchical_null.ci95_low_ratio - 1.0)
                .abs()
                .max((hierarchical_null.ci95_high_ratio - 1.0).abs())
                > 0.05,
            "hierarchical A/A CI must expose between-query uncertainty"
        );

        let mut state = DecisionState::default();
        let cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            state: &mut state,
        };
        evaluate_qg6(&mut target, Some(&evidence));
        assert!(state.quarantined);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.inconclusive_equivalence"
                && reason.message.contains("hierarchical A/A")
        }));
    }

    #[test]
    fn qg6_gate_never_borrows_hierarchy_from_an_unrelated_same_fixture_cell() {
        let ratios = [[0.95; 3]; 4];
        let (artifact, mut evidence) = qg6_current_pair("candidate", ratios);
        let mut unrelated = evidence.cells[0].clone();
        unrelated.spec.unit = "ns".to_owned();
        unrelated.spec.role = EvidenceRole::Diagnostic;
        let EvidenceCellBody::Paired {
            hierarchical: Some(estimate),
            ..
        } = &mut unrelated.body
        else {
            unreachable!("QG-6 must carry a hierarchical estimate");
        };
        estimate.ci95_low_ratio = 2.0;
        estimate.ci95_high_ratio = 2.0;
        evidence.cells.insert(0, unrelated);

        let mut state = DecisionState::default();
        let cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            state: &mut state,
        };
        evaluate_qg6(&mut target, Some(&evidence));
        assert!(
            !state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.gate_target_missed"
                    || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
            }),
            "an unrelated same-fixture cell influenced the QG-6 target: {:?}",
            state.reasons
        );

        evidence.cells.pop();
        let mut missing_state = DecisionState::default();
        let missing_cells = validate_artifact(
            &artifact,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut missing_state,
        );
        let mut missing_target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &missing_cells,
            activated: true,
            state: &mut missing_state,
        };
        evaluate_qg6(&mut missing_target, Some(&evidence));
        assert!(
            missing_state
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.qg6_hierarchical_evidence_missing")
        );
    }

    #[test]
    fn qg6_reproduction_uses_hierarchical_grouping_not_equal_flat_multisets() {
        let candidate_groups = [
            [0.95, 0.95, 0.95],
            [0.95, 0.95, 0.95],
            [0.95, 1.10, 1.10],
            [1.10, 1.10, 1.10],
        ];
        let rerun_groups = [
            [0.95, 0.95, 1.10],
            [0.95, 0.95, 1.10],
            [0.95, 0.95, 1.10],
            [0.95, 1.10, 1.10],
        ];
        let (_, candidate) = qg6_current_pair("candidate", candidate_groups);
        let (_, rerun) = qg6_current_pair("rerun", rerun_groups);
        let candidate_pair = match &candidate.cells[0].body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        let rerun_pair = match &rerun.cells[0].body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        assert!(
            candidate_pair
                .reproduces_within(rerun_pair)
                .expect("flat reproduction compatibility"),
            "fixture must reproduce under the former flat estimator"
        );

        let mut state = DecisionState::default();
        compare_current_evidence_reproduction(&candidate, &rerun, &mut state);
        assert!(state.quarantined);
        assert!(
            state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
            })
        );
    }

    #[test]
    fn qg6_null_reproduction_uses_hierarchical_grouping_not_equal_flat_multisets() {
        let candidate_null_groups = [
            [0.95, 0.95, 0.95],
            [0.95, 0.95, 0.95],
            [0.95, 1.10, 1.10],
            [1.10, 1.10, 1.10],
        ];
        let rerun_null_groups = [
            [0.95, 0.95, 1.10],
            [0.95, 0.95, 1.10],
            [0.95, 0.95, 1.10],
            [0.95, 1.10, 1.10],
        ];
        let (_, candidate) =
            qg6_current_pair_with_null("candidate", [[0.95; 3]; 4], candidate_null_groups);
        let (_, rerun) = qg6_current_pair_with_null("rerun", [[0.95; 3]; 4], rerun_null_groups);
        let candidate_pair = match &candidate.cells[0].body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        let rerun_pair = match &rerun.cells[0].body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        assert_eq!(
            projected_ratio_distribution(&candidate_pair.null_samples),
            projected_ratio_distribution(&rerun_pair.null_samples),
            "fixture must have identical legacy flat A/A projections"
        );

        let mut state = DecisionState::default();
        compare_current_evidence_reproduction(&candidate, &rerun, &mut state);
        assert!(state.quarantined);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                && reason.message.contains("A/A null")
        }));
    }

    #[test]
    fn qg6_legacy_flat_null_projection_never_vetoes_reproduction() {
        let ratios = [[0.95; 3]; 4];
        let (candidate, _) = qg6_current_pair("candidate", ratios);
        let (mut rerun, _) = qg6_current_pair("rerun", ratios);
        let flat_null = rerun
            .cells
            .iter_mut()
            .find(|cell| cell.engine == "paired_null")
            .expect("flat A/A compatibility projection");
        flat_null.distribution.p50 = 1.50;
        flat_null.distribution.median_ci95_low = 1.49;
        flat_null.distribution.median_ci95_high = 1.51;

        let mut state = DecisionState::default();
        let candidate_cells = validate_artifact(
            &candidate,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        let rerun_cells =
            validate_artifact(&rerun, PerfGate::Qg6, &"b".repeat(64), "rerun", &mut state);
        compare_reproduction(
            &candidate,
            &rerun,
            &candidate_cells,
            &rerun_cells,
            &mut state,
        );
        assert!(
            !state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.reproduction_failed"
                    && reason.message.contains("paired_null")
            }),
            "flat A/A projection leaked into QG-6 reproduction: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg6_legacy_flat_paired_projections_never_veto_baseline_comparison() {
        let ratios = [[0.95; 3]; 4];
        let (baseline, _) = qg6_current_pair("baseline", ratios);
        let (mut candidate, _) = qg6_current_pair("candidate", ratios);
        for cell in &mut candidate.cells {
            match cell.engine.as_str() {
                "paired_ab" => {
                    cell.distribution.p50 = 2.0;
                    cell.distribution.median_ci95_low = 1.9;
                    cell.distribution.median_ci95_high = 2.1;
                }
                "paired_null" => {
                    cell.distribution.p50 = 1.5;
                    cell.distribution.median_ci95_low = 1.4;
                    cell.distribution.median_ci95_high = 1.6;
                }
                _ => {}
            }
        }

        let mut state = DecisionState::default();
        let baseline_cells = validate_artifact(
            &baseline,
            PerfGate::Qg6,
            &"b".repeat(64),
            "baseline",
            &mut state,
        );
        let candidate_cells = validate_artifact(
            &candidate,
            PerfGate::Qg6,
            &"b".repeat(64),
            "candidate",
            &mut state,
        );
        let mut comparisons = Vec::new();
        compare_baseline(
            &baseline,
            &candidate,
            &baseline_cells,
            &candidate_cells,
            PerfRatchetMode::Promotion,
            &mut comparisons,
            &mut state,
        );
        assert!(
            !state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.invalid_null_control"
                    || (matches!(
                        reason.code.as_str(),
                        "perf.ratchet.regression_detected" | "perf.ratchet.inconclusive_regression"
                    ) && reason.message.contains("paired_ab"))
            }),
            "flat QG-6 paired projections leaked into baseline decisions: {:?}",
            state.reasons
        );
        assert!(
            comparisons
                .iter()
                .all(|comparison| comparison.engine != "paired_ab"),
            "flat QG-6 effect projection leaked into comparison output"
        );
    }

    #[test]
    fn reproducible_pass_over_pass_regression_blocks() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 140.0, 100.0);
        let mut rerun = qg2_artifact("new", 140.0, 100.0);
        rerun.run_id = "rerun".to_owned();
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.regression_detected")
        );
    }

    #[test]
    fn high_cv_is_provenance_and_does_not_quarantine_a_decidable_claim() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let mut candidate = qg2_artifact("new", 161.0, 100.0);
        candidate.cells[0].distribution.cv_pct = 47.0;
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Allow);
        assert!(result.reasons.iter().all(|reason| {
            reason.code != "perf.ratchet.insufficient_samples"
                && reason.code != "perf.ratchet.inconclusive_paired_claim"
        }));
    }

    #[test]
    fn target_point_estimate_is_inconclusive_when_median_ci_crosses_threshold() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let mut candidate = qg2_artifact("new", 160.0, 100.0);
        candidate.cells[2].distribution.median_ci95_low = 1.49;
        candidate.cells[2].distribution.median_ci95_high = 1.71;
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();

        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_ci_inconclusive"
                && reason.message.contains("[1.490000, 1.710000]")
        }));
    }

    #[test]
    fn target_miss_blocks_only_when_entire_median_ci_fails_threshold() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let mut candidate = qg2_artifact("new", 140.0, 100.0);
        candidate.cells[2].distribution.median_ci95_low = 1.31;
        candidate.cells[2].distribution.median_ci95_high = 1.49;
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();

        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("[1.310000, 1.490000]")
        }));
    }

    #[test]
    fn same_revision_reproduction_mismatch_quarantines() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 160.0, 100.0);
        // Keep the rerun independently above QG-2's 1.5x target while
        // exceeding the predeclared 5% reproduction tolerance.
        let rerun = qg2_artifact("new", 151.0, 100.0);
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.reproduction_failed")
        );
    }

    #[test]
    fn rerun_from_a_different_elf_is_quarantined() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();
        rerun.bench_elf_sha256 = "d".repeat(64);
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.rerun_elf_mismatch")
        );
    }

    #[test]
    fn inactive_gate_remains_provisional_even_when_target_passes() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let mut rerun = qg2_artifact("new", 161.0, 100.0);
        rerun.run_id = "rerun".to_owned();
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            false,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.gate_inactive")
        );
    }

    #[test]
    fn regression_alarm_allows_cross_revision_non_regression() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let result = evaluate(
            &baseline,
            &candidate,
            None,
            false,
            PerfRatchetMode::RegressionAlarm,
        );
        assert_eq!(result.decision, PerfGateDecision::Allow);
    }

    #[test]
    fn ci_overlapping_apparent_regression_is_quarantined_not_blocked() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let mut candidate = qg2_artifact("new", 150.0, 100.0);
        candidate.cells[0].distribution.median_ci95_low = 140.0;
        candidate.cells[0].distribution.median_ci95_high = 170.0;
        candidate.cells[2].distribution.median_ci95_low = 1.4;
        candidate.cells[2].distribution.median_ci95_high = 1.7;
        let mut rerun = candidate.clone();
        rerun.run_id = "rerun".to_owned();
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
    }

    #[test]
    fn activated_bootstrap_can_establish_first_measured_baseline() {
        let mut baseline = qg2_artifact("unmeasured", 0.0, 1.0);
        baseline.machine_fingerprint = "unmeasured".to_owned();
        baseline.bench_elf_sha256 = "unmeasured".to_owned();
        baseline.run_window = "unmeasured".to_owned();
        baseline.run_id = "unmeasured".to_owned();
        baseline.cells.clear();
        baseline.laws_attested = false;
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let mut rerun = qg2_artifact("new", 161.0, 100.0);
        rerun.run_id = "rerun".to_owned();
        let result = evaluate(
            &baseline,
            &candidate,
            Some(&rerun),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(result.decision, PerfGateDecision::Allow);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.bootstrap_promotion")
        );
    }

    #[test]
    fn bootstrap_cannot_satisfy_pr_regression_alarm() {
        let mut baseline = qg2_artifact("unmeasured", 0.0, 1.0);
        baseline.machine_fingerprint = "unmeasured".to_owned();
        baseline.bench_elf_sha256 = "unmeasured".to_owned();
        baseline.run_window = "unmeasured".to_owned();
        baseline.run_id = "unmeasured".to_owned();
        baseline.cells.clear();
        baseline.laws_attested = false;
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let result = evaluate(
            &baseline,
            &candidate,
            None,
            false,
            PerfRatchetMode::RegressionAlarm,
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
    }

    #[test]
    fn reused_or_cross_window_rerun_is_quarantined() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let reused = candidate.clone();
        let reused_result = evaluate(
            &baseline,
            &candidate,
            Some(&reused),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(reused_result.decision, PerfGateDecision::Quarantine);
        assert!(
            reused_result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.rerun_identity_reused" })
        );

        let mut cross_window = candidate.clone();
        cross_window.run_id = "rerun".to_owned();
        cross_window.run_window = "other-window".to_owned();
        let cross_window_result = evaluate(
            &baseline,
            &candidate,
            Some(&cross_window),
            true,
            PerfRatchetMode::Promotion,
        );
        assert_eq!(cross_window_result.decision, PerfGateDecision::Quarantine);
        assert!(
            cross_window_result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.rerun_window_mismatch" })
        );
    }
}
