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
    PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS, PerfCellResult, PerfGate, PerfGateArtifact,
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

    if request.mode == PerfRatchetMode::Promotion && candidate_is_complete(gate, &candidate_cells) {
        evaluate_gate_targets(
            request.candidate,
            &candidate_cells,
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

    PerfRatchetEvaluation {
        schema_version: PERF_RATCHET_SCHEMA_VERSION.to_owned(),
        gate,
        mode: request.mode,
        gate_activated: request.gate_activated,
        decision: state.decision(),
        reasons: state.reasons,
        comparisons,
        evidence: request.evidence,
        history_updates: Vec::new(),
    }
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
        if !validate_null_control(null, role, state) {
            continue;
        }

        let null_floor = (null.distribution.median_ci95_low - 1.0)
            .abs()
            .max((null.distribution.median_ci95_high - 1.0).abs());
        if gate == PerfGate::Qg6 {
            let equivalence_half_width = 0.10;
            let floor_fits = PERF_NULL_MARGIN_MULTIPLIER * null_floor <= equivalence_half_width;
            let claim_fits = claim.distribution.median_ci95_low >= 1.0 - equivalence_half_width
                && claim.distribution.median_ci95_high <= 1.0 + equivalence_half_width;
            if !floor_fits || !claim_fits {
                state.quarantine(
                    "perf.ratchet.inconclusive_equivalence",
                    format!(
                        "{role} {}/{} cannot establish ±10% equivalence: A/B median CI \
                         [{:.6}, {:.6}], A/A median CI [{:.6}, {:.6}], required null margin {:.1}x",
                        key.fixture,
                        key.metric,
                        claim.distribution.median_ci95_low,
                        claim.distribution.median_ci95_high,
                        null.distribution.median_ci95_low,
                        null.distribution.median_ci95_high,
                        PERF_NULL_MARGIN_MULTIPLIER,
                    ),
                );
            }
            continue;
        }

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
    fn value(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
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
        self.cells.get(&key).map(|cell| cell.distribution.p50)
    }

    fn p95(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
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
        self.cells.get(&key).map(|cell| cell.distribution.p95)
    }

    fn p99(&mut self, fixture: &str, metric: &str, engine: &str) -> Option<f64> {
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
        self.cells.get(&key).map(|cell| cell.distribution.p99)
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
}

fn evaluate_gate_targets(
    artifact: &PerfGateArtifact,
    cells: &BTreeMap<CellKey, &PerfCellResult>,
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
        PerfGate::Qg6 => evaluate_qg6(&mut target),
        PerfGate::Qg7 => evaluate_qg7(&mut target),
        PerfGate::Qg8 => evaluate_qg8(&mut target),
        PerfGate::Qg9 => evaluate_qg9(&mut target),
        PerfGate::Qg10 => evaluate_qg10(&mut target),
    }
}

fn evaluate_qg1(target: &mut GateTargetEvaluator<'_, '_>) {
    for corpus in ["medium", "xlarge"] {
        let fixture = format!("bulk/{corpus}/8/positions_on");
        if let Some(ratio) =
            target.value(&fixture, "docs_per_second_quill_over_tantivy", "paired_ab")
        {
            target.target(
                ratio >= 3.0,
                format!("QG-1 {fixture} ratio {ratio:.6} is below 3.0"),
            );
        }
        let tokenize_fixture = format!("tokenize_only/{corpus}");
        if let (Some(index), Some(tokenize)) = (
            target.value(&fixture, "docs_per_second", "quill"),
            target.value(
                &tokenize_fixture,
                "tokenize_docs_per_second",
                "quill_tokenizer",
            ),
        ) {
            let ceiling_ratio = index / tokenize.max(f64::MIN_POSITIVE);
            target.target(
                ceiling_ratio >= 0.60,
                format!(
                    "QG-1 {corpus} indexing/tokenize ceiling ratio {ceiling_ratio:.6} is below 0.60"
                ),
            );
        }
    }
}

fn evaluate_qg2(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some(ratio) = target.value(
        "bulk/medium/1/positions_on",
        "docs_per_second_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target(
            ratio >= 1.5,
            format!("QG-2 single-thread ratio {ratio:.6} is below 1.5"),
        );
    }
}

fn evaluate_qg3(target: &mut GateTargetEvaluator<'_, '_>) {
    if let Some(initial) = target.value("watch/medium/initial", "docs_per_second", "quill") {
        target.target(
            initial >= 20_000.0,
            format!("QG-3 initial throughput {initial:.3} docs/s is below 20000"),
        );
    }
    for topology in ["inprocess", "freshprocess"] {
        let fixture = format!("watch/medium/5000/{topology}");
        if let Some(updates) = target.value(&fixture, "updates_per_second", "quill") {
            target.target(
                updates >= 5_000.0,
                format!("QG-3 {topology} throughput {updates:.3} updates/s is below 5000"),
            );
        }
        if let Some(p95) = target.p95(&fixture, "update_to_searchable_ms", "quill") {
            target.target(
                p95 <= 25.0,
                format!("QG-3 {topology} update-to-searchable p95 {p95:.3}ms exceeds 25ms"),
            );
        }
    }
    if let Some(ratio) = target.value(
        "watch/medium/5000/inprocess",
        "update_to_searchable_ms_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target(
            ratio <= 0.25,
            format!("QG-3 in-process visibility ratio {ratio:.6} exceeds 0.25"),
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
    if let Some(ratio) = target.value(
        "compaction/xlarge/20pct",
        "wall_clock_ms_quill_over_tantivy",
        "paired_ab",
    ) {
        target.target(
            ratio <= 0.20,
            format!("QG-5 20% compaction ratio {ratio:.6} exceeds 0.20"),
        );
    }
}

fn evaluate_qg6(target: &mut GateTargetEvaluator<'_, '_>) {
    let fixtures = target
        .cells
        .keys()
        .filter(|key| key.engine == "paired_ab" && key.metric == "latency_ms_quill_over_tantivy")
        .map(|key| key.fixture.clone())
        .collect::<Vec<_>>();
    for fixture in fixtures {
        if let Some(ratio) = target.value(&fixture, "latency_ms_quill_over_tantivy", "paired_ab") {
            target.target(
                (0.90..=1.10).contains(&ratio),
                format!("QG-6 {fixture} p50 ratio {ratio:.6} is outside [0.90, 1.10]"),
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
    use crate::{DistributionSummary, PerfCellResult};

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
            gate_activated: activated,
            mode,
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
    fn same_revision_reproduction_mismatch_quarantines() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 160.0, 100.0);
        let rerun = qg2_artifact("new", 145.0, 100.0);
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
