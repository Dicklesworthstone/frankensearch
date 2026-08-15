//! Pass-over-pass evaluation for committed Quill performance artifacts.
//!
//! The benchmark harness emits measurements; this module decides whether a
//! result may advance the committed `.bench-history` baseline. It deliberately
//! keeps noisy results in quarantine and requires a same-revision rerun before
//! a performance result can be promoted. An explicit unmeasured bootstrap may
//! establish the first admitted baseline even when the target is missed; that
//! MISS is retained in the decision reasons and remains claim-ineligible.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::perf::PERF_NULL_MARGIN_MULTIPLIER;
use crate::perf_evidence::QG6_NULL_EFFECT_MARGIN;
use crate::{
    DistributionSummary, EvidenceCellBody, EvidenceRole, ExecutionProfileId, HardwareClassId,
    MachineClassRegistry, MachineProfileKey, PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS,
    PerfApplicabilityPlan, PerfApplicabilityPlanBinding, PerfCellApplicability, PerfCellResult,
    PerfEvidenceArtifact, PerfExecutionProvenance, PerfGate, PerfGateArtifact, PerfMatrixSpec,
    Qg1ExpectedAuthority, Qg6ScheduleAuthority, VerifiedRunnerIdentity,
};

/// Version of the machine-readable ratchet decision artifact.
pub const PERF_RATCHET_SCHEMA_VERSION: &str = "quill-perf-ratchet-v4";
/// Strict schema for the immutable pointer to an admitted history artifact pair.
pub const PERF_HISTORY_POINTER_SCHEMA_VERSION: &str = "frankensearch.perf-history-pointer.v3";
/// Maximum directional pass-over-pass regression admitted for a cell.
pub const PERF_MAX_REGRESSION_PCT: f64 = 5.0;
/// Maximum disagreement admitted between same-revision candidate reruns.
pub const PERF_MAX_REPRODUCTION_DELTA_PCT: f64 = 5.0;
/// Robust-z value retained as diagnostic provenance beside CI-gated decisions.
pub const PERF_REGRESSION_ROBUST_Z: f64 = 3.0;
/// Largest drift from identity admitted for an A/A null control's median.
///
/// Bounds the null's *accuracy*. Its precision is bounded separately, and in
/// the opposite direction, by the paired A/A floor a claim must clear. See
/// [`validate_null_control`].
pub const PERF_MAX_NULL_MEDIAN_DRIFT_PCT: f64 = 2.0;

const MAD_SCALE: f64 = 1.4826;
const MAD_EPSILON: f64 = 1.0e-12;
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

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
    /// Content-addressed history objects planned before the decision record is
    /// written. Publication occurs only after that record is durable.
    pub history_updates: Vec<PerfEvidenceFile>,
}

/// Inputs to one ratchet evaluation.
pub struct PerfRatchetRequest<'a> {
    /// Prior committed history artifact, if one exists.
    pub baseline: Option<&'a PerfGateArtifact>,
    /// Hash-sealed current-schema evidence bound to the baseline receipt.
    pub baseline_evidence: Option<&'a PerfEvidenceArtifact>,
    /// First candidate measurement.
    pub candidate: &'a PerfGateArtifact,
    /// Same-revision candidate rerun. Required in promotion mode.
    pub rerun: Option<&'a PerfGateArtifact>,
    /// Hash-sealed, raw-recomputable evidence for the candidate artifact.
    pub candidate_evidence: Option<&'a PerfEvidenceArtifact>,
    /// Hash-sealed, raw-recomputable evidence for the candidate rerun.
    pub rerun_evidence: Option<&'a PerfEvidenceArtifact>,
    /// Canonical hardware/profile expected by the caller. This can only check,
    /// never set, the identity derived from verified receipts.
    pub expected_machine_profile: Option<MachineProfileKey>,
    /// Independently admitted exact candidate runner receipt.
    pub candidate_runner_identity: Option<&'a VerifiedRunnerIdentity>,
    /// Independently admitted exact rerun runner receipt.
    pub rerun_runner_identity: Option<&'a VerifiedRunnerIdentity>,
    /// Whether the normative gate manifest marks the gate active.
    pub gate_activated: bool,
    /// Evaluation purpose.
    pub mode: PerfRatchetMode,
    /// SHA-256 of the normative TOML manifest.
    pub expected_manifest_sha256: &'a str,
    /// Content-addressed evidence paths.
    pub evidence: Vec<PerfEvidenceFile>,
}

/// Externally retained QG-1 lifecycle authorities for each ratchet role.
///
/// The slices are deliberately independent: QG-1 receipts are issued for one
/// raw stream in one invocation and must never be admitted under another
/// ratchet role's authority set.
#[derive(Debug, Clone, Copy)]
pub struct PerfRatchetQg1AuthoritySets<'a> {
    /// Authorities retained for the committed baseline invocation.
    pub baseline: &'a [&'a Qg1ExpectedAuthority],
    /// Authorities retained for the first candidate invocation.
    pub candidate: &'a [&'a Qg1ExpectedAuthority],
    /// Authorities retained for the same-revision rerun invocation.
    pub rerun: &'a [&'a Qg1ExpectedAuthority],
}

impl<'a> PerfRatchetQg1AuthoritySets<'a> {
    /// No external authority is available. QG-1 evidence consequently fails
    /// closed while non-QG-1 evidence retains its normal integrity checks.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            baseline: &[],
            candidate: &[],
            rerun: &[],
        }
    }

    fn for_role(self, role: PerfRatchetEvidenceRole) -> &'a [&'a Qg1ExpectedAuthority] {
        match role {
            PerfRatchetEvidenceRole::Baseline => self.baseline,
            PerfRatchetEvidenceRole::Candidate => self.candidate,
            PerfRatchetEvidenceRole::Rerun => self.rerun,
        }
    }
}

/// Externally retained QG-6 pre-timing schedule authorities for each ratchet
/// role.
///
/// Baseline, candidate, and rerun come from independent invocations. Their
/// authority sets therefore remain disjoint inputs instead of one pooled set
/// that could authenticate a swapped artifact.
#[derive(Debug, Clone, Copy)]
pub struct PerfRatchetQg6AuthoritySets<'a> {
    /// Authorities retained for the committed baseline invocation.
    pub baseline: &'a [&'a Qg6ScheduleAuthority],
    /// Authorities retained for the first candidate invocation.
    pub candidate: &'a [&'a Qg6ScheduleAuthority],
    /// Authorities retained for the same-revision rerun invocation.
    pub rerun: &'a [&'a Qg6ScheduleAuthority],
}

impl<'a> PerfRatchetQg6AuthoritySets<'a> {
    /// No external authority is available. QG-6 evidence consequently fails
    /// closed while evidence for other gates remains unaffected.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            baseline: &[],
            candidate: &[],
            rerun: &[],
        }
    }

    fn for_role(self, role: PerfRatchetEvidenceRole) -> &'a [&'a Qg6ScheduleAuthority] {
        match role {
            PerfRatchetEvidenceRole::Baseline => self.baseline,
            PerfRatchetEvidenceRole::Candidate => self.candidate,
            PerfRatchetEvidenceRole::Rerun => self.rerun,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PerfRatchetEvidenceRole {
    Baseline,
    Candidate,
    Rerun,
}

impl PerfRatchetEvidenceRole {
    const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
            Self::Rerun => "rerun",
        }
    }
}

impl fmt::Display for PerfRatchetEvidenceRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
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

fn reconstruct_applicability_plan(
    binding: &PerfApplicabilityPlanBinding,
    gate: PerfGate,
    role: &str,
    state: &mut DecisionState,
) -> Option<PerfApplicabilityPlan> {
    if binding.gate != gate {
        state.fatal(
            "perf.ratchet.applicability_plan_gate_mismatch",
            format!(
                "{role} applicability plan is for {}, expected {gate}",
                binding.gate
            ),
        );
        return None;
    }
    let registry = match MachineClassRegistry::frozen() {
        Ok(registry) => registry,
        Err(error) => {
            state.fatal(
                "perf.ratchet.machine_registry_invalid",
                format!("frozen machine registry rejected {role} planning: {error}"),
            );
            return None;
        }
    };
    let matrix = PerfMatrixSpec::complete();
    let plan = match matrix.applicability_plan(&registry, binding.profile, gate) {
        Ok(plan) => plan,
        Err(error) => {
            state.fatal(
                "perf.ratchet.applicability_plan_invalid",
                format!(
                    "{role} applicability plan for {:?} {gate} cannot be reconstructed: {error}",
                    binding.profile
                ),
            );
            return None;
        }
    };
    if plan.binding != *binding {
        state.fatal(
            "perf.ratchet.applicability_plan_binding_mismatch",
            format!(
                "{role} applicability-plan binding does not equal the frozen registry and \
                 complete canonical {gate} matrix projection for {:?}",
                binding.profile
            ),
        );
        return None;
    }
    Some(plan)
}

fn candidate_applicability_plan(
    request: &PerfRatchetRequest<'_>,
    state: &mut DecisionState,
) -> Option<PerfApplicabilityPlan> {
    let Some(binding) = request.candidate.applicability_plan.as_ref() else {
        state.fatal(
            "perf.ratchet.measured_applicability_plan_missing",
            "measured candidate threshold artifact requires an exact applicability-plan \
             binding",
        );
        return None;
    };
    reconstruct_applicability_plan(binding, request.candidate.gate, "candidate", state)
}

fn validate_threshold_plan_scope(
    artifact: &PerfGateArtifact,
    plan: &PerfApplicabilityPlan,
    role: &str,
    state: &mut DecisionState,
) {
    if is_explicit_bootstrap(artifact) {
        if role != "baseline" {
            state.fatal(
                "perf.ratchet.bootstrap_candidate_forbidden",
                format!("{role} cannot be the unmeasured bootstrap sentinel"),
            );
        }
        return;
    }
    if let Err(error) = artifact.verify_current_measured_contract() {
        state.fatal(
            "perf.ratchet.threshold_verified_reload_failed",
            format!("{role} measured threshold failed strict reconstruction: {error}"),
        );
        return;
    }
    match artifact.applicability_plan.as_ref() {
        Some(binding) if binding == plan.binding() => {}
        Some(binding) => state.fatal(
            "perf.ratchet.threshold_applicability_plan_mismatch",
            format!(
                "{role} threshold binds {:?} {} plan {}, expected {:?} {} plan {}",
                binding.profile,
                binding.gate,
                binding.applicability_plan_sha256,
                plan.binding.profile,
                plan.binding.gate,
                plan.binding.applicability_plan_sha256
            ),
        ),
        None => state.fatal(
            "perf.ratchet.measured_applicability_plan_missing",
            format!(
                "{role} measured v8 threshold artifact requires an exact applicability-plan \
                 binding"
            ),
        ),
    }
}

/// Evaluate a candidate against the committed pass-over-pass baseline.
#[must_use]
pub fn evaluate_perf_ratchet(request: PerfRatchetRequest<'_>) -> PerfRatchetEvaluation {
    evaluate_perf_ratchet_against_authorities(
        request,
        PerfRatchetQg1AuthoritySets::empty(),
        PerfRatchetQg6AuthoritySets::empty(),
    )
}

/// Evaluate a candidate against the committed baseline with externally
/// retained QG-1 lifecycle authorities for each independent invocation.
#[must_use]
pub fn evaluate_perf_ratchet_against_qg1_authorities(
    request: PerfRatchetRequest<'_>,
    qg1_authority_sets: PerfRatchetQg1AuthoritySets<'_>,
) -> PerfRatchetEvaluation {
    evaluate_perf_ratchet_against_authorities(
        request,
        qg1_authority_sets,
        PerfRatchetQg6AuthoritySets::empty(),
    )
}

/// Evaluate a candidate with every externally retained QG-1 lifecycle and
/// QG-6 pre-timing schedule authority, partitioned by invocation role.
#[must_use]
pub fn evaluate_perf_ratchet_against_authorities(
    request: PerfRatchetRequest<'_>,
    qg1_authority_sets: PerfRatchetQg1AuthoritySets<'_>,
    qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
) -> PerfRatchetEvaluation {
    let gate = request.candidate.gate;
    let mut state = DecisionState::default();
    let Some(plan) = candidate_applicability_plan(&request, &mut state) else {
        return finish_evaluation(
            gate,
            request.mode,
            request.gate_activated,
            state,
            Vec::new(),
            request.evidence,
        );
    };
    if request.mode == PerfRatchetMode::Promotion {
        validate_machine_profile_promotion(
            &request,
            &plan,
            qg1_authority_sets,
            qg6_authority_sets,
            &mut state,
        );
        if state.fatal || state.blocked || state.quarantined {
            return finish_evaluation(
                gate,
                request.mode,
                request.gate_activated,
                state,
                Vec::new(),
                request.evidence,
            );
        }
    } else {
        state.note(
            "perf.ratchet.legacy_regression_alarm_nonpromotable",
            "regression-alarm mode may read current-schema threshold projections without \
             promotion evidence but can never update profile-qualified history",
        );
    }
    let require_current_evidence = request.mode == PerfRatchetMode::Promotion;
    evaluate_perf_ratchet_inner(
        request,
        qg1_authority_sets,
        qg6_authority_sets,
        state,
        require_current_evidence,
    )
}

/// The append-only quarantine register, embedded at build time.
///
/// bd-h4sqj: embedding rather than reading a path is deliberate. The register
/// travels with the binary, so a structurally invalid sweep cannot be admitted
/// by pointing the tool at some other `.bench-history` directory — a refusal
/// that can be sidestepped by changing the working directory is not a refusal.
/// The cost is that appending a record needs a rebuild to take effect, which for
/// an append-only, git-versioned evidence register is the correct propagation
/// path anyway. This mirrors how the QG sentinels are already embedded here.
const EMBEDDED_QUARANTINE_REGISTER: &str = include_str!("../../../.bench-history/QUARANTINE.jsonl");

/// Refuse promotion of evidence measured at a structurally invalid revision.
///
/// INTEGRITY IS NOT ADMISSIBILITY. An artifact from a quarantined sweep passes
/// every check above this one: its bytes verify, its seal matches, and its
/// summaries recompute from their own raw samples. That is precisely why the
/// quarantine has to be a separate screen — nothing else in this function can
/// see the defect, because the defect is in how the numbers were produced, not
/// in the artifact that records them.
///
/// The refusal is `fatal` rather than `quarantine` on purpose.
/// [`PerfGateDecision::Quarantine`] means evidence is "noisy, incomplete,
/// incompatible, or still provisional" — a state a rerun can leave. A
/// structurally invalid sweep never leaves it: re-measuring reproduces the same
/// invalid shape, so inviting a retry would be misleading. `fatal` yields
/// [`PerfGateDecision::Block`], and matches this file's convention that `fatal`
/// carries contract violations while `block` carries measurement outcomes.
fn reject_quarantined_revision(
    role: &str,
    evidence: &PerfEvidenceArtifact,
    state: &mut DecisionState,
) {
    match crate::perf_evidence::PerfQuarantineRegister::from_jsonl(EMBEDDED_QUARANTINE_REGISTER) {
        Ok(register) => {
            if let Err(error) = register.screen(evidence) {
                state.fatal(
                    "perf.ratchet.quarantined_revision",
                    format!("{role} {error}"),
                );
            }
        }
        Err(error) => {
            // A register that will not parse must never read as "nothing is
            // quarantined". Failing closed here means a malformed register
            // blocks promotion instead of silently disarming the screen.
            state.fatal(
                "perf.ratchet.quarantine_register_unusable",
                format!("embedded quarantine register cannot be read: {error}"),
            );
        }
    }
}

fn validate_machine_profile_promotion(
    request: &PerfRatchetRequest<'_>,
    plan: &PerfApplicabilityPlan,
    qg1_authority_sets: PerfRatchetQg1AuthoritySets<'_>,
    qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
    state: &mut DecisionState,
) {
    let Some(expected_profile) = request.expected_machine_profile else {
        state.fatal(
            "perf.ratchet.machine_profile_expectation_missing",
            "promotion requires an expected canonical hardware/profile identity",
        );
        return;
    };
    if expected_profile != plan.binding.profile {
        state.fatal(
            "perf.ratchet.machine_profile_mismatch",
            format!(
                "candidate applicability plan binds {:?}, caller expected {expected_profile:?}",
                plan.binding.profile
            ),
        );
        return;
    }
    let baseline_is_bootstrap = request.baseline.is_some_and(|baseline| {
        is_explicit_bootstrap_for(
            baseline,
            request.candidate.gate,
            request.expected_manifest_sha256,
        )
    });
    let roles = [
        (
            PerfRatchetEvidenceRole::Baseline,
            request.baseline,
            request.baseline_evidence,
            request
                .baseline_evidence
                .and_then(|evidence| evidence.machine_class.identity()),
        ),
        (
            PerfRatchetEvidenceRole::Candidate,
            Some(request.candidate),
            request.candidate_evidence,
            request.candidate_runner_identity,
        ),
        (
            PerfRatchetEvidenceRole::Rerun,
            request.rerun,
            request.rerun_evidence,
            request.rerun_runner_identity,
        ),
    ];
    let mut admitted = Vec::with_capacity(roles.len());
    for (role, artifact, evidence, external_identity) in roles {
        if role == PerfRatchetEvidenceRole::Baseline && baseline_is_bootstrap {
            if evidence.is_some() || external_identity.is_some() {
                state.fatal(
                    "perf.ratchet.bootstrap_identity_fabricated",
                    "the exact unmeasured baseline must not carry fabricated current evidence or \
                     a runner receipt",
                );
            }
            continue;
        }
        let (Some(artifact), Some(evidence), Some(external_identity)) =
            (artifact, evidence, external_identity)
        else {
            state.quarantine(
                "perf.ratchet.machine_identity_incomplete",
                format!(
                    "promotion requires threshold evidence, current evidence, and a verified \
                     runner receipt for {role}"
                ),
            );
            continue;
        };
        validate_threshold_plan_scope(artifact, plan, role.label(), state);
        if evidence.applicability_plan != plan.binding {
            state.fatal(
                "perf.ratchet.evidence_applicability_plan_mismatch",
                format!(
                    "{role} v6 evidence does not bind the exact candidate profile/applicability \
                     plan"
                ),
            );
            continue;
        }
        let artifact_bytes = match serde_json::to_vec_pretty(artifact) {
            Ok(bytes) => bytes,
            Err(error) => {
                state.fatal(
                    "perf.ratchet.threshold_source_invalid",
                    format!("{role} threshold object is not canonically serializable: {error}"),
                );
                continue;
            }
        };
        if let Err(error) = external_identity.verify() {
            state.fatal(
                "perf.ratchet.runner_receipt_rejected",
                format!("{role} runner receipt no longer verifies: {error}"),
            );
            continue;
        }
        if external_identity.profile() != plan.binding.profile
            || external_identity.capacity_semantics() != plan.capacity_semantics
            || plan.execution_capacity != Some(external_identity.execution_capacity())
            || plan.max_exercised_cell_width != Some(external_identity.max_exercised_cell_width())
        {
            state.fatal(
                "perf.ratchet.runner_applicability_envelope_mismatch",
                format!(
                    "{role} runner profile/capacity/maximum envelope does not equal the exact \
                     applicability plan"
                ),
            );
            continue;
        }
        if let Err(error) = external_identity.verify_threshold_artifact(&artifact_bytes) {
            state.fatal(
                "perf.ratchet.runner_receipt_threshold_mismatch",
                format!(
                    "{role} runner receipt does not seal the exact threshold artifact: {error}"
                ),
            );
            continue;
        }
        let Some(artifact_manifest) = external_identity.artifact_manifest() else {
            state.fatal(
                "perf.ratchet.runner_artifact_manifest_missing",
                format!("{role} runner identity has no exact artifact manifest"),
            );
            continue;
        };
        let artifact_manifest = artifact_manifest.manifest();
        if artifact_manifest.gate() != artifact.gate.label()
            || artifact_manifest.run_id() != artifact.run_id
            || artifact_manifest.run_window() != artifact.run_window
            || artifact_manifest.applicability_plan() != plan.binding()
        {
            state.fatal(
                "perf.ratchet.runner_manifest_run_mismatch",
                format!(
                    "{role} artifact manifest names a different gate, run, or applicability plan"
                ),
            );
            continue;
        }
        if let Err(error) = evidence.verify_integrity_against_authorities(
            qg1_authority_sets.for_role(role),
            qg6_authority_sets.for_role(role),
        ) {
            state.fatal(
                "perf.ratchet.machine_evidence_integrity_failed",
                format!("{role} current evidence no longer verifies: {error}"),
            );
            continue;
        }
        // Screened immediately after integrity, because a quarantined artifact
        // passes that check: it is intact evidence of an invalid measurement.
        reject_quarantined_revision(role.label(), evidence, state);
        let Some(bound_identity) = evidence.machine_class.identity() else {
            state.quarantine(
                "perf.ratchet.machine_identity_unverified",
                format!("{role} current evidence has no registry-verified runner identity"),
            );
            continue;
        };
        if evidence.gate_decision.is_some() || !evidence.ratchet_admissible() {
            state.quarantine(
                "perf.ratchet.current_evidence_not_admissible",
                format!(
                    "{role} current evidence is not a provisional, claim-eligible ratchet input"
                ),
            );
            continue;
        }
        if bound_identity != external_identity {
            state.fatal(
                "perf.ratchet.runner_receipt_artifact_mismatch",
                format!(
                    "{role} verified runner identity is not the exact identity bound into current \
                     evidence"
                ),
            );
            continue;
        }
        validate_execution_projection_binding(
            role.label(),
            artifact,
            evidence,
            external_identity,
            plan,
            state,
        );
        if external_identity.profile() != expected_profile {
            state.fatal(
                "perf.ratchet.machine_profile_mismatch",
                format!(
                    "{role} receipt derives profile {:?}, caller expected {expected_profile:?}",
                    external_identity.profile()
                ),
            );
        }
        let expected_gate = request.candidate.gate.label();
        let expected_destination = match external_identity.profile().latest_basename(expected_gate)
        {
            Ok(destination) => destination,
            Err(error) => {
                state.fatal(
                    "perf.ratchet.machine_destination_invalid",
                    format!("{role} profile destination rejected: {error}"),
                );
                continue;
            }
        };
        if external_identity.admission_context().gate != expected_gate
            || external_identity.admission_context().destination_basename != expected_destination
        {
            state.fatal(
                "perf.ratchet.machine_destination_mismatch",
                format!(
                    "{role} receipt was admitted for {}/{} rather than {expected_gate}/\
                     {expected_destination}",
                    external_identity.admission_context().gate,
                    external_identity.admission_context().destination_basename
                ),
            );
        }
        if evidence.gate != artifact.gate {
            state.fatal(
                "perf.ratchet.machine_evidence_gate_mismatch",
                format!(
                    "{role} current evidence is for {}, threshold artifact is for {}",
                    evidence.gate, artifact.gate
                ),
            );
        }
        admitted.push((role, external_identity));
    }
    let candidate_producer = admitted
        .iter()
        .find(|(role, _)| *role == PerfRatchetEvidenceRole::Candidate)
        .and_then(|(_, identity)| identity.build().get("producer"));
    let rerun_producer = admitted
        .iter()
        .find(|(role, _)| *role == PerfRatchetEvidenceRole::Rerun)
        .and_then(|(_, identity)| identity.build().get("producer"));
    if let (Some(candidate_producer), Some(rerun_producer)) = (candidate_producer, rerun_producer)
        && candidate_producer != rerun_producer
    {
        state.fatal(
            "perf.ratchet.candidate_rerun_producer_mismatch",
            "fresh candidate and immediate rerun must share the exact typed-finalizer contract, \
             source, Cargo.lock, and executing-ELF identity",
        );
    }
    let candidate_benchmark_executable = admitted
        .iter()
        .find(|(role, _)| *role == PerfRatchetEvidenceRole::Candidate)
        .and_then(|(_, identity)| identity.build().get("executable_sha256"))
        .and_then(Value::as_str);
    let rerun_benchmark_executable = admitted
        .iter()
        .find(|(role, _)| *role == PerfRatchetEvidenceRole::Rerun)
        .and_then(|(_, identity)| identity.build().get("executable_sha256"))
        .and_then(Value::as_str);
    if let (Some(candidate_benchmark_executable), Some(rerun_benchmark_executable)) =
        (candidate_benchmark_executable, rerun_benchmark_executable)
        && candidate_benchmark_executable != rerun_benchmark_executable
    {
        state.fatal(
            "perf.ratchet.candidate_rerun_benchmark_executable_mismatch",
            "fresh candidate and immediate rerun must execute byte-identical benchmark ELFs",
        );
    }
    let expected_admitted = if baseline_is_bootstrap { 2 } else { 3 };
    if admitted.len() == expected_admitted {
        if admitted
            .windows(2)
            .any(|pair| !pair[0].1.same_execution_identity(pair[1].1))
        {
            state.fatal(
                "perf.ratchet.mixed_machine_identity",
                "measured promotion roles do not share one canonical registry, profile, hardware \
                 fingerprint, capacity envelope, and execution-identity hash",
            );
        }
        let receipt_digests = admitted
            .iter()
            .map(|(_, identity)| identity.receipt_sha256())
            .collect::<BTreeSet<_>>();
        if receipt_digests.len() != expected_admitted {
            state.quarantine(
                "perf.ratchet.runner_receipt_reused",
                "every measured promotion role requires an independently sealed completion receipt",
            );
        }
        let current_evidence = [
            request.baseline_evidence,
            request.candidate_evidence,
            request.rerun_evidence,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if current_evidence.len() == expected_admitted {
            if current_evidence
                .iter()
                .any(|evidence| evidence.policy != current_evidence[0].policy)
            {
                state.fatal(
                    "perf.ratchet.mixed_evidence_policy",
                    "all measured promotion roles must share one exact A/A and evidence policy",
                );
            }
            let run_ids = current_evidence
                .iter()
                .map(|evidence| evidence.provenance.run_id.as_str())
                .collect::<BTreeSet<_>>();
            if run_ids.len() != expected_admitted {
                state.quarantine(
                    "perf.ratchet.run_identity_reused",
                    "every measured promotion role requires a distinct evidence run ID",
                );
            }
            let builds = current_evidence
                .iter()
                .map(|evidence| &evidence.provenance.build)
                .collect::<Vec<_>>();
            if builds
                .iter()
                .any(|build| build.command_sha256 != builds[0].command_sha256)
                || builds
                    .iter()
                    .any(|build| build.environment_sha256 != builds[0].environment_sha256)
            {
                state.fatal(
                    "perf.ratchet.mixed_command_identity",
                    "all measured promotion roles must share exact NUL-delimited argv and \
                     controlled-environment SHA-256 identities",
                );
            }
            if builds.iter().any(|build| {
                build.rustc_version != builds[0].rustc_version
                    || build.target_triple != builds[0].target_triple
                    || build.build_profile != builds[0].build_profile
                    || build.cargo_features != builds[0].cargo_features
            }) {
                state.fatal(
                    "perf.ratchet.mixed_build_context",
                    "baseline, candidate, and rerun must share rustc, target, profile, and feature \
                     context",
                );
            }
        }
    }
}

fn validate_execution_projection_binding(
    role: &str,
    artifact: &PerfGateArtifact,
    evidence: &PerfEvidenceArtifact,
    identity: &VerifiedRunnerIdentity,
    plan: &PerfApplicabilityPlan,
    state: &mut DecisionState,
) {
    let Some(projected) = artifact.execution.as_ref() else {
        state.fatal(
            "perf.ratchet.execution_projection_missing",
            format!("{role} threshold artifact has no execution projection"),
        );
        return;
    };
    let sealed = &evidence.provenance.machine.execution;
    if projected != sealed {
        state.fatal(
            "perf.ratchet.execution_projection_evidence_mismatch",
            format!(
                "{role} threshold execution projection differs from its sealed current evidence"
            ),
        );
    }
    if evidence.provenance.machine.logical_cpus != projected.process_available_threads {
        state.fatal(
            "perf.ratchet.execution_projection_concurrency_mismatch",
            format!(
                "{role} machine logical_cpus does not equal projected process-available threads"
            ),
        );
    }

    let hardware = identity.hardware();
    let receipt_physical = hardware
        .get("physical_cores")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let receipt_logical = hardware
        .get("logical_cpus")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    if receipt_physical != Some(projected.physical_cores)
        || receipt_logical != Some(projected.logical_threads)
    {
        state.fatal(
            "perf.ratchet.execution_projection_receipt_topology_mismatch",
            format!(
                "{role} projected physical/logical topology does not equal the verified receipt"
            ),
        );
    }
    let projected_isa = serde_json::to_value(&projected.runtime_detected_isa)
        .expect("runtime ISA vector is representable as JSON");
    if hardware.get("runtime_detected_isa") != Some(&projected_isa) {
        state.fatal(
            "perf.ratchet.execution_projection_receipt_isa_mismatch",
            format!(
                "{role} projected runtime ISA does not equal the independently admitted receipt"
            ),
        );
    }

    if artifact.applicability_plan.as_ref() != Some(plan.binding())
        || evidence.applicability_plan != plan.binding
        || identity.profile() != plan.binding.profile
        || identity.capacity_semantics() != plan.capacity_semantics
        || plan.execution_capacity != Some(identity.execution_capacity())
        || plan.max_exercised_cell_width != Some(identity.max_exercised_cell_width())
    {
        state.fatal(
            "perf.ratchet.execution_projection_plan_envelope_mismatch",
            format!(
                "{role} threshold, evidence, and verified receipt do not share one exact \
                 profile/capacity/maximum applicability envelope"
            ),
        );
        return;
    }

    let matrix = PerfMatrixSpec::complete();
    let planned_widths = matrix
        .for_gate(artifact.gate)
        .into_iter()
        .zip(&plan.cells)
        .filter(|(_, cell)| cell.applicability.is_runnable())
        .map(|(_, cell)| cell.configured_threads)
        .collect::<BTreeSet<_>>();
    let projected_widths = projected
        .configured_engine_thread_widths
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let Some(execution_capacity) = plan
        .execution_capacity
        .and_then(|value| usize::try_from(value).ok())
    else {
        state.fatal(
            "perf.ratchet.execution_projection_plan_envelope_mismatch",
            format!("{role} applicability plan has no fixed execution capacity"),
        );
        return;
    };
    let Some(maximum) = plan
        .max_exercised_cell_width
        .and_then(|value| usize::try_from(value).ok())
    else {
        state.fatal(
            "perf.ratchet.execution_projection_plan_envelope_mismatch",
            format!("{role} applicability plan has no fixed maximum exercised width"),
        );
        return;
    };
    if maximum > execution_capacity
        || usize::try_from(projected.execution_capacity) != Ok(execution_capacity)
        || usize::try_from(projected.max_exercised_cell_width) != Ok(maximum)
        || !projected.matches_capacity_semantics(plan.capacity_semantics)
    {
        state.fatal(
            "perf.ratchet.execution_projection_capacity_mismatch",
            format!(
                "{role} projected execution capacity or maximum differs from the verified \
                 applicability-plan envelope, or process availability contradicts its capacity \
                 semantics"
            ),
        );
    }
    if projected_widths != planned_widths
        || projected.configured_engine_thread_widths.len() != planned_widths.len()
        || planned_widths.iter().next_back().copied() != Some(maximum)
    {
        state.fatal(
            "perf.ratchet.execution_projection_widths_mismatch",
            format!(
                "{role} projected engine widths do not equal the exact runnable applicability plan"
            ),
        );
    }
    for cell in &evidence.cells {
        if let Some(witness) = cell.spec.concurrency_witness.as_ref()
            && (witness.configured_threads > maximum
                || !planned_widths.contains(&witness.configured_threads))
        {
            state.fatal(
                "perf.ratchet.concurrency_witness_receipt_mismatch",
                format!(
                    "{role} cell {} materialized width {} outside the exact runnable plan or \
                     admitted receipt maximum",
                    cell.cell_id, witness.configured_threads
                ),
            );
        }
    }

    let observed_logical_threads = identity
        .execution_start()
        .get("observed_logical_cpu_ids")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .filter(|count| *count > 0);
    if observed_logical_threads.is_some_and(|count| projected.process_available_threads > count)
        || (observed_logical_threads.is_some() && projected.cpu_affinity_allowed_list.is_none())
    {
        state.fatal(
            "perf.ratchet.execution_projection_cpuset_mismatch",
            format!(
                "{role} projected process concurrency or affinity exceeds the verified receipt"
            ),
        );
    }
}

fn evaluate_perf_ratchet_inner(
    request: PerfRatchetRequest<'_>,
    qg1_authority_sets: PerfRatchetQg1AuthoritySets<'_>,
    qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
    mut state: DecisionState,
    require_current_evidence: bool,
) -> PerfRatchetEvaluation {
    let gate = request.candidate.gate;
    let Some(plan) = candidate_applicability_plan(&request, &mut state) else {
        return finish_evaluation(
            gate,
            request.mode,
            request.gate_activated,
            state,
            Vec::new(),
            request.evidence,
        );
    };
    validate_threshold_plan_scope(request.candidate, &plan, "candidate", &mut state);
    if let Some(baseline) = request.baseline {
        validate_threshold_plan_scope(baseline, &plan, "baseline", &mut state);
    }
    if let Some(rerun) = request.rerun {
        validate_threshold_plan_scope(rerun, &plan, "rerun", &mut state);
    }
    if state.fatal {
        return finish_evaluation(
            gate,
            request.mode,
            request.gate_activated,
            state,
            Vec::new(),
            request.evidence,
        );
    }
    let baseline_is_bootstrap = request.baseline.is_some_and(|baseline| {
        is_explicit_bootstrap_for(baseline, gate, request.expected_manifest_sha256)
    });
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
                &plan,
                request.expected_manifest_sha256,
                PerfRatchetEvidenceRole::Candidate,
                qg1_authority_sets.for_role(PerfRatchetEvidenceRole::Candidate),
                qg6_authority_sets.for_role(PerfRatchetEvidenceRole::Candidate),
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
        None if require_current_evidence => {
            state.quarantine(
                "perf.ratchet.missing_current_candidate_evidence",
                "promotion requires a hash-sealed current-schema candidate evidence artifact",
            );
            None
        }
        None => None,
    };

    if request.mode == PerfRatchetMode::Promotion {
        validate_complete_gate(&plan, &candidate_cells, request.gate_activated, &mut state);
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
        match request.baseline_evidence {
            Some(evidence) => {
                if !validate_current_evidence(
                    evidence,
                    baseline,
                    &baseline_cells,
                    &plan,
                    request.expected_manifest_sha256,
                    PerfRatchetEvidenceRole::Baseline,
                    qg1_authority_sets.for_role(PerfRatchetEvidenceRole::Baseline),
                    qg6_authority_sets.for_role(PerfRatchetEvidenceRole::Baseline),
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
            }
            None if require_current_evidence && !baseline_is_bootstrap => state.quarantine(
                "perf.ratchet.missing_current_baseline_evidence",
                "promotion requires hash-sealed current-schema baseline evidence",
            ),
            None => {}
        }
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
            "no committed baseline exists for this gate and machine profile",
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
                        &plan,
                        request.expected_manifest_sha256,
                        PerfRatchetEvidenceRole::Rerun,
                        qg1_authority_sets.for_role(PerfRatchetEvidenceRole::Rerun),
                        qg6_authority_sets.for_role(PerfRatchetEvidenceRole::Rerun),
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
                None if require_current_evidence => {
                    state.quarantine(
                        "perf.ratchet.missing_current_rerun_evidence",
                        "promotion requires a hash-sealed current-schema rerun evidence artifact",
                    );
                    None
                }
                None => None,
            };
            if candidate_is_complete(&plan, &rerun_cells) {
                // Promotion requires both independent passes to satisfy every
                // gate target. Reproduction tolerance cannot substitute for
                // independently clearing a threshold; QG-6 additionally
                // consumes the rerun's hierarchical CI/null-margin evidence.
                evaluate_gate_targets(
                    rerun,
                    &rerun_cells,
                    rerun_evidence,
                    request.gate_activated,
                    baseline_is_bootstrap,
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

    if request.mode == PerfRatchetMode::Promotion && candidate_is_complete(&plan, &candidate_cells)
    {
        evaluate_gate_targets(
            request.candidate,
            &candidate_cells,
            candidate_evidence,
            request.gate_activated,
            baseline_is_bootstrap,
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
    let explicit_bootstrap = is_explicit_bootstrap(artifact);
    if !explicit_bootstrap && let Some(binding) = artifact.applicability_plan.as_ref() {
        let _ = reconstruct_applicability_plan(binding, artifact.gate, role, state);
    } else if !explicit_bootstrap {
        state.fatal(
            "perf.ratchet.measured_applicability_plan_missing",
            format!(
                "{role} measured v8 threshold artifact requires an exact applicability-plan \
                 binding"
            ),
        );
    }
    if artifact.run_window.trim().is_empty() || artifact.run_id.trim().is_empty() {
        state.fatal(
            "perf.ratchet.missing_run_identity",
            format!("{role} must record non-empty run_window and run_id values"),
        );
    }
    if !explicit_bootstrap
        && !artifact
            .execution
            .as_ref()
            .is_some_and(PerfExecutionProvenance::is_complete)
    {
        state.fatal(
            "perf.ratchet.missing_execution_provenance",
            format!(
                "{role} must record host identity, physical/logical topology, effective threads, \
                 runtime ISA, and affinity/cpuset provenance"
            ),
        );
    }
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

fn expected_evidence_cells(
    plan: &PerfApplicabilityPlan,
) -> BTreeMap<EvidenceCellKey, EvidenceRole> {
    PerfMatrixSpec::complete()
        .for_gate(plan.binding.gate)
        .into_iter()
        .zip(&plan.cells)
        .filter_map(|(spec, classification)| {
            let role = match classification.applicability {
                PerfCellApplicability::Required => EvidenceRole::Required,
                PerfCellApplicability::Diagnostic => EvidenceRole::Diagnostic,
                PerfCellApplicability::NotApplicable => return None,
            };
            Some((
                EvidenceCellKey::from_parts(&spec.fixture, &spec.metric, metric_unit(&spec.metric)),
                role,
            ))
        })
        .collect()
}

fn validate_current_evidence(
    evidence: &PerfEvidenceArtifact,
    legacy: &PerfGateArtifact,
    legacy_cells: &BTreeMap<CellKey, &PerfCellResult>,
    plan: &PerfApplicabilityPlan,
    expected_manifest_sha256: &str,
    role: PerfRatchetEvidenceRole,
    qg1_authorities: &[&Qg1ExpectedAuthority],
    qg6_authorities: &[&Qg6ScheduleAuthority],
    state: &mut DecisionState,
) -> bool {
    let role = role.label();
    if evidence.applicability_plan != plan.binding {
        state.fatal(
            "perf.ratchet.evidence_applicability_plan_mismatch",
            format!(
                "{role} v6 evidence does not bind the exact threshold profile/applicability plan"
            ),
        );
        return false;
    }
    if let Err(error) =
        evidence.verify_integrity_against_authorities(qg1_authorities, qg6_authorities)
    {
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

    let expected = expected_evidence_cells(plan);
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
        if normative.is_none() {
            state.fatal(
                "perf.ratchet.current_evidence_cell_not_runnable",
                format!(
                    "{role} current evidence measured noncanonical or NotApplicable cell \
                     {}/{}/{}",
                    key.fixture, key.metric, key.unit
                ),
            );
        }
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
        if matches!(legacy.gate, PerfGate::Qg1 | PerfGate::Qg8)
            && normative == Some(EvidenceRole::Required)
        {
            let expected_threads = PerfMatrixSpec::complete()
                .for_gate(legacy.gate)
                .into_iter()
                .find(|spec| spec.fixture == cell.spec.fixture && spec.metric == cell.spec.metric)
                .and_then(|spec| spec.threads);
            let observed_threads = cell
                .spec
                .concurrency_witness
                .as_ref()
                .map(|witness| witness.configured_threads);
            if observed_threads != expected_threads {
                state.fatal(
                    "perf.ratchet.concurrency_witness_matrix_mismatch",
                    format!(
                        "{role} cell {} concurrency witness {:?} does not match normative width \
                         {:?}",
                        cell.cell_id, observed_threads, expected_threads
                    ),
                );
            }
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
    let unexpected = actual
        .iter()
        .filter(|(key, actual_role)| expected.get(*key) != Some(*actual_role))
        .count();
    if unexpected != 0 {
        state.fatal(
            "perf.ratchet.current_evidence_plan_mismatch",
            format!(
                "{role} current evidence is outside the exact runnable {} plan: {unexpected} \
                 unexpected or wrong-role cells",
                legacy.gate
            ),
        );
    }
    if missing != 0 {
        state.quarantine(
            "perf.ratchet.current_evidence_incomplete_plan",
            format!(
                "{role} current evidence omits {missing} Required or Diagnostic cells from the \
                 exact runnable {} plan",
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
            treatment_arm_null,
            qg6_protocol,
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
                && (hierarchical.is_none() || hierarchical_null.is_none() || qg6_protocol.is_none())
            {
                state.fatal(
                    "perf.ratchet.qg6_joint_tail_evidence_missing",
                    format!(
                        "{role} QG-6 cell {} lacks a verified hierarchical compatibility estimate \
                         or formal six-arm joint tail estimate",
                        cell.cell_id
                    ),
                );
            }
            if normative {
                reconcile_current_cell_with_projection(
                    cell,
                    paired,
                    treatment_arm_null.as_deref(),
                    qg6_protocol.as_deref(),
                    legacy_cells,
                    role,
                    state,
                );
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
    treatment_arm_null: Option<&crate::PairedExperimentResult>,
    qg6_protocol: Option<&crate::Qg6FormalProtocolEvidence>,
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
    let treatment_null = legacy_cells.get(&ratio("quill_over_quill", "paired_null_quill"));
    let projected_effect = projected_ratio_distribution(&paired.effect_samples);
    let projected_null = projected_ratio_distribution(&paired.null_samples);
    let qg6_absolute = if cell.spec.gate == PerfGate::Qg6 {
        match crate::project_qg6_effect_leaf_distributions(&paired.effect_samples, &paired.config) {
            Ok(distributions) => Some(distributions),
            Err(error) => {
                state.fatal(
                    "perf.ratchet.qg6_leaf_projection_invalid",
                    format!(
                        "{role} cell {} cannot reconstruct authenticated QG-6 search leaves: \
                         {error}",
                        cell.cell_id
                    ),
                );
                return;
            }
        }
    } else {
        None
    };
    let expected_treatment = qg6_absolute
        .as_ref()
        .map_or(&paired.effect.treatment, |projection| &projection.treatment);
    let expected_control = qg6_absolute
        .as_ref()
        .map_or(&paired.effect.control, |projection| &projection.control);
    let treatment_null_aligned = match cell.spec.gate {
        PerfGate::Qg1 => treatment_arm_null.is_some_and(|evidence| {
            treatment_null.is_some_and(|projected| {
                projected_ratio_distribution(&evidence.null_samples)
                    .as_ref()
                    .is_some_and(|summary| projected.distribution == *summary)
            })
        }),
        PerfGate::Qg6 => qg6_protocol.is_some_and(|evidence| {
            treatment_null.is_some_and(|projected| {
                projected_ratio_distribution(&evidence.quill_null_samples)
                    .as_ref()
                    .is_some_and(|summary| projected.distribution == *summary)
            })
        }),
        _ => treatment_arm_null.is_none() && qg6_protocol.is_none() && treatment_null.is_none(),
    };
    let aligned = treatment_null_aligned
        && treatment.is_some_and(|projected| projected.distribution == *expected_treatment)
        && control.is_some_and(|projected| projected.distribution == *expected_control)
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
                "{role} cell {} does not reproduce both absolute arms plus A/B and required \
                 per-arm A/A medians in its legacy threshold projection",
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
    if candidate.applicability_plan != rerun.applicability_plan {
        state.quarantine(
            "perf.ratchet.current_rerun_applicability_plan_mismatch",
            "candidate and rerun current evidence must share one exact profile/applicability plan",
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

    let (candidate_tail, rerun_tail) = match (&candidate_cell.body, &rerun_cell.body) {
        (
            EvidenceCellBody::Paired {
                qg6_protocol: Some(candidate_protocol),
                ..
            },
            EvidenceCellBody::Paired {
                qg6_protocol: Some(rerun_protocol),
                ..
            },
        ) => (&candidate_protocol.joint_tail, &rerun_protocol.joint_tail),
        _ => {
            state.quarantine(
                "perf.ratchet.qg6_joint_tail_reproduction_missing",
                format!(
                    "QG-6 reproduction requires formal joint-tail estimates for {}",
                    candidate_cell.cell_id
                ),
            );
            return;
        }
    };
    let compatible_tail_topology = candidate_tail.schema_version == rerun_tail.schema_version
        && candidate_tail.query_count == rerun_tail.query_count
        && candidate_tail.units_per_query == rerun_tail.units_per_query
        && candidate_tail.leaves_per_arm_per_unit == rerun_tail.leaves_per_arm_per_unit
        && candidate_tail.bootstrap_resamples == rerun_tail.bootstrap_resamples;
    if !compatible_tail_topology {
        state.quarantine(
            "perf.ratchet.qg6_joint_tail_reproduction_incompatible",
            format!(
                "QG-6 candidate and rerun joint-tail topology differs for {}",
                candidate_cell.cell_id
            ),
        );
        return;
    }

    for (comparison, candidate, rerun) in [
        ("Quill/Tantivy", &candidate_tail.effect, &rerun_tail.effect),
        (
            "Tantivy/Tantivy",
            &candidate_tail.tantivy_null,
            &rerun_tail.tantivy_null,
        ),
        (
            "Quill/Quill",
            &candidate_tail.quill_null,
            &rerun_tail.quill_null,
        ),
    ] {
        for (quantile, candidate_ratio, rerun_ratio) in [
            ("p50", candidate.p50_ratio, rerun.p50_ratio),
            ("p99", candidate.p99_ratio, rerun.p99_ratio),
        ] {
            let delta = (candidate_ratio.ln() - rerun_ratio.ln()).abs();
            if !delta.is_finite() || delta > candidate_pair.config.max_reproduction_delta_log {
                state.quarantine(
                    "perf.ratchet.qg6_joint_tail_reproduction_failed",
                    format!(
                        "QG-6 joint-tail {comparison} {quantile} candidate and rerun differ by \
                         {delta:.6} log-ratio for {}",
                        candidate_cell.cell_id
                    ),
                );
            }
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
        let mut nulls = vec![null];
        if gate == PerfGate::Qg1 {
            let treatment_null_key = CellKey {
                fixture: key.fixture.clone(),
                metric: format!("{metric_stem}_quill_over_quill"),
                engine: "paired_null_quill".to_owned(),
                unit: "ratio".to_owned(),
            };
            let Some(treatment_null) = cells.get(&treatment_null_key).copied() else {
                state.quarantine(
                    "perf.ratchet.missing_treatment_arm_null_control",
                    format!(
                        "{role} QG-1 paired claim {}/{} has no same-invocation Quill/Quill \
                         A/A null row",
                        key.fixture, key.metric
                    ),
                );
                continue;
            };
            nulls.push(treatment_null);
        }
        let mut nulls_valid = true;
        for null in &nulls {
            nulls_valid &= validate_null_control(null, role, state);
        }
        if !nulls_valid {
            continue;
        }

        let null_ci_low = nulls
            .iter()
            .map(|null| null.distribution.median_ci95_low)
            .fold(f64::INFINITY, f64::min);
        let null_ci_high = nulls
            .iter()
            .map(|null| null.distribution.median_ci95_high)
            .fold(f64::NEG_INFINITY, f64::max);
        let null_floor = (null_ci_low - 1.0).abs().max((null_ci_high - 1.0).abs());

        let effect = (claim.distribution.p50 - 1.0).abs();
        let outside_null =
            claim.distribution.p50 < null_ci_low || claim.distribution.p50 > null_ci_high;
        if !outside_null || effect < PERF_NULL_MARGIN_MULTIPLIER * null_floor {
            state.quarantine(
                "perf.ratchet.inconclusive_paired_claim",
                format!(
                    "{role} {}/{} median {:.6} does not clear A/A median CI \
                     [{:.6}, {:.6}] with the required {:.1}x margin",
                    key.fixture,
                    key.metric,
                    claim.distribution.p50,
                    null_ci_low,
                    null_ci_high,
                    PERF_NULL_MARGIN_MULTIPLIER,
                ),
            );
        }
    }
}

fn expected_gate_keys(plan: &PerfApplicabilityPlan) -> BTreeSet<CellKey> {
    PerfMatrixSpec::complete()
        .for_gate(plan.binding.gate)
        .into_iter()
        .zip(&plan.cells)
        .filter(|(_, classification)| classification.applicability.is_runnable())
        .flat_map(|(spec, _)| {
            if plan.binding.gate == PerfGate::Qg10 {
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
            let mut keys = vec![
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
            ];
            if matches!(plan.binding.gate, PerfGate::Qg1 | PerfGate::Qg6) {
                keys.push(CellKey {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_quill_over_quill", spec.metric),
                    engine: "paired_null_quill".to_owned(),
                    unit: "ratio".to_owned(),
                });
            }
            keys
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
    plan: &PerfApplicabilityPlan,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
) -> bool {
    candidate_cells.keys().cloned().collect::<BTreeSet<_>>() == expected_gate_keys(plan)
}

fn validate_complete_gate(
    plan: &PerfApplicabilityPlan,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
    activated: bool,
    state: &mut DecisionState,
) {
    if candidate_is_complete(plan, candidate_cells) {
        return;
    }
    let gate = plan.binding.gate;
    let expected = expected_gate_keys(plan);
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
    let explicit_bootstrap = is_explicit_bootstrap(baseline);
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
    if baseline.applicability_plan != candidate.applicability_plan {
        state.fatal(
            "perf.ratchet.history_applicability_plan_mismatch",
            "baseline and candidate history objects do not share one exact profile/applicability \
             plan",
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
    if baseline.execution != candidate.execution {
        state.quarantine(
            "perf.ratchet.execution_provenance_mismatch",
            "baseline and candidate differ in host topology, effective threads, ISA, or \
             affinity/cpuset provenance",
        );
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
            && matches!(
                key.engine.as_str(),
                "paired_ab" | "paired_null" | "paired_null_quill"
            )
        {
            // QG-6's flat paired rows are compatibility projections. Current
            // hierarchical evidence owns effect and null inference.
            continue;
        }
        if matches!(key.engine.as_str(), "paired_null" | "paired_null_quill") {
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
                        "oracle row {}/{}/{} moved more than {:.1}%; rerun on a quiet same-profile host",
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

/// Whether an artifact is the one exact current-schema unmeasured baseline
/// sentinel. This does not validate its gate or manifest binding.
#[must_use]
pub fn is_explicit_bootstrap(artifact: &PerfGateArtifact) -> bool {
    artifact.schema_version == PERF_ARTIFACT_SCHEMA_VERSION
        && artifact.applicability_plan.is_none()
        && artifact.cells.is_empty()
        && artifact.machine_fingerprint == "unmeasured"
        && artifact.bench_elf_sha256 == "unmeasured"
        && artifact.execution.is_none()
        && artifact.git_rev == "unmeasured"
        && artifact.run_window == "unmeasured"
        && artifact.run_id == "unmeasured"
        && artifact.corpus_manifest_hash == ZERO_SHA256
        && !artifact.laws_attested
}

/// Whether an artifact is the exact unmeasured baseline sentinel for one
/// evaluated gate and manifest.
#[must_use]
pub fn is_explicit_bootstrap_for(
    artifact: &PerfGateArtifact,
    gate: PerfGate,
    expected_manifest_sha256: &str,
) -> bool {
    is_explicit_bootstrap(artifact)
        && artifact.gate == gate
        && artifact.manifest_sha256 == expected_manifest_sha256
}

fn compare_reproduction(
    candidate: &PerfGateArtifact,
    rerun: &PerfGateArtifact,
    candidate_cells: &BTreeMap<CellKey, &PerfCellResult>,
    rerun_cells: &BTreeMap<CellKey, &PerfCellResult>,
    state: &mut DecisionState,
) {
    if candidate.applicability_plan != rerun.applicability_plan {
        state.quarantine(
            "perf.ratchet.rerun_applicability_plan_mismatch",
            "candidate and rerun thresholds must share one exact profile/applicability plan",
        );
        return;
    }
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
    if candidate.execution != rerun.execution {
        state.quarantine(
            "perf.ratchet.rerun_execution_provenance_mismatch",
            "candidate and rerun must share host topology, effective threads, ISA, and \
             affinity/cpuset provenance",
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
            && matches!(
                key.engine.as_str(),
                "paired_ab" | "paired_null" | "paired_null_quill"
            )
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

/// Admit an A/A control on its **accuracy** — is its median at 1.0? — and never
/// on how precisely it was measured.
///
/// # Why not "the median CI must contain 1.0"
///
/// That clause, which this replaced (`bd-pjh09`), coupled admission to the
/// null's precision in the wrong direction: a tighter null has a narrower CI,
/// so it is *more* likely to exclude 1.0 and quarantine its own row — whatever
/// its residual bias, and whatever the size of the effect being measured. It
/// punished exactly the measurement quality this harness exists to buy, and it
/// made admission a property of the host's noise rather than of the code.
///
/// Measured on one ELF across `taskset`-pinned cores, a reproducible planted
/// effect held its ratio to 1.16% while that clause flipped its verdict on 6 of
/// 20 cells, purely as a function of which core and how many rounds. Every one
/// of those vetoed nulls had a median within 0.21% of 1.0 — they were rejected
/// for being precise, not for being wrong.
///
/// The CI stays in the quarantine message as provenance, and the null's spread
/// still sets the floor a claim has to clear elsewhere; only the median decides
/// whether the sampler itself can be trusted.
fn validate_null_control(cell: &PerfCellResult, role: &str, state: &mut DecisionState) -> bool {
    let drift_pct = (cell.distribution.p50 - 1.0).abs() * 100.0;
    let unbiased = drift_pct <= PERF_MAX_NULL_MEDIAN_DRIFT_PCT;
    if !unbiased {
        state.quarantine(
            "perf.ratchet.invalid_null_control",
            format!(
                "{role} {}/{}/{} A/A median {:.6} drifts {drift_pct:.3}% from 1.0 \
                 (limit {PERF_MAX_NULL_MEDIAN_DRIFT_PCT:.3}%; median CI [{:.6}, {:.6}] \
                 and cv_pct={:.3} are provenance only)",
                cell.fixture,
                cell.metric,
                cell.engine,
                cell.distribution.p50,
                cell.distribution.median_ci95_low,
                cell.distribution.median_ci95_high,
                cell.distribution.cv_pct,
            ),
        );
    }
    unbiased
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
    observe_only: bool,
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
        if self.observe_only && !self.activated {
            self.state
                .note("perf.ratchet.bootstrap_target_missed", message);
        } else if self.activated {
            self.state.block("perf.ratchet.gate_target_missed", message);
        } else {
            self.state
                .quarantine("perf.ratchet.provisional_target_missed", message);
        }
    }

    fn target_inconclusive(&mut self, message: impl Into<String>) {
        self.state
            .quarantine("perf.ratchet.gate_target_ci_inconclusive", message);
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
            self.target_inconclusive(format!("{message}; the CI crosses {threshold:.6}"));
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
            self.target_inconclusive(format!("{message}; the CI crosses {threshold:.6}"));
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
            self.target_inconclusive(format!(
                "{message}; the median CI overlaps but is not contained in \
                 [{allowed_low:.6}, {allowed_high:.6}]"
            ));
        }
    }
}

fn evaluate_gate_targets(
    artifact: &PerfGateArtifact,
    cells: &BTreeMap<CellKey, &PerfCellResult>,
    current_evidence: Option<&PerfEvidenceArtifact>,
    activated: bool,
    observe_only: bool,
    state: &mut DecisionState,
) {
    let mut target = GateTargetEvaluator {
        artifact,
        cells,
        activated,
        observe_only,
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
    // The e6.1 generator has landed, so this re-baselined ratchet pin matches
    // the xlarge QG-5 cell emitted by `PerfMatrixSpec`.
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
                        "QG-6 {fixture} cannot establish +/-10% equivalence: hierarchical A/A \
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

        let joint_tail =
            current_evidence.and_then(|artifact| exact_qg6_joint_tail_cell(artifact, &fixture));
        if let Some(tail) = joint_tail {
            let ci_low = tail.effect.p50_ci95_low_ratio;
            let ci_high = tail.effect.p50_ci95_high_ratio;
            target.target_interval_ci(
                ci_low,
                ci_high,
                0.90,
                1.10,
                format!(
                    "QG-6 {fixture} joint true-leaf p50 ratio CI \
                    [{ci_low:.6}, {ci_high:.6}] is not contained in [0.90, 1.10]"
                ),
            );
            target.target_lower_ci(
                tail.effect.p99_ci95_low_ratio,
                tail.effect.p99_ci95_high_ratio,
                1.0,
                format!(
                    "QG-6 {fixture} joint true-leaf p99 ratio CI [{:.6}, {:.6}] does not clear \
                     oracle parity",
                    tail.effect.p99_ci95_low_ratio, tail.effect.p99_ci95_high_ratio,
                ),
            );
            for (engine, null) in [
                ("Tantivy/Tantivy", &tail.tantivy_null),
                ("Quill/Quill", &tail.quill_null),
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
                        target.state.quarantine(
                            "perf.ratchet.inconclusive_equivalence",
                            format!(
                                "QG-6 {fixture} {engine} {quantile} null CI \
                                 [{low:.6}, {high:.6}] fails identity containment or the required \
                                 {PERF_NULL_MARGIN_MULTIPLIER:.1}x margin",
                            ),
                        );
                    }
                }
            }
        } else {
            target.state.quarantine(
                "perf.ratchet.qg6_joint_tail_evidence_missing",
                format!(
                    "QG-6 {fixture} requires verified six-arm query-first joint p50/p99 evidence; \
                     flat threshold projections are not decision inputs"
                ),
            );
        }
    }
}

fn exact_qg6_joint_tail_cell<'a>(
    artifact: &'a PerfEvidenceArtifact,
    fixture: &str,
) -> Option<&'a crate::Qg6JointTailEstimate> {
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
            qg6_protocol: Some(protocol),
            ..
        } if paired.provenance.corpus_sha256 == artifact.provenance.corpus.corpus_sha256
            && paired.provenance.input_identity == cell.spec.input_identity
            && cell.claim_eligible() =>
        {
            Some(&protocol.joint_tail)
        }
        _ => None,
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

/// Reviewed QG-8 own-arm scaling floor. x86 uses 16-vs-4; Apple scheduler
/// profiles use the same 1.8x floor on their widest runnable cell (8-vs-4).
const QG8_OWN_SCALING_THRESHOLD: f64 = 1.8;

struct Qg8ScalingComparison {
    wide_threads: u32,
    baseline_threads: u32,
    label: &'static str,
}

fn qg8_scaling_comparison(
    plan: &PerfApplicabilityPlanBinding,
) -> Result<Qg8ScalingComparison, String> {
    match (
        plan.profile.hardware_class_id(),
        plan.profile.execution_profile_id(),
    ) {
        (
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64 | ExecutionProfileId::Smt2_128,
        ) => Ok(Qg8ScalingComparison {
            wide_threads: 16,
            baseline_threads: 4,
            label: "16-thread/4-thread",
        }),
        (HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10)
        | (HardwareClassId::M5Macos, ExecutionProfileId::Scheduler14) => Ok(Qg8ScalingComparison {
            wide_threads: 8,
            baseline_threads: 4,
            label: "8-thread/4-thread",
        }),
        (HardwareClassId::X86VpsOvh, ExecutionProfileId::X86Diagnostic) => {
            Err("QG-8 rejects diagnostic x86-vps-ovh/x86-diagnostic evidence".to_owned())
        }
        (hardware, profile) => Err(format!(
            "QG-8 has no scaling contract for {}/{}",
            hardware.as_str(),
            profile.as_str()
        )),
    }
}

fn evaluate_qg8(target: &mut GateTargetEvaluator<'_, '_>) {
    let Some(plan) = target.artifact.applicability_plan.as_ref() else {
        target.target(
            false,
            "QG-8 requires an applicability-plan hardware/profile identity",
        );
        return;
    };
    let comparison = match qg8_scaling_comparison(plan) {
        Ok(comparison) => comparison,
        Err(message) => {
            target.target(false, message);
            return;
        }
    };
    let wide_fixture = format!("scaling/xlarge/{}/positions_on", comparison.wide_threads);
    let baseline_fixture = format!(
        "scaling/xlarge/{}/positions_on",
        comparison.baseline_threads
    );
    if let (Some(wide), Some(baseline)) = (
        target.value(&wide_fixture, "docs_per_second", "quill"),
        target.value(&baseline_fixture, "docs_per_second", "quill"),
    ) {
        let ratio = wide / baseline.max(f64::MIN_POSITIVE);
        target.target(
            ratio >= QG8_OWN_SCALING_THRESHOLD,
            format!(
                "QG-8 {} scaling {ratio:.6} is below {QG8_OWN_SCALING_THRESHOLD}",
                comparison.label
            ),
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
    use crate::perf_evidence::qg6_test_fixture;
    use crate::{
        BuildIdentity, CorpusIdentity, DistributionSummary, EvidenceArtifactError, EvidenceCell,
        EvidenceCellSpec, EvidencePolicy, EvidenceProvenance, MachineIdentity,
        PairedEstimatorConfig, PeakRssEvidence, PerfCellResult, PerfRawSample, PerfSampleArm,
        PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance, estimate_paired_experiment,
        seeded_balanced_pair_order,
    };
    use sha2::{Digest, Sha256};

    const TEST_MACHINE_FINGERPRINT: &str =
        "linux-x86_64-test-machine-128thread-AMD_Ryzen_Threadripper_PRO_5995WX_64-Cores";
    const QG6_TEST_SCHEDULE_SEED: u64 = 0x5156_0006;

    fn reseal_evidence_without_verification(evidence: &mut PerfEvidenceArtifact) {
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
    }

    fn seal_evidence(
        evidence: &mut PerfEvidenceArtifact,
        qg6_authorities: &[&Qg6ScheduleAuthority],
    ) {
        reseal_evidence_without_verification(evidence);
        evidence
            .verify_integrity_against_authorities(&[], qg6_authorities)
            .expect("test evidence must be integrity-valid after sealing");
    }

    fn threshold_artifact_bytes(artifact: &PerfGateArtifact) -> Vec<u8> {
        serde_json::to_vec_pretty(artifact).expect("test threshold artifact bytes")
    }

    fn try_bind_test_evidence(
        artifact: &PerfGateArtifact,
        evidence: &mut PerfEvidenceArtifact,
        run_label: &str,
        qg6_authorities: &[&Qg6ScheduleAuthority],
    ) -> Result<(), EvidenceArtifactError> {
        evidence.machine_class = crate::MachineClassEvidenceBinding::unverified(
            "sealed runner receipt has not been bound",
        );
        evidence.gate_decision = None;
        evidence.artifact_sha256.clear();
        reseal_evidence_without_verification(evidence);
        evidence.verify_integrity_against_authorities(&[], qg6_authorities)?;
        let evidence_bytes =
            serde_json::to_vec_pretty(evidence).expect("pre-binding evidence bytes");
        let threshold_bytes = threshold_artifact_bytes(artifact);
        let build = &evidence.provenance.build;
        let identity = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            artifact.gate.label(),
            &build.git_revision,
            build
                .cargo_lock_sha256
                .as_deref()
                .expect("test Cargo.lock digest"),
            &build.executable_sha256,
            &build.command_sha256,
            build
                .environment_sha256
                .as_deref()
                .expect("test environment digest"),
            run_label,
            &artifact.run_id,
            &artifact.run_window,
            &threshold_bytes,
            &evidence_bytes,
        );
        evidence.bind_machine_class_identity_against_authorities(
            identity,
            &threshold_bytes,
            &evidence_bytes,
            &[],
            qg6_authorities,
        )?;
        reseal_evidence_without_verification(evidence);
        evidence.verify_integrity_against_authorities(&[], qg6_authorities)
    }

    fn bind_test_evidence(
        artifact: &PerfGateArtifact,
        evidence: &mut PerfEvidenceArtifact,
        run_label: &str,
        qg6_authorities: &[&Qg6ScheduleAuthority],
    ) {
        try_bind_test_evidence(artifact, evidence, run_label, qg6_authorities)
            .expect("bind test evidence to exact receipt artifacts");
    }

    fn mutate_cell_sample_provenance(
        evidence: &mut PerfEvidenceArtifact,
        mut mutate: impl FnMut(&mut PerfSampleProvenance),
    ) {
        fn mutate_experiment(
            experiment: &mut crate::PairedExperimentResult,
            mutate: &mut impl FnMut(&mut PerfSampleProvenance),
        ) {
            mutate(&mut experiment.provenance);
            for sample in experiment
                .effect_samples
                .iter_mut()
                .chain(&mut experiment.null_samples)
            {
                mutate(&mut sample.provenance);
            }
        }

        for cell in &mut evidence.cells {
            if let EvidenceCellBody::Paired {
                paired,
                treatment_arm_null,
                qg6_protocol,
                ..
            } = &mut cell.body
            {
                mutate_experiment(paired, &mut mutate);
                if let Some(treatment_arm_null) = treatment_arm_null {
                    mutate_experiment(treatment_arm_null, &mut mutate);
                }
                if let Some(qg6_protocol) = qg6_protocol {
                    for sample in &mut qg6_protocol.quill_null_samples {
                        mutate(&mut sample.provenance);
                    }
                }
            }
        }
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

    /// An A/A control is admitted on its median, not on how wide its CI is.
    ///
    /// Both fixtures are shapes the retired straddle clause got backwards
    /// (`bd-pjh09`): it quarantined the precise one for excluding 1.0, and
    /// waved the biased one through because its wide CI happened to span 1.0.
    #[test]
    fn null_control_admits_on_median_accuracy_not_on_ci_width() {
        let null_cell = |p50: f64, low: f64, high: f64| PerfCellResult {
            fixture: "bulk/medium/1/positions_on".to_owned(),
            metric: "docs_per_second_tantivy_over_tantivy".to_owned(),
            engine: "paired_null".to_owned(),
            unit: "ratio".to_owned(),
            distribution: DistributionSummary {
                median_ci95_low: low,
                median_ci95_high: high,
                ..distribution(p50)
            },
        };

        // Precise: a CI 0.0007% wide sitting entirely above 1.0, median 0.02%
        // off identity. The retired clause quarantined this.
        let precise = null_cell(1.0002, 1.000_201, 1.000_208);
        assert!(
            !(precise.distribution.median_ci95_low <= 1.0
                && 1.0 <= precise.distribution.median_ci95_high),
            "fixture must exclude 1.0 or it does not exercise the retired clause"
        );
        let mut state = DecisionState::default();
        assert!(validate_null_control(&precise, "candidate", &mut state));
        assert!(
            state.reasons.is_empty(),
            "a precise null must not be quarantined: {:?}",
            state.reasons
        );

        // Biased: median 5% off identity, CI spanning 1.0 with room. The
        // retired clause admitted this without complaint.
        let biased = null_cell(1.05, 0.90, 1.20);
        assert!(
            biased.distribution.median_ci95_low <= 1.0
                && 1.0 <= biased.distribution.median_ci95_high,
            "fixture must contain 1.0 or it does not exercise the retired clause"
        );
        let mut state = DecisionState::default();
        assert!(!validate_null_control(&biased, "candidate", &mut state));
        assert!(
            state
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.invalid_null_control"),
            "a biased null must quarantine: {:?}",
            state.reasons
        );

        // The tolerance is a boundary. Both fixtures sit a clear 1% inside and
        // outside it rather than exactly on it, so neither assertion depends on
        // f64 rounding through the subtraction.
        let drift = PERF_MAX_NULL_MEDIAN_DRIFT_PCT / 100.0;
        let at_edge = null_cell(1.0 + drift * 0.99, 0.90, 1.20);
        let past_edge = null_cell(1.0 + drift * 1.01, 0.90, 1.20);
        let mut state = DecisionState::default();
        assert!(validate_null_control(&at_edge, "candidate", &mut state));
        assert!(!validate_null_control(&past_edge, "candidate", &mut state));
    }

    fn test_profile() -> MachineProfileKey {
        MachineProfileKey::new(
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

    fn normalized_manifest_sha256() -> String {
        plan_binding(PerfGate::Qg2).normalized_perf_manifest_sha256
    }

    fn execution_provenance(gate: PerfGate) -> PerfExecutionProvenance {
        let plan = applicability_plan(gate);
        PerfExecutionProvenance {
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
        }
    }

    fn explicit_bootstrap(gate: PerfGate) -> PerfGateArtifact {
        PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate,
            applicability_plan: None,
            bench_elf_sha256: "unmeasured".to_owned(),
            machine_fingerprint: "unmeasured".to_owned(),
            execution: None,
            git_rev: "unmeasured".to_owned(),
            run_window: "unmeasured".to_owned(),
            run_id: "unmeasured".to_owned(),
            corpus_manifest_hash: ZERO_SHA256.to_owned(),
            manifest_sha256: normalized_manifest_sha256(),
            cells: Vec::new(),
            laws_attested: false,
        }
    }

    fn qg2_artifact(revision: &str, quill: f64, oracle: f64) -> PerfGateArtifact {
        let revision = if revision == "new" {
            "1".repeat(40)
        } else if revision == "old" {
            "0".repeat(40)
        } else {
            revision.to_owned()
        };
        let ratio = quill / oracle;
        PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg2,
            applicability_plan: Some(plan_binding(PerfGate::Qg2)),
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
            execution: Some(execution_provenance(PerfGate::Qg2)),
            git_rev: revision.clone(),
            run_window: "test-window".to_owned(),
            run_id: format!("{revision}-{quill}-{oracle}"),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: normalized_manifest_sha256(),
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

    fn qg5_target_artifact(ratio: f64) -> PerfGateArtifact {
        let mut artifact = qg2_artifact("new", 1.0, 1.0);
        artifact.gate = PerfGate::Qg5;
        artifact.applicability_plan = Some(plan_binding(PerfGate::Qg5));
        artifact.cells = vec![PerfCellResult {
            fixture: "compaction/xlarge/20pct".to_owned(),
            metric: "wall_clock_ms_quill_over_tantivy".to_owned(),
            engine: "paired_ab".to_owned(),
            unit: "ratio".to_owned(),
            distribution: distribution(ratio),
        }];
        artifact
    }

    fn qg5_target_decision(ratio: f64) -> DecisionState {
        let artifact = qg5_target_artifact(ratio);
        let cells = artifact
            .cells
            .iter()
            .map(|cell| (CellKey::from(cell), cell))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        evaluate_gate_targets(&artifact, &cells, None, true, false, &mut state);
        state
    }

    #[test]
    fn qg5_xlarge_rebaseline_rejects_a_fourfold_loss() {
        let state = qg5_target_decision(0.25);

        assert_eq!(state.decision(), PerfGateDecision::Block);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("QG-5 20% compaction")
        }));
    }

    #[test]
    fn qg5_xlarge_rebaseline_accepts_the_fivefold_threshold() {
        let state = qg5_target_decision(0.20);

        assert_eq!(state.decision(), PerfGateDecision::Allow);
    }

    fn profile_key(
        hardware: crate::HardwareClassId,
        profile: crate::ExecutionProfileId,
    ) -> MachineProfileKey {
        MachineProfileKey::new(hardware, profile).expect("registered hardware/profile pair")
    }

    fn applicability_plan_for(profile: MachineProfileKey, gate: PerfGate) -> PerfApplicabilityPlan {
        PerfMatrixSpec::complete()
            .applicability_plan(
                &MachineClassRegistry::frozen().expect("frozen machine registry"),
                profile,
                gate,
            )
            .expect("canonical applicability plan")
    }

    fn qg7_cell(
        fixture: &str,
        metric: &str,
        engine: &str,
        unit: &str,
        value: f64,
    ) -> PerfCellResult {
        PerfCellResult {
            fixture: fixture.to_owned(),
            metric: metric.to_owned(),
            engine: engine.to_owned(),
            unit: unit.to_owned(),
            distribution: distribution(value),
        }
    }

    /// Exactly the four rows `evaluate_qg7` reads for one corpus.
    fn qg7_corpus_cells(
        corpus: &str,
        rss_ratio: f64,
        bytes_ratio: f64,
        quill_off_bytes: f64,
        oracle_on_bytes: f64,
    ) -> Vec<PerfCellResult> {
        vec![
            qg7_cell(
                &format!("memory/{corpus}/positions_on"),
                "peak_rss_bytes_quill_over_tantivy",
                "paired_ab",
                "ratio",
                rss_ratio,
            ),
            qg7_cell(
                &format!("size/{corpus}/positions_on"),
                "index_bytes_per_document_quill_over_tantivy",
                "paired_ab",
                "ratio",
                bytes_ratio,
            ),
            qg7_cell(
                &format!("size/{corpus}/positions_off"),
                "index_bytes_per_document",
                "quill",
                "bytes/doc",
                quill_off_bytes,
            ),
            qg7_cell(
                &format!("size/{corpus}/positions_on"),
                "index_bytes_per_document",
                "tantivy",
                "bytes/doc",
                oracle_on_bytes,
            ),
        ]
    }

    fn qg7_target_decision(cells: Vec<PerfCellResult>) -> DecisionState {
        let mut artifact = qg2_artifact("new", 1.0, 1.0);
        artifact.gate = PerfGate::Qg7;
        artifact.cells = cells;
        let cells = artifact
            .cells
            .iter()
            .map(|cell| (CellKey::from(cell), cell))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        evaluate_gate_targets(&artifact, &cells, None, true, false, &mut state);
        state
    }

    /// The three published QG-7 bounds, pinned at and one step past each, plus
    /// the fail-closed refusal when the positions-off denominator is absent.
    ///
    /// `evaluate_qg7` reads medium and xlarge, and a row it cannot find is
    /// itself a quarantine, so every case supplies both corpora and varies only
    /// the medium rows under test.
    #[test]
    fn qg7_pins_its_three_thresholds_and_refuses_a_missing_denominator() {
        let passing = |corpus: &str| qg7_corpus_cells(corpus, 1.0, 1.15, 80.0, 100.0);

        let mut at_threshold = passing("medium");
        at_threshold.extend(passing("xlarge"));
        let state = qg7_target_decision(at_threshold);
        assert!(
            state.reasons.iter().all(|reason| {
                reason.code != "perf.ratchet.gate_target_missed"
                    && reason.code != "perf.ratchet.target_cell_missing"
            }),
            "QG-7 must admit exactly its published bounds (1.0, 1.15, 0.80): {:?}",
            state.reasons
        );

        for (label, medium, needle) in [
            (
                "rss",
                qg7_corpus_cells("medium", 1.000_001, 1.15, 80.0, 100.0),
                "RSS ratio",
            ),
            (
                "bytes_per_document",
                qg7_corpus_cells("medium", 1.0, 1.150_001, 80.0, 100.0),
                "bytes/doc ratio",
            ),
            (
                "positions_off",
                qg7_corpus_cells("medium", 1.0, 1.15, 80.001, 100.0),
                "positions-off/default-oracle ratio",
            ),
        ] {
            let mut cells = medium;
            cells.extend(passing("xlarge"));
            let state = qg7_target_decision(cells);
            assert!(
                state.reasons.iter().any(|reason| {
                    reason.code == "perf.ratchet.gate_target_missed"
                        && reason.message.contains(needle)
                }),
                "QG-7 {label} must reject one step past its published bound: {:?}",
                state.reasons
            );
        }

        // Drop the oracle positions-on bytes/doc row that the positions-off
        // ratio divides by. The gate must refuse the run rather than silently
        // skip the comparison it can no longer compute.
        let mut missing = passing("medium");
        missing.retain(|cell| {
            !(cell.engine == "tantivy" && cell.metric == "index_bytes_per_document")
        });
        missing.extend(passing("xlarge"));
        let state = qg7_target_decision(missing);
        assert!(
            state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.target_cell_missing"
                    && reason
                        .message
                        .contains("size/medium/positions_on/index_bytes_per_document/tantivy")
            }),
            "a missing QG-7 denominator must quarantine rather than skip: {:?}",
            state.reasons
        );
    }

    fn qg8_cell(threads: u32, docs_per_second: f64) -> PerfCellResult {
        PerfCellResult {
            fixture: format!("scaling/xlarge/{threads}/positions_on"),
            metric: "docs_per_second".to_owned(),
            engine: "quill".to_owned(),
            unit: "docs/s".to_owned(),
            distribution: distribution(docs_per_second),
        }
    }

    fn qg8_target_artifact(
        profile: MachineProfileKey,
        cells: Vec<PerfCellResult>,
    ) -> PerfGateArtifact {
        let plan = applicability_plan_for(profile, PerfGate::Qg8);
        let mut artifact = qg2_artifact("new", 1.0, 1.0);
        artifact.gate = PerfGate::Qg8;
        artifact.applicability_plan = Some(plan.binding);
        artifact.cells = cells;
        artifact
    }

    fn qg8_target_decision(
        profile: MachineProfileKey,
        cells: Vec<PerfCellResult>,
    ) -> DecisionState {
        let artifact = qg8_target_artifact(profile, cells);
        let cells = artifact
            .cells
            .iter()
            .map(|cell| (CellKey::from(cell), cell))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        evaluate_gate_targets(&artifact, &cells, None, true, false, &mut state);
        state
    }

    fn trj_physical() -> MachineProfileKey {
        profile_key(
            crate::HardwareClassId::TrjZen35995wx,
            crate::ExecutionProfileId::Physical64,
        )
    }

    fn m4_scheduler10() -> MachineProfileKey {
        profile_key(
            crate::HardwareClassId::M4Macos,
            crate::ExecutionProfileId::Scheduler10,
        )
    }

    #[test]
    fn qg8_trj_accepts_reviewed_16_vs_4_at_threshold() {
        let state = qg8_target_decision(
            trj_physical(),
            vec![qg8_cell(4, 1000.0), qg8_cell(16, 1800.0)],
        );
        assert!(
            state
                .reasons
                .iter()
                .all(|reason| reason.code != "perf.ratchet.gate_target_missed"),
            "TRJ 16/4 at 1.8x must not miss the QG-8 target: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg8_trj_rejects_16_vs_4_below_threshold() {
        let state = qg8_target_decision(
            trj_physical(),
            vec![qg8_cell(4, 1000.0), qg8_cell(16, 1799.0)],
        );
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("16-thread/4-thread")
        }));
    }

    #[test]
    fn qg8_trj_does_not_treat_8_as_the_wide_cell() {
        let state = qg8_target_decision(
            trj_physical(),
            vec![qg8_cell(4, 1000.0), qg8_cell(8, 5000.0)],
        );
        assert!(
            state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.target_cell_missing"
                    && reason.message.contains("scaling/xlarge/16/positions_on")
            }),
            "TRJ must still require the 16-thread cell: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg8_m4_accepts_reviewed_8_vs_4_at_threshold_without_width_16() {
        let state = qg8_target_decision(
            m4_scheduler10(),
            vec![qg8_cell(4, 1000.0), qg8_cell(8, 1800.0)],
        );
        assert!(
            state.reasons.iter().all(|reason| {
                reason.code != "perf.ratchet.gate_target_missed"
                    && reason.code != "perf.ratchet.target_cell_missing"
            }),
            "M4 8/4 at 1.8x must not require width 16 or 10: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg8_m4_rejects_8_vs_4_below_threshold() {
        let state = qg8_target_decision(
            m4_scheduler10(),
            vec![qg8_cell(4, 1000.0), qg8_cell(8, 1799.0)],
        );
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("8-thread/4-thread")
        }));
    }

    #[test]
    fn qg8_m4_rejects_missing_width_8_even_if_16_is_present() {
        let state = qg8_target_decision(
            m4_scheduler10(),
            vec![qg8_cell(4, 1000.0), qg8_cell(16, 5000.0)],
        );
        assert!(
            state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.target_cell_missing"
                    && reason.message.contains("scaling/xlarge/8/positions_on")
            }),
            "M4 must not treat a substituted 16-thread cell as the wide arm: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg8_m4_rejects_missing_width_4() {
        let state = qg8_target_decision(m4_scheduler10(), vec![qg8_cell(8, 1800.0)]);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.target_cell_missing"
                && reason.message.contains("scaling/xlarge/4/positions_on")
        }));
    }

    #[test]
    fn qg8_m4_does_not_invent_a_width_10_cell() {
        let state = qg8_target_decision(
            m4_scheduler10(),
            vec![qg8_cell(4, 1000.0), qg8_cell(8, 1800.0)],
        );
        assert!(
            state
                .reasons
                .iter()
                .all(|reason| { !reason.message.contains("scaling/xlarge/10/positions_on") }),
            "scheduler-10 must not invent a width-10 QG-8 cell: {:?}",
            state.reasons
        );
    }

    #[test]
    fn qg8_rejects_diagnostic_x86_profile() {
        let mut binding = plan_binding(PerfGate::Qg8);
        binding.profile = profile_key(
            crate::HardwareClassId::X86VpsOvh,
            crate::ExecutionProfileId::X86Diagnostic,
        );
        let mut artifact = qg8_target_artifact(
            trj_physical(),
            vec![qg8_cell(4, 1000.0), qg8_cell(16, 1800.0)],
        );
        artifact.applicability_plan = Some(binding);
        let cells = artifact
            .cells
            .iter()
            .map(|cell| (CellKey::from(cell), cell))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        evaluate_gate_targets(&artifact, &cells, None, true, false, &mut state);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("diagnostic x86")
        }));
    }

    #[test]
    fn qg8_rejects_m4_m5_profile_substitution() {
        assert!(
            MachineProfileKey::new(
                crate::HardwareClassId::M4Macos,
                crate::ExecutionProfileId::Scheduler14,
            )
            .is_err(),
            "M5 scheduler-14 must not bind onto m4-macos"
        );
        assert!(
            MachineProfileKey::new(
                crate::HardwareClassId::M5Macos,
                crate::ExecutionProfileId::Scheduler10,
            )
            .is_err(),
            "M4 scheduler-10 must not bind onto m5-macos"
        );
    }

    #[test]
    fn qg8_without_applicability_plan_fails_closed() {
        let mut artifact = qg8_target_artifact(
            trj_physical(),
            vec![qg8_cell(4, 1000.0), qg8_cell(16, 1800.0)],
        );
        artifact.applicability_plan = None;
        let cells = artifact
            .cells
            .iter()
            .map(|cell| (CellKey::from(cell), cell))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        evaluate_gate_targets(&artifact, &cells, None, true, false, &mut state);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.gate_target_missed"
                && reason.message.contains("applicability-plan")
        }));
    }

    fn qg2_current_pair(
        revision: &str,
        run_id: &str,
        quill: f64,
        oracle: f64,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let revision = if revision == "new" {
            "1".repeat(40)
        } else if revision == "old" {
            "0".repeat(40)
        } else {
            revision.to_owned()
        };
        let scope = crate::perf::perf_operation_scope(
            PerfGate::Qg2,
            "bulk/medium/1/positions_on",
            "docs_per_second",
        );
        let sample_provenance = PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "c".repeat(64),
            corpus_sha256: "a".repeat(64),
            input_identity: None,
            worker_id: TEST_MACHINE_FINGERPRINT.to_owned(),
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
                        qg6_sample_binding: None,
                        qg1_sample_binding: None,
                        tantivy_config_sha256: None,
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
                qg6_semantic_contract: None,
                cold_cache: None,
                concurrency_witness: None,
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
            applicability_plan: Some(plan_binding(PerfGate::Qg2)),
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
            execution: Some(execution_provenance(PerfGate::Qg2)),
            git_rev: revision.clone(),
            run_window: "test-window".to_owned(),
            run_id: run_id.to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: normalized_manifest_sha256(),
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
            plan_binding(PerfGate::Qg2),
            EvidencePolicy::predeclared(),
            EvidenceProvenance {
                run_id: run_id.to_owned(),
                run_window: "test-window".to_owned(),
                manifest_sha256: normalized_manifest_sha256(),
                build: BuildIdentity {
                    executable_sha256: "c".repeat(64),
                    git_revision: revision,
                    git_dirty: false,
                    worktree_state_sha256: None,
                    cargo_lock_sha256: Some("e".repeat(64)),
                    command_sha256: "f".repeat(64),
                    environment_sha256: Some("d".repeat(64)),
                    rustc_version: "rustc test".to_owned(),
                    target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                    build_profile: "release-perf".to_owned(),
                    cargo_features: vec!["perf-harness".to_owned()],
                },
                machine: MachineIdentity {
                    fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
                    os: "linux".to_owned(),
                    arch: "x86_64".to_owned(),
                    logical_cpus: 64,
                    execution: execution_provenance(PerfGate::Qg2),
                    cpu_governor: Some("performance".to_owned()),
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
        bind_test_evidence(&artifact, &mut evidence, run_id, &[]);
        (artifact, evidence)
    }

    fn qg6_fixture_authority_for_cell<const ROUNDS: usize>(
        query_class: crate::PerfQueryClass,
        k: usize,
        document_count: u64,
        searches_per_sample: usize,
        full_top_k_receipts: bool,
        schedule_seed: u64,
    ) -> Qg6ScheduleAuthority {
        let (identity, contract) = if full_top_k_receipts {
            qg6_test_fixture::contract_for_full_top_k(query_class, document_count, k)
        } else {
            qg6_test_fixture::contract_for(query_class, document_count, k)
        };
        Qg6ScheduleAuthority::for_experiment(
            crate::Qg6ExperimentIdentity {
                corpus_sha256: identity.prepared_corpus_sha256,
                query_manifest_sha256: identity.query_manifest_sha256,
                config_contract_sha256: identity.config_contract_sha256,
                document_count: contract.document_count,
                k: contract.k,
            },
            contract.groups.len(),
            ROUNDS,
            searches_per_sample,
            schedule_seed,
        )
        .expect("independently construct QG-6 fixture authority")
    }

    fn qg6_current_pair<const GROUPS: usize>(
        run_id: &str,
        group_ratios: [[f64; 3]; GROUPS],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_current_pair_with_null(run_id, group_ratios, [[1.0; 3]; GROUPS])
    }

    fn qg6_current_pair_with_null<const GROUPS: usize>(
        run_id: &str,
        effect_group_ratios: [[f64; 3]; GROUPS],
        null_group_ratios: [[f64; 3]; GROUPS],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let authority = qg6_fixture_authority_for_cell::<3>(
            crate::PerfQueryClass::Identifier,
            10,
            100_000,
            1,
            false,
            QG6_TEST_SCHEDULE_SEED,
        );
        qg6_current_pair_for_cell(
            run_id,
            effect_group_ratios,
            null_group_ratios,
            crate::PerfQueryClass::Identifier,
            10,
            100_000,
            "query/identifier/k10/100k",
            false,
            1,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            false,
            &authority,
        )
    }

    #[derive(Clone, Copy)]
    enum Qg6TestLeafProfile {
        Uniform,
        HiddenEffectTail,
        EffectP99 {
            baseline_numerator: u64,
            exceptional_numerator: u64,
        },
        QuillNullP99 {
            exceptional_numerator: u64,
            denominator: u64,
        },
    }

    fn qg6_test_leaf_latencies(
        profile: Qg6TestLeafProfile,
        comparison: crate::Qg6Comparison,
        sample: &PerfRawSample,
        parent_latency_ns: u64,
        searches_per_sample: usize,
    ) -> Vec<u64> {
        let uniform = || vec![parent_latency_ns; searches_per_sample];
        match profile {
            Qg6TestLeafProfile::Uniform => uniform(),
            Qg6TestLeafProfile::HiddenEffectTail if comparison == crate::Qg6Comparison::Effect => {
                let mut leaves = uniform();
                if sample.arm == PerfSampleArm::Treatment
                    && sample.group_id == Some(crate::QG6_QUERY_GROUP_IDS[0])
                {
                    *leaves.last_mut().expect("positive QG-6 leaf count") = parent_latency_ns * 100;
                }
                leaves
            }
            Qg6TestLeafProfile::EffectP99 {
                baseline_numerator,
                exceptional_numerator,
            } if comparison == crate::Qg6Comparison::Effect
                && sample.arm == PerfSampleArm::Treatment =>
            {
                let mut leaves = uniform();
                let exceptional_latency_ns = parent_latency_ns
                    .checked_mul(exceptional_numerator)
                    .expect("bounded QG-6 exceptional effect latency")
                    / baseline_numerator;
                for leaf in leaves.iter_mut().rev().take(2) {
                    *leaf = exceptional_latency_ns;
                }
                leaves
            }
            Qg6TestLeafProfile::QuillNullP99 {
                exceptional_numerator,
                denominator,
            } if comparison == crate::Qg6Comparison::QuillNull
                && sample.arm == PerfSampleArm::Treatment =>
            {
                let mut leaves = uniform();
                let exceptional_latency_ns = parent_latency_ns
                    .checked_mul(exceptional_numerator)
                    .expect("bounded QG-6 exceptional null latency")
                    / denominator;
                for leaf in leaves.iter_mut().rev().take(2) {
                    *leaf = exceptional_latency_ns;
                }
                leaves
            }
            _ => uniform(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qg6_current_pair_for_cell<const GROUPS: usize, const ROUNDS: usize>(
        run_id: &str,
        effect_group_ratios: [[f64; ROUNDS]; GROUPS],
        null_group_ratios: [[f64; ROUNDS]; GROUPS],
        query_class: crate::PerfQueryClass,
        k: usize,
        document_count: u64,
        fixture: &str,
        hidden_leaf_tail: bool,
        searches_per_sample: usize,
        bootstrap_resamples: usize,
        full_top_k_receipts: bool,
        authority: &Qg6ScheduleAuthority,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_current_pair_for_cell_with_leaf_profile(
            run_id,
            effect_group_ratios,
            null_group_ratios,
            query_class,
            k,
            document_count,
            fixture,
            if hidden_leaf_tail {
                Qg6TestLeafProfile::HiddenEffectTail
            } else {
                Qg6TestLeafProfile::Uniform
            },
            searches_per_sample,
            bootstrap_resamples,
            full_top_k_receipts,
            authority,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qg6_current_pair_for_cell_with_leaf_profile<const GROUPS: usize, const ROUNDS: usize>(
        run_id: &str,
        effect_group_ratios: [[f64; ROUNDS]; GROUPS],
        null_group_ratios: [[f64; ROUNDS]; GROUPS],
        query_class: crate::PerfQueryClass,
        k: usize,
        document_count: u64,
        fixture: &str,
        leaf_profile: Qg6TestLeafProfile,
        searches_per_sample: usize,
        bootstrap_resamples: usize,
        full_top_k_receipts: bool,
        authority: &Qg6ScheduleAuthority,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        assert!(GROUPS > 0 && ROUNDS >= 2 && searches_per_sample > 0);
        let scope = crate::perf::perf_operation_scope(PerfGate::Qg6, fixture, "latency_ms");
        let (input_identity, semantic_contract) = if full_top_k_receipts {
            qg6_test_fixture::contract_for_full_top_k(query_class, document_count, k)
        } else {
            qg6_test_fixture::contract_for(query_class, document_count, k)
        };
        let sample_provenance = PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: "c".repeat(64),
            corpus_sha256: "a".repeat(64),
            input_identity: Some(input_identity.clone()),
            worker_id: TEST_MACHINE_FINGERPRINT.to_owned(),
            build_profile: "release-perf".to_owned(),
        };
        let order =
            seeded_balanced_pair_order(crate::QG6_QUERY_GROUPS * ROUNDS, authority.schedule_seed)
                .expect("balanced QG-6 pair order");
        let stream = |group_ratios: &[[f64; ROUNDS]; GROUPS], sample_base: u64| {
            let mut samples = Vec::with_capacity(crate::QG6_QUERY_GROUPS * ROUNDS * 2);
            let mut ordinal = 0_usize;
            for (group_index, group_id) in crate::QG6_QUERY_GROUP_IDS.into_iter().enumerate() {
                let ratios = &group_ratios[group_index % GROUPS];
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
                            group_id: Some(group_id),
                            qg6_sample_binding: None,
                            qg1_sample_binding: None,
                            tantivy_config_sha256: None,
                        });
                    }
                    ordinal += 1;
                }
            }
            samples
        };
        let mut effect_samples = stream(&effect_group_ratios, 0);
        let mut null_samples = stream(&null_group_ratios, 100_000);
        let mut quill_null_samples = stream(&[[1.0; ROUNDS]; GROUPS], 200_000);
        qg6_test_fixture::attach_stream_against_schedule_authority_with_leaf_latencies(
            &mut effect_samples,
            crate::Qg6Comparison::Effect,
            authority,
            &input_identity,
            &semantic_contract,
            |sample, parent_latency_ns| {
                qg6_test_leaf_latencies(
                    leaf_profile,
                    crate::Qg6Comparison::Effect,
                    sample,
                    parent_latency_ns,
                    searches_per_sample,
                )
            },
        );
        for (samples, comparison) in [
            (&mut null_samples[..], crate::Qg6Comparison::TantivyNull),
            (&mut quill_null_samples[..], crate::Qg6Comparison::QuillNull),
        ] {
            qg6_test_fixture::attach_stream_against_schedule_authority_with_leaf_latencies(
                samples,
                comparison,
                authority,
                &input_identity,
                &semantic_contract,
                |sample, parent_latency_ns| {
                    qg6_test_leaf_latencies(
                        leaf_profile,
                        comparison,
                        sample,
                        parent_latency_ns,
                        searches_per_sample,
                    )
                },
            );
        }
        let mut estimator_config = PairedEstimatorConfig::predeclared(QG6_TEST_SCHEDULE_SEED);
        estimator_config.bootstrap_resamples = bootstrap_resamples;
        let paired = estimate_paired_experiment(&effect_samples, &null_samples, &estimator_config)
            .expect("QG-6 paired evidence");
        let quill_null_projection =
            projected_ratio_distribution(&quill_null_samples).expect("QG-6 Quill-null projection");
        let protocol = crate::Qg6FormalProtocolEvidence::new_against_authority(
            &paired,
            quill_null_samples,
            authority,
            &input_identity,
            &semantic_contract,
        )
        .expect("QG-6 ratchet fixture formal protocol");
        let mut cell = EvidenceCell::evaluate(
            EvidenceCellSpec {
                gate: PerfGate::Qg6,
                fixture: fixture.to_owned(),
                metric: "latency_ms".to_owned(),
                unit: "ms".to_owned(),
                role: EvidenceRole::Required,
                input_identity: Some(input_identity),
                qg6_semantic_contract: Some(semantic_contract),
                cold_cache: None,
                concurrency_witness: None,
            },
            paired,
            &EvidencePolicy::predeclared(),
        )
        .expect("QG-6 evidence cell");
        cell.attach_qg6_formal_protocol_against_authority(
            protocol,
            &EvidencePolicy::predeclared(),
            authority,
        )
        .expect("attach QG-6 ratchet fixture formal protocol");
        let paired = match &cell.body {
            EvidenceCellBody::Paired { paired, .. } => paired,
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
        };
        let leaf_distributions =
            crate::project_qg6_effect_leaf_distributions(&paired.effect_samples, &paired.config)
                .expect("QG-6 effect-leaf projection");
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg6,
            applicability_plan: Some(plan_binding(PerfGate::Qg6)),
            bench_elf_sha256: "c".repeat(64),
            machine_fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
            execution: Some(execution_provenance(PerfGate::Qg6)),
            git_rev: "1".repeat(40),
            run_window: "test-window".to_owned(),
            run_id: run_id.to_owned(),
            corpus_manifest_hash: "a".repeat(64),
            manifest_sha256: normalized_manifest_sha256(),
            cells: vec![
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms".to_owned(),
                    engine: "quill".to_owned(),
                    unit: "ms".to_owned(),
                    distribution: leaf_distributions.treatment,
                },
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms".to_owned(),
                    engine: "tantivy".to_owned(),
                    unit: "ms".to_owned(),
                    distribution: leaf_distributions.control,
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
                PerfCellResult {
                    fixture: fixture.to_owned(),
                    metric: "latency_ms_quill_over_quill".to_owned(),
                    engine: "paired_null_quill".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: quill_null_projection,
                },
            ],
            laws_attested: true,
        };
        let mut evidence = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            plan_binding(PerfGate::Qg6),
            EvidencePolicy::predeclared(),
            EvidenceProvenance {
                run_id: run_id.to_owned(),
                run_window: "test-window".to_owned(),
                manifest_sha256: normalized_manifest_sha256(),
                build: BuildIdentity {
                    executable_sha256: "c".repeat(64),
                    git_revision: "1".repeat(40),
                    git_dirty: false,
                    worktree_state_sha256: None,
                    cargo_lock_sha256: Some("e".repeat(64)),
                    command_sha256: "f".repeat(64),
                    environment_sha256: Some("d".repeat(64)),
                    rustc_version: "rustc test".to_owned(),
                    target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                    build_profile: "release-perf".to_owned(),
                    cargo_features: vec!["perf-harness".to_owned()],
                },
                machine: MachineIdentity {
                    fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
                    os: "linux".to_owned(),
                    arch: "x86_64".to_owned(),
                    logical_cpus: 64,
                    execution: execution_provenance(PerfGate::Qg6),
                    cpu_governor: Some("performance".to_owned()),
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
                    document_count,
                    content_bytes: None,
                    generator_seed: 42,
                    generator_revision: "test-v1".to_owned(),
                },
            },
            vec![cell],
        )
        .expect("QG-6 evidence artifact");
        bind_test_evidence(&artifact, &mut evidence, run_id, &[authority]);
        (artifact, evidence)
    }

    fn qg6_complete_pair<const GROUPS: usize>(
        run_id: &str,
        group_ratios: [[f64; 3]; GROUPS],
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_complete_pair_with_shape(
            run_id,
            group_ratios,
            1,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            false,
        )
    }

    fn qg6_complete_pair_with_leaf_profile<const GROUPS: usize>(
        run_id: &str,
        group_ratios: [[f64; 3]; GROUPS],
        leaf_profile: Qg6TestLeafProfile,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_complete_pair_with_shape_and_seed_and_leaf_profile(
            run_id,
            group_ratios,
            128,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            false,
            QG6_TEST_SCHEDULE_SEED,
            leaf_profile,
        )
    }

    fn qg6_fixture_authorities_for_shape<const ROUNDS: usize>(
        searches_per_sample: usize,
        full_top_k_receipts: bool,
    ) -> Vec<Qg6ScheduleAuthority> {
        qg6_fixture_authorities_for_shape_and_seed::<ROUNDS>(
            searches_per_sample,
            full_top_k_receipts,
            QG6_TEST_SCHEDULE_SEED,
        )
    }

    fn qg6_fixture_authorities_for_shape_and_seed<const ROUNDS: usize>(
        searches_per_sample: usize,
        full_top_k_receipts: bool,
        schedule_seed: u64,
    ) -> Vec<Qg6ScheduleAuthority> {
        PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg6)
            .into_iter()
            .map(|spec| {
                let query_class = spec.query_class.expect("QG-6 query class");
                let k = spec.k.expect("QG-6 k");
                let document_count = spec.document_count.expect("QG-6 document count");
                qg6_fixture_authority_for_cell::<ROUNDS>(
                    query_class,
                    k,
                    document_count,
                    searches_per_sample,
                    full_top_k_receipts,
                    schedule_seed,
                )
            })
            .collect()
    }

    fn qg6_default_fixture_authorities() -> Vec<Qg6ScheduleAuthority> {
        qg6_fixture_authorities_for_shape::<3>(1, false)
    }

    fn qg6_complete_pair_with_shape<const GROUPS: usize, const ROUNDS: usize>(
        run_id: &str,
        group_ratios: [[f64; ROUNDS]; GROUPS],
        searches_per_sample: usize,
        bootstrap_resamples: usize,
        full_top_k_receipts: bool,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_complete_pair_with_shape_and_seed(
            run_id,
            group_ratios,
            searches_per_sample,
            bootstrap_resamples,
            full_top_k_receipts,
            QG6_TEST_SCHEDULE_SEED,
        )
    }

    fn qg6_complete_pair_with_shape_and_seed<const GROUPS: usize, const ROUNDS: usize>(
        run_id: &str,
        group_ratios: [[f64; ROUNDS]; GROUPS],
        searches_per_sample: usize,
        bootstrap_resamples: usize,
        full_top_k_receipts: bool,
        schedule_seed: u64,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        qg6_complete_pair_with_shape_and_seed_and_leaf_profile(
            run_id,
            group_ratios,
            searches_per_sample,
            bootstrap_resamples,
            full_top_k_receipts,
            schedule_seed,
            Qg6TestLeafProfile::Uniform,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qg6_complete_pair_with_shape_and_seed_and_leaf_profile<
        const GROUPS: usize,
        const ROUNDS: usize,
    >(
        run_id: &str,
        group_ratios: [[f64; ROUNDS]; GROUPS],
        searches_per_sample: usize,
        bootstrap_resamples: usize,
        full_top_k_receipts: bool,
        schedule_seed: u64,
        leaf_profile: Qg6TestLeafProfile,
    ) -> (PerfGateArtifact, PerfEvidenceArtifact) {
        let mut rows = Vec::new();
        let mut cells = Vec::new();
        let authorities = qg6_fixture_authorities_for_shape_and_seed::<ROUNDS>(
            searches_per_sample,
            full_top_k_receipts,
            schedule_seed,
        );
        let mut artifact_template = None;
        let mut evidence_template = None;
        for (spec, authority) in PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg6)
            .into_iter()
            .zip(&authorities)
        {
            let query_class = spec.query_class.expect("QG-6 query class");
            let k = spec.k.expect("QG-6 k");
            let document_count = spec.document_count.expect("QG-6 document count");
            let (cell_artifact, mut cell_evidence) = qg6_current_pair_for_cell_with_leaf_profile(
                run_id,
                group_ratios,
                [[1.0; ROUNDS]; GROUPS],
                query_class,
                k,
                document_count,
                &spec.fixture,
                leaf_profile,
                searches_per_sample,
                bootstrap_resamples,
                full_top_k_receipts,
                authority,
            );
            rows.extend(cell_artifact.cells.iter().cloned());
            cells.push(cell_evidence.cells.remove(0));
            artifact_template.get_or_insert(cell_artifact);
            evidence_template.get_or_insert(cell_evidence);
        }
        let mut artifact = artifact_template.expect("QG-6 matrix is non-empty");
        artifact.cells = rows;
        let template_evidence = evidence_template.expect("QG-6 matrix is non-empty");
        let mut evidence = PerfEvidenceArtifact::assemble(
            PerfGate::Qg6,
            plan_binding(PerfGate::Qg6),
            template_evidence.policy,
            template_evidence.provenance,
            cells,
        )
        .expect("complete QG-6 evidence");
        let authority_refs = authorities.iter().collect::<Vec<_>>();
        bind_test_evidence(&artifact, &mut evidence, run_id, &authority_refs);
        (artifact, evidence)
    }

    fn mutate_qg6_prepared_input(
        evidence: &mut PerfEvidenceArtifact,
        field: &str,
        replacement: &str,
    ) {
        let cell = evidence.cells.first().expect("QG-6 evidence cell");
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
        let cell = evidence.cells.first_mut().expect("QG-6 evidence cell");
        cell.spec.input_identity = Some(identity.clone());
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
    }

    fn evaluate(
        baseline: &PerfGateArtifact,
        candidate: &PerfGateArtifact,
        rerun: Option<&PerfGateArtifact>,
        activated: bool,
        mode: PerfRatchetMode,
    ) -> PerfRatchetEvaluation {
        evaluate_perf_ratchet_inner(
            PerfRatchetRequest {
                baseline: Some(baseline),
                baseline_evidence: None,
                candidate,
                rerun,
                candidate_evidence: None,
                rerun_evidence: None,
                expected_machine_profile: None,
                candidate_runner_identity: None,
                rerun_runner_identity: None,
                gate_activated: activated,
                mode,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg1AuthoritySets::empty(),
            PerfRatchetQg6AuthoritySets::empty(),
            DecisionState::default(),
            false,
        )
    }

    fn evaluate_with_current(
        baseline: &PerfGateArtifact,
        baseline_evidence: Option<&PerfEvidenceArtifact>,
        candidate: &PerfGateArtifact,
        rerun: Option<&PerfGateArtifact>,
        candidate_evidence: Option<&PerfEvidenceArtifact>,
        rerun_evidence: Option<&PerfEvidenceArtifact>,
        qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
    ) -> PerfRatchetEvaluation {
        let expected_machine_profile = candidate_evidence
            .and_then(|evidence| evidence.machine_class.identity())
            .map(crate::VerifiedRunnerIdentity::profile);
        evaluate_perf_ratchet_against_authorities(
            PerfRatchetRequest {
                baseline: Some(baseline),
                baseline_evidence,
                candidate,
                rerun,
                candidate_evidence,
                rerun_evidence,
                expected_machine_profile,
                candidate_runner_identity: candidate_evidence
                    .and_then(|evidence| evidence.machine_class.identity()),
                rerun_runner_identity: rerun_evidence
                    .and_then(|evidence| evidence.machine_class.identity()),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg1AuthoritySets::empty(),
            qg6_authority_sets,
        )
    }

    fn assert_qg6_target_passes(
        artifact: &PerfGateArtifact,
        evidence: &PerfEvidenceArtifact,
        role: &str,
        qg6_authority_refs: &[&Qg6ScheduleAuthority],
    ) {
        evidence
            .verify_integrity_against_authorities(&[], qg6_authority_refs)
            .expect("QG-6 test evidence must be bound and resealed against its schedule authority");

        let mut state = DecisionState::default();
        let cells = validate_artifact(
            artifact,
            PerfGate::Qg6,
            &normalized_manifest_sha256(),
            role,
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact,
            cells: &cells,
            activated: true,
            observe_only: false,
            state: &mut state,
        };
        evaluate_qg6(&mut target, Some(evidence));
        assert!(
            !state.fatal && !state.blocked && !state.quarantined,
            "{role} must independently clear every QG-6 target: {:?}",
            state.reasons
        );
    }

    fn evaluate_verified_promotion_request(
        request: PerfRatchetRequest<'_>,
        qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
    ) -> PerfRatchetEvaluation {
        evaluate_perf_ratchet_against_authorities(
            request,
            PerfRatchetQg1AuthoritySets::empty(),
            qg6_authority_sets,
        )
    }

    fn evaluate_verified_promotion(
        baseline: &PerfGateArtifact,
        baseline_evidence: &PerfEvidenceArtifact,
        candidate: &PerfGateArtifact,
        candidate_evidence: &PerfEvidenceArtifact,
        rerun: &PerfGateArtifact,
        rerun_evidence: &PerfEvidenceArtifact,
        expected_machine_profile: MachineProfileKey,
        qg6_authority_sets: PerfRatchetQg6AuthoritySets<'_>,
    ) -> PerfRatchetEvaluation {
        evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(baseline),
                baseline_evidence: Some(baseline_evidence),
                candidate,
                rerun: Some(rerun),
                candidate_evidence: Some(candidate_evidence),
                rerun_evidence: Some(rerun_evidence),
                expected_machine_profile: Some(expected_machine_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            qg6_authority_sets,
        )
    }

    /// bd-h4sqj: the whole point of the quarantine. This artifact is INTACT —
    /// it was assembled and bound by the same helper every other promotion test
    /// uses, so it verifies, seals, and recomputes. Only its measured revision
    /// belongs to the structurally invalid 193d2e3f sweep, and that alone must
    /// block promotion.
    #[test]
    fn evidence_measured_at_a_quarantined_revision_blocks_promotion() {
        let (_artifact, mut evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        evidence.provenance.build.git_revision =
            "193d2e3fa1b2c3d4e5f60718293a4b5c6d7e8f90".to_owned();

        let mut state = DecisionState::default();
        reject_quarantined_revision("baseline", &evidence, &mut state);

        assert_eq!(
            state.decision(),
            PerfGateDecision::Block,
            "a quarantined revision must BLOCK, not merely quarantine: Quarantine means \
             'rerun may help', and re-measuring a structurally invalid sweep reproduces the \
             same invalid shape"
        );
        assert!(
            state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.quarantined_revision"
                    && reason.message.contains("baseline")
            }),
            "the refusal must name its role and reason: {:#?}",
            state.reasons
        );
    }

    /// The screen must DISCRIMINATE. Asserting only the refusal above would
    /// still pass if the quarantine degenerated into blocking every artifact,
    /// which would take the whole ratchet offline while looking like coverage.
    #[test]
    fn evidence_from_an_unquarantined_revision_is_not_blocked_by_the_quarantine_screen() {
        let (_artifact, evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let mut state = DecisionState::default();
        reject_quarantined_revision("baseline", &evidence, &mut state);

        assert_eq!(
            state.decision(),
            PerfGateDecision::Allow,
            "an unquarantined revision must pass the quarantine screen untouched: {:#?}",
            state.reasons
        );
    }

    /// Guards the `include_str!` wiring itself. If the register path breaks, the
    /// file is emptied, or a record is dropped, the screen would silently admit
    /// the sweep it exists to refuse — so the embedded bytes are asserted to
    /// still cover every named revision.
    #[test]
    fn the_embedded_quarantine_register_covers_every_named_sweep_revision() {
        let register =
            crate::perf_evidence::PerfQuarantineRegister::from_jsonl(EMBEDDED_QUARANTINE_REGISTER)
                .expect("the embedded quarantine register must parse");
        for revision in [
            "193d2e3fa1b2c3d4e5f60718293a4b5c6d7e8f90",
            "544ffeb0112233445566778899aabbccddeeff00",
            "e0dc6ba3ffeeddccbbaa99887766554433221100",
        ] {
            assert!(
                register.quarantine_of(revision).is_some(),
                "{revision} must remain quarantined by the embedded register"
            );
        }
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
    fn complete_verified_qg6_matrix_reaches_joint_tail_promotion() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();
        let retained_authority_sets = PerfRatchetQg6AuthoritySets {
            baseline: &retained_authority_refs,
            candidate: &retained_authority_refs,
            rerun: &retained_authority_refs,
        };

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            retained_authority_sets,
        );

        assert_eq!(result.decision, PerfGateDecision::Allow, "{result:#?}");

        let mut tampered_evidence = candidate_evidence.clone();
        let EvidenceCellBody::Paired { paired, .. } = &mut tampered_evidence.cells[0].body else {
            unreachable!("QG-6 must be paired");
        };
        paired.effect_samples[0]
            .qg6_sample_binding
            .as_mut()
            .expect("QG-6 semantic binding")
            .result_sequence_sha256 = "0".repeat(64);
        let tampered = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &tampered_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            retained_authority_sets,
        );
        assert_eq!(
            tampered.decision,
            PerfGateDecision::Block,
            "tampered QG-6 evidence escaped authenticated replay: {tampered:#?}"
        );
        assert!(
            tampered
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.machine_evidence_integrity_failed" })
        );
    }

    #[test]
    fn qg6_fixture_evaluator_requires_explicit_external_authorities() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let missing = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets::empty(),
        );
        assert_eq!(missing.decision, PerfGateDecision::Block, "{missing:#?}");
        assert!(missing.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.machine_evidence_integrity_failed"
                && reason.message.contains("independently retained")
        }));
        assert!(missing.comparisons.is_empty());

        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();
        let retained = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );
        assert_eq!(retained.decision, PerfGateDecision::Allow, "{retained:#?}");
    }

    #[test]
    fn qg6_ratchet_refuses_a_foreign_external_schedule_authority() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let baseline_authorities = retained_authorities.iter().collect::<Vec<_>>();
        let mut candidate_authorities = retained_authorities.iter().collect::<Vec<_>>();
        let rerun_authorities = retained_authorities.iter().collect::<Vec<_>>();
        let original = candidate_authorities[0];
        let foreign = Qg6ScheduleAuthority::for_experiment(
            original.identity.clone(),
            original.query_count,
            original.rounds_per_query,
            original.searches_per_sample,
            original.schedule_seed ^ 1,
        )
        .expect("foreign but valid QG-6 authority");
        candidate_authorities[0] = &foreign;

        let result = evaluate_perf_ratchet_against_authorities(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg1AuthoritySets::empty(),
            PerfRatchetQg6AuthoritySets {
                baseline: &baseline_authorities,
                candidate: &candidate_authorities,
                rerun: &rerun_authorities,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block, "{result:#?}");
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.machine_evidence_integrity_failed" })
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn qg6_fixture_evaluator_refuses_self_authenticated_foreign_authorities() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let foreign_seed = QG6_TEST_SCHEDULE_SEED ^ 1;
        let (candidate, candidate_evidence) = qg6_complete_pair_with_shape_and_seed(
            "candidate",
            ratios,
            1,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            false,
            foreign_seed,
        );
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let foreign_authorities =
            qg6_fixture_authorities_for_shape_and_seed::<3>(1, false, foreign_seed);
        let foreign_authority_refs = foreign_authorities.iter().collect::<Vec<_>>();
        candidate_evidence
            .verify_integrity_against_authorities(&[], &foreign_authority_refs)
            .expect("foreign-seeded fixture must be internally self-consistent");
        for (candidate_cell, rerun_cell) in
            candidate_evidence.cells.iter().zip(&rerun_evidence.cells)
        {
            let EvidenceCellBody::Paired {
                paired: candidate_paired,
                ..
            } = &candidate_cell.body
            else {
                panic!("QG-6 candidate cell must be paired");
            };
            let EvidenceCellBody::Paired {
                paired: rerun_paired,
                ..
            } = &rerun_cell.body
            else {
                panic!("QG-6 rerun cell must be paired");
            };
            assert_eq!(candidate_paired.config, rerun_paired.config);
        }
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block, "{result:#?}");
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.machine_evidence_integrity_failed"
                && reason.message.contains("independently retained")
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn complete_qg6_production_cardinality_wire_fits_assembly_cap() {
        let (_, evidence) = qg6_complete_pair_with_shape(
            "qg6-production-wire-size",
            [[1.0; 2]; 4],
            crate::QG6_TIMED_SEARCHES_PER_SAMPLE,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            true,
        );
        assert_eq!(evidence.cells.len(), 20, "canonical QG-6 matrix drifted");
        let expected_rounds = crate::PERF_MIN_RUNS
            .div_ceil(crate::QG6_QUERY_GROUPS)
            .max(EvidencePolicy::predeclared().min_group_pairs);
        assert_eq!(
            expected_rounds, 2,
            "production QG-6 round cardinality changed; update the wire proof"
        );
        let mut sample_count = 0_usize;
        let mut leaf_count = 0_usize;
        for cell in &evidence.cells {
            let EvidenceCellBody::Paired {
                paired,
                qg6_protocol: Some(protocol),
                ..
            } = &cell.body
            else {
                panic!("every QG-6 cell must carry formal six-arm evidence");
            };
            assert_eq!(
                protocol.schedule_authority.rounds_per_query,
                expected_rounds
            );
            assert_eq!(
                protocol.schedule_authority.searches_per_sample,
                crate::QG6_TIMED_SEARCHES_PER_SAMPLE,
            );
            assert_eq!(
                paired.config.bootstrap_resamples,
                crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            );
            assert_eq!(
                protocol.joint_tail.bootstrap_resamples,
                crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            );
            for sample in paired
                .effect_samples
                .iter()
                .chain(&paired.null_samples)
                .chain(&protocol.quill_null_samples)
            {
                sample_count += 1;
                leaf_count += sample
                    .qg6_sample_binding
                    .as_ref()
                    .expect("production-cardinality timed sample")
                    .timed_sample
                    .timing_leaves
                    .len();
            }
        }
        assert_eq!(
            sample_count,
            20 * crate::QG6_QUERY_GROUPS * expected_rounds * 6,
        );
        assert_eq!(
            leaf_count,
            20 * crate::QG6_QUERY_GROUPS
                * expected_rounds
                * 6
                * crate::QG6_TIMED_SEARCHES_PER_SAMPLE,
        );
        let expected_hit_count = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg6)
            .into_iter()
            .map(|spec| {
                crate::QG6_QUERY_GROUPS * crate::Qg6ArmRole::ALL.len() * spec.k.expect("QG-6 k")
            })
            .sum::<usize>();
        let observed_hit_count = evidence
            .cells
            .iter()
            .map(|cell| {
                cell.spec
                    .qg6_semantic_contract
                    .as_ref()
                    .expect("production-cardinality semantic contract")
                    .groups
                    .iter()
                    .map(|group| {
                        crate::Qg6ArmRole::ALL
                            .into_iter()
                            .map(|role| group.roles.get(role).ordered_hits.len())
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();
        assert_eq!(observed_hit_count, expected_hit_count);

        let bytes = serde_json::to_vec_pretty(&evidence)
            .expect("serialize complete production-cardinality QG-6 evidence");
        let qg6_authorities =
            qg6_fixture_authorities_for_shape::<2>(crate::QG6_TIMED_SEARCHES_PER_SAMPLE, true);
        let qg6_authority_refs = qg6_authorities.iter().collect::<Vec<_>>();
        let reparsed = PerfEvidenceArtifact::from_verified_slice_against_authorities(
            &bytes,
            &[],
            &qg6_authority_refs,
        )
        .expect("reparse complete production-cardinality QG-6 evidence");
        assert_eq!(reparsed, evidence, "compact QG-6 wire changed evidence");
        const MIN_WIRE_HEADROOM_PERCENT: usize = 5;
        let cap = crate::perf_assembly::PERF_ASSEMBLY_MAX_ARTIFACT_BYTES;
        let maximum_with_headroom = cap * (100 - MIN_WIRE_HEADROOM_PERCENT) / 100;
        eprintln!(
            "complete production-cardinality QG-6 evidence wire: {} bytes (cap {}, required maximum {})",
            bytes.len(),
            cap,
            maximum_with_headroom,
        );
        assert!(
            bytes.len() <= maximum_with_headroom,
            "complete compact QG-6 evidence is {} bytes, cap is {} bytes, required headroom is {}%",
            bytes.len(),
            cap,
            MIN_WIRE_HEADROOM_PERCENT,
        );
    }

    #[test]
    fn qg6_bad_joint_p50_interval_blocks_promotion() {
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[0.70; 3]; 4]);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", [[0.70; 3]; 4]);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block, "{result:#?}");
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.gate_target_missed"),
            "bad p50 interval did not reach the gate target: {result:#?}"
        );
    }

    #[test]
    fn qg6_bad_joint_p99_upper_interval_blocks_promotion() {
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.01; 3]; 4]);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", [[1.01; 3]; 4]);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block, "{result:#?}");
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.gate_target_missed"),
            "bad p99 interval did not reach the gate target: {result:#?}"
        );
    }

    #[test]
    fn qg6_joint_p99_interval_straddling_parity_quarantines() {
        let baseline_ratios = [[1.0; 3]; crate::QG6_QUERY_GROUPS];
        let mut straddling_ratios = [[0.97; 3]; crate::QG6_QUERY_GROUPS];
        straddling_ratios[crate::QG6_QUERY_GROUPS - 1] = [1.04; 3];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", baseline_ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", straddling_ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", straddling_ratios);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine, "{result:#?}");
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.gate_target_ci_inconclusive"
                    && reason.message.contains("joint true-leaf p99")
            }),
            "straddling p99 interval did not remain inconclusive: {result:#?}"
        );
        assert!(
            !result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.gate_target_missed"),
            "straddling p99 interval was misclassified as a hard loss: {result:#?}"
        );
    }

    #[test]
    fn qg6_baseline_regression_and_joint_tail_failure_are_decision_inputs() {
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.10; 3]; 4]);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", [[1.10; 3]; 4]);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert!(
            result.comparisons.iter().any(|comparison| {
                comparison.engine == "quill" && comparison.threshold_exceeded
            }),
            "fixture did not exercise the numeric regression path: {result:#?}"
        );
        assert_eq!(result.decision, PerfGateDecision::Block, "{result:#?}");
    }

    #[test]
    fn threshold_execution_projection_cannot_override_sealed_evidence() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (mut candidate, mut candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        candidate
            .execution
            .as_mut()
            .expect("current threshold projection")
            .host_identity = "caller-forged-host".to_owned();
        let candidate_qg6_authorities = qg6_default_fixture_authorities();
        let candidate_qg6_authority_refs = candidate_qg6_authorities.iter().collect::<Vec<_>>();
        bind_test_evidence(
            &candidate,
            &mut candidate_evidence,
            "candidate-forged-projection",
            &candidate_qg6_authority_refs,
        );
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            expected_profile,
            PerfRatchetQg6AuthoritySets {
                baseline: &candidate_qg6_authority_refs,
                candidate: &candidate_qg6_authority_refs,
                rerun: &candidate_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.execution_projection_evidence_mismatch"
        }));
    }

    #[test]
    fn sealed_execution_projection_cannot_override_verified_receipt_topology() {
        let ratios = [[1.0; 3]; 4];
        let (mut candidate, mut candidate_evidence) = qg6_complete_pair("candidate", ratios);
        candidate
            .execution
            .as_mut()
            .expect("current threshold projection")
            .physical_cores = 63;
        candidate_evidence
            .provenance
            .machine
            .execution
            .physical_cores = 63;
        let candidate_qg6_authorities = qg6_default_fixture_authorities();
        let candidate_qg6_authority_refs = candidate_qg6_authorities.iter().collect::<Vec<_>>();
        let error = try_bind_test_evidence(
            &candidate,
            &mut candidate_evidence,
            "candidate-forged-topology",
            &candidate_qg6_authority_refs,
        )
        .expect_err("forged topology must fail before ratchet evaluation");
        assert!(matches!(
            error,
            EvidenceArtifactError::InvalidProvenance { ref reason }
                if reason.contains("topology") && reason.contains("verified runner hardware")
        ));
    }

    #[test]
    fn sealed_execution_projection_must_equal_verified_profile_capacity() {
        let (mut artifact, mut evidence) =
            qg2_current_pair("new", "candidate-capacity", 160.0, 100.0);
        let identity = evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .clone();
        artifact
            .execution
            .as_mut()
            .expect("current threshold projection")
            .process_available_threads = 63;
        evidence.provenance.machine.logical_cpus = 63;
        evidence
            .provenance
            .machine
            .execution
            .process_available_threads = 63;
        let plan = applicability_plan(PerfGate::Qg2);
        let mut state = DecisionState::default();

        validate_execution_projection_binding(
            "candidate",
            &artifact,
            &evidence,
            &identity,
            &plan,
            &mut state,
        );

        assert_eq!(state.decision(), PerfGateDecision::Block);
        assert!(state.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.execution_projection_capacity_mismatch"
        }));
    }

    #[test]
    fn expected_profile_cannot_relabel_verified_evidence() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion(
            &baseline,
            &baseline_evidence,
            &candidate,
            &candidate_evidence,
            &rerun,
            &rerun_evidence,
            MachineProfileKey::new(
                crate::HardwareClassId::TrjZen35995wx,
                crate::ExecutionProfileId::Smt2_128,
            )
            .expect("canonical mismatched profile"),
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.machine_profile_mismatch"
                && reason.message.contains("Smt2_128")
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn legacy_baseline_cannot_mix_with_current_promotion_evidence() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, _) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: None,
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.machine_identity_incomplete"
                && reason.message.contains("baseline")
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn resealed_stale_registry_binding_is_rejected_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, mut candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let external_candidate = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .clone();
        let expected_profile = external_candidate.profile();
        let mut binding =
            serde_json::to_value(&candidate_evidence.machine_class).expect("binding JSON");
        binding["identity"]["canonicalization"]["registry_sha256"] =
            serde_json::Value::String("0".repeat(64));
        candidate_evidence.machine_class =
            serde_json::from_value(binding).expect("stale binding JSON");
        reseal_evidence_without_verification(&mut candidate_evidence);
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: Some(&external_candidate),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.machine_evidence_integrity_failed"
                || reason.code == "perf.ratchet.runner_receipt_artifact_mismatch"
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn mixed_applicability_plan_hashes_are_rejected_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (mut rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        rerun
            .applicability_plan
            .as_mut()
            .expect("measured plan")
            .applicability_plan_sha256 = "0".repeat(64);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.threshold_verified_reload_failed"
                && reason.message.contains("rerun")
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn measured_threshold_without_applicability_plan_is_rejected_before_comparison() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (mut candidate, candidate_evidence) =
            qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        candidate.applicability_plan = None;

        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(test_profile()),
            candidate_runner_identity: candidate_evidence.machine_class.identity(),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.measured_applicability_plan_missing"
            })
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn applicability_plan_profile_mutation_is_rejected_before_comparison() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (mut candidate, candidate_evidence) =
            qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        candidate
            .applicability_plan
            .as_mut()
            .expect("measured plan")
            .profile = MachineProfileKey::new(
            crate::HardwareClassId::TrjZen35995wx,
            crate::ExecutionProfileId::Smt2_128,
        )
        .expect("alternate canonical profile");

        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(test_profile()),
            candidate_runner_identity: candidate_evidence.machine_class.identity(),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.applicability_plan_binding_mismatch"
            })
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn evidence_applicability_binding_mutation_is_rejected_before_comparison() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, mut candidate_evidence) =
            qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        candidate_evidence
            .applicability_plan
            .applicability_plan_sha256 = "0".repeat(64);
        reseal_evidence_without_verification(&mut candidate_evidence);

        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(test_profile()),
            candidate_runner_identity: candidate_evidence.machine_class.identity(),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.evidence_applicability_plan_mismatch"
            })
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn runner_capacity_binding_mutation_is_rejected_before_comparison() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        let mut identity_value = serde_json::to_value(
            candidate_evidence
                .machine_class
                .identity()
                .expect("candidate identity"),
        )
        .expect("candidate identity JSON");
        identity_value["execution_capacity"] = Value::from(63_u64);
        let mutated_identity: VerifiedRunnerIdentity =
            serde_json::from_value(identity_value).expect("capacity-mutated identity");

        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(test_profile()),
            candidate_runner_identity: Some(&mutated_identity),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.runner_receipt_rejected")
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn runner_gate_maximum_must_equal_applicability_plan_maximum() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        let build = &candidate_evidence.provenance.build;
        let candidate_evidence_bytes =
            serde_json::to_vec_pretty(&candidate_evidence).expect("candidate evidence bytes");
        let qg1_identity = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg1.label(),
            &build.git_revision,
            build
                .cargo_lock_sha256
                .as_deref()
                .expect("test Cargo.lock digest"),
            &build.executable_sha256,
            &build.command_sha256,
            build
                .environment_sha256
                .as_deref()
                .expect("test environment digest"),
            "candidate-qg1-maximum",
            &candidate.run_id,
            &candidate.run_window,
            &threshold_artifact_bytes(&candidate),
            &candidate_evidence_bytes,
        );
        assert_eq!(qg1_identity.max_exercised_cell_width(), 64);
        assert_eq!(
            applicability_plan(PerfGate::Qg2).max_exercised_cell_width,
            Some(1)
        );

        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(test_profile()),
            candidate_runner_identity: Some(&qg1_identity),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.runner_applicability_envelope_mismatch"
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn plan_coverage_includes_required_and_diagnostic_once_and_excludes_na() {
        let plan = applicability_plan(PerfGate::Qg1);
        let complete_matrix = PerfMatrixSpec::complete();
        let matrix = complete_matrix.for_gate(PerfGate::Qg1);
        let threshold_keys = expected_gate_keys(&plan);
        let evidence_cells = expected_evidence_cells(&plan);
        let mut saw_diagnostic = false;
        let mut saw_not_applicable = false;

        for (spec, classification) in matrix.into_iter().zip(&plan.cells) {
            let evidence_key =
                EvidenceCellKey::from_parts(&spec.fixture, &spec.metric, metric_unit(&spec.metric));
            let paired_metric = format!("{}_quill_over_tantivy", spec.metric);
            let oracle_null_metric = format!("{}_tantivy_over_tantivy", spec.metric);
            let quill_null_metric = format!("{}_quill_over_quill", spec.metric);
            let threshold_count = threshold_keys
                .iter()
                .filter(|key| {
                    key.fixture == spec.fixture
                        && (key.metric == spec.metric
                            || key.metric == paired_metric
                            || key.metric == oracle_null_metric
                            || key.metric == quill_null_metric)
                })
                .count();

            match classification.applicability {
                PerfCellApplicability::Required => {
                    assert_eq!(
                        evidence_cells.get(&evidence_key),
                        Some(&EvidenceRole::Required)
                    );
                    assert_eq!(threshold_count, 5);
                }
                PerfCellApplicability::Diagnostic => {
                    saw_diagnostic = true;
                    assert_eq!(
                        evidence_cells.get(&evidence_key),
                        Some(&EvidenceRole::Diagnostic)
                    );
                    assert_eq!(threshold_count, 5);
                }
                PerfCellApplicability::NotApplicable => {
                    saw_not_applicable = true;
                    assert!(!evidence_cells.contains_key(&evidence_key));
                    assert_eq!(threshold_count, 0);
                }
            }
        }

        assert!(
            saw_diagnostic,
            "QG-1 fixture must exercise Diagnostic coverage"
        );
        assert!(
            saw_not_applicable,
            "physical-64 QG-1 fixture must exercise NotApplicable coverage"
        );
    }

    #[test]
    fn filtered_threshold_and_evidence_are_never_claim_eligible() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let (mut candidate, mut candidate_evidence) =
            qg2_current_pair("new", "candidate", 160.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 160.0, 100.0);
        candidate.cells.pop();
        candidate_evidence.cells.clear();
        reseal_evidence_without_verification(&mut candidate_evidence);

        let result = evaluate_with_current(
            &baseline,
            None,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets::empty(),
        );

        assert_ne!(result.decision, PerfGateDecision::Allow);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.threshold_verified_reload_failed"
                    || reason.code == "perf.ratchet.incomplete_matrix"
                    || reason.code == "perf.ratchet.current_evidence_integrity_failed"
                    || reason.code == "perf.ratchet.current_evidence_incomplete_plan"
            }),
            "filtered artifacts escaped every strict rejection seam: {:?}",
            result.reasons
        );
    }

    #[test]
    fn candidate_and_rerun_cannot_mix_typed_finalizer_executables() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, mut rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let rerun_qg6_authorities = qg6_default_fixture_authorities();
        let rerun_qg6_authority_refs = rerun_qg6_authorities.iter().collect::<Vec<_>>();
        rerun_evidence.machine_class = crate::MachineClassEvidenceBinding::unverified(
            "sealed runner receipt has not been bound",
        );
        rerun_evidence.artifact_sha256.clear();
        seal_evidence(&mut rerun_evidence, &rerun_qg6_authority_refs);
        let threshold_bytes = threshold_artifact_bytes(&rerun);
        let evidence_bytes =
            serde_json::to_vec_pretty(&rerun_evidence).expect("rerun pre-binding evidence");
        let build = &rerun_evidence.provenance.build;
        let alternate_producer =
            crate::machine_class_registry::admitted_test_identity_for_artifacts_with_producer(
                PerfGate::Qg6.label(),
                &build.git_revision,
                build
                    .cargo_lock_sha256
                    .as_deref()
                    .expect("test Cargo.lock digest"),
                &build.executable_sha256,
                &build.command_sha256,
                build
                    .environment_sha256
                    .as_deref()
                    .expect("test environment digest"),
                &"6".repeat(64),
                "rerun-producer-mismatch",
                &rerun.run_id,
                &rerun.run_window,
                &threshold_bytes,
                &evidence_bytes,
            );
        rerun_evidence
            .bind_machine_class_identity_against_authorities(
                alternate_producer.clone(),
                &threshold_bytes,
                &evidence_bytes,
                &[],
                &rerun_qg6_authority_refs,
            )
            .expect("bind alternate producer identity");
        seal_evidence(&mut rerun_evidence, &rerun_qg6_authority_refs);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: Some(&alternate_producer),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &rerun_qg6_authority_refs,
                candidate: &rerun_qg6_authority_refs,
                rerun: &rerun_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.candidate_rerun_producer_mismatch" })
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn candidate_and_rerun_cannot_mix_benchmark_executables() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, mut rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let rerun_qg6_authorities = qg6_default_fixture_authorities();
        let rerun_qg6_authority_refs = rerun_qg6_authorities.iter().collect::<Vec<_>>();
        let alternate_executable = "7".repeat(64);
        rerun_evidence
            .provenance
            .build
            .executable_sha256
            .clone_from(&alternate_executable);
        mutate_cell_sample_provenance(&mut rerun_evidence, |provenance| {
            provenance
                .executable_sha256
                .clone_from(&alternate_executable);
        });
        rerun_evidence.machine_class = crate::MachineClassEvidenceBinding::unverified(
            "sealed runner receipt has not been bound",
        );
        rerun_evidence.artifact_sha256.clear();
        seal_evidence(&mut rerun_evidence, &rerun_qg6_authority_refs);
        let threshold_bytes = threshold_artifact_bytes(&rerun);
        let evidence_bytes =
            serde_json::to_vec_pretty(&rerun_evidence).expect("rerun pre-binding evidence");
        let build = &rerun_evidence.provenance.build;
        let alternate_benchmark =
            crate::machine_class_registry::admitted_test_identity_for_artifacts(
                PerfGate::Qg6.label(),
                &build.git_revision,
                build
                    .cargo_lock_sha256
                    .as_deref()
                    .expect("test Cargo.lock digest"),
                &build.executable_sha256,
                &build.command_sha256,
                build
                    .environment_sha256
                    .as_deref()
                    .expect("test environment digest"),
                "rerun-benchmark-mismatch",
                &rerun.run_id,
                &rerun.run_window,
                &threshold_bytes,
                &evidence_bytes,
            );
        rerun_evidence
            .bind_machine_class_identity_against_authorities(
                alternate_benchmark.clone(),
                &threshold_bytes,
                &evidence_bytes,
                &[],
                &rerun_qg6_authority_refs,
            )
            .expect("bind alternate benchmark identity");
        seal_evidence(&mut rerun_evidence, &rerun_qg6_authority_refs);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: Some(&alternate_benchmark),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &rerun_qg6_authority_refs,
                candidate: &rerun_qg6_authority_refs,
                rerun: &rerun_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.candidate_rerun_benchmark_executable_mismatch"
        }));
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn candidate_diagnostic_mutation_is_nonpromotable_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, mut candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        candidate_evidence.force_no_claim(
            "evidence.test_diagnostic_mutation",
            "candidate was explicitly downgraded after collection",
        );
        let candidate_qg6_authorities = qg6_default_fixture_authorities();
        let candidate_qg6_authority_refs = candidate_qg6_authorities.iter().collect::<Vec<_>>();
        bind_test_evidence(
            &candidate,
            &mut candidate_evidence,
            "candidate-diagnostic",
            &candidate_qg6_authority_refs,
        );
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &candidate_qg6_authority_refs,
                candidate: &candidate_qg6_authority_refs,
                rerun: &candidate_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.current_evidence_not_admissible")
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn mixed_aa_evidence_policy_is_rejected_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, mut rerun_evidence) = qg6_complete_pair("rerun", ratios);
        rerun_evidence.policy.max_raw_samples += 1;
        reseal_evidence_without_verification(&mut rerun_evidence);
        assert!(matches!(
            rerun_evidence.verify_integrity(),
            Err(EvidenceArtifactError::InvalidPolicy { .. })
        ));
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.machine_evidence_integrity_failed")
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn reused_cross_role_run_id_is_rejected_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (mut baseline, mut baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        baseline.run_id.clone_from(&candidate.run_id);
        baseline_evidence
            .provenance
            .run_id
            .clone_from(&candidate_evidence.provenance.run_id);
        let reused_run_id = candidate_evidence.provenance.run_id.clone();
        mutate_cell_sample_provenance(&mut baseline_evidence, |provenance| {
            provenance.run_id.clone_from(&reused_run_id);
        });
        let baseline_qg6_authorities = qg6_default_fixture_authorities();
        let baseline_qg6_authority_refs = baseline_qg6_authorities.iter().collect::<Vec<_>>();
        bind_test_evidence(
            &baseline,
            &mut baseline_evidence,
            "baseline-reused-run",
            &baseline_qg6_authority_refs,
        );
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &baseline_qg6_authority_refs,
                candidate: &baseline_qg6_authority_refs,
                rerun: &baseline_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.run_identity_reused")
        );
        assert!(result.comparisons.is_empty());
    }

    #[test]
    fn mixed_exact_argv_identity_is_rejected_before_comparison() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, mut baseline_evidence) = qg6_complete_pair("baseline", ratios);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", ratios);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", ratios);
        let baseline_qg6_authorities = qg6_default_fixture_authorities();
        let baseline_qg6_authority_refs = baseline_qg6_authorities.iter().collect::<Vec<_>>();
        baseline_evidence.machine_class = crate::MachineClassEvidenceBinding::unverified(
            "sealed runner receipt has not been bound",
        );
        baseline_evidence.provenance.build.command_sha256 = "0".repeat(64);
        baseline_evidence.artifact_sha256.clear();
        seal_evidence(&mut baseline_evidence, &baseline_qg6_authority_refs);
        let threshold_bytes = threshold_artifact_bytes(&baseline);
        let evidence_bytes =
            serde_json::to_vec_pretty(&baseline_evidence).expect("alternate argv evidence");
        let alternate_identity =
            crate::machine_class_registry::admitted_test_identity_for_artifacts(
                PerfGate::Qg6.label(),
                &"1".repeat(40),
                &"e".repeat(64),
                &"c".repeat(64),
                &"0".repeat(64),
                &"d".repeat(64),
                "baseline",
                &baseline.run_id,
                &baseline.run_window,
                &threshold_bytes,
                &evidence_bytes,
            );
        baseline_evidence
            .bind_machine_class_identity_against_authorities(
                alternate_identity,
                &threshold_bytes,
                &evidence_bytes,
                &[],
                &baseline_qg6_authority_refs,
            )
            .expect("bind alternate exact argv identity");
        seal_evidence(&mut baseline_evidence, &baseline_qg6_authority_refs);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();

        let result = evaluate_verified_promotion_request(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: Some(&baseline_evidence),
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_evidence.machine_class.identity(),
                rerun_runner_identity: rerun_evidence.machine_class.identity(),
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg6AuthoritySets {
                baseline: &baseline_qg6_authority_refs,
                candidate: &baseline_qg6_authority_refs,
                rerun: &baseline_qg6_authority_refs,
            },
        );

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.mixed_command_identity")
        );
        assert!(result.comparisons.is_empty());
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

        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.threshold_verified_reload_failed"
                && reason.message.contains("rerun")
        }));
    }

    #[test]
    fn current_evidence_reconciles_with_threshold_projection_and_rerun() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 161.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 161.0, 100.0);
        let result = evaluate_with_current(
            &baseline,
            None,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets::empty(),
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
            None,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets::empty(),
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
        let result = evaluate_perf_ratchet_inner(
            PerfRatchetRequest {
                baseline: Some(&baseline),
                baseline_evidence: None,
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: None,
                rerun_evidence: None,
                expected_machine_profile: None,
                candidate_runner_identity: None,
                rerun_runner_identity: None,
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            },
            PerfRatchetQg1AuthoritySets::empty(),
            PerfRatchetQg6AuthoritySets::empty(),
            DecisionState::default(),
            true,
        );
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
            None,
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets::empty(),
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
            &normalized_manifest_sha256(),
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
        assert!(
            !state.fatal,
            "unexpected fatal reasons: {:?}",
            state.reasons
        );
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
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.0; 3]; 4]);
        let mut rerun_ratios = [[1.0; 3]; crate::QG6_QUERY_GROUPS];
        for ratios in &mut rerun_ratios[..7] {
            *ratios = [0.80; 3];
        }
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", rerun_ratios);
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

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
            Some(&baseline_evidence),
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
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
        let (baseline, baseline_evidence) = qg6_complete_pair("baseline", [[1.0; 3]; 4]);
        let (candidate, candidate_evidence) = qg6_complete_pair("candidate", [[1.0; 3]; 4]);
        let (rerun, rerun_evidence) = qg6_complete_pair("rerun", [[1.01; 3]; 4]);
        let retained_authorities = qg6_default_fixture_authorities();
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();

        let result = evaluate_with_current(
            &baseline,
            Some(&baseline_evidence),
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );
        assert_eq!(result.decision, PerfGateDecision::Block);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.gate_target_missed"
                    && reason.message.contains("joint true-leaf p99")
                    && reason.message.contains("does not clear oracle parity")
            }),
            "{result:#?}"
        );
        assert!(!result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
        }));
    }

    #[test]
    fn qg6_gate_uses_hierarchical_ci_when_flat_projection_would_false_pass() {
        let mut ratios = [[0.91; 3]; crate::QG6_QUERY_GROUPS];
        for group in &mut ratios[9..] {
            *group = [0.91, 1.20, 1.20];
        }
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            observe_only: false,
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut state,
        );
        validate_paired_evidence(PerfGate::Qg6, &cells, "candidate", &mut state);
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            observe_only: false,
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
        let effect_ratios = [[0.95; 3]; crate::QG6_QUERY_GROUPS];
        let mut null_ratios = [[1.0; 3]; crate::QG6_QUERY_GROUPS];
        for group in &mut null_ratios[9..] {
            *group = [1.0, 1.08, 1.08];
        }
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            observe_only: false,
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut state,
        );
        let mut target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &cells,
            activated: true,
            observe_only: false,
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut missing_state,
        );
        let mut missing_target = GateTargetEvaluator {
            artifact: &artifact,
            cells: &missing_cells,
            activated: true,
            observe_only: false,
            state: &mut missing_state,
        };
        evaluate_qg6(&mut missing_target, Some(&evidence));
        assert!(
            missing_state
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.qg6_joint_tail_evidence_missing")
        );
    }

    #[test]
    fn qg6_reproduction_uses_hierarchical_grouping_not_equal_flat_multisets() {
        // Two ratio values with log-spread 0.0445: inside the window where the
        // hierarchical median-of-group-medians delta (spread/2 = 0.02225)
        // exceeds max_reproduction_delta_log (ln 1.02) while every half or
        // order-subset median stays within the bd-yo5by effect drift and
        // order-effect bounds (ln 1.05) no matter how blocks land — the
        // previous 0.95/1.10 fixture drifted between halves and now correctly
        // classifies InvalidExperiment before reproduction is evaluated. Both
        // runs share one flat multiset (7 low, 5 high), so the former flat
        // estimator still calls them reproductions of each other.
        let candidate_groups = [
            [0.978, 0.978, 0.978],
            [0.978, 0.978, 0.978],
            [0.978, 1.0225, 1.0225],
            [1.0225, 1.0225, 1.0225],
        ];
        let rerun_groups = [
            [0.978, 0.978, 1.0225],
            [0.978, 0.978, 1.0225],
            [0.978, 0.978, 1.0225],
            [0.978, 1.0225, 1.0225],
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
    fn qg6_reproduction_rejects_p99_only_effect_tail_drift_after_independent_target_passes() {
        let ratios = [[0.98; 3]; 4];
        let (baseline, baseline_evidence) = qg6_complete_pair_with_leaf_profile(
            "baseline",
            ratios,
            Qg6TestLeafProfile::EffectP99 {
                baseline_numerator: 98,
                exceptional_numerator: 98,
            },
        );
        let (candidate, candidate_evidence) = qg6_complete_pair_with_leaf_profile(
            "candidate",
            ratios,
            Qg6TestLeafProfile::EffectP99 {
                baseline_numerator: 98,
                exceptional_numerator: 98,
            },
        );
        let (rerun, rerun_evidence) = qg6_complete_pair_with_leaf_profile(
            "rerun",
            ratios,
            Qg6TestLeafProfile::EffectP99 {
                baseline_numerator: 98,
                exceptional_numerator: 100,
            },
        );
        let retained_authorities = qg6_fixture_authorities_for_shape::<3>(128, false);
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();
        assert_qg6_target_passes(
            &candidate,
            &candidate_evidence,
            "candidate",
            &retained_authority_refs,
        );
        assert_qg6_target_passes(&rerun, &rerun_evidence, "rerun", &retained_authority_refs);

        let candidate_tail =
            exact_qg6_joint_tail_cell(&candidate_evidence, "query/identifier/k10/100k")
                .expect("candidate joint tail");
        let rerun_tail = exact_qg6_joint_tail_cell(&rerun_evidence, "query/identifier/k10/100k")
            .expect("rerun joint tail");
        let p50_delta =
            (candidate_tail.effect.p50_ratio.ln() - rerun_tail.effect.p50_ratio.ln()).abs();
        let p99_delta =
            (candidate_tail.effect.p99_ratio.ln() - rerun_tail.effect.p99_ratio.ln()).abs();
        assert!(
            p50_delta
                <= crate::PairedEstimatorConfig::predeclared(QG6_TEST_SCHEDULE_SEED)
                    .max_reproduction_delta_log,
            "fixture must hold the effect p50 constant"
        );
        assert!(
            p99_delta
                > crate::PairedEstimatorConfig::predeclared(QG6_TEST_SCHEDULE_SEED)
                    .max_reproduction_delta_log,
            "fixture must exceed reproduction tolerance through effect p99 alone"
        );
        assert_eq!(candidate_tail.tantivy_null, rerun_tail.tantivy_null);
        assert_eq!(candidate_tail.quill_null, rerun_tail.quill_null);

        let result = evaluate_with_current(
            &baseline,
            Some(&baseline_evidence),
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine, "{result:#?}");
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_joint_tail_reproduction_failed"
                && reason.message.contains("Quill/Tantivy p99")
        }));
        assert!(!result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                || reason.code == "perf.ratchet.reproduction_failed"
                || reason.code == "perf.ratchet.gate_target_missed"
                || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
        }));
    }

    #[test]
    fn qg6_reproduction_rejects_p99_only_quill_null_tail_drift_after_independent_target_passes() {
        let ratios = [[1.0; 3]; 4];
        let (baseline, baseline_evidence) =
            qg6_complete_pair_with_leaf_profile("baseline", ratios, Qg6TestLeafProfile::Uniform);
        let (candidate, candidate_evidence) =
            qg6_complete_pair_with_leaf_profile("candidate", ratios, Qg6TestLeafProfile::Uniform);
        let (rerun, rerun_evidence) = qg6_complete_pair_with_leaf_profile(
            "rerun",
            ratios,
            Qg6TestLeafProfile::QuillNullP99 {
                exceptional_numerator: 1_021,
                denominator: 1_000,
            },
        );
        let retained_authorities = qg6_fixture_authorities_for_shape::<3>(128, false);
        let retained_authority_refs = retained_authorities.iter().collect::<Vec<_>>();
        assert_qg6_target_passes(
            &candidate,
            &candidate_evidence,
            "candidate",
            &retained_authority_refs,
        );
        assert_qg6_target_passes(&rerun, &rerun_evidence, "rerun", &retained_authority_refs);

        let candidate_tail =
            exact_qg6_joint_tail_cell(&candidate_evidence, "query/identifier/k10/100k")
                .expect("candidate joint tail");
        let rerun_tail = exact_qg6_joint_tail_cell(&rerun_evidence, "query/identifier/k10/100k")
            .expect("rerun joint tail");
        let p50_delta =
            (candidate_tail.quill_null.p50_ratio.ln() - rerun_tail.quill_null.p50_ratio.ln()).abs();
        let p99_delta =
            (candidate_tail.quill_null.p99_ratio.ln() - rerun_tail.quill_null.p99_ratio.ln()).abs();
        assert!(
            p50_delta
                <= crate::PairedEstimatorConfig::predeclared(QG6_TEST_SCHEDULE_SEED)
                    .max_reproduction_delta_log,
            "fixture must hold the Quill-null p50 constant"
        );
        assert!(
            p99_delta
                > crate::PairedEstimatorConfig::predeclared(QG6_TEST_SCHEDULE_SEED)
                    .max_reproduction_delta_log,
            "fixture must exceed reproduction tolerance through Quill-null p99 alone"
        );
        assert_eq!(candidate_tail.effect, rerun_tail.effect);
        assert_eq!(candidate_tail.tantivy_null, rerun_tail.tantivy_null);

        let result = evaluate_with_current(
            &baseline,
            Some(&baseline_evidence),
            &candidate,
            Some(&rerun),
            Some(&candidate_evidence),
            Some(&rerun_evidence),
            PerfRatchetQg6AuthoritySets {
                baseline: &retained_authority_refs,
                candidate: &retained_authority_refs,
                rerun: &retained_authority_refs,
            },
        );
        assert_eq!(result.decision, PerfGateDecision::Quarantine, "{result:#?}");
        assert!(result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_joint_tail_reproduction_failed"
                && reason.message.contains("Quill/Quill p99")
        }));
        assert!(!result.reasons.iter().any(|reason| {
            reason.code == "perf.ratchet.qg6_hierarchical_reproduction_failed"
                || reason.code == "perf.ratchet.reproduction_failed"
                || reason.code == "perf.ratchet.gate_target_missed"
                || reason.code == "perf.ratchet.gate_target_ci_inconclusive"
        }));
    }

    #[test]
    fn qg6_ratchet_requires_leaf_derived_absolute_projection() {
        let authority = qg6_fixture_authority_for_cell::<3>(
            crate::PerfQueryClass::Identifier,
            10,
            100_000,
            3,
            false,
            QG6_TEST_SCHEDULE_SEED,
        );
        let (mut artifact, evidence) = qg6_current_pair_for_cell(
            "leaf-tail-projection",
            [[1.0; 3]; 4],
            [[1.0; 3]; 4],
            crate::PerfQueryClass::Identifier,
            10,
            100_000,
            "query/identifier/k10/100k",
            true,
            3,
            crate::perf::PERF_BOOTSTRAP_RESAMPLES,
            false,
            &authority,
        );
        let cell = evidence.cells.first().expect("QG-6 evidence cell");
        let (paired, protocol) = match &cell.body {
            EvidenceCellBody::Paired {
                paired,
                qg6_protocol: Some(protocol),
                ..
            } => (paired, protocol.as_ref()),
            EvidenceCellBody::Facts { .. } => unreachable!("QG-6 must be paired"),
            EvidenceCellBody::Paired {
                qg6_protocol: None, ..
            } => unreachable!("QG-6 must carry formal protocol evidence"),
        };
        let leaf_treatment = artifact
            .cells
            .iter()
            .find(|row| row.engine == "quill")
            .expect("QG-6 Quill absolute row")
            .distribution
            .clone();
        assert!(
            leaf_treatment.p99 > paired.effect.treatment.p99,
            "fixture hides a treatment tail behind equal parent medians"
        );

        let rows = artifact
            .cells
            .iter()
            .map(|row| (CellKey::from(row), row))
            .collect::<BTreeMap<_, _>>();
        let mut state = DecisionState::default();
        reconcile_current_cell_with_projection(
            cell,
            paired,
            None,
            Some(protocol),
            &rows,
            "candidate",
            &mut state,
        );
        assert!(
            !state.fatal,
            "leaf projection must reconcile: {:?}",
            state.reasons
        );

        artifact
            .cells
            .iter_mut()
            .find(|row| row.engine == "quill")
            .expect("QG-6 Quill absolute row")
            .distribution = paired.effect.treatment.clone();
        let parent_rows = artifact
            .cells
            .iter()
            .map(|row| (CellKey::from(row), row))
            .collect::<BTreeMap<_, _>>();
        let mut parent_state = DecisionState::default();
        reconcile_current_cell_with_projection(
            cell,
            paired,
            None,
            Some(protocol),
            &parent_rows,
            "candidate",
            &mut parent_state,
        );
        assert!(
            parent_state.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.current_evidence_projection_mismatch"
            })
        );
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
            &normalized_manifest_sha256(),
            "candidate",
            &mut state,
        );
        let rerun_cells = validate_artifact(
            &rerun,
            PerfGate::Qg6,
            &normalized_manifest_sha256(),
            "rerun",
            &mut state,
        );
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
            &normalized_manifest_sha256(),
            "baseline",
            &mut state,
        );
        let candidate_cells = validate_artifact(
            &candidate,
            PerfGate::Qg6,
            &normalized_manifest_sha256(),
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
    fn public_regression_alarm_reads_legacy_thresholds_without_promotion_inputs() {
        let baseline = qg2_artifact("old", 160.0, 100.0);
        let candidate = qg2_artifact("new", 161.0, 100.0);
        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: None,
            candidate_evidence: None,
            rerun_evidence: None,
            expected_machine_profile: None,
            candidate_runner_identity: None,
            rerun_runner_identity: None,
            gate_activated: false,
            mode: PerfRatchetMode::RegressionAlarm,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });

        assert_eq!(result.decision, PerfGateDecision::Allow);
        assert!(
            result.reasons.iter().any(|reason| {
                reason.code == "perf.ratchet.legacy_regression_alarm_nonpromotable"
            })
        );
        assert!(
            result
                .reasons
                .iter()
                .all(|reason| !reason.code.contains("missing_current"))
        );
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
    fn activated_bootstrap_target_miss_blocks_baseline_replacement() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 110.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 110.0, 100.0);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(expected_profile),
            candidate_runner_identity: candidate_evidence.machine_class.identity(),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: true,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });
        assert_eq!(
            result.decision,
            PerfGateDecision::Block,
            "activated bootstrap target-miss reasons: {:#?}",
            result.reasons
        );
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.gate_target_missed")
        );
        assert!(
            result
                .reasons
                .iter()
                .all(|reason| reason.code != "perf.ratchet.bootstrap_target_missed"),
            "an activated gate must not re-enter the bootstrap target-miss concession"
        );
    }

    #[test]
    fn inactive_bootstrap_target_miss_remains_observational() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 110.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 110.0, 100.0);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let result = evaluate_perf_ratchet(PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: None,
            candidate: &candidate,
            rerun: Some(&rerun),
            candidate_evidence: Some(&candidate_evidence),
            rerun_evidence: Some(&rerun_evidence),
            expected_machine_profile: Some(expected_profile),
            candidate_runner_identity: candidate_evidence.machine_class.identity(),
            rerun_runner_identity: rerun_evidence.machine_class.identity(),
            gate_activated: false,
            mode: PerfRatchetMode::Promotion,
            expected_manifest_sha256: &normalized_manifest_sha256(),
            evidence: Vec::new(),
        });
        assert_eq!(result.decision, PerfGateDecision::Quarantine);
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.code == "perf.ratchet.bootstrap_target_missed")
        );
        assert!(
            result
                .reasons
                .iter()
                .all(|reason| reason.code != "perf.ratchet.gate_target_missed")
        );
    }

    #[test]
    fn preserved_v6_placeholders_are_canonical_but_not_current_v7_sentinels() {
        let manifest = include_str!("../../../docs/contracts/quill-perf-gates.toml");
        let manifest_sha256 = crate::perf_manifest_contract_sha256(manifest);
        for raw in [
            include_str!("../../../.bench-history/QG-1.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-2.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-3.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-4.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-5.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-6.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-7.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-8.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-9.unmeasured.latest.json"),
            include_str!("../../../.bench-history/QG-10.unmeasured.latest.json"),
        ] {
            let artifact =
                serde_json::from_str::<PerfGateArtifact>(raw).expect("committed placeholder");
            assert_eq!(
                raw.as_bytes(),
                serde_json::to_vec_pretty(&artifact)
                    .expect("canonical committed placeholder")
                    .as_slice(),
                "{} placeholder bytes are not exact canonical pretty JSON",
                artifact.gate
            );
            assert_eq!(artifact.schema_version, "quill-perf-artifact-v6");
            assert!(
                !is_explicit_bootstrap_for(&artifact, artifact.gate, &manifest_sha256),
                "{} legacy placeholder must not satisfy the current-schema sentinel exemption",
                artifact.gate
            );
        }
    }

    #[test]
    fn every_current_schema_bootstrap_sentinel_is_committed_and_exact() {
        let manifest = include_str!("../../../docs/contracts/quill-perf-gates.toml");
        let manifest_sha256 = crate::perf_manifest_contract_sha256(manifest);
        let version = PERF_ARTIFACT_SCHEMA_VERSION
            .strip_prefix("quill-perf-artifact-")
            .expect("current threshold schema is version-tagged");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.bench-history");
        for gate in PerfGate::ALL {
            let path = root.join(format!("{}.{version}.unmeasured.latest.json", gate.label()));
            let raw = std::fs::read(&path)
                .unwrap_or_else(|error| panic!("{} is missing: {error}", path.display()));
            let artifact = serde_json::from_slice::<PerfGateArtifact>(&raw)
                .unwrap_or_else(|error| panic!("{} is malformed: {error}", path.display()));
            let mut canonical =
                serde_json::to_vec_pretty(&artifact).expect("canonical current placeholder");
            canonical.push(b'\n');
            assert_eq!(raw, canonical, "{} is not canonical", path.display());
            assert!(
                is_explicit_bootstrap_for(&artifact, gate, &manifest_sha256),
                "{gate} current placeholder must satisfy the narrow sentinel exemption"
            );
        }
    }

    #[test]
    fn bootstrap_exemption_rejects_near_misses_and_missing_measured_receipts() {
        let exact = explicit_bootstrap(PerfGate::Qg2);
        let (candidate, candidate_evidence) = qg2_current_pair("new", "candidate", 110.0, 100.0);
        let (rerun, rerun_evidence) = qg2_current_pair("new", "rerun", 110.0, 100.0);
        let expected_profile = candidate_evidence
            .machine_class
            .identity()
            .expect("candidate identity")
            .profile();
        let evaluate = |baseline: &PerfGateArtifact,
                        baseline_evidence: Option<&PerfEvidenceArtifact>,
                        _baseline_identity: Option<&VerifiedRunnerIdentity>,
                        candidate_identity: Option<&VerifiedRunnerIdentity>,
                        rerun_identity: Option<&VerifiedRunnerIdentity>| {
            evaluate_perf_ratchet(PerfRatchetRequest {
                baseline: Some(baseline),
                baseline_evidence,
                candidate: &candidate,
                rerun: Some(&rerun),
                candidate_evidence: Some(&candidate_evidence),
                rerun_evidence: Some(&rerun_evidence),
                expected_machine_profile: Some(expected_profile),
                candidate_runner_identity: candidate_identity,
                rerun_runner_identity: rerun_identity,
                gate_activated: true,
                mode: PerfRatchetMode::Promotion,
                expected_manifest_sha256: &normalized_manifest_sha256(),
                evidence: Vec::new(),
            })
        };

        let mut almost = exact.clone();
        almost.git_rev = "almost-unmeasured".to_owned();
        let almost_result = evaluate(
            &almost,
            None,
            None,
            candidate_evidence.machine_class.identity(),
            rerun_evidence.machine_class.identity(),
        );
        assert_eq!(almost_result.decision, PerfGateDecision::Quarantine);
        assert!(
            almost_result
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.machine_identity_incomplete" })
        );

        let mut measured_v3 = qg2_artifact("old", 160.0, 100.0);
        measured_v3.schema_version = crate::LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3.to_owned();
        measured_v3.applicability_plan = None;
        let measured_v3_result = evaluate(
            &measured_v3,
            None,
            None,
            candidate_evidence.machine_class.identity(),
            rerun_evidence.machine_class.identity(),
        );
        assert_eq!(measured_v3_result.decision, PerfGateDecision::Quarantine);

        let missing_candidate = evaluate(
            &exact,
            None,
            None,
            None,
            rerun_evidence.machine_class.identity(),
        );
        assert_eq!(missing_candidate.decision, PerfGateDecision::Quarantine);
        let missing_rerun = evaluate(
            &exact,
            None,
            None,
            candidate_evidence.machine_class.identity(),
            None,
        );
        assert_eq!(missing_rerun.decision, PerfGateDecision::Quarantine);

        let fabricated_baseline = evaluate(
            &exact,
            Some(&candidate_evidence),
            candidate_evidence.machine_class.identity(),
            candidate_evidence.machine_class.identity(),
            rerun_evidence.machine_class.identity(),
        );
        assert_eq!(fabricated_baseline.decision, PerfGateDecision::Block);
        assert!(
            fabricated_baseline
                .reasons
                .iter()
                .any(|reason| { reason.code == "perf.ratchet.bootstrap_identity_fabricated" })
        );
    }

    #[test]
    fn bootstrap_cannot_satisfy_pr_regression_alarm() {
        let baseline = explicit_bootstrap(PerfGate::Qg2);
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
