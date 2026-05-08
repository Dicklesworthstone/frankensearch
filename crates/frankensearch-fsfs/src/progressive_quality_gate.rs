//! Progressive search release quality gate contracts.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Write as _;

pub const PROGRESSIVE_RELEASE_GATE_KIND: &str = "fsfs_progressive_release_quality_gate_pack";
pub const PROGRESSIVE_RELEASE_GATE_SCHEMA_VERSION: &str =
    "fsfs-progressive-release-quality-gate-v1";
pub const PROGRESSIVE_RELEASE_GATE_MATRIX_VERSION: &str = "progressive-release-gate-matrix-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Pass,
    FailClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureLane {
    HashOnly,
    Hybrid,
    Durable,
    Full,
}

impl FeatureLane {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HashOnly => "hash-only",
            Self::Hybrid => "hybrid",
            Self::Durable => "durable",
            Self::Full => "full",
        }
    }

    #[must_use]
    pub const fn feature_flags(self) -> &'static str {
        match self {
            Self::HashOnly => "hash",
            Self::Hybrid => "hybrid",
            Self::Durable => "durable",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    Initial,
    Refined,
    RefinementFailed,
}

impl PhaseKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Refined => "refined",
            Self::RefinementFailed => "refinement_failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    NdcgAt10,
    Mrr,
    RecallAt10,
}

impl MetricKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NdcgAt10 => "nDCG@10",
            Self::Mrr => "MRR",
            Self::RecallAt10 => "Recall@10",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    GateEventsJsonl,
    SummaryJson,
    SummaryMarkdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingKind {
    MissingFeatureLane,
    MissingRchCommand,
    MissingPhaseContract,
    PhaseContractFailed,
    LexicalFallbackRegression,
    OrderingRegression,
    QualityEnvelopeRegression,
    MissingArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureLaneGate {
    pub lane: FeatureLane,
    pub feature_flags: Vec<String>,
    pub cargo_check_command: String,
    pub behavior_test_command: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseContractGate {
    pub lane: FeatureLane,
    pub phase: PhaseKind,
    pub status: GateStatus,
    pub reason_code: String,
    pub preserves_initial_results: bool,
    pub result_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalFallbackGate {
    pub lane: FeatureLane,
    pub lexical_enabled: bool,
    pub semantic_failure_mode: String,
    pub initial_results_observed: bool,
    pub hard_failure_observed: bool,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderingGate {
    pub lane: FeatureLane,
    pub stable: bool,
    pub tie_break_keys: Vec<String>,
    pub repeated_run_doc_ids: Vec<String>,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityEnvelopeGate {
    pub lane: FeatureLane,
    pub metric: MetricKind,
    pub observed: f64,
    pub baseline: f64,
    pub minimum: f64,
    pub max_regression: f64,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateArtifact {
    pub kind: ArtifactKind,
    pub path: String,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseGateInput {
    pub run_id: String,
    pub lanes: Vec<FeatureLaneGate>,
    pub phase_contracts: Vec<PhaseContractGate>,
    pub lexical_fallback: Vec<LexicalFallbackGate>,
    pub ordering: Vec<OrderingGate>,
    pub quality_envelopes: Vec<QualityEnvelopeGate>,
    pub artifacts: Vec<GateArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateFinding {
    pub kind: FindingKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane: Option<FeatureLane>,
    pub reason_code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateSummary {
    pub verdict: GateVerdict,
    pub lane_count: u32,
    pub phase_contract_count: u32,
    pub lexical_fallback_count: u32,
    pub ordering_contract_count: u32,
    pub quality_envelope_count: u32,
    pub artifact_count: u32,
    pub finding_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseGatePack {
    pub kind: String,
    pub schema_version: String,
    pub matrix_version: String,
    pub input: ReleaseGateInput,
    pub summary: GateSummary,
    pub findings: Vec<GateFinding>,
    pub events_jsonl_path: String,
    pub summary_json_path: String,
    pub summary_markdown_path: String,
    pub replay_command: String,
    pub human_summary: String,
}

impl ReleaseGatePack {
    #[must_use]
    pub fn from_input(input: ReleaseGateInput) -> Self {
        let mut findings = evaluate(&input);
        findings.sort_by(|left, right| {
            (
                left.lane,
                left.kind,
                left.reason_code.as_str(),
                left.message.as_str(),
            )
                .cmp(&(
                    right.lane,
                    right.kind,
                    right.reason_code.as_str(),
                    right.message.as_str(),
                ))
        });

        let verdict = if findings.is_empty() {
            GateVerdict::Pass
        } else {
            GateVerdict::FailClosed
        };
        let summary = GateSummary {
            verdict,
            lane_count: usize_to_u32(input.lanes.len()),
            phase_contract_count: usize_to_u32(input.phase_contracts.len()),
            lexical_fallback_count: usize_to_u32(input.lexical_fallback.len()),
            ordering_contract_count: usize_to_u32(input.ordering.len()),
            quality_envelope_count: usize_to_u32(input.quality_envelopes.len()),
            artifact_count: usize_to_u32(input.artifacts.len()),
            finding_count: usize_to_u32(findings.len()),
        };

        let events_jsonl_path = format!("runs/{}/progressive_gate/gate-events.jsonl", input.run_id);
        let summary_json_path = format!("runs/{}/progressive_gate/summary.json", input.run_id);
        let summary_markdown_path = format!("runs/{}/progressive_gate/summary.md", input.run_id);
        let replay_command = format!(
            "scripts/check_progressive_release_quality_gate.sh --mode all --run-id {}",
            input.run_id
        );
        let human_summary = render_human_summary(&input, verdict, &findings);

        Self {
            kind: PROGRESSIVE_RELEASE_GATE_KIND.to_owned(),
            schema_version: PROGRESSIVE_RELEASE_GATE_SCHEMA_VERSION.to_owned(),
            matrix_version: PROGRESSIVE_RELEASE_GATE_MATRIX_VERSION.to_owned(),
            input,
            summary,
            findings,
            events_jsonl_path,
            summary_json_path,
            summary_markdown_path,
            replay_command,
            human_summary,
        }
    }

    #[must_use]
    pub fn passed(&self) -> bool {
        self.summary.verdict == GateVerdict::Pass
    }
}

#[must_use]
pub fn default_release_gate_input(run_id: &str) -> ReleaseGateInput {
    let lanes = required_lanes()
        .into_iter()
        .map(default_lane_gate)
        .collect();
    let phase_contracts = required_lanes()
        .into_iter()
        .flat_map(default_phase_contracts)
        .collect();
    let lexical_fallback = [FeatureLane::Hybrid, FeatureLane::Durable, FeatureLane::Full]
        .into_iter()
        .map(default_lexical_fallback)
        .collect();
    let ordering = required_lanes()
        .into_iter()
        .map(default_ordering_gate)
        .collect();
    let quality_envelopes = required_lanes()
        .into_iter()
        .flat_map(default_quality_envelopes)
        .collect();

    ReleaseGateInput {
        run_id: run_id.to_owned(),
        lanes,
        phase_contracts,
        lexical_fallback,
        ordering,
        quality_envelopes,
        artifacts: vec![
            GateArtifact {
                kind: ArtifactKind::GateEventsJsonl,
                path: format!("runs/{run_id}/progressive_gate/gate-events.jsonl"),
                format: "jsonl".to_owned(),
            },
            GateArtifact {
                kind: ArtifactKind::SummaryJson,
                path: format!("runs/{run_id}/progressive_gate/summary.json"),
                format: "json".to_owned(),
            },
            GateArtifact {
                kind: ArtifactKind::SummaryMarkdown,
                path: format!("runs/{run_id}/progressive_gate/summary.md"),
                format: "markdown".to_owned(),
            },
        ],
    }
}

fn evaluate(input: &ReleaseGateInput) -> Vec<GateFinding> {
    let mut findings = Vec::new();
    check_lanes(input, &mut findings);
    check_phase_contracts(input, &mut findings);
    check_lexical_fallback(input, &mut findings);
    check_ordering(input, &mut findings);
    check_quality_envelopes(input, &mut findings);
    check_artifacts(input, &mut findings);
    findings
}

fn check_lanes(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    let observed: BTreeSet<FeatureLane> = input.lanes.iter().map(|gate| gate.lane).collect();
    for lane in required_lanes() {
        if !observed.contains(&lane) {
            findings.push(finding(
                FindingKind::MissingFeatureLane,
                Some(lane),
                "RELEASE_GATE_MISSING_FEATURE_LANE",
                format!("missing required {} feature lane", lane.as_str()),
            ));
        }
    }

    for lane in &input.lanes {
        if !is_rch_cargo_command(&lane.cargo_check_command)
            || !is_rch_cargo_command(&lane.behavior_test_command)
        {
            findings.push(finding(
                FindingKind::MissingRchCommand,
                Some(lane.lane),
                "RELEASE_GATE_COMMAND_NOT_RCH_BACKED",
                format!(
                    "{} lane command is not explicitly rch-backed",
                    lane.lane.as_str()
                ),
            ));
        }
        if lane.feature_flags.is_empty() {
            findings.push(finding(
                FindingKind::MissingFeatureLane,
                Some(lane.lane),
                "RELEASE_GATE_EMPTY_FEATURE_FLAGS",
                format!("{} lane has no feature flags", lane.lane.as_str()),
            ));
        }
    }
}

fn check_phase_contracts(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    for lane in required_lanes() {
        for phase in [
            PhaseKind::Initial,
            PhaseKind::Refined,
            PhaseKind::RefinementFailed,
        ] {
            let Some(contract) = input
                .phase_contracts
                .iter()
                .find(|contract| contract.lane == lane && contract.phase == phase)
            else {
                findings.push(finding(
                    FindingKind::MissingPhaseContract,
                    Some(lane),
                    "RELEASE_GATE_MISSING_PHASE_CONTRACT",
                    format!(
                        "{} lane missing {} phase contract",
                        lane.as_str(),
                        phase.as_str()
                    ),
                ));
                continue;
            };

            if contract.status != GateStatus::Pass {
                findings.push(finding(
                    FindingKind::PhaseContractFailed,
                    Some(lane),
                    contract.reason_code.as_str(),
                    format!(
                        "{} lane {} phase contract failed",
                        lane.as_str(),
                        phase.as_str()
                    ),
                ));
            }
            if phase == PhaseKind::Initial && contract.result_count == 0 {
                findings.push(finding(
                    FindingKind::PhaseContractFailed,
                    Some(lane),
                    "RELEASE_GATE_INITIAL_EMPTY",
                    format!("{} lane initial phase produced no results", lane.as_str()),
                ));
            }
            if phase == PhaseKind::RefinementFailed
                && (!contract.preserves_initial_results || contract.result_count == 0)
            {
                findings.push(finding(
                    FindingKind::PhaseContractFailed,
                    Some(lane),
                    "RELEASE_GATE_REFINEMENT_FAILED_DROPS_INITIAL",
                    format!(
                        "{} lane refinement failure does not preserve initial results",
                        lane.as_str()
                    ),
                ));
            }
        }
    }
}

fn check_lexical_fallback(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    for lane in [FeatureLane::Hybrid, FeatureLane::Durable, FeatureLane::Full] {
        let Some(contract) = input
            .lexical_fallback
            .iter()
            .find(|contract| contract.lane == lane)
        else {
            findings.push(finding(
                FindingKind::LexicalFallbackRegression,
                Some(lane),
                "RELEASE_GATE_MISSING_LEXICAL_FALLBACK",
                format!("{} lane missing lexical fallback guard", lane.as_str()),
            ));
            continue;
        };

        if !contract.lexical_enabled
            || !contract.initial_results_observed
            || contract.hard_failure_observed
        {
            findings.push(finding(
                FindingKind::LexicalFallbackRegression,
                Some(lane),
                contract.reason_code.as_str(),
                format!("{} lane lexical fallback regressed", lane.as_str()),
            ));
        }
    }
}

fn check_ordering(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    for lane in required_lanes() {
        let Some(contract) = input.ordering.iter().find(|contract| contract.lane == lane) else {
            findings.push(finding(
                FindingKind::OrderingRegression,
                Some(lane),
                "RELEASE_GATE_MISSING_ORDERING_CONTRACT",
                format!(
                    "{} lane missing deterministic ordering guard",
                    lane.as_str()
                ),
            ));
            continue;
        };

        if !contract.stable
            || contract.tie_break_keys.is_empty()
            || contract.repeated_run_doc_ids.is_empty()
        {
            findings.push(finding(
                FindingKind::OrderingRegression,
                Some(lane),
                contract.reason_code.as_str(),
                format!("{} lane deterministic ordering regressed", lane.as_str()),
            ));
        }
    }
}

fn check_quality_envelopes(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    for lane in required_lanes() {
        for metric in [
            MetricKind::NdcgAt10,
            MetricKind::Mrr,
            MetricKind::RecallAt10,
        ] {
            let Some(envelope) = input
                .quality_envelopes
                .iter()
                .find(|envelope| envelope.lane == lane && envelope.metric == metric)
            else {
                findings.push(finding(
                    FindingKind::QualityEnvelopeRegression,
                    Some(lane),
                    "RELEASE_GATE_MISSING_QUALITY_ENVELOPE",
                    format!(
                        "{} lane missing {} envelope",
                        lane.as_str(),
                        metric.as_str()
                    ),
                ));
                continue;
            };

            let allowed_floor = envelope.baseline - envelope.max_regression;
            if !envelope.observed.is_finite()
                || envelope.observed < envelope.minimum
                || envelope.observed < allowed_floor
            {
                findings.push(finding(
                    FindingKind::QualityEnvelopeRegression,
                    Some(lane),
                    envelope.reason_code.as_str(),
                    format!(
                        "{} lane {} envelope regressed",
                        lane.as_str(),
                        metric.as_str()
                    ),
                ));
            }
        }
    }
}

fn check_artifacts(input: &ReleaseGateInput, findings: &mut Vec<GateFinding>) {
    for kind in [
        ArtifactKind::GateEventsJsonl,
        ArtifactKind::SummaryJson,
        ArtifactKind::SummaryMarkdown,
    ] {
        let Some(artifact) = input
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == kind)
        else {
            findings.push(finding(
                FindingKind::MissingArtifact,
                None,
                "RELEASE_GATE_MISSING_ARTIFACT",
                format!("missing required {kind:?} artifact"),
            ));
            continue;
        };

        if artifact.path.is_empty() || artifact.format.is_empty() {
            findings.push(finding(
                FindingKind::MissingArtifact,
                None,
                "RELEASE_GATE_EMPTY_ARTIFACT_PATH",
                format!("{kind:?} artifact has empty path or format"),
            ));
        }
    }
}

fn render_human_summary(
    input: &ReleaseGateInput,
    verdict: GateVerdict,
    findings: &[GateFinding],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Progressive Release Quality Gate");
    let _ = writeln!(out);
    let _ = writeln!(out, "run_id: {}", input.run_id);
    let _ = writeln!(out, "verdict: {verdict:?}");
    let _ = writeln!(out);
    let _ = writeln!(out, "| Lane | Features | Cargo check | Behavior test |");
    let _ = writeln!(out, "|---|---|---|---|");
    for lane in &input.lanes {
        let _ = writeln!(
            out,
            "| {} | {} | `{}` | `{}` |",
            lane.lane.as_str(),
            lane.feature_flags.join(","),
            lane.cargo_check_command,
            lane.behavior_test_command
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "| Finding | Lane | Reason |");
    let _ = writeln!(out, "|---|---|---|");
    if findings.is_empty() {
        let _ = writeln!(out, "| none | all | RELEASE_GATE_PASS |");
    } else {
        for finding in findings {
            let lane = finding.lane.map_or("global", FeatureLane::as_str);
            let _ = writeln!(
                out,
                "| {:?} | {} | {} |",
                finding.kind, lane, finding.reason_code
            );
        }
    }
    out
}

fn required_lanes() -> [FeatureLane; 4] {
    [
        FeatureLane::HashOnly,
        FeatureLane::Hybrid,
        FeatureLane::Durable,
        FeatureLane::Full,
    ]
}

fn default_lane_gate(lane: FeatureLane) -> FeatureLaneGate {
    let feature_flags = lane.feature_flags();
    let target = format!(
        "/tmp/rch_target_frankensearch_${{AGENT_NAME:-agent}}_bd_pkl0_3_{}",
        lane.as_str().replace('-', "_")
    );
    FeatureLaneGate {
        lane,
        feature_flags: vec![feature_flags.to_owned()],
        cargo_check_command: format!(
            "CARGO_TARGET_DIR={target} RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo check -p frankensearch --lib --no-default-features --features {feature_flags}"
        ),
        behavior_test_command: format!(
            "CARGO_TARGET_DIR={target} RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo test -p frankensearch-fsfs --lib progressive_release_quality_gate -- --nocapture"
        ),
        reason_code: "RELEASE_GATE_LANE_CONFIGURED".to_owned(),
    }
}

fn default_phase_contracts(lane: FeatureLane) -> Vec<PhaseContractGate> {
    vec![
        PhaseContractGate {
            lane,
            phase: PhaseKind::Initial,
            status: GateStatus::Pass,
            reason_code: "RELEASE_GATE_INITIAL_OK".to_owned(),
            preserves_initial_results: true,
            result_count: 10,
        },
        PhaseContractGate {
            lane,
            phase: PhaseKind::Refined,
            status: GateStatus::Pass,
            reason_code: "RELEASE_GATE_REFINED_OK".to_owned(),
            preserves_initial_results: true,
            result_count: 10,
        },
        PhaseContractGate {
            lane,
            phase: PhaseKind::RefinementFailed,
            status: GateStatus::Pass,
            reason_code: "RELEASE_GATE_REFINEMENT_FAILED_PRESERVES_INITIAL".to_owned(),
            preserves_initial_results: true,
            result_count: 10,
        },
    ]
}

fn default_lexical_fallback(lane: FeatureLane) -> LexicalFallbackGate {
    LexicalFallbackGate {
        lane,
        lexical_enabled: true,
        semantic_failure_mode: "quality_timeout".to_owned(),
        initial_results_observed: true,
        hard_failure_observed: false,
        reason_code: "RELEASE_GATE_LEXICAL_FALLBACK_OK".to_owned(),
    }
}

fn default_ordering_gate(lane: FeatureLane) -> OrderingGate {
    OrderingGate {
        lane,
        stable: true,
        tie_break_keys: vec!["score_total_cmp".to_owned(), "doc_id".to_owned()],
        repeated_run_doc_ids: vec![
            format!("{}:doc-alpha", lane.as_str()),
            format!("{}:doc-beta", lane.as_str()),
            format!("{}:doc-gamma", lane.as_str()),
        ],
        reason_code: "RELEASE_GATE_ORDERING_STABLE".to_owned(),
    }
}

fn default_quality_envelopes(lane: FeatureLane) -> Vec<QualityEnvelopeGate> {
    let lane_offset = match lane {
        FeatureLane::HashOnly => 0.0,
        FeatureLane::Hybrid => 0.04,
        FeatureLane::Durable => 0.03,
        FeatureLane::Full => 0.05,
    };

    vec![
        QualityEnvelopeGate {
            lane,
            metric: MetricKind::NdcgAt10,
            observed: 0.76 + lane_offset,
            baseline: 0.75 + lane_offset,
            minimum: 0.70,
            max_regression: 0.02,
            reason_code: "RELEASE_GATE_NDCG_STABLE".to_owned(),
        },
        QualityEnvelopeGate {
            lane,
            metric: MetricKind::Mrr,
            observed: 0.78 + lane_offset,
            baseline: 0.77 + lane_offset,
            minimum: 0.70,
            max_regression: 0.02,
            reason_code: "RELEASE_GATE_MRR_STABLE".to_owned(),
        },
        QualityEnvelopeGate {
            lane,
            metric: MetricKind::RecallAt10,
            observed: 0.84 + lane_offset,
            baseline: 0.83 + lane_offset,
            minimum: 0.75,
            max_regression: 0.02,
            reason_code: "RELEASE_GATE_RECALL_STABLE".to_owned(),
        },
    ]
}

fn is_rch_cargo_command(command: &str) -> bool {
    command.contains("rch exec -- cargo") && command.contains("CARGO_TARGET_DIR")
}

fn finding(
    kind: FindingKind,
    lane: Option<FeatureLane>,
    reason_code: impl Into<String>,
    message: impl Into<String>,
) -> GateFinding {
    GateFinding {
        kind,
        lane,
        reason_code: reason_code.into(),
        message: message.into(),
    }
}

fn usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_release_quality_gate_default_pack_passes() {
        let input = default_release_gate_input("unit-pass");
        let report = ReleaseGatePack::from_input(input);

        assert!(report.passed());
        assert_eq!(report.kind, PROGRESSIVE_RELEASE_GATE_KIND);
        assert_eq!(
            report.schema_version,
            PROGRESSIVE_RELEASE_GATE_SCHEMA_VERSION
        );
        assert_eq!(
            report.matrix_version,
            PROGRESSIVE_RELEASE_GATE_MATRIX_VERSION
        );
        assert_eq!(report.summary.verdict, GateVerdict::Pass);
        assert_eq!(report.summary.lane_count, 4);
        assert_eq!(report.summary.phase_contract_count, 12);
        assert_eq!(report.summary.lexical_fallback_count, 3);
        assert_eq!(report.summary.ordering_contract_count, 4);
        assert_eq!(report.summary.quality_envelope_count, 12);
        assert_eq!(report.summary.artifact_count, 3);
        assert!(report.findings.is_empty());
        assert!(
            report
                .replay_command
                .contains("scripts/check_progressive_release_quality_gate.sh")
        );
        assert!(report.events_jsonl_path.ends_with("gate-events.jsonl"));
        assert!(report.summary_json_path.ends_with("summary.json"));
        assert!(report.summary_markdown_path.ends_with("summary.md"));
        assert!(report.human_summary.contains("| hash-only |"));
        assert!(report.human_summary.contains("RELEASE_GATE_PASS"));

        for lane in &report.input.lanes {
            assert!(lane.cargo_check_command.contains("rch exec -- cargo"));
            assert!(lane.behavior_test_command.contains("rch exec -- cargo"));
        }

        let json = serde_json::to_string_pretty(&report).expect("serialize gate report");
        let roundtrip: ReleaseGatePack =
            serde_json::from_str(&json).expect("deserialize gate report");
        assert_eq!(roundtrip, report);
    }

    #[test]
    fn progressive_release_quality_gate_fails_closed_with_reason_codes() {
        let mut input = default_release_gate_input("unit-fail");
        input.phase_contracts.retain(|contract| {
            !(contract.lane == FeatureLane::Hybrid && contract.phase == PhaseKind::Refined)
        });
        if let Some(fallback) = input
            .lexical_fallback
            .iter_mut()
            .find(|fallback| fallback.lane == FeatureLane::Durable)
        {
            fallback.hard_failure_observed = true;
            fallback.reason_code = "RELEASE_GATE_LEXICAL_HARD_FAILURE".to_owned();
        }
        if let Some(ordering) = input
            .ordering
            .iter_mut()
            .find(|ordering| ordering.lane == FeatureLane::HashOnly)
        {
            ordering.stable = false;
            ordering.reason_code = "RELEASE_GATE_ORDERING_CHANGED".to_owned();
        }
        if let Some(envelope) = input.quality_envelopes.iter_mut().find(|envelope| {
            envelope.lane == FeatureLane::Full && envelope.metric == MetricKind::RecallAt10
        }) {
            envelope.observed = 0.20;
            envelope.reason_code = "RELEASE_GATE_RECALL_REGRESSION".to_owned();
        }
        input
            .artifacts
            .retain(|artifact| artifact.kind != ArtifactKind::SummaryMarkdown);

        let report = ReleaseGatePack::from_input(input);
        assert_eq!(report.summary.verdict, GateVerdict::FailClosed);
        assert!(!report.passed());

        let reason_codes: Vec<&str> = report
            .findings
            .iter()
            .map(|finding| finding.reason_code.as_str())
            .collect();
        assert!(reason_codes.contains(&"RELEASE_GATE_MISSING_PHASE_CONTRACT"));
        assert!(reason_codes.contains(&"RELEASE_GATE_LEXICAL_HARD_FAILURE"));
        assert!(reason_codes.contains(&"RELEASE_GATE_ORDERING_CHANGED"));
        assert!(reason_codes.contains(&"RELEASE_GATE_RECALL_REGRESSION"));
        assert!(reason_codes.contains(&"RELEASE_GATE_MISSING_ARTIFACT"));

        let finding_keys: Vec<(Option<FeatureLane>, FindingKind, &str)> = report
            .findings
            .iter()
            .map(|finding| (finding.lane, finding.kind, finding.reason_code.as_str()))
            .collect();
        let mut sorted_keys = finding_keys.clone();
        sorted_keys.sort_unstable();
        assert_eq!(
            finding_keys, sorted_keys,
            "findings should be deterministic"
        );
        assert!(report.human_summary.contains("FailClosed"));
    }
}
