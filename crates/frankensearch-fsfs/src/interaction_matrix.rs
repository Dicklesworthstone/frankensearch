use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionGatePolicy {
    pub schema: String, // "interaction-matrix-gate-policy-v1"
    pub generated_at: String,
    pub bead: String,
    pub pass_threshold: String,
    pub required_tests: Vec<String>,
    pub required_failure_artifacts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LaneOwnership {
    pub lane_id: String,
    pub owner_lane: String,
    pub bead_refs: Vec<String>,
    pub escalation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionLaneOwnership {
    pub schema: String, // "interaction-lane-ownership-v1"
    pub generated_at: String,
    pub bead: String,
    pub lanes: Vec<LaneOwnership>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionMatrixGateSummary {
    pub schema: String, // "composition-matrix-gate-summary-v1"
    pub generated_at: String,
    pub bead: String,
    pub matrix_anchor: String,
    pub required_fields: Vec<String>,
    pub fallback_contract: String,
    pub required_interaction_tests: Vec<String>,
    pub ownership_artifact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EscalationMetadata {
    pub thread_id: String,
    pub ownership_artifact: String,
    pub summary_contract: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionFailureSummary {
    pub schema: String, // "interaction-failure-summary-v1"
    pub generated_at: String,
    pub bead: String,
    pub workflow: String,
    pub run_url: String,
    pub replay_command: String,
    pub required_artifacts: Vec<String>,
    pub escalation_playbook: String,
    pub escalation_metadata: EscalationMetadata,
}
