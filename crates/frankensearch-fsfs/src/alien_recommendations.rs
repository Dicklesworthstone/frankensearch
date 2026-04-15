use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchemaVersion1;

impl Serialize for SchemaVersion1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(1)
    }
}

impl<'de> Deserialize<'de> for SchemaVersion1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        if value == 1 {
            Ok(Self)
        } else {
            Err(de::Error::invalid_value(
                de::Unexpected::Unsigned(u64::from(value)),
                &"schema version 1",
            ))
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationCardKind {
    #[serde(rename = "fsfs_alien_recommendation_card")]
    Current,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationBundleKind {
    #[serde(rename = "fsfs_alien_recommendation_bundle")]
    Current,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Subsystem {
    IngestionPolicy,
    DegradationScheduler,
    RankingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BudgetedMode {
    pub latency_budget_ms: u32,
    pub memory_budget_mb: u32,
    pub retry_budget: u32,
    pub on_exhaustion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FallbackTrigger {
    pub condition: String,
    pub fallback_action: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IsomorphismProofPlan {
    pub invariants: Vec<String>,
    pub baseline_harness: String,
    pub replay_checks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestField {
    Seed,
    ConfigHash,
    Subsystem,
    PolicyVersion,
    ScenarioId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReproArtifacts {
    pub manifest_fields: Vec<ManifestField>,
    pub artifact_outputs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RollbackPlan {
    pub rollback_command: String,
    pub abort_conditions: Vec<String>,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecommendationCard {
    pub kind: RecommendationCardKind,
    pub v: SchemaVersion1,
    pub subsystem: Subsystem,
    pub ev_score: f64,
    pub priority_tier: String, // "A", "B", "C"
    pub adoption_wedge: String,
    pub budgeted_mode: BudgetedMode,
    pub fallback_trigger: FallbackTrigger,
    pub baseline_comparator: String,
    pub isomorphism_proof_plan: IsomorphismProofPlan,
    pub repro_artifacts: ReproArtifacts,
    pub rollback_plan: RollbackPlan,
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecommendationBundle {
    pub kind: RecommendationBundleKind,
    pub v: SchemaVersion1,
    pub cards: Vec<RecommendationCard>,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{RecommendationBundle, RecommendationCard};

    fn valid_card() -> Value {
        json!({
            "kind": "fsfs_alien_recommendation_card",
            "v": 1,
            "subsystem": "ingestion_policy",
            "ev_score": 3.9,
            "priority_tier": "A",
            "adoption_wedge": "Start with code/docs roots and extend to mixed corpora after stability proof.",
            "budgeted_mode": {
                "latency_budget_ms": 35,
                "memory_budget_mb": 96,
                "retry_budget": 1,
                "on_exhaustion": "Switch low-value sources to deferred ingest queue."
            },
            "fallback_trigger": {
                "condition": "ingest_queue_p95_ms > 250 for 3 windows",
                "fallback_action": "index_later",
                "reason_code": "FSFS_INGEST_QUEUE_PRESSURE_TRIP"
            },
            "baseline_comparator": "Naive immediate indexing for every discovered artifact.",
            "isomorphism_proof_plan": {
                "invariants": ["Deterministic include/exclude decision for identical path snapshots."],
                "baseline_harness": "tests/e2e_ingest_replay.sh",
                "replay_checks": ["decision_trace_equivalence"]
            },
            "repro_artifacts": {
                "manifest_fields": ["seed", "config_hash", "subsystem", "scenario_id"],
                "artifact_outputs": ["ingest_decisions.jsonl", "ingest_summary.json"],
                "replay_command": "cargo test -p frankensearch --test ingest_replay"
            },
            "rollback_plan": {
                "rollback_command": "fsfsctl policy set ingest.naive=true",
                "abort_conditions": ["recall_drop_gt_5pct"]
            }
        })
    }

    #[test]
    fn recommendation_card_rejects_wrong_kind() {
        let mut value = valid_card();
        value["kind"] = json!("wrong_kind");

        let error =
            serde_json::from_value::<RecommendationCard>(value).expect_err("reject bad kind");

        assert!(error.to_string().contains("fsfs_alien_recommendation_card"));
    }

    #[test]
    fn recommendation_card_rejects_wrong_version() {
        let mut value = valid_card();
        value["v"] = json!(2);

        let error =
            serde_json::from_value::<RecommendationCard>(value).expect_err("reject bad version");

        assert!(error.to_string().contains("schema version 1"));
    }

    #[test]
    fn recommendation_bundle_rejects_unknown_fields() {
        let mut value = json!({
            "kind": "fsfs_alien_recommendation_bundle",
            "v": 1,
            "cards": [valid_card()]
        });
        value["extra"] = json!(true);

        let error =
            serde_json::from_value::<RecommendationBundle>(value).expect_err("reject extra field");

        assert!(error.to_string().contains("unknown field `extra`"));
    }
}
