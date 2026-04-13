use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Window {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "6h")]
    SixHours,
    #[serde(rename = "24h")]
    TwentyFourHours,
    #[serde(rename = "3d")]
    ThreeDays,
    #[serde(rename = "1w")]
    OneWeek,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricId {
    SearchLatencyP95,
    QueryFailureRate,
    StaleIndexLag,
    EmbeddingBacklogAge,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WindowBudget {
    pub budget_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloMetricDefinition {
    pub metric_id: MetricId,
    pub unit: String,
    pub objective_threshold: f64,
    pub objective_bad_ratio: f64,
    pub reason_code_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FormulaCase {
    pub bad_events: u64,
    pub total_events: u64,
    pub expected_bad_ratio: f64,
    pub expected_consumed: f64,
    pub expected_remaining: f64,
    pub expected_burn_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestVector {
    pub metric_id: MetricId,
    pub inputs: HashMap<Window, FormulaCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloContractDefinition {
    pub kind: String, // "slo_contract_definition"
    pub v: u32,       // 1
    pub generated_ts: String,
    pub formula_version: String, // "v1"
    pub windows: HashMap<Window, WindowBudget>,
    pub metrics: Vec<SloMetricDefinition>,
    pub test_vectors: Vec<TestVector>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BaselineMethod {
    RollingMean,
    Ewma,
    SeasonalMedian,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineContext {
    pub method: BaselineMethod,
    pub baseline_value: f64,
    pub lookback_points: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Deviation {
    pub absolute: f64,
    pub relative_pct: f64,
    pub z_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Suppression {
    pub is_suppressed: bool,
    pub policy_id: Option<String>,
    pub until_ts: Option<String>,
    pub suppress_reason_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceBand {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Confidence {
    pub score: f64,
    pub band: ConfidenceBand,
    pub evidence_points: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SloAnomalyEvent {
    pub kind: String, // "slo_anomaly_event"
    pub v: u32,       // 1
    pub ts: String,
    pub event_id: String,
    pub project_key: String,
    pub instance_id: String,
    pub metric_id: MetricId,
    pub window: Window,
    pub reason_code: String,
    pub severity: String, // "info", "warn", "critical"
    pub baseline: BaselineContext,
    pub observed_value: f64,
    pub deviation: Deviation,
    pub suppression: Suppression,
    pub confidence: Confidence,
}
