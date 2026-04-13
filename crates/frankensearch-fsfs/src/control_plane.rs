use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Topic {
    Search,
    Embedding,
    Index,
    Resource,
    Anomaly,
    Lifecycle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FleetSummary {
    pub detected_instances: u32,
    pub healthy_instances: u32,
    pub degraded_instances: u32,
    pub stale_instances: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Started,
    Healthy,
    Degraded,
    Stale,
    Stopped,
    Recovering,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SloStatus {
    Green,
    Yellow,
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstanceHealth {
    pub lifecycle_state: LifecycleState,
    pub slo_status: SloStatus,
    pub error_budget_consumed_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchMetrics {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub qps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingMetrics {
    pub queue_depth: u32,
    pub throughput_eps: f64,
    pub fail_rate_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexMetrics {
    pub docs: u64,
    pub index_bytes: u64,
    pub stale_ratio_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceMetrics {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bps: f64,
    pub io_write_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatestMetrics {
    pub search: SearchMetrics,
    pub embedding: EmbeddingMetrics,
    pub index: IndexMetrics,
    pub resource: ResourceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    None,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnomalySummary {
    pub active_count: u32,
    pub max_severity: Severity,
    pub last_anomaly_ts: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LagMetrics {
    pub ingest_lag_ms_p50: f64,
    pub ingest_lag_ms_p95: f64,
    pub stream_queue_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstanceSnapshot {
    pub instance_id: String,
    pub project_key: String,
    pub host_name: String,
    pub attribution_confidence: f64,
    pub health: InstanceHealth,
    pub latest_metrics: LatestMetrics,
    pub anomaly_summary: AnomalySummary,
    pub lag: LagMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotResponse {
    pub kind: String, // "snapshot_response"
    pub v: u32,       // 1
    pub snapshot_id: String,
    pub generated_ts: String,
    pub fleet_summary: FleetSummary,
    pub instances: Vec<InstanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamSubscribe {
    pub kind: String, // "stream_subscribe"
    pub v: u32,       // 1
    pub client_id: String,
    pub topics: Vec<Topic>,
    pub project_filter: Option<Vec<String>>,
    pub resume_cursor: Option<String>,
    pub max_inflight: u32,
    pub heartbeat_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FrameType {
    Event,
    Control,
    Heartbeat,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventPayload {
    pub event_id: String,
    pub topic: Topic,
    pub instance_id: String,
    pub project_key: String,
    pub root_request_id: String,
    pub body: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlType {
    Backpressure,
    ReconnectAdvisory,
    Sampling,
    TopologyChange,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackpressureState {
    Normal,
    Constrained,
    Dropping,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlPayload {
    pub control_type: ControlType,
    pub backpressure_state: Option<BackpressureState>,
    pub dropped_count_window: Option<u64>,
    pub sampling_ratio: Option<f64>,
    pub retry_after_ms: Option<u32>,
    pub resume_cursor_hint: Option<String>,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeartbeatPayload {
    pub queue_depth: u32,
    pub max_inflight: u32,
    pub unacked: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    pub retry_after_ms: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StreamPayload {
    Event(EventPayload),
    Control(ControlPayload),
    Heartbeat(HeartbeatPayload),
    Error(ErrorPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamFrame {
    pub kind: String, // "stream_frame"
    pub v: u32,       // 1
    pub frame_type: FrameType,
    pub cursor: Option<String>,
    pub producer_ts: Option<String>,
    pub dispatch_ts: String,
    pub lag_ms: u64,
    pub payload: StreamPayload,
}
