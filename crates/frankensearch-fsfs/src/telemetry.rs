use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstanceIdentity {
    pub instance_id: String,
    pub project_key: String,
    pub host_name: String,
    pub pid: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryCorrelation {
    pub event_id: String,
    pub root_request_id: String,
    pub parent_event_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueryClass {
    Empty,
    Identifier,
    ShortKeyword,
    NaturalLanguage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchPhase {
    Initial,
    Refined,
    RefinementFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchQuery {
    pub text: String,
    pub class: QueryClass,
    pub phase: SearchPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResults {
    pub result_count: u32,
    pub lexical_count: u32,
    pub semantic_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchMetrics {
    pub latency_us: u64,
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "search"
    pub instance: InstanceIdentity,
    pub correlation: TelemetryCorrelation,
    pub query: SearchQuery,
    pub results: SearchResults,
    pub metrics: SearchMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStage {
    Fast,
    Quality,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingJob {
    pub job_id: String,
    pub queue_depth: u32,
    pub doc_count: u32,
    pub stage: EmbeddingStage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbedderTier {
    Hash,
    Fast,
    Quality,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedderIdentity {
    pub id: String,
    pub tier: EmbedderTier,
    pub dimension: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "embedding"
    pub instance: InstanceIdentity,
    pub correlation: TelemetryCorrelation,
    pub job: EmbeddingJob,
    pub embedder: EmbedderIdentity,
    pub status: EmbeddingStatus,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexOperation {
    Build,
    Rebuild,
    Append,
    Compact,
    Repair,
    Snapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexInventory {
    pub words: u64,
    pub tokens: u64,
    pub lines: u64,
    pub bytes: u64,
    pub docs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    Started,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "index"
    pub instance: InstanceIdentity,
    pub correlation: TelemetryCorrelation,
    pub operation: IndexOperation,
    pub inventory: IndexInventory,
    pub dimension: u32,
    pub quantization: String,
    pub status: IndexStatus,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceSample {
    pub cpu_pct: f64,
    pub rss_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub interval_ms: u32,
    pub load_avg_1m: Option<f64>,
    pub pressure_profile: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "resource"
    pub instance: InstanceIdentity,
    pub correlation: TelemetryCorrelation,
    pub sample: ResourceSample,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Started,
    Stopped,
    Healthy,
    Degraded,
    Stale,
    Recovering,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TelemetrySeverity {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "lifecycle"
    pub instance: InstanceIdentity,
    pub correlation: TelemetryCorrelation,
    pub state: LifecycleState,
    pub severity: TelemetrySeverity,
    pub reason: Option<String>,
    pub uptime_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TelemetryPayload {
    Search(SearchEvent),
    Embedding(EmbeddingEvent),
    Index(IndexEvent),
    Resource(ResourceEvent),
    Lifecycle(LifecycleEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryEnvelope {
    pub v: u32,
    pub ts: String,
    pub event: TelemetryPayload,
}
