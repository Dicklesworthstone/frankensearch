use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorType {
    DiscoveryFailed,
    StorageError,
    StreamDisconnected,
    SchemaMismatch,
    IngestionOverflow,
    AttributionFailed,
    TelemetryGap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SeverityClass {
    Fatal,
    Degraded,
    Transient,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UiSurface {
    Toast,
    StatusBadge,
    FullScreenPanel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorVariant {
    pub error_type: ErrorType,
    pub default_severity: SeverityClass,
    pub ui_surface: UiSurface,
    pub status_badge: String,
    pub recovery_guidance: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorCatalog {
    pub kind: String, // "control_plane_error_catalog"
    pub v: u32,       // 1
    pub variants: Vec<ErrorVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Correlation {
    pub root_request_id: Option<String>,
    pub parent_event_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMapping {
    pub surface: UiSurface,
    pub status_badge: String,
    pub escalation_after_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecoveryInfo {
    pub retry_policy: String, // "none", "immediate", "exponential_backoff"
    pub operator_steps: Vec<String>,
    pub suggested_commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlPlaneErrorEvent {
    pub kind: String, // "control_plane_error_event"
    pub v: u32,       // 1
    pub ts: String,
    pub event_id: String,
    pub error_type: ErrorType,
    pub severity_class: SeverityClass,
    pub reason_code: String,
    pub message: String,
    pub project_key: String,
    pub instance_id: Option<String>,
    pub correlation: Correlation,
    pub retry_count: u32,
    pub recoverable: bool,
    pub ui_mapping: UiMapping,
    pub recovery: RecoveryInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Window {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErrorAggregation {
    pub kind: String, // "control_plane_error_aggregation"
    pub v: u32,       // 1
    pub window: Window,
    pub error_type: ErrorType,
    pub project_key: String,
    pub instance_id: Option<String>,
    pub reason_code: String,
    pub occurrences: u32,
    pub first_seen_ts: String,
    pub last_seen_ts: String,
    pub escalated: bool,
    pub aggregation_reason_code: String,
}
