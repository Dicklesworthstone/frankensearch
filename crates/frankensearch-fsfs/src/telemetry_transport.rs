use serde::{Deserialize, Serialize};
use serde_json::Number;
use crate::control_plane::Topic;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrimaryTransportConfig {
    #[serde(rename = "type")]
    pub transport_type: String, // "unix_domain_socket"
    pub socket_path_template: String,
    pub framing: String,
    pub supported_codecs: Vec<String>,
    pub auth_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FallbackTransportConfig {
    #[serde(rename = "type")]
    pub transport_type: String, // "jsonl_file"
    pub path_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackpressureConfig {
    pub strategy: String, // "drop_not_block"
    pub drop_counter_required: bool,
    pub max_inflight_min: u32,
    pub block_instance_search: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportLifecycleConfig {
    pub handshake_required: bool,
    pub heartbeat_required: bool,
    pub resume_required: bool,
    pub disconnect_behavior: String, // "graceful_or_retry"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiConsumerConfig {
    pub mode: String, // "fan_out"
    pub max_consumers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportSecurityConfig {
    pub local_only: bool,
    pub network_transport_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportSloTargets {
    pub p95_delivery_lag_ms_target: Number,
    pub throughput_eps_target: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryTransportContractDefinition {
    pub kind: String, // "telemetry_transport_contract_definition"
    pub v: u32,       // 1
    pub primary_transport: PrimaryTransportConfig,
    pub fallback_transport: FallbackTransportConfig,
    pub backpressure: BackpressureConfig,
    pub lifecycle: TransportLifecycleConfig,
    pub multi_consumer: MultiConsumerConfig,
    pub security: TransportSecurityConfig,
    pub slo_targets: TransportSloTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportEndpoint {
    pub kind: String, // "telemetry_transport_endpoint"
    pub v: u32,       // 1
    pub instance_id: String,
    pub project_key: String,
    pub socket_path: String,
    pub fallback_jsonl_path: String,
    pub framing: String,
    pub codec: String,
    pub auth_mode: String,
    pub heartbeat_ms: u32,
    pub max_inflight: u32,
    pub drop_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubscribeFrame {
    pub kind: String, // "telemetry_transport_subscribe"
    pub v: u32,       // 1
    pub connection_id: String,
    pub instance_id: String,
    pub topic_filter: Vec<Topic>,
    pub resume_cursor: Option<String>,
    pub heartbeat_ms: u32,
    pub max_inflight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    UnixDomainSocket,
    JsonlFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<Topic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heartbeat_seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportStreamFrame {
    pub kind: String, // "telemetry_transport_stream_frame"
    pub v: u32,       // 1
    pub frame_type: String, // "event", "control", "heartbeat", "error"
    pub transport: TransportType,
    pub connection_id: String,
    pub sequence: u64,
    pub producer_ts: String,
    pub dispatch_ts: String,
    pub lag_ms: Number,
    pub dropped_since_last: u64,
    pub payload: TransportPayload,
}
