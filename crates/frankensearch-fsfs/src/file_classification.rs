use serde::{Deserialize, Serialize};
use serde_json::Number;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SniffHeuristics {
    pub max_probe_bytes: u32,
    pub binary_byte_threshold_pct: Number,
    pub high_bit_ratio_threshold_pct: Number,
    pub null_byte_hard_binary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EncodingDetector {
    Bom,
    Utf8Validation,
    IcuHeuristic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationPolicy {
    Utf8Nfc,
    Utf8NfcLossy,
    RejectNonText,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnknownEncodingAction {
    Skip,
    LossyDecode,
    Quarantine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EncodingPolicy {
    pub primary_detectors: Vec<EncodingDetector>,
    pub normalization: NormalizationPolicy,
    pub unknown_encoding_action: UnknownEncodingAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TruncatedAction {
    Quarantine,
    Skip,
    IndexPartialWithFlag,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumMismatchAction {
    Quarantine,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorruptPartialPolicy {
    pub truncated_action: TruncatedAction,
    pub checksum_mismatch_action: ChecksumMismatchAction,
    pub max_recovery_prefix_bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfidenceSignals {
    pub min_confidence_for_text: f64,
    pub min_confidence_for_lossy_decode: f64,
    pub emit_required_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileClassificationContractDefinition {
    pub kind: String, // "fsfs_file_classification_contract_definition"
    pub v: u32,       // 1
    pub sniff_heuristics: SniffHeuristics,
    pub encoding_policy: EncodingPolicy,
    pub corrupt_partial_policy: CorruptPartialPolicy,
    pub confidence_signals: ConfidenceSignals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SniffFeatures {
    pub null_bytes: u32,
    pub non_printable_ratio: f64,
    pub high_bit_ratio: f64,
    pub bom: String, // "none", "utf8", "utf16le", "utf16be"
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectedType {
    Text,
    Binary,
    Archive,
    Partial,
    Corrupt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestAction {
    Index,
    Skip,
    Quarantine,
    IndexPartialWithFlag,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownstreamSignals {
    pub utility_penalty: f64,
    pub skip_candidate: bool,
    pub requires_manual_review: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileClassificationDecision {
    pub kind: String, // "fsfs_file_classification_decision"
    pub v: u32,       // 1
    pub path: String,
    pub size_bytes: u64,
    pub probe_bytes: u64,
    pub sniff_features: SniffFeatures,
    pub detected_type: DetectedType,
    pub detected_encoding: String,
    pub normalization_applied: String, // "utf8_nfc", "utf8_nfc_lossy", "none"
    pub ingest_action: IngestAction,
    pub classification_confidence: f64,
    pub encoding_confidence: f64,
    pub reason_code: String,
    pub downstream_signals: DownstreamSignals,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    Truncated,
    ChecksumMismatch,
    DecodeError,
    IoShortRead,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileClassificationCorruptEvent {
    pub kind: String, // "fsfs_file_classification_corrupt_event"
    pub v: u32,       // 1
    pub path: String,
    pub error_class: ErrorClass,
    pub bytes_recovered: u64,
    pub ingest_action: IngestAction,
    pub reason_code: String,
}
