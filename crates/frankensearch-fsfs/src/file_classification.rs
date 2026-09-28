use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Number;
use std::collections::BTreeSet;
use std::path::Path;

pub const FILE_CLASSIFICATION_CONTRACT_KIND: &str = "fsfs_file_classification_contract_definition";
pub const FILE_CLASSIFICATION_DECISION_KIND: &str = "fsfs_file_classification_decision";
pub const FILE_CLASSIFICATION_CORRUPT_EVENT_KIND: &str = "fsfs_file_classification_corrupt_event";
pub const FILE_CLASSIFICATION_SCHEMA_VERSION: u32 = 1;

const BOM_NONE: &str = "none";
const BOM_UTF8: &str = "utf8";
const BOM_UTF16_LE: &str = "utf16le";
const BOM_UTF16_BE: &str = "utf16be";
const ENCODING_NONE: &str = "none";
const ENCODING_UTF8: &str = "utf-8";
const ENCODING_UTF16_LE: &str = "utf-16le";
const ENCODING_UTF16_BE: &str = "utf-16be";
const ENCODING_UNKNOWN_8BIT: &str = "unknown-8bit";
const NORMALIZATION_NONE: &str = "none";
const FSFS_TEXT_UTF8_BOM: &str = "FSFS_TEXT_UTF8_BOM";
const FSFS_TEXT_UTF8_HIGH_CONFIDENCE: &str = "FSFS_TEXT_UTF8_HIGH_CONFIDENCE";
const FSFS_TEXT_UTF16_REQUIRES_TRANSCODE: &str = "FSFS_TEXT_UTF16_REQUIRES_TRANSCODE";
const FSFS_TEXT_HEURISTIC_LOSSY_DECODE: &str = "FSFS_TEXT_HEURISTIC_LOSSY_DECODE";
const FSFS_TEXT_HEURISTIC_SKIP: &str = "FSFS_TEXT_HEURISTIC_SKIP";
const FSFS_TEXT_HEURISTIC_QUARANTINE: &str = "FSFS_TEXT_HEURISTIC_QUARANTINE";
const FSFS_BINARY_NULL_BYTE_DETECTED: &str = "FSFS_BINARY_NULL_BYTE_DETECTED";
const FSFS_BINARY_HEURISTIC_THRESHOLD: &str = "FSFS_BINARY_HEURISTIC_THRESHOLD";
const FSFS_ARCHIVE_EXTENSION_BLOCKED: &str = "FSFS_ARCHIVE_EXTENSION_BLOCKED";
const FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY: &str = "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY";
const FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE: &str = "FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE";
const FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE: &str = "FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE";
const FSFS_PARTIAL_HEURISTIC_SKIP: &str = "FSFS_PARTIAL_HEURISTIC_SKIP";
const FSFS_PARTIAL_HEURISTIC_QUARANTINE: &str = "FSFS_PARTIAL_HEURISTIC_QUARANTINE";
const FSFS_CORRUPT_CHECKSUM_MISMATCH: &str = "FSFS_CORRUPT_CHECKSUM_MISMATCH";
const FSFS_CORRUPT_DECODE_ERROR: &str = "FSFS_CORRUPT_DECODE_ERROR";
const FSFS_CORRUPT_IO_SHORT_READ: &str = "FSFS_CORRUPT_IO_SHORT_READ";
pub const FSFS_SENSITIVE_PRIVATE_KEY: &str = "FSFS_SENSITIVE_PRIVATE_KEY";
const CONFIDENCE_SIGNAL_FIELD_CLASSIFICATION_CONFIDENCE: &str = "classification_confidence";
const CONFIDENCE_SIGNAL_FIELD_ENCODING_CONFIDENCE: &str = "encoding_confidence";
const CONFIDENCE_SIGNAL_FIELD_REASON_CODE: &str = "reason_code";
const CONFIDENCE_SIGNAL_FIELD_UTILITY_PENALTY: &str = "utility_penalty";
const CONFIDENCE_SIGNAL_FIELD_SKIP_CANDIDATE: &str = "skip_candidate";
const CONFIDENCE_SIGNAL_FIELD_REQUIRES_MANUAL_REVIEW: &str = "requires_manual_review";
const MIN_PROBE_BYTES: u32 = 256;

const ALLOWED_CONFIDENCE_SIGNAL_FIELDS: [&str; 6] = [
    CONFIDENCE_SIGNAL_FIELD_CLASSIFICATION_CONFIDENCE,
    CONFIDENCE_SIGNAL_FIELD_ENCODING_CONFIDENCE,
    CONFIDENCE_SIGNAL_FIELD_REASON_CODE,
    CONFIDENCE_SIGNAL_FIELD_UTILITY_PENALTY,
    CONFIDENCE_SIGNAL_FIELD_SKIP_CANDIDATE,
    CONFIDENCE_SIGNAL_FIELD_REQUIRES_MANUAL_REVIEW,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SniffHeuristics {
    pub max_probe_bytes: u32,
    pub binary_byte_threshold_pct: Number,
    pub high_bit_ratio_threshold_pct: Number,
    pub null_byte_hard_binary: bool,
}

impl SniffHeuristics {
    #[must_use]
    pub fn binary_byte_threshold_pct(&self) -> f64 {
        self.binary_byte_threshold_pct.as_f64().unwrap_or(30.0)
    }

    #[must_use]
    pub fn high_bit_ratio_threshold_pct(&self) -> f64 {
        self.high_bit_ratio_threshold_pct.as_f64().unwrap_or(60.0)
    }
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

impl NormalizationPolicy {
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Utf8Nfc => "utf8_nfc",
            Self::Utf8NfcLossy => "utf8_nfc_lossy",
            Self::RejectNonText => NORMALIZATION_NONE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnknownEncodingAction {
    Skip,
    LossyDecode,
    Quarantine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
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

impl TruncatedAction {
    #[must_use]
    pub const fn ingest_action(&self) -> IngestAction {
        match self {
            Self::Quarantine => IngestAction::Quarantine,
            Self::Skip => IngestAction::Skip,
            Self::IndexPartialWithFlag => IngestAction::IndexPartialWithFlag,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumMismatchAction {
    Quarantine,
    Skip,
}

impl ChecksumMismatchAction {
    #[must_use]
    pub const fn ingest_action(&self) -> IngestAction {
        match self {
            Self::Quarantine => IngestAction::Quarantine,
            Self::Skip => IngestAction::Skip,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorruptPartialPolicy {
    pub truncated_action: TruncatedAction,
    pub checksum_mismatch_action: ChecksumMismatchAction,
    pub max_recovery_prefix_bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceSignals {
    pub min_confidence_for_text: f64,
    pub min_confidence_for_lossy_decode: f64,
    pub emit_required_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FileClassificationContractDefinition {
    pub kind: String,
    pub v: u32,
    pub sniff_heuristics: SniffHeuristics,
    pub encoding_policy: EncodingPolicy,
    pub corrupt_partial_policy: CorruptPartialPolicy,
    pub confidence_signals: ConfidenceSignals,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileClassificationContractDefinition {
    kind: String,
    v: u32,
    sniff_heuristics: SniffHeuristics,
    encoding_policy: EncodingPolicy,
    corrupt_partial_policy: CorruptPartialPolicy,
    confidence_signals: ConfidenceSignals,
}

impl Default for FileClassificationContractDefinition {
    fn default() -> Self {
        Self {
            kind: FILE_CLASSIFICATION_CONTRACT_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            sniff_heuristics: SniffHeuristics {
                max_probe_bytes: 8_192,
                binary_byte_threshold_pct: Number::from(30),
                high_bit_ratio_threshold_pct: Number::from(60),
                null_byte_hard_binary: true,
            },
            encoding_policy: EncodingPolicy {
                primary_detectors: vec![
                    EncodingDetector::Bom,
                    EncodingDetector::Utf8Validation,
                    EncodingDetector::IcuHeuristic,
                ],
                normalization: NormalizationPolicy::Utf8Nfc,
                unknown_encoding_action: UnknownEncodingAction::Quarantine,
            },
            corrupt_partial_policy: CorruptPartialPolicy {
                truncated_action: TruncatedAction::IndexPartialWithFlag,
                checksum_mismatch_action: ChecksumMismatchAction::Quarantine,
                max_recovery_prefix_bytes: 16_384,
            },
            confidence_signals: ConfidenceSignals {
                min_confidence_for_text: 0.8,
                min_confidence_for_lossy_decode: 0.9,
                emit_required_fields: vec![
                    CONFIDENCE_SIGNAL_FIELD_CLASSIFICATION_CONFIDENCE.to_string(),
                    CONFIDENCE_SIGNAL_FIELD_ENCODING_CONFIDENCE.to_string(),
                    CONFIDENCE_SIGNAL_FIELD_REASON_CODE.to_string(),
                    CONFIDENCE_SIGNAL_FIELD_UTILITY_PENALTY.to_string(),
                    CONFIDENCE_SIGNAL_FIELD_SKIP_CANDIDATE.to_string(),
                    CONFIDENCE_SIGNAL_FIELD_REQUIRES_MANUAL_REVIEW.to_string(),
                ],
            },
        }
    }
}

impl<'de> Deserialize<'de> for FileClassificationContractDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawFileClassificationContractDefinition::deserialize(deserializer)?;
        let contract = Self {
            kind: raw.kind,
            v: raw.v,
            sniff_heuristics: raw.sniff_heuristics,
            encoding_policy: raw.encoding_policy,
            corrupt_partial_policy: raw.corrupt_partial_policy,
            confidence_signals: raw.confidence_signals,
        };
        contract.validate().map_err(de::Error::custom)?;
        Ok(contract)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SniffFeatures {
    pub null_bytes: u32,
    pub non_printable_ratio: f64,
    pub high_bit_ratio: f64,
    pub bom: String,
}

impl SniffFeatures {
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                null_bytes: 0,
                non_printable_ratio: 0.0,
                high_bit_ratio: 0.0,
                bom: BOM_NONE.to_string(),
            };
        }

        // Branchless `u64` counters (saturate-cast to `u32` at the end) instead of
        // per-byte `u32::saturating_add`: a saturating add is not a plain add, so it
        // blocks auto-vectorization of this per-file content scan. The three
        // predicates are independent, so `count += u64::from(cond)` lets LLVM emit
        // three SIMD histograms. Bit-identical: a `u64` counter cannot overflow for
        // any real `&[u8]` (len ≤ usize ≤ u64), and the final
        // `u32::try_from(..).unwrap_or(u32::MAX)` reproduces the same `u32::MAX`
        // saturation the old per-byte `saturating_add` produced.
        let mut null_bytes = 0_u64;
        let mut non_printable = 0_u64;
        let mut high_bit = 0_u64;

        for &byte in bytes {
            null_bytes += u64::from(byte == 0);
            non_printable += u64::from(is_non_printable(byte));
            high_bit += u64::from(byte >= 0x80);
        }

        let null_bytes = u32::try_from(null_bytes).unwrap_or(u32::MAX);
        let non_printable = u32::try_from(non_printable).unwrap_or(u32::MAX);
        let high_bit = u32::try_from(high_bit).unwrap_or(u32::MAX);

        let sample_len = bytes.len() as f64;
        Self {
            null_bytes,
            non_printable_ratio: f64::from(non_printable) / sample_len,
            high_bit_ratio: f64::from(high_bit) / sample_len,
            bom: detect_bom(bytes).to_string(),
        }
    }
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
#[serde(deny_unknown_fields)]
pub struct DownstreamSignals {
    pub utility_penalty: f64,
    pub skip_candidate: bool,
    pub requires_manual_review: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FileClassificationDecision {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub size_bytes: u64,
    pub probe_bytes: u64,
    pub sniff_features: SniffFeatures,
    pub detected_type: DetectedType,
    pub detected_encoding: String,
    pub normalization_applied: String,
    pub ingest_action: IngestAction,
    pub classification_confidence: f64,
    pub encoding_confidence: f64,
    pub reason_code: String,
    pub downstream_signals: DownstreamSignals,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileClassificationDecision {
    kind: String,
    v: u32,
    path: String,
    size_bytes: u64,
    probe_bytes: u64,
    sniff_features: SniffFeatures,
    detected_type: DetectedType,
    detected_encoding: String,
    normalization_applied: String,
    ingest_action: IngestAction,
    classification_confidence: f64,
    encoding_confidence: f64,
    reason_code: String,
    downstream_signals: DownstreamSignals,
}

impl<'de> Deserialize<'de> for FileClassificationDecision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawFileClassificationDecision::deserialize(deserializer)?;
        let decision = Self {
            kind: raw.kind,
            v: raw.v,
            path: raw.path,
            size_bytes: raw.size_bytes,
            probe_bytes: raw.probe_bytes,
            sniff_features: raw.sniff_features,
            detected_type: raw.detected_type,
            detected_encoding: raw.detected_encoding,
            normalization_applied: raw.normalization_applied,
            ingest_action: raw.ingest_action,
            classification_confidence: raw.classification_confidence,
            encoding_confidence: raw.encoding_confidence,
            reason_code: raw.reason_code,
            downstream_signals: raw.downstream_signals,
        };
        decision.validate().map_err(de::Error::custom)?;
        Ok(decision)
    }
}

impl FileClassificationDecision {
    #[must_use]
    pub fn satisfies_contract(&self) -> bool {
        self.validate().is_ok()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    Truncated,
    ChecksumMismatch,
    DecodeError,
    IoShortRead,
}

impl ErrorClass {
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Truncated => FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY,
            Self::ChecksumMismatch => FSFS_CORRUPT_CHECKSUM_MISMATCH,
            Self::DecodeError => FSFS_CORRUPT_DECODE_ERROR,
            Self::IoShortRead => FSFS_CORRUPT_IO_SHORT_READ,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FileClassificationCorruptEvent {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub error_class: ErrorClass,
    pub bytes_recovered: u64,
    pub ingest_action: IngestAction,
    pub reason_code: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFileClassificationCorruptEvent {
    kind: String,
    v: u32,
    path: String,
    error_class: ErrorClass,
    bytes_recovered: u64,
    ingest_action: IngestAction,
    reason_code: String,
}

impl<'de> Deserialize<'de> for FileClassificationCorruptEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawFileClassificationCorruptEvent::deserialize(deserializer)?;
        let event = Self {
            kind: raw.kind,
            v: raw.v,
            path: raw.path,
            error_class: raw.error_class,
            bytes_recovered: raw.bytes_recovered,
            ingest_action: raw.ingest_action,
            reason_code: raw.reason_code,
        };
        event.validate().map_err(de::Error::custom)?;
        Ok(event)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityState {
    Clean,
    Truncated,
    ChecksumMismatch,
    DecodeError,
    IoShortRead,
}

impl IntegrityState {
    #[must_use]
    pub const fn error_class(self) -> Option<ErrorClass> {
        match self {
            Self::Clean => None,
            Self::Truncated => Some(ErrorClass::Truncated),
            Self::ChecksumMismatch => Some(ErrorClass::ChecksumMismatch),
            Self::DecodeError => Some(ErrorClass::DecodeError),
            Self::IoShortRead => Some(ErrorClass::IoShortRead),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileClassificationInput<'a> {
    pub path: &'a str,
    pub size_bytes: u64,
    pub probe: &'a [u8],
    pub integrity: IntegrityState,
}

impl<'a> FileClassificationInput<'a> {
    #[must_use]
    pub fn from_bytes(path: &'a str, bytes: &'a [u8]) -> Self {
        Self {
            path,
            size_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            probe: bytes,
            integrity: IntegrityState::Clean,
        }
    }

    #[must_use]
    pub fn with_integrity(path: &'a str, bytes: &'a [u8], integrity: IntegrityState) -> Self {
        Self {
            integrity,
            ..Self::from_bytes(path, bytes)
        }
    }
}

impl FileClassificationContractDefinition {
    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            FILE_CLASSIFICATION_CONTRACT_KIND,
            "kind must be fsfs_file_classification_contract_definition",
        )?;
        validate_schema_version(self.v)?;
        self.sniff_heuristics.validate()?;
        self.encoding_policy.validate()?;
        self.confidence_signals.validate()?;
        Ok(())
    }

    #[must_use]
    pub fn classify(&self, input: FileClassificationInput<'_>) -> FileClassificationDecision {
        let capped_probe = self.cap_probe(input.probe);
        let sniff_features = SniffFeatures::from_bytes(capped_probe);
        let probe_len = u64::try_from(capped_probe.len()).unwrap_or(u64::MAX);

        if let Some(error_class) = input.integrity.error_class() {
            if !matches!(error_class, ErrorClass::Truncated) {
                return self.corrupt_decision(input, probe_len, sniff_features, error_class);
            }
        }

        if is_archive_path(input.path) {
            return FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Archive,
                detected_encoding: ENCODING_NONE.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Skip,
                classification_confidence: 0.99,
                encoding_confidence: 0.0,
                reason_code: FSFS_ARCHIVE_EXTENSION_BLOCKED.to_string(),
                downstream_signals: downstream(0.95, true, false),
            };
        }

        let utf8_valid = std::str::from_utf8(capped_probe).is_ok();
        if self.is_binary(&sniff_features, utf8_valid) {
            return FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features: sniff_features.clone(),
                detected_type: DetectedType::Binary,
                detected_encoding: ENCODING_NONE.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Skip,
                classification_confidence: if sniff_features.null_bytes > 0 {
                    0.99
                } else {
                    0.9
                },
                encoding_confidence: 0.0,
                reason_code: if sniff_features.null_bytes > 0 {
                    FSFS_BINARY_NULL_BYTE_DETECTED.to_string()
                } else {
                    FSFS_BINARY_HEURISTIC_THRESHOLD.to_string()
                },
                downstream_signals: downstream(0.9, true, false),
            };
        }

        // Private keys are a sensitive class: the scope-privacy contract
        // forbids persisting, emitting or displaying their content.
        if contains_private_key_armor(capped_probe) {
            return FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Text,
                detected_encoding: if utf8_valid { "utf-8" } else { "8bit" }.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Skip,
                classification_confidence: 0.99,
                encoding_confidence: if utf8_valid { 1.0 } else { 0.5 },
                reason_code: FSFS_SENSITIVE_PRIVATE_KEY.to_string(),
                downstream_signals: downstream(1.0, true, false),
            };
        }

        let encoding = detect_encoding(capped_probe, &sniff_features, utf8_valid);
        if matches!(input.integrity, IntegrityState::Truncated) {
            return self.partial_decision(input, probe_len, sniff_features, encoding);
        }

        self.text_decision(input, probe_len, sniff_features, encoding)
    }

    #[must_use]
    pub fn classify_bytes(&self, path: &str, bytes: &[u8]) -> FileClassificationDecision {
        self.classify(FileClassificationInput::from_bytes(path, bytes))
    }

    #[must_use]
    pub fn classify_with_integrity(
        &self,
        path: &str,
        bytes: &[u8],
        integrity: IntegrityState,
    ) -> FileClassificationDecision {
        self.classify(FileClassificationInput::with_integrity(
            path, bytes, integrity,
        ))
    }

    #[must_use]
    pub fn build_corrupt_event(
        &self,
        path: impl Into<String>,
        error_class: ErrorClass,
        bytes_recovered: u64,
    ) -> FileClassificationCorruptEvent {
        let ingest_action = match error_class {
            ErrorClass::Truncated => self.corrupt_partial_policy.truncated_action.ingest_action(),
            ErrorClass::ChecksumMismatch => self
                .corrupt_partial_policy
                .checksum_mismatch_action
                .ingest_action(),
            ErrorClass::DecodeError | ErrorClass::IoShortRead => IngestAction::Quarantine,
        };

        FileClassificationCorruptEvent {
            kind: FILE_CLASSIFICATION_CORRUPT_EVENT_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            path: path.into(),
            error_class,
            bytes_recovered,
            ingest_action,
            reason_code: error_class.reason_code().to_string(),
        }
    }

    #[must_use]
    fn cap_probe<'a>(&self, probe: &'a [u8]) -> &'a [u8] {
        let max_probe =
            usize::try_from(self.sniff_heuristics.max_probe_bytes).unwrap_or(usize::MAX);
        &probe[..probe.len().min(max_probe)]
    }

    #[must_use]
    fn is_binary(&self, sniff_features: &SniffFeatures, utf8_valid: bool) -> bool {
        if self.sniff_heuristics.null_byte_hard_binary && sniff_features.null_bytes > 0 {
            return true;
        }

        let non_printable_pct = sniff_features.non_printable_ratio * 100.0;
        if non_printable_pct >= self.sniff_heuristics.binary_byte_threshold_pct() {
            return true;
        }

        let high_bit_pct = sniff_features.high_bit_ratio * 100.0;
        !utf8_valid && high_bit_pct >= self.sniff_heuristics.high_bit_ratio_threshold_pct()
    }

    #[must_use]
    fn corrupt_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        error_class: ErrorClass,
    ) -> FileClassificationDecision {
        let ingest_action = match error_class {
            ErrorClass::ChecksumMismatch => self
                .corrupt_partial_policy
                .checksum_mismatch_action
                .ingest_action(),
            ErrorClass::DecodeError | ErrorClass::IoShortRead => IngestAction::Quarantine,
            ErrorClass::Truncated => self.corrupt_partial_policy.truncated_action.ingest_action(),
        };

        FileClassificationDecision {
            kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            path: input.path.to_string(),
            size_bytes: input.size_bytes,
            probe_bytes: probe_len,
            sniff_features,
            detected_type: DetectedType::Corrupt,
            detected_encoding: ENCODING_NONE.to_string(),
            normalization_applied: NORMALIZATION_NONE.to_string(),
            ingest_action,
            classification_confidence: 0.95,
            encoding_confidence: 0.0,
            reason_code: error_class.reason_code().to_string(),
            downstream_signals: downstream(1.0, true, true),
        }
    }

    #[must_use]
    fn partial_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        encoding: EncodingAssessment,
    ) -> FileClassificationDecision {
        match encoding {
            EncodingAssessment::Utf8 { bom } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Partial,
                detected_encoding: ENCODING_UTF8.to_string(),
                normalization_applied: self.encoding_policy.normalization.label().to_string(),
                ingest_action: self.corrupt_partial_policy.truncated_action.ingest_action(),
                classification_confidence: clamp_unit(
                    self.confidence_signals.min_confidence_for_text.max(0.82),
                ),
                encoding_confidence: if bom == BOM_UTF8 { 1.0 } else { 0.98 },
                reason_code: FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY.to_string(),
                downstream_signals: downstream(
                    if matches!(
                        self.corrupt_partial_policy.truncated_action,
                        TruncatedAction::IndexPartialWithFlag
                    ) {
                        0.35
                    } else {
                        0.8
                    },
                    true,
                    true,
                ),
            },
            EncodingAssessment::Utf16 { label } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Partial,
                detected_encoding: label.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Quarantine,
                classification_confidence: 0.9,
                encoding_confidence: 0.99,
                reason_code: FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE.to_string(),
                downstream_signals: downstream(0.8, true, true),
            },
            EncodingAssessment::Heuristic8Bit { confidence } => {
                let lossy_allowed =
                    confidence >= self.confidence_signals.min_confidence_for_lossy_decode;
                let (ingest_action, normalization_applied, reason_code) =
                    match self.encoding_policy.unknown_encoding_action {
                        UnknownEncodingAction::LossyDecode if lossy_allowed => (
                            self.corrupt_partial_policy.truncated_action.ingest_action(),
                            NormalizationPolicy::Utf8NfcLossy.label().to_string(),
                            FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE.to_string(),
                        ),
                        UnknownEncodingAction::LossyDecode | UnknownEncodingAction::Quarantine => (
                            IngestAction::Quarantine,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_PARTIAL_HEURISTIC_QUARANTINE.to_string(),
                        ),
                        UnknownEncodingAction::Skip => (
                            IngestAction::Skip,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_PARTIAL_HEURISTIC_SKIP.to_string(),
                        ),
                    };

                FileClassificationDecision {
                    kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                    v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                    path: input.path.to_string(),
                    size_bytes: input.size_bytes,
                    probe_bytes: probe_len,
                    sniff_features,
                    detected_type: DetectedType::Partial,
                    detected_encoding: ENCODING_UNKNOWN_8BIT.to_string(),
                    normalization_applied,
                    ingest_action,
                    classification_confidence: clamp_unit(confidence.max(0.7)),
                    encoding_confidence: confidence,
                    reason_code,
                    downstream_signals: downstream(0.65, true, true),
                }
            }
        }
    }

    #[must_use]
    fn text_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        encoding: EncodingAssessment,
    ) -> FileClassificationDecision {
        match encoding {
            EncodingAssessment::Utf8 { bom } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Text,
                detected_encoding: ENCODING_UTF8.to_string(),
                normalization_applied: self.encoding_policy.normalization.label().to_string(),
                ingest_action: IngestAction::Index,
                classification_confidence: clamp_unit(
                    self.confidence_signals.min_confidence_for_text.max(0.95),
                ),
                encoding_confidence: if bom == BOM_UTF8 { 1.0 } else { 0.98 },
                reason_code: if bom == BOM_UTF8 {
                    FSFS_TEXT_UTF8_BOM.to_string()
                } else {
                    FSFS_TEXT_UTF8_HIGH_CONFIDENCE.to_string()
                },
                downstream_signals: downstream(0.02, false, false),
            },
            EncodingAssessment::Utf16 { label } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Text,
                detected_encoding: label.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Quarantine,
                classification_confidence: 0.92,
                encoding_confidence: 0.99,
                reason_code: FSFS_TEXT_UTF16_REQUIRES_TRANSCODE.to_string(),
                downstream_signals: downstream(0.7, true, true),
            },
            EncodingAssessment::Heuristic8Bit { confidence } => {
                let lossy_allowed =
                    confidence >= self.confidence_signals.min_confidence_for_lossy_decode;
                let (ingest_action, normalization_applied, reason_code, utility_penalty) =
                    match self.encoding_policy.unknown_encoding_action {
                        UnknownEncodingAction::LossyDecode if lossy_allowed => (
                            IngestAction::Index,
                            NormalizationPolicy::Utf8NfcLossy.label().to_string(),
                            FSFS_TEXT_HEURISTIC_LOSSY_DECODE.to_string(),
                            0.25,
                        ),
                        UnknownEncodingAction::LossyDecode | UnknownEncodingAction::Quarantine => (
                            IngestAction::Quarantine,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_TEXT_HEURISTIC_QUARANTINE.to_string(),
                            0.75,
                        ),
                        UnknownEncodingAction::Skip => (
                            IngestAction::Skip,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_TEXT_HEURISTIC_SKIP.to_string(),
                            0.85,
                        ),
                    };

                FileClassificationDecision {
                    kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                    v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                    path: input.path.to_string(),
                    size_bytes: input.size_bytes,
                    probe_bytes: probe_len,
                    sniff_features,
                    detected_type: DetectedType::Text,
                    detected_encoding: ENCODING_UNKNOWN_8BIT.to_string(),
                    normalization_applied,
                    ingest_action: ingest_action.clone(),
                    classification_confidence: clamp_unit(confidence.max(0.8)),
                    encoding_confidence: confidence,
                    reason_code,
                    downstream_signals: downstream(
                        utility_penalty,
                        !matches!(ingest_action, IngestAction::Index),
                        !matches!(ingest_action, IngestAction::Index),
                    ),
                }
            }
        }
    }
}

impl SniffHeuristics {
    fn validate(&self) -> Result<(), &'static str> {
        if self.max_probe_bytes < MIN_PROBE_BYTES {
            return Err("sniff_heuristics.max_probe_bytes must be >= 256");
        }
        validate_percentage(
            self.binary_byte_threshold_pct.as_f64(),
            "sniff_heuristics.binary_byte_threshold_pct must be between 0 and 100",
        )?;
        validate_percentage(
            self.high_bit_ratio_threshold_pct.as_f64(),
            "sniff_heuristics.high_bit_ratio_threshold_pct must be between 0 and 100",
        )?;
        if !self.null_byte_hard_binary {
            return Err("sniff_heuristics.null_byte_hard_binary must be true");
        }
        Ok(())
    }
}

impl EncodingPolicy {
    fn validate(&self) -> Result<(), &'static str> {
        if self.primary_detectors.len() < 2 {
            return Err("encoding_policy.primary_detectors must contain at least two detectors");
        }
        if self.primary_detectors[0] != EncodingDetector::Bom {
            return Err("encoding_policy.primary_detectors[0] must be bom");
        }
        if self.primary_detectors[1] != EncodingDetector::Utf8Validation {
            return Err("encoding_policy.primary_detectors[1] must be utf8_validation");
        }
        Ok(())
    }
}

impl ConfidenceSignals {
    fn validate(&self) -> Result<(), &'static str> {
        validate_unit_interval(
            self.min_confidence_for_text,
            "confidence_signals.min_confidence_for_text must be between 0 and 1",
        )?;
        validate_unit_interval(
            self.min_confidence_for_lossy_decode,
            "confidence_signals.min_confidence_for_lossy_decode must be between 0 and 1",
        )?;
        if self.emit_required_fields.len() < 4 {
            return Err(
                "confidence_signals.emit_required_fields must contain at least four fields",
            );
        }

        let mut unique_fields = BTreeSet::new();
        for field in &self.emit_required_fields {
            if !ALLOWED_CONFIDENCE_SIGNAL_FIELDS.contains(&field.as_str()) {
                return Err(
                    "confidence_signals.emit_required_fields contains an unsupported field",
                );
            }
            if !unique_fields.insert(field.as_str()) {
                return Err("confidence_signals.emit_required_fields must be unique");
            }
        }

        Ok(())
    }
}

impl SniffFeatures {
    fn validate(&self) -> Result<(), &'static str> {
        validate_unit_interval(
            self.non_printable_ratio,
            "sniff_features.non_printable_ratio must be between 0 and 1",
        )?;
        validate_unit_interval(
            self.high_bit_ratio,
            "sniff_features.high_bit_ratio must be between 0 and 1",
        )?;
        if !matches!(
            self.bom.as_str(),
            BOM_NONE | BOM_UTF8 | BOM_UTF16_LE | BOM_UTF16_BE
        ) {
            return Err("sniff_features.bom must be one of none|utf8|utf16le|utf16be");
        }
        Ok(())
    }
}

impl DownstreamSignals {
    fn validate(&self) -> Result<(), &'static str> {
        validate_unit_interval(
            self.utility_penalty,
            "downstream_signals.utility_penalty must be between 0 and 1",
        )
    }
}

impl FileClassificationDecision {
    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            FILE_CLASSIFICATION_DECISION_KIND,
            "kind must be fsfs_file_classification_decision",
        )?;
        validate_schema_version(self.v)?;
        if self.path.is_empty() {
            return Err("path must not be empty");
        }
        self.sniff_features.validate()?;
        validate_unit_interval(
            self.classification_confidence,
            "classification_confidence must be between 0 and 1",
        )?;
        validate_unit_interval(
            self.encoding_confidence,
            "encoding_confidence must be between 0 and 1",
        )?;
        if !is_reason_code(&self.reason_code) {
            return Err("reason_code must match ^FSFS_[A-Z0-9_]+$");
        }
        if !is_allowed_detected_encoding(&self.detected_encoding) {
            return Err("detected_encoding must match the schema contract");
        }
        if !matches!(
            self.normalization_applied.as_str(),
            "utf8_nfc" | "utf8_nfc_lossy" | NORMALIZATION_NONE
        ) {
            return Err("normalization_applied must match the schema contract");
        }
        self.downstream_signals.validate()?;

        let encoding_is_none = self.detected_encoding == ENCODING_NONE;
        let normalization_is_none = self.normalization_applied == NORMALIZATION_NONE;
        match self.detected_type {
            DetectedType::Binary => {
                if !encoding_is_none || !normalization_is_none {
                    return Err("binary decisions must use none encoding and normalization");
                }
            }
            DetectedType::Archive | DetectedType::Corrupt => {
                if !encoding_is_none || !normalization_is_none {
                    return Err(
                        "archive/corrupt decisions must use none encoding and normalization",
                    );
                }
                if !matches!(
                    self.ingest_action,
                    IngestAction::Skip | IngestAction::Quarantine
                ) {
                    return Err("archive/corrupt decisions must skip or quarantine");
                }
            }
            DetectedType::Partial => {
                if encoding_is_none {
                    return Err("partial decisions must provide a detected encoding");
                }
                if matches!(self.ingest_action, IngestAction::Index) {
                    return Err("partial decisions must not use ingest_action=index");
                }
            }
            DetectedType::Text => {
                if encoding_is_none {
                    return Err("text decisions must provide a detected encoding");
                }
            }
        }

        Ok(())
    }
}

impl FileClassificationCorruptEvent {
    fn validate(&self) -> Result<(), &'static str> {
        validate_kind(
            &self.kind,
            FILE_CLASSIFICATION_CORRUPT_EVENT_KIND,
            "kind must be fsfs_file_classification_corrupt_event",
        )?;
        validate_schema_version(self.v)?;
        if self.path.is_empty() {
            return Err("path must not be empty");
        }
        if !is_reason_code(&self.reason_code) {
            return Err("reason_code must match ^FSFS_[A-Z0-9_]+$");
        }
        if matches!(self.ingest_action, IngestAction::Index) {
            return Err("corrupt events must not use ingest_action=index");
        }
        if !matches!(
            (self.error_class, &self.ingest_action),
            (
                ErrorClass::Truncated,
                IngestAction::Skip | IngestAction::Quarantine | IngestAction::IndexPartialWithFlag
            ) | (
                ErrorClass::ChecksumMismatch | ErrorClass::DecodeError | ErrorClass::IoShortRead,
                IngestAction::Skip | IngestAction::Quarantine
            )
        ) {
            return Err("corrupt event ingest_action does not satisfy the schema contract");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EncodingAssessment {
    Utf8 { bom: &'static str },
    Utf16 { label: &'static str },
    Heuristic8Bit { confidence: f64 },
}

#[must_use]
fn is_non_printable(byte: u8) -> bool {
    !matches!(byte, b'\t' | b'\n' | b'\r') && (byte < 0x20 || byte == 0x7f)
}

#[must_use]
fn detect_bom(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        BOM_UTF8
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        BOM_UTF16_LE
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        BOM_UTF16_BE
    } else {
        BOM_NONE
    }
}

#[must_use]
fn detect_encoding(
    _bytes: &[u8],
    sniff_features: &SniffFeatures,
    utf8_valid: bool,
) -> EncodingAssessment {
    match sniff_features.bom.as_str() {
        BOM_UTF16_LE => EncodingAssessment::Utf16 {
            label: ENCODING_UTF16_LE,
        },
        BOM_UTF16_BE => EncodingAssessment::Utf16 {
            label: ENCODING_UTF16_BE,
        },
        BOM_UTF8 if utf8_valid => EncodingAssessment::Utf8 { bom: BOM_UTF8 },
        _ if utf8_valid => EncodingAssessment::Utf8 { bom: BOM_NONE },
        _ => {
            let confidence = clamp_unit(
                1.0 - (sniff_features.non_printable_ratio * 0.35)
                    - (sniff_features.high_bit_ratio * 0.1),
            );
            EncodingAssessment::Heuristic8Bit { confidence }
        }
    }
}

#[must_use]
fn downstream(
    utility_penalty: f64,
    skip_candidate: bool,
    requires_manual_review: bool,
) -> DownstreamSignals {
    DownstreamSignals {
        utility_penalty,
        skip_candidate,
        requires_manual_review,
    }
}

#[must_use]
fn clamp_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn validate_schema_version(value: u32) -> Result<(), &'static str> {
    if value == FILE_CLASSIFICATION_SCHEMA_VERSION {
        Ok(())
    } else {
        Err("schema version 1")
    }
}

fn validate_kind(actual: &str, expected: &str, message: &'static str) -> Result<(), &'static str> {
    if actual == expected {
        Ok(())
    } else {
        Err(message)
    }
}

fn validate_unit_interval(value: f64, message: &'static str) -> Result<(), &'static str> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(message)
    }
}

fn validate_percentage(value: Option<f64>, message: &'static str) -> Result<(), &'static str> {
    match value {
        Some(value) if (0.0..=100.0).contains(&value) => Ok(()),
        _ => Err(message),
    }
}

fn is_reason_code(value: &str) -> bool {
    value.len() > 5
        && value.starts_with("FSFS_")
        && value
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_allowed_detected_encoding(value: &str) -> bool {
    value == ENCODING_NONE
        || (!value.is_empty()
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')))
}

#[must_use]
/// Whether the bytes hold PEM or PGP private-key material: a line starting
/// with `-----BEGIN … PRIVATE KEY` followed, after at most a few armor header
/// lines, by a base64 body line. Source code that only names the header (a
/// PEM parser's constant) is not key material and stays indexable.
fn contains_private_key_armor(probe: &[u8]) -> bool {
    const BEGIN: &[u8] = b"-----BEGIN ";
    const PRIVATE_KEY: &[u8] = b"PRIVATE KEY";
    const MIN_BODY_LINE: usize = 32;
    let mut lines = probe.split(|&byte| byte == b'\n');
    while let Some(line) = lines.next() {
        let line = line.trim_ascii();
        let is_header = line.starts_with(BEGIN)
            && line
                .windows(PRIVATE_KEY.len())
                .any(|window| window == PRIVATE_KEY);
        if !is_header {
            continue;
        }
        // PEM `Proc-Type:`/`DEK-Info:` and PGP `Version:`/`Comment:` headers
        // and a blank separator may precede the body.
        for body in lines.by_ref().take(4) {
            let body = body.trim_ascii();
            if body.is_empty() || body.contains(&b':') {
                continue;
            }
            if body.len() >= MIN_BODY_LINE
                && body
                    .iter()
                    .all(|&byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
            {
                return true;
            }
            break;
        }
    }
    false
}

fn is_archive_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if [
        ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".tbz2", ".txz",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
    {
        return true;
    }

    Path::new(&lower)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension,
                "7z" | "apk"
                    | "bz2"
                    | "crate"
                    | "dmg"
                    | "ear"
                    | "gz"
                    | "iso"
                    | "jar"
                    | "pkg"
                    | "rar"
                    | "tar"
                    | "war"
                    | "whl"
                    | "xz"
                    | "zip"
                    | "zst"
            )
        })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ChecksumMismatchAction, DetectedType, ErrorClass, FILE_CLASSIFICATION_CONTRACT_KIND,
        FILE_CLASSIFICATION_CORRUPT_EVENT_KIND, FILE_CLASSIFICATION_DECISION_KIND,
        FILE_CLASSIFICATION_SCHEMA_VERSION, FSFS_SENSITIVE_PRIVATE_KEY,
        FileClassificationContractDefinition, FileClassificationDecision, IngestAction,
        IntegrityState, NormalizationPolicy, TruncatedAction,
    };

    use super::{SniffFeatures, is_non_printable};

    fn sniff_reference(bytes: &[u8]) -> (u32, f64, f64) {
        if bytes.is_empty() {
            return (0, 0.0, 0.0);
        }
        let mut null_bytes = 0_u32;
        let mut non_printable = 0_u32;
        let mut high_bit = 0_u32;
        for &byte in bytes {
            if byte == 0 {
                null_bytes = null_bytes.saturating_add(1);
            }
            if is_non_printable(byte) {
                non_printable = non_printable.saturating_add(1);
            }
            if byte >= 0x80 {
                high_bit = high_bit.saturating_add(1);
            }
        }
        let len = bytes.len() as f64;
        (
            null_bytes,
            f64::from(non_printable) / len,
            f64::from(high_bit) / len,
        )
    }

    #[test]
    fn sniff_features_vectorized_matches_scalar_reference() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0u8; 64],
            vec![0xFFu8; 100],
            b"hello world\tplain\ntext\r".to_vec(),
            (0u8..=255).cycle().take(4099).collect(),
            (0u8..=255).collect(),
        ];
        for bytes in &cases {
            let got = SniffFeatures::from_bytes(bytes);
            let (rn, rnp, rhb) = sniff_reference(bytes);
            assert_eq!(
                got.null_bytes,
                rn,
                "null_bytes mismatch (len={})",
                bytes.len()
            );
            assert_eq!(
                got.non_printable_ratio.to_bits(),
                rnp.to_bits(),
                "np ratio mismatch (len={})",
                bytes.len()
            );
            assert_eq!(
                got.high_bit_ratio.to_bits(),
                rhb.to_bits(),
                "hb ratio mismatch (len={})",
                bytes.len()
            );
        }
    }

    #[test]
    fn default_contract_matches_documented_defaults() {
        let contract = FileClassificationContractDefinition::default();
        assert_eq!(contract.kind, FILE_CLASSIFICATION_CONTRACT_KIND);
        assert_eq!(contract.v, FILE_CLASSIFICATION_SCHEMA_VERSION);
        assert_eq!(contract.sniff_heuristics.max_probe_bytes, 8_192);
        assert_eq!(
            contract.sniff_heuristics.binary_byte_threshold_pct.as_u64(),
            Some(30)
        );
        assert_eq!(
            contract
                .sniff_heuristics
                .high_bit_ratio_threshold_pct
                .as_u64(),
            Some(60)
        );
        assert!(contract.sniff_heuristics.null_byte_hard_binary);
        assert_eq!(
            contract.encoding_policy.normalization,
            NormalizationPolicy::Utf8Nfc
        );
        assert_eq!(
            contract.corrupt_partial_policy.truncated_action,
            TruncatedAction::IndexPartialWithFlag
        );
        assert_eq!(
            contract.corrupt_partial_policy.checksum_mismatch_action,
            ChecksumMismatchAction::Quarantine
        );
    }

    #[test]
    fn classify_utf8_text_indexes_normally() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/src/lib.rs",
            b"pub fn score() {\n    println!(\"hello\");\n}\n",
        );

        assert_eq!(decision.kind, FILE_CLASSIFICATION_DECISION_KIND);
        assert_eq!(decision.detected_type, DetectedType::Text);
        assert_eq!(decision.detected_encoding, "utf-8");
        assert_eq!(decision.normalization_applied, "utf8_nfc");
        assert_eq!(decision.ingest_action, IngestAction::Index);
        assert_eq!(decision.reason_code, "FSFS_TEXT_UTF8_HIGH_CONFIDENCE");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_private_key_material_skips_but_pem_parsers_index() {
        let contract = FileClassificationContractDefinition::default();
        let body = "MIIEowIBAAKCAQEAwl8KX2rUqbD9Qm2p0xa7Ubq3Qjp8Ksb3S4DbBqvAbRTsrMpk\n";
        let keys = [
            format!("-----BEGIN RSA PRIVATE KEY-----\n{body}-----END RSA PRIVATE KEY-----\n"),
            format!("-----BEGIN OPENSSH PRIVATE KEY-----\n{body}"),
            format!(
                "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,00\n\n{body}"
            ),
            format!("notes\n  -----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: GnuPG\n\n{body}"),
        ];
        for key in &keys {
            let decision = contract.classify_bytes("/workspace/deploy/server.pem", key.as_bytes());
            assert_eq!(decision.ingest_action, IngestAction::Skip, "{key}");
            assert_eq!(decision.reason_code, FSFS_SENSITIVE_PRIVATE_KEY, "{key}");
            assert!(decision.satisfies_contract(), "{key}");
        }
        // Naming the header is not key material: a PEM parser, a certificate
        // and prose about private keys stay indexable.
        for text in [
            "const HEADER: &str = \"-----BEGIN PRIVATE KEY-----\";\nfn parse() {}\n".to_owned(),
            format!("-----BEGIN CERTIFICATE-----\n{body}-----END CERTIFICATE-----\n"),
            "-----BEGIN RSA PRIVATE KEY-----\n(paste your key here)\n".to_owned(),
            "Rotate the private key yearly.\n".to_owned(),
        ] {
            let decision = contract.classify_bytes("/workspace/src/pem.rs", text.as_bytes());
            assert_eq!(decision.ingest_action, IngestAction::Index, "{text}");
        }
    }

    #[test]
    fn classify_binary_with_null_bytes_skips() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/tmp/blob.bin",
            &[0, 17, 2, 144, 0, 31, 7, 0, 255],
        );

        assert_eq!(decision.detected_type, DetectedType::Binary);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.normalization_applied, "none");
        assert_eq!(decision.ingest_action, IngestAction::Skip);
        assert_eq!(decision.reason_code, "FSFS_BINARY_NULL_BYTE_DETECTED");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_archive_path_skips_before_text_decode() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/build/cache.tar.gz",
            b"this would otherwise be valid utf8",
        );

        assert_eq!(decision.detected_type, DetectedType::Archive);
        assert_eq!(decision.ingest_action, IngestAction::Skip);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.reason_code, "FSFS_ARCHIVE_EXTENSION_BLOCKED");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_truncated_utf8_text_uses_partial_policy() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_with_integrity(
            "/workspace/logs/partial.md",
            b"# heading\nbody that was cut off",
            IntegrityState::Truncated,
        );

        assert_eq!(decision.detected_type, DetectedType::Partial);
        assert_eq!(decision.detected_encoding, "utf-8");
        assert_eq!(decision.ingest_action, IngestAction::IndexPartialWithFlag);
        assert_eq!(decision.reason_code, "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY");
        assert!(decision.downstream_signals.requires_manual_review);
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_checksum_mismatch_quarantines_corrupt_input() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_with_integrity(
            "/workspace/bad/corrupt.txt",
            b"definitely not trustworthy",
            IntegrityState::ChecksumMismatch,
        );

        assert_eq!(decision.detected_type, DetectedType::Corrupt);
        assert_eq!(decision.ingest_action, IngestAction::Quarantine);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.reason_code, "FSFS_CORRUPT_CHECKSUM_MISMATCH");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_unknown_encoding_obeys_quarantine_policy() {
        let contract = FileClassificationContractDefinition::default();
        let bytes = &[0x93, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x94];
        let decision = contract.classify_bytes("/workspace/notes/cp1252.txt", bytes);

        assert_eq!(decision.detected_type, DetectedType::Text);
        assert_eq!(decision.detected_encoding, "unknown-8bit");
        assert_eq!(decision.ingest_action, IngestAction::Quarantine);
        assert_eq!(decision.reason_code, "FSFS_TEXT_HEURISTIC_QUARANTINE");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn build_corrupt_event_respects_partial_and_corrupt_actions() {
        let contract = FileClassificationContractDefinition::default();

        let truncated = contract.build_corrupt_event(
            "/workspace/tmp/partial.log",
            ErrorClass::Truncated,
            4_096,
        );
        assert_eq!(truncated.kind, FILE_CLASSIFICATION_CORRUPT_EVENT_KIND);
        assert_eq!(truncated.ingest_action, IngestAction::IndexPartialWithFlag);
        assert_eq!(truncated.reason_code, "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY");

        let decode_error =
            contract.build_corrupt_event("/workspace/tmp/bad.log", ErrorClass::DecodeError, 0);
        assert_eq!(decode_error.ingest_action, IngestAction::Quarantine);
        assert_eq!(decode_error.reason_code, "FSFS_CORRUPT_DECODE_ERROR");
    }

    #[test]
    fn contract_definition_rejects_wrong_kind() {
        let mut value = json!(FileClassificationContractDefinition::default());
        value["kind"] = json!("wrong_kind");

        let error = serde_json::from_value::<FileClassificationContractDefinition>(value)
            .expect_err("reject bad kind");

        assert!(
            error
                .to_string()
                .contains("fsfs_file_classification_contract_definition")
        );
    }

    #[test]
    fn contract_definition_rejects_wrong_version() {
        let mut value = json!(FileClassificationContractDefinition::default());
        value["v"] = json!(2);

        let error = serde_json::from_value::<FileClassificationContractDefinition>(value)
            .expect_err("reject bad version");

        assert!(error.to_string().contains("schema version 1"));
    }

    #[test]
    fn decision_rejects_unknown_fields() {
        let contract = FileClassificationContractDefinition::default();
        let mut value = json!(contract.classify_bytes("/workspace/src/lib.rs", b"hello world\n"));
        value["extra"] = json!(true);

        let error = serde_json::from_value::<FileClassificationDecision>(value)
            .expect_err("reject extra field");

        assert!(error.to_string().contains("unknown field `extra`"));
    }

    #[test]
    fn decision_rejects_partial_index_action() {
        let contract = FileClassificationContractDefinition::default();
        let mut value = json!(contract.classify_with_integrity(
            "/workspace/logs/partial.md",
            b"# heading\nbody that was cut off",
            IntegrityState::Truncated,
        ));
        value["ingest_action"] = json!("index");

        let error = serde_json::from_value::<FileClassificationDecision>(value)
            .expect_err("reject bad partial action");

        assert!(
            error
                .to_string()
                .contains("partial decisions must not use ingest_action=index")
        );
    }

    #[test]
    fn corrupt_event_rejects_non_truncated_partial_action() {
        let contract = FileClassificationContractDefinition::default();
        let mut value = json!(contract.build_corrupt_event(
            "/workspace/tmp/bad.log",
            ErrorClass::DecodeError,
            0,
        ));
        value["ingest_action"] = json!("index_partial_with_flag");

        let error = serde_json::from_value::<super::FileClassificationCorruptEvent>(value)
            .expect_err("reject invalid corrupt event action");

        assert!(error.to_string().contains("corrupt event ingest_action"));
    }
}
