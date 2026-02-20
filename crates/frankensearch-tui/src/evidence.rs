//! JSONL evidence hooks for replay, postmortem, and explainability.
//!
//! Implements the evidence-jsonl-v1 contract (see `docs/evidence-jsonl-contract.md`
//! and `schemas/evidence-jsonl-v1.schema.json`). Evidence events are emitted as
//! JSONL lines to an `EvidenceWriter` sink.
//!
//! All evidence payloads go through mandatory redaction — raw sensitive fields
//! are never emitted.

use serde::{Deserialize, Serialize};

use crate::determinism::ReplayMetadata;

// ─── Evidence Envelope ──────────────────────────────────────────────────────

/// Top-level JSONL envelope per the evidence-jsonl-v1 schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEnvelope {
    /// Schema version (always 1).
    pub v: u32,
    /// ISO 8601 timestamp.
    pub ts: String,
    /// The evidence event payload.
    pub event: EvidenceEvent,
}

// ─── Evidence Event ─────────────────────────────────────────────────────────

/// Evidence event types per the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEventType {
    /// A decision was made (e.g., phase transition, model selection).
    Decision,
    /// An alert was raised (e.g., SLO breach).
    Alert,
    /// Degradation detected (e.g., high latency, error rate).
    Degradation,
    /// State transition (e.g., screen change, overlay open/close).
    Transition,
    /// Replay marker for synchronization.
    ReplayMarker,
}

/// Reason code with human-readable explanation and severity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReason {
    /// Machine-stable reason code (e.g., `search.phase.refinement_failed`).
    pub code: String,
    /// Human-readable explanation.
    pub human: String,
    /// Severity level.
    pub severity: EvidenceSeverity,
}

/// Evidence severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSeverity {
    /// Informational.
    Info,
    /// Warning.
    Warn,
    /// Error.
    Error,
}

/// Trace context for event correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceTrace {
    /// Root request ID (ULID).
    pub root_request_id: String,
    /// Parent event ID (null for root events).
    pub parent_event_id: Option<String>,
}

/// Redaction metadata — declares what transforms were applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRedaction {
    /// Policy version (e.g., "v1").
    pub policy_version: String,
    /// Whether the source data contained sensitive fields.
    pub contains_sensitive_source: bool,
    /// List of transforms applied.
    pub transforms_applied: Vec<RedactionTransform>,
}

/// Redaction transforms per the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionTransform {
    /// SHA-256 one-way hash.
    HashSha256,
    /// Truncated preview (max 120 chars).
    TruncatePreview,
    /// Complete removal.
    Drop,
    /// Path tokenization (project-relative only).
    PathTokenize,
    /// Partial masking.
    MaskPartial,
}

/// Sanitized evidence payload (only allowed fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvidencePayload {
    /// SHA-256 hash of the query text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_hash: Option<String>,
    /// Truncated query preview (max 120 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_preview: Option<String>,
    /// Resource profile at time of event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_profile: Option<String>,
    /// Lag in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lag_ms: Option<u64>,
    /// Count of dropped items.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropped_count: Option<u64>,
    /// Free-form notes (max 500 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Full evidence event (inside the envelope).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEvent {
    /// Unique event ID (ULID format).
    pub event_id: String,
    /// Event type.
    #[serde(rename = "type")]
    pub event_type: EvidenceEventType,
    /// Project key.
    pub project_key: String,
    /// Instance ID (ULID format).
    pub instance_id: String,
    /// Trace context.
    pub trace: EvidenceTrace,
    /// Reason code + human text + severity.
    pub reason: EvidenceReason,
    /// Replay metadata.
    pub replay: ReplayMetadata,
    /// Redaction metadata.
    pub redaction: EvidenceRedaction,
    /// Sanitized payload.
    pub payload: EvidencePayload,
}

// ─── Evidence Writer ────────────────────────────────────────────────────────

/// Sink for evidence JSONL lines.
///
/// Implementations write serialized evidence envelopes. The default
/// [`VecWriter`] collects into memory for testing. Production code
/// would use a file or ring-buffer writer.
pub trait EvidenceSink: Send {
    /// Write a single evidence envelope as a JSONL line.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or I/O fails.
    fn write(&mut self, envelope: &EvidenceEnvelope) -> Result<(), EvidenceWriteError>;
}

/// Error type for evidence writing.
#[derive(Debug)]
pub enum EvidenceWriteError {
    /// JSON serialization failed.
    Serialization(serde_json::Error),
    /// I/O error (for file-backed sinks).
    Io(std::io::Error),
}

impl std::fmt::Display for EvidenceWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(e) => write!(f, "evidence serialization error: {e}"),
            Self::Io(e) => write!(f, "evidence I/O error: {e}"),
        }
    }
}

impl std::error::Error for EvidenceWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for EvidenceWriteError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e)
    }
}

impl From<std::io::Error> for EvidenceWriteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ─── Vec Writer (Testing) ───────────────────────────────────────────────────

/// In-memory evidence sink for testing.
///
/// Collects all emitted envelopes into a `Vec` for inspection.
pub struct VecWriter {
    /// Collected envelopes.
    entries: Vec<EvidenceEnvelope>,
}

impl VecWriter {
    /// Create a new empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Get all collected envelopes.
    #[must_use]
    pub fn entries(&self) -> &[EvidenceEnvelope] {
        &self.entries
    }

    /// Number of collected entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries have been collected.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Export all entries as JSONL (one JSON object per line).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut output = String::new();
        for entry in &self.entries {
            let line = serde_json::to_string(entry)?;
            output.push_str(&line);
            output.push('\n');
        }
        Ok(output)
    }
}

impl Default for VecWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl EvidenceSink for VecWriter {
    fn write(&mut self, envelope: &EvidenceEnvelope) -> Result<(), EvidenceWriteError> {
        // Validate serializes cleanly (catches schema issues early).
        let _json = serde_json::to_string(envelope)?;
        self.entries.push(envelope.clone());
        Ok(())
    }
}

// ─── Noop Writer ────────────────────────────────────────────────────────────

/// No-op evidence sink that discards all events.
///
/// Used when evidence logging is disabled.
pub struct NoopWriter;

impl EvidenceSink for NoopWriter {
    fn write(&mut self, _envelope: &EvidenceEnvelope) -> Result<(), EvidenceWriteError> {
        Ok(())
    }
}

// ─── Builder Helpers ────────────────────────────────────────────────────────

/// Build an evidence envelope with the current timestamp.
///
/// Caller provides the event; this function wraps it in the v1 envelope.
#[must_use]
pub fn wrap_envelope(event: EvidenceEvent) -> EvidenceEnvelope {
    // Use a simple ISO 8601 timestamp (no external crate needed).
    let ts = format_iso8601_now();
    EvidenceEnvelope { v: 1, ts, event }
}

/// Format the current time as an ISO 8601 string.
///
/// Uses `SystemTime` to produce a UTC timestamp without external crate deps.
fn format_iso8601_now() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Simple UTC timestamp from epoch seconds.
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Days since epoch to Y-M-D (simplified leap year calculation).
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
const fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Shift to March-based calendar for simpler leap year handling.
    let era_days = days + 719_468; // Days from 0000-03-01 to Unix epoch.
    let era = era_days / 146_097;
    let day_of_era = era_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

/// Build a minimal redaction metadata for non-sensitive events.
#[must_use]
pub fn redaction_none() -> EvidenceRedaction {
    EvidenceRedaction {
        policy_version: "v1".to_string(),
        contains_sensitive_source: false,
        transforms_applied: vec![RedactionTransform::Drop],
    }
}

/// Build a redaction metadata indicating query content was hashed.
#[must_use]
pub fn redaction_query_hashed() -> EvidenceRedaction {
    EvidenceRedaction {
        policy_version: "v1".to_string(),
        contains_sensitive_source: true,
        transforms_applied: vec![
            RedactionTransform::HashSha256,
            RedactionTransform::TruncatePreview,
        ],
    }
}

#[cfg(test)]
mod tests {
    use crate::determinism::ReplayMetadata;

    use super::*;

    fn sample_event() -> EvidenceEvent {
        EvidenceEvent {
            event_id: "01HQXYZ1234567890ABCDEFGHIJ".to_string(),
            event_type: EvidenceEventType::Decision,
            project_key: "/data/projects/test".to_string(),
            instance_id: "01HQXYZ9876543210ABCDEFGHIJ".to_string(),
            trace: EvidenceTrace {
                root_request_id: "01HQXYZREQUEST00000000000A".to_string(),
                parent_event_id: None,
            },
            reason: EvidenceReason {
                code: "search.phase.initial_complete".to_string(),
                human: "Initial search phase completed".to_string(),
                severity: EvidenceSeverity::Info,
            },
            replay: ReplayMetadata::live(),
            redaction: redaction_none(),
            payload: EvidencePayload::default(),
        }
    }

    #[test]
    fn evidence_envelope_serde_roundtrip() {
        let event = sample_event();
        let envelope = wrap_envelope(event);

        assert_eq!(envelope.v, 1);
        assert!(!envelope.ts.is_empty());

        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: EvidenceEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.v, 1);
        assert_eq!(decoded.event.event_type, EvidenceEventType::Decision);
    }

    #[test]
    fn evidence_event_type_serde() {
        for event_type in [
            EvidenceEventType::Decision,
            EvidenceEventType::Alert,
            EvidenceEventType::Degradation,
            EvidenceEventType::Transition,
            EvidenceEventType::ReplayMarker,
        ] {
            let json = serde_json::to_string(&event_type).unwrap();
            let decoded: EvidenceEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, event_type);
        }
    }

    #[test]
    fn evidence_severity_serde() {
        for severity in [
            EvidenceSeverity::Info,
            EvidenceSeverity::Warn,
            EvidenceSeverity::Error,
        ] {
            let json = serde_json::to_string(&severity).unwrap();
            let decoded: EvidenceSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, severity);
        }
    }

    #[test]
    fn redaction_transform_serde() {
        for transform in [
            RedactionTransform::HashSha256,
            RedactionTransform::TruncatePreview,
            RedactionTransform::Drop,
            RedactionTransform::PathTokenize,
            RedactionTransform::MaskPartial,
        ] {
            let json = serde_json::to_string(&transform).unwrap();
            let decoded: RedactionTransform = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, transform);
        }
    }

    #[test]
    fn vec_writer_collects() {
        let mut writer = VecWriter::new();
        assert!(writer.is_empty());

        let event = sample_event();
        let envelope = wrap_envelope(event);
        writer.write(&envelope).unwrap();

        assert_eq!(writer.len(), 1);
        assert_eq!(writer.entries()[0].v, 1);
    }

    #[test]
    fn vec_writer_to_jsonl() {
        let mut writer = VecWriter::new();

        let event1 = sample_event();
        writer.write(&wrap_envelope(event1)).unwrap();

        let mut event2 = sample_event();
        event2.event_type = EvidenceEventType::Alert;
        writer.write(&wrap_envelope(event2)).unwrap();

        let jsonl = writer.to_jsonl().unwrap();
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line should be valid JSON.
        for line in &lines {
            let _: EvidenceEnvelope = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn noop_writer_discards() {
        let mut writer = NoopWriter;
        let event = sample_event();
        let envelope = wrap_envelope(event);
        // Should not error.
        writer.write(&envelope).unwrap();
    }

    #[test]
    fn evidence_with_deterministic_replay() {
        let mut event = sample_event();
        event.replay = ReplayMetadata::deterministic(42, 16, 100);

        let envelope = wrap_envelope(event);
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("\"deterministic\""));
        assert!(json.contains("\"seed\":42"));
        assert!(json.contains("\"tick_ms\":16"));
        assert!(json.contains("\"frame_seq\":100"));
    }

    #[test]
    fn evidence_with_query_redaction() {
        let mut event = sample_event();
        event.redaction = redaction_query_hashed();
        event.payload.query_hash = Some("a".repeat(64));
        event.payload.query_preview = Some("how does search work".to_string());

        let envelope = wrap_envelope(event);
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("hash_sha256"));
        assert!(json.contains("truncate_preview"));
        assert!(json.contains("\"contains_sensitive_source\":true"));
    }

    #[test]
    fn redaction_none_has_drop() {
        let r = redaction_none();
        assert!(!r.contains_sensitive_source);
        assert_eq!(r.transforms_applied, vec![RedactionTransform::Drop]);
    }

    #[test]
    fn evidence_payload_default_empty() {
        let payload = EvidencePayload::default();
        let json = serde_json::to_string(&payload).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn evidence_payload_with_fields() {
        let payload = EvidencePayload {
            lag_ms: Some(42),
            notes: Some("test note".to_string()),
            ..EvidencePayload::default()
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("\"lag_ms\":42"));
        assert!(json.contains("\"notes\":\"test note\""));
        // Fields that are None should be omitted.
        assert!(!json.contains("query_hash"));
    }

    #[test]
    fn format_iso8601_produces_valid_format() {
        let ts = format_iso8601_now();
        // Should match YYYY-MM-DDTHH:MM:SSZ pattern.
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
    }

    #[test]
    fn evidence_write_error_display() {
        let err = EvidenceWriteError::Io(std::io::Error::other("test error"));
        let msg = err.to_string();
        assert!(msg.contains("evidence I/O error"));
    }
}
