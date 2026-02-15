//! Streaming query protocol for fsfs agent CLI mode.
//!
//! This module defines machine-stable streaming frames emitted in `--stream`
//! mode for NDJSON and TOON outputs.

use std::io;

use frankensearch_core::{SearchError, SearchResult};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::evidence::{ValidationResult, ValidationViolation};
use crate::explanation_payload::FsfsExplanationPayload;
use crate::output_schema::{OutputError, OutputWarning, exit_code_for, output_error_from};

const SUBSYSTEM: &str = "fsfs_stream_protocol";
const TOON_DEFAULT_DELIMITER: char = ',';

/// Current stream protocol version.
pub const STREAM_PROTOCOL_VERSION: u32 = 1;

/// Stable stream schema identifier.
pub const STREAM_SCHEMA_VERSION: &str = "fsfs.stream.query.v1";

/// Record separator prefix for TOON stream frames (`0x1E`).
///
/// TOON payloads may span multiple lines, so stream transport prefixes each
/// frame with this byte to provide deterministic frame boundaries.
pub const TOON_STREAM_RECORD_SEPARATOR: char = '\u{001E}';
/// Byte form of [`TOON_STREAM_RECORD_SEPARATOR`].
pub const TOON_STREAM_RECORD_SEPARATOR_BYTE: u8 = 0x1E;

/// High-level event taxonomy for stream frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEventKind {
    /// Stream initialization acknowledgement (always first, exactly once).
    Started,
    /// Index/query pipeline progress update.
    Progress,
    /// Ranked result payload.
    Result,
    /// Explainability payload for a selected result.
    Explain,
    /// Non-fatal warning payload.
    Warning,
    /// Terminal stream state with exit and retry semantics (always last, exactly once).
    Terminal,
}

/// Started event payload, emitted exactly once as the first stream frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamStartedEvent {
    /// Unique stream identifier for correlation (ULID format).
    pub stream_id: String,
    /// The query being searched.
    pub query: String,
    /// Output format used for this stream.
    pub format: String,
}

/// Progress event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamProgressEvent {
    /// Stable stage name (`retrieve.fast`, `retrieve.quality`, `rerank`, etc).
    pub stage: String,
    /// Completed unit count for the stage.
    pub completed_units: u64,
    /// Total unit count when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_units: Option<u64>,
    /// Stable reason code for this progress update.
    pub reason_code: String,
    /// Human-readable message.
    pub message: String,
}

/// Result event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamResultEvent<T> {
    /// One-based rank in the emitted stream.
    pub rank: u64,
    /// Result payload.
    pub item: T,
}

/// Explain event payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamExplainEvent {
    /// Rich explanation payload.
    pub explanation: FsfsExplanationPayload,
}

/// Warning event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamWarningEvent {
    /// Structured warning payload.
    pub warning: OutputWarning,
}

/// Final stream status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamTerminalStatus {
    /// Stream completed successfully.
    Completed,
    /// Stream failed with a categorized error.
    Failed,
    /// Stream terminated due to cancellation/interrupt.
    Cancelled,
}

/// Stable failure categories for deterministic recovery handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamFailureCategory {
    Config,
    Index,
    Model,
    Resource,
    Io,
    Internal,
}

/// Retry guidance attached to terminal frames.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum StreamRetryDirective {
    /// Do not retry.
    None,
    /// Retry is recommended after the specified backoff.
    RetryAfterMs {
        delay_ms: u64,
        next_attempt: u32,
        max_attempts: u32,
    },
    /// Retry budget is exhausted.
    RetryExhausted { exhausted_after: u32 },
}

/// Terminal event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamTerminalEvent {
    /// Terminal state.
    pub status: StreamTerminalStatus,
    /// Deterministic process exit code mapping.
    pub exit_code: i32,
    /// Failure category for failed streams.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_category: Option<StreamFailureCategory>,
    /// Structured failure payload for failed/cancelled streams.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<OutputError>,
    /// Retry guidance.
    pub retry: StreamRetryDirective,
}

/// Stream event payload union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", content = "payload", rename_all = "snake_case")]
pub enum StreamEvent<T> {
    Started(StreamStartedEvent),
    Progress(StreamProgressEvent),
    Result(StreamResultEvent<T>),
    Explain(Box<StreamExplainEvent>),
    Warning(StreamWarningEvent),
    Terminal(StreamTerminalEvent),
}

impl<T> StreamEvent<T> {
    /// Event-kind discriminator.
    #[must_use]
    pub const fn kind(&self) -> StreamEventKind {
        match self {
            Self::Started(_) => StreamEventKind::Started,
            Self::Progress(_) => StreamEventKind::Progress,
            Self::Result(_) => StreamEventKind::Result,
            Self::Explain(_) => StreamEventKind::Explain,
            Self::Warning(_) => StreamEventKind::Warning,
            Self::Terminal(_) => StreamEventKind::Terminal,
        }
    }
}

/// Canonical stream frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamFrame<T> {
    /// Stream protocol version.
    pub v: u32,
    /// Stable schema identifier.
    pub schema_version: String,
    /// Stable stream identifier (ULID/UUID).
    pub stream_id: String,
    /// Monotonic sequence number for deterministic replay.
    pub seq: u64,
    /// RFC3339 UTC timestamp.
    pub ts: String,
    /// Command that emitted the frame (`search`).
    pub command: String,
    /// Event payload.
    #[serde(flatten)]
    pub event: StreamEvent<T>,
}

impl<T> StreamFrame<T> {
    /// Construct a stream frame.
    #[must_use]
    pub fn new(
        stream_id: impl Into<String>,
        seq: u64,
        ts: impl Into<String>,
        command: impl Into<String>,
        event: StreamEvent<T>,
    ) -> Self {
        Self {
            v: STREAM_PROTOCOL_VERSION,
            schema_version: STREAM_SCHEMA_VERSION.to_owned(),
            stream_id: stream_id.into(),
            seq,
            ts: ts.into(),
            command: command.into(),
            event,
        }
    }
}

/// Build a successful terminal payload.
#[must_use]
pub const fn terminal_event_completed() -> StreamTerminalEvent {
    StreamTerminalEvent {
        status: StreamTerminalStatus::Completed,
        exit_code: crate::adapters::cli::exit_code::OK,
        failure_category: None,
        error: None,
        retry: StreamRetryDirective::None,
    }
}

/// Build a terminal payload from a failure with deterministic retry guidance.
#[must_use]
pub fn terminal_event_from_error(
    err: &SearchError,
    attempt: u32,
    max_attempts: u32,
) -> StreamTerminalEvent {
    if matches!(err, SearchError::Cancelled { .. }) {
        return StreamTerminalEvent {
            status: StreamTerminalStatus::Cancelled,
            exit_code: crate::adapters::cli::exit_code::INTERRUPTED,
            failure_category: Some(StreamFailureCategory::Resource),
            error: Some(output_error_from(err)),
            retry: StreamRetryDirective::None,
        };
    }

    let retry = retry_directive_for_error(err, attempt, max_attempts);
    StreamTerminalEvent {
        status: StreamTerminalStatus::Failed,
        exit_code: exit_code_for(err),
        failure_category: Some(failure_category_for_error(err)),
        error: Some(output_error_from(err)),
        retry,
    }
}

/// Deterministic failure categorization.
#[must_use]
pub const fn failure_category_for_error(err: &SearchError) -> StreamFailureCategory {
    match err {
        SearchError::InvalidConfig { .. } | SearchError::QueryParseError { .. } => {
            StreamFailureCategory::Config
        }
        SearchError::IndexCorrupted { .. }
        | SearchError::IndexVersionMismatch { .. }
        | SearchError::DimensionMismatch { .. }
        | SearchError::IndexNotFound { .. } => StreamFailureCategory::Index,
        SearchError::EmbedderUnavailable { .. }
        | SearchError::EmbeddingFailed { .. }
        | SearchError::ModelNotFound { .. }
        | SearchError::ModelLoadFailed { .. }
        | SearchError::RerankerUnavailable { .. }
        | SearchError::RerankFailed { .. } => StreamFailureCategory::Model,
        SearchError::SearchTimeout { .. }
        | SearchError::FederatedInsufficientResponses { .. }
        | SearchError::QueueFull { .. }
        | SearchError::Cancelled { .. } => StreamFailureCategory::Resource,
        SearchError::Io(_) | SearchError::HashMismatch { .. } | SearchError::DurabilityDisabled => {
            StreamFailureCategory::Io
        }
        SearchError::SubsystemError { .. } => StreamFailureCategory::Internal,
    }
}

/// Whether an error class should trigger retry guidance.
#[must_use]
pub const fn is_retryable_error(err: &SearchError) -> bool {
    matches!(
        err,
        SearchError::EmbeddingFailed { .. }
            | SearchError::SearchTimeout { .. }
            | SearchError::FederatedInsufficientResponses { .. }
            | SearchError::RerankFailed { .. }
            | SearchError::Io(_)
            | SearchError::QueueFull { .. }
            | SearchError::SubsystemError { .. }
    )
}

/// Exponential backoff policy for retry guidance.
#[must_use]
pub const fn retry_backoff_ms(attempt: u32) -> u64 {
    const BASE_MS: u64 = 250;
    const CAP_MS: u64 = 30_000;

    let shift = if attempt > 10 { 10 } else { attempt };
    let delay = BASE_MS << shift;
    if delay > CAP_MS { CAP_MS } else { delay }
}

const fn retry_directive_for_error(
    err: &SearchError,
    attempt: u32,
    max_attempts: u32,
) -> StreamRetryDirective {
    if !is_retryable_error(err) || max_attempts == 0 {
        return StreamRetryDirective::None;
    }

    if attempt < max_attempts {
        return StreamRetryDirective::RetryAfterMs {
            delay_ms: retry_backoff_ms(attempt),
            next_attempt: attempt.saturating_add(1),
            max_attempts,
        };
    }

    StreamRetryDirective::RetryExhausted {
        exhausted_after: max_attempts,
    }
}

/// Encode one stream frame as NDJSON (single-line JSON + trailing newline).
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
///
/// # Panics
///
/// Panics if `serde_json` produces non-UTF-8 output (cannot happen in practice).
pub fn encode_stream_frame_ndjson<T>(frame: &StreamFrame<T>) -> SearchResult<String>
where
    T: Serialize,
{
    // Pre-size buffer: typical stream frame is ~2 KB.
    let mut buf = Vec::with_capacity(2048);
    serde_json::to_writer(&mut buf, frame).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to encode stream frame as NDJSON: {source}"
        ))),
    })?;
    buf.push(b'\n');
    // serde_json always produces valid UTF-8; unwrap is safe here.
    Ok(String::from_utf8(buf).expect("serde_json output is valid UTF-8"))
}

/// Decode one NDJSON frame line.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if deserialization fails.
pub fn decode_stream_frame_ndjson<T>(line: &str) -> SearchResult<StreamFrame<T>>
where
    T: DeserializeOwned,
{
    serde_json::from_str(line.trim_end()).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to decode NDJSON stream frame: {source}"
        ))),
    })
}

/// Encode one stream frame as TOON text.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization/encoding fails.
pub fn encode_stream_frame_toon<T>(frame: &StreamFrame<T>) -> SearchResult<String>
where
    T: Serialize,
{
    let mut value = serde_json::to_value(frame).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to project stream frame to JSON value: {source}"
        ))),
    })?;
    prepare_toon_value_for_lossless_strings(&mut value)?;

    toon_rust::encode(&value, None).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to encode stream frame as TOON: {source}"
        ))),
    })
}

/// Decode one TOON stream frame.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if decoding/deserialization fails.
pub fn decode_stream_frame_toon<T>(input: &str) -> SearchResult<StreamFrame<T>>
where
    T: DeserializeOwned,
{
    let value = toon_rust::decode(input, None).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to decode TOON stream frame: {source}"
        ))),
    })?;

    serde_json::from_value(value).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to deserialize TOON stream frame: {source}"
        ))),
    })
}

/// Validate stream frame invariants.
#[must_use]
pub fn validate_stream_frame<T>(frame: &StreamFrame<T>) -> ValidationResult {
    let mut violations = Vec::new();

    if frame.v != STREAM_PROTOCOL_VERSION {
        violations.push(ValidationViolation {
            field: "v".into(),
            code: "stream.schema_version.mismatch".into(),
            message: format!(
                "expected stream protocol version {STREAM_PROTOCOL_VERSION}, found {}",
                frame.v
            ),
        });
    }

    if frame.schema_version != STREAM_SCHEMA_VERSION {
        violations.push(ValidationViolation {
            field: "schema_version".into(),
            code: "stream.schema_tag.mismatch".into(),
            message: format!(
                "expected schema_version {STREAM_SCHEMA_VERSION}, found {}",
                frame.schema_version
            ),
        });
    }

    if frame.stream_id.is_empty() {
        violations.push(ValidationViolation {
            field: "stream_id".into(),
            code: "stream.stream_id.empty".into(),
            message: "stream_id must be non-empty".into(),
        });
    }

    if frame.ts.is_empty() {
        violations.push(ValidationViolation {
            field: "ts".into(),
            code: "stream.timestamp.empty".into(),
            message: "ts must be non-empty RFC3339".into(),
        });
    }

    if frame.command.is_empty() {
        violations.push(ValidationViolation {
            field: "command".into(),
            code: "stream.command.empty".into(),
            message: "command must be non-empty".into(),
        });
    }

    if let StreamEvent::Terminal(terminal) = &frame.event {
        validate_terminal_event(terminal, &mut violations);
    }

    ValidationResult {
        valid: violations.is_empty(),
        violations,
    }
}

fn validate_terminal_event(
    terminal: &StreamTerminalEvent,
    violations: &mut Vec<ValidationViolation>,
) {
    match terminal.status {
        StreamTerminalStatus::Completed => {
            if terminal.exit_code != crate::adapters::cli::exit_code::OK {
                violations.push(ValidationViolation {
                    field: "terminal.exit_code".into(),
                    code: "stream.terminal.completed_nonzero_exit".into(),
                    message: "completed terminal event must use exit_code 0".into(),
                });
            }
            if terminal.error.is_some() {
                violations.push(ValidationViolation {
                    field: "terminal.error".into(),
                    code: "stream.terminal.completed_has_error".into(),
                    message: "completed terminal event must not include error payload".into(),
                });
            }
            if terminal.failure_category.is_some() {
                violations.push(ValidationViolation {
                    field: "terminal.failure_category".into(),
                    code: "stream.terminal.completed_has_category".into(),
                    message: "completed terminal event must not include failure category".into(),
                });
            }
            if !matches!(terminal.retry, StreamRetryDirective::None) {
                violations.push(ValidationViolation {
                    field: "terminal.retry".into(),
                    code: "stream.terminal.completed_has_retry".into(),
                    message: "completed terminal event must not include retry guidance".into(),
                });
            }
        }
        StreamTerminalStatus::Failed => {
            if terminal.exit_code == crate::adapters::cli::exit_code::OK {
                violations.push(ValidationViolation {
                    field: "terminal.exit_code".into(),
                    code: "stream.terminal.failed_zero_exit".into(),
                    message: "failed terminal event must use non-zero exit code".into(),
                });
            }
            if terminal.failure_category.is_none() {
                violations.push(ValidationViolation {
                    field: "terminal.failure_category".into(),
                    code: "stream.terminal.failed_missing_category".into(),
                    message: "failed terminal event must include failure_category".into(),
                });
            }
            if terminal.error.is_none() {
                violations.push(ValidationViolation {
                    field: "terminal.error".into(),
                    code: "stream.terminal.failed_missing_error".into(),
                    message: "failed terminal event must include error payload".into(),
                });
            }
        }
        StreamTerminalStatus::Cancelled => {
            if terminal.exit_code != crate::adapters::cli::exit_code::INTERRUPTED {
                violations.push(ValidationViolation {
                    field: "terminal.exit_code".into(),
                    code: "stream.terminal.cancelled_exit_mismatch".into(),
                    message: "cancelled terminal event must use interrupted exit code".into(),
                });
            }
            if !matches!(terminal.retry, StreamRetryDirective::None) {
                violations.push(ValidationViolation {
                    field: "terminal.retry".into(),
                    code: "stream.terminal.cancelled_has_retry".into(),
                    message: "cancelled terminal event must not include retry guidance".into(),
                });
            }
        }
    }

    if let StreamRetryDirective::RetryAfterMs {
        delay_ms,
        next_attempt,
        max_attempts,
    } = terminal.retry
    {
        if delay_ms == 0 {
            violations.push(ValidationViolation {
                field: "terminal.retry.delay_ms".into(),
                code: "stream.retry.delay_zero".into(),
                message: "retry delay must be greater than zero".into(),
            });
        }
        if next_attempt == 0 || next_attempt > max_attempts {
            violations.push(ValidationViolation {
                field: "terminal.retry.next_attempt".into(),
                code: "stream.retry.next_attempt.invalid".into(),
                message: "next_attempt must be within retry budget".into(),
            });
        }
    }
}

fn prepare_toon_value_for_lossless_strings(value: &mut serde_json::Value) -> SearchResult<()> {
    match value {
        serde_json::Value::String(token) => {
            if should_wrap_toon_string_token(token) {
                let wrapped =
                    serde_json::to_string(token).map_err(|source| SearchError::SubsystemError {
                        subsystem: SUBSYSTEM,
                        source: Box::new(io::Error::other(format!(
                            "failed to prepare TOON stream string token: {source}"
                        ))),
                    })?;
                *token = wrapped;
            }
        }
        serde_json::Value::Array(values) => {
            for item in values {
                prepare_toon_value_for_lossless_strings(item)?;
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values_mut() {
                prepare_toon_value_for_lossless_strings(item)?;
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
    Ok(())
}

fn should_wrap_toon_string_token(token: &str) -> bool {
    !toon_encoder_would_quote_string(token, TOON_DEFAULT_DELIMITER)
        && !toon_unquoted_token_roundtrips_as_same_string(token)
}

fn toon_encoder_would_quote_string(token: &str, delimiter: char) -> bool {
    token.contains(delimiter)
        || token.contains(' ')
        || token.contains('\n')
        || token.contains('\t')
        || token == "true"
        || token == "false"
        || token == "null"
        || token.parse::<f64>().is_ok()
}

fn toon_unquoted_token_roundtrips_as_same_string(token: &str) -> bool {
    let probe = format!("v: {token}");
    match toon_rust::decode(&probe, None) {
        Ok(serde_json::Value::Object(map)) => {
            map.get("v") == Some(&serde_json::Value::String(token.to_owned()))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    fn sample_frame(event: StreamEvent<Vec<String>>) -> StreamFrame<Vec<String>> {
        StreamFrame::new(
            "01JAH9A2W8F8Q6GQ4C7M3N2P1R",
            1,
            "2026-02-14T12:00:00Z",
            "search",
            event,
        )
    }

    #[test]
    fn event_kind_discriminator_is_stable() {
        assert_eq!(
            StreamEvent::<()>::Started(StreamStartedEvent {
                stream_id: "01TEST".into(),
                query: "hello".into(),
                format: "jsonl".into(),
            })
            .kind(),
            StreamEventKind::Started
        );
        assert_eq!(
            StreamEvent::<()>::Progress(StreamProgressEvent {
                stage: "retrieve.fast".into(),
                completed_units: 1,
                total_units: Some(2),
                reason_code: "query.phase.initial".into(),
                message: "phase 1 complete".into(),
            })
            .kind(),
            StreamEventKind::Progress
        );
        assert_eq!(
            StreamEvent::<()>::Terminal(terminal_event_completed()).kind(),
            StreamEventKind::Terminal
        );
    }

    #[test]
    fn started_event_ndjson_roundtrip() {
        let frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".into(),
            query: "test query".into(),
            format: "jsonl".into(),
        }));

        let encoded = encode_stream_frame_ndjson(&frame).expect("encode started ndjson");
        assert!(encoded.contains("\"event\":\"started\""));
        assert!(encoded.contains("\"query\":\"test query\""));

        let decoded: StreamFrame<Vec<String>> =
            decode_stream_frame_ndjson(&encoded).expect("decode started ndjson");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn started_event_toon_roundtrip() {
        let frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".into(),
            query: "hello world".into(),
            format: "toon".into(),
        }));

        let toon = encode_stream_frame_toon(&frame).expect("encode started toon");
        let decoded: StreamFrame<Vec<String>> =
            decode_stream_frame_toon(&toon).expect("decode started toon");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn validate_started_frame() {
        let frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "01TEST".into(),
            query: "q".into(),
            format: "jsonl".into(),
        }));
        let validation = validate_stream_frame(&frame);
        assert!(validation.valid);
    }

    #[test]
    fn ndjson_roundtrip_result_frame() {
        let frame = sample_frame(StreamEvent::Result(StreamResultEvent {
            rank: 1,
            item: vec!["doc-1".to_owned(), "doc-2".to_owned()],
        }));

        let encoded = encode_stream_frame_ndjson(&frame).expect("encode ndjson");
        assert!(encoded.ends_with('\n'));

        let decoded: StreamFrame<Vec<String>> =
            decode_stream_frame_ndjson(&encoded).expect("decode ndjson");
        assert_eq!(decoded, frame);
    }

    #[test]
    fn toon_roundtrip_preserves_ambiguous_string_tokens() {
        let frame = sample_frame(StreamEvent::Result(StreamResultEvent {
            rank: 1,
            item: vec![
                "1abc".to_owned(),
                "\"quoted\"".to_owned(),
                "[array-like".to_owned(),
                "null1".to_owned(),
                "-abc".to_owned(),
            ],
        }));

        let toon = encode_stream_frame_toon(&frame).expect("encode toon");
        let decoded: StreamFrame<Vec<String>> =
            decode_stream_frame_toon(&toon).expect("decode toon");

        assert_eq!(decoded, frame);
    }

    #[test]
    fn retry_backoff_is_capped() {
        assert_eq!(retry_backoff_ms(0), 250);
        assert_eq!(retry_backoff_ms(1), 500);
        assert_eq!(retry_backoff_ms(10), 30_000);
        assert_eq!(retry_backoff_ms(99), 30_000);
    }

    #[test]
    fn terminal_event_from_error_non_retryable_config() {
        let err = SearchError::InvalidConfig {
            field: "cli.stream".into(),
            value: "true".into(),
            reason: "stream mode requires jsonl or toon".into(),
        };
        let terminal = terminal_event_from_error(&err, 0, 3);

        assert_eq!(terminal.status, StreamTerminalStatus::Failed);
        assert_eq!(
            terminal.exit_code,
            crate::adapters::cli::exit_code::USAGE_ERROR
        );
        assert_eq!(
            terminal.failure_category,
            Some(StreamFailureCategory::Config)
        );
        assert!(matches!(terminal.retry, StreamRetryDirective::None));
        assert_eq!(
            terminal.error.as_ref().map(|e| e.code.as_str()),
            Some("invalid_config")
        );
    }

    #[test]
    fn terminal_event_from_error_retryable_io() {
        let err = SearchError::Io(io::Error::other("network jitter"));
        let terminal = terminal_event_from_error(&err, 1, 3);

        assert_eq!(terminal.status, StreamTerminalStatus::Failed);
        assert_eq!(terminal.failure_category, Some(StreamFailureCategory::Io));
        assert!(matches!(
            terminal.retry,
            StreamRetryDirective::RetryAfterMs {
                delay_ms: 500,
                next_attempt: 2,
                max_attempts: 3,
            }
        ));
    }

    #[test]
    fn terminal_event_from_error_retry_exhausted() {
        let err = SearchError::EmbeddingFailed {
            model: "miniLM".into(),
            source: Box::new(io::Error::other("transient")),
        };
        let terminal = terminal_event_from_error(&err, 3, 3);

        assert!(matches!(
            terminal.retry,
            StreamRetryDirective::RetryExhausted { exhausted_after: 3 }
        ));
    }

    #[test]
    fn terminal_event_from_error_cancelled() {
        let err = SearchError::Cancelled {
            phase: "query".into(),
            reason: "user interrupt".into(),
        };
        let terminal = terminal_event_from_error(&err, 1, 3);

        assert_eq!(terminal.status, StreamTerminalStatus::Cancelled);
        assert_eq!(
            terminal.exit_code,
            crate::adapters::cli::exit_code::INTERRUPTED
        );
        assert!(matches!(terminal.retry, StreamRetryDirective::None));
    }

    #[test]
    fn validate_completed_terminal_contract() {
        let frame = sample_frame(StreamEvent::Terminal(terminal_event_completed()));
        let validation = validate_stream_frame(&frame);
        assert!(validation.valid);
    }

    #[test]
    fn validate_rejects_invalid_terminal_invariants() {
        let frame = sample_frame(StreamEvent::Terminal(StreamTerminalEvent {
            status: StreamTerminalStatus::Completed,
            exit_code: 2,
            failure_category: Some(StreamFailureCategory::Config),
            error: Some(OutputError::new("invalid_config", "bad", 2)),
            retry: StreamRetryDirective::RetryAfterMs {
                delay_ms: 0,
                next_attempt: 2,
                max_attempts: 1,
            },
        }));

        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.completed_nonzero_exit")
        );
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.retry.delay_zero")
        );
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.retry.next_attempt.invalid")
        );
    }

    #[test]
    fn full_stream_lifecycle_ndjson() {
        // Simulate a complete stream: Started → Progress → Result → Warning → Terminal
        let stream_id = "01JAH9LIFECYCLE";
        let ts = "2026-02-14T12:00:00Z";
        let mut output = Vec::new();

        let events: Vec<StreamEvent<Vec<String>>> = vec![
            StreamEvent::Started(StreamStartedEvent {
                stream_id: stream_id.into(),
                query: "lifecycle test".into(),
                format: "jsonl".into(),
            }),
            StreamEvent::Progress(StreamProgressEvent {
                stage: "retrieve.fast".into(),
                completed_units: 100,
                total_units: Some(100),
                reason_code: "query.phase.initial".into(),
                message: "fast retrieval complete".into(),
            }),
            StreamEvent::Result(StreamResultEvent {
                rank: 1,
                item: vec!["doc-1".to_owned()],
            }),
            StreamEvent::Warning(StreamWarningEvent {
                warning: OutputWarning::new("degraded_mode", "quality tier skipped"),
            }),
            StreamEvent::Terminal(terminal_event_completed()),
        ];

        for (i, event) in events.iter().enumerate() {
            let frame = StreamFrame::new(stream_id, i as u64, ts, "search", event.clone());
            let line = encode_stream_frame_ndjson(&frame).unwrap();
            output.push(line);
        }

        // Verify all lines decode correctly and sequence is monotonic
        assert_eq!(output.len(), 5);
        for (i, line) in output.iter().enumerate() {
            let frame: StreamFrame<Vec<String>> = decode_stream_frame_ndjson(line).unwrap();
            assert_eq!(frame.seq, i as u64);
            assert_eq!(frame.stream_id, stream_id);
        }

        // First event is Started, last is Terminal
        let first: StreamFrame<Vec<String>> = decode_stream_frame_ndjson(&output[0]).unwrap();
        assert_eq!(first.event.kind(), StreamEventKind::Started);
        let last: StreamFrame<Vec<String>> = decode_stream_frame_ndjson(&output[4]).unwrap();
        assert_eq!(last.event.kind(), StreamEventKind::Terminal);
    }

    #[test]
    fn validate_requires_schema_identity() {
        let mut frame = sample_frame(StreamEvent::Result(StreamResultEvent {
            rank: 1,
            item: vec!["doc-1".to_owned()],
        }));
        frame.schema_version = "fsfs.stream.query.v2".to_owned();

        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.schema_tag.mismatch")
        );
    }

    // ─── bd-zp7z tests begin ───

    #[test]
    fn protocol_constants() {
        assert_eq!(STREAM_PROTOCOL_VERSION, 1);
        assert_eq!(STREAM_SCHEMA_VERSION, "fsfs.stream.query.v1");
        assert_eq!(TOON_STREAM_RECORD_SEPARATOR, '\u{001E}');
        assert_eq!(TOON_STREAM_RECORD_SEPARATOR_BYTE, 0x1E);
        assert_eq!(SUBSYSTEM, "fsfs_stream_protocol");
    }

    #[test]
    fn event_kind_result() {
        let event = StreamEvent::<String>::Result(StreamResultEvent {
            rank: 1,
            item: "doc".into(),
        });
        assert_eq!(event.kind(), StreamEventKind::Result);
    }

    #[test]
    fn event_kind_warning() {
        let event = StreamEvent::<()>::Warning(StreamWarningEvent {
            warning: OutputWarning::new("test_warn", "warning message"),
        });
        assert_eq!(event.kind(), StreamEventKind::Warning);
    }

    #[test]
    fn stream_frame_new_sets_version_and_schema() {
        let frame: StreamFrame<()> = StreamFrame::new(
            "stream-1",
            0,
            "2026-01-01T00:00:00Z",
            "search",
            StreamEvent::Terminal(terminal_event_completed()),
        );
        assert_eq!(frame.v, STREAM_PROTOCOL_VERSION);
        assert_eq!(frame.schema_version, STREAM_SCHEMA_VERSION);
        assert_eq!(frame.stream_id, "stream-1");
        assert_eq!(frame.seq, 0);
        assert_eq!(frame.ts, "2026-01-01T00:00:00Z");
        assert_eq!(frame.command, "search");
    }

    #[test]
    fn failure_category_index_errors() {
        assert_eq!(
            failure_category_for_error(&SearchError::IndexCorrupted {
                path: "test".into(),
                detail: "bad".into(),
            }),
            StreamFailureCategory::Index
        );
        assert_eq!(
            failure_category_for_error(&SearchError::IndexNotFound {
                path: "test".into(),
            }),
            StreamFailureCategory::Index
        );
        assert_eq!(
            failure_category_for_error(&SearchError::DimensionMismatch {
                expected: 256,
                found: 384,
            }),
            StreamFailureCategory::Index
        );
    }

    #[test]
    fn failure_category_model_errors() {
        assert_eq!(
            failure_category_for_error(&SearchError::EmbedderUnavailable {
                model: "test".into(),
                reason: "not loaded".into(),
            }),
            StreamFailureCategory::Model
        );
        assert_eq!(
            failure_category_for_error(&SearchError::RerankFailed {
                model: "test".into(),
                source: "fail".into(),
            }),
            StreamFailureCategory::Model
        );
        assert_eq!(
            failure_category_for_error(&SearchError::ModelNotFound {
                name: "test".into(),
            }),
            StreamFailureCategory::Model
        );
    }

    #[test]
    fn failure_category_resource_errors() {
        assert_eq!(
            failure_category_for_error(&SearchError::SearchTimeout {
                budget_ms: 100,
                elapsed_ms: 200,
            }),
            StreamFailureCategory::Resource
        );
        assert_eq!(
            failure_category_for_error(&SearchError::QueueFull {
                pending: 50,
                capacity: 100
            }),
            StreamFailureCategory::Resource
        );
    }

    #[test]
    fn failure_category_io_errors() {
        assert_eq!(
            failure_category_for_error(&SearchError::Io(io::Error::other("disk"))),
            StreamFailureCategory::Io
        );
        assert_eq!(
            failure_category_for_error(&SearchError::DurabilityDisabled),
            StreamFailureCategory::Io
        );
    }

    #[test]
    fn failure_category_internal_errors() {
        assert_eq!(
            failure_category_for_error(&SearchError::SubsystemError {
                subsystem: "test",
                source: "bug".into(),
            }),
            StreamFailureCategory::Internal
        );
    }

    #[test]
    fn is_retryable_error_true_cases() {
        assert!(is_retryable_error(&SearchError::EmbeddingFailed {
            model: "m".into(),
            source: Box::new(io::Error::other("e")),
        }));
        assert!(is_retryable_error(&SearchError::SearchTimeout {
            budget_ms: 100,
            elapsed_ms: 200,
        }));
        assert!(is_retryable_error(&SearchError::Io(io::Error::other("e"))));
        assert!(is_retryable_error(&SearchError::QueueFull {
            pending: 5,
            capacity: 10
        }));
        assert!(is_retryable_error(&SearchError::SubsystemError {
            subsystem: "test",
            source: "e".into(),
        }));
        assert!(is_retryable_error(&SearchError::RerankFailed {
            model: "m".into(),
            source: "e".into(),
        }));
    }

    #[test]
    fn is_retryable_error_false_cases() {
        assert!(!is_retryable_error(&SearchError::InvalidConfig {
            field: "f".into(),
            value: "v".into(),
            reason: "r".into(),
        }));
        assert!(!is_retryable_error(&SearchError::IndexCorrupted {
            path: "p".into(),
            detail: "d".into(),
        }));
        assert!(!is_retryable_error(&SearchError::DimensionMismatch {
            expected: 256,
            found: 384,
        }));
        assert!(!is_retryable_error(&SearchError::Cancelled {
            phase: "p".into(),
            reason: "r".into(),
        }));
    }

    #[test]
    fn validate_empty_stream_id() {
        let mut frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "x".into(),
            query: "q".into(),
            format: "jsonl".into(),
        }));
        frame.stream_id = String::new();
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.stream_id.empty")
        );
    }

    #[test]
    fn validate_empty_timestamp() {
        let mut frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "x".into(),
            query: "q".into(),
            format: "jsonl".into(),
        }));
        frame.ts = String::new();
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.timestamp.empty")
        );
    }

    #[test]
    fn validate_empty_command() {
        let mut frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "x".into(),
            query: "q".into(),
            format: "jsonl".into(),
        }));
        frame.command = String::new();
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.command.empty")
        );
    }

    #[test]
    fn validate_failed_terminal_missing_category_and_error() {
        let frame = sample_frame(StreamEvent::Terminal(StreamTerminalEvent {
            status: StreamTerminalStatus::Failed,
            exit_code: 0,
            failure_category: None,
            error: None,
            retry: StreamRetryDirective::None,
        }));
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.failed_zero_exit")
        );
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.failed_missing_category")
        );
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.failed_missing_error")
        );
    }

    #[test]
    fn validate_cancelled_terminal_exit_mismatch() {
        let frame = sample_frame(StreamEvent::Terminal(StreamTerminalEvent {
            status: StreamTerminalStatus::Cancelled,
            exit_code: 1,
            failure_category: None,
            error: None,
            retry: StreamRetryDirective::None,
        }));
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.cancelled_exit_mismatch")
        );
    }

    #[test]
    fn validate_cancelled_terminal_with_retry() {
        let frame = sample_frame(StreamEvent::Terminal(StreamTerminalEvent {
            status: StreamTerminalStatus::Cancelled,
            exit_code: crate::adapters::cli::exit_code::INTERRUPTED,
            failure_category: None,
            error: None,
            retry: StreamRetryDirective::RetryAfterMs {
                delay_ms: 100,
                next_attempt: 1,
                max_attempts: 3,
            },
        }));
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.terminal.cancelled_has_retry")
        );
    }

    #[test]
    fn stream_terminal_status_serde_roundtrip() {
        for status in [
            StreamTerminalStatus::Completed,
            StreamTerminalStatus::Failed,
            StreamTerminalStatus::Cancelled,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: StreamTerminalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn stream_failure_category_serde_roundtrip() {
        for cat in [
            StreamFailureCategory::Config,
            StreamFailureCategory::Index,
            StreamFailureCategory::Model,
            StreamFailureCategory::Resource,
            StreamFailureCategory::Io,
            StreamFailureCategory::Internal,
        ] {
            let json = serde_json::to_string(&cat).unwrap();
            let back: StreamFailureCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(back, cat);
        }
    }

    #[test]
    fn stream_retry_directive_serde_none() {
        let d = StreamRetryDirective::None;
        let json = serde_json::to_string(&d).unwrap();
        let back: StreamRetryDirective = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn stream_retry_directive_serde_retry_after() {
        let d = StreamRetryDirective::RetryAfterMs {
            delay_ms: 500,
            next_attempt: 2,
            max_attempts: 5,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: StreamRetryDirective = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn stream_retry_directive_serde_exhausted() {
        let d = StreamRetryDirective::RetryExhausted { exhausted_after: 3 };
        let json = serde_json::to_string(&d).unwrap();
        let back: StreamRetryDirective = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn decode_ndjson_invalid_json() {
        let result = decode_stream_frame_ndjson::<String>("not valid json{{{");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SearchError::SubsystemError { .. }));
    }

    #[test]
    fn retry_directive_max_attempts_zero() {
        // max_attempts=0 should always return None regardless of error type.
        let err = SearchError::Io(io::Error::other("transient"));
        let directive = retry_directive_for_error(&err, 0, 0);
        assert!(matches!(directive, StreamRetryDirective::None));
    }

    #[test]
    fn retry_backoff_progressive() {
        // Verify backoff increases progressively.
        let mut prev = 0;
        for attempt in 0..8 {
            let delay = retry_backoff_ms(attempt);
            assert!(
                delay >= prev,
                "backoff should be non-decreasing: attempt {attempt}, {delay} >= {prev}"
            );
            prev = delay;
        }
    }

    #[test]
    fn terminal_event_from_error_max_attempts_zero() {
        // max_attempts=0 means no retries regardless of error type.
        let err = SearchError::Io(io::Error::other("transient"));
        let terminal = terminal_event_from_error(&err, 0, 0);
        assert!(matches!(terminal.retry, StreamRetryDirective::None));
    }

    #[test]
    fn stream_event_kind_serde_roundtrip() {
        for kind in [
            StreamEventKind::Started,
            StreamEventKind::Progress,
            StreamEventKind::Result,
            StreamEventKind::Explain,
            StreamEventKind::Warning,
            StreamEventKind::Terminal,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: StreamEventKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn progress_event_total_units_none_skipped_in_json() {
        let event = StreamProgressEvent {
            stage: "retrieve.fast".into(),
            completed_units: 50,
            total_units: None,
            reason_code: "query.phase.initial".into(),
            message: "in progress".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("total_units"));
    }

    #[test]
    fn validate_wrong_protocol_version() {
        let mut frame = sample_frame(StreamEvent::Started(StreamStartedEvent {
            stream_id: "x".into(),
            query: "q".into(),
            format: "jsonl".into(),
        }));
        frame.v = 99;
        let validation = validate_stream_frame(&frame);
        assert!(!validation.valid);
        assert!(
            validation
                .violations
                .iter()
                .any(|v| v.code == "stream.schema_version.mismatch")
        );
    }

    // ─── bd-zp7z tests end ───
}
