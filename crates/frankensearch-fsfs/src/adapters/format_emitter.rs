//! Unified output format emitter for fsfs CLI responses.
//!
//! Dispatches serialization of [`OutputEnvelope`] payloads based on the
//! selected [`OutputFormat`], ensuring semantic parity across JSON, TOON,
//! JSONL, CSV, and human-readable table modes.
//!
//! # Parity Contract
//!
//! JSON and TOON modes produce semantically equivalent output: encoding an
//! envelope as TOON and decoding it back yields the same `serde_json::Value`
//! as direct JSON serialization. This is verified by [`verify_json_toon_parity`].

use std::io::{self, Write};

use frankensearch_core::{SearchError, SearchResult};
use serde::Serialize;

use super::cli::OutputFormat;
use crate::output_schema::{OutputEnvelope, OutputMeta, encode_envelope_toon};
use crate::stream_protocol::{
    StreamFrame, TOON_STREAM_RECORD_SEPARATOR_BYTE, encode_stream_frame_ndjson,
    encode_stream_frame_toon,
};

const SUBSYSTEM: &str = "fsfs_format_emitter";

// ─── Format Emission ────────────────────────────────────────────────────────

/// Emit an [`OutputEnvelope`] to a writer in the requested format.
///
/// # Supported formats
///
/// | Format | Behaviour |
/// |--------|-----------|
/// | `Json` | Pretty-printed JSON with 2-space indent |
/// | `Toon` | TOON encoding via `toon-rust` |
/// | `Jsonl` | Compact single-line JSON (no trailing newline added by serializer) |
/// | `Table` | Human-readable key/value table |
/// | `Csv` | Not yet implemented for envelopes (returns error) |
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
pub fn emit_envelope<T, W>(
    envelope: &OutputEnvelope<T>,
    format: OutputFormat,
    writer: &mut W,
) -> SearchResult<()>
where
    T: Serialize,
    W: Write,
{
    match format {
        OutputFormat::Json => emit_json(envelope, writer),
        OutputFormat::Toon => emit_toon(envelope, writer),
        OutputFormat::Jsonl => emit_jsonl(envelope, writer),
        OutputFormat::Table => emit_table(envelope, writer),
        OutputFormat::Csv => Err(SearchError::InvalidConfig {
            field: "format".into(),
            value: "csv".into(),
            reason: "CSV format is not supported for envelope output; use json or toon".into(),
        }),
    }
}

/// Emit an [`OutputEnvelope`] as a string in the requested format.
///
/// Convenience wrapper around [`emit_envelope`] that returns the output as
/// a `String`.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
pub fn emit_envelope_string<T>(
    envelope: &OutputEnvelope<T>,
    format: OutputFormat,
) -> SearchResult<String>
where
    T: Serialize,
{
    let mut buf = Vec::new();
    emit_envelope(envelope, format, &mut buf)?;
    String::from_utf8(buf).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "output contained invalid UTF-8: {source}"
        ))),
    })
}

/// Emit a single stream frame in NDJSON or TOON stream transport format.
///
/// TOON stream frames are emitted as:
/// - `0x1E` record-separator prefix
/// - TOON payload bytes
/// - trailing newline (`\\n`)
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when `format` is not `jsonl` or `toon`.
pub fn emit_stream_frame<T, W>(
    frame: &StreamFrame<T>,
    format: OutputFormat,
    writer: &mut W,
) -> SearchResult<()>
where
    T: Serialize,
    W: Write,
{
    match format {
        OutputFormat::Jsonl => {
            let line = encode_stream_frame_ndjson(frame)?;
            writer.write_all(line.as_bytes()).map_err(write_err)
        }
        OutputFormat::Toon => {
            let toon = encode_stream_frame_toon(frame)?;
            writer
                .write_all(&[TOON_STREAM_RECORD_SEPARATOR_BYTE])
                .map_err(write_err)?;
            writer.write_all(toon.as_bytes()).map_err(write_err)?;
            writer.write_all(b"\n").map_err(write_err)
        }
        _ => Err(SearchError::InvalidConfig {
            field: "format".into(),
            value: format.to_string(),
            reason: "stream mode supports only jsonl and toon".into(),
        }),
    }
}

/// Emit a single stream frame and return the transport payload as a string.
///
/// # Errors
///
/// Returns `SearchError` when stream emission fails.
pub fn emit_stream_frame_string<T>(
    frame: &StreamFrame<T>,
    format: OutputFormat,
) -> SearchResult<String>
where
    T: Serialize,
{
    let mut buf = Vec::new();
    emit_stream_frame(frame, format, &mut buf)?;
    String::from_utf8(buf).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "stream output contained invalid UTF-8: {source}"
        ))),
    })
}

// ─── JSON ───────────────────────────────────────────────────────────────────

fn emit_json<T: Serialize, W: Write>(
    envelope: &OutputEnvelope<T>,
    writer: &mut W,
) -> SearchResult<()> {
    serde_json::to_writer_pretty(writer, envelope).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to serialize envelope as JSON: {source}"
        ))),
    })
}

// ─── TOON ───────────────────────────────────────────────────────────────────

fn emit_toon<T: Serialize, W: Write>(
    envelope: &OutputEnvelope<T>,
    writer: &mut W,
) -> SearchResult<()> {
    let toon = encode_envelope_toon(envelope)?;
    writer
        .write_all(toon.as_bytes())
        .map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(source),
        })
}

// ─── JSONL ──────────────────────────────────────────────────────────────────

fn emit_jsonl<T: Serialize, W: Write>(
    envelope: &OutputEnvelope<T>,
    writer: &mut W,
) -> SearchResult<()> {
    serde_json::to_writer(&mut *writer, envelope).map_err(|source| {
        SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "failed to serialize envelope as JSONL: {source}"
            ))),
        }
    })?;
    writer
        .write_all(b"\n")
        .map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(source),
        })
}

// ─── Table ──────────────────────────────────────────────────────────────────

fn emit_table<T: Serialize, W: Write>(
    envelope: &OutputEnvelope<T>,
    writer: &mut W,
) -> SearchResult<()> {
    // Table mode renders a human-readable summary. For errors, show the
    // error code and message. For success, show a JSON representation of
    // the data since we don't have type-specific table renderers yet.
    if envelope.ok {
        if let Some(data) = &envelope.data {
            let json = serde_json::to_string_pretty(data).map_err(|source| {
                SearchError::SubsystemError {
                    subsystem: SUBSYSTEM,
                    source: Box::new(io::Error::other(format!(
                        "failed to serialize data for table display: {source}"
                    ))),
                }
            })?;
            write!(writer, "{json}").map_err(write_err)?;
        }
    } else if let Some(error) = &envelope.error {
        write!(writer, "error: [{}] {}", error.code, error.message).map_err(write_err)?;
        if let Some(field) = &error.field {
            write!(writer, " (field: {field})").map_err(write_err)?;
        }
        if let Some(context) = &error.context {
            write!(writer, "\n\n  {context}").map_err(write_err)?;
        }
        if let Some(suggestion) = &error.suggestion {
            write!(writer, "\n\n  Fix: {suggestion}").map_err(write_err)?;
        }
    }

    for warning in &envelope.warnings {
        write!(writer, "\nwarning: [{}] {}", warning.code, warning.message).map_err(write_err)?;
    }

    if let Some(ms) = envelope.meta.duration_ms {
        write!(writer, "\n({ms}ms)").map_err(write_err)?;
    }

    Ok(())
}

fn write_err(source: io::Error) -> SearchError {
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(source),
    }
}

// ─── Parity Verification ────────────────────────────────────────────────────

/// Verify that JSON and TOON representations of an envelope are semantically
/// equivalent at the typed level.
///
/// TOON may transform certain value types during encoding/decoding (e.g.,
/// ISO 8601 timestamps become date literals). This function verifies parity
/// by comparing the *typed deserialization* result rather than raw
/// `serde_json::Value` trees, which correctly accounts for TOON's type
/// coercions.
///
/// The check serializes the envelope as JSON and TOON, deserializes both
/// back to `serde_json::Value` via the typed `OutputEnvelope` intermediary,
/// and compares the re-serialized values.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if encoding fails or parity is
/// violated.
pub fn verify_json_toon_parity<T>(envelope: &OutputEnvelope<T>) -> SearchResult<()>
where
    T: Serialize + serde::de::DeserializeOwned,
{
    use crate::output_schema::decode_envelope_toon;

    // JSON path: serialize → deserialize → re-serialize to Value
    let json_text =
        serde_json::to_string(envelope).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "parity check: failed to serialize to JSON: {source}"
            ))),
        })?;
    let json_roundtrip: OutputEnvelope<T> =
        serde_json::from_str(&json_text).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "parity check: failed to deserialize JSON: {source}"
            ))),
        })?;
    let json_value =
        serde_json::to_value(&json_roundtrip).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "parity check: failed to re-serialize JSON roundtrip: {source}"
            ))),
        })?;

    // TOON path: encode → decode (with coercion) → re-serialize to Value
    let toon_text = encode_envelope_toon(envelope)?;
    let toon_roundtrip: OutputEnvelope<T> =
        decode_envelope_toon(&toon_text).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "parity check: failed to decode TOON: {source}"
            ))),
        })?;
    let toon_value =
        serde_json::to_value(&toon_roundtrip).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "parity check: failed to re-serialize TOON roundtrip: {source}"
            ))),
        })?;

    if json_value != toon_value {
        return Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "JSON/TOON parity violation: re-serialized values differ.\nJSON: {json_value}\nTOON: {toon_value}"
            ))),
        });
    }

    Ok(())
}

// ─── Format Detection ───────────────────────────────────────────────────────

/// Create an [`OutputMeta`] with the format field set from an [`OutputFormat`].
#[must_use]
pub fn meta_for_format(command: &str, format: OutputFormat) -> OutputMeta {
    OutputMeta::new(command, format.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output_schema::{
        OutputEnvelope, OutputError, OutputErrorCode, OutputMeta, OutputWarning, OutputWarningCode,
        decode_envelope_toon,
    };
    use crate::stream_protocol::{
        StreamEvent, StreamFrame, StreamResultEvent, TOON_STREAM_RECORD_SEPARATOR_BYTE,
        decode_stream_frame_ndjson, decode_stream_frame_toon,
    };

    fn sample_ts() -> &'static str {
        "2026-02-14T12:00:00Z"
    }

    fn sample_meta(format: &str) -> OutputMeta {
        OutputMeta::new("search", format)
    }

    fn sample_stream_frame() -> StreamFrame<Vec<String>> {
        StreamFrame::new(
            "01JAH9A2W8F8Q6GQ4C7M3N2P1R",
            1,
            sample_ts(),
            "search",
            StreamEvent::Result(StreamResultEvent {
                rank: 1,
                item: vec!["doc-1".to_owned(), "doc-2".to_owned()],
            }),
        )
    }

    // ─── JSON emission ──────────────────────────────────────────────────

    #[test]
    fn emit_json_success_envelope() {
        let env = OutputEnvelope::success(vec!["doc-1", "doc-2"], sample_meta("json"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Json).unwrap();
        assert!(output.contains("\"ok\": true"));
        assert!(output.contains("\"doc-1\""));
        assert!(output.contains("\"doc-2\""));
        assert!(output.contains("\"format\": \"json\""));
    }

    #[test]
    fn emit_json_error_envelope() {
        let err = OutputError::new(OutputErrorCode::INDEX_NOT_FOUND, "not found", 1);
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("json"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Json).unwrap();
        assert!(output.contains("\"ok\": false"));
        assert!(output.contains("\"index_not_found\""));
    }

    // ─── TOON emission ──────────────────────────────────────────────────

    #[test]
    fn emit_toon_success_envelope() {
        let env = OutputEnvelope::success(vec!["doc-1", "doc-2"], sample_meta("toon"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Toon).unwrap();
        // TOON output should be non-empty and decodable
        assert!(!output.is_empty());
        let decoded: OutputEnvelope<Vec<String>> = decode_envelope_toon(&output).unwrap();
        assert!(decoded.ok);
        assert_eq!(decoded.data.as_ref().unwrap(), &["doc-1", "doc-2"]);
    }

    #[test]
    fn emit_toon_error_envelope() {
        let err = OutputError::new(OutputErrorCode::INVALID_CONFIG, "bad value", 2)
            .with_field("quality_weight");
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("toon"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Toon).unwrap();
        assert!(!output.is_empty());
        let decoded: OutputEnvelope<()> = decode_envelope_toon(&output).unwrap();
        assert!(!decoded.ok);
        assert_eq!(decoded.error.as_ref().unwrap().code, "invalid_config");
    }

    #[test]
    fn emit_toon_with_warnings() {
        let env =
            OutputEnvelope::success(42u32, sample_meta("toon"), sample_ts()).with_warnings(vec![
                OutputWarning::new(OutputWarningCode::DEGRADED_MODE, "quality tier skipped"),
                OutputWarning::new(OutputWarningCode::FAST_ONLY_RESULTS, "fast only"),
            ]);
        let output = emit_envelope_string(&env, OutputFormat::Toon).unwrap();
        let decoded: OutputEnvelope<u32> = decode_envelope_toon(&output).unwrap();
        assert_eq!(decoded.warnings.len(), 2);
        assert_eq!(decoded.warnings[0].code, "degraded_mode");
        assert_eq!(decoded.warnings[1].code, "fast_only_results");
    }

    // ─── JSONL emission ─────────────────────────────────────────────────

    #[test]
    fn emit_jsonl_is_single_line() {
        let env = OutputEnvelope::success("data", sample_meta("jsonl"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Jsonl).unwrap();
        // JSONL: single line ending with \n
        let lines: Vec<&str> = output.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 1, "JSONL must be a single line, got: {output}");
        assert!(output.ends_with('\n'));
        // Should be valid JSON
        let _: OutputEnvelope<String> = serde_json::from_str(lines[0]).unwrap();
    }

    // ─── Stream emission ────────────────────────────────────────────────

    #[test]
    fn emit_stream_frame_jsonl_roundtrip() {
        let frame = sample_stream_frame();
        let output = emit_stream_frame_string(&frame, OutputFormat::Jsonl).unwrap();
        assert!(output.ends_with('\n'));

        let decoded: StreamFrame<Vec<String>> = decode_stream_frame_ndjson(&output).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn emit_stream_frame_toon_with_record_separator() {
        let frame = sample_stream_frame();
        let output = emit_stream_frame_string(&frame, OutputFormat::Toon).unwrap();
        let bytes = output.as_bytes();
        assert_eq!(
            bytes.first().copied(),
            Some(TOON_STREAM_RECORD_SEPARATOR_BYTE)
        );
        assert!(output.ends_with('\n'));

        let toon_payload = &output[1..output.len() - 1];
        let decoded: StreamFrame<Vec<String>> = decode_stream_frame_toon(toon_payload).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn emit_stream_frame_rejects_non_stream_format() {
        let frame = sample_stream_frame();
        let err = emit_stream_frame_string(&frame, OutputFormat::Json).unwrap_err();
        assert!(
            err.to_string()
                .contains("stream mode supports only jsonl and toon")
        );
    }

    // ─── Table emission ─────────────────────────────────────────────────

    #[test]
    fn emit_table_success_shows_data() {
        let env = OutputEnvelope::success("hello world", sample_meta("table"), sample_ts())
            .with_warnings(vec![OutputWarning::new("degraded_mode", "degraded")]);
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("hello world"));
        assert!(output.contains("warning: [degraded_mode]"));
    }

    #[test]
    fn emit_table_error_shows_code_and_message() {
        let err = OutputError::new("io_error", "disk full", 1);
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("table"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("error: [io_error] disk full"));
    }

    #[test]
    fn emit_table_error_with_field() {
        let err = OutputError::new("invalid_config", "bad", 2).with_field("timeout_ms");
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("table"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("(field: timeout_ms)"));
    }

    #[test]
    fn emit_table_shows_duration() {
        let meta = OutputMeta::new("search", "table").with_duration_ms(42);
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("(42ms)"));
    }

    // ─── CSV unsupported ────────────────────────────────────────────────

    #[test]
    fn emit_csv_returns_error() {
        let env = OutputEnvelope::success("ok", sample_meta("csv"), sample_ts());
        let err = emit_envelope_string(&env, OutputFormat::Csv).unwrap_err();
        assert!(err.to_string().contains("CSV format is not supported"));
    }

    // ─── JSON/TOON Parity ───────────────────────────────────────────────

    #[test]
    fn json_toon_parity_success_envelope() {
        let env = OutputEnvelope::success(
            vec![
                "result-a".to_string(),
                "result-b".to_string(),
                "result-c".to_string(),
            ],
            OutputMeta::new("search", "json").with_duration_ms(15),
            sample_ts(),
        );
        verify_json_toon_parity(&env).expect("parity should hold for success envelope");
    }

    #[test]
    fn json_toon_parity_error_envelope() {
        let err = OutputError::new(OutputErrorCode::SEARCH_TIMEOUT, "timeout after 50ms", 1);
        let env: OutputEnvelope<()> = OutputEnvelope::error(
            err,
            OutputMeta::new("search", "json").with_duration_ms(50),
            sample_ts(),
        );
        verify_json_toon_parity(&env).expect("parity should hold for error envelope");
    }

    #[test]
    fn json_toon_parity_with_warnings() {
        #[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq)]
        struct SearchSummary {
            hits: u32,
            scores: Vec<f64>,
        }

        let env = OutputEnvelope::success(
            SearchSummary {
                hits: 42,
                scores: vec![0.9, 0.85, 0.7],
            },
            OutputMeta::new("search", "json")
                .with_duration_ms(33)
                .with_request_id("01JAH9TEST"),
            sample_ts(),
        )
        .with_warnings(vec![
            OutputWarning::new(OutputWarningCode::DEGRADED_MODE, "quality tier skipped"),
            OutputWarning::new(OutputWarningCode::RERANK_SKIPPED, "reranker circuit open"),
        ]);
        verify_json_toon_parity(&env).expect("parity should hold with warnings");
    }

    #[test]
    fn json_toon_parity_empty_data() {
        let env = OutputEnvelope::success(
            Vec::<String>::new(),
            OutputMeta::new("search", "json"),
            sample_ts(),
        );
        verify_json_toon_parity(&env).expect("parity should hold for empty results");
    }

    #[test]
    fn json_toon_parity_with_optional_fields() {
        // Envelope with all optional fields absent
        let env = OutputEnvelope::success(
            "minimal".to_string(),
            OutputMeta::new("status", "json"),
            sample_ts(),
        );
        verify_json_toon_parity(&env).expect("parity should hold with minimal fields");

        // Envelope with all optional fields present
        let env = OutputEnvelope::success(
            "full".to_string(),
            OutputMeta::new("status", "json")
                .with_duration_ms(100)
                .with_request_id("01FULL"),
            sample_ts(),
        )
        .with_warnings(vec![OutputWarning::new(
            "schema_version_newer",
            "v2 detected",
        )]);
        verify_json_toon_parity(&env).expect("parity should hold with all optional fields");
    }

    // ─── meta_for_format ────────────────────────────────────────────────

    #[test]
    fn meta_for_format_sets_format_string() {
        let meta = meta_for_format("search", OutputFormat::Toon);
        assert_eq!(meta.format, "toon");
        assert_eq!(meta.command, "search");

        let meta = meta_for_format("status", OutputFormat::Json);
        assert_eq!(meta.format, "json");
    }

    // ─── Cross-format roundtrip ─────────────────────────────────────────

    #[test]
    fn json_and_toon_produce_same_typed_envelope() {
        let env = OutputEnvelope::success(
            vec![1u32, 2, 3, 4, 5],
            OutputMeta::new("search", "json").with_duration_ms(7),
            sample_ts(),
        );

        let json_str = emit_envelope_string(&env, OutputFormat::Json).unwrap();
        let toon_str = emit_envelope_string(&env, OutputFormat::Toon).unwrap();

        // Decode both via typed deserialization
        let json_env: OutputEnvelope<Vec<u32>> = serde_json::from_str(&json_str).unwrap();
        let toon_env: OutputEnvelope<Vec<u32>> = decode_envelope_toon(&toon_str).unwrap();

        // Compare structural fields (timestamps may differ in representation)
        assert_eq!(json_env.ok, toon_env.ok);
        assert_eq!(json_env.v, toon_env.v);
        assert_eq!(json_env.data, toon_env.data);
        assert_eq!(json_env.meta.command, toon_env.meta.command);
        assert_eq!(json_env.meta.duration_ms, toon_env.meta.duration_ms);
    }

    #[test]
    fn jsonl_and_json_produce_same_decoded_value() {
        let env =
            OutputEnvelope::success("same-data", OutputMeta::new("search", "json"), sample_ts());

        let expected_text = emit_envelope_string(&env, OutputFormat::Json).unwrap();
        let line_mode_text = emit_envelope_string(&env, OutputFormat::Jsonl).unwrap();

        let expected_value: serde_json::Value = serde_json::from_str(&expected_text).unwrap();
        let line_mode_value: serde_json::Value =
            serde_json::from_str(line_mode_text.trim()).unwrap();

        assert_eq!(
            expected_value, line_mode_value,
            "JSON and JSONL decoded values must match"
        );
    }
}
