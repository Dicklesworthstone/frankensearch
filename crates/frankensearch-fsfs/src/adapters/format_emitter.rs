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

use std::env;
use std::fmt::Write as _;
use std::io::{self, IsTerminal, Write};

use frankensearch_core::{SearchError, SearchResult};
use serde::Serialize;

use super::cli::OutputFormat;
use crate::output_schema::{
    OutputEnvelope, OutputMeta, SearchHitPayload, SearchPayload, encode_envelope_toon,
};
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
/// | `Csv` | RFC4180-compatible rows for search payloads, generic payloads, and envelope errors |
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
        OutputFormat::Csv => emit_csv(envelope, writer),
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
            let value =
                serde_json::to_value(data).map_err(|source| SearchError::SubsystemError {
                    subsystem: SUBSYSTEM,
                    source: Box::new(io::Error::other(format!(
                        "failed to project data for table display: {source}"
                    ))),
                })?;

            if let Ok(search_payload) = serde_json::from_value::<SearchPayload>(value.clone()) {
                write!(
                    writer,
                    "{}",
                    render_search_table(&search_payload, envelope.meta.duration_ms)
                )
                .map_err(write_err)?;
                return Ok(());
            }

            let json = serde_json::to_string_pretty(&value).map_err(|source| {
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

fn emit_csv<T: Serialize, W: Write>(
    envelope: &OutputEnvelope<T>,
    writer: &mut W,
) -> SearchResult<()> {
    if envelope.ok {
        let data = envelope
            .data
            .as_ref()
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "format".into(),
                value: "csv".into(),
                reason: "csv success output requires a payload".into(),
            })?;
        let payload_json =
            serde_json::to_string(data).map_err(|source| SearchError::SubsystemError {
                subsystem: SUBSYSTEM,
                source: Box::new(io::Error::other(format!(
                    "failed to project data for csv display: {source}"
                ))),
            })?;

        if envelope.meta.command.eq_ignore_ascii_case("search")
            && let Ok(search_payload) = serde_json::from_str::<SearchPayload>(&payload_json)
        {
            return write_search_payload_csv(&search_payload, writer);
        }

        return write_generic_payload_csv(&payload_json, writer);
    }

    write_csv_row(
        writer,
        &[
            "ok".to_owned(),
            "error_code".to_owned(),
            "error_message".to_owned(),
            "error_field".to_owned(),
            "error_suggestion".to_owned(),
            "error_context".to_owned(),
        ],
    )?;

    if let Some(error) = &envelope.error {
        write_csv_row(
            writer,
            &[
                "false".to_owned(),
                error.code.clone(),
                error.message.clone(),
                error.field.clone().unwrap_or_default(),
                error.suggestion.clone().unwrap_or_default(),
                error.context.clone().unwrap_or_default(),
            ],
        )
    } else {
        write_csv_row(
            writer,
            &[
                "false".to_owned(),
                "unknown_error".to_owned(),
                "envelope marked as error but no error payload provided".to_owned(),
                String::new(),
                String::new(),
                String::new(),
            ],
        )
    }
}

fn write_generic_payload_csv<W: Write>(payload_json: &str, writer: &mut W) -> SearchResult<()> {
    write_csv_row(writer, &["data_json".to_owned()])?;
    write_csv_row(writer, &[payload_json.to_owned()])
}

fn write_search_payload_csv<W: Write>(payload: &SearchPayload, writer: &mut W) -> SearchResult<()> {
    write_csv_row(
        writer,
        &[
            "query".to_owned(),
            "phase".to_owned(),
            "total_candidates".to_owned(),
            "returned_hits".to_owned(),
            "rank".to_owned(),
            "path".to_owned(),
            "score".to_owned(),
            "in_both_sources".to_owned(),
            "lexical_rank".to_owned(),
            "semantic_rank".to_owned(),
            "snippet".to_owned(),
        ],
    )?;

    for hit in &payload.hits {
        let lexical_rank = hit
            .lexical_rank
            .map(|rank| rank.saturating_add(1).to_string())
            .unwrap_or_default();
        let semantic_rank = hit
            .semantic_rank
            .map(|rank| rank.saturating_add(1).to_string())
            .unwrap_or_default();
        write_csv_row(
            writer,
            &[
                payload.query.clone(),
                payload.phase.to_string(),
                payload.total_candidates.to_string(),
                payload.returned_hits.to_string(),
                hit.rank.to_string(),
                hit.path.clone(),
                format!("{:.6}", hit.score),
                hit.in_both_sources.to_string(),
                lexical_rank,
                semantic_rank,
                hit.snippet.clone().unwrap_or_default(),
            ],
        )?;
    }

    Ok(())
}

fn write_csv_row<W: Write>(writer: &mut W, fields: &[String]) -> SearchResult<()> {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            writer.write_all(b",").map_err(write_err)?;
        }
        write_csv_field(writer, field)?;
    }
    writer.write_all(b"\n").map_err(write_err)
}

fn write_csv_field<W: Write>(writer: &mut W, field: &str) -> SearchResult<()> {
    let needs_quotes = field.contains(',') || field.contains('\n') || field.contains('\r');
    if !needs_quotes && !field.contains('"') {
        return writer.write_all(field.as_bytes()).map_err(write_err);
    }

    writer.write_all(b"\"").map_err(write_err)?;
    for ch in field.chars() {
        if ch == '"' {
            writer.write_all(b"\"\"").map_err(write_err)?;
        } else {
            let mut utf8 = [0_u8; 4];
            let encoded = ch.encode_utf8(&mut utf8);
            writer.write_all(encoded.as_bytes()).map_err(write_err)?;
        }
    }
    writer.write_all(b"\"").map_err(write_err)
}

fn render_search_table(payload: &SearchPayload, duration_ms: Option<u64>) -> String {
    let color_enabled = should_use_ansi_color();
    let width = detect_terminal_width();
    render_search_table_with_options(payload, duration_ms, color_enabled, width)
}

/// Render search results as a table while honoring the CLI `--no-color` flag.
#[must_use]
pub(crate) fn render_search_table_for_cli(
    payload: &SearchPayload,
    duration_ms: Option<u64>,
    no_color: bool,
) -> String {
    let color_enabled = !no_color && should_use_ansi_color();
    let width = detect_terminal_width();
    render_search_table_with_options(payload, duration_ms, color_enabled, width)
}

fn render_search_table_with_options(
    payload: &SearchPayload,
    duration_ms: Option<u64>,
    color_enabled: bool,
    width: usize,
) -> String {
    let mut out = String::new();
    let query_terms = collect_query_terms(&payload.query);
    let total_ms = duration_ms.unwrap_or(0);
    let snippet_width = width.saturating_sub(34).max(32);
    let phase = payload.phase.to_string().to_ascii_uppercase();
    let phase_label = paint(&phase, "1;34", color_enabled);
    let _ = writeln!(
        out,
        "PHASE {phase_label}: {} hit(s) for \"{}\"",
        payload.returned_hits, payload.query
    );

    if payload.is_empty() {
        let _ = writeln!(
            out,
            "No results for \"{}\". Try broadening your search or checking the index with fsfs status.",
            payload.query
        );
        let _ = writeln!(out, "{} results in {total_ms}ms", payload.returned_hits);
        return out;
    }

    for hit in &payload.hits {
        let (path_text, line_number) = split_path_and_line_number(&hit.path);
        let path = paint(path_text, "1;36", color_enabled);
        let line_segment = line_number
            .map(|line| format!(":{}", paint(line, "32", color_enabled)))
            .unwrap_or_default();
        let score = paint(
            &format!("{:.3}", hit.score),
            score_color_code(hit.score),
            color_enabled,
        );
        let source_badge = source_badge(hit, color_enabled);

        let _ = write!(
            out,
            "{:>3}. {}{}  score={}  {}",
            hit.rank, path, line_segment, score, source_badge
        );

        if let (Some(lexical_rank), Some(semantic_rank)) = (hit.lexical_rank, hit.semantic_rank) {
            let _ = write!(out, " [L{} S{}]", lexical_rank + 1, semantic_rank + 1);
        }
        let _ = writeln!(out);
        if let Some(snippet) = hit.snippet.as_deref() {
            let clipped = truncate_for_width(snippet.trim(), snippet_width);
            let highlighted = highlight_query_terms(&clipped, &query_terms, color_enabled);
            let _ = writeln!(out, "     {highlighted}");
        }
    }

    let _ = writeln!(out, "{} results in {total_ms}ms", payload.returned_hits);
    out
}

fn should_use_ansi_color() -> bool {
    if env_var_disables_color() {
        return false;
    }
    std::io::stdout().is_terminal()
}

fn env_var_disables_color() -> bool {
    env::var("FRANKENSEARCH_NO_COLOR")
        .ok()
        .is_some_and(|value| truthy_env(&value))
        || env::var("NO_COLOR")
            .ok()
            .is_some_and(|value| truthy_env(&value))
}

fn truthy_env(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return true;
    }
    !matches!(trimmed, "0" | "false" | "FALSE" | "False")
}

fn detect_terminal_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|width| *width >= 40)
        .unwrap_or(100)
}

fn score_color_code(score: f64) -> &'static str {
    if score >= 0.8 {
        "32"
    } else if score >= 0.5 {
        "33"
    } else {
        "31"
    }
}

fn paint(text: &str, style: &str, color_enabled: bool) -> String {
    if color_enabled {
        format!("\u{1b}[{style}m{text}\u{1b}[0m")
    } else {
        text.to_owned()
    }
}

fn split_path_and_line_number(path: &str) -> (&str, Option<&str>) {
    if let Some((left, right)) = path.rsplit_once(':')
        && !right.is_empty()
        && right.chars().all(|ch| ch.is_ascii_digit())
    {
        return (left, Some(right));
    }
    (path, None)
}

fn source_badge(hit: &SearchHitPayload, color_enabled: bool) -> String {
    if hit.in_both_sources {
        return paint("[both]", "1;32", color_enabled);
    }
    if hit.lexical_rank.is_some() {
        return paint("[lexical]", "33", color_enabled);
    }
    if hit.semantic_rank.is_some() {
        return paint("[semantic]", "36", color_enabled);
    }
    paint("[unknown]", "90", color_enabled)
}

fn truncate_for_width(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
}

fn collect_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .filter(|term| !term.is_empty())
        .collect()
}

fn highlight_query_terms(text: &str, query_terms: &[String], color_enabled: bool) -> String {
    if query_terms.is_empty() {
        return text.to_owned();
    }

    text.split_whitespace()
        .map(|token| {
            let token_lower = token.to_ascii_lowercase();
            if query_terms
                .iter()
                .any(|term| token_lower.contains(term.as_str()))
            {
                paint(token, "1;33", color_enabled)
            } else {
                token.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
        SearchHitPayload, SearchOutputPhase, SearchPayload, decode_envelope_toon,
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
    fn emit_table_search_payload_renders_ranked_hits() {
        let payload = SearchPayload::new(
            "how auth works",
            SearchOutputPhase::Initial,
            2,
            vec![
                SearchHitPayload {
                    rank: 1,
                    path: "src/auth.rs".to_owned(),
                    score: 0.923,
                    snippet: Some("fn authenticate(token: &str) -> bool".to_owned()),
                    lexical_rank: Some(0),
                    semantic_rank: Some(1),
                    in_both_sources: true,
                },
                SearchHitPayload {
                    rank: 2,
                    path: "docs/auth.md".to_owned(),
                    score: 0.811,
                    snippet: None,
                    lexical_rank: Some(2),
                    semantic_rank: None,
                    in_both_sources: false,
                },
            ],
        );
        let env = OutputEnvelope::success(payload, sample_meta("table"), sample_ts());

        let output = emit_envelope_string(&env, OutputFormat::Table).expect("render table");
        assert!(output.contains("PHASE INITIAL"));
        assert!(output.contains("1. src/auth.rs"));
        assert!(output.contains("score=0.923"));
        assert!(output.contains("[both]"));
        assert!(output.contains("[lexical]"));
        assert!(output.contains("fn authenticate(token: &str) -> bool"));
        assert!(output.contains("2 results in 0ms"));
    }

    #[test]
    fn render_search_table_empty_results_shows_guidance() {
        let payload = SearchPayload::new("auth middleware", SearchOutputPhase::Initial, 0, vec![]);
        let output = render_search_table_with_options(&payload, Some(19), false, 80);
        assert!(output.contains("No results for \"auth middleware\"."));
        assert!(output.contains("checking the index with fsfs status"));
        assert!(output.contains("0 results in 19ms"));
    }

    #[test]
    fn render_search_table_uses_ansi_when_color_enabled() {
        let payload = SearchPayload::new(
            "auth",
            SearchOutputPhase::Refined,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "src/auth.rs:45".to_owned(),
                score: 0.91,
                snippet: Some("auth middleware validates bearer token".to_owned()),
                lexical_rank: Some(0),
                semantic_rank: Some(0),
                in_both_sources: true,
            }],
        );
        let output = render_search_table_with_options(&payload, Some(42), true, 80);
        assert!(output.contains("\u{1b}["));
        assert!(output.contains("1 results in 42ms"));
    }

    #[test]
    fn env_var_disables_color_treats_empty_as_true() {
        assert!(truthy_env(""));
        assert!(!truthy_env("0"));
        assert!(!truthy_env("false"));
        assert!(truthy_env("1"));
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
    fn emit_table_error_shows_suggestion() {
        let err = OutputError::new("model_not_found", "model X not found", 78)
            .with_suggestion("fsfs download-models --model X")
            .with_context("Models are required for semantic search");
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("table"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(
            output.contains("Fix: fsfs download-models"),
            "table output should include suggestion: {output}"
        );
        assert!(
            output.contains("Models are required"),
            "table output should include context: {output}"
        );
    }

    #[test]
    fn emit_table_error_without_suggestion_unchanged() {
        let err = OutputError::new("cancelled", "cancelled", 130);
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("table"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("error: [cancelled] cancelled"));
        assert!(
            !output.contains("Fix:"),
            "should not show Fix section when no suggestion"
        );
    }

    #[test]
    fn emit_table_shows_duration() {
        let meta = OutputMeta::new("search", "table").with_duration_ms(42);
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Table).unwrap();
        assert!(output.contains("(42ms)"));
    }

    // ─── CSV emission ────────────────────────────────────────────────────

    #[test]
    fn emit_csv_search_payload_renders_header_and_rows() {
        let payload = SearchPayload::new(
            "auth middleware",
            SearchOutputPhase::Refined,
            3,
            vec![
                SearchHitPayload {
                    rank: 1,
                    path: "src/auth.rs:42".to_owned(),
                    score: 0.9234,
                    snippet: Some("middleware validates bearer tokens".to_owned()),
                    lexical_rank: Some(0),
                    semantic_rank: Some(1),
                    in_both_sources: true,
                },
                SearchHitPayload {
                    rank: 2,
                    path: "docs/auth guide.md".to_owned(),
                    score: 0.811,
                    snippet: Some("quoted \"token\" snippet, with comma".to_owned()),
                    lexical_rank: None,
                    semantic_rank: Some(2),
                    in_both_sources: false,
                },
            ],
        );
        let env = OutputEnvelope::success(payload, sample_meta("csv"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Csv).expect("csv output");
        let mut lines = output.lines();
        assert_eq!(
            lines.next().unwrap_or_default(),
            "query,phase,total_candidates,returned_hits,rank,path,score,in_both_sources,lexical_rank,semantic_rank,snippet"
        );
        assert!(
            lines.next().unwrap_or_default().contains("auth middleware,refined,3,2,1,src/auth.rs:42,0.923400,true,1,2,middleware validates bearer tokens")
        );
        assert!(lines.next().unwrap_or_default().contains(
            "docs/auth guide.md,0.811000,false,,3,\"quoted \"\"token\"\" snippet, with comma\""
        ));
    }

    #[test]
    fn emit_csv_error_outputs_error_row() {
        let err = OutputError::new(OutputErrorCode::MODEL_NOT_FOUND, "model missing", 78)
            .with_field("model")
            .with_suggestion("fsfs download-models --model all-MiniLM-L6-v2");
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta("csv"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Csv).expect("csv error output");
        let mut lines = output.lines();
        assert_eq!(
            lines.next().unwrap_or_default(),
            "ok,error_code,error_message,error_field,error_suggestion,error_context"
        );
        assert!(
            lines
                .next()
                .unwrap_or_default()
                .contains("false,model_not_found,model missing,model,fsfs download-models --model all-MiniLM-L6-v2,")
        );
    }

    #[test]
    fn emit_csv_non_search_success_payload_outputs_json_cell() {
        let env = OutputEnvelope::success("ok", sample_meta("csv"), sample_ts());
        let output = emit_envelope_string(&env, OutputFormat::Csv).expect("csv output");
        let mut lines = output.lines();
        assert_eq!(lines.next().unwrap_or_default(), "data_json");
        assert_eq!(lines.next().unwrap_or_default(), "\"\"\"ok\"\"\"");
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

    // ─── render_search_table_for_cli ────────────────────────────────────

    fn sample_search_payload() -> SearchPayload {
        SearchPayload::new(
            "how does auth work",
            SearchOutputPhase::Refined,
            3,
            vec![
                SearchHitPayload {
                    rank: 1,
                    path: "src/auth/middleware.rs:45".to_owned(),
                    score: 0.923,
                    snippet: Some("JWT validation middleware checks Bearer token".to_owned()),
                    lexical_rank: Some(0),
                    semantic_rank: Some(1),
                    in_both_sources: true,
                },
                SearchHitPayload {
                    rank: 2,
                    path: "src/auth/login.rs:12".to_owned(),
                    score: 0.811,
                    snippet: Some("Login handler with bcrypt password hashing".to_owned()),
                    lexical_rank: Some(2),
                    semantic_rank: None,
                    in_both_sources: false,
                },
                SearchHitPayload {
                    rank: 3,
                    path: "docs/auth.md".to_owned(),
                    score: 0.421,
                    snippet: None,
                    lexical_rank: None,
                    semantic_rank: Some(3),
                    in_both_sources: false,
                },
            ],
        )
    }

    #[test]
    fn render_search_table_for_cli_no_color_strips_ansi() {
        let payload = sample_search_payload();
        let output = render_search_table_for_cli(&payload, Some(42), true);
        assert!(
            !output.contains("\u{1b}["),
            "no_color=true must strip all ANSI escapes: {output}"
        );
        assert!(output.contains("REFINED"));
        assert!(output.contains("src/auth/middleware.rs"));
        assert!(output.contains("0.923"));
        assert!(output.contains("3 results in 42ms"));
    }

    #[test]
    fn render_search_table_for_cli_includes_source_badges() {
        let payload = sample_search_payload();
        let output = render_search_table_for_cli(&payload, Some(10), true);
        assert!(output.contains("[both]"), "hit in both sources: {output}");
        assert!(output.contains("[lexical]"), "lexical-only hit: {output}");
        assert!(output.contains("[semantic]"), "semantic-only hit: {output}");
    }

    #[test]
    fn render_search_table_for_cli_shows_score_gradient() {
        let payload = SearchPayload::new(
            "test scores",
            SearchOutputPhase::Initial,
            3,
            vec![
                SearchHitPayload {
                    rank: 1,
                    path: "high.rs".to_owned(),
                    score: 0.923,
                    snippet: None,
                    lexical_rank: Some(0),
                    semantic_rank: None,
                    in_both_sources: false,
                },
                SearchHitPayload {
                    rank: 2,
                    path: "medium.rs".to_owned(),
                    score: 0.65,
                    snippet: None,
                    lexical_rank: None,
                    semantic_rank: Some(0),
                    in_both_sources: false,
                },
                SearchHitPayload {
                    rank: 3,
                    path: "low.rs".to_owned(),
                    score: 0.32,
                    snippet: None,
                    lexical_rank: None,
                    semantic_rank: Some(1),
                    in_both_sources: false,
                },
            ],
        );
        let output = render_search_table_with_options(&payload, Some(5), true, 100);
        // High score (>= 0.8) gets green (code 32)
        assert!(
            output.contains("\u{1b}[32m0.923"),
            "high score should be green: {output}"
        );
        // Medium score (>= 0.5 but < 0.8) gets yellow (code 33)
        assert!(
            output.contains("\u{1b}[33m0.650"),
            "medium score should be yellow: {output}"
        );
        // Low score (< 0.5) gets red (code 31)
        assert!(
            output.contains("\u{1b}[31m0.320"),
            "low score should be red: {output}"
        );
    }

    #[test]
    fn render_search_table_for_cli_highlights_query_terms_in_snippet() {
        let payload = SearchPayload::new(
            "bearer token",
            SearchOutputPhase::Initial,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "src/auth.rs".to_owned(),
                score: 0.9,
                snippet: Some("validates bearer token from header".to_owned()),
                lexical_rank: Some(0),
                semantic_rank: Some(0),
                in_both_sources: true,
            }],
        );
        let output = render_search_table_with_options(&payload, Some(5), true, 100);
        // Query terms "bearer" and "token" should be highlighted with bold yellow
        assert!(
            output.contains("\u{1b}[1;33mbearer\u{1b}[0m"),
            "query term 'bearer' should be highlighted: {output}"
        );
        assert!(
            output.contains("\u{1b}[1;33mtoken\u{1b}[0m"),
            "query term 'token' should be highlighted: {output}"
        );
    }

    #[test]
    fn render_search_table_for_cli_splits_path_and_line_number() {
        let payload = SearchPayload::new(
            "test",
            SearchOutputPhase::Initial,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "src/lib.rs:42".to_owned(),
                score: 0.75,
                snippet: None,
                lexical_rank: Some(0),
                semantic_rank: None,
                in_both_sources: false,
            }],
        );
        let output = render_search_table_with_options(&payload, Some(1), true, 100);
        // Line number should be separated with green coloring
        assert!(
            output.contains("\u{1b}[32m42\u{1b}[0m"),
            "line number should be green: {output}"
        );
    }

    #[test]
    fn render_search_table_for_cli_empty_shows_guidance() {
        let payload = SearchPayload::new("obscure query", SearchOutputPhase::Initial, 0, vec![]);
        let output = render_search_table_for_cli(&payload, Some(3), true);
        assert!(output.contains("No results for \"obscure query\"."));
        assert!(output.contains("fsfs status"));
        assert!(output.contains("0 results in 3ms"));
    }

    #[test]
    fn render_search_table_for_cli_truncates_long_snippets() {
        let long_snippet = "a".repeat(200);
        let payload = SearchPayload::new(
            "test",
            SearchOutputPhase::Initial,
            1,
            vec![SearchHitPayload {
                rank: 1,
                path: "src/long.rs".to_owned(),
                score: 0.8,
                snippet: Some(long_snippet),
                lexical_rank: None,
                semantic_rank: Some(0),
                in_both_sources: false,
            }],
        );
        // Narrow width to force truncation (width 60 means snippet_width = max(60-34, 32) = 32)
        let output = render_search_table_with_options(&payload, Some(1), false, 60);
        assert!(
            output.contains('…'),
            "long snippet should be truncated with ellipsis: {output}"
        );
    }

    #[test]
    fn render_search_table_for_cli_phase_labels_correct() {
        for (phase, expected) in [
            (SearchOutputPhase::Initial, "INITIAL"),
            (SearchOutputPhase::Refined, "REFINED"),
            (SearchOutputPhase::RefinementFailed, "REFINEMENT_FAILED"),
        ] {
            let payload = SearchPayload::new("q", phase, 0, vec![]);
            let output = render_search_table_for_cli(&payload, Some(1), true);
            assert!(
                output.contains(expected),
                "phase {expected} not found in: {output}"
            );
        }
    }
}
