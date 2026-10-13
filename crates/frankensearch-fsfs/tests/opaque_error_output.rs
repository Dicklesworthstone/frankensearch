//! Public diagnostic boundaries must not format caller-owned error sources.
//! These are serializer/contract tests, not model or end-to-end search evidence.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_core::SearchError;
use frankensearch_fsfs::output_schema::{
    CompatibilityMode, OutputEnvelope, OutputMeta, decode_envelope_toon,
    encode_envelope_toon, error_code_for, exit_code_for, output_error_from, validate_envelope,
};
use frankensearch_fsfs::stream_protocol::{
    StreamEvent, StreamFailureCategory, StreamFrame, StreamRetryDirective, StreamTerminalStatus,
    decode_stream_frame_ndjson, decode_stream_frame_toon, encode_stream_frame_ndjson,
    encode_stream_frame_toon, terminal_event_from_error, validate_stream_frame,
};

const PRIVATE: &str = "private-provider-response-canary";
const TIMESTAMP: &str = "2026-10-13T18:00:00Z";

#[derive(Clone)]
struct OpaqueSource {
    body: Arc<str>,
    formatted: Arc<AtomicUsize>,
    reject_format: bool,
}

impl fmt::Display for OpaqueSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.formatted.fetch_add(1, Ordering::Relaxed);
        if self.reject_format {
            Err(fmt::Error)
        } else {
            formatter.write_str(&self.body)
        }
    }
}

impl fmt::Debug for OpaqueSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl Error for OpaqueSource {}

fn source(reject_format: bool) -> OpaqueSource {
    OpaqueSource {
        body: format!("{PRIVATE}\n\u{001b}[2J{}", "x".repeat(1 << 20)).into(),
        formatted: Arc::new(AtomicUsize::new(0)),
        reject_format,
    }
}

fn failures(source: &OpaqueSource) -> Vec<(SearchError, &'static str, i32, StreamFailureCategory)> {
    vec![
        (
            SearchError::EmbeddingFailed {
                model: "fixture-model".to_owned(),
                source: Box::new(source.clone()),
            },
            "embedding_failed",
            1,
            StreamFailureCategory::Model,
        ),
        (
            SearchError::ModelLoadFailed {
                path: PathBuf::from("/models/fixture/model.onnx"),
                source: Box::new(source.clone()),
            },
            "model_load_failed",
            78,
            StreamFailureCategory::Model,
        ),
        (
            SearchError::RerankFailed {
                model: "fixture-reranker".to_owned(),
                source: Box::new(source.clone()),
            },
            "rerank_failed",
            1,
            StreamFailureCategory::Model,
        ),
        (
            SearchError::Io(io::Error::new(io::ErrorKind::PermissionDenied, source.clone())),
            "io_error",
            1,
            StreamFailureCategory::Io,
        ),
        (
            SearchError::SubsystemError {
                subsystem: "private-provider-response-canary",
                source: Box::new(source.clone()),
            },
            "subsystem_error",
            1,
            StreamFailureCategory::Internal,
        ),
    ]
}

fn assert_public_text(text: &str) {
    assert!(!text.contains(PRIVATE));
    assert!(!text.contains("[2J"));
    assert!(!text.contains("xxxxxxxxxxxxxxxx"));
    assert!(text.len() < 4_096, "opaque diagnostics must not grow with source size");
}

#[test]
fn json_and_toon_envelopes_do_not_render_opaque_sources() {
    let source = source(false);
    for (error, code, exit, _) in failures(&source) {
        let output = output_error_from(&error);
        assert_eq!(output.code, code);
        assert_eq!(error_code_for(&error), code);
        assert_eq!(output.exit_code, exit);
        assert_eq!(exit_code_for(&error), exit);
        assert!(output.field.is_none());
        let envelope: OutputEnvelope<()> =
            OutputEnvelope::error(output, OutputMeta::new("search", "json"), TIMESTAMP);
        assert!(validate_envelope(&envelope, CompatibilityMode::Strict).valid);
        let json = serde_json::to_string(&envelope).unwrap();
        assert_public_text(&json);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["ok"], false);
        assert!(value.get("data").is_none());
        assert_eq!(value["error"]["code"], code);
        let toon = encode_envelope_toon(&envelope).unwrap();
        assert_public_text(&toon);
        let decoded: OutputEnvelope<()> = decode_envelope_toon(&toon).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), value);
    }
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}

#[test]
fn terminal_streams_keep_failure_and_retry_contracts_without_provider_bodies() {
    let source = source(false);
    for (error, code, exit, category) in failures(&source) {
        let terminal = terminal_event_from_error(&error, 1, 3);
        assert_eq!(terminal.status, StreamTerminalStatus::Failed);
        assert_eq!(terminal.exit_code, exit);
        assert_eq!(terminal.failure_category, Some(category));
        assert_eq!(terminal.error.as_ref().unwrap().code, code);
        let retryable = code != "model_load_failed";
        let expected_retry = if retryable {
            StreamRetryDirective::RetryAfterMs {
                delay_ms: 500,
                next_attempt: 2,
                max_attempts: 3,
            }
        } else {
            StreamRetryDirective::None
        };
        assert_eq!(terminal.retry, expected_retry);
        let frame: StreamFrame<()> = StreamFrame::new(
            "diagnostic-stream", 2, TIMESTAMP, "search", StreamEvent::Terminal(terminal),
        );
        assert!(validate_stream_frame(&frame).valid);
        let ndjson = encode_stream_frame_ndjson(&frame).unwrap();
        assert_public_text(&ndjson);
        assert_eq!(ndjson.lines().count(), 1);
        let decoded: StreamFrame<()> = decode_stream_frame_ndjson(&ndjson).unwrap();
        assert_eq!(decoded, frame);
        let toon = encode_stream_frame_toon(&frame).unwrap();
        assert_public_text(&toon);
        let decoded: StreamFrame<()> = decode_stream_frame_toon(&toon).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(terminal_event_from_error(&error, 0, 0).retry, StreamRetryDirective::None);
        let exhausted = terminal_event_from_error(&error, 3, 3);
        assert_eq!(exhausted.retry, if retryable {
            StreamRetryDirective::RetryExhausted { exhausted_after: 3 }
        } else {
            StreamRetryDirective::None
        });
    }
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}

#[test]
fn a_failing_display_implementation_cannot_break_public_error_construction() {
    // Display returning fmt::Error makes ToString panic. No catch_unwind hides
    // that bug here: construction must not invoke the implementation at all.
    let source = source(true);
    for (error, _, _, _) in failures(&source) {
        let output = output_error_from(&error);
        assert_public_text(&serde_json::to_string(&output).unwrap());
        let terminal = terminal_event_from_error(&error, 0, 1);
        assert_public_text(&serde_json::to_string(&terminal).unwrap());
    }
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}

#[test]
fn public_conversion_preserves_original_local_source_identity() {
    let source = source(false);
    for (error, _, _, _) in failures(&source) {
        let _ = output_error_from(&error);
        let original: &(dyn Error + 'static) = match &error {
            SearchError::EmbeddingFailed { source, .. }
            | SearchError::ModelLoadFailed { source, .. }
            | SearchError::RerankFailed { source, .. }
            | SearchError::SubsystemError { source, .. } => source.as_ref(),
            SearchError::Io(error) => error.get_ref().unwrap(),
            _ => unreachable!("fixture contains only opaque-source failures"),
        };
        let original = original.downcast_ref::<OpaqueSource>().unwrap();
        assert!(Arc::ptr_eq(&original.body, &source.body));
        assert!(Arc::ptr_eq(&original.formatted, &source.formatted));
        assert_eq!(original.body.len(), source.body.len());
    }
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}

struct CyclicSource {
    visited: Arc<AtomicUsize>,
    formatted: Arc<AtomicUsize>,
}

impl fmt::Display for CyclicSource {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.formatted.fetch_add(1, Ordering::Relaxed);
        Err(fmt::Error)
    }
}

impl fmt::Debug for CyclicSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl Error for CyclicSource {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.visited.fetch_add(1, Ordering::Relaxed);
        Some(self)
    }
}

#[test]
fn cyclic_error_chains_have_a_fixed_diagnostic_work_bound() {
    let visited = Arc::new(AtomicUsize::new(0));
    let formatted = Arc::new(AtomicUsize::new(0));
    let error = SearchError::SubsystemError {
        subsystem: "quill",
        source: Box::new(CyclicSource {
            visited: Arc::clone(&visited),
            formatted: Arc::clone(&formatted),
        }),
    };
    let output = output_error_from(&error);
    assert!(output.suggestion.is_none());
    assert_eq!(visited.load(Ordering::Relaxed), 64);
    let terminal = terminal_event_from_error(&error, 0, 1);
    assert_eq!(terminal.status, StreamTerminalStatus::Failed);
    assert_eq!(visited.load(Ordering::Relaxed), 128);
    assert_eq!(formatted.load(Ordering::Relaxed), 0);
}

#[derive(Debug)]
struct Layer(Box<dyn Error + Send + Sync>);

impl fmt::Display for Layer {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}

impl Error for Layer {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref())
    }
}

fn schema_mismatch() -> frankensearch_quill::KeeperError {
    frankensearch_quill::KeeperError::SchemaMismatch {
        path: PathBuf::from(PRIVATE),
        expected: 1,
        found: 2,
    }
}

#[test]
fn schema_advice_uses_types_with_a_limit_not_matching_error_text() {
    for wrappers in [0, 1, 63, 64, 80] {
        let mut inner: Box<dyn Error + Send + Sync> = Box::new(schema_mismatch());
        for _ in 0..wrappers {
            inner = Box::new(Layer(inner));
        }
        let error = SearchError::SubsystemError { subsystem: "quill", source: inner };
        let output = output_error_from(&error);
        assert_eq!(output.suggestion.as_ref().is_some_and(|s| s.contains("older fsfs")), wrappers < 64);
        assert_public_text(&serde_json::to_string(&output).unwrap());
    }
    let wrapped = SearchError::SubsystemError {
        subsystem: "quill",
        source: Box::new(io::Error::other(schema_mismatch())),
    };
    let output = output_error_from(&wrapped);
    assert!(output.suggestion.as_deref().unwrap().contains("fsfs index <directory>"));
    assert_public_text(&serde_json::to_string(&output).unwrap());
    let impostor = SearchError::SubsystemError {
        subsystem: "quill",
        source: Box::new(io::Error::other("SchemaMismatch: older fsfs")),
    };
    assert!(output_error_from(&impostor).suggestion.is_none());
}

#[test]
fn model_labels_and_model_load_paths_cannot_expand_opaque_diagnostics() {
    let source = source(true);
    for model in [
        format!("https://{PRIVATE}/{}", "x".repeat(4_096)),
        format!("{PRIVATE}\nforged"),
        "m".repeat(129),
    ] {
        let error = SearchError::EmbeddingFailed { model: model.clone(), source: Box::new(source.clone()) };
        let output = output_error_from(&error);
        assert!(output.message.contains("MODEL_ID"));
        assert_public_text(&serde_json::to_string(&output).unwrap());
        let error = SearchError::RerankFailed { model, source: Box::new(source.clone()) };
        assert!(output_error_from(&error).message.contains("MODEL_ID"));
    }
    for path in [
        "x".repeat(1_025),
        format!("/{PRIVATE}/\nmodel.onnx"),
        format!("/{PRIVATE}/\u{001b}[2Jmodel.onnx"),
        format!("/{PRIVATE}/\u{202e}model.onnx"),
        format!("/{PRIVATE}/\u{2066}model.onnx"),
    ] {
        let error = SearchError::ModelLoadFailed { path: path.into(), source: Box::new(source.clone()) };
        let output = output_error_from(&error);
        assert!(output.suggestion.as_ref().unwrap().contains("<redacted-model-path>"));
        assert_public_text(&serde_json::to_string(&output).unwrap());
    }
    // Ordinary local model paths remain useful; this boundary does not claim
    // to redact valid caller-supplied paths for export as anonymous telemetry.
    for path in ["/models/fixture/model.onnx".to_owned(), "m".repeat(1_024)] {
        let error = SearchError::ModelLoadFailed { path: path.clone().into(), source: Box::new(source.clone()) };
        let output = output_error_from(&error);
        assert!(output.suggestion.unwrap().contains(&path));
    }
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}

#[cfg(unix)]
#[test]
fn non_utf8_model_paths_are_redacted_without_lossy_allocation() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let source = source(true);
    let error = SearchError::ModelLoadFailed {
        path: PathBuf::from(OsString::from_vec(vec![b'/', 0xff, b'm'])),
        source: Box::new(source.clone()),
    };
    let output = output_error_from(&error);
    assert!(output.suggestion.unwrap().contains("<redacted-model-path>"));
    assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
}
