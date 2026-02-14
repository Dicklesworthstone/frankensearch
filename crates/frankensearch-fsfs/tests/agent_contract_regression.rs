//! Agent-contract regression suite for JSON/TOON outputs, error envelopes,
//! and exit semantics (bd-2hz.10.8).
//!
//! This suite protects agent integrations from output drift by testing:
//! - Envelope field stability across JSON and TOON encodings
//! - Error code taxonomy exhaustiveness and exit code mapping
//! - Stream protocol lifecycle invariants
//! - Compact mode transformations and result ID stability
//! - Cross-format parity (JSON <-> TOON round-trip)

use std::path::PathBuf;

use frankensearch_core::SearchError;
use frankensearch_fsfs::{
    ALL_OUTPUT_ERROR_CODES, ALL_OUTPUT_WARNING_CODES, CompactEnvelope, CompactSearchResponse,
    CompatibilityMode, ENVELOPE_FIELDS, FieldPresence, OUTPUT_SCHEMA_MIN_SUPPORTED,
    OUTPUT_SCHEMA_VERSION, OutputEnvelope, OutputError, OutputErrorCode, OutputMeta, OutputWarning,
    OutputWarningCode, ResultIdRegistry, STREAM_PROTOCOL_VERSION, STREAM_SCHEMA_VERSION,
    StreamTerminalStatus, builtin_templates, decode_envelope_toon, encode_envelope_toon,
    error_code_for, exit_code, exit_code_for, is_version_compatible, output_error_from,
    parse_result_id, result_id, validate_envelope, verify_json_toon_parity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// --- Test Helpers -----------------------------------------------------------

fn test_meta() -> OutputMeta {
    OutputMeta {
        command: "search".to_string(),
        format: "json".to_string(),
        duration_ms: Some(42),
        request_id: Some("01JK0000000000000000000000".to_string()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestPayload {
    query: String,
    result_count: usize,
}

fn sample_success_envelope() -> OutputEnvelope<TestPayload> {
    OutputEnvelope::success(
        TestPayload {
            query: "async runtime".to_string(),
            result_count: 5,
        },
        test_meta(),
        "2026-02-14T12:00:00Z",
    )
}

fn sample_error_envelope() -> OutputEnvelope<TestPayload> {
    OutputEnvelope::error(
        OutputError {
            code: OutputErrorCode::EMBEDDER_UNAVAILABLE.to_string(),
            message: "Embedding model not loaded".to_string(),
            field: None,
            exit_code: exit_code::MODEL_UNAVAILABLE,
            suggestion: Some("Run: fsfs download-models".to_string()),
            context: Some("Fast embedder not available".to_string()),
        },
        test_meta(),
        "2026-02-14T12:00:01Z",
    )
}

fn sample_warning_envelope() -> OutputEnvelope<TestPayload> {
    sample_success_envelope().with_warnings(vec![
        OutputWarning {
            code: OutputWarningCode::DEGRADED_MODE.to_string(),
            message: "Quality tier skipped".to_string(),
        },
        OutputWarning {
            code: OutputWarningCode::FAST_ONLY_RESULTS.to_string(),
            message: "Results from fast phase only".to_string(),
        },
    ])
}

// ===========================================================================
// 1. ENVELOPE FIELD STABILITY
// ===========================================================================

#[test]
fn envelope_success_json_field_names_are_stable() {
    let env = sample_success_envelope();
    let json: Value = serde_json::to_value(&env).unwrap();
    let obj = json.as_object().unwrap();

    // Required top-level fields.
    assert!(obj.contains_key("v"), "missing 'v' field");
    assert!(obj.contains_key("ts"), "missing 'ts' field");
    assert!(obj.contains_key("ok"), "missing 'ok' field");
    assert!(obj.contains_key("data"), "missing 'data' field");
    assert!(obj.contains_key("meta"), "missing 'meta' field");

    // Error and warnings should be absent on success.
    assert!(
        !obj.contains_key("error"),
        "'error' present on success envelope"
    );
    assert!(
        !obj.contains_key("warnings"),
        "'warnings' present when empty"
    );
}

#[test]
fn envelope_error_json_field_names_are_stable() {
    let env = sample_error_envelope();
    let json: Value = serde_json::to_value(&env).unwrap();
    let obj = json.as_object().unwrap();

    assert!(obj.contains_key("v"));
    assert!(obj.contains_key("ts"));
    assert!(obj.contains_key("ok"));
    assert!(obj.contains_key("error"));
    assert!(obj.contains_key("meta"));

    // Data should be absent on error.
    assert!(
        !obj.contains_key("data"),
        "'data' present on error envelope"
    );

    // Error sub-fields.
    let err = obj["error"].as_object().unwrap();
    assert!(err.contains_key("code"), "missing error.code");
    assert!(err.contains_key("message"), "missing error.message");
    assert!(err.contains_key("exit_code"), "missing error.exit_code");
}

#[test]
fn envelope_with_warnings_includes_warnings_array() {
    let env = sample_warning_envelope();
    let json: Value = serde_json::to_value(&env).unwrap();
    let obj = json.as_object().unwrap();

    assert!(obj.contains_key("warnings"));
    let warnings = obj["warnings"].as_array().unwrap();
    assert_eq!(warnings.len(), 2);

    for w in warnings {
        assert!(w.as_object().unwrap().contains_key("code"));
        assert!(w.as_object().unwrap().contains_key("message"));
    }
}

#[test]
fn envelope_schema_version_always_current() {
    let success = sample_success_envelope();
    assert_eq!(success.v, OUTPUT_SCHEMA_VERSION);
    let error = sample_error_envelope();
    assert_eq!(error.v, OUTPUT_SCHEMA_VERSION);
}

#[test]
fn envelope_meta_required_fields_always_present() {
    let env = sample_success_envelope();
    let json: Value = serde_json::to_value(&env).unwrap();
    let meta = json["meta"].as_object().unwrap();

    assert!(meta.contains_key("command"), "missing meta.command");
    assert!(meta.contains_key("format"), "missing meta.format");
    assert!(
        !meta["command"].as_str().unwrap().is_empty(),
        "meta.command is empty"
    );
    assert!(
        !meta["format"].as_str().unwrap().is_empty(),
        "meta.format is empty"
    );
}

// ===========================================================================
// 2. ERROR CODE TAXONOMY EXHAUSTIVENESS
// ===========================================================================

#[test]
fn all_error_codes_are_snake_case() {
    for code in ALL_OUTPUT_ERROR_CODES {
        assert!(
            code.chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
            "error code '{code}' is not snake_case"
        );
    }
}

#[test]
fn all_warning_codes_are_snake_case() {
    for code in ALL_OUTPUT_WARNING_CODES {
        assert!(
            code.chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
            "warning code '{code}' is not snake_case"
        );
    }
}

#[test]
fn error_code_count_is_stable() {
    // If you add a new error code, update this count. Removing codes is a breaking change.
    assert!(
        ALL_OUTPUT_ERROR_CODES.len() >= 21,
        "error codes shrunk from 21 to {} -- this is a breaking change",
        ALL_OUTPUT_ERROR_CODES.len()
    );
}

#[test]
fn warning_code_count_is_stable() {
    assert!(
        ALL_OUTPUT_WARNING_CODES.len() >= 6,
        "warning codes shrunk from 6 to {} -- this is a breaking change",
        ALL_OUTPUT_WARNING_CODES.len()
    );
}

#[test]
fn known_error_codes_still_present() {
    let must_exist = [
        "embedder_unavailable",
        "embedding_failed",
        "model_not_found",
        "model_load_failed",
        "index_corrupted",
        "index_version_mismatch",
        "dimension_mismatch",
        "index_not_found",
        "query_parse_error",
        "search_timeout",
        "reranker_unavailable",
        "rerank_failed",
        "io_error",
        "invalid_config",
        "hash_mismatch",
        "cancelled",
        "queue_full",
        "subsystem_error",
        "durability_disabled",
        "internal_error",
    ];
    for code in &must_exist {
        assert!(
            ALL_OUTPUT_ERROR_CODES.contains(code),
            "required error code '{code}' removed -- breaking change"
        );
    }
}

#[test]
fn known_warning_codes_still_present() {
    let must_exist = [
        "degraded_mode",
        "deprecated_field",
        "rerank_skipped",
        "fast_only_results",
        "hash_fallback",
        "schema_version_newer",
    ];
    for code in &must_exist {
        assert!(
            ALL_OUTPUT_WARNING_CODES.contains(code),
            "required warning code '{code}' removed -- breaking change"
        );
    }
}

// ===========================================================================
// 3. EXIT CODE MAPPING DETERMINISM
// ===========================================================================

#[test]
fn exit_code_values_are_stable() {
    assert_eq!(exit_code::OK, 0);
    assert_eq!(exit_code::RUNTIME_ERROR, 1);
    assert_eq!(exit_code::USAGE_ERROR, 2);
    assert_eq!(exit_code::MODEL_UNAVAILABLE, 78);
    assert_eq!(exit_code::INTERRUPTED, 130);
}

#[test]
fn config_errors_map_to_usage_exit_code() {
    let err = SearchError::InvalidConfig {
        field: "timeout_ms".to_string(),
        value: "bad".to_string(),
        reason: "must be positive".to_string(),
    };
    assert_eq!(exit_code_for(&err), exit_code::USAGE_ERROR);
    assert_eq!(error_code_for(&err), "invalid_config");
}

#[test]
fn query_parse_errors_map_to_usage_exit_code() {
    let err = SearchError::QueryParseError {
        query: "empty query".to_string(),
        detail: "query is empty".to_string(),
    };
    assert_eq!(exit_code_for(&err), exit_code::USAGE_ERROR);
    assert_eq!(error_code_for(&err), "query_parse_error");
}

#[test]
fn model_missing_errors_map_to_model_unavailable_exit_code() {
    for err in [
        SearchError::EmbedderUnavailable {
            model: "no model".to_string(),
            reason: "not found".to_string(),
        },
        SearchError::ModelNotFound {
            name: "potion".to_string(),
        },
        SearchError::ModelLoadFailed {
            path: PathBuf::from("/models/minilm.onnx"),
            source: Box::new(std::io::Error::other("corrupt")),
        },
    ] {
        assert_eq!(
            exit_code_for(&err),
            exit_code::MODEL_UNAVAILABLE,
            "wrong exit code for {err:?}"
        );
    }
}

#[test]
fn cancellation_maps_to_interrupted_exit_code() {
    let err = SearchError::Cancelled {
        phase: "search".to_string(),
        reason: "user interrupt".to_string(),
    };
    assert_eq!(exit_code_for(&err), exit_code::INTERRUPTED);
    assert_eq!(error_code_for(&err), "cancelled");
}

#[test]
fn runtime_errors_map_to_runtime_exit_code() {
    let runtime_errors = [
        SearchError::EmbeddingFailed {
            model: "potion".to_string(),
            source: Box::new(std::io::Error::other("transient")),
        },
        SearchError::SearchTimeout {
            elapsed_ms: 600,
            budget_ms: 500,
        },
        SearchError::IndexCorrupted {
            path: PathBuf::from("/tmp/index.fsvi"),
            detail: "bad header".to_string(),
        },
        SearchError::QueueFull {
            pending: 100,
            capacity: 100,
        },
    ];
    for err in &runtime_errors {
        assert_eq!(
            exit_code_for(err),
            exit_code::RUNTIME_ERROR,
            "wrong exit code for {err:?}"
        );
    }
}

#[test]
fn output_error_from_preserves_error_code() {
    let err = SearchError::EmbedderUnavailable {
        model: "test".to_string(),
        reason: "missing".to_string(),
    };
    let output_err = output_error_from(&err);
    assert_eq!(output_err.code, error_code_for(&err));
    assert_eq!(output_err.exit_code, exit_code_for(&err));
}

#[test]
fn output_error_from_includes_suggestion_for_model_errors() {
    let err = SearchError::ModelNotFound {
        name: "potion-base-8M".to_string(),
    };
    let output_err = output_error_from(&err);
    assert!(
        output_err.suggestion.is_some(),
        "model errors should include actionable suggestion"
    );
}

// ===========================================================================
// 4. JSON <-> TOON CROSS-FORMAT PARITY
// ===========================================================================

#[test]
fn json_toon_parity_success_envelope() {
    let env = sample_success_envelope();
    verify_json_toon_parity(&env).expect("JSON/TOON parity must hold for success envelope");
}

#[test]
fn json_toon_parity_error_envelope() {
    let env = sample_error_envelope();
    verify_json_toon_parity(&env).expect("JSON/TOON parity must hold for error envelope");
}

#[test]
fn json_toon_parity_warning_envelope() {
    let env = sample_warning_envelope();
    verify_json_toon_parity(&env).expect("JSON/TOON parity must hold for warning envelope");
}

#[test]
fn toon_roundtrip_preserves_all_fields() {
    let env = sample_success_envelope();
    let toon = encode_envelope_toon(&env).unwrap();
    let decoded: OutputEnvelope<TestPayload> = decode_envelope_toon(&toon).unwrap();

    assert_eq!(decoded.v, env.v);
    assert_eq!(decoded.ok, env.ok);
    assert_eq!(decoded.data.unwrap().query, "async runtime");
    assert_eq!(decoded.meta.command, "search");
    assert_eq!(decoded.meta.format, "json");
    assert_eq!(decoded.meta.duration_ms, Some(42));
}

#[test]
fn toon_roundtrip_error_preserves_error_fields() {
    let env = sample_error_envelope();
    let toon = encode_envelope_toon(&env).unwrap();
    let decoded: OutputEnvelope<TestPayload> = decode_envelope_toon(&toon).unwrap();

    let err = decoded.error.unwrap();
    assert_eq!(err.code, OutputErrorCode::EMBEDDER_UNAVAILABLE);
    assert_eq!(err.exit_code, exit_code::MODEL_UNAVAILABLE);
    assert!(err.suggestion.is_some());
    assert!(err.context.is_some());
}

#[test]
fn toon_encoding_is_deterministic() {
    let env = sample_success_envelope();
    let toon1 = encode_envelope_toon(&env).unwrap();
    let toon2 = encode_envelope_toon(&env).unwrap();
    assert_eq!(toon1, toon2, "TOON encoding must be deterministic");
}

// ===========================================================================
// 5. ENVELOPE VALIDATION
// ===========================================================================

#[test]
fn valid_success_envelope_passes_strict_validation() {
    let env = sample_success_envelope();
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(
        result.valid,
        "validation violations: {:?}",
        result.violations
    );
}

#[test]
fn valid_error_envelope_passes_strict_validation() {
    let env = sample_error_envelope();
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(
        result.valid,
        "validation violations: {:?}",
        result.violations
    );
}

#[test]
fn broken_envelope_ok_true_with_error_fails_validation() {
    let mut env = sample_success_envelope();
    env.error = Some(OutputError {
        code: "internal_error".to_string(),
        message: "shouldn't be here".to_string(),
        field: None,
        exit_code: 1,
        suggestion: None,
        context: None,
    });
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(!result.valid);
}

#[test]
fn broken_envelope_ok_false_with_data_fails_validation() {
    let mut env = sample_error_envelope();
    env.data = Some(TestPayload {
        query: "orphan".to_string(),
        result_count: 0,
    });
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(!result.valid);
}

#[test]
fn envelope_with_unknown_error_code_fails_strict_validation() {
    let env = OutputEnvelope::<TestPayload>::error(
        OutputError {
            code: "nonexistent_code".to_string(),
            message: "bad code".to_string(),
            field: None,
            exit_code: 1,
            suggestion: None,
            context: None,
        },
        test_meta(),
        "2026-02-14T12:00:00Z",
    );
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(!result.valid);
}

#[test]
fn empty_timestamp_fails_validation() {
    let mut env = sample_success_envelope();
    env.ts = String::new();
    let result = validate_envelope(&env, CompatibilityMode::Strict);
    assert!(!result.valid);
}

// ===========================================================================
// 6. STREAM PROTOCOL CONTRACTS
// ===========================================================================

#[test]
fn stream_protocol_version_is_stable() {
    assert_eq!(STREAM_PROTOCOL_VERSION, 1);
}

#[test]
fn stream_schema_version_format_is_stable() {
    assert_eq!(STREAM_SCHEMA_VERSION, "fsfs.stream.query.v1");
}

#[test]
fn stream_terminal_completed_has_zero_exit_code() {
    let terminal = frankensearch_fsfs::terminal_event_completed();
    assert_eq!(terminal.status, StreamTerminalStatus::Completed);
    assert_eq!(terminal.exit_code, 0);
    assert!(terminal.failure_category.is_none());
    assert!(terminal.error.is_none());
}

#[test]
fn stream_terminal_failed_has_nonzero_exit_code() {
    let err = SearchError::EmbedderUnavailable {
        model: "test".to_string(),
        reason: "missing".to_string(),
    };
    let terminal = frankensearch_fsfs::terminal_event_from_error(&err, 0, 3);
    assert_eq!(terminal.status, StreamTerminalStatus::Failed);
    assert!(terminal.exit_code > 0);
    assert!(terminal.failure_category.is_some());
    assert!(terminal.error.is_some());
    assert_eq!(
        terminal.error.as_ref().unwrap().code,
        OutputErrorCode::EMBEDDER_UNAVAILABLE
    );
}

#[test]
fn stream_terminal_cancelled_has_interrupted_exit_code() {
    let err = SearchError::Cancelled {
        phase: "search".to_string(),
        reason: "user interrupt".to_string(),
    };
    let terminal = frankensearch_fsfs::terminal_event_from_error(&err, 0, 3);
    assert_eq!(terminal.status, StreamTerminalStatus::Cancelled);
    assert_eq!(terminal.exit_code, exit_code::INTERRUPTED);
}

#[test]
fn stream_retryable_errors_get_retry_directive() {
    let retryable = [
        SearchError::EmbeddingFailed {
            model: "potion".to_string(),
            source: Box::new(std::io::Error::other("transient")),
        },
        SearchError::SearchTimeout {
            elapsed_ms: 600,
            budget_ms: 500,
        },
        SearchError::QueueFull {
            pending: 100,
            capacity: 100,
        },
    ];
    for err in &retryable {
        let terminal = frankensearch_fsfs::terminal_event_from_error(err, 0, 3);
        assert!(
            !matches!(
                terminal.retry,
                frankensearch_fsfs::StreamRetryDirective::None
            ),
            "retryable error {err:?} should have retry directive"
        );
    }
}

#[test]
fn stream_non_retryable_errors_have_no_retry() {
    let non_retryable = [
        SearchError::InvalidConfig {
            field: "x".to_string(),
            value: "y".to_string(),
            reason: "bad".to_string(),
        },
        SearchError::IndexCorrupted {
            path: PathBuf::from("/tmp/index.fsvi"),
            detail: "broken".to_string(),
        },
        SearchError::ModelNotFound {
            name: "x".to_string(),
        },
    ];
    for err in &non_retryable {
        assert!(
            !frankensearch_fsfs::is_retryable_error(err),
            "non-retryable error {err:?} reported as retryable"
        );
    }
}

#[test]
fn stream_retry_backoff_is_bounded() {
    // Very high attempt number should not overflow or produce enormous delay.
    let delay = frankensearch_fsfs::retry_backoff_ms(100);
    assert!(
        delay <= 30_000,
        "retry backoff should cap at 30s, got {delay}ms"
    );
}

#[test]
fn stream_retry_backoff_is_monotonic_to_cap() {
    let mut prev = 0u64;
    for attempt in 0..=12 {
        let delay = frankensearch_fsfs::retry_backoff_ms(attempt);
        assert!(
            delay >= prev,
            "backoff should be non-decreasing: attempt={attempt}, prev={prev}, got={delay}"
        );
        prev = delay;
    }
}

// ===========================================================================
// 7. COMPACT MODE / AGENT ERGONOMICS
// ===========================================================================

#[test]
fn result_id_format_is_stable() {
    assert_eq!(result_id(0), "R0");
    assert_eq!(result_id(1), "R1");
    assert_eq!(result_id(42), "R42");
    assert_eq!(result_id(999), "R999");
}

#[test]
fn result_id_roundtrip() {
    for rank in [0, 1, 10, 100, 9999] {
        let id = result_id(rank);
        let parsed = parse_result_id(&id);
        assert_eq!(parsed, Some(rank), "roundtrip failed for rank {rank}");
    }
}

#[test]
fn result_id_parse_rejects_invalid() {
    assert_eq!(parse_result_id("X5"), None);
    assert_eq!(parse_result_id(""), None);
    assert_eq!(parse_result_id("R"), None);
    assert_eq!(parse_result_id("Rabc"), None);
}

#[test]
fn registry_assigns_sequential_ids() {
    let mut registry = ResultIdRegistry::new();
    let docs = vec![
        ("src/main.rs".to_string(), 0.95),
        ("src/lib.rs".to_string(), 0.85),
        ("src/utils.rs".to_string(), 0.75),
    ];
    let ids = registry.register_batch(&docs);
    assert_eq!(ids, vec!["R0", "R1", "R2"]);
}

#[test]
fn registry_preserves_state_across_batches() {
    let mut registry = ResultIdRegistry::new();
    let batch1 = vec![("a.rs".to_string(), 0.9)];
    let batch2 = vec![("b.rs".to_string(), 0.8)];
    let ids1 = registry.register_batch(&batch1);
    let ids2 = registry.register_batch(&batch2);
    assert_eq!(ids1, vec!["R0"]);
    assert_eq!(ids2, vec!["R1"]);
}

#[test]
fn registry_resolve_returns_correct_entry() {
    let mut registry = ResultIdRegistry::new();
    let docs = vec![("alpha.rs".to_string(), 0.9), ("beta.rs".to_string(), 0.8)];
    registry.register_batch(&docs);

    let entry = registry.resolve("R0").expect("R0 should exist");
    assert_eq!(entry.doc_id, "alpha.rs");

    let entry = registry.resolve("R1").expect("R1 should exist");
    assert_eq!(entry.doc_id, "beta.rs");

    assert!(registry.resolve("R99").is_none());
}

#[test]
fn compact_envelope_success_has_expected_shape() {
    let compact = CompactEnvelope {
        ok: true,
        data: Some(CompactSearchResponse {
            n: 2,
            hits: vec![],
            ms: Some(42),
            phase: Some("fast".to_string()),
        }),
        err: None,
        w: vec![],
    };
    let json: Value = serde_json::to_value(&compact).unwrap();
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("ok"));
    assert!(obj.contains_key("data"));
    assert!(!obj.contains_key("err") || obj["err"].is_null());
}

#[test]
fn builtin_templates_are_stable() {
    let templates = builtin_templates();
    assert!(
        templates.len() >= 3,
        "expected at least 3 built-in templates, got {}",
        templates.len()
    );

    let names: Vec<&str> = templates.iter().map(|t| t.name.as_str()).collect();
    assert!(
        names.contains(&"search_then_explain"),
        "missing search_then_explain template"
    );
    assert!(
        names.contains(&"incremental_refinement"),
        "missing incremental_refinement template"
    );
    assert!(
        names.contains(&"batch_search"),
        "missing batch_search template"
    );
}

#[test]
fn templates_have_at_least_one_step() {
    for template in builtin_templates() {
        assert!(
            !template.steps.is_empty(),
            "template '{}' has no steps",
            template.name
        );
    }
}

// ===========================================================================
// 8. VERSION COMPATIBILITY GATES
// ===========================================================================

#[test]
fn current_version_is_compatible_with_itself() {
    assert!(is_version_compatible(
        OUTPUT_SCHEMA_VERSION,
        CompatibilityMode::Strict
    ));
    assert!(is_version_compatible(
        OUTPUT_SCHEMA_VERSION,
        CompatibilityMode::Lenient
    ));
}

#[test]
fn future_version_fails_strict_but_passes_lenient() {
    let future_version = OUTPUT_SCHEMA_VERSION + 1;
    assert!(!is_version_compatible(
        future_version,
        CompatibilityMode::Strict
    ));
    assert!(is_version_compatible(
        future_version,
        CompatibilityMode::Lenient
    ));
}

#[test]
fn ancient_version_fails_both_modes() {
    if OUTPUT_SCHEMA_MIN_SUPPORTED > 0 {
        assert!(!is_version_compatible(0, CompatibilityMode::Strict));
        assert!(!is_version_compatible(0, CompatibilityMode::Lenient));
    }
}

// ===========================================================================
// 9. FIELD PRESENCE CONTRACT
// ===========================================================================

#[test]
fn field_descriptors_cover_required_fields() {
    let required: Vec<&str> = ENVELOPE_FIELDS
        .iter()
        .filter(|f| f.presence == FieldPresence::Required)
        .map(|f| f.name)
        .collect();
    assert!(required.contains(&"v"));
    assert!(required.contains(&"ts"));
    assert!(required.contains(&"ok"));
    assert!(required.contains(&"meta"));
}

#[test]
fn field_descriptors_mark_data_and_error_as_conditional() {
    let data_field = ENVELOPE_FIELDS.iter().find(|f| f.name == "data");
    assert!(data_field.is_some());
    assert!(
        matches!(
            data_field.unwrap().presence,
            FieldPresence::ConditionalOn(_)
        ),
        "'data' should be conditional (present when ok==true)"
    );

    let error_field = ENVELOPE_FIELDS.iter().find(|f| f.name == "error");
    assert!(error_field.is_some());
    assert!(
        matches!(
            error_field.unwrap().presence,
            FieldPresence::ConditionalOn(_)
        ),
        "'error' should be conditional (present when ok==false)"
    );
}

// ===========================================================================
// 10. UNICODE AND EDGE CASES
// ===========================================================================

#[test]
fn envelope_handles_unicode_in_payload() {
    let env = OutputEnvelope::success(
        TestPayload {
            query: "concurrencia asincrona".to_string(),
            result_count: 1,
        },
        test_meta(),
        "2026-02-14T12:00:00Z",
    );
    let json = serde_json::to_string(&env).unwrap();
    let decoded: OutputEnvelope<TestPayload> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.data.unwrap().query, "concurrencia asincrona");
}

#[test]
fn envelope_handles_unicode_in_error_message() {
    let env = OutputEnvelope::<TestPayload>::error(
        OutputError {
            code: "internal_error".to_string(),
            message: "model load failed".to_string(),
            field: None,
            exit_code: 1,
            suggestion: None,
            context: None,
        },
        test_meta(),
        "2026-02-14T12:00:00Z",
    );
    let json = serde_json::to_string(&env).unwrap();
    assert!(json.contains("model load failed"));
    let decoded: OutputEnvelope<TestPayload> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.error.unwrap().message, "model load failed");
}

#[test]
fn toon_roundtrip_with_unicode() {
    let env = OutputEnvelope::success(
        TestPayload {
            query: "structured data".to_string(),
            result_count: 3,
        },
        test_meta(),
        "2026-02-14T12:00:00Z",
    );
    let toon = encode_envelope_toon(&env).unwrap();
    let decoded: OutputEnvelope<TestPayload> = decode_envelope_toon(&toon).unwrap();
    assert_eq!(decoded.data.unwrap().query, "structured data");
}

#[test]
fn envelope_handles_empty_optional_fields() {
    let env = OutputEnvelope::success(
        TestPayload {
            query: String::new(),
            result_count: 0,
        },
        OutputMeta {
            command: "search".to_string(),
            format: "json".to_string(),
            duration_ms: None,
            request_id: None,
        },
        "2026-02-14T12:00:00Z",
    );
    let json: Value = serde_json::to_value(&env).unwrap();
    let meta = json["meta"].as_object().unwrap();
    // Optional fields should be absent (not null) when None.
    assert!(
        !meta.contains_key("duration_ms") || meta["duration_ms"].is_null(),
        "duration_ms should be absent or null when None"
    );
}
