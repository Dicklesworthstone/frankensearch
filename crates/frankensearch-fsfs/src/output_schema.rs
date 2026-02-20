//! Versioned JSON output schema and forward/backward compatibility contract.
//!
//! This module defines the machine-consumable JSON envelope emitted by fsfs in
//! `--format json`, `--format jsonl`, and `--format toon` modes. Every response
//! wraps its payload in an [`OutputEnvelope`] that provides:
//!
//! - **Schema version**: integer `v` field for version negotiation.
//! - **Success/error bifurcation**: `ok` boolean, with `data` or `error` present.
//! - **Stable error codes**: [`OutputErrorCode`] constants that map 1:1 from
//!   [`SearchError`] variants.
//! - **Field optionality**: explicit [`FieldPresence`] descriptors for every
//!   envelope field.
//! - **Compatibility mode**: [`CompatibilityMode`] controls whether unknown
//!   fields are rejected (strict) or ignored (lenient).
//!
//! # Compatibility Contract
//!
//! - **Same major version**: Fields may be added but never removed or renamed.
//!   Optional fields may become required, but required fields never become
//!   optional. Enum variants may be added but never removed.
//! - **Major version bump**: Breaking changes (field removal, rename, type
//!   change) require incrementing [`OUTPUT_SCHEMA_VERSION`]. Consumers in
//!   [`CompatibilityMode::Lenient`] should tolerate unknown fields from newer
//!   minor schema revisions within the same major version.
//! - **Deprecation window**: Deprecated fields carry a `since` version and
//!   optional `removed_in` version. They remain present (possibly null) until
//!   the `removed_in` version.

use std::fmt;
use std::io;

use frankensearch_core::{SearchError, SearchResult};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::evidence::{ValidationResult, ValidationViolation};

const SUBSYSTEM: &str = "fsfs_output_schema";
const TOON_DEFAULT_DELIMITER: char = ',';

// ─── Schema Version ─────────────────────────────────────────────────────────

/// Current output schema version. Incremented on breaking changes.
pub const OUTPUT_SCHEMA_VERSION: u32 = 1;

/// Minimum schema version that consumers in lenient mode should accept.
pub const OUTPUT_SCHEMA_MIN_SUPPORTED: u32 = 1;
const KNOWN_OUTPUT_FORMATS: &[&str] = &["table", "json", "jsonl", "csv", "toon"];

// ─── Compatibility Mode ─────────────────────────────────────────────────────

/// Controls how a consumer should interpret the output schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityMode {
    /// Consumer rejects unknown fields and requires exact version match.
    /// Use for CI pipelines and regression suites.
    Strict,
    /// Consumer ignores unknown fields and accepts any version >= `min_supported`.
    /// Use for interactive tooling and forward-compatible consumers.
    #[default]
    Lenient,
}

impl fmt::Display for CompatibilityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strict => write!(f, "strict"),
            Self::Lenient => write!(f, "lenient"),
        }
    }
}

// ─── Output Envelope ────────────────────────────────────────────────────────

/// Top-level JSON envelope wrapping every fsfs CLI response.
///
/// # Invariants
///
/// - `ok == true` implies `data` is `Some` and `error` is `None`.
/// - `ok == false` implies `error` is `Some` and `data` is `None`.
/// - `v` is always [`OUTPUT_SCHEMA_VERSION`].
/// - `ts` is always RFC 3339 UTC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputEnvelope<T> {
    /// Schema version (always [`OUTPUT_SCHEMA_VERSION`]).
    pub v: u32,
    /// RFC 3339 UTC timestamp of response generation.
    pub ts: String,
    /// Whether the operation completed successfully.
    pub ok: bool,
    /// Success payload. Present when `ok == true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    /// Error payload. Present when `ok == false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<OutputError>,
    /// Non-fatal warnings (e.g., degraded mode, deprecated fields).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<OutputWarning>,
    /// Request metadata (command, format, timing).
    pub meta: OutputMeta,
}

impl<T> OutputEnvelope<T> {
    /// Create a success envelope.
    #[must_use]
    pub fn success(data: T, meta: OutputMeta, ts: impl Into<String>) -> Self {
        Self {
            v: OUTPUT_SCHEMA_VERSION,
            ts: ts.into(),
            ok: true,
            data: Some(data),
            error: None,
            warnings: Vec::new(),
            meta,
        }
    }

    /// Create an error envelope.
    #[must_use]
    pub fn error(error: OutputError, meta: OutputMeta, ts: impl Into<String>) -> Self {
        Self {
            v: OUTPUT_SCHEMA_VERSION,
            ts: ts.into(),
            ok: false,
            data: None,
            error: Some(error),
            warnings: Vec::new(),
            meta,
        }
    }

    /// Attach warnings to the envelope.
    #[must_use]
    pub fn with_warnings(mut self, warnings: Vec<OutputWarning>) -> Self {
        self.warnings = warnings;
        self
    }
}

/// Encode an output envelope into TOON text via `toon-rust`.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` when JSON projection or TOON
/// encoding fails.
pub fn encode_envelope_toon<T>(envelope: &OutputEnvelope<T>) -> SearchResult<String>
where
    T: Serialize,
{
    let mut value =
        serde_json::to_value(envelope).map_err(|source| SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "failed to project output envelope to JSON value: {source}"
            ))),
        })?;
    prepare_toon_value_for_lossless_strings(&mut value)?;

    toon_rust::encode(&value, None).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to encode output envelope as TOON: {source}"
        ))),
    })
}

/// Decode TOON text into an output envelope via `toon-rust`.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` when TOON decoding or JSON-to-envelope
/// deserialization fails.
pub fn decode_envelope_toon<T>(input: &str) -> SearchResult<OutputEnvelope<T>>
where
    T: DeserializeOwned,
{
    let value = toon_rust::decode(input, None).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to decode TOON output envelope: {source}"
        ))),
    })?;

    serde_json::from_value(value).map_err(|source| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(format!(
            "failed to deserialize TOON payload into output envelope: {source}"
        ))),
    })
}

fn prepare_toon_value_for_lossless_strings(value: &mut serde_json::Value) -> SearchResult<()> {
    match value {
        serde_json::Value::String(token) => {
            if should_wrap_toon_string_token(token) {
                let wrapped =
                    serde_json::to_string(token).map_err(|source| SearchError::SubsystemError {
                        subsystem: SUBSYSTEM,
                        source: Box::new(io::Error::other(format!(
                            "failed to prepare TOON string token for lossless encoding: {source}"
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

// ─── Output Error ───────────────────────────────────────────────────────────

/// Structured error object with a machine-stable code.
///
/// Error codes are drawn from [`OutputErrorCode`] constants and map 1:1
/// from [`SearchError`] variants.
///
/// The optional `suggestion` and `context` fields follow the three-part
/// error message pattern: (1) what happened (`message`), (2) why it
/// matters (`context`), (3) what to do about it (`suggestion`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputError {
    /// Machine-stable error code (`snake_case`, never changes within a major version).
    pub code: String,
    /// Human-readable error message. May change across versions.
    pub message: String,
    /// The field that caused the error (for validation/config errors).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Suggested CLI exit code.
    pub exit_code: i32,
    /// Actionable suggestion for how to fix the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    /// Additional context explaining why the error matters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl OutputError {
    /// Create a new output error.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>, exit_code: i32) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            field: None,
            exit_code,
            suggestion: None,
            context: None,
        }
    }

    /// Attach a field path to this error.
    #[must_use]
    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }

    /// Attach an actionable fix suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Attach explanatory context about why this error matters.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

// ─── Output Warning ────────────────────────────────────────────────────────

/// Non-fatal warning attached to a response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputWarning {
    /// Machine-stable warning code (`snake_case`).
    pub code: String,
    /// Human-readable warning message.
    pub message: String,
}

impl OutputWarning {
    /// Create a new output warning.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

// ─── Output Meta ────────────────────────────────────────────────────────────

/// Request metadata embedded in every output envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputMeta {
    /// Command that produced this response (e.g., "search", "status").
    pub command: String,
    /// Output format used (e.g., "json", "jsonl", "toon").
    pub format: String,
    /// Wall-clock duration of the operation in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Unique request ID for correlation (ULID format when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl OutputMeta {
    /// Create metadata for a command.
    #[must_use]
    pub fn new(command: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            format: format.into(),
            duration_ms: None,
            request_id: None,
        }
    }

    /// Attach a duration.
    #[must_use]
    pub const fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Attach a request ID.
    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }
}

// ─── Search Payload ────────────────────────────────────────────────────────

/// Progressive search phase for CLI output payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchOutputPhase {
    Initial,
    Refined,
    RefinementFailed,
}

impl fmt::Display for SearchOutputPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial => write!(f, "initial"),
            Self::Refined => write!(f, "refined"),
            Self::RefinementFailed => write!(f, "refinement_failed"),
        }
    }
}

/// One ranked search hit inside a [`SearchPayload`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHitPayload {
    pub rank: usize,
    pub path: String,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_rank: Option<usize>,
    pub in_both_sources: bool,
}

/// Search payload embedded in `OutputEnvelope::data` for `fsfs search`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchPayload {
    pub query: String,
    pub phase: SearchOutputPhase,
    pub total_candidates: usize,
    pub returned_hits: usize,
    pub hits: Vec<SearchHitPayload>,
}

impl SearchPayload {
    #[must_use]
    pub fn new(
        query: impl Into<String>,
        phase: SearchOutputPhase,
        total_candidates: usize,
        hits: Vec<SearchHitPayload>,
    ) -> Self {
        let returned_hits = hits.len();
        Self {
            query: query.into(),
            phase,
            total_candidates,
            returned_hits,
            hits,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}

// ─── Stable Error Codes ─────────────────────────────────────────────────────

/// Machine-stable error code constants for JSON output.
///
/// These map 1:1 from [`SearchError`] variants
/// and are guaranteed to remain stable within a major schema version. Consumers
/// should match on these codes, never on the `message` field.
pub struct OutputErrorCode;

impl OutputErrorCode {
    /// Embedding model is not available.
    pub const EMBEDDER_UNAVAILABLE: &str = "embedder_unavailable";
    /// Embedding inference failed (transient).
    pub const EMBEDDING_FAILED: &str = "embedding_failed";
    /// Model files not found.
    pub const MODEL_NOT_FOUND: &str = "model_not_found";
    /// Model files exist but failed to load.
    pub const MODEL_LOAD_FAILED: &str = "model_load_failed";
    /// Vector index file is corrupted.
    pub const INDEX_CORRUPTED: &str = "index_corrupted";
    /// Index file version mismatch.
    pub const INDEX_VERSION_MISMATCH: &str = "index_version_mismatch";
    /// Query vector dimension does not match index.
    pub const DIMENSION_MISMATCH: &str = "dimension_mismatch";
    /// Vector index file not found.
    pub const INDEX_NOT_FOUND: &str = "index_not_found";
    /// Query string could not be parsed.
    pub const QUERY_PARSE_ERROR: &str = "query_parse_error";
    /// Search phase exceeded its time budget.
    pub const SEARCH_TIMEOUT: &str = "search_timeout";
    /// Federated search got insufficient shard responses.
    pub const FEDERATED_INSUFFICIENT: &str = "federated_insufficient_responses";
    /// Reranking model is not available.
    pub const RERANKER_UNAVAILABLE: &str = "reranker_unavailable";
    /// Reranking inference failed (transient).
    pub const RERANK_FAILED: &str = "rerank_failed";
    /// I/O error during file operations.
    pub const IO_ERROR: &str = "io_error";
    /// Configuration value is invalid.
    pub const INVALID_CONFIG: &str = "invalid_config";
    /// File hash mismatch (corrupted download).
    pub const HASH_MISMATCH: &str = "hash_mismatch";
    /// Operation cancelled via structured concurrency.
    pub const CANCELLED: &str = "cancelled";
    /// Embedding queue is full (backpressure).
    pub const QUEUE_FULL: &str = "queue_full";
    /// Optional subsystem error (storage, durability, fts5).
    pub const SUBSYSTEM_ERROR: &str = "subsystem_error";
    /// Durability feature is not compiled in.
    pub const DURABILITY_DISABLED: &str = "durability_disabled";
    /// Unknown or unclassified error.
    pub const INTERNAL: &str = "internal_error";
}

/// All stable error codes for enumeration and validation.
pub const ALL_OUTPUT_ERROR_CODES: &[&str] = &[
    OutputErrorCode::EMBEDDER_UNAVAILABLE,
    OutputErrorCode::EMBEDDING_FAILED,
    OutputErrorCode::MODEL_NOT_FOUND,
    OutputErrorCode::MODEL_LOAD_FAILED,
    OutputErrorCode::INDEX_CORRUPTED,
    OutputErrorCode::INDEX_VERSION_MISMATCH,
    OutputErrorCode::DIMENSION_MISMATCH,
    OutputErrorCode::INDEX_NOT_FOUND,
    OutputErrorCode::QUERY_PARSE_ERROR,
    OutputErrorCode::SEARCH_TIMEOUT,
    OutputErrorCode::FEDERATED_INSUFFICIENT,
    OutputErrorCode::RERANKER_UNAVAILABLE,
    OutputErrorCode::RERANK_FAILED,
    OutputErrorCode::IO_ERROR,
    OutputErrorCode::INVALID_CONFIG,
    OutputErrorCode::HASH_MISMATCH,
    OutputErrorCode::CANCELLED,
    OutputErrorCode::QUEUE_FULL,
    OutputErrorCode::SUBSYSTEM_ERROR,
    OutputErrorCode::DURABILITY_DISABLED,
    OutputErrorCode::INTERNAL,
];

// ─── Warning Codes ──────────────────────────────────────────────────────────

/// Machine-stable warning code constants.
pub struct OutputWarningCode;

impl OutputWarningCode {
    /// Search operated in degraded mode (e.g., quality tier skipped).
    pub const DEGRADED_MODE: &str = "degraded_mode";
    /// A deprecated field was accessed or emitted.
    pub const DEPRECATED_FIELD: &str = "deprecated_field";
    /// Reranking was skipped (circuit open or fast-only mode).
    pub const RERANK_SKIPPED: &str = "rerank_skipped";
    /// Results are from the fast (initial) phase only.
    pub const FAST_ONLY_RESULTS: &str = "fast_only_results";
    /// Embedding fell back to hash embedder.
    pub const HASH_FALLBACK: &str = "hash_fallback";
    /// Schema version is newer than consumer expects.
    pub const SCHEMA_NEWER: &str = "schema_version_newer";
}

/// All stable warning codes for enumeration.
pub const ALL_OUTPUT_WARNING_CODES: &[&str] = &[
    OutputWarningCode::DEGRADED_MODE,
    OutputWarningCode::DEPRECATED_FIELD,
    OutputWarningCode::RERANK_SKIPPED,
    OutputWarningCode::FAST_ONLY_RESULTS,
    OutputWarningCode::HASH_FALLBACK,
    OutputWarningCode::SCHEMA_NEWER,
];

// ─── Field Optionality ──────────────────────────────────────────────────────

/// Describes the presence contract for a field within the output schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldPresence {
    /// Field is always present and non-null.
    Required,
    /// Field may be absent or null.
    Optional,
    /// Field is present only when a condition holds (e.g., "ok == true").
    ConditionalOn(&'static str),
    /// Field is deprecated: still emitted but scheduled for removal.
    Deprecated {
        /// Schema version when deprecation was announced.
        since: u32,
        /// Schema version when the field will be removed (if known).
        removed_in: Option<u32>,
    },
}

impl fmt::Display for FieldPresence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Required => write!(f, "required"),
            Self::Optional => write!(f, "optional"),
            Self::ConditionalOn(cond) => write!(f, "conditional({cond})"),
            Self::Deprecated { since, removed_in } => {
                write!(f, "deprecated(since=v{since}")?;
                if let Some(v) = removed_in {
                    write!(f, ", removed_in=v{v}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Describes a single field in the output schema for introspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDescriptor {
    /// Dot-separated field path (e.g., "`meta.duration_ms`").
    pub name: &'static str,
    /// Presence contract.
    pub presence: FieldPresence,
    /// Human-readable description of the field.
    pub description: &'static str,
}

/// Field descriptors for the top-level output envelope.
pub const ENVELOPE_FIELDS: &[FieldDescriptor] = &[
    FieldDescriptor {
        name: "v",
        presence: FieldPresence::Required,
        description: "Schema version integer",
    },
    FieldDescriptor {
        name: "ts",
        presence: FieldPresence::Required,
        description: "RFC 3339 UTC timestamp",
    },
    FieldDescriptor {
        name: "ok",
        presence: FieldPresence::Required,
        description: "Whether the operation succeeded",
    },
    FieldDescriptor {
        name: "data",
        presence: FieldPresence::ConditionalOn("ok == true"),
        description: "Success payload",
    },
    FieldDescriptor {
        name: "error",
        presence: FieldPresence::ConditionalOn("ok == false"),
        description: "Error payload",
    },
    FieldDescriptor {
        name: "warnings",
        presence: FieldPresence::Optional,
        description: "Non-fatal warnings (omitted when empty)",
    },
    FieldDescriptor {
        name: "meta",
        presence: FieldPresence::Required,
        description: "Request metadata",
    },
    FieldDescriptor {
        name: "meta.command",
        presence: FieldPresence::Required,
        description: "Command name that produced this response",
    },
    FieldDescriptor {
        name: "meta.format",
        presence: FieldPresence::Required,
        description: "Output format used",
    },
    FieldDescriptor {
        name: "meta.duration_ms",
        presence: FieldPresence::Optional,
        description: "Wall-clock operation duration in milliseconds",
    },
    FieldDescriptor {
        name: "meta.request_id",
        presence: FieldPresence::Optional,
        description: "Unique request correlation ID (ULID)",
    },
    FieldDescriptor {
        name: "error.code",
        presence: FieldPresence::ConditionalOn("ok == false"),
        description: "Machine-stable error code",
    },
    FieldDescriptor {
        name: "error.message",
        presence: FieldPresence::ConditionalOn("ok == false"),
        description: "Human-readable error message",
    },
    FieldDescriptor {
        name: "error.field",
        presence: FieldPresence::Optional,
        description: "Field that caused a validation/config error",
    },
    FieldDescriptor {
        name: "error.exit_code",
        presence: FieldPresence::ConditionalOn("ok == false"),
        description: "Suggested CLI exit code",
    },
];

// ─── SearchError → OutputError ──────────────────────────────────────────────

/// Map a [`SearchError`] to its stable output
/// error code string.
#[must_use]
pub const fn error_code_for(err: &frankensearch_core::SearchError) -> &'static str {
    use frankensearch_core::SearchError;
    match err {
        SearchError::EmbedderUnavailable { .. } => OutputErrorCode::EMBEDDER_UNAVAILABLE,
        SearchError::EmbeddingFailed { .. } => OutputErrorCode::EMBEDDING_FAILED,
        SearchError::ModelNotFound { .. } => OutputErrorCode::MODEL_NOT_FOUND,
        SearchError::ModelLoadFailed { .. } => OutputErrorCode::MODEL_LOAD_FAILED,
        SearchError::IndexCorrupted { .. } => OutputErrorCode::INDEX_CORRUPTED,
        SearchError::IndexVersionMismatch { .. } => OutputErrorCode::INDEX_VERSION_MISMATCH,
        SearchError::DimensionMismatch { .. } => OutputErrorCode::DIMENSION_MISMATCH,
        SearchError::IndexNotFound { .. } => OutputErrorCode::INDEX_NOT_FOUND,
        SearchError::QueryParseError { .. } => OutputErrorCode::QUERY_PARSE_ERROR,
        SearchError::SearchTimeout { .. } => OutputErrorCode::SEARCH_TIMEOUT,
        SearchError::FederatedInsufficientResponses { .. } => {
            OutputErrorCode::FEDERATED_INSUFFICIENT
        }
        SearchError::RerankerUnavailable { .. } => OutputErrorCode::RERANKER_UNAVAILABLE,
        SearchError::RerankFailed { .. } => OutputErrorCode::RERANK_FAILED,
        SearchError::Io(_) => OutputErrorCode::IO_ERROR,
        SearchError::InvalidConfig { .. } => OutputErrorCode::INVALID_CONFIG,
        SearchError::HashMismatch { .. } => OutputErrorCode::HASH_MISMATCH,
        SearchError::Cancelled { .. } => OutputErrorCode::CANCELLED,
        SearchError::QueueFull { .. } => OutputErrorCode::QUEUE_FULL,
        SearchError::SubsystemError { .. } => OutputErrorCode::SUBSYSTEM_ERROR,
        SearchError::DurabilityDisabled => OutputErrorCode::DURABILITY_DISABLED,
    }
}

/// Map a [`SearchError`] to its suggested
/// exit code.
#[must_use]
pub const fn exit_code_for(err: &frankensearch_core::SearchError) -> i32 {
    use crate::adapters::cli::exit_code;
    use frankensearch_core::SearchError;
    match err {
        SearchError::InvalidConfig { .. } | SearchError::QueryParseError { .. } => {
            exit_code::USAGE_ERROR
        }
        SearchError::EmbedderUnavailable { .. }
        | SearchError::ModelNotFound { .. }
        | SearchError::ModelLoadFailed { .. } => exit_code::MODEL_UNAVAILABLE,
        SearchError::Cancelled { .. } => exit_code::INTERRUPTED,
        _ => exit_code::RUNTIME_ERROR,
    }
}

/// Convert a [`SearchError`] into a structured
/// [`OutputError`] with actionable suggestion and context.
#[must_use]
pub fn output_error_from(err: &frankensearch_core::SearchError) -> OutputError {
    use frankensearch_core::SearchError;

    let code = error_code_for(err);
    let exit = exit_code_for(err);
    let message = err.to_string();

    let field = match err {
        SearchError::InvalidConfig { field, .. } => Some(field.clone()),
        _ => None,
    };

    OutputError {
        code: code.to_owned(),
        message,
        field,
        exit_code: exit,
        suggestion: suggestion_for_error(err),
        context: context_for_error(err),
    }
}

/// Return an actionable fix suggestion for a [`SearchError`].
///
/// Each suggestion tells the user exactly what command to run or
/// environment variable to set in order to resolve the error.
#[must_use]
fn suggestion_for_error(err: &frankensearch_core::SearchError) -> Option<String> {
    use frankensearch_core::SearchError;

    match err {
        SearchError::EmbedderUnavailable { model, .. } => Some(format!(
            "Default semantic models are bundled and should auto-install on first use.\n\
             If this failed, set FRANKENSEARCH_MODEL_DIR to a writable directory.\n\
             Optional: fsfs download-models --model {model} (for alternate model sets)."
        )),
        SearchError::ModelNotFound { name } => Some(format!(
            "Default semantic models are bundled but not available in the active cache path.\n\
             Ensure FRANKENSEARCH_MODEL_DIR is writable and run: fsfs status\n\
             Optional override: fsfs download-models --model {name}\n\
             Check cache: fsfs doctor --check-models"
        )),
        SearchError::ModelLoadFailed { path, .. } => Some(format!(
            "The model file at {} may be corrupted.\n\
             Try: fsfs download-models --force --model <name>\n\
             Or run: fsfs doctor --fix",
            path.display()
        )),
        SearchError::IndexNotFound { path } => Some(format!(
            "Run: fsfs index <directory>\n\
             Expected index at: {}",
            path.display()
        )),
        SearchError::IndexCorrupted { path, .. } => Some(format!(
            "Run: fsfs index --force <directory>\n\
             This will rebuild the index at: {}",
            path.display()
        )),
        SearchError::IndexVersionMismatch { expected, found } => Some(format!(
            "The index was built with format v{found} but this version of fsfs expects v{expected}.\n\
             Run: fsfs index --force <directory> to rebuild."
        )),
        SearchError::DimensionMismatch { expected, found } => Some(format!(
            "The index was built with {expected}-dim embeddings but the current embedder produces {found}-dim vectors.\n\
             Run: fsfs index --force <directory> to rebuild with the current embedder."
        )),
        SearchError::InvalidConfig { field, reason, .. } => Some(format!(
            "Check the '{field}' setting in your configuration.\n\
             {reason}\n\
             Config files: fsfs.toml (project) or ~/.config/fsfs/config.toml (user)"
        )),
        SearchError::Io(_) => Some(
            "Check file permissions and available disk space.\n\
             Bundled default semantic models require ~600MB in the model cache; search indices vary by corpus size."
                .to_owned(),
        ),
        SearchError::SearchTimeout {
            budget_ms, ..
        } => Some(format!(
            "Increase the timeout: fsfs search --timeout {}\n\
             Or reduce result count: fsfs search --limit 5",
            budget_ms.saturating_mul(2)
        )),
        SearchError::HashMismatch { path, .. } => Some(format!(
            "The file at {} may be corrupted or tampered with.\n\
             Re-download: fsfs download-models --force",
            path.display()
        )),
        SearchError::QueueFull { capacity, .. } => Some(format!(
            "The embedding queue is at capacity ({capacity}).\n\
             Wait for current jobs to complete or increase queue capacity in config."
        )),
        SearchError::DurabilityDisabled => Some(
            "Enable the 'durability' feature: cargo install frankensearch-fsfs --features durability"
                .to_owned(),
        ),
        SearchError::RerankerUnavailable { .. } => Some(
            "Enable the 'rerank' feature: cargo install frankensearch-fsfs --features rerank\n\
             Or use --no-rerank to skip reranking."
                .to_owned(),
        ),
        _ => None,
    }
}

/// Return explanatory context about why a [`SearchError`] matters.
#[must_use]
fn context_for_error(err: &frankensearch_core::SearchError) -> Option<String> {
    use frankensearch_core::SearchError;

    match err {
        SearchError::EmbedderUnavailable { .. } => Some(
            "Without an embedding model, semantic search is unavailable. \
             Lexical (keyword) search via BM25 may still work if a Tantivy index exists."
                .to_owned(),
        ),
        SearchError::ModelNotFound { .. } => Some(
            "Default semantic models are bundled in fsfs and materialized into the local cache on first use. \
             If models are still missing, check model-dir permissions and cache path selection."
                .to_owned(),
        ),
        SearchError::ModelLoadFailed { .. } => Some(
            "A corrupted model file can produce wrong results or crash the ONNX runtime. \
             Re-downloading ensures integrity."
                .to_owned(),
        ),
        SearchError::IndexNotFound { .. } => Some(
            "No search index exists yet. You need to index a directory of files \
             before you can search."
                .to_owned(),
        ),
        SearchError::IndexCorrupted { .. } => Some(
            "A corrupted index will produce incorrect or missing results. \
             Rebuilding with --force creates a fresh index from source files."
                .to_owned(),
        ),
        SearchError::DimensionMismatch { .. } => Some(
            "The index was built with a different embedder than the one currently configured. \
             Rebuild the index to match the active embedder."
                .to_owned(),
        ),
        SearchError::HashMismatch { .. } => Some(
            "File integrity verification failed. This can happen after network errors \
             during download or disk corruption."
                .to_owned(),
        ),
        _ => None,
    }
}

// ─── Schema Validation ──────────────────────────────────────────────────────

/// Validate an output envelope against the schema contract.
///
/// Checks:
/// 1. `v` matches [`OUTPUT_SCHEMA_VERSION`] (or is within range for lenient mode).
/// 2. `ts` is non-empty.
/// 3. `ok == true` implies `data.is_some()` and `error.is_none()`.
/// 4. `ok == false` implies `error.is_some()` and `data.is_none()`.
/// 5. Error code, if present, is a recognized stable code.
/// 6. `meta.command` and `meta.format` are non-empty.
/// 7. In strict mode, `meta.format` must be a canonical known format label.
#[must_use]
pub fn validate_envelope<T>(
    envelope: &OutputEnvelope<T>,
    mode: CompatibilityMode,
) -> ValidationResult {
    let mut violations = Vec::new();
    validate_schema_version(envelope, mode, &mut violations);
    validate_timestamp(envelope, &mut violations);
    validate_envelope_shape(envelope, &mut violations);
    validate_error_code(envelope, &mut violations);
    validate_meta_fields(envelope, mode, &mut violations);

    ValidationResult {
        valid: violations.is_empty(),
        violations,
    }
}

fn validate_schema_version<T>(
    envelope: &OutputEnvelope<T>,
    mode: CompatibilityMode,
    violations: &mut Vec<ValidationViolation>,
) {
    match mode {
        CompatibilityMode::Strict => {
            if envelope.v != OUTPUT_SCHEMA_VERSION {
                push_violation(
                    violations,
                    "v",
                    "output.schema_version.mismatch",
                    format!(
                        "expected schema version {OUTPUT_SCHEMA_VERSION}, found {}",
                        envelope.v
                    ),
                );
            }
        }
        CompatibilityMode::Lenient => {
            if envelope.v < OUTPUT_SCHEMA_MIN_SUPPORTED {
                push_violation(
                    violations,
                    "v",
                    "output.schema_version.too_old",
                    format!(
                        "schema version {} is below minimum supported {OUTPUT_SCHEMA_MIN_SUPPORTED}",
                        envelope.v
                    ),
                );
            }
        }
    }
}

fn validate_timestamp<T>(envelope: &OutputEnvelope<T>, violations: &mut Vec<ValidationViolation>) {
    if envelope.ts.is_empty() {
        push_violation(
            violations,
            "ts",
            "output.timestamp.empty",
            "timestamp must be non-empty RFC 3339 UTC",
        );
    }
}

fn validate_envelope_shape<T>(
    envelope: &OutputEnvelope<T>,
    violations: &mut Vec<ValidationViolation>,
) {
    if envelope.ok {
        if envelope.data.is_none() {
            push_violation(
                violations,
                "data",
                "output.data.missing_on_success",
                "data must be present when ok == true",
            );
        }
        if envelope.error.is_some() {
            push_violation(
                violations,
                "error",
                "output.error.present_on_success",
                "error must be absent when ok == true",
            );
        }
    } else {
        if envelope.error.is_none() {
            push_violation(
                violations,
                "error",
                "output.error.missing_on_failure",
                "error must be present when ok == false",
            );
        }
        if envelope.data.is_some() {
            push_violation(
                violations,
                "data",
                "output.data.present_on_failure",
                "data must be absent when ok == false",
            );
        }
    }
}

fn validate_error_code<T>(envelope: &OutputEnvelope<T>, violations: &mut Vec<ValidationViolation>) {
    if let Some(error) = &envelope.error
        && !ALL_OUTPUT_ERROR_CODES.contains(&error.code.as_str())
    {
        push_violation(
            violations,
            "error.code",
            "output.error_code.unrecognized",
            format!("unrecognized error code: {}", error.code),
        );
    }
}

fn validate_meta_fields<T>(
    envelope: &OutputEnvelope<T>,
    mode: CompatibilityMode,
    violations: &mut Vec<ValidationViolation>,
) {
    if envelope.meta.command.is_empty() {
        push_violation(
            violations,
            "meta.command",
            "output.meta.command_empty",
            "meta.command must be non-empty",
        );
    }
    if envelope.meta.format.is_empty() {
        push_violation(
            violations,
            "meta.format",
            "output.meta.format_empty",
            "meta.format must be non-empty",
        );
    } else if matches!(mode, CompatibilityMode::Strict)
        && !KNOWN_OUTPUT_FORMATS.contains(&envelope.meta.format.as_str())
    {
        push_violation(
            violations,
            "meta.format",
            "output.meta.format_unrecognized",
            format!(
                "meta.format must be one of: {}",
                KNOWN_OUTPUT_FORMATS.join(", ")
            ),
        );
    }
}

fn push_violation(
    violations: &mut Vec<ValidationViolation>,
    field: &str,
    code: &str,
    message: impl Into<String>,
) {
    violations.push(ValidationViolation {
        field: field.to_owned(),
        code: code.to_owned(),
        message: message.into(),
    });
}

/// Check whether a schema version is compatible under the given mode.
#[must_use]
pub const fn is_version_compatible(version: u32, mode: CompatibilityMode) -> bool {
    match mode {
        CompatibilityMode::Strict => version == OUTPUT_SCHEMA_VERSION,
        CompatibilityMode::Lenient => version >= OUTPUT_SCHEMA_MIN_SUPPORTED,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn sample_meta() -> OutputMeta {
        OutputMeta::new("search", "json")
    }

    fn sample_ts() -> &'static str {
        "2026-02-14T12:00:00Z"
    }

    #[test]
    fn search_output_phase_display_matches_wire_values() {
        assert_eq!(SearchOutputPhase::Initial.to_string(), "initial");
        assert_eq!(SearchOutputPhase::Refined.to_string(), "refined");
        assert_eq!(
            SearchOutputPhase::RefinementFailed.to_string(),
            "refinement_failed"
        );
    }

    #[test]
    fn search_payload_serde_roundtrip() {
        let payload = SearchPayload::new(
            "how does auth work",
            SearchOutputPhase::Initial,
            3,
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
        assert!(!payload.is_empty());

        let json = serde_json::to_string(&payload).expect("serialize payload");
        let decoded: SearchPayload = serde_json::from_str(&json).expect("deserialize payload");
        assert_eq!(decoded, payload);
        assert_eq!(decoded.returned_hits, 2);
    }

    // ─── OutputEnvelope construction ────────────────────────────────────

    #[test]
    fn success_envelope_invariants() {
        let env: OutputEnvelope<String> =
            OutputEnvelope::success("hello".to_owned(), sample_meta(), sample_ts());
        assert!(env.ok);
        assert_eq!(env.data.as_deref(), Some("hello"));
        assert!(env.error.is_none());
        assert_eq!(env.v, OUTPUT_SCHEMA_VERSION);
        assert!(env.warnings.is_empty());
    }

    #[test]
    fn error_envelope_invariants() {
        let err = OutputError::new(OutputErrorCode::INDEX_NOT_FOUND, "not found", 1);
        let env: OutputEnvelope<String> = OutputEnvelope::error(err, sample_meta(), sample_ts());
        assert!(!env.ok);
        assert!(env.data.is_none());
        assert!(env.error.is_some());
        assert_eq!(env.error.as_ref().unwrap().code, "index_not_found");
    }

    #[test]
    fn envelope_with_warnings() {
        let env: OutputEnvelope<u32> =
            OutputEnvelope::success(42, sample_meta(), sample_ts()).with_warnings(vec![
                OutputWarning::new(OutputWarningCode::DEGRADED_MODE, "quality tier skipped"),
            ]);
        assert_eq!(env.warnings.len(), 1);
        assert_eq!(env.warnings[0].code, "degraded_mode");
    }

    // ─── Serde round-trips ─────────────────────────────────────────────

    #[test]
    fn success_envelope_serde_roundtrip() {
        let env = OutputEnvelope::success(vec!["result1", "result2"], sample_meta(), sample_ts());
        let json = serde_json::to_string(&env).unwrap();
        let decoded: OutputEnvelope<Vec<String>> = serde_json::from_str(&json).unwrap();
        assert!(decoded.ok);
        assert_eq!(decoded.data.as_ref().unwrap().len(), 2);
        assert_eq!(decoded.v, OUTPUT_SCHEMA_VERSION);
    }

    #[test]
    fn error_envelope_serde_roundtrip() {
        let err = OutputError::new(OutputErrorCode::INVALID_CONFIG, "bad value", 2)
            .with_field("quality_weight");
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta(), sample_ts());
        let json = serde_json::to_string(&env).unwrap();
        let decoded: OutputEnvelope<()> = serde_json::from_str(&json).unwrap();
        assert!(!decoded.ok);
        let e = decoded.error.as_ref().unwrap();
        assert_eq!(e.code, "invalid_config");
        assert_eq!(e.field.as_deref(), Some("quality_weight"));
        assert_eq!(e.exit_code, 2);
    }

    #[test]
    fn optional_fields_omitted_in_json() {
        let env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        let json = serde_json::to_string(&env).unwrap();
        assert!(!json.contains("\"error\""));
        assert!(!json.contains("\"warnings\""));
        assert!(!json.contains("\"duration_ms\""));
        assert!(!json.contains("\"request_id\""));
    }

    #[test]
    fn warning_serde_roundtrip() {
        let w = OutputWarning::new("degraded_mode", "test warning");
        let json = serde_json::to_string(&w).unwrap();
        let decoded: OutputWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, w);
    }

    #[test]
    fn toon_roundtrip_preserves_success_envelope_semantics() {
        let env = OutputEnvelope::success(
            vec!["doc-1", "doc-2"],
            OutputMeta::new("search", "toon"),
            sample_ts(),
        )
        .with_warnings(vec![OutputWarning::new(
            OutputWarningCode::DEGRADED_MODE,
            "quality tier skipped",
        )]);

        let toon = encode_envelope_toon(&env).expect("encode toon");
        let decoded: OutputEnvelope<Vec<String>> =
            decode_envelope_toon(&toon).expect("decode toon");

        let original_value = serde_json::to_value(&env).expect("serialize original");
        let decoded_value = serde_json::to_value(&decoded).expect("serialize decoded");
        assert_eq!(decoded_value, original_value);
    }

    #[test]
    fn toon_roundtrip_preserves_error_envelope_semantics() {
        let err = OutputError::new(OutputErrorCode::INVALID_CONFIG, "bad value", 2)
            .with_field("meta.format");
        let env: OutputEnvelope<()> =
            OutputEnvelope::error(err, OutputMeta::new("status", "toon"), sample_ts());

        let toon = encode_envelope_toon(&env).expect("encode toon");
        let decoded: OutputEnvelope<()> = decode_envelope_toon(&toon).expect("decode toon");

        let original_value = serde_json::to_value(&env).expect("serialize original");
        let decoded_value = serde_json::to_value(&decoded).expect("serialize decoded");
        assert_eq!(decoded_value, original_value);
    }

    #[test]
    fn toon_roundtrip_preserves_ambiguous_string_tokens() {
        let env = OutputEnvelope::success(
            vec![
                "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                "1abc".to_owned(),
                "\"quoted\"".to_owned(),
                "[array-like".to_owned(),
                "null1".to_owned(),
                "-abc".to_owned(),
            ],
            OutputMeta::new("search", "toon").with_request_id("01JAH9A2W8F8Q6GQ4C7M3N2P1R"),
            sample_ts(),
        );

        let toon = encode_envelope_toon(&env).expect("encode toon");
        let decoded: OutputEnvelope<Vec<String>> =
            decode_envelope_toon(&toon).expect("decode toon");

        assert_eq!(decoded.ts, env.ts);
        assert_eq!(decoded.data, env.data);
        assert_eq!(decoded.meta.request_id, env.meta.request_id);
    }

    #[test]
    fn toon_string_token_wrap_decision_matches_parser_behavior() {
        assert!(should_wrap_toon_string_token("2026-02-14T12:00:00Z"));
        assert!(should_wrap_toon_string_token("1abc"));
        assert!(should_wrap_toon_string_token("\"quoted\""));
        assert!(should_wrap_toon_string_token("[array-like"));
        assert!(should_wrap_toon_string_token("null1"));
        assert!(should_wrap_toon_string_token("-abc"));
        assert!(!should_wrap_toon_string_token("doc-1"));
    }

    // ─── OutputMeta ────────────────────────────────────────────────────

    #[test]
    fn meta_builder_chain() {
        let meta = OutputMeta::new("search", "json")
            .with_duration_ms(42)
            .with_request_id("01JAH9A2W8F8Q6GQ4C7M3N2P1R");
        assert_eq!(meta.command, "search");
        assert_eq!(meta.duration_ms, Some(42));
        assert_eq!(
            meta.request_id.as_deref(),
            Some("01JAH9A2W8F8Q6GQ4C7M3N2P1R")
        );
    }

    #[test]
    fn meta_serde_roundtrip() {
        let meta = OutputMeta::new("status", "jsonl").with_duration_ms(100);
        let json = serde_json::to_string(&meta).unwrap();
        let decoded: OutputMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, meta);
    }

    // ─── OutputError ───────────────────────────────────────────────────

    #[test]
    fn output_error_with_field() {
        let e = OutputError::new("invalid_config", "bad", 2).with_field("timeout_ms");
        assert_eq!(e.field.as_deref(), Some("timeout_ms"));
    }

    #[test]
    fn output_error_without_field_omits_in_json() {
        let e = OutputError::new("io_error", "disk full", 1);
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("\"field\""));
    }

    // ─── CompatibilityMode ─────────────────────────────────────────────

    #[test]
    fn compatibility_mode_display() {
        assert_eq!(CompatibilityMode::Strict.to_string(), "strict");
        assert_eq!(CompatibilityMode::Lenient.to_string(), "lenient");
    }

    #[test]
    fn compatibility_mode_serde_roundtrip() {
        for mode in [CompatibilityMode::Strict, CompatibilityMode::Lenient] {
            let json = serde_json::to_string(&mode).unwrap();
            let decoded: CompatibilityMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    #[test]
    fn compatibility_mode_default_is_lenient() {
        assert_eq!(CompatibilityMode::default(), CompatibilityMode::Lenient);
    }

    // ─── Error Codes ───────────────────────────────────────────────────

    #[test]
    fn all_error_codes_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &code in ALL_OUTPUT_ERROR_CODES {
            assert!(seen.insert(code), "duplicate error code: {code}");
        }
    }

    #[test]
    fn all_error_codes_are_snake_case() {
        for &code in ALL_OUTPUT_ERROR_CODES {
            assert!(
                code.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "error code is not snake_case: {code}"
            );
        }
    }

    #[test]
    fn all_warning_codes_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &code in ALL_OUTPUT_WARNING_CODES {
            assert!(seen.insert(code), "duplicate warning code: {code}");
        }
    }

    // ─── SearchError mapping ───────────────────────────────────────────

    #[test]
    #[allow(clippy::too_many_lines)]
    fn error_code_for_all_variants() {
        use frankensearch_core::SearchError;

        let cases: Vec<(SearchError, &str)> = vec![
            (
                SearchError::EmbedderUnavailable {
                    model: "m".into(),
                    reason: "r".into(),
                },
                OutputErrorCode::EMBEDDER_UNAVAILABLE,
            ),
            (
                SearchError::ModelNotFound { name: "m".into() },
                OutputErrorCode::MODEL_NOT_FOUND,
            ),
            (
                SearchError::IndexNotFound {
                    path: PathBuf::from("/tmp"),
                },
                OutputErrorCode::INDEX_NOT_FOUND,
            ),
            (
                SearchError::IndexCorrupted {
                    path: PathBuf::from("/tmp"),
                    detail: "bad".into(),
                },
                OutputErrorCode::INDEX_CORRUPTED,
            ),
            (
                SearchError::IndexVersionMismatch {
                    expected: 1,
                    found: 2,
                },
                OutputErrorCode::INDEX_VERSION_MISMATCH,
            ),
            (
                SearchError::DimensionMismatch {
                    expected: 256,
                    found: 384,
                },
                OutputErrorCode::DIMENSION_MISMATCH,
            ),
            (
                SearchError::QueryParseError {
                    query: "q".into(),
                    detail: "d".into(),
                },
                OutputErrorCode::QUERY_PARSE_ERROR,
            ),
            (
                SearchError::SearchTimeout {
                    elapsed_ms: 100,
                    budget_ms: 50,
                },
                OutputErrorCode::SEARCH_TIMEOUT,
            ),
            (
                SearchError::FederatedInsufficientResponses {
                    required: 2,
                    received: 1,
                },
                OutputErrorCode::FEDERATED_INSUFFICIENT,
            ),
            (
                SearchError::RerankerUnavailable { model: "m".into() },
                OutputErrorCode::RERANKER_UNAVAILABLE,
            ),
            (
                SearchError::InvalidConfig {
                    field: "f".into(),
                    value: "v".into(),
                    reason: "r".into(),
                },
                OutputErrorCode::INVALID_CONFIG,
            ),
            (
                SearchError::HashMismatch {
                    path: PathBuf::from("/tmp"),
                    expected: "a".into(),
                    actual: "b".into(),
                },
                OutputErrorCode::HASH_MISMATCH,
            ),
            (
                SearchError::Cancelled {
                    phase: "p".into(),
                    reason: "r".into(),
                },
                OutputErrorCode::CANCELLED,
            ),
            (
                SearchError::QueueFull {
                    pending: 10,
                    capacity: 10,
                },
                OutputErrorCode::QUEUE_FULL,
            ),
            (
                SearchError::DurabilityDisabled,
                OutputErrorCode::DURABILITY_DISABLED,
            ),
        ];

        for (err, expected_code) in &cases {
            assert_eq!(error_code_for(err), *expected_code, "mismatch for {err:?}");
        }
    }

    #[test]
    fn exit_code_for_usage_errors() {
        use crate::adapters::cli::exit_code;
        use frankensearch_core::SearchError;

        let config_err = SearchError::InvalidConfig {
            field: "f".into(),
            value: "v".into(),
            reason: "r".into(),
        };
        assert_eq!(exit_code_for(&config_err), exit_code::USAGE_ERROR);

        let parse_err = SearchError::QueryParseError {
            query: "q".into(),
            detail: "d".into(),
        };
        assert_eq!(exit_code_for(&parse_err), exit_code::USAGE_ERROR);
    }

    #[test]
    fn exit_code_for_cancelled_is_interrupted() {
        use crate::adapters::cli::exit_code;
        use frankensearch_core::SearchError;

        let err = SearchError::Cancelled {
            phase: "p".into(),
            reason: "r".into(),
        };
        assert_eq!(exit_code_for(&err), exit_code::INTERRUPTED);
    }

    #[test]
    fn exit_code_for_model_unavailable() {
        use crate::adapters::cli::exit_code;
        use frankensearch_core::SearchError;

        let err = SearchError::EmbedderUnavailable {
            model: "MiniLM".into(),
            reason: "feature not enabled".into(),
        };
        assert_eq!(exit_code_for(&err), exit_code::MODEL_UNAVAILABLE);

        let err = SearchError::ModelNotFound {
            name: "potion-128M".into(),
        };
        assert_eq!(exit_code_for(&err), exit_code::MODEL_UNAVAILABLE);

        let err = SearchError::ModelLoadFailed {
            path: PathBuf::from("/models/broken.onnx"),
            source: Box::new(std::io::Error::other("corrupted")),
        };
        assert_eq!(exit_code_for(&err), exit_code::MODEL_UNAVAILABLE);
    }

    #[test]
    fn exit_code_for_runtime_errors() {
        use crate::adapters::cli::exit_code;
        use frankensearch_core::SearchError;

        let err = SearchError::IndexNotFound {
            path: PathBuf::from("/tmp"),
        };
        assert_eq!(exit_code_for(&err), exit_code::RUNTIME_ERROR);
    }

    #[test]
    fn output_error_from_preserves_field() {
        use frankensearch_core::SearchError;

        let err = SearchError::InvalidConfig {
            field: "quality_weight".into(),
            value: "-1".into(),
            reason: "must be positive".into(),
        };
        let out = output_error_from(&err);
        assert_eq!(out.code, "invalid_config");
        assert_eq!(out.field.as_deref(), Some("quality_weight"));
        assert_eq!(out.exit_code, 2);
    }

    #[test]
    fn output_error_from_non_config_has_no_field() {
        use frankensearch_core::SearchError;

        let err = SearchError::IndexNotFound {
            path: PathBuf::from("/tmp"),
        };
        let out = output_error_from(&err);
        assert!(out.field.is_none());
    }

    // ─── Helpful Error Messages (bd-2w7x.31) ──────────────────────────

    #[test]
    fn suggestion_for_model_not_found() {
        use frankensearch_core::SearchError;

        let err = SearchError::ModelNotFound {
            name: "potion-128m".into(),
        };
        let out = output_error_from(&err);
        let suggestion = out.suggestion.unwrap();
        assert!(
            suggestion.contains("fsfs download-models"),
            "suggestion should tell user to download: {suggestion}"
        );
        assert!(
            suggestion.contains("potion-128m"),
            "suggestion should mention the missing model: {suggestion}"
        );
    }

    #[test]
    fn suggestion_for_embedder_unavailable() {
        use frankensearch_core::SearchError;

        let err = SearchError::EmbedderUnavailable {
            model: "MiniLM-L6-v2".into(),
            reason: "feature not enabled".into(),
        };
        let out = output_error_from(&err);
        let suggestion = out.suggestion.unwrap();
        assert!(suggestion.contains("FRANKENSEARCH_MODEL_DIR"));
        assert!(suggestion.contains("MiniLM-L6-v2"));
    }

    #[test]
    fn suggestion_for_index_not_found() {
        use frankensearch_core::SearchError;

        let err = SearchError::IndexNotFound {
            path: PathBuf::from("/home/user/.frankensearch"),
        };
        let out = output_error_from(&err);
        let suggestion = out.suggestion.unwrap();
        assert!(
            suggestion.contains("fsfs index"),
            "should tell user to create index: {suggestion}"
        );
    }

    #[test]
    fn suggestion_for_index_corrupted() {
        use frankensearch_core::SearchError;

        let err = SearchError::IndexCorrupted {
            path: PathBuf::from("/data/index.fsvi"),
            detail: "CRC mismatch".into(),
        };
        let out = output_error_from(&err);
        let suggestion = out.suggestion.unwrap();
        assert!(suggestion.contains("--force"));
    }

    #[test]
    fn context_for_model_errors() {
        use frankensearch_core::SearchError;

        let err = SearchError::ModelNotFound {
            name: "potion".into(),
        };
        let out = output_error_from(&err);
        let context = out.context.unwrap();
        assert!(
            context.contains("bundled"),
            "context should mention bundled models: {context}"
        );
    }

    #[test]
    fn context_for_index_not_found() {
        use frankensearch_core::SearchError;

        let err = SearchError::IndexNotFound {
            path: PathBuf::from("/tmp"),
        };
        let out = output_error_from(&err);
        let context = out.context.unwrap();
        assert!(context.contains("index"));
    }

    #[test]
    fn output_error_suggestion_omitted_in_json_when_none() {
        let err = OutputError::new("cancelled", "user cancelled", 130);
        let json = serde_json::to_string(&err).unwrap();
        assert!(
            !json.contains("suggestion"),
            "suggestion should be omitted from JSON when None"
        );
        assert!(
            !json.contains("context"),
            "context should be omitted from JSON when None"
        );
    }

    #[test]
    fn output_error_suggestion_present_in_json_when_set() {
        let err = OutputError::new("model_not_found", "not found", 78)
            .with_suggestion("fsfs download-models")
            .with_context("Models needed for semantic search");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("suggestion"));
        assert!(json.contains("fsfs download-models"));
        assert!(json.contains("context"));
        assert!(json.contains("Models needed"));
    }

    #[test]
    fn output_error_serde_roundtrip_with_hints() {
        let err = OutputError::new("io_error", "disk full", 1)
            .with_suggestion("free disk space")
            .with_context("not enough room");
        let json = serde_json::to_string(&err).unwrap();
        let restored: OutputError = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.suggestion.as_deref(), Some("free disk space"));
        assert_eq!(restored.context.as_deref(), Some("not enough room"));
    }

    #[test]
    fn no_suggestion_for_cancelled() {
        use frankensearch_core::SearchError;

        let err = SearchError::Cancelled {
            phase: "search".into(),
            reason: "user interrupt".into(),
        };
        let out = output_error_from(&err);
        assert!(
            out.suggestion.is_none(),
            "cancelled errors should not have suggestions"
        );
    }

    // ─── Validation ────────────────────────────────────────────────────

    #[test]
    fn valid_success_envelope_passes_strict() {
        let env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(result.valid, "violations: {:?}", result.violations);
    }

    #[test]
    fn valid_error_envelope_passes_strict() {
        let err = OutputError::new(OutputErrorCode::IO_ERROR, "disk full", 1);
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta(), sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(result.valid, "violations: {:?}", result.violations);
    }

    #[test]
    fn wrong_version_fails_strict() {
        let mut env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        env.v = 999;
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.schema_version.mismatch")
        );
    }

    #[test]
    fn future_version_passes_lenient() {
        let mut env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        env.v = OUTPUT_SCHEMA_VERSION + 5;
        let result = validate_envelope(&env, CompatibilityMode::Lenient);
        assert!(result.valid, "violations: {:?}", result.violations);
    }

    #[test]
    fn old_version_fails_lenient() {
        let mut env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        env.v = 0;
        let result = validate_envelope(&env, CompatibilityMode::Lenient);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.schema_version.too_old")
        );
    }

    #[test]
    fn empty_timestamp_fails() {
        let mut env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        env.ts = String::new();
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.timestamp.empty")
        );
    }

    #[test]
    fn success_with_error_fails() {
        let mut env = OutputEnvelope::success("ok", sample_meta(), sample_ts());
        env.error = Some(OutputError::new("io_error", "bad", 1));
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.error.present_on_success")
        );
    }

    #[test]
    fn failure_without_error_fails() {
        let mut env: OutputEnvelope<()> = OutputEnvelope::success((), sample_meta(), sample_ts());
        env.ok = false;
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.error.missing_on_failure")
        );
    }

    #[test]
    fn failure_with_data_fails() {
        let err = OutputError::new(OutputErrorCode::IO_ERROR, "bad", 1);
        let mut env = OutputEnvelope::error(err, sample_meta(), sample_ts());
        env.data = Some("leaked");
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.data.present_on_failure")
        );
    }

    #[test]
    fn success_without_data_fails() {
        let mut env: OutputEnvelope<String> =
            OutputEnvelope::success("x".into(), sample_meta(), sample_ts());
        env.data = None;
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.data.missing_on_success")
        );
    }

    #[test]
    fn unrecognized_error_code_fails() {
        let err = OutputError::new("totally_bogus_code", "bad", 1);
        let env: OutputEnvelope<()> = OutputEnvelope::error(err, sample_meta(), sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.error_code.unrecognized")
        );
    }

    #[test]
    fn empty_meta_command_fails() {
        let meta = OutputMeta::new("", "json");
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.meta.command_empty")
        );
    }

    #[test]
    fn empty_meta_format_fails() {
        let meta = OutputMeta::new("search", "");
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.meta.format_empty")
        );
    }

    #[test]
    fn strict_mode_rejects_unrecognized_meta_format() {
        let meta = OutputMeta::new("search", "yaml");
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "output.meta.format_unrecognized")
        );
    }

    #[test]
    fn lenient_mode_allows_unrecognized_meta_format() {
        let meta = OutputMeta::new("search", "yaml");
        let env = OutputEnvelope::success("ok", meta, sample_ts());
        let result = validate_envelope(&env, CompatibilityMode::Lenient);
        assert!(result.valid, "{:?}", result.violations);
    }

    #[test]
    fn strict_mode_accepts_all_known_meta_formats() {
        for format in KNOWN_OUTPUT_FORMATS {
            let meta = OutputMeta::new("search", *format);
            let env = OutputEnvelope::success("ok", meta, sample_ts());
            let result = validate_envelope(&env, CompatibilityMode::Strict);
            assert!(result.valid, "format `{format}` should be accepted");
        }
    }

    // ─── Version compatibility ─────────────────────────────────────────

    #[test]
    fn version_compatibility_strict() {
        assert!(is_version_compatible(
            OUTPUT_SCHEMA_VERSION,
            CompatibilityMode::Strict
        ));
        assert!(!is_version_compatible(
            OUTPUT_SCHEMA_VERSION + 1,
            CompatibilityMode::Strict
        ));
        assert!(!is_version_compatible(0, CompatibilityMode::Strict));
    }

    #[test]
    fn version_compatibility_lenient() {
        assert!(is_version_compatible(
            OUTPUT_SCHEMA_VERSION,
            CompatibilityMode::Lenient
        ));
        assert!(is_version_compatible(
            OUTPUT_SCHEMA_VERSION + 10,
            CompatibilityMode::Lenient
        ));
        assert!(!is_version_compatible(0, CompatibilityMode::Lenient));
    }

    // ─── FieldPresence ─────────────────────────────────────────────────

    #[test]
    fn field_presence_display() {
        assert_eq!(FieldPresence::Required.to_string(), "required");
        assert_eq!(FieldPresence::Optional.to_string(), "optional");
        assert_eq!(
            FieldPresence::ConditionalOn("ok == true").to_string(),
            "conditional(ok == true)"
        );
        assert_eq!(
            FieldPresence::Deprecated {
                since: 1,
                removed_in: Some(3)
            }
            .to_string(),
            "deprecated(since=v1, removed_in=v3)"
        );
        assert_eq!(
            FieldPresence::Deprecated {
                since: 2,
                removed_in: None
            }
            .to_string(),
            "deprecated(since=v2)"
        );
    }

    // ─── ENVELOPE_FIELDS ───────────────────────────────────────────────

    #[test]
    fn envelope_fields_cover_all_top_level_fields() {
        let top_level = ["v", "ts", "ok", "data", "error", "warnings", "meta"];
        for field in top_level {
            assert!(
                ENVELOPE_FIELDS.iter().any(|f| f.name == field),
                "missing field descriptor for: {field}"
            );
        }
    }

    #[test]
    fn envelope_fields_have_unique_names() {
        let mut seen = std::collections::HashSet::new();
        for f in ENVELOPE_FIELDS {
            assert!(
                seen.insert(f.name),
                "duplicate field descriptor: {}",
                f.name
            );
        }
    }

    #[test]
    fn envelope_field_count() {
        // 7 top-level + 4 meta + 4 error = 15
        assert_eq!(ENVELOPE_FIELDS.len(), 15);
    }

    // ─── Multiple violations ───────────────────────────────────────────

    #[test]
    fn multiple_violations_accumulated() {
        let mut env: OutputEnvelope<String> =
            OutputEnvelope::success("x".into(), OutputMeta::new("", ""), sample_ts());
        env.v = 999;
        env.ts = String::new();
        env.data = None;
        let result = validate_envelope(&env, CompatibilityMode::Strict);
        assert!(!result.valid);
        // Should have at least version + timestamp + missing data violations
        assert!(result.violations.len() >= 3);
    }
}
